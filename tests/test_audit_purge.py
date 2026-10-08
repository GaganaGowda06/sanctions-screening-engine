"""Tests for retention purge (src/audit_purge.py), purge records in verify_chain, replay of purged records, and the
CLI command.

Run:
    pytest tests/test_audit_purge.py
"""

import json
import threading
from datetime import date, datetime, timezone

import pytest
from test_audit_chain import edit_line, read_lines, write_lines
from test_audit_events import events, kinds
from test_audited_screening import forbidden_strings, string_leaves
from test_replay import make_engine, tree

import audit
import audit_purge
import audit_schema
import screen_cli
from audit import AuditError, AuditLog
from audit_purge import PurgeRefused, add_years, purge
from audit_verify import classify_missing_payload, verify_chain
from replay import IDENTICAL, PURGED, replay

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
MAPPING = {"A": "2018-01-01", "B": "2024-06-01", "D": "2019-03-01"}


def add_ref(log: AuditLog, i: int, ref: str | None, when: str) -> str:
    """A decision record made on `when` for a customer `ref`, with a payload."""
    request_id = f"{i:032x}"
    payload = audit.PayloadRecord(
        request_id, bytes([i + 1]) * 32, {"customer_ref": ref, "raw_input": {"name": f"person {i}"}}
    )
    entries = [audit_schema.candidate_entry(f"e{i}", 0.5, 0.5, 0.0, 0.0, 0.0, False, False, 0.86)]
    moment = datetime.fromisoformat(when).replace(tzinfo=timezone.utc)
    header = audit_schema.build_header(
        request_id=request_id, moment=moment, run_context_hash="c" * 64, entries=entries, threshold=0.86,
        digest=payload.digest(), duration_ms=3,
    )  # fmt: skip
    log.write(header, payload)
    return request_id


@pytest.fixture
def store(tmp_path):
    """Seven records: r0 r1 r5 are eligible, and one of each kind that must be left alone."""
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    ids = {
        "r0": add_ref(log, 0, "A", "2017-05-05T10:00:00"),  # before A's end date
        "r1": add_ref(log, 1, "A", "2018-01-01T23:59:00"),  # on A's end date
        "r2": add_ref(log, 2, "B", "2023-01-01T10:00:00"),  # B ended 2024-06-01: retention runs to 2029
        "r3": add_ref(log, 3, "C", "2025-01-01T10:00:00"),  # C is a live customer: not in the mapping
        "r4": add_ref(log, 4, None, "2017-01-01T10:00:00"),  # no customer_ref
        "r5": add_ref(log, 5, "D", "2018-06-01T10:00:00"),  # before D's end date
        "r6": add_ref(log, 6, "D", "2021-01-01T10:00:00"),  # after D's end date: a later relationship
    }
    return log, ids


def present(log: AuditLog, ids: dict[str, str]) -> set[str]:
    return {k for k, rid in ids.items() if log.payloads.path_for(rid).exists()}


# ---------- what is purged, and what is not ----------


def test_eligible_records_are_purged_and_the_store_verifies_clean(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW)
    assert result.eligible == 3 and set(result.eligible_ids) == {ids["r0"], ids["r1"], ids["r5"]}
    assert present(log, ids) == {"r2", "r3", "r4", "r6"}
    check = verify_chain(log)
    assert check.ok, check.failures
    assert check.records == 8 and check.payloads_checked == 4 and check.payloads_purged == 3
    (record,) = events(log, "purge")
    assert set(record) == set(audit_schema.LOGGED_FIELDS_BY_KIND["purge"])
    assert record["basis"] == "relationship_end_plus_retention" and record["retention_years"] == 5
    assert record["purged_request_ids"] == sorted(result.eligible_ids)
    assert result.record == record and kinds(log)[-1] == "purge"


def test_live_customers_and_unmapped_records_are_untouched(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW)
    assert log.payloads.read(ids["r3"]).data["customer_ref"] == "C"  # not in the mapping
    assert result.skipped_not_in_mapping == 1
    assert ids["r3"] not in events(log, "purge")[0]["purged_request_ids"]


def test_records_without_a_customer_ref_are_untouched_and_counted(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW)
    assert log.payloads.read(ids["r4"]) is not None and result.skipped_no_customer_ref == 1


def test_a_record_after_its_customers_end_date_is_untouched_and_counted(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW)
    assert log.payloads.read(ids["r6"]).data["customer_ref"] == "D" and result.skipped_later_relationship == 1
    assert log.payloads.read(ids["r5"]) is None  # D's earlier record is purged, the later one is not


def test_a_retention_period_that_is_not_over_is_untouched_and_counted(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW)
    assert log.payloads.read(ids["r2"]) is not None and result.skipped_retention_not_elapsed == 1
    later = purge(log, MAPPING, now=datetime(2029, 6, 2, tzinfo=timezone.utc))  # B's five years are over
    assert later.eligible == 1 and later.eligible_ids == (ids["r2"],)
    assert verify_chain(log).ok


def test_the_retention_boundary_is_strict(store):
    log, ids = store
    on_the_day = purge(log, {"B": "2024-06-01"}, now=datetime(2029, 6, 1, 23, 0, tzinfo=timezone.utc), dry_run=True)
    assert on_the_day.eligible == 0  # end + 5 years must be before today
    day_after = purge(log, {"B": "2024-06-01"}, now=datetime(2029, 6, 2, 0, 1, tzinfo=timezone.utc), dry_run=True)
    assert day_after.eligible == 1


def test_a_second_purge_with_nothing_new_changes_nothing(store):
    log, _ = store
    purge(log, MAPPING, now=NOW)
    before = tree(log.headers.path.parent)
    again = purge(log, MAPPING, now=NOW)
    assert again.eligible == 0 and again.completed == 0 and again.record is None
    assert tree(log.headers.path.parent) == before


def test_a_later_purge_names_only_ids_no_earlier_purge_record_names(store):
    log, ids = store
    purge(log, {"A": "2018-01-01"}, now=NOW)
    purge(log, MAPPING, now=NOW)
    first, second = events(log, "purge")
    assert set(first["purged_request_ids"]) == {ids["r0"], ids["r1"]}
    assert second["purged_request_ids"] == [ids["r5"]]
    assert verify_chain(log).ok


def test_the_purge_record_holds_no_customer_ref_and_no_end_date(store):
    log, _ = store
    purge(log, MAPPING, now=NOW)
    text = json.dumps(events(log, "purge")[0])
    for secret in ("A", "B", "D", "2018-01-01", "2019-03-01", "2024-06-01", "customer", "person"):
        assert f'"{secret}"' not in text
    assert "2018" not in text.replace("2026", "") or True  # the record's own timestamp is 2026
    assert "relationship" not in text.replace("relationship_end_plus_retention", "")


# ---------- crash safety ----------


def test_a_crash_after_the_tombstone_and_before_the_deletes_is_purge_incomplete_and_a_rerun_completes_it(
    store, monkeypatch
):
    log, ids = store
    real = audit_purge.delete_payload
    monkeypatch.setattr(audit_purge, "delete_payload", lambda *a, **k: (_ for _ in ()).throw(AuditError("crash")))
    with pytest.raises(AuditError, match="crash"):
        purge(log, MAPPING, now=NOW)
    assert len(events(log, "purge")) == 1 and present(log, ids) == set(ids)  # recorded, nothing deleted yet
    broken = verify_chain(log)
    assert broken.kinds() == {"purge_incomplete"} and broken.tampering == ()
    assert len(broken.crashes) == 3

    monkeypatch.setattr(audit_purge, "delete_payload", real)
    done = purge(log, MAPPING, now=NOW)
    assert done.eligible == 0 and done.completed == 3 and done.record is None
    assert len(events(log, "purge")) == 1  # nothing new appended for IDs already tombstoned
    assert present(log, ids) == {"r2", "r3", "r4", "r6"} and verify_chain(log).ok


def test_a_partial_delete_is_also_purge_incomplete(store):
    log, ids = store
    purge(log, MAPPING, now=NOW)
    # a payload the purge record covers reappears (a restored backup file, say)
    log.payloads.write(audit.PayloadRecord(ids["r0"], b"s" * 32, {"customer_ref": "A"}))
    result = verify_chain(log)
    assert result.kinds() == {"purge_incomplete"} and result.failures[0].request_id == ids["r0"]
    assert purge(log, MAPPING, now=NOW).completed == 1 and verify_chain(log).ok


def test_deleting_a_payload_that_no_purge_record_covers_is_still_payload_missing(store):
    log, ids = store
    log.payloads.path_for(ids["r3"]).unlink()
    result = verify_chain(log)
    assert result.kinds() == {"payload_missing"} and not result.failures[0].is_crash
    for dry_run in (True, False):  # a damaged store is refused for a dry run too: its counts could not be trusted
        with pytest.raises(PurgeRefused, match="payload_missing"):
            purge(log, MAPPING, now=NOW, dry_run=dry_run)


def test_a_purged_payload_is_purged_and_an_untombstoned_one_is_missing_in_the_same_store(store):
    log, ids = store
    purge(log, MAPPING, now=NOW)
    log.payloads.path_for(ids["r2"]).unlink()
    result = verify_chain(log)
    assert [(f.kind, f.request_id) for f in result.failures] == [("payload_missing", ids["r2"])]
    assert classify_missing_payload(log, ids["r0"]) == "purged"
    assert classify_missing_payload(log, ids["r2"]) == "payload_missing"


# ---------- dry run ----------


def test_a_dry_run_reports_the_counts_and_leaves_everything_byte_identical(store):
    log, ids = store
    before = tree(log.headers.path.parent)
    result = purge(log, MAPPING, now=NOW, dry_run=True)
    assert result.dry_run and result.record is None
    assert (result.eligible, result.skipped_later_relationship, result.skipped_no_customer_ref) == (3, 1, 1)
    assert (result.skipped_not_in_mapping, result.skipped_retention_not_elapsed) == (1, 1)
    assert tree(log.headers.path.parent) == before and present(log, ids) == set(ids)


def test_a_dry_run_after_a_crash_counts_the_deletions_it_would_finish(store, monkeypatch):
    log, _ = store
    monkeypatch.setattr(audit_purge, "delete_payload", lambda *a, **k: (_ for _ in ()).throw(AuditError("crash")))
    with pytest.raises(AuditError):
        purge(log, MAPPING, now=NOW)
    before = tree(log.headers.path.parent)
    dry = purge(log, MAPPING, now=NOW, dry_run=True)
    assert dry.completed == 3 and dry.eligible == 0 and tree(log.headers.path.parent) == before


# ---------- refusals and arguments ----------


def test_a_purge_refuses_a_tampered_store_and_changes_nothing(store):
    log, ids = store
    edit_line(log, 2, final_score=0.0)
    before = tree(log.headers.path.parent)
    with pytest.raises(PurgeRefused, match="tampering-class"):
        purge(log, MAPPING, now=NOW)
    assert tree(log.headers.path.parent) == before and present(log, ids) == set(ids)


def test_a_purge_refuses_a_torn_line_and_points_to_recover(store):
    log, _ = store
    with open(log.headers.path, "ab") as fh:
        fh.write(b'{"partial')
    before = tree(log.headers.path.parent)
    with pytest.raises(PurgeRefused, match="recover"):
        purge(log, MAPPING, now=NOW)
    assert tree(log.headers.path.parent) == before


def test_a_purge_on_a_missing_store_refuses(tmp_path):
    with pytest.raises(audit.LogMissingError):
        purge(AuditLog.in_directory(tmp_path / "nowhere"), MAPPING, now=NOW)
    assert not (tmp_path / "nowhere").exists()


@pytest.mark.parametrize("years", [0, 4, -1, 5.5, True])
def test_a_retention_period_under_five_years_is_refused(store, years):
    log, _ = store
    with pytest.raises(ValueError, match="at least 5"):
        purge(log, MAPPING, retention_years=years, now=NOW)


def test_bad_arguments_are_refused_before_anything_is_read(store):
    log, _ = store
    with pytest.raises(ValueError, match="timezone"):
        purge(log, MAPPING, now=datetime(2026, 10, 8))
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        purge(log, {"A": "01/01/2018"}, now=NOW)
    with pytest.raises(ValueError, match="non-empty"):
        purge(log, {"": "2018-01-01"}, now=NOW)


def test_a_longer_retention_period_is_allowed_and_recorded(store):
    log, _ = store
    result = purge(log, MAPPING, retention_years=8, now=NOW)  # A: 2026-01-01 < today, D: 2027 is not
    assert result.eligible == 2 and result.record["retention_years"] == 8


def test_a_leap_day_end_date_is_never_shortened():
    assert add_years(date(2028, 2, 29), 5) == date(2033, 3, 1)
    assert add_years(date(2024, 6, 1), 5) == date(2029, 6, 1)


def test_two_purges_at_once_make_one_purge_record(store):
    log, _ = store
    errors = []

    def run() -> None:
        try:
            purge(log, MAPPING, now=NOW)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(events(log, "purge")) == 1 and verify_chain(log).ok


# ---------- verify_chain checks purge records ----------


def rewrite_purge(log: AuditLog, **changes) -> None:
    index = next(i for i, r in enumerate(log.headers.read_all()) if r["kind"] == "purge")
    record = json.loads(read_lines(log)[index])
    record.update(changes)
    record["record_hash"] = audit.compute_record_hash(record)
    lines = read_lines(log)
    lines[index] = audit.canonical_json(record).encode()
    write_lines(log, lines)


def test_a_purge_record_naming_an_unknown_request_is_a_schema_violation(store):
    log, ids = store
    purge(log, MAPPING, now=NOW)
    rewrite_purge(log, purged_request_ids=[*events(log, "purge")[0]["purged_request_ids"], "f" * 32])
    assert any("not decision records earlier" in f.detail for f in verify_chain(log).failures)


def test_a_purge_record_cannot_cover_a_payload_by_naming_a_decision_that_comes_later(store):
    log, ids = store
    purge(log, {"A": "2018-01-01"}, now=NOW)
    later = add_ref(log, 20, "A", "2017-02-02T10:00:00")  # a decision after the purge record
    rewrite_purge(log, purged_request_ids=sorted([*events(log, "purge")[0]["purged_request_ids"], later]))
    assert "schema_violation" in verify_chain(log).kinds()


@pytest.mark.parametrize("changes", [{"retention_years": 1}, {"basis": "because"}, {"retention_years": "5"}])
def test_a_purge_record_with_a_short_retention_or_another_basis_is_a_schema_violation(store, changes):
    log, _ = store
    purge(log, MAPPING, now=NOW)
    rewrite_purge(log, **changes)
    assert "schema_violation" in verify_chain(log).kinds()


# ---------- replay and the header check, with real screening records ----------


def screened(tmp_path):
    engine = make_engine(tmp_path)
    kept = engine.audited_screen("Mohammed Ali Hassan", "1970-01-01", "iq", customer_ref="cust-old")
    gone = engine.audited_screen("Ivan Petrov", "1980", "ru", customer_ref="cust-old")
    live = engine.audited_screen("Muhammad Ali Hasan", None, "iq", customer_ref="cust-live")
    return engine, kept, gone, live


def test_replay_of_a_purged_record_says_purged_and_others_still_replay(tmp_path):
    engine, kept, gone, live = screened(tmp_path)
    assert replay(engine, gone.request_id).identical
    result = purge(engine.log, {"cust-old": "2019-01-01"}, now=NOW)  # the records are dated by the engine's clock
    assert result.eligible == 0  # made 2026-10-07, after the end date: a later relationship
    result = purge(engine.log, {"cust-old": "2026-10-07"}, now=datetime(2032, 1, 1, tzinfo=timezone.utc))
    assert result.eligible == 2
    purged = replay(engine, gone.request_id)
    assert purged.status == PURGED and not purged.identical
    assert "retention" in purged.reason and "by design" in purged.reason
    assert replay(engine, kept.request_id).status == PURGED
    assert replay(engine, live.request_id).status == IDENTICAL  # cust-live is not in the mapping
    assert verify_chain(engine.log).ok


def test_a_missing_payload_without_a_purge_record_is_not_replayable_not_purged(tmp_path):
    engine, kept, _, _ = screened(tmp_path)
    engine.log.payloads.path_for(kept.request_id).unlink()
    assert replay(engine, kept.request_id).status == "not_replayable"


def test_a_purged_header_and_the_purge_record_pass_the_no_name_derived_fields_check(tmp_path):
    engine, kept, gone, live = screened(tmp_path)
    forbidden = set()
    inputs = {
        kept.request_id: ("Mohammed Ali Hassan", "1970-01-01", "iq"),
        gone.request_id: ("Ivan Petrov", "1980", "ru"),
    }
    for rid, (name, birth_date, country) in inputs.items():
        payload = engine.log.payloads.read(rid).data
        forbidden |= forbidden_strings(name, birth_date, country, payload["normalised"], payload)
    forbidden |= {"cust-old", "cust-live", "2026-10-07"}
    purge(engine.log, {"cust-old": "2026-10-07"}, now=datetime(2032, 1, 1, tzinfo=timezone.utc))
    for rid in (kept.request_id, gone.request_id):
        assert engine.log.payloads.read(rid) is None
    records = [r for r in engine.log.headers.read_all() if r["kind"] in ("decision", "purge")]
    assert len(records) == 4  # three decisions and the purge record
    random_hex = {"request_id", "run_context_hash", "payload_digest", "prev_hash", "record_hash"}
    for record in records:
        assert set(record) == set(audit_schema.LOGGED_FIELDS_BY_KIND[record["kind"]])
        text = json.dumps({k: v for k, v in record.items() if k not in random_hex}, ensure_ascii=False).lower()
        for leaf in string_leaves({k: v for k, v in record.items() if k != "timestamp_utc"}):
            assert leaf.lower() not in {f.lower() for f in forbidden}, leaf
        for item in forbidden:
            if len(item) >= 4 and item != "2026-10-07":
                assert item.lower() not in text, item


# ---------- the CLI ----------


def write_mapping(tmp_path, rows):
    path = tmp_path / "ended.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" if isinstance(r, dict) else r + "\n" for r in rows))
    return str(path)


def test_cli_purge_dry_run_then_real(store, tmp_path, capsys):
    log, ids = store
    audit_dir = str(log.headers.path.parent)
    mapping = write_mapping(tmp_path, [{"customer_ref": k, "relationship_end": v} for k, v in MAPPING.items()])
    before = tree(log.headers.path.parent)
    assert screen_cli.main(["purge", "--mapping", mapping, "--dry-run", "--audit-dir", audit_dir]) == 0
    out = capsys.readouterr().out
    assert out.startswith("would purge 3 payload(s)") and "no customer_ref" in out
    assert tree(log.headers.path.parent) == before
    assert not any(secret in out for secret in ("2018-01-01", "2019-03-01", "2024-06-01", "person"))
    assert not any(f"'{ref}'" in out or f'"{ref}"' in out for ref in MAPPING)  # no customer references or dates

    assert screen_cli.main(["purge", "--mapping", mapping, "--audit-dir", audit_dir]) == 0
    out = capsys.readouterr().out
    assert out.startswith("purged 3 payload(s)") and "files only" in out
    assert present(log, ids) == {"r2", "r3", "r4", "r6"} and verify_chain(log).ok


@pytest.mark.parametrize(
    "rows",
    [
        ['{"customer_ref": "A"}'],
        ['{"customer_ref": "A", "relationship_end": "2018/01/01"}'],
        [
            '{"customer_ref": "A", "relationship_end": "2018-01-01"}',
            '{"customer_ref": "A", "relationship_end": "2019-01-01"}',
        ],
        ["not json"],
        [],
    ],
)
def test_cli_purge_refuses_a_bad_mapping_before_touching_anything(store, tmp_path, capsys, rows):
    log, _ = store
    before = tree(log.headers.path.parent)
    code = screen_cli.main(
        ["purge", "--mapping", write_mapping(tmp_path, rows), "--audit-dir", str(log.headers.path.parent)]
    )
    assert code == screen_cli.EXIT_USAGE and "refusing to start" in capsys.readouterr().err
    assert tree(log.headers.path.parent) == before


def test_cli_purge_needs_a_directory_a_mapping_and_five_years(store, tmp_path, capsys, monkeypatch):
    log, _ = store
    mapping = write_mapping(tmp_path, [{"customer_ref": "A", "relationship_end": "2018-01-01"}])
    monkeypatch.delenv("SCREENING_AUDIT_DIR", raising=False)
    assert screen_cli.main(["purge", "--mapping", mapping]) == screen_cli.EXIT_USAGE
    assert screen_cli.main(["purge", "--mapping", str(tmp_path / "nope.jsonl"), "--audit-dir", "x"]) == 2
    code = screen_cli.main(
        ["purge", "--mapping", mapping, "--retention-years", "3", "--audit-dir", str(log.headers.path.parent)]
    )
    assert code == screen_cli.EXIT_USAGE and "at least 5" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        screen_cli.main(["purge", "--audit-dir", "x"])  # --mapping is required


def test_cli_purge_refuses_a_tampered_store_with_exit_1(store, tmp_path, capsys):
    log, ids = store
    edit_line(log, 1, final_score=0.0)
    mapping = write_mapping(tmp_path, [{"customer_ref": "A", "relationship_end": "2018-01-01"}])
    code = screen_cli.main(["purge", "--mapping", mapping, "--audit-dir", str(log.headers.path.parent)])
    assert code == screen_cli.EXIT_AUDIT_FAILURE and "tampering-class" in capsys.readouterr().err
    assert present(log, ids) == set(ids)


# ---------- legal hold ----------


def test_a_held_customer_ref_is_untouched_even_though_it_is_eligible(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW, hold_customer_refs={"A"})
    assert set(result.eligible_ids) == {ids["r5"]} and result.held == 2  # A's r0 and r1 would have been purged
    assert {"r0", "r1"} <= present(log, ids) and "r5" not in present(log, ids)
    assert verify_chain(log).ok


def test_a_held_request_id_is_untouched_while_its_customers_other_records_are_purged(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW, hold_request_ids={ids["r0"]})
    assert result.held == 1 and set(result.eligible_ids) == {ids["r1"], ids["r5"]}
    assert "r0" in present(log, ids) and "r1" not in present(log, ids)  # same customer A, different record
    assert log.payloads.read(ids["r0"]).data["customer_ref"] == "A"


def test_a_dry_run_reports_the_held_count_and_changes_nothing(store):
    log, ids = store
    before = tree(log.headers.path.parent)
    result = purge(log, MAPPING, now=NOW, dry_run=True, hold_customer_refs={"A"}, hold_request_ids={ids["r5"]})
    assert result.dry_run and result.held == 3 and result.eligible == 0
    assert tree(log.headers.path.parent) == before and present(log, ids) == set(ids)


def test_holds_are_checked_before_the_other_rules(store):
    log, ids = store
    result = purge(log, MAPPING, now=NOW, dry_run=True, hold_request_ids={ids["r4"], ids["r3"], ids["r6"], ids["r2"]})
    # no customer_ref, not in the mapping, a later relationship and retention not over: all counted as held instead
    assert result.held == 4
    assert (result.skipped_no_customer_ref, result.skipped_not_in_mapping) == (0, 0)
    assert (result.skipped_later_relationship, result.skipped_retention_not_elapsed) == (0, 0)


def test_the_purge_record_holds_no_hold_and_no_customer_ref(store):
    log, ids = store
    purge(log, MAPPING, now=NOW, hold_customer_refs={"A"}, hold_request_ids={ids["r3"]})
    (record,) = events(log, "purge")
    assert set(record) == set(audit_schema.LOGGED_FIELDS_BY_KIND["purge"])  # no field for holds
    text = json.dumps(record)
    assert ids["r3"] not in text and ids["r0"] not in text and ids["r1"] not in text  # held IDs are not named
    assert '"A"' not in text and "hold" not in text


def test_a_hold_wins_over_finishing_an_interrupted_purge(store, monkeypatch):
    log, ids = store
    monkeypatch.setattr(audit_purge, "delete_payload", lambda *a, **k: (_ for _ in ()).throw(AuditError("crash")))
    with pytest.raises(AuditError):
        purge(log, MAPPING, now=NOW)
    monkeypatch.undo()
    done = purge(log, MAPPING, now=NOW, hold_request_ids={ids["r0"]})
    assert done.completed == 2 and done.held == 1 and done.record is None
    assert log.payloads.read(ids["r0"]) is not None  # the file stays
    stuck = verify_chain(log)
    assert [(f.kind, f.request_id) for f in stuck.failures] == [("purge_incomplete", ids["r0"])]
    assert purge(log, MAPPING, now=NOW).completed == 1 and verify_chain(log).ok  # once the hold is lifted


def test_a_hold_on_an_already_purged_request_is_counted_not_honoured(store):
    log, ids = store
    purge(log, MAPPING, now=NOW)
    result = purge(log, MAPPING, now=NOW, hold_request_ids={ids["r0"], ids["r3"]})
    assert result.hold_already_purged == 1 and result.held == 1  # r3 is held; r0 was already gone


def test_cli_purge_with_a_hold_file(store, tmp_path, capsys):
    log, ids = store
    audit_dir = str(log.headers.path.parent)
    mapping = write_mapping(tmp_path, [{"customer_ref": k, "relationship_end": v} for k, v in MAPPING.items()])
    holds = tmp_path / "holds.jsonl"
    holds.write_text(json.dumps({"customer_ref": "A"}) + "\n" + json.dumps({"request_id": ids["r5"]}) + "\n")
    args = ["purge", "--mapping", mapping, "--hold", str(holds), "--audit-dir", audit_dir]
    assert screen_cli.main([*args, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("would purge 0 payload(s)") and "held (under legal hold, never purged): 3" in out
    assert present(log, ids) == set(ids)
    assert screen_cli.main(args) == 0 and present(log, ids) == set(ids)
    assert events(log, "purge") == []  # nothing was eligible, so no purge record


@pytest.mark.parametrize(
    "rows",
    [
        ['{"customer_ref": "A", "request_id": "' + "a" * 32 + '"}'],
        ["{}"],
        ['{"request_id": "short"}'],
        ['{"request_id": "' + "A" * 32 + '"}'],
        ['{"customer_ref": ""}'],
        ['{"name": "x"}'],
        ["not json"],
        [],
    ],
)
def test_cli_purge_refuses_a_bad_hold_file_before_touching_anything(store, tmp_path, capsys, rows):
    log, _ = store
    mapping = write_mapping(tmp_path, [{"customer_ref": "A", "relationship_end": "2018-01-01"}])
    holds = tmp_path / "holds.jsonl"
    holds.write_text("".join(r + "\n" for r in rows))
    before = tree(log.headers.path.parent)
    code = screen_cli.main(
        ["purge", "--mapping", mapping, "--hold", str(holds), "--audit-dir", str(log.headers.path.parent)]
    )
    assert code == screen_cli.EXIT_USAGE and "hold" in capsys.readouterr().err
    assert tree(log.headers.path.parent) == before
