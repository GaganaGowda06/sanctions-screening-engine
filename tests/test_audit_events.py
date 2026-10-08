"""Tests for record kinds: engine_start events, replay's refusal of non-decisions, schema checks, and recovery from a
torn final line (src/audit_recover.py).

Run:
    pytest tests/test_audit_events.py
"""

import hashlib
import json
import threading
from datetime import datetime, timezone

import pytest
from test_audit_chain import add, edit_line, read_lines, write_lines
from test_replay import ENTITIES, make_engine, tree

import audit
import audit_recover
import audit_schema
import run_context as rc
import screen_cli
from audit import AuditError, AuditLog
from audit_recover import RecoveryRefused, recover
from audit_verify import verify_chain
from audited_screening import AuditedEngine
from replay import NOT_A_DECISION, replay
from screening import ScreeningConfig, ScreeningEngine

MOMENT = datetime(2026, 10, 8, 9, 30, 0, 123456, tzinfo=timezone.utc)


def kinds(log: AuditLog) -> list[str]:
    return [r["kind"] for r in log.headers.read_all()]


def events(log: AuditLog, kind: str) -> list[dict]:
    return [r for r in log.headers.read_all() if r["kind"] == kind]


@pytest.fixture
def log(tmp_path) -> AuditLog:
    return AuditLog.create_new(tmp_path / "audit", fsync=False)


# ---------- engine_start ----------


def test_the_first_engine_build_logs_engine_start(tmp_path):
    engine = make_engine(tmp_path)
    (start,) = events(engine.log, "engine_start")
    assert set(start) == set(audit_schema.LOGGED_FIELDS_BY_KIND["engine_start"])
    assert start["context_hash"] == engine.context_hash and start["previous_context_hash"] is None
    assert start["changed_fields"] == [] and start["schema_version"] == audit_schema.SCHEMA_VERSION
    assert verify_chain(engine.log).ok and kinds(engine.log) == ["engine_start"]
    assert list((tmp_path / "audit" / "payloads").iterdir()) == []  # events have no payload


def test_an_identical_rebuild_logs_nothing(tmp_path):
    first = make_engine(tmp_path)
    before = first.log.headers.path.read_bytes()
    second = make_engine(tmp_path, log=first.log)
    third = make_engine(tmp_path, log=first.log)
    assert second.context_hash == third.context_hash == first.context_hash
    assert first.log.headers.path.read_bytes() == before and kinds(first.log) == ["engine_start"]


def test_a_threshold_change_logs_engine_start_naming_the_threshold_field(tmp_path):
    first = make_engine(tmp_path)
    second = make_engine(tmp_path, log=first.log, config=ScreeningConfig(global_threshold=0.85))
    start1, start2 = events(first.log, "engine_start")
    assert start2["context_hash"] == second.context_hash != first.context_hash
    assert start2["previous_context_hash"] == first.context_hash
    assert "config.global_threshold" in start2["changed_fields"]
    assert "config_file_sha256" not in start2["changed_fields"]  # only what differs
    assert verify_chain(first.log).ok


def newer_list(directory):
    directory.mkdir()
    body = b'{"id":"e1"}\n{"id":"e2"}\n'
    path = directory / "entities.ftm.json"
    path.write_bytes(body)
    (directory / "VERSION").write_text(f"version=20260202000000-new\nsha1={hashlib.sha1(body).hexdigest()}\n")  # noqa: S324
    return path


def test_a_snapshot_change_names_the_snapshot_fields(tmp_path):
    first = make_engine(tmp_path)
    other_engine = ScreeningEngine.from_entities(ENTITIES, ScreeningConfig())
    other_context = rc.build_run_context(
        other_engine,
        newer_list(tmp_path / "list2"),
        git=lambda: ("a" * 40, False),
        versions=lambda: {"rapidfuzz": "3.14.6"},
    )
    AuditedEngine(other_engine, other_context, first.log)
    start = events(first.log, "engine_start")[-1]
    assert {"snapshot_id", "snapshot_sha256"} <= set(start["changed_fields"])


def test_going_back_to_an_earlier_context_logs_again(tmp_path):
    first = make_engine(tmp_path)
    make_engine(tmp_path, log=first.log, config=ScreeningConfig(global_threshold=0.85))
    make_engine(tmp_path, log=first.log)  # back to the first configuration
    starts = events(first.log, "engine_start")
    assert len(starts) == 3 and starts[2]["context_hash"] == first.context_hash
    assert starts[2]["previous_context_hash"] == starts[1]["context_hash"]


def test_replay_builds_log_nothing_even_with_another_context(tmp_path):
    first = make_engine(tmp_path)
    before = tree(tmp_path / "audit")
    make_engine(tmp_path, log=first.log, config=ScreeningConfig(global_threshold=0.5), for_screening=False)
    make_engine(tmp_path, log=first.log, for_screening=False)
    assert tree(tmp_path / "audit") == before


def test_engine_start_holds_field_names_only_never_values(tmp_path):
    first = make_engine(tmp_path)
    second = make_engine(tmp_path, log=first.log, config=ScreeningConfig(global_threshold=0.85, blend=0.4))
    start = events(first.log, "engine_start")[-1]
    paths = {p for p, *_ in rc._differences(first.run_context.as_dict(), second.run_context.as_dict())}
    assert set(start["changed_fields"]) == paths
    text = json.dumps(start)
    for value in ("0.85", "0.4", "0.86", "a" * 40):
        assert f'"{value}"' not in text
    assert all(isinstance(f, str) and f.replace(".", "").replace("_", "").isalnum() for f in start["changed_fields"])


def test_a_missing_previous_context_file_is_recorded_as_such(tmp_path):
    first = make_engine(tmp_path)
    (first.contexts_dir / f"{first.context_hash}.json").unlink()
    make_engine(tmp_path, log=first.log, config=ScreeningConfig(global_threshold=0.85))
    assert events(first.log, "engine_start")[-1]["changed_fields"] == ["context_file_missing"]


def test_engines_starting_together_log_one_engine_start(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    prepared = [make_engine(tmp_path, log=log, for_screening=False) for _ in range(6)]  # built one at a time
    errors = []

    def start(engine: AuditedEngine) -> None:
        try:
            AuditedEngine(engine.engine, engine.run_context, log)  # the part under test: the start record
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(e,)) for e in prepared]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and kinds(log) == ["engine_start"]


def test_an_engine_that_cannot_log_its_start_does_not_start(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    add(log, 0)
    edit_line(log, 0, final_score=0.0)  # the chain's head is now damaged
    with pytest.raises(AuditError):
        make_engine(tmp_path, log=log)


def test_screening_still_works_between_engine_starts(tmp_path):
    engine = make_engine(tmp_path)
    for i in range(3):
        engine.audited_screen("Ivan Petrov", customer_ref=f"c{i}")
    assert kinds(engine.log) == ["engine_start", "decision", "decision", "decision"]
    result = verify_chain(engine.log)
    assert result.ok and result.records == 4 and result.payloads_checked == 3


# ---------- replay and verify handle each kind ----------


def test_replay_refuses_an_engine_start_record_with_a_clear_message(tmp_path):
    engine = make_engine(tmp_path)
    (start,) = events(engine.log, "engine_start")
    result = replay(engine, start["request_id"])
    assert result.status == NOT_A_DECISION and not result.identical
    assert "engine_start" in result.reason and "not a screening decision" in result.reason


def test_replay_refuses_a_recovery_record_too(log):
    log.headers.path.write_bytes(b"")
    add(log, 0)
    with open(log.headers.path, "ab") as fh:
        fh.write(b'{"torn')
    recover(log, clock=lambda: MOMENT)
    (record,) = events(log, "recovery")
    engine_less = type("E", (), {"log": log})()  # replay reads only the header here
    result = replay(engine_less, record["request_id"])
    assert result.status == NOT_A_DECISION and "recovery" in result.reason


def test_a_record_of_an_unknown_kind_or_with_extra_fields_is_a_schema_violation(log):
    for i in range(3):
        add(log, i)
    record = json.loads(read_lines(log)[1])
    record["kind"] = "mystery"
    record["record_hash"] = audit.compute_record_hash(record)
    lines = read_lines(log)
    lines[1] = audit.canonical_json(record).encode()
    write_lines(log, lines)
    assert "schema_violation" in verify_chain(log).kinds()

    log2_lines = read_lines(log)
    other = json.loads(log2_lines[1])
    other["kind"], other["extra"] = "decision", "x"
    other["record_hash"] = audit.compute_record_hash(other)
    log2_lines[1] = audit.canonical_json(other).encode()
    write_lines(log, log2_lines)
    result = verify_chain(log)
    assert any(f.kind == "schema_violation" and "extra" in f.detail for f in result.failures)


def test_each_kind_has_its_own_exact_allow_list(tmp_path):
    engine = make_engine(tmp_path)
    engine.audited_screen("Ivan Petrov")
    for record in engine.log.headers.read_all():
        assert set(record) == set(audit_schema.LOGGED_FIELDS_BY_KIND[record["kind"]])
    assert set(audit_schema.FIELDS_BY_KIND) == set(audit_schema.KINDS)
    assert "payload_digest" not in audit_schema.ENGINE_START_FIELDS + audit_schema.RECOVERY_FIELDS


# ---------- recovery from a crash during a write ----------


def torn(log: AuditLog, fragment: bytes = b'{"candidates":[{"entity_id":"e1","sco') -> bytes:
    with open(log.headers.path, "ab") as fh:
        fh.write(fragment)
    return fragment


def orphan(log: AuditLog, i: int = 90, text: str = "left behind") -> str:
    """A payload whose header was never written."""
    rid = f"{i:032x}"
    log.payloads.write(audit.PayloadRecord(rid, b"s" * 32, {"raw_input": {"name": text}}))
    return rid


def test_recovery_on_a_torn_line_gives_a_clean_chain_with_one_recovery_record(log):
    for i in range(3):
        add(log, i)
    complete = log.headers.path.read_bytes()
    fragment = torn(log)
    assert verify_chain(log).kinds() == {"torn_final_line"}
    outcome = recover(log, clock=lambda: MOMENT)

    result = verify_chain(log)
    assert result.ok, result.failures
    (record,) = events(log, "recovery")
    assert record["failure_types"] == ["torn_final_line"]
    assert record["quarantine_size"] == len(fragment) == outcome.quarantine_size
    assert record["quarantine_sha256"] == hashlib.sha256(fragment).hexdigest()
    assert record["orphan_request_id"] is None and record["orphan_payload_sha256"] is None
    quarantined = (log.headers.path.parent / "quarantine" / record["quarantine_file"]).read_bytes()
    assert quarantined == fragment and hashlib.sha256(quarantined).hexdigest() == record["quarantine_sha256"]
    assert record["quarantine_file"] == "20261008T093000123456Z.bin"
    assert log.headers.path.read_bytes().startswith(complete)  # the complete lines were not touched
    assert log.headers.path.read_bytes().count(b"\n") == 4 and result.records == 4
    assert log.headers.read_checkpoint()["count"] == 4


def test_a_torn_line_plus_its_orphan_payload_gives_a_clean_chain_with_both_recorded(log):
    add(log, 0)
    rid = orphan(log)  # the crash came after the payload write and part way through the header write
    payload_bytes = log.payloads.path_for(rid).read_bytes()
    fragment = torn(log)
    assert verify_chain(log).kinds() == {"torn_final_line", "orphan_payload"}
    outcome = recover(log, clock=lambda: MOMENT)

    result = verify_chain(log)
    assert result.ok, result.failures
    assert not log.payloads.path_for(rid).exists()  # deleted, not quarantined
    assert not any(rid in p.name for p in (log.headers.path.parent / "quarantine").iterdir())
    (record,) = events(log, "recovery")
    assert record["failure_types"] == ["torn_final_line", "orphan_payload"]
    assert record["quarantine_sha256"] == hashlib.sha256(fragment).hexdigest()
    assert record["orphan_request_id"] == rid == outcome.orphan_request_id
    assert record["orphan_payload_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert "left behind" not in json.dumps(record)  # no payload content in the chain
    assert all(
        "left behind" not in p.read_text(errors="ignore") for p in (log.headers.path.parent).rglob("*") if p.is_file()
    )


def test_an_orphan_only_crash_before_any_header_byte_gives_a_clean_chain(log):
    add(log, 0)
    rid = orphan(log)
    assert verify_chain(log).kinds() == {"orphan_payload"}
    before_log = log.headers.path.read_bytes()
    outcome = recover(log, clock=lambda: MOMENT)
    assert verify_chain(log).ok
    (record,) = events(log, "recovery")
    assert record["failure_types"] == ["orphan_payload"] and record["orphan_request_id"] == rid
    assert (
        record["quarantine_file"] is None and record["quarantine_sha256"] is None and record["quarantine_size"] is None
    )
    assert outcome.quarantine_file is None
    assert not (log.headers.path.parent / "quarantine").exists()  # no torn bytes, no quarantine
    assert log.headers.path.read_bytes().startswith(before_log)


def test_recovery_when_the_very_first_record_was_torn(log):
    fragment = torn(log, b'{"schema_ver')
    recover(log, clock=lambda: MOMENT)
    assert verify_chain(log).ok
    (record,) = log.headers.read_all()
    assert record["kind"] == "recovery" and record["seq"] == 0 and record["quarantine_size"] == len(fragment)


def test_two_orphan_payloads_are_refused_and_nothing_changes(log):
    add(log, 0)
    orphan(log, 90)
    orphan(log, 91)
    torn(log)
    before = snapshot(log)
    with pytest.raises(RecoveryRefused, match="2 orphan payloads") as stop:
        recover(log, clock=lambda: MOMENT)
    assert "orphan_payload" in stop.value.kinds
    assert snapshot(log) == before and not (log.headers.path.parent / "quarantine").exists()
    for rid in (f"{90:032x}", f"{91:032x}"):
        assert log.payloads.read(rid) is not None


def test_two_orphans_without_a_torn_line_are_refused_too(log):
    add(log, 0)
    orphan(log, 90)
    orphan(log, 91)
    before = snapshot(log)
    with pytest.raises(RecoveryRefused, match="2 orphan payloads"):
        recover(log, clock=lambda: MOMENT)
    assert snapshot(log) == before


def test_a_file_in_the_payload_directory_that_is_not_a_payload_is_not_deleted(log):
    add(log, 0)
    stray = log.payloads.directory / "notes.txt"
    stray.write_text("not ours")
    before = snapshot(log)
    with pytest.raises(RecoveryRefused):
        recover(log, clock=lambda: MOMENT)
    assert snapshot(log) == before and stray.exists()


def test_a_crash_between_the_recovery_record_and_the_deletion_is_finished_by_running_recovery_again(log, monkeypatch):
    add(log, 0)
    rid = orphan(log)
    real = audit_recover.delete_payload
    monkeypatch.setattr(audit_recover, "delete_payload", lambda *a, **k: (_ for _ in ()).throw(AuditError("crash")))
    with pytest.raises(AuditError, match="crash"):
        recover(log, clock=lambda: MOMENT)
    assert log.payloads.read(rid) is not None and len(events(log, "recovery")) == 1  # recorded, not yet deleted
    assert verify_chain(log).kinds() == {"orphan_payload"}
    monkeypatch.setattr(audit_recover, "delete_payload", real)
    outcome = recover(log, clock=lambda: MOMENT)
    assert outcome.record is None  # no second recovery record
    assert len(events(log, "recovery")) == 1 and log.payloads.read(rid) is None and verify_chain(log).ok


def test_screening_works_after_recovery(tmp_path):
    engine = make_engine(tmp_path)
    engine.audited_screen("Ivan Petrov")
    rid = orphan(engine.log)
    with open(engine.log.headers.path, "ab") as fh:
        fh.write(b'{"partial')
    with pytest.raises(AuditError, match="torn"):
        engine.audited_screen("Mohammed Ali Hassan")
    recover(engine.log, clock=lambda: MOMENT)
    again = engine.audited_screen("Mohammed Ali Hassan")
    assert again.matches and kinds(engine.log) == ["engine_start", "decision", "recovery", "decision"]
    assert verify_chain(engine.log).ok and engine.log.payloads.read(rid) is None
    assert replay(engine, again.request_id).identical
    rebuilt = make_engine(tmp_path, log=engine.log)  # a new process after the recovery starts normally
    rebuilt.audited_screen("Acme Trading")
    assert verify_chain(engine.log).ok


def snapshot(log: AuditLog) -> dict[str, bytes]:
    return tree(log.headers.path.parent)


def test_recovery_refuses_a_clean_log_and_changes_nothing(log):
    add(log, 0)
    before = snapshot(log)
    with pytest.raises(RecoveryRefused, match="nothing to recover"):
        recover(log, clock=lambda: MOMENT)
    assert snapshot(log) == before and not (log.headers.path.parent / "quarantine").exists()


def damaged_store(log: AuditLog, how: str) -> None:
    """A store with a torn line and an orphan, and one tampering-class failure of the named type."""
    for i in range(4):
        add(log, i)
    if how == "edited_header":
        edit_line(log, 1, final_score=0.0)
    elif how == "broken_link":
        lines = read_lines(log)
        write_lines(log, lines[:1] + lines[2:])
    elif how == "tail_truncated":
        write_lines(log, read_lines(log)[:2])
    elif how == "payload_digest_mismatch":
        path = log.payloads.path_for(f"{1:032x}")
        stored = json.loads(path.read_text())
        stored["data"]["raw_input"]["name"] = "changed"
        path.write_text(json.dumps(stored))
    elif how == "payload_missing":
        log.payloads.path_for(f"{1:032x}").unlink()
    elif how == "checkpoint_missing":
        log.headers.checkpoint_path.unlink()
    elif how == "checkpoint_mismatch":
        log.headers.checkpoint_path.write_text(audit.canonical_json({"count": 4, "record_hash": "e" * 64}))
    elif how == "log_missing":
        log.headers.path.unlink()
        log.headers.checkpoint_path.unlink()
    elif how == "schema_violation":
        record = json.loads(read_lines(log)[1])
        record["kind"] = "mystery"
        record["record_hash"] = audit.compute_record_hash(record)
        lines = read_lines(log)
        lines[1] = audit.canonical_json(record).encode()
        write_lines(log, lines)
    elif how == "quarantine_mismatch":
        torn(log)
        recover(log, clock=lambda: MOMENT)
        (next((log.headers.path.parent / "quarantine").iterdir())).write_bytes(b"swapped")
    else:
        raise AssertionError(how)
    if log.headers.path.exists():
        orphan(log)
        torn(log)


@pytest.mark.parametrize(
    "how",
    [
        "edited_header", "broken_link", "tail_truncated", "payload_digest_mismatch", "payload_missing",
        "checkpoint_missing", "checkpoint_mismatch", "schema_violation", "quarantine_mismatch",
    ],
)  # fmt: skip
def test_every_tampering_type_is_still_refused_with_every_file_unchanged(log, how):
    damaged_store(log, how)
    before = snapshot(log)
    if how == "checkpoint_missing":  # the store counts as not there: refused before it is even verified
        with pytest.raises(audit.LogMissingError):
            recover(log, clock=lambda: MOMENT)
    else:
        with pytest.raises(RecoveryRefused, match="tampering-class") as stop:
            recover(log, clock=lambda: MOMENT)
        assert how in stop.value.kinds and how in str(stop.value)
    assert snapshot(log) == before


def test_a_missing_log_file_is_refused_as_log_missing(log):
    damaged_store(log, "log_missing")
    before = snapshot(log)
    with pytest.raises(audit.LogMissingError):
        recover(log, clock=lambda: MOMENT)
    assert snapshot(log) == before


def test_findings_other_than_the_two_crash_signatures_are_refused(log):
    add(log, 0)
    saved = log.headers.checkpoint_path.read_text()
    add(log, 1)
    log.headers.checkpoint_path.write_text(saved)  # checkpoint_behind: a different crash, repaired by the next append
    assert verify_chain(log).kinds() == {"checkpoint_behind"}
    before = snapshot(log)
    with pytest.raises(RecoveryRefused, match="other than"):
        recover(log, clock=lambda: MOMENT)
    assert snapshot(log) == before


def test_recovery_refuses_a_missing_store(tmp_path):
    with pytest.raises(audit.LogMissingError):
        recover(AuditLog.in_directory(tmp_path / "nowhere"))
    assert not (tmp_path / "nowhere").exists()


def test_a_quarantine_name_collision_stops_recovery_before_anything_is_cut_or_deleted(log):
    add(log, 0)
    rid = orphan(log)
    torn(log)
    quarantine = log.headers.path.parent / "quarantine"
    quarantine.mkdir()
    (quarantine / "20261008T093000123456Z.bin").write_bytes(b"already here")
    before = snapshot(log)
    with pytest.raises(AuditError, match="could not quarantine"):
        recover(log, clock=lambda: MOMENT)
    assert snapshot(log) == before and log.payloads.read(rid) is not None  # still torn, nothing lost
    assert verify_chain(log).kinds() == {"torn_final_line", "orphan_payload"}


def test_a_quarantine_file_that_no_longer_matches_its_recovery_record_is_reported(log):
    add(log, 0)
    torn(log)
    outcome = recover(log, clock=lambda: MOMENT)
    assert verify_chain(log).ok
    (log.headers.path.parent / "quarantine" / outcome.quarantine_file).write_bytes(b"swapped")
    assert verify_chain(log).kinds() == {"quarantine_mismatch"}
    (log.headers.path.parent / "quarantine" / outcome.quarantine_file).unlink()
    assert verify_chain(log).ok  # a file that is gone is not a chain failure


def rewrite_recovery(log: AuditLog, **changes) -> None:
    index = next(i for i, r in enumerate(log.headers.read_all()) if r["kind"] == "recovery")
    record = json.loads(read_lines(log)[index])
    record.update(changes)
    record["record_hash"] = audit.compute_record_hash(record)
    lines = read_lines(log)
    lines[index] = audit.canonical_json(record).encode()
    write_lines(log, lines)


def test_a_recovery_record_cannot_name_a_path_outside_quarantine(log):
    add(log, 0)
    torn(log)
    recover(log, clock=lambda: MOMENT)
    rewrite_recovery(log, quarantine_file="../headers.jsonl")
    assert "schema_violation" in verify_chain(log).kinds()


def test_an_inconsistent_recovery_record_is_a_schema_violation(log):
    add(log, 0)
    torn(log)
    recover(log, clock=lambda: MOMENT)
    rewrite_recovery(log, quarantine_sha256=None)  # the torn-line fields only partly set
    assert any("partly set" in f.detail for f in verify_chain(log).failures)


# ---------- the CLI command ----------


def test_cli_recover_exits_0_on_a_crash_and_1_on_a_clean_log(log, capsys):
    add(log, 0)
    audit_dir = str(log.headers.path.parent)
    assert screen_cli.main(["recover", "--audit-dir", audit_dir]) == 1
    assert "nothing to recover" in capsys.readouterr().err
    rid = orphan(log)
    torn(log)
    assert screen_cli.main(["recover", "--audit-dir", audit_dir]) == 0
    out = capsys.readouterr().out
    assert out.startswith("recovered:") and "quarantine/" in out and rid in out
    assert verify_chain(log).ok
    assert screen_cli.main(["recover", "--audit-dir", audit_dir]) == 1  # clean again


def test_cli_recover_needs_an_audit_directory(monkeypatch, capsys):
    monkeypatch.delenv("SCREENING_AUDIT_DIR", raising=False)
    assert screen_cli.main(["recover"]) == screen_cli.EXIT_USAGE
    assert "refusing to start" in capsys.readouterr().err
