"""Tests for batch monitoring metrics (src/audit_metrics.py) and duration_ms in decision headers.

The hand-built log has 20 decisions with chosen scores, names and durations, so every figure in the report can be
worked out by hand (see the table in `SCENARIO`).

Run:
    pytest tests/test_audit_metrics.py
"""

import json
from datetime import datetime, timezone

import pytest
from test_audit_chain import edit_line
from test_audited_screening import forbidden_strings, string_leaves
from test_replay import make_engine, tree

import audit
import audit_metrics
import audit_schema
import screen_cli
from audit import AuditError, AuditLog
from audit_metrics import MetricsConfig, MetricsRefused, bin_index, compute_metrics, percentile
from audit_purge import purge
from audit_verify import verify_chain

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
CLOCK = lambda: NOW  # noqa: E731
CTX_X, CTX_Y = "a" * 64, "b" * 64
LATIN, CYRILLIC, ARABIC = "John Smith", "Иван Петров", "محمد علي"
BAND_05 = MetricsConfig(near_threshold_band=0.05, min_cell=3)

# (name, final score or None, duration_ms). Threshold 0.86 throughout.
SCENARIO = (
    [(LATIN, s, d) for s, d in zip([0.10, 0.10, 0.30, 0.55, 0.84, 0.86, 0.91, 1.0], range(1, 9), strict=True)]
    + [(ARABIC, s, d) for s, d in zip([0.20, 0.50, 0.87, 0.88, 0.95, 0.89], range(9, 15), strict=True)]
    + [(CYRILLIC, s, d) for s, d in zip([0.05, 0.60, 0.70, 0.865], range(15, 19), strict=True)]
    + [(LATIN, None, 19), (LATIN, None, 20)]
)


def add_decision(log, i, name, score, duration, context=CTX_X, ref=None, when="2026-10-07T10:00:00") -> str:
    request_id = f"{i:032x}"
    payload = audit.PayloadRecord(
        request_id, bytes([i % 250 + 1]) * 32, {"customer_ref": ref, "raw_input": {"name": name}}
    )
    entries = (
        [] if score is None else [audit_schema.candidate_entry("e1", score, score, 0.0, 0.0, 0.0, False, False, 0.86)]
    )
    header = audit_schema.build_header(
        request_id=request_id, moment=datetime.fromisoformat(when).replace(tzinfo=timezone.utc),
        run_context_hash=context, entries=entries, threshold=0.86, digest=payload.digest(), duration_ms=duration,
    )  # fmt: skip
    log.write(header, payload)
    return request_id


@pytest.fixture
def scenario(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i, (name, score, duration) in enumerate(SCENARIO):
        add_decision(log, i, name, score, duration)
    return log


def assert_not_recoverable(report: dict) -> None:
    """No hidden count can be worked out as a total minus the visible counts: a group never has exactly one hidden
    per-script cell, and the window never has exactly one hidden group while it still gives the decision total."""
    for group in report["groups"]:
        if not isinstance(group["count"], int):
            assert set(group) == {"run_context_hash", "count"}  # a suppressed group shows nothing else
            continue
        hidden = [k for k, v in group["per_script"].items() if not isinstance(v["count"], int)]
        assert len(hidden) != 1, f"one hidden cell {hidden} could be recovered by subtraction: {group['per_script']}"
        assert all(isinstance(v["count"], int) or set(v) == {"count"} for v in group["per_script"].values())
    hidden_groups = [g for g in report["groups"] if not isinstance(g["count"], int)]
    if hidden_groups:
        assert report["window"]["decisions"] == "withheld"  # else the one hidden group is total minus the others


def measured(*args, **kwargs):
    """compute_metrics, plus the check that nothing hidden in the report can be recovered by subtraction."""
    report, path = compute_metrics(*args, **kwargs)
    assert_not_recoverable(report)
    return report, path


def only_group(report: dict) -> dict:
    (group,) = report["groups"]
    return group


# ---------- exact metrics on a hand-built log ----------


def test_a_hand_built_log_gives_exactly_the_expected_metrics(scenario):
    report, path = measured(scenario.headers.path.parent, 0, 19, BAND_05, CLOCK)
    group = only_group(report)
    assert group["run_context_hash"] == CTX_X
    assert group["count"] == 20 and group["alerts"] == 8 and group["alert_rate"] == 0.4
    expected_bins = [0] * 20
    for index, n in {1: 1, 2: 2, 4: 1, 6: 1, 10: 1, 11: 1, 12: 1, 14: 1, 16: 1, 17: 5, 18: 1, 19: 2}.items():
        expected_bins[index] = n
    assert group["score_histogram"] == {"counts": expected_bins, "no_candidates": 2}
    assert sum(expected_bins) + 2 == 20
    assert group["near_threshold"] == {"count": 7, "rate": 0.35}  # 0.84 0.86 0.91 | 0.87 0.88 0.89 | 0.865
    assert group["per_script"] == {
        "Latin": {"count": 10, "alert_rate": 0.3},
        "Cyrillic": {"count": 4, "alert_rate": 0.25},
        "Arabic": {"count": 6, "alert_rate": 4 / 6},
    }
    assert group["duration_ms"] == {"p50": 10, "p95": 19, "p99": 20}
    stamp = "2026-10-07T10:00:00.000000+00:00"
    assert report["window"] == {
        "from_seq": 0, "to_seq": 19, "records": 20, "decisions": 20,
        "first_timestamp_utc": stamp, "last_timestamp_utc": stamp,
    }  # fmt: skip
    assert report["histogram"]["bins"] == 20 and report["histogram"]["edges"][:3] == [0.0, 0.05, 0.1]
    assert report["generated_at_utc"] == "2026-10-08T12:00:00+00:00"
    assert json.loads(path.read_text()) == report and path.name == "0-19.json"


def test_a_sub_window_measures_only_its_own_records(scenario):
    report, path = measured(scenario.headers.path.parent, 8, 13, BAND_05, CLOCK)  # the six Arabic records
    group = only_group(report)
    assert (
        group["count"] == 6
        and group["alerts"] == 4
        and group["per_script"] == {"Arabic": {"count": 6, "alert_rate": 4 / 6}}
    )
    assert group["duration_ms"] == {"p50": 11, "p95": 14, "p99": 14} and path.name == "8-13.json"


def test_the_near_threshold_boundary_is_inclusive(scenario):
    # 0.91 is exactly one band above 0.86: counted despite float noise in 0.91 - 0.86
    assert 0.91 - 0.86 > 0.05
    report, _ = measured(scenario.headers.path.parent, 6, 6, MetricsConfig(0.05, 1), CLOCK)
    assert only_group(report)["near_threshold"]["count"] == 1


def test_the_histogram_bin_edges_are_fixed_and_a_score_of_one_is_in_the_last_bin():
    assert [bin_index(s) for s in (0.0, 0.049, 0.05, 0.3, 0.55, 0.85, 0.95, 1.0)] == [0, 0, 1, 6, 11, 17, 19, 19]
    assert audit_metrics.BINS == 20


def test_percentiles_are_nearest_rank():
    values = list(range(1, 21))
    assert [percentile(values, p) for p in (50, 95, 99)] == [10, 19, 20]
    assert percentile([5], 99) == 5 and percentile([], 50) is None


# ---------- contexts are never mixed ----------


def test_two_contexts_in_a_window_give_two_separate_groups(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i in range(8):
        add_decision(log, i, LATIN, 0.9 if i % 2 else 0.1, 10, context=CTX_X)
    for i in range(8, 12):
        add_decision(log, i, LATIN, 0.95, 40, context=CTX_Y)
    add_decision(log, 12, LATIN, 0.1, 10, context=CTX_X)  # back to the first context
    report, _ = measured(log.headers.path.parent, 0, 12, BAND_05, CLOCK)
    x, y = report["groups"]
    assert (x["run_context_hash"], x["count"], x["alerts"]) == (CTX_X, 9, 4)
    assert (y["run_context_hash"], y["count"], y["alerts"]) == (CTX_Y, 4, 4)
    assert y["duration_ms"] == {"p50": 40, "p95": 40, "p99": 40} and x["duration_ms"]["p99"] == 10
    assert y["score_histogram"]["counts"][19] == 4 and x["score_histogram"]["counts"][19] == 0


def test_a_real_threshold_change_starts_a_new_group(tmp_path):
    from screening import ScreeningConfig

    first = make_engine(tmp_path)
    first.audited_screen("Ivan Petrov")
    second = make_engine(tmp_path, log=first.log, config=ScreeningConfig(global_threshold=0.5))
    second.audited_screen("Ivan Petrov")
    report, _ = measured(
        first.log.headers.path.parent, 0, len(first.log.headers.read_all()) - 1, MetricsConfig(0.05, 1), CLOCK
    )
    assert [g["run_context_hash"] for g in report["groups"]] == [first.context_hash, second.context_hash]
    assert report["window"]["records"] == 4 and report["window"]["decisions"] == 2  # two engine_start events left out


# ---------- script, suppression ----------


def test_a_purged_payload_counts_as_unavailable(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i in range(6):
        add_decision(log, i, ARABIC, 0.9, 5, ref="R", when="2017-01-01T10:00:00")
    for i in range(6, 9):
        add_decision(log, i, LATIN, 0.1, 5, ref="live")
    purge(log, {"R": "2018-01-01"}, now=NOW)
    assert verify_chain(log).ok
    report, _ = measured(log.headers.path.parent, 0, 9, BAND_05, CLOCK)  # seq 9 is the purge record
    group = only_group(report)
    assert group["per_script"] == {
        "Latin": {"count": 3, "alert_rate": 0.0},
        "unavailable": {"count": 6, "alert_rate": 1.0},
    }
    assert report["window"]["records"] == 10 and report["window"]["decisions"] == 9


def test_a_missing_or_damaged_payload_without_a_purge_record_is_refused_not_counted_as_unavailable(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    ids = [add_decision(log, i, LATIN, 0.1, 5) for i in range(3)]
    log.payloads.path_for(ids[0]).unlink()
    with pytest.raises(MetricsRefused, match="payload_missing"):
        compute_metrics(log.headers.path.parent, 0, 2, BAND_05, CLOCK)
    log.payloads.path_for(ids[0]).write_text("{broken")
    with pytest.raises(MetricsRefused, match="payload_digest_mismatch"):
        compute_metrics(log.headers.path.parent, 0, 2, BAND_05, CLOCK)
    assert not (log.headers.path.parent / "monitoring").exists()


def test_a_group_below_min_cell_is_suppressed(scenario):
    report, _ = measured(scenario.headers.path.parent, 0, 19, MetricsConfig(0.05, 10), CLOCK)
    per_script = only_group(report)["per_script"]
    assert per_script["Latin"] == {"count": 10, "alert_rate": 0.3}  # 10 is not below 10
    assert per_script["Arabic"] == {"count": "<10"} and per_script["Cyrillic"] == {
        "count": "<10"
    }  # two hidden: no third
    assert "alert_rate" not in per_script["Arabic"] and "alert_rate" not in per_script["Cyrillic"]
    assert only_group(report)["count"] == 20  # the group as a whole is reported


def test_min_cell_comes_from_the_config_and_names_itself_in_the_suppressed_count(scenario):
    report, _ = measured(scenario.headers.path.parent, 0, 19, MetricsConfig(0.05, 5), CLOCK)
    per_script = only_group(report)["per_script"]
    assert per_script["Cyrillic"] == {"count": "<5"}  # primary: 4 is below 5
    assert per_script["Arabic"] == {"count": "withheld"}  # secondary: the smallest visible cell (6) goes too
    assert per_script["Latin"]["count"] == 10


def test_the_script_comes_from_the_stage_3_detection_function(tmp_path):
    import screening

    assert audit_metrics.query_script is screening.query_script  # not a second implementation
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i, name in enumerate(["John Smith", "Иван Петров", "محمد علي", "田中太郎"]):
        add_decision(log, i, name, 0.1, 1)
    report, _ = measured(log.headers.path.parent, 0, 3, MetricsConfig(0.05, 1), CLOCK)
    assert set(only_group(report)["per_script"]) == {"Latin", "Cyrillic", "Arabic", "other"}


# ---------- config ----------


def test_the_default_config_is_the_file_and_bad_values_are_refused(tmp_path):
    assert MetricsConfig.load() == MetricsConfig(0.05, 10)
    assert MetricsConfig.load(tmp_path / "missing.json") == MetricsConfig()
    custom = tmp_path / "c.json"
    custom.write_text('{"near_threshold_band": 0.1, "min_cell": 20}')
    assert MetricsConfig.load(custom) == MetricsConfig(0.1, 20)
    for band, cell in ((-0.1, 10), (1.5, 10), (0.05, 0), (0.05, True), (0.05, 2.5)):
        with pytest.raises(ValueError):
            MetricsConfig(band, cell)


# ---------- nothing personal, no request IDs, nothing changed ----------


def test_no_report_field_holds_a_name_derived_value_or_a_request_id(tmp_path):
    engine = make_engine(tmp_path)
    inputs = [
        ("Mohammed Ali Hassan", "1970-01-01", "iq"),
        ("Ivan Petrov", "1980", "ru"),
        ("Zebulon Quixotic", None, None),
    ]
    request_ids, forbidden = [], set()
    for name, birth_date, country in inputs:
        result = engine.audited_screen(name, birth_date, country, customer_ref=f"cust-{name[:3]}")
        request_ids.append(result.request_id)
        payload = engine.log.payloads.read(result.request_id).data
        forbidden |= forbidden_strings(name, birth_date, country, payload["normalised"], payload)
    forbidden |= {"cust-Moh", "cust-Iva", "cust-Zeb"}
    report, path = measured(engine.log.headers.path.parent, 0, 3, MetricsConfig(0.05, 1), CLOCK)
    text = json.dumps(report).lower()
    for item in forbidden:
        if len(item) >= 4:
            assert item.lower() not in text, item
    for leaf in string_leaves(report):
        assert leaf.lower() not in {f.lower() for f in forbidden}, leaf
    assert not any(rid in text for rid in request_ids) and not any(rid in path.read_text() for rid in request_ids)
    assert not any(rid in text for rid in [h["request_id"] for h in engine.log.headers.read_all()])  # events too


def test_the_report_has_only_the_expected_keys(scenario):
    report, _ = measured(scenario.headers.path.parent, 0, 19, BAND_05, CLOCK)
    assert set(report) == {
        "report_schema_version", "generated_at_utc", "window", "config", "integrity", "histogram", "groups",
    }  # fmt: skip
    assert set(only_group(report)) == {
        "run_context_hash", "count", "alerts", "alert_rate", "score_histogram", "near_threshold", "per_script",
        "duration_ms",
    }  # fmt: skip


def test_metrics_leave_the_audit_log_byte_identical(scenario):
    directory = scenario.headers.path.parent
    before = tree(directory)
    compute_metrics(directory, 0, 19, BAND_05, CLOCK)
    after = tree(directory)
    assert set(after) - set(before) == {"monitoring/0-19.json"}  # the only new file
    assert {k: v for k, v in after.items() if k in before} == before  # every existing file is untouched
    assert verify_chain(scenario).ok


def test_a_report_for_a_window_is_never_overwritten(scenario):
    directory = scenario.headers.path.parent
    _, path = measured(directory, 0, 19, BAND_05, CLOCK)
    first = path.read_text()
    with pytest.raises(AuditError, match="already exists"):
        compute_metrics(directory, 0, 19, MetricsConfig(0.2, 1), CLOCK)
    assert path.read_text() == first


@pytest.mark.parametrize("window", [(-1, 3), (5, 2), (0, 20), (0, 999)])
def test_a_bad_window_is_refused_and_writes_nothing(scenario, window):
    before = tree(scenario.headers.path.parent)
    with pytest.raises(ValueError):
        compute_metrics(scenario.headers.path.parent, *window, BAND_05, CLOCK)
    assert tree(scenario.headers.path.parent) == before


def test_metrics_on_a_missing_store_refuse_and_create_nothing(tmp_path):
    with pytest.raises(audit.LogMissingError):
        compute_metrics(tmp_path / "nowhere", 0, 1)
    assert not (tmp_path / "nowhere").exists()


# ---------- duration_ms ----------


def test_every_decision_header_has_a_whole_non_negative_duration_ms(tmp_path):
    engine = make_engine(tmp_path)
    for name in ("Ivan Petrov", "Mohammed Ali Hassan", "محمد علي حسن", "Zebulon Quixotic"):
        engine.audited_screen(name)
    decisions = [h for h in engine.log.headers.read_all() if h["kind"] == "decision"]
    assert len(decisions) == 4
    assert all(isinstance(h["duration_ms"], int) and not isinstance(h["duration_ms"], bool) for h in decisions)
    assert all(h["duration_ms"] >= 0 for h in decisions)
    assert "duration_ms" in audit_schema.HEADER_FIELDS and "duration_ms" not in audit_schema.ENGINE_START_FIELDS
    assert verify_chain(engine.log).ok


@pytest.mark.parametrize("bad", [-1, 1.5, "3", None, True])
def test_a_decision_with_a_bad_duration_is_a_schema_violation(tmp_path, bad):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i in range(3):
        add_decision(log, i, LATIN, 0.1, 5)
    edit_line_rehashed(log, 1, duration_ms=bad)
    assert any(f.kind == "schema_violation" and "duration_ms" in f.detail for f in verify_chain(log).failures)


def test_a_decision_with_no_duration_is_a_schema_violation(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    add_decision(log, 0, LATIN, 0.1, 5)
    record = json.loads(log.headers.path.read_bytes().splitlines()[0])
    del record["duration_ms"]
    record["record_hash"] = audit.compute_record_hash(record)
    log.headers.path.write_bytes(audit.canonical_json(record).encode() + b"\n")
    assert "schema_violation" in verify_chain(log).kinds()


def edit_line_rehashed(log: AuditLog, index: int, **changes) -> None:
    lines = log.headers.path.read_bytes().split(b"\n")[:-1]
    record = json.loads(lines[index])
    record.update(changes)
    record["record_hash"] = audit.compute_record_hash(record)
    lines[index] = audit.canonical_json(record).encode()
    log.headers.path.write_bytes(b"".join(line + b"\n" for line in lines))


def test_the_schema_version_was_bumped_for_duration_ms():
    assert audit_schema.SCHEMA_VERSION == 4


def test_replay_does_not_compare_the_duration(tmp_path):
    from replay import replay

    engine = make_engine(tmp_path)
    result = engine.audited_screen("Mohammed Ali Hassan")
    assert replay(engine, result.request_id).identical  # timing differs from run to run and is not a decision field
    assert "duration_ms" not in audit_schema.DECISION_FIELDS


# ---------- the CLI ----------


def test_cli_metrics_writes_the_report_and_prints_counts_only(scenario, capsys):
    audit_dir = str(scenario.headers.path.parent)
    assert screen_cli.main(["metrics", "--from", "0", "--to", "19", "--audit-dir", audit_dir]) == 0
    out = capsys.readouterr().out
    assert "window 0-19: 20 decision(s) of 20 record(s), 1 run context group(s)" in out
    assert (scenario.headers.path.parent / "monitoring" / "0-19.json").exists()
    assert not any(h["request_id"] in out for h in scenario.headers.read_all())
    assert (
        screen_cli.main(["metrics", "--from", "0", "--to", "19", "--audit-dir", audit_dir])
        == screen_cli.EXIT_AUDIT_FAILURE
    )
    assert "already exists" in capsys.readouterr().err


def test_cli_metrics_refuses_a_bad_window_and_a_missing_directory(scenario, capsys, monkeypatch):
    audit_dir = str(scenario.headers.path.parent)
    assert screen_cli.main(["metrics", "--from", "5", "--to", "99", "--audit-dir", audit_dir]) == screen_cli.EXIT_USAGE
    assert "last seq" in capsys.readouterr().err
    monkeypatch.delenv("SCREENING_AUDIT_DIR", raising=False)
    assert screen_cli.main(["metrics", "--from", "0", "--to", "1"]) == screen_cli.EXIT_USAGE
    with pytest.raises(SystemExit):
        screen_cli.main(["metrics", "--audit-dir", audit_dir])  # --from and --to are required


# ---------- group-level and secondary suppression ----------


def hand_log(tmp_path, groups: dict[str, list[tuple[str, float]]], refs: dict[str, str] | None = None) -> AuditLog:
    """A log with the given (name, score) decisions per context, written context by context."""
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    i = 0
    for context, rows in groups.items():
        for name, score in rows:
            add_decision(log, i, name, score, 5, context=context, ref=(refs or {}).get(name))
            i += 1
    return log


def test_a_group_below_min_cell_is_fully_suppressed(tmp_path):
    log = hand_log(tmp_path, {CTX_X: [(LATIN, 0.9)] * 12, CTX_Y: [(LATIN, 0.1)] * 4})
    report, _ = measured(log.headers.path.parent, 0, 15, MetricsConfig(0.05, 10), CLOCK)
    big, small = report["groups"]
    assert big["count"] == 12 and "alert_rate" in big and "score_histogram" in big and "duration_ms" in big
    assert small == {"run_context_hash": CTX_Y, "count": "<10"}  # no rates, histogram, durations, or script cells
    assert report["window"]["decisions"] == "withheld"  # else 16 - 12 would give the hidden 4


def test_with_no_suppressed_group_the_window_reports_its_decision_total(scenario):
    report, _ = measured(scenario.headers.path.parent, 0, 19, BAND_05, CLOCK)
    assert report["window"]["decisions"] == 20


def test_a_single_suppressed_cell_triggers_a_second_the_smallest_visible_one(tmp_path):
    rows = [(LATIN, 0.1)] * 12 + [(ARABIC, 0.9)] * 3 + [(CYRILLIC, 0.1)] * 7 + [("田中太郎", 0.1)] * 9
    log = hand_log(tmp_path, {CTX_X: rows})
    report, _ = measured(log.headers.path.parent, 0, 30, MetricsConfig(0.05, 5), CLOCK)
    per_script = only_group(report)["per_script"]
    assert per_script["Arabic"] == {"count": "<5"}  # 3 is below 5
    assert per_script["Cyrillic"] == {"count": "withheld"}  # the smallest of the rest (7 < 9 < 12)
    assert per_script["Latin"]["count"] == 12 and per_script["other"]["count"] == 9
    hidden = [k for k, v in per_script.items() if not isinstance(v["count"], int)]
    assert sorted(hidden) == ["Arabic", "Cyrillic"]  # two hidden: neither can be recovered by subtraction


def test_unavailable_is_not_the_cell_withheld_while_another_is_visible(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i in range(12):
        add_decision(log, i, LATIN, 0.1, 5)
    for i in range(12, 15):
        add_decision(log, i, ARABIC, 0.9, 5)
    for i in range(15, 20):
        add_decision(log, i, LATIN, 0.1, 5, ref="R", when="2017-01-01T10:00:00")
    purge(log, {"R": "2018-01-01"}, now=NOW)  # five records become "unavailable"
    report, _ = measured(log.headers.path.parent, 0, 20, MetricsConfig(0.05, 4), CLOCK)
    per_script = only_group(report)["per_script"]
    assert per_script["Arabic"] == {"count": "<4"}
    assert per_script["Latin"] == {"count": "withheld"}  # the smallest visible cell that is not "unavailable"
    assert per_script["unavailable"]["count"] == 5  # stays visible


def test_when_only_unavailable_is_visible_it_is_withheld_so_the_hidden_cell_is_not_recoverable(tmp_path):
    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i in range(3):
        add_decision(log, i, ARABIC, 0.9, 5)
    for i in range(3, 15):
        add_decision(log, i, LATIN, 0.1, 5, ref="R", when="2017-01-01T10:00:00")
    purge(log, {"R": "2018-01-01"}, now=NOW)
    report, _ = measured(log.headers.path.parent, 0, 15, MetricsConfig(0.05, 5), CLOCK)
    per_script = only_group(report)["per_script"]
    assert per_script == {"Arabic": {"count": "<5"}, "unavailable": {"count": "withheld"}}


def test_no_report_in_this_file_lets_a_hidden_count_be_recovered(scenario, tmp_path):
    """The invariant is also asserted by `measured` on every report above; here it runs over many min_cell values."""
    for min_cell in range(1, 25):
        sub = tmp_path / f"m{min_cell}"
        log = AuditLog.create_new(sub / "audit", fsync=False)
        for i, (name, score, duration) in enumerate(SCENARIO):
            add_decision(log, i, name, score, duration)
        report, _ = measured(log.headers.path.parent, 0, 19, MetricsConfig(0.05, min_cell), CLOCK)
        assert_not_recoverable(report)


# ---------- integrity first ----------


def test_a_tampered_log_is_refused_and_no_report_is_written(scenario):
    edit_line(scenario, 4, final_score=0.0)
    with pytest.raises(MetricsRefused, match="edited_header") as stop:
        compute_metrics(scenario.headers.path.parent, 0, 19, BAND_05, CLOCK)
    assert "edited_header" in stop.value.kinds
    assert not (scenario.headers.path.parent / "monitoring").exists()


def test_tampering_outside_the_window_still_refuses(scenario):
    edit_line(scenario, 19, final_score=0.0)
    with pytest.raises(MetricsRefused):
        compute_metrics(scenario.headers.path.parent, 0, 5, BAND_05, CLOCK)


def test_a_log_with_an_orphan_payload_produces_a_report_listing_it(scenario):
    orphan = audit.PayloadRecord("f" * 32, b"s" * 32, {"raw_input": {"name": "left behind"}})
    scenario.payloads.write(orphan)
    report, path = measured(scenario.headers.path.parent, 0, 19, BAND_05, CLOCK)
    assert report["integrity"] == {"tampering_findings": 0, "crash_findings": {"orphan_payload": 1}}
    assert "f" * 32 not in path.read_text() and "left behind" not in path.read_text()  # listed by type and count


def test_a_clean_log_lists_no_findings(scenario):
    report, _ = measured(scenario.headers.path.parent, 0, 19, BAND_05, CLOCK)
    assert report["integrity"] == {"tampering_findings": 0, "crash_findings": {}}


def test_a_torn_tail_and_an_interrupted_purge_are_listed_by_type(tmp_path, monkeypatch):
    import audit_purge

    log = AuditLog.create_new(tmp_path / "audit", fsync=False)
    for i in range(4):
        add_decision(log, i, LATIN, 0.1, 5, ref="R", when="2017-01-01T10:00:00")
    monkeypatch.setattr(audit_purge, "delete_payload", lambda *a, **k: (_ for _ in ()).throw(AuditError("crash")))
    with pytest.raises(AuditError):
        purge(log, {"R": "2018-01-01"}, now=NOW)
    with open(log.headers.path, "ab") as fh:
        fh.write(b'{"torn')
    report, _ = measured(log.headers.path.parent, 0, 3, MetricsConfig(0.05, 1), CLOCK)
    assert report["integrity"]["crash_findings"] == {"purge_incomplete": 4, "torn_final_line": 1}


# ---------- the config file is recorded ----------


def test_the_report_records_the_sha256_of_the_config_file(scenario, tmp_path):
    import hashlib

    config_path = tmp_path / "monitoring.json"
    config_path.write_text('{"near_threshold_band": 0.05, "min_cell": 3}')
    loaded = MetricsConfig.load(config_path)
    report, _ = measured(scenario.headers.path.parent, 0, 19, loaded, CLOCK)
    assert report["config"]["file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    other = tmp_path / "other.json"
    other.write_text('{"near_threshold_band": 0.05,  "min_cell": 3}')  # same values, different bytes
    report2, _ = measured(scenario.headers.path.parent, 0, 18, MetricsConfig.load(other), CLOCK)
    assert report2["config"]["file_sha256"] != report["config"]["file_sha256"]
    assert measured(scenario.headers.path.parent, 0, 17, BAND_05, CLOCK)[0]["config"]["file_sha256"] is None


def test_the_default_config_hash_is_that_of_the_shipped_file(scenario):
    import hashlib

    report, _ = measured(scenario.headers.path.parent, 0, 19, None, CLOCK)
    assert report["config"]["file_sha256"] == hashlib.sha256(audit_metrics.CONFIG_PATH.read_bytes()).hexdigest()
