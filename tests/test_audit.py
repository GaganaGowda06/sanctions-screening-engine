"""Tests for src/audit.py: canonical JSON, the payload digest, candidate truncation, and the two stores.

Run:
    pytest tests/test_audit.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import audit
import audit_schema
from audit import AuditError, AuditLog, PayloadRecord, PayloadStore

SRC = Path(audit.__file__).resolve().parent
MOMENT = datetime(2026, 10, 7, 12, 30, 15, 123456, tzinfo=timezone.utc)


def entry(entity_id: str, score: float, threshold: float = 0.86) -> dict:
    return audit_schema.candidate_entry(entity_id, score, score, 0.0, 0.0, 0.0, False, False, threshold)


# ---------- canonical JSON ----------


def test_canonical_json_ignores_key_order_and_keeps_non_ascii():
    a = {"b": 1, "a": {"y": [1, 2], "x": "محمد"}}
    b = {"a": {"x": "محمد", "y": [1, 2]}, "b": 1}
    assert audit.canonical_json(a) == audit.canonical_json(b)
    assert audit.canonical_json(a) == '{"a":{"x":"محمد","y":[1,2]},"b":1}'


def test_canonical_json_refuses_nan_and_infinity():
    with pytest.raises(ValueError):
        audit.canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        audit.canonical_json({"x": float("inf")})


def test_canonical_json_and_digest_are_stable_across_runs_and_hash_seeds():
    code = (
        "import audit\n"
        "import audit_schema\n"
        "data = {'name': 'محمد علي', 'tokens': ['a', 'b'], 'keys': sorted({'t:b', 's:a', 't:a'}), 'n': 0.1 + 0.2}\n"
        "print(audit.canonical_json(data)); print(audit.payload_digest(bytes(range(32)), data))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=SRC,
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONPATH": str(SRC), "PYTHONHASHSEED": seed, "PATH": ""},
        ).stdout  # fmt: skip
        for seed in ("1", "2", "3")
    }
    assert len(outputs) == 1


# ---------- salt and digest ----------


def test_digest_is_sha256_of_salt_then_canonical_json_in_utf8():
    salt, data = bytes(range(32)), {"name": "محمد", "a": 1}
    expected = hashlib.sha256(salt + '{"a":1,"name":"محمد"}'.encode()).hexdigest()
    assert audit.payload_digest(salt, data) == expected


def test_digest_changes_with_the_salt_and_with_the_data():
    data = {"name": "Mohammed Ali"}
    salt = audit.new_salt()
    assert audit.payload_digest(salt, data) != audit.payload_digest(audit.new_salt(), data)
    assert audit.payload_digest(salt, data) != audit.payload_digest(salt, {"name": "Mohammed Alj"})
    assert audit.payload_digest(salt, data) == audit.payload_digest(salt, dict(data))


def test_salts_are_at_least_16_bytes_and_random():
    salts = {audit.new_salt() for _ in range(50)}
    assert len(salts) == 50
    assert all(len(s) >= 16 for s in salts)
    assert audit.SALT_BYTES >= 16


def test_request_ids_are_unique_and_valid():
    ids = {audit.new_request_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == 32 and set(i) <= set("0123456789abcdef") for i in ids)


# ---------- time ----------


def test_timestamps_must_be_utc():
    assert audit_schema.utc_timestamp(MOMENT) == "2026-10-07T12:30:15.123456+00:00"
    with pytest.raises(ValueError):
        audit_schema.utc_timestamp(datetime(2026, 10, 7, 12, 0))
    with pytest.raises(ValueError):
        audit_schema.utc_timestamp(datetime(2026, 10, 7, 12, 0, tzinfo=timezone(timedelta(hours=1))))


# ---------- candidate truncation ----------


def test_top_ten_are_kept_when_nothing_reaches_the_threshold():
    entries = [entry(f"e{i:02d}", 0.5 - i / 1000) for i in range(30)]
    kept = audit_schema.choose_candidates(entries)
    assert [e["entity_id"] for e in kept] == [f"e{i:02d}" for i in range(10)]


def test_every_candidate_at_or_above_the_threshold_is_kept_past_the_top_ten():
    entries = [entry(f"hit{i:02d}", 0.99 - i / 1000) for i in range(15)] + [
        entry(f"low{i:02d}", 0.3) for i in range(20)
    ]
    kept = audit_schema.choose_candidates(entries)
    assert [e["entity_id"] for e in kept][:15] == [f"hit{i:02d}" for i in range(15)]
    assert sum(e["at_or_above_threshold"] for e in kept) == 15
    assert len(kept) == 15  # the top ten are all hits, so no low candidate is added


def test_a_score_exactly_at_the_threshold_is_a_hit():
    entries = [entry(f"low{i}", 0.80) for i in range(12)] + [entry("edge", 0.86)]
    kept = audit_schema.choose_candidates(entries)
    assert kept[0]["entity_id"] == "edge" and kept[0]["at_or_above_threshold"]
    assert len(kept) == 10


def test_ties_are_ordered_by_entity_id():
    kept = audit_schema.choose_candidates([entry("b", 0.9), entry("a", 0.9), entry("c", 0.9)])
    assert [e["entity_id"] for e in kept] == ["a", "b", "c"]


def header_for(entries, threshold=0.86, **kwargs):
    return audit_schema.build_header(
        request_id="a" * 32, moment=MOMENT, run_context_hash="c" * 64, entries=entries, threshold=threshold,
        digest="d" * 64, duration_ms=7, **kwargs,
    )  # fmt: skip


def test_header_has_exactly_the_allowed_fields_and_candidate_fields():
    header = header_for([entry("e1", 0.95), entry("e2", 0.4)])
    assert tuple(header) == audit_schema.HEADER_FIELDS
    assert all(tuple(c) == audit_schema.CANDIDATE_FIELDS for c in header["candidates"])


def test_header_decision_and_matches():
    alert = header_for([entry("e2", 0.90), entry("e1", 0.95), entry("e3", 0.4)])
    assert alert["decision"] == "alert"
    assert alert["matched_entity_id"] == "e1" and alert["matched_entity_ids"] == ["e1", "e2"]
    assert alert["final_score"] == 0.95 and alert["candidates_scored"] == 3
    clear = header_for([entry("e3", 0.4)])
    assert clear["decision"] == "clear" and clear["matched_entity_id"] is None and clear["matched_entity_ids"] == []
    assert clear["final_score"] == 0.4
    none = header_for([])
    assert none["decision"] == "clear" and none["final_score"] is None and none["candidates_scored"] == 0


def test_header_counts_all_scored_candidates_not_only_the_logged_ones():
    header = header_for([entry(f"e{i}", 0.3) for i in range(40)])
    assert header["candidates_scored"] == 40 and len(header["candidates"]) == 10


def test_header_has_the_timestamp_and_no_retention_date():
    header = header_for([])
    assert header["timestamp_utc"] == "2026-10-07T12:30:15.123456+00:00"
    assert "retention_until" not in header and "retention_until" not in audit_schema.HEADER_FIELDS


# ---------- the header log ----------


def test_header_log_appends_one_line_per_header_and_never_rewrites(tmp_path):
    log = AuditLog.create_new(tmp_path / "store").headers
    tmp_path = tmp_path / "store"
    first, second = header_for([entry("e1", 0.9)]), header_for([entry("e2", 0.5)])
    log.append(first)
    before = (tmp_path / "headers.jsonl").read_bytes()
    log.append(second)
    after = (tmp_path / "headers.jsonl").read_bytes()
    assert after.startswith(before) and after.count(b"\n") == 2
    one, two = log.read_all()
    assert {k: v for k, v in one.items() if k not in audit_schema.CHAIN_FIELDS} == first
    assert {k: v for k, v in two.items() if k not in audit_schema.CHAIN_FIELDS} == second
    assert tuple(sorted(one)) == tuple(sorted(audit_schema.LOGGED_HEADER_FIELDS))


def test_header_log_failure_raises_audit_error(tmp_path):
    (tmp_path / "blocker").write_text("a file, not a directory")
    with pytest.raises(AuditError):
        audit.HeaderLog(tmp_path / "blocker" / "headers.jsonl").append({"a": 1})
    with pytest.raises(AuditError):
        AuditLog.create_new(tmp_path / "store").headers.append({"a": float("nan")})


# ---------- the payload store ----------


def test_payload_round_trip_keeps_salt_and_non_ascii_text(tmp_path):
    store = PayloadStore(tmp_path / "payloads")
    record = PayloadRecord("b" * 32, audit.new_salt(), {"raw_input": {"name": "محمد علي"}})
    store.write(record)
    assert store.read("b" * 32) == record
    assert store.read("c" * 32) is None
    assert record.digest() == audit.payload_digest(record.salt, record.data)


def test_payload_is_never_overwritten(tmp_path):
    store = PayloadStore(tmp_path)
    store.write(PayloadRecord("b" * 32, b"s" * 32, {"x": 1}))
    with pytest.raises(AuditError):
        store.write(PayloadRecord("b" * 32, b"t" * 32, {"x": 2}))
    assert store.read("b" * 32).data == {"x": 1}


@pytest.mark.parametrize("bad", ["../escape", "a" * 31, "A" * 32, "", "a/b", "a" * 32 + ".json"])
def test_request_ids_that_could_escape_the_directory_are_refused(tmp_path, bad):
    with pytest.raises(AuditError):
        PayloadStore(tmp_path).path_for(bad)


def test_payload_write_failure_raises_audit_error(tmp_path):
    (tmp_path / "blocker").write_text("x")
    with pytest.raises(AuditError):
        PayloadStore(tmp_path / "blocker" / "p").write(PayloadRecord("b" * 32, b"s" * 32, {"x": 1}))


def test_audit_log_removes_the_payload_if_the_header_cannot_be_written(tmp_path):
    log = AuditLog.create_new(tmp_path / "store")
    (tmp_path / "store" / "headers.jsonl").unlink()
    (tmp_path / "store" / "headers.jsonl").mkdir()  # a directory where the log file should be
    record = PayloadRecord("b" * 32, b"s" * 32, {"x": 1})
    with pytest.raises(AuditError):
        log.write({"request_id": "b" * 32}, record)
    assert log.payloads.read("b" * 32) is None


def test_audit_log_writes_both_when_it_can(tmp_path):
    log = AuditLog.create_new(tmp_path / "store")
    tmp_path = tmp_path / "store"
    record = PayloadRecord("b" * 32, b"s" * 32, {"x": 1})
    header = header_for([entry("e1", 0.9)])
    log.write(header, record)
    (logged,) = log.headers.read_all()
    assert {k: v for k, v in logged.items() if k not in audit_schema.CHAIN_FIELDS} == header
    assert log.payloads.read("b" * 32) == record
    assert json.loads((tmp_path / "payloads" / ("b" * 32 + ".json")).read_text())["salt_hex"] == (b"s" * 32).hex()
