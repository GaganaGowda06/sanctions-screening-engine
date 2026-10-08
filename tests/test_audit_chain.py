"""Tests for the hash chain in src/audit.py (HeaderLog) and verify_chain() in src/audit_verify.py.

Run:
    pytest tests/test_audit_chain.py
"""

import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

import audit
from audit import AuditError, AuditLog, CheckpointError, PayloadRecord
from audit_verify import verify_chain

SRC = Path(audit.__file__).resolve().parent
MOMENT = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def add(log: AuditLog, i: int) -> str:
    request_id = f"{i:032x}"
    payload = PayloadRecord(request_id, bytes([i % 256]) * 32, {"raw_input": {"name": f"name {i}"}})
    entries = [audit.candidate_entry(f"e{i}", 0.5, 0.5, 0.0, 0.0, 0.0, False, False, 0.86)]
    header = audit.build_header(
        request_id=request_id, moment=MOMENT, run_context_hash="c" * 64, entries=entries, threshold=0.86,
        digest=payload.digest(),
    )  # fmt: skip
    log.write(header, payload)
    return request_id


@pytest.fixture
def log(tmp_path) -> AuditLog:
    return AuditLog.create_new(tmp_path / "audit", fsync=False)


def filled(log: AuditLog, n: int = 5) -> AuditLog:
    for i in range(n):
        add(log, i)
    return log


def read_lines(log: AuditLog) -> list[bytes]:
    return log.headers.path.read_bytes().split(b"\n")[:-1]


def write_lines(log: AuditLog, lines: list[bytes]) -> None:
    log.headers.path.write_bytes(b"".join(line + b"\n" for line in lines))


def edit_line(log: AuditLog, index: int, **changes) -> dict:
    lines = read_lines(log)
    record = json.loads(lines[index])
    record.update(changes)
    lines[index] = audit.canonical_json(record).encode()
    write_lines(log, lines)
    return record


# ---------- the chain itself ----------


def test_a_clean_chain_passes(log):
    filled(log, 5)
    result = verify_chain(log)
    assert result.ok and result.failures == ()
    assert (result.records, result.payloads_checked) == (5, 5)


def test_an_initialised_empty_log_is_clean(log):
    result = verify_chain(log)
    assert result.ok and result.records == 0
    assert log.headers.read_checkpoint() == {"count": 0, "record_hash": audit.GENESIS_PREV_HASH}
    assert log.headers.path.read_bytes() == b"" and log.payloads.directory.is_dir()


def test_create_new_refuses_a_directory_that_exists_and_is_not_empty(log):
    before = snapshot(log.headers.path.parent)
    with pytest.raises(AuditError, match="not empty"):
        AuditLog.create_new(log.headers.path.parent)
    filled(log, 2)
    with pytest.raises(AuditError, match="not empty"):
        AuditLog.create_new(log.headers.path.parent)
    assert snapshot(log.headers.path.parent) >= before and verify_chain(log).ok and verify_chain(log).records == 2


def test_create_new_accepts_an_existing_empty_directory_and_refuses_a_file(tmp_path):
    (tmp_path / "empty").mkdir()
    assert verify_chain(AuditLog.create_new(tmp_path / "empty", fsync=False)).ok
    (tmp_path / "afile").write_text("x")
    with pytest.raises(AuditError):
        AuditLog.create_new(tmp_path / "afile")


def test_appending_to_a_store_that_was_never_initialised_fails_with_log_missing_and_creates_nothing(tmp_path):
    log = AuditLog.in_directory(tmp_path / "nowhere" / "audit", fsync=False)
    with pytest.raises(audit.LogMissingError, match="log_missing"):
        add(log, 0)
    assert not (tmp_path / "nowhere").exists()
    (tmp_path / "bare").mkdir()
    with pytest.raises(audit.LogMissingError, match="init"):
        add(AuditLog.in_directory(tmp_path / "bare", fsync=False), 0)
    assert list((tmp_path / "bare").iterdir()) == []


def test_a_wiped_store_is_refused_not_started_fresh(log):
    filled(log, 3)
    import shutil

    shutil.rmtree(log.headers.path.parent)
    with pytest.raises(audit.LogMissingError, match="log_missing"):
        add(log, 9)
    assert not log.headers.path.parent.exists()  # the refusal did not recreate it


def test_an_emptied_store_directory_is_refused_too(log):
    filled(log, 3)
    for path in log.headers.path.parent.iterdir():
        if path.is_file():
            path.unlink()
    with pytest.raises(audit.LogMissingError):
        add(log, 9)
    assert not log.headers.path.exists()


def test_a_missing_checkpoint_on_an_empty_log_is_refused_on_append(log):
    log.headers.checkpoint_path.unlink()
    with pytest.raises(audit.LogMissingError, match="checkpoint"):
        add(log, 0)


def snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")} if root.exists() else set()


def test_verifying_a_nonexistent_path_returns_log_missing_and_creates_nothing(tmp_path):
    log = AuditLog.in_directory(tmp_path / "nowhere" / "audit", fsync=False)
    result = verify_chain(log)
    assert result.kinds() == {"log_missing"} and not result.ok and not result.failures[0].is_crash
    assert snapshot(tmp_path) == set()  # not even the parent directory


def test_verifying_an_existing_empty_directory_returns_log_missing_and_creates_at_most_the_lock_file(tmp_path):
    (tmp_path / "audit").mkdir()
    result = verify_chain(AuditLog.in_directory(tmp_path / "audit", fsync=False))
    assert result.kinds() == {"log_missing"}
    assert snapshot(tmp_path) <= {"audit", "audit/headers.jsonl.lock"}


def test_deleting_the_headers_and_the_checkpoint_returns_log_missing(log):
    filled(log, 3)
    log.headers.path.unlink()
    log.headers.checkpoint_path.unlink()
    result = verify_chain(log)
    assert "log_missing" in result.kinds() and not result.ok
    assert "checkpoint_missing" not in result.kinds()  # one clear finding, not a pile


def test_deleting_only_the_headers_is_log_missing_and_truncation(log):
    filled(log, 3)
    log.headers.path.unlink()
    assert verify_chain(log).kinds() >= {"log_missing", "tail_truncated"}


def test_an_empty_headers_file_with_no_checkpoint_is_checkpoint_missing(log):
    log.headers.checkpoint_path.unlink()
    assert verify_chain(log).kinds() == {"checkpoint_missing"}


def test_verifying_does_not_change_an_existing_store(log):
    filled(log, 3)
    before = snapshot(log.headers.path.parent)
    verify_chain(log)
    assert snapshot(log.headers.path.parent) == before


def test_the_first_record_uses_the_genesis_hash_and_each_record_links_to_the_one_before(log):
    filled(log, 4)
    records = log.headers.read_all()
    assert records[0]["prev_hash"] == audit.GENESIS_PREV_HASH == "0" * 64
    for i, record in enumerate(records):
        assert record["seq"] == i
        if i:
            assert record["prev_hash"] == records[i - 1]["record_hash"]


def test_record_hash_is_sha256_of_the_canonical_json_with_prev_hash_and_without_record_hash(log):
    import hashlib

    filled(log, 2)
    record = log.headers.read_all()[1]
    body = {k: v for k, v in record.items() if k != "record_hash"}
    assert "prev_hash" in body and "record_hash" not in body
    assert record["record_hash"] == hashlib.sha256(audit.canonical_json(body).encode("utf-8")).hexdigest()
    assert record["record_hash"] == audit.compute_record_hash(record)


def test_a_header_to_append_may_not_carry_chain_fields(log):
    with pytest.raises(AuditError):
        log.headers.append({"seq": 3, "request_id": "a" * 32})


# ---------- one test per failure type ----------


def test_an_edited_header_is_reported(log):
    filled(log, 5)
    edit_line(log, 2, decision="alert")
    result = verify_chain(log)
    assert "edited_header" in result.kinds()
    assert any(f.kind == "edited_header" and f.seq == 2 for f in result.tampering)
    assert not result.crashes


def test_a_rehashed_edit_is_caught_by_the_next_link(log):
    filled(log, 5)
    record = json.loads(read_lines(log)[2])
    record["decision"] = "alert"
    record["record_hash"] = audit.compute_record_hash(record)
    lines = read_lines(log)
    lines[2] = audit.canonical_json(record).encode()
    write_lines(log, lines)
    result = verify_chain(log)
    assert "edited_header" not in result.kinds()
    assert any(f.kind == "broken_link" and f.seq == 3 for f in result.failures)


def test_a_removed_line_is_a_broken_link(log):
    filled(log, 5)
    lines = read_lines(log)
    write_lines(log, lines[:2] + lines[3:])
    result = verify_chain(log)
    assert "broken_link" in result.kinds()


def test_swapped_lines_are_a_broken_link(log):
    filled(log, 5)
    lines = read_lines(log)
    lines[1], lines[2] = lines[2], lines[1]
    write_lines(log, lines)
    assert "broken_link" in verify_chain(log).kinds()


def test_a_line_that_is_not_json_is_an_edited_header(log):
    filled(log, 3)
    lines = read_lines(log)
    lines[1] = b"{not json"
    write_lines(log, lines)
    assert "edited_header" in verify_chain(log).kinds()


def test_an_edited_payload_fails_its_digest(log):
    filled(log, 4)
    rid = f"{2:032x}"
    path = log.payloads.path_for(rid)
    stored = json.loads(path.read_text())
    stored["data"]["raw_input"]["name"] = "someone else"
    path.write_text(json.dumps(stored))
    result = verify_chain(log)
    assert [f.kind for f in result.failures] == ["payload_digest_mismatch"]
    assert result.failures[0].request_id == rid and not result.failures[0].is_crash


def test_a_replaced_salt_fails_the_digest(log):
    filled(log, 3)
    path = log.payloads.path_for(f"{1:032x}")
    stored = json.loads(path.read_text())
    stored["salt_hex"] = "00" * 32
    path.write_text(json.dumps(stored))
    assert verify_chain(log).kinds() == {"payload_digest_mismatch"}


def test_an_unreadable_payload_is_a_digest_failure(log):
    filled(log, 3)
    log.payloads.path_for(f"{1:032x}").write_text("{broken")
    assert verify_chain(log).kinds() == {"payload_digest_mismatch"}


def test_a_missing_payload_is_a_failure_and_not_a_crash(log):
    filled(log, 3)
    rid = f"{1:032x}"
    log.payloads.path_for(rid).unlink()
    result = verify_chain(log)
    assert result.kinds() == {"payload_missing"}
    assert result.failures[0].request_id == rid and not result.failures[0].is_crash
    assert result.payloads_checked == 2


def test_an_orphan_payload_is_reported_as_a_crash_not_tampering(log):
    filled(log, 3)
    orphan = PayloadRecord("f" * 32, b"s" * 32, {"raw_input": {"name": "left behind"}})
    log.payloads.write(orphan)  # a crash after the payload write, before the header write
    result = verify_chain(log)
    assert result.kinds() == {"orphan_payload"}
    assert result.failures[0].request_id == "f" * 32 and result.failures[0].is_crash
    assert result.tampering == () and len(result.crashes) == 1
    assert "left behind" not in repr(result)  # the report never carries payload content


def test_a_torn_final_line_is_reported_as_a_crash_and_the_rest_still_verifies(log):
    filled(log, 3)
    complete = read_lines(log)[-1]
    with open(log.headers.path, "ab") as fh:
        fh.write(complete[: len(complete) // 2])  # a record cut off mid-write, no newline
    result = verify_chain(log)
    assert result.kinds() == {"torn_final_line"}
    assert result.tampering == () and result.records == 3
    assert len(log.headers.read_all()) == 3


def test_the_log_refuses_to_append_after_a_torn_line(log):
    filled(log, 2)
    with open(log.headers.path, "ab") as fh:
        fh.write(b'{"partial')
    with pytest.raises(AuditError, match="torn"):
        add(log, 99)
    assert log.payloads.read(f"{99:032x}") is None  # the payload for the refused call was removed


def test_a_crash_between_the_line_and_the_checkpoint_is_reported_as_a_crash_and_repaired(log, monkeypatch):
    filled(log, 3)
    original = log.headers._write_checkpoint
    monkeypatch.setattr(log.headers, "_write_checkpoint", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(CheckpointError):
        add(log, 3)
    monkeypatch.setattr(log.headers, "_write_checkpoint", original)
    assert log.payloads.read(f"{3:032x}") is not None  # the header is in the log, so the payload stays
    result = verify_chain(log)
    assert result.kinds() == {"checkpoint_behind"} and result.tampering == ()
    add(log, 4)  # the next append repairs the checkpoint
    assert verify_chain(log).ok and verify_chain(log).records == 5


# ---------- checkpoint repair on append ----------


def checkpoint_text(log: AuditLog) -> str:
    return log.headers.checkpoint_path.read_text()


def test_behind_by_one_with_a_valid_link_is_repaired_and_the_append_continues(log):
    filled(log, 3)
    saved = checkpoint_text(log)  # count 3
    add(log, 3)
    log.headers.checkpoint_path.write_text(saved)  # the crash: record 3 is in the log, the checkpoint says 3
    assert verify_chain(log).kinds() == {"checkpoint_behind"}
    add(log, 4)
    result = verify_chain(log)
    assert result.ok and result.records == 5
    assert log.headers.read_checkpoint()["count"] == 5


def test_behind_by_one_with_a_bad_link_is_refused(log):
    filled(log, 3)
    add(log, 3)
    log.headers.checkpoint_path.write_text(audit.canonical_json({"count": 3, "record_hash": "e" * 64}))
    before = log.headers.path.read_bytes()
    with pytest.raises(AuditError, match="does not match its checkpoint"):
        add(log, 4)
    assert log.headers.path.read_bytes() == before  # nothing was appended
    assert log.headers.read_checkpoint()["record_hash"] == "e" * 64  # and the evidence was not overwritten
    assert verify_chain(log).kinds() == {"checkpoint_mismatch"}


def test_behind_by_two_is_refused(log):
    filled(log, 3)
    saved = checkpoint_text(log)  # count 3
    add(log, 3)
    add(log, 4)
    log.headers.checkpoint_path.write_text(saved)
    before = log.headers.path.read_bytes()
    with pytest.raises(AuditError, match="does not match its checkpoint"):
        add(log, 5)
    assert log.headers.path.read_bytes() == before and checkpoint_text(log) == saved
    assert verify_chain(log).kinds() == {"checkpoint_mismatch"}


def test_behind_by_one_with_an_edited_last_record_is_refused(log):
    filled(log, 3)
    saved = checkpoint_text(log)
    add(log, 3)
    log.headers.checkpoint_path.write_text(saved)
    edit_line(log, 3, decision="alert")
    with pytest.raises(AuditError, match="own hash"):
        add(log, 4)
    assert checkpoint_text(log) == saved


# ---------- tail truncation and the checkpoint ----------


def test_the_checkpoint_records_the_count_and_last_hash(log):
    filled(log, 4)
    checkpoint = log.headers.read_checkpoint()
    assert checkpoint == {"count": 4, "record_hash": log.headers.read_all()[-1]["record_hash"]}


@pytest.mark.parametrize("remove", [1, 2, 5])
def test_tail_truncation_is_detected(log, remove):
    filled(log, 5)
    write_lines(log, read_lines(log)[:-remove])
    result = verify_chain(log)
    assert "tail_truncated" in result.kinds()
    assert not any(f.is_crash for f in result.failures if f.kind == "tail_truncated")


def test_the_log_refuses_to_append_to_a_truncated_log(log):
    filled(log, 5)
    write_lines(log, read_lines(log)[:3])
    with pytest.raises(AuditError, match="tail truncated"):
        add(log, 50)
    assert verify_chain(log).kinds() >= {"tail_truncated"}  # still evident: the checkpoint was not overwritten
    assert log.headers.read_checkpoint()["count"] == 5


def test_a_deleted_checkpoint_is_reported_and_blocks_appends(log):
    filled(log, 3)
    log.headers.checkpoint_path.unlink()
    assert verify_chain(log).kinds() == {"checkpoint_missing"}
    with pytest.raises(AuditError):
        add(log, 9)


def test_a_wrong_checkpoint_hash_is_a_mismatch(log):
    filled(log, 3)
    log.headers.checkpoint_path.write_text(audit.canonical_json({"count": 3, "record_hash": "e" * 64}))
    assert verify_chain(log).kinds() == {"checkpoint_mismatch"}


def test_an_unreadable_checkpoint_is_a_mismatch(log):
    filled(log, 3)
    log.headers.checkpoint_path.write_text("{broken")
    assert verify_chain(log).kinds() == {"checkpoint_mismatch"}


def test_the_log_refuses_to_append_after_its_last_record_was_edited(log):
    filled(log, 3)
    edit_line(log, 2, decision="alert")
    with pytest.raises(AuditError, match="own hash"):
        add(log, 8)


def test_a_wiped_log_with_a_checkpoint_is_truncation(log):
    filled(log, 3)
    log.headers.path.write_bytes(b"")
    assert "tail_truncated" in verify_chain(log).kinds()
    with pytest.raises(AuditError):
        add(log, 7)


# ---------- concurrent writers ----------


def test_parallel_processes_write_one_valid_chain(tmp_path):
    directory = tmp_path / "audit"
    worker = (
        "import sys\n"
        "from datetime import datetime, timezone\n"
        "import audit\n"
        "from audit import AuditLog, PayloadRecord\n"
        "w, n = int(sys.argv[1]), int(sys.argv[2])\n"
        "log = AuditLog.in_directory(sys.argv[3], fsync=False)\n"
        "for i in range(n):\n"
        "    rid = f'{w * 1000 + i:032x}'\n"
        "    p = PayloadRecord(rid, bytes([w]) * 32, {'raw_input': {'name': f'w{w} {i}'}})\n"
        "    e = [audit.candidate_entry('e', 0.5, 0.5, 0.0, 0.0, 0.0, False, False, 0.86)]\n"
        "    h = audit.build_header(request_id=rid, moment=datetime(2026, 1, 1, tzinfo=timezone.utc),\n"
        "        run_context_hash='c' * 64, entries=e, threshold=0.86, digest=p.digest())\n"
        "    log.write(h, p)\n"
    )
    workers, each = 6, 25
    AuditLog.create_new(directory, fsync=False)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(w), str(each), str(directory)],
            cwd=SRC,
            env={"PYTHONPATH": str(SRC), "PATH": ""},
            stderr=subprocess.PIPE,
        )  # fmt: skip
        for w in range(workers)
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode()
    log = AuditLog.in_directory(directory, fsync=False)
    result = verify_chain(log)
    assert result.ok, result.failures
    assert result.records == workers * each
    records = log.headers.read_all()
    assert [r["seq"] for r in records] == list(range(workers * each))
    assert len({r["request_id"] for r in records}) == workers * each
    assert log.headers.read_checkpoint()["count"] == workers * each


def test_parallel_threads_write_one_valid_chain(log):
    errors = []

    def run(worker: int) -> None:
        try:
            for i in range(15):
                add(log, worker * 1000 + i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    result = verify_chain(log)
    assert result.ok, result.failures
    assert result.records == 120


# ---------- verification alongside writers ----------


def test_verify_running_alongside_threaded_writers_reports_nothing(log):
    stop = threading.Event()
    results, errors = [], []

    def write(worker: int) -> None:
        try:
            for i in range(25):
                add(log, worker * 1000 + i)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def verify() -> None:
        while not stop.is_set():
            results.append(verify_chain(log))

    writers = [threading.Thread(target=write, args=(w,)) for w in range(4)]
    verifier = threading.Thread(target=verify)
    verifier.start()
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    verifier.join()
    assert errors == []
    assert len(results) >= 3, "the verifier should have run several times while writers were active"
    assert all(r.ok for r in results), [f for r in results for f in r.failures][:5]
    assert not any("orphan_payload" in r.kinds() for r in results)
    assert verify_chain(log).records == 100


def test_verify_running_alongside_writer_processes_reports_no_orphans(tmp_path):
    directory = tmp_path / "audit"
    worker = (
        "import sys\n"
        "from datetime import datetime, timezone\n"
        "import audit\n"
        "from audit import AuditLog, PayloadRecord\n"
        "w, n = int(sys.argv[1]), int(sys.argv[2])\n"
        "log = AuditLog.in_directory(sys.argv[3], fsync=False)\n"
        "for i in range(n):\n"
        "    rid = f'{w * 1000 + i:032x}'\n"
        "    p = PayloadRecord(rid, bytes([w]) * 32, {'raw_input': {'name': f'w{w} {i}'}})\n"
        "    e = [audit.candidate_entry('e', 0.5, 0.5, 0.0, 0.0, 0.0, False, False, 0.86)]\n"
        "    h = audit.build_header(request_id=rid, moment=datetime(2026, 1, 1, tzinfo=timezone.utc),\n"
        "        run_context_hash='c' * 64, entries=e, threshold=0.86, digest=p.digest())\n"
        "    log.write(h, p)\n"
    )
    log = AuditLog.create_new(
        directory, fsync=False
    )  # a verifier started before the store exists would report log_missing
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(w), "40", str(directory)],
            cwd=SRC,
            env={"PYTHONPATH": str(SRC), "PATH": ""},
            stderr=subprocess.PIPE,
        )  # fmt: skip
        for w in range(4)
    ]
    results = []
    while any(p.poll() is None for p in procs):
        results.append(verify_chain(log))
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode()
    assert results, "the verifier should have run while the writers were active"
    assert all(r.ok for r in results), [f for r in results for f in r.failures][:5]
    assert verify_chain(log).records == 160
