"""verify_chain(): check the header log, its checkpoint and the payload store, and say what is wrong.

Failure kinds, split into tampering and crash signatures so a crash is never reported as an attack:

  tampering
    edited_header          a record's hash does not match its content (or the line is not valid JSON)
    broken_link            a record's seq or prev_hash does not follow the record before it (a line was removed,
                           inserted, moved, or an edited record was rehashed)
    payload_digest_mismatch  a stored payload does not match the digest in its header (or cannot be read)
    payload_missing        a header whose payload file is gone. No purge exists yet; once one does, only payloads
                           covered by a chained purge tombstone are exempt (step 7)
    tail_truncated         the checkpoint records more records than the log has: the last lines were removed
    checkpoint_mismatch    the checkpoint's hash does not match the record it points at
    checkpoint_missing     the log has records and no checkpoint file
  crash
    torn_final_line        the file ends without a newline: a write was cut short. The line is not counted
    checkpoint_behind      the log is exactly one record ahead of the checkpoint: a crash between the two writes
    orphan_payload         a payload file with no header: a crash between the payload write and the header write.
                           It holds personal data that nothing points to.

Writers write the payload and the header under the log's exclusive lock, and verify_chain holds the same lock for
its whole scan, so a verification that runs alongside writers sees only whole records and reports no false orphan.

The checkpoint is tamper-evident, not tamper-proof: whoever can rewrite the log can rewrite the checkpoint.
"""

import json
from dataclasses import dataclass, field

from audit import GENESIS_PREV_HASH, AuditError, AuditLog, compute_record_hash

TAMPERING = frozenset(
    {
        "edited_header",
        "broken_link",
        "payload_digest_mismatch",
        "payload_missing",
        "tail_truncated",
        "checkpoint_mismatch",
        "checkpoint_missing",
    }
)
CRASH = frozenset({"torn_final_line", "checkpoint_behind", "orphan_payload"})


@dataclass(frozen=True)
class Failure:
    kind: str
    detail: str
    seq: int | None = None
    request_id: str | None = None

    @property
    def is_crash(self) -> bool:
        return self.kind in CRASH


@dataclass(frozen=True)
class VerifyResult:
    records: int
    payloads_checked: int
    failures: tuple[Failure, ...] = field(default_factory=tuple)

    @property
    def tampering(self) -> tuple[Failure, ...]:
        return tuple(f for f in self.failures if not f.is_crash)

    @property
    def crashes(self) -> tuple[Failure, ...]:
        return tuple(f for f in self.failures if f.is_crash)

    @property
    def ok(self) -> bool:
        return not self.failures

    def kinds(self) -> set[str]:
        return {f.kind for f in self.failures}


def verify_chain(log: AuditLog) -> VerifyResult:
    """Walk the whole header log, then the checkpoint, then every payload. Never raises for a bad log; it reports."""
    with log.headers.locked():  # one consistent view of the log, its checkpoint and the payload store
        return _verify_locked(log)


def _verify_locked(log: AuditLog) -> VerifyResult:
    headers = log.headers
    failures: list[Failure] = []
    raw = headers.path.read_bytes() if headers.path.exists() else b""
    try:
        checkpoint = headers.read_checkpoint()
    except ValueError:
        checkpoint = {"count": None, "record_hash": None}

    lines = raw.split(b"\n")
    torn = lines.pop()  # empty after a final newline, or the torn fragment
    if torn:
        failures.append(Failure("torn_final_line", f"{len(torn)} bytes after the last newline, not counted"))

    records: list[dict] = []
    expected_prev: str | None = GENESIS_PREV_HASH
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("not an object")
        except ValueError as exc:
            failures.append(Failure("edited_header", f"line {i} is not valid JSON ({exc})", seq=i))
            expected_prev = None
            continue
        records.append(record)
        if compute_record_hash(record) != record.get("record_hash"):
            failures.append(
                Failure("edited_header", f"record {i} does not match its hash", i, record.get("request_id"))
            )
        if record.get("seq") != i:
            failures.append(
                Failure("broken_link", f"record {i} has seq {record.get('seq')!r}", i, record.get("request_id"))
            )
        elif expected_prev is not None and record.get("prev_hash") != expected_prev:
            failures.append(
                Failure("broken_link", f"record {i} does not follow the record before it", i, record.get("request_id"))
            )
        expected_prev = record.get("record_hash")

    n = len(lines)
    if checkpoint is None:
        if n:
            failures.append(Failure("checkpoint_missing", f"the log has {n} records and no checkpoint"))
    elif checkpoint.get("count") is None:
        failures.append(Failure("checkpoint_mismatch", "the checkpoint file cannot be read"))
    else:
        count, head = checkpoint["count"], checkpoint["record_hash"]
        if count > n:
            failures.append(Failure("tail_truncated", f"the checkpoint records {count} records, the log has {n}"))
        elif count == n:
            if n and records and head != records[-1].get("record_hash") and len(records) == n:
                failures.append(
                    Failure("checkpoint_mismatch", "the last record is not the one the checkpoint recorded")
                )
        else:
            at = records[count - 1].get("record_hash") if count and len(records) >= count else None
            if count and at != head:
                failures.append(
                    Failure("checkpoint_mismatch", f"record {count - 1} is not the one the checkpoint recorded")
                )
            elif n - count == 1:
                failures.append(Failure("checkpoint_behind", "the log is one record ahead of its checkpoint"))
            else:
                failures.append(
                    Failure("checkpoint_mismatch", f"the log is {n - count} records ahead of its checkpoint")
                )

    request_ids = {r.get("request_id") for r in records}
    checked = 0
    for r in records:
        rid = r.get("request_id")
        if not isinstance(rid, str):
            continue
        try:
            payload = log.payloads.read(rid)
        except (AuditError, ValueError, KeyError, OSError) as exc:
            failures.append(
                Failure(
                    "payload_digest_mismatch", f"the payload cannot be read ({type(exc).__name__})", r.get("seq"), rid
                )
            )
            continue
        if payload is None:
            failures.append(Failure("payload_missing", "the header's payload file is gone", r.get("seq"), rid))
            continue
        checked += 1
        if payload.request_id != rid or payload.digest() != r.get("payload_digest"):
            failures.append(
                Failure(
                    "payload_digest_mismatch", "the payload does not match the digest in its header", r.get("seq"), rid
                )
            )

    directory = log.payloads.directory
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.suffix == ".json" and path.stem in request_ids:
                continue
            failures.append(Failure("orphan_payload", f"{path.name}: no header points to it", request_id=path.stem))

    return VerifyResult(len(records), checked, tuple(failures))
