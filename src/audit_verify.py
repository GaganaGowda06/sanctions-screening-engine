"""verify_chain(): check the header log, its checkpoint and the payload store, and say what is wrong.

Failure kinds, split into tampering and crash signatures so a crash is never reported as an attack:

  tampering
    log_missing            the audit directory, or the headers file, does not exist. A store that was initialised
                           has both a headers file and a checkpoint, so a directory with neither is not clean
    schema_violation       a record of an unknown kind, or whose fields are not exactly its kind's allow-list
                           (audit_schema.LOGGED_FIELDS_BY_KIND)
    quarantine_mismatch    a recovery record names a quarantine file that is there but does not match the SHA-256
                           and size the record holds
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

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from audit import GENESIS_PREV_HASH, AuditError, AuditLog, compute_record_hash
from audit_schema import KIND_DECISION, KIND_RECOVERY, LOGGED_FIELDS_BY_KIND

TAMPERING = frozenset(
    {
        "log_missing",
        "schema_violation",
        "quarantine_mismatch",
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
    """Walk the whole header log, then the checkpoint, then every payload. Never raises for a bad log; it reports.

    Verification creates nothing: with no audit directory it returns log_missing at once. In an existing directory
    it may create the lock file, and nothing else. Deleting the whole audit directory leaves nothing to compare
    with, so it is undetectable locally; external anchoring covers it in production (see the project notes)."""
    if not log.headers.path.parent.is_dir():
        return VerifyResult(0, 0, (Failure("log_missing", "the audit directory does not exist"),))
    with log.headers.locked(
        create_directory=False
    ):  # one consistent view of the log, its checkpoint and the payload store
        return _verify_locked(log)


def check_schema(record: dict, seq: int) -> list[Failure]:
    allowed = LOGGED_FIELDS_BY_KIND.get(record.get("kind"))
    rid = record.get("request_id") if isinstance(record.get("request_id"), str) else None
    if allowed is None:
        return [Failure("schema_violation", f"record {seq} has an unknown kind {record.get('kind')!r}", seq, rid)]
    if set(record) != set(allowed):
        extra, missing = sorted(set(record) - set(allowed)), sorted(set(allowed) - set(record))
        return [Failure("schema_violation", f"record {seq} fields differ: extra {extra}, missing {missing}", seq, rid)]
    return []


def check_recovery(record: dict, seq: int, directory: Path) -> list[Failure]:
    """A recovery record must be consistent (the torn-line fields all present or all None, same for the orphan
    fields, and failure_types matching), and its quarantine file, if still there, must be the bytes it describes. A
    file that is gone is not a failure here: it is evidence, not part of the chain."""
    rid = record.get("request_id") if isinstance(record.get("request_id"), str) else None
    name = record.get("quarantine_file")
    torn = [record.get(k) for k in ("quarantine_file", "quarantine_sha256", "quarantine_size")]
    orphan = [record.get(k) for k in ("orphan_request_id", "orphan_payload_sha256")]
    expected_types = (["torn_final_line"] if torn[0] is not None else []) + (
        ["orphan_payload"] if orphan[0] is not None else []
    )
    problems = []
    if any(v is None for v in torn) and any(v is not None for v in torn):
        problems.append("the quarantine fields are only partly set")
    if any(v is None for v in orphan) and any(v is not None for v in orphan):
        problems.append("the orphan fields are only partly set")
    if record.get("failure_types") != expected_types or not expected_types:
        problems.append("failure_types does not match the fields that are set")
    if name is not None and (not isinstance(name, str) or Path(name).name != name or not name):
        problems.append("the quarantine file is not a plain file name")
    if problems:
        return [Failure("schema_violation", f"record {seq}: {'; '.join(problems)}", seq, rid)]
    if name is None:
        return []
    path = directory / name
    if not path.is_file():
        return []
    data = path.read_bytes()
    same_hash = hashlib.sha256(data).hexdigest() == record.get("quarantine_sha256")
    if not same_hash or len(data) != record.get("quarantine_size"):
        return [Failure("quarantine_mismatch", f"{name} does not match the recovery record {seq}", seq, rid)]
    return []


def _verify_locked(log: AuditLog) -> VerifyResult:
    headers = log.headers
    failures: list[Failure] = []
    headers_exist = headers.path.exists()
    raw = headers.path.read_bytes() if headers_exist else b""
    try:
        checkpoint = headers.read_checkpoint()
    except ValueError:
        checkpoint = {"count": None, "record_hash": None}
    if not headers_exist:
        failures.append(Failure("log_missing", "the headers file does not exist"))

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
        failures += check_schema(record, i)
        if record.get("kind") == KIND_RECOVERY:
            failures += check_recovery(record, i, headers.path.parent / "quarantine")
        expected_prev = record.get("record_hash")

    n = len(lines)
    if checkpoint is None:
        if headers_exist:  # an initialised store always has a checkpoint, even for an empty log
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

    decisions = [r for r in records if r.get("kind") == KIND_DECISION]  # only decisions have payloads
    request_ids = {r.get("request_id") for r in decisions}
    checked = 0
    for r in decisions:
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


def classify_missing_payload(log: AuditLog, request_id: str) -> str:
    """Why a record's payload is not there, using verify_chain's classification: "purged" for a payload covered by a
    chained purge tombstone, otherwise "payload_missing", the tampering-class failure verify_chain reports. No purge
    exists yet, so no tombstone can, and the answer is always "payload_missing" (step 7 adds the tombstone check)."""
    return "payload_missing"
