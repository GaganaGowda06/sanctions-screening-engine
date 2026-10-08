"""Operator recovery after a crash during a write: a torn final line and/or one orphan payload.

An append is two writes under one lock: the payload, then the header. A crash can leave the log ending in a partial
line (the header write was cut), a payload with no header (the crash came before any header byte), or both. The log
then refuses to append, so screening stops until someone decides. This is that decision, made visible.

  1. Under the log's exclusive lock, verify the whole store. Recovery goes ahead only if every finding is
     torn_final_line or orphan_payload, and at least one is present. Any tampering-class failure, or any other
     finding, is refused with nothing changed. So is a clean log.
  2. More than one orphan payload is refused, nothing changed. Only one append can be in flight under the lock, so a
     real crash leaves at most one; several mean something else happened.
  3. A torn line is written to <audit_dir>/quarantine/<timestamp>.bin (exclusive create, flushed) before anything is
     removed, and the log is truncated back to its last complete line.
  4. One recovery record (kind "recovery") is appended to the chain with the SHA-256 and size of the torn bytes (if
     any) and the request ID and payload file SHA-256 of the orphan (if any). No payload content.
  5. The orphan payload file is deleted. It is deleted, not quarantined: its screening never produced a decision, so
     there is no retention basis for keeping the personal data in it.

The record is appended before the file is deleted. If the process dies between the two, the orphan is still there,
a recovery record already names it, and running recovery again deletes the file and appends nothing new.

The quarantine directory is evidence; do not delete it. Deletion here is deletion of a file only: copies in backups
or filesystem snapshots are outside its reach.
"""

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from audit import AuditError, AuditLog, new_request_id
from audit_schema import KIND_RECOVERY, build_recovery_header
from audit_verify import _verify_locked

QUARANTINE_DIR = "quarantine"
RECOVERABLE = frozenset({"torn_final_line", "orphan_payload"})
_PAYLOAD_NAME = re.compile(r"[0-9a-f]{32}\.json")


class RecoveryRefused(AuditError):
    """Recovery was refused. Nothing was changed."""

    def __init__(self, message: str, kinds: frozenset[str] = frozenset()) -> None:
        super().__init__(message)
        self.kinds = kinds


@dataclass(frozen=True)
class RecoveryOutcome:
    quarantine_file: str | None
    quarantine_sha256: str | None
    quarantine_size: int | None
    orphan_request_id: str | None
    orphan_payload_sha256: str | None
    record: dict | None  # None when only finishing a deletion an earlier recovery record already covers


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def recover(log: AuditLog, clock: Callable[[], datetime] = utc_now) -> RecoveryOutcome:
    """Quarantine a torn final line and/or delete one orphan payload, and record it in the chain. Raises
    RecoveryRefused, changing nothing, unless the findings are exactly those two crash signatures; raises AuditError
    if the store is missing or a write fails."""
    headers = log.headers
    headers.require_initialised()
    with headers.locked(create_directory=False):
        result = _verify_locked(log)
        kinds = frozenset(result.kinds())
        if result.tampering:
            names = ", ".join(sorted({f.kind for f in result.tampering}))
            raise RecoveryRefused(
                f"refusing to recover: the store has tampering-class failures ({names}). Recovery is for a crash, "
                "not for damage someone may have done. Nothing was changed.",
                kinds,
            )
        other = kinds - RECOVERABLE
        if other:
            raise RecoveryRefused(
                "refusing to recover: findings other than a torn line or an orphan payload "
                f"({', '.join(sorted(other))}). Nothing was changed.",
                kinds,
            )
        if not kinds:
            raise RecoveryRefused(
                "nothing to recover: no torn final line and no orphan payload. Nothing was changed.", kinds
            )
        orphans = [f for f in result.failures if f.kind == "orphan_payload"]
        if len(orphans) > 1:
            raise RecoveryRefused(
                f"refusing to recover: {len(orphans)} orphan payloads. Only one append can be in flight under the "
                "lock, so a crash leaves at most one; more means something else happened. Nothing was changed.",
                kinds,
            )
        orphan_path = None
        if orphans:
            orphan_path = log.payloads.directory / f"{orphans[0].request_id}.json"
            if not _PAYLOAD_NAME.fullmatch(orphan_path.name) or not orphan_path.is_file():
                raise RecoveryRefused(
                    "refusing to recover: the orphan is not a payload file this store wrote. Nothing was changed.",
                    kinds,
                )
        orphan_bytes = orphan_path.read_bytes() if orphan_path else None
        orphan_id = orphan_path.stem if orphan_path else None
        orphan_sha = hashlib.sha256(orphan_bytes).hexdigest() if orphan_bytes is not None else None

        # An earlier run appended the record and died before deleting the file: finish the deletion only.
        if (
            orphan_id
            and "torn_final_line" not in kinds
            and any(
                r.get("kind") == KIND_RECOVERY and r.get("orphan_request_id") == orphan_id for r in headers.read_all()
            )
        ):
            delete_payload(orphan_path, fsync=log.payloads.fsync)
            return RecoveryOutcome(None, None, None, orphan_id, orphan_sha, None)

        moment = clock()
        name = sha = size = None
        if "torn_final_line" in kinds:
            raw = headers.path.read_bytes()
            cut = raw.rfind(b"\n") + 1  # 0 if there is no complete line at all
            torn = raw[cut:]
            name = moment.strftime("%Y%m%dT%H%M%S%fZ") + ".bin"
            sha, size = hashlib.sha256(torn).hexdigest(), len(torn)
            directory = headers.path.parent / QUARANTINE_DIR
            try:
                directory.mkdir(exist_ok=True)
                fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.write(fd, torn)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                fd = os.open(headers.path, os.O_WRONLY)
                try:
                    os.ftruncate(fd, cut)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise AuditError(f"could not quarantine the torn line: {exc}") from exc

        record = headers.append_locked(
            build_recovery_header(
                request_id=new_request_id(),
                moment=moment,
                quarantine_file=name,
                quarantine_sha256=sha,
                quarantine_size=size,
                orphan_request_id=orphan_id,
                orphan_payload_sha256=orphan_sha,
            )
        )
        if orphan_path:
            delete_payload(orphan_path, fsync=log.payloads.fsync)
    return RecoveryOutcome(name, sha, size, orphan_id, orphan_sha, record)


def delete_payload(path: Path, fsync: bool = True) -> None:
    """Delete one payload file (the file only: backups and snapshots are outside this module's reach)."""
    try:
        path.unlink(missing_ok=True)
        if fsync:
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except OSError as exc:
        raise AuditError(f"could not delete {path.name}: {exc}") from exc
