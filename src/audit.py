"""Audit records: a header that can be kept, and a payload that can be deleted.

Why two parts. An audit record has to answer "why was this name flagged or cleared, under what" by replay, and
the Money Laundering Regulations 2017 (regulation 40) require records to be kept for at least five years from the
end of the business relationship and the personal data in them deleted after that. So a record is split:

  header   request ID, UTC time, run context hash, the candidates with numeric scores, the decision and a digest
           of the payload. Nothing in it comes from the customer's name, birth date or country, so it can stay in
           an append-only log after the payload is gone. (The numeric scores and the birth date and country
           adjustments are derived from the input, but they are numbers, and they are what replay compares.)
           It carries no retention date: the period runs from the end of the relationship, which the matcher
           does not know, so a purge takes that date from the caller (step 7, see the project notes).
  payload  the optional customer_ref (the caller's opaque link to a customer), the raw input, every normalised
           form, the name tokens, the retrieval keys, the matched names, and a random salt. It lives in a
           separate store, one file per request, and can be deleted on its own.

payload_digest = SHA-256(salt + canonical JSON of the payload data). The digest binds the header to the payload
and, because the salt is random and lives only with the payload, cannot be recomputed by guessing a name once the
payload is deleted.

Canonical JSON: sorted keys, separators "," and ":", ensure_ascii off, encoded as UTF-8. Not NaN or infinity.

The log and the store raise AuditError on any failure. Nothing here swallows an error.

Standard library only.
"""

import fcntl
import json
import os
import re
import secrets
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from audit_schema import CHAIN_FIELDS

SALT_BYTES = 32  # at least 16 are required
_REQUEST_ID = re.compile(r"[0-9a-f]{32}")

GENESIS_PREV_HASH = "0" * 64  # prev_hash of the first record


class AuditError(RuntimeError):
    """A record could not be written. The screening call that needed it must fail."""


class LogMissingError(AuditError):
    """The audit store does not exist or is incomplete. Nothing is screened and nothing is created."""


INIT_COMMAND = "python src/screen_cli.py init --audit-dir"


class AuditConfigError(AuditError):
    """The audit store is not configured. A production entry point refuses to start."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def new_salt() -> bytes:
    return secrets.token_bytes(SALT_BYTES)


def new_request_id() -> str:
    return uuid.uuid4().hex


def payload_digest(salt: bytes, data: Mapping[str, object]) -> str:
    """SHA-256 over the salt followed by the canonical JSON of the payload data, UTF-8."""
    return sha256(salt + canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PayloadRecord:
    request_id: str
    salt: bytes
    data: dict

    def digest(self) -> str:
        return payload_digest(self.salt, self.data)

    def as_json(self) -> str:
        return canonical_json({"request_id": self.request_id, "salt_hex": self.salt.hex(), "data": self.data})

    @classmethod
    def from_json(cls, text: str) -> "PayloadRecord":
        raw = json.loads(text)
        return cls(raw["request_id"], bytes.fromhex(raw["salt_hex"]), raw["data"])


def compute_record_hash(record: Mapping[str, object]) -> str:
    """SHA-256 of the canonical JSON of a logged header, including prev_hash and excluding record_hash."""
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return sha256(canonical_json(body).encode("utf-8")).hexdigest()


class CheckpointError(AuditError):
    """The header line was written but the head checkpoint could not be updated. The header is in the log."""


class HeaderLog:
    """Append-only, hash-chained JSONL file of headers: one canonical JSON object per line.

    On append the log adds seq (0 for the first record), prev_hash (the previous record_hash, or GENESIS_PREV_HASH)
    and record_hash (see compute_record_hash). Appends are serialised by an exclusive flock on `<log>.lock`, held
    from reading the head to writing the checkpoint, so concurrent writers, in threads or processes, cannot fork the
    chain. After each append `<log>.head` is rewritten with the record count and last record_hash. The chain alone
    cannot show that the last N lines were deleted; the checkpoint can. A local checkpoint is tamper-evident, not
    tamper-proof: whoever can rewrite the log can rewrite the checkpoint (production anchors the head outside, in
    WORM storage or a periodic signed export).

    The log refuses to append, and the call fails, when its head is not trustworthy: a torn final line from a crash
    mid-write, an edited last record, or a checkpoint that does not match (a log shorter than its checkpoint is the
    tail-truncation signature). Appending past any of those would hide them. A checkpoint exactly one record behind
    the log, where that record's prev_hash is the checkpoint's hash and its own hash verifies, is the signature of a
    crash between the two writes and is repaired; every other mismatch is refused. flock is POSIX only and not
    reentrant across descriptors: code that holds locked() must call append_locked(), not append().
    """

    def __init__(self, path: Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.checkpoint_path = self.path.with_name(self.path.name + ".head")

    @contextmanager
    def locked(self, create_directory: bool = True) -> Iterator[None]:
        """Hold the exclusive lock. With create_directory False (verification) nothing is made except the lock
        file, and only inside a directory that already exists."""
        if create_directory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing releases the lock

    def require_initialised(self) -> None:
        """Raise LogMissingError unless the directory, the headers file and the checkpoint all exist. Creates
        nothing. A screening run that finds the store missing refuses rather than starting a fresh one, because a
        wiped trail and a new deployment look the same from here."""
        directory = self.path.parent
        missing = (
            "the directory does not exist"
            if not directory.is_dir()
            else "the headers file does not exist"
            if not self.path.exists()
            else "the checkpoint does not exist"
            if not self.checkpoint_path.exists()
            else None
        )
        if missing:
            raise LogMissingError(
                f"log_missing: the audit store at {directory} is not there ({missing}). Screening will not start a "
                f"new one. If this is a new deployment, create it with: {INIT_COMMAND} {directory}. If it is not, "
                "the audit trail has been lost: investigate before anything else."
            )

    def create_genesis(self) -> None:
        """Make an empty store: an empty headers file (exclusive create) and the genesis checkpoint (count 0, the
        genesis hash). Called only by AuditLog.create_new()."""
        try:
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            self._write_checkpoint(0, GENESIS_PREV_HASH)
        except OSError as exc:
            raise AuditError(f"could not create the header log {self.path}: {exc}") from exc

    def read_checkpoint(self) -> dict | None:
        """The stored head {"count", "record_hash"}, or None if there is no checkpoint file."""
        if not self.checkpoint_path.exists():
            return None
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

    def _write_checkpoint(self, count: int, record_hash: str) -> None:
        tmp = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        data = canonical_json({"count": count, "record_hash": record_hash}).encode("utf-8")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            if self.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self.checkpoint_path)
        if self.fsync:
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def _last_line(self) -> bytes | None:
        """The last complete line, or None for an empty or missing file. Raises AuditError on a torn tail."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with open(self.path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                raise AuditError(f"{self.path} ends in a torn line (a crash mid-write); not appending")
            end = fh.seek(0, os.SEEK_END) - 1  # offset of the final newline
            tail = b""
            position = end
            while position > 0:
                step = min(8192, position)
                position -= step
                fh.seek(position)
                tail = fh.read(step) + tail
                if b"\n" in tail:
                    break
            return tail.rsplit(b"\n", 1)[-1]

    def _next_link(self) -> tuple[int, str]:
        """(seq, prev_hash) for the next record, after checking the head against the checkpoint."""
        last = self._last_line()
        checkpoint = self.read_checkpoint()
        if last is None:
            if checkpoint is not None and checkpoint["count"] > 0:
                raise AuditError(f"{self.path} is empty but its checkpoint records {checkpoint['count']} records")
            if checkpoint is None:
                raise LogMissingError(f"log_missing: {self.checkpoint_path} does not exist; not appending")
            return 0, GENESIS_PREV_HASH
        try:
            head = json.loads(last)
        except ValueError as exc:
            raise AuditError(f"the last line of {self.path} is not valid JSON: {exc}") from exc
        if compute_record_hash(head) != head.get("record_hash"):
            raise AuditError(f"the last record of {self.path} does not match its own hash; not appending")
        if checkpoint is None:
            raise AuditError(f"{self.checkpoint_path} is missing for a non-empty log; not appending")
        count, seq = checkpoint["count"], head["seq"]
        if count == seq + 1 and checkpoint["record_hash"] == head["record_hash"]:
            return seq + 1, head["record_hash"]
        if count == seq and checkpoint["record_hash"] == head["prev_hash"]:
            # Exactly one line past the checkpoint, linked to it, and the line's own hash verified above: a crash
            # between the line and the checkpoint. Repair the checkpoint, then continue.
            try:
                self._write_checkpoint(seq + 1, head["record_hash"])
            except OSError as exc:
                raise AuditError(f"could not repair the checkpoint of {self.path}: {exc}") from exc
            return seq + 1, head["record_hash"]
        if count > seq + 1:
            raise AuditError(f"{self.path} has {seq + 1} records but its checkpoint records {count}: tail truncated")
        raise AuditError(f"the head of {self.path} does not match its checkpoint; not appending")

    def append(self, header: Mapping[str, object]) -> dict:
        """Chain and append a header, taking the lock. Returns the record as written. Raises AuditError."""
        try:
            self.require_initialised()
            with self.locked(create_directory=False):
                return self.append_locked(header)
        except AuditError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise AuditError(f"could not append to the header log {self.path}: {exc}") from exc

    def append_locked(self, header: Mapping[str, object]) -> dict:
        """append() for a caller that already holds locked(). The lock covers reading the head to writing the
        checkpoint, so two writers cannot take the same seq."""
        clash = set(CHAIN_FIELDS) & set(header)
        if clash:
            raise AuditError(f"a header to append must not carry {sorted(clash)}")
        self.require_initialised()  # again, under the lock: the store may have been wiped since the first check
        try:
            seq, prev_hash = self._next_link()
            record = {**header, "seq": seq, "prev_hash": prev_hash}
            record["record_hash"] = compute_record_hash(record)
            line = (canonical_json(record) + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                if os.write(fd, line) != len(line):
                    raise OSError("short write")
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)
            try:
                self._write_checkpoint(seq + 1, record["record_hash"])
            except OSError as exc:
                raise CheckpointError(f"header written but the checkpoint could not be: {exc}") from exc
            return record
        except AuditError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise AuditError(f"could not append to the header log {self.path}: {exc}") from exc

    def last_of_kind(self, kind: str) -> dict | None:
        """The most recent complete record of a kind, or None."""
        for record in reversed(self.read_all()):
            if record.get("kind") == kind:
                return record
        return None

    def read_all(self) -> list[dict]:
        """Every complete line. A torn final line (no newline) is left out; verify_chain reports it."""
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        lines.pop()  # the empty piece after the last newline, or the torn fragment
        return [json.loads(line) for line in lines if line.strip()]


class PayloadStore:
    """One file per request ID in a directory. Files are created exclusively, so a payload is never overwritten."""

    def __init__(self, directory: Path, fsync: bool = True) -> None:
        self.directory = Path(directory)
        self.fsync = fsync

    def path_for(self, request_id: str) -> Path:
        if not _REQUEST_ID.fullmatch(request_id):
            raise AuditError(f"not a request ID: {request_id!r}")
        return self.directory / f"{request_id}.json"

    def write(self, record: PayloadRecord) -> None:
        path = self.path_for(record.request_id)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                data = record.as_json().encode("utf-8")
                if os.write(fd, data) != len(data):
                    raise OSError("short write")
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, TypeError, ValueError) as exc:
            raise AuditError(f"could not write the payload {path}: {exc}") from exc

    def read(self, request_id: str) -> PayloadRecord | None:
        path = self.path_for(request_id)
        if not path.exists():
            return None
        return PayloadRecord.from_json(path.read_text(encoding="utf-8"))

    def remove(self, request_id: str) -> None:
        """Delete a payload. Used to undo a payload whose header could not be written; the purge is a later step."""
        self.path_for(request_id).unlink(missing_ok=True)


@dataclass(frozen=True)
class AuditLog:
    headers: HeaderLog
    payloads: PayloadStore

    @classmethod
    def in_directory(cls, directory: Path, fsync: bool = True) -> "AuditLog":
        directory = Path(directory)
        return cls(HeaderLog(directory / "headers.jsonl", fsync), PayloadStore(directory / "payloads", fsync))

    @classmethod
    def create_new(cls, directory: Path, fsync: bool = True) -> "AuditLog":
        """Create a new, empty store: the directory, an empty headers file, the genesis checkpoint and the payload
        directory. Refuses if the directory already exists and is not empty. This is the only way a store is made;
        screening never makes one."""
        directory = Path(directory)
        try:
            if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
                raise AuditError(f"{directory} already exists and is not empty; refusing to initialise it")
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AuditError(f"could not create {directory}: {exc}") from exc
        log = cls.in_directory(directory, fsync)
        log.headers.create_genesis()
        try:
            log.payloads.directory.mkdir()
        except OSError as exc:
            raise AuditError(f"could not create {log.payloads.directory}: {exc}") from exc
        return log

    def write(self, header: Mapping[str, object], payload: PayloadRecord) -> None:
        """Payload first, then header, both under the header log's exclusive lock, so a verifier that takes the same
        lock never sees a payload whose header is still to come. If the header cannot be written the payload is
        removed again. After a crash between the two writes an orphan payload can remain; verify_chain reports it."""
        self.headers.require_initialised()
        try:
            with self.headers.locked(create_directory=False):
                self.payloads.write(payload)
                try:
                    self.headers.append_locked(header)
                except CheckpointError:
                    raise  # the header is in the log, so its payload must stay
                except AuditError:
                    try:
                        self.payloads.remove(payload.request_id)
                    except OSError:
                        pass
                    raise
        except AuditError:
            raise
        except OSError as exc:
            raise AuditError(f"could not write the audit record: {exc}") from exc
