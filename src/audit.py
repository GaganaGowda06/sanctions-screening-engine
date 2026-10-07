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

import json
import os
import re
import secrets
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

HEADER_SCHEMA_VERSION = 1
SALT_BYTES = 32  # at least 16 are required
TOP_CANDIDATES = 10
_REQUEST_ID = re.compile(r"[0-9a-f]{32}")

# The only keys a header may have. A test checks a built header against this list, so a new field cannot slip in.
HEADER_FIELDS = (
    "schema_version",
    "request_id",
    "timestamp_utc",
    "run_context_hash",
    "candidates_scored",
    "candidates",
    "final_score",
    "threshold",
    "decision",
    "matched_entity_id",
    "matched_entity_ids",
    "payload_digest",
)
CANDIDATE_FIELDS = (
    "entity_id",
    "score",
    "name_score",
    "skeleton_score",
    "birth_date_adjustment",
    "country_adjustment",
    "arabic_pair",
    "skeleton_used",
    "at_or_above_threshold",
)


class AuditError(RuntimeError):
    """A record could not be written. The screening call that needed it must fail."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def new_salt() -> bytes:
    return secrets.token_bytes(SALT_BYTES)


def new_request_id() -> str:
    return uuid.uuid4().hex


def payload_digest(salt: bytes, data: Mapping[str, object]) -> str:
    """SHA-256 over the salt followed by the canonical JSON of the payload data, UTF-8."""
    return sha256(salt + canonical_json(data).encode("utf-8")).hexdigest()


def utc_timestamp(moment: datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("audit timestamps must be timezone-aware UTC")
    return moment.isoformat(timespec="microseconds")


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


def candidate_entry(
    entity_id: str,
    score: float,
    name_score: float,
    skeleton_score: float,
    birth_date_adjustment: float,
    country_adjustment: float,
    arabic_pair: bool,
    skeleton_used: bool,
    threshold: float,
) -> dict:
    return {
        "entity_id": entity_id,
        "score": score,
        "name_score": name_score,
        "skeleton_score": skeleton_score,
        "birth_date_adjustment": birth_date_adjustment,
        "country_adjustment": country_adjustment,
        "arabic_pair": arabic_pair,
        "skeleton_used": skeleton_used,
        "at_or_above_threshold": score >= threshold,
    }


def choose_candidates(entries: list[dict], top: int = TOP_CANDIDATES) -> list[dict]:
    """The top `top` by score, plus every candidate at or above the threshold, best first, ties by entity ID.

    `entries` are candidate_entry() dicts for all scored candidates. Every at-or-above-threshold candidate is kept
    even if it is ranked past `top`.
    """
    ordered = sorted(entries, key=lambda e: (-e["score"], e["entity_id"]))
    return [e for i, e in enumerate(ordered) if i < top or e["at_or_above_threshold"]]


def build_header(
    *,
    request_id: str,
    moment: datetime,
    run_context_hash: str,
    entries: list[dict],
    threshold: float,
    digest: str,
    top: int = TOP_CANDIDATES,
) -> dict:
    ordered = sorted(entries, key=lambda e: (-e["score"], e["entity_id"]))
    matched = [e["entity_id"] for e in ordered if e["at_or_above_threshold"]]
    return {
        "schema_version": HEADER_SCHEMA_VERSION,
        "request_id": request_id,
        "timestamp_utc": utc_timestamp(moment),
        "run_context_hash": run_context_hash,
        "candidates_scored": len(entries),
        "candidates": choose_candidates(entries, top),
        "final_score": ordered[0]["score"] if ordered else None,  # the best candidate's score, alert or not
        "threshold": threshold,
        "decision": "alert" if matched else "clear",
        "matched_entity_id": matched[0] if matched else None,
        "matched_entity_ids": matched,
        "payload_digest": digest,
    }


class HeaderLog:
    """Append-only JSONL file of headers: one canonical JSON object per line. There is no way to edit or remove a
    line through this class. Each line is written with one write() on a descriptor opened with O_APPEND and then
    flushed to disk."""

    def __init__(self, path: Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync

    def append(self, header: Mapping[str, object]) -> None:
        try:
            line = (canonical_json(header) + "\n").encode("utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                if os.write(fd, line) != len(line):
                    raise OSError("short write")
                if self.fsync:
                    os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, TypeError, ValueError) as exc:
            raise AuditError(f"could not append to the header log {self.path}: {exc}") from exc

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


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

    def write(self, header: Mapping[str, object], payload: PayloadRecord) -> None:
        """Payload first, then header. If the header cannot be written the payload is removed again, so there is no
        header without a payload that matches it and no payload without a header unless removal also fails."""
        self.payloads.write(payload)
        try:
            self.headers.append(header)
        except AuditError:
            try:
                self.payloads.remove(payload.request_id)
            except OSError:
                pass
            raise
