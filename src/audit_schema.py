"""What a header says: the field lists, the decision summary, and how a header is built.

This module defines header content and nothing about how headers are stored. Storage, the hash chain, verification
and replay are in audit.py, audit_verify.py and replay.py and are not hashed into the run context; this file is
(project notes: the hash covers what a record says, not how it is stored). The schema version below and TOP_CANDIDATES
are also run context fields, so a record made under another schema or another candidate cut-off cannot be replayed
as if nothing changed.

Header content never comes from the customer's name, birth date or country. The numeric scores and the birth date
and country adjustments are derived from them, but they are numbers, and they are what replay compares.

Standard library only.
"""

from datetime import datetime, timezone

SCHEMA_VERSION = 4  # written into every header; bump it when a header's fields or meaning change
TOP_CANDIDATES = 10

# Every record has a kind. Only decision records have a payload; the others are events about the system itself.
KIND_DECISION = "decision"  # a name was screened
KIND_ENGINE_START = "engine_start"  # an engine started under a run context the chain had not seen last
KIND_RECOVERY = "recovery"  # an operator recovered from a crash during a write
KIND_PURGE = "purge"  # payloads were deleted at the end of their retention period
KINDS = (KIND_DECISION, KIND_ENGINE_START, KIND_RECOVERY, KIND_PURGE)
PURGE_BASIS = "relationship_end_plus_retention"
MIN_RETENTION_YEARS = 5  # regulation 40 of the Money Laundering Regulations 2017

# The only keys a decision header may have. A test checks a built header against this list, so a new field cannot
# slip in. Each kind has its own list (FIELDS_BY_KIND below); `request_id` is the record's ID for every kind.
HEADER_FIELDS = (
    "schema_version",
    "kind",
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
    "duration_ms",
    "payload_digest",
)
# Added by the log when a header is appended: its position, the hash of the record before it, and its own hash.
CHAIN_FIELDS = ("seq", "prev_hash", "record_hash")
LOGGED_HEADER_FIELDS = HEADER_FIELDS + CHAIN_FIELDS
# engine_start: the engine's run context hash, the one before it, and the NAMES of the context fields that differ. Never
# values (they are not personal data, but a name list is all an auditor needs; the stored context files hold values).
ENGINE_START_FIELDS = (
    "schema_version",
    "kind",
    "request_id",
    "timestamp_utc",
    "context_hash",
    "previous_context_hash",
    "changed_fields",
)
# recovery: what an operator removed after a crash, identified by hash and size, never by content. failure_types lists
# "torn_final_line" (a partial last line, moved to quarantine/) and/or "orphan_payload" (a payload whose header was
# never written: deleted, because no decision was made and there is no retention basis). Fields for a part that did
# not happen are None. The orphan's request_id names no person; its payload file hash lets the deletion be checked
# against a backup.
RECOVERY_FIELDS = (
    "schema_version",
    "kind",
    "request_id",
    "timestamp_utc",
    "failure_types",
    "quarantine_file",
    "quarantine_sha256",
    "quarantine_size",
    "orphan_request_id",
    "orphan_payload_sha256",
)
# purge: the request IDs whose payloads are deleted, the retention period, and the basis. No customer_ref and no end
# dates: they would link records back to a person. The record is appended before any file is deleted, so a payload that
# a purge record covers is gone or about to be.
PURGE_FIELDS = (
    "schema_version",
    "kind",
    "request_id",
    "timestamp_utc",
    "basis",
    "retention_years",
    "purged_request_ids",
)
FIELDS_BY_KIND = {
    KIND_DECISION: HEADER_FIELDS,
    KIND_ENGINE_START: ENGINE_START_FIELDS,
    KIND_RECOVERY: RECOVERY_FIELDS,
    KIND_PURGE: PURGE_FIELDS,
}
LOGGED_FIELDS_BY_KIND = {kind: fields + CHAIN_FIELDS for kind, fields in FIELDS_BY_KIND.items()}
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


def utc_timestamp(moment: datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("audit timestamps must be timezone-aware UTC")
    return moment.isoformat(timespec="microseconds")


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


DECISION_FIELDS = (
    "candidates_scored",
    "candidates",
    "final_score",
    "threshold",
    "decision",
    "matched_entity_id",
    "matched_entity_ids",
)


def decision_summary(entries: list[dict], threshold: float, top: int = TOP_CANDIDATES) -> dict:
    """The part of a header that is the matcher's answer, and nothing else: the candidates kept, how many were
    scored, the best score, the threshold, the decision and the matched IDs. build_header() uses it and so does
    replay, so the two cannot disagree about what a decision is."""
    ordered = sorted(entries, key=lambda e: (-e["score"], e["entity_id"]))
    matched = [e["entity_id"] for e in ordered if e["at_or_above_threshold"]]
    return {
        "candidates_scored": len(entries),
        "candidates": choose_candidates(entries, top),
        "final_score": ordered[0]["score"] if ordered else None,  # the best candidate's score, alert or not
        "threshold": threshold,
        "decision": "alert" if matched else "clear",
        "matched_entity_id": matched[0] if matched else None,
        "matched_entity_ids": matched,
    }


def build_header(
    *,
    request_id: str,
    moment: datetime,
    run_context_hash: str,
    entries: list[dict],
    threshold: float,
    digest: str,
    duration_ms: int,
    top: int = TOP_CANDIDATES,
) -> dict:
    """`duration_ms` is the wall time of the engine's evaluate() call for this input, an integer number of
    milliseconds: a number about the system, not about the customer."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_DECISION,
        "request_id": request_id,
        "timestamp_utc": utc_timestamp(moment),
        "run_context_hash": run_context_hash,
        **decision_summary(entries, threshold, top),
        "duration_ms": duration_ms,
        "payload_digest": digest,
    }


def build_engine_start_header(
    *,
    request_id: str,
    moment: datetime,
    context_hash: str,
    previous_context_hash: str | None,
    changed_fields: list[str],
) -> dict:
    """An engine started under `context_hash`. `previous_context_hash` is None for the first start in the chain.
    `changed_fields` are field names (such as "config.global_threshold" or "snapshot_id"), never values."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_ENGINE_START,
        "request_id": request_id,
        "timestamp_utc": utc_timestamp(moment),
        "context_hash": context_hash,
        "previous_context_hash": previous_context_hash,
        "changed_fields": sorted(changed_fields),
    }


def build_recovery_header(
    *,
    request_id: str,
    moment: datetime,
    quarantine_file: str | None = None,
    quarantine_sha256: str | None = None,
    quarantine_size: int | None = None,
    orphan_request_id: str | None = None,
    orphan_payload_sha256: str | None = None,
) -> dict:
    failure_types = []
    if quarantine_file is not None:
        failure_types.append("torn_final_line")
    if orphan_request_id is not None:
        failure_types.append("orphan_payload")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_RECOVERY,
        "request_id": request_id,
        "timestamp_utc": utc_timestamp(moment),
        "failure_types": failure_types,
        "quarantine_file": quarantine_file,
        "quarantine_sha256": quarantine_sha256,
        "quarantine_size": quarantine_size,
        "orphan_request_id": orphan_request_id,
        "orphan_payload_sha256": orphan_payload_sha256,
    }


def build_purge_header(
    *, request_id: str, moment: datetime, retention_years: int, purged_request_ids: list[str]
) -> dict:
    """The tombstone for a purge: which payloads are deleted, under what rule. Request IDs only."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_PURGE,
        "request_id": request_id,
        "timestamp_utc": utc_timestamp(moment),
        "basis": PURGE_BASIS,
        "retention_years": retention_years,
        "purged_request_ids": sorted(purged_request_ids),
    }
