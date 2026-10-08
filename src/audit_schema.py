"""What a header says: the field lists, the decision summary, and how a header is built.

This module defines header content and nothing about how headers are stored. Storage, the hash chain, verification
and replay are in audit.py, audit_verify.py and replay.py and are not hashed into the run context; this file is
(the project notes: the hash covers what a record says, not how it is stored). The schema version below and TOP_CANDIDATES
are also run context fields, so a record made under another schema or another candidate cut-off cannot be replayed
as if nothing changed.

Header content never comes from the customer's name, birth date or country. The numeric scores and the birth date
and country adjustments are derived from them, but they are numbers, and they are what replay compares.

Standard library only.
"""

from datetime import datetime, timezone

SCHEMA_VERSION = 1  # written into every header; bump it when a header's fields or meaning change
TOP_CANDIDATES = 10

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
# Added by the log when a header is appended: its position, the hash of the record before it, and its own hash.
CHAIN_FIELDS = ("seq", "prev_hash", "record_hash")
LOGGED_HEADER_FIELDS = HEADER_FIELDS + CHAIN_FIELDS
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
    top: int = TOP_CANDIDATES,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "timestamp_utc": utc_timestamp(moment),
        "run_context_hash": run_context_hash,
        **decision_summary(entries, threshold, top),
        "payload_digest": digest,
    }
