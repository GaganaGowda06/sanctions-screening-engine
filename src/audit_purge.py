"""Retention purge: delete payloads whose retention period has ended, and say so in the chain first.

Regulation 40 of the Money Laundering Regulations 2017 requires records to be kept for at least five years, counted
from the end of the business relationship, and personal data deleted after that. The matcher does not know when a
relationship ends, so the caller supplies it: a mapping of customer_ref to relationship end date. A timestamp-based
expiry would delete records of customers who are still customers.

A decision's payload is eligible when all of these hold:
  - the payload has a customer_ref, and it is in the mapping;
  - the record's timestamp is on or before the end date (a record made after the end date belongs to a later
    relationship, and is skipped and counted);
  - the end date plus the retention period is before today.
Never eligible: no customer_ref; a customer_ref not in the mapping (a live customer, as far as this store knows); a
record dated after its customer's end date; a retention period not yet over.

Legal hold. The caller can also pass holds: customer_refs and/or request_ids whose records are never purged, whatever
their end date. Deleting records that are relevant to an investigation is an offence (regulation 87 of the Money
Laundering Regulations 2017), so a hold is checked first, before any eligibility rule, and a dry run shows it. The
hold list is the caller's responsibility: this store cannot know what is relevant to an investigation. A hold also
stops the finishing of an interrupted purge for a held record (its file stays and verify_chain keeps reporting
purge_incomplete until the hold is lifted). A hold on a request that was already purged cannot be honoured; it is
counted so the caller knows. Holds are not written to the purge record or anywhere else in the chain.

Order of work, under the log's exclusive lock:
  1. verify the store; refuse on any tampering-class failure or anything other than the crash signatures this module
     can finish (purge_incomplete) or ignore (orphan_payload);
  2. work out the eligible request IDs, and the IDs already covered by an earlier purge record whose file is still
     there (a purge that died between its record and its deletes);
  3. append ONE chained purge record naming the newly eligible request IDs, the retention period and the basis
     "relationship_end_plus_retention". It holds no customer_ref and no end date: they would link records to a person;
  4. delete the payload files.
Running purge again after a crash between 3 and 4 deletes the leftover files and appends nothing for IDs a purge
record already names. verify_chain treats a missing payload that a purge record covers as purged, not a failure.

A dry run does steps 1 and 2 and reports counts. It changes nothing.

THIS DELETES FILES ONLY. It is not a secure erase. Copies in backups, filesystem snapshots, journals, swap or
anywhere else the bytes went are outside its reach, and production retention has to cover them: encrypt payloads at
rest with per-record or per-period keys and destroy the keys, or expire backups on the same schedule.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone

from audit import AuditError, AuditLog, new_request_id
from audit_recover import delete_payload
from audit_schema import KIND_DECISION, KIND_PURGE, MIN_RETENTION_YEARS, build_purge_header
from audit_verify import _verify_locked

ALLOWED_FINDINGS = frozenset({"purge_incomplete", "orphan_payload"})


class PurgeRefused(AuditError):
    """The purge did not run. Nothing was changed."""

    def __init__(self, message: str, kinds: frozenset[str] = frozenset()) -> None:
        super().__init__(message)
        self.kinds = kinds


@dataclass(frozen=True)
class PurgeResult:
    dry_run: bool
    eligible: int  # payloads newly covered by this run's purge record (or that would be, in a dry run)
    completed: int  # payloads an earlier purge record covers whose files were still there
    skipped_no_customer_ref: int
    skipped_not_in_mapping: int  # live customers, as far as this store knows
    skipped_later_relationship: int  # dated after their customer's end date
    skipped_retention_not_elapsed: int
    held: int  # records under legal hold: never purged, checked before the other rules
    hold_already_purged: int  # held request IDs whose payloads an earlier purge record had already covered
    record: dict | None  # the purge record appended, None for a dry run or when nothing was newly eligible
    eligible_ids: tuple[str, ...]


def add_years(day: date, years: int) -> date:
    """The same day `years` later; 29 February becomes 1 March so a retention period is never shortened."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return date(day.year + years, 3, 1)


def parse_mapping(mapping: Mapping[str, date | str]) -> dict[str, date]:
    """customer_ref -> end date. Raises ValueError for an empty reference or a date that is not YYYY-MM-DD."""
    parsed = {}
    for ref, end in mapping.items():
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"customer_ref must be a non-empty string, got {ref!r}")
        if isinstance(end, datetime):
            end = end.date()
        if not isinstance(end, date):
            try:
                end = date.fromisoformat(end)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"relationship end for {ref!r} must be a date (YYYY-MM-DD)") from exc
        parsed[ref] = end
    return parsed


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def purge(
    log: AuditLog,
    mapping: Mapping[str, date | str],
    retention_years: int = MIN_RETENTION_YEARS,
    now: datetime | None = None,
    dry_run: bool = False,
    hold_customer_refs: Collection[str] = (),
    hold_request_ids: Collection[str] = (),
    clock: Callable[[], datetime] = utc_now,
) -> PurgeResult:
    """Purge payloads whose retention period has ended (see the module docstring). `now` is the date the retention
    period is measured to (default: the current UTC time). Raises ValueError for a bad argument, PurgeRefused if the
    store is not in a state to purge, AuditError on a write failure."""
    if (
        not isinstance(retention_years, int)
        or isinstance(retention_years, bool)
        or retention_years < MIN_RETENTION_YEARS
    ):
        raise ValueError(f"retention_years must be an integer of at least {MIN_RETENTION_YEARS} (regulation 40)")
    ends = parse_mapping(mapping)
    hold_refs, hold_ids = frozenset(hold_customer_refs), frozenset(hold_request_ids)
    now = now if now is not None else clock()
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    today = now.astimezone(timezone.utc).date()

    headers = log.headers
    headers.require_initialised()
    with headers.locked(create_directory=False):
        result = _verify_locked(log)
        kinds = frozenset(result.kinds())
        if result.tampering:
            names = ", ".join(sorted({f.kind for f in result.tampering}))
            raise PurgeRefused(
                f"refusing to purge: the store has tampering-class failures ({names}). Nothing was changed.", kinds
            )
        if kinds - ALLOWED_FINDINGS:
            raise PurgeRefused(
                f"refusing to purge: the store has findings ({', '.join(sorted(kinds - ALLOWED_FINDINGS))}) that "
                "need the recover command first. Nothing was changed.",
                kinds,
            )

        records = headers.read_all()
        tombstoned = {i for r in records if r.get("kind") == KIND_PURGE for i in (r.get("purged_request_ids") or [])}
        eligible, completing = [], []
        counts = {"no_ref": 0, "not_in_mapping": 0, "later": 0, "not_elapsed": 0, "held": 0}

        def held(rid: str, ref: str | None) -> bool:
            return rid in hold_ids or (ref is not None and ref in hold_refs)

        for header in records:
            if header.get("kind") != KIND_DECISION:
                continue
            rid = header["request_id"]
            if rid in tombstoned:
                if log.payloads.path_for(rid).exists():
                    payload = log.payloads.read(rid)
                    if held(rid, payload.data.get("customer_ref") if payload else None):
                        counts["held"] += 1  # the hold wins over finishing an earlier purge
                    else:
                        completing.append(rid)
                continue
            payload = log.payloads.read(rid)  # verify_chain already refused a missing or damaged one
            ref = payload.data.get("customer_ref") if payload else None
            if held(rid, ref):
                counts["held"] += 1  # checked before every eligibility rule
            elif ref is None:
                counts["no_ref"] += 1
            elif ref not in ends:
                counts["not_in_mapping"] += 1
            elif datetime.fromisoformat(header["timestamp_utc"]).astimezone(timezone.utc).date() > ends[ref]:
                counts["later"] += 1
            elif not add_years(ends[ref], retention_years) < today:
                counts["not_elapsed"] += 1
            else:
                eligible.append(rid)
        too_late = len(hold_ids & tombstoned) - sum(
            1 for rid in hold_ids & tombstoned if log.payloads.path_for(rid).exists()
        )  # held request IDs whose payload file is already gone

        record = None
        if not dry_run:
            if eligible:
                record = headers.append_locked(
                    build_purge_header(
                        request_id=new_request_id(),
                        moment=clock(),
                        retention_years=retention_years,
                        purged_request_ids=eligible,
                    )
                )
            for rid in [*eligible, *completing]:
                delete_payload(log.payloads.path_for(rid), fsync=log.payloads.fsync)
    return PurgeResult(
        dry_run,
        len(eligible),
        len(completing),
        counts["no_ref"],
        counts["not_in_mapping"],
        counts["later"],
        counts["not_elapsed"],
        counts["held"],
        too_late,
        record,
        tuple(sorted(eligible)),
    )
