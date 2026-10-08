"""Batch monitoring metrics from the audit log: aggregate numbers only, one report per window of records.

    report, path = compute_metrics(audit_dir, from_seq, to_seq)

The window is the records with seq from_seq to to_seq inclusive (seq is a record's position in the chain). Only
decision records are measured; other kinds in the window are counted and left out. Records are grouped by run context
hash and metrics are never mixed across contexts: a threshold change or a new list snapshot starts a new group, so a
shift in the numbers cannot be blamed on traffic when the engine itself changed. Per group:

  count, alert rate
  final score histogram: 20 equal bins on [0, 1], fixed so a later PSI compares like with like. Bin i is
      [i/20, (i+1)/20) and the last bin includes 1.0. Records with no candidate at all have no final score and are
      counted separately (no_candidates)
  near-threshold count and rate: |final score - the record's threshold| <= near_threshold_band (from config; the
      difference is rounded to 9 places so a score exactly one band away counts, whatever the float noise)
  per-script count and alert rate. The script of a record is query_script() (the Stage 3 script detection in
      screening.py) applied to the raw name in its payload. A payload that is purged, missing, unreadable or does not
      match its digest counts as "unavailable"
  duration_ms p50, p95, p99 (nearest rank: the value at position ceil(p/100 * n) of the sorted durations)

Small-cell suppression (min_cell from config):
  - a context group with fewer than min_cell decisions reports its count as "<min_cell" and nothing else: no rates,
    histogram or durations;
  - a per-script cell with fewer than min_cell records reports its count as "<min_cell" and no rate;
  - secondary suppression: if exactly one per-script cell of a group is suppressed, the smallest remaining visible
    cell (not "unavailable") is withheld too, because otherwise the hidden count is the group total minus the visible
    cells. Its count shows as "withheld". With no such cell, "unavailable" is withheld instead. Two or more hidden
    cells cannot be told apart by subtraction;
  - the same arithmetic works across groups through the window's decision total, so when any group is suppressed
    window.decisions is "withheld".
Counts that are shown are exact; what is hidden cannot be recovered by subtraction.

Integrity first. compute_metrics verifies the whole chain (verify_chain's checks) under the lock and refuses on any
tampering-class finding: metrics from a log someone may have edited would look authoritative and mean nothing.
Crash-class findings (an orphan payload, a torn tail, an interrupted purge) are allowed and listed in the report by
type and count, never by request ID.

The report is JSON in <audit_dir>/monitoring/<from_seq>-<to_seq>.json, created exclusively (an existing report for the
window is never overwritten), and holds no request ID, name, customer reference or payload content. The window must
lie inside the log, so the same window always names the same records. Metrics read the log under its lock and
write nothing to it. They do not verify the chain: run verify_chain first if the log's integrity is in doubt.

Standard library only, apart from the project's own modules.
"""

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from audit import AuditError, AuditLog
from audit_schema import KIND_DECISION
from audit_verify import _verify_locked
from screening import query_script

BINS = 20  # fixed: later drift (PSI) compares histograms over the same bins
REPORT_SCHEMA_VERSION = 1
CONFIG_PATH = Path(__file__).with_name("monitoring_config.json")
SCRIPTS = ("Latin", "Cyrillic", "Arabic", "other", "unavailable")


@dataclass(frozen=True)
class MetricsConfig:
    near_threshold_band: float = 0.05
    min_cell: int = 10
    source_sha256: str | None = field(default=None, compare=False)  # SHA-256 of the file this was loaded from

    def __post_init__(self) -> None:
        if not isinstance(self.near_threshold_band, (int, float)) or not 0 <= self.near_threshold_band <= 1:
            raise ValueError("near_threshold_band must be a number from 0 to 1")
        if not isinstance(self.min_cell, int) or isinstance(self.min_cell, bool) or self.min_cell < 1:
            raise ValueError("min_cell must be a whole number of at least 1")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "MetricsConfig":
        """The saved configuration, or the defaults if the file does not exist."""
        if not path.exists():
            return cls()
        raw = path.read_bytes()
        data = json.loads(raw)
        return cls(data["near_threshold_band"], data["min_cell"], hashlib.sha256(raw).hexdigest())


def bin_index(score: float) -> int:
    """The histogram bin of a final score in [0, 1]. The epsilon keeps a score such as 0.3 out of the wrong bin."""
    return min(max(int(math.floor(score * BINS + 1e-9)), 0), BINS - 1)


def percentile(sorted_values: list[int], p: int) -> int | None:
    """Nearest-rank percentile of an already sorted list (None for an empty list)."""
    if not sorted_values:
        return None
    return sorted_values[max(math.ceil(p / 100 * len(sorted_values)), 1) - 1]


def record_script(log: AuditLog, header: dict) -> str:
    """The script of a decision's name, from its payload, or "unavailable"."""
    try:
        payload = log.payloads.read(header["request_id"])
        if payload is None or payload.digest() != header.get("payload_digest"):
            return "unavailable"
        return query_script(payload.data["raw_input"]["name"])
    except (AuditError, ValueError, KeyError, TypeError, OSError):
        return "unavailable"


def rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def suppress_script_cells(per_script: dict[str, dict], total: int, min_cell: int) -> dict[str, dict]:
    """Primary suppression is done by the caller ("<min_cell" cells). If exactly one cell is hidden, withhold the
    smallest remaining visible one too (excluding "unavailable" unless nothing else is visible)."""
    hidden = [k for k, v in per_script.items() if not isinstance(v["count"], int)]
    if len(hidden) != 1:
        return per_script
    visible = [k for k, v in per_script.items() if isinstance(v["count"], int)]
    pool = [k for k in visible if k != "unavailable"] or visible
    if not pool:
        return per_script
    smallest = min(pool, key=lambda k: (per_script[k]["count"], SCRIPTS.index(k)))
    per_script[smallest] = {"count": "withheld"}
    return per_script


def group_metrics(rows: list[dict], config: MetricsConfig, context_hash: str) -> dict:
    count = len(rows)
    if count < config.min_cell:
        return {"run_context_hash": context_hash, "count": f"<{config.min_cell}"}
    alerts = sum(r["decision"] == "alert" for r in rows)
    counts = [0] * BINS
    no_candidates = near = 0
    for r in rows:
        if r["final_score"] is None:
            no_candidates += 1
            continue
        counts[bin_index(r["final_score"])] += 1
        if round(abs(r["final_score"] - r["threshold"]), 9) <= config.near_threshold_band:
            near += 1
    per_script: dict[str, dict] = {}
    for script in SCRIPTS:
        cell = [r for r in rows if r["script"] == script]
        if not cell:
            continue
        if len(cell) < config.min_cell:
            per_script[script] = {"count": f"<{config.min_cell}"}
        else:
            per_script[script] = {
                "count": len(cell),
                "alert_rate": rate(sum(r["decision"] == "alert" for r in cell), len(cell)),
            }
    per_script = suppress_script_cells(per_script, count, config.min_cell)
    durations = sorted(r["duration_ms"] for r in rows)
    return {
        "run_context_hash": context_hash,
        "count": count,
        "alerts": alerts,
        "alert_rate": rate(alerts, count),
        "score_histogram": {"counts": counts, "no_candidates": no_candidates},
        "near_threshold": {"count": near, "rate": rate(near, count)},
        "per_script": per_script,
        "duration_ms": {
            "p50": percentile(durations, 50),
            "p95": percentile(durations, 95),
            "p99": percentile(durations, 99),
        },
    }


class MetricsRefused(AuditError):
    """The log failed verification. Nothing was written."""

    def __init__(self, message: str, kinds: frozenset[str] = frozenset()) -> None:
        super().__init__(message)
        self.kinds = kinds


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compute_metrics(
    audit_dir: Path,
    from_seq: int,
    to_seq: int,
    config: MetricsConfig | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> tuple[dict, Path]:
    """Measure the decision records with seq in [from_seq, to_seq], write the report, and return it with its path.

    Raises ValueError for a bad window, LogMissingError if the store is not there, and AuditError if the report for
    this window already exists or cannot be written."""
    for name, value in (("from_seq", from_seq), ("to_seq", to_seq)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a whole number of at least 0")
    if from_seq > to_seq:
        raise ValueError("from_seq must not be after to_seq")
    config = config or MetricsConfig.load()
    log = AuditLog.in_directory(audit_dir)
    log.headers.require_initialised()
    with log.headers.locked(create_directory=False):
        verification = _verify_locked(log)
        if verification.tampering:
            names = ", ".join(sorted({f.kind for f in verification.tampering}))
            raise MetricsRefused(
                f"refusing to measure: verify_chain found tampering-class failures ({names}). Metrics from a log "
                "that may have been edited would mean nothing. Nothing was written.",
                frozenset(f.kind for f in verification.tampering),
            )
        records = log.headers.read_all()
        if to_seq >= len(records):
            raise ValueError(f"the window ends at seq {to_seq} but the log's last seq is {len(records) - 1}")
        window = records[from_seq : to_seq + 1]
        decisions = [r for r in window if r.get("kind") == KIND_DECISION]
        rows = [
            {
                "context": r["run_context_hash"],
                "decision": r["decision"],
                "final_score": r["final_score"],
                "threshold": r["threshold"],
                "duration_ms": r["duration_ms"],
                "script": record_script(log, r),
            }
            for r in decisions
        ]
    by_context: dict[str, list[dict]] = defaultdict(list)
    for row in rows:  # dict order is first appearance in the chain
        by_context[row["context"]].append(row)
    groups = [group_metrics(group, config, context) for context, group in by_context.items()]
    any_suppressed = any(not isinstance(g["count"], int) for g in groups)
    crash = Counter(f.kind for f in verification.failures)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": clock().astimezone(timezone.utc).isoformat(timespec="seconds"),
        "window": {
            "from_seq": from_seq,
            "to_seq": to_seq,
            "records": len(window),
            "decisions": "withheld" if any_suppressed else len(decisions),
            "first_timestamp_utc": window[0]["timestamp_utc"] if window else None,
            "last_timestamp_utc": window[-1]["timestamp_utc"] if window else None,
        },
        "config": {
            "near_threshold_band": config.near_threshold_band,
            "min_cell": config.min_cell,
            "file_sha256": config.source_sha256,  # null if the config was built in code rather than loaded
        },
        "integrity": {"tampering_findings": 0, "crash_findings": dict(sorted(crash.items()))},
        "histogram": {"bins": BINS, "edges": [i / BINS for i in range(BINS + 1)]},
        "groups": groups,
    }
    return report, write_report(Path(audit_dir), report, from_seq, to_seq)


def write_report(audit_dir: Path, report: dict, from_seq: int, to_seq: int) -> Path:
    path = audit_dir / "monitoring" / f"{from_seq}-{to_seq}.json"
    try:
        path.parent.mkdir(exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise AuditError(f"a report for window {from_seq}-{to_seq} already exists; it is never overwritten") from None
    except OSError as exc:
        raise AuditError(f"could not create the report {path}: {exc}") from exc
    try:
        os.write(fd, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
    except OSError as exc:
        raise AuditError(f"could not write the report {path}: {exc}") from exc
    finally:
        os.close(fd)
    return path
