"""The monitoring report in plain markdown: <audit_dir>/monitoring/<from>-<to>.md.

Run:
    python src/screen_cli.py report --from 0 --to 999 --audit-dir DIR

This module assembles outputs that already exist and calculates nothing of its own: verify_chain's status and finding
counts (audit_verify), the metrics report for the window (audit_metrics: read if its JSON is there, else written by
compute_metrics), the drift comparison with the matching baseline (audit_drift), and the latest gate report in reports/
(list_gate). It adds no figure that those do not give. Aggregate numbers only: no names and no request IDs, because none
of its sources holds them.

Exit status is the worst status: 0 ok, 3 insufficient data, 10 warn, 20 alert, 2 refused. Refused means the log failed
verification with tampering-class findings, or no baseline matches any run context in the window; nothing is written.
The gate result is shown for information and does not change the exit status: it belongs to a list update, not to the
traffic in this window.

The plain-writing rules are enforced by tests: no em dash, no horizontal rule, only headings, paragraphs and tables.
"""

import json
import os
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from audit import AuditError, AuditLog
from audit_drift import BaselineMismatch, DriftConfig, compare_to_baseline, load_baseline
from audit_metrics import MetricsConfig, MetricsRefused, compute_metrics, utc_now
from audit_verify import verify_chain

REPO = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO / "baseline"
REPORT_DIR = REPO / "reports"
GATE_GLOB = "stage4_list_update_gate_*.json"

EXIT_BY_STATUS = {"ok": 0, "insufficient_data": 3, "warn": 10, "alert": 20}
EXIT_REFUSED = 2

# The two stated gaps. The README's Stage 4 section says the same things in the same words; a test checks it.
GAP_CONSTRUCTED = (
    "Monitoring is demonstrated on constructed batches, not production traffic. The baseline and the stable, mix-shift "
    "and degraded batches are samples of a spent hold-out, so how often a live deployment would warn or alert is "
    "unknown."
)
GAP_NO_LABELS = (
    "There are no analyst dispositions, so live precision and recall cannot be measured, and neither can the "
    "false-alert burden (the 22.6% Arabic result of Stage 3): live traffic has no labels saying which alerts were "
    "wrong. A change in an alert rate is a reason to look, not a measurement of harm."
)
LIMITS_PARAGRAPHS = (
    GAP_CONSTRUCTED,
    GAP_NO_LABELS,
    "The deduplication model is outside the decision path and is not monitored.",
    "Duration figures depend on the machine and are shown for information only.",
)


class ReportRefused(AuditError):
    """The report cannot be made: the log failed verification, or no baseline matches. Nothing was written."""


# ---------- small formatting helpers ----------


def short(value: str | None) -> str:
    return "none" if not value else value[:12] + "..."


def num(value: object, places: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


# ---------- sections ----------


def integrity_section(result: object) -> list[str]:
    crash = Counter(f.kind for f in result.crashes)  # type: ignore[attr-defined]
    lines = ["## Integrity", ""]
    status = "clean" if result.ok else "crash-class findings only"  # type: ignore[attr-defined]
    lines += table(
        ["Item", "Value"],
        [
            ["verify_chain", status],
            ["Records checked", result.records],  # type: ignore[attr-defined]
            ["Payloads checked", result.payloads_checked],  # type: ignore[attr-defined]
            ["Payloads purged (covered by a purge record)", result.payloads_purged],  # type: ignore[attr-defined]
            ["Tampering-class findings", len(result.tampering)],  # type: ignore[attr-defined]
        ],
    )
    if crash:
        lines += ["Crash-class findings by type:", ""]
        lines += table(["Finding", "Count"], [[k, v] for k, v in sorted(crash.items())])
    else:
        lines += ["No crash-class findings.", ""]
    return lines


def window_section(report: dict) -> list[str]:
    w = report["window"]
    lines = ["## Window and run contexts", ""]
    lines += table(
        ["Item", "Value"],
        [
            ["First record (seq)", w["from_seq"]],
            ["Last record (seq)", w["to_seq"]],
            ["Records", w["records"]],
            ["Decisions", w["decisions"]],
            ["First timestamp (UTC)", w["first_timestamp_utc"] or "none"],
            ["Last timestamp (UTC)", w["last_timestamp_utc"] or "none"],
            ["Metrics generated (UTC)", report["generated_at_utc"]],
        ],
    )
    rows = [[short(g["run_context_hash"]), short(g.get("decision_context_hash")), g["count"]] for g in report["groups"]]
    lines += ["Run context groups. Metrics are never mixed across run contexts.", ""]
    lines += table(["Run context", "Decision context", "Decisions"], rows)
    return lines


def group_section(group: dict, report: dict) -> list[str]:
    lines = [f"### Run context {short(group['run_context_hash'])}", ""]
    if not isinstance(group["count"], int):
        return [
            *lines,
            f"Fewer decisions than min_cell ({report['config']['min_cell']}): {group['count']}. Nothing else is shown.",
            "",
        ]
    nt = group["near_threshold"]
    d = group["duration_ms"]
    lines += table(
        ["Item", "Value"],
        [
            ["Decisions", group["count"]],
            ["Alerts", group["alerts"]],
            ["Alert rate", num(group["alert_rate"])],
            ["Near-threshold decisions", nt["count"]],
            ["Near-threshold rate", num(nt["rate"])],
            ["No candidates", group["score_histogram"]["no_candidates"]],
            ["Duration p50, p95, p99 in ms (information only)", f"{d['p50']}, {d['p95']}, {d['p99']}"],
        ],
    )
    counts = group["score_histogram"]["counts"]
    bins = len(counts)
    edges = [f"{i / bins:.2f} to {(i + 1) / bins:.2f}" for i in range(bins)]
    lines += ["Final score histogram (20 fixed bins):", ""]
    lines += table(["Score bin", "Decisions"], [[e, c] for e, c in zip(edges, counts, strict=True)])
    rows = []
    for script, cell in group["per_script"].items():
        if isinstance(cell["count"], int):
            rows.append([script, cell["count"], num(cell["alert_rate"])])
        else:
            rows.append([script, cell["count"], "suppressed"])
    lines += ["By script. Cells below min_cell are suppressed as in the JSON report.", ""]
    lines += table(["Script", "Decisions", "Alert rate"], rows)
    return lines


def metrics_section(report: dict) -> list[str]:
    c = report["config"]
    lines = ["## Metrics", ""]
    lines += [
        f"Near-threshold band {c['near_threshold_band']}, min_cell {c['min_cell']}, "
        f"config file sha256 {short(c['file_sha256'])}.",
        "",
    ]
    for group in report["groups"]:
        lines += group_section(group, report)
    return lines


def psi_row(label: str, item: dict) -> list[object]:
    if item["value"] is None:
        return [label, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", item["status"], "n/a"]
    return [
        label,
        num(item["value"]),
        item["n"],
        item["n_baseline"],
        num(item["null_p95"]),
        num(item["null_p99"]),
        num(item["warn_limit"]),
        num(item["alert_limit"]),
        item["status"],
        item.get("corrected_status", "n/a"),
    ]


def drift_section(results: list[dict], missing: list[str]) -> list[str]:
    lines = ["## Drift against the baseline", ""]
    for result in results:
        lines += [f"### Decision context {short(result['decision_context_hash'])}", ""]
        lines += [f"Overall status: {result['overall']}.", ""]
        if "reason" in result:
            lines += [result["reason"], ""]
            continue
        m = result["multiplicity"]
        if m:
            lines += [
                f"The overall status is the worst of k = {m['k']} PSI comparisons and so uses Bonferroni-corrected "
                f"percentiles: {num(m['warn_percentile'], 4)} for warn (alpha {m['alpha_warn']}) and "
                f"{num(m['alert_percentile'], 4)} for alert (alpha {m['alpha_alert']}). Overall status without the "
                f"correction: {result['overall_uncorrected']}.",
                "",
            ]
        lines += [
            f"Live decisions {result['count_live']}, baseline decisions {result['count_baseline']}. "
            "Limits are the larger of the convention (0.1 and 0.25) and the pooled two-sample null percentile. "
            "Each status uses the uncorrected percentiles; the last column uses the corrected ones.",
            "",
        ]
        rows = [psi_row("Score", result["score_psi"]), psi_row("Input mix", result["input_mix_psi"])]
        rows += [psi_row(f"{s} score", e["score_psi"]) for s, e in result["per_script"].items()]
        lines += table(
            ["PSI comparison", "PSI", "n live", "n baseline", "Null p95", "Null p99", "Warn limit", "Alert limit",
             "Status", "Corrected status"],
            rows,
        )  # fmt: skip
        other = [["Near-threshold rate", num(result["near_threshold"]["baseline_rate"]),
                  num(result["near_threshold"]["live_rate"]), f"{result['near_threshold']['abs_diff']:+.4f}", "n/a",
                  result["near_threshold"]["status"]]]  # fmt: skip
        for script, e in result["per_script"].items():
            a = e["alert_rate"]
            other.append([f"{script} alert rate", num(a["baseline"]), num(a["live"]), f"{a['abs_diff']:+.4f}",
                          num(a["ratio"]), a["status"]])  # fmt: skip
        lines += table(["Rate comparison", "Baseline", "Live", "Difference", "Ratio", "Status"], other)
        for script, why in result["skipped_scripts"].items():
            lines += [f"{script} was not compared: {why}.", ""]
        if result["note_thin_data"]:
            lines += [f"Thin data: {result['note_thin_data']}.", ""]
        if result["other_matching_contexts"]:
            lines += [
                f"Other run contexts with the same decisions were not compared: "
                f"{len(result['other_matching_contexts'])}.",
                "",
            ]
    for context in missing:
        lines += [f"No baseline for decision context {short(context)}; that group was not compared.", ""]
    return lines


def latest_gate_report(reports_dir: Path) -> tuple[Path, dict] | None:
    """The gate report with the newest new-snapshot version (versions start with a timestamp), or None."""
    found = []
    for path in Path(reports_dir).glob(GATE_GLOB):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            found.append((data["new_snapshot"], path.name, path, data))
        except (OSError, ValueError, KeyError):
            continue
    if not found:
        return None
    _, _, path, data = max(found, key=lambda t: (t[0], t[1]))
    return path, data


def interval(metric: dict) -> str:
    low, high = metric["delta_ci95"]
    return f"{num(metric['delta'])} ({num(low)} to {num(high)})"


def gate_section(found: tuple[Path, dict] | None) -> list[str]:
    lines = ["## Latest list-update gate report", ""]
    if found is None:
        return [*lines, "No gate report was found in reports/.", ""]
    path, g = found
    label = g["label"]
    if label == "REVERSE-TIME CHURN TEST":
        lines += [
            "REVERSE-TIME CHURN TEST. This run took the current snapshot as the old list and an older snapshot as the "
            "new one. It is not a forecast of a future update.",
            "",
        ]
    p, m, r = g["provenance"], g["metrics_on_resolved_set"], g["resolution"]
    f1, ar = m["f1"], m["arabic_burden"]
    lines += table(
        ["Item", "Value"],
        [
            ["File", path.name],
            ["Label", label],
            ["Direction", g["direction"]],
            ["Old snapshot", g["old_snapshot"]],
            ["New snapshot", g["new_snapshot"]],
            ["Hold-out", p["holdout"]],
            ["Hold-out sha256", short(p["holdout_sha256"])],
            ["Engine", p["engine"]],
            ["Positives kept, remapped, dropped", f"{r['kept']}, {r['remapped']}, {r['dropped']}"],
            ["F1 old to new", f"{num(f1['old'])} to {num(f1['new'])}"],
            ["F1 delta (95% interval)", interval(f1)],
            ["Arabic burden old to new", f"{num(ar['old'])} to {num(ar['new'])}"],
            ["Arabic burden delta (95% interval)", interval(ar)],
            ["Gate", "PASS" if g["gate"]["passed"] else "FAIL"],
        ],
    )  # fmt: skip
    lines += ["Figures the headline numbers must not be confused with:", ""]
    lines += table(["Figure", "What it is"], [[k, v] for k, v in p["not_the_same_as"].items()])
    lines += ["The gate result is shown for information and does not change this report's exit status.", ""]
    return lines


def limits_section() -> list[str]:
    return ["## Limits of this report", "", *[line for p in LIMITS_PARAGRAPHS for line in (p, "")]]


# ---------- assembling ----------


def find_baselines(report: dict, baseline_dir: Path) -> tuple[list[dict], list[str]]:
    """The baseline for each distinct decision context in the report, and the contexts that have none."""
    baselines, missing = [], []
    for context in dict.fromkeys(g.get("decision_context_hash") for g in report["groups"]):
        path = Path(baseline_dir) / f"{context}.json"
        if context and path.exists():
            baselines.append(load_baseline(path))
        else:
            missing.append(context or "unknown")
    return baselines, missing


def metrics_for(
    audit_dir: Path, from_seq: int, to_seq: int, config: MetricsConfig | None, clock: Callable[[], datetime]
) -> dict:
    """The window's metrics report: read if its JSON exists, else made (and written) by compute_metrics."""
    path = Path(audit_dir) / "monitoring" / f"{from_seq}-{to_seq}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return compute_metrics(audit_dir, from_seq, to_seq, config, clock)[0]


def build_report(
    audit_dir: Path,
    from_seq: int,
    to_seq: int,
    *,
    baseline_dir: Path = BASELINE_DIR,
    reports_dir: Path = REPORT_DIR,
    metrics_config: MetricsConfig | None = None,
    drift_config: DriftConfig | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> tuple[str, int]:
    """The markdown and the exit status. Raises ReportRefused (tampering, or no matching baseline) and writes nothing
    in that case except what compute_metrics already writes for a window it accepts."""
    log = AuditLog.in_directory(audit_dir)
    log.headers.require_initialised()
    verification = verify_chain(log)
    if verification.tampering:
        names = ", ".join(sorted({f.kind for f in verification.tampering}))
        raise ReportRefused(
            f"refusing to report: verify_chain found tampering-class failures ({names}). Nothing was written."
        )
    try:
        report = metrics_for(audit_dir, from_seq, to_seq, metrics_config, clock)
    except MetricsRefused as exc:
        raise ReportRefused(str(exc)) from exc
    baselines, missing = find_baselines(report, baseline_dir)
    if not baselines:
        raise ReportRefused(
            f"refusing to report: no baseline for the decision context(s) in this window in {baseline_dir}. A baseline "
            "only applies to the decision context it was measured under."
        )
    results = []
    for baseline in baselines:
        try:
            results.append(compare_to_baseline(report, baseline, drift_config))
        except BaselineMismatch as exc:
            raise ReportRefused(str(exc)) from exc
    lines = [
        "# Monitoring report",
        "",
        "Aggregate figures only. No names and no request IDs appear in this report.",
        "",
        *integrity_section(verification),
        *window_section(report),
        *metrics_section(report),
        *drift_section(results, missing),
        *gate_section(latest_gate_report(reports_dir)),
        *limits_section(),
    ]
    status = max((r["overall"] for r in results), key=lambda s: EXIT_BY_STATUS[s])
    return "\n".join(lines).rstrip("\n") + "\n", EXIT_BY_STATUS[status]


def write_markdown(audit_dir: Path, text: str, from_seq: int, to_seq: int) -> Path:
    path = Path(audit_dir) / "monitoring" / f"{from_seq}-{to_seq}.md"
    try:
        path.parent.mkdir(exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise AuditError(f"a report for window {from_seq}-{to_seq} already exists; it is never overwritten") from None
    except OSError as exc:
        raise AuditError(f"could not create the report {path}: {exc}") from exc
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    except OSError as exc:
        raise AuditError(f"could not write the report {path}: {exc}") from exc
    finally:
        os.close(fd)
    return path
