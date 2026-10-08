"""Drift: compare a monitoring report with a frozen baseline measured under the same run context.

    result = compare_to_baseline(report, baseline)

A baseline is a monitoring report (src/audit_metrics.py) of a seeded sample of the frozen hold-out, screened through
the audited path, saved as baseline/<context_hash>.json (src/make_baseline.py). It applies only to the run context it
was measured under: a different threshold, code, list or library is a different engine, and comparing live traffic
with a baseline from another engine would blame the traffic for a change the engine made. A report whose context hash
differs is refused, with a field-level diff when the baseline file carries its context.

What is compared, for the group of the report that has the baseline's context:
  score PSI           population stability index over the 20 fixed score bins
  input-mix PSI       PSI over the share of decisions per script (plus a "suppressed" bucket for hidden cells)
  per-script score PSI  for scripts that are visible (unsuppressed) in both, with enough decisions in both
  per-script alert rate versus baseline: both values, the absolute difference and the ratio
  near-threshold rate versus baseline (absolute difference)
  no-candidate rate (information only)

PSI = sum over bins of (live share - baseline share) * ln(live share / baseline share), where a share below
psi_epsilon (an empty bin) is replaced by psi_epsilon so the logarithm exists. Shares are not renormalised afterwards.

Limits are the larger of a convention and the sampling null. PSI has a noise floor: even if nothing changed, two samples
of a few hundred decisions give a PSI above zero, and over 20 bins that floor reaches the conventional 0.1 warn limit
(measured on this project's baseline: two random halves of the same 400 Latin decisions have a median PSI of 0.10). The
null therefore has to include the noise of BOTH samples, the baseline's as well as the live one's. For every PSI
comparison the baseline and live histograms are pooled into one set of proportions (what the two samples would share if
nothing had changed); then, 1,000 times with a fixed seed, n_base and n_live decisions are drawn multinomially from the
pooled proportions and the PSI between the two draws is taken. Then
    warn   when PSI > max(psi_warn, null p95)       (psi_warn is 0.1)
    alert  when PSI > max(psi_alert, null p99)      (psi_alert is 0.25)
The floors 0.1 and 0.25 are an industry convention from credit scorecard monitoring, not a value derived from this
matcher or this data, and they assume large samples. Percentiles are nearest rank. Every comparison reports its PSI, n,
the null percentiles and the limits used.

Multiplicity. A report makes k PSI comparisons (the score, the script mix, and one per script with enough decisions),
and the overall status is the worst of them, so at 5% per comparison a clean batch would warn far more often than 5% of
the time. The overall status therefore uses Bonferroni-corrected percentiles: the warn limit is the null's
1 - alpha_warn/k percentile (alpha_warn 0.05) and the alert limit its 1 - alpha_alert/k percentile (alpha_alert 0.01),
each still no lower than its convention floor. Per-comparison statuses keep the uncorrected percentiles and are shown
beside the corrected ones; k and the corrected percentiles are reported. The alert-rate and near-threshold comparisons
are not PSI and keep fixed limits (judgement): the alert-rate status needs both the ratio (the larger of ratio and its
inverse) and the absolute difference past their limits, so a rate moving from 1% to 2% is not an alert, and at a limit
is past it.

Durations are machine-dependent, so they are shown beside the baseline's as information and never compared or given a
status.

Thin data gets no status. If the live or baseline group, or a script cell, has fewer than min_psi_count decisions the
comparison is reported as "insufficient_data" with no PSI.

What live monitoring can and cannot see. It tracks per-script alert rates and score distributions against a hold-out
baseline. It cannot track false-alert burden (the share of clean customers who are alerted, the 22.6% Arabic result
of Stage 3): that needs labels saying which alerts were wrong, and live traffic has none. A drift in alert rate is
a signal to look, not a measurement of harm.
"""

import json
import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audit_metrics import CONFIG_PATH, SCRIPTS
from run_context import SCORE_PATH_SOURCES, diff_contexts

BASELINE_SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "baseline"
SEVERITY = {"ok": 0, "warn": 1, "alert": 2}


class BaselineMismatch(ValueError):
    """The baseline was measured under a different decision context (or the report has no group for it)."""


@dataclass(frozen=True)
class DriftConfig:
    min_psi_count: int = 200
    psi_epsilon: float = 0.0001
    psi_warn: float = 0.1  # convention floors: the limit is the larger of these and the sampling null
    psi_alert: float = 0.25
    psi_null_draws: int = 1000
    psi_null_seed: int = 11
    psi_warn_percentile: int = 95
    psi_alert_percentile: int = 99
    psi_alpha_warn: float = 0.05  # overall status: Bonferroni, warn at the 1 - alpha/k percentile of the null
    psi_alpha_alert: float = 0.01
    alert_rate_ratio_warn: float = 1.5
    alert_rate_ratio_alert: float = 2.0
    alert_rate_abs_diff_warn: float = 0.03
    alert_rate_abs_diff_alert: float = 0.06
    near_threshold_diff_warn: float = 0.03
    near_threshold_diff_alert: float = 0.06

    def __post_init__(self) -> None:
        if not isinstance(self.min_psi_count, int) or isinstance(self.min_psi_count, bool) or self.min_psi_count < 1:
            raise ValueError("min_psi_count must be a whole number of at least 1")
        if not 0 < self.psi_epsilon < 0.01:
            raise ValueError("psi_epsilon must be a small positive number")
        if not isinstance(self.psi_null_draws, int) or self.psi_null_draws < 100:
            raise ValueError("psi_null_draws must be a whole number of at least 100")
        if not 0 < self.psi_alpha_alert <= self.psi_alpha_warn < 0.5:
            raise ValueError("need 0 < psi_alpha_alert <= psi_alpha_warn < 0.5")
        if not 50 < self.psi_warn_percentile < self.psi_alert_percentile < 100:
            raise ValueError("need 50 < psi_warn_percentile < psi_alert_percentile < 100")
        for name in ("psi", "alert_rate_ratio", "alert_rate_abs_diff", "near_threshold_diff"):
            warn, alert = getattr(self, f"{name}_warn"), getattr(self, f"{name}_alert")
            if not 0 < warn <= alert:
                raise ValueError(f"{name}: need 0 < warn <= alert")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "DriftConfig":
        """The drift settings from the monitoring config file, or the defaults if the file does not exist."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})


def status_for(value: float, warn: float, alert: float) -> str:
    """ok below warn, warn from warn up to (not including) alert, alert from alert up. A value at a limit is past it."""
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def psi(baseline: list[int], live: list[int], epsilon: float = 0.0001) -> float | None:
    """Population stability index of two count vectors over the same bins; None if either is empty."""
    n_base, n_live = sum(baseline), sum(live)
    if not n_base or not n_live:
        return None
    total = 0.0
    for base_n, live_n in zip(baseline, live, strict=True):
        p, q = max(base_n / n_base, epsilon), max(live_n / n_live, epsilon)
        total += (q - p) * math.log(q / p)
    return total


def insufficient(reason: str, n: int | None = None) -> dict:
    return {"value": None, "status": "insufficient_data", "reason": reason, "n": n}


def psi_status(value: float, warn_limit: float, alert_limit: float) -> str:
    """alert above the alert limit, warn above the warn limit, otherwise ok. At a limit is not above it."""
    return "alert" if value > alert_limit else "warn" if value > warn_limit else "ok"


def null_distribution(baseline: list[int], live: list[int], config: DriftConfig, name: str) -> np.ndarray:
    """Sorted PSI values of the pooled two-sample null: the two histograms are pooled into one set of proportions, and
    config.psi_null_draws times n_base and n_live decisions are drawn multinomially from it and their PSI is taken. The
    seed is the configured one plus a CRC of the comparison's name, so each comparison has its own fixed stream."""
    base_counts, live_counts = np.asarray(baseline, dtype=float), np.asarray(live, dtype=float)
    pooled = base_counts + live_counts
    shares = pooled / pooled.sum()
    n_base, n_live = int(base_counts.sum()), int(live_counts.sum())
    rng = np.random.default_rng(config.psi_null_seed + zlib.crc32(name.encode()))
    p = np.maximum(rng.multinomial(n_base, shares, size=config.psi_null_draws) / n_base, config.psi_epsilon)
    q = np.maximum(rng.multinomial(n_live, shares, size=config.psi_null_draws) / n_live, config.psi_epsilon)
    return np.sort(((q - p) * np.log(q / p)).sum(axis=1))


def rank(values: np.ndarray, fraction: float) -> float:
    """Nearest-rank percentile: the value at position ceil(fraction * n) of the sorted values."""
    return float(values[min(max(math.ceil(fraction * len(values)) - 1, 0), len(values) - 1)])


def null_psi(baseline: list[int], live: list[int], config: DriftConfig, name: str) -> tuple[float, float]:
    """(warn percentile, alert percentile) of the pooled two-sample null, 95th and 99th by default."""
    values = null_distribution(baseline, live, config, name)
    return rank(values, config.psi_warn_percentile / 100), rank(values, config.psi_alert_percentile / 100)


def corrected_fractions(k: int, config: DriftConfig) -> tuple[float, float]:
    """The Bonferroni percentiles for k comparisons, as fractions: 1 - alpha_warn/k and 1 - alpha_alert/k."""
    return 1 - config.psi_alpha_warn / k, 1 - config.psi_alpha_alert / k


def psi_enough(n_base: int, n_live: int, config: DriftConfig) -> bool:
    return n_base >= config.min_psi_count and n_live >= config.min_psi_count


def psi_result(
    baseline: list[int], live: list[int], config: DriftConfig, n_base: int, n_live: int, name: str, k: int | None = None
) -> dict:
    """One PSI comparison with its pooled two-sample null. n_base and n_live decide whether there is enough data. With k
    (the number of PSI comparisons in the report) the Bonferroni-corrected limits and status are added."""
    n_draw = sum(live)
    if not psi_enough(n_base, n_live, config):
        return insufficient(f"{min(n_base, n_live)} decisions, fewer than min_psi_count {config.min_psi_count}", n_draw)
    value = psi(baseline, live, config.psi_epsilon)
    if value is None:
        return insufficient("no scored decisions", n_draw)
    values = null_distribution(baseline, live, config, name)
    p95, p99 = rank(values, config.psi_warn_percentile / 100), rank(values, config.psi_alert_percentile / 100)
    warn_limit, alert_limit = max(config.psi_warn, p95), max(config.psi_alert, p99)
    result = {
        "value": value,
        "status": psi_status(value, warn_limit, alert_limit),
        "n": n_draw,
        "n_baseline": sum(baseline),
        "null_p95": p95,
        "null_p99": p99,
        "warn_limit": warn_limit,
        "alert_limit": alert_limit,
    }
    if k:
        warn_fraction, alert_fraction = corrected_fractions(k, config)
        corrected_warn = max(config.psi_warn, rank(values, warn_fraction))
        corrected_alert = max(config.psi_alert, rank(values, alert_fraction))
        result.update(
            corrected_warn_limit=corrected_warn,
            corrected_alert_limit=corrected_alert,
            corrected_status=psi_status(value, corrected_warn, corrected_alert),
        )
    return result


def script_counts(group: Mapping) -> dict[str, int]:
    """Visible per-script counts of a group, plus whatever is hidden as one "suppressed" bucket."""
    visible = {k: v["count"] for k, v in group["per_script"].items() if isinstance(v["count"], int)}
    hidden = group["count"] - sum(visible.values())
    return {**visible, "suppressed": hidden} if hidden else visible


def worst(statuses: list[str]) -> str:
    known = [s for s in statuses if s in SEVERITY]
    return max(known, key=SEVERITY.__getitem__) if known else "insufficient_data"


def compare_groups(base: Mapping, live: Mapping, config: DriftConfig) -> dict:
    """Compare a live group with the baseline group of the same context."""
    if not isinstance(base["count"], int) or not isinstance(live["count"], int):
        reason = "a group is suppressed (fewer decisions than min_cell)"
        return {"overall": "insufficient_data", "reason": reason}
    n_base, n_live = base["count"], live["count"]
    thin = n_base < config.min_psi_count or n_live < config.min_psi_count
    thin_reason = f"{min(n_base, n_live)} decisions, fewer than min_psi_count {config.min_psi_count}"

    # Phase 1: which PSI comparisons will have enough data. k of them are made, and k sets the multiplicity correction.
    visible = {}
    for script in SCRIPTS:
        b_cell, l_cell = base["per_script"].get(script), live["per_script"].get(script)
        if b_cell and l_cell and isinstance(b_cell["count"], int) and isinstance(l_cell["count"], int):
            visible[script] = (b_cell, l_cell)
    k = 0
    if psi_enough(n_base, n_live, config):
        k += 2  # the score and the script mix
    k += sum(psi_enough(b_cell["count"], l_cell["count"], config) for b_cell, l_cell in visible.values())
    k = k or None

    base_hist, live_hist = base["score_histogram"]["counts"], live["score_histogram"]["counts"]
    score = psi_result(base_hist, live_hist, config, n_base, n_live, "score", k)

    cats = sorted(set(script_counts(base)) | set(script_counts(live)))
    base_mix, live_mix = script_counts(base), script_counts(live)
    mix = psi_result(
        [base_mix.get(c, 0) for c in cats], [live_mix.get(c, 0) for c in cats], config, n_base, n_live, "input_mix", k
    )
    mix["categories"] = cats

    per_script, skipped = {}, {}
    for script in SCRIPTS:
        b_cell, l_cell = base["per_script"].get(script), live["per_script"].get(script)
        if (
            b_cell is None
            or l_cell is None
            or not isinstance(b_cell["count"], int)
            or not isinstance(l_cell["count"], int)
        ):
            if b_cell is not None or l_cell is not None:
                skipped[script] = "not visible (suppressed, withheld or absent) in both"
            continue
        entry: dict = {"count_baseline": b_cell["count"], "count_live": l_cell["count"]}
        cell_thin = min(b_cell["count"], l_cell["count"]) < config.min_psi_count
        entry["score_psi"] = psi_result(
            b_cell["score_histogram"]["counts"],
            l_cell["score_histogram"]["counts"],
            config,
            b_cell["count"],
            l_cell["count"],
            f"script:{script}",
            k,
        )
        entry["alert_rate"] = alert_rate_result(b_cell["alert_rate"], l_cell["alert_rate"], config, cell_thin)
        per_script[script] = entry

    near = near_threshold_result(base, live, config, thin)
    no_cand_base = base["score_histogram"]["no_candidates"] / n_base
    no_cand_live = live["score_histogram"]["no_candidates"] / n_live

    def counted(item: dict) -> str:
        return item.get("corrected_status", item["status"])  # the overall status uses the corrected PSI statuses

    statuses = [counted(score), counted(mix), near["status"]]
    uncorrected = [score["status"], mix["status"], near["status"]]
    for entry in per_script.values():
        statuses += [counted(entry["score_psi"]), entry["alert_rate"]["status"]]
        uncorrected += [entry["score_psi"]["status"], entry["alert_rate"]["status"]]
    multiplicity = None
    if k:
        warn_fraction, alert_fraction = corrected_fractions(k, config)
        multiplicity = {
            "k": k,
            "alpha_warn": config.psi_alpha_warn,
            "alpha_alert": config.psi_alpha_alert,
            "warn_percentile": round(100 * warn_fraction, 4),
            "alert_percentile": round(100 * alert_fraction, 4),
            "why": "the overall status is the worst of k PSI comparisons, so it uses Bonferroni-corrected percentiles",
        }
    return {
        "overall": worst(statuses),
        "overall_uncorrected": worst(uncorrected),
        "multiplicity": multiplicity,
        "count_baseline": n_base,
        "count_live": n_live,
        "score_psi": score,
        "input_mix_psi": mix,
        "per_script": per_script,
        "skipped_scripts": skipped,
        "near_threshold": near,
        "no_candidate_rate": {"baseline": no_cand_base, "live": no_cand_live, "abs_diff": no_cand_live - no_cand_base},
        "note_thin_data": thin_reason if thin else None,
        "duration_ms": {
            "informational": True,
            "note": "machine-dependent: shown, never compared and never given a status",
            "baseline": base.get("duration_ms"),
            "live": live.get("duration_ms"),
        },
    }


def alert_rate_result(base_rate: float, live_rate: float, config: DriftConfig, thin: bool) -> dict:
    diff = live_rate - base_rate
    ratio = live_rate / base_rate if base_rate > 0 else None
    result = {"baseline": base_rate, "live": live_rate, "abs_diff": diff, "ratio": ratio}
    if thin:
        return {**result, "status": "insufficient_data", "reason": "fewer decisions than min_psi_count"}
    if ratio is None:
        ratio_dev = math.inf if live_rate > 0 else 1.0
    elif ratio == 0:
        ratio_dev = math.inf
    else:
        ratio_dev = round(max(ratio, 1 / ratio), 9)  # rounded so a value at a limit is past it despite float noise
    magnitude = round(abs(diff), 9)
    if ratio_dev >= config.alert_rate_ratio_alert and magnitude >= config.alert_rate_abs_diff_alert:
        status = "alert"
    elif ratio_dev >= config.alert_rate_ratio_warn and magnitude >= config.alert_rate_abs_diff_warn:
        status = "warn"
    else:
        status = "ok"
    return {**result, "status": status}


def near_threshold_result(base: Mapping, live: Mapping, config: DriftConfig, thin: bool) -> dict:
    base_rate, live_rate = base["near_threshold"]["rate"], live["near_threshold"]["rate"]
    result = {"baseline_rate": base_rate, "live_rate": live_rate, "abs_diff": live_rate - base_rate}
    if thin:
        return {**result, "status": "insufficient_data", "reason": "fewer decisions than min_psi_count"}
    return {
        **result,
        "status": status_for(
            abs(live_rate - base_rate), config.near_threshold_diff_warn, config.near_threshold_diff_alert
        ),
    }


def decision_dict(context: Mapping) -> dict:
    """The decision-relevant part of a stored context dict (see RunContext.decision_context)."""
    return {
        "config": context.get("config"),
        "engine_settings": context.get("engine_settings"),
        "snapshot_id": context.get("snapshot_id"),
        "snapshot_sha256": context.get("snapshot_sha256"),
        "rarity_table_sha256": context.get("rarity_table_sha256"),
        "scoring_sources": {k: v for k, v in (context.get("artefacts") or {}).items() if k in SCORE_PATH_SOURCES},
        "libraries": context.get("libraries"),
        "python_version": context.get("python_version"),
        "unidata_version": context.get("unidata_version"),
    }


def compare_to_baseline(
    report: Mapping, baseline: Mapping, config: DriftConfig | None = None, live_contexts: Mapping | None = None
) -> dict:
    """Compare a monitoring report with a baseline. The match is on decision_context_hash: the same configuration,
    list snapshot, scoring code and libraries, whatever the git commit or audit schema. Raises BaselineMismatch if the
    report has no group with the baseline's decision context. If several groups match (the same decisions made from
    different commits) the one with the most decisions is compared and the others are named in
    `other_matching_contexts`. `live_contexts` (run context hash -> context dict, such as the audit store's contexts/
    files) lets a refusal say which decision fields differ."""
    config = config or DriftConfig.load()
    wanted = baseline["decision_context_hash"]
    matches = [g for g in report["groups"] if g.get("decision_context_hash") == wanted]
    if not matches:
        have = [g.get("decision_context_hash") for g in report["groups"]]
        shown = ", ".join(str(h)[:12] + "..." for h in have) if have else "no decisions"
        message = (
            f"refusing to compare: the baseline was measured under decision context {wanted[:12]}..., and a baseline "
            f"only applies to the decision context it was measured under (the configuration, list snapshot, scoring "
            f"code and libraries that decide decisions). This report has {shown}."
        )
        stored = baseline.get("context")
        if stored and live_contexts:
            for group in report["groups"]:
                live_ctx = live_contexts.get(group["run_context_hash"])
                if live_ctx:
                    diff = diff_contexts(decision_dict(stored), decision_dict(live_ctx))
                    message += f"\n  decision fields that differ in {group['run_context_hash'][:12]}...: " + "; ".join(
                        diff
                    )
        raise BaselineMismatch(message)
    live = max(matches, key=lambda g: g["count"] if isinstance(g["count"], int) else -1)
    base = next(g for g in baseline["report"]["groups"] if g.get("decision_context_hash") == wanted)
    result = compare_groups(base, live, config)
    result["decision_context_hash"] = wanted
    result["run_context_hash"] = live["run_context_hash"]
    result["baseline_run_context_hash"] = baseline["run_context_hash"]
    result["other_matching_contexts"] = [g["run_context_hash"] for g in matches if g is not live]
    return result


def load_baseline(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"{path}: not a baseline file this version understands")
    return data


def save_baseline(report: Mapping, context: Mapping, sample: Mapping, directory: Path = BASELINE_DIR) -> Path:
    """Write baseline/<decision_context_hash>.json (exclusive create: a frozen baseline is never silently replaced).
    `context` is the stored run context dict of the engine that made the sample."""
    (group,) = report["groups"]
    decision_hash = group["decision_context_hash"]
    if not decision_hash:
        raise ValueError("the report's group has no decision_context_hash (its context file is missing)")
    path = directory / f"{decision_hash}.json"
    body = {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "decision_context_hash": decision_hash,
        "run_context_hash": group["run_context_hash"],
        "context": dict(context),
        "sample": dict(sample),
        "report": dict(report),
    }
    directory.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path
