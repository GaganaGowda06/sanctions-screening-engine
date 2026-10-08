"""Constructed batches for drift: a stable one, a script-mix shift, and degraded inputs, each compared with a baseline.

Run:
    python src/run_drift_batches.py --work-dir /tmp/drift_work --out reports/stage4_drift_batches.json

All of it is hold-out data screened through the audited path in fresh audit directories under --work-dir. This
demonstrates what the drift comparison does on batches we construct; it is not production traffic, and the numbers
say nothing about how a real stream would look. The batches:

  stable     a fresh sample (disjoint from the baseline sample) drawn the same way. Expected: every status ok.
  mix_shift  mostly Arabic-script queries, so the input mix moves a lot, but each script's queries are the same kind
             of query as in the baseline. Expected: input-mix PSI alert; within-script score PSI ok.
  degraded   a fresh sample with the birth date dropped and the name cut to its first token. Expected: score PSI and
             the near-threshold rate move (warn or alert).

The expected pattern is written down here before the numbers are looked at, and the report records both the pattern
and what was observed. If a batch does not behave as expected the report says so; the limits are not adjusted to fit.

A single stable batch cannot be asserted against a probabilistic limit: even with correct limits some clean batches will
warn. So the stable expectation is also checked over many batches: `repeated_stable` draws 50 seeded batches from the
hold-out queries that are not in the baseline sample (the pool is screened once through the audited path; each batch is
a seeded sample of it, measured with the same group metrics code and compared with the same baseline), and the share
whose overall status is warn or worse, and alert, are reported. The expectation is warn-or-worse at most 10% and alert
at most 2%. The original single stable batch is still run and reported.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from audit import AuditLog
from audit_drift import DriftConfig, compare_to_baseline, load_baseline, psi, save_baseline
from audit_metrics import MetricsConfig, group_metrics
from audited_screening import AuditedEngine
from make_baseline import (
    DEFAULT_SEED,
    HOLDOUT,
    Transform,
    load_queries,
    sample_indices,
    screen_queries,
    whole_log_metrics,
)
from profile_data import DEFAULT_INPUT

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "stable": "overall ok",
    "mix_shift": "input-mix PSI alert; every computed within-script score PSI ok",
    "degraded": "score PSI warn or alert; near-threshold rate warn or alert",
}


def script_mix(queries: list[dict], indices: list[int]) -> dict[str, int]:
    return dict(sorted(Counter(queries[i]["script"] for i in indices).items()))


def split_half_psi(counts: list[int], seed: int, reps: int = 400, epsilon: float = 0.0001) -> dict:
    """What "no real change" looks like: the PSI between two random halves of the same sample, repeated. A drift limit
    below this is crossed by sampling noise alone."""
    population = [i for i, c in enumerate(counts) for _ in range(c)]
    rng = random.Random(seed)
    half = len(population) // 2
    values = []
    for _ in range(reps):
        rng.shuffle(population)
        a, b = [0] * len(counts), [0] * len(counts)
        for x in population[:half]:
            a[x] += 1
        for x in population[half : 2 * half]:
            b[x] += 1
        values.append(psi(a, b, epsilon))
    values.sort()
    return {
        "decisions_per_half": half,
        "median": values[reps // 2],
        "p90": values[int(reps * 0.9)],
        "p95": values[int(reps * 0.95)],
    }


def noise_floor(group: dict, seed: int) -> dict:
    """Split-half PSI of the baseline group and of each of its visible per-script cells."""
    out = {"all": split_half_psi(group["score_histogram"]["counts"], seed)}
    for script, cell in group["per_script"].items():
        if isinstance(cell["count"], int):
            out[script] = split_half_psi(cell["score_histogram"]["counts"], seed)
    return out


def repeated_stable(
    list_path: Path,
    queries: list[dict],
    baseline_idx: list[int],
    baseline: dict,
    work_dir: Path,
    seed: int,
    batches: int = 50,
    batch_size: int = 800,
    drift_config: DriftConfig | None = None,
) -> dict:
    """Overall status of many seeded stable batches. The pool (hold-out queries outside the baseline sample) is screened
    once through the audited path; each batch is a seeded sample of it."""
    pool = [i for i in range(len(queries)) if i not in set(baseline_idx)]
    directory = work_dir / "pool_audit"
    AuditLog.create_new(directory)
    engine = AuditedEngine.from_file(list_path, directory)
    screen_queries(engine, queries, pool)
    records = [h for h in AuditLog.in_directory(directory).headers.read_all() if h["kind"] == "decision"]
    rows = [
        {
            "decision": h["decision"],
            "final_score": h["final_score"],
            "threshold": h["threshold"],
            "duration_ms": h["duration_ms"],
            "script": queries[i]["script"],
        }
        for h, i in zip(records, pool, strict=True)
    ]
    run_hash = engine.context_hash
    decision_hash = engine.run_context.decision_context_hash
    metrics_config = MetricsConfig.load()
    overall, uncorrected, ks = [], [], []
    comparisons = {"warn_or_worse": 0, "alert": 0, "total": 0}
    for b in range(batches):
        chosen = random.Random(f"{seed}-stable-{b}").sample(range(len(rows)), batch_size)
        group = group_metrics([rows[i] for i in chosen], metrics_config, run_hash, decision_hash)
        result = compare_to_baseline({"groups": [group]}, baseline, drift_config)
        overall.append(result["overall"])
        uncorrected.append(result["overall_uncorrected"])
        ks.append((result["multiplicity"] or {}).get("k"))
        items = [result["score_psi"], result["input_mix_psi"]] + [e["score_psi"] for e in result["per_script"].values()]
        for item in items:
            if item["value"] is not None:
                comparisons["total"] += 1
                comparisons["warn_or_worse"] += item["status"] in ("warn", "alert")
                comparisons["alert"] += item["status"] == "alert"

    def share(statuses: list[str], levels: tuple[str, ...]) -> float:
        return sum(s in levels for s in statuses) / len(statuses)

    return {
        "batches": batches,
        "batch_size": batch_size,
        "pool_size": len(pool),
        "expected": "overall warn-or-worse rate <= 10% and overall alert rate <= 2%",
        "overall_warn_or_worse_rate": share(overall, ("warn", "alert")),
        "overall_alert_rate": share(overall, ("alert",)),
        "overall_counts": dict(sorted(Counter(overall).items())),
        "uncorrected_overall_warn_or_worse_rate": share(uncorrected, ("warn", "alert")),
        "uncorrected_overall_alert_rate": share(uncorrected, ("alert",)),
        "k_values": dict(sorted(Counter(map(str, ks)).items())),
        "per_comparison_uncorrected": {
            **comparisons,
            "warn_or_worse_rate": comparisons["warn_or_worse"] / max(comparisons["total"], 1),
            "alert_rate": comparisons["alert"] / max(comparisons["total"], 1),
        },
    }


def degrade(query: dict) -> tuple[str, None, str | None]:
    """Birth date dropped, name cut to its first token, country kept."""
    return query["name"].split()[0], None, query["country"]


def pattern_check(name: str, result: dict) -> tuple[bool, dict]:
    """Does the result show the expected pattern? Also the numbers that decided it."""
    within = {s: e["score_psi"] for s, e in result["per_script"].items() if e["score_psi"]["value"] is not None}
    observed = {
        "overall": result["overall"],
        "score_psi": result["score_psi"],
        "input_mix_psi": result["input_mix_psi"],
        "within_script_score_psi": {s: {"value": v["value"], "status": v["status"]} for s, v in within.items()},
        "near_threshold": result["near_threshold"],
    }
    if name == "stable":
        return result["overall"] == "ok", observed
    if name == "mix_shift":
        ok = (
            result["input_mix_psi"]["status"] == "alert"
            and bool(within)
            and all(v["status"] == "ok" for v in within.values())
        )
        return ok, observed
    ok = result["score_psi"]["status"] in ("warn", "alert") and result["near_threshold"]["status"] in ("warn", "alert")
    return ok, observed


def run_scenario(
    list_path: Path, queries: list[dict], indices: list[int], directory: Path, transform: Transform | None = None
) -> dict:
    AuditLog.create_new(directory)
    engine = AuditedEngine.from_file(list_path, directory)
    screen_queries(engine, queries, indices, transform)
    return whole_log_metrics(directory)


def run_batches(
    list_path: Path,
    work_dir: Path,
    holdout: Path = HOLDOUT,
    baseline_size: int = 1500,
    stable_size: int = 800,
    shift_arabic: int = 400,
    shift_other: int = 400,
    degraded_size: int = 800,
    stable_repeats: int = 50,
    seed: int = DEFAULT_SEED,
    drift_config: DriftConfig | None = None,
) -> dict:
    work_dir = Path(work_dir)
    queries = load_queries(holdout)
    baseline_idx = sample_indices(queries, baseline_size, seed)
    fresh = frozenset(baseline_idx)
    arabic_pool = [i for i, q in enumerate(queries) if q["script"] == "Arabic" and i not in fresh]
    other_pool = [i for i, q in enumerate(queries) if q["script"] != "Arabic" and i not in fresh]
    rng = random.Random(seed + 1)
    shift_idx = rng.sample(arabic_pool, min(shift_arabic, len(arabic_pool))) + rng.sample(
        other_pool, min(shift_other, len(other_pool))
    )
    batches = {
        "stable": (sample_indices(queries, stable_size, seed + 2, fresh), None),
        "mix_shift": (shift_idx, None),
        "degraded": (sample_indices(queries, degraded_size, seed + 3, fresh), degrade),
    }

    base_report = run_scenario(list_path, queries, baseline_idx, work_dir / "baseline_audit")
    context_hash = base_report["groups"][0]["run_context_hash"]
    base_context = json.loads((work_dir / "baseline_audit" / "contexts" / f"{context_hash}.json").read_text())
    baseline_path = save_baseline(
        base_report,
        base_context,
        {"holdout": Path(holdout).name, "seed": seed, "size": baseline_size},
        work_dir / "baseline",
    )
    baseline = load_baseline(baseline_path)

    metrics = MetricsConfig.load()
    out = {
        "what_this_is": "constructed batches of hold-out data through the audited path; not production traffic",
        "run_context_hash": context_hash,
        "holdout": Path(holdout).name,
        "seed": seed,
        "baseline": {"size": baseline_size, "script_mix": script_mix(queries, baseline_idx)},
        "metrics_config": {"near_threshold_band": metrics.near_threshold_band, "min_cell": metrics.min_cell},
        "drift_config": (drift_config or DriftConfig.load()).__dict__,
        "noise_floor_of_the_baseline_sample": {
            "what_this_is": "PSI between two random halves of the same baseline sample (400 shuffles): the PSI that "
            "sampling noise alone produces at that sample size",
            **noise_floor(base_report["groups"][0], seed),
        },
        "scenarios": {},
    }
    out["repeated_stable"] = repeated_stable(
        list_path, queries, baseline_idx, baseline, work_dir, seed, stable_repeats, stable_size, drift_config
    )
    for name, (indices, transform) in batches.items():
        report = run_scenario(list_path, queries, indices, work_dir / f"{name}_audit", transform)
        result = compare_to_baseline(report, baseline, drift_config)
        ok, observed = pattern_check(name, result)
        out["scenarios"][name] = {
            "size": len(indices),
            "script_mix": script_mix(queries, indices),
            "expected": EXPECTED[name],
            "expected_pattern_seen": ok,
            "observed": observed,
            "result": result,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", required=True, type=Path, help="a NEW directory for the batches' audit stores")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "stage4_drift_batches.json")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    out = run_batches(args.input, args.work_dir, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for name, s in out["scenarios"].items():
        print(
            f"{name}: expected [{s['expected']}] -> {'seen' if s['expected_pattern_seen'] else 'NOT seen'}; "
            f"overall {s['result']['overall']}"
        )
    r = out["repeated_stable"]
    print(
        f"repeated stable ({r['batches']} batches of {r['batch_size']}): overall warn-or-worse "
        f"{r['overall_warn_or_worse_rate']:.0%}, alert {r['overall_alert_rate']:.0%} (expected <= 10% and <= 2%)"
    )
    print(f"wrote {args.out}")
    repeated_ok = r["overall_warn_or_worse_rate"] <= 0.10 and r["overall_alert_rate"] <= 0.02
    return 0 if repeated_ok and all(s["expected_pattern_seen"] for s in out["scenarios"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
