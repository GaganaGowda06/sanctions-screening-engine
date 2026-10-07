"""Tune the screening configuration on dev, report on test, and write the configuration file.

Step 1, scoring parameters on the deployed candidate pool. Step 3c tuned them on a pool that kept a
candidate if any blend scored it at least 0.2, which the deployed engine does not do. Here each
blend is scored on its own, so the pool is the one the engine would really keep. Skeleton floor and
lift, the blend of character ratio and rarity-weighted score, and one global alert threshold are
chosen for the best dev F1.

Step 2, per-script thresholds for Latin, Cyrillic and Arabic queries. For each script, on dev:
  - recall must stay at or above that script's recall at the global threshold, and
  - within that, take the threshold with the lowest false positive rate (ties go to the lower
    threshold, which keeps more recall at the same false positive rate).
The grid stops at 0.99, so no script is left alerting only on a perfect score. "Other" scripts
(Han, kana, Hebrew, Thai and the rest) use the global threshold: their sample is too small to tune.

Everything is reported on the step 3c test queries, plus "other" negatives and same-script "other"
positives that step 3c did not have. The final hold-out is not read or scored. Its entities are
regenerated in memory only, to keep new queries away from them, and checked against the frozen checksum.

Run:
    python src/evaluate_screening.py
Writes reports/stage2_screening.json and src/screening_config.json.
"""

import argparse
import hashlib
import itertools
import json
import platform
import random
import sys
import time
from pathlib import Path

import evaluate_scoring as ev
from entities import load_named_entities
from evaluate_candidates import percentile
from profile_data import DEFAULT_INPUT
from scoring import Screener
from screening import CONFIG_PATH, SCRIPTS, ScreeningConfig, ScreeningEngine, query_script

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "stage2_screening.json"
BLENDS = (0.0, 0.5, 1.0)
TUNED_SCRIPTS = ("Latin", "Cyrillic", "Arabic")
EXTRA_PLAN = {"other_same_script": [("dev", 100), ("test", 200)]}
# Step 4 as first reported (floor 0.4, lift 0.3, blend 1.0, one threshold of 0.87), on the deployed pool.
STEP_4_BASELINE = {"recall": 0.7045, "precision": 0.8182, "false_positive_rate": 0.079}


def run_queries(screener: Screener, queries: list[dict], blend: float) -> list[float]:
    """Score components for each query with one blend, so the candidate pool is the deployed one."""
    key, latencies = f"comps_{blend}", []
    for q in queries:
        start = time.perf_counter()
        q[key] = screener.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=(blend,))
        latencies.append(time.perf_counter() - start)
    return latencies


def view(queries: list[dict], blend: float) -> list[dict]:
    """The queries with the components of one blend in place, for ev.final_scores."""
    return [{**q, "comps": q[f"comps_{blend}"]} for q in queries]


def recall_at(pos_rows: list[tuple], script: str, threshold: float) -> float:
    rows = [r for r in pos_rows if r[0] in ev.HEADLINE_GROUPS and r[5] == script]
    return sum(r[2] is not None and r[2] >= threshold for r in rows) / len(rows) if rows else 0.0


def fpr_at(neg_rows: list[tuple], script: str, threshold: float) -> float:
    rows = [r for r in neg_rows if r[5] == script]
    return sum(r[3] >= threshold for r in rows) / len(rows) if rows else 0.0


def choose_script_thresholds(
    pos_rows: list[tuple], neg_rows: list[tuple], global_threshold: float
) -> dict[str, float]:
    """Per script: lowest false positive rate with recall no lower than at the global threshold.

    "Other" gets the global threshold. A script with no positives keeps the global threshold too.
    """
    chosen = dict.fromkeys(SCRIPTS, global_threshold)
    for script in TUNED_SCRIPTS:
        if not any(r[0] in ev.HEADLINE_GROUPS and r[5] == script for r in pos_rows):
            continue
        floor_recall = recall_at(pos_rows, script, global_threshold)
        feasible = [t for t in ev.THRESHOLDS if recall_at(pos_rows, script, t) >= floor_recall]
        lowest = min(fpr_at(neg_rows, script, t) for t in feasible)
        chosen[script] = next(t for t in feasible if fpr_at(neg_rows, script, t) == lowest)
    return chosen


def spread(fpr_by_script: dict[str, float | None], scripts: tuple[str, ...]) -> float:
    values = [fpr_by_script[s] for s in scripts if fpr_by_script[s] is not None]
    return round(max(values) - min(values), 4) if values else 0.0


def build_query_sets(entities: list[dict], targets: list[dict]) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """The step 3c dev and test queries plus the step 4 additions, each with its query script.

    Returns the positives, the negatives, and the names each positive's entity keeps in the list. Never reads
    the final hold-out: its entities are regenerated in memory, only to keep new queries away from them.
    """
    persons = [e for e in targets if e["schema"] == "Person"]
    dob_pool = [d for e in persons for d in e["birth_dates"][:1]]
    country_pool = [c for e in persons for c in e["countries"][:1]]

    # Step 3c dev and test queries, unchanged.
    rng = random.Random(ev.SEED)
    positives, kept, near, synthetic, used = ev.build_dev_test_queries(entities, targets, dob_pool, country_pool, rng)

    # Keep new queries off the final hold-out's entities. Regenerated in memory, never read from disk.
    final_queries, final_kept = ev.build_final_holdout(entities, targets, dob_pool, country_pool, used)
    frozen = (ROOT / "holdout" / "final_holdout.sha256").read_text().strip()
    if hashlib.sha256(ev.holdout_text(final_queries, final_kept).encode()).hexdigest() != frozen:
        sys.exit("The regenerated final hold-out does not match the frozen checksum; refusing to continue.")
    reserved = used | {q.get("true_id") or q.get("entity") for q in final_queries}

    # New for step 4, on their own random stream: same-script "other" positives, extra and "other" negatives.
    extra_rng = random.Random(f"{ev.SEED}-extra")
    extra_pos, extra_kept = ev.build_positives(
        targets, extra_rng, EXTRA_PLAN, exclude=reserved, key="{seed}-extra-{group}"
    )
    extra_neg = ev.build_synthetic(entities, targets, dob_pool, country_pool, extra_rng, ("dev_extra", "test_other"))
    for q in extra_neg:
        if q["split"] == "test_other":
            q["split"] = "test"

    all_positives = positives + extra_pos
    negatives = synthetic + near + extra_neg
    for q in all_positives + negatives:
        q["script"] = query_script(q["name"])
    print(
        f"queries: {len(positives)} + {len(extra_pos)} positive, {len(synthetic) + len(extra_neg)} synthetic,"
        f" {len(near)} near-miss",
        file=sys.stderr,
    )
    return all_positives, negatives, {**kept, **extra_kept}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--max-df", type=int, default=1000)
    args = parser.parse_args()
    print(f"Python {platform.python_version()}", file=sys.stderr)

    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    all_positives, negatives, held = build_query_sets(entities, targets)

    held_screener = Screener.build([{**e, "names": held.get(e["id"], e["names"])} for e in targets], max_df=args.max_df)
    full = Screener.build(targets, max_df=args.max_df)

    def pick(queries: list[dict], *splits: str, headline: bool = False) -> list[dict]:
        return [q for q in queries if q["split"] in splits and (not headline or q["group"] in ev.HEADLINE_GROUPS)]

    # ---- Step 1: floor, lift, blend and the global threshold, on dev, one blend's pool at a time.
    dev_pos, dev_neg = pick(all_positives, "dev", headline=True), pick(negatives, "dev")
    best = (0.0, 0.0, 0.0, 0.0, -1.0)  # blend, floor, lift, threshold, F1
    marginals: dict[str, dict[str, float]] = {"blend": {}, "skeleton_floor": {}, "skeleton_lift": {}}
    for blend in BLENDS:
        run_queries(held_screener, dev_pos, blend)
        run_queries(full, dev_neg, blend)
        pos_view, neg_view = view(dev_pos, blend), view(dev_neg, blend)
        for floor, lift in itertools.product(ev.SKELETON_FLOORS, ev.SKELETON_LIFTS):
            params = ev.Params(floor, lift, 0)
            pos_rows = ev.final_scores(pos_view, params, True, True)
            neg_rows = ev.final_scores(neg_view, params, True, True)
            top = -1.0
            for t in ev.THRESHOLDS:
                f1 = ev.metrics_at(pos_rows, neg_rows, t)["f1"]
                top = max(top, f1)
                if f1 >= best[4]:
                    best = (blend, floor, lift, t, f1)
            for name, value in (("blend", blend), ("skeleton_floor", floor), ("skeleton_lift", lift)):
                marginals[name][str(value)] = max(marginals[name].get(str(value), -1.0), top)
        print(f"blend {blend} tuned on its own pool; best so far {best}", file=sys.stderr)
    blend, floor, lift, global_t, dev_f1 = best
    params = ev.Params(floor, lift, 0)

    # ---- Step 2: the rest of the queries, scored with the chosen blend only.
    rest_pos = [q for q in all_positives if f"comps_{blend}" not in q]
    rest_neg = [q for q in negatives if f"comps_{blend}" not in q]
    latencies = run_queries(held_screener, rest_pos, blend)
    run_queries(full, rest_neg, blend)

    def rows(queries: list[dict]) -> list[tuple]:
        return ev.final_scores(view(queries, blend), params, True, True)

    dev_pos_rows, dev_neg_rows = rows(pick(all_positives, "dev")), rows(pick(negatives, "dev", "dev_extra"))
    test_pos_rows, test_neg_rows = rows(pick(all_positives, "test")), rows(pick(negatives, "test"))
    thresholds = choose_script_thresholds(dev_pos_rows, dev_neg_rows, global_t)

    single = ev.metrics_at(test_pos_rows, test_neg_rows, global_t)
    per_script = ev.metrics_at(test_pos_rows, test_neg_rows, thresholds)
    dev_detail = {
        s: {
            "recall_at_global": round(recall_at(dev_pos_rows, s, global_t), 4),
            "recall_at_chosen": round(recall_at(dev_pos_rows, s, thresholds[s]), 4),
            "fpr_at_global": round(fpr_at(dev_neg_rows, s, global_t), 4),
            "fpr_at_chosen": round(fpr_at(dev_neg_rows, s, thresholds[s]), 4),
        }
        for s in TUNED_SCRIPTS
    }

    # ---- The engine itself: same scores as the evaluation, and how long a call takes.
    config = ScreeningConfig(floor, lift, blend, thresholds)
    engine = ScreeningEngine(full, config)
    test_neg = pick(negatives, "test")
    sample = random.Random(1).sample(range(len(test_neg)), 300)
    mismatches, engine_ms = 0, []
    for i in sample:
        q, row = test_neg[i], test_neg_rows[i]
        start = time.perf_counter()
        result = engine.screen(q["name"], q["dob"], q["country"])
        engine_ms.append((time.perf_counter() - start) * 1000)
        if (row[3] >= thresholds[row[5]]) != bool(result) or (result and abs(result[0].score - row[3]) > 1e-9):
            mismatches += 1

    result = {
        "python_version": platform.python_version(),
        "tuned_on_deployed_candidate_pool": {
            "blend": blend,
            "skeleton_floor": floor,
            "skeleton_lift": lift,
            "global_threshold": global_t,
            "dev_f1": dev_f1,
            "dev_best_f1_by_parameter_value": marginals,
        },
        "thresholds": thresholds,
        "dev_per_script": dev_detail,
        "dev_negatives_per_script": {s: sum(r[5] == s for r in dev_neg_rows) for s in SCRIPTS},
        "dev_positives_per_script": {
            s: sum(r[5] == s and r[0] in ev.HEADLINE_GROUPS for r in dev_pos_rows) for s in SCRIPTS
        },
        "test_negatives_per_script": {s: sum(r[5] == s for r in test_neg_rows) for s in SCRIPTS},
        "fpr_by_script_test": {"single_threshold": single["fpr_by_script"], "per_script": per_script["fpr_by_script"]},
        "recall_by_script_test": {
            "single_threshold": single["recall_by_script"],
            "per_script": per_script["recall_by_script"],
        },
        "fpr_spread_latin_cyrillic_arabic_test": {
            "single_threshold": spread(single["fpr_by_script"], TUNED_SCRIPTS),
            "per_script": spread(per_script["fpr_by_script"], TUNED_SCRIPTS),
        },
        "step_4_baseline_as_reported": STEP_4_BASELINE,
        "test_overall_single_threshold": single,
        "test_overall_per_script_thresholds": per_script,
        "engine": {
            "calls_checked_against_evaluation": len(sample),
            "mismatches": mismatches,
            "call_ms_median": round(percentile(engine_ms, 0.5), 2),
            "call_ms_p95": round(percentile(engine_ms, 0.95), 2),
        },
        "positive_screen_ms": {
            "median": round(percentile(latencies, 0.5) * 1000, 2),
            "p95": round(percentile(latencies, 0.95) * 1000, 2),
        },
        "final_holdout": {
            "checked_against_checksum": (ROOT / "holdout" / "final_holdout.sha256").read_text().strip()[:12],
            "scored": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    args.config.write_text(
        json.dumps(
            {
                "skeleton_floor": floor,
                "skeleton_lift": lift,
                "blend": blend,
                "thresholds": thresholds,
                "global_threshold": global_t,
                "per_script": False,
                "tuned_on": "dev: scoring parameters on the deployed candidate pool; per-script thresholds for the "
                "lowest false positive rate at no loss of recall; other scripts use the global threshold",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.output} and {args.config}")


if __name__ == "__main__":
    main()
