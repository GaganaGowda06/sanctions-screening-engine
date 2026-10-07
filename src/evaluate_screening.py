"""Tune per-script alert thresholds on dev, report on test, and write the screening configuration.

The scoring parameters are those chosen in step 3c (skeleton floor 0.4, lift 0.3, rarity-weighted
name score). Only the alert threshold changes: one per query script (Latin, Cyrillic, Arabic, other).

Equalising false positive rates:
  target   the pooled dev false positive rate of Latin, Cyrillic and Arabic at the single global
           threshold of step 3c, so the overall alert volume stays about where it was
  rule     for each script, the lowest threshold whose dev false positive rate is at or under the
           target (a threshold of 1.0 alerts only on a perfect score)
Dev negatives for this are the step 3c dev negatives plus extra synthetic ones, since Arabic and
"other" have few. "Other" scripts (Han, kana, Hebrew, Thai) get synthetic negatives built the same way
as the rest, drawn evenly from those four. Everything is reported on the step 3c test queries, plus
"other" negatives and same-script "other" positives that step 3c did not have.

The final hold-out is not read or scored. Its entities are regenerated in memory only, to keep new
queries away from them, and checked against the frozen checksum.

Run:
    python src/evaluate_screening.py
Writes reports/stage2_screening.json and src/screening_config.json.
"""

import argparse
import hashlib
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
FLOOR, LIFT, BLEND = 0.4, 0.3, 1.0  # step 3c
GLOBAL_THRESHOLD = 0.87  # step 3c
PARAMS = ev.Params(FLOOR, LIFT, 0)
BLENDS = (BLEND,)
SCRIPT_THRESHOLDS = [*ev.THRESHOLDS, 1.0]
EXTRA_PLAN = {"other_same_script": [("dev", 100), ("test", 200)]}


def run_queries(screener: Screener, queries: list[dict]) -> list[float]:
    latencies = []
    for q in queries:
        start = time.perf_counter()
        q["comps"] = screener.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=BLENDS)
        latencies.append(time.perf_counter() - start)
    return latencies


def rate(rows: list[tuple], threshold: float) -> float:
    return sum(r[3] >= threshold for r in rows) / len(rows) if rows else 0.0


def choose_thresholds(dev_neg_rows: list[tuple], target: float) -> dict[str, float]:
    """For each script, the lowest threshold whose dev false positive rate is at or under the target."""
    chosen = {}
    for script in SCRIPTS:
        rows = [r for r in dev_neg_rows if r[5] == script]
        chosen[script] = next((t for t in SCRIPT_THRESHOLDS if rate(rows, t) <= target), SCRIPT_THRESHOLDS[-1])
    return chosen


def spread(fpr_by_script: dict[str, float | None], scripts: tuple[str, ...]) -> float:
    values = [fpr_by_script[s] for s in scripts if fpr_by_script[s] is not None]
    return round(max(values) - min(values), 4) if values else 0.0


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
    persons = [e for e in targets if e["schema"] == "Person"]
    dob_pool = [d for e in persons for d in e["birth_dates"][:1]]
    country_pool = [c for e in persons for c in e["countries"][:1]]

    # Step 3c dev and test queries, unchanged.
    rng = random.Random(ev.SEED)
    positives, kept, near, synthetic, used = ev.build_dev_test_queries(entities, targets, dob_pool, country_pool, rng)

    # Keep new queries off the final hold-out's entities. Regenerated in memory, never read from disk.
    final_queries, final_kept = ev.build_final_holdout(entities, targets, dob_pool, country_pool, used)
    frozen = (ROOT / "holdout" / "final_holdout.sha256").read_text().strip()
    regenerated = hashlib.sha256(ev.holdout_text(final_queries, final_kept).encode()).hexdigest()
    if regenerated != frozen:
        sys.exit("The regenerated final hold-out does not match the frozen checksum; refusing to continue.")
    reserved = used | {q.get("true_id") or q.get("entity") for q in final_queries}

    # New for step 4, on their own random stream: same-script "other" positives, extra and "other" negatives.
    extra_rng = random.Random(f"{ev.SEED}-extra")
    extra_pos, extra_kept = ev.build_positives(
        targets, extra_rng, EXTRA_PLAN, exclude=reserved, key="{seed}-extra-{group}"
    )
    extra_neg = ev.build_synthetic(
        entities, targets, dob_pool, country_pool, extra_rng, ("dev_extra", "test_other")
    )
    for q in extra_neg:
        if q["split"] == "test_other":
            q["split"] = "test"
    print(
        f"queries: {len(positives)} + {len(extra_pos)} positive, {len(synthetic)} + {len(extra_neg)} synthetic,"
        f" {len(near)} near-miss",
        file=sys.stderr,
    )

    all_positives = positives + extra_pos
    negatives = synthetic + near + extra_neg
    for q in all_positives + negatives:
        q["script"] = query_script(q["name"])

    # Components: positives against a list with their own names held out, negatives against the full list.
    held = {**kept, **extra_kept}
    held_list = [{**e, "names": held.get(e["id"], e["names"])} for e in targets]
    screener = Screener.build(held_list, max_df=args.max_df, blend=BLEND)
    latencies = run_queries(screener, all_positives)
    del screener, held_list
    full = Screener.build(targets, max_df=args.max_df, blend=BLEND)
    run_queries(full, negatives)

    dev_pos = [q for q in all_positives if q["split"] == "dev"]
    test_pos = [q for q in all_positives if q["split"] == "test"]
    dev_neg = [q for q in negatives if q["split"] in ("dev", "dev_extra")]
    test_neg = [q for q in negatives if q["split"] == "test"]
    rows = {name: ev.final_scores(qs, PARAMS, True, True) for name, qs in
            (("dev_pos", dev_pos), ("test_pos", test_pos), ("dev_neg", dev_neg), ("test_neg", test_neg))}  # fmt: skip

    # Target and per-script thresholds, from dev only.
    dev_core = [r for q, r in zip(dev_neg, rows["dev_neg"], strict=True) if q["split"] == "dev"]
    dev_core_head = [r for r in dev_core if r[1] in ev.HEADLINE_NEG_SCRIPTS]
    target = rate(dev_core_head, GLOBAL_THRESHOLD)
    thresholds = choose_thresholds(rows["dev_neg"], target)

    def report(pos_rows: list[tuple], neg_rows: list[tuple], threshold: float | dict[str, float]) -> dict:
        return ev.metrics_at(pos_rows, neg_rows, threshold)

    before = report(rows["test_pos"], rows["test_neg"], GLOBAL_THRESHOLD)
    after = report(rows["test_pos"], rows["test_neg"], thresholds)
    dev_before = report(rows["dev_pos"], rows["dev_neg"], GLOBAL_THRESHOLD)
    dev_after = report(rows["dev_pos"], rows["dev_neg"], thresholds)

    # The engine itself: same scores as the evaluation, and how long a call takes.
    config = ScreeningConfig(FLOOR, LIFT, BLEND, thresholds)
    engine = ScreeningEngine(full, config)
    sample = random.Random(1).sample(range(len(test_neg)), 300)
    mismatches, engine_ms = 0, []
    for i in sample:
        q, row = test_neg[i], rows["test_neg"][i]
        start = time.perf_counter()
        result = engine.screen(q["name"], q["dob"], q["country"])
        engine_ms.append((time.perf_counter() - start) * 1000)
        expected_alert = row[3] >= thresholds[row[5]]
        top = result[0].score if result else None
        if expected_alert != bool(result) or (result and abs(top - row[3]) > 1e-9):
            mismatches += 1

    examples = {
        "Vladimir Putin, 1952-10-07, ru": [m.as_dict() for m in engine.screen("Vladimir Putin", "1952-10-07", "ru", 2)],
        "Vladimir Putin, no birth date or country": [m.as_dict() for m in engine.screen("Vladimir Putin", limit=2)],
    }

    result = {
        "python_version": platform.python_version(),
        "scoring": {"skeleton_floor": FLOOR, "skeleton_lift": LIFT, "blend": BLEND},
        "global_threshold_step_3c": GLOBAL_THRESHOLD,
        "target_fpr_from_dev": round(target, 4),
        "thresholds": thresholds,
        "dev_negatives_per_script": {s: sum(r[5] == s for r in rows["dev_neg"]) for s in SCRIPTS},
        "test_negatives_per_script": {s: sum(r[5] == s for r in rows["test_neg"]) for s in SCRIPTS},
        "fpr_by_script": {
            "dev_global": dev_before["fpr_by_script"],
            "dev_per_script": dev_after["fpr_by_script"],
            "test_global": before["fpr_by_script"],
            "test_per_script": after["fpr_by_script"],
        },
        "fpr_spread_latin_cyrillic_arabic": {
            "test_global": spread(before["fpr_by_script"], ev.HEADLINE_NEG_SCRIPTS),
            "test_per_script": spread(after["fpr_by_script"], ev.HEADLINE_NEG_SCRIPTS),
        },
        "fpr_spread_all_four_scripts": {
            "test_global": spread(before["fpr_by_script"], SCRIPTS),
            "test_per_script": spread(after["fpr_by_script"], SCRIPTS),
        },
        "test_overall_global_threshold": before,
        "test_overall_per_script_thresholds": after,
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
        "examples": examples,
        "final_holdout": {"checked_against_checksum": frozen[:12], "scored": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    args.config.write_text(
        json.dumps(
            {
                "skeleton_floor": FLOOR,
                "skeleton_lift": LIFT,
                "blend": BLEND,
                "thresholds": thresholds,
                "tuned_on": "dev, to equalise the false positive rate across scripts",
                "target_fpr": round(target, 4),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.output} and {args.config}")


if __name__ == "__main__":
    main()
