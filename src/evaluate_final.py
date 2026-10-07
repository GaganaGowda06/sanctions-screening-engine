"""Final evaluation on the frozen hold-out. Run once, with the frozen configuration, no tuning.

Order of work:
  1. Check the SHA-256 of holdout/final_holdout.json against the checksum recorded when it was frozen.
  2. Check that src/screening_config.json is the frozen configuration (one global threshold 0.86,
     blend 0.5, skeleton lift 0.2, floor 0.6, per-script thresholds off).
  3. Score every hold-out query. Positives are screened against a list with their own names held out,
     negatives against the full list, exactly as in dev and test.
  4. Report overall, by direction and by script, with 95% confidence intervals, the false positive rate
     as alerts per 1,000 clean names, and a comparison with the step 4 test-set figures.

The script refuses to run a second time (it will not overwrite reports/final_evaluation.json), because
looking at a hold-out again after seeing its results is tuning on it.

Confidence intervals: Wilson score intervals for proportions, bootstrap percentile intervals (2,000
resamples of queries) for F1.

Run:
    python src/evaluate_final.py
Writes reports/final_evaluation.json.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import evaluate_scoring as ev
from entities import load_named_entities
from evaluate_candidates import percentile
from profile_data import DEFAULT_INPUT
from scoring import Screener
from screening import CONFIG_PATH, ScreeningConfig, ScreeningEngine, query_script

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "holdout" / "final_holdout.json"
HOLDOUT_SHA = ROOT / "holdout" / "final_holdout.sha256"
TEST_REPORT = ROOT / "reports" / "stage2_screening.json"
DEFAULT_OUTPUT = ROOT / "reports" / "final_evaluation.json"
FROZEN = ScreeningConfig(
    skeleton_floor=0.6,
    skeleton_lift=0.2,
    blend=0.5,
    thresholds={"Latin": 0.86, "Cyrillic": 0.88, "Arabic": 0.86, "other": 0.86},
    global_threshold=0.86,
    per_script=False,
)
SCRIPTS = ("Latin", "Cyrillic", "Arabic")
GROUPS = tuple(g for g in ev.POSITIVE_GROUPS if g in ev.HEADLINE_GROUPS)
BOOTSTRAP_RESAMPLES = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def estimate(k: int, n: int) -> dict:
    low, high = wilson(k, n)
    return {"value": round(k / n, 4) if n else None, "ci95": [round(low, 4), round(high, 4)], "k": k, "n": n}


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value of a z-test for equal proportions."""
    if not n1 or not n2:
        return 1.0
    pooled = (k1 + k2) / (n1 + n2)
    spread = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 1.0 if spread == 0 else math.erfc(abs(k1 / n1 - k2 / n2) / spread / math.sqrt(2))


def bootstrap_f1(pos_codes: list[int], neg_alerts: list[bool], seed: int = 5) -> tuple[float, float]:
    """95% percentile interval for F1.

    pos_codes: 1 = the true entity was alerted, 2 = only a wrong entity was, 0 = nothing was.
    """
    rng = random.Random(seed)
    scores = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        pos = rng.choices(pos_codes, k=len(pos_codes))
        tp, wrong = pos.count(1), pos.count(2)
        fp = wrong + sum(rng.choices(neg_alerts, k=len(neg_alerts)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(pos_codes)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    scores.sort()
    return (scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores)) - 1])


def code_version() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compare(label: str, est: dict, test_value: float, test_n: int | None) -> dict:
    """One line of the test-versus-hold-out table. `est` is an estimate() result, or has just value and ci95."""
    low, high = est["ci95"]
    k, n = est.get("k"), est.get("n")
    if est["value"] is None:  # nothing of this kind in the set, so nothing to compare
        return {"figure": label, "test": round(test_value, 4), "holdout": None, "difference": None,
                "holdout_ci95": [low, high], "test_inside_holdout_ci": None, "p_value": None}  # fmt: skip
    p_value = None
    if k is not None and n and test_n:
        p_value = round(two_proportion_p(k, n, round(test_value * test_n), test_n), 4)
    return {
        "figure": label,
        "test": round(test_value, 4),
        "holdout": est["value"],
        "difference": round(est["value"] - test_value, 4),
        "holdout_ci95": [low, high],
        "test_inside_holdout_ci": low <= test_value <= high,
        "p_value": p_value,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--checksum", type=Path, default=HOLDOUT_SHA)
    parser.add_argument("--test-report", type=Path, default=TEST_REPORT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-df", type=int, default=1000)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    if args.output.exists():
        sys.exit(f"{args.output} exists: the final evaluation has been run. It is not run twice.")

    # 1. Checksum first.
    actual = hashlib.sha256(args.holdout.read_bytes()).hexdigest()
    recorded = args.checksum.read_text().strip()
    if actual != recorded:
        sys.exit(f"Hold-out checksum mismatch: file {actual[:12]}, recorded {recorded[:12]}. Not scoring.")
    print(f"hold-out checksum ok ({actual[:12]})", file=sys.stderr)

    # 2. Frozen configuration, and a test report that was produced with it.
    config = ScreeningConfig.load(args.config)
    if config != FROZEN:
        sys.exit(f"{args.config} is not the frozen configuration: {config}")
    test_report = json.loads(args.test_report.read_text())
    tuned = test_report["tuned_on_deployed_candidate_pool"]
    if (tuned["blend"], tuned["skeleton_floor"], tuned["skeleton_lift"], tuned["global_threshold"]) != (
        FROZEN.blend, FROZEN.skeleton_floor, FROZEN.skeleton_lift, FROZEN.global_threshold
    ):  # fmt: skip
        sys.exit("The test report was not produced with the frozen configuration.")
    test = test_report["test_overall_single_threshold"]
    threshold = FROZEN.global_threshold

    # 3. Score.
    payload = json.loads(args.holdout.read_text())
    queries, kept = payload["queries"], payload["kept_names"]
    for q in queries:
        q["script"] = query_script(q["name"])
    positives = [q for q in queries if q["kind"] == "positive"]
    negatives = [q for q in queries if q["kind"] != "positive"]

    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    held_screener = Screener.build(
        [{**e, "names": kept.get(e["id"], e["names"])} for e in targets], max_df=args.max_df, blend=FROZEN.blend
    )
    full_screener = Screener.build(targets, max_df=args.max_df, blend=FROZEN.blend)
    for screener, group in ((held_screener, positives), (full_screener, negatives)):
        for q in group:
            q["comps"] = screener.components(
                q["name"], q["dob"], q["country"], always=q["true_id"], blends=(FROZEN.blend,)
            )
    params = ev.Params(FROZEN.skeleton_floor, FROZEN.skeleton_lift, 0)
    pos_rows = ev.final_scores(positives, params, True, True)
    neg_rows = ev.final_scores(negatives, params, True, True)
    point = ev.metrics_at(pos_rows, neg_rows, threshold)

    # The deployed engine must agree with the components on every alert.
    mismatches, call_ms = 0, []
    for engine, group, rows in (
        (ScreeningEngine(held_screener, FROZEN), positives, pos_rows),
        (ScreeningEngine(full_screener, FROZEN), negatives, neg_rows),
    ):
        for q, row in zip(group, rows, strict=True):
            start = time.perf_counter()
            matches = engine.screen(q["name"], q["dob"], q["country"])
            call_ms.append((time.perf_counter() - start) * 1000)
            ids = {m.entity_id for m in matches}
            if q["kind"] == "positive":
                expected_true = row[2] is not None and row[2] >= threshold
                any_alert = expected_true or row[3] >= threshold
                consistent = (q["true_id"] in ids) == expected_true and bool(ids) == any_alert
            else:
                consistent = bool(ids) == (row[3] >= threshold)
            mismatches += not consistent

    # 4. Numbers with intervals.
    head = [r for r in pos_rows if r[0] in ev.HEADLINE_GROUPS]
    head_neg = [r for r in neg_rows if r[1] in ev.HEADLINE_NEG_SCRIPTS]
    true_alert = [r[2] is not None and r[2] >= threshold for r in head]
    wrong_only = [(not t) and r[3] >= threshold for t, r in zip(true_alert, head, strict=True)]
    neg_alert = [r[3] >= threshold for r in head_neg]
    tp, wrong, fp_neg = sum(true_alert), sum(wrong_only), sum(neg_alert)
    top1 = sum(r[4] for r in head)
    codes = [1 if t else 2 if w else 0 for t, w in zip(true_alert, wrong_only, strict=True)]
    f1_low, f1_high = bootstrap_f1(codes, neg_alert)
    overall = {
        "recall": estimate(tp, len(head)),
        "precision": estimate(tp, tp + fp_neg + wrong),
        "f1": {"value": point["f1"], "ci95": [round(f1_low, 4), round(f1_high, 4)], "method": "bootstrap"},
        "top1_accuracy": estimate(top1, len(head)),
        "false_positive_rate": estimate(fp_neg, len(head_neg)),
    }
    assert abs(overall["recall"]["value"] - point["recall"]) < 1e-3
    assert abs(overall["false_positive_rate"]["value"] - point["false_positive_rate"]) < 1e-3

    def positives_in(group: str | None = None, script: str | None = None) -> list[tuple[tuple, bool]]:
        return [(r, t) for r, t in zip(head, true_alert, strict=True)
                if (group is None or r[0] == group) and (script is None or r[5] == script)]  # fmt: skip

    def negatives_in(script: str | None = None, kind: str | None = None) -> list[bool]:
        return [a for a, r in zip(neg_alert, head_neg, strict=True)
                if (script is None or r[1] == script) and (kind is None or r[0] == kind)]  # fmt: skip

    by_direction = {}
    for group in GROUPS:
        rows = positives_in(group)
        by_direction[group] = estimate(sum(t for _, t in rows), len(rows))
    other = [r for r in pos_rows if r[0] == "other_to_latin"]
    by_direction["other_to_latin (known limitation, not in the headline)"] = estimate(
        sum(r[2] is not None and r[2] >= threshold for r in other), len(other)
    )
    by_script = {}
    for script in SCRIPTS:
        rows, negs = positives_in(script=script), negatives_in(script)
        by_script[script] = {
            "recall": estimate(sum(t for _, t in rows), len(rows)),
            "false_positive_rate": estimate(sum(negs), len(negs)),
        }
    kinds = {kind: negatives_in(kind=kind) for kind in ("synthetic", "near_miss")}
    per_1000 = {"clean_name_mix": "synthetic recombined names and near-miss namesakes, as sampled"}

    def per_thousand(label: str, flags: list[bool]) -> None:
        if not flags:
            per_1000[label] = {"alerts_per_1000_clean_names": None, "ci95": None, "names": 0}
            return
        low, high = wilson(sum(flags), len(flags))
        per_1000[label] = {
            "alerts_per_1000_clean_names": round(1000 * sum(flags) / len(flags), 1),
            "ci95": [round(1000 * low, 1), round(1000 * high, 1)],
            "names": len(flags),
        }

    per_thousand("all", neg_alert)
    for kind, flags in kinds.items():
        per_thousand(kind, flags)
    for script in SCRIPTS:
        per_thousand(f"{script} queries", negatives_in(script))

    # 5. Test set against hold-out.
    test_groups, test_scripts = test["recall_by_group"], test["recall_by_script"]
    neg_test_scripts = test_report["test_negatives_per_script"]
    n_pos_test = {g: ev.POSITIVE_GROUPS[g][1] for g in GROUPS}
    script_pos_test = {
        "Latin": n_pos_test["variant"] + n_pos_test["latin_to_cyr"] + n_pos_test["latin_to_ar"],
        "Cyrillic": n_pos_test["cyr_to_latin"],
        "Arabic": n_pos_test["ar_to_latin"],
    }
    synthetic_est = estimate(sum(kinds["synthetic"]), len(kinds["synthetic"]))
    near_est = estimate(sum(kinds["near_miss"]), len(kinds["near_miss"]))
    table = [
        compare("recall", overall["recall"], test["recall"], 2000),
        compare("precision", overall["precision"], test["precision"], None),
        compare("f1", overall["f1"], test["f1"], None),
        compare("top-1 accuracy", overall["top1_accuracy"], test["top1_accuracy"], 2000),
        compare("false positive rate", overall["false_positive_rate"], test["false_positive_rate"], 2000),
        compare("FPR, synthetic names", synthetic_est, test["fpr_synthetic"], sum(ev.SYNTHETIC["test"].values())),
        compare("FPR, near-miss namesakes", near_est, test["fpr_near_miss"], ev.NEAR_MISS["test"]),
    ]
    for group in GROUPS:
        table.append(compare(f"recall, {group}", by_direction[group], test_groups[group], n_pos_test[group]))
    for script in SCRIPTS:
        recall, fpr = by_script[script]["recall"], by_script[script]["false_positive_rate"]
        table.append(compare(f"recall, {script} queries", recall, test_scripts[script], script_pos_test[script]))
        table.append(compare(f"FPR, {script} queries", fpr, test["fpr_by_script"][script], neg_test_scripts[script]))
    dev_f1 = tuned["dev_f1"]

    outside = [r["figure"] for r in table if r["test_inside_holdout_ci"] is False]
    report = {
        "python_version": platform.python_version(),
        "code_commit": code_version(),
        "checksum": {"holdout_sha256": actual, "recorded": recorded, "ok": True},
        "frozen_config": {
            "skeleton_floor": FROZEN.skeleton_floor, "skeleton_lift": FROZEN.skeleton_lift, "blend": FROZEN.blend,
            "global_threshold": FROZEN.global_threshold, "per_script_thresholds": FROZEN.per_script,
        },  # fmt: skip
        "queries": {"positive_headline": len(head), "negative": len(head_neg), "positive_other_to_latin": len(other)},
        "overall": overall,
        "by_direction": by_direction,
        "by_script": by_script,
        "false_positives_per_1000": per_1000,
        "f1_dev_test_holdout": {"dev": dev_f1, "test": test["f1"], "holdout": overall["f1"]["value"]},
        "test_vs_holdout": table,
        "figures_with_test_value_outside_holdout_ci": outside,
        "engine_check": {"queries": len(queries), "mismatches": mismatches},
        "engine_call_ms": {"median": round(percentile(call_ms, 0.5), 2), "p95": round(percentile(call_ms, 0.95), 2)},
    }  # fmt: skip
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
