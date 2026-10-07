"""Baseline alert-rate disparity across name scripts, on the third screening hold-out. Engine unchanged.

The fairness question: on clean names (names that match no listed entity), how much more often does the
frozen engine raise an alert for Arabic-script names than for Latin or Cyrillic ones? Every alert on a clean
name is a false alert that lands on a customer.

Headline: the alert rate on synthetic clean names, per script, and the ratio and difference against Latin
and Cyrillic. Synthetic names are the like-for-like comparison, because near-miss namesakes exist almost only
in Latin and Cyrillic (the list has almost no Arabic-only namesakes with conflicting details), so mixing them
in would compare different kinds of name. The all-negatives figure is reported too, with that caveat.

Intervals (95%): Wilson for each rate. Ratios and differences use a bootstrap percentile interval
(5,000 resamples of the clean names within each script).

Order of work: check the hold-out checksum, check the frozen configuration, score once, write the report.
The script refuses to run a second time. It does not change the engine or any threshold.

Run:
    python src/measure_alert_disparity.py
Writes reports/fairness_baseline.json.
"""

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import evaluate_scoring as ev
from entities import load_named_entities
from evaluate_final import FROZEN, SCRIPTS, code_version, estimate
from profile_data import DEFAULT_INPUT
from scoring import Screener
from screening import CONFIG_PATH, ScreeningConfig, query_script

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "holdout" / "screening_holdout_v3.json"
HOLDOUT_SHA = ROOT / "holdout" / "screening_holdout_v3.sha256"
DEFAULT_OUTPUT = ROOT / "reports" / "fairness_baseline.json"
RESAMPLES = 5000


def bootstrap_contrast(a: list[bool], b: list[bool], seed: int = 7) -> dict:
    """Ratio (a rate / b rate) and difference (a minus b, in points) with 95% percentile intervals."""
    rng = random.Random(seed)
    ratios, diffs = [], []
    for _ in range(RESAMPLES):
        ra = sum(rng.choices(a, k=len(a))) / len(a)
        rb = sum(rng.choices(b, k=len(b))) / len(b)
        diffs.append(100 * (ra - rb))
        if rb > 0:
            ratios.append(ra / rb)
    ratios.sort()
    diffs.sort()
    ra, rb = sum(a) / len(a), sum(b) / len(b)
    pick = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {
        "ratio": round(ra / rb, 2) if rb else None,
        "ratio_ci95": [round(pick(ratios, 0.025), 2), round(pick(ratios, 0.975), 2)],
        "difference_points": round(100 * (ra - rb), 1),
        "difference_ci95": [round(pick(diffs, 0.025), 1), round(pick(diffs, 0.975), 1)],
        "bootstrap_resamples_with_zero_reference_rate": RESAMPLES - len(ratios),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-df", type=int, default=1000)
    parser.add_argument(
        "--engine",
        choices=("baseline", "current"),
        default="baseline",
        help="baseline: the Stage 2 frozen engine (skeleton_idf off); current: the engine as it is now",
    )
    args = parser.parse_args(argv)
    skeleton_idf = args.engine == "current"
    if args.engine == "current" and args.output == DEFAULT_OUTPUT:
        args.output = ROOT / "reports" / "fairness_current.json"
    print(f"Python {platform.python_version()}", file=sys.stderr)

    if args.output.exists():
        sys.exit(f"{args.output} exists: the baseline has been measured. It is not run twice.")
    actual = hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()
    if actual != HOLDOUT_SHA.read_text().strip():
        sys.exit("Hold-out checksum mismatch. Not scoring.")
    config = ScreeningConfig.load(CONFIG_PATH)
    if config != FROZEN:
        sys.exit(f"{CONFIG_PATH} is not the frozen configuration: {config}")

    payload = json.loads(HOLDOUT.read_text())
    queries, kept = payload["queries"], payload["kept_names"]
    for q in queries:
        q["script"] = query_script(q["name"])
    positives = [q for q in queries if q["kind"] == "positive"]
    negatives = [q for q in queries if q["kind"] != "positive"]

    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    held = Screener.build([{**e, "names": kept.get(e["id"], e["names"])} for e in targets],
                          max_df=args.max_df, blend=FROZEN.blend, skeleton_idf=skeleton_idf)  # fmt: skip
    full = Screener.build(targets, max_df=args.max_df, blend=FROZEN.blend, skeleton_idf=skeleton_idf)
    for screener, group in ((held, positives), (full, negatives)):
        for q in group:
            q["comps"] = screener.components(q["name"], q["dob"], q["country"], always=q["true_id"],
                                             blends=(FROZEN.blend,))  # fmt: skip
    params = ev.Params(FROZEN.skeleton_floor, FROZEN.skeleton_lift, 0)
    threshold = FROZEN.global_threshold
    pos_rows = ev.final_scores(positives, params, True, True)
    neg_rows = ev.final_scores(negatives, params, True, True)

    # neg row: (kind, script, ..., best wrong score at index 3)
    clean = [(r[0], r[1], r[3] >= threshold) for r in neg_rows]

    def flags(script: str, kind: str | None = None) -> list[bool]:
        return [a for k, s, a in clean if s == script and (kind is None or k == kind)]

    def per_script(kind: str | None) -> dict:
        out = {}
        for script in (*SCRIPTS, "other"):
            f = flags(script, kind)
            if not f:
                out[script] = {"names": 0}
                continue
            e = estimate(sum(f), len(f))
            out[script] = {"alert_rate": e["value"], "ci95": e["ci95"], "alerts": e["k"], "names": e["n"],
                           "alerts_per_1000": round(1000 * e["value"], 1)}  # fmt: skip
        return out

    def contrasts(kind: str | None) -> dict:
        arabic = flags("Arabic", kind)
        return {f"Arabic vs {ref}": bootstrap_contrast(arabic, flags(ref, kind)) for ref in ("Latin", "Cyrillic")}

    pooled_ref = flags("Latin", "synthetic") + flags("Cyrillic", "synthetic")
    head_pos = [r for r in pos_rows if r[0] in ev.HEADLINE_GROUPS]
    recall = {}
    for script in SCRIPTS:
        rows = [r for r in head_pos if r[5] == script]
        k = sum(r[2] is not None and r[2] >= threshold for r in rows)
        recall[script] = estimate(k, len(rows))

    report = {
        "python_version": platform.python_version(),
        "code_commit": code_version(),
        "holdout": {
            "path": "holdout/screening_holdout_v3.json",
            "sha256": actual,
            "checksum_ok": True,
            "clean_names": len(negatives),
            "positive_queries": len(positives),
        },  # fmt: skip
        "engine": {
            "global_threshold": threshold,
            "blend": FROZEN.blend,
            "skeleton_floor": FROZEN.skeleton_floor,
            "skeleton_lift": FROZEN.skeleton_lift,
            "per_script_thresholds": False,
            "changed": skeleton_idf,
            "engine_option": args.engine,
            "skeleton_idf": skeleton_idf,
        },  # fmt: skip
        "metric": "share of clean names (matching no listed entity) that raise an alert, by query script",
        "headline_synthetic_clean_names": {
            "per_script": per_script("synthetic"),
            "contrasts": contrasts("synthetic"),
            "arabic_vs_latin_and_cyrillic_pooled": bootstrap_contrast(flags("Arabic", "synthetic"), pooled_ref),
        },
        "all_clean_names_not_like_for_like": {
            "note": "includes near-miss namesakes, which are almost all Latin or Cyrillic",
            "per_script": per_script(None),
            "contrasts": contrasts(None),
            "near_miss_names_per_script": {s: len(flags(s, "near_miss")) for s in (*SCRIPTS, "other")},
        },
        "recall_by_query_script_for_context": recall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"headline": report["headline_synthetic_clean_names"],
                      "all": report["all_clean_names_not_like_for_like"]["per_script"],
                      "recall": recall}, indent=1, ensure_ascii=False))  # fmt: skip


if __name__ == "__main__":
    main()
