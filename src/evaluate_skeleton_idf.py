"""Stage 3 step 2: does rarity-weighting the skeleton cut Arabic false alerts? Dev, then test. No hold-out.

One change against the frozen engine: the skeleton similarity is a rarity-weighted token overlap over
skeleton tokens (weights from the document frequency of each skeleton token over the listed entities)
instead of a whole-string ratio. Nothing else moves: same blend, skeleton floor and lift, same global
alert threshold of 0.86, nothing retuned.

Reported on dev and on test for the baseline and the fix: per-script false positive rate (synthetic clean
names, which are the like-for-like comparison; all clean names too), per-script recall, overall recall,
precision and F1, and the Arabic to Latin and Arabic to Cyrillic alert-rate ratios with bootstrap intervals.
The acceptance rule was set before running: Arabic false positive rate clearly down, Arabic recall at or
above 65%, Latin and Cyrillic false positive rate and recall unchanged within noise (95% intervals overlap).

The three hold-outs are not read: build_query_sets regenerates the first one in memory only to keep new
queries away from its entities.

Run:
    python src/evaluate_skeleton_idf.py
Writes reports/stage3_skeleton_idf.json.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import evaluate_scoring as ev
from entities import load_named_entities
from evaluate_final import estimate
from evaluate_screening import build_query_sets
from measure_alert_disparity import bootstrap_contrast
from profile_data import DEFAULT_INPUT
from scoring import Screener
from screening import CONFIG_PATH, ScreeningConfig

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "stage3_skeleton_idf.json"
SCRIPTS = ("Latin", "Cyrillic", "Arabic")


def overlap(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def evaluate(
    positives: list[dict], negatives: list[dict], screeners: tuple[Screener, Screener], config: ScreeningConfig
) -> dict:
    held, full = screeners
    for q in positives:
        q["comps"] = held.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=(config.blend,))
    for q in negatives:
        q["comps"] = full.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=(config.blend,))
    params = ev.Params(config.skeleton_floor, config.skeleton_lift, 0)
    t = config.global_threshold
    pos_rows = ev.final_scores(positives, params, True, True)
    neg_rows = ev.final_scores(negatives, params, True, True)
    point = ev.metrics_at(pos_rows, neg_rows, t)
    head = [r for r in pos_rows if r[0] in ev.HEADLINE_GROUPS]

    def alerts(script: str, kind: str | None) -> list[bool]:
        return [r[3] >= t for r in neg_rows if r[1] == script and (kind is None or r[0] == kind)]

    out = {
        "overall": {k: point[k] for k in ("recall", "precision", "f1", "false_positive_rate")},
        "fpr_synthetic": {}, "fpr_all_clean": {}, "recall": {},
    }  # fmt: skip
    for script in SCRIPTS:
        for key, kind in (("fpr_synthetic", "synthetic"), ("fpr_all_clean", None)):
            f = alerts(script, kind)
            out[key][script] = estimate(sum(f), len(f))
        rows = [r for r in head if r[5] == script]
        out["recall"][script] = estimate(sum(r[2] is not None and r[2] >= t for r in rows), len(rows))
    arabic = alerts("Arabic", "synthetic")
    out["arabic_ratio_synthetic"] = {
        ref: bootstrap_contrast(arabic, alerts(ref, "synthetic")) for ref in ("Latin", "Cyrillic")
    }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-df", type=int, default=1000)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    config = ScreeningConfig.load(CONFIG_PATH)
    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    all_pos, all_neg, kept = build_query_sets(entities, targets)
    held_entities = [{**e, "names": kept.get(e["id"], e["names"])} for e in targets]

    report = {"engine": "frozen config, threshold 0.86", "change": "rarity-weighted skeleton tokens"}
    sets = {
        "dev": ([q for q in all_pos if q["split"] == "dev" and q["group"] in ev.HEADLINE_GROUPS],
                [q for q in all_neg if q["split"] in ("dev", "dev_extra") and q["script"] in SCRIPTS]),
        "test": ([q for q in all_pos if q["split"] == "test" and q["group"] in ev.HEADLINE_GROUPS],
                 [q for q in all_neg if q["split"] == "test" and q["script"] in SCRIPTS]),
    }  # fmt: skip
    for label, skeleton_idf in (("baseline", False), ("rarity_weighted_skeleton", True)):
        screeners = (
            Screener.build(held_entities, max_df=args.max_df, blend=config.blend, skeleton_idf=skeleton_idf),
            Screener.build(targets, max_df=args.max_df, blend=config.blend, skeleton_idf=skeleton_idf),
        )
        for split, (pos, neg) in sets.items():
            report.setdefault(split, {})[label] = evaluate(pos, neg, screeners, config)
            print(f"{split} {label} done", file=sys.stderr)

    verdict = {}
    for split in sets:
        base, fix = report[split]["baseline"], report[split]["rarity_weighted_skeleton"]
        fpr_base, fpr_fix = base["fpr_synthetic"]["Arabic"], fix["fpr_synthetic"]["Arabic"]
        checks = {
            # "clearly down": the new interval sits wholly under the old one, or the rate at least halved
            "arabic_fpr_clearly_down": fpr_fix["ci95"][1] < fpr_base["ci95"][0]
            or fpr_fix["value"] <= fpr_base["value"] / 2,
            "arabic_recall_at_least_65": fix["recall"]["Arabic"]["value"] >= 0.65,
        }  # fmt: skip
        for script in ("Latin", "Cyrillic"):
            for key in ("fpr_synthetic", "recall"):
                checks[f"{script}_{key}_unchanged"] = overlap(base[key][script]["ci95"], fix[key][script]["ci95"])
        verdict[split] = {**checks, "all_pass": all(checks.values())}
    report["acceptance"] = verdict
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    for split in sets:
        for label in ("baseline", "rarity_weighted_skeleton"):
            r = report[split][label]
            print(split, label, "overall", r["overall"])
            for key in ("fpr_synthetic", "recall"):
                print("  ", key, {s: (r[key][s]["value"], r[key][s]["ci95"]) for s in SCRIPTS})
            print("   ratio vs Latin", r["arabic_ratio_synthetic"]["Latin"]["ratio"],
                  r["arabic_ratio_synthetic"]["Latin"]["ratio_ci95"])  # fmt: skip
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
