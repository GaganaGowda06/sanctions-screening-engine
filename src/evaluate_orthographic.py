"""Stage 3 step 5: an orthographic Arabic skeleton that keeps long vowels and the initial alif. Dev, then test.

For Arabic-involved pairs only, the skeleton is built by `orthographic_token` (src/scoring.py) instead of the
vowel-free one, and compared with a stricter token match (ratio 85, not 70), so Mohammed (mhmd) and Mahmoud
(mhmuwd) are different tokens. Latin-with-Latin and Cyrillic pairs are untouched. Threshold, blend, floor and
lift are unchanged. The flag is `orthographic` on the Screener, off by default.

First two direct checks: Mohammed against Mahmoud must no longer share a skeleton token, and spellings of one
name must still share one. Then, on dev and on test: per-script false positive rate (synthetic clean names) and
recall for the current engine and the new one, the Arabic to Latin and Arabic to Cyrillic ratios, and an exact
McNemar test of the new engine against the current one. Acceptance as before: Arabic FPR clearly down (new
interval wholly under the old, or at least halved), Arabic recall at or above 65%, Latin and Cyrillic rates and
recall unchanged within noise.

Run:
    python src/evaluate_orthographic.py
Writes reports/stage3_orthographic.json.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import evaluate_scoring as ev
from entities import load_named_entities
from evaluate_screening import build_query_sets
from evaluate_skeleton_particles import overlap, paired, score, summarise
from profile_data import DEFAULT_INPUT
from scoring import ORTHO_CUTOFF, Screener, orthographic_form
from screening import CONFIG_PATH, ScreeningConfig

OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "stage3_orthographic.json"
SCRIPTS = ("Latin", "Cyrillic", "Arabic")
SAME_NAME = {
    "Mohammed": ["Muhammad", "Mohamed", "Mhmd", "Mohammad"],
    "Mahmoud": ["Mahmood", "Mahmud", "Mhmwd"],
    "Hussein": ["Husayn", "Hosein", "Hsyn", "Hussain"],
    "Ahmed": ["Ahmad", "Ahmd"],
    "Rashid": ["Rasheed", "Rshyd"],
    "Ibrahim": ["Ibraheem", "Ebrahim"],
    "Khalid": ["Khalid", "Khlyd"],
    "Yusuf": ["Youssef", "Yousef", "Yusef", "Ysf"],
    "Omar": ["Umar", "Omer"],
}


def token_ratio(a: str, b: str) -> int:
    from rapidfuzz import fuzz

    return round(fuzz.ratio(orthographic_form(a.lower()), orthographic_form(b.lower())))


def spelling_checks() -> dict:
    kept = {name: {v: token_ratio(name, v) >= ORTHO_CUTOFF for v in variants} for name, variants in SAME_NAME.items()}
    total = sum(len(v) for v in kept.values())
    return {
        "mohammed_vs_mahmoud_ratio": token_ratio("Mohammed", "Mahmoud"),
        "mohammed_vs_mahmoud_still_match": token_ratio("Mohammed", "Mahmoud") >= ORTHO_CUTOFF,
        "variants_still_matching": f"{sum(sum(v.values()) for v in kept.values())} of {total}",
        "detail": kept,
    }


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
    sets = {
        "dev": (
            [q for q in all_pos if q["split"] == "dev" and q["group"] in ev.HEADLINE_GROUPS],
            [q for q in all_neg if q["split"] in ("dev", "dev_extra") and q["script"] in SCRIPTS],
        ),
        "test": (
            [q for q in all_pos if q["split"] == "test" and q["group"] in ev.HEADLINE_GROUPS],
            [q for q in all_neg if q["split"] == "test" and q["script"] in SCRIPTS],
        ),
    }
    runs: dict[str, dict] = {}
    for label, kwargs in (("current", {}), ("new", {"orthographic": True})):
        screeners = (
            Screener.build(held_entities, max_df=args.max_df, blend=config.blend, **kwargs),
            Screener.build(targets, max_df=args.max_df, blend=config.blend, **kwargs),
        )
        for split, (pos, neg) in sets.items():
            runs[f"{split}/{label}"] = score([dict(q) for q in pos], [dict(q) for q in neg], screeners, config)
            print(f"{split} {label} done", file=sys.stderr)

    report: dict = {"change": "orthographic skeleton for Arabic-involved pairs", "spelling_checks": spelling_checks()}
    verdict = {}
    for split in sets:
        cur, new = runs[f"{split}/current"], runs[f"{split}/new"]
        a, b = summarise(cur), summarise(new)
        report[split] = {"current": a, "new": b, "mcnemar": paired(cur, new)}
        fa, fb = a["fpr_synthetic"]["Arabic"], b["fpr_synthetic"]["Arabic"]
        checks = {
            "arabic_fpr_clearly_down": fb["ci95"][1] < fa["ci95"][0] or fb["value"] <= fa["value"] / 2,
            "arabic_recall_at_least_65": b["recall"]["Arabic"]["value"] >= 0.65,
        }
        for script in ("Latin", "Cyrillic"):
            for key in ("fpr_synthetic", "recall"):
                checks[f"{script}_{key}_unchanged"] = overlap(a[key][script]["ci95"], b[key][script]["ci95"])
        verdict[split] = {**checks, "all_pass": all(checks.values())}
    report["acceptance"] = verdict
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in report["spelling_checks"].items() if k != "detail"}, indent=1))
    for split in sets:
        for label in ("current", "new"):
            r = report[split][label]
            print(split, label, r["overall"])
            for key in ("fpr_synthetic", "recall"):
                print("  ", key, {s: (r[key][s]["value"], r[key][s]["ci95"]) for s in SCRIPTS})
            rl = r["arabic_ratio_synthetic"]["Latin"]
            print("   Arabic/Latin", rl["ratio"], rl["ratio_ci95"])
        print("  mcnemar", json.dumps(report[split]["mcnemar"]))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
