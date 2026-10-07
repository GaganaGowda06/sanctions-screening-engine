"""Stage 3 step 3: are Arabic particles evidence? Dev, then test. No hold-out is read.

One change against the current engine (rarity-weighted skeleton, on by default): before the skeleton is
compared, Arabic particles (al, el, bin, ibn, bint, abu, umm) and any skeleton token of one consonant are left
out, on both the query and the entity side. A name left with no skeleton token has no skeleton evidence.
The threshold, blend, floor and lift are unchanged and nothing is retuned.

First, the hunch that a full match on common tokens still scores 1.0 under the rarity-weighted skeleton:
checked on two toy pairs and on the real dev clean names that alert with a skeleton similarity of 1.0.

Then, on dev and on test: per-script false positive rate (synthetic clean names) and recall for the current
engine and the new one, the Arabic to Latin and Arabic to Cyrillic ratios with bootstrap intervals, and an
exact McNemar test of the new engine against the current one on the same queries. The acceptance bar is the
earlier one: Arabic false positive rate clearly down (new interval wholly under the old, or at least
halved), Arabic recall at or above 65%, Latin and Cyrillic rates and recall unchanged within noise.

Run:
    python src/evaluate_skeleton_particles.py
Writes reports/stage3_skeleton_particles.json.
"""

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import evaluate_scoring as ev
from entities import load_named_entities
from evaluate_final import estimate
from evaluate_screening import build_query_sets
from measure_alert_disparity import bootstrap_contrast
from profile_data import DEFAULT_INPUT
from scoring import Components, Screener, combine
from screening import CONFIG_PATH, ScreeningConfig

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "stage3_skeleton_particles.json"
SCRIPTS = ("Latin", "Cyrillic", "Arabic")


def mcnemar(old: list[bool], new: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired yes/no outcomes."""
    only_old = sum(o and not n for o, n in zip(old, new, strict=True))
    only_new = sum(n and not o for o, n in zip(old, new, strict=True))
    m = only_old + only_new
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, k) for k in range(min(only_old, only_new) + 1)) / 2**m)
    return {"only_current": only_old, "only_new": only_new, "p_value": round(p, 4)}


def score(
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
    return {
        "t": t,
        "pos": pos_rows,
        "neg": neg_rows,
        "neg_q": negatives,
        "metrics": ev.metrics_at(pos_rows, neg_rows, t),
    }


def summarise(run: dict) -> dict:
    t, neg, head = run["t"], run["neg"], [r for r in run["pos"] if r[0] in ev.HEADLINE_GROUPS]
    out = {"overall": {k: run["metrics"][k] for k in ("recall", "precision", "f1", "false_positive_rate")}}
    out["fpr_synthetic"] = {}
    out["recall"] = {}
    for script in SCRIPTS:
        f = [r[3] >= t for r in neg if r[1] == script and r[0] == "synthetic"]
        out["fpr_synthetic"][script] = estimate(sum(f), len(f))
        rows = [r for r in head if r[5] == script]
        out["recall"][script] = estimate(sum(r[2] is not None and r[2] >= t for r in rows), len(rows))
    arabic = [r[3] >= t for r in neg if r[1] == "Arabic" and r[0] == "synthetic"]
    out["arabic_ratio_synthetic"] = {
        ref: bootstrap_contrast(arabic, [r[3] >= t for r in neg if r[1] == ref and r[0] == "synthetic"])
        for ref in ("Latin", "Cyrillic")
    }
    return out


def paired(cur: dict, new: dict) -> dict:
    t = cur["t"]
    out = {"fpr_synthetic": {}, "recall": {}}
    for script in SCRIPTS:
        idx = [i for i, r in enumerate(cur["neg"]) if r[1] == script and r[0] == "synthetic"]
        out["fpr_synthetic"][script] = mcnemar(
            [cur["neg"][i][3] >= t for i in idx], [new["neg"][i][3] >= t for i in idx]
        )
        idx = [i for i, r in enumerate(cur["pos"]) if r[0] in ev.HEADLINE_GROUPS and r[5] == script]
        out["recall"][script] = mcnemar(
            [cur["pos"][i][2] is not None and cur["pos"][i][2] >= t for i in idx],
            [new["pos"][i][2] is not None and new["pos"][i][2] >= t for i in idx],
        )
    return out


def hunch(entities: list[dict], dev_neg: list[dict], config: ScreeningConfig, cur: dict, new: dict) -> dict:
    """Does a full match on common tokens still score 1.0 under the weighted skeleton, and does this step fix it?"""
    filler = [{"id": f"f{i}", "names": [f"Zxq{i}rt Wvb{i}nk"]} for i in range(60)]
    pairs = {
        "particles and one consonant: 'Ali Abu' vs 'Abu Ali'": ("Ali Abu", "Abu Ali"),
        "particles around a rare name: 'Bin Al Khalidi' vs 'Khalidi'": ("Bin Al Khalidi", "Khalidi"),
        "common tokens, no particles: 'Mohammed Ahmed' vs 'Ahmed Mohammed'": ("Mohammed Ahmed", "Ahmed Mohammed"),
    }
    common = [{"id": f"c{i}", "names": [f"Mohammed Ahmed Q{i}"]} for i in range(40)]
    toy = {}
    for label, (query, name) in pairs.items():
        row = {}
        for tag, kwargs in (("current", {}), ("new", {"skeleton_particles": True})):
            screener = Screener.build(
                filler + [{"id": "t", "names": [name]}] + common,
                blend=config.blend,
                **kwargs,
            )
            hits = [c.skeleton for c in screener.components(query) if c.entity_id == "t"]
            row[tag] = round(hits[0], 3) if hits else None
        toy[label] = row
    # Real dev clean names: alerts whose best entity has a skeleton similarity of 1.0 under the current engine.
    t = cur["t"]
    summary = {}
    for script in SCRIPTS:
        idx = [i for i, q in enumerate(dev_neg) if q["script"] == script]
        full_sk = []
        for i in idx:
            comps = cur["neg_q"][i]["comps"]
            full_sk.append(any(c.skeleton >= 0.999 and combine_ok(c, config, t) for c in comps))
        summary[script] = {
            "clean_names": len(idx),
            "alert_with_skeleton_1.0_current": sum(full_sk),
            "alert_current": sum(cur["neg"][i][3] >= t for i in idx),
            "alert_new": sum(new["neg"][i][3] >= t for i in idx),
            "alert_with_skeleton_1.0_new": sum(
                any(c.skeleton >= 0.999 and combine_ok(c, config, t) for c in new["neg_q"][i]["comps"]) for i in idx
            ),
        }
    return {"toy_pairs_skeleton_similarity": toy, "real_dev_clean_names": summary}


def combine_ok(c: Components, config: ScreeningConfig, t: float) -> bool:
    return combine(c, config.skeleton_floor, config.skeleton_lift, 0) >= t


def overlap(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


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
    for label, kwargs in (("current", {}), ("new", {"skeleton_particles": True})):
        screeners = (
            Screener.build(held_entities, max_df=args.max_df, blend=config.blend, **kwargs),
            Screener.build(targets, max_df=args.max_df, blend=config.blend, **kwargs),
        )
        for split, (pos, neg) in sets.items():
            runs[f"{split}/{label}"] = score([dict(q) for q in pos], [dict(q) for q in neg], screeners, config)
            print(f"{split} {label} done", file=sys.stderr)

    report = {
        "engine": "frozen config, threshold 0.86, rarity-weighted skeleton",
        "change": "particles and one-consonant tokens left out of the skeleton",
    }
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
    report["hunch_full_match_on_common_tokens"] = hunch(
        entities, sets["dev"][1], config, runs["dev/current"], runs["dev/new"]
    )
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    for split in sets:
        for label in ("current", "new"):
            r = report[split][label]
            print(split, label, r["overall"])
            for key in ("fpr_synthetic", "recall"):
                print("  ", key, {s: (r[key][s]["value"], r[key][s]["ci95"]) for s in SCRIPTS})
            rl = r["arabic_ratio_synthetic"]["Latin"]
            print("   Arabic/Latin", rl["ratio"], rl["ratio_ci95"])
        print("  mcnemar", json.dumps(report[split]["mcnemar"]))
    print(json.dumps(report["hunch_full_match_on_common_tokens"], indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
