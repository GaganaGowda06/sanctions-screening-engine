"""Stage 3 step 4: hold back skeleton matches that rest only on common tokens. Dev, then test. No hold-out.

Part 1, measure first (dev, current engine). Of the false alerts on clean names, how many are full-name
collisions (every skeleton token of the query has a partner in the listed name) and how many are partial
matches, by query script; how many needed the skeleton to reach the threshold; and how many of the
skeleton-driven ones rest only on common tokens (skeleton token in at least COMMON_DF listed entities).

Part 2, the fix. When the skeleton carried a match (without it the entity is under the threshold) and every
matched skeleton token is common, the match goes to a "needs secondary identifier" tier instead of a full
alert. If the query supplies a birth date or a country, it is scored as normal. Nothing is retuned.

Reported on dev and on test: full-alert false positive rate per script (synthetic clean names), full-alert
recall per script, how many true matches land in the downgraded tier, the Arabic to Latin and Arabic to
Cyrillic ratios, and an exact McNemar test of the full-alert outcome against the current engine on the same
queries. Acceptance: Arabic full-alert FPR clearly down (new interval wholly under the old, or at least
halved), Arabic full-alert recall at or above 65%, Latin and Cyrillic rates and recall unchanged within noise.

Run:
    python src/evaluate_common_tier.py
Writes reports/stage3_common_tier.json.
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
from evaluate_skeleton_particles import mcnemar, overlap
from measure_alert_disparity import bootstrap_contrast
from profile_data import DEFAULT_INPUT
from scoring import Components, Screener, combine, needs_secondary_identifier, scoring_form, skeleton_form
from screening import CONFIG_PATH, ScreeningConfig

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "stage3_common_tier.json"
SCRIPTS = ("Latin", "Cyrillic", "Arabic")
COMMON_DF = 50


def scored(q: dict, config: ScreeningConfig) -> list[tuple[Components, float, bool]]:
    """(components, final score, held back for want of a secondary identifier) for each candidate."""
    supplied = bool(q["dob"] or q["country"])
    out = []
    for c in q["comps"]:
        score = combine(c, config.skeleton_floor, config.skeleton_lift, 0)
        down = needs_secondary_identifier(
            c, config.global_threshold, supplied, 0,
            skeleton_floor=config.skeleton_floor, skeleton_lift=config.skeleton_lift,
        )  # fmt: skip
        out.append((c, score, down))
    return out


def outcome_rows(queries: list[dict], config: ScreeningConfig) -> dict:
    """Per query: current alert, full alert, and for positives true-match flags, in both engines."""
    t = config.global_threshold
    rows = []
    for q in queries:
        cands = scored(q, config)
        cur_other = max((s for c, s, _ in cands if c.entity_id != q["true_id"]), default=0.0)
        full_other = max((s for c, s, d in cands if c.entity_id != q["true_id"] and not d), default=0.0)
        true = next(((s, d) for c, s, d in cands if c.entity_id == q["true_id"]), None)
        rows.append({
            "kind": q["kind"], "group": q["group"], "script": q["script"],
            "cur_other": cur_other >= t, "full_other": full_other >= t,
            "cur_true": true is not None and true[0] >= t,
            "full_true": true is not None and true[0] >= t and not true[1],
            "tier_true": true is not None and true[0] >= t and true[1],
        })  # fmt: skip
    return {"rows": rows}


def collisions(negatives: list[dict], full: Screener, config: ScreeningConfig) -> dict:
    t = config.global_threshold
    out = {}
    for script in SCRIPTS:
        slot = {"clean_names": 0, "alerts": 0, "skeleton_carried": 0, "full_name_collision": 0, "partial_match": 0,
                "collision_common_only": 0, "partial_common_only": 0}  # fmt: skip
        for q in (q for q in negatives if q["script"] == script):
            slot["clean_names"] += 1
            cands = scored(q, config)
            if not cands:
                continue
            c, score, _ = max(cands, key=lambda x: x[1])
            if score < t:
                continue
            slot["alerts"] += 1
            if min(1.0, max(0.0, c.names[0] + c.dob + c.country)) >= t:
                continue  # the name alone carried it
            slot["skeleton_carried"] += 1
            pairs, n = full.skeleton_match(skeleton_form(scoring_form(q["name"])), full.records[c.entity_id])
            kind = "full_name_collision" if n and len(pairs) == n else "partial_match"
            slot[kind] += 1
            slot["collision_common_only" if kind == "full_name_collision" else "partial_common_only"] += c.common_only
        out[script] = slot
    return out


def summarise(rows: list[dict]) -> dict:
    out = {"fpr_synthetic": {}, "recall": {}}
    for engine, alert, true in (("current", "cur_other", "cur_true"), ("full_tier", "full_other", "full_true")):
        fpr, recall = {}, {}
        for script in SCRIPTS:
            neg = [r[alert] for r in rows if r["kind"] == "synthetic" and r["script"] == script]
            fpr[script] = estimate(sum(neg), len(neg))
            pos = [r[true] for r in rows if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS
                   and r["script"] == script]  # fmt: skip
            recall[script] = estimate(sum(pos), len(pos))
        out["fpr_synthetic"][engine], out["recall"][engine] = fpr, recall
        arabic = [r[alert] for r in rows if r["kind"] == "synthetic" and r["script"] == "Arabic"]
        out.setdefault("arabic_ratio_synthetic", {})[engine] = {
            ref: bootstrap_contrast(arabic, [r[alert] for r in rows if r["kind"] == "synthetic" and r["script"] == ref])
            for ref in ("Latin", "Cyrillic")
        }
    head = [r for r in rows if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS]
    out["true_matches_in_downgraded_tier"] = {
        s: {"tier": sum(r["tier_true"] for r in head if r["script"] == s), "of_true_alerts":
            sum(r["cur_true"] for r in head if r["script"] == s)} for s in SCRIPTS
    }  # fmt: skip
    out["clean_names_held_back_entirely"] = {
        s: sum(r["cur_other"] and not r["full_other"] for r in rows if r["kind"] != "positive" and r["script"] == s)
        for s in SCRIPTS
    }
    return out


def paired(rows: list[dict]) -> dict:
    out = {"fpr_synthetic": {}, "recall": {}}
    for s in SCRIPTS:
        neg = [r for r in rows if r["kind"] == "synthetic" and r["script"] == s]
        out["fpr_synthetic"][s] = mcnemar([r["cur_other"] for r in neg], [r["full_other"] for r in neg])
        pos = [r for r in rows if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS and r["script"] == s]
        out["recall"][s] = mcnemar([r["cur_true"] for r in pos], [r["full_true"] for r in pos])
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
    held = Screener.build(
        [{**e, "names": kept.get(e["id"], e["names"])} for e in targets],
        max_df=args.max_df, blend=config.blend, common_df=COMMON_DF,
    )  # fmt: skip
    full = Screener.build(targets, max_df=args.max_df, blend=config.blend, common_df=COMMON_DF)
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
    report: dict = {"common_df": COMMON_DF, "threshold": config.global_threshold}
    verdict = {}
    for split, (pos, neg) in sets.items():
        for q in pos:
            q["comps"] = held.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=(config.blend,))
        for q in neg:
            q["comps"] = full.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=(config.blend,))
        rows = outcome_rows(pos + neg, config)["rows"]
        if split == "dev":
            report["part1_collisions_dev_current_engine"] = collisions(neg, full, config)
            report["supplied_birth_date_or_country_share_dev_negatives"] = round(
                sum(bool(q["dob"] or q["country"]) for q in neg) / len(neg), 3
            )
        summary = summarise(rows)
        report[split] = {**summary, "mcnemar_current_vs_full_tier": paired(rows)}
        a, b = summary["fpr_synthetic"]["current"], summary["fpr_synthetic"]["full_tier"]
        ra, rb = summary["recall"]["current"], summary["recall"]["full_tier"]
        checks = {
            "arabic_fpr_clearly_down": b["Arabic"]["ci95"][1] < a["Arabic"]["ci95"][0]
            or b["Arabic"]["value"] <= a["Arabic"]["value"] / 2,
            "arabic_full_alert_recall_at_least_65": rb["Arabic"]["value"] >= 0.65,
        }
        for s in ("Latin", "Cyrillic"):
            checks[f"{s}_fpr_unchanged"] = overlap(a[s]["ci95"], b[s]["ci95"])
            checks[f"{s}_recall_unchanged"] = overlap(ra[s]["ci95"], rb[s]["ci95"])
        verdict[split] = {**checks, "all_pass": all(checks.values())}
        print(f"{split} done", file=sys.stderr)
    report["acceptance"] = verdict
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    print(json.dumps(report["part1_collisions_dev_current_engine"], indent=1))
    print("supplied share", report["supplied_birth_date_or_country_share_dev_negatives"])
    for split in sets:
        r = report[split]
        for engine in ("current", "full_tier"):
            print(split, engine)
            print(
                "  fpr",
                {s: (r["fpr_synthetic"][engine][s]["value"], r["fpr_synthetic"][engine][s]["ci95"]) for s in SCRIPTS},
            )
            print("  recall", {s: (r["recall"][engine][s]["value"], r["recall"][engine][s]["ci95"]) for s in SCRIPTS})
            x = r["arabic_ratio_synthetic"][engine]["Latin"]
            print("  Arabic/Latin", x["ratio"], x["ratio_ci95"])
        print("  tier", r["true_matches_in_downgraded_tier"], r["clean_names_held_back_entirely"])
        print("  mcnemar", json.dumps(r["mcnemar_current_vs_full_tier"]))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
