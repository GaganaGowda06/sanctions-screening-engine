"""Error analysis of the headline F1: every false negative and every false positive, classified.

Run (after the hold-out is frozen; reads it, never writes it):
    python analysis/error_analysis.py --holdout v2 --engine stage2     # the 0.777 figure
    python analysis/error_analysis.py --holdout v3 --engine current    # the shipped engine

Definitions fixed before the individual cases were read (the cutoffs came from the score distribution of the misses):

False negative: a headline positive whose true entity's final score is under the threshold (0.86).
  Severity, from the true entity's final score and an independent name similarity (analysis/textsim.py):
    dangerous   any of: the true entity was scored and its final score is 0.80 or more (within 0.06 of the threshold);
                the query and the nearest listed alias are near-identical (same-script, similarity 0.85 or more); every
                token of the query appears exactly in one listed alias (an acronym or a shortened name), with a
                token of four characters or more. The reasons are recorded, so each can be counted alone.
    borderline  final score 0.70 to under 0.80
    weak        final score under 0.70
    no signal   the true entity was never scored: no candidate retrieved, or under the matcher's minimum name score
  Cause, first rule that applies:
    non_transliterated_script  the query, or every alias kept, is in a script the transliteration does not cover
    org_translation            the entity is not a Person and the pair crosses scripts (Cyrillic or Arabic with Latin)
    missing_attribute          the query lacked a birth date or country, and giving the entity's own value alerts
    unvowelled_cross_script    the pair crosses between Arabic script and Latin (Arabic leaves short vowels unwritten)
    other                      everything else, described by how the query's tokens relate to the nearest alias
False positive: a headline negative that alerts, or a positive that alerts on some entity other than its own.
  Cause of the alert, from the alerting entity's score parts:
    attribute_boost   the score is over the threshold only because a matching birth date or country was added
    skeleton          the skeleton similarity lifted the name score over the threshold
    name_score        the name score alone is over the threshold
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import textsim  # noqa: E402

from profile_data import name_scripts  # noqa: E402  (script detection only, to say which script a name is in)
from screening import ScreeningEngine, query_script  # noqa: E402

DANGEROUS_SCORE = 0.80
BORDERLINE_SCORE = 0.70
NEAR_IDENTICAL = 0.85
CROSS_SCRIPT = {"cyr_to_latin", "latin_to_cyr", "ar_to_latin", "latin_to_ar"}
ARABIC_PAIRS = {"ar_to_latin", "latin_to_ar"}


def script_class(name: str) -> str:
    return query_script(name)


def is_other_script(name: str) -> bool:
    return bool(name_scripts(name) - {"Latin", "Cyrillic", "Arabic"})


def restored_alerts(engine: ScreeningEngine, q: dict, entity: dict) -> bool:
    """Would the query alert on its own entity if it carried the entity's birth date and country?"""
    dob = entity["birth_dates"][0] if entity["birth_dates"] else q["dob"]
    country = entity["countries"][0] if entity["countries"] else q["country"]
    got = common.score_query(engine, q, dob=dob, country=country)
    return got["true"] is not None and got["true"]["score"] >= got["threshold"]


def classify_false_negative(r: dict, q: dict, entity: dict, kept: list[str], engine: ScreeningEngine) -> dict:
    t = r["true"]
    alias, sim = textsim.best_alias(r["name"], kept)
    same_script = bool(alias) and script_class(alias) == script_class(r["name"]) and r["group"] == "variant"
    if t is None:
        severity = "no signal"
    elif t["score"] >= DANGEROUS_SCORE:
        severity = "dangerous"
    elif t["score"] >= BORDERLINE_SCORE:
        severity = "borderline"
    else:
        severity = "weak"
    reasons = []
    if t is not None and t["score"] >= DANGEROUS_SCORE:
        reasons.append("final score 0.80 or more")
    if same_script and sim >= NEAR_IDENTICAL:
        reasons.append("near-identical text")
    q_tokens = textsim.tokens(r["name"])
    if q_tokens and max(map(len, q_tokens)) >= 4 and any(set(q_tokens) <= set(textsim.tokens(n)) for n in kept):
        reasons.append("query tokens all appear in an alias")
    if reasons:
        severity = "dangerous"
    band = (
        None
        if not same_script
        else ("under 0.5" if sim < 0.5 else "0.5 to 0.85" if sim < NEAR_IDENTICAL else "0.85 or more")
    )
    missing = r["dob"] is None or r["country"] is None
    flips = missing and restored_alerts(engine, q, entity)
    if is_other_script(r["name"]) or (kept and all(is_other_script(n) for n in kept)):
        cause = "non_transliterated_script"
    elif entity["schema"] != "Person" and r["group"] in CROSS_SCRIPT:
        cause = "org_translation"
    elif flips:
        cause = "missing_attribute"
    elif r["group"] in ARABIC_PAIRS:
        cause = "unvowelled_cross_script"
    else:
        cause = "other"
    detail = textsim.token_relation(r["name"], alias) if alias and r["group"] == "variant" else None
    return {
        "group": r["group"],
        "query_script": r["script"],
        "schema": entity["schema"],
        "severity": severity,
        "cause": cause,
        "other_detail": detail if cause == "other" else None,
        "final_score": None if t is None else t["score"],
        "name_score": None if t is None else t["name"],
        "skeleton_score": None if t is None else t["skeleton"],
        "independent_similarity": round(sim, 4) if same_script else None,
        "similarity_band": band,
        "dangerous_reasons": reasons,
        "dob_missing": r["dob"] is None,
        "country_missing": r["country"] is None,
        "would_alert_with_attributes": bool(flips),
        "also_alerted_on_another_entity": r["best_other"] >= r["threshold"],
        "query": r["name"],
        "nearest_kept_alias": alias,
        "entity_id": r["true_id"],
    }


def false_positive_cause(a: dict, threshold: float) -> str:
    if a["score"] - max(a["dob_adj"], 0) - max(a["country_adj"], 0) < threshold <= a["score"] and (
        a["dob_adj"] > 0 or a["country_adj"] > 0
    ):
        return "attribute_boost"
    if a["skeleton_used"]:
        return "skeleton"
    return "name_score"


def classify_false_positive(r: dict, kind: str, names_of: dict[str, list[str]], imitated: dict) -> dict:
    top = (
        r["alerts"][0]
        if r["kind"] != "positive" or r["true"] is None
        else next((a for a in r["alerts"] if a["id"] != r["true_id"]), r["alerts"][0])
    )
    names = names_of.get(top["id"], [])
    q_tokens = set(textsim.tokens(r["name"]))
    shared = max((len(q_tokens & set(textsim.tokens(n))) for n in names), default=0)
    alias, sim = textsim.best_alias(r["name"], names)
    return {
        "kind": kind,
        "query_script": r["script"],
        "group": r["group"],
        "cause": false_positive_cause(top, r["threshold"]),
        "arabic_pair": top["arabic"],
        "score": top["score"],
        "name_score": top["name"],
        "skeleton_score": top["skeleton"],
        "exact_tokens_shared_with_nearest_alias": shared,
        "query_tokens": len(q_tokens),
        "independent_similarity": round(sim, 4),
        "n_entities_alerted": r["n_alerts"],
        "alerted_entity_is_the_one_imitated": kind == "near_miss" and imitated.get(r["index"]) == top["id"],
        "query": r["name"],
        "alerting_alias": alias,
    }


def tabulate(items: list[dict], *keys: str) -> dict:
    out: dict = defaultdict(int)
    for item in items:
        out["/".join(str(item[k]) for k in keys)] += 1
    return dict(sorted(out.items()))


def analyse(holdout_path: Path, engine_kind: str) -> dict:
    snap = common.read_snapshot(common.CURRENT_SNAPSHOT)
    queries, kept, digest = common.load_holdout(holdout_path)
    pos_engine, neg_engine = common.build_engines(engine_kind, snap, kept)
    rows = common.score_holdout(queries, pos_engine, neg_engine)
    metrics = common.headline_metrics(rows)
    by_index = {q["index"]: q for q in queries}
    imitated = {q["index"]: q.get("entity") for q in queries if q["kind"] == "near_miss"}

    head_pos = [r for r in rows if r["kind"] == "positive" and r["group"] in common.ev.HEADLINE_GROUPS]
    misses = [r for r in head_pos if not (r["true"] and r["true"]["score"] >= r["threshold"])]
    fns = [
        classify_false_negative(r, by_index[r["index"]], snap.screened[r["true_id"]], kept[r["true_id"]], pos_engine)
        for r in misses
    ]

    names_pos = {eid: kept.get(eid, e["names"]) for eid, e in snap.screened.items()}
    names_full = {eid: e["names"] for eid, e in snap.screened.items()}
    fps = []
    for r in rows:
        if r["kind"] == "positive":
            if (
                r["group"] in common.ev.HEADLINE_GROUPS
                and not (r["true"] and r["true"]["score"] >= r["threshold"])
                and r["best_other"] >= r["threshold"]
            ):
                fps.append(classify_false_positive(r, "wrong_entity_on_positive", names_pos, imitated))
        elif r["group"] in common.ev.HEADLINE_NEG_SCRIPTS and r["best_other"] >= r["threshold"]:
            fps.append(classify_false_positive(r, r["kind"], names_full, imitated))

    outside = [r for r in rows if r["kind"] == "positive" and r["group"] not in common.ev.HEADLINE_GROUPS]
    outside_hits = sum(1 for r in outside if r["true"] and r["true"]["score"] >= r["threshold"])
    all_pos = [r for r in rows if r["kind"] == "positive"]
    all_hits = sum(1 for r in all_pos if r["true"] and r["true"]["score"] >= r["threshold"])
    dangerous = sorted((f for f in fns if f["severity"] == "dangerous"),
                       key=lambda f: -(f["final_score"] if f["final_score"] is not None else 0))  # fmt: skip
    summary = {
        "holdout": holdout_path.name,
        "holdout_sha256": digest,
        "engine": engine_kind,
        "snapshot": snap.version,
        "headline": metrics,
        "false_negatives": {
            "count": len(fns),
            "by_cause": tabulate(fns, "cause"),
            "by_severity": tabulate(fns, "severity"),
            "by_cause_and_severity": tabulate(fns, "cause", "severity"),
            "by_group": tabulate(fns, "group"),
            "by_group_and_severity": tabulate(fns, "group", "severity"),
            "by_schema": tabulate(fns, "schema"),
            "other_cause_detail": tabulate([f for f in fns if f["cause"] == "other"], "other_detail"),
            "also_alerted_on_another_entity": sum(f["also_alerted_on_another_entity"] for f in fns),
            "dob_missing": sum(f["dob_missing"] for f in fns),
            "country_missing": sum(f["country_missing"] for f in fns),
            "would_alert_with_attributes": sum(f["would_alert_with_attributes"] for f in fns),
            "dangerous_total": len(dangerous),
            "dangerous_by_reason": {
                reason: sum(1 for f in dangerous if reason in f["dangerous_reasons"])
                for reason in ("final score 0.80 or more", "near-identical text", "query tokens all appear in an alias")
            },
            "dangerous_only_by_reason": tabulate([{"r": " + ".join(f["dangerous_reasons"])} for f in dangerous], "r"),
            "dangerous_by_group": tabulate(dangerous, "group"),
            "dangerous_by_schema": tabulate(dangerous, "schema"),
            "dangerous_not_scored_at_all": sum(1 for f in dangerous if f["final_score"] is None),
            "cause_by_group": tabulate(fns, "cause", "group"),
            "other_by_schema_and_band": tabulate(
                [f for f in fns if f["cause"] == "other"], "schema", "similarity_band"
            ),
            "variant_by_similarity_band_and_severity": tabulate(
                [f for f in fns if f["group"] == "variant"], "similarity_band", "severity"
            ),
            "dangerous_that_would_alert_with_attributes": sum(f["would_alert_with_attributes"] for f in dangerous),
        },
        "false_positives": {
            "count": len(fps),
            "by_kind": tabulate(fps, "kind"),
            "by_kind_and_script": tabulate(fps, "kind", "query_script"),
            "by_cause": tabulate(fps, "cause"),
            "by_cause_and_script": tabulate(fps, "cause", "query_script"),
            "by_kind_and_cause": tabulate(fps, "kind", "cause"),
            "arabic_pair_alerts": sum(f["arabic_pair"] for f in fps),
            "synthetic_by_exact_tokens_shared": tabulate(
                [f for f in fps if f["kind"] == "synthetic"], "exact_tokens_shared_with_nearest_alias"
            ),
            "near_miss_alerts_on_the_imitated_entity": sum(f["alerted_entity_is_the_one_imitated"] for f in fps),
        },
        "outside_the_headline": {
            "positives_not_in_headline": len(outside),
            "of_which_found": outside_hits,
            "all_positives": len(all_pos),
            "all_positives_found": all_hits,
            "recall_over_all_positives": all_hits / len(all_pos),
            "by_group": {
                g: {
                    "n": sum(1 for r in outside if r["group"] == g),
                    "found": sum(
                        1 for r in outside if r["group"] == g and r["true"] and r["true"]["score"] >= r["threshold"]
                    ),
                }
                for g in sorted({r["group"] for r in outside})
            },
        },
    }
    return {"summary": summary, "dangerous_false_negatives": dangerous, "false_negatives": fns, "false_positives": fps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", choices=("v2", "v3"), required=True)
    parser.add_argument("--engine", choices=("stage2", "current"), required=True)
    args = parser.parse_args()
    path = common.HOLDOUT_V2 if args.holdout == "v2" else common.HOLDOUT_V3
    result = analyse(path, args.engine)
    out = common.OUT / f"analysis_error_analysis_{args.holdout}_{args.engine}.json"
    if out.exists():
        sys.exit(f"{out.name} already exists; it is never overwritten")
    common.write_json(out, result)
    print(common.json.dumps(result["summary"], indent=1, sort_keys=True))  # type: ignore[attr-defined]
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
