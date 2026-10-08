"""How much of the headline F1 rests on the way the labels were made.

Run:
    python analysis/label_dependence.py

Measured on the second screening hold-out with the Stage 2 engine (the 0.777 configuration), and read-only:

  A. Attributes. Positives carry their own entity's birth date and country (each dropped with probability 0.3), and
     never a wrong one. F1 with both, with one, and with neither, using the same stored scores (evaluate_scoring's
     four modes). The near-miss negatives are labelled clean only because their birth date and country conflict with
     the listed person's: their false-alert rate with the attributes withheld shows how much of that label is the
     matcher's own penalty.
  B. Scope. The headline leaves out two positive groups and one negative group that the engine is known to fail on.
     The same F1 over every query of the hold-out.
  C. Weak positives. Positives are aliases of a listed entity, whatever kind of alias it is. Recall by how similar the
     query is to the nearest alias the list still holds (an independent character measure, analysis/textsim.py).
  E. What a hit is. A positive counts as found if its own entity is among the alerts, however many other entities also
     alert and whatever they rank. The extra alerts on hits are not counted anywhere in the headline precision.
  D. Duplicate entities. A wrong-entity alert on a positive counts as a false positive. Count the ones whose alerted
     entity holds an alias identical to one of the true entity's aliases (an unmerged duplicate in the list).
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import textsim  # noqa: E402
import tuning_transfer as tt  # noqa: E402

import evaluate_screening as es  # noqa: E402
from screening import ScreeningConfig  # noqa: E402

ev = common.ev


def attribute_modes(snapshot: common.Snapshot, queries: list[dict], kept: dict) -> dict:
    config = ScreeningConfig.load()
    held = tt.engine_variant(snapshot, config, kept, 1000, False).screener
    full = tt.engine_variant(snapshot, config, None, 1000, False).screener
    pos = [dict(q) for q in queries if q["kind"] == "positive"]
    neg = [dict(q) for q in queries if q["kind"] != "positive"]
    es.run_queries(held, pos, 0.5)  # the deployed candidate pool, as dev scored it
    es.run_queries(full, neg, 0.5)
    pos, neg = es.view(pos, 0.5), es.view(neg, 0.5)
    out = {}
    for mode, (use_dob, use_country) in ev.MODES.items():
        params = ev.Params(0.6, 0.2, 0)
        m = ev.metrics_at(
            ev.final_scores(pos, params, use_dob, use_country), ev.final_scores(neg, params, use_dob, use_country), 0.86
        )
        out[mode] = {
            k: m[k] for k in ("recall", "precision", "f1", "false_positive_rate", "fpr_synthetic", "fpr_near_miss")
        }
    return out


def scope(rows: list[dict]) -> dict:
    def measure(pos: list[dict], neg: list[dict]) -> dict:
        hit = [bool(r["true"]) and r["true"]["score"] >= r["threshold"] for r in pos]
        wrong = [(not h) and r["best_other"] >= r["threshold"] for h, r in zip(hit, pos, strict=True)]
        fa = sum(r["best_other"] >= r["threshold"] for r in neg)
        tp, wr = sum(hit), sum(wrong)
        precision, recall = tp / (tp + wr + fa), tp / len(pos)
        return {"positives": len(pos), "negatives": len(neg), "true_alerts": tp, "wrong_entity_alerts": wr,
                "false_alerts": fa, "recall": recall, "precision": precision,
                "f1": 2 * precision * recall / (precision + recall)}  # fmt: skip

    pos_all = [r for r in rows if r["kind"] == "positive"]
    neg_all = [r for r in rows if r["kind"] != "positive"]
    head_pos = [r for r in pos_all if r["group"] in ev.HEADLINE_GROUPS]
    head_neg = [r for r in neg_all if r["group"] in ev.HEADLINE_NEG_SCRIPTS]
    return {
        "headline": measure(head_pos, head_neg),
        "every_query": measure(pos_all, neg_all),
        "every_query_with_the_two_excluded_positive_groups_only": measure(pos_all, head_neg),
        "excluded_negatives_other_script_alert_rate": sum(
            r["best_other"] >= r["threshold"] for r in neg_all if r["group"] not in ev.HEADLINE_NEG_SCRIPTS
        )
        / sum(1 for r in neg_all if r["group"] not in ev.HEADLINE_NEG_SCRIPTS),
    }


def weak_positives(rows: list[dict], kept: dict) -> dict:
    bands: dict[str, list[bool]] = {
        "under 0.3": [],
        "0.3 to 0.5": [],
        "0.5 to 0.7": [],
        "0.7 to 0.85": [],
        "0.85 or more": [],
    }
    for r in rows:
        if r["kind"] != "positive" or r["group"] != "variant":
            continue
        _, sim = textsim.best_alias(r["name"], kept[r["true_id"]])
        key = ("under 0.3" if sim < 0.3 else "0.3 to 0.5" if sim < 0.5 else "0.5 to 0.7" if sim < 0.7
               else "0.7 to 0.85" if sim < 0.85 else "0.85 or more")  # fmt: skip
        bands[key].append(bool(r["true"]) and r["true"]["score"] >= r["threshold"])
    return {
        band: {"variant_positives": len(v), "found": sum(v), "recall": sum(v) / len(v) if v else None}
        for band, v in bands.items()
    }


def hit_definition(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS]
    hits = [r for r in pos if r["true"] and r["true"]["score"] >= r["threshold"]]
    extra = [r for r in hits if r["n_alerts"] > 1]
    extra_alerts = sum(r["n_alerts"] - 1 for r in extra)
    not_top = [r for r in hits if r["alerts"] and r["alerts"][0]["id"] != r["true_id"]]
    wrong = sum(1 for r in pos if r not in hits and r["best_other"] >= r["threshold"])
    fa = sum(
        r["best_other"] >= r["threshold"]
        for r in rows if r["kind"] != "positive" and r["group"] in ev.HEADLINE_NEG_SCRIPTS
    )  # fmt: skip
    tp = len(hits)
    entity_precision = tp / (tp + fa + wrong + extra_alerts)
    recall = tp / len(pos)
    return {
        "hits": tp,
        "hits_where_other_entities_also_alerted": len(extra),
        "extra_entity_alerts_on_hits": extra_alerts,
        "hits_where_another_entity_scored_higher_than_the_true_one": len(not_top),
        "precision_per_customer_alert_as_in_the_headline": tp / (tp + fa + wrong),
        "precision_if_every_extra_entity_alert_counted_as_false": entity_precision,
        "f1_if_every_extra_entity_alert_counted_as_false": 2 * entity_precision * recall / (entity_precision + recall),
    }


def duplicates(rows: list[dict], snapshot: common.Snapshot) -> dict:
    wrong = [
        r for r in rows
        if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS and r["best_other"] >= r["threshold"]
        and not (r["true"] and r["true"]["score"] >= r["threshold"])
    ]  # fmt: skip
    same_alias, very_similar = 0, 0
    for r in wrong:
        other = next((a for a in r["alerts"] if a["id"] != r["true_id"]), None)
        if other is None:
            continue
        true_names = {" ".join(sorted(textsim.tokens(n))) for n in snapshot.screened[r["true_id"]]["names"]}
        other_names = snapshot.screened[other["id"]]["names"]
        if any(" ".join(sorted(textsim.tokens(n))) in true_names for n in other_names):
            same_alias += 1
        elif (
            max(
                (textsim.similarity(a, b) for a in other_names for b in snapshot.screened[r["true_id"]]["names"]),
                default=0,
            )
            >= 0.9
        ):
            very_similar += 1
    return {
        "wrong_entity_alerts_on_positives": len(wrong),
        "alerted_entity_shares_an_identical_alias": same_alias,
        "alerted_entity_has_an_alias_at_least_0.9_similar": very_similar,
    }


def main() -> None:
    out_path = common.OUT / "analysis_label_dependence.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; it is never overwritten")
    snapshot = common.read_snapshot(common.CURRENT_SNAPSHOT)
    queries, kept, digest = common.load_holdout(common.HOLDOUT_V2)
    pos_engine, neg_engine = common.build_engines("stage2", snapshot, kept)
    rows = common.score_holdout(queries, pos_engine, neg_engine)
    result = {
        "holdout": common.HOLDOUT_V2.name,
        "holdout_sha256": digest,
        "engine": "stage2",
        "snapshot": snapshot.version,
        "A_attribute_modes": attribute_modes(snapshot, queries, kept),
        "B_scope": scope(rows),
        "C_recall_by_similarity_of_the_variant_to_the_nearest_kept_alias": weak_positives(rows, kept),
        "D_possible_unmerged_duplicates": duplicates(rows, snapshot),
        "E_what_a_hit_is": hit_definition(rows),
        "positives_by_group": dict(Counter(q["group"] for q in queries if q["kind"] == "positive")),
        "negatives_by_kind_and_group": dict(
            Counter(f"{q['kind']}/{q['group']}" for q in queries if q["kind"] != "positive")
        ),
    }
    common.write_json(out_path, result)
    print(common.json.dumps(result, indent=1))  # type: ignore[attr-defined]
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
