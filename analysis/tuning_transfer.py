"""Does what was tuned on dev transfer to a hold-out that tuning never touched?

Run:
    python analysis/tuning_transfer.py

What was tuned, and on what (the project notes, README, reports/stage2_screening.json, reports/stage3_skeleton_idf.json):
  floor, lift, blend, global threshold   grid search on dev (step 4); frozen at 0.6, 0.2, 0.5, 0.86
  per-script thresholds                  chosen on dev (0.86 / 0.88 / 0.86); kept for reference and switched off
  rarity-weighted skeleton               Stage 3, judged on dev and test; kept
  max_df 1000                            the candidate index's frequent-token cutoff, set in Stage 2

The slice used here is the second screening hold-out (holdout/screening_holdout_v2.json). It was built from entities
used in no dev, test or earlier hold-out query, and it was read for reporting once (its F1 of 0.777) and never used to
choose any parameter. Stage 3 chose the rarity skeleton on dev and test and read the THIRD hold-out only to report it,
so the second hold-out is untouched by that decision too. Reading it again here is analysis, not tuning: nothing found
here changes a parameter, and it is stated that this hold-out is now read a second time.

Numbers it produces:
  1. The frozen configuration's F1, recall and per-script alert rates on the slice, against the dev (tuned) and test
     numbers.
  2. The same grid dev searched (75 parameter sets x 50 thresholds) run on the slice: where the frozen choice ranks, how
     far it is from the best the slice would have picked, and whether dev's preference order carries over.
  3. Per-script thresholds against the single threshold on the slice.
  4. The rarity skeleton on/off on the slice, against its dev and test effect.
  5. max_df 250, 500, 1000, 2000, 5000 on the slice.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

import evaluate_screening as es  # noqa: E402
from list_gate import Snapshot  # noqa: E402
from scoring import Screener  # noqa: E402
from screening import ScreeningConfig, ScreeningEngine  # noqa: E402

ev = common.ev
FROZEN = (0.5, 0.6, 0.2)  # blend, floor, lift
DEPLOYED_BLEND = 0.5
MAX_DFS = (250, 500, 1000, 2000, 5000)


def engine_variant(
    snapshot: Snapshot, config: ScreeningConfig, held: dict | None, max_df: int, skeleton_idf: bool
) -> ScreeningEngine:
    entities = [
        {
            "id": e["id"],
            "names": (held or {}).get(e["id"], e["names"]),
            "birth_dates": e["birth_dates"],
            "countries": e["countries"],
        }
        for e in snapshot.screened.values()
    ]
    return ScreeningEngine(
        Screener.build(entities, max_df=max_df, blend=config.blend, skeleton_idf=skeleton_idf), config
    )


def per_script(rows: list[dict]) -> dict:
    out = {}
    for script in common.ev.HEADLINE_NEG_SCRIPTS:
        neg = [r for r in rows if r["kind"] != "positive" and r["group"] == script]
        pos = [
            r for r in rows if r["kind"] == "positive" and r["group"] in ev.HEADLINE_GROUPS and r["script"] == script
        ]
        out[script] = {
            "alert_rate_on_clean_names": sum(r["best_other"] >= r["threshold"] for r in neg) / len(neg),
            "clean_names": len(neg),
            "recall": sum(bool(r["true"]) and r["true"]["score"] >= r["threshold"] for r in pos) / len(pos),
            "positives": len(pos),
        }
    return out


def run_variant(
    snapshot: Snapshot, queries: list[dict], kept: dict, config: ScreeningConfig, max_df: int, skeleton_idf: bool
) -> dict:
    pos = engine_variant(snapshot, config, kept, max_df, skeleton_idf)
    neg = engine_variant(snapshot, config, None, max_df, skeleton_idf)
    rows = common.score_holdout(queries, pos, neg)
    m = common.headline_metrics(rows)
    return {"max_df": max_df, "skeleton_idf": skeleton_idf, **m, "per_script": per_script(rows)}


def grid(snapshot: Snapshot, queries: list[dict], kept: dict) -> dict:
    """The dev grid on the slice, run as dev ran it (evaluate_screening.py): the Stage 2 engine (no rarity skeleton),
    each blend scored on its own candidate pool, then floor x lift x threshold for that pool."""
    config = ScreeningConfig.load()
    held = engine_variant(snapshot, config, kept, 1000, False).screener
    full = engine_variant(snapshot, config, None, 1000, False).screener
    pos = [dict(q) for q in queries if q["kind"] == "positive" and q["group"] in ev.HEADLINE_GROUPS]
    neg = [dict(q) for q in queries if q["kind"] != "positive" and q["group"] in ev.HEADLINE_NEG_SCRIPTS]
    results = []
    for blend in ev.BLENDS:
        es.run_queries(held, pos, blend)
        es.run_queries(full, neg, blend)
        pos_view, neg_view = es.view(pos, blend), es.view(neg, blend)
        for floor, lift in itertools.product(ev.SKELETON_FLOORS, ev.SKELETON_LIFTS):
            params = ev.Params(floor, lift, 0)
            pos_rows = ev.final_scores(pos_view, params, True, True)
            neg_rows = ev.final_scores(neg_view, params, True, True)
            for t in ev.THRESHOLDS:
                m = ev.metrics_at(pos_rows, neg_rows, t)
                results.append((m["f1"], (blend, floor, lift), t, m["recall"], m["precision"]))
    deployed = (es.view(pos, DEPLOYED_BLEND), es.view(neg, DEPLOYED_BLEND))
    return {"results": results, "deployed_pos": deployed[0], "deployed_neg": deployed[1]}


def main() -> None:
    out_path = common.OUT / "analysis_tuning_transfer.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; it is never overwritten")
    snapshot = common.read_snapshot(common.CURRENT_SNAPSHOT)
    queries, kept, digest = common.load_holdout(common.HOLDOUT_V2)
    config = ScreeningConfig.load()
    reported = json.loads((common.ROOT / "reports" / "stage2_screening.json").read_text())
    first = json.loads((common.ROOT / "reports" / "final_evaluation.json").read_text())

    # 2. The dev grid on the slice.
    g = grid(snapshot, queries, kept)
    ranked = sorted(g["results"], key=lambda r: (-r[0], -r[2]))
    frozen = next(r for r in g["results"] if r[1] == FROZEN and r[2] == 0.86)
    best = ranked[0]
    rank = 1 + sum(r[0] > frozen[0] for r in g["results"])
    within = sum(r[0] >= best[0] - 0.005 for r in g["results"])

    def best_by(index: int) -> dict:
        out: dict = {}
        for r in g["results"]:
            key = str(r[1][index])
            out[key] = max(out.get(key, -1.0), r[0])
        return {k: round(v, 4) for k, v in sorted(out.items())}

    dev_marginals = reported["tuned_on_deployed_candidate_pool"]["dev_best_f1_by_parameter_value"]
    grid_summary = {
        "configurations": len(g["results"]),
        "frozen_choice": {
            "blend": 0.5,
            "floor": 0.6,
            "lift": 0.2,
            "threshold": 0.86,
            "f1_on_slice": round(frozen[0], 4),
        },
        "best_on_slice": {
            "blend": best[1][0],
            "floor": best[1][1],
            "lift": best[1][2],
            "threshold": best[2],
            "f1": round(best[0], 4),
        },
        "regret_f1": round(best[0] - frozen[0], 4),
        "rank_of_frozen_choice": rank,
        "configurations_within_0.005_of_best": within,
        "slice_best_f1_by_blend": best_by(0),
        "slice_best_f1_by_floor": best_by(1),
        "slice_best_f1_by_lift": best_by(2),
        "dev_best_f1_by_blend": dev_marginals["blend"],
        "dev_best_f1_by_floor": dev_marginals["skeleton_floor"],
        "dev_best_f1_by_lift": dev_marginals["skeleton_lift"],
        "f1_by_threshold_at_frozen_params": {
            str(t): round(next(r[0] for r in g["results"] if r[1] == FROZEN and r[2] == t), 4)
            for t in (0.78, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92)
        },
    }

    # 3. Per-script thresholds against the single threshold, on the slice (Stage 2 engine, frozen parameters).
    frozen_params = ev.Params(FROZEN[1], FROZEN[2], 0)
    pos_rows = ev.final_scores(g["deployed_pos"], frozen_params, True, True)
    neg_rows = ev.final_scores(g["deployed_neg"], frozen_params, True, True)
    thresholds = dict(config.thresholds)
    single = ev.metrics_at(pos_rows, neg_rows, 0.86)
    scripted = ev.metrics_at(pos_rows, neg_rows, thresholds)
    per_script_thresholds = {
        "thresholds_chosen_on_dev": thresholds,
        "single_threshold": {
            k: single[k]
            for k in ("recall", "precision", "f1", "false_positive_rate", "fpr_by_script", "recall_by_script")
        },
        "per_script_thresholds": {
            k: scripted[k]
            for k in ("recall", "precision", "f1", "false_positive_rate", "fpr_by_script", "recall_by_script")
        },
        "dev": reported["dev_per_script"],
        "test": {
            "single": reported["fpr_by_script_test"]["single_threshold"],
            "per_script": reported["fpr_by_script_test"]["per_script"],
        },
    }

    # 4 and 5. Rarity skeleton on/off and max_df, on the slice.
    variants = {}
    for skeleton_idf in (False, True):
        for max_df in MAX_DFS:
            key = f"{'rarity_skeleton' if skeleton_idf else 'stage2_skeleton'}_max_df_{max_df}"
            variants[key] = run_variant(snapshot, queries, kept, config, max_df, skeleton_idf)
            print(key, round(variants[key]["f1"], 4), file=sys.stderr, flush=True)

    result = {
        "slice": {"holdout": common.HOLDOUT_V2.name, "sha256": digest, "snapshot": snapshot.version},
        "tuned_numbers": {
            "dev_f1_at_frozen_config": reported["tuned_on_deployed_candidate_pool"]["dev_f1"],
            "test_f1": reported["test_overall_single_threshold"]["f1"],
            "test_recall": reported["test_overall_single_threshold"]["recall"],
            "test_fpr_by_script": reported["test_overall_single_threshold"]["fpr_by_script"],
            "test_recall_by_script": reported["test_overall_single_threshold"]["recall_by_script"],
            "dev_per_script_at_global": reported["dev_per_script"],
            "first_holdout_f1": first["overall"]["f1"]["value"],
        },
        "grid_on_slice": grid_summary,
        "per_script_thresholds": per_script_thresholds,
        "variants_on_slice": variants,
    }
    common.write_json(out_path, result)
    print(json.dumps({k: result[k] for k in ("tuned_numbers", "grid_on_slice")}, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
