"""How much recall is lost because the entity was never retrieved, not because it scored low?

Run (after error_analysis.py and adversarial.py have written their reports):
    python analysis/retrieval_gap.py

A query reaches scoring only through the candidate index. When the true entity is not among the retrieved candidates
it gets no score at all (the "no signal" misses in the error analysis, and the adversarial slips whose true-entity
score is null). This forces the engine's own scoring code onto that one entity, by making the index return it, and
records the final score it would have had. The scoring code is the one screen() uses; only the retrieval step is
overridden, and only inside this script.

Reports: of the misses that had no score, how many would have alerted had the entity been retrieved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from screening import ScreeningEngine  # noqa: E402


def forced_score(
    engine: ScreeningEngine, name: str, dob: str | None, country: str | None, entity_id: str
) -> dict | None:
    """The engine's score for one entity as if retrieval had returned it."""
    index = engine.screener.index
    original = index.query
    index.query = lambda _name: [(entity_id, 0)]  # type: ignore[method-assign]
    try:
        got = common.score_query(engine, {"true_id": entity_id, "dob": dob, "country": country}, name=name)
    finally:
        index.query = original  # type: ignore[method-assign]
    return got["true"]


def tally(results: list[dict | None], threshold: float) -> dict:
    scored = [r for r in results if r is not None]
    return {
        "cases": len(results),
        "would_have_been_scored": len(scored),
        "would_have_alerted": sum(r["score"] >= threshold for r in scored),
        "would_have_scored_0.80_to_under_threshold": sum(0.80 <= r["score"] < threshold for r in scored),
        "still_under_0.80": sum(r["score"] < 0.80 for r in scored) + (len(results) - len(scored)),
    }


def main() -> None:
    out_path = common.OUT / "analysis_retrieval_gap.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; it is never overwritten")
    snapshot = common.read_snapshot(common.CURRENT_SNAPSHOT)
    result = {"snapshot": snapshot.version, "threshold": common.THRESHOLD}
    for label, holdout_path, kind, name in (
        ("second_holdout_stage2_engine", common.HOLDOUT_V2, "stage2", "v2_stage2"),
        ("third_holdout_shipped_engine", common.HOLDOUT_V3, "current", "v3_current"),
    ):
        queries, kept, _ = common.load_holdout(holdout_path)
        by_entity = {q["true_id"]: q for q in queries if q["kind"] == "positive"}
        engine, _ = common.build_engines(kind, snapshot, kept)
        report = common.json.loads((common.OUT / f"analysis_error_analysis_{name}.json").read_text())  # type: ignore[attr-defined]
        unretrieved = [f for f in report["false_negatives"] if f["final_score"] is None]
        forced = [
            forced_score(
                engine,
                f["query"],
                by_entity[f["entity_id"]]["dob"],
                by_entity[f["entity_id"]]["country"],
                f["entity_id"],
            )
            for f in unretrieved
        ]
        result[label] = {
            "headline_false_negatives": len(report["false_negatives"]),
            "false_negatives_with_no_score_at_all": tally(forced, common.THRESHOLD),
        }
        if kind == "current":
            adv = common.json.loads((common.OUT / "analysis_adversarial_v3_current.json").read_text())  # type: ignore[attr-defined]
            char = ("vowel_swap", "vowel_delete", "consonant_double", "consonant_delete")
            slips = [c for c in adv["all_slip_cases"] if c["family"] in char and c["variant_score"] is None]
            forced_adv = [
                forced_score(
                    engine,
                    c["variant"],
                    by_entity[c["entity_id"]]["dob"],
                    by_entity[c["entity_id"]]["country"],
                    c["entity_id"],
                )
                for c in slips
            ]
            result["adversarial_character_edit_slips_with_no_score_at_all"] = tally(forced_adv, common.THRESHOLD)
    common.write_json(out_path, result)
    print(common.json.dumps(result, indent=1))  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
