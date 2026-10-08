"""Why the gate's F1 (0.8010) differs from the frozen hold-out F1 (0.777): the same hold-out under two engines.

Run:
    python src/f1_provenance.py

The project quotes several F1 and burden figures, and they are not comparable without knowing which hold-out and which
engine is behind each:

  0.776  the Stage 2 engine on the FIRST hold-out        (reports/final_evaluation.json)
  0.777  the Stage 2 engine on the SECOND hold-out       (reports/screening_holdout_v2_evaluation.json)
  0.8010 the CURRENT engine on the THIRD hold-out        (the list-update gate, reports/stage4_list_update_gate_*.json)

Two things differ between 0.777 and 0.8010: the hold-out (a different sample of queries and entities) and the engine
(Stage 3 added the rarity-weighted skeleton). This script separates them on the third hold-out, the only one still used,
as a fixed regression benchmark, never to tune anything: the Stage 2 engine (rarity-weighted skeleton off) and the
current engine are scored on the same queries with the same headline definition. The first two hold-outs are not read
again; their figures are taken from the reports that were written when they were scored once.

Writes reports/stage4_f1_provenance.json. Aggregate numbers only.
"""

import json
import sys

import evaluate_scoring as ev
from list_gate import (
    CURRENT_SNAPSHOT,
    HOLDOUT,
    REPORT_DIR,
    Snapshot,
    engine_for,
    f1_from,
    headline_arrays,
    load_holdout,
    read_snapshot,
    screen_set,
)
from scoring import Screener
from screening import ScreeningConfig, ScreeningEngine

OUT = REPORT_DIR / "stage4_f1_provenance.json"


def stage2_engine(snapshot: Snapshot, config: ScreeningConfig, held: dict | None = None) -> ScreeningEngine:
    """The engine as it was at the end of Stage 2: no rarity-weighted skeleton."""
    held = held or {}
    entities = [
        {
            "id": e["id"],
            "names": held.get(e["id"], e["names"]),
            "birth_dates": e["birth_dates"],
            "countries": e["countries"],
        }
        for e in snapshot.screened.values()
    ]
    screener = Screener.build(entities, max_df=1000, blend=config.blend, skeleton_idf=False)
    return ScreeningEngine(screener, config)


def score(queries: list[dict], held_engine: ScreeningEngine, full_engine: ScreeningEngine) -> list[dict]:
    results: list[dict | None] = [None] * len(queries)
    for positive, engine in ((True, held_engine), (False, full_engine)):
        idx = [i for i, q in enumerate(queries) if (q["kind"] == "positive") == positive]
        out = screen_set(engine, [queries[i] for i in idx], [queries[i]["true_id"] for i in idx])
        for i, r in zip(idx, out, strict=True):
            results[i] = r
    return results  # type: ignore[return-value]


def headline(queries: list[dict], results: list[dict]) -> dict:
    a = headline_arrays(queries, results, results)
    n_pos = len(a["pos_tp_old"])
    tp, wrong, fa = a["pos_tp_old"].sum(), a["pos_wrong_old"].sum(), a["neg_old"].sum()
    n_ar = int(a["arabic"].sum())
    return {
        "headline_positives": int(n_pos),
        "headline_negatives": int(len(a["neg_old"])),
        "recall": float(tp / n_pos),
        "precision": float(tp / (tp + fa + wrong)),
        "f1": f1_from(tp, wrong, fa, n_pos),
        "arabic_burden": float((a["neg_old"] * a["arabic"]).sum() / n_ar) if n_ar else None,
    }


def main() -> int:
    config = ScreeningConfig.load()
    snapshot = read_snapshot(CURRENT_SNAPSHOT)
    queries, kept = load_holdout(HOLDOUT)
    held = {q["true_id"]: kept[q["true_id"]] for q in queries if q["kind"] == "positive"}
    stage2 = headline(queries, score(queries, stage2_engine(snapshot, config, held), stage2_engine(snapshot, config)))
    current = headline(queries, score(queries, engine_for(snapshot, config, held), engine_for(snapshot, config)))
    first = json.loads((REPORT_DIR / "final_evaluation.json").read_text())["overall"]["f1"]["value"]
    second = json.loads((REPORT_DIR / "screening_holdout_v2_evaluation.json").read_text())["overall"]["f1"]["value"]
    out = {
        "what_this_is": "the same third hold-out under the Stage 2 engine and the current engine; the figures for the "
        "first two hold-outs are copied from the reports written when they were scored once",
        "snapshot": snapshot.version,
        "third_holdout": {"stage2_engine": stage2, "current_engine": current},
        "first_holdout_stage2_engine": {"report": "final_evaluation.json", "f1": first},
        "second_holdout_stage2_engine": {"report": "screening_holdout_v2_evaluation.json", "f1": second},
        "reading": "the change from 0.777 to 0.8010 is the sum of a hold-out effect (the same Stage 2 engine on a "
        "different sample of queries and entities) and an engine effect (the Stage 3 rarity-weighted skeleton). Both "
        "engines here run on today's snapshot, not the one the earlier hold-outs were scored on, and the third "
        "hold-out is scored over its full 2,000 headline positives; the gate's 0.8010 is over the positives that "
        "survive resolution",
        "hold_out_effect_stage2_engine": stage2["f1"] - second,
        "engine_effect_on_third_holdout": current["f1"] - stage2["f1"],
        "headline_definition": f"positive groups {sorted(ev.HEADLINE_GROUPS)}; negatives in Latin, Cyrillic, Arabic",
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
