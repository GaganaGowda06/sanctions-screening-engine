"""Why does the trained matcher do badly at screening? Swap parts of its coefficients and see.

On the Pairs data the model beats the hand-tuned engine, but on screening queries it does much worse. The
name scores look fine; the suspect is the learned weights on birth date, country and identifier, because in
the Pairs data a missing attribute goes with matches (a pair with no country on either side is 85% matches,
one whose countries agree 66%), so the model gives an agreeing country a negative weight.

Four variants of the same model, scored on the first 600 dev positives and 600 dev negatives of the screening
set, thresholds chosen on that same sample, so the figures compare the variants with one another and are
not a fair estimate of performance:
  as_trained           the fitted coefficients
  country_agree_zero   only the country_agree weight set to zero
  names_only           every attribute weight set to zero
  hand_set_attributes  attribute weights set by hand to the signs and rough sizes the hand-tuned engine uses
Neither final hold-out is used.

Run:
    python src/diagnose_screening.py
Writes reports/pairs_diagnostic.json.
"""

import argparse
import json
import multiprocessing
import platform
import sys
from pathlib import Path

import compare_matchers as cm
import evaluate_scoring as ev
from entities import load_named_entities
from evaluate_screening import build_query_sets
from idf_table import load_table
from matcher_model import MODEL_PATH
from profile_data import DEFAULT_INPUT
from scoring import Screener

REPORT = Path(__file__).resolve().parent.parent / "reports" / "pairs_diagnostic.json"
BEST_KEYS = ("recall", "precision", "f1", "false_positive_rate")
TEN_KEYS = ("recall", "precision", "top1_accuracy", "false_positive_rate")
ATTRIBUTE_WEIGHTS = (
    "birth_agree", "birth_close", "birth_conflict", "country_agree", "country_conflict",
    "identifier_agree", "identifier_conflict",
)  # fmt: skip


def variants(model: dict) -> dict[str, dict]:
    def changed(**weights: float) -> dict:
        copy = json.loads(json.dumps(model))
        copy["coefficients"].update(weights)
        return copy

    return {
        "as_trained": model,
        "country_agree_zero": changed(country_agree=0.0),
        "names_only": changed(**dict.fromkeys(ATTRIBUTE_WEIGHTS, 0.0)),
        "hand_set_attributes": changed(
            birth_agree=0.5, birth_close=0.0, birth_conflict=-4.0, country_agree=0.3, country_conflict=-2.0,
            identifier_agree=0.5, identifier_conflict=-2.0,
        ),  # fmt: skip
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--queries", type=int, default=600)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    model = json.loads(args.model.read_text())
    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    positives, negatives, held = build_query_sets(entities, targets)
    df, documents = load_table()
    held_screener = Screener.build(
        [{**e, "names": held.get(e["id"], e["names"])} for e in targets], idf_table=(df, documents)
    )
    full_screener = Screener.build(targets, idf_table=(df, documents))
    pos = [q for q in positives if q["split"] == "dev" and q["group"] in ev.HEADLINE_GROUPS][: args.queries]
    neg = [q for q in negatives if q["split"] == "dev"][: args.queries]

    report = {"queries": {"positive": len(pos), "negative": len(neg)}, "variants": {}}
    for label, variant in variants(model).items():
        context = multiprocessing.get_context("spawn")
        with context.Pool(args.workers, cm._init, (str(args.input), held, df, documents, variant)) as pool:
            pos_rows = cm.trained_rows(pool, held_screener, pos, True)
            neg_rows = cm.trained_rows(pool, full_screener, neg, False)
        best = cm.choose_threshold_for_f1(pos_rows, neg_rows)
        at_best = ev.metrics_at(pos_rows, neg_rows, best)
        ten = cm.choose_threshold_for_fpr(pos_rows, neg_rows, 0.10)
        at_ten = ev.metrics_at(pos_rows, neg_rows, ten)
        report["variants"][label] = {
            "coefficients": {k: round(v, 3) for k, v in variant["coefficients"].items()},
            "best_f1": {"threshold": best, **{k: at_best[k] for k in BEST_KEYS}},
            "at_false_positive_rate_up_to_10_percent": {"threshold": ten, **{k: at_ten[k] for k in TEN_KEYS}},
        }
        print(f"{label}: recall at FPR<=10% {at_ten['recall']}", file=sys.stderr)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
