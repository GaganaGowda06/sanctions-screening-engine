"""Compare the trained matcher with the hand-tuned engine, on the Pairs test set and on the old screening test set.

Both systems use the same rarity table (the deployment list's). Neither final hold-out is read.

Part A, Pairs test (cleaned entity pairs). Each pair is scored by
  hand-tuned  the frozen engine's rule applied to two entities: the blended name score, raised by the
              consonant skeleton (in full if either entity has an Arabic-script name, otherwise by at most 0.2
              once the name score is 0.6), plus birth date (+0.05 agree, -0.30 conflict) and country (+0.03
              agree, -0.15 conflict), against the frozen threshold 0.86. Identifiers are not used by that engine.
  trained     the logistic regression over all eight features, served from the coefficients file with the
              standard library only, against the threshold that maximised F1 on Pairs dev.
Reported per script slice and separately for pairs that share an entity with train and pairs that do not.

Part B, screening test (the dev and test queries of steps 3c and 4). The same candidates are retrieved for both
systems. The hand-tuned engine is the frozen configuration. The trained model scores the query against every
candidate; its threshold is chosen on the screening dev queries (best F1), and also set to match the
hand-tuned engine's dev false positive rate. Reported per script of the query: recall, precision, false
positive rate.

Run:
    python src/compare_matchers.py
Writes reports/pairs_comparison.json.
"""

import argparse
import json
import multiprocessing
import multiprocessing.pool
import platform
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import evaluate_scoring as ev
from build_pair_features import OUT_DIR, read_table
from entities import load_named_entities
from evaluate_screening import build_query_sets
from idf_table import load_table
from matcher_model import MODEL_PATH, LogisticMatcher
from pair_features import SLICES, Forms, make_idf, pair_features
from profile_data import DEFAULT_INPUT, PROPS_PATH
from scoring import Screener

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "pairs_comparison.json"
HAND_THRESHOLD = 0.86  # the frozen configuration
FLOOR, LIFT, BLEND = 0.6, 0.2, 0.5
THRESHOLDS = [round(0.02 + 0.01 * i, 2) for i in range(98)]
SCREEN_PARAMS = ev.Params(FLOOR, LIFT, 0)


# ---------- Part A: pairs ----------


def hand_tuned_pair_score(row: dict) -> float:
    """The frozen engine's score for two entities, from the pair's features."""
    name = float(row["name_score"])
    skeleton = float(row["skeleton_score"])
    if int(row["arabic_involved"]):
        name = max(name, skeleton)
    elif name >= FLOOR:
        name = max(name, min(skeleton, name + LIFT))
    name += {"agree": 0.05, "conflict": -0.30}.get(row["birth_state"], 0.0)
    name += {"agree": 0.03, "conflict": -0.15}.get(row["country_state"], 0.0)
    return min(1.0, max(0.0, name))


def counts_at(rows: list[dict], scores: list[float], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for row, score in zip(rows, scores, strict=True):
        alert, label = score >= threshold, row["label"] == "1"
        tp += alert and label
        fp += alert and not label
        fn += (not alert) and label
        tn += (not alert) and not label
    return {
        "pairs": len(rows),
        "matches": tp + fn,
        "non_matches": fp + tn,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
    }


def best_f1(rows: list[dict], scores: list[float], grid: list[float]) -> float:
    best, best_f1_value = grid[0], -1.0
    for t in grid:
        c = counts_at(rows, scores, t)
        recall, precision = c["recall"] or 0.0, c["precision"] or 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        if f1 >= best_f1_value:
            best, best_f1_value = t, f1
    return best


def matched_recall_threshold(rows: list[dict], scores: list[float], target_recall: float) -> float:
    """The highest threshold whose recall is still at least `target_recall`."""
    chosen = THRESHOLDS[0]
    for t in THRESHOLDS:
        if (counts_at(rows, scores, t)["recall"] or 0.0) >= target_recall:
            chosen = t
    return chosen


def sliced(rows: list[dict], scores: list[float], threshold: float) -> dict:
    out = {"all": counts_at(rows, scores, threshold)}
    for name in SLICES:
        picked = [(r, s) for r, s in zip(rows, scores, strict=True) if r["script_slice"] == name]
        if picked:
            out[name] = counts_at([r for r, _ in picked], [s for _, s in picked], threshold)
    return out


def part_a(tables: Path, model: LogisticMatcher, model_threshold: float) -> dict:
    from pair_features import auc

    dev, test = (read_table(tables / f"{s}.csv.gz") for s in ("dev", "test"))
    score = {
        "hand_tuned": {name: [hand_tuned_pair_score(r) for r in rows] for name, rows in (("dev", dev), ("test", test))},
        "trained": {name: [model.probability(r) for r in rows] for name, rows in (("dev", dev), ("test", test))},
    }
    hand_dev_threshold = best_f1(dev, score["hand_tuned"]["dev"], [t for t in THRESHOLDS if t >= 0.5])
    hand_dev_recall = counts_at(dev, score["hand_tuned"]["dev"], HAND_THRESHOLD)["recall"]
    matched = matched_recall_threshold(dev, score["trained"]["dev"], hand_dev_recall)
    systems = {
        "hand_tuned_frozen_threshold": ("hand_tuned", HAND_THRESHOLD),
        "hand_tuned_threshold_retuned_on_pairs_dev": ("hand_tuned", hand_dev_threshold),
        "trained_threshold_best_f1_on_pairs_dev": ("trained", model_threshold),
        "trained_threshold_matching_hand_tuned_dev_recall": ("trained", matched),
    }
    subsets = {
        "all_pairs": lambda r: True,
        "share_an_entity_with_train": lambda r: r["shares_entity_with_train"] == "1",
        "no_entity_in_train": lambda r: r["shares_entity_with_train"] == "0",
    }
    results: dict = {}
    for label, (kind, threshold) in systems.items():
        results[label] = {"threshold": threshold}
        for subset, keep in subsets.items():
            idx = [i for i, r in enumerate(test) if keep(r)]
            results[label][subset] = sliced([test[i] for i in idx], [score[kind]["test"][i] for i in idx], threshold)
    labels_dev = [int(r["label"]) for r in dev]
    labels_test = [int(r["label"]) for r in test]
    return {
        "test_pairs": len(test),
        "auc": {
            kind: {
                "dev": round(auc(score[kind]["dev"], labels_dev), 4),
                "test": round(auc(score[kind]["test"], labels_test), 4),
            }
            for kind in score
        },
        "thresholds": {
            "hand_tuned_frozen": HAND_THRESHOLD,
            "hand_tuned_best_f1_on_pairs_dev": hand_dev_threshold,
            "trained_best_f1_on_pairs_dev": model_threshold,
            "trained_matching_hand_tuned_dev_recall": matched,
        },
        "systems": results,
    }


# ---------- Part B: screening ----------

_w: dict = {}


def _init(path: str, overrides: dict[str, list[str]], df: dict[str, int], documents: int, model: dict) -> None:
    props = {s: list(v["identifier"]) for s, v in json.loads(PROPS_PATH.read_text())["schemas"].items()}
    _w["entities"] = {e["id"]: e for e in load_named_entities(Path(path)) if e["target"]}
    _w["overrides"] = overrides
    _w["idf"] = make_idf(df, documents)
    _w["model"] = LogisticMatcher(model["coefficients"], model["intercept"])
    _w["props"] = props
    _w["cache"] = {}


def _listed(entity_id: str, held_out: bool) -> Forms:
    key = (entity_id, held_out)
    cache = _w["cache"]
    if key not in cache:
        if len(cache) > 15_000:
            cache.clear()
        entity = _w["entities"][entity_id]
        names = _w["overrides"].get(entity_id, entity["names"]) if held_out else entity["names"]
        cache[key] = Forms(
            {"id": entity_id, "caption": names[0], "schema": entity["schema"],
             "properties": {"name": names[1:], "birthDate": entity["birth_dates"], "country": entity["countries"]}},
            _w["props"],
        )  # fmt: skip
    return cache[key]


def _score(task: tuple) -> tuple:
    """(task id, probability of the true entity or None, highest probability of any other candidate)."""
    task_id, name, dob, country, candidates, true_id, held_out = task
    query = Forms(
        {"id": "query", "caption": name, "schema": "Person",
         "properties": {"birthDate": [dob] if dob else [], "country": [country] if country else []}},
        _w["props"],
    )  # fmt: skip
    best_other, true_p = 0.0, None
    for candidate in candidates:
        p = _w["model"].probability(pair_features(query, _listed(candidate, held_out), _w["idf"]))
        if candidate == true_id:
            true_p = p
        else:
            best_other = max(best_other, p)
    return task_id, true_p, best_other


def trained_rows(
    pool: "multiprocessing.pool.Pool", screener: Screener, queries: list[dict], held_out: bool
) -> list[tuple]:
    tasks = []
    for i, q in enumerate(queries):
        candidates = [entity_id for entity_id, _ in screener.index.query(q["name"])]
        tasks.append((i, q["name"], q["dob"], q["country"], candidates, q["true_id"], held_out))
    results = {}
    for task_id, true_p, other in pool.imap_unordered(_score, tasks, chunksize=20):
        results[task_id] = (true_p, other)
    rows = []
    for i, q in enumerate(queries):
        true_p, other = results[i]
        group_or_kind = q["group"] if q["kind"] == "positive" else q["kind"]
        top1 = true_p is not None and true_p > 0 and true_p >= other
        rows.append((group_or_kind, q["group"], true_p, other, top1, q["script"]))
    return rows


def hand_rows(screener: Screener, queries: list[dict]) -> list[tuple]:
    for q in queries:
        q["comps"] = screener.components(q["name"], q["dob"], q["country"], always=q["true_id"], blends=(BLEND,))
    rows = ev.final_scores(queries, SCREEN_PARAMS, True, True)
    for q in queries:
        del q["comps"]
    return rows


def choose_threshold_for_f1(pos: list[tuple], neg: list[tuple]) -> float:
    best, best_f1_value = THRESHOLDS[0], -1.0
    for t in THRESHOLDS:
        f1 = ev.metrics_at(pos, neg, t)["f1"]
        if f1 >= best_f1_value:
            best, best_f1_value = t, f1
    return best


def choose_threshold_for_fpr(pos: list[tuple], neg: list[tuple], target: float) -> float:
    """The lowest threshold (most recall) whose dev false positive rate is at most `target`."""
    for t in THRESHOLDS:
        if ev.metrics_at(pos, neg, t)["false_positive_rate"] <= target:
            return t
    return THRESHOLDS[-1]


SUMMARY_KEYS = (
    "recall", "precision", "f1", "top1_accuracy", "false_positive_rate", "fpr_synthetic", "fpr_near_miss",
    "recall_by_script", "precision_by_script", "fpr_by_script", "top1_by_script", "recall_by_group",
)  # fmt: skip


def summary(metrics: dict) -> dict:
    return {k: metrics[k] for k in SUMMARY_KEYS}


def part_b(args: argparse.Namespace, model_json: dict) -> dict:
    entities = load_named_entities(args.input)
    targets = [e for e in entities if e["target"]]
    positives, negatives, held = build_query_sets(entities, targets)
    df, documents = load_table()
    held_screener = Screener.build(
        [{**e, "names": held.get(e["id"], e["names"])} for e in targets],
        blend=BLEND,
        idf_table=(df, documents),
        skeleton_idf=False,
    )
    full_screener = Screener.build(targets, blend=BLEND, idf_table=(df, documents), skeleton_idf=False)

    dev_pos = [q for q in positives if q["split"] == "dev" and q["group"] in ev.HEADLINE_GROUPS]
    dev_neg = [q for q in negatives if q["split"] == "dev"]
    test_pos = [q for q in positives if q["split"] == "test"]
    test_neg = [q for q in negatives if q["split"] == "test"]
    if args.limit:
        dev_pos, dev_neg, test_pos, test_neg = (
            list(islice(x, args.limit)) for x in (dev_pos, dev_neg, test_pos, test_neg)
        )
    print(
        f"screening queries: dev {len(dev_pos)}+{len(dev_neg)}, test {len(test_pos)}+{len(test_neg)}", file=sys.stderr
    )

    start = time.perf_counter()
    hand = {
        "dev_pos": hand_rows(held_screener, dev_pos), "dev_neg": hand_rows(full_screener, dev_neg),
        "test_pos": hand_rows(held_screener, test_pos), "test_neg": hand_rows(full_screener, test_neg),
    }  # fmt: skip
    print(f"hand-tuned scored in {time.perf_counter() - start:.0f}s", file=sys.stderr)

    start = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    initargs = (str(args.input), held, df, documents, model_json)
    with context.Pool(args.workers, _init, initargs) as pool:
        trained = {
            "dev_pos": trained_rows(pool, held_screener, dev_pos, True),
            "dev_neg": trained_rows(pool, full_screener, dev_neg, False),
            "test_pos": trained_rows(pool, held_screener, test_pos, True),
            "test_neg": trained_rows(pool, full_screener, test_neg, False),
        }
    print(f"trained model scored in {time.perf_counter() - start:.0f}s", file=sys.stderr)

    hand_dev = ev.metrics_at(hand["dev_pos"], hand["dev_neg"], HAND_THRESHOLD)
    f1_threshold = choose_threshold_for_f1(trained["dev_pos"], trained["dev_neg"])
    fpr_threshold = choose_threshold_for_fpr(trained["dev_pos"], trained["dev_neg"], hand_dev["false_positive_rate"])
    systems = {
        "hand_tuned_frozen": (hand, HAND_THRESHOLD),
        "trained_threshold_best_f1_on_screening_dev": (trained, f1_threshold),
        "trained_threshold_matching_hand_tuned_dev_fpr": (trained, fpr_threshold),
    }
    results = {}
    for label, (rows, threshold) in systems.items():
        results[label] = {
            "threshold": threshold,
            "dev": summary(ev.metrics_at(rows["dev_pos"], rows["dev_neg"], threshold)),
            "test": summary(ev.metrics_at(rows["test_pos"], rows["test_neg"], threshold)),
        }
    return {
        "queries": {
            "dev_positive": len(dev_pos),
            "dev_negative": len(dev_neg),
            "test_positive": len(test_pos),
            "test_negative": len(test_neg),
            "test_negative_by_script": dict(Counter(q["script"] for q in test_neg)),
        },
        "systems": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tables", type=Path, default=OUT_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--limit", type=int, help="a trial: only the first N screening queries of each kind")
    parser.add_argument("--skip-screening", action="store_true")
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    model_json = json.loads(args.model.read_text())
    model = LogisticMatcher(model_json["coefficients"], model_json["intercept"])
    report = {"python_version": platform.python_version(), "model": str(args.model.name), "C": model_json["C"]}
    report["pairs_test"] = part_a(args.tables, model, model_json["dev_f1_threshold"])
    if not args.skip_screening:
        report["screening_test"] = part_b(args, model_json)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
