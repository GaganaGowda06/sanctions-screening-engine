"""Final evaluation on the Pairs final hold-out. Run once.

Scores the cleaned pairs of the final hold-out with the constrained matcher (models/pairs_lr_v2.json, at the
threshold chosen on Pairs dev) and with the hand-tuned engine's rule applied to two entities (the frozen
threshold, and the threshold that was best on Pairs dev). Reported overall, by script slice, and separately for
pairs that share an entity with a train pair and pairs that do not, with Wilson intervals for the rates and a
bootstrap interval for the AUC.

This is the only code that may ask for the hold-out split (pairs.iter_split allow_final). Before scoring it
recomputes the two membership digests of the hold-out and checks them against the frozen manifest and cleaning
record, and checks the dataset checksum, so only the frozen pairs are scored. Nothing is tuned: the model, its
threshold and the hand-tuned thresholds all come from earlier steps. It refuses to run a second time.

Run:
    python src/evaluate_pairs_final.py
Writes reports/pairs_final_evaluation.json.
"""

import argparse
import hashlib
import json
import multiprocessing
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import build_pair_features as bpf
from compare_matchers import counts_at, hand_tuned_pair_score, sliced
from evaluate_final import code_version, wilson
from idf_table import load_table
from matcher_model import LogisticMatcher
from pair_features import SLICES
from pairs import PAIRS_PATH, ROOT, SPLIT_DIR, iter_split, load_clean_mask
from profile_data import PROPS_PATH
from train_matcher import auc as auc_fast
from train_matcher import bootstrap_auc

MODEL = ROOT / "models" / "pairs_lr_v2.json"
COMPARISON = ROOT / "reports" / "pairs_comparison_v2.json"
OUTPUT = ROOT / "reports" / "pairs_final_evaluation.json"
HAND_FROZEN = 0.86
SUBSETS = {
    "all_pairs": lambda r: True,
    "share_an_entity_with_train": lambda r: r["shares_entity_with_train"] == "1",
    "no_entity_in_train": lambda r: r["shares_entity_with_train"] == "0",
}


def with_intervals(counts: dict, rows: list[dict], scores: list[float], threshold: float) -> dict:
    """counts_at plus Wilson intervals for recall, precision and the false positive rate."""
    tp = sum(s >= threshold and r["label"] == "1" for r, s in zip(rows, scores, strict=True))
    alerts = sum(s >= threshold for s in scores)
    out = dict(counts)
    for name, k, n in (
        ("recall", tp, counts["matches"]),
        ("precision", tp, alerts),
        ("false_positive_rate", alerts - tp, counts["non_matches"]),
    ):
        low, high = wilson(k, n)
        out[f"{name}_ci95"] = [round(low, 4), round(high, 4)] if n else None
    return out


def digests(split_dir: Path, path: Path) -> tuple[list[tuple[int, dict]], str, str]:
    """The kept hold-out pairs, plus the membership digest of every hold-out pair and of the kept line numbers."""
    mask = load_clean_mask(split_dir)
    membership, kept_lines = hashlib.sha256(), hashlib.sha256()
    kept: list[tuple[int, dict]] = []
    for index, record in iter_split("final", allow_final=True, path=path, split_dir=split_dir):
        membership.update(f"{index}\t{record['left']['id']}\t{record['right']['id']}\n".encode())
        if mask[index] == "1":
            kept_lines.update(f"{index}\n".encode())
            kept.append((index, record))
    return kept, membership.hexdigest(), kept_lines.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PAIRS_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--comparison", type=Path, default=COMPARISON, help="report holding the hand-tuned thresholds")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)
    if args.output.exists():
        sys.exit(f"{args.output} exists: the Pairs final evaluation has been run. It is not run twice.")

    # 1. The frozen files must be the ones that were frozen.
    manifest = json.loads((args.split_dir / "manifest.json").read_text())
    cleaning = json.loads((args.split_dir / "cleaning.json").read_text())
    source = json.loads((args.input.parent / "SOURCE.json").read_text())
    dataset_ok = source["sha256"] == manifest["dataset_sha256"]
    if not dataset_ok or cleaning["assignment_sha256"] != manifest["assignment_sha256"]:
        sys.exit("The dataset or the cleaning record does not match the frozen split manifest. Not scoring.")
    kept, membership, kept_lines = digests(args.split_dir, args.input)
    if membership != manifest["final_holdout_membership_sha256"]:
        sys.exit("The hold-out pairs are not the frozen membership. Not scoring.")
    if kept_lines != cleaning["final_holdout_kept_line_numbers_sha256"]:
        sys.exit("The cleaned hold-out pairs are not the frozen ones. Not scoring.")
    print(f"hold-out membership ok ({membership[:12]}), {len(kept):,} cleaned pairs", file=sys.stderr)

    saved = json.loads(args.model.read_text())
    model = LogisticMatcher(saved["coefficients"], saved["intercept"])
    model_threshold = saved["dev_f1_threshold"]
    thresholds = json.loads(args.comparison.read_text())["pairs_test"]["thresholds"]
    hand_retuned = thresholds["hand_tuned_best_f1_on_pairs_dev"]

    # 2. Features, exactly as for train, dev and test. Train entities only flag which pairs share an entity.
    train_entities: set[str] = set()
    for _, record in iter_split("train", clean=True, path=args.input, split_dir=args.split_dir):
        train_entities.update((record["left"]["id"], record["right"]["id"]))
    document_frequency, documents = load_table()
    identifier_props = {s: list(v["identifier"]) for s, v in json.loads(PROPS_PATH.read_text())["schemas"].items()}
    start = time.perf_counter()
    rows: list[dict] = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, bpf._init, (document_frequency, documents, identifier_props)) as pool:
        for batch in pool.imap(bpf.featurise, bpf.chunks(iter(kept), bpf.CHUNK)):
            for row in batch:
                seen = row["left_id"] in train_entities or row["right_id"] in train_entities
                rows.append({**row, "label": str(row["label"]), "shares_entity_with_train": "1" if seen else "0"})
    print(f"featurised in {time.perf_counter() - start:.0f}s", file=sys.stderr)

    # 3. Score and report.
    labels = np.array([int(r["label"]) for r in rows])
    systems = {
        "constrained_model": ([model.probability(r) for r in rows], model_threshold),
        "hand_tuned_frozen_threshold": ([hand_tuned_pair_score(r) for r in rows], HAND_FROZEN),
        "hand_tuned_threshold_best_on_pairs_dev": ([hand_tuned_pair_score(r) for r in rows], hand_retuned),
    }
    rng = np.random.default_rng(23)
    results: dict = {}
    for label, (scores, threshold) in systems.items():
        results[label] = {"threshold": threshold}
        score_array = np.array(scores)
        for subset, keep in SUBSETS.items():
            idx = [i for i, r in enumerate(rows) if keep(r)]
            subset_rows, subset_scores = [rows[i] for i in idx], [scores[i] for i in idx]
            counts = counts_at(subset_rows, subset_scores, threshold)
            y, s = labels[idx], score_array[idx]
            area = auc_fast(y, s)
            results[label][subset] = {
                **with_intervals(counts, subset_rows, subset_scores, threshold),
                "auc": None if area is None else round(area, 4),
                "auc_ci95": None if area is None else bootstrap_auc(y, s, rng, reps=300),
            }
        for name, keep in (
            ("by_script_slice_all_pairs", SUBSETS["all_pairs"]),
            ("by_script_slice_share_an_entity", SUBSETS["share_an_entity_with_train"]),
            ("by_script_slice_no_entity", SUBSETS["no_entity_in_train"]),
        ):
            picked = [(r, s) for r, s in zip(rows, scores, strict=True) if keep(r)]
            results[label][name] = sliced([r for r, _ in picked], [s for _, s in picked], threshold)

    test = json.loads(args.comparison.read_text())["pairs_test"]
    report = {
        "python_version": platform.python_version(),
        "code_commit": code_version(),
        "frozen": {
            "dataset_sha256": manifest["dataset_sha256"],
            "assignment_sha256": manifest["assignment_sha256"],
            "holdout_membership_sha256": membership,
            "cleaned_holdout_kept_lines_sha256": kept_lines,
            "model": args.model.name,
            "model_threshold_chosen_on_pairs_dev": model_threshold,
        },
        "pairs": len(rows),
        "matches": int(labels.sum()),
        "share_an_entity_with_train": sum(r["shares_entity_with_train"] == "1" for r in rows),
        "script_slices": dict(Counter(r["script_slice"] for r in rows)),
        "systems": results,
        "pairs_test_for_comparison": {"auc": test["auc"], "thresholds": test["thresholds"]},
    }
    assert set(SLICES) >= set(report["script_slices"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
