"""Threshold sweep over the saved per-pair table (analysis only; the engine and its 0.86 threshold are not changed).

Reads data/external/pairs_scores_table.npz (made by pairs_external_table.py), checks that the table reproduces the
stored 0.86 figures in reports/pairs_external_scores.json exactly, then sweeps the decision threshold 0.50 to 0.99 in
0.01 steps (pair is a match when its forward score >= the threshold) for three slices: all pairs, pairs with no
shared name, and person_person pairs. Writes reports/pairs_external_threshold_sweep.json (aggregates only).

The best threshold is chosen on the same pairs it is reported on, so it is an in-sample optimum; on the first
maximum when several thresholds tie. Run:
    python analysis/pairs_external_sweep.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TABLE = common.ROOT / "data" / "external" / "pairs_scores_table.npz"
REPORT = common.OUT / "pairs_external_scores.json"
OUT = common.OUT / "pairs_external_threshold_sweep.json"
THRESHOLDS = [round(0.50 + i / 100, 2) for i in range(50)]
SLICES = {
    "all_pairs": ("overall", lambda t: np.ones(len(t["label"]), dtype=bool)),
    "no_shared_name": ("shares_no_identical_name", lambda t: ~t["shared"]),
    "person_person": ("by_entity_type/person_person", lambda t: t["kind"] == "person_person"),
}


def counts(label: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(((label == 1) & pred).sum())
    fp = int(((label == 0) & pred).sum())
    fn = int(((label == 1) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def stored(report: dict, path: str) -> dict:
    node = report["left_names_as_queries_against_right"]
    for part in path.split("/"):
        node = node[part]
    return node


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT.name} already exists")
    t = np.load(TABLE)
    report = json.loads(REPORT.read_text())
    result = {"table_pairs": len(t["label"]), "direction": "forward (left names as queries against right)"}
    for name, (path, mask_of) in SLICES.items():
        mask = mask_of(t)
        label, score = t["label"][mask], t["forward"][mask]
        at_086 = counts(label, score, 0.86)
        ref = stored(report, path)
        for key in ("tp", "fp", "fn"):
            if at_086[key] != ref[key]:
                sys.exit(f"MISMATCH {name} {key}: table {at_086[key]} stored {ref[key]}")
        if at_086["f1"] != ref["f1"]:
            sys.exit(f"MISMATCH {name} f1: table {at_086['f1']!r} stored {ref['f1']!r}")
        sweep = [counts(label, score, th) for th in THRESHOLDS]
        best = max(sweep, key=lambda r: r["f1"])  # first maximum
        result[name] = {
            "pairs": int(mask.sum()), "positives": int((label == 1).sum()), "reproduces_stored_086": True,
            "at_086": at_086, "best": best, "f1_gain": best["f1"] - at_086["f1"], "sweep": sweep,
        }  # fmt: skip
        print(
            f"{name}: pairs {mask.sum()}, 0.86 reproduces stored (tp {at_086['tp']}, fp {at_086['fp']}, "
            f"fn {at_086['fn']}, f1 {at_086['f1']:.4f}); best {best['threshold']} f1 {best['f1']:.4f} "
            f"P {best['precision']:.4f} R {best['recall']:.4f}"
        )
    common.write_json(OUT, result)


if __name__ == "__main__":
    main()
