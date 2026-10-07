"""Train the logistic regression matcher on the Pairs feature tables and examine it.

1. L2-regularised logistic regression fitted on train, the regularisation strength (C) chosen on dev
   by log loss.
2. Ablation: each of the eight features dropped in turn, the model refitted with the same C, dev AUC change
   overall and by script slice.
3. Coefficient stability: bootstrap resamples of the train pairs, the spread of every coefficient.
4. Leakage check: the model's AUC on dev and test pairs that share an entity with train, against pairs
   that do not, with the same gap for the raw features.
Dev and test are the cleaned Pairs splits. The final hold-out is not read.

Writes models/pairs_lr_v1.json (the coefficients) and reports/pairs_training.json.

Run:
    python src/train_matcher.py
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from build_pair_features import OUT_DIR, read_table
from idf_table import MANIFEST_PATH
from matcher_model import FEATURE_INPUTS, INPUTS, NAME_SCORES, ROOT, STATE_LEVELS, STATE_PREFIX
from pair_features import SLICES

MODEL_OUT = ROOT / "models" / "pairs_lr_v1.json"
REPORT = ROOT / "reports" / "pairs_training.json"
FEATURES_REPORT = ROOT / "reports" / "pairs_features.json"
LOG_LOSS_TOLERANCE = 0.0005
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
BOOTSTRAPS = 100
SEED = 17


class Table:
    """One split as arrays: the model inputs, the label, and the columns to slice by."""

    def __init__(self, rows: list[dict]) -> None:
        self.n = len(rows)
        self.y = np.array([int(r["label"]) for r in rows])
        self.slice = np.array([r["script_slice"] for r in rows])
        self.shares = np.array([r["shares_entity_with_train"] for r in rows])
        self.raw = {name: np.array([float(r[name]) for r in rows]) for name in NAME_SCORES}
        columns = dict(self.raw)
        for state, levels in STATE_LEVELS.items():
            values = np.array([r[state] for r in rows])
            for level in levels:
                columns[f"{STATE_PREFIX[state]}_{level}"] = (values == level).astype(float)
        self.columns = columns

    def matrix(self, inputs: tuple[str, ...]) -> np.ndarray:
        return np.column_stack([self.columns[name] for name in inputs])


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(table: Table, inputs: tuple[str, ...], c: float, rows: np.ndarray | None = None) -> LogisticRegression:
    x, y = table.matrix(inputs), table.y
    if rows is not None:
        x, y = x[rows], y[rows]
    return LogisticRegression(C=c, l1_ratio=0.0, max_iter=2000, tol=1e-8).fit(x, y)


def probabilities(model: LogisticRegression, table: Table, inputs: tuple[str, ...]) -> np.ndarray:
    return model.predict_proba(table.matrix(inputs))[:, 1]


def auc_by_slice(table: Table, score: np.ndarray) -> dict[str, float | None]:
    out = {"all": auc(table.y, score)}
    for name in SLICES:
        mask = table.slice == name
        out[name] = auc(table.y[mask], score[mask]) if mask.any() else None
    return {k: None if v is None else round(v, 4) for k, v in out.items()}


def bootstrap_auc(y: np.ndarray, score: np.ndarray, rng: np.random.Generator, reps: int = 300) -> list[float]:
    values = []
    for _ in range(reps):
        pick = rng.integers(0, len(y), len(y))
        value = auc(y[pick], score[pick])
        if value is not None:
            values.append(value)
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for t in np.arange(0.02, 0.99, 0.01):
        predicted = p >= t
        tp = int((predicted & (y == 1)).sum())
        fp = int((predicted & (y == 0)).sum())
        fn = int((~predicted & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 >= best[1]:
            best = (round(float(t), 2), f1)
    return best


def one_bootstrap(x: np.ndarray, y: np.ndarray, c: float, seed: int) -> list[float]:
    pick = np.random.default_rng(seed).integers(0, len(y), len(y))
    model = LogisticRegression(C=c, l1_ratio=0.0, max_iter=2000, tol=1e-8).fit(x[pick], y[pick])
    return [*model.coef_[0], model.intercept_[0]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=OUT_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}", file=sys.stderr)

    train, dev, test = (Table(read_table(args.tables / f"{s}.csv.gz")) for s in ("train", "dev", "test"))
    print(f"train {train.n:,}, dev {dev.n:,}, test {test.n:,}", file=sys.stderr)

    # 1. Regularisation on dev.
    tuning = []
    for c in C_GRID:
        model = fit(train, INPUTS, c)
        p_dev = probabilities(model, dev, INPUTS)
        tuning.append(
            {
                "C": c,
                "train_log_loss": round(log_loss(train.y, probabilities(model, train, INPUTS)), 5),
                "dev_log_loss": round(log_loss(dev.y, p_dev), 5),
                "dev_auc": round(auc(dev.y, p_dev), 4),
            }
        )
        print(f"C={c:g}: dev log loss {tuning[-1]['dev_log_loss']}, AUC {tuning[-1]['dev_auc']}", file=sys.stderr)
    # Dev log loss is flat once C is moderate, so the best value alone would pick an almost unregularised model.
    # Take the most regularised C within LOG_LOSS_TOLERANCE of the best dev log loss.
    lowest = min(t["dev_log_loss"] for t in tuning)
    best_c = min(t["C"] for t in tuning if t["dev_log_loss"] <= lowest + LOG_LOSS_TOLERANCE)
    model = fit(train, INPUTS, best_c)
    coefficients = dict(zip(INPUTS, (float(w) for w in model.coef_[0]), strict=True))
    intercept = float(model.intercept_[0])
    p = {name: probabilities(model, t, INPUTS) for name, t in (("train", train), ("dev", dev), ("test", test))}
    threshold, dev_f1 = best_f1_threshold(dev.y, p["dev"])

    def prf(table: Table, probability: np.ndarray, t: float) -> dict:
        predicted = probability >= t
        tp = int((predicted & (table.y == 1)).sum())
        fp = int((predicted & (table.y == 0)).sum())
        fn = int((~predicted & (table.y == 1)).sum())
        tn = int((~predicted & (table.y == 0)).sum())
        return {"recall": round(tp / (tp + fn), 4), "precision": round(tp / (tp + fp), 4),
                "false_positive_rate": round(fp / (fp + tn), 4)}  # fmt: skip

    performance = {
        name: {"auc": round(auc(t.y, p[name]), 4), "log_loss": round(log_loss(t.y, p[name]), 4),
               **prf(t, p[name], threshold)}
        for name, t in (("train", train), ("dev", dev), ("test", test))
    }  # fmt: skip

    # 2. Ablation.
    full_by_slice = auc_by_slice(dev, p["dev"])
    ablation = {}
    for feature, dropped in FEATURE_INPUTS.items():
        inputs = tuple(i for i in INPUTS if i not in dropped)
        reduced = fit(train, inputs, best_c)
        p_reduced = probabilities(reduced, dev, inputs)
        by_slice = auc_by_slice(dev, p_reduced)
        ablation[feature] = {
            "dev_auc": by_slice["all"],
            "auc_change": {k: None if v is None else round(v - full_by_slice[k], 4) for k, v in by_slice.items()},
            "dev_log_loss_change": round(log_loss(dev.y, p_reduced) - log_loss(dev.y, p["dev"]), 4),
        }
        change = ablation[feature]["auc_change"]["all"]
        print(f"dropped {feature}: dev AUC {by_slice['all']} ({change:+})", file=sys.stderr)

    # 3. Coefficient stability.
    x_train = train.matrix(INPUTS)
    runs = Parallel(n_jobs=-2)(
        delayed(one_bootstrap)(x_train, train.y, best_c, SEED + i) for i in range(args.bootstraps)
    )
    runs = np.array(runs)
    names = [*INPUTS, "intercept"]
    full = [*coefficients.values(), intercept]
    stability = {}
    for i, name in enumerate(names):
        column = runs[:, i]
        low, high = np.percentile(column, [2.5, 97.5])
        stability[name] = {
            "fitted": round(full[i], 4),
            "bootstrap_mean": round(float(column.mean()), 4),
            "bootstrap_std": round(float(column.std(ddof=1)), 4),
            "ci95": [round(float(low), 4), round(float(high), 4)],
            "same_sign_as_fitted": round(float(np.mean(np.sign(column) == np.sign(full[i]))), 3),
        }
    name_idx = [INPUTS.index(n) for n in NAME_SCORES]
    sums = runs[:, name_idx].sum(axis=1)
    stability["sum_of_the_five_name_score_coefficients"] = {
        "fitted": round(sum(coefficients[n] for n in NAME_SCORES), 4),
        "bootstrap_std": round(float(sums.std(ddof=1)), 4),
        "ci95": [round(float(x), 4) for x in np.percentile(sums, [2.5, 97.5])],
    }

    # 4. Leakage check.
    rng = np.random.default_rng(SEED)
    leakage = {}
    for name, table in (("dev", dev), ("test", test)):
        entry = {}
        for label, mask in (
            ("shares_entity_with_train", table.shares == "1"),
            ("no_entity_in_train", table.shares == "0"),
        ):
            y, score = table.y[mask], p[name][mask]
            entry[label] = {
                "pairs": int(mask.sum()),
                "non_matches": int((y == 0).sum()),
                "model_auc": round(auc(y, score), 4),
                "model_auc_ci95": bootstrap_auc(y, score, rng),
                **{f"raw_{n}_auc": round(auc(y, table.raw[n][mask]), 4) for n in ("name_score", "weighted_score")},
            }
        seen, unseen = entry["shares_entity_with_train"], entry["no_entity_in_train"]
        entry["gap_shares_minus_no_entity"] = {
            "model": round(seen["model_auc"] - unseen["model_auc"], 4),
            **{
                f"raw_{n}": round(
                    entry["shares_entity_with_train"][f"raw_{n}_auc"] - entry["no_entity_in_train"][f"raw_{n}_auc"], 4
                )
                for n in ("name_score", "weighted_score")
            },
        }
        leakage[name] = entry

    report = {
        "python_version": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "inputs": list(INPUTS),
        "regularisation_tuned_on_dev": {
            "grid": tuning,
            "chosen_C": best_c,
            "criterion": f"most regularised C within {LOG_LOSS_TOLERANCE} of the best dev log loss",
        },
        "coefficients": {k: round(v, 4) for k, v in coefficients.items()} | {"intercept": round(intercept, 4)},
        "dev_f1_threshold": {"threshold": threshold, "dev_f1": round(dev_f1, 4)},
        "performance_at_dev_f1_threshold": performance,
        "auc_by_script_slice_dev": full_by_slice,
        "auc_by_script_slice_test": auc_by_slice(test, p["test"]),
        "ablation_dev": ablation,
        "coefficient_stability": {"bootstrap_resamples": args.bootstraps, "resampling": "train pairs, with replacement",
                                  "coefficients": stability},  # fmt: skip
        "leakage_check": leakage,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(
        json.dumps(
            {
                "model": "L2 logistic regression over twelve inputs; see src/matcher_model.py",
                "C": best_c,
                "inputs": list(INPUTS),
                "coefficients": coefficients,
                "intercept": intercept,
                "dev_f1_threshold": threshold,
                "trained_on_feature_table_sha256": json.loads(FEATURES_REPORT.read_text())["tables"]["train"]["sha256"],
                "idf_table_sha256": json.loads(MANIFEST_PATH.read_text())["table_sha256"],
                "scikit_learn": sklearn.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.model} and {args.report}")


if __name__ == "__main__":
    main()
