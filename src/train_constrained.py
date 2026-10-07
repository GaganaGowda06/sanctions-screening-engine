"""Retrain the matcher with two name features and sign-constrained attribute weights.

Step 3's model learned attribute weights from the Pairs data that do not transfer to screening: a missing
attribute goes with matches there, so an agreeing country got a negative weight. Here the weights are kept
sensible by construction.

  Features   two name scores, name_score (the current engine's) and weighted_score (rarity-weighted tokens),
             and the attribute states as indicators with "missing" as the reference level, so a missing
             attribute has exactly zero effect.
  Signs      every agree weight >= 0 and every conflict weight <= 0. The one-year-apart birth state ("close")
             is neither; it is held <= 0, because in the data it goes with non-matches (6.5% match).
             The name weights are not constrained.
  Fit        L2 regularised logistic regression with the same objective as scikit-learn,
             C * sum(log loss) + 0.5 * |w|^2 with the intercept unpenalised, minimised by bounded L-BFGS
             (scipy) because scikit-learn cannot take bounds. C is chosen on Pairs dev, by the rule used before:
             the most regularised C within 0.0005 of the best dev log loss.
The alert threshold is NOT chosen here: it is chosen on the screening dev queries (compare_matchers.py).
Dev and test are the cleaned Pairs splits; the final hold-out is not read.

Writes models/pairs_lr_v2.json and reports/pairs_training_v2.json.

Run:
    python src/train_constrained.py
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from build_pair_features import OUT_DIR, read_table
from idf_table import MANIFEST_PATH
from matcher_model import ROOT
from train_matcher import (
    C_GRID,
    FEATURES_REPORT,
    LOG_LOSS_TOLERANCE,
    Table,
    auc,
    auc_by_slice,
    best_f1_threshold,
    log_loss,
)

ATTRIBUTE_INPUTS = (
    "birth_agree", "birth_close", "birth_conflict", "country_agree", "country_conflict",
    "identifier_agree", "identifier_conflict",
)  # fmt: skip
TWO_NAME_SCORES = ("name_score", "weighted_score")
FIVE_NAME_SCORES = ("name_score", "skeleton_score", "weighted_score", "phonetic_score", "arabic_score")
INPUTS = TWO_NAME_SCORES + ATTRIBUTE_INPUTS
MODEL_OUT = ROOT / "models" / "pairs_lr_v2.json"
REPORT = ROOT / "reports" / "pairs_training_v2.json"


def bounds_for(name: str) -> tuple[float | None, float | None]:
    if name.endswith("_agree"):
        return (0.0, None)
    if name.endswith(("_conflict", "_close")):
        return (None, 0.0)
    return (None, None)


def fit_constrained(
    x: np.ndarray, y: np.ndarray, c: float, bounds: list[tuple[float | None, float | None]]
) -> tuple[np.ndarray, float]:
    """Minimise C * sum(log loss) + 0.5 * |w|^2 over weights within `bounds`, intercept free and unpenalised."""
    sign = 2.0 * y - 1.0
    n_features = x.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = theta[:n_features], theta[n_features]
        margin = sign * (x @ w + b)
        loss = c * np.logaddexp(0.0, -margin).sum() + 0.5 * w @ w
        slope = -c * sign * expit(-margin)  # derivative of the loss term with respect to z
        return float(loss), np.append(x.T @ slope + w, slope.sum())

    start = np.zeros(n_features + 1)
    result = minimize(
        objective, start, jac=True, method="L-BFGS-B", bounds=[*bounds, (None, None)],
        options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-9},
    )  # fmt: skip
    return result.x[:n_features], float(result.x[n_features])


def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return expit(x @ w + b)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=OUT_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument(
        "--five-name-scores", action="store_true",
        help="diagnostic: keep skeleton, phonetic and Arabic-folded scores too (the delivered model drops them)",
    )  # fmt: skip
    args = parser.parse_args(argv)
    inputs = (FIVE_NAME_SCORES if args.five_name_scores else TWO_NAME_SCORES) + ATTRIBUTE_INPUTS
    print(f"Python {platform.python_version()}, scipy {scipy.__version__}", file=sys.stderr)

    train, dev, test = (Table(read_table(args.tables / f"{s}.csv.gz")) for s in ("train", "dev", "test"))
    x_train, x_dev, x_test = (t.matrix(inputs) for t in (train, dev, test))
    bounds = [bounds_for(name) for name in inputs]

    tuning = []
    for c in C_GRID:
        w, b = fit_constrained(x_train, train.y, c, bounds)
        p_dev = predict(x_dev, w, b)
        tuning.append(
            {"C": c, "dev_log_loss": round(log_loss(dev.y, p_dev), 5), "dev_auc": round(auc(dev.y, p_dev), 4)}
        )
        print(f"C={c:g}: dev log loss {tuning[-1]['dev_log_loss']}, AUC {tuning[-1]['dev_auc']}", file=sys.stderr)
    lowest = min(t["dev_log_loss"] for t in tuning)
    best_c = min(t["C"] for t in tuning if t["dev_log_loss"] <= lowest + LOG_LOSS_TOLERANCE)
    w, b = fit_constrained(x_train, train.y, best_c, bounds)
    free = LogisticRegression(C=best_c, l1_ratio=0.0, max_iter=2000, tol=1e-8).fit(x_train, train.y)

    def scores(table: Table, matrix: np.ndarray) -> dict:
        p_constrained = predict(matrix, w, b)
        p_free = free.predict_proba(matrix)[:, 1]

        def both(p: np.ndarray) -> dict:
            return {"auc": round(auc(table.y, p), 4), "log_loss": round(log_loss(table.y, p), 4)}

        return {"constrained": both(p_constrained), "unconstrained_same_features": both(p_free)}

    p_dev, p_test = predict(x_dev, w, b), predict(x_test, w, b)
    threshold, dev_f1 = best_f1_threshold(dev.y, p_dev)
    coefficients = dict(zip(inputs, (float(v) for v in w), strict=True))
    free_coefficients = dict(zip(inputs, (float(v) for v in free.coef_[0]), strict=True))
    report = {
        "python_version": platform.python_version(),
        "scipy": scipy.__version__,
        "inputs": list(inputs),
        "constraints": {name: {"lower": lo, "upper": hi} for name, (lo, hi) in zip(inputs, bounds, strict=True)},
        "regularisation_tuned_on_pairs_dev": {
            "grid": tuning,
            "chosen_C": best_c,
            "criterion": f"most regularised C within {LOG_LOSS_TOLERANCE} of the best dev log loss",
        },
        "coefficients": {k: round(v, 4) for k, v in coefficients.items()} | {"intercept": round(b, 4)},
        "unconstrained_coefficients_same_features": {k: round(v, 4) for k, v in free_coefficients.items()}
        | {"intercept": round(float(free.intercept_[0]), 4)},
        "binding_constraints": [n for n, v in coefficients.items() if abs(v) < 1e-6 and bounds_for(n) != (None, None)],
        "pairs_auc_and_log_loss": {"dev": scores(dev, x_dev), "test": scores(test, x_test)},
        "pairs_test_auc": round(auc(test.y, p_test), 4),
        "pairs_test_auc_by_script_slice": auc_by_slice(test, p_test),
        "pairs_dev_f1_threshold_for_reference_only": {"threshold": threshold, "dev_f1": round(dev_f1, 4)},
        "alert_threshold": "chosen on the screening dev queries, see reports/pairs_comparison_v2.json",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(
        json.dumps(
            {
                "model": "sign-constrained L2 logistic regression, nine inputs; see src/train_constrained.py",
                "C": best_c,
                "inputs": list(inputs),
                "coefficients": coefficients,
                "intercept": b,
                "constraints": report["constraints"],
                "dev_f1_threshold": threshold,
                "trained_on_feature_table_sha256": json.loads(FEATURES_REPORT.read_text())["tables"]["train"]["sha256"],
                "idf_table_sha256": json.loads(MANIFEST_PATH.read_text())["table_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.model} and {args.report}")


if __name__ == "__main__":
    main()
