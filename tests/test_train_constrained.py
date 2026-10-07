"""Tests for src/train_constrained.py: the sign-constrained logistic regression.

Run:
    pytest tests/test_train_constrained.py
"""

import json
import random

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import build_pair_features as bpf
import matcher_model as mm
import pair_features as pf
import train_constrained as tc

FREE = (None, None)


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    z = 1.5 * x[:, 0] - 1.0 * x[:, 1] + 0.5 * x[:, 2] + 0.3
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return x, y


def test_with_slack_bounds_it_is_ordinary_l2_logistic_regression():
    x, y = synthetic()
    reference = LogisticRegression(C=2.0, l1_ratio=0.0, max_iter=5000, tol=1e-10).fit(x, y)
    for bounds in ([FREE] * 4, [(-10.0, 10.0)] * 4):
        w, b = tc.fit_constrained(x, y, 2.0, bounds)
        assert w == pytest.approx(reference.coef_[0], abs=2e-3)
        assert b == pytest.approx(reference.intercept_[0], abs=2e-3)


def test_a_binding_bound_pins_the_weight_and_costs_some_fit():
    x, y = synthetic()
    free_w, free_b = tc.fit_constrained(x, y, 2.0, [FREE] * 4)
    assert free_w[1] < -0.5  # the data want a negative weight on the second feature
    w, b = tc.fit_constrained(x, y, 2.0, [FREE, (0.0, None), FREE, FREE])
    assert w[1] == pytest.approx(0.0, abs=1e-9)  # held at the bound
    upper_w, _ = tc.fit_constrained(x, y, 2.0, [FREE, FREE, (None, 0.0), FREE])
    assert upper_w[2] <= 1e-9  # an upper bound of zero on a positive weight
    assert tc.log_loss(y, tc.predict(x, w, b)) > tc.log_loss(y, tc.predict(x, free_w, free_b))


def test_the_intercept_is_not_penalised():
    x, y = synthetic()
    w, b = tc.fit_constrained(x, y, 1e-6, [FREE] * 4)  # a huge penalty flattens the weights
    assert np.abs(w).max() < 1e-3
    base_rate = y.mean()
    assert b == pytest.approx(np.log(base_rate / (1 - base_rate)), abs=0.05)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("birth_agree", (0.0, None)), ("country_agree", (0.0, None)), ("identifier_agree", (0.0, None)),
        ("birth_conflict", (None, 0.0)), ("country_conflict", (None, 0.0)), ("identifier_conflict", (None, 0.0)),
        ("birth_close", (None, 0.0)),
        ("name_score", (None, None)), ("weighted_score", (None, None)),
    ],
)  # fmt: skip
def test_bounds_for(name, expected):
    assert tc.bounds_for(name) == expected


def test_the_two_name_feature_set_and_the_attribute_levels():
    assert tc.TWO_NAME_SCORES == ("name_score", "weighted_score")
    assert set(tc.ATTRIBUTE_INPUTS) == {i for i in mm.INPUTS if i not in mm.NAME_SCORES}
    assert len(tc.INPUTS) == 9 and len(tc.FIVE_NAME_SCORES + tc.ATTRIBUTE_INPUTS) == 12


def fake_rows(split, n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = int(rng.random() < 0.6)
        name = min(1.0, max(0.0, rng.gauss(0.9 if label else 0.45, 0.12)))
        # Country "agree" goes with non-matches here, the way a missing-attribute artefact would push it.
        country = rng.choice(["agree", "missing", "missing"] if label else ["agree", "agree", "missing"])
        scores = [round(min(1.0, max(0.0, name + rng.gauss(0, 0.05))), 4) for _ in range(5)]
        birth = rng.choice(["agree", "missing"]) if label else rng.choice(["conflict", "missing", "missing"])
        rows.append([i, split, label, "" if split == "train" else rng.choice([0, 1]), "Person", "Person",
                     "Latin on both sides only", rng.choice(pf.SLICES), 0, *scores, birth, country,
                     "missing"])  # fmt: skip
    return rows


def test_training_end_to_end_signs_and_serving(tmp_path):
    tables = tmp_path / "tables"
    for split, n, seed in (("train", 1500, 1), ("dev", 500, 2), ("test", 500, 3)):
        bpf.write_table(tables / f"{split}.csv.gz", iter(fake_rows(split, n, seed)))
    model_path, report_path = tmp_path / "m.json", tmp_path / "r.json"
    tc.main(["--tables", str(tables), "--model", str(model_path), "--report", str(report_path)])

    saved = json.loads(model_path.read_text())
    coefficients = saved["coefficients"]
    assert list(coefficients) == list(tc.INPUTS)
    for name, weight in coefficients.items():
        low, high = tc.bounds_for(name)
        assert low is None or weight >= low - 1e-9
        assert high is None or weight <= high + 1e-9
    report = json.loads(report_path.read_text())
    assert report["unconstrained_coefficients_same_features"]["country_agree"] < 0  # the artefact to be removed
    assert "country_agree" in report["binding_constraints"]
    assert coefficients["country_agree"] == pytest.approx(0.0, abs=1e-9)
    assert report["pairs_test_auc"] > 0.9

    # The standard-library server gives the probabilities the fitted weights give.
    rows = bpf.read_table(tables / "dev.csv.gz")
    x = tc.Table(rows).matrix(tc.INPUTS)
    served = mm.LogisticMatcher(coefficients, saved["intercept"])
    expected = tc.predict(x, np.array([coefficients[n] for n in tc.INPUTS]), saved["intercept"])
    for k in range(0, 500, 41):
        assert served.probability(rows[k]) == pytest.approx(expected[k], abs=1e-9)


def test_the_committed_model_obeys_its_constraints():
    saved = json.loads((mm.ROOT / "models" / "pairs_lr_v2.json").read_text())
    assert list(saved["coefficients"]) == list(tc.INPUTS)
    for name, weight in saved["coefficients"].items():
        low, high = tc.bounds_for(name)
        assert (low is None or weight >= low - 1e-9) and (high is None or weight <= high + 1e-9), name
    assert saved["coefficients"]["country_agree"] == pytest.approx(0.0, abs=1e-9)
