"""Tests for src/train_matcher.py, on small synthetic feature tables.

Run:
    pytest tests/test_train_matcher.py
"""

import json
import random

import numpy as np
import pytest

import build_pair_features as bpf
import matcher_model as mm
import pair_features as pf
import train_matcher as tm


def fake_rows(split, n, seed):
    """Rows where a high name score and an agreeing birth year mean a match, with some noise."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = int(rng.random() < 0.6)
        name = min(1.0, max(0.0, rng.gauss(0.9 if label else 0.45, 0.12)))
        birth = rng.choice(["agree", "missing"]) if label else rng.choice(["conflict", "missing", "missing"])
        scores = [round(min(1.0, max(0.0, name + rng.gauss(0, 0.05))), 4) for _ in range(5)]
        rows.append([i, split, label, "" if split == "train" else rng.choice([0, 1]), "Person", "Person",
                     "Latin on both sides only", rng.choice(pf.SLICES), 0, *scores, birth,
                     rng.choice(["agree", "missing", "conflict"]), "missing"])  # fmt: skip
    return rows


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    path = tmp_path_factory.mktemp("tables")
    for split, n, seed in (("train", 1500, 1), ("dev", 500, 2), ("test", 500, 3)):
        bpf.write_table(path / f"{split}.csv.gz", iter(fake_rows(split, n, seed)))
    return path


def test_auc_agrees_with_the_standard_library_version():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    score = np.round(rng.random(400), 1)  # many ties
    assert tm.auc(y, score) == pytest.approx(pf.auc(score.tolist(), y.tolist()))
    assert tm.auc(np.array([1, 1]), np.array([0.2, 0.9])) is None


def test_log_loss():
    assert tm.log_loss(np.array([1, 0]), np.array([0.9, 0.2])) == pytest.approx(-(np.log(0.9) + np.log(0.8)) / 2)
    assert tm.log_loss(np.array([1]), np.array([1.0])) < 1e-9  # clipped, not infinite


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    threshold, f1 = tm.best_f1_threshold(y, p)
    assert f1 == 1.0 and 0.3 < threshold <= 0.6


def test_table_encoding_matches_the_serving_encoding(tables):
    rows = bpf.read_table(tables / "dev.csv.gz")
    table = tm.Table(rows)
    matrix = table.matrix(mm.INPUTS)
    for k in (0, 7, 100, 333):
        served = mm.encode(rows[k])
        assert matrix[k].tolist() == pytest.approx([served[name] for name in mm.INPUTS])


def test_fit_learns_the_planted_signal(tables):
    train = tm.Table(bpf.read_table(tables / "train.csv.gz"))
    model = tm.fit(train, mm.INPUTS, 1.0)
    weights = dict(zip(mm.INPUTS, model.coef_[0], strict=True))
    assert weights["birth_conflict"] < 0 < weights["birth_agree"]
    assert sum(weights[n] for n in mm.NAME_SCORES) > 3  # the name scores carry a large positive total weight


def test_training_end_to_end_and_the_served_model_equals_sklearn(tables, tmp_path):
    model_path, report_path = tmp_path / "model.json", tmp_path / "report.json"
    tm.main(["--tables", str(tables), "--model", str(model_path), "--report", str(report_path), "--bootstraps", "4"])
    report = json.loads(report_path.read_text())
    assert set(report["ablation_dev"]) == set(mm.FEATURE_INPUTS)
    assert report["regularisation_tuned_on_dev"]["chosen_C"] in tm.C_GRID
    assert set(report["leakage_check"]) == {"dev", "test"}
    assert report["coefficient_stability"]["bootstrap_resamples"] == 4
    assert report["performance_at_dev_f1_threshold"]["dev"]["auc"] > 0.9

    # The standard-library server must give exactly the probabilities scikit-learn gives.
    saved = json.loads(model_path.read_text())
    served = mm.LogisticMatcher(saved["coefficients"], saved["intercept"])
    train = tm.Table(bpf.read_table(tables / "train.csv.gz"))
    dev_rows = bpf.read_table(tables / "dev.csv.gz")
    dev = tm.Table(dev_rows)
    expected = tm.probabilities(tm.fit(train, mm.INPUTS, saved["C"]), dev, mm.INPUTS)
    for k in range(0, 500, 37):
        assert served.probability(dev_rows[k]) == pytest.approx(expected[k], abs=1e-6)


def test_ablation_drops_exactly_the_inputs_of_one_feature():
    for feature, dropped in mm.FEATURE_INPUTS.items():
        kept = tuple(i for i in mm.INPUTS if i not in dropped)
        assert len(kept) == len(mm.INPUTS) - len(dropped), feature
