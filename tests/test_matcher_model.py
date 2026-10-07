"""Tests for the serving side of the trained matcher (src/matcher_model.py).

Run:
    pytest tests/test_matcher_model.py
"""

import json
import math

import pytest

import matcher_model as mm


def features(**overrides):
    base = {"name_score": 0.0, "skeleton_score": 0.0, "weighted_score": 0.0, "phonetic_score": 0.0,
            "arabic_score": 0.0, "birth_state": "missing", "country_state": "missing",
            "identifier_state": "missing"}  # fmt: skip
    return base | overrides


def test_inputs_and_features_line_up():
    assert len(mm.INPUTS) == 12 and len(set(mm.INPUTS)) == 12
    supplied = [name for names in mm.FEATURE_INPUTS.values() for name in names]
    assert sorted(supplied) == sorted(mm.INPUTS)  # every input belongs to exactly one of the eight features
    assert len(mm.FEATURE_INPUTS) == 8


def test_encode_states_as_indicators_with_missing_as_the_reference():
    row = mm.encode(features(name_score="0.8", birth_state="conflict", country_state="agree"))
    assert row["name_score"] == 0.8
    assert (row["birth_agree"], row["birth_close"], row["birth_conflict"]) == (0.0, 0.0, 1.0)
    assert (row["country_agree"], row["country_conflict"]) == (1.0, 0.0)
    assert (row["identifier_agree"], row["identifier_conflict"]) == (0.0, 0.0)
    assert all(v == 0.0 for k, v in mm.encode(features()).items())  # nothing to compare, nothing scores


def test_probability_is_the_logistic_of_the_weighted_sum():
    model = mm.LogisticMatcher({"name_score": 2.0, "birth_conflict": -3.0}, intercept=-1.0)
    z = 2.0 * 0.8 - 1.0
    assert model.probability(features(name_score=0.8)) == pytest.approx(1 / (1 + math.exp(-z)))
    z = 2.0 * 0.8 - 1.0 - 3.0
    assert model.probability(features(name_score=0.8, birth_state="conflict")) == pytest.approx(
        1 / (1 + math.exp(-z))
    )
    assert mm.LogisticMatcher({}, 0.0).probability(features()) == 0.5


def test_probability_does_not_overflow():
    model = mm.LogisticMatcher({"name_score": 1000.0}, intercept=0.0)
    assert model.probability(features(name_score=1.0)) == pytest.approx(1.0)
    assert 0.0 <= model.probability(features(name_score=-1.0)) < 1e-300


def test_a_model_loads_from_json(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"coefficients": {"name_score": 4.0}, "intercept": -2.0}))
    assert mm.LogisticMatcher.load(path).probability(features(name_score=0.5)) == pytest.approx(0.5)


def test_the_committed_model_is_complete():
    data = json.loads(mm.MODEL_PATH.read_text())
    assert data["inputs"] == list(mm.INPUTS) and set(data["coefficients"]) == set(mm.INPUTS)
    assert data["C"] > 0 and 0 < data["dev_f1_threshold"] < 1
    assert len(data["idf_table_sha256"]) == 64 and len(data["trained_on_feature_table_sha256"]) == 64
