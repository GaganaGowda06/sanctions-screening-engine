"""Tests for the metric and query-building helpers in src/evaluate_scoring.py, on tiny hand-built data.

Run:
    pytest tests/test_evaluate_scoring.py
"""

import random
import re

import pytest

import evaluate_scoring as ev
from normalise import normalise
from scoring import Components

PARAMS = ev.Params(skeleton_floor=0.5, skeleton_lift=0.1, penalty_index=0)


def names(value):
    return (value,) * len(ev.LENGTH_PENALTIES)


def query(kind, group, comps, true_id=None):
    return {"kind": kind, "group": group, "true_id": true_id, "comps": comps}


def test_final_scores_find_true_score_best_other_and_top1():
    comps = [Components("t", names(0.9), 0.0, 0, 0), Components("o", names(0.8), 0.0, 0, 0)]
    q = query("positive", "variant", comps, "t")
    row = ev.final_scores([q], PARAMS, True, True)[0]
    assert row == ("variant", "variant", pytest.approx(0.9), pytest.approx(0.8), True)


def test_final_scores_attributes_can_change_the_top_entity():
    comps = [Components("t", names(0.7), 0.0, 0.10, 0.0), Components("o", names(0.75), 0.0, 0.0, 0.0)]
    q = query("positive", "variant", comps, "t")
    assert ev.final_scores([q], PARAMS, True, True)[0][4] is True  # 0.8 against 0.75
    assert ev.final_scores([q], PARAMS, False, True)[0][4] is False  # 0.7 against 0.75


def test_final_scores_missing_true_entity():
    q = query("positive", "variant", [Components("o", names(0.8), 0.0, 0, 0)], "t")
    row = ev.final_scores([q], PARAMS, True, True)[0]
    assert row[2] is None and row[4] is False


def test_final_scores_negative_uses_kind_and_script():
    q = query("synthetic", "Arabic", [Components("o", names(0.6), 0.0, 0, 0)])
    assert ev.final_scores([q], PARAMS, True, True) == [("synthetic", "Arabic", None, pytest.approx(0.6), False)]


POS = [
    ("variant", "variant", 0.9, 0.0, True),  # true entity alerted at 0.8: true positive
    ("variant", "variant", 0.7, 0.85, False),  # true entity below, a wrong entity alerts: false positive
    ("variant", "variant", None, 0.0, False),  # nothing alerts: false negative
    ("other_to_latin", "other_to_latin", None, 0.0, False),  # reported by group, not in the headline
]
NEG = [
    ("synthetic", "Latin", None, 0.9, False),  # alerts
    ("synthetic", "Arabic", None, 0.5, False),
    ("near_miss", "Latin", None, 0.85, False),  # alerts
    ("near_miss", "Cyrillic", None, 0.2, False),
]


def test_metrics_at_threshold_0_8():
    m = ev.metrics_at(POS, NEG, 0.8)
    # Headline positives are the first three rows: 1 TP, 1 wrong-entity alert, 1 FN.
    # False positives: 2 negatives alerted + 1 wrong-entity alert = 3.
    assert m["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["precision"] == pytest.approx(1 / 4, abs=1e-4)
    assert m["f1"] == pytest.approx(2 * 0.25 * (1 / 3) / (0.25 + 1 / 3), abs=1e-4)
    assert m["top1_accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["false_positive_rate"] == 0.5  # 2 of 4 negatives
    assert m["fpr_synthetic"] == 0.5 and m["fpr_near_miss"] == 0.5
    assert m["fpr_synthetic_by_script"] == {"Latin": 1.0, "Cyrillic": None, "Arabic": 0.0}
    # Alerted: positives 1 and 2 (the wrong-entity alert counts), plus 2 negatives, out of 3 + 4 queries.
    assert m["alert_rate_overall"] == pytest.approx(4 / 7, abs=1e-4)
    assert m["recall_by_group"]["variant"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recall_by_group"]["other_to_latin"] == 0.0
    assert m["recall_by_group"]["cyr_to_latin"] is None  # no queries in that group


def test_metrics_at_a_threshold_nothing_reaches():
    m = ev.metrics_at(POS, NEG, 0.99)
    assert (m["recall"], m["precision"], m["f1"], m["false_positive_rate"]) == (0.0, 0.0, 0.0, 0.0)


def test_tune_picks_the_highest_threshold_that_separates_perfectly():
    dev_pos = [query("positive", "variant", [Components("t", names(0.9), 0.0, 0, 0)], "t") for _ in range(3)]
    dev_neg = [query("synthetic", "Latin", [Components("n", names(0.6), 0.0, 0, 0)]) for _ in range(3)]
    params, threshold, f1, marginals = ev.tune(dev_pos, dev_neg, True, True)
    assert (threshold, f1) == (0.9, 1.0)  # separation holds for 0.61 to 0.90, ties go to the higher end
    assert isinstance(params, ev.Params)
    assert set(marginals) == {"skeleton_floor", "skeleton_lift", "length_penalty"}
    assert set(marginals["length_penalty"]) == {"0.0", "0.2", "0.4", "0.6"}


def test_conflicting_birth_date_differs_by_at_least_eight_years():
    rng = random.Random(0)
    for _ in range(20):
        result = ev.conflicting_birth_date(rng, ["1970", "1972-03-04"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", result)
        assert min(abs(int(result[:4]) - y) for y in (1970, 1972)) >= 2
    assert ev.conflicting_birth_date(rng, ["unknown"]) is None


def person(eid, names, dobs=(), countries=(), target=True):
    return {"id": eid, "schema": "Person", "target": target, "names": list(names),
            "birth_dates": list(dobs), "countries": list(countries)}  # fmt: skip


def test_near_miss_has_conflicting_birth_date_and_country(monkeypatch):
    monkeypatch.setattr(ev, "NEAR_MISS", {"dev": 1, "test": 0})
    targets = [person("e1", ["Ivan Petrov"], ["1970"], ["ru"])]
    out = ev.build_near_misses(targets, set(), ["us", "ru", "de"], random.Random(1))
    assert len(out) == 1
    q = out[0]
    assert q["name"] == "Ivan Petrov" and q["true_id"] is None and q["split"] == "dev"
    assert q["country"] != "ru"
    assert abs(int(q["dob"][:4]) - 1970) >= 8


def test_near_miss_skips_entities_already_used_and_those_without_attributes(monkeypatch):
    monkeypatch.setattr(ev, "NEAR_MISS", {"dev": 1, "test": 0})
    targets = [person("e1", ["Ivan Petrov"], ["1970"], ["ru"]), person("e2", ["Anna Sidorova"])]
    assert ev.build_near_misses(targets, {"e1"}, ["us"], random.Random(1)) == []


def test_synthetic_names_match_no_existing_name(monkeypatch):
    monkeypatch.setattr(ev, "SYNTHETIC", {"dev": {"Latin": 5}, "test": {"Latin": 5, "Cyrillic": 0}})
    names = ["Ivan Petrov", "Anna Sidorova", "Boris Kuznetsov", "Maria Orlova"]
    people = [person(f"e{i}", [n]) for i, n in enumerate(names)]
    existing = {normalise(n) for n in names}
    existing_sorted = {" ".join(sorted(n.split())) for n in existing}
    out = ev.build_synthetic(people, people, ["1970"], ["ru"], random.Random(2))
    assert len(out) == 10
    for q in out:
        norm = normalise(q["name"])
        assert norm not in existing and " ".join(sorted(norm.split())) not in existing_sorted
        assert q["kind"] == "synthetic" and q["group"] == "Latin" and q["true_id"] is None
        assert q["dob"] in (None, "1970") and q["country"] in (None, "ru")
