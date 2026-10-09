"""Tests for the external-validation plumbing (analysis/pairs_external.py and pairs_external_score.py).

A first scoring run was thrown away because each left entity was scored against itself (the record tuple was in the
wrong order, so every score was 1.0). These tests pin the orientation, the protocol and the arithmetic. They need no
data.

Run:
    pytest tests/test_pairs_external.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import pairs_external as pe  # noqa: E402
import pairs_external_score as ps  # noqa: E402


def entity(eid: str, schema: str, **props: list[str]) -> dict:
    return {"id": eid, "caption": eid, "schema": schema, "properties": props}


PAIRS = [
    {
        "judgement": "positive",
        "left": entity(
            "L1", "Person", name=["Ivan Petrov"], alias=["Iwan Petrow"], birthDate=["1970-01-01"], country=["ru"]
        ),
        "right": entity("R1", "Person", name=["Petrov Ivan Ivanovich"], birthDate=["1970"], nationality=["ru"]),
    },
    {
        "judgement": "negative",
        "left": entity("L2", "Company", name=["Acme"]),
        "right": entity("R2", "Company", name=["Zeta"]),
    },
    {
        "judgement": "positive",
        "left": entity("L3", "Occupancy", name=["x"]),
        "right": entity("R3", "Occupancy", name=["y"]),
    },
    {
        "judgement": "positive",
        "left": entity("L1", "Person", name=["Ivan Petrov"]),
        "right": entity("R1", "Person", name=["Ivan Petrov"]),
    },
    {"judgement": "positive", "left": entity("L5", "Person", name=["A"]), "right": entity("R5", "Person")},
    {"judgement": "positive", "left": entity("L6", "Person", name=["B"]), "right": entity("R6", "Person", name=["B"])},
    {"judgement": "negative", "left": entity("R6", "Person", name=["B"]), "right": entity("L6", "Person", name=["B"])},
]


@pytest.fixture
def stream(monkeypatch):
    monkeypatch.setattr(pe, "iter_pairs", lambda path=None: iter(enumerate(PAIRS)))
    out = list(ps.tasks())
    return [t for t in out if t[0] != "stats"], dict(out[-1][1])


def test_the_task_carries_the_left_names_as_queries_and_the_right_entity_as_the_record(stream):
    tasks, _ = stream
    label, kind, script, cross, ln, rn, l_dob, l_ctry, r_dob, r_ctry, l_rec, r_rec, bd, ct, shared, schemas = tasks[0]
    assert label == 1 and kind == "person_person" and script == "Latin" and cross is False
    assert ln == ["Ivan Petrov", "Iwan Petrow"] and rn == ["Petrov Ivan Ivanovich"]
    assert (l_dob, l_ctry, r_dob, r_ctry) == ("1970-01-01", "ru", "1970", "ru")
    assert l_rec == (ln, ["1970-01-01"], ["ru"])  # the LEFT record is the left entity ...
    assert r_rec == (rn, ["1970"], ["ru"])  # ... and the RIGHT record is the right one (they were swapped once)
    assert bd is True and ct is True and shared is False  # 'Ivan Petrov' is not 'Petrov Ivan Ivanovich'
    assert schemas == "Person/Person"


def test_work_scores_the_left_names_against_the_right_record_and_the_reverse_the_other_way(monkeypatch, stream):
    tasks, _ = stream
    calls = []

    def fake(names, dob, country, record_args):
        calls.append((tuple(names), dob, country, tuple(record_args[0])))
        return 0.5

    monkeypatch.setattr(ps, "score_against", fake)
    ps.work(tasks[0])
    assert calls[0] == (("Ivan Petrov", "Iwan Petrow"), "1970-01-01", "ru", ("Petrov Ivan Ivanovich",))
    assert calls[1] == (("Petrov Ivan Ivanovich",), "1970", "ru", ("Ivan Petrov", "Iwan Petrow"))
    assert calls[0][0] != calls[0][3]  # a query is never scored against its own entity


def test_relationship_records_duplicates_conflicts_and_nameless_sides_are_counted_not_scored(stream):
    tasks, stats = stream
    assert len(tasks) == 2  # pair 0 and the company pair; the duplicate of L1/R1 is not scored again
    assert stats["not_an_entity_pair"] == 1  # Occupancy
    assert stats["duplicate_line"] == 2  # L1/R1 again, and R6/L6 (the same unordered pair as L6/R6)
    assert stats["a_side_has_no_name_property"] == 1
    assert stats.get("conflicting_labels_left_out", 0) == 1  # L6/R6 was labelled both ways


def test_entity_types_scripts_and_cross_script():
    assert pe.entity_type(entity("a", "Person"), entity("b", "Person")) == "person_person"
    assert pe.entity_type(entity("a", "Company"), entity("b", "Organization")) == "organisation_organisation"
    assert pe.entity_type(entity("a", "Person"), entity("b", "LegalEntity")) == "person_organisation"
    assert pe.entity_type(entity("a", "Vessel"), entity("b", "Vessel")) == "other_entity_types"
    assert pe.script_bucket(["John Smith"]) == "Latin"
    assert pe.script_bucket(["John Smith", "Иван"]) == "Cyrillic"
    assert pe.script_bucket(["Иван", "محمد"]) == "Arabic"  # the engine's order: Arabic wins over Cyrillic
    assert pe.cross_script(["John Smith"], ["Иван Петров"]) and not pe.cross_script(["John", "Иван"], ["Иван"])


def test_pair_key_is_unordered():
    a, b = PAIRS[5], PAIRS[6]
    assert pe.pair_key(a) == pe.pair_key(b)


def test_confusion_and_rates_by_hand():
    label = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    c = ps.confusion(label, pred)
    assert c == {"tp": 3, "fp": 1, "fn": 1, "tn": 5}
    r = ps.rates(c)
    assert r["precision"] == 0.75 and r["recall"] == 0.75 and r["f1"] == 0.75
    assert r["false_positive_rate"] == pytest.approx(1 / 6) and r["positive_share"] == 0.4
    # the same recall and false-alert rate with as many negatives as positives: precision = r / (r + f)
    assert r["precision_if_half_the_pairs_were_positive"] == pytest.approx(0.75 / (0.75 + 1 / 6))


def test_a_constant_score_has_auc_one_half_and_a_perfect_one_has_auc_one():
    label = np.array([1, 0, 1, 0])
    assert ps.auc(label, np.ones(4)) == 0.5
    assert ps.auc(label, np.array([0.9, 0.1, 0.8, 0.2])) == 1.0
    assert ps.auc(label, np.array([0.1, 0.9, 0.2, 0.8])) == 0.0
    assert ps.auc(np.ones(3, dtype=int), np.arange(3.0)) is None


def test_the_f1_interval_brackets_the_point_estimate_and_is_reproducible():
    c = {"tp": 700, "fp": 100, "fn": 200, "tn": 400}
    lo, hi = ps.f1_interval(c)
    f1 = ps.rates(c)["f1"]
    assert lo < f1 < hi and [lo, hi] == ps.f1_interval(c) and hi - lo < 0.06


def test_the_protocol_uses_the_engines_one_threshold():
    assert pe.THRESHOLD == 0.86


def test_a_shared_name_is_the_same_tokens_in_any_order_ignoring_case_and_accents():
    assert ps.shares_a_name(["Ivan Petrov", "x"], ["PETROV, ivan"])
    assert ps.shares_a_name(["José García"], ["jose garcia"])
    assert not ps.shares_a_name(["Ivan Petrov"], ["Ivan Ivanovich Petrov"])
    assert not ps.shares_a_name(["Ivan"], ["Ivan Petrov", "Petrov"])
