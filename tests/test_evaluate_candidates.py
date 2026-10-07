"""Tests for the helpers in src/evaluate_candidates.py, on tiny hand-built data.

Run:
    pytest tests/test_evaluate_candidates.py
"""

import pytest

from candidates import CandidateIndex
from evaluate_candidates import evaluate, held_out, index_entities, percentile, script_class


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Mohammed Ali", "Latin"),
        ("Владимир Путин", "Cyrillic"),
        ("محمد علي", "Arabic"),
        ("张伟", "other"),
        ("たなか 张", "other"),
        ("Gazprom Нефть", None),  # mixed Latin and Cyrillic
        ("123", None),  # no letters
    ],
)
def test_script_class(name, expected):
    assert script_class(name) == expected


def test_held_out_keeps_only_the_opposite_side():
    latin, cyrillic, mixed = "Ivan Petrov", "Иван Петров", "Ivan Петров"
    # Querying with Cyrillic: only purely Latin names stay in the index.
    assert [held_out(n, "Cyrillic", "to_latin") for n in (latin, cyrillic, mixed)] == [False, True, True]
    # Querying with Latin: only purely Cyrillic names stay.
    assert [held_out(n, "Cyrillic", "from_latin") for n in (latin, cyrillic, mixed)] == [True, False, True]


def entity(eid, names, schema="Person"):
    return {"id": eid, "schema": schema, "target": True, "names": names}


def test_index_entities_removes_names_only_for_sampled_entities():
    sampled = entity("e1", ["Ivan Petrov", "Иван Петров"])
    other = entity("e2", ["Anna Petrova", "Анна Петрова"])
    condition = {"dev": [sampled], "test": [], "cls": "Cyrillic", "direction": "to_latin"}
    assert dict(index_entities([sampled, other], condition)) == {
        "e1": ["Ivan Petrov"],
        "e2": ["Anna Petrova", "Анна Петрова"],
    }
    assert dict(index_entities([sampled, other], None))["e1"] == ["Ivan Petrov", "Иван Петров"]


def test_percentile():
    assert percentile([], 0.5) == 0.0
    assert percentile([5, 1, 3], 0.5) == 3
    assert percentile(list(range(101)), 0.95) == 95


def test_evaluate_counts_hits_and_misses():
    ivan = entity("e1", ["Ivan Petrov", "Иван Петров"])
    dragon = entity("e2", ["Star Dragon", "星龍"], schema="Company")
    condition = {"dev": [ivan, dragon], "test": [], "cls": "Cyrillic", "direction": "to_latin"}
    index = CandidateIndex.build(index_entities([ivan, dragon, entity("e3", ["Maria Garcia"])], condition))
    result = evaluate(index, [ivan, dragon], {"e1": "Иван Петров", "e2": "星龍"}, examples=1)
    # Ivan is found through "ivan petrov". The Han name shares no key with "Star Dragon".
    assert result["queries"] == 2
    assert result["candidate_recall"] == 0.5
    assert result["candidate_recall_person"] == 1.0  # one Person query, found
    assert result["candidate_recall_non_person"] == 0.0  # one Company query, missed
    assert result["missed_examples"][0]["query"] == "星龍"
