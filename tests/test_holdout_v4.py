"""The fourth screening hold-out: frozen, checksummed, and disjoint from every earlier set.

Dev and test entities cannot be checked here without data/, so the builder checks them when it builds (it exits if any
entity was used before); these tests check what the frozen files alone can show.

Run:
    pytest tests/test_holdout_v4.py
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

HOLDOUT = Path(__file__).resolve().parent.parent / "holdout"
NAMES = ("final_holdout", "screening_holdout_v2", "screening_holdout_v3", "screening_holdout_v4")


def load(name: str) -> dict:
    return json.loads((HOLDOUT / f"{name}.json").read_text(encoding="utf-8"))


def entities(payload: dict) -> set[str]:
    return {q.get("true_id") or q.get("entity") for q in payload["queries"] if q.get("true_id") or q.get("entity")}


@pytest.mark.parametrize("name", NAMES)
def test_each_frozen_holdout_matches_its_recorded_checksum(name):
    recorded = (HOLDOUT / f"{name}.sha256").read_text().strip()
    assert hashlib.sha256((HOLDOUT / f"{name}.json").read_bytes()).hexdigest() == recorded


def test_the_fourth_holdout_shares_no_entity_with_the_first_three():
    v4 = entities(load("screening_holdout_v4"))
    assert len(v4) == 2900
    for name in NAMES[:3]:
        assert not v4 & entities(load(name)), name


def test_the_fourth_holdout_has_the_same_design_as_the_second_and_third():
    def shape(name: str) -> dict:
        return dict(Counter(f"{q['kind']}/{q['group']}" for q in load(name)["queries"]))

    v4 = shape("screening_holdout_v4")
    assert v4["positive/variant"] == 800 and v4["positive/other_same_script"] == 200
    assert sum(v for k, v in v4.items() if k.startswith("positive/")) == 2400
    assert sum(v for k, v in v4.items() if k.startswith("synthetic/")) == 1900
    assert sum(v for k, v in v4.items() if k.startswith("near_miss/")) == 500
    assert load("screening_holdout_v4")["tag"] == "holdout4"
    assert {k.split("/")[0] + "/" + k.split("/")[1] for k in v4} == {
        k.split("/")[0] + "/" + k.split("/")[1] for k in shape("screening_holdout_v3")
    }


def test_every_positive_has_a_list_of_kept_names():
    payload = load("screening_holdout_v4")
    assert all(q["true_id"] in payload["kept_names"] for q in payload["queries"] if q["kind"] == "positive")
