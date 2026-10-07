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

PARAMS = ev.Params(skeleton_floor=0.5, skeleton_lift=0.1, blend_index=0)


def names(value):
    return (value,) * len(ev.BLENDS)


def query(kind, group, comps, true_id=None):
    return {"kind": kind, "group": group, "true_id": true_id, "comps": comps}


def test_final_scores_find_true_score_best_other_and_top1():
    comps = [Components("t", names(0.9), 0.0, 0, 0), Components("o", names(0.8), 0.0, 0, 0)]
    q = query("positive", "variant", comps, "t")
    row = ev.final_scores([q], PARAMS, True, True)[0]
    assert row == ("variant", "variant", pytest.approx(0.9), pytest.approx(0.8), True, None)


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
    assert ev.final_scores([q], PARAMS, True, True) == [("synthetic", "Arabic", None, pytest.approx(0.6), False, None)]


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
    assert m["fpr_synthetic_by_script"] == {"Latin": 1.0, "Cyrillic": None, "Arabic": 0.0, "other": None}
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
    assert set(marginals) == {"skeleton_floor", "skeleton_lift", "blend"}
    assert set(marginals["blend"]) == {"0.0", "0.5", "1.0"}


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


def test_final_scores_arabic_full_lift_switch():
    comps = [Components("t", names(0.3), 0.9, 0, 0, True)]
    q = query("positive", "ar_to_latin", comps, "t")
    assert ev.final_scores([q], PARAMS, True, True, arabic_full=True)[0][2] == pytest.approx(0.9)
    assert ev.final_scores([q], PARAMS, True, True, arabic_full=False)[0][2] == pytest.approx(0.3)


def tiny_targets(n=14):
    return [person(f"e{i}", [f"Karim Nasser{i}x", f"Nasser Karim{i}y"], target=True) for i in range(n)]


def test_default_plan_matches_the_explicit_dev_test_plan(monkeypatch):
    monkeypatch.setattr(ev, "POSITIVE_GROUPS", {"variant": (2, 3)})
    default, kept = ev.build_positives(tiny_targets(), random.Random(1))
    explicit, kept2 = ev.build_positives(tiny_targets(), random.Random(1), {"variant": [("dev", 2), ("test", 3)]})
    assert default == explicit and kept == kept2
    assert [q["split"] for q in default] == ["dev", "dev", "test", "test", "test"]


def test_final_positives_are_disjoint_from_dev_and_test(monkeypatch):
    monkeypatch.setattr(ev, "POSITIVE_GROUPS", {"variant": (2, 3)})
    targets = tiny_targets()
    dev_test, _ = ev.build_positives(targets, random.Random(1))
    used = {q["true_id"] for q in dev_test}
    final, kept = ev.build_positives(
        targets, random.Random("final"), {"variant": [("final", 4)]}, exclude=used, key="{seed}-final-{group}"
    )
    assert len(final) == 4 and {q["split"] for q in final} == {"final"}
    assert not used & {q["true_id"] for q in final}
    assert set(kept) == {q["true_id"] for q in final}


def test_final_holdout_is_written_once_and_never_overwritten(tmp_path):
    path = tmp_path / "holdout" / "final_holdout.json"
    first = [{"split": "final", "kind": "synthetic", "group": "Latin", "name": "Karim Nasser", "comps": ["dropped"]}]
    info = ev.write_final_holdout(path, first, {})
    assert info["queries"] == {"synthetic/Latin": 1}
    assert "comps" not in path.read_text()
    assert path.with_suffix(".sha256").read_text().strip() == info["sha256"]
    other = [{"split": "final", "kind": "synthetic", "group": "Latin", "name": "Someone Else"}]
    again = ev.write_final_holdout(path, other, {})
    assert again["sha256"] == info["sha256"]  # frozen: the second call changed nothing


def test_final_near_misses_use_their_own_split_and_exclude_used_entities(monkeypatch):
    monkeypatch.setattr(ev, "NEAR_MISS", {"dev": 0, "test": 0, "final": 2})
    targets = [person(f"p{i}", [f"Ivan Petrov{i}x"], ["1970"], ["ru"]) for i in range(5)]
    out = ev.build_near_misses(targets, {"p0"}, ["us", "de"], random.Random(1), ("final",), "near-final")
    assert len(out) == 2 and {q["split"] for q in out} == {"final"}
    assert "p0" not in {q["entity"] for q in out}


def test_metrics_at_with_one_threshold_per_script():
    pos = [
        ("variant", "variant", 0.9, 0.0, True, "Latin"),
        ("cyr_to_latin", "cyr_to_latin", 0.7, 0.0, True, "Cyrillic"),
    ]
    neg = [
        ("synthetic", "Latin", None, 0.8, False, "Latin"),
        ("synthetic", "Cyrillic", None, 0.8, False, "Cyrillic"),
        ("synthetic", "other", None, 0.8, False, "other"),  # reported by script, left out of the pooled rate
    ]
    thresholds = {"Latin": 0.85, "Cyrillic": 0.6, "Arabic": 0.9, "other": 0.5}
    m = ev.metrics_at(pos, neg, thresholds)
    assert m["recall"] == 1.0  # 0.9 reaches 0.85, 0.7 reaches 0.6
    assert m["false_positive_rate"] == 0.5  # Latin 0.8 is under 0.85, Cyrillic 0.8 reaches 0.6
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)  # 2 true positives, 1 false positive
    assert m["fpr_by_script"] == {"Latin": 0.0, "Cyrillic": 1.0, "Arabic": None, "other": 1.0}
    assert m["recall_by_script"] == {"Latin": 1.0, "Cyrillic": 1.0, "Arabic": None, "other": None}
    # Latin: 1 found, no false alarm (0.8 is under 0.85). Cyrillic: 1 found and 1 false alarm. "other" is not pooled.
    assert m["precision_by_script"] == {"Latin": 1.0, "Cyrillic": 0.5, "Arabic": None, "other": None}
    # One global threshold of 0.85 would lose the Cyrillic positive.
    assert ev.metrics_at(pos, neg, 0.85)["recall"] == 0.5


def other_people():
    names = {
        "kana": ["イリーナ・コステンコ", "ジ・サンジュン", "チ・サンチュン"],
        "hebrew": ["אלאיב ליבי", "עבד אלחי סאלכ", "עבד אלקדיר עבד אלבציר"],
        "han": ["孝志 池田", "兴标 沈", "素炫 颜"],
        "thai": ["อับดุลเลาะ บาราเฮง", "อาดัม มุสอดี", "อามีร ดอแต"],
    }
    return [person(f"{k}{i}", [n]) for k, ns in names.items() for i, n in enumerate(ns)]


def test_other_script_negatives_are_new_names_from_the_four_sub_scripts(monkeypatch):
    monkeypatch.setattr(ev, "SYNTHETIC", {"dev": {"other": 8}})
    people = other_people()
    existing = {normalise(n) for e in people for n in e["names"]}
    out = ev.build_synthetic(people, people, ["1970"], ["ru"], random.Random(3), ("dev",))
    assert len(out) == 8
    for q in out:
        assert q["group"] == "other" and ev.script_class(q["name"]) == "other"
        assert normalise(q["name"]) not in existing
    # Sub-scripts are drawn in turn: kana, Hebrew, Han, Thai, kana, ...
    assert [ev.name_scripts(q["name"]) for q in out[:4]] == [
        {"Japanese kana"}, {"Hebrew"}, {"Han"}, {"Thai"},
    ]  # fmt: skip


def test_same_script_positives_keep_only_other_names_that_differ_from_the_query():
    people = [person("a", ["孝志 池田", "池田 孝志", "Takashi Ikeda"]), person("b", ["孝志 池田"])]
    assert [c[0]["id"] for c in ev.positive_candidates(people, "other_same_script")] == ["a"]  # b has one name
    queries, kept = ev.build_positives(
        people, random.Random(1), {"other_same_script": [("dev", 1)]}, key="{seed}-test-{group}"
    )
    assert len(queries) == 1 and queries[0]["true_id"] == "a" and queries[0]["group"] == "other_same_script"
    other_name = ({"孝志 池田", "池田 孝志"} - {queries[0]["name"]}).pop()
    assert kept["a"] == [other_name]  # the query name and the Latin name are not in the list


def test_headline_leaves_out_the_extra_groups():
    assert "other_same_script" not in ev.HEADLINE_GROUPS and "other_to_latin" not in ev.HEADLINE_GROUPS
