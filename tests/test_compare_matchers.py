"""Tests for the hand-checkable parts of src/compare_matchers.py.

Run:
    pytest tests/test_compare_matchers.py
"""

import json

import pytest

import build_pair_features as bpf
import compare_matchers as cm
import evaluate_scoring as ev
import matcher_model as mm


def row(name=0.0, skeleton=0.0, arabic=0, birth="missing", country="missing", label=1, slice_="Latin", shares="0"):
    return {"name_score": name, "skeleton_score": skeleton, "arabic_involved": arabic, "birth_state": birth,
            "country_state": country, "label": str(label), "script_slice": slice_,
            "shares_entity_with_train": shares}  # fmt: skip


@pytest.mark.parametrize(
    ("pair", "expected"),
    [
        (row(name=0.9, skeleton=0.5, country="conflict"), 0.75),  # 0.9 stays, then -0.15
        (row(name=0.5, skeleton=0.9), 0.5),  # below the 0.6 floor the skeleton does nothing
        (row(name=0.7, skeleton=1.0), 0.9),  # lifted by at most 0.2
        (row(name=0.3, skeleton=0.9, arabic=1), 0.9),  # Arabic-involved: the skeleton stands in in full
        (row(name=0.3, skeleton=0.9, arabic=1, birth="conflict"), 0.6),  # then 0.9 - 0.30
        (row(name=0.95, birth="agree", country="agree"), 1.0),  # 0.95 + 0.05 + 0.03, clipped
        (row(name=0.2, birth="conflict", country="conflict"), 0.0),  # clipped at zero
        (row(name=0.9, birth="close", country="missing"), 0.9),  # a year apart and missing do nothing
    ],
)
def test_hand_tuned_pair_score(pair, expected):
    assert cm.hand_tuned_pair_score(pair) == pytest.approx(expected)


def test_counts_at():
    rows = [row(label=1), row(label=1), row(label=0), row(label=0), row(label=0)]
    scores = [0.9, 0.4, 0.8, 0.3, 0.2]
    c = cm.counts_at(rows, scores, 0.5)  # alerts: pairs 1 and 3
    assert (c["pairs"], c["matches"], c["non_matches"]) == (5, 2, 3)
    assert (c["recall"], c["precision"]) == (0.5, 0.5)
    assert c["false_positive_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_matched_recall_and_best_f1_thresholds():
    rows = [row(label=1), row(label=1), row(label=1), row(label=0), row(label=0)]
    scores = [0.9, 0.8, 0.5, 0.45, 0.1]
    assert cm.matched_recall_threshold(rows, scores, 1.0) == 0.5  # all three matches need a threshold of 0.5 or less
    assert cm.matched_recall_threshold(rows, scores, 2 / 3) == 0.8
    assert cm.best_f1(rows, scores, [t / 100 for t in range(5, 100, 5)]) == 0.5


def test_sliced_splits_by_script_slice():
    rows = [row(slice_="Latin", label=1), row(slice_="Arabic", label=1), row(slice_="Arabic", label=0)]
    out = cm.sliced(rows, [0.9, 0.9, 0.9], 0.5)
    assert set(out) == {"all", "Latin", "Arabic"}
    assert out["Arabic"]["pairs"] == 2 and out["Arabic"]["precision"] == 0.5


def screening_rows(scores_true, scores_other, negatives):
    positives = [
        ("variant", "variant", t, o, t is not None and t >= o, "Latin")
        for t, o in zip(scores_true, scores_other, strict=True)
    ]
    negs = [("synthetic", "Latin", None, s, False, "Latin") for s in negatives]
    return positives, negs


def test_screening_threshold_choices():
    pos, neg = screening_rows([0.9, 0.8, 0.7, 0.6], [0.1] * 4, [0.85, 0.5, 0.3, 0.2])
    # One false alarm in four (25%) is allowed: from 0.51 only the 0.85 negative alerts, while at 0.50 two do.
    assert cm.choose_threshold_for_fpr(pos, neg, 0.25) == 0.51
    assert cm.choose_threshold_for_fpr(pos, neg, 0.0) == 0.86  # none allowed: above the 0.85 negative
    best = cm.choose_threshold_for_f1(pos, neg)
    assert ev.metrics_at(pos, neg, best)["f1"] == max(ev.metrics_at(pos, neg, t)["f1"] for t in cm.THRESHOLDS)


def test_part_a_on_small_tables(tmp_path):
    head = ["Person", "Person", "Latin on both sides only"]
    rows = []
    for i in range(40):
        match = i % 2 == 0
        name = 0.95 if match else 0.3
        rows.append([i, "dev", int(match), 0, *head, "Latin", 0, *[name] * 5, "missing", "missing", "missing"])
    for split in ("dev", "test"):
        bpf.write_table(tmp_path / f"{split}.csv.gz", iter([[*r[:1], split, *r[2:]] for r in rows]))
    model = mm.LogisticMatcher({"name_score": 10.0}, intercept=-5.0)
    result = cm.part_a(tmp_path, model, model_threshold=0.5)
    trained = result["systems"]["trained_threshold_best_f1_on_pairs_dev"]["all_pairs"]["all"]
    hand = result["systems"]["hand_tuned_frozen_threshold"]["all_pairs"]["all"]
    assert trained["recall"] == 1.0 and trained["false_positive_rate"] == 0.0
    assert hand["recall"] == 1.0 and hand["false_positive_rate"] == 0.0
    assert result["auc"]["trained"]["test"] == 1.0 and result["test_pairs"] == 40
    assert json.dumps(result)  # serialisable
