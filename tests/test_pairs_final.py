"""Tests for src/evaluate_pairs_final.py on a tiny fake dataset. The real hold-out is never read.

Run:
    pytest tests/test_pairs_final.py
"""

import gzip
import json

import pytest

import build_pair_splits as bps
import clean_pairs
import evaluate_pairs_final as epf


def record(i, positive):
    def side(tag, names):
        return {"id": f"{tag}{i}", "caption": names[0], "schema": "Person",
                "properties": {"name": names, "birthDate": ["1970"] if positive else ["1950"]}}  # fmt: skip

    same = [f"Person Number{i}x Alpha"]
    other = same if positive else [f"Different Name{i}y Beta"]
    right = side("r", other)
    right["properties"]["birthDate"] = ["1970"]
    return {"judgement": "positive" if positive else "negative", "left": side("l", same), "right": right}


@pytest.fixture
def world(tmp_path, monkeypatch):
    records = [record(i, i % 3 != 0) for i in range(900)]
    path = tmp_path / "pairs-20251209.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in records) + "\n")
    (tmp_path / "SOURCE.json").write_text(json.dumps({"file": path.name, "sha256": "x" * 64}))
    split_dir = tmp_path / "splits"
    common = ["--input", str(path), "--split-dir", str(split_dir)]
    bps.main([*common, "--report", str(tmp_path / "splits.json")])
    clean_pairs.main(common)
    model = tmp_path / "model.json"
    model.write_text(json.dumps({
        "coefficients": {"name_score": 6.0, "weighted_score": 6.0, "birth_agree": 1.0, "birth_conflict": -3.0},
        "intercept": -6.0, "dev_f1_threshold": 0.5,
    }))  # fmt: skip
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"pairs_test": {"auc": {"trained": {"test": 1.0}},
        "thresholds": {"hand_tuned_best_f1_on_pairs_dev": 0.9}}}))  # fmt: skip
    monkeypatch.setattr(epf, "load_table", lambda: ({"person": 5, "alpha": 5, "different": 5, "beta": 5}, 100))
    args = ["--input", str(path), "--split-dir", str(split_dir), "--model", str(model),
            "--comparison", str(comparison), "--workers", "1"]  # fmt: skip
    return tmp_path, split_dir, args


def test_scores_only_the_frozen_cleaned_holdout(world):
    tmp_path, split_dir, args = world
    out = tmp_path / "out.json"
    epf.main([*args, "--output", str(out)])
    report = json.loads(out.read_text())
    cleaning = json.loads((split_dir / "cleaning.json").read_text())
    expected = sum(cleaning["by_split_after_cleaning"][k]["pairs"] for k in ("final", "final_clean"))
    assert report["pairs"] == expected > 0
    manifest = json.loads((split_dir / "manifest.json").read_text())
    assert report["frozen"]["holdout_membership_sha256"] == manifest["final_holdout_membership_sha256"]
    assert set(report["systems"]) == {"constrained_model", "hand_tuned_frozen_threshold",
                                      "hand_tuned_threshold_best_on_pairs_dev"}  # fmt: skip


def test_rates_have_intervals_and_the_split_by_shared_entity_is_there(world):
    tmp_path, _, args = world
    out = tmp_path / "out.json"
    epf.main([*args, "--output", str(out)])
    model = json.loads(out.read_text())["systems"]["constrained_model"]
    assert set(model) >= {"threshold", "all_pairs", "share_an_entity_with_train", "no_entity_in_train"}
    everything = model["all_pairs"]
    assert everything["auc"] == 1.0  # identical names against clearly different ones
    for name in ("recall", "precision", "false_positive_rate"):
        low, high = everything[f"{name}_ci95"]
        assert low <= everything[name] <= high
    shared, unshared = model["share_an_entity_with_train"], model["no_entity_in_train"]
    assert everything["pairs"] == shared["pairs"] + unshared["pairs"]


def test_it_refuses_to_run_twice(world):
    tmp_path, _, args = world
    out = tmp_path / "out.json"
    epf.main([*args, "--output", str(out)])
    before = out.read_text()
    with pytest.raises(SystemExit, match="not run twice"):
        epf.main([*args, "--output", str(out)])
    assert out.read_text() == before


def test_it_refuses_a_tampered_membership_digest(world):
    tmp_path, split_dir, args = world
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["final_holdout_membership_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="not the frozen membership"):
        epf.main([*args, "--output", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()


def test_it_refuses_a_different_dataset(world):
    tmp_path, _, args = world
    (tmp_path / "SOURCE.json").write_text(json.dumps({"file": "x", "sha256": "y" * 64}))
    with pytest.raises(SystemExit, match="does not match the frozen split manifest"):
        epf.main([*args, "--output", str(tmp_path / "out.json")])


def test_the_interval_helper():
    rows = [{"label": "1"}, {"label": "1"}, {"label": "0"}, {"label": "0"}]
    counts = {"pairs": 4, "matches": 2, "non_matches": 2, "recall": 0.5, "precision": 0.5, "false_positive_rate": 0.5}
    out = epf.with_intervals(counts, rows, [0.9, 0.1, 0.8, 0.2], 0.5)
    assert out["recall_ci95"][0] < 0.5 < out["recall_ci95"][1]
    assert out["precision_ci95"] is not None and out["false_positive_rate_ci95"] is not None
