"""The final hold-out must stay exactly as frozen and must not be read by any scoring or tuning code.

Run:
    pytest tests/test_holdout_frozen.py
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "holdout" / "final_holdout.json"


def test_final_holdout_matches_its_recorded_checksum():
    recorded = (ROOT / "holdout" / "final_holdout.sha256").read_text().strip()
    assert hashlib.sha256(HOLDOUT.read_bytes()).hexdigest() == recorded


def test_only_the_generator_and_the_final_evaluation_name_the_final_holdout_file():
    # evaluate_scoring.py wrote it once, and evaluate_final.py scores it once. Nothing else may name the file,
    # so nothing else can read it, tune on it or score it.
    allowed = {"evaluate_scoring.py", "evaluate_final.py"}
    for path in (ROOT / "src").glob("*.py"):
        if path.name not in allowed:
            assert "final_holdout.json" not in path.read_text(), path.name


def test_the_final_evaluation_scored_the_frozen_file():
    import json

    report = json.loads((ROOT / "reports" / "final_evaluation.json").read_text())
    assert report["checksum"]["ok"] is True
    assert report["checksum"]["holdout_sha256"] == hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()
    assert report["engine_check"]["mismatches"] == 0


HOLDOUT2 = ROOT / "holdout" / "screening_holdout_v2.json"


def test_screening_holdout_v2_matches_its_recorded_checksum_once_built():
    if not HOLDOUT2.exists():
        return
    recorded = (ROOT / "holdout" / "screening_holdout_v2.sha256").read_text().strip()
    assert hashlib.sha256(HOLDOUT2.read_bytes()).hexdigest() == recorded


def test_only_the_builder_names_the_second_screening_holdout_file():
    # build_screening_holdout.py writes it once. The final evaluation of the trained matcher will be the only
    # other file allowed to name it, and it does not exist yet.
    for path in (ROOT / "src").glob("*.py"):
        # the third and fourth hold-outs' builders read the second one only to keep their own entities disjoint
        builders = ("build_screening_holdout.py", "build_screening_holdout_v3.py", "build_screening_holdout_v4.py")
        if path.name not in builders:
            assert "screening_holdout_v2.json" not in path.read_text(), path.name


def test_the_second_holdouts_were_each_scored_once_on_the_frozen_files():
    import json

    screening = json.loads((ROOT / "reports" / "screening_holdout_v2_evaluation.json").read_text())
    assert screening["checksum"]["ok"] is True
    assert screening["checksum"]["holdout_sha256"] == hashlib.sha256(HOLDOUT2.read_bytes()).hexdigest()
    assert screening["engine_check"]["mismatches"] == 0

    pairs_report = json.loads((ROOT / "reports" / "pairs_final_evaluation.json").read_text())
    manifest = json.loads((ROOT / "splits" / "pairs" / "manifest.json").read_text())
    assert pairs_report["frozen"]["holdout_membership_sha256"] == manifest["final_holdout_membership_sha256"]
    assert pairs_report["frozen"]["dataset_sha256"] == manifest["dataset_sha256"]
    assert pairs_report["code_commit"] == screening["code_commit"]  # scored by the same committed code


def test_third_screening_holdout_matches_its_checksum_and_is_disjoint_from_the_second():
    import json

    path = ROOT / "holdout" / "screening_holdout_v3.json"
    recorded = (ROOT / "holdout" / "screening_holdout_v3.sha256").read_text().strip()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded

    def ids(p):
        return {q["true_id"] for q in json.loads(p.read_text())["queries"] if q.get("true_id")}

    assert not ids(path) & ids(HOLDOUT2)
