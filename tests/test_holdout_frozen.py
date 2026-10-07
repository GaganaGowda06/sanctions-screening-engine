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
        if path.name != "build_screening_holdout.py":
            assert "screening_holdout_v2.json" not in path.read_text(), path.name
