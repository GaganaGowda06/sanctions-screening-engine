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
