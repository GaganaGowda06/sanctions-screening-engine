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


def test_only_the_generator_touches_the_final_holdout_file():
    # evaluate_scoring.py writes it once. Nothing else may name the file, so nothing else can read or score it.
    for path in (ROOT / "src").glob("*.py"):
        if path.name != "evaluate_scoring.py":
            assert "final_holdout.json" not in path.read_text(), path.name
