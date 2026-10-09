"""The terminal demo (scripts/demo.py) runs every step against a throwaway store and leaves the real one alone.

It runs against the downloaded list, so it takes about 20 seconds, and is skipped where the list is absent.

Marked slow: it runs with the slow suite, not the default one.

Run:
    pytest -m slow tests/test_demo.py
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "data" / "sanctions" / "entities.ftm.json"


@pytest.mark.slow
@pytest.mark.skipif(not LIST.exists(), reason="needs the downloaded list (python src/download_data.py)")
def test_the_demo_runs_all_eight_steps_and_never_touches_the_real_store(tmp_path):
    # The real list is used because a fuzzy match (a misspelling that alerts with a score below 1.0) needs a list big
    # enough for the name's rarity weights; on a few-entity list every one-letter variant scores exactly 1.0.
    real_store = tmp_path / "real-audit"  # pointed to by the environment; the demo must neither use nor create it
    env = {"PATH": "/usr/bin:/bin", "SCREENING_AUDIT_DIR": str(real_store)}

    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo.py"), "--input", str(LIST)],
        capture_output=True, text=True, env=env, timeout=120,
    )  # fmt: skip

    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert [int(n) for n in re.findall(r"^=== Step (\d+):", out, re.M)] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert "FAILED" not in out and "All 8 steps succeeded" in out
    for line in (
        "Decision: ALERT",
        "edit applied: one consonant",
        "Score: exact name",
        "with a lower score",
        "Audit clock simulated to",
        "real deployments never backdate.",
        "Decision: CLEAR",
        "Replay status: identical",
        "Replay after purge: purged",
        "the chain still verifies after the purge",
    ):
        assert line in out, line
    store = re.search(r"Throwaway audit store: (\S+)", out).group(1)
    assert not Path(store).exists() and not real_store.exists()
