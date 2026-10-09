"""The terminal demo (scripts/demo.py) runs every step against a throwaway store and leaves the real one alone.

The list is a few entities in a temporary file (with its VERSION file), so the run takes seconds.

Run:
    pytest tests/test_demo.py
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FTM = [
    {"id": "e1", "schema": "Person", "target": True,
     "properties": {"name": ["Mohammed Ali Hassan"], "birthDate": ["1970-01-01"], "country": ["iq"]}},
    {"id": "e2", "schema": "Person", "target": True,
     "properties": {"name": ["Ivan Petrov"], "birthDate": ["1980-05-05"], "country": ["ru"]}},
    {"id": "e4", "schema": "Company", "target": True, "properties": {"name": ["Acme Trading Limited"]}},
]  # fmt: skip


def test_the_demo_runs_all_eight_steps_and_never_touches_the_real_store(tmp_path):
    directory = tmp_path / "list"
    directory.mkdir()
    body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in FTM).encode("utf-8")
    entities = directory / "entities.ftm.json"
    entities.write_bytes(body)
    (directory / "VERSION").write_text(f"version=20260101000000-test\nsha1={hashlib.sha1(body).hexdigest()}\n")  # noqa: S324
    real_store = tmp_path / "real-audit"  # pointed to by the environment; the demo must neither use nor create it
    env = {"PATH": "/usr/bin:/bin", "SCREENING_AUDIT_DIR": str(real_store)}

    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo.py"), "--input", str(entities)],
        capture_output=True, text=True, env=env, timeout=120,
    )  # fmt: skip

    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert [int(n) for n in re.findall(r"^=== Step (\d+):", out, re.M)] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert "FAILED" not in out and "All 8 steps succeeded" in out
    for line in (
        "Decision: ALERT",
        "transformation applied: token reorder",
        "Score: exact name",
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
