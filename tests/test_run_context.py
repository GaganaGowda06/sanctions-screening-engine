"""Tests for src/run_context.py.

The context is built from a small engine, a tiny list file and copies of the score-path sources in a temporary
directory, so a test can change one input at a time and see the hash move. One test builds the real context from
the downloaded list and is skipped when the list is not on disk.

Run:
    pytest tests/test_run_context.py
"""

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import run_context as rc
from scoring import Adjustments, Screener
from screening import ScreeningConfig, ScreeningEngine

ENTITIES = [
    {"id": "e1", "names": ["Mohammed Ali Hassan"], "birth_dates": ["1970-01-01"], "countries": ["iq"]},
    {"id": "e2", "names": ["Ivan Petrov"], "birth_dates": [], "countries": ["ru"]},
    {"id": "e3", "names": ["Acme Trading Limited"], "birth_dates": [], "countries": []},
]
FIXED_GIT = lambda: ("a" * 40, False)  # noqa: E731
FIXED_VERSIONS = lambda: {"rapidfuzz": "3.14.6"}  # noqa: E731


def write_list(
    directory: Path, body: bytes = b'{"id":"e1"}\n{"id":"e2"}\n', version: str = "20260101000000-abc"
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "entities.ftm.json"
    path.write_bytes(body)
    sha1 = hashlib.sha1(body).hexdigest()  # noqa: S324
    (directory / "VERSION").write_text(f"version={version}\nsha1={sha1}\nsize_bytes={len(body)}\n")
    return path


@pytest.fixture
def world(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in rc.SCORE_PATH_SOURCES:
        shutil.copy(rc.SRC_DIR / name, src / name)
    config_path = tmp_path / "screening_config.json"
    config_path.write_text(json.dumps({"global_threshold": 0.86}))
    pins = tmp_path / "requirements.txt"
    pins.write_text("rapidfuzz==3.14.6\n")
    entities = write_list(tmp_path / "list")
    return {"src": src, "config_path": config_path, "pins": pins, "entities": entities}


def make(world, engine=None, **overrides):
    engine = engine or ScreeningEngine.from_entities(ENTITIES, ScreeningConfig())
    kwargs = {
        "src_dir": world["src"], "config_path": world["config_path"], "pins_path": world["pins"],
        "git": FIXED_GIT, "versions": FIXED_VERSIONS,
    }  # fmt: skip
    kwargs.update(overrides)
    return rc.build_run_context(engine, world.get("entities_override", world["entities"]), **kwargs)


# ---------- determinism ----------


def test_same_inputs_give_the_same_context_and_hash(world):
    a, b = make(world), make(world)
    assert a == b
    assert a.context_hash == b.context_hash
    assert len(a.context_hash) == 64


def test_two_engines_built_from_the_same_list_give_the_same_hash(world):
    first = make(world, ScreeningEngine.from_entities(ENTITIES, ScreeningConfig()))
    second = make(world, ScreeningEngine.from_entities(list(ENTITIES), ScreeningConfig()))
    assert first.context_hash == second.context_hash


def test_the_context_is_immutable_and_hashable(world):
    ctx = make(world)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.git_sha = "b" * 40  # type: ignore[misc]
    hash(ctx)  # every field is a string, number, bool, None or a tuple of such


def test_round_trip_through_a_dict_keeps_the_hash(world):
    ctx = make(world)
    as_json = json.loads(json.dumps(ctx.as_dict()))
    assert rc.RunContext.from_dict(as_json) == ctx
    assert rc.RunContext.from_dict(as_json).context_hash == ctx.context_hash


# ---------- every field moves the hash ----------


def mutate(ctx: rc.RunContext, name: str) -> rc.RunContext:
    value = getattr(ctx, name)
    if isinstance(value, bool):
        new = not value
    elif isinstance(value, int):
        new = value + 1
    elif isinstance(value, str):
        new = value + "x"
    elif value is None:
        new = "x"
    else:  # a tuple of pairs
        new = (*value, ("added", "x"))
    return dataclasses.replace(ctx, **{name: new})


@pytest.mark.parametrize("name", [f.name for f in dataclasses.fields(rc.RunContext)])
def test_changing_any_field_changes_the_hash(world, name):
    ctx = make(world)
    assert mutate(ctx, name).context_hash != ctx.context_hash


@pytest.mark.parametrize("name", ["artefacts", "libraries"])
def test_changing_one_entry_of_a_mapping_field_changes_the_hash(world, name):
    ctx = make(world)
    for i in range(len(getattr(ctx, name))):
        entries = list(getattr(ctx, name))
        key, value = entries[i]
        entries[i] = (key, value + "x")
        assert dataclasses.replace(ctx, **{name: tuple(entries)}).context_hash != ctx.context_hash


# ---------- real inputs move the hash ----------


def test_editing_a_score_path_source_file_changes_the_hash(world):
    before = make(world)
    path = world["src"] / "normalise.py"
    path.write_text(path.read_text() + "\n# edited\n")
    after = make(world)
    assert after.context_hash != before.context_hash
    assert dict(after.artefacts)["normalise.py"] != dict(before.artefacts)["normalise.py"]
    assert dict(after.artefacts)["scoring.py"] == dict(before.artefacts)["scoring.py"]


def test_every_source_file_is_hashed(world):
    assert {name for name, _ in make(world).artefacts} == set(rc.SCORE_PATH_SOURCES)


def test_a_different_threshold_changes_the_hash(world):
    base = make(world)
    other = make(world, ScreeningEngine.from_entities(ENTITIES, ScreeningConfig(global_threshold=0.85)))
    assert other.context_hash != base.context_hash


def test_a_different_skeleton_setting_blend_or_scope_changes_the_hash(world):
    base = make(world)
    for config in (ScreeningConfig(skeleton_lift=0.25), ScreeningConfig(blend=0.4), ScreeningConfig(per_script=True)):
        assert make(world, ScreeningEngine.from_entities(ENTITIES, config)).context_hash != base.context_hash


def test_settings_that_are_not_in_the_config_are_in_the_hash(world):
    base = make(world)
    config = ScreeningConfig()
    for kwargs in ({"skeleton_idf": False}, {"max_df": 500}, {"adjustments": Adjustments(country_agree=0.04)}):
        screener = Screener.build(ENTITIES, blend=config.blend, **kwargs)
        assert make(world, ScreeningEngine(screener, config)).context_hash != base.context_hash, kwargs


def test_a_changed_config_file_changes_the_hash(world):
    base = make(world)
    world["config_path"].write_text(json.dumps({"global_threshold": 0.87}))
    assert make(world).config_file_sha256 != base.config_file_sha256
    assert make(world).context_hash != base.context_hash


def test_a_missing_config_file_is_recorded_as_none(world):
    world["config_path"].unlink()
    assert make(world).config_file_sha256 is None


def test_a_different_list_changes_snapshot_and_rarity_hashes(world, tmp_path):
    base = make(world)
    other_list = write_list(tmp_path / "list2", body=b'{"id":"e1"}\n{"id":"e9"}\n', version="20260102000000-def")
    changed = make({**world, "entities_override": other_list})
    assert changed.snapshot_id != base.snapshot_id and changed.snapshot_sha256 != base.snapshot_sha256
    assert changed.context_hash != base.context_hash


def test_the_rarity_table_comes_from_the_built_engine_not_the_file(world):
    base = make(world)
    smaller = make(world, ScreeningEngine.from_entities(ENTITIES[:2], ScreeningConfig()))
    assert smaller.rarity_table_sha256 != base.rarity_table_sha256
    assert smaller.snapshot_sha256 == base.snapshot_sha256


def test_a_library_version_change_changes_the_hash(world):
    base = make(world)
    assert make(world, versions=lambda: {"rapidfuzz": "3.14.7"}).context_hash != base.context_hash


def test_the_pins_file_is_hashed_and_a_lockfile_is_none_when_absent(world):
    base = make(world)
    assert base.lockfile_sha256 is None
    world["pins"].write_text("rapidfuzz==3.14.7\n")
    assert make(world).pins_file_sha256 != base.pins_file_sha256
    lock = world["pins"].with_name("lock.txt")
    lock.write_text("rapidfuzz==3.14.6 --hash=sha256:00\n")
    assert make(world, lockfile_path=lock).lockfile_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()


def test_git_sha_and_dirty_flag_are_in_the_hash(world):
    base = make(world)
    assert make(world, git=lambda: ("b" * 40, False)).context_hash != base.context_hash
    assert make(world, git=lambda: ("a" * 40, True)).context_hash != base.context_hash


# ---------- the list is checked, not trusted ----------


def test_a_list_that_does_not_match_its_recorded_sha1_is_refused(world):
    world["entities"].write_bytes(world["entities"].read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match the SHA-1"):
        make(world)


def test_a_list_without_a_version_file_is_refused(world):
    world["entities"].with_name("VERSION").unlink()
    with pytest.raises(FileNotFoundError):
        make(world)


# ---------- git state ----------


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_state_reads_sha_and_dirty_flag(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("1")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "one")
    sha, dirty = rc.git_state(repo)
    assert len(sha) == 40 and dirty is False
    (repo / "a.txt").write_text("2")
    assert rc.git_state(repo) == (sha, True)
    git(repo, "checkout", "-q", "a.txt")
    (repo / "new.txt").write_text("untracked")
    assert rc.git_state(repo) == (sha, True)


def test_git_state_outside_a_repository_is_unknown_and_dirty(tmp_path):
    assert rc.git_state(tmp_path) == ("unknown", True)


# ---------- the lists of artefacts match what the matcher really imports ----------


def test_the_score_path_lists_match_what_importing_screening_loads():
    code = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import screening\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(json.dumps(sorted(m for m in new if not m.startswith('_') and m not in sys.stdlib_module_names)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=rc.SRC_DIR, capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(rc.SRC_DIR), "PATH": ""},
    )  # fmt: skip
    loaded = set(json.loads(out.stdout))
    local = {name for name in loaded if (rc.SRC_DIR / f"{name}.py").exists()}
    assert local == {name.removesuffix(".py") for name in rc.SCORE_PATH_SOURCES}
    assert loaded - local - {"cython_runtime"} == set(rc.SCORE_PATH_LIBRARIES)


def test_the_candidate_cap_mirrors_the_literal_in_screening():
    source = (rc.SRC_DIR / "screening.py").read_text()
    assert f"keep={rc.SCREEN_CANDIDATE_CAP:_}" in source


# ---------- the real list ----------

REAL_LIST = rc.ROOT / "data" / "sanctions" / "entities.ftm.json"


@pytest.mark.skipif(not REAL_LIST.exists(), reason="the OpenSanctions file is not downloaded")
def test_real_context_is_deterministic_and_names_the_installed_library():
    engine = ScreeningEngine.from_file(REAL_LIST)
    first = rc.build_run_context(engine, REAL_LIST)
    second = rc.build_run_context(engine, REAL_LIST)
    assert first == second
    assert dict(first.libraries)["rapidfuzz"] == "3.14.6"
    assert first.snapshot_id == (REAL_LIST.with_name("VERSION").read_text().splitlines()[0].split("=", 1)[1])
    assert first.config_file_sha256 == hashlib.sha256(rc.CONFIG_PATH.read_bytes()).hexdigest()
