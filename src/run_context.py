"""Run context: everything that decides what the matcher answers, as one immutable, hashable object.

An audit record has to say not only what the matcher decided but under what. Replay (a later step) can only be
exact if the code, configuration, list and libraries are the same, so each record carries the hash of the run
context it was made under, and replay refuses to run under a different one.

What goes in, and why it is there (found by reading the score path, see the project notes running notes):
  code            git SHA and a dirty-tree flag, plus the SHA-256 of each source file on the score path. The
                  normalisation and transliteration tables are Python source, so their hash is the source hash.
  config          the ScreeningConfig values, and the SHA-256 of src/screening_config.json when it exists.
  engine          score-affecting settings that ScreeningConfig does not hold: max_df, the skeleton flags, the
                  birth date and country adjustment magnitudes, the length penalty, the 10,000 candidate cap.
  list            the OpenSanctions dataset version, and the SHA-256 of the entities file that was loaded.
  rarity table    computed from the list when the engine is built, not read from a file, so it is hashed from the
                  built engine (token document frequencies and the skeleton document frequencies).
  libraries       installed version of every third-party library on the score path (rapidfuzz).
  interpreter     Python version and implementation, and the Unicode database version (normalisation uses it).
  pins file       SHA-256 of requirements.txt. It pins versions but is not a lockfile (no hashes, no transitive
                  packages), so lockfile_sha256 stays None until a real lockfile exists.

Deterministic: no clock, no randomness, no paths. The same inputs give the same context hash. Standard library
only, apart from reading the version of the libraries it names.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from scoring import Screener
from screening import CANDIDATE_CAP, CONFIG_PATH, ScreeningEngine

SCHEMA_VERSION = 1
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

# Source files imported on the score path, checked against the interpreter in tests/test_run_context.py.
SCORE_PATH_SOURCES = ("candidates.py", "entities.py", "normalise.py", "profile_data.py", "scoring.py", "screening.py")
# Third-party libraries imported on the score path. Checked in the same test.
SCORE_PATH_LIBRARIES = ("rapidfuzz",)
# ScreeningEngine.screen() asks the screener for this many candidates per query.
SCREEN_CANDIDATE_CAP = CANDIDATE_CAP


def canonical_json(value: object) -> str:
    """Sorted keys, fixed separators, ASCII only: the same value always gives the same text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pairs(mapping: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    """A mapping as a sorted tuple of pairs, so the context holds no mutable dict."""
    return tuple(sorted(mapping.items()))


@dataclass(frozen=True)
class RunContext:
    schema_version: int
    git_sha: str  # "unknown" if the code is not in a git repository
    git_dirty: bool  # True if there are uncommitted or untracked changes, or the state could not be read
    config: str  # ScreeningConfig values as canonical JSON (a string, so the context holds no mutable dict)
    config_file_sha256: str | None  # src/screening_config.json, None if the file does not exist
    engine_settings: str  # score-affecting settings that ScreeningConfig does not hold, as canonical JSON
    snapshot_id: str  # OpenSanctions dataset version
    snapshot_sha256: str  # SHA-256 of the entities file that was loaded
    rarity_table_sha256: str  # document frequencies the engine computed from that list
    artefacts: tuple[tuple[str, str], ...]  # source file name -> SHA-256
    libraries: tuple[tuple[str, str], ...]  # library -> installed version
    python_version: str
    python_implementation: str
    unidata_version: str
    pins_file_sha256: str | None  # requirements.txt: pins versions, but is not a lockfile
    lockfile_sha256: str | None  # None: the project has no lockfile

    def as_dict(self) -> dict:
        """JSON-ready form: config and settings become objects, artefacts and libraries become mappings.
        Round-trips through from_dict()."""
        out = asdict(self)
        for name in ("config", "engine_settings"):
            out[name] = json.loads(out[name])
        for name in ("artefacts", "libraries"):
            out[name] = {k: v for k, v in out[name]}
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RunContext":
        values = dict(data)
        for name in ("config", "engine_settings"):
            values[name] = canonical_json(values[name])
        for name in ("artefacts", "libraries"):
            values[name] = pairs(values[name])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    def canonical(self) -> str:
        """The text that is hashed: every field as stored, so the hash does not depend on parsing."""
        return canonical_json(asdict(self))

    @property
    def context_hash(self) -> str:
        return sha256_text(self.canonical())


def git_state(root: Path = ROOT) -> tuple[str, bool]:
    """(HEAD SHA, dirty). Dirty if anything is modified, staged or untracked and not ignored. Unreadable counts as
    dirty with SHA "unknown", so a context never claims a clean tree it could not check."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", True
    return sha, bool(status.strip())


def rarity_table_sha256(screener: Screener) -> str:
    """Hash of the token document frequencies the screener computed (or was given) and, when the skeleton is
    rarity-weighted, the skeleton document frequencies. Reads the screener's own tables, not a file."""
    payload = {
        "documents": screener._n,
        "token_df": sorted(screener._df.items()),
        "skeleton_df": sorted(screener._skeleton_df.items()) if screener.skeleton_idf else None,
    }
    return sha256_text(canonical_json(payload))


def engine_settings(screener: Screener) -> dict[str, object]:
    return {
        "max_df": screener.index.max_df,
        "skeleton_idf": screener.skeleton_idf,
        "skeleton_particles": screener.skeleton_particles,
        "common_df": screener.common_df,
        "orthographic": screener.orthographic,
        "length_penalty": screener.length_penalty,
        "adjustments": asdict(screener.adjustments),
        "candidate_cap": SCREEN_CANDIDATE_CAP,
    }


def snapshot_identity(entities_path: Path) -> tuple[str, str]:
    """(dataset version, SHA-256 of the entities file), checked against the VERSION file next to it.

    download_data.py writes VERSION with the dataset version and the SHA-1 of the file it verified. If the SHA-1 of
    the file on disk no longer matches, the list is not the one that was downloaded and no context is built.
    """
    version_file = entities_path.with_name("VERSION")
    if not version_file.exists():
        raise FileNotFoundError(f"{version_file} is missing: the list snapshot cannot be identified")
    recorded = dict(line.split("=", 1) for line in version_file.read_text().splitlines() if "=" in line)
    version, sha1 = recorded.get("version"), recorded.get("sha1")
    if not version or not sha1:
        raise ValueError(f"{version_file} has no version or sha1")
    sha256, check = hashlib.sha256(), hashlib.sha1()  # noqa: S324  (the SHA-1 is the downloader's own record)
    with open(entities_path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            sha256.update(chunk)
            check.update(chunk)
    if check.hexdigest() != sha1:
        raise ValueError(f"{entities_path} does not match the SHA-1 recorded in {version_file}")
    return version, sha256.hexdigest()


def library_versions(names: tuple[str, ...] = SCORE_PATH_LIBRARIES) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def build_run_context(
    engine: ScreeningEngine,
    entities_path: Path,
    *,
    src_dir: Path = SRC_DIR,
    config_path: Path = CONFIG_PATH,
    pins_path: Path | None = ROOT / "requirements.txt",
    lockfile_path: Path | None = None,
    git: Callable[[], tuple[str, bool]] = git_state,
    versions: Callable[[], dict[str, str]] = library_versions,
) -> RunContext:
    """The run context of an engine that was built from `entities_path`. Raises if the list cannot be identified."""
    git_sha, dirty = git()
    snapshot_id, snapshot_sha = snapshot_identity(entities_path)
    return RunContext(
        schema_version=SCHEMA_VERSION,
        git_sha=git_sha,
        git_dirty=dirty,
        config=canonical_json(asdict(engine.config)),
        config_file_sha256=sha256_file(config_path) if config_path.exists() else None,
        engine_settings=canonical_json(engine_settings(engine.screener)),
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha,
        rarity_table_sha256=rarity_table_sha256(engine.screener),
        artefacts=pairs({name: sha256_file(src_dir / name) for name in SCORE_PATH_SOURCES}),
        libraries=pairs(versions()),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        unidata_version=unicodedata.unidata_version,
        pins_file_sha256=sha256_file(pins_path) if pins_path is not None and pins_path.exists() else None,
        lockfile_sha256=sha256_file(lockfile_path) if lockfile_path is not None and lockfile_path.exists() else None,
    )
