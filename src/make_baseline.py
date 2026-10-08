"""Make the drift baseline: a seeded sample of the frozen hold-out, screened through the audited path.

Run:
    python src/make_baseline.py --audit-dir /tmp/baseline_audit --size 1500 --seed 11

The sample is screened with audited_screen into a dedicated, new audit directory (never the production one), the
metrics for the whole log are computed, and the report is saved as baseline/<context_hash>.json together with the run
context it was measured under and a description of the sample (hold-out file and checksum, seed, size, script mix). The
file holds aggregate numbers only: no query text, no request ID. A test checks every committed baseline against the
names, birth dates and countries of its own sample.

The hold-out is list-derived test data (positives are aliases of listed entities, negatives are recombined listed
tokens), not customer data, and it is spent: it is used here as a fixed regression benchmark, never to tune anything.

A baseline is named by, and applies to, the decision context hash of the engine that made it: the configuration and
threshold, the list snapshot, the scoring and normalisation source files, the rarity table and the library and Python
versions. A new git commit with the same decisions keeps matching; a changed threshold, list or scoring code does not,
and the drift comparison then refuses. After such a change make a new baseline (delete the old file on purpose: a frozen
baseline is never silently replaced).
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from audit import AuditError, AuditLog
from audit_drift import BASELINE_DIR, save_baseline
from audit_metrics import MetricsConfig, compute_metrics
from audited_screening import AuditedEngine
from profile_data import DEFAULT_INPUT
from screening import query_script

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "holdout" / "screening_holdout_v3.json"
DEFAULT_SIZE = 1500
DEFAULT_SEED = 11


def load_queries(path: Path = HOLDOUT) -> list[dict]:
    """The hold-out's queries, each with its script (Stage 3 detection)."""
    queries = json.loads(Path(path).read_text(encoding="utf-8"))["queries"]
    for q in queries:
        q["script"] = query_script(q["name"])
    return queries


def sample_indices(queries: list[dict], size: int, seed: int, exclude: frozenset[int] = frozenset()) -> list[int]:
    """A seeded sample of query positions, never any in `exclude`."""
    pool = [i for i in range(len(queries)) if i not in exclude]
    if size > len(pool):
        raise ValueError(f"asked for {size} queries but only {len(pool)} are available")
    return random.Random(seed).sample(pool, size)


Transform = Callable[[dict], tuple[str, str | None, str | None]]


def screen_queries(
    engine: AuditedEngine, queries: list[dict], indices: list[int], transform: Transform | None = None
) -> int:
    """Screen the chosen queries through audited_screen. `transform(query) -> (name, dob, country)` can degrade them."""
    for i in indices:
        q = queries[i]
        name, dob, country = transform(q) if transform else (q["name"], q["dob"], q["country"])
        engine.audited_screen(name, dob, country)
    return len(indices)


def whole_log_metrics(audit_dir: Path, config: MetricsConfig | None = None) -> dict:
    last = len(AuditLog.in_directory(audit_dir).headers.read_all()) - 1
    report, _ = compute_metrics(audit_dir, 0, last, config)
    return report


def build_baseline(
    list_path: Path,
    audit_dir: Path,
    out_dir: Path = BASELINE_DIR,
    holdout: Path = HOLDOUT,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    config: MetricsConfig | None = None,
) -> tuple[Path, list[int]]:
    """Screen the sample into `audit_dir` (created here, must not exist or be empty) and save the baseline file."""
    queries = load_queries(holdout)
    indices = sample_indices(queries, size, seed)
    AuditLog.create_new(audit_dir)
    engine = AuditedEngine.from_file(list_path, audit_dir)
    screen_queries(engine, queries, indices)
    report = whole_log_metrics(audit_dir, config)
    sample = {
        "holdout": Path(holdout).name,
        "holdout_sha256": hashlib.sha256(Path(holdout).read_bytes()).hexdigest(),
        "seed": seed,
        "size": size,
        "script_mix": dict(sorted(Counter(queries[i]["script"] for i in indices).items())),
    }
    return save_baseline(report, engine.run_context.as_dict(), sample, out_dir), indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit-dir", required=True, type=Path, help="a NEW directory for the baseline's audit store")
    parser.add_argument("--out-dir", type=Path, default=BASELINE_DIR, help="where baseline/<hash>.json goes")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="entities file to screen against")
    args = parser.parse_args(argv)
    try:
        path, indices = build_baseline(args.input, args.audit_dir, args.out_dir, args.holdout, args.size, args.seed)
    except (AuditError, ValueError, FileExistsError) as exc:
        print(f"baseline not made: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path} from {len(indices)} hold-out queries (seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
