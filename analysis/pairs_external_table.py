"""Re-run the pairs scoring once and keep the per-pair table (the first run saved only aggregates).

Same engine, same tasks, same worker as pairs_external_score.py; nothing is tuned and the report there is not touched.
The table holds, per scored pair: label, kind (entity-type pair), shared-name flag, forward score, reverse score.
No names, IDs or other pair content. It goes to data/external/ (git-ignored; the data is CC BY-NC 4.0).

Run:
    python analysis/pairs_external_table.py
"""

import multiprocessing
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import pairs_external_score as ps  # noqa: E402

OUT = common.ROOT / "data" / "external" / "pairs_scores_table.npz"


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT.name} already exists; the table is saved once")
    started = time.time()
    ps.BUILT, _ = ps.build_engine()
    rows, stats = [], Counter()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(max(multiprocessing.cpu_count() - 2, 1), initializer=ps.init_worker) as pool:

        def stream() -> Iterator[tuple]:
            for t in ps.tasks():
                if t[0] == "stats":
                    stats.update(t[1])
                else:
                    yield t

        for i, r in enumerate(pool.imap(ps.work, stream(), chunksize=400)):
            rows.append(r)
            if i % 50000 == 0:
                print(f"{i} pairs scored, {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    # work() returns (label, kind, script, cross, bd, ct, forward, reverse, capped, shared, schemas)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        label=np.array([r[0] for r in rows], dtype="i1"),
        kind=np.array([r[1] for r in rows], dtype="U30"),
        shared=np.array([r[9] for r in rows], dtype=bool),
        forward=np.array([r[6] for r in rows], dtype="f8"),
        reverse=np.array([r[7] for r in rows], dtype="f8"),
    )
    print(f"wrote {OUT}, {len(rows)} pairs, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
