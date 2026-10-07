"""Build the feature table of the Pairs data: one row per cleaned entity pair, train / dev / test.

Features are described in pair_features.py. Nothing is trained here. The final hold-out is not read:
only the train, dev and test splits are featurised, from the cleaned pairs (entity pairs, no conflicting
labels, no duplicates).

Rarity weights (inverse document frequency of name tokens) come from the entities in the train pairs only,
so no dev or test name influences a feature. The list the engine screens has its own token statistics; a
model trained here would meet different weights there, which is worth remembering when it is deployed.

Each dev and test row records whether either of its entities also appears in a train pair
(shares_entity_with_train), because every metric is reported separately for the two kinds.

Tables are written to data/pairs/features/{train,dev,test}.csv.gz (derived data, not committed). The report
reports/pairs_features.json holds counts, checksums and how well each feature separates matches from
non-matches on dev.

Run:
    python src/build_pair_features.py
    python src/build_pair_features.py --limit 2000      # a quick trial on the first 2,000 pairs of each split
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from pair_features import (
    SCORES,
    STATES,
    Forms,
    auc,
    make_idf,
    name_tokens,
    pair_features,
    script_class,
    state_rates,
)
from pairs import PAIRS_PATH, ROOT, SPLIT_DIR, iter_split
from profile_data import PROPS_PATH

OUT_DIR = ROOT / "data" / "pairs" / "features"
REPORT = ROOT / "reports" / "pairs_features.json"
SPLITS = ("train", "dev", "test")
COLUMNS = [
    "index", "split", "label", "shares_entity_with_train", "schema_left", "schema_right", "script_class",
    *SCORES, *STATES,
]  # fmt: skip
CHUNK = 400

_state: dict = {}


def _init(document_frequency: dict[str, int], documents: int, identifier_props: dict[str, list[str]]) -> None:
    _state["idf"] = make_idf(document_frequency, documents)
    _state["props"] = identifier_props


def _forms(entity: dict) -> Forms:
    # Not cached by entity id: the same id carries different properties in different pairs (74% of the ids
    # that appear in more than one pair), so a cache would make the features depend on which pair came first.
    return Forms(entity, _state["props"])


def featurise(chunk: list[tuple[int, dict]]) -> list[list]:
    """Rows without the split and the shares-entity flag, which the parent adds."""
    rows = []
    for index, record in chunk:
        a, b = _forms(record["left"]), _forms(record["right"])
        f = pair_features(a, b, _state["idf"])
        rows.append(
            [index, int(record["judgement"] == "positive"), record["left"].get("schema"),
             record["right"].get("schema"), script_class(a, b), record["left"]["id"], record["right"]["id"],
             *(round(f[s], 4) for s in SCORES), *(f[s] for s in STATES)]
        )  # fmt: skip
    return rows


def chunks(items: Iterator[tuple[int, dict]], size: int) -> Iterator[list[tuple[int, dict]]]:
    while batch := list(islice(items, size)):
        yield batch


def write_table(path: Path, rows: Iterator[list]) -> str:
    """Write rows to a gzip CSV whose bytes do not depend on the clock. Returns its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # No file name and no time in the gzip header, so identical rows give identical bytes.
    with open(path, "wb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
            writer = csv.writer(text)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    var_x, var_y = sum((a - mean_x) ** 2 for a in x), sum((b - mean_y) ** 2 for b in y)
    return cov / (var_x * var_y) ** 0.5 if var_x and var_y else 0.0


def separation(rows: list[dict], rates: dict[str, dict[str, float]]) -> dict:
    """How well each feature tells matches from non-matches: AUC, and the mean value for each label."""
    labels = [int(r["label"]) for r in rows]
    out = {"pairs": len(rows), "positive": sum(labels), "negative": len(rows) - sum(labels)}
    for name in SCORES:
        values = [float(r[name]) for r in rows]
        pos = [v for v, label in zip(values, labels, strict=True) if label]
        neg = [v for v, label in zip(values, labels, strict=True) if not label]
        out[name] = {
            "auc": None if auc(values, labels) is None else round(auc(values, labels), 4),
            "mean_match": round(sum(pos) / len(pos), 4) if pos else None,
            "mean_non_match": round(sum(neg) / len(neg), 4) if neg else None,
        }
    for name in STATES:
        states = [r[name] for r in rows]
        score = [rates[name].get(s, 0.5) for s in states]
        by_state = {}
        for state in sorted(set(states)):
            chosen = [label for s, label in zip(states, labels, strict=True) if s == state]
            by_state[state] = {"pairs": len(chosen), "match_share": round(sum(chosen) / len(chosen), 4)}
        out[name] = {
            "auc_of_train_match_rate": None if auc(score, labels) is None else round(auc(score, labels), 4),
            "by_state": by_state,
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PAIRS_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--limit", type=int, help="only the first N cleaned pairs of each split (a trial run)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}, {args.workers} workers", file=sys.stderr)

    ftm = json.loads(PROPS_PATH.read_text())["schemas"]
    identifier_props = {s: list(v["identifier"]) for s, v in ftm.items()}

    def pairs_of(split: str) -> Iterator[tuple[int, dict]]:
        stream = iter_split(split, clean=True, path=args.input, split_dir=args.split_dir)
        return islice(stream, args.limit) if args.limit else stream

    # Pass 1: token document frequencies and the set of entity ids, from train pairs only. An id counts once,
    # with the properties of the first pair it appears in.
    start = time.perf_counter()
    document_frequency: Counter = Counter()
    train_entities: set[str] = set()
    for _, record in pairs_of("train"):
        for side in (record["left"], record["right"]):
            if side["id"] not in train_entities:
                train_entities.add(side["id"])
                document_frequency.update(name_tokens(side))
    print(f"idf from {len(train_entities):,} train entities in {time.perf_counter() - start:.0f}s", file=sys.stderr)

    report: dict = {
        "python_version": platform.python_version(),
        "limit": args.limit,
        "features": {"scores": list(SCORES), "states": list(STATES)},
        "idf_entities": len(train_entities),
        "tables": {},
    }
    rates: dict[str, dict[str, float]] = {}
    dev_rows: list[dict] = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        args.workers, _init, (dict(document_frequency), len(train_entities), identifier_props)
    ) as pool:
        for split in SPLITS:
            start = time.perf_counter()
            counts = {"pairs": 0, "positive": 0, "shares_entity": 0}
            states = {name: ([], []) for name in STATES}  # (states, labels) kept for train, to get match rates

            def rows(split: str = split, counts: dict = counts, states: dict = states) -> Iterator[list]:
                for batch in pool.imap(featurise, chunks(pairs_of(split), CHUNK)):
                    for row in batch:
                        shares = int(split != "train" and (row[5] in train_entities or row[6] in train_entities))
                        counts["pairs"] += 1
                        counts["positive"] += row[1]
                        counts["shares_entity"] += shares
                        if split == "train":
                            for k, name in enumerate(STATES):
                                states[name][0].append(row[-3 + k])
                                states[name][1].append(row[1])
                        yield [row[0], split, row[1], shares if split != "train" else "", *row[2:5], *row[7:]]

            path = args.out / f"{split}.csv.gz"
            digest = write_table(path, rows())
            if split == "train":
                rates = {name: state_rates(*states[name]) for name in STATES}
            report["tables"][split] = {
                **counts,
                "positive_share": round(counts["positive"] / counts["pairs"], 4) if counts["pairs"] else None,
                "sha256": digest,
                "seconds": round(time.perf_counter() - start),
            }
            print(f"{split}: {counts['pairs']:,} pairs in {time.perf_counter() - start:.0f}s", file=sys.stderr)
            if split == "dev":
                dev_rows = read_table(path)

    report["train_match_rate_by_state"] = {k: {s: round(r, 4) for s, r in v.items()} for k, v in rates.items()}
    report["separation_on_dev"] = {
        "all": separation(dev_rows, rates),
        "shares_entity_with_train": separation([r for r in dev_rows if r["shares_entity_with_train"] == "1"], rates),
        "no_entity_in_train": separation([r for r in dev_rows if r["shares_entity_with_train"] == "0"], rates),
    }
    columns = {name: [float(r[name]) for r in dev_rows] for name in SCORES}
    report["score_correlations_on_dev"] = {
        a: {b: round(pearson(columns[a], columns[b]), 3) for b in SCORES} for a in SCORES
    }
    classes = sorted({r["script_class"] for r in dev_rows})
    report["separation_on_dev_by_script_class"] = {
        label: {
            c: separation([r for r in dev_rows if r["script_class"] == c and keep(r)], rates)
            for c in classes
        }
        for label, keep in (
            ("all", lambda r: True),
            ("shares_entity_with_train", lambda r: r["shares_entity_with_train"] == "1"),
            ("no_entity_in_train", lambda r: r["shares_entity_with_train"] == "0"),
        )
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
