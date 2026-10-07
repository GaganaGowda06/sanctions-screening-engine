"""Carve the OpenSanctions Pairs dataset into train / dev / test and a new frozen final hold-out.

The old hold-out, made for the earlier hand-tuned matcher, is spent.
This is a fresh one for the trained matcher.

Fractions: train 70%, dev 10%, test 10%, final hold-out 10%.

How a pair gets its split. The file is ordered by source (the start of it is 87% positive and the end
2%), so position plays no part. Each pair belongs to a group, the group is hashed with a fixed seed, and
the hash picks the split:
  - A pair in a small connected component (up to 100 pairs, entities linked by pairs) is grouped with its
    whole component, so those entities appear in one split only.
  - A pair in a larger component, and one huge component holds 35% of the file, is grouped by its
    unordered entity pair. That keeps duplicates (8,280 pairs appear more than once) together but cannot
    keep entities apart: a hub entity can sit in 376 pairs. Entity overlap between splits is measured
    and reported rather than hidden.
  - Final hold-out pairs whose two entities appear in no pair of any other split get the code 4
    ("clean"): the final evaluation can report on those separately.
All 755,540 pairs are assigned, including the relationship and role records that are out of scope for name
matching; pairs.in_scope() tells them apart.

Freezing: the assignment, a digest of the pair ids in file order and a digest of the final hold-out
membership are written with the dataset checksum to splits/pairs/. If the files exist they are never
overwritten: the assignment is recomputed and must match, or the script stops.

Run:
    python src/build_pair_splits.py
Writes splits/pairs/assignment.txt.gz, splits/pairs/manifest.json and reports/pairs_splits.json.
"""

import argparse
import gzip
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from pairs import PAIRS_PATH, ROOT, SPLIT_CODES, SPLIT_DIR, in_scope, iter_pairs
from profile_pairs import UnionFind

SEED = "opensanctions-pairs-split-v1"
FRACTIONS = (("train", 0.70), ("dev", 0.10), ("test", 0.10), ("final", 0.10))
LARGE_COMPONENT_PAIRS = 100
DEFAULT_REPORT = ROOT / "reports" / "pairs_splits.json"


def bucket(group: str) -> float:
    """A number in [0, 1) that depends only on the seed and the group."""
    digest = hashlib.sha256(f"{SEED}|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_for(group: str) -> str:
    position, edge = bucket(group), 0.0
    for name, fraction in FRACTIONS:
        edge += fraction
        if position < edge:
            return name
    return FRACTIONS[-1][0]


def assign(keys: list[tuple[str, str]]) -> str:
    """The split code of every pair, in the order given. Pure function of the pair ids."""
    uf = UnionFind()
    for left, right in keys:
        uf.union(left, right)
    pairs_in_component: Counter = Counter()
    smallest_id: dict[str, str] = {}
    for left, right in keys:
        root = uf.find(left)
        pairs_in_component[root] += 1
        smallest_id[root] = min(smallest_id.get(root, left), left, right)

    codes = []
    for left, right in keys:
        root = uf.find(left)
        if pairs_in_component[root] <= LARGE_COMPONENT_PAIRS:
            group = f"component|{smallest_id[root]}"
        else:
            group = "pair|" + "|".join(sorted((left, right)))
        codes.append(SPLIT_CODES[split_for(group)])

    elsewhere = {entity for (left, right), code in zip(keys, codes, strict=True) if code != SPLIT_CODES["final"]
                 for entity in (left, right)}  # fmt: skip
    clean = SPLIT_CODES["final_clean"]
    return "".join(
        clean if code == SPLIT_CODES["final"] and left not in elsewhere and right not in elsewhere else code
        for (left, right), code in zip(keys, codes, strict=True)
    )


def membership_digest(keys: list[tuple[str, str]], codes: str) -> str:
    final = {SPLIT_CODES["final"], SPLIT_CODES["final_clean"]}
    digest = hashlib.sha256()
    for index, ((left, right), code) in enumerate(zip(keys, codes, strict=True)):
        if code in final:
            digest.update(f"{index}\t{left}\t{right}\n".encode())
    return digest.hexdigest()


def pair_keys_digest(keys: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for left, right in keys:
        digest.update(f"{left}\t{right}\n".encode())
    return digest.hexdigest()


def statistics(keys: list[tuple[str, str]], codes: str, positive: bytearray, scope: bytearray) -> dict:
    """Size, label balance and entity overlap with train, per split."""
    names = {code: name for name, code in SPLIT_CODES.items()}
    train_entities = {e for (a, b), c in zip(keys, codes, strict=True) if c == SPLIT_CODES["train"] for e in (a, b)}
    out = {}
    for code in "01234":
        rows = [i for i, c in enumerate(codes) if c == code]
        in_scope_rows = [i for i in rows if scope[i]]
        shares_entity = [i for i in rows if keys[i][0] in train_entities or keys[i][1] in train_entities]
        out[names[code]] = {
            "pairs": len(rows),
            "positive": sum(positive[i] for i in rows),
            "positive_share": round(sum(positive[i] for i in rows) / len(rows), 4) if rows else None,
            "entity_pairs": len(in_scope_rows),
            "entity_pairs_positive_share": round(sum(positive[i] for i in in_scope_rows) / len(in_scope_rows), 4)
            if in_scope_rows
            else None,
            "pairs_sharing_an_entity_with_a_train_pair": len(shares_entity),
            "share_sharing_an_entity_with_a_train_pair": round(len(shares_entity) / len(rows), 4) if rows else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PAIRS_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(f"Python {platform.python_version()}", file=sys.stderr)

    keys: list[tuple[str, str]] = []
    positive, scope = bytearray(), bytearray()
    schemas_by_pair = []
    for _, record in iter_pairs(args.input):
        keys.append((record["left"]["id"], record["right"]["id"]))
        positive.append(record["judgement"] == "positive")
        scope.append(in_scope(record))
        schemas_by_pair.append(record["left"].get("schema"))
    codes = assign(keys)

    manifest_path, assignment_path = args.split_dir / "manifest.json", args.split_dir / "assignment.txt.gz"
    source = json.loads((args.input.parent / "SOURCE.json").read_text())
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if hashlib.sha256(codes.encode()).hexdigest() != recorded["assignment_sha256"]:
            sys.exit("The recomputed assignment differs from the frozen one. The frozen files are not overwritten.")
        manifest = recorded
        print("Frozen assignment reproduced exactly.", file=sys.stderr)
    else:
        args.split_dir.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(assignment_path, "wb", mtime=0) as fh:  # mtime 0 keeps the file bytes reproducible
            fh.write((codes + "\n").encode())
        manifest = {
            "dataset_file": source["file"],
            "dataset_sha256": source["sha256"],
            "seed": SEED,
            "fractions": dict(FRACTIONS),
            "large_component_pairs": LARGE_COMPONENT_PAIRS,
            "codes": SPLIT_CODES,
            "pairs": len(keys),
            "assignment_sha256": hashlib.sha256(codes.encode()).hexdigest(),
            "pair_keys_sha256": pair_keys_digest(keys),
            "final_holdout_membership_sha256": membership_digest(keys, codes),
            "rule": "see src/build_pair_splits.py",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report = {
        "manifest": manifest,
        "splits": statistics(keys, codes, positive, scope),
        "schemas_left_side_by_split": {
            name: dict(Counter(s for s, c in zip(schemas_by_pair, codes, strict=True) if c == code).most_common(6))
            for name, code in (("train", "0"), ("final", "3"))
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report} and the frozen split in {args.split_dir}")


if __name__ == "__main__":
    main()
