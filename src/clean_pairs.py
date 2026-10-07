"""Clean the OpenSanctions Pairs dataset: entity pairs only, no conflicting labels, no duplicates.

Decisions applied:
  1. Keep only pairs of name-bearing entities (pairs.ENTITY_SCHEMAS). Relationship and role records
     (Occupancy, Succession, Position, Ownership, ...) are dropped.
  2. Drop every pair whose two entities were labelled both positive and negative on different lines
     (conflicting labels), all of its lines. Deduplicate the rest: of the lines that name the same
     unordered pair of entities, the first in file order is kept.

The split is NOT changed. Duplicates were kept in one split by the split rule, so deduplicating cannot
move a pair across splits. The result is a mask over the frozen split: one character per pair in file
order, "1" kept and "0" dropped, written to splits/pairs/clean_mask.txt.gz with counts and checksums in
splits/pairs/cleaning.json. If those files exist they are never overwritten: the mask is recomputed and
must match, or the script stops.

Run:
    python src/clean_pairs.py
"""

import argparse
import gzip
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from pairs import PAIRS_PATH, SPLIT_CODES, SPLIT_DIR, in_scope, iter_pairs, load_assignment


def unordered_key(record: dict) -> str:
    return "|".join(sorted((record["left"]["id"], record["right"]["id"])))


def clean_mask(path: Path) -> tuple[str, dict]:
    """The keep mask over every line of the file, and what was dropped and why."""
    first_line: dict[str, int] = {}
    labels: dict[str, set[str]] = {}
    lines_per_key: Counter = Counter()
    scope = bytearray()
    keys: list[str | None] = []
    for index, record in iter_pairs(path):
        inside = in_scope(record)
        scope.append(inside)
        if not inside:
            keys.append(None)
            continue
        key = unordered_key(record)
        keys.append(key)
        first_line.setdefault(key, index)
        labels.setdefault(key, set()).add(record["judgement"])
        lines_per_key[key] += 1
    conflicting = {k for k, v in labels.items() if len(v) > 1}
    mask = []
    dropped = Counter()
    for index, key in enumerate(keys):
        if key is None:
            mask.append("0")
            dropped["not_an_entity_pair"] += 1
        elif key in conflicting:
            mask.append("0")
            dropped["conflicting_label_lines"] += 1
        elif first_line[key] != index:
            mask.append("0")
            dropped["duplicate_lines"] += 1
        else:
            mask.append("1")
    stats = {
        "lines": len(keys),
        "entity_pair_lines": sum(scope),
        "dropped": dict(dropped),
        "conflicting_unordered_pairs": len(conflicting),
        "unordered_pairs_that_appear_more_than_once_before": sum(1 for n in lines_per_key.values() if n > 1),
        "kept": mask.count("1"),
    }
    return "".join(mask), stats


def per_split(codes: str, mask: str, positives: bytearray) -> dict:
    names = {code: name for name, code in SPLIT_CODES.items()}
    out = {}
    for code in "01234":
        kept = [i for i, (c, m) in enumerate(zip(codes, mask, strict=True)) if c == code and m == "1"]
        out[names[code]] = {
            "pairs": len(kept),
            "positive": sum(positives[i] for i in kept),
            "positive_share": round(sum(positives[i] for i in kept) / len(kept), 4) if kept else None,
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PAIRS_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    codes, manifest = load_assignment(args.split_dir)
    mask, stats = clean_mask(args.input)
    positives = bytearray(record["judgement"] == "positive" for _, record in iter_pairs(args.input))
    final = {SPLIT_CODES["final"], SPLIT_CODES["final_clean"]}
    final_digest = hashlib.sha256()
    for index, (code, keep) in enumerate(zip(codes, mask, strict=True)):
        if code in final and keep == "1":
            final_digest.update(f"{index}\n".encode())
    result = {
        "dataset_sha256": manifest["dataset_sha256"],
        "assignment_sha256": manifest["assignment_sha256"],
        **stats,
        "mask_sha256": hashlib.sha256(mask.encode()).hexdigest(),
        "final_holdout_kept_line_numbers_sha256": final_digest.hexdigest(),
        "by_split_after_cleaning": per_split(codes, mask, positives),
    }
    info_path, mask_path = args.split_dir / "cleaning.json", args.split_dir / "clean_mask.txt.gz"
    if info_path.exists():
        recorded = json.loads(info_path.read_text())
        if recorded["mask_sha256"] != result["mask_sha256"]:
            sys.exit("The recomputed mask differs from the frozen one. The frozen files are not overwritten.")
        print("Frozen cleaning mask reproduced exactly.", file=sys.stderr)
    else:
        with gzip.GzipFile(mask_path, "wb", mtime=0) as fh:
            fh.write((mask + "\n").encode())
        info_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
