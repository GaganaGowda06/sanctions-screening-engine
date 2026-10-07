"""Tests for the pair loader (src/pairs.py), the split rule (src/build_pair_splits.py) and the frozen split files.

Run:
    pytest tests/test_pair_splits.py
"""

import gzip
import hashlib
import json
import random
from collections import Counter

import pytest

import build_pair_splits as bps
import pairs

TRAIN, DEV, TEST, FINAL, CLEAN = "0", "1", "2", "3", "4"


def independent_pairs(n):
    return [(f"a{i}", f"b{i}") for i in range(n)]


def test_assignment_is_deterministic_and_does_not_depend_on_order():
    keys = independent_pairs(500)
    codes = bps.assign(keys)
    assert codes == bps.assign(keys)
    shuffled = keys[:]
    random.Random(1).shuffle(shuffled)
    by_pair = dict(zip(keys, codes, strict=True))
    assert bps.assign(shuffled) == "".join(by_pair[k] for k in shuffled)


def test_fractions_are_close_to_the_targets():
    codes = bps.assign(independent_pairs(20_000))
    counts = Counter("3" if c == CLEAN else c for c in codes)
    for code, (_, fraction) in zip((TRAIN, DEV, TEST, FINAL), bps.FRACTIONS, strict=True):
        assert counts[code] / 20_000 == pytest.approx(fraction, abs=0.012)


def test_duplicates_in_either_order_land_together():
    keys = independent_pairs(200)
    flipped = [(b, a) for a, b in keys]
    codes = bps.assign(keys + flipped + keys)
    first, second, third = codes[:200], codes[200:400], codes[400:]
    assert first == second == third


def test_a_small_component_is_never_split_across_sets():
    # 300 chains a-b-c-d (three pairs each, one component each).
    chains = [(f"n{i}a", f"n{i}b") for i in range(300)] + [(f"n{i}b", f"n{i}c") for i in range(300)]
    chains += [(f"n{i}c", f"n{i}d") for i in range(300)]
    codes = bps.assign(chains)
    for i in range(300):
        members = {codes[i], codes[300 + i], codes[600 + i]}
        assert len(members) == 1, f"chain {i} was split: {members}"
    assert len({codes[i] for i in range(300)}) > 1  # the chains themselves do spread over several sets


def test_pairs_of_a_large_component_are_assigned_one_by_one_and_leave_overlap():
    star = [("hub", f"leaf{i}") for i in range(400)]  # one component of 400 pairs, over the limit
    codes = bps.assign(star)
    assert len(set(codes)) > 1
    assert CLEAN not in codes  # every final pair shares the hub with pairs elsewhere
    assert bps.assign(star) == codes


def test_final_pairs_sharing_no_entity_with_other_sets_are_marked_clean():
    keys = independent_pairs(400)
    codes = bps.assign(keys)
    assert FINAL not in codes and CLEAN in codes  # independent pairs never share an entity
    # Add pairs linking the final pairs to train pairs: they stop being clean.
    train = [k for k, c in zip(keys, codes, strict=True) if c == TRAIN]
    final = [k for k, c in zip(keys, codes, strict=True) if c == CLEAN]
    bridge = [(final[0][0], train[0][0])]
    new_codes = bps.assign(keys + bridge)
    assert new_codes[keys.index(final[0])] != CLEAN


def test_the_old_split_rule_constants():
    assert sum(f for _, f in bps.FRACTIONS) == pytest.approx(1.0)
    assert bps.bucket("x") == bps.bucket("x") and 0 <= bps.bucket("x") < 1


def make_pairs_file(path, n=60):
    rows = []
    for i in range(n):
        schema = "Person" if i % 3 else "Occupancy"
        left = {"id": f"l{i}", "schema": schema, "properties": {}}
        right = {"id": f"r{i}", "schema": schema, "properties": {}}
        rows.append({"judgement": "positive" if i % 2 else "negative", "left": left, "right": right})
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def freeze(tmp_path, rows, path):
    keys = [(r["left"]["id"], r["right"]["id"]) for r in rows]
    codes = bps.assign(keys)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    with gzip.GzipFile(split_dir / "assignment.txt.gz", "wb", mtime=0) as fh:
        fh.write((codes + "\n").encode())
    manifest = {
        "assignment_sha256": hashlib.sha256(codes.encode()).hexdigest(),
        "pair_keys_sha256": bps.pair_keys_digest(keys),
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest))
    return split_dir, codes


def test_iter_split_returns_only_the_requested_pairs(tmp_path):
    path = tmp_path / "pairs.json.gz"
    rows = make_pairs_file(path)
    split_dir, codes = freeze(tmp_path, rows, path)
    train = list(pairs.iter_split("train", path=path, split_dir=split_dir))
    assert [i for i, _ in train] == [i for i, c in enumerate(codes) if c == TRAIN]
    assert all(rec["left"]["id"] == f"l{i}" for i, rec in train)
    scoped = list(pairs.iter_split("train", path=path, split_dir=split_dir, in_scope_only=True))
    assert all(pairs.in_scope(rec) for _, rec in scoped) and len(scoped) < len(train)


def test_the_final_holdout_is_refused_unless_explicitly_allowed(tmp_path):
    path = tmp_path / "pairs.json.gz"
    split_dir, _ = freeze(tmp_path, make_pairs_file(path), path)
    for name in ("final", "final_clean"):
        with pytest.raises(PermissionError, match="final hold-out"):
            next(pairs.iter_split(name, path=path, split_dir=split_dir))
    assert list(pairs.iter_split("final", allow_final=True, path=path, split_dir=split_dir)) is not None


def test_an_assignment_is_rejected_for_a_different_file(tmp_path):
    path = tmp_path / "pairs.json.gz"
    rows = make_pairs_file(path)
    split_dir, _ = freeze(tmp_path, rows, path)
    other = tmp_path / "other.json.gz"
    rows[5]["left"]["id"] = "someone_else"
    with gzip.open(other, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="not the one"):
        list(pairs.iter_split("train", path=other, split_dir=split_dir))


def test_a_tampered_assignment_fails_its_checksum(tmp_path):
    path = tmp_path / "pairs.json.gz"
    split_dir, codes = freeze(tmp_path, make_pairs_file(path), path)
    with gzip.GzipFile(split_dir / "assignment.txt.gz", "wb", mtime=0) as fh:
        fh.write((("1" if codes[0] != "1" else "2") + codes[1:] + "\n").encode())
    with pytest.raises(ValueError, match="checksum"):
        pairs.load_assignment(split_dir)


def test_scope_flags_relationship_records_out():
    entity = {"left": {"schema": "Person"}, "right": {"schema": "Company"}}
    link = {"left": {"schema": "Occupancy"}, "right": {"schema": "Occupancy"}}
    mixed = {"left": {"schema": "Person"}, "right": {"schema": "Position"}}
    assert pairs.in_scope(entity) and not pairs.in_scope(link) and not pairs.in_scope(mixed)


def test_only_the_final_evaluation_may_ask_for_the_final_split():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    askers = {p.name for p in src.glob("*.py") if "allow_final=True" in p.read_text()}
    assert askers == {"evaluate_pairs_final.py"}


def test_the_committed_split_files_agree_with_each_other():
    split_dir = pairs.SPLIT_DIR
    if not (split_dir / "manifest.json").exists():
        pytest.skip("splits not built yet")
    codes, manifest = pairs.load_assignment(split_dir)  # raises if the checksum is wrong
    assert len(codes) == manifest["pairs"]
    assert set(codes) <= set("01234")
    counts = Counter("3" if c == "4" else c for c in codes)
    for code, (_, fraction) in zip("0123", bps.FRACTIONS, strict=True):
        assert counts[code] / len(codes) == pytest.approx(fraction, abs=0.01)
    assert manifest["seed"] == bps.SEED and manifest["large_component_pairs"] == bps.LARGE_COMPONENT_PAIRS
