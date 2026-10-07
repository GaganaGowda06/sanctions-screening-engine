"""Load the OpenSanctions Pairs labelled dataset: one JSON record per line, two entities and a judgement.

    {"judgement": "positive" | "negative", "left": {entity}, "right": {entity}}

Entities are FollowTheMoney entities (id, caption, schema, datasets, properties, ...). The file is gzip
compressed JSON lines. It is NOT in random order (the start of the file is mostly positive), so never
take a prefix or a slice of it as a sample: split by the assignment in splits/pairs/ instead.

Standard library only.
"""

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from profile_data import DEFAULT_INPUT

ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = DEFAULT_INPUT.parent.parent / "pairs" / "pairs-20251209.json.gz"
SPLIT_DIR = ROOT / "splits" / "pairs"
# One character per pair, in file order. "4" is a final hold-out pair whose entities appear in no other split.
SPLIT_CODES = {"train": "0", "dev": "1", "test": "2", "final": "3", "final_clean": "4"}
SPLIT_MEMBERS = {
    "train": {"0"},
    "dev": {"1"},
    "test": {"2"},
    "final": {"3", "4"},
    "final_clean": {"4"},
}
FINAL_SPLITS = {"final", "final_clean"}

# Schemas of things that have names: matching these is the screening task. The rest of the file labels
# relationship and role records (Occupancy, Succession, Position, Ownership, Family, Directorship, ...).
ENTITY_SCHEMAS = frozenset(
    {"Person", "Organization", "Company", "LegalEntity", "PublicBody", "Vessel", "Airplane", "Security", "CryptoWallet"}
)


def in_scope(record: dict) -> bool:
    """True if both sides are name-bearing entities rather than relationship or role records."""
    return record["left"].get("schema") in ENTITY_SCHEMAS and record["right"].get("schema") in ENTITY_SCHEMAS


def iter_pairs(path: Path = PAIRS_PATH) -> Iterator[tuple[int, dict]]:
    """(line number from 0, record) for every pair, streaming. Never loads the file whole."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            yield index, json.loads(line)


def pair_key_digest_update(digest: "hashlib._Hash", record: dict) -> None:
    digest.update(f"{record['left']['id']}\t{record['right']['id']}\n".encode())


def load_assignment(split_dir: Path = SPLIT_DIR) -> tuple[str, dict]:
    """The split code of every pair in file order, and the manifest. Verified against the manifest checksum."""
    manifest = json.loads((split_dir / "manifest.json").read_text())
    with gzip.open(split_dir / "assignment.txt.gz", "rt") as fh:
        codes = fh.read().strip()
    if hashlib.sha256(codes.encode()).hexdigest() != manifest["assignment_sha256"]:
        raise ValueError("The split assignment does not match its recorded checksum.")
    return codes, manifest


def iter_split(
    split: str,
    *,
    allow_final: bool = False,
    in_scope_only: bool = False,
    path: Path = PAIRS_PATH,
    split_dir: Path = SPLIT_DIR,
) -> Iterator[tuple[int, dict]]:
    """Pairs of one split, streaming. The final hold-out is refused unless allow_final is True.

    Only the final evaluation may pass allow_final. The whole file is read, and at the end the order of
    its pair ids is checked against the manifest, so an assignment cannot be applied to a different file.
    """
    if split in FINAL_SPLITS and not allow_final:
        raise PermissionError(f"The {split!r} split is the frozen final hold-out and is only for the final evaluation.")
    codes, manifest = load_assignment(split_dir)
    wanted = SPLIT_MEMBERS[split]
    digest = hashlib.sha256()
    count = 0
    for index, record in iter_pairs(path):
        pair_key_digest_update(digest, record)
        count += 1
        if count > len(codes):
            raise ValueError("The pairs file has more pairs than the split assignment.")
        if codes[index] in wanted and (not in_scope_only or in_scope(record)):
            yield index, record
    if count != len(codes) or digest.hexdigest() != manifest["pair_keys_sha256"]:
        raise ValueError("The pairs file is not the one the split assignment was made for.")
