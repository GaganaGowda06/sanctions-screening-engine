"""The rarity table: how many entities of the deployment list carry each name token.

One table is used for training and for serving, so a token weighs the same in both. It is built from
the named target entities of the list the engine screens (the downloaded OpenSanctions file), with the
same name normalisation the scoring uses (normalise.py, legal-form words stripped). A token is counted
once per entity however many of the entity's names contain it.

The table is written to data/idf/deployment_idf.json.gz and described, with its checksum, in
models/idf_manifest.json, which is committed. Loading checks the checksum.

Standard library only.
"""

import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from scoring import scoring_form

ROOT = Path(__file__).resolve().parent.parent
TABLE_PATH = ROOT / "data" / "idf" / "deployment_idf.json.gz"
MANIFEST_PATH = ROOT / "models" / "idf_manifest.json"


def build_table(entities: Iterable[dict]) -> tuple[Counter, int]:
    """(document frequency of every name token, number of entities) over entities with a names list."""
    frequency: Counter = Counter()
    documents = 0
    for entity in entities:
        tokens = {t for name in entity["names"] for t in scoring_form(name).split()}
        if tokens:
            documents += 1
            frequency.update(tokens)
    return frequency, documents


def write_table(path: Path, frequency: Counter, documents: int) -> str:
    """Write the table so its bytes depend only on its content. Returns the SHA-256 of the file."""
    payload = json.dumps({"documents": documents, "df": dict(sorted(frequency.items()))}, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            text.write(payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_table(path: Path = TABLE_PATH, manifest_path: Path = MANIFEST_PATH) -> tuple[dict[str, int], int]:
    """(document frequency, documents). Refuses a table whose checksum is not the recorded one."""
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["table_sha256"]:
        raise ValueError(f"{path} does not match the checksum in {manifest_path}.")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["df"], data["documents"]
