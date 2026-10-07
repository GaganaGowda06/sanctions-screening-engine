"""Build the deployment rarity table (see idf_table.py) from the downloaded sanctions list.

Run:
    python src/build_idf.py
Writes data/idf/deployment_idf.json.gz and models/idf_manifest.json.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

from entities import load_named_entities
from idf_table import MANIFEST_PATH, TABLE_PATH, build_table, write_table
from profile_data import DEFAULT_INPUT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()}", file=sys.stderr)

    targets = [e for e in load_named_entities(args.input) if e["target"]]
    frequency, documents = build_table(targets)
    sha = write_table(args.table, frequency, documents)
    version_file = args.input.with_name("VERSION")
    version = dict(line.split("=", 1) for line in version_file.read_text().splitlines() if "=" in line)
    manifest = {
        "entities": documents,
        "tokens": len(frequency),
        "table_sha256": sha,
        "dataset_version": version.get("version"),
        "dataset_sha1": version.get("sha1"),
        "source": "named target entities of the OpenSanctions sanctions list the engine screens",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
