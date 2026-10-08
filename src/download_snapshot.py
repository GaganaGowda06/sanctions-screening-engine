"""Download another published OpenSanctions "sanctions" snapshot, for the list-update regression gate.

Run:
    python src/download_snapshot.py --after-current   # the latest snapshot, if newer than data/sanctions/VERSION
    python src/download_snapshot.py --date 20260701   # a dated snapshot (used for the reverse-time churn test)

The snapshot goes to data/snapshots/<version>/ (git-ignored) as entities.ftm.json plus index.json and a VERSION file
in the format download_data.py writes. The file is only kept if its SHA-1 and size match the index. If --after-current
finds nothing newer it says so and exits 3: the gate can then be run in the other direction on an older snapshot, and
every result of that run is labelled a REVERSE-TIME CHURN TEST.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRENT_VERSION_FILE = ROOT / "data" / "sanctions" / "VERSION"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
LATEST_INDEX = "https://data.opensanctions.org/datasets/latest/sanctions/index.json"
DATED_INDEX = "https://data.opensanctions.org/datasets/{date}/sanctions/index.json"
RESOURCE = "entities.ftm.json"
CHUNK = 1024 * 1024


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def version_stamp(version: str) -> str:
    """The sortable timestamp at the front of a version such as 20261006074704-hib."""
    return version.split("-")[0]


def current_version(path: Path = CURRENT_VERSION_FILE) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("version="):
            return line.split("=", 1)[1]
    raise ValueError(f"no version= line in {path}")


def find_resource(index: dict) -> dict:
    for resource in index.get("resources", []):
        if resource.get("name") == RESOURCE:
            return resource
    raise ValueError(f"no resource named {RESOURCE!r} in the index")


def newer_than(current: str, index: dict) -> bool:
    return version_stamp(index["version"]) > version_stamp(current)


def download_snapshot(index: dict, directory: Path = SNAPSHOT_DIR) -> Path:
    """Download the entities file of `index` into directory/<version>/ and keep it only if SHA-1 and size match."""
    resource = find_resource(index)
    target_dir = directory / index["version"]
    target = target_dir / RESOURCE
    if target.exists():
        return target
    target_dir.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    sha1, done = hashlib.sha1(), 0  # noqa: S324  (the index publishes SHA-1)
    with urllib.request.urlopen(resource["url"], timeout=60) as response, open(part, "wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            sha1.update(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:8.1f} MB", end="", flush=True)
    print()
    if sha1.hexdigest() != resource["checksum"] or (resource.get("size") and done != resource["size"]):
        part.unlink(missing_ok=True)
        raise ValueError(f"checksum or size mismatch for {index['version']}: got {sha1.hexdigest()} / {done} bytes")
    part.replace(target)
    (target_dir / "index.json").write_text(json.dumps(index, indent=2))
    (target_dir / "VERSION").write_text(
        "\n".join(
            [
                f"version={index['version']}",
                f"sha1={sha1.hexdigest()}",
                f"size_bytes={done}",
                f"url={resource['url']}",
                f"verified_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            ]
        )
        + "\n"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--after-current", action="store_true", help="the latest snapshot, if newer than data/sanctions")
    group.add_argument("--date", help="a dated snapshot, YYYYMMDD")
    args = parser.parse_args(argv)
    index = fetch_json(LATEST_INDEX if args.after_current else DATED_INDEX.format(date=args.date))
    here = current_version()
    print(f"current {here}; remote {index['version']}")
    if args.after_current and not newer_than(here, index):
        print("no newer snapshot is published")
        return 3
    try:
        path = download_snapshot(index)
    except ValueError as exc:
        print(f"not kept: {exc}", file=sys.stderr)
        return 1
    print(f"verified against the index and saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
