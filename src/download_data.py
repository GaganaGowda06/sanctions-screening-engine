"""Download the OpenSanctions "sanctions" collection as FollowTheMoney entities.

Re-runnable: skips the download when the saved index.json has the same version
and the entities file is already on disk with the expected size.

Provenance: writes data/sanctions/VERSION (dataset version and the SHA1 of the file
as downloaded and verified) so a later run can confirm it holds the same data.

Run:
    python src/download_data.py            # download if new version
    python src/download_data.py --force     # re-download regardless
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

INDEX_URL = "https://data.opensanctions.org/datasets/latest/sanctions/index.json"
RESOURCE_NAME = "entities.ftm.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sanctions"
CHUNK = 1024 * 1024


def fetch_index() -> dict:
    with urllib.request.urlopen(INDEX_URL, timeout=60) as resp:
        return json.load(resp)


def find_resource(index: dict) -> dict:
    for res in index.get("resources", []):
        if res.get("name") == RESOURCE_NAME:
            return res
    sys.exit(f"No resource named {RESOURCE_NAME!r} in index.json")


def already_current(index: dict, resource: dict, target: Path, index_path: Path) -> bool:
    if not (target.exists() and index_path.exists()):
        return False
    try:
        saved = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return False
    return saved.get("version") == index.get("version") and target.stat().st_size == resource.get("size")


def sha1_of(path: Path) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            sha1.update(chunk)
    return sha1.hexdigest()


def write_version_file(path: Path, index: dict, resource: dict, sha1: str) -> None:
    lines = [
        f"version={index.get('version')}",
        f"sha1={sha1}",
        f"size_bytes={resource.get('size')}",
        f"url={resource['url']}",
        f"verified_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ]
    path.write_text("\n".join(lines) + "\n")


def download(url: str, part: Path, expected_size: int | None) -> str:
    sha1 = hashlib.sha1()
    done = 0
    start = time.time()
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or expected_size or 0)
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            sha1.update(chunk)
            done += len(chunk)
            rate = done / max(time.time() - start, 1e-6) / 1e6
            pct = f"{100 * done / total:5.1f}%" if total else "   ?%"
            print(f"\r  {done / 1e6:8.1f} MB {pct}  {rate:5.1f} MB/s", end="", flush=True)
    print()
    return sha1.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if the saved version matches")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / RESOURCE_NAME
    part = target.with_name(target.name + ".part")
    index_path = DATA_DIR / "index.json"
    version_path = DATA_DIR / "VERSION"

    print(f"Python {platform.python_version()}")
    index = fetch_index()
    resource = find_resource(index)
    print(f"Remote version {index.get('version')}, {resource.get('size', 0) / 1e6:.1f} MB")

    if not args.force and already_current(index, resource, target, index_path):
        print(f"Already up to date: {target}. Use --force to re-download.")
        if not version_path.exists():
            actual = sha1_of(target)
            if actual != resource.get("checksum"):
                sys.exit(f"Existing file fails SHA1 check ({actual}). Re-run with --force.")
            write_version_file(version_path, index, resource, actual)
            print(f"SHA1 OK ({actual}). Wrote {version_path}")
        return

    print(f"Downloading {resource['url']}")
    actual = download(resource["url"], part, resource.get("size"))
    expected = resource.get("checksum")
    if actual != expected:
        part.unlink(missing_ok=True)
        sys.exit(f"SHA1 mismatch: expected {expected}, got {actual}. Partial file deleted.")

    part.replace(target)
    index_path.write_text(json.dumps(index, indent=2))
    write_version_file(version_path, index, resource, actual)
    print(f"SHA1 OK ({actual}). Saved {target}, {index_path} and {version_path}")


if __name__ == "__main__":
    main()
