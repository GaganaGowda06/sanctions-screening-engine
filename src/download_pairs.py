"""Download the OpenSanctions Pairs labelled dataset (analyst-labelled entity pairs).

Source: OpenSanctions, "Matcher training data", snapshot of 2025-12-09. Hugging Face hosts a mirror
of the same file (sanctions-er-anon/opensanctions_pairs).

Licence: non-commercial use only (CC-BY-NC-4.0 on the Hugging Face dataset card; "non-commercial use"
on the OpenSanctions page). Commercial use needs a separate licence from OpenSanctions. The paper
describing the dataset is licensed separately (CC BY 4.0 on arXiv), which is not the data licence.

Cite:
  Smith, C., Sesodia, M., Lindenberg, F., Schroeder de Witt, C. (2026). OpenSanctions Pairs:
  Large-Scale Entity Matching with LLMs. arXiv:2603.11051.
  OpenSanctions. Pairs: Cross-Referencing Training Data.
  https://www.opensanctions.org/docs/opensource/pairs/ (snapshot dated 2025-12-09).

Checks: the size, the MD5 that OpenSanctions publishes as the file's ETag, and the SHA-256 that
Hugging Face publishes for the mirror. A mismatch deletes the partial file and exits.
Re-runnable: skips the download if the file is already there and verified. --force overrides.

Run:
    python src/download_pairs.py
"""

import argparse
import hashlib
import json
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://data.opensanctions.org/contrib/training/pairs-20251209.json.gz"
MIRROR = "https://huggingface.co/datasets/sanctions-er-anon/opensanctions_pairs/resolve/main/pairs.json.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pairs"
FILE_NAME = "pairs-20251209.json.gz"
CHUNK = 1024 * 1024


def head(url: str) -> dict[str, str]:
    """Response headers of a HEAD request, lower-cased, without following the cross-host redirect."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: object, **kwargs: object) -> None:
            return None

    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener.open(request, timeout=60) as response:
            return {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as error:  # a 302 to the CDN still carries the headers we want
        return {k.lower(): v for k, v in error.headers.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download again even if the file is verified")
    args = parser.parse_args()
    print(f"Python {platform.python_version()}", file=sys.stderr)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target, part, source_path = DATA_DIR / FILE_NAME, DATA_DIR / (FILE_NAME + ".part"), DATA_DIR / "SOURCE.json"
    origin = head(URL)
    expected_size = int(origin["content-length"])
    expected_md5 = origin.get("etag", "").strip('"')
    mirror = head(MIRROR)
    expected_sha256 = mirror.get("x-linked-etag", "").strip('"') or None
    if len(expected_md5) != 32:
        sys.exit(f"The ETag {expected_md5!r} is not an MD5, so the file cannot be verified. Stopping.")

    if not args.force and target.exists() and source_path.exists() and target.stat().st_size == expected_size:
        print(f"Already downloaded and verified: {target}. Use --force to download again.")
        return

    print(f"Downloading {URL} ({expected_size / 1e6:.0f} MB)")
    md5, sha256, done = hashlib.md5(), hashlib.sha256(), 0  # noqa: S324 (a published checksum, not security)
    with urllib.request.urlopen(URL, timeout=60) as response, open(part, "wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            md5.update(chunk)
            sha256.update(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:8.1f} MB {100 * done / expected_size:5.1f}%", end="", flush=True)
    print()

    problems = []
    if done != expected_size:
        problems.append(f"size {done} != {expected_size}")
    if md5.hexdigest() != expected_md5:
        problems.append(f"MD5 {md5.hexdigest()} != {expected_md5}")
    if expected_sha256 and sha256.hexdigest() != expected_sha256:
        problems.append(f"SHA-256 {sha256.hexdigest()} != {expected_sha256}")
    if problems:
        part.unlink(missing_ok=True)
        sys.exit("Verification failed, partial file deleted: " + "; ".join(problems))

    part.replace(target)
    source_path.write_text(
        json.dumps(
            {
                "file": FILE_NAME,
                "url": URL,
                "mirror": MIRROR,
                "snapshot": "2025-12-09",
                "size_bytes": done,
                "md5": md5.hexdigest(),
                "md5_verified_against": "ETag published by data.opensanctions.org",
                "sha256": sha256.hexdigest(),
                "sha256_verified_against": (
                    "x-linked-etag published by Hugging Face" if expected_sha256 else "not checked"
                ),
                "licence": "non-commercial use only (CC-BY-NC-4.0 per the Hugging Face dataset card)",
                "citation": "Smith, Sesodia, Lindenberg, Schroeder de Witt (2026), OpenSanctions Pairs: "
                "Large-Scale Entity Matching with LLMs, arXiv:2603.11051; OpenSanctions, Pairs: "
                "Cross-Referencing Training Data, snapshot 2025-12-09",
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified (size, MD5 and SHA-256 {sha256.hexdigest()[:12]}...). Saved {target}")


if __name__ == "__main__":
    main()
