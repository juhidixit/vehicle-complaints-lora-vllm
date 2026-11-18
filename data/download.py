"""Download the NHTSA complaints flat file and its layout document, and check the layout.

    python data/download.py --out data/raw

The flat file is public-domain US government data, updated daily by NHTSA. The download date,
the server's Last-Modified header and the SHA-256 are recorded in data/raw/download_meta.json,
which build_splits.py copies into results/data_stats.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from vclora.nhtsa import FLAT_URL, LAYOUT_URL, verify_layout

UA = {"User-Agent": "vehicle-complaints-lora-vllm data script"}


def fetch(url: str, dest: Path) -> dict:
    req = urllib.request.Request(url, headers=UA)
    sha = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        last_modified = resp.headers.get("Last-Modified")
        while chunk := resp.read(1 << 20):
            sha.update(chunk)
            fh.write(chunk)
    return {"url": url, "bytes": dest.stat().st_size, "sha256": sha.hexdigest(),
            "last_modified": last_modified}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--skip-if-present", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    layout_path = args.out / "CMPL.txt"
    fetch(LAYOUT_URL, layout_path)
    found = verify_layout(layout_path.read_text(encoding="latin-1"))
    print(f"layout OK: {len(found)} documented fields match our field order")

    zip_path = args.out / "FLAT_CMPL.zip"
    meta_path = args.out / "download_meta.json"
    if args.skip_if_present and zip_path.exists() and meta_path.exists():
        print("flat file already present")
        return
    tmp = zip_path.with_suffix(".part")
    meta = fetch(FLAT_URL, tmp)
    shutil.move(tmp, zip_path)
    meta["downloaded_at_utc"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["layout_fields_checked"] = len(found)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
