#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GROUP_DIRS = {
    "carrillo_2020": Path("data/raw/carrillo_2020"),
    "dreyfuss_2013": Path("data/raw/dreyfuss_2013"),
    "ensembl_fungi": Path("data/raw/ensembl_fungi"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public data files listed in config/source_manifest.csv")
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--manifest", default="config/source_manifest.csv", help="Source manifest relative to project root or absolute path")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_manifest(root: Path, manifest: str) -> Path:
    path = Path(manifest)
    return path if path.is_absolute() else root / path


def download(url: str, out: Path, retries: int, overwrite: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0 and not overwrite:
        print(f"present\t{out.name}")
        return
    tmp = out.with_suffix(out.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    headers = {"User-Agent": "Mozilla/5.0 phenome-architecture-downloader"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if tmp.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            tmp.replace(out)
            print(f"downloaded\t{out.name}")
            return
        except (HTTPError, URLError, RuntimeError, TimeoutError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    manifest = resolve_manifest(root, args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Source manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        group_dir = GROUP_DIRS.get(row["group"], Path("data/raw/other"))
        out = root / group_dir / row["filename"]
        download(row["url"], out, args.retries, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
