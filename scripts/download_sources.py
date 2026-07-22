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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public source files")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="config/source_manifest.csv")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def download(url: str, output: Path, retries: int, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not overwrite:
        print(f"present\t{output.relative_to(output.parents[2])}")
        return
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    headers = {"User-Agent": "Mozilla/5.0 neurospora-phenome-architecture"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if temporary.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            temporary.replace(output)
            print(f"downloaded\t{output.name}")
            return
        except (HTTPError, URLError, RuntimeError, TimeoutError, OSError) as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    manifest = resolve_path(root, args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Source manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        output = root / "data" / "raw" / row.get("group", "other") / row["filename"]
        try:
            download(row["url"], output, args.retries, args.overwrite)
        except Exception:
            if as_bool(row.get("required", "true")):
                raise
            print(f"optional source unavailable\t{row['source_id']}", file=sys.stderr)


if __name__ == "__main__":
    main()
