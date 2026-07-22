#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phenome_architecture.io import sha256_file


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory downloaded source files")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="config/source_manifest.csv")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = pd.read_csv(manifest_path)
    records: list[dict[str, object]] = []
    missing_required: list[str] = []
    for _, row in manifest.iterrows():
        relative_path = Path("data") / "raw" / str(row["group"]) / str(row["filename"])
        path = root / relative_path
        exists = path.exists() and path.stat().st_size > 0
        required = as_bool(row.get("required", True))
        if required and not exists:
            missing_required.append(str(row["source_id"]))
        records.append(
            {
                "source_id": row["source_id"],
                "group": row["group"],
                "filename": row["filename"],
                "relative_path": relative_path.as_posix(),
                "required": required,
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": sha256_file(path) if exists else "",
                "url": row["url"],
                "description": row["description"],
            }
        )
    output = pd.DataFrame(records)
    output_dir = root / "reports" / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "source_inventory.csv", index=False)
    print(output[["source_id", "required", "exists", "bytes"]].to_string(index=False))
    if missing_required:
        raise SystemExit("Required source files are missing: " + ", ".join(missing_required))


if __name__ == "__main__":
    main()
