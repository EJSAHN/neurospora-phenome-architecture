from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phenome_architecture.io import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    args = parser.parse_args()
    root = Path(args.project_root)
    manifest_path = root / "config" / "source_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    records = []
    for _, row in manifest.iterrows():
        if row["group"] == "carrillo_2020":
            folder = root / "data" / "raw" / "carrillo_2020"
        elif row["group"] == "dreyfuss_2013":
            folder = root / "data" / "raw" / "dreyfuss_2013"
        elif row["group"] == "ensembl_fungi":
            folder = root / "data" / "raw" / "ensembl_fungi"
        else:
            folder = root / "data" / "raw" / "other"
        path = folder / row["filename"]
        records.append(
            {
                "source_id": row["source_id"],
                "group": row["group"],
                "filename": row["filename"],
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.stat().st_size > 0 else "",
                "path": str(path),
                "url": row["url"],
                "description": row["description"],
            }
        )
    out = pd.DataFrame(records)
    out_dir = root / "reports" / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "source_inventory.csv", index=False)
    print(out[["source_id", "exists", "bytes"]].to_string(index=False))
    if not bool(out.loc[out["source_id"] == "carrillo_2020_additional_file_1", "exists"].iloc[0]):
        raise SystemExit("Required source missing: carrillo_2020_additional_file_1")


if __name__ == "__main__":
    main()
