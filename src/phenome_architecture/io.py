from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "unnamed"


def find_header_row(raw: pd.DataFrame, required: str = "NCU Number") -> int:
    required_lower = required.lower()
    for idx in range(raw.shape[0]):
        values = [str(x).strip().lower() for x in raw.iloc[idx].tolist()]
        if required_lower in values:
            return idx
    raise ValueError(f"Could not find a header row containing {required!r}")


def read_excel_table(path: Path, sheet_name: str, required: str = "NCU Number") -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_idx = find_header_row(raw, required=required)
    header = raw.iloc[header_idx].tolist()
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")
    df = df.loc[:, [c for c in df.columns if not pd.isna(c)]]
    return df.reset_index(drop=True)


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None
