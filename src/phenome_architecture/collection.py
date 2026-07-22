from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

NCU_RE = re.compile(r"\bNCU\s*0*(\d{5})(?:\.\d+)?\b", re.IGNORECASE)
FGSC_RE = re.compile(r"\bFGSC\s*#?\s*(\d{4,5})\b", re.IGNORECASE)
INVALID_RE = re.compile(
    r"incorrect|southern\s+wrong|questionable|inauthentic|wrong\s+strain|"
    r"unable\s+to\s+revive|awaiting\s+replacement|recalled|soft\s+mutation|secondary\s+mutation",
    re.IGNORECASE,
)
UNAVAILABLE_RE = re.compile(
    r"not\s+available|unavailable|not\s+achieved|unable\s+to\s+obtain|"
    r"unable\s+to\s+recover|failed|no\s+strain|not\s+made",
    re.IGNORECASE,
)
HET_RE = re.compile(r"\bheterokary(?:on|otic|ons)?\b", re.IGNORECASE)
HOMO_RE = re.compile(r"\bhomokary(?:on|otic|ons)?\b", re.IGNORECASE)
HET_UNAVAILABLE_RE = re.compile(
    r"heterokary(?:on|otic|ons)?[^|;]{0,45}(?:not\s+available|unavailable|not\s+achieved|failed|unable\s+to\s+recover)",
    re.IGNORECASE,
)
HOMO_UNAVAILABLE_RE = re.compile(
    r"homokary(?:on|otic|ons)?[^|;]{0,45}(?:not\s+available|unavailable|not\s+achieved|failed|unable\s+to\s+obtain|unable\s+to\s+recover)",
    re.IGNORECASE,
)
NONESSENTIAL_RE = re.compile(r"non[- ]?essential|not\s+essential", re.IGNORECASE)
ESSENTIAL_RE = re.compile(r"\bessential\b|predicted\s+lethal|null\s+lethal", re.IGNORECASE)


def canonical_ncu(value: object) -> str:
    """Return an annotation-version-independent NCU identifier or an empty string."""
    match = NCU_RE.search(str(value))
    return f"NCU{match.group(1)}" if match else ""


def extract_ncu_ids(value: object) -> set[str]:
    """Extract every annotation-version-independent NCU identifier from a value."""
    return {f"NCU{match.group(1)}" for match in NCU_RE.finditer(str(value))}


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def normalize_header(value: object) -> str:
    text = safe_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unnamed"


def find_header_row(raw: pd.DataFrame, max_rows: int = 80) -> int | None:
    """Locate a likely header row in heterogeneous KO workbooks."""
    for idx in range(min(max_rows, len(raw))):
        cells = [safe_text(x).lower() for x in raw.iloc[idx].tolist()]
        joined = " | ".join(cells)
        has_gene = any(token in joined for token in ("ncu", "gene", "orf", "locus"))
        has_strain = any(token in joined for token in ("fgsc", "strain", "homokary", "heterokary"))
        if has_gene and has_strain:
            return idx
    return None


def _infer_fgsc_ids(row: Mapping[str, object], row_text: str, ncu_id: str) -> list[str]:
    found = set(FGSC_RE.findall(row_text))
    for key, value in row.items():
        key_low = str(key).lower()
        if "fgsc" not in key_low and "strain" not in key_low:
            continue
        text = safe_text(value)
        for token in re.findall(r"\b\d{4,5}\b", text):
            if token != ncu_id.removeprefix("NCU"):
                found.add(token)
    return sorted(found, key=int)


def _infer_record_status(
    row: Mapping[str, object], row_text: str, fgsc_ids: Sequence[str]
) -> tuple[str, str]:
    """Classify an individual strain record, preserving confidence of the inference.

    Availability statements are interpreted at the karyon level. For example,
    ``homokaryon not available; heterokaryon retained`` is classified as a
    heterokaryon record rather than as a generic unavailable record.
    """
    if INVALID_RE.search(row_text):
        return "invalid_or_unavailable", "explicit"

    row_low = row_text.lower()
    homo_unavailable = bool(HOMO_UNAVAILABLE_RE.search(row_text))
    hetero_unavailable = bool(HET_UNAVAILABLE_RE.search(row_text))
    homo_present = bool(HOMO_RE.search(row_text)) and not homo_unavailable
    hetero_present = bool(HET_RE.search(row_text)) and not hetero_unavailable

    # Explicit status/type columns are more reliable than unlabeled strain IDs.
    for key, value in row.items():
        key_low = str(key).lower()
        val_low = safe_text(value).lower()
        if "status" in key_low or "type" in key_low or "kary" in key_low:
            if val_low in {"het", "hetero", "heterokaryon", "heterokaryotic", "heterokaryon only"}:
                hetero_present = True
            if val_low in {"hom", "homo", "homokaryon", "homokaryotic", "homokaryon available"}:
                homo_present = True

    # A verified homokaryon takes precedence when both record types exist.
    if homo_present:
        return "homokaryon", "explicit"
    if hetero_present:
        return "heterokaryon", "explicit"
    if UNAVAILABLE_RE.search(row_low):
        return "invalid_or_unavailable", "explicit"

    # In the archived availability workbook, heterokaryons are normally labeled.
    # A listed, non-invalid KO record without a heterokaryon label is therefore
    # treated as an available homokaryon, but the inference is flagged.
    if fgsc_ids:
        return "homokaryon", "inferred_from_available_record"
    return "unresolved", "insufficient_information"


def _records_from_dataframe(
    df: pd.DataFrame, sheet_name: str, source_file: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row_index, series in df.iterrows():
        row = {str(k): v for k, v in series.items()}
        row_text = " | ".join(safe_text(v) for v in row.values() if safe_text(v))
        ncu_ids: set[str] = set()
        for value in row.values():
            ncu_ids.update(extract_ncu_ids(value))
        ncu_ids.update(extract_ncu_ids(row_text))
        ncu_ids = sorted(ncu_ids)
        if not ncu_ids:
            continue
        for ncu_id in ncu_ids:
            fgsc_ids = _infer_fgsc_ids(row, row_text, ncu_id)
            status, confidence = _infer_record_status(row, row_text, fgsc_ids)
            records.append(
                {
                    "ncu_id": ncu_id,
                    "record_status": status,
                    "status_confidence": confidence,
                    "fgsc_ids": ";".join(fgsc_ids),
                    "n_fgsc_ids": len(fgsc_ids),
                    "source_file": source_file,
                    "source_sheet": sheet_name,
                    "source_row": int(row_index) + 1,
                    "raw_record": row_text[:4000],
                }
            )
    return records


def parse_ko_workbook(path: Path) -> pd.DataFrame:
    """Parse an official Neurospora KO availability workbook with schema auto-detection."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"KO availability workbook not found or empty: {path}")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unknown extension is not supported and will be removed",
                category=UserWarning,
            )
            excel = pd.ExcelFile(path)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not read {path.name}. Install xlrd>=2.0 for legacy .xls files."
        ) from exc
    records: list[dict[str, object]] = []
    try:
        for sheet in excel.sheet_names:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Unknown extension is not supported and will be removed",
                    category=UserWarning,
                )
                raw = pd.read_excel(excel, sheet_name=sheet, header=None, dtype=object)
            raw = raw.dropna(how="all")
            if raw.empty:
                continue
            header_idx = find_header_row(raw)
            if header_idx is None:
                frame = raw.copy()
                frame.columns = [f"column_{i+1}" for i in range(frame.shape[1])]
            else:
                headers = [normalize_header(x) for x in raw.iloc[header_idx].tolist()]
                # Excel sheets occasionally repeat a column name. Make names unique.
                seen: defaultdict[str, int] = defaultdict(int)
                unique_headers: list[str] = []
                for header in headers:
                    seen[header] += 1
                    unique_headers.append(
                        header if seen[header] == 1 else f"{header}_{seen[header]}"
                    )
                frame = raw.iloc[header_idx + 1 :].copy()
                frame.columns = unique_headers
            frame = frame.dropna(how="all").reset_index(drop=True)
            records.extend(_records_from_dataframe(frame, sheet, path.name))
    finally:
        excel.close()
    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError(
            f"No NCU identifiers could be parsed from {path}. "
            "Open the workbook and confirm that it is the Neurospora KO availability file."
        )
    return out.drop_duplicates().reset_index(drop=True)


def load_exclusions(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["fgsc_id", "ncu_id", "reason", "source_url"])
    out = pd.read_csv(path, dtype=str).fillna("")
    if "ncu_id" in out:
        out["ncu_id"] = out["ncu_id"].map(canonical_ncu)
    if "fgsc_id" in out:
        out["fgsc_id"] = (
            out["fgsc_id"].astype(str).str.extract(r"(\d{4,5})", expand=False).fillna("")
        )
    return out


def apply_exclusions(records: pd.DataFrame, exclusions: pd.DataFrame) -> pd.DataFrame:
    out = records.copy()
    excluded_fgsc = set(exclusions.get("fgsc_id", pd.Series(dtype=str)).dropna().astype(str))
    excluded_ncu = set(exclusions.get("ncu_id", pd.Series(dtype=str)).dropna().astype(str))

    def record_excluded(row: pd.Series) -> bool:
        ids = set(str(row.get("fgsc_ids", "")).split(";")) - {""}
        return bool(ids & excluded_fgsc) or str(row.get("ncu_id", "")) in excluded_ncu

    out["excluded_from_status_analysis"] = out.apply(record_excluded, axis=1)
    reason_by_fgsc = dict(zip(exclusions.get("fgsc_id", []), exclusions.get("reason", [])))
    reason_by_ncu = dict(zip(exclusions.get("ncu_id", []), exclusions.get("reason", [])))

    def exclusion_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        for fgsc in str(row.get("fgsc_ids", "")).split(";"):
            if fgsc in reason_by_fgsc:
                reasons.append(reason_by_fgsc[fgsc])
        ncu = str(row.get("ncu_id", ""))
        if ncu in reason_by_ncu:
            reasons.append(reason_by_ncu[ncu])
        return "; ".join(sorted(set(x for x in reasons if x)))

    out["status_exclusion_reason"] = out.apply(exclusion_reason, axis=1)
    return out


def aggregate_gene_status(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse strain-level records to one conservative status per NCU gene.

    Two status calls are retained:

    ``ko_gene_status``
        Uses explicit records plus archived available-strain records that can be
        interpreted as homokaryons. This is the primary, inclusive call.

    ``explicit_ko_gene_status``
        Uses only records whose karyon status was explicitly stated in the
        source workbook. This provides a conservative sensitivity analysis and
        prevents unlabeled available records from silently driving the result.
    """
    rows: list[dict[str, object]] = []
    for ncu_id, group in records.groupby("ncu_id", sort=True):
        valid = group[~group["excluded_from_status_analysis"].fillna(False)].copy()
        valid_homo = valid[valid["record_status"] == "homokaryon"]
        valid_hetero = valid[valid["record_status"] == "heterokaryon"]
        explicit = valid[valid["status_confidence"].astype(str).eq("explicit")].copy()
        explicit_homo = explicit[explicit["record_status"] == "homokaryon"]
        explicit_hetero = explicit[explicit["record_status"] == "heterokaryon"]
        inferred_homo = valid[
            valid["record_status"].eq("homokaryon")
            & ~valid["status_confidence"].astype(str).eq("explicit")
        ]

        if not valid_homo.empty:
            status = "homokaryon_available"
        elif not valid_hetero.empty:
            status = "heterokaryon_only"
        elif not valid.empty and (valid["record_status"] == "invalid_or_unavailable").any():
            status = "invalid_or_unavailable_only"
        elif not valid.empty:
            status = "unresolved"
        else:
            status = "invalid_or_unavailable_only"

        if not explicit_homo.empty:
            explicit_status = "homokaryon_available"
        elif not explicit_hetero.empty:
            explicit_status = "heterokaryon_only"
        elif not explicit.empty and (explicit["record_status"] == "invalid_or_unavailable").any():
            explicit_status = "invalid_or_unavailable_only"
        else:
            explicit_status = "unresolved_no_explicit_call"

        fgsc_ids = sorted(
            {
                token
                for value in group["fgsc_ids"].fillna("")
                for token in str(value).split(";")
                if token
            },
            key=int,
        )
        rows.append(
            {
                "ncu_id": ncu_id,
                "ko_gene_status": status,
                "explicit_ko_gene_status": explicit_status,
                "has_homokaryon": not valid_homo.empty,
                "has_heterokaryon": not valid_hetero.empty,
                "has_explicit_homokaryon": not explicit_homo.empty,
                "has_explicit_heterokaryon": not explicit_hetero.empty,
                "primary_status_uses_inference": bool(
                    status == "homokaryon_available"
                    and explicit_homo.empty
                    and not inferred_homo.empty
                ),
                "n_source_records": len(group),
                "n_valid_records": len(valid),
                "n_explicit_records": len(explicit),
                "n_inferred_records": int(
                    (valid["status_confidence"].astype(str) != "explicit").sum()
                ),
                "n_excluded_records": int(
                    group["excluded_from_status_analysis"].fillna(False).sum()
                ),
                "fgsc_ids": ";".join(fgsc_ids),
                "status_confidence": (
                    ";".join(sorted(set(valid["status_confidence"].dropna().astype(str))))
                    if not valid.empty
                    else ""
                ),
                "source_sheets": ";".join(sorted(set(group["source_sheet"].dropna().astype(str)))),
                "status_notes": ";".join(
                    sorted(set(group["status_exclusion_reason"].dropna().astype(str)) - {""})
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_gff_gene_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    opener = __import__("gzip").open if path.suffix == ".gz" else open
    genes: set[str] = set()
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2].lower() not in {"gene", "mrna", "transcript"}:
                continue
            ncu = canonical_ncu(parts[8])
            if ncu:
                genes.add(ncu)
    return genes


def parse_sbml_genes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {canonical_ncu(match.group(0)) for match in NCU_RE.finditer(text)} - {""}


def _essential_rows_from_sheet(
    path: Path, sheet: str, raw: pd.DataFrame
) -> list[dict[str, object]]:
    sheet_low = sheet.lower()
    sheet_is_essential = (
        "essential" in sheet_low
        and "nonessential" not in sheet_low
        and "non-essential" not in sheet_low
    )
    header_idx = find_header_row(raw)
    if header_idx is None:
        frame = raw.copy()
        frame.columns = [f"column_{i+1}" for i in range(frame.shape[1])]
    else:
        headers = [normalize_header(x) for x in raw.iloc[header_idx].tolist()]
        frame = raw.iloc[header_idx + 1 :].copy()
        frame.columns = headers
    rows: list[dict[str, object]] = []
    for idx, series in frame.dropna(how="all").iterrows():
        row_text = " | ".join(safe_text(x) for x in series.tolist() if safe_text(x))
        ncu_ids: set[str] = set()
        for value in series.tolist():
            ncu_ids.update(extract_ncu_ids(value))
        ncu_ids.update(extract_ncu_ids(row_text))
        ncu_ids = sorted(ncu_ids)
        if not ncu_ids:
            continue
        row_is_nonessential = bool(NONESSENTIAL_RE.search(row_text))
        row_is_essential = bool(ESSENTIAL_RE.search(row_text)) and not row_is_nonessential
        if not (sheet_is_essential or row_is_essential):
            continue
        for ncu in ncu_ids:
            rows.append(
                {
                    "ncu_id": ncu,
                    "dreyfuss_predicted_essential": True,
                    "source_file": path.name,
                    "source_sheet": sheet,
                    "source_row": int(idx) + 1,
                    "prediction_evidence": (
                        "sheet_title" if sheet_is_essential and not row_is_essential else "row_text"
                    ),
                    "raw_prediction_record": row_text[:3000],
                }
            )
    return rows


def parse_dreyfuss_essentiality(
    path: Path | None, essential_genes_csv: Path | None = None
) -> pd.DataFrame:
    """Extract predicted-essential NCU genes from the Dreyfuss supplementary workbook.

    An optional one-column CSV (ncu_id) can supply an explicit essential-gene list
    when the supplementary workbook cannot be parsed consistently.
    """
    if essential_genes_csv is not None and essential_genes_csv.exists():
        provided = pd.read_csv(essential_genes_csv, dtype=str)
        if "ncu_id" not in provided.columns:
            raise ValueError(
                f"Essential-gene file must contain an ncu_id column: {essential_genes_csv}"
            )
        provided["ncu_id"] = provided["ncu_id"].map(canonical_ncu)
        provided = provided[provided["ncu_id"] != ""].drop_duplicates("ncu_id")
        provided["dreyfuss_predicted_essential"] = True
        provided["source_file"] = essential_genes_csv.name
        provided["source_sheet"] = "provided_gene_list"
        provided["prediction_evidence"] = "provided_gene_list"
        return provided
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "ncu_id",
                "dreyfuss_predicted_essential",
                "source_file",
                "source_sheet",
                "prediction_evidence",
            ]
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unknown extension is not supported and will be removed",
                category=UserWarning,
            )
            excel = pd.ExcelFile(path)
    except ImportError as exc:
        raise RuntimeError("Install xlrd>=2.0 to read the Dreyfuss legacy .xls workbook") from exc
    rows: list[dict[str, object]] = []
    try:
        for sheet in excel.sheet_names:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Unknown extension is not supported and will be removed",
                    category=UserWarning,
                )
                raw = pd.read_excel(excel, sheet_name=sheet, header=None, dtype=object).dropna(
                    how="all"
                )
            rows.extend(_essential_rows_from_sheet(path, sheet, raw))
    finally:
        excel.close()
    out = pd.DataFrame(rows)
    if out.empty:
        warnings.warn(
            f"No predicted-essential genes were detected in {path.name}. "
            "Inspect the workbook and provide config/dreyfuss_essential_genes.csv if needed.",
            RuntimeWarning,
        )
        return pd.DataFrame(
            columns=[
                "ncu_id",
                "dreyfuss_predicted_essential",
                "source_file",
                "source_sheet",
                "prediction_evidence",
            ]
        )
    return (
        out.sort_values(["ncu_id", "source_sheet"]).drop_duplicates("ncu_id").reset_index(drop=True)
    )


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray([1.0 if pd.isna(x) else float(x) for x in p_values], dtype=float)
    if len(p) == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def fisher_enrichment(member: pd.Series, outcome: pd.Series) -> dict[str, float | int]:
    member = member.fillna(False).astype(bool)
    outcome = outcome.fillna(False).astype(bool)
    a = int((member & outcome).sum())
    b = int((member & ~outcome).sum())
    c = int((~member & outcome).sum())
    d = int((~member & ~outcome).sum())
    odds, p_value = stats.fisher_exact([[a, b], [c, d]])
    # Haldane-Anscombe correction gives finite log OR and CI when a cell is zero.
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return {
        "member_outcome": a,
        "member_other": b,
        "nonmember_outcome": c,
        "nonmember_other": d,
        "outcome_rate_member": a / max(a + b, 1),
        "outcome_rate_nonmember": c / max(c + d, 1),
        "odds_ratio": float(odds),
        "log2_odds_ratio_corrected": float(log_or / math.log(2)),
        "odds_ratio_ci95_low_corrected": float(math.exp(log_or - 1.96 * se)),
        "odds_ratio_ci95_high_corrected": float(math.exp(log_or + 1.96 * se)),
        "fisher_p": float(p_value),
    }


def binary_concordance(predicted: pd.Series, observed: pd.Series) -> dict[str, float | int]:
    predicted = predicted.fillna(False).astype(bool)
    observed = observed.fillna(False).astype(bool)
    tp = int((predicted & observed).sum())
    fp = int((predicted & ~observed).sum())
    fn = int((~predicted & observed).sum())
    tn = int((~predicted & ~observed).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else float("nan")
    odds, p = stats.fisher_exact([[tp, fp], [fn, tn]])
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_predictive_value": ppv,
        "negative_predictive_value": npv,
        "matthews_correlation_coefficient": mcc,
        "odds_ratio": float(odds),
        "fisher_p": float(p),
    }
