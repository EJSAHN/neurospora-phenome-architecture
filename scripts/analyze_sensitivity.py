#!/usr/bin/env python
"""Evaluate discretization and gene-class annotation sensitivity."""

from __future__ import annotations

import argparse
import itertools
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

NORMAL_VALUES = {"normal", "normal_range", "normal range", "wild_type", "wild-type", "wt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze discretization and annotation sensitivity"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--random-seed", type=int, default=20260601)
    parser.add_argument("--min-class-size", type=int, default=5)
    parser.add_argument("--output-workbook", default="")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_file(root: Path, preferred: Sequence[Path], patterns: Sequence[str]) -> Optional[Path]:
    hit = first_existing(preferred)
    if hit is not None:
        return hit
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def read_csv_required(path: Optional[Path], label: str) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Could not find {label}. Run scripts/prepare_phenotype_matrix.py first."
        )
    return pd.read_csv(path)


def canonical_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray([1.0 if pd.isna(x) else float(x) for x in pvals], dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q_sorted, 1.0)
    return out


def safe_fisher(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    try:
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        return float(odds), float(p)
    except Exception:
        return np.nan, np.nan


def log2_or_corrected(a: int, b: int, c: int, d: int) -> float:
    return float(np.log2(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))))


def assay_axis(assay: str) -> str:
    low = assay.lower()
    if "basal" in low or "aerial" in low or "hypha" in low:
        return "growth"
    if "conidia" in low:
        return "asexual_development"
    if "protoperithe" in low or "perithe" in low or "ascospore" in low:
        return "sexual_development"
    return "other"


def normalize_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
        .fillna("missing")
    )


def quantile_label(x: pd.Series, probs: Sequence[float], labels: Sequence[str]) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    if vals.notna().sum() < 5 or vals.nunique(dropna=True) <= 3:
        return vals.astype("string").fillna("missing")
    qs = vals.quantile(probs).to_numpy(dtype=float)
    # Prevent duplicate cut points from crashing; ties are common in phenotype tables.
    qs = np.maximum.accumulate(qs)
    out = []
    for v in vals:
        if pd.isna(v):
            out.append("missing")
            continue
        j = int(np.searchsorted(qs, float(v), side="right"))
        j = min(j, len(labels) - 1)
        out.append(labels[j])
    return pd.Series(out, index=x.index, dtype="string")


def qcut_label(x: pd.Series, q: int, prefix: str) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    if vals.notna().sum() < 5 or vals.nunique(dropna=True) <= 3:
        return vals.astype("string").fillna("missing")
    try:
        codes = pd.qcut(vals.rank(method="first"), q=q, labels=False, duplicates="drop")
    except Exception:
        codes = pd.Series(np.zeros(len(vals), dtype=int), index=x.index)
    return codes.map(lambda z: "missing" if pd.isna(z) else f"{prefix}{int(z) + 1}").astype(
        "string"
    )


def equal_width_label(x: pd.Series, bins: int) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    if vals.notna().sum() < 5 or vals.nunique(dropna=True) <= 3:
        return vals.astype("string").fillna("missing")
    try:
        codes = pd.cut(vals, bins=bins, labels=False, include_lowest=True)
    except Exception:
        codes = pd.Series(np.zeros(len(vals), dtype=int), index=x.index)
    return codes.map(lambda z: "missing" if pd.isna(z) else f"eq{int(z) + 1}").astype("string")


def zscore_label(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    mu = vals.mean()
    sd = vals.std(ddof=1)
    if vals.notna().sum() < 5 or not np.isfinite(sd) or sd == 0:
        return vals.astype("string").fillna("missing")
    z = (vals - mu) / sd
    labels = []
    for v in z:
        if pd.isna(v):
            labels.append("missing")
        elif v <= -1.5:
            labels.append("very_low")
        elif v <= -0.5:
            labels.append("low")
        elif v <= 0.5:
            labels.append("normal_range")
        elif v <= 1.5:
            labels.append("high")
        else:
            labels.append("very_high")
    return pd.Series(labels, index=x.index, dtype="string")


def discretize_continuous(series: pd.Series, scheme_id: str) -> pd.Series:
    if scheme_id == "quantile5_original_10_25_75_90":
        return quantile_label(
            series,
            [0.10, 0.25, 0.75, 0.90],
            ["very_low", "low", "normal_range", "high", "very_high"],
        )
    if scheme_id == "quantile5_wide_normal_05_20_80_95":
        return quantile_label(
            series,
            [0.05, 0.20, 0.80, 0.95],
            ["very_low", "low", "normal_range", "high", "very_high"],
        )
    if scheme_id == "quantile5_narrow_normal_15_30_70_85":
        return quantile_label(
            series,
            [0.15, 0.30, 0.70, 0.85],
            ["very_low", "low", "normal_range", "high", "very_high"],
        )
    if scheme_id == "quantile3_tertiles":
        return qcut_label(series, 3, "q")
    if scheme_id == "quantile4_quartiles":
        return qcut_label(series, 4, "q")
    if scheme_id == "quantile6_sextiles":
        return qcut_label(series, 6, "q")
    if scheme_id == "equal_width5":
        return equal_width_label(series, 5)
    if scheme_id == "zscore5_mean_sd":
        return zscore_label(series)
    raise ValueError(f"Unknown continuous discretization scheme: {scheme_id}")


def infer_assays(
    raw: pd.DataFrame, discrete: pd.DataFrame, catalog: Optional[pd.DataFrame]
) -> List[str]:
    if catalog is not None and "assay" in catalog.columns:
        assays = [str(a) for a in catalog["assay"].dropna().tolist() if str(a) in discrete.columns]
        if assays:
            return assays
    return [c for c in discrete.columns if c not in {"ncu_id", "gene_name", "published_cluster"}]


def infer_continuous_assays(
    raw: pd.DataFrame, assays: List[str], catalog: Optional[pd.DataFrame]
) -> List[str]:
    out = []
    if catalog is not None and {"assay", "assay_type"}.issubset(catalog.columns):
        for _, row in catalog.iterrows():
            assay = str(row["assay"])
            if assay in assays and "continuous" in str(row["assay_type"]).lower():
                out.append(assay)
    if out:
        return out
    for assay in assays:
        numeric = pd.to_numeric(raw[assay], errors="coerce")
        if (
            numeric.notna().sum() >= max(10, int(0.80 * len(raw)))
            and numeric.nunique(dropna=True) > 8
        ):
            out.append(assay)
    return out


def build_discrete_matrix(
    raw: pd.DataFrame,
    baseline: pd.DataFrame,
    assays: List[str],
    continuous_assays: List[str],
    scheme_id: str,
) -> pd.DataFrame:
    if scheme_id == "baseline_current_file":
        return baseline[["ncu_id", "gene_name"] + assays + ["published_cluster"]].copy()
    out = raw[["ncu_id", "gene_name"]].copy()
    for assay in assays:
        if assay in continuous_assays:
            out[assay] = discretize_continuous(raw[assay], scheme_id)
        else:
            out[assay] = normalize_category(raw[assay])
    out["published_cluster"] = raw["published_cluster"].astype(str)
    return out


def contingency_fast(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = np.asarray(true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if len(true) == 0:
        return np.zeros((0, 0), dtype=np.int64)
    n_true = int(true.max()) + 1
    n_pred = int(pred.max()) + 1
    return np.bincount(true * n_pred + pred, minlength=n_true * n_pred).reshape(n_true, n_pred)


def metrics_fast(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    cont = contingency_fast(true, pred).astype(float)
    n = float(cont.sum())
    if n <= 1:
        return {"nmi": 0.0, "ari": 0.0, "purity": 1.0}
    row = cont.sum(axis=1)
    col = cont.sum(axis=0)
    nz = cont > 0
    expected = np.outer(row, col)
    mi = float(np.sum((cont[nz] / n) * np.log((cont[nz] * n) / expected[nz])))
    h_true = float(-np.sum((row[row > 0] / n) * np.log(row[row > 0] / n)))
    h_pred = float(-np.sum((col[col > 0] / n) * np.log(col[col > 0] / n)))
    nmi = 2.0 * mi / (h_true + h_pred) if (h_true + h_pred) > 0 else 1.0
    purity = float(cont.max(axis=0).sum() / n) if cont.shape[1] else np.nan
    comb = lambda x: x * (x - 1.0) / 2.0
    sum_comb = float(comb(cont).sum())
    sum_rows = float(comb(row).sum())
    sum_cols = float(comb(col).sum())
    total = float(comb(n))
    if total == 0:
        ari = 0.0
    else:
        expected_index = sum_rows * sum_cols / total
        max_index = 0.5 * (sum_rows + sum_cols)
        denom = max_index - expected_index
        ari = (sum_comb - expected_index) / denom if denom != 0 else 0.0
    return {"nmi": float(nmi), "ari": float(ari), "purity": float(purity)}


class EncodedPhenome:
    def __init__(self, discrete: pd.DataFrame, assays: List[str]):
        self.discrete = discrete.reset_index(drop=True)
        self.assays = list(assays)
        self.m = len(assays)
        self.n = len(discrete)
        self.cluster_codes = pd.factorize(discrete["published_cluster"].astype(str), sort=True)[0]
        code_cols = []
        for assay in assays:
            codes, _ = pd.factorize(discrete[assay].astype(str), sort=True)
            code_cols.append(codes.astype(np.int16))
        self.codes = np.vstack(code_cols).T if code_cols else np.empty((self.n, 0), dtype=np.int16)
        if self.codes.shape[1]:
            self.bases = self.codes.max(axis=0).astype(np.int64) + 1
            self.bases[self.bases < 2] = 2
        else:
            self.bases = np.asarray([], dtype=np.int64)
        self._labels: Dict[int, np.ndarray] = {}
        self._nprofiles: Dict[int, int] = {}

    @property
    def full_mask(self) -> int:
        return (1 << self.m) - 1

    def mask_to_indices(self, mask: int) -> List[int]:
        return [i for i in range(self.m) if mask & (1 << i)]

    def mask_to_assays(self, mask: int) -> List[str]:
        return [self.assays[i] for i in self.mask_to_indices(mask)]

    def labels_for_mask(self, mask: int) -> np.ndarray:
        if mask in self._labels:
            return self._labels[mask]
        idx = self.mask_to_indices(mask)
        if not idx:
            labels = np.zeros(self.n, dtype=np.int32)
        else:
            hashed = np.zeros(self.n, dtype=np.int64)
            for j in idx:
                hashed = hashed * int(self.bases[j]) + self.codes[:, j].astype(np.int64)
            labels = np.unique(hashed, return_inverse=True)[1].astype(np.int32)
        self._labels[mask] = labels
        self._nprofiles[mask] = int(labels.max() + 1) if len(labels) else 0
        return labels

    def n_profiles(self, mask: int) -> int:
        if mask not in self._nprofiles:
            self.labels_for_mask(mask)
        return self._nprofiles[mask]


def all_nonempty_masks(m: int) -> Iterable[int]:
    return range(1, (1 << m))


def subset_metrics(enc: EncodedPhenome) -> Tuple[pd.DataFrame, Dict[str, float]]:
    full_labels = enc.labels_for_mask(enc.full_mask)
    full_profiles = enc.n_profiles(enc.full_mask)
    full_metrics = metrics_fast(enc.cluster_codes, full_labels)
    full_metrics["profiles"] = full_profiles
    rows = []
    for mask in all_nonempty_masks(enc.m):
        labels = enc.labels_for_mask(mask)
        n_profiles = int(np.unique(labels).size)
        met = metrics_fast(enc.cluster_codes, labels)
        rows.append(
            {
                "mask": mask,
                "n_assays": len(enc.mask_to_indices(mask)),
                "assay_set": ";".join(enc.mask_to_assays(mask)),
                "n_profiles": n_profiles,
                "profile_fraction_of_full": n_profiles / full_profiles if full_profiles else np.nan,
                "cluster_nmi": met["nmi"],
                "cluster_ari": met["ari"],
                "cluster_purity": met["purity"],
                "cluster_nmi_fraction_of_full": (
                    met["nmi"] / full_metrics["nmi"] if full_metrics["nmi"] > 0 else np.nan
                ),
                "cluster_purity_fraction_of_full": (
                    met["purity"] / full_metrics["purity"] if full_metrics["purity"] > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows), full_metrics


def exact_minimal(
    subsets: pd.DataFrame, full_metrics: Dict[str, float], max_sets: int = 5000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    objectives = [
        ("profile_fraction", "profile_fraction_of_full", 1.0),
        ("cluster_nmi", "cluster_nmi", full_metrics["nmi"]),
        ("cluster_purity", "cluster_purity", full_metrics["purity"]),
    ]
    rows, set_rows = [], []
    for objective, col, full_value in objectives:
        for frac in [0.80, 0.90, 0.95, 0.99, 1.00]:
            target = full_value * frac
            elig = subsets[subsets[col] >= target - 1e-12]
            if elig.empty:
                rows.append(
                    {
                        "objective": objective,
                        "target_fraction_of_full": frac,
                        "minimal_n_assays": np.nan,
                        "n_equivalent_minimal_sets": 0,
                    }
                )
                continue
            kmin = int(elig["n_assays"].min())
            at = elig[elig["n_assays"] == kmin].sort_values(
                [col, "profile_fraction_of_full"], ascending=False
            )
            rows.append(
                {
                    "objective": objective,
                    "target_fraction_of_full": frac,
                    "metric_column": col,
                    "full_metric_value": full_value,
                    "target_metric_value": target,
                    "minimal_n_assays": kmin,
                    "n_equivalent_minimal_sets": int(len(at)),
                    "example_assay_set": at.iloc[0]["assay_set"],
                    "best_metric_at_minimal_size": float(at[col].max()),
                    "best_profile_fraction_at_minimal_size": float(
                        at["profile_fraction_of_full"].max()
                    ),
                }
            )
            for rank, (_, hit) in enumerate(at.head(max_sets).iterrows(), start=1):
                set_rows.append(
                    {
                        "objective": objective,
                        "target_fraction_of_full": frac,
                        "rank_within_target": rank,
                        "mask": int(hit["mask"]),
                        "n_assays": int(hit["n_assays"]),
                        "assay_set": hit["assay_set"],
                        "metric_value": float(hit[col]),
                        "profile_fraction_of_full": float(hit["profile_fraction_of_full"]),
                        "cluster_nmi": float(hit["cluster_nmi"]),
                        "cluster_purity": float(hit["cluster_purity"]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(set_rows)


def baseline_agreement(
    baseline_enc: EncodedPhenome,
    enc: EncodedPhenome,
    scheme_id: str,
    full_metrics: Dict[str, float],
) -> Dict[str, float | str | int]:
    base_labels = baseline_enc.labels_for_mask(baseline_enc.full_mask)
    labels = enc.labels_for_mask(enc.full_mask)
    met = metrics_fast(base_labels, labels)
    return {
        "scheme_id": scheme_id,
        "baseline_profile_nmi": met["nmi"],
        "baseline_profile_ari": met["ari"],
        "n_full_profiles": enc.n_profiles(enc.full_mask),
        "full_cluster_nmi": full_metrics["nmi"],
        "full_cluster_ari": full_metrics["ari"],
        "full_cluster_purity": full_metrics["purity"],
    }


def run_discretization_sensitivity(
    raw: pd.DataFrame, baseline: pd.DataFrame, assays: List[str], continuous_assays: List[str]
) -> Dict[str, pd.DataFrame]:
    schemes = pd.DataFrame(
        [
            {
                "scheme_id": "baseline_current_file",
                "scheme_family": "current",
                "description": "Current processed matrix generated by scripts/prepare_phenotype_matrix.py",
            },
            {
                "scheme_id": "quantile5_original_10_25_75_90",
                "scheme_family": "quantile",
                "description": "Five-state quantile binning using 10/25/75/90 percentile cut points",
            },
            {
                "scheme_id": "quantile5_wide_normal_05_20_80_95",
                "scheme_family": "quantile",
                "description": "Five-state quantile binning with a wider central normal range",
            },
            {
                "scheme_id": "quantile5_narrow_normal_15_30_70_85",
                "scheme_family": "quantile",
                "description": "Five-state quantile binning with a narrower central normal range",
            },
            {
                "scheme_id": "quantile3_tertiles",
                "scheme_family": "quantile",
                "description": "Three quantile states",
            },
            {
                "scheme_id": "quantile4_quartiles",
                "scheme_family": "quantile",
                "description": "Four quantile states",
            },
            {
                "scheme_id": "quantile6_sextiles",
                "scheme_family": "quantile",
                "description": "Six quantile states",
            },
            {
                "scheme_id": "equal_width5",
                "scheme_family": "equal_width",
                "description": "Five equal-width bins over the observed range",
            },
            {
                "scheme_id": "zscore5_mean_sd",
                "scheme_family": "zscore",
                "description": "Five states using mean and standard-deviation thresholds",
            },
        ]
    )
    baseline_enc = EncodedPhenome(baseline, assays)
    summary_rows, exact_rows, set_rows, best_rows, agreement_rows = [], [], [], [], []
    for _, scheme in schemes.iterrows():
        scheme_id = str(scheme["scheme_id"])
        disc = build_discrete_matrix(raw, baseline, assays, continuous_assays, scheme_id)
        enc = EncodedPhenome(disc, assays)
        subsets, full_metrics = subset_metrics(enc)
        exact, sets = exact_minimal(subsets, full_metrics)
        for frame in [subsets, exact, sets]:
            frame.insert(0, "scheme_id", scheme_id)
        exact_rows.append(exact)
        set_rows.append(sets)
        agreement_rows.append(baseline_agreement(baseline_enc, enc, scheme_id, full_metrics))
        summary_rows.append(
            {
                "scheme_id": scheme_id,
                "n_full_profiles": enc.n_profiles(enc.full_mask),
                "full_cluster_nmi": full_metrics["nmi"],
                "full_cluster_ari": full_metrics["ari"],
                "full_cluster_purity": full_metrics["purity"],
            }
        )
        best = subsets.groupby("n_assays", as_index=False).agg(
            best_profile_fraction=("profile_fraction_of_full", "max"),
            best_cluster_nmi=("cluster_nmi", "max"),
            best_cluster_nmi_fraction=("cluster_nmi_fraction_of_full", "max"),
            best_cluster_purity_fraction=("cluster_purity_fraction_of_full", "max"),
        )
        best.insert(0, "scheme_id", scheme_id)
        best_rows.append(best)
    exact_all = pd.concat(exact_rows, ignore_index=True)
    sets_all = pd.concat(set_rows, ignore_index=True)
    best_all = pd.concat(best_rows, ignore_index=True)
    agreement = pd.DataFrame(agreement_rows)
    assay_inclusion_rows = []
    for objective, target in [
        ("cluster_nmi", 0.95),
        ("cluster_nmi", 1.0),
        ("profile_fraction", 0.90),
        ("profile_fraction", 0.95),
        ("profile_fraction", 1.0),
    ]:
        sub = sets_all[
            (sets_all["objective"] == objective)
            & np.isclose(sets_all["target_fraction_of_full"], target)
        ]
        for scheme_id, g in sub.groupby("scheme_id"):
            total_sets = max(len(g), 1)
            for assay in assays:
                present = g["assay_set"].astype(str).str.split(";").map(lambda vals: assay in vals)
                assay_inclusion_rows.append(
                    {
                        "objective": objective,
                        "target_fraction_of_full": target,
                        "scheme_id": scheme_id,
                        "assay": assay,
                        "axis": assay_axis(assay),
                        "inclusion_fraction_within_minimal_sets": (
                            float(present.mean()) if len(present) else 0.0
                        ),
                        "n_minimal_sets_for_scheme": total_sets,
                    }
                )
    inclusion = pd.DataFrame(assay_inclusion_rows)
    if not inclusion.empty:
        scheme_count = (
            inclusion.groupby(["objective", "target_fraction_of_full"])["scheme_id"]
            .nunique()
            .rename("n_schemes")
        )
        agg = inclusion.groupby(
            ["objective", "target_fraction_of_full", "assay", "axis"], as_index=False
        ).agg(
            mean_inclusion_fraction=("inclusion_fraction_within_minimal_sets", "mean"),
            schemes_with_any_inclusion=(
                "inclusion_fraction_within_minimal_sets",
                lambda x: int((x > 0).sum()),
            ),
        )
        agg = agg.merge(
            scheme_count.reset_index(), on=["objective", "target_fraction_of_full"], how="left"
        )
        agg["scheme_presence_fraction"] = agg["schemes_with_any_inclusion"] / agg[
            "n_schemes"
        ].replace(0, np.nan)
    else:
        agg = pd.DataFrame()
    return {
        "disc_schemes": schemes,
        "disc_summary": pd.DataFrame(summary_rows),
        "disc_baseline_agreement": agreement,
        "disc_exact_minimal": exact_all,
        "disc_minimal_sets": sets_all,
        "disc_best_by_k": best_all,
        "disc_assay_inclusion": inclusion,
        "disc_assay_stability": agg,
    }


def abnormality_table(discrete: pd.DataFrame, assays: List[str]) -> pd.DataFrame:
    out = discrete[["ncu_id", "gene_name", "published_cluster"]].copy()
    for assay in assays:
        out[f"abnormal__{assay}"] = (
            ~discrete[assay].astype(str).str.strip().str.lower().isin(NORMAL_VALUES)
        )
    for axis in ["growth", "asexual_development", "sexual_development"]:
        cols = [f"abnormal__{a}" for a in assays if assay_axis(a) == axis]
        if cols:
            out[f"{axis}_abnormal_any"] = out[cols].any(axis=1)
            out[f"{axis}_abnormal_count"] = out[cols].sum(axis=1).astype(int)
    ab_cols = [f"abnormal__{a}" for a in assays]
    out["total_abnormal_count"] = out[ab_cols].sum(axis=1).astype(int)
    out["any_abnormal"] = out[ab_cols].any(axis=1)
    return out


def keyword_series(annot: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    cls = (
        annot.get("gene_classification", pd.Series([""] * len(annot)))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    desc = (
        annot.get("product_description", pd.Series([""] * len(annot)))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    gene = annot.get("gene_name", pd.Series([""] * len(annot))).fillna("").astype(str).str.lower()
    return cls, desc, gene


def build_gene_class_modes(annot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cls, desc, gene = keyword_series(annot)
    tm = pd.to_numeric(
        annot.get("n_transmembrane_helices", pd.Series([0] * len(annot))),
        errors="coerce",
    ).fillna(0)
    ph = pd.to_numeric(
        annot.get("n_phosphorylation_sites", pd.Series([0] * len(annot))),
        errors="coerce",
    )
    ph_cutoff = ph.dropna().quantile(0.75) if ph.notna().any() else np.inf
    yeast = (
        canonical_bool(annot["has_yeast_ortholog"])
        if "has_yeast_ortholog" in annot.columns
        else pd.Series(False, index=annot.index)
    )
    common = {
        "yeast_ortholog": yeast,
        "no_yeast_ortholog": ~yeast,
        "transmembrane_any": tm >= 1,
        "transmembrane_5plus": tm >= 5,
        "high_phosphorylation_q75": ph.fillna(0) >= ph_cutoff,
    }
    strict = {
        "TF": cls.eq("TF"),
        "GPCR": cls.eq("GPCR"),
        "ser_thr_kinase": cls.eq("STKIN"),
        "phosphatase": cls.eq("PASE"),
    }
    default = {
        "TF": strict["TF"]
        | desc.str.contains("transcription factor|zn2cys6|c2h2|bzip|myb|homeobox|ap2", regex=True),
        "GPCR": strict["GPCR"]
        | desc.str.contains("g-protein-coupled|gpcr", regex=True)
        | gene.str.match(r"gpr-"),
        "ser_thr_kinase": strict["ser_thr_kinase"]
        | desc.str.contains("serine/threonine|protein kinase|map kinase|kinase", regex=True),
        "phosphatase": strict["phosphatase"] | desc.str.contains("phosphatase", regex=True),
    }
    broad = {
        "TF": default["TF"]
        | desc.str.contains("dna-binding|zinc finger|helix-turn-helix|regulator", regex=True),
        "GPCR": default["GPCR"]
        | ((tm >= 6) & desc.str.contains("receptor|membrane|signal", regex=True)),
        "ser_thr_kinase": default["ser_thr_kinase"]
        | desc.str.contains("phosphotransferase|kinase domain", regex=True),
        "phosphatase": default["phosphatase"]
        | desc.str.contains("pp2c|protein phosphatase|phosphoprotein phosphatase", regex=True),
    }
    frames: list[pd.DataFrame] = []
    for mode, definitions in (("strict", strict), ("default", default), ("broad", broad)):
        frame = annot[["ncu_id", "gene_name"]].copy()
        frame["mode"] = mode
        for column, values in {**common, **definitions}.items():
            frame[column] = values.fillna(False).astype(bool).to_numpy()
        frames.append(frame)
    rules = [
        {
            "mode": "strict",
            "gene_class": "TF",
            "rule_description": "Gene classification equals TF",
            "source_columns": "gene_classification",
        },
        {
            "mode": "strict",
            "gene_class": "GPCR",
            "rule_description": "Gene classification equals GPCR",
            "source_columns": "gene_classification",
        },
        {
            "mode": "strict",
            "gene_class": "ser_thr_kinase",
            "rule_description": "Gene classification equals STKIN",
            "source_columns": "gene_classification",
        },
        {
            "mode": "strict",
            "gene_class": "phosphatase",
            "rule_description": "Gene classification equals PASE",
            "source_columns": "gene_classification",
        },
        {
            "mode": "default",
            "gene_class": "TF",
            "rule_description": "Strict TF plus common transcription-factor keywords",
            "source_columns": "gene_classification;product_description",
        },
        {
            "mode": "default",
            "gene_class": "GPCR",
            "rule_description": "Strict GPCR plus GPCR keywords and gpr-* gene names",
            "source_columns": "gene_classification;product_description;gene_name",
        },
        {
            "mode": "default",
            "gene_class": "ser_thr_kinase",
            "rule_description": "Strict STKIN plus kinase-related product-description keywords",
            "source_columns": "gene_classification;product_description",
        },
        {
            "mode": "default",
            "gene_class": "phosphatase",
            "rule_description": "Strict PASE plus phosphatase product-description keywords",
            "source_columns": "gene_classification;product_description",
        },
        {
            "mode": "broad",
            "gene_class": "TF",
            "rule_description": "Default TF plus broader DNA-binding and regulator keywords",
            "source_columns": "gene_classification;product_description",
        },
        {
            "mode": "broad",
            "gene_class": "GPCR",
            "rule_description": "Default GPCR plus a receptor-like multi-pass transmembrane rule",
            "source_columns": "gene_classification;product_description;n_transmembrane_helices",
        },
        {
            "mode": "broad",
            "gene_class": "ser_thr_kinase",
            "rule_description": "Default kinase plus broader kinase-domain terms",
            "source_columns": "gene_classification;product_description",
        },
        {
            "mode": "broad",
            "gene_class": "phosphatase",
            "rule_description": "Default phosphatase plus PP2C and protein-phosphatase terms",
            "source_columns": "gene_classification;product_description",
        },
        {
            "mode": "all",
            "gene_class": "yeast_ortholog",
            "rule_description": "Yeast ortholog annotation is present",
            "source_columns": "has_yeast_ortholog",
        },
        {
            "mode": "all",
            "gene_class": "no_yeast_ortholog",
            "rule_description": "Yeast ortholog annotation is absent",
            "source_columns": "has_yeast_ortholog",
        },
        {
            "mode": "all",
            "gene_class": "transmembrane_any",
            "rule_description": "At least one predicted transmembrane helix",
            "source_columns": "n_transmembrane_helices",
        },
        {
            "mode": "all",
            "gene_class": "transmembrane_5plus",
            "rule_description": "At least five predicted transmembrane helices",
            "source_columns": "n_transmembrane_helices",
        },
        {
            "mode": "all",
            "gene_class": "high_phosphorylation_q75",
            "rule_description": "Phosphorylation-site annotation count is at or above the 75th percentile",
            "source_columns": "n_phosphorylation_sites",
        },
    ]
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rules)


def gene_class_counts(modes: pd.DataFrame, min_class_size: int) -> pd.DataFrame:
    class_cols = [c for c in modes.columns if c not in {"ncu_id", "gene_name", "mode"}]
    rows = []
    for mode, group in modes.groupby("mode"):
        for col in class_cols:
            n = int(group[col].fillna(False).astype(bool).sum())
            rows.append(
                {
                    "mode": mode,
                    "gene_class": col,
                    "n_genes": n,
                    "fraction_of_phenome": n / len(group),
                    "passes_min_class_size": n >= min_class_size,
                }
            )
    return pd.DataFrame(rows)


def gene_class_overlap(modes: pd.DataFrame) -> pd.DataFrame:
    class_cols = [c for c in modes.columns if c not in {"ncu_id", "gene_name", "mode"}]
    rows = []
    for mode, group in modes.groupby("mode"):
        for a, b in itertools.combinations(class_cols, 2):
            x = group[a].fillna(False).astype(bool).to_numpy()
            y = group[b].fillna(False).astype(bool).to_numpy()
            both = int((x & y).sum())
            union = int((x | y).sum())
            # Phi coefficient for binary overlap.
            table = np.array(
                [[both, int((x & ~y).sum())], [int((~x & y).sum()), int((~x & ~y).sum())]],
                dtype=float,
            )
            denom = math.sqrt(table.sum(axis=0).prod() * table.sum(axis=1).prod())
            phi = (
                ((table[0, 0] * table[1, 1]) - (table[0, 1] * table[1, 0])) / denom
                if denom
                else np.nan
            )
            rows.append(
                {
                    "mode": mode,
                    "class_a": a,
                    "class_b": b,
                    "n_a": int(x.sum()),
                    "n_b": int(y.sum()),
                    "n_both": both,
                    "jaccard": both / union if union else np.nan,
                    "phi": phi,
                }
            )
    return pd.DataFrame(rows)


def gene_definition_breakdown(annot: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    strict = modes[modes["mode"] == "strict"].set_index("ncu_id")
    default = modes[modes["mode"] == "default"].set_index("ncu_id")
    broad = modes[modes["mode"] == "broad"].set_index("ncu_id")
    rows = []
    for cls_name in ["TF", "GPCR", "ser_thr_kinase", "phosphatase"]:
        s = strict[cls_name].astype(bool)
        d = default[cls_name].astype(bool)
        b = broad[cls_name].astype(bool)
        rows.append(
            {
                "gene_class": cls_name,
                "strict_n": int(s.sum()),
                "default_n": int(d.sum()),
                "broad_n": int(b.sum()),
                "default_added_by_keyword": int((d & ~s).sum()),
                "broad_added_beyond_default": int((b & ~d).sum()),
                "strict_fraction_of_default": int(s.sum()) / max(int(d.sum()), 1),
                "default_fraction_of_broad": int(d.sum()) / max(int(b.sum()), 1),
            }
        )
    return pd.DataFrame(rows)


def annotation_coverage(annot: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "product_description",
        "gene_classification",
        "n_phosphorylation_sites",
        "n_transmembrane_helices",
    ]
    rows = []
    for field in fields:
        values = annot.get(field, pd.Series(np.nan, index=annot.index))
        missing = values.isna() | values.astype("string").str.strip().eq("")
        rows.append(
            {
                "annotation_field": field,
                "n_genes": len(annot),
                "n_missing": int(missing.sum()),
                "fraction_missing": float(missing.mean()) if len(annot) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def enrichment_by_mode(
    modes: pd.DataFrame, abnormal: pd.DataFrame, min_class_size: int
) -> pd.DataFrame:
    rows = []
    axes = ["growth", "asexual_development", "sexual_development"]
    class_cols = [c for c in modes.columns if c not in {"ncu_id", "gene_name", "mode"}]
    for mode, group in modes.groupby("mode"):
        data = abnormal.merge(group, on=["ncu_id", "gene_name"], how="inner")
        for gene_class in class_cols:
            mem = data[gene_class].fillna(False).astype(bool)
            if mem.sum() < min_class_size or (~mem).sum() < min_class_size:
                continue
            for axis in axes:
                col = f"{axis}_abnormal_any"
                if col not in data.columns:
                    continue
                ab = data[col].astype(bool)
                x1, x0 = int((mem & ab).sum()), int((mem & ~ab).sum())
                y1, y0 = int((~mem & ab).sum()), int((~mem & ~ab).sum())
                odds, p = safe_fisher(x1, x0, y1, y0)
                rows.append(
                    {
                        "mode": mode,
                        "gene_class": gene_class,
                        "axis": axis,
                        "n_class_genes": int(mem.sum()),
                        "class_abnormal": x1,
                        "class_normal": x0,
                        "nonclass_abnormal": y1,
                        "nonclass_normal": y0,
                        "abnormal_rate_class": x1 / max(x1 + x0, 1),
                        "abnormal_rate_nonclass": y1 / max(y1 + y0, 1),
                        "odds_ratio": odds,
                        "log2_odds_ratio_corrected": log2_or_corrected(x1, x0, y1, y0),
                        "fisher_p": p,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q"] = bh_fdr(out["fisher_p"])
    return out


def mode_stability(enrichment: pd.DataFrame) -> pd.DataFrame:
    if enrichment.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (gene_class, axis), group in enrichment.groupby(["gene_class", "axis"]):
        values = group.set_index("mode")
        signs = np.sign(values["log2_odds_ratio_corrected"].fillna(0).to_numpy())
        nonzero = signs[signs != 0]
        sign_consistent = bool(len(nonzero) > 0 and np.all(nonzero == nonzero[0]))
        significant_modes = int((values["fdr_q"] < 0.05).sum())
        rows.append(
            {
                "gene_class": gene_class,
                "axis": axis,
                "n_modes_tested": int(values.shape[0]),
                "sign_consistent_across_modes": sign_consistent,
                "n_modes_fdr_lt_0_05": significant_modes,
                "min_fdr_q": float(values["fdr_q"].min()),
                "max_fdr_q": float(values["fdr_q"].max()),
                "strict_log2_or": (
                    float(values.loc["strict", "log2_odds_ratio_corrected"])
                    if "strict" in values.index
                    else np.nan
                ),
                "default_log2_or": (
                    float(values.loc["default", "log2_odds_ratio_corrected"])
                    if "default" in values.index
                    else np.nan
                ),
                "broad_log2_or": (
                    float(values.loc["broad", "log2_odds_ratio_corrected"])
                    if "broad" in values.index
                    else np.nan
                ),
                "stability_class": (
                    "stable"
                    if sign_consistent and significant_modes >= 2
                    else "definition_sensitive"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["stability_class", "min_fdr_q", "gene_class", "axis"])


def write_workbook(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    ensure_dir(path.parent)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            frame = df if df is not None else pd.DataFrame()
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.book[name[:31]]
            ws.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                values = frame[col].head(250).fillna("").astype(str).tolist() if len(frame) else []
                width = min(max([len(str(col))] + [len(v) for v in values]) + 2, 45)
                ws.column_dimensions[ws.cell(1, i).column_letter].width = max(width, 10)
            for cell in ws[1]:
                cell.style = "Headline 4"


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    processed = root / "data" / "processed"
    tables = ensure_dir(root / "reports" / "tables")
    source = ensure_dir(root / "reports" / "source_data" / "sensitivity")

    raw = read_csv_required(
        find_file(root, [processed / "phenotype_matrix_raw.csv"], ["**/phenotype_matrix_raw.csv"]),
        "phenotype_matrix_raw.csv",
    )
    baseline = read_csv_required(
        find_file(
            root,
            [processed / "phenotype_matrix_discrete.csv"],
            ["**/phenotype_matrix_discrete.csv"],
        ),
        "phenotype_matrix_discrete.csv",
    )
    annot = read_csv_required(
        find_file(root, [processed / "gene_annotations.csv"], ["**/gene_annotations.csv"]),
        "gene_annotations.csv",
    )
    catalog_path = find_file(root, [processed / "assay_catalog.csv"], ["**/assay_catalog.csv"])
    catalog = pd.read_csv(catalog_path) if catalog_path else None
    assays = infer_assays(raw, baseline, catalog)
    continuous_assays = infer_continuous_assays(raw, assays, catalog)
    if not continuous_assays:
        raise ValueError("No continuous assays were detected")

    sensitivity = run_discretization_sensitivity(raw, baseline, assays, continuous_assays)
    abnormal = abnormality_table(baseline, assays)
    modes, rulebook = build_gene_class_modes(annot)
    counts = gene_class_counts(modes, args.min_class_size)
    overlaps = gene_class_overlap(modes)
    definition_breakdown = gene_definition_breakdown(annot, modes)
    coverage = annotation_coverage(annot)
    enrichment = enrichment_by_mode(modes, abnormal, args.min_class_size)
    stability = mode_stability(enrichment)

    summary = pd.DataFrame(
        [
            {
                "n_mutants": len(baseline),
                "n_assays": len(assays),
                "continuous_assays_tested": ";".join(continuous_assays),
                "n_discretization_schemes": int(len(sensitivity["disc_schemes"])),
                "n_gene_class_modes": int(modes["mode"].nunique()),
                "min_class_size": args.min_class_size,
                "random_seed": args.random_seed,
            }
        ]
    )
    sheets: Dict[str, pd.DataFrame] = {
        "summary_sensitivity": summary,
        **sensitivity,
        "gene_class_rulebook": rulebook,
        "gene_class_counts": counts,
        "gene_class_overlap": overlaps,
        "gene_class_definition_breakdown": definition_breakdown,
        "annotation_coverage": coverage,
        "gene_class_axis_enrich_modes": (
            enrichment.sort_values("fdr_q") if not enrichment.empty else enrichment
        ),
        "gene_class_mode_stability": stability,
        "abnormality_table": abnormal,
    }
    output = (
        Path(args.output_workbook)
        if args.output_workbook
        else tables / "phenome_sensitivity_results.xlsx"
    )
    if not output.is_absolute():
        output = root / output
    write_workbook(output, sheets)
    for name, frame in sheets.items():
        if name != "abnormality_table":
            frame.to_csv(source / f"{name}.csv", index=False)

    print(f"Sensitivity analysis complete: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
