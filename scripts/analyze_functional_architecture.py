#!/usr/bin/env python
"""Functional-architecture analyses for the Neurospora deletion phenome.

Adds three analysis layers:
  1) published-cluster recovery from reduced assay panels;
  2) gene-class enrichment across phenotype axes;
  3) Dreyfuss iJDZ836 metabolic-model integration.

Adds three exploratory concept layers:
  4) marginal-cost / marginal-information curves;
  5) exposure-count phenomics;
  6) assay-substitutability orbits among equivalent minimal panels.

All paths are resolved from --project-root. No local machine paths are hard-coded.
"""

from __future__ import annotations

import argparse
import itertools
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure, normalized_mutual_info_score

NORMAL_VALUES = {"normal", "normal_range", "normal range", "wild_type", "wild-type", "wt"}
NCU_RE = re.compile(r"NCU\d{5}", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Neurospora phenome-architecture analyses.")
    p.add_argument("--project-root", default=".", help="Project root")
    p.add_argument("--shapley-permutations", type=int, default=10000)
    p.add_argument("--bootstrap-iterations", type=int, default=1000)
    p.add_argument("--null-iterations", type=int, default=1000, help="Cluster-label-shuffle null iterations")
    p.add_argument("--assay-permutation-null-iterations", type=int, default=-1, help="Assay-column permutation null iterations; default=min(null,200)")
    p.add_argument("--random-seed", type=int, default=20260601)
    p.add_argument("--max-minimal-sets", type=int, default=5000)
    p.add_argument("--output-workbook", default="")
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_file(root: Path, preferred: Sequence[Path], patterns: Sequence[str]) -> Optional[Path]:
    for p in preferred:
        if p.exists():
            return p
    for pattern in patterns:
        hits = list(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def read_csv_required(path: Optional[Path], label: str) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Could not find {label}. Run scripts/prepare_phenotype_matrix.py first.")
    return pd.read_csv(path)


def canonical_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def assay_axis(assay: str) -> str:
    low = assay.lower()
    if "basal" in low or "aerial" in low or "hypha" in low:
        return "growth"
    if "conidia" in low:
        return "asexual_development"
    if "protoperithe" in low or "perithe" in low or "ascospore" in low:
        return "sexual_development"
    return "other"


def default_assay_cost(assay: str) -> float:
    low = assay.lower()
    if "basal" in low or "aerial" in low:
        return 1.0
    if "conidia" in low:
        return 1.5
    if "protoperithe" in low:
        return 2.0
    if "perithe" in low or "ascospore" in low:
        return 2.5
    return 1.0


def assay_columns(discrete: pd.DataFrame, catalog: Optional[pd.DataFrame]) -> List[str]:
    if catalog is not None and "assay" in catalog.columns:
        assays = [a for a in catalog["assay"].dropna().astype(str).tolist() if a in discrete.columns]
        if assays:
            return assays
    return [c for c in discrete.columns if c not in {"ncu_id", "gene_name", "published_cluster"}]


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


def contingency_fast(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = np.asarray(true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if len(true) == 0:
        return np.zeros((0, 0), dtype=np.int64)
    n_true = int(true.max()) + 1
    n_pred = int(pred.max()) + 1
    flat = true * n_pred + pred
    return np.bincount(flat, minlength=n_true * n_pred).reshape(n_true, n_pred)


def metrics_fast(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    # Fast contingency-based NMI/ARI/purity for many repeated subset evaluations.
    cont = contingency_fast(true, pred).astype(float)
    n = float(cont.sum())
    if n <= 1:
        return {"nmi": 0.0, "ari": 0.0, "purity": 1.0, "homogeneity": 0.0, "completeness": 0.0, "v_measure": 0.0}
    row = cont.sum(axis=1)
    col = cont.sum(axis=0)
    nz = cont > 0
    expected = np.outer(row, col)
    mi = float(np.sum((cont[nz] / n) * np.log((cont[nz] * n) / expected[nz])))
    h_true = float(-np.sum((row[row > 0] / n) * np.log(row[row > 0] / n)))
    h_pred = float(-np.sum((col[col > 0] / n) * np.log(col[col > 0] / n)))
    nmi = 2.0 * mi / (h_true + h_pred) if (h_true + h_pred) > 0 else 1.0
    homogeneity = mi / h_true if h_true > 0 else 1.0
    completeness = mi / h_pred if h_pred > 0 else 1.0
    v_measure = nmi
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
    return {"nmi": float(nmi), "ari": float(ari), "purity": float(purity), "homogeneity": float(homogeneity), "completeness": float(completeness), "v_measure": float(v_measure)}


class EncodedPhenome:
    def __init__(self, discrete: pd.DataFrame, assays: List[str]):
        self.discrete = discrete
        self.assays = assays
        self.m = len(assays)
        self.n = len(discrete)
        self.cluster_codes = pd.factorize(discrete["published_cluster"].astype(str), sort=True)[0]
        code_cols = []
        self.value_maps = {}
        for a in assays:
            codes, uniques = pd.factorize(discrete[a].astype(str), sort=True)
            code_cols.append(codes.astype(np.int16))
            self.value_maps[a] = list(uniques)
        self.codes = np.vstack(code_cols).T
        self._label_cache: Dict[int, np.ndarray] = {}
        self._nprofile_cache: Dict[int, int] = {}

    def mask_to_assays(self, mask: int) -> List[str]:
        return [self.assays[i] for i in range(self.m) if mask & (1 << i)]

    def mask_to_indices(self, mask: int) -> List[int]:
        return [i for i in range(self.m) if mask & (1 << i)]

    def labels_for_mask(self, mask: int, codes: Optional[np.ndarray] = None) -> np.ndarray:
        if codes is None and mask in self._label_cache:
            return self._label_cache[mask]
        idx = self.mask_to_indices(mask)
        if len(idx) == 0:
            labels = np.zeros(self.n if codes is None else codes.shape[0], dtype=np.int32)
        else:
            arr = self.codes[:, idx] if codes is None else codes[:, idx]
            labels = np.unique(arr, axis=0, return_inverse=True)[1].astype(np.int32)
        if codes is None:
            self._label_cache[mask] = labels
            self._nprofile_cache[mask] = int(labels.max() + 1) if len(labels) else 0
        return labels

    def n_profiles(self, mask: int) -> int:
        if mask not in self._nprofile_cache:
            self.labels_for_mask(mask)
        return self._nprofile_cache[mask]

    @property
    def full_mask(self) -> int:
        return (1 << self.m) - 1


def all_nonempty_masks(m: int) -> Iterable[int]:
    return range(1, (1 << m))


def build_subset_metrics(enc: EncodedPhenome) -> Tuple[pd.DataFrame, Dict[str, float]]:
    full_labels = enc.labels_for_mask(enc.full_mask)
    full_profiles = enc.n_profiles(enc.full_mask)
    full_metrics = metrics_fast(enc.cluster_codes, full_labels)
    full_metrics["n_profiles"] = full_profiles
    rows = []
    for mask in all_nonempty_masks(enc.m):
        labels = enc.labels_for_mask(mask)
        nprof = int(labels.max() + 1)
        met = metrics_fast(enc.cluster_codes, labels)
        assays = enc.mask_to_assays(mask)
        rows.append({
            "mask": mask,
            "n_assays": len(assays),
            "assay_set": ";".join(assays),
            "n_profiles": nprof,
            "profile_fraction_of_full": nprof / full_profiles,
            **{f"cluster_{k}": v for k, v in met.items()},
            "cluster_nmi_fraction_of_full": met["nmi"] / full_metrics["nmi"] if full_metrics["nmi"] > 0 else np.nan,
            "cluster_ari_fraction_of_full": met["ari"] / full_metrics["ari"] if full_metrics["ari"] > 0 else np.nan,
            "cluster_purity_fraction_of_full": met["purity"] / full_metrics["purity"] if full_metrics["purity"] > 0 else np.nan,
            "cluster_v_fraction_of_full": met["v_measure"] / full_metrics["v_measure"] if full_metrics["v_measure"] > 0 else np.nan,
            "total_unit_cost": float(len(assays)),
            "total_default_labor_cost": float(sum(default_assay_cost(a) for a in assays)),
        })
    return pd.DataFrame(rows), full_metrics


def summarize_exact_minimal(subsets: pd.DataFrame, full_metrics: Dict[str, float], max_sets: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    objectives = [
        ("profile_fraction", "profile_fraction_of_full", 1.0),
        ("cluster_nmi", "cluster_nmi", full_metrics["nmi"]),
        ("cluster_ari", "cluster_ari", full_metrics["ari"]),
        ("cluster_purity", "cluster_purity", full_metrics["purity"]),
        ("cluster_v_measure", "cluster_v_measure", full_metrics["v_measure"]),
    ]
    rows, set_rows = [], []
    for obj, col, full in objectives:
        for frac in [0.80, 0.90, 0.95, 0.99, 1.00]:
            target = full * frac
            elig = subsets[subsets[col] >= target - 1e-12]
            if elig.empty:
                rows.append({"objective": obj, "target_fraction_of_full": frac, "metric_column": col, "full_metric_value": full,
                             "target_metric_value": target, "minimal_n_assays": np.nan, "n_equivalent_minimal_sets": 0,
                             "example_assay_set": "", "best_metric_at_minimal_size": np.nan,
                             "best_profile_fraction_at_minimal_size": np.nan, "min_default_labor_cost_at_target": np.nan,
                             "min_unit_cost_at_target": np.nan})
                continue
            kmin = int(elig["n_assays"].min())
            at = elig[elig["n_assays"] == kmin].sort_values([col, "profile_fraction_of_full"], ascending=False)
            rows.append({"objective": obj, "target_fraction_of_full": frac, "metric_column": col, "full_metric_value": full,
                         "target_metric_value": target, "minimal_n_assays": kmin, "n_equivalent_minimal_sets": int(len(at)),
                         "example_assay_set": at.iloc[0]["assay_set"], "best_metric_at_minimal_size": float(at[col].max()),
                         "best_profile_fraction_at_minimal_size": float(at["profile_fraction_of_full"].max()),
                         "min_default_labor_cost_at_target": float(elig["total_default_labor_cost"].min()),
                         "min_unit_cost_at_target": float(elig["total_unit_cost"].min())})
            for rank, (_, hit) in enumerate(at.head(max_sets).iterrows(), 1):
                set_rows.append({"objective": obj, "target_fraction_of_full": frac, "rank_within_target": rank,
                                 "mask": int(hit["mask"]), "n_assays": int(hit["n_assays"]), "assay_set": hit["assay_set"],
                                 "metric_value": float(hit[col]), "profile_fraction_of_full": float(hit["profile_fraction_of_full"]),
                                 "cluster_nmi": float(hit["cluster_nmi"]), "cluster_purity": float(hit["cluster_purity"]),
                                 "total_default_labor_cost": float(hit["total_default_labor_cost"])})
    return pd.DataFrame(rows), pd.DataFrame(set_rows)


def greedy_curve(enc: EncodedPhenome, objective: str) -> pd.DataFrame:
    selected_mask = 0
    remaining = set(range(enc.m))
    full_profiles = enc.n_profiles(enc.full_mask)
    full_nmi = metrics_fast(enc.cluster_codes, enc.labels_for_mask(enc.full_mask))["nmi"]
    prev_profiles, prev_nmi, prev_cost = 1, 0.0, 0.0
    rows = []
    for step in range(1, enc.m + 1):
        best = None
        for i in remaining:
            mask = selected_mask | (1 << i)
            labels = enc.labels_for_mask(mask)
            nprof = int(labels.max() + 1)
            nmi = metrics_fast(enc.cluster_codes, labels)["nmi"]
            if objective == "cluster_nmi":
                score = (nmi, nprof)
            elif objective == "cost_aware_profiles":
                score = ((nprof - prev_profiles) / default_assay_cost(enc.assays[i]), nprof, nmi)
            else:
                score = (nprof, nmi)
            if best is None or score > best[0]:
                best = (score, i, nprof, nmi)
        _, i, nprof, nmi = best
        selected_mask |= (1 << i)
        remaining.remove(i)
        selected = enc.mask_to_assays(selected_mask)
        cost = sum(default_assay_cost(a) for a in selected)
        mcost = cost - prev_cost
        rows.append({"objective": objective, "step": step, "added_assay": enc.assays[i], "mask": selected_mask,
                     "selected_assays": ";".join(selected), "n_profiles": nprof, "profile_fraction_of_full": nprof / full_profiles,
                     "cluster_nmi": nmi, "cluster_nmi_fraction_of_full": nmi / full_nmi if full_nmi > 0 else np.nan,
                     "cumulative_default_labor_cost": cost, "marginal_default_labor_cost": mcost,
                     "marginal_profile_gain": nprof - prev_profiles, "marginal_cluster_nmi_gain": nmi - prev_nmi,
                     "profiles_per_default_labor_cost": (nprof - prev_profiles) / mcost if mcost > 0 else np.nan,
                     "nmi_gain_per_default_labor_cost": (nmi - prev_nmi) / mcost if mcost > 0 else np.nan})
        prev_profiles, prev_nmi, prev_cost = nprof, nmi, cost
    return pd.DataFrame(rows)


def shapley_cluster(enc: EncodedPhenome, permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    gains = {a: [] for a in enc.assays}
    for _ in range(permutations):
        order = list(rng.permutation(enc.m))
        mask, prev = 0, 0.0
        for i in order:
            mask |= (1 << int(i))
            val = metrics_fast(enc.cluster_codes, enc.labels_for_mask(mask))["nmi"]
            gains[enc.assays[int(i)]].append(val - prev)
            prev = val
    rows = []
    for assay, vals in gains.items():
        arr = np.asarray(vals)
        rows.append({"assay": assay, "axis": assay_axis(assay), "mean_marginal_cluster_nmi_gain": arr.mean(),
                     "sd_marginal_cluster_nmi_gain": arr.std(ddof=1) if len(arr) > 1 else 0.0,
                     "median_marginal_cluster_nmi_gain": np.median(arr), "permutations": permutations})
    return pd.DataFrame(rows).sort_values("mean_marginal_cluster_nmi_gain", ascending=False)


def bootstrap_reduced(enc: EncodedPhenome, exact_summary: pd.DataFrame, subsets: pd.DataFrame, iterations: int, rng: np.random.Generator) -> pd.DataFrame:
    candidates = []
    for obj, frac in [("profile_fraction", 0.80), ("profile_fraction", 0.90), ("cluster_nmi", 0.95), ("cluster_nmi", 0.99)]:
        hit = exact_summary[(exact_summary["objective"] == obj) & np.isclose(exact_summary["target_fraction_of_full"], frac)]
        if not hit.empty and hit.iloc[0]["example_assay_set"]:
            # recover mask from subset table
            mask = int(subsets.loc[subsets["assay_set"] == hit.iloc[0]["example_assay_set"], "mask"].iloc[0])
            candidates.append((f"{obj}_{frac:g}", mask))
    for k in [6, 8]:
        at = subsets[subsets["n_assays"] == k].sort_values("cluster_nmi", ascending=False)
        if not at.empty:
            candidates.append((f"best_cluster_nmi_k{k}", int(at.iloc[0]["mask"])))
    # de-duplicate
    seen, unique = set(), []
    for name, mask in candidates:
        key = (name, mask)
        if key not in seen:
            seen.add(key); unique.append((name, mask))
    candidates = unique
    all_labels = {mask: enc.labels_for_mask(mask) for _, mask in candidates}
    full_labels = enc.labels_for_mask(enc.full_mask)
    rows = []
    for i in range(iterations):
        idx = rng.integers(0, enc.n, size=enc.n)
        y = enc.cluster_codes[idx]
        full_nmi = metrics_fast(y, full_labels[idx])["nmi"]
        for name, mask in candidates:
            lab = all_labels[mask][idx]
            nmi = metrics_fast(y, lab)["nmi"]
            rows.append({"iteration": i + 1, "subset_name": name, "mask": mask, "n_assays": len(enc.mask_to_assays(mask)),
                         "assay_set": ";".join(enc.mask_to_assays(mask)), "bootstrap_full_cluster_nmi": full_nmi,
                         "bootstrap_subset_cluster_nmi": nmi, "subset_fraction_of_full_nmi": nmi / full_nmi if full_nmi > 0 else np.nan,
                         "bootstrap_subset_profiles": int(np.unique(lab).size)})
    return pd.DataFrame(rows)


def null_cluster(enc: EncodedPhenome, subsets: pd.DataFrame, label_null_iters: int, colperm_iters: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Targeted null model for cluster recovery.

    This evaluates biologically relevant panel candidates rather than maximizing
    across all 1,023 subsets at every null iteration. That keeps the targeted null analysis computationally efficient
    and gives a direct answer: does the full panel, greedy sequence, or exact
    reduced candidate preserve published-cluster structure better than random
    label or assay-column shuffles?
    """
    # Candidate masks: full panel, best subset at each size by cluster NMI, and
    # best subset at each size by profile recovery.
    candidate_rows = []
    full = subsets[subsets["mask"] == enc.full_mask].iloc[0]
    candidate_rows.append(("full_panel", int(full["mask"])))
    for k, g in subsets.groupby("n_assays"):
        best_nmi = g.sort_values("cluster_nmi", ascending=False).iloc[0]
        best_prof = g.sort_values(["profile_fraction_of_full", "cluster_nmi"], ascending=False).iloc[0]
        candidate_rows.append((f"best_cluster_nmi_k{int(k)}", int(best_nmi["mask"])))
        candidate_rows.append((f"best_profile_k{int(k)}", int(best_prof["mask"])))
    # Deduplicate masks while retaining a readable first name.
    seen = {}
    for name, mask in candidate_rows:
        if mask not in seen:
            seen[mask] = name
    candidates = [(name, mask) for mask, name in seen.items()]

    observed = {}
    labels = {}
    for name, mask in candidates:
        lab = enc.labels_for_mask(mask)
        labels[mask] = lab
        observed[mask] = metrics_fast(enc.cluster_codes, lab)["nmi"]

    rows = []
    for it in range(label_null_iters):
        y = rng.permutation(enc.cluster_codes)
        for name, mask in candidates:
            val = metrics_fast(y, labels[mask])["nmi"]
            rows.append({"iteration": it + 1, "null_type": "cluster_label_shuffle", "candidate_name": name,
                         "mask": mask, "n_assays": len(enc.mask_to_assays(mask)), "assay_set": ";".join(enc.mask_to_assays(mask)),
                         "observed_cluster_nmi": observed[mask], "null_cluster_nmi": val})
    for it in range(colperm_iters):
        codes = enc.codes.copy()
        for j in range(enc.m):
            codes[:, j] = rng.permutation(codes[:, j])
        for name, mask in candidates:
            lab = enc.labels_for_mask(mask, codes=codes)
            val = metrics_fast(enc.cluster_codes, lab)["nmi"]
            rows.append({"iteration": it + 1, "null_type": "assay_column_permutation", "candidate_name": name,
                         "mask": mask, "n_assays": len(enc.mask_to_assays(mask)), "assay_set": ";".join(enc.mask_to_assays(mask)),
                         "observed_cluster_nmi": observed[mask], "null_cluster_nmi": val})
    nulls = pd.DataFrame(rows)
    summary_rows = []
    for (ntype, name, mask), g in nulls.groupby(["null_type", "candidate_name", "mask"]):
        vals = g["null_cluster_nmi"].to_numpy(dtype=float)
        obs = float(g["observed_cluster_nmi"].iloc[0])
        summary_rows.append({"null_type": ntype, "candidate_name": name, "mask": int(mask),
                             "n_assays": int(g["n_assays"].iloc[0]), "assay_set": g["assay_set"].iloc[0],
                             "observed_cluster_nmi": obs, "null_mean_cluster_nmi": vals.mean(),
                             "null_sd_cluster_nmi": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                             "empirical_p_ge_observed": (np.sum(vals >= obs) + 1) / (len(vals) + 1),
                             "null_iterations": len(vals)})
    return pd.DataFrame(summary_rows), nulls


def parse_sbml_model(root: Path) -> Tuple[pd.DataFrame, set, Optional[Path]]:
    path = find_file(root, [root/"data/raw/dreyfuss_2013/pcbi.1003126.s001.xml"], ["data/raw/dreyfuss_2013/*.xml", "**/pcbi.1003126.s001.xml"])
    if path is None:
        return pd.DataFrame(columns=["ncu_id", "sbml_reaction_count", "sbml_reaction_examples"]), set(), None
    text = path.read_text(encoding="utf-8", errors="ignore")
    genes = {m.group(0).upper() for m in NCU_RE.finditer(text)}
    counts, examples = defaultdict(int), defaultdict(list)
    rxn_pat = re.compile(r"<[^>]*reaction\b.*?</[^>]*reaction>", re.I | re.S)
    id_pat = re.compile(r"\bid=['\"]([^'\"]+)['\"]")
    name_pat = re.compile(r"\bname=['\"]([^'\"]+)['\"]")
    for rxn in rxn_pat.finditer(text):
        block = rxn.group(0)
        rgenes = {m.group(0).upper() for m in NCU_RE.finditer(block)}
        if not rgenes: continue
        rid = id_pat.search(block); rname = name_pat.search(block)
        label = rid.group(1) if rid else "reaction"
        if rname: label = f"{label}:{rname.group(1)}"
        for g in rgenes:
            counts[g] += 1
            if len(examples[g]) < 5: examples[g].append(label)
    df = pd.DataFrame([{"ncu_id": g, "sbml_reaction_count": int(counts.get(g,0)), "sbml_reaction_examples": ";".join(examples.get(g, []))} for g in sorted(genes)])
    return df, genes, path


def parse_dreyfuss_tables(root: Path) -> pd.DataFrame:
    raw = root/"data/raw/dreyfuss_2013"
    rows = []
    if raw.exists():
        for path in raw.glob("pcbi.1003126.s00*.xls*"):
            try: xl = pd.ExcelFile(path)
            except Exception: continue
            for sheet in xl.sheet_names:
                try: df = pd.read_excel(path, sheet_name=sheet, dtype=str)
                except Exception: continue
                text = "\n".join(df.fillna("").astype(str).values.ravel().tolist())
                for g in sorted({m.group(0).upper() for m in NCU_RE.finditer(text)}):
                    rows.append({"ncu_id": g, "source_file": path.name, "source_sheet": sheet})
    if not rows:
        return pd.DataFrame(columns=["ncu_id", "mentioned_in_dreyfuss_supp_tables", "dreyfuss_supplementary_annotations"])
    tmp = pd.DataFrame(rows)
    return tmp.groupby("ncu_id", as_index=False).agg(mentioned_in_dreyfuss_supp_tables=("source_file", lambda x: True), dreyfuss_supplementary_annotations=("source_file", lambda x: ";".join(sorted(set(x)))))


def build_gene_classes(annot: pd.DataFrame, sbml_genes: set) -> pd.DataFrame:
    out = annot[["ncu_id"]].copy()
    out["gene_name"] = annot["gene_name"] if "gene_name" in annot else ""
    cls = annot.get("gene_classification", pd.Series([""]*len(annot))).fillna("").astype(str).str.upper()
    desc = annot.get("product_description", pd.Series([""]*len(annot))).fillna("").astype(str).str.lower()
    gname = annot.get("gene_name", pd.Series([""]*len(annot))).fillna("").astype(str).str.lower()
    out["metabolic_carrillo_sheet"] = canonical_bool(annot["is_metabolic_gene_carrillo_sheet"]) if "is_metabolic_gene_carrillo_sheet" in annot else False
    out["metabolic_iJDZ836_sbml"] = annot["ncu_id"].astype(str).str.upper().isin(sbml_genes)
    out["metabolic_any"] = out["metabolic_carrillo_sheet"] | out["metabolic_iJDZ836_sbml"]
    out["yeast_ortholog"] = canonical_bool(annot["has_yeast_ortholog"]) if "has_yeast_ortholog" in annot else False
    out["no_yeast_ortholog"] = ~out["yeast_ortholog"]
    out["TF"] = cls.eq("TF") | desc.str.contains("transcription factor|zn2cys6|c2h2|bzip|myb|homeobox|ap2", regex=True)
    out["GPCR"] = cls.eq("GPCR") | desc.str.contains("g-protein-coupled|gpcr", regex=True) | gname.str.match(r"gpr-")
    out["ser_thr_kinase"] = cls.eq("STKIN") | desc.str.contains("serine/threonine|protein kinase|map kinase|kinase", regex=True)
    out["phosphatase"] = cls.eq("PASE") | desc.str.contains("phosphatase", regex=True)
    tm = pd.to_numeric(annot.get("n_transmembrane_helices", pd.Series([0]*len(annot))), errors="coerce").fillna(0)
    ph = pd.to_numeric(annot.get("n_phosphorylation_sites", pd.Series([0]*len(annot))), errors="coerce")
    out["transmembrane_any"] = tm >= 1
    out["transmembrane_5plus"] = tm >= 5
    out["high_phosphorylation_q75"] = ph.fillna(0) >= (ph.dropna().quantile(0.75) if ph.notna().any() else np.inf)
    return out


def abnormality_table(discrete: pd.DataFrame, assays: List[str]) -> pd.DataFrame:
    out = discrete[["ncu_id", "gene_name", "published_cluster"]].copy()
    for a in assays:
        out[f"abnormal__{a}"] = ~discrete[a].astype(str).str.strip().str.lower().isin(NORMAL_VALUES)
    for axis in ["growth", "asexual_development", "sexual_development"]:
        cols = [f"abnormal__{a}" for a in assays if assay_axis(a) == axis]
        if cols:
            out[f"{axis}_abnormal_any"] = out[cols].any(axis=1)
            out[f"{axis}_abnormal_count"] = out[cols].sum(axis=1).astype(int)
    allcols = [f"abnormal__{a}" for a in assays]
    out["total_abnormal_count"] = out[allcols].sum(axis=1).astype(int)
    out["any_abnormal"] = out[allcols].any(axis=1)
    return out


def enrichment_tables(discrete: pd.DataFrame, annot: pd.DataFrame, gene_classes: pd.DataFrame, abnormal: pd.DataFrame, assays: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = abnormal.merge(gene_classes, on=["ncu_id", "gene_name"], how="left")
    class_cols = [c for c in gene_classes.columns if c not in {"ncu_id", "gene_name"} and data[c].fillna(False).astype(bool).sum() >= 5]
    assay_rows, axis_rows, cluster_rows = [], [], []
    for gc in class_cols:
        mem = data[gc].fillna(False).astype(bool)
        for a in assays:
            ab = data[f"abnormal__{a}"].astype(bool)
            x1, x0 = int((mem&ab).sum()), int((mem&~ab).sum())
            y1, y0 = int((~mem&ab).sum()), int((~mem&~ab).sum())
            odds, p = safe_fisher(x1,x0,y1,y0)
            assay_rows.append({"gene_class": gc, "assay": a, "axis": assay_axis(a), "n_class_genes": int(mem.sum()), "class_abnormal": x1, "class_normal": x0,
                               "nonclass_abnormal": y1, "nonclass_normal": y0, "abnormal_rate_class": x1/max(x1+x0,1), "abnormal_rate_nonclass": y1/max(y1+y0,1),
                               "odds_ratio": odds, "log2_odds_ratio_corrected": log2_or_corrected(x1,x0,y1,y0), "fisher_p": p})
        for axis in ["growth", "asexual_development", "sexual_development"]:
            col = f"{axis}_abnormal_any"
            if col not in data: continue
            ab = data[col].astype(bool)
            x1, x0 = int((mem&ab).sum()), int((mem&~ab).sum())
            y1, y0 = int((~mem&ab).sum()), int((~mem&~ab).sum())
            odds, p = safe_fisher(x1,x0,y1,y0)
            axis_rows.append({"gene_class": gc, "axis": axis, "n_class_genes": int(mem.sum()), "class_abnormal": x1, "class_normal": x0,
                              "nonclass_abnormal": y1, "nonclass_normal": y0, "abnormal_rate_class": x1/max(x1+x0,1), "abnormal_rate_nonclass": y1/max(y1+y0,1),
                              "odds_ratio": odds, "log2_odds_ratio_corrected": log2_or_corrected(x1,x0,y1,y0), "fisher_p": p})
        for clus in sorted(data["published_cluster"].unique()):
            inc = data["published_cluster"].eq(clus)
            x1, x0 = int((mem&inc).sum()), int((mem&~inc).sum())
            y1, y0 = int((~mem&inc).sum()), int((~mem&~inc).sum())
            odds, p = safe_fisher(x1,x0,y1,y0)
            cluster_rows.append({"gene_class": gc, "published_cluster": clus, "n_cluster_genes": int(inc.sum()), "n_class_genes": int(mem.sum()),
                                 "class_in_cluster": x1, "class_outside_cluster": x0, "nonclass_in_cluster": y1, "nonclass_outside_cluster": y0,
                                 "class_rate_in_cluster": x1/max(x1+y1,1), "class_rate_outside_cluster": x0/max(x0+y0,1),
                                 "odds_ratio": odds, "log2_odds_ratio_corrected": log2_or_corrected(x1,x0,y1,y0), "fisher_p": p})
    assay_enrich, axis_enrich, cluster_enrich = pd.DataFrame(assay_rows), pd.DataFrame(axis_rows), pd.DataFrame(cluster_rows)
    for df in [assay_enrich, axis_enrich, cluster_enrich]:
        if not df.empty: df["fdr_q"] = bh_fdr(df["fisher_p"])
    exposure = data[["ncu_id","gene_name","published_cluster","growth_abnormal_count","asexual_development_abnormal_count","sexual_development_abnormal_count","total_abnormal_count","any_abnormal"] + class_cols].copy()
    exposure["axis_exposure_signature"] = exposure["growth_abnormal_count"].astype(str)+"|"+exposure["asexual_development_abnormal_count"].astype(str)+"|"+exposure["sexual_development_abnormal_count"].astype(str)
    exposure["total_exposure_label"] = exposure["total_abnormal_count"].astype(str)
    exp_rows = []
    for col in ["total_exposure_label", "axis_exposure_signature"]:
        codes = pd.factorize(exposure[col].astype(str))[0]
        met = metrics_fast(pd.factorize(exposure["published_cluster"].astype(str))[0], codes)
        exp_rows.append({"table": "exposure_cluster_metrics", "exposure_model": col, "n_exposure_states": int(exposure[col].nunique()), **{f"cluster_{k}": v for k,v in met.items()}})
    for gc in class_cols:
        mem = exposure[gc].fillna(False).astype(bool)
        if mem.sum() < 5 or (~mem).sum() < 5: continue
        try: u, p = stats.mannwhitneyu(exposure.loc[mem,"total_abnormal_count"], exposure.loc[~mem,"total_abnormal_count"])
        except Exception: u,p = np.nan,np.nan
        exp_rows.append({"table": "exposure_class_stats", "gene_class": gc, "n_class_genes": int(mem.sum()), "mean_total_abnormal_count_class": exposure.loc[mem,"total_abnormal_count"].mean(),
                         "mean_total_abnormal_count_nonclass": exposure.loc[~mem,"total_abnormal_count"].mean(), "median_total_abnormal_count_class": exposure.loc[mem,"total_abnormal_count"].median(),
                         "median_total_abnormal_count_nonclass": exposure.loc[~mem,"total_abnormal_count"].median(), "mannwhitney_u": u, "mannwhitney_p": p})
    exposure_summary = pd.DataFrame(exp_rows)
    if "mannwhitney_p" in exposure_summary:
        mask = exposure_summary["mannwhitney_p"].notna()
        exposure_summary.loc[mask, "fdr_q"] = bh_fdr(exposure_summary.loc[mask, "mannwhitney_p"])
    return assay_enrich, axis_enrich, cluster_enrich, exposure, exposure_summary


def class_specific_compression(enc: EncodedPhenome, gene_classes: pd.DataFrame) -> pd.DataFrame:
    data = enc.discrete[["ncu_id", "gene_name"]].merge(gene_classes, on=["ncu_id","gene_name"], how="left")
    rows = []
    for gc in [c for c in gene_classes.columns if c not in {"ncu_id", "gene_name"}]:
        row_idx = np.flatnonzero(data[gc].fillna(False).astype(bool).to_numpy())
        if len(row_idx) < 15: continue
        full_lab = enc.labels_for_mask(enc.full_mask)[row_idx]
        full_prof = int(np.unique(full_lab).size)
        if full_prof <= 1: continue
        tmp = []
        for mask in all_nonempty_masks(enc.m):
            lab = enc.labels_for_mask(mask)[row_idx]
            tmp.append((mask, int(np.unique(lab).size)))
        for frac in [0.80, 0.90, 1.00]:
            elig = [(mask,npf) for mask,npf in tmp if npf/full_prof >= frac - 1e-12]
            if not elig: continue
            kmin = min(len(enc.mask_to_assays(mask)) for mask,_ in elig)
            best = sorted([(mask,npf) for mask,npf in elig if len(enc.mask_to_assays(mask))==kmin], key=lambda x:x[1], reverse=True)[0]
            rows.append({"gene_class": gc, "n_class_genes": len(row_idx), "full_profiles_within_class": full_prof, "target_profile_fraction": frac,
                         "minimal_n_assays": kmin, "example_assay_set": ";".join(enc.mask_to_assays(best[0])), "profiles_at_minimal_size": best[1],
                         "profile_fraction_at_minimal_size": best[1]/full_prof})
    return pd.DataFrame(rows)


def metabolic_layer(annot: pd.DataFrame, abnormal: pd.DataFrame, sbml_df: pd.DataFrame, mentions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    met = annot[["ncu_id", "gene_name"]].copy()
    met = met.merge(sbml_df, on="ncu_id", how="left")
    met["in_iJDZ836_sbml"] = met["sbml_reaction_count"].notna()
    met["sbml_reaction_count"] = met["sbml_reaction_count"].fillna(0).astype(int)
    met["sbml_reaction_examples"] = met["sbml_reaction_examples"].fillna("")
    met["in_carrillo_metabolic_sheet"] = canonical_bool(annot["is_metabolic_gene_carrillo_sheet"]) if "is_metabolic_gene_carrillo_sheet" in annot else False
    met = met.merge(mentions, on="ncu_id", how="left")
    met["mentioned_in_dreyfuss_supp_tables"] = met["mentioned_in_dreyfuss_supp_tables"].fillna(False).astype(bool)
    met["dreyfuss_supplementary_annotations"] = met["dreyfuss_supplementary_annotations"].fillna("")
    met["metabolic_any"] = met["in_iJDZ836_sbml"] | met["in_carrillo_metabolic_sheet"] | met["mentioned_in_dreyfuss_supp_tables"]
    data = abnormal.merge(met, on=["ncu_id", "gene_name"], how="left")
    rows = []
    for definition in ["in_iJDZ836_sbml", "in_carrillo_metabolic_sheet", "mentioned_in_dreyfuss_supp_tables", "metabolic_any"]:
        mem = data[definition].fillna(False).astype(bool)
        for axis in ["growth", "asexual_development", "sexual_development"]:
            col = f"{axis}_abnormal_any"
            if col not in data: continue
            ab = data[col].astype(bool)
            x1,x0 = int((mem&ab).sum()), int((mem&~ab).sum())
            y1,y0 = int((~mem&ab).sum()), int((~mem&~ab).sum())
            odds,p = safe_fisher(x1,x0,y1,y0)
            rows.append({"metabolic_definition": definition, "axis": axis, "n_metabolic_genes": int(mem.sum()), "metabolic_abnormal": x1, "metabolic_normal": x0,
                         "nonmetabolic_abnormal": y1, "nonmetabolic_normal": y0, "abnormal_rate_metabolic": x1/max(x1+x0,1), "abnormal_rate_nonmetabolic": y1/max(y1+y0,1),
                         "odds_ratio": odds, "log2_odds_ratio_corrected": log2_or_corrected(x1,x0,y1,y0), "fisher_p": p})
    axis = pd.DataFrame(rows)
    if not axis.empty: axis["fdr_q"] = bh_fdr(axis["fisher_p"])
    bias_rows = []
    for definition, sub in axis.groupby("metabolic_definition"):
        vals = dict(zip(sub["axis"], sub["log2_odds_ratio_corrected"]))
        growth = vals.get("growth", np.nan)
        dev = np.nanmean([vals.get("asexual_development", np.nan), vals.get("sexual_development", np.nan)])
        bias_rows.append({"table": "beadle_growth_bias", "metabolic_definition": definition, "growth_log2_or": growth,
                          "mean_development_log2_or": dev, "beadle_growth_bias_score": growth-dev if not pd.isna(growth) and not pd.isna(dev) else np.nan,
                          "interpretation": "positive means metabolic membership is more growth-axis biased than development-axis biased"})
    model = data[data["in_iJDZ836_sbml"].fillna(False)].copy()
    if len(model) >= 5 and model["sbml_reaction_count"].nunique() > 1:
        for col in ["growth_abnormal_count", "asexual_development_abnormal_count", "sexual_development_abnormal_count", "total_abnormal_count"]:
            if col not in model: continue
            rho,p = stats.spearmanr(model["sbml_reaction_count"], model[col])
            bias_rows.append({"table": "reaction_count_correlations", "x": "sbml_reaction_count", "y": col, "n_model_genes": len(model), "spearman_rho": rho, "spearman_p": p})
    return met, axis, pd.DataFrame(bias_rows)


def substitutability_orbits(exact_sets: pd.DataFrame, assays: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if exact_sets.empty: return pd.DataFrame(), pd.DataFrame()
    edge_rows, orbit_rows = [], []
    for (obj, target), group in exact_sets.groupby(["objective", "target_fraction_of_full"]):
        sets = [frozenset(str(x).split(";")) for x in group["assay_set"].dropna().astype(str)]
        edges = set()
        for s1,s2 in itertools.combinations(sets,2):
            diff = s1.symmetric_difference(s2)
            if len(diff) == 2:
                edges.add(tuple(sorted(diff)))
        for a,b in sorted(edges): edge_rows.append({"objective": obj, "target_fraction_of_full": target, "assay_a": a, "assay_b": b})
        parent = {a:a for a in assays}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x,y):
            rx,ry = find(x),find(y)
            if rx != ry: parent[ry] = rx
        for a,b in edges: union(a,b)
        comps = defaultdict(list)
        for a in assays: comps[find(a)].append(a)
        oid = 0
        for comp in comps.values():
            if len(comp)>1:
                oid += 1
                orbit_rows.append({"objective": obj, "target_fraction_of_full": target, "orbit_id": oid, "orbit_size": len(comp), "assay_orbit": ";".join(sorted(comp)),
                                   "note": "assays connected by one-for-one swaps among equivalent minimal sets"})
    return pd.DataFrame(edge_rows), pd.DataFrame(orbit_rows)


def write_workbook(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    ensure_dir(path.parent)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df = df if df is not None else pd.DataFrame()
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.book[name[:31]]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, 1):
                vals = df[col].head(200).fillna("").astype(str).tolist() if len(df) else []
                width = min(max([len(str(col))] + [len(v) for v in vals]) + 2, 45)
                ws.column_dimensions[ws.cell(1, i).column_letter].width = max(width, 10)
            for c in ws[1]:
                c.style = "Headline 4"


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    processed = root/"data/processed"
    tables = ensure_dir(root/"reports/tables")
    source = ensure_dir(root/"reports/source_data")
    discrete = read_csv_required(find_file(root, [processed/"phenotype_matrix_discrete.csv", root/"phenotype_matrix_discrete.csv"], ["**/phenotype_matrix_discrete.csv"]), "phenotype_matrix_discrete.csv")
    raw_path = find_file(root, [processed/"phenotype_matrix_raw.csv", root/"phenotype_matrix_raw.csv"], ["**/phenotype_matrix_raw.csv"])
    raw = pd.read_csv(raw_path) if raw_path else pd.DataFrame()
    annot = read_csv_required(find_file(root, [processed/"gene_annotations.csv", root/"gene_annotations.csv"], ["**/gene_annotations.csv"]), "gene_annotations.csv")
    catalog_path = find_file(root, [processed/"assay_catalog.csv", root/"assay_catalog.csv"], ["**/assay_catalog.csv"])
    catalog = pd.read_csv(catalog_path) if catalog_path else None
    assays = assay_columns(discrete, catalog)
    if "published_cluster" not in discrete: raise ValueError("published_cluster column required.")
    enc = EncodedPhenome(discrete, assays)

    sbml_df, sbml_genes, sbml_path = parse_sbml_model(root)
    mentions = parse_dreyfuss_tables(root)
    gene_classes = build_gene_classes(annot, sbml_genes)
    abnormal = abnormality_table(discrete, assays)

    subsets, full_metrics = build_subset_metrics(enc)
    exact_summary, exact_sets = summarize_exact_minimal(subsets, full_metrics, args.max_minimal_sets)
    greedy_profiles = greedy_curve(enc, "profiles")
    greedy_cluster = greedy_curve(enc, "cluster_nmi")
    greedy_cost = greedy_curve(enc, "cost_aware_profiles")
    cluster_shapley = shapley_cluster(enc, args.shapley_permutations, np.random.default_rng(args.random_seed + 1))
    bootstrap = bootstrap_reduced(enc, exact_summary, subsets, args.bootstrap_iterations, np.random.default_rng(args.random_seed + 2))
    colperm_iters = min(args.null_iterations, 200) if args.assay_permutation_null_iterations < 0 else args.assay_permutation_null_iterations
    null_summary, null_iters = null_cluster(enc, subsets, args.null_iterations, colperm_iters, np.random.default_rng(args.random_seed + 3))

    assay_enrich, axis_enrich, cluster_enrich, exposure, exposure_summary = enrichment_tables(discrete, annot, gene_classes, abnormal, assays)
    class_compression = class_specific_compression(enc, gene_classes)
    met_table, met_axis, met_summary = metabolic_layer(annot, abnormal, sbml_df, mentions)
    edges, orbits = substitutability_orbits(exact_sets, assays)
    marginal_cost = pd.concat([greedy_profiles.assign(curve="greedy_profile_resolution"), greedy_cluster.assign(curve="greedy_cluster_nmi"), greedy_cost.assign(curve="cost_aware_profile_resolution")], ignore_index=True)

    summary = pd.DataFrame([{ "n_mutants": len(discrete), "n_assays": len(assays), "n_published_clusters": int(discrete["published_cluster"].nunique()),
                              "full_discrete_profiles": enc.n_profiles(enc.full_mask), "full_profile_vs_cluster_nmi": full_metrics["nmi"],
                              "full_profile_vs_cluster_ari": full_metrics["ari"], "full_profile_vs_cluster_purity": full_metrics["purity"],
                              "sbml_model_reference": "iJDZ836 SBML model (Dreyfuss et al. 2013)" if sbml_path else "not_found", "n_sbml_genes_detected": len(sbml_genes),
                              "shapley_permutations": args.shapley_permutations, "bootstrap_iterations": args.bootstrap_iterations,
                              "cluster_label_null_iterations": args.null_iterations, "assay_permutation_null_iterations": colperm_iters, "random_seed": args.random_seed }])

    sheets = {
        "analysis_summary": summary,
        "cluster_all_subsets": subsets.sort_values(["n_assays", "cluster_nmi", "profile_fraction_of_full"], ascending=[True, False, False]),
        "cluster_exact_minimal": exact_summary,
        "cluster_minimal_sets": exact_sets,
        "cluster_greedy_profiles": greedy_profiles,
        "cluster_greedy_nmi": greedy_cluster,
        "cluster_shapley": cluster_shapley,
        "cluster_bootstrap": bootstrap,
        "cluster_null_summary": null_summary,
        "cluster_null_iterations": null_iters,
        "gene_class_membership": gene_classes,
        "gene_class_assay_enrich": assay_enrich.sort_values("fdr_q") if not assay_enrich.empty else assay_enrich,
        "gene_class_axis_enrich": axis_enrich.sort_values("fdr_q") if not axis_enrich.empty else axis_enrich,
        "gene_class_cluster_enrich": cluster_enrich.sort_values("fdr_q") if not cluster_enrich.empty else cluster_enrich,
        "gene_class_compression": class_compression,
        "exposure_gene_table": exposure,
        "exposure_summary": exposure_summary,
        "metabolic_gene_table": met_table,
        "metabolic_axis_enrich": met_axis,
        "metabolic_summary": met_summary,
        "marginal_cost_curves": marginal_cost,
        "assay_substitution_edges": edges,
        "assay_substitution_orbits": orbits,
        "abnormality_table": abnormal,
        "assay_catalog": catalog if catalog is not None else pd.DataFrame({"assay": assays, "axis": [assay_axis(a) for a in assays]}),
    }
    out = Path(args.output_workbook) if args.output_workbook else tables/"phenome_architecture_results.xlsx"
    write_workbook(out, sheets)
    subsets.to_csv(source/"cluster_all_subset_metrics.csv", index=False)
    axis_enrich.to_csv(source/"gene_class_axis_enrichment.csv", index=False)
    met_table.to_csv(source/"metabolic_gene_table.csv", index=False)
    exposure.to_csv(source/"exposure_gene_table.csv", index=False)
    print(f"Functional-architecture analysis complete: {out}")
    print(summary.to_string(index=False))
    key = exact_summary[(exact_summary["objective"].isin(["profile_fraction", "cluster_nmi", "cluster_purity"])) & (exact_summary["target_fraction_of_full"].isin([0.9,0.95,1.0]))]
    if not key.empty:
        print("\nKey exact-minimal targets:")
        print(key[["objective","target_fraction_of_full","minimal_n_assays","n_equivalent_minimal_sets","example_assay_set"]].to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
