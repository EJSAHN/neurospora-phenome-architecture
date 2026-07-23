from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from phenome_architecture.analysis import ProfileCounter

NORMAL_VALUES = {"normal", "normal_range", "normal range", "wild_type", "wild-type", "wt"}


def canonical_state(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .fillna("missing")
    )


def is_abnormal(series: pd.Series) -> pd.Series:
    return ~canonical_state(series).isin({x.replace(" ", "_") for x in NORMAL_VALUES})


def ordinal_state_codes(series: pd.Series) -> pd.Series:
    """Map ordered phenotype categories to numeric scores for correlation summaries."""
    states = canonical_state(series)
    fixed = {
        "very_low": 0.0,
        "low": 1.0,
        "normal_range": 2.0,
        "normal": 2.0,
        "high": 3.0,
        "very_high": 4.0,
    }
    output = pd.Series(np.nan, index=states.index, dtype=float)
    for label, value in fixed.items():
        output.loc[states.eq(label)] = value
    for prefix in ("q", "eq"):
        mask = states.str.fullmatch(rf"{prefix}\d+")
        if mask.any():
            output.loc[mask] = (
                states.loc[mask].str.extract(r"(\d+)", expand=False).astype(float) - 1.0
            )
    unknown = output.isna() & ~states.eq("missing")
    if unknown.any():
        labels = sorted(states.loc[unknown].unique())
        fallback = {label: float(index) for index, label in enumerate(labels)}
        output.loc[unknown] = states.loc[unknown].map(fallback)
    return output


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x.loc[valid].nunique() < 2 or y.loc[valid].nunique() < 2:
        return float("nan"), float("nan"), int(valid.sum())
    result = (
        stats.pearsonr(x.loc[valid], y.loc[valid])
        if method == "pearson"
        else stats.spearmanr(x.loc[valid], y.loc[valid])
    )
    return float(result.statistic), float(result.pvalue), int(valid.sum())


def cramer_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(canonical_state(x), canonical_state(y))
    if table.empty:
        return float("nan")
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n <= 1:
        return float("nan")
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(math.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def pairwise_discrete_dependence(discrete: pd.DataFrame, assays: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for assay_a, assay_b in itertools.combinations(assays, 2):
        a = canonical_state(discrete[assay_a])
        b = canonical_state(discrete[assay_b])
        rows.append(
            {
                "assay_a": assay_a,
                "assay_b": assay_b,
                "normalized_mutual_information": float(normalized_mutual_info_score(a, b)),
                "adjusted_rand_index": float(adjusted_rand_score(a, b)),
                "cramers_v": cramer_v(a, b),
                "n_assay_a_states": int(a.nunique()),
                "n_assay_b_states": int(b.nunique()),
            }
        )
    return pd.DataFrame(rows)


def continuous_trait_association(
    raw: pd.DataFrame,
    discrete: pd.DataFrame,
    assay_a: str,
    assay_b: str,
) -> pd.DataFrame:
    x = pd.to_numeric(raw[assay_a], errors="coerce")
    y = pd.to_numeric(raw[assay_b], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        raise ValueError(
            f"Fewer than three paired numeric observations for {assay_a!r} and {assay_b!r}"
        )
    pearson = stats.pearsonr(x[valid], y[valid])
    spearman = stats.spearmanr(x[valid], y[valid])
    a_disc = canonical_state(discrete.loc[valid, assay_a])
    b_disc = canonical_state(discrete.loc[valid, assay_b])
    a_ordinal = ordinal_state_codes(a_disc)
    b_ordinal = ordinal_state_codes(b_disc)
    ordinal_pearson_r, ordinal_pearson_p, ordinal_n = safe_correlation(
        a_ordinal, b_ordinal, "pearson"
    )
    ordinal_spearman_rho, ordinal_spearman_p, _ = safe_correlation(a_ordinal, b_ordinal, "spearman")
    return pd.DataFrame(
        [
            {
                "assay_a": assay_a,
                "assay_b": assay_b,
                "n_paired": int(valid.sum()),
                "pearson_r": float(pearson.statistic),
                "pearson_r_squared": float(pearson.statistic**2),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "discrete_n_paired": ordinal_n,
                "discrete_ordinal_pearson_r": ordinal_pearson_r,
                "discrete_ordinal_pearson_r_squared": (
                    ordinal_pearson_r**2 if not np.isnan(ordinal_pearson_r) else np.nan
                ),
                "discrete_ordinal_pearson_p": ordinal_pearson_p,
                "discrete_ordinal_spearman_rho": ordinal_spearman_rho,
                "discrete_ordinal_spearman_p": ordinal_spearman_p,
                "discrete_normalized_mutual_information": float(
                    normalized_mutual_info_score(a_disc, b_disc)
                ),
                "discrete_adjusted_rand_index": float(adjusted_rand_score(a_disc, b_disc)),
                "discrete_cramers_v": cramer_v(a_disc, b_disc),
            }
        ]
    )


GROWTH_DISCRETIZATION_SCHEMES: dict[str, tuple[str, object] | None] = {
    "baseline_current_file": None,
    "quantile5_original_10_25_75_90": ("quantile", [0.10, 0.25, 0.75, 0.90]),
    "quantile5_wide_normal_05_20_80_95": ("quantile", [0.05, 0.20, 0.80, 0.95]),
    "quantile5_narrow_normal_15_30_70_85": ("quantile", [0.15, 0.30, 0.70, 0.85]),
    "quantile3_tertiles": ("qcut", 3),
    "quantile4_quartiles": ("qcut", 4),
    "quantile6_sextiles": ("qcut", 6),
    "equal_width5": ("equal_width", 5),
    "zscore5_mean_sd": ("zscore", 5),
}


def _quantile_categories(series: pd.Series, probabilities: Sequence[float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    cutpoints = np.maximum.accumulate(values.quantile(probabilities).to_numpy(dtype=float))
    labels = ["very_low", "low", "normal_range", "high", "very_high"]
    output: list[str] = []
    for value in values:
        if pd.isna(value):
            output.append("missing")
        else:
            output.append(
                labels[
                    min(
                        int(np.searchsorted(cutpoints, float(value), side="right")), len(labels) - 1
                    )
                ]
            )
    return pd.Series(output, index=series.index, dtype="string")


def _rank_quantile_categories(series: pd.Series, q: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    codes = pd.qcut(values.rank(method="first"), q=q, labels=False, duplicates="drop")
    return codes.map(lambda value: "missing" if pd.isna(value) else f"q{int(value) + 1}").astype(
        "string"
    )


def _equal_width_categories(series: pd.Series, bins: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    codes = pd.cut(values, bins=bins, labels=False, include_lowest=True)
    return codes.map(lambda value: "missing" if pd.isna(value) else f"eq{int(value) + 1}").astype(
        "string"
    )


def _zscore_categories(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    z = (values - values.mean()) / values.std(ddof=1)
    output: list[str] = []
    for value in z:
        if pd.isna(value):
            output.append("missing")
        elif value <= -1.5:
            output.append("very_low")
        elif value <= -0.5:
            output.append("low")
        elif value <= 0.5:
            output.append("normal_range")
        elif value <= 1.5:
            output.append("high")
        else:
            output.append("very_high")
    return pd.Series(output, index=series.index, dtype="string")


def _discretize_for_scheme(series: pd.Series, scheme: tuple[str, object]) -> pd.Series:
    method, parameter = scheme
    if method == "quantile":
        return _quantile_categories(series, parameter)  # type: ignore[arg-type]
    if method == "qcut":
        return _rank_quantile_categories(series, int(parameter))
    if method == "equal_width":
        return _equal_width_categories(series, int(parameter))
    if method == "zscore":
        return _zscore_categories(series)
    raise ValueError(f"Unknown discretization method: {method}")


def growth_discretization_sensitivity(
    raw: pd.DataFrame,
    discrete: pd.DataFrame,
    assay_a: str,
    assay_b: str,
) -> pd.DataFrame:
    """Quantify association between the two continuous assays across all coding schemes."""
    rows: list[dict[str, object]] = []
    for scheme_id, scheme in GROWTH_DISCRETIZATION_SCHEMES.items():
        if scheme is None:
            a = canonical_state(discrete[assay_a])
            b = canonical_state(discrete[assay_b])
        else:
            a = canonical_state(_discretize_for_scheme(raw[assay_a], scheme))
            b = canonical_state(_discretize_for_scheme(raw[assay_b], scheme))
        a_ordinal = ordinal_state_codes(a)
        b_ordinal = ordinal_state_codes(b)
        ordinal_pearson_r, ordinal_pearson_p, ordinal_n = safe_correlation(
            a_ordinal, b_ordinal, "pearson"
        )
        ordinal_spearman_rho, ordinal_spearman_p, _ = safe_correlation(
            a_ordinal, b_ordinal, "spearman"
        )
        rows.append(
            {
                "scheme_id": scheme_id,
                "n_paired": ordinal_n,
                "ordinal_pearson_r": ordinal_pearson_r,
                "ordinal_pearson_r_squared": (
                    ordinal_pearson_r**2 if not np.isnan(ordinal_pearson_r) else np.nan
                ),
                "ordinal_pearson_p": ordinal_pearson_p,
                "ordinal_spearman_rho": ordinal_spearman_rho,
                "ordinal_spearman_p": ordinal_spearman_p,
                "normalized_mutual_information": float(normalized_mutual_info_score(a, b)),
                "adjusted_rand_index": float(adjusted_rand_score(a, b)),
                "cramers_v": cramer_v(a, b),
                "n_categories_assay_a": int(a.nunique()),
                "n_categories_assay_b": int(b.nunique()),
            }
        )
    return pd.DataFrame(rows)


def developmental_dependency_table(
    discrete: pd.DataFrame,
    ordered_assays: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for upstream_idx, downstream_idx in itertools.combinations(range(len(ordered_assays)), 2):
        upstream = ordered_assays[upstream_idx]
        downstream = ordered_assays[downstream_idx]
        up_ab = is_abnormal(discrete[upstream])
        down_ab = is_abnormal(discrete[downstream])
        a = int((up_ab & down_ab).sum())
        b = int((up_ab & ~down_ab).sum())
        c = int((~up_ab & down_ab).sum())
        d = int((~up_ab & ~down_ab).sum())
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        log_or = math.log((aa * dd) / (bb * cc))
        se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        rows.append(
            {
                "upstream_assay": upstream,
                "downstream_assay": downstream,
                "upstream_stage_order": upstream_idx + 1,
                "downstream_stage_order": downstream_idx + 1,
                "both_abnormal": a,
                "upstream_abnormal_downstream_normal": b,
                "upstream_normal_downstream_abnormal": c,
                "both_normal": d,
                "p_downstream_abnormal_given_upstream_abnormal": a / max(a + b, 1),
                "p_downstream_abnormal_given_upstream_normal": c / max(c + d, 1),
                "odds_ratio": float(odds),
                "log2_odds_ratio_corrected": float(log_or / math.log(2)),
                "odds_ratio_ci95_low_corrected": float(math.exp(log_or - 1.96 * se)),
                "odds_ratio_ci95_high_corrected": float(math.exp(log_or + 1.96 * se)),
                "fisher_p": float(p),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(
                        canonical_state(discrete[upstream]), canonical_state(discrete[downstream])
                    )
                ),
                "cramers_v": cramer_v(discrete[upstream], discrete[downstream]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        pvals = out["fisher_p"].to_numpy(dtype=float)
        order = np.argsort(pvals)
        q = pvals[order] * len(pvals) / np.arange(1, len(pvals) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q_out = np.empty_like(q)
        q_out[order] = np.minimum(q, 1.0)
        out["fdr_q"] = q_out
    return out


def build_module_matrix(
    discrete: pd.DataFrame, module_catalog: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"module", "component_assay", "axis", "stage_order"}
    missing = required - set(module_catalog.columns)
    if missing:
        raise ValueError(f"Module catalog is missing columns: {sorted(missing)}")
    module_frame = discrete[
        [c for c in ["ncu_id", "gene_name", "published_cluster"] if c in discrete.columns]
    ].copy()
    catalog_rows: list[dict[str, object]] = []
    for module, group in module_catalog.groupby("module", sort=False):
        components = [str(x) for x in group["component_assay"] if str(x) in discrete.columns]
        if not components:
            continue
        if len(components) == 1:
            module_frame[module] = canonical_state(discrete[components[0]])
        else:
            module_frame[module] = discrete[components].apply(
                lambda row: "|".join(
                    f"{component}={canonical_state(pd.Series([row[component]])).iloc[0]}"
                    for component in components
                ),
                axis=1,
            )
        catalog_rows.append(
            {
                "module": module,
                "axis": str(group["axis"].iloc[0]),
                "stage_order": int(group["stage_order"].iloc[0]),
                "n_component_assays": len(components),
                "component_assays": ";".join(components),
            }
        )
    return module_frame, pd.DataFrame(catalog_rows)


def contingency_metrics(true_labels: pd.Series, predicted_labels: pd.Series) -> dict[str, float]:
    true = pd.factorize(true_labels.astype(str), sort=True)[0]
    pred = pd.factorize(predicted_labels.astype(str), sort=True)[0]
    table = pd.crosstab(true, pred).to_numpy(dtype=float)
    n = table.sum()
    purity = table.max(axis=0).sum() / n if n else float("nan")
    return {
        "cluster_nmi": float(normalized_mutual_info_score(true, pred)),
        "cluster_ari": float(adjusted_rand_score(true, pred)),
        "cluster_purity": float(purity),
    }


def exact_module_subset_metrics(
    module_frame: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modules = catalog.sort_values("stage_order")["module"].tolist()
    codes = encode_categorical_matrix(module_frame, modules)
    true_codes = pd.factorize(module_frame["published_cluster"].astype(str), sort=True)[0]
    full_mask = (1 << len(modules)) - 1
    full_labels = labels_from_mask(codes, full_mask)
    full_profiles = int(full_labels.max()) + 1
    full_metrics = fast_cluster_metrics(true_codes, full_labels)
    rows: list[dict[str, object]] = []
    for mask in range(1, full_mask + 1):
        combo = [modules[index] for index in range(len(modules)) if mask & (1 << index)]
        labels = labels_from_mask(codes, mask)
        metrics = fast_cluster_metrics(true_codes, labels)
        n_profiles = int(labels.max()) + 1
        components = catalog[catalog["module"].isin(combo)]
        rows.append(
            {
                "mask": mask,
                "n_modules": len(combo),
                "n_component_assays": int(components["n_component_assays"].sum()),
                "module_set": ";".join(combo),
                "component_assay_set": ";".join(components["component_assays"].tolist()),
                "n_profiles": n_profiles,
                "profile_fraction_of_full": n_profiles / full_profiles,
                **metrics,
                "cluster_nmi_fraction_of_full": (
                    metrics["cluster_nmi"] / full_metrics["cluster_nmi"]
                    if full_metrics["cluster_nmi"]
                    else float("nan")
                ),
                "cluster_ari_fraction_of_full": (
                    metrics["cluster_ari"] / full_metrics["cluster_ari"]
                    if full_metrics["cluster_ari"]
                    else float("nan")
                ),
                "cluster_purity_fraction_of_full": (
                    metrics["cluster_purity"] / full_metrics["cluster_purity"]
                    if full_metrics["cluster_purity"]
                    else float("nan")
                ),
            }
        )
    all_metrics = pd.DataFrame(rows)
    exact_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    objectives = {
        "module_profile_recovery": "profile_fraction_of_full",
        "module_cluster_nmi_recovery": "cluster_nmi_fraction_of_full",
        "module_cluster_ari_recovery": "cluster_ari_fraction_of_full",
        "module_cluster_purity_recovery": "cluster_purity_fraction_of_full",
    }
    for objective, metric in objectives.items():
        for target in (0.80, 0.90, 0.95, 0.99, 1.00):
            eligible = all_metrics[all_metrics[metric] >= target - 1e-12]
            if eligible.empty:
                continue
            min_modules = int(eligible["n_modules"].min())
            minimal = eligible[eligible["n_modules"] == min_modules].sort_values(
                metric, ascending=False
            )
            best = minimal.iloc[0]
            exact_rows.append(
                {
                    "objective": objective,
                    "target_fraction_of_full": target,
                    "metric_column": metric,
                    "minimal_n_modules": min_modules,
                    "minimal_component_assays": int(minimal["n_component_assays"].min()),
                    "n_equivalent_minimal_sets": len(minimal),
                    "example_module_set": best["module_set"],
                    "example_component_assay_set": best["component_assay_set"],
                    "best_metric_at_minimal_size": float(best[metric]),
                }
            )
            for rank, (_, row) in enumerate(minimal.iterrows(), start=1):
                set_rows.append(
                    {
                        "objective": objective,
                        "target_fraction_of_full": target,
                        "rank_within_target": rank,
                        **row.to_dict(),
                    }
                )
    return all_metrics, pd.DataFrame(exact_rows), pd.DataFrame(set_rows)


def all_assay_substitutability_matrix(
    minimal_sets: pd.DataFrame,
    assays: Sequence[str],
    objective: str = "cluster_nmi_recovery",
    target: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = minimal_sets[
        (minimal_sets["objective"].astype(str) == objective)
        & np.isclose(minimal_sets["target_fraction_of_full"].astype(float), target)
    ].copy()
    assay_sets = [frozenset(str(value).split(";")) for value in subset["assay_set"].dropna()]
    edge_counts: dict[tuple[str, str], int] = {}
    for set_a, set_b in itertools.combinations(assay_sets, 2):
        difference = set_a.symmetric_difference(set_b)
        if len(difference) == 2:
            pair = tuple(sorted(difference))
            edge_counts[pair] = edge_counts.get(pair, 0) + 1
    matrix = pd.DataFrame(0, index=list(assays), columns=list(assays), dtype=int)
    edge_rows: list[dict[str, object]] = []
    for (assay_a, assay_b), count in sorted(edge_counts.items()):
        if assay_a in matrix.index and assay_b in matrix.columns:
            matrix.loc[assay_a, assay_b] = count
            matrix.loc[assay_b, assay_a] = count
        edge_rows.append(
            {
                "objective": objective,
                "target_fraction_of_full": target,
                "assay_a": assay_a,
                "assay_b": assay_b,
                "n_one_for_one_swaps": count,
            }
        )
    for assay in matrix.index:
        if assay in matrix.columns:
            matrix.loc[assay, assay] = 0
    matrix.insert(0, "assay", matrix.index)
    matrix = matrix.reset_index(drop=True)
    return matrix, pd.DataFrame(edge_rows)


def minimal_set_assay_frequency(
    minimal_sets: pd.DataFrame,
    assays: Sequence[str],
    objective: str = "cluster_nmi_recovery",
    target: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize assay participation in equivalent exact minimal sets."""
    subset = minimal_sets[
        (minimal_sets["objective"].astype(str) == objective)
        & np.isclose(minimal_sets["target_fraction_of_full"].astype(float), target)
    ].copy()
    assay_sets = [
        frozenset(token for token in str(value).split(";") if token)
        for value in subset.get("assay_set", pd.Series(dtype=str)).dropna()
    ]
    n_sets = len(assay_sets)
    frequency = pd.DataFrame(
        [
            {
                "objective": objective,
                "target_fraction_of_full": target,
                "assay": assay,
                "n_equivalent_minimal_sets": n_sets,
                "n_sets_containing_assay": sum(assay in assay_set for assay_set in assay_sets),
                "fraction_of_minimal_sets": (
                    sum(assay in assay_set for assay_set in assay_sets) / n_sets
                    if n_sets
                    else np.nan
                ),
            }
            for assay in assays
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "objective": objective,
                "target_fraction_of_full": target,
                "n_equivalent_minimal_sets": n_sets,
                "minimal_set_size": (len(next(iter(assay_sets))) if assay_sets else np.nan),
            }
        ]
    )
    return metadata, frequency


def encode_categorical_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    coded = []
    for column in columns:
        codes = pd.factorize(canonical_state(frame[column]), sort=True)[0]
        coded.append(codes.astype(np.int16, copy=False))
    return np.vstack(coded).T if coded else np.empty((len(frame), 0), dtype=np.int16)


def labels_from_mask(codes: np.ndarray, mask: int) -> np.ndarray:
    indices = [index for index in range(codes.shape[1]) if mask & (1 << index)]
    if not indices:
        return np.zeros(codes.shape[0], dtype=np.int32)
    return np.unique(codes[:, indices], axis=0, return_inverse=True)[1].astype(np.int32)


def fast_cluster_metrics(true_codes: np.ndarray, pred_codes: np.ndarray) -> dict[str, float]:
    true_codes = np.asarray(true_codes, dtype=np.int64)
    pred_codes = np.asarray(pred_codes, dtype=np.int64)
    n_true = int(true_codes.max()) + 1 if len(true_codes) else 0
    n_pred = int(pred_codes.max()) + 1 if len(pred_codes) else 0
    if n_true == 0 or n_pred == 0:
        return {"cluster_nmi": np.nan, "cluster_ari": np.nan, "cluster_purity": np.nan}
    flat = true_codes * n_pred + pred_codes
    table = np.bincount(flat, minlength=n_true * n_pred).reshape(n_true, n_pred).astype(float)
    n = table.sum()
    row = table.sum(axis=1)
    col = table.sum(axis=0)
    nonzero = table > 0
    expected = np.outer(row, col)
    mutual_information = float(
        np.sum((table[nonzero] / n) * np.log((table[nonzero] * n) / expected[nonzero]))
    )
    h_true = float(-np.sum((row[row > 0] / n) * np.log(row[row > 0] / n)))
    h_pred = float(-np.sum((col[col > 0] / n) * np.log(col[col > 0] / n)))
    nmi = 2 * mutual_information / (h_true + h_pred) if h_true + h_pred > 0 else 1.0
    purity = float(table.max(axis=0).sum() / n)
    comb = lambda values: values * (values - 1) / 2
    sum_comb = float(comb(table).sum())
    sum_rows = float(comb(row).sum())
    sum_cols = float(comb(col).sum())
    total = float(comb(n))
    expected_index = sum_rows * sum_cols / total if total else 0.0
    max_index = 0.5 * (sum_rows + sum_cols)
    denominator = max_index - expected_index
    ari = (sum_comb - expected_index) / denominator if denominator else 0.0
    return {"cluster_nmi": nmi, "cluster_ari": ari, "cluster_purity": purity}
