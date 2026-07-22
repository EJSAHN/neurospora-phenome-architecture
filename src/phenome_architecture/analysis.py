from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


class ProfileCounter:
    """Fast unique-profile counter for small categorical assay panels."""

    def __init__(self, matrix: pd.DataFrame, assay_cols: list[str]):
        self.assay_cols = list(assay_cols)
        self.col_index = {c: i for i, c in enumerate(self.assay_cols)}
        coded = []
        for col in self.assay_cols:
            codes, _ = pd.factorize(matrix[col].astype(str), sort=True)
            coded.append(codes.astype(np.int16, copy=False))
        self.arr = (
            np.vstack(coded).T.astype(np.int64, copy=False)
            if coded
            else np.empty((matrix.shape[0], 0), dtype=np.int64)
        )
        if self.arr.shape[1]:
            self.bases = self.arr.max(axis=0).astype(np.int64) + 1
            self.bases[self.bases < 2] = 2
        else:
            self.bases = np.asarray([], dtype=np.int64)
        self.cache: dict[tuple[int, ...], int] = {tuple(): 1}

    def count(self, columns: Iterable[str]) -> int:
        idxs = tuple(sorted(self.col_index[c] for c in columns))
        if idxs in self.cache:
            return self.cache[idxs]
        # Exact mixed-radix encoding of the selected categorical columns.
        # This avoids slow repeated unique(axis=0) calls for small assay panels.
        h = np.zeros(self.arr.shape[0], dtype=np.int64)
        for idx in idxs:
            h = h * self.bases[idx] + self.arr[:, idx]
        value = int(np.unique(h).shape[0])
        self.cache[idxs] = value
        return value


def discretize_continuous(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 5 or x.nunique(dropna=True) <= 3:
        return x.astype("string").fillna("missing")
    q = x.quantile([0.10, 0.25, 0.75, 0.90]).to_dict()

    def label(v: float) -> str:
        if pd.isna(v):
            return "missing"
        if v <= q[0.10]:
            return "very_low"
        if v <= q[0.25]:
            return "low"
        if v <= q[0.75]:
            return "normal_range"
        if v <= q[0.90]:
            return "high"
        return "very_high"

    return x.map(label).astype("string")


def encode_phenotypes(df: pd.DataFrame, assay_cols: list[str]) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for col in assay_cols:
        s = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        is_continuous = (
            numeric.notna().sum() >= max(10, int(0.80 * len(s)))
            and numeric.nunique(dropna=True) > 8
        )
        if is_continuous:
            encoded[col] = discretize_continuous(s)
        else:
            encoded[col] = (
                s.astype("string")
                .str.strip()
                .str.replace(r"\s+", "_", regex=True)
                .str.lower()
                .fillna("missing")
            )
    return encoded


def n_profiles(matrix: pd.DataFrame, columns: Iterable[str]) -> int:
    columns = list(columns)
    if not columns:
        return 1
    return matrix[columns].astype(str).drop_duplicates().shape[0]


def greedy_accumulation(
    matrix: pd.DataFrame, assay_cols: list[str], counter: ProfileCounter | None = None
) -> pd.DataFrame:
    counter = counter or ProfileCounter(matrix, assay_cols)
    selected: list[str] = []
    remaining = list(assay_cols)
    records = []
    current = counter.count(selected)
    full = counter.count(assay_cols)
    for step in range(1, len(assay_cols) + 1):
        candidates = []
        for assay in remaining:
            new_cols = selected + [assay]
            profiles = counter.count(new_cols)
            candidates.append((profiles, profiles - current, assay))
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        profiles, gain, assay = candidates[0]
        selected.append(assay)
        remaining.remove(assay)
        current = profiles
        records.append(
            {
                "step": step,
                "assay_added": assay,
                "n_assays": len(selected),
                "n_profiles": profiles,
                "gain": gain,
                "resolution_fraction": profiles / full if full else np.nan,
                "selected_assays": ";".join(selected),
            }
        )
    return pd.DataFrame(records)


def exact_minimal_sets(
    matrix: pd.DataFrame,
    assay_cols: list[str],
    thresholds=(0.80, 0.90, 1.00),
    counter: ProfileCounter | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counter = counter or ProfileCounter(matrix, assay_cols)
    full = counter.count(assay_cols)
    summary_records = []
    set_records = []
    for threshold in thresholds:
        target = int(np.ceil(threshold * full))
        found: list[tuple[str, ...]] = []
        found_profiles = None
        for k in range(1, len(assay_cols) + 1):
            for combo in itertools.combinations(assay_cols, k):
                profiles = counter.count(combo)
                if profiles >= target:
                    found.append(combo)
                    found_profiles = (
                        profiles if found_profiles is None else max(found_profiles, profiles)
                    )
            if found:
                break
        summary_records.append(
            {
                "target_fraction": threshold,
                "full_profiles": full,
                "target_profiles": target,
                "minimal_n_assays": len(found[0]) if found else np.nan,
                "n_equivalent_minimal_sets": len(found),
                "example_assay_set": ";".join(found[0]) if found else "",
                "max_profiles_at_minimal_size": found_profiles,
            }
        )
        for combo in found:
            set_records.append(
                {
                    "target_fraction": threshold,
                    "n_assays": len(combo),
                    "n_profiles": counter.count(combo),
                    "assay_set": ";".join(combo),
                }
            )
    return pd.DataFrame(summary_records), pd.DataFrame(set_records)


def shapley_like_contribution(
    matrix: pd.DataFrame, assay_cols: list[str], n_permutations: int = 1000, random_state: int = 1
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    counter = ProfileCounter(matrix, assay_cols)
    contributions = {a: [] for a in assay_cols}
    for _ in range(n_permutations):
        order = list(rng.permutation(assay_cols))
        selected: list[str] = []
        previous = counter.count(selected)
        for assay in order:
            selected.append(assay)
            current = counter.count(selected)
            contributions[assay].append(current - previous)
            previous = current
    records = []
    full = counter.count(assay_cols)
    for assay, values in contributions.items():
        arr = np.asarray(values, dtype=float)
        records.append(
            {
                "assay": assay,
                "mean_marginal_gain": arr.mean(),
                "sd_marginal_gain": arr.std(ddof=1),
                "median_marginal_gain": np.median(arr),
                "normalized_contribution": arr.mean() / max(full - 1, 1),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("mean_marginal_gain", ascending=False)
        .reset_index(drop=True)
    )


def bootstrap_stability(
    matrix: pd.DataFrame, assay_cols: list[str], n_iterations: int = 200, random_state: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    inclusion = {a: {"k80": 0, "k90": 0, "k100": 0, "first": 0} for a in assay_cols}
    summary = []
    n = matrix.shape[0]
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        sample = matrix.iloc[idx].reset_index(drop=True)
        counter = ProfileCounter(sample, assay_cols)
        greedy = greedy_accumulation(sample, assay_cols, counter)
        full = counter.count(assay_cols)
        row = {"iteration": i + 1, "full_profiles": full}
        for label, threshold in [("k80", 0.80), ("k90", 0.90), ("k100", 1.00)]:
            target = np.ceil(threshold * full)
            hit = greedy.loc[greedy["n_profiles"] >= target]
            k = int(hit.iloc[0]["n_assays"]) if not hit.empty else len(assay_cols)
            row[label] = k
            selected = set(greedy.iloc[:k]["assay_added"].tolist())
            for assay in selected:
                inclusion[assay][label] += 1
        first_assay = str(greedy.iloc[0]["assay_added"])
        inclusion[first_assay]["first"] += 1
        summary.append(row)
    inc_records = []
    for assay, vals in inclusion.items():
        rec = {"assay": assay}
        for key, count in vals.items():
            rec[f"{key}_inclusion_frequency"] = count / n_iterations
        inc_records.append(rec)
    return pd.DataFrame(summary), pd.DataFrame(inc_records).sort_values(
        "k90_inclusion_frequency", ascending=False
    ).reset_index(drop=True)


def column_permutation_null(
    matrix: pd.DataFrame, assay_cols: list[str], n_iterations: int = 200, random_state: int = 3
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    records = []
    observed_counter = ProfileCounter(matrix, assay_cols)
    observed_greedy = greedy_accumulation(matrix, assay_cols, observed_counter)
    observed_exact, _ = exact_minimal_sets(matrix, assay_cols, counter=observed_counter)
    observed_k100 = int(
        observed_exact.loc[observed_exact["target_fraction"] == 1.0, "minimal_n_assays"].iloc[0]
    )
    full = observed_counter.count(assay_cols)
    for i in range(n_iterations):
        shuffled = matrix.copy()
        for col in assay_cols:
            shuffled[col] = rng.permutation(shuffled[col].astype(str).values)
        counter = ProfileCounter(shuffled, assay_cols)
        exact_summary, _ = exact_minimal_sets(shuffled, assay_cols, counter=counter)
        greedy = greedy_accumulation(shuffled, assay_cols, counter)
        k100 = int(
            exact_summary.loc[exact_summary["target_fraction"] == 1.0, "minimal_n_assays"].iloc[0]
        )
        top_gain_fraction = float(greedy.iloc[0]["gain"] / max(full - 1, 1))
        records.append(
            {
                "iteration": i + 1,
                "null_exact_k100": k100,
                "null_top_gain_fraction": top_gain_fraction,
            }
        )
    out = pd.DataFrame(records)
    out.attrs["observed_exact_k100"] = observed_k100
    out.attrs["observed_top_gain_fraction"] = float(
        observed_greedy.iloc[0]["gain"] / max(full - 1, 1)
    )
    return out


def pairwise_redundancy(matrix: pd.DataFrame, assay_cols: list[str]) -> pd.DataFrame:
    records = []
    for a, b in itertools.product(assay_cols, assay_cols):
        score = normalized_mutual_info_score(matrix[a].astype(str), matrix[b].astype(str))
        records.append({"assay_1": a, "assay_2": b, "normalized_mutual_information": score})
    return pd.DataFrame(records)
