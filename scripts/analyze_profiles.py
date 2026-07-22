from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phenome_architecture.analysis import (
    bootstrap_stability,
    column_permutation_null,
    exact_minimal_sets,
    greedy_accumulation,
    n_profiles,
    pairwise_redundancy,
    shapley_like_contribution,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--shapley-permutations", type=int, default=10000)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--null-iterations", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.project_root)
    processed = root / "data" / "processed"
    out_dir = root / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(processed / "phenotype_matrix_discrete.csv")
    annotations = pd.read_csv(processed / "gene_annotations.csv")
    assay_catalog = pd.read_csv(processed / "assay_catalog.csv")
    assay_cols = assay_catalog["assay"].tolist()
    assay_matrix = matrix[assay_cols].astype(str)

    full_profiles = n_profiles(assay_matrix, assay_cols)
    summary = pd.DataFrame(
        [
            {
                "n_mutants": matrix.shape[0],
                "n_assays": len(assay_cols),
                "full_discrete_phenotype_profiles": full_profiles,
                "reference_clusters": matrix["published_cluster"].nunique(),
                "shapley_permutations": args.shapley_permutations,
                "bootstrap_iterations": args.bootstrap_iterations,
                "null_iterations": args.null_iterations,
            }
        ]
    )

    greedy = greedy_accumulation(assay_matrix, assay_cols)
    exact_summary, exact_sets = exact_minimal_sets(assay_matrix, assay_cols)
    shapley = shapley_like_contribution(
        assay_matrix, assay_cols, args.shapley_permutations, args.random_state
    )
    boot_summary, boot_inclusion = bootstrap_stability(
        assay_matrix, assay_cols, args.bootstrap_iterations, args.random_state + 1
    )
    null = column_permutation_null(
        assay_matrix, assay_cols, args.null_iterations, args.random_state + 2
    )
    redundancy = pairwise_redundancy(assay_matrix, assay_cols)

    observed_exact_k100 = int(
        exact_summary.loc[exact_summary["target_fraction"] == 1.0, "minimal_n_assays"].iloc[0]
    )
    null_summary = pd.DataFrame(
        [
            {
                "metric": "exact_k100",
                "observed": observed_exact_k100,
                "null_mean": null["null_exact_k100"].mean(),
                "null_sd": null["null_exact_k100"].std(ddof=1),
                "empirical_p_lower_or_equal": (
                    1 + (null["null_exact_k100"] <= observed_exact_k100).sum()
                )
                / (len(null) + 1),
                "empirical_p_upper_or_equal": (
                    1 + (null["null_exact_k100"] >= observed_exact_k100).sum()
                )
                / (len(null) + 1),
            }
        ]
    )

    # Simple role table for assays.
    merged = shapley.merge(
        greedy[["assay_added", "step"]].rename(
            columns={"assay_added": "assay", "step": "greedy_step"}
        ),
        on="assay",
        how="left",
    )
    merged = merged.merge(boot_inclusion, on="assay", how="left")
    q_high = merged["mean_marginal_gain"].quantile(0.75)
    q_low = merged["mean_marginal_gain"].quantile(0.25)

    def role(row):
        if row["mean_marginal_gain"] >= q_high and row["k90_inclusion_frequency"] >= 0.5:
            return "phenome_bottleneck_assay"
        if row["mean_marginal_gain"] >= q_high:
            return "high_information_assay"
        if row["mean_marginal_gain"] <= q_low:
            return "low_information_or_anchor_assay"
        return "supporting_assay"

    merged["information_role"] = merged.apply(role, axis=1)
    assay_roles = merged.sort_values(["greedy_step", "mean_marginal_gain"], ascending=[True, False])

    workbook = out_dir / "phenome_profile_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")
        greedy.to_excel(writer, index=False, sheet_name="greedy_accumulation")
        exact_summary.to_excel(writer, index=False, sheet_name="exact_minimal_summary")
        exact_sets.to_excel(writer, index=False, sheet_name="exact_minimal_sets")
        shapley.to_excel(writer, index=False, sheet_name="shapley_contribution")
        assay_roles.to_excel(writer, index=False, sheet_name="assay_roles")
        boot_summary.to_excel(writer, index=False, sheet_name="bootstrap_summary")
        boot_inclusion.to_excel(writer, index=False, sheet_name="bootstrap_assay_inclusion")
        null_summary.to_excel(writer, index=False, sheet_name="null_summary")
        null.to_excel(writer, index=False, sheet_name="column_permutation_null")
        redundancy.to_excel(writer, index=False, sheet_name="assay_redundancy_nmi")
        assay_catalog.to_excel(writer, index=False, sheet_name="assay_catalog")
        annotations.to_excel(writer, index=False, sheet_name="gene_annotations")

    print(f"Analysis complete: {workbook}")
    print(summary.to_string(index=False))
    print(exact_summary.to_string(index=False))


if __name__ == "__main__":
    main()
