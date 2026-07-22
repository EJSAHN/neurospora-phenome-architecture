#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenome_architecture.dependency import (
    all_assay_substitutability_matrix,
    build_module_matrix,
    continuous_trait_association,
    developmental_dependency_table,
    encode_categorical_matrix,
    exact_module_subset_metrics,
    fast_cluster_metrics,
    growth_discretization_sensitivity,
    labels_from_mask,
    minimal_set_assay_frequency,
    pairwise_discrete_dependence,
)

GROWTH_ASSAYS = [
    "Basal hyphae growth rate (mm/day)",
    "Aerial hyphae height (total mm)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze assay dependence and module-level recovery"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--module-catalog", default="config/assay_modules.csv")
    parser.add_argument("--output-workbook", default="reports/tables/assay_dependency_results.xlsx")
    parser.add_argument("--substitutability-target", type=float, default=0.90)
    return parser.parse_args()


def assay_columns(discrete: pd.DataFrame, catalog: pd.DataFrame | None) -> list[str]:
    if catalog is not None and "assay" in catalog.columns:
        assays = [str(value) for value in catalog["assay"] if str(value) in discrete.columns]
        if assays:
            return assays
    return [
        column
        for column in discrete.columns
        if column not in {"ncu_id", "gene_name", "published_cluster"}
    ]


def all_subset_metrics(
    discrete: pd.DataFrame,
    assays: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    codes = encode_categorical_matrix(discrete, assays)
    true_codes = pd.factorize(discrete["published_cluster"].astype(str), sort=True)[0]
    full_mask = (1 << len(assays)) - 1
    full_labels = labels_from_mask(codes, full_mask)
    full_profiles = int(full_labels.max()) + 1
    full_metrics = fast_cluster_metrics(true_codes, full_labels)
    rows: list[dict[str, object]] = []
    for mask in range(1, full_mask + 1):
        combo = [assays[index] for index in range(len(assays)) if mask & (1 << index)]
        labels = labels_from_mask(codes, mask)
        metrics = fast_cluster_metrics(true_codes, labels)
        n_profiles = int(labels.max()) + 1
        rows.append(
            {
                "mask": mask,
                "n_assays": len(combo),
                "assay_set": ";".join(combo),
                "n_profiles": n_profiles,
                "profile_fraction_of_full": n_profiles / full_profiles,
                **metrics,
                "cluster_nmi_fraction_of_full": (
                    metrics["cluster_nmi"] / full_metrics["cluster_nmi"]
                    if full_metrics["cluster_nmi"]
                    else np.nan
                ),
                "cluster_ari_fraction_of_full": (
                    metrics["cluster_ari"] / full_metrics["cluster_ari"]
                    if full_metrics["cluster_ari"]
                    else np.nan
                ),
                "cluster_purity_fraction_of_full": (
                    metrics["cluster_purity"] / full_metrics["cluster_purity"]
                    if full_metrics["cluster_purity"]
                    else np.nan
                ),
            }
        )
    summary = {"full_profiles": full_profiles, **full_metrics}
    return pd.DataFrame(rows), summary


def exact_minimal_sets(subsets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    objectives = {
        "profile_recovery": "profile_fraction_of_full",
        "cluster_nmi_recovery": "cluster_nmi_fraction_of_full",
        "cluster_ari_recovery": "cluster_ari_fraction_of_full",
        "cluster_purity_recovery": "cluster_purity_fraction_of_full",
    }
    exact_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    for objective, metric in objectives.items():
        for target in (0.80, 0.90, 0.95, 0.99, 1.00):
            eligible = subsets[subsets[metric] >= target - 1e-12]
            if eligible.empty:
                continue
            minimum = int(eligible["n_assays"].min())
            minimal = eligible[eligible["n_assays"] == minimum].sort_values(
                metric,
                ascending=False,
            )
            best = minimal.iloc[0]
            exact_rows.append(
                {
                    "objective": objective,
                    "target_fraction_of_full": target,
                    "metric_column": metric,
                    "minimal_n_assays": minimum,
                    "n_equivalent_minimal_sets": len(minimal),
                    "example_assay_set": best["assay_set"],
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
    return pd.DataFrame(exact_rows), pd.DataFrame(set_rows)


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.book[name[:31]]
            sheet.freeze_panes = "A2"
            for index, column in enumerate(frame.columns, start=1):
                values = frame[column].head(300).astype("string").fillna("").tolist()
                width = min(
                    max([len(str(column))] + [len(value) for value in values]) + 2,
                    55,
                )
                sheet.column_dimensions[sheet.cell(1, index).column_letter].width = max(
                    width,
                    10,
                )
            for cell in sheet[1]:
                cell.style = "Headline 4"


def build_summary(
    discrete: pd.DataFrame,
    assays: Sequence[str],
    full_summary: dict[str, float],
    continuous: pd.DataFrame,
    dependencies: pd.DataFrame,
    assay_exact: pd.DataFrame,
    module_exact: pd.DataFrame,
    substitution_metadata: pd.DataFrame,
) -> pd.DataFrame:
    record: dict[str, object] = {
        "n_mutants": len(discrete),
        "n_assays": len(assays),
        "full_profiles": int(full_summary["full_profiles"]),
        "full_cluster_nmi": float(full_summary["cluster_nmi"]),
    }
    if not continuous.empty:
        row = continuous.iloc[0]
        record.update(
            {
                "raw_growth_pearson_r": row.get("pearson_r", np.nan),
                "raw_growth_r_squared": row.get("pearson_r_squared", np.nan),
                "raw_growth_spearman_rho": row.get("spearman_rho", np.nan),
                "discrete_ordinal_r_squared": row.get("discrete_ordinal_pearson_r_squared", np.nan),
                "discrete_nmi": row.get("discrete_normalized_mutual_information", np.nan),
                "discrete_cramers_v": row.get("discrete_cramers_v", np.nan),
            }
        )
    if not dependencies.empty:
        strongest = dependencies.sort_values("normalized_mutual_information", ascending=False).iloc[
            0
        ]
        record.update(
            {
                "strongest_developmental_upstream": strongest["upstream_assay"],
                "strongest_developmental_downstream": strongest["downstream_assay"],
                "strongest_developmental_nmi": strongest["normalized_mutual_information"],
                "strongest_developmental_fdr": strongest["fdr_q"],
            }
        )
    assay_row = assay_exact[
        (assay_exact["objective"] == "cluster_nmi_recovery")
        & np.isclose(assay_exact["target_fraction_of_full"], 0.95)
    ]
    if not assay_row.empty:
        record["assay_level_95pct_cluster_nmi_minimum"] = int(assay_row.iloc[0]["minimal_n_assays"])
    module_row = module_exact[
        (module_exact["objective"] == "module_cluster_nmi_recovery")
        & np.isclose(module_exact["target_fraction_of_full"], 0.95)
    ]
    if not module_row.empty:
        record["module_level_95pct_cluster_nmi_minimum"] = int(
            module_row.iloc[0]["minimal_n_modules"]
        )
        record["module_level_component_assays"] = int(
            module_row.iloc[0]["minimal_component_assays"]
        )
    if not substitution_metadata.empty:
        row = substitution_metadata.iloc[0]
        record["substitutability_target_fraction"] = float(row["target_fraction_of_full"])
        record["n_equivalent_minimal_sets"] = int(row["n_equivalent_minimal_sets"])
        record["n_nonzero_swap_pairs"] = int(row["n_nonzero_swap_pairs"])
    return pd.DataFrame([record])


def main() -> None:
    args = parse_args()
    if not (0 < args.substitutability_target <= 1):
        raise ValueError("--substitutability-target must be in the interval (0, 1]")
    root = Path(args.project_root).expanduser().resolve()
    processed = root / "data" / "processed"
    raw_path = processed / "phenotype_matrix_raw.csv"
    discrete_path = processed / "phenotype_matrix_discrete.csv"
    catalog_path = processed / "assay_catalog.csv"
    if not raw_path.exists() or not discrete_path.exists():
        raise FileNotFoundError(
            "Processed phenotype matrices are missing. Run scripts/prepare_phenotype_matrix.py first."
        )
    raw = pd.read_csv(raw_path)
    discrete = pd.read_csv(discrete_path)
    catalog = pd.read_csv(catalog_path) if catalog_path.exists() else None
    assays = assay_columns(discrete, catalog)

    missing_growth = [
        assay
        for assay in GROWTH_ASSAYS
        if assay not in raw.columns or assay not in discrete.columns
    ]
    if missing_growth:
        raise ValueError(f"Expected continuous assays are missing: {missing_growth}")
    continuous = continuous_trait_association(raw, discrete, GROWTH_ASSAYS[0], GROWTH_ASSAYS[1])
    growth_sensitivity = growth_discretization_sensitivity(
        raw, discrete, GROWTH_ASSAYS[0], GROWTH_ASSAYS[1]
    )
    pairwise = pairwise_discrete_dependence(discrete, assays)

    module_path = Path(args.module_catalog).expanduser()
    if not module_path.is_absolute():
        module_path = root / module_path
    module_catalog = pd.read_csv(module_path)
    sexual_assays = [
        assay
        for assay in module_catalog.sort_values("stage_order")
        .loc[module_catalog["axis"] == "sexual_development", "component_assay"]
        .tolist()
        if assay in discrete.columns
    ]
    dependencies = developmental_dependency_table(discrete, sexual_assays)
    module_frame, module_definitions = build_module_matrix(discrete, module_catalog)
    module_metrics, module_exact, module_sets = exact_module_subset_metrics(
        module_frame, module_definitions
    )

    subsets, full_summary = all_subset_metrics(discrete, assays)
    assay_exact, minimal_sets = exact_minimal_sets(subsets)
    substitution_matrix, substitution_edges = all_assay_substitutability_matrix(
        minimal_sets, assays, target=args.substitutability_target
    )
    substitution_metadata, substitution_frequency = minimal_set_assay_frequency(
        minimal_sets, assays, target=args.substitutability_target
    )
    substitution_metadata["n_nonzero_swap_pairs"] = len(substitution_edges)
    summary = build_summary(
        discrete,
        assays,
        full_summary,
        continuous,
        dependencies,
        assay_exact,
        module_exact,
        substitution_metadata,
    )

    output = Path(args.output_workbook).expanduser()
    if not output.is_absolute():
        output = root / output
    write_workbook(
        output,
        {
            "analysis_summary": summary,
            "continuous_growth_relation": continuous,
            "growth_coding_sensitivity": growth_sensitivity,
            "pairwise_assay_dependence": pairwise,
            "developmental_dependencies": dependencies,
            "module_definitions": module_definitions,
            "module_all_subsets": module_metrics,
            "module_exact_minimal": module_exact,
            "module_minimal_sets": module_sets,
            "assay_exact_minimal": assay_exact,
            "assay_minimal_sets": minimal_sets,
            "substitution_metadata": substitution_metadata,
            "assay_presence_in_sets": substitution_frequency,
            "all_assay_substitution": substitution_matrix,
            "substitution_edges": substitution_edges,
        },
    )

    source_dir = root / "reports" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    continuous.to_csv(source_dir / "continuous_growth_relation.csv", index=False)
    growth_sensitivity.to_csv(source_dir / "growth_coding_sensitivity.csv", index=False)
    dependencies.to_csv(source_dir / "developmental_assay_dependencies.csv", index=False)
    module_metrics.to_csv(source_dir / "module_subset_metrics.csv", index=False)
    substitution_metadata.to_csv(source_dir / "substitution_metadata.csv", index=False)
    substitution_frequency.to_csv(source_dir / "assay_presence_in_minimal_sets.csv", index=False)
    substitution_matrix.to_csv(source_dir / "all_assay_substitutability.csv", index=False)
    substitution_edges.to_csv(source_dir / "assay_substitution_edges.csv", index=False)

    print(f"Assay-dependency analysis complete: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
