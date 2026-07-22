#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INTERPRETABLE_STATUSES = {"homokaryon_available", "heterokaryon_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate collection-selection and assay-dependency outputs"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--min-valid-status", type=int, default=1000)
    parser.add_argument("--min-explicit-status", type=int, default=1000)
    parser.add_argument(
        "--min-complete-homokaryon-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--expected-substitutability-target",
        type=float,
        default=0.90,
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required output is missing or empty: {path}")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return (
        series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
    )


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    master_path = root / "data" / "processed" / "collection_gene_master.csv"
    preparation_workbook = root / "reports" / "tables" / "collection_status_preparation.xlsx"
    collection_workbook = root / "reports" / "tables" / "collection_selection_results.xlsx"
    dependency_workbook = root / "reports" / "tables" / "assay_dependency_results.xlsx"
    for path in [
        master_path,
        preparation_workbook,
        collection_workbook,
        dependency_workbook,
    ]:
        require_file(path)

    master = pd.read_csv(master_path)
    required_master_columns = {
        "ko_gene_status",
        "explicit_ko_gene_status",
        "homokaryon_available",
        "heterokaryon_only",
        "explicit_homokaryon_available",
        "explicit_heterokaryon_only",
        "in_complete_10_assay_matrix",
        "has_ko_status_record",
        "in_ijdz836_model",
    }
    missing = sorted(required_master_columns - set(master.columns))
    if missing:
        raise RuntimeError(
            f"The collection master table is missing required columns: {missing}. "
            "Rerun scripts/prepare_collection_status.py."
        )

    valid_status = master["ko_gene_status"].isin(INTERPRETABLE_STATUSES)
    explicit_status = master["explicit_ko_gene_status"].isin(INTERPRETABLE_STATUSES)
    n_valid = int(valid_status.sum())
    n_explicit = int(explicit_status.sum())
    homokaryon = as_bool(master["homokaryon_available"])
    heterokaryon = as_bool(master["heterokaryon_only"])
    explicit_homo = as_bool(master["explicit_homokaryon_available"])
    explicit_hetero = as_bool(master["explicit_heterokaryon_only"])
    complete = as_bool(master["in_complete_10_assay_matrix"])
    known = as_bool(master["has_ko_status_record"])

    if n_valid < args.min_valid_status:
        raise RuntimeError(
            f"Only {n_valid} genes received interpretable primary KO status; "
            f"the required minimum is {args.min_valid_status}."
        )
    if n_explicit < args.min_explicit_status:
        raise RuntimeError(
            f"Only {n_explicit} genes received interpretable explicit KO status; "
            f"the required minimum is {args.min_explicit_status}."
        )
    if int(homokaryon.sum()) == 0 or int(heterokaryon.sum()) == 0:
        raise RuntimeError(
            "Both homokaryon-available and heterokaryon-only genes must be present in the primary analysis."
        )
    if int(explicit_homo.sum()) == 0 or int(explicit_hetero.sum()) == 0:
        raise RuntimeError(
            "Both homokaryon-available and heterokaryon-only genes must be present in the explicit-only sensitivity analysis."
        )

    complete_known = complete & known
    if int(complete_known.sum()) == 0:
        raise RuntimeError("No complete-matrix genes could be matched to KO status records.")
    complete_homo_fraction = float(homokaryon[complete_known].mean())
    if complete_homo_fraction < args.min_complete_homokaryon_fraction:
        raise RuntimeError(
            f"Only {complete_homo_fraction:.1%} of complete-matrix genes with known KO status were classified "
            "as homokaryon available. Inspect KO status parsing before analysis."
        )

    complete_summary = pd.read_excel(
        preparation_workbook,
        sheet_name="complete_status_summary",
    )
    complete_status_check = pd.read_excel(
        preparation_workbook,
        sheet_name="complete_status_check",
    )
    total_row = complete_summary[complete_summary["status_category"].eq("complete_matrix_total")]
    homo_row = complete_summary[
        complete_summary["status_category"].eq("current_homokaryon_available")
    ]
    if total_row.empty or homo_row.empty:
        raise RuntimeError(
            "Complete-matrix status summary is missing required total or homokaryon categories."
        )
    expected_discrepancies = int(total_row.iloc[0]["n_genes"]) - int(homo_row.iloc[0]["n_genes"])
    if expected_discrepancies != len(complete_status_check):
        raise RuntimeError(
            "Complete-matrix discrepancy counts are inconsistent between summary and status-check sheets."
        )

    mapping = pd.read_excel(
        preparation_workbook,
        sheet_name="identifier_mapping_summary",
    )
    required_mapping_stages = {
        "Reported by Dreyfuss et al. (2013)",
        "Unique NCU identifiers parsed from SBML",
        "Parsed identifiers in current genome annotation",
        "Parsed identifiers with interpretable KO status",
        "Included in complete 10-assay matrix",
    }
    observed_mapping_stages = set(mapping["mapping_stage"].astype(str))
    if not required_mapping_stages.issubset(observed_mapping_stages):
        raise RuntimeError(
            "Identifier mapping summary does not contain all required mapping stages."
        )

    enrichment = pd.read_excel(
        collection_workbook,
        sheet_name="selection_enrichment",
    )
    observed_scopes = set(enrichment["status_scope"].dropna().astype(str))
    if not {"all_interpretable_calls", "explicit_only"}.issubset(observed_scopes):
        raise RuntimeError(
            "Collection-selection output must contain primary and explicit-only status scopes."
        )

    continuous = pd.read_excel(
        dependency_workbook,
        sheet_name="continuous_growth_relation",
    )
    if continuous.empty or not (0 <= float(continuous.iloc[0]["pearson_r_squared"]) <= 1):
        raise RuntimeError("Continuous growth-trait association output is invalid.")

    substitution_metadata = pd.read_excel(
        dependency_workbook,
        sheet_name="substitution_metadata",
    )
    if substitution_metadata.empty:
        raise RuntimeError("Substitutability metadata are missing.")
    target = float(substitution_metadata.iloc[0]["target_fraction_of_full"])
    if not np.isclose(target, args.expected_substitutability_target):
        raise RuntimeError(
            f"Substitutability target is {target:.3f}; expected {args.expected_substitutability_target:.3f}."
        )
    n_sets = int(substitution_metadata.iloc[0]["n_equivalent_minimal_sets"])
    if n_sets < 2:
        raise RuntimeError(
            f"Only {n_sets} equivalent minimal set was found at the substitutability target; "
            "one-for-one substitutions cannot be interpreted."
        )

    substitution = pd.read_excel(
        dependency_workbook,
        sheet_name="all_assay_substitution",
    )
    if len(substitution) != 10:
        raise RuntimeError(
            f"Expected a 10-assay substitutability matrix; found {len(substitution)} rows."
        )
    matrix_values = substitution.drop(columns=["assay"]).to_numpy(dtype=float)
    if not np.any(matrix_values > 0):
        raise RuntimeError(
            "The all-assay substitutability matrix contains no nonzero swaps. "
            "Use a recovery target with multiple equivalent minimal sets."
        )

    frequency = pd.read_excel(
        dependency_workbook,
        sheet_name="assay_presence_in_sets",
    )
    if len(frequency) != 10:
        raise RuntimeError(
            f"Expected assay-frequency output for 10 assays; found {len(frequency)} rows."
        )

    print("Validation passed.")
    print(
        f"Primary KO statuses: {n_valid:,} "
        f"({int(homokaryon.sum()):,} homokaryon available; {int(heterokaryon.sum()):,} heterokaryon only)"
    )
    print(
        f"Explicit-only KO statuses: {n_explicit:,} "
        f"({int(explicit_homo.sum()):,} homokaryon available; {int(explicit_hetero.sum()):,} heterokaryon only)"
    )
    print(f"Complete-matrix genes with current homokaryon status: {complete_homo_fraction:.1%}")
    print(
        f"Complete-matrix current-status discrepancies retained for status check: {len(complete_status_check):,}"
    )
    print(f"Raw growth-trait R^2: {float(continuous.iloc[0]['pearson_r_squared']):.4f}")
    print(
        "Discretized ordinal growth-trait R^2: "
        f"{float(continuous.iloc[0]['discrete_ordinal_pearson_r_squared']):.4f}"
    )
    print(
        f"Substitutability target: {target:.0%}; equivalent minimal sets: {n_sets}; "
        f"nonzero swap pairs: {int(substitution_metadata.iloc[0]['n_nonzero_swap_pairs'])}"
    )
    print("All 10 assays are present in the substitutability matrix.")


if __name__ == "__main__":
    main()
