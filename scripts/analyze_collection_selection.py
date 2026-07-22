#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenome_architecture.collection import bh_fdr, binary_concordance, fisher_enrichment

CLASS_LABELS = {
    "in_ijdz836_model": "iJDZ836 model genes",
    "in_carrillo_metabolic_sheet": "Curated metabolic genes",
    "metabolic_union": "Metabolic genes (union)",
    "dreyfuss_predicted_essential": "Dreyfuss-predicted essential genes",
    "transcription_factor": "Transcription factors",
    "gpcr": "GPCR genes",
    "ser_thr_kinase": "Ser/Thr kinases",
    "phosphatase": "Phosphatases",
    "transmembrane_any": "Transmembrane genes",
    "transmembrane_5plus": "Genes with at least five TM helices",
    "high_phosphorylation_q75": "High phosphorylation-site genes",
    "has_yeast_ortholog": "Genes with yeast orthologs",
}

STATUS_SCOPES = {
    "all_interpretable_calls": {
        "status_column": "ko_gene_status",
        "homokaryon_column": "homokaryon_available",
        "heterokaryon_column": "heterokaryon_only",
        "description": "Explicit calls and archived available-strain records interpreted as homokaryons",
    },
    "explicit_only": {
        "status_column": "explicit_ko_gene_status",
        "homokaryon_column": "explicit_homokaryon_available",
        "heterokaryon_column": "explicit_heterokaryon_only",
        "description": "Records with an explicit karyon-status call",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze knockout-collection recovery and phenotype-matrix inclusion"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--min-class-size", type=int, default=5)
    parser.add_argument(
        "--output-workbook", default="reports/tables/collection_selection_results.xlsx"
    )
    return parser.parse_args()


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


def as_bool(series: pd.Series) -> pd.Series:
    """Convert booleans serialized through CSV without treating 'False' as true."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def scope_masks(master: pd.DataFrame, status_scope: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    if status_scope not in STATUS_SCOPES:
        raise ValueError(f"Unknown status scope: {status_scope}")
    config = STATUS_SCOPES[status_scope]
    status = master[config["status_column"]].fillna("").astype(str)
    interpretable = status.isin(["homokaryon_available", "heterokaryon_only"])
    homokaryon = as_bool(master[config["homokaryon_column"]])
    heterokaryon = as_bool(master[config["heterokaryon_column"]])
    return interpretable, homokaryon, heterokaryon


def category_metrics(master: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    subset = master[mask].copy()
    in_model = as_bool(subset["in_ijdz836_model"]) if len(subset) else pd.Series(dtype=bool)
    predicted = (
        as_bool(subset["dreyfuss_predicted_essential"]) if len(subset) else pd.Series(dtype=bool)
    )
    return {
        "n_genes": len(subset),
        "n_ijdz836_genes": int(in_model.sum()) if len(subset) else 0,
        "fraction_ijdz836": float(in_model.mean()) if len(subset) else np.nan,
        "n_predicted_essential": int(predicted.sum()) if len(subset) else 0,
        "fraction_predicted_essential_among_model_genes": (
            float(predicted[in_model].mean()) if len(subset) and in_model.sum() else np.nan
        ),
    }


def collection_overview(master: pd.DataFrame) -> pd.DataFrame:
    primary_interpretable, homokaryon, heterokaryon = scope_masks(
        master,
        "all_interpretable_calls",
    )
    complete = as_bool(master["in_complete_10_assay_matrix"])
    current_match = complete & homokaryon
    current_status_check = complete & ~homokaryon
    categories = [
        (
            1,
            "Genome annotation",
            "universe",
            "",
            as_bool(master["in_genome_annotation"]),
        ),
        (
            2,
            "KO status recorded",
            "status_universe",
            "Genome annotation",
            as_bool(master["has_ko_status_record"]),
        ),
        (
            3,
            "Interpretable KO status",
            "status_universe",
            "KO status recorded",
            primary_interpretable,
        ),
        (
            4,
            "Homokaryon available",
            "status_branch",
            "Interpretable KO status",
            homokaryon,
        ),
        (
            5,
            "Heterokaryon only",
            "status_branch",
            "Interpretable KO status",
            heterokaryon,
        ),
        (
            6,
            "Complete 10-assay matrix",
            "phenome",
            "Carrillo complete-data matrix",
            complete,
        ),
        (
            7,
            "Complete matrix: current homokaryon match",
            "phenome_match",
            "Complete 10-assay matrix",
            current_match,
        ),
        (
            8,
            "Complete matrix: current-status check",
            "status_check",
            "Complete 10-assay matrix",
            current_status_check,
        ),
    ]
    rows: list[dict[str, object]] = []
    for order, label, category_type, parent, mask in categories:
        rows.append(
            {
                "display_order": order,
                "collection_category": label,
                "category_type": category_type,
                "parent_category": parent,
                **category_metrics(master, mask),
            }
        )
    return pd.DataFrame(rows)


def status_by_class(
    master: pd.DataFrame,
    class_columns: list[str],
    status_scope: str,
) -> pd.DataFrame:
    interpretable, homokaryon, heterokaryon = scope_masks(master, status_scope)
    valid = master[interpretable].copy()
    valid_homo = homokaryon.loc[valid.index]
    valid_hetero = heterokaryon.loc[valid.index]
    rows: list[dict[str, object]] = []
    for column in class_columns:
        data = valid
        homo_mask = valid_homo
        hetero_mask = valid_hetero
        universe = "all genes with interpretable KO status"
        if column == "dreyfuss_predicted_essential":
            model_mask = as_bool(valid["in_ijdz836_model"])
            data = valid[model_mask].copy()
            homo_mask = valid_homo.loc[data.index]
            hetero_mask = valid_hetero.loc[data.index]
            universe = "iJDZ836 genes with interpretable KO status"
        member = as_bool(data[column])
        for status_label, status_mask in [
            ("homokaryon_available", homo_mask),
            ("heterokaryon_only", hetero_mask),
        ]:
            rows.append(
                {
                    "status_scope": status_scope,
                    "status_scope_description": STATUS_SCOPES[status_scope]["description"],
                    "analysis_universe": universe,
                    "gene_class": column,
                    "gene_class_label": CLASS_LABELS.get(column, column),
                    "ko_gene_status": status_label,
                    "n_genes_in_status": int(status_mask.sum()),
                    "n_class_genes": int((member & status_mask).sum()),
                    "class_fraction_in_status": (
                        float(member[status_mask].mean()) if status_mask.sum() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def enrichment_table(
    master: pd.DataFrame,
    class_columns: list[str],
    gate: str,
    min_class_size: int,
    status_scope: str,
) -> pd.DataFrame:
    interpretable, homokaryon, heterokaryon = scope_masks(master, status_scope)
    if gate == "homokaryon_recovery":
        data = master[interpretable].copy()
        outcome = heterokaryon.loc[data.index]
        outcome_label = "heterokaryon_only"
        comparison = "heterokaryon-only vs homokaryon-available"
    elif gate == "complete_phenome_inclusion":
        data = master[homokaryon].copy()
        outcome = as_bool(data["in_complete_10_assay_matrix"])
        outcome_label = "included_in_complete_10_assay_matrix"
        comparison = "complete 10-assay inclusion among homokaryon-available genes"
    else:
        raise ValueError(f"Unknown gate: {gate}")

    rows: list[dict[str, object]] = []
    for column in class_columns:
        class_data = data
        class_outcome = outcome
        universe = "all genes eligible for the selection gate"
        if column == "dreyfuss_predicted_essential":
            model_mask = as_bool(data["in_ijdz836_model"])
            class_data = data[model_mask].copy()
            class_outcome = outcome.loc[class_data.index]
            universe = "iJDZ836 genes eligible for the selection gate"
        member = as_bool(class_data[column])
        if member.sum() < min_class_size or (~member).sum() < min_class_size:
            continue
        stats = fisher_enrichment(member, class_outcome)
        rows.append(
            {
                "status_scope": status_scope,
                "status_scope_description": STATUS_SCOPES[status_scope]["description"],
                "selection_gate": gate,
                "analysis_universe": universe,
                "comparison": comparison,
                "outcome": outcome_label,
                "gene_class": column,
                "gene_class_label": CLASS_LABELS.get(column, column),
                "n_genes_analyzed": len(class_data),
                "n_class_genes": int(member.sum()),
                **stats,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q"] = bh_fdr(out["fisher_p"])
        out["direction"] = np.where(
            out["log2_odds_ratio_corrected"] > 0,
            "enriched_for_outcome",
            "depleted_for_outcome",
        )
        out = out.sort_values(["fdr_q", "fisher_p"]).reset_index(drop=True)
    return out


def essentiality_concordance(
    master: pd.DataFrame,
    status_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interpretable, _, heterokaryon = scope_masks(master, status_scope)
    data = master[interpretable & as_bool(master["in_ijdz836_model"])].copy()
    predicted = as_bool(data["dreyfuss_predicted_essential"])
    observed = heterokaryon.loc[data.index]
    summary = pd.DataFrame(
        [
            {
                "status_scope": status_scope,
                "status_scope_description": STATUS_SCOPES[status_scope]["description"],
                "n_model_genes_analyzed": len(data),
                **binary_concordance(predicted, observed),
            }
        ]
    )
    counts = pd.crosstab(
        np.where(predicted, "Predicted essential", "Not predicted essential"),
        np.where(observed, "Heterokaryon only", "Homokaryon available"),
    )
    counts.index.name = "Dreyfuss prediction"
    counts = counts.reset_index()
    counts.insert(0, "status_scope", status_scope)
    return summary, counts


def compare_status_scopes(enrichment: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        "selection_gate",
        "analysis_universe",
        "comparison",
        "outcome",
        "gene_class",
        "gene_class_label",
    ]
    value_columns = [
        "n_genes_analyzed",
        "n_class_genes",
        "log2_odds_ratio_corrected",
        "fisher_p",
        "fdr_q",
    ]
    primary = enrichment[enrichment["status_scope"] == "all_interpretable_calls"][
        key_columns + value_columns
    ].copy()
    explicit = enrichment[enrichment["status_scope"] == "explicit_only"][
        key_columns + value_columns
    ].copy()
    primary = primary.rename(columns={column: f"primary_{column}" for column in value_columns})
    explicit = explicit.rename(columns={column: f"explicit_{column}" for column in value_columns})
    out = primary.merge(explicit, on=key_columns, how="outer")
    if out.empty:
        return out
    out["delta_log2_odds_ratio_explicit_minus_primary"] = (
        out["explicit_log2_odds_ratio_corrected"] - out["primary_log2_odds_ratio_corrected"]
    )
    out["same_effect_direction"] = np.sign(out["primary_log2_odds_ratio_corrected"]) == np.sign(
        out["explicit_log2_odds_ratio_corrected"]
    )
    out["significant_in_both_scopes"] = (out["primary_fdr_q"] < 0.05) & (
        out["explicit_fdr_q"] < 0.05
    )
    out["directionally_stable"] = out["same_effect_direction"] & out["significant_in_both_scopes"]
    return out.sort_values(["selection_gate", "gene_class_label"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    master_path = root / "data" / "processed" / "collection_gene_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(
            "collection_gene_master.csv not found. Run scripts/prepare_collection_status.py first."
        )
    master = pd.read_csv(master_path)
    for column in set(CLASS_LABELS) | {
        "in_genome_annotation",
        "has_ko_status_record",
        "homokaryon_available",
        "heterokaryon_only",
        "explicit_homokaryon_available",
        "explicit_heterokaryon_only",
        "in_complete_10_assay_matrix",
    }:
        if column not in master.columns:
            master[column] = False
        master[column] = as_bool(master[column])
    if "explicit_ko_gene_status" not in master.columns:
        raise ValueError(
            "explicit_ko_gene_status is missing. Run scripts/prepare_collection_status.py again."
        )

    class_columns = [column for column in CLASS_LABELS if column in master.columns]
    overview = collection_overview(master)
    status_tables: list[pd.DataFrame] = []
    enrichment_tables: list[pd.DataFrame] = []
    concordance_tables: list[pd.DataFrame] = []
    concordance_count_tables: list[pd.DataFrame] = []
    for status_scope in STATUS_SCOPES:
        status_tables.append(status_by_class(master, class_columns, status_scope))
        for gate in ("homokaryon_recovery", "complete_phenome_inclusion"):
            enrichment_tables.append(
                enrichment_table(master, class_columns, gate, args.min_class_size, status_scope)
            )
        concordance, counts = essentiality_concordance(master, status_scope)
        concordance_tables.append(concordance)
        concordance_count_tables.append(counts)

    status_classes = pd.concat(status_tables, ignore_index=True)
    enrichment = pd.concat(enrichment_tables, ignore_index=True)
    concordance = pd.concat(concordance_tables, ignore_index=True)
    concordance_counts = pd.concat(concordance_count_tables, ignore_index=True)
    scope_comparison = compare_status_scopes(enrichment)

    preparation_workbook = root / "reports" / "tables" / "collection_status_preparation.xlsx"
    identifier_mapping = (
        pd.read_excel(preparation_workbook, sheet_name="identifier_mapping_summary")
        if preparation_workbook.exists()
        else pd.DataFrame()
    )
    complete_status_summary = (
        pd.read_excel(preparation_workbook, sheet_name="complete_status_summary")
        if preparation_workbook.exists()
        else pd.DataFrame()
    )
    complete_status_check = (
        pd.read_excel(preparation_workbook, sheet_name="complete_status_check")
        if preparation_workbook.exists()
        else pd.DataFrame()
    )

    output = Path(args.output_workbook).expanduser()
    if not output.is_absolute():
        output = root / output
    write_workbook(
        output,
        {
            "collection_overview": overview,
            "status_by_gene_class": status_classes,
            "selection_enrichment": enrichment,
            "status_scope_comparison": scope_comparison,
            "essentiality_concordance": concordance,
            "essentiality_counts": concordance_counts,
            "identifier_mapping": identifier_mapping,
            "complete_status_summary": complete_status_summary,
            "complete_status_check": complete_status_check,
            "gene_master": master,
        },
    )

    source_dir = root / "reports" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(source_dir / "collection_overview.csv", index=False)
    status_classes.to_csv(source_dir / "collection_status_by_gene_class.csv", index=False)
    enrichment.to_csv(source_dir / "collection_selection_enrichment.csv", index=False)
    scope_comparison.to_csv(source_dir / "collection_status_scope_comparison.csv", index=False)
    concordance.to_csv(source_dir / "essentiality_concordance.csv", index=False)
    concordance_counts.to_csv(source_dir / "essentiality_concordance_counts.csv", index=False)

    print(f"Collection-selection analysis complete: {output}")
    key = enrichment[
        (enrichment["selection_gate"] == "homokaryon_recovery")
        & enrichment["gene_class"].isin(["in_ijdz836_model", "dreyfuss_predicted_essential"])
    ]
    if not key.empty:
        print(
            key[
                [
                    "status_scope",
                    "gene_class_label",
                    "log2_odds_ratio_corrected",
                    "fisher_p",
                    "fdr_q",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
