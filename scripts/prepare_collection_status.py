#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT_SRC))

from phenome_architecture.collection import (  # noqa: E402
    aggregate_gene_status,
    apply_exclusions,
    canonical_ncu,
    load_exclusions,
    parse_dreyfuss_essentiality,
    parse_gff_gene_ids,
    parse_ko_workbook,
    parse_sbml_genes,
)
from phenome_architecture.io import read_excel_table  # noqa: E402


NCU_RE = re.compile(r"NCU\d{5}", re.IGNORECASE)
INTERPRETABLE_STATUSES = {"homokaryon_available", "heterokaryon_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Neurospora knockout-collection status and genome-wide annotations"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--ko-workbook",
        default="",
        help="Official KO availability workbook; auto-detected if omitted",
    )
    parser.add_argument("--ko-status-exclusions", default="config/ko_status_exclusions.csv")
    parser.add_argument(
        "--dreyfuss-essential-genes",
        default="config/dreyfuss_essential_genes.csv",
    )
    parser.add_argument(
        "--reported-model-gene-count",
        type=int,
        default=836,
        help="Gene count reported for iJDZ836 in Dreyfuss et al. (2013)",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return (
        series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
    )


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def find_first(root: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def canonicalize_id_column(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    source = next((column for column in candidates if column in out.columns), None)
    if source is None:
        raise ValueError(f"No NCU identifier column found. Available columns: {list(out.columns)}")
    out["ncu_id"] = out[source].map(canonical_ncu)
    return out[out["ncu_id"] != ""].copy()


def build_carrillo_annotations(
    main_file: Path | None,
    yeast_file: Path | None,
) -> pd.DataFrame:
    if main_file is None:
        return pd.DataFrame(columns=["ncu_id"])
    frames: list[pd.DataFrame] = []

    product = read_excel_table(main_file, "Product descriptions")
    product = canonicalize_id_column(product, ["NCU Number", "Gene Number", "NCU"])
    description_col = next(
        (
            column
            for column in ["Product Description", "product_description", "Description"]
            if column in product.columns
        ),
        None,
    )
    product_out = product[["ncu_id"]].copy()
    product_out["product_description"] = (
        product[description_col].astype(str) if description_col else ""
    )
    frames.append(product_out.drop_duplicates("ncu_id"))

    features = read_excel_table(main_file, "Protein features")
    features = canonicalize_id_column(features, ["NCU Number", "Gene Number", "NCU"])
    feature_out = features[["ncu_id"]].copy()
    rename_map = {
        "Gene Classification": "gene_classification",
        "# of Phosphorylation Sites": "n_phosphorylation_sites",
        "# Transmembrane Helices": "n_transmembrane_helices",
    }
    for source, target in rename_map.items():
        feature_out[target] = features[source] if source in features.columns else np.nan
    frames.append(feature_out.drop_duplicates("ncu_id"))

    metabolic = read_excel_table(main_file, "Metabolic genes")
    metabolic = canonicalize_id_column(metabolic, ["NCU Number", "Gene Number", "NCU"])
    metabolic_out = metabolic[["ncu_id"]].drop_duplicates()
    metabolic_out["in_carrillo_metabolic_sheet"] = True
    frames.append(metabolic_out)

    annotation = frames[0]
    for frame in frames[1:]:
        annotation = annotation.merge(frame, on="ncu_id", how="outer")

    if yeast_file is not None:
        try:
            yeast = read_excel_table(
                yeast_file,
                "Yeast Ortholog Summary",
                required="Gene Number",
            )
            yeast = canonicalize_id_column(
                yeast,
                ["Gene Number", "NCU Number", "NCU"],
            )
            ortholog_cols = [
                column
                for column in yeast.columns
                if str(column).lower().startswith("yeast ortholog")
            ]
            yeast_out = yeast[["ncu_id"]].copy()
            yeast_out["has_yeast_ortholog"] = (
                yeast[ortholog_cols].notna().any(axis=1) if ortholog_cols else False
            )
            annotation = annotation.merge(
                yeast_out.drop_duplicates("ncu_id"),
                on="ncu_id",
                how="outer",
            )
        except Exception as exc:
            print(
                f"warning: yeast ortholog sheet was not parsed: {exc}",
                file=sys.stderr,
            )

    annotation["in_carrillo_metabolic_sheet"] = as_bool(
        annotation.get(
            "in_carrillo_metabolic_sheet",
            pd.Series(False, index=annotation.index),
        )
    )
    annotation["has_yeast_ortholog"] = as_bool(
        annotation.get(
            "has_yeast_ortholog",
            pd.Series(False, index=annotation.index),
        )
    )
    return annotation.drop_duplicates("ncu_id").reset_index(drop=True)


def add_annotation_classes(annotation: pd.DataFrame) -> pd.DataFrame:
    out = annotation.copy()
    description = (
        out.get("product_description", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    classification = (
        out.get("gene_classification", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    phosphorylation_source = (
        out["n_phosphorylation_sites"]
        if "n_phosphorylation_sites" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    transmembrane_source = (
        out["n_transmembrane_helices"]
        if "n_transmembrane_helices" in out.columns
        else pd.Series(0, index=out.index)
    )
    phosphorylation = pd.to_numeric(phosphorylation_source, errors="coerce")
    transmembrane = pd.to_numeric(transmembrane_source, errors="coerce").fillna(0)
    out["transcription_factor"] = classification.eq("TF") | description.str.contains(
        "transcription factor|zn2cys6|c2h2|bzip|myb|homeobox|ap2",
        regex=True,
    )
    out["gpcr"] = classification.eq("GPCR") | description.str.contains(
        "g-protein-coupled|gpcr",
        regex=True,
    )
    out["ser_thr_kinase"] = classification.eq("STKIN") | description.str.contains(
        "serine/threonine|protein kinase|map kinase|kinase",
        regex=True,
    )
    out["phosphatase"] = classification.eq("PASE") | description.str.contains(
        "phosphatase",
        regex=True,
    )
    out["transmembrane_any"] = transmembrane >= 1
    out["transmembrane_5plus"] = transmembrane >= 5
    threshold = phosphorylation.dropna().quantile(0.75) if phosphorylation.notna().any() else np.inf
    out["high_phosphorylation_q75"] = phosphorylation.fillna(0) >= threshold
    return out


def phenotype_gene_sets(root: Path) -> tuple[set[str], set[str]]:
    """Return genes in the processed raw matrix and in its complete-data subset.

    The core workflow currently writes a complete-data raw matrix, so the two
    sets are often identical. They are retained separately to avoid calling the
    processed 1,168-gene matrix the full set of all genes phenotyped by Carrillo.
    """
    raw_path = first_existing(
        [
            root / "data" / "processed" / "phenotype_matrix_raw.csv",
            root / "phenotype_matrix_raw.csv",
        ]
    ) or find_first(root, ["**/phenotype_matrix_raw.csv"])
    discrete_path = first_existing(
        [
            root / "data" / "processed" / "phenotype_matrix_discrete.csv",
            root / "phenotype_matrix_discrete.csv",
        ]
    ) or find_first(root, ["**/phenotype_matrix_discrete.csv"])

    processed_genes: set[str] = set()
    complete_genes: set[str] = set()
    if raw_path is not None:
        raw = pd.read_csv(raw_path)
        if "ncu_id" in raw.columns:
            raw["ncu_id"] = raw["ncu_id"].map(canonical_ncu)
            processed_genes = set(raw["ncu_id"]) - {""}
            phenotype_cols = [
                column
                for column in raw.columns
                if column not in {"ncu_id", "gene_name", "published_cluster"}
            ]
            complete = raw.dropna(subset=phenotype_cols) if phenotype_cols else raw
            complete_genes = set(complete["ncu_id"]) - {""}
    if discrete_path is not None and not complete_genes:
        discrete = pd.read_csv(discrete_path)
        if "ncu_id" in discrete.columns:
            complete_genes = set(discrete["ncu_id"].map(canonical_ncu)) - {""}
            processed_genes |= complete_genes
    return processed_genes, complete_genes


def complete_matrix_status_check(
    master: pd.DataFrame,
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = as_bool(master["in_complete_10_assay_matrix"])
    homokaryon = as_bool(master["homokaryon_available"])
    status_rows = master[complete & ~homokaryon].copy()

    def status_category(row: pd.Series) -> str:
        if not bool(row.get("has_ko_status_record", False)):
            return "no_current_ko_status_record"
        status = str(row.get("ko_gene_status", ""))
        if status == "heterokaryon_only":
            return "current_heterokaryon_only"
        if status == "invalid_or_unavailable_only":
            return "current_invalid_or_unavailable_only"
        if status == "unresolved":
            return "current_status_unresolved"
        return "other_nonhomokaryon_status"

    if not status_rows.empty:
        status_rows.insert(1, "status_category", status_rows.apply(status_category, axis=1))
        status_rows.insert(2, "status_check_required", True)
        status_rows.insert(
            3,
            "status_note",
            "Current availability status may differ from the strain status recorded when phenotypes were collected.",
        )
        status_rows = status_rows.sort_values(["status_category", "ncu_id"]).reset_index(drop=True)
    else:
        status_rows = pd.DataFrame(
            columns=[
                "ncu_id",
                "status_category",
                "status_check_required",
                "status_note",
            ]
        )

    category_counts = (
        status_rows.groupby("status_category", dropna=False).size().rename("n_genes").reset_index()
        if not status_rows.empty
        else pd.DataFrame(columns=["status_category", "n_genes"])
    )
    summary_rows = [
        {
            "status_category": "complete_matrix_total",
            "n_genes": int(complete.sum()),
        },
        {
            "status_category": "current_homokaryon_available",
            "n_genes": int((complete & homokaryon).sum()),
        },
    ]
    summary = pd.concat(
        [pd.DataFrame(summary_rows), category_counts],
        ignore_index=True,
    )
    summary["fraction_of_complete_matrix"] = summary["n_genes"] / max(int(complete.sum()), 1)

    status_ids = set(status_rows.get("ncu_id", pd.Series(dtype=str)).astype(str))
    status_records = records[records["ncu_id"].isin(status_ids)].copy()
    status_records = status_records.sort_values(
        ["ncu_id", "status_confidence", "record_status", "source_row"]
    ).reset_index(drop=True)
    return summary, status_rows, status_records


def identifier_mapping_tables(
    master: pd.DataFrame,
    sbml_genes: set[str],
    essential_genes: set[str],
    genome_genes: set[str],
    reported_model_gene_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    in_model = as_bool(master["in_ijdz836_model"])
    predicted_essential = as_bool(master["dreyfuss_predicted_essential"])
    has_status = as_bool(master["has_ko_status_record"])
    interpretable = master["ko_gene_status"].isin(INTERPRETABLE_STATUSES)
    homo = as_bool(master["homokaryon_available"])
    hetero = as_bool(master["heterokaryon_only"])
    complete = as_bool(master["in_complete_10_assay_matrix"])
    in_genome = as_bool(master["in_genome_annotation"])

    rows = [
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Reported by Dreyfuss et al. (2013)",
            "n_genes": int(reported_model_gene_count),
            "count_source": "published model description",
        },
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Unique NCU identifiers parsed from SBML",
            "n_genes": len(sbml_genes),
            "count_source": "SBML parsing",
        },
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Parsed identifiers in current genome annotation",
            "n_genes": int((in_model & in_genome).sum()),
            "count_source": "NC12 GFF mapping",
        },
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Parsed identifiers with any KO status record",
            "n_genes": int((in_model & has_status).sum()),
            "count_source": "current KO availability workbook",
        },
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Parsed identifiers with interpretable KO status",
            "n_genes": int((in_model & interpretable).sum()),
            "count_source": "current KO availability workbook",
        },
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Homokaryon available",
            "n_genes": int((in_model & homo).sum()),
            "count_source": "current KO availability workbook",
        },
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Heterokaryon only",
            "n_genes": int((in_model & hetero).sum()),
            "count_source": "current KO availability workbook",
        },
        {
            "mapping_series": "iJDZ836 model genes",
            "mapping_stage": "Included in complete 10-assay matrix",
            "n_genes": int((in_model & complete).sum()),
            "count_source": "processed Carrillo matrix",
        },
        {
            "mapping_series": "Dreyfuss predicted-essential genes",
            "mapping_stage": "Unique identifiers parsed from supplementary output",
            "n_genes": len(essential_genes),
            "count_source": "supplementary workbook parsing",
        },
        {
            "mapping_series": "Dreyfuss predicted-essential genes",
            "mapping_stage": "Parsed identifiers in current genome annotation",
            "n_genes": int((predicted_essential & in_genome).sum()),
            "count_source": "NC12 GFF mapping",
        },
        {
            "mapping_series": "Dreyfuss predicted-essential genes",
            "mapping_stage": "Parsed identifiers with interpretable KO status",
            "n_genes": int((predicted_essential & interpretable).sum()),
            "count_source": "current KO availability workbook",
        },
        {
            "mapping_series": "Dreyfuss predicted-essential genes",
            "mapping_stage": "Homokaryon available",
            "n_genes": int((predicted_essential & homo).sum()),
            "count_source": "current KO availability workbook",
        },
        {
            "mapping_series": "Dreyfuss predicted-essential genes",
            "mapping_stage": "Heterokaryon only",
            "n_genes": int((predicted_essential & hetero).sum()),
            "count_source": "current KO availability workbook",
        },
        {
            "mapping_series": "Dreyfuss predicted-essential genes",
            "mapping_stage": "Included in complete 10-assay matrix",
            "n_genes": int((predicted_essential & complete).sum()),
            "count_source": "processed Carrillo matrix",
        },
    ]
    summary = pd.DataFrame(rows)

    mapping_check = master[in_model | predicted_essential].copy()

    def mapping_issue(row: pd.Series) -> str:
        issues: list[str] = []
        if not bool(row.get("in_genome_annotation", False)):
            issues.append("not_in_current_genome_annotation")
        if not bool(row.get("has_ko_status_record", False)):
            issues.append("no_current_ko_status_record")
        elif str(row.get("ko_gene_status", "")) not in INTERPRETABLE_STATUSES:
            issues.append("ko_status_not_interpretable")
        if not bool(row.get("in_complete_10_assay_matrix", False)):
            issues.append("not_in_complete_10_assay_matrix")
        return ";".join(issues) if issues else "fully_mapped_to_complete_matrix"

    mapping_check.insert(1, "mapping_issue", mapping_check.apply(mapping_issue, axis=1))
    mapping_check.insert(
        2,
        "mapping_source",
        np.where(
            in_model.loc[mapping_check.index] & predicted_essential.loc[mapping_check.index],
            "iJDZ836_model;predicted_essential",
            np.where(
                in_model.loc[mapping_check.index],
                "iJDZ836_model",
                "predicted_essential",
            ),
        ),
    )
    mapping_check = mapping_check.sort_values(
        ["mapping_source", "mapping_issue", "ncu_id"]
    ).reset_index(drop=True)
    return summary, mapping_check


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.book[name[:31]]
            sheet.freeze_panes = "A2"
            for column_index, column in enumerate(frame.columns, start=1):
                values = frame[column].head(300).astype("string").fillna("").tolist()
                width = min(
                    max([len(str(column))] + [len(value) for value in values]) + 2,
                    55,
                )
                sheet.column_dimensions[sheet.cell(1, column_index).column_letter].width = max(
                    width, 10
                )
            for cell in sheet[1]:
                cell.style = "Headline 4"


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    table_dir = root / "reports" / "tables"
    source_dir = root / "reports" / "source_data"
    table_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    ko_path = (
        Path(args.ko_workbook).expanduser()
        if args.ko_workbook
        else first_existing(
            [
                root / "data" / "raw" / "neurospora_ko" / "Available_KO_Strains.xlsx",
                root / "data" / "raw" / "neurospora_ko" / "UpdatedPlateList1217.xls",
            ]
        )
    )
    if ko_path is None or not ko_path.exists():
        raise FileNotFoundError(
            "KO availability workbook is missing. Run scripts/download_sources.py or place "
            "Available_KO_Strains.xlsx in data/raw/neurospora_ko/."
        )
    if not ko_path.is_absolute():
        ko_path = root / ko_path

    exclusions_path = Path(args.ko_status_exclusions)
    if not exclusions_path.is_absolute():
        exclusions_path = root / exclusions_path
    records = parse_ko_workbook(ko_path)
    records = apply_exclusions(records, load_exclusions(exclusions_path))
    gene_status = aggregate_gene_status(records)

    carrillo_main = first_existing(
        [root / "data" / "raw" / "carrillo_2020" / "12864_2020_7131_MOESM1_ESM.xlsx"]
    ) or find_first(root, ["**/12864_2020_7131_MOESM1_ESM.xlsx"])
    carrillo_yeast = first_existing(
        [root / "data" / "raw" / "carrillo_2020" / "12864_2020_7131_MOESM6_ESM.xlsx"]
    ) or find_first(root, ["**/12864_2020_7131_MOESM6_ESM.xlsx"])
    annotation = add_annotation_classes(build_carrillo_annotations(carrillo_main, carrillo_yeast))

    sbml_path = first_existing(
        [root / "data" / "raw" / "dreyfuss_2013" / "pcbi.1003126.s001.xml"]
    ) or find_first(root, ["**/pcbi.1003126.s001.xml"])
    sbml_genes = parse_sbml_genes(sbml_path)

    dreyfuss_path = first_existing(
        [
            root / "data" / "raw" / "dreyfuss_2013" / "pcbi.1003126.s008.xls",
            root / "data" / "raw" / "dreyfuss_2013" / "pcbi.1003126.s008.xlsx",
        ]
    ) or find_first(root, ["**/pcbi.1003126.s008.xls", "**/pcbi.1003126.s008.xlsx"])
    essential_gene_list = Path(args.dreyfuss_essential_genes)
    if not essential_gene_list.is_absolute():
        essential_gene_list = root / essential_gene_list
    dreyfuss_essential = parse_dreyfuss_essentiality(
        dreyfuss_path,
        essential_gene_list if essential_gene_list.exists() else None,
    )
    essential_genes = set(dreyfuss_essential.get("ncu_id", []))

    gff_path = first_existing(
        [root / "data" / "raw" / "ensembl_fungi" / "Neurospora_crassa.NC12.62.gff3.gz"]
    ) or find_first(root, ["**/Neurospora_crassa*.gff3.gz", "**/Neurospora_crassa*.gff3"])
    genome_genes = parse_gff_gene_ids(gff_path)
    processed_genes, complete_genes = phenotype_gene_sets(root)

    all_genes = sorted(
        genome_genes
        | set(gene_status["ncu_id"])
        | set(annotation.get("ncu_id", []))
        | sbml_genes
        | essential_genes
        | processed_genes
        | complete_genes
    )
    master = pd.DataFrame({"ncu_id": all_genes})
    master = master.merge(gene_status, on="ncu_id", how="left")
    master = master.merge(annotation, on="ncu_id", how="left")
    master["in_genome_annotation"] = master["ncu_id"].isin(genome_genes)
    master["in_ijdz836_model"] = master["ncu_id"].isin(sbml_genes)
    master["dreyfuss_predicted_essential"] = master["ncu_id"].isin(essential_genes)
    master["in_processed_phenotype_matrix"] = master["ncu_id"].isin(processed_genes)
    master["in_complete_10_assay_matrix"] = master["ncu_id"].isin(complete_genes)
    master["has_ko_status_record"] = master["ko_gene_status"].notna()
    master["homokaryon_available"] = master["ko_gene_status"].eq("homokaryon_available")
    master["heterokaryon_only"] = master["ko_gene_status"].eq("heterokaryon_only")
    master["has_explicit_status_call"] = master["explicit_ko_gene_status"].isin(
        INTERPRETABLE_STATUSES
    )
    master["explicit_homokaryon_available"] = master["explicit_ko_gene_status"].eq(
        "homokaryon_available"
    )
    master["explicit_heterokaryon_only"] = master["explicit_ko_gene_status"].eq("heterokaryon_only")
    carrillo_metabolic = master.get(
        "in_carrillo_metabolic_sheet",
        pd.Series(False, index=master.index),
    )
    master["metabolic_union"] = master["in_ijdz836_model"] | as_bool(carrillo_metabolic)

    bool_columns = [
        "in_carrillo_metabolic_sheet",
        "has_yeast_ortholog",
        "transcription_factor",
        "gpcr",
        "ser_thr_kinase",
        "phosphatase",
        "transmembrane_any",
        "transmembrane_5plus",
        "high_phosphorylation_q75",
        "primary_status_uses_inference",
    ]
    for column in bool_columns:
        if column not in master.columns:
            master[column] = False
        master[column] = as_bool(master[column])

    status_summary, complete_status_check, complete_status_records = complete_matrix_status_check(
        master,
        records,
    )
    mapping_summary, mapping_check = identifier_mapping_tables(
        master,
        sbml_genes,
        essential_genes,
        genome_genes,
        args.reported_model_gene_count,
    )

    complete_known = master[
        as_bool(master["in_complete_10_assay_matrix"]) & as_bool(master["has_ko_status_record"])
    ].copy()
    complete_homokaryon_fraction = (
        float(as_bool(complete_known["homokaryon_available"]).mean())
        if len(complete_known)
        else np.nan
    )

    explicit_valid = master["explicit_ko_gene_status"].isin(INTERPRETABLE_STATUSES)
    summary = pd.DataFrame(
        [
            {
                "ko_workbook": ko_path.name,
                "n_strain_records_parsed": len(records),
                "n_genes_with_ko_records": int(master["has_ko_status_record"].sum()),
                "n_homokaryon_available": int(master["homokaryon_available"].sum()),
                "n_heterokaryon_only": int(master["heterokaryon_only"].sum()),
                "n_explicit_interpretable_status": int(explicit_valid.sum()),
                "n_explicit_homokaryon_available": int(
                    master["explicit_homokaryon_available"].sum()
                ),
                "n_explicit_heterokaryon_only": int(master["explicit_heterokaryon_only"].sum()),
                "n_primary_calls_using_inference": int(
                    master["primary_status_uses_inference"].sum()
                ),
                "n_invalid_or_unresolved": int(
                    master["ko_gene_status"]
                    .isin(["invalid_or_unavailable_only", "unresolved"])
                    .sum()
                ),
                "n_genome_annotation_genes": len(genome_genes),
                "n_ijdz836_genes_parsed": len(sbml_genes),
                "n_ijdz836_genes_reported": args.reported_model_gene_count,
                "n_dreyfuss_predicted_essential": len(essential_genes),
                "n_processed_phenotype_matrix": len(processed_genes),
                "n_complete_10_assay": len(complete_genes),
                "processed_matrix_equals_complete_matrix": processed_genes == complete_genes,
                "n_complete_with_ko_status": len(complete_known),
                "n_complete_not_current_homokaryon_available": len(complete_status_check),
                "fraction_complete_with_current_homokaryon_status": complete_homokaryon_fraction,
            }
        ]
    )

    records.to_csv(processed / "ko_strain_records.csv", index=False)
    gene_status.to_csv(processed / "ko_gene_status.csv", index=False)
    master.to_csv(processed / "collection_gene_master.csv", index=False)
    dreyfuss_essential.to_csv(
        processed / "dreyfuss_predicted_essential_genes.csv",
        index=False,
    )

    records.to_csv(source_dir / "ko_strain_records.csv", index=False)
    master.to_csv(source_dir / "collection_gene_master.csv", index=False)
    status_summary.to_csv(source_dir / "complete_matrix_status_summary.csv", index=False)
    complete_status_check.to_csv(
        source_dir / "complete_matrix_status_check.csv",
        index=False,
    )
    complete_status_records.to_csv(
        source_dir / "complete_matrix_status_records.csv",
        index=False,
    )
    mapping_summary.to_csv(
        source_dir / "identifier_mapping_summary.csv",
        index=False,
    )
    mapping_check.to_csv(source_dir / "identifier_mapping_check.csv", index=False)

    write_workbook(
        table_dir / "collection_status_preparation.xlsx",
        {
            "summary": summary,
            "identifier_mapping_summary": mapping_summary,
            "complete_status_summary": status_summary,
            "complete_status_check": complete_status_check,
            "complete_status_records": complete_status_records,
            "gene_master": master,
            "gene_status": gene_status,
            "strain_records": records,
            "identifier_mapping_check": mapping_check,
            "dreyfuss_essential": dreyfuss_essential,
        },
    )
    print(summary.to_string(index=False))
    print("Complete-matrix current-status summary:")
    print(status_summary.to_string(index=False))
    print("Identifier mapping summary:")
    print(mapping_summary.to_string(index=False))
    print(f"Prepared collection-level data: {processed / 'collection_gene_master.csv'}")


if __name__ == "__main__":
    main()
