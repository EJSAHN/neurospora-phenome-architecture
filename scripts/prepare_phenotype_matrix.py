from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phenome_architecture.analysis import encode_phenotypes
from phenome_architecture.io import first_existing, read_excel_table


def merge_left(base: pd.DataFrame, other: pd.DataFrame, key: str = "NCU Number") -> pd.DataFrame:
    keep_cols = [c for c in other.columns if c != key]
    return base.merge(other[[key] + keep_cols], on=key, how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    args = parser.parse_args()
    root = Path(args.project_root)
    raw_dir = root / "data" / "raw" / "carrillo_2020"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_file = first_existing(
        [
            raw_dir / "12864_2020_7131_MOESM1_ESM.xlsx",
            raw_dir / "MOESM1.xlsx",
        ]
    )
    if main_file is None:
        raise FileNotFoundError(
            "Carrillo Additional file 1 was not found in data/raw/carrillo_2020"
        )

    phen = read_excel_table(main_file, "Phenotypes")
    product = read_excel_table(main_file, "Product descriptions")
    features = read_excel_table(main_file, "Protein features")
    chrom = read_excel_table(main_file, "Chromosome location")
    metabolic = read_excel_table(main_file, "Metabolic genes")

    id_cols = ["NCU Number", "Consolidated Gene Names"]
    phenotype_cols = [
        "Basal hyphae growth rate (mm/day)",
        "Aerial hyphae height (total mm)",
        "Conidia Number",
        "Conidia Morphology",
        "Protoperithecia Number",
        "Protoperithecial Morphology",
        "Perithecia Number",
        "Perithecia Morphology",
        "Ascospore Number",
        "Ascospore Morphology",
    ]
    phenotype_cols = [c for c in phenotype_cols if c in phen.columns]

    phen = phen.dropna(subset=["NCU Number"]).copy()
    phen["NCU Number"] = phen["NCU Number"].astype(str).str.strip()
    discrete = encode_phenotypes(phen, phenotype_cols)
    discrete.insert(0, "gene_name", phen["Consolidated Gene Names"].astype(str))
    discrete.insert(0, "ncu_id", phen["NCU Number"].astype(str))
    discrete["published_cluster"] = phen["Cluster"].astype(str)

    raw = phen[id_cols + phenotype_cols + ["Cluster"]].copy()
    raw = raw.rename(
        columns={
            "NCU Number": "ncu_id",
            "Consolidated Gene Names": "gene_name",
            "Cluster": "published_cluster",
        }
    )

    annot = phen[["NCU Number", "Consolidated Gene Names", "Cluster"]].rename(
        columns={
            "NCU Number": "NCU Number",
            "Consolidated Gene Names": "gene_name",
            "Cluster": "published_cluster",
        }
    )
    product_small = product.rename(columns={"Product Description": "product_description"})
    features_small = features.rename(
        columns={
            "Gene Classification": "gene_classification",
            "# of Phosphorylation Sites": "n_phosphorylation_sites",
            "# Transmembrane Helices": "n_transmembrane_helices",
        }
    )
    chrom_small = chrom.rename(columns={"Chromosome": "chromosome"})
    metabolic_small = metabolic[["NCU Number"]].copy()
    metabolic_small["is_metabolic_gene_carrillo_sheet"] = True

    annot = annot.merge(
        product_small[["NCU Number", "product_description"]], on="NCU Number", how="left"
    )
    annot = annot.merge(
        features_small[
            [
                "NCU Number",
                "gene_classification",
                "n_phosphorylation_sites",
                "n_transmembrane_helices",
            ]
        ],
        on="NCU Number",
        how="left",
    )
    annot = annot.merge(chrom_small[["NCU Number", "chromosome"]], on="NCU Number", how="left")
    annot = annot.merge(metabolic_small, on="NCU Number", how="left")
    annot["is_metabolic_gene_carrillo_sheet"] = annot["is_metabolic_gene_carrillo_sheet"].fillna(
        False
    )
    annot = annot.rename(columns={"NCU Number": "ncu_id"})

    yeast_file = first_existing([raw_dir / "12864_2020_7131_MOESM6_ESM.xlsx"])
    if yeast_file is not None:
        yeast = read_excel_table(yeast_file, "Yeast Ortholog Summary", required="Gene Number")
        yeast = yeast.rename(
            columns={"Gene Number": "ncu_id", "Gene Name": "yeast_sheet_gene_name"}
        )
        yeast["has_yeast_ortholog"] = (
            yeast[[c for c in yeast.columns if str(c).startswith("Yeast Ortholog")]]
            .notna()
            .any(axis=1)
        )
        yeast = yeast[["ncu_id", "has_yeast_ortholog"]].drop_duplicates("ncu_id")
        annot = annot.merge(yeast, on="ncu_id", how="left")
        annot["has_yeast_ortholog"] = annot["has_yeast_ortholog"].fillna(False)
    else:
        annot["has_yeast_ortholog"] = False

    assay_catalog = pd.DataFrame(
        {
            "assay": phenotype_cols,
            "assay_type": [
                (
                    "continuous_binned"
                    if c in ["Basal hyphae growth rate (mm/day)", "Aerial hyphae height (total mm)"]
                    else "categorical"
                )
                for c in phenotype_cols
            ],
        }
    )

    raw.to_csv(out_dir / "phenotype_matrix_raw.csv", index=False)
    discrete.to_csv(out_dir / "phenotype_matrix_discrete.csv", index=False)
    annot.to_csv(out_dir / "gene_annotations.csv", index=False)
    assay_catalog.to_csv(out_dir / "assay_catalog.csv", index=False)

    print(f"Prepared phenotype matrix: {discrete.shape[0]} mutants x {len(phenotype_cols)} assays")
    print(f"Processed outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
