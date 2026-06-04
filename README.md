# neurospora-phenome-architecture

Analysis code for quantifying assay-panel compressibility, published-cluster recovery, gene-class enrichment, and metabolic-model contrast in the *Neurospora crassa* deletion phenome.

## Repository description

This repository reproduces a public-data analysis of the *N. crassa* deletion-phenome data from Carrillo et al. and the iJDZ836 metabolic model from Dreyfuss et al. The workflow asks how much of the deletion-phenome structure is retained when classical growth and developmental assays are reduced to smaller subsets.

The repository contains the computational analysis workflow and source-data export steps.

## Inputs

All source files are public and are listed in `config/source_manifest.csv`. The workflow downloads:

- Carrillo et al. phenotype and annotation supplementary workbooks.
- Dreyfuss et al. iJDZ836 metabolic-model files.
- Ensembl Fungi NC12 genome annotation files.

Downloaded files are stored under `data/raw/`, which is ignored by Git.

## Quick start

```bash
conda env create -f environment.yml
conda activate neurospora-phenome
python scripts/run_pipeline.py --project-root .
```

For an existing local copy of the public input files, skip the download step:

```bash
python scripts/run_pipeline.py --project-root . --skip-download
```

## Main workflow

```bash
python scripts/download_sources.py --project-root .
python scripts/inventory_sources.py --project-root .
python scripts/prepare_phenotype_matrix.py --project-root .
python scripts/analyze_profiles.py --project-root . --shapley-permutations 10000 --bootstrap-iterations 1000 --null-iterations 1000
python scripts/analyze_functional_architecture.py --project-root . --shapley-permutations 10000 --bootstrap-iterations 1000 --null-iterations 1000
python scripts/analyze_sensitivity.py --project-root . --random-set-iterations 2000
```

## Outputs

The workflow writes processed matrices, analysis workbooks, and source-data tables to:

```text
data/processed/
reports/tables/
reports/source_data/
```

Key analysis workbooks are:

```text
reports/tables/phenome_profile_results.xlsx
reports/tables/phenome_architecture_results.xlsx
reports/tables/phenome_sensitivity_results.xlsx
```

Generated input data, processed data, and reports are ignored by Git.

## Notes on reproducibility

All stochastic procedures use explicit random seeds. Exact subset searches enumerate all non-empty subsets of the 10-assay panel.
