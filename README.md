# neurospora-phenome-architecture

Reproducible analysis of phenotype-panel structure and knockout-collection recovery in the *Neurospora crassa* deletion resource.

## Overview

The workflow combines five related analyses:

1. recovery of discrete phenotype profiles from reduced assay panels;
2. preservation of a reference phenotype-cluster architecture;
3. sensitivity to continuous-trait discretization and annotation rules;
4. homokaryon and heterokaryon recovery across the knockout collection;
5. dependence-aware analysis of growth and developmental assays.

All inputs are public. File names, source descriptions, and download locations are listed in `config/source_manifest.csv`.

## Requirements

Python 3.10 or later is required. A Conda environment can be created with:

```bash
conda env create -f environment.yml
conda activate neurospora-phenome
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

## Run the workflow

From the repository root:

```bash
python scripts/run_pipeline.py --project-root .
```

To use source files that are already present under `data/raw/`:

```bash
python scripts/run_pipeline.py --project-root . --skip-download
```

A knockout-availability workbook can be supplied explicitly:

```bash
python scripts/run_pipeline.py --project-root . --skip-download --ko-workbook path/to/Available_KO_Strains.xlsx
```

The default workflow uses 10,000 random assay orderings and 1,000 bootstrap and null iterations for profile and cluster analyses. The one-for-one substitutability analysis uses the 90% reference-cluster NMI target because that target contains multiple equivalent exact minimal sets. Alternative values can be supplied through the command-line options of `run_pipeline.py`.

## Workflow steps

```bash
python scripts/download_sources.py --project-root .
python scripts/inventory_sources.py --project-root .
python scripts/prepare_phenotype_matrix.py --project-root .
python scripts/analyze_profiles.py --project-root .
python scripts/analyze_functional_architecture.py --project-root .
python scripts/analyze_sensitivity.py --project-root .
python scripts/prepare_collection_status.py --project-root .
python scripts/analyze_collection_selection.py --project-root .
python scripts/analyze_assay_dependencies.py --project-root .
python scripts/validate_results.py --project-root .
```

## Outputs

Generated files are written to:

```text
data/processed/
reports/tables/
reports/source_data/
```

Primary workbooks:

```text
reports/tables/phenome_profile_results.xlsx
reports/tables/phenome_architecture_results.xlsx
reports/tables/phenome_sensitivity_results.xlsx
reports/tables/collection_status_preparation.xlsx
reports/tables/collection_selection_results.xlsx
reports/tables/assay_dependency_results.xlsx
```

The workflow exports numerical results and tabular source data. Plot generation is intentionally separate from the analysis pipeline.

## Reproducibility and validation

All stochastic procedures use explicit random seeds. The assay-level search enumerates all 1,023 non-empty subsets of the 10-assay panel. The dependency-aware search enumerates all 63 non-empty subsets of six biological modules.

Knockout-status analyses are reported for two status scopes. The primary scope includes explicit calls and archived available-strain records that can be interpreted as homokaryons. The explicit-only scope excludes inferred calls. Current knockout availability is not assumed to reproduce stock status at the time of phenotype collection; unmatched records are retained in status-check tables.

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

The final workflow step verifies matrix dimensions, knockout-status coverage, identifier mapping, continuous-trait association outputs, module recovery, and the all-assay substitutability matrix.

## Repository structure

```text
config/                         analysis settings and source manifest
scripts/                        command-line workflow
src/phenome_architecture/       reusable analysis functions
tests/                          unit tests
reports/source_data/            generated tabular exports
```

Raw inputs, processed matrices, and generated workbooks are excluded from version control.

## Data sources

- Carrillo AJ et al. (2020). *BMC Genomics* 21:755. https://doi.org/10.1186/s12864-020-07131-7
- Dreyfuss JM et al. (2013). *PLoS Computational Biology* 9:e1003126. https://doi.org/10.1371/journal.pcbi.1003126
- Ensembl Fungi, *Neurospora crassa* NC12 genome annotation.
- Neurospora Functional Genomics Project knockout-availability records.

## License

MIT License.
