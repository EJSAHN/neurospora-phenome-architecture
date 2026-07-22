#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Neurospora phenome-architecture workflow")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--ko-workbook", default="")
    parser.add_argument("--profile-shapley", type=int, default=10000)
    parser.add_argument("--profile-bootstrap", type=int, default=1000)
    parser.add_argument("--profile-null", type=int, default=1000)
    parser.add_argument("--architecture-shapley", type=int, default=10000)
    parser.add_argument("--architecture-bootstrap", type=int, default=1000)
    parser.add_argument("--architecture-null", type=int, default=1000)
    parser.add_argument("--substitutability-target", type=float, default=0.90)
    parser.add_argument("--reported-model-gene-count", type=int, default=836)
    return parser.parse_args()


def run(script_dir: Path, script: str, *arguments: str) -> None:
    command = [sys.executable, str(script_dir / script), *arguments]
    print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    common = ["--project-root", str(root)]

    if not args.skip_download:
        run(script_dir, "download_sources.py", *common)
    run(script_dir, "inventory_sources.py", *common)
    run(script_dir, "prepare_phenotype_matrix.py", *common)
    run(
        script_dir,
        "analyze_profiles.py",
        *common,
        "--shapley-permutations",
        str(args.profile_shapley),
        "--bootstrap-iterations",
        str(args.profile_bootstrap),
        "--null-iterations",
        str(args.profile_null),
    )
    run(
        script_dir,
        "analyze_functional_architecture.py",
        *common,
        "--shapley-permutations",
        str(args.architecture_shapley),
        "--bootstrap-iterations",
        str(args.architecture_bootstrap),
        "--null-iterations",
        str(args.architecture_null),
    )
    run(script_dir, "analyze_sensitivity.py", *common)

    prepare_arguments = [
        *common,
        "--reported-model-gene-count",
        str(args.reported_model_gene_count),
    ]
    if args.ko_workbook:
        prepare_arguments.extend(["--ko-workbook", args.ko_workbook])
    run(script_dir, "prepare_collection_status.py", *prepare_arguments)
    run(script_dir, "analyze_collection_selection.py", *common)
    run(
        script_dir,
        "analyze_assay_dependencies.py",
        *common,
        "--substitutability-target",
        str(args.substitutability_target),
    )
    run(
        script_dir,
        "validate_results.py",
        *common,
        "--expected-substitutability-target",
        str(args.substitutability_target),
    )


if __name__ == "__main__":
    main()
