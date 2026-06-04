#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the phenome-architecture analysis workflow")
    p.add_argument("--project-root", default=".")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--profile-shapley", type=int, default=10000)
    p.add_argument("--profile-bootstrap", type=int, default=1000)
    p.add_argument("--profile-null", type=int, default=1000)
    p.add_argument("--architecture-shapley", type=int, default=10000)
    p.add_argument("--architecture-bootstrap", type=int, default=1000)
    p.add_argument("--architecture-null", type=int, default=1000)
    p.add_argument("--random-set-iterations", type=int, default=2000)
    return p.parse_args()


def call(script: str, *args: str) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve().parent / script), *args]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = str(Path(args.project_root).expanduser().resolve())
    common = ["--project-root", root]
    if not args.skip_download:
        call("download_sources.py", *common)
    call("inventory_sources.py", *common)
    call("prepare_phenotype_matrix.py", *common)
    call("analyze_profiles.py", *common, "--shapley-permutations", str(args.profile_shapley), "--bootstrap-iterations", str(args.profile_bootstrap), "--null-iterations", str(args.profile_null))
    call("analyze_functional_architecture.py", *common, "--shapley-permutations", str(args.architecture_shapley), "--bootstrap-iterations", str(args.architecture_bootstrap), "--null-iterations", str(args.architecture_null))
    call("analyze_sensitivity.py", *common, "--random-set-iterations", str(args.random_set_iterations))


if __name__ == "__main__":
    main()
