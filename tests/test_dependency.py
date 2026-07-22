from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenome_architecture.dependency import (  # noqa: E402
    all_assay_substitutability_matrix,
    build_module_matrix,
    continuous_trait_association,
    developmental_dependency_table,
    exact_module_subset_metrics,
    minimal_set_assay_frequency,
)


class DependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        n = 80
        self.raw = pd.DataFrame(
            {
                "ncu_id": [f"NCU{i:05d}" for i in range(n)],
                "gene_name": [f"g{i}" for i in range(n)],
                "published_cluster": np.repeat(np.arange(8), 10),
                "Basal hyphae growth rate (mm/day)": rng.normal(size=n),
                "Aerial hyphae height (total mm)": rng.normal(size=n),
            }
        )
        self.discrete = self.raw[["ncu_id", "gene_name", "published_cluster"]].copy()
        self.discrete["Basal hyphae growth rate (mm/day)"] = pd.qcut(
            self.raw["Basal hyphae growth rate (mm/day)"],
            5,
            labels=False,
        ).astype(str)
        self.discrete["Aerial hyphae height (total mm)"] = pd.qcut(
            self.raw["Aerial hyphae height (total mm)"],
            5,
            labels=False,
        ).astype(str)
        for name in [
            "Conidia Number",
            "Conidia Morphology",
            "Protoperithecia Number",
            "Protoperithecial Morphology",
            "Perithecia Number",
            "Perithecia Morphology",
            "Ascospore Number",
            "Ascospore Morphology",
        ]:
            self.discrete[name] = np.where(
                rng.random(n) < 0.25,
                "abnormal",
                "normal",
            )

    def test_continuous_metrics(self) -> None:
        result = continuous_trait_association(
            self.raw,
            self.discrete,
            "Basal hyphae growth rate (mm/day)",
            "Aerial hyphae height (total mm)",
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(0 <= result.loc[0, "pearson_r_squared"] <= 1)
        self.assertTrue(0 <= result.loc[0, "discrete_ordinal_pearson_r_squared"] <= 1)

    def test_module_search(self) -> None:
        catalog = pd.read_csv(Path(__file__).resolve().parents[1] / "config" / "assay_modules.csv")
        module_frame, module_defs = build_module_matrix(self.discrete, catalog)
        metrics, exact, sets = exact_module_subset_metrics(module_frame, module_defs)
        self.assertEqual(metrics["n_modules"].max(), 6)
        self.assertFalse(exact.empty)
        self.assertFalse(sets.empty)

    def test_developmental_dependencies(self) -> None:
        assays = ["Protoperithecia Number", "Perithecia Number", "Ascospore Number"]
        output = developmental_dependency_table(self.discrete, assays)
        self.assertEqual(len(output), 3)
        self.assertIn("fdr_q", output.columns)

    def test_substitutability_at_target_with_equivalent_sets(self) -> None:
        assays = ["growth_a", "growth_b", "development_a", "development_b"]
        minimal_sets = pd.DataFrame(
            {
                "objective": ["cluster_nmi_recovery"] * 3,
                "target_fraction_of_full": [0.90] * 3,
                "assay_set": [
                    "growth_a;growth_b;development_a",
                    "growth_a;growth_b;development_b",
                    "growth_a;development_a;development_b",
                ],
            }
        )
        matrix, edges = all_assay_substitutability_matrix(
            minimal_sets,
            assays,
            target=0.90,
        )
        metadata, frequency = minimal_set_assay_frequency(
            minimal_sets,
            assays,
            target=0.90,
        )
        values = matrix.drop(columns="assay").to_numpy()
        self.assertTrue((values > 0).any())
        self.assertGreaterEqual(len(edges), 1)
        self.assertEqual(int(metadata.loc[0, "n_equivalent_minimal_sets"]), 3)
        self.assertEqual(len(frequency), 4)


if __name__ == "__main__":
    unittest.main()
