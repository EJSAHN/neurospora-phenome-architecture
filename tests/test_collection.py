from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phenome_architecture.collection import (  # noqa: E402
    aggregate_gene_status,
    apply_exclusions,
    load_exclusions,
    parse_ko_workbook,
)


class CollectionParserTests(unittest.TestCase):
    def test_status_precedence_and_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook = root / "ko.xlsx"
            frame = pd.DataFrame(
                {
                    "NCU Number": [
                        "NCU00001.2",
                        "NCU00002.2",
                        "NCU00003.2",
                        "NCU00002.2",
                    ],
                    "FGSC #": [11201, 11202, 11459, 11203],
                    "Status": [
                        "homokaryon",
                        "heterokaryon",
                        "heterokaryon",
                        "homokaryon",
                    ],
                    "Notes": ["validated", "validated", "Southern wrong", "validated"],
                }
            )
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="Available strains", index=False, startrow=2)
            exclusions = root / "exclusions.csv"
            pd.DataFrame(
                {
                    "fgsc_id": [11459],
                    "ncu_id": ["NCU00003"],
                    "reason": ["invalid"],
                    "source_url": ["test"],
                }
            ).to_csv(exclusions, index=False)
            records = parse_ko_workbook(workbook)
            records = apply_exclusions(records, load_exclusions(exclusions))
            status = aggregate_gene_status(records).set_index("ncu_id")
            self.assertEqual(
                status.loc["NCU00001", "ko_gene_status"],
                "homokaryon_available",
            )
            self.assertEqual(
                status.loc["NCU00002", "ko_gene_status"],
                "homokaryon_available",
            )
            self.assertEqual(
                status.loc["NCU00002", "explicit_ko_gene_status"],
                "homokaryon_available",
            )
            self.assertEqual(
                status.loc["NCU00003", "ko_gene_status"],
                "invalid_or_unavailable_only",
            )

    def test_homokaryon_unavailable_but_heterokaryon_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook = root / "ko.xlsx"
            frame = pd.DataFrame(
                {
                    "NCU Number": ["NCU00004.2", "NCU00005.2"],
                    "Heterokaryon FGSC": [22004, ""],
                    "Notes": [
                        "homokaryon not available; heterokaryon retained",
                        "homokaryon not achieved after repeated attempts",
                    ],
                }
            )
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="Available strains", index=False, startrow=1)
            records = parse_ko_workbook(workbook)
            records = apply_exclusions(records, load_exclusions(None))
            status = aggregate_gene_status(records).set_index("ncu_id")
            self.assertEqual(
                status.loc["NCU00004", "ko_gene_status"],
                "heterokaryon_only",
            )
            self.assertEqual(
                status.loc["NCU00004", "explicit_ko_gene_status"],
                "heterokaryon_only",
            )
            self.assertEqual(
                status.loc["NCU00005", "ko_gene_status"],
                "invalid_or_unavailable_only",
            )

    def test_inclusive_and_explicit_statuses_are_separate(self) -> None:
        records = pd.DataFrame(
            {
                "ncu_id": ["NCU00008", "NCU00008"],
                "record_status": ["heterokaryon", "homokaryon"],
                "status_confidence": ["explicit", "inferred_from_available_record"],
                "fgsc_ids": ["22008", "22009"],
                "source_sheet": ["test", "test"],
                "excluded_from_status_analysis": [False, False],
                "status_exclusion_reason": ["", ""],
            }
        )
        status = aggregate_gene_status(records).set_index("ncu_id")
        self.assertEqual(
            status.loc["NCU00008", "ko_gene_status"],
            "homokaryon_available",
        )
        self.assertEqual(
            status.loc["NCU00008", "explicit_ko_gene_status"],
            "heterokaryon_only",
        )
        self.assertTrue(status.loc["NCU00008", "primary_status_uses_inference"])

    def test_multiple_ncu_ids_in_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workbook = Path(temporary) / "ko.xlsx"
            frame = pd.DataFrame(
                {
                    "Gene record": ["NCU00006.1 and NCU00007.2"],
                    "FGSC strain": [22006],
                    "Status": ["homokaryon"],
                }
            )
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="Available strains", index=False)
            records = parse_ko_workbook(workbook)
            self.assertEqual(set(records["ncu_id"]), {"NCU00006", "NCU00007"})


if __name__ == "__main__":
    unittest.main()
