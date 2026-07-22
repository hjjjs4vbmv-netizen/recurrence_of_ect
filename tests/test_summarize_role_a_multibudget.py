import csv
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_role_a_multibudget as summary


class RoleAMultibudgetSummaryTest(unittest.TestCase):
    def make_rows(self):
        rows = []
        for budget in summary.BUDGETS:
            for nfe in summary.NFES:
                for seed in summary.SEEDS:
                    for method in summary.METHODS:
                        adjustment = -0.01 if method == "adaptive_v1" else 0.0
                        rows.append({
                            "method": method,
                            "training_seed": seed,
                            "budget_kimg": budget,
                            "nfe": nfe,
                            "kid5k": 1 + seed + adjustment,
                            "fid5k": 10 + seed + adjustment,
                            "checkpoint_sha256": f"sha-{method}-{seed}-{budget}",
                        })
        return rows

    def test_complete_matrix_pairs_and_aggregates(self):
        rows = summary.validate_matrix(self.make_rows())
        paired = summary.pair_rows(rows)
        aggregate = summary.aggregate_rows(rows, paired)
        self.assertEqual(len(rows), 36)
        self.assertEqual(len(paired), 18)
        self.assertEqual(len(aggregate), 12)
        self.assertTrue(all(row["delta_kid5k"] < 0 for row in paired))
        self.assertTrue(all(row["adaptive_wins"] == 3 for row in aggregate))

    def test_csv_requires_exact_role_a_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary.INPUT_COLUMNS))
                writer.writeheader()
            self.assertEqual(summary.read_rows([path]), [])

    def test_incomplete_matrix_is_rejected(self):
        with self.assertRaises(SystemExit):
            summary.validate_matrix(self.make_rows()[:-1])


if __name__ == "__main__":
    unittest.main()
