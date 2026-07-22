from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = REPO_ROOT / "scripts" / "analyze_paired_results.py"


def sha(identifier: int) -> str:
    return f"{identifier:064x}"


def write_summary(path: Path, method: str, seed: int) -> None:
    fieldnames = [
        "attempted_iteration",
        "successful_optimizer_steps",
        "processed_kimg",
        "schedule",
        "loss",
        "step_skipped",
        "adaptive_active",
        "correction",
        "r_over_t_mean",
        "gap_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for iteration, budget in enumerate((8, 16, 32, 64), start=1):
            writer.writerow(
                {
                    "attempted_iteration": iteration,
                    "successful_optimizer_steps": iteration,
                    "processed_kimg": budget,
                    "schedule": method,
                    "loss": 10 - iteration + seed * 0.01,
                    "step_skipped": "0",
                    "adaptive_active": "1" if method == "adaptive_v1" else "0",
                    "correction": "0.01" if method == "adaptive_v1" else "0",
                    "r_over_t_mean": "0.97" if method == "adaptive_v1" else "0.98",
                    "gap_mean": "0.03" if method == "adaptive_v1" else "0.02",
                }
            )


class RoleCAnalyzerTests(unittest.TestCase):
    def write_complete_fixture(self, root: Path) -> tuple[Path, Path]:
        metrics_path = root / "metrics.csv"
        records_path = root / "training_records.csv"
        metric_fields = [
            "method", "training_seed", "budget_kimg", "nfe", "kid_5k", "fid_5k", "checkpoint_sha256", "mid_t", "sampling_seed", "num_generated",
        ]
        record_fields = ["method", "training_seed", "budget_kimg", "checkpoint_sha256", "training_summary_csv", "run_dir"]
        with metrics_path.open("w", newline="", encoding="utf-8") as metric_handle, records_path.open("w", newline="", encoding="utf-8") as record_handle:
            metric_writer = csv.DictWriter(metric_handle, fieldnames=metric_fields)
            record_writer = csv.DictWriter(record_handle, fieldnames=record_fields)
            metric_writer.writeheader()
            record_writer.writeheader()
            for method_index, method in enumerate(("sigmoid", "adaptive_v1")):
                for seed in (0, 1, 2):
                    summary = root / f"{method}-seed{seed}" / "train_summary.csv"
                    write_summary(summary, method, seed)
                    for budget in (16, 32, 64):
                        digest = sha(method_index * 100 + seed * 10 + budget // 16)
                        record_writer.writerow(
                            {
                                "method": method,
                                "training_seed": seed,
                                "budget_kimg": budget,
                                "checkpoint_sha256": digest,
                                "training_summary_csv": summary,
                                "run_dir": summary.parent,
                            }
                        )
                        for nfe in (1, 2):
                            fixed_kid = 0.020 + budget / 100000 + seed / 1000000 + nfe / 10000000
                            # Adaptive improves in all settings for seeds 0/1 and is modestly worse for seed 2.
                            delta = -0.001 if seed in (0, 1) else 0.0002
                            value = fixed_kid if method == "sigmoid" else fixed_kid + delta
                            metric_writer.writerow(
                                {
                                    "method": method,
                                    "training_seed": seed,
                                    "budget_kimg": budget,
                                    "nfe": nfe,
                                    "kid_5k": value,
                                    "fid_5k": value * 300,
                                    "checkpoint_sha256": digest,
                                    "mid_t": "0.821" if nfe == 2 else "",
                                    "sampling_seed": "0-4999",
                                    "num_generated": 5000,
                                }
                            )
        return metrics_path, records_path

    def test_complete_matrix_writes_all_role_c_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            output = root / "role_c"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(output), "--require-complete"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in (
                "per_seed_metrics.csv",
                "paired_differences.csv",
                "aggregate_results.csv",
                "quality_vs_budget.png",
                "controller_vs_budget.png",
                "FINAL_CONCLUSION.md",
            ):
                self.assertTrue((output / name).is_file(), name)
                self.assertGreater((output / name).stat().st_size, 0, name)
            with (output / "paired_differences.csv").open(newline="", encoding="utf-8") as handle:
                paired_rows = list(csv.DictReader(handle))
            self.assertEqual(len(paired_rows), 36)  # 3 seeds × 3 budgets × 2 NFEs × KID/FID.
            first_kid = next(row for row in paired_rows if row["metric"] == "kid_5k" and row["training_seed"] == "0")
            self.assertLess(float(first_kid["delta_adaptive_minus_fixed"]), 0)
            self.assertEqual(first_kid["num_generated"], "5000")
            with (output / "aggregate_results.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 12)  # 2 metrics × 3 budgets × 2 NFEs.
            conclusion = (output / "FINAL_CONCLUSION.md").read_text(encoding="utf-8")
            self.assertIn("Adaptive 表现出初步优势", conclusion)
            self.assertNotIn("Current verdict: INCOMPLETE", conclusion)
            self.assertIn("Training and controller relationships", conclusion)
            self.assertIn("paired_delta_vs_correction", conclusion)

    def test_partial_matrix_is_reported_as_incomplete_without_failing_rolling_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows[:-1])
            output = root / "partial"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(output)],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Current verdict: INCOMPLETE", (output / "FINAL_CONCLUSION.md").read_text(encoding="utf-8"))

    def test_missing_sampling_provenance_is_not_misreported_as_missing_metric_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["sampling_seed"] = ""
                row["num_generated"] = ""
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            output = root / "missing_provenance"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(output)],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            conclusion = (output / "FINAL_CONCLUSION.md").read_text(encoding="utf-8")
            self.assertIn("Current verdict: INCOMPLETE", conclusion)
            self.assertIn("evaluation provenance is incomplete", conclusion)
            self.assertIn("Metric coverage is complete", conclusion)
            self.assertNotIn("matrix is not yet available", conclusion)

    def test_rejects_metric_that_exists_for_only_one_arm_of_a_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["method"] == "adaptive_v1" and row["training_seed"] == "0" and row["budget_kimg"] == "16" and row["nfe"] == "1":
                    row["fid_5k"] = ""
                    break
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "bad")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("present for only one arm", completed.stderr)

    def test_rejects_checkpoint_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["checkpoint_sha256"] = sha(999)
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "sha-mismatch")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("checkpoint SHA mismatch", completed.stderr)

    def test_rejects_reusing_one_checkpoint_for_both_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            fixed_sha = next(
                row["checkpoint_sha256"]
                for row in metric_rows
                if row["method"] == "sigmoid" and row["training_seed"] == "0" and row["budget_kimg"] == "16"
            )
            for row in metric_rows:
                if row["method"] == "adaptive_v1" and row["training_seed"] == "0" and row["budget_kimg"] == "16":
                    row["checkpoint_sha256"] = fixed_sha
            for row in record_rows:
                if row["method"] == "adaptive_v1" and row["training_seed"] == "0" and row["budget_kimg"] == "16":
                    row["checkpoint_sha256"] = fixed_sha
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
                writer.writeheader()
                writer.writerows(metric_rows)
            with records.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=record_rows[0].keys())
                writer.writeheader()
                writer.writerows(record_rows)
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "same-checkpoint")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("same checkpoint SHA", completed.stderr)

    def test_rejects_a_nonfrozen_two_step_midpoint_even_when_arms_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["nfe"] == "2":
                    row["mid_t"] = "0.7"
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "wrong-midpoint")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires the frozen mid_t=0.821", completed.stderr)

    def test_rejects_metrics_outside_the_frozen_budget_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["budget_kimg"] = "8"
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "wrong-budget")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("budget_kimg must be one of", completed.stderr)

    def test_final_conclusion_requires_matched_five_thousand_image_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["budget_kimg"] == "16" and row["nfe"] == "1":
                    row["num_generated"] = "512"
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            output = root / "not-final"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(output), "--require-complete"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("Current verdict: INCOMPLETE", (output / "FINAL_CONCLUSION.md").read_text(encoding="utf-8"))

    def test_rejects_undertrained_training_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            target = next(row for row in record_rows if row["method"] == "sigmoid" and row["training_seed"] == "0" and row["budget_kimg"] == "16")
            summary_path = Path(target["training_summary_csv"])
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
                fields = handle.seek(0) or None
            # Keep only the 8-kimg row: Role C must reject it for a 16-kimg checkpoint.
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerow(summary_rows[0])
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "undertrained")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("reaches only 8 kimg", completed.stderr)

    def test_rejects_training_summary_with_the_wrong_schedule_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            target = next(row for row in record_rows if row["method"] == "adaptive_v1" and row["training_seed"] == "0")
            summary_path = Path(target["training_summary_csv"])
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            for row in summary_rows:
                row["schedule"] = "sigmoid"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "wrong-schedule")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("does not match training record method", completed.stderr)

    def test_nonfinite_training_loss_blocks_a_final_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            target = next(row for row in record_rows if row["method"] == "adaptive_v1" and row["training_seed"] == "2")
            summary_path = Path(target["training_summary_csv"])
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            summary_rows[-1]["loss"] = "nan"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            output = root / "nonfinite"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(output), "--require-complete"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            conclusion = (output / "FINAL_CONCLUSION.md").read_text(encoding="utf-8")
            self.assertIn("Current verdict: INCOMPLETE", conclusion)
            self.assertIn("non-finite loss", conclusion)

    def test_out_of_bound_controller_correction_blocks_a_final_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            target = next(row for row in record_rows if row["method"] == "adaptive_v1" and row["training_seed"] == "1")
            summary_path = Path(target["training_summary_csv"])
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            summary_rows[-1]["correction"] = "0.051"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            output = root / "out-of-bound-correction"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(output), "--require-complete"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("correction exceeds the configured bound", (output / "FINAL_CONCLUSION.md").read_text(encoding="utf-8"))

    def test_nonfinite_controller_telemetry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            target = next(row for row in record_rows if row["method"] == "adaptive_v1" and row["training_seed"] == "2")
            summary_path = Path(target["training_summary_csv"])
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            summary_rows[-1]["r_over_t_mean"] = "nan"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(root / "nonfinite-controller")],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("non-finite r_over_t_mean telemetry", completed.stderr)

    def test_missing_successful_step_telemetry_blocks_a_final_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            target = next(row for row in record_rows if row["method"] == "sigmoid" and row["training_seed"] == "2")
            summary_path = Path(target["training_summary_csv"])
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            summary_rows[-1]["successful_optimizer_steps"] = ""
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            output = root / "missing-successful-step"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-records", str(records), "--outdir", str(output), "--require-complete"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("successful optimizer-step telemetry", (output / "FINAL_CONCLUSION.md").read_text(encoding="utf-8"))

    def test_training_root_ignores_off_matrix_smokes_and_loads_protocol_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics, records = self.write_complete_fixture(root)
            with records.open(newline="", encoding="utf-8") as handle:
                record_rows = list(csv.DictReader(handle))
            training_root = root / "role_b_runs"
            for index, row in enumerate(record_rows):
                run_dir = training_root / f"run-{index}"
                run_dir.mkdir(parents=True)
                shutil.copyfile(row["training_summary_csv"], run_dir / "train_summary.csv")
                (run_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "schedule": row["method"],
                            "seed": int(row["training_seed"]),
                            "processed_kimg": float(row["budget_kimg"]),
                            "network_snapshot_sha256": row["checkpoint_sha256"],
                        }
                    ),
                    encoding="utf-8",
                )
            smoke = training_root / "old-activation-smoke"
            smoke.mkdir()
            (smoke / "metadata.json").write_text(
                json.dumps(
                    {
                        "schedule": "adaptive_v1",
                        "seed": 0,
                        "processed_kimg": 4.096,
                        "network_snapshot_sha256": sha(888),
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(ANALYZER), "--metrics", str(metrics), "--training-root", str(training_root), "--outdir", str(root / "root-scan"), "--require-complete"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
