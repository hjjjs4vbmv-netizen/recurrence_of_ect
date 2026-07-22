import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import collect_role_a_quality_results
from scripts import run_role_a_quality_evaluation


class RoleAQualityEvaluationTest(unittest.TestCase):
    def make_manifest(self, root: Path) -> Path:
        cells = []
        for budget in (64, 32, 16):
            for seed in range(3):
                for method in ("sigmoid", "adaptive_v1"):
                    checkpoint = root / f"{method}-seed{seed}-{budget}k.pkl"
                    checkpoint.write_bytes(f"{method}-{seed}-{budget}".encode())
                    cells.append({
                        "method": method,
                        "training_seed": seed,
                        "budget_kimg": budget,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": run_role_a_quality_evaluation.sha256_file(checkpoint),
                    })
        path = root / "checkpoints.json"
        path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        return path

    def test_smoke_is_seed0_16k_512_samples_with_exact_repeats(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cells = run_role_a_quality_evaluation.load_cells(self.make_manifest(root))
            selected = run_role_a_quality_evaluation.select_cells(cells, "smoke", 16)
            jobs = run_role_a_quality_evaluation.build_jobs(
                selected, root / "cifar.zip", root / "out", "smoke", "both", 29700
            )
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                self.assertEqual(job["training_seed"], 0)
                self.assertEqual(job["budget_kimg"], 16)
                self.assertEqual(job["sample_count"], 512)
                self.assertEqual(job["metric_repeats"], 2)
                command = " ".join(job["command"])
                self.assertIn("--sample-seeds=0-511", command)
                self.assertIn("--metrics=kid512_full,fid512_full", command)
                self.assertIn("--fp16=False", command)

    def test_formal_requires_complete_budget_and_keeps_seed_cells_separate(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cells = run_role_a_quality_evaluation.load_cells(self.make_manifest(root))
            selected = run_role_a_quality_evaluation.select_cells(cells, "formal", 64)
            jobs = run_role_a_quality_evaluation.build_jobs(
                selected, root / "cifar.zip", root / "out", "formal", "both", 29700
            )
            self.assertEqual(len(jobs), 12)
            self.assertEqual(
                {(job["method"], job["training_seed"], job["nfe"]) for job in jobs},
                {
                    (method, seed, nfe)
                    for method in ("sigmoid", "adaptive_v1")
                    for seed in range(3)
                    for nfe in (1, 2)
                },
            )
            for job in jobs:
                command = " ".join(job["command"])
                self.assertIn("--sample-seeds=0-4999", command)
                self.assertIn("--metrics=kid5k_full,fid5k_full", command)
                self.assertEqual(job["budget_kimg"], 64)
            with self.assertRaises(SystemExit):
                run_role_a_quality_evaluation.select_cells(selected[:-1], "formal", 64)

    def test_collector_writes_exact_required_columns_and_checks_repeats(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = []
            for method_index, method in enumerate(("sigmoid", "adaptive_v1")):
                for nfe in (1, 2):
                    output = root / method / f"nfe{nfe}"
                    output.mkdir(parents=True)
                    metric_names = ["kid512_full", "fid512_full"]
                    for metric_index, metric in enumerate(metric_names):
                        value = 1.0 + method_index + nfe / 10 + metric_index / 100
                        payload = {"metric": metric, "results": {metric: value}}
                        (output / f"metric-{metric}.jsonl").write_text(
                            json.dumps(payload) + "\n" + json.dumps(payload) + "\n",
                            encoding="utf-8",
                        )
                    jobs.append({
                        "method": method,
                        "training_seed": 0,
                        "budget_kimg": 16,
                        "nfe": nfe,
                        "sample_count": 512,
                        "sample_seeds": "0-511",
                        "metric_repeats": 2,
                        "metric_names": metric_names,
                        "checkpoint_sha256": f"sha-{method}",
                        "output_directory": str(output),
                        "status": "completed",
                    })
            manifest = {
                "protocol": "role-a-multibudget-quality-v1",
                "status": "completed",
                "phase": "smoke",
                "metric_mode": "both",
                "metric_names": ["kid512_full", "fid512_full"],
                "precision": "fp32",
                "sample_count": 512,
                "sample_seeds": "0-511",
                "dataset": "/data/cifar.zip",
                "dataset_sha256": "dataset-sha",
                "reference_real_count": 50000,
                "feature_detector_url": "detector",
                "jobs": jobs,
            }
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            outdir = root / "summary"
            collect_role_a_quality_results.main([
                "--eval-root", str(root), "--outdir", str(outdir)
            ])
            with (outdir / "role_a_metrics.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(
                    reader.fieldnames,
                    ["Method", "Train seed", "Budget", "NFE", "KID", "FID", "Checkpoint SHA"],
                )
            self.assertEqual(len(rows), 4)
            summary = json.loads((outdir / "role_a_metrics.json").read_text())
            self.assertTrue(summary["repeat_results_exact"])
            self.assertTrue(summary["repeat_results_numerically_consistent"])
            self.assertTrue(summary["reference_identity_consistent"])

    def test_collector_accepts_roundoff_but_records_nonexact_repeats(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metric-kid512_full.jsonl"
            payloads = [
                {"metric": "kid512_full", "results": {"kid512_full": value}}
                for value in (0.5, 0.5000001)
            ]
            path.write_text(
                "".join(json.dumps(payload) + "\n" for payload in payloads),
                encoding="utf-8",
            )
            value, exact, consistent = collect_role_a_quality_results.read_metric(
                path, "kid512_full", 2
            )
            self.assertEqual(value, 0.5)
            self.assertFalse(exact)
            self.assertTrue(consistent)


if __name__ == "__main__":
    unittest.main()
