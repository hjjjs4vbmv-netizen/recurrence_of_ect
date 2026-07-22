import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from scripts import build_blind_ab
from scripts import build_final_conclusion
from scripts import collect_final_quality_results
from scripts import collect_final_stability
from scripts import run_final_evaluation_matrix
from scripts import score_blind_ab


class FinalEvaluationTest(unittest.TestCase):
    def make_manifest(self, root: Path):
        cells = []
        for seed in range(3):
            for schedule in ("sigmoid", "adaptive_v1"):
                checkpoint = root / f"{schedule}_seed{seed}.pkl"
                checkpoint.write_bytes(f"{schedule}-{seed}".encode())
                result_dir = root / f"{schedule}_seed{seed}_result"
                cells.append({
                    "schedule": schedule,
                    "training_seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": run_final_evaluation_matrix.sha256_file(checkpoint),
                    "training_result_dir": str(result_dir),
                })
        manifest = root / "checkpoints.json"
        manifest.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        return manifest, cells

    def test_frozen_runner_builds_exact_matrix(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _ = self.make_manifest(root)
            data = root / "cifar.zip"
            data.write_bytes(b"test-data")
            cells = run_final_evaluation_matrix.load_cells(manifest, allow_missing=False)
            quantitative = run_final_evaluation_matrix.quantitative_commands(
                cells, data, root / "out", 29600, "primary"
            )
            visual = run_final_evaluation_matrix.visual_commands(cells, root / "out")
            self.assertEqual(len(quantitative), 12)
            self.assertEqual(len(visual), 6)
            self.assertEqual(
                {(job["schedule"], job["training_seed"], job["nfe"]) for job in quantitative},
                {(schedule, seed, nfe) for schedule in ("sigmoid", "adaptive_v1") for seed in range(3) for nfe in (1, 2)},
            )
            for job in quantitative:
                command = " ".join(job["command"])
                self.assertIn("--sample-seeds=0-4999", command)
                self.assertIn("--metric-repeats=1", command)
                self.assertIn("--metrics=kid5k_full,fid5k_full", command)
                self.assertIn("--fp16=False", command)

    def make_metric_matrix(self, root: Path):
        for seed in range(3):
            for schedule in ("sigmoid", "adaptive_v1"):
                for nfe in (1, 2):
                    cell = root / "quantitative" / schedule / f"seed{seed}" / f"nfe{nfe}"
                    cell.mkdir(parents=True)
                    fixed_base = 10 + seed + nfe
                    values = {
                        "kid5k_full": fixed_base - (0.25 if schedule == "adaptive_v1" else 0),
                        "fid5k_full": fixed_base + 5 - (0.5 if schedule == "adaptive_v1" else 0),
                    }
                    for metric, value in values.items():
                        payload = {"metric": metric, "results": {metric: value}}
                        (cell / f"metric-{metric}.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def test_quantitative_collector_preserves_pairing(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_metric_matrix(root)
            outdir = root / "summary"
            collect_final_quality_results.main(["--eval-root", str(root), "--outdir", str(outdir)])
            summary = json.loads((outdir / "quantitative_summary.json").read_text())
            for nfe in (1, 2):
                kid = summary["summary_by_nfe"][str(nfe)]["kid5k_full"]
                self.assertAlmostEqual(kid["mean_delta"], -0.25)
                self.assertEqual(kid["adaptive_fixed_tie_seed_counts"], [3, 0, 0])

    def make_visual_samples(self, manifest: Path, sample_root: Path):
        cells = build_blind_ab.load_cells(manifest)
        for (schedule, training_seed), cell in cells.items():
            for nfe in (1, 2):
                for seed in range(16):
                    path = sample_root / cell["checkpoint_id"] / f"nfe{nfe}" / "images" / f"seed{seed:06d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    color = (20 + training_seed * 30, 40 + nfe * 40, 80 if schedule == "sigmoid" else 180)
                    Image.new("RGB", (32, 32), color).save(path)

    def test_blind_ballot_is_balanced_and_scores_complete_raters(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _ = self.make_manifest(root)
            sample_root = root / "samples"
            self.make_visual_samples(manifest, sample_root)
            public = root / "public"
            key_path = root / "private" / "key.csv"
            build_blind_ab.main([
                "--manifest", str(manifest),
                "--sample-root", str(sample_root),
                "--outdir", str(public),
                "--key-out", str(key_path),
            ])
            metadata = json.loads((public / "metadata.json").read_text())
            self.assertEqual(metadata["trial_count"], 96)
            self.assertEqual(metadata["side_balance"], {"adaptive_on_A": 48, "adaptive_on_B": 48})

            with key_path.open(newline="", encoding="utf-8") as handle:
                key_rows = list(csv.DictReader(handle))
            responses = []
            for rater in range(3):
                response_path = root / f"rater{rater}.csv"
                with response_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["trial_id", "rater_id", "preference"])
                    writer.writeheader()
                    for row in key_rows:
                        preference = "A" if row["A_schedule"] == "adaptive_v1" else "B"
                        writer.writerow({"trial_id": row["trial_id"], "rater_id": f"R{rater}", "preference": preference})
                responses.append(str(response_path))
            score_dir = root / "scores"
            score_blind_ab.main([
                "--key", str(key_path), "--responses", *responses, "--outdir", str(score_dir)
            ])
            score = json.loads((score_dir / "blind_ab_summary.json").read_text())
            overall = next(row for row in score["summary"] if row["stratum"] == "overall")
            self.assertEqual(overall["adaptive_v1"], 288)
            self.assertEqual(overall["sigmoid"], 0)

    def make_stability_results(self, cells):
        for cell in cells:
            result_dir = Path(cell["training_result_dir"])
            result_dir.mkdir()
            schedule = cell["schedule"]
            seed = cell["training_seed"]
            metadata = {
                "schedule": schedule,
                "seed": seed,
                "processed_kimg": 16.0,
                "nan_count": 0,
                "inf_count": 0,
                "skipped_steps": 9,
                "successful_optimizer_steps": 116,
                "final_grad_scale": 128,
                "peak_vram_mib": 2500 + seed,
                "wall_time_seconds": 100 + seed,
                "final_adaptive_active": True,
                "final_signal_updates": 32,
                "network_snapshot_sha256": cell["checkpoint_sha256"],
            }
            (result_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            fieldnames = ["attempted_iteration", "loss", "step_skipped", "schedule"]
            with (result_dir / "train_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for iteration in range(1, 126):
                    writer.writerow({
                        "attempted_iteration": iteration,
                        "loss": 10 + iteration / 100,
                        "step_skipped": "true" if iteration <= 9 else "false",
                        "schedule": schedule,
                    })

    def test_stability_and_final_conclusion_builders(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, cells = self.make_manifest(root)
            self.make_stability_results(cells)
            summary_dir = root / "summary"
            collect_final_stability.main(["--manifest", str(manifest), "--outdir", str(summary_dir)])
            stability = json.loads((summary_dir / "training_stability.json").read_text())
            self.assertTrue(stability["all_six_runs_complete"])
            self.assertEqual(stability["summary_by_schedule"]["adaptive_v1"]["controller_activated_runs"], 3)

    def test_one_page_conclusion_uses_locked_primary_direction(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quantitative = root / "quantitative"
            blind = root / "blind"
            stability = root / "stability"
            for path in (quantitative, blind, stability):
                path.mkdir()
            quantitative_summary = {
                "primary_metric": "kid5k_full",
                "summary_by_nfe": {
                    str(nfe): {
                        "kid5k_full": {
                            "mean_delta": -0.2,
                            "sample_sd_delta": 0.05,
                            "adaptive_fixed_tie_seed_counts": [3, 0, 0],
                        }
                    }
                    for nfe in (1, 2)
                },
            }
            (quantitative / "quantitative_summary.json").write_text(
                json.dumps(quantitative_summary), encoding="utf-8"
            )
            with (quantitative / "quantitative_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["schedule", "training_seed", "nfe", "kid5k_full"])
                writer.writeheader()
                for seed in range(3):
                    for nfe in (1, 2):
                        writer.writerow({"schedule": "sigmoid", "training_seed": seed, "nfe": nfe, "kid5k_full": 2.0})
                        writer.writerow({"schedule": "adaptive_v1", "training_seed": seed, "nfe": nfe, "kid5k_full": 1.8})
            (blind / "blind_ab_summary.json").write_text(json.dumps({
                "complete_raters": 3,
                "summary": [{
                    "stratum": "overall", "adaptive_v1": 160, "sigmoid": 100, "tie": 28,
                    "adaptive_tie_half_score": 174 / 288,
                }],
            }), encoding="utf-8")
            (stability / "training_stability.json").write_text(json.dumps({
                "all_six_runs_complete": True,
                "all_losses_finite": True,
                "summary_by_schedule": {"adaptive_v1": {"controller_activated_runs": 3}},
            }), encoding="utf-8")
            output = root / "FINAL_CONCLUSION.md"
            build_final_conclusion.main([
                "--quantitative-dir", str(quantitative),
                "--blind-dir", str(blind),
                "--stability-dir", str(stability),
                "--output", str(output),
            ])
            text = output.read_text(encoding="utf-8")
            self.assertIn("方向性支持 Adaptive v1", text)
            self.assertIn("不是标准 FID-50k benchmark", text)


if __name__ == "__main__":
    unittest.main()
