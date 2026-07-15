import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from scripts import export_results
from scripts import launch_training


def stat(mean):
    return {"num": 1, "mean": mean, "std": 0.0}


class LaunchTrainingTests(unittest.TestCase):
    def test_environment_expansion_reports_missing_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text('{"path": "${DAY2_REQUIRED_PATH}"}\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "DAY2_REQUIRED_PATH"):
                    launch_training.load_config(config)

    def test_manifest_checkpoint_is_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = run_dir / "training-state-latest.pt"
            snapshot = run_dir / "network-snapshot-latest.pkl"
            state.write_bytes(b"state")
            snapshot.write_bytes(b"snapshot")
            (run_dir / "training-state-000010.pt").write_bytes(b"old-state")
            (run_dir / "network-snapshot-000010.pkl").write_bytes(b"old-snapshot")
            (run_dir / "checkpoint-latest.json").write_text(
                json.dumps(
                    {
                        "tick": 12,
                        "state": state.name,
                        "snapshot": snapshot.name,
                        "state_bytes": state.stat().st_size,
                        "snapshot_bytes": snapshot.stat().st_size,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = launch_training.find_resume_checkpoint(run_dir)
            self.assertEqual(checkpoint.tick, 12)
            self.assertEqual(checkpoint.source, "latest-manifest")

    def test_invalid_manifest_falls_back_to_newest_numbered_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "checkpoint-latest.json").write_text("{}", encoding="utf-8")
            for tick in (2, 9):
                (run_dir / f"training-state-{tick:06d}.pt").write_bytes(b"state")
                (run_dir / f"network-snapshot-{tick:06d}.pkl").write_bytes(b"snapshot")
            checkpoint = launch_training.find_resume_checkpoint(run_dir)
            self.assertEqual(checkpoint.tick, 9)
            self.assertEqual(checkpoint.source, "numbered-pair")

    def test_newer_numbered_pair_wins_over_stale_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            latest_state = run_dir / "training-state-latest.pt"
            latest_snapshot = run_dir / "network-snapshot-latest.pkl"
            latest_state.write_bytes(b"state")
            latest_snapshot.write_bytes(b"snapshot")
            (run_dir / "checkpoint-latest.json").write_text(
                json.dumps(
                    {
                        "tick": 3,
                        "state": latest_state.name,
                        "snapshot": latest_snapshot.name,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "training-state-000004.pt").write_bytes(b"state")
            (run_dir / "network-snapshot-000004.pkl").write_bytes(b"snapshot")
            checkpoint = launch_training.find_resume_checkpoint(run_dir)
            self.assertEqual(checkpoint.tick, 4)
            self.assertEqual(checkpoint.source, "numbered-pair")

    def test_dry_run_accepts_path_overrides_without_environment_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            data = root / "data.zip"
            transfer = root / "transfer.pkl"
            run_dir = root / "run"
            data.write_bytes(b"data")
            transfer.write_bytes(b"checkpoint")
            config.write_text(
                json.dumps(
                    {
                        "paths": {
                            "run_dir": "${ECT_RUN_DIR}",
                            "data": "${ECT_DATA_PATH}",
                            "transfer": "${ECT_TRANSFER_PATH}",
                        },
                        "launcher": {"gpus": 1, "port": 29501},
                        "train": {"--duration": 0.001, "--batch": 10},
                    }
                ),
                encoding="utf-8",
            )
            result = launch_training.main(
                [
                    "--config",
                    str(config),
                    "--run-dir",
                    str(run_dir),
                    "--data",
                    str(data),
                    "--transfer",
                    str(transfer),
                    "--resume",
                    "none",
                    "--allow-dirty",
                    "--dry-run",
                ]
            )
            self.assertEqual(result, 0)
            self.assertFalse(run_dir.exists())


class ExportResultsTests(unittest.TestCase):
    def test_complete_bundle_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            output_dir = root / "bundle"
            run_dir.mkdir()
            options = {
                "dataset_kwargs": {"resolution": 32},
                "total_kimg": 2,
                "batch_size": 10,
            }
            (run_dir / "training_options.json").write_text(
                json.dumps(options), encoding="utf-8"
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "submitted_config": {
                            "experiment": {
                                "name": "unit-test",
                                "owner": "A",
                                "purpose": "verify exporter",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            records = []
            for tick, ratio in ((1, 0.5), (2, 0.75)):
                records.append(
                    {
                        "Progress/tick": stat(tick),
                        "Progress/kimg": stat(tick),
                        "Loss/loss": stat(1.0 / tick),
                        "Schedule/stage": stat(tick - 1),
                        "Schedule/ratio": stat(ratio),
                        "Timing/total_sec": stat(tick * 10),
                        "timestamp": 1000 + tick,
                    }
                )
            (run_dir / "stats.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
            )
            (run_dir / "metric-fid50k_full.jsonl").write_text(
                json.dumps(
                    {
                        "metric": "fid50k_full",
                        "results": {"fid50k_full": 4.2},
                        "snapshot_pkl": "network-snapshot-latest.pkl",
                        "total_time": 12.0,
                        "num_gpus": 1,
                        "timestamp": 1002,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            Image.new("RGB", (512, 512), "#336699").save(run_dir / "final.png")
            (run_dir / "training-state-latest.pt").write_bytes(b"state")
            (run_dir / "network-snapshot-latest.pkl").write_bytes(b"snapshot")

            result = export_results.main(
                ["--run-dir", str(run_dir), "--output-dir", str(output_dir)]
            )
            self.assertEqual(result, 0)
            expected = {
                "config.yaml",
                "metadata.json",
                "metrics.csv",
                "train_summary.csv",
                "samples_64.png",
                "schedule_curve.png",
                "notes.md",
            }
            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            with Image.open(output_dir / "samples_64.png") as sample:
                self.assertEqual(sample.size, (256, 256))
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["metric_rows"], 1)
            self.assertEqual(metadata["schedule_points"], 2)


if __name__ == "__main__":
    unittest.main()
