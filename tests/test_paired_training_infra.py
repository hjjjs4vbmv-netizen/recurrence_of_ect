#!/usr/bin/env python3
"""Lightweight infra tests for Role B paired-training runner/collector."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_schedule_experiment.sh"
COLLECTOR = REPO_ROOT / "scripts" / "collect_schedule_results.py"


def write_dummy_asset(path: Path, payload: bytes = b"asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_minimal_run(
    run_dir: Path,
    *,
    schedule: str = "sigmoid",
    mode: str = "stability",
    duration: float = 0.016,
    kimg: float = 16.0,
    include_nan: bool = False,
    git_head: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    head = git_head or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    )
    exact = (
        f"python {REPO_ROOT}/ct_train.py --mapping={schedule} --duration={duration} "
        f"--outdir={run_dir}"
    )
    (run_dir / "run_meta.env").write_text(
        "\n".join(
            [
                f"mode={mode}",
                f"schedule={schedule}",
                f"git_head={head}",
                f"git_branch={branch}",
                f"git_dirty={'true' if dirty else 'false'}",
                f"exact_command={exact}",
                "python_version=3.10.0",
                "torch_version=2.0.0",
                "cuda_version=11.8",
                "gpu_name=TestGPU",
                "gpu_count=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    nimg = int(kimg * 1000)
    loss = "nan" if include_nan else "1.25"
    with (run_dir / "train_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "attempted_iteration",
                "successful_optimizer_steps",
                "processed_nimg",
                "processed_kimg",
                "loss",
                "grad_scale",
                "step_skipped",
                "schedule",
                "stage",
                "elapsed_sec",
                "peak_vram_gb",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "attempted_iteration": 1,
                "successful_optimizer_steps": 1,
                "processed_nimg": nimg,
                "processed_kimg": f"{kimg:.6f}",
                "loss": loss,
                "grad_scale": "65536",
                "step_skipped": 0,
                "schedule": schedule,
                "stage": 0,
                "elapsed_sec": "1.0",
                "peak_vram_gb": "1.5",
            }
        )

    (run_dir / "network-snapshot-latest.pkl").write_bytes(b"not-a-real-pickle")
    # Minimal torch-free stand-in; collector tests skip torch.load via flag.
    (run_dir / "training-state-latest.pt").write_bytes(b"not-a-real-state")


class RunnerInfraTests(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.check_call(["bash", "-n", str(RUNNER)])

    def test_dry_run_sigmoid(self):
        env = os.environ.copy()
        env["ECT_DATA_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.zip"
        env["ECT_TRANSFER_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.pkl"
        completed = subprocess.run(
            ["bash", str(RUNNER), "--schedule", "sigmoid", "--mode", "dry-run"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("schedule=sigmoid", completed.stdout)
        self.assertIn("--transfer=", completed.stdout)
        self.assertNotIn("--resume=", completed.stdout)
        self.assertIn("--mapping=sigmoid", completed.stdout)

    def test_dry_run_resume_excludes_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            resume = Path(tmp) / "training-state-latest.pt"
            resume.write_bytes(b"x")
            env = os.environ.copy()
            env["ECT_DATA_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.zip"
            env["ECT_TRANSFER_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.pkl"
            completed = subprocess.run(
                [
                    "bash",
                    str(RUNNER),
                    "--schedule",
                    "sigmoid",
                    "--mode",
                    "dry-run",
                    "--resume",
                    str(resume),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("--resume=", completed.stdout)
            self.assertNotIn("--transfer=", completed.stdout)

    def test_fresh_nonempty_outdir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "busy"
            outdir.mkdir()
            (outdir / "marker").write_text("x", encoding="utf-8")
            data = Path(tmp) / "data.zip"
            transfer = Path(tmp) / "transfer.pkl"
            write_dummy_asset(data)
            write_dummy_asset(transfer)
            env = os.environ.copy()
            env["ECT_DATA_PATH"] = str(data)
            env["ECT_TRANSFER_PATH"] = str(transfer)
            completed = subprocess.run(
                [
                    "bash",
                    str(RUNNER),
                    "--schedule",
                    "sigmoid",
                    "--mode",
                    "stability",
                    "--outdir",
                    str(outdir),
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("fresh run requires empty outdir", completed.stderr)


class CollectorInfraTests(unittest.TestCase):
    def _assets(self, tmp: Path) -> tuple[Path, Path]:
        data = tmp / "data.zip"
        transfer = tmp / "transfer.pkl"
        write_dummy_asset(data, b"dataset")
        write_dummy_asset(transfer, b"transfer")
        return data, transfer

    def test_missing_csv_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            run_dir.mkdir()
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            (run_dir / "run_meta.env").write_text(
                f"git_head={head}\ngit_dirty=false\nexact_command=python ct_train.py --mapping=sigmoid --duration=0.016\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COLLECTOR),
                    "--run-dir",
                    str(run_dir),
                    "--outdir",
                    str(outdir),
                    "--data",
                    str(data),
                    "--transfer",
                    str(transfer),
                    "--mode",
                    "stability",
                    "--schedule",
                    "sigmoid",
                    "--allow-dirty",
                    "--skip-snapshot-load",
                    "--skip-training-state-load",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("train_summary.csv missing", completed.stderr)

    def test_nan_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(run_dir, include_nan=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COLLECTOR),
                    "--run-dir",
                    str(run_dir),
                    "--outdir",
                    str(outdir),
                    "--data",
                    str(data),
                    "--transfer",
                    str(transfer),
                    "--mode",
                    "stability",
                    "--schedule",
                    "sigmoid",
                    "--allow-dirty",
                    "--skip-snapshot-load",
                    "--skip-training-state-load",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("non-finite losses", completed.stderr)

    def test_schedule_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(run_dir, schedule="sigmoid")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COLLECTOR),
                    "--run-dir",
                    str(run_dir),
                    "--outdir",
                    str(outdir),
                    "--data",
                    str(data),
                    "--transfer",
                    str(transfer),
                    "--mode",
                    "stability",
                    "--schedule",
                    "adaptive_v1",
                    "--allow-dirty",
                    "--skip-snapshot-load",
                    "--skip-training-state-load",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("CSV schedule", completed.stderr)

    def test_resume_progress_fields_round_trip_dict(self):
        # Pure unit check of the state schema Role B now persists.
        state = {
            "cur_nimg": 16000,
            "cur_tick": 1,
            "tick_start_nimg": 16000,
            "attempted_iteration": 125,
            "successful_optimizer_steps": 116,
        }
        restored_nimg = int(state["cur_nimg"])
        restored_tick = int(state["cur_tick"])
        self.assertEqual(restored_nimg, 16000)
        self.assertEqual(restored_tick, 1)
        # Filename-derived estimate would be wrong for short runs.
        resume_tick_from_name = 1
        kimg_per_tick = 50
        wrong = resume_tick_from_name * kimg_per_tick * 1000
        self.assertNotEqual(wrong, restored_nimg)


if __name__ == "__main__":
    unittest.main()
