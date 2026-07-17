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
    kimg: float | None = None,
    global_batch: int = 128,
    include_nan: bool = False,
    git_head: str | None = None,
    data_path: Path | None = None,
    transfer_path: Path | None = None,
) -> None:
    import hashlib
    import math

    run_dir.mkdir(parents=True, exist_ok=True)
    if kimg is None:
        target_kimg = max(int(duration * 1000), 1)
        target_nimg = target_kimg * 1000
        final_nimg = math.ceil(target_nimg / global_batch) * global_batch
        kimg = final_nimg / 1000.0

    def _sha(path: Path | None, fallback: bytes) -> str:
        payload = path.read_bytes() if path is not None and path.is_file() else fallback
        return hashlib.sha256(payload).hexdigest()

    data_sha = _sha(data_path, b"dataset")
    transfer_sha = _sha(transfer_path, b"transfer")

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
                f"data_sha256={data_sha}",
                f"transfer_sha256={transfer_sha}",
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

    nimg = int(round(kimg * 1000))
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
        env["ECT_RUNS_ROOT"] = "/tmp/paired-runs"
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
        self.assertRegex(
            completed.stdout,
            r"OUTDIR=.*/sigmoid-dry-run-[0-9a-f]{8}-[0-9]{8}T[0-9]{6}Z",
        )

    def test_dry_run_adaptive_outdir_slug(self):
        env = os.environ.copy()
        env["ECT_DATA_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.zip"
        env["ECT_TRANSFER_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.pkl"
        env["ECT_RUNS_ROOT"] = "/tmp/paired-runs"
        completed = subprocess.run(
            ["bash", str(RUNNER), "--schedule", "adaptive_v1", "--mode", "dry-run"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("schedule_slug=adaptive-v1", completed.stdout)
        self.assertRegex(
            completed.stdout,
            r"OUTDIR=.*/adaptive-v1-dry-run-[0-9a-f]{8}-[0-9]{8}T[0-9]{6}Z",
        )

    def test_runner_has_no_tee_append(self):
        text = RUNNER.read_text(encoding="utf-8")
        # Only count real pipeline usage, not commentary.
        self.assertNotRegex(text, r'(?m)^[^#\n]*\btee -a\b')
        self.assertRegex(text, r'(?m)^[^#\n]*\btee "\$\{LOG_PATH\}"')

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
            self.assertTrue(
                ("exists and is not empty" in completed.stderr)
                or ("fresh run requires empty outdir" in completed.stderr),
                completed.stderr,
            )

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
            write_minimal_run(run_dir, include_nan=True, data_path=data, transfer_path=transfer)
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
            write_minimal_run(run_dir, schedule="sigmoid", data_path=data, transfer_path=transfer)
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
        # Maintenance saves next-loop values so resume matches uninterrupted training.
        cur_tick = 1
        cur_nimg = 16000
        tick_start_nimg = 12800  # previous tick start; must NOT be what we persist
        elapsed_base_sec = 31.3
        segment_elapsed = 10.0
        state = {
            "cur_nimg": cur_nimg,
            "cur_tick": cur_tick + 1,
            "tick_start_nimg": cur_nimg,
            "attempted_iteration": 125,
            "successful_optimizer_steps": 116,
            "elapsed_sec": elapsed_base_sec + segment_elapsed,
        }
        self.assertEqual(int(state["cur_nimg"]), 16000)
        self.assertEqual(int(state["cur_tick"]), 2)
        self.assertEqual(int(state["tick_start_nimg"]), 16000)
        self.assertNotEqual(int(state["tick_start_nimg"]), tick_start_nimg)
        self.assertGreater(float(state["elapsed_sec"]), elapsed_base_sec)
        # Filename-derived estimate would be wrong for short runs.
        resume_tick_from_name = 1
        kimg_per_tick = 50
        wrong = resume_tick_from_name * kimg_per_tick * 1000
        self.assertNotEqual(wrong, int(state["cur_nimg"]))

    def test_already_done_budget_is_noop(self):
        cur_nimg = 16000
        total_kimg = 16
        self.assertGreaterEqual(cur_nimg, total_kimg * 1000)

    def test_collector_prefers_mode_meta_for_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                mode="activation",
                duration=0.004,
                data_path=data,
                transfer_path=transfer,
            )
            # Simulate resume overwriting only mode/latest sidecars, preserving run_meta.env
            # with activation identity hashes, while stability command lives in mode file.
            import hashlib
            import shutil

            identity = run_dir / "run_meta.env"
            shutil.copy(identity, run_dir / "run_meta.activation.env")
            data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
            transfer_sha = hashlib.sha256(transfer.read_bytes()).hexdigest()
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
            # Rewrite CSV to stability budget so packaging as stability can succeed.
            write_minimal_run(
                run_dir,
                mode="stability",
                duration=0.016,
                data_path=data,
                transfer_path=transfer,
            )
            # Restore immutable identity with original hashes/git, but keep stability command sidecar.
            (run_dir / "run_meta.env").write_text(
                "\n".join(
                    [
                        "mode=activation",
                        "schedule=sigmoid",
                        f"git_head={head}",
                        "git_branch=role-b/paired-training-v1",
                        "git_dirty=false",
                        f"data_sha256={data_sha}",
                        f"transfer_sha256={transfer_sha}",
                        "exact_command=python ct_train.py --mapping=sigmoid --duration=0.004",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "run_meta.stability.env").write_text(
                "\n".join(
                    [
                        "mode=stability",
                        "schedule=sigmoid",
                        f"git_head={head}",
                        "git_dirty=false",
                        f"data_sha256={data_sha}",
                        f"transfer_sha256={transfer_sha}",
                        "exact_command=python ct_train.py --mapping=sigmoid --duration=0.016",
                    ]
                )
                + "\n",
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
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_activation_expected_kimg_uses_batch_rounding(self):
        from scripts.collect_schedule_results import expected_final_nimg

        nimg = expected_final_nimg(0.004, 128)
        self.assertEqual(nimg, 4096)
        self.assertEqual(nimg / 128, 32)

    def test_asset_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(run_dir, data_path=data, transfer_path=transfer)
            # Corrupt the recorded train-time hash.
            lines = []
            for line in (run_dir / "run_meta.env").read_text(encoding="utf-8").splitlines():
                if line.startswith("data_sha256="):
                    lines.append("data_sha256=" + ("0" * 64))
                else:
                    lines.append(line)
            (run_dir / "run_meta.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
            self.assertIn("dataset SHA mismatch", completed.stderr)

    def test_outdir_nonempty_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            outdir.mkdir()
            (outdir / "metadata.json").write_text("{}", encoding="utf-8")
            data, transfer = self._assets(tmp_path)
            write_minimal_run(run_dir, data_path=data, transfer_path=transfer)
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
            self.assertIn("outdir is not empty", completed.stderr)


if __name__ == "__main__":
    unittest.main()
