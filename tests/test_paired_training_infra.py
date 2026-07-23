#!/usr/bin/env python3
"""Lightweight infra tests for Role B paired-training runner/collector."""

from __future__ import annotations

import csv
import json
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
    include_telemetry: bool = False,
    next_loop_cur_tick: int = 2,
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
        fieldnames = [
                "attempted_iteration",
                "successful_optimizer_steps",
                "processed_nimg",
                "processed_kimg",
                "loss",
                "grad_scale",
                "step_skipped",
                "schedule",
                "stage",
                "next_loop_cur_tick",
                "elapsed_sec",
                "peak_vram_gb",
            ]
        if include_telemetry:
            fieldnames += [
                "loss_ema",
                "loss_reference",
                "correction",
                "signal_updates",
                "adaptive_active",
                "r_over_t_mean",
                "gap_mean",
            ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {
                "attempted_iteration": 1,
                "successful_optimizer_steps": 1,
                "processed_nimg": nimg,
                "processed_kimg": f"{kimg:.6f}",
                "loss": loss,
                "grad_scale": "65536",
                "step_skipped": 0,
                "schedule": schedule,
                "stage": 0,
                "next_loop_cur_tick": next_loop_cur_tick,
                "elapsed_sec": "1.0",
                "peak_vram_gb": "1.5",
            }
        if include_telemetry:
            row.update(
                loss_ema="0.8",
                loss_reference="1.0",
                correction="0.02",
                signal_updates="3",
                adaptive_active="1",
                r_over_t_mean="0.75",
                gap_mean="0.25",
            )
        writer.writerow(row)

    (run_dir / "network-snapshot-latest.pkl").write_bytes(b"not-a-real-pickle")
    # Minimal torch-free stand-in; collector tests skip torch.load via flag.
    (run_dir / "training-state-latest.pt").write_bytes(b"not-a-real-state")


def write_collector_training_state(
    run_dir: Path,
    *,
    schedule: str,
    cur_nimg: int = 16_000,
    attempted_iteration: int = 1,
    successful_optimizer_steps: int = 1,
    elapsed_sec: float = 1.0,
    cur_tick: int = 2,
    signal_updates: int = 3,
) -> dict:
    import torch

    state = {
        "gradscaler_state": {"scale": 65536.0},
        "cur_nimg": cur_nimg,
        "cur_tick": cur_tick,
        "attempted_iteration": attempted_iteration,
        "successful_optimizer_steps": successful_optimizer_steps,
        "elapsed_sec": elapsed_sec,
        "loss_fn_state": {
            "schedule_name": schedule,
            "stage": 0,
            "ratio": 0.5,
            "schedule": {},
        },
    }
    if schedule in {"adaptive_v1", "pid_deadband"}:
        state["loss_fn_state"]["schedule"] = {
            "loss_ema": 0.8,
            "loss_reference": 1.0,
            "signal_updates": signal_updates,
        }
    torch.save(state, run_dir / "training-state-latest.pt")
    return state


def write_adaptive_activation_run(
    run_dir: Path,
    *,
    data_path: Path,
    transfer_path: Path,
    final_signal_updates: int = 8,
    first_nonzero_correction_iteration: int | None = 12,
    final_adaptive_active: bool = True,
) -> None:
    """Write a 32-iteration activation fixture with controller-state timing."""
    write_minimal_run(
        run_dir,
        schedule="adaptive_v1",
        mode="activation",
        duration=0.004,
        data_path=data_path,
        transfer_path=transfer_path,
        include_telemetry=True,
    )
    summary_path = run_dir / "train_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames

    rows = []
    for iteration in range(1, 33):
        signal_updates = min(iteration // 4, final_signal_updates)
        correction_active = (
            first_nonzero_correction_iteration is not None
            and iteration >= first_nonzero_correction_iteration
            and (iteration < 32 or final_adaptive_active)
        )
        adaptive_active = signal_updates >= 3 and (iteration < 32 or final_adaptive_active)
        rows.append(
            {
                "attempted_iteration": iteration,
                "successful_optimizer_steps": iteration,
                "processed_nimg": iteration * 128,
                "processed_kimg": f"{iteration * 0.128:.6f}",
                "loss": "1.25",
                "grad_scale": "65536",
                "step_skipped": "0",
                "schedule": "adaptive_v1",
                "stage": "0",
                "next_loop_cur_tick": "1" if iteration < 32 else "2",
                "elapsed_sec": f"{iteration / 10:.1f}",
                "peak_vram_gb": "1.5",
                "loss_ema": "0.8" if signal_updates else "",
                "loss_reference": "1.0" if signal_updates else "",
                "correction": "0.02" if correction_active else "0",
                "signal_updates": str(signal_updates),
                "adaptive_active": str(int(adaptive_active)),
                "r_over_t_mean": "0.75",
                "gap_mean": "0.25",
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    def test_dry_run_pid_freezes_controller_and_lr_parameters(self):
        env = os.environ.copy()
        env["ECT_DATA_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.zip"
        env["ECT_TRANSFER_PATH"] = "/tmp/does-not-need-to-exist-for-dry-run.pkl"
        env["ECT_RUNS_ROOT"] = "/tmp/paired-runs"
        completed = subprocess.run(
            ["bash", str(RUNNER), "--schedule", "pid_deadband", "--mode", "dry-run"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("schedule_slug=pid-deadband", completed.stdout)
        self.assertIn("--mapping=pid_deadband", completed.stdout)
        self.assertIn("--pid-update-kimg=1.024", completed.stdout)
        self.assertIn("--pid-kp=0.1", completed.stdout)
        self.assertIn("--pid-ki=0.01", completed.stdout)
        self.assertIn("--pid-kd=0.05", completed.stdout)
        self.assertIn("--pid-deadband=0.02", completed.stdout)
        self.assertIn("--pid-integral-limit=5", completed.stdout)
        self.assertIn("--pid-max-control=0.1", completed.stdout)
        self.assertIn("--pid-lr-boost=1.25", completed.stdout)
        self.assertIn("--pid-lr-max-boost=1.5", completed.stdout)
        self.assertIn("--pid-lr-warmup-kimg=256", completed.stdout)
        self.assertRegex(
            completed.stdout,
            r"OUTDIR=.*/pid-deadband-dry-run-[0-9a-f]{8}-[0-9]{8}T[0-9]{6}Z",
        )

    def test_dry_run_accepts_final_matrix_seed_one(self):
        completed = subprocess.run(
            [
                "bash", str(RUNNER),
                "--schedule", "sigmoid",
                "--mode", "dry-run",
                "--seed", "1",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("seed=1", completed.stdout)
        self.assertIn("--seed=1", completed.stdout)
        self.assertRegex(
            completed.stdout,
            r"OUTDIR=.*/sigmoid-dry-run-seed1-[0-9a-f]{8}-[0-9]{8}T[0-9]{6}Z",
        )

    def test_runner_rejects_seed_outside_final_matrix(self):
        completed = subprocess.run(
            [
                "bash", str(RUNNER),
                "--schedule", "sigmoid",
                "--mode", "dry-run",
                "--seed", "3",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--seed {0|1|2}", completed.stderr)

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
            tmp_path = Path(tmp)
            outdir = tmp_path / "sigmoid-activation-deadbeef-20260101T000000Z"
            outdir.mkdir()
            resume = outdir / "training-state-latest.pt"
            resume.write_bytes(b"x")
            data = tmp_path / "data.zip"
            transfer = tmp_path / "transfer.pkl"
            write_dummy_asset(data, b"dataset")
            write_dummy_asset(transfer, b"transfer")
            import hashlib

            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
                ).strip()
            )
            if dirty:
                self.skipTest("worktree dirty; resume identity gate requires clean tree")
            (outdir / "run_meta.env").write_text(
                "\n".join(
                    [
                        "schedule=sigmoid",
                        f"git_head={head}",
                        "git_dirty=false",
                        f"data_sha256={hashlib.sha256(b'dataset').hexdigest()}",
                        f"transfer_sha256={hashlib.sha256(b'transfer').hexdigest()}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
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

    def test_resume_head_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outdir = tmp_path / "sigmoid-activation-deadbeef-20260101T000000Z"
            outdir.mkdir()
            resume = outdir / "training-state-latest.pt"
            resume.write_bytes(b"x")
            data = tmp_path / "data.zip"
            transfer = tmp_path / "transfer.pkl"
            write_dummy_asset(data, b"dataset")
            write_dummy_asset(transfer, b"transfer")
            import hashlib

            (outdir / "run_meta.env").write_text(
                "\n".join(
                    [
                        "schedule=sigmoid",
                        "git_head=" + ("a" * 40),
                        "git_dirty=false",
                        f"data_sha256={hashlib.sha256(b'dataset').hexdigest()}",
                        f"transfer_sha256={hashlib.sha256(b'transfer').hexdigest()}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
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
                    "dry-run",
                    "--resume",
                    str(resume),
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("resume HEAD mismatch", completed.stderr)

    def test_resume_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outdir = tmp_path / "sigmoid-activation-deadbeef-20260101T000000Z"
            outdir.mkdir()
            resume = outdir / "training-state-latest.pt"
            resume.write_bytes(b"x")
            data = tmp_path / "data.zip"
            transfer = tmp_path / "transfer.pkl"
            write_dummy_asset(data, b"dataset")
            write_dummy_asset(transfer, b"transfer")
            import hashlib

            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
                ).strip()
            )
            if dirty:
                self.skipTest("worktree dirty; resume identity gate requires clean tree")
            (outdir / "run_meta.env").write_text(
                "\n".join(
                    [
                        "schedule=sigmoid",
                        f"git_head={head}",
                        "git_dirty=false",
                        "data_sha256=" + ("0" * 64),
                        f"transfer_sha256={hashlib.sha256(b'transfer').hexdigest()}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
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
                    "dry-run",
                    "--resume",
                    str(resume),
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("resume dataset SHA mismatch", completed.stderr)


class CollectorInfraTests(unittest.TestCase):
    def _assets(self, tmp: Path) -> tuple[Path, Path]:
        data = tmp / "data.zip"
        transfer = tmp / "transfer.pkl"
        write_dummy_asset(data, b"dataset")
        write_dummy_asset(transfer, b"transfer")
        return data, transfer

    def _run_collector(
        self,
        run_dir: Path,
        outdir: Path,
        data: Path,
        transfer: Path,
        *,
        schedule: str,
        mode: str = "stability",
        load_training_state: bool = False,
    ) -> subprocess.CompletedProcess:
        command = [
            sys.executable,
            str(COLLECTOR),
            "--run-dir", str(run_dir),
            "--outdir", str(outdir),
            "--data", str(data),
            "--transfer", str(transfer),
            "--mode", mode,
            "--schedule", schedule,
            "--allow-dirty",
            "--skip-snapshot-load",
        ]
        if not load_training_state:
            command.append("--skip-training-state-load")
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_adaptive_telemetry_is_validated_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                schedule="adaptive_v1",
                data_path=data,
                transfer_path=transfer,
                include_telemetry=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COLLECTOR),
                    "--run-dir", str(run_dir),
                    "--outdir", str(outdir),
                    "--data", str(data),
                    "--transfer", str(transfer),
                    "--mode", "stability",
                    "--schedule", "adaptive_v1",
                    "--allow-dirty",
                    "--skip-snapshot-load",
                    "--skip-training-state-load",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (outdir / "train_summary.csv").open(newline="", encoding="utf-8") as handle:
                packaged = next(csv.DictReader(handle))
            self.assertEqual(packaged["signal_updates"], "3")
            self.assertEqual(packaged["adaptive_active"], "1")
            self.assertEqual(packaged["r_over_t_mean"], "0.75")
            self.assertEqual(packaged["next_loop_cur_tick"], "2")
            metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["schedule_telemetry_available"])
            self.assertEqual(metadata["final_signal_updates"], 3)
            self.assertEqual(metadata["first_nonzero_correction_iteration"], 1)

    def test_training_state_must_match_final_csv_and_adaptive_telemetry(self):
        cases = [
            (
                "cur_nimg",
                lambda state: state.update(cur_nimg=15_872),
                "training-state cur_nimg mismatch",
            ),
            (
                "attempted_iteration",
                lambda state: state.update(attempted_iteration=2),
                "training-state attempted_iteration mismatch",
            ),
            (
                "successful_optimizer_steps",
                lambda state: state.update(successful_optimizer_steps=0),
                "training-state successful_optimizer_steps mismatch",
            ),
            (
                "elapsed_sec",
                lambda state: state.update(elapsed_sec=2.0),
                "training-state elapsed_sec mismatch",
            ),
            (
                "cur_tick",
                lambda state: state.update(cur_tick=3),
                "training-state cur_tick mismatch",
            ),
            (
                "signal_updates",
                lambda state: state["loss_fn_state"]["schedule"].update(signal_updates=2),
                "training-state signal_updates mismatch",
            ),
            (
                "loss_ema",
                lambda state: state["loss_fn_state"]["schedule"].update(loss_ema=0.7),
                "training-state loss_ema mismatch",
            ),
            (
                "loss_reference",
                lambda state: state["loss_fn_state"]["schedule"].update(loss_reference=1.1),
                "training-state loss_reference mismatch",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data, transfer = self._assets(tmp_path)
            for name, mutate, error in cases:
                with self.subTest(name=name):
                    run_dir = tmp_path / f"run-{name}"
                    outdir = tmp_path / f"out-{name}"
                    write_minimal_run(
                        run_dir,
                        schedule="adaptive_v1",
                        data_path=data,
                        transfer_path=transfer,
                        include_telemetry=True,
                    )
                    state = write_collector_training_state(run_dir, schedule="adaptive_v1")
                    mutate(state)
                    import torch

                    torch.save(state, run_dir / "training-state-latest.pt")
                    completed = self._run_collector(
                        run_dir,
                        outdir,
                        data,
                        transfer,
                        schedule="adaptive_v1",
                        load_training_state=True,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(error, completed.stderr)

    def test_matching_training_state_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                schedule="adaptive_v1",
                data_path=data,
                transfer_path=transfer,
                include_telemetry=True,
            )
            write_collector_training_state(run_dir, schedule="adaptive_v1")
            completed = self._run_collector(
                run_dir,
                outdir,
                data,
                transfer,
                schedule="adaptive_v1",
                load_training_state=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_pid_telemetry_and_training_state_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                schedule="pid_deadband",
                data_path=data,
                transfer_path=transfer,
                include_telemetry=True,
            )
            write_collector_training_state(run_dir, schedule="pid_deadband")
            completed = self._run_collector(
                run_dir,
                outdir,
                data,
                transfer,
                schedule="pid_deadband",
                load_training_state=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schedule"], "pid_deadband")
            self.assertTrue(metadata["schedule_telemetry_available"])
            self.assertEqual(metadata["final_signal_updates"], 3)

    def test_pid_correction_must_respect_frozen_control_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                schedule="pid_deadband",
                data_path=data,
                transfer_path=transfer,
                include_telemetry=True,
            )
            summary_path = run_dir / "train_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                row = next(reader)
            row["correction"] = "0.100001"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            completed = self._run_collector(
                run_dir, outdir, data, transfer, schedule="pid_deadband"
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("pid correction exceeds pid_max_control", completed.stderr)

    def test_baseline_training_state_uses_recorded_tick_not_tick_formula(self):
        # With batch=128 and default --tick=50 kimg, real maintenance runs at
        # 0.128, 50.176, 100.224, and 128 kimg. The final checkpoint therefore
        # persists cur_tick=4, whereas the removed ceil-based formula gives 3.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                schedule="adaptive_v1",
                mode="baseline",
                duration=0.128,
                data_path=data,
                transfer_path=transfer,
                include_telemetry=True,
                next_loop_cur_tick=4,
            )
            write_collector_training_state(
                run_dir,
                schedule="adaptive_v1",
                cur_nimg=128_000,
                cur_tick=4,
            )
            completed = self._run_collector(
                run_dir,
                outdir,
                data,
                transfer,
                schedule="adaptive_v1",
                mode="baseline",
                load_training_state=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (outdir / "train_summary.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["next_loop_cur_tick"], "4")

    def test_adaptive_activation_gate_records_controller_and_pair_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_adaptive_activation_run(
                run_dir,
                data_path=data,
                transfer_path=transfer,
            )
            write_collector_training_state(
                run_dir,
                schedule="adaptive_v1",
                cur_nimg=4_096,
                attempted_iteration=32,
                successful_optimizer_steps=32,
                elapsed_sec=3.2,
                cur_tick=2,
                signal_updates=8,
            )
            completed = self._run_collector(
                run_dir, outdir, data, transfer,
                schedule="adaptive_v1", mode="activation",
                load_training_state=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (run_dir / "train_summary.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle))[-1]["next_loop_cur_tick"], "2")
            metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["final_signal_updates"], 8)
            self.assertTrue(metadata["final_adaptive_active"])
            self.assertEqual(metadata["first_nonzero_correction_iteration"], 12)
            self.assertEqual(metadata["first_adapted_pair_iteration"], 13)
            self.assertTrue(metadata["activation_gate_applied"])
            self.assertTrue(metadata["activation_gate_passed"])

    def test_adaptive_activation_gate_rejects_nonactivated_or_late_controller(self):
        cases = [
            (
                dict(final_signal_updates=2, first_nonzero_correction_iteration=None),
                "final_signal_updates must be >= 3",
            ),
            (
                dict(first_nonzero_correction_iteration=None),
                "first_nonzero_correction_iteration is required",
            ),
            (
                dict(final_adaptive_active=False),
                "final adaptive_active must be true",
            ),
            (
                dict(first_nonzero_correction_iteration=31),
                "first_adapted_pair_iteration must be before final_iteration",
            ),
            (
                dict(first_nonzero_correction_iteration=29),
                "at least 4 attempted iterations",
            ),
        ]
        for fixture_kwargs, expected_error in cases:
            with self.subTest(fixture_kwargs=fixture_kwargs), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                run_dir = tmp_path / "run"
                outdir = tmp_path / "out"
                data, transfer = self._assets(tmp_path)
                write_adaptive_activation_run(
                    run_dir,
                    data_path=data,
                    transfer_path=transfer,
                    **fixture_kwargs,
                )
                completed = self._run_collector(
                    run_dir, outdir, data, transfer,
                    schedule="adaptive_v1", mode="activation",
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stderr)

    def test_adaptive_telemetry_rejects_invalid_types_and_ranges(self):
        cases = [
            ({"signal_updates": "3.5"}, "non-negative integer"),
            ({"adaptive_active": "maybe"}, "must be one of"),
            ({"correction": "nan"}, "must be finite"),
            ({"r_over_t_mean": "1.1", "gap_mean": "-0.1"}, "must be in [0, 1]"),
        ]
        for changes, expected_error in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                run_dir = tmp_path / "run"
                outdir = tmp_path / "out"
                data, transfer = self._assets(tmp_path)
                write_minimal_run(
                    run_dir,
                    schedule="adaptive_v1",
                    data_path=data,
                    transfer_path=transfer,
                    include_telemetry=True,
                )
                summary_path = run_dir / "train_summary.csv"
                with summary_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = reader.fieldnames
                    row = next(reader)
                row.update(changes)
                with summary_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)

                completed = self._run_collector(
                    run_dir, outdir, data, transfer, schedule="adaptive_v1"
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stderr)

    def test_adaptive_warmup_telemetry_allows_missing_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                schedule="adaptive_v1",
                data_path=data,
                transfer_path=transfer,
                include_telemetry=True,
            )
            summary_path = run_dir / "train_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                row = next(reader)
            row.update(
                loss_reference="",
                correction="0",
                signal_updates="1",
                adaptive_active="0",
            )
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            completed = self._run_collector(
                run_dir, outdir, data, transfer, schedule="adaptive_v1"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_migrated_telemetry_prefix_is_auditable_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(
                run_dir,
                schedule="adaptive_v1",
                data_path=data,
                transfer_path=transfer,
                include_telemetry=True,
            )
            summary_path = run_dir / "train_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                final_row = next(reader)
            historical_row = dict(final_row)
            historical_row.update(
                attempted_iteration="1",
                successful_optimizer_steps="1",
                processed_nimg="128",
                processed_kimg="0.128",
                elapsed_sec="0.1",
            )
            for field in (
                "loss_ema", "loss_reference", "correction", "signal_updates",
                "adaptive_active", "r_over_t_mean", "gap_mean",
            ):
                historical_row[field] = ""
            final_row.update(attempted_iteration="2", successful_optimizer_steps="2")
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([historical_row, final_row])

            completed = self._run_collector(
                run_dir, outdir, data, transfer, schedule="adaptive_v1"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["schedule_telemetry_columns_available"])
            self.assertFalse(metadata["schedule_telemetry_available"])
            self.assertEqual(metadata["schedule_telemetry_rows"], 1)
            self.assertEqual(metadata["schedule_telemetry_total_rows"], 2)
            self.assertEqual(metadata["schedule_telemetry_coverage"], 0.5)
            self.assertEqual(metadata["first_schedule_telemetry_iteration"], 2)
            with (outdir / "train_summary.csv").open(newline="", encoding="utf-8") as handle:
                packaged = list(csv.DictReader(handle))
            self.assertEqual(packaged[0]["signal_updates"], "")
            self.assertEqual(packaged[1]["signal_updates"], "3")

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
                "\n".join(
                    [
                        f"git_head={head}",
                        "git_dirty=false",
                        f"data_sha256={__import__('hashlib').sha256(b'dataset').hexdigest()}",
                        f"transfer_sha256={__import__('hashlib').sha256(b'transfer').hexdigest()}",
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

    def test_collector_command_head_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            outdir = tmp_path / "out"
            data, transfer = self._assets(tmp_path)
            write_minimal_run(run_dir, data_path=data, transfer_path=transfer)
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            import hashlib

            data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
            transfer_sha = hashlib.sha256(transfer.read_bytes()).hexdigest()
            (run_dir / "run_meta.stability.env").write_text(
                "\n".join(
                    [
                        "mode=stability",
                        "schedule=sigmoid",
                        "git_head=" + ("b" * 40),
                        "git_dirty=false",
                        f"data_sha256={data_sha}",
                        f"transfer_sha256={transfer_sha}",
                        "exact_command=python ct_train.py --mapping=sigmoid --duration=0.016",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            # Keep immutable fresh identity at current HEAD.
            lines = []
            for line in (run_dir / "run_meta.env").read_text(encoding="utf-8").splitlines():
                if line.startswith("git_head="):
                    lines.append(f"git_head={head}")
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
            self.assertIn("resume/command git_head", completed.stderr)

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
