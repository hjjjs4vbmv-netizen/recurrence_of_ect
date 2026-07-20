#!/usr/bin/env python3
"""Validate a paired-schedule run and package compact results for the repo.

Fails closed if train_summary.csv is missing, loss has NaN/Inf, checkpoints are
unreadable, gradscaler_state is absent, train-time git metadata disagrees with
the packaging HEAD, or CLI schedule/mode/duration disagree with the CSV / command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path


TELEMETRY_FIELDS = (
    "loss_ema",
    "loss_reference",
    "correction",
    "signal_updates",
    "adaptive_active",
    "r_over_t_mean",
    "gap_mean",
)

OUTPUT_FIELDS = (
    "attempted_iteration",
    "successful_optimizer_steps",
    "processed_kimg",
    "loss",
    "grad_scale",
    "step_skipped",
    "schedule",
    "stage",
    *TELEMETRY_FIELDS,
    "seconds",
    "peak_vram_mib",
)

MODE_DURATION_MIMG = {
    "activation": 0.004,
    "stability": 0.016,
    "baseline": 0.128,
}


def expected_final_nimg(duration_mimg: float, global_batch: int) -> int:
    """Match ct_train/ct_training_loop discrete batch completion.

    total_kimg = max(int(duration_mimg * 1000), 1)
    training stops once cur_nimg >= total_kimg * 1000 after a full batch.
    """
    if global_batch <= 0:
        fail(f"global_batch must be positive, got {global_batch}")
    target_kimg = max(int(duration_mimg * 1000), 1)
    target_nimg = target_kimg * 1000
    return math.ceil(target_nimg / global_batch) * global_batch


def fail(message: str) -> None:
    raise SystemExit(f"[collect_schedule_results] ERROR: {message}")


def run_git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"git {' '.join(args)} failed: {exc}")
    return completed.stdout.strip()


def collect_packaging_git_metadata(repo_root: Path) -> dict:
    head = run_git(repo_root, "rev-parse", "HEAD")
    branch = run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    porcelain = run_git(repo_root, "status", "--porcelain")
    return {
        "packaging_git_commit": head,
        "packaging_git_branch": branch,
        "packaging_git_dirty": bool(porcelain),
    }


def parse_run_meta(path: Path) -> dict[str, str]:
    if not path.is_file():
        fail(f"run_meta.env missing: {path}")
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key] = value
    return meta


def load_identity_and_command_meta(run_dir: Path, mode: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split immutable train identity from the packaging-mode command meta.

    run_meta.env is written once on the fresh segment and must not be overwritten.
    Resume segments write run_meta.<mode>.env / run_meta.latest.env.
    """
    identity_path = run_dir / "run_meta.env"
    identity = parse_run_meta(identity_path)
    command_candidates = [
        run_dir / f"run_meta.{mode}.env",
        run_dir / "run_meta.latest.env",
        identity_path,
    ]
    command_meta = None
    for path in command_candidates:
        if path.is_file():
            command_meta = parse_run_meta(path)
            break
    if command_meta is None:
        fail(f"no run_meta sidecar found under {run_dir}")
    return identity, command_meta


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_strict_bool(value: str | None, field: str) -> bool:
    text = "" if value is None else str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    fail(f"{field} must be one of 0/1/false/true, got {value!r}")


def parse_finite_float(value: str | None, field: str) -> float:
    text = "" if value is None else str(value).strip()
    if text == "":
        fail(f"{field} must not be empty")
    try:
        number = float(text)
    except ValueError:
        fail(f"{field} must be numeric, got {value!r}")
    if not math.isfinite(number):
        fail(f"{field} must be finite, got {value!r}")
    return number


def parse_optional_float(value: str | None, field: str) -> float | None:
    text = "" if value is None else str(value).strip()
    return None if text == "" else parse_finite_float(text, field)


def parse_nonnegative_integer(value: str | None, field: str) -> int:
    number = parse_finite_float(value, field)
    if not number.is_integer() or number < 0:
        fail(f"{field} must be a non-negative integer, got {value!r}")
    return int(number)


def load_rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"train_summary.csv missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        fail(f"train_summary.csv has no data rows: {path}")
    required = {
        "attempted_iteration",
        "successful_optimizer_steps",
        "processed_nimg",
        "loss",
        "grad_scale",
        "step_skipped",
        "elapsed_sec",
        "peak_vram_gb",
        "schedule",
    }
    if "processed_nimg" not in fieldnames and "nimg" in fieldnames:
        fieldnames.add("processed_nimg")
        for row in rows:
            row["processed_nimg"] = row["nimg"]
    if "attempted_iteration" not in fieldnames and "update" in fieldnames:
        fieldnames.add("attempted_iteration")
        for row in rows:
            row["attempted_iteration"] = row["update"]
    if "successful_optimizer_steps" not in fieldnames:
        fieldnames.add("successful_optimizer_steps")
        skipped_so_far = 0
        for idx, row in enumerate(rows, start=1):
            if parse_boolish(row.get("step_skipped", "0")):
                skipped_so_far += 1
            row["successful_optimizer_steps"] = str(idx - skipped_so_far)
    if "peak_vram_gb" not in fieldnames and "peak_gpu_mem_gb" in fieldnames:
        fieldnames.add("peak_vram_gb")
        for row in rows:
            row["peak_vram_gb"] = row["peak_gpu_mem_gb"]
    if "schedule" not in fieldnames:
        fieldnames.add("schedule")
        for row in rows:
            row.setdefault("schedule", "")
    missing = required - fieldnames
    if missing:
        fail(f"train_summary.csv missing columns: {sorted(missing)}")
    return rows


def choose_snapshot(run_dir: Path) -> Path:
    latest = run_dir / "network-snapshot-latest.pkl"
    if latest.is_file() and latest.stat().st_size > 0:
        return latest
    numbered = sorted(run_dir.glob("network-snapshot-[0-9][0-9][0-9][0-9][0-9][0-9].pkl"))
    numbered = [path for path in numbered if path.stat().st_size > 0]
    if not numbered:
        fail(f"no non-empty network-snapshot under {run_dir}")
    return numbered[-1]


def load_snapshot(snapshot: Path, repo_root: Path) -> None:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    with snapshot.open("rb") as handle:
        pickle.load(handle)


def choose_training_state(run_dir: Path) -> Path:
    latest = run_dir / "training-state-latest.pt"
    if latest.is_file() and latest.stat().st_size > 0:
        return latest
    numbered = sorted(run_dir.glob("training-state-[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    numbered = [path for path in numbered if path.stat().st_size > 0]
    if not numbered:
        fail(f"no non-empty training-state under {run_dir}")
    return numbered[-1]


def extract_exact_command(run_meta: dict[str, str], log_path: Path | None, exact_command_file: Path | None) -> str:
    if exact_command_file is not None and exact_command_file.is_file():
        for line in exact_command_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("exact_command="):
                text = line[len("exact_command=") :].strip()
                if text:
                    return text
    if "exact_command" in run_meta and run_meta["exact_command"].strip():
        return run_meta["exact_command"].strip()
    if log_path is not None and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("exact_command="):
                text = line[len("exact_command=") :].strip()
                if text:
                    return text
    fail("exact_command not found in run_meta.env / log / --exact-command-file")


def extract_mapping_from_command(exact_command: str) -> str | None:
    try:
        tokens = shlex.split(exact_command)
    except ValueError:
        tokens = exact_command.split()
    for token in tokens:
        if token.startswith("--mapping="):
            return token.split("=", 1)[1]
        if token.startswith("--schedule="):
            return token.split("=", 1)[1]
    return None


def extract_duration_from_command(exact_command: str) -> float | None:
    try:
        tokens = shlex.split(exact_command)
    except ValueError:
        tokens = exact_command.split()
    for token in tokens:
        if token.startswith("--duration="):
            return float(token.split("=", 1)[1])
    return None


def collect_runtime_metadata(run_meta: dict[str, str]) -> dict:
    meta = {
        "python_version": run_meta.get("python_version") or sys.version.split()[0],
        "platform": run_meta.get("platform") or platform.platform(),
    }
    if "torch_version" in run_meta:
        meta["torch_version"] = run_meta.get("torch_version")
        meta["cuda_version"] = run_meta.get("cuda_version")
        meta["gpu_name"] = run_meta.get("gpu_name") or None
        meta["gpu_count"] = int(run_meta["gpu_count"]) if run_meta.get("gpu_count") else 0
        return meta
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["gpu_count"] = torch.cuda.device_count()
        else:
            meta["gpu_name"] = None
            meta["gpu_count"] = 0
    except Exception as exc:  # noqa: BLE001
        meta["torch_import_error"] = str(exc)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--wall-time", type=Path)
    parser.add_argument("--exact-command-file", type=Path)
    parser.add_argument("--data", type=Path, required=True, help="Dataset zip for SHA256 metadata")
    parser.add_argument("--transfer", type=Path, required=True, help="EDM transfer pickle for SHA256 metadata")
    parser.add_argument("--mode", required=True, choices=sorted(MODE_DURATION_MIMG))
    parser.add_argument("--schedule", required=True, choices=("sigmoid", "adaptive_v1", "const"))
    parser.add_argument("--duration-mimg", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--batch-gpu", type=int, default=16)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit packaging when the packaging worktree is dirty (not for formal evidence)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty --outdir (default: fail closed)",
    )
    parser.add_argument(
        "--skip-snapshot-load",
        action="store_true",
        help="Skip pickle.load of network snapshot (tests only)",
    )
    parser.add_argument(
        "--skip-training-state-load",
        action="store_true",
        help="Skip torch.load of training-state (tests only)",
    )
    args = parser.parse_args(argv)

    expected_duration = MODE_DURATION_MIMG[args.mode]
    if args.duration_mimg is None:
        args.duration_mimg = expected_duration
    elif not math.isclose(args.duration_mimg, expected_duration, rel_tol=0.0, abs_tol=1e-12):
        fail(
            f"--duration-mimg={args.duration_mimg} disagrees with mode={args.mode} "
            f"(expected {expected_duration})"
        )

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        fail(f"run dir does not exist: {run_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    packaging_git = collect_packaging_git_metadata(repo_root)
    if packaging_git["packaging_git_dirty"] and not args.allow_dirty:
        fail("git worktree is dirty; commit/stash first or pass --allow-dirty for preliminary packaging")

    run_meta, command_meta = load_identity_and_command_meta(run_dir, args.mode)
    train_git_head = run_meta.get("git_head")
    if not train_git_head or train_git_head == "unknown":
        fail("run_meta.env missing git_head from training time")
    if train_git_head != packaging_git["packaging_git_commit"]:
        fail(
            f"train-time git_head={train_git_head} != packaging HEAD="
            f"{packaging_git['packaging_git_commit']}"
        )
    train_git_dirty = parse_boolish(run_meta.get("git_dirty", "false"))
    if train_git_dirty and not args.allow_dirty:
        fail("run_meta.env records git_dirty=true; refuse formal packaging without --allow-dirty")

    # Dual provenance: every resume segment meta must agree with the immutable fresh identity.
    cmd_git_head = command_meta.get("git_head")
    if not cmd_git_head or cmd_git_head == "unknown":
        fail("command meta missing git_head")
    if cmd_git_head != train_git_head:
        fail(
            f"resume/command git_head={cmd_git_head} != fresh git_head={train_git_head}"
        )
    cmd_git_dirty = parse_boolish(command_meta.get("git_dirty", "true"))
    if cmd_git_dirty and not args.allow_dirty:
        fail("command meta records git_dirty=true; refuse formal packaging without --allow-dirty")
    cmd_data_sha = command_meta.get("data_sha256")
    cmd_transfer_sha = command_meta.get("transfer_sha256")
    fresh_data_sha = run_meta.get("data_sha256")
    fresh_transfer_sha = run_meta.get("transfer_sha256")
    if not fresh_data_sha or fresh_data_sha in {"missing", "unknown"}:
        fail("run_meta.env missing training-time data_sha256")
    if not fresh_transfer_sha or fresh_transfer_sha in {"missing", "unknown"}:
        fail("run_meta.env missing training-time transfer_sha256")
    if not cmd_data_sha or cmd_data_sha in {"missing", "unknown"}:
        fail("command meta missing data_sha256")
    if not cmd_transfer_sha or cmd_transfer_sha in {"missing", "unknown"}:
        fail("command meta missing transfer_sha256")
    if cmd_data_sha != fresh_data_sha:
        fail(
            f"command data_sha256={cmd_data_sha} != fresh data_sha256={fresh_data_sha}"
        )
    if cmd_transfer_sha != fresh_transfer_sha:
        fail(
            f"command transfer_sha256={cmd_transfer_sha} != fresh "
            f"transfer_sha256={fresh_transfer_sha}"
        )

    source_csv = run_dir / "train_summary.csv"
    rows = load_rows(source_csv)

    schedules = {str(row.get("schedule", "")).strip() for row in rows}
    if len(schedules) != 1:
        fail(f"train_summary.csv schedule not unique: {sorted(schedules)}")
    csv_schedule = next(iter(schedules))
    if csv_schedule != args.schedule:
        fail(f"CSV schedule={csv_schedule!r} != --schedule={args.schedule!r}")
    telemetry_columns = [field in rows[0] for field in TELEMETRY_FIELDS]
    if any(telemetry_columns) and not all(telemetry_columns):
        fail("train_summary.csv has a partial schedule telemetry schema")
    telemetry_columns_available = all(telemetry_columns)
    if args.schedule == "adaptive_v1" and not telemetry_columns_available:
        fail("adaptive_v1 train_summary.csv missing stable schedule telemetry columns")

    exact_command = extract_exact_command(command_meta, args.log, args.exact_command_file)
    cmd_schedule = extract_mapping_from_command(exact_command)
    if cmd_schedule is None:
        fail("exact_command missing --mapping=/--schedule=")
    if cmd_schedule != args.schedule:
        fail(f"exact_command schedule={cmd_schedule!r} != --schedule={args.schedule!r}")
    cmd_duration = extract_duration_from_command(exact_command)
    if cmd_duration is None:
        fail("exact_command missing --duration=")
    if not math.isclose(cmd_duration, args.duration_mimg, rel_tol=0.0, abs_tol=1e-12):
        fail(f"exact_command duration={cmd_duration} != expected {args.duration_mimg}")

    losses: list[float] = []
    grad_scales: list[float] = []
    skipped = 0
    packaged: list[dict] = []
    nan_count = 0
    inf_count = 0
    previous_signal_updates = 0
    telemetry_started = False
    telemetry_rows = 0
    first_telemetry_iteration = None

    for row in rows:
        loss = float(row["loss"])
        if math.isnan(loss):
            nan_count += 1
        if math.isinf(loss):
            inf_count += 1
        step_skipped = parse_boolish(row["step_skipped"])
        if step_skipped:
            skipped += 1
        nimg = float(row["processed_nimg"])
        peak_gb = float(row["peak_vram_gb"])
        grad_scale = float(row["grad_scale"])
        losses.append(loss)
        grad_scales.append(grad_scale)
        telemetry = {
            "loss_ema": None,
            "loss_reference": None,
            "correction": None,
            "signal_updates": None,
            "adaptive_active": None,
            "r_over_t_mean": None,
            "gap_mean": None,
        }
        row_has_telemetry = telemetry_columns_available and any(
            str(row.get(field, "")).strip() for field in TELEMETRY_FIELDS
        )
        if telemetry_columns_available and not row_has_telemetry:
            if telemetry_started:
                fail(f"schedule telemetry gap at attempted_iteration={row['attempted_iteration']}")
        elif row_has_telemetry:
            telemetry_started = True
            telemetry_rows += 1
            if first_telemetry_iteration is None:
                first_telemetry_iteration = int(float(row["attempted_iteration"]))
            telemetry.update(
                loss_ema=parse_optional_float(row["loss_ema"], "loss_ema"),
                loss_reference=parse_optional_float(row["loss_reference"], "loss_reference"),
                correction=parse_finite_float(row["correction"], "correction"),
                signal_updates=parse_nonnegative_integer(row["signal_updates"], "signal_updates"),
                adaptive_active=parse_strict_bool(row["adaptive_active"], "adaptive_active"),
                r_over_t_mean=parse_finite_float(row["r_over_t_mean"], "r_over_t_mean"),
                gap_mean=parse_finite_float(row["gap_mean"], "gap_mean"),
            )
            if telemetry["signal_updates"] < previous_signal_updates:
                fail("schedule signal_updates is not monotonic")
            previous_signal_updates = telemetry["signal_updates"]
            if not 0 <= telemetry["r_over_t_mean"] <= 1:
                fail(f"r_over_t_mean must be in [0, 1], got {telemetry['r_over_t_mean']}")
            if not 0 <= telemetry["gap_mean"] <= 1:
                fail(f"gap_mean must be in [0, 1], got {telemetry['gap_mean']}")
            if not math.isclose(
                telemetry["r_over_t_mean"] + telemetry["gap_mean"],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                fail(f"r_over_t_mean + gap_mean != 1 at attempted_iteration={row['attempted_iteration']}")
            if (telemetry["loss_ema"] is None) != (telemetry["loss_reference"] is None):
                fail("loss_ema and loss_reference must both be present or both be empty")
            if telemetry["loss_ema"] is not None and telemetry["loss_ema"] <= 0:
                fail(f"loss_ema must be > 0, got {telemetry['loss_ema']}")
            if telemetry["loss_reference"] is not None and telemetry["loss_reference"] <= 0:
                fail(f"loss_reference must be > 0, got {telemetry['loss_reference']}")
            if telemetry["signal_updates"] == 0 and telemetry["loss_ema"] is not None:
                fail("signal_updates=0 requires empty loss_ema/loss_reference")
            if telemetry["signal_updates"] > 0 and telemetry["loss_ema"] is None:
                fail("positive signal_updates requires loss_ema/loss_reference")
            if not telemetry["adaptive_active"] and telemetry["correction"] != 0:
                fail("nonzero correction requires adaptive_active=true")
            if telemetry["adaptive_active"] and telemetry["signal_updates"] == 0:
                fail("adaptive_active=true requires positive signal_updates")
            if args.schedule != "adaptive_v1" and (
                telemetry["signal_updates"] != 0
                or telemetry["adaptive_active"]
                or telemetry["correction"] != 0
                or telemetry["loss_ema"] is not None
            ):
                fail(f"fixed schedule {args.schedule!r} contains adaptive controller telemetry")

        packaged.append(
            {
                "attempted_iteration": int(float(row["attempted_iteration"])),
                "successful_optimizer_steps": int(float(row["successful_optimizer_steps"])),
                "processed_kimg": nimg / 1000.0,
                "loss": loss,
                "grad_scale": grad_scale,
                "step_skipped": "true" if step_skipped else "false",
                "schedule": row.get("schedule", args.schedule),
                "stage": row.get("stage", ""),
                **telemetry,
                "seconds": float(row["elapsed_sec"]),
                "peak_vram_mib": peak_gb * 1024.0,
            }
        )

    if args.schedule == "adaptive_v1" and telemetry_rows == 0:
        fail("adaptive_v1 train_summary.csv contains no populated schedule telemetry rows")

    attempted = len(packaged)
    successful = attempted - skipped
    last_attempted = packaged[-1]["attempted_iteration"]
    last_successful = packaged[-1]["successful_optimizer_steps"]
    if last_attempted != attempted:
        fail(f"attempted_iteration mismatch: csv_last={last_attempted} row_count={attempted}")
    if last_successful != successful:
        fail(
            f"successful_optimizer_steps mismatch: csv_last={last_successful} "
            f"derived={successful}"
        )
    if successful + skipped != attempted:
        fail(
            f"update identity failed: successful({successful}) + skipped({skipped}) "
            f"!= attempted({attempted})"
        )
    if nan_count or inf_count:
        fail(f"non-finite losses: nan_count={nan_count} inf_count={inf_count}")

    final_kimg = packaged[-1]["processed_kimg"]
    expected_nimg = expected_final_nimg(args.duration_mimg, args.global_batch)
    expected_kimg = expected_nimg / 1000.0
    if not math.isclose(final_kimg, expected_kimg, rel_tol=0.0, abs_tol=1e-6):
        fail(
            f"final processed_kimg={final_kimg} != expected {expected_kimg} "
            f"(duration_mimg={args.duration_mimg}, batch={args.global_batch}, "
            f"expected_nimg={expected_nimg}) for mode={args.mode}"
        )

    if args.log is not None and args.log.is_file():
        hits = [
            line
            for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines()
            if re.search(r"nan|inf", line, flags=re.IGNORECASE)
        ]
        print(f"[collect_schedule_results] log nan|inf grep hits: {len(hits)}")

    snapshot = choose_snapshot(run_dir)
    if not args.skip_snapshot_load:
        load_snapshot(snapshot, repo_root)
        print(f"[collect_schedule_results] loaded snapshot: {snapshot}")

    training_state = choose_training_state(run_dir)
    if not args.skip_training_state_load:
        import torch

        state = torch.load(training_state, map_location="cpu")
        if not isinstance(state, dict):
            fail(f"training-state is not a dict: {training_state}")
        if "gradscaler_state" not in state:
            fail(f"gradscaler_state missing in {training_state}")
        gradscaler_state = state["gradscaler_state"]
        if gradscaler_state is None or gradscaler_state == {}:
            fail(f"gradscaler_state empty in {training_state}")
        for key in ("cur_nimg", "cur_tick", "attempted_iteration", "successful_optimizer_steps", "elapsed_sec"):
            if key not in state:
                fail(f"{key} missing in {training_state}")
        print(f"[collect_schedule_results] loaded training-state: {training_state}")

    wall_time = None
    if args.wall_time is not None and args.wall_time.is_file():
        wall_text = args.wall_time.read_text(encoding="utf-8").strip()
        if wall_text:
            wall_time = float(wall_text.splitlines()[-1])
    if wall_time is None:
        wall_time = float(packaged[-1]["seconds"])

    dataset_sha = sha256_file(args.data)
    transfer_sha = sha256_file(args.transfer)
    if not dataset_sha:
        fail(f"dataset SHA256 unavailable: {args.data}")
    if not transfer_sha:
        fail(f"transfer SHA256 unavailable: {args.transfer}")
    train_data_sha = run_meta.get("data_sha256")
    train_transfer_sha = run_meta.get("transfer_sha256")
    if dataset_sha != train_data_sha:
        fail(
            f"dataset SHA mismatch: packaging={dataset_sha} "
            f"train-time={train_data_sha}"
        )
    if transfer_sha != train_transfer_sha:
        fail(
            f"transfer SHA mismatch: packaging={transfer_sha} "
            f"train-time={train_transfer_sha}"
        )

    runtime = collect_runtime_metadata(command_meta)
    evidence_class = (
        "preliminary"
        if packaging_git["packaging_git_dirty"] or train_git_dirty or args.allow_dirty
        else "formal_candidate"
    )

    metadata = {
        "git_commit": train_git_head,
        "git_branch": run_meta.get("git_branch"),
        "git_dirty": train_git_dirty,
        **packaging_git,
        "exact_command": exact_command,
        "seed": args.seed,
        "global_batch": args.global_batch,
        "batch_gpu": args.batch_gpu,
        "processed_kimg": packaged[-1]["processed_kimg"],
        "attempted_iterations": attempted,
        "successful_optimizer_steps": successful,
        "skipped_steps": skipped,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "initial_grad_scale": grad_scales[0],
        "final_grad_scale": grad_scales[-1],
        "wall_time_seconds": wall_time,
        "peak_vram_mib": max(row["peak_vram_mib"] for row in packaged),
        "peak_vram_source": "torch.cuda.max_memory_allocated",
        "network_snapshot": str(snapshot),
        "training_state": str(training_state),
        "network_snapshot_sha256": sha256_file(snapshot),
        "training_state_sha256": sha256_file(training_state),
        "dataset_sha256": dataset_sha,
        "transfer_sha256": transfer_sha,
        "gradscaler_state_saved": True,
        "metrics_enabled": False,
        "mode": args.mode,
        "schedule": args.schedule,
        "schedule_telemetry_columns_available": telemetry_columns_available,
        "schedule_telemetry_available": telemetry_rows == len(packaged),
        "schedule_telemetry_rows": telemetry_rows,
        "schedule_telemetry_total_rows": len(packaged),
        "schedule_telemetry_coverage": telemetry_rows / len(packaged),
        "first_schedule_telemetry_iteration": first_telemetry_iteration,
        "final_signal_updates": next(
            (
                row["signal_updates"] for row in reversed(packaged)
                if row["signal_updates"] is not None
            ),
            None,
        ),
        "first_nonzero_correction_iteration": next(
            (
                row["attempted_iteration"] for row in packaged
                if row["correction"] is not None and row["correction"] != 0
            ),
            None,
        ),
        "duration_mimg": args.duration_mimg,
        "evidence_class": evidence_class,
        **runtime,
    }

    outdir = args.outdir
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        fail(f"outdir is not empty: {outdir}; pass --overwrite to replace packaged evidence")
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "train_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in packaged:
            writer.writerow(
                {
                    "attempted_iteration": row["attempted_iteration"],
                    "successful_optimizer_steps": row["successful_optimizer_steps"],
                    "processed_kimg": f"{row['processed_kimg']:.6f}",
                    "loss": f"{row['loss']:.8f}",
                    "grad_scale": f"{row['grad_scale']:.8g}",
                    "step_skipped": row["step_skipped"],
                    "schedule": row["schedule"],
                    "stage": row["stage"],
                    "loss_ema": "" if row["loss_ema"] is None else f"{row['loss_ema']:.12g}",
                    "loss_reference": "" if row["loss_reference"] is None else f"{row['loss_reference']:.12g}",
                    "correction": "" if row["correction"] is None else f"{row['correction']:.12g}",
                    "signal_updates": "" if row["signal_updates"] is None else row["signal_updates"],
                    "adaptive_active": "" if row["adaptive_active"] is None else int(row["adaptive_active"]),
                    "r_over_t_mean": "" if row["r_over_t_mean"] is None else f"{row['r_over_t_mean']:.12g}",
                    "gap_mean": "" if row["gap_mean"] is None else f"{row['gap_mean']:.12g}",
                    "seconds": f"{row['seconds']:.6f}",
                    "peak_vram_mib": f"{row['peak_vram_mib']:.6f}",
                }
            )

    metadata_path = outdir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"[collect_schedule_results] wrote {summary_path}")
    print(f"[collect_schedule_results] wrote {metadata_path}")
    print(
        f"[collect_schedule_results] PASS attempted={attempted} "
        f"successful={successful} skipped={skipped}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - surface collector failures clearly
        if isinstance(exc, SystemExit):
            raise
        fail(str(exc))
