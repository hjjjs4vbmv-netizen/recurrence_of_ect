#!/usr/bin/env python3
"""Launch one reproducible ECT training run from a dependency-free config."""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
NUMBERED_STATE_PATTERN = re.compile(r"training-state-(\d{6})\.pt$")
RESERVED_TRAIN_OPTIONS = {
    "--data",
    "--dry_run",
    "--nosubdir",
    "--outdir",
    "--resume",
    "--resume-tick",
    "--transfer",
}


@dataclass(frozen=True)
class ResumeCheckpoint:
    state: str
    snapshot: str
    tick: int
    source: str


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def append_jsonl(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def expand_environment(value, missing, environment=None):
    environment = os.environ if environment is None else environment
    if isinstance(value, dict):
        return {
            key: expand_environment(item, missing, environment)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_environment(item, missing, environment) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name = match.group(1)
        if name not in environment:
            missing.add(name)
            return match.group(0)
        return environment[name]

    return ENV_PATTERN.sub(replace, value)


def load_config(path: Path, environment=None):
    """Load JSON-compatible YAML without adding another runtime dependency."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise ValueError(f"config not found: {path}") from err
    except json.JSONDecodeError as err:
        raise ValueError(
            f"{path} must use JSON-compatible YAML syntax: {err}"
        ) from err
    missing = set()
    config = expand_environment(raw, missing, environment)
    if missing:
        variables = ", ".join(sorted(missing))
        raise ValueError(f"set the required environment variables first: {variables}")
    return config


def file_is_usable(path: Path, expected_size=None):
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    return expected_size is None or path.stat().st_size == int(expected_size)


def _safe_child(run_dir: Path, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"invalid checkpoint filename in manifest: {name!r}")
    return run_dir / name


def find_resume_checkpoint(run_dir: Path):
    """Return the newest complete checkpoint pair, preferring manifest on a tie."""
    candidates = []
    manifest_path = run_dir / "checkpoint-latest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = _safe_child(run_dir, manifest["state"])
            snapshot = _safe_child(run_dir, manifest["snapshot"])
            tick = int(manifest["tick"])
            if tick < 0:
                raise ValueError("negative checkpoint tick")
            if file_is_usable(state, manifest.get("state_bytes")) and file_is_usable(
                snapshot, manifest.get("snapshot_bytes")
            ):
                candidates.append(
                    ResumeCheckpoint(
                        str(state.resolve()),
                        str(snapshot.resolve()),
                        tick,
                        "latest-manifest",
                    )
                )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    if run_dir.is_dir():
        for state in run_dir.glob("training-state-[0-9][0-9][0-9][0-9][0-9][0-9].pt"):
            match = NUMBERED_STATE_PATTERN.fullmatch(state.name)
            if match is None:
                continue
            tick = int(match.group(1))
            snapshot = run_dir / f"network-snapshot-{tick:06d}.pkl"
            if file_is_usable(state) and file_is_usable(snapshot):
                candidates.append(
                    ResumeCheckpoint(
                        str(state.resolve()),
                        str(snapshot.resolve()),
                        tick,
                        "numbered-pair",
                    )
                )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item.tick, item.source == "latest-manifest"),
    )


def explicit_resume_checkpoint(state: Path, resume_tick=None):
    if not file_is_usable(state):
        raise ValueError(f"resume state is missing or empty: {state}")
    match = re.fullmatch(r"training-state-(\d+|latest)\.pt", state.name)
    if match is None:
        raise ValueError("resume file must be named training-state-<tick>.pt or training-state-latest.pt")
    checkpoint_id = match.group(1)
    snapshot = state.with_name(f"network-snapshot-{checkpoint_id}.pkl")
    if not file_is_usable(snapshot):
        raise ValueError(f"matching network snapshot is missing or empty: {snapshot}")
    if resume_tick is not None:
        tick = int(resume_tick)
    elif checkpoint_id != "latest":
        tick = int(checkpoint_id)
    else:
        manifest_checkpoint = find_resume_checkpoint(state.parent)
        if manifest_checkpoint is None or Path(manifest_checkpoint.state) != state.resolve():
            raise ValueError("latest checkpoint requires checkpoint-latest.json or --resume-tick")
        tick = manifest_checkpoint.tick
    if tick < 0:
        raise ValueError("resume tick must not be negative")
    return ResumeCheckpoint(
        str(state.resolve()), str(snapshot.resolve()), tick, "explicit"
    )


def git_value(repo_root: Path, *args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata(repo_root: Path):
    commit = git_value(repo_root, "rev-parse", "HEAD")
    status = git_value(repo_root, "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status), "branch": git_value(repo_root, "branch", "--show-current")}


def gpu_metadata():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    fields = ["index", "name", "uuid", "memory_total_mib", "driver_version"]
    return [dict(zip(fields, (part.strip() for part in line.split(",")))) for line in output.splitlines() if line.strip()]


def format_train_option(name, value):
    if not isinstance(name, str) or not name.startswith("--"):
        raise ValueError(f"training option must start with '--': {name!r}")
    if name in RESERVED_TRAIN_OPTIONS:
        raise ValueError(f"{name} is managed by the launcher and cannot appear in train")
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, bool):
            item = "True" if item else "False"
        result.append(f"{name}={item}")
    return result


def build_command(repo_root: Path, run_dir: Path, data: Path, config, checkpoint, transfer):
    launcher = config.get("launcher", {})
    gpus = int(launcher.get("gpus", 1))
    port = int(launcher.get("port", 29501))
    if gpus < 1:
        raise ValueError("launcher.gpus must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("launcher.port must be between 1 and 65535")
    train = config.get("train")
    if not isinstance(train, dict) or not train:
        raise ValueError("config.train must be a non-empty object")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={gpus}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=localhost:{port}",
        str(repo_root / "ct_train.py"),
        f"--outdir={run_dir}",
        f"--data={data}",
        "--nosubdir",
    ]
    for name, value in train.items():
        command.extend(format_train_option(name, value))
    if checkpoint is not None:
        command.extend(
            [f"--resume={checkpoint.state}", f"--resume-tick={checkpoint.tick}"]
        )
    else:
        command.append(f"--transfer={transfer}")
    return command


def has_existing_training_output(run_dir: Path):
    names = ["training_options.json", "stats.jsonl", "log.txt"]
    return any((run_dir / name).exists() for name in names) or any(
        run_dir.glob("training-state-*.pt")
    )


def resolve_path(value, config_dir: Path):
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--resume", help="auto, none, or a training-state path")
    parser.add_argument("--resume-tick", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.expanduser().resolve()
    try:
        config_environment = dict(os.environ)
        if args.run_dir is not None:
            config_environment["ECT_RUN_DIR"] = str(args.run_dir)
        if args.data is not None:
            config_environment["ECT_DATA_PATH"] = str(args.data)
        if args.transfer is not None:
            config_environment["ECT_TRANSFER_PATH"] = str(args.transfer)
        config = load_config(config_path, config_environment)
        config.setdefault("launcher", {})
        if args.gpus is not None:
            config["launcher"]["gpus"] = args.gpus
        if args.port is not None:
            config["launcher"]["port"] = args.port
        paths = config.get("paths", {})
        config_dir = config_path.parent
        run_dir = (args.run_dir or resolve_path(paths["run_dir"], config_dir)).resolve()
        data = (args.data or resolve_path(paths["data"], config_dir)).resolve()
        transfer_value = args.transfer or resolve_path(paths["transfer"], config_dir)
        transfer = transfer_value.resolve()

        git = git_metadata(repo_root)
        if git["commit"] is None:
            raise ValueError(f"not a Git repository: {repo_root}")
        if git["dirty"] and not args.allow_dirty:
            raise ValueError("working tree is dirty; commit first or use --allow-dirty for a preliminary run")
        if not data.is_file():
            raise ValueError(f"dataset not found: {data}")

        resume_mode = args.resume
        if resume_mode is None:
            resume_mode = "auto" if config["launcher"].get("auto_resume", True) else "none"
        if resume_mode == "auto":
            checkpoint = find_resume_checkpoint(run_dir)
        elif resume_mode == "none":
            checkpoint = None
            if has_existing_training_output(run_dir):
                raise ValueError(f"refusing a fresh run over existing output: {run_dir}")
        else:
            checkpoint = explicit_resume_checkpoint(Path(resume_mode).expanduser().resolve(), args.resume_tick)
        if checkpoint is None and not transfer.is_file():
            raise ValueError(f"transfer checkpoint not found: {transfer}")
        if resume_mode == "auto" and checkpoint is None and has_existing_training_output(run_dir):
            raise ValueError(f"existing run has no complete checkpoint pair: {run_dir}")

        command = build_command(repo_root, run_dir, data, config, checkpoint, transfer)
    except (KeyError, TypeError, ValueError) as err:
        raise SystemExit(f"[train] ERROR: {err}") from err

    print("[train] " + shlex.join(command), flush=True)
    if args.dry_run:
        print("[train] dry run; no files changed and training was not started")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    invocation_id = str(uuid.uuid4())
    metadata = {
        "format_version": 1,
        "invocation_id": invocation_id,
        "status": "running",
        "started_at": utc_now(),
        "config_file": str(config_path),
        "config_sha256": config_sha256,
        "submitted_config": config,
        "git": git,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": gpu_metadata(),
        },
        "paths": {"run_dir": str(run_dir), "data": str(data), "transfer": str(transfer)},
        "resume": asdict(checkpoint) if checkpoint is not None else None,
        "command": command,
    }
    atomic_write_json(run_dir / "run_metadata.json", metadata)
    append_jsonl(run_dir / "launcher_history.jsonl", metadata)

    try:
        completed = subprocess.run(command, cwd=repo_root, check=False)
        exit_code = completed.returncode
        if exit_code == 0:
            metadata["status"] = "completed"
        elif exit_code in {-15, -2, 130, 143}:
            metadata["status"] = "interrupted"
        else:
            metadata["status"] = "failed"
    except KeyboardInterrupt:
        exit_code = 130
        metadata["status"] = "interrupted"
    metadata["ended_at"] = utc_now()
    metadata["exit_code"] = exit_code
    atomic_write_json(run_dir / "run_metadata.json", metadata)
    append_jsonl(
        run_dir / "launcher_history.jsonl",
        {
            "invocation_id": invocation_id,
            "status": metadata["status"],
            "ended_at": metadata["ended_at"],
            "exit_code": exit_code,
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
