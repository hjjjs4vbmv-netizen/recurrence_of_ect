#!/usr/bin/env python3
"""Validate the frozen Day 1 ECT runtime and report the observed environment."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


# env.yml pins these packages directly. The Day 1 validated runtime additionally
# freezes huggingface-hub because newer API removals can break diffusers 0.26.3.
EXPECTED = {
    "torch": "2.3.0",
    "diffusers": "0.26.3",
    "accelerate": "0.27.2",
    "huggingface-hub": "0.23.4",
}
REQUIRED = [
    "numpy",
    "scipy",
    "Pillow",
    "click",
    "requests",
    "psutil",
    "tqdm",
    "imageio",
    "imageio-ffmpeg",
    "pyspng",
]
IMPORTS = [
    "torch",
    "numpy",
    "scipy",
    "PIL",
    "click",
    "requests",
    "psutil",
    "tqdm",
    "imageio",
    "pyspng",
    "diffusers",
    "accelerate",
    "huggingface_hub",
]


def git_value(repo_root: Path, *args: str):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    versions = {}
    errors = []
    for package in [*EXPECTED, *REQUIRED]:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package: {package}")
            continue
        versions[package] = actual
        expected = EXPECTED.get(package)
        if expected is not None:
            if package == "torch":
                matches = actual.startswith(expected)
            else:
                matches = actual == expected
            if not matches:
                errors.append(f"{package}: expected {expected}, got {actual}")

    import_status = {}
    for module in IMPORTS:
        try:
            importlib.import_module(module)
            import_status[module] = "ok"
        except Exception as exc:  # Import compatibility matters, not just installation.
            import_status[module] = f"failed: {type(exc).__name__}: {exc}"
            errors.append(f"cannot import {module}: {type(exc).__name__}: {exc}")

    try:
        import torch
    except ImportError:
        torch = None

    cuda_available = bool(torch and torch.cuda.is_available())
    if not args.allow_no_cuda and not cuda_available:
        errors.append("CUDA is unavailable")
    if sys.version_info[:3] != (3, 9, 18):
        errors.append(f"expected Python 3.9.18, got {platform.python_version()}")

    repo_root = Path(__file__).resolve().parents[1]
    status = git_value(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    report = {
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "git_branch": git_value(repo_root, "branch", "--show-current"),
        "git_dirty": bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "imports": import_status,
        "cuda": {
            "available": cuda_available,
            "runtime": torch.version.cuda if torch else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
        },
        "status": "passed" if not errors else "failed",
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("environment validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
