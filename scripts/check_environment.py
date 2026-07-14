#!/usr/bin/env python3
"""Validate the pinned ECT runtime and write a reproducibility report."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "torch": "2.3.0",
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "Pillow": "10.2.0",
    "click": "8.1.7",
    "requests": "2.31.0",
    "psutil": "5.9.8",
    "tqdm": "4.66.2",
    "imageio": "2.34.0",
    "imageio-ffmpeg": "0.4.9",
    "pyspng": "0.1.1",
    "diffusers": "0.26.3",
    "accelerate": "0.27.2",
    "huggingface-hub": "0.20.3",
}


def get_commit(repo_root: Path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    versions = {}
    errors = []
    for package, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package: {package}")
            continue
        versions[package] = actual
        if package == "torch":
            if not actual.startswith(expected):
                errors.append(f"{package}: expected {expected}, got {actual}")
        elif actual != expected:
            errors.append(f"{package}: expected {expected}, got {actual}")

    try:
        import torch
    except ImportError as exc:
        errors.append(f"cannot import torch: {exc}")
        torch = None

    cuda_available = bool(torch and torch.cuda.is_available())
    if not args.allow_no_cuda and not cuda_available:
        errors.append("CUDA is unavailable")
    if sys.version_info[:3] != (3, 9, 18):
        errors.append(f"expected Python 3.9.18, got {platform.python_version()}")

    repo_root = Path(__file__).resolve().parents[1]
    report = {
        "git_commit": get_commit(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
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
        "status": "ok" if not errors else "failed",
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("Environment validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
