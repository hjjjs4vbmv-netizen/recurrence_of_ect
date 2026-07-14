#!/usr/bin/env python3
"""Verify prepared datasets and checkpoints without importing project code."""

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(path: Path, expected_count: int, expected_resolution: int):
    if not path.is_file():
        raise SystemExit(f"dataset not found: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"CRC failure: {corrupt}")
            names = archive.namelist()
            images = sorted(name for name in names if name.lower().endswith(".png"))
            if len(images) != expected_count:
                raise ValueError(f"expected {expected_count} images, found {len(images)}")
            if "dataset.json" not in names:
                raise ValueError("dataset.json is missing")
            metadata = json.loads(archive.read("dataset.json"))
            labels = metadata.get("labels")
            if labels is not None and len(labels) != expected_count:
                raise ValueError(
                    f"expected {expected_count} labels, found {len(labels)}"
                )
            for image_name in (images[0], images[-1]):
                header = archive.read(image_name)[:24]
                if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"invalid PNG header: {image_name}")
                width, height = struct.unpack(">II", header[16:24])
                if (width, height) != (expected_resolution, expected_resolution):
                    raise ValueError(
                        f"{image_name} is {width}x{height}, expected "
                        f"{expected_resolution}x{expected_resolution}"
                    )
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"dataset verification failed: {exc}") from exc

    return {
        "type": "dataset",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "images": expected_count,
        "resolution": [expected_resolution, expected_resolution],
        "labels": labels is not None,
    }


def verify_checkpoint(path: Path, expected_sha256: str):
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    size = path.stat().st_size
    if size < 10 * 1024 * 1024:
        raise SystemExit(f"checkpoint is unexpectedly small: {size} bytes")
    with path.open("rb") as handle:
        prefix = handle.read(2)
    if not prefix.startswith(b"\x80"):
        raise SystemExit("checkpoint does not look like a binary pickle")
    actual_sha256 = sha256(path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "type": "checkpoint",
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": actual_sha256,
    }


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="asset_type", required=True)

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("--path", type=Path, required=True)
    dataset_parser.add_argument("--expected-count", type=int, default=50000)
    dataset_parser.add_argument("--expected-resolution", type=int, default=32)
    dataset_parser.add_argument("--output", type=Path)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--path", type=Path, required=True)
    checkpoint_parser.add_argument("--expected-sha256", default="")
    checkpoint_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.asset_type == "dataset":
        report = verify_dataset(
            args.path, args.expected_count, args.expected_resolution
        )
    else:
        report = verify_checkpoint(args.path, args.expected_sha256)

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
