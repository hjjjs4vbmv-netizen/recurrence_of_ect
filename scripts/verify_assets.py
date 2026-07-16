#!/usr/bin/env python3
"""Verify ECT datasets and transfer checkpoints without changing the assets."""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OFFICIAL_CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_tarball(path: Path, expected_md5: str) -> dict:
    if not path.is_file():
        raise ValueError(f"source tarball not found: {path}")
    actual_md5 = digest(path, "md5")
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"CIFAR-10 tarball MD5 mismatch: expected {expected_md5}, got {actual_md5}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "md5": actual_md5,
        "expected_md5": expected_md5.lower(),
        "md5_ok": True,
    }


def verify_dataset(
    path: Path,
    tarball: Path,
    expected_md5: str,
    expected_count: int,
    expected_labels: int,
    expected_resolution: int,
) -> dict:
    """Validate archive integrity, every image header, labels, and project loading."""
    if not path.is_file():
        raise ValueError(f"dataset not found: {path}")

    source = verify_tarball(tarball, expected_md5)
    sample_names = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"ZIP CRC failure: {corrupt}")

            names = archive.namelist()
            images = sorted(name for name in names if name.lower().endswith(".png"))
            if len(images) != expected_count:
                raise ValueError(f"expected {expected_count} PNGs, found {len(images)}")
            if "dataset.json" not in names:
                raise ValueError("dataset.json is missing")

            metadata = json.loads(archive.read("dataset.json"))
            labels = metadata.get("labels")
            if not isinstance(labels, list):
                raise ValueError("dataset.json must contain a labels list")
            if len(labels) != expected_labels:
                raise ValueError(
                    f"expected {expected_labels} labels, found {len(labels)}"
                )
            label_names = [entry[0] for entry in labels if isinstance(entry, list) and len(entry) == 2]
            if len(label_names) != expected_labels:
                raise ValueError("every label must be a [filename, class] pair")
            if sorted(label_names) != images:
                raise ValueError("dataset labels do not map one-to-one to the PNG entries")

            expected_size = (expected_resolution, expected_resolution)
            for image_name in images:
                with Image.open(io.BytesIO(archive.read(image_name))) as image:
                    if image.size != expected_size:
                        raise ValueError(
                            f"{image_name} is {image.size}, expected {expected_size}"
                        )
                    if image.mode != "RGB":
                        raise ValueError(f"{image_name} is {image.mode}, expected RGB")

            sample_names = [images[0], images[len(images) // 2], images[-1]]
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"dataset archive verification failed: {exc}") from exc

    # Import only after archive checks so failures clearly identify the project loader stage.
    from training.dataset import ImageFolderDataset

    dataset = ImageFolderDataset(path=str(path), use_labels=True)
    try:
        if len(dataset) != expected_count:
            raise ValueError(
                f"ImageFolderDataset length is {len(dataset)}, expected {expected_count}"
            )
        if not dataset.has_labels:
            raise ValueError("ImageFolderDataset did not expose class labels")

        loader_samples = []
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            image, label = dataset[index]
            if image.shape != (3, expected_resolution, expected_resolution):
                raise ValueError(f"dataset sample {index} has shape {image.shape}")
            if image.dtype != np.uint8:
                raise ValueError(f"dataset sample {index} has dtype {image.dtype}")
            loader_samples.append(
                {
                    "index": index,
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "label_shape": list(label.shape),
                }
            )
    finally:
        dataset.close()

    return {
        "type": "dataset",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest(path, "sha256"),
        "sha256_policy": "record_only_not_a_unique_content_requirement",
        "source_tarball": source,
        "zip_crc_ok": True,
        "images": expected_count,
        "labels": expected_labels,
        "resolution": [expected_resolution, expected_resolution],
        "color_mode": "RGB",
        "archive_samples": sample_names,
        "imagefolderdataset": {
            "length": expected_count,
            "has_labels": True,
            "samples": loader_samples,
        },
    }


def verify_checkpoint(path: Path, expected_sha256: str) -> dict:
    if not path.is_file():
        raise ValueError(f"checkpoint not found: {path}")
    size = path.stat().st_size
    if size < 10 * 1024 * 1024:
        raise ValueError(f"checkpoint is unexpectedly small: {size} bytes")
    with path.open("rb") as handle:
        prefix = handle.read(2)
    if not prefix.startswith(b"\x80"):
        raise ValueError("checkpoint does not look like a binary pickle")
    actual_sha256 = digest(path, "sha256")
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "type": "checkpoint",
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256.lower() if expected_sha256 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="asset_type", required=True)

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("--path", type=Path, required=True)
    dataset_parser.add_argument("--tarball", type=Path, required=True)
    dataset_parser.add_argument("--expected-md5", default=OFFICIAL_CIFAR10_MD5)
    dataset_parser.add_argument("--expected-count", type=int, default=50000)
    dataset_parser.add_argument("--expected-labels", type=int, default=50000)
    dataset_parser.add_argument("--expected-resolution", type=int, default=32)
    dataset_parser.add_argument("--output", type=Path)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--path", type=Path, required=True)
    checkpoint_parser.add_argument("--expected-sha256", default="")
    checkpoint_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        if args.asset_type == "dataset":
            report = verify_dataset(
                args.path,
                args.tarball,
                args.expected_md5,
                args.expected_count,
                args.expected_labels,
                args.expected_resolution,
            )
        else:
            report = verify_checkpoint(args.path, args.expected_sha256)
    except ValueError as exc:
        raise SystemExit(f"asset verification failed: {exc}") from exc

    report["status"] = "passed"
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
