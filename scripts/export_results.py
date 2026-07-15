#!/usr/bin/env python3
"""Export a compact, Git-friendly result bundle from one ECT run."""

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw


SUMMARY_COLUMNS = [
    "tick",
    "kimg",
    "loss",
    "schedule_stage",
    "schedule_ratio",
    "total_sec",
    "sec_per_tick",
    "sec_per_kimg",
    "cpu_mem_gb",
    "gpu_mem_gb",
    "gpu_reserved_gb",
    "timestamp",
]
SCHEDULE_LOG_PATTERN = re.compile(
    r"Update scheduler at\s+(\d+)\s+ticks,\s+([0-9.eE+-]+)\s+kimg,\s+ratio\s+([0-9.eE+-]+)"
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def stats_value(record, name):
    value = record.get(name)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def read_stats(run_dir: Path):
    path = run_dir / "stats.jsonl"
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as err:
            raise ValueError(f"invalid stats.jsonl line {line_number}: {err}") from err
    return records


def summary_rows(stats):
    mapping = {
        "tick": "Progress/tick",
        "kimg": "Progress/kimg",
        "loss": "Loss/loss",
        "schedule_stage": "Schedule/stage",
        "schedule_ratio": "Schedule/ratio",
        "total_sec": "Timing/total_sec",
        "sec_per_tick": "Timing/sec_per_tick",
        "sec_per_kimg": "Timing/sec_per_kimg",
        "cpu_mem_gb": "Resources/cpu_mem_gb",
        "gpu_mem_gb": "Resources/peak_gpu_mem_gb",
        "gpu_reserved_gb": "Resources/peak_gpu_mem_reserved_gb",
        "timestamp": "timestamp",
    }
    return [
        {column: stats_value(record, source) for column, source in mapping.items()}
        for record in stats
    ]


def write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(run_dir: Path):
    rows = []
    for path in sorted(run_dir.glob("metric-*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"invalid {path.name} line {line_number}: {err}") from err
            results = record.get("results", {})
            for result_name, value in results.items():
                rows.append(
                    {
                        "metric": record.get("metric"),
                        "result": result_name,
                        "value": value,
                        "snapshot_pkl": record.get("snapshot_pkl"),
                        "total_time_seconds": record.get("total_time"),
                        "num_gpus": record.get("num_gpus"),
                        "timestamp": record.get("timestamp"),
                        "source_file": path.name,
                    }
                )
    return rows


def choose_sample(run_dir: Path):
    final = run_dir / "final.png"
    if final.is_file():
        return final
    numbered = [
        path for path in run_dir.glob("*.png") if re.fullmatch(r"\d{6}\.png", path.name)
    ]
    return max(numbered, key=lambda item: item.name) if numbered else None


def write_placeholder(path: Path, text):
    image = Image.new("RGB", (768, 512), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 747, 491), outline="#9ca3af", width=3)
    draw.text((48, 228), text, fill="#374151")
    image.save(path)


def export_samples_64(source: Path, output: Path, resolution):
    with Image.open(source) as image:
        image = image.convert("RGB")
        if resolution is None:
            if image.width % 16 != 0 or image.height % 16 != 0:
                raise ValueError("cannot infer sample cell size; dataset resolution is missing")
            resolution = min(image.width // 16, image.height // 16)
        width = int(resolution) * 8
        height = int(resolution) * 8
        if image.width < width or image.height < height:
            raise ValueError(f"sample grid is smaller than 8x8 cells: {source}")
        image.crop((0, 0, width, height)).save(output)


def schedule_points(stats, log_path: Path):
    points = []
    for record in stats:
        x = stats_value(record, "Progress/kimg")
        y = stats_value(record, "Schedule/ratio")
        if x is not None and y is not None:
            points.append((float(x), float(y)))
    if points or not log_path.is_file():
        return points
    for match in SCHEDULE_LOG_PATTERN.finditer(
        log_path.read_text(encoding="utf-8", errors="replace")
    ):
        points.append((float(match.group(2)), float(match.group(3))))
    return points


def draw_schedule_curve(points, output: Path):
    width, height = 1200, 720
    left, top, right, bottom = 100, 65, 1140, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 22), "Schedule ratio over training", fill="#111827")
    draw.line((left, top, left, bottom), fill="#111827", width=2)
    draw.line((left, bottom, right, bottom), fill="#111827", width=2)
    draw.text((right - 70, bottom + 30), "kimg", fill="#374151")
    draw.text((20, top - 10), "ratio", fill="#374151")
    for index in range(6):
        ratio = index / 5
        y = bottom - ratio * (bottom - top)
        draw.line((left - 6, y, right, y), fill="#e5e7eb", width=1)
        draw.text((52, y - 7), f"{ratio:.1f}", fill="#6b7280")
    if not points:
        draw.text((430, 330), "No schedule data exported", fill="#6b7280")
        image.save(output)
        return
    points = sorted(points)
    min_x = min(item[0] for item in points)
    max_x = max(item[0] for item in points)
    if max_x == min_x:
        max_x = min_x + 1

    def project(point):
        x, ratio = point
        px = left + (x - min_x) / (max_x - min_x) * (right - left)
        py = bottom - max(0.0, min(1.0, ratio)) * (bottom - top)
        return px, py

    projected = [project(point) for point in points]
    if len(projected) > 1:
        draw.line(projected, fill="#2563eb", width=4, joint="curve")
    for px, py in projected:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#1d4ed8")
    draw.text((left, bottom + 30), f"{min_x:g}", fill="#6b7280")
    draw.text((right - 35, bottom + 30), f"{max_x:g}", fill="#6b7280")
    image.save(output)


def checkpoint_inventory(run_dir: Path):
    inventory = []
    for pattern in ("network-snapshot-*.pkl", "training-state-*.pt"):
        for path in sorted(run_dir.glob(pattern)):
            inventory.append({"name": path.name, "bytes": path.stat().st_size})
    return inventory


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--notes-file", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"[export] ERROR: run directory not found: {run_dir}")
    try:
        options = load_json(run_dir / "training_options.json")
        metadata = load_json(run_dir / "run_metadata.json", {})
        stats = read_stats(run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as err:
        raise SystemExit(f"[export] ERROR: {err}") from err

    missing = []
    if options is None:
        missing.append("training_options.json")
    if not stats:
        missing.append("stats.jsonl")
    sample_source = choose_sample(run_dir)
    if sample_source is None:
        missing.append("final or numbered sample PNG")
    if missing and not args.allow_incomplete:
        raise SystemExit("[export] ERROR: incomplete run: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    submitted_config = metadata.get("submitted_config") if isinstance(metadata, dict) else None
    config_export = {
        "format_version": 1,
        "submitted_config": submitted_config,
        "resolved_training_options": options,
    }
    (output_dir / "config.yaml").write_text(
        json.dumps(config_export, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = summary_rows(stats)
    write_csv(output_dir / "train_summary.csv", SUMMARY_COLUMNS, rows)
    metric_columns = [
        "metric",
        "result",
        "value",
        "snapshot_pkl",
        "total_time_seconds",
        "num_gpus",
        "timestamp",
        "source_file",
    ]
    metrics = metric_rows(run_dir)
    write_csv(output_dir / "metrics.csv", metric_columns, metrics)

    resolution = None
    if isinstance(options, dict):
        resolution = options.get("dataset_kwargs", {}).get("resolution")
    try:
        if sample_source is not None:
            export_samples_64(sample_source, output_dir / "samples_64.png", resolution)
        else:
            write_placeholder(output_dir / "samples_64.png", "Samples unavailable in this preliminary run")
    except (OSError, ValueError) as err:
        if not args.allow_incomplete:
            raise SystemExit(f"[export] ERROR: {err}") from err
        write_placeholder(output_dir / "samples_64.png", f"Sample export failed: {err}")

    points = schedule_points(stats, run_dir / "log.txt")
    draw_schedule_curve(points, output_dir / "schedule_curve.png")

    export_metadata = {
        "format_version": 1,
        "exported_at": utc_now(),
        "run_dir": str(run_dir),
        "run_metadata": metadata,
        "last_tick": rows[-1]["tick"] if rows else None,
        "last_kimg": rows[-1]["kimg"] if rows else None,
        "metric_rows": len(metrics),
        "schedule_points": len(points),
        "sample_source": sample_source.name if sample_source else None,
        "checkpoints": checkpoint_inventory(run_dir),
        "missing_inputs": missing,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(export_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    user_notes = ""
    if args.notes_file:
        user_notes = args.notes_file.read_text(encoding="utf-8").strip()
    experiment = submitted_config.get("experiment", {}) if isinstance(submitted_config, dict) else {}
    note_lines = [
        f"# {experiment.get('name', run_dir.name)}",
        "",
        f"- Owner: {experiment.get('owner', 'not recorded')}",
        f"- Purpose: {experiment.get('purpose', 'not recorded')}",
        f"- Final kimg: {export_metadata['last_kimg']}",
        f"- Metric rows: {len(metrics)}",
        f"- Missing inputs: {', '.join(missing) if missing else 'none'}",
        "",
        "## Observations",
        "",
        user_notes or "Add the result interpretation, anomalies, and follow-up decision here.",
        "",
    ]
    (output_dir / "notes.md").write_text("\n".join(note_lines), encoding="utf-8")

    required = [
        "config.yaml",
        "metadata.json",
        "metrics.csv",
        "train_summary.csv",
        "samples_64.png",
        "schedule_curve.png",
        "notes.md",
    ]
    absent = [name for name in required if not (output_dir / name).is_file()]
    if absent:
        raise SystemExit("[export] ERROR: failed to create: " + ", ".join(absent))
    print(f"[export] wrote {len(required)} files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
