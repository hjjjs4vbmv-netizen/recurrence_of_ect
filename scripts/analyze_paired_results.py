#!/usr/bin/env python3
"""Aggregate the frozen Fixed sigmoid vs Adaptive v1 experiment matrix.

Role A supplies one evaluation row per method / training seed / budget / NFE;
Role B supplies checkpoint identities and training/controller telemetry.  This
script joins those records, computes *paired* (same training seed) quality
deltas, produces the Role-C figures, and writes a guarded conclusion draft.

The script intentionally does not pool generated images across training seeds
or infer a conclusion from an incomplete matrix.  Intermediate data are useful
and can be rendered with the default settings, but the conclusion remains
``INCOMPLETE`` until the expected 3 x 3 x 2 paired matrix is present.

Expected metrics CSV (Role A; column aliases are accepted)::

    method,training_seed,budget_kimg,nfe,kid_5k,fid_5k,checkpoint_sha256
    sigmoid,0,16,1,0.0124,3.81,<64-character checkpoint SHA256>
    adaptive_v1,0,16,1,0.0119,3.72,<64-character checkpoint SHA256>

Optional columns ``mid_t``, ``sampling_seed``, ``num_generated`` and
``metric_source`` are carried into the normalized output.  When supplied,
sampling seed, image count, and NFE=2 mid_t must agree inside each pair.

Training records CSV (Role B; optional while a run is still in progress)::

    method,training_seed,budget_kimg,checkpoint_sha256,training_summary_csv,run_dir

Alternatively, pass one or more ``--training-root`` directories containing
Role-B ``metadata.json`` files beside ``train_summary.csv``.  A training record
is joined by method, training seed and checkpoint budget, and its checkpoint
SHA must exactly match Role A's metrics row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

# Keep matplotlib from attempting to write under an immutable home directory on
# compute nodes and CI workers.  The caller can override this location.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ect-xdg-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("sigmoid", "adaptive_v1")
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_BUDGETS = (16.0, 32.0, 64.0)
EXPECTED_NFES = (1, 2)
EXPECTED_IMAGE_COUNT = 5000
METRICS = ("kid_5k", "fid_5k")


class AnalysisError(ValueError):
    """Raised when supplied Role-A/B evidence violates the frozen protocol."""


@dataclass(frozen=True)
class MetricRow:
    method: str
    training_seed: int
    budget_kimg: float
    nfe: int
    kid_5k: float | None
    fid_5k: float | None
    checkpoint_sha256: str
    mid_t: float | None
    sampling_seed: str
    num_generated: int | None
    metric_source: str

    @property
    def key(self) -> tuple[str, int, float, int]:
        return (self.method, self.training_seed, self.budget_kimg, self.nfe)


@dataclass(frozen=True)
class TrainingRecord:
    method: str
    training_seed: int
    budget_kimg: float
    checkpoint_sha256: str
    training_summary_csv: Path | None
    run_dir: str

    @property
    def key(self) -> tuple[str, int, float]:
        return (self.method, self.training_seed, self.budget_kimg)


def fail(message: str) -> None:
    raise AnalysisError(message)


def canonical_method(value: str) -> str:
    normalized = re.sub(r"[\s_-]+", "", str(value).strip().lower())
    aliases = {
        "sigmoid": "sigmoid",
        "fixedsigmoid": "sigmoid",
        "fixed": "sigmoid",
        "adaptivev1": "adaptive_v1",
    }
    if normalized not in aliases:
        fail(f"unknown method {value!r}; expected Fixed sigmoid/sigmoid or Adaptive v1/adaptive_v1")
    return aliases[normalized]


def parse_float(value: Any, field: str, *, allow_blank: bool = False, minimum: float | None = None) -> float | None:
    text = "" if value is None else str(value).strip()
    if text == "":
        if allow_blank:
            return None
        fail(f"{field} must not be blank")
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        fail(f"{field} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        fail(f"{field} must be >= {minimum}, got {number}")
    return number


def parse_int(value: Any, field: str, *, allow_blank: bool = False, minimum: int | None = None) -> int | None:
    number = parse_float(value, field, allow_blank=allow_blank)
    if number is None:
        return None
    if not number.is_integer():
        fail(f"{field} must be an integer, got {value!r}")
    result = int(number)
    if minimum is not None and result < minimum:
        fail(f"{field} must be >= {minimum}, got {result}")
    return result


def parse_budget(value: Any, field: str) -> float:
    text = str(value).strip().lower().replace("kimg", "").strip()
    result = parse_float(text, field, minimum=0)
    assert result is not None
    return float(result)


def canonical_budget(value: float, expected_budgets: Iterable[float]) -> float:
    for expected in expected_budgets:
        if math.isclose(value, expected, rel_tol=0, abs_tol=1e-6):
            return float(expected)
    return round(float(value), 6)


def parse_sha256(value: Any, field: str) -> str:
    digest = "" if value is None else str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        fail(f"{field} must be a 64-character lowercase/uppercase SHA256, got {value!r}")
    return digest


def parse_bool(value: Any, field: str) -> bool | None:
    text = "" if value is None else str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    fail(f"{field} must be true/false (or 1/0), got {value!r}")


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            fail(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        fail(f"CSV has no rows: {path}")
    return rows


def value_from_aliases(row: dict[str, str], aliases: Iterable[str], field: str, row_number: int, *, required: bool = True) -> str:
    normalized = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    if required:
        fail(f"row {row_number}: missing required column {field}; accepted names: {', '.join(aliases)}")
    return ""


def read_metrics(path: Path, expected_budgets: Iterable[float]) -> list[MetricRow]:
    metrics: list[MetricRow] = []
    seen: set[tuple[str, int, float, int]] = set()
    allowed_budgets = {float(budget) for budget in expected_budgets}
    for row_number, row in enumerate(csv_rows(path), start=2):
        method = canonical_method(value_from_aliases(row, ("method", "schedule"), "method", row_number))
        training_seed = parse_int(
            value_from_aliases(row, ("training_seed", "train_seed", "seed"), "training_seed", row_number),
            f"row {row_number} training_seed",
            minimum=0,
        )
        budget = canonical_budget(
            parse_budget(value_from_aliases(row, ("budget_kimg", "budget"), "budget_kimg", row_number), f"row {row_number} budget_kimg"),
            expected_budgets,
        )
        if budget not in allowed_budgets:
            fail(f"row {row_number}: budget_kimg must be one of {sorted(allowed_budgets)}, got {budget:g}")
        if training_seed not in EXPECTED_SEEDS:
            fail(f"row {row_number}: training_seed must be one of {EXPECTED_SEEDS}, got {training_seed}")
        nfe = parse_int(value_from_aliases(row, ("nfe",), "nfe", row_number), f"row {row_number} nfe", minimum=1)
        if nfe not in EXPECTED_NFES:
            fail(f"row {row_number}: nfe must be one of {EXPECTED_NFES}, got {nfe}")
        kid = parse_float(
            value_from_aliases(row, ("kid_5k", "kid5k", "kid"), "kid_5k", row_number, required=False),
            f"row {row_number} kid_5k",
            allow_blank=True,
            minimum=0,
        )
        fid = parse_float(
            value_from_aliases(row, ("fid_5k", "fid5k", "fid"), "fid_5k", row_number, required=False),
            f"row {row_number} fid_5k",
            allow_blank=True,
            minimum=0,
        )
        if kid is None and fid is None:
            fail(f"row {row_number}: at least one of kid_5k or fid_5k is required")
        checkpoint_sha256 = parse_sha256(
            value_from_aliases(
                row,
                ("checkpoint_sha256", "checkpoint_sha", "checkpoint_sha256sum"),
                "checkpoint_sha256",
                row_number,
            ),
            f"row {row_number} checkpoint_sha256",
        )
        mid_t = parse_float(
            value_from_aliases(row, ("mid_t",), "mid_t", row_number, required=False),
            f"row {row_number} mid_t",
            allow_blank=True,
            minimum=0,
        )
        if nfe == 2 and mid_t is None:
            mid_t = 0.821
        if nfe == 2 and not math.isclose(mid_t, 0.821, rel_tol=0, abs_tol=1e-9):
            fail(f"row {row_number}: NFE=2 requires the frozen mid_t=0.821, got {mid_t}")
        if nfe == 1 and mid_t is not None:
            fail(f"row {row_number}: NFE=1 must not specify mid_t, got {mid_t}")
        sampling_seed = value_from_aliases(
            row, ("sampling_seed", "eval_seed", "evaluation_seed"), "sampling_seed", row_number, required=False
        ).strip()
        num_generated = parse_int(
            value_from_aliases(row, ("num_generated", "image_count", "num_images"), "num_generated", row_number, required=False),
            f"row {row_number} num_generated",
            allow_blank=True,
            minimum=1,
        )
        metric_source = value_from_aliases(
            row, ("metric_source", "source", "metrics_path"), "metric_source", row_number, required=False
        ).strip()
        assert training_seed is not None and nfe is not None
        result = MetricRow(
            method=method,
            training_seed=training_seed,
            budget_kimg=budget,
            nfe=nfe,
            kid_5k=kid,
            fid_5k=fid,
            checkpoint_sha256=checkpoint_sha256,
            mid_t=mid_t,
            sampling_seed=sampling_seed,
            num_generated=num_generated,
            metric_source=metric_source,
        )
        if result.key in seen:
            fail(f"duplicate metrics row for method/seed/budget/NFE={result.key}")
        seen.add(result.key)
        metrics.append(result)
    return sorted(metrics, key=lambda item: (item.budget_kimg, item.nfe, item.training_seed, item.method))


def training_record_from_row(row: dict[str, str], row_number: int, expected_budgets: Iterable[float], source: Path) -> TrainingRecord:
    method = canonical_method(value_from_aliases(row, ("method", "schedule"), "method", row_number))
    seed = parse_int(
        value_from_aliases(row, ("training_seed", "train_seed", "seed"), "training_seed", row_number),
        f"row {row_number} training_seed",
        minimum=0,
    )
    budget = canonical_budget(
        parse_budget(value_from_aliases(row, ("budget_kimg", "budget"), "budget_kimg", row_number), f"row {row_number} budget_kimg"),
        expected_budgets,
    )
    sha = parse_sha256(
        value_from_aliases(row, ("checkpoint_sha256", "checkpoint_sha", "network_snapshot_sha256"), "checkpoint_sha256", row_number),
        f"row {row_number} checkpoint_sha256",
    )
    summary_text = value_from_aliases(
        row, ("training_summary_csv", "train_summary_csv", "training_summary"), "training_summary_csv", row_number, required=False
    ).strip()
    summary_path = None
    if summary_text:
        candidate = Path(summary_text)
        summary_path = candidate if candidate.is_absolute() else source.parent / candidate
    run_dir = value_from_aliases(row, ("run_dir", "training_run_dir"), "run_dir", row_number, required=False).strip()
    assert seed is not None
    return TrainingRecord(method, seed, budget, sha, summary_path, run_dir)


def read_training_records(path: Path, expected_budgets: Iterable[float]) -> list[TrainingRecord]:
    return [
        training_record_from_row(row, row_number, expected_budgets, path)
        for row_number, row in enumerate(csv_rows(path), start=2)
    ]


def records_from_training_root(root: Path, expected_budgets: Iterable[float]) -> list[TrainingRecord]:
    if not root.is_dir():
        fail(f"training root is not a directory: {root}")
    records: list[TrainingRecord] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"cannot parse training metadata {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            continue
        if not {"schedule", "seed", "processed_kimg", "network_snapshot_sha256"}.issubset(metadata):
            continue
        try:
            method = canonical_method(metadata["schedule"])
            seed = parse_int(metadata["seed"], f"{metadata_path} seed", minimum=0)
            raw_budget = parse_budget(metadata["processed_kimg"], f"{metadata_path} processed_kimg")
            # A shared run root can retain engineering smokes and old activation
            # probes. They are not checkpoints in the frozen 16/32/64-kimg
            # matrix and must not create duplicate records for this analysis.
            if not any(math.isclose(raw_budget, budget, rel_tol=0, abs_tol=1e-6) for budget in expected_budgets):
                continue
            budget = canonical_budget(raw_budget, expected_budgets)
            sha = parse_sha256(metadata["network_snapshot_sha256"], f"{metadata_path} network_snapshot_sha256")
        except AnalysisError:
            raise
        summary = metadata_path.parent / "train_summary.csv"
        records.append(
            TrainingRecord(
                method=method,
                training_seed=int(seed),
                budget_kimg=budget,
                checkpoint_sha256=sha,
                training_summary_csv=summary if summary.is_file() else None,
                run_dir=str(metadata_path.parent),
            )
        )
    return records


def index_training_records(records: Iterable[TrainingRecord]) -> dict[tuple[str, int, float], TrainingRecord]:
    indexed: dict[tuple[str, int, float], TrainingRecord] = {}
    for record in records:
        if record.key in indexed:
            previous = indexed[record.key]
            fail(
                "duplicate training record for "
                f"{record.key}: {previous.checkpoint_sha256} and {record.checkpoint_sha256}; "
                "supply one immutable checkpoint record per method/seed/budget"
            )
        indexed[record.key] = record
    return indexed


def optional_number(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def optional_bool(value: Any) -> bool | None:
    text = "" if value is None else str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def summarize_training(
    summary_path: Path | None,
    budget_kimg: float,
    adaptive_max_adjust: float,
    expected_method: str,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "training_summary_available": False,
        "training_processed_kimg": None,
        "training_attempted_iterations": None,
        "training_successful_optimizer_steps": None,
        "training_successful_steps_valid": None,
        "training_amp_skipped": None,
        "training_amp_skip_rate": None,
        "training_amp_skip_telemetry_complete": None,
        "training_nan_count": None,
        "training_inf_count": None,
        "training_trailing_loss_mean": None,
        "training_trailing_loss_std": None,
        "training_final_loss": None,
        "adaptive_active": None,
        "final_correction": None,
        "correction_abs_max": None,
        "correction_saturation_fraction": None,
        "correction_sign_changes": None,
        "correction_bound_valid": None,
        "final_r_over_t_mean": None,
        "final_gap_mean": None,
        "pair_ratio_valid": None,
        "pair_gap_valid": None,
        "controller_telemetry_complete": None,
    }
    if summary_path is None:
        return fields
    if not summary_path.is_file():
        fail(f"training summary referenced by a record does not exist: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        fail(f"training summary has no rows: {summary_path}")
    selected: list[tuple[float, dict[str, str]]] = []
    prior_processed: float | None = None
    for row_number, row in enumerate(rows, start=2):
        processed = optional_number(row.get("processed_kimg"))
        if processed is None:
            nimg = optional_number(row.get("processed_nimg"))
            processed = None if nimg is None else nimg / 1000.0
        if processed is None:
            fail(f"training summary {summary_path} row {row_number} has no numeric processed_kimg or processed_nimg")
        if prior_processed is not None and processed < prior_processed - 1e-6:
            fail(f"training summary {summary_path} has non-monotonic processed progress at row {row_number}")
        prior_processed = processed
        if processed is not None and processed <= budget_kimg + 1e-6:
            selected.append((processed, row))
    if not selected:
        fail(f"training summary {summary_path} has no row at or before {budget_kimg:g} kimg")
    final_processed, _ = selected[-1]
    if final_processed < budget_kimg - 1e-6:
        fail(
            f"training summary {summary_path} reaches only {final_processed:g} kimg before the "
            f"{budget_kimg:g}-kimg checkpoint"
        )
    selected_rows = [row for _, row in selected]
    observed_schedules = set()
    for row_number, row in enumerate(selected_rows, start=2):
        schedule = str(row.get("schedule", "")).strip()
        if not schedule:
            fail(f"training summary {summary_path} row {row_number} is missing schedule")
        observed_schedules.add(canonical_method(schedule))
    if observed_schedules != {expected_method}:
        fail(
            f"training summary {summary_path} schedule {sorted(observed_schedules)} does not match "
            f"training record method {expected_method!r}"
        )

    losses = [optional_number(row.get("loss")) for row in selected_rows]
    finite_losses = [value for value in losses if value is not None]
    fields["training_summary_available"] = True
    fields["training_processed_kimg"] = final_processed
    fields["training_attempted_iterations"] = len(selected_rows)
    successful_values = [optional_number(row.get("successful_optimizer_steps")) for row in selected_rows]
    if all(value is not None and value >= 0 and value.is_integer() for value in successful_values):
        successful_steps = [int(value) for value in successful_values]
        if any(right < left for left, right in zip(successful_steps, successful_steps[1:])):
            fail(f"training summary {summary_path} has non-monotonic successful_optimizer_steps")
        fields["training_successful_optimizer_steps"] = successful_steps[-1]
        fields["training_successful_steps_valid"] = True
    else:
        fields["training_successful_steps_valid"] = False
    skipped = [optional_bool(row.get("step_skipped")) for row in selected_rows]
    skipped_count = sum(value is True for value in skipped)
    fields["training_amp_skipped"] = skipped_count if any(value is not None for value in skipped) else None
    fields["training_amp_skip_rate"] = skipped_count / len(selected_rows) if any(value is not None for value in skipped) else None
    fields["training_amp_skip_telemetry_complete"] = all(value is not None for value in skipped)
    fields["training_nan_count"] = sum(
        1 for row in selected_rows if str(row.get("loss", "")).strip().lower() == "nan"
    )
    fields["training_inf_count"] = sum(
        1
        for row in selected_rows
        if str(row.get("loss", "")).strip().lower()
        in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    )
    # A malformed non-numeric loss must never be silently treated as finite.
    malformed_loss_count = sum(
        1
        for row, value in zip(selected_rows, losses)
        if value is None and str(row.get("loss", "")).strip().lower() not in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    )
    fields["training_nan_count"] += malformed_loss_count
    if finite_losses:
        trailing_count = min(100, max(1, math.ceil(len(finite_losses) * 0.1)))
        trailing = finite_losses[-trailing_count:]
        fields["training_trailing_loss_mean"] = mean(trailing)
        fields["training_trailing_loss_std"] = stdev(trailing) if len(trailing) >= 2 else 0.0
        fields["training_final_loss"] = finite_losses[-1]

    raw_corrections = [str(row.get("correction", "")).strip() for row in selected_rows]
    corrections = [optional_number(value) for value in raw_corrections]
    if any(text and value is None for text, value in zip(raw_corrections, corrections)):
        fail(f"training summary {summary_path} contains non-finite correction telemetry")
    fields["correction_bound_valid"] = all(
        value is None or abs(value) <= adaptive_max_adjust + 1e-9 for value in corrections
    )
    finite_corrections = [value for value in corrections if value is not None]
    if finite_corrections:
        fields["final_correction"] = finite_corrections[-1]
        fields["correction_abs_max"] = max(abs(value) for value in finite_corrections)
        threshold = adaptive_max_adjust * 0.95
        fields["correction_saturation_fraction"] = sum(abs(value) >= threshold for value in finite_corrections) / len(finite_corrections) if threshold > 0 else 0.0
        signed = [1 if value > 0 else -1 for value in finite_corrections if value != 0]
        fields["correction_sign_changes"] = sum(left != right for left, right in zip(signed, signed[1:]))
    raw_active = [str(row.get("adaptive_active", "")).strip() for row in selected_rows]
    active = [optional_bool(value) for value in raw_active]
    if any(text and value is None for text, value in zip(raw_active, active)):
        fail(f"training summary {summary_path} contains invalid adaptive_active telemetry")
    if active[-1] is not None:
        fields["adaptive_active"] = active[-1]
    raw_ratios = [str(row.get("r_over_t_mean", "")).strip() for row in selected_rows]
    raw_gaps = [str(row.get("gap_mean", "")).strip() for row in selected_rows]
    ratios = [optional_number(value) for value in raw_ratios]
    gaps = [optional_number(value) for value in raw_gaps]
    if any(text and value is None for text, value in zip(raw_ratios, ratios)):
        fail(f"training summary {summary_path} contains non-finite r_over_t_mean telemetry")
    if any(text and value is None for text, value in zip(raw_gaps, gaps)):
        fail(f"training summary {summary_path} contains non-finite gap_mean telemetry")
    fields["controller_telemetry_complete"] = (
        all(value is not None for value in corrections)
        and all(value is not None for value in active)
        and all(value is not None for value in ratios)
        and all(value is not None for value in gaps)
    )
    paired_values = [(ratio, gap) for ratio, gap in zip(ratios, gaps) if ratio is not None and gap is not None]
    if paired_values:
        fields["final_r_over_t_mean"], fields["final_gap_mean"] = paired_values[-1]
        fields["pair_ratio_valid"] = all(0 <= ratio <= 1 for ratio, _ in paired_values)
        fields["pair_gap_valid"] = all(0 <= gap <= 1 and math.isclose(ratio + gap, 1.0, rel_tol=0, abs_tol=1e-5) for ratio, gap in paired_values)
    return fields


def join_metrics_and_training(
    metrics: Iterable[MetricRow],
    records: dict[tuple[str, int, float], TrainingRecord],
    adaptive_max_adjust: float,
) -> list[dict[str, Any]]:
    cached_summaries: dict[tuple[Path | None, float, str], dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for metric in metrics:
        record = records.get((metric.method, metric.training_seed, metric.budget_kimg))
        row: dict[str, Any] = {
            "method": metric.method,
            "training_seed": metric.training_seed,
            "budget_kimg": metric.budget_kimg,
            "nfe": metric.nfe,
            "mid_t": metric.mid_t,
            "sampling_seed": metric.sampling_seed,
            "num_generated": metric.num_generated,
            "kid_5k": metric.kid_5k,
            "fid_5k": metric.fid_5k,
            "checkpoint_sha256": metric.checkpoint_sha256,
            "metric_source": metric.metric_source,
            "run_dir": "",
            "training_summary_csv": "",
        }
        if record is None:
            row.update(summarize_training(None, metric.budget_kimg, adaptive_max_adjust, metric.method))
        else:
            if record.checkpoint_sha256 != metric.checkpoint_sha256:
                fail(
                    "checkpoint SHA mismatch for "
                    f"{metric.method}, seed={metric.training_seed}, budget={metric.budget_kimg:g} kimg: "
                    f"Role A={metric.checkpoint_sha256}, Role B={record.checkpoint_sha256}"
                )
            cache_key = (record.training_summary_csv, metric.budget_kimg, metric.method)
            if cache_key not in cached_summaries:
                cached_summaries[cache_key] = summarize_training(
                    record.training_summary_csv, metric.budget_kimg, adaptive_max_adjust, metric.method
                )
            row.update(cached_summaries[cache_key])
            row["run_dir"] = record.run_dir
            row["training_summary_csv"] = "" if record.training_summary_csv is None else str(record.training_summary_csv)
        output.append(row)
    return output


def verify_pair_settings(fixed: dict[str, Any], adaptive: dict[str, Any]) -> bool:
    if fixed["checkpoint_sha256"] == adaptive["checkpoint_sha256"]:
        fail(
            f"fixed and adaptive use the same checkpoint SHA for seed={fixed['training_seed']}, "
            f"budget={fixed['budget_kimg']}, NFE={fixed['nfe']}"
        )
    if fixed["nfe"] == 2 and not math.isclose(float(fixed["mid_t"]), float(adaptive["mid_t"]), rel_tol=0, abs_tol=1e-9):
        fail(
            f"mid_t mismatch for seed={fixed['training_seed']}, budget={fixed['budget_kimg']}, NFE=2: "
            f"fixed={fixed['mid_t']}, adaptive={adaptive['mid_t']}"
        )
    for field in ("sampling_seed", "num_generated"):
        left, right = fixed[field], adaptive[field]
        if left not in {None, ""} and right not in {None, ""} and left != right:
            fail(
                f"{field} mismatch for seed={fixed['training_seed']}, budget={fixed['budget_kimg']}, NFE={fixed['nfe']}: "
                f"fixed={left!r}, adaptive={right!r}"
            )
    return (
        all(fixed[field] not in {None, ""} and adaptive[field] not in {None, ""} for field in ("sampling_seed", "num_generated"))
        and int(fixed["num_generated"]) == EXPECTED_IMAGE_COUNT
    )


def paired_differences(per_seed: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_seed:
        key = (int(row["training_seed"]), float(row["budget_kimg"]), int(row["nfe"]))
        grouped[key][str(row["method"])] = row
    output: list[dict[str, Any]] = []
    for (seed, budget, nfe), arms in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        if set(arms) != set(METHODS):
            continue  # Rolling reports are intentionally allowed to be partial.
        fixed, adaptive = arms["sigmoid"], arms["adaptive_v1"]
        settings_verified = verify_pair_settings(fixed, adaptive)
        for metric in METRICS:
            fixed_value, adaptive_value = fixed[metric], adaptive[metric]
            if (fixed_value is None) != (adaptive_value is None):
                fail(
                    f"{metric} is present for only one arm at seed={seed}, budget={budget:g}, NFE={nfe}; "
                    "fixed and adaptive must use the same metric"
                )
            if fixed_value is None:
                continue
            delta = float(adaptive_value) - float(fixed_value)
            output.append(
                {
                    "metric": metric,
                    "training_seed": seed,
                    "budget_kimg": budget,
                    "nfe": nfe,
                    "fixed_value": fixed_value,
                    "adaptive_value": adaptive_value,
                    "delta_adaptive_minus_fixed": delta,
                    "adaptive_better": delta < 0,
                    "paired_sampling_settings_verified": settings_verified,
                    "sampling_seed": fixed["sampling_seed"],
                    "num_generated": fixed["num_generated"],
                    "fixed_checkpoint_sha256": fixed["checkpoint_sha256"],
                    "adaptive_checkpoint_sha256": adaptive["checkpoint_sha256"],
                }
            )
    return output


def aggregate_differences(pairs: Iterable[dict[str, Any]], expected_seeds: Iterable[int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(str(row["metric"]), float(row["budget_kimg"]), int(row["nfe"]))].append(row)
    output: list[dict[str, Any]] = []
    expected_seed_set = set(expected_seeds)
    for (metric, budget, nfe), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        rows = sorted(rows, key=lambda item: int(item["training_seed"]))
        deltas = [float(row["delta_adaptive_minus_fixed"]) for row in rows]
        fixed_values = [float(row["fixed_value"]) for row in rows]
        adaptive_values = [float(row["adaptive_value"]) for row in rows]
        seeds = {int(row["training_seed"]) for row in rows}
        count = len(rows)
        output.append(
            {
                "metric": metric,
                "budget_kimg": budget,
                "nfe": nfe,
                "paired_seed_count": count,
                "expected_seed_count": len(expected_seed_set),
                "coverage_complete": seeds == expected_seed_set,
                "adaptive_better_seed_count": sum(delta < 0 for delta in deltas),
                "adaptive_worse_seed_count": sum(delta > 0 for delta in deltas),
                "fixed_mean": mean(fixed_values),
                "fixed_std": stdev(fixed_values) if count >= 2 else 0.0,
                "adaptive_mean": mean(adaptive_values),
                "adaptive_std": stdev(adaptive_values) if count >= 2 else 0.0,
                "delta_mean": mean(deltas),
                "delta_std": stdev(deltas) if count >= 2 else 0.0,
                "delta_sem": (stdev(deltas) / math.sqrt(count)) if count >= 2 else 0.0,
                "paired_sampling_settings_verified": all(row["paired_sampling_settings_verified"] for row in rows),
            }
        )
    return output


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    """Return descriptive Pearson r, or None when the relationship is undefined."""
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return None
    x_mean, y_mean = mean(x_values), mean(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    denominator = math.sqrt(sum(value * value for value in x_centered) * sum(value * value for value in y_centered))
    if denominator == 0:
        return None
    return sum(left * right for left, right in zip(x_centered, y_centered)) / denominator


def mechanism_relationships(per_seed: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe training/controller-to-quality relationships without p-values.

    Each NFE is analysed separately. This prevents the intentionally different
    one-step and two-step samplers from masquerading as a controller effect.
    """
    adaptive_rows = {
        (int(row["training_seed"]), float(row["budget_kimg"]), int(row["nfe"])): row
        for row in per_seed
        if row["method"] == "adaptive_v1"
    }
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        key = (int(pair["training_seed"]), float(pair["budget_kimg"]), int(pair["nfe"]))
        adaptive = adaptive_rows.get(key)
        if adaptive is not None:
            grouped[(str(pair["metric"]), int(pair["nfe"]))].append((pair, adaptive))
    relationships: list[dict[str, Any]] = []
    specifications = (
        ("adaptive_quality_vs_trailing_loss", "training_trailing_loss_mean", "adaptive_value"),
        ("paired_delta_vs_correction", "final_correction", "delta_adaptive_minus_fixed"),
        ("paired_delta_vs_gap", "final_gap_mean", "delta_adaptive_minus_fixed"),
    )
    for (metric, nfe), rows in sorted(grouped.items()):
        for relation, x_field, y_field in specifications:
            x_values: list[float] = []
            y_values: list[float] = []
            for pair, adaptive in rows:
                x_value = adaptive.get(x_field)
                y_value = pair.get(y_field)
                if x_value is not None and y_value is not None:
                    x_values.append(float(x_value))
                    y_values.append(float(y_value))
            relationships.append(
                {
                    "metric": metric,
                    "nfe": nfe,
                    "relationship": relation,
                    "n": len(x_values),
                    "pearson_r": pearson_correlation(x_values, y_values),
                }
            )
    return relationships


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_number(row.get(field)) for field in fieldnames})


def plot_quality(per_seed: list[dict[str, Any]], output_path: Path) -> None:
    available = [metric for metric in METRICS if any(row[metric] is not None for row in per_seed)]
    if not available:
        fail("cannot plot quality without KID or FID values")
    figure, axes = plt.subplots(len(available), len(EXPECTED_NFES), figsize=(11, 4.2 * len(available)), squeeze=False)
    colours = {"sigmoid": "#356aa0", "adaptive_v1": "#d95f02"}
    labels = {"sigmoid": "Fixed sigmoid", "adaptive_v1": "Adaptive v1"}
    for metric_index, metric in enumerate(available):
        for nfe_index, nfe in enumerate(EXPECTED_NFES):
            axis = axes[metric_index][nfe_index]
            for method in METHODS:
                budgets, averages, spreads = [], [], []
                for budget in EXPECTED_BUDGETS:
                    values = [
                        float(row[metric])
                        for row in per_seed
                        if row["method"] == method and row["nfe"] == nfe and row["budget_kimg"] == budget and row[metric] is not None
                    ]
                    if values:
                        budgets.append(budget)
                        averages.append(mean(values))
                        spreads.append(stdev(values) if len(values) >= 2 else 0.0)
                if budgets:
                    axis.errorbar(budgets, averages, yerr=spreads, marker="o", capsize=4, linewidth=2, color=colours[method], label=labels[method])
            axis.set_title(f"{metric.replace('_', '-').upper()} · NFE={nfe}" + (" · mid_t=0.821" if nfe == 2 else ""))
            axis.set_xlabel("Training budget (kimg)")
            axis.set_ylabel(f"{metric.replace('_', '-').upper()} (lower is better)")
            axis.set_xticks(EXPECTED_BUDGETS)
            axis.grid(alpha=0.25)
            axis.legend()
    figure.suptitle("Paired generation quality by training budget (mean ± sample std across training seeds)", y=1.01)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def values_by_budget(rows: Iterable[dict[str, Any]], method: str, field: str) -> tuple[list[float], list[float], list[float]]:
    budgets, averages, spreads = [], [], []
    for budget in EXPECTED_BUDGETS:
        values = [
            float(row[field])
            for row in rows
            if row["method"] == method and row["budget_kimg"] == budget and row[field] is not None
        ]
        if values:
            budgets.append(budget)
            averages.append(mean(values))
            spreads.append(stdev(values) if len(values) >= 2 else 0.0)
    return budgets, averages, spreads


def plot_controller(per_seed: list[dict[str, Any]], output_path: Path) -> None:
    unique_training_rows: dict[tuple[str, int, float], dict[str, Any]] = {}
    for row in per_seed:
        unique_training_rows[(row["method"], row["training_seed"], row["budget_kimg"])] = row
    training_rows = list(unique_training_rows.values())
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colours = {"sigmoid": "#356aa0", "adaptive_v1": "#d95f02"}
    labels = {"sigmoid": "Fixed sigmoid", "adaptive_v1": "Adaptive v1"}
    for field, axis, title, ylabel, methods in (
        ("final_correction", axes[0], "Controller correction", "Final correction", ("adaptive_v1",)),
        ("final_gap_mean", axes[1], "Final training-pair gap", "mean((t-r)/t)", METHODS),
        ("training_trailing_loss_mean", axes[2], "Trailing training loss", "mean loss (last 10%, max 100 rows)", METHODS),
    ):
        found = False
        for method in methods:
            budgets, averages, spreads = values_by_budget(training_rows, method, field)
            if budgets:
                found = True
                axis.errorbar(budgets, averages, yerr=spreads, marker="o", capsize=4, linewidth=2, color=colours[method], label=labels[method])
        axis.set_title(title)
        axis.set_xlabel("Checkpoint budget (kimg)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(EXPECTED_BUDGETS)
        axis.grid(alpha=0.25)
        if found:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No training telemetry supplied", ha="center", va="center", transform=axis.transAxes)
    figure.suptitle("Controller and training-stability telemetry (mean ± sample std across training seeds)", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def expected_quality_settings() -> set[tuple[float, int]]:
    return {(budget, nfe) for budget in EXPECTED_BUDGETS for nfe in EXPECTED_NFES}


def primary_metric_availability(aggregate: list[dict[str, Any]], expected_seeds: Iterable[int]) -> tuple[str | None, str]:
    """Return the usable metric and why no final-quality verdict is possible.

    Complete paired values and complete evaluation provenance are distinct
    requirements.  A CSV imported from a legacy Role-A report can have all
    36 numerical values while omitting sampling_seed/num_generated; calling
    that a missing metric matrix would hide the actual data-freeze issue.
    """
    expected = expected_quality_settings()
    expected_count = len(set(expected_seeds))
    complete_but_unverified = False
    for metric in METRICS:  # KID is deliberately first: it is the frozen primary metric.
        covered = {
            (float(row["budget_kimg"]), int(row["nfe"]))
            for row in aggregate
            if (
                row["metric"] == metric
                and row["paired_seed_count"] == expected_count
                and row["coverage_complete"]
            )
        }
        verified = {
            (float(row["budget_kimg"]), int(row["nfe"]))
            for row in aggregate
            if (
                row["metric"] == metric
                and row["paired_seed_count"] == expected_count
                and row["coverage_complete"]
                and row["paired_sampling_settings_verified"]
            )
        }
        if verified == expected:
            return metric, "COMPLETE"
        if covered == expected:
            complete_but_unverified = True
    if complete_but_unverified:
        return None, "SAMPLING_PROVENANCE_INCOMPLETE"
    return None, "METRIC_COVERAGE_INCOMPLETE"


def training_stability_status(
    per_seed: list[dict[str, Any]], expected_seeds: Iterable[int], skip_rate_tolerance: float
) -> tuple[str, list[str]]:
    unique: dict[tuple[str, int, float], dict[str, Any]] = {}
    for row in per_seed:
        unique[(row["method"], int(row["training_seed"]), float(row["budget_kimg"]))] = row
    expected = {(method, seed, budget) for method in METHODS for seed in expected_seeds for budget in EXPECTED_BUDGETS}
    missing = expected - set(unique)
    if missing:
        return "INCOMPLETE", [f"missing {len(missing)} training records"]
    findings: list[str] = []
    for key in sorted(expected):
        row = unique[key]
        if not row["training_summary_available"]:
            return "INCOMPLETE", [f"training summary unavailable for {key}"]
        if row["training_successful_steps_valid"] is not True:
            return "INCOMPLETE", [f"successful optimizer-step telemetry unavailable or invalid for {key}"]
        if row["training_amp_skip_telemetry_complete"] is not True:
            return "INCOMPLETE", [f"AMP skip telemetry unavailable or invalid for {key}"]
        if row["controller_telemetry_complete"] is not True:
            return "INCOMPLETE", [f"controller telemetry unavailable for {key}"]
        if (row["training_nan_count"] or 0) > 0 or (row["training_inf_count"] or 0) > 0:
            findings.append(f"non-finite loss recorded for {key}")
        if row["pair_ratio_valid"] is not True or row["pair_gap_valid"] is not True:
            findings.append(f"invalid r/t or gap telemetry for {key}")
        if row["correction_bound_valid"] is not True:
            findings.append(f"correction exceeds the configured bound for {key}")
        if key[0] == "adaptive_v1" and row["adaptive_active"] is not True:
            findings.append(f"adaptive controller did not activate for {key}")
    for seed in expected_seeds:
        for budget in EXPECTED_BUDGETS:
            fixed = unique[("sigmoid", seed, budget)]
            adaptive = unique[("adaptive_v1", seed, budget)]
            fixed_rate, adaptive_rate = fixed["training_amp_skip_rate"], adaptive["training_amp_skip_rate"]
            if fixed_rate is None or adaptive_rate is None:
                return "INCOMPLETE", [f"AMP skip telemetry unavailable for seed={seed}, budget={budget:g}"]
            if float(adaptive_rate) > float(fixed_rate) + skip_rate_tolerance:
                findings.append(
                    f"adaptive AMP skip rate exceeds fixed by more than {skip_rate_tolerance:.1%} at seed={seed}, budget={budget:g}"
                )
    return ("PASS", []) if not findings else ("FAIL", findings)


def conclusion_from_aggregate(
    aggregate: list[dict[str, Any]], per_seed: list[dict[str, Any]], expected_seeds: Iterable[int], skip_rate_tolerance: float
) -> dict[str, Any]:
    primary, quality_availability = primary_metric_availability(aggregate, expected_seeds)
    stability, stability_findings = training_stability_status(per_seed, expected_seeds, skip_rate_tolerance)
    if primary is None:
        if quality_availability == "SAMPLING_PROVENANCE_INCOMPLETE":
            reason = (
                "The complete paired KID/FID matrix is available, but its evaluation provenance is incomplete: "
                "each fixed/adaptive pair must record the same nonempty sampling_seed and num_generated=5000."
            )
        else:
            reason = "The complete 3-seed × 16/32/64 kimg × NFE=1/2 paired matrix is not yet available for one common metric."
        return {
            "label": "INCOMPLETE",
            "primary_metric": None,
            "quality_availability": quality_availability,
            "stability": stability,
            "stability_findings": stability_findings,
            "reason": reason,
            "setting_signals": [],
        }
    expected_count = len(set(expected_seeds))
    primary_rows = [row for row in aggregate if row["metric"] == primary]
    signal_by_setting: dict[tuple[float, int], dict[str, Any]] = {}
    for row in primary_rows:
        key = (float(row["budget_kimg"]), int(row["nfe"]))
        complete = row["paired_seed_count"] == expected_count and row["coverage_complete"]
        wins, losses, delta = int(row["adaptive_better_seed_count"]), int(row["adaptive_worse_seed_count"]), float(row["delta_mean"])
        signal_by_setting[key] = {
            "budget_kimg": key[0],
            "nfe": key[1],
            "complete": complete,
            "wins": wins,
            "losses": losses,
            "delta_mean": delta,
            "advantage": complete and wins >= 2 and delta < 0,
            "regression": complete and losses >= 2 and delta > 0,
        }
    settings = [signal_by_setting[key] for key in sorted(signal_by_setting)]
    advantages = [item for item in settings if item["advantage"]]
    regressions = [item for item in settings if item["regression"]]
    high_budget_advantages = []
    for item in advantages:
        if item["budget_kimg"] not in {32.0, 64.0}:
            continue
        peer = signal_by_setting.get((item["budget_kimg"], 1 if item["nfe"] == 2 else 2))
        if peer is not None and not peer["regression"]:
            high_budget_advantages.append(item)
    if stability != "PASS":
        label = "INCOMPLETE"
        reason = "Quality matrix is complete, but the frozen stability/controller safety requirement is not yet satisfied."
    elif advantages and regressions:
        label = "混合"
        reason = "Stable positive and negative paired settings coexist, so the effect is sensitive to budget or NFE."
    elif high_budget_advantages:
        label = "Adaptive 表现出初步优势"
        reason = "At least one 32/64 kimg setting has ≥2/3 adaptive seed wins and a lower three-seed mean, while its other NFE is not a stable regression."
    elif regressions:
        label = "负向"
        reason = "At least one complete setting has ≥2/3 adaptive seed losses and a worse three-seed mean, with no stable positive setting."
    else:
        label = "持平"
        reason = "No complete setting meets the pre-frozen repeated-advantage or repeated-regression rule."
    return {
        "label": label,
        "primary_metric": primary,
        "quality_availability": quality_availability,
        "stability": stability,
        "stability_findings": stability_findings,
        "reason": reason,
        "setting_signals": settings,
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":--"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(result)


def write_conclusion(
    path: Path,
    conclusion: dict[str, Any],
    aggregate: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    expected_seeds: Iterable[int],
    skip_rate_tolerance: float,
) -> None:
    expected_count = len(set(expected_seeds))
    lines = ["# Fixed sigmoid vs Adaptive v1 — Role C conclusion", "", f"## Current verdict: {conclusion['label']}", "", conclusion["reason"], ""]
    primary = conclusion["primary_metric"]
    if primary is None:
        if conclusion["quality_availability"] == "SAMPLING_PROVENANCE_INCOMPLETE":
            lines += [
                "The report is deliberately provisional: all numerical pairs are present, but it does not make a quality claim until every pair records matching sampling_seed values and num_generated=5000.",
                "",
            ]
        else:
            lines += [
                "The report is deliberately provisional: it does not make a quality claim until one common metric covers every expected paired setting.",
                "",
            ]
    else:
        lines += [
            f"Primary metric: **{primary.replace('_', '-').upper()}** (lower is better). KID is preferred whenever its full frozen matrix is available; FID becomes the common fallback only when KID is incomplete.",
            "",
        ]
    lines += [f"Training/controller stability gate: **{conclusion['stability']}**."]
    if conclusion["stability_findings"]:
        lines += ["", "Stability findings:"] + [f"- {finding}" for finding in conclusion["stability_findings"]]
    lines += ["", "## Paired quality summary", ""]
    summary_rows = []
    for row in aggregate:
        summary_rows.append(
            [
                row["metric"].replace("_", "-").upper(),
                f"{row['budget_kimg']:g}",
                str(row["nfe"]),
                f"{row['paired_seed_count']}/{expected_count}",
                f"{row['adaptive_better_seed_count']}/{row['paired_seed_count']}",
                f"{row['fixed_mean']:.6g} ± {row['fixed_std']:.3g}",
                f"{row['adaptive_mean']:.6g} ± {row['adaptive_std']:.3g}",
                f"{row['delta_mean']:.6g} ± {row['delta_std']:.3g}",
            ]
        )
    lines += [
        markdown_table(
            ["Metric", "Budget (kimg)", "NFE", "Paired seeds", "Adaptive wins", "Fixed mean ± std", "Adaptive mean ± std", "Δ adaptive − fixed ± std"],
            summary_rows,
        ),
        "",
        "Negative Δ means Adaptive v1 is better. Standard deviations are sample SD across paired training seeds; no p-value is inferred from n=3.",
        "",
        "## Pre-frozen decision checks",
        "",
    ]
    if conclusion["setting_signals"]:
        signal_rows = []
        for item in conclusion["setting_signals"]:
            signal_rows.append(
                [
                    f"{item['budget_kimg']:g}",
                    str(item["nfe"]),
                    f"{item['wins']}/3",
                    f"{item['losses']}/3",
                    f"{item['delta_mean']:.6g}",
                    "yes" if item["advantage"] else "no",
                    "yes" if item["regression"] else "no",
                ]
            )
        lines += [
            markdown_table(
                ["Budget (kimg)", "NFE", "Adaptive wins", "Adaptive losses", "Mean Δ", "Repeated advantage", "Repeated regression"],
                signal_rows,
            ),
            "",
        ]
    else:
        if conclusion["quality_availability"] == "SAMPLING_PROVENANCE_INCOMPLETE":
            lines += ["Metric coverage is complete, but sampling provenance is not verified for every setting.", ""]
        else:
            lines += ["No common complete metric matrix is available yet.", ""]
    lines += ["## Training and controller relationships", ""]
    if relationships:
        relationship_rows = []
        for item in relationships:
            relationship_rows.append(
                [
                    item["metric"].replace("_", "-").upper(),
                    str(item["nfe"]),
                    item["relationship"],
                    str(item["n"]),
                    "undefined" if item["pearson_r"] is None else f"{item['pearson_r']:.4f}",
                ]
            )
        lines += [
            markdown_table(["Metric", "NFE", "Relationship", "n", "Pearson r"], relationship_rows),
            "",
            "These are descriptive correlations across available adaptive runs, separated by NFE. They are not significance tests and do not establish causality.",
            "",
        ]
    else:
        lines += ["No paired quality/controller observations are available yet.", ""]
    lines += [
        "## Guardrails applied",
        "",
        "- Only fixed/adaptive rows with the same training seed, checkpoint budget and NFE are differenced.",
        "- KID/FID are never substituted across arms; a metric appearing for only one arm is rejected.",
        "- If supplied, sampling seed, generated-image count, and NFE=2 `mid_t` must agree inside each pair.",
        "- The stability gate requires finite losses, legal `r/t` and gap telemetry, active adaptive controller telemetry, and an adaptive AMP-skip rate no more than "
        f"{skip_rate_tolerance:.1%} above the paired fixed run.",
        "- The conclusion is not upgraded from partial coverage, a single seed, or a single favorable NFE setting.",
        "",
        f"Paired metric rows currently available: {len(pairs)}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


PER_SEED_FIELDS = [
    "method", "training_seed", "budget_kimg", "nfe", "mid_t", "sampling_seed", "num_generated", "kid_5k", "fid_5k",
    "checkpoint_sha256", "metric_source", "run_dir", "training_summary_csv", "training_summary_available", "training_processed_kimg",
    "training_attempted_iterations", "training_successful_optimizer_steps", "training_successful_steps_valid", "training_amp_skipped", "training_amp_skip_rate", "training_amp_skip_telemetry_complete",
    "training_nan_count", "training_inf_count", "training_trailing_loss_mean", "training_trailing_loss_std", "training_final_loss",
    "adaptive_active", "final_correction", "correction_abs_max", "correction_saturation_fraction", "correction_sign_changes", "correction_bound_valid",
    "final_r_over_t_mean", "final_gap_mean", "pair_ratio_valid", "pair_gap_valid", "controller_telemetry_complete",
]

PAIR_FIELDS = [
    "metric", "training_seed", "budget_kimg", "nfe", "fixed_value", "adaptive_value", "delta_adaptive_minus_fixed",
    "adaptive_better", "paired_sampling_settings_verified", "sampling_seed", "num_generated", "fixed_checkpoint_sha256", "adaptive_checkpoint_sha256",
]

AGGREGATE_FIELDS = [
    "metric", "budget_kimg", "nfe", "paired_seed_count", "expected_seed_count", "coverage_complete", "adaptive_better_seed_count",
    "adaptive_worse_seed_count", "fixed_mean", "fixed_std", "adaptive_mean", "adaptive_std", "delta_mean", "delta_std",
    "delta_sem", "paired_sampling_settings_verified",
]


def comma_separated_numbers(value: str, field: str, *, integer: bool) -> tuple[int | float, ...]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = parse_int(item, field, minimum=0) if integer else parse_float(item, field, minimum=0)
        assert parsed is not None
        values.append(parsed)
    if not values:
        fail(f"{field} must include at least one value")
    if len(set(values)) != len(values):
        fail(f"{field} contains duplicate values: {value!r}")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", required=True, type=Path, help="Role-A unified per-checkpoint metrics CSV")
    parser.add_argument("--training-records", type=Path, help="Role-B checkpoint/training-summary CSV")
    parser.add_argument("--training-root", action="append", type=Path, default=[], help="Directory to recursively scan for Role-B metadata.json files")
    parser.add_argument("--outdir", required=True, type=Path, help="Empty or existing output directory for Role-C deliverables")
    parser.add_argument("--expected-seeds", default="0,1,2", help="Comma-separated frozen training seeds (default: 0,1,2)")
    parser.add_argument("--expected-budgets", default="16,32,64", help="Comma-separated frozen budgets in kimg (default: 16,32,64)")
    parser.add_argument("--adaptive-max-adjust", type=float, default=0.05, help="Controller saturation reference (default: 0.05)")
    parser.add_argument("--skip-rate-tolerance", type=float, default=0.02, help="Allowed adaptive-minus-fixed AMP skip-rate increase for stability gate (default: 0.02)")
    parser.add_argument("--require-complete", action="store_true", help="Return a non-zero status unless complete quality and training-stability evidence is present")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.adaptive_max_adjust) or args.adaptive_max_adjust < 0:
        fail("--adaptive-max-adjust must be finite and >= 0")
    if not math.isfinite(args.skip_rate_tolerance) or args.skip_rate_tolerance < 0:
        fail("--skip-rate-tolerance must be finite and >= 0")
    expected_seeds = tuple(int(value) for value in comma_separated_numbers(args.expected_seeds, "expected_seeds", integer=True))
    expected_budgets = tuple(float(value) for value in comma_separated_numbers(args.expected_budgets, "expected_budgets", integer=False))
    if tuple(expected_budgets) != EXPECTED_BUDGETS or tuple(expected_seeds) != EXPECTED_SEEDS:
        # The matrix itself is frozen by the task.  Alternate values are useful for a smoke fixture only,
        # but the deliverable labels always retain the original 16/32/64, seeds 0/1/2 protocol.
        fail("Role-C production analysis is frozen to training seeds 0,1,2 and budgets 16,32,64 kimg")
    metrics = read_metrics(args.metrics, expected_budgets)
    training_records: list[TrainingRecord] = []
    if args.training_records is not None:
        training_records.extend(read_training_records(args.training_records, expected_budgets))
    for root in args.training_root:
        training_records.extend(records_from_training_root(root, expected_budgets))
    indexed_records = index_training_records(training_records)
    per_seed = join_metrics_and_training(metrics, indexed_records, args.adaptive_max_adjust)
    pairs = paired_differences(per_seed)
    aggregate = aggregate_differences(pairs, expected_seeds)
    conclusion = conclusion_from_aggregate(aggregate, per_seed, expected_seeds, args.skip_rate_tolerance)
    relationships = mechanism_relationships(per_seed, pairs)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "per_seed_metrics.csv", per_seed, PER_SEED_FIELDS)
    write_csv(args.outdir / "paired_differences.csv", pairs, PAIR_FIELDS)
    write_csv(args.outdir / "aggregate_results.csv", aggregate, AGGREGATE_FIELDS)
    plot_quality(per_seed, args.outdir / "quality_vs_budget.png")
    plot_controller(per_seed, args.outdir / "controller_vs_budget.png")
    write_conclusion(
        args.outdir / "FINAL_CONCLUSION.md",
        conclusion,
        aggregate,
        pairs,
        relationships,
        expected_seeds,
        args.skip_rate_tolerance,
    )

    print(f"[analyze_paired_results] wrote Role-C deliverables to {args.outdir}")
    print(f"[analyze_paired_results] verdict: {conclusion['label']}")
    if args.require_complete and conclusion["label"] == "INCOMPLETE":
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"[analyze_paired_results] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
