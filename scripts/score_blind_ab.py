#!/usr/bin/env python3
"""Validate locked A/B ballots, unblind them, and summarize preferences."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


VALID_PREFERENCES = {"A", "B", "TIE"}


def fail(message: str) -> None:
    raise SystemExit(f"[score_blind_ab] ERROR: {message}")


def read_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def load_key(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    keyed = {row.get("trial_id", ""): row for row in rows}
    if len(rows) != 96 or len(keyed) != 96 or "" in keyed:
        fail("private key must contain exactly 96 unique trial IDs")
    for row in rows:
        if {row.get("A_schedule"), row.get("B_schedule")} != {"sigmoid", "adaptive_v1"}:
            fail(f"invalid side key for {row.get('trial_id')}")
    return keyed


def load_responses(paths: list[Path], key: dict[str, dict], min_raters: int) -> list[dict]:
    responses = []
    seen = set()
    rater_trials = defaultdict(set)
    for path in paths:
        for row in read_csv(path):
            trial_id = row.get("trial_id", "").strip()
            rater_id = row.get("rater_id", "").strip()
            preference = row.get("preference_A_B_TIE", row.get("preference", "")).strip().upper()
            if trial_id not in key:
                fail(f"unknown trial ID in {path}: {trial_id}")
            if not rater_id:
                fail(f"missing rater_id for {trial_id} in {path}")
            if preference not in VALID_PREFERENCES:
                fail(f"invalid preference for {trial_id}/{rater_id}: {preference!r}")
            pair = (rater_id, trial_id)
            if pair in seen:
                fail(f"duplicate response for {rater_id}/{trial_id}")
            seen.add(pair)
            rater_trials[rater_id].add(trial_id)
            responses.append({"trial_id": trial_id, "rater_id": rater_id, "preference": preference})
    if len(rater_trials) < min_raters:
        fail(f"found {len(rater_trials)} complete raters; protocol requires at least {min_raters}")
    expected_trials = set(key)
    for rater_id, trials in rater_trials.items():
        if trials != expected_trials:
            fail(f"rater {rater_id} has {len(trials)}/96 trials; incomplete ballots are not scored")
    return responses


def counter() -> dict:
    return {"adaptive_v1": 0, "sigmoid": 0, "tie": 0, "judgments": 0}


def add_result(counts: dict, winner: str) -> None:
    counts[winner] += 1
    counts["judgments"] += 1


def finalize(label: str, counts: dict) -> dict:
    non_ties = counts["adaptive_v1"] + counts["sigmoid"]
    judgments = counts["judgments"]
    return {
        "stratum": label,
        **counts,
        "adaptive_share_excluding_ties": None if non_ties == 0 else counts["adaptive_v1"] / non_ties,
        "adaptive_tie_half_score": None if judgments == 0 else (counts["adaptive_v1"] + 0.5 * counts["tie"]) / judgments,
    }


def summarize(responses: list[dict], key: dict[str, dict]) -> list[dict]:
    groups = defaultdict(counter)
    for response in responses:
        item = key[response["trial_id"]]
        preference = response["preference"]
        winner = "tie" if preference == "TIE" else item[f"{preference}_schedule"]
        training_seed = int(item["training_seed"])
        nfe = int(item["nfe"])
        rater_id = response["rater_id"]
        for label in (
            "overall",
            f"nfe={nfe}",
            f"training_seed={training_seed}",
            f"training_seed={training_seed},nfe={nfe}",
            f"rater={rater_id}",
        ):
            add_result(groups[label], winner)
    labels = ["overall", "nfe=1", "nfe=2"]
    labels += [f"training_seed={seed}" for seed in range(3)]
    labels += [f"training_seed={seed},nfe={nfe}" for seed in range(3) for nfe in (1, 2)]
    labels += sorted(label for label in groups if label.startswith("rater="))
    return [finalize(label, groups[label]) for label in labels]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], rater_count: int) -> None:
    lines = [
        "# Blinded A/B visual preference summary",
        "",
        f"Locked complete ballots: {rater_count}. Preferences are descriptive judgments on 96 method-blinded paired trials per rater.",
        "",
        "| Stratum | Adaptive wins | Fixed wins | Ties | Adaptive share (ties excluded) | Adaptive score (ties=0.5) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        share = "—" if row["adaptive_share_excluding_ties"] is None else f"{row['adaptive_share_excluding_ties']:.3f}"
        score = "—" if row["adaptive_tie_half_score"] is None else f"{row['adaptive_tie_half_score']:.3f}"
        lines.append(
            f"| {row['stratum']} | {row['adaptive_v1']} | {row['sigmoid']} | {row['tie']} | {share} | {score} |"
        )
    lines.extend([
        "",
        "The trials are repeated judgments nested within raters and training seeds; no binomial significance claim is made from the raw judgment count.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-raters", type=int, default=3)
    args = parser.parse_args(argv)

    key = load_key(args.key)
    responses = load_responses(args.responses, key, args.min_raters)
    rows = summarize(responses, key)
    rater_count = len({response["rater_id"] for response in responses})
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "blind_ab_summary.csv", rows)
    payload = {
        "schema_version": 1,
        "method_blinded": True,
        "complete_raters": rater_count,
        "trials_per_rater": 96,
        "judgments": len(responses),
        "summary": rows,
    }
    (outdir / "blind_ab_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(outdir / "blind_ab_summary.md", rows, rater_count)
    print(f"Scored {len(responses)} judgments from {rater_count} complete raters")


if __name__ == "__main__":
    main()
