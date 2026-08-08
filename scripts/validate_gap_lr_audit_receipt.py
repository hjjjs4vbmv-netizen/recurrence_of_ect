#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

EXPERIMENT_ID = "gap_lr_matched_q128_s3_v1"
C0_DEFINITION = "dot(delta_1_3,delta_1_0)/dot(delta_1_3,delta_1_3)"
CHECKPOINTS = [32, 64, 128, 256]


def fail(message):
    raise SystemExit("AUDIT RECEIPT REJECTED: " + message)


def need(obj, key, where):
    if not isinstance(obj, dict) or key not in obj:
        fail("missing %s.%s" % (where, key))
    return obj[key]


def truth(obj, key, where):
    if need(obj, key, where) is not True:
        fail("%s.%s must be true" % (where, key))


def number(value, name, positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        fail(name + " must be numeric")
    if not math.isfinite(value) or (positive and value <= 0):
        fail(name + " must be finite" + (" and positive" if positive else ""))
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def git(repo, *args):
    return subprocess.check_output(
        ["git"] + list(args),
        cwd=str(repo),
        text=True,
    ).strip()


def validate_state(state, index):
    where = "optimizer_mechanism_gate.states[%d]" % index
    steps = need(state, "successful_optimizer_steps", where)
    if not isinstance(steps, int) or steps < 6:
        fail(where + ".successful_optimizer_steps must be >= 6")

    for key in (
        "same_pre_state",
        "paired_minibatch_rng",
        "amp_invariants_passed",
        "virtual_step_non_committing",
        "parameter_hash_unchanged",
        "optimizer_hash_unchanged",
        "gradscaler_hash_unchanged",
    ):
        truth(state, key, where)

    for key in (
        "state_id",
        "checkpoint_sha256",
        "pre_parameter_sha256",
        "pre_optimizer_sha256",
        "pre_gradscaler_sha256",
        "radam_branch",
        "amp_skip_history_sha256",
        "paired_input_sha256",
    ):
        value = need(state, key, where)
        if not isinstance(value, str) or not value:
            fail("%s.%s must be a nonempty string" % (where, key))

    metrics = need(state, "metrics", where)
    for key in (
        "raw_a_star",
        "raw_residual",
        "update_s_star",
        "update_c_star",
        "update_cosine",
        "reference_update_norm",
        "candidate_update_norm",
        "R_opt",
        "H_exact",
        "H_thresholded",
        "off_support_energy",
        "excluded_reference_energy",
        "abs_H_exact_minus_R_opt",
        "abs_H_thresholded_minus_R_opt",
    ):
        number(
            need(metrics, key, where + ".metrics"),
            where + ".metrics." + key,
        )

    for key in ("h_update", "h_moment_ideal", "layerwise"):
        value = need(metrics, key, where + ".metrics")
        if not isinstance(value, dict) or not value:
            fail("%s.metrics.%s must be a nonempty object" % (where, key))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--transfer", required=True, type=Path)
    args = parser.parse_args()

    if not args.receipt.is_file():
        fail("receipt file does not exist")
    if not args.data.is_file() or not args.transfer.is_file():
        fail("data and transfer must be regular files")

    receipt = json.loads(args.receipt.read_text())
    if need(receipt, "schema_version", "receipt") != 1:
        fail("unsupported schema_version")
    if need(receipt, "experiment_id", "receipt") != EXPERIMENT_ID:
        fail("experiment_id mismatch")
    if need(receipt, "status", "receipt") != "passed":
        fail("status is not passed")
    if need(receipt, "verdict", "receipt") != "formal_launch_allowed":
        fail("verdict does not allow formal launch")

    source = need(receipt, "source", "receipt")
    training_commit = need(source, "training_code_commit", "receipt.source")
    protocol_commit = need(source, "protocol_commit", "receipt.source")
    if git(args.repo, "rev-parse", "HEAD") != protocol_commit:
        fail("HEAD does not equal receipt protocol_commit")
    git(args.repo, "cat-file", "-e", training_commit + "^{commit}")
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            training_commit,
            "--",
            "ct_train.py",
            "training",
        ],
        cwd=str(args.repo),
    )
    if result.returncode != 0:
        fail("working training code differs from training_code_commit")
    if digest(args.data) != need(source, "dataset_sha256", "receipt.source"):
        fail("dataset hash mismatch")
    if digest(args.transfer) != need(
        source, "transfer_checkpoint_sha256", "receipt.source"
    ):
        fail("transfer checkpoint hash mismatch")

    fresh = need(receipt, "fresh_linearized_control", "receipt")
    if need(fresh, "status", "fresh_linearized_control") != "passed":
        fail("fresh-linearized control did not pass")
    if need(fresh, "c0_definition", "fresh_linearized_control") != C0_DEFINITION:
        fail("c0 definition mismatch")
    c0_star = number(
        need(fresh, "c0_star", "fresh_linearized_control"),
        "c0_star",
        positive=True,
    )
    for key in (
        "paired_inputs",
        "amp_invariants_passed",
        "state_hashes_unchanged",
    ):
        truth(fresh, key, "fresh_linearized_control")

    gate = need(receipt, "optimizer_mechanism_gate", "receipt")
    if need(gate, "status", "optimizer_mechanism_gate") != "passed":
        fail("nonzero-state optimizer mechanism gate did not pass")
    states = need(gate, "states", "optimizer_mechanism_gate")
    if not isinstance(states, list) or len(states) < 3:
        fail("optimizer gate requires at least three audited states")
    for index, state in enumerate(states):
        validate_state(state, index)

    longitudinal = need(receipt, "longitudinal_audit", "receipt")
    if need(
        longitudinal, "checkpoint_kimg", "longitudinal_audit"
    ) != CHECKPOINTS:
        fail("longitudinal checkpoint axis mismatch")
    truth(
        longitudinal,
        "identical_counterfactual_state_required",
        "longitudinal_audit",
    )

    print(format(c0_star, ".17g"))


if __name__ == "__main__":
    try:
        main()
    except (json.JSONDecodeError, subprocess.CalledProcessError) as error:
        fail(str(error))
