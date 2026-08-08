import copy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from scripts import validate_gap_lr_audit_receipt as validator


def valid_state():
    metrics = {
        "raw_a_star": 1.0,
        "raw_residual": 0.01,
        "update_s_star": 1.0,
        "update_c_star": 1.0,
        "update_cosine": 0.99,
        "reference_update_norm": 2.0,
        "candidate_update_norm": 2.0,
        "R_opt": 0.02,
        "H_exact": 0.02,
        "H_thresholded": 0.021,
        "off_support_energy": 0.0,
        "excluded_reference_energy": 0.0,
        "abs_H_exact_minus_R_opt": 0.0,
        "abs_H_thresholded_minus_R_opt": 0.001,
        "h_update": {"weighted_mean": 1.0},
        "h_moment_ideal": {"weighted_mean": 1.0},
        "layerwise": {"count": 1},
    }
    return {
        "state_id": "step-6",
        "successful_optimizer_steps": 6,
        "same_pre_state": True,
        "paired_minibatch_rng": True,
        "amp_invariants_passed": True,
        "virtual_step_non_committing": True,
        "parameter_hash_unchanged": True,
        "optimizer_hash_unchanged": True,
        "gradscaler_hash_unchanged": True,
        "checkpoint_sha256": "a" * 64,
        "pre_parameter_sha256": "b" * 64,
        "pre_optimizer_sha256": "c" * 64,
        "pre_gradscaler_sha256": "d" * 64,
        "radam_branch": "rectified",
        "amp_skip_history_sha256": "e" * 64,
        "paired_input_sha256": "f" * 64,
        "metrics": metrics,
    }


class ReceiptValidatorTests(unittest.TestCase):
    def test_valid_nonzero_state_passes(self):
        validator.validate_state(valid_state(), 0)

    def test_fewer_than_six_successful_steps_rejected(self):
        state = valid_state()
        state["successful_optimizer_steps"] = 5
        with self.assertRaisesRegex(SystemExit, "must be >= 6"):
            validator.validate_state(state, 0)

    def test_missing_history_gauge_rejected(self):
        state = valid_state()
        del state["metrics"]["h_moment_ideal"]
        with self.assertRaisesRegex(SystemExit, "missing"):
            validator.validate_state(state, 0)

    def test_nonfinite_metric_rejected(self):
        state = valid_state()
        state["metrics"]["R_opt"] = float("nan")
        with self.assertRaisesRegex(SystemExit, "must be finite"):
            validator.validate_state(state, 0)

    def test_git_helper_uses_cwd_instead_of_dash_c(self):
        repo = Path("/tmp/example-repository")
        with mock.patch.object(
            validator.subprocess,
            "check_output",
            return_value="abc123\\n",
        ) as check_output:
            result = validator.git(repo, "rev-parse", "HEAD")

        self.assertEqual(result, "abc123")
        check_output.assert_called_once_with(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            text=True,
        )

    def test_missing_receipt_fails_closed(self):
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "validate_gap_lr_audit_receipt.py"),
                "--receipt",
                "/definitely/missing/receipt.json",
                "--repo",
                str(repo),
                "--data",
                "/definitely/missing/data.zip",
                "--transfer",
                "/definitely/missing/model.pkl",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("receipt file does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
