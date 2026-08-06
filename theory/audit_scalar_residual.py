#!/usr/bin/env python3
"""Reproduce the scalar-rescaling audit for the PR #34 stop-gradient toy.

The script intentionally reimplements the small A/H/T operators from their
definitions instead of importing ``true_sg_operator.py``.  This keeps the audit
independent of the implementation being checked and avoids plotting/pandas
dependencies.  The output is deterministic for a fixed seed and sample count.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from toy_core import base_gap_sigmoid, sample_t


GAPS = np.round(np.arange(0.5, 1.5001, 0.05), 4)
BUDGETS = (20, 50, 100, 200, 500, 1000)
MODES = ("fixed", "H_matched", "A_matched")


def features(t: np.ndarray, delta0: np.ndarray, g: float, t_min: float = 1e-3):
    """Return residual feature v_g(t) and online Jacobian J_t."""
    delta = np.minimum(g * delta0, t - t_min)
    r = t - delta
    v = np.stack([delta, t**2 - r**2], axis=-1)
    J = np.stack([t, t**2], axis=-1)
    return v, J


def mean_operator(sigma_d: float, v: np.ndarray, J: np.ndarray) -> np.ndarray:
    """A_g = sigma_d^2 E[J_t v_g(t)^T]."""
    return sigma_d**2 * (J.T @ v / len(v))


def loss_hessian(sigma_d: float, v: np.ndarray) -> np.ndarray:
    """H_g = sigma_d^2 E[v_g(t) v_g(t)^T]."""
    return sigma_d**2 * (v.T @ v / len(v))


def second_moment_operator(
    sigma_d: float,
    eta: float,
    A: np.ndarray,
    v: np.ndarray,
    J: np.ndarray,
) -> np.ndarray:
    """Return T_g on the symmetric basis [M00, M01, M11]."""
    bases = (
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    )
    coefficient = (3.0 * eta**2 * sigma_d**4 / len(v)) * np.einsum(
        "na,nb,ni,nj->abij", v, v, J, J
    )
    columns = []
    for basis in bases:
        updated = basis - eta * (A @ basis + basis @ A.T)
        updated += np.einsum("abij,ab->ij", coefficient, basis)
        columns.append([updated[0, 0], updated[0, 1], updated[1, 1]])
    return np.asarray(columns).T


def finite_horizon_errors(T: np.ndarray, beta0: np.ndarray) -> dict[int, float]:
    """Return Tr(M_K) for every requested budget."""
    moment = np.array(
        [beta0[0] ** 2, beta0[0] * beta0[1], beta0[1] ** 2], dtype=float
    )
    errors = {}
    previous = 0
    for budget in BUDGETS:
        for _ in range(budget - previous):
            moment = T @ moment
        errors[budget] = float(moment[0] + moment[2])
        previous = budget
    return errors


def relative_frobenius(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.linalg.norm(numerator) / np.linalg.norm(denominator))


def audit(sample_count: int, seed: int, sigma_d: float):
    t = sample_t(sample_count, rng=np.random.default_rng(seed))
    delta0 = base_gap_sigmoid(t)
    beta0 = np.ones(2) * 1e-2

    v1, J = features(t, delta0, 1.0)
    A1 = mean_operator(sigma_d, v1, J)
    H1 = loss_hessian(sigma_d, v1)
    eta1 = 0.005 / max(abs(np.linalg.eigvals(A1)))
    T1 = second_moment_operator(sigma_d, eta1, A1, v1, J)

    rows = []
    for g in GAPS:
        v, Jg = features(t, delta0, float(g))
        A = mean_operator(sigma_d, v, Jg)
        H = loss_hessian(sigma_d, v)
        scalar = float(np.sum(A * A1) / np.sum(A1 * A1))

        eta_by_mode = {
            "fixed": eta1,
            "H_matched": eta1
            * float(np.linalg.eigvalsh(H1)[-1] / np.linalg.eigvalsh(H)[-1]),
            "A_matched": eta1 / scalar,
        }
        errors_by_mode = {}
        for mode in MODES:
            T = second_moment_operator(sigma_d, eta_by_mode[mode], A, v, Jg)
            errors_by_mode[mode] = finite_horizon_errors(T, beta0)

        sample_residual = Jg[:, :, None] * (v - scalar * v1)[:, None, :]
        sample_operator = Jg[:, :, None] * v[:, None, :]
        sample_rms_relative = float(
            np.sqrt(
                np.mean(np.sum(sample_residual**2, axis=(1, 2)))
                / np.mean(np.sum(sample_operator**2, axis=(1, 2)))
            )
        )
        T_matched = second_moment_operator(
            sigma_d, eta_by_mode["A_matched"], A, v, Jg
        )

        row = {
            "g": float(g),
            "a_g": scalar,
            "eta_fixed": eta_by_mode["fixed"],
            "eta_H_matched": eta_by_mode["H_matched"],
            "eta_A_matched": eta_by_mode["A_matched"],
            "A_residual_relative": relative_frobenius(A - scalar * A1, A),
            "sample_operator_rms_residual_relative": sample_rms_relative,
            "T_update_residual_relative": relative_frobenius(
                (T_matched - np.eye(3)) - (T1 - np.eye(3)), T1 - np.eye(3)
            ),
        }
        for mode in MODES:
            for budget in BUDGETS:
                row[f"E_{mode}_K{budget}"] = errors_by_mode[mode][budget]
        rows.append(row)

    return rows


def write_csv(rows, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    for field in (
        "A_residual_relative",
        "sample_operator_rms_residual_relative",
        "T_update_residual_relative",
    ):
        print(f"max {field}: {max(row[field] for row in rows):.9e}")
    for mode in MODES:
        for budget in BUDGETS:
            field = f"E_{mode}_K{budget}"
            values = np.asarray([row[field] for row in rows])
            g_star = rows[int(np.argmin(values))]["g"]
            spread = float((values.max() - values.min()) / values.min())
            print(f"{mode:9s} K={budget:4d}: g*={g_star:.2f}, spread={spread:.9e}")


def parse_args():
    default_output = Path(__file__).with_name("audit_scalar_residual.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-d", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = audit(args.sample_count, args.seed, args.sigma_d)
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
