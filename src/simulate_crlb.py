#!/usr/bin/env python3
# ============================================================
# simulate_crlb.py
# ============================================================
# Monte Carlo verification of Fisher-information / CRLB bound
# for normal diffusion-time inference under the model:
#
#   X ~ N(0, 2 D t I_d)  with known D and independent samples
#
# For this Gaussian model:
#   I_1(t) = d / (2 t^2)
#   I_N(t) = N * d / (2 t^2)
#   CRLB: Var(t_hat) >= 2 t^2 / (d N)
#
# We use the unbiased estimator based on the sufficient statistic:
#   t_hat = (1/(2 D d N)) * sum_{i=1}^N ||X_i||^2
# which is efficient for the Gaussian family and attains the CRLB.
#
# Output: empirical Var(t_hat) vs theoretical CRLB.
#
# Author: Alexander Yashin
# ============================================================

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np


def simulate_many(rng: np.random.Generator, D: float, t: float, d: int, N: int, n_trials: int) -> np.ndarray:
    """
    Vectorized simulation of n_trials independent repetitions.

    Draw samples:
        X_{k,i} ~ N(0, 2 D t I_d),  k=1..n_trials, i=1..N
    Return:
        t_hat[k] for each trial, using the quadratic sufficient statistic.
    """
    if D <= 0.0:
        raise ValueError("D must be positive.")
    if t <= 0.0:
        raise ValueError("t must be positive.")
    if d <= 0 or N <= 0 or n_trials <= 1:
        raise ValueError("d, N must be positive integers and n_trials must be > 1.")

    sigma2 = 2.0 * D * t  # variance per coordinate
    # Shape: (n_trials, N, d)
    X = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=(n_trials, N, d))
    # Sum over particles and coordinates per trial:
    r2_sum = np.sum(X * X, axis=(1, 2))
    t_hat = r2_sum / (2.0 * D * d * N)
    return t_hat


def run_grid(
    D: float = 1.0,
    d: int = 2,
    t_values: Optional[np.ndarray] = None,
    N_values: Optional[List[int]] = None,
    n_trials: int = 20000,
    seed: int = 123,
) -> List[Dict[str, float]]:
    """
    For each (t, N), run Monte Carlo and compare:
      empirical Var(t_hat)  vs  CRLB(t,N) = 2 t^2 / (d N)

    Notes:
    - Assumes D is known (no nuisance-parameter penalty).
    - Assumes independent samples; correlated data should use N_eff < N.
    """
    if t_values is None:
        t_values = np.logspace(-2, 1, 7)  # 0.01 .. 10
    if N_values is None:
        N_values = [10, 30, 100, 300, 1000]

    rng = np.random.default_rng(seed)
    results: List[Dict[str, float]] = []

    for t in t_values:
        t = float(t)
        for N in N_values:
            N_int = int(N)
            hats = simulate_many(rng, D=D, t=t, d=d, N=N_int, n_trials=n_trials)

            mean_hat = float(np.mean(hats))
            var_hat = float(np.var(hats, ddof=1))
            crlb = 2.0 * (t ** 2) / (d * N_int)

            results.append(
                {
                    "t": t,
                    "N": N_int,
                    "mean(t_hat)": mean_hat,
                    "Var(t_hat)": var_hat,
                    "CRLB": float(crlb),
                    "ratio Var/CRLB": float(var_hat / crlb),
                }
            )
    return results


def print_results(results: List[Dict[str, float]], max_rows: Optional[int] = None) -> None:
    """
    Pretty print a subset (or all).
    """
    header = "t        N     mean(t_hat)     Var(t_hat)        CRLB        Var/CRLB"
    print(header)
    print("-" * len(header))

    rows = results if max_rows is None else results[:max_rows]
    for r in rows:
        print(
            f"{r['t']:>6.3g}  "
            f"{int(r['N']):>5d}  "
            f"{r['mean(t_hat)']:>12.6g}  "
            f"{r['Var(t_hat)']:>12.6g}  "
            f"{r['CRLB']:>10.6g}  "
            f"{r['ratio Var/CRLB']:>9.4f}"
        )


def save_results_txt(results: List[Dict[str, float]], out_path: str) -> None:
    """
    Save machine-readable output (same columns as in the printed table).
    """
    out = np.array(
        [
            [r["t"], r["N"], r["mean(t_hat)"], r["Var(t_hat)"], r["CRLB"], r["ratio Var/CRLB"]]
            for r in results
        ],
        dtype=float,
    )
    np.savetxt(
        out_path,
        out,
        header="t   N   mean_t_hat   var_t_hat   CRLB   var_over_crlb",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo verification of CRLB for diffusion-time inference: X ~ N(0, 2 D t I_d)."
    )
    p.add_argument("--D", type=float, default=1.0, help="Diffusion coefficient D (must be > 0).")
    p.add_argument("--d", type=int, default=2, help="Dimension d (positive integer).")
    p.add_argument("--trials", type=int, default=30000, help="Monte Carlo trials per (t,N) point.")
    p.add_argument("--seed", type=int, default=20260112, help="RNG seed for reproducibility.")
    p.add_argument(
        "--out",
        type=str,
        default="crlb_mc_results.txt",
        help="Output path for results table (text).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Default scenario: dimension typical for SPT, D normalized to 1.
    results = run_grid(D=args.D, d=args.d, n_trials=args.trials, seed=args.seed)

    print("Monte Carlo verification of CRLB for diffusion-time inference")
    print("Model: X ~ N(0, 2 D t I_d), estimator t_hat = (sum ||X||^2)/(2 D d N)")
    print(f"Parameters: D={args.D}, d={args.d}, trials per point={args.trials}, seed={args.seed}")
    print()

    print_results(results)

    ratios = np.array([r["ratio Var/CRLB"] for r in results], dtype=float)
    print()
    print(f"Sanity check: Var/CRLB ratio range over grid: min={ratios.min():.4f}, max={ratios.max():.4f}")

    # Save machine-readable output
    save_results_txt(results, args.out)
    print(f"Saved: {args.out}")
