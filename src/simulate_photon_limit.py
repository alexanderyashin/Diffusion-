#!/usr/bin/env python3
# ============================================================
# simulate_photon_limit.py
# ============================================================
# Numerical verification of the photon-limited temporal
# distinguishability scaling:
#     Δt_min ∝ Φ^{-1/3}
#
# IMPORTANT (role separation):
# - t0 is the elapsed physical diffusion time parameter we want to infer.
# - Δt is the acquisition/exposure window that determines photon count:
#       N_gamma ~ Poisson(Φ Δt).
# - Self-consistency arises because achieving smaller Δt requires photons,
#   but photons require exposure time.
#
# This script verifies the Φ^{-1/3} scaling in a way consistent with the
# manuscript's photon-limited derivation.
#
# Author: Alexander Yashin
# ============================================================

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    d: int = 2                  # spatial dimension
    D: float = 1.0e-12          # diffusion coefficient [m^2/s]
    sigma0: float = 200e-9      # PSF width / localization floor [m]
    t0: float = 5.0e-3          # elapsed time parameter to be inferred [s]
    kappa: float = 2.0          # Var(sigma^2_hat) = kappa * sigma^4 / N_gamma
    z: float = 1.0              # confidence multiplier (1.0 = 1σ)
    n_iter: int = 25            # fixed-point iterations
    n_trials: int = 2000        # Monte Carlo trials per Φ (keep CI-friendly)
    seed: int = 42              # RNG seed


def analytic_dt_min(phi: float, p: Params) -> float:
    """
    Closed-form self-consistent prediction (deterministic, no Poisson noise).

    Model:
      sigma_obs^2(t0) = sigma0^2 + 2 d D t0
      Var(t_hat) ≈ [kappa / (4 d^2 D^2)] * sigma_obs^4 / N_gamma
      N_gamma = phi * Δt
      Self-consistency: Δt = z * sqrt(Var(t_hat))

    => Δt^3 = z^2 * [kappa / (4 d^2 D^2)] * sigma_obs^4 / phi
    """
    if phi <= 0.0:
        raise ValueError("phi must be positive.")
    sigma2 = p.sigma0**2 + 2.0 * p.d * p.D * p.t0
    pref = (p.z**2) * (p.kappa / (4.0 * (p.d**2) * (p.D**2))) * (sigma2**2)
    dt = (pref / phi) ** (1.0 / 3.0)
    return float(dt)


def fixed_point_mc(phi: float, p: Params, rng: np.random.Generator) -> np.ndarray:
    """
    Monte Carlo fixed-point iteration with Poisson photon counts.

    Returns:
      dt_estimates: array of length n_trials containing Δt_min estimates.
    """
    # Initialize Δt guesses (same start for all trials; fixed-point will converge).
    dt = np.full(p.n_trials, 1e-3, dtype=float)

    # sigma_obs depends on elapsed time parameter t0, not on Δt.
    sigma2 = p.sigma0**2 + 2.0 * p.d * p.D * p.t0

    # Precompute constant piece of Var(t_hat) excluding 1/N_gamma.
    # Var(t_hat) = A / N_gamma, where A = kappa * sigma_obs^4 / (4 d^2 D^2)
    A = p.kappa * (sigma2**2) / (4.0 * (p.d**2) * (p.D**2))

    for _ in range(p.n_iter):
        # Photon counts per trial
        lam = phi * dt
        # Avoid pathological lam=0
        lam = np.maximum(lam, 1e-12)
        N_gamma = rng.poisson(lam=lam)

        # Enforce a minimal photon count to avoid division blowups; if too low, extend dt.
        too_low = N_gamma < 5
        if np.any(too_low):
            dt[too_low] *= 2.0
            # Update lam for these in next iteration; keep others going.
            N_gamma = np.where(too_low, 5, N_gamma)

        var_t = A / N_gamma.astype(float)

        # Self-consistent update: Δt := z * sqrt(Var(t_hat))
        dt = p.z * np.sqrt(var_t)

    return dt


def fit_slope(log_x: np.ndarray, log_y: np.ndarray) -> Tuple[float, float]:
    """
    Fit log_y = a + b log_x; return (b, SE_b).
    """
    x = np.asarray(log_x, dtype=float)
    y = np.asarray(log_y, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 points for slope SE.")

    X = np.vstack([np.ones_like(x), x]).T
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta[0], beta[1]

    y_hat = a + b * x
    resid = y - y_hat
    s2 = np.sum(resid**2) / (n - 2)
    Sxx = np.sum((x - np.mean(x))**2)
    se_b = np.sqrt(s2 / Sxx)
    return float(b), float(se_b)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify photon-limited scaling Δt_min ∝ Φ^{-1/3}.")
    ap.add_argument("--t0", type=float, default=5.0e-3, help="Elapsed diffusion time parameter t0 [s].")
    ap.add_argument("--D", type=float, default=1.0e-12, help="Diffusion coefficient D [m^2/s].")
    ap.add_argument("--d", type=int, default=2, help="Dimension d.")
    ap.add_argument("--sigma0", type=float, default=200e-9, help="PSF/localization floor sigma0 [m].")
    ap.add_argument("--kappa", type=float, default=2.0, help="Variance-estimator coefficient kappa (ideal Gaussian -> 2).")
    ap.add_argument("--z", type=float, default=1.0, help="Confidence multiplier z (1.0=1σ, 1.96=95%%).")
    ap.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials per Φ.")
    ap.add_argument("--iters", type=int, default=25, help="Fixed-point iterations.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--out", type=str, default="photon_limit_results.txt", help="Output file.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    p = Params(
        d=args.d,
        D=args.D,
        sigma0=args.sigma0,
        t0=args.t0,
        kappa=args.kappa,
        z=args.z,
        n_iter=args.iters,
        n_trials=args.trials,
        seed=args.seed,
    )

    # Photon flux range [photons / second]
    phi_values = np.logspace(3, 7, 20)

    rng = np.random.default_rng(seed=p.seed)

    # Analytic baseline (deterministic)
    dt_analytic = np.array([analytic_dt_min(phi, p) for phi in phi_values], dtype=float)

    # Monte Carlo verification
    dt_mc_mean = []
    dt_mc_std = []

    for phi in phi_values:
        dt_samples = fixed_point_mc(phi=float(phi), p=p, rng=rng)
        dt_mc_mean.append(float(np.mean(dt_samples)))
        dt_mc_std.append(float(np.std(dt_samples, ddof=1)))

    dt_mc_mean = np.array(dt_mc_mean, dtype=float)
    dt_mc_std = np.array(dt_mc_std, dtype=float)

    # Log-log fit for scaling using Monte Carlo means
    log_phi = np.log(phi_values)
    log_dt = np.log(dt_mc_mean)
    slope, slope_se = fit_slope(log_phi, log_dt)

    print("Photon-limited temporal resolution scaling (self-consistent)")
    print("----------------------------------------------------------")
    print("Model roles: elapsed parameter t0 fixed; acquisition window Δt controls photons Nγ ~ Poisson(ΦΔt)")
    print(f"Parameters: d={p.d}, D={p.D:.3e} m^2/s, sigma0={p.sigma0:.3e} m, t0={p.t0:.3e} s")
    print(f"Monte Carlo: trials/Φ={p.n_trials}, iterations={p.n_iter}, seed={p.seed}")
    print()
    print(f"Fitted scaling exponent (MC means): {slope:.4f} ± {slope_se:.4f} (SE)")
    print("Expected theoretical value: -1/3 ≈ -0.3333")
    print()

    print("Φ [photons/s]    Δt_min_MC [s]    std_MC [s]     Δt_min_analytic [s]")
    for phi, dtm, dts, dta in zip(phi_values, dt_mc_mean, dt_mc_std, dt_analytic):
        print(f"{phi:10.3e}    {dtm:12.4e}   {dts:10.3e}    {dta:12.4e}")

    # Save results
    np.savetxt(
        args.out,
        np.column_stack((phi_values, dt_mc_mean, dt_mc_std, dt_analytic)),
        header="Phi [photons/s]   Delta_t_min_MC_mean [s]   Delta_t_min_MC_std [s]   Delta_t_min_analytic [s]",
    )
    print()
    print(f"Saved: {args.out}")
