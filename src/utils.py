#!/usr/bin/env python3
# ============================================================
# utils.py
# ============================================================
# Shared utility functions for simulations in the repository.
#
# Author: Alexander Yashin
# ============================================================

import numpy as np


def set_seed(seed: int = 0):
    """
    Initialize and return a NumPy random generator
    with a fixed seed for reproducibility.
    """
    return np.random.default_rng(seed)


def loglog_fit(x, y):
    """
    Perform a log-log linear fit:
        log(y) = a * log(x) + b

    Returns:
        slope a, intercept b
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("loglog_fit requires positive x and y.")

    logx = np.log(x)
    logy = np.log(y)

    slope, intercept = np.polyfit(logx, logy, 1)
    return slope, intercept


def confidence_interval(data, level=0.68):
    """
    Compute a symmetric confidence interval assuming
    approximately normal statistics.

    Parameters:
        data : array-like
        level : confidence level (default: 0.68 ~ 1 sigma)

    Returns:
        (mean, lower, upper)
    """
    data = np.asarray(data, dtype=float)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    z = {
        0.68: 1.0,
        0.95: 1.96,
        0.99: 2.576,
    }.get(level, None)

    if z is None:
        raise ValueError("Unsupported confidence level.")

    return mean, mean - z * std, mean + z * std


def effective_sample_size(n, autocorr_time=1.0):
    """
    Estimate effective sample size for correlated data.

    Parameters:
        n : total number of samples
        autocorr_time : integrated autocorrelation time

    Returns:
        n_eff = n / (2 * tau)
    """
    if autocorr_time <= 0:
        raise ValueError("autocorr_time must be positive.")

    return n / (2.0 * autocorr_time)
