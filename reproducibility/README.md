# Reproducibility Notes

This directory documents the steps required to reproduce
all analytical figures and numerical results reported
in the accompanying paper:

> *Information-Theoretic Limits of Time Distinguishability in Diffusion  
> and Anomalous Diffusion*

The goal is to ensure that an independent researcher
can verify the results without ambiguity or hidden assumptions.

---

## Computational Environment

The numerical results were generated using:

- Python ≥ 3.9
- NumPy ≥ 1.23

Optional (only for plotting and post-processing):

- Matplotlib ≥ 3.6
- SciPy ≥ 1.9

No GPU acceleration is required.
All simulations run on a standard laptop.

---

## Randomness and Seeds

All simulation scripts fix random seeds explicitly.

Stochastic variability affects numerical prefactors
but **does not affect scaling exponents**.

Any systematic change in scaling behavior
must therefore be treated as a genuine discrepancy,
not numerical noise.

---

## Analytical Reproducibility

All analytical results in the paper are derived symbolically
from explicit probability distributions and standard
information-theoretic identities.

They do not depend on numerical solvers.

Every equation in the main text and appendices
can be independently verified using standard tools
(e.g. symbolic algebra or manual derivation).

---

## Numerical Reproducibility

### Directory Structure

All numerical experiments are located in:

src/

makefile
Code kopieren

### 1. Fisher-Information / CRLB Verification

Run:

```bash
python simulate_crlb.py
Expected outputs:

Console output listing:

empirical Var(t_hat),

theoretical CRLB,

ratio Var(t_hat) / CRLB.

A text file:

Code kopieren
crlb_mc_results.txt
Acceptance criterion:

The ratio Var(t_hat) / CRLB should be close to 1
(typically within ±5–10%) across a wide range of t and N.

This confirms that the estimator saturates the
Cramér–Rao lower bound for the Gaussian diffusion model.

2. Photon-Limited Regime
Run:

bash
Code kopieren
python simulate_photon_limit.py
Expected outputs:

Console output reporting a fitted log–log slope.

A text file:

Code kopieren
photon_limit_results.txt
Acceptance criterion:

The fitted scaling exponent should satisfy:

nginx
Code kopieren
slope ≈ −1/3
typically within the range:

Code kopieren
−0.36 ≤ slope ≤ −0.30
This verifies the self-consistent photon-limited
temporal resolution scaling

Δ
𝑡
min
⁡
∝
Φ
−
1
/
3
.
Δt 
min
​
 ∝Φ 
−1/3
 .
Interpretation of Deviations
If numerical results deviate from analytical predictions,
possible causes include:

Violation of model assumptions
(e.g. correlations, non-Gaussian noise),

Insufficient Monte Carlo statistics,

Implementation errors.

Persistent deviations under controlled conditions
constitute valid falsifications of the framework.

Scope
These notes address reproducibility of scaling laws only.

They do not claim experimental validation,
which must be performed independently
using physical measurement systems.

yaml
Code kopieren
