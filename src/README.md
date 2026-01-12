# Simulation Code — Temporal Distinguishability Limits

This directory contains all numerical simulations used to verify
the analytical results presented in the paper:

**Information-Theoretic Limits of Time Distinguishability in Diffusion
and Anomalous Diffusion**

All scripts are self-contained, require only NumPy, and are fully
reproducible.

---

## Contents

- `simulate_crlb.py`  
  Monte Carlo verification of the Fisher-information / Cramér–Rao
  bound for time inference in normal diffusion.

- `simulate_photon_limit.py`  
  Self-consistent simulation of the photon-limited temporal
  distinguishability regime, yielding the nontrivial scaling  
  \[
  \Delta t_{\min} \propto \Phi^{-1/3}.
  \]

- `utils.py`  
  Shared utility functions (random seeds, log–log fits, confidence
  intervals, effective sample size).

---

## Requirements

- Python ≥ 3.9  
- NumPy ≥ 1.20  

No additional libraries are required.

---

## How to Run

### 1. Fisher-information / CRLB verification

```bash
python simulate_crlb.py
Expected outcome:

The printed ratio

V
a
r
(
𝑡
^
)
/
C
R
L
B
Var( 
t
^
 )/CRLB
should be close to 1 across a wide range of 
𝑡
t and 
𝑁
N.

This confirms that the estimator achieves the theoretical bound.

2. Photon-limited regime
bash
Code kopieren
python simulate_photon_limit.py
Expected outcome:

The fitted log–log slope should be close to 
−
1
/
3
−1/3.

Typical output:

yaml
Code kopieren
Fitted scaling exponent: -0.33
Expected theoretical value: -1/3 ≈ -0.333
This verifies the self-consistent photon-limited temporal resolution
predicted analytically.

Reproducibility Notes
All scripts fix random seeds by default.

Output data files (*.txt) are written to the working directory
for independent inspection.

Changing physical parameters (D, sigma0, dimension d)
does not alter scaling exponents, only prefactors.

Relation to the Paper
These simulations correspond directly to:

Appendix A — Analytical derivations

Appendix B — Numerical verification

No part of the analytical argument depends on simulation results;
simulations serve as independent verification only.

License
All code is released under the MIT License.


