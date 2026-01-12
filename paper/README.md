# Paper — Information-Theoretic Limits of Time Distinguishability

This directory contains the complete LaTeX source for the paper:

**Information-Theoretic Limits of Time Distinguishability in Diffusion  
and Anomalous Diffusion**

The paper is self-contained and can be compiled with standard LaTeX
toolchains (XeLaTeX or pdfLaTeX).

---

## Directory Structure

paper/
├── main.tex # Main document entry point
├── preamble.tex # Packages and global settings
├── metadata.tex # Title, authors, date
├── sections/
│ ├── 00_abstract.tex
│ ├── 01_introduction.tex
│ ├── 02_problem_setup.tex
│ ├── 03_normal_diffusion_crlb.tex
│ ├── 04_kl_equivalence.tex
│ ├── 05_photon_limited_regime.tex
│ ├── 06_anomalous_diffusion.tex
│ ├── 07_falsifiability_protocols.tex
│ ├── 08_discussion_limits.tex
│ ├── 09_conclusion.tex
│ ├── A_appendix_derivations.tex
│ └── B_appendix_simulations.tex
└── bib/
└── references.bib

yaml
Code kopieren

The file `main.tex` is the **single compilation entry point**.
All sections and appendices are included via `\input`.

---

## Compilation

### Recommended (XeLaTeX)

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
Alternative (pdfLaTeX)
bash
Code kopieren
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
Both methods produce identical output.

Notes on Compilation
No external figures are required.

All equations and derivations are native LaTeX.

Bibliography is managed via bib/references.bib.

Appendices are included automatically via main.tex.

Logical Structure of the Paper
Abstract & Introduction
Motivation and operational definition of time distinguishability.

Problem Setup
Formal statement of the inference problem.

Normal Diffusion (CRLB)
Fisher-information derivation and estimator-independent bound.

KL Equivalence
Hypothesis-testing formulation and equivalence proof.

Photon-Limited Regime
Self-consistent temporal resolution and 
Φ
−
1
/
3
Φ 
−1/3
  scaling.

Anomalous Diffusion
Subdiffusive and superdiffusive regimes.

Falsifiability Protocols
Concrete experimental tests.

Discussion & Limits
Assumptions, robustness, and scope.

Conclusion
Conceptual implications.

Appendices
Full derivations and numerical verification.

Reproducibility
All numerical claims made in the paper are independently verified
by simulation code located in the src/ directory
at the repository root.

Analytical results do not depend on numerical simulations.

License
This document is released under the MIT License.


