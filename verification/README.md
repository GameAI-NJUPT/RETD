# Exact certificate verification

This directory contains two deterministic verifiers from the manuscript's
pre-submission August 2026 source. They are not training programs, Monte Carlo
experiments, or general-purpose RETD stability tests. No plotting code or
plotting dependency is included.

## Requirements

- Python 3.11 or later (release checks used Python 3.12.10).
- SymPy for the Baird verifier (release checks used SymPy 1.14.0).
- The two-state verifier uses only the Python standard library.

Install the Baird dependency with `python -m pip install sympy`.

## Run from the repository root

```sh
python verification/verify_two_state_p3over10_c4_exact_20260823.py --output verification/outputs/two_state_recheck.json
python verification/verify_baird_retd_certificate.py --output verification/outputs/baird_recheck.json
```

Both scripts also work from another working directory when invoked by their
absolute paths. Their default candidate and output paths are relative to the
script, not to the current working directory. Both refuse to overwrite an
existing output: select a new `--output` filename when repeating a check.

The Baird verifier accepts an explicit `--candidate` path. The intended and
audited use is the supplied frozen candidate at
`certificates/baird_retd_alpha_0p01_c_0p5_candidate_20260810.json`; this interface
does not make the hard-coded analytic tail bounds valid for arbitrary models.
Only the candidate's historical `fit_script` metadata was normalized from a
local absolute path to an upstream package-relative source identifier. That
identifier is not a bundled file or runtime dependency. Every other JSON key
and value is deeply identical to the upstream candidate, so the mathematical
candidate is unchanged, but the complete file is not byte-identical. No
candidate fitting or optimization is performed by these verifiers.

## What is checked

- **Two states:** exact rational matrix arithmetic and enclosed logarithms
  verify the fixed zero-reward model with behavior target-action probability
  `3/10`, discount `9/10`, features `(1, 2)`, step size `1/10`, and RETD
  regularizer `4`. The certificate checks ETD's positive cycle-log lower bound
  and RETD's negative expected cycle-log-norm upper bound. Success is `PASS`
  with every entry of `checks` equal to `true`.
- **Baird:** exact rational matrices, symmetry-reduced positive-definiteness
  checks, and enclosed logarithms verify the supplied quadratic-form
  certificate at step size `1/100` and regularizer `1/2`. Success is
  `certified`, proving the stated physical-time Lyapunov upper bound
  `-1/10000` for this fixed model.

These are matrix-product certificates. They do not by themselves certify
nonzero-reward prediction error, finite stationary moments, or a nonlinear
two-regularizer readout.

## Source preservation and release checks

In the scripts, only script-relative candidate/output path constants have
changed. Every function, including each `main`, has the same Python AST as
its upstream counterpart; all model constants are unchanged. The candidate
has only the metadata normalization described above, verified by deep JSON
equality after excluding `fit_script`. `provenance.json` records the distinct
source/release hashes, allowed path changes, and actual release-time checks,
including a fresh Baird verification after this metadata change.

The two-state output retains the original schema's `date: 2026-08-23` and two
historical `numerical_full_*` estimates. They are not a new run timestamp or
newly recomputed Monte Carlo estimates. The release-time UTC timestamps and
fresh exact-check summaries are recorded separately in `provenance.json`.
Printed decimal bounds summarize exact rational comparisons; the decisions
are not made from floating-point rounding.

Generated `outputs/` files are local verification artifacts, not required
inputs and not part of the source release. No historical output archive is
bundled.
