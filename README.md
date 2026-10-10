# Regularized Emphatic Temporal-Difference Learning (RETD)

Numerical code accompanying the manuscript **Regularized Emphatic
Temporal-Difference Learning**. This release is derived from the August 2026
submission sources, not from subsequent theoretical extensions.

**Plotting code and figure files are intentionally excluded.** The manuscript,
editorial correspondence, and later theory drafts are also not included.

## Scope

| Directory | Contents |
| --- | --- |
| `q1/` | Numerical two-state and Baird counterexample experiments, including the locked ETD/RETD configurations and baseline implementations. |
| `reproducibility/` | Diminishing-step fixed-point recovery (Q3), additional prediction tasks (Q4), and their shared numerical implementation and frozen protocol. |
| `verification/` | Exact two-state and Baird RETD certificate checkers and the fixed Baird certificate candidate. |

See each directory's README for runnable commands, configuration details, and
the changes made while separating numerical code from plotting code.

This is **not a one-command reproduction of every paper figure**. In particular,
the Q2 parameter-grid/phase-map pipeline and historical large trajectory archives
are not included. The Q1 release provides a numerical runner rather than the
historical plotting/archive-reuse entry points. Numerical results and figures
from the original submission have not been uploaded as newly rerun results.

## Algorithm identity

Each RETD recursion uses the original **single-c RETD** convention: `c` is excess
leakage, so the auxiliary update contains the factor `1 + c`. The original Q3
fixed-point recovery experiment also runs copies at two regularization values
and combines their readouts. Those original recovery routines are included;
later post-submission algorithm or theory modifications are not. The RETD
emphatic follow-on trace is not clipped or replaced. Consult the source and
frozen protocol for update order, initialization, step-size schedules,
environment definitions, and readout conventions.

The certificate programs verify particular mathematical certificates. Passing
those checks does not establish an unrestricted finite-variance or prediction-risk
guarantee for arbitrary off-policy tasks.

## Environment

The release checks use Python 3.12.10, NumPy 2.3.1, and SymPy 1.14.0. Install the
pinned numerical dependencies into an isolated environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
```

No plotting library is required. SymPy is used by the exact Baird verifier.

## Reproducibility and limitations

- The full experiment defaults and frozen configurations preserve the original
  source settings; full runs can be expensive.
- Short smoke runs check software execution only. They are not paper-scale
  replications, independent empirical validation, or new scientific results.
- The numerical algorithms retained from the original source are compared at
  the Python abstract-syntax-tree level; `provenance.json` files identify source
  hashes, retained functions, and permitted packaging/metadata changes.
- New numerical output is written locally and is excluded from version control.
- No license has been selected for this repository; no additional reuse license
  is granted by this release.

## Provenance

Source packages: `jmlr_submission_revised_20260826` and
`jmlr_submission_theory_revised_20260828`. The latter's eleven result artifacts
match the corresponding artifacts in the August 29 submission by SHA-256.
That check establishes artifact correspondence, not a fresh experimental rerun.

Release date: 2026-10-10.
