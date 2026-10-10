# Regularized Emphatic Temporal-Difference Learning (RETD)

Numerical code accompanying **Regularized Emphatic Temporal-Difference
Learning**.

Plotting code and figure files are not included.

## Scope

| Directory | Contents |
| --- | --- |
| `q1/` | Numerical two-state and Baird counterexample experiments, including the locked ETD/RETD configurations and baseline implementations. |
| `reproducibility/` | Diminishing-step fixed-point recovery (Q3), additional prediction tasks (Q4), and their shared numerical implementation and frozen protocol. |
| `verification/` | Exact two-state and Baird RETD certificate checkers and the fixed Baird certificate candidate. |

See each directory's README for execution commands, configurations, and outputs.

The Q2 parameter-grid/phase-map pipeline and large trajectory archives are not
included.

## RETD implementation

Each RETD recursion uses the **single-c RETD** convention: `c` is excess
leakage, so the auxiliary update contains the factor `1 + c`. The Q3
fixed-point recovery experiment also runs copies at two regularization values
and combines their readouts. The RETD
emphatic follow-on trace is not clipped or replaced. Consult the source and
frozen protocol for update order, initialization, step-size schedules,
environment definitions, and readout conventions.

The exact certificate verifiers check the two-state and Baird configurations
documented in `verification/README.md`.

## Environment

Tested with Python 3.12.10, NumPy 2.3.1, and SymPy 1.14.0. Install the
pinned numerical dependencies into an isolated environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
```

No plotting library is required. SymPy is used by the exact Baird verifier.

## Reproducibility

- Experiment settings, seeds, and output definitions are documented in each
  directory. Full runs can be expensive.
- Smoke checks use reduced run counts and horizons to verify execution; use
  the full-run commands for the paper's experimental settings.
- `provenance.json` files record source hashes and verification results.
- New numerical output is written locally and is excluded from version control.
