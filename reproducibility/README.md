# Numerical reproducibility: Q3 and Q4

This directory contains numerical experiment code only. It does not generate
manuscript images or depend on visualization libraries. Python 3.10 or later and
NumPy are required. The release software checks used Python 3.12.10 and NumPy
2.3.1 on Windows; other platforms were not tested in this release audit.

The original algorithms, numerical calculations, seeds, tuning budgets, and
formal experiment defaults are preserved. The changes are limited to removing
Q3's visualization implementation and making protocol/output locations and
receipt file identifiers portable. No scientific result was recomputed at the
formal 10,000-run scale during release preparation.

## Files

| File | Purpose |
| --- | --- |
| `run_diminishing_recovery_20260826.py` | Q3: diminishing-stepsize, nonzero-reward fixed-point recovery on a bounded-trace two-state task. |
| `run_additional_tasks_10000.py` | Q4: vectorized 10,000-test-seed experiments on Boyan and three random-walk feature representations. |
| `formal_suite.py` | Shared environment, coordinate, algorithm, and protocol utilities; also the original separate multi-task experiment runner. |
| `frozen_protocol_20260810.json` | Byte-identical original frozen protocol, including the shared 24-candidate tuning grids. |
| `smoke_numerics.py` | Short, explicitly non-formal numerical software check. |
| `provenance.json` | Upstream/release hashes, exact AST-equivalence scope, commands, and release-check results. |
| `SHA256SUMS` | Checksums of the files in this directory at release preparation. |

Run the commands below from this directory. Existing result folders are not
overwritten; choose a new output folder when repeating a run.

## Q3: formal diminishing-stepsize recovery

```sh
python -B run_diminishing_recovery_20260826.py --output results/q3_10000runs
```

The retained defaults are 10,000 paired runs, 20,000 transitions, seed 2026082601,
stepsize `5/(t+100)`, `c1=0.5`, `c2=2.0`, and final-20% averaging. The output
contains the configuration, analytic fixed-point quantities, one row per run,
summary statistics, and checksums in a completion receipt.

The metric is RMS prediction error **to the analytic ETD fixed prediction** on
this task, not a claim of general constant-stepsize or heavy-tail prediction
risk. Raw RETD, model-based single-c recovery, model-free two-c recovery, and ETD
use paired trajectories. The two-c denominator and invalid flag are retained in
the per-run output. A short run can produce unreliable or invalid readouts;
successful execution is not a theorem about a ratio estimator.

## Q4: formal additional tasks

```sh
python -B run_additional_tasks_10000.py
```

This runs the original full schedule and writes to
`results/additional_tasks_10000runs_20260826/`:

- 10 validation seeds, 24 candidate configurations per method/environment;
- 10,000 held-out test seeds, 202608300000 through 202608309999;
- 5,000 transitions for Boyan and 10,000 for each random-walk representation;
- GTD2, TDC, TDRC, ETD, TETD(4), and RETD, with common random numbers within each
  environment/seed and the original final-20% evaluation policy.

Outputs include candidate scores, selected configurations, aggregate numerical
learning curves, per-seed tail errors, endpoint mean/sample-standard-deviation
and quantile summaries, and checksums. **RETD's Q4 reported prediction is the
model-based aligned single-c readout**, using analytically computed task
coordinates, while the other methods use their ordinary parameter prediction.
This is not a comparison of raw RETD against raw ETD. Divergence detection and
error clipping at 1,000 follow the original implementation; finite recorded
metrics must not be interpreted as uncensored finite risk.

## Quick software check

```sh
python -B smoke_numerics.py --output results/software_smoke
```

The helper invokes the same numerical functions with Q3 at 16 runs/400 steps
and Q4 at 100 steps per environment, 10 validation seeds, 16 test seeds, all six
methods, and all 24 candidates. Its overrides exist only in memory. It does not
edit the frozen protocol, alter formal defaults, or provide evidence for the
paper's 10,000-run results. The receipt, per-run/per-seed values, and summaries
are marked as a software check.

## Separate legacy suite runner

```sh
python -B formal_suite.py --mode smoke --output-root results --run-id legacy_smoke
python -B formal_suite.py --mode full --output-root results --run-id legacy_full
```

This independent runner defaults to the adjacent frozen protocol. Its original
`smoke` mode uses two validation seeds, two test seeds, and two candidates, but
retains the protocol's full horizons. Its `full` mode uses 10 validation seeds
and 30 test seeds. Neither command replaces the Q3/Q4 10,000-run commands above.
The suite includes trajectory records and additional two-c recovery diagnostics.

## Release audit and limits

The original frozen protocol is byte-identical. For each Python experiment
module, the complete release AST equals the original AST after an explicitly
enumerated set of non-numerical removals and portability edits. All unaffected
function/class ASTs are checked separately. This includes the updates, sampling,
initialization, configuration selection, readouts, and numerical summaries;
it is not merely an import or compilation check.

Q3, all four Q4 environments with all six methods, the public short-run helper,
and the separate legacy suite were exercised at reduced scale. The legacy
release check additionally used a separate in-memory-derived 100-step protocol,
not a modification to the frozen file. Both experiment CLIs exited successfully
with empty stderr. The detailed audit tool and generated validation artifacts
are retained outside this release directory. They are not required to run the
released numerical code. `provenance.json` identifies the checks and records
their limits without embedding private absolute paths.

The checks establish source preservation and reduced-scale executability, not
independent mathematical correctness, a new formal experiment, or reproduction
of every manuscript result. The original invalid-denominator and divergence
policies are preserved rather than silently changed during packaging.
