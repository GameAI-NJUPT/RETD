# Q1: locked counterexample experiments (numerical code only)

This release computes numerical results for the manuscript's two zero-reward
counterexamples. It contains no plotting code, images, or historical raw arrays.
The numerical runner uses Python 3.10+ syntax and requires NumPy. Validation
here used Python 3.12.10 and NumPy 2.3.1; use Python 3.12 for the recorded AST
fingerprint checks, since Python's AST schema can differ between minor versions.
Other versions have not been checked for bitwise agreement.

## Run from the repository root

Short software check (not a scientific replication):

```sh
python q1/verify_provenance.py
python q1/run_locked.py --runs 10 --steps 40 --curve-stride 10 --tail-start 32 --output-dir q1/outputs/smoke
```

Full released locked protocol (expensive; not executed during release QA):

```sh
python q1/run_locked.py --output-dir q1/outputs/formal_locked
```

Select only one environment with `--environment new_two_state` or
`--environment baird_seven_state`. Run `python q1/run_locked.py --help` for all
options. Existing nonempty output directories are rejected, not overwritten.
Each environment's trajectories are held in memory; the default trajectory
arrays alone require roughly 500 MB. Environments are processed sequentially.

## Default protocol and relation to the original scripts

| Setting | New two-state | Baird seven-state |
|---|---|---|
| Held-out runs | 10,000 | 10,000 |
| Transitions per run | 5,000 | 5,000 |
| Inclusive seed range | 202608260000--202608269999 | 202608280000--202608289999 |
| Discount | 0.9 | 0.99 |
| Behavior target/solid-action probability | 0.3 | 1/7 |
| ETD stepsize | 0.1 | 0.01 |
| RETD stepsize and c | 0.1, 4 | 0.01, 0.5 |

Initial features, parameters, follow-on trace, simultaneous RETD updates and
the transition generator are copied unchanged from the original numerical
functions. Rewards are zero. The remaining methods are GTD(0), GTD2, TDC, TDRC
and TETD; their independently selected configurations are read from the included
`selected_configs.json`. The locked ETD/RETD settings replace the tuned settings
in that file. TETD's selected depth is 4 in both environments. Every method
receives the same trajectory for each seed.

The original two-state locked entry point recomputed ETD/RETD but loaded the
other five methods' frozen outputs from an earlier archive. This release instead
recomputes all seven methods using the same preserved simulator, configurations
and seed range. The original Baird locked entry point already recomputed all
seven methods. No historical output archive is needed. A full numerical rerun
has NOT been performed here, so this release does not claim that the reported
10,000-run paper results have been reproduced or are byte-for-byte identical.

## Statistics and files

Each environment writes `summary.csv`, `per_seed.csv`, `configs.json`,
`curves.npz` and `receipt.json`. The receipt records settings, software versions
and file hashes. Nondefault runs are explicitly labeled custom/software checks.

- RMSVE is over all states against the zero value function.
- Finite-time growth is `log(RMSVE_t / RMSVE_0) / t`. The original implementation
  clips RMSVE to `[1e-300, 1e100]` when forming this logarithm.
- Standard deviations are sample standard deviations (`ddof=1`), not standard
  errors. There is no temporal smoothing. At time zero the growth statistics are
  undefined and stored as NaN, exactly as in the original simulator.
- The tail average includes transitions 4,000 through 5,000, i.e. 1,001 samples.
  This inclusive definition is deliberately unchanged.
- The original overflow policy is retained: nonfinite RMSVE or RMSVE above
  `1e100` flags a run; current parameter/auxiliary/omega values are reset to zero,
  the current diagnostic is capped, and final RMSVE of any flagged run is capped.
  These diagnostics are not an uncapped estimator of an infinite population MSE.
- `curves.npz` contains per-method aggregate raw means/stds and growth
  means/stds/fractions expanding. `per_seed.csv` stores tail and final RMSVE and
  divergence flags; it does not store every seed's full trajectory.

## Preserved core and coverage limits

`numerical_core.py` preserves `environments`, `generate_trajectory`, `configs`,
`rmsve`, `simulate`, `config_key` and `select_configs`, including their original
function names and ASTs. `finite_time_growth.py` preserves `simulate_growth`.
All three core dataclasses and the small serialization/hash helpers are also
unchanged. `provenance.json` lists source hashes, preserved-node AST hashes,
changes and release QA. `verify_provenance.py` validates this record locally;
`--source-dir` optionally checks against an original experiments directory.

The old end-to-end tuning/archive-building CLIs and figure-specific historical
loaders are not released. Their numerical candidate generation/selection
functions remain available as APIs, but this CLI does not rerun validation.
The older tuned suite used 200/50 validation runs and 10,000/100 test runs for
two-state/Baird respectively; those older test counts must not be confused with
the two 10,000-run locked protocols above. Plot-only processing of the separate
historical two-state raw archive is omitted. Q2, Q3, Q4, theory certificates and
nonzero-reward tasks are outside this directory's coverage.
