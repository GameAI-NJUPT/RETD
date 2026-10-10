#!/usr/bin/env python3
"""Run Q1's locked protocols from frozen configurations, without archived data."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import numerical_core as base
    from .finite_time_growth import simulate_growth
except ImportError:
    import numerical_core as base
    from finite_time_growth import simulate_growth

HERE = Path(__file__).resolve().parent
SELECTED_CONFIGS = HERE / "selected_configs.json"
PROTOCOLS = {
    "new_two_state": {
        "seed_start": 202608260000,
        "parameter_rule": "counterexample_locked",
        "locked": {"ETD": {"alpha": 0.1}, "RETD": {"alpha": 0.1, "c": 4.0}},
    },
    "baird_seven_state": {
        "seed_start": 202608280000,
        "parameter_rule": "section5_locked",
        "locked": {"ETD": {"alpha": 0.01}, "RETD": {"alpha": 0.01, "c": 0.5}},
    },
}
CURVE_KEYS = (
    "curve_steps", "raw_mean", "raw_std", "growth_mean", "growth_std",
    "fraction_expanding",
)


def configurations(environment: str) -> dict[str, dict[str, float | int]]:
    with SELECTED_CONFIGS.open(encoding="utf-8") as handle:
        selected = json.load(handle)
    result = {method: selected[f"{environment}:{method}"] for method in base.METHODS}
    result.update(PROTOCOLS[environment]["locked"])
    return result


def run_environment(environment: str, args: argparse.Namespace) -> dict[str, Any]:
    protocol = PROTOCOLS[environment]
    seed_start = protocol["seed_start"] if args.seed_start is None else args.seed_start
    seeds = np.arange(seed_start, seed_start + args.runs, dtype=np.int64)
    env = base.environments()[environment]
    selected = configurations(environment)
    output_dir = args.output_dir / environment
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Generating paired trajectories: {environment}, {args.runs} runs, {args.steps} steps", flush=True)
    trajectory = base.generate_trajectory(env, seeds)
    summary_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    for method in args.methods:
        config = selected[method]
        print(f"Running {method}: {base.config_key(config)}", flush=True)
        output = simulate_growth(env, trajectory, method, config)
        tail = output["tail_rmsve"]
        summary_rows.append({
            "environment": environment,
            "method": method,
            "config": base.config_key(config),
            "parameter_rule": protocol["parameter_rule"] if method in protocol["locked"] else "validation_selected",
            "test_runs": len(seeds),
            "final_growth_mean": float(output["growth_mean"][-1]),
            "final_growth_sample_std": float(output["growth_std"][-1]),
            "final_fraction_expanding": float(output["fraction_expanding"][-1]),
            "tail_rmsve_mean": float(np.mean(tail)),
            "tail_rmsve_sample_std": float(np.std(tail, ddof=1)),
            "tail_rmsve_median": float(np.median(tail)),
            "tail_rmsve_q25": float(np.quantile(tail, 0.25)),
            "tail_rmsve_q75": float(np.quantile(tail, 0.75)),
            "divergence_fraction": float(np.mean(output["diverged"])),
        })
        for seed, tail_value, final_value, diverged in zip(
            seeds, tail, output["final_rmsve"], output["diverged"], strict=True
        ):
            seed_rows.append({
                "environment": environment,
                "method": method,
                "seed": int(seed),
                "tail_rmsve": float(tail_value),
                "final_rmsve": float(final_value),
                "diverged": int(diverged),
            })
        for key in CURVE_KEYS:
            archive[f"{method}__{key}"] = output[key]
    base.write_csv(output_dir / "summary.csv", summary_rows)
    base.write_csv(output_dir / "per_seed.csv", seed_rows)
    base.write_json(output_dir / "configs.json", {m: selected[m] for m in args.methods})
    np.savez_compressed(output_dir / "curves.npz", **archive)
    full_protocol = (
        args.runs == 10_000 and args.steps == 5_000 and args.curve_stride == 10
        and args.tail_start == 4_000 and seed_start == protocol["seed_start"]
        and tuple(args.methods) == base.METHODS
    )
    receipt = {
        "schema": "q1-locked-numerical-release-v1",
        "status": "COMPLETED_NUMERICAL_RUN",
        "environment": environment,
        "matches_released_locked_protocol": full_protocol,
        "warning": None if full_protocol else "Custom/software-check run, not the full manuscript experiment.",
        "runs": args.runs,
        "steps": args.steps,
        "curve_stride": args.curve_stride,
        "tail_start_inclusive": args.tail_start,
        "seed_range_inclusive": [int(seeds[0]), int(seeds[-1])],
        "methods": args.methods,
        "same_trajectory_per_seed_for_all_methods": True,
        "zero_rewards": True,
        "metric": "natural-log(RMSVE_t/RMSVE_0)/t, with original numerical floor/cap",
        "sample_std_ddof": 1,
        "temporal_smoothing": False,
        "historical_curve_reuse": False,
        "selected_configs_sha256": base.sha256_file(SELECTED_CONFIGS),
        "numerical_core_sha256": base.sha256_file(HERE / "numerical_core.py"),
        "growth_sha256": base.sha256_file(HERE / "finite_time_growth.py"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "artifacts": {
            name: base.sha256_file(output_dir / name)
            for name in ("summary.csv", "per_seed.csv", "configs.json", "curves.npz")
        },
    }
    base.write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("both", *PROTOCOLS), default="both")
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--curve-stride", type=int, default=10)
    parser.add_argument("--tail-start", type=int, default=4_000)
    parser.add_argument("--seed-start", type=int, default=None,
                        help="Override one environment's first seed; forbidden with --environment both.")
    parser.add_argument("--methods", nargs="+", choices=base.METHODS, default=list(base.METHODS))
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "locked")
    args = parser.parse_args(argv)
    if args.runs < 2:
        parser.error("--runs must be at least 2 because statistics use sample standard deviation.")
    if args.steps < 1 or args.curve_stride < 1 or args.steps % args.curve_stride:
        parser.error("--steps must be a positive multiple of --curve-stride.")
    if not 1 <= args.tail_start <= args.steps:
        parser.error("Require 1 <= --tail-start <= --steps; short checks must override --tail-start explicitly.")
    if args.seed_start is not None and (args.environment == "both" or args.seed_start < 0):
        parser.error("--seed-start must be nonnegative and select exactly one environment.")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must not repeat methods.")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is nonempty; choose a new directory to preserve prior results.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base.STEPS = args.steps
    base.CURVE_STRIDE = args.curve_stride
    base.TAIL_START = args.tail_start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environments = list(PROTOCOLS) if args.environment == "both" else [args.environment]
    for environment in environments:
        receipt = run_environment(environment, args)
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
