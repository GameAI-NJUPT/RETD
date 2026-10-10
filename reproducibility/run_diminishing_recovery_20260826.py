"""Formal diminishing-stepsize fixed-point-recovery experiment.

The experiment is intentionally separate from the zero-reward counterexamples.
It uses a bounded-trace nonzero-reward two-state prediction task and compares
raw RETD, model-based single-c recovery, model-free two-c recovery, and ETD on
paired trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def prediction_error(theta: np.ndarray, theta_e: float) -> np.ndarray:
    # phi=(1, 3/5); RMS error to the ETD fixed prediction.
    return np.abs(theta - theta_e) * math.sqrt((1.0 + 0.6**2) / 2.0)


def run_experiment(
    output_root: Path,
    runs: int,
    steps: int,
    seed: int,
    b1: float,
    b2: float,
    c1: float,
    c2: float,
    tail_fraction: float,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    config = {
        "schema_version": 1,
        "status": "frozen_before_run",
        "purpose": "nonzero-reward diminishing-stepsize fixed-point recovery",
        "environment": {
            "states": [0, 1],
            "actions": ["left", "right"],
            "behavior_right_probability": 0.75,
            "target_action": "right",
            "left_destination": 0,
            "right_destination": 1,
            "reward": "1 iff current_state=1 and action=right; 0 otherwise",
            "gamma": 0.5,
            "phi": [1.0, 0.6],
            "initial_state": "independent stationary Bernoulli(0.75)",
            "F0": 1.0,
            "theta0": 0.0,
            "omega0": 0.0,
        },
        "stepsize": {"form": "B1/(t+B2)", "B1": b1, "B2": b2},
        "regularizers": [c1, c2],
        "runs": runs,
        "steps": steps,
        "seed": seed,
        "tail_fraction": tail_fraction,
        "paired_trajectories": True,
        "invalid_denominator_threshold": 1e-8,
        "reported_statistics": ["mean", "sample_std", "median", "IQR", "min", "max"],
    }
    config_path = output_root / "config.json"
    write_json(config_path, config)

    # Exact mean-system quantities for this task.
    A = 49.0 / 100.0
    q = 7.0 / 10.0
    b = 21.0 / 20.0
    beta = 7.0 / 4.0
    theta_e = b / A
    u = q / A
    s = q * u
    m_e = beta - q * theta_e
    analytic = {
        "A": A,
        "q": q,
        "b": b,
        "beta": beta,
        "theta_E": theta_e,
        "u_A_inverse_q": u,
        "s_q_A_inverse_q": s,
        "m_E": m_e,
        "c1": {
            "c": c1,
            "omega": m_e / (1.0 + c1 - s),
            "theta": theta_e - u * m_e / (1.0 + c1 - s),
        },
        "c2": {
            "c": c2,
            "omega": m_e / (1.0 + c2 - s),
            "theta": theta_e - u * m_e / (1.0 + c2 - s),
        },
    }
    write_json(output_root / "analytic_solution.json", analytic)

    rng = np.random.default_rng(seed)
    state = rng.random(runs) < 0.75
    F = np.ones(runs, dtype=float)
    theta_e_alg = np.zeros(runs, dtype=float)
    theta1 = np.zeros(runs, dtype=float)
    theta2 = np.zeros(runs, dtype=float)
    omega1 = np.zeros(runs, dtype=float)
    omega2 = np.zeros(runs, dtype=float)
    tail_start = int(math.floor((1.0 - tail_fraction) * steps))
    tail_count = steps - tail_start
    sums = {
        "theta_e": np.zeros(runs),
        "theta1": np.zeros(runs),
        "theta2": np.zeros(runs),
        "omega1": np.zeros(runs),
        "omega2": np.zeros(runs),
    }

    start = time.perf_counter()
    for t in range(steps):
        alpha = b1 / (t + b2)
        right = rng.random(runs) < 0.75
        rho = (4.0 / 3.0) * right
        phi = np.where(state, 0.6, 1.0)
        next_phi = np.where(right, 0.6, 1.0)
        reward = np.logical_and(state, right).astype(float)

        delta_e = reward + 0.5 * next_phi * theta_e_alg - phi * theta_e_alg
        g_e = F * rho * delta_e
        theta_e_alg += alpha * g_e * phi

        delta1 = reward + 0.5 * next_phi * theta1 - phi * theta1
        g1 = F * rho * delta1
        theta1 += alpha * (g1 - omega1) * phi
        omega1 += alpha * (g1 - (1.0 + c1) * omega1)

        delta2 = reward + 0.5 * next_phi * theta2 - phi * theta2
        g2 = F * rho * delta2
        theta2 += alpha * (g2 - omega2) * phi
        omega2 += alpha * (g2 - (1.0 + c2) * omega2)

        F = 1.0 + 0.5 * rho * F
        state = right

        if t >= tail_start:
            sums["theta_e"] += theta_e_alg
            sums["theta1"] += theta1
            sums["theta2"] += theta2
            sums["omega1"] += omega1
            sums["omega2"] += omega2

    wall_seconds = time.perf_counter() - start
    averages = {name: values / tail_count for name, values in sums.items()}
    denominator = averages["omega1"] - averages["omega2"]
    invalid = np.abs(denominator) <= 1e-8
    two_c_theta = (
        averages["omega1"] * averages["theta2"]
        - averages["omega2"] * averages["theta1"]
    ) / denominator
    single1_theta = averages["theta1"] + u * averages["omega1"]
    single2_theta = averages["theta2"] + u * averages["omega2"]

    metrics = {
        "raw_c1": prediction_error(averages["theta1"], theta_e),
        "raw_c2": prediction_error(averages["theta2"], theta_e),
        "single_c1": prediction_error(single1_theta, theta_e),
        "single_c2": prediction_error(single2_theta, theta_e),
        "two_c": prediction_error(two_c_theta, theta_e),
        "etd": prediction_error(averages["theta_e"], theta_e),
    }

    per_seed_path = output_root / "per_run.csv"
    with per_seed_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "run", "theta_etd", "theta_c1", "omega_c1", "theta_c2", "omega_c2",
            "denominator", "invalid", "raw_c1_error", "raw_c2_error",
            "single_c1_error", "single_c2_error", "two_c_error", "etd_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(runs):
            writer.writerow(
                {
                    "run": index,
                    "theta_etd": averages["theta_e"][index],
                    "theta_c1": averages["theta1"][index],
                    "omega_c1": averages["omega1"][index],
                    "theta_c2": averages["theta2"][index],
                    "omega_c2": averages["omega2"][index],
                    "denominator": denominator[index],
                    "invalid": bool(invalid[index]),
                    "raw_c1_error": metrics["raw_c1"][index],
                    "raw_c2_error": metrics["raw_c2"][index],
                    "single_c1_error": metrics["single_c1"][index],
                    "single_c2_error": metrics["single_c2"][index],
                    "two_c_error": metrics["two_c"][index],
                    "etd_error": metrics["etd"][index],
                }
            )

    summary = {
        "runs": runs,
        "steps": steps,
        "tail_count": tail_count,
        "invalid_count": int(np.sum(invalid)),
        "denominator_absolute": summarize(np.abs(denominator)),
        "metrics": {name: summarize(values) for name, values in metrics.items()},
        "wall_seconds": wall_seconds,
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)

    receipt = {
        "status": "complete",
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(config_path),
        "outputs": {
            per_seed_path.name: sha256(per_seed_path),
            summary_path.name: sha256(summary_path),
            "analytic_solution.json": sha256(output_root / "analytic_solution.json"),
        },
    }
    write_json(output_root / "receipt.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026082601)
    parser.add_argument("--b1", type=float, default=5.0)
    parser.add_argument("--b2", type=float, default=100.0)
    parser.add_argument("--c1", type=float, default=0.5)
    parser.add_argument("--c2", type=float, default=2.0)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    args = parser.parse_args()
    run_experiment(
        output_root=args.output,
        runs=args.runs,
        steps=args.steps,
        seed=args.seed,
        b1=args.b1,
        b2=args.b2,
        c1=args.c1,
        c2=args.c2,
        tail_fraction=args.tail_fraction,
    )


if __name__ == "__main__":
    main()
