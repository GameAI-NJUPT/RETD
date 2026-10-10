"""Short numerical software check, not a reproduction of the formal results."""

from __future__ import annotations

import argparse
import copy
import platform
from pathlib import Path

import numpy as np

import formal_suite as suite
import run_additional_tasks_10000 as additional
import run_diminishing_recovery_20260826 as recovery


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "software_smoke")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError("Choose a fresh output folder; existing results are never overwritten")
    root.mkdir(parents=True)
    recovery.run_experiment(
        output_root=root / "q3", runs=16, steps=400, seed=2026082601,
        b1=5.0, b2=100.0, c1=0.5, c2=2.0, tail_fraction=0.2,
    )
    protocol = suite.load_protocol(HERE / "frozen_protocol_20260810.json")
    shortened = copy.deepcopy(protocol)
    for specification in shortened["environments"].values():
        specification["steps"] = 100
    selected_all, validation_rows, curves, seeds, summaries = {}, [], [], [], []
    original_test_seeds = additional.TEST_SEEDS
    try:
        additional.TEST_SEEDS = original_test_seeds[:16].copy()
        for environment in additional.environments(shortened).values():
            coordinates = suite.analysis_coordinates(environment)
            path = additional.path_arrays(environment, additional.VALIDATION_SEEDS)
            selected = {}
            for method in additional.METHODS:
                candidates = additional.config_candidates(method, protocol)
                assert len(candidates) == 24
                selected[method], rows = additional.simulate_validation_method(
                    environment, coordinates, method, candidates, path,
                )
                validation_rows.extend(rows)
            selected_all[environment.name] = selected
            environment_curves, environment_seeds, environment_summaries = additional.test_environment(
                environment, coordinates, selected,
            )
            curves.extend(environment_curves)
            seeds.extend(environment_seeds)
            summaries.extend(environment_summaries)
    finally:
        additional.TEST_SEEDS = original_test_seeds
    assert len(validation_rows) == 4 * 6 * 24 and len(seeds) == 4 * 6 * 16
    assert all(np.isfinite(row["tail_rmsve"]) for row in seeds)
    suite.write_json(root / "q4" / "smoke_protocol.json", shortened)
    suite.write_json(root / "q4" / "selected_configs.json", selected_all)
    for name, rows in (
        ("validation_candidates.csv", validation_rows),
        ("aggregate_learning_curves.csv", curves),
        ("per_seed_tail_rmsve.csv", seeds),
        ("endpoint_summary.csv", summaries),
    ):
        additional.write_csv(root / "q4" / name, rows)
    suite.write_json(root / "smoke_receipt.json", {
        "status": "complete_software_check_only",
        "formal_reproduction": False,
        "python": platform.python_version(), "numpy": np.__version__,
        "q3_runs": 16, "q3_steps": 400,
        "q4_steps_per_environment": 100,
        "q4_validation_seeds": additional.VALIDATION_SEEDS.tolist(),
        "q4_test_seeds": original_test_seeds[:16].tolist(),
        "q4_candidate_count_per_method_environment": 24,
        "q4_per_seed_rows": len(seeds),
        "q4_recorded_metrics_finite": True,
        "metric_caveat": "The original divergence detection and clipping policy is unchanged.",
        "frozen_protocol_sha256": suite.sha256_file(HERE / "frozen_protocol_20260810.json"),
        "source_sha256": {name: suite.sha256_file(HERE / name) for name in (
            "formal_suite.py", "run_additional_tasks_10000.py",
            "run_diminishing_recovery_20260826.py", "smoke_numerics.py",
        )},
    })
    suite.write_manifest(root)
    print("PASS: short Q3/Q4 numerical software check; no formal experiment was rerun")


if __name__ == "__main__":
    main()
