from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import formal_suite as suite


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "frozen_protocol_20260810.json"
RESULT_ROOT = HERE / "results" / "additional_tasks_10000runs_20260826"
METHODS = ["GTD2", "TDC", "TDRC", "ETD", "TETD(4)", "RETD"]
VALIDATION_SEEDS = np.arange(2026081000, 2026081010, dtype=np.uint64)
TEST_SEEDS = np.arange(202608300000, 202608310000, dtype=np.uint64)
STRIDE = 10
TAIL_FRACTION = 0.2
DIVERGENCE_THRESHOLD = 1000.0


@dataclass
class MethodState:
    theta: np.ndarray
    weight: np.ndarray
    omega: np.ndarray
    followon: np.ndarray
    history: np.ndarray
    diverged: np.ndarray
    tail_sum: np.ndarray
    tail_count: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def u01(seeds: np.ndarray, step: int, stream: int = 0) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = seeds + np.uint64(step + 1) * np.uint64(0x9E3779B97F4A7C15)
        x = x + np.uint64(stream + 1) * np.uint64(0xD1B54A32D192ED03)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def environments(protocol: dict) -> dict[str, suite.Environment]:
    boyan = suite.boyan_environment(int(protocol["environments"]["boyan_chain"]["steps"]))
    base = suite.random_walk_environment(
        int(protocol["environments"]["off_policy_random_walk"]["steps"])
    )
    tabular = np.eye(5, dtype=float)
    inverted = np.full((5, 5), 0.5, dtype=float)
    np.fill_diagonal(inverted, 0.0)
    dependent = base.phi.copy()
    variants = [
        replace(base, name="random_walk_tabular", phi=tabular, theta0=np.zeros(5)),
        replace(base, name="random_walk_inverted", phi=inverted, theta0=np.zeros(5)),
        replace(base, name="random_walk_dependent", phi=dependent, theta0=np.zeros(3)),
    ]
    return {environment.name: environment for environment in [boyan, *variants]}


def initial_states(environment: suite.Environment, seeds: np.ndarray) -> np.ndarray:
    if environment.kind == "random_walk":
        return np.full(len(seeds), 2, dtype=np.int16)
    return np.floor(13.0 * u01(seeds, -1, 17)).astype(np.int16)


def transition(
    environment: suite.Environment,
    state: np.ndarray,
    seeds: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random_value = u01(seeds, step, 0)
    if environment.kind == "boyan":
        next_state = np.empty_like(state)
        reward = np.empty(len(state), dtype=float)
        rho = np.empty(len(state), dtype=float)
        at_zero = state == 0
        at_one = state == 1
        active = ~(at_zero | at_one)
        next_state[at_zero] = np.floor(13.0 * u01(seeds[at_zero], step, 1)).astype(np.int16)
        reward[at_zero] = 1.5
        rho[at_zero] = 1.0
        next_state[at_one] = 0
        reward[at_one] = 1.5
        rho[at_one] = 1.0
        first = active & (random_value < 0.4)
        second = active & ~first
        next_state[first] = state[first] - 2
        reward[first] = 1.0
        rho[first] = 1.25
        next_state[second] = state[second] - 1
        reward[second] = 2.0
        rho[second] = 5.0 / 6.0
        reset = np.zeros(len(state), dtype=bool)
        return next_state.copy(), reward, rho, reset, next_state

    right = random_value >= 0.5
    candidate = state + np.where(right, 1, -1).astype(np.int16)
    terminal = (candidate < 0) | (candidate >= 5)
    next_state = np.where(terminal, -1, candidate).astype(np.int16)
    reward = (candidate >= 5).astype(float)
    rho = np.where(right, 1.2, 0.8)
    new_current = np.where(terminal, 2, candidate).astype(np.int16)
    return next_state, reward, rho, terminal, new_current


def path_arrays(
    environment: suite.Environment,
    seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.empty((environment.steps, len(seeds)), dtype=np.int16)
    next_states = np.empty_like(states)
    rewards = np.empty((environment.steps, len(seeds)), dtype=np.float64)
    rhos = np.empty_like(rewards)
    resets = np.empty((environment.steps, len(seeds)), dtype=bool)
    current = initial_states(environment, seeds)
    for step in range(environment.steps):
        states[step] = current
        nxt, reward, rho, reset, current = transition(environment, current, seeds, step)
        next_states[step] = nxt
        rewards[step] = reward
        rhos[step] = rho
        resets[step] = reset
    return states, next_states, rewards, rhos, resets


def config_candidates(method: str, protocol: dict) -> list[dict[str, float | int]]:
    return suite.algorithm_configs(method, protocol)


def output_error(
    environment: suite.Environment,
    coordinates: suite.Coordinates,
    method: str,
    theta: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    output_theta = theta
    if method == "RETD":
        recovery_vector = coordinates.basis @ coordinates.u
        output_theta = theta + omega[..., None] * recovery_vector
    prediction = output_theta @ environment.phi.T
    target = suite.target_values(environment)
    return np.sqrt(np.mean(np.square(prediction - target), axis=-1))


def truncated_followon(history: np.ndarray, gamma: float) -> np.ndarray:
    followon = np.ones(history.shape[:-1], dtype=float)
    product = np.ones_like(followon)
    for lag in range(history.shape[-1]):
        product *= history[..., lag]
        followon += (gamma ** (lag + 1)) * product
    return followon


def simulate_validation_method(
    environment: suite.Environment,
    coordinates: suite.Coordinates,
    method: str,
    configs: list[dict[str, float | int]],
    path: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    states, next_states, rewards, rhos, resets = path
    config_count = len(configs)
    run_count = states.shape[1]
    dimension = environment.phi.shape[1]
    theta = np.broadcast_to(environment.theta0, (config_count, run_count, dimension)).copy()
    weight = np.zeros_like(theta)
    omega = np.zeros((config_count, run_count), dtype=float)
    followon = np.ones((config_count, run_count), dtype=float)
    history = np.zeros((config_count, run_count, 4), dtype=float)
    tail_sum = np.zeros((config_count, run_count), dtype=float)
    tail_count = 0
    tail_start = int((1.0 - TAIL_FRACTION) * environment.steps)
    alpha = np.asarray([float(config["alpha"]) for config in configs])[:, None]
    aux = np.asarray([float(config.get("aux_alpha", 0.0)) for config in configs])[:, None]
    regularizer = np.asarray([float(config.get("regularizer", 0.0)) for config in configs])[:, None]
    c_value = np.asarray([float(config.get("c", 0.0)) for config in configs])[:, None]
    for step in range(environment.steps):
        state = states[step]
        next_state = next_states[step]
        phi = environment.phi[state]
        phi_next = np.zeros_like(phi)
        valid_next = next_state >= 0
        phi_next[valid_next] = environment.phi[next_state[valid_next]]
        reward = rewards[step][None, :]
        rho = rhos[step][None, :]
        phi_b = phi[None, :, :]
        phi_next_b = phi_next[None, :, :]
        delta = reward + environment.gamma * np.sum(theta * phi_next_b, axis=-1) - np.sum(theta * phi_b, axis=-1)
        if method == "GTD2":
            correction = np.sum(phi_b * weight, axis=-1)
            theta += alpha[..., None] * rho[..., None] * (phi_b - environment.gamma * phi_next_b) * correction[..., None]
            weight += aux[..., None] * (rho * delta - correction)[..., None] * phi_b
        elif method == "TDC":
            correction = np.sum(phi_b * weight, axis=-1)
            theta += alpha[..., None] * rho[..., None] * (
                delta[..., None] * phi_b - environment.gamma * phi_next_b * correction[..., None]
            )
            weight += aux[..., None] * (rho * delta - correction)[..., None] * phi_b
        elif method == "TDRC":
            correction = np.sum(phi_b * weight, axis=-1)
            theta += alpha[..., None] * rho[..., None] * (
                delta[..., None] * phi_b - environment.gamma * phi_next_b * correction[..., None]
            )
            weight += alpha[..., None] * (
                (rho * delta - correction)[..., None] * phi_b - regularizer[..., None] * weight
            )
        elif method == "ETD":
            theta += alpha[..., None] * (followon * rho * delta)[..., None] * phi_b
        elif method == "TETD(4)":
            truncated = truncated_followon(history, environment.gamma)
            theta += alpha[..., None] * (truncated * rho * delta)[..., None] * phi_b
            history[..., 1:] = history[..., :-1]
            history[..., 0] = rho
            history[:, resets[step], :] = 0.0
        elif method == "RETD":
            signal = followon * rho * delta
            old_omega = omega.copy()
            theta += alpha[..., None] * (signal - old_omega)[..., None] * phi_b
            omega += alpha * (signal - (1.0 + c_value) * old_omega)
        else:
            raise KeyError(method)
        if method in {"ETD", "RETD"}:
            followon = 1.0 + environment.gamma * rho * followon
            followon[:, resets[step]] = 1.0
        if (step + 1) % STRIDE == 0 and step + 1 >= tail_start:
            errors = output_error(environment, coordinates, method, theta, omega)
            errors = np.where(np.isfinite(errors), np.minimum(errors, DIVERGENCE_THRESHOLD), DIVERGENCE_THRESHOLD)
            tail_sum += errors
            tail_count += 1
    tail_metric = tail_sum / tail_count
    rows: list[dict[str, object]] = []
    candidates: list[tuple[float, float, float, int]] = []
    for index, config in enumerate(configs):
        values = tail_metric[index]
        divergence_fraction = float(np.mean(values >= DIVERGENCE_THRESHOLD))
        median = float(np.median(values))
        score = float(np.median(np.log10(np.maximum(values, 1.0e-12))) + 5.0 * divergence_fraction)
        candidates.append((score, divergence_fraction, median, index))
        rows.append(
            {
                "environment": environment.name,
                "method": method,
                "config_id": suite.config_id(config),
                "validation_score": score,
                "divergence_fraction": divergence_fraction,
                "median_tail_rmsve": median,
            }
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2], float(configs[item[3]]["alpha"]), suite.config_id(configs[item[3]])))
    return configs[candidates[0][3]], rows


def initialize_method_state(
    environment: suite.Environment,
    run_count: int,
) -> MethodState:
    dimension = environment.phi.shape[1]
    return MethodState(
        theta=np.broadcast_to(environment.theta0, (run_count, dimension)).copy(),
        weight=np.zeros((run_count, dimension), dtype=float),
        omega=np.zeros(run_count, dtype=float),
        followon=np.ones(run_count, dtype=float),
        history=np.zeros((run_count, 4), dtype=float),
        diverged=np.zeros(run_count, dtype=bool),
        tail_sum=np.zeros(run_count, dtype=float),
        tail_count=0,
    )


def test_environment(
    environment: suite.Environment,
    coordinates: suite.Coordinates,
    selected: dict[str, dict[str, float | int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    run_count = len(TEST_SEEDS)
    current = initial_states(environment, TEST_SEEDS)
    states = {method: initialize_method_state(environment, run_count) for method in METHODS}
    curve_rows: list[dict[str, object]] = []
    checkpoint_steps = set(range(0, environment.steps + 1, STRIDE))
    tail_start = int((1.0 - TAIL_FRACTION) * environment.steps)

    def record(step: int) -> None:
        for method in METHODS:
            method_state = states[method]
            errors = output_error(
                environment,
                coordinates,
                method,
                method_state.theta,
                method_state.omega,
            )
            invalid = ~np.isfinite(errors) | (errors > DIVERGENCE_THRESHOLD)
            method_state.diverged |= invalid
            errors = np.where(method_state.diverged, DIVERGENCE_THRESHOLD, errors)
            if step >= tail_start:
                method_state.tail_sum += errors
                method_state.tail_count += 1
            curve_rows.append(
                {
                    "environment": environment.name,
                    "method": method,
                    "step": step,
                    "runs": run_count,
                    "mean": float(np.mean(errors)),
                    "sample_std": float(np.std(errors, ddof=1)),
                }
            )

    record(0)
    for step in range(environment.steps):
        state = current.copy()
        next_state, reward, rho, reset, current = transition(environment, current, TEST_SEEDS, step)
        phi = environment.phi[state]
        phi_next = np.zeros_like(phi)
        valid_next = next_state >= 0
        phi_next[valid_next] = environment.phi[next_state[valid_next]]
        for method in METHODS:
            method_state = states[method]
            config = selected[method]
            alpha = float(config["alpha"])
            active = ~method_state.diverged
            delta = reward + environment.gamma * np.sum(method_state.theta * phi_next, axis=1) - np.sum(method_state.theta * phi, axis=1)
            if method == "GTD2":
                correction = np.sum(phi * method_state.weight, axis=1)
                method_state.theta[active] += alpha * rho[active, None] * (
                    phi[active] - environment.gamma * phi_next[active]
                ) * correction[active, None]
                method_state.weight[active] += float(config["aux_alpha"]) * (
                    rho[active] * delta[active] - correction[active]
                )[:, None] * phi[active]
            elif method == "TDC":
                correction = np.sum(phi * method_state.weight, axis=1)
                method_state.theta[active] += alpha * rho[active, None] * (
                    delta[active, None] * phi[active]
                    - environment.gamma * phi_next[active] * correction[active, None]
                )
                method_state.weight[active] += float(config["aux_alpha"]) * (
                    rho[active] * delta[active] - correction[active]
                )[:, None] * phi[active]
            elif method == "TDRC":
                correction = np.sum(phi * method_state.weight, axis=1)
                method_state.theta[active] += alpha * rho[active, None] * (
                    delta[active, None] * phi[active]
                    - environment.gamma * phi_next[active] * correction[active, None]
                )
                method_state.weight[active] += alpha * (
                    (rho[active] * delta[active] - correction[active])[:, None] * phi[active]
                    - float(config["regularizer"]) * method_state.weight[active]
                )
            elif method == "ETD":
                method_state.theta[active] += alpha * (
                    method_state.followon[active] * rho[active] * delta[active]
                )[:, None] * phi[active]
            elif method == "TETD(4)":
                truncated = truncated_followon(method_state.history, environment.gamma)
                method_state.theta[active] += alpha * (
                    truncated[active] * rho[active] * delta[active]
                )[:, None] * phi[active]
                method_state.history[:, 1:] = method_state.history[:, :-1]
                method_state.history[:, 0] = rho
                method_state.history[reset] = 0.0
            elif method == "RETD":
                signal = method_state.followon * rho * delta
                old_omega = method_state.omega.copy()
                method_state.theta[active] += alpha * (signal[active] - old_omega[active])[:, None] * phi[active]
                method_state.omega[active] += alpha * (
                    signal[active] - (1.0 + float(config["c"])) * old_omega[active]
                )
            if method in {"ETD", "RETD"}:
                method_state.followon = 1.0 + environment.gamma * rho * method_state.followon
                method_state.followon[reset] = 1.0
            finite = np.all(np.isfinite(method_state.theta), axis=1)
            finite &= np.all(np.isfinite(method_state.weight), axis=1)
            finite &= np.isfinite(method_state.omega)
            new_invalid = ~finite
            if np.any(new_invalid):
                method_state.diverged |= new_invalid
                method_state.theta[new_invalid] = 0.0
                method_state.weight[new_invalid] = 0.0
                method_state.omega[new_invalid] = 0.0
        completed = step + 1
        if completed in checkpoint_steps:
            record(completed)
        if completed % 500 == 0:
            print(f"PROGRESS environment={environment.name} step={completed}/{environment.steps}", flush=True)

    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        method_state = states[method]
        tail = method_state.tail_sum / method_state.tail_count
        for seed, value, diverged in zip(TEST_SEEDS, tail, method_state.diverged, strict=True):
            seed_rows.append(
                {
                    "environment": environment.name,
                    "method": method,
                    "seed": int(seed),
                    "tail_rmsve": float(value),
                    "diverged": int(diverged),
                }
            )
        summary_rows.append(
            {
                "environment": environment.name,
                "method": method,
                "runs": run_count,
                "config_id": suite.config_id(selected[method]),
                "mean": float(np.mean(tail)),
                "sample_std": float(np.std(tail, ddof=1)),
                "median": float(np.median(tail)),
                "q25": float(np.quantile(tail, 0.25)),
                "q75": float(np.quantile(tail, 0.75)),
                "divergence_fraction": float(np.mean(method_state.diverged)),
            }
        )
    return curve_rows, seed_rows, summary_rows


def main() -> None:
    if RESULT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True)
    protocol = suite.load_protocol(PROTOCOL_PATH)
    envs = environments(protocol)
    coordinates = {name: suite.analysis_coordinates(environment) for name, environment in envs.items()}
    selected_all: dict[str, dict[str, dict[str, float | int]]] = {}
    validation_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for environment in envs.values():
        print(f"VALIDATION environment={environment.name}", flush=True)
        path = path_arrays(environment, VALIDATION_SEEDS)
        selected_all[environment.name] = {}
        for method in METHODS:
            selected, rows = simulate_validation_method(
                environment,
                coordinates[environment.name],
                method,
                config_candidates(method, protocol),
                path,
            )
            selected_all[environment.name][method] = selected
            validation_rows.extend(rows)
            print(
                f"SELECTED environment={environment.name} method={method} config={suite.config_id(selected)}",
                flush=True,
            )
    write_csv(RESULT_ROOT / "validation_candidates.csv", validation_rows)
    write_json(RESULT_ROOT / "selected_configs.json", selected_all)

    all_curves: list[dict[str, object]] = []
    all_seeds: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []
    for environment in envs.values():
        print(f"TEST environment={environment.name} runs={len(TEST_SEEDS)}", flush=True)
        curves, seeds, summaries = test_environment(
            environment,
            coordinates[environment.name],
            selected_all[environment.name],
        )
        all_curves.extend(curves)
        all_seeds.extend(seeds)
        all_summaries.extend(summaries)
    write_csv(RESULT_ROOT / "aggregate_learning_curves.csv", all_curves)
    write_csv(RESULT_ROOT / "per_seed_tail_rmsve.csv", all_seeds)
    write_csv(RESULT_ROOT / "endpoint_summary.csv", all_summaries)
    write_json(
        RESULT_ROOT / "completion_receipt.json",
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_path": PROTOCOL_PATH.name,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "script_sha256": sha256(Path(__file__).resolve()),
            "validation_seed_count": len(VALIDATION_SEEDS),
            "test_seed_count": len(TEST_SEEDS),
            "validation_seeds": [int(seed) for seed in VALIDATION_SEEDS],
            "test_seed_first": int(TEST_SEEDS[0]),
            "test_seed_last": int(TEST_SEEDS[-1]),
            "common_random_numbers_within_environment_and_seed": True,
            "test_seed_stream": "counter-based SplitMix64, one stream per integer seed",
            "candidate_budget_per_method_environment": 24,
            "methods": METHODS,
            "environments": list(envs),
            "wall_seconds": time.perf_counter() - started,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    manifest_lines = []
    for path in sorted(RESULT_ROOT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            manifest_lines.append(f"{sha256(path)}  {path.name}")
    (RESULT_ROOT / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"COMPLETED={RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
