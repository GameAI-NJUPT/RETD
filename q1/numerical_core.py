"""Frozen Q1 numerical core; scientific function bodies are unchanged upstream ASTs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STEPS = 5_000
CURVE_STRIDE = 10
TAIL_START = 4_000
DIV_THRESHOLD = 1.0e100
METHODS = ("ETD", "RETD", "GTD", "GTD2", "TDC", "TDRC", "TETD")


@dataclass(frozen=True)
class Environment:
    name: str
    gamma: float
    phi: np.ndarray
    theta0: np.ndarray
    kind: str


@dataclass(frozen=True)
class Trajectory:
    state: np.ndarray
    next_state: np.ndarray
    rho: np.ndarray


@dataclass
class RunOutput:
    tail_rmsve: np.ndarray
    final_rmsve: np.ndarray
    diverged: np.ndarray
    curve_steps: np.ndarray | None = None
    curve_mean: np.ndarray | None = None
    curve_std: np.ndarray | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environments() -> dict[str, Environment]:
    two = Environment(
        name="new_two_state",
        gamma=0.9,
        phi=np.array([[1.0], [2.0]], dtype=float),
        theta0=np.ones(1, dtype=float),
        kind="two_state",
    )
    phi = np.zeros((7, 8), dtype=float)
    for state in range(7):
        phi[state, state] = 2.0
        phi[state, 7] = 1.0
    phi[6, 6] = 1.0
    phi[6, 7] = 2.0
    theta0 = np.ones(8, dtype=float)
    theta0[6] = 10.0
    baird = Environment(
        name="baird_seven_state",
        gamma=0.99,
        phi=phi,
        theta0=theta0,
        kind="baird",
    )
    return {two.name: two, baird.name: baird}


def generate_trajectory(env: Environment, seeds: np.ndarray) -> Trajectory:
    runs = len(seeds)
    state = np.empty((STEPS, runs), dtype=np.int8)
    next_state = np.empty((STEPS, runs), dtype=np.int8)
    rho = np.empty((STEPS, runs), dtype=np.float64)
    for column, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        if env.kind == "two_state":
            current = 1 if rng.random() < 0.3 else 0
            target = rng.random(STEPS) < 0.3
            for step in range(STEPS):
                state[step, column] = current
                current = 1 if target[step] else 0
                next_state[step, column] = current
                rho[step, column] = 10.0 / 3.0 if target[step] else 0.0
        elif env.kind == "baird":
            current = int(rng.integers(0, 7))
            solid = rng.random(STEPS) < (1.0 / 7.0)
            dashed_next = rng.integers(0, 6, size=STEPS)
            for step in range(STEPS):
                state[step, column] = current
                current = 6 if solid[step] else int(dashed_next[step])
                next_state[step, column] = current
                rho[step, column] = 7.0 if solid[step] else 0.0
        else:
            raise KeyError(env.kind)
    return Trajectory(state=state, next_state=next_state, rho=rho)


def configs(method: str) -> list[dict[str, float | int]]:
    single = np.geomspace(1.0e-5, 1.0e-1, 12)
    main = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2)
    ratios = (0.1, 1.0, 10.0)
    if method == "ETD":
        result = [{"alpha": float(alpha)} for alpha in single]
    elif method == "RETD":
        result = [
            {"alpha": alpha, "c": c}
            for alpha in main
            for c in (0.5, 4.0, 8.0)
        ]
    elif method in {"GTD", "GTD2", "TDC"}:
        result = [
            {"alpha": alpha, "aux_alpha": alpha * ratio}
            for alpha in main
            for ratio in ratios
        ]
    elif method == "TDRC":
        result = [
            {"alpha": alpha, "regularizer": regularizer}
            for alpha in main
            for regularizer in (0.1, 1.0, 10.0)
        ]
    elif method == "TETD":
        result = [
            {"alpha": alpha, "depth": depth}
            for alpha in main
            for depth in (4, 6, 10)
        ]
    else:
        raise KeyError(method)
    if len(result) != 12:
        raise AssertionError((method, len(result)))
    return result


def rmsve(env: Environment, theta: np.ndarray) -> np.ndarray:
    prediction = theta @ env.phi.T
    return np.sqrt(np.mean(prediction * prediction, axis=1))


def simulate(
    env: Environment,
    trajectory: Trajectory,
    method: str,
    config: dict[str, float | int],
    record_curve: bool,
) -> RunOutput:
    runs = trajectory.state.shape[1]
    dim = env.phi.shape[1]
    theta = np.repeat(env.theta0[None, :], runs, axis=0)
    auxiliary = np.zeros((runs, dim), dtype=np.float64)
    omega = np.zeros(runs, dtype=np.float64)
    followon = np.ones(runs, dtype=np.float64)
    alpha = float(config["alpha"])
    diverged = np.zeros(runs, dtype=bool)
    tail_sum = np.zeros(runs, dtype=np.float64)
    tail_count = 0
    curve_steps = np.arange(0, STEPS + 1, CURVE_STRIDE, dtype=int)
    curve_mean = np.empty(len(curve_steps), dtype=np.float64) if record_curve else None
    curve_std = np.empty(len(curve_steps), dtype=np.float64) if record_curve else None
    curve_index = 0

    if method == "TETD":
        depth = int(config["depth"])
        products = np.zeros((depth, runs), dtype=np.float64)
        gamma_powers = env.gamma ** np.arange(1, depth + 1)
    else:
        products = np.empty((0, runs), dtype=np.float64)
        gamma_powers = np.empty(0)

    initial = rmsve(env, theta)
    if record_curve:
        curve_mean[0] = float(np.mean(initial))
        curve_std[0] = float(np.std(initial, ddof=1))
        curve_index = 1

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(STEPS):
            states = trajectory.state[step]
            next_states = trajectory.next_state[step]
            rho = trajectory.rho[step]
            phi = env.phi[states]
            phi_next = env.phi[next_states]
            delta = env.gamma * np.sum(phi_next * theta, axis=1) - np.sum(phi * theta, axis=1)

            if method == "ETD":
                theta += alpha * (followon * rho * delta)[:, None] * phi
            elif method == "RETD":
                signal = followon * rho * delta
                old_omega = omega.copy()
                theta += alpha * (signal - old_omega)[:, None] * phi
                omega += alpha * (signal - (1.0 + float(config["c"])) * old_omega)
            elif method == "GTD":
                # Original GTD(0): the target-action substream estimates E[delta phi].
                selected = rho > 0.0
                old_auxiliary = auxiliary.copy()
                auxiliary[selected] += float(config["aux_alpha"]) * (
                    delta[selected, None] * phi[selected] - auxiliary[selected]
                )
                correction = np.sum(phi * old_auxiliary, axis=1)
                theta[selected] += alpha * (
                    phi[selected] - env.gamma * phi_next[selected]
                ) * correction[selected, None]
            elif method == "GTD2":
                correction = np.sum(phi * auxiliary, axis=1)
                old_auxiliary = auxiliary.copy()
                theta += alpha * rho[:, None] * (
                    phi - env.gamma * phi_next
                ) * correction[:, None]
                auxiliary = old_auxiliary + float(config["aux_alpha"]) * (
                    rho * delta - correction
                )[:, None] * phi
            elif method == "TDC":
                correction = np.sum(phi * auxiliary, axis=1)
                old_auxiliary = auxiliary.copy()
                theta += alpha * rho[:, None] * (
                    delta[:, None] * phi - env.gamma * phi_next * correction[:, None]
                )
                auxiliary = old_auxiliary + float(config["aux_alpha"]) * (
                    rho * delta - correction
                )[:, None] * phi
            elif method == "TDRC":
                correction = np.sum(phi * auxiliary, axis=1)
                old_auxiliary = auxiliary.copy()
                theta += alpha * rho[:, None] * (
                    delta[:, None] * phi - env.gamma * phi_next * correction[:, None]
                )
                auxiliary = old_auxiliary + alpha * (
                    (rho * delta - correction)[:, None] * phi
                    - float(config["regularizer"]) * old_auxiliary
                )
            elif method == "TETD":
                truncated = 1.0 + np.sum(gamma_powers[:, None] * products, axis=0)
                theta += alpha * (truncated * rho * delta)[:, None] * phi
            else:
                raise KeyError(method)

            if method in {"ETD", "RETD"}:
                followon = 1.0 + env.gamma * rho * followon
            elif method == "TETD":
                if products.shape[0] > 1:
                    products[1:] = rho[None, :] * products[:-1].copy()
                products[0] = rho

            values = rmsve(env, theta)
            bad = (~np.isfinite(values)) | (values > DIV_THRESHOLD)
            diverged |= bad
            if np.any(bad):
                theta[bad] = 0.0
                auxiliary[bad] = 0.0
                omega[bad] = 0.0
                values[bad] = DIV_THRESHOLD
            if step + 1 >= TAIL_START:
                tail_sum += values
                tail_count += 1
            if record_curve and (step + 1) % CURVE_STRIDE == 0:
                curve_mean[curve_index] = float(np.mean(values))
                curve_std[curve_index] = float(np.std(values, ddof=1))
                curve_index += 1

    final = rmsve(env, theta)
    final[diverged] = DIV_THRESHOLD
    return RunOutput(
        tail_rmsve=tail_sum / max(tail_count, 1),
        final_rmsve=final,
        diverged=diverged,
        curve_steps=curve_steps if record_curve else None,
        curve_mean=curve_mean,
        curve_std=curve_std,
    )


def config_key(config: dict[str, float | int]) -> str:
    return ";".join(f"{key}={config[key]}" for key in sorted(config))


def select_configs(
    envs: dict[str, Environment],
    validation: dict[str, Trajectory],
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, float | int]] = {}
    rows: list[dict[str, Any]] = []
    for env_name, env in envs.items():
        for method in METHODS:
            best_score = math.inf
            best_config: dict[str, float | int] | None = None
            for candidate in configs(method):
                output = simulate(env, validation[env_name], method, candidate, False)
                score = float(
                    np.median(np.log10(np.maximum(output.tail_rmsve, 1.0e-300)))
                    + 10.0 * np.mean(output.diverged)
                )
                rows.append(
                    {
                        "environment": env_name,
                        "method": method,
                        "config": config_key(candidate),
                        "score": score,
                        "tail_median": float(np.median(output.tail_rmsve)),
                        "divergence_fraction": float(np.mean(output.diverged)),
                    }
                )
                if score < best_score:
                    best_score = score
                    best_config = candidate
            if best_config is None:
                raise RuntimeError((env_name, method))
            selected[f"{env_name}:{method}"] = best_config
            print(
                f"selected environment={env_name} method={method} "
                f"config={config_key(best_config)} score={best_score:.8g}",
                flush=True,
            )
    return selected, rows
