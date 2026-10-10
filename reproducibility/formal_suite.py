from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "frozen_protocol_20260810.json"
EPS = 1.0e-12


@dataclass(frozen=True)
class Environment:
    name: str
    gamma: float
    phi: np.ndarray
    theta0: np.ndarray
    p_mu_current: np.ndarray
    p_pi_bootstrap: np.ndarray
    r_pi: np.ndarray
    evaluation_states: np.ndarray
    steps: int
    kind: str

    @property
    def num_states(self) -> int:
        return int(self.phi.shape[0])


@dataclass(frozen=True)
class Coordinates:
    basis: np.ndarray
    phi_r: np.ndarray
    d_mu: np.ndarray
    f: np.ndarray
    A: np.ndarray
    b: np.ndarray
    q: np.ndarray
    beta: float
    u: np.ndarray
    s: float
    theta_e_r: np.ndarray
    theta_e: np.ndarray
    v_e: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    state: np.ndarray
    next_state: np.ndarray
    reward: np.ndarray
    rho: np.ndarray
    reset_after: np.ndarray


@dataclass(frozen=True)
class RunResult:
    steps: np.ndarray
    primary_curve: np.ndarray
    aligned_curve: np.ndarray
    raw_curve: np.ndarray
    divergence_step: int | None
    tail_primary: float
    tail_aligned: float
    tail_raw: float
    tail_theta: np.ndarray
    tail_omega: float
    final_theta: np.ndarray
    final_omega: float
    runtime_seconds: float


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol["status"] != "frozen_before_formal_rerun":
        raise ValueError("The formal suite requires a pre-run frozen protocol")
    if protocol["retd_update"]["c_semantics"].find("1+c") < 0:
        raise ValueError("RETD c convention is not the manuscript convention")
    return protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stationary_distribution(p: np.ndarray) -> np.ndarray:
    n = p.shape[0]
    system = p.T - np.eye(n)
    system[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    d = np.linalg.solve(system, rhs)
    d = np.maximum(d, 0.0)
    if not np.isclose(d.sum(), 1.0):
        raise ValueError("Invalid stationary distribution")
    return d / d.sum()


def analysis_coordinates(env: Environment) -> Coordinates:
    d_mu = stationary_distribution(env.p_mu_current)
    f = np.linalg.solve(
        np.eye(env.num_states) - env.gamma * env.p_pi_bootstrap.T,
        d_mu,
    )
    emphatic = np.diag(f)
    A_full = (
        env.phi.T
        @ emphatic
        @ (np.eye(env.num_states) - env.gamma * env.p_pi_bootstrap)
        @ env.phi
    )
    b_full = env.phi.T @ emphatic @ env.r_pi
    q_full = env.phi.T @ d_mu
    beta = float(f @ env.r_pi)
    _, singular, vt = np.linalg.svd(env.phi, full_matrices=False)
    tol = max(env.phi.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tol))
    basis = vt[:rank].T
    phi_r = env.phi @ basis
    A = basis.T @ A_full @ basis
    b = basis.T @ b_full
    q = basis.T @ q_full
    if np.linalg.matrix_rank(A) != rank:
        raise np.linalg.LinAlgError(f"Singular reduced ETD matrix for {env.name}")
    theta_e_r = np.linalg.solve(A, b)
    u = np.linalg.solve(A, q)
    return Coordinates(
        basis=basis,
        phi_r=phi_r,
        d_mu=d_mu,
        f=f,
        A=A,
        b=b,
        q=q,
        beta=beta,
        u=u,
        s=float(q @ u),
        theta_e_r=theta_e_r,
        theta_e=basis @ theta_e_r,
        v_e=phi_r @ theta_e_r,
    )


def baird_environment(steps: int) -> Environment:
    phi = np.zeros((7, 8), dtype=float)
    for state in range(7):
        phi[state, state] = 2.0
        phi[state, 7] = 1.0
    phi[6, 6] = 1.0
    phi[6, 7] = 2.0
    p_mu = np.full((7, 7), 1.0 / 7.0)
    p_pi = np.zeros((7, 7))
    p_pi[:, 6] = 1.0
    theta0 = np.ones(8)
    theta0[6] = 10.0
    return Environment(
        name="baird_seven_state",
        gamma=0.99,
        phi=phi,
        theta0=theta0,
        p_mu_current=p_mu,
        p_pi_bootstrap=p_pi,
        r_pi=np.zeros(7),
        evaluation_states=np.arange(7),
        steps=steps,
        kind="baird",
    )


def prediction_two_state_environment(steps: int) -> Environment:
    return Environment(
        name="prediction_two_state",
        gamma=0.9,
        phi=np.array([[1.0], [0.6]]),
        theta0=np.ones(1),
        p_mu_current=np.full((2, 2), 0.5),
        p_pi_bootstrap=np.array([[0.0, 1.0], [0.0, 1.0]]),
        r_pi=np.array([0.0, 1.0]),
        evaluation_states=np.arange(2),
        steps=steps,
        kind="prediction_two_state",
    )


def boyan_environment(steps: int) -> Environment:
    phi = np.array(
        [
            [1, 0, 0, 0],
            [0.75, 0.25, 0, 0],
            [0.5, 0.5, 0, 0],
            [0.25, 0.75, 0, 0],
            [0, 1, 0, 0],
            [0, 0.75, 0.25, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.25, 0.75, 0],
            [0, 0, 1, 0],
            [0, 0, 0.75, 0.25],
            [0, 0, 0.5, 0.5],
            [0, 0, 0.25, 0.75],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    n = phi.shape[0]
    p_mu = np.zeros((n, n))
    p_pi = np.zeros((n, n))
    p_mu[0] = 1.0 / n
    p_pi[0] = 1.0 / n
    p_mu[1, 0] = 1.0
    p_pi[1, 0] = 1.0
    for state in range(2, n):
        p_mu[state, state - 2] = 0.4
        p_mu[state, state - 1] = 0.6
        p_pi[state, state - 2] = 0.5
        p_pi[state, state - 1] = 0.5
    return Environment(
        name="boyan_chain",
        gamma=0.9,
        phi=phi,
        theta0=np.ones(4),
        p_mu_current=p_mu,
        p_pi_bootstrap=p_pi,
        r_pi=np.full(n, 1.5),
        evaluation_states=np.arange(n),
        steps=steps,
        kind="boyan",
    )


def random_walk_environment(steps: int) -> Environment:
    phi = np.array(
        [
            [1.0, 0.0, 0.0],
            np.sqrt([0.5, 0.5, 0.0]),
            np.sqrt([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            np.sqrt([0.0, 0.5, 0.5]),
            [0.0, 0.0, 1.0],
        ]
    )
    n = 5
    p_mu_current = np.zeros((n, n))
    p_pi_bootstrap = np.zeros((n, n))
    for state in range(n):
        if state > 0:
            p_mu_current[state, state - 1] += 0.5
            p_pi_bootstrap[state, state - 1] += 0.4
        else:
            p_mu_current[state, 2] += 0.5
        if state < n - 1:
            p_mu_current[state, state + 1] += 0.5
            p_pi_bootstrap[state, state + 1] += 0.6
        else:
            p_mu_current[state, 2] += 0.5
    return Environment(
        name="off_policy_random_walk",
        gamma=0.9,
        phi=phi,
        theta0=np.zeros(3),
        p_mu_current=p_mu_current,
        p_pi_bootstrap=p_pi_bootstrap,
        r_pi=np.array([0.0, 0.0, 0.0, 0.0, 0.6]),
        evaluation_states=np.arange(5),
        steps=steps,
        kind="random_walk",
    )


def build_environments(protocol: dict[str, Any]) -> dict[str, Environment]:
    specs = protocol["environments"]
    envs = [
        baird_environment(int(specs["baird_seven_state"]["steps"])),
        prediction_two_state_environment(int(specs["prediction_two_state"]["steps"])),
        boyan_environment(int(specs["boyan_chain"]["steps"])),
        random_walk_environment(int(specs["off_policy_random_walk"]["steps"])),
    ]
    return {env.name: env for env in envs}


def generate_trajectory(env: Environment, seed: int) -> Trajectory:
    rng = np.random.default_rng(seed)
    n = env.steps
    state = np.empty(n, dtype=np.int16)
    next_state = np.empty(n, dtype=np.int16)
    reward = np.empty(n, dtype=float)
    rho = np.empty(n, dtype=float)
    reset_after = np.zeros(n, dtype=bool)
    current = 2 if env.kind == "random_walk" else int(rng.integers(env.num_states))
    for step in range(n):
        state[step] = current
        if env.kind == "baird":
            solid = bool(rng.random() < 1.0 / 7.0)
            nxt = 6 if solid else int(rng.integers(0, 6))
            rew = 0.0
            ratio = 7.0 if solid else 0.0
            current = nxt
        elif env.kind == "prediction_two_state":
            right = bool(rng.random() >= 0.5)
            nxt = 1 if right else 0
            rew = 1.0 if current == 1 and right else 0.0
            ratio = 2.0 if right else 0.0
            current = nxt
        elif env.kind == "boyan":
            if current == 0:
                nxt = int(rng.integers(0, 13))
                rew = 1.5
                ratio = 1.0
            elif current == 1:
                nxt = 0
                rew = 1.5
                ratio = 1.0
            elif rng.random() < 0.4:
                nxt = current - 2
                rew = 1.0
                ratio = 1.25
            else:
                nxt = current - 1
                rew = 2.0
                ratio = 5.0 / 6.0
            current = nxt
        elif env.kind == "random_walk":
            right = bool(rng.random() >= 0.5)
            candidate = current + (1 if right else -1)
            ratio = 1.2 if right else 0.8
            terminal = candidate < 0 or candidate >= 5
            rew = 1.0 if candidate >= 5 else 0.0
            nxt = -1 if terminal else candidate
            reset_after[step] = terminal
            current = 2 if terminal else candidate
        else:
            raise KeyError(env.kind)
        next_state[step] = nxt
        reward[step] = rew
        rho[step] = ratio
    return Trajectory(state, next_state, reward, rho, reset_after)


def target_values(env: Environment) -> np.ndarray:
    return np.linalg.solve(
        np.eye(env.num_states) - env.gamma * env.p_pi_bootstrap,
        env.r_pi,
    )


def config_id(config: dict[str, Any]) -> str:
    parts = []
    for key in sorted(config):
        value = config[key]
        parts.append(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    return ";".join(parts)


def algorithm_configs(method: str, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tuning = protocol["tuning"]
    if method in {"TD", "ETD"} or method.startswith("TETD"):
        depth = int(method.removeprefix("TETD(").removesuffix(")")) if method.startswith("TETD") else None
        configs = [
            {"alpha": float(alpha), **({"depth": depth} if depth is not None else {})}
            for alpha in tuning["single_stepsize_grid"]
        ]
    elif method in {"GTD2", "TDC"}:
        configs = [
            {"alpha": float(alpha), "aux_alpha": float(alpha) * float(ratio)}
            for alpha in tuning["two_stepsize_main_grid"]
            for ratio in tuning["auxiliary_to_main_ratios"]
        ]
    elif method == "TDRC":
        configs = [
            {"alpha": float(alpha), "regularizer": float(regularizer)}
            for alpha in tuning["two_stepsize_main_grid"]
            for regularizer in tuning["tdrc_regularizers"]
        ]
    elif method == "RETD":
        configs = [
            {"alpha": float(alpha), "c": float(c_value)}
            for alpha in tuning["two_stepsize_main_grid"]
            for c_value in tuning["retd_c_values"]
        ]
    else:
        raise KeyError(method)
    budget = int(tuning["candidate_budget_per_algorithm_environment"])
    if len(configs) != budget:
        raise AssertionError(f"{method} has {len(configs)} candidates, expected {budget}")
    return configs


def truncated_followon(history: list[float], gamma: float, depth: int) -> float:
    value = 1.0
    product = 1.0
    for lag in range(1, min(depth, len(history)) + 1):
        product *= history[-lag]
        value += gamma**lag * product
    return value


def run_algorithm(
    env: Environment,
    coords: Coordinates,
    trajectory: Trajectory,
    method: str,
    config: dict[str, Any],
    protocol: dict[str, Any],
    keep_curve: bool,
) -> RunResult:
    started = time.perf_counter()
    alpha = float(config["alpha"])
    threshold = float(protocol["evaluation"]["main_divergence_threshold"])
    stride = int(protocol["evaluation"]["curve_stride"])
    tail_fraction = float(protocol["evaluation"]["tail_fraction"])
    theta = env.theta0.copy()
    weight = np.zeros_like(theta)
    omega = 0.0
    followon = 1.0
    history: list[float] = []
    true_v = target_values(env)
    eval_idx = env.evaluation_states
    zero_phi = np.zeros(theta.shape[0])
    checkpoint_steps = np.arange(0, env.steps + 1, stride, dtype=int)
    if checkpoint_steps[-1] != env.steps:
        checkpoint_steps = np.append(checkpoint_steps, env.steps)
    primary_curve = np.empty(len(checkpoint_steps))
    aligned_curve = np.empty(len(checkpoint_steps))
    raw_curve = np.empty(len(checkpoint_steps))
    tail_start_step = int(math.floor((1.0 - tail_fraction) * env.steps))
    tail_theta_sum = np.zeros_like(theta)
    tail_omega_sum = 0.0
    tail_count = 0
    divergence_step: int | None = None

    def metrics() -> tuple[float, float, float]:
        raw_value = env.phi @ theta
        if method == "RETD":
            recovered_value = raw_value + env.phi @ (coords.basis @ coords.u) * omega
            output_value = recovered_value
        else:
            recovered_value = raw_value
            output_value = raw_value
        primary = float(np.sqrt(np.mean((output_value[eval_idx] - true_v[eval_idx]) ** 2)))
        aligned = float(np.sqrt(np.mean((recovered_value[eval_idx] - coords.v_e[eval_idx]) ** 2)))
        raw = float(np.sqrt(np.mean((raw_value[eval_idx] - true_v[eval_idx]) ** 2)))
        return primary, aligned, raw

    primary_curve[0], aligned_curve[0], raw_curve[0] = metrics()
    checkpoint_index = 1
    for index in range(env.steps):
        state = int(trajectory.state[index])
        next_state = int(trajectory.next_state[index])
        phi = env.phi[state]
        phi_next = zero_phi if next_state < 0 else env.phi[next_state]
        reward = float(trajectory.reward[index])
        rho = float(trajectory.rho[index])
        delta = reward + env.gamma * float(phi_next @ theta) - float(phi @ theta)
        if method == "TD":
            theta += alpha * rho * delta * phi
        elif method == "GTD2":
            correction = float(phi @ weight)
            theta += alpha * rho * (phi - env.gamma * phi_next) * correction
            weight += float(config["aux_alpha"]) * (rho * delta - correction) * phi
        elif method == "TDC":
            correction = float(phi @ weight)
            theta += alpha * rho * (delta * phi - env.gamma * phi_next * correction)
            weight += float(config["aux_alpha"]) * (rho * delta - correction) * phi
        elif method == "TDRC":
            correction = float(phi @ weight)
            theta += alpha * rho * (delta * phi - env.gamma * phi_next * correction)
            weight += alpha * (
                (rho * delta - correction) * phi - float(config["regularizer"]) * weight
            )
        elif method == "ETD":
            theta += alpha * followon * rho * delta * phi
        elif method.startswith("TETD"):
            truncated = truncated_followon(history, env.gamma, int(config["depth"]))
            theta += alpha * truncated * rho * delta * phi
        elif method == "RETD":
            signal = followon * rho * delta
            old_omega = omega
            theta += alpha * (signal - old_omega) * phi
            omega += alpha * (signal - (1.0 + float(config["c"])) * old_omega)
        else:
            raise KeyError(method)

        if method in {"ETD", "RETD"}:
            followon = 1.0 + env.gamma * rho * followon
        if method.startswith("TETD"):
            history.append(rho)
            depth = int(config["depth"])
            if len(history) > depth:
                del history[: len(history) - depth]
        if bool(trajectory.reset_after[index]):
            followon = 1.0
            history.clear()

        completed_step = index + 1
        if completed_step >= tail_start_step:
            tail_theta_sum += theta
            tail_omega_sum += omega
            tail_count += 1
        if completed_step == checkpoint_steps[checkpoint_index]:
            primary, aligned, raw = metrics()
            values = np.concatenate([theta, weight, np.array([omega, primary, aligned, raw])])
            if not np.all(np.isfinite(values)) or primary > threshold:
                divergence_step = completed_step
                primary = threshold
                aligned = min(aligned, threshold) if math.isfinite(aligned) else threshold
                raw = min(raw, threshold) if math.isfinite(raw) else threshold
            primary_curve[checkpoint_index] = primary
            aligned_curve[checkpoint_index] = aligned
            raw_curve[checkpoint_index] = raw
            checkpoint_index += 1
            if divergence_step is not None:
                primary_curve[checkpoint_index:] = threshold
                aligned_curve[checkpoint_index:] = threshold
                raw_curve[checkpoint_index:] = threshold
                break

    if tail_count == 0:
        tail_theta = theta.copy()
        tail_omega = omega
    else:
        tail_theta = tail_theta_sum / tail_count
        tail_omega = tail_omega_sum / tail_count
    tail_checkpoint = checkpoint_steps >= tail_start_step
    result_steps = checkpoint_steps if keep_curve else np.array([0, env.steps], dtype=int)
    result_primary = primary_curve if keep_curve else primary_curve[[0, -1]]
    result_aligned = aligned_curve if keep_curve else aligned_curve[[0, -1]]
    result_raw = raw_curve if keep_curve else raw_curve[[0, -1]]
    return RunResult(
        steps=result_steps,
        primary_curve=result_primary,
        aligned_curve=result_aligned,
        raw_curve=result_raw,
        divergence_step=divergence_step,
        tail_primary=float(np.mean(primary_curve[tail_checkpoint])),
        tail_aligned=float(np.mean(aligned_curve[tail_checkpoint])),
        tail_raw=float(np.mean(raw_curve[tail_checkpoint])),
        tail_theta=tail_theta,
        tail_omega=float(tail_omega),
        final_theta=theta.copy(),
        final_omega=float(omega),
        runtime_seconds=time.perf_counter() - started,
    )


def validation_score(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    values = np.asarray([float(row["tail_primary"]) for row in rows])
    divergence = np.asarray([bool(row["diverged"]) for row in rows])
    divergence_fraction = float(np.mean(divergence))
    median = float(np.median(values))
    score = float(np.median(np.log10(np.maximum(values, EPS))) + 5.0 * divergence_fraction)
    return score, divergence_fraction, median


def select_configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_config.setdefault(str(row["config_id"]), []).append(row)
    candidates = []
    for identifier, subset in by_config.items():
        score, divergence_fraction, median = validation_score(subset)
        config = json.loads(str(subset[0]["config_json"]))
        candidates.append(
            {
                "config_id": identifier,
                "config": config,
                "score": score,
                "divergence_fraction": divergence_fraction,
                "median_tail_primary": median,
                "alpha": float(config["alpha"]),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["score"],
            item["divergence_fraction"],
            item["median_tail_primary"],
            item["alpha"],
            item["config_id"],
        )
    )
    return candidates[0]


def robust_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    output = {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
    }
    output["iqr"] = output["q75"] - output["q25"]
    return output


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    seed: int,
    replicates: int = 10000,
) -> tuple[float, float, float]:
    if left.shape != right.shape:
        raise ValueError("Paired arrays must have the same shape")
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    draws = difference[indices].mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def save_trajectory(path: Path, trajectory: Trajectory, env: Environment, seed: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_arrays(
        trajectory.state,
        trajectory.next_state,
        trajectory.reward,
        trajectory.rho,
        trajectory.reset_after,
    )
    np.savez_compressed(
        path,
        state=trajectory.state,
        next_state=trajectory.next_state,
        reward=trajectory.reward,
        rho=trajectory.rho,
        reset_after=trajectory.reset_after,
        environment=env.name,
        seed=seed,
        sha256=digest,
    )
    return digest


def environment_receipt(env: Environment, coords: Coordinates) -> dict[str, Any]:
    true_v = target_values(env)
    return {
        "environment": env.name,
        "kind": env.kind,
        "gamma": env.gamma,
        "steps": env.steps,
        "phi": env.phi,
        "theta0": env.theta0,
        "p_mu_current": env.p_mu_current,
        "p_pi_bootstrap": env.p_pi_bootstrap,
        "r_pi": env.r_pi,
        "evaluation_states": env.evaluation_states,
        "true_values": true_v,
        "d_mu": coords.d_mu,
        "emphatic_weights": coords.f,
        "A_reduced": coords.A,
        "b_reduced": coords.b,
        "q_reduced": coords.q,
        "A_inverse_q": coords.u,
        "q_A_inverse_q": coords.s,
        "etd_fixed_prediction": coords.v_e,
        "feature_rank": int(coords.basis.shape[1]),
        "condition_A": float(np.linalg.cond(coords.A)),
    }


def run_validation(
    root: Path,
    protocol: dict[str, Any],
    envs: dict[str, Environment],
    coords: dict[str, Coordinates],
    seeds: list[int],
    candidate_limit: int | None,
) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trajectory_cache: dict[tuple[str, int], Trajectory] = {}
    for env in envs.values():
        for seed in seeds:
            trajectory = generate_trajectory(env, seed)
            trajectory_cache[(env.name, seed)] = trajectory
            save_trajectory(
                root / "trajectories" / "validation" / env.name / f"seed_{seed}.npz",
                trajectory,
                env,
                seed,
            )
    for env in envs.values():
        for method in protocol["algorithms"]:
            configs = algorithm_configs(method, protocol)
            if candidate_limit is not None:
                configs = configs[:candidate_limit]
            for config in configs:
                identifier = config_id(config)
                for seed in seeds:
                    result = run_algorithm(
                        env,
                        coords[env.name],
                        trajectory_cache[(env.name, seed)],
                        method,
                        config,
                        protocol,
                        keep_curve=False,
                    )
                    rows.append(
                        {
                            "environment": env.name,
                            "method": method,
                            "config_id": identifier,
                            "config_json": json.dumps(config, sort_keys=True),
                            "seed": seed,
                            "tail_primary": result.tail_primary,
                            "tail_aligned": result.tail_aligned,
                            "tail_raw": result.tail_raw,
                            "diverged": result.divergence_step is not None,
                            "divergence_step": result.divergence_step,
                            "runtime_seconds": result.runtime_seconds,
                        }
                    )
    write_csv(root / "validation" / "validation_seed_metrics.csv", rows)
    selected: dict[str, dict[str, Any]] = {}
    selection_rows = []
    for env in envs.values():
        selected[env.name] = {}
        for method in protocol["algorithms"]:
            subset = [row for row in rows if row["environment"] == env.name and row["method"] == method]
            winner = select_configuration(subset)
            selected[env.name][method] = winner
            selection_rows.append(
                {
                    "environment": env.name,
                    "method": method,
                    "config_id": winner["config_id"],
                    "config_json": json.dumps(winner["config"], sort_keys=True),
                    "validation_score": winner["score"],
                    "divergence_fraction": winner["divergence_fraction"],
                    "median_tail_primary": winner["median_tail_primary"],
                }
            )
    write_csv(root / "validation" / "selected_configurations.csv", selection_rows)
    write_json(root / "validation" / "selected_configurations.json", selected)
    return selected


def run_test(
    root: Path,
    protocol: dict[str, Any],
    envs: dict[str, Environment],
    coords: dict[str, Coordinates],
    seeds: list[int],
    selected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    seed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    trajectory_cache: dict[tuple[str, int], Trajectory] = {}
    for env in envs.values():
        for seed in seeds:
            trajectory = generate_trajectory(env, seed)
            trajectory_cache[(env.name, seed)] = trajectory
            save_trajectory(
                root / "trajectories" / "test" / env.name / f"seed_{seed}.npz",
                trajectory,
                env,
                seed,
            )
    for env in envs.values():
        for method in protocol["algorithms"]:
            config = selected[env.name][method]["config"]
            for seed in seeds:
                result = run_algorithm(
                    env,
                    coords[env.name],
                    trajectory_cache[(env.name, seed)],
                    method,
                    config,
                    protocol,
                    keep_curve=True,
                )
                seed_rows.append(
                    {
                        "environment": env.name,
                        "method": method,
                        "config_id": config_id(config),
                        "seed": seed,
                        "tail_primary": result.tail_primary,
                        "tail_aligned": result.tail_aligned,
                        "tail_raw": result.tail_raw,
                        "diverged": result.divergence_step is not None,
                        "divergence_step": result.divergence_step,
                        "right_censored": result.divergence_step is None,
                        "runtime_seconds": result.runtime_seconds,
                    }
                )
                seed_root = root / "test" / "per_seed" / env.name / method.replace("/", "_")
                seed_root.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    seed_root / f"seed_{seed}.npz",
                    step=result.steps,
                    primary=result.primary_curve,
                    aligned=result.aligned_curve,
                    raw=result.raw_curve,
                    final_theta=result.final_theta,
                    final_omega=result.final_omega,
                    tail_theta=result.tail_theta,
                    tail_omega=result.tail_omega,
                )
                for step, primary, aligned, raw in zip(
                    result.steps,
                    result.primary_curve,
                    result.aligned_curve,
                    result.raw_curve,
                    strict=True,
                ):
                    curve_rows.append(
                        {
                            "environment": env.name,
                            "method": method,
                            "seed": seed,
                            "step": int(step),
                            "primary_rmsve": float(primary),
                            "aligned_rmsve": float(aligned),
                            "raw_rmsve": float(raw),
                        }
                    )
    write_csv(root / "test" / "test_seed_metrics.csv", seed_rows)
    write_csv(root / "test" / "test_curves_per_seed.csv", curve_rows)
    summary_rows: list[dict[str, Any]] = []
    for env in envs.values():
        for method in protocol["algorithms"]:
            subset = [row for row in seed_rows if row["environment"] == env.name and row["method"] == method]
            summary = robust_summary(float(row["tail_primary"]) for row in subset)
            summary_rows.append(
                {
                    "environment": env.name,
                    "method": method,
                    "seeds": len(subset),
                    "divergence_count": sum(bool(row["diverged"]) for row in subset),
                    **summary,
                }
            )
    write_csv(root / "test" / "robust_endpoint_summary.csv", summary_rows)
    aggregate_curve_rows: list[dict[str, Any]] = []
    for env in envs.values():
        for method in protocol["algorithms"]:
            subset = [row for row in curve_rows if row["environment"] == env.name and row["method"] == method]
            for step in sorted({int(row["step"]) for row in subset}):
                values = [float(row["primary_rmsve"]) for row in subset if int(row["step"]) == step]
                aggregate_curve_rows.append(
                    {
                        "environment": env.name,
                        "method": method,
                        "step": step,
                        **robust_summary(values),
                    }
                )
    write_csv(root / "test" / "aggregate_learning_curves.csv", aggregate_curve_rows)
    paired_rows: list[dict[str, Any]] = []
    for env_index, env in enumerate(envs.values()):
        retd = np.asarray(
            [
                float(row["tail_primary"])
                for row in seed_rows
                if row["environment"] == env.name and row["method"] == "RETD"
            ]
        )
        for method_index, method in enumerate(protocol["algorithms"]):
            if method == "RETD":
                continue
            comparator = np.asarray(
                [
                    float(row["tail_primary"])
                    for row in seed_rows
                    if row["environment"] == env.name and row["method"] == method
                ]
            )
            mean_delta, lower, upper = paired_bootstrap_interval(
                retd,
                comparator,
                seed=2026081300 + 100 * env_index + method_index,
            )
            paired_rows.append(
                {
                    "environment": env.name,
                    "left_method": "RETD",
                    "right_method": method,
                    "mean_paired_tail_difference": mean_delta,
                    "bootstrap_95_lower": lower,
                    "bootstrap_95_upper": upper,
                    "replicates": 10000,
                }
            )
    write_csv(root / "test" / "paired_bootstrap_summary.csv", paired_rows)
    survival_rows: list[dict[str, Any]] = []
    for env in envs.values():
        checkpoints = sorted({0, env.steps // 10, env.steps // 4, env.steps // 2, 3 * env.steps // 4, env.steps})
        for method in protocol["algorithms"]:
            subset = [row for row in seed_rows if row["environment"] == env.name and row["method"] == method]
            for checkpoint in checkpoints:
                survived = sum(
                    row["divergence_step"] is None or int(row["divergence_step"]) > checkpoint
                    for row in subset
                )
                survival_rows.append(
                    {
                        "environment": env.name,
                        "method": method,
                        "step": checkpoint,
                        "at_risk_initial": len(subset),
                        "survived": survived,
                        "survival_probability": survived / len(subset),
                    }
                )
    write_csv(root / "test" / "divergence_survival.csv", survival_rows)
    return seed_rows


def two_c_configs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    spec = protocol["two_c_recovery"]
    configs = [
        {"alpha": float(alpha), "c1": float(pair[0]), "c2": float(pair[1])}
        for alpha in spec["alpha_grid"]
        for pair in spec["c_pairs"]
    ]
    if len(configs) != int(spec["candidate_budget"]):
        raise AssertionError("Two-c candidate budget mismatch")
    return configs


def two_c_readout(
    first: RunResult,
    second: RunResult,
    threshold: float,
) -> tuple[np.ndarray | None, float]:
    denominator = first.tail_omega - second.tail_omega
    if not math.isfinite(denominator) or abs(denominator) < threshold:
        return None, denominator
    readout = (
        first.tail_omega * second.tail_theta - second.tail_omega * first.tail_theta
    ) / denominator
    if not np.all(np.isfinite(readout)):
        return None, denominator
    return readout, denominator


def run_two_c(
    root: Path,
    protocol: dict[str, Any],
    envs: dict[str, Environment],
    coords: dict[str, Coordinates],
    validation_seeds: list[int],
    test_seeds: list[int],
    candidate_limit: int | None,
) -> None:
    spec = protocol["two_c_recovery"]
    threshold = float(spec["absolute_denominator_threshold"])
    validation_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    configs = two_c_configs(protocol)
    if candidate_limit is not None:
        configs = configs[:candidate_limit]
    for env_name in spec["environments"]:
        env = envs[env_name]
        coord = coords[env_name]
        trajectories = {seed: generate_trajectory(env, seed) for seed in validation_seeds}
        for config in configs:
            for seed in validation_seeds:
                base = {"alpha": config["alpha"]}
                first = run_algorithm(
                    env,
                    coord,
                    trajectories[seed],
                    "RETD",
                    {**base, "c": config["c1"]},
                    protocol,
                    keep_curve=False,
                )
                second = run_algorithm(
                    env,
                    coord,
                    trajectories[seed],
                    "RETD",
                    {**base, "c": config["c2"]},
                    protocol,
                    keep_curve=False,
                )
                readout, denominator = two_c_readout(first, second, threshold)
                error = (
                    float(np.sqrt(np.mean((env.phi @ readout - coord.v_e) ** 2)))
                    if readout is not None
                    else float(protocol["evaluation"]["main_divergence_threshold"])
                )
                validation_rows.append(
                    {
                        "environment": env_name,
                        "config_id": config_id(config),
                        "config_json": json.dumps(config, sort_keys=True),
                        "seed": seed,
                        "readout_error": error,
                        "denominator": denominator,
                        "invalid": readout is None,
                        "branch_diverged": first.divergence_step is not None or second.divergence_step is not None,
                    }
                )
        candidates = []
        for config in configs:
            identifier = config_id(config)
            subset = [row for row in validation_rows if row["environment"] == env_name and row["config_id"] == identifier]
            errors = np.asarray([float(row["readout_error"]) for row in subset])
            invalid_fraction = float(np.mean([bool(row["invalid"]) for row in subset]))
            divergence_fraction = float(np.mean([bool(row["branch_diverged"]) for row in subset]))
            score = float(np.median(np.log10(np.maximum(errors, EPS))) + 5 * invalid_fraction + 5 * divergence_fraction)
            candidates.append(
                {
                    "config": config,
                    "config_id": identifier,
                    "score": score,
                    "invalid_fraction": invalid_fraction,
                    "divergence_fraction": divergence_fraction,
                    "median_error": float(np.median(errors)),
                }
            )
        candidates.sort(
            key=lambda item: (
                item["score"],
                item["invalid_fraction"],
                item["divergence_fraction"],
                item["median_error"],
                item["config_id"],
            )
        )
        selected[env_name] = candidates[0]
    write_csv(root / "two_c" / "validation_seed_metrics.csv", validation_rows)
    write_json(root / "two_c" / "selected_configurations.json", selected)

    test_rows: list[dict[str, Any]] = []
    for env_name in spec["environments"]:
        env = envs[env_name]
        coord = coords[env_name]
        config = selected[env_name]["config"]
        for seed in test_seeds:
            trajectory = generate_trajectory(env, seed)
            first = run_algorithm(
                env,
                coord,
                trajectory,
                "RETD",
                {"alpha": config["alpha"], "c": config["c1"]},
                protocol,
                keep_curve=False,
            )
            second = run_algorithm(
                env,
                coord,
                trajectory,
                "RETD",
                {"alpha": config["alpha"], "c": config["c2"]},
                protocol,
                keep_curve=False,
            )
            readout, denominator = two_c_readout(first, second, threshold)
            model_first = first.tail_theta + coord.basis @ coord.u * first.tail_omega
            model_second = second.tail_theta + coord.basis @ coord.u * second.tail_omega
            if readout is None:
                readout_error = math.nan
                model_gap = math.nan
                primary_error = math.nan
            else:
                readout_value = env.phi @ readout
                readout_error = float(np.sqrt(np.mean((readout_value - coord.v_e) ** 2)))
                model_gap = float(
                    0.5
                    * (
                        np.linalg.norm(readout - model_first)
                        + np.linalg.norm(readout - model_second)
                    )
                )
                primary_error = float(np.sqrt(np.mean((readout_value - target_values(env)) ** 2)))
            test_rows.append(
                {
                    "environment": env_name,
                    "seed": seed,
                    "alpha": config["alpha"],
                    "c1": config["c1"],
                    "c2": config["c2"],
                    "omega1": first.tail_omega,
                    "omega2": second.tail_omega,
                    "denominator": denominator,
                    "invalid": readout is None,
                    "branch_diverged": first.divergence_step is not None or second.divergence_step is not None,
                    "readout_etd_fixed_point_error": readout_error,
                    "readout_true_value_error": primary_error,
                    "readout_model_based_parameter_gap": model_gap,
                    "theta1_json": json.dumps(first.tail_theta.tolist()),
                    "theta2_json": json.dumps(second.tail_theta.tolist()),
                }
            )
    write_csv(root / "two_c" / "test_seed_metrics.csv", test_rows)
    summary_rows: list[dict[str, Any]] = []
    for env_name in spec["environments"]:
        subset = [row for row in test_rows if row["environment"] == env_name]
        valid = [row for row in subset if not bool(row["invalid"])]
        error_summary = robust_summary(
            float(row["readout_etd_fixed_point_error"]) for row in valid
        ) if valid else {key: math.nan for key in ["mean", "sample_std", "median", "q10", "q25", "q50", "q75", "q90", "iqr"]}
        summary_rows.append(
            {
                "environment": env_name,
                "test_seeds": len(subset),
                "valid_count": len(valid),
                "invalid_count": len(subset) - len(valid),
                "invalid_rate": (len(subset) - len(valid)) / len(subset),
                "branch_divergence_count": sum(bool(row["branch_diverged"]) for row in subset),
                **{f"readout_error_{key}": value for key, value in error_summary.items()},
            }
        )
    write_csv(root / "two_c" / "robust_summary.csv", summary_rows)


def write_manifest(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    output_root = Path(args.output_root).resolve()
    run_id = args.run_id or (
        f"formal_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    root = output_root / run_id
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    envs = build_environments(protocol)
    coords = {name: analysis_coordinates(env) for name, env in envs.items()}
    for name, env in envs.items():
        write_json(root / "environments" / f"{name}.json", environment_receipt(env, coords[name]))
    validation_seeds = [int(seed) for seed in protocol["randomness"]["validation_seeds"]]
    test_seeds = [int(seed) for seed in protocol["randomness"]["test_seeds"]]
    candidate_limit = None
    if args.mode == "smoke":
        validation_seeds = validation_seeds[:2]
        test_seeds = test_seeds[:2]
        candidate_limit = 2
    started = time.perf_counter()
    selected = run_validation(
        root,
        protocol,
        envs,
        coords,
        validation_seeds,
        candidate_limit,
    )
    run_test(root, protocol, envs, coords, test_seeds, selected)
    run_two_c(
        root,
        protocol,
        envs,
        coords,
        validation_seeds,
        test_seeds,
        candidate_limit,
    )
    write_json(
        root / "completion_receipt.json",
        {
            "status": "complete",
            "mode": args.mode,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256_file(protocol_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "validation_seeds": validation_seeds,
            "test_seeds": test_seeds,
            "candidate_limit": candidate_limit,
            "environment_count": len(envs),
            "algorithm_count": len(protocol["algorithms"]),
            "wall_seconds": time.perf_counter() - started,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    write_manifest(root)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(HERE / "results"))
    parser.add_argument("--run-id")
    return parser.parse_args()


if __name__ == "__main__":
    completed = run(parse_args())
    print(f"COMPLETED={completed}")
