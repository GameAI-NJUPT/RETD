"""Frozen finite-time log-growth simulation; no historical data dependency."""
from __future__ import annotations

from typing import Any
import numpy as np

try:
    from . import numerical_core as base
except ImportError:
    import numerical_core as base


def simulate_growth(
    env: Any,
    trajectory: Any,
    method: str,
    config: dict[str, float | int],
) -> dict[str, np.ndarray]:
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
    curve_steps = np.arange(0, base.STEPS + 1, base.CURVE_STRIDE, dtype=int)
    raw_mean = np.empty(len(curve_steps), dtype=np.float64)
    raw_std = np.empty(len(curve_steps), dtype=np.float64)
    growth_mean = np.full(len(curve_steps), np.nan, dtype=np.float64)
    growth_std = np.full(len(curve_steps), np.nan, dtype=np.float64)
    fraction_expanding = np.full(len(curve_steps), np.nan, dtype=np.float64)
    curve_index = 1

    if method == "TETD":
        depth = int(config["depth"])
        products = np.zeros((depth, runs), dtype=np.float64)
        gamma_powers = env.gamma ** np.arange(1, depth + 1)
    else:
        products = np.empty((0, runs), dtype=np.float64)
        gamma_powers = np.empty(0)

    initial = base.rmsve(env, theta)
    raw_mean[0] = float(np.mean(initial))
    raw_std[0] = float(np.std(initial, ddof=1))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(base.STEPS):
            states = trajectory.state[step]
            next_states = trajectory.next_state[step]
            rho = trajectory.rho[step]
            phi = env.phi[states]
            phi_next = env.phi[next_states]
            delta = (
                env.gamma * np.sum(phi_next * theta, axis=1)
                - np.sum(phi * theta, axis=1)
            )

            if method == "ETD":
                theta += alpha * (followon * rho * delta)[:, None] * phi
            elif method == "RETD":
                signal = followon * rho * delta
                old_omega = omega.copy()
                theta += alpha * (signal - old_omega)[:, None] * phi
                omega += alpha * (
                    signal - (1.0 + float(config["c"])) * old_omega
                )
            elif method == "GTD":
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
                    delta[:, None] * phi
                    - env.gamma * phi_next * correction[:, None]
                )
                auxiliary = old_auxiliary + float(config["aux_alpha"]) * (
                    rho * delta - correction
                )[:, None] * phi
            elif method == "TDRC":
                correction = np.sum(phi * auxiliary, axis=1)
                old_auxiliary = auxiliary.copy()
                theta += alpha * rho[:, None] * (
                    delta[:, None] * phi
                    - env.gamma * phi_next * correction[:, None]
                )
                auxiliary = old_auxiliary + alpha * (
                    (rho * delta - correction)[:, None] * phi
                    - float(config["regularizer"]) * old_auxiliary
                )
            elif method == "TETD":
                truncated = 1.0 + np.sum(
                    gamma_powers[:, None] * products, axis=0
                )
                theta += alpha * (truncated * rho * delta)[:, None] * phi
            else:
                raise KeyError(method)

            if method in {"ETD", "RETD"}:
                followon = 1.0 + env.gamma * rho * followon
            elif method == "TETD":
                if products.shape[0] > 1:
                    products[1:] = rho[None, :] * products[:-1].copy()
                products[0] = rho

            values = base.rmsve(env, theta)
            bad = (~np.isfinite(values)) | (values > base.DIV_THRESHOLD)
            diverged |= bad
            if np.any(bad):
                theta[bad] = 0.0
                auxiliary[bad] = 0.0
                omega[bad] = 0.0
                values[bad] = base.DIV_THRESHOLD
            if step + 1 >= base.TAIL_START:
                tail_sum += values
                tail_count += 1

            if (step + 1) % base.CURVE_STRIDE == 0:
                finite_growth = (
                    np.log(np.clip(values, 1.0e-300, base.DIV_THRESHOLD))
                    - np.log(initial)
                ) / float(step + 1)
                raw_mean[curve_index] = float(np.mean(values))
                raw_std[curve_index] = float(np.std(values, ddof=1))
                growth_mean[curve_index] = float(np.mean(finite_growth))
                growth_std[curve_index] = float(np.std(finite_growth, ddof=1))
                fraction_expanding[curve_index] = float(
                    np.mean(finite_growth > 0.0)
                )
                curve_index += 1

    final = base.rmsve(env, theta)
    final[diverged] = base.DIV_THRESHOLD
    return {
        "tail_rmsve": tail_sum / max(tail_count, 1),
        "final_rmsve": final,
        "diverged": diverged,
        "curve_steps": curve_steps,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "growth_mean": growth_mean,
        "growth_std": growth_std,
        "fraction_expanding": fraction_expanding,
    }
