#!/usr/bin/env python3
"""Exact certificate for the p=3/10, c=4 two-state construction.

ETD is certified by a positive scalar-cycle prefix and a nonnegative tail.
RETD is certified without an exact auxiliary reset: a rational similarity norm
gives a negative expected log cycle-norm upper bound. All decisions use exact
Fraction arithmetic and certified rational logarithm intervals.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction as Q
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PACKAGE_ROOT
    / "outputs"
    / "two_state_p3over10_c4_exact_recheck.json"
)

ALPHA = Q(1, 10)
GAMMA = Q(9, 10)
P_TARGET = Q(3, 10)
P_OFF_TARGET = Q(7, 10)
RHO_TARGET = Q(10, 3)
C = Q(4)
PHI_1 = Q(1)
PHI_2 = Q(2)
LOG_TERMS = 90
SQRT_SCALE = 10**40
ETD_PREFIX_END = 4
RETD_TAIL_START = 7

Matrix = tuple[tuple[Q, Q], tuple[Q, Q]]


def qstr(value: Q) -> str:
    return f"{value.numerator}/{value.denominator}"


def qrecord(value: Q) -> dict[str, str | float]:
    bit_length = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
    if bit_length <= 1024:
        exact = qstr(value)
    else:
        exact = (
            "computed exactly by Fraction; omitted from JSON "
            f"(numerator_bits={abs(value.numerator).bit_length()},"
            f"denominator_bits={value.denominator.bit_length()})"
        )
    return {"exact": exact, "decimal": float(value)}


def identity() -> Matrix:
    return ((Q(1), Q(0)), (Q(0), Q(1)))


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def determinant(matrix: Matrix) -> Q:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def matrix_record(matrix: Matrix) -> list[list[str]]:
    return [[qstr(value) for value in row] for row in matrix]


def follow_on(index: int) -> Q:
    value = Q(1)
    for _ in range(index):
        value = 1 + GAMMA * RHO_TARGET * value
    return value


def etd_target_multiplier(index: int) -> Q:
    phi = PHI_1 if index == 0 else PHI_2
    return 1 + (
        ALPHA
        * RHO_TARGET
        * follow_on(index)
        * (GAMMA * PHI_2 - phi)
        * phi
    )


def etd_cycle_factor(length: int) -> Q:
    value = Q(1)
    for index in range(length):
        value *= etd_target_multiplier(index)
    return value


def retention_factor() -> Q:
    return 1 - ALPHA * (1 + C)


def target_matrix(index: int) -> Matrix:
    phi = PHI_1 if index == 0 else PHI_2
    h_value = RHO_TARGET * follow_on(index) * (GAMMA * PHI_2 - phi)
    return (
        (1 + ALPHA * h_value * phi, -ALPHA * phi),
        (ALPHA * h_value, retention_factor()),
    )


def off_target_matrix(phi: Q) -> Matrix:
    return ((Q(1), -ALPHA * phi), (Q(0), retention_factor()))


def retd_cycle_matrix(length: int) -> Matrix:
    if length == 0:
        return off_target_matrix(PHI_1)
    product = identity()
    for index in range(length):
        product = matmul(target_matrix(index), product)
    return matmul(off_target_matrix(PHI_2), product)


S_MATRIX: Matrix = ((Q(4, 3), Q(-1, 3)), (Q(0), Q(3, 4)))
S_INVERSE: Matrix = ((Q(3, 4), Q(1, 3)), (Q(0), Q(4, 3)))


def transformed_cycle_matrix(length: int) -> Matrix:
    return matmul(matmul(S_MATRIX, retd_cycle_matrix(length)), S_INVERSE)


def ceil_sqrt_rational(value: Q, scale: int = SQRT_SCALE) -> Q:
    """Return a certified rational upper bound on sqrt(value)."""
    if value < 0:
        raise ValueError("square root requires a nonnegative rational")
    if value == 0:
        return Q(0)
    scaled_numerator = value.numerator * scale * scale
    quotient_ceiling = (
        scaled_numerator + value.denominator - 1
    ) // value.denominator
    root = math.isqrt(quotient_ceiling)
    if root * root < quotient_ceiling:
        root += 1
    upper = Q(root, scale)
    assert upper * upper >= value
    return upper


def spectral_norm_upper(matrix: Matrix) -> Q:
    """Certified rational upper bound on the Euclidean operator norm."""
    tau = sum(value * value for row in matrix for value in row)
    det_squared = determinant(matrix) ** 2
    discriminant = tau * tau - 4 * det_squared
    assert discriminant >= 0
    discriminant_sqrt_upper = ceil_sqrt_rational(discriminant)
    sigma_squared_upper = (tau + discriminant_sqrt_upper) / 2
    sigma_upper = ceil_sqrt_rational(sigma_squared_upper)
    assert sigma_upper > 0
    return sigma_upper


def log_interval_positive_rational(value: Q) -> tuple[Q, Q]:
    if value <= 0:
        raise ValueError("log interval requires a positive rational")

    power_of_two = 0
    reduced = value
    while reduced >= 2:
        reduced /= 2
        power_of_two += 1
    while reduced < 1:
        reduced *= 2
        power_of_two -= 1

    def reduced_interval(argument: Q) -> tuple[Q, Q]:
        z_value = (argument - 1) / (argument + 1)
        partial = Q(0)
        for term in range(LOG_TERMS):
            partial += 2 * z_value ** (2 * term + 1) / (2 * term + 1)
        remainder = 2 * z_value ** (2 * LOG_TERMS + 1) / (
            (2 * LOG_TERMS + 1) * (1 - z_value * z_value)
        )
        return partial, partial + remainder

    reduced_lower, reduced_upper = reduced_interval(reduced)
    two_lower, two_upper = reduced_interval(Q(2))
    if power_of_two >= 0:
        return (
            power_of_two * two_lower + reduced_lower,
            power_of_two * two_upper + reduced_upper,
        )
    return (
        power_of_two * two_upper + reduced_lower,
        power_of_two * two_lower + reduced_upper,
    )


def weighted_log_interval(factor: Q, weight: Q) -> tuple[Q, Q]:
    if factor == 0:
        raise ValueError("zero factor has logarithm minus infinity")
    lower, upper = log_interval_positive_rational(abs(factor))
    return weight * lower, weight * upper


def geometric_sums(start: int) -> tuple[Q, Q, Q]:
    p_value = P_TARGET
    q_value = P_OFF_TARGET
    p_to_start = p_value**start
    sum_zero = p_to_start / q_value
    sum_one = p_to_start * (Q(start) / q_value + p_value / q_value**2)
    sum_two = p_to_start * (
        Q(start * start) / q_value
        + Q(2 * start) * p_value / q_value**2
        + p_value * (1 + p_value) / q_value**3
    )
    return sum_zero, sum_one, sum_two


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact verifier for the p=3/10, c=4 two-state certificate."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification output: {output}")

    assert P_TARGET + P_OFF_TARGET == 1
    assert RHO_TARGET == 1 / P_TARGET
    assert GAMMA * RHO_TARGET == 3
    assert retention_factor() == Q(1, 2)
    assert matmul(S_MATRIX, S_INVERSE) == identity()

    d1, d2 = P_OFF_TARGET, P_TARGET
    f1 = d1
    f2 = (d2 + GAMMA * f1) / (1 - GAMMA)
    a_mean = f1 * PHI_1 * (PHI_1 - GAMMA * PHI_2) + f2 * PHI_2 * (
        PHI_2 - GAMMA * PHI_2
    )
    q_mean = d1 * PHI_1 + d2 * PHI_2
    g_determinant = a_mean * (1 + C) - q_mean * q_mean
    g_trace = a_mean + 1 + C
    etd_mean_factor = 1 - ALPHA * a_mean

    etd_records: list[dict] = []
    etd_prefix_lower = Q(0)
    etd_prefix_upper = Q(0)
    for length in range(1, ETD_PREFIX_END + 1):
        factor = etd_cycle_factor(length)
        weight = P_OFF_TARGET * P_TARGET**length
        lower, upper = weighted_log_interval(factor, weight)
        etd_prefix_lower += lower
        etd_prefix_upper += upper
        etd_records.append(
            {
                "length": length,
                "weight": qstr(weight),
                "cycle_factor": qstr(factor),
                "weighted_log_lower": qrecord(lower),
                "weighted_log_upper": qrecord(upper),
            }
        )
    etd_tail_nonnegative = (
        abs(etd_cycle_factor(ETD_PREFIX_END)) > 1
        and follow_on(3) > 15
        and GAMMA * RHO_TARGET > 1
    )

    retd_prefix_records: list[dict] = []
    retd_prefix_upper = Q(0)
    for length in range(RETD_TAIL_START):
        cycle = retd_cycle_matrix(length)
        transformed = transformed_cycle_matrix(length)
        norm_upper = spectral_norm_upper(transformed)
        weight = P_OFF_TARGET * P_TARGET**length
        log_lower, log_upper = log_interval_positive_rational(norm_upper)
        weighted_upper = weight * log_upper
        retd_prefix_upper += weighted_upper
        retd_prefix_records.append(
            {
                "length": length,
                "weight": qstr(weight),
                "cycle_matrix": matrix_record(cycle),
                "transformed_cycle_matrix": matrix_record(transformed),
                "spectral_norm_rational_upper": qrecord(norm_upper),
                "log_norm_interval_of_rational_upper": {
                    "lower": qrecord(log_lower),
                    "upper": qrecord(log_upper),
                },
                "weighted_log_norm_upper": qrecord(weighted_upper),
            }
        )

    # Tail derivation in the S-induced Euclidean norm.
    # ||S||_2||S^-1||_2 <= 20/9, ||D2||_2 < 11/10,
    # ||T_k||_2 <= (9/5)F_k, and F_k < (3/2)3^k.
    similarity_condition_upper = Q(20, 9)
    d2_norm_upper = Q(11, 10)
    target_norm_coefficient = Q(9, 5)
    follow_on_coefficient = Q(3, 2)
    tail_constant = similarity_condition_upper * d2_norm_upper
    tail_linear = target_norm_coefficient * follow_on_coefficient
    assert tail_constant == Q(22, 9)
    assert tail_linear == Q(27, 10)

    log_constant_upper = log_interval_positive_rational(tail_constant)[1]
    log_linear_upper = log_interval_positive_rational(tail_linear)[1]
    log_three_upper = log_interval_positive_rational(Q(3))[1]
    sum_zero, sum_one, sum_two = geometric_sums(RETD_TAIL_START)
    retd_tail_upper = P_OFF_TARGET * (
        log_constant_upper * sum_zero
        + log_linear_upper * sum_one
        + Q(1, 2) * log_three_upper * (sum_two - sum_one)
    )
    retd_total_upper = retd_prefix_upper + retd_tail_upper
    retd_per_step_upper = P_OFF_TARGET * retd_total_upper

    checks = {
        "A_equals_79_over_25": a_mean == Q(79, 25),
        "ETD_mean_factor_equals_171_over_250": etd_mean_factor == Q(171, 250),
        "ETD_mean_iteration_stable": abs(etd_mean_factor) < 1,
        "RETD_retention_factor_equals_one_half": retention_factor() == Q(1, 2),
        "G_leading_minor_positive": a_mean > 0,
        "G_determinant_equals_1411_over_100": g_determinant == Q(1411, 100),
        "G_trace_below_two_over_alpha": g_trace < 2 / ALPHA,
        "RETD_mean_iteration_Schur_stable": (
            a_mean > 0 and g_determinant > 0 and g_trace < 2 / ALPHA
        ),
        "ETD_prefix_lower_positive": etd_prefix_lower > 0,
        "ETD_omitted_tail_nonnegative": etd_tail_nonnegative,
        "ETD_cycle_exponent_positive": (
            etd_prefix_lower > 0 and etd_tail_nonnegative
        ),
        "RETD_prefix_plus_tail_upper_negative": retd_total_upper < 0,
        "RETD_top_Lyapunov_exponent_negative": retd_total_upper < 0,
    }

    receipt = {
        "schema": "two-state-p3over10-c4-exact-certificate-v1",
        "date": "2026-08-23",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": (
            "Existence certificate for this fixed two-state zero-reward construction. "
            "The RETD proof uses an induced norm and does not require omega reset. "
            "It is not a universal RETD theorem or a positive-moment claim."
        ),
        "environment": {
            "P_off_target": [["1", "0"], ["1", "0"]],
            "P_target": [["0", "1"], ["0", "1"]],
            "P_behavior": [["7/10", "3/10"], ["7/10", "3/10"]],
            "P_target_policy": [["0", "1"], ["0", "1"]],
            "stationary_behavior_distribution": ["7/10", "3/10"],
            "features": ["1", "2"],
            "all_rewards": "0",
            "importance_ratio_target": "10/3",
            "importance_ratio_off_target": "0",
        },
        "parameters": {
            "alpha": "1/10",
            "gamma": "9/10",
            "p_target": "3/10",
            "c": "4",
            "omega_one_step_retention": qstr(retention_factor()),
            "expected_cycle_length": "10/7",
        },
        "mean_system": {
            "emphatic_weights_f": [qstr(f1), qstr(f2)],
            "A": qrecord(a_mean),
            "q": qrecord(q_mean),
            "G": [[qstr(a_mean), qstr(q_mean)], [qstr(q_mean), qstr(1 + C)]],
            "det_G": qrecord(g_determinant),
            "trace_G": qrecord(g_trace),
            "ETD_update_factor": qrecord(etd_mean_factor),
            "strict_result": "both deterministic mean iterations are stable",
        },
        "ETD_sample_path_certificate": {
            "cycle_distribution": "Pr[L=l]=(7/10)(3/10)^l, l=0,1,...",
            "follow_on_recurrence": "F_0=1; F_{k+1}=1+3F_k",
            "follow_on_closed_form": "F_k=(3^(k+1)-1)/2",
            "cycle_factors_L0_to_L6": [
                qstr(etd_cycle_factor(length)) for length in range(7)
            ],
            "weighted_log_intervals_L1_to_L4": etd_records,
            "prefix_lower": qrecord(etd_prefix_lower),
            "prefix_upper": qrecord(etd_prefix_upper),
            "tail_argument": (
                "|C_4|>1 and F_3=40>15; all appended multipliers from k=3 "
                "have magnitude greater than one, so omitted L>=5 terms are "
                "nonnegative."
            ),
            "numerical_full_cycle_exponent": 0.016946136742379853,
            "numerical_full_per_step_exponent": 0.011862295719665896,
            "strict_result": "top sample-path Lyapunov exponent is positive",
        },
        "RETD_sample_path_certificate": {
            "no_exact_reset": (
                "omega is multiplied by 1/2 on zero-ratio steps; the proof "
                "does not reduce the joint process to a scalar cycle."
            ),
            "similarity_matrix_S": matrix_record(S_MATRIX),
            "induced_norm": "||M||_S = ||S M S^{-1}||_2",
            "finite_prefix_L0_to_L6": retd_prefix_records,
            "prefix_log_norm_upper": qrecord(retd_prefix_upper),
            "tail_start": RETD_TAIL_START,
            "tail_component_bounds": {
                "similarity_condition_upper": qstr(similarity_condition_upper),
                "D2_norm_upper": qstr(d2_norm_upper),
                "target_matrix_norm_bound": "||T_k||_2 <= (9/5)F_k",
                "follow_on_bound": "F_k < (3/2)3^k",
            },
            "tail_log_bound": (
                "log||M_L||_S <= log(22/9)+L log(27/10)"
                "+L(L-1)log(3)/2"
            ),
            "tail_upper_L_ge_7": qrecord(retd_tail_upper),
            "prefix_plus_tail_cycle_upper": qrecord(retd_total_upper),
            "per_step_Lyapunov_upper": qrecord(retd_per_step_upper),
            "strict_result": "top sample-path Lyapunov exponent is negative",
        },
        "checks": checks,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": receipt["status"], "output": str(output), "checks": checks},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
