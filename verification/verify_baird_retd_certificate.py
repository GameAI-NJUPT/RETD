from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp


sys.set_int_max_str_digits(0)


HERE = Path(__file__).resolve().parent
DEFAULT_CANDIDATE = (
    HERE
    / "certificates"
    / "baird_retd_alpha_0p01_c_0p5_candidate_20260810.json"
)
DEFAULT_OUTPUT = (
    HERE
    / "outputs"
    / "baird_retd_alpha_0p01_c_0p5_exact_verification_20260810.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rational(value: str | int) -> sp.Rational:
    return sp.Rational(str(value))


def fraction(value: sp.Rational) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def exact_summary(value: sp.Rational | Fraction) -> dict[str, str | int]:
    if isinstance(value, sp.Rational):
        numerator = int(value.p)
        denominator = int(value.q)
    else:
        numerator = value.numerator
        denominator = value.denominator
    representation = f"{numerator}/{denominator}".encode("ascii")
    return {
        "sign": 1 if numerator > 0 else (-1 if numerator < 0 else 0),
        "numerator_digits": len(str(abs(numerator))),
        "denominator_digits": len(str(denominator)),
        "sha256": hashlib.sha256(representation).hexdigest(),
    }


def gram() -> sp.Matrix:
    matrix = sp.ones(7, 7)
    for index in range(7):
        matrix[index, index] = 5
    for index in range(6):
        matrix[index, 6] = 2
        matrix[6, index] = 2
    return matrix


def sample_factor(
    state: int,
    next_state: int,
    xfr: sp.Rational,
    alpha: sp.Rational,
    c_value: sp.Rational,
    gamma: sp.Rational,
) -> sp.Matrix:
    basis = sp.eye(7)
    k_state = gram()[:, state]
    bellman = basis[:, state] - gamma * basis[:, next_state]
    block = sp.zeros(8, 8)
    block[:7, :7] = xfr * k_state * bellman.T
    block[:7, 7] = k_state
    block[7, :7] = xfr * bellman.T
    block[7, 7] = 1 + c_value
    return sp.eye(8) - alpha * block


def cycle_matrix(
    upper: int,
    length: int,
    alpha: sp.Rational,
    c_value: sp.Rational,
    gamma: sp.Rational,
) -> sp.Matrix:
    if length == 0:
        return sample_factor(upper, upper, sp.Rational(0), alpha, c_value, gamma)
    product = sample_factor(upper, 6, sp.Rational(7), alpha, c_value, gamma)
    followon = 1 + 7 * gamma
    for _ in range(1, length):
        product = sample_factor(6, 6, 7 * followon, alpha, c_value, gamma) * product
        followon = 1 + 7 * gamma * followon
    return sample_factor(6, 0, sp.Rational(0), alpha, c_value, gamma) * product


def symmetric_matrix(parameters: dict[str, Any]) -> sp.Matrix:
    matrix = sp.zeros(8, 8)
    diagonal = rational(parameters["upper_diagonal"])
    off_diagonal = rational(parameters["upper_off_diagonal"])
    for row in range(6):
        for column in range(6):
            matrix[row, column] = diagonal if row == column else off_diagonal
        matrix[row, 6] = matrix[6, row] = rational(parameters["upper_lower"])
        matrix[row, 7] = matrix[7, row] = rational(parameters["upper_omega"])
    matrix[6, 6] = rational(parameters["lower_lower"])
    matrix[6, 7] = matrix[7, 6] = rational(parameters["lower_omega"])
    matrix[7, 7] = rational(parameters["omega_omega"])
    return matrix


def symmetry_positive_definite(matrix: sp.Matrix, label: str) -> dict[str, dict[str, str | int]]:
    if matrix != matrix.T:
        raise AssertionError(f"{label} is not symmetric")
    reference_diagonal = matrix[0, 0]
    reference_off = matrix[0, 1]
    reference_lower = matrix[0, 6]
    reference_omega = matrix[0, 7]
    for row in range(6):
        for column in range(6):
            expected = reference_diagonal if row == column else reference_off
            if matrix[row, column] != expected:
                raise AssertionError(f"{label} breaks upper-state permutation symmetry")
        if matrix[row, 6] != reference_lower or matrix[row, 7] != reference_omega:
            raise AssertionError(f"{label} breaks upper-state coupling symmetry")
    contrast = sp.factor(reference_diagonal - reference_off)
    reduced = sp.Matrix(
        [
            [6 * (reference_diagonal + 5 * reference_off), 6 * reference_lower, 6 * reference_omega],
            [6 * reference_lower, matrix[6, 6], matrix[6, 7]],
            [6 * reference_omega, matrix[7, 6], matrix[7, 7]],
        ]
    )
    minors = [sp.factor(reduced[:size, :size].det()) for size in range(1, 4)]
    if contrast <= 0 or any(value <= 0 for value in minors):
        raise AssertionError(f"{label} is not positive definite")
    return {
        "contrast_eigenvalue": exact_summary(contrast),
        "reduced_leading_minor_1": exact_summary(minors[0]),
        "reduced_leading_minor_2": exact_summary(minors[1]),
        "reduced_leading_minor_3": exact_summary(minors[2]),
    }


def log_unit_interval(value: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    if not (Fraction(1) <= value <= Fraction(2)):
        raise ValueError(value)
    z_value = (value - 1) / (value + 1)
    total = Fraction(0)
    for index in range(terms):
        power = 2 * index + 1
        total += z_value**power / power
    lower = 2 * total
    next_power = 2 * terms + 1
    remainder = 2 * z_value**next_power / (next_power * (1 - z_value**2))
    return lower, lower + remainder


def log_interval(value: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    if value <= 0:
        raise ValueError("log input must be positive")
    exponent = 0
    scaled = value
    while scaled < 1:
        scaled *= 2
        exponent -= 1
    while scaled >= 2:
        scaled /= 2
        exponent += 1
    unit_lower, unit_upper = log_unit_interval(scaled, terms)
    two_lower, two_upper = log_unit_interval(Fraction(2), terms)
    if exponent >= 0:
        return (
            unit_lower + exponent * two_lower,
            unit_upper + exponent * two_upper,
        )
    return (
        unit_lower + exponent * two_upper,
        unit_upper + exponent * two_lower,
    )


def decimal_string(value: Fraction, digits: int = 18) -> str:
    return f"{float(value):.{digits}g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only exact verifier for the frozen Baird RETD certificate."
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite verification output: {args.output}")
    with args.candidate.open("r", encoding="utf-8") as handle:
        candidate = json.load(handle)
    if candidate.get("stage") != "fit_output_unverified":
        raise ValueError("Verifier accepts only a frozen, unverified fit output")

    problem = candidate["problem"]
    settings = candidate["exact_verifier_settings"]
    alpha = rational(problem["alpha"])
    c_value = rational(problem["c"])
    gamma = rational(problem["gamma"])
    solid_probability = rational(problem["solid_probability"])
    lmax = int(problem["finite_max_run_length"])
    epsilon = rational(problem["claimed_physical_time_margin"])
    terms = int(settings["log_series_terms"])
    p_lower = rational(settings["p_lower_bound"])
    p_upper = rational(settings["p_upper_bound"])
    slack = rational(settings["generalized_eigenvalue_slack"])

    p_matrix = symmetric_matrix(candidate["quadratic_form"])
    p_proof = symmetry_positive_definite(p_matrix, "P")
    p_lower_proof = symmetry_positive_definite(p_matrix - p_lower * sp.eye(8), "P-p_lower I")
    p_upper_proof = symmetry_positive_definite(p_upper * sp.eye(8) - p_matrix, "p_upper I-P")

    direction_data = candidate["tangent_direction"]
    direction = sp.Matrix(
        [rational(direction_data["common_upper"])] * 6
        + [rational(direction_data["lower"]), rational(direction_data["omega"])]
    )
    direction_denominator = sp.factor((direction.T * p_matrix * direction)[0])
    if direction_denominator <= 0:
        raise AssertionError("Invalid tangent direction")

    total_matrix = sp.zeros(8, 8)
    finite_log_upper = Fraction(0)
    finite_records: list[dict[str, str | int]] = []
    finish_probability = 1 - solid_probability
    for length in range(lmax + 1):
        probability = finish_probability * solid_probability**length
        qbar = sp.zeros(8, 8)
        for upper in range(6):
            cycle = cycle_matrix(upper, length, alpha, c_value, gamma)
            qbar += cycle.T * p_matrix * cycle / 6
        beta = sp.factor((direction.T * qbar * direction)[0] / direction_denominator)
        if beta <= 0:
            raise AssertionError(f"Nonpositive tangent at L={length}")
        total_matrix += probability * qbar / beta
        _, beta_log_upper = log_interval(fraction(beta), terms)
        finite_log_upper += fraction(probability) * (beta_log_upper - 1)
        finite_records.append(
            {
                "run_length": length,
                "probability": str(probability),
                "beta_decimal": decimal_string(fraction(beta)),
                "beta_exact": exact_summary(beta),
                "log_beta_upper_decimal": decimal_string(beta_log_upper),
                "log_beta_upper_exact": exact_summary(beta_log_upper),
            }
        )

    finite_mass = 1 - solid_probability ** (lmax + 1)
    generalized_bound = finite_mass + slack
    matrix_gap = sp.factor(generalized_bound) * p_matrix - total_matrix
    matrix_gap_proof = symmetry_positive_definite(matrix_gap, "lambda P-S")
    finite_upper = (finite_log_upper + fraction(generalized_bound)) / 2

    # Analytic tail.  Every solid factor has norm at most (357/200) F_k,
    # every dashed factor at most 357/200, and
    # F_k <= (693/593)(693/100)^k.  The exact spectral enclosure
    # p_lower I < P < p_upper I converts this to the P norm.
    first_tail_length = lmax + 1
    tail_probability = fraction(solid_probability**first_tail_length)
    tail_l = tail_probability * (Fraction(first_tail_length) + Fraction(1, 6))
    tail_l_falling = tail_probability * (
        Fraction(first_tail_length * (first_tail_length - 1))
        + Fraction(first_tail_length, 3)
        + Fraction(1, 18)
    )
    _, log_a = log_interval(Fraction(357, 200), terms)
    _, log_b = log_interval(Fraction(693, 593), terms)
    _, log_q = log_interval(Fraction(693, 100), terms)
    _, log_kappa = log_interval(fraction(p_upper / p_lower), terms)
    tail_upper = (
        (tail_l + tail_probability) * log_a
        + tail_l * log_b
        + tail_l_falling * log_q / 2
        + tail_probability * log_kappa / 2
    )

    expected_cycle_length = Fraction(7, 6)
    final_upper = finite_upper + tail_upper + fraction(epsilon) * expected_cycle_length
    if final_upper >= 0:
        raise AssertionError(f"Certificate failed: final upper bound {final_upper}")

    output = {
        "schema_version": 1,
        "status": "certified",
        "claim": "RETD top physical-time Lyapunov exponent <= -1/10000",
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256(Path(__file__)),
        "arithmetic": "exact rational matrices and rational log-series enclosures",
        "problem": problem,
        "p_positive_definite": p_proof,
        "p_lower_enclosure": p_lower_proof,
        "p_upper_enclosure": p_upper_proof,
        "finite_records": finite_records,
        "finite_mass": str(finite_mass),
        "generalized_eigenvalue_upper": str(generalized_bound),
        "matrix_gap_positive_definite": matrix_gap_proof,
        "finite_projective_drift_upper_exact": exact_summary(finite_upper),
        "finite_projective_drift_upper_decimal": decimal_string(finite_upper),
        "tail_projective_drift_upper_exact": exact_summary(tail_upper),
        "tail_projective_drift_upper_decimal": decimal_string(tail_upper),
        "epsilon_times_expected_cycle_length": str(fraction(epsilon) * expected_cycle_length),
        "final_certificate_upper_exact": exact_summary(final_upper),
        "final_certificate_upper_decimal": decimal_string(final_upper),
        "log_series_terms": terms,
        "sympy_version": sp.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: output[key] for key in ["status", "claim", "finite_projective_drift_upper_decimal", "tail_projective_drift_upper_decimal", "final_certificate_upper_decimal"]}, indent=2))


if __name__ == "__main__":
    main()
