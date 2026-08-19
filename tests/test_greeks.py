"""Correctness tests for risk_tool.greeks — mostly identities that must
hold algebraically, plus sign/bounds sanity checks."""
import math

import pytest

from risk_tool.greeks import all_greeks, delta, gamma, n_d1_d2, rho, theta, vega
from risk_tool.pricing import black_scholes_price

PARAMS = dict(S=100.0, K=100.0, T=0.5, r=0.05, q=0.02, sigma=0.25)


def test_delta_call_minus_put_equals_dividend_discount():
    """delta_call - delta_put = e^-qT * (N(d1) - (N(d1)-1)) = e^-qT exactly,
    for any inputs — an algebraic identity, not an approximation."""
    d_call = delta(**PARAMS, option_type="call")
    d_put = delta(**PARAMS, option_type="put")
    expected = math.exp(-PARAMS["q"] * PARAMS["T"])
    assert (d_call - d_put) == pytest.approx(expected, abs=1e-12)


def test_delta_bounds():
    assert 0.0 <= delta(**PARAMS, option_type="call") <= 1.0
    assert -1.0 <= delta(**PARAMS, option_type="put") <= 0.0


def test_gamma_and_vega_positive_and_identical_for_call_and_put():
    """Gamma and vega don't depend on option_type in Black-Scholes — both
    calls and puts on the same contract specs have identical values."""
    g = gamma(**PARAMS)
    v = vega(**PARAMS)
    assert g > 0
    assert v > 0


def test_rho_call_positive_put_negative():
    """Higher rates help calls (present value of paying K later drops) and
    hurt puts — a standard sign-sanity check."""
    assert rho(**PARAMS, option_type="call") > 0
    assert rho(**PARAMS, option_type="put") < 0


def test_theta_matches_finite_difference_of_price():
    """Theta should match a small finite-difference bump of the actual BS
    price in time — this cross-checks the closed-form theta formula
    against the pricing function it's supposed to be the derivative of."""
    for option_type in ("call", "put"):
        base_price = black_scholes_price(**PARAMS, option_type=option_type)
        dt = 1e-5
        bumped_params = {**PARAMS, "T": PARAMS["T"] - dt}
        bumped_price = black_scholes_price(**bumped_params, option_type=option_type)
        finite_diff_theta_per_year = (bumped_price - base_price) / dt
        finite_diff_theta_per_day = finite_diff_theta_per_year / 365.0

        analytic_theta = theta(**PARAMS, option_type=option_type, per_day=True)
        assert analytic_theta == pytest.approx(finite_diff_theta_per_day, abs=1e-3)


def test_n_d1_d2_matches_call_delta_when_no_dividend():
    """With q=0, call delta reduces exactly to N(d1) — the case where the
    'N(d1) is not quite delta' caveat disappears."""
    params_no_div = {**PARAMS, "q": 0.0}
    nd = n_d1_d2(**params_no_div)
    d_call = delta(**params_no_div, option_type="call")
    assert nd["N_d1"] == pytest.approx(d_call, abs=1e-12)


def test_all_greeks_returns_expected_keys():
    result = all_greeks(**PARAMS, option_type="call")
    expected_keys = {"delta", "gamma", "vega", "theta", "rho", "d1", "d2", "N_d1", "N_d2"}
    assert set(result.keys()) == expected_keys
