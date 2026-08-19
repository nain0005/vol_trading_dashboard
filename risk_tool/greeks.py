"""Greeks derived from the Black-Scholes d1/d2, plus the raw N(d1)/N(d2)
values the strike-selection model needs directly.

Vega and gamma have the same formula for calls and puts; delta, theta, and
rho differ by sign and by which cumulative-normal term they use. All
include the continuous dividend yield q — dropping it is a common source of
silently-wrong deltas on dividend-paying names.
"""
from __future__ import annotations

import math

from scipy.stats import norm

from risk_tool.pricing import D1D2, OptionType, _check_option_type, compute_d1_d2

TRADING_DAYS_PER_YEAR = 365.0  # theta-per-day convention; use 252 for a trading-day convention instead


def delta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType) -> float:
    """dPrice/dS.

        Call: e^-qT * N(d1)
        Put:  e^-qT * (N(d1) - 1)  =  -e^-qT * N(-d1)

    Equals bare N(d1) only when q = 0 — the dividend discount factor is not
    optional and matters most for high-yield underlyings.
    """
    _check_option_type(option_type)
    d = compute_d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        return disc_q * norm.cdf(d.d1)
    return disc_q * (norm.cdf(d.d1) - 1.0)


def gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """d(delta)/dS — identical formula for calls and puts:

        e^-qT * phi(d1) / (S * sigma * sqrt(T))
    """
    d = compute_d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm.pdf(d.d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dPrice/dsigma, per 1.00 (100 vol points) of sigma — identical formula
    for calls and puts:

        S * e^-qT * phi(d1) * sqrt(T)

    Divide by 100 for the conventional "price change per 1 vol point" quote.
    """
    d = compute_d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm.pdf(d.d1) * math.sqrt(T)


def theta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType, per_day: bool = True) -> float:
    """dPrice/dt — time decay, expressed as the option's value change per
    unit of time passing (negative for a long option in the typical case).
    Annualized by the raw formula; divided by 365 when per_day=True.
    """
    _check_option_type(option_type)
    d = compute_d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    decay_term = -S * disc_q * norm.pdf(d.d1) * sigma / (2 * math.sqrt(T))
    if option_type == "call":
        raw = decay_term - r * K * disc_r * norm.cdf(d.d2) + q * S * disc_q * norm.cdf(d.d1)
    else:
        raw = decay_term + r * K * disc_r * norm.cdf(-d.d2) - q * S * disc_q * norm.cdf(-d.d1)

    return raw / TRADING_DAYS_PER_YEAR if per_day else raw


def rho(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType) -> float:
    """dPrice/dr, per 1.00 (100%) change in r. Divide by 100 for "per 1% rate move"."""
    _check_option_type(option_type)
    d = compute_d1_d2(S, K, T, r, q, sigma)
    disc_r = math.exp(-r * T)
    if option_type == "call":
        return K * T * disc_r * norm.cdf(d.d2)
    return -K * T * disc_r * norm.cdf(-d.d2)


def n_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> dict[str, float]:
    """Raw d1, d2, N(d1), N(d2) — exposed explicitly per spec.

    N(d2) is the risk-neutral ITM probability for a CALL (use N(-d2) for a
    put — see pricing.itm_probability). N(d1) is the delta-adjacent term,
    equal to call delta only when q = 0. Both are risk-neutral quantities,
    not real-world probabilities.
    """
    d: D1D2 = compute_d1_d2(S, K, T, r, q, sigma)
    return {"d1": d.d1, "d2": d.d2, "N_d1": norm.cdf(d.d1), "N_d2": norm.cdf(d.d2)}


def all_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType) -> dict[str, float]:
    """Convenience bundle used by strike_selection.py and the dashboard."""
    return {
        "delta": delta(S, K, T, r, q, sigma, option_type),
        "gamma": gamma(S, K, T, r, q, sigma),
        "vega": vega(S, K, T, r, q, sigma),
        "theta": theta(S, K, T, r, q, sigma, option_type),
        "rho": rho(S, K, T, r, q, sigma, option_type),
        **n_d1_d2(S, K, T, r, q, sigma),
    }
