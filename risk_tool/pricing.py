"""Black-Scholes pricing, the CRR binomial tree for American options, and
implied-volatility inversion.

Every function here is pure and stateless — no I/O, no Streamlit, no config
object required — so they're directly unit-testable and reusable from a
CLI, a dashboard, or live IBKR data alike.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _check_option_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


@dataclass(frozen=True)
class D1D2:
    d1: float
    d2: float


def compute_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> D1D2:
    """The two Black-Scholes standardized distances.

        d1 = (ln(S/K) + (r - q + sigma^2/2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    d2 is the risk-neutral z-score of the terminal log-price relative to the
    strike: N(d2) is the risk-neutral probability the option finishes ITM
    for a call, N(-d2) for a put. d1 is the analogous quantity that appears
    in delta once you account for the S*e^-qT drift term — see greeks.py.
    Both are RISK-NEUTRAL constructs, not real-world return distributions.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return D1D2(d1=d1, d2=d2)


def black_scholes_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType) -> float:
    """European option price under Black-Scholes-Merton with continuous
    dividend yield q.

        Call = S*e^-qT*N(d1) - K*e^-rT*N(d2)
        Put  = K*e^-rT*N(-d2) - S*e^-qT*N(-d1)

    Assumes: constant sigma and r over the option's life, continuous
    trading, no transaction costs, log-normal terminal price distribution.
    None of these hold exactly in real markets — see README for what that
    means for this tool's outputs.
    """
    _check_option_type(option_type)
    d = compute_d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d.d1) - K * disc_r * norm.cdf(d.d2)
    return K * disc_r * norm.cdf(-d.d2) - S * disc_q * norm.cdf(-d.d1)


def itm_probability(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType) -> float:
    """Risk-neutral probability of finishing in-the-money: N(d2) for a call,
    N(-d2) for a put.

    THIS IS NOT a real-world / statistical probability. It's the
    probability under the risk-neutral measure implied by treating the
    option's IV as the true vol and the stock as drifting at the
    risk-free rate — an artifact of no-arbitrage pricing, not a forecast
    of where the stock will actually go. Use it to price and rank setups,
    never to justify a directional call on its own.
    """
    _check_option_type(option_type)
    d = compute_d1_d2(S, K, T, r, q, sigma)
    return norm.cdf(d.d2) if option_type == "call" else norm.cdf(-d.d2)


def binomial_crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    american: bool = True,
    n_steps: int = 500,
) -> float:
    """Cox-Ross-Rubinstein binomial tree, early-exercise aware for American options.

        u = e^(sigma*sqrt(dt)),  d = 1/u
        p = (e^((r-q)*dt) - d) / (u - d)      [risk-neutral up probability]
        discount = e^(-r*dt)

    Backward induction: at each node, the American value is
    max(intrinsic value, discounted continuation value); the European
    value just takes the continuation value and never checks intrinsic
    value early. This is exactly where American and European pricing
    diverge — see tests/test_pricing.py for the case that proves it
    (American calls on non-dividend stock should NEVER early-exercise, so
    American == European there; American puts do diverge).
    """
    _check_option_type(option_type)
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside (0,1) — dt is too "
            "large relative to r, q, sigma. Increase n_steps."
        )

    # Terminal payoffs across all n_steps+1 terminal nodes.
    terminal_prices = [S * (u**j) * (d ** (n_steps - j)) for j in range(n_steps + 1)]
    if option_type == "call":
        values = [max(px - K, 0.0) for px in terminal_prices]
    else:
        values = [max(K - px, 0.0) for px in terminal_prices]

    # Roll backward one time step at a time, checking early exercise at
    # every node when american=True.
    for step in range(n_steps - 1, -1, -1):
        next_values = [0.0] * (step + 1)
        for j in range(step + 1):
            continuation = disc * (p * values[j + 1] + (1 - p) * values[j])
            if american:
                spot_at_node = S * (u**j) * (d ** (step - j))
                intrinsic = max(spot_at_node - K, 0.0) if option_type == "call" else max(K - spot_at_node, 0.0)
                next_values[j] = max(continuation, intrinsic)
            else:
                next_values[j] = continuation
        values = next_values

    return values[0]


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    initial_guess: float = 0.3,
    max_iter: int = 100,
    tol: float = 1e-8,
    bounds: tuple[float, float] = (1e-4, 5.0),
) -> float:
    """Solve for sigma such that black_scholes_price(..., sigma) == market_price.

    Primary method is Newton-Raphson using vega as the derivative:

        sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

    This converges quadratically near the root but is unstable whenever
    vega is small — which happens for deep ITM/OTM contracts, where price
    is nearly flat in sigma. In that regime we fall back to Brent's method
    (scipy.optimize.brentq), a bracketed bisection-family solver that's
    slower but guaranteed to converge given a sign change across the
    bracket, which is exactly the case where Newton-Raphson can't be
    trusted.
    """
    _check_option_type(option_type)
    from risk_tool.greeks import vega as vega_fn  # local import: avoids a pricing<->greeks import cycle

    sigma = initial_guess
    for _ in range(max_iter):
        try:
            price = black_scholes_price(S, K, T, r, q, sigma, option_type)
        except ValueError:
            break  # sigma wandered non-positive; drop to Brent's method
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        v = vega_fn(S, K, T, r, q, sigma)
        if v < 1e-8:
            break  # vega too small for Newton-Raphson to make progress
        sigma -= diff / v
        if sigma <= 0:
            break  # stepped into an invalid region

    def objective(s: float) -> float:
        return black_scholes_price(S, K, T, r, q, s, option_type) - market_price

    lo, hi = bounds
    try:
        return brentq(objective, lo, hi, xtol=tol, maxiter=max_iter)
    except ValueError as exc:
        raise ValueError(
            f"Could not solve implied vol within bounds {bounds} for "
            f"market_price={market_price}: {exc}. The price may be outside "
            "any value Black-Scholes can produce for this contract "
            "(e.g. below intrinsic value, or a data error)."
        ) from exc
