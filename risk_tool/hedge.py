"""Beta-hedge sizing: offset a directional position (shares or delta-bearing
options) with a correlated hedge instrument — an ETF, or a futures contract
via its per-unit contract multiplier.

The core idea: a position's exposure to the hedge instrument isn't its
notional, it's its DELTA-adjusted exposure (shares held as-is; delta *
contracts * shares_per_contract for options, since an option moves less
than 1:1 with the underlying). Convert that to a dollar amount, scale by
beta (the regression slope of the underlying's returns on the hedge
instrument's returns), and that's the dollar amount of hedge instrument to
take the *opposite* side of the position's implied hedge-instrument
exposure.

What this does NOT do:
  - Offset theta or vega on an options position — only directional
    (delta) risk correlated with the hedge instrument.
  - Guarantee a profitable trade. It reduces the correlated portion of the
    P&L swing; the position can still lose money on its idiosyncratic
    (non-hedge-explained) move, which R² tells you the size of.
  - Stay hedged over time without rebalancing. Option delta changes with
    spot/time (gamma/theta), and beta itself is a rolling-window estimate
    that drifts — both mean this hedge decays and needs periodic
    recomputation, not a "set and forget" position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BetaEstimate:
    beta: float  # OLS slope: underlying_return ~= beta * hedge_return
    r_squared: float  # fraction of the underlying's return variance the hedge instrument explains
    n_obs: int


def simple_returns(prices: pd.Series) -> pd.Series:
    """Period-over-period % change, first (NaN) row dropped."""
    return prices.pct_change().dropna()


def estimate_beta(underlying_returns: pd.Series, hedge_returns: pd.Series) -> BetaEstimate:
    """OLS slope of underlying_returns on hedge_returns, plus R².

    Aligns on index (so callers can pass returns computed independently,
    e.g. from two get_equity_historicals() pulls on different calendars)
    and only uses overlapping observations.

    This is a rolling-window estimate over whatever history you feed it —
    re-run it periodically rather than sizing a hedge once and forgetting
    it; beta over the last month and beta over the last year routinely
    disagree.
    """
    idx = underlying_returns.dropna().index.intersection(hedge_returns.dropna().index)
    u = underlying_returns.loc[idx].to_numpy(dtype=float)
    h = hedge_returns.loc[idx].to_numpy(dtype=float)
    n = len(u)
    if n < 3:
        raise ValueError(f"Need at least 3 overlapping return observations to estimate beta, got {n}.")

    h_mean, u_mean = h.mean(), u.mean()
    cov = float(np.sum((h - h_mean) * (u - u_mean)))
    var_h = float(np.sum((h - h_mean) ** 2))
    var_u = float(np.sum((u - u_mean) ** 2))
    if var_h == 0:
        raise ValueError("Hedge instrument returns have zero variance — can't estimate beta.")

    beta = cov / var_h
    r_squared = (cov**2) / (var_u * var_h) if var_u > 0 else 0.0
    return BetaEstimate(beta=beta, r_squared=r_squared, n_obs=n)


def option_position_exposure_shares(delta: float, contracts: float, shares_per_contract: float = 100.0) -> float:
    """Delta-equivalent signed share exposure of an option position.

    `delta` carries the sign your broker quotes it with — negative for a
    long put or short call, positive for a long call or short put. Pass a
    negative `contracts` for a short options position if your delta is
    already signed for a long position (or just sign delta yourself,
    whichever is less error-prone for the caller)."""
    return delta * contracts * shares_per_contract


def share_position_exposure_shares(quantity: float, side: str) -> float:
    """side: 'long' or 'short'. quantity should be positive (unsigned)."""
    side = side.lower()
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    if quantity < 0:
        raise ValueError("quantity must be non-negative — use side to set direction.")
    return quantity if side == "long" else -quantity


@dataclass
class HedgeSizeResult:
    exposure_shares: float  # signed, delta-equivalent shares of the underlying
    exposure_dollars: float  # signed
    hedge_dollars: float  # signed; positive = go long the hedge instrument, negative = go short
    hedge_units: float  # signed; ETF/stock shares (multiplier=1) or futures contracts (multiplier=barrels/contract)
    direction: str  # "long" or "short" — which side of the hedge instrument to take


def size_hedge(
    exposure_shares: float,
    underlying_price: float,
    beta: float,
    hedge_price: float,
    hedge_contract_multiplier: float = 1.0,
) -> HedgeSizeResult:
    """The one function to call from outside this module for sizing.

        hedge_dollars = -(exposure_shares * underlying_price) * beta
        hedge_units   = hedge_dollars / (hedge_price * hedge_contract_multiplier)

    A net-short position (negative exposure_shares) with positive beta
    produces positive hedge_dollars — i.e. buy/long the hedge instrument,
    since the short position already carries implicit short exposure to
    anything positively correlated with the underlying. Flip the position
    to net-long, or use a negative beta, and the sign flips accordingly —
    read `direction` / the sign of `hedge_units` rather than reasoning
    about it by hand each time.

    hedge_contract_multiplier: 1.0 for an ETF/stock hedge (hedge_units is
    then a plain share count). For a futures hedge, pass the contract's
    units-per-contract (e.g. 1000 for CME WTI/CL, 100 for Micro WTI/MCL)
    with hedge_price quoted per unit (per barrel) — hedge_units then comes
    out in contracts.
    """
    if underlying_price <= 0:
        raise ValueError("underlying_price must be positive")
    if hedge_price <= 0:
        raise ValueError("hedge_price must be positive")
    if hedge_contract_multiplier <= 0:
        raise ValueError("hedge_contract_multiplier must be positive")

    exposure_dollars = exposure_shares * underlying_price
    hedge_dollars = -exposure_dollars * beta
    hedge_units = hedge_dollars / (hedge_price * hedge_contract_multiplier)

    return HedgeSizeResult(
        exposure_shares=exposure_shares,
        exposure_dollars=exposure_dollars,
        hedge_dollars=hedge_dollars,
        hedge_units=hedge_units,
        direction="long" if hedge_units >= 0 else "short",
    )


def same_underlying_hedge_shares(net_delta_shares: float) -> float:
    """Exact share hedge to flip an option position's aggregate delta-
    equivalent exposure to zero, when hedging with shares of the SAME
    underlying the options are on. Unlike size_hedge's cross-asset case,
    there's no beta/price-ratio conversion here — a share of the
    underlying has a delta of exactly 1 against itself, so the hedge is
    just the sign-flipped net delta. Kept as a tiny named function (not a
    bare minus sign in the UI layer) so the "ratio is exactly 1" reasoning
    is documented once, in the one place it's true."""
    return -net_delta_shares


def beta_across_windows(
    underlying_returns: pd.Series,
    hedge_returns: pd.Series,
    windows: tuple[int, ...] = (30, 60, 90, 180),
) -> dict[int, BetaEstimate | None]:
    """Re-run estimate_beta over several different TRAILING windows (in
    trading days) carved out of the SAME two return series — no new
    statistics, just calling the existing, tested OLS estimator on
    different slices of history.

    Why this matters: size_hedge's beta is a single point estimate over
    whatever one lookback window you happened to pick. That number can
    look precise while being highly sensitive to the window — a beta of
    0.6 over the last 30 days and 0.2 over the last 180 days are BOTH
    "the beta," and a hedge sized on either one alone hides that
    disagreement. Comparing several windows surfaces it: estimates that
    cluster together mean the hedge ratio is on reasonably solid ground;
    estimates that swing widely mean "the beta" is itself a fragile
    number this quarter, and the hedge should be treated as rough at best
    (and rebalanced/re-estimated more often) regardless of how many
    decimal places size_hedge prints.

    A window longer than the available overlapping history maps to None
    (skipped) rather than silently shrinking to whatever data exists —
    that way the caller can tell "not enough history for a 180-day
    estimate" apart from "estimated it, and the answer happens to be
    small." Windows are trailing-most-recent: the 30-day estimate uses
    the last 30 overlapping observations, not the first 30.
    """
    idx = underlying_returns.dropna().index.intersection(hedge_returns.dropna().index)
    idx = idx.sort_values()
    results: dict[int, BetaEstimate | None] = {}
    for w in windows:
        if len(idx) < w:
            results[w] = None
            continue
        window_idx = idx[-w:]
        try:
            results[w] = estimate_beta(underlying_returns.loc[window_idx], hedge_returns.loc[window_idx])
        except ValueError:
            results[w] = None
    return results


def option_intrinsic_pl(
    spot_at_scenario: float,
    strike: float,
    premium: float,
    option_type: str,
    contracts: float,
    shares_per_contract: float = 100.0,
) -> float:
    """P&L at expiration (intrinsic value only — no remaining time value) for
    an option position. Use a negative `contracts` for a short position;
    `premium` should be signed the same way you paid/received it (positive
    for what you paid on a long position)."""
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    intrinsic = max(spot_at_scenario - strike, 0.0) if option_type == "call" else max(strike - spot_at_scenario, 0.0)
    return (intrinsic - premium) * contracts * shares_per_contract


def share_position_pl(price_at_scenario: float, entry_price: float, exposure_shares: float) -> float:
    """P&L for a share position. exposure_shares is signed (+ long, - short),
    as returned by share_position_exposure_shares."""
    return exposure_shares * (price_at_scenario - entry_price)


def hedge_instrument_pl(hedge_price_at_scenario: float, hedge_price_now: float, hedge_units: float) -> float:
    """P&L for the hedge leg. hedge_units is signed (+ long, - short), as
    returned by size_hedge — pass it straight through."""
    return hedge_units * (hedge_price_at_scenario - hedge_price_now)
