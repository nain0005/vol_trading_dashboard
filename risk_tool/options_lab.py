"""Options Lab: scenario analysis on top of any option_strategy.OptionLeg
combination -- a P&L surface across spot x time, aggregate Greeks across a
spot range, and shock scenarios (a live what-if slider, or a discrete
earnings/IV-crush event).

Nothing here is a new pricing model. Everything is built on two functions
that already exist and are already tested elsewhere in this codebase:
  - option_strategy.strategy_pl_today -- Black-Scholes-reprices every leg
    at ANY remaining time, not just "right now" (T_years=0 already falls
    back to exact intrinsic value, which is what makes the P&L grid's
    expiration row match option_strategy.analyze_strategy exactly).
  - risk_tool.greeks.all_greeks -- per-leg Greeks at a given spot/time/vol.
This module is UI-facing repricing at different points in (spot, time,
vol) space, not a new model.

Constant-vol simplification -- read this before trusting the P&L grid or
the earnings simulator: every leg keeps the vol you gave it except where
you explicitly shock it. There is no vol *surface* here -- no skew that
moves as spot moves, no smile, no term-structure change as time passes
(other than the vol level you set staying fixed). Treat outputs as "what
if price/vol move along this specific path, holding everything else
about the vol assumption fixed" -- not a full forward-looking risk model
that accounts for how IV itself would realistically shift with the
market.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from risk_tool.greeks import all_greeks
from risk_tool.option_strategy import OptionLeg, strategy_pl_today

_GREEK_KEYS = ("delta", "gamma", "theta", "vega")


def shock_legs(legs: list[OptionLeg], iv_shift: float = 0.0, iv_mult: float = 1.0) -> list[OptionLeg]:
    """Copy of legs with each leg's iv scaled (iv_mult) then shifted
    (iv_shift, additive vol points) -- e.g. iv_mult=0.6 models a 40%
    relative post-earnings IV crush, iv_shift=+0.02 layers on a further
    +2-point event-day bump. Floored at a tiny positive number so
    Black-Scholes never receives sigma<=0. Legs with no iv set pass
    through unchanged (they'll still fail strategy_pl_today/
    strategy_greeks's "every leg needs iv" requirement, same as today)."""
    shocked = []
    for leg in legs:
        if leg.iv is None:
            shocked.append(leg)
            continue
        new_iv = max(leg.iv * iv_mult + iv_shift, 1e-4)
        shocked.append(replace(leg, iv=new_iv))
    return shocked


def strategy_greeks(legs: list[OptionLeg], spot: float, T_years: float, r: float = 0.05, q: float = 0.0) -> dict:
    """Aggregate delta/gamma/theta/vega across all legs at a given spot and
    time remaining. Every leg must have iv set (same requirement as
    strategy_pl_today). Returns all-zero at/past expiration (T_years<=0) --
    Greeks aren't defined there in this model, not "small but nonzero"."""
    totals = {k: 0.0 for k in _GREEK_KEYS}
    if T_years <= 0:
        return totals
    for leg in legs:
        if leg.iv is None:
            raise ValueError(f"leg at strike {leg.strike} has no iv set -- strategy_greeks needs every leg's current IV")
        g = all_greeks(spot, leg.strike, T_years, r, q, leg.iv, leg.option_type)
        weight = leg.contracts * leg.shares_per_contract
        for k in _GREEK_KEYS:
            totals[k] += g[k] * weight
    return totals


def pl_grid(
    legs: list[OptionLeg],
    spot_now: float,
    dte: int,
    r: float = 0.05,
    q: float = 0.0,
    spot_range_pct: float = 0.20,
    n_spot_points: int = 41,
    n_time_points: int = 12,
) -> pd.DataFrame:
    """P&L surface in long form: one row per (days_forward, spot) cell.

    days_forward always includes 0 (today) and dte (expiration) exactly,
    plus n_time_points-2 evenly spaced steps between -- the expiration row
    reprices via T_years<=0, which strategy_pl_today already handles as
    exact intrinsic value, so it matches
    option_strategy.analyze_strategy's payoff exactly, not an
    approximation of it.
    """
    if dte < 0:
        raise ValueError("dte must be >= 0")
    spots = np.linspace(spot_now * (1 - spot_range_pct), spot_now * (1 + spot_range_pct), n_spot_points)
    days_forward = sorted(set([0, dte] + list(np.linspace(0, dte, n_time_points).astype(int).tolist())))

    rows = []
    for d in days_forward:
        T_remaining = max(dte - d, 0) / 365.0
        for s in spots:
            pl = strategy_pl_today(legs, float(s), T_remaining, r, q)
            rows.append({"days_forward": d, "spot": float(s), "pl": pl})
    return pd.DataFrame(rows)


def greeks_curve(
    legs: list[OptionLeg],
    spot_now: float,
    T_years: float,
    r: float = 0.05,
    q: float = 0.0,
    spot_range_pct: float = 0.20,
    n_points: int = 41,
) -> pd.DataFrame:
    """One row per spot level with the strategy's aggregate Greeks at that
    spot, time-to-expiration held fixed at T_years -- "what would Greeks
    look like if spot jumped there right now", not a forward simulation
    (pl_grid covers the time dimension separately)."""
    spots = np.linspace(spot_now * (1 - spot_range_pct), spot_now * (1 + spot_range_pct), n_points)
    rows = [{"spot": float(s), **strategy_greeks(legs, float(s), T_years, r, q)} for s in spots]
    return pd.DataFrame(rows)


@dataclass
class ScenarioResult:
    spot: float
    days_forward: int
    T_remaining: float
    pl: float
    greeks: dict  # empty dict at/past expiration


def evaluate_scenario(
    legs: list[OptionLeg],
    spot_now: float,
    dte: int,
    spot_move_pct: float = 0.0,
    iv_shift: float = 0.0,
    iv_mult: float = 1.0,
    days_forward: int = 0,
    r: float = 0.05,
    q: float = 0.0,
) -> ScenarioResult:
    """The one function behind both the live what-if sliders and the
    earnings/IV-crush simulator -- they're the same operation (move spot,
    shock vol, advance time, reprice), just different default magnitudes
    and framing in the UI."""
    scenario_legs = shock_legs(legs, iv_shift=iv_shift, iv_mult=iv_mult)
    spot_scenario = spot_now * (1 + spot_move_pct)
    T_remaining = max(dte - days_forward, 0) / 365.0
    pl = strategy_pl_today(scenario_legs, spot_scenario, T_remaining, r, q)
    greeks = strategy_greeks(scenario_legs, spot_scenario, T_remaining, r, q) if T_remaining > 0 else {}
    return ScenarioResult(spot=spot_scenario, days_forward=days_forward, T_remaining=T_remaining, pl=pl, greeks=greeks)
