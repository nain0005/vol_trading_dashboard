"""Auto strike-selection for four defined-structure multi-leg strategies:
bear put spread, bull put spread, straddle, strangle.

Builds directly on two modules that already exist:
  - strike_selection.py's core idea — compare a structure's market price
    against YOUR OWN vol view; see that module's docstring for the full
    argument that this is the only honest source of edge here. Nothing
    below changes that argument, only extends it from one leg to two.
  - option_strategy.py's exact piecewise-linear payoff analysis (max
    profit, max loss, breakevens) — reused as-is, not reimplemented.

Every candidate is built from strikes ACTUALLY LISTED in the live chain
you pass in (not synthetic price levels), and each leg is priced at ITS
OWN market IV from that chain — skew-aware, not a flat-vol assumption.

Probability of profit is a blended-IV approximation: a structure has legs
at different strikes with (generally) different market IVs, and getting
one P(profit) number requires picking ONE terminal distribution. We use
a single lognormal at the strikes' average IV, then get a real
(if simplified) probability by treating the breakeven itself as a
synthetic strike: P(S_T < breakeven) is exactly itm_probability(...,
K=breakeven, option_type="put"); P(S_T > breakeven) is exactly
itm_probability(..., K=breakeven, option_type="call"). This reuses
existing, tested math rather than inventing new probability code — but
averaging two different IVs into one sigma is still an approximation,
most defensible when the legs' IVs are close (usually true for strikes a
handful of increments apart).

All dollar figures throughout (net_cost, max_profit, max_loss, edge_ev,
my_vol_fair_value) are PER SHARE, matching strike_selection.py's
convention — multiply by contract_multiplier (100) for per-contract
dollars, same as that module's callers already do.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from risk_tool.config import DEFAULT_CONFIG, RiskConfig
from risk_tool.option_strategy import OptionLeg, analyze_strategy
from risk_tool.pricing import black_scholes_price, itm_probability

STRATEGIES = ("bear_put_spread", "bull_put_spread", "straddle", "strangle")


@dataclass
class ChainLeg:
    """One strike's market data, pulled from the live chain."""

    strike: float
    option_type: str  # "call" / "put"
    iv: float
    mid: float


def _lookup(chain: pd.DataFrame, strike: float, option_type: str) -> ChainLeg | None:
    rows = chain[(chain["strike"] == strike) & (chain["type"] == option_type)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if row["iv"] <= 0 or row["mid"] <= 0:
        return None
    return ChainLeg(strike=strike, option_type=option_type, iv=float(row["iv"]), mid=float(row["mid"]))


def _to_option_legs(resolved: list[tuple[str, ChainLeg]]) -> list[OptionLeg]:
    # shares_per_contract=1.0 keeps every downstream number per-share, so
    # this module's outputs stay directly comparable to strike_selection.py's.
    return [
        OptionLeg(
            option_type=cl.option_type,
            strike=cl.strike,
            premium=cl.mid,
            contracts=1.0 if side == "long" else -1.0,
            shares_per_contract=1.0,
        )
        for side, cl in resolved
    ]


def _prob_profit(strategy: str, breakevens: list[float], S: float, T: float, r: float, q: float, blended_iv: float) -> float | None:
    """None means "undefined", not "zero" -- e.g. a breakeven at or below
    $0 (possible with a deep/degenerate spread, or loose synthetic demo
    IVs that don't enforce a realistic no-arbitrage relationship between
    strikes) isn't a strike Black-Scholes can price against; itm_probability
    requires a strictly positive strike, so guard rather than crash."""
    if strategy == "bear_put_spread":
        if len(breakevens) < 1 or breakevens[0] <= 0:
            return None
        return itm_probability(S, breakevens[0], T, r, q, blended_iv, "put")
    if strategy == "bull_put_spread":
        if len(breakevens) < 1 or breakevens[0] <= 0:
            return None
        return itm_probability(S, breakevens[0], T, r, q, blended_iv, "call")
    if strategy in ("straddle", "strangle"):
        if len(breakevens) < 2:
            return None
        lo, hi = min(breakevens), max(breakevens)
        if lo <= 0 or hi <= 0:
            return None
        return itm_probability(S, lo, T, r, q, blended_iv, "put") + itm_probability(S, hi, T, r, q, blended_iv, "call")
    return None


@dataclass
class SpreadCandidate:
    strategy: str
    legs: list  # list of (side, ChainLeg)
    net_cost: float  # per share; positive = debit paid, negative = credit received
    max_profit: float | None  # per share; None = unlimited
    max_loss: float | None  # per share; None = unlimited (negative when finite)
    breakevens: list[float]
    risk_reward: float | None  # max_profit / abs(max_loss), only when both finite
    prob_profit: float | None
    my_vol_fair_value: float | None  # per share, at the vol you supplied
    edge_ev: float | None  # my_vol_fair_value - net_cost; None if no my_vol given
    is_poor_setup: bool

    def strike_label(self) -> str:
        return " / ".join(f"{side} ${cl.strike:.2f}{cl.option_type[0].upper()}" for side, cl in self.legs)


def evaluate_candidate(
    strategy: str,
    leg_specs: list[tuple[str, str, float]],  # (side, option_type, strike)
    chain: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    q: float,
    config: RiskConfig = DEFAULT_CONFIG,
    my_vol: float | None = None,
) -> SpreadCandidate | None:
    """None means one or more legs aren't quoted (no live IV/mid) in the chain."""
    resolved: list[tuple[str, ChainLeg]] = []
    for side, opt_type, strike in leg_specs:
        cl = _lookup(chain, strike, opt_type)
        if cl is None:
            return None
        resolved.append((side, cl))

    opt_legs = _to_option_legs(resolved)
    profile = analyze_strategy(opt_legs)

    risk_reward = None
    if profile.max_profit is not None and profile.max_loss is not None and profile.max_loss < 0:
        risk_reward = profile.max_profit / abs(profile.max_loss)
    is_poor_setup = bool(risk_reward is not None and risk_reward < config.min_risk_reward_ratio)

    blended_iv = sum(cl.iv for _, cl in resolved) / len(resolved)
    prob_profit = _prob_profit(strategy, profile.breakevens, S, T, r, q, blended_iv)

    my_vol_fair_value = None
    edge_ev = None
    if my_vol:
        my_vol_fair_value = sum(
            black_scholes_price(S, cl.strike, T, r, q, my_vol, cl.option_type) * (1 if side == "long" else -1)
            for side, cl in resolved
        )
        edge_ev = my_vol_fair_value - profile.net_debit

    return SpreadCandidate(
        strategy=strategy,
        legs=resolved,
        net_cost=profile.net_debit,
        max_profit=profile.max_profit,
        max_loss=profile.max_loss,
        breakevens=profile.breakevens,
        risk_reward=risk_reward,
        prob_profit=prob_profit,
        my_vol_fair_value=my_vol_fair_value,
        edge_ev=edge_ev,
        is_poor_setup=is_poor_setup,
    )


def _nearby_strikes(all_strikes: list[float], spot: float, strike_increment: float, num_each_side: int) -> list[float]:
    atm = round(spot / strike_increment) * strike_increment
    lo, hi = atm - strike_increment * num_each_side, atm + strike_increment * num_each_side
    return sorted(k for k in all_strikes if lo - 1e-6 <= k <= hi + 1e-6)


def _bear_put_spread_specs(strikes: list[float]) -> list[list[tuple[str, str, float]]]:
    return [
        [("long", "put", long_k), ("short", "put", short_k)]
        for long_k in strikes
        for short_k in strikes
        if short_k < long_k
    ]


def _bull_put_spread_specs(strikes: list[float]) -> list[list[tuple[str, str, float]]]:
    return [
        [("short", "put", short_k), ("long", "put", long_k)]
        for short_k in strikes
        for long_k in strikes
        if long_k < short_k
    ]


def _straddle_specs(strikes: list[float]) -> list[list[tuple[str, str, float]]]:
    return [[("long", "call", k), ("long", "put", k)] for k in strikes]


def _strangle_specs(strikes: list[float], spot: float) -> list[list[tuple[str, str, float]]]:
    calls = [k for k in strikes if k > spot]
    puts = [k for k in strikes if k < spot]
    return [[("long", "call", c), ("long", "put", p)] for c in calls for p in puts]


def find_best_spreads(
    strategy: str,
    chain: pd.DataFrame,
    spot: float,
    T: float,
    r: float,
    q: float,
    strike_increment: float,
    num_each_side: int = 6,
    config: RiskConfig = DEFAULT_CONFIG,
    my_vol: float | None = None,
    top_n: int = 10,
) -> list[SpreadCandidate]:
    """The one function to call from outside this module. Returns up to
    top_n candidates, ranked by edge EV when my_vol is given (the honest
    edge signal), or by risk:reward otherwise (a market-neutral fallback —
    with no vol view of your own, there's no edge number to rank by)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if chain.empty:
        return []

    all_strikes = sorted(chain["strike"].unique())
    strikes = _nearby_strikes(all_strikes, spot, strike_increment, num_each_side)

    if strategy == "bear_put_spread":
        specs = _bear_put_spread_specs(strikes)
    elif strategy == "bull_put_spread":
        specs = _bull_put_spread_specs(strikes)
    elif strategy == "straddle":
        specs = _straddle_specs(strikes)
    else:
        specs = _strangle_specs(strikes, spot)

    candidates = [
        c for leg_specs in specs if (c := evaluate_candidate(strategy, leg_specs, chain, spot, T, r, q, config, my_vol)) is not None
    ]

    if my_vol:
        candidates.sort(key=lambda c: c.edge_ev if c.edge_ev is not None else float("-inf"), reverse=True)
    else:
        candidates.sort(key=lambda c: c.risk_reward if c.risk_reward is not None else (c.prob_profit or 0.0), reverse=True)

    return candidates[:top_n]
