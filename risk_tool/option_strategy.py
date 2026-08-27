"""Multi-leg option strategy payoff at expiration — combines any number of
long/short call/put legs on the same underlying into one payoff function,
and derives max profit, max loss, and breakeven(s) from it.

No new pricing math here: a short leg is exactly a long leg with negated
contracts (you flip from paying the premium to receiving it, and from
owning the intrinsic value to owing it) — see
hedge.option_intrinsic_pl, which this module calls directly. The payoff
function of any such combination is piecewise-linear in the underlying
price, with kinks only at the legs' strikes, which is what makes max/min
and breakeven-finding exact rather than approximate: extrema of a
piecewise-linear function only occur at its kinks or at the domain
boundary (spot = 0), and this checks all of them plus the asymptotic slope
past the highest strike for unlimited profit/loss.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk_tool.hedge import option_intrinsic_pl
from risk_tool.pricing import black_scholes_price

UNBOUNDED_SLOPE_EPS = 1e-6


@dataclass
class OptionLeg:
    option_type: str  # "call" or "put"
    strike: float
    premium: float  # per-share premium, always a positive magnitude
    contracts: float  # signed: positive = long, negative = short
    shares_per_contract: float = 100.0
    iv: float | None = None  # current implied vol, e.g. 0.35 — only needed for strategy_pl_today


def leg_pl(leg: OptionLeg, spot_at_expiration: float) -> float:
    return option_intrinsic_pl(
        spot_at_expiration, leg.strike, leg.premium, leg.option_type, leg.contracts, leg.shares_per_contract
    )


def strategy_pl(legs: list[OptionLeg], spot_at_expiration: float) -> float:
    """Combined P&L of every leg at a given terminal underlying price."""
    return sum(leg_pl(leg, spot_at_expiration) for leg in legs)


def leg_pl_today(leg: OptionLeg, spot: float, T_years: float, r: float = 0.05, q: float = 0.0) -> float:
    """Mark-to-market P&L for one leg right now — reprices via Black-Scholes
    at the leg's own current IV and the shared time remaining, instead of
    assuming only intrinsic value (which is only correct exactly at
    expiration). Requires leg.iv to be set."""
    if leg.iv is None:
        raise ValueError(f"leg at strike {leg.strike} has no iv set — strategy_pl_today needs every leg's current IV")
    if T_years <= 0:
        return leg_pl(leg, spot)  # at/past expiration, Black-Scholes needs T>0 — intrinsic value is exact here anyway
    price = black_scholes_price(spot, leg.strike, T_years, r, q, leg.iv, leg.option_type)
    return (price - leg.premium) * leg.contracts * leg.shares_per_contract


def strategy_pl_today(legs: list[OptionLeg], spot: float, T_years: float, r: float = 0.05, q: float = 0.0) -> float:
    """Combined mark-to-market P&L right now (not at expiration) — the
    "live" curve: what the strategy is actually worth today, given how much
    time value and vol premium remain, rather than only its terminal shape."""
    return sum(leg_pl_today(leg, spot, T_years, r, q) for leg in legs)


def net_debit(legs: list[OptionLeg]) -> float:
    """Total dollars paid to open (positive = net debit paid; negative = net
    credit received). Sum of each leg's premium * signed contracts * multiplier."""
    return sum(leg.premium * leg.contracts * leg.shares_per_contract for leg in legs)


@dataclass
class StrategyProfile:
    max_profit: float | None  # None = unlimited (e.g. a naked long call)
    max_loss: float | None  # None = unlimited (e.g. a naked short call)
    breakevens: list[float]
    net_debit: float  # positive = paid; negative = received (net credit)


def analyze_strategy(legs: list[OptionLeg]) -> StrategyProfile:
    """Exact max profit / max loss / breakeven(s) for any combination of
    calls/puts on one underlying, found from the payoff's piecewise-linear
    structure rather than a numeric search."""
    if not legs:
        raise ValueError("legs must be non-empty")

    strikes = sorted(set(leg.strike for leg in legs))
    highest_strike = strikes[-1]

    # Slope of the combined payoff for spot > highest_strike: beyond every
    # leg's strike, each leg's intrinsic value moves 1-for-1 with spot (its
    # sign depending on call/long vs call/short — puts stop mattering
    # entirely up here), so this sum is exact, not a numeric approximation.
    slope_beyond_highest = sum(
        (1.0 if leg.option_type == "call" else 0.0) * (1 if leg.contracts > 0 else -1) * abs(leg.contracts) * leg.shares_per_contract
        for leg in legs
    )

    # Candidate breakpoints: spot=0 (the domain floor) plus every strike.
    # A piecewise-linear function's extrema over a bounded domain can only
    # occur at these breakpoints.
    candidates = [0.0] + strikes
    values = [(p, strategy_pl(legs, p)) for p in candidates]

    unlimited_profit = slope_beyond_highest > UNBOUNDED_SLOPE_EPS
    unlimited_loss = slope_beyond_highest < -UNBOUNDED_SLOPE_EPS

    finite_values = [v for _, v in values]
    max_profit = None if unlimited_profit else max(finite_values)
    max_loss = None if unlimited_loss else min(finite_values)

    # Breakevens: linear-interpolate zero-crossings between consecutive
    # evaluated points, including the unbounded tail (approximated by a
    # point far past the last strike, using the exact asymptotic slope).
    points = list(values)
    if slope_beyond_highest != 0:
        far_spot = highest_strike + 10_000.0
        far_value = values[-1][1] + slope_beyond_highest * (far_spot - highest_strike)
        points.append((far_spot, far_value))

    breakevens: list[float] = []
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if v0 == 0:
            breakevens.append(round(p0, 6))
        elif (v0 < 0) != (v1 < 0):  # sign change across this segment
            t = -v0 / (v1 - v0)
            breakevens.append(round(p0 + t * (p1 - p0), 6))
    if points[-1][1] == 0:
        breakevens.append(round(points[-1][0], 6))
    breakevens = sorted(set(breakevens))

    return StrategyProfile(
        max_profit=max_profit,
        max_loss=max_loss,
        breakevens=breakevens,
        net_debit=net_debit(legs),
    )
