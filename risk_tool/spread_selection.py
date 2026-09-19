"""Auto strike-selection for eight defined-structure multi-leg strategies:
bear put spread, bull put spread, bull call spread, bear call spread,
straddle, strangle, iron condor, iron butterfly.

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

Two-breakeven structures split into two mirror-image families:
straddle/strangle (both legs bought — long vol) profit OUTSIDE the two
breakevens, so P(profit) sums both tails. Iron condor/iron butterfly
(both spreads sold, wings bought for protection — short vol, defined
risk) profit BETWEEN the two breakevens, so P(profit) is 1 minus both
tails. Same two itm_probability calls, opposite combination — see
_prob_profit.

All dollar figures throughout (net_cost, max_profit, max_loss, edge_ev,
my_vol_fair_value) are PER SHARE, matching strike_selection.py's
convention — multiply by contract_multiplier (100) for per-contract
dollars, same as that module's callers already do.

Strike search has two alternative sources (find_best_spreads's
strike_source argument):
  - "increment" (the default, unchanged from before): a window of
    consecutive listed strikes at a flat $ increment around ATM. Simple,
    predictable, no dependency on how OI happens to be shaped today.
  - "oi_percentile": strikes placed at chosen percentiles of a normal or
    Student-t curve FIT TO OPEN INTEREST BY STRIKE (oi_distribution.py),
    snapped to the nearest strike actually listed. Read oi_distribution's
    module docstring before using this — in short, "OI is concentrated
    near strike K" is a statement about where EXISTING POSITIONS already
    sit (retail round-number clustering, market-maker hedging flow, and
    stale positions all look the same in this number), not a forecast,
    not "smart money," and not the same thing as max_pain(). Using
    OI-implied percentiles to place strikes is a bet that current
    positioning is informative about where liquidity/interest will
    concentrate at expiration — a real, defensible idea, but a different
    and weaker claim than "this is where the stock is going."
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from risk_tool.config import DEFAULT_CONFIG, RiskConfig
from risk_tool.oi_distribution import DistributionFit, distribution_percentile_strike
from risk_tool.option_strategy import OptionLeg, analyze_strategy
from risk_tool.pricing import black_scholes_price, itm_probability

STRATEGIES = (
    "bear_put_spread",
    "bull_put_spread",
    "bull_call_spread",
    "bear_call_spread",
    "straddle",
    "strangle",
    "iron_condor",
    "iron_butterfly",
)

# Human-readable labels for UI dropdowns / comparison tables — kept here
# (not in dashboard.py) so the strategy set and its display names can't
# drift apart.
STRATEGY_LABELS = {
    "bear_put_spread": "Bear put spread (bearish, defined risk)",
    "bull_put_spread": "Bull put spread (bullish/neutral, credit)",
    "bull_call_spread": "Bull call spread (bullish, defined-risk debit)",
    "bear_call_spread": "Bear call spread (bearish/neutral, defined-risk credit)",
    "straddle": "Straddle (long vol, big move either direction)",
    "strangle": "Strangle (long vol, cheaper than a straddle)",
    "iron_condor": "Iron condor (short vol, defined risk, 4 legs)",
    "iron_butterfly": "Iron butterfly (short vol, defined risk, 4 legs, tighter body)",
}


@dataclass
class ChainLeg:
    """One strike's market data, pulled from the live chain."""

    strike: float
    option_type: str  # "call" / "put"
    iv: float
    mid: float
    spread_pct: float | None = None  # (ask-bid)/mid * 100, when the chain has it — liquidity flag, not a pricing input


def _lookup(chain: pd.DataFrame, strike: float, option_type: str) -> ChainLeg | None:
    rows = chain[(chain["strike"] == strike) & (chain["type"] == option_type)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if row["iv"] <= 0 or row["mid"] <= 0:
        return None
    # A negative spread_pct means ask < bid -- a crossed quote. Real chains
    # occasionally have these on dead/no-interest strikes (and synthetic
    # demo pricing can produce them at deep OTM strikes where the modeled
    # premium rounds toward zero); treat it as "no reliable liquidity
    # reading" rather than display a nonsense negative percentage.
    spread_pct = float(row["spread_pct"]) if "spread_pct" in chain.columns and pd.notna(row["spread_pct"]) and row["spread_pct"] >= 0 else None
    return ChainLeg(strike=strike, option_type=option_type, iv=float(row["iv"]), mid=float(row["mid"]), spread_pct=spread_pct)


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
    # Single-breakeven debit/credit verticals: bear put spread and bear call
    # spread are both bearish/neutral — profit BELOW the one breakeven.
    # Bull put spread and bull call spread are both bullish/neutral —
    # profit ABOVE it. Same math regardless of which leg is the call vs.
    # the put; only the payoff's direction matters.
    if strategy in ("bear_put_spread", "bear_call_spread"):
        if len(breakevens) < 1 or breakevens[0] <= 0:
            return None
        return itm_probability(S, breakevens[0], T, r, q, blended_iv, "put")
    if strategy in ("bull_put_spread", "bull_call_spread"):
        if len(breakevens) < 1 or breakevens[0] <= 0:
            return None
        return itm_probability(S, breakevens[0], T, r, q, blended_iv, "call")
    if strategy in ("straddle", "strangle"):
        # Long vol, both legs bought: profit OUTSIDE the two breakevens
        # (a big enough move either direction), so sum both tails.
        if len(breakevens) < 2:
            return None
        lo, hi = min(breakevens), max(breakevens)
        if lo <= 0 or hi <= 0:
            return None
        return itm_probability(S, lo, T, r, q, blended_iv, "put") + itm_probability(S, hi, T, r, q, blended_iv, "call")
    if strategy in ("iron_condor", "iron_butterfly"):
        # Short vol, both spreads sold with wings bought for protection:
        # profit is the mirror image of straddle/strangle — BETWEEN the
        # two breakevens (spot stays range-bound), not outside them.
        # P(lo < S_T < hi) = P(S_T > lo) - P(S_T > hi), and
        # itm_probability(..., "call") is exactly P(S_T > K) under this
        # model, so this is one subtraction of the same building block
        # straddle/strangle add.
        if len(breakevens) < 2:
            return None
        lo, hi = min(breakevens), max(breakevens)
        if lo <= 0 or hi <= 0:
            return None
        return itm_probability(S, lo, T, r, q, blended_iv, "call") - itm_probability(S, hi, T, r, q, blended_iv, "call")
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
    avg_spread_pct: float | None  # mean (ask-bid)/mid*100 across legs — liquidity flag, not part of the pricing/edge math

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
    # A near-zero max_loss (credit happens to land almost exactly at the
    # spread width, common with round-number synthetic/demo premiums) is
    # floating-point noise around a TRUE risk of ~$0, not a real number to
    # divide by -- dividing by e.g. -8.88e-16 produces a meaningless
    # multi-quadrillion "ratio" instead of the "essentially riskless"
    # reality. Treat anything tighter than a tenth of a cent as undefined,
    # matching this module's existing None-means-undefined convention.
    if profile.max_profit is not None and profile.max_loss is not None and profile.max_loss < -1e-4:
        risk_reward = profile.max_profit / abs(profile.max_loss)
    is_poor_setup = bool(risk_reward is not None and risk_reward < config.min_risk_reward_ratio)

    blended_iv = sum(cl.iv for _, cl in resolved) / len(resolved)
    prob_profit = _prob_profit(strategy, profile.breakevens, S, T, r, q, blended_iv)

    spread_pcts = [cl.spread_pct for _, cl in resolved if cl.spread_pct is not None]
    avg_spread_pct = sum(spread_pcts) / len(spread_pcts) if spread_pcts else None

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
        avg_spread_pct=avg_spread_pct,
    )


def _nearby_strikes(all_strikes: list[float], spot: float, strike_increment: float, num_each_side: int) -> list[float]:
    atm = round(spot / strike_increment) * strike_increment
    lo, hi = atm - strike_increment * num_each_side, atm + strike_increment * num_each_side
    return sorted(k for k in all_strikes if lo - 1e-6 <= k <= hi + 1e-6)


def _nearest_strike(strikes: list[float], target: float) -> float:
    """The listed strike closest to `target` -- the one nearest-strike
    "snapping" rule used everywhere this module needs to turn an
    arbitrary price level into something actually tradable (iron
    butterfly's ATM body below, and OI-percentile strike search)."""
    return min(strikes, key=lambda k: abs(k - target))


def _oi_percentile_strikes(all_strikes: list[float], fit: DistributionFit, percentiles: tuple[float, ...]) -> list[float]:
    """Strikes at each of `percentiles` (0-1) of a fitted OI-by-strike
    distribution (see oi_distribution.py and this module's docstring for
    what that fit does and doesn't mean), each snapped to the nearest
    strike ACTUALLY LISTED in the chain via _nearest_strike — you can't
    trade a strike that doesn't exist. Deduplicated and sorted: adjacent
    percentiles can snap to the same listed strike (common on a chain
    with wide $ increments or a tight fit), which legitimately shrinks
    the search set below len(percentiles) rather than being an error."""
    if not all_strikes:
        return []
    raw = (distribution_percentile_strike(fit, p) for p in percentiles)
    return sorted({_nearest_strike(all_strikes, r) for r in raw})


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


def _bull_call_spread_specs(strikes: list[float]) -> list[list[tuple[str, str, float]]]:
    """Buy the lower call, sell the higher call — bullish, debit, defined
    risk. Mirror image of _bear_put_spread_specs (calls instead of puts,
    long the cheaper/lower strike instead of the more expensive one)."""
    return [
        [("long", "call", low_k), ("short", "call", high_k)]
        for low_k in strikes
        for high_k in strikes
        if high_k > low_k
    ]


def _bear_call_spread_specs(strikes: list[float]) -> list[list[tuple[str, str, float]]]:
    """Sell the lower call, buy the higher call — bearish/neutral, credit,
    defined risk. Mirror image of _bull_put_spread_specs."""
    return [
        [("short", "call", low_k), ("long", "call", high_k)]
        for low_k in strikes
        for high_k in strikes
        if high_k > low_k
    ]


def _iron_condor_specs(strikes: list[float], spot: float) -> list[list[tuple[str, str, float]]]:
    """A bull put spread (both legs OTM, below spot) plus a bear call
    spread (both legs OTM, above spot), sharing one net-credit ticket.
    Every combination here keeps put strikes strictly below spot and call
    strikes strictly above it — the defining "both sides OTM" shape of a
    real iron condor, not just any 4-leg combination that happens to net
    a credit. `evaluate_candidate`/`analyze_strategy` still derive
    max profit/loss/breakevens exactly from the combined 4-leg payoff;
    nothing about the risk here is assumed or hand-computed."""
    puts = [k for k in strikes if k < spot]
    calls = [k for k in strikes if k > spot]
    return [
        [("long", "put", put_long), ("short", "put", put_short), ("short", "call", call_short), ("long", "call", call_long)]
        for put_long in puts
        for put_short in puts
        if put_short > put_long
        for call_short in calls
        for call_long in calls
        if call_long > call_short
    ]


def _iron_butterfly_specs(strikes: list[float], spot: float) -> list[list[tuple[str, str, float]]]:
    """A short straddle at the single strike closest to spot (the "body"),
    protected by a long call and a long put further out (the "wings").
    Unlike iron condor, the body is fixed at ATM rather than searched —
    that's what makes it a butterfly (a single short strike) instead of a
    condor (a short RANGE): searching every possible body strike would
    mostly just reproduce iron condor's wider-body candidates under a
    different name, so this only varies wing width around the one
    genuinely ATM body."""
    if not strikes:
        return []
    body = _nearest_strike(strikes, spot)
    lower_wings = [k for k in strikes if k < body]
    upper_wings = [k for k in strikes if k > body]
    return [
        [("long", "put", wing_lo), ("short", "put", body), ("short", "call", body), ("long", "call", wing_hi)]
        for wing_lo in lower_wings
        for wing_hi in upper_wings
    ]


DEFAULT_OI_PERCENTILES = (0.16, 0.5, 0.84)  # median plus the classic "~1 sigma" wings


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
    strike_source: str = "increment",
    oi_fit: DistributionFit | None = None,
    oi_percentiles: tuple[float, ...] = DEFAULT_OI_PERCENTILES,
) -> list[SpreadCandidate]:
    """The one function to call from outside this module. Returns up to
    top_n candidates, ranked by edge EV when my_vol is given (the honest
    edge signal), or by risk:reward otherwise (a market-neutral fallback —
    with no vol view of your own, there's no edge number to rank by).

    strike_source picks how the strike search set is built:
      - "increment" (default, unchanged): _nearby_strikes — a window of
        strike_increment-spaced strikes around ATM, num_each_side each way.
      - "oi_percentile": strikes at `oi_percentiles` of `oi_fit` (a
        DistributionFit from oi_distribution.fit_oi_distribution — the
        CALLER fits it and decides normal vs. t, since that's a judgment
        call this module shouldn't make silently), snapped to listed
        strikes. Raises ValueError if oi_fit is None — there's no fit to
        search without one; fit_oi_distribution() itself returns None on
        a chain with too few OI-priced strikes (fewer than 3), so callers
        need to handle that upstream before reaching here.
      strike_increment/num_each_side are ignored when strike_source is
      "oi_percentile" (kept as required positional args so this stays a
      backwards-compatible addition, not a signature break)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strike_source not in ("increment", "oi_percentile"):
        raise ValueError(f"strike_source must be 'increment' or 'oi_percentile', got {strike_source!r}")
    if chain.empty:
        return []

    all_strikes = sorted(chain["strike"].unique())
    if strike_source == "oi_percentile":
        if oi_fit is None:
            raise ValueError("oi_fit is required when strike_source='oi_percentile' — fit one with oi_distribution.fit_oi_distribution(chain) first.")
        strikes = _oi_percentile_strikes(all_strikes, oi_fit, oi_percentiles)
    else:
        strikes = _nearby_strikes(all_strikes, spot, strike_increment, num_each_side)

    if strategy == "bear_put_spread":
        specs = _bear_put_spread_specs(strikes)
    elif strategy == "bull_put_spread":
        specs = _bull_put_spread_specs(strikes)
    elif strategy == "bull_call_spread":
        specs = _bull_call_spread_specs(strikes)
    elif strategy == "bear_call_spread":
        specs = _bear_call_spread_specs(strikes)
    elif strategy == "straddle":
        specs = _straddle_specs(strikes)
    elif strategy == "strangle":
        specs = _strangle_specs(strikes, spot)
    elif strategy == "iron_condor":
        specs = _iron_condor_specs(strikes, spot)
    else:
        specs = _iron_butterfly_specs(strikes, spot)

    candidates = [
        c for leg_specs in specs if (c := evaluate_candidate(strategy, leg_specs, chain, spot, T, r, q, config, my_vol)) is not None
    ]

    if my_vol:
        candidates.sort(key=lambda c: c.edge_ev if c.edge_ev is not None else float("-inf"), reverse=True)
    else:
        candidates.sort(key=lambda c: c.risk_reward if c.risk_reward is not None else (c.prob_profit or 0.0), reverse=True)

    return candidates[:top_n]


def compare_strategies(
    chain: pd.DataFrame,
    spot: float,
    T: float,
    r: float,
    q: float,
    strike_increment: float,
    num_each_side: int = 6,
    config: RiskConfig = DEFAULT_CONFIG,
    my_vol: float | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
    strike_source: str = "increment",
    oi_fit: DistributionFit | None = None,
    oi_percentiles: tuple[float, ...] = DEFAULT_OI_PERCENTILES,
) -> dict[str, SpreadCandidate | None]:
    """Run find_best_spreads for every strategy in `strategies` against the
    SAME already-fetched chain, and return each one's single best
    candidate (None if that strategy has no valid candidate in this
    strike window — e.g. an iron condor needs listed strikes on both
    sides of spot; a thin/one-sided chain can starve it while straddles
    still work fine).

    This is "what's the single best options trade available right now,
    across structures" — the answer a per-strategy tool can't give without
    the user re-running the search once per strategy by hand.

    strike_source/oi_fit/oi_percentiles pass straight through to
    find_best_spreads — see that function's docstring.

    No extra network I/O: `chain` is a DataFrame the caller already fetched
    once (cached upstream in dashboard.py). Evaluating every strategy here
    costs extra in-process payoff/probability math only — cheap relative to
    the one chain fetch, not a multiplied API cost."""
    return {
        strat: (
            find_best_spreads(
                strat, chain, spot, T, r, q, strike_increment, num_each_side=num_each_side, config=config, my_vol=my_vol,
                top_n=1, strike_source=strike_source, oi_fit=oi_fit, oi_percentiles=oi_percentiles,
            ) or [None]
        )[0]
        for strat in strategies
    }
