"""Expected-value strike ranking, and the realized-vs-implied comparison
that's the actual source of edge in this tool.

IMPORTANT — read this before trusting any number this module produces:

The "EV" here is NOT the option's true expected payoff at expiry. It's the
expected value of a specific, mechanical plan: pay the premium, and exit
at either your own profit-target or your own stop-loss rule (the same
rule risk_manager.py enforces). That's a deliberate choice — it makes the
EV model consistent with how you actually trade, instead of an idealized
"hold to expiry" payoff nobody following a discipline system actually
takes. The tradeoff: max_gain/risk_reward here are relative to YOUR
profit target, not the option's theoretical (unbounded, for a call)
upside — don't confuse the two.

P_win = N(d2) (or N(-d2) for puts) is a RISK-NEUTRAL probability: it's
what the option's price implies about the odds of finishing ITM if the
stock drifted at the risk-free rate and the given sigma were the true
vol. It is NOT a real-world forecast, and by itself it doesn't tell you
whether a trade has positive real-world expected value — pricing built on
market IV is, by construction, close to a fair (zero-edge) bet before
frictions.

The actual, honest edge in this framework can only come from ONE place:
believing your own volatility estimate (realized vol, or a GARCH
forecast) is a better estimate of what will actually happen than the
market's IV. `compare_implied_vs_realized_ev` makes that comparison
explicit and first-class, rather than burying it as a side note.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk_tool.config import DEFAULT_CONFIG, RiskConfig
from risk_tool.greeks import all_greeks
from risk_tool.pricing import OptionType, black_scholes_price, itm_probability


@dataclass
class StrikeEV:
    strike: float
    option_type: OptionType
    premium: float
    p_win_risk_neutral: float  # N(d2)/N(-d2) at the sigma actually passed in — LABEL AS RISK-NEUTRAL, not a forecast
    profit_if_win: float  # premium * profit_target_pct — profit AT YOUR OWN TARGET, not at expiry
    max_loss: float  # = premium (worst case for a long option)
    ev: float
    risk_reward: float  # profit_if_win / max_loss, relative to your own target — see module docstring
    is_poor_setup: bool  # risk_reward below config.min_risk_reward_ratio
    delta: float
    gamma: float
    vega: float
    theta: float


def evaluate_strike(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    config: RiskConfig = DEFAULT_CONFIG,
    profit_target_pct: float | None = None,
) -> StrikeEV:
    """EV for one strike: EV = P_win * profit_if_win - P_lose * premium.

    `sigma` is whatever vol you pass in — market IV for the risk-neutral
    view, or your own realized/forecast vol for the "fair" view (see
    compare_implied_vs_realized_ev). The premium itself always uses
    whichever sigma is passed in too, so if you want "market premium, my
    probability," call this twice with the same premium held fixed
    manually (compare_implied_vs_realized_ev does exactly that).
    """
    target_pct = profit_target_pct if profit_target_pct is not None else config.default_profit_target_pct

    premium = black_scholes_price(S, K, T, r, q, sigma, option_type)
    p_win = itm_probability(S, K, T, r, q, sigma, option_type)
    p_lose = 1.0 - p_win

    profit_if_win = premium * target_pct
    max_loss = premium
    ev = p_win * profit_if_win - p_lose * max_loss
    risk_reward = profit_if_win / max_loss if max_loss > 0 else float("inf")
    greeks = all_greeks(S, K, T, r, q, sigma, option_type)

    return StrikeEV(
        strike=K,
        option_type=option_type,
        premium=premium,
        p_win_risk_neutral=p_win,
        profit_if_win=profit_if_win,
        max_loss=max_loss,
        ev=ev,
        risk_reward=risk_reward,
        is_poor_setup=bool(risk_reward < config.min_risk_reward_ratio),
        delta=greeks["delta"],
        gamma=greeks["gamma"],
        vega=greeks["vega"],
        theta=greeks["theta"],
    )


def generate_candidate_strikes(spot: float, strike_increment: float, num_each_side: int) -> list[float]:
    """ATM strike (spot rounded to the nearest increment) plus num_each_side
    strikes above and below it."""
    if strike_increment <= 0:
        raise ValueError("strike_increment must be positive")
    if num_each_side < 0:
        raise ValueError("num_each_side must be >= 0")
    atm = round(spot / strike_increment) * strike_increment
    return sorted(atm + i * strike_increment for i in range(-num_each_side, num_each_side + 1))


def rank_strikes_by_ev(
    S: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    strikes: list[float],
    config: RiskConfig = DEFAULT_CONFIG,
    profit_target_pct: float | None = None,
    max_premium_dollars: float | None = None,
    contract_multiplier: int = 100,
) -> list[StrikeEV]:
    """Evaluate every candidate strike and return them ranked by EV,
    descending. If max_premium_dollars is given (e.g. your position-size
    cap for one contract), strikes whose premium * multiplier exceeds it
    are dropped entirely rather than just flagged — you can't take them at
    that size regardless of how good their EV looks.
    """
    results = []
    for K in strikes:
        strike_ev = evaluate_strike(S, K, T, r, q, sigma, option_type, config, profit_target_pct)
        if max_premium_dollars is not None and strike_ev.premium * contract_multiplier > max_premium_dollars:
            continue
        results.append(strike_ev)
    return sorted(results, key=lambda s: s.ev, reverse=True)


@dataclass
class RealizedVsImpliedEV:
    strike: float
    option_type: OptionType
    market_premium: float  # priced at market IV — this is what you actually pay
    market_iv: float
    realized_or_forecast_vol: float
    vol_spread: float  # realized_or_forecast_vol - market_iv; positive = you think it'll be MORE volatile than priced
    risk_neutral_ev: StrikeEV  # EV using market IV for both price and probability (the "textbook", ~zero-edge view)
    edge_ev: StrikeEV  # EV using YOUR vol for probability, but the market's actual premium — this is the edge check


def compare_implied_vs_realized_ev(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_iv: float,
    realized_or_forecast_vol: float,
    option_type: OptionType,
    config: RiskConfig = DEFAULT_CONFIG,
    profit_target_pct: float | None = None,
) -> RealizedVsImpliedEV:
    """The first-class realized-vs-implied comparison.

    risk_neutral_ev prices AND assigns probability using market_iv — this
    is close to a fair-value view by construction (options pricing is
    close to zero-edge before frictions when your vol assumption matches
    the market's).

    edge_ev keeps the premium you'd ACTUALLY pay (priced at market_iv,
    since that's the real quote) but assigns win probability using YOUR
    OWN volatility estimate. If your realized/forecast vol differs
    meaningfully from market IV, this is where a real, quantifiable edge
    (or a real, quantifiable reason to avoid the trade) shows up. This is
    the actual edge this tool can offer — everything else is bookkeeping
    and discipline enforcement.
    """
    market_premium = black_scholes_price(S, K, T, r, q, market_iv, option_type)

    risk_neutral_ev = evaluate_strike(S, K, T, r, q, market_iv, option_type, config, profit_target_pct)

    # edge_ev: probability from realized/forecast vol, but priced (premium,
    # profit_if_win, max_loss) off the market's actual premium.
    target_pct = profit_target_pct if profit_target_pct is not None else config.default_profit_target_pct
    p_win_edge = itm_probability(S, K, T, r, q, realized_or_forecast_vol, option_type)
    p_lose_edge = 1.0 - p_win_edge
    profit_if_win = market_premium * target_pct
    edge_ev_value = p_win_edge * profit_if_win - p_lose_edge * market_premium
    edge_greeks = all_greeks(S, K, T, r, q, market_iv, option_type)  # Greeks quoted off the tradable (market-IV) contract
    edge_ev = StrikeEV(
        strike=K,
        option_type=option_type,
        premium=market_premium,
        p_win_risk_neutral=p_win_edge,  # computed with YOUR vol — see field docstring caveat, still risk-neutral-style math
        profit_if_win=profit_if_win,
        max_loss=market_premium,
        ev=edge_ev_value,
        risk_reward=profit_if_win / market_premium if market_premium > 0 else float("inf"),
        is_poor_setup=bool((profit_if_win / market_premium if market_premium > 0 else float("inf")) < config.min_risk_reward_ratio),
        delta=edge_greeks["delta"],
        gamma=edge_greeks["gamma"],
        vega=edge_greeks["vega"],
        theta=edge_greeks["theta"],
    )

    return RealizedVsImpliedEV(
        strike=K,
        option_type=option_type,
        market_premium=market_premium,
        market_iv=market_iv,
        realized_or_forecast_vol=realized_or_forecast_vol,
        vol_spread=realized_or_forecast_vol - market_iv,
        risk_neutral_ev=risk_neutral_ev,
        edge_ev=edge_ev,
    )
