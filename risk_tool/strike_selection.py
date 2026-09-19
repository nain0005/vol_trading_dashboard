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
from risk_tool.option_strategy import OptionLeg, analyze_strategy
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


# ---------------------------------------------------------------------------
# Short premium (income) strikes: selling a naked put or call instead of
# buying one. This is NOT a mirror image of the long side with signs
# flipped — read the dataclass docstring before using mechanical_max_loss
# or theoretical_max_loss for anything.
# ---------------------------------------------------------------------------


@dataclass
class ShortStrikeEV:
    """EV for selling (writing) one naked strike.

    Two different "max loss" numbers are reported here, and conflating
    them is the single easiest way to misuse this dataclass:

    - `mechanical_max_loss` is the loss booked under YOUR OWN pre-committed
      exit rule: buy the contract back once its price grows to
      config.short_stop_loss_multiple times the premium you received
      (default 2x — i.e. stop out once you'd be paying back twice what you
      collected). This is the number `ev`, `risk_reward`, and
      `is_poor_setup` are actually computed from, and it's the number
      `sizing.size_short_position` needs — it plays the same role for a
      short position that `premium` (max loss for a long) plays for a
      long one: a FINITE, rule-defined amount at risk you're actually
      committing to respect.
    - `theoretical_max_loss` is the option's true worst case if you never
      exit, as a POSITIVE dollar magnitude (matching mechanical_max_loss's
      convention, and the long side's `max_loss = premium` convention —
      NOT option_strategy.StrategyProfile.max_loss's own sign convention,
      where a loss is negative; this field is renormalized from that):
      None (genuinely, mathematically unbounded — the underlying has no
      ceiling) for a short call, or strike - premium_received for a
      short put (finite only because the underlying can't trade below
      $0). It is NOT used anywhere in the EV/Kelly math below — it exists
      purely so this number is never hidden from you. A short call's
      unbounded risk does not become bounded just because you plan to
      follow a stop-loss rule; a rule you fail to execute (a gap through
      your stop, a halted underlying, an assignment) leaves the true
      exposure open. Treat mechanical_max_loss as "risk under discipline,"
      not "worst case."
    """

    strike: float
    option_type: OptionType
    premium_received: float  # per share, credited immediately at entry (this is what you receive, not what you risk)
    p_win_risk_neutral: float  # risk-neutral probability of expiring OTM (i.e. keeping the premium) — see module docstring caveats on risk-neutral probabilities
    profit_if_win: float  # premium_received * min(profit_target_pct, 1.0) — capped, since you can never keep more than 100% of a credit
    mechanical_max_loss: float  # loss at YOUR stop-loss rule — see class docstring; this is what ev/risk_reward/sizing use
    theoretical_max_loss: float | None  # true worst case at expiration; None = unbounded (short call). NOT used in ev/risk_reward/sizing.
    ev: float
    risk_reward: float  # profit_if_win / mechanical_max_loss
    is_poor_setup: bool
    delta: float  # POSITION greeks (short = negated from the raw per-contract greeks), matching how option_positions delta is signed elsewhere in this codebase
    gamma: float
    vega: float
    theta: float


def evaluate_short_strike(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    config: RiskConfig = DEFAULT_CONFIG,
    profit_target_pct: float | None = None,
) -> ShortStrikeEV:
    """EV for selling one strike (naked short put or naked short call).

    `sigma` plays the same role it does in evaluate_strike: whatever vol
    you pass in prices the premium AND sets the win probability. Pass
    market IV for the risk-neutral view; see
    compare_implied_vs_realized_ev_short for the realized-vs-implied edge
    check on the short side.

    Win probability is the COMPLEMENT of the long side's ITM probability:
    a short option "wins" (keeps the premium) when it finishes OTM, and
    P(OTM) = 1 - P(ITM) = N(-d2) for a call / N(d2) for a put — which is
    exactly itm_probability() called with the OPPOSITE option_type at the
    same strike/sigma (N(d2)+N(-d2)=1 by construction), so this reuses
    itm_probability rather than deriving a new formula.
    """
    target_pct = profit_target_pct if profit_target_pct is not None else config.default_profit_target_pct
    # A short position can never keep more than 100% of the credit it
    # received (the option's value has a floor of $0) — unlike a long
    # position, where profit_target_pct=1.0 means "the premium doubled,"
    # not "you collected the whole premium." Capping here, rather than
    # silently letting a >1.0 target imply an impossible >100%-of-credit
    # profit, is the honest behavior.
    capped_target_pct = min(target_pct, 1.0)

    premium = black_scholes_price(S, K, T, r, q, sigma, option_type)
    flipped_type: OptionType = "put" if option_type == "call" else "call"
    p_win = itm_probability(S, K, T, r, q, sigma, flipped_type)
    p_lose = 1.0 - p_win

    if config.short_stop_loss_multiple <= 1.0:
        raise ValueError(f"config.short_stop_loss_multiple must be > 1.0, got {config.short_stop_loss_multiple}")
    profit_if_win = premium * capped_target_pct
    mechanical_max_loss = premium * (config.short_stop_loss_multiple - 1.0)

    short_leg = OptionLeg(option_type=option_type, strike=K, premium=premium, contracts=-1.0, shares_per_contract=1.0)
    raw_max_loss = analyze_strategy([short_leg]).max_loss  # option_strategy's own convention: a signed P&L, negative = a loss
    # Renormalize to a positive magnitude, matching mechanical_max_loss's convention and the long side's
    # StrikeEV.max_loss ("= premium", also a positive magnitude) — None stays None (genuinely unbounded).
    theoretical_max_loss = -raw_max_loss if raw_max_loss is not None else None

    ev = p_win * profit_if_win - p_lose * mechanical_max_loss
    risk_reward = profit_if_win / mechanical_max_loss if mechanical_max_loss > 0 else float("inf")
    greeks = all_greeks(S, K, T, r, q, sigma, option_type)

    return ShortStrikeEV(
        strike=K,
        option_type=option_type,
        premium_received=premium,
        p_win_risk_neutral=p_win,
        profit_if_win=profit_if_win,
        mechanical_max_loss=mechanical_max_loss,
        theoretical_max_loss=theoretical_max_loss,
        ev=ev,
        risk_reward=risk_reward,
        is_poor_setup=bool(risk_reward < config.min_risk_reward_ratio),
        delta=-greeks["delta"],
        gamma=-greeks["gamma"],
        vega=-greeks["vega"],
        theta=-greeks["theta"],
    )


def rank_short_strikes_by_ev(
    S: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
    strikes: list[float],
    config: RiskConfig = DEFAULT_CONFIG,
    profit_target_pct: float | None = None,
) -> list[ShortStrikeEV]:
    """Evaluate every candidate strike for selling premium and return them
    ranked by EV, descending. No premium cap here (unlike
    rank_strikes_by_ev) — selling premium credits your account rather than
    costing you money up front, so there's no "can't afford it" filter
    analogous to the long side's max_premium_dollars."""
    results = [evaluate_short_strike(S, K, T, r, q, sigma, option_type, config, profit_target_pct) for K in strikes]
    return sorted(results, key=lambda s: s.ev, reverse=True)


@dataclass
class ShortRealizedVsImpliedEV:
    strike: float
    option_type: OptionType
    market_premium: float  # priced at market IV — this is what you'd actually receive
    market_iv: float
    realized_or_forecast_vol: float
    vol_spread: float  # realized_or_forecast_vol - market_iv
    risk_neutral_ev: ShortStrikeEV  # EV using market IV for both price and probability (~zero-edge by construction)
    edge_ev: ShortStrikeEV  # EV using YOUR vol for probability, market's actual premium — the edge check


def compare_implied_vs_realized_ev_short(
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
) -> ShortRealizedVsImpliedEV:
    """The short-premium analog of compare_implied_vs_realized_ev.

    Selling premium has an edge when YOUR vol estimate is LOWER than
    market IV (you believe the market is overpricing how much the
    underlying will actually move, so the credit you collect is more than
    fair compensation for the real risk) — the mirror image of the long
    side, where edge comes from your vol being HIGHER than market IV.
    Nothing here special-cases that direction, though: edge_ev's sign
    falls out of the same math regardless of which way vol_spread points,
    exactly like the long side.
    """
    market_premium = black_scholes_price(S, K, T, r, q, market_iv, option_type)
    risk_neutral_ev = evaluate_short_strike(S, K, T, r, q, market_iv, option_type, config, profit_target_pct)

    target_pct = profit_target_pct if profit_target_pct is not None else config.default_profit_target_pct
    capped_target_pct = min(target_pct, 1.0)
    flipped_type: OptionType = "put" if option_type == "call" else "call"
    p_win_edge = itm_probability(S, K, T, r, q, realized_or_forecast_vol, flipped_type)
    p_lose_edge = 1.0 - p_win_edge

    if config.short_stop_loss_multiple <= 1.0:
        raise ValueError(f"config.short_stop_loss_multiple must be > 1.0, got {config.short_stop_loss_multiple}")
    profit_if_win = market_premium * capped_target_pct
    mechanical_max_loss = market_premium * (config.short_stop_loss_multiple - 1.0)

    short_leg = OptionLeg(option_type=option_type, strike=K, premium=market_premium, contracts=-1.0, shares_per_contract=1.0)
    raw_max_loss = analyze_strategy([short_leg]).max_loss
    theoretical_max_loss = -raw_max_loss if raw_max_loss is not None else None

    edge_ev_value = p_win_edge * profit_if_win - p_lose_edge * mechanical_max_loss
    edge_greeks = all_greeks(S, K, T, r, q, market_iv, option_type)  # Greeks quoted off the tradable (market-IV) contract
    risk_reward = profit_if_win / mechanical_max_loss if mechanical_max_loss > 0 else float("inf")
    edge_ev = ShortStrikeEV(
        strike=K,
        option_type=option_type,
        premium_received=market_premium,
        p_win_risk_neutral=p_win_edge,  # computed with YOUR vol — see field docstring caveat, still risk-neutral-style math
        profit_if_win=profit_if_win,
        mechanical_max_loss=mechanical_max_loss,
        theoretical_max_loss=theoretical_max_loss,
        ev=edge_ev_value,
        risk_reward=risk_reward,
        is_poor_setup=bool(risk_reward < config.min_risk_reward_ratio),
        delta=-edge_greeks["delta"],
        gamma=-edge_greeks["gamma"],
        vega=-edge_greeks["vega"],
        theta=-edge_greeks["theta"],
    )

    return ShortRealizedVsImpliedEV(
        strike=K,
        option_type=option_type,
        market_premium=market_premium,
        market_iv=market_iv,
        realized_or_forecast_vol=realized_or_forecast_vol,
        vol_spread=realized_or_forecast_vol - market_iv,
        risk_neutral_ev=risk_neutral_ev,
        edge_ev=edge_ev,
    )
