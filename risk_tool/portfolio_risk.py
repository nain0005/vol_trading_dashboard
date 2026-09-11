"""Portfolio-level risk: cross-position correlation/covariance, delta-normal
and historical Value-at-Risk, and full-repricing stress tests across a mixed
equity + multi-underlying option book.

Every risk number shown elsewhere in this app describes ONE position at a
time (one option's Greeks, one strike's edge). None of that answers the
question that actually matters once a book has more than one name in it: if
the market moves against you all at once, how much does the WHOLE thing
lose, and how much of the apparent diversification is fake because two
positions are secretly the same bet wearing different tickers? That's what
this module is for.

Two VaR methods are offered side by side deliberately, because they fail in
different, complementary ways:
  - Delta-normal (parametric): assumes returns are jointly normal and that
    P&L is LINEAR in each name's return (a first-order/delta approximation
    -- no gamma, no correlations that spike toward 1 in a real crash). Fast
    and smooth, but understates tail risk for anything with convexity (i.e.
    any options position), and understates co-movement in exactly the
    scenario you'd most want it to be conservative about.
  - Historical simulation: replays each day's ACTUAL historical return per
    name (so real fat tails and real historical co-movement come through
    directly, no normality assumption needed) but is still a linear,
    delta-only repricing per name, and is only as good as the lookback
    window you feed it -- a calm lookback understates risk, a crash-era
    lookback overstates it for a calm regime.

Neither method captures options convexity for a LARGE move -- that's what
the stress-test functions at the bottom are for: they reprice every option
leg exactly via Black-Scholes at the shocked spot/IV instead of linearizing
around today's Greeks, at the cost of only covering the specific scenarios
you ask about rather than a continuous distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_tool.option_strategy import OptionLeg
from risk_tool.pricing import black_scholes_price

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Exposure:
    """One underlying's net dollar sensitivity to its OWN 1.0 (=100%) return
    -- i.e. d(portfolio $)/d(return) -- the "dollar delta" that variance-
    covariance VaR is built on. For a share position this is just
    shares * price; for an option position it's option delta (signed,
    multiplier-adjusted -- the same shares-equivalent convention
    risk_tool.hedge.option_position_exposure_shares uses) * spot. When a
    book holds both stock and options on the same underlying, net the two
    into ONE Exposure per symbol before passing in -- otherwise the same
    name's risk gets double-counted as if it were two independent bets."""

    symbol: str
    dollar_delta: float


def align_returns(price_histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily log returns for every symbol, aligned on date (inner join --
    only dates common to ALL symbols survive). Each history needs 'date'
    and 'close' columns, the same shape get_equity_historicals returns.

    Inner-joining shrinks the usable window down to the shortest common
    history (e.g. a recent IPO drags everyone else's sample down too) --
    that's a real tradeoff, not a bug: a correlation/covariance matrix
    needs every pair measured over the SAME dates, or the numbers aren't
    actually comparable to each other.
    """
    series = {}
    for symbol, hist in price_histories.items():
        if hist is None or hist.empty or len(hist) < 2:
            continue
        closes = hist.set_index("date")["close"].sort_index()
        log_ret = np.log(closes / closes.shift(1)).dropna()
        if not log_ret.empty:
            series[symbol] = log_ret
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, join="inner")


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily log returns -- the honest, detrended
    co-movement signal (see vol_analysis.correlation_stats for why this is
    preferred over price-level correlation, which is inflated by any shared
    trend)."""
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Need aligned return history for at least 2 symbols to build a correlation matrix.")
    return returns.corr()


def covariance_matrix(returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    """Sample covariance of daily log returns. Leave annualize=False for
    VaR use (parametric_var scales the final dollar number by
    sqrt(horizon_days) itself, since VaR wants a specific short horizon,
    not "priced for a year from now"); annualize=True is only for display
    (e.g. comparing against realized_vol's annualized single-name numbers).
    """
    if returns.empty:
        raise ValueError("returns is empty.")
    cov = returns.cov()
    return cov * TRADING_DAYS_PER_YEAR if annualize else cov


_Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def _z_score(confidence: float) -> float:
    if confidence not in _Z_SCORES:
        raise ValueError(f"confidence must be one of {sorted(_Z_SCORES)}, got {confidence}")
    return _Z_SCORES[confidence]


@dataclass
class VarResult:
    method: str
    confidence: float
    horizon_days: int
    portfolio_dollar_vol: float  # std dev of estimated/simulated P&L over the horizon
    var_dollars: float  # POSITIVE number = a loss estimate's magnitude
    diversification_ratio: float | None = None  # var_dollars / sum(|single-name VaRs|); <1 means correlation is netting risk down


def parametric_var(
    exposures: list[Exposure], returns: pd.DataFrame, confidence: float = 0.95, horizon_days: int = 1
) -> VarResult:
    """Delta-normal / variance-covariance VaR.

        portfolio daily $ variance = w^T Sigma w

    where w_i is exposure_i.dollar_delta (d$/dReturn_i) and Sigma is the
    DAILY return covariance matrix -- exactly the same linearization
    risk_tool.hedge uses for a single-name beta hedge (one OLS slope),
    generalized to N correlated names via matrix form. Multiplying by
    sqrt(horizon_days) scales a 1-day sigma to an N-day one under the
    (also approximate) assumption that daily returns are IID -- real
    markets have volatility clustering (see realized_vol.py's GARCH docs),
    so this understates multi-day risk when vol is already elevated.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if not exposures:
        raise ValueError("exposures must be non-empty")
    symbols = [e.symbol for e in exposures]
    missing = [s for s in symbols if s not in returns.columns]
    if missing:
        raise ValueError(f"No return history for: {missing}")

    w = np.array([e.dollar_delta for e in exposures])
    sigma = returns[symbols].cov().to_numpy()
    daily_variance = float(w @ sigma @ w)
    daily_vol = math.sqrt(max(daily_variance, 0.0))
    horizon_vol = daily_vol * math.sqrt(horizon_days)
    z = _z_score(confidence)
    var_dollars = z * horizon_vol

    single_name_vars = [abs(w[i]) * math.sqrt(max(sigma[i, i], 0.0)) * math.sqrt(horizon_days) * z for i in range(len(w))]
    gross = sum(single_name_vars)
    diversification_ratio = (var_dollars / gross) if gross > 0 else None

    return VarResult(
        method="parametric (delta-normal)",
        confidence=confidence,
        horizon_days=horizon_days,
        portfolio_dollar_vol=horizon_vol,
        var_dollars=var_dollars,
        diversification_ratio=diversification_ratio,
    )


def historical_var(
    exposures: list[Exposure], returns: pd.DataFrame, confidence: float = 0.95, horizon_days: int = 1
) -> tuple[VarResult, pd.Series]:
    """Historical-simulation VaR: replay each day's ACTUAL joint return
    across symbols (so real non-normal co-movement -- including the
    "everything correlates to 1" effect in a genuine crash -- comes
    through directly, no normality assumption needed) and linearly reprice
    the CURRENT book each day with TODAY's dollar-delta exposures, i.e.
    "what would this exact book have lost on each of the last N trading
    days." Returns both the VaR estimate and the full simulated daily P&L
    series (e.g. for a histogram). horizon_days scaling is still sqrt(time)
    since only a 1-day empirical distribution is available, so it carries
    the same IID caveat as the parametric method despite using real
    historical returns for the 1-day shape.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if not exposures:
        raise ValueError("exposures must be non-empty")
    symbols = [e.symbol for e in exposures]
    missing = [s for s in symbols if s not in returns.columns]
    if missing:
        raise ValueError(f"No return history for: {missing}")
    if len(returns) < 20:
        raise ValueError("Need at least 20 days of overlapping history for a historical simulation.")

    w = np.array([e.dollar_delta for e in exposures])
    daily_pnl = returns[symbols].to_numpy() @ w
    pnl_series = pd.Series(daily_pnl, index=returns.index, name="simulated_daily_pnl")

    loss_percentile = (1 - confidence) * 100
    one_day_var = -float(np.percentile(pnl_series, loss_percentile))
    horizon_var = max(one_day_var, 0.0) * math.sqrt(horizon_days)
    daily_vol = float(pnl_series.std(ddof=1))

    result = VarResult(
        method="historical simulation",
        confidence=confidence,
        horizon_days=horizon_days,
        portfolio_dollar_vol=daily_vol * math.sqrt(horizon_days),
        var_dollars=horizon_var,
    )
    return result, pnl_series


def equity_stress_pl(shares: float, spot: float, shock_pct: float) -> float:
    """Equity P&L under a spot shock -- exact, since equity payoff is
    already linear in price (no repricing model needed, unlike options)."""
    return shares * spot * shock_pct


def option_leg_stress_pl(
    leg: OptionLeg,
    spot: float,
    T_years: float,
    shock_pct: float,
    iv_shock_pts: float = 0.0,
    r: float = 0.05,
    q: float = 0.0,
) -> float:
    """P&L of one option leg between now and a shocked (spot, IV) scenario
    -- Black-Scholes value at the shocked point minus Black-Scholes value
    right now, both at the leg's OWN iv (shifted by iv_shock_pts for the
    shocked side only). This is a full repricing, not a Greeks-based
    approximation, so it captures gamma/vega convexity that the linear
    parametric/historical VaR above misses entirely for a large move --
    the whole reason this function exists alongside them.

    Requires leg.iv (current IV) to be set. At/past expiration (T_years<=0)
    IV no longer applies to value -- both sides fall back to intrinsic
    value and iv_shock_pts is ignored, since there's no time value left to
    shock.
    """
    if leg.iv is None:
        raise ValueError(f"leg at strike {leg.strike} has no iv set -- stress testing needs current IV")
    shocked_spot = spot * (1 + shock_pct)

    if T_years <= 0:
        current_value = _intrinsic(spot, leg.strike, leg.option_type)
        shocked_value = _intrinsic(shocked_spot, leg.strike, leg.option_type)
    else:
        current_value = black_scholes_price(spot, leg.strike, T_years, r, q, leg.iv, leg.option_type)
        shocked_iv = max(leg.iv + iv_shock_pts, 0.001)
        shocked_value = black_scholes_price(shocked_spot, leg.strike, T_years, r, q, shocked_iv, leg.option_type)

    return (shocked_value - current_value) * leg.contracts * leg.shares_per_contract


def _intrinsic(spot: float, strike: float, option_type: str) -> float:
    return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
