"""How often does a stock actually move +/-2sigma or +/-3sigma, and is it
"overdue" by its own history? Two genuinely different questions bundled
into one screener -- read the caveat below before trusting the "overdue"
framing on its own.

sigma here is a TRAILING ROLLING estimate (default 20 trading days),
shifted by one day so day t's own return never leaks into the sigma used
to judge day t (no look-ahead). Same close-to-close estimator as
realized_vol.close_to_close_vol, just computed on a rolling window
instead of the whole sample, and left un-annualized since a z-score
needs daily sigma in the same units as the daily return.

IMPORTANT CAVEAT -- read before trusting the "overdue" framing:
"It's been longer than average since the last 2-sigma move" does NOT, on
its own, mean tomorrow is more likely to be one. If exceedance days were
truly i.i.d. (a fixed per-day probability, independent across days --
the assumption baked into both the empirical and model-implied
recurrence numbers below), being overdue changes nothing: a memoryless
process has no memory, the same fallacy as expecting a coin to owe you
tails after ten heads.

What COULD make "overdue" meaningful is different: volatility itself is
mean-reverting and clusters (this is exactly what GARCH models capture).
A long stretch with no big move often means realized vol has genuinely
been unusually LOW, and low-vol regimes tend to eventually revert toward
the stock's own long-run average -- that IS a real, evidence-based reason
to expect vol (and the odds of a big move) to pick back up. The correct
signal for that is current_vol_ratio below (< 1 = compressed), not the
raw days-since-last count by itself. Read them together; the recurrence
gap alone is not a probability forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ROLLING_WINDOW = 20


def rolling_daily_sigma(close: pd.Series, window: int = DEFAULT_ROLLING_WINDOW) -> pd.Series:
    """Trailing rolling std of daily log returns, shifted by one day so
    sigma for day t never includes day t's own return (no look-ahead)."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).std(ddof=1).shift(1)


def zscores(close: pd.Series, window: int = DEFAULT_ROLLING_WINDOW) -> pd.Series:
    """Each day's log return divided by its trailing rolling sigma -- how
    many standard deviations of RECENT realized vol (not full-sample)
    that day's move was."""
    log_returns = np.log(close / close.shift(1))
    sigma = rolling_daily_sigma(close, window)
    return log_returns / sigma


@dataclass
class SigmaMoveStats:
    symbol: str
    threshold: float
    n_observations: int
    n_exceedances: int
    empirical_avg_days_between: float | None  # None if fewer than 2 exceedances observed
    fitted_t_df: float  # degrees of freedom from the Student-t fit to the z-score series -- lower = fatter tails
    model_avg_days_between: float  # from the fitted distribution's tail probability -- always defined, even with 0-1 exceedances
    days_since_last_exceedance: int | None  # None if no exceedance ever observed in the sample
    current_vol_ratio: float | None  # trailing `window`-day realized vol / full-sample realized vol; <1 = compressed now
    overdue_ratio: float | None  # days_since_last / empirical_avg_days_between; >1 = longer than the historical average gap (see module caveat -- not a forecast on its own)


def analyze_symbol(
    symbol: str, close: pd.Series, threshold: float = 2.0, window: int = DEFAULT_ROLLING_WINDOW
) -> SigmaMoveStats | None:
    """None if there isn't enough price history to compute a rolling sigma
    at all (fewer than window+2 usable closes)."""
    z = zscores(close, window).dropna()
    n_obs = len(z)
    if n_obs < 5:
        return None

    exceed = z.abs() > threshold
    n_exceed = int(exceed.sum())
    exceed_positions = np.flatnonzero(exceed.to_numpy())

    empirical_avg = float(np.diff(exceed_positions).mean()) if len(exceed_positions) >= 2 else None
    days_since_last = int(n_obs - 1 - exceed_positions[-1]) if len(exceed_positions) >= 1 else None

    # Student-t fit to the z-score series itself (already sigma-normalized,
    # so location should sit near 0 for a well-behaved name) -- fix loc=0
    # so the fit stays meaningful even on shorter samples.
    df_fit, loc_fit, scale_fit = stats.t.fit(z.to_numpy(), floc=0.0)
    tail_p = 2.0 * stats.t.sf(threshold, df=df_fit, loc=loc_fit, scale=scale_fit)
    model_avg = (1.0 / tail_p) if tail_p > 0 else float("inf")

    log_returns = np.log(close / close.shift(1)).dropna()
    recent_vol = float(log_returns.tail(window).std(ddof=1)) if len(log_returns) >= window else None
    full_vol = float(log_returns.std(ddof=1))
    vol_ratio = (recent_vol / full_vol) if (recent_vol is not None and full_vol > 0) else None

    overdue_ratio = (days_since_last / empirical_avg) if (days_since_last is not None and empirical_avg) else None

    return SigmaMoveStats(
        symbol=symbol, threshold=threshold, n_observations=n_obs, n_exceedances=n_exceed,
        empirical_avg_days_between=empirical_avg, fitted_t_df=float(df_fit), model_avg_days_between=model_avg,
        days_since_last_exceedance=days_since_last, current_vol_ratio=vol_ratio, overdue_ratio=overdue_ratio,
    )
