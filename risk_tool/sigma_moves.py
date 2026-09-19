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


@dataclass
class CrashProbability:
    direction: str  # "down" (a "crash"), "up", or "either"
    threshold: float
    forward_days: int
    current_run_length: int  # trading days since the last exceedance (0 = today itself was one)
    n_observations: int
    baseline_daily_prob: float | None  # unconditional P(exceedance on any given day) = n_exceed / n_obs
    baseline_prob_within_horizon: float | None  # memoryless 1-(1-p)^forward_days -- the "if this were pure chance" baseline
    empirical_conditional_prob_within_horizon: float | None  # the actual answer to the user's question -- see docstring
    n_historical_analogs: int  # how many historical points had a quiet run >= current_run_length -- sample size behind the conditional number


def crash_probability_given_quiet_run(
    close: pd.Series,
    threshold: float = 1.0,
    direction: str = "down",
    forward_days: int = 5,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> CrashProbability | None:
    """Answers the actual question "given it HASN'T moved > threshold-sigma
    in a while, what's the empirical chance it does within the next
    `forward_days` -- and is that actually higher than pure chance, or is
    the drought meaningless?" Two numbers, deliberately shown side by side
    rather than collapsed into one:

    - baseline_prob_within_horizon: what you'd expect if exceedance days
      were i.i.d. (a coin with the stock's own overall hit rate, flipped
      fresh every day, no memory of the drought at all) -- the
      pure-chance / memoryless null hypothesis.
    - empirical_conditional_prob_within_horizon: what ACTUALLY happened,
      historically, specifically after quiet runs at least as long as the
      one happening right now. Computed by walking every day in the
      series, finding every point where a quiet streak of at least
      current_run_length had just been reached, and checking what
      fraction of THOSE historical analogs saw an exceedance within the
      next `forward_days` days.

    If the empirical number sits well above the baseline, that's real
    evidence -- for THIS stock, THIS threshold -- that quiet spells
    historically preceded a move more often than chance alone would
    predict (consistent with vol mean-reversion, see the module-level
    caveat). If it's close to or below baseline, the drought has been
    genuinely uninformative for this name and the "it's overdue" instinct
    isn't supported by its own history. Either answer is a real answer;
    this function doesn't assume which one you'll get.

    Caveats that don't go away just because this is now empirical:
    - The historical analog windows OVERLAP (day 50 and day 51 both being
      "10+ quiet days" share 9 of the same days), so n_historical_analogs
      is not a count of independent observations -- treat it as "how much
      raw historical evidence," not a textbook sample size. Small values
      (rule of thumb: under ~20) mean the empirical number is noisy;
      that's surfaced explicitly rather than hidden.
    - This is still just pattern-matching this one stock's own past --
      it's not a causal model and it can't see genuinely new information
      (an unannounced earnings date, a macro regime change) that has no
      precedent in the lookback window.

    None if there's no exceedance in the sample at all (a baseline rate
    of exactly 0 makes "conditional on how long since the last one"
    undefined) or too little price history to compute sigma at all.
    """
    if direction not in ("down", "up", "either"):
        raise ValueError(f"direction must be 'down', 'up', or 'either', got {direction!r}")

    z = zscores(close, window).dropna()
    n_obs = len(z)
    if n_obs < 5:
        return None

    z_arr = z.to_numpy()
    if direction == "down":
        exceed = z_arr <= -threshold
    elif direction == "up":
        exceed = z_arr >= threshold
    else:
        exceed = np.abs(z_arr) >= threshold

    result = _crash_probability_from_exceedances(exceed, forward_days)
    if result is None:
        return None
    baseline_daily_prob, baseline_within_horizon, current_run, n_analogs, empirical_prob = result

    return CrashProbability(
        direction=direction, threshold=threshold, forward_days=forward_days, current_run_length=current_run,
        n_observations=n_obs, baseline_daily_prob=baseline_daily_prob, baseline_prob_within_horizon=baseline_within_horizon,
        empirical_conditional_prob_within_horizon=empirical_prob, n_historical_analogs=n_analogs,
    )


def _crash_probability_from_exceedances(exceed: np.ndarray, forward_days: int):
    """Pure boolean-array core of crash_probability_given_quiet_run, split
    out so the conditional-probability logic can be hand-verified directly
    against a constructed True/False pattern without needing to reverse-
    engineer a price series that produces specific z-scores. None if there
    are no exceedances in the array at all (undefined baseline rate)."""
    n_obs = len(exceed)
    n_exceed = int(exceed.sum())
    if n_exceed == 0:
        return None
    baseline_daily_prob = n_exceed / n_obs
    baseline_within_horizon = 1.0 - (1.0 - baseline_daily_prob) ** forward_days

    # Current run length: consecutive quiet (non-exceedance) days counting
    # back from the most recent observation. 0 means today itself exceeded.
    current_run = 0
    for exceeded_that_day in exceed[::-1]:
        if exceeded_that_day:
            break
        current_run += 1

    # Every historical point i where a quiet run of AT LEAST current_run
    # days had just been reached (i.e. day i and the current_run-1 days
    # before it are all quiet) is one analog. Among those, what fraction
    # saw an exceedance within the next forward_days days?
    #
    # current_run == 0 needs its own branch, not just the general window
    # formula with current_run=0 plugged in: that formula slices
    # exceed[i+1:i+1] for every i, which is an EMPTY slice regardless of
    # i -- meaning it passes vacuously (day i doesn't even get checked)
    # and every single day in history ends up "qualifying," exceedance or
    # not. current_run == 0 specifically means today WAS an exceedance,
    # so the matching historical state is "a day where an exceedance ALSO
    # just happened" -- i.e. exceed[i] is True -- not "any day at all."
    n_analogs = 0
    n_hits = 0
    if current_run == 0:
        qualifying_days = np.flatnonzero(exceed)
    else:
        qualifying_days = [
            i for i in range(current_run - 1, n_obs)
            if not exceed[i - current_run + 1 : i + 1].any()
        ]
    for i in qualifying_days:
        n_analogs += 1
        future = exceed[i + 1 : i + 1 + forward_days]
        if future.any():
            n_hits += 1

    empirical_prob = (n_hits / n_analogs) if n_analogs > 0 else None
    return baseline_daily_prob, baseline_within_horizon, current_run, n_analogs, empirical_prob
