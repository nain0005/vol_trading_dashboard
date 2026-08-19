"""Realized/historical volatility estimators, plus GARCH(1,1) and
EGARCH(1,1) forecasts — the alternative to market IV that
strike_selection.py compares against to find the actual realized-vs-implied
edge.

Three close-to-close/range estimators, in increasing order of statistical
efficiency (lower variance for the same sample size), because they use
progressively more of each day's price action:
    - close-to-close: only closing prices
    - Parkinson: adds the day's high/low range
    - Garman-Klass: adds open too, and is the most efficient of the three
      for a pure GBM process with no drift or jumps

Plus two conditional-vol models that, unlike the three estimators above,
produce a forward-looking forecast rather than a single backward-looking
number:
    - GARCH(1,1) (`fit_garch_11`): symmetric — a big move up and a big move
      down of the same size raise forecast vol by the same amount.
    - EGARCH(1,1) (`fit_egarch_11`): adds the "leverage effect" — equity
      vol empirically rises more after a down move than after an up move
      of the same size — via an asymmetry term, and models log-variance
      directly so it needs no positivity constraints on its parameters.

Both accept `dist="normal"` (default) or `dist="t"` innovations. Real
equity returns have fatter tails than a normal distribution — large moves
are more common than a Gaussian would predict — so `dist="t"` fits a
Student-t degrees-of-freedom parameter (`nu`) alongside the variance
parameters by maximum likelihood; a low fitted `nu` (roughly < 10) is
itself a diagnostic that tails matter for that name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

TRADING_DAYS_PER_YEAR = 252


def close_to_close_vol(close: pd.Series, annualization_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized std of log returns — the simplest, noisiest, and most
    common realized-vol estimator. Ignores everything except closing
    prices, so it throws away the intraday range information the other two
    estimators use."""
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < 2:
        raise ValueError("Need at least 3 closing prices to estimate volatility.")
    return float(log_returns.std(ddof=1) * math.sqrt(annualization_factor))


def parkinson_vol(high: pd.Series, low: pd.Series, annualization_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """Parkinson (1980) range-based estimator:

        sigma^2 = (1 / (4*n*ln2)) * sum(ln(H_i/L_i)^2)

    More efficient than close-to-close because the day's high/low range
    carries information about volatility that happened between the close
    prices. Assumes no overnight gaps/drift and continuous trading, so it
    tends to underestimate vol on gappy names.
    """
    if len(high) != len(low):
        raise ValueError("high and low must be the same length")
    n = len(high)
    if n < 1:
        raise ValueError("Need at least 1 day of high/low data.")
    log_hl = np.log(high.to_numpy() / low.to_numpy())
    variance = float(np.sum(log_hl**2) / (4 * n * math.log(2)))
    return math.sqrt(variance * annualization_factor)


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Garman-Klass (1980) estimator — adds the open/close relationship on
    top of Parkinson's high/low range:

        sigma^2 = (1/n) * sum[ 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2 ]

    The most statistically efficient of the three here under the assumption
    of continuous GBM with zero drift — but that assumption (no jumps, no
    overnight drift) is exactly what real markets violate around earnings,
    gaps, and trend days, so treat it as one estimate among several, not
    ground truth.
    """
    n = len(close)
    if not (len(open_) == len(high) == len(low) == n):
        raise ValueError("open, high, low, close must all be the same length")
    if n < 1:
        raise ValueError("Need at least 1 day of OHLC data.")

    log_hl = np.log(high.to_numpy() / low.to_numpy())
    log_co = np.log(close.to_numpy() / open_.to_numpy())
    term = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
    variance = float(np.sum(term) / n)
    if variance < 0:
        raise ValueError(
            "Garman-Klass variance came out negative — this can happen with "
            "very few, unusual-looking bars (e.g. O=H=L=C). Use more data."
        )
    return math.sqrt(variance * annualization_factor)


@dataclass
class GarchFit:
    """A fitted GARCH(1,1): sigma_t^2 = omega + alpha*r_{t-1}^2 + beta*sigma_{t-1}^2."""

    omega: float
    alpha: float
    beta: float
    mean_return: float
    demeaned_returns: np.ndarray = field(repr=False)
    conditional_variance: np.ndarray = field(repr=False)  # fitted sigma_t^2, same length as demeaned_returns

    @property
    def long_run_variance(self) -> float:
        """omega / (1 - alpha - beta) — the variance GARCH reverts to as the
        forecast horizon grows, given alpha+beta < 1 (stationarity)."""
        return self.omega / (1 - self.alpha - self.beta)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def fit_garch_11(returns: pd.Series) -> GarchFit:
    """Fit GARCH(1,1) by maximum likelihood (Gaussian conditional density).

    Returns are demeaned first — GARCH models the variance of the
    *unexpected* return, not the level. The negative log-likelihood is:

        NLL = 0.5 * sum( ln(2*pi*sigma_t^2) + r_t^2 / sigma_t^2 )

    minimized subject to omega > 0, alpha >= 0, beta >= 0, alpha+beta < 1
    (the stationarity condition — without it the variance process doesn't
    revert to a long-run level and 'annualized vol' stops being meaningful).
    """
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 30:
        raise ValueError("GARCH(1,1) needs a reasonably long return history (30+ points) to fit sensibly.")
    mean_r = float(r.mean())
    r = r - mean_r
    sample_var = float(r.var(ddof=1))

    def neg_log_likelihood(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10  # infeasible region — push the optimizer away
        n = len(r)
        sigma2 = np.empty(n)
        sigma2[0] = sample_var
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2 = np.maximum(sigma2, 1e-12)  # guard log(0) for numerically degenerate params
        return 0.5 * float(np.sum(np.log(2 * np.pi * sigma2) + r**2 / sigma2))

    initial_guess = [sample_var * 0.05, 0.05, 0.90]
    bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999)]
    result = minimize(neg_log_likelihood, initial_guess, bounds=bounds, method="L-BFGS-B")
    omega, alpha, beta = result.x

    # Recompute the fitted variance series once more at the optimum so we
    # can return it (avoids re-deriving it from scratch at forecast time).
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = sample_var
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]

    return GarchFit(omega=omega, alpha=alpha, beta=beta, mean_return=mean_r, demeaned_returns=r, conditional_variance=sigma2)


def garch_forecast_vol(fit: GarchFit, horizon_days: int = 1, annualization_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """h-day-ahead annualized vol forecast from a fitted GARCH(1,1).

    One-step: sigma^2_{t+1} = omega + alpha*r_t^2 + beta*sigma_t^2
    Multi-step: the forecast mean-reverts geometrically toward the long-run
    variance at rate (alpha+beta) per day:

        sigma^2_{t+h} = LRV + (alpha+beta)^(h-1) * (sigma^2_{t+1} - LRV)

    As horizon_days -> infinity this converges to the long-run variance —
    i.e. GARCH's forecast of "how volatile will it be eventually" is just
    the unconditional variance, regardless of today's conditions.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    last_sigma2 = fit.conditional_variance[-1]
    last_return = fit.demeaned_returns[-1]
    one_step_var = fit.omega + fit.alpha * last_return**2 + fit.beta * last_sigma2

    lrv = fit.long_run_variance
    if horizon_days == 1:
        horizon_var = one_step_var
    else:
        horizon_var = lrv + (fit.persistence ** (horizon_days - 1)) * (one_step_var - lrv)

    return math.sqrt(max(horizon_var, 0.0) * annualization_factor)
