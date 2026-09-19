"""Fit normal and Student-t curves to open interest BY STRIKE -- a map of
where positioning/liquidity is concentrated, not a forecast of anything.

Read this before using any of it to pick strikes:

  - This fits a curve to CURRENT OPEN INTEREST, a snapshot of existing
    positions -- it says where capital already sits, not where the stock
    is going. Don't read "OI is concentrated near strike K" as "the
    market expects K," any more than a crowded parking lot means the
    store is about to have a sale.
  - "High OI at a strike" does NOT mean "institutional smart money."
    It's equally consistent with retail congregating at round-number
    strikes, market-maker delta-hedging flow, or old positions nobody's
    closed. This module makes no claim about WHO holds the open
    interest, only where it is.
  - This is a genuinely different question from max_pain() below (the
    strike that minimizes total payout to option holders at expiry --
    a specific, well-defined calculation) or from sigma_moves.py's
    return-distribution fitting (which fits actual historical PRICE
    MOVEMENT, not today's static positioning). Don't conflate "OI sits
    above a fitted normal curve" with "max pain" -- they're unrelated
    numbers that happen to both involve a strike axis.

Fitting method: OI at each strike is expanded into a weighted pseudo-
sample (proportional counts summing to a fixed total, not raw OI, since
only the SHAPE matters) so the same, already-tested `scipy.stats.t.fit`
machinery sigma_moves.py uses can fit both distributions consistently.

Model comparison uses AIC, not raw fit error -- this matters. A Student-t
has one more free parameter than a normal (df, on top of loc/scale) and
NESTS the normal as the df-to-infinity limit, so it can always match or
beat a normal's raw fit error, even on data with zero real excess
kurtosis -- comparing SSE directly would make "t wins" close to a
foregone conclusion rather than a real finding. AIC (2k - 2*log-
likelihood) penalizes that extra parameter, so "t" only actually wins
when the fatter tail earns its keep. A huge fitted df (hundreds or more)
on a "t" result means the fit converged to something normal-equivalent
in practice, regardless of which label technically won -- surface the df
itself, not just the winning label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

PSEUDO_SAMPLE_SIZE = 5000


@dataclass
class DistributionFit:
    kind: str  # "normal" or "t"
    params: dict  # normal: {"mean","std"}; t: {"df","loc","scale"}
    sse: float  # sum-of-squared-error between fitted PDF and the actual OI shape at the observed strikes -- display only, see aic
    aic: float  # Akaike Information Criterion (2k - 2*log-likelihood) -- the actual model-comparison criterion, see module note


@dataclass
class OIDistributionResult:
    strikes: np.ndarray
    oi_weights: np.ndarray  # OI at each strike, normalized to sum to 1 -- the empirical "density" being fit
    normal_fit: DistributionFit
    t_fit: DistributionFit
    better_fit: str  # "normal" or "t", whichever has the lower SSE


def _oi_by_strike(chain: pd.DataFrame) -> pd.DataFrame:
    """Call + put open interest combined per strike -- total positioning
    at that price level, not split by side."""
    return chain.groupby("strike", as_index=False)["open_interest"].sum().sort_values("strike").reset_index(drop=True)


def fit_oi_distribution(chain: pd.DataFrame) -> OIDistributionResult | None:
    """None if there are fewer than 3 strikes with any open interest at
    all -- can't fit a distribution's shape to 0-2 points."""
    by_strike = _oi_by_strike(chain)
    by_strike = by_strike[by_strike["open_interest"] > 0]
    if len(by_strike) < 3:
        return None

    strikes = by_strike["strike"].to_numpy(dtype=float)
    oi = by_strike["open_interest"].to_numpy(dtype=float)
    weights = oi / oi.sum()

    # Expand into a pseudo-sample with counts proportional to OI, summing
    # to PSEUDO_SAMPLE_SIZE -- shape-preserving, independent of whether
    # this chain's OI totals in the hundreds or the hundreds of thousands.
    counts = np.round(weights * PSEUDO_SAMPLE_SIZE).astype(int)
    counts = np.maximum(counts, 0)
    if counts.sum() == 0:
        return None
    pseudo_sample = np.repeat(strikes, counts)

    normal_mean, normal_std = float(np.mean(pseudo_sample)), float(np.std(pseudo_sample, ddof=1))
    if normal_std <= 0:
        return None
    normal_pdf = stats.norm.pdf(strikes, loc=normal_mean, scale=normal_std)
    normal_sse = float(np.sum((normal_pdf / normal_pdf.sum() - weights) ** 2)) if normal_pdf.sum() > 0 else float("inf")
    normal_ll = float(np.sum(stats.norm.logpdf(pseudo_sample, loc=normal_mean, scale=normal_std)))
    normal_aic = 2 * 2 - 2 * normal_ll  # k=2: mean, std

    t_df, t_loc, t_scale = stats.t.fit(pseudo_sample)
    t_pdf = stats.t.pdf(strikes, df=t_df, loc=t_loc, scale=t_scale)
    t_sse = float(np.sum((t_pdf / t_pdf.sum() - weights) ** 2)) if t_pdf.sum() > 0 else float("inf")
    t_ll = float(np.sum(stats.t.logpdf(pseudo_sample, df=t_df, loc=t_loc, scale=t_scale)))
    t_aic = 2 * 3 - 2 * t_ll  # k=3: df, loc, scale

    normal_fit = DistributionFit(kind="normal", params={"mean": normal_mean, "std": normal_std}, sse=normal_sse, aic=normal_aic)
    t_fit = DistributionFit(kind="t", params={"df": float(t_df), "loc": float(t_loc), "scale": float(t_scale)}, sse=t_sse, aic=t_aic)

    return OIDistributionResult(
        strikes=strikes, oi_weights=weights, normal_fit=normal_fit, t_fit=t_fit,
        better_fit="t" if t_aic < normal_aic else "normal",
    )


def distribution_percentile_strike(fit: DistributionFit, percentile: float) -> float:
    """The strike at a given percentile (0-1) of a fitted distribution --
    e.g. percentile=0.16 on a normal fit is approximately the fitted
    "-1 sigma" strike. Use this to place spread strikes at OI-implied
    percentiles instead of a flat dollar increment."""
    if not (0.0 < percentile < 1.0):
        raise ValueError(f"percentile must be in (0, 1), got {percentile}")
    if fit.kind == "normal":
        return float(stats.norm.ppf(percentile, loc=fit.params["mean"], scale=fit.params["std"]))
    return float(stats.t.ppf(percentile, df=fit.params["df"], loc=fit.params["loc"], scale=fit.params["scale"]))


def max_pain(chain: pd.DataFrame) -> float | None:
    """The strike that minimizes TOTAL payout option WRITERS owe at
    expiration, summed across every listed strike's open interest --
    the actual, standard "max pain" calculation. This is a real,
    specific number, NOT the same thing as where fit_oi_distribution()
    says OI is concentrated (see module docstring) -- it's included here
    because it's cheap to compute from the same chain and genuinely
    related, not because the two numbers usually coincide.

    None if there's no open interest anywhere in the chain."""
    by_strike_type = chain.groupby(["strike", "type"], as_index=False)["open_interest"].sum()
    strikes = sorted(by_strike_type["strike"].unique())
    if not strikes or by_strike_type["open_interest"].sum() <= 0:
        return None

    calls = by_strike_type[by_strike_type["type"] == "call"].set_index("strike")["open_interest"]
    puts = by_strike_type[by_strike_type["type"] == "put"].set_index("strike")["open_interest"]

    total_payout_by_settle = {}
    for settle in strikes:
        # Every call struck below settle finishes ITM by (settle - strike)
        # per contract; every put struck above settle finishes ITM by
        # (strike - settle). Writers owe that, times open interest.
        call_payout = sum(max(settle - k, 0.0) * oi for k, oi in calls.items())
        put_payout = sum(max(k - settle, 0.0) * oi for k, oi in puts.items())
        total_payout_by_settle[settle] = call_payout + put_payout

    return min(total_payout_by_settle, key=total_payout_by_settle.get)
