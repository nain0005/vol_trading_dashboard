"""Correctness tests for spread_selection's OI-percentile strike search
(strike_source="oi_percentile") -- the alternative to the flat $-increment
window that instead places strikes at percentiles of a distribution fit
to open interest by strike (risk_tool.oi_distribution).

Reuses test_spread_selection.py's synthetic chain-building approach (mid
prices from black_scholes_price at a known sigma) plus a hand-designed
open_interest column with a KNOWN, symmetric shape so the resulting
percentile strikes are independently predictable via scipy directly, not
just "the code agrees with itself."
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_tool.oi_distribution import DistributionFit, fit_oi_distribution
from risk_tool.pricing import black_scholes_price
from risk_tool.spread_selection import (
    DEFAULT_OI_PERCENTILES,
    compare_strategies,
    find_best_spreads,
)

S, T, r, q, MARKET_IV = 100.0, 0.25, 0.03, 0.0, 0.30
STRIKES = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]


def _make_chain_with_oi(oi_by_strike: dict) -> pd.DataFrame:
    rows = []
    for k in STRIKES:
        for opt_type in ("call", "put"):
            price = black_scholes_price(S, k, T, r, q, MARKET_IV, opt_type)
            rows.append({"strike": k, "type": opt_type, "iv": MARKET_IV, "mid": price, "open_interest": oi_by_strike.get(k, 0)})
    return pd.DataFrame(rows)


def _bell_shaped_oi() -> dict:
    # Symmetric bell around 100, std ~5 -- same shape family as
    # test_oi_distribution.py's "prefers normal" case, so better_fit is
    # predictable and the fitted mean/std are close to the true generator.
    return {k: int(round(np.exp(-((k - 100.0) ** 2) / (2 * 5.0**2)) * 1000)) for k in STRIKES}


def test_oi_percentile_strikes_match_independently_computed_scipy_values():
    chain = _make_chain_with_oi(_bell_shaped_oi())
    fit_result = fit_oi_distribution(chain)
    assert fit_result is not None
    fit = fit_result.normal_fit  # force normal regardless of AIC winner -- deterministic expected values below

    # Independently compute the expected snapped strikes: scipy ppf at
    # each percentile, then snap by hand to the nearest of STRIKES.
    expected_raw = [stats.norm.ppf(p, loc=fit.params["mean"], scale=fit.params["std"]) for p in DEFAULT_OI_PERCENTILES]
    expected_snapped = sorted({min(STRIKES, key=lambda k: abs(k - raw)) for raw in expected_raw})

    candidates = find_best_spreads(
        "straddle", chain, S, T, r, q, strike_increment=5.0,
        strike_source="oi_percentile", oi_fit=fit, oi_percentiles=DEFAULT_OI_PERCENTILES,
    )
    assert candidates  # straddle has one leg pair per searched strike, so this should be non-empty
    strikes_used = sorted({c.legs[0][1].strike for c in candidates})
    assert strikes_used == pytest.approx(expected_snapped)


def test_oi_percentile_search_is_narrower_than_the_full_increment_window_here():
    # With only 3 default percentiles collapsing to a handful of strikes
    # (vs. num_each_side=6 -> up to 13 strikes for "increment"), the
    # oi_percentile candidate set should be a STRICT subset of strikes
    # actually searched by increment mode on the same chain.
    chain = _make_chain_with_oi(_bell_shaped_oi())
    fit = fit_oi_distribution(chain).normal_fit

    pct_candidates = find_best_spreads(
        "straddle", chain, S, T, r, q, strike_increment=5.0, num_each_side=6,
        strike_source="oi_percentile", oi_fit=fit, top_n=100,
    )
    inc_candidates = find_best_spreads(
        "straddle", chain, S, T, r, q, strike_increment=5.0, num_each_side=6, strike_source="increment", top_n=100,
    )
    pct_strikes = {c.legs[0][1].strike for c in pct_candidates}
    inc_strikes = {c.legs[0][1].strike for c in inc_candidates}
    assert pct_strikes.issubset(inc_strikes)
    assert len(pct_strikes) < len(inc_strikes)


def test_increment_mode_is_unaffected_by_the_new_parameters_existing():
    # Default strike_source, no oi_fit passed -- must behave exactly as
    # before the oi_percentile feature was added (regression guard).
    chain = _make_chain_with_oi(_bell_shaped_oi())
    candidates = find_best_spreads("straddle", chain, S, T, r, q, strike_increment=5.0, num_each_side=6, top_n=100)
    strikes_used = sorted({c.legs[0][1].strike for c in candidates})
    assert strikes_used == [k for k in STRIKES if 70.0 <= k <= 130.0]


def test_oi_percentile_without_a_fit_raises_valueerror():
    chain = _make_chain_with_oi(_bell_shaped_oi())
    with pytest.raises(ValueError):
        find_best_spreads("straddle", chain, S, T, r, q, strike_increment=5.0, strike_source="oi_percentile", oi_fit=None)


def test_invalid_strike_source_raises_valueerror():
    chain = _make_chain_with_oi(_bell_shaped_oi())
    fit = fit_oi_distribution(chain).normal_fit
    with pytest.raises(ValueError):
        find_best_spreads("straddle", chain, S, T, r, q, strike_increment=5.0, strike_source="bogus", oi_fit=fit)


def test_too_few_oi_priced_strikes_leaves_fit_oi_distribution_none_and_caller_must_handle_it():
    # Degenerate chain: OI at only 2 strikes -- fit_oi_distribution
    # correctly returns None (see test_oi_distribution.py), and a caller
    # that naively passes that None straight to strike_source="oi_percentile"
    # gets the same explicit ValueError as test_oi_percentile_without_a_fit,
    # not a silent fallback or a crash deeper in the fit math.
    sparse_oi = {100.0: 500, 105.0: 300}
    chain = _make_chain_with_oi(sparse_oi)
    fit_result = fit_oi_distribution(chain)
    assert fit_result is None
    with pytest.raises(ValueError):
        find_best_spreads("straddle", chain, S, T, r, q, strike_increment=5.0, strike_source="oi_percentile", oi_fit=fit_result)


def test_compare_strategies_threads_oi_percentile_params_through():
    chain = _make_chain_with_oi(_bell_shaped_oi())
    fit = fit_oi_distribution(chain).normal_fit
    result = compare_strategies(
        chain, S, T, r, q, strike_increment=5.0, strike_source="oi_percentile", oi_fit=fit,
        strategies=("straddle", "strangle"),
    )
    assert set(result.keys()) == {"straddle", "strangle"}
    # At least straddle should produce a candidate: there's OI-implied
    # strikes on both sides of spot in this symmetric setup.
    assert result["straddle"] is not None


def test_body_snapping_helper_reused_by_iron_butterfly_matches_nearest_strike_choice():
    # iron_butterfly's ATM "body" strike selection and oi_percentile's
    # snapping now share the same _nearest_strike helper -- confirm
    # iron_butterfly still picks the single nearest-to-spot strike as its
    # body (regression guard on the refactor, independent of OI at all).
    from risk_tool.spread_selection import _iron_butterfly_specs

    strikes = [90.0, 95.0, 98.0, 103.0, 110.0]
    specs = _iron_butterfly_specs(strikes, spot=100.0)
    bodies = {spec[1][2] for spec in specs}  # index 1 is ("short","put",body)
    assert bodies == {98.0}  # 98 is nearer to 100 than 103 (2 vs 3)
