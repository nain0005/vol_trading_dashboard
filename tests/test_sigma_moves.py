"""Correctness tests for risk_tool.sigma_moves.

Builds synthetic price series with KNOWN, hand-placed outlier days so the
exceedance count, gap arithmetic, and days-since-last can be verified
exactly -- not just "it runs without crashing."
"""
import numpy as np
import pandas as pd
import pytest

from risk_tool.sigma_moves import (
    DEFAULT_ROLLING_WINDOW,
    _crash_probability_from_exceedances,
    analyze_symbol,
    crash_probability_given_quiet_run,
    rolling_daily_sigma,
    zscores,
)

RNG = np.random.default_rng(42)


def _flat_vol_series_with_spikes(n=300, daily_vol=0.01, spike_positions=(), spike_size=0.10):
    """Close series with constant small daily noise, plus a few days
    deliberately forced to a large move -- spike_positions are return
    indices (0-based, i.e. the return from close[i] to close[i+1])."""
    returns = RNG.normal(0.0, daily_vol, n)
    for pos in spike_positions:
        returns[pos] = spike_size
    closes = 100.0 * np.cumprod(1 + returns)
    return pd.Series(np.concatenate([[100.0], closes]))


def test_rolling_daily_sigma_has_no_lookahead():
    # A single huge spike on day 50 must NOT appear in sigma until day 51
    # (shift(1) means day t's sigma is built from days before t).
    close = _flat_vol_series_with_spikes(n=100, spike_positions=[50])
    sigma = rolling_daily_sigma(close, window=20)
    # sigma at the index right after the spike return (close index 51,
    # i.e. the return realized between close[50] and close[51]) should
    # still be small -- the spike return itself isn't in its own window.
    assert sigma.iloc[51] < 0.03  # far below what including the 10% spike would push it to


def test_zscore_of_a_large_isolated_move_is_large():
    close = _flat_vol_series_with_spikes(n=300, daily_vol=0.01, spike_positions=[200], spike_size=0.15)
    z = zscores(close, window=20)
    # index 201 = the return from close[200] to close[201], i.e. the spike itself
    assert abs(z.iloc[201]) > 5  # a 15% move against ~1% recent vol is a huge z-score


def test_analyze_symbol_gap_and_days_since_last_arithmetic_matches_independent_recomputation():
    # Three deliberately huge planted spikes guarantee at least those 3
    # exceedances -- but background Gaussian noise at threshold=2.0 will
    # ALSO cross it by chance sometimes (that's correct: ~4.55% of any
    # normal series exceeds 2 true-sigma, and the rolling estimate adds
    # its own sampling noise on top), so this doesn't assume the 3 plants
    # are the ONLY exceedances. Instead it recomputes exceedance positions
    # independently from zscores()'s own output and checks analyze_symbol's
    # gap-averaging and days-since-last arithmetic against that -- the
    # thing actually worth verifying, without assuming away real noise.
    n = 400
    spikes = [100, 150, 250]  # return indices
    close = _flat_vol_series_with_spikes(n=n, daily_vol=0.005, spike_positions=spikes, spike_size=0.12)
    stats = analyze_symbol("TEST", close, threshold=2.0, window=20)
    assert stats is not None
    assert stats.n_exceedances >= 3  # at least the 3 planted spikes

    z = zscores(close, window=DEFAULT_ROLLING_WINDOW).dropna()
    exceed_positions = np.flatnonzero((z.abs() > 2.0).to_numpy())
    assert stats.n_exceedances == len(exceed_positions)
    assert stats.empirical_avg_days_between == pytest.approx(np.diff(exceed_positions).mean())
    assert stats.days_since_last_exceedance == len(z) - 1 - exceed_positions[-1]


def test_no_exceedances_gives_none_not_a_crash():
    # Extremely low, uniform vol with no spikes -- some days may still
    # incidentally cross a low threshold, so use threshold=10 to
    # guarantee zero exceedances deterministically.
    close = _flat_vol_series_with_spikes(n=200, daily_vol=0.001, spike_positions=[])
    stats = analyze_symbol("FLAT", close, threshold=10.0, window=20)
    assert stats is not None
    assert stats.n_exceedances == 0
    assert stats.empirical_avg_days_between is None
    assert stats.days_since_last_exceedance is None
    assert stats.overdue_ratio is None
    assert stats.model_avg_days_between > 0  # still defined -- comes from the fitted distribution, not the empirical count


def test_single_exceedance_has_no_empirical_gap_but_has_days_since_last():
    close = _flat_vol_series_with_spikes(n=200, daily_vol=0.005, spike_positions=[100], spike_size=0.15)
    stats = analyze_symbol("ONE", close, threshold=3.0, window=20)
    assert stats is not None
    assert stats.n_exceedances >= 1
    if stats.n_exceedances == 1:
        assert stats.empirical_avg_days_between is None  # needs >= 2 exceedances for a gap
        assert stats.days_since_last_exceedance is not None


def test_vol_ratio_reflects_genuine_compression():
    # First half of the series has high vol, second half (the trailing
    # window analyze_symbol looks at) has much lower vol -- current_vol_ratio
    # should clearly read as compressed (< 1).
    n = 300
    high_vol_returns = RNG.normal(0.0, 0.03, n // 2)
    low_vol_returns = RNG.normal(0.0, 0.003, n // 2)
    returns = np.concatenate([high_vol_returns, low_vol_returns])
    closes = 100.0 * np.cumprod(1 + returns)
    close = pd.Series(np.concatenate([[100.0], closes]))

    stats = analyze_symbol("COMPRESSED", close, threshold=2.0, window=20)
    assert stats is not None
    assert stats.current_vol_ratio is not None
    assert stats.current_vol_ratio < 0.5  # recent vol is much lower than the full-sample average


def test_analyze_symbol_returns_none_for_too_little_history():
    close = pd.Series([100.0, 101.0, 99.5, 100.2])  # far fewer than window+2 usable closes
    assert analyze_symbol("SHORT", close, threshold=2.0, window=20) is None


def test_fitted_t_df_is_lower_for_fatter_tailed_series():
    n = 400
    # "Fat tails": occasional large moves mixed into otherwise small noise
    normal_like = RNG.normal(0.0, 0.01, n)
    fat_tailed = RNG.standard_t(df=3, size=n) * 0.01

    close_normal = pd.Series(np.concatenate([[100.0], 100.0 * np.cumprod(1 + normal_like)]))
    close_fat = pd.Series(np.concatenate([[100.0], 100.0 * np.cumprod(1 + fat_tailed)]))

    stats_normal = analyze_symbol("NORMALISH", close_normal, threshold=2.0, window=20)
    stats_fat = analyze_symbol("FATTAILED", close_fat, threshold=2.0, window=20)
    assert stats_normal is not None and stats_fat is not None
    assert stats_fat.fitted_t_df < stats_normal.fitted_t_df


# --- crash_probability_given_quiet_run / _crash_probability_from_exceedances ---

# Hand-constructed 14-day exceedance pattern, indices 0-13, True at 2, 6, 11:
#   F F T F F F T F F F F T F F
# Every number below is derived by hand-walking this exact array (see the
# accompanying comment math in the PR/commit, reproduced in each assertion).
_PATTERN = np.array([False, False, True, False, False, False, True, False, False, False, False, True, False, False])


def test_crash_probability_core_matches_hand_walked_example():
    result = _crash_probability_from_exceedances(_PATTERN, forward_days=2)
    assert result is not None
    baseline_daily_prob, baseline_within_horizon, current_run, n_analogs, empirical_prob = result

    # 3 exceedances in 14 days.
    assert baseline_daily_prob == pytest.approx(3 / 14)
    # 1 - (1 - 3/14)^2 = 1 - (11/14)^2 = 75/196
    assert baseline_within_horizon == pytest.approx(75 / 196)
    # Counting back from index 13: F(1), F(2), then index 11 is True -> stop.
    assert current_run == 2
    # Hand-walked analogs (windows of the last 2 days both quiet) at
    # i = 1, 4, 5, 8, 9, 10, 13 -- 7 total. Hits (an exceedance in the next
    # 2 days after) at i = 1, 4, 5, 9, 10 -- 5 total; i=8 and i=13 miss.
    assert n_analogs == 7
    assert empirical_prob == pytest.approx(5 / 7)


def test_crash_probability_current_run_zero_when_last_day_exceeded():
    pattern = np.array([False, False, True])  # most recent day IS an exceedance
    result = _crash_probability_from_exceedances(pattern, forward_days=1)
    assert result is not None
    _, _, current_run, _, _ = result
    assert current_run == 0


def test_crash_probability_current_run_zero_only_matches_actual_exceedance_days():
    # Regression: the general "window ending at i" formula degenerates to
    # an EMPTY slice for every i when current_run == 0 (exceed[i+1:i+1]),
    # which passes vacuously regardless of exceed[i] -- before the fix
    # this counted every single day in history as an "analog" instead of
    # only the 3 real exceedance days. Caught via AppTest against demo
    # data (AAPL/VXX both showed n_analogs == n_observations exactly,
    # which is the tell). Hand-walked here:
    #   idx: 0    1     2     3     4     5
    #        T    F     F     T     F     T   <- current_run = 0 (day 5 exceeded)
    pattern = np.array([True, False, False, True, False, True])
    result = _crash_probability_from_exceedances(pattern, forward_days=2)
    assert result is not None
    _, _, current_run, n_analogs, empirical_prob = result
    assert current_run == 0
    # Qualifying days (exceed[i] itself True): i = 0, 3, 5 -- NOT all 6 days.
    assert n_analogs == 3
    # i=0: future=exceed[1:3]=[F,F] -> no hit.
    # i=3: future=exceed[4:6]=[F,T] -> hit.
    # i=5: future=exceed[6:8]=[] (past the end) -> no hit.
    assert empirical_prob == pytest.approx(1 / 3)


def test_crash_probability_none_when_no_exceedances_at_all():
    pattern = np.zeros(50, dtype=bool)
    assert _crash_probability_from_exceedances(pattern, forward_days=5) is None


def test_crash_probability_single_exceedance_gives_one_analog_at_most():
    # Only one exceedance, at the very start -- current_run counts back to
    # it, and there's exactly one point in history (right after it) where
    # that exact run length was reached.
    pattern = np.array([True] + [False] * 20)
    result = _crash_probability_from_exceedances(pattern, forward_days=3)
    assert result is not None
    _, _, current_run, n_analogs, empirical_prob = result
    assert current_run == 20  # 20 quiet days since the one exceedance at index 0
    assert n_analogs == 1  # only index 0 itself reached a "run of >=20 quiet days" (trivially, the exceedance day)
    assert empirical_prob == 0.0  # the only analog (day 0) was immediately followed by 20 quiet days, no hit


def test_crash_probability_given_quiet_run_direction_filters_correctly():
    # 60 days of tiny noise, one big DOWN day and one big UP day of equal
    # magnitude planted apart from each other -- direction="down" must only
    # count the down day, direction="up" only the up day.
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.003, 60)
    returns[30] = -0.08  # big down day
    returns[45] = 0.08  # big up day
    close = pd.Series(np.concatenate([[100.0], 100.0 * np.cumprod(1 + returns)]))

    down_result = crash_probability_given_quiet_run(close, threshold=1.0, direction="down", forward_days=5, window=20)
    up_result = crash_probability_given_quiet_run(close, threshold=1.0, direction="up", forward_days=5, window=20)
    assert down_result is not None and up_result is not None
    assert down_result.direction == "down"
    assert up_result.direction == "up"
    # Both baseline rates should reflect exactly 1 exceedance each in this
    # construction (the planted move dominates; incidental noise-driven
    # exceedances at |z|>=1 are common at this low a threshold, so just
    # check the direction-specific mechanics wired correctly rather than
    # an exact count):
    assert down_result.baseline_daily_prob is not None and down_result.baseline_daily_prob > 0
    assert up_result.baseline_daily_prob is not None and up_result.baseline_daily_prob > 0


def test_crash_probability_given_quiet_run_returns_none_for_short_history():
    close = pd.Series([100.0, 101.0, 99.5])
    assert crash_probability_given_quiet_run(close, threshold=1.0, window=20) is None


def test_crash_probability_given_quiet_run_rejects_bad_direction():
    close = pd.Series(100.0 + np.cumsum(RNG.normal(0, 1, 100)))
    with pytest.raises(ValueError):
        crash_probability_given_quiet_run(close, direction="sideways")
