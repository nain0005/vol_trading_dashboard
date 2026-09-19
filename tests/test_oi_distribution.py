"""Correctness tests for risk_tool.oi_distribution.

Builds OI-by-strike patterns with a KNOWN shape (symmetric/bell-shaped vs.
deliberately fat-tailed) so which distribution "wins" is verifiable, and
a small hand-constructed chain for max_pain() where the total payout at
each candidate settle price is computed by hand.
"""
import numpy as np
import pandas as pd
import pytest

from risk_tool.oi_distribution import (
    distribution_percentile_strike,
    fit_oi_distribution,
    max_pain,
)


def _chain_from_oi(strike_oi: dict, opt_type: str = "call") -> pd.DataFrame:
    return pd.DataFrame([{"strike": k, "type": opt_type, "open_interest": oi} for k, oi in strike_oi.items()])


def test_returns_none_with_fewer_than_three_priced_strikes():
    assert fit_oi_distribution(_chain_from_oi({100.0: 500, 105.0: 300})) is None


def test_returns_none_when_all_open_interest_is_zero():
    assert fit_oi_distribution(_chain_from_oi({95.0: 0, 100.0: 0, 105.0: 0})) is None


def test_bell_shaped_oi_prefers_normal_fit():
    # Symmetric, thin-tailed OI around 100 -- a textbook bell shape.
    strikes = np.arange(90, 111, 1.0)
    oi = np.round(np.exp(-((strikes - 100.0) ** 2) / (2 * 3.0**2)) * 1000).astype(int)
    result = fit_oi_distribution(_chain_from_oi(dict(zip(strikes, oi))))
    assert result is not None
    assert result.better_fit == "normal"
    assert result.normal_fit.params["mean"] == pytest.approx(100.0, abs=1.0)


def test_fat_tailed_oi_prefers_t_fit():
    # Most OI tight around 100, but with real weight parked way out in
    # the wings (deep OTM hedges) -- a normal curve can't explain both
    # the tight center AND the far wings at once nearly as well as a
    # heavy-tailed t can.
    strike_oi = {100.0: 3000, 99.0: 2000, 101.0: 2000, 98.0: 1000, 102.0: 1000}
    strike_oi.update({70.0: 400, 130.0: 400, 60.0: 300, 140.0: 300, 50.0: 250, 150.0: 250})
    result = fit_oi_distribution(_chain_from_oi(strike_oi))
    assert result is not None
    assert result.better_fit == "t"
    assert result.t_fit.params["df"] < 30  # meaningfully non-normal tail weight


def test_percentile_strike_at_half_matches_the_fitted_center():
    strikes = np.arange(90, 111, 1.0)
    oi = np.round(np.exp(-((strikes - 100.0) ** 2) / (2 * 3.0**2)) * 1000).astype(int)
    result = fit_oi_distribution(_chain_from_oi(dict(zip(strikes, oi))))
    assert result is not None
    median_strike = distribution_percentile_strike(result.normal_fit, 0.5)
    assert median_strike == pytest.approx(result.normal_fit.params["mean"], abs=0.5)


def test_percentile_strike_rejects_out_of_range_input():
    fit = fit_oi_distribution(_chain_from_oi({95.0: 100, 100.0: 200, 105.0: 100})).normal_fit
    with pytest.raises(ValueError):
        distribution_percentile_strike(fit, 1.5)
    with pytest.raises(ValueError):
        distribution_percentile_strike(fit, 0.0)


def test_max_pain_matches_hand_computed_total_payout():
    # Calls concentrated low, puts concentrated high -- symmetric setup,
    # hand-computed total payout at each candidate settle:
    #   settle=95:  call_payout=0,   put_payout=0*0+5*5+10*10=125 -> 125
    #   settle=100: call_payout=10*5+5*0+0*0=50, put_payout=0*0+5*0+10*5=50 -> 100
    #   settle=105: call_payout=10*10+5*5+0*0=125, put_payout=0 -> 125
    # Minimum total payout (100) is at settle=100.
    chain = pd.concat([
        _chain_from_oi({95.0: 10, 100.0: 5, 105.0: 0}, opt_type="call"),
        _chain_from_oi({95.0: 0, 100.0: 5, 105.0: 10}, opt_type="put"),
    ])
    assert max_pain(chain) == pytest.approx(100.0)


def test_max_pain_shifts_toward_the_side_with_less_open_interest():
    # All the OI is calls struck at 100 -- writers are hurt most by a
    # settle ABOVE 100, so max pain (minimizing writer payout) should
    # land AT or below 100, not above it.
    chain = _chain_from_oi({95.0: 100, 100.0: 500, 105.0: 100}, opt_type="call")
    result = max_pain(chain)
    assert result is not None
    assert result <= 100.0


def test_max_pain_returns_none_with_no_open_interest():
    chain = _chain_from_oi({95.0: 0, 100.0: 0, 105.0: 0}, opt_type="call")
    assert max_pain(chain) is None
