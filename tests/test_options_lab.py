"""Correctness tests for risk_tool.options_lab.

Key thing to verify: this module invents no new pricing math, so its
outputs must be provably consistent with the modules it wraps --
pl_grid's expiration row must match option_strategy.analyze_strategy's
exact payoff, greeks_curve's at-the-money point must match a direct
all_greeks call, and shock_legs must apply iv_mult before iv_shift.
"""
import pytest

from risk_tool.greeks import all_greeks
from risk_tool.option_strategy import OptionLeg, analyze_strategy, strategy_pl
from risk_tool.options_lab import (
    evaluate_scenario,
    greeks_curve,
    pl_grid,
    shock_legs,
    strategy_greeks,
)

S, T, r, q, IV = 100.0, 30 / 365.0, 0.03, 0.0, 0.30


def _long_call_leg(strike=100.0, premium=3.0, iv=IV):
    return OptionLeg(option_type="call", strike=strike, premium=premium, contracts=1.0, iv=iv)


def test_shock_legs_applies_mult_before_shift():
    legs = [_long_call_leg(iv=0.30)]
    shocked = shock_legs(legs, iv_shift=0.05, iv_mult=0.5)
    assert shocked[0].iv == pytest.approx(0.30 * 0.5 + 0.05)


def test_shock_legs_floors_iv_above_zero():
    legs = [_long_call_leg(iv=0.10)]
    shocked = shock_legs(legs, iv_shift=-10.0, iv_mult=1.0)  # would go deeply negative unfloored
    assert shocked[0].iv > 0


def test_shock_legs_leaves_legs_without_iv_untouched():
    leg = OptionLeg(option_type="call", strike=100.0, premium=3.0, contracts=1.0, iv=None)
    shocked = shock_legs([leg], iv_shift=0.1, iv_mult=2.0)
    assert shocked[0].iv is None


def test_strategy_greeks_matches_direct_all_greeks_for_a_single_leg():
    legs = [_long_call_leg()]
    result = strategy_greeks(legs, S, T, r, q)
    expected = all_greeks(S, 100.0, T, r, q, IV, "call")
    assert result["delta"] == pytest.approx(expected["delta"] * 100.0)  # shares_per_contract default 100
    assert result["gamma"] == pytest.approx(expected["gamma"] * 100.0)


def test_strategy_greeks_short_leg_negates_sign():
    long_legs = [_long_call_leg()]
    short_legs = [OptionLeg(option_type="call", strike=100.0, premium=3.0, contracts=-1.0, iv=IV)]
    long_g = strategy_greeks(long_legs, S, T, r, q)
    short_g = strategy_greeks(short_legs, S, T, r, q)
    assert short_g["delta"] == pytest.approx(-long_g["delta"])


def test_strategy_greeks_is_zero_at_expiration():
    legs = [_long_call_leg()]
    result = strategy_greeks(legs, S, 0.0, r, q)
    assert all(v == 0.0 for v in result.values())


def test_strategy_greeks_requires_iv_on_every_leg():
    leg = OptionLeg(option_type="call", strike=100.0, premium=3.0, contracts=1.0, iv=None)
    with pytest.raises(ValueError):
        strategy_greeks([leg], S, T, r, q)


def test_pl_grid_expiration_row_matches_analyze_strategy_exactly():
    legs = [
        OptionLeg(option_type="put", strike=105.0, premium=6.0, contracts=1.0, iv=IV),
        OptionLeg(option_type="put", strike=95.0, premium=2.0, contracts=-1.0, iv=IV),
    ]
    dte = 30
    grid = pl_grid(legs, spot_now=S, dte=dte, r=r, q=q, n_spot_points=9, n_time_points=4)
    expiration_rows = grid[grid["days_forward"] == dte]
    assert not expiration_rows.empty
    for _, row in expiration_rows.iterrows():
        assert row["pl"] == pytest.approx(strategy_pl(legs, row["spot"]))


def test_pl_grid_includes_zero_and_dte_days_forward_even_with_coarse_time_points():
    legs = [_long_call_leg()]
    grid = pl_grid(legs, spot_now=S, dte=45, n_spot_points=5, n_time_points=3)
    days = set(grid["days_forward"].unique())
    assert 0 in days
    assert 45 in days


def test_pl_grid_rejects_negative_dte():
    with pytest.raises(ValueError):
        pl_grid([_long_call_leg()], spot_now=S, dte=-1)


def test_greeks_curve_atm_point_matches_direct_call():
    legs = [_long_call_leg(strike=100.0)]
    curve = greeks_curve(legs, spot_now=100.0, T_years=T, r=r, q=q, spot_range_pct=0.0, n_points=1)
    expected = all_greeks(100.0, 100.0, T, r, q, IV, "call")
    assert curve.iloc[0]["delta"] == pytest.approx(expected["delta"] * 100.0)


def test_evaluate_scenario_no_shock_matches_strategy_pl_today_baseline():
    legs = [_long_call_leg()]
    scenario = evaluate_scenario(legs, spot_now=S, dte=30, r=r, q=q)
    assert scenario.spot == pytest.approx(S)
    assert scenario.days_forward == 0
    assert scenario.pl == pytest.approx(strategy_pl_today_baseline(legs, S, 30 / 365.0, r, q))


def strategy_pl_today_baseline(legs, spot, T_years, r, q):
    from risk_tool.option_strategy import strategy_pl_today as _spt
    return _spt(legs, spot, T_years, r, q)


def test_evaluate_scenario_spot_move_and_iv_crush_combine():
    legs = [_long_call_leg(strike=100.0, premium=3.0, iv=0.50)]
    scenario = evaluate_scenario(legs, spot_now=100.0, dte=30, spot_move_pct=0.10, iv_mult=0.5, days_forward=1)
    assert scenario.spot == pytest.approx(110.0)
    assert scenario.days_forward == 1
    # A 10% spot pop should still leave meaningful positive P&L on a long
    # ATM-struck call even after a 50% relative vol crush the next day.
    assert scenario.pl > 0


def test_evaluate_scenario_at_expiration_has_no_greeks():
    legs = [_long_call_leg()]
    scenario = evaluate_scenario(legs, spot_now=S, dte=5, days_forward=5)
    assert scenario.T_remaining == 0.0
    assert scenario.greeks == {}
