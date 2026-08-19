"""Correctness tests for risk_tool.strike_selection.

Two things matter most here: the EV arithmetic itself (checked against a
manual calculation), and that compare_implied_vs_realized_ev actually uses
market IV vs. your own vol estimate in exactly the documented places (not
accidentally mixing them up) — that wiring correctness is the whole point
of the realized-vs-implied feature.
"""
import pytest

from risk_tool.config import RiskConfig
from risk_tool.pricing import black_scholes_price, itm_probability
from risk_tool.strike_selection import (
    compare_implied_vs_realized_ev,
    evaluate_strike,
    generate_candidate_strikes,
    rank_strikes_by_ev,
)

S, T, r, q, sigma = 100.0, 0.25, 0.03, 0.0, 0.30


def test_generate_candidate_strikes_rounds_to_atm_and_spans_both_sides():
    strikes = generate_candidate_strikes(spot=101.0, strike_increment=5.0, num_each_side=2)
    assert strikes == [90.0, 95.0, 100.0, 105.0, 110.0]


def test_generate_candidate_strikes_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        generate_candidate_strikes(spot=100, strike_increment=0, num_each_side=2)
    with pytest.raises(ValueError):
        generate_candidate_strikes(spot=100, strike_increment=5, num_each_side=-1)


def test_evaluate_strike_ev_matches_manual_calculation():
    config = RiskConfig(default_profit_target_pct=1.0, min_risk_reward_ratio=2.0)
    K = 105.0
    result = evaluate_strike(S, K, T, r, q, sigma, "call", config)

    expected_premium = black_scholes_price(S, K, T, r, q, sigma, "call")
    expected_p_win = itm_probability(S, K, T, r, q, sigma, "call")
    expected_profit_if_win = expected_premium * 1.0
    expected_ev = expected_p_win * expected_profit_if_win - (1 - expected_p_win) * expected_premium

    assert result.premium == pytest.approx(expected_premium)
    assert result.p_win_risk_neutral == pytest.approx(expected_p_win)
    assert result.ev == pytest.approx(expected_ev)
    assert result.risk_reward == pytest.approx(1.0)  # profit_target_pct=1.0 -> ratio is always 1:1 by construction
    assert result.is_poor_setup is True  # 1:1 < the 2:1 minimum


def test_higher_profit_target_improves_risk_reward_and_can_clear_poor_setup_flag():
    config = RiskConfig(min_risk_reward_ratio=2.0)
    poor = evaluate_strike(S, 105.0, T, r, q, sigma, "call", config, profit_target_pct=1.0)
    good = evaluate_strike(S, 105.0, T, r, q, sigma, "call", config, profit_target_pct=2.5)
    assert poor.is_poor_setup is True
    assert good.is_poor_setup is False
    assert good.risk_reward > poor.risk_reward


def test_call_itm_probability_decreases_monotonically_with_strike():
    """A well-known, safe monotonicity fact: for a call, higher strikes are
    strictly less likely to finish ITM, holding everything else fixed."""
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    probs = [evaluate_strike(S, K, T, r, q, sigma, "call").p_win_risk_neutral for K in strikes]
    assert probs == sorted(probs, reverse=True)


def test_rank_strikes_by_ev_sorts_descending():
    strikes = generate_candidate_strikes(spot=S, strike_increment=5, num_each_side=3)
    ranked = rank_strikes_by_ev(S, T, r, q, sigma, "call", strikes)
    evs = [s.ev for s in ranked]
    assert evs == sorted(evs, reverse=True)
    assert len(ranked) == len(strikes)


def test_rank_strikes_drops_strikes_over_the_premium_cap():
    strikes = generate_candidate_strikes(spot=S, strike_increment=5, num_each_side=3)
    cheapest_premium = min(evaluate_strike(S, K, T, r, q, sigma, "call").premium for K in strikes)
    tight_cap = cheapest_premium * 100 * 1.01  # only the single cheapest strike should survive
    ranked = rank_strikes_by_ev(S, T, r, q, sigma, "call", strikes, max_premium_dollars=tight_cap)
    assert len(ranked) >= 1
    assert all(s.premium * 100 <= tight_cap for s in ranked)


class TestRealizedVsImpliedEV:
    def test_uses_market_iv_and_realized_vol_in_exactly_the_right_places(self):
        K = 105.0
        market_iv = 0.30
        realized = 0.45
        result = compare_implied_vs_realized_ev(S, K, T, r, q, market_iv, realized, "call")

        expected_market_premium = black_scholes_price(S, K, T, r, q, market_iv, "call")
        assert result.market_premium == pytest.approx(expected_market_premium)

        # risk_neutral_ev must be priced AND probability-assigned with market_iv throughout.
        assert result.risk_neutral_ev.premium == pytest.approx(expected_market_premium)
        assert result.risk_neutral_ev.p_win_risk_neutral == pytest.approx(itm_probability(S, K, T, r, q, market_iv, "call"))

        # edge_ev must keep the MARKET premium (that's what you actually pay)
        # but assign probability using the REALIZED/forecast vol.
        assert result.edge_ev.premium == pytest.approx(expected_market_premium)
        assert result.edge_ev.p_win_risk_neutral == pytest.approx(itm_probability(S, K, T, r, q, realized, "call"))

        assert result.vol_spread == pytest.approx(realized - market_iv)

    def test_vol_spread_sign_matches_realized_minus_implied(self):
        result_higher = compare_implied_vs_realized_ev(S, 105.0, T, r, q, market_iv=0.30, realized_or_forecast_vol=0.45, option_type="call")
        result_lower = compare_implied_vs_realized_ev(S, 105.0, T, r, q, market_iv=0.30, realized_or_forecast_vol=0.15, option_type="call")
        assert result_higher.vol_spread > 0
        assert result_lower.vol_spread < 0
