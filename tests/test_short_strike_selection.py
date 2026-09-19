"""Correctness tests for the short-premium (income) additions to
risk_tool.strike_selection: evaluate_short_strike, rank_short_strikes_by_ev,
and compare_implied_vs_realized_ev_short.

Mirrors the verification style of test_strike_selection.py — every EV
number is checked against an independent manual calculation, not just
re-run through the same code path. theoretical_max_loss is checked against
a closed-form derivation (K - premium for a short put; None for a short
call) rather than just trusting option_strategy.analyze_strategy, since
that's the number this module explicitly promises never to hide.
"""
import math

import pytest

from risk_tool.config import RiskConfig
from risk_tool.pricing import black_scholes_price, itm_probability
from risk_tool.strike_selection import (
    compare_implied_vs_realized_ev_short,
    evaluate_short_strike,
    rank_short_strikes_by_ev,
)

S, T, r, q, sigma = 100.0, 0.25, 0.03, 0.0, 0.30


class TestEvaluateShortStrikePut:
    def test_matches_manual_calculation(self):
        config = RiskConfig(short_stop_loss_multiple=2.0, min_risk_reward_ratio=2.0)
        K = 95.0
        result = evaluate_short_strike(S, K, T, r, q, sigma, "put", config, profit_target_pct=1.0)

        expected_premium = black_scholes_price(S, K, T, r, q, sigma, "put")
        # Short put "wins" (keeps the premium) when it expires OTM, i.e. S_T > K —
        # exactly the condition a CALL finishes ITM under, so the complement is
        # itm_probability(..., "call") at the same K/sigma.
        expected_p_win = itm_probability(S, K, T, r, q, sigma, "call")
        expected_mech_loss = expected_premium * (2.0 - 1.0)
        expected_profit_if_win = expected_premium * 1.0
        expected_ev = expected_p_win * expected_profit_if_win - (1 - expected_p_win) * expected_mech_loss

        assert result.premium_received == pytest.approx(expected_premium)
        assert result.p_win_risk_neutral == pytest.approx(expected_p_win)
        assert result.mechanical_max_loss == pytest.approx(expected_mech_loss)
        assert result.profit_if_win == pytest.approx(expected_profit_if_win)
        assert result.ev == pytest.approx(expected_ev)
        assert result.risk_reward == pytest.approx(1.0)  # profit_if_win == mechanical_max_loss by construction here

    def test_theoretical_max_loss_is_strike_minus_premium_bounded_at_zero_underlying(self):
        """A short put's true worst case is being assigned at $0 underlying:
        you keep the premium but owe the full strike — K - premium per
        share. Checked against a closed-form derivation independent of
        option_strategy.analyze_strategy's own (already-tested) machinery."""
        K = 95.0
        result = evaluate_short_strike(S, K, T, r, q, sigma, "put")
        expected = K - result.premium_received
        assert result.theoretical_max_loss == pytest.approx(expected)

    def test_win_probability_increases_as_strike_moves_further_otm(self):
        """The further below spot the short put's strike sits, the less
        likely it finishes ITM, so the more likely you keep the premium —
        win probability should rise monotonically as strike falls."""
        strikes = [95.0, 90.0, 85.0, 80.0]
        wins = [evaluate_short_strike(S, K, T, r, q, sigma, "put").p_win_risk_neutral for K in strikes]
        assert wins == sorted(wins)  # ascending as strike falls further OTM

    def test_profit_target_above_one_is_capped_not_amplified(self):
        """A short position can never keep more than 100% of the credit it
        received — profit_target_pct > 1.0 must be clipped, not silently
        allowed to imply an impossible >100%-of-credit profit."""
        K = 95.0
        capped = evaluate_short_strike(S, K, T, r, q, sigma, "put", profit_target_pct=2.5)
        normal = evaluate_short_strike(S, K, T, r, q, sigma, "put", profit_target_pct=1.0)
        assert capped.profit_if_win == pytest.approx(normal.profit_if_win)
        assert capped.profit_if_win == pytest.approx(capped.premium_received)


class TestEvaluateShortStrikeCall:
    def test_theoretical_max_loss_is_unbounded(self):
        """A naked short call's true worst case is genuinely unbounded —
        the underlying has no ceiling. This must come through as None, not
        some very large finite number."""
        K = 105.0
        result = evaluate_short_strike(S, K, T, r, q, sigma, "call")
        assert result.theoretical_max_loss is None

    def test_mechanical_max_loss_is_still_finite_despite_unbounded_theoretical_risk(self):
        """The EV/Kelly framing needs a finite number to work with even
        though the true risk is unbounded — mechanical_max_loss (the
        stop-loss-rule loss) must be finite and positive regardless."""
        K = 105.0
        result = evaluate_short_strike(S, K, T, r, q, sigma, "call", RiskConfig(short_stop_loss_multiple=3.0))
        expected = result.premium_received * (3.0 - 1.0)
        assert result.mechanical_max_loss == pytest.approx(expected)
        assert math.isfinite(result.ev)
        assert math.isfinite(result.risk_reward)

    def test_win_probability_complements_the_long_call_itm_probability(self):
        K = 105.0
        short = evaluate_short_strike(S, K, T, r, q, sigma, "call")
        long_itm_prob = itm_probability(S, K, T, r, q, sigma, "call")
        assert short.p_win_risk_neutral == pytest.approx(1.0 - long_itm_prob)

    def test_invalid_stop_loss_multiple_rejected(self):
        with pytest.raises(ValueError):
            evaluate_short_strike(S, 105.0, T, r, q, sigma, "call", RiskConfig(short_stop_loss_multiple=1.0))
        with pytest.raises(ValueError):
            evaluate_short_strike(S, 105.0, T, r, q, sigma, "call", RiskConfig(short_stop_loss_multiple=0.5))


class TestPositionGreeksAreNegatedFromRawContractGreeks:
    def test_short_call_delta_is_negative_of_long_call_delta(self):
        """Position greeks match how option_positions delta is signed
        elsewhere in this codebase (app/data_fetch.py, app/demo_data.py):
        raw_greek * sign(position). A short call's raw delta is positive;
        the position's delta must come through negative."""
        from risk_tool.greeks import all_greeks

        K = 105.0
        raw = all_greeks(S, K, T, r, q, sigma, "call")
        short = evaluate_short_strike(S, K, T, r, q, sigma, "call")
        assert short.delta == pytest.approx(-raw["delta"])
        assert short.vega == pytest.approx(-raw["vega"])


def test_rank_short_strikes_by_ev_sorts_descending_and_covers_every_strike():
    strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    ranked = rank_short_strikes_by_ev(S, T, r, q, sigma, "put", strikes)
    evs = [s.ev for s in ranked]
    assert evs == sorted(evs, reverse=True)
    assert len(ranked) == len(strikes)


class TestCompareImpliedVsRealizedEvShort:
    def test_fair_scenario_realized_equals_implied_gives_no_edge(self):
        """Sanity/null-result check (house style — see
        test_parity_arbitrage.py, test_sigma_moves.py): if your own vol
        view EXACTLY matches market IV, there is no edge to find, and
        edge_ev must come out identical to risk_neutral_ev — not just
        close, but the same numbers, since the math collapses to the same
        formula when realized_or_forecast_vol == market_iv."""
        K = 95.0
        result = compare_implied_vs_realized_ev_short(S, K, T, r, q, market_iv=0.30, realized_or_forecast_vol=0.30, option_type="put")
        assert result.vol_spread == pytest.approx(0.0)
        assert result.edge_ev.ev == pytest.approx(result.risk_neutral_ev.ev)
        assert result.edge_ev.p_win_risk_neutral == pytest.approx(result.risk_neutral_ev.p_win_risk_neutral)
        assert result.edge_ev.premium_received == pytest.approx(result.risk_neutral_ev.premium_received)

    def test_uses_market_iv_and_realized_vol_in_exactly_the_right_places(self):
        K = 95.0
        market_iv = 0.30
        realized = 0.15  # lower than market IV -- the case where selling premium has a real edge
        result = compare_implied_vs_realized_ev_short(S, K, T, r, q, market_iv, realized, "put")

        expected_market_premium = black_scholes_price(S, K, T, r, q, market_iv, "put")
        assert result.market_premium == pytest.approx(expected_market_premium)

        # risk_neutral_ev must be priced AND probability-assigned with market_iv throughout.
        assert result.risk_neutral_ev.premium_received == pytest.approx(expected_market_premium)
        assert result.risk_neutral_ev.p_win_risk_neutral == pytest.approx(itm_probability(S, K, T, r, q, market_iv, "call"))

        # edge_ev must keep the MARKET premium (what you'd actually receive)
        # but assign win probability using the REALIZED/forecast vol.
        assert result.edge_ev.premium_received == pytest.approx(expected_market_premium)
        assert result.edge_ev.p_win_risk_neutral == pytest.approx(itm_probability(S, K, T, r, q, realized, "call"))

        assert result.vol_spread == pytest.approx(realized - market_iv)

    def test_lower_realized_vol_than_market_iv_improves_short_premium_edge(self):
        """Selling premium has an edge when YOUR vol view is LOWER than
        market IV (the market is overpricing how much the underlying will
        move) — the mirror image of the long side's edge condition."""
        K = 95.0
        low_vol_view = compare_implied_vs_realized_ev_short(S, K, T, r, q, market_iv=0.30, realized_or_forecast_vol=0.15, option_type="put")
        high_vol_view = compare_implied_vs_realized_ev_short(S, K, T, r, q, market_iv=0.30, realized_or_forecast_vol=0.45, option_type="put")
        assert low_vol_view.edge_ev.ev > high_vol_view.edge_ev.ev
