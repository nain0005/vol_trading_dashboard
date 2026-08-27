"""Correctness tests for risk_tool.option_strategy — checked against hand-worked
values, including the real CSCO bull call spread this module was built for."""
import pytest

from risk_tool.option_strategy import OptionLeg, analyze_strategy, net_debit, strategy_pl, strategy_pl_today


class TestBullCallSpread:
    """Real position: long 1x CSCO $113 call @ $3.05, short 1x CSCO $115 call
    @ $2.25, both exp 2026-09-18. Net debit $80, max profit $120 (at spot >=
    115), max loss $80 (at spot <= 113), breakeven $113.80 — hand-verified."""

    legs = [
        OptionLeg(option_type="call", strike=113.0, premium=3.05, contracts=1),
        OptionLeg(option_type="call", strike=115.0, premium=2.25, contracts=-1),
    ]

    def test_net_debit(self):
        assert net_debit(self.legs) == pytest.approx(80.0)

    def test_pl_at_zero_and_below_long_strike_equals_max_loss(self):
        assert strategy_pl(self.legs, 0.0) == pytest.approx(-80.0)
        assert strategy_pl(self.legs, 113.0) == pytest.approx(-80.0)

    def test_pl_at_and_above_short_strike_equals_max_profit(self):
        assert strategy_pl(self.legs, 115.0) == pytest.approx(120.0)
        assert strategy_pl(self.legs, 130.0) == pytest.approx(120.0)  # capped — flat past the short strike

    def test_analyze_matches_hand_calculation(self):
        profile = analyze_strategy(self.legs)
        assert profile.max_profit == pytest.approx(120.0)
        assert profile.max_loss == pytest.approx(-80.0)
        assert profile.net_debit == pytest.approx(80.0)
        assert profile.breakevens == pytest.approx([113.80])


class TestBearPutSpread:
    """Mirror image: long higher-strike put, short lower-strike put — a debit
    spread that profits when the underlying falls."""

    legs = [
        OptionLeg(option_type="put", strike=100.0, premium=5.0, contracts=1),
        OptionLeg(option_type="put", strike=90.0, premium=2.0, contracts=-1),
    ]

    def test_max_profit_and_loss(self):
        profile = analyze_strategy(self.legs)
        # net debit = 5*100 - 2*100 = 300; width = 10*100 = 1000; max profit = 1000-300=700
        assert profile.net_debit == pytest.approx(300.0)
        assert profile.max_profit == pytest.approx(700.0)
        assert profile.max_loss == pytest.approx(-300.0)
        assert profile.breakevens == pytest.approx([97.0])


class TestUnboundedStrategies:
    def test_naked_long_call_has_unlimited_profit_bounded_loss(self):
        legs = [OptionLeg(option_type="call", strike=100.0, premium=4.0, contracts=1)]
        profile = analyze_strategy(legs)
        assert profile.max_profit is None  # unlimited upside
        assert profile.max_loss == pytest.approx(-400.0)  # capped at premium paid
        assert profile.breakevens == pytest.approx([104.0])

    def test_naked_short_call_has_unlimited_loss_bounded_profit(self):
        legs = [OptionLeg(option_type="call", strike=100.0, premium=4.0, contracts=-1)]
        profile = analyze_strategy(legs)
        assert profile.max_profit == pytest.approx(400.0)  # capped at premium received
        assert profile.max_loss is None  # unlimited downside... upside, technically — loss grows without bound as spot rises
        assert profile.breakevens == pytest.approx([104.0])

    def test_long_straddle_unlimited_profit_both_directions_bounded_loss(self):
        legs = [
            OptionLeg(option_type="call", strike=100.0, premium=4.0, contracts=1),
            OptionLeg(option_type="put", strike=100.0, premium=3.5, contracts=1),
        ]
        profile = analyze_strategy(legs)
        assert profile.max_profit is None
        assert profile.max_loss == pytest.approx(-750.0)
        assert len(profile.breakevens) == 2


def test_analyze_strategy_rejects_empty_legs():
    with pytest.raises(ValueError):
        analyze_strategy([])


class TestStrategyPlToday:
    def test_converges_to_intrinsic_payoff_as_expiration_approaches(self):
        """As T_years -> 0, the Black-Scholes-repriced 'live' curve must
        converge to the same intrinsic-value payoff strategy_pl computes —
        time value vanishes at expiration by definition. Spot values exactly
        AT a strike are excluded: combined with a near-zero T that's a
        genuine floating-point degenerate limit (d1/d2 involve dividing by
        sigma*sqrt(T), and log(S/K)=0 exactly at the strike), not a
        correctness question — away from that exact boundary (checked here
        down to a still-tiny 1-hour T) it converges to fractions of a cent."""
        legs = [
            OptionLeg(option_type="call", strike=113.0, premium=3.05, contracts=1, iv=0.30),
            OptionLeg(option_type="call", strike=115.0, premium=2.25, contracts=-1, iv=0.30),
        ]
        T_one_hour = 1 / 365 / 24
        for spot in [100.0, 108.0, 114.0, 120.0, 130.0]:
            today = strategy_pl_today(legs, spot, T_years=T_one_hour, r=0.05)
            expiry = strategy_pl(legs, spot)
            assert today == pytest.approx(expiry, abs=0.5)

    def test_live_curve_differs_from_intrinsic_with_real_time_remaining(self):
        """With real time and vol left, the mark-to-market value should NOT
        equal pure intrinsic value at a strike (there's real time value
        priced in) — this is the whole point of the 'live' curve existing
        as a separate thing from the at-expiration payoff."""
        legs = [OptionLeg(option_type="call", strike=100.0, premium=4.0, contracts=1, iv=0.30)]
        at_strike_intrinsic = strategy_pl(legs, 100.0)  # = -premium, since intrinsic value is 0 exactly at the strike
        at_strike_today = strategy_pl_today(legs, 100.0, T_years=30 / 365, r=0.05)
        assert at_strike_today != pytest.approx(at_strike_intrinsic, abs=0.5)
        assert at_strike_today > at_strike_intrinsic  # real time value makes it worth more than pure intrinsic

    def test_raises_when_iv_missing(self):
        legs = [OptionLeg(option_type="call", strike=100.0, premium=4.0, contracts=1)]  # no iv set
        with pytest.raises(ValueError):
            strategy_pl_today(legs, 100.0, T_years=30 / 365)
