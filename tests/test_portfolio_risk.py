"""Correctness tests for risk_tool.portfolio_risk.

Key things to hand-verify: align_returns' inner join and log-return math,
correlation/covariance recovering known values on constructed series,
parametric VaR's w^T Sigma w against a manual 2-asset calculation, historical
VaR's percentile logic on a small deterministic P&L series, and
option_leg_stress_pl's repricing matching direct black_scholes_price calls
(same pattern test_options_lab.py uses -- this module invents no new pricing
math, so its output must be provably consistent with the modules it wraps).
"""
import math

import numpy as np
import pandas as pd
import pytest

from risk_tool.option_strategy import OptionLeg
from risk_tool.pricing import black_scholes_price
from risk_tool.portfolio_risk import (
    Exposure,
    align_returns,
    correlation_matrix,
    covariance_matrix,
    equity_stress_pl,
    historical_var,
    option_leg_stress_pl,
    parametric_var,
)


def _hist(dates, closes):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": closes})


class TestAlignReturns:
    def test_known_log_returns(self):
        dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
        hist = {"AAA": _hist(dates, [100.0, 110.0, 99.0])}
        returns = align_returns(hist)
        expected = [math.log(110.0 / 100.0), math.log(99.0 / 110.0)]
        assert list(returns["AAA"].round(6)) == [round(v, 6) for v in expected]

    def test_inner_joins_on_overlapping_dates_only(self):
        # BBB is missing 2026-01-03 -- that date must drop out of the aligned frame.
        a = _hist(["2026-01-01", "2026-01-02", "2026-01-03"], [100.0, 101.0, 102.0])
        b = _hist(["2026-01-01", "2026-01-02"], [50.0, 51.0])
        returns = align_returns({"AAA": a, "BBB": b})
        assert len(returns) == 1  # only one return computable for BBB, and it's the only overlapping row

    def test_empty_or_short_history_is_skipped_not_raised(self):
        a = _hist(["2026-01-01", "2026-01-02"], [100.0, 101.0])
        b = pd.DataFrame({"date": [], "close": []})
        returns = align_returns({"AAA": a, "BBB": b})
        assert "BBB" not in returns.columns
        assert "AAA" in returns.columns


class TestCorrelationAndCovariance:
    def test_perfectly_correlated_series(self):
        idx = pd.date_range("2026-01-01", periods=50)
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 50)
        returns = pd.DataFrame({"A": r, "B": 2 * r}, index=idx)  # B is an exact linear function of A
        corr = correlation_matrix(returns)
        assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-9)

    def test_covariance_diagonal_matches_variance(self):
        idx = pd.date_range("2026-01-01", periods=30)
        vals = np.linspace(-0.02, 0.02, 30)
        returns = pd.DataFrame({"A": vals, "B": vals[::-1]}, index=idx)
        cov = covariance_matrix(returns)
        assert cov.loc["A", "A"] == pytest.approx(float(pd.Series(vals).var(ddof=1)))

    def test_annualize_scales_by_trading_days(self):
        idx = pd.date_range("2026-01-01", periods=10)
        returns = pd.DataFrame({"A": np.linspace(-0.01, 0.01, 10)}, index=idx)
        raw = covariance_matrix(returns, annualize=False)
        ann = covariance_matrix(returns, annualize=True)
        assert ann.loc["A", "A"] == pytest.approx(raw.loc["A", "A"] * 252)

    def test_single_symbol_raises(self):
        idx = pd.date_range("2026-01-01", periods=5)
        returns = pd.DataFrame({"A": [0.01] * 5}, index=idx)
        with pytest.raises(ValueError):
            correlation_matrix(returns)


class TestParametricVar:
    def test_single_asset_matches_hand_calculation(self):
        # One symbol: portfolio vol = |dollar_delta| * sigma. VaR_95 = 1.6449 * that.
        idx = pd.date_range("2026-01-01", periods=100)
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.02, 100)
        returns = pd.DataFrame({"AAA": r}, index=idx)
        exposures = [Exposure("AAA", dollar_delta=10_000.0)]
        result = parametric_var(exposures, returns, confidence=0.95, horizon_days=1)

        daily_sigma = float(pd.Series(r).std(ddof=1))
        expected_vol = 10_000.0 * daily_sigma
        expected_var = 1.6449 * expected_vol
        assert result.portfolio_dollar_vol == pytest.approx(expected_vol, rel=1e-9)
        assert result.var_dollars == pytest.approx(expected_var, rel=1e-4)
        assert result.diversification_ratio == pytest.approx(1.0)  # only one name -- nothing to diversify

    def test_two_uncorrelated_assets_diversify_below_sum_of_parts(self):
        idx = pd.date_range("2026-01-01", periods=200)
        rng = np.random.default_rng(2)
        a = rng.normal(0, 0.02, 200)
        b = rng.normal(0, 0.02, 200)  # independent of a by construction
        returns = pd.DataFrame({"A": a, "B": b}, index=idx)
        exposures = [Exposure("A", 10_000.0), Exposure("B", 10_000.0)]
        result = parametric_var(exposures, returns)
        assert result.diversification_ratio < 1.0  # uncorrelated names net down vs. naive sum

    def test_perfectly_correlated_assets_do_not_diversify(self):
        idx = pd.date_range("2026-01-01", periods=100)
        rng = np.random.default_rng(3)
        a = rng.normal(0, 0.02, 100)
        returns = pd.DataFrame({"A": a, "B": a * 1.5}, index=idx)  # B is a deterministic function of A -> corr = 1
        exposures = [Exposure("A", 10_000.0), Exposure("B", 10_000.0)]
        result = parametric_var(exposures, returns)
        assert result.diversification_ratio == pytest.approx(1.0, abs=1e-6)

    def test_horizon_scales_by_sqrt_time(self):
        idx = pd.date_range("2026-01-01", periods=60)
        rng = np.random.default_rng(4)
        returns = pd.DataFrame({"A": rng.normal(0, 0.015, 60)}, index=idx)
        exposures = [Exposure("A", 5_000.0)]
        one_day = parametric_var(exposures, returns, horizon_days=1)
        ten_day = parametric_var(exposures, returns, horizon_days=10)
        assert ten_day.var_dollars == pytest.approx(one_day.var_dollars * math.sqrt(10), rel=1e-9)

    def test_missing_symbol_raises(self):
        idx = pd.date_range("2026-01-01", periods=30)
        returns = pd.DataFrame({"A": [0.01] * 30}, index=idx)
        with pytest.raises(ValueError):
            parametric_var([Exposure("ZZZ", 1000.0)], returns)

    def test_invalid_confidence_raises(self):
        idx = pd.date_range("2026-01-01", periods=30)
        returns = pd.DataFrame({"A": [0.01] * 30}, index=idx)
        with pytest.raises(ValueError):
            parametric_var([Exposure("A", 1000.0)], returns, confidence=0.5)


class TestHistoricalVar:
    def test_known_percentile_on_deterministic_series(self):
        # 100 symmetric-ish "returns" -1% to +1%(step) so the 5th percentile of
        # simulated P&L is exactly derivable from np.percentile directly.
        idx = pd.date_range("2026-01-01", periods=100)
        rets = np.linspace(-0.05, 0.05, 100)
        returns = pd.DataFrame({"A": rets}, index=idx)
        exposures = [Exposure("A", 10_000.0)]
        result, pnl = historical_var(exposures, returns, confidence=0.95)

        expected_pnl = rets * 10_000.0
        expected_var = -float(np.percentile(expected_pnl, 5))
        assert result.var_dollars == pytest.approx(max(expected_var, 0.0), rel=1e-9)
        assert list(pnl.round(6)) == [round(v, 6) for v in expected_pnl]

    def test_short_history_raises(self):
        idx = pd.date_range("2026-01-01", periods=5)
        returns = pd.DataFrame({"A": [0.01] * 5}, index=idx)
        with pytest.raises(ValueError):
            historical_var([Exposure("A", 1000.0)], returns)

    def test_horizon_scales_by_sqrt_time(self):
        idx = pd.date_range("2026-01-01", periods=50)
        rng = np.random.default_rng(5)
        returns = pd.DataFrame({"A": rng.normal(0, 0.02, 50)}, index=idx)
        exposures = [Exposure("A", 8_000.0)]
        one_day, _ = historical_var(exposures, returns, horizon_days=1)
        four_day, _ = historical_var(exposures, returns, horizon_days=4)
        assert four_day.var_dollars == pytest.approx(one_day.var_dollars * math.sqrt(4), rel=1e-9)


class TestEquityStressPl:
    def test_long_shares_down_move_loses_money(self):
        assert equity_stress_pl(shares=100.0, spot=50.0, shock_pct=-0.10) == pytest.approx(-500.0)

    def test_short_shares_down_move_gains_money(self):
        assert equity_stress_pl(shares=-100.0, spot=50.0, shock_pct=-0.10) == pytest.approx(500.0)

    def test_zero_shock_is_zero_pl(self):
        assert equity_stress_pl(shares=100.0, spot=50.0, shock_pct=0.0) == 0.0


class TestOptionLegStressPl:
    S, T, r, q = 100.0, 30 / 365.0, 0.03, 0.0

    def _leg(self, option_type="call", strike=100.0, iv=0.30, contracts=1.0):
        return OptionLeg(option_type=option_type, strike=strike, premium=3.0, contracts=contracts, iv=iv)

    def test_zero_shock_is_zero_pl(self):
        leg = self._leg()
        pl = option_leg_stress_pl(leg, self.S, self.T, shock_pct=0.0, r=self.r, q=self.q)
        assert pl == pytest.approx(0.0, abs=1e-9)

    def test_matches_direct_black_scholes_repricing(self):
        leg = self._leg(strike=100.0, iv=0.30, contracts=2.0)
        shock_pct, iv_shock = 0.05, 0.10
        pl = option_leg_stress_pl(leg, self.S, self.T, shock_pct, iv_shock_pts=iv_shock, r=self.r, q=self.q)

        current = black_scholes_price(self.S, 100.0, self.T, self.r, self.q, 0.30, "call")
        shocked = black_scholes_price(self.S * 1.05, 100.0, self.T, self.r, self.q, 0.40, "call")
        expected = (shocked - current) * 2.0 * 100.0
        assert pl == pytest.approx(expected)

    def test_short_leg_negates_long_leg_pl(self):
        long_leg = self._leg(contracts=1.0)
        short_leg = self._leg(contracts=-1.0)
        long_pl = option_leg_stress_pl(long_leg, self.S, self.T, shock_pct=-0.10, r=self.r, q=self.q)
        short_pl = option_leg_stress_pl(short_leg, self.S, self.T, shock_pct=-0.10, r=self.r, q=self.q)
        assert short_pl == pytest.approx(-long_pl)

    def test_down_move_with_vol_spike_helps_a_long_put_more_than_flat_vol(self):
        put = self._leg(option_type="put", strike=100.0, iv=0.30)
        pl_flat_vol = option_leg_stress_pl(put, self.S, self.T, shock_pct=-0.10, iv_shock_pts=0.0, r=self.r, q=self.q)
        pl_vol_spike = option_leg_stress_pl(put, self.S, self.T, shock_pct=-0.10, iv_shock_pts=0.15, r=self.r, q=self.q)
        assert pl_vol_spike > pl_flat_vol  # extra vega tailwind on top of the same spot move

    def test_at_expiration_falls_back_to_intrinsic_and_ignores_iv_shock(self):
        call = self._leg(option_type="call", strike=100.0, iv=0.30)
        pl = option_leg_stress_pl(call, spot=110.0, T_years=0.0, shock_pct=0.0, iv_shock_pts=99.0, r=self.r, q=self.q)
        # current value = intrinsic(110, 100) = 10; shocked spot = 110*(1+0) = 110 -> same intrinsic -> 0 pl
        assert pl == pytest.approx(0.0)

        pl_move = option_leg_stress_pl(call, spot=100.0, T_years=0.0, shock_pct=0.10, iv_shock_pts=99.0, r=self.r, q=self.q)
        # current intrinsic(100,100)=0; shocked spot=110 -> intrinsic=10; ignoring the iv_shock_pts entirely
        assert pl_move == pytest.approx(10.0 * 1.0 * 100.0)

    def test_requires_iv(self):
        leg = OptionLeg(option_type="call", strike=100.0, premium=3.0, contracts=1.0, iv=None)
        with pytest.raises(ValueError):
            option_leg_stress_pl(leg, self.S, self.T, shock_pct=0.05)
