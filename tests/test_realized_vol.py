"""Correctness tests for risk_tool.realized_vol.

For the range estimators (Parkinson, Garman-Klass) we cross-check the
module's output against the same formula recomputed independently in the
test, on a small hand-built OHLC sample — this catches implementation bugs
(off-by-one in n, wrong annualization, wrong log base) without relying on
recovering a "true" simulated vol, which is inherently noisy and would make
the test flaky.
"""
import math

import numpy as np
import pandas as pd
import pytest

from risk_tool.realized_vol import (
    EGarchFit,
    close_to_close_vol,
    egarch_forecast_vol,
    fit_egarch_11,
    fit_garch_11,
    garch_forecast_vol,
    garman_klass_vol,
    parkinson_vol,
)


def test_close_to_close_vol_matches_independent_calculation():
    closes = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0])
    expected_log_returns = np.diff(np.log(closes.to_numpy()))
    expected = expected_log_returns.std(ddof=1) * math.sqrt(252)
    assert close_to_close_vol(closes, annualization_factor=252) == pytest.approx(expected, rel=1e-9)


def test_close_to_close_vol_requires_enough_data():
    with pytest.raises(ValueError):
        close_to_close_vol(pd.Series([100.0]))


def test_parkinson_matches_independent_calculation():
    high = pd.Series([102.0, 105.0, 103.0])
    low = pd.Series([99.0, 101.0, 100.0])
    log_hl = np.log(high.to_numpy() / low.to_numpy())
    expected_var = np.sum(log_hl**2) / (4 * len(high) * math.log(2))
    expected = math.sqrt(expected_var * 252)
    assert parkinson_vol(high, low, annualization_factor=252) == pytest.approx(expected, rel=1e-9)


def test_parkinson_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        parkinson_vol(pd.Series([100.0, 101.0]), pd.Series([99.0]))


def test_garman_klass_matches_independent_calculation():
    open_ = pd.Series([100.0, 102.0, 101.0])
    high = pd.Series([103.0, 104.0, 102.5])
    low = pd.Series([99.0, 101.0, 100.0])
    close = pd.Series([102.0, 101.5, 102.0])

    log_hl = np.log(high.to_numpy() / low.to_numpy())
    log_co = np.log(close.to_numpy() / open_.to_numpy())
    term = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
    expected = math.sqrt((np.sum(term) / len(close)) * 252)

    assert garman_klass_vol(open_, high, low, close, annualization_factor=252) == pytest.approx(expected, rel=1e-9)


def test_garman_klass_and_parkinson_agree_in_magnitude_on_realistic_data():
    """Sanity check, not a precision check: both range estimators should
    land in the same ballpark on the same realistic-looking data, since
    they're measuring the same underlying phenomenon."""
    rng = np.random.default_rng(42)
    n = 100
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))

    p = parkinson_vol(pd.Series(high), pd.Series(low))
    gk = garman_klass_vol(pd.Series(open_), pd.Series(high), pd.Series(low), pd.Series(close))
    assert 0.05 < p < 1.0
    assert 0.05 < gk < 1.0
    assert p == pytest.approx(gk, rel=0.5)  # same ballpark, not a precision match


class TestGarch:
    @pytest.fixture
    def returns(self):
        """Synthetic returns with genuine volatility clustering (a GARCH-like
        process), so the fit has real structure to recover rather than pure
        noise."""
        rng = np.random.default_rng(7)
        n = 800
        omega, alpha, beta = 1e-6, 0.08, 0.88
        sigma2 = np.empty(n)
        r = np.empty(n)
        sigma2[0] = omega / (1 - alpha - beta)
        r[0] = rng.normal(0, math.sqrt(sigma2[0]))
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
            r[t] = rng.normal(0, math.sqrt(sigma2[t]))
        return pd.Series(r)

    def test_fit_is_stationary_and_well_posed(self, returns):
        fit = fit_garch_11(returns)
        assert fit.omega > 0
        assert fit.alpha >= 0
        assert fit.beta >= 0
        assert fit.persistence < 1.0  # stationarity — required for long_run_variance to make sense
        assert fit.long_run_variance > 0

    def test_requires_minimum_history(self):
        with pytest.raises(ValueError):
            fit_garch_11(pd.Series(np.random.default_rng(0).normal(0, 0.01, 10)))

    def test_forecast_converges_to_long_run_vol_at_long_horizon(self, returns):
        fit = fit_garch_11(returns)
        long_horizon_vol = garch_forecast_vol(fit, horizon_days=5000)
        expected_long_run_vol = math.sqrt(fit.long_run_variance * 252)
        assert long_horizon_vol == pytest.approx(expected_long_run_vol, rel=1e-3)

    def test_one_step_forecast_is_positive_and_finite(self, returns):
        fit = fit_garch_11(returns)
        vol = garch_forecast_vol(fit, horizon_days=1)
        assert vol > 0
        assert math.isfinite(vol)

    def test_rejects_invalid_horizon(self, returns):
        fit = fit_garch_11(returns)
        with pytest.raises(ValueError):
            garch_forecast_vol(fit, horizon_days=0)


class TestEgarch:
    @pytest.fixture
    def returns(self):
        """Synthetic returns from a genuine EGARCH(1,1) process with a real
        negative gamma (leverage effect), so the fit has real asymmetric
        structure to recover rather than pure noise."""
        rng = np.random.default_rng(11)
        n = 1000
        omega, alpha, beta, gamma = -0.10, 0.10, 0.95, -0.08
        e_abs_z = math.sqrt(2 / math.pi)
        ln_sigma2 = np.empty(n)
        r = np.empty(n)
        ln_sigma2[0] = omega / (1 - beta)
        z0 = rng.normal(0, 1)
        r[0] = z0 * math.sqrt(math.exp(ln_sigma2[0]))
        z_prev = z0
        for t in range(1, n):
            ln_sigma2[t] = omega + beta * ln_sigma2[t - 1] + alpha * (abs(z_prev) - e_abs_z) + gamma * z_prev
            z_prev = rng.normal(0, 1)
            r[t] = z_prev * math.sqrt(math.exp(ln_sigma2[t]))
        return pd.Series(r)

    def test_fit_is_stationary_and_well_posed(self, returns):
        fit = fit_egarch_11(returns)
        assert -1.0 < fit.beta < 1.0
        assert fit.persistence < 1.0
        assert fit.long_run_variance > 0

    def test_recovers_a_negative_gamma_from_data_with_real_leverage_effect(self, returns):
        """The whole point of EGARCH over GARCH: fit on data actually
        generated with a leverage effect (gamma=-0.08 in the fixture above)
        should recover a negative gamma, not just noise around zero."""
        fit = fit_egarch_11(returns)
        assert fit.gamma < 0

    def test_requires_minimum_history(self):
        with pytest.raises(ValueError):
            fit_egarch_11(pd.Series(np.random.default_rng(0).normal(0, 0.01, 10)))

    def test_rejects_zero_variance_returns(self):
        with pytest.raises(ValueError):
            fit_egarch_11(pd.Series([0.01] * 40))

    def test_forecast_converges_to_long_run_vol_at_long_horizon(self, returns):
        fit = fit_egarch_11(returns)
        long_horizon_vol = egarch_forecast_vol(fit, horizon_days=5000)
        expected_long_run_vol = math.sqrt(fit.long_run_variance * 252)
        assert long_horizon_vol == pytest.approx(expected_long_run_vol, rel=1e-2)

    def test_one_step_forecast_is_positive_and_finite(self, returns):
        fit = fit_egarch_11(returns)
        vol = egarch_forecast_vol(fit, horizon_days=1)
        assert vol > 0
        assert math.isfinite(vol)

    def test_rejects_invalid_horizon(self, returns):
        fit = fit_egarch_11(returns)
        with pytest.raises(ValueError):
            egarch_forecast_vol(fit, horizon_days=0)

    def test_leverage_effect_a_down_move_raises_the_forecast_more_than_an_equal_up_move(self):
        """This is the actual behavior EGARCH exists for — direct answer to
        'the stock just fell 9% and 5%, does that raise my near-term vol
        forecast more than a +9%/+5% rally would have?'. Build two identical
        fitted models differing only in the SIGN of the most recent return,
        with a realistic negative gamma, and check the down-move forecast
        comes out higher."""
        common = dict(omega=-0.10, alpha=0.10, beta=0.90, gamma=-0.15, mean_return=0.0)
        prior_ln_var = math.log(0.04 / 252)  # ~20% annualized vol level going into the shock

        down_fit = EGarchFit(
            **common,
            demeaned_returns=np.array([-0.09]),
            log_conditional_variance=np.array([prior_ln_var]),
        )
        up_fit = EGarchFit(
            **common,
            demeaned_returns=np.array([0.09]),
            log_conditional_variance=np.array([prior_ln_var]),
        )

        down_vol = egarch_forecast_vol(down_fit, horizon_days=1)
        up_vol = egarch_forecast_vol(up_fit, horizon_days=1)
        assert down_vol > up_vol
