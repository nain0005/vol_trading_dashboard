"""Correctness tests for risk_tool.hedge — direction sign-flipping and the
exposure/beta/hedge math are the properties most worth pinning down."""
import numpy as np
import pandas as pd
import pytest

from risk_tool.hedge import (
    estimate_beta,
    hedge_instrument_pl,
    option_intrinsic_pl,
    option_position_exposure_shares,
    share_position_exposure_shares,
    share_position_pl,
    simple_returns,
    size_hedge,
)


def test_simple_returns_known_values():
    prices = pd.Series([100.0, 110.0, 99.0])
    ret = simple_returns(prices)
    assert list(ret.round(4)) == [0.1, -0.1]


class TestEstimateBeta:
    def test_recovers_known_slope(self):
        # Construct hedge_returns and underlying_returns = 0.5 * hedge_returns exactly (no noise).
        rng = np.random.default_rng(0)
        hedge_ret = pd.Series(rng.normal(0, 0.01, 200))
        underlying_ret = 0.5 * hedge_ret
        est = estimate_beta(underlying_ret, hedge_ret)
        assert est.beta == pytest.approx(0.5, abs=1e-9)
        assert est.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_aligns_on_index_and_uses_only_overlap(self):
        hedge_ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=[0, 1, 2, 3, 4])
        underlying_ret = pd.Series([0.02, 0.04, -0.02, 0.06], index=[0, 1, 2, 3])  # missing index 4
        est = estimate_beta(underlying_ret, hedge_ret)
        assert est.n_obs == 4
        assert est.beta == pytest.approx(2.0, abs=1e-9)

    def test_rejects_too_few_observations(self):
        with pytest.raises(ValueError):
            estimate_beta(pd.Series([0.01, 0.02]), pd.Series([0.01, 0.02]))

    def test_rejects_zero_variance_hedge_returns(self):
        with pytest.raises(ValueError):
            estimate_beta(pd.Series([0.01, 0.02, 0.03]), pd.Series([0.0, 0.0, 0.0]))


class TestExposureHelpers:
    def test_option_position_exposure_shares_long_put_is_negative(self):
        # 5 contracts, delta -0.46 (long put), 100 shares/contract -> -230 shares equivalent
        exposure = option_position_exposure_shares(delta=-0.46, contracts=5, shares_per_contract=100)
        assert exposure == pytest.approx(-230.0)

    def test_share_position_exposure_shares_signs(self):
        assert share_position_exposure_shares(100, "long") == 100.0
        assert share_position_exposure_shares(100, "short") == -100.0

    def test_share_position_exposure_shares_rejects_bad_side(self):
        with pytest.raises(ValueError):
            share_position_exposure_shares(100, "sideways")

    def test_share_position_exposure_shares_rejects_negative_quantity(self):
        with pytest.raises(ValueError):
            share_position_exposure_shares(-10, "long")


class TestSizeHedge:
    def test_short_position_positive_beta_hedges_long(self):
        # Matches the XOM/USO worked example from the risk_tool README-adjacent chat:
        # short 100 shares @ $165, beta 0.3447 -> hedge dollars positive (buy), ~43 USO shares @ $130.44.
        exposure = share_position_exposure_shares(100, "short")
        result = size_hedge(exposure_shares=exposure, underlying_price=165.00, beta=0.3447, hedge_price=130.44)
        assert result.exposure_dollars == pytest.approx(-16_500.0)
        assert result.hedge_dollars > 0
        assert result.direction == "long"
        assert result.hedge_units == pytest.approx(43.3, abs=0.5)

    def test_long_position_positive_beta_hedges_short(self):
        exposure = share_position_exposure_shares(100, "long")
        result = size_hedge(exposure_shares=exposure, underlying_price=165.00, beta=0.3447, hedge_price=130.44)
        assert result.hedge_dollars < 0
        assert result.direction == "short"

    def test_negative_beta_flips_direction(self):
        exposure = share_position_exposure_shares(100, "short")
        result = size_hedge(exposure_shares=exposure, underlying_price=165.00, beta=-0.5, hedge_price=130.44)
        assert result.direction == "short"

    def test_futures_multiplier_divides_units_down(self):
        exposure = option_position_exposure_shares(delta=-0.4576, contracts=5, shares_per_contract=100)
        etf_result = size_hedge(exposure_shares=exposure, underlying_price=165.00, beta=0.3447, hedge_price=65.00, hedge_contract_multiplier=1.0)
        mcl_result = size_hedge(exposure_shares=exposure, underlying_price=165.00, beta=0.3447, hedge_price=65.00, hedge_contract_multiplier=100.0)
        assert mcl_result.hedge_units == pytest.approx(etf_result.hedge_units / 100.0)

    def test_rejects_non_positive_prices_and_multiplier(self):
        with pytest.raises(ValueError):
            size_hedge(exposure_shares=-100, underlying_price=0, beta=0.3, hedge_price=100)
        with pytest.raises(ValueError):
            size_hedge(exposure_shares=-100, underlying_price=100, beta=0.3, hedge_price=0)
        with pytest.raises(ValueError):
            size_hedge(exposure_shares=-100, underlying_price=100, beta=0.3, hedge_price=100, hedge_contract_multiplier=0)


class TestOptionIntrinsicPl:
    def test_matches_the_worked_xom_put_example(self):
        # 12x $162.50 put, entry $1.99, XOM falls 10% to $149.37 at expiration.
        spot_at_scenario = 165.965 * 0.90
        pl = option_intrinsic_pl(spot_at_scenario, strike=162.50, premium=1.99, option_type="put", contracts=12)
        intrinsic = 162.50 - spot_at_scenario
        assert pl == pytest.approx((intrinsic - 1.99) * 12 * 100)
        assert pl > 0  # deep enough ITM to be profitable net of premium

    def test_otm_at_expiration_loses_exactly_the_premium(self):
        # Put expires worthless when spot stays above strike.
        pl = option_intrinsic_pl(200.0, strike=162.50, premium=1.99, option_type="put", contracts=12)
        assert pl == pytest.approx(-1.99 * 12 * 100)

    def test_call_intrinsic_direction(self):
        pl_up = option_intrinsic_pl(220.0, strike=200.0, premium=5.0, option_type="call", contracts=1)
        pl_down = option_intrinsic_pl(180.0, strike=200.0, premium=5.0, option_type="call", contracts=1)
        assert pl_up > 0
        assert pl_down == pytest.approx(-5.0 * 100)

    def test_rejects_invalid_option_type(self):
        with pytest.raises(ValueError):
            option_intrinsic_pl(100.0, strike=100.0, premium=1.0, option_type="straddle", contracts=1)


class TestSharePositionPl:
    def test_long_gains_when_price_rises(self):
        assert share_position_pl(price_at_scenario=110.0, entry_price=100.0, exposure_shares=100) == pytest.approx(1000.0)

    def test_short_gains_when_price_falls(self):
        assert share_position_pl(price_at_scenario=90.0, entry_price=100.0, exposure_shares=-100) == pytest.approx(1000.0)


class TestHedgeInstrumentPl:
    def test_long_hedge_gains_when_hedge_price_rises(self):
        assert hedge_instrument_pl(hedge_price_at_scenario=140.0, hedge_price_now=130.0, hedge_units=163) == pytest.approx(1630.0)

    def test_short_hedge_gains_when_hedge_price_falls(self):
        assert hedge_instrument_pl(hedge_price_at_scenario=120.0, hedge_price_now=130.0, hedge_units=-50) == pytest.approx(500.0)
