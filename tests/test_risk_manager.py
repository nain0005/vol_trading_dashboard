"""Correctness tests for risk_tool.risk_manager — each rule is tested for
both its triggered and not-triggered branch, using numbers you can check
by hand."""
import pytest

from risk_tool.config import RiskConfig
from risk_tool.risk_manager import (
    Position,
    check_delta_exit,
    check_iv_crush_exit,
    check_portfolio_governors,
    check_profit_target,
    check_stop_loss,
    check_theta_exit,
    PortfolioState,
    pct_pnl,
)

CONFIG = RiskConfig(
    profit_target_pct=1.00,
    stop_loss_pct=-0.50,
    delta_exit_threshold=0.15,
    theta_exit_dte_threshold=3,
    iv_crush_threshold_points=0.20,
)


def make_position(**overrides) -> Position:
    defaults = dict(
        symbol="TEST",
        option_type="call",
        strike=100.0,
        spot=105.0,
        dte=20,
        quantity=1,
        entry_premium=2.00,
        current_premium=2.00,
        entry_iv=0.30,
        current_iv=0.30,
        delta=0.50,
    )
    defaults.update(overrides)
    return Position(**defaults)


def test_pct_pnl_basic():
    pos = make_position(entry_premium=2.00, current_premium=3.00)
    assert pct_pnl(pos) == pytest.approx(0.5)


class TestProfitTarget:
    def test_triggers_at_exactly_100pct_gain(self):
        pos = make_position(entry_premium=2.00, current_premium=4.00)  # +100%
        signal = check_profit_target(pos, CONFIG)
        assert signal.triggered is True
        assert "100%" in signal.reason or "1" in signal.reason

    def test_does_not_trigger_below_target(self):
        pos = make_position(entry_premium=2.00, current_premium=3.00)  # +50%
        assert check_profit_target(pos, CONFIG).triggered is False


class TestStopLoss:
    def test_triggers_at_exactly_neg_50pct(self):
        pos = make_position(entry_premium=2.00, current_premium=1.00)  # -50%
        assert check_stop_loss(pos, CONFIG).triggered is True

    def test_does_not_trigger_above_stop(self):
        pos = make_position(entry_premium=2.00, current_premium=1.50)  # -25%
        assert check_stop_loss(pos, CONFIG).triggered is False


class TestDeltaExit:
    def test_triggers_when_delta_decays_below_threshold(self):
        pos = make_position(delta=0.10)
        assert check_delta_exit(pos, CONFIG).triggered is True

    def test_does_not_trigger_with_healthy_delta(self):
        pos = make_position(delta=0.45)
        assert check_delta_exit(pos, CONFIG).triggered is False

    def test_uses_absolute_value_for_puts(self):
        pos = make_position(option_type="put", delta=-0.10)
        assert check_delta_exit(pos, CONFIG).triggered is True


class TestThetaExit:
    def test_triggers_when_near_expiry_and_otm_call(self):
        pos = make_position(option_type="call", spot=95.0, strike=100.0, dte=2)  # OTM call, 2 DTE
        assert check_theta_exit(pos, CONFIG).triggered is True

    def test_does_not_trigger_when_near_expiry_but_itm(self):
        pos = make_position(option_type="call", spot=110.0, strike=100.0, dte=2)  # ITM call, 2 DTE
        assert check_theta_exit(pos, CONFIG).triggered is False

    def test_does_not_trigger_when_otm_but_plenty_of_time(self):
        pos = make_position(option_type="call", spot=95.0, strike=100.0, dte=30)
        assert check_theta_exit(pos, CONFIG).triggered is False

    def test_put_otm_direction_is_correct(self):
        # A put is OTM when spot > strike.
        pos = make_position(option_type="put", spot=110.0, strike=100.0, dte=1)
        assert check_theta_exit(pos, CONFIG).triggered is True
        pos_itm = make_position(option_type="put", spot=90.0, strike=100.0, dte=1)
        assert check_theta_exit(pos_itm, CONFIG).triggered is False


class TestIvCrushExit:
    def test_triggers_when_iv_drops_past_threshold(self):
        pos = make_position(entry_iv=0.50, current_iv=0.25)  # 25-point drop
        assert check_iv_crush_exit(pos, CONFIG).triggered is True

    def test_does_not_trigger_on_small_iv_move(self):
        pos = make_position(entry_iv=0.50, current_iv=0.45)  # 5-point drop
        assert check_iv_crush_exit(pos, CONFIG).triggered is False

    def test_does_not_trigger_when_iv_rises(self):
        pos = make_position(entry_iv=0.30, current_iv=0.45)
        assert check_iv_crush_exit(pos, CONFIG).triggered is False


class TestPortfolioGovernors:
    def test_halts_when_daily_loss_limit_breached(self):
        config = RiskConfig(daily_max_loss_pct=0.03)
        state = PortfolioState(account_size=100_000, daily_pnl=-3_500, net_delta_dollars=0, net_vega_dollars=0)
        result = check_portfolio_governors(state, config)
        assert result.halted is True
        assert any("Daily P&L" in r for r in result.reasons)

    def test_does_not_halt_within_daily_loss_limit(self):
        config = RiskConfig(daily_max_loss_pct=0.03)
        state = PortfolioState(account_size=100_000, daily_pnl=-1_000, net_delta_dollars=0, net_vega_dollars=0)
        result = check_portfolio_governors(state, config)
        assert result.halted is False

    def test_delta_cap_enforced_when_set(self):
        config = RiskConfig(daily_max_loss_pct=1.0, max_net_delta_dollars=10_000)
        state = PortfolioState(account_size=100_000, daily_pnl=0, net_delta_dollars=15_000, net_vega_dollars=0)
        result = check_portfolio_governors(state, config)
        assert result.halted is True
        assert any("delta" in r.lower() for r in result.reasons)

    def test_delta_cap_not_enforced_when_unset(self):
        config = RiskConfig(daily_max_loss_pct=1.0, max_net_delta_dollars=None)
        state = PortfolioState(account_size=100_000, daily_pnl=0, net_delta_dollars=999_999, net_vega_dollars=0)
        result = check_portfolio_governors(state, config)
        assert result.halted is False

    def test_vega_cap_enforced_when_set(self):
        config = RiskConfig(daily_max_loss_pct=1.0, max_net_vega_dollars=500)
        state = PortfolioState(account_size=100_000, daily_pnl=0, net_delta_dollars=0, net_vega_dollars=800)
        result = check_portfolio_governors(state, config)
        assert result.halted is True
        assert any("vega" in r.lower() for r in result.reasons)
