"""Correctness tests for risk_tool.sizing — the hard-cap-always-wins
guarantee is the most important property to verify here."""
import pytest

from risk_tool.config import RiskConfig
from risk_tool.sizing import half_kelly_fraction, kelly_fraction, size_position, size_short_position


def test_kelly_fraction_known_formula_value():
    # p=0.6, b=1 -> f* = (0.6*1 - 0.4)/1 = 0.2
    assert kelly_fraction(p_win=0.6, b=1.0) == pytest.approx(0.2)


def test_kelly_fraction_negative_when_no_edge():
    # p=0.4, b=1 -> f* = (0.4 - 0.6)/1 = -0.2 — "don't take this bet"
    assert kelly_fraction(p_win=0.4, b=1.0) == pytest.approx(-0.2)


def test_kelly_fraction_scales_with_better_odds():
    assert kelly_fraction(p_win=0.5, b=3.0) > kelly_fraction(p_win=0.5, b=1.0)


def test_kelly_fraction_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        kelly_fraction(p_win=1.5, b=1.0)
    with pytest.raises(ValueError):
        kelly_fraction(p_win=0.5, b=0.0)


def test_half_kelly_is_exactly_half_of_full_kelly():
    assert half_kelly_fraction(0.6, 1.0) == pytest.approx(kelly_fraction(0.6, 1.0) / 2)


class TestSizePosition:
    def test_hard_cap_always_wins_when_kelly_suggests_more(self):
        """Construct a huge apparent edge (high p_win, big payout) so full
        Kelly would suggest betting a large fraction of the account — the
        hard cap must still win."""
        config = RiskConfig(use_half_kelly=False, max_position_pct_of_account=0.05)
        result = size_position(
            account_size=100_000,
            premium_per_contract=1.0,
            p_win=0.95,
            profit_if_win_per_contract=5.0,  # b = 5, huge edge at p=0.95
            config=config,
        )
        assert result.capped_by_hard_limit is True
        assert result.recommended_dollar_size == pytest.approx(5_000)  # 5% of 100k
        assert result.kelly_dollar_size > result.recommended_dollar_size

    def test_kelly_wins_when_smaller_than_cap(self):
        """A modest, realistic edge should size well under the 5% cap —
        Kelly's (halved) suggestion should be what's actually used.

        p=0.52, b=1 -> full Kelly f*=(0.52-0.48)/1=0.04, half-Kelly=0.02 (2%),
        clear of the 5% cap (chosen deliberately, not coincidentally equal to it)."""
        config = RiskConfig(use_half_kelly=True, max_position_pct_of_account=0.05)
        result = size_position(
            account_size=100_000,
            premium_per_contract=2.0,
            p_win=0.52,
            profit_if_win_per_contract=2.0,  # b = 1
            config=config,
        )
        assert result.capped_by_hard_limit is False
        assert result.recommended_dollar_size == pytest.approx(result.kelly_dollar_size)
        assert result.recommended_dollar_size < 5_000

    def test_negative_edge_floors_to_zero_size(self):
        config = RiskConfig(use_half_kelly=True)
        result = size_position(
            account_size=100_000,
            premium_per_contract=2.0,
            p_win=0.3,
            profit_if_win_per_contract=2.0,  # b = 1, p=0.3 -> negative Kelly
            config=config,
        )
        assert result.kelly_fraction_raw < 0
        assert result.kelly_fraction_used == 0.0
        assert result.recommended_dollar_size == 0.0
        assert result.recommended_contracts == 0

    def test_half_kelly_config_flag_actually_halves_the_size(self):
        common_kwargs = dict(account_size=100_000, premium_per_contract=2.0, p_win=0.6, profit_if_win_per_contract=2.0)
        full = size_position(**common_kwargs, config=RiskConfig(use_half_kelly=False, max_position_pct_of_account=1.0))
        half = size_position(**common_kwargs, config=RiskConfig(use_half_kelly=True, max_position_pct_of_account=1.0))
        assert half.kelly_dollar_size == pytest.approx(full.kelly_dollar_size / 2)

    def test_recommended_contracts_is_floor_of_dollar_size(self):
        config = RiskConfig(use_half_kelly=False, max_position_pct_of_account=0.05)
        result = size_position(account_size=10_000, premium_per_contract=1.37, p_win=0.6, profit_if_win_per_contract=1.37, config=config)
        contract_cost = 1.37 * 100
        assert result.recommended_contracts == int(result.recommended_dollar_size // contract_cost)

    def test_rejects_non_positive_account_or_premium(self):
        with pytest.raises(ValueError):
            size_position(account_size=0, premium_per_contract=1.0, p_win=0.5, profit_if_win_per_contract=1.0)
        with pytest.raises(ValueError):
            size_position(account_size=10_000, premium_per_contract=0, p_win=0.5, profit_if_win_per_contract=1.0)


class TestSizeShortPosition:
    def test_uses_mechanical_max_loss_not_premium_received_as_kellys_denominator(self):
        """The whole point of size_short_position: it must size off
        mechanical_max_loss_per_contract, NOT premium_received_per_contract.
        Construct the two so they'd give very different answers if the
        wrong one were used, and check against a hand-computed Kelly value
        tied to the CORRECT (mechanical_max_loss) denominator.

        premium_received=5.0 (what you'd collect), mechanical_max_loss=1.0
        (your stop-loss-rule risk), profit_if_win=1.0 -> b = 1.0 (using the
        correct denominator). p=0.6, b=1 -> f* = (0.6*1-0.4)/1 = 0.2, same
        known formula as test_kelly_fraction_known_formula_value above. If
        premium_received (5.0) were used instead, b would be 0.2 and f*
        would come out negative (no edge) -- a completely different,
        wrong, answer.
        """
        config = RiskConfig(use_half_kelly=False, max_position_pct_of_account=1.0)
        result = size_short_position(
            account_size=100_000,
            premium_received_per_contract=5.0,
            mechanical_max_loss_per_contract=1.0,
            p_win=0.6,
            profit_if_win_per_contract=1.0,
            config=config,
        )
        assert result.kelly_fraction_raw == pytest.approx(0.2)
        # Kelly dollars = f* * account_size; contract cost is mechanical_max_loss * 100, not premium_received * 100.
        assert result.kelly_dollar_size == pytest.approx(0.2 * 100_000)

    def test_matches_size_position_called_directly_with_mechanical_loss_as_premium(self):
        """size_short_position should be a pure, transparent remapping —
        identical output to calling size_position with
        premium_per_contract=mechanical_max_loss_per_contract directly."""
        config = RiskConfig(use_half_kelly=True, max_position_pct_of_account=0.05)
        direct = size_position(account_size=50_000, premium_per_contract=2.0, p_win=0.55, profit_if_win_per_contract=3.0, config=config)
        via_wrapper = size_short_position(
            account_size=50_000,
            premium_received_per_contract=9.0,  # deliberately different from mechanical_max_loss -- must be ignored by the Kelly math
            mechanical_max_loss_per_contract=2.0,
            p_win=0.55,
            profit_if_win_per_contract=3.0,
            config=config,
        )
        assert via_wrapper == direct

    def test_rejects_non_positive_mechanical_max_loss(self):
        with pytest.raises(ValueError):
            size_short_position(
                account_size=100_000,
                premium_received_per_contract=5.0,
                mechanical_max_loss_per_contract=0.0,
                p_win=0.6,
                profit_if_win_per_contract=1.0,
            )

    def test_naked_short_call_with_unbounded_theoretical_risk_still_sizes_finitely(self):
        """The scenario this function exists for: a short call's true max
        loss is unbounded (no finite dollar amount for that), but sizing
        under the mechanical stop-loss rule must still produce a finite,
        sane recommendation rather than crashing or returning inf/nan."""
        import math

        result = size_short_position(
            account_size=100_000,
            premium_received_per_contract=3.0,
            mechanical_max_loss_per_contract=3.0,  # short_stop_loss_multiple=2.0 -> loss = premium*(2-1)
            p_win=0.75,
            profit_if_win_per_contract=3.0,
        )
        assert math.isfinite(result.recommended_dollar_size)
        assert math.isfinite(result.kelly_dollar_size)
        assert result.recommended_contracts >= 0
