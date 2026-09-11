"""Correctness tests for app.performance -- FIFO round-trip matching and
win-rate stats, on hand-constructed synthetic fill data (no live Robinhood
session needed)."""
import pandas as pd
import pytest

from app.performance import match_round_trips, round_trips_to_frame, win_rate_by_instrument_type, win_rate_stats


def fill(date, symbol, side, qty, price, instrument_type="equity", contract_id=None):
    row = {"date": pd.Timestamp(date, tz="UTC"), "symbol": symbol, "instrument_type": instrument_type, "side": side, "quantity": qty, "price": price}
    if contract_id is not None:
        row["contract_id"] = contract_id
    return row


def frame(*fills):
    return pd.DataFrame(list(fills))


class TestMatchRoundTrips:
    def test_empty_history_returns_no_trips(self):
        assert match_round_trips(pd.DataFrame()) == []

    def test_simple_long_round_trip(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-05", "AAPL", "sell", 10, 110.0),
        )
        trips = match_round_trips(hist)
        assert len(trips) == 1
        t = trips[0]
        assert t.direction == "long"
        assert t.quantity == 10
        assert t.realized_pl == pytest.approx(100.0)  # (110-100)*10
        assert t.is_win is True

    def test_simple_short_round_trip(self):
        hist = frame(
            fill("2026-01-01", "TSLA", "sell", 5, 200.0),
            fill("2026-01-05", "TSLA", "buy", 5, 180.0),
        )
        trips = match_round_trips(hist)
        assert len(trips) == 1
        t = trips[0]
        assert t.direction == "short"
        assert t.realized_pl == pytest.approx(100.0)  # (200-180)*5
        assert t.is_win is True

    def test_losing_trade_is_flagged_not_a_win(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-05", "AAPL", "sell", 10, 90.0),
        )
        trips = match_round_trips(hist)
        assert trips[0].realized_pl == pytest.approx(-100.0)
        assert trips[0].is_win is False

    def test_fifo_matches_oldest_lot_first(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-02", "AAPL", "buy", 10, 120.0),
            fill("2026-01-05", "AAPL", "sell", 10, 130.0),
        )
        trips = match_round_trips(hist)
        assert len(trips) == 1
        assert trips[0].entry_price == pytest.approx(100.0)  # matched the FIRST lot, not the second
        assert trips[0].quantity == 10

    def test_partial_fill_splits_into_multiple_trips(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-02", "AAPL", "buy", 10, 120.0),
            fill("2026-01-05", "AAPL", "sell", 15, 130.0),
        )
        trips = match_round_trips(hist)
        assert len(trips) == 2
        assert trips[0].quantity == 10 and trips[0].entry_price == pytest.approx(100.0)
        assert trips[1].quantity == 5 and trips[1].entry_price == pytest.approx(120.0)

    def test_option_trade_applies_100x_multiplier(self):
        hist = frame(
            fill("2026-01-01", "SPY", "buy", 2, 5.0, instrument_type="option"),
            fill("2026-01-05", "SPY", "sell", 2, 7.0, instrument_type="option"),
        )
        trips = match_round_trips(hist)
        assert trips[0].realized_pl == pytest.approx((7.0 - 5.0) * 2 * 100)

    def test_different_symbols_are_matched_independently(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-01", "TSLA", "buy", 5, 200.0),
            fill("2026-01-05", "AAPL", "sell", 10, 110.0),
        )
        trips = match_round_trips(hist)
        assert len(trips) == 1
        assert trips[0].symbol == "AAPL"  # TSLA still open, not closed, so not a round trip yet

    def test_unmatched_open_position_produces_no_trip(self):
        hist = frame(fill("2026-01-01", "AAPL", "buy", 10, 100.0))
        assert match_round_trips(hist) == []

    def test_two_different_contracts_on_same_underlying_do_not_cross_match(self):
        # Two different AAPL option contracts (different strikes) with
        # OVERLAPPING open windows -- exactly the scenario that corrupted
        # results before contract_id-based grouping existed. Contract A is
        # a clean winner, contract B a clean loser; if they were wrongly
        # matched together (old symbol-only behavior) the fills would pair
        # up in fill order instead, producing different trips/P&L.
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 5, 2.00, instrument_type="option", contract_id="AAPL-150C"),
            fill("2026-01-02", "AAPL", "buy", 5, 9.00, instrument_type="option", contract_id="AAPL-160C"),
            fill("2026-01-03", "AAPL", "sell", 5, 3.00, instrument_type="option", contract_id="AAPL-150C"),  # A: win
            fill("2026-01-04", "AAPL", "sell", 5, 8.00, instrument_type="option", contract_id="AAPL-160C"),  # B: loss
        )
        trips = match_round_trips(hist)
        assert len(trips) == 2
        by_contract = {t.entry_price: t for t in trips}
        assert by_contract[2.00].exit_price == pytest.approx(3.00)  # A matched with A, not B
        assert by_contract[2.00].is_win is True
        assert by_contract[9.00].exit_price == pytest.approx(8.00)  # B matched with B, not A
        assert by_contract[9.00].is_win is False

    def test_missing_contract_id_column_falls_back_to_symbol_only_grouping(self):
        # No contract_id column at all (e.g. an older cached order_history) --
        # must not KeyError, and behaves like the pre-fix symbol-only match.
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0, instrument_type="option"),
            fill("2026-01-05", "AAPL", "sell", 10, 110.0, instrument_type="option"),
        )
        trips = match_round_trips(hist)
        assert len(trips) == 1


class TestWinRateStats:
    def test_empty_trips_returns_none_fields_not_zero(self):
        stats = win_rate_stats(round_trips_to_frame([]))
        assert stats["total_trades"] == 0
        assert stats["win_rate"] is None
        assert stats["avg_win"] is None

    def test_mixed_wins_and_losses(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-02", "AAPL", "sell", 10, 110.0),  # win: +100
            fill("2026-01-03", "TSLA", "buy", 10, 200.0),
            fill("2026-01-04", "TSLA", "sell", 10, 190.0),  # loss: -100
        )
        trips_df = round_trips_to_frame(match_round_trips(hist))
        stats = win_rate_stats(trips_df)
        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["avg_win"] == pytest.approx(100.0)
        assert stats["avg_loss"] == pytest.approx(-100.0)
        assert stats["profit_factor"] == pytest.approx(1.0)

    def test_all_wins_has_no_defined_profit_factor_denominator_but_returns_none_for_no_losses(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-02", "AAPL", "sell", 10, 110.0),
        )
        stats = win_rate_stats(round_trips_to_frame(match_round_trips(hist)))
        assert stats["win_rate"] == 1.0
        assert stats["profit_factor"] is None  # zero gross loss -- undefined, not infinity


class TestWinRateByInstrumentType:
    def test_splits_stats_per_instrument_type(self):
        hist = frame(
            fill("2026-01-01", "AAPL", "buy", 10, 100.0),
            fill("2026-01-02", "AAPL", "sell", 10, 110.0, instrument_type="equity"),
            fill("2026-01-03", "SPY", "buy", 1, 5.0, instrument_type="option"),
            fill("2026-01-04", "SPY", "sell", 1, 4.0, instrument_type="option"),
        )
        by_type = win_rate_by_instrument_type(round_trips_to_frame(match_round_trips(hist)))
        row = {r["instrument_type"]: r for _, r in by_type.iterrows()}
        assert row["equity"]["win_rate"] == pytest.approx(1.0)
        assert row["option"]["win_rate"] == pytest.approx(0.0)
