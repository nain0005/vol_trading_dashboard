"""Correctness tests for the pure helpers in app.data_fetch — no robin_stocks
mocking needed since these don't touch the network, except where noted."""
from unittest.mock import patch

import pytest

from app.data_fetch import get_open_orders, option_unrealized_pl


def test_long_leg_profits_when_mark_rises_above_entry():
    # Bought at $3.00, now worth $5.00 -> +$200 on 1 contract.
    assert option_unrealized_pl("long", avg_price=3.0, mark_price=5.0, qty=1) == pytest.approx(200.0)


def test_long_leg_loses_when_mark_falls_below_entry():
    assert option_unrealized_pl("long", avg_price=5.0, mark_price=3.0, qty=1) == pytest.approx(-200.0)


def test_short_leg_profits_when_mark_falls_below_entry_credit():
    # Sold for a $7.58 credit, now worth $5.20 -> profit (816) on 5 contracts... (7.58-5.20)*5*100
    assert option_unrealized_pl("short", avg_price=-758.0 / 100, mark_price=5.20, qty=5) == pytest.approx((7.58 - 5.20) * 5 * 100)


def test_short_leg_loses_when_mark_rises_above_entry_credit():
    # Sold for $2.00, now worth $6.00 -> lost $400 on 1 contract.
    assert option_unrealized_pl("short", avg_price=-2.0, mark_price=6.0, qty=1) == pytest.approx(-400.0)


def test_crwd_bear_put_spread_regression():
    """Reproduces the exact bug report: a same-day bear put spread showing an
    impossible -$7,618 loss on a position with a $1,738 max loss. The old
    formula (mark_price - avg_price) * signed_qty * multiplier double-applied
    Robinhood's sign convention on the short leg, adding a phantom -$7,580
    (= 2 * 7.58 * 5 * 100) regardless of the actual mark price — this checks
    the fixed short-leg formula no longer does that, using a mark price
    representative of a same-day open (near the entry price)."""
    long_pl = option_unrealized_pl("long", avg_price=1105.6 / 100, mark_price=11.20, qty=5)  # tiny move since open
    short_pl = option_unrealized_pl("short", avg_price=-758.0 / 100, mark_price=7.70, qty=5)  # tiny move since open
    combined = long_pl + short_pl
    # A same-day, small-move mark-to-market P&L should be a small number —
    # nowhere near the -7618 the bug produced, and well within the spread's
    # own [-1738, +2012] max-loss/max-profit bounds.
    assert -1738.0 <= combined <= 2012.0
    assert abs(combined) < 500.0


def test_multiplier_scales_linearly():
    assert option_unrealized_pl("long", avg_price=2.0, mark_price=3.0, qty=1, multiplier=100) == pytest.approx(100.0)
    assert option_unrealized_pl("long", avg_price=2.0, mark_price=3.0, qty=1, multiplier=1) == pytest.approx(1.0)


class TestGetOpenOrdersSurvivesRobinStocksBugs:
    """Reproduces a real crash seen in production: robin_stocks' own
    get_all_open_stock_orders does `item['cancel']` on every entry without
    checking for None first, and Robinhood's API occasionally hands back a
    list containing a None entry (order data still settling, a purged
    cancelled order, or similar API noise) -- when it does, robin_stocks
    raises TypeError: 'NoneType' object is not subscriptable from INSIDE
    its own list comprehension, before ever returning to us. That crashed
    the whole dashboard page load. get_open_orders() must degrade to an
    empty result for the side that failed, not take the page down."""

    def test_typeerror_from_equity_orders_call_is_caught(self):
        with patch("app.data_fetch.rh.orders.get_all_open_stock_orders", side_effect=TypeError("'NoneType' object is not subscriptable")):
            with patch("app.data_fetch.rh.orders.get_all_open_option_orders", return_value=[]):
                result = get_open_orders()
        assert result.empty

    def test_typeerror_from_option_orders_call_is_caught(self):
        with patch("app.data_fetch.rh.orders.get_all_open_stock_orders", return_value=[]):
            with patch("app.data_fetch.rh.orders.get_all_open_option_orders", side_effect=TypeError("'NoneType' object is not subscriptable")):
                result = get_open_orders()
        assert result.empty

    def test_none_entries_in_a_successfully_returned_list_are_skipped_not_crashed(self):
        # A defensive second layer: even if robin_stocks itself doesn't
        # raise (a different version, or Robinhood fixes their API), a
        # None slipping through the returned list must not crash our own
        # iteration either.
        good_order = {
            "instrument": "https://api.robinhood.com/instruments/abc/", "side": "buy",
            "quantity": "10", "price": "150.00", "state": "queued", "created_at": "2026-09-16T00:00:00Z",
        }
        with patch("app.data_fetch.rh.orders.get_all_open_stock_orders", return_value=[None, good_order]):
            with patch("app.data_fetch.rh.orders.get_all_open_option_orders", return_value=[None]):
                with patch("app.data_fetch.rh.stocks.get_symbol_by_url", return_value="AAPL"):
                    result = get_open_orders()
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"

    def test_normal_data_still_flows_through_unaffected(self):
        good_equity = {
            "instrument": "https://api.robinhood.com/instruments/abc/", "side": "buy",
            "quantity": "10", "price": "150.00", "state": "queued", "created_at": "2026-09-16T00:00:00Z",
        }
        good_option = {"chain_symbol": "SPY", "direction": "debit", "quantity": "1", "price": "2.50", "state": "queued", "created_at": "2026-09-16T00:00:00Z"}
        with patch("app.data_fetch.rh.orders.get_all_open_stock_orders", return_value=[good_equity]):
            with patch("app.data_fetch.rh.orders.get_all_open_option_orders", return_value=[good_option]):
                with patch("app.data_fetch.rh.stocks.get_symbol_by_url", return_value="AAPL"):
                    result = get_open_orders()
        assert len(result) == 2
        assert set(result["instrument_type"]) == {"equity", "option"}
