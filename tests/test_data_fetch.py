"""Correctness tests for the pure helpers in app.data_fetch — no robin_stocks
mocking needed since these don't touch the network."""
import pytest

from app.data_fetch import option_unrealized_pl


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
