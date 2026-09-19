"""Correctness tests for app.demo_data.get_earnings_dates — the DEMO_MODE
stand-in for data_fetch.get_earnings_dates, used so the Risk Tool's
earnings-window warning still has something to check against with
DEMO_MODE=true and no live Robinhood session."""
from datetime import date, timedelta

from app.demo_data import get_earnings_dates


def test_returns_sorted_dates_including_one_within_120_days():
    today = date.today()
    dates = get_earnings_dates("AAPL")
    assert dates == sorted(dates)
    # The deterministic "next" earnings date is always within [today, today+119].
    future_dates = [d for d in dates if d >= today]
    assert len(future_dates) >= 1
    assert (future_dates[0] - today).days <= 119


def test_is_deterministic_for_the_same_symbol():
    assert get_earnings_dates("AAPL") == get_earnings_dates("AAPL")


def test_different_symbols_generally_get_different_schedules():
    """Not a mathematical guarantee (a hash collision is technically
    possible), but with a handful of distinct real tickers the schedules
    should not all coincide -- this catches an accidental symbol-independent
    implementation (e.g. forgetting to seed on `symbol`)."""
    schedules = {t: tuple(get_earnings_dates(t)) for t in ["AAPL", "MSFT", "TSLA", "SPY", "NVDA"]}
    assert len(set(schedules.values())) > 1


def test_includes_prior_quarters_roughly_91_days_apart():
    dates = get_earnings_dates("AAPL")
    assert len(dates) == 4
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    for gap in gaps:
        assert abs(gap - 91) <= 1  # generated with exact 91-day spacing; tolerate off-by-one from sorting edge cases
