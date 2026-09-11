"""Correctness tests for the portfolio-level book_* helpers in
app.vol_analysis -- the glue that turns data_fetch's raw position
DataFrames into risk_tool.portfolio_risk's plain Exposure/stress inputs.
The netting behavior (shares + options on the same underlying combining
into ONE exposure) is the property most worth pinning down, since getting
that wrong silently double- or under-counts risk on any hedged position.
"""
import pandas as pd
import pytest

from app.vol_analysis import book_exposures, book_stress_pl, book_underlyings


def equity_row(symbol, quantity):
    return {"symbol": symbol, "quantity": quantity}


def option_row(symbol, option_type, side, strike, dte, quantity, avg_price, iv, delta):
    return {
        "symbol": symbol, "type": option_type, "side": side, "strike": strike, "dte": dte,
        "quantity": quantity, "avg_price": avg_price, "implied_volatility": iv, "delta": delta,
    }


class TestBookUnderlyings:
    def test_union_of_equity_and_option_symbols(self):
        eq = pd.DataFrame([equity_row("AAPL", 100.0)])
        opt = pd.DataFrame([option_row("XOM", "put", "long", 100.0, 30, 1.0, 2.0, 0.3, -30.0)])
        assert book_underlyings(eq, opt) == ["AAPL", "XOM"]

    def test_dedupes_symbol_held_both_ways(self):
        eq = pd.DataFrame([equity_row("AAPL", 100.0)])
        opt = pd.DataFrame([option_row("AAPL", "call", "short", 110.0, 20, 2.0, 3.0, 0.25, -40.0)])
        assert book_underlyings(eq, opt) == ["AAPL"]

    def test_both_empty_returns_empty_list(self):
        empty = pd.DataFrame()
        assert book_underlyings(empty, empty) == []


class TestBookExposures:
    def test_equity_only_dollar_delta_is_shares_times_spot(self):
        eq = pd.DataFrame([equity_row("AAPL", 100.0)])
        opt = pd.DataFrame()
        exposures = book_exposures(eq, opt, {"AAPL": 150.0})
        assert len(exposures) == 1
        assert exposures[0].symbol == "AAPL"
        assert exposures[0].dollar_delta == pytest.approx(15_000.0)

    def test_option_only_dollar_delta_is_delta_shares_times_spot(self):
        eq = pd.DataFrame()
        opt = pd.DataFrame([option_row("SPY", "call", "long", 450.0, 30, 5.0, 4.0, 0.16, 250.0)])
        exposures = book_exposures(eq, opt, {"SPY": 440.0})
        assert exposures[0].dollar_delta == pytest.approx(250.0 * 440.0)

    def test_shares_and_options_on_same_underlying_net_together(self):
        # Long 100 shares of XOM (dollar delta = 100*spot), short a call whose
        # net delta is already -40 shares-equivalent -- must net to 60 shares
        # of dollar exposure, ONE Exposure entry, not two independent ones.
        eq = pd.DataFrame([equity_row("XOM", 100.0)])
        opt = pd.DataFrame([option_row("XOM", "call", "short", 115.0, 25, 1.0, 1.5, 0.28, -40.0)])
        exposures = book_exposures(eq, opt, {"XOM": 110.0})
        assert len(exposures) == 1
        assert exposures[0].dollar_delta == pytest.approx((100.0 - 40.0) * 110.0)

    def test_symbol_missing_a_spot_price_is_skipped(self):
        eq = pd.DataFrame([equity_row("AAPL", 100.0), equity_row("ZZZ", 10.0)])
        exposures = book_exposures(eq, pd.DataFrame(), {"AAPL": 150.0})  # no ZZZ spot
        assert [e.symbol for e in exposures] == ["AAPL"]

    def test_near_zero_net_exposure_is_dropped(self):
        # Shares and an equal-and-opposite option delta cancel to ~0 -- no
        # meaningful risk left on this name, shouldn't show up in the list.
        eq = pd.DataFrame([equity_row("XOM", 40.0)])
        opt = pd.DataFrame([option_row("XOM", "call", "short", 115.0, 25, 1.0, 1.5, 0.28, -40.0)])
        exposures = book_exposures(eq, opt, {"XOM": 110.0})
        assert exposures == []


class TestBookStressPl:
    def test_equity_only_matches_hand_calc(self):
        eq = pd.DataFrame([equity_row("AAPL", 100.0)])
        result = book_stress_pl(eq, pd.DataFrame(), {"AAPL": 150.0}, shock_pct=-0.10)
        assert result["equity_pl"] == pytest.approx(100.0 * 150.0 * -0.10)
        assert result["option_pl"] == 0.0
        assert result["total_pl"] == pytest.approx(result["equity_pl"])
        assert result["skipped"] == 0

    def test_option_leg_matches_direct_black_scholes_repricing(self):
        from risk_tool.pricing import black_scholes_price

        opt = pd.DataFrame([option_row("XOM", "put", "long", 100.0, 30, 1.0, 3.0, 0.30, -20.0)])
        result = book_stress_pl(pd.DataFrame(), opt, {"XOM": 100.0}, shock_pct=-0.05, iv_shock_pts=0.05, r=0.03, q=0.0)

        T = 30 / 365.0
        current = black_scholes_price(100.0, 100.0, T, 0.03, 0.0, 0.30, "put")
        shocked = black_scholes_price(95.0, 100.0, T, 0.03, 0.0, 0.35, "put")
        expected = (shocked - current) * 1.0 * 100.0  # long 1 contract
        assert result["option_pl"] == pytest.approx(expected)

    def test_missing_spot_is_skipped_and_counted(self):
        eq = pd.DataFrame([equity_row("ZZZ", 10.0)])
        result = book_stress_pl(eq, pd.DataFrame(), {}, shock_pct=-0.10)
        assert result["total_pl"] == 0.0
        assert result["skipped"] == 1

    def test_option_missing_iv_or_dte_is_skipped_and_counted(self):
        opt = pd.DataFrame([option_row("XOM", "put", "long", 100.0, None, 1.0, 3.0, None, -20.0)])
        result = book_stress_pl(pd.DataFrame(), opt, {"XOM": 100.0}, shock_pct=-0.10)
        assert result["option_pl"] == 0.0
        assert result["skipped"] == 1

    def test_zero_shock_is_zero_total_pl(self):
        eq = pd.DataFrame([equity_row("AAPL", 50.0)])
        opt = pd.DataFrame([option_row("AAPL", "call", "long", 150.0, 20, 2.0, 4.0, 0.30, 60.0)])
        result = book_stress_pl(eq, opt, {"AAPL": 150.0}, shock_pct=0.0, iv_shock_pts=0.0)
        assert result["total_pl"] == pytest.approx(0.0, abs=1e-9)

    def test_short_option_leg_gains_when_underlying_falls(self):
        opt = pd.DataFrame([option_row("SPY", "call", "short", 460.0, 15, 3.0, 2.0, 0.15, -90.0)])
        result = book_stress_pl(pd.DataFrame(), opt, {"SPY": 450.0}, shock_pct=-0.05)
        assert result["option_pl"] > 0  # short call, underlying drops -> position gains
