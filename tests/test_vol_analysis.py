"""Correctness tests for the portfolio-level book_* helpers in
app.vol_analysis -- the glue that turns data_fetch's raw position
DataFrames into risk_tool.portfolio_risk's plain Exposure/stress inputs.
The netting behavior (shares + options on the same underlying combining
into ONE exposure) is the property most worth pinning down, since getting
that wrong silently double- or under-counts risk on any hedged position.
"""
import pandas as pd
import pytest

from app.vol_analysis import book_exposures, book_stress_pl, book_underlyings, otm_iv_curve, vol_surface_grid


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


def _chain_row(strike, option_type, iv):
    return {"strike": strike, "type": option_type, "iv": iv}


class TestOtmIvCurve:
    def test_stitches_puts_below_spot_and_calls_at_or_above(self):
        chain = pd.DataFrame(
            [
                _chain_row(90, "call", 0.30), _chain_row(95, "call", 0.28), _chain_row(100, "call", 0.25),
                _chain_row(90, "put", 0.35), _chain_row(95, "put", 0.32), _chain_row(100, "put", 0.26),
            ]
        )
        curve = otm_iv_curve(chain, spot=97.0)
        # below spot -> puts (90, 95); at/above spot -> calls (100)
        assert list(curve["strike"]) == [90.0, 95.0, 100.0]
        assert list(curve["iv"]) == [0.35, 0.32, 0.25]

    def test_drops_unquoted_strikes(self):
        chain = pd.DataFrame([_chain_row(90, "put", 0.0), _chain_row(95, "put", -1.0), _chain_row(100, "call", 0.25)])
        curve = otm_iv_curve(chain, spot=97.0)
        assert list(curve["strike"]) == [100.0]

    def test_empty_chain_or_no_spot(self):
        assert otm_iv_curve(pd.DataFrame(), 100.0).empty
        assert otm_iv_curve(pd.DataFrame([_chain_row(100, "call", 0.25)]), 0.0).empty


class TestVolSurfaceGrid:
    def test_fewer_than_two_usable_expirations_returns_empty(self):
        curve = otm_iv_curve(pd.DataFrame([_chain_row(100, "call", 0.25), _chain_row(90, "put", 0.30)]), 95.0)
        assert vol_surface_grid({"2026-10-16": (30, curve)}).empty
        # a curve with < 2 rows also doesn't count as usable
        single_row_curve = otm_iv_curve(pd.DataFrame([_chain_row(100, "call", 0.25)]), 95.0)
        assert vol_surface_grid(
            {"2026-10-16": (30, curve), "2026-11-20": (60, single_row_curve)}
        ).empty

    def test_two_expirations_produce_non_degenerate_interpolated_grid(self):
        near = otm_iv_curve(
            pd.DataFrame([_chain_row(90, "put", 0.35), _chain_row(95, "put", 0.32), _chain_row(100, "call", 0.25)]),
            97.0,
        )
        far = otm_iv_curve(
            pd.DataFrame([_chain_row(85, "put", 0.36), _chain_row(95, "put", 0.33), _chain_row(105, "call", 0.27)]),
            97.0,
        )
        grid = vol_surface_grid({"near": (30, near), "far": (60, far)}, n_strike_points=15)
        assert not grid.empty
        assert grid["expiration"].nunique() == 2
        assert grid["strike"].nunique() == 15  # same shared grid for both expirations
        assert grid["iv"].dropna().nunique() > 5  # real interpolated variation, not a flat/degenerate surface
        # a strike known to be inside BOTH curves' quoted range must interpolate, not be NaN
        near_row = grid[(grid["expiration"] == "near") & (grid["strike"].between(90, 100))]
        assert near_row["iv"].notna().any()

    def test_points_outside_an_expirations_own_strike_range_are_nan_not_extrapolated(self):
        near = otm_iv_curve(pd.DataFrame([_chain_row(95, "put", 0.32), _chain_row(100, "call", 0.25)]), 97.0)
        far = otm_iv_curve(pd.DataFrame([_chain_row(70, "put", 0.40), _chain_row(130, "call", 0.20)]), 97.0)
        grid = vol_surface_grid({"near": (30, near), "far": (60, far)}, n_strike_points=20)
        near_rows = grid[grid["expiration"] == "near"]
        # far curve spans 70-130, near curve only spans 95-100 -- most of the
        # shared grid falls outside near's own quoted range
        assert near_rows["iv"].isna().sum() > 0
        assert near_rows["iv"].notna().sum() > 0
