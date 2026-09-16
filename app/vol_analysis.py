"""Book-level volatility exposure analysis on top of raw position DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_tool.option_strategy import OptionLeg
from risk_tool.portfolio_risk import Exposure, equity_stress_pl, option_leg_stress_pl


def aggregate_greeks(option_positions: pd.DataFrame) -> dict:
    if option_positions.empty:
        return {"net_delta": 0.0, "net_theta": 0.0, "net_vega": 0.0, "net_gamma": 0.0}
    return {
        "net_delta": float(option_positions["delta"].sum()),
        "net_theta": float(option_positions["theta"].sum()),
        "net_vega": float(option_positions["vega"].sum()),
        "net_gamma": float(option_positions["gamma"].sum()),
    }


def vol_etp_exposure(equity_positions: pd.DataFrame) -> pd.DataFrame:
    """Directional (non-options) exposure to VIX-linked ETPs specifically."""
    if equity_positions.empty:
        return equity_positions
    return equity_positions[equity_positions["is_vol_ticker"]].copy()


def premium_by_underlying(option_positions: pd.DataFrame) -> pd.DataFrame:
    """Net premium (market value) and vega grouped by underlying — which
    names are actually driving the book's vol exposure."""
    if option_positions.empty:
        return option_positions
    grouped = (
        option_positions.groupby("symbol")
        .agg(
            contracts=("quantity", "sum"),
            market_value=("market_value", "sum"),
            unrealized_pl=("unrealized_pl", "sum"),
            net_delta=("delta", "sum"),
            net_theta=("theta", "sum"),
            net_vega=("vega", "sum"),
            avg_iv=("implied_volatility", "mean"),
        )
        .reset_index()
        .sort_values("net_vega", key=abs, ascending=False)
    )
    return grouped


def near_expiry(option_positions: pd.DataFrame, within_days: int = 7) -> pd.DataFrame:
    if option_positions.empty:
        return option_positions
    return option_positions[option_positions["dte"].fillna(999) <= within_days].sort_values("dte")


def skew_metrics(chain: pd.DataFrame, spot: float) -> dict:
    """ATM IV and 25-delta risk reversal (call IV minus put IV) — the two
    standard single-number summaries of a skew curve's level and slant."""
    if chain.empty or not spot:
        return {"atm_iv": None, "risk_reversal_25d": None}

    calls = chain[chain["type"] == "call"]
    puts = chain[chain["type"] == "put"]

    atm_ivs = []
    if not calls.empty:
        atm_ivs.append(calls.loc[(calls["strike"] - spot).abs().idxmin(), "iv"])
    if not puts.empty:
        atm_ivs.append(puts.loc[(puts["strike"] - spot).abs().idxmin(), "iv"])
    atm_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else None

    risk_reversal = None
    if not calls.empty and not puts.empty:
        call_25d = calls.loc[(calls["delta"] - 0.25).abs().idxmin()]
        put_25d = puts.loc[(puts["delta"] + 0.25).abs().idxmin()]
        risk_reversal = float(call_25d["iv"] - put_25d["iv"])

    return {"atm_iv": atm_iv, "risk_reversal_25d": risk_reversal}


def otm_iv_curve(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Per-strike IV using the standard OTM-stitched convention: put IV
    below spot, call IV at/above spot. OTM contracts are more liquid and
    don't carry the early-exercise / deep-ITM quoting noise that a naive
    single-side curve would, so this is the usual way to reduce a chain's
    two curves (calls, puts) to the one "the market's IV at this strike"
    curve a vol surface needs. Drops unquoted (iv<=0 or missing) strikes
    rather than fabricating a value. Returns columns strike, iv, sorted
    and deduped by strike."""
    if chain.empty or not spot:
        return pd.DataFrame(columns=["strike", "iv"])
    calls = chain[(chain["type"] == "call") & (chain["strike"] >= spot)][["strike", "iv"]]
    puts = chain[(chain["type"] == "put") & (chain["strike"] < spot)][["strike", "iv"]]
    combined = pd.concat([calls, puts], ignore_index=True).dropna(subset=["iv"])
    combined = combined[combined["iv"] > 0]
    return combined.sort_values("strike").drop_duplicates(subset="strike").reset_index(drop=True)


def vol_surface_grid(curves: dict, n_strike_points: int = 30) -> pd.DataFrame:
    """Stitch several expirations' otm_iv_curve() results into one
    strike x expiration x IV long-form grid for a go.Surface.

    curves: {expiration_label: (dte, otm_iv_curve_df)}, each df needing
    columns strike/iv with >=2 rows -- entries not meeting that are
    skipped (a single-strike curve can't be interpolated).

    Every curve is linearly interpolated onto ONE shared strike grid
    spanning the union of all curves' strikes, so every expiration lines
    up on the same x-axis (real chains rarely share identical strikes
    expiration to expiration). Points outside a given expiration's own
    quoted strike range are left NaN rather than extrapolated -- past the
    edge of what that expiration actually quoted, plotly draws a gap
    instead of us guessing a number.

    Returns columns: expiration, dte, strike, iv (iv may be NaN).
    Empty (0 rows) if fewer than 2 usable expirations are supplied.
    """
    usable = {exp: (dte, curve) for exp, (dte, curve) in curves.items() if len(curve) >= 2}
    if len(usable) < 2:
        return pd.DataFrame(columns=["expiration", "dte", "strike", "iv"])

    lo = min(curve["strike"].min() for _, curve in usable.values())
    hi = max(curve["strike"].max() for _, curve in usable.values())
    if lo >= hi:
        return pd.DataFrame(columns=["expiration", "dte", "strike", "iv"])
    strike_grid = np.linspace(lo, hi, n_strike_points)

    rows = []
    for exp, (dte, curve) in usable.items():
        curve_strikes = curve["strike"].to_numpy()
        curve_ivs = curve["iv"].to_numpy()
        s_lo, s_hi = curve_strikes.min(), curve_strikes.max()
        interp_iv = np.interp(strike_grid, curve_strikes, curve_ivs, left=np.nan, right=np.nan)
        interp_iv = np.where((strike_grid < s_lo) | (strike_grid > s_hi), np.nan, interp_iv)
        for strike, iv in zip(strike_grid, interp_iv):
            rows.append({"expiration": exp, "dte": dte, "strike": float(strike), "iv": float(iv) if pd.notna(iv) else None})
    return pd.DataFrame(rows)


def cumulative_return(close: pd.Series) -> pd.Series:
    """Indexed % return from the first observation — lets two differently-priced
    series be overlaid on one chart on a common scale."""
    if close.empty:
        return close
    return (close / close.iloc[0] - 1.0) * 100.0


def correlation_stats(hist_a: pd.DataFrame, hist_b: pd.DataFrame) -> dict:
    """Price-level and daily-return correlation between two get_equity_historicals()
    frames (each needs 'date' and 'close'), aligned on date.

    Price-level correlation is usually inflated by any shared long-run trend —
    two names that both drifted up over the window will look correlated even if
    their day-to-day moves are unrelated. Return correlation strips the trend
    out and is the more honest co-movement signal."""
    merged = pd.merge(hist_a[["date", "close"]], hist_b[["date", "close"]], on="date", suffixes=("_a", "_b"))
    if len(merged) < 3:
        return {"price_corr": None, "return_corr": None, "n_obs": len(merged)}

    price_corr = merged["close_a"].corr(merged["close_b"])
    ret_a = merged["close_a"].pct_change().dropna()
    ret_b = merged["close_b"].pct_change().dropna()
    return_corr = ret_a.corr(ret_b)

    return {
        "price_corr": float(price_corr) if pd.notna(price_corr) else None,
        "return_corr": float(return_corr) if pd.notna(return_corr) else None,
        "n_obs": len(merged),
    }


def held_contracts_in_chain(
    option_positions: pd.DataFrame, chain: pd.DataFrame, symbol: str, expiration: str
) -> pd.DataFrame:
    """Your open option positions that fall within the given chain lookup,
    so they can be marked on the skew/spread/open-interest charts for that
    underlying+expiration."""
    if option_positions.empty or chain.empty:
        return option_positions.iloc[0:0]

    held = option_positions[(option_positions["symbol"] == symbol) & (option_positions["expiration"] == expiration)]
    if held.empty:
        return held

    return held.merge(
        chain[["strike", "type", "iv", "spread", "spread_pct", "mid", "open_interest", "volume"]],
        on=["strike", "type"], how="left",
    )


def book_underlyings(equity_positions: pd.DataFrame, option_positions: pd.DataFrame) -> list[str]:
    """Every distinct underlying symbol held, either as shares or as an
    option -- the set of names a portfolio-level risk view (correlation,
    VaR, stress test) needs price history for."""
    symbols: set[str] = set()
    if not equity_positions.empty:
        symbols |= set(equity_positions["symbol"].dropna().unique())
    if not option_positions.empty:
        symbols |= set(option_positions["symbol"].dropna().unique())
    return sorted(symbols)


def book_exposures(
    equity_positions: pd.DataFrame, option_positions: pd.DataFrame, spot_by_symbol: dict
) -> list[Exposure]:
    """Net dollar-delta Exposure per underlying, for risk_tool.portfolio_risk's
    VaR functions.

    Shares and options on the SAME underlying are netted into one Exposure
    before returning -- e.g. long 100 shares of XOM plus short calls whose
    delta offsets 40 of those shares nets to a 60-share-equivalent dollar
    exposure, not two separate 100- and -40-share bets. Getting this netting
    right is the entire point of a portfolio (rather than per-position) risk
    view: it's what lets a covered call correctly show up as LESS risky than
    the same amount of naked stock.

    Symbols with no entry in spot_by_symbol are silently skipped (no price
    to convert shares-equivalent into a dollar figure) rather than raising,
    so one bad/delisted quote doesn't take down the whole portfolio view.
    """
    totals: dict[str, float] = {}

    if not equity_positions.empty:
        for _, row in equity_positions.iterrows():
            spot = spot_by_symbol.get(row["symbol"])
            if not spot:
                continue
            totals[row["symbol"]] = totals.get(row["symbol"], 0.0) + float(row["quantity"]) * spot

    if not option_positions.empty:
        net_delta_by_symbol = option_positions.groupby("symbol")["delta"].sum()  # already signed, multiplier-adjusted shares-equivalent
        for symbol, net_delta_shares in net_delta_by_symbol.items():
            spot = spot_by_symbol.get(symbol)
            if not spot:
                continue
            totals[symbol] = totals.get(symbol, 0.0) + float(net_delta_shares) * spot

    return [Exposure(symbol=s, dollar_delta=v) for s, v in totals.items() if abs(v) > 1e-9]


def book_stress_pl(
    equity_positions: pd.DataFrame,
    option_positions: pd.DataFrame,
    spot_by_symbol: dict,
    shock_pct: float,
    iv_shock_pts: float = 0.0,
    r: float = 0.05,
    q: float = 0.0,
) -> dict:
    """Full-repricing P&L of the whole book under one scenario: every
    underlying's spot moves by shock_pct simultaneously (a market-wide
    move, the standard stress-test scenario -- not a per-symbol shock),
    and every option leg is repriced exactly via Black-Scholes rather than
    approximated from today's Greeks (see
    risk_tool.portfolio_risk.option_leg_stress_pl for why that matters for
    a large move). Equity legs are exact by construction (linear payoff).

    iv_shock_pts applies to every option leg's IV uniformly -- a real
    selloff doesn't crush/expand every name's vol by the same number of
    points, but a single shared shock keeps the scenario legible instead
    of requiring a per-symbol vol view for a portfolio-wide stress test.

    Positions on an underlying with no entry in spot_by_symbol, or an
    option with a missing/non-positive DTE or IV, are skipped and counted
    in `skipped` rather than silently dropped from the total unexplained.
    """
    equity_pl = 0.0
    option_pl = 0.0
    skipped = 0

    if not equity_positions.empty:
        for _, row in equity_positions.iterrows():
            spot = spot_by_symbol.get(row["symbol"])
            if not spot:
                skipped += 1
                continue
            equity_pl += equity_stress_pl(float(row["quantity"]), spot, shock_pct)

    if not option_positions.empty:
        for _, row in option_positions.iterrows():
            spot = spot_by_symbol.get(row["symbol"])
            iv = row.get("implied_volatility")
            dte = row.get("dte")
            if not spot or iv is None or pd.isna(iv) or iv <= 0 or dte is None or pd.isna(dte):
                skipped += 1
                continue
            contracts = float(row["quantity"]) if row["side"] == "long" else -float(row["quantity"])
            leg = OptionLeg(option_type=row["type"], strike=float(row["strike"]), premium=float(row["avg_price"]), contracts=contracts, iv=float(iv))
            T_years = max(float(dte), 0.0) / 365.0
            option_pl += option_leg_stress_pl(leg, spot, T_years, shock_pct, iv_shock_pts=iv_shock_pts, r=r, q=q)

    return {"equity_pl": equity_pl, "option_pl": option_pl, "total_pl": equity_pl + option_pl, "skipped": skipped}
