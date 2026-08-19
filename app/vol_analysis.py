"""Book-level volatility exposure analysis on top of raw position DataFrames."""
from __future__ import annotations

import pandas as pd


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
    so they can be marked on the skew/spread charts for that underlying+expiration."""
    if option_positions.empty or chain.empty:
        return option_positions.iloc[0:0]

    held = option_positions[(option_positions["symbol"] == symbol) & (option_positions["expiration"] == expiration)]
    if held.empty:
        return held

    return held.merge(chain[["strike", "type", "iv", "spread", "spread_pct", "mid"]], on=["strike", "type"], how="left")
