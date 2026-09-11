"""Synthetic data for DEMO_MODE — a drop-in replacement for data_fetch.py with
the exact same function names and return shapes, so dashboard.py doesn't know
or care which one it's talking to.

No network calls, no Robinhood account, nothing real. Every number here is
generated deterministically from a seed derived from the symbol name (so the
same ticker always produces the same synthetic chart/price across reruns,
which matters for a demo someone might click around in more than once) plus
one shared "market factor" all symbols partially load onto, so cross-symbol
correlation in the Correlation Explorer looks like something real rather than
either 0 or 1.

Option prices/Greeks are computed with risk_tool's actual Black-Scholes engine
(not hand-picked), so the numbers you see are internally consistent — this
part of the demo is genuinely running the same math as the real thing, just
fed a synthetic spot/vol instead of a live quote.
"""
from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from risk_tool.greeks import all_greeks
from risk_tool.pricing import black_scholes_price

RISK_FREE_RATE = 0.05
DIVIDEND_YIELD = 0.0
MARKET_SEED = 20260101  # fixed — the shared factor every symbol partially loads onto


def _seed(symbol: str, salt: str = "") -> int:
    return int(hashlib.sha256(f"{symbol}:{salt}".encode()).hexdigest(), 16) % (2**32)


def get_vol_tickers() -> list[str]:
    raw = os.getenv("VOL_TICKERS", "VXX,UVXY,SVXY,VIXY,UVIX,SVIX")
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _market_factor(n_days: int) -> np.ndarray:
    rng = np.random.default_rng(MARKET_SEED)
    return rng.normal(0.0003, 0.010, n_days)


def _price_series(symbol: str, n_days: int = 260) -> pd.DataFrame:
    """Deterministic synthetic daily OHLC via a one-factor return model:
    daily_return = drift + beta * market_factor + idiosyncratic_noise."""
    rng = np.random.default_rng(_seed(symbol, "series"))
    base_price = float(rng.uniform(25, 420))
    beta = float(rng.uniform(0.2, 1.5))
    idio_vol = float(rng.uniform(0.008, 0.022))
    drift = float(rng.uniform(-0.0004, 0.0007))

    market = _market_factor(n_days)
    idio = rng.normal(0.0, idio_vol, n_days)
    daily_returns = drift + beta * market + idio

    closes = base_price * np.cumprod(1 + daily_returns)
    intraday_range = np.abs(rng.normal(0.0, idio_vol * 0.6, n_days))
    highs = closes * (1 + intraday_range)
    lows = closes * (1 - intraday_range)
    opens = np.concatenate([[base_price], closes[:-1]])

    end = datetime.now(timezone.utc).date()
    dates = pd.bdate_range(end=end, periods=n_days)

    return pd.DataFrame(
        {"date": pd.to_datetime(dates), "open": opens, "high": highs, "low": lows, "close": closes}
    )


def get_underlying_price(symbol: str) -> float:
    return float(_price_series(symbol, n_days=5)["close"].iloc[-1])


def get_stock_quote(symbol: str) -> dict:
    rng = np.random.default_rng(_seed(symbol, "quote"))
    last = get_underlying_price(symbol)
    spread = last * float(rng.uniform(0.0005, 0.002))
    bid, ask = last - spread / 2, last + spread / 2
    return {"symbol": symbol, "bid": bid, "ask": ask, "mark": (bid + ask) / 2, "last_trade_price": last}


def get_equity_historicals(symbol: str, interval: str = "day", span: str = "year") -> pd.DataFrame:
    n_days = {"month": 22, "3month": 65, "year": 260, "5year": 1260}.get(span, 260)
    return _price_series(symbol, n_days=n_days)


def get_crypto_historicals(symbol: str, interval: str = "day", span: str = "year") -> pd.DataFrame:
    return get_equity_historicals(symbol, interval=interval, span=span)


def get_crypto_quote(symbol: str) -> dict:
    return get_stock_quote(symbol)


def get_portfolio_overview() -> dict:
    equity = 128_450.32
    prev_equity = 127_209.77
    day_pl = equity - prev_equity
    return {
        "equity": equity,
        "cash": 18_220.10,
        "buying_power": 36_440.20,
        "day_pl": day_pl,
        "day_pl_pct": day_pl / prev_equity * 100,
        "market_value": 110_230.22,
    }


def get_portfolio_history(span: str = "month", interval: str = "day") -> pd.DataFrame:
    rng = np.random.default_rng(MARKET_SEED + 1)
    n_days = 30
    end_equity = 128_450.32
    daily_returns = rng.normal(0.0012, 0.006, n_days)
    factors = np.cumprod(1 + daily_returns[::-1])[::-1]
    equity_path = end_equity / factors
    dates = pd.bdate_range(end=datetime.now(timezone.utc).date(), periods=n_days)
    return pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity_path})


_EQUITY_BOOK = [
    {"symbol": "XOM", "quantity": 300.0, "cost_mult": 0.94, "is_vol_ticker": False},
    {"symbol": "AAPL", "quantity": 150.0, "cost_mult": 1.06, "is_vol_ticker": False},
    {"symbol": "VXX", "quantity": 500.0, "cost_mult": 1.18, "is_vol_ticker": True},
]


def get_equity_positions() -> pd.DataFrame:
    rows = []
    for pos in _EQUITY_BOOK:
        last_price = get_underlying_price(pos["symbol"])
        avg_cost = last_price * pos["cost_mult"]
        market_value = pos["quantity"] * last_price
        cost_basis = pos["quantity"] * avg_cost
        rows.append(
            {
                "symbol": pos["symbol"],
                "quantity": pos["quantity"],
                "avg_cost": avg_cost,
                "last_price": last_price,
                "market_value": market_value,
                "unrealized_pl": market_value - cost_basis,
                "unrealized_pl_pct": (last_price - avg_cost) / avg_cost * 100,
                "is_vol_ticker": pos["is_vol_ticker"],
            }
        )
    return pd.DataFrame(rows)


_OPTION_BOOK = [
    {"symbol": "XOM", "type": "put", "side": "long", "strike_mult": 0.98, "dte": 30, "quantity": 10.0, "sigma": 0.27},
    {"symbol": "SPY", "type": "call", "side": "long", "strike_mult": 1.02, "dte": 45, "quantity": 5.0, "sigma": 0.16},
    {"symbol": "VXX", "type": "call", "side": "short", "strike_mult": 1.15, "dte": 20, "quantity": 8.0, "sigma": 0.85},
]


def get_option_positions() -> pd.DataFrame:
    rows = []
    today = datetime.now(timezone.utc).date()
    for pos in _OPTION_BOOK:
        spot = get_underlying_price(pos["symbol"])
        strike = round(spot * pos["strike_mult"] / 0.5) * 0.5
        expiration = today + timedelta(days=pos["dte"])
        T = pos["dte"] / 365.0
        sigma = pos["sigma"]

        mark_price = black_scholes_price(spot, strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, sigma, pos["type"])
        greeks = all_greeks(spot, strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, sigma, pos["type"])
        entry_drift = 0.85 if pos["side"] == "long" else 1.15
        avg_price = mark_price * entry_drift

        qty = pos["quantity"]
        multiplier = 100
        signed_qty = qty if pos["side"] == "long" else -qty

        rows.append(
            {
                "symbol": pos["symbol"],
                "type": pos["type"],
                "side": pos["side"],
                "strike": strike,
                "expiration": expiration.isoformat(),
                "dte": pos["dte"],
                "quantity": qty,
                "avg_price": avg_price,
                "mark_price": mark_price,
                "market_value": mark_price * qty * multiplier,
                "unrealized_pl": (mark_price - avg_price) * signed_qty * multiplier,
                "delta": greeks["delta"] * signed_qty * multiplier,
                "theta": greeks["theta"] * signed_qty * multiplier,
                "vega": greeks["vega"] * signed_qty * multiplier,
                "gamma": greeks["gamma"] * signed_qty * multiplier,
                "implied_volatility": sigma,
            }
        )
    return pd.DataFrame(rows)


def get_open_orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"instrument_type": "equity", "symbol": "AAPL", "side": "buy", "quantity": 25.0, "price": get_underlying_price("AAPL") * 0.97, "state": "open", "created_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()},
            {"instrument_type": "option", "symbol": "SPY", "side": "sell", "quantity": 2.0, "price": 4.35, "state": "open", "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()},
        ]
    )


def get_order_history(days_back: int = 3650) -> pd.DataFrame:
    rng = np.random.default_rng(_seed("order_history", str(days_back)))
    symbols = ["XOM", "AAPL", "TSLA", "MSFT", "SPY"]
    rows = []
    now = datetime.now(timezone.utc)
    order_id = 1000

    for i in range(18):
        symbol = symbols[i % len(symbols)]
        contract_id = f"{symbol}-{i}"  # distinct per synthetic position, same schema as real contract_id
        is_option = bool(rng.integers(0, 2))
        days_ago_open = int(rng.uniform(5, min(days_back, 180)))
        entry_price = get_underlying_price(symbol) * float(rng.uniform(0.85, 1.15))
        qty = float(rng.integers(1, 10)) if is_option else float(rng.integers(10, 200))
        move = float(rng.normal(0.01, 0.06))
        exit_price = entry_price * (1 + move)
        multiplier = 100 if is_option else 1

        open_date = now - timedelta(days=days_ago_open)
        close_date = open_date + timedelta(days=int(rng.uniform(1, min(days_ago_open, 20) + 1)))

        rows.append({
            "date": open_date, "instrument_type": "option" if is_option else "equity", "symbol": symbol,
            "contract_id": contract_id, "side": "buy", "quantity": qty, "price": entry_price,
            "amount": -qty * entry_price * multiplier, "fees": 0.0, "order_id": str(order_id),
            **({"strategy": "opening"} if is_option else {}),
        })
        rows.append({
            "date": close_date, "instrument_type": "option" if is_option else "equity", "symbol": symbol,
            "contract_id": contract_id, "side": "sell", "quantity": qty, "price": exit_price,
            "amount": qty * exit_price * multiplier, "fees": 0.0, "order_id": str(order_id + 1),
            **({"strategy": "closing"} if is_option else {}),
        })
        order_id += 2

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def get_stock_quote_batch(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([get_stock_quote(s) for s in symbols])


def get_vol_ticker_quotes() -> pd.DataFrame:
    rows = []
    for t in get_vol_tickers():
        last = get_underlying_price(t)
        rng = np.random.default_rng(_seed(t, "prevclose"))
        prev_close = last * float(rng.uniform(0.97, 1.03))
        rows.append({"symbol": t, "last_price": last, "prev_close": prev_close, "change_pct": (last - prev_close) / prev_close * 100})
    return pd.DataFrame(rows)


def get_option_chain_expirations(symbol: str) -> list[str]:
    today = datetime.now(timezone.utc).date()
    days_to_friday = (4 - today.weekday()) % 7 or 7
    first_friday = today + timedelta(days=days_to_friday)
    return [(first_friday + timedelta(weeks=w)).isoformat() for w in range(6)]


def get_option_chain_skew(symbol: str, expiration_date: str) -> pd.DataFrame:
    spot = get_underlying_price(symbol)
    today = datetime.now(timezone.utc).date()
    exp = datetime.fromisoformat(expiration_date).date()
    dte = max((exp - today).days, 1)
    T = dte / 365.0

    rng = np.random.default_rng(_seed(symbol, f"chain:{expiration_date}"))
    atm_iv = float(rng.uniform(0.18, 0.45))
    increment = 1.0 if spot < 50 else (2.5 if spot < 150 else 5.0)
    strikes = [round((spot + k * increment) / increment) * increment for k in range(-7, 8)]

    rows = []
    for strike in strikes:
        moneyness = math.log(strike / spot)
        iv = max(atm_iv + 0.35 * moneyness**2 - 0.05 * moneyness, 0.05)  # smile + mild skew
        for option_type in ("call", "put"):
            price = black_scholes_price(spot, strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, iv, option_type)
            delta = all_greeks(spot, strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, iv, option_type)["delta"]
            spread_pct = 2.0 + 6.0 * abs(moneyness)
            spread = price * spread_pct / 100
            bid, ask = max(price - spread / 2, 0.01), price + spread / 2
            rows.append({
                "strike": strike, "type": option_type, "iv": iv, "delta": delta,
                "bid": round(bid, 2), "ask": round(ask, 2),
                "volume": int(rng.uniform(5, 800)), "open_interest": int(rng.uniform(50, 5000)),
            })

    df = pd.DataFrame(rows)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = (df["spread"] / df["mid"].replace(0, pd.NA) * 100).fillna(0.0)
    return df.sort_values(["type", "strike"]).reset_index(drop=True)
