"""All Robinhood data pulls, isolated from Streamlit/UI concerns.

Every function returns plain pandas DataFrames (or dicts for single-value
summaries) so app.py stays presentation-only and this module is unit
testable without a live session.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import robin_stocks.robinhood as rh


def get_vol_tickers() -> list[str]:
    raw = os.getenv("VOL_TICKERS", "VXX,UVXY,SVXY,VIXY,UVIX,SVIX")
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def get_portfolio_overview() -> dict:
    profile = rh.profiles.load_portfolio_profile() or {}
    account = rh.profiles.load_account_profile() or {}

    equity = float(profile.get("equity") or 0)
    prev_equity = float(profile.get("equity_previous_close") or equity)
    day_pl = equity - prev_equity
    day_pl_pct = (day_pl / prev_equity * 100) if prev_equity else 0.0

    return {
        "equity": equity,
        "cash": float(account.get("cash") or profile.get("withdrawable_amount") or 0),
        "buying_power": float(account.get("buying_power") or 0),
        "day_pl": day_pl,
        "day_pl_pct": day_pl_pct,
        "market_value": float(profile.get("market_value") or 0),
    }


def get_portfolio_history(span: str = "month", interval: str = "day") -> pd.DataFrame:
    hist = rh.account.get_historical_portfolio(interval=interval, span=span) or {}
    equities = hist.get("equity_historicals", [])
    if not equities:
        return pd.DataFrame(columns=["date", "equity"])
    df = pd.DataFrame(equities)
    df["date"] = pd.to_datetime(df["begins_at"])
    df["equity"] = df["adjusted_close_equity"].astype(float)
    return df[["date", "equity"]]


def get_equity_positions() -> pd.DataFrame:
    positions = rh.account.get_open_stock_positions() or []
    rows = []
    for pos in positions:
        qty = float(pos.get("quantity") or 0)
        if qty == 0:
            continue
        instrument_url = pos.get("instrument")
        symbol = rh.stocks.get_symbol_by_url(instrument_url) if instrument_url else pos.get("symbol")
        quote = rh.stocks.get_latest_price(symbol)[0] if symbol else None
        last_price = float(quote) if quote else 0.0
        avg_cost = float(pos.get("average_buy_price") or 0)
        market_value = qty * last_price
        cost_basis = qty * avg_cost
        rows.append(
            {
                "symbol": symbol,
                "quantity": qty,
                "avg_cost": avg_cost,
                "last_price": last_price,
                "market_value": market_value,
                "unrealized_pl": market_value - cost_basis,
                "unrealized_pl_pct": ((last_price - avg_cost) / avg_cost * 100) if avg_cost else 0.0,
                "is_vol_ticker": symbol in get_vol_tickers(),
            }
        )
    return pd.DataFrame(rows)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def option_unrealized_pl(side: str, avg_price: float, mark_price: float, qty: float, multiplier: int = 100) -> float:
    """Unrealized P&L for one option leg.

    Robinhood reports `average_price` already sign-flipped for short
    positions (negative — representing a credit received), which is NOT the
    same convention as `mark_price` (always a plain positive market quote).
    Naively doing (mark_price - avg_price) * signed_qty double-applies that
    sign for shorts, producing a phantom loss roughly 2x the true premium
    regardless of the actual current price. Unsign avg_price explicitly and
    branch on side instead — the only correct way to combine a
    sign-carrying field with a sign-free one."""
    avg_price_magnitude = abs(avg_price)
    if side == "long":
        return (mark_price - avg_price_magnitude) * qty * multiplier
    return (avg_price_magnitude - mark_price) * qty * multiplier


def get_option_positions() -> pd.DataFrame:
    positions = rh.options.get_open_option_positions() or []
    rows = []
    for pos in positions:
        qty = _safe_float(pos.get("quantity"))
        if qty == 0:
            continue

        instrument_data = rh.options.get_option_instrument_data_by_id(pos["option_id"]) or {}
        symbol = pos.get("chain_symbol")
        strike = _safe_float(instrument_data.get("strike_price"))
        exp_date = instrument_data.get("expiration_date")
        opt_type = instrument_data.get("type")  # call / put
        side = pos.get("type")  # long / short

        market_data_list = rh.options.get_option_market_data_by_id(pos["option_id"]) or []
        market_data = market_data_list[0] if market_data_list else {}

        mark_price = _safe_float(market_data.get("mark_price"))
        avg_price = _safe_float(pos.get("average_price")) / 100  # RH stores per-contract *100
        multiplier = 100
        signed_qty = qty if side == "long" else -qty

        dte = None
        if exp_date:
            try:
                dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - datetime.now(timezone.utc).date()).days
            except ValueError:
                dte = None

        rows.append(
            {
                "symbol": symbol,
                "type": opt_type,
                "side": side,
                "strike": strike,
                "expiration": exp_date,
                "dte": dte,
                "quantity": qty,
                "avg_price": avg_price,
                "mark_price": mark_price,
                "market_value": mark_price * qty * multiplier,
                "unrealized_pl": option_unrealized_pl(side, avg_price, mark_price, qty, multiplier),
                "delta": _safe_float(market_data.get("delta")) * signed_qty * multiplier,
                "theta": _safe_float(market_data.get("theta")) * signed_qty * multiplier,
                "vega": _safe_float(market_data.get("vega")) * signed_qty * multiplier,
                "gamma": _safe_float(market_data.get("gamma")) * signed_qty * multiplier,
                "implied_volatility": _safe_float(market_data.get("implied_volatility")),
            }
        )
    return pd.DataFrame(rows)


def get_open_orders() -> pd.DataFrame:
    # robin_stocks' own get_all_open_stock_orders has a bug: it does
    # `item['cancel']` on every entry Robinhood returns without checking
    # for None first, and Robinhood's API occasionally includes a None
    # entry in this list (order data still settling, a cancelled-and-
    # purged order, or similar transient API noise) -- when it does,
    # robin_stocks crashes with "TypeError: 'NoneType' object is not
    # subscriptable" before ever returning to us. That's unfixable from
    # our side except by catching it here rather than letting one bad
    # entry take down the whole dashboard page load.
    try:
        equity_orders = rh.orders.get_all_open_stock_orders() or []
    except TypeError:
        equity_orders = []
    try:
        option_orders = rh.orders.get_all_open_option_orders() or []
    except TypeError:
        option_orders = []

    rows = []
    for o in equity_orders:
        if not o:
            continue
        symbol = rh.stocks.get_symbol_by_url(o.get("instrument")) if o.get("instrument") else None
        rows.append(
            {
                "instrument_type": "equity",
                "symbol": symbol,
                "side": o.get("side"),
                "quantity": _safe_float(o.get("quantity")),
                "price": _safe_float(o.get("price")),
                "state": o.get("state"),
                "created_at": o.get("created_at"),
            }
        )
    for o in option_orders:
        if not o:
            continue
        rows.append(
            {
                "instrument_type": "option",
                "symbol": o.get("chain_symbol"),
                "side": o.get("direction"),
                "quantity": _safe_float(o.get("quantity")),
                "price": _safe_float(o.get("price")),
                "state": o.get("state"),
                "created_at": o.get("created_at"),
            }
        )
    return pd.DataFrame(rows)


def get_order_history(days_back: int = 3650) -> pd.DataFrame:
    """Normalized fill history across equities + options, for the journal.

    days_back defaults to ~10 years (effectively "all time" for any real
    account) rather than a short recent window -- Robinhood's underlying
    get_all_stock_orders/get_all_option_orders calls already return your
    full history unbounded; this filter only trims what's shown, so there's
    no reason to default it short and hide older round trips from win-rate
    stats."""
    equity_orders = rh.orders.get_all_stock_orders() or []
    option_orders = rh.orders.get_all_option_orders() or []

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
    rows = []

    for o in equity_orders:
        if o.get("state") != "filled":
            continue
        created = pd.to_datetime(o.get("created_at"), utc=True, errors="coerce")
        if pd.isna(created) or created < cutoff:
            continue
        symbol = rh.stocks.get_symbol_by_url(o.get("instrument")) if o.get("instrument") else None
        qty = _safe_float(o.get("cumulative_quantity") or o.get("quantity"))
        avg_price = _safe_float(o.get("average_price"))
        rows.append(
            {
                "date": created,
                "instrument_type": "equity",
                "symbol": symbol,
                "contract_id": symbol,  # every equity fill on a symbol is the same instrument
                "side": o.get("side"),
                "quantity": qty,
                "price": avg_price,
                "amount": qty * avg_price * (1 if o.get("side") == "sell" else -1),
                "fees": _safe_float(o.get("fees") or 0),
                "order_id": o.get("id"),
            }
        )

    for o in option_orders:
        if o.get("state") != "filled":
            continue
        created = pd.to_datetime(o.get("created_at"), utc=True, errors="coerce")
        if pd.isna(created) or created < cutoff:
            continue
        qty = _safe_float(o.get("processed_quantity") or o.get("quantity"))
        # `price` is per-share premium (e.g. 2.18); `processed_premium` is the
        # TOTAL dollar amount for the whole fill (price * qty * 100) -- using
        # it as a per-share price here inflated realized P&L by ~qty*100x.
        # Only fall back to processed_premium (normalized back to per-share)
        # when `price` itself is missing.
        avg_price = _safe_float(o.get("price"))
        if not avg_price and qty:
            avg_price = _safe_float(o.get("processed_premium")) / (qty * 100)
        legs = o.get("legs") or [{}]
        opening = (o.get("opening_strategy") or "").strip() != ""
        # legs[0]['option'] is a per-contract instrument URL -- unique per
        # (underlying, strike, expiration, type), unlike chain_symbol which
        # is shared by every contract on the same underlying. Without this,
        # match_round_trips FIFO-matches fills across DIFFERENT contracts
        # whenever you hold more than one on the same underlying at once.
        contract_id = (legs[0].get("option") if legs else None) or o.get("chain_symbol")
        rows.append(
            {
                "date": created,
                "instrument_type": "option",
                "symbol": o.get("chain_symbol"),
                "contract_id": contract_id,
                "side": legs[0].get("side") if legs else o.get("direction"),
                "quantity": qty,
                "price": avg_price,
                "amount": qty * avg_price * 100 * (1 if (legs[0].get("side") if legs else None) == "sell" else -1),
                "fees": 0.0,
                "order_id": o.get("id"),
                "strategy": "opening" if opening else "closing",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def get_underlying_price(symbol: str) -> float:
    price = rh.stocks.get_latest_price(symbol)
    return _safe_float(price[0]) if price else 0.0


def get_stock_quote(symbol: str) -> dict:
    """Best bid/ask for the underlying plus a mark price computed as their
    midpoint — Robinhood's stock quote endpoint has no mark field of its own
    (unlike options), so we derive it the same way brokers do."""
    quotes = rh.stocks.get_quotes(symbol) or []
    q = quotes[0] if quotes else {}

    bid = _safe_float(q.get("bid_price"))
    ask = _safe_float(q.get("ask_price"))
    last = _safe_float(q.get("last_trade_price"))
    mark = (bid + ask) / 2 if bid and ask else last

    return {"symbol": symbol, "bid": bid, "ask": ask, "mark": mark, "last_trade_price": last}


def get_option_chain_expirations(symbol: str) -> list[str]:
    chain = rh.options.get_chains(symbol) or {}
    return sorted(chain.get("expiration_dates") or [])


def get_option_chain_skew(symbol: str, expiration_date: str) -> pd.DataFrame:
    """Full call/put chain for one expiration, with IV/greeks per strike.

    Zero-IV rows (no live quote) are dropped — they're dead strikes, not
    real skew data, and would just show up as noise pinned to the x-axis.

    A single expiration can have 300+ contracts, and Robinhood's API needs
    one HTTP round-trip per contract for market data — done sequentially
    that's 30-40+ seconds. These calls are independent and I/O-bound (each
    thread is just waiting on the network), so a thread pool cuts this to a
    few seconds without changing any of the per-contract logic below.
    """
    contracts = rh.options.find_tradable_options(symbol, expirationDate=expiration_date, optionType=None) or []
    contract_ids = [c["id"] for c in contracts if c.get("id")]

    def fetch_market_data(contract_id: str) -> dict:
        market_data_list = rh.options.get_option_market_data_by_id(contract_id) or []
        return market_data_list[0] if market_data_list else {}

    with ThreadPoolExecutor(max_workers=20) as executor:
        market_data_by_id = dict(zip(contract_ids, executor.map(fetch_market_data, contract_ids)))

    rows = []
    for contract in contracts:
        contract_id = contract.get("id")
        if not contract_id:
            continue
        md = market_data_by_id.get(contract_id, {})
        iv = _safe_float(md.get("implied_volatility"))
        if iv <= 0:
            continue
        rows.append(
            {
                "strike": _safe_float(contract.get("strike_price")),
                "type": contract.get("type"),
                "iv": iv,
                "delta": _safe_float(md.get("delta")),
                "bid": _safe_float(md.get("bid_price")),
                "ask": _safe_float(md.get("ask_price")),
                "volume": _safe_float(md.get("volume")),
                "open_interest": _safe_float(md.get("open_interest")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = (df["spread"] / df["mid"].replace(0, pd.NA) * 100).fillna(0.0)

    return df.sort_values(["type", "strike"]).reset_index(drop=True)


def get_equity_historicals(symbol: str, interval: str = "day", span: str = "year") -> pd.DataFrame:
    """Daily OHLC history, used to feed risk_tool.realized_vol (close-to-close,
    Parkinson, Garman-Klass, GARCH all need real price history, not just a quote)."""
    bars = rh.stocks.get_stock_historicals(symbol, interval=interval, span=span, bounds="regular") or []
    rows = [
        {
            "date": bar.get("begins_at"),
            "open": _safe_float(bar.get("open_price")),
            "high": _safe_float(bar.get("high_price")),
            "low": _safe_float(bar.get("low_price")),
            "close": _safe_float(bar.get("close_price")),
        }
        for bar in bars
        if bar  # an unrecognized symbol returns [None] rather than [] or raising
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_crypto_historicals(symbol: str, interval: str = "day", span: str = "year") -> pd.DataFrame:
    """Same shape as get_equity_historicals, for crypto hedge instruments
    (e.g. XLM) that aren't on the equities historicals endpoint. Crypto
    trades 24/7 so bounds is fixed at '24_7' rather than 'regular'."""
    bars = rh.crypto.get_crypto_historicals(symbol, interval=interval, span=span, bounds="24_7") or []
    rows = [
        {
            "date": bar.get("begins_at"),
            "open": _safe_float(bar.get("open_price")),
            "high": _safe_float(bar.get("high_price")),
            "low": _safe_float(bar.get("low_price")),
            "close": _safe_float(bar.get("close_price")),
        }
        for bar in bars
        if bar
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_crypto_quote(symbol: str) -> dict:
    q = rh.crypto.get_crypto_quote(symbol) or {}
    bid = _safe_float(q.get("bid_price"))
    ask = _safe_float(q.get("ask_price"))
    mark = _safe_float(q.get("mark_price")) or ((bid + ask) / 2 if bid and ask else 0.0)
    return {"symbol": q.get("symbol", symbol), "bid": bid, "ask": ask, "mark": mark, "last_trade_price": mark}


def get_vol_ticker_quotes() -> pd.DataFrame:
    tickers = get_vol_tickers()
    if not tickers:
        return pd.DataFrame()
    quotes = rh.stocks.get_quotes(tickers) or []
    rows = []
    for t, q in zip(tickers, quotes):
        if not q:
            continue
        last = _safe_float(q.get("last_trade_price"))
        prev_close = _safe_float(q.get("previous_close"))
        rows.append(
            {
                "symbol": t,
                "last_price": last,
                "prev_close": prev_close,
                "change_pct": ((last - prev_close) / prev_close * 100) if prev_close else 0.0,
            }
        )
    return pd.DataFrame(rows)
