"""Realized round-trip trade matching + win-rate stats, built from
data_fetch.get_order_history's fill log.

FIFO-matches each (symbol, contract) group's chronological buy/sell fills
into closed round-trip trades (long: buy-then-sell; short: sell-then-buy)
-- the same mechanics a broker's own realized-P&L accounting uses, so
partial fills and re-entries net out correctly. Options carry the
standard 100-share contract multiplier; equities don't.

Matched per exact contract, not just underlying symbol: data_fetch's
get_order_history() pulls each option fill's `legs[0]['option']`
instrument URL as `contract_id` -- unique per (underlying, strike,
expiration, type), unlike the chain symbol which every contract on the
same underlying shares. If you hold two different contracts on the same
underlying at once, they're matched independently and can't cross-match.
(order_history without a contract_id column -- e.g. hand-built test
fixtures -- falls back to symbol-only grouping.)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

_CONTRACT_MULTIPLIER = {"equity": 1, "option": 100}


@dataclass
class _Lot:
    qty: float
    price: float
    date: pd.Timestamp


@dataclass
class RoundTrip:
    symbol: str
    instrument_type: str
    direction: str  # "long" (buy opened it) or "short" (sell opened it)
    quantity: float
    entry_price: float
    exit_price: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    realized_pl: float
    is_win: bool


def match_round_trips(order_history: pd.DataFrame) -> list[RoundTrip]:
    """Chronological FIFO match per (symbol, instrument_type, contract) group.
    A buy first closes any open short lots (oldest first), then opens/adds
    to a long lot with whatever quantity is left over; a sell is the
    mirror."""
    if order_history.empty:
        return []

    group_cols = ["symbol", "instrument_type"]
    if "contract_id" in order_history.columns:
        group_cols.append("contract_id")

    trips: list[RoundTrip] = []
    for group_key, group in order_history.groupby(group_cols):
        symbol, instrument_type = group_key[0], group_key[1]
        multiplier = _CONTRACT_MULTIPLIER.get(instrument_type, 1)
        long_lots: deque[_Lot] = deque()
        short_lots: deque[_Lot] = deque()

        for _, fill in group.sort_values("date").iterrows():
            side = str(fill.get("side") or "").lower()
            qty = float(fill["quantity"])
            price = float(fill["price"])
            date = fill["date"]
            if qty <= 0 or price <= 0 or side not in ("buy", "sell"):
                continue

            opposing = short_lots if side == "buy" else long_lots
            remaining = qty
            while remaining > 1e-9 and opposing:
                lot = opposing[0]
                matched = min(remaining, lot.qty)
                direction = "short" if side == "buy" else "long"
                pl = ((lot.price - price) if side == "buy" else (price - lot.price)) * matched * multiplier
                trips.append(RoundTrip(symbol, instrument_type, direction, matched, lot.price, price, lot.date, date, pl, pl > 0))
                lot.qty -= matched
                remaining -= matched
                if lot.qty <= 1e-9:
                    opposing.popleft()

            if remaining > 1e-9:
                (long_lots if side == "buy" else short_lots).append(_Lot(remaining, price, date))

    trips.sort(key=lambda t: t.exit_date)
    return trips


_TRIP_COLUMNS = [
    "symbol", "instrument_type", "direction", "quantity", "entry_price", "exit_price",
    "entry_date", "exit_date", "realized_pl", "is_win", "holding_days",
]


def round_trips_to_frame(trips: list[RoundTrip]) -> pd.DataFrame:
    if not trips:
        return pd.DataFrame(columns=_TRIP_COLUMNS)
    df = pd.DataFrame([t.__dict__ for t in trips])
    df["holding_days"] = (df["exit_date"] - df["entry_date"]).dt.total_seconds() / 86400
    return df[_TRIP_COLUMNS]


def win_rate_stats(trips_df: pd.DataFrame) -> dict:
    """None fields mean undefined (no trades / no wins / no losses), not zero."""
    if trips_df.empty:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": None,
            "avg_win": None, "avg_loss": None, "expectancy": None,
            "total_realized_pl": 0.0, "profit_factor": None,
        }
    wins = trips_df[trips_df["is_win"]]
    losses = trips_df[~trips_df["is_win"]]
    total = len(trips_df)
    win_rate = len(wins) / total
    avg_win = float(wins["realized_pl"].mean()) if not wins.empty else None
    avg_loss = float(losses["realized_pl"].mean()) if not losses.empty else None
    gross_loss = -float(losses["realized_pl"].sum()) if not losses.empty else 0.0
    gross_win = float(wins["realized_pl"].sum()) if not wins.empty else 0.0
    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": win_rate * (avg_win or 0.0) + (1 - win_rate) * (avg_loss or 0.0),
        "total_realized_pl": float(trips_df["realized_pl"].sum()),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
    }


def win_rate_by_instrument_type(trips_df: pd.DataFrame) -> pd.DataFrame:
    if trips_df.empty:
        return pd.DataFrame(columns=["instrument_type", "total_trades", "wins", "win_rate", "total_realized_pl"])
    rows = []
    for instrument_type, group in trips_df.groupby("instrument_type"):
        stats = win_rate_stats(group)
        rows.append({
            "instrument_type": instrument_type,
            "total_trades": stats["total_trades"],
            "wins": stats["wins"],
            "win_rate": stats["win_rate"],
            "total_realized_pl": stats["total_realized_pl"],
        })
    return pd.DataFrame(rows)
