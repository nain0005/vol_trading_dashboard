"""Local, growing history of open-interest snapshots per (symbol, expiration).

Robinhood's API only exposes a CURRENT snapshot of open interest -- there's
no historical OI time series to pull. This module is how day-over-day OI
changes and an OI-over-time surface become possible at all: each time the
Vol Skew tab is viewed for a given (symbol, expiration), today's chain OI
gets appended here (once per calendar day, idempotently). History only
exists from whenever logging started, and only grows on days you actually
open that chain -- this app has no background scheduler/daemon, so there's
no such thing as continuous logging while you're not looking at it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "oi_history"

_HISTORY_COLUMNS = ["date", "strike", "type", "open_interest", "volume"]
_CHANGE_COLUMNS = _HISTORY_COLUMNS + ["prior_open_interest", "oi_change", "oi_pct_change", "days_since_prior"]


def _history_path(symbol: str, expiration: str) -> Path:
    return HISTORY_DIR / f"{symbol.strip().upper()}_{expiration}.csv"


def log_snapshot(symbol: str, expiration: str, chain: pd.DataFrame, snapshot_date: str | None = None) -> bool:
    """Append today's (strike, type, open_interest, volume) rows for this
    chain, unless today's date is already logged. Returns True if a new
    snapshot was written, False if today was already logged or chain was
    empty."""
    if chain.empty:
        return False
    date_str = snapshot_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    path = _history_path(symbol, expiration)
    if path.exists():
        existing_dates = pd.read_csv(path, usecols=["date"])["date"].astype(str)
        if date_str in existing_dates.values:
            return False

    snapshot = chain[["strike", "type", "open_interest", "volume"]].copy()
    snapshot.insert(0, "date", date_str)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(path, mode="a", header=not path.exists(), index=False)
    return True


def load_history(symbol: str, expiration: str) -> pd.DataFrame:
    """All logged snapshots for this (symbol, expiration), sorted by date.
    Empty DataFrame (same columns, zero rows) if nothing's been logged yet."""
    path = _history_path(symbol, expiration)
    if not path.exists():
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "type", "strike"]).reset_index(drop=True)


def daily_oi_change(history: pd.DataFrame) -> pd.DataFrame:
    """Change in open interest per (strike, type) between CONSECUTIVE
    LOGGED snapshots -- not necessarily consecutive calendar days, since
    logging only happens on days this chain was actually viewed.
    `days_since_prior` makes that gap visible rather than silently
    labeling a multi-day gap "daily". First observation of each
    (strike, type) has no prior snapshot to compare against -- those rows
    carry NaN change/pct-change/days-since-prior, not zero."""
    if history.empty:
        return pd.DataFrame(columns=_CHANGE_COLUMNS)

    df = history.sort_values(["strike", "type", "date"]).copy()
    grouped = df.groupby(["strike", "type"])
    df["prior_open_interest"] = grouped["open_interest"].shift(1)
    df["oi_change"] = df["open_interest"] - df["prior_open_interest"]
    df["oi_pct_change"] = df["oi_change"] / df["prior_open_interest"].replace(0, pd.NA)
    df["days_since_prior"] = grouped["date"].diff().dt.days
    return df.sort_values(["date", "type", "strike"]).reset_index(drop=True)
