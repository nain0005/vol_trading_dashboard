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

import os
import uuid
from pathlib import Path

import pandas as pd

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "oi_history"

_HISTORY_COLUMNS = ["date", "strike", "type", "open_interest", "volume"]
_CHANGE_COLUMNS = _HISTORY_COLUMNS + ["prior_open_interest", "oi_change", "oi_pct_change", "days_since_prior"]


def _history_path(symbol: str, expiration: str) -> Path:
    return HISTORY_DIR / f"{symbol.strip().upper()}_{expiration}.csv"


_NUMERIC_COLUMNS = ["strike", "open_interest", "volume"]


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a history CSV defensively -- a concurrent-write race (see
    log_snapshot) or any other partial/corrupted write can leave a stray
    duplicate header mid-file, which shows up as the literal string "date"
    (etc.) in what should be a data row. That single bad row forces pandas
    to read EVERY column as strings (a mixed numeric/text column can't be
    inferred as numeric), not just the "date" column -- so a fix that only
    re-coerces "date" leaves every other field silently a string. Drop any
    row that doesn't parse as a real date, then explicitly re-coerce the
    numeric columns back to floats (any further-unparseable values are
    themselves dropped too, since they're already-anomalous data)."""
    if not path.exists():
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    df = pd.read_csv(path)
    if df.empty:
        return df
    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    bad = parsed_date.isna()
    for col in _NUMERIC_COLUMNS:
        bad = bad | pd.to_numeric(df[col], errors="coerce").isna()
    if bad.any():
        df = df.loc[~bad].copy()
        parsed_date = parsed_date.loc[~bad]
    df["date"] = parsed_date
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col])
    return df


def log_snapshot(symbol: str, expiration: str, chain: pd.DataFrame, snapshot_date: str | None = None) -> bool:
    """Append today's (strike, type, open_interest, volume) rows for this
    chain, unless today's date is already logged. Returns True if a new
    snapshot was written, False if today was already logged or chain was
    empty.

    Writes via read-everything -> merge -> atomic replace (not CSV append
    mode) specifically because Streamlit Community Cloud can run multiple
    concurrent sessions against the same deployment, each independently
    calling this for the same (symbol, expiration): append mode's
    `header=not path.exists()` check is a classic TOCTOU race -- two
    sessions can both see "file doesn't exist yet" and both write a
    header, leaving a stray header row stuck mid-file that later crashes
    date parsing (this happened in production; see _read_raw's docstring
    for the defensive reader-side fix, and this function for the actual
    prevention). os.replace is atomic on both POSIX and Windows, so a
    concurrent writer's result is always either the fully-old or the
    fully-new file, never a partial mix of both."""
    if chain.empty:
        return False
    date_str = snapshot_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    path = _history_path(symbol, expiration)
    existing = _read_raw(path)
    if not existing.empty and date_str in existing["date"].dt.strftime("%Y-%m-%d").values:
        return False

    snapshot = chain[["strike", "type", "open_interest", "volume"]].copy()
    snapshot.insert(0, "date", date_str)
    if not existing.empty:
        existing = existing.copy()
        existing["date"] = existing["date"].dt.strftime("%Y-%m-%d")
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)  # atomic on POSIX and Windows
    return True


def load_history(symbol: str, expiration: str) -> pd.DataFrame:
    """All logged snapshots for this (symbol, expiration), sorted by date.
    Empty DataFrame (same columns, zero rows) if nothing's been logged yet."""
    df = _read_raw(_history_path(symbol, expiration))
    if df.empty:
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
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
