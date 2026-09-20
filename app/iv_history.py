"""Local, growing history of ATM implied vol snapshots per (symbol,
expiration) -- the IV analog of app.oi_history, same reasoning and same
idempotent-per-day pattern.

Robinhood's API only exposes a CURRENT option chain (hence a current ATM
IV) -- there's no historical IV time series to pull, the same limitation
oi_history.py documents for open interest. This module is how an actual
"IV rank" / "IV percentile" (today's IV relative to its OWN logged range)
becomes possible at all: each time the Vol Skew tab is viewed for a given
(symbol, expiration), today's ATM IV gets appended here (once per calendar
day, idempotently). History only exists from whenever logging started,
and only grows on days you actually open that chain -- this app has no
background scheduler/daemon, so there's no continuous logging while
you're not looking at it. On day one (and for a long while after) there
simply isn't enough logged history for a real percentile -- that's the
correct, honest state to show, not a bug to paper over.

Caveat worth being explicit about: history here is kept per (symbol,
expiration), and as calendar days pass, days-to-expiration for that same
expiration keeps shrinking. Comparing today's ATM IV against a reading
logged weeks ago is therefore partly comparing across different points on
the term structure, not a pure apples-to-apples "vol history" -- term
structure alone can drift IV even with no change in the market's actual
vol outlook. Real IV-rank products usually solve this with a fixed
constant-maturity (e.g. 30-day) interpolated IV; this app doesn't have
enough chain history to interpolate a constant-maturity series yet, so it
logs the specific expiration you're actually looking at instead. That's a
real, known simplification -- flagged here and in the dashboard copy, not
hidden.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "iv_history"

_HISTORY_COLUMNS = ["date", "atm_iv"]

# Below this many logged observations, min/max-based IV rank is little more
# than "is today above or below the one or two other days we've seen" --
# technically computable but not a number worth trusting. Chosen to roughly
# match a couple of trading weeks, the same spirit as oi_history gating its
# surface/day-over-day views on "at least 2 logged days".
MIN_OBSERVATIONS_FOR_RANK = 10


def _history_path(symbol: str, expiration: str) -> Path:
    return HISTORY_DIR / f"{symbol.strip().upper()}_{expiration}.csv"


def _read_raw(path: Path) -> pd.DataFrame:
    """Read defensively -- see log_snapshot's docstring for the concurrent-
    write race this guards against. A stray duplicate header row mid-file
    shows up as the literal string "date"/"atm_iv" where real values
    should be, and that one bad row forces pandas to read the WHOLE
    atm_iv column as strings (mixed numeric/text can't infer as numeric),
    not just flag that one row -- so re-coercing only "date" would leave
    every atm_iv value silently a string. Drop any row that doesn't parse
    as a real date or a real number, then explicitly re-coerce atm_iv
    back to float."""
    if not path.exists():
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    df = pd.read_csv(path)
    if df.empty:
        return df
    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    bad = parsed_date.isna() | pd.to_numeric(df["atm_iv"], errors="coerce").isna()
    if bad.any():
        df = df.loc[~bad].copy()
        parsed_date = parsed_date.loc[~bad]
    df["date"] = parsed_date
    df["atm_iv"] = pd.to_numeric(df["atm_iv"])
    return df


def log_snapshot(symbol: str, expiration: str, atm_iv: float | None, snapshot_date: str | None = None) -> bool:
    """Append today's ATM IV for this (symbol, expiration), unless today's
    date is already logged or atm_iv is None (chain had no usable ATM
    quote). Returns True if a new snapshot was written.

    Writes via read-everything -> merge -> atomic replace, not CSV append
    mode -- see oi_history.log_snapshot's docstring (same fix, same reason:
    concurrent Streamlit Cloud sessions racing the append-mode
    `header=not path.exists()` check corrupted a live deployment's file)."""
    if atm_iv is None:
        return False
    date_str = snapshot_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    path = _history_path(symbol, expiration)
    existing = _read_raw(path)
    if not existing.empty and date_str in existing["date"].dt.strftime("%Y-%m-%d").values:
        return False

    snapshot = pd.DataFrame([{"date": date_str, "atm_iv": float(atm_iv)}])
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
    """All logged ATM IV snapshots for this (symbol, expiration), sorted by
    date. Empty DataFrame (same columns, zero rows) if nothing's logged."""
    df = _read_raw(_history_path(symbol, expiration))
    if df.empty:
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    return df.sort_values("date").reset_index(drop=True)


def iv_rank_percentile(history: pd.DataFrame, current_iv: float | None) -> dict:
    """Where current_iv sits relative to the logged history for this
    (symbol, expiration).

    - iv_rank: (current - min) / (max - min) over history, i.e. "how far
      up the logged range is today's IV" -- the classic 0-100% IV Rank.
      None if the range is degenerate (max == min: every logged reading,
      including possibly today's, is identical -- there's no range to
      rank a position within, so this is undefined rather than an
      arbitrary 0% or 100%, matching this codebase's None-means-undefined
      convention elsewhere (see risk:reward in spread_selection.py)).
    - iv_percentile: fraction of logged observations at or below
      current_iv, i.e. "what % of days was IV this low or lower" -- well
      defined even when iv_rank isn't (a flat history where current
      equals every logged value is a valid, non-degenerate percentile:
      100%, it's been at least this low/high every day logged).
    - n_observations: how many days are logged, so the UI can show/gate
      on sample size honestly.
    - min_iv / max_iv: the logged range itself, for direct display.

    min_iv/max_iv describe the logged HISTORY alone, so they're populated
    whenever any history exists at all -- even on a day current_iv itself
    is None (e.g. today's chain had no usable ATM quote), so a caller can
    still show "logged range so far" without it silently going missing on
    exactly the day the live reading failed. iv_rank/iv_percentile are
    None when there's nothing logged yet, current_iv is None, or history
    has fewer than 2 points (see MIN_OBSERVATIONS_FOR_RANK for the higher
    bar used to actually surface a "trustworthy" rank in the UI -- this
    function itself only guards against literal undefined math, not
    sample-size confidence)."""
    n = len(history)
    if n == 0:
        return {"iv_rank": None, "iv_percentile": None, "n_observations": 0, "min_iv": None, "max_iv": None}

    values = history["atm_iv"]
    min_iv, max_iv = float(values.min()), float(values.max())

    if n < 2 or current_iv is None:
        return {"iv_rank": None, "iv_percentile": None, "n_observations": n, "min_iv": min_iv, "max_iv": max_iv}

    iv_rank = None if max_iv == min_iv else (current_iv - min_iv) / (max_iv - min_iv)
    iv_percentile = float((values <= current_iv).mean())

    return {
        "iv_rank": iv_rank,
        "iv_percentile": iv_percentile,
        "n_observations": n,
        "min_iv": min_iv,
        "max_iv": max_iv,
    }
