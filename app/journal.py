"""Trade journal: normalize fill history into an exportable CSV log.

Per-trade notes/tags (the WHY behind a fill, freeform tags to find it
again later) live in app.journal_notes, keyed by order_id, and get merged
onto the table this module builds -- see render_journal in dashboard.py
for the UI that edits them and app.journal_notes for the storage."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports"


def build_journal(order_history: pd.DataFrame) -> pd.DataFrame:
    if order_history.empty:
        return order_history
    journal = order_history.copy()
    journal["date"] = pd.to_datetime(journal["date"]).dt.tz_convert(None)
    journal = journal.sort_values("date", ascending=False)
    return journal[
        [c for c in ["date", "instrument_type", "symbol", "side", "quantity", "price", "amount", "fees", "strategy", "order_id"] if c in journal.columns]
    ]


def export_csv(journal: pd.DataFrame, filename: str | None = None) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"trade_journal_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    path = EXPORT_DIR / filename
    journal.to_csv(path, index=False)
    return path
