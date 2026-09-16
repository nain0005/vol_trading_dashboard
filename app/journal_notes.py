"""Local, persistent store of per-trade notes/tags for the journal.

Order history (app.data_fetch.get_order_history) already carries an
`order_id` column, unique per filled order (Robinhood's own order id, or a
synthetic-but-still-unique one in demo mode). Notes/tags live in a
separate local file keyed by that `order_id`, not bolted onto the order
history itself -- order history is a live pull from Robinhood every time
the page loads, so anything we wrote directly onto it would be lost on the
next refresh. This mirrors app.oi_history's own reasoning for keeping
logged data in its own local file rather than trying to persist it inside
a value that gets refetched from the broker.

One CSV, one row per order_id (last write wins) -- not append-only like
oi_history, since a note is a single current value per trade, not a
growing time series. Tags are freeform (no fixed vocabulary): stored as a
single semicolon-separated string per row and split/joined at the edges,
same spirit as everything else in this app that shows a human a plain
CSV they could open themselves.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NOTES_PATH = Path(__file__).resolve().parent.parent / "data" / "journal_notes.csv"

_COLUMNS = ["order_id", "tags", "notes", "updated_at"]


def load_notes() -> pd.DataFrame:
    """All saved notes/tags, one row per order_id. Empty (but correctly
    shaped) DataFrame if nothing has been saved yet."""
    if not NOTES_PATH.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(NOTES_PATH, dtype={"order_id": str})
    df["tags"] = df["tags"].fillna("")
    df["notes"] = df["notes"].fillna("")
    return df[_COLUMNS]


def parse_tags(tags: str) -> list[str]:
    """'earnings; mistake ; thesis-break' -> ['earnings', 'mistake', 'thesis-break'].
    Blank/whitespace-only entries are dropped; order and case are preserved
    (tags are freeform text, not a controlled vocabulary -- no normalization
    beyond trimming whitespace around each one)."""
    if not tags:
        return []
    return [t.strip() for t in tags.split(";") if t.strip()]


def format_tags(tags: list[str]) -> str:
    return "; ".join(t.strip() for t in tags if t.strip())


def save_note(order_id: str, tags: str = "", notes: str = "") -> pd.DataFrame:
    """Upsert the note/tags for one order_id (last write wins). Passing
    both fields blank deletes the row entirely -- an emptied-out note
    shouldn't leave a dead placeholder row in the file forever. Returns
    the full notes table after the write."""
    order_id = str(order_id)
    df = load_notes()
    df = df[df["order_id"] != order_id]

    tags = (tags or "").strip()
    notes = (notes or "").strip()
    if tags or notes:
        new_row = pd.DataFrame(
            [{"order_id": order_id, "tags": tags, "notes": notes, "updated_at": pd.Timestamp.now().isoformat()}]
        )
        df = pd.concat([df, new_row], ignore_index=True)

    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[_COLUMNS].to_csv(NOTES_PATH, index=False)
    return df[_COLUMNS]


def get_note(order_id: str) -> dict:
    """Current {tags, notes} for one order_id -- empty strings if nothing
    saved yet, never a missing-key error, so callers can always index it."""
    df = load_notes()
    match = df[df["order_id"] == str(order_id)]
    if match.empty:
        return {"tags": "", "notes": ""}
    row = match.iloc[0]
    return {"tags": row["tags"], "notes": row["notes"]}


def merge_notes(journal_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join saved tags/notes onto a journal table by order_id. Rows
    with no saved note get empty strings, not NaN, so downstream text
    search/filtering never has to special-case missing values."""
    if journal_df.empty:
        out = journal_df.copy()
        out["tags"] = pd.Series(dtype=str)
        out["notes"] = pd.Series(dtype=str)
        return out

    notes = load_notes()[["order_id", "tags", "notes"]].copy()
    out = journal_df.copy()
    out["order_id"] = out["order_id"].astype(str)
    notes["order_id"] = notes["order_id"].astype(str)
    merged = out.merge(notes, on="order_id", how="left")
    merged["tags"] = merged["tags"].fillna("")
    merged["notes"] = merged["notes"].fillna("")
    return merged


def all_tags(journal_with_notes: pd.DataFrame) -> list[str]:
    """Sorted, deduplicated tag vocabulary actually in use across the given
    (already-merged) journal -- used to populate a filter dropdown from
    real data rather than a hardcoded list, since tags are freeform."""
    if journal_with_notes.empty or "tags" not in journal_with_notes.columns:
        return []
    seen: set[str] = set()
    for tags in journal_with_notes["tags"]:
        seen.update(parse_tags(tags))
    return sorted(seen)


def filter_journal(journal_with_notes: pd.DataFrame, tag: str | None = None, text: str | None = None) -> pd.DataFrame:
    """Filter an already-merged (tags/notes present) journal by an exact
    tag match (case-insensitive) and/or free-text found in symbol, notes,
    or tags (case-insensitive substring). Either filter left blank/None is
    skipped; both together are AND'd."""
    df = journal_with_notes
    if df.empty:
        return df

    if tag:
        tag_lower = tag.strip().lower()
        mask = df["tags"].apply(lambda t: tag_lower in [x.lower() for x in parse_tags(t)])
        df = df[mask]

    if text:
        text_lower = text.strip().lower()
        haystack = (
            df.get("symbol", pd.Series("", index=df.index)).astype(str)
            + " " + df.get("notes", pd.Series("", index=df.index)).astype(str)
            + " " + df.get("tags", pd.Series("", index=df.index)).astype(str)
        )
        df = df[haystack.str.lower().str.contains(text_lower, na=False, regex=False)]

    return df
