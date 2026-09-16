"""Correctness tests for app.journal_notes.

Points NOTES_PATH at a pytest tmp_path for every test so nothing here
touches real saved notes, and covers: upsert-by-order_id (last write
wins), delete-on-blank, tag parsing/formatting round-trips, merging notes
onto a journal table without introducing NaNs, and tag/text filtering
including degenerate cases (empty journal, tag with special characters,
free-text search with zero results).
"""
import pandas as pd
import pytest

from app import journal_notes


@pytest.fixture(autouse=True)
def _isolated_notes_path(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_notes, "NOTES_PATH", tmp_path / "journal_notes.csv")


def test_load_notes_empty_when_nothing_saved():
    df = journal_notes.load_notes()
    assert df.empty
    assert list(df.columns) == ["order_id", "tags", "notes", "updated_at"]


def test_save_and_get_note_round_trips():
    journal_notes.save_note("abc123", tags="earnings; thesis-break", notes="Sized too big into print")
    note = journal_notes.get_note("abc123")
    assert note["tags"] == "earnings; thesis-break"
    assert note["notes"] == "Sized too big into print"


def test_get_note_for_unknown_order_id_returns_empty_strings_not_error():
    note = journal_notes.get_note("does-not-exist")
    assert note == {"tags": "", "notes": ""}


def test_save_note_is_an_upsert_last_write_wins():
    journal_notes.save_note("abc123", tags="earnings", notes="first draft")
    journal_notes.save_note("abc123", tags="earnings; revised", notes="second draft")
    df = journal_notes.load_notes()
    assert len(df) == 1  # not duplicated
    note = journal_notes.get_note("abc123")
    assert note["tags"] == "earnings; revised"
    assert note["notes"] == "second draft"


def test_save_note_with_both_fields_blank_deletes_the_row():
    journal_notes.save_note("abc123", tags="earnings", notes="some note")
    assert len(journal_notes.load_notes()) == 1
    journal_notes.save_note("abc123", tags="", notes="")
    assert journal_notes.load_notes().empty


def test_different_order_ids_are_independent_rows():
    journal_notes.save_note("order-1", tags="mistake", notes="chased the move")
    journal_notes.save_note("order-2", tags="thesis", notes="earnings beat expected")
    df = journal_notes.load_notes()
    assert len(df) == 2
    assert journal_notes.get_note("order-1")["tags"] == "mistake"
    assert journal_notes.get_note("order-2")["tags"] == "thesis"


def test_order_id_is_stored_and_matched_as_string_not_int():
    # order_ids from real data can be numeric-looking strings ("1000") or
    # UUIDs -- this must not silently coerce/compare as an int.
    journal_notes.save_note(1000, tags="rolled", notes="")
    assert journal_notes.get_note("1000")["tags"] == "rolled"


def test_parse_tags_splits_trims_and_drops_blanks():
    assert journal_notes.parse_tags("earnings; mistake ;  thesis-break ;; ") == ["earnings", "mistake", "thesis-break"]


def test_parse_tags_handles_empty_string():
    assert journal_notes.parse_tags("") == []


def test_format_tags_round_trips_with_parse_tags():
    tags = ["earnings", "mistake", "thesis-break"]
    formatted = journal_notes.format_tags(tags)
    assert journal_notes.parse_tags(formatted) == tags


def test_merge_notes_fills_missing_with_empty_strings_not_nan():
    journal = pd.DataFrame([
        {"order_id": "1", "symbol": "AAPL"},
        {"order_id": "2", "symbol": "MSFT"},
    ])
    journal_notes.save_note("1", tags="earnings", notes="good entry")
    merged = journal_notes.merge_notes(journal)
    assert merged.loc[merged["order_id"] == "1", "tags"].iloc[0] == "earnings"
    assert merged.loc[merged["order_id"] == "2", "tags"].iloc[0] == ""
    assert merged.loc[merged["order_id"] == "2", "notes"].iloc[0] == ""
    assert not merged["tags"].isna().any()
    assert not merged["notes"].isna().any()


def test_merge_notes_on_empty_journal_returns_empty_with_tag_notes_columns():
    empty = pd.DataFrame(columns=["order_id", "symbol"])
    merged = journal_notes.merge_notes(empty)
    assert merged.empty
    assert "tags" in merged.columns
    assert "notes" in merged.columns


def test_all_tags_is_sorted_deduplicated_vocabulary_in_use():
    journal = pd.DataFrame([
        {"order_id": "1", "tags": "earnings; mistake"},
        {"order_id": "2", "tags": "mistake; thesis"},
        {"order_id": "3", "tags": ""},
    ])
    assert journal_notes.all_tags(journal) == ["earnings", "mistake", "thesis"]


def test_all_tags_on_empty_journal_returns_empty_list():
    assert journal_notes.all_tags(pd.DataFrame(columns=["order_id", "tags"])) == []


def test_filter_journal_by_tag_is_case_insensitive_exact_match():
    journal = pd.DataFrame([
        {"order_id": "1", "symbol": "AAPL", "tags": "Earnings", "notes": ""},
        {"order_id": "2", "symbol": "MSFT", "tags": "mistake", "notes": ""},
        {"order_id": "3", "symbol": "TSLA", "tags": "earnings-adjacent", "notes": ""},  # not an exact tag match
    ])
    result = journal_notes.filter_journal(journal, tag="earnings")
    assert list(result["order_id"]) == ["1"]  # exact tag match only, not a substring of "earnings-adjacent"


def test_filter_journal_by_free_text_searches_symbol_notes_and_tags():
    journal = pd.DataFrame([
        {"order_id": "1", "symbol": "AAPL", "tags": "earnings", "notes": "sized too big"},
        {"order_id": "2", "symbol": "MSFT", "tags": "", "notes": "clean entry"},
    ])
    assert list(journal_notes.filter_journal(journal, text="too big")["order_id"]) == ["1"]
    assert list(journal_notes.filter_journal(journal, text="msft")["order_id"]) == ["2"]
    assert list(journal_notes.filter_journal(journal, text="earnings")["order_id"]) == ["1"]


def test_filter_journal_with_zero_results_returns_empty_not_error():
    journal = pd.DataFrame([{"order_id": "1", "symbol": "AAPL", "tags": "earnings", "notes": ""}])
    result = journal_notes.filter_journal(journal, text="nonexistent-needle")
    assert result.empty


def test_filter_journal_handles_tag_with_special_characters():
    journal = pd.DataFrame([
        {"order_id": "1", "symbol": "AAPL", "tags": "P&L review (Q3)", "notes": ""},
        {"order_id": "2", "symbol": "MSFT", "tags": "normal-tag", "notes": ""},
    ])
    result = journal_notes.filter_journal(journal, tag="P&L review (Q3)")
    assert list(result["order_id"]) == ["1"]
    # regex metacharacters in free-text search must be treated literally, not as a pattern
    text_result = journal_notes.filter_journal(journal, text="(Q3)")
    assert list(text_result["order_id"]) == ["1"]


def test_filter_journal_on_empty_journal_returns_empty():
    empty = pd.DataFrame(columns=["order_id", "symbol", "tags", "notes"])
    assert journal_notes.filter_journal(empty, tag="anything").empty
    assert journal_notes.filter_journal(empty, text="anything").empty


def test_filter_journal_with_no_filters_returns_everything_unchanged():
    journal = pd.DataFrame([
        {"order_id": "1", "symbol": "AAPL", "tags": "earnings", "notes": ""},
        {"order_id": "2", "symbol": "MSFT", "tags": "", "notes": ""},
    ])
    result = journal_notes.filter_journal(journal)
    assert len(result) == 2
