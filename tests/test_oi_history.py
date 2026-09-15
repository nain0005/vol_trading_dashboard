"""Correctness tests for app.oi_history.

Points HISTORY_DIR at a pytest tmp_path for every test so nothing here
touches real logged data, and covers: idempotent per-day logging, correct
day-over-day change/pct-change/days-since-prior arithmetic, and the
first-observation-has-no-prior (NaN, not zero) edge case.
"""
import pandas as pd
import pytest

from app import oi_history


@pytest.fixture(autouse=True)
def _isolated_history_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(oi_history, "HISTORY_DIR", tmp_path / "oi_history")


def _chain(rows):
    return pd.DataFrame(rows, columns=["strike", "type", "open_interest", "volume", "iv", "mid"])


def test_log_snapshot_writes_and_load_history_reads_it_back():
    chain = _chain([
        {"strike": 100.0, "type": "call", "open_interest": 500, "volume": 20, "iv": 0.3, "mid": 2.5},
        {"strike": 100.0, "type": "put", "open_interest": 300, "volume": 10, "iv": 0.3, "mid": 2.1},
    ])
    wrote = oi_history.log_snapshot("AAPL", "2026-12-18", chain, snapshot_date="2026-09-15")
    assert wrote is True

    hist = oi_history.load_history("AAPL", "2026-12-18")
    assert len(hist) == 2
    assert set(hist["open_interest"]) == {500, 300}
    assert (hist["date"] == pd.Timestamp("2026-09-15")).all()


def test_logging_the_same_day_twice_is_a_no_op():
    chain = _chain([{"strike": 100.0, "type": "call", "open_interest": 500, "volume": 20, "iv": 0.3, "mid": 2.5}])
    first = oi_history.log_snapshot("AAPL", "2026-12-18", chain, snapshot_date="2026-09-15")
    second = oi_history.log_snapshot("AAPL", "2026-12-18", chain, snapshot_date="2026-09-15")
    assert first is True
    assert second is False
    assert len(oi_history.load_history("AAPL", "2026-12-18")) == 1  # not duplicated


def test_logging_a_different_day_appends_not_overwrites():
    chain_day1 = _chain([{"strike": 100.0, "type": "call", "open_interest": 500, "volume": 20, "iv": 0.3, "mid": 2.5}])
    chain_day2 = _chain([{"strike": 100.0, "type": "call", "open_interest": 550, "volume": 30, "iv": 0.3, "mid": 2.6}])
    oi_history.log_snapshot("AAPL", "2026-12-18", chain_day1, snapshot_date="2026-09-15")
    oi_history.log_snapshot("AAPL", "2026-12-18", chain_day2, snapshot_date="2026-09-16")

    hist = oi_history.load_history("AAPL", "2026-12-18")
    assert len(hist) == 2
    assert sorted(hist["open_interest"]) == [500, 550]


def test_log_snapshot_returns_false_for_empty_chain():
    assert oi_history.log_snapshot("AAPL", "2026-12-18", pd.DataFrame(), snapshot_date="2026-09-15") is False


def test_load_history_empty_when_nothing_logged():
    hist = oi_history.load_history("NOPE", "2026-12-18")
    assert hist.empty
    assert list(hist.columns) == ["date", "strike", "type", "open_interest", "volume"]


def test_daily_oi_change_arithmetic_across_three_snapshots():
    for date_str, oi in [("2026-09-01", 400), ("2026-09-03", 500), ("2026-09-08", 450)]:
        chain = _chain([{"strike": 100.0, "type": "call", "open_interest": oi, "volume": 10, "iv": 0.3, "mid": 2.0}])
        oi_history.log_snapshot("AAPL", "2026-12-18", chain, snapshot_date=date_str)

    hist = oi_history.load_history("AAPL", "2026-12-18")
    changes = oi_history.daily_oi_change(hist)
    assert len(changes) == 3

    first, second, third = changes.iloc[0], changes.iloc[1], changes.iloc[2]
    assert pd.isna(first["prior_open_interest"])  # no prior snapshot -- NaN, not 0
    assert pd.isna(first["oi_change"])

    assert second["oi_change"] == pytest.approx(100)  # 500 - 400
    assert second["oi_pct_change"] == pytest.approx(0.25)  # 100/400
    assert second["days_since_prior"] == 2  # 09-03 minus 09-01

    assert third["oi_change"] == pytest.approx(-50)  # 450 - 500
    assert third["oi_pct_change"] == pytest.approx(-0.10)
    assert third["days_since_prior"] == 5  # 09-08 minus 09-03


def test_daily_oi_change_handles_zero_prior_without_a_divide_by_zero_crash():
    for date_str, oi in [("2026-09-01", 0), ("2026-09-02", 50)]:
        chain = _chain([{"strike": 100.0, "type": "call", "open_interest": oi, "volume": 10, "iv": 0.3, "mid": 2.0}])
        oi_history.log_snapshot("AAPL", "2026-12-18", chain, snapshot_date=date_str)

    hist = oi_history.load_history("AAPL", "2026-12-18")
    changes = oi_history.daily_oi_change(hist)
    second = changes.iloc[1]
    assert second["oi_change"] == pytest.approx(50)
    assert pd.isna(second["oi_pct_change"])  # undefined (0 -> 50), not inf and not a crash


def test_daily_oi_change_tracks_each_strike_and_type_independently():
    day1 = _chain([
        {"strike": 100.0, "type": "call", "open_interest": 400, "volume": 10, "iv": 0.3, "mid": 2.0},
        {"strike": 100.0, "type": "put", "open_interest": 200, "volume": 5, "iv": 0.3, "mid": 1.5},
        {"strike": 105.0, "type": "call", "open_interest": 300, "volume": 8, "iv": 0.3, "mid": 1.0},
    ])
    day2 = _chain([
        {"strike": 100.0, "type": "call", "open_interest": 450, "volume": 12, "iv": 0.3, "mid": 2.1},
        {"strike": 100.0, "type": "put", "open_interest": 180, "volume": 6, "iv": 0.3, "mid": 1.4},
        {"strike": 105.0, "type": "call", "open_interest": 320, "volume": 9, "iv": 0.3, "mid": 1.1},
    ])
    oi_history.log_snapshot("AAPL", "2026-12-18", day1, snapshot_date="2026-09-15")
    oi_history.log_snapshot("AAPL", "2026-12-18", day2, snapshot_date="2026-09-16")

    changes = oi_history.daily_oi_change(oi_history.load_history("AAPL", "2026-12-18"))
    latest = changes[changes["date"] == pd.Timestamp("2026-09-16")].set_index(["strike", "type"])
    assert latest.loc[(100.0, "call"), "oi_change"] == pytest.approx(50)
    assert latest.loc[(100.0, "put"), "oi_change"] == pytest.approx(-20)
    assert latest.loc[(105.0, "call"), "oi_change"] == pytest.approx(20)


def test_different_symbol_or_expiration_is_a_separate_history():
    chain = _chain([{"strike": 100.0, "type": "call", "open_interest": 500, "volume": 20, "iv": 0.3, "mid": 2.5}])
    oi_history.log_snapshot("AAPL", "2026-12-18", chain, snapshot_date="2026-09-15")
    oi_history.log_snapshot("AAPL", "2026-11-20", chain, snapshot_date="2026-09-15")
    oi_history.log_snapshot("MSFT", "2026-12-18", chain, snapshot_date="2026-09-15")

    assert len(oi_history.load_history("AAPL", "2026-12-18")) == 1
    assert len(oi_history.load_history("AAPL", "2026-11-20")) == 1
    assert len(oi_history.load_history("MSFT", "2026-12-18")) == 1
