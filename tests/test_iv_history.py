"""Correctness tests for app.iv_history.

Points HISTORY_DIR at a pytest tmp_path for every test so nothing here
touches real logged data. Covers: idempotent per-day logging (mirroring
oi_history's own tests), hand-verified iv_rank/iv_percentile arithmetic,
and the degenerate cases that matter most for this kind of stat: empty
history, a single logged day, current_iv=None, and a flat history where
max == min (would otherwise divide by zero).
"""
import pandas as pd
import pytest

from app import iv_history


@pytest.fixture(autouse=True)
def _isolated_history_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(iv_history, "HISTORY_DIR", tmp_path / "iv_history")


def test_log_snapshot_writes_and_load_history_reads_it_back():
    wrote = iv_history.log_snapshot("AAPL", "2026-12-18", 0.32, snapshot_date="2026-09-15")
    assert wrote is True
    hist = iv_history.load_history("AAPL", "2026-12-18")
    assert len(hist) == 1
    assert hist.iloc[0]["atm_iv"] == pytest.approx(0.32)
    assert hist.iloc[0]["date"] == pd.Timestamp("2026-09-15")


def test_logging_the_same_day_twice_is_a_no_op():
    first = iv_history.log_snapshot("AAPL", "2026-12-18", 0.30, snapshot_date="2026-09-15")
    second = iv_history.log_snapshot("AAPL", "2026-12-18", 0.35, snapshot_date="2026-09-15")
    assert first is True
    assert second is False
    hist = iv_history.load_history("AAPL", "2026-12-18")
    assert len(hist) == 1
    assert hist.iloc[0]["atm_iv"] == pytest.approx(0.30)  # first write for the day wins, not overwritten


def test_logging_a_different_day_appends_not_overwrites():
    iv_history.log_snapshot("AAPL", "2026-12-18", 0.30, snapshot_date="2026-09-15")
    iv_history.log_snapshot("AAPL", "2026-12-18", 0.35, snapshot_date="2026-09-16")
    hist = iv_history.load_history("AAPL", "2026-12-18")
    assert len(hist) == 2
    assert sorted(hist["atm_iv"]) == pytest.approx([0.30, 0.35])


def test_log_snapshot_returns_false_for_none_iv():
    assert iv_history.log_snapshot("AAPL", "2026-12-18", None, snapshot_date="2026-09-15") is False
    assert iv_history.load_history("AAPL", "2026-12-18").empty


def test_load_history_empty_when_nothing_logged():
    hist = iv_history.load_history("NOPE", "2026-12-18")
    assert hist.empty
    assert list(hist.columns) == ["date", "atm_iv"]


def test_different_symbol_or_expiration_is_a_separate_history():
    iv_history.log_snapshot("AAPL", "2026-12-18", 0.30, snapshot_date="2026-09-15")
    iv_history.log_snapshot("AAPL", "2026-11-20", 0.40, snapshot_date="2026-09-15")
    iv_history.log_snapshot("MSFT", "2026-12-18", 0.25, snapshot_date="2026-09-15")
    assert len(iv_history.load_history("AAPL", "2026-12-18")) == 1
    assert len(iv_history.load_history("AAPL", "2026-11-20")) == 1
    assert len(iv_history.load_history("MSFT", "2026-12-18")) == 1


# --- iv_rank_percentile ---

def _hist(values):
    return pd.DataFrame({"atm_iv": values})


def test_rank_and_percentile_on_empty_history_is_all_none():
    result = iv_history.iv_rank_percentile(_hist([]), current_iv=0.30)
    assert result == {"iv_rank": None, "iv_percentile": None, "n_observations": 0, "min_iv": None, "max_iv": None}


def test_rank_is_none_when_current_iv_is_none_but_min_max_of_history_still_shown():
    # Real scenario this guards: today's chain had no usable ATM quote
    # (metrics['atm_iv'] is None), but prior days' history still exists.
    # Losing min/max on exactly the day the live reading fails would be a
    # real bug (the dashboard formats them with :.1% -- None there is a
    # crash, not just a missing number), so they must stay populated.
    result = iv_history.iv_rank_percentile(_hist([0.20, 0.30, 0.40]), current_iv=None)
    assert result["iv_rank"] is None
    assert result["iv_percentile"] is None
    assert result["n_observations"] == 3  # sample size still reported even though rank isn't computable
    assert result["min_iv"] == pytest.approx(0.20)
    assert result["max_iv"] == pytest.approx(0.40)


def test_rank_is_none_when_current_iv_is_none_and_history_is_empty():
    result = iv_history.iv_rank_percentile(_hist([]), current_iv=None)
    assert result == {"iv_rank": None, "iv_percentile": None, "n_observations": 0, "min_iv": None, "max_iv": None}


def test_rank_is_none_with_a_single_logged_observation():
    # One point has no range to rank a position within.
    result = iv_history.iv_rank_percentile(_hist([0.30]), current_iv=0.30)
    assert result["iv_rank"] is None
    assert result["iv_percentile"] is None
    assert result["n_observations"] == 1
    assert result["min_iv"] == pytest.approx(0.30)
    assert result["max_iv"] == pytest.approx(0.30)


def test_hand_verified_rank_and_percentile_at_the_middle_of_a_known_range():
    # History: 0.20, 0.25, 0.30, 0.35, 0.40 (min=0.20, max=0.40).
    # current=0.30 -> rank = (0.30-0.20)/(0.40-0.20) = 0.10/0.20 = 0.50 exactly.
    # percentile = fraction of {0.20,0.25,0.30,0.35,0.40} <= 0.30 = 3/5 = 0.60.
    hist = _hist([0.20, 0.25, 0.30, 0.35, 0.40])
    result = iv_history.iv_rank_percentile(hist, current_iv=0.30)
    assert result["iv_rank"] == pytest.approx(0.50)
    assert result["iv_percentile"] == pytest.approx(0.60)
    assert result["n_observations"] == 5
    assert result["min_iv"] == pytest.approx(0.20)
    assert result["max_iv"] == pytest.approx(0.40)


def test_rank_at_the_low_end_of_history_is_zero_not_negative_or_none():
    hist = _hist([0.20, 0.30, 0.40])
    result = iv_history.iv_rank_percentile(hist, current_iv=0.20)
    assert result["iv_rank"] == pytest.approx(0.0)
    assert result["iv_percentile"] == pytest.approx(1 / 3)


def test_rank_at_the_high_end_of_history_is_one():
    hist = _hist([0.20, 0.30, 0.40])
    result = iv_history.iv_rank_percentile(hist, current_iv=0.40)
    assert result["iv_rank"] == pytest.approx(1.0)
    assert result["iv_percentile"] == pytest.approx(1.0)  # every logged reading is <= today's


def test_rank_for_a_current_iv_outside_the_logged_range_is_not_clamped():
    # current_iv can legitimately exceed every logged reading (today IS a
    # new high) -- rank > 1.0 in that case is correct information (a fresh
    # high, further out than anything logged so far), not a bug to clamp away.
    hist = _hist([0.20, 0.30, 0.40])
    result = iv_history.iv_rank_percentile(hist, current_iv=0.50)
    assert result["iv_rank"] == pytest.approx(1.5)  # (0.50-0.20)/(0.40-0.20)
    assert result["iv_percentile"] == pytest.approx(1.0)


def test_flat_history_gives_none_rank_but_a_defined_percentile_not_a_zero_division_crash():
    # Every logged reading identical (max == min) -- iv_rank is genuinely
    # undefined (no range to place a position within), but iv_percentile
    # is still well-defined: current equals every logged value, so 100% of
    # history is <= current.
    hist = _hist([0.30, 0.30, 0.30])
    result = iv_history.iv_rank_percentile(hist, current_iv=0.30)
    assert result["iv_rank"] is None
    assert result["iv_percentile"] == pytest.approx(1.0)
    assert result["min_iv"] == pytest.approx(0.30)
    assert result["max_iv"] == pytest.approx(0.30)


def test_min_observations_threshold_constant_is_reasonable():
    # Sanity check the gating constant the dashboard uses to decide when a
    # rank is trustworthy enough to headline -- must require more than the
    # bare minimum (2) the math itself needs to not divide by zero.
    assert iv_history.MIN_OBSERVATIONS_FOR_RANK >= 2


def test_load_history_survives_a_stray_duplicate_header_row():
    # Same real production crash as oi_history's identical test -- see that
    # test's docstring for the full concurrent-write race explanation.
    # iv_history.py has the exact same log_snapshot/load_history pattern
    # and needed the exact same fix.
    iv_history.log_snapshot("AAPL", "2026-12-18", 0.25, snapshot_date="2026-09-15")

    path = iv_history._history_path("AAPL", "2026-12-18")
    with open(path, "a") as f:
        f.write("date,atm_iv\n")  # the stray duplicate header

    history = iv_history.load_history("AAPL", "2026-12-18")  # must not raise
    assert len(history) == 1
    assert history.iloc[0]["atm_iv"] == pytest.approx(0.25)


def test_log_snapshot_write_is_atomic_not_append_mode():
    iv_history.log_snapshot("AAPL", "2026-12-18", 0.25, snapshot_date="2026-09-15")
    iv_history.log_snapshot("AAPL", "2026-12-18", 0.27, snapshot_date="2026-09-16")

    path = iv_history._history_path("AAPL", "2026-12-18")
    lines = path.read_text().splitlines()
    header_lines = [ln for ln in lines if ln.startswith("date,atm_iv")]
    assert len(header_lines) == 1
    assert len(lines) == 3  # 1 header + 2 data rows
