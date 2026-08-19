"""End-to-end smoke tests for the CLI — verifies the whole pipeline (strike
generation -> EV ranking -> sizing -> entry/exit levels) wires together and
produces sane output for both the market-IV-only and realized-vs-implied
code paths."""
from risk_tool.cli import run


def test_cli_runs_end_to_end_with_market_iv_only(capsys):
    exit_code = run([
        "--ticker", "AAPL",
        "--spot", "190",
        "--iv", "0.32",
        "--dte", "30",
        "--direction", "call",
        "--account", "50000",
        "--strike-increment", "5",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Recommended strike" in captured.out
    assert "Position sizing" in captured.out
    assert "Pre-committed entry/exit levels" in captured.out


def test_cli_runs_end_to_end_with_realized_vol_edge_check(capsys):
    exit_code = run([
        "--ticker", "AAPL",
        "--spot", "190",
        "--iv", "0.32",
        "--dte", "30",
        "--direction", "put",
        "--account", "50000",
        "--strike-increment", "5",
        "--realized-vol", "0.45",
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Realized-vs-implied edge check" in captured.out
    assert "EDGE EV" in captured.out


def test_cli_full_kelly_flag_changes_output(capsys):
    run([
        "--ticker", "AAPL", "--spot", "190", "--iv", "0.32", "--dte", "30",
        "--direction", "call", "--account", "50000", "--strike-increment", "5", "--full-kelly",
    ])
    captured = capsys.readouterr()
    assert "full-Kelly" in captured.out
