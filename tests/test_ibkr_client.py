"""MockIBKRClient tests — this is the fully testable half of ibkr_client.py.
The real IBKRClient requires a live TWS/Gateway connection and isn't
exercised here; its methods are thin and import ib_insync lazily precisely
so this module stays importable and testable without one."""
import pytest

from risk_tool.ibkr_client import LiveOptionPosition, LiveQuote, MockIBKRClient


def test_mock_client_connect_disconnect_tracks_state():
    client = MockIBKRClient()
    assert client.connected is False
    client.connect()
    assert client.connected is True
    client.disconnect()
    assert client.connected is False


def test_mock_client_returns_configured_quote():
    quote = LiveQuote(symbol="AAPL", bid=150.0, ask=150.2, last=150.1, mark=150.1)
    client = MockIBKRClient(quotes={"AAPL": quote})
    assert client.get_quote("AAPL") == quote


def test_mock_client_raises_for_unconfigured_symbol():
    client = MockIBKRClient()
    with pytest.raises(KeyError):
        client.get_quote("MSFT")


def test_mock_client_returns_configured_positions():
    pos = LiveOptionPosition(
        symbol="AAPL", option_type="call", strike=150.0, expiration="2026-08-21",
        quantity=2, avg_cost=3.50, current_price=4.00, delta=0.45, implied_vol=0.28,
    )
    client = MockIBKRClient(positions=[pos])
    assert client.get_option_positions() == [pos]


def test_mock_client_defaults_to_empty():
    client = MockIBKRClient()
    assert client.get_option_positions() == []
