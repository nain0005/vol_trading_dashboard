"""Thin wrapper around ib_insync for live IBKR quotes and option position
monitoring.

OPTIONAL — nothing in pricing.py, greeks.py, sizing.py, risk_manager.py, or
strike_selection.py imports this module or requires a live connection.
Every core model runs entirely on manual inputs. This module exists purely
to feed those models live numbers instead of typing them in by hand, and to
let risk_manager watch your real IBKR positions if you want that.

MockIBKRClient implements the same interface with deterministic fake data,
so the CLI/dashboard code path can be exercised and tested without TWS or
IB Gateway running at all.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveQuote:
    symbol: str
    bid: float
    ask: float
    last: float
    mark: float  # (bid + ask) / 2, same convention used elsewhere in this project


@dataclass
class LiveOptionPosition:
    symbol: str
    option_type: str  # "call" or "put"
    strike: float
    expiration: str  # "YYYY-MM-DD"
    quantity: int
    avg_cost: float  # per-contract entry cost
    current_price: float  # per-contract mark; 0.0 if not available (see IBKRClient docstring)
    delta: float | None
    implied_vol: float | None


class MockIBKRClient:
    """Deterministic fake data for development and tests. Configure it with
    whatever quotes/positions your test needs; it never touches a network."""

    def __init__(
        self,
        quotes: dict[str, LiveQuote] | None = None,
        positions: list[LiveOptionPosition] | None = None,
    ):
        self._quotes = quotes or {}
        self._positions = positions or []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_quote(self, symbol: str) -> LiveQuote:
        if symbol not in self._quotes:
            raise KeyError(f"No mock quote configured for {symbol!r}")
        return self._quotes[symbol]

    def get_option_positions(self) -> list[LiveOptionPosition]:
        return list(self._positions)


class IBKRClient:
    """Real ib_insync-backed client. Requires TWS or IB Gateway running
    locally with the API enabled — default port 7497 is IB's paper-trading
    port; use 7496 for a live account, at your own risk.

    ib_insync is imported lazily inside each method rather than at module
    load time, so importing risk_tool.ibkr_client never fails just because
    you haven't set up IBKR — it only matters once you actually call
    connect().
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def connect(self) -> None:
        from ib_insync import IB

        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()

    def _require_connection(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("Not connected to IBKR — call connect() first (requires TWS/IB Gateway running).")

    def get_quote(self, symbol: str) -> LiveQuote:
        from ib_insync import Stock

        self._require_connection()
        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        ticker = self._ib.reqMktData(contract, "", False, False)
        self._ib.sleep(1)  # ib_insync needs a beat for the snapshot fields to populate
        bid, ask, last = ticker.bid, ticker.ask, ticker.last
        mark = (bid + ask) / 2 if bid and ask else last
        return LiveQuote(symbol=symbol, bid=bid, ask=ask, last=last, mark=mark)

    def get_option_positions(self) -> list[LiveOptionPosition]:
        """NOTE: current_price/delta/implied_vol are left unset (None/0.0)
        here — filling them in requires a live market-data subscription per
        option contract (reqMktData with the options generic tick list),
        which is a meaningfully bigger integration than position listing
        and isn't wired up yet. Feed those fields from the dashboard's
        existing Robinhood option-chain data instead in the meantime, or
        extend this method if you have the IBKR options market data
        subscription needed to fill them in properly.
        """
        self._require_connection()
        positions: list[LiveOptionPosition] = []
        for pos in self._ib.positions():
            contract = pos.contract
            if contract.secType != "OPT":
                continue
            positions.append(
                LiveOptionPosition(
                    symbol=contract.symbol,
                    option_type="call" if contract.right == "C" else "put",
                    strike=contract.strike,
                    expiration=contract.lastTradeDateOrContractMonth,
                    quantity=int(pos.position),
                    avg_cost=pos.avgCost / 100,  # ib_insync reports option avgCost scaled by the 100x multiplier
                    current_price=0.0,
                    delta=None,
                    implied_vol=None,
                )
            )
        return positions
