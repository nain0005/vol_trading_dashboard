"""Put-call parity arbitrage: conversions and reversals.

The classic, precise meaning of "ATM IV arbitrage" -- at the SAME strike
and expiration, the call's implied vol and the put's implied vol are tied
together by put-call parity, a no-arbitrage relationship that holds
independent of any vol model (Black-Scholes, whatever the market is
actually using, doesn't matter -- it's pure replication, not a pricing
model). If they diverge by more than the bid/ask spread can explain,
that's a real, quantifiable, tradeable edge: a conversion or reversal
locks it in. This module answers "how much, and which direction," priced
off REAL executable bid/ask quotes, not mid.

Put-call parity (continuous dividend yield q, continuous compounding r):
    C - P = S*e^(-qT) - K*e^(-rT)
The right-hand side is the "synthetic forward value" F -- what a
long-call/short-put combo is worth today purely from carrying the
underlying to expiration, with no assumption about volatility at all.

Two trades exploit a violation, and they're mirror images of each other:
  - CONVERSION: buy stock, buy the put, sell the call. A guaranteed K at
    expiration (call assigned above K, put exercised at/below K, either
    way you deliver stock for K). Profitable when the call is priced RICH
    relative to the put (more precisely: when call_bid - put_ask exceeds
    the theoretical forward value F).
  - REVERSAL: short stock, sell the put, buy the call. Mirror image --
    profitable when the PUT is priced rich relative to the call.

IMPORTANT -- why this is NOT textbook riskless arbitrage for real equity
options, read before trusting a positive edge number:
  - Robinhood equity options are AMERICAN-style. Put-call parity in its
    exact form above is a EUROPEAN result. The short option leg in either
    trade (the call in a conversion, the put in a reversal) can be
    exercised against you early, which can force you to unwind before
    expiration at a price you don't control -- this is the single biggest
    reason a real conversion/reversal desk's edge is smaller than the
    naive formula suggests, and it's most likely to bite around dividend
    ex-dates (early call assignment to capture the dividend).
  - Financing a short-stock leg (the reversal) has a real, variable
    borrow cost/rebate that isn't in `r` here -- hard-to-borrow names can
    erase an apparent reversal edge entirely.
  - `q` (the dividend yield used above) has to be the RIGHT forecast for
    the dividends actually paid over the option's life, not a trailing
    average -- a surprise dividend change moves F directly.
  - Real execution means crossing the spread on the STOCK leg too (not
    modeled here, only the two option legs use real bid/ask) plus
    commissions and margin/capital cost for the trade's holding period.
This function reports the edge as a matter of pricing arithmetic, priced
off real bid/ask so it's not mid-price noise -- it is not a claim that
executing it nets that amount risk-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from risk_tool.pricing import implied_volatility


def synthetic_forward_value(S: float, K: float, T: float, r: float, q: float) -> float:
    """F = S*e^(-qT) - K*e^(-rT) -- the theoretical C-P under put-call
    parity. Positive when the forward is above the strike (calls should
    be worth more than puts), negative when below."""
    return S * math.exp(-q * T) - K * math.exp(-r * T)


def conversion_edge(call_bid: float, put_ask: float, S: float, K: float, T: float, r: float, q: float) -> float:
    """PV edge (per share) of buy stock + buy put (at ask) + sell call (at
    bid), locking in K at expiration. Positive = profitable on paper --
    see module docstring for why "on paper" is doing real work here."""
    return call_bid - put_ask - synthetic_forward_value(S, K, T, r, q)


def reversal_edge(call_ask: float, put_bid: float, S: float, K: float, T: float, r: float, q: float) -> float:
    """PV edge (per share) of short stock + sell put (at bid) + buy call
    (at ask), the mirror image of conversion_edge. Positive = profitable
    on paper -- same caveats apply, plus stock borrow cost specifically."""
    return synthetic_forward_value(S, K, T, r, q) + put_bid - call_ask


@dataclass
class ParityArbitrage:
    strike: float
    call_iv: float | None  # solved from call MID price with this module's own r/q -- may differ slightly from the chain's own IV if it used different assumptions
    put_iv: float | None
    iv_spread: float | None  # call_iv - put_iv; the "ATM IV arbitrage" headline number, None if either side's IV didn't solve
    conversion_edge: float  # per share, PV; positive = a conversion (sell call, buy put+stock) looks profitable on paper
    reversal_edge: float  # per share, PV; positive = a reversal (buy call, sell put+stock) looks profitable on paper
    best_direction: str | None  # "conversion", "reversal", or None if neither edge is positive
    best_edge: float  # max(conversion_edge, reversal_edge, 0) -- 0 if neither side clears


def analyze_strike(
    call_bid: float, call_ask: float, put_bid: float, put_ask: float,
    S: float, K: float, T: float, r: float, q: float,
) -> ParityArbitrage | None:
    """The one function to call per strike. None if any of the four
    quotes is non-positive (a strike with no real two-sided market --
    can't price an executable edge off a quote that doesn't exist)."""
    if call_bid <= 0 or call_ask <= 0 or put_bid <= 0 or put_ask <= 0:
        return None
    if T <= 0:
        return None

    conv = conversion_edge(call_bid, put_ask, S, K, T, r, q)
    rev = reversal_edge(call_ask, put_bid, S, K, T, r, q)

    if conv > 0 and conv >= rev:
        best_direction, best_edge = "conversion", conv
    elif rev > 0:
        best_direction, best_edge = "reversal", rev
    else:
        best_direction, best_edge = None, 0.0

    call_mid = (call_bid + call_ask) / 2
    put_mid = (put_bid + put_ask) / 2
    call_iv = put_iv = None
    iv_spread = None
    try:
        call_iv = implied_volatility(call_mid, S, K, T, r, q, "call")
    except ValueError:
        pass
    try:
        put_iv = implied_volatility(put_mid, S, K, T, r, q, "put")
    except ValueError:
        pass
    if call_iv is not None and put_iv is not None:
        iv_spread = call_iv - put_iv

    return ParityArbitrage(
        strike=K, call_iv=call_iv, put_iv=put_iv, iv_spread=iv_spread,
        conversion_edge=conv, reversal_edge=rev, best_direction=best_direction, best_edge=best_edge,
    )
