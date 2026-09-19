"""Correctness tests for risk_tool.parity_arbitrage.

Checks the put-call parity arithmetic against an independently-computed
theoretical forward value (math.exp, not a re-use of the function under
test), verifies conversion/reversal edges are correct mirror images of
each other, and that a genuinely fairly-priced two-sided market (built
from actual Black-Scholes prices split into a symmetric bid/ask) shows
no free lunch -- crossing a real spread should never look profitable.
"""
import math

import pytest

from risk_tool.pricing import black_scholes_price
from risk_tool.parity_arbitrage import analyze_strike, conversion_edge, reversal_edge, synthetic_forward_value

S, K, T, r, q = 100.0, 100.0, 0.25, 0.05, 0.0


def test_synthetic_forward_value_matches_independent_calc():
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert synthetic_forward_value(S, K, T, r, q) == pytest.approx(expected)


def test_synthetic_forward_value_with_dividends_and_different_strike():
    S2, K2, T2, r2, q2 = 250.0, 240.0, 0.5, 0.04, 0.02
    expected = S2 * math.exp(-q2 * T2) - K2 * math.exp(-r2 * T2)
    assert synthetic_forward_value(S2, K2, T2, r2, q2) == pytest.approx(expected)


def test_conversion_edge_positive_when_call_priced_rich_relative_to_put():
    # C - P (mid) = 6.10 - 4.60 = 1.50, deliberately above the theoretical
    # forward value -- the call is rich relative to the put.
    F = synthetic_forward_value(S, K, T, r, q)
    call_bid, call_ask = 6.00, 6.20
    put_bid, put_ask = 4.50, 4.70

    conv = conversion_edge(call_bid, put_ask, S, K, T, r, q)
    rev = reversal_edge(call_ask, put_bid, S, K, T, r, q)

    assert conv == pytest.approx(call_bid - put_ask - F)
    assert rev == pytest.approx(F + put_bid - call_ask)
    assert conv > 0  # the rich-call direction should show a positive conversion edge
    assert rev < 0  # and a negative (unprofitable) reversal edge -- mirror image


def test_reversal_edge_positive_when_put_priced_rich_relative_to_call():
    F = synthetic_forward_value(S, K, T, r, q)
    # Now flip it: put priced rich relative to the call.
    call_bid, call_ask = 4.50, 4.70
    put_bid, put_ask = 6.00, 6.20

    conv = conversion_edge(call_bid, put_ask, S, K, T, r, q)
    rev = reversal_edge(call_ask, put_bid, S, K, T, r, q)
    assert rev > 0
    assert conv < 0


def test_analyze_strike_picks_the_profitable_direction_and_sizes_it():
    F = synthetic_forward_value(S, K, T, r, q)
    call_bid, call_ask = 6.00, 6.20
    put_bid, put_ask = 4.50, 4.70
    result = analyze_strike(call_bid, call_ask, put_bid, put_ask, S, K, T, r, q)
    assert result is not None
    assert result.best_direction == "conversion"
    assert result.best_edge == pytest.approx(call_bid - put_ask - F)
    assert result.best_edge > 0


def test_analyze_strike_iv_spread_sign_matches_mispricing_direction():
    # Same rich-call setup -- call should solve to a HIGHER implied vol
    # than the put (it's priced richer), so iv_spread = call_iv - put_iv
    # should be positive, consistent with the positive conversion edge.
    result = analyze_strike(6.00, 6.20, 4.50, 4.70, S, K, T, r, q)
    assert result is not None
    assert result.call_iv is not None and result.put_iv is not None
    assert result.iv_spread == pytest.approx(result.call_iv - result.put_iv)
    assert result.iv_spread > 0


def test_fairly_priced_symmetric_spread_market_has_no_positive_edge():
    # Both call and put priced off the SAME Black-Scholes vol (parity holds
    # exactly at mid), then split into a symmetric +/-1% bid/ask spread --
    # a genuinely fair, two-sided market. Crossing a real spread should
    # never show a profitable conversion OR reversal; that's the "no free
    # lunch" sanity check this module's honesty depends on.
    sigma = 0.25
    call_mid = black_scholes_price(S, K, T, r, q, sigma, "call")
    put_mid = black_scholes_price(S, K, T, r, q, sigma, "put")
    call_bid, call_ask = call_mid * 0.99, call_mid * 1.01
    put_bid, put_ask = put_mid * 0.99, put_mid * 1.01

    result = analyze_strike(call_bid, call_ask, put_bid, put_ask, S, K, T, r, q)
    assert result is not None
    assert result.conversion_edge < 0
    assert result.reversal_edge < 0
    assert result.best_direction is None
    assert result.best_edge == 0.0


def test_fairly_priced_market_has_near_zero_iv_spread():
    sigma = 0.30
    call_mid = black_scholes_price(S, K, T, r, q, sigma, "call")
    put_mid = black_scholes_price(S, K, T, r, q, sigma, "put")
    result = analyze_strike(call_mid * 0.999, call_mid * 1.001, put_mid * 0.999, put_mid * 1.001, S, K, T, r, q)
    assert result is not None
    assert result.iv_spread == pytest.approx(0.0, abs=1e-3)


def test_analyze_strike_returns_none_for_a_missing_quote():
    assert analyze_strike(0.0, 6.20, 4.50, 4.70, S, K, T, r, q) is None
    assert analyze_strike(6.00, 6.20, -1.0, 4.70, S, K, T, r, q) is None


def test_analyze_strike_returns_none_at_or_past_expiration():
    assert analyze_strike(6.00, 6.20, 4.50, 4.70, S, K, 0.0, r, q) is None
