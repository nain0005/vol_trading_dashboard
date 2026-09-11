"""Correctness tests for risk_tool.spread_selection.

Builds a small synthetic chain (mid prices computed via black_scholes_price
at a known sigma, so expected payoff numbers can be hand-verified) and
checks: exact net-debit/max-profit/max-loss/breakeven arithmetic for a
bear put spread and a bull put spread, that the blended-IV probability
matches direct itm_probability calls at the same breakevens, that edge_ev
correctly compares market price against a supplied "my vol", and that
find_best_spreads ranks by edge EV when a vol view is supplied.
"""
import pandas as pd
import pytest

from risk_tool.config import RiskConfig
from risk_tool.pricing import black_scholes_price, itm_probability
from risk_tool.spread_selection import (
    evaluate_candidate,
    find_best_spreads,
)

S, T, r, q, MARKET_IV = 100.0, 0.25, 0.03, 0.0, 0.30
STRIKES = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]


def _make_chain(sigma: float = MARKET_IV) -> pd.DataFrame:
    rows = []
    for k in STRIKES:
        for opt_type in ("call", "put"):
            price = black_scholes_price(S, k, T, r, q, sigma, opt_type)
            rows.append({"strike": k, "type": opt_type, "iv": sigma, "mid": price})
    return pd.DataFrame(rows)


def test_bear_put_spread_net_debit_and_bounds_match_manual_calc():
    chain = _make_chain()
    long_premium = black_scholes_price(S, 105.0, T, r, q, MARKET_IV, "put")
    short_premium = black_scholes_price(S, 95.0, T, r, q, MARKET_IV, "put")
    expected_debit = long_premium - short_premium

    cand = evaluate_candidate(
        "bear_put_spread",
        [("long", "put", 105.0), ("short", "put", 95.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    assert cand.net_cost == pytest.approx(expected_debit)
    assert cand.max_loss == pytest.approx(-expected_debit)  # capped at the debit paid
    assert cand.max_profit == pytest.approx((105.0 - 95.0) - expected_debit)  # width - debit
    assert cand.breakevens == pytest.approx([105.0 - expected_debit])
    assert cand.risk_reward == pytest.approx(cand.max_profit / abs(cand.max_loss))


def test_bull_put_spread_is_a_net_credit_with_capped_loss():
    chain = _make_chain()
    short_premium = black_scholes_price(S, 105.0, T, r, q, MARKET_IV, "put")
    long_premium = black_scholes_price(S, 95.0, T, r, q, MARKET_IV, "put")
    expected_credit = short_premium - long_premium  # net_debit convention: negative = credit received

    cand = evaluate_candidate(
        "bull_put_spread",
        [("short", "put", 105.0), ("long", "put", 95.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    assert cand.net_cost == pytest.approx(-expected_credit)
    assert cand.max_profit == pytest.approx(expected_credit)  # keep the whole credit if spot stays above breakeven
    assert cand.max_loss == pytest.approx(-((105.0 - 95.0) - expected_credit))
    assert cand.breakevens == pytest.approx([105.0 - expected_credit])


def test_bear_put_spread_prob_profit_matches_direct_itm_probability_at_breakeven():
    chain = _make_chain()
    cand = evaluate_candidate(
        "bear_put_spread",
        [("long", "put", 105.0), ("short", "put", 95.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    breakeven = cand.breakevens[0]
    expected = itm_probability(S, breakeven, T, r, q, MARKET_IV, "put")
    assert cand.prob_profit == pytest.approx(expected)


def test_straddle_has_two_breakevens_and_prob_profit_sums_both_tails():
    chain = _make_chain()
    cand = evaluate_candidate("straddle", [("long", "call", 100.0), ("long", "put", 100.0)], chain, S, T, r, q)
    assert cand is not None
    assert len(cand.breakevens) == 2
    lo, hi = min(cand.breakevens), max(cand.breakevens)
    expected = itm_probability(S, lo, T, r, q, MARKET_IV, "put") + itm_probability(S, hi, T, r, q, MARKET_IV, "call")
    assert cand.prob_profit == pytest.approx(expected)
    assert cand.max_profit is None  # long call leg is unbounded upside


def test_edge_ev_is_positive_when_my_vol_exceeds_market_iv_for_a_long_structure():
    chain = _make_chain()  # priced at MARKET_IV = 0.30
    higher_vol = 0.45
    cand = evaluate_candidate(
        "straddle", [("long", "call", 100.0), ("long", "put", 100.0)], chain, S, T, r, q,
        my_vol=higher_vol,
    )
    assert cand is not None
    # A long straddle repriced at a HIGHER vol than it was bought at should
    # look cheap relative to that view — positive edge.
    assert cand.edge_ev > 0
    assert cand.my_vol_fair_value == pytest.approx(
        black_scholes_price(S, 100.0, T, r, q, higher_vol, "call") + black_scholes_price(S, 100.0, T, r, q, higher_vol, "put")
    )


def test_evaluate_candidate_returns_none_for_an_unquoted_strike():
    chain = _make_chain()
    cand = evaluate_candidate("bear_put_spread", [("long", "put", 999.0), ("short", "put", 95.0)], chain, S, T, r, q)
    assert cand is None


def test_find_best_spreads_ranks_by_edge_ev_when_my_vol_given():
    chain = _make_chain()
    results = find_best_spreads(
        "bull_put_spread", chain, S, T, r, q, strike_increment=5.0, num_each_side=3, my_vol=0.15,
    )
    assert results
    edge_evs = [c.edge_ev for c in results]
    assert edge_evs == sorted(edge_evs, reverse=True)


def test_find_best_spreads_falls_back_to_risk_reward_with_no_vol_view():
    chain = _make_chain()
    results = find_best_spreads("bear_put_spread", chain, S, T, r, q, strike_increment=5.0, num_each_side=3)
    assert results
    assert all(c.edge_ev is None for c in results)
    rrs = [c.risk_reward for c in results if c.risk_reward is not None]
    assert rrs == sorted(rrs, reverse=True)


def test_find_best_spreads_rejects_unknown_strategy():
    chain = _make_chain()
    with pytest.raises(ValueError):
        find_best_spreads("iron_condor", chain, S, T, r, q, strike_increment=5.0)


def test_min_risk_reward_flags_poor_setup():
    chain = _make_chain()
    strict_config = RiskConfig(min_risk_reward_ratio=100.0)  # nothing will clear this bar
    cand = evaluate_candidate(
        "bear_put_spread", [("long", "put", 105.0), ("short", "put", 95.0)], chain, S, T, r, q, config=strict_config,
    )
    assert cand.is_poor_setup is True
