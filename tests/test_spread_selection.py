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
from risk_tool.option_strategy import OptionLeg, analyze_strategy
from risk_tool.pricing import black_scholes_price, itm_probability
from risk_tool.spread_selection import (
    STRATEGIES,
    compare_strategies,
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
        find_best_spreads("condor_time_machine", chain, S, T, r, q, strike_increment=5.0)


def test_prob_profit_is_none_not_a_crash_when_net_debit_exactly_equals_spread_width():
    # Found via streamlit.testing.v1.AppTest exercising the live Spread
    # Selector form against demo-mode AAPL data (2026-09-18 expiration):
    # when net_debit exactly equals the spread width, max_profit is
    # exactly 0 at spot=$0 (both puts fully in the money: payoff there is
    # width - net_debit = 0) -- option_strategy.analyze_strategy correctly
    # treats that as a zero-crossing and reports breakevens=[0.0, ...],
    # which is mathematically real but not a valid Black-Scholes strike
    # (compute_d1_d2 requires S and K strictly positive). This crashed
    # with "S and K must be positive" inside itm_probability before the
    # <= 0 guard existed. Reproduced here with the exact real premiums
    # that triggered it, not a hand-constructed approximation.
    chain = pd.DataFrame([
        {"strike": 100.0, "type": "put", "iv": 0.198160, "mid": 12.180},
        {"strike": 95.0, "type": "put", "iv": 0.196956, "mid": 7.180},
    ])
    spot = 87.75595402838555
    cand = evaluate_candidate("bear_put_spread", [("long", "put", 100.0), ("short", "put", 95.0)], chain, spot, 5 / 365.0, 0.05, 0.0)
    assert cand is not None
    assert cand.max_profit == pytest.approx(0.0, abs=1e-6)  # confirms this test hits the exact degenerate case
    assert cand.breakevens[0] == pytest.approx(0.0, abs=1e-6)
    assert cand.prob_profit is None  # not a crash, not a silently wrong number


def test_min_risk_reward_flags_poor_setup():
    chain = _make_chain()
    strict_config = RiskConfig(min_risk_reward_ratio=100.0)  # nothing will clear this bar
    cand = evaluate_candidate(
        "bear_put_spread", [("long", "put", 105.0), ("short", "put", 95.0)], chain, S, T, r, q, config=strict_config,
    )
    assert cand.is_poor_setup is True


# --- Call verticals (bull call spread, bear call spread) ---


def test_bull_call_spread_net_debit_and_bounds_match_manual_calc():
    chain = _make_chain()
    long_premium = black_scholes_price(S, 95.0, T, r, q, MARKET_IV, "call")
    short_premium = black_scholes_price(S, 105.0, T, r, q, MARKET_IV, "call")
    expected_debit = long_premium - short_premium  # long the cheaper (lower-strike) call, sell the pricier one? -- for calls, LOWER strike is worth MORE, so this is a genuine debit

    cand = evaluate_candidate(
        "bull_call_spread",
        [("long", "call", 95.0), ("short", "call", 105.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    assert cand.net_cost == pytest.approx(expected_debit)
    assert cand.net_cost > 0  # genuinely a debit: long call strike is cheaper priced but still a net outlay
    assert cand.max_loss == pytest.approx(-expected_debit)  # capped at the debit paid
    assert cand.max_profit == pytest.approx((105.0 - 95.0) - expected_debit)  # width - debit
    assert cand.breakevens == pytest.approx([95.0 + expected_debit])
    assert cand.risk_reward == pytest.approx(cand.max_profit / abs(cand.max_loss))


def test_bull_call_spread_prob_profit_matches_direct_itm_probability_at_breakeven():
    chain = _make_chain()
    cand = evaluate_candidate("bull_call_spread", [("long", "call", 95.0), ("short", "call", 105.0)], chain, S, T, r, q)
    assert cand is not None
    breakeven = cand.breakevens[0]
    expected = itm_probability(S, breakeven, T, r, q, MARKET_IV, "call")  # bullish: profits ABOVE the breakeven
    assert cand.prob_profit == pytest.approx(expected)


def test_bear_call_spread_is_a_net_credit_with_capped_loss():
    chain = _make_chain()
    short_premium = black_scholes_price(S, 95.0, T, r, q, MARKET_IV, "call")
    long_premium = black_scholes_price(S, 105.0, T, r, q, MARKET_IV, "call")
    expected_credit = short_premium - long_premium  # sell the pricier lower-strike call, buy the cheaper higher-strike call

    cand = evaluate_candidate(
        "bear_call_spread",
        [("short", "call", 95.0), ("long", "call", 105.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    assert cand.net_cost == pytest.approx(-expected_credit)
    assert cand.max_profit == pytest.approx(expected_credit)
    assert cand.max_loss == pytest.approx(-((105.0 - 95.0) - expected_credit))
    assert cand.breakevens == pytest.approx([95.0 + expected_credit])


def test_bear_call_spread_prob_profit_matches_direct_itm_probability_at_breakeven():
    chain = _make_chain()
    cand = evaluate_candidate("bear_call_spread", [("short", "call", 95.0), ("long", "call", 105.0)], chain, S, T, r, q)
    assert cand is not None
    breakeven = cand.breakevens[0]
    expected = itm_probability(S, breakeven, T, r, q, MARKET_IV, "put")  # bearish: profits BELOW the breakeven
    assert cand.prob_profit == pytest.approx(expected)


# --- Iron condor ---


def test_iron_condor_max_profit_loss_breakevens_match_manual_calc():
    # Symmetric wings (5-wide both sides) keep the closed-form clean:
    # max profit = net credit, max loss = -(wing width - credit),
    # breakevens = short strike -+ credit -- verified independently against
    # option_strategy.analyze_strategy on a hand-built leg list before being
    # written as the expected values here.
    chain = _make_chain()
    put_long_p = black_scholes_price(S, 85.0, T, r, q, MARKET_IV, "put")
    put_short_p = black_scholes_price(S, 90.0, T, r, q, MARKET_IV, "put")
    call_short_p = black_scholes_price(S, 110.0, T, r, q, MARKET_IV, "call")
    call_long_p = black_scholes_price(S, 115.0, T, r, q, MARKET_IV, "call")
    expected_credit = (put_short_p - put_long_p) + (call_short_p - call_long_p)

    cand = evaluate_candidate(
        "iron_condor",
        [("long", "put", 85.0), ("short", "put", 90.0), ("short", "call", 110.0), ("long", "call", 115.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    assert cand.net_cost == pytest.approx(-expected_credit)
    assert cand.max_profit == pytest.approx(expected_credit)
    assert cand.max_loss == pytest.approx(-(5.0 - expected_credit))
    assert cand.breakevens == pytest.approx(sorted([90.0 - expected_credit, 110.0 + expected_credit]))
    assert cand.risk_reward == pytest.approx(cand.max_profit / abs(cand.max_loss))


def test_iron_condor_prob_profit_is_the_mirror_image_of_strangle_between_breakevens():
    chain = _make_chain()
    cand = evaluate_candidate(
        "iron_condor",
        [("long", "put", 85.0), ("short", "put", 90.0), ("short", "call", 110.0), ("long", "call", 115.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    lo, hi = min(cand.breakevens), max(cand.breakevens)
    # Profit is BETWEEN the breakevens (short vol) -- P(lo<S<hi) = P(S>lo) - P(S>hi).
    expected = itm_probability(S, lo, T, r, q, MARKET_IV, "call") - itm_probability(S, hi, T, r, q, MARKET_IV, "call")
    assert cand.prob_profit == pytest.approx(expected)
    assert 0.0 < cand.prob_profit < 1.0


def test_iron_condor_asymmetric_wings_matches_independent_analyze_strategy_call():
    # Edge case: wings of very different widths (15-wide puts, 5-wide
    # calls) -- risk is NOT symmetric here (all the loss risk sits on the
    # put side), so the clean "credit / (width-credit)" formula from the
    # symmetric test above does NOT apply. Recomputes expected values by
    # calling option_strategy.analyze_strategy directly on a hand-built
    # OptionLeg list -- an independent path from evaluate_candidate's
    # chain-lookup-based one -- rather than asserting a possibly-wrong
    # hand-derived formula.
    chain = _make_chain()
    legs = [
        OptionLeg("put", 85.0, black_scholes_price(S, 85.0, T, r, q, MARKET_IV, "put"), 1, 1.0),
        OptionLeg("put", 100.0, black_scholes_price(S, 100.0, T, r, q, MARKET_IV, "put"), -1, 1.0),
        OptionLeg("call", 100.0, black_scholes_price(S, 100.0, T, r, q, MARKET_IV, "call"), -1, 1.0),
        OptionLeg("call", 105.0, black_scholes_price(S, 105.0, T, r, q, MARKET_IV, "call"), 1, 1.0),
    ]
    expected = analyze_strategy(legs)

    cand = evaluate_candidate(
        "iron_condor",
        [("long", "put", 85.0), ("short", "put", 100.0), ("short", "call", 100.0), ("long", "call", 105.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    assert cand.net_cost == pytest.approx(expected.net_debit)
    assert cand.max_profit == pytest.approx(expected.max_profit)
    assert cand.max_loss == pytest.approx(expected.max_loss)
    assert cand.breakevens == pytest.approx(expected.breakevens)
    # All the defined risk sits on the wide put side here -- max loss should
    # be a much bigger multiple of the credit than a symmetric condor's.
    assert abs(cand.max_loss) > cand.max_profit


def test_find_best_spreads_iron_condor_only_returns_otm_otm_combos():
    chain = _make_chain()
    results = find_best_spreads("iron_condor", chain, S, T, r, q, strike_increment=5.0, num_each_side=3, top_n=50)
    assert results
    for c in results:
        legs_by_type_side = {(side, cl.option_type): cl.strike for side, cl in c.legs}
        put_long, put_short = legs_by_type_side[("long", "put")], legs_by_type_side[("short", "put")]
        call_short, call_long = legs_by_type_side[("short", "call")], legs_by_type_side[("long", "call")]
        # Both spreads properly OTM/OTM, and the two spreads don't overlap:
        # put strikes strictly below spot, call strikes strictly above it.
        assert put_long < put_short < S < call_short < call_long


# --- Iron butterfly ---


def test_iron_butterfly_max_profit_loss_breakevens_match_manual_calc():
    chain = _make_chain()
    put_wing_p = black_scholes_price(S, 90.0, T, r, q, MARKET_IV, "put")
    short_put_p = black_scholes_price(S, 100.0, T, r, q, MARKET_IV, "put")
    short_call_p = black_scholes_price(S, 100.0, T, r, q, MARKET_IV, "call")
    call_wing_p = black_scholes_price(S, 110.0, T, r, q, MARKET_IV, "call")
    expected_credit = (short_put_p - put_wing_p) + (short_call_p - call_wing_p)

    cand = evaluate_candidate(
        "iron_butterfly",
        [("long", "put", 90.0), ("short", "put", 100.0), ("short", "call", 100.0), ("long", "call", 110.0)],
        chain, S, T, r, q,
    )
    assert cand is not None
    assert cand.net_cost == pytest.approx(-expected_credit)
    assert cand.max_profit == pytest.approx(expected_credit)
    assert cand.max_loss == pytest.approx(-(10.0 - expected_credit))  # symmetric 10-wide wings here
    assert cand.breakevens == pytest.approx(sorted([100.0 - expected_credit, 100.0 + expected_credit]))


def test_find_best_spreads_iron_butterfly_body_is_always_the_nearest_strike_to_spot():
    chain = _make_chain()  # STRIKES = [85, 90, 95, 100, 105, 110, 115], spot=100 -> ATM strike is exactly 100
    results = find_best_spreads("iron_butterfly", chain, S, T, r, q, strike_increment=5.0, num_each_side=3, top_n=50)
    assert results
    for c in results:
        short_strikes = {cl.strike for side, cl in c.legs if side == "short"}
        assert short_strikes == {100.0}  # both short legs (the "body") pinned to the ATM strike


# --- Cross-strategy comparison ---


def test_compare_strategies_returns_one_best_candidate_per_strategy():
    chain = _make_chain()
    result = compare_strategies(chain, S, T, r, q, strike_increment=5.0, num_each_side=3)
    assert set(result.keys()) == set(STRATEGIES)
    for strat, cand in result.items():
        assert cand is not None  # this synthetic chain has strikes on both sides of spot for every strategy
        assert cand.strategy == strat


def test_compare_strategies_handles_a_strategy_with_zero_valid_candidates():
    # A chain with strikes only ABOVE spot starves any structure that
    # specifically requires a strike BELOW spot (strangle needs an OTM put
    # below spot; iron_condor needs OTM puts below spot; iron_butterfly's
    # ATM body would have no lower wing to pair with) while structures that
    # don't care about strike-vs-spot placement (verticals, which just need
    # two strikes of the same type) still work fine -- exactly the "one
    # strategy has zero candidates, another has several" case that
    # shouldn't crash the comparison.
    rows = []
    for k in [105.0, 110.0, 115.0, 120.0]:
        for opt_type in ("call", "put"):
            price = black_scholes_price(S, k, T, r, q, MARKET_IV, opt_type)
            rows.append({"strike": k, "type": opt_type, "iv": MARKET_IV, "mid": price})
    lopsided_chain = pd.DataFrame(rows)

    result = compare_strategies(lopsided_chain, S, T, r, q, strike_increment=5.0, num_each_side=4)
    assert result["strangle"] is None
    assert result["iron_condor"] is None
    assert result["iron_butterfly"] is None
    assert result["bull_call_spread"] is not None
    assert result["bear_call_spread"] is not None
    assert result["bear_put_spread"] is not None  # doesn't care about spot placement, just needs two put strikes
