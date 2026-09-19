"""Correctness tests for risk_tool.spread_portfolio.

Uses small, exactly-representable candidate sets (costs/edges chosen so
the discretized DP buckets land on exact boundaries at the default
resolution) so results can be verified against a hand-computed or
brute-force-enumerated optimum, not just "it ran."
"""
import itertools

import pytest

from risk_tool.spread_portfolio import (
    PortfolioCandidate,
    from_spread_candidates,
    optimize_portfolio,
)
from risk_tool.spread_selection import ChainLeg, SpreadCandidate


def _cand(label, cost, edge_ev, max_loss=None):
    return PortfolioCandidate(label=label, cost=cost, edge_ev=edge_ev, max_loss=max_loss)


def test_classic_knapsack_example_matches_textbook_optimum():
    # Textbook 0/1 knapsack: weights [10, 20, 30], values [60, 100, 120],
    # capacity 50 -- the well-known optimal answer is items 2+3 (value
    # 220, weight exactly 50), NOT item 1 despite its best value/cost
    # ratio (6.0 vs 5.0 and 4.0) -- a pure ratio-greedy would grab item 1
    # first and then be unable to fit both 2 and 3 (30+20=50 already
    # uses the whole budget with nothing left for item 1's 10).
    candidates = [_cand("A", 10, 60), _cand("B", 20, 100), _cand("C", 30, 120)]
    result = optimize_portfolio(candidates, capital_budget=50)
    assert {c.label for c in result.selected} == {"B", "C"}
    assert result.total_edge == pytest.approx(220.0)
    assert result.total_cost == pytest.approx(50.0)
    assert result.excluded == []


def test_brute_force_cross_check_on_a_larger_candidate_set():
    # Independent verification: for 8 candidates, exhaustively enumerate
    # every subset (2**8 = 256, trivial), keep the best-edge subset that
    # fits the budget, and confirm the DP finds the SAME total edge.
    candidates = [
        _cand("c0", 12, 31), _cand("c1", 7, 18), _cand("c2", 25, 55),
        _cand("c3", 3, 9), _cand("c4", 18, 40), _cand("c5", 9, 22),
        _cand("c6", 14, 33), _cand("c7", 22, 47),
    ]
    budget = 45.0

    best_edge = 0.0
    for r in range(len(candidates) + 1):
        for combo in itertools.combinations(candidates, r):
            cost = sum(c.cost for c in combo)
            if cost <= budget:
                best_edge = max(best_edge, sum(c.edge_ev for c in combo))

    result = optimize_portfolio(candidates, capital_budget=budget)
    assert result.total_edge == pytest.approx(best_edge)
    assert result.total_cost <= budget


def test_risk_cap_excludes_a_single_candidate_whose_own_risk_exceeds_it():
    x = _cand("X", 10, 40, max_loss=5)
    y = _cand("Y", 10, 60, max_loss=100)
    # Budget easily fits both (cost 20); Y's OWN max_loss (100) alone
    # already exceeds the risk cap (10), so Y must be excluded outright
    # regardless of what else is picked -- not just "excluded from the
    # optimal combo," excluded from consideration entirely.
    result = optimize_portfolio([x, y], capital_budget=20, max_total_risk=10)
    assert [c.label for c in result.selected] == ["X"]
    assert result.total_edge == pytest.approx(40.0)
    assert result.total_max_loss == pytest.approx(5.0)
    assert ("Y", "max_loss alone exceeds the max-total-risk cap") in result.excluded


def test_risk_cap_forces_a_genuine_joint_tradeoff_not_just_per_item_filtering():
    # Both P and Q are individually well within both budget and risk cap,
    # and BOTH fit the capital budget together -- but their COMBINED risk
    # (8+8=16) exceeds the cap (12), so only one can be taken. Q has the
    # higher edge, so Q alone is optimal -- this can only be found by
    # jointly tracking cost AND risk together, not by filtering candidates
    # one at a time.
    p = _cand("P", 10, 30, max_loss=8)
    q = _cand("Q", 10, 40, max_loss=8)
    result = optimize_portfolio([p, q], capital_budget=20, max_total_risk=12)
    assert [c.label for c in result.selected] == ["Q"]
    assert result.total_edge == pytest.approx(40.0)
    assert result.total_max_loss == pytest.approx(8.0)


def test_ties_are_broken_deterministically_by_candidate_order():
    a = _cand("first", 10, 50)
    b = _cand("second", 10, 50)  # identical cost and edge to "first"
    result = optimize_portfolio([a, b], capital_budget=10)
    assert [c.label for c in result.selected] == ["first"]


def test_negative_edge_candidate_is_never_selected_even_with_ample_budget():
    good = _cand("good", 10, 25)
    bad = _cand("bad", 10, -5)
    result = optimize_portfolio([good, bad], capital_budget=1000)
    assert [c.label for c in result.selected] == ["good"]
    assert result.total_edge == pytest.approx(25.0)


def test_zero_eligible_candidates_returns_empty_selection():
    result = optimize_portfolio([], capital_budget=1000)
    assert result.selected == []
    assert result.total_edge == 0.0
    assert result.total_cost == 0.0
    assert result.total_max_loss_is_bounded is True


def test_budget_too_small_for_even_the_cheapest_candidate():
    result = optimize_portfolio([_cand("only", 100, 500)], capital_budget=10)
    assert result.selected == []
    assert ("only", "cost alone exceeds the capital budget") in result.excluded


def test_negative_cost_candidate_raises_valueerror():
    with pytest.raises(ValueError):
        optimize_portfolio([_cand("bad_cost", -5, 10)], capital_budget=100)


def test_undefined_risk_candidate_excluded_only_when_risk_cap_is_active():
    undefined = _cand("undefined_risk", 10, 20, max_loss=None)
    # No risk cap: undefined-risk candidates are perfectly fine to include.
    result_no_cap = optimize_portfolio([undefined], capital_budget=100)
    assert [c.label for c in result_no_cap.selected] == ["undefined_risk"]
    assert result_no_cap.total_max_loss_is_bounded is False

    # A risk cap is requested: can't check an undefined risk against it,
    # so it's excluded up front rather than silently treated as zero risk.
    result_with_cap = optimize_portfolio([undefined], capital_budget=100, max_total_risk=50)
    assert result_with_cap.selected == []
    assert any(label == "undefined_risk" for label, _ in result_with_cap.excluded)


def test_from_spread_candidates_prices_debit_at_net_cost_and_credit_at_margin():
    long_leg = ("long", ChainLeg(strike=100.0, option_type="call", iv=0.3, mid=2.0))
    short_leg = ("short", ChainLeg(strike=105.0, option_type="call", iv=0.28, mid=0.5))

    debit = SpreadCandidate(
        strategy="bull_call_spread", legs=[long_leg, short_leg], net_cost=1.5, max_profit=3.5,
        max_loss=-1.5, breakevens=[101.5], risk_reward=2.33, prob_profit=0.4,
        my_vol_fair_value=2.0, edge_ev=0.5, is_poor_setup=False, avg_spread_pct=2.0,
    )
    credit = SpreadCandidate(
        strategy="bear_call_spread", legs=[short_leg, long_leg], net_cost=-1.5, max_profit=1.5,
        max_loss=-3.5, breakevens=[106.5], risk_reward=0.43, prob_profit=0.6,
        my_vol_fair_value=-1.0, edge_ev=0.5, is_poor_setup=False, avg_spread_pct=2.0,
    )
    credit_undefined_risk = SpreadCandidate(
        strategy="bear_call_spread", legs=[short_leg, long_leg], net_cost=-1.5, max_profit=1.5,
        max_loss=None, breakevens=[106.5], risk_reward=None, prob_profit=0.6,
        my_vol_fair_value=-1.0, edge_ev=0.5, is_poor_setup=False, avg_spread_pct=2.0,
    )
    no_edge = SpreadCandidate(
        strategy="bull_call_spread", legs=[long_leg, short_leg], net_cost=1.5, max_profit=3.5,
        max_loss=-1.5, breakevens=[101.5], risk_reward=2.33, prob_profit=0.4,
        my_vol_fair_value=None, edge_ev=None, is_poor_setup=False, avg_spread_pct=2.0,
    )

    out = from_spread_candidates([debit, credit, credit_undefined_risk, no_edge], label_prefix="AAPL")
    assert len(out) == 2  # undefined-risk credit and no-edge candidates are both skipped

    debit_out = next(c for c in out if "bull_call_spread" in c.label)
    assert debit_out.cost == pytest.approx(1.5 * 100)  # net debit * contract multiplier
    assert debit_out.max_loss == pytest.approx(1.5 * 100)
    assert debit_out.edge_ev == pytest.approx(0.5 * 100)

    credit_out = next(c for c in out if "bear_call_spread" in c.label)
    assert credit_out.cost == pytest.approx(3.5 * 100)  # margin = max_loss magnitude, NOT the negative net credit
    assert credit_out.max_loss == pytest.approx(3.5 * 100)


def test_result_never_exceeds_the_real_budget_or_risk_cap_despite_bucket_rounding():
    # Costs deliberately chosen to NOT divide evenly into the default
    # bucket size, so ceiling rounding actually kicks in -- the real
    # (undiscretized) totals must still respect both caps exactly.
    candidates = [
        _cand("odd1", 7.13, 19.0, max_loss=6.9),
        _cand("odd2", 11.77, 25.0, max_loss=10.1),
        _cand("odd3", 3.02, 8.0, max_loss=2.5),
    ]
    result = optimize_portfolio(candidates, capital_budget=17.0, max_total_risk=15.0)
    assert result.total_cost <= 17.0 + 1e-9
    assert result.total_max_loss <= 15.0 + 1e-9
