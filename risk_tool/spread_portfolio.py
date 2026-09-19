"""Pick the best COMBINATION of option-spread candidates under a capital
budget (and, optionally, a max-total-risk cap) -- a 0/1 knapsack, not a
per-ticker ranking.

Why this is a different question from "what's the single best trade per
ticker" (spread_selection.compare_strategies): that function ranks each
ticker/strategy's candidates independently and hands you a list of
"bests" -- it never asks whether, say, spending your whole budget on
ticker A's #1 candidate beats splitting it between A's #2 and B's #1
instead. That's a genuine combinatorial trade-off (limited capital,
multiple competing uses for it), which is exactly the 0/1 knapsack shape:
each candidate is taken whole or not at all (you can't buy 0.6 of a
spread), each has a cost that consumes a shared budget, and the goal is
to maximize total value (here, edge_ev) subject to that budget.

EXACT vs. greedy: this module solves the knapsack EXACTLY via dynamic
programming over a DISCRETIZED capital budget (and, when a risk cap is
given, a second discretized risk axis) -- not a greedy "sort by edge/cost
ratio and take until the budget runs out" heuristic. Greedy knapsack is
provably NOT optimal in general (a classic counterexample: two items that
together fill the budget with more total value than the single
best-ratio item alone, which greedy would grab first and then have no
room left for the pair) and the candidate universe here is realistically
small -- a handful of tickers times up to 8 strategies times a handful of
top candidates each, "well under a few hundred" -- so exact DP is
genuinely affordable. The discretization (see MAX_CAPITAL_BUCKETS /
MAX_RISK_BUCKETS below) is what keeps the DP's state space bounded regardless
of how large a dollar budget you type in; it is NOT a concession to
avoid writing the exact algorithm, it's what makes "exact" tractable at
all for a knapsack (pseudo-polynomial in the capacity, not the dollar
value of the capacity) -- see the discretization note below for the
feasibility guarantee this buys you.

WHAT "cost" MEANS -- READ BEFORE BUILDING CANDIDATES:
This module treats `cost` as "capital required to open one contract of
this position" and simply enforces sum(cost of selected) <= capital_budget.
It does NOT know or care whether a candidate is a net debit or a net
credit -- that modeling decision belongs to the caller, and getting it
wrong silently breaks the whole optimization:
  - A DEBIT spread's capital requirement is the net premium you pay --
    use net_cost (per share, from spread_selection.SpreadCandidate) times
    the contract multiplier.
  - A CREDIT spread's capital requirement is NOT its (negative) net_cost
    -- you RECEIVE money opening it. What actually ties up capital is the
    broker's margin/collateral requirement, which for a defined-risk
    credit spread is the position's max_loss magnitude (the width the
    broker could be on the hook for, minus the credit already collected
    is already netted into max_loss by option_strategy.analyze_strategy).
    Passing a negative "cost" here would make a credit spread look like
    it GENERATES budget rather than consumes it, which this module
    explicitly refuses (see validation below) precisely because that's
    not how real margin works and would let the optimizer "launder"
    unlimited free capital through credit spreads.
`from_spread_candidates()` below applies exactly this rule when building
a PortfolioCandidate from a spread_selection.SpreadCandidate, so prefer
it over hand-rolling the cost field yourself.

DISCRETIZATION AND THE FEASIBILITY GUARANTEE:
Both the capital budget and (if given) the risk cap are divided into
buckets, and each candidate's real dollar cost/risk is rounded UP
(ceiling) to the nearest whole bucket before the DP runs. Rounding up,
not to-nearest or down, is the deliberate choice that makes the result's
real (non-discretized) total cost and total risk provably never exceed
what you asked for -- every selected candidate's bucket already covers
its true cost, so the sum of true costs is bounded by the sum of bucket
costs, which the DP keeps within budget. That guarantee holds at ANY
resolution; what resolution buys you is closeness to the true continuous
optimum -- coarse buckets can make the DP miss a combination that (in
real dollars) fits exactly at the budget boundary, because several
items' individual roundings can stack up past the discretized cap even
though the true sum doesn't. This module picks buckets ADAPTIVELY: as
fine as whole US cents (the actual granularity option premiums are
already quoted in, so cent buckets lose nothing real) up to a bucket-count
ceiling chosen to keep the DP fast; only budgets/caps large enough to
exceed that ceiling coarsen beyond a cent, and even then the feasibility
guarantee above still holds exactly -- only how close the result gets to
the true continuous optimum degrades. See MAX_CAPITAL_BUCKETS (no risk
cap) and MAX_CAPITAL_BUCKETS_WITH_RISK / MAX_RISK_BUCKETS (risk cap
active, where the DP table is the PRODUCT of both axes' bucket counts, so
each axis gets a smaller ceiling to keep that product tractable).

TIES: when two combinations achieve the same total edge, this module
deterministically keeps whichever was found first in candidate list
order (a later candidate only replaces an earlier choice when it
STRICTLY improves total edge) -- pass candidates in whatever order you'd
prefer to win ties (e.g. more liquid / tighter-spread candidates first).

UNDEFINED RISK: a candidate with max_loss=None (unlimited-risk structure,
e.g. a naked short leg slipping through) can't be checked against a
max_total_risk cap by definition, so whenever a risk cap is requested,
such candidates are excluded up front rather than silently treated as
zero risk -- see `excluded` in the result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CONTRACT_MULTIPLIER = 100  # standard equity option contract size -- matches the "multiply by 100 for per-contract dollars" convention used throughout dashboard.py

CENT = 0.01  # option premiums are already quoted to the cent -- bucketing this fine loses no real precision

# Bucket-count ceilings -- see "DISCRETIZATION AND THE FEASIBILITY GUARANTEE" above. Below these
# ceilings, resolution is exactly one cent (no rounding loss beyond what option pricing already has);
# above them, buckets coarsen just enough to keep the DP table size, and therefore runtime, bounded.
MAX_CAPITAL_BUCKETS = 500_000  # 1D case (no risk cap): exact-to-the-cent for budgets up to $5,000
MAX_CAPITAL_BUCKETS_WITH_RISK = 1500  # 2D case: table size is this * MAX_RISK_BUCKETS, so each axis gets a smaller ceiling
MAX_RISK_BUCKETS = 1500


@dataclass
class PortfolioCandidate:
    """One indivisible (0/1 -- take it whole or not at all) option spread
    candidate for the optimizer. `cost` and `max_loss` are PER CONTRACT
    (i.e. already multiplied by CONTRACT_MULTIPLIER if built by hand),
    matching what `capital_budget`/`max_total_risk` are denominated in."""

    label: str
    cost: float  # capital required to open one contract -- see module docstring for debit vs. credit spread semantics; must be >= 0
    edge_ev: float  # per-contract expected-value edge (the quantity maximized); can be negative -- the optimizer just won't pick it
    max_loss: float | None = None  # per-contract max dollar loss if defined risk; None = undefined/unlimited


@dataclass
class PortfolioSelection:
    selected: list[PortfolioCandidate]
    total_cost: float
    total_edge: float
    total_max_loss: float  # sum of max_loss over SELECTED candidates only (0.0 if none selected); undefined-risk candidates can only appear here when no max_total_risk cap was requested, in which case this sum is not a real risk bound -- see total_max_loss_is_bounded
    total_max_loss_is_bounded: bool  # False if any selected candidate has max_loss=None (only possible when max_total_risk was None) -- total_max_loss is a partial sum, not a real cap, in that case
    capital_budget: float
    max_total_risk: float | None
    capital_bucket_size: float  # dollars per capital DP bucket -- the resolution of the "exact within discretization" guarantee
    risk_bucket_size: float | None
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (label, reason) for candidates dropped before the DP even ran -- e.g. cost alone exceeds the budget, or undefined risk with a risk cap active


def from_spread_candidates(candidates, label_prefix: str, contract_multiplier: int = CONTRACT_MULTIPLIER) -> list[PortfolioCandidate]:
    """Build PortfolioCandidates from spread_selection.SpreadCandidate
    objects, applying the debit-vs-credit capital rule from the module
    docstring: a net DEBIT (net_cost > 0) ties up its premium; a net
    CREDIT (net_cost <= 0) ties up margin equal to max_loss's magnitude,
    not the (negative) premium received. `label_prefix` is typically the
    ticker (and strategy, if you're mixing strategies) so candidates from
    different tickers/strategies stay distinguishable in the result.

    Candidates with edge_ev=None (no vol view was supplied upstream) are
    skipped -- there's nothing for the optimizer to maximize without an
    edge number, and silently treating None as 0 would make "no view"
    indistinguishable from "a genuinely breakeven trade."
    """
    out = []
    for i, c in enumerate(candidates):
        if c.edge_ev is None:
            continue
        if c.net_cost > 0:
            cost = c.net_cost * contract_multiplier
        else:
            # Net credit: capital at risk is the margin requirement, which
            # for a defined-risk credit spread is max_loss's magnitude
            # (already net of the credit received, from analyze_strategy).
            # An undefined-risk credit candidate (max_loss=None) has no
            # sound capital number here -- skip it rather than guess.
            if c.max_loss is None:
                continue
            cost = abs(c.max_loss) * contract_multiplier
        max_loss = abs(c.max_loss) * contract_multiplier if c.max_loss is not None else None
        out.append(
            PortfolioCandidate(
                label=f"{label_prefix} {c.strategy} {c.strike_label()}",
                cost=cost,
                edge_ev=c.edge_ev * contract_multiplier,
                max_loss=max_loss,
            )
        )
    return out


def _bucket_index(value: float, bucket_size: float, n_buckets: int) -> int:
    """Ceiling-rounded bucket index for `value` in [0, n_buckets], or
    n_buckets + 1 (a sentinel meaning "doesn't fit at all") if value is
    positive but bucket_size is 0 (a zero or negative budget/cap can only
    afford exactly-zero-cost items)."""
    if value <= 0:
        return 0
    if bucket_size <= 0:
        return n_buckets + 1
    return min(n_buckets + 1, math.ceil(value / bucket_size))


def _adaptive_buckets(amount: float, max_buckets: int) -> tuple[int, float]:
    """Number of buckets and dollars-per-bucket for `amount`, as fine as
    exactly one cent when that stays within max_buckets, coarsening only
    when the amount is large enough to need more buckets than the ceiling
    allows -- see "DISCRETIZATION AND THE FEASIBILITY GUARANTEE" in the
    module docstring. amount <= 0 still returns at least 1 bucket (so the
    DP arrays are constructible) with bucket_size 0 -- _bucket_index
    handles that degenerate case (only exactly-zero-cost items fit)."""
    if amount <= 0:
        return 1, 0.0
    exact_buckets = math.ceil(amount / CENT)
    n_buckets = min(exact_buckets, max_buckets)
    return n_buckets, amount / n_buckets


def optimize_portfolio(
    candidates: list[PortfolioCandidate],
    capital_budget: float,
    max_total_risk: float | None = None,
    max_capital_buckets: int | None = None,
    max_risk_buckets: int = MAX_RISK_BUCKETS,
) -> PortfolioSelection:
    """The one function to call from outside this module. Exact 0/1
    knapsack DP over a discretized capital budget (and, if max_total_risk
    is given, jointly over a second discretized risk axis) -- see module
    docstring for what "exact" and "discretized" mean here and why greedy
    isn't used.

    max_capital_buckets defaults to MAX_CAPITAL_BUCKETS when no risk cap is
    requested, or MAX_CAPITAL_BUCKETS_WITH_RISK when one is (the 2D DP
    table is max_capital_buckets * max_risk_buckets, so both default
    smaller together to keep that product fast) -- override either to
    trade resolution for speed or vice versa.

    Raises ValueError if any candidate has cost < 0 (see module docstring
    -- that's almost always a credit spread's raw net_cost used by
    mistake instead of its margin requirement)."""
    if max_capital_buckets is None:
        max_capital_buckets = MAX_CAPITAL_BUCKETS if max_total_risk is None else MAX_CAPITAL_BUCKETS_WITH_RISK

    for c in candidates:
        if c.cost < 0:
            raise ValueError(
                f"PortfolioCandidate {c.label!r} has negative cost ({c.cost!r}) -- this module requires "
                "capital cost, not net premium; a credit spread's cost is its margin requirement "
                "(≈ max_loss magnitude), not its negative net premium. See module docstring."
            )

    excluded: list[tuple[str, str]] = []
    eligible: list[PortfolioCandidate] = []
    for c in candidates:
        if max_total_risk is not None and c.max_loss is None:
            excluded.append((c.label, "undefined/unlimited risk -- can't be checked against a max-total-risk cap"))
            continue
        if c.cost > max(capital_budget, 0.0):
            excluded.append((c.label, "cost alone exceeds the capital budget"))
            continue
        if max_total_risk is not None and abs(c.max_loss) > max(max_total_risk, 0.0):
            excluded.append((c.label, "max_loss alone exceeds the max-total-risk cap"))
            continue
        eligible.append(c)

    capital_buckets, capital_bucket_size = _adaptive_buckets(max(capital_budget, 0.0), max_capital_buckets)
    if max_total_risk is not None:
        risk_buckets, risk_bucket_size = _adaptive_buckets(max(max_total_risk, 0.0), max_risk_buckets)
    else:
        risk_buckets, risk_bucket_size = None, None

    empty_result = PortfolioSelection(
        selected=[], total_cost=0.0, total_edge=0.0, total_max_loss=0.0, total_max_loss_is_bounded=True,
        capital_budget=capital_budget, max_total_risk=max_total_risk,
        capital_bucket_size=capital_bucket_size, risk_bucket_size=risk_bucket_size, excluded=excluded,
    )
    if not eligible:
        return empty_result

    cost_buckets = [_bucket_index(c.cost, capital_bucket_size, capital_buckets) for c in eligible]

    if max_total_risk is None:
        selection = _knapsack_1d(eligible, cost_buckets, capital_buckets)
    else:
        risk_buckets_idx = [_bucket_index(abs(c.max_loss), risk_bucket_size, risk_buckets) for c in eligible]
        selection = _knapsack_2d(eligible, cost_buckets, risk_buckets_idx, capital_buckets, risk_buckets)

    if not selection:
        return empty_result

    total_cost = sum(c.cost for c in selection)
    total_edge = sum(c.edge_ev for c in selection)
    total_max_loss_is_bounded = all(c.max_loss is not None for c in selection)
    total_max_loss = sum(abs(c.max_loss) for c in selection if c.max_loss is not None)

    return PortfolioSelection(
        selected=selection, total_cost=total_cost, total_edge=total_edge, total_max_loss=total_max_loss,
        total_max_loss_is_bounded=total_max_loss_is_bounded, capital_budget=capital_budget, max_total_risk=max_total_risk,
        capital_bucket_size=capital_bucket_size, risk_bucket_size=risk_bucket_size, excluded=excluded,
    )


def _knapsack_1d(eligible: list[PortfolioCandidate], cost_buckets: list[int], capital_buckets: int) -> list[PortfolioCandidate]:
    n = len(eligible)
    dp = np.zeros(capital_buckets + 1)
    choice = np.zeros((n, capital_buckets + 1), dtype=bool)

    for i, cb in enumerate(cost_buckets):
        prev = dp.copy()
        if cb <= capital_buckets:
            width = capital_buckets + 1 - cb
            shifted = prev[:width] + eligible[i].edge_ev
            region = prev[cb:]
            take = shifted > region
            dp[cb:] = np.where(take, shifted, region)
            choice[i, cb:] = take

    best_bucket = int(np.argmax(dp))
    selected = []
    bucket = best_bucket
    for i in range(n - 1, -1, -1):
        if choice[i, bucket]:
            selected.append(eligible[i])
            bucket -= cost_buckets[i]
    selected.reverse()
    return selected


def _knapsack_2d(
    eligible: list[PortfolioCandidate], cost_buckets: list[int], risk_bucket_idx: list[int], capital_buckets: int, risk_buckets: int,
) -> list[PortfolioCandidate]:
    n = len(eligible)
    dp = np.zeros((capital_buckets + 1, risk_buckets + 1))
    choice = np.zeros((n, capital_buckets + 1, risk_buckets + 1), dtype=bool)

    for i in range(n):
        cb, rb = cost_buckets[i], risk_bucket_idx[i]
        prev = dp.copy()
        if cb <= capital_buckets and rb <= risk_buckets:
            cwidth = capital_buckets + 1 - cb
            rwidth = risk_buckets + 1 - rb
            shifted = prev[:cwidth, :rwidth] + eligible[i].edge_ev
            region = prev[cb:, rb:]
            take = shifted > region
            dp[cb:, rb:] = np.where(take, shifted, region)
            choice[i, cb:, rb:] = take

    best_flat = int(np.argmax(dp))
    best_c, best_r = np.unravel_index(best_flat, dp.shape)
    selected = []
    c_idx, r_idx = int(best_c), int(best_r)
    for i in range(n - 1, -1, -1):
        if choice[i, c_idx, r_idx]:
            selected.append(eligible[i])
            c_idx -= cost_buckets[i]
            r_idx -= risk_bucket_idx[i]
    selected.reverse()
    return selected
