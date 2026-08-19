"""Kelly Criterion position sizing, with a mandatory hard cap.

Kelly answers "how much of my capital should I risk on a bet with edge
(p, b)", ASSUMING p and b are known exactly. In reality p comes from a
model (risk-neutral N(d2), or your own vol forecast) and b comes from your
own profit-target/stop-loss rule — both are estimates, not certainties.
Full Kelly is famously aggressive and compounds an overestimated edge into
real drawdown fast; half-Kelly (the default here) trades some long-run
growth rate for a large cut in variance. The hard %-of-account cap exists
for the same reason, and it always wins regardless of what Kelly says:
size_position() is the only function you should call from outside this
module, and the cap is applied as its last step, unconditionally.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk_tool.config import DEFAULT_CONFIG, RiskConfig


def kelly_fraction(p_win: float, b: float) -> float:
    """f* = (p*b - q) / b

    p = probability of winning, q = 1-p, b = odds received (profit per $1
    risked, if you win). Can come out negative — that means the bet has no
    edge at these odds and Kelly says "don't take it," not "go short."
    Callers should floor the result at 0 before sizing anything.
    """
    if not (0.0 <= p_win <= 1.0):
        raise ValueError(f"p_win must be in [0, 1], got {p_win}")
    if b <= 0:
        raise ValueError(f"b (odds received) must be positive, got {b}")
    q = 1.0 - p_win
    return (p_win * b - q) / b


def half_kelly_fraction(p_win: float, b: float) -> float:
    return kelly_fraction(p_win, b) / 2.0


@dataclass
class PositionSizeResult:
    kelly_fraction_raw: float  # full Kelly f*, can be negative (no edge)
    kelly_fraction_used: float  # floored at 0, then halved if config.use_half_kelly
    kelly_dollar_size: float
    hard_cap_dollar_size: float
    recommended_dollar_size: float  # min(kelly_dollar_size, hard_cap_dollar_size) — the hard cap always wins ties
    recommended_contracts: int
    capped_by_hard_limit: bool  # True if the hard cap is what actually bound the size (not Kelly)


def size_position(
    account_size: float,
    premium_per_contract: float,
    p_win: float,
    profit_if_win_per_contract: float,
    config: RiskConfig = DEFAULT_CONFIG,
    contract_multiplier: int = 100,
) -> PositionSizeResult:
    """The one function to call from outside this module.

    b (odds received) is derived from your own profit-target/stop-loss
    rule: profit_if_win_per_contract / premium_per_contract, i.e. "if I win,
    how many dollars do I make per dollar risked, given I'll take profit at
    my own rule rather than hold to expiry." This keeps sizing consistent
    with the same discipline risk_manager.py enforces on exit.

    The hard cap (config.max_position_pct_of_account of account_size) is
    computed unconditionally and the final recommendation is always
    min(kelly_size, hard_cap) — there is no parameter here that can make
    Kelly override the cap.
    """
    if account_size <= 0:
        raise ValueError("account_size must be positive")
    if premium_per_contract <= 0:
        raise ValueError("premium_per_contract must be positive")

    b = profit_if_win_per_contract / premium_per_contract
    f_raw = kelly_fraction(p_win, b)
    f_used = max(f_raw, 0.0)
    if config.use_half_kelly:
        f_used /= 2.0

    kelly_dollars = f_used * account_size
    hard_cap_dollars = config.max_position_pct_of_account * account_size
    recommended_dollars = min(kelly_dollars, hard_cap_dollars)

    contract_cost = premium_per_contract * contract_multiplier
    recommended_contracts = int(recommended_dollars // contract_cost) if contract_cost > 0 else 0

    return PositionSizeResult(
        kelly_fraction_raw=f_raw,
        kelly_fraction_used=f_used,
        kelly_dollar_size=kelly_dollars,
        hard_cap_dollar_size=hard_cap_dollars,
        recommended_dollar_size=recommended_dollars,
        recommended_contracts=recommended_contracts,
        capped_by_hard_limit=bool(hard_cap_dollars < kelly_dollars),
    )
