"""Exit-rule engine: pre-committed, mechanical exit conditions evaluated
against every open position on every update, plus portfolio-level
governors that halt new entries.

The entire point of this module is to replace gut-feel exits with rules you
committed to before you had a position-specific reason to rationalize
holding. Every rule reports WHY it fired, with the actual numbers, so the
output is auditable rather than a bare yes/no.
"""
from __future__ import annotations

from dataclasses import dataclass

from risk_tool.config import DEFAULT_CONFIG, RiskConfig


@dataclass
class Position:
    symbol: str
    option_type: str  # "call" or "put"
    strike: float
    spot: float
    dte: int
    quantity: int  # positive = long contracts; this module reasons about long-premium positions
    entry_premium: float  # premium paid per contract at entry
    current_premium: float  # current mark price per contract
    entry_iv: float  # IV at entry (annualized, e.g. 0.35)
    current_iv: float  # current IV
    delta: float  # current option delta (signed: positive for calls, negative for puts)


@dataclass
class ExitSignal:
    rule: str
    triggered: bool
    reason: str
    numbers: dict


def pct_pnl(position: Position) -> float:
    """(current - entry) / entry, on premium per contract."""
    if position.entry_premium <= 0:
        raise ValueError("entry_premium must be positive")
    return (position.current_premium - position.entry_premium) / position.entry_premium


def check_profit_target(position: Position, config: RiskConfig = DEFAULT_CONFIG) -> ExitSignal:
    pnl = pct_pnl(position)
    triggered = pnl >= config.profit_target_pct
    reason = (
        f"P&L {pnl:+.1%} has reached the +{config.profit_target_pct:.0%} profit target "
        f"(entry ${position.entry_premium:.2f} -> current ${position.current_premium:.2f})."
        if triggered
        else f"P&L {pnl:+.1%} is below the +{config.profit_target_pct:.0%} profit target."
    )
    return ExitSignal("profit_target", triggered, reason, {"pct_pnl": pnl, "threshold": config.profit_target_pct})


def check_stop_loss(position: Position, config: RiskConfig = DEFAULT_CONFIG) -> ExitSignal:
    pnl = pct_pnl(position)
    triggered = pnl <= config.stop_loss_pct
    reason = (
        f"P&L {pnl:+.1%} has breached the {config.stop_loss_pct:.0%} stop loss "
        f"(entry ${position.entry_premium:.2f} -> current ${position.current_premium:.2f})."
        if triggered
        else f"P&L {pnl:+.1%} is above the {config.stop_loss_pct:.0%} stop loss."
    )
    return ExitSignal("stop_loss", triggered, reason, {"pct_pnl": pnl, "threshold": config.stop_loss_pct})


def check_delta_exit(position: Position, config: RiskConfig = DEFAULT_CONFIG) -> ExitSignal:
    """For a long option, delta decaying toward zero means the position has
    become a low-probability lottery ticket rather than the directional
    exposure you originally bought — this closes it out regardless of
    current P&L."""
    abs_delta = abs(position.delta)
    triggered = abs_delta < config.delta_exit_threshold
    reason = (
        f"|delta| {abs_delta:.3f} has fallen below the {config.delta_exit_threshold:.2f} exit threshold — "
        "this option barely resembles the original directional bet anymore."
        if triggered
        else f"|delta| {abs_delta:.3f} is still above the {config.delta_exit_threshold:.2f} exit threshold."
    )
    return ExitSignal("delta_exit", triggered, reason, {"abs_delta": abs_delta, "threshold": config.delta_exit_threshold})


def check_theta_exit(position: Position, config: RiskConfig = DEFAULT_CONFIG) -> ExitSignal:
    """Close OTM positions once DTE is too low for the thesis to plausibly
    still play out — theta burn accelerates fastest in the final days, and
    an OTM option this close to expiry is fighting decay with no help from
    intrinsic value."""
    is_otm = (position.option_type == "call" and position.spot < position.strike) or (
        position.option_type == "put" and position.spot > position.strike
    )
    triggered = position.dte <= config.theta_exit_dte_threshold and is_otm
    reason = (
        f"DTE {position.dte} <= {config.theta_exit_dte_threshold} and position is OTM "
        f"(spot ${position.spot:.2f} vs strike ${position.strike:.2f}) — theta burn with no help from intrinsic value."
        if triggered
        else f"DTE {position.dte}, OTM={is_otm} — theta exit not triggered."
    )
    return ExitSignal(
        "theta_exit", triggered, reason, {"dte": position.dte, "threshold": config.theta_exit_dte_threshold, "is_otm": is_otm}
    )


def check_iv_crush_exit(position: Position, config: RiskConfig = DEFAULT_CONFIG) -> ExitSignal:
    """If IV has dropped sharply since entry (e.g. post-earnings crush), the
    vega tailwind you may have been counting on is gone even if the
    underlying moved your way — this exit is about vol risk specifically,
    independent of the delta/theta/P&L rules above."""
    iv_drop = position.entry_iv - position.current_iv
    triggered = iv_drop >= config.iv_crush_threshold_points
    reason = (
        f"IV dropped {iv_drop:.1%} since entry (entry {position.entry_iv:.1%} -> current {position.current_iv:.1%}), "
        f"past the {config.iv_crush_threshold_points:.0%}-point crush threshold."
        if triggered
        else f"IV drop {iv_drop:.1%} is within the {config.iv_crush_threshold_points:.0%}-point crush threshold."
    )
    return ExitSignal(
        "iv_crush_exit", triggered, reason, {"iv_drop": iv_drop, "threshold": config.iv_crush_threshold_points}
    )


def evaluate_all_rules(position: Position, config: RiskConfig = DEFAULT_CONFIG) -> list[ExitSignal]:
    return [
        check_profit_target(position, config),
        check_stop_loss(position, config),
        check_delta_exit(position, config),
        check_theta_exit(position, config),
        check_iv_crush_exit(position, config),
    ]


def any_exit_triggered(position: Position, config: RiskConfig = DEFAULT_CONFIG) -> bool:
    return any(signal.triggered for signal in evaluate_all_rules(position, config))


@dataclass
class PortfolioState:
    account_size: float
    daily_pnl: float  # today's realized + unrealized P&L in dollars; negative = a loss
    net_delta_dollars: float  # sum over positions of delta * spot * quantity * multiplier
    net_vega_dollars: float  # sum over positions of vega * quantity * multiplier


@dataclass
class GovernorResult:
    halted: bool
    reasons: list[str]


def check_portfolio_governors(state: PortfolioState, config: RiskConfig = DEFAULT_CONFIG) -> GovernorResult:
    """Portfolio-level circuit breakers, independent of any single
    position's exit rules: a daily max-loss halt on new entries, and
    optional net-Greek exposure caps (unset by default — see config.py;
    there's no universal correct value, it's your own risk tolerance)."""
    reasons: list[str] = []

    daily_loss_limit = -abs(config.daily_max_loss_pct) * state.account_size
    if state.daily_pnl <= daily_loss_limit:
        reasons.append(
            f"Daily P&L ${state.daily_pnl:,.2f} has breached the max-loss governor "
            f"(${daily_loss_limit:,.2f}, {config.daily_max_loss_pct:.0%} of account) — halt new entries for today."
        )

    if config.max_net_delta_dollars is not None and abs(state.net_delta_dollars) > config.max_net_delta_dollars:
        reasons.append(
            f"Net delta exposure ${state.net_delta_dollars:,.2f} exceeds the "
            f"${config.max_net_delta_dollars:,.2f} cap — reduce directional exposure before adding more."
        )

    if config.max_net_vega_dollars is not None and abs(state.net_vega_dollars) > config.max_net_vega_dollars:
        reasons.append(
            f"Net vega exposure ${state.net_vega_dollars:,.2f} exceeds the "
            f"${config.max_net_vega_dollars:,.2f} cap — reduce vol exposure before adding more."
        )

    return GovernorResult(halted=len(reasons) > 0, reasons=reasons)
