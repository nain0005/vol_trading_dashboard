"""CLI entry point: manual-input strike selection, sizing, and entry/exit
levels. No live connection required — every input is a flag.

Example:
    python -m risk_tool.cli --ticker AAPL --spot 190 --iv 0.32 --dte 30 \\
        --direction call --account 50000 --strike-increment 5

With your own vol view for the realized-vs-implied edge check:
    python -m risk_tool.cli --ticker AAPL --spot 190 --iv 0.32 --dte 30 \\
        --direction call --account 50000 --strike-increment 5 --realized-vol 0.45
"""
from __future__ import annotations

import argparse
import sys

from risk_tool.config import DEFAULT_CONFIG, RiskConfig
from risk_tool.sizing import size_position
from risk_tool.strike_selection import compare_implied_vs_realized_ev, evaluate_strike, generate_candidate_strikes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Options risk-management and strike-selection tool.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--spot", type=float, required=True, help="Current underlying price")
    parser.add_argument("--iv", type=float, required=True, help="Market implied vol, e.g. 0.32 for 32%%")
    parser.add_argument("--dte", type=int, required=True, help="Days to expiry")
    parser.add_argument("--direction", choices=["call", "put"], required=True)
    parser.add_argument("--account", type=float, required=True, help="Account size in dollars")
    parser.add_argument("--strike-increment", type=float, required=True, help="Strike spacing for this underlying")
    parser.add_argument("--r", type=float, default=DEFAULT_CONFIG.risk_free_rate, help="Risk-free rate")
    parser.add_argument("--q", type=float, default=DEFAULT_CONFIG.dividend_yield, help="Dividend yield")
    parser.add_argument("--num-strikes", type=int, default=DEFAULT_CONFIG.num_strikes_each_side, help="Candidate strikes per side of ATM")
    parser.add_argument("--profit-target-pct", type=float, default=DEFAULT_CONFIG.default_profit_target_pct, help="e.g. 1.0 for +100%%")
    parser.add_argument("--max-position-pct", type=float, default=DEFAULT_CONFIG.max_position_pct_of_account, help="Hard cap, e.g. 0.05 for 5%% of account")
    parser.add_argument("--full-kelly", action="store_true", help="Use full Kelly instead of the half-Kelly default")
    parser.add_argument(
        "--realized-vol",
        type=float,
        default=None,
        help="Your own realized/forecast vol — enables the realized-vs-implied edge comparison and ranks by it",
    )
    return parser


def _print_strike_row(rank: int, strike_ev, extra: dict | None = None) -> None:
    flag = " [POOR SETUP]" if strike_ev.is_poor_setup else ""
    print(
        f"  #{rank}  K=${strike_ev.strike:<8.2f} premium=${strike_ev.premium:<7.2f} "
        f"P_win(risk-neutral)={strike_ev.p_win_risk_neutral:6.1%}  EV=${strike_ev.ev:+7.2f}  "
        f"R:R={strike_ev.risk_reward:5.2f}:1{flag}"
    )
    print(
        f"       delta={strike_ev.delta:+.3f}  theta/day={strike_ev.theta:+.3f}  vega={strike_ev.vega:.3f}"
    )
    if extra:
        for label, value in extra.items():
            print(f"       {label}: {value}")


def run(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = RiskConfig(
        risk_free_rate=args.r,
        dividend_yield=args.q,
        num_strikes_each_side=args.num_strikes,
        default_profit_target_pct=args.profit_target_pct,
        max_position_pct_of_account=args.max_position_pct,
        use_half_kelly=not args.full_kelly,
    )
    T = args.dte / 365.0
    strikes = generate_candidate_strikes(args.spot, args.strike_increment, config.num_strikes_each_side)

    print(f"\n{args.ticker} — spot ${args.spot:.2f}, {args.direction} thesis, {args.dte} DTE, IV {args.iv:.1%}")
    print(f"Config: profit target +{config.profit_target_pct:.0%}, stop {config.stop_loss_pct:.0%}, "
          f"{'half' if config.use_half_kelly else 'full'}-Kelly, hard cap {config.max_position_pct_of_account:.0%} of account\n")

    if args.realized_vol is not None:
        print(f"Realized-vs-implied edge check: market IV {args.iv:.1%} vs. your vol {args.realized_vol:.1%} "
              f"(spread {args.realized_vol - args.iv:+.1%})\n")
        comparisons = [
            compare_implied_vs_realized_ev(args.spot, K, T, config.risk_free_rate, config.dividend_yield, args.iv, args.realized_vol, args.direction, config)
            for K in strikes
        ]
        comparisons.sort(key=lambda c: c.edge_ev.ev, reverse=True)
        top = comparisons[:3]
        print("Top strikes by EDGE EV (your vol view, market's actual premium):")
        for i, c in enumerate(top, start=1):
            _print_strike_row(i, c.edge_ev, extra={
                "risk-neutral EV (market IV throughout)": f"${c.risk_neutral_ev.ev:+.2f}",
            })
        best_strike_ev = top[0].edge_ev
        best_p_win = best_strike_ev.p_win_risk_neutral
    else:
        ranked = sorted(
            (evaluate_strike(args.spot, K, T, config.risk_free_rate, config.dividend_yield, args.iv, args.direction, config) for K in strikes),
            key=lambda s: s.ev,
            reverse=True,
        )
        top = ranked[:3]
        print("Top strikes by risk-neutral EV (market IV throughout — see strike_selection.py docstring on what this does and doesn't mean):")
        for i, s in enumerate(top, start=1):
            _print_strike_row(i, s)
        best_strike_ev = top[0]
        best_p_win = best_strike_ev.p_win_risk_neutral

    print(f"\nRecommended strike: ${best_strike_ev.strike:.2f} {args.direction} @ ${best_strike_ev.premium:.2f}\n")

    sizing = size_position(
        account_size=args.account,
        premium_per_contract=best_strike_ev.premium,
        p_win=best_p_win,
        profit_if_win_per_contract=best_strike_ev.profit_if_win,
        config=config,
    )
    print("Position sizing:")
    print(f"  Kelly fraction (raw):        {sizing.kelly_fraction_raw:+.3f}")
    print(f"  Kelly fraction (used):       {sizing.kelly_fraction_used:.3f}")
    print(f"  Kelly-suggested size:        ${sizing.kelly_dollar_size:,.2f}")
    print(f"  Hard cap ({config.max_position_pct_of_account:.0%} of account):   ${sizing.hard_cap_dollar_size:,.2f}")
    print(f"  >>> Recommended size:        ${sizing.recommended_dollar_size:,.2f}  ({sizing.recommended_contracts} contracts)"
          f"{'  [capped by hard limit]' if sizing.capped_by_hard_limit else ''}")

    entry = best_strike_ev.premium
    profit_target_price = entry * (1 + config.profit_target_pct)
    stop_price = entry * (1 + config.stop_loss_pct)
    print("\nPre-committed entry/exit levels:")
    print(f"  Entry premium:               ${entry:.2f}")
    print(f"  Profit target (+{config.profit_target_pct:.0%}):       ${profit_target_price:.2f}")
    print(f"  Stop loss ({config.stop_loss_pct:.0%}):           ${stop_price:.2f}")
    print(f"  Delta exit if |delta| falls below {config.delta_exit_threshold:.2f}")
    print(f"  Theta exit if DTE <= {config.theta_exit_dte_threshold} and still OTM")
    print(f"  IV-crush exit if IV drops >= {config.iv_crush_threshold_points:.0%} points from entry ({args.iv:.1%})")
    print()
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
