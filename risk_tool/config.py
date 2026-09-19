"""Central configuration for every threshold used across the risk tool.

Nothing in pricing.py, greeks.py, strike_selection.py, sizing.py, or
risk_manager.py should hardcode a threshold — it comes from a RiskConfig
instance, so the whole system's discipline rules live in one auditable
place you can inspect, log, and change without touching model code.

Defaults marked "pick your own" don't have a universally correct value —
they depend on your account size and risk tolerance, not on any model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    # --- Market defaults (override per call whenever you have better inputs) ---
    risk_free_rate: float = 0.05  # annualized, continuously compounded
    dividend_yield: float = 0.0  # annualized continuous dividend yield q

    # --- Strike selection ---
    num_strikes_each_side: int = 4  # candidate strikes around ATM, per side
    min_risk_reward_ratio: float = 2.0  # strikes below this are flagged "poor setup"
    default_profit_target_pct: float = 1.00  # used as "profit_if_win" basis when no explicit target price is given

    # --- Position sizing (Kelly) ---
    use_half_kelly: bool = True  # edge estimates are noisy; half-Kelly is the safer default
    max_position_pct_of_account: float = 0.05  # HARD cap — always wins, see sizing.py

    # --- Short (credit) strike selection ---
    short_stop_loss_multiple: float = 2.0  # buy back a short option once its price grows to this multiple of the
    # premium you received (e.g. 2.0 = stop out once you'd pay back 2x what you collected). This is the FINITE,
    # rule-defined loss strike_selection.ShortStrikeEV's EV/Kelly math is built on — see that module's docstring for
    # why it is explicitly NOT the same thing as a short position's true theoretical max loss (unbounded for a
    # naked short call). Must be > 1.0.

    # --- Exit rules ---
    profit_target_pct: float = 1.00  # close at +100% gain on premium paid
    stop_loss_pct: float = -0.50  # close at -50% loss on premium paid
    delta_exit_threshold: float = 0.15  # close a long option if |delta| falls below this
    theta_exit_dte_threshold: int = 3  # DTE floor for the OTM theta-burn exit
    iv_crush_threshold_points: float = 0.20  # close if IV has dropped this many vol points since entry

    # --- Portfolio governors ("pick your own" — set to what your account can tolerate) ---
    daily_max_loss_pct: float = 0.03  # halt new entries once today's P&L hits -3% of account
    max_net_delta_dollars: float | None = None  # None = unset/unenforced until you set it
    max_net_vega_dollars: float | None = None  # None = unset/unenforced until you set it

    # --- Numerics ---
    binomial_steps: int = 500  # CRR tree steps for American pricing
    iv_solver_max_iter: int = 100
    iv_solver_tol: float = 1e-8
    iv_solver_bounds: tuple[float, float] = (1e-4, 5.0)  # Brent fallback search range for sigma


DEFAULT_CONFIG = RiskConfig()
