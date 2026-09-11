# risk_tool

An options risk-management and strike-selection tool. It sizes positions,
ranks strikes by expected value, and enforces pre-committed exits. **It
does not predict direction.** You bring the thesis (ticker, direction,
timeframe); this tool prices it, sizes it, and holds you to your own exit
rules.

Every module is pure Python with no UI/network dependency — `pricing.py`,
`greeks.py`, `strike_selection.py`, `sizing.py`, `risk_manager.py`,
`realized_vol.py`, `hedge.py`, `option_strategy.py`, `spread_selection.py`,
`options_lab.py`, and `portfolio_risk.py` all run on manual inputs and are
independently unit tested (see `../tests/`). `ibkr_client.py` and the
Robinhood-backed dashboard tabs are optional data sources layered on top.

## Quickstart

```bash
# from the vol-dashboard/ directory, with .venv activated
python3 -m pytest tests/ -v          # 213 tests as of this writing

python3 -m risk_tool.cli \
  --ticker AAPL --spot 190 --iv 0.32 --dte 30 \
  --direction call --account 50000 --strike-increment 5

# add your own vol view for the realized-vs-implied edge check:
python3 -m risk_tool.cli \
  --ticker AAPL --spot 190 --iv 0.32 --dte 30 \
  --direction call --account 50000 --strike-increment 5 --realized-vol 0.45
```

Or use the **Risk Tool** tab in the Streamlit dashboard (`streamlit run
dashboard.py`) — same models, plus it can auto-fetch spot/IV from your live
Robinhood connection and monitor your actual open positions against the
exit rules in real time.

## Models, one by one

### 1. Pricing & Greeks (`pricing.py`, `greeks.py`)

Black-Scholes-Merton with continuous dividend yield `q`:

```
d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ√T)
d2 = d1 - σ√T
Call = S·e^-qT·N(d1) - K·e^-rT·N(d2)
Put  = K·e^-rT·N(-d2) - S·e^-qT·N(-d1)
```

`N(d2)` is the risk-neutral probability of finishing in-the-money; the
module exposes raw `N(d1)`/`N(d2)` directly. Delta is `e^-qT·N(d1)` for a
call — **not** bare `N(d1)` unless `q=0`, a common source of quietly-wrong
deltas on dividend payers.

American options are priced with a **Cox-Ross-Rubinstein binomial tree**
(`binomial_crr_price`), which checks early exercise at every node. Two
facts anchor its tests: an American call on a non-dividend stock must
price identically to European (never optimal to exercise early), while an
American put can be worth strictly more (early exercise has real value
there).

**Implied vol** (`implied_volatility`) uses Newton-Raphson first
(quadratic convergence via vega), falling back to Brent's method
(`scipy.optimize.brentq`) whenever vega is too small for Newton-Raphson to
be trusted — deep ITM/OTM contracts, mainly.

**Assumptions and where they break**: constant `σ` and `r` over the
option's life (real vol is not constant — that's the entire reason
`realized_vol.py` and the GARCH forecast exist), continuous trading with no
transaction costs or bid/ask spread, and log-normal terminal prices (no
jumps — earnings gaps and other jump risk are not modeled at all).

### 2. Strike selection & EV (`strike_selection.py`)

**Read the module docstring before trusting any number here** — the short
version:

- `EV = P_win · profit_if_win − P_lose · premium`, where `P_win = N(d2)`
  (or `N(-d2)` for puts).
- `profit_if_win` is your **profit-target rule's** payout
  (`premium × profit_target_pct`, default +100%), not the option's
  theoretical (unbounded, for a call) payoff at expiry. This is
  deliberate: it keeps the EV consistent with how `risk_manager.py`
  actually makes you exit, instead of an idealized hold-to-expiry payoff
  nobody following the discipline rules ever takes.
- Consequently, `risk_reward` here is relative to **your own target**, not
  the option's max theoretical gain — don't read it as "this call can only
  make 2x."
- `P_win` is risk-neutral: it's what the option's price implies about the
  odds of finishing ITM if the stock drifted at the risk-free rate and the
  given `σ` were the true vol. **It is not a forecast.** Pricing built on
  market IV is, by construction, close to a zero-edge bet before
  frictions — ranking strikes by risk-neutral EV mostly tells you about
  cost-efficiency and convexity trade-offs across strikes, not "which
  strike is more likely to make you money."

**The actual edge**: `compare_implied_vs_realized_ev` is the first-class
feature this whole module exists to support. It prices the option once, at
market IV (that's the real premium you pay), but computes `P_win` **twice**
— once with market IV (the "risk-neutral" reference view) and once with
your own realized/forecast vol (the "edge" view). If your vol estimate
differs meaningfully from the market's, the difference between those two
EVs is a genuine, quantifiable statement about whether you think this
option is mispriced — not a probability of the stock going up.

### 3. Position sizing (`sizing.py`)

Full Kelly: `f* = (p·b − q)/b`, where `b` is the payout odds implied by
your own profit-target/stop-loss rule. **Half-Kelly is the default**
(`config.use_half_kelly`) because `p` and `b` are estimates, not known
constants — full Kelly is notorious for compounding an overestimated edge
into real drawdown fast.

**The hard cap always wins.** `size_position()` is the only function meant
to be called from outside this module; it computes Kelly, then
unconditionally takes `min(kelly_size, cap_size)` as its last step. There
is no parameter, flag, or code path elsewhere in this package that can
produce a size larger than `config.max_position_pct_of_account` (default
5%) of your account — see `tests/test_sizing.py::test_hard_cap_always_wins_when_kelly_suggests_more`.

**Assumption that breaks fastest**: Kelly assumes you actually know your
edge (`p` and `b`) with some precision. You don't — `p` is either a
risk-neutral model output or your own vol guess, and `b` depends on you
actually exiting at your stated target. Half-Kelly and the hard cap both
exist specifically to blunt the damage when that assumption is wrong,
which is most of the time.

### 4. Risk manager — exit rules (`risk_manager.py`)

Five mechanical, pre-committed rules, each returning **why** it fired with
the actual numbers, not just a boolean:

| Rule | Default | Fires when |
|---|---|---|
| Profit target | +100% | current P&L ≥ target |
| Stop loss | −50% | current P&L ≤ stop |
| Delta exit | 0.15 | `\|delta\|` decays below threshold |
| Theta exit | DTE ≤ 3 | position is OTM AND DTE at/under threshold |
| IV-crush exit | 20 pts | IV has dropped ≥ threshold since entry |

Plus portfolio-level governors (`check_portfolio_governors`): a **daily
max-loss halt** on new entries (default 3% of account) and optional net
delta/vega dollar caps (unset by default — there's no universal correct
value, set them to what your account can actually tolerate).

**Known limitation, dashboard-specific**: the dashboard doesn't persist a
trade log, so "entry IV" for the IV-crush rule is approximated as
whatever IV the dashboard first observed for a position in the current
session — not the real entry IV from when you opened the trade. The
IV-crush exit is only meaningful once the dashboard has been continuously
watching a position since you opened it. The CLI and direct
`risk_manager.Position` usage don't have this problem if you pass the real
entry IV yourself.

### 5. Realized volatility (`realized_vol.py`)

Three historical-vol estimators, in increasing order of statistical
efficiency (more of each day's price action used per estimate):

- **Close-to-close**: `σ = std(log returns) × √252`. Simplest, noisiest.
- **Parkinson**: uses the day's high/low range,
  `σ² = (1/(4n·ln2))·Σln(H/L)²`. More efficient, but assumes no
  overnight gaps.
- **Garman-Klass**: adds open/close, most efficient of the three under
  continuous-GBM-with-no-drift — an assumption that breaks around
  earnings, gaps, and trend days, exactly when you most want a correct
  vol read.

**GARCH(1,1)** (`fit_garch_11`, `garch_forecast_vol`): fit by maximum
likelihood on demeaned returns, `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}`,
constrained to `α+β < 1` for stationarity. The forecast mean-reverts
geometrically toward the long-run variance `ω/(1-α-β)` as the horizon
grows — GARCH's answer to "how volatile eventually" is just the
unconditional variance, regardless of today's conditions. Symmetric: an
up move and a down move of the same size raise the forecast by the same
amount.

**EGARCH(1,1)** (`fit_egarch_11`, `egarch_forecast_vol`): adds the
"leverage effect" — equity vol empirically rises more after a down move
than an up move of the same size — via an asymmetry term `γ`, modeling
log-variance directly (`ln σ²_t = ω + β·ln σ²_{t-1} + α(|z_{t-1}|-E|z|) +
γ·z_{t-1}`) so `ω`, `α`, `γ` need no positivity constraints, only `|β|<1`.
A fitted `γ < 0` is the asymmetry the model exists to capture; a fitted `ν`
below ~10 under `dist="t"` is a diagnostic that fat tails matter for that
name. Both GARCH and EGARCH accept `dist="t"` innovations as an
alternative to the normal default.

### 6. Portfolio risk (`portfolio_risk.py`)

Everything above describes ONE position or ONE underlying. This module
answers the book-level question instead: correlation/covariance across
every underlying held (`correlation_matrix`, `covariance_matrix`, on
date-aligned daily log returns), delta-normal variance-covariance VaR
(`parametric_var` — `w^T Σ w` linearized around today's net dollar delta
per name) and historical-simulation VaR (`historical_var` — replays each
day's actual joint return against today's exposures, so real fat tails and
real historical co-movement come through without a normality assumption),
and market-wide stress tests that fully reprice every option leg via
Black-Scholes at a shocked spot/IV (`option_leg_stress_pl`) rather than
linearizing — the one place here that captures gamma/vega convexity for a
large move. See the dashboard's **Portfolio Risk** tab, or the module
docstring for the full tradeoffs between the two VaR methods.

## Architecture notes

- `config.py` is the only place thresholds live — nothing else hardcodes a
  number. Build a custom `RiskConfig(...)` and pass it through if you want
  different discipline rules than the defaults.
- `ibkr_client.py` is optional and mockable (`MockIBKRClient`) — nothing in
  the models above imports it or requires a live connection. It exists to
  feed the models live numbers instead of typing them by hand, and isn't
  fully wired for option Greeks/IV yet (see its docstring) since that needs
  a live IBKR options market-data subscription this environment couldn't
  test against.
- The dashboard's Risk Tool tab is a thin UI layer over these same
  functions — it doesn't duplicate any model logic.

## What this tool will never do

It will not forecast price direction, and no probability it prints is a
real-world prediction — every `P_win`/ITM-probability value is explicitly
risk-neutral (see `strike_selection.py`'s module docstring for exactly what
that means and doesn't mean). If you want an edge, it has to come from your
own volatility view being better than the market's, compared explicitly via
`compare_implied_vs_realized_ev` — that's the one lever this tool gives you
beyond bookkeeping and discipline enforcement.
