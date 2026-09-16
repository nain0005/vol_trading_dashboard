# Vol Trading Dashboard

A local dashboard for a mixed options + VIX-ETP book, backed directly by your
Robinhood account via [`robin_stocks`](https://github.com/jmfernandes/robin_stocks).
Everything runs on your machine — your credentials go straight to Robinhood
and are never sent anywhere else.

This is personal-use software that connects to **your own** brokerage
account — it's not a hosted product or a multi-tenant service, and it isn't
meant to run against anyone else's account or be exposed beyond `localhost`
(see Notes/limitations below).

## What this demonstrates

A few things this codebase is meant to show, for anyone skimming it:

- **Options pricing & Greeks from scratch** — Black-Scholes, an American
  binomial tree, Brent's-method implied vol solving, all independently
  verified against known textbook values and put-call parity
  (`risk_tool/pricing.py`, `risk_tool/greeks.py`).
- **Real risk management, not just signals** — Kelly-criterion sizing with a
  hard cap that always wins regardless of what Kelly suggests
  (`risk_tool/sizing.py`), pre-committed exit rules evaluated live against
  open positions (`risk_tool/risk_manager.py`), and portfolio-level delta/vega
  governors that can halt new entries.
- **Statistics applied correctly, not just called** — GARCH/EGARCH vol
  forecasting fit by maximum likelihood (`risk_tool/realized_vol.py`), a
  beta-hedge calculator (`risk_tool/hedge.py`) that separates price-level
  correlation (inflated by shared trend) from return correlation (the honest
  co-movement signal), and portfolio-level delta-normal + historical
  Value-at-Risk with full Black-Scholes stress-test repricing
  (`risk_tool/portfolio_risk.py`).
- **Test discipline** — the entire `risk_tool/` package is pure, dependency-injected,
  and independently pytest-covered (237 passing cases: `pytest tests/ -v`) —
  it's also usable as a standalone CLI with no Streamlit/Robinhood dependency
  at all (`python3 -m risk_tool.cli --help`).
- **Production-adjacent app structure** — Robinhood I/O is fully isolated from
  presentation (`app/data_fetch.py` returns plain DataFrames, `dashboard.py` is
  UI-only), so the business logic is testable without a live session or network
  access.

## What it shows

- **Overview** — equity, cash, buying power, day P&L, portfolio value over time.
- **Positions & Greeks** — equity + option positions, with per-contract delta,
  theta, vega, gamma, and implied volatility, plus a near-expiry warning.
- **Vol Exposure** — book-level net delta/theta/vega/gamma, vega & theta
  broken down by underlying, and your VIX-ETP holdings (VXX/UVXY/SVXY/etc.,
  configurable) with live quotes.
- **Portfolio Risk** — cross-position risk the other tabs don't answer: net
  dollar exposure per underlying (shares netted against option delta so a
  covered call correctly reads as less risky than naked stock), a
  return-correlation heatmap across everything you hold, side-by-side
  parametric (delta-normal) and historical Value-at-Risk with a simulated
  daily P&L histogram, a market-wide stress test that fully reprices every
  option leg via Black-Scholes at a shocked spot/IV (not a linear Greeks
  approximation) across a range of preset market moves, and a 3D stress
  surface sweeping spot move × IV shock independently across the whole book
  (same repricing, two axes instead of one).
- **Sigma Screener** — ranks tickers by how overdue they are for a 2σ/3σ
  move relative to their own trailing realized vol, using both an empirical
  recurrence estimate and a fitted Student-t tail model — with an explicit
  methodology note on why "overdue" alone isn't a forecast.
- **Vol Skew** — live IV-by-strike and bid/ask-spread-by-strike for any
  underlying/expiration (with your held contracts marked on both charts), a
  term-structure view (ATM IV across expirations vs. trailing 20d/60d
  realized vol), a full 3D volatility surface (strike × expiration × IV,
  OTM-stitched and interpolated onto a shared strike grid), open
  interest × volume and day-over-day OI change by strike, and a 3D open
  interest surface (strike × date) built from a local snapshot log since
  Robinhood exposes no OI history endpoint.
- **Risk Tool** — strike selection by expected value, Kelly-based position
  sizing with a non-overridable hard cap, pre-committed entry/exit levels,
  and a live monitor that runs exit rules against your actual open
  positions. Does not predict direction — see `risk_tool/README.md` for the
  full model-by-model writeup (math, assumptions, limitations). Also usable
  standalone: `python3 -m risk_tool.cli --help`.
- **Spread Selector** — ranks real, listed strikes for bear put spreads, bull
  put spreads, straddles, and strangles against your own realized/GARCH/
  EGARCH vol view (edge EV), not just risk:reward.
- **Strategy Payoff** — exact max profit/loss/breakeven(s) for any multi-leg
  combination of calls/puts you build by hand, plus a live mark-to-market
  curve.
- **Options Lab** — a 3D P&L surface (spot × days-forward, toggleable to a
  2D heatmap) with the live what-if scenario marked on it, Greeks
  sensitivity curves, and an earnings/IV-crush simulator for any structure —
  build one from scratch, import an open position, or send one over from the
  Spread Selector or Strategy Payoff tabs.
- **Delta Hedge** — sizes a beta-hedge (shares or futures) for a shares or
  options position against any correlated instrument, with beta estimated
  live from price history, plus a movable P&L scenario chart (drag to any
  bearish or bullish move) for the resulting hedged position.
- **Orders & History** — open orders and recent fills.
- **Win Rate** — realized round-trip trades FIFO-matched from your fill history (matched per exact
  option contract, not just underlying symbol, so two different contracts on the same underlying
  held at once can't cross-match), with win rate, avg win/loss, profit factor, a cumulative
  realized-P&L chart, and a breakdown by equity vs. options.
- **Journal / Export** — normalized trade log with a CSV download button
  (and an option to save a timestamped copy into `exports/`).

## Setup

```bash
cd vol-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set ROBINHOOD_USERNAME / ROBINHOOD_PASSWORD, and optionally
# ROBINHOOD_TOTP_SECRET if you have app-based 2FA (see below), and
# VOL_TICKERS if your vol-ETP watchlist differs from the default.

streamlit run dashboard.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

### Demo mode (no Robinhood account required)

```bash
DEMO_MODE=true streamlit run dashboard.py
```

Runs the exact same UI on synthetic data (`app/demo_data.py`) — no Robinhood
credentials, no login screen, no real account anywhere in the loop. Prices,
positions, and history are generated deterministically (seeded per symbol),
but option prices/Greeks are computed with the real `risk_tool` Black-Scholes
engine, so that part is genuinely the same math the live version uses.

This is what a **public deployment** should run — e.g. on
[Streamlit Community Cloud](https://share.streamlit.io): point it at this
repo, set the main file to `dashboard.py`, and add `DEMO_MODE = "true"` under
the app's Secrets (not `.env` — that file never leaves your machine and isn't
part of the repo). Never deploy this app publicly connected to a real
Robinhood account — see Notes/limitations below for why.

### Login & 2FA

- If Robinhood challenges you for an SMS/app code on first login, the app
  will show a code input field — enter it and click "Log in" again.
- To skip that prompt on future runs, set `ROBINHOOD_TOTP_SECRET` in `.env`
  to your authenticator app's base32 secret (the same one you'd scan as a QR
  code when setting up 2FA) — the app generates the current code automatically.
- `robin_stocks` caches a session token locally (`~/.tokens/robinhood.pickle`)
  so you generally won't be re-prompted every single run.

### Notes / limitations

- `VOL_TICKERS` in `.env` controls which symbols count as "vol ETPs" for the
  dedicated exposure view — edit it to match what you actually trade.
- Robinhood's API doesn't expose the raw VIX index quote, so vol-ETP tracking
  uses the ETPs themselves (VXX, UVXY, etc.) as the proxy, not `^VIX` directly.
- The trade journal covers **filled** orders from the last ~10 years by
  default (`data_fetch.get_order_history(days_back=3650)`) — effectively
  all-time for any real account; pass a smaller `days_back` if you want a
  shorter window.
- This is a personal-use tool with no auth layer of its own — don't expose
  `streamlit run` beyond `localhost` (e.g. don't tunnel it to the public
  internet) since it holds your live Robinhood session.

## Project layout

```
dashboard.py           Streamlit entrypoint / UI
app/auth.py            Robinhood login (+ MFA) handling
app/data_fetch.py      All robin_stocks calls, returned as plain DataFrames
app/vol_analysis.py    Greeks aggregation, vega/theta by underlying, expiry checks, cross-symbol
                        correlation, portfolio exposure netting + stress-test aggregation
app/journal.py         Trade journal normalization + CSV export
app/performance.py     FIFO round-trip trade matching + win-rate stats (per exact contract)
app/colors.py          Shared chart/UI color tokens (also drives dashboard.py's custom CSS)
risk_tool/             Pricing, Greeks, strike selection, sizing, exit rules, beta-hedge sizing,
                        multi-leg strategy payoff, spread selection, options lab, portfolio-level
                        VaR/stress testing — see risk_tool/README.md
tests/                 pytest suite (run: pytest tests/ -v)
exports/               CSV journal exports land here (gitignored)
```
