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
  forecasting fit by maximum likelihood (`risk_tool/realized_vol.py`), and a
  beta-hedge calculator (`risk_tool/hedge.py`) that separates price-level
  correlation (inflated by shared trend) from return correlation (the honest
  co-movement signal), sized off delta-adjusted exposure rather than notional.
- **Test discipline** — the entire `risk_tool/` package is pure, dependency-injected,
  and independently pytest-covered (122 passing cases: `pytest tests/ -v`) —
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
- **Vol Skew** — live IV-by-strike and bid/ask-spread-by-strike for any
  underlying/expiration, with your held contracts marked on both charts.
- **Risk Tool** — strike selection by expected value, Kelly-based position
  sizing with a non-overridable hard cap, pre-committed entry/exit levels,
  and a live monitor that runs exit rules against your actual open
  positions. Does not predict direction — see `risk_tool/README.md` for the
  full model-by-model writeup (math, assumptions, limitations). Also usable
  standalone: `python3 -m risk_tool.cli --help`.
- **Correlation Explorer** — cumulative-return chart and price-level/return
  correlation for any two symbols you type in.
- **Hedge Calculator** — sizes a beta-hedge (shares or futures) for a shares
  or options position against any correlated instrument, with beta estimated
  live from price history, plus a movable P&L scenario chart (drag to any
  bearish or bullish move) for the resulting hedged position.
- **Orders & History** — open orders and recent fills.
- **Win Rate** — realized round-trip trades FIFO-matched from your fill history, with win rate, avg
  win/loss, profit factor, a cumulative realized-P&L chart, and a breakdown by equity vs. options.
  Equity matching is exact; options are matched per underlying symbol only (fill history has no
  per-contract identity) — see the caveat in the tab itself.
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
- The trade journal covers **filled** orders from the last 90 days by default
  (`data_fetch.get_order_history(days_back=...)`); increase that if you want
  a longer lookback.
- This is a personal-use tool with no auth layer of its own — don't expose
  `streamlit run` beyond `localhost` (e.g. don't tunnel it to the public
  internet) since it holds your live Robinhood session.

## Project layout

```
dashboard.py           Streamlit entrypoint / UI
app/auth.py            Robinhood login (+ MFA) handling
app/data_fetch.py      All robin_stocks calls, returned as plain DataFrames
app/vol_analysis.py    Greeks aggregation, vega/theta by underlying, expiry checks, cross-symbol correlation
app/journal.py         Trade journal normalization + CSV export
app/performance.py     FIFO round-trip trade matching + win-rate stats
app/colors.py          Shared chart color tokens
risk_tool/             Pricing, Greeks, strike selection, sizing, exit rules, beta-hedge sizing — see risk_tool/README.md
tests/                 pytest suite for risk_tool (run: pytest tests/ -v)
exports/               CSV journal exports land here (gitignored)
```
