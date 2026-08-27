
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import colors, data_fetch, journal, performance, vol_analysis
from app.auth import ensure_logged_in, logout
from risk_tool import option_strategy
from risk_tool import realized_vol as rv
from risk_tool import risk_manager
from risk_tool import sizing as risk_sizing
from risk_tool import strike_selection
from risk_tool.config import DEFAULT_CONFIG, RiskConfig

st.set_page_config(page_title="Vol Trading Dashboard", page_icon="📉", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    return {
        "overview": data_fetch.get_portfolio_overview(),
        "history": data_fetch.get_portfolio_history(),
        "equity_positions": data_fetch.get_equity_positions(),
        "option_positions": data_fetch.get_option_positions(),
        "open_orders": data_fetch.get_open_orders(),
        "order_history": data_fetch.get_order_history(),
        "vol_quotes": data_fetch.get_vol_ticker_quotes(),
    }


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def render_overview(d: dict):
    ov = d["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio equity", money(ov["equity"]))
    c2.metric("Cash", money(ov["cash"]))
    c3.metric("Buying power", money(ov["buying_power"]))
    c4.metric("Day P&L", money(ov["day_pl"]), f"{ov['day_pl_pct']:.2f}%")

    hist = d["history"]
    if not hist.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["equity"],
                mode="lines",
                line=dict(color=colors.CATEGORICAL[0], width=2),
                hovertemplate="%{x|%b %d}<br>$%{y:,.0f}<extra></extra>",
                name="Equity",
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor=colors.SURFACE,
            paper_bgcolor=colors.SURFACE,
            xaxis=dict(showgrid=False, color=colors.INK_MUTED),
            yaxis=dict(showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No portfolio history returned yet.")


def render_positions(d: dict):
    st.subheader("Equity positions")
    eq = d["equity_positions"]
    if eq.empty:
        st.caption("No open equity positions.")
    else:
        st.dataframe(
            eq,
            use_container_width=True,
            hide_index=True,
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.0f"),
                "avg_cost": st.column_config.NumberColumn(format="$%.2f"),
                "last_price": st.column_config.NumberColumn(format="$%.2f"),
                "market_value": st.column_config.NumberColumn(format="$%.2f"),
                "unrealized_pl": st.column_config.NumberColumn(format="$%.2f"),
                "unrealized_pl_pct": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    st.subheader("Option positions")
    opt = d["option_positions"]
    if opt.empty:
        st.caption("No open option positions.")
    else:
        display_cols = [
            "symbol", "type", "side", "strike", "expiration", "dte", "quantity",
            "avg_price", "mark_price", "market_value", "unrealized_pl",
            "implied_volatility", "delta", "theta", "vega", "gamma",
        ]
        st.dataframe(
            opt[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "strike": st.column_config.NumberColumn(format="$%.2f"),
                "avg_price": st.column_config.NumberColumn(format="$%.2f"),
                "mark_price": st.column_config.NumberColumn(format="$%.2f"),
                "market_value": st.column_config.NumberColumn(format="$%.2f"),
                "unrealized_pl": st.column_config.NumberColumn(format="$%.2f"),
                "implied_volatility": st.column_config.NumberColumn(format="percent"),
                "delta": st.column_config.NumberColumn(format="%.2f"),
                "theta": st.column_config.NumberColumn(format="%.2f"),
                "vega": st.column_config.NumberColumn(format="%.2f"),
                "gamma": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        near = vol_analysis.near_expiry(opt)
        if not near.empty:
            st.warning(f"{len(near)} option position(s) expiring within 7 days — check assignment/roll risk.")


def render_vol_exposure(d: dict):
    opt = d["option_positions"]
    eq = d["equity_positions"]
    greeks = vol_analysis.aggregate_greeks(opt)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net delta (shares-equiv.)", f"{greeks['net_delta']:,.0f}")
    c2.metric("Net theta ($/day)", money(greeks["net_theta"]))
    c3.metric("Net vega ($/vol pt)", money(greeks["net_vega"]))
    c4.metric("Net gamma", f"{greeks['net_gamma']:,.2f}")

    st.subheader("Vega & theta by underlying")
    by_underlying = vol_analysis.premium_by_underlying(opt)
    if by_underlying.empty:
        st.caption("No option positions to break down.")
    else:
        fig = go.Figure()
        bar_colors = [colors.DIVERGING_POS if v >= 0 else colors.DIVERGING_NEG for v in by_underlying["net_vega"]]
        fig.add_trace(
            go.Bar(
                x=by_underlying["symbol"],
                y=by_underlying["net_vega"],
                marker_color=bar_colors,
                hovertemplate="%{x}<br>Net vega: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor=colors.SURFACE,
            paper_bgcolor=colors.SURFACE,
            xaxis=dict(showgrid=False, color=colors.INK_MUTED),
            yaxis=dict(showgrid=True, gridcolor=colors.GRIDLINE, zerolinecolor=colors.INK_MUTED, color=colors.INK_MUTED),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_underlying, use_container_width=True, hide_index=True)

    st.subheader("VIX-linked ETP holdings")
    vol_holdings = vol_analysis.vol_etp_exposure(eq)
    quotes = d["vol_quotes"]
    if not quotes.empty:
        cols = st.columns(len(quotes))
        for col, (_, row) in zip(cols, quotes.iterrows()):
            col.metric(row["symbol"], f"${row['last_price']:.2f}", f"{row['change_pct']:.2f}%")
    if vol_holdings.empty:
        st.caption("No direct positions in your configured vol-ETP list (see VOL_TICKERS in .env).")
    else:
        st.dataframe(vol_holdings, use_container_width=True, hide_index=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_expirations(symbol: str):
    return data_fetch.get_option_chain_expirations(symbol)


@st.cache_data(ttl=120, show_spinner="Pulling option chain...")
def fetch_chain(symbol: str, expiration: str):
    return data_fetch.get_option_chain_skew(symbol, expiration)


@st.fragment(run_every=5)
def render_live_quote(symbol: str):
    quote = data_fetch.get_stock_quote(symbol)
    c1, c2, c3 = st.columns(3)
    c1.metric("Best bid", f"${quote['bid']:.2f}")
    c2.metric("Best ask", f"${quote['ask']:.2f}")
    c3.metric("Mark (mid)", f"${quote['mark']:.2f}")
    st.caption(f"Last trade ${quote['last_trade_price']:.2f} · refreshes every 5s")


def render_vol_skew(d: dict):
    st.subheader("Volatility & liquidity skew explorer")
    st.caption(
        "Look up the live option chain for any underlying — your held contracts for the "
        "selected expiration are marked on both charts."
    )

    held_positions = d["option_positions"]
    if "skew_symbol_input" not in st.session_state:
        default_symbol = held_positions.iloc[0]["symbol"] if not held_positions.empty else ""
        st.session_state["skew_symbol_input"] = default_symbol

    symbol = st.text_input("Underlying ticker", key="skew_symbol_input", placeholder="e.g. SPY").strip().upper()
    if not symbol:
        st.caption("Enter a ticker to load its expirations.")
        return

    try:
        expirations = fetch_expirations(symbol)
    except Exception as exc:
        st.error(f"Couldn't load expirations for {symbol}: {exc}")
        return
    if not expirations:
        st.warning(f"No option chain found for {symbol}. Check the ticker and that it has listed options.")
        return

    col_a, col_b = st.columns([4, 1])
    with col_a:
        expiration = st.selectbox("Expiration", expirations, key="skew_expiration")
    with col_b:
        st.write("")
        if st.button("Refresh chain"):
            fetch_chain.clear()

    try:
        chain = fetch_chain(symbol, expiration)
    except Exception as exc:
        st.error(f"Couldn't load the option chain: {exc}")
        return
    if chain.empty:
        st.warning("No quoted contracts returned for this expiration.")
        return

    st.markdown("##### Live underlying quote")
    render_live_quote(symbol)

    spot = data_fetch.get_stock_quote(symbol)["mark"]
    metrics = vol_analysis.skew_metrics(chain, spot)

    c1, c2 = st.columns(2)
    c1.metric("ATM IV", f"{metrics['atm_iv']:.1%}" if metrics["atm_iv"] is not None else "—")
    c2.metric(
        "25Δ risk reversal (C − P)",
        f"{metrics['risk_reversal_25d']:.1%}" if metrics["risk_reversal_25d"] is not None else "—",
    )

    calls = chain[chain["type"] == "call"]
    puts = chain[chain["type"] == "put"]
    held = vol_analysis.held_contracts_in_chain(held_positions, chain, symbol, expiration)

    def held_marker_trace(y_col: str, hover_suffix: str, tickformat: str):
        if held.empty:
            return None
        return go.Scatter(
            x=held["strike"],
            y=held[y_col],
            mode="markers",
            name="Your position",
            marker=dict(symbol="diamond-open", size=15, color=colors.STATUS_WARNING, line=dict(width=2)),
            customdata=held[["side", "quantity", "type"]],
            hovertemplate=(
                "Your position — %{customdata[1]}x %{customdata[0]} %{customdata[2]}"
                f"<br>Strike $%{{x:.2f}}<br>{hover_suffix} %{{y:{tickformat}}}<extra></extra>"
            ),
        )

    st.markdown("##### Volatility skew (IV by strike)")
    fig_iv = go.Figure()
    fig_iv.add_trace(
        go.Scatter(
            x=calls["strike"], y=calls["iv"], mode="lines+markers", name="Calls",
            line=dict(color=colors.CATEGORICAL[0], width=2),
            marker=dict(size=6),
            hovertemplate="Strike $%{x:.2f}<br>IV %{y:.1%}<extra>Call</extra>",
        )
    )
    fig_iv.add_trace(
        go.Scatter(
            x=puts["strike"], y=puts["iv"], mode="lines+markers", name="Puts",
            line=dict(color=colors.CATEGORICAL[5], width=2),
            marker=dict(size=6),
            hovertemplate="Strike $%{x:.2f}<br>IV %{y:.1%}<extra>Put</extra>",
        )
    )
    iv_marker = held_marker_trace("iv", "IV", ".1%")
    if iv_marker:
        fig_iv.add_trace(iv_marker)
    if spot:
        fig_iv.add_vline(x=spot, line=dict(color=colors.INK_MUTED, dash="dash", width=1), annotation_text="Mark", annotation_position="top")
    fig_iv.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=colors.SURFACE,
        paper_bgcolor=colors.SURFACE,
        xaxis=dict(title="Strike", showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(title="Implied volatility", showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_iv, use_container_width=True)

    st.markdown("##### Bid/ask spread skew (liquidity by strike)")
    st.caption("Spread shown as % of mid price — the standard way to compare liquidity across strikes with very different premiums.")
    fig_spread = go.Figure()
    fig_spread.add_trace(
        go.Scatter(
            x=calls["strike"], y=calls["spread_pct"], mode="lines+markers", name="Calls",
            line=dict(color=colors.CATEGORICAL[0], width=2),
            marker=dict(size=6),
            hovertemplate="Strike $%{x:.2f}<br>Spread %{y:.1f}%<extra>Call</extra>",
        )
    )
    fig_spread.add_trace(
        go.Scatter(
            x=puts["strike"], y=puts["spread_pct"], mode="lines+markers", name="Puts",
            line=dict(color=colors.CATEGORICAL[5], width=2),
            marker=dict(size=6),
            hovertemplate="Strike $%{x:.2f}<br>Spread %{y:.1f}%<extra>Put</extra>",
        )
    )
    spread_marker = held_marker_trace("spread_pct", "Spread", ".1f")
    if spread_marker:
        fig_spread.add_trace(spread_marker)
    if spot:
        fig_spread.add_vline(x=spot, line=dict(color=colors.INK_MUTED, dash="dash", width=1), annotation_text="Mark", annotation_position="top")
    fig_spread.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=colors.SURFACE,
        paper_bgcolor=colors.SURFACE,
        xaxis=dict(title="Strike", showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(title="Spread (% of mid)", showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_spread, use_container_width=True)

    st.dataframe(
        chain,
        use_container_width=True,
        hide_index=True,
        column_config={
            "strike": st.column_config.NumberColumn(format="$%.2f"),
            "iv": st.column_config.NumberColumn(format="percent"),
            "delta": st.column_config.NumberColumn(format="%.3f"),
            "bid": st.column_config.NumberColumn(format="$%.2f"),
            "ask": st.column_config.NumberColumn(format="$%.2f"),
            "mid": st.column_config.NumberColumn(format="$%.2f"),
            "spread": st.column_config.NumberColumn(format="$%.2f"),
            "spread_pct": st.column_config.NumberColumn(format="%.1f%%"),
            "volume": st.column_config.NumberColumn(format="%.0f"),
            "open_interest": st.column_config.NumberColumn(format="%.0f"),
        },
    )


@st.cache_data(ttl=3600, show_spinner="Pulling price history...")
def fetch_price_history(symbol: str):
    return data_fetch.get_equity_historicals(symbol)


def _strike_ev_table(strike_evs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strike": s.strike,
                "premium": s.premium,
                "p_win (risk-neutral)": s.p_win_risk_neutral,
                "EV": s.ev,
                "risk:reward": s.risk_reward,
                "setup": "Poor (< min R:R)" if s.is_poor_setup else "OK",
                "delta": s.delta,
                "theta/day": s.theta,
                "vega": s.vega,
            }
            for s in strike_evs
        ]
    )


STRIKE_EV_COLUMN_CONFIG = {
    "strike": st.column_config.NumberColumn(format="$%.2f"),
    "premium": st.column_config.NumberColumn(format="$%.2f"),
    "p_win (risk-neutral)": st.column_config.NumberColumn(format="percent"),
    "EV": st.column_config.NumberColumn(format="$%+.2f"),
    "risk-neutral EV": st.column_config.NumberColumn(format="$%+.2f"),
    "risk:reward": st.column_config.NumberColumn(format="%.2f:1"),
    "delta": st.column_config.NumberColumn(format="%+.3f"),
    "theta/day": st.column_config.NumberColumn(format="%+.3f"),
    "vega": st.column_config.NumberColumn(format="%.3f"),
}


def render_risk_tool(d: dict):
    st.subheader("Options risk management & strike selection")
    st.caption(
        "This tool does not predict direction. You supply the thesis (ticker, direction, timeframe) — "
        "it prices candidate strikes on expected value, sizes the position, and gives you pre-committed "
        "exit levels. All win probabilities are risk-neutral (derived from option prices), not forecasts."
    )

    account_equity = float(d["overview"].get("equity") or 0.0)

    with st.form("risk_tool_inputs"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", value=st.session_state.get("skew_symbol_input", "")).strip().upper()
        direction = c2.selectbox("Direction", ["call", "put"])
        dte = c3.number_input("Days to expiry", min_value=1, value=30, step=1)

        c4, c5, c6 = st.columns(3)
        spot_input = c4.number_input("Spot price ($, 0 = auto-fetch)", min_value=0.0, value=0.0, step=0.5)
        iv_input = c5.number_input("Market IV (%, 0 = use chain ATM IV)", min_value=0.0, value=0.0, step=1.0)
        strike_increment = c6.number_input("Strike increment ($)", min_value=0.5, value=5.0, step=0.5)

        c7, c8 = st.columns(2)
        account_override = c7.number_input("Account size ($)", min_value=0.0, value=account_equity, step=1000.0)
        use_realized = c8.checkbox("Compute realized vol from 1yr price history (adds the realized-vs-implied edge check)")

        submitted = st.form_submit_button("Analyze", type="primary")

    if not submitted:
        st.info("Fill in a thesis above and click Analyze.")
        return
    if not ticker:
        st.error("Enter a ticker.")
        return

    spot = spot_input if spot_input > 0 else data_fetch.get_stock_quote(ticker)["mark"]
    if not spot:
        st.error(f"Couldn't get a live price for {ticker}.")
        return

    market_iv = iv_input / 100 if iv_input > 0 else None
    if market_iv is None:
        try:
            expirations = data_fetch.get_option_chain_expirations(ticker)
            today = pd.Timestamp.now().normalize()
            target_date = today + pd.Timedelta(days=int(dte))
            nearest_exp = min(expirations, key=lambda e: abs((pd.Timestamp(e) - target_date).days))
            chain = data_fetch.get_option_chain_skew(ticker, nearest_exp)
            market_iv = vol_analysis.skew_metrics(chain, spot)["atm_iv"]
        except Exception as exc:
            st.error(f"Couldn't determine market IV from the option chain ({exc}) — enter it manually above.")
            return
    if not market_iv:
        st.error("Couldn't determine market IV — enter it manually above.")
        return

    realized_or_forecast_vol = None
    if use_realized:
        try:
            hist = fetch_price_history(ticker)
            realized_or_forecast_vol = rv.close_to_close_vol(hist["close"])
        except Exception as exc:
            st.warning(f"Couldn't compute realized vol ({exc}) — continuing with market IV only.")

    config = DEFAULT_CONFIG
    T = dte / 365.0
    strikes = strike_selection.generate_candidate_strikes(spot, strike_increment, config.num_strikes_each_side)

    if realized_or_forecast_vol:
        st.markdown("##### Realized-vs-implied edge check")
        spread = realized_or_forecast_vol - market_iv
        c1, c2, c3 = st.columns(3)
        c1.metric("Market IV", f"{market_iv:.1%}")
        c2.metric("Realized vol (close-to-close, 1yr)", f"{realized_or_forecast_vol:.1%}")
        c3.metric("Spread (realized − implied)", f"{spread:+.1%}")
        st.caption(
            "Positive spread = your realized vol exceeds market IV (options may be cheap relative to actual "
            "movement); negative = the opposite. This is the only place an actual edge can come from in this "
            "model — see strike_selection.py's module docstring."
        )

        comparisons = [
            strike_selection.compare_implied_vs_realized_ev(
                spot, K, T, config.risk_free_rate, config.dividend_yield, market_iv, realized_or_forecast_vol, direction, config
            )
            for K in strikes
        ]
        comparisons.sort(key=lambda c: c.edge_ev.ev, reverse=True)

        st.markdown("##### Strikes ranked by edge EV (your vol view, market's actual premium)")
        edge_table = _strike_ev_table([c.edge_ev for c in comparisons])
        edge_table.insert(edge_table.columns.get_loc("EV") + 1, "risk-neutral EV", [c.risk_neutral_ev.ev for c in comparisons])
        st.dataframe(edge_table, use_container_width=True, hide_index=True, column_config=STRIKE_EV_COLUMN_CONFIG)
        best = comparisons[0].edge_ev
        best_p_win = best.p_win_risk_neutral
    else:
        ranked = strike_selection.rank_strikes_by_ev(spot, T, config.risk_free_rate, config.dividend_yield, market_iv, direction, strikes, config)
        st.markdown("##### Strikes ranked by risk-neutral EV (market IV throughout)")
        st.dataframe(_strike_ev_table(ranked), use_container_width=True, hide_index=True, column_config=STRIKE_EV_COLUMN_CONFIG)
        best = ranked[0]
        best_p_win = best.p_win_risk_neutral

    st.markdown("##### Recommended strike")
    c1, c2, c3 = st.columns(3)
    c1.metric("Strike", f"${best.strike:.2f}")
    c2.metric("Premium", f"${best.premium:.2f}")
    c3.metric("EV", f"${best.ev:+.2f}")
    if best.is_poor_setup:
        st.warning(f"Risk/reward {best.risk_reward:.2f}:1 is below your {config.min_risk_reward_ratio:.1f}:1 minimum — poor setup by this rule, even though it's the best of the candidates.")

    sizing_result = risk_sizing.size_position(
        account_size=account_override or 1.0,
        premium_per_contract=best.premium,
        p_win=best_p_win,
        profit_if_win_per_contract=best.profit_if_win,
        config=config,
    )
    st.markdown("##### Position sizing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kelly fraction used", f"{sizing_result.kelly_fraction_used:.1%}")
    c2.metric("Kelly-suggested size", f"${sizing_result.kelly_dollar_size:,.0f}")
    c3.metric("Hard cap (5% default)", f"${sizing_result.hard_cap_dollar_size:,.0f}")
    c4.metric("Recommended", f"${sizing_result.recommended_dollar_size:,.0f}", f"{sizing_result.recommended_contracts} contracts")
    if sizing_result.capped_by_hard_limit:
        st.caption("Capped by the hard %-of-account limit, not by Kelly — the cap always wins.")

    st.markdown("##### Pre-committed entry/exit levels")
    entry = best.premium
    c1, c2 = st.columns(2)
    c1.metric("Profit target", f"${entry * (1 + config.profit_target_pct):.2f}", f"+{config.profit_target_pct:.0%}")
    c2.metric("Stop loss", f"${entry * (1 + config.stop_loss_pct):.2f}", f"{config.stop_loss_pct:.0%}")
    st.caption(
        f"Also exits if |delta| < {config.delta_exit_threshold}, DTE <= {config.theta_exit_dte_threshold} while still OTM, "
        f"or IV drops >= {config.iv_crush_threshold_points:.0%} points from entry ({market_iv:.1%})."
    )


def render_position_monitor(d: dict):
    st.subheader("Live position monitor — exit rules on your actual open positions")
    positions = d["option_positions"]
    if positions.empty:
        st.caption("No open option positions to monitor.")
        return

    config = DEFAULT_CONFIG
    st.session_state.setdefault("entry_iv_cache", {})

    for _, pos in positions.iterrows():
        cache_key = f"{pos['symbol']}_{pos['strike']}_{pos['expiration']}_{pos['type']}_{pos['side']}"
        if cache_key not in st.session_state["entry_iv_cache"]:
            st.session_state["entry_iv_cache"][cache_key] = pos["implied_volatility"]
        entry_iv = st.session_state["entry_iv_cache"][cache_key]

        try:
            spot = data_fetch.get_underlying_price(pos["symbol"])
        except Exception:
            spot = pos["strike"]

        dte = int(pos["dte"]) if pd.notna(pos["dte"]) else 999
        per_contract_delta = pos["delta"] / (pos["quantity"] * 100) if pos["quantity"] else 0.0
        label = f"{pos['symbol']} ${pos['strike']:.2f} {pos['type']} exp {pos['expiration']} ({pos['side']} x{pos['quantity']:.0f})"

        if pos["avg_price"] <= 0:
            # risk_manager.Position's %P&L exit rules (profit target, stop
            # loss) are built for LONG-premium positions only — see its
            # module docstring. Robinhood reports average_price as negative
            # for short positions (a credit received, not a debit paid),
            # which fails pct_pnl's entry_premium > 0 requirement by design,
            # not because the data is missing. Skip rather than crash; a
            # zero avg_price (assignment/exercise/very old fills) hits this
            # same branch too, so the message covers both.
            reason = "it's a short position (risk_manager's %P&L rules are long-only)" if pos["avg_price"] < 0 else "Robinhood reports no entry price for it"
            st.warning(f"**{label}** — skipped: {reason}, so %P&L-based exit rules can't be evaluated.")
            continue

        position = risk_manager.Position(
            symbol=pos["symbol"],
            option_type=pos["type"],
            strike=pos["strike"],
            spot=spot,
            dte=dte,
            quantity=int(pos["quantity"]),
            entry_premium=pos["avg_price"],
            current_premium=pos["mark_price"],
            entry_iv=entry_iv,
            current_iv=pos["implied_volatility"],
            delta=per_contract_delta,
        )
        signals = risk_manager.evaluate_all_rules(position, config)
        triggered = [s for s in signals if s.triggered]

        if triggered:
            st.error(f"**{label}** — {len(triggered)} exit rule(s) triggered")
            for s in triggered:
                st.write(f"- **{s.rule}**: {s.reason}")
        else:
            st.success(f"**{label}** — no exit rules triggered")
        with st.expander("Full rule breakdown"):
            for s in signals:
                marker = "🔴" if s.triggered else "⚪"
                st.write(f"{marker} **{s.rule}**: {s.reason}")

    st.caption(
        "Entry IV is approximated as the IV first observed by this dashboard for each position — not the true "
        "trade-entry IV, which isn't persisted anywhere. The IV-crush exit is only meaningful once the dashboard "
        "has been watching a position continuously since you opened it."
    )

    st.markdown("###### Portfolio governors")
    net_delta_dollars = float(positions["delta"].sum())
    net_vega_dollars = float(positions["vega"].sum())
    daily_pnl = float(d["overview"].get("day_pl") or 0.0)

    c1, c2 = st.columns(2)
    max_delta_input = c1.number_input("Max net delta ($, 0 = no cap)", min_value=0.0, value=0.0, step=1000.0)
    max_vega_input = c2.number_input("Max net vega ($, 0 = no cap)", min_value=0.0, value=0.0, step=100.0)

    governor_config = RiskConfig(
        daily_max_loss_pct=config.daily_max_loss_pct,
        max_net_delta_dollars=max_delta_input if max_delta_input > 0 else None,
        max_net_vega_dollars=max_vega_input if max_vega_input > 0 else None,
    )
    state = risk_manager.PortfolioState(
        account_size=float(d["overview"].get("equity") or 1.0),
        daily_pnl=daily_pnl,
        net_delta_dollars=net_delta_dollars,
        net_vega_dollars=net_vega_dollars,
    )
    governor_result = risk_manager.check_portfolio_governors(state, governor_config)

    c1, c2, c3 = st.columns(3)
    c1.metric("Day P&L", f"${daily_pnl:,.2f}")
    c2.metric("Net delta ($)", f"{net_delta_dollars:,.0f}")
    c3.metric("Net vega ($)", f"{net_vega_dollars:,.2f}")
    if governor_result.halted:
        st.error("**New entries halted:**")
        for reason in governor_result.reasons:
            st.write(f"- {reason}")
    else:
        st.success("No portfolio governors triggered — new entries not halted.")


def _render_strategy_profile(
    legs: list, underlying_symbol: str, key_suffix: str,
    current_pl: float | None = None, dte: int | None = None, live_capable: bool = False,
):
    """Shared summary-metrics + payoff-chart renderer for the Strategy Payoff
    tab — used both for auto-detected open positions and the manual what-if
    builder below. When live_capable (every leg has a current IV) and dte is
    a real positive number, plots a second curve: mark-to-market P&L *today*
    (Black-Scholes-repriced at each leg's current IV and the real time
    remaining) alongside the plain at-expiration intrinsic-value payoff —
    those two are NOT the same thing before expiration, and time value is
    exactly the gap between them."""
    profile = option_strategy.analyze_strategy(legs)

    cols = st.columns(5 if current_pl is not None else 4)
    cols[0].metric("Max profit", money(profile.max_profit) if profile.max_profit is not None else "Unlimited")
    cols[1].metric("Max loss", money(profile.max_loss) if profile.max_loss is not None else "Unlimited")
    debit_label = "paid" if profile.net_debit >= 0 else "received"
    cols[2].metric("Net debit / credit", f"{money(abs(profile.net_debit))} ({debit_label})")
    cols[3].metric("Breakeven(s)", ", ".join(f"${b:,.2f}" for b in profile.breakevens) if profile.breakevens else "—")
    if current_pl is not None:
        pct_of_max = f" ({current_pl / profile.max_profit:+.0%} of max profit)" if profile.max_profit else None
        cols[4].metric("Current unrealized P&L (live, Robinhood)", money(current_pl), pct_of_max)

    if dte is not None:
        st.caption(f"{dte} day(s) to expiration.")

    spot = None
    if underlying_symbol:
        try:
            spot = data_fetch.get_stock_quote(underlying_symbol)["mark"]
        except Exception:
            spot = None

    strikes = [leg.strike for leg in legs]
    low_strike, high_strike = min(strikes), max(strikes)
    center = spot if spot else (low_strike + high_strike) / 2
    pad = max(high_strike - low_strike, center * 0.15, 5.0)
    x_min = max(0.0, min(low_strike, center) - pad)
    x_max = max(high_strike, center) + pad

    n_points = 200
    xs = [x_min + i * (x_max - x_min) / (n_points - 1) for i in range(n_points)]
    ys_expiry = [option_strategy.strategy_pl(legs, x) for x in xs]

    show_live = live_capable and dte is not None and dte > 0
    ys_live = None
    if show_live:
        T_years = dte / 365.0
        try:
            ys_live = [option_strategy.strategy_pl_today(legs, x, T_years) for x in xs]
        except ValueError:
            show_live = False

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys_expiry, mode="lines", line=dict(color=colors.CATEGORICAL[0], width=2.5), name="P&L at expiration"))
    if show_live:
        fig.add_trace(go.Scatter(x=xs, y=ys_live, mode="lines", line=dict(color=colors.CATEGORICAL[7], width=2.5), name="P&L today (live, current IV)"))
    fig.add_hline(y=0, line=dict(color=colors.INK_MUTED, dash="dash", width=1))
    for be in profile.breakevens:
        if x_min <= be <= x_max:
            fig.add_vline(x=be, line=dict(color=colors.STATUS_WARNING, dash="dot", width=1), annotation_text=f"BE ${be:,.2f}", annotation_position="top")
    if spot and x_min <= spot <= x_max:
        fig.add_vline(x=spot, line=dict(color=colors.INK_MUTED, dash="dash", width=1), annotation_text="Spot", annotation_position="bottom")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor=colors.SURFACE,
        paper_bgcolor=colors.SURFACE,
        xaxis=dict(title="Underlying price ($)", showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(title="P&L ($)", showgrid=True, gridcolor=colors.GRIDLINE, zerolinecolor=colors.INK_MUTED, color=colors.INK_MUTED, tickprefix="$"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if show_live else dict(),
        showlegend=show_live,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"strategy_chart_{key_suffix}")
    if show_live:
        st.caption(
            "Blue = P&L at expiration (intrinsic value only). Orange = P&L today, repriced via Black-Scholes at each "
            "leg's current implied vol and the real time remaining — this is the honest 'if I closed it right now' "
            "curve. They converge onto each other as expiration approaches and time value decays to zero."
        )
    else:
        st.caption(
            "P&L at expiration (intrinsic value only, no remaining time value) — the standard payoff-diagram convention. "
            "\"Unlimited\" means the payoff keeps moving in that direction indefinitely past the highest strike, not a "
            "very large but finite number."
        )


def render_strategy_payoff(d: dict):
    st.subheader("Multi-leg option strategy payoff")
    st.caption(
        "Max profit, max loss, and breakeven(s) for any combination of calls/puts on one underlying — exact, "
        "from the payoff's piecewise-linear shape at expiration, plus a live mark-to-market curve using each "
        "leg's current implied vol and real time remaining for your actual open positions."
    )

    opt = d["option_positions"]
    combos = []
    if not opt.empty:
        for (symbol, expiration), group in opt.groupby(["symbol", "expiration"]):
            if len(group) >= 2:
                combos.append((symbol, expiration, group))

    st.markdown("##### Your running trades")
    if not combos:
        st.caption("No open positions with 2+ legs on the same underlying/expiration — nothing to auto-detect right now.")
    for symbol, expiration, group in combos:
        legs = [
            option_strategy.OptionLeg(
                option_type=row["type"], strike=float(row["strike"]), premium=abs(float(row["avg_price"])),
                contracts=float(row["quantity"]) if row["side"] == "long" else -float(row["quantity"]),
                iv=float(row["implied_volatility"]) if pd.notna(row["implied_volatility"]) and row["implied_volatility"] > 0 else None,
            )
            for _, row in group.iterrows()
        ]
        leg_desc = " / ".join(
            f"{row['side']} {row['quantity']:.0f}x ${row['strike']:.2f}{row['type'][0].upper()}" for _, row in group.iterrows()
        )
        with st.expander(f"**{symbol}** exp {expiration} — {leg_desc}", expanded=True):
            current_pl = float(group["unrealized_pl"].sum())
            dte = int(group["dte"].iloc[0]) if pd.notna(group["dte"].iloc[0]) else None
            live_capable = all(leg.iv is not None for leg in legs)
            _render_strategy_profile(legs, symbol, key_suffix=f"{symbol}_{expiration}", current_pl=current_pl, dte=dte, live_capable=live_capable)

    st.divider()
    st.markdown("##### Explore a hypothetical strategy")
    with st.form("strategy_payoff_inputs"):
        underlying_symbol = st.text_input("Underlying symbol (centers the chart's price range)", value="").strip().upper()
        dte_input = st.number_input("Days to expiration (for the live curve)", min_value=0, value=30, step=1)
        n_legs = st.number_input("Number of legs", min_value=1, max_value=4, value=2, step=1)

        leg_inputs = []
        for i in range(int(n_legs)):
            st.markdown(f"###### Leg {i + 1}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            side = c1.selectbox("Side", ["Long", "Short"], key=f"leg_side_{i}")
            opt_type = c2.selectbox("Type", ["Call", "Put"], key=f"leg_type_{i}")
            strike = c3.number_input("Strike ($)", min_value=0.0, value=0.0, step=0.5, key=f"leg_strike_{i}")
            premium = c4.number_input("Premium ($/share)", min_value=0.0, value=0.0, step=0.01, key=f"leg_premium_{i}")
            contracts = c5.number_input("Contracts", min_value=1.0, value=1.0, step=1.0, key=f"leg_contracts_{i}")
            iv_pct = c6.number_input("IV (%)", min_value=0.0, value=30.0, step=1.0, key=f"leg_iv_{i}")
            leg_inputs.append((side, opt_type, strike, premium, contracts, iv_pct))

        submitted = st.form_submit_button("Analyze", type="primary")

    if not submitted:
        st.info("Set up your legs above and click Analyze.")
        return

    legs = [
        option_strategy.OptionLeg(
            option_type=opt_type.lower(), strike=strike, premium=premium,
            contracts=contracts if side == "Long" else -contracts,
            iv=(iv_pct / 100.0) if iv_pct > 0 else None,
        )
        for side, opt_type, strike, premium, contracts, iv_pct in leg_inputs
        if strike > 0
    ]
    if not legs:
        st.error("Enter at least one leg with a strike > 0.")
        return

    live_capable = all(leg.iv is not None for leg in legs)
    _render_strategy_profile(legs, underlying_symbol, key_suffix="manual", dte=int(dte_input), live_capable=live_capable)


def render_orders(d: dict):
    st.subheader("Open orders")
    open_orders = d["open_orders"]
    if open_orders.empty:
        st.caption("No open orders.")
    else:
        st.dataframe(open_orders, use_container_width=True, hide_index=True)

    st.subheader("Recent fills (last 90 days)")
    hist = d["order_history"]
    if hist.empty:
        st.caption("No recent fills.")
    else:
        st.dataframe(
            hist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "price": st.column_config.NumberColumn(format="$%.2f"),
                "amount": st.column_config.NumberColumn(format="$%.2f"),
                "fees": st.column_config.NumberColumn(format="$%.2f"),
            },
        )


def render_journal(d: dict):
    hist = d["order_history"]
    log = journal.build_journal(hist)
    st.subheader("Trade journal")
    if log.empty:
        st.caption("No filled trades in range to journal yet.")
        return

    st.dataframe(log, use_container_width=True, hide_index=True)
    csv_bytes = log.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"trade_journal_{pd.Timestamp.now():%Y%m%d}.csv",
        mime="text/csv",
        type="primary",
    )
    if st.button("Save a copy to exports/"):
        path = journal.export_csv(log)
        st.success(f"Saved to {path}")


def render_win_rate(d: dict):
    st.subheader("Win rate")
    st.caption(
        "Realized round-trip trades, FIFO-matched from your filled order history (same 90-day window as "
        "Orders & History). Equity matching is exact; options are matched per underlying symbol only, since "
        "fill history doesn't carry per-contract strike/expiration identity — if you hold multiple different "
        "contracts on the same underlying at once, fills across them can get cross-matched. Only **closed** "
        "trades count here; open positions with no matching exit aren't included."
    )

    trips_df = performance.round_trips_to_frame(performance.match_round_trips(d["order_history"]))
    if trips_df.empty:
        st.caption("No closed round-trip trades in the last 90 days yet.")
        return

    stats = performance.win_rate_stats(trips_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win rate", f"{stats['win_rate']:.1%}", f"{stats['wins']}W / {stats['losses']}L")
    c2.metric("Total realized P&L", money(stats["total_realized_pl"]))
    c3.metric("Avg win / avg loss", f"{money(stats['avg_win'] or 0)} / {money(stats['avg_loss'] or 0)}")
    c4.metric("Profit factor", f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "—")

    st.markdown("##### By instrument type")
    by_type = performance.win_rate_by_instrument_type(trips_df)
    st.dataframe(
        by_type,
        use_container_width=True,
        hide_index=True,
        column_config={
            "win_rate": st.column_config.NumberColumn(format="percent"),
            "total_realized_pl": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    st.markdown("##### Cumulative realized P&L")
    chart_df = trips_df.sort_values("exit_date")
    cum_pl = chart_df["realized_pl"].cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["exit_date"],
            y=cum_pl,
            mode="lines",
            line=dict(color=colors.CATEGORICAL[0], width=2),
            hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=colors.SURFACE,
        paper_bgcolor=colors.SURFACE,
        xaxis=dict(showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(showgrid=True, gridcolor=colors.GRIDLINE, zerolinecolor=colors.INK_MUTED, color=colors.INK_MUTED, tickprefix="$"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Closed trades")
    st.dataframe(
        trips_df.sort_values("exit_date", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "entry_price": st.column_config.NumberColumn(format="$%.2f"),
            "exit_price": st.column_config.NumberColumn(format="$%.2f"),
            "realized_pl": st.column_config.NumberColumn(format="$%.2f"),
            "holding_days": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def main():
    st.title("📉 Vol Trading Dashboard")

    if not ensure_logged_in():
        return

    top_l, top_r = st.columns([6, 1])
    with top_l:
        st.caption("Data refreshes every 60s while the app is open. Use Refresh for an immediate pull.")
    with top_r:
        if st.button("Refresh"):
            load_data.clear()
        if st.button("Log out"):
            logout()
            st.rerun()

    d = load_data()

    tabs = st.tabs(
        [
            "Overview", "Positions & Greeks", "Vol Exposure", "Vol Skew", "Risk Tool",
            "Strategy Payoff", "Orders & History", "Win Rate", "Journal / Export",
        ]
    )
    with tabs[0]:
        render_overview(d)
    with tabs[1]:
        render_positions(d)
    with tabs[2]:
        render_vol_exposure(d)
    with tabs[3]:
        render_vol_skew(d)
    with tabs[4]:
        render_risk_tool(d)
        st.divider()
        render_position_monitor(d)
    with tabs[5]:
        render_strategy_payoff(d)
    with tabs[6]:
        render_orders(d)
    with tabs[7]:
        render_win_rate(d)
    with tabs[8]:
        render_journal(d)


if __name__ == "__main__":
    main()
