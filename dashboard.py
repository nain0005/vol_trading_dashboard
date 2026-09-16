from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import colors, journal, oi_history, performance, vol_analysis
from app.auth import ensure_logged_in, is_demo_mode, logout
from risk_tool import hedge
from risk_tool import option_strategy
from risk_tool import options_lab
from risk_tool import portfolio_risk
from risk_tool import realized_vol as rv
from risk_tool import risk_manager
from risk_tool import sigma_moves
from risk_tool import sizing as risk_sizing
from risk_tool import spread_selection
from risk_tool import strike_selection
from risk_tool.config import DEFAULT_CONFIG, RiskConfig

if is_demo_mode():
    from app import demo_data as data_fetch  # same function names/shapes as data_fetch — synthetic, no live account
else:
    from app import data_fetch

st.set_page_config(page_title="Vol Trading Dashboard", page_icon="📉", layout="wide")


def inject_custom_css():
    """One CSS block, sourced entirely from app/colors.py tokens, that turns
    the default Streamlit chrome into something closer to a real trading
    terminal: a branded header, elevated metric tiles instead of bare
    numbers, an underlined active-tab indicator, and a monospace/tabular
    numeral face for anything that's actually a price or a Greek so columns
    of numbers line up the way they would in a real quote screen.

    Deliberately scoped to stable, widely-documented Streamlit selectors
    (data-testid attributes, baseweb tab parts) rather than deep structural
    hacks — those selectors have moved across major Streamlit versions
    before and will again, so anything here breaking should degrade to
    "looks like plain Streamlit," never to a broken layout.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        /* ---- App header ---- */
        .vt-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.9rem 1.4rem;
            margin: -1rem -1rem 1.2rem -1rem;
            background: linear-gradient(135deg, {colors.CHROME_DEEP} 0%, #0f1b2e 55%, #16324f 100%);
            border-bottom: 1px solid rgba(76,154,255,0.35);
            border-radius: 0 0 14px 14px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.45);
        }}
        .vt-header-left {{ display: flex; align-items: center; gap: 0.75rem; }}
        .vt-header-icon {{ font-size: 1.9rem; line-height: 1; }}
        .vt-header-title {{
            color: #fdfdfc; font-size: 1.28rem; font-weight: 700; letter-spacing: 0.01em; margin: 0;
        }}
        .vt-header-subtitle {{
            color: rgba(253,253,252,0.68); font-size: 0.8rem; font-weight: 400; margin: 0.1rem 0 0 0;
        }}
        .vt-header-badge {{
            font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
            padding: 0.3rem 0.65rem; border-radius: 999px; text-transform: uppercase; white-space: nowrap;
        }}
        .vt-header-badge.live {{ background: rgba(12,163,12,0.18); color: #7be07b; border: 1px solid rgba(12,163,12,0.4); }}
        .vt-header-badge.demo {{ background: rgba(250,178,25,0.18); color: #ffcf6b; border: 1px solid rgba(250,178,25,0.45); }}

        /* ---- Metric tiles ---- */
        [data-testid="stMetric"] {{
            background: {colors.SURFACE_RAISED};
            border: 1px solid {colors.GRIDLINE};
            border-radius: 10px;
            padding: 0.85rem 1rem 0.7rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: {colors.CATEGORICAL[0]};
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 1px {colors.CATEGORICAL[0]}, 0 6px 16px rgba(76,154,255,0.15);
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
            color: {colors.INK_MUTED};
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-weight: 600;
            color: {colors.INK_PRIMARY};
        }}
        [data-testid="stMetricDelta"] {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.2rem; border-bottom: 1px solid {colors.GRIDLINE};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 2.4rem; padding: 0 0.9rem; font-weight: 500; font-size: 0.88rem; color: {colors.INK_SECONDARY};
            border-radius: 8px 8px 0 0;
        }}
        .stTabs [data-baseweb="tab"]:hover {{ background: {colors.ACCENT_SOFT}; color: {colors.INK_PRIMARY}; }}
        .stTabs [aria-selected="true"] {{ color: {colors.CATEGORICAL[0]} !important; font-weight: 600; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {colors.CATEGORICAL[0]}; height: 2.5px; }}

        /* ---- Dataframes / tables: numeric alignment feel ---- */
        [data-testid="stDataFrame"] {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}

        /* ---- Expanders (methodology notes) ---- */
        [data-testid="stExpander"] {{
            border: 1px solid {colors.GRIDLINE}; border-radius: 10px; background: {colors.SURFACE_RAISED};
        }}

        /* ---- Buttons ---- */
        .stButton button {{ border-radius: 8px; font-weight: 500; }}
        .stButton button[kind="primary"] {{ background: {colors.CATEGORICAL[0]}; border-color: {colors.CATEGORICAL[0]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(demo: bool):
    badge_class = "demo" if demo else "live"
    badge_text = "Demo mode" if demo else "Live · Robinhood"
    st.markdown(
        f"""
        <div class="vt-header">
            <div class="vt-header-left">
                <span class="vt-header-icon">📉</span>
                <div>
                    <p class="vt-header-title">Vol Trading Dashboard</p>
                    <p class="vt-header-subtitle">Options &amp; volatility book — positions, Greeks, portfolio risk, and execution in one place</p>
                </div>
            </div>
            <span class="vt-header-badge {badge_class}">{badge_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


@st.cache_data(ttl=120, show_spinner=False)
def fetch_spot_prices(symbols: tuple) -> dict:
    prices = {}
    for sym in symbols:
        try:
            prices[sym] = data_fetch.get_underlying_price(sym)
        except Exception:
            continue
    return prices


_STRESS_SHOCKS_PCT = [-0.20, -0.15, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.15, 0.20]


def render_portfolio_risk(d: dict):
    st.subheader("Portfolio Risk")
    st.caption(
        "Correlation, Value-at-Risk, and stress tests across your WHOLE book at once. Every other tab looks at "
        "one position or one underlying at a time; this is the only place that asks 'if everything moves "
        "together, how much do I actually lose' — which is a different question from 'how risky is this one "
        "trade,' and usually the more important one once a book has more than a couple of names in it."
    )
    eq, opt = d["equity_positions"], d["option_positions"]
    underlyings = vol_analysis.book_underlyings(eq, opt)
    if len(underlyings) < 1:
        st.info("No open equity or option positions to analyze.")
        return

    with st.expander("Methodology / what this does and doesn't do", expanded=False):
        st.markdown(
            "- **Correlation/covariance** use daily log returns, inner-joined on date across every underlying "
            "you hold — a name with a short price history (e.g. a recent listing) shrinks the usable window for "
            "everyone, since a correlation matrix only means something if every pair is measured over the same "
            "dates.\n"
            "- **Net exposure** combines shares and option delta on the SAME underlying into one dollar figure "
            "before anything downstream sees it — a covered call correctly shows up as less risky than the same "
            "amount of naked stock, not as two unrelated bets.\n"
            "- **Parametric (delta-normal) VaR** assumes returns are jointly normal and that P&L moves linearly "
            "with each name's return (no gamma). Fast and smooth, but it understates tail risk for anything "
            "convex (any option) and understates how correlations spike toward 1 in a real crash.\n"
            "- **Historical VaR** replays your book's ACTUAL simulated P&L on each of the last N trading days — "
            "real fat tails and real historical co-movement come through, but it's still a linear, delta-only "
            "repricing per name, and it's only as good as the lookback window (a calm window understates risk; "
            "a crash-era window overstates it for a calm market).\n"
            "- **Neither VaR method captures options convexity for a large move.** That's what the stress test "
            "below is for: it reprices every option leg exactly via Black-Scholes at the shocked spot/IV, at the "
            "cost of only covering the specific scenarios you ask about instead of a continuous distribution.\n"
            "- All of this is a **market-wide, single-shock scenario** (every underlying moves by the same % at "
            "once) — it does not model one name gapping alone, or correlations breaking down, or your own "
            "positions' delta/gamma changing as time passes before the shock hits."
        )

    spot_by_symbol = fetch_spot_prices(tuple(underlyings))
    missing_spot = [s for s in underlyings if s not in spot_by_symbol]
    if missing_spot:
        st.warning(f"Couldn't get a live price for: {', '.join(missing_spot)} — excluded below.")

    histories = {}
    for sym in underlyings:
        if sym not in spot_by_symbol:
            continue
        try:
            hist = fetch_price_history(sym)
            if not hist.empty:
                histories[sym] = hist
        except Exception:
            continue

    returns = portfolio_risk.align_returns(histories)
    symbols_with_history = list(returns.columns)

    exposures = vol_analysis.book_exposures(eq, opt, spot_by_symbol)
    exposures_by_symbol = {e.symbol: e for e in exposures}

    st.markdown("##### Net exposure by underlying")
    st.caption("Shares and option delta on the same name netted together — dollar amount that moves 1:1 with a 100% move in that underlying.")
    if not exposures:
        st.caption("No net directional exposure (fully delta-neutral, or no positions priced).")
    else:
        exp_df = pd.DataFrame(
            [{"symbol": e.symbol, "net dollar delta": e.dollar_delta} for e in sorted(exposures, key=lambda e: -abs(e.dollar_delta))]
        )
        st.dataframe(
            exp_df, use_container_width=True, hide_index=True,
            column_config={"net dollar delta": st.column_config.NumberColumn(format="$%,.0f")},
        )

    st.divider()
    st.markdown("##### Correlation matrix (daily returns)")
    if len(symbols_with_history) < 2:
        st.caption("Need overlapping price history for at least 2 underlyings to build a correlation matrix.")
    else:
        corr = portfolio_risk.correlation_matrix(returns)
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0, colors.DIVERGING_NEG], [0.5, colors.DIVERGING_MID], [1, colors.DIVERGING_POS]],
                zmid=0, zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
                colorbar=dict(title="corr"),
                hovertemplate="%{y} vs %{x}<br>corr %{z:.2f}<extra></extra>",
            )
        )
        fig_corr.update_layout(
            height=max(280, 40 * len(corr)),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor=colors.SURFACE, paper_bgcolor=colors.SURFACE,
            xaxis=dict(color=colors.INK_MUTED), yaxis=dict(color=colors.INK_MUTED, autorange="reversed"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(f"{len(returns)} overlapping trading days used.")

    st.divider()
    st.markdown("##### Value-at-Risk")
    var_exposures = [exposures_by_symbol[s] for s in symbols_with_history if s in exposures_by_symbol]
    if len(var_exposures) < 1:
        st.caption("No priced net exposure with overlapping return history — can't estimate VaR.")
    else:
        vc1, vc2 = st.columns(2)
        confidence = vc1.select_slider("Confidence", options=[0.90, 0.95, 0.975, 0.99], value=0.95, key="var_confidence")
        horizon_days = vc2.slider("Horizon (trading days)", min_value=1, max_value=20, value=1, key="var_horizon")

        try:
            param = portfolio_risk.parametric_var(var_exposures, returns, confidence=confidence, horizon_days=horizon_days)
        except Exception as exc:
            param = None
            st.warning(f"Parametric VaR unavailable: {exc}")

        hist_result, hist_pnl = None, None
        try:
            hist_result, hist_pnl = portfolio_risk.historical_var(var_exposures, returns, confidence=confidence, horizon_days=horizon_days)
        except Exception as exc:
            st.caption(f"Historical VaR unavailable ({exc}) — needs at least 20 overlapping trading days.")

        m1, m2, m3, m4 = st.columns(4)
        if param:
            m1.metric(f"Parametric VaR ({confidence:.0%}, {horizon_days}d)", money(param.var_dollars))
            m2.metric(
                "Diversification ratio", f"{param.diversification_ratio:.0%}" if param.diversification_ratio is not None else "—",
                help="VaR / sum of each name's standalone VaR. Below 100% means correlation is netting risk down across your book; at/near 100% means your positions are effectively one correlated bet.",
            )
        if hist_result:
            m3.metric(f"Historical VaR ({confidence:.0%}, {horizon_days}d)", money(hist_result.var_dollars))
            ov_equity = d["overview"].get("equity") if d.get("overview") else None
            if ov_equity:
                m4.metric("Historical VaR, % of equity", f"{hist_result.var_dollars / ov_equity:.1%}")

        if hist_pnl is not None and not hist_pnl.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(x=hist_pnl, marker_color=colors.CATEGORICAL[0], nbinsx=30, hovertemplate="P&L $%{x:,.0f}<extra></extra>")
            )
            if hist_result:
                fig_hist.add_vline(x=-hist_result.var_dollars, line=dict(color=colors.STATUS_CRITICAL, dash="dash", width=2), annotation_text=f"{confidence:.0%} VaR")
            fig_hist.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor=colors.SURFACE, paper_bgcolor=colors.SURFACE,
                xaxis=dict(title="Simulated 1-day book P&L ($)", color=colors.INK_MUTED),
                yaxis=dict(showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, title="Days"),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Histogram of what your CURRENT book would have made/lost on each of the last N trading days, at today's exposures.")

    st.divider()
    st.markdown("##### Stress test — market-wide shock")
    st.caption(
        "Every underlying moves by the same % at once; every option leg is fully repriced via Black-Scholes "
        "(not a linear Greeks approximation) at the shocked spot and IV."
    )
    sc1, sc2 = st.columns(2)
    vol_pts_per_10pct_down = sc1.slider(
        "IV expansion on down moves (points per 10% the market drops)", min_value=0.0, max_value=15.0, value=4.0, step=0.5,
        help="Applied only to negative shocks — vol reliably rises in selloffs (the 'leverage effect' also discussed in the Risk Tool's EGARCH model), so a flat-IV down-move scenario is optimistic. Up moves are left at flat IV by default since post-selloff vol relief isn't nearly as reliable.",
    )
    config = DEFAULT_CONFIG

    rows = []
    total_skipped = 0
    for shock in _STRESS_SHOCKS_PCT:
        iv_shock = vol_pts_per_10pct_down * max(0.0, -shock) / 0.10
        result = vol_analysis.book_stress_pl(eq, opt, spot_by_symbol, shock, iv_shock_pts=iv_shock, r=config.risk_free_rate, q=config.dividend_yield)
        total_skipped = max(total_skipped, result["skipped"])
        rows.append(
            {
                "market move": shock, "IV shock (pts)": iv_shock,
                "equity P&L": result["equity_pl"], "option P&L": result["option_pl"], "total P&L": result["total_pl"],
            }
        )
    stress_df = pd.DataFrame(rows)

    fig_stress = go.Figure()
    bar_colors = [colors.DIVERGING_NEG if v < 0 else colors.DIVERGING_POS for v in stress_df["total P&L"]]
    fig_stress.add_trace(
        go.Bar(
            x=[f"{v:+.0%}" for v in stress_df["market move"]], y=stress_df["total P&L"], marker_color=bar_colors,
            hovertemplate="%{x}<br>Total P&L $%{y:,.0f}<extra></extra>",
        )
    )
    fig_stress.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=colors.SURFACE, paper_bgcolor=colors.SURFACE,
        xaxis=dict(title="Market move", showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(title="Total book P&L ($)", showgrid=True, gridcolor=colors.GRIDLINE, zerolinecolor=colors.INK_MUTED, color=colors.INK_MUTED),
        showlegend=False,
    )
    st.plotly_chart(fig_stress, use_container_width=True)
    st.dataframe(
        stress_df, use_container_width=True, hide_index=True,
        column_config={
            "market move": st.column_config.NumberColumn(format="percent"),
            "IV shock (pts)": st.column_config.NumberColumn(format="%.1f"),
            "equity P&L": st.column_config.NumberColumn(format="$%,.0f"),
            "option P&L": st.column_config.NumberColumn(format="$%,.0f"),
            "total P&L": st.column_config.NumberColumn(format="$%,.0f"),
        },
    )
    if total_skipped:
        st.caption(f"{total_skipped} position(s) skipped in the stress test (missing live price, IV, or DTE).")

    st.divider()
    st.markdown("##### Stress surface — spot move × IV shock (independent grid)")
    st.caption(
        "The bar chart above ties the IV shock to the spot shock via one slider (the leverage-effect "
        "assumption). This surface decouples them — every combination of spot move and IV shock, "
        "independently — same book_stress_pl repricing as above, swept over two axes instead of one, so you "
        "can see how much the total depends on vol shocking more or less than that fixed assumption."
    )
    gc1, gc2 = st.columns(2)
    spot_shock_max = gc1.slider("Spot shock range (±%)", min_value=5, max_value=40, value=25, step=5, key="stress_surface_spot_range") / 100.0
    # Slider is in "points" (percentage points, e.g. 15 -> IV +/- 15pp) for
    # readability; book_stress_pl's iv_shock_pts is added directly onto IV's
    # own 0-1 fraction scale (see option_leg_stress_pl / its test), so it
    # needs the /100 fraction form, not the raw slider value.
    iv_shock_max_points = gc2.slider("IV shock range (± points)", min_value=5, max_value=30, value=15, step=5, key="stress_surface_iv_range")
    iv_shock_max_frac = iv_shock_max_points / 100.0

    n_grid_points = 11
    spot_shocks = np.linspace(-spot_shock_max, spot_shock_max, n_grid_points)
    iv_shocks_frac = np.linspace(-iv_shock_max_frac, iv_shock_max_frac, n_grid_points)
    z = np.zeros((n_grid_points, n_grid_points))
    for i, iv_s in enumerate(iv_shocks_frac):
        for j, sp_s in enumerate(spot_shocks):
            grid_result = vol_analysis.book_stress_pl(
                eq, opt, spot_by_symbol, float(sp_s), iv_shock_pts=float(iv_s),
                r=config.risk_free_rate, q=config.dividend_yield,
            )
            z[i, j] = grid_result["total_pl"]

    zmax_surface = float(np.abs(z).max()) or 1.0
    fig_stress_surface = go.Figure(
        data=go.Surface(
            x=spot_shocks, y=iv_shocks_frac * 100.0, z=z,
            colorscale=[[0, colors.DIVERGING_NEG], [0.5, colors.DIVERGING_MID], [1, colors.DIVERGING_POS]],
            cmid=0, cmin=-zmax_surface, cmax=zmax_surface,
            colorbar=dict(title="P&L ($)"),
            hovertemplate="Spot move %{x:+.0%}<br>IV shock %{y:+.1f}pts<br>Total P&L $%{z:,.0f}<extra></extra>",
        )
    )
    fig_stress_surface.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor=colors.SURFACE,
        scene=dict(
            xaxis=dict(title="Spot move", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, tickformat=".0%"),
            yaxis=dict(title="IV shock (points)", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
            zaxis=dict(title="Total book P&L ($)", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
    )
    st.plotly_chart(fig_stress_surface, use_container_width=True)
    st.caption("Positive IV shock = vol expands; negative = vol contracts. Applied uniformly across every option leg.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history_span(symbol: str, span: str):
    return data_fetch.get_equity_historicals(symbol, span=span)


def render_sigma_screener(d: dict):
    st.subheader("Sigma move screener")
    st.caption(
        "For each ticker: how often has it actually moved 2σ+ or 3σ+ (relative to its own trailing rolling "
        "realized vol), and is it currently \"overdue\" by that history?"
    )
    with st.expander("Methodology — read this before trusting the ranking", expanded=False):
        st.markdown(
            "- **Sigma is a trailing rolling estimate** (20 trading days by default), shifted by one day so a "
            "day's own return never inflates the sigma used to judge it — no look-ahead.\n"
            "- **Two recurrence numbers, not one.** *Empirical* is the actual average gap between past "
            "exceedance days in this ticker's own history. *Model* comes from fitting a Student-t distribution "
            "to its z-scores (captures fat tails) and inverting the tail probability — more stable on names "
            "with very few historical exceedances, where the empirical average is noisy or undefined (shown "
            "as — when there are fewer than 2).\n"
            "- **⚠️ The 'overdue' ratio is NOT a forecast on its own.** If big moves were truly independent "
            "day to day, being overdue changes nothing going forward — the same logic as a coin not owing you "
            "tails after ten heads in a row. What legitimately CAN matter is the **vol compression ratio**: "
            "current rolling realized vol vs. this ticker's own full-sample average. A ratio well below 1 "
            "means vol has genuinely dropped, and volatility is mean-reverting — that's real evidence, unlike "
            "the raw day-count alone. Read the overdue ratio and vol compression together; the day count by "
            "itself is not a probability."
        )

    held_symbols = set()
    if not d["equity_positions"].empty:
        held_symbols |= set(d["equity_positions"]["symbol"])
    if not d["option_positions"].empty:
        held_symbols |= set(d["option_positions"]["symbol"])
    default_tickers = ", ".join(sorted(held_symbols)) if held_symbols else ", ".join(data_fetch.get_vol_tickers())

    with st.form("sigma_screener_inputs"):
        c1, c2, c3 = st.columns(3)
        tickers_input = c1.text_input("Tickers (comma-separated)", value=default_tickers)
        span = c2.selectbox(
            "Lookback", ["3month", "year", "5year"], index=2,
            format_func=lambda s: {"3month": "3 months", "year": "1 year", "5year": "5 years"}[s],
        )
        window = int(c3.number_input(
            "Rolling vol window (days)", min_value=5, max_value=60, value=sigma_moves.DEFAULT_ROLLING_WINDOW, step=5,
        ))
        submitted = st.form_submit_button("Run screener", type="primary")

    if not submitted:
        st.info("Enter tickers above and click Run screener.")
        return

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
        return

    rows = []
    skipped = []
    for ticker in tickers:
        try:
            hist = fetch_price_history_span(ticker, span)
        except Exception as exc:
            skipped.append(f"{ticker} ({exc})")
            continue
        if hist.empty:
            skipped.append(f"{ticker} (no price history)")
            continue

        close = hist["close"]
        s2 = sigma_moves.analyze_symbol(ticker, close, threshold=2.0, window=window)
        s3 = sigma_moves.analyze_symbol(ticker, close, threshold=3.0, window=window)
        if s2 is None or s3 is None:
            skipped.append(f"{ticker} (not enough history for a {window}-day rolling window)")
            continue

        rows.append(
            {
                "symbol": ticker,
                "days observed": s2.n_observations,
                "2σ count": s2.n_exceedances,
                "2σ empirical avg days": s2.empirical_avg_days_between,
                "2σ model avg days": s2.model_avg_days_between,
                "2σ days since last": s2.days_since_last_exceedance,
                "2σ overdue ratio": s2.overdue_ratio,
                "3σ count": s3.n_exceedances,
                "3σ empirical avg days": s3.empirical_avg_days_between,
                "3σ model avg days": s3.model_avg_days_between,
                "3σ days since last": s3.days_since_last_exceedance,
                "3σ overdue ratio": s3.overdue_ratio,
                "vol compression (recent/full)": s2.current_vol_ratio,
                "fitted t df (2σ fit)": s2.fitted_t_df,
            }
        )

    if skipped:
        st.caption("Skipped: " + "; ".join(skipped))
    if not rows:
        st.warning("No tickers produced usable results.")
        return

    table = pd.DataFrame(rows).sort_values("2σ overdue ratio", ascending=False, na_position="last")
    table["watch (overdue + compressed)"] = (table["2σ overdue ratio"].fillna(0) > 1) & (
        table["vol compression (recent/full)"].fillna(1) < 0.85
    )

    st.markdown("##### Screener results")
    st.caption(
        "Sorted by 2σ overdue ratio, descending — click a column header to re-sort. \"Watch\" flags rows where "
        "the overdue ratio AND vol compression agree (the combination with an actual evidence-based case "
        "behind it, per the methodology above), not the day count alone."
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "2σ empirical avg days": st.column_config.NumberColumn(format="%.1f"),
            "2σ model avg days": st.column_config.NumberColumn(format="%.1f"),
            "2σ overdue ratio": st.column_config.NumberColumn(format="%.2f"),
            "3σ empirical avg days": st.column_config.NumberColumn(format="%.1f"),
            "3σ model avg days": st.column_config.NumberColumn(format="%.1f"),
            "3σ overdue ratio": st.column_config.NumberColumn(format="%.2f"),
            "vol compression (recent/full)": st.column_config.NumberColumn(format="%.2f"),
            "fitted t df (2σ fit)": st.column_config.NumberColumn(
                format="%.1f",
                help="Lower = fatter tails (more prone to big moves than a normal distribution predicts). "
                "Below ~10 is a common rule of thumb for 'tails matter here.'",
            ),
        },
    )


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

    oi_history.log_snapshot(symbol, expiration, chain)

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

    st.markdown("##### Open interest × volume by strike")
    st.caption(
        "Open interest alone shows where positions have built up over time; volume alone only shows today's "
        "activity. Multiplying them highlights strikes with BOTH — a large existing book AND real trading "
        "today — rather than a strike that's merely old-and-static or merely noisy-today. This is a raw "
        "activity-intensity measure, not \"max pain\" (a different, specific calculation based on option "
        "writers' assignment cost) — don't conflate the two."
    )
    chain_activity = chain.copy()
    chain_activity["oi_volume"] = chain_activity["open_interest"] * chain_activity["volume"]
    calls_activity = chain_activity[chain_activity["type"] == "call"]
    puts_activity = chain_activity[chain_activity["type"] == "put"]

    fig_oi = go.Figure()
    fig_oi.add_trace(
        go.Bar(
            x=calls_activity["strike"], y=calls_activity["oi_volume"], name="Calls",
            marker=dict(color=colors.CATEGORICAL[0]),
            customdata=calls_activity[["open_interest", "volume"]],
            hovertemplate="Strike $%{x:.2f}<br>OI×Vol %{y:,.0f}<br>OI %{customdata[0]:,.0f} · Vol %{customdata[1]:,.0f}<extra>Call</extra>",
        )
    )
    fig_oi.add_trace(
        go.Bar(
            x=puts_activity["strike"], y=puts_activity["oi_volume"], name="Puts",
            marker=dict(color=colors.CATEGORICAL[5]),
            customdata=puts_activity[["open_interest", "volume"]],
            hovertemplate="Strike $%{x:.2f}<br>OI×Vol %{y:,.0f}<br>OI %{customdata[0]:,.0f} · Vol %{customdata[1]:,.0f}<extra>Put</extra>",
        )
    )
    if not held.empty:
        held = held.copy()
        held["oi_volume"] = held["open_interest"] * held["volume"]
    oi_marker = held_marker_trace("oi_volume", "OI×Vol", ",.0f")
    if oi_marker:
        fig_oi.add_trace(oi_marker)
    if spot and (not chain_activity.empty):
        fig_oi.add_vline(x=spot, line=dict(color=colors.INK_MUTED, dash="dash", width=1), annotation_text="Mark", annotation_position="top")
    if not chain_activity.empty and chain_activity["oi_volume"].max() > 0:
        top_row = chain_activity.loc[chain_activity["oi_volume"].idxmax()]
        fig_oi.add_vline(
            x=top_row["strike"], line=dict(color=colors.STATUS_GOOD, dash="dot", width=1.5),
            annotation_text=f"Most active ${top_row['strike']:,.2f}", annotation_position="bottom",
        )
    fig_oi.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        barmode="group",
        plot_bgcolor=colors.SURFACE,
        paper_bgcolor=colors.SURFACE,
        xaxis=dict(title="Strike", showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(title="Open interest × volume", showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_oi, use_container_width=True)

    oi_hist = oi_history.load_history(symbol, expiration)
    n_days_logged = oi_hist["date"].nunique() if not oi_hist.empty else 0

    st.markdown("##### Day-over-day change in open interest")
    st.caption(
        "Robinhood only exposes a current OI snapshot, not history — this builds up from a local log captured "
        "each time you view this ticker+expiration, starting today. It only grows on days you actually check "
        "back, so a gap can span more than one calendar day — that's shown per bar, not hidden."
    )
    if n_days_logged < 2:
        st.info(
            f"Logging started — {n_days_logged} day logged so far for {symbol} {expiration}. "
            "Come back another day (view this same ticker + expiration again) to see day-over-day changes."
        )
    else:
        changes = oi_history.daily_oi_change(oi_hist)
        latest_date = changes["date"].max()
        latest = changes[changes["date"] == latest_date].dropna(subset=["oi_change"])
        if latest.empty:
            st.caption("No comparable prior snapshot for any strike on the latest logged date yet.")
        else:
            gap_days = int(latest["days_since_prior"].iloc[0]) if pd.notna(latest["days_since_prior"].iloc[0]) else None
            gap_note = f" (spanning {gap_days} day(s) since the prior snapshot)" if gap_days and gap_days != 1 else ""
            st.caption(f"Latest logged date: {latest_date.date()}{gap_note}.")
            latest_calls = latest[latest["type"] == "call"]
            latest_puts = latest[latest["type"] == "put"]
            bar_colors_calls = [colors.DIVERGING_POS if v >= 0 else colors.DIVERGING_NEG for v in latest_calls["oi_change"]]
            bar_colors_puts = [colors.DIVERGING_POS if v >= 0 else colors.DIVERGING_NEG for v in latest_puts["oi_change"]]
            fig_change = go.Figure()
            fig_change.add_trace(
                go.Bar(
                    x=latest_calls["strike"], y=latest_calls["oi_change"], name="Calls",
                    marker=dict(color=bar_colors_calls),
                    customdata=latest_calls[["oi_pct_change"]],
                    hovertemplate="Strike $%{x:.2f}<br>Δ OI %{y:+,.0f}<br>%{customdata[0]:+.1%}<extra>Call</extra>",
                )
            )
            fig_change.add_trace(
                go.Bar(
                    x=latest_puts["strike"], y=latest_puts["oi_change"], name="Puts",
                    marker=dict(color=bar_colors_puts, opacity=0.65),
                    customdata=latest_puts[["oi_pct_change"]],
                    hovertemplate="Strike $%{x:.2f}<br>Δ OI %{y:+,.0f}<br>%{customdata[0]:+.1%}<extra>Put</extra>",
                )
            )
            fig_change.add_hline(y=0, line=dict(color=colors.INK_MUTED, width=1))
            if spot:
                fig_change.add_vline(x=spot, line=dict(color=colors.INK_MUTED, dash="dash", width=1), annotation_text="Mark", annotation_position="top")
            fig_change.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                barmode="group",
                plot_bgcolor=colors.SURFACE,
                paper_bgcolor=colors.SURFACE,
                xaxis=dict(title="Strike", showgrid=False, color=colors.INK_MUTED),
                yaxis=dict(title="Δ Open interest", showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, zerolinecolor=colors.INK_MUTED),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_change, use_container_width=True)
            st.caption("Calls shown solid, puts shown at reduced opacity — green = OI increased since the prior snapshot, red = decreased. Bar height is the raw contract count change; hover for the % change.")

    st.markdown("##### Open interest surface — strike × time")
    st.caption(
        "A true 'live' 3D view of how OI has actually moved, not a single-day snapshot — fills in as more days "
        f"get logged. {n_days_logged} day(s) logged for {symbol} {expiration} so far."
    )
    if n_days_logged < 2:
        st.info("Need at least 2 logged days to draw a surface — check back on another day this chain is viewed.")
    else:
        surface_type = st.radio("Side", ["call", "put"], key="oi_surface_type", horizontal=True)
        side_hist = oi_hist[oi_hist["type"] == surface_type]
        pivot = side_hist.pivot_table(index="date", columns="strike", values="open_interest", aggfunc="last")
        pivot = pivot.sort_index()
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            st.caption("Not enough (date × strike) coverage yet to render a surface for this side.")
        else:
            fig_surface = go.Figure(
                data=go.Surface(
                    x=pivot.columns,
                    y=pivot.index,
                    z=pivot.values,
                    colorscale=[[0, colors.SURFACE_RAISED], [0.5, colors.CATEGORICAL[1]], [1, colors.CATEGORICAL[0]]],
                    colorbar=dict(title="OI"),
                    hovertemplate="Strike $%{x:.2f}<br>%{y}<br>OI %{z:,.0f}<extra></extra>",
                )
            )
            fig_surface.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor=colors.SURFACE,
                scene=dict(
                    xaxis=dict(title="Strike", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
                    yaxis=dict(title="Date", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
                    zaxis=dict(title="Open interest", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
                ),
            )
            st.plotly_chart(fig_surface, use_container_width=True)

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

    st.divider()
    render_term_structure(symbol, expirations, spot)

    st.divider()
    render_vol_surface(symbol, expirations, spot)


@st.cache_data(ttl=3600, show_spinner="Pulling price history...")
def fetch_price_history(symbol: str):
    return data_fetch.get_equity_historicals(symbol)


def render_term_structure(symbol: str, expirations: list, spot: float):
    """ATM IV across expirations, benchmarked against trailing realized vol.

    Reuses fetch_chain (already cached) per expiration rather than a new data
    source — expensive part is one full chain fetch per expiration, so this
    is opt-in via a button/slider instead of running on every page load.
    """
    st.markdown("##### Term structure & IV vs. realized vol")
    st.caption(
        "ATM implied vol across expirations. Upward slope (contango) is the normal state; "
        "a front-month hump (backwardation) usually flags event risk (earnings, macro print) priced into "
        "the near-dated options. Dotted lines are trailing realized vol — where ATM IV sits above them, "
        "options are pricing more movement than has actually occurred recently."
    )

    max_available = min(12, len(expirations))
    n = st.slider(
        "Expirations to include", min_value=min(3, max_available), max_value=max_available,
        value=min(6, max_available), key="term_structure_n",
    )
    if st.button("Load term structure", key="load_term_structure"):
        today = date.today()
        rows = []
        for exp in expirations[:n]:
            try:
                chain = fetch_chain(symbol, exp)
            except Exception:
                continue
            if chain.empty:
                continue
            metrics = vol_analysis.skew_metrics(chain, spot)
            if metrics["atm_iv"] is None:
                continue
            dte = (pd.Timestamp(exp).date() - today).days
            rows.append({"expiration": exp, "dte": max(dte, 0), "atm_iv": metrics["atm_iv"]})
        st.session_state["term_structure_data"] = pd.DataFrame(rows)
        st.session_state["term_structure_symbol"] = symbol

    ts = st.session_state.get("term_structure_data")
    if ts is None or ts.empty or st.session_state.get("term_structure_symbol") != symbol:
        st.caption("Click **Load term structure** to pull ATM IV across the selected expirations.")
        return

    rv_20 = rv_60 = None
    try:
        hist = fetch_price_history(symbol)
        if not hist.empty:
            if len(hist) >= 21:
                rv_20 = rv.close_to_close_vol(hist["close"].tail(21))
            if len(hist) >= 61:
                rv_60 = rv.close_to_close_vol(hist["close"].tail(61))
    except Exception:
        pass

    ts = ts.sort_values("dte")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts["dte"], y=ts["atm_iv"], mode="lines+markers", name="ATM IV",
            line=dict(color=colors.CATEGORICAL[0], width=2), marker=dict(size=8),
            customdata=ts["expiration"],
            hovertemplate="%{customdata}<br>DTE %{x}<br>ATM IV %{y:.1%}<extra></extra>",
        )
    )
    if rv_20 is not None:
        fig.add_hline(
            y=rv_20, line=dict(color=colors.STATUS_GOOD, dash="dot", width=1.5),
            annotation_text="20d realized", annotation_position="right",
        )
    if rv_60 is not None:
        fig.add_hline(
            y=rv_60, line=dict(color=colors.INK_MUTED, dash="dot", width=1.5),
            annotation_text="60d realized", annotation_position="right",
        )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=colors.SURFACE,
        paper_bgcolor=colors.SURFACE,
        xaxis=dict(title="Days to expiration", showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(title="Implied volatility", showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, tickformat=".0%"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    if rv_20 is not None:
        front_iv = ts.iloc[0]["atm_iv"]
        spread = front_iv - rv_20
        c1, c2 = st.columns(2)
        c1.metric("Front-month ATM IV − 20d realized", f"{spread:+.1%}", help="Positive = options pricing more movement than has recently occurred (rich). Negative = cheap.")
        slope = ts.iloc[-1]["atm_iv"] - ts.iloc[0]["atm_iv"]
        c2.metric("Term structure slope (back − front)", f"{slope:+.1%}", help="Positive = contango (normal). Negative = backwardation (event risk priced near-term).")


def render_vol_surface(symbol: str, expirations: list, spot: float):
    """Full strike x expiration IV surface -- term structure above only
    tracks ATM IV; this is the same idea extended across every strike at
    once, the classic "vol surface" desks actually mean by that term.

    Reuses fetch_chain per expiration (already cached), same as
    render_term_structure just above -- so this is opt-in via its own
    button rather than automatic, for the same reason: N expirations
    means N full chain fetches.
    """
    st.markdown("##### Volatility surface — strike × expiration")
    st.caption(
        "IV at every strike across several expirations, not just ATM. Each expiration's curve uses the "
        "OTM-stitched convention (put IV below spot, call IV at/above — the more liquid, less exercise-noisy "
        "side) and is linearly interpolated onto one shared strike grid so expirations with different listed "
        "strikes still line up; a gap in the surface means that expiration didn't quote anything near that "
        "strike, not a real IV of zero."
    )
    max_available = min(12, len(expirations))
    n = st.slider(
        "Expirations to include", min_value=min(3, max_available), max_value=max_available,
        value=min(6, max_available), key="vol_surface_n",
    )
    if st.button("Load vol surface", key="load_vol_surface"):
        today = date.today()
        curves = {}
        for exp in expirations[:n]:
            try:
                chain = fetch_chain(symbol, exp)
            except Exception:
                continue
            if chain.empty:
                continue
            curve = vol_analysis.otm_iv_curve(chain, spot)
            dte = max((pd.Timestamp(exp).date() - today).days, 0)
            curves[exp] = (dte, curve)
        st.session_state["vol_surface_data"] = vol_analysis.vol_surface_grid(curves)
        st.session_state["vol_surface_symbol"] = symbol

    grid = st.session_state.get("vol_surface_data")
    if grid is None or st.session_state.get("vol_surface_symbol") != symbol:
        st.caption("Click **Load vol surface** to pull per-strike IV across the selected expirations.")
        return
    if grid.empty:
        st.info("Need at least 2 expirations with 2+ quoted OTM strikes each to build a surface — try including more expirations.")
        return

    pivot = grid.pivot_table(index="dte", columns="strike", values="iv", aggfunc="mean").sort_index()
    fig_surface = go.Figure(
        data=go.Surface(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.values,
            colorscale=[[0, colors.SURFACE_RAISED], [0.5, colors.CATEGORICAL[1]], [1, colors.CATEGORICAL[0]]],
            colorbar=dict(title="IV", tickformat=".0%"),
            hovertemplate="Strike $%{x:.2f}<br>DTE %{y}<br>IV %{z:.1%}<extra></extra>",
        )
    )
    fig_surface.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor=colors.SURFACE,
        scene=dict(
            xaxis=dict(title="Strike", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
            yaxis=dict(title="Days to expiration", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
            zaxis=dict(title="Implied volatility", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, tickformat=".0%"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
    )
    st.plotly_chart(fig_surface, use_container_width=True)
    st.caption(f"{grid['expiration'].nunique()} expiration(s), {pivot.shape[1]} strike points. Gaps are missing coverage, not zero IV.")


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
        vol_method = c8.selectbox(
            "Vol estimate for edge check (1yr history)",
            ["None (market IV only)", "Close-to-close (realized)", "GARCH(1,1) forecast", "EGARCH(1,1) forecast (asymmetric — weights recent down-moves more)"],
        )

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
    if vol_method != "None (market IV only)":
        try:
            hist = fetch_price_history(ticker)
            log_returns = np.log(hist["close"] / hist["close"].shift(1)).dropna()
            if vol_method == "Close-to-close (realized)":
                realized_or_forecast_vol = rv.close_to_close_vol(hist["close"])
            elif vol_method == "GARCH(1,1) forecast":
                garch_fit = rv.fit_garch_11(log_returns)
                realized_or_forecast_vol = rv.garch_forecast_vol(garch_fit, horizon_days=max(dte, 1))
            else:
                egarch_fit = rv.fit_egarch_11(log_returns)
                realized_or_forecast_vol = rv.egarch_forecast_vol(egarch_fit, horizon_days=max(dte, 1))
        except Exception as exc:
            st.warning(f"Couldn't compute {vol_method} ({exc}) — continuing with market IV only.")

    config = DEFAULT_CONFIG
    T = dte / 365.0
    strikes = strike_selection.generate_candidate_strikes(spot, strike_increment, config.num_strikes_each_side)

    if realized_or_forecast_vol:
        st.markdown("##### Realized-vs-implied edge check")
        spread = realized_or_forecast_vol - market_iv
        c1, c2, c3 = st.columns(3)
        c1.metric("Market IV", f"{market_iv:.1%}")
        c2.metric(vol_method, f"{realized_or_forecast_vol:.1%}")
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

    if live_capable and dte is not None and dte > 0:
        if st.button("Send to Options Lab →", key=f"send_to_lab_{key_suffix}"):
            st.session_state["lab_import"] = {
                "legs": [
                    {"option_type": leg.option_type, "strike": leg.strike, "premium": leg.premium, "contracts": leg.contracts, "iv": leg.iv}
                    for leg in legs
                ],
                "underlying_symbol": underlying_symbol,
                "dte": dte,
                "spot": spot,
            }
            st.success("Sent — open the **Options Lab** tab to explore this structure's P&L surface, Greeks, and scenarios.")
    elif not live_capable:
        st.caption("Options Lab needs every leg's current IV — not available for this setup (manual entries with IV = 0 are treated as unset).")


_SPREAD_STRATEGY_LABELS = {label: key for key, label in spread_selection.STRATEGY_LABELS.items()}


def render_spread_selector(d: dict):
    st.subheader("Spread & straddle strike selector")
    st.caption(
        "Searches strikes actually listed in the live chain and ranks candidates by comparing the market's "
        "price for the structure against YOUR OWN volatility view — same edge concept as the Risk Tool tab, "
        "extended to two legs. Read the methodology note below before trusting any ranking."
    )
    with st.expander("Methodology / what this does and doesn't do", expanded=False):
        st.markdown(
            "- Every candidate comes from strikes **actually listed** in the live option chain for the "
            "expiration you pick — not synthetic price levels — so it's something you could really place.\n"
            "- Max profit, max loss, and breakeven(s) are **exact**, from the payoff's piecewise-linear shape "
            "(same engine as the Strategy Payoff tab), not simulated.\n"
            "- **Edge EV** reprices every leg at your own realized/forecast vol instead of market IV, sums with "
            "the correct signs, and compares that 'fair value' to what you'd actually pay/receive at market "
            "prices. This is the only place real, quantifiable edge can come from here — it is **not** a "
            "directional prediction. With no vol view selected, candidates are ranked by risk:reward instead.\n"
            "- **Probability of profit** is a blended-IV approximation: it averages the two legs' implied vols "
            "into one assumed lognormal distribution, then measures the odds of finishing past the "
            "breakeven(s). Least reliable when the legs' IVs are far apart (e.g. a wide strangle across a "
            "steep skew).\n"
            "- For straddles/strangles specifically: **edge EV naturally shrinks toward deep ITM/OTM strikes**, "
            "since a leg with little extrinsic value left has little vega exposure to your vol view being "
            "right or wrong. The top-ranked candidate can end up being the one with the *least* vega at stake "
            "rather than the strongest actual view — sanity-check the strikes against where you'd realistically "
            "expect the move, not just the ranking.\n"
            "- Search is bounded to strikes within your chosen window of ATM, not the entire chain.\n"
            "- **Avg leg spread %** in the results table is a liquidity flag, not a pricing input — the average "
            "(ask-bid)/mid across the structure's legs. A mathematically great candidate on wide-spread strikes "
            "can cost more to actually enter/exit than its edge is worth; nothing here adjusts for that "
            "automatically, so treat a high number as a reason to check the live bid/ask before placing it.\n"
            "- **Compare all strategies** runs every structure above against the SAME already-fetched chain — "
            "no extra network calls, just more in-process payoff math — and shows each one's single best "
            "candidate side by side, so you can see which structure actually has the best setup right now "
            "instead of re-running this search once per strategy by hand."
        )

    with st.form("spread_selector_inputs"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", value=st.session_state.get("skew_symbol_input", ""), key="spread_selector_ticker").strip().upper()
        strategy_label = c2.selectbox("Strategy", list(_SPREAD_STRATEGY_LABELS.keys()), key="spread_selector_strategy")
        strike_increment = c3.number_input("Strike increment ($)", min_value=0.5, value=5.0, step=0.5, key="spread_selector_increment")

        c4, c5 = st.columns(2)
        num_each_side = c4.slider("Strikes to search, each side of ATM", min_value=2, max_value=15, value=6, key="spread_selector_width")
        vol_method = c5.selectbox(
            "Vol estimate for edge check (1yr history)",
            ["None (rank by risk:reward instead)", "Close-to-close (realized)", "GARCH(1,1) forecast", "EGARCH(1,1) forecast (asymmetric)"],
            key="spread_selector_vol_method",
        )
        compare_all = st.checkbox(
            "Also compare all strategies (best candidate from each, side by side)",
            key="spread_selector_compare_all",
        )
        submitted = st.form_submit_button("Find best strikes", type="primary")

    if not submitted:
        st.info("Pick a ticker and strategy above, then click Find best strikes.")
        return
    if not ticker:
        st.error("Enter a ticker.")
        return

    strategy = _SPREAD_STRATEGY_LABELS[strategy_label]

    try:
        expirations = fetch_expirations(ticker)
    except Exception as exc:
        st.error(f"Couldn't load expirations for {ticker}: {exc}")
        return
    if not expirations:
        st.warning(f"No option chain found for {ticker}. Check the ticker and that it has listed options.")
        return
    expiration = st.selectbox("Expiration", expirations, key="spread_selector_expiration")

    try:
        chain = fetch_chain(ticker, expiration)
    except Exception as exc:
        st.error(f"Couldn't load the option chain: {exc}")
        return
    if chain.empty:
        st.warning("No quoted contracts returned for this expiration.")
        return

    try:
        spot = data_fetch.get_stock_quote(ticker)["mark"]
    except Exception as exc:
        st.error(f"Couldn't get a live price for {ticker}: {exc}")
        return
    if not spot:
        st.error(f"Couldn't get a live price for {ticker}.")
        return

    dte = max((pd.Timestamp(expiration).date() - date.today()).days, 1)
    T = dte / 365.0
    config = DEFAULT_CONFIG

    my_vol = None
    if vol_method != "None (rank by risk:reward instead)":
        try:
            hist = fetch_price_history(ticker)
            log_returns = np.log(hist["close"] / hist["close"].shift(1)).dropna()
            if vol_method == "Close-to-close (realized)":
                my_vol = rv.close_to_close_vol(hist["close"])
            elif vol_method == "GARCH(1,1) forecast":
                garch_fit = rv.fit_garch_11(log_returns)
                my_vol = rv.garch_forecast_vol(garch_fit, horizon_days=dte)
            else:
                egarch_fit = rv.fit_egarch_11(log_returns)
                my_vol = rv.egarch_forecast_vol(egarch_fit, horizon_days=dte)
        except Exception as exc:
            st.warning(f"Couldn't compute {vol_method} ({exc}) — ranking by risk:reward instead.")

    candidates = spread_selection.find_best_spreads(
        strategy, chain, spot, T, config.risk_free_rate, config.dividend_yield,
        strike_increment, num_each_side=int(num_each_side), config=config, my_vol=my_vol,
    )
    if not candidates:
        st.warning("No valid candidates found in this strike window — try a wider search or a different expiration.")
        return

    st.markdown(f"##### Top candidates — {dte} DTE, {'ranked by edge EV' if my_vol else 'ranked by risk:reward'}")
    rows = [
        {
            "strikes": c.strike_label(),
            "net debit/credit": c.net_cost,
            "max profit": c.max_profit if c.max_profit is not None else float("inf"),
            "max loss": c.max_loss if c.max_loss is not None else float("-inf"),
            "risk:reward": c.risk_reward,
            "P(profit)": c.prob_profit,
            "edge EV": c.edge_ev,
            "breakeven(s)": ", ".join(f"${b:.2f}" for b in c.breakevens),
            "avg leg spread %": c.avg_spread_pct,
            "setup": "Poor (< min R:R)" if c.is_poor_setup else "OK",
        }
        for c in candidates
    ]
    table = pd.DataFrame(rows)
    column_config = {
        "net debit/credit": st.column_config.NumberColumn(format="$%+.2f", help="Per share. Positive = debit paid, negative = credit received."),
        "max profit": st.column_config.NumberColumn(format="$%.2f", help="Per share."),
        "max loss": st.column_config.NumberColumn(format="$%.2f", help="Per share."),
        "risk:reward": st.column_config.NumberColumn(format="%.2f:1"),
        "P(profit)": st.column_config.NumberColumn(format="percent"),
        "edge EV": st.column_config.NumberColumn(format="$%+.2f", help="Per share. Your-vol fair value minus market cost."),
        "avg leg spread %": st.column_config.NumberColumn(format="%.1f%%", help="Average (ask-bid)/mid across the structure's legs — a liquidity flag, not part of the edge/ranking math. Higher = more slippage to actually get in/out."),
    }
    if my_vol is None:
        table = table.drop(columns=["edge EV"])
        column_config.pop("edge EV")
    st.dataframe(table, use_container_width=True, hide_index=True, column_config=column_config)
    st.caption("$ figures are per share — multiply by 100 for per-contract dollars.")

    if my_vol:
        c1, c2 = st.columns(2)
        c1.metric(vol_method, f"{my_vol:.1%}")
        c2.metric("Best candidate's edge EV", f"${candidates[0].edge_ev:+.2f}/share")

    st.markdown("##### Best candidate — payoff diagram")
    best = candidates[0]
    legs_for_payoff = [
        option_strategy.OptionLeg(
            option_type=cl.option_type, strike=cl.strike, premium=cl.mid,
            contracts=1.0 if side == "long" else -1.0, iv=cl.iv,
        )
        for side, cl in best.legs
    ]
    live_capable = all(leg.iv is not None for leg in legs_for_payoff)
    _render_strategy_profile(
        legs_for_payoff, ticker, key_suffix=f"spread_selector_{ticker}_{expiration}", dte=dte, live_capable=live_capable,
    )

    if compare_all:
        st.divider()
        st.markdown("##### Compare all strategies — best candidate from each")
        st.caption(
            "Every structure run against this SAME chain (already fetched above, no extra network calls) — "
            "ranked the same way as the single-strategy table (edge EV if you picked a vol estimate, "
            "risk:reward otherwise). 'No candidate' means this strike window/chain didn't have a valid setup "
            "for that structure (e.g. an iron condor needs listed strikes on both sides of spot)."
        )
        comparison = spread_selection.compare_strategies(
            chain, spot, T, config.risk_free_rate, config.dividend_yield,
            strike_increment, num_each_side=int(num_each_side), config=config, my_vol=my_vol,
        )
        comp_rows = []
        for strat_key, cand in comparison.items():
            label = spread_selection.STRATEGY_LABELS[strat_key]
            if cand is None:
                comp_rows.append({
                    "strategy": label, "strikes": "—", "net debit/credit": None, "max profit": None,
                    "max loss": None, "risk:reward": None, "P(profit)": None, "edge EV": None, "avg leg spread %": None,
                })
            else:
                comp_rows.append({
                    "strategy": label,
                    "strikes": cand.strike_label(),
                    "net debit/credit": cand.net_cost,
                    "max profit": cand.max_profit if cand.max_profit is not None else float("inf"),
                    "max loss": cand.max_loss if cand.max_loss is not None else float("-inf"),
                    "risk:reward": cand.risk_reward,
                    "P(profit)": cand.prob_profit,
                    "edge EV": cand.edge_ev,
                    "avg leg spread %": cand.avg_spread_pct,
                })
        comp_table = pd.DataFrame(comp_rows)
        comp_column_config = {
            "net debit/credit": st.column_config.NumberColumn(format="$%+.2f", help="Per share."),
            "max profit": st.column_config.NumberColumn(format="$%.2f", help="Per share."),
            "max loss": st.column_config.NumberColumn(format="$%.2f", help="Per share."),
            "risk:reward": st.column_config.NumberColumn(format="%.2f:1"),
            "P(profit)": st.column_config.NumberColumn(format="percent"),
            "edge EV": st.column_config.NumberColumn(format="$%+.2f", help="Per share."),
            "avg leg spread %": st.column_config.NumberColumn(format="%.1f%%"),
        }
        if my_vol is None:
            comp_table = comp_table.drop(columns=["edge EV"])
            comp_column_config.pop("edge EV")
        rank_col = "edge EV" if my_vol else "risk:reward"
        if rank_col in comp_table.columns and comp_table[rank_col].notna().any():
            comp_table = comp_table.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)
        st.dataframe(comp_table, use_container_width=True, hide_index=True, column_config=comp_column_config)


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


def render_options_lab(d: dict):
    st.subheader("Options Lab")
    st.caption(
        "P&L across spot price and time, Greeks sensitivity, and shock scenarios for any option structure — "
        "build one from scratch, import an open position, or click **Send to Options Lab** on a Spread Selector "
        "or Strategy Payoff result."
    )
    with st.expander("Methodology / what this does and doesn't do", expanded=False):
        st.markdown(
            "- Every chart reprices via Black-Scholes at each leg's **own IV**, held fixed except where you "
            "explicitly shock it. There is no vol surface here — no skew that moves as spot moves, no smile — "
            "so treat this as *'what if price/vol move along this specific path,'* not a full forward-looking "
            "model of how IV itself would realistically shift.\n"
            "- The P&L grid's expiration row is **exact** intrinsic value (same engine as Strategy Payoff); "
            "every other cell is a Black-Scholes mark-to-market estimate for that point in time.\n"
            "- Greeks are aggregated across all legs, signed and multiplier-adjusted — a short leg's Greeks "
            "subtract, not add. Greeks are undefined (shown as zero) at/past expiration.\n"
            "- The earnings simulator applies the **same** relative IV crush to every leg — real post-earnings "
            "crush can differ leg-to-leg (front-week options typically crush harder than back-month)."
        )

    st.markdown("##### Strategy")
    source = st.radio(
        "Source", ["Build manually", "Import an open position", "Use last Send-to-Lab import"],
        key="lab_source", horizontal=True,
    )

    active = None
    if source == "Build manually":
        with st.form("lab_manual_inputs"):
            c1, c2, c3 = st.columns(3)
            m_symbol = c1.text_input("Underlying (for spot price + centering)", value=st.session_state.get("skew_symbol_input", "")).strip().upper()
            m_dte = int(c2.number_input("Days to expiration", min_value=1, value=30, step=1, key="lab_manual_dte"))
            n_legs = int(c3.number_input("Number of legs", min_value=1, max_value=4, value=2, step=1, key="lab_manual_nlegs"))
            leg_inputs = []
            for i in range(n_legs):
                st.markdown(f"###### Leg {i + 1}")
                lc1, lc2, lc3, lc4, lc5, lc6 = st.columns(6)
                side = lc1.selectbox("Side", ["Long", "Short"], key=f"lab_leg_side_{i}")
                opt_type = lc2.selectbox("Type", ["Call", "Put"], key=f"lab_leg_type_{i}")
                strike = lc3.number_input("Strike ($)", min_value=0.0, value=0.0, step=0.5, key=f"lab_leg_strike_{i}")
                premium = lc4.number_input("Premium ($/share)", min_value=0.0, value=0.0, step=0.01, key=f"lab_leg_premium_{i}")
                contracts = lc5.number_input("Contracts", min_value=1.0, value=1.0, step=1.0, key=f"lab_leg_contracts_{i}")
                iv_pct = lc6.number_input("IV (%)", min_value=0.1, value=30.0, step=1.0, key=f"lab_leg_iv_{i}")
                leg_inputs.append((side, opt_type, strike, premium, contracts, iv_pct))
            submitted = st.form_submit_button("Build", type="primary")
        if submitted:
            built_legs = [
                {"option_type": t.lower(), "strike": k, "premium": p, "contracts": (c if s == "Long" else -c), "iv": iv / 100.0}
                for s, t, k, p, c, iv in leg_inputs
                if k > 0
            ]
            if not built_legs:
                st.error("Enter at least one leg with a strike > 0.")
            else:
                st.session_state["lab_manual_strategy"] = {"legs": built_legs, "underlying_symbol": m_symbol, "dte": m_dte, "spot": None}
        active = st.session_state.get("lab_manual_strategy")
        if not active:
            st.info("Fill in the legs above and click Build.")
            return

    elif source == "Import an open position":
        opt = d["option_positions"]
        combos = []
        if not opt.empty:
            for (symbol, expiration), group in opt.groupby(["symbol", "expiration"]):
                combos.append((symbol, expiration, group))
        if not combos:
            st.caption("No open option positions to import.")
            return
        labels = [f"{symbol} exp {expiration} ({len(group)} leg(s))" for symbol, expiration, group in combos]
        idx = st.selectbox("Position", range(len(labels)), format_func=lambda i: labels[i], key="lab_position_select")
        symbol, expiration, group = combos[idx]
        built_legs = [
            {
                "option_type": row["type"], "strike": float(row["strike"]), "premium": abs(float(row["avg_price"])),
                "contracts": float(row["quantity"]) if row["side"] == "long" else -float(row["quantity"]),
                "iv": float(row["implied_volatility"]) if pd.notna(row["implied_volatility"]) and row["implied_volatility"] > 0 else None,
            }
            for _, row in group.iterrows()
        ]
        dte_val = int(group["dte"].iloc[0]) if pd.notna(group["dte"].iloc[0]) else 30
        if not all(leg["iv"] is not None for leg in built_legs):
            st.warning("One or more legs on this position are missing a live IV — Options Lab needs it on every leg.")
            return
        active = {"legs": built_legs, "underlying_symbol": symbol, "dte": dte_val, "spot": None}

    else:  # Use last Send-to-Lab import
        active = st.session_state.get("lab_import")
        if not active:
            st.info("Nothing sent yet — click **Send to Options Lab** on a result in Strategy Payoff or Spread Selector, or switch source above.")
            return
        st.caption(f"Loaded: {active['underlying_symbol']}, {len(active['legs'])} leg(s), {active['dte']} DTE.")

    legs = [option_strategy.OptionLeg(**leg) for leg in active["legs"]]
    underlying_symbol = active["underlying_symbol"]
    dte = active["dte"]

    spot = active.get("spot")
    if not spot and underlying_symbol:
        try:
            spot = data_fetch.get_stock_quote(underlying_symbol)["mark"]
        except Exception:
            spot = None
    if not spot:
        strikes = [leg.strike for leg in legs]
        spot = sum(strikes) / len(strikes)
        st.caption(f"No live quote for {underlying_symbol or 'this underlying'} — centering on the average strike (${spot:,.2f}) instead.")

    config = DEFAULT_CONFIG
    T_years = dte / 365.0

    st.divider()
    profile = option_strategy.analyze_strategy(legs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max profit", money(profile.max_profit) if profile.max_profit is not None else "Unlimited")
    c2.metric("Max loss", money(profile.max_loss) if profile.max_loss is not None else "Unlimited")
    debit_label = "paid" if profile.net_debit >= 0 else "received"
    c3.metric("Net debit / credit", f"{money(abs(profile.net_debit))} ({debit_label})")
    c4.metric("Breakeven(s)", ", ".join(f"${b:,.2f}" for b in profile.breakevens) if profile.breakevens else "—")

    st.markdown("##### Live what-if")
    st.caption("Drag to reprice instantly — the scenario point is marked on the P&L surface and Greeks curves below.")
    wc1, wc2, wc3 = st.columns(3)
    spot_move_pct = wc1.slider("Spot move (%)", -30.0, 30.0, 0.0, step=1.0, key="lab_whatif_spot") / 100.0
    days_forward = wc2.slider("Days forward", 0, dte, 0, step=1, key="lab_whatif_days")
    iv_shock_pct = wc3.slider("IV shock (%, relative)", -60.0, 60.0, 0.0, step=5.0, key="lab_whatif_iv") / 100.0

    scenario = options_lab.evaluate_scenario(
        legs, spot, dte, spot_move_pct=spot_move_pct, iv_mult=(1 + iv_shock_pct),
        days_forward=days_forward, r=config.risk_free_rate, q=config.dividend_yield,
    )
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Scenario spot", f"${scenario.spot:,.2f}")
    sc2.metric("Scenario P&L", money(scenario.pl))
    if scenario.greeks:
        sc3.metric("Delta", f"{scenario.greeks['delta']:+.2f}")
        sc4.metric("Theta/day", f"{scenario.greeks['theta']:+.2f}")
        sc5.metric("Vega", f"{scenario.greeks['vega']:+.2f}")
    else:
        sc3.metric("Delta", "—")
        sc4.metric("Theta/day", "—")
        sc5.metric("Vega", "—")
        st.caption("At/past expiration — Greeks aren't defined here.")

    st.markdown("##### P&L surface (spot × days forward)")
    st.caption("Row/floor 0 = right now; the last row = expiration (exact intrinsic value). The marker is your what-if scenario above.")
    pl_view = st.radio("View", ["3D surface", "Heatmap"], key="lab_pl_view", horizontal=True)
    spot_range_pct = st.slider("Spot range (± % around current)", min_value=0.05, max_value=0.50, value=0.20, step=0.05, key="lab_heatmap_range")
    grid = options_lab.pl_grid(legs, spot, dte, config.risk_free_rate, config.dividend_yield, spot_range_pct=spot_range_pct)
    pivot = grid.pivot(index="days_forward", columns="spot", values="pl")
    zmax = float(pivot.to_numpy().__abs__().max()) or 1.0

    if pl_view == "3D surface":
        fig_hm = go.Figure(
            data=go.Surface(
                x=pivot.columns, y=pivot.index, z=pivot.values,
                colorscale=[[0, colors.DIVERGING_NEG], [0.5, colors.DIVERGING_MID], [1, colors.DIVERGING_POS]],
                cmid=0, cmin=-zmax, cmax=zmax,
                colorbar=dict(title="P&L ($)"),
                hovertemplate="Spot $%{x:,.2f}<br>Days forward %{y}<br>P&L $%{z:,.0f}<extra></extra>",
            )
        )
        fig_hm.add_trace(
            go.Scatter3d(
                x=[scenario.spot], y=[scenario.days_forward], z=[scenario.pl], mode="markers",
                marker=dict(symbol="diamond", size=6, color="white", line=dict(color=colors.INK_PRIMARY, width=1)),
                name="Scenario", hovertemplate="Scenario<br>Spot $%{x:,.2f}<br>Days forward %{y}<br>P&L $%{z:,.0f}<extra></extra>",
            )
        )
        fig_hm.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor=colors.SURFACE,
            showlegend=False,
            scene=dict(
                xaxis=dict(title="Underlying price ($)", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
                yaxis=dict(title="Days forward", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
                zaxis=dict(title="P&L ($)", backgroundcolor=colors.SURFACE, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            ),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index,
                colorscale=[[0, colors.DIVERGING_NEG], [0.5, colors.DIVERGING_MID], [1, colors.DIVERGING_POS]],
                zmid=0, zmin=-zmax, zmax=zmax,
                colorbar=dict(title="P&L ($)"),
                hovertemplate="Spot $%{x:,.2f}<br>Days forward %{y}<br>P&L $%{z:,.0f}<extra></extra>",
            )
        )
        fig_hm.add_trace(
            go.Scatter(
                x=[scenario.spot], y=[scenario.days_forward], mode="markers",
                marker=dict(symbol="x", size=14, color="white", line=dict(color=colors.INK_PRIMARY, width=2)),
                name="Scenario", hovertemplate="Scenario<br>Spot $%{x:,.2f}<br>Days forward %{y}<extra></extra>",
            )
        )
        if spot:
            fig_hm.add_vline(x=spot, line=dict(color=colors.INK_MUTED, dash="dash", width=1), annotation_text="Spot now", annotation_position="top")
        fig_hm.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor=colors.SURFACE, paper_bgcolor=colors.SURFACE,
            xaxis=dict(title="Underlying price ($)", color=colors.INK_MUTED),
            yaxis=dict(title="Days forward from today", color=colors.INK_MUTED),
            showlegend=False,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("##### Greeks sensitivity (vs. spot, at current IV & time)")
    curve = options_lab.greeks_curve(legs, spot, T_years, config.risk_free_rate, config.dividend_yield, spot_range_pct=spot_range_pct)
    greek_colors = {"delta": colors.CATEGORICAL[0], "gamma": colors.CATEGORICAL[1], "theta": colors.CATEGORICAL[5], "vega": colors.CATEGORICAL[4]}
    grid_cols = st.columns(2)
    for i, greek in enumerate(("delta", "gamma", "theta", "vega")):
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=curve["spot"], y=curve[greek], mode="lines", line=dict(color=greek_colors[greek], width=2)))
        if spot:
            fig_g.add_vline(x=spot, line=dict(color=colors.INK_MUTED, dash="dash", width=1))
        fig_g.add_vline(x=scenario.spot, line=dict(color=colors.STATUS_WARNING, dash="dot", width=1.5))
        fig_g.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor=colors.SURFACE, paper_bgcolor=colors.SURFACE,
            title=dict(text=greek.capitalize(), font=dict(size=13)),
            xaxis=dict(showgrid=False, color=colors.INK_MUTED),
            yaxis=dict(showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED),
        )
        grid_cols[i % 2].plotly_chart(fig_g, use_container_width=True, key=f"lab_greek_{greek}")
    st.caption("Gray dashed line = current spot. Orange dotted line = your what-if scenario spot.")

    st.divider()
    st.markdown("##### Earnings / IV-crush simulator")
    st.caption(
        "Models a discrete event: a spot gap plus a vol crush (implied vol typically drops sharply right after "
        "an earnings print), evaluated the day after. Same repricing engine as the sliders above, framed as a "
        "one-shot event instead of a live drag."
    )
    ec1, ec2, ec3 = st.columns(3)
    earnings_spot_move = ec1.number_input("Expected spot move on the print (%)", min_value=-50.0, max_value=50.0, value=5.0, step=0.5, key="lab_earnings_spot") / 100.0
    iv_crush_pct = ec2.number_input("IV crush (%, relative drop)", min_value=0.0, max_value=95.0, value=40.0, step=5.0, key="lab_earnings_ivcrush")
    earnings_days_forward = int(ec3.number_input("Days forward (day after the print)", min_value=0, max_value=dte, value=min(1, dte), step=1, key="lab_earnings_days"))

    if st.button("Run earnings scenario", key="lab_run_earnings"):
        before = options_lab.evaluate_scenario(legs, spot, dte, days_forward=0, r=config.risk_free_rate, q=config.dividend_yield)
        after = options_lab.evaluate_scenario(
            legs, spot, dte, spot_move_pct=earnings_spot_move, iv_mult=(1 - iv_crush_pct / 100.0),
            days_forward=earnings_days_forward, r=config.risk_free_rate, q=config.dividend_yield,
        )
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Spot: before → after", f"${before.spot:,.2f} → ${after.spot:,.2f}")
        ac2.metric("P&L: before → after", f"{money(before.pl)} → {money(after.pl)}", money(after.pl - before.pl))
        vega_before = before.greeks.get("vega") if before.greeks else None
        ac3.metric(
            "Vega before the event", f"{vega_before:+.2f}" if vega_before is not None else "—",
            help="Positive vega positions lose the most from an IV crush; negative vega positions benefit.",
        )
        st.caption(
            f"Modeled crush: every leg's IV × {(1 - iv_crush_pct / 100.0):.2f} ({iv_crush_pct:.0f}% relative drop), "
            "applied uniformly across legs — see the methodology note above for why that's a simplification."
        )


_HEDGE_PRESETS = {
    "XLM (crypto)": ("XLM", "crypto"),
    "USO — oil ETF (Robinhood has no 'USOIL' ticker)": ("USO", "stock"),
    "SPY": ("SPY", "stock"),
    "Custom": (None, None),
}


def render_delta_hedge(d: dict):
    st.subheader("Delta hedge builder")
    st.caption(
        "Sizes a hedge in a *different* instrument (an ETF, a commodity proxy, a crypto) against a position's "
        "delta-equivalent exposure, scaled by that instrument's historical beta to the position's underlying — "
        "not a same-underlying options hedge. Useful when you want to offset directional risk with something "
        "liquid you can actually trade (e.g. hedge an oil-sensitive equity position with USO, or a small-cap "
        "growth book with XLM as a risk-appetite proxy)."
    )
    with st.expander("What this does and doesn't do", expanded=False):
        st.markdown(
            "- Offsets the **correlated portion** of directional (delta) risk only — not theta or vega on an "
            "options position.\n"
            "- Does **not** guarantee a profitable trade. The position can still lose money on its "
            "idiosyncratic move; **R²** below tells you how much of the move the hedge instrument actually "
            "explains — a low R² means this hedge is doing much less than the notional suggests.\n"
            "- **Decays over time.** Option delta moves with spot/time (gamma/theta), and beta is a rolling "
            "estimate that drifts — recompute and rebalance periodically, this isn't set-and-forget."
        )

    col_pos, col_hedge = st.columns(2)

    with col_pos:
        st.markdown("##### Position to hedge")
        position_ticker = st.text_input("Underlying ticker", key="hedge_pos_ticker", placeholder="e.g. AXP").strip().upper()
        position_mode = st.radio("Position type", ["Shares / ETF", "Option"], key="hedge_pos_mode", horizontal=True)
        if position_mode == "Shares / ETF":
            qty = st.number_input("Quantity (unsigned)", min_value=0.0, value=100.0, step=1.0, key="hedge_pos_qty")
            side = st.selectbox("Side", ["long", "short"], key="hedge_pos_side")
        else:
            delta = st.number_input(
                "Delta (signed — e.g. -0.45 for a long put, +0.30 for a short put)",
                value=0.45, step=0.01, format="%.2f", key="hedge_pos_delta",
            )
            contracts = st.number_input("Contracts (unsigned)", min_value=0.0, value=1.0, step=1.0, key="hedge_pos_contracts")
            contract_side = st.selectbox("Position side", ["long", "short"], key="hedge_pos_contract_side")

    with col_hedge:
        st.markdown("##### Hedge instrument")
        preset_label = st.selectbox("Quick pick", list(_HEDGE_PRESETS.keys()), key="hedge_preset")
        preset_ticker, preset_type = _HEDGE_PRESETS[preset_label]
        if preset_ticker:
            hedge_ticker = preset_ticker
            hedge_type = preset_type
            st.text_input("Hedge ticker", value=hedge_ticker, disabled=True, key="hedge_ticker_display")
        else:
            hedge_ticker = st.text_input("Hedge ticker", key="hedge_ticker_custom", placeholder="e.g. UNG, DBA, BTC").strip().upper()
            hedge_type = st.radio("Instrument type", ["stock", "crypto"], key="hedge_type_custom", horizontal=True)
        lookback_days = st.slider("Beta lookback window (trading days)", min_value=20, max_value=252, value=90, key="hedge_lookback")
        multiplier = st.number_input(
            "Hedge contract multiplier", min_value=0.0001, value=1.0, step=1.0, key="hedge_multiplier",
            help="1.0 for a plain ETF/stock/crypto hedge. Set to the units-per-contract if you're actually "
            "sizing a futures hedge (e.g. 1000 for CME WTI/CL, 100 for Micro WTI/MCL) with the hedge price "
            "quoted per unit.",
        )

    if not position_ticker or not hedge_ticker:
        st.caption("Enter both a position ticker and a hedge ticker to compute a hedge size.")
        return

    if not st.button("Calculate hedge", type="primary", key="hedge_calculate"):
        return

    try:
        underlying_hist = fetch_price_history(position_ticker)
    except Exception as exc:
        st.error(f"Couldn't load price history for {position_ticker}: {exc}")
        return
    try:
        hedge_hist = (
            data_fetch.get_crypto_historicals(hedge_ticker) if hedge_type == "crypto" else fetch_price_history(hedge_ticker)
        )
    except Exception as exc:
        st.error(f"Couldn't load price history for {hedge_ticker}: {exc}")
        return

    if underlying_hist.empty or len(underlying_hist) < 4:
        st.error(f"Not enough price history for {position_ticker} — check the ticker.")
        return
    if hedge_hist.empty or len(hedge_hist) < 4:
        st.error(f"Not enough price history for {hedge_ticker} — check the ticker (Robinhood may list it under a different symbol, e.g. USO instead of USOIL).")
        return

    underlying_close = underlying_hist.set_index("date")["close"].tail(lookback_days + 1)
    hedge_close = hedge_hist.set_index("date")["close"].tail(lookback_days + 1)
    underlying_returns = hedge.simple_returns(underlying_close)
    hedge_returns = hedge.simple_returns(hedge_close)

    try:
        beta_est = hedge.estimate_beta(underlying_returns, hedge_returns)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        underlying_price = data_fetch.get_stock_quote(position_ticker)["mark"] or float(underlying_close.iloc[-1])
    except Exception:
        underlying_price = float(underlying_close.iloc[-1])
    try:
        hedge_price = (
            data_fetch.get_crypto_quote(hedge_ticker)["mark"] if hedge_type == "crypto" else data_fetch.get_stock_quote(hedge_ticker)["mark"]
        ) or float(hedge_close.iloc[-1])
    except Exception:
        hedge_price = float(hedge_close.iloc[-1])

    if position_mode == "Shares / ETF":
        exposure_shares = hedge.share_position_exposure_shares(qty, side)
    else:
        signed_contracts = contracts if contract_side == "long" else -contracts
        exposure_shares = hedge.option_position_exposure_shares(delta, signed_contracts)

    result = hedge.size_hedge(exposure_shares, underlying_price, beta_est.beta, hedge_price, multiplier)

    st.markdown("##### Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beta (to hedge instrument)", f"{beta_est.beta:.2f}", help=f"OLS slope over {beta_est.n_obs} overlapping daily returns.")
    c2.metric("R²", f"{beta_est.r_squared:.1%}", help="Fraction of the position's return variance this hedge instrument explains. Low R² = a lot of unhedged idiosyncratic risk remains.")
    c3.metric("Delta-equivalent exposure", money(result.exposure_dollars))
    c4.metric("Hedge notional", money(result.hedge_dollars))

    action = "Buy / go long" if result.direction == "long" else "Sell / go short"
    unit_word = "contracts" if multiplier != 1.0 else "shares" if hedge_type == "stock" else "units"
    st.success(f"**{action} {abs(result.hedge_units):,.2f} {unit_word} of {hedge_ticker}** to hedge this position's correlated exposure.")

    if beta_est.r_squared < 0.15:
        st.warning(f"R² is only {beta_est.r_squared:.1%} — {hedge_ticker} explains very little of {position_ticker}'s recent moves. This hedge will leave most of the position's risk unhedged.")

    st.markdown("##### Indexed price comparison")
    idx = pd.DataFrame({
        position_ticker: vol_analysis.cumulative_return(underlying_close),
        hedge_ticker: vol_analysis.cumulative_return(hedge_close),
    }).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx.index, y=idx[position_ticker], mode="lines", name=position_ticker, line=dict(color=colors.CATEGORICAL[0], width=2)))
    fig.add_trace(go.Scatter(x=idx.index, y=idx[hedge_ticker], mode="lines", name=hedge_ticker, line=dict(color=colors.CATEGORICAL[5], width=2)))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=colors.SURFACE,
        paper_bgcolor=colors.SURFACE,
        xaxis=dict(showgrid=False, color=colors.INK_MUTED),
        yaxis=dict(title="Indexed return", showgrid=True, gridcolor=colors.GRIDLINE, color=colors.INK_MUTED, ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_orders(d: dict):
    st.subheader("Open orders")
    open_orders = d["open_orders"]
    if open_orders.empty:
        st.caption("No open orders.")
    else:
        st.dataframe(open_orders, use_container_width=True, hide_index=True)

    st.subheader("Recent fills (all-time)")
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
        "Realized round-trip trades, FIFO-matched **per exact contract** from your full filled order history "
        "(same all-time data as Orders & History) — each option fill is matched against its own specific "
        "strike/expiration/type, so holding multiple different contracts on the same underlying at once no "
        "longer risks cross-matching. Only **closed** trades count here; open positions with no matching exit "
        "aren't included."
    )

    trips_df = performance.round_trips_to_frame(performance.match_round_trips(d["order_history"]))
    if trips_df.empty:
        st.caption("No closed round-trip trades yet.")
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
    inject_custom_css()
    render_header(demo=is_demo_mode())

    if is_demo_mode():
        st.info(
            "🎭 **Demo mode** — every number on this page is synthetic (generated by `app/demo_data.py`), "
            "regenerated deterministically per session. This is not a real brokerage account, and no live "
            "trading connection exists in this deployment. The pricing/Greeks math itself (`risk_tool/`) is "
            "the real engine — only the input data is fake.",
            icon="🎭",
        )

    if not ensure_logged_in():
        return

    top_l, top_r = st.columns([6, 1])
    with top_l:
        st.caption("Data refreshes every 60s while the app is open. Use Refresh for an immediate pull.")
    with top_r:
        if st.button("Refresh"):
            load_data.clear()
        if not is_demo_mode() and st.button("Log out"):
            logout()
            st.rerun()

    d = load_data()

    tabs = st.tabs(
        [
            "Overview", "Positions & Greeks", "Vol Exposure", "Portfolio Risk", "Sigma Screener", "Vol Skew",
            "Risk Tool", "Spread Selector", "Strategy Payoff", "Options Lab", "Delta Hedge", "Orders & History",
            "Win Rate", "Journal / Export",
        ]
    )
    with tabs[0]:
        render_overview(d)
    with tabs[1]:
        render_positions(d)
    with tabs[2]:
        render_vol_exposure(d)
    with tabs[3]:
        render_portfolio_risk(d)
    with tabs[4]:
        render_sigma_screener(d)
    with tabs[5]:
        render_vol_skew(d)
    with tabs[6]:
        render_risk_tool(d)
        st.divider()
        render_position_monitor(d)
    with tabs[7]:
        render_spread_selector(d)
    with tabs[8]:
        render_strategy_payoff(d)
    with tabs[9]:
        render_options_lab(d)
    with tabs[10]:
        render_delta_hedge(d)
    with tabs[11]:
        render_orders(d)
    with tabs[12]:
        render_win_rate(d)
    with tabs[13]:
        render_journal(d)


if __name__ == "__main__":
    main()
