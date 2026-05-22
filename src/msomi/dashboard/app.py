"""Msomi Control Center — Streamlit dashboard."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure src/ is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from msomi.core.config import get_config
from msomi.core.database import init_db
from msomi.data.feeds import fetch_latest_price, fetch_ohlcv
from msomi.journal.logger import TradeJournal
from msomi.risk.circuit_breaker import CircuitBreaker
from msomi.signals.engine import SignalEngine

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Msomi",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Global dark polish */
    [data-testid="stAppViewContainer"] { background: #0e1117; }
    [data-testid="stSidebar"] { background: #141720; border-right: 1px solid #1f2235; }

    /* Signal cards */
    .sig-card {
        background: #141720;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #3b3f5c;
    }
    .sig-long  { border-left-color: #2ecc8a; }
    .sig-short { border-left-color: #e05a5a; }

    /* Price tiles */
    .price-tile {
        background: #141720;
        border: 1px solid #1f2235;
        border-radius: 10px;
        padding: 14px 12px;
        text-align: center;
    }
    .price-up   { color: #2ecc8a; }
    .price-down { color: #e05a5a; }

    /* Circuit breaker status */
    .cb-ok      { background:#0d2b1e; border:1px solid #2ecc8a; border-radius:8px; padding:12px 16px; }
    .cb-tripped { background:#2b0d0d; border:1px solid #e05a5a; border-radius:8px; padding:12px 16px; }

    /* Metric delta colour fix */
    [data-testid="stMetricDelta"] svg { display:none; }

    .stTabs [data-baseweb="tab"] { color: #8890a8; }
    .stTabs [aria-selected="true"] { color: #c9a84c; border-bottom-color: #c9a84c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Init ─────────────────────────────────────────────────────────────────────


@st.cache_resource
def _init():
    cfg = get_config()
    init_db(cfg.data.db_url)
    journal = TradeJournal()
    engine = SignalEngine(cfg)
    breaker = CircuitBreaker(
        account_balance=cfg.account.balance,
        daily_loss_limit_pct=cfg.risk.daily_loss_limit_pct,
        max_consecutive_losses=cfg.risk.max_consecutive_losses,
    )
    return cfg, journal, engine, breaker


cfg, journal, signal_engine, circuit_breaker = _init()

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _score_color(score: int) -> str:
    if score >= 75:
        return "#2ecc8a"
    if score >= 60:
        return "#c9a84c"
    return "#8890a8"


def _candlestick_chart(symbol: str, timeframe: str = "1h", periods: int = 120) -> go.Figure:
    """Fetch OHLCV and return a dark-theme candlestick figure."""
    try:
        df = fetch_ohlcv(symbol, timeframe=timeframe, periods=periods)
    except Exception:
        return go.Figure()

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#2ecc8a",
            decreasing_line_color="#e05a5a",
            increasing_fillcolor="#2ecc8a",
            decreasing_fillcolor="#e05a5a",
            name=symbol,
        )
    ])

    # Add volume as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["volume"],
        marker_color=["#2ecc8a" if c >= o else "#e05a5a"
                      for c, o in zip(df["close"], df["open"])],
        opacity=0.3,
        yaxis="y2",
        name="Volume",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        title=dict(text=f"{symbol} · {timeframe}", font=dict(size=14, color="#c9a84c")),
        yaxis=dict(showgrid=True, gridcolor="#1f2235", side="right"),
        yaxis2=dict(overlaying="y", side="left", showgrid=False, showticklabels=False, range=[0, df["volume"].max() * 6]),
        xaxis=dict(showgrid=True, gridcolor="#1f2235"),
    )
    return fig


def _equity_fig(initial_balance: float) -> go.Figure:
    eq_df = journal.equity_curve(initial_balance)
    fig = go.Figure()
    if eq_df.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=eq_df["opened_at"],
        y=eq_df["balance"],
        mode="lines",
        line=dict(color="#c9a84c", width=2),
        fill="tozeroy",
        fillcolor="rgba(201,168,76,0.07)",
        name="Balance",
    ))
    # Colour area above/below starting balance
    fig.add_hline(
        y=initial_balance,
        line_dash="dot",
        line_color="#3b3f5c",
        annotation_text=f"Start ${initial_balance:,.0f}",
        annotation_font_color="#8890a8",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#1f2235"),
        yaxis=dict(showgrid=True, gridcolor="#1f2235", side="right"),
    )
    return fig


def _circuit_breaker_widget():
    state = circuit_breaker.state
    if state.is_tripped:
        st.markdown(
            f'<div class="cb-tripped">⛔ <strong>Circuit Breaker TRIPPED</strong> — {state.reason}<br/>'
            f'<small>Daily P&L: <strong>${state.daily_pnl:+.2f}</strong> '
            f'({state.daily_loss_pct*100:.1f}%)</small></div>',
            unsafe_allow_html=True,
        )
    else:
        streak = state.consecutive_losses
        streak_warn = f"⚠️ {streak}/{circuit_breaker.max_streak} loss streak" if streak > 0 else "✅ Clear"
        st.markdown(
            f'<div class="cb-ok">🟢 <strong>Circuit Breaker OK</strong> — {streak_warn}<br/>'
            f'<small>Daily P&L: <strong>${state.daily_pnl:+.2f}</strong> '
            f'· {state.trades_today} trade(s) today</small></div>',
            unsafe_allow_html=True,
        )


# ─── Page implementations ────────────────────────────────────────────────────


def _page_live_feed():
    st.title("📡 Live Feed")

    # Top KPI row
    perf_7d = journal.performance_summary(days=7)
    perf_30d = journal.performance_summary(days=30)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Trades (7d)", perf_7d.get("total_trades", 0))
    k2.metric(
        "Win Rate (7d)",
        f"{perf_7d.get('win_rate', 0)*100:.0f}%",
        delta=f"{(perf_7d.get('win_rate', 0) - perf_30d.get('win_rate', 0))*100:+.0f}% vs 30d",
    )
    k3.metric(
        "P&L (7d)",
        f"${perf_7d.get('total_pnl', 0):+.2f}",
    )
    k4.metric("Profit Factor (30d)", f"{perf_30d.get('profit_factor', 0):.2f}")
    k5.metric("Streak", f"{perf_30d.get('total_trades',0)} trades")

    st.divider()

    # Circuit breaker + equity side by side
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Risk Status")
        _circuit_breaker_widget()
        st.markdown("")

        st.subheader("Account")
        rc = cfg.risk
        st.markdown(
            f"**Balance:** ${cfg.account.balance:,.2f} {cfg.account.currency}  \n"
            f"**Risk/Trade:** {rc.per_trade_pct*100:.1f}%  \n"
            f"**Daily Limit:** {rc.daily_loss_limit_pct*100:.1f}%  \n"
            f"**Max Streak:** {rc.max_consecutive_losses} losses  \n"
            f"**Min R:R:** {rc.min_risk_reward}:1"
        )

    with right:
        st.subheader("Equity Curve")
        fig = _equity_fig(cfg.account.balance)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet — equity curve will appear here.")

    st.divider()

    # Live prices — all watchlist symbols
    st.subheader("Watchlist Prices")
    forex_syms = cfg.watchlist.forex
    crypto_syms = cfg.watchlist.crypto

    st.caption("Forex")
    fcols = st.columns(len(forex_syms))
    for i, sym in enumerate(forex_syms):
        price = fetch_latest_price(sym)
        with fcols[i]:
            label = sym.replace("=X", "")
            st.markdown(
                f'<div class="price-tile"><strong>{label}</strong><br/>'
                f'<span class="price-up">{price:.5f}</span></div>' if price else
                f'<div class="price-tile"><strong>{label}</strong><br/><span style="color:#8890a8">N/A</span></div>',
                unsafe_allow_html=True,
            )

    st.caption("Crypto")
    ccols = st.columns(len(crypto_syms))
    for i, sym in enumerate(crypto_syms):
        price = fetch_latest_price(sym)
        with ccols[i]:
            label = sym.replace("-USD", "")
            st.markdown(
                f'<div class="price-tile"><strong>{label}</strong><br/>'
                f'<span class="price-up">${price:,.2f}</span></div>' if price else
                f'<div class="price-tile"><strong>{label}</strong><br/><span style="color:#8890a8">N/A</span></div>',
                unsafe_allow_html=True,
            )

    # Auto-refresh toggle
    st.divider()
    if st.button("🔄 Refresh Prices", use_container_width=False):
        st.rerun()


def _page_charts():
    st.title("📈 Charts")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Symbol", cfg.watchlist.all_symbols, key="chart_sym")
    with col2:
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1, key="chart_tf")
    with col3:
        periods = st.selectbox("Candles", [60, 120, 200, 500], index=1, key="chart_periods")

    with st.spinner(f"Loading {symbol} [{timeframe}]…"):
        fig = _candlestick_chart(symbol, timeframe, periods)

    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load chart data.")

    # Indicator snapshot below chart
    with st.spinner("Computing indicators…"):
        snap = signal_engine.snapshot_only(symbol, timeframe=timeframe)

    if snap:
        st.subheader("Indicator Snapshot")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trend", snap.trend_direction)
        c2.metric("RSI", f"{snap.rsi:.1f}", snap.rsi_signal)
        c3.metric("MACD Hist", f"{snap.macd_hist:.5f}",
                  "↑ bullish" if snap.macd_crossover > 0 else ("↓ bearish" if snap.macd_crossover < 0 else "–"))
        c4.metric("BB Position", snap.bb_position)
        c5.metric("Vol Ratio", f"{snap.volume_ratio:.2f}×",
                  "Spike 🔥" if snap.volume_spike else None)
        c6.metric("ATR%", f"{snap.atr_pct*100:.2f}%")

        # RSI gauge
        rsi_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=snap.rsi,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "RSI", "font": {"color": "#c9a84c"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8890a8"},
                "bar": {"color": "#c9a84c"},
                "bgcolor": "#141720",
                "bordercolor": "#1f2235",
                "steps": [
                    {"range": [0, 30], "color": "#0d2b1e"},
                    {"range": [30, 70], "color": "#141720"},
                    {"range": [70, 100], "color": "#2b0d0d"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": snap.rsi},
            },
        ))
        rsi_fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font={"color": "#8890a8"},
            height=200,
            margin=dict(l=20, r=20, t=30, b=0),
        )

        ga, gb, gc = st.columns(3)
        with ga:
            st.plotly_chart(rsi_fig, use_container_width=True)

        # MACD mini chart
        try:
            df_full = fetch_ohlcv(symbol, timeframe=timeframe, periods=periods)
            from msomi.signals.indicators import IndicatorEngine
            ind = IndicatorEngine()
            df_full = ind.compute(df_full)
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Bar(
                x=df_full.index[-60:],
                y=df_full["macd_hist"].iloc[-60:],
                marker_color=["#2ecc8a" if v >= 0 else "#e05a5a" for v in df_full["macd_hist"].iloc[-60:]],
                name="MACD Histogram",
            ))
            macd_fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=200, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                title=dict(text="MACD Histogram", font=dict(color="#c9a84c", size=12)),
                xaxis=dict(showgrid=True, gridcolor="#1f2235"),
                yaxis=dict(showgrid=True, gridcolor="#1f2235"),
            )
            with gb:
                st.plotly_chart(macd_fig, use_container_width=True)

            # Volume chart
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Bar(
                x=df_full.index[-60:],
                y=df_full["volume"].iloc[-60:],
                marker_color=["#2ecc8a" if c >= o else "#e05a5a"
                              for c, o in zip(df_full["close"].iloc[-60:], df_full["open"].iloc[-60:])],
                name="Volume",
            ))
            vol_fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=200, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                title=dict(text="Volume", font=dict(color="#c9a84c", size=12)),
                xaxis=dict(showgrid=True, gridcolor="#1f2235"),
                yaxis=dict(showgrid=True, gridcolor="#1f2235"),
            )
            with gc:
                st.plotly_chart(vol_fig, use_container_width=True)
        except Exception:
            pass


def _page_signals():
    st.title("🎯 Signals")

    tab1, tab2 = st.tabs(["Recent Signals", "Scan Now"])

    with tab1:
        signals = journal.recent_signals(limit=30)
        if not signals:
            st.info("No signals fired yet. Run `msomi scan` or use the Scan Now tab.")
        else:
            for s in signals:
                direction = s.get("direction", "")
                score = s.get("confidence_score", 0) or 0
                entry = s.get("entry_price", 0) or 0
                sl = s.get("stop_loss", 0) or 0
                tp = s.get("take_profit", 0) or 0
                rr = s.get("risk_reward", 0) or 0
                ts = ""
                if s.get("created_at"):
                    try:
                        ts = datetime.fromisoformat(s["created_at"]).strftime("%b %d %H:%M")
                    except Exception:
                        ts = str(s["created_at"])[:16]

                card_class = "sig-long" if direction == "LONG" else "sig-short"
                dir_color = "#2ecc8a" if direction == "LONG" else "#e05a5a"
                dir_arrow = "▲" if direction == "LONG" else "▼"
                sc = _score_color(score)
                status = s.get("status", "")

                st.markdown(
                    f"""
                    <div class="sig-card {card_class}">
                      <div style="display:flex; justify-content:space-between; align-items:center">
                        <span><strong style="font-size:1.05em">{s.get('symbol','')}</strong>
                        &nbsp;<span style="color:#8890a8">[{s.get('timeframe','')}]</span>
                        &nbsp;<span style="color:{dir_color}">{dir_arrow} {direction}</span></span>
                        <span>
                          <span style="background:{sc}22;color:{sc};padding:2px 8px;border-radius:5px;font-size:0.85em">
                            Score {score}/100</span>
                          &nbsp;<span style="color:#8890a8;font-size:0.8em">{ts}</span>
                          &nbsp;<span style="color:#3b3f5c;font-size:0.8em">{status}</span>
                        </span>
                      </div>
                      <div style="margin-top:8px; font-size:0.9em; color:#8890a8">
                        Entry <code style="color:#e0e0e0">{entry:.5f}</code>
                        &nbsp;·&nbsp; SL <code style="color:#e05a5a">{sl:.5f}</code>
                        &nbsp;·&nbsp; TP <code style="color:#2ecc8a">{tp:.5f}</code>
                        &nbsp;·&nbsp; R:R <code style="color:#c9a84c">{rr:.2f}</code>
                        &nbsp;·&nbsp; RSI <code>{s.get('rsi') or '–'}</code>
                        &nbsp;·&nbsp; Trend <code>{s.get('ema_trend') or '–'}</code>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if s.get("ai_analysis"):
                    with st.expander(f"🤖 AI Analysis — {s.get('symbol','')}"):
                        st.write(s["ai_analysis"])

    with tab2:
        st.subheader("On-Demand Scan")
        col1, col2, col3 = st.columns(3)
        with col1:
            scan_symbol = st.selectbox("Symbol", cfg.watchlist.all_symbols, key="scan_sym")
        with col2:
            scan_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1, key="scan_tf")
        with col3:
            st.markdown("<br/>", unsafe_allow_html=True)
            run_scan = st.button("🔍 Scan", type="primary", use_container_width=True)

        if run_scan:
            with st.spinner(f"Scanning {scan_symbol} [{scan_tf}]…"):
                event = signal_engine.evaluate_symbol(scan_symbol, timeframe=scan_tf)

            if event:
                s = event.signal
                dir_color = "#2ecc8a" if s.direction == "LONG" else "#e05a5a"
                st.markdown(
                    f"<h3 style='color:{dir_color}'>{'▲' if s.direction=='LONG' else '▼'} "
                    f"{s.direction} Signal — Score {s.score}/100</h3>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Entry", f"{s.entry_price:.5f}")
                c2.metric("Stop Loss", f"{s.stop_loss:.5f}")
                c3.metric("Take Profit", f"{s.take_profit:.5f}")
                c4.metric("R:R", f"{s.risk_reward:.2f}:1")

                st.markdown("**Confluence reasons:**")
                for reason in s.reasons:
                    st.write(f"• {reason}")

                # Score breakdown bar chart
                if s.components:
                    comp_fig = go.Figure(go.Bar(
                        x=list(s.components.values()),
                        y=list(s.components.keys()),
                        orientation="h",
                        marker_color="#c9a84c",
                    ))
                    comp_fig.update_layout(
                        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        height=220, margin=dict(l=0, r=0, t=10, b=0),
                        title=dict(text="Score Breakdown", font=dict(color="#c9a84c", size=12)),
                        xaxis=dict(range=[0, max(s.components.values()) * 1.2]),
                    )
                    st.plotly_chart(comp_fig, use_container_width=True)
            else:
                st.info(f"No signal above threshold ({cfg.signals.min_confidence_score}) for {scan_symbol} [{scan_tf}].")

            snap = signal_engine.snapshot_only(scan_symbol, timeframe=scan_tf)
            if snap:
                st.divider()
                st.subheader("Indicator Snapshot")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Trend", snap.trend_direction)
                c2.metric("RSI", f"{snap.rsi:.1f}", snap.rsi_signal)
                c3.metric("MACD Hist", f"{snap.macd_hist:.5f}")
                c4.metric("BB Position", snap.bb_position)
                c5.metric("Volume Ratio", f"{snap.volume_ratio:.2f}×")


def _page_journal():
    st.title("📚 Trade Journal")

    perf = journal.performance_summary(days=30)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Trades (30d)", perf.get("total_trades", 0))
    k2.metric("Win Rate", f"{perf.get('win_rate', 0)*100:.1f}%")
    k3.metric("Total P&L", f"${perf.get('total_pnl', 0):+.2f}")
    k4.metric("Avg Win", f"${perf.get('avg_win', 0):.2f}")
    k5.metric("Profit Factor", f"{perf.get('profit_factor', 0):.2f}")

    st.divider()

    tab1, tab2 = st.tabs(["Trades", "Analytics"])

    with tab1:
        trades = journal.recent_trades(limit=100)
        if not trades:
            st.info("No trades recorded yet.")
        else:
            df = pd.DataFrame(trades)
            # Rename for display
            rename = {
                "opened_at": "Opened", "closed_at": "Closed", "symbol": "Symbol",
                "direction": "Dir", "entry_price": "Entry", "exit_price": "Exit",
                "pnl": "P&L", "pnl_pct": "P&L%", "outcome": "Result",
                "position_size": "Size", "risk_amount": "Risk $",
                "emotion_tag": "Emotion", "strategy": "Strategy",
            }
            display_cols = [c for c in rename if c in df.columns]
            df_display = df[display_cols].rename(columns=rename)

            # Colour P&L column
            def _colour_pnl(val):
                if isinstance(val, (int, float)):
                    color = "#2ecc8a" if val > 0 else "#e05a5a" if val < 0 else "#8890a8"
                    return f"color: {color}"
                return ""

            st.dataframe(
                df_display.style.applymap(_colour_pnl, subset=["P&L", "P&L%"] if "P&L%" in df_display.columns else ["P&L"]),
                use_container_width=True,
                height=450,
            )

    with tab2:
        trades = journal.recent_trades(limit=200)
        if len(trades) < 2:
            st.info("Need at least 2 trades for analytics.")
        else:
            df = pd.DataFrame(trades)

            col_a, col_b = st.columns(2)

            with col_a:
                # Win/Loss distribution
                if "outcome" in df.columns:
                    outcome_counts = df["outcome"].value_counts()
                    pie_fig = go.Figure(go.Pie(
                        labels=outcome_counts.index.tolist(),
                        values=outcome_counts.values.tolist(),
                        marker_colors=["#2ecc8a", "#e05a5a", "#c9a84c", "#8890a8"],
                        hole=0.45,
                    ))
                    pie_fig.update_layout(
                        template="plotly_dark", paper_bgcolor="#0e1117",
                        height=280, margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(text="Outcome Distribution", font=dict(color="#c9a84c")),
                        legend=dict(font=dict(color="#8890a8")),
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

            with col_b:
                # P&L by symbol
                if "pnl" in df.columns and "symbol" in df.columns:
                    sym_pnl = df.groupby("symbol")["pnl"].sum().sort_values()
                    bar_fig = go.Figure(go.Bar(
                        x=sym_pnl.values,
                        y=sym_pnl.index.tolist(),
                        orientation="h",
                        marker_color=["#2ecc8a" if v >= 0 else "#e05a5a" for v in sym_pnl.values],
                    ))
                    bar_fig.update_layout(
                        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        height=280, margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(text="P&L by Symbol", font=dict(color="#c9a84c")),
                        xaxis=dict(showgrid=True, gridcolor="#1f2235"),
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

            # Equity curve
            st.subheader("Equity Curve")
            eq_fig = _equity_fig(cfg.account.balance)
            if eq_fig.data:
                st.plotly_chart(eq_fig, use_container_width=True)

            # Rolling win rate
            if "outcome" in df.columns:
                df_sorted = df.sort_values("opened_at") if "opened_at" in df.columns else df
                df_sorted["win"] = df_sorted["outcome"] == "WIN"
                df_sorted["rolling_wr"] = df_sorted["win"].rolling(10, min_periods=1).mean() * 100
                wr_fig = go.Figure(go.Scatter(
                    x=list(range(len(df_sorted))),
                    y=df_sorted["rolling_wr"],
                    mode="lines",
                    line=dict(color="#c9a84c", width=2),
                    name="Rolling WR (10)",
                ))
                wr_fig.add_hline(y=50, line_dash="dot", line_color="#3b3f5c")
                wr_fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    height=220, margin=dict(l=0, r=0, t=30, b=0),
                    title=dict(text="Rolling Win Rate (10 trades)", font=dict(color="#c9a84c")),
                    yaxis=dict(ticksuffix="%", showgrid=True, gridcolor="#1f2235"),
                    xaxis=dict(showgrid=True, gridcolor="#1f2235", title="Trade #"),
                )
                st.plotly_chart(wr_fig, use_container_width=True)


def _page_backtest():
    st.title("🔬 Backtest")

    with st.form("bt_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bt_symbol = st.selectbox("Symbol", cfg.watchlist.all_symbols)
        with col2:
            bt_tf = st.selectbox("Timeframe", ["1h", "4h", "1d"])
        with col3:
            bt_start = st.date_input("Start", value=pd.to_datetime(cfg.backtest.default_start))
        with col4:
            bt_end = st.date_input("End", value=pd.to_datetime(cfg.backtest.default_end))

        min_score_input = st.slider("Min Confidence Score", 40, 90, cfg.signals.min_confidence_score)
        run = st.form_submit_button("▶ Run Backtest", type="primary", use_container_width=True)

    if run:
        from msomi.backtest.engine import BacktestEngine

        with st.spinner(f"Backtesting {bt_symbol} [{bt_tf}] from {bt_start} to {bt_end}…"):
            engine = BacktestEngine(cfg)
            report = engine.run(
                symbol=bt_symbol,
                timeframe=bt_tf,
                start=str(bt_start),
                end=str(bt_end),
                min_score=min_score_input,
            )

        # KPI strip
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Trades", report.total_trades)
        m2.metric("Win Rate", f"{report.win_rate:.1f}%")
        m3.metric("Sharpe", f"{report.sharpe_ratio:.2f}")
        m4.metric("Max DD", f"{report.max_drawdown_pct:.1f}%")
        m5.metric("Return", f"{report.total_return_pct:+.1f}%")
        m6.metric("Profit Factor", f"{report.profit_factor:.2f}" if report.profit_factor else "N/A")

        # Equity curve
        if report.equity_curve:
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                y=report.equity_curve,
                mode="lines",
                line=dict(color="#c9a84c", width=2),
                fill="tozeroy",
                fillcolor="rgba(201,168,76,0.07)",
            ))
            eq_fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                title=dict(text=f"Equity Curve — {bt_symbol}", font=dict(color="#c9a84c")),
                xaxis=dict(title="Trade #", showgrid=True, gridcolor="#1f2235"),
                yaxis=dict(title="Equity ($)", showgrid=True, gridcolor="#1f2235", side="right"),
            )
            st.plotly_chart(eq_fig, use_container_width=True)

        # Raw stats
        with st.expander("Full Report"):
            st.code(report.summary())

        # Trades table
        if report.trades:
            trades_df = pd.DataFrame([
                {
                    "Entry Time": t.entry_time,
                    "Exit Time": t.exit_time,
                    "Dir": t.direction,
                    "Entry": round(t.entry_price, 5),
                    "Exit": round(t.exit_price, 5),
                    "P&L%": round(t.pnl_pct, 2),
                    "Outcome": t.outcome,
                    "Score": t.score,
                }
                for t in report.trades
            ])
            st.dataframe(trades_df, use_container_width=True, height=350)


def _page_settings():
    st.title("⚙️ Settings")

    col_l, col_r = st.columns(2)

    with col_l:
        with st.expander("Risk Configuration", expanded=True):
            st.metric("Risk per Trade", f"{cfg.risk.per_trade_pct*100:.1f}%")
            st.metric("Daily Loss Limit", f"{cfg.risk.daily_loss_limit_pct*100:.1f}%")
            st.metric("Weekly Drawdown Limit", f"{cfg.risk.weekly_drawdown_limit_pct*100:.1f}%")
            st.metric("Max Consecutive Losses", cfg.risk.max_consecutive_losses)
            st.metric("Min R:R", f"{cfg.risk.min_risk_reward}:1")
            st.metric("Max Open Positions", cfg.risk.max_open_positions)

        with st.expander("AI Configuration"):
            st.metric("Provider", cfg.ai.provider.upper())
            model = cfg.ai.model_anthropic if cfg.ai.provider == "anthropic" else cfg.ai.model_openai
            st.metric("Model", model)
            st.metric("Max Tokens", cfg.ai.max_tokens)
            st.metric("Temperature", cfg.ai.temperature)

    with col_r:
        with st.expander("Signal Configuration", expanded=True):
            st.metric("Min Confidence Score", cfg.signals.min_confidence_score)
            st.metric("Primary Timeframe", cfg.signals.timeframes.get("primary", "1h"))
            st.metric("EMA Fast", cfg.signals.indicators.ema_fast)
            st.metric("EMA Slow", cfg.signals.indicators.ema_slow)
            st.metric("EMA Trend", cfg.signals.indicators.ema_trend)
            st.metric("RSI Period", cfg.signals.indicators.rsi_period)
            st.metric("ATR Period", cfg.signals.indicators.atr_period)

        with st.expander("Watchlist"):
            st.markdown("**Forex:**")
            st.code("  ".join(cfg.watchlist.forex))
            st.markdown("**Crypto:**")
            st.code("  ".join(cfg.watchlist.crypto))

    st.info("📝 Edit `config/settings.yaml` to change any setting, then restart the dashboard.")


# ─── Sidebar & routing ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h2 style='color:#c9a84c;margin-bottom:0'>📡 Msomi</h2>"
        "<p style='color:#8890a8;margin-top:2px;font-size:0.85em'>Trading Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        ["Live Feed", "Charts", "Signals", "Journal", "Backtest", "Settings"],
        index=0,
    )

    st.divider()

    # Mini circuit breaker pill in sidebar
    state = circuit_breaker.state
    if state.is_tripped:
        st.markdown("🔴 **Circuit Breaker TRIPPED**")
    else:
        streak = state.consecutive_losses
        color = "#e05a5a" if streak >= cfg.risk.max_consecutive_losses - 1 else "#2ecc8a"
        st.markdown(f"<span style='color:{color}'>🟢 CB OK · {streak} loss streak</span>", unsafe_allow_html=True)

    st.caption(f"v{cfg.app.version} · {cfg.app.env}")
    st.caption(f"{datetime.utcnow().strftime('%H:%M UTC')}")

# Route
if page == "Live Feed":
    _page_live_feed()
elif page == "Charts":
    _page_charts()
elif page == "Signals":
    _page_signals()
elif page == "Journal":
    _page_journal()
elif page == "Backtest":
    _page_backtest()
elif page == "Settings":
    _page_settings()
