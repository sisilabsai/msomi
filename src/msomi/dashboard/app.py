"""Msomi Control Center — Streamlit dashboard."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure src/ is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from msomi.core.config import get_config
from msomi.core.database import init_db
from msomi.data.feeds import fetch_latest_price
from msomi.journal.logger import TradeJournal
from msomi.risk.manager import RiskManager
from msomi.signals.engine import SignalEngine

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Msomi — Control Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .metric-card {
        background: #1c1f2e;
        border: 1px solid #252840;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .signal-card {
        background: #1c1f2e;
        border-left: 3px solid #c9a84c;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .signal-long { border-left-color: #2ecc8a; }
    .signal-short { border-left-color: #e05a5a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Init ─────────────────────────────────────────────────────────────────────


@st.cache_resource
def _init():
    cfg = get_config()
    init_db(cfg.data.db_url)
    return cfg, TradeJournal(), SignalEngine(cfg)


cfg, journal, signal_engine = _init()


# ─── Page implementations ────────────────────────────────────────────────────


def _page_live_feed():
    st.title("📡 Live Feed")
    st.caption("Real-time watchlist and market overview")

    col1, col2, col3, col4 = st.columns(4)
    perf = journal.performance_summary(days=7)
    with col1:
        st.metric("Trades (7d)", perf.get("total_trades", 0))
    with col2:
        wr = perf.get("win_rate", 0)
        st.metric("Win Rate", f"{wr:.0f}%")
    with col3:
        pnl = perf.get("total_pnl", 0)
        st.metric("P&L (7d)", f"${pnl:+.2f}")
    with col4:
        pf = perf.get("profit_factor", 0)
        st.metric("Profit Factor", f"{pf:.2f}")

    st.divider()

    # Watchlist prices
    st.subheader("Watchlist")
    symbols = cfg.watchlist.all_symbols
    price_cols = st.columns(min(len(symbols), 5))
    for i, sym in enumerate(symbols[:5]):
        price = fetch_latest_price(sym)
        with price_cols[i]:
            st.metric(sym, f"{price:.4f}" if price else "N/A")

    # Equity curve
    st.subheader("Equity Curve")
    eq_df = journal.equity_curve(cfg.account.balance)
    if not eq_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=eq_df["time"],
                y=eq_df["equity"],
                mode="lines",
                line=dict(color="#c9a84c", width=2),
                fill="tozeroy",
                fillcolor="rgba(201,168,76,0.08)",
                name="Equity",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades yet. Start trading to see your equity curve.")


def _page_signals():
    st.title("🎯 Signals")

    tab1, tab2 = st.tabs(["Recent Signals", "Scan Now"])

    with tab1:
        signals = journal.recent_signals(limit=30)
        if not signals:
            st.info("No signals fired yet.")
        else:
            for s in signals:
                direction_color = "signal-long" if s["direction"] == "LONG" else "signal-short"
                score_color = "#2ecc8a" if s["score"] >= 75 else "#c9a84c" if s["score"] >= 60 else "#8890a8"
                ts = s["created_at"].strftime("%Y-%m-%d %H:%M") if s["created_at"] else ""

                st.markdown(
                    f"""
                    <div class="signal-card {direction_color}">
                        <strong>{s['symbol']}</strong> [{s['timeframe']}] &nbsp;
                        <span style="color:{'#2ecc8a' if s['direction']=='LONG' else '#e05a5a'}">
                            ▲ {s['direction']}</span> &nbsp;
                        <span style="color:{score_color}">Score: {s['score']}/100</span> &nbsp;
                        <small style="color:#8890a8">{ts}</small><br/>
                        Entry: <code>{s['entry']:.5f}</code> &nbsp;
                        SL: <code>{s['sl']:.5f}</code> &nbsp;
                        TP: <code>{s['tp']:.5f}</code> &nbsp;
                        R:R <code>{s['rr']:.2f}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if s.get("ai_analysis"):
                    with st.expander("🤖 AI Analysis"):
                        st.write(s["ai_analysis"])

    with tab2:
        st.subheader("Manual Scan")
        col1, col2 = st.columns(2)
        with col1:
            scan_symbol = st.selectbox("Symbol", cfg.watchlist.all_symbols)
        with col2:
            scan_tf = st.selectbox("Timeframe", ["15m", "1h", "4h"])

        if st.button("🔍 Scan Now", type="primary"):
            with st.spinner(f"Scanning {scan_symbol}…"):
                event = signal_engine.evaluate_symbol(scan_symbol, timeframe=scan_tf)

            if event:
                s = event.signal
                st.success(f"Signal found! {s.direction} | Score: {s.score}/100")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Entry", f"{s.entry_price:.5f}")
                col_b.metric("Stop Loss", f"{s.stop_loss:.5f}")
                col_c.metric("Take Profit", f"{s.take_profit:.5f}")
                col_d.metric("R:R", f"{s.risk_reward:.2f}:1")

                st.markdown("**Why:**")
                for reason in s.reasons:
                    st.write(f"• {reason}")
            else:
                st.info(f"No signal above threshold ({cfg.signals.min_confidence_score}) for {scan_symbol} [{scan_tf}].")

            snap = signal_engine.snapshot_only(scan_symbol, timeframe=scan_tf)
            if snap:
                st.divider()
                st.subheader("Indicator Snapshot")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("RSI", f"{snap.rsi:.1f}", snap.rsi_signal)
                c2.metric("MACD Hist", f"{snap.macd_hist:.5f}", snap.macd_crossover)
                c3.metric("Trend", snap.trend_direction)
                c4.metric("Volume Ratio", f"{snap.volume_ratio:.1f}×")


def _page_journal():
    st.title("📚 Trade Journal")

    perf = journal.performance_summary(days=30)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Trades (30d)", perf.get("total_trades", 0))
    col2.metric("Win Rate", f"{perf.get('win_rate', 0):.0f}%")
    col3.metric("P&L", f"${perf.get('total_pnl', 0):+.2f}")
    col4.metric("Avg Win", f"${perf.get('avg_win', 0):.2f}")
    col5.metric("Profit Factor", f"{perf.get('profit_factor', 0):.2f}")

    st.divider()
    trades = journal.recent_trades(limit=50)
    if trades:
        df = pd.DataFrame(trades)
        display_cols = ["opened_at", "symbol", "direction", "entry", "exit", "pnl", "pnl_pct", "outcome", "strategy"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, height=400)
    else:
        st.info("No trades recorded yet.")


def _page_backtest():
    st.title("🔬 Backtest")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bt_symbol = st.selectbox("Symbol", cfg.watchlist.all_symbols, key="bt_sym")
    with col2:
        bt_tf = st.selectbox("Timeframe", ["1h", "4h", "1d"], key="bt_tf")
    with col3:
        bt_start = st.date_input("Start", value=pd.to_datetime(cfg.backtest.default_start))
    with col4:
        bt_end = st.date_input("End", value=pd.to_datetime(cfg.backtest.default_end))

    min_score_input = st.slider("Min Confidence Score", 40, 90, cfg.signals.min_confidence_score)

    if st.button("▶ Run Backtest", type="primary"):
        from msomi.backtest.engine import BacktestEngine

        with st.spinner(f"Running backtest for {bt_symbol}…"):
            engine = BacktestEngine(cfg)
            report = engine.run(
                symbol=bt_symbol,
                timeframe=bt_tf,
                start=str(bt_start),
                end=str(bt_end),
                min_score=min_score_input,
            )

        st.code(report.summary())

        # Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades", report.total_trades)
        m2.metric("Win Rate", f"{report.win_rate:.1f}%")
        m3.metric("Sharpe", f"{report.sharpe_ratio:.2f}")
        m4.metric("Max Drawdown", f"{report.max_drawdown_pct:.1f}%")
        m5.metric("Total Return", f"{report.total_return_pct:+.1f}%")

        # Equity curve
        if report.equity_curve:
            st.subheader("Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=report.equity_curve,
                mode="lines",
                line=dict(color="#c9a84c"),
                name="Equity",
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Trades table
        if report.trades:
            trades_df = pd.DataFrame([
                {
                    "Entry Time": t.entry_time,
                    "Exit Time": t.exit_time,
                    "Direction": t.direction,
                    "Entry": t.entry_price,
                    "Exit": t.exit_price,
                    "P&L%": t.pnl_pct,
                    "Outcome": t.outcome,
                    "Score": t.score,
                }
                for t in report.trades
            ])
            st.dataframe(trades_df, use_container_width=True)


def _page_settings():
    st.title("⚙️ Settings")

    with st.expander("Risk Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk per Trade", f"{cfg.risk.per_trade_pct*100:.1f}%")
            st.metric("Daily Loss Limit", f"{cfg.risk.daily_loss_limit_pct*100:.1f}%")
        with col2:
            st.metric("Max Consecutive Losses", cfg.risk.max_consecutive_losses)
            st.metric("Min R:R", f"{cfg.risk.min_risk_reward}:1")

    with st.expander("Signal Configuration"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Min Confidence Score", cfg.signals.min_confidence_score)
            st.metric("Primary Timeframe", cfg.signals.timeframes.get("primary", "1h"))
        with col2:
            st.metric("EMA Trend Period", cfg.signals.indicators.ema_trend)
            st.metric("RSI Period", cfg.signals.indicators.rsi_period)

    with st.expander("AI Configuration"):
        st.metric("Provider", cfg.ai.provider)
        st.metric("Model", cfg.ai.model_anthropic if cfg.ai.provider == "anthropic" else cfg.ai.model_openai)

    with st.expander("Watchlist"):
        st.markdown("**Forex:**")
        st.code(", ".join(cfg.watchlist.forex))
        st.markdown("**Crypto:**")
        st.code(", ".join(cfg.watchlist.crypto))

    st.info("To modify settings, edit `config/settings.yaml` and restart.")


# ─── Sidebar & routing (after page functions are defined) ─────────────────────

with st.sidebar:
    st.markdown("## 📡 Msomi")
    st.caption("Personal Trading Intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Live Feed", "Signals", "Journal", "Backtest", "Settings"],
        index=0,
    )

    st.divider()
    st.caption(f"v{cfg.app.version} · {cfg.app.env}")
    st.caption(f"Updated: {datetime.utcnow().strftime('%H:%M UTC')}")

# Route to selected page
if page == "Live Feed":
    _page_live_feed()
elif page == "Signals":
    _page_signals()
elif page == "Journal":
    _page_journal()
elif page == "Backtest":
    _page_backtest()
elif page == "Settings":
    _page_settings()
