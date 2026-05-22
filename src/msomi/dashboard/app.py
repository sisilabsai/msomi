"""Msomi Control Center — Streamlit dashboard (full-featured build)."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Ensure src/ is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from msomi.core.config import get_config
from msomi.core.database import init_db
from msomi.data.feeds import fetch_latest_price, fetch_ohlcv
from msomi.journal.logger import TradeJournal
from msomi.risk.circuit_breaker import CircuitBreaker
from msomi.signals.engine import SignalEngine
from msomi.signals.indicators import IndicatorEngine

st.set_page_config(page_title="Msomi", page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"] { background: #141720; border-right: 1px solid #1f2235; }
.sig-card { background:#141720;border-radius:10px;padding:14px 18px;margin-bottom:10px;border-left:4px solid #3b3f5c; }
.sig-long { border-left-color:#2ecc8a; }
.sig-short { border-left-color:#e05a5a; }
.price-tile { background:#141720;border:1px solid #1f2235;border-radius:10px;padding:14px 12px;text-align:center; }
.price-up { color:#2ecc8a; }
.price-down { color:#e05a5a; }
.opp-card-long { background:#141720;border-radius:12px;padding:16px 20px;margin-bottom:12px;border:1px solid #2ecc8a44;border-left:5px solid #2ecc8a; }
.opp-card-short { background:#141720;border-radius:12px;padding:16px 20px;margin-bottom:12px;border:1px solid #e05a5a44;border-left:5px solid #e05a5a; }
.mkt-table { width:100%;border-collapse:collapse;font-size:.9em; }
.mkt-table th { padding:8px 10px;text-align:left;color:#8890a8;border-bottom:1px solid #1f2235;font-weight:500; }
.mkt-table td { padding:8px 10px;border-bottom:1px solid #141720; }
.mkt-table tr:hover td { background:#141720; }
.cb-ok { background:#0d2b1e;border:1px solid #2ecc8a;border-radius:8px;padding:12px 16px; }
.cb-tripped { background:#2b0d0d;border:1px solid #e05a5a;border-radius:8px;padding:12px 16px; }
.session-pill { display:inline-block;border-radius:5px;padding:3px 10px;margin:2px;font-size:0.78em;font-weight:600; }
.session-open { background:#0d2b1e;color:#2ecc8a;border:1px solid #2ecc8a; }
.session-closed { background:#1a1a2e;color:#3b3f5c;border:1px solid #252840; }
[data-testid="stMetricDelta"] svg { display:none; }
.stTabs [data-baseweb="tab"] { color:#8890a8; }
.stTabs [aria-selected="true"] { color:#c9a84c;border-bottom-color:#c9a84c; }
</style>""", unsafe_allow_html=True)

SESSIONS = {"Sydney":(22,7),"Tokyo":(0,9),"London":(8,17),"New York":(13,22)}
TIMEFRAMES = ["5m","15m","1h","4h","1d"]

@st.cache_resource
def _init():
    cfg = get_config()
    init_db(cfg.data.db_url)
    journal = TradeJournal()
    engine = SignalEngine(cfg)
    ind = cfg.signals.indicators
    ie = IndicatorEngine(ema_fast=ind.ema_fast,ema_slow=ind.ema_slow,ema_trend=ind.ema_trend,
                         rsi_period=ind.rsi_period,rsi_overbought=ind.rsi_overbought,
                         rsi_oversold=ind.rsi_oversold,macd_fast=ind.macd_fast,
                         macd_slow=ind.macd_slow,macd_signal=ind.macd_signal,
                         bb_period=ind.bb_period,bb_std=ind.bb_std,atr_period=ind.atr_period)
    breaker = CircuitBreaker(account_balance=cfg.account.balance,
                             daily_loss_limit_pct=cfg.risk.daily_loss_limit_pct,
                             max_consecutive_losses=cfg.risk.max_consecutive_losses)
    return cfg, journal, engine, ie, breaker

cfg, journal, signal_engine, ind_engine, circuit_breaker = _init()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_color(score):
    return "#2ecc8a" if score>=75 else "#c9a84c" if score>=60 else "#8890a8"

def _session_open(name):
    h0,h1=SESSIONS[name]; now=datetime.now(timezone.utc).hour
    return (now>=h0 or now<h1) if h0>h1 else h0<=now<h1

def _session_bar():
    parts=[]
    for n in SESSIONS:
        cls="session-open" if _session_open(n) else "session-closed"
        parts.append(f'<span class="session-pill {cls}">{n}</span>')
    return " ".join(parts)

def _get_chart_df(symbol,timeframe,periods,show_emas=True,show_bb=True,show_vwap=True):
    try:
        df=fetch_ohlcv(symbol,timeframe=timeframe,periods=periods)
        df=ind_engine.compute(df)
    except Exception:
        return go.Figure(),None
    traces=[go.Candlestick(x=df.index,open=df["open"],high=df["high"],low=df["low"],
                           close=df["close"],increasing_line_color="#2ecc8a",
                           decreasing_line_color="#e05a5a",increasing_fillcolor="#2ecc8a",
                           decreasing_fillcolor="#e05a5a",name=symbol)]
    ind_cfg=cfg.signals.indicators
    if show_emas:
        traces+=[
            go.Scatter(x=df.index,y=df["ema_fast"],mode="lines",
                       line=dict(color="#c9a84c",width=1),name=f"EMA{ind_cfg.ema_fast}",hoverinfo="skip"),
            go.Scatter(x=df.index,y=df["ema_slow"],mode="lines",
                       line=dict(color="#7b9fe0",width=1),name=f"EMA{ind_cfg.ema_slow}",hoverinfo="skip"),
            go.Scatter(x=df.index,y=df["ema_trend"],mode="lines",
                       line=dict(color="#e07bbb",width=1.5,dash="dot"),name=f"EMA{ind_cfg.ema_trend}",hoverinfo="skip"),
        ]
    if show_bb:
        traces+=[
            go.Scatter(x=df.index,y=df["bb_upper"],mode="lines",
                       line=dict(color="rgba(90,130,200,0.4)",width=1),name="BB Upper",hoverinfo="skip"),
            go.Scatter(x=df.index,y=df["bb_lower"],mode="lines",
                       line=dict(color="rgba(90,130,200,0.4)",width=1),
                       fill="tonexty",fillcolor="rgba(90,130,200,0.05)",name="BB Lower",hoverinfo="skip"),
            go.Scatter(x=df.index,y=df["bb_mid"],mode="lines",
                       line=dict(color="rgba(90,130,200,0.25)",width=1,dash="dot"),name="BB Mid",hoverinfo="skip"),
        ]
    if show_vwap and "vwap" in df.columns and df["vwap"].notna().any():
        traces.append(go.Scatter(x=df.index,y=df["vwap"],mode="lines",
                                 line=dict(color="#ff9f43",width=1.5,dash="dash"),
                                 name="VWAP",hoverinfo="skip"))
    vol_c=["#2ecc8a" if c>=o else "#e05a5a" for c,o in zip(df["close"],df["open"])]
    traces.append(go.Bar(x=df.index,y=df["volume"],marker_color=vol_c,opacity=0.25,
                         yaxis="y2",name="Volume",hoverinfo="skip"))
    fig=go.Figure(data=traces)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                      height=450,margin=dict(l=0,r=0,t=35,b=0),xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h",font=dict(size=10,color="#8890a8"),
                                  bgcolor="rgba(0,0,0,0)",x=0,y=1.05),
                      title=dict(text=f"{symbol}  ·  {timeframe}",font=dict(size=14,color="#c9a84c")),
                      yaxis=dict(showgrid=True,gridcolor="#1a1d2e",side="right"),
                      yaxis2=dict(overlaying="y",side="left",showgrid=False,showticklabels=False,
                                  range=[0,df["volume"].max()*6]),
                      xaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
    return fig,df

def _equity_fig(initial_balance):
    eq_df=journal.equity_curve(initial_balance)
    fig=go.Figure()
    if eq_df.empty: return fig
    fig.add_trace(go.Scatter(x=eq_df["opened_at"],y=eq_df["balance"],mode="lines",
                             line=dict(color="#c9a84c",width=2),fill="tozeroy",
                             fillcolor="rgba(201,168,76,0.07)",name="Balance"))
    fig.add_hline(y=initial_balance,line_dash="dot",line_color="#3b3f5c",
                  annotation_text=f"Start ${initial_balance:,.0f}",annotation_font_color="#8890a8")
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                      height=300,margin=dict(l=0,r=0,t=10,b=0),showlegend=False,
                      xaxis=dict(showgrid=True,gridcolor="#1a1d2e"),
                      yaxis=dict(showgrid=True,gridcolor="#1a1d2e",side="right"))
    return fig

def _drawdown_fig(initial_balance):
    eq_df=journal.equity_curve(initial_balance)
    fig=go.Figure()
    if eq_df.empty: return fig
    bal=eq_df["balance"].values
    peak=np.maximum.accumulate(bal)
    dd=(bal-peak)/peak*100
    fig.add_trace(go.Scatter(x=eq_df["opened_at"],y=dd,mode="lines",
                             line=dict(color="#e05a5a",width=1.5),fill="tozeroy",
                             fillcolor="rgba(224,90,90,0.08)",name="Drawdown"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                      height=200,margin=dict(l=0,r=0,t=10,b=0),showlegend=False,
                      title=dict(text="Drawdown",font=dict(color="#e05a5a",size=12)),
                      yaxis=dict(ticksuffix="%",showgrid=True,gridcolor="#1a1d2e",side="right"),
                      xaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
    return fig

def _cb_widget():
    state=circuit_breaker.state
    if state.is_tripped:
        st.markdown(f'<div class="cb-tripped">⛔ <strong>Circuit Breaker TRIPPED</strong> — {state.reason}<br/>'
                    f'<small>Daily P&L: <strong>${state.daily_pnl:+.2f}</strong> '
                    f'({state.daily_loss_pct*100:.1f}%)</small></div>',unsafe_allow_html=True)
    else:
        streak=state.consecutive_losses
        sw=f"⚠️ {streak}/{circuit_breaker.max_streak} loss streak" if streak>0 else "✅ Clear"
        st.markdown(f'<div class="cb-ok">🟢 <strong>Circuit Breaker OK</strong> — {sw}<br/>'
                    f'<small>Daily P&L: <strong>${state.daily_pnl:+.2f}</strong> '
                    f'· {state.trades_today} trade(s) today</small></div>',unsafe_allow_html=True)

# ── Live Feed helpers ────────────────────────────────────────────────────────

def _run_market_scan(symbols, timeframe):
    """Scan all symbols and return list of {sym, price, snap, event}."""
    results = []
    bar = st.progress(0, text=f"Scanning {len(symbols)} symbols on {timeframe}…")
    for i, sym in enumerate(symbols):
        try:
            event = signal_engine.evaluate_symbol(sym, timeframe=timeframe)
            snap = signal_engine.snapshot_only(sym, timeframe=timeframe)
            price = fetch_latest_price(sym)
            results.append({"sym": sym, "price": price, "snap": snap, "event": event})
        except Exception:
            results.append({"sym": sym, "price": None, "snap": None, "event": None})
        bar.progress((i + 1) / len(symbols), text=f"Scanned {sym}…")
    bar.empty()
    return results

def _opportunity_card(r):
    """Render a full signal card with entry/SL/TP/reasons."""
    s = r["event"].signal
    sym = r["sym"]
    snap = r.get("snap")
    price = r.get("price")
    direction = s.direction
    score = s.score
    dc = "#2ecc8a" if direction == "LONG" else "#e05a5a"
    arrow = "▲ LONG" if direction == "LONG" else "▼ SHORT"
    sc = _score_color(score)
    risk_dollar = cfg.account.balance * cfg.risk.per_trade_pct
    sl_dist = abs(s.entry_price - s.stop_loss)
    price_str = f"${price:,.5f}" if price else "–"
    rsi_str = f"RSI {snap.rsi:.0f}" if snap else ""
    trend_str = snap.trend_direction if snap else ""
    bb_str = snap.bb_position if snap else ""
    vol_str = f"Vol ×{snap.volume_ratio:.1f}{'  🔥' if snap.volume_spike else ''}" if snap else ""
    squeeze_warn = "<br/><span style='color:#c9a84c;font-size:.8em'>🗜️ BB Squeeze — watch for breakout</span>" if snap and snap.bb_squeeze else ""
    reasons_html = "".join(f"<li style='margin-bottom:2px'>{reason}</li>" for reason in s.reasons[:5])
    card_cls = "opp-card-long" if direction == "LONG" else "opp-card-short"
    st.markdown(f"""
    <div class="{card_cls}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
        <div>
          <span style="font-size:1.25em;font-weight:700;color:#e0e0e0">{sym.replace('=X','').replace('-USD','')}</span>
          <span style="color:{dc};font-size:1.1em;font-weight:600;margin-left:12px">{arrow}</span>
          <span style="background:{sc}22;color:{sc};padding:3px 10px;border-radius:6px;font-size:.85em;margin-left:8px;font-weight:600">{score}/100</span>
        </div>
        <div style="text-align:right;color:#8890a8;font-size:.82em">
          {rsi_str} &nbsp;·&nbsp; Trend: <strong style='color:#e0e0e0'>{trend_str}</strong> &nbsp;·&nbsp; {bb_str}<br/>
          <span style="color:#e07bbb">{vol_str}</span> &nbsp;·&nbsp; Price: <strong style='color:#e0e0e0'>{price_str}</strong>{squeeze_warn}
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:10px">
        <div style="background:#0e1117;border-radius:6px;padding:8px;text-align:center">
          <div style="color:#8890a8;font-size:.7em;text-transform:uppercase">Entry</div>
          <div style="color:#e0e0e0;font-weight:600;font-family:monospace">{s.entry_price:.5f}</div>
        </div>
        <div style="background:#0e1117;border-radius:6px;padding:8px;text-align:center">
          <div style="color:#8890a8;font-size:.7em;text-transform:uppercase">Stop Loss</div>
          <div style="color:#e05a5a;font-weight:600;font-family:monospace">{s.stop_loss:.5f}</div>
        </div>
        <div style="background:#0e1117;border-radius:6px;padding:8px;text-align:center">
          <div style="color:#8890a8;font-size:.7em;text-transform:uppercase">Take Profit</div>
          <div style="color:#2ecc8a;font-weight:600;font-family:monospace">{s.take_profit:.5f}</div>
        </div>
        <div style="background:#0e1117;border-radius:6px;padding:8px;text-align:center">
          <div style="color:#8890a8;font-size:.7em;text-transform:uppercase">R:R / Risk $</div>
          <div style="color:#c9a84c;font-weight:600">{s.risk_reward:.1f}:1 &nbsp;/&nbsp; ${risk_dollar:.0f}</div>
        </div>
      </div>
      <ul style="color:#8890a8;font-size:.85em;margin:4px 0 0;padding-left:18px">{reasons_html}</ul>
    </div>""", unsafe_allow_html=True)

def _market_overview_table(results):
    """Render a compact market overview with live indicator values."""
    header = (
        "<table class='mkt-table'>"
        "<thead><tr>"
        "<th>Pair</th><th style='text-align:right'>Price</th>"
        "<th style='text-align:center'>Trend</th><th style='text-align:center'>vs EMA200</th>"
        "<th style='text-align:center'>RSI</th><th style='text-align:center'>MACD</th>"
        "<th style='text-align:center'>BB Zone</th><th style='text-align:center'>Vol</th>"
        "<th style='text-align:center'>ATR%</th><th style='text-align:center'>Signal</th>"
        "</tr></thead><tbody>"
    )
    rows_html = ""
    for r in results:
        snap = r.get("snap")
        price = r.get("price")
        sym = r["sym"]
        label = sym.replace("=X", "").replace("-USD", "")
        if price is None and snap is None:
            continue
        price_str = (f"${price:,.2f}" if "-USD" in sym else f"{price:.5f}") if price else "–"
        trend = snap.trend_direction if snap else "–"
        tc = "#2ecc8a" if trend == "UP" else "#e05a5a" if trend == "DOWN" else "#8890a8"
        ta = "▲" if trend == "UP" else "▼" if trend == "DOWN" else "–"
        ema_pos = snap.price_vs_ema_trend if snap else "–"
        ema_c = "#2ecc8a" if snap and "ABOVE" in (snap.price_vs_ema_trend or "") else "#e05a5a" if snap and "BELOW" in (snap.price_vs_ema_trend or "") else "#8890a8"
        rsi = snap.rsi if snap else None
        rsi_c = "#e05a5a" if rsi and rsi > 70 else "#2ecc8a" if rsi and rsi < 30 else "#c9a84c" if rsi else "#8890a8"
        rsi_s = f"{rsi:.0f}" if rsi else "–"
        macd_c_val = snap.macd_crossover if snap else "NONE"
        macd_a = "↑" if macd_c_val == "BULLISH" else "↓" if macd_c_val == "BEARISH" else "–"
        macd_c = "#2ecc8a" if macd_c_val == "BULLISH" else "#e05a5a" if macd_c_val == "BEARISH" else "#8890a8"
        bb = snap.bb_position if snap else "–"
        vol = f"×{snap.volume_ratio:.1f}" if snap else "–"
        vol_c = "#e07bbb" if snap and snap.volume_spike else "#8890a8"
        atr_s = f"{snap.atr_pct*100:.2f}%" if snap else "–"
        event = r.get("event")
        if event:
            sig_d = event.signal.direction
            sig_s = event.signal.score
            sig_str = f"{'▲' if sig_d=='LONG' else '▼'} {sig_s}"
            sig_c = "#2ecc8a" if sig_d == "LONG" else "#e05a5a"
        else:
            sig_str = "–"; sig_c = "#3b3f5c"
        rows_html += (
            f"<tr>"
            f"<td style='font-weight:600;color:#e0e0e0'>{label}</td>"
            f"<td style='text-align:right;font-family:monospace;color:#e0e0e0'>{price_str}</td>"
            f"<td style='text-align:center;color:{tc};font-weight:600'>{ta} {trend}</td>"
            f"<td style='text-align:center;color:{ema_c};font-size:.82em'>{ema_pos}</td>"
            f"<td style='text-align:center;color:{rsi_c};font-weight:600'>{rsi_s}</td>"
            f"<td style='text-align:center;color:{macd_c};font-weight:700;font-size:1.1em'>{macd_a}</td>"
            f"<td style='text-align:center;color:#8890a8;font-size:.82em'>{bb}</td>"
            f"<td style='text-align:center;color:{vol_c}'>{vol}</td>"
            f"<td style='text-align:center;color:#8890a8'>{atr_s}</td>"
            f"<td style='text-align:center;color:{sig_c};font-weight:700'>{sig_str}</td>"
            f"</tr>"
        )
    st.markdown(header + rows_html + "</tbody></table>", unsafe_allow_html=True)

# ── Page: Live Feed ───────────────────────────────────────────────────────────

def _page_live_feed():
    st.title("📡 Live Feed")
    now_utc = datetime.now(timezone.utc)

    # ── Session bar + clock ───────────────────────────────────────────────────
    cl, cr = st.columns([3, 1])
    with cl:
        st.markdown(_session_bar(), unsafe_allow_html=True)
        st.caption(f"🕐 {now_utc.strftime('%H:%M:%S UTC')}  —  Green = session currently open")
    with cr:
        do_scan = st.button("🔄 Scan Markets", use_container_width=True, type="primary")
        primary_tf = cfg.signals.timeframes.get("primary", "1h") if hasattr(cfg.signals, "timeframes") else "1h"
        if isinstance(primary_tf, dict): primary_tf = primary_tf.get("primary", "1h")
        st.caption(f"Primary TF: **{primary_tf}**")

    # ── Auto-scan: run on first load or on demand ─────────────────────────────
    all_syms = cfg.watchlist.all_symbols
    scan_stale = True
    if "lf_scan_ts" in st.session_state and not do_scan:
        age = (now_utc - st.session_state["lf_scan_ts"]).total_seconds()
        scan_stale = age > 300  # refresh every 5 min

    if scan_stale or do_scan:
        results = _run_market_scan(all_syms, primary_tf)
        st.session_state["lf_results"] = results
        st.session_state["lf_scan_ts"] = now_utc
    else:
        results = st.session_state.get("lf_results", [])
        age_s = int((now_utc - st.session_state["lf_scan_ts"]).total_seconds())
        st.caption(f"📌 Cached scan — {age_s}s ago. Auto-refreshes every 5 min.")

    # ── Active signals (opportunities) ───────────────────────────────────────
    opportunities = [r for r in results if r.get("event")]
    if opportunities:
        st.markdown(f"### 🚨 {len(opportunities)} Active Signal{'s' if len(opportunities)!=1 else ''} — {primary_tf}")
        for r in sorted(opportunities, key=lambda x: x["event"].signal.score, reverse=True):
            _opportunity_card(r)
    else:
        st.info(
            f"📊 No confirmed signals above **{cfg.signals.min_confidence_score}** on {primary_tf} right now.  "
            f"All symbols are being monitored — check the **Heatmap** or **Charts** for setups building up."
        )
    st.divider()

    # ── Market overview table ─────────────────────────────────────────────────
    st.markdown("### 📊 Market Snapshot")
    st.caption("Live prices + indicator state for every watched symbol")
    if results:
        _market_overview_table(results)
    st.divider()

    # ── Risk status + equity ─────────────────────────────────────────────────
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Risk Status")
        _cb_widget()
        st.markdown("<br/>", unsafe_allow_html=True)
        rc = cfg.risk
        st.markdown(
            f"**Balance:** ${cfg.account.balance:,.2f} {cfg.account.currency}  \n"
            f"**Risk/Trade:** {rc.per_trade_pct*100:.1f}%  ·  **Max $:** ${cfg.account.balance*rc.per_trade_pct:,.0f}  \n"
            f"**Daily Limit:** {rc.daily_loss_limit_pct*100:.1f}%  \n"
            f"**Max Streak:** {rc.max_consecutive_losses} losses  \n"
            f"**Min R:R:** {rc.min_risk_reward}:1"
        )
    with right:
        st.subheader("Performance (30d)")
        p7 = journal.performance_summary(days=7)
        p30 = journal.performance_summary(days=30)
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Trades (7d)", p7.get("total_trades", 0))
        k2.metric("Win Rate (7d)", f"{p7.get('win_rate',0)*100:.0f}%",
                  delta=f"{(p7.get('win_rate',0)-p30.get('win_rate',0))*100:+.0f}% vs 30d")
        k3.metric("P&L (7d)", f"${p7.get('total_pnl',0):+.2f}")
        k4.metric("Profit Factor", f"{p30.get('profit_factor',0):.2f}")
        k5.metric("Avg Win", f"${p30.get('avg_win',0):.2f}")
        k6.metric("Avg Loss", f"${p30.get('avg_loss',0):.2f}")
        fig = _equity_fig(cfg.account.balance)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    if st.checkbox("Auto-refresh every 60 s", value=False):
        import time
        bar = st.progress(0, text="Refreshing in 60 s…")
        for i in range(60):
            time.sleep(1)
            bar.progress((i+1)/60, text=f"Refreshing in {59-i} s…")
        st.session_state.pop("lf_results", None)
        st.session_state.pop("lf_scan_ts", None)
        st.rerun()

# ── Page: Charts ─────────────────────────────────────────────────────────────

def _page_charts():
    st.title("📈 Charts")
    c1,c2,c3=st.columns([2,1,1])
    with c1: symbol=st.selectbox("Symbol",cfg.watchlist.all_symbols,key="chart_sym")
    with c2: timeframe=st.selectbox("Timeframe",["5m","15m","1h","4h","1d"],index=2,key="chart_tf")
    with c3: periods=st.selectbox("Candles",[60,120,200,500],index=1,key="chart_periods")

    o1,o2,o3=st.columns(3)
    show_emas=o1.checkbox("EMA Overlays",value=True)
    show_bb=o2.checkbox("Bollinger Bands",value=True)
    show_vwap=o3.checkbox("VWAP",value=True)

    with st.spinner(f"Loading {symbol} [{timeframe}]…"):
        fig,df=_get_chart_df(symbol,timeframe,periods,show_emas,show_bb,show_vwap)

    if not fig.data:
        st.warning("Could not load chart data."); return
    st.plotly_chart(fig,use_container_width=True)

    if df is not None and not df.empty:
        s1,s2,s3=st.columns(3)
        tail=60
        with s1:
            rsi_fig=go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index[-tail:],y=df["rsi"].iloc[-tail:],
                                         mode="lines",line=dict(color="#c9a84c",width=1.5),name="RSI"))
            rsi_fig.add_hline(y=70,line_dash="dot",line_color="#e05a5a",line_width=1)
            rsi_fig.add_hline(y=30,line_dash="dot",line_color="#2ecc8a",line_width=1)
            rsi_fig.add_hrect(y0=70,y1=100,fillcolor="rgba(224,90,90,0.05)",line_width=0)
            rsi_fig.add_hrect(y0=0,y1=30,fillcolor="rgba(46,204,138,0.05)",line_width=0)
            rsi_fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                  height=200,margin=dict(l=0,r=0,t=30,b=0),showlegend=False,
                                  title=dict(text="RSI (14)",font=dict(color="#c9a84c",size=12)),
                                  yaxis=dict(range=[0,100],showgrid=True,gridcolor="#1a1d2e"),
                                  xaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
            st.plotly_chart(rsi_fig,use_container_width=True)
        with s2:
            mc=["#2ecc8a" if v>=0 else "#e05a5a" for v in df["macd_hist"].iloc[-tail:]]
            mf=go.Figure()
            mf.add_trace(go.Bar(x=df.index[-tail:],y=df["macd_hist"].iloc[-tail:],marker_color=mc,name="Hist"))
            mf.add_trace(go.Scatter(x=df.index[-tail:],y=df["macd"].iloc[-tail:],
                                    mode="lines",line=dict(color="#7b9fe0",width=1),name="MACD"))
            mf.add_trace(go.Scatter(x=df.index[-tail:],y=df["macd_signal"].iloc[-tail:],
                                    mode="lines",line=dict(color="#e07bbb",width=1),name="Signal"))
            mf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                              height=200,margin=dict(l=0,r=0,t=30,b=0),showlegend=False,
                              title=dict(text="MACD (12,26,9)",font=dict(color="#c9a84c",size=12)),
                              xaxis=dict(showgrid=True,gridcolor="#1a1d2e"),
                              yaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
            st.plotly_chart(mf,use_container_width=True)
        with s3:
            vc=["#2ecc8a" if c>=o else "#e05a5a"
                for c,o in zip(df["close"].iloc[-tail:],df["open"].iloc[-tail:])]
            vf=go.Figure(go.Bar(x=df.index[-tail:],y=df["volume"].iloc[-tail:],marker_color=vc,name="Volume"))
            vf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                             height=200,margin=dict(l=0,r=0,t=30,b=0),showlegend=False,
                             title=dict(text="Volume",font=dict(color="#c9a84c",size=12)),
                             xaxis=dict(showgrid=True,gridcolor="#1a1d2e"),
                             yaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
            st.plotly_chart(vf,use_container_width=True)

    snap=signal_engine.snapshot_only(symbol,timeframe=timeframe)
    if snap:
        st.subheader("Indicator Values")
        c1,c2,c3,c4,c5,c6,c7=st.columns(7)
        c1.metric("Trend",snap.trend_direction)
        c2.metric("vs EMA200",snap.price_vs_ema_trend)
        c3.metric("RSI",f"{snap.rsi:.1f}",snap.rsi_signal)
        # FIX: macd_crossover is a str, not a number
        macd_delta=("↑ bullish" if snap.macd_crossover=="BULLISH"
                    else ("↓ bearish" if snap.macd_crossover=="BEARISH" else "–"))
        c4.metric("MACD Hist",f"{snap.macd_hist:.5f}",macd_delta)
        c5.metric("BB Zone",snap.bb_position)
        c6.metric("Vol Ratio",f"{snap.volume_ratio:.2f}×","🔥 Spike" if snap.volume_spike else None)
        c7.metric("ATR%",f"{snap.atr_pct*100:.2f}%")
        if snap.bb_squeeze:
            st.warning("🗜️ Bollinger Band squeeze — potential breakout incoming.")

        rg=go.Figure(go.Indicator(mode="gauge+number",value=snap.rsi,
                                   domain={"x":[0,1],"y":[0,1]},
                                   title={"text":"RSI","font":{"color":"#c9a84c"}},
                                   gauge={"axis":{"range":[0,100],"tickcolor":"#8890a8"},
                                          "bar":{"color":"#c9a84c"},
                                          "bgcolor":"#141720","bordercolor":"#1f2235",
                                          "steps":[{"range":[0,30],"color":"#0d2b1e"},
                                                   {"range":[30,70],"color":"#141720"},
                                                   {"range":[70,100],"color":"#2b0d0d"}],
                                          "threshold":{"line":{"color":"white","width":2},
                                                       "thickness":0.75,"value":snap.rsi}}))
        rg.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                         font={"color":"#8890a8"},height=200,margin=dict(l=20,r=20,t=30,b=0))
        g1,_=st.columns([1,2])
        with g1: st.plotly_chart(rg,use_container_width=True)

# ── Page: Heatmap ─────────────────────────────────────────────────────────────

def _page_heatmap():
    st.title("🔥 Signal Heatmap")
    st.caption("Confluence score for every symbol × timeframe. Green=bullish · Red=bearish · Grey=no signal")

    sel_tfs=st.multiselect("Timeframes",TIMEFRAMES,default=["15m","1h","4h"],key="hm_tfs")
    if not sel_tfs:
        st.warning("Select at least one timeframe."); return

    symbols=cfg.watchlist.all_symbols
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🔄 Re-scan", type="primary", use_container_width=True):
            st.session_state.pop("hm_rows", None)
            st.session_state.pop("hm_tfs_cached", None)

    # Auto-load on first visit
    cached_tfs = st.session_state.get("hm_tfs_cached")
    if cached_tfs != sel_tfs or "hm_rows" not in st.session_state:
        st.session_state.pop("hm_rows", None)

    total=len(symbols)*len(sel_tfs)
    bar=st.progress(0,text="Scanning markets…")
    rows=[]
    done=0
    for sym in symbols:
        row={"Symbol":sym.replace("=X","").replace("-USD","")}
        for tf in sel_tfs:
            try:
                event=signal_engine.evaluate_symbol(sym,timeframe=tf)
                snap=signal_engine.snapshot_only(sym,timeframe=tf)
                if event:
                    row[tf]=event.signal.score if event.signal.direction=="LONG" else -event.signal.score
                elif snap:
                    row[tf]=20 if snap.trend_direction=="UP" else (-20 if snap.trend_direction=="DOWN" else 0)
                else:
                    row[tf]=0
            except Exception:
                row[tf]=0
            done+=1
            bar.progress(done/total,text=f"Scanned {sym} [{tf}]…")
        rows.append(row)
    bar.empty()
    st.session_state["hm_rows"] = rows
    st.session_state["hm_tfs_cached"] = sel_tfs

    rows = st.session_state["hm_rows"]
    df_heat=pd.DataFrame(rows).set_index("Symbol")
    z=df_heat.values
    text=[[f"{int(v):+d}" if v!=0 else "–" for v in r] for r in z]
    hf=go.Figure(go.Heatmap(z=z,x=sel_tfs,y=df_heat.index.tolist(),
                             text=text,texttemplate="%{text}",
                             colorscale=[[0,"#6b0000"],[0.35,"#e05a5a"],[0.48,"#3b3f5c"],
                                         [0.52,"#3b3f5c"],[0.65,"#2ecc8a"],[1,"#0a5e38"]],
                             zmid=0,zmin=-100,zmax=100,showscale=True,
                             colorbar=dict(tickvals=[-100,-50,0,50,100],
                                           ticktext=["Short 100","Short 50","Neutral","Long 50","Long 100"],
                                           tickfont=dict(color="#8890a8"))))
    hf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                     height=max(300,len(symbols)*38+80),margin=dict(l=0,r=0,t=20,b=0),
                     xaxis=dict(side="top"),yaxis=dict(autorange="reversed"))
    st.plotly_chart(hf,use_container_width=True)

    st.subheader("Confirmed Signals")
    flat=[{"Symbol":r["Symbol"],"Timeframe":tf,"Direction":"LONG" if r.get(tf,0)>0 else "SHORT",
           "Score":abs(r.get(tf,0))}
          for r in rows for tf in sel_tfs if abs(r.get(tf,0))>=cfg.signals.min_confidence_score]
    if flat:
        st.dataframe(pd.DataFrame(flat).sort_values("Score",ascending=False),
                     use_container_width=True,height=250)
    else:
        st.info(f"No confirmed signals above threshold ({cfg.signals.min_confidence_score}).")

# ── Page: Signals ─────────────────────────────────────────────────────────────

def _page_signals():
    st.title("🎯 Signals")
    tab1,tab2=st.tabs(["Recent Signals","Scan Now"])
    with tab1:
        f1,f2,f3=st.columns(3)
        fdir=f1.selectbox("Direction",["All","LONG","SHORT"],key="sf_dir")
        fsym=f2.selectbox("Symbol",["All"]+cfg.watchlist.all_symbols,key="sf_sym")
        fsc=f3.slider("Min Score",0,100,0,key="sf_sc")

        sigs=journal.recent_signals(limit=100)
        if fdir!="All": sigs=[s for s in sigs if s.get("direction")==fdir]
        if fsym!="All": sigs=[s for s in sigs if s.get("symbol")==fsym]
        sigs=[s for s in sigs if (s.get("confidence_score") or 0)>=fsc]
        st.caption(f"{len(sigs)} signals")

        if not sigs:
            st.info("No signals match your filters.")
        else:
            for s in sigs:
                direction=s.get("direction","")
                score=s.get("confidence_score",0) or 0
                entry=s.get("entry_price",0) or 0
                sl=s.get("stop_loss",0) or 0
                tp=s.get("take_profit",0) or 0
                rr=s.get("risk_reward",0) or 0
                ts=""
                if s.get("created_at"):
                    try: ts=datetime.fromisoformat(s["created_at"]).strftime("%b %d %H:%M")
                    except Exception: ts=str(s["created_at"])[:16]
                cc="sig-long" if direction=="LONG" else "sig-short"
                dc="#2ecc8a" if direction=="LONG" else "#e05a5a"
                da="▲" if direction=="LONG" else "▼"
                sc=_score_color(score)
                st.markdown(f"""<div class="sig-card {cc}">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span><strong>{s.get('symbol','')}</strong> <span style="color:#8890a8">[{s.get('timeframe','')}]</span>
                    <span style="color:{dc}"> {da} {direction}</span></span>
                    <span><span style="background:{sc}22;color:{sc};padding:2px 8px;border-radius:5px;font-size:.85em">
                    Score {score}/100</span> <span style="color:#8890a8;font-size:.8em">{ts}</span></span>
                  </div>
                  <div style="margin-top:8px;font-size:.9em;color:#8890a8">
                    Entry <code style="color:#e0e0e0">{entry:.5f}</code> &nbsp;·&nbsp;
                    SL <code style="color:#e05a5a">{sl:.5f}</code> &nbsp;·&nbsp;
                    TP <code style="color:#2ecc8a">{tp:.5f}</code> &nbsp;·&nbsp;
                    R:R <code style="color:#c9a84c">{rr:.2f}</code> &nbsp;·&nbsp;
                    RSI <code>{s.get('rsi') or '–'}</code> &nbsp;·&nbsp;
                    Trend <code>{s.get('ema_trend') or '–'}</code>
                  </div></div>""",unsafe_allow_html=True)
                if s.get("ai_analysis"):
                    with st.expander(f"🤖 AI — {s.get('symbol','')}"): st.write(s["ai_analysis"])

    with tab2:
        st.subheader("On-Demand Scan")
        sc1,sc2,sc3=st.columns(3)
        with sc1: scan_sym=st.selectbox("Symbol",cfg.watchlist.all_symbols,key="scan_sym")
        with sc2: scan_tf=st.selectbox("Timeframe",["5m","15m","1h","4h","1d"],index=2,key="scan_tf")
        with sc3:
            st.markdown("<br/>",unsafe_allow_html=True)
            run_scan=st.button("🔍 Scan",type="primary",use_container_width=True)

        if run_scan:
            with st.spinner(f"Scanning {scan_sym} [{scan_tf}]…"):
                event=signal_engine.evaluate_symbol(scan_sym,timeframe=scan_tf)
            if event:
                s=event.signal
                dc="#2ecc8a" if s.direction=="LONG" else "#e05a5a"
                st.markdown(f"<h3 style='color:{dc}'>{'▲' if s.direction=='LONG' else '▼'} "
                            f"{s.direction} — Score {s.score}/100</h3>",unsafe_allow_html=True)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Entry",f"{s.entry_price:.5f}"); c2.metric("SL",f"{s.stop_loss:.5f}")
                c3.metric("TP",f"{s.take_profit:.5f}"); c4.metric("R:R",f"{s.risk_reward:.2f}:1")
                for reason in s.reasons: st.write(f"• {reason}")
                if s.components:
                    cf=go.Figure(go.Bar(x=list(s.components.values()),y=list(s.components.keys()),
                                        orientation="h",marker_color="#c9a84c"))
                    cf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                     height=220,margin=dict(l=0,r=0,t=10,b=0),
                                     title=dict(text="Score Breakdown",font=dict(color="#c9a84c",size=12)))
                    st.plotly_chart(cf,use_container_width=True)
            else:
                st.info(f"No signal above {cfg.signals.min_confidence_score} for {scan_sym} [{scan_tf}].")

            snap=signal_engine.snapshot_only(scan_sym,timeframe=scan_tf)
            if snap:
                st.divider(); st.subheader("Snapshot")
                c1,c2,c3,c4,c5=st.columns(5)
                c1.metric("Trend",snap.trend_direction); c2.metric("RSI",f"{snap.rsi:.1f}",snap.rsi_signal)
                c3.metric("MACD",snap.macd_crossover); c4.metric("BB",snap.bb_position)
                c5.metric("Vol Ratio",f"{snap.volume_ratio:.2f}×")

# ── Page: Journal ─────────────────────────────────────────────────────────────

def _page_journal():
    st.title("📚 Trade Journal")
    perf=journal.performance_summary(days=30)
    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Trades (30d)",perf.get("total_trades",0))
    k2.metric("Win Rate",f"{perf.get('win_rate',0)*100:.1f}%")
    k3.metric("Total P&L",f"${perf.get('total_pnl',0):+.2f}")
    k4.metric("Avg Win",f"${perf.get('avg_win',0):.2f}")
    k5.metric("Avg Loss",f"${perf.get('avg_loss',0):.2f}")
    k6.metric("Profit Factor",f"{perf.get('profit_factor',0):.2f}")
    st.divider()

    tab1,tab2,tab3=st.tabs(["Trades","Analytics","P&L Calendar"])

    with tab1:
        trades=journal.recent_trades(limit=200)
        if not trades: st.info("No trades recorded yet.")
        else:
            df=pd.DataFrame(trades)
            rename={"opened_at":"Opened","closed_at":"Closed","symbol":"Symbol","direction":"Dir",
                    "entry_price":"Entry","exit_price":"Exit","pnl":"P&L","pnl_pct":"P&L%",
                    "outcome":"Result","position_size":"Size","risk_amount":"Risk $",
                    "emotion_tag":"Emotion","strategy":"Strategy"}
            dcols=[c for c in rename if c in df.columns]
            dd=df[dcols].rename(columns=rename)
            def _cp(v):
                if isinstance(v,(int,float)): return f"color:{'#2ecc8a' if v>0 else '#e05a5a' if v<0 else '#8890a8'}"
                return ""
            pcols=[c for c in ["P&L","P&L%"] if c in dd.columns]
            st.dataframe(dd.style.applymap(_cp,subset=pcols),use_container_width=True,height=450)

    with tab2:
        trades=journal.recent_trades(limit=500)
        if len(trades)<2: st.info("Need at least 2 trades.")
        else:
            df=pd.DataFrame(trades)
            ef=_equity_fig(cfg.account.balance)
            if ef.data: st.plotly_chart(ef,use_container_width=True)
            df2=_drawdown_fig(cfg.account.balance)
            if df2.data: st.plotly_chart(df2,use_container_width=True)

            ca,cb=st.columns(2)
            with ca:
                if "outcome" in df.columns:
                    oc=df["outcome"].value_counts()
                    pf=go.Figure(go.Pie(labels=oc.index.tolist(),values=oc.values.tolist(),
                                        marker_colors=["#2ecc8a","#e05a5a","#c9a84c","#8890a8"],hole=0.45))
                    pf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",height=280,
                                     margin=dict(l=0,r=0,t=30,b=0),
                                     title=dict(text="Outcome Distribution",font=dict(color="#c9a84c")),
                                     legend=dict(font=dict(color="#8890a8")))
                    st.plotly_chart(pf,use_container_width=True)
            with cb:
                if "pnl" in df.columns and "symbol" in df.columns:
                    sp=df.groupby("symbol")["pnl"].sum().sort_values()
                    bf=go.Figure(go.Bar(x=sp.values,y=sp.index.tolist(),orientation="h",
                                        marker_color=["#2ecc8a" if v>=0 else "#e05a5a" for v in sp.values]))
                    bf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                     height=280,margin=dict(l=0,r=0,t=30,b=0),
                                     title=dict(text="P&L by Symbol",font=dict(color="#c9a84c")),
                                     xaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
                    st.plotly_chart(bf,use_container_width=True)

            if "outcome" in df.columns:
                ds=df.sort_values("opened_at").reset_index(drop=True) if "opened_at" in df.columns else df.copy().reset_index(drop=True)
                ds["win"]=ds["outcome"]=="WIN"
                ds["rwr"]=ds["win"].rolling(10,min_periods=1).mean()*100
                wf=go.Figure(go.Scatter(x=ds.index,y=ds["rwr"],mode="lines",
                                        line=dict(color="#c9a84c",width=2),name="Rolling WR"))
                wf.add_hline(y=50,line_dash="dot",line_color="#3b3f5c")
                wf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                 height=220,margin=dict(l=0,r=0,t=30,b=0),
                                 title=dict(text="Rolling Win Rate (10 trades)",font=dict(color="#c9a84c")),
                                 yaxis=dict(ticksuffix="%",showgrid=True,gridcolor="#1a1d2e"),
                                 xaxis=dict(showgrid=True,gridcolor="#1a1d2e",title="Trade #"))
                st.plotly_chart(wf,use_container_width=True)

            if "pnl" in df.columns:
                cb1,cb2=st.columns(2)
                show_cols=[c for c in ["symbol","direction","pnl","pnl_pct","outcome"] if c in df.columns]
                with cb1:
                    st.markdown("**Top 5 Winners**")
                    st.dataframe(df.nlargest(5,"pnl")[show_cols],use_container_width=True,hide_index=True)
                with cb2:
                    st.markdown("**Top 5 Losers**")
                    st.dataframe(df.nsmallest(5,"pnl")[show_cols],use_container_width=True,hide_index=True)

    with tab3:
        trades=journal.recent_trades(limit=500)
        if not trades: st.info("No trades to show.")
        else:
            df=pd.DataFrame(trades)
            if "opened_at" in df.columns and "pnl" in df.columns:
                df["date"]=pd.to_datetime(df["opened_at"]).dt.date
                daily=df.groupby("date")["pnl"].sum().reset_index()
                daily["date"]=pd.to_datetime(daily["date"])
                daily["week"]=daily["date"].dt.isocalendar().week.astype(int)
                daily["dow"]=daily["date"].dt.dayofweek
                daily["month"]=daily["date"].dt.strftime("%b %Y")
                months=daily["month"].unique().tolist()
                sel_m=st.selectbox("Month",months[::-1],key="cal_m")
                mdf=daily[daily["month"]==sel_m]
                if not mdf.empty:
                    weeks=sorted(mdf["week"].unique())
                    days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    z=np.full((7,len(weeks)),np.nan)
                    txt=[["" for _ in weeks] for _ in range(7)]
                    for _,row in mdf.iterrows():
                        wi=weeks.index(row["week"]); di=int(row["dow"])
                        if di<7: z[di][wi]=row["pnl"]; txt[di][wi]=f"${row['pnl']:+.2f}"
                    cf=go.Figure(go.Heatmap(z=z,x=[f"W{w}" for w in weeks],y=days,
                                            text=txt,texttemplate="%{text}",
                                            colorscale=[[0,"#6b0000"],[0.45,"#e05a5a"],
                                                        [0.5,"#1a1d2e"],[0.55,"#2ecc8a"],[1,"#0a5e38"]],
                                            zmid=0,showscale=True,
                                            colorbar=dict(tickfont=dict(color="#8890a8"))))
                    cf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                     height=300,margin=dict(l=0,r=0,t=30,b=0),
                                     title=dict(text=f"Daily P&L — {sel_m}",font=dict(color="#c9a84c")),
                                     xaxis=dict(side="top"),yaxis=dict(autorange="reversed"))
                    st.plotly_chart(cf,use_container_width=True)
            else:
                st.info("Trade date data not available.")

# ── Page: Planner ─────────────────────────────────────────────────────────────

def _page_planner():
    st.title("🧮 Trade Planner")
    st.caption("Calculate exact position size, dollar risk, and multiple TP levels.")
    pl,pr=st.columns([1,1])
    with pl:
        st.subheader("Inputs")
        balance=st.number_input("Account Balance ($)",value=cfg.account.balance,min_value=100.0,step=100.0,key="pl_bal")
        risk_pct=st.slider("Risk per Trade (%)",0.25,5.0,float(cfg.risk.per_trade_pct*100),step=0.25,key="pl_risk")
        asset_type=st.radio("Asset Type",["Forex","Crypto","Other"],horizontal=True,key="pl_type")
        entry_price=st.number_input("Entry Price",value=1.10000,format="%.5f",key="pl_entry")
        sl_price=st.number_input("Stop Loss Price",value=1.09500,format="%.5f",key="pl_sl")
        use_atr=st.checkbox("Suggest SL from ATR",value=False,key="pl_atr")
        if use_atr:
            atr_sym=st.selectbox("Symbol",cfg.watchlist.all_symbols,key="pl_atr_sym")
            atr_tf=st.selectbox("Timeframe",["1h","4h","1d"],key="pl_atr_tf")
            if st.button("Get ATR",key="pl_get_atr"):
                snap=signal_engine.snapshot_only(atr_sym,timeframe=atr_tf)
                if snap:
                    st.success(f"ATR = {snap.atr:.5f}  ({snap.atr_pct*100:.2f}% of price)")
                    st.info(f"1×ATR SL → long: {entry_price-snap.atr:.5f}  |  short: {entry_price+snap.atr:.5f}")
    with pr:
        st.subheader("Position Sizing")
        if entry_price==sl_price:
            st.error("Entry and SL cannot be equal.")
        else:
            dollar_risk=balance*(risk_pct/100)
            sl_dist=abs(entry_price-sl_price)
            direction="LONG" if entry_price>sl_price else "SHORT"
            if asset_type=="Forex":
                pip_sz=0.0001
                sl_pips=sl_dist/pip_sz
                units=dollar_risk/(sl_dist*100_000)
                m1,m2=st.columns(2)
                m1.metric("Dollar Risk",f"${dollar_risk:.2f}"); m2.metric("Direction",direction)
                m1.metric("SL Distance",f"{sl_pips:.1f} pips"); m2.metric("Lots",f"{units:.4f}")
            else:
                units=dollar_risk/sl_dist
                m1,m2=st.columns(2)
                m1.metric("Dollar Risk",f"${dollar_risk:.2f}"); m2.metric("Direction",direction)
                m1.metric("SL Distance",f"{sl_dist:.5f}"); m2.metric("Units",f"{units:.4f}")

            st.subheader("Take-Profit Levels")
            rrs=[1.0,1.5,2.0,2.5,3.0,4.0,5.0]
            rows=[]
            for rr in rrs:
                tp=entry_price+sl_dist*rr if direction=="LONG" else entry_price-sl_dist*rr
                rows.append({"R:R":f"{rr:.1f}:1","TP Price":round(tp,5),
                             "Profit ($)":round(dollar_risk*rr,2),
                             "Return (%)":round(dollar_risk*rr/balance*100,3)})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

            st.divider()
            p90=journal.performance_summary(days=90)
            w=p90.get("win_rate",0.5) or 0.5
            aw=p90.get("avg_win") or 0.0
            al=abs(p90.get("avg_loss") or 0.0)
            if al>0 and aw>0:
                b=aw/al
                kelly=max(0.0,w-(1-w)/b)*100
            else:
                b=0.0; kelly=0.0
            st.markdown(f"**Kelly Criterion (90d):** {kelly:.1f}%  \n"
                        f"Win rate: {w*100:.0f}%  ·  Avg W/L: {f'{b:.2f}' if b else 'N/A'}  \n"
                        f"<small style='color:#8890a8'>½ Kelly = {kelly/2:.1f}% (recommended) "
                        f"{'· Not enough trade history yet' if not (al and aw) else ''}</small>",
                        unsafe_allow_html=True)

# ── Page: Backtest ────────────────────────────────────────────────────────────

def _page_backtest():
    st.title("🔬 Backtest")
    with st.form("bt_form"):
        c1,c2,c3,c4=st.columns(4)
        with c1: bt_sym=st.selectbox("Symbol",cfg.watchlist.all_symbols)
        with c2: bt_tf=st.selectbox("Timeframe",["1h","4h","1d"])
        with c3: bt_start=st.date_input("Start",value=pd.to_datetime(cfg.backtest.default_start))
        with c4: bt_end=st.date_input("End",value=pd.to_datetime(cfg.backtest.default_end))
        min_sc=st.slider("Min Score",40,90,cfg.signals.min_confidence_score)
        run=st.form_submit_button("▶ Run Backtest",type="primary",use_container_width=True)

    if run:
        from msomi.backtest.engine import BacktestEngine
        with st.spinner(f"Backtesting {bt_sym} [{bt_tf}] {bt_start}→{bt_end}…"):
            engine=BacktestEngine(cfg)
            report=engine.run(symbol=bt_sym,timeframe=bt_tf,
                              start=str(bt_start),end=str(bt_end),min_score=min_sc)

        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("Trades",report.total_trades); m2.metric("Win Rate",f"{report.win_rate:.1f}%")
        m3.metric("Sharpe",f"{report.sharpe_ratio:.2f}"); m4.metric("Max DD",f"{report.max_drawdown_pct:.1f}%")
        m5.metric("Return",f"{report.total_return_pct:+.1f}%")
        m6.metric("Profit Factor",f"{report.profit_factor:.2f}" if report.profit_factor else "N/A")

        if report.equity_curve:
            arr=np.array(report.equity_curve)
            ef=go.Figure(go.Scatter(y=arr,mode="lines",line=dict(color="#c9a84c",width=2),
                                    fill="tozeroy",fillcolor="rgba(201,168,76,0.07)"))
            ef.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                             height=320,margin=dict(l=0,r=0,t=10,b=0),
                             title=dict(text=f"Equity — {bt_sym}",font=dict(color="#c9a84c")),
                             xaxis=dict(title="Trade #",showgrid=True,gridcolor="#1a1d2e"),
                             yaxis=dict(title="Equity ($)",showgrid=True,gridcolor="#1a1d2e",side="right"))
            st.plotly_chart(ef,use_container_width=True)

            peak=np.maximum.accumulate(arr); dd=(arr-peak)/peak*100
            ddf=go.Figure(go.Scatter(y=dd,mode="lines",line=dict(color="#e05a5a",width=1.5),
                                     fill="tozeroy",fillcolor="rgba(224,90,90,0.08)"))
            ddf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                              height=200,margin=dict(l=0,r=0,t=10,b=0),
                              title=dict(text="Drawdown",font=dict(color="#e05a5a",size=12)),
                              yaxis=dict(ticksuffix="%",showgrid=True,gridcolor="#1a1d2e",side="right"),
                              xaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
            st.plotly_chart(ddf,use_container_width=True)

        with st.expander("Full Report"): st.code(report.summary())

        if report.trades:
            st.dataframe(pd.DataFrame([{"Entry":t.entry_time,"Exit":t.exit_time,"Dir":t.direction,
                                         "Entry $":round(t.entry_price,5),"Exit $":round(t.exit_price,5),
                                         "P&L%":round(t.pnl_pct,2),"Outcome":t.outcome,"Score":t.score}
                                        for t in report.trades]),use_container_width=True,height=350)

# ── Page: Settings ────────────────────────────────────────────────────────────

def _page_settings():
    st.title("⚙️ Settings")
    cl,cr=st.columns(2)
    with cl:
        with st.expander("Risk",expanded=True):
            rc=cfg.risk
            st.metric("Risk/Trade",f"{rc.per_trade_pct*100:.1f}%")
            st.metric("Daily Loss Limit",f"{rc.daily_loss_limit_pct*100:.1f}%")
            st.metric("Weekly DD Limit",f"{rc.weekly_drawdown_limit_pct*100:.1f}%")
            st.metric("Max Consecutive Losses",rc.max_consecutive_losses)
            st.metric("Min R:R",f"{rc.min_risk_reward}:1")
            st.metric("Max Open Positions",rc.max_open_positions)
        with st.expander("AI"):
            st.metric("Provider",cfg.ai.provider.upper())
            st.metric("Model",cfg.ai.model_anthropic if cfg.ai.provider=="anthropic" else cfg.ai.model_openai)
            st.metric("Max Tokens",cfg.ai.max_tokens)
            st.metric("Temperature",cfg.ai.temperature)
    with cr:
        with st.expander("Signals",expanded=True):
            ind=cfg.signals.indicators
            st.metric("Min Score",cfg.signals.min_confidence_score)
            st.metric("Primary TF",cfg.signals.timeframes.get("primary","1h"))
            st.metric("EMA Fast",ind.ema_fast); st.metric("EMA Slow",ind.ema_slow)
            st.metric("EMA Trend",ind.ema_trend); st.metric("RSI Period",ind.rsi_period)
            st.metric("ATR Period",ind.atr_period); st.metric("BB Period",ind.bb_period)
        with st.expander("Watchlist"):
            st.markdown("**Forex:**"); st.code("  ".join(cfg.watchlist.forex))
            st.markdown("**Crypto:**"); st.code("  ".join(cfg.watchlist.crypto))
    st.info("📝 Edit `config/settings.yaml` to change settings, then restart the dashboard.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='color:#c9a84c;margin-bottom:0'>📡 Msomi</h2>"
                "<p style='color:#8890a8;margin-top:2px;font-size:0.85em'>Trading Intelligence</p>",
                unsafe_allow_html=True)
    st.divider()
    page=st.radio("Navigate",
                  ["Live Feed","Charts","Heatmap","Signals","Journal","Planner","Backtest","Settings"],
                  index=0)
    st.divider()
    state=circuit_breaker.state
    if state.is_tripped:
        st.markdown("🔴 **Circuit Breaker TRIPPED**")
    else:
        streak=state.consecutive_losses
        color="#e05a5a" if streak>=cfg.risk.max_consecutive_losses-1 else "#2ecc8a"
        st.markdown(f"<span style='color:{color}'>🟢 CB OK · {streak} loss streak</span>",
                    unsafe_allow_html=True)
    st.markdown("<br/>",unsafe_allow_html=True)
    for name in SESSIONS:
        st.caption(f"{'🟢' if _session_open(name) else '⚫'} {name}")
    st.divider()
    st.caption(f"v{cfg.app.version} · {cfg.app.env}")
    st.caption(f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}")

# ── Route ─────────────────────────────────────────────────────────────────────

if page=="Live Feed": _page_live_feed()
elif page=="Charts": _page_charts()
elif page=="Heatmap": _page_heatmap()
elif page=="Signals": _page_signals()
elif page=="Journal": _page_journal()
elif page=="Planner": _page_planner()
elif page=="Backtest": _page_backtest()
elif page=="Settings": _page_settings()
