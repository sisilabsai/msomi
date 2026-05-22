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

# ── Analysis helpers ────────────────────────────────────────────────────────

def _next_candle_close(timeframe: str):
    """Return (close_dt_utc, minutes_remaining, friendly_str) for current candle."""
    tf_mins={"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
    mins=tf_mins.get(timeframe,60)
    now=datetime.now(timezone.utc)
    total_m=now.hour*60+now.minute
    candle_end_m=((total_m//mins)+1)*mins
    close_h=(candle_end_m%1440)//60
    close_min=candle_end_m%60
    close_dt=now.replace(hour=close_h,minute=close_min,second=0,microsecond=0)
    if close_dt<=now: close_dt=close_dt+__import__("datetime").timedelta(days=1)
    remaining=max(1,int((close_dt-now).total_seconds()//60))
    h_r=remaining//60; m_r=remaining%60
    friendly=f"{h_r}h {m_r}m" if h_r>0 else f"{m_r}m"
    return close_dt,remaining,friendly

def _plain_english_analysis(snap):
    """Plain-English bullet list explaining current indicator state."""
    lines = []
    if snap.trend_direction == "UP":
        lines.append("✅ **Trend is UP** — price is above its moving averages. Conditions favour buyers.")
    elif snap.trend_direction == "DOWN":
        lines.append("🔴 **Trend is DOWN** — price is below moving averages. Conditions favour sellers.")
    else:
        lines.append("⚪ **Trend is mixed** — no clear direction yet. Best to wait for a clearer signal.")
    if snap.price_vs_ema_trend:
        if "ABOVE" in snap.price_vs_ema_trend:
            lines.append("✅ **Above the 200 EMA** — the most important long-term bullish signal. Institutional money buys above this line.")
        elif "BELOW" in snap.price_vs_ema_trend:
            lines.append("⚠️ **Below the 200 EMA** — long-term trend is bearish. Be very careful with buy trades here.")
    if snap.rsi > 70:
        lines.append(f"⚠️ **RSI {snap.rsi:.0f} — Overbought.** Market moved up too fast. A pullback is likely. Avoid buying, watch for sells.")
    elif snap.rsi < 30:
        lines.append(f"✅ **RSI {snap.rsi:.0f} — Oversold.** Market dropped too fast. A bounce is likely. Watch for a buy opportunity.")
    elif snap.rsi > 55:
        lines.append(f"✅ **RSI {snap.rsi:.0f}** — Bullish momentum, not yet overbought. Good conditions for buyers.")
    elif snap.rsi < 45:
        lines.append(f"🔴 **RSI {snap.rsi:.0f}** — Bearish momentum. Sellers are in control.")
    else:
        lines.append(f"⚪ **RSI {snap.rsi:.0f}** — Neutral. The market is balanced between buyers and sellers.")
    if snap.macd_crossover == "BULLISH":
        lines.append("✅ **MACD bullish crossover** — momentum just flipped upward. This is an entry signal many traders use.")
    elif snap.macd_crossover == "BEARISH":
        lines.append("🔴 **MACD bearish crossover** — momentum just flipped downward. Sellers are taking control.")
    elif snap.macd_hist > 0:
        lines.append("⚪ **MACD histogram positive** — bullish momentum ongoing, no fresh crossover yet.")
    else:
        lines.append("⚪ **MACD histogram negative** — bearish pressure, no fresh crossover yet.")
    if snap.bb_position:
        bp = snap.bb_position.upper()
        if "UPPER" in bp:
            lines.append("⚠️ **Near upper Bollinger Band** — price is stretched to the upside. Possible reversal zone. Don't chase longs here.")
        elif "LOWER" in bp:
            lines.append("✅ **Near lower Bollinger Band** — price has stretched to the downside. Possible bounce zone.")
        else:
            lines.append("⚪ **Inside Bollinger Bands** — market is in a normal range, not at an extreme.")
    if snap.bb_squeeze:
        lines.append("⚡ **Bollinger Band Squeeze** — volatility compressing. A large breakout move is building. Watch for direction!")
    if snap.volume_spike:
        lines.append(f"🔥 **Volume spike ({snap.volume_ratio:.1f}× normal)** — professionals are actively moving in. This confirms the move.")
    elif snap.volume_ratio > 1.5:
        lines.append(f"📈 **Above-average volume ({snap.volume_ratio:.1f}×)** — good confirmation of the move.")
    return lines

def _verdict_from_snap(snap, event=None):
    """Return (label, color, emoji, score, subtitle) for current conditions."""
    if event:
        s = event.signal
        if s.direction == "LONG":
            label = "STRONG BUY" if s.score >= 75 else "BUY"
            color = "#2ecc8a"; emoji = "⬆"
        else:
            label = "STRONG SELL" if s.score >= 75 else "SELL"
            color = "#e05a5a"; emoji = "⬇"
        return label, color, emoji, s.score, f"Confluence score: {s.score}/100"
    if snap is None:
        return "NO DATA", "#3b3f5c", "–", 0, "Could not load data"
    bull = 0; bear = 0
    if snap.trend_direction == "UP": bull += 2
    elif snap.trend_direction == "DOWN": bear += 2
    if snap.price_vs_ema_trend and "ABOVE" in (snap.price_vs_ema_trend or ""): bull += 1
    elif snap.price_vs_ema_trend and "BELOW" in (snap.price_vs_ema_trend or ""): bear += 1
    if snap.rsi < 40: bull += 1
    elif snap.rsi > 60: bear += 1
    if snap.macd_crossover == "BULLISH": bull += 2
    elif snap.macd_crossover == "BEARISH": bear += 2
    net = bull - bear
    score = min(90, 50 + abs(net) * 7)
    if net >= 4: return "STRONG BUY", "#2ecc8a", "⬆", score, f"{bull} bullish signals, {bear} bearish"
    elif net >= 2: return "BUY", "#2ecc8a", "↑", score, f"Bullish leaning — {bull} bull vs {bear} bear"
    elif net <= -4: return "STRONG SELL", "#e05a5a", "⬇", score, f"{bear} bearish signals, {bull} bullish"
    elif net <= -2: return "SELL", "#e05a5a", "↓", score, f"Bearish leaning — {bear} bear vs {bull} bull"
    else: return "WAIT / NEUTRAL", "#8890a8", "↔", 50, f"Mixed signals — no clear edge ({bull} bull, {bear} bear)"

def _strategy_health(report):
    """Return (emoji, color, rating, description) for a backtest report."""
    if report.total_trades < 5:
        return "⚠️", "#c9a84c", "Too few trades", "Run over a longer period or lower the min score for meaningful results."
    score = 0
    if report.win_rate >= 55: score += 2
    elif report.win_rate >= 45: score += 1
    pf = report.profit_factor if report.profit_factor != float("inf") else 999
    if pf >= 2.0: score += 2
    elif pf >= 1.5: score += 1
    if report.sharpe_ratio >= 1.5: score += 2
    elif report.sharpe_ratio >= 0.5: score += 1
    if report.max_drawdown_pct <= 10: score += 2
    elif report.max_drawdown_pct <= 20: score += 1
    if score >= 7: return "🏆", "#2ecc8a", "Excellent", "Strong historical performance. High win rate, low drawdown, positive expectancy."
    elif score >= 5: return "✅", "#7bc67e", "Good", "Solid strategy. Profitable with manageable risk."
    elif score >= 3: return "⚠️", "#c9a84c", "Average", "Makes money but needs better market conditions or refined parameters."
    else: return "❌", "#e05a5a", "Poor", "This configuration has not performed well historically. Try different settings."

# ── Page: Trade Analyzer ─────────────────────────────────────────────────────

def _page_charts():
    st.title("🎯 Trade Analyzer")
    st.caption("Pick a market. Get a clear recommendation with exact entry, stop loss, and take profit.")
    c1,c2,c3=st.columns([2,1,1])
    with c1: symbol=st.selectbox("Symbol",cfg.watchlist.all_symbols,key="chart_sym")
    with c2: timeframe=st.selectbox("Timeframe",["5m","15m","1h","4h","1d"],index=2,key="chart_tf")
    with c3: periods=st.selectbox("Candles",[60,120,200,500],index=1,key="chart_periods")

    o1,o2,o3=st.columns(3)
    show_emas=o1.checkbox("EMA Lines",value=True)
    show_bb=o2.checkbox("Bollinger Bands",value=True)
    show_vwap=o3.checkbox("VWAP",value=True)

    with st.spinner(f"Analyzing {symbol} [{timeframe}]…"):
        fig,df=_get_chart_df(symbol,timeframe,periods,show_emas,show_bb,show_vwap)
        event=signal_engine.evaluate_symbol(symbol,timeframe=timeframe)
        snap=signal_engine.snapshot_only(symbol,timeframe=timeframe)
        price=fetch_latest_price(symbol)

    if not fig.data:
        st.warning("Could not load data for this symbol."); return
    st.plotly_chart(fig,use_container_width=True)

    # ── RSI / MACD / Volume sub-panels ─────────────────────────────────────
    if df is not None and not df.empty:
        tail=60
        s1,s2,s3=st.columns(3)
        with s1:
            rsi_fig=go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index[-tail:],y=df["rsi"].iloc[-tail:],
                                         mode="lines",line=dict(color="#c9a84c",width=1.5),name="RSI"))
            rsi_fig.add_hline(y=70,line_dash="dot",line_color="#e05a5a",line_width=1)
            rsi_fig.add_hline(y=30,line_dash="dot",line_color="#2ecc8a",line_width=1)
            rsi_fig.add_hrect(y0=70,y1=100,fillcolor="rgba(224,90,90,0.05)",line_width=0)
            rsi_fig.add_hrect(y0=0,y1=30,fillcolor="rgba(46,204,138,0.05)",line_width=0)
            rsi_fig.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                  height=180,margin=dict(l=0,r=0,t=30,b=0),showlegend=False,
                                  title=dict(text="RSI",font=dict(color="#c9a84c",size=12)),
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
                              height=180,margin=dict(l=0,r=0,t=30,b=0),showlegend=False,
                              title=dict(text="MACD",font=dict(color="#c9a84c",size=12)),
                              xaxis=dict(showgrid=True,gridcolor="#1a1d2e"),
                              yaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
            st.plotly_chart(mf,use_container_width=True)
        with s3:
            vc=["#2ecc8a" if c>=o else "#e05a5a"
                for c,o in zip(df["close"].iloc[-tail:],df["open"].iloc[-tail:])]
            vf=go.Figure(go.Bar(x=df.index[-tail:],y=df["volume"].iloc[-tail:],marker_color=vc,name="Volume"))
            vf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                             height=180,margin=dict(l=0,r=0,t=30,b=0),showlegend=False,
                             title=dict(text="Volume",font=dict(color="#c9a84c",size=12)),
                             xaxis=dict(showgrid=True,gridcolor="#1a1d2e"),
                             yaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
            st.plotly_chart(vf,use_container_width=True)

    st.divider()

    # ── Trade Recommendation ─────────────────────────────────────────────────
    st.markdown("### What should I do?")
    if snap is None:
        st.warning("Could not compute indicators. Try a different symbol or timeframe."); return

    label,color,emoji,score,subtitle=_verdict_from_snap(snap,event)
    analysis=_plain_english_analysis(snap)
    left,right=st.columns([1,2])

    with left:
        st.markdown(f"""
        <div style="background:{color}15;border:2px solid {color};border-radius:16px;
             padding:28px 20px;text-align:center;margin-bottom:16px">
          <div style="font-size:2.8em;margin-bottom:4px">{emoji}</div>
          <div style="color:{color};font-size:1.7em;font-weight:700;letter-spacing:1px">{label}</div>
          <div style="color:#8890a8;font-size:.85em;margin-top:8px">{subtitle}</div>
        </div>""",unsafe_allow_html=True)
        if event:
            s=event.signal
            sl_dist=abs(s.entry_price-s.stop_loss)
            dollar_risk=cfg.account.balance*cfg.risk.per_trade_pct
            if "=X" in symbol:
                size_str=f"{dollar_risk/(sl_dist*100_000):.4f} lots" if sl_dist>0 else "–"
            else:
                size_str=f"{dollar_risk/sl_dist:.4f} units" if sl_dist>0 else "–"
            potential_profit=dollar_risk*s.risk_reward
            st.markdown(f"""
            <div style="background:#141720;border-radius:12px;padding:16px">
              <div style="color:#8890a8;font-size:.75em;text-transform:uppercase;margin-bottom:12px;letter-spacing:1px">Trade Plan</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div><div style="color:#8890a8;font-size:.7em">Entry</div>
                  <div style="font-family:monospace;color:#e0e0e0;font-weight:600">{s.entry_price:.5f}</div></div>
                <div><div style="color:#8890a8;font-size:.7em">Stop Loss</div>
                  <div style="font-family:monospace;color:#e05a5a;font-weight:600">{s.stop_loss:.5f}</div></div>
                <div><div style="color:#8890a8;font-size:.7em">Take Profit</div>
                  <div style="font-family:monospace;color:#2ecc8a;font-weight:600">{s.take_profit:.5f}</div></div>
                <div><div style="color:#8890a8;font-size:.7em">R:R Ratio</div>
                  <div style="color:#c9a84c;font-weight:600">{s.risk_reward:.1f}:1</div></div>
              </div>
              <div style="margin-top:14px;border-top:1px solid #1f2235;padding-top:12px">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                  <div><div style="color:#8890a8;font-size:.7em">Position Size</div>
                    <div style="color:#e0e0e0;font-weight:600">{size_str}</div></div>
                  <div><div style="color:#8890a8;font-size:.7em">Max You Lose</div>
                    <div style="color:#e05a5a;font-weight:600">${dollar_risk:.2f}</div></div>
                  <div><div style="color:#8890a8;font-size:.7em">Potential Gain</div>
                    <div style="color:#2ecc8a;font-weight:600">${potential_profit:.2f}</div></div>
                  <div><div style="color:#8890a8;font-size:.7em">Account Risk</div>
                    <div style="color:#c9a84c;font-weight:600">{cfg.risk.per_trade_pct*100:.1f}%</div></div>
                </div>
              </div>
            </div>""",unsafe_allow_html=True)
        else:
            st.info("No confirmed signal at the current threshold.  \nTry a different timeframe, or check the **Live Feed** for active opportunities.")
            if snap.atr_pct>0:
                ep=price or snap.vwap or 1.0
                sl_long=ep-snap.atr; sl_short=ep+snap.atr
                tp_long=ep+snap.atr*cfg.risk.min_risk_reward
                tp_short=ep-snap.atr*cfg.risk.min_risk_reward
                st.markdown(f"""**ATR-based levels** (if you want to set manual orders):  
*Long:* Entry `{ep:.5f}` · SL `{sl_long:.5f}` · TP `{tp_long:.5f}`  
*Short:* Entry `{ep:.5f}` · SL `{sl_short:.5f}` · TP `{tp_short:.5f}`""")

    with right:
        st.markdown("#### What each indicator is saying")
        for line in analysis:
            st.markdown(line)
        if snap.bb_squeeze:
            st.warning("⚡ **Action required:** A breakout is forming. Set a price alert and be ready to act quickly.")
        if event:
            st.markdown("#### Why this signal fired")
            for reason in event.signal.reasons:
                st.markdown(f"• {reason}")
            if event.signal.components:
                cf=go.Figure(go.Bar(x=list(event.signal.components.values()),
                                    y=list(event.signal.components.keys()),
                                    orientation="h",marker_color="#c9a84c"))
                cf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                  height=200,margin=dict(l=0,r=0,t=10,b=0),
                                  title=dict(text="Score Breakdown",font=dict(color="#c9a84c",size=12)),
                                  xaxis=dict(showgrid=True,gridcolor="#1a1d2e"),
                                  yaxis=dict(showgrid=True,gridcolor="#1a1d2e"))
                st.plotly_chart(cf,use_container_width=True)

    # ── Exact Next Steps ──────────────────────────────────────────────────
    if snap and event and label not in ("WAIT / NEUTRAL","NO DATA"):
        st.divider()
        s=event.signal
        sym_clean=symbol.replace("=X","").replace("-USD","")
        close_dt,mins_left,friendly_left=_next_candle_close(timeframe)
        close_str=close_dt.strftime("%H:%M UTC")
        dollar_risk=cfg.account.balance*cfg.risk.per_trade_pct
        pot_gain=dollar_risk*s.risk_reward
        direction_word="BUY" if s.direction=="LONG" else "SELL"
        atr_guard=snap.atr
        is_forex="=X" in symbol
        sl_pips=abs(s.entry_price-s.stop_loss)/0.0001 if is_forex else abs(s.entry_price-s.stop_loss)
        unit_label="pips" if is_forex else "pts"
        dc="#2ecc8a" if s.direction=="LONG" else "#e05a5a"
        st.markdown(f"<h3 style='color:{dc}'>Your Exact Next Steps — Do This Now</h3>",unsafe_allow_html=True)
        steps=[
            f"📱 **Open your broker app** (PocketOption, MetaTrader, or whichever you use)",
            f"🔍 **Search for `{sym_clean}`** and open its chart",
            f"📊 **Confirm the price is near `{s.entry_price:.5f}`** — if price has moved more than `{atr_guard:.5f}` ({atr_guard/s.entry_price*100:.2f}%), skip this trade and wait for the next candle",
            f"{'\u2705' if s.direction=='LONG' else '\u274c'} **Place a {direction_word} order at `{s.entry_price:.5f}`** ({sym_clean})",
            f"🛑 **Set Stop Loss at `{s.stop_loss:.5f}`** — that's `{sl_pips:.1f} {unit_label}` away. This limits your max loss to **${dollar_risk:.2f}** ({cfg.risk.per_trade_pct*100:.0f}% of account)",
            f"🎯 **Set Take Profit at `{s.take_profit:.5f}`** — if price reaches this you earn **${pot_gain:.2f}** (R:R = {s.risk_reward:.1f}:1)",
            f"⏰ **Come back at {close_str}** (in ~{friendly_left}) when the {timeframe} candle closes — then re-evaluate",
            f"📝 **Screenshot your entry** or click Save below to track your result",
        ]
        for i,step in enumerate(steps,1):
            st.markdown(f"**{i}.** {step}")
        st.info(f"⚠️ Always use your Stop Loss. Never risk money you can't afford to lose. Signal confidence: **{s.score}/100**.")
        if st.button("📌 Save signal to AI Tracker",key="ta_save_sig",type="secondary"):
            try:
                journal.log_signal(event,ai_analysis=f"Score {s.score} | " + " | ".join(s.reasons[:3]))
                st.success("✅ Saved! Go to **AI Tracker** to monitor the result.")
            except Exception as exc:
                st.error(f"Could not save: {exc}")

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

# ── Page: Strategy Lab ───────────────────────────────────────────────────────

def _page_backtest():
    st.title("🔬 Strategy Lab")
    st.caption("See how this strategy would have performed historically. Build confidence before risking real money.")

    # Defaults (overridden by expander widgets when open)
    bt_sym=cfg.watchlist.all_symbols[0]; bt_tf="1h"
    bt_start=pd.to_datetime(cfg.backtest.default_start).date()
    bt_end=pd.to_datetime(cfg.backtest.default_end).date()
    min_sc=40; run_bt=False

    with st.expander("⚙️ Settings", expanded=False):
        c1,c2,c3,c4=st.columns(4)
        with c1: bt_sym=st.selectbox("Symbol",cfg.watchlist.all_symbols,key="sl_sym")
        with c2: bt_tf=st.selectbox("Timeframe",["1h","4h","1d"],key="sl_tf")
        with c3: bt_start=st.date_input("Start",value=pd.to_datetime(cfg.backtest.default_start),key="sl_start")
        with c4: bt_end=st.date_input("End",value=pd.to_datetime(cfg.backtest.default_end),key="sl_end")
        min_sc=st.slider(
            "Minimum Signal Score — lower = more trades found, higher = only the best setups",
            30,90,40,
            help="Your live threshold is 65. Set to 40 here to get enough historical trades to analyse."
        )
        run_bt=st.button("▶ Run Simulation",type="primary",use_container_width=True,key="sl_run")

    # Auto-run on first load
    cache_key=f"sl_{bt_sym}_{bt_tf}_{min_sc}_{bt_start}_{bt_end}"
    if "sl_cache_key" not in st.session_state or st.session_state["sl_cache_key"]!=cache_key:
        run_bt=True

    if run_bt:
        from msomi.backtest.engine import BacktestEngine
        with st.spinner(f"Simulating {bt_sym} [{bt_tf}] {bt_start} → {bt_end} at min score {min_sc}…"):
            engine=BacktestEngine(cfg)
            report=engine.run(symbol=bt_sym,timeframe=bt_tf,
                              start=str(bt_start),end=str(bt_end),min_score=min_sc)
        st.session_state["sl_report"]=report
        st.session_state["sl_cache_key"]=cache_key

    report=st.session_state.get("sl_report")
    if report is None:
        st.info("Configure settings above and click **Run Simulation**."); return

    # ── Zero trades explanation ───────────────────────────────────────────────
    if report.total_trades==0:
        st.markdown(f"""
        <div style="background:#1a1520;border-left:4px solid #c9a84c;border-radius:10px;padding:20px 24px">
          <h4 style="color:#c9a84c;margin:0 0 8px">⚠️ No trades found (min score {min_sc})</h4>
          <p style="color:#8890a8;margin:0">The strategy did not find signals above {min_sc} for {bt_sym} [{bt_tf}] in this period.</p>
          <p style="color:#8890a8;margin:8px 0 0"><strong style="color:#e0e0e0">Try this:</strong><br/>
          • Open Settings above and lower the min score to <strong>30–40</strong><br/>
          • Extend the date range (start earlier)<br/>
          • Try a more volatile pair like <strong>BTC-USD</strong> or a higher timeframe like <strong>4h</strong></p>
        </div>""",unsafe_allow_html=True)
        return

    # ── Strategy Health card ──────────────────────────────────────────────────
    h_emoji,h_color,h_rating,h_desc=_strategy_health(report)
    scale=cfg.account.balance/report.initial_capital
    user_final=report.final_capital*scale
    user_profit=user_final-cfg.account.balance
    st.markdown(f"""
    <div style="background:{h_color}12;border:1px solid {h_color}44;border-left:5px solid {h_color};
         border-radius:12px;padding:20px 24px;margin-bottom:20px">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
        <div>
          <span style="font-size:1.8em">{h_emoji}</span>
          <span style="color:{h_color};font-size:1.3em;font-weight:700;margin-left:10px">{h_rating} Strategy</span>
          <div style="color:#8890a8;margin-top:6px">{h_desc}</div>
        </div>
        <div style="text-align:right">
          <div style="color:#8890a8;font-size:.8em">Starting with your ${cfg.account.balance:,.0f}</div>
          <div style="font-size:1.6em;font-weight:700;color:{'#2ecc8a' if user_profit>=0 else '#e05a5a'}">${user_final:,.2f}</div>
          <div style="color:{'#2ecc8a' if user_profit>=0 else '#e05a5a'};font-size:.9em">
            {'+' if user_profit>=0 else ''}{user_profit:,.2f} ({report.total_return_pct:+.1f}%)</div>
        </div>
      </div>
    </div>""",unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1,m2,m3,m4,m5,m6=st.columns(6)
    m1.metric("Total Trades",report.total_trades)
    m2.metric("Win Rate",f"{report.win_rate:.1f}%",delta=f"{report.win_rate-50:+.1f}% vs 50%")
    pf_val=report.profit_factor if report.profit_factor!=float("inf") else 999
    m3.metric("Profit Factor",f"{pf_val:.2f}" if pf_val<999 else "∞")
    m4.metric("Sharpe Ratio",f"{report.sharpe_ratio:.2f}")
    m5.metric("Max Drawdown",f"{report.max_drawdown_pct:.1f}%")
    m6.metric("Return",f"{report.total_return_pct:+.1f}%")
    st.divider()

    # ── Plain-English explanation ─────────────────────────────────────────────
    with st.expander("📖 What do these numbers mean?",expanded=True):
        wr=report.win_rate; dd=report.max_drawdown_pct; ret=report.total_return_pct
        wr_txt="excellent" if wr>=60 else "good" if wr>=50 else "below average"
        pf_txt="strong" if pf_val>=2 else "positive" if pf_val>=1 else "losing"
        dd_txt="very low" if dd<=10 else "moderate" if dd<=20 else "high"
        st.markdown(
            f"Over **{report.start_date} → {report.end_date}** this strategy executed **{report.total_trades} trades** "
            f"on **{bt_sym} [{bt_tf}]**.\n\n"
            f"- **{report.wins} wins** and **{report.losses} losses** — a **{wr:.1f}%** win rate, which is {wr_txt}. "
            f"*(Even a 45% win rate can be profitable if your wins are bigger than your losses.)*\n"
            f"- **Profit factor {pf_val:.2f}** — for every $1 lost, you made ${pf_val:.2f}. "
            f"Above 1.0 = profitable; above 1.5 = {pf_txt}.\n"
            f"- **Max drawdown {dd:.1f}%** — the worst peak-to-trough loss. "
            f"That's ${cfg.account.balance*dd/100:.0f} on your ${cfg.account.balance:,.0f} account. "
            f"{dd_txt.title()} risk.\n"
            f"- **Total return {ret:+.1f}%** — your ${cfg.account.balance:,.0f} would have become **${user_final:,.0f}** over this period."
        )

    # ── Equity curve with trade markers ──────────────────────────────────────
    if report.equity_curve:
        arr=np.array(report.equity_curve)*scale
        idx=list(range(len(arr)))
        ef=go.Figure()
        ef.add_trace(go.Scatter(x=idx,y=arr,mode="lines",
                                line=dict(color="#c9a84c",width=2.5),
                                fill="tozeroy",fillcolor="rgba(201,168,76,0.07)",
                                name="Account Balance"))
        ef.add_hline(y=cfg.account.balance,line_dash="dot",line_color="#3b3f5c",
                     annotation_text=f"Start ${cfg.account.balance:,.0f}",
                     annotation_font_color="#8890a8")
        if report.trades:
            win_x=[i+1 for i,t in enumerate(report.trades) if t.outcome=="WIN" and i+1<len(arr)]
            win_y=[arr[i] for i in win_x]
            loss_x=[i+1 for i,t in enumerate(report.trades) if t.outcome=="LOSS" and i+1<len(arr)]
            loss_y=[arr[i] for i in loss_x]
            ef.add_trace(go.Scatter(x=win_x,y=win_y,mode="markers",
                                    marker=dict(symbol="triangle-up",size=10,color="#2ecc8a"),name="Win"))
            ef.add_trace(go.Scatter(x=loss_x,y=loss_y,mode="markers",
                                    marker=dict(symbol="triangle-down",size=10,color="#e05a5a"),name="Loss"))
        ef.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                         height=360,margin=dict(l=0,r=0,t=10,b=0),
                         title=dict(text=f"Account Balance per Trade — {bt_sym} [{bt_tf}]",font=dict(color="#c9a84c")),
                         xaxis=dict(title="Trade #",showgrid=True,gridcolor="#1a1d2e"),
                         yaxis=dict(title="Balance ($)",showgrid=True,gridcolor="#1a1d2e",side="right",tickprefix="$"),
                         legend=dict(orientation="h",font=dict(color="#8890a8"),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(ef,use_container_width=True)

        peak=np.maximum.accumulate(arr); dd_arr=(arr-peak)/peak*100
        ddf=go.Figure(go.Scatter(x=idx,y=dd_arr,mode="lines",
                                  line=dict(color="#e05a5a",width=1.5),
                                  fill="tozeroy",fillcolor="rgba(224,90,90,0.08)"))
        ddf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                           height=180,margin=dict(l=0,r=0,t=10,b=0),
                           title=dict(text="Drawdown from peak",font=dict(color="#e05a5a",size=12)),
                           yaxis=dict(ticksuffix="%",showgrid=True,gridcolor="#1a1d2e",side="right"),
                           xaxis=dict(title="Trade #",showgrid=True,gridcolor="#1a1d2e"))
        st.plotly_chart(ddf,use_container_width=True)

    # ── Win/Loss breakdown ────────────────────────────────────────────────────
    if report.trades:
        ca,cb=st.columns(2)
        with ca:
            wlf=go.Figure(go.Bar(x=["Wins","Losses","Breakeven"],
                                  y=[report.wins,report.losses,report.breakevens],
                                  marker_color=["#2ecc8a","#e05a5a","#8890a8"],
                                  text=[report.wins,report.losses,report.breakevens],
                                  textposition="outside"))
            wlf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                               height=220,margin=dict(l=0,r=0,t=30,b=0),
                               title=dict(text="Win / Loss Split",font=dict(color="#c9a84c")),
                               yaxis=dict(showgrid=True,gridcolor="#1a1d2e"),xaxis=dict(showgrid=False))
            st.plotly_chart(wlf,use_container_width=True)
        with cb:
            pnl_vals=[t.pnl_pct for t in report.trades]
            colors=["#2ecc8a" if v>=0 else "#e05a5a" for v in pnl_vals]
            pnlf=go.Figure(go.Bar(x=list(range(1,len(pnl_vals)+1)),y=pnl_vals,
                                   marker_color=colors,name="P&L %"))
            pnlf.add_hline(y=0,line_color="#3b3f5c",line_width=1)
            pnlf.update_layout(template="plotly_dark",paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                                height=220,margin=dict(l=0,r=0,t=30,b=0),
                                title=dict(text="P&L per Trade (%)",font=dict(color="#c9a84c")),
                                xaxis=dict(title="Trade #",showgrid=True,gridcolor="#1a1d2e"),
                                yaxis=dict(ticksuffix="%",showgrid=True,gridcolor="#1a1d2e",side="right"))
            st.plotly_chart(pnlf,use_container_width=True)

        with st.expander("📋 All Trades",expanded=False):
            rows=[{"#":i+1,
                   "Date":t.entry_time.strftime("%Y-%m-%d") if hasattr(t.entry_time,"strftime") else str(t.entry_time)[:10],
                   "Dir":t.direction,
                   "Entry":round(t.entry_price,5),"Exit":round(t.exit_price,5),
                   "P&L%":round(t.pnl_pct,2),
                   "P&L $":round(t.pnl*(cfg.account.balance/report.initial_capital),2),
                   "Result":t.outcome,"Score":t.score}
                  for i,t in enumerate(report.trades)]
            df_t=pd.DataFrame(rows)
            def _co(v): return "color:#2ecc8a" if v=="WIN" else "color:#e05a5a" if v=="LOSS" else ""
            def _cp(v): return f"color:{'#2ecc8a' if v>=0 else '#e05a5a'}"
            st.dataframe(df_t.style.applymap(_co,subset=["Result"]).applymap(_cp,subset=["P&L%","P&L $"]),
                         use_container_width=True,hide_index=True,height=400)

# ── Page: AI Prediction Tracker ──────────────────────────────────────────────────

def _page_predictions():
    st.title("🤖 AI Signal Tracker")
    st.caption("Every signal this system fires is auto-tracked here. Watch predictions play out against the real market.")

    sigs=journal.recent_signals(limit=200)
    if not sigs:
        st.info("🔍 No signals tracked yet.  \nGo to **Trade Analyzer** or wait for **Live Feed** to auto-scan. Every fired signal is saved automatically.")
        return

    # Resolve each signal against current price
    predictions=[]
    correct=0; resolved=0
    for sig in sigs:
        sym=sig.get("symbol","")
        direction=sig.get("direction","")
        entry=sig.get("entry_price") or 0
        tp=sig.get("take_profit") or 0
        sl=sig.get("stop_loss") or 0
        score=sig.get("confidence_score") or 0
        ts=sig.get("created_at")
        try: current=fetch_latest_price(sym)
        except Exception: current=None
        if current and entry:
            if direction=="LONG":
                pnl_pct=(current-entry)/entry*100
                if current>=tp: outcome="✅ TP HIT"; oc="#2ecc8a"; correct+=1; resolved+=1
                elif current<=sl: outcome="❌ SL HIT"; oc="#e05a5a"; resolved+=1
                elif current>entry: outcome="📈 In Profit"; oc="#7bc67e"
                else: outcome="📉 In Loss"; oc="#c9a84c"
            else:
                pnl_pct=(entry-current)/entry*100
                if current<=tp: outcome="✅ TP HIT"; oc="#2ecc8a"; correct+=1; resolved+=1
                elif current>=sl: outcome="❌ SL HIT"; oc="#e05a5a"; resolved+=1
                elif current<entry: outcome="📈 In Profit"; oc="#7bc67e"
                else: outcome="📉 In Loss"; oc="#c9a84c"
        else:
            pnl_pct=0; outcome="⏳ Loading…"; oc="#3b3f5c"
        predictions.append(dict(sym=sym,direction=direction,entry=entry,tp=tp,sl=sl,
                                score=score,ts=ts,current=current,pnl_pct=pnl_pct,
                                outcome=outcome,oc=oc))

    # ── Summary header ───────────────────────────────────────────────────
    accuracy=correct/resolved*100 if resolved>0 else 0
    acc_color="#2ecc8a" if accuracy>=55 else "#c9a84c" if accuracy>=45 else "#e05a5a"
    a1,a2,a3,a4=st.columns(4)
    a1.metric("Total Signals",len(predictions))
    a2.metric("Resolved",resolved,help="TP or SL was hit")
    a3.metric("AI Accuracy",f"{accuracy:.0f}%",delta=f"{accuracy-50:+.0f}% vs random")
    a4.metric("Pending",len(predictions)-resolved,help="Trade still open")
    if resolved>=5:
        st.markdown(f"""
        <div style="background:{acc_color}12;border-left:4px solid {acc_color};border-radius:8px;padding:14px 20px;margin:10px 0">
          <strong style="color:{acc_color};font-size:1.1em">
            {'This AI has been right ' + str(correct) + ' out of ' + str(resolved) + ' resolved calls (' + f"{accuracy:.0f}" + '%).'}
          </strong><br/>
          <span style="color:#8890a8;font-size:.9em">
            {'Solid edge.' if accuracy>=60 else 'Room to improve — review the losing signals for patterns.' if accuracy>=45 else 'More data needed. Check if market conditions have changed.'}
          </span>
        </div>""",unsafe_allow_html=True)
    st.divider()

    # ── Prediction cards ───────────────────────────────────────────────────
    for p in predictions[:40]:
        dc="#2ecc8a" if p["direction"]=="LONG" else "#e05a5a"
        arrow="▲" if p["direction"]=="LONG" else "▼"
        sym_c=(p["sym"] or "").replace("=X","").replace("-USD","")
        price_str=f"{p['current']:.5f}" if p["current"] else "–"
        pnl_str=f"{p['pnl_pct']:+.2f}%" if p["pnl_pct"] else "–"
        pnl_c="#2ecc8a" if p["pnl_pct"]>0 else "#e05a5a" if p["pnl_pct"]<0 else "#8890a8"
        ts_str=""
        if p["ts"]:
            try: ts_str=datetime.fromisoformat(str(p["ts"])).strftime("%b %d %H:%M")
            except Exception: ts_str=str(p["ts"])[:16]
        sc=_score_color(p["score"])
        # movement description
        if p["current"] and p["entry"]:
            move_pct=abs(p["pnl_pct"])
            move_dir="rose" if p["pnl_pct"]>0 else "fell"
            if p["direction"]=="LONG":
                verdict_text=f"Price {move_dir} {move_pct:.2f}% — {'in your favour' if p['pnl_pct']>0 else 'against you'}"
            else:
                verdict_text=f"Price {move_dir} {move_pct:.2f}% — {'in your favour' if p['pnl_pct']>0 else 'against you'}"
        else:
            verdict_text="Waiting for price data…"
        st.markdown(f"""
        <div style="background:#141720;border-radius:10px;padding:14px 18px;margin-bottom:8px;
             border-left:4px solid {p['oc']}">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
            <div>
              <span style="font-size:1.1em;font-weight:700;color:#e0e0e0">{sym_c}</span>
              <span style="color:{dc};font-weight:600;margin:0 10px">{arrow} {p['direction']}</span>
              <span style="background:{sc}22;color:{sc};padding:2px 8px;border-radius:5px;font-size:.82em">Score {p['score']}</span>
            </div>
            <div style="text-align:right">
              <span style="color:{p['oc']};font-weight:700;font-size:1.05em">{p['outcome']}</span>
              <span style="color:{pnl_c};margin-left:12px;font-weight:600">{pnl_str}</span>
              <span style="color:#8890a8;font-size:.8em;margin-left:12px">{ts_str}</span>
            </div>
          </div>
          <div style="margin-top:8px;display:flex;gap:20px;font-size:.85em;flex-wrap:wrap">
            <span style="color:#8890a8">Predicted entry <code style="color:#e0e0e0">{p['entry']:.5f}</code></span>
            <span style="color:#8890a8">Current <code style="color:#c9a84c">{price_str}</code></span>
            <span style="color:#8890a8">TP <code style="color:#2ecc8a">{p['tp']:.5f}</code></span>
            <span style="color:#8890a8">SL <code style="color:#e05a5a">{p['sl']:.5f}</code></span>
          </div>
          <div style="margin-top:6px;color:#5a6080;font-size:.82em">{verdict_text}</div>
        </div>""",unsafe_allow_html=True)

# ── Page: Settings ────────────────────────────────────────────────────────────

def _page_settings():
    st.title("⚙️ Settings")
    st.caption("Adjust your trading parameters. Changes are written to `config/settings.yaml` instantly.")

    import yaml as _yaml
    _settings_path=os.path.normpath(
        os.path.join(os.path.dirname(__file__),"..","..","..","config","settings.yaml")
    )

    ind=cfg.signals.indicators
    tab1,tab2,tab3=st.tabs(["💰 Account & Risk","📁 Signals & Strategy","📌 Watchlist"])

    with tab1:
        st.subheader("Account")
        new_balance=st.number_input("Account Balance ($)",value=float(cfg.account.balance),
                                    min_value=10.0,step=100.0,key="st_bal")
        st.caption("This is the balance used for position sizing and risk calculations.")
        st.subheader("Risk Management")
        new_risk_pct=st.slider("Risk per Trade (%)",0.25,5.0,float(cfg.risk.per_trade_pct*100),
                               step=0.25,key="st_rpt",
                               help="% of your balance you're willing to lose per trade. Beginners: use 1-2%.")
        st.caption(f"💡 At {new_risk_pct:.2f}% risk on ${new_balance:,.0f} you risk **${new_balance*new_risk_pct/100:.2f}** per trade.")
        new_daily_loss=st.slider("Daily Loss Limit (%)",1.0,30.0,float(cfg.risk.daily_loss_limit_pct*100),
                                 step=1.0,key="st_dll",
                                 help="Circuit breaker trips when your day's loss hits this.")
        new_max_streak=st.number_input("Circuit Breaker: Max Consecutive Losses",
                                       value=int(cfg.risk.max_consecutive_losses),
                                       min_value=1,max_value=20,step=1,key="st_mcs")
        new_min_rr=st.slider("Minimum R:R Ratio",1.0,5.0,float(cfg.risk.min_risk_reward),
                             step=0.5,key="st_mrr",
                             help="Only take trades where potential gain is at least this many times your risk.")
        new_max_pos=st.number_input("Max Open Positions",value=int(cfg.risk.max_open_positions),
                                    min_value=1,max_value=20,step=1,key="st_mop")

    with tab2:
        st.subheader("Signal Engine")
        new_min_score=st.slider("Minimum Confidence Score (live signals)",30,90,
                                int(cfg.signals.min_confidence_score),key="st_msc",
                                help="Only fire a live signal if the confluence score is above this.")
        st.caption(f"💡 Currently set to **{new_min_score}**. Higher = fewer, higher-quality signals. Lower = more signals, lower quality.")
        tfs=["5m","15m","1h","4h","1d"]
        cur_primary=cfg.signals.timeframes.get("primary","1h")
        new_primary_tf=st.selectbox("Primary Timeframe",tfs,
                                    index=tfs.index(cur_primary) if cur_primary in tfs else 2,
                                    key="st_ptf")
        st.subheader("Indicator Periods")
        c1,c2=st.columns(2)
        with c1:
            new_ema_fast=st.number_input("EMA Fast",value=int(ind.ema_fast),min_value=5,max_value=50,key="st_ef")
            new_ema_slow=st.number_input("EMA Slow",value=int(ind.ema_slow),min_value=20,max_value=100,key="st_es")
            new_rsi_period=st.number_input("RSI Period",value=int(ind.rsi_period),min_value=5,max_value=30,key="st_rp")
        with c2:
            new_bb_period=st.number_input("BB Period",value=int(ind.bb_period),min_value=10,max_value=50,key="st_bp")
            new_atr_period=st.number_input("ATR Period",value=int(ind.atr_period),min_value=5,max_value=30,key="st_ap")

    with tab3:
        st.subheader("Forex Watchlist")
        st.caption("One symbol per line. Include the `=X` suffix for forex pairs.")
        new_forex=st.text_area("Forex",value="\n".join(cfg.watchlist.forex),height=160,key="st_fx")
        st.subheader("Crypto Watchlist")
        st.caption("One symbol per line. Include `-USD` suffix.")
        new_crypto=st.text_area("Crypto",value="\n".join(cfg.watchlist.crypto),height=120,key="st_cr")

    st.divider()
    if st.button("💾 Save All Settings",type="primary",use_container_width=True,key="st_save"):
        try:
            with open(_settings_path) as f: raw=_yaml.safe_load(f)
            raw["account"]["balance"]=float(new_balance)
            raw["risk"]["per_trade_pct"]=round(new_risk_pct/100,4)
            raw["risk"]["daily_loss_limit_pct"]=round(new_daily_loss/100,4)
            raw["risk"]["max_consecutive_losses"]=int(new_max_streak)
            raw["risk"]["min_risk_reward"]=float(new_min_rr)
            raw["risk"]["max_open_positions"]=int(new_max_pos)
            raw["signals"]["min_confidence_score"]=int(new_min_score)
            raw["signals"]["timeframes"]["primary"]=new_primary_tf
            raw["signals"]["indicators"]["ema_fast"]=int(new_ema_fast)
            raw["signals"]["indicators"]["ema_slow"]=int(new_ema_slow)
            raw["signals"]["indicators"]["rsi_period"]=int(new_rsi_period)
            raw["signals"]["indicators"]["bb_period"]=int(new_bb_period)
            raw["signals"]["indicators"]["atr_period"]=int(new_atr_period)
            raw["watchlist"]["forex"]=[x.strip() for x in new_forex.strip().split("\n") if x.strip()]
            raw["watchlist"]["crypto"]=[x.strip() for x in new_crypto.strip().split("\n") if x.strip()]
            with open(_settings_path,"w") as f:
                _yaml.dump(raw,f,default_flow_style=False,allow_unicode=True,sort_keys=False)
            st.success("✅ Settings saved to `config/settings.yaml`. **Restart the dashboard** to apply changes.")
            st.balloons()
        except Exception as e:
            st.error(f"Failed to save settings: {e}")
    st.info("💡 Tip: after saving, press **Ctrl+C** in the terminal and run `msomi dashboard` again to reload.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='color:#c9a84c;margin-bottom:0'>📡 Msomi</h2>"
                "<p style='color:#8890a8;margin-top:2px;font-size:0.85em'>Trading Intelligence</p>",
                unsafe_allow_html=True)
    st.divider()
    page=st.radio("Navigate",
                  ["Live Feed","Trade Analyzer","Heatmap","Signals","Journal","AI Tracker","Risk Calculator","Strategy Lab","Settings"],
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
elif page=="Trade Analyzer": _page_charts()
elif page=="Heatmap": _page_heatmap()
elif page=="Signals": _page_signals()
elif page=="Journal": _page_journal()
elif page=="AI Tracker": _page_predictions()
elif page=="Risk Calculator": _page_planner()
elif page=="Strategy Lab": _page_backtest()
elif page=="Settings": _page_settings()
