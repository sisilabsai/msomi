"""Backtesting engine — strategy simulation on historical OHLCV data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from msomi.core.config import MsomiConfig, get_config
from msomi.data.feeds import fetch_ohlcv
from msomi.signals.confluence import ConfluenceEngine
from msomi.signals.indicators import IndicatorEngine

logger = logging.getLogger(__name__)


# ─── Results ──────────────────────────────────────────────────────────────────


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pct: float
    score: int
    outcome: str  # WIN | LOSS | BREAKEVEN


@dataclass
class BacktestReport:
    strategy: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0

    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    final_capital: float = 0.0

    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  Backtest: {self.strategy} | {self.symbol} [{self.timeframe}]",
            f"  Period: {self.start_date} → {self.end_date}",
            f"{'='*60}",
            f"  Total Trades : {self.total_trades}",
            f"  Win Rate     : {self.win_rate:.1f}%  ({self.wins}W / {self.losses}L)",
            f"  Profit Factor: {self.profit_factor:.2f}",
            f"  Sharpe Ratio : {self.sharpe_ratio:.2f}",
            f"  Max Drawdown : {self.max_drawdown_pct:.1f}%",
            f"  Total Return : {self.total_return_pct:+.1f}%",
            f"  Final Capital: ${self.final_capital:,.2f}  (started ${self.initial_capital:,.2f})",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ─── Engine ───────────────────────────────────────────────────────────────────


class BacktestEngine:
    """
    Event-driven backtester for Msomi's confluence strategy.

    Iterates bar-by-bar, evaluates signals at each candle, and
    simulates trade entry/exit with commission and slippage.
    """

    def __init__(self, config: Optional[MsomiConfig] = None) -> None:
        self.cfg = config or get_config()
        bc = self.cfg.backtest

        self.initial_capital = bc.initial_capital
        self.commission_pct = bc.commission_pct
        self.slippage_pct = bc.slippage_pct

        ind = self.cfg.signals.indicators
        wt = self.cfg.signals.weights
        self.indicator_engine = IndicatorEngine(
            ema_fast=ind.ema_fast,
            ema_slow=ind.ema_slow,
            ema_trend=ind.ema_trend,
            rsi_period=ind.rsi_period,
            rsi_overbought=ind.rsi_overbought,
            rsi_oversold=ind.rsi_oversold,
            macd_fast=ind.macd_fast,
            macd_slow=ind.macd_slow,
            macd_signal=ind.macd_signal,
            bb_period=ind.bb_period,
            bb_std=ind.bb_std,
            atr_period=ind.atr_period,
        )
        self.confluence_engine = ConfluenceEngine(
            trend_weight=wt.trend_alignment,
            rsi_weight=wt.rsi_signal,
            macd_weight=wt.macd_signal,
            bb_weight=wt.bb_signal,
            vwap_weight=wt.vwap_signal,
            volume_weight=wt.volume_confirm,
        )

    def run(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        min_score: Optional[int] = None,
        strategy_name: str = "Confluence-v1",
    ) -> BacktestReport:
        """
        Run a backtest for a single symbol over the given date range.

        Returns a BacktestReport with full trade list and metrics.
        """
        start = start or self.cfg.backtest.default_start
        end = end or self.cfg.backtest.default_end
        min_score = min_score if min_score is not None else self.cfg.signals.min_confidence_score

        logger.info("Running backtest: %s [%s] %s → %s", symbol, timeframe, start, end)

        df = fetch_ohlcv(symbol, timeframe=timeframe, start=start, end=end, periods=9999)
        if df.empty or len(df) < 250:
            logger.warning("Insufficient data for backtest of %s", symbol)
            return BacktestReport(
                strategy=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end,
                initial_capital=self.initial_capital,
            )

        df_ind = self.indicator_engine.compute(df)
        lookback = max(self.cfg.signals.indicators.ema_trend, 50) + 10
        trades, equity = self._simulate(df_ind, symbol, min_score, lookback)

        report = BacktestReport(
            strategy=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            initial_capital=self.initial_capital,
            trades=trades,
            equity_curve=equity,
        )
        self._compute_metrics(report)
        logger.info(report.summary())
        return report

    def run_multi(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, BacktestReport]:
        """Run backtests for multiple symbols."""
        return {sym: self.run(sym, timeframe, start, end) for sym in symbols}

    # ── Simulation ─────────────────────────────────────────────────────────────

    def _simulate(
        self,
        df: pd.DataFrame,
        symbol: str,
        min_score: int,
        lookback: int,
    ) -> tuple[list[BacktestTrade], list[float]]:
        capital = self.initial_capital
        equity: list[float] = [capital]
        trades: list[BacktestTrade] = []
        in_trade = False
        trade_dir = ""
        entry_price = 0.0
        sl = 0.0
        tp = 0.0
        entry_time = df.index[0]
        entry_score = 0

        for i in range(lookback, len(df)):
            window = df.iloc[: i + 1]
            close = float(df["close"].iloc[i])
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])
            ts = df.index[i]

            # ── Manage open trade ─────────────────────────────────────────────
            if in_trade:
                exit_price = None
                outcome = None

                if trade_dir == "LONG":
                    if low <= sl:
                        exit_price = sl
                        outcome = "LOSS"
                    elif high >= tp:
                        exit_price = tp
                        outcome = "WIN"
                elif trade_dir == "SHORT":
                    if high >= sl:
                        exit_price = sl
                        outcome = "LOSS"
                    elif low <= tp:
                        exit_price = tp
                        outcome = "WIN"

                if exit_price and outcome:
                    slippage = exit_price * self.slippage_pct
                    effective_exit = exit_price - slippage if trade_dir == "LONG" else exit_price + slippage
                    commission = capital * self.commission_pct * 2  # entry + exit

                    pnl_pct = (effective_exit - entry_price) / entry_price
                    if trade_dir == "SHORT":
                        pnl_pct = -pnl_pct
                    pnl = capital * pnl_pct - commission
                    capital += pnl

                    trades.append(BacktestTrade(
                        entry_time=entry_time,
                        exit_time=ts,
                        symbol=symbol,
                        direction=trade_dir,
                        entry_price=entry_price,
                        exit_price=effective_exit,
                        stop_loss=sl,
                        take_profit=tp,
                        pnl=round(pnl, 4),
                        pnl_pct=round(pnl_pct * 100, 4),
                        score=entry_score,
                        outcome=outcome,
                    ))
                    equity.append(capital)
                    in_trade = False
                continue

            # ── Look for new signal ───────────────────────────────────────────
            snap = self.indicator_engine.snapshot(window)
            if snap is None:
                continue

            scored = self.confluence_engine.score(snap)
            if scored is None or scored.score < min_score:
                continue

            # Simulate entry with slippage
            slip = close * self.slippage_pct
            if scored.direction == "LONG":
                entry_price = close + slip
                sl = scored.stop_loss
                tp = scored.take_profit
            else:
                entry_price = close - slip
                sl = scored.stop_loss
                tp = scored.take_profit

            in_trade = True
            trade_dir = scored.direction
            entry_time = ts
            entry_score = scored.score

        return trades, equity

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, report: BacktestReport) -> None:
        trades = report.trades
        if not trades:
            return

        report.total_trades = len(trades)
        report.wins = sum(1 for t in trades if t.outcome == "WIN")
        report.losses = sum(1 for t in trades if t.outcome == "LOSS")
        report.breakevens = sum(1 for t in trades if t.outcome == "BREAKEVEN")

        report.win_rate = report.wins / report.total_trades * 100 if report.total_trades else 0

        win_pnls = [t.pnl for t in trades if t.outcome == "WIN"]
        loss_pnls = [t.pnl for t in trades if t.outcome == "LOSS"]

        report.avg_win_pct = float(np.mean([t.pnl_pct for t in trades if t.outcome == "WIN"])) if win_pnls else 0
        report.avg_loss_pct = float(np.mean([t.pnl_pct for t in trades if t.outcome == "LOSS"])) if loss_pnls else 0

        gross_profit = sum(p for p in win_pnls if p > 0)
        gross_loss = abs(sum(p for p in loss_pnls if p < 0))
        report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        if len(report.equity_curve) > 1:
            eq = np.array(report.equity_curve)
            returns = np.diff(eq) / eq[:-1]
            report.sharpe_ratio = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252))
            report.final_capital = float(eq[-1])
            report.total_return_pct = (eq[-1] - report.initial_capital) / report.initial_capital * 100

            # Max drawdown
            peak = np.maximum.accumulate(eq)
            drawdowns = (peak - eq) / peak * 100
            report.max_drawdown_pct = float(drawdowns.max())

            ann_return = report.total_return_pct / max(len(eq) / (252 * 24), 1)
            report.calmar_ratio = ann_return / report.max_drawdown_pct if report.max_drawdown_pct > 0 else 0
        else:
            report.final_capital = report.initial_capital
