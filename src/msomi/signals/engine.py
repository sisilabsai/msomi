"""Signal detection orchestrator — ties data, indicators, and confluence together."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from msomi.core.config import MsomiConfig, get_config
from msomi.data.feeds import fetch_multiple, fetch_ohlcv
from msomi.signals.confluence import ConfluenceEngine, ScoredSignal
from msomi.signals.indicators import IndicatorEngine, IndicatorSnapshot

logger = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    """A fired signal ready for distribution."""

    symbol: str
    timeframe: str
    signal: ScoredSignal
    fired_at: datetime


class SignalEngine:
    """
    Watches a symbol list and fires SignalEvents when confluence score
    meets the configured minimum threshold.
    """

    def __init__(self, config: Optional[MsomiConfig] = None) -> None:
        self.cfg = config or get_config()
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

        self.min_score = self.cfg.signals.min_confidence_score
        self.timeframe = self.cfg.signals.timeframes.get("primary", "1h")
        self.cache_dir = self.cfg.data.cache_dir

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(
        self,
        symbols: Optional[list[str]] = None,
        timeframe: Optional[str] = None,
    ) -> list[SignalEvent]:
        """
        Scan all watched symbols, return list of SignalEvents that pass
        the minimum confidence threshold.
        """
        symbols = symbols or self.cfg.watchlist.all_symbols
        tf = timeframe or self.timeframe

        data = fetch_multiple(
            symbols,
            timeframe=tf,
            periods=self.cfg.signals.lookback_periods + 50,
            cache_dir=self.cache_dir,
        )

        events: list[SignalEvent] = []
        for sym, df in data.items():
            event = self._evaluate(sym, tf, df)
            if event:
                events.append(event)
                logger.info(
                    "Signal fired: %s [%s] %s score=%d",
                    sym,
                    tf,
                    event.signal.direction,
                    event.signal.score,
                )

        return events

    def evaluate_symbol(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[SignalEvent]:
        """Evaluate a single symbol. Optionally pass in pre-fetched df."""
        tf = timeframe or self.timeframe
        if df is None:
            df = fetch_ohlcv(
                symbol,
                timeframe=tf,
                periods=self.cfg.signals.lookback_periods + 50,
                cache_dir=self.cache_dir,
            )
        return self._evaluate(symbol, tf, df)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Optional[SignalEvent]:
        if df.empty:
            return None

        snapshot = self.indicator_engine.snapshot(df)
        if snapshot is None:
            return None

        scored = self.confluence_engine.score(snapshot)
        if scored is None:
            return None

        if scored.score < self.min_score:
            logger.debug(
                "%s: score %d below threshold %d", symbol, scored.score, self.min_score
            )
            return None

        return SignalEvent(
            symbol=symbol,
            timeframe=timeframe,
            signal=scored,
            fired_at=datetime.utcnow(),
        )

    def snapshot_only(self, symbol: str, timeframe: Optional[str] = None) -> Optional[IndicatorSnapshot]:
        """Return raw indicator snapshot without scoring (for dashboard display)."""
        tf = timeframe or self.timeframe
        df = fetch_ohlcv(symbol, timeframe=tf, periods=300, cache_dir=self.cache_dir)
        if df.empty:
            return None
        return self.indicator_engine.snapshot(df)
