"""Confluence scoring engine — combines indicator signals into a 0–100 score."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from msomi.signals.indicators import IndicatorSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ScoredSignal:
    """Output of the confluence engine."""

    direction: str                  # "LONG" | "SHORT"
    score: int                      # 0–100
    components: dict[str, int]      # per-indicator contribution
    reasons: list[str]              # human-readable reasons list
    snapshot: IndicatorSnapshot

    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.score >= 0  # actual threshold applied externally

    @property
    def short_summary(self) -> str:
        return f"{self.direction} | score={self.score} | entry={self.entry_price:.5f} SL={self.stop_loss:.5f} TP={self.take_profit:.5f}"


class ConfluenceEngine:
    """
    Scores a trading setup 0–100 by combining indicator signals.

    Weights (must sum to 100):
      trend_alignment:  25
      rsi_signal:       20
      macd_signal:      20
      bb_signal:        15
      vwap_signal:      10
      volume_confirm:   10
    """

    def __init__(
        self,
        trend_weight: int = 25,
        rsi_weight: int = 20,
        macd_weight: int = 20,
        bb_weight: int = 15,
        vwap_weight: int = 10,
        volume_weight: int = 10,
        min_atr_pct: float = 0.002,   # minimum 0.2% ATR for trade viability
        rr_multiplier: float = 2.0,    # TP = entry ± rr_multiplier * ATR
        sl_atr_mult: float = 1.5,
    ) -> None:
        self.weights = {
            "trend": trend_weight,
            "rsi": rsi_weight,
            "macd": macd_weight,
            "bb": bb_weight,
            "vwap": vwap_weight,
            "volume": volume_weight,
        }
        self.min_atr_pct = min_atr_pct
        self.rr_multiplier = rr_multiplier
        self.sl_atr_mult = sl_atr_mult

    def score(self, snap: IndicatorSnapshot) -> Optional[ScoredSignal]:
        """
        Evaluate indicator snapshot and return a ScoredSignal or None if
        there is no clear directional bias.
        """
        long_score, short_score = 0, 0
        long_reasons: list[str] = []
        short_reasons: list[str] = []
        long_components: dict[str, int] = {}
        short_components: dict[str, int] = {}

        # ── 1. Trend alignment (EMA stack) ────────────────────────────────────
        w = self.weights["trend"]
        if snap.trend_direction == "UP" and snap.price_vs_ema_trend == "ABOVE":
            pts = w
            long_score += pts
            long_components["trend"] = pts
            long_reasons.append(f"EMA stack bullish (fast > slow > {int(snap.ema_trend)}-EMA)")
        elif snap.trend_direction == "DOWN" and snap.price_vs_ema_trend == "BELOW":
            pts = w
            short_score += pts
            short_components["trend"] = pts
            short_reasons.append(f"EMA stack bearish (fast < slow < {int(snap.ema_trend)}-EMA)")
        elif snap.trend_direction == "UP":
            pts = int(w * 0.5)
            long_score += pts
            long_components["trend"] = pts
            long_reasons.append("Partial bullish EMA alignment")
        elif snap.trend_direction == "DOWN":
            pts = int(w * 0.5)
            short_score += pts
            short_components["trend"] = pts
            short_reasons.append("Partial bearish EMA alignment")

        # ── 2. RSI ────────────────────────────────────────────────────────────
        w = self.weights["rsi"]
        if snap.rsi_signal == "OVERSOLD":
            pts = w
            long_score += pts
            long_components["rsi"] = pts
            long_reasons.append(f"RSI oversold ({snap.rsi:.1f})")
        elif snap.rsi_signal == "OVERBOUGHT":
            pts = w
            short_score += pts
            short_components["rsi"] = pts
            short_reasons.append(f"RSI overbought ({snap.rsi:.1f})")
        elif 45 <= snap.rsi <= 55:
            # Neutral territory — slight lean based on trend
            pts = int(w * 0.3)
            if snap.trend_direction == "UP":
                long_score += pts
                long_components["rsi"] = pts
            elif snap.trend_direction == "DOWN":
                short_score += pts
                short_components["rsi"] = pts

        # ── 3. MACD crossover / histogram ─────────────────────────────────────
        w = self.weights["macd"]
        if snap.macd_crossover == "BULLISH":
            pts = w
            long_score += pts
            long_components["macd"] = pts
            long_reasons.append("MACD bullish crossover")
        elif snap.macd_crossover == "BEARISH":
            pts = w
            short_score += pts
            short_components["macd"] = pts
            short_reasons.append("MACD bearish crossover")
        elif snap.macd_hist > 0:
            pts = int(w * 0.5)
            long_score += pts
            long_components["macd"] = pts
            long_reasons.append("MACD histogram positive")
        elif snap.macd_hist < 0:
            pts = int(w * 0.5)
            short_score += pts
            short_components["macd"] = pts
            short_reasons.append("MACD histogram negative")

        # ── 4. Bollinger Bands ────────────────────────────────────────────────
        w = self.weights["bb"]
        if snap.bb_position == "BELOW_LOWER":
            pts = w
            long_score += pts
            long_components["bb"] = pts
            long_reasons.append("Price below lower Bollinger Band (mean reversion)")
        elif snap.bb_position == "ABOVE_UPPER":
            pts = w
            short_score += pts
            short_components["bb"] = pts
            short_reasons.append("Price above upper Bollinger Band (mean reversion)")
        elif snap.bb_squeeze:
            pts = int(w * 0.6)
            long_score += pts
            short_score += pts
            long_components["bb"] = pts
            short_components["bb"] = pts
            long_reasons.append("BB squeeze — breakout expected")
            short_reasons.append("BB squeeze — breakout expected")
        elif snap.bb_position == "NEAR_LOWER":
            pts = int(w * 0.4)
            long_score += pts
            long_components["bb"] = pts
            long_reasons.append("Price near lower BB")
        elif snap.bb_position == "NEAR_UPPER":
            pts = int(w * 0.4)
            short_score += pts
            short_components["bb"] = pts
            short_reasons.append("Price near upper BB")

        # ── 5. VWAP ───────────────────────────────────────────────────────────
        w = self.weights["vwap"]
        if snap.vwap and snap.price_vs_vwap == "ABOVE":
            pts = w
            long_score += pts
            long_components["vwap"] = pts
            long_reasons.append("Price above VWAP (bullish intraday bias)")
        elif snap.vwap and snap.price_vs_vwap == "BELOW":
            pts = w
            short_score += pts
            short_components["vwap"] = pts
            short_reasons.append("Price below VWAP (bearish intraday bias)")

        # ── 6. Volume confirmation ─────────────────────────────────────────────
        w = self.weights["volume"]
        if snap.volume_spike:
            long_score += w
            short_score += w
            long_components["volume"] = w
            short_components["volume"] = w
            long_reasons.append(f"Volume spike ({snap.volume_ratio:.1f}× avg)")
            short_reasons.append(f"Volume spike ({snap.volume_ratio:.1f}× avg)")
        elif snap.volume_ratio > 1.2:
            pts = int(w * 0.5)
            long_score += pts
            short_score += pts
            long_components["volume"] = pts
            short_components["volume"] = pts

        # ── Determine direction ───────────────────────────────────────────────
        if long_score == short_score:
            return None  # No clear bias

        if long_score > short_score:
            direction = "LONG"
            final_score = min(long_score, 100)
            components = long_components
            reasons = long_reasons
        else:
            direction = "SHORT"
            final_score = min(short_score, 100)
            components = short_components
            reasons = short_reasons

        # ── ATR viability ─────────────────────────────────────────────────────
        if snap.atr_pct < self.min_atr_pct:
            reasons.append(f"⚠ Low volatility (ATR={snap.atr_pct*100:.3f}%)")
            final_score = max(0, final_score - 15)

        # ── Compute entry / SL / TP ───────────────────────────────────────────
        entry = snap.close
        atr = snap.atr or snap.close * 0.005

        if direction == "LONG":
            sl = entry - self.sl_atr_mult * atr
            tp = entry + self.rr_multiplier * self.sl_atr_mult * atr
        else:
            sl = entry + self.sl_atr_mult * atr
            tp = entry - self.rr_multiplier * self.sl_atr_mult * atr

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0.0

        return ScoredSignal(
            direction=direction,
            score=final_score,
            components=components,
            reasons=reasons,
            snapshot=snap,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            risk_reward=rr,
        )
