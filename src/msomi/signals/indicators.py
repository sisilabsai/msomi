"""Technical indicator calculations using pandas-ta."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # type: ignore[import-untyped]
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False

logger = logging.getLogger(__name__)


# ─── Result containers ────────────────────────────────────────────────────────


@dataclass
class IndicatorSnapshot:
    """Snapshot of all indicator values at the latest candle."""

    # Trend
    ema_fast: float
    ema_slow: float
    ema_trend: float
    trend_direction: str          # "UP" | "DOWN" | "NEUTRAL"
    price_vs_ema_trend: str       # "ABOVE" | "BELOW"

    # RSI
    rsi: float
    rsi_signal: str               # "OVERSOLD" | "OVERBOUGHT" | "NEUTRAL" | "BULLISH_DIV" | "BEARISH_DIV"

    # MACD
    macd: float
    macd_signal: float
    macd_hist: float
    macd_crossover: str           # "BULLISH" | "BEARISH" | "NONE"

    # Bollinger Bands
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_position: str              # "ABOVE_UPPER" | "BELOW_LOWER" | "NEAR_UPPER" | "NEAR_LOWER" | "MID"
    bb_squeeze: bool

    # ATR
    atr: float
    atr_pct: float                # ATR as % of close

    # VWAP
    vwap: Optional[float]
    price_vs_vwap: str            # "ABOVE" | "BELOW" | "AT"

    # Volume
    volume_ratio: float           # current_vol / avg_vol(20)
    volume_spike: bool

    # Close
    close: float
    timestamp: pd.Timestamp


# ─── Indicator computation ────────────────────────────────────────────────────


class IndicatorEngine:
    """Computes all technical indicators on an OHLCV DataFrame."""

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        ema_trend: int = 200,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicator columns to df. Returns enriched copy."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── EMAs ──────────────────────────────────────────────────────────────
        df["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()
        df["ema_trend"] = close.ewm(span=self.ema_trend, adjust=False).mean()

        # ── RSI ───────────────────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ── MACD ──────────────────────────────────────────────────────────────
        ema_fast_line = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow_line = close.ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast_line - ema_slow_line
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std(ddof=0)
        df["bb_upper"] = bb_mid + self.bb_std * bb_std
        df["bb_mid"] = bb_mid
        df["bb_lower"] = bb_mid - self.bb_std * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)

        # ── ATR ───────────────────────────────────────────────────────────────
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(com=self.atr_period - 1, adjust=False).mean()
        df["atr_pct"] = df["atr"] / close

        # ── VWAP (session) ────────────────────────────────────────────────────
        if "volume" in df.columns and df["volume"].sum() > 0:
            tp = (high + low + close) / 3
            df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
        else:
            df["vwap"] = np.nan

        # ── Volume ratio ──────────────────────────────────────────────────────
        if "volume" in df.columns:
            df["volume_ma20"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)
        else:
            df["volume_ratio"] = 1.0

        return df

    def snapshot(self, df: pd.DataFrame) -> Optional[IndicatorSnapshot]:
        """Return an IndicatorSnapshot for the most recent candle."""
        if len(df) < max(self.ema_trend, 50) + 10:
            logger.debug("Not enough data for indicators (%d rows)", len(df))
            return None

        df = self.compute(df)
        row = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(row["close"])

        # Trend direction
        if row["ema_fast"] > row["ema_slow"] > row["ema_trend"]:
            trend = "UP"
        elif row["ema_fast"] < row["ema_slow"] < row["ema_trend"]:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        price_vs_trend = "ABOVE" if close > row["ema_trend"] else "BELOW"

        # RSI signal
        rsi = float(row["rsi"])
        if rsi < self.rsi_oversold:
            rsi_sig = "OVERSOLD"
        elif rsi > self.rsi_overbought:
            rsi_sig = "OVERBOUGHT"
        else:
            rsi_sig = "NEUTRAL"

        # MACD crossover
        prev_cross = float(prev["macd_hist"]) if not pd.isna(prev["macd_hist"]) else 0
        curr_hist = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
        if prev_cross <= 0 < curr_hist:
            macd_cross = "BULLISH"
        elif prev_cross >= 0 > curr_hist:
            macd_cross = "BEARISH"
        else:
            macd_cross = "NONE"

        # BB position
        bb_u = float(row["bb_upper"])
        bb_l = float(row["bb_lower"])
        bb_m = float(row["bb_mid"])
        band_width = bb_u - bb_l
        if close > bb_u:
            bb_pos = "ABOVE_UPPER"
        elif close < bb_l:
            bb_pos = "BELOW_LOWER"
        elif close > bb_m + 0.25 * band_width:
            bb_pos = "NEAR_UPPER"
        elif close < bb_m - 0.25 * band_width:
            bb_pos = "NEAR_LOWER"
        else:
            bb_pos = "MID"

        # VWAP
        vwap = float(row["vwap"]) if not pd.isna(row.get("vwap", float("nan"))) else None
        if vwap:
            diff_pct = (close - vwap) / vwap
            if diff_pct > 0.001:
                pvwap = "ABOVE"
            elif diff_pct < -0.001:
                pvwap = "BELOW"
            else:
                pvwap = "AT"
        else:
            pvwap = "AT"

        vol_ratio = float(row["volume_ratio"]) if not pd.isna(row.get("volume_ratio", float("nan"))) else 1.0

        return IndicatorSnapshot(
            ema_fast=float(row["ema_fast"]),
            ema_slow=float(row["ema_slow"]),
            ema_trend=float(row["ema_trend"]),
            trend_direction=trend,
            price_vs_ema_trend=price_vs_trend,
            rsi=rsi,
            rsi_signal=rsi_sig,
            macd=float(row["macd"]) if not pd.isna(row["macd"]) else 0.0,
            macd_signal=float(row["macd_signal"]) if not pd.isna(row["macd_signal"]) else 0.0,
            macd_hist=curr_hist,
            macd_crossover=macd_cross,
            bb_upper=bb_u,
            bb_mid=bb_m,
            bb_lower=bb_l,
            bb_position=bb_pos,
            bb_squeeze=bool(row.get("bb_squeeze", False)),
            atr=float(row["atr"]) if not pd.isna(row["atr"]) else 0.0,
            atr_pct=float(row["atr_pct"]) if not pd.isna(row["atr_pct"]) else 0.0,
            vwap=vwap,
            price_vs_vwap=pvwap,
            volume_ratio=vol_ratio,
            volume_spike=vol_ratio > 2.0,
            close=close,
            timestamp=df.index[-1],
        )
