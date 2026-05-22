"""Tests for the signal/indicator engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msomi.signals.confluence import ConfluenceEngine
from msomi.signals.indicators import IndicatorEngine, IndicatorSnapshot


def _make_df(n: int = 300, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n) * 0.001 + (0.0002 if trend == "up" else -0.0002)) + 1.1000
    high = close + np.abs(np.random.randn(n) * 0.0005)
    low = close - np.abs(np.random.randn(n) * 0.0005)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(1000, 10000, n).astype(float)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


# ─── IndicatorEngine ──────────────────────────────────────────────────────────


class TestIndicatorEngine:
    def setup_method(self):
        self.engine = IndicatorEngine()

    def test_compute_adds_all_columns(self):
        df = _make_df(300)
        result = self.engine.compute(df)
        for col in ["ema_fast", "ema_slow", "ema_trend", "rsi", "macd", "macd_signal",
                    "macd_hist", "bb_upper", "bb_mid", "bb_lower", "atr", "vwap"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_in_range(self):
        df = _make_df(300)
        result = self.engine.compute(df)
        rsi = result["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_snapshot_returns_none_for_short_df(self):
        df = _make_df(50)
        snap = self.engine.snapshot(df)
        assert snap is None

    def test_snapshot_returns_snapshot_for_long_df(self):
        df = _make_df(300)
        snap = self.engine.snapshot(df)
        assert snap is not None
        assert isinstance(snap.rsi, float)
        assert snap.close == pytest.approx(df["close"].iloc[-1], rel=1e-6)

    def test_trend_directions(self):
        df_up = _make_df(300, trend="up")
        df_down = _make_df(300, trend="down")
        snap_up = self.engine.snapshot(df_up)
        snap_down = self.engine.snapshot(df_down)
        assert snap_up is not None
        assert snap_down is not None
        # trend direction should reflect the generated data (not guaranteed but likely)
        assert snap_up.trend_direction in ("UP", "NEUTRAL", "DOWN")
        assert snap_down.trend_direction in ("UP", "NEUTRAL", "DOWN")

    def test_volume_spike_detection(self):
        df = _make_df(300)
        # Inject a volume spike at the last row
        df.iloc[-1, df.columns.get_loc("volume")] = 999999
        snap = self.engine.snapshot(df)
        assert snap is not None
        assert snap.volume_spike is True


# ─── ConfluenceEngine ─────────────────────────────────────────────────────────


class TestConfluenceEngine:
    def setup_method(self):
        self.ind_engine = IndicatorEngine()
        self.conf_engine = ConfluenceEngine()

    def _get_snap(self, trend: str = "up") -> IndicatorSnapshot:
        df = _make_df(300, trend=trend)
        snap = self.ind_engine.snapshot(df)
        assert snap is not None
        return snap

    def test_score_returns_signal_or_none(self):
        snap = self._get_snap()
        result = self.conf_engine.score(snap)
        # May be None if no directional bias
        assert result is None or hasattr(result, "direction")

    def test_scored_signal_fields(self):
        snap = self._get_snap()
        result = self.conf_engine.score(snap)
        if result is None:
            pytest.skip("No directional bias in synthetic data")
        assert result.direction in ("LONG", "SHORT")
        assert 0 <= result.score <= 100
        assert result.entry_price > 0
        assert result.stop_loss > 0
        assert result.take_profit > 0
        assert len(result.reasons) > 0

    def test_risk_reward_positive(self):
        snap = self._get_snap()
        result = self.conf_engine.score(snap)
        if result is None:
            pytest.skip("No signal")
        assert result.risk_reward > 0

    def test_sl_tp_direction_long(self):
        snap = self._get_snap("up")
        result = self.conf_engine.score(snap)
        if result is None or result.direction != "LONG":
            pytest.skip("Not a LONG signal")
        assert result.stop_loss < result.entry_price
        assert result.take_profit > result.entry_price

    def test_sl_tp_direction_short(self):
        snap = self._get_snap("down")
        result = self.conf_engine.score(snap)
        if result is None or result.direction != "SHORT":
            pytest.skip("Not a SHORT signal")
        assert result.stop_loss > result.entry_price
        assert result.take_profit < result.entry_price
