"""Tests for risk management modules."""

from __future__ import annotations

import pytest

from msomi.risk.circuit_breaker import CircuitBreaker
from msomi.risk.position_sizer import PositionSizer


# ─── PositionSizer ────────────────────────────────────────────────────────────


class TestPositionSizer:
    def test_basic_calculation(self):
        sizer = PositionSizer(account_balance=1000, risk_per_trade_pct=0.02, min_risk_reward=1.5)
        pos = sizer.calculate(entry=1.1000, stop_loss=1.0950, take_profit=1.1100)
        assert pos.is_viable
        assert pos.risk_amount == pytest.approx(20.0, rel=1e-4)
        assert pos.units > 0
        assert pos.risk_reward >= 1.5

    def test_rejects_poor_rr(self):
        sizer = PositionSizer(account_balance=1000, risk_per_trade_pct=0.02, min_risk_reward=2.0)
        pos = sizer.calculate(entry=1.1000, stop_loss=1.0980, take_profit=1.1010)
        assert not pos.is_viable
        assert "R:R" in pos.rejection_reason

    def test_rejects_zero_risk(self):
        sizer = PositionSizer(account_balance=1000, risk_per_trade_pct=0.02)
        pos = sizer.calculate(entry=1.1000, stop_loss=1.1000, take_profit=1.1100)
        assert not pos.is_viable

    def test_balance_update(self):
        sizer = PositionSizer(account_balance=1000, risk_per_trade_pct=0.02)
        sizer.update_balance(2000)
        pos = sizer.calculate(entry=1.0, stop_loss=0.99, take_profit=1.03)
        assert pos.risk_amount == pytest.approx(40.0, rel=1e-4)

    def test_invalid_risk_pct(self):
        with pytest.raises(ValueError):
            PositionSizer(account_balance=1000, risk_per_trade_pct=1.5)

    def test_short_position_sizing(self):
        sizer = PositionSizer(account_balance=1000, risk_per_trade_pct=0.01, min_risk_reward=1.5)
        pos = sizer.calculate(entry=50000, stop_loss=51000, take_profit=47000)
        assert pos.is_viable
        assert pos.risk_reward >= 1.5


# ─── CircuitBreaker ───────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_allows_trading_initially(self):
        cb = CircuitBreaker(account_balance=1000, daily_loss_limit_pct=0.10)
        allowed, reason = cb.can_trade()
        assert allowed
        assert reason == ""

    def test_trips_on_daily_loss_limit(self):
        cb = CircuitBreaker(account_balance=1000, daily_loss_limit_pct=0.10)
        # Lose exactly 10%
        cb.record_trade(-100.0)
        allowed, reason = cb.can_trade()
        assert not allowed
        assert "Daily loss" in reason

    def test_trips_on_consecutive_losses(self):
        cb = CircuitBreaker(account_balance=1000, daily_loss_limit_pct=0.50, max_consecutive_losses=3)
        cb.record_trade(-5.0)
        cb.record_trade(-5.0)
        cb.record_trade(-5.0)
        allowed, _ = cb.can_trade()
        assert not allowed

    def test_resets_consecutive_on_win(self):
        cb = CircuitBreaker(account_balance=1000, max_consecutive_losses=3)
        cb.record_trade(-5.0)
        cb.record_trade(-5.0)
        cb.record_trade(20.0)  # win resets streak
        state = cb.state
        assert state.consecutive_losses == 0

    def test_daily_pnl_tracking(self):
        cb = CircuitBreaker(account_balance=1000)
        cb.record_trade(50.0)
        cb.record_trade(-20.0)
        state = cb.state
        assert state.daily_pnl == pytest.approx(30.0, rel=1e-6)

    def test_manual_reset(self):
        cb = CircuitBreaker(account_balance=1000, daily_loss_limit_pct=0.10)
        cb.record_trade(-200.0)
        assert not cb.can_trade()[0]
        cb.reset_day()
        assert cb.can_trade()[0]
