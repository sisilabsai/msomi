"""Risk manager — orchestrates position sizing and circuit breaker."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from msomi.core.config import MsomiConfig, get_config
from msomi.risk.circuit_breaker import CircuitBreaker, CircuitState
from msomi.risk.position_sizer import PositionSize, PositionSizer
from msomi.signals.confluence import ScoredSignal

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Full risk assessment for a proposed trade."""

    allowed: bool
    rejection_reason: str
    position: Optional[PositionSize]
    circuit_state: CircuitState
    session_quality: float   # 0–1 multiplier


class RiskManager:
    """
    Central risk orchestrator.

    Usage:
        rm = RiskManager(balance=1000)
        assessment = rm.assess(signal)
        if assessment.allowed:
            # place trade
            rm.record_outcome(pnl)
    """

    def __init__(
        self,
        balance: float,
        config: Optional[MsomiConfig] = None,
    ) -> None:
        self.cfg = config or get_config()
        rc = self.cfg.risk

        self.sizer = PositionSizer(
            account_balance=balance,
            risk_per_trade_pct=rc.per_trade_pct,
            min_risk_reward=rc.min_risk_reward,
        )
        self.breaker = CircuitBreaker(
            account_balance=balance,
            daily_loss_limit_pct=rc.daily_loss_limit_pct,
            max_consecutive_losses=rc.max_consecutive_losses,
        )
        self._balance = balance

    # ── Public API ─────────────────────────────────────────────────────────────

    def assess(self, signal: ScoredSignal) -> RiskAssessment:
        """
        Full risk assessment for a given signal.
        Returns RiskAssessment with allowed=True/False and position details.
        """
        circuit_state = self.breaker.state

        # 1. Circuit breaker check
        allowed, reason = self.breaker.can_trade()
        if not allowed:
            return RiskAssessment(
                allowed=False,
                rejection_reason=reason,
                position=None,
                circuit_state=circuit_state,
                session_quality=1.0,
            )

        # 2. Position sizing & R:R check
        position = self.sizer.calculate(
            entry=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        if not position.is_viable:
            return RiskAssessment(
                allowed=False,
                rejection_reason=position.rejection_reason,
                position=position,
                circuit_state=circuit_state,
                session_quality=1.0,
            )

        return RiskAssessment(
            allowed=True,
            rejection_reason="",
            position=position,
            circuit_state=circuit_state,
            session_quality=1.0,
        )

    def record_outcome(self, pnl: float) -> CircuitState:
        """Record completed trade P&L and update circuit breaker."""
        new_balance = self._balance + pnl
        self.update_balance(new_balance)
        state = self.breaker.record_trade(pnl)
        logger.info(
            "Trade outcome: P&L=%.4f | Balance=%.2f | Daily P&L=%.4f | Streak=%d",
            pnl,
            self._balance,
            state.daily_pnl,
            state.consecutive_losses,
        )
        return state

    def update_balance(self, balance: float) -> None:
        self._balance = balance
        self.sizer.update_balance(balance)
        self.breaker.update_balance(balance)

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def circuit_state(self) -> CircuitState:
        return self.breaker.state
