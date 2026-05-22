"""Daily loss circuit breaker and streak detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class CircuitState:
    is_tripped: bool = False
    reason: str = ""
    date: str = ""
    daily_pnl: float = 0.0
    daily_loss_pct: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0


class CircuitBreaker:
    """
    Enforces hard daily loss limits and consecutive loss streaks.

    Once tripped for the day, no new trades should be taken.
    """

    def __init__(
        self,
        account_balance: float,
        daily_loss_limit_pct: float = 0.10,
        max_consecutive_losses: int = 3,
    ) -> None:
        self.balance = account_balance
        self.daily_limit_pct = daily_loss_limit_pct
        self.max_streak = max_consecutive_losses

        self._today = str(date.today())
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._tripped: bool = False
        self._trip_reason: str = ""
        self._trades_today: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_trade(self, pnl: float) -> CircuitState:
        """
        Record a completed trade outcome. Returns current circuit state.

        Args:
            pnl: Profit/loss in account currency (negative = loss)
        """
        self._maybe_reset_day()
        self._daily_pnl += pnl
        self._trades_today += 1

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._check_conditions()
        return self.state

    def can_trade(self) -> tuple[bool, str]:
        """Returns (allowed, reason). Call before placing any trade."""
        self._maybe_reset_day()
        if self._tripped:
            return False, self._trip_reason
        return True, ""

    def reset_day(self) -> None:
        """Manually reset for a new session (use with care)."""
        self._today = str(date.today())
        self._daily_pnl = 0.0
        self._trades_today = 0
        self._tripped = False
        self._trip_reason = ""
        # Note: consecutive_losses persists across days intentionally

    def update_balance(self, new_balance: float) -> None:
        self.balance = new_balance

    @property
    def state(self) -> CircuitState:
        daily_loss_pct = abs(self._daily_pnl) / self.balance if self._daily_pnl < 0 else 0
        return CircuitState(
            is_tripped=self._tripped,
            reason=self._trip_reason,
            date=self._today,
            daily_pnl=round(self._daily_pnl, 4),
            daily_loss_pct=round(daily_loss_pct, 4),
            consecutive_losses=self._consecutive_losses,
            trades_today=self._trades_today,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _check_conditions(self) -> None:
        if self._tripped:
            return

        # Daily loss limit
        if self._daily_pnl < 0:
            loss_pct = abs(self._daily_pnl) / self.balance
            if loss_pct >= self.daily_limit_pct:
                self._trip(
                    f"Daily loss limit hit: {loss_pct*100:.1f}% (limit: {self.daily_limit_pct*100:.0f}%)"
                )
                return

        # Consecutive losses
        if self._consecutive_losses >= self.max_streak:
            self._trip(
                f"{self._consecutive_losses} consecutive losses — pausing for the session"
            )

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self._trip_reason = reason
        logger.warning("⛔ Circuit breaker tripped: %s", reason)

    def _maybe_reset_day(self) -> None:
        today = str(date.today())
        if today != self._today:
            logger.info("New trading day — resetting daily P&L and circuit breaker")
            self._today = today
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._tripped = False
            self._trip_reason = ""
