"""Position sizing and risk calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    units: float          # number of units / contracts
    risk_amount: float    # $ amount at risk
    risk_pct: float       # % of balance at risk
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    is_viable: bool
    rejection_reason: str = ""


class PositionSizer:
    """Calculates position size based on a fixed % risk model."""

    def __init__(
        self,
        account_balance: float,
        risk_per_trade_pct: float = 0.02,
        min_risk_reward: float = 1.5,
        pip_value: float = 1.0,      # for forex: varies per pair
    ) -> None:
        if not 0 < risk_per_trade_pct <= 0.5:
            raise ValueError("risk_per_trade_pct must be between 0 and 0.5")
        self.balance = account_balance
        self.risk_pct = risk_per_trade_pct
        self.min_rr = min_risk_reward
        self.pip_value = pip_value

    def calculate(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float,
        balance_override: float | None = None,
    ) -> PositionSize:
        """
        Calculate position size.

        Args:
            entry: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            balance_override: Use this balance instead of self.balance

        Returns:
            PositionSize dataclass
        """
        balance = balance_override or self.balance
        risk_amount = balance * self.risk_pct

        risk_per_unit = abs(entry - stop_loss)
        reward_per_unit = abs(take_profit - entry)

        if risk_per_unit == 0:
            return PositionSize(
                units=0,
                risk_amount=0,
                risk_pct=0,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=0,
                is_viable=False,
                rejection_reason="Stop-loss equals entry price",
            )

        rr = reward_per_unit / risk_per_unit

        if rr < self.min_rr:
            return PositionSize(
                units=0,
                risk_amount=0,
                risk_pct=0,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=round(rr, 2),
                is_viable=False,
                rejection_reason=f"R:R {rr:.2f} below minimum {self.min_rr}",
            )

        units = risk_amount / risk_per_unit

        return PositionSize(
            units=round(units, 4),
            risk_amount=round(risk_amount, 4),
            risk_pct=self.risk_pct,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=round(rr, 2),
            is_viable=True,
        )

    def update_balance(self, new_balance: float) -> None:
        self.balance = new_balance
