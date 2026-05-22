"""Trade journal — SQLite persistence and performance analytics."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from msomi.core.config import get_config, get_settings
from msomi.core.database import (
    Direction,
    DailyStats,
    Signal,
    SignalStatus,
    Trade,
    TradeOutcome,
    get_session,
    init_db,
)
from msomi.signals.engine import SignalEvent

logger = logging.getLogger(__name__)


class TradeJournal:
    """
    Persistent trade and signal logging with analytics queries.

    All writes are auto-committed; reads return plain dicts or DataFrames
    so callers don't need SQLAlchemy awareness.
    """

    def __init__(self) -> None:
        cfg = get_config()
        init_db(cfg.data.db_url)

    # ── Write API ─────────────────────────────────────────────────────────────

    def log_signal(
        self,
        event: SignalEvent,
        ai_analysis: str = "",
        ai_risk_note: str = "",
    ) -> int:
        """Persist a fired signal; returns the new signal DB id."""
        sig = event.signal
        snap = sig.snapshot

        record = Signal(
            symbol=event.symbol,
            timeframe=event.timeframe,
            direction=Direction(sig.direction),
            confidence_score=sig.score,
            entry_price=sig.entry_price,
            stop_loss=sig.stop_loss,
            take_profit=sig.take_profit,
            risk_reward=sig.risk_reward,
            rsi=snap.rsi if snap else None,
            macd_hist=snap.macd_hist if snap else None,
            ema_trend=snap.trend_direction if snap else None,
            bb_position=snap.bb_position if snap else None,
            atr=snap.atr if snap else None,
            ai_analysis=ai_analysis or None,
            ai_risk_note=ai_risk_note or None,
            status=SignalStatus.FIRED,
        )

        with get_session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def log_trade_open(
        self,
        signal_id: Optional[int],
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        risk_amount: float,
        strategy: str = "confluence",
        session_name: str = "",
        emotion_tag: str = "",
        confidence_at_entry: int = 0,
        notes: str = "",
    ) -> int:
        """Record a trade open; returns the trade DB id."""
        record = Trade(
            signal_id=signal_id,
            symbol=symbol,
            direction=Direction(direction),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            outcome=TradeOutcome.OPEN,
            strategy=strategy or None,
            session=session_name or None,
            emotion_tag=emotion_tag or None,
            confidence_at_entry=confidence_at_entry or None,
            notes=notes or None,
        )

        with get_session() as session:
            session.add(record)
            # Update signal status to ACTED
            if signal_id:
                sig = session.get(Signal, signal_id)
                if sig:
                    sig.status = SignalStatus.ACTED
            session.commit()
            session.refresh(record)
            return record.id

    def log_trade_close(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        outcome: str = "WIN",
        ai_review: str = "",
        notes: str = "",
    ) -> None:
        """Update a trade with close data and compute daily stats."""
        with get_session() as session:
            trade = session.get(Trade, trade_id)
            if not trade:
                logger.warning("Trade %d not found", trade_id)
                return

            trade.closed_at = datetime.utcnow()
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.pnl_pct = pnl_pct
            trade.outcome = TradeOutcome(outcome)
            if ai_review:
                trade.ai_review = ai_review
            if notes:
                trade.notes = (trade.notes or "") + "\n" + notes

            session.commit()

        # Update daily stats
        self._update_daily_stats(pnl, outcome)

    # ── Read API ──────────────────────────────────────────────────────────────

    def recent_signals(self, limit: int = 20) -> list[dict]:
        """Return the most recent signals as plain dicts."""
        with get_session() as session:
            rows = (
                session.query(Signal)
                .order_by(Signal.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._signal_to_dict(r) for r in rows]

    def recent_trades(self, limit: int = 20) -> list[dict]:
        """Return the most recent trades as plain dicts."""
        with get_session() as session:
            rows = (
                session.query(Trade)
                .order_by(Trade.opened_at.desc())
                .limit(limit)
                .all()
            )
            return [self._trade_to_dict(r) for r in rows]

    def performance_summary(self, days: int = 30) -> dict:
        """
        Aggregated performance stats over the last `days` days.

        Returns keys: total_trades, wins, losses, breakevens, win_rate,
        profit_factor, avg_win, avg_loss, best_trade, worst_trade,
        total_pnl, avg_rr, period.
        """
        since = datetime.utcnow() - timedelta(days=days)

        with get_session() as session:
            trades = (
                session.query(Trade)
                .filter(
                    Trade.opened_at >= since,
                    Trade.outcome != TradeOutcome.OPEN,
                )
                .all()
            )

        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "breakevens": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "total_pnl": 0.0,
                "avg_rr": 0.0,
                "period": f"last {days} days",
            }

        pnls = [t.pnl for t in trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakevens = [p for p in pnls if p == 0]

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        win_count = len([t for t in trades if t.outcome == TradeOutcome.WIN])
        loss_count = len([t for t in trades if t.outcome == TradeOutcome.LOSS])
        be_count = len([t for t in trades if t.outcome == TradeOutcome.BREAKEVEN])
        total = len(trades)

        rr_values = [
            abs((t.take_profit - t.entry_price) / (t.entry_price - t.stop_loss))
            for t in trades
            if t.stop_loss and t.entry_price != t.stop_loss
        ]

        return {
            "total_trades": total,
            "wins": win_count,
            "losses": loss_count,
            "breakevens": be_count,
            "win_rate": win_count / total if total > 0 else 0.0,
            "profit_factor": profit_factor,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "best_trade": max(pnls) if pnls else 0.0,
            "worst_trade": min(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
            "avg_rr": sum(rr_values) / len(rr_values) if rr_values else 0.0,
            "period": f"last {days} days",
        }

    def equity_curve(self, initial_balance: float = 1000.0) -> pd.DataFrame:
        """
        Build a cumulative equity DataFrame from closed trades.

        Returns a DataFrame with columns: [opened_at, pnl, balance].
        """
        with get_session() as session:
            trades = (
                session.query(Trade)
                .filter(Trade.outcome != TradeOutcome.OPEN)
                .order_by(Trade.opened_at.asc())
                .all()
            )

        if not trades:
            return pd.DataFrame(columns=["opened_at", "pnl", "balance"])

        rows = []
        balance = initial_balance
        for t in trades:
            pnl = t.pnl or 0.0
            balance += pnl
            rows.append({"opened_at": t.opened_at, "pnl": pnl, "balance": balance})

        return pd.DataFrame(rows)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_daily_stats(self, pnl: float, outcome: str) -> None:
        """Upsert today's DailyStats row."""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        with get_session() as session:
            stats = session.query(DailyStats).filter_by(date=today).first()
            if not stats:
                stats = DailyStats(date=today)
                session.add(stats)

            stats.total_trades += 1
            stats.total_pnl = (stats.total_pnl or 0.0) + pnl

            if outcome == TradeOutcome.WIN or outcome == "WIN":
                stats.wins += 1
                stats.gross_profit = (stats.gross_profit or 0.0) + pnl
                stats.consecutive_losses = 0
            elif outcome == TradeOutcome.LOSS or outcome == "LOSS":
                stats.losses += 1
                stats.gross_loss = (stats.gross_loss or 0.0) + abs(pnl)
                stats.consecutive_losses = (stats.consecutive_losses or 0) + 1
            else:
                stats.breakevens += 1

            session.commit()

    @staticmethod
    def _signal_to_dict(s: Signal) -> dict:
        return {
            "id": s.id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "symbol": s.symbol,
            "timeframe": s.timeframe,
            "direction": s.direction.value if s.direction else None,
            "confidence_score": s.confidence_score,
            "entry_price": s.entry_price,
            "stop_loss": s.stop_loss,
            "take_profit": s.take_profit,
            "risk_reward": s.risk_reward,
            "rsi": s.rsi,
            "macd_hist": s.macd_hist,
            "ema_trend": s.ema_trend,
            "status": s.status.value if s.status else None,
            "ai_analysis": s.ai_analysis,
        }

    @staticmethod
    def _trade_to_dict(t: Trade) -> dict:
        return {
            "id": t.id,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            "symbol": t.symbol,
            "direction": t.direction.value if t.direction else None,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "position_size": t.position_size,
            "risk_amount": t.risk_amount,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "outcome": t.outcome.value if t.outcome else None,
            "emotion_tag": t.emotion_tag,
            "confidence_at_entry": t.confidence_at_entry,
            "notes": t.notes,
            "strategy": t.strategy,
        }
