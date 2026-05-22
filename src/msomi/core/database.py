"""SQLAlchemy database models and session management."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


# ─── Enums ────────────────────────────────────────────────────────────────────


class Direction(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStatus(str, enum.Enum):
    FIRED = "FIRED"
    ACTED = "ACTED"
    MISSED = "MISSED"
    EXPIRED = "EXPIRED"


class TradeOutcome(str, enum.Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


# ─── Base ─────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ─── Models ───────────────────────────────────────────────────────────────────


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    timeframe: Mapped[str] = mapped_column(String(10))
    direction: Mapped[Direction] = mapped_column(Enum(Direction))
    confidence_score: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    risk_reward: Mapped[float] = mapped_column(Float)

    # Indicator values at signal time
    rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_hist: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema_trend: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    bb_position: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    atr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # AI narration
    ai_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_risk_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[SignalStatus] = mapped_column(Enum(SignalStatus), default=SignalStatus.FIRED)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    trades: Mapped[list["Trade"]] = relationship("Trade", back_populates="signal")


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[Optional[int]] = mapped_column(ForeignKey("signals.id"), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    symbol: Mapped[str] = mapped_column(String(20), index=True)
    direction: Mapped[Direction] = mapped_column(Enum(Direction))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)

    position_size: Mapped[float] = mapped_column(Float)
    risk_amount: Mapped[float] = mapped_column(Float)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    outcome: Mapped[TradeOutcome] = mapped_column(Enum(TradeOutcome), default=TradeOutcome.OPEN)

    # Journal fields
    emotion_tag: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    confidence_at_entry: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_review: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    strategy: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    session: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    signal: Mapped[Optional["Signal"]] = relationship("Signal", back_populates="trades")


class DailyStats(Base):
    __tablename__ = "daily_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), unique=True, index=True)

    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    breakevens: Mapped[int] = mapped_column(Integer, default=0)

    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    gross_profit: Mapped[float] = mapped_column(Float, default=0.0)
    gross_loss: Mapped[float] = mapped_column(Float, default=0.0)

    starting_balance: Mapped[float] = mapped_column(Float, default=0.0)
    ending_balance: Mapped[float] = mapped_column(Float, default=0.0)

    circuit_breaker_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    consecutive_losses: Mapped[int] = mapped_column(Integer, default=0)

    ai_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    strategy_name: Mapped[str] = mapped_column(String(100))
    symbol: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(10))
    start_date: Mapped[str] = mapped_column(String(10))
    end_date: Mapped[str] = mapped_column(String(10))

    initial_capital: Mapped[float] = mapped_column(Float)
    final_capital: Mapped[float] = mapped_column(Float)
    total_return_pct: Mapped[float] = mapped_column(Float)
    annualized_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    total_trades: Mapped[int] = mapped_column(Integer)
    win_rate: Mapped[float] = mapped_column(Float)
    avg_win: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown_pct: Mapped[float] = mapped_column(Float)
    calmar_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# ─── Engine & Session ─────────────────────────────────────────────────────────


_engine = None
_SessionLocal = None


def init_db(url: str = "sqlite:///./data/msomi.db") -> None:
    """Initialize database, create tables, enable WAL mode for SQLite."""
    global _engine, _SessionLocal

    import os
    os.makedirs("./data", exist_ok=True)

    _engine = create_engine(
        url,
        connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
        echo=False,
    )

    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def get_session() -> Session:
    """Return a new database session. Caller is responsible for closing."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()
