"""Core configuration using Pydantic v2 + YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Sub-models ───────────────────────────────────────────────────────────────


class AccountConfig(BaseModel):
    balance: float = 1000.0
    currency: str = "USD"


class RiskConfig(BaseModel):
    per_trade_pct: float = 0.02
    daily_loss_limit_pct: float = 0.10
    weekly_drawdown_limit_pct: float = 0.20
    max_consecutive_losses: int = 3
    min_risk_reward: float = 1.5
    max_open_positions: int = 5
    correlation_limit: float = 0.7

    @field_validator("per_trade_pct", "daily_loss_limit_pct", "weekly_drawdown_limit_pct")
    @classmethod
    def pct_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        return v


class IndicatorConfig(BaseModel):
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    vwap_session: str = "D"


class WeightsConfig(BaseModel):
    trend_alignment: int = 25
    rsi_signal: int = 20
    macd_signal: int = 20
    bb_signal: int = 15
    vwap_signal: int = 10
    volume_confirm: int = 10


class SignalsConfig(BaseModel):
    min_confidence_score: int = Field(65, ge=0, le=100)
    lookback_periods: int = 200
    timeframes: dict[str, Any] = Field(
        default_factory=lambda: {"primary": "1h", "confirmation": ["15m", "4h"]}
    )
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    weights: WeightsConfig = Field(default_factory=WeightsConfig)


class SessionWindow(BaseModel):
    start: str
    end: str
    weight: float = 1.0


class SessionsConfig(BaseModel):
    london: SessionWindow = Field(default_factory=lambda: SessionWindow(start="08:00", end="17:00", weight=1.2))
    new_york: SessionWindow = Field(default_factory=lambda: SessionWindow(start="13:00", end="22:00", weight=1.2))
    asian: SessionWindow = Field(default_factory=lambda: SessionWindow(start="23:00", end="08:00", weight=0.8))
    avoid_news_window_minutes: int = 30


class AIConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model_anthropic: str = "claude-3-5-sonnet-20241022"
    model_openai: str = "gpt-4o"
    max_tokens: int = 800
    temperature: float = 0.3
    narrate_signals: bool = True
    weekly_review: bool = True
    post_trade_review: bool = True


class BacktestConfig(BaseModel):
    default_start: str = "2022-01-01"
    default_end: str = "2024-12-31"
    initial_capital: float = 10_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    monte_carlo_runs: int = 300


class AlertsConfig(BaseModel):
    telegram_enabled: bool = True
    email_enabled: bool = False
    quiet_hours: dict[str, str] = Field(default_factory=lambda: {"start": "23:00", "end": "06:00"})
    eod_report_time: str = "22:00"


class DataConfig(BaseModel):
    forex_source: Literal["yfinance", "twelvedata"] = "yfinance"
    crypto_source: Literal["yfinance", "binance"] = "yfinance"
    cache_dir: str = "./data/cache"
    db_url: str = "sqlite:///./data/msomi.db"


class WatchlistConfig(BaseModel):
    forex: list[str] = Field(
        default_factory=lambda: ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
    )
    crypto: list[str] = Field(
        default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
    )

    @property
    def all_symbols(self) -> list[str]:
        return self.forex + self.crypto


class AppConfig(BaseModel):
    name: str = "Msomi"
    version: str = "0.1.0"
    log_level: str = "INFO"
    env: str = "development"


class MsomiConfig(BaseModel):
    """Root configuration model."""

    app: AppConfig = Field(default_factory=AppConfig)
    account: AccountConfig = Field(default_factory=AccountConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    watchlist: WatchlistConfig = Field(default_factory=WatchlistConfig)


# ─── Settings (env vars) ──────────────────────────────────────────────────────


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AI Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ai_provider: Literal["anthropic", "openai"] = "anthropic"

    # Data feeds
    twelve_data_api_key: str = ""
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Account
    account_balance: float = 1000.0
    account_currency: str = "USD"

    # Risk
    risk_per_trade: float = 0.02
    daily_loss_limit: float = 0.10
    max_consecutive_losses: int = 3

    # App
    log_level: str = "INFO"
    env: str = "development"
    database_url: str = "sqlite:///./data/msomi.db"


# ─── Loaders ──────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parents[3] / "config" / "settings.yaml"


def load_yaml_config(path: Path = _CONFIG_PATH) -> MsomiConfig:
    """Load and validate YAML config."""
    if not path.exists():
        return MsomiConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return MsomiConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> MsomiConfig:
    return load_yaml_config()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
