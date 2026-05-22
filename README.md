# Msomi 📡

> *msomi — the educated one (Swahili)*

**A personal AI-powered trading intelligence system for Forex and Crypto.**

Msomi gives one trader a systematic, data-driven edge by combining technical analysis, AI market commentary, automated risk management, and backtested strategies — delivered to your Telegram before every trade.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Intelligence Layer                  │
│  Claude / GPT-4 · Signal narration · Regime detect  │
├─────────────────────────────────────────────────────┤
│                    Signal Layer                      │
│  EMA · RSI · MACD · BB · ATR · VWAP · Confluence   │
├─────────────────────────────────────────────────────┤
│                     Risk Layer                       │
│  Position sizing · Circuit breaker · Streak detect  │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
msomi/
├── config/
│   └── settings.yaml          # All configuration
├── src/msomi/
│   ├── core/
│   │   ├── config.py          # Pydantic v2 config models
│   │   └── database.py        # SQLAlchemy models + session
│   ├── data/
│   │   └── feeds.py           # yfinance / Binance data pipeline
│   ├── signals/
│   │   ├── indicators.py      # EMA, RSI, MACD, BB, ATR, VWAP
│   │   ├── confluence.py      # Scored signal engine (0–100)
│   │   └── engine.py          # Signal scan orchestrator
│   ├── risk/
│   │   ├── position_sizer.py  # Dynamic % risk position sizing
│   │   ├── circuit_breaker.py # Daily loss limits + streak detect
│   │   └── manager.py         # Risk orchestrator
│   ├── ai/
│   │   └── analyst.py         # Claude / OpenAI signal narration
│   ├── backtest/
│   │   └── engine.py          # Bar-by-bar backtester
│   ├── alerts/
│   │   ├── telegram.py        # Rich Telegram signal cards
│   │   └── scheduler.py       # APScheduler job management
│   ├── journal/
│   │   └── logger.py          # Trade logging + analytics
│   ├── dashboard/
│   │   └── app.py             # Streamlit control center
│   └── cli.py                 # Typer CLI
└── tests/
    ├── test_signals.py
    ├── test_risk.py
    └── test_config.py
```

## Quick Start

### 1. Install

```bash
# Clone and set up
git clone <your-repo>
cd msomi

# Install in editable mode
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

Key `.env` variables:
- `ANTHROPIC_API_KEY` — Claude AI narration
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — signal alerts
- `ACCOUNT_BALANCE` — your trading account size

Adjust strategy settings in `config/settings.yaml`.

### 3. Run

```bash
# One-time scan of all watched symbols
msomi scan

# Scan a specific symbol
msomi scan --symbol EURUSD=X --timeframe 1h

# Continuous watch (scans every 60 min)
msomi watch --interval 60

# Backtest a symbol
msomi backtest EURUSD=X --timeframe 1h --start 2023-01-01

# Launch Streamlit dashboard
msomi dashboard

# Performance summary
msomi journal --days 30
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Indicators | pandas-ta, custom numpy |
| Backtesting | Custom event-driven engine |
| AI | Anthropic Claude / OpenAI GPT-4 |
| Data (Forex) | yfinance / Twelve Data |
| Data (Crypto) | yfinance / Binance API |
| Database | SQLite → PostgreSQL (SQLAlchemy) |
| Dashboard | Streamlit + Plotly |
| Alerts | python-telegram-bot |
| Scheduler | APScheduler |
| Config | YAML + Pydantic v2 |
| CLI | Typer + Rich |
| Testing | pytest |

## Risk Disclaimer

⚠️ Trading Forex and Crypto carries significant financial risk. Msomi is a personal research and analysis tool — not financial advice. Always start with paper trading or the absolute minimum deposit before risking real capital. Never trade money you cannot afford to lose.

---

*Built for the educated trader.*
# msomi
