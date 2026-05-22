"""
Msomi CLI — main entry point.

Usage:
  msomi scan              # scan all symbols once
  msomi watch             # continuous scanning loop
  msomi backtest          # run backtests
  msomi dashboard         # launch Streamlit dashboard
  msomi journal           # print performance summary
"""

from __future__ import annotations

import logging
import sys
import time

import typer
from rich.console import Console
from rich.table import Table

from msomi.core.config import get_config, get_settings
from msomi.core.database import init_db

app = typer.Typer(
    name="msomi",
    help="Personal AI-powered trading intelligence system",
    add_completion=False,
)
console = Console()


def _setup(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_config()
    init_db(cfg.data.db_url)


# ─── Commands ─────────────────────────────────────────────────────────────────


@app.command()
def scan(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Single symbol to scan"),
    timeframe: str = typer.Option(None, "--timeframe", "-t", help="Override timeframe"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Scan watchlist (or a specific symbol) for signals."""
    _setup("DEBUG" if verbose else "INFO")
    from msomi.ai.analyst import AIAnalyst
    from msomi.alerts.telegram import TelegramAlerter
    from msomi.journal.logger import TradeJournal
    from msomi.risk.manager import RiskManager
    from msomi.signals.engine import SignalEngine

    cfg = get_config()
    settings = get_settings()

    engine = SignalEngine(cfg)
    risk = RiskManager(balance=settings.account_balance, config=cfg)
    analyst = AIAnalyst(cfg)
    alerter = TelegramAlerter(cfg)
    journal = TradeJournal()

    symbols = [symbol] if symbol else None

    console.rule("[bold gold1]Msomi Signal Scan[/]")
    events = engine.scan(symbols=symbols, timeframe=timeframe)

    if not events:
        console.print("[dim]No signals above threshold.[/]")
        return

    for event in events:
        s = event.signal
        assessment = risk.assess(s)
        ai_text = analyst.narrate_signal(event.symbol, event.timeframe, s)

        # Log to journal
        journal.log_signal(event, ai_analysis=ai_text)

        # Alert
        alerter.send_signal(event, assessment, ai_text)

        # Print to console
        color = "green" if s.direction == "LONG" else "red"
        console.print(
            f"\n[bold {color}]▲ {s.direction}[/] [bold]{event.symbol}[/] [{event.timeframe}] "
            f"Score=[bold]{s.score}[/]/100"
        )
        console.print(f"  Entry: {s.entry_price:.5f}  SL: {s.stop_loss:.5f}  TP: {s.take_profit:.5f}  R:R: {s.risk_reward:.2f}")

        if not assessment.allowed:
            console.print(f"  [red]⛔ Risk blocked: {assessment.rejection_reason}[/]")
        else:
            pos = assessment.position
            if pos:
                console.print(f"  [green]✓ Size: {pos.units:.4f}  Risk: ${pos.risk_amount:.2f}[/]")

        if ai_text:
            console.print(f"\n  [dim]{ai_text[:300]}…[/]")


@app.command()
def watch(
    interval: int = typer.Option(60, "--interval", "-i", help="Scan interval in minutes"),
    timeframe: str = typer.Option(None, "--timeframe", "-t"),
) -> None:
    """Run continuous signal scanning loop."""
    _setup()
    console.print(f"[bold gold1]Msomi[/] watching markets every [bold]{interval}[/] minutes…")
    console.print("Press Ctrl+C to stop.\n")

    try:
        while True:
            scan(symbol=None, timeframe=timeframe, verbose=False)
            console.print(f"\n[dim]Next scan in {interval} minutes…[/]")
            time.sleep(interval * 60)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")


@app.command()
def backtest(
    symbol: str = typer.Argument("EURUSD=X", help="Symbol to backtest"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t"),
    start: str = typer.Option(None, "--start"),
    end: str = typer.Option(None, "--end"),
    min_score: int = typer.Option(None, "--min-score", "-m"),
) -> None:
    """Run a backtest for a symbol."""
    _setup()
    from msomi.backtest.engine import BacktestEngine

    cfg = get_config()
    engine = BacktestEngine(cfg)

    console.rule(f"[bold gold1]Backtest: {symbol}[/]")
    report = engine.run(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        min_score=min_score,
    )
    console.print(report.summary())

    # Trade table
    if report.trades:
        table = Table(title="Trades", show_header=True)
        table.add_column("Entry Time")
        table.add_column("Dir")
        table.add_column("Entry")
        table.add_column("Exit")
        table.add_column("P&L%")
        table.add_column("Outcome")
        for t in report.trades[-20:]:
            color = "green" if t.outcome == "WIN" else "red"
            table.add_row(
                str(t.entry_time)[:16],
                t.direction,
                f"{t.entry_price:.4f}",
                f"{t.exit_price:.4f}",
                f"{t.pnl_pct:+.2f}%",
                f"[{color}]{t.outcome}[/]",
            )
        console.print(table)


@app.command()
def journal_cmd(days: int = typer.Option(30, "--days", "-d")) -> None:
    """Print performance summary from the trade journal."""
    _setup()
    from msomi.journal.logger import TradeJournal

    cfg = get_config()
    j = TradeJournal()
    perf = j.performance_summary(days=days)

    console.rule(f"[bold gold1]Performance — Last {days} days[/]")
    table = Table(show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")
    for key, val in perf.items():
        if key == "trade_breakdown":
            continue
        table.add_row(key.replace("_", " ").title(), str(round(val, 4) if isinstance(val, float) else val))
    console.print(table)


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    from pathlib import Path

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print("[bold gold1]Launching Msomi Dashboard…[/]")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", "8501"],
        check=False,
    )


# rename to avoid clash with typer.command name
app.command(name="journal")(journal_cmd)

if __name__ == "__main__":
    app()
