"""Telegram alerter — rich signal cards and reports via python-telegram-bot."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from msomi.core.config import MsomiConfig, get_config, get_settings
from msomi.risk.manager import RiskAssessment
from msomi.signals.engine import SignalEvent

logger = logging.getLogger(__name__)

# Characters that must be escaped in MarkdownV2
_MDV2_SPECIAL = r"\_*[]()~`>#+-=|{}.!"


def _esc(text: str) -> str:
    """Escape all MarkdownV2 special characters."""
    return re.sub(r"([\_\*\[\]\(\)\~\`\>\#\+\-\=\|\{\}\.\!])", r"\\\1", str(text))


class TelegramAlerter:
    """
    Sends Msomi alerts to a Telegram chat using python-telegram-bot.

    Falls back to plain text if MarkdownV2 parsing fails.
    """

    def __init__(self, config: Optional[MsomiConfig] = None) -> None:
        self.cfg = config or get_config()
        self._settings = get_settings()
        self._token: Optional[str] = self._settings.telegram_bot_token
        self._chat_id: Optional[str] = self._settings.telegram_chat_id
        self._enabled = bool(
            self._token and self._chat_id
            and self._token != "your_telegram_bot_token_here"
            and self.cfg.alerts.telegram_enabled
        )
        if not self._enabled:
            logger.info("Telegram alerts disabled (token/chat_id not configured)")

    # ── Public API ────────────────────────────────────────────────────────────

    def send_signal(
        self,
        event: SignalEvent,
        assessment: RiskAssessment,
        ai_analysis: str = "",
    ) -> None:
        """Send a rich signal card to Telegram."""
        if not self._enabled:
            return

        sig = event.signal
        direction_emoji = "🟢 LONG" if sig.direction == "LONG" else "🔴 SHORT"
        risk_line = (
            f"⛔ *Blocked*: {_esc(assessment.rejection_reason)}"
            if not assessment.allowed
            else (
                f"✅ Size: `{assessment.position.units:.4f}` "
                f"\\| Risk: `${assessment.position.risk_amount:.2f}`"
                if assessment.position
                else "✅ Viable"
            )
        )

        reasons_text = "\n".join(f"  • {_esc(r)}" for r in sig.reasons[:5])
        ai_snippet = _esc(ai_analysis[:280] + "…") if ai_analysis and len(ai_analysis) > 280 else _esc(ai_analysis)

        message = (
            f"*{_esc(event.symbol)}* \\[{_esc(event.timeframe)}\\] — {direction_emoji}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Score: *{sig.score}/100*\n"
            f"Entry: `{sig.entry_price:.5f}`\n"
            f"Stop Loss: `{sig.stop_loss:.5f}`\n"
            f"Take Profit: `{sig.take_profit:.5f}`\n"
            f"R:R: `{sig.risk_reward:.2f}`\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"*Risk Assessment*\n"
            f"{risk_line}\n"
        )

        if reasons_text:
            message += f"━━━━━━━━━━━━━━━━━\n*Confluence*\n{reasons_text}\n"

        if ai_snippet:
            message += f"━━━━━━━━━━━━━━━━━\n_{ai_snippet}_\n"

        self._send(message, parse_mode="MarkdownV2")

    def send_circuit_breaker(self, reason: str, daily_pnl: float) -> None:
        """Send a circuit breaker trip alert."""
        if not self._enabled:
            return
        message = (
            f"⚠️ *CIRCUIT BREAKER TRIGGERED*\n"
            f"Reason: {_esc(reason)}\n"
            f"Daily P&L: `{daily_pnl:+.2f}`\n"
            f"_Trading halted for today\\. Resume tomorrow\\._"
        )
        self._send(message, parse_mode="MarkdownV2")

    def send_eod_report(self, stats: dict) -> None:
        """Send end-of-day performance summary."""
        if not self._enabled:
            return

        total = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        pnl = stats.get("total_pnl", 0.0)
        win_rate = (wins / total * 100) if total > 0 else 0.0
        pnl_emoji = "📈" if pnl >= 0 else "📉"

        message = (
            f"{pnl_emoji} *End of Day Report*\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Trades: `{total}` \\| W: `{wins}` L: `{losses}`\n"
            f"Win Rate: `{win_rate:.1f}%`\n"
            f"P&L: `{pnl:+.2f}`\n"
        )

        pf = stats.get("profit_factor")
        if pf is not None:
            message += f"Profit Factor: `{pf:.2f}`\n"

        ai_summary = stats.get("ai_summary", "")
        if ai_summary:
            message += f"━━━━━━━━━━━━━━━━━\n_{_esc(ai_summary[:300])}_\n"

        self._send(message, parse_mode="MarkdownV2")

    def send_text(self, text: str) -> None:
        """Send a plain text message."""
        if not self._enabled:
            return
        self._send(text, parse_mode=None)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send(self, text: str, parse_mode: Optional[str] = "MarkdownV2") -> None:
        """Send message, handling sync/async contexts and retrying on parse error."""
        try:
            self._send_async(text, parse_mode)
        except Exception as exc:
            if parse_mode == "MarkdownV2":
                logger.warning("MarkdownV2 send failed (%s), retrying as plain text", exc)
                try:
                    # Strip markdown and retry
                    plain = re.sub(r"[*_`\\]", "", text)
                    self._send_async(plain, None)
                except Exception as exc2:
                    logger.error("Telegram send failed: %s", exc2)
            else:
                logger.error("Telegram send failed: %s", exc)

    def _send_async(self, text: str, parse_mode: Optional[str]) -> None:
        """Dispatch to the correct event loop context."""
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context — schedule as a task
            loop.create_task(self._do_send(text, parse_mode))
        except RuntimeError:
            # No running event loop — use asyncio.run
            asyncio.run(self._do_send(text, parse_mode))

    async def _do_send(self, text: str, parse_mode: Optional[str]) -> None:
        try:
            from telegram import Bot  # lazy import
        except ImportError:
            logger.warning("python-telegram-bot not installed; skipping alert")
            return

        bot = Bot(token=self._token)
        kwargs: dict = {"chat_id": self._chat_id, "text": text}
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        async with bot:
            await bot.send_message(**kwargs)
