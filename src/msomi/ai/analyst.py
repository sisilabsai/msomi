"""AI analyst — Claude/OpenAI signal narration with deterministic fallback."""

from __future__ import annotations

import logging
from typing import Optional

from msomi.core.config import MsomiConfig, get_config
from msomi.signals.confluence import ScoredSignal

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────

_SIGNAL_PROMPT = """\
You are a professional Forex/Crypto trading analyst. Analyze this trading signal and provide a concise 3-4 paragraph assessment.

Symbol: {symbol}
Timeframe: {timeframe}
Direction: {direction}
Confidence Score: {score}/100
Entry: {entry_price}
Stop Loss: {stop_loss}
Take Profit: {take_profit}
Risk/Reward: {risk_reward:.2f}

Indicator readings:
- EMA Trend: {ema_trend}
- RSI: {rsi:.1f} (signal: {rsi_signal})
- MACD Histogram: {macd_hist:.6f} (crossover: {crossover})
- Bollinger Band position: {bb_position}
- ATR (volatility): {atr:.5f} ({atr_pct:.2f}% of price)
- VWAP: price is {price_vs_vwap}
- Volume ratio: {volume_ratio:.2f}x average

Confluence components:
{components}

Paragraph 1 — Market context: What does the current price action and trend suggest?
Paragraph 2 — Indicator confluence: Which indicators support or contradict this setup?
Paragraph 3 — Risk considerations: Key risks, invalidation levels, and what to watch.
Paragraph 4 — Recommendation: Clear, actionable guidance with conviction level.

Keep each paragraph to 2-3 sentences. Be direct and specific.
"""

_WEEKLY_REVIEW_PROMPT = """\
You are a trading performance coach. Review this week's trading stats and provide actionable feedback.

Period: {period}
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}
Total P&L: {total_pnl:+.2f}
Average Win: {avg_win:+.2f}
Average Loss: {avg_loss:+.2f}
Best Trade: {best_trade:+.2f}
Worst Trade: {worst_trade:+.2f}
Consecutive Losses (max): {max_streak}

Provide:
1. Performance summary (2-3 sentences)
2. Key strengths this week
3. Key areas for improvement
4. Specific actions for next week

Be honest, constructive, and specific. Focus on process, not just outcomes.
"""

_POST_TRADE_PROMPT = """\
You are a trading coach reviewing a completed trade. Provide a brief post-trade review.

Symbol: {symbol}
Direction: {direction}
Entry: {entry_price} → Exit: {exit_price}
P&L: {pnl:+.2f} ({pnl_pct:+.2f}%)
Outcome: {outcome}
Confidence at entry: {confidence}/10
Emotion tag: {emotion}
Notes: {notes}

Provide a 2-3 sentence review covering: (1) was the trade process correct regardless of outcome, (2) what could be improved, (3) one key lesson.
"""


class AIAnalyst:
    """
    AI-powered signal narration using Claude or OpenAI.

    Falls back to a deterministic narration if API keys are not configured
    or if the API call fails.
    """

    def __init__(self, config: Optional[MsomiConfig] = None) -> None:
        self.cfg = config or get_config()
        self._ai_cfg = self.cfg.ai

    # ── Public API ────────────────────────────────────────────────────────────

    def narrate_signal(
        self,
        symbol: str,
        timeframe: str,
        signal: ScoredSignal,
    ) -> str:
        """Generate a plain-language analysis of a trading signal."""
        if not self._ai_cfg.narrate_signals:
            return self._fallback_narration(symbol, timeframe, signal)

        snap = signal.snapshot
        components_text = "\n".join(
            f"  - {k}: {v}/100" for k, v in signal.components.items()
        )

        prompt = _SIGNAL_PROMPT.format(
            symbol=symbol,
            timeframe=timeframe,
            direction=signal.direction,
            score=signal.score,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward=signal.risk_reward,
            ema_trend=snap.trend_direction,
            rsi=snap.rsi,
            rsi_signal=snap.rsi_signal,
            macd_hist=snap.macd_hist,
            crossover="bullish" if snap.macd_crossover > 0 else ("bearish" if snap.macd_crossover < 0 else "none"),
            bb_position=snap.bb_position,
            atr=snap.atr,
            atr_pct=snap.atr_pct * 100,
            price_vs_vwap=snap.price_vs_vwap,
            volume_ratio=snap.volume_ratio,
            components=components_text,
        )

        try:
            return self._call_ai(prompt)
        except Exception as exc:
            logger.warning("AI narration failed (%s), using fallback", exc)
            return self._fallback_narration(symbol, timeframe, signal)

    def weekly_review(self, stats: dict) -> str:
        """Generate a structured weekly performance review."""
        prompt = _WEEKLY_REVIEW_PROMPT.format(
            period=stats.get("period", "this week"),
            total_trades=stats.get("total_trades", 0),
            win_rate=stats.get("win_rate", 0.0) * 100,
            profit_factor=stats.get("profit_factor", 0.0),
            total_pnl=stats.get("total_pnl", 0.0),
            avg_win=stats.get("avg_win", 0.0),
            avg_loss=stats.get("avg_loss", 0.0),
            best_trade=stats.get("best_trade", 0.0),
            worst_trade=stats.get("worst_trade", 0.0),
            max_streak=stats.get("max_consecutive_losses", 0),
        )
        try:
            return self._call_ai(prompt)
        except Exception as exc:
            logger.warning("Weekly review AI call failed: %s", exc)
            return self._fallback_weekly(stats)

    def post_trade_review(self, context: dict) -> str:
        """Brief post-trade analysis for the journal."""
        prompt = _POST_TRADE_PROMPT.format(
            symbol=context.get("symbol", ""),
            direction=context.get("direction", ""),
            entry_price=context.get("entry_price", 0.0),
            exit_price=context.get("exit_price", 0.0),
            pnl=context.get("pnl", 0.0),
            pnl_pct=context.get("pnl_pct", 0.0),
            outcome=context.get("outcome", ""),
            confidence=context.get("confidence_at_entry", 0),
            emotion=context.get("emotion_tag", "neutral"),
            notes=context.get("notes", "none"),
        )
        try:
            return self._call_ai(prompt)
        except Exception as exc:
            logger.warning("Post-trade review AI call failed: %s", exc)
            return ""

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_ai(self, prompt: str) -> str:
        """Route to the configured AI provider."""
        if self._ai_cfg.provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic  # lazy import
        except ImportError as exc:
            raise RuntimeError("anthropic package not installed") from exc

        from msomi.core.config import get_settings
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=self._ai_cfg.model_anthropic,
            max_tokens=self._ai_cfg.max_tokens,
            temperature=self._ai_cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai  # lazy import
        except ImportError as exc:
            raise RuntimeError("openai package not installed") from exc

        from msomi.core.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=self._ai_cfg.model_openai,
            max_tokens=self._ai_cfg.max_tokens,
            temperature=self._ai_cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _fallback_narration(
        self,
        symbol: str,
        timeframe: str,
        signal: ScoredSignal,
    ) -> str:
        """Deterministic narration when AI is unavailable."""
        snap = signal.snapshot
        reasons_text = "; ".join(signal.reasons[:3]) if signal.reasons else "multiple factors aligned"

        direction_word = "bullish" if signal.direction == "LONG" else "bearish"
        rsi_note = (
            "RSI is in oversold territory supporting a bounce"
            if snap.rsi < 35
            else "RSI is in overbought territory suggesting caution"
            if snap.rsi > 65
            else f"RSI at {snap.rsi:.1f} is neutral"
        )

        return (
            f"{symbol} [{timeframe}] — {direction_word.upper()} setup with confidence {signal.score}/100. "
            f"Key factors: {reasons_text}. "
            f"{rsi_note}. "
            f"Entry {signal.entry_price:.5f}, SL {signal.stop_loss:.5f}, "
            f"TP {signal.take_profit:.5f} (R:R {signal.risk_reward:.2f}). "
            f"Trend direction is {snap.trend_direction}. "
            f"Manage risk strictly — never risk more than your plan allows."
        )

    def _fallback_weekly(self, stats: dict) -> str:
        total = stats.get("total_trades", 0)
        wr = stats.get("win_rate", 0.0) * 100
        pnl = stats.get("total_pnl", 0.0)
        pf = stats.get("profit_factor", 0.0)

        outlook = "positive" if pnl > 0 else "challenging"
        return (
            f"Weekly Review: {total} trades with {wr:.1f}% win rate — a {outlook} week. "
            f"Profit factor: {pf:.2f}. Total P&L: {pnl:+.2f}. "
            f"Continue following your plan, respect stop losses, and review every trade."
        )
