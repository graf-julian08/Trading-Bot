"""
notifications.py — Telegram Notification Service
==================================================
Sends formatted messages to a Telegram chat via the Bot API using aiohttp.
No heavy Telegram bot framework required — just simple HTTPS POST calls.

Rate-limit safe: Telegram allows ~30 messages/second.  We add a small
cooldown between messages to stay well within limits.

Usage:
    notifier = TelegramNotifier()
    await notifier.send_trade_opened(symbol="BTC/USDT", side="buy", ...)
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Optional

import aiohttp

from config import cfg

logger = logging.getLogger(__name__)

# ============================================================================
# Telegram Notifier
# ============================================================================

class TelegramNotifier:
    """Async Telegram message sender via Bot API."""

    _BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    # Minimum seconds between consecutive messages (rate-limit guard).
    _COOLDOWN = 0.05  # 50 ms → max ~20 msg/s (well under 30/s limit)

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._token = bot_token or cfg.TELEGRAM_BOT_TOKEN
        self._chat_id = chat_id or cfg.TELEGRAM_CHAT_ID
        self._enabled = enabled if enabled is not None else cfg.TELEGRAM_ENABLED
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_send_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Core Send
    # ------------------------------------------------------------------

    async def _send_raw(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a raw text message to the configured Telegram chat.

        Returns True on success, False otherwise. Never raises — all
        exceptions are caught and logged so the trading loop is never
        interrupted by a notification failure.
        """
        if not self._enabled:
            logger.debug("Telegram disabled — skipping message.")
            return False

        if not self._token or not self._chat_id:
            logger.warning("Telegram token or chat_id not configured.")
            return False

        # Rate-limit cooldown
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_send_time
        if elapsed < self._COOLDOWN:
            await asyncio.sleep(self._COOLDOWN - elapsed)

        url = self._BASE_URL.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            session = await self._ensure_session()
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                self._last_send_time = asyncio.get_event_loop().time()
                if resp.status == 200:
                    logger.debug("Telegram message sent successfully.")
                    return True
                else:
                    body = await resp.text()
                    logger.error("Telegram API error %d: %s", resp.status, body)
                    return False
        except asyncio.TimeoutError:
            logger.error("Telegram send timed out.")
            return False
        except Exception:
            logger.error("Telegram send failed:\n%s", traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # High-Level Message Templates
    # ------------------------------------------------------------------

    async def send_trade_opened(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        stop_loss: float,
        take_profit: float,
        ml_confidence: float,
    ) -> bool:
        """Notify the user that a new trade has been opened."""
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        text = (
            f"{emoji} <b>Trade Opened</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Price:</b> {price:.8g}\n"
            f"<b>Amount:</b> {amount:.8g}\n"
            f"<b>Stop Loss:</b> {stop_loss:.8g}\n"
            f"<b>Take Profit:</b> {take_profit:.8g}\n"
            f"<b>ML Confidence:</b> {ml_confidence:.1%}\n"
        )
        return await self._send_raw(text)

    async def send_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
    ) -> bool:
        """Notify the user that a trade has been closed."""
        emoji = "✅" if pnl >= 0 else "❌"
        text = (
            f"{emoji} <b>Trade Closed</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Entry:</b> {entry_price:.8g}\n"
            f"<b>Exit:</b> {exit_price:.8g}\n"
            f"<b>PnL:</b> {pnl:+.4f} ({pnl_pct:+.2%})\n"
        )
        return await self._send_raw(text)

    async def send_daily_summary(
        self,
        date_str: str,
        starting_equity: float,
        ending_equity: float,
        realised_pnl: float,
        trade_count: int,
    ) -> bool:
        """Send end-of-day performance summary."""
        change_pct = (
            (ending_equity - starting_equity) / starting_equity
            if starting_equity > 0
            else 0.0
        )
        text = (
            f"📊 <b>Daily Summary — {date_str}</b>\n"
            f"<b>Starting Equity:</b> {starting_equity:.2f}\n"
            f"<b>Ending Equity:</b> {ending_equity:.2f}\n"
            f"<b>Change:</b> {change_pct:+.2%}\n"
            f"<b>Realised PnL:</b> {realised_pnl:+.4f}\n"
            f"<b>Trades:</b> {trade_count}\n"
        )
        return await self._send_raw(text)

    async def send_kill_switch_alert(
        self, daily_pnl_pct: float, threshold: float
    ) -> bool:
        """URGENT: Kill switch has been triggered."""
        text = (
            f"🚨🚨🚨 <b>KILL SWITCH TRIGGERED</b> 🚨🚨🚨\n\n"
            f"Daily drawdown <b>{daily_pnl_pct:+.2%}</b> exceeded "
            f"limit of <b>{threshold:.2%}</b>.\n\n"
            f"<b>All trading has been halted.</b>\n"
            f"Manual intervention required to resume."
        )
        return await self._send_raw(text)

    async def send_error_alert(self, error_msg: str) -> bool:
        """Send an error/exception alert."""
        # Truncate to avoid Telegram's 4096-char limit.
        truncated = error_msg[:3500]
        text = (
            f"⚠️ <b>Bot Error</b>\n"
            f"<pre>{truncated}</pre>"
        )
        return await self._send_raw(text)

    async def send_emergency_exit_alert(
        self,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        trade_id: int,
        attempts: int,
    ) -> bool:
        """
        CRITICAL: Dead-Man's Switch — all exit attempts have failed.

        This is the highest-priority alert. It means the bot has an
        open position it CANNOT close, and human intervention is needed.
        """
        text = (
            f"🚨🚨🚨 <b>CRITICAL: MANUAL INTERVENTION REQUIRED</b> 🚨🚨🚨\n\n"
            f"<b>Dead-Man's Switch activated!</b>\n"
            f"The bot FAILED to close a position after <b>{attempts}</b> attempts.\n\n"
            f"<b>Trade ID:</b> {trade_id}\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Amount:</b> {amount:.8g}\n"
            f"<b>Entry Price:</b> {entry_price:.8g}\n\n"
            f"⚠️ <b>This position is STILL OPEN and UNPROTECTED.</b>\n"
            f"Log into the exchange IMMEDIATELY and close it manually."
        )
        return await self._send_raw(text)

    async def send_model_retrained(
        self, symbol: str, accuracy: float, candle_count: int
    ) -> bool:
        """Notify that the ML model was retrained."""
        text = (
            f"🧠 <b>Model Retrained</b>\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Accuracy:</b> {accuracy:.2%}\n"
            f"<b>Training Candles:</b> {candle_count}\n"
        )
        return await self._send_raw(text)

    async def send_startup(self, pairs: list, paper_mode: bool) -> bool:
        """Notify that the bot has started."""
        mode = "📝 PAPER" if paper_mode else "💰 LIVE"
        text = (
            f"🚀 <b>Trading Bot Started</b>\n"
            f"<b>Mode:</b> {mode}\n"
            f"<b>Pairs:</b> {', '.join(pairs)}\n"
        )
        return await self._send_raw(text)

    async def send_shutdown(self, reason: str = "Manual shutdown") -> bool:
        """Notify that the bot is shutting down."""
        text = (
            f"🛑 <b>Trading Bot Stopped</b>\n"
            f"<b>Reason:</b> {reason}\n"
        )
        return await self._send_raw(text)
