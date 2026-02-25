"""
Notification service — pluggable backends for trade alerts.
Supports: console (default), telegram, slack, sms (stubs ready).
"""

import json
import logging
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NotificationBackend(ABC):
    """Abstract base for notification delivery."""

    @abstractmethod
    def send(self, title: str, message: str, level: str = "info") -> bool:
        """Send a notification. Returns True if sent successfully."""
        ...


class ConsoleBackend(NotificationBackend):
    """Print notifications to stdout/logs."""

    def send(self, title: str, message: str, level: str = "info") -> bool:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "📢")
        print(f"\n{icon} [{ts}] {title}")
        for line in message.strip().split("\n"):
            print(f"   {line}")
        return True


class TelegramBackend(NotificationBackend):
    """Send via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, title: str, message: str, level: str = "info") -> bool:
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "📢")
        text = f"{icon} *{title}*\n```\n{message}\n```"
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            return resp.ok
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


class SlackBackend(NotificationBackend):
    """Send via Slack incoming webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, title: str, message: str, level: str = "info") -> bool:
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "📢")
        text = f"{icon} *{title}*\n```{message}```"
        try:
            resp = requests.post(
                self.webhook_url,
                json={"text": text},
                timeout=10,
            )
            return resp.ok
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False


class SMSBackend(NotificationBackend):
    """Send via generic HTTP SMS gateway (e.g. Twilio, Vonage)."""

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key

    def send(self, title: str, message: str, level: str = "info") -> bool:
        text = f"{title}: {message[:160]}"
        try:
            resp = requests.post(
                self.endpoint,
                json={"message": text, "api_key": self.api_key},
                timeout=10,
            )
            return resp.ok
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return False


class NotificationService:
    """Unified notification service with multiple backends."""

    def __init__(self, config: dict):
        self.backends: list[NotificationBackend] = []

        backend_name = config.get("backend", "console")

        # Always add console
        self.backends.append(ConsoleBackend())

        if backend_name == "telegram" and config.get("telegram_bot_token"):
            self.backends.append(TelegramBackend(
                config["telegram_bot_token"],
                config["telegram_chat_id"],
            ))

        if backend_name == "slack" and config.get("slack_webhook_url"):
            self.backends.append(SlackBackend(config["slack_webhook_url"]))

        if backend_name == "sms" and config.get("sms_endpoint"):
            self.backends.append(SMSBackend(
                config["sms_endpoint"],
                config["sms_api_key"],
            ))

    def notify(self, title: str, message: str, level: str = "info"):
        """Send notification to all configured backends."""
        for backend in self.backends:
            try:
                backend.send(title, message, level)
            except Exception as e:
                logger.error(f"Notification failed ({type(backend).__name__}): {e}")

    def trade_opened(self, slug: str, signal: str, token_id: str, price: float,
                     size: float, shares: float):
        self.notify("TRADE OPENED", (
            f"Market:  {slug}\n"
            f"Signal:  {signal}\n"
            f"Token:   {token_id[:20]}...\n"
            f"Price:   ${price:.4f}\n"
            f"Size:    ${size:.2f}\n"
            f"Shares:  {shares:.2f}"
        ), "info")

    def trade_result(self, slug: str, signal: str, outcome: str, profit: float,
                     capital: float, daily_pnl: float, total_pnl: float):
        level = "success" if profit > 0 else "warning"
        self.notify("MARKET RESULT", (
            f"Market:    {slug}\n"
            f"Signal:    {signal}\n"
            f"Outcome:   {outcome}\n"
            f"Profit:    ${profit:+.2f}\n"
            f"Capital:   ${capital:.2f}\n"
            f"Daily PnL: ${daily_pnl:+.2f}\n"
            f"Total PnL: ${total_pnl:+.2f}"
        ), level)

    def daily_loss_hit(self, daily_loss: float, max_loss: float):
        self.notify("⛔ DAILY LOSS LIMIT HIT", (
            f"Daily loss: ${daily_loss:.2f}\n"
            f"Max allowed: ${max_loss:.2f}\n"
            f"Bot is pausing until next day."
        ), "error")

    def bot_started(self, config_summary: str):
        self.notify("BOT STARTED", config_summary, "info")

    def bot_error(self, error: str):
        self.notify("BOT ERROR", error, "error")

    def redemption_result(self, slug: str, amount: float, success: bool):
        level = "success" if success else "error"
        status = "SUCCESS" if success else "FAILED"
        self.notify(f"REDEMPTION {status}", (
            f"Market: {slug}\n"
            f"Amount: ${amount:.2f}"
        ), level)
