"""Actionable error metadata without arbitrary exception bodies or frame locals."""

from pathlib import Path
from datetime import timedelta
import math
import re
import traceback
from urllib.parse import urlsplit, urlunsplit

from telegram.error import BadRequest, ChatMigrated, Forbidden, RetryAfter

PROVIDER_CODES = {
    "invalid_api_key",
    "authentication_error",
    "permission_denied",
    "model_not_found",
    "invalid_request_error",
    "rate_limit_exceeded",
    "insufficient_quota",
    "context_length_exceeded",
    "server_error",
}
STATUS_REASONS = {
    400: "invalid_request",
    401: "authentication_failed",
    403: "permission_denied",
    404: "resource_or_model_not_found",
    429: "rate_or_quota_limit",
}
SAFE_MESSAGES = {
    "Incomplete or empty Grok response",
    "Search execution was not verified",
    "Empty Gemini response",
}
TELEGRAM_REASONS = {
    "message thread not found": "message_thread_not_found",
    "replied message not found": "replied_message_not_found",
    "reply message not found": "replied_message_not_found",
    "message to be replied not found": "replied_message_not_found",
    "not enough rights to send text messages": "insufficient_send_rights",
    "not enough rights to send text messages to the chat": "insufficient_send_rights",
    "bot was blocked by the user": "bot_blocked_by_user",
    "bot was kicked from the group chat": "bot_kicked_from_chat",
    "bot was kicked from the supergroup chat": "bot_kicked_from_chat",
    "bot is not a member of the supergroup chat": "bot_not_chat_member",
    "chat not found": "chat_not_found",
    "message is too long": "message_too_long",
}
DOWNLOAD_HOSTS = {
    "facebook.com",
    "fb.watch",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
    "redd.it",
    "x.com",
    "twitter.com",
    "youtube.com",
    "youtu.be",
}


def telegram_details(exception):
    """Use fixed reason labels and numeric fields, never an arbitrary description.

    PTB errors can also be constructed locally or wrap transport errors. Their
    class is not proof that the message is a fixed Bot API server description.
    """
    if isinstance(exception, (BadRequest, Forbidden)):
        message = exception.message.casefold().removeprefix("forbidden: ")
        reason = TELEGRAM_REASONS.get(message)
        return [f"telegram_error={reason}"] if reason else []
    if isinstance(exception, RetryAfter):
        delay = exception.retry_after
        if isinstance(delay, timedelta):
            delay = delay.total_seconds()
        if type(delay) in (int, float) and math.isfinite(delay) and delay >= 0:
            return ["telegram_error=flood_control", f"retry_after_seconds={delay:g}"]
    if isinstance(exception, ChatMigrated) and type(exception.new_chat_id) is int:
        return ["telegram_error=chat_migrated", f"new_chat_id={exception.new_chat_id}"]
    return []


def exception_summary(exception):
    """Format allowlisted metadata and frame locations, including chained causes.

    Do not call str/repr on an exception or format_exception: SDK/Telegram error
    messages, subprocess args/stderr and chained causes may contain private input.
    Unknown messages stay omitted; a bounded arbitrary message is still arbitrary.
    """
    parts, seen = [], set()
    current = exception
    while isinstance(current, BaseException) and id(current) not in seen and len(parts) < 4:
        seen.add(id(current))
        details = [f"type={type(current).__name__}"]
        details.extend(telegram_details(current))
        status = getattr(current, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            details.extend([f"status={status}", f"reason={STATUS_REASONS.get(status, 'http_error')}"])
        code = getattr(current, "code", None)
        if isinstance(code, str) and code in PROVIDER_CODES:
            details.append(f"code={code}")
        request_id = getattr(current, "request_id", None)
        if isinstance(request_id, str) and re.fullmatch(
            r"(?:[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|req_[a-zA-Z0-9]{16,64})", request_id
        ):
            details.append(f"request_id={request_id}")
        if isinstance(current, TimeoutError):
            details.append("reason=timeout")
        returncode = getattr(current, "returncode", None)
        if type(returncode) is int:
            details.append(f"exit_code={returncode}")
        if current.args and isinstance(current.args[0], str) and current.args[0] in SAFE_MESSAGES:
            details.append(f"message={current.args[0]}")
        frames = []
        for frame, line in traceback.walk_tb(current.__traceback__):
            filename = Path(frame.f_code.co_filename).name
            function = frame.f_code.co_name
            if re.fullmatch(r"[\w.-]+\.py", filename) and re.fullmatch(r"[\w<>]+", function):
                frames.append(f"{filename}:{line}:{function}")
        if frames:
            details.append("stack=" + ">".join(frames[-6:]))
        parts.append(" ".join(details))
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return (" caused_by=[".join(parts) + "]" * max(0, len(parts) - 1))[:1600]


def download_diagnostic_url(text):
    """Only a standalone supported-site URL; discard credentials/query/fragment."""
    if not isinstance(text, str) or not re.fullmatch(r"https?://\S{1,2048}", text.strip()):
        return None
    try:
        parsed = urlsplit(text.strip())
        host = parsed.hostname
        if not host or not any(host == domain or host.endswith("." + domain) for domain in DOWNLOAD_HOSTS):
            return None
        if any(ord(char) < 32 for char in parsed.path):
            return None
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return None
