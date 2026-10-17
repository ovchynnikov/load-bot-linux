"""
This module handles error logging and notification for the Telegram bot.

It defines an error handler that logs errors and optionally sends error messages
to a list of admin users if configured to do so. The behavior is controlled by
environment variables that specify admin chat IDs and whether to send error
notifications.

Dependencies:
- telegram: For interacting with the Telegram Bot API.
"""

import os
from typing import Optional
from telegram import Update
from telegram.ext import CallbackContext
from logger import error, debug
from safe_diagnostics import download_diagnostic_url, exception_summary

admins_chat_ids = os.getenv("ADMINS_CHAT_IDS")
send_error_to_admin = os.getenv("SEND_ERROR_TO_ADMIN", "False").lower() == "true"


async def error_handler(update: Optional[Update], context: CallbackContext):
    """Log the error and send a message to the admins.
    Works only if SEND_ERROR_TO_ADMIN=True in .env file. and ADMINS_CHAT_IDS is set.
    If SEND_ERROR_TO_ADMIN=False, the error will be logged but not sent to admins.
    Works only for Exceptions errors that are not handled by the bot code.
    """

    user = update.effective_user if isinstance(update, Update) else None
    user_id = user.id if user else None
    diagnostic = exception_summary(context.error)
    message = update.effective_message if isinstance(update, Update) else None
    download_url = download_diagnostic_url(message.text) if message else None
    error("Unhandled bot error user_id=%s %s download_url=%s", user_id, diagnostic, download_url)
    debug("send_error_to_admin: %s, admin_chat_id: %s", send_error_to_admin, admins_chat_ids)
    if send_error_to_admin and admins_chat_ids:
        admin_ids = admins_chat_ids.split(",")  # Split the string into a list of IDs
        summary = "Помилка бота: " if os.getenv("LANGUAGE", "uk").lower() in {"uk", "ua"} else "Bot error: "
        for admin_chat_id in admin_ids:
            try:
                await context.bot.send_message(
                    chat_id=admin_chat_id.strip(),
                    text=f"{summary}{diagnostic}; user_id={user_id}"
                    + (f"; download_url={download_url}" if download_url else ""),
                    disable_web_page_preview=True,
                    parse_mode=None,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error("Admin error notification failed %s", exception_summary(exc))
    else:
        debug("Admin chat IDs are not set; error message not sent to admins.")
