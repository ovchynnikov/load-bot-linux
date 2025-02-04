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

admins_chat_ids = os.getenv("ADMINS_CHAT_IDS")
send_error_to_admin = os.getenv("SEND_ERROR_TO_ADMIN", "False").lower() == "true"


async def error_handler(update: Optional[Update], context: CallbackContext):
    """Log the error and send a message to the admins.
    Works only if SEND_ERROR_TO_ADMIN=True in .env file. and ADMINS_CHAT_IDS is set.
    If SEND_ERROR_TO_ADMIN=False, the error will be logged but not sent to admins.
    Works only for Exceptions errors that are not handled by the bot code.
    """

    if update is None:
        error("Update is None. Cannot log error.")
        return  # Exit the function if update is not valid
    if update.effective_sender is None:
        error("Effective sender is None. Cannot log error.")
        return  # Exit the function if effective_sender is not valid
    username = update.effective_sender.username
    error("An error occurred while processing the update from %s: %s", username, context.error)
    # Log the error
    debug("Update %s caused error %s", update, context.error)
    debug("send_error_to_admin: %s, admin_chat_id: %s", send_error_to_admin, admins_chat_ids)
    if send_error_to_admin and admins_chat_ids:
        admin_ids = admins_chat_ids.split(",")  # Split the string into a list of IDs
        for admin_chat_id in admin_ids:
            await context.bot.send_message(
                chat_id=admin_chat_id.strip(),  # Strip any whitespace
                text=f"`{context.error}` \n\nWho triggered the error: `@{username}`.\nUrl was {update.message.text}",
                disable_web_page_preview=True,
                parse_mode='Markdown',
            )
    else:
        debug("Admin chat IDs are not set; error message not sent to admins.")
