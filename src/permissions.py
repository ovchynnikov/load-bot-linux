"""Module for handling user permissions and access control.
This module provides functionality to check if users are allowed to access
the bot based on their Telegram usernames and chat IDs configured in environment variables.
"""

import os
from typing import Optional
from telegram import Update

allowed_usernames = [x.strip() for x in os.getenv("ALLOWED_USERNAMES", "").split(",") if x]
allowed_chat_ids = [int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x]
limit_bot_access = os.getenv("LIMIT_BOT_ACCESS", "False")


# Check if user or chat is not allowed. Returns True if not allowed, False if allowed
def is_user_or_chat_not_allowed(username: Optional[str], chat_id: int) -> bool:
    """Check if username or chat_id is not in the allowed lists.

    Args:
        username: Telegram username to check
        chat: Telegram chat ID to check

    Returns:
        True if neither user nor chat is allowed, False if either is allowed
    """
    # default case when no limits are set
    if limit_bot_access == "False":
        return False

    # If chat_id is allowed, grant access regardless of username
    if chat_id in allowed_chat_ids:
        return False

    # Otherwise check if username is allowed
    return username not in allowed_usernames


# Function to inform the user they are not allowed to use the bot
async def inform_user_not_allowed(update: Update) -> None:
    """
    Informs the user that they are not allowed to use the bot.

    This function sends a message to the user indicating that they do not have permission
    to use the bot. It only responds if the chat type is private.

    Args:
        update (telegram.Update): Represents the incoming update from the Telegram bot.

    Returns:
        None
    """
    if update.effective_chat.type == "private":
        await update.message.reply_text(
            f"You are not allowed to use this bot.\n "
            f"[Username]:  {update.effective_user.username}\n "
            f"[Chat ID]: {update.effective_chat.id}",
            reply_to_message_id=update.message.message_id
        )


supported_sites = [
    "**https://",
    "facebook.com/",
    "instagram.com/",
    "tiktok.com/",
    "reddit.com/",
    "x.com/",
    "youtube.com/shorts",
]
