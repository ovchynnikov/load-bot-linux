"""Bounded plain-text Telegram delivery without regenerating on send failure."""

import asyncio

from llm_text import split_reply


def message_topic_id(message):
    """A non-forum reply chain's thread ID is not a Bot API topic ID."""
    return message.message_thread_id if message.is_topic_message else None


async def send_reply(update, text, on_first_sent=None, timeout=120, on_final_sent=None):
    """Thread every chunk to its triggering message and preserve forum topics."""
    async with asyncio.timeout(timeout):
        chunks = split_reply(text)
        for index, chunk in enumerate(chunks):
            await update.message.reply_text(
                chunk,
                reply_to_message_id=update.message.message_id,
                message_thread_id=message_topic_id(update.message),
                do_quote=False,
                parse_mode=None,
                disable_web_page_preview=True,
            )
            if index == len(chunks) - 1 and on_final_sent is not None:
                await on_final_sent()
            if index == 0 and on_first_sent is not None:
                await on_first_sent()
