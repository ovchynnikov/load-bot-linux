"""Download videos from tiktok, x(twitter), reddit, youtube shorts, instagram reels and many more"""

import os
import random
import json
from functools import lru_cache
from dotenv import load_dotenv
from telegram import Update
from telegram.error import TimedOut, NetworkError, TelegramError
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram.constants import MessageEntityType
from logger import error, info
from video_utils import compress_video, download_video, cleanup_file
from permissions import inform_user_not_allowed, is_user_or_chat_not_allowed, supported_sites

load_dotenv()

language = os.getenv("LANGUAGE", "ua").lower()  # Default to Ukrainian if not set


# Cache responses from JSON file
@lru_cache(maxsize=1)
def load_responses():
    """Function loading bot responses based on language setting."""

    filename = "responses_ua.json" if language == "ua" else "responses_en.json"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data["responses"]
    except FileNotFoundError:
        # Return a minimal set of responses if no response files found
        not_found_responses = {
            "en": "Sorry, I'm having trouble loading my responses right now! 😅",
            "ua": "Вибачте, у мене проблеми із завантаженням відповідей! 😅",
        }
        return not_found_responses[language]


responses = load_responses()


def spoiler_in_message(entities):
    """
    Checks if any of the provided message entities contain a spoiler.

    This function iterates through the list of message entities and checks if
    any of them have the type `MessageEntityType.SPOILER`.

    Args:
        entities (list): A list of `MessageEntity` objects from a Telegram message.
                         These entities describe parts of the message (e.g., bold text,
                         spoilers, mentions, etc.).

    Returns:
        bool: True if a spoiler entity is found, False otherwise.

    Example:
        entities = [
            MessageEntity(length=65, offset=0, type=MessageEntityType.SPOILER),
            MessageEntity(length=10, offset=70, type=MessageEntityType.BOLD)
        ]
        spoiler_in_message(entities)  # Returns: True
    """
    if entities:
        for entity in entities:
            if entity.type == MessageEntityType.SPOILER:
                return True
    return False


def is_bot_mentioned(message_text: str) -> bool:
    """
    Checks if the bot is mentioned in the message text.

    Args:
        message_text (str): The text of the message to check.

    Returns:
        bool: True if the bot is mentioned, False otherwise.
    """
    bot_trigger_words = ["ботяра", "bot_health"]
    return any(word in message_text.lower().split() for word in bot_trigger_words)


def clean_url(message_text: str) -> str:
    """
    Cleans the URL from the message text by removing unwanted characters.

    Args:
        message_text (str): The text of the message containing the URL.

    Returns:
        str: The cleaned URL.
    """
    return message_text.replace("**", "") if message_text.startswith("**") else message_text


def is_large_file(file_path: str, max_size_mb: int = 50) -> bool:
    """
    Checks if the file size exceeds the specified maximum size.

    Args:
        file_path (str): The path to the file to check.
        max_size_mb (int): The maximum file size in megabytes (default is 50MB).

    Returns:
        bool: True if the file size exceeds the maximum size, False otherwise.
    """
    return os.path.exists(file_path) and (os.path.getsize(file_path) / (1024 * 1024)) > max_size_mb


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):  # pylint: disable=unused-argument
    """
    Handles incoming messages from the Telegram bot.

    This function processes text messages sent to the bot and determines the appropriate response
    based on the message content. It supports specific keywords and URLs, such as Instagram Reels,
    Facebook Reels, YouTube Shorts, TikTok, Reddit, and X/Twitter, and attempts to download and send
    the corresponding video to the user.

    Parameters:
        update (telegram.Update): Represents the incoming update from the Telegram bot.
        context (ContextTypes.DEFAULT_TYPE): The context object for the handler.

    Behavior:
        - If the message contains "ботяра" (case insensitive), responds with a random response
          from a predefined list of bot responses.
        - If the message contains an Instagram Stories URL, informs the user that downloading is not supported.
        - If the message contains a supported URL
          (Instagram Reels, Facebook Reels, YouTube Shorts,
          TikTok, Reddit, X/Twitter):
            - Downloads and optionally compresses the video.
            - Sends the video back to the user via Telegram.
            - Preserves any spoiler tags present in the original message.
            - Cleans up temporary files after sending the video.
        - Handles error cases with appropriate user feedback, ensuring a smooth user experience.

    Returns:
        None
    """
    if not update.message or not update.message.text:
        return

    message_text = update.message.text.strip()

    # Handle bot mention response
    if is_bot_mentioned(message_text):
        await respond_with_bot_message(update)
        return

    # Ignore if message doesn't contain http
    if "http" not in message_text:
        return

    # Check if user is not allowed
    if is_user_or_chat_not_allowed(update.effective_user.username, update.effective_chat.id):
        await inform_user_not_allowed(update)
        return

    # Handle Instagram stories
    if "instagram.com/stories/" in message_text:
        instagram_stories_responses = {
            "ua": "Сторіз не можу скачати.",
            "en": "Instagram stories not supported."
        }
        await update.message.reply_text(instagram_stories_responses[language])
        return

    message_text = message_text.replace("** ", "**")

    # Check if URL is from a supported site. Ignore if it's from a group or channel
    if not any(site in message_text for site in supported_sites):
        if update.effective_chat.type == "private":
            not_supported_responses = {
                "ua": "Цей сайт не підтримується. Спробуйте додати ** перед https://",
                "en": "This site is not supported. Try adding ** before the https://"
            }
            await update.message.reply_text(not_supported_responses[language])
            return  # Stop further execution after sending the reply
        return

    try:
        # Remove '**' prefix and any spaces if present
        url = clean_url(message_text)

        # Download the video
        video_path = download_video(url)

        # Check if video was downloaded
        if not video_path or not os.path.exists(video_path):
            return

        # Compress video if it's larger than 50MB
        if is_large_file(video_path):
            compress_video(video_path)

        # Check for spoiler flag
        has_spoiler = spoiler_in_message(update.message.entities)

        # Send the video to the chat
        await send_video(update, video_path, has_spoiler)

    finally:
        # Clean up temporary files
        if video_path and os.path.exists(video_path):
            cleanup_file(video_path)


async def respond_with_bot_message(update: Update) -> None:
    """
    Responds to the user with a random bot response when the bot is mentioned.

    Args:
        update (telegram.Update): Represents the incoming update from the Telegram bot.

    Returns:
        None
    """
    response_message = random.choice(responses)  # Select a random response from the predefined list
    await update.message.reply_text(
        f"{response_message}\n"
        f"[Chat ID]: {update.effective_chat.id}\n"
        f"[Username]: {update.effective_user.username}"
    )


async def send_video(update: Update, video_path: str, has_spoiler: bool) -> None:
    """
    Sends the video to the chat.

    Args:
        update (telegram.Update): Represents the incoming update from the Telegram bot.
        video_path (str): The path to the video file to send.
        has_spoiler (bool): Indicates if the message contains a spoiler.

    Returns:
        None
    """
    try:
        with open(video_path, 'rb') as video_file:
            await update.message.chat.send_video(
                video=video_file,
                has_spoiler=has_spoiler,
                disable_notification=True,
                write_timeout=8000,
                read_timeout=8000,
            )
    except TimedOut as e:
        error("Telegram timeout while sending video. %s", e)
    except (NetworkError, TelegramError) as e:
        await update.message.reply_text(
            f"Error sending video: {str(e)}. Please try again later."
        )


def main():
    """
    Entry point for the Telegram bot application.

    This function initializes the bot, sets up message handling, and starts the bot's polling loop.

    Steps:
        1. Retrieves the bot token from the environment variable `BOT_TOKEN`.
        2. Builds a Telegram bot application using the `Application.builder()` method.
        3. Adds a message handler to process all text messages (excluding commands) using the
           `handle_message` function.
        4. Prints a message to indicate the bot has started.
        5. Starts the bot's polling loop, allowing it to listen for incoming updates until
           manually stopped (Ctrl+C).

    Dependencies:
        - Requires the `BOT_TOKEN` environment variable to be set with the bot's token.
        - Depends on `handle_message` for processing incoming messages.

    Notes:
        - Designed to be run as the `__main__` function in a Python script.
        - Uses the `telegram.ext.Application` and `MessageHandler` from the Telegram Bot API library.

    Returns:
        None
    """
    bot_token = os.getenv("BOT_TOKEN")
    application = Application.builder().token(bot_token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    info("Bot started. Ctrl+C to stop")
    application.run_polling()


if __name__ == "__main__":
    main()
