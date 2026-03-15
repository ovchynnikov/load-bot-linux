"""Download videos from tiktok, x(twitter), reddit, youtube shorts, instagram reels and many more"""

import base64
import os
import random
import json
import asyncio
import re
import time
import traceback
from datetime import datetime
from typing import Optional
import google.generativeai as genai
from openai import AsyncOpenAI

try:
    import xai_sdk
except ImportError:
    xai_sdk = None
from functools import lru_cache
from collections import defaultdict
from dotenv import load_dotenv
from telegram import Update, InputMediaPhoto, InputMediaVideo
from telegram.error import TimedOut, NetworkError, TelegramError
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram.constants import MessageEntityType
from telegram.request import HTTPXRequest
from logger import error, info, debug
from general_error_handler import error_handler
from permissions import inform_user_not_allowed, is_user_or_chat_not_allowed, supported_sites
from cleanup import cleanup
from db_storage import BotStorage
from video_utils import (
    compress_video,
    download_media,
    get_video_dimensions,
    is_video_duration_over_limits,
    is_video_too_long_to_download,
)

load_dotenv()

# Default to Ukrainian if not set
language = os.getenv("LANGUAGE", "uk").lower()
# Add backward compatibility for old language setting
if language == "ua":
    language = "uk"

# Reply with user data for Healthcheck
send_user_info_with_healthcheck = os.getenv("SEND_USER_INFO_WITH_HEALTHCHECK", "False").lower() == "true"
USE_LLM = os.getenv("USE_LLM", "False").lower() == "true"
USE_CONVERSATION_CONTEXT = os.getenv("USE_CONVERSATION_CONTEXT", "True").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "grok").lower()  # gemini or grok
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-latest")
GROK_IMG_MODEL = os.getenv("GROK_IMG_MODEL", "grok-imagine-image")
TELEGRAM_CONNECT_TIMEOUT = 60
TELEGRAM_POOL_TIMEOUT = 30
TELEGRAM_READ_TIMEOUT = 120
TELEGRAM_WRITE_TIMEOUT = 120
MAX_PROMPT_LEN = 1000


def get_image_caption():
    """Get localized image caption."""
    if language == "uk":
        return "Ось ваше зображення 🖼️"
    else:
        return "Here's your image 🖼️"


IMAGE_CAPTION_STUB = get_image_caption()  # Legacy reference for compatibility
IMAGE_TIMEOUT_SEC = 30.0

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure Grok API
grok_client = None
if GROK_API_KEY:
    grok_client = AsyncOpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

# Configure xAI image API client (grok-imagine-image)
xai_client = None
if GROK_API_KEY and xai_sdk is not None:
    try:
        xai_client = xai_sdk.Client(api_key=GROK_API_KEY)
    except Exception as e:  # pylint: disable=broad-except
        error("Failed to initialize xai_sdk.Client: %s", e)
        xai_client = None

# Rate limiting for LLM APIs
llm_rate_limit = defaultdict(list)  # {user_id: [timestamp1, timestamp2, ...]}
llm_daily_limit = defaultdict(lambda: {"count": 0, "date": ""})  # {user_id: {count, date}}

# Rate limiting for Image Generation
img_gen_rate_limit = defaultdict(list)  # {user_id: [timestamp1, timestamp2, ...]}
img_gen_daily_limit = defaultdict(lambda: {"count": 0, "date": ""})  # {user_id: {count, date}}
LLM_RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", "50"))  # LLM Requests per minute per user
LLM_RPD_LIMIT = int(os.getenv("LLM_RPD_LIMIT", "500"))  # LLM Requests per day per user
IMG_GEN_RPM_LIMIT = int(os.getenv("IMG_GEN_RPM_LIMIT", "1"))  # Image Generation Requests per minute per user
IMG_GEN_RPD_LIMIT = int(os.getenv("IMG_GEN_RPD_LIMIT", "25"))  # Image Generation Requests per day per user

# Conversation context storage: {user_id: [(user_msg, bot_response), ...]}
conversation_context = defaultdict(list)
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "3"))  # Keep last N exchanges
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "500"))  # Max chars per message in context

# User activity tracking for cleanup
user_last_seen = defaultdict(float)  # {user_id: timestamp}
USER_CLEANUP_TTL_DAYS = int(os.getenv("USER_CLEANUP_TTL_DAYS", "3"))  # Days before user data expires
USER_CLEANUP_INTERVAL_HOURS = int(os.getenv("USER_CLEANUP_INTERVAL_HOURS", "24"))  # Cleanup interval

# Allowed LLM providers
ALLOWED_PROVIDERS = {"grok", "gemini"}

# Initialize database storage
db_storage = BotStorage()

# Cleanup task reference
cleanup_task = None


# Cache responses from JSON file
@lru_cache(maxsize=1)
def load_responses():
    """Function loading bot responses based on language setting."""

    filename = "responses_uk.json" if language == "uk" else "responses_en.json"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data["responses"]
    except FileNotFoundError:
        # Return a minimal set of responses if no response files found
        not_found_responses = {
            "en": "Sorry, I'm having trouble loading my responses right now! 😅",
            "uk": "Вибачте, у мене проблеми із завантаженням відповідей! 😅",
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
    # Remove leading/trailing whitespace and convert to lowercase
    cleaned_text = message_text.strip().lower()
    for word in bot_trigger_words:
        if cleaned_text.startswith(word):
            # Check if it's followed by space, punctuation, or is the whole message
            if len(cleaned_text) == len(word) or not cleaned_text[len(word)].isalpha():
                return True
    return False


def extract_image_prompt(message_text: str) -> Optional[str]:
    """Extract image generation prompt for commands like 'ботяра, image: ...'."""
    if not message_text:
        return None

    lower = message_text.lower()
    # Match bot command for image generation: ботяра, image: prompt
    match = re.search(r"ботяра[^\w\d]*image\s*:\s*(.+)", lower)
    if match:
        prompt = match.group(1).strip()
        return prompt or None

    # Fallback for english trigger
    match = re.search(r"bot\s*:\s*image\s*:\s*(.+)", lower)
    if match:
        prompt = match.group(1).strip()
        return prompt or None

    return None


async def generate_image_and_send(update: Update, prompt: str) -> None:
    """Generate image through Grok image API and send to Telegram."""
    if not prompt:
        await update.message.reply_text(
            (
                "Вкажіть, що саме потрібно згенерувати після 'botyara, image:'"
                if language == "uk"
                else "Please specify what you want to generate after 'bot, image:'"
            ),
            reply_to_message_id=update.message.message_id,
        )
        return

    if not GROK_API_KEY:
        await update.message.reply_text(
            (
                "Grok API key не налаштовано. Будь ласка, встановіть GROK_API_KEY."
                if language == "uk"
                else "Grok API key is not configured. Please set GROK_API_KEY."
            ),
            reply_to_message_id=update.message.message_id,
        )
        return

    if not xai_sdk or not xai_client:
        await update.message.reply_text(
            (
                "xAI клієнт недоступний. Перевірте встановлення xai-sdk та GROK_API_KEY."
                if language == "uk"
                else "xAI client is unavailable. Please check xai-sdk installation and GROK_API_KEY."
            ),
            reply_to_message_id=update.message.message_id,
        )
        return

    # Rate limiting for image generation
    user_id = update.effective_user.id
    current_time = time.time()

    # Load img_gen data from DB on first access
    if user_id not in img_gen_daily_limit:
        user_data = await asyncio.to_thread(db_storage.load_user_data, user_id)
        if user_data:
            img_gen_rate_limit[user_id] = user_data["img_gen_rate_limit_timestamps"]
            img_gen_daily_limit[user_id] = {
                "count": user_data["img_gen_daily_count"],
                "date": user_data["img_gen_daily_date"],
            }

    # Clean old timestamps (older than 60 seconds)
    img_gen_rate_limit[user_id] = [t for t in img_gen_rate_limit[user_id] if current_time - t < 60]

    if len(img_gen_rate_limit[user_id]) >= IMG_GEN_RPM_LIMIT:
        debug("Image gen RPM limit hit for user %s", user_id)
        await update.message.reply_text(
            (
                "Вибачте, забагато запитів на генерацію зображень. Почекайте хвилину."
                if language == "uk"
                else "Sorry, too many image generation requests. Please wait a minute."
            ),
            reply_to_message_id=update.message.message_id,
        )
        return

    # Check daily image gen limit
    today = datetime.now().strftime("%Y-%m-%d")
    if img_gen_daily_limit[user_id]["date"] != today:
        img_gen_daily_limit[user_id] = {"count": 0, "date": today}

    if img_gen_daily_limit[user_id]["count"] >= IMG_GEN_RPD_LIMIT:
        debug("Image gen RPD limit hit for user %s", user_id)
        await update.message.reply_text(
            (
                "Вибачте, денний ліміт генерації зображень вичерпано. Спробуйте завтра."
                if language == "uk"
                else "Sorry, daily image generation limit reached. Try again tomorrow."
            ),
            reply_to_message_id=update.message.message_id,
        )
        return

    # Tentatively add current request timestamp (will be removed on failure)
    img_gen_rate_limit[user_id].append(current_time)

    prompt = prompt[:MAX_PROMPT_LEN].strip()

    try:
        image_response = await asyncio.wait_for(
            asyncio.to_thread(
                xai_client.image.sample,
                prompt=prompt,
                model=GROK_IMG_MODEL,
            ),
            timeout=IMAGE_TIMEOUT_SEC,
        )

        image_url = getattr(image_response, "url", None)
        image_b64 = getattr(image_response, "image", None) if not image_url else None

        if not image_url and not image_b64:
            raise ValueError("Не вдалося отримати результат з xAI API")

        if image_url:
            await update.message.reply_photo(photo=image_url, caption=get_image_caption())
        else:
            file_bytes = base64.b64decode(image_b64)
            await update.message.reply_photo(photo=file_bytes, caption=get_image_caption())

        # Increment daily limit only after successful generation
        img_gen_daily_limit[user_id]["count"] += 1

        # Save img_gen rate limit data to DB (best-effort, targeted update only)
        async def save_img_gen_to_db():
            try:
                await asyncio.to_thread(
                    db_storage.update_user_image_limits,
                    user_id,
                    img_gen_rate_limit[user_id],
                    img_gen_daily_limit[user_id]["count"],
                    img_gen_daily_limit[user_id]["date"],
                )
            except Exception as db_error:  # pylint: disable=broad-except
                error("Failed to save img_gen data to database: %s", db_error)

        asyncio.create_task(save_img_gen_to_db())

    except asyncio.TimeoutError:
        error("Image generation timed out for prompt: %.100s", prompt)
        # Remove tentative timestamp on failure
        if img_gen_rate_limit[user_id] and img_gen_rate_limit[user_id][-1] == current_time:
            img_gen_rate_limit[user_id].pop()
        await update.message.reply_text(
            (
                "Генерація зайняла надто багато часу. Спробуйте пізніше."
                if language == "uk"
                else "Image generation took too long. Please try again later."
            ),
            reply_to_message_id=update.message.message_id,
        )
    except Exception as e:  # pylint: disable=broad-except
        error("Image generation failed: %s", e)
        # Remove tentative timestamp on failure
        if img_gen_rate_limit[user_id] and img_gen_rate_limit[user_id][-1] == current_time:
            img_gen_rate_limit[user_id].pop()
        await update.message.reply_text(
            (
                "Вибачте, не вдалося згенерувати зображення. Спробуйте пізніше."
                if language == "uk"
                else "Sorry, I couldn't generate the image. Please try again later."
            ),
            reply_to_message_id=update.message.message_id,
        )


def clean_url(message_text: str) -> str:
    """
    Cleans the URL from the message text by removing unwanted characters and usernames.

    Args:
        message_text (str): The text of the message containing the URL.

    Returns:
        str: The cleaned URL.
    """
    # Remove markdown formatting
    url = message_text.replace("**", "")

    # Split by space and take the first part (the URL)
    url = url.split()[0]

    # Remove any @username from the end of the URL only if it's an Instagram URL
    if "instagram.com" in url:
        url = url.split('@')[0]

    return url


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
    throttle = False
    video_path = None
    if not update.message or not update.message.text:
        return

    debug("Received a new message: %s", update.message.text)

    message_text = update.message.text.strip()
    # Check if user is not allowed
    if is_user_or_chat_not_allowed(update.effective_user.username, update.effective_chat.id):
        await inform_user_not_allowed(update)
        return

    # Handle bot mention response
    bot_mentioned = is_bot_mentioned(message_text)
    debug("Bot mentioned check: %s for message: %s", bot_mentioned, message_text)
    debug("USE_LLM setting: %s", USE_LLM)
    debug("LLM_PROVIDER: %s", LLM_PROVIDER)

    if bot_mentioned:
        cleaned_text = message_text.strip().lower()

        # Health check always takes priority, even with LLM enabled
        if cleaned_text.startswith("bot_health"):
            # Check if it's a pure health check command (no additional parameters like 'image:')
            if "image:" not in cleaned_text:
                debug("Health check command detected")
                await respond_with_bot_message(update)
                return

        image_prompt = extract_image_prompt(message_text)
        if image_prompt:
            debug("Bot image command detected with prompt: %s", image_prompt)
            await generate_image_and_send(update, image_prompt)
            return

        if USE_LLM:
            debug("Calling LLM response function")
            await respond_with_llm_message(update)
        else:
            debug("Calling regular bot response function")
            await respond_with_bot_message(update)
        return

    # Ignore if message doesn't contain http
    if "http" not in message_text:
        return

    message_text = message_text.replace("** ", "**")

    # Check if URL is from a supported site. Ignore if it's from a group or channel
    if not any(site in message_text for site in supported_sites):
        if update.effective_chat.type == "private":
            not_supported_responses = {
                "uk": "Цей сайт не підтримується. Спробуйте додати ** перед https://",
                "en": "This site is not supported. Try adding ** before the https://",
            }
            await update.message.reply_text(
                not_supported_responses[language],
                reply_to_message_id=update.message.message_id,
            )
            return  # Stop further execution after sending the reply
        return

    debug("Cleaning URL from message text.")
    url = clean_url(message_text)
    debug("Cleaned URL: %s", url)

    if is_video_too_long_to_download(url):
        debug("Video is too long to process.")
        await update.message.reply_text(
            "The video is too long to send (over 12 minutes).",
            reply_to_message_id=update.message.message_id,
        )
        return
    debug("Video is not too long or metadata is not available. Starting download.")

    try:
        # media_path can be string or list of strings
        media_path = []  # Initilize empty list of media paths
        video_path = []  # Initilize empty list of video paths
        pic_path = []  # Initilize empty list of picture paths
        return_path = download_media(url)
        # Create a list of media paths
        if isinstance(return_path, list):
            media_path.extend(return_path)
        else:
            media_path.append(return_path)

        # Ensure that not more than 10 media files are processed
        if len(media_path) > 10:
            debug("Too many media files to process, enabling throttle. Amount: %s", len(media_path))
            throttle = True

        for pathobj in media_path:

            # Create a lists of video and picture paths
            if pathobj.endswith(".mp4"):
                # do not process if video is too long
                if is_video_duration_over_limits(pathobj):
                    await update.message.reply_text(
                        "The video is too long to send (over 12min).",
                        reply_to_message_id=update.message.message_id,
                    )
                    continue  # Drop the video and continue to the next one
                # Compress the video if it's too large
                if is_large_file(pathobj):
                    compress_video(pathobj)
                    if is_large_file(pathobj):
                        await update.message.reply_text(
                            "The video is too large to send (over 50MB).",
                            reply_to_message_id=update.message.message_id,
                        )
                        continue  # Stop further execution for this video
                video_path.append(pathobj)

            elif pathobj.endswith((".jpg", ".jpeg", ".png")):
                pic_path.append(pathobj)

        if len(video_path) > 1:
            # Group videos
            video_path = [video_path[i : i + 2] for i in range(0, len(video_path), 2)]
            # TODO: Implement a total size calculation for the group of videos  # pylint: disable=fixme
            # and resort to sending them one by one if the total size exceeds the limit

        if len(pic_path) > 1:
            # Group pictures
            pic_path = [pic_path[i : i + 10] for i in range(0, len(pic_path), 10)]
            debug("Grouped pictures length: %s", len(pic_path))

        for video in video_path:

            if isinstance(video, str):  # Skip spoiler check for media groups
                # Check for spoiler flag
                has_spoiler = spoiler_in_message(update.message.entities)
            else:
                has_spoiler = False

            # Send the video to the chat
            await send_video(update, video, has_spoiler)
            # wait 5 seconds before sending the next media throttle is enabled
            if throttle:
                await asyncio.sleep(15)

        for pic in pic_path:
            # Send the picture to the chat
            await send_pic(update, pic)
            # wait 5 seconds before sending the next video if throttle is enabled
            if throttle:
                await asyncio.sleep(15)

    finally:
        if media_path:
            cleanup(media_path)


async def respond_with_bot_message(update: Update) -> None:
    """
    Responds to the user with a random bot response when the bot is mentioned.

    Args:
        update (telegram.Update): Represents the incoming update from the Telegram bot.

    Returns:
        None
    """
    response_message = random.choice(responses)  # Select a random response from the predefined list
    info(" requested [Chat ID]: %s by the user %s", update.effective_chat.id, update.effective_user.username)

    if send_user_info_with_healthcheck:
        response_message += f"\n[Chat ID]: {update.effective_chat.id}\n[Username]: {update.effective_user.username}"
    await update.message.reply_text(
        f"{response_message}",
        reply_to_message_id=update.message.message_id,
    )


async def send_video(update: Update, video, has_spoiler: bool) -> None:
    """
    Sends the video to the chat.

    Args:
        update (telegram.Update): Represents the incoming update from the Telegram bot.
        video (str or list): The path to the video file to send. If a list is provided,
            the videos will be sent as a media group with up to 2 videos per group.
        has_spoiler (bool): Indicates if the message contains a spoiler.

    Returns:
        None
    """
    # Send the single video
    if isinstance(video, str):
        width, height = get_video_dimensions(video)
        try:
            with open(video, 'rb') as video_file:
                await update.message.chat.send_video(
                    video=video_file,
                    width=width,
                    height=height,
                    has_spoiler=has_spoiler,
                    disable_notification=True,
                    connect_timeout=TELEGRAM_CONNECT_TIMEOUT,
                    write_timeout=TELEGRAM_WRITE_TIMEOUT,
                    read_timeout=TELEGRAM_READ_TIMEOUT,
                    reply_to_message_id=update.message.message_id,
                )
        except TimedOut as e:
            error("Telegram timeout while sending video. %s", e)
        except (NetworkError, TelegramError) as e:
            await update.message.reply_text(
                f"Error sending video: {str(e)}. Please try again later.",
                reply_to_message_id=update.message.message_id,
            )
        finally:
            video_file.close()

    # Send the group of videos
    elif isinstance(video, list):
        media_group = []  # Initilize empty list of media groups
        opened_files = []  # Initilize empty list of opened files
        for video_file in video:
            file = open(video_file, 'rb')
            opened_files.append(file)
            width, height = get_video_dimensions(video_file)
            media_group.append(InputMediaVideo(file, width=width, height=height))
        debug("Sending a group with number of videos: %s", len(media_group))
        try:
            await update.message.chat.send_media_group(
                media=media_group,
                disable_notification=True,
                connect_timeout=TELEGRAM_CONNECT_TIMEOUT,
                write_timeout=TELEGRAM_WRITE_TIMEOUT,
                read_timeout=TELEGRAM_READ_TIMEOUT,
            )
        except TimedOut as e:
            error("Telegram timeout while sending group of videos. %s", e)
        except (NetworkError, TelegramError) as e:
            await update.message.reply_text(
                f"Error sending group of videos: {str(e)}. Please try again later.",
                reply_to_message_id=update.message.message_id,
            )
        finally:
            for file in opened_files:
                file.close()


async def send_pic(update: Update, pic) -> None:
    """
    Sends the picture to the chat.
    Args:
        update (telegram.Update): Represents the incoming update from the Telegram bot.
        pic (str or list): The path to the picture file to send. If a list is provided,
            the pictures will be sent as a media group with up to 10 pictures per group.
    Returns:
        None
    """
    if isinstance(pic, str):
        # Send the single picture
        try:
            with open(pic, 'rb') as pic_file:
                await update.message.chat.send_photo(
                    photo=pic_file,
                    disable_notification=True,
                    connect_timeout=TELEGRAM_CONNECT_TIMEOUT,
                    write_timeout=TELEGRAM_WRITE_TIMEOUT,
                    read_timeout=TELEGRAM_READ_TIMEOUT,
                )
        except TimedOut as e:
            error("Telegram timeout while sending picture. %s", e)
        except (NetworkError, TelegramError) as e:
            await update.message.reply_text(
                f"Error sending picture: {str(e)}. Please try again later.",
                reply_to_message_id=update.message.message_id,
            )
        finally:
            pic_file.close()

    elif isinstance(pic, list):
        media_group = []  # Initilize empty list of media groups
        opened_files = []  # Initilize empty list of opened files
        for pic_file in pic:
            file = open(pic_file, 'rb')
            opened_files.append(file)
            media_group.append(InputMediaPhoto(file))
        debug("Sending a group with number of pictures: %s", len(media_group))
        # Send the media group
        try:
            await update.message.chat.send_media_group(
                media=media_group,
                disable_notification=True,
                connect_timeout=TELEGRAM_CONNECT_TIMEOUT,
                write_timeout=TELEGRAM_WRITE_TIMEOUT,
                read_timeout=TELEGRAM_READ_TIMEOUT,
            )
        except TimedOut as e:
            error("Telegram timeout while sending group of pictures. %s", e)
        except (NetworkError, TelegramError) as e:
            await update.message.reply_text(
                f"Error sending group of pictures: {str(e)}. Please try again later.",
                reply_to_message_id=update.message.message_id,
            )
        finally:
            for file in opened_files:
                file.close()


async def respond_with_llm_message(update):
    """Handle LLM responses when bot is mentioned using Gemini or Grok API."""
    debug("LLM response function called")
    message_text = update.message.text
    # Remove bot mention and any punctuation after it
    prompt = re.sub(r'ботяра[^\w\s]*', '', message_text.lower()).strip()
    debug("Original message: %s", message_text)
    debug("Processed prompt: %s", prompt)

    # Validate LLM provider
    if LLM_PROVIDER not in ALLOWED_PROVIDERS:
        bot_response = (
            f"Вибачте, провайдер '{LLM_PROVIDER}' не підтримується. Доступні: {', '.join(ALLOWED_PROVIDERS)}"
            if language == "uk"
            else f"Sorry, provider '{LLM_PROVIDER}' is not supported. Available: {', '.join(ALLOWED_PROVIDERS)}"
        )
        await update.message.reply_text(bot_response)
        return

    # Check if API is configured
    if LLM_PROVIDER == "grok" and not GROK_API_KEY:
        bot_response = (
            "Вибачте, Grok AI сервіс не налаштовано."
            if language == "uk"
            else "Sorry, Grok AI service is not configured."
        )
        await update.message.reply_text(bot_response)
        return
    elif LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        bot_response = (
            "Вибачте, Gemini AI сервіс не налаштовано."
            if language == "uk"
            else "Sorry, Gemini AI service is not configured."
        )
        await update.message.reply_text(bot_response)
        return

    # Rate limiting check
    user_id = update.effective_user.id
    current_time = time.time()

    # Update last seen timestamp
    user_last_seen[user_id] = current_time

    # Load user data from database on first access
    if user_id not in llm_daily_limit:
        debug("Loading user data from database for user_id: %s", user_id)
        user_data = await asyncio.to_thread(db_storage.load_user_data, user_id)
        if user_data:
            debug(
                "Found user data in database: context=%d messages, rate_limit=%d timestamps, daily=%d/%s",
                len(user_data["conversation_context"]),
                len(user_data["rate_limit_timestamps"]),
                user_data["daily_count"],
                user_data["daily_date"],
            )
            conversation_context[user_id] = user_data["conversation_context"]
            llm_rate_limit[user_id] = user_data["rate_limit_timestamps"]
            llm_daily_limit[user_id] = {"count": user_data["daily_count"], "date": user_data["daily_date"]}
            # Only update last_seen if DB value is newer
            if user_id not in user_last_seen or user_data["last_seen"] > user_last_seen[user_id]:
                user_last_seen[user_id] = user_data["last_seen"]
        else:
            debug("No existing data found in database for user_id: %s", user_id)

    # Clean old timestamps (older than 60 seconds)
    llm_rate_limit[user_id] = [t for t in llm_rate_limit[user_id] if current_time - t < 60]

    if len(llm_rate_limit[user_id]) >= LLM_RPM_LIMIT:
        debug("Rate limit hit for user %s", user_id)
        bot_response = (
            "Вибачте, забагато запитів. Почекайте хвилину."
            if language == "uk"
            else "Sorry, too many requests. Please wait a minute."
        )
        await update.message.reply_text(bot_response)
        return

    # Check daily limit
    today = datetime.now().strftime("%Y-%m-%d")
    if llm_daily_limit[user_id]["date"] != today:
        llm_daily_limit[user_id] = {"count": 0, "date": today}

    if llm_daily_limit[user_id]["count"] >= LLM_RPD_LIMIT:
        debug("Daily limit hit for user %s", user_id)
        bot_response = (
            "Вибачте, денний ліміт запитів вичерпано. Спробуйте завтра."
            if language == "uk"
            else "Sorry, daily request limit reached. Try again tomorrow."
        )
        await update.message.reply_text(bot_response)
        return

    # Tentatively add current request timestamp (will be removed on failure)
    llm_rate_limit[user_id].append(current_time)

    try:
        # Check if user is asking for image generation and modify prompt
        image_keywords = [
            # 'картинку',
            # 'картинка',
            # 'зображення',
            # 'image',
            # 'фото',
            # 'picture',
            # 'згенеруй',
            # 'generate',
            # 'створи',
            # 'create',
            # 'покажи',
            # 'покажи мне',
            # 'покажи мені',
        ]
        # Check both original message and processed prompt
        original_text = message_text.lower()
        if any(word in original_text for word in image_keywords) or any(
            word in prompt.lower() for word in image_keywords
        ):
            # Directly respond to image requests without calling Gemini
            debug("Image generation request detected, sending direct response")
            if language == "uk":
                bot_response = "Вибачте, я не можу генерувати зображення, але можу детально описати те, що ви просите! Наприклад, я можу розповісти про машину: її колір, форму, особливості дизайну тощо. Що саме вас цікавить?"
            else:
                bot_response = "Sorry, I can't generate images, but I can describe in detail what you're asking for! For example, I can tell you about a car: its color, shape, design features, etc. What specifically interests you?"

            await update.message.reply_text(bot_response)
            # Remove tentative timestamp since no API call was made
            llm_rate_limit[user_id].pop()
            return

        # Prepare prompt with context
        debug("Original prompt: %s", prompt)

        # Build context from previous messages if enabled
        if USE_CONVERSATION_CONTEXT:
            context_messages = (
                conversation_context[user_id][-MAX_CONTEXT_MESSAGES:] if conversation_context[user_id] else []
            )
        else:
            context_messages = []

        # Create prompt with context if available
        if language == "uk":
            user_label = "Користувач"
            assistant_label = "Асистент"
            instruction = "Відповідай українською мовою як дружній асистент. Не вітайся і не прощайся."
        else:
            user_label = "User"
            assistant_label = "Assistant"
            instruction = "Answer in English as a friendly assistant. Don't greet or say goodbye."

        if context_messages:
            context_str = "\n".join(
                [f"{user_label}: {msg}\n{assistant_label}: {resp}" for msg, resp in context_messages]
            )
            if language == "uk":
                safe_prompt = (
                    f"Попередня розмова:\n{context_str}\n\nПоточне питання користувача: {prompt}\n\n{instruction}"
                )
            else:
                safe_prompt = (
                    f"Previous conversation:\n{context_str}\n\nCurrent user question: {prompt}\n\n{instruction}"
                )
        else:
            if language == "uk":
                safe_prompt = f"{instruction} Питання користувача: {prompt}"
            else:
                safe_prompt = f"{instruction} User question: {prompt}"

        debug("Modified safe prompt with context: %s", safe_prompt[:200])

        # Call appropriate LLM provider
        if LLM_PROVIDER == "grok":
            debug("Using Grok API with model: %s", GROK_MODEL)
            bot_response = await call_grok_api(safe_prompt, update)
        else:
            debug("Using Gemini API with model: %s", GEMINI_MODEL)
            bot_response = await call_gemini_api(safe_prompt, prompt, update)

        # Increment daily limit only after successful API call
        llm_daily_limit[user_id]["count"] += 1

        # Store conversation in context if enabled
        if USE_CONVERSATION_CONTEXT:
            truncated_prompt = prompt[:MAX_CONTEXT_CHARS]
            truncated_response = bot_response[:MAX_CONTEXT_CHARS]
            conversation_context[user_id].append((truncated_prompt, truncated_response))
            # Keep only last MAX_CONTEXT_MESSAGES
            if len(conversation_context[user_id]) > MAX_CONTEXT_MESSAGES:
                conversation_context[user_id] = conversation_context[user_id][-MAX_CONTEXT_MESSAGES:]

        # Send reply first, then save to DB (best-effort persistence)
        await update.message.reply_text(bot_response)

        # Save user data to database (best-effort, don't fail on DB errors)
        async def save_to_db():
            try:
                debug(
                    "Saving user data to database: user_id=%s, context=%d messages, daily=%d/%s",
                    user_id,
                    len(conversation_context[user_id]),
                    llm_daily_limit[user_id]["count"],
                    llm_daily_limit[user_id]["date"],
                )

                # Build save arguments, only including image gen data if explicitly set for this user
                save_kwargs = {
                    "user_id": user_id,
                    "conversation_context": conversation_context[user_id],
                    "rate_limit_timestamps": llm_rate_limit[user_id],
                    "daily_count": llm_daily_limit[user_id]["count"],
                    "daily_date": llm_daily_limit[user_id]["date"],
                    "last_seen": user_last_seen[user_id],
                }

                # Only include image gen data if user has actually interacted with image generation
                if user_id in img_gen_rate_limit:
                    save_kwargs["img_gen_rate_limit_timestamps"] = img_gen_rate_limit[user_id]
                if user_id in img_gen_daily_limit:
                    save_kwargs["img_gen_daily_count"] = img_gen_daily_limit[user_id]["count"]
                    save_kwargs["img_gen_daily_date"] = img_gen_daily_limit[user_id]["date"]

                await asyncio.to_thread(db_storage.save_user_data, **save_kwargs)
            except Exception as db_error:  # pylint: disable=broad-except
                error("Failed to save user data to database: %s", db_error)

        asyncio.create_task(save_to_db())

    except Exception as e:  # pylint: disable=broad-except
        # Remove tentative timestamp on failure
        if llm_rate_limit[user_id] and llm_rate_limit[user_id][-1] == current_time:
            llm_rate_limit[user_id].pop()

        error_msg = str(e)
        error("Error in LLM API request: %s (Type: %s)", error_msg, type(e).__name__)
        error("Full traceback: %s", traceback.format_exc())

        # Check for rate limit (429) error
        if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            error("Rate limit exceeded (429) - Too many requests to LLM API")
            bot_response = (
                "Вибачте, перевищено ліміт запитів до AI. Спробуйте пізніше."
                if language == "uk"
                else "Sorry, AI request limit exceeded. Please try again later."
            )
        else:
            bot_response = (
                "Вибачте, я не можу згенерувати відповідь."
                if language == "uk"
                else "Sorry, I encountered an error while processing your request."
            )

        await update.message.reply_text(bot_response)


async def call_grok_api(safe_prompt: str, update) -> str:
    """Call Grok API and return response. Raises exception on failure."""
    plain_text_instruction = "Provide the entire response exclusively as plain text. Do not use any Markdown formatting (no **bold**, *italics*, # headers, or lists). The response must be text only. Provide concise, short answers. Aim for 1-3 sentences."
    max_retries = 2
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            response = await grok_client.chat.completions.create(
                model=GROK_MODEL,
                messages=[
                    {"role": "system", "content": plain_text_instruction},
                    {"role": "user", "content": safe_prompt},
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as retry_error:  # pylint: disable=broad-exception-caught
            error_msg = str(retry_error)
            if (
                "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower()
            ) and attempt < max_retries - 1:
                debug(
                    "Rate limit hit, waiting %s seconds before retry (attempt %s/%s)",
                    retry_delay,
                    attempt + 1,
                    max_retries,
                )
                wait_msg = (
                    f"Перевищено ліміт запитів. Зачекайте {retry_delay} секунд, я спробую ще раз..."
                    if language == "uk"
                    else f"Rate limit exceeded. Waiting {retry_delay} seconds before retrying..."
                )
                await update.message.reply_text(wait_msg)
                await asyncio.sleep(retry_delay)
            else:
                raise


async def call_gemini_api(safe_prompt: str, prompt: str, update) -> str:
    """Call Gemini API and return response. Raises exception on failure."""
    plain_text_instruction = "Provide the entire response exclusively as plain text. Do not use any Markdown formatting (no **bold**, *italics*, # headers, or lists). The response must be text only. Provide concise, short answers. Aim for 1-3 sentences."
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=plain_text_instruction)
    safety_settings = {
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    }
    contents = [{'role': 'user', 'parts': [safe_prompt]}]

    max_retries = 2
    retry_delay = 60
    response = None

    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                contents,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=30,
                    max_output_tokens=1024,
                ),
                safety_settings=safety_settings,
            )
            debug("Successfully received response from Gemini API")
            break
        except Exception as retry_error:  # pylint: disable=broad-exception-caught
            error_msg = str(retry_error)
            if (
                "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower()
            ) and attempt < max_retries - 1:
                debug(
                    "Rate limit hit, waiting %s seconds before retry (attempt %s/%s)",
                    retry_delay,
                    attempt + 1,
                    max_retries,
                )
                wait_msg = (
                    f"Перевищено ліміт запитів. Зачекайте {retry_delay} секунд, я спробую ще раз..."
                    if language == "uk"
                    else f"Rate limit exceeded. Waiting {retry_delay} seconds before retrying..."
                )
                await update.message.reply_text(wait_msg)
                await asyncio.sleep(retry_delay)
            else:
                raise

    # Check if response was set after retries
    if response is None:
        raise Exception("Failed to get response after retries")  # pylint: disable=broad-exception-raised

    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        debug("Response candidate finish_reason: %s", getattr(candidate, 'finish_reason', 'None'))
        debug("Response candidate safety_ratings: %s", getattr(candidate, 'safety_ratings', 'None'))

        if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
            debug("Safety filter triggered - finish_reason: 2, trying simpler approach")
            fallback_instruction = (
                "Відповідь українською мовою: дай загальну інформацію про: "
                if language == "uk"
                else "Answer in English: give general information about: "
            )
            simple_response = await asyncio.to_thread(
                model.generate_content,
                fallback_instruction + prompt,
                safety_settings=safety_settings,
            )
            if simple_response.text:
                prefix = "Ось загальна інформація: " if language == "uk" else "Here's general information: "
                return f"{prefix}{simple_response.text.strip()}"
            else:
                error_msg = (
                    "Вибачте, не можу надати детальну відповідь на це питання."
                    if language == "uk"
                    else "Sorry, I can't provide a detailed answer to this question."
                )
                raise Exception(error_msg)  # pylint: disable=broad-exception-raised
        elif response.text:
            # Remove Markdown formatting
            bot_response = response.text.strip()
            bot_response = re.sub(r'\*+', '', bot_response)
            bot_response = bot_response.replace('*', '').replace('`', '').replace('#', '')
            return bot_response
        else:
            raise Exception("Вибачте, я не можу згенерувати відповідь.")  # pylint: disable=broad-exception-raised
    else:
        raise Exception("Вибачте, я не можу згенерувати відповідь.")  # pylint: disable=broad-exception-raised


async def cleanup_stale_users():
    """Remove inactive users from memory and database to prevent unbounded growth."""
    while True:
        try:
            await asyncio.sleep(USER_CLEANUP_INTERVAL_HOURS * 3600)
            ttl_seconds = USER_CLEANUP_TTL_DAYS * 86400

            # Get stale users from database
            stale_users = await asyncio.to_thread(db_storage.get_stale_users, ttl_seconds)

            for user_id in stale_users:
                # Remove from memory
                if user_id in conversation_context:
                    del conversation_context[user_id]
                if user_id in llm_rate_limit:
                    del llm_rate_limit[user_id]
                if user_id in llm_daily_limit:
                    del llm_daily_limit[user_id]
                if user_id in img_gen_rate_limit:
                    del img_gen_rate_limit[user_id]
                if user_id in img_gen_daily_limit:
                    del img_gen_daily_limit[user_id]
                if user_id in user_last_seen:
                    del user_last_seen[user_id]
                # Remove from database
                await asyncio.to_thread(db_storage.delete_user_data, user_id)

            if stale_users:
                info("Cleaned up %d inactive users (TTL: %d days)", len(stale_users), USER_CLEANUP_TTL_DAYS)
        except Exception as cleanup_error:  # pylint: disable=broad-except
            error("Error in cleanup_stale_users: %s", cleanup_error)
            error("Full traceback: %s", traceback.format_exc())
            await asyncio.sleep(60)  # Wait before retrying


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
    request = HTTPXRequest(
        connect_timeout=TELEGRAM_CONNECT_TIMEOUT,
        pool_timeout=TELEGRAM_POOL_TIMEOUT,
        read_timeout=TELEGRAM_READ_TIMEOUT,
        write_timeout=TELEGRAM_WRITE_TIMEOUT,
    )
    application = Application.builder().token(bot_token).request(request).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # This handler will receive every error which happens in your bot
    application.add_error_handler(error_handler)

    # Start cleanup task after event loop is running
    async def post_init(app):  # pylint: disable=unused-argument
        global cleanup_task  # pylint: disable=global-statement
        cleanup_task = asyncio.create_task(cleanup_stale_users())

    # Cancel cleanup task and close DB on shutdown
    async def post_shutdown(app):  # pylint: disable=unused-argument
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        # Close database connection
        try:
            db_storage.close()
            debug("Database connection closed")
        except Exception as e:  # pylint: disable=broad-except
            error("Error closing database: %s", e)

    application.post_init = post_init
    application.post_shutdown = post_shutdown

    info("Bot started. Ctrl+C to stop")
    application.run_polling()


if __name__ == "__main__":
    main()
