"""Download videos from tiktok, x(twitter), reddit, youtube shorts, instagram reels and many more"""

import os
import random
import json
import asyncio
import re
import google.generativeai as genai
from functools import lru_cache
from dotenv import load_dotenv
from telegram import Update, InputMediaPhoto, InputMediaVideo
from telegram.error import TimedOut, NetworkError, TelegramError
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram.constants import MessageEntityType
from logger import error, info, debug
from general_error_handler import error_handler
from permissions import inform_user_not_allowed, is_user_or_chat_not_allowed, supported_sites
from cleanup import cleanup
from video_utils import (
    compress_video,
    download_media,
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TELEGRAM_WRITE_TIMEOUT = 8000
TELEGRAM_READ_TIMEOUT = 8000

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


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
    debug("GEMINI_API_KEY configured: %s", bool(GEMINI_API_KEY))

    if bot_mentioned:
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

        try:
            with open(video, 'rb') as video_file:
                await update.message.chat.send_video(
                    video=video_file,
                    has_spoiler=has_spoiler,
                    disable_notification=True,
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
            media_group.append(InputMediaVideo(file))
        debug("Sending a group with number of videos: %s", len(media_group))
        try:
            await update.message.chat.send_media_group(
                media=media_group,
                disable_notification=True,
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
    """Handle LLM responses when bot is mentioned using Google Gemini API."""
    debug("LLM response function called")
    message_text = update.message.text
    # Remove bot mention and any punctuation after it
    prompt = re.sub(r'ботяра[^\w\s]*', '', message_text.lower()).strip()
    debug("Original message: %s", message_text)
    debug("Processed prompt: %s", prompt)

    if not GEMINI_API_KEY:
        # debug("GEMINI_API_KEY not configured")
        await update.message.reply_text("Sorry, AI service is not configured.")
        return

    try:
        # Check if user is asking for image generation and modify prompt
        image_keywords = [
            'картинку',
            'картинка',
            'зображення',
            'image',
            'фото',
            'picture',
            'згенеруй',
            'generate',
            'створи',
            'create',
            'покажи',
            'покажи мне',
            'покажи мені',
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
            return

        # Initialize the Gemini model
        debug("Initializing Gemini model: gemini-2.5-flash")
        plain_text_instruction = "Provide the entire response exclusively as plain text. Do not use any Markdown formatting (no **bold**, *italics*, # headers, or lists). The response must be text only. Provide concise, short answers. Aim for 1-3 sentences."

        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=plain_text_instruction)

        # Try different approach - rephrase any potentially problematic prompts
        debug("Original prompt: %s", prompt)
        safe_prompt = f"Відповідай українською мовою як дружній асистент. Питання користувача: {prompt}"
        debug("Modified safe prompt: %s", safe_prompt)

        # Generate response using Gemini with both safety settings and safe prompting
        debug("Sending request to Gemini API")
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        contents = [{'role': 'user', 'parts': [safe_prompt]}]
        response = await asyncio.to_thread(
            model.generate_content,
            contents,  # Pass the simplified list here
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=30,
                max_output_tokens=1024,
            ),
            safety_settings=safety_settings,
        )
        # debug("Successfully received response from Gemini API")

        # Handle response with safety filter checks
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            debug("Response candidate finish_reason: %s", getattr(candidate, 'finish_reason', 'None'))
            debug("Response candidate safety_ratings: %s", getattr(candidate, 'safety_ratings', 'None'))

            if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                debug("Safety filter triggered - finish_reason: 2, trying simpler approach")
                # Try a much simpler, generic response for blocked content
                try:
                    simple_response = await asyncio.to_thread(
                        model.generate_content,
                        "Відповідь українською мовою: дай загальну інформацію про: " + prompt,
                        safety_settings=safety_settings,
                    )
                    if simple_response.text:
                        bot_response = f"Можу розповісти загалом: {simple_response.text.strip()}"
                    else:
                        bot_response = (
                            "Вибачте, не можу надати детальну відповідь на це питання."
                            if language == "uk"
                            else "Sorry, I can't provide a detailed answer to this question."
                        )
                except:  # --- IGNORE --- # pylint: disable=bare-except
                    bot_response = (
                        "Вибачте, не можу надати детальну відповідь на це питання."
                        if language == "uk"
                        else "Sorry, I can't provide a detailed answer to this question."
                    )
            elif response.text:
                # Remove Markdown formatting from response
                bot_response = response.text.strip()
                # Remove common Markdown syntax
                bot_response = re.sub(r'\*+', '', bot_response)  # Bold text
                bot_response = bot_response.replace('*', '')  # Italic text
                bot_response = bot_response.replace('`', '')  # Code blocks
                bot_response = bot_response.replace('#', '')  # Headers
            else:
                bot_response = (
                    "Вибачте, я не можу згенерувати відповідь."
                    if language == "uk"
                    else "Sorry, I couldn't generate a response."
                )
        else:
            bot_response = (
                "Вибачте, я не можу згенерувати відповідь."
                if language == "uk"
                else "Sorry, I couldn't generate a response."
            )

        await update.message.reply_text(bot_response)

    except (ValueError, RuntimeError) as e:
        error("Error in Gemini API request: %s", e)
        await update.message.reply_text(
            "Вибачте, я не можу згенерувати відповідь."
            if language == "uk"
            else "Sorry, I encountered an error while processing your request."
        )
    except Exception as e:  # pylint: disable=broad-except
        error("Unexpected error in Gemini API request: %s", e)
        await update.message.reply_text(
            "Вибачте, я не можу згенерувати відповідь."
            if language == "uk"
            else "Sorry, I encountered an unexpected error while processing your request."
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
    # This handler will receive every error which happens in your bot
    application.add_error_handler(error_handler)
    info("Bot started. Ctrl+C to stop")
    application.run_polling()


if __name__ == "__main__":
    main()
