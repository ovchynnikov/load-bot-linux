"""Download route: supported-URL detection, media download/compression and Telegram sending."""

import asyncio
import os
from contextlib import ExitStack
from telegram import Update, InputMediaPhoto, InputMediaVideo
from telegram.error import TimedOut, NetworkError, TelegramError
from telegram.constants import MessageEntityType
from logger import error, debug
from i18n import get_language
from permissions import supported_sites
from cleanup import cleanup
from safe_diagnostics import exception_summary
from video_utils import (
    compress_video,
    download_media,
    get_video_dimensions,
    is_video_duration_over_limits,
    is_video_too_long_to_download,
)

language = get_language()

TELEGRAM_CONNECT_TIMEOUT = 60
TELEGRAM_POOL_TIMEOUT = 30
TELEGRAM_READ_TIMEOUT = 120
TELEGRAM_WRITE_TIMEOUT = 120


def localized(uk: str, en: str) -> str:
    """Pick the reply text for the configured language."""
    return uk if language == "uk" else en


async def run_blocking(function, *args, discard=None):
    """Run blocking work in a thread that the caller owns until it finishes.

    Cancelling the caller - even repeatedly - does not abandon the worker: we keep waiting
    until the thread returns, hand a finished result to `discard` (e.g. cleanup of a
    downloaded temp dir), then re-raise. This keeps the handler's `finally` cleanup from
    deleting files a worker is still writing.
    """
    task = asyncio.ensure_future(asyncio.to_thread(function, *args))
    cancellation = None
    while not task.done():
        try:
            # asyncio.wait never cancels the task it waits for, unlike awaiting the task directly.
            await asyncio.wait({task})
        except asyncio.CancelledError as exc:
            cancellation = exc  # keep draining; later cancellations are folded into this one
    if cancellation is not None:
        if discard is not None and not task.cancelled() and task.exception() is None:
            discard(task.result())
        raise cancellation
    return task.result()


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


async def handle_download(update: Update) -> None:
    """Download media from a supported URL in the message and send it back to the chat."""
    throttle = False
    video_path = None
    message_text = update.message.text.strip()

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

    if await run_blocking(is_video_too_long_to_download, url):
        debug("Video is too long to process.")
        await update.message.reply_text(
            localized(
                "Відео задовге для надсилання (понад 12 хвилин).", "The video is too long to send (over 12 minutes)."
            ),
            reply_to_message_id=update.message.message_id,
        )
        return
    debug("Video is not too long or metadata is not available. Starting download.")

    try:
        # media_path can be string or list of strings
        media_path = []  # Initilize empty list of media paths
        video_path = []  # Initilize empty list of video paths
        pic_path = []  # Initilize empty list of picture paths
        return_path = await run_blocking(download_media, url, discard=cleanup)
        if not return_path:
            await update.message.reply_text(
                (
                    "Не вдалося завантажити медіа з цього посилання 😔"
                    if language == "uk"
                    else "Couldn't download media from this link 😔"
                ),
                reply_to_message_id=update.message.message_id,
            )
            return
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
                if await run_blocking(is_video_duration_over_limits, pathobj):
                    await update.message.reply_text(
                        localized(
                            "Відео задовге для надсилання (понад 12 хвилин).",
                            "The video is too long to send (over 12 minutes).",
                        ),
                        reply_to_message_id=update.message.message_id,
                    )
                    continue  # Drop the video and continue to the next one
                # Compress the video if it's too large
                if is_large_file(pathobj):
                    await run_blocking(compress_video, pathobj)
                    if is_large_file(pathobj):
                        await update.message.reply_text(
                            localized(
                                "Відео завелике для надсилання (понад 50 МБ).",
                                "The video is too large to send (over 50MB).",
                            ),
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
        width, height = await run_blocking(get_video_dimensions, video)
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
            error("Telegram timeout while sending video. %s", exception_summary(e))
        except (NetworkError, TelegramError) as e:
            error("Telegram error while sending video. %s", exception_summary(e))
            await update.message.reply_text(
                localized(
                    "Не вдалося надіслати відео. Спробуйте пізніше.", "Couldn't send the video. Please try again later."
                ),
                reply_to_message_id=update.message.message_id,
            )

    # Send the group of videos
    elif isinstance(video, list):
        # ExitStack closes every opened file even if a later open or probe fails.
        with ExitStack() as files:
            media_group = []  # Initilize empty list of media groups
            for video_file in video:
                file = files.enter_context(open(video_file, 'rb'))
                width, height = await run_blocking(get_video_dimensions, video_file)
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
                error("Telegram timeout while sending group of videos. %s", exception_summary(e))
            except (NetworkError, TelegramError) as e:
                error("Telegram error while sending group of videos. %s", exception_summary(e))
                await update.message.reply_text(
                    localized(
                        "Не вдалося надіслати відео. Спробуйте пізніше.",
                        "Couldn't send the videos. Please try again later.",
                    ),
                    reply_to_message_id=update.message.message_id,
                )


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
            error("Telegram timeout while sending picture. %s", exception_summary(e))
        except (NetworkError, TelegramError) as e:
            error("Telegram error while sending picture. %s", exception_summary(e))
            await update.message.reply_text(
                localized(
                    "Не вдалося надіслати зображення. Спробуйте пізніше.",
                    "Couldn't send the picture. Please try again later.",
                ),
                reply_to_message_id=update.message.message_id,
            )

    elif isinstance(pic, list):
        # ExitStack closes every opened file even if a later open fails.
        with ExitStack() as files:
            media_group = [InputMediaPhoto(files.enter_context(open(pic_file, 'rb'))) for pic_file in pic]
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
                error("Telegram timeout while sending group of pictures. %s", exception_summary(e))
            except (NetworkError, TelegramError) as e:
                error("Telegram error while sending group of pictures. %s", exception_summary(e))
                await update.message.reply_text(
                    localized(
                        "Не вдалося надіслати зображення. Спробуйте пізніше.",
                        "Couldn't send the pictures. Please try again later.",
                    ),
                    reply_to_message_id=update.message.message_id,
                )
