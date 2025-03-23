# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring

import os
import re
import subprocess
import tempfile
from pathlib import Path
import yt_dlp
from dotenv import load_dotenv
from logger import debug, error

load_dotenv()  # Load environment variables from .env file

# Retrieve the INSTACOOKIES environment variable and set a default value
instagram_cookies = os.getenv("INSTACOOKIES", "false").lower() == 'true'
LIBX264 = "libx264"
LIBX265 = "libx265"
# Retrieve the H_CODEC environment variable and set a default value
h_codec = os.getenv("H_CODEC", LIBX265)  # Default lib to compress is libx265 to save space
use_gpu_compressing = os.getenv("USE_GPU_COMPRESSING", "false").lower() == 'true'
debug("Codec configured: %s", h_codec)
debug("Use GPU compressing: %s", use_gpu_compressing)

# Check if INSTACOOKIES is True and the required file exists
if instagram_cookies:
    if not os.path.exists("instagram_cookies.txt"):
        error("INSTACOOKIES is True but 'instagram_cookies.txt' not found.")
        # Set to False if the file is not found
        instagram_cookies = False  # pylint: disable=invalid-name
else:
    debug("INSTACOOKIES is False or cookies file not found")


def get_video_metadata(url):
    """
    Extract metadata from a video URL without downloading the content.

    Args:
        url (str): The URL of the video to analyze

    Returns:
        dict: Video metadata including duration, format, etc., or None if extraction fails

    Raises:
        yt_dlp.utils.ExtractorError: When video information cannot be extracted
    """
    ydl_opts = {
        'noplaylist': True,
        'quiet': True,
    }
    debug("Getting video metadata for: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)  # fetch metadata only
            debug("Video metadata extracted")
            return info_dict
        except (yt_dlp.utils.ExtractorError, yt_dlp.utils.DownloadError) as e:
            debug("Extracting metadata failed: %s", e)
            return None
        except (ValueError, AttributeError) as e:  # Catch specific validation errors
            debug("Invalid video URL or metadata format for %s: %s", url, e)
            return None


def is_video_duration_over_limits(video_path: str, max_duration: int = 720) -> bool:
    """
    Checks if the video file is of a suitable size for compression.

    Args:
        video_path (str): The path to the video file to check.
        max_size_mb (int): The maximum file size in megabytes (default is 50MB).

    Returns:
        bool: True if the video file size is greater than the maximum size, False otherwise.
    """
    duration = get_video_duration(video_path)
    if duration:
        return duration > max_duration
    return False


def is_video_too_long_to_download(url, max_duration_minutes=12):
    """
    Checks if the video duration exceeds the specified maximum duration.

    Args:
        url (str): The URL of the video to check.
        max_duration_minutes (int): The maximum video duration in minutes (default is 12 minutes).

    Returns:
        bool: True if the video duration exceeds the maximum duration, False otherwise.
    """
    debug("Checking if video is too long to download: %s", url)
    metadata = get_video_metadata(url)
    if metadata and 'duration' in metadata:
        debug("Video duration: %s seconds. Max duration is %s seconds", metadata['duration'], max_duration_minutes * 60)
        return metadata['duration'] > (max_duration_minutes * 60)

    debug("No video duration found in metadata for %s", url)
    return False


def compress_video(input_path):
    """
    Compress video for 50MB with use of FFmpeg.

    Parameters:
        input_path (str): Path to original video.
    """
    temp_output = tempfile.mktemp(suffix=".mp4")
    # Caclulation of file size. 40 means MB
    # target_size_bytes = 40 * 1024 * 1024
    duration = get_video_duration(input_path)
    if not duration:
        raise ValueError("Get video duration failed.")

    # bitrate caclulation kb/s (bit/sec -> kb/sec)
    # target_bitrate_kbps = (target_size_bytes * 8) / duration / 1000
    debug("Starting compression for video: %s", input_path)

    command = [
        "nice",
        "-n",
        "19",
        "ffmpeg",
        *(["-hwaccel", "vaapi"] if use_gpu_compressing else []),
        *(["-vaapi_device", "/dev/dri/renderD128"] if use_gpu_compressing else []),
        "-i",
        input_path,
        "-qp",
        "35",
        # *(["-b:v", f"{target_bitrate_kbps}k"] if h_codes == LIBX264 else []),
        *(["-vf", "format=nv12,hwupload,scale=-2:720"] if use_gpu_compressing else ["-vf", "scale=-2:720"]),
        *(
            ["-c:v", "h264_vaapi"]
            if use_gpu_compressing and h_codec == LIBX264
            else (["-c:v", "h265_vaapi"] if use_gpu_compressing and h_codec == LIBX265 else ["-c:v", h_codec])
        ),  # Use hardware GPU acceleration or software codec based on settings
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-y",
        temp_output,
    ]

    try:
        subprocess.run(command, check=True)
        if os.path.exists(temp_output):
            os.replace(temp_output, input_path)
    except subprocess.CalledProcessError as e:
        error("Error while compressing: %s", e)
    debug("Compression completed for video: %s", input_path)


def get_video_duration(video_path):
    """
    Gets video duration in seconds.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        error("Error getting video duration: %s", e)
        return None


def download_instagram_media(url, temp_dir):
    """
    Downloads Instagram media using gallery-dl.

    This function uses the `gallery-dl` command-line tool to download media from Instagram.
    The media is stored in a temporary directory. The function returns the path to the downloaded
    media file if successful.

    Parameters:
        url (str): The URL of the Instagram media to download.
        temp_dir (str): The path to the temporary directory to store the downloaded media.

    Returns:
        str: The path to the downloaded media file if successful, or None if the download fails.
    """
    result_path = None  # Initialize the result variable

    # Validate Instagram URL
    if not re.match(r'^https?://(?:www\.)?instagram\.com/[^/]+/[^/]+/?.*$', url):
        error("Invalid Instagram URL")
        return None

    # Ensure that reels not in the URL
    if "reel" in url:
        error("Reels should be handled by the yt-dlp. Abort.")
        return None

    command = [
        "gallery-dl",  # Assuming gallery-dl is installed and in the PATH
        *(["--cookies", "instagram_cookies.txt"] if instagram_cookies else []),
        url,
        "-d",
        temp_dir,
    ]

    try:
        debug("Running gallery-dl command: %s", command)
        subprocess.run(command, check=True, timeout=120)
        result_path = []  # Initialize the result variable as a empty list
        # Use Path.rglob to recursively search for files in the temp directory
        # as the output contains subdirectories and may contain multiple files
        for file in [str(file) for file in Path(temp_dir).rglob("*")]:
            if file.endswith((".mp4", ".jpg", ".jpeg", ".png")):
                # Append the file path to the result list
                result_path.append(file)
        if not result_path:
            error("No media files found in the gallery-dl output")
            return None

        return result_path  # Return the result variable if successful
    except subprocess.CalledProcessError as e:
        error("Error downloading Instagram media: %s", e)
    except subprocess.TimeoutExpired as e:
        error("Download process timed out: %s", e)
    except (OSError, IOError) as e:
        error("File system error occurred: %s", e)

    return result_path  # Return the result variable at the end


def download_media(url):
    """
    Downloads a video from the specified URL using yt-dlp and saves it as an MP4 file.

    This function uses the `yt-dlp` command-line tool to download a video. The video is stored
    in a temporary directory with a filename based on the video's title. The function
    returns the path to the downloaded video file if successful. If the download fails, "[Instagram]"
    and "No video formats found!" present in the error message, the function will invoke
    download_instagram_media based on `gallery-dl`, and will return a list of media (may contain
    videos, pictures or mix of them).

    Parameters:
        url (str): The URL of the video to download.

    Returns:
        list: The list of path to the downloaded media files if successful, or None if the download fails.

    Exceptions:
        Handles exceptions for subprocess errors, timeouts, or unexpected errors during the
        download process. Logs the errors if debugging is enabled.
    """
    temp_dir = tempfile.mkdtemp()
    command = [
        "yt-dlp",  # Assuming yt-dlp is installed and in the PATH
        *(["--cookies", "instagram_cookies.txt"] if instagram_cookies else []),
        "-S",
        "vcodec:h264,fps,res,acodec:m4a",
        url,
        "-o",
        os.path.join(temp_dir, "%(id)s.%(ext)s"),
    ]

    debug("Downloading video from URL: %s", url)
    debug("Downloading video to temp_dir full path: %s", os.path.abspath(temp_dir))
    result_path = temp_dir  # Initialize the result variable

    try:
        subprocess.run(command, check=True, timeout=120)
        for filename in os.listdir(temp_dir):
            if filename.endswith(".mp4"):
                result_path = os.path.join(temp_dir, filename)
                debug("Downloaded video found at path: %s", result_path)
                break  # Exit the loop once the file is found
    except subprocess.CalledProcessError as e:
        error("Error downloading video: %s", e)
        if "[Instagram]" in str(e) and "No video formats found!" in str(e):
            debug("The yt-dlp did not find Instagram reels, trying gallery-dl for images")
            try:
                result_path = download_instagram_media(url, temp_dir)
                if result_path:
                    debug("Successfully downloaded Instagram media using gallery-dl")
                    return result_path
                else:
                    error("Failed to download Instagram media using gallery-dl")
            except Exception as gallery_dl_error:  # pylint: disable=broad-except
                error("Unexpected error during Instagram download images by gallery-dl: %s", gallery_dl_error)
    except subprocess.TimeoutExpired as e:
        error("Download process timed out: %s", e)
    except (OSError, IOError) as e:
        error("File system error occurred: %s", e)
    except yt_dlp.utils.DownloadError as e:
        error("Download error occurred: %s", e)
    except yt_dlp.utils.ExtractorError as e:
        error("Extractor error occurred: %s", e)

    return result_path  # Return the result variable at the end
