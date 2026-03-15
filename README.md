# Video Downloader Bot

![python-version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11_|_3.12_|_3.13-blue.svg)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linters](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/linets.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/linets.yml)
[![Publish Docker image](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-actions-push-image.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-actions-push-image.yml)
[![Push to Remote](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-action-push-to-remote.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-action-push-to-remote.yml)

A Telegram bot that downloads videos from 1000+ platforms (YouTube, Instagram, TikTok, Reddit, X, Facebook, etc.) with automatic compression and optional AI chat capabilities.

## Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Quick Start

### Docker (recommended)

```bash
# Create .env with your bot token
echo "BOT_TOKEN=your_token_here" > .env

# Run
docker run -d --name downloader-bot --restart always --env-file .env \
  -v bot-data:/bot/data \
  ovchynnikov/load-bot-linux:latest
```

### Systemd

```bash
git clone https://github.com/ovchynnikov/load-bot-linux.git
cd load-bot-linux
pip install -r src/requirements.txt
sudo apt install ffmpeg

# Create service (see Deploy with Linux Service section below)
```

### Docker Compose

```bash
git clone https://github.com/ovchynnikov/load-bot-linux.git
cd load-bot-linux
docker-compose up -d
```

Get a Telegram bot token from [@BotFather](https://t.me/botfather), then send `bot_health` to test.

## Features

- Downloads from 1000+ video platforms
- Automatic compression to fit Telegram's 50 MB limit
- GPU acceleration (Intel VAAPI)
- Instagram Stories/Carousels with automatic fallback
- Optional AI chat (Grok or Google Gemini)
- Conversation history per user
- Access control via allowlist (by username or chat ID)
- Error reporting to admin chats
- Multi-language support (Ukrainian, English)

## Setup

### Prerequisites

- Python 3.9+
- FFmpeg
- Linux OS

### Get Bot Token

1. Chat with [@BotFather](https://t.me/botfather) on Telegram
2. Create a bot and copy the token
3. Add to `.env` file

### Environment Variables

**Required:**
- `BOT_TOKEN` - Your Telegram bot token

**Optional - Basic:**
- `LANGUAGE` - `en` or `uk` (default: uk)
- `LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR (default: INFO)

**Optional - Video Processing:**
- `H_CODEC` - `libx265` (smaller) or `libx264` (default: libx265)
- `USE_GPU_COMPRESSING` - Enable Intel VAAPI (default: false)
- `INSTACOOKIES` - Use Instagram cookies file (default: false)

**Optional - Access Control:**
- `LIMIT_BOT_ACCESS` - Restrict to allowlist (default: false)
- `ALLOWED_USERNAMES` - Comma-separated usernames
- `ALLOWED_CHAT_IDS` - Comma-separated chat IDs

**Optional - Error Reporting:**
- `SEND_ERROR_TO_ADMIN` - Forward errors to admin (default: false)
- `ADMINS_CHAT_IDS` - Comma-separated admin chat IDs

**Optional - AI/LLM (Grok or Gemini):**
- `USE_LLM` - Enable AI chat (default: false)
- `LLM_PROVIDER` - `grok` or `gemini` (default: grok)
- `GROK_API_KEY` - xAI API key (get from https://console.grok.ai)
- `GEMINI_API_KEY` - Google API key (get from https://aistudio.google.com)
- `LLM_RPM_LIMIT` - Requests per minute (default: 50)
- `LLM_RPD_LIMIT` - Requests per day (default: 500)
- `IMG_GEN_RPM_LIMIT` - Image generations per minute (default: 1)
- `IMG_GEN_RPD_LIMIT` - Image generations per day (default: 25)
- `MAX_CONTEXT_MESSAGES` - Messages to remember (default: 3)
- `MAX_CONTEXT_CHARS` - Max chars per message (default: 500)

**Optional - Cleanup:**
- `USER_CLEANUP_TTL_DAYS` - Remove inactive users after N days (default: 3)
- `USER_CLEANUP_INTERVAL_HOURS` - Cleanup interval (default: 24)

### Example .env

```ini
BOT_TOKEN=123456789:ABCDEFghijklmnopqrstuvwxyz
LANGUAGE=en
LIMIT_BOT_ACCESS=false
ALLOWED_USERNAMES=
ALLOWED_CHAT_IDS=
H_CODEC=libx265
USE_GPU_COMPRESSING=false
INSTACOOKIES=false
SEND_ERROR_TO_ADMIN=false
ADMINS_CHAT_IDS=
USE_LLM=false
LLM_PROVIDER=grok
```

## Deploy with Docker

### Basic

```bash
docker run -d --name downloader-bot --restart always --env-file .env \
  -v bot-data:/bot/data \
  ovchynnikov/load-bot-linux:latest
```

### With Instagram Cookies

```bash
docker run -d --name downloader-bot --restart always --env-file .env \
  -v bot-data:/bot/data \
  -v /path/to/instagram_cookies.txt:/bot/instagram_cookies.txt \
  ovchynnikov/load-bot-linux:latest
```

Enable `INSTACOOKIES=true` in `.env`.

### With GPU (Intel)

```bash
docker run -d --name downloader-bot --restart always --env-file .env \
  -v bot-data:/bot/data \
  --device /dev/dri:/dev/dri \
  --group-add video \
  ovchynnikov/load-bot-linux:latest
```

Set `USE_GPU_COMPRESSING=true` in `.env`.

### Build Custom Image

```bash
git clone https://github.com/ovchynnikov/load-bot-linux.git
cd load-bot-linux
docker build . -t downloader-bot:latest
docker run -d --name downloader-bot --restart always --env-file .env \
  -v bot-data:/bot/data \
  downloader-bot:latest
```

## Deploy with Linux Service (Systemd)

<details>
  <summary>Click to expand</summary>

### Install

```bash
git clone https://github.com/ovchynnikov/load-bot-linux.git
cd load-bot-linux
pip install -r src/requirements.txt
sudo apt install ffmpeg
sudo chmod a+rx $(which yt-dlp)
```

### Create Service

```bash
sudo nano /etc/systemd/system/downloader-bot.service
```

```ini
[Unit]
Description=Video Downloader Bot Service
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/bot
ExecStart=/usr/bin/python3 /path/to/bot/main.py
Restart=always
RestartSec=5
Environment="BOT_TOKEN=your_token_here"
Environment="LANGUAGE=en"
Environment="LOG_LEVEL=INFO"

[Install]
WantedBy=multi-user.target
```

### Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable downloader-bot.service
sudo systemctl start downloader-bot.service
sudo systemctl status downloader-bot.service
```

### View Logs

```bash
journalctl -u downloader-bot.service -f
```

</details>

## Usage

### Send a Video URL

Simply send any supported platform URL:

```
https://youtube.com/shorts/video_id
https://www.instagram.com/reel/ABC123/
https://www.tiktok.com/@user/video/123456
```

### Download Full YouTube Video

Prefix with `**`:

```
**https://www.youtube.com/watch?v=video_id
```

### Check Bot Status

Send `bot_health` or `ботяра` to the bot. It will respond with status.

### AI Chat (if enabled)

Send any message and the bot will respond using Grok or Gemini.

### Generate Image (Grok only)

```
image: a sunset over mountains
```

## Supported Platforms

- Instagram (Reels, Stories, Carousels)
- Facebook Reels
- TikTok
- YouTube (Shorts and full videos)
- Reddit
- X.com (Twitter)
- 1000+ others via yt-dlp

See the [full list](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).

## Instagram Stories and Carousels

To download private/age-restricted content:

1. Export cookies using a browser extension
2. Save as `instagram_cookies.txt`
3. Mount in Docker or place in working directory
4. Set `INSTACOOKIES=true`

The bot automatically falls back to gallery-dl if yt-dlp fails.

## Access Control

Restrict bot access to specific users or groups:

```ini
LIMIT_BOT_ACCESS=true
ALLOWED_USERNAMES=username1,username2
ALLOWED_CHAT_IDS=12345,67890
```

To get your IDs, send `bot_health` to the bot.

## Error Reporting

Forward errors to admin chats:

```ini
SEND_ERROR_TO_ADMIN=true
ADMINS_CHAT_IDS=12345,67890
```

## AI/LLM Chat

Optional integration with language models.

### Setup Grok (xAI)

1. Sign up at https://console.grok.ai
2. Get API key
3. Set in `.env`:
   ```ini
   USE_LLM=true
   LLM_PROVIDER=grok
   GROK_API_KEY=xai-your-key
   ```

### Setup Gemini (Google)

1. Get API key at https://aistudio.google.com
2. Set in `.env`:
   ```ini
   USE_LLM=true
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-key
   ```

### Usage

- Send a message: bot responds with AI
- Image generation (Grok): `image: prompt`
- Bot remembers conversation history (configurable)

## Troubleshooting

### Bot not responding

```bash
# Check if running
docker ps | grep downloader-bot

# View logs
docker logs downloader-bot

# Systemd logs
journalctl -u downloader-bot.service -n 50
```

Send `bot_health` to test.

### Video download fails

- Check if platform is supported
- For YouTube, use `**` prefix for full videos
- Check available disk space
- Enable debug logging: `LOG_LEVEL=DEBUG`

### Instagram downloads don't work

- Set up cookies file (see Instagram section)
- Enable `INSTACOOKIES=true`
- Ensure cookies are valid

### GPU not working

Check if Intel GPU is present:

```bash
vainfo
```

If not found, install drivers:

```bash
sudo apt install intel-media-va-driver-non-free
```

### Database locked

```bash
docker restart downloader-bot
```

Or clear database (WARNING: loses user data):

```bash
docker exec downloader-bot rm /bot/data/bot.db
docker restart downloader-bot
```

### Out of memory

Enable GPU compression: `USE_GPU_COMPRESSING=true`

## Contributing

Contributions welcome. Please:

1. Check existing issues
2. Open an issue or fork and submit a PR
3. Follow code style (black, type hints)

To set up development:

```bash
git clone https://github.com/yourusername/load-bot-linux.git
cd load-bot-linux
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Credits

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Video downloader
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) - Telegram API
- [gallery-dl](https://github.com/mikf/gallery-dl) - Media gallery downloader
- [FFmpeg](https://ffmpeg.org) - Video processing

---

Backend code uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) which is released under The [Unlicense](https://unlicense.org/). All rights for yt-dlp belong to their respective authors.
