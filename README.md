# Video Downloader Bot

![python-version](https://img.shields.io/badge/python-3.11_|_3.12_|_3.13_|_3.14-blue.svg)
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
cp .env.example .env  # then fill in BOT_TOKEN and other values
docker compose up -d
```

Cookie files can be mounted by uncommenting the cookie lines in `docker-compose.yml`.

Get a Telegram bot token from [@BotFather](https://t.me/botfather), then send `bot_health` to test.

## Features

- Downloads from 1000+ video platforms
- Automatic compression to fit Telegram's 50 MB limit
- GPU acceleration (Intel VAAPI)
- Instagram Stories/Carousels with automatic fallback
- Access control via allowlist (by username or chat ID)
- Error reporting to admin chats
- Multi-language support (Ukrainian, English)
- Optional AI chat (Grok or Google Gemini)
- Conversation context prompt history per user for AI

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

<details>
  <summary>Click to expand</summary>

**Optional - Basic:**
- `LANGUAGE` - `en` or `uk` (default: uk)
- `LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR (default: INFO)

**Optional - Video Processing:**
- `H_CODEC` - `libx265` (smaller) or `libx264` (default: libx265)
- `USE_GPU_COMPRESSING` - Enable Intel VAAPI (default: false)
- `INSTACOOKIES` - Use Instagram cookies file (default: false)
- `REDDITCOOKIES` - Use Reddit cookies file `reddit_cookies.txt` (default: false)

**Optional - Access Control:**
- `LIMIT_BOT_ACCESS` - Restrict to allowlist (default: false)
- `ALLOWED_USERNAMES` - Comma-separated usernames
- `ALLOWED_CHAT_IDS` - Comma-separated chat IDs

**Optional - Error Reporting:**
- `SEND_ERROR_TO_ADMIN` - Forward errors to admin (default: false)
- `ADMINS_CHAT_IDS` - Comma-separated admin chat IDs
- `SEND_USER_INFO_WITH_HEALTHCHECK` - Send user info when bot_health command is triggered

**Optional - AI/LLM (Grok or Gemini):**
- `USE_LLM` - Enable AI chat (default: false)
- `LLM_PROVIDER` - `grok` or `gemini` (default: grok)
- `GROK_API_KEY` - xAI API key (get from https://console.grok.ai)
- `GROK_MODEL` - Text model (default: `grok-4.3`; previously `grok-4-latest`). Existing environment overrides take precedence.
- `GROK_REASONING_EFFORT` - `none`, `low`, `medium`, or `high` (default: `low`)
- `GEMINI_API_KEY` - Google API key (get from https://aistudio.google.com)
- `LLM_RPM_LIMIT` - LLM Requests (prompts) per minute (default: 50)
- `LLM_RPD_LIMIT` - LLM Requests per day (default: 500)
- `GROK_IMG_MODEL` - default: grok-imagine-image
- `IMG_GEN_RPM_LIMIT` - Image generations per minute (default: 1)
- `IMG_GEN_RPD_LIMIT` - Image generations per day (default: 25)
- `MAX_CONTEXT_MESSAGES` - LLM Messages (prompts) to remember per user (default: 3)
- `MAX_CONTEXT_CHARS` - Max chars per message (default: 500)

**Optional - Cleanup:**
- `USER_CLEANUP_TTL_DAYS` - Remove LLM context messages (prompts) for inactive users after N days (default: 3)
- `USER_CLEANUP_INTERVAL_HOURS` - Cleanup interval (default: 24)

</details>

### Grok web search

Enable `USE_LLM=true`, `LLM_PROVIDER=grok`, and `GROK_WEB_SEARCH_ENABLED=true` with
a valid `GROK_API_KEY`. The existing asynchronous OpenAI-compatible client uses
xAI's Responses API. No separate search service or key is needed.

With `GROK_WEB_SEARCH_MODE=auto` (default), Grok decides when to search. Use
`ботяра, web: your question` to require search explicitly, or select `explicit`
mode to offer search only for that prefix. Prompts and URLs keep their original
case. Image commands, health checks, Reddit cookies, and downloads keep their
existing routing. Explicit web requests with Gemini receive a labeled ordinary
answer; they do not switch providers.

Search admission defaults to **20 requests per user and 50 across the bot per UTC
day** (`GROK_WEB_SEARCH_RPD_LIMIT`, `GROK_WEB_SEARCH_GLOBAL_RPD_LIMIT`). Reservations
are durable, atomic, and released only after a completed response proves that no
search ran. Failures and ambiguous timeouts retain them. Zero quota disables
search admission. These are request quotas: a request/turn can make multiple paid
tool calls. `GROK_WEB_SEARCH_MAX_TURNS=3` bounds hosted turns, not dollars.

On a failed search, one tool-free fallback is attempted and clearly labeled as
unverified. When quotas block auto mode, the first delivered reply per user/UTC
day includes a notice; later auto replies that day are ordinary chat. Explicit
`web:` requests remain labeled every time. Notices survive restarts. The bot
rechecks quota availability on each request because no-search reservations can
be released later.

Relevant timeouts and bounds (all documented in `.env.example`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `GROK_REQUEST_TIMEOUT_SEC` | 120 | Overall provider work, excluding queue/delivery |
| `GROK_SEARCH_TIMEOUT_SEC` | 75 | Search attempt budget |
| `GROK_FALLBACK_TIMEOUT_SEC` | 30 | Reserved fallback budget; search + fallback must fit the overall budget |
| `GROK_MAX_OUTPUT_TOKENS` | 4096 | Generation budget, including reasoning |
| `LLM_MAX_CONCURRENT_REQUESTS` | 2 | Active LLM jobs |
| `LLM_MAX_PENDING_REQUESTS` | 10 | Accepted jobs waiting for a user lock/active slot |
| `LLM_MAX_JOBS_PER_USER` | 3 | Total accepted jobs per user, active plus pending |
| `LLM_QUEUE_TIMEOUT_SEC` | 120 | Maximum admission wait |

LLM jobs run in the background, in order per user, with best-effort typing while
queued/generating/sending. Existing synchronous download processing can still
delay the event loop. Replies include numbered source URLs when available, have
previews disabled, and split safely into threaded plain-text messages. Conversation
context remains per user across chats and keeps the configured truncation limits.
Grok requests use `store=False`; this controls response storage, not every aspect
of provider data retention. Usage logs include tool/token counts and reported cost
ticks without adding raw LLM prompts or provider payloads. Error diagnostics include
safe HTTP status/error codes, known Telegram failure reasons, retry delays and stack
locations; arbitrary exception messages and frame locals are omitted. Optional LLM settings are validated only with `USE_LLM=true`;
invalid enabled settings fail startup, while disabled LLM settings do not block downloads.

Persist `/bot/data` in Dokku/Docker so deployments do not reset quotas or history.
Back up SQLite before rollout. For an already configured Dokku bot, after deploying
the new code, the non-secret search settings can be applied with:

```bash
dokku config:set YOUR_APP_NAME GROK_MODEL=grok-4.3 GROK_REASONING_EFFORT=low \
  GROK_WEB_SEARCH_ENABLED=true GROK_WEB_SEARCH_MODE=auto \
  GROK_WEB_SEARCH_RPD_LIMIT=20 GROK_WEB_SEARCH_GLOBAL_RPD_LIMIT=50
```

Set `GROK_WEB_SEARCH_ENABLED=false` to stop new search admission. In-flight work
may still finish/be billed. This switch does not undo the Responses migration or
background processing; retain the previous deployment for a full rollback.
Confirm Dokku's actual stop grace period: a forced container stop can interrupt
in-flight replies while their durable search reservations remain counted.

### Development verification

```bash
black --check --diff --skip-string-normalization --line-length 120 src/ tests/
(cd src && pylint --disable=R,C,W1203,W0105 . && PYTHONPATH="$PWD" pylint --disable=R,C,W1203,W0105 ../tests/)
PYTHONPATH=src python -m unittest discover -s tests -v
jq empty src/responses_*.json
```

Tests use fake provider/Telegram objects, temporary SQLite, and an isolated bot
startup with dummy credentials. They require no real API keys, do not start production
polling, or spend API credits. Rotating your Grok key does not affect this suite;
no token file is needed in `tests/`. Recorded compatibility cases are stored in
`tests/fixtures/grok-compatibility-results.json` and replayed without network calls.

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

Reddit works the same way: mount `reddit_cookies.txt` to `/bot/reddit_cookies.txt` and set `REDDITCOOKIES=true`.

Cookie mounts must be writable (no `:ro`): yt-dlp saves the cookie jar back after each run.
Cookie files are never copied into the image (see `.dockerignore`); always provide them at runtime.

Keep `.env` free of inline comments (`KEY=value  # comment`): `docker run --env-file` treats
everything after `=` as the value, including the comment.

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

Send `bot_health` to the bot. It will respond with status.

### AI Chat (if enabled)

Send `ботяра, ` any message and the bot will respond using Grok or Gemini.

### Generate Image (Grok only)

```
ботяра, image: a sunset over mountains
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

## Reddit

Reddit often requires a logged-in session to download videos. Without cookies such links fail and the bot
replies that it couldn't download the media.

1. Log in to Reddit in a browser and export cookies (Netscape format) using a browser extension
2. Save as `reddit_cookies.txt`
3. Mount in Docker (`/bot/reddit_cookies.txt`) or place in working directory
4. Set `REDDITCOOKIES=true`

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
- Image generation (Grok): `ботяра, image: prompt`
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

### Reddit downloads don't work

- Set up cookies file (see Reddit section)
- Enable `REDDITCOOKIES=true`

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
