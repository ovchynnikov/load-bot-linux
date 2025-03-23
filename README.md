# Video Downloader Bot Setup Guide

![python-version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11_|_3.12_|_3.13-blue.svg)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linters](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/linets.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/linets.yml)
[![Publish Docker image](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-actions-push-image.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-actions-push-image.yml)
[![Push to Remote](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-action-push-to-remote.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-action-push-to-remote.yml)

This guide provides step-by-step instructions on installation and running the Video Downloader bot on a Linux system.
- Backend code uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) which is released under The [Unlicense](https://unlicense.org/). All rights for yt-dlp belong to their respective authors.
---

## Deploy with Docker


Prerequisites:
  1. Create `.env` file in the project root folder with your token (mandatory) and access configuration (optional). Use `.env.example` as a reference.
  2. Clone the repo
  ```sh
    git clone https://github.com/ovchynnikov/load-bot-linux.git
  ```
Build and run the container
```
docker build . -t downloader-bot:latest
```
```
docker run -d --name downloader-bot --restart always --env-file .env downloader-bot:latest
```
or use a built image from **Docker hub**
```
docker run -d --name downloader-bot --restart always --env-file .env ovchynnikov/load-bot-linux:latest
```
or if you use instagram cookies
```
docker run -d --name downloader-bot --restart always --env-file .env -v /absolute/path/to/instagram_cookies.txt:/bot/instagram_cookies.txt ovchynnikov/load-bot-linux:latest
```
or if you want use GPU power of intel chip and set USE_GPU_COMPRESSING=True variable
```
docker run --rm --device /dev/dri:/dev/dri .....
```
Alternatively, you can use **docker-compose**
```
docker-compose build
```
```
docker-compose up
```
---

## Deploy with Linux Service (daemon)
<details>
  <summary>Click to expand</summary>
 1. Clone and Install
Clone the repo
```sh
git clone https://github.com/ovchynnikov/load-bot-linux.git
```

 2. Install dependencies
```bash
pip install -r scr/requirements.txt
```
```sh
sudo apt update && sudo apt install ffmpeg -y
```
 3. Change permissions for the yt-dlp
```
sudo chmod a+rx $(which yt-dlp)
```

 4. Create and configure Linux service
```sh
sudo nano /etc/systemd/system/downloader-bot.service
```

Add the following configuration to the file:
```ini
[Unit]
Description=Video Downloader Bot Service
After=network.target

[Service]
User=your_linux_user                                   # <====== REPLACE `your_linux_user` with the username that will run the bot.
WorkingDirectory=/path/to/your/bot                     # <====== REPLACE THIS with the absolute path to your bot's folder.
ExecStart=/usr/bin/python3 /path/to/your/bot/main.py   # <====== REPLACE THIS with the command to start your bot. Adjust if you're using a virtual environment.
Restart=always                                         # Ensures the bot restarts automatically if it crashes.
RestartSec=5
Environment="BOT_TOKEN=your_bot_token"                 # <====== REPLACE THIS with your bot token.
Environment="LOG_LEVEL=INFO"
Environment="LIMIT_BOT_ACCESS=False"                   # <====== REPLACE THIS (value is optional. False by default) Type: Boolean
Environment="ALLOWED_USERNAMES="                       # <====== REPLACE THIS (value is optional) Type: string separated by commas. Example: ALLOWED_USERNAMES=username1,username2,username3
Environment="ALLOWED_CHAT_IDS="                        # <====== REPLACE THIS (value is optional) Type: string separated by commas.  Example: ALLOWED_CHAT_IDS=12349,12345,123456
Environment="INSTACOOKIES=False"                       # <====== REPLACE THIS (value is optional) Type: Boolean. False by default.
Environment="ADMINS_CHAT_IDS="                         # <====== REPLACE THIS (value is optional) Type: string separated by commas. IDS to send Exceptions errors to private messages. Get this from bot health check
Environment="SEND_ERROR_TO_ADMIN=True"                 # <====== REPLACE THIS (value is optional) Type: Boolean.  Send errors to admins in private messages
Environment="H_CODEC=libx265"                          # <====== REPLACE THIS (value is optional) Type: String. libx265 or libx264 
Environment="USE_GPU_COMPRESSING=False"                # <====== Enable to use GPU for video compression using Intel chip and VAAPI. False by default

[Install]
WantedBy=multi-user.target
```

 5. Start the Bot Service

Reload the systemd daemon and start the bot service:

```bash
sudo systemctl daemon-reload
```
```bash
sudo systemctl enable downloader-bot.service
```
```bash
sudo systemctl start downloader-bot.service
```
```bash
sudo systemctl status downloader-bot.service
```

 6. Troubleshooting

- Check the status of the service:
  ```sh
  sudo systemctl status downloader-bot.service
  ```
- View logs for more details:
  ```sh
  journalctl -u downloader-bot.service
  ```
</details>

## How to use the bot

 1. Create Your Token for the Telegram Bot
- Follow this guide to create your Telegram bot and obtain the bot token:
  [How to Get Your Bot Token](https://www.freecodecamp.org/news/how-to-create-a-telegram-bot-using-python/).
  Make sure you put token in `.env` file

 2. Health Check
- Verify the bot is running by sending a message with the trigger word:
  ```sh
  bot_health
  ```
  or
  ```sh
  ботяра
  ```

  If the bot is active, it will respond accordingly.

 3. Once the bot is created and the Linux service or Docker image is running:
  Send a URL from **YouTube Shorts**, **Instagram Reels**, or similar platforms to the bot.
  Example:
  ```
  https://youtube.com/shorts/kaTxVLGd6IE?si=YaUM3gYjr1kcXqTm
  ```
  Wait for the bot to process the URL and respond.

## Supported platforms by default
```
instagram reels
facebook reels
tiktok
reddit
x.com
youtube shorts
```

### Download videos from other sources.
Videos shorter than 10 minutes usually work fine. The Telegram limitation for a video is 50 MB.
- To download the full video from YouTube add two asterisks before the url address.
Example:
```
  **https://www.youtube.com/watch?v=rxdu3whDVSM or with a space ** https://www.youtube.com/watch?v=rxdu3whDVSM
```
The expected waiting time for videos up to 10 minutes is 3-10 minutes depending on the internet speed.
- Full list of supported sites here: [yt-dlp Supported Sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

### Instagram Stories and Reels credentials
- To download Instagram stories and reels you need to create a cookies file `instagram_cookies.txt` in the `bot` folder and set env var `INSTACOOKIES` to `True`.
- You can use the `instagram_cookies_example.txt` file as a reference from the `src` folder of the repo.
- Suggestion on how to get the file: easy export with [chrome extension](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
- When you run the bot with Docker, place `instagram_cookies.txt` to the folder with your `.env` file and add `-v instagram_cookies.txt:/bot/instagram_cookies.txt` to the start command
- The bot supports downloading Instagram stories and carousels with pictures using `gallery-dl` when `yt-dlp` fails.
- The bot will automatically fall back to `gallery-dl` for Instagram URLs without `reels` when `yt-dlp` fails, multiple media files is supported.
- The same cookies file used for `yt-dlp` can be used for `gallery-dl`

## Access Control with Safe List
The bot can use 'Safelist' to restrict access for users or groups.
Ensure these variables are set in your `.env` file, without them or with the chat ID and username.
You can get your `chat_id` and `username` by setting `LIMIT_BOT_ACCESS=True` first. Then, send the word `bot_health` or `ботяра`, and the bot will answer you with the chat ID and username.
The priority for allowed Group Chat is highest. All users in the Group Chat can use the bot even if they do not have access to it in private chat.
- When `LIMIT_BOT_ACCESS=True` to use the bot in private messages add the username to the `ALLOWED_USERNAMES` variable or chat ID to `ALLOWED_CHAT_IDS`.
- If you want a bot in your Group Chat with restrictions, leave `ALLOWED_CHAT_IDS` empty and define the `ALLOWED_USERNAMES` variable list.
```ini
LIMIT_BOT_ACCESS=False  # If True, the bot will only work for users in ALLOWED_USERNAMES or ALLOWED_CHAT_IDS
ALLOWED_USERNAMES= # a list of allowed usernames as strings separated by commas. Example: ALLOWED_USERNAMES=username1,username2,username3
ALLOWED_CHAT_IDS= # a list of allowed chat IDs as strings separated by commas. Example: ALLOWED_CHAT_IDS=-412349,12345,123456
```

## Troubleshooting
If you sent a link to the bot and got no response, send the word `bot_health` or `ботяра` to the bot to check if it's working.

- in .env file set IDS to send Exceptions errors to in private messages to Admins. Get these ids from bot healthcheck.
Works only for Exceptions errors that are not handled by the bot code.

```ini
ADMINS_CHAT_IDS="your_admins_chat_id_here"  # ADMINS_CHAT_IDS=chatid_1,chatid_2,chatid_3
```

- in .env file set SEND_ERROR_TO_ADMIN=True to send errors to admins in private messages

```ini
SEND_ERROR_TO_ADMIN=True
```
---
