# Video Downloader Bot Setup Guide

![python-version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11_|_3.12_|_3.13-blue.svg)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linters](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/linets.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/linets.yml)
[![Publish Docker image](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-actions-push-image.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-actions-push-image.yml)
[![Push to Remote](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-action-push-to-remote.yml/badge.svg)](https://github.com/ovchynnikov/load-bot-linux/actions/workflows/github-action-push-to-remote.yml)

This guide provides step-by-step instructions to install and run the Video Downloader bot on a Linux system.
- Backend code uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) which is released under [The Unlicense](https://unlicense.org/). All rights for yt-dlp belong to its respective authors.
---

## Deploy with Docker


Edit `.env` file with your secrets and run the container. Use `.env.example` as a reference.

```
docker build . -t downloader-bot:latest
docker run -d --name downloader-bot --restart always --env-file .env downloader-bot:latest
```
or use builded image from Docker hub
```
docker run -d --name downloader-bot --restart always --env-file .env ovchynnikov/load-bot-linux:latest
```
Alternativelly you can use docker-compose
```
docker-compose build
docker-compose up
```
---

## Alternatively, you can use Linux Service (daemon)
### 1. Clone and Install
Clone the repo
```sh
git clone https://github.com/ovchynnikov/load-bot-linux.git
```

Install dependencies
```bash
pip install -r scr/requirements.txt
```
```sh
sudo apt update && sudo apt install ffmpeg -y
```
- Change permissions for the yt-dlp
```
sudo chmod a+rx $(which yt-dlp)
```

### Create a Linux service

```sh
sudo nano /etc/systemd/system/downloader-bot.service
```

### 2. Configure the Service File

Create a service file for the bot using the following command:
```bash
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
Environment="ALLOWED_CHAT_IDS="                        # <====== REPLACE THIS (value is optional) Type: string separated by commas  Example: ALLOWED_CHAT_IDS=12349,12345,123456

[Install]
WantedBy=multi-user.target
```

## 3. Start the Bot Service

Reload the systemd daemon and start the bot service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable downloader-bot.service
sudo systemctl start downloader-bot.service
sudo systemctl status downloader-bot.service
```

## Troubleshooting

- Check the status of the service:
  ```sh
  sudo systemctl status downloader-bot.service
  ```
- View logs for more details:
  ```sh
  journalctl -u downloader-bot.service
  ```

## How to use the bot

### 1. Create Your Token for the Telegram Bot
- Follow this guide to create your Telegram bot and obtain the bot token:
  [How to Get Your Bot Token](https://www.freecodecamp.org/news/how-to-create-a-telegram-bot-using-python/).
  Make sure you put token in `.env` file

### 2. Health Check
- Verify the bot is running by sending a message with the trigger word:
  ```sh
  bot_health
  ```
  or
  ```sh
  ботяра
  ```

  If the bot is active, it will respond accordingly.

### 3. Once the bot is created and the Linux service or Docker image is running:
  1. Send a URL from **YouTube Shorts**, **Instagram Reels**, or similar platforms to the bot.
  Example:
  ```
  https://youtube.com/shorts/kaTxVLGd6IE?si=YaUM3gYjr1kcXqTm
  ```
  2. Wait for the bot to process the URL and respond.

### Supported platforms by default:
```
facebook
instagram reels
tiktok
reddit
x.com
youtube shorts
```

### Additionally, the bot can download videos from other sources. Videos shorter than 10 minutes usually work fine. The Telegram limitation for a video is 50 MB.
- To download the full video from YouTube add two asterisks before the url address.
Example:
```
  **https://www.youtube.com/watch?v=rxdu3whDVSM or with a space ** https://www.youtube.com/watch?v=rxdu3whDVSM
```
- Full list of supported sites here: [yt-dlp Supported Sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

### The bot can use 'Safelist' to restrict access for users or groups.
Ensure these variables are set in your `.env` file, without them or with the chat ID and username.
You can get your `chat_id` or `username` by setting `LIMIT_BOT_ACCESS=True` first. Then, send a link, and the bot will answer you with the chat ID and username.
- Allowed Group Chat priority is highest. All users in the Group Chat can use the bot even if they have no access to the bot in private chat.
- When `LIMIT_BOT_ACCESS=True` to use the bot in private messages add the username to the `ALLOWED_USERNAMES` variable.
- If you want a bot in your Group Chat with restrictions, leave `ALLOWED_CHAT_IDS` empty and define the `ALLOWED_USERNAMES` variable list.
```ini
LIMIT_BOT_ACCESS=False  # If True, the bot will only work for users in ALLOWED_USERNAMES or ALLOWED_CHAT_IDS
ALLOWED_USERNAMES= # a list of allowed usernames as strings separated by commas. Example: ALLOWED_USERNAMES=username1,username2,username3
ALLOWED_CHAT_IDS= # a list of allowed chat IDs as strings separated by commas. Example: ALLOWED_CHAT_IDS=12349,12345,123456
```
---
