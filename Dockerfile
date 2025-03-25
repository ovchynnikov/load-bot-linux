FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/ovchynnikov/load-bot-linux" \
        org.opencontainers.image.licenses="MIT" \
        org.opencontainers.image.title="Social media content download bot" \
        org.opencontainers.image.description="Telegram bot to download videos from tiktok, x(twitter), reddit, youtube shorts, instagram reels and many more"

RUN --mount=type=bind,target=/tmp/requirements.txt,source=src/requirements.txt \
    apt-get update && apt-get install --no-install-recommends -y \
    software-properties-common \
    ffmpeg \
    libva-drm2 \
    libva-x11-2 \
    libva2 \
    vainfo \
    && add-apt-repository --component non-free \
    && apt-get update \
    && apt install intel-media-va-driver-non-free \
    && apt-get update \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /bot

WORKDIR /bot

# https://stackoverflow.com/questions/58701233/docker-logs-erroneously-appears-empty-until-container-stops
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
