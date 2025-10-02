FROM python:3.12

LABEL org.opencontainers.image.source="https://github.com/ovchynnikov/load-bot-linux" \
        org.opencontainers.image.licenses="MIT" \
        org.opencontainers.image.title="Social media content download bot" \
        org.opencontainers.image.description="Telegram bot to download videos from tiktok, x(twitter), reddit, youtube shorts, instagram reels and many more"

RUN --mount=type=bind,target=/tmp/requirements.txt,source=src/requirements.txt \
    echo "deb http://deb.debian.org/debian bookworm main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb-src http://deb.debian.org/debian bookworm main contrib non-free" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libva-drm2 libva-x11-2 libva2 vainfo \
    # Uncomment the line below for Intel servers (comment out for ARM/M4 MacBook local development) \
    && apt-get install --no-install-recommends -y intel-media-va-driver-non-free \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /bot

WORKDIR /bot

# https://stackoverflow.com/questions/58701233/docker-logs-erroneously-appears-empty-until-container-stops
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
