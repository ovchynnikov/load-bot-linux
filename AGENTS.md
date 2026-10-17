# AGENTS.md

Telegram bot that downloads videos from TikTok, X/Twitter, Reddit, YouTube Shorts,
Instagram Reels, etc., and (optionally) answers chat mentions via an LLM and generates
images. Python 3.14, long-polling, all code lives in `src/`.

## Commands

Run from the repo root unless noted.

```bash
# Lint — CI runs exactly these; keep them green before finishing.
black --check --diff --skip-string-normalization --line-length 120 src/
black --check --diff --skip-string-normalization --line-length 120 tests/
# Pylint MUST run from src/, in two separate invocations as in CI.
(cd src && pylint --disable=R,C,W1203,W0105 .)
(cd src && PYTHONPATH="$PWD" pylint --disable=R,C,W1203,W0105 ../tests/)
PYTHONPATH=src python -m unittest discover -s tests -v

# Auto-format
black --skip-string-normalization --line-length 120 src/

# Run locally (needs a populated .env in repo root + ffmpeg installed)
cd src && python main.py

# Run in Docker (recommended)
docker-compose up -d
```

Match `.github/workflows/linets.yml` when verifying. Do not combine the two Pylint
invocations: that can mask test import errors. The separate test check needs the
explicit source `PYTHONPATH` above.

The test suite uses dummy keys and offline transports, including bot startup checks;
real API keys are not required. Also validate JSON (`jq empty src/responses_*.json tests/fixtures/*.json`).

## Architecture

Entry point is `src/main.py:main()` (`run_polling`). One `MessageHandler` routes every
message: it either detects a supported URL and downloads/compresses the media, or, when
the message mentions the bot ("ботяра"), calls the LLM.

- `main.py` — Telegram handlers, LLM calls (Grok + Gemini), image generation, rate limits.
- `video_utils.py` — download (`yt-dlp` / `gallery-dl`) and `ffmpeg` compression.
- `db_storage.py` — SQLite (`BotStorage`) for per-user context / rate-limit state.
- `permissions.py` — access control (`LIMIT_BOT_ACCESS`), supported-site list.
- `cleanup.py`, `logger.py`, `general_error_handler.py` — housekeeping, logging, error routing.
- `responses_uk.json` / `responses_en.json` — user-facing strings, picked by `LANGUAGE`.

## Conventions

- Formatting is enforced: black, line length **120**, string quotes left as-is
  (`--skip-string-normalization`). Don't reformat quotes or wrap at 88.
- Pylint runs with `R,C,W1203,W0105` disabled — don't "fix" lazy-logging (`%s`) into f-strings.
- **i18n:** every user-facing string exists in both `responses_uk.json` and
  `responses_en.json`, or is branched inline on `language == "uk"` (default `uk`). Never
  hardcode a single-language reply. Keep both JSON files in sync.
- Config comes from env vars read via `os.getenv` with defaults; document new ones in
  `.env.example`. Feature toggles: `USE_LLM`, `LLM_PROVIDER` (`grok`|`gemini`),
  `USE_CONVERSATION_CONTEXT`, `LIMIT_BOT_ACCESS`.
- Keep the shared LLM style/persona in the constants defined in `main.py`
  (`PLAIN_TEXT_INSTRUCTION` and the per-language `instruction`) — don't duplicate prompt
  text at call sites.

## Gotchas

- **Secrets:** never commit `.env`, `.env.*` (except `.env.example`), or
  `instagram_cookies.txt`. SQLite data (`src/data/`, `*.db`) is gitignored too.
- Downloading needs system `ffmpeg`; it isn't a pip dependency.
- LLM features are off unless `USE_LLM=true` and the matching API key is set.
- Dependency version bumps (`yt-dlp`, `gallery-dl`) are routine and land via small commits.

## Commit / PR

- Small, focused commits. Prefixes seen in history: `feat:`, `fix:`, `chore:`.
- Bump dependency versions in `src/requirements.txt`; note yt-dlp/gallery-dl bumps in the
  commit message.
