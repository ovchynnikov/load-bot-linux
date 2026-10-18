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

# Run in Docker (self-hosting; production uses Dokku)
docker compose up -d
```

Match `.github/workflows/linets.yml` when verifying. Do not combine the two Pylint
invocations: that can mask test import errors. The separate test check needs the
explicit source `PYTHONPATH` above.

The test suite uses dummy keys and offline transports, including bot startup checks;
real API keys are not required. Also validate JSON (`jq empty src/responses_*.json tests/fixtures/*.json`).

## Architecture

Entry point is `src/main.py:main()` (`run_polling`). One `MessageHandler`
(`handle_message`) routes every text message: access check → `bot_health` → image
command → LLM mention → supported-URL download.

- `main.py` — routing, image generation, LLM orchestration (Grok + Gemini), rate limits, startup/shutdown.
- `downloads.py` — download route: URL cleanup, download/compress through worker threads, Telegram media sending.
- `video_utils.py` — `yt-dlp` / `gallery-dl` download, `ffmpeg` compression, `ffprobe` probing, cookies.
- `grok_api.py` — xAI Responses adapter: search reservation, execution evidence, one tool-free fallback.
- `llm_config.py` — validated LLM/search settings (read only when `USE_LLM=true`).
- `llm_jobs.py` — `JobGate` (per-user lock, then global semaphore, pending/per-user caps), typing leases, `storage_call`.
- `llm_text.py` — command parsing, localized LLM messages, citation rendering, UTF-16-safe splitting.
- `llm_delivery.py` — `send_reply` (threaded, chunked, topic-safe plain text).
- `safe_diagnostics.py` — `exception_summary`: loggable error metadata without raw exception text.
- `db_storage.py` — SQLite (`BotStorage`): per-user context/limits plus search quota, reservation and notice tables.
- `permissions.py` — access control (`LIMIT_BOT_ACCESS`), supported-site list.
- `cleanup.py`, `logger.py`, `general_error_handler.py` — temp-file cleanup, logging, error routing/admin alerts.
- `responses_uk.json` / `responses_en.json` — random replies for mentions without LLM, picked by `LANGUAGE`.

LLM request flow: `handle_message` → `schedule_llm_message` admits a `JobGate` ticket and starts
a PTB background task (`application.create_task`) with a typing lease → `run_llm_job` waits for
the user's lock and a global slot → `respond_with_llm_message` checks limits and builds messages →
`GrokService.generate` reserves search quota, calls Responses, settles the reservation, falls back
once without tools if needed → `send_reply` delivers → context is persisted.

Download flow: `handle_message` → `downloads.handle_download` → metadata/download/compress run in
worker threads via `run_blocking` (the handler waits for the worker even on cancellation) → media
is sent → the temp directory is cleaned in `finally`. Updates are still handled one at a time.

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
  (`PLAIN_TEXT_INSTRUCTION`, `PERSONA_INSTRUCTIONS`, `SEARCH_INSTRUCTION`) — don't duplicate
  prompt text at call sites.

## Gotchas

- **Secrets:** never commit `.env`, `.env.*` (except `.env.example`), `instagram_cookies.txt` or
  `reddit_cookies.txt`. SQLite data (`src/data/`, `*.db`) is gitignored too. Cookie files are
  provided at runtime as writable mounts (yt-dlp saves the jar back); `.dockerignore` keeps them
  out of images.
- `.env.example` has no inline comments on purpose: `docker run --env-file` keeps `# ...` as
  part of the value.
- `LIMIT_BOT_ACCESS` fails closed: only `false`/`0`/`no`/`off` (case-insensitive) disable it.
- Every push to `main` deploys (Dokku push + Docker image); this is intended. Work directly on
  `main`; don't create branches unless the user asks.
- Never pass `message.message_thread_id` directly to Telegram; use `llm_delivery.message_topic_id`
  (non-forum reply chains also carry a thread ID).
- Log exceptions with `safe_diagnostics.exception_summary`, not `str(exc)` — SDK error text
  can echo prompts.
- Search quota state lives in its own tables; never add columns to `user_data` (its LLM save
  path upserts selected fields only, keep it that way).
- Blocking work (subprocess, yt-dlp, ffmpeg) must not run directly on the event loop; use
  `downloads.run_blocking` / `llm_jobs.storage_call`.
- Downloading needs system `ffmpeg`; it isn't a pip dependency.
- LLM features are off unless `USE_LLM=true` and the matching API key is set.
- Dependency version bumps (`yt-dlp`, `gallery-dl`) are routine and land via small commits.

## Commit / PR

- Small, focused commits. Prefixes seen in history: `feat:`, `fix:`, `chore:`.
- Bump dependency versions in `src/requirements.txt`; note yt-dlp/gallery-dl bumps in the
  commit message.
