# Add Grok API support and improve LLM integration

## Summary
This PR adds Grok API as an alternative LLM provider alongside Gemini, implements conversation context tracking, adds configurable rate limiting, improves error handling, and optimizes token usage through context truncation.

## Changes

### 1. Grok API Integration
- Add Grok API support using OpenAI-compatible client
- Add environment variables: `LLM_PROVIDER`, `GROK_API_KEY`, `GROK_MODEL`
- Unified LLM approach: renamed `gemini_*` variables to `llm_*` for provider-agnostic naming
- Add `openai>=1.0.0` dependency to requirements.txt

### 2. Conversation Context
- Implement conversation history tracking per user
- Add `USE_CONVERSATION_CONTEXT` flag (default: True)
- Add `MAX_CONTEXT_MESSAGES` to control number of exchanges stored (default: 3)
- Add `MAX_CONTEXT_CHARS` to limit token usage by truncating stored messages (default: 500 chars)
- Context is included in prompts to maintain conversation flow
- **Token optimization**: Reduces context size by ~75% (from ~6000 to ~1500 chars for 3 exchanges)

### 3. Rate Limiting
- Implement per-user rate limiting for LLM APIs
- Add `LLM_RPM_LIMIT` (requests per minute, default: 50)
- Add `LLM_RPD_LIMIT` (requests per day, default: 500)
- Automatic cleanup of old timestamps
- User-friendly rate limit messages in Ukrainian and English

### 4. Error Handling & Logging
- Add proper handling for 429 (Too Many Requests) errors from LLM APIs
- Add detailed error logging with full traceback for debugging
- Add retry logic with 60-second delay for rate limit errors (max 2 attempts)
- Distinguish between rate limit errors and other API failures in user messages
- Log exception type along with error message
- Add error logging when API key is not configured but `USE_LLM=True`

### 5. Code Quality
- Fix Black formatting issues (line breaks, spacing)
- Add `# pylint: disable=broad-exception-caught` comments for retry logic
- Add Ukrainian translations for all new error messages

### 6. Configuration
- Add missing `USE_LLM` variable to `.env.example`
- Add all new LLM-related variables to `.env.example`
- Add configuration check messages in both Ukrainian and English

## Environment Variables Added
```ini
USE_LLM=False                          # Enable LLM responses
LLM_PROVIDER=grok                      # grok or gemini
GROK_API_KEY=your_grok_api_key
GROK_MODEL=grok-4-latest
USE_CONVERSATION_CONTEXT=True          # Enable conversation history
MAX_CONTEXT_MESSAGES=3                 # Number of exchanges to remember
MAX_CONTEXT_CHARS=500                  # Max chars per message in context (token optimization)
LLM_RPM_LIMIT=50                       # Requests per minute per user
LLM_RPD_LIMIT=500                      # Requests per day per user
```

## Benefits
- **Flexibility**: Choose between Grok (480 RPM, 4M TPM) and Gemini (5 RPM, 20 RPD)
- **Better UX**: Conversation context makes bot responses more relevant
- **Cost optimization**: Context truncation saves ~75% of tokens
- **Reliability**: Automatic retry on rate limits with user feedback
- **Debuggability**: Full error logging makes issues easy to diagnose
- **Protection**: Rate limiting prevents API quota exhaustion

## Testing
1. Set `LOG_LEVEL=DEBUG` to see detailed API logs and error traces
2. Test with `USE_LLM=True` and both `LLM_PROVIDER=grok` and `LLM_PROVIDER=gemini`
3. Test conversation context by asking follow-up questions
4. Test rate limiting by making multiple rapid requests

## Related Issues
- Fixes issue where bot returns generic error without logging actual API errors
- Fixes missing conversation context causing bot to "forget" previous messages
- Fixes token waste from storing full LLM responses in context
