"""Validated configuration for bounded LLM jobs and hosted web search."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSettings:
    """Configuration is loaded explicitly, after the application's dotenv setup."""

    model: str = "grok-4.3"
    reasoning: str = "low"
    search_enabled: bool = False
    search_mode: str = "auto"
    user_quota: int = 20
    global_quota: int = 50
    timeout: float = 120
    search_timeout: float = 75
    fallback_timeout: float = 30
    max_turns: int = 3
    max_output_tokens: int = 4096
    concurrency: int = 2
    pending: int = 10
    jobs_per_user: int = 3
    queue_timeout: float = 120

    def __post_init__(self):
        if self.search_mode not in {"auto", "explicit"}:
            raise ValueError("GROK_WEB_SEARCH_MODE must be auto or explicit")
        if self.reasoning not in {"none", "low", "medium", "high"}:
            raise ValueError("Invalid GROK_REASONING_EFFORT")
        if not self.model or min(self.user_quota, self.global_quota) < 0:
            raise ValueError("Model must be nonempty and search quotas must be nonnegative")
        if min(self.max_turns, self.max_output_tokens, self.concurrency, self.pending, self.jobs_per_user) < 1:
            raise ValueError("Turn, token, concurrency, pending and per-user job limits must be positive")
        if not all(
            0 < value < float("inf")
            for value in (self.timeout, self.search_timeout, self.fallback_timeout, self.queue_timeout)
        ):
            raise ValueError("LLM timeouts must be positive and finite")
        if self.search_timeout + self.fallback_timeout > self.timeout:
            raise ValueError("Search and fallback budgets must fit GROK_REQUEST_TIMEOUT_SEC")

    @classmethod
    def from_env(cls):
        """Read only documented settings; zero quotas explicitly disable admission."""
        return cls(
            model=os.getenv("GROK_MODEL", "grok-4.3"),
            reasoning=os.getenv("GROK_REASONING_EFFORT", "low").lower(),
            search_enabled=os.getenv("GROK_WEB_SEARCH_ENABLED", "false").lower() == "true",
            search_mode=os.getenv("GROK_WEB_SEARCH_MODE", "auto").lower(),
            user_quota=int(os.getenv("GROK_WEB_SEARCH_RPD_LIMIT", "20")),
            global_quota=int(os.getenv("GROK_WEB_SEARCH_GLOBAL_RPD_LIMIT", "50")),
            timeout=float(os.getenv("GROK_REQUEST_TIMEOUT_SEC", "120")),
            search_timeout=float(os.getenv("GROK_SEARCH_TIMEOUT_SEC", "75")),
            fallback_timeout=float(os.getenv("GROK_FALLBACK_TIMEOUT_SEC", "30")),
            max_turns=int(os.getenv("GROK_WEB_SEARCH_MAX_TURNS", "3")),
            max_output_tokens=int(os.getenv("GROK_MAX_OUTPUT_TOKENS", "4096")),
            concurrency=int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "2")),
            pending=int(os.getenv("LLM_MAX_PENDING_REQUESTS", "10")),
            jobs_per_user=int(os.getenv("LLM_MAX_JOBS_PER_USER", "3")),
            queue_timeout=float(os.getenv("LLM_QUEUE_TIMEOUT_SEC", "120")),
        )
