"""Responses API adapter and durable hosted-search policy, independent of Telegram."""

import asyncio
import logging
import time
from dataclasses import dataclass

from llm_jobs import storage_call
from safe_diagnostics import exception_summary

LOG = logging.getLogger(__name__)


class InvalidResponse(ValueError):
    """The provider did not return a usable terminal answer."""


@dataclass
class GrokResult:
    text: str
    search_state: str = "zero"
    usage: dict | None = None
    notice: str | None = None
    notice_day: str | None = None
    source_urls: tuple | None = None
    refund_eligible: bool = False


def count(value):
    return value if type(value) is int and value >= 0 else None


def parse_response(response, duration=None):
    """Missing provider extensions are unknown, never an implicit zero count."""
    raw = response.model_dump()
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    output = raw.get("output") or []
    text = "\n".join(
        content.get("text", "")
        for item in output
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    ).strip()
    LOG.info(
        "Grok response id=%s model=%s status=%s seconds=%s tokens=%s tools=%s cost_ticks=%s",
        raw.get("id"),
        raw.get("model"),
        raw.get("status"),
        duration,
        count(usage.get("total_tokens")),
        count(usage.get("num_server_side_tools_used")),
        count(usage.get("cost_in_usd_ticks")),
    )
    if raw.get("status") != "completed" or not text:
        raise InvalidResponse("Incomplete or empty Grok response")
    annotations = [
        content["annotations"]
        for item in output
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text" and isinstance(content.get("annotations"), list)
    ]
    source_urls = (
        tuple(
            annotation["url"]
            for group in annotations
            for annotation in group
            if annotation.get("type") == "url_citation" and isinstance(annotation.get("url"), str)
        )
        if annotations
        else None
    )
    tools = [item for item in output if item.get("type", "").endswith("_call")]
    web = [item for item in tools if item.get("type") == "web_search_call"]
    details = usage.get("server_side_tool_usage_details")
    detailed = count(details.get("web_search_calls")) if isinstance(details, dict) else None
    aggregate = count(usage.get("num_server_side_tools_used"))
    malformed = (
        (raw.get("usage") is not None and not isinstance(raw.get("usage"), dict))
        or ("num_server_side_tools_used" in usage and aggregate is None)
        or ("server_side_tool_usage_details" in usage and not isinstance(details, dict))
    )
    if isinstance(details, dict) and "web_search_calls" in details and detailed is None:
        malformed = True
    contradictory = aggregate is not None and detailed is not None and aggregate != detailed
    completed_web = any(item.get("status") == "completed" for item in web)
    state = "unknown"
    refund_eligible = False
    if completed_web:
        state = "used"
    elif web:
        state = "failed"
    elif not tools and not malformed and not contradictory and (aggregate == 0 or detailed == 0):
        if aggregate in (None, 0) and detailed in (None, 0):
            state = "zero"
            refund_eligible = True
    if (
        malformed
        or contradictory
        or state == "unknown"
        or (completed_web and (detailed == 0 or aggregate == 0 or (detailed is None and aggregate is None)))
    ):
        LOG.warning("Grok usage anomaly search_state=%s; zero-use refund unavailable", state)
    return GrokResult(text, state, usage, source_urls=source_urls, refund_eligible=refund_eligible)


class GrokService:
    """No transparent search retries; failure gets at most one tool-free fallback."""

    def __init__(self, client, storage, settings):
        self.client = client
        self.storage = storage
        self.settings = settings

    async def _request(self, messages, search=None):
        arguments = {
            "model": self.settings.model,
            "input": messages,
            "reasoning": {"effort": self.settings.reasoning},
            "max_output_tokens": self.settings.max_output_tokens,
            "store": False,
        }
        if search is not None:
            arguments.update(
                tools=[{"type": "web_search"}], tool_choice=search, extra_body={"max_turns": self.settings.max_turns}
            )
        else:
            instruction = (
                "Web tools are unavailable for this request. Do not claim online verification or invent citations."
            )
            if messages and messages[0]["role"] == "system":
                arguments["input"] = [{**messages[0], "content": messages[0]["content"] + "\n" + instruction}]
                arguments["input"].extend(messages[1:])
            else:
                arguments["input"] = [{"role": "system", "content": instruction}] + messages
        started = time.monotonic()
        response = await self.client.responses.create(**arguments)
        return parse_response(response, round(time.monotonic() - started, 3))

    async def generate(self, messages, user_id, explicit=False):
        """The outer provider deadline excludes admission waits and Telegram delivery."""
        async with asyncio.timeout(self.settings.timeout):
            return await self._generate(messages, user_id, explicit)

    async def _generate(self, messages, user_id, explicit):
        wants_search = self.settings.search_enabled and (explicit or self.settings.search_mode == "auto")
        if not wants_search:
            result = await self._request(messages)
            result.notice = "fallback" if explicit else None
            return result
        day = self.storage.utc_day()
        admission = asyncio.create_task(
            storage_call(
                self.storage.reserve_search, user_id, self.settings.user_quota, self.settings.global_quota, day
            )
        )
        try:
            reservation = await asyncio.shield(admission)
        except asyncio.CancelledError:
            # This cancellation happened before provider dispatch. Complete the DB
            # worker, undo its reservation, then propagate cancellation.
            reservation = await admission
            if reservation:
                await storage_call(self.storage.settle_search, reservation, True)
            raise
        if reservation is None:
            notice = explicit or not await storage_call(self.storage.search_notice_shown, user_id, day)
            result = await self._request(messages)
            result.notice = ("fallback" if explicit else "quota") if notice else None
            result.notice_day = day if notice and not explicit else None
            return result
        try:
            async with asyncio.timeout(self.settings.search_timeout):
                result = await self._request(messages, "required" if explicit else "auto")
                await storage_call(self.storage.settle_search, reservation, result.refund_eligible)
                if result.search_state == "failed" or (explicit and result.search_state != "used"):
                    raise InvalidResponse("Search execution was not verified")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.warning("Grok search failed %s; trying tool-free fallback", exception_summary(exc))
            await storage_call(self.storage.settle_search, reservation, False)
            async with asyncio.timeout(self.settings.fallback_timeout):
                result = await self._request(messages)
            result.notice = "fallback"
            return result
        return result
