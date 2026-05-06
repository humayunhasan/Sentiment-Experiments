"""Anthropic Claude Haiku."""

from __future__ import annotations

import logging
import time
from typing import Any

from anthropic import AsyncAnthropic

from app.config import Settings, get_settings
from app.models.schemas import empty_fallback
from app.models_providers.base import SentimentModel, build_prompt, parse_llm_json
from app.utils.retry import llm_retry

logger = logging.getLogger(__name__)


class ClaudeSentimentModel(SentimentModel):
    name = "claude_haiku"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Claude model.")
        self._client = AsyncAnthropic(api_key=self._settings.anthropic_api_key)

    @llm_retry()
    async def _call_api(self, prompt: str) -> str:
        msg = await self._client.messages.create(
            model=self._settings.anthropic_model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=self._settings.request_timeout_s,
        )
        parts: list[str] = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    async def analyze(self, text: str) -> dict[str, Any]:
        prompt = build_prompt(text)
        t0 = time.perf_counter()
        raw = ""
        try:
            raw = await self._call_api(prompt)
            parsed = parse_llm_json(raw)
            parsed["latency_ms"] = (time.perf_counter() - t0) * 1000
            parsed["raw_response"] = raw
            parsed["error"] = None
            return parsed
        except Exception as e:
            logger.exception("Claude analyze failed: %s", e)
            fb = empty_fallback(f"claude_error: {e!s}", str(e))
            fb["latency_ms"] = (time.perf_counter() - t0) * 1000
            fb["raw_response"] = raw
            fb["error"] = str(e)
            return fb
