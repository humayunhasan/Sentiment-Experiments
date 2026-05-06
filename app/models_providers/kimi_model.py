"""Kimi / Moonshot via OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.models_providers.base import SentimentModel, build_prompt, parse_llm_json
from app.utils.retry import llm_retry

logger = logging.getLogger(__name__)


def _missing_key_result() -> dict[str, Any]:
    return {
        "overall_sentiment": None,
        "entities": [],
        "confidence": 0,
        "reason": "",
        "raw_response": None,
        "latency_ms": 0,
        "error": "KIMI_API_KEY missing",
    }


class KimiSentimentModel(SentimentModel):
    name = "kimi"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
        if self._settings.kimi_api_key:
            self._client = AsyncOpenAI(
                api_key=self._settings.kimi_api_key,
                base_url=self._settings.kimi_base_url,
            )

    @llm_retry()
    async def _call_api(self, prompt: str) -> str:
        if self._client is None:
            raise RuntimeError("KIMI client not initialized")
        # Moonshot/Kimi requires temperature=1 for chat completions (0 is rejected).
        resp = await self._client.chat.completions.create(
            model=self._settings.kimi_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            timeout=self._settings.request_timeout_s,
        )
        choice = resp.choices[0].message.content
        if not choice:
            return ""
        return choice

    async def analyze(self, text: str) -> dict[str, Any]:
        if not self._settings.kimi_api_key or self._client is None:
            return _missing_key_result()

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
            logger.exception("Kimi analyze failed: %s", e)
            return {
                "overall_sentiment": None,
                "entities": [],
                "confidence": 0,
                "reason": "",
                "raw_response": raw or None,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "error": str(e),
            }
