"""DeepSeek via OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.models.schemas import empty_fallback
from app.models_providers.base import SentimentModel, build_prompt, parse_llm_json
from app.utils.retry import llm_retry

logger = logging.getLogger(__name__)


class DeepSeekSentimentModel(SentimentModel):
    name = "deepseek"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek model.")
        self._client = AsyncOpenAI(
            api_key=self._settings.deepseek_api_key,
            base_url=self._settings.deepseek_base_url,
        )

    @llm_retry()
    async def _call_api(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=self._settings.request_timeout_s,
        )
        choice = resp.choices[0].message.content
        if not choice:
            return ""
        return choice

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
            logger.exception("DeepSeek analyze failed: %s", e)
            fb = empty_fallback(f"deepseek_error: {e!s}", str(e))
            fb["latency_ms"] = (time.perf_counter() - t0) * 1000
            fb["raw_response"] = raw
            fb["error"] = str(e)
            return fb
