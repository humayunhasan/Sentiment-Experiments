"""OpenAI Chat Completions (GPT-5 mini / configurable)."""

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


def _openai_model_accepts_temperature(model_id: str) -> bool:
    """GPT-5 family rejects fixed temperature=0; omit the parameter for those models."""
    m = (model_id or "").strip().lower()
    if m.startswith("gpt-5"):
        return False
    return True


class OpenAISentimentModel(SentimentModel):
    name = "gpt_5_mini"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI model.")
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    @llm_retry()
    async def _call_api(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self._settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self._settings.request_timeout_s,
        }
        if _openai_model_accepts_temperature(self._settings.openai_model):
            kwargs["temperature"] = 0
        resp = await self._client.chat.completions.create(**kwargs)
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
            logger.exception("OpenAI analyze failed: %s", e)
            fb = empty_fallback(f"openai_error: {e!s}", str(e))
            fb["latency_ms"] = (time.perf_counter() - t0) * 1000
            fb["raw_response"] = raw
            fb["error"] = str(e)
            return fb
