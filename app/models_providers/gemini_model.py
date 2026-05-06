"""Google Gemini Flash."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import google.generativeai as genai

from app.config import Settings, get_settings
from app.models.schemas import empty_fallback
from app.models_providers.base import SentimentModel, build_prompt, parse_llm_json
from app.utils.retry import llm_retry

logger = logging.getLogger(__name__)


def _sync_generate(model_name: str, prompt: str) -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    text = ""
    try:
        text = resp.text or ""
    except Exception:
        for c in getattr(resp, "candidates", []) or []:
            parts = getattr(c.content, "parts", None) or []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    text += t
    return text


class GeminiSentimentModel(SentimentModel):
    name = "gemini_flash"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini model.")
        genai.configure(api_key=self._settings.gemini_api_key)

    @llm_retry()
    async def _call_api(self, prompt: str) -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_generate, self._settings.gemini_model, prompt),
            timeout=self._settings.request_timeout_s,
        )

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
            logger.exception("Gemini analyze failed: %s", e)
            fb = empty_fallback(f"gemini_error: {e!s}", str(e))
            fb["latency_ms"] = (time.perf_counter() - t0) * 1000
            fb["raw_response"] = raw
            fb["error"] = str(e)
            return fb
