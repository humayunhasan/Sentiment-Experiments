"""Local Hugging Face sentiment via configurable models."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from app.config import Settings, get_settings
from app.models.schemas import empty_fallback
from app.models_providers.base import SentimentModel

logger = logging.getLogger(__name__)

_pipelines: dict[str, Any] = {}


def _map_label_to_sentiment(label: str) -> str:
    u = (label or "").upper().strip()
    if u in ("NEGATIVE", "LABEL_0"):
        return "negative"
    if u in ("NEUTRAL", "LABEL_1"):
        return "neutral"
    if u in ("POSITIVE", "LABEL_2"):
        return "positive"
    if "NEGATIVE" in u:
        return "negative"
    if "NEUTRAL" in u:
        return "neutral"
    if "POSITIVE" in u:
        return "positive"
    return "neutral"


def _get_pipeline(model_id: str):
    global _pipelines
    if model_id not in _pipelines:
        # Requirements: Set env vars before importing transformers to avoid torchvision conflicts
        os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
        os.environ["TRANSFORMERS_NO_VISION"] = "1"

        # Requirement: Import inside function to avoid top-level issues
        from transformers import pipeline

        # Requirement: Use "text-classification" task
        _pipelines[model_id] = pipeline("text-classification", model=model_id)
    return _pipelines[model_id]


def _predict_sync(model_id: str, text: str) -> dict[str, Any]:
    pipe = _get_pipeline(model_id)
    truncated = (text or "")[:2048]
    raw_out = pipe(truncated)
    if isinstance(raw_out, list) and raw_out:
        out = raw_out[0]
    elif isinstance(raw_out, dict):
        out = raw_out
    else:
        out = {"label": "neutral", "score": 0.0}

    label_raw = str(out.get("label", "neutral"))
    score = float(out.get("score", 0.0))
    mapped = _map_label_to_sentiment(label_raw)
    raw_blob = json.dumps(out, default=str)

    return {
        "overall_sentiment": mapped,
        "entities": [],
        "confidence": max(0.0, min(1.0, score)),
        "reason": f"Local Hugging Face classification using {model_id}",
        "raw_response": raw_blob,
    }


class HFTabularisaiModel(SentimentModel):
    name = "hf_tabularisai"

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._model_id = s.hf_tabularisai_model

    async def analyze(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            parsed = await asyncio.to_thread(_predict_sync, self._model_id, text)
            parsed["latency_ms"] = (time.perf_counter() - t0) * 1000
            parsed["error"] = None
            return parsed
        except Exception as e:
            # Requirement: Return a clean error and do not crash the experiment
            logger.exception("HF tabularisai failed for model %s: %s", self._model_id, e)
            fb = empty_fallback(f"hf_error: {e!s}", str(e))
            fb["latency_ms"] = (time.perf_counter() - t0) * 1000
            fb["raw_response"] = ""
            fb["error"] = str(e)
            return fb
