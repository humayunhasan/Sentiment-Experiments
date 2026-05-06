"""NLTK VADER — lexicon-based overall sentiment only."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from app.models.schemas import empty_fallback
from app.models_providers.base import SentimentModel

logger = logging.getLogger(__name__)

_vader_ready = False


def _ensure_vader() -> None:
    global _vader_ready
    if _vader_ready:
        return
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    _vader_ready = True


def _compound_to_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _analyze_sync(text: str) -> dict[str, Any]:
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text or "")
    compound = float(scores.get("compound", 0.0))
    label = _compound_to_label(compound)
    conf = min(1.0, abs(compound))
    return {
        "overall_sentiment": label,
        "entities": [],
        "confidence": conf,
        "reason": f"VADER compound={compound:.4f} pos={scores.get('pos')} neg={scores.get('neg')} neu={scores.get('neu')}",
    }


class NLTKVaderModel(SentimentModel):
    name = "nltk_vader"

    def __init__(self) -> None:
        pass

    async def analyze(self, text: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            parsed = await asyncio.to_thread(_analyze_sync, text)
            parsed["latency_ms"] = (time.perf_counter() - t0) * 1000
            parsed["raw_response"] = parsed["reason"]
            parsed["error"] = None
            return parsed
        except Exception as e:
            logger.exception("NLTK VADER failed: %s", e)
            fb = empty_fallback(f"nltk_error: {e!s}", str(e))
            fb["latency_ms"] = (time.perf_counter() - t0) * 1000
            fb["raw_response"] = ""
            fb["error"] = str(e)
            return fb
