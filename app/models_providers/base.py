"""Base sentiment model interface and shared parsing."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from app.models.schemas import VALID_ENTITY_SENTIMENT, VALID_OVERALL, empty_fallback

logger = logging.getLogger(__name__)

SENTIMENT_PROMPT_TEMPLATE = """You are a sentiment analysis system for YouTube comments.

Return ONLY valid JSON:

{
  "overall_sentiment": "positive" | "negative" | "neutral" | "mixed",
  "entities": [
    {
      "entity": "name",
      "sentiment": "positive" | "negative" | "neutral",
      "evidence": "text snippet"
    }
  ],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}

Rules:
- Mixed sentiment if both positive and negative exist
- Extract entities if mentioned
- No extra text outside JSON

Comment:
{text}"""


def build_prompt(text: str) -> str:
    return SENTIMENT_PROMPT_TEMPLATE.replace("{text}", text)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json_blob(raw: str) -> str:
    raw = raw.strip()
    m = _JSON_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw


def normalize_result(parsed: dict[str, Any]) -> dict[str, Any]:
    overall = parsed.get("overall_sentiment", "neutral")
    if not isinstance(overall, str) or overall not in VALID_OVERALL:
        overall = "neutral"

    entities_out: list[dict[str, Any]] = []
    entities = parsed.get("entities")
    if isinstance(entities, list):
        for e in entities:
            if not isinstance(e, dict):
                continue
            ent = str(e.get("entity", "")).strip()
            sent = e.get("sentiment", "neutral")
            if not isinstance(sent, str) or sent not in VALID_ENTITY_SENTIMENT:
                sent = "neutral"
            ev = e.get("evidence", "")
            if not isinstance(ev, str):
                ev = str(ev)
            if ent:
                entities_out.append(
                    {"entity": ent, "sentiment": sent, "evidence": ev[:2000]}
                )

    conf = parsed.get("confidence", 0.0)
    try:
        c = float(conf)
    except (TypeError, ValueError):
        c = 0.0
    c = max(0.0, min(1.0, c))

    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "overall_sentiment": overall,
        "entities": entities_out,
        "confidence": c,
        "reason": reason[:4000],
    }


def parse_llm_json(raw: str) -> dict[str, Any]:
    blob = extract_json_blob(raw)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"json_decode_error: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("root_must_be_object")
    return normalize_result(data)


class SentimentModel(ABC):
    """Async sentiment analyzer returning a uniform JSON-like dict."""

    name: str

    @abstractmethod
    async def analyze(self, text: str) -> dict[str, Any]:
        """Return dict with overall_sentiment, entities, confidence, reason."""


class MissingKeySentimentModel(SentimentModel):
    """Placeholder when API credentials are not configured."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self._reason = reason

    async def analyze(self, text: str) -> dict[str, Any]:
        fb = empty_fallback(self._reason, self._reason)
        fb["latency_ms"] = 0.0
        fb["raw_response"] = ""
        fb["error"] = self._reason
        return fb
