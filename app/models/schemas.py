"""Pydantic-style document shapes (plain dicts used at runtime; types for clarity)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

SentimentLabel = Literal["positive", "negative", "neutral", "mixed"]


class EntitySentiment(TypedDict, total=False):
    entity: str
    sentiment: Literal["positive", "negative", "neutral"]
    evidence: str


class ModelResultPayload(TypedDict, total=False):
    overall_sentiment: SentimentLabel
    entities: list[EntitySentiment]
    confidence: float
    reason: str
    latency_ms: float
    error: str | None
    raw_response: str | None


class MajorityLabelDoc(TypedDict, total=False):
    overall_sentiment: SentimentLabel
    votes: dict[str, int]
    eligible_models: int
    tie_breaker_applied: bool


class ExperimentDocument(TypedDict, total=False):
    comment_id: str
    text: str
    experiment_id: str
    results: dict[str, ModelResultPayload]
    majority_label: MajorityLabelDoc | None
    model_agreement: dict[str, bool] | None


def empty_fallback(reason: str, error: str | None = None) -> dict[str, Any]:
    return {
        "overall_sentiment": "neutral",
        "entities": [],
        "confidence": 0.0,
        "reason": reason,
        "error": error,
    }


VALID_OVERALL: frozenset[str] = frozenset({"positive", "negative", "neutral", "mixed"})
VALID_ENTITY_SENTIMENT: frozenset[str] = frozenset({"positive", "negative", "neutral"})
