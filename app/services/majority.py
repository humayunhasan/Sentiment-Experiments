"""Majority vote over model overall_sentiment labels."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.config import Settings, get_settings
from app.models.schemas import VALID_OVERALL


def build_model_agreement(
    results: dict[str, dict[str, Any]],
    truth: str,
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for mk, payload in results.items():
        if not isinstance(payload, dict):
            out[mk] = False
            continue
        if payload.get("error") is not None:
            out[mk] = False
            continue
        pred = payload.get("overall_sentiment")
        out[mk] = isinstance(pred, str) and pred == truth
    return out


def compute_majority_label(result_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Majority vote on `overall_sentiment` across models.

    Ignores models that have a truthy `error` field or invalid/missing labels.
    Tie among top labels -> overall_sentiment "neutral".
    """
    labels: list[str] = []
    for _model_key, payload in result_dict.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("error") is not None:
            continue
        lab = payload.get("overall_sentiment")
        if lab is None:
            continue
        if isinstance(lab, str) and lab in VALID_OVERALL:
            labels.append(lab)

    if not labels:
        return {
            "overall_sentiment": "neutral",
            "votes": {},
            "eligible_models": 0,
            "tie_breaker_applied": True,
        }

    counts = Counter(labels)
    max_votes = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == max_votes]
    tie_breaker = len(winners) > 1
    overall = "neutral" if tie_breaker else winners[0]

    return {
        "overall_sentiment": overall,
        "votes": dict(counts),
        "eligible_models": len(labels),
        "tie_breaker_applied": tie_breaker,
    }


async def recompute_majority_for_experiment(
    experiment_id: str,
    *,
    settings: Settings | None = None,
) -> int:
    """
    Recompute `majority_label` and `model_agreement` from stored `results`
    for every document with the given experiment_id.
    """
    from motor.motor_asyncio import AsyncIOMotorCollection

    from app.db import get_results_collection

    s = settings or get_settings()
    coll: AsyncIOMotorCollection[Any] = get_results_collection(s)

    cursor = coll.find({"experiment_id": experiment_id})
    updated = 0
    async for doc in cursor:
        results = doc.get("results") or {}
        if not isinstance(results, dict):
            continue
        majority = compute_majority_label(results)
        truth = str(majority.get("overall_sentiment", "neutral"))
        agreement = build_model_agreement(results, truth)
        await coll.update_one(
            {"_id": doc["_id"]},
            {"$set": {"majority_label": majority, "model_agreement": agreement}},
        )
        updated += 1
    return updated
