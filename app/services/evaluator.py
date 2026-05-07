"""Aggregate evaluation vs majority pseudo-label."""

from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from app.config import Settings, get_settings
from app.db import get_results_collection

logger = logging.getLogger(__name__)


def model_keys_for_report(settings: Settings | None = None) -> list[str]:
    """Keys that may appear in stored results (matches runner output keys)."""
    s = settings or get_settings()
    keys: list[str] = [
        "gpt_5_mini",
        "gemini_flash",
    ]
    if s.enable_anthropic:
        keys.append("claude_haiku")
    keys.extend(
        [
            "deepseek",
            "nltk_vader",
        ]
    )
    if s.enable_kimi:
        keys.append("kimi")
    if s.enable_hf_tabularisai:
        keys.append("hf_tabularisai")
    return keys


# Superset for evaluating older runs that may contain claude / hf regardless of current flags
ALL_KNOWN_MODEL_KEYS: list[str] = [
    "gpt_5_mini",
    "gemini_flash",
    "claude_haiku",
    "deepseek",
    "kimi",
    "nltk_vader",
    "hf_tabularisai",
]


async def evaluate_experiment(
    experiment_id: str,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Per-model stats vs majority for all documents with `majority_label` set.
    """
    s = settings or get_settings()
    coll: AsyncIOMotorCollection[Any] = get_results_collection(s)

    cursor = coll.find({"experiment_id": experiment_id, "majority_label": {"$ne": None}})
    docs = await cursor.to_list(length=None)
    total_docs = len(docs)
    if total_docs == 0:
        logger.warning("No documents for experiment_id=%s with majority_label.", experiment_id)
        return {
            "experiment_id": experiment_id,
            "total_documents": 0,
            "models": {},
            "leaderboard": [],
        }

    model_keys = list(dict.fromkeys(model_keys_for_report(s) + ALL_KNOWN_MODEL_KEYS))

    stats: dict[str, dict[str, float]] = {
        k: {
            "matches": 0.0,
            "errors": 0.0,
            "latency_sum_ms": 0.0,
            "latency_count": 0.0,
        }
        for k in model_keys
    }

    for doc in docs:
        maj = doc.get("majority_label") or {}
        truth = maj.get("overall_sentiment")
        if not isinstance(truth, str):
            continue

        results = doc.get("results") or {}
        for mk in model_keys:
            st = stats[mk]
            payload = results.get(mk)
            if not isinstance(payload, dict):
                continue
            if payload.get("error") is not None:
                st["errors"] += 1.0
            else:
                pred = payload.get("overall_sentiment")
                if isinstance(pred, str) and pred == truth:
                    st["matches"] += 1.0
            lat = payload.get("latency_ms")
            try:
                if lat is not None and payload.get("error") is None:
                    st["latency_sum_ms"] += float(lat)
                    st["latency_count"] += 1.0
            except (TypeError, ValueError):
                pass

    out_models: dict[str, Any] = {}
    for mk, st in stats.items():
        err_count = int(st["errors"])
        maj_matches = int(st["matches"])
        total = total_docs
        match_rate = maj_matches / total if total > 0 else 0.0
        err_rate = err_count / total if total > 0 else 0.0
        avg_lat = (
            st["latency_sum_ms"] / st["latency_count"] if st["latency_count"] > 0 else None
        )
        out_models[mk] = {
            "model_name": mk,
            "total": total,
            "majority_matches": maj_matches,
            "majority_match_rate": match_rate,
            "error_count": err_count,
            "error_rate": err_rate,
            "avg_latency_ms": avg_lat,
        }

    leaderboard = sorted(
        out_models.items(),
        key=lambda x: (-x[1]["majority_match_rate"], x[1]["avg_latency_ms"] or 1e9),
    )

    return {
        "experiment_id": experiment_id,
        "total_documents": total_docs,
        "models": out_models,
        "leaderboard": [v for _k, v in leaderboard],
    }
