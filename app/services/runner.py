"""End-to-end async experiment runner."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection
from tqdm import tqdm

from app.config import Settings, get_settings
from app.db import get_results_collection
from app.models_providers.base import MissingKeySentimentModel, SentimentModel
from app.models_providers.deepseek_model import DeepSeekSentimentModel
from app.models_providers.gemini_model import GeminiSentimentModel
from app.models_providers.hf_tabularisai_model import HFTabularisaiModel
from app.models_providers.kimi_model import KimiSentimentModel
from app.models_providers.nltk_model import NLTKVaderModel
from app.models_providers.openai_model import OpenAISentimentModel
from app.services.majority import build_model_agreement, compute_majority_label
from app.services.sampler import sample_random_comments

logger = logging.getLogger(__name__)


def comment_id_from_doc(doc: dict[str, Any]) -> str:
    """Prefer `comment_id` when present; otherwise stringify `_id`."""
    raw = doc.get("comment_id")
    if raw is not None and str(raw).strip() != "":
        return str(raw)
    return str(doc.get("_id", ""))


def comment_text_from_doc(doc: dict[str, Any]) -> str:
    """Prefer `text_display`, then `text`, then `comment_text`."""
    for key in ("text_display", "text", "comment_text"):
        val = doc.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            if val.strip():
                return val
        else:
            s = str(val).strip()
            if s:
                return s
    return ""


def build_models(settings: Settings | None = None) -> list[SentimentModel]:
    """
    Active models (fixed keys in Mongo):
    gpt_5_mini, gemini_flash, deepseek, kimi, nltk_vader, hf_tabularisai
    Optional: claude_haiku when ENABLE_ANTHROPIC=true.
    """
    s = settings or get_settings()
    models: list[SentimentModel] = []

    if s.openai_api_key:
        models.append(OpenAISentimentModel(s))
    else:
        models.append(
            MissingKeySentimentModel("gpt_5_mini", "OPENAI_API_KEY not configured")
        )

    if s.gemini_api_key:
        models.append(GeminiSentimentModel(s))
    else:
        models.append(
            MissingKeySentimentModel("gemini_flash", "GEMINI_API_KEY not configured")
        )

    if s.enable_anthropic:
        from app.models_providers.claude_model import ClaudeSentimentModel

        if s.anthropic_api_key:
            models.append(ClaudeSentimentModel(s))
        else:
            models.append(
                MissingKeySentimentModel(
                    "claude_haiku", "ANTHROPIC_API_KEY not configured"
                )
            )

    if s.deepseek_api_key:
        models.append(DeepSeekSentimentModel(s))
    else:
        models.append(
            MissingKeySentimentModel("deepseek", "DEEPSEEK_API_KEY not configured")
        )

    if s.enable_kimi:
        models.append(KimiSentimentModel(s))

    models.append(NLTKVaderModel())

    if s.enable_hf_tabularisai:
        models.append(HFTabularisaiModel(s))

    return models


async def _run_models_for_text(
    models: list[SentimentModel],
    text: str,
) -> dict[str, dict[str, Any]]:
    async def one(m: SentimentModel) -> tuple[str, dict[str, Any]]:
        try:
            r = await m.analyze(text)
            return m.name, r
        except Exception as e:
            logger.exception("Unexpected failure for model %s: %s", m.name, e)
            from app.models.schemas import empty_fallback

            fb = empty_fallback(f"unexpected: {e!s}", str(e))
            fb["latency_ms"] = 0.0
            fb["raw_response"] = ""
            fb["error"] = str(e)
            return m.name, fb

    pairs = await asyncio.gather(*(one(m) for m in models))
    return {k: v for k, v in pairs}


async def _persist_result(
    coll: AsyncIOMotorCollection[Any],
    comment_id: str,
    text: str,
    experiment_id: str,
    results: dict[str, dict[str, Any]],
) -> None:
    majority = compute_majority_label(results)
    truth = str(majority.get("overall_sentiment", "neutral"))
    model_agreement = build_model_agreement(results, truth)

    doc = {
        "comment_id": comment_id,
        "text": text,
        "experiment_id": experiment_id,
        "results": results,
        "majority_label": majority,
        "model_agreement": model_agreement,
    }
    await coll.update_one(
        {"comment_id": comment_id, "experiment_id": experiment_id},
        {"$set": doc},
        upsert=True,
    )


async def run_experiment(
    experiment_id: str | None = None,
    *,
    sample_limit: int | None = None,
    settings: Settings | None = None,
) -> str:
    """
    Sample comments, run all models per comment (bounded concurrency),
    and upsert one document per comment into `RESULTS_COLLECTION`.
    """
    s = settings or get_settings()
    exp_id = experiment_id or str(uuid.uuid4())
    limit = sample_limit if sample_limit is not None else s.sample_limit

    logger.info("Starting experiment_id=%s sample_limit=%s", exp_id, limit)

    comments = await sample_random_comments(limit=limit, settings=s)
    models = build_models(s)

    coll = get_results_collection(s)

    sem = asyncio.Semaphore(max(1, s.concurrent_comments))

    async def process_comment(doc: dict[str, Any]) -> None:
        async with sem:
            cid = comment_id_from_doc(doc)
            text = comment_text_from_doc(doc)
            results = await _run_models_for_text(models, text)
            await _persist_result(coll, cid, text, exp_id, results)

    tasks = [asyncio.create_task(process_comment(c)) for c in comments]
    for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="comments"):
        try:
            await t
        except Exception as e:
            logger.exception("Comment task failed: %s", e)

    logger.info("Experiment %s finished (%s comments).", exp_id, len(comments))
    return exp_id
