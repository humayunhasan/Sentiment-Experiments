"""Random sampling from the configured comments collection."""

from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import OperationFailure

from app.config import Settings, get_settings
from app.db import get_comments_collection

logger = logging.getLogger(__name__)


async def sample_random_comments(
    limit: int = 10000,
    *,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Sample up to `limit` documents using MongoDB $sample on `COMMENTS_COLLECTION`."""
    s = settings or get_settings()
    coll: AsyncIOMotorCollection[Any] = get_comments_collection(s)
    pipeline: list[dict[str, Any]] = [{"$sample": {"size": limit}}]
    cursor = coll.aggregate(pipeline)
    try:
        out = await cursor.to_list(length=limit)
    except OperationFailure as e:
        if e.code in (13, 18):
            raise RuntimeError(
                "MongoDB rejected the request (authentication required or invalid "
                "credentials). Use a MONGO_URI that includes a username and password "
                "(for example "
                "'mongodb://USER:PASSWORD@host:27017/?authSource=admin'), or set "
                "MONGO_USERNAME and MONGO_PASSWORD (and optionally MONGO_AUTH_SOURCE) "
                "alongside a host-only MONGO_URI."
            ) from e
        raise
    logger.info(
        "sample_random_comments: collection=%s retrieved %s documents (limit=%s)",
        s.comments_collection,
        len(out),
        limit,
    )
    return out
