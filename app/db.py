"""MongoDB async client helpers (Motor)."""

from __future__ import annotations

from typing import Any

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from app.config import Settings, get_settings

_client: AsyncIOMotorClient[Any] | None = None


def get_client(settings: Settings | None = None) -> AsyncIOMotorClient[Any]:
    global _client
    if _client is None:
        s = settings or get_settings()
        _client = AsyncIOMotorClient(
            s.mongo_uri,
            serverSelectionTimeoutMS=30000,
        )
    return _client


def get_db(settings: Settings | None = None) -> AsyncIOMotorDatabase[Any]:
    s = settings or get_settings()
    return get_client(s)[s.db_name]


def get_comments_collection(
    settings: Settings | None = None,
) -> AsyncIOMotorCollection[Any]:
    """Motor collection for source comments (`COMMENTS_COLLECTION`)."""
    s = settings or get_settings()
    return get_db(s)[s.comments_collection]


def get_results_collection(
    settings: Settings | None = None,
) -> AsyncIOMotorCollection[Any]:
    """Motor collection for experiment outputs (`RESULTS_COLLECTION`)."""
    s = settings or get_settings()
    return get_db(s)[s.results_collection]


async def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
