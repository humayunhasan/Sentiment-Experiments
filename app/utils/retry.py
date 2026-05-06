"""Retry helpers using tenacity (supports async functions in tenacity 8+)."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Network / transient errors common to HTTP clients
DEFAULT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    OPENAI_RETRY = (APIConnectionError, RateLimitError, APITimeoutError)
except Exception:  # pragma: no cover
    OPENAI_RETRY = tuple()

try:
    import httpx

    HTTPX_RETRY = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
except Exception:  # pragma: no cover
    HTTPX_RETRY = tuple()

try:
    import asyncio as _asyncio

    _ASYNC_TIMEOUT: tuple[type[BaseException], ...] = (_asyncio.TimeoutError,)
except Exception:  # pragma: no cover
    _ASYNC_TIMEOUT = tuple()

MERGED_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    *DEFAULT_RETRY_EXCEPTIONS,
    *OPENAI_RETRY,
    *HTTPX_RETRY,
    *_ASYNC_TIMEOUT,
)


def llm_retry(
    *,
    attempts: int = 4,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exceptions: tuple[type[BaseException], ...] | None = None,
):
    """Retry decorator for LLM HTTP calls."""
    exc = exceptions if exceptions is not None else MERGED_RETRY_EXCEPTIONS
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        retry=retry_if_exception_type(exc),
        reraise=True,
    )


__all__ = ["llm_retry", "DEFAULT_RETRY_EXCEPTIONS", "MERGED_RETRY_EXCEPTIONS"]
