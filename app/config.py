"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

from dotenv import load_dotenv

# Load .env from project root (parent of sentiment_exp/) or cwd
_root = Path(__file__).resolve().parent.parent
_workspace = _root.parent
_env_paths = [
    _root / ".env",
    _workspace / ".env",
    Path.cwd() / ".env",
]
for _p in _env_paths:
    if _p.is_file():
        load_dotenv(_p)
        break
else:
    load_dotenv()


def _get(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return v


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    raw = _get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _resolve_mongo_uri(
    uri: str,
    username: str | None,
    password: str | None,
    auth_source: str | None,
) -> str:
    """
    If MONGO_USERNAME and MONGO_PASSWORD are set and the URI has no credentials,
    inject userinfo and optionally append authSource to the query string.
    """
    if not username or not password:
        return uri
    parsed = urlparse(uri)
    if parsed.username is not None:
        return uri
    if parsed.netloc and "@" in parsed.netloc:
        return uri

    user_q = quote_plus(username, safe="")
    pass_q = quote_plus(password, safe="")
    netloc = f"{user_q}:{pass_q}@{parsed.netloc}"

    q_pairs = list(parse_qsl(parsed.query, keep_blank_values=True))
    keys_lower = {k.lower() for k, _ in q_pairs}
    if auth_source and "authsource" not in keys_lower:
        q_pairs.append(("authSource", auth_source))
    new_query = urlencode(q_pairs)

    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        )
    )


@dataclass(frozen=True)
class Settings:
    mongo_uri: str
    db_name: str
    comments_collection: str
    results_collection: str

    openai_api_key: str | None
    gemini_api_key: str | None
    anthropic_api_key: str | None
    deepseek_api_key: str | None
    kimi_api_key: str | None

    openai_model: str
    gemini_model: str
    anthropic_model: str
    deepseek_model: str
    deepseek_base_url: str
    kimi_model: str
    kimi_base_url: str

    enable_anthropic: bool
    enable_hf_tabularisai: bool
    hf_tabularisai_model: str

    sample_limit: int
    concurrent_comments: int
    request_timeout_s: float


def get_settings() -> Settings:
    mongo_uri = _get("MONGO_URI")
    db_name = _get("DB_NAME", "sentiment")
    if not mongo_uri or not db_name:
        raise RuntimeError("MONGO_URI and DB_NAME must be set in environment.")

    mongo_user = _get("MONGO_USERNAME")
    mongo_password = _get("MONGO_PASSWORD")
    mongo_auth_source = _get("MONGO_AUTH_SOURCE", "admin")
    mongo_uri = _resolve_mongo_uri(
        mongo_uri,
        mongo_user,
        mongo_password,
        mongo_auth_source,
    )

    return Settings(
        mongo_uri=mongo_uri,
        db_name=db_name,
        comments_collection=_get("COMMENTS_COLLECTION", "youtube_comments"),
        results_collection=_get(
            "RESULTS_COLLECTION", "sentiment_experiment_results"
        ),
        openai_api_key=_get("OPENAI_API_KEY"),
        gemini_api_key=_get("GEMINI_API_KEY"),
        anthropic_api_key=_get("ANTHROPIC_API_KEY"),
        deepseek_api_key=_get("DEEPSEEK_API_KEY"),
        kimi_api_key=_get("KIMI_API_KEY"),
        openai_model=_get("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_model=_get("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        anthropic_model=_get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        kimi_model=_get("KIMI_MODEL", "kimi-k2.6"),
        kimi_base_url=_get("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
        enable_anthropic=_get_bool("ENABLE_ANTHROPIC", False),
        enable_hf_tabularisai=_get_bool("ENABLE_HF_TABULARISAI", True),
        hf_tabularisai_model=_get(
            "HF_TABULARISAI_MODEL",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        ),
        sample_limit=_get_int("SAMPLE_LIMIT", 10000),
        concurrent_comments=_get_int("CONCURRENT_COMMENTS", 5),
        request_timeout_s=_get_float("REQUEST_TIMEOUT_S", 120.0),
    )
