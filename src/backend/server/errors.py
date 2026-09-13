"""Map exceptions to stable client error codes. Never send str(exc): details stay in the server log."""

from __future__ import annotations

import httpx
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langgraph.errors import NodeTimeoutError
from qdrant_client.http.exceptions import ApiException

from keeper.nodes import EmptyGeneration

MESSAGES = {
    "empty_generation": "The Keeper returned an empty reply.",
    "timeout": "The model took too long to answer.",
    "rate_limited": "The model provider is rate limiting requests.",
    "upstream_unavailable": "The model provider is unavailable.",
    "internal_error": "Something went wrong on the server.",
}


def error_event(code: str, retryable: bool) -> dict:
    return {"code": code, "retryable": retryable, "message": MESSAGES.get(code, code)}


def map_error(exc: BaseException) -> dict:
    if isinstance(exc, EmptyGeneration):
        return error_event("empty_generation", True)
    if isinstance(exc, (NodeTimeoutError, ModelTimeoutError, TimeoutError, httpx.TimeoutException)):
        return error_event("timeout", True)
    if isinstance(exc, ModelRateLimitError):
        return error_event("rate_limited", True)
    if isinstance(exc, (ModelAPIError, ModelConnectionError, httpx.TransportError, ConnectionError, ApiException)):
        return error_event("upstream_unavailable", True)
    return error_event("internal_error", False)
