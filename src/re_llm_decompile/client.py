"""HTTP client for OpenAI-compatible chat-completions APIs.

Used by the MCP tools. Pure httpx + tenacity, no SDK lock-in. Works
with vLLM, Ollama (``/v1``), OpenAI, and any compatible endpoint.

Model selection (Cycle 1 fix, T1.2): the ``LLM_DECOMPILE_MODEL`` env var
is preferred; if absent or unknown to the endpoint, ``resolve_model()``
queries ``/v1/models`` and picks the first non-embed / non-moe model.
The result is cached for the process lifetime. The chat path
automatically retries on the fallback model if the configured one
404s (decompile_function surfaces the fallback as a ``WARNING``).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ── Configuration ──────────────────────────────────────────────────────


# A4 fix (v2.8.0): structured exception so callers can surface
# HTTP status / endpoint URL / response body to the agent instead
# of catching a raw httpx.HTTPStatusError and losing context.
class LLMCallError(RuntimeError):
    """Raised when the LLM endpoint returns a non-2xx or refuses the call.

    Attributes:
        status_code: HTTP status (or 0 for network / timeout errors)
        url: full URL that was hit
        body: response body (truncated) or the underlying exception repr
        is_cloud_model: True if the resolved model name contains ``:cloud``
            (or other known cloud markers); the most common cause of 403
            in this server is a cloud-bridge model that refuses
            programmatic calls
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        url: str = "",
        body: str = "",
        is_cloud_model: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = body
        self.is_cloud_model = is_cloud_model


def is_cloud_model(name: str) -> bool:
    """Return True if *name* names a cloud-backed model.

    Heuristic: any model whose name contains ``:cloud`` or starts with
    a known cloud-provider prefix (``openrouter/``, ``anthropic/``,
    ``openai/``). The Ollama cloud bridge serves models like
    ``deepseek-v4-flash:cloud`` that refuse programmatic / unauthenticated
    chat completions (HTTP 403) and are the dominant cause of the A4
    silent failure observed in r03-stress.
    """
    if not name:
        return False
    if ":cloud" in name:
        return True
    for prefix in ("openrouter/", "anthropic/", "openai/"):
        if name.startswith(prefix):
            return True
    return False


def get_endpoint() -> str:
    return os.environ.get("LLM_DECOMPILE_ENDPOINT", "http://localhost:11434/v1").rstrip("/")


def get_model() -> str:
    """Return the user-requested model name (env-var default).

    Does NOT do the fallback resolution — callers that want
    auto-fallback should call :func:`resolve_model` instead. Use
    :func:`get_model` only when you specifically want the raw
    env-var value (e.g. for the check_endpoint response shape).

    Cycle 2 fix: default changed from ``llm4decompile`` to
    ``deepseek-v4-flash:cloud`` (the cloud model that the user's
    Ollama endpoint actually serves per run 2026-06-06-r01).
    ``llm4decompile`` is not in the registry; the auto-fallback
    kicked in and picked ``llama3.2:3b`` which produced HTTP 500
    on decompile prompts. Use the cloud model directly to avoid the
    fallback path entirely.

    v2.8.0 (A4): the cloud default itself proved bad — Ollama's cloud
    bridge returns HTTP 403 to programmatic chat completions (per
    stress test agent reports). Set
    ``LLM_DECOMPILE_DISABLE_CLOUD=1`` to skip cloud models in the
    auto-fallback path, OR set ``LLM_DECOMPILE_MODEL`` explicitly to
    a local model name (``llama3.2:3b``, ``llm4decompile``, etc.).
    The default value is preserved for backward compat; the truthful
    health-check now surfaces WARN when this default is resolved.
    """
    return os.environ.get("LLM_DECOMPILE_MODEL", "deepseek-v4-flash:cloud")


def get_api_key() -> str:
    return os.environ.get("LLM_DECOMPILE_API_KEY", "")


def cloud_disabled() -> bool:
    """Return True when the user has set ``LLM_DECOMPILE_DISABLE_CLOUD=1``
    (or any truthy value). When True, cloud-backed models are skipped
    during auto-fallback resolution.
    """
    return os.environ.get("LLM_DECOMPILE_DISABLE_CLOUD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ── HTTP transport ────────────────────────────────────────────────────


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    key = get_api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _post(path: str, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    url = f"{get_endpoint()}{path}"
    with httpx.Client(timeout=timeout, headers=_headers()) as client:
        r = client.post(url, json=body)
        # A4 fix (v2.8.0): raise structured LLMCallError instead of
        # httpx.HTTPStatusError so callers can surface a useful
        # remediation hint (especially for the 403 cloud-bridge case).
        if not (200 <= r.status_code < 300):
            model_name = body.get("model", "") if isinstance(body, dict) else ""
            body_text = ""
            try:
                body_text = r.text[:512] if r.text else ""
            except Exception:  # noqa: BLE001
                body_text = "<unreadable response body>"
            raise LLMCallError(
                f"LLM endpoint returned HTTP {r.status_code} for model "
                f"{model_name!r} at {url}",
                status_code=r.status_code,
                url=url,
                body=body_text,
                is_cloud_model=is_cloud_model(model_name),
            )
        return r.json()


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _get(path: str, timeout: float = 10.0) -> dict[str, Any]:
    """GET against the configured endpoint. Used for ``/v1/models`` (OpenAI
    compat) and ``/api/tags`` (native Ollama)."""
    url = f"{get_endpoint()}{path}"
    with httpx.Client(timeout=timeout, headers=_headers()) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


# ── Public API ─────────────────────────────────────────────────────────


def list_models() -> list[str]:
    """Return the list of model names the endpoint serves.

    OpenAI-compatible endpoints expose ``GET /v1/models``. Native Ollama
    (without the ``/v1`` shim) exposes ``GET /api/tags`` instead. We try
    OpenAI first, then fall back to Ollama's native endpoint.

    Previously this POSTed to ``/v1/models`` — both endpoints reject POST
    with 405 Method Not Allowed, which is what tripped the user-reported
    ``check_endpoint`` failure.
    """
    try:
        data = _get("/models")
        models = data.get("data", [])
        if isinstance(models, list):
            return [m.get("id", str(m)) for m in models if isinstance(m, dict)]
    except httpx.HTTPStatusError:
        pass
    try:
        data = _get("/api/tags")
        models = data.get("models", [])
        if isinstance(models, list):
            return [m.get("name", str(m)) for m in models if isinstance(m, dict)]
    except Exception:  # noqa: BLE001
        return []
    return []


# ── Model auto-resolution (Cycle 1 / T1.2) ─────────────────────────────


# Token substrings that indicate an embedding / non-LLM model. If the
# configured `LLM_DECOMPILE_MODEL` is one of these, the fallback path
# is mandatory (and so is anything else we'd auto-pick).
_EMBED_TOKENS = ("embed", "embedding", "-moe", "moe:")


# Cycle 2 fix: fidelity-aware fallback preference list. The prior
# implementation picked the first non-embed model, which on a stock
# Ollama install is `llama3.2:3b` — a 3B general-purpose chat model
# with poor decompilation fidelity. The new preference list
# prefers code-specialized / larger-parameter models when available,
# falling back to general-purpose models only if no coder is present.
_FALLBACK_PREFERENCE = (
    "deepseek-coder",
    "qwen2.5-coder",
    "qwen-coder",
    "codellama",
    "codeqwen",
    "starcoder",
    "wizardcoder",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-coder-v2",
    "deepseek-r1",
    "qwen2.5",
    "qwen2",
    "llama3.1",
    "llama3.2",
    "llama3",
    "mistral",
    "gemma",
    "phi",
)


def _is_chat_model(name: str) -> bool:
    """Return True if *name* looks like a chat / instruct model.

    Heuristic: exclude names containing embed / moe tokens. Anything
    else is treated as chat. This is intentionally permissive — we'd
    rather fall back to a non-ideal chat model than to fail.
    """
    n = name.lower()
    return not any(tok in n for tok in _EMBED_TOKENS)


def _pick_fallback_model(available: list[str]) -> str | None:
    """Return the highest-fidelity non-embed model in *available*.

    Cycle 2 fix: the prior implementation returned the first
    non-embed model, which on most stock Ollama installs is
    `llama3.2:3b` — a 3B general-purpose chat model with poor
    decompilation fidelity. The new implementation scores each
    candidate by the order in ``_FALLBACK_PREFERENCE`` (code-specialized
    first, then larger / coder-flavored chat models, then general
    purpose) and returns the highest-scoring one.

    Returns None if every available model looks like an embed. The
    caller (chat path) decides what to do with None — typically
    surface a clear ``ERROR`` so the user can fix
    ``LLM_DECOMPILE_MODEL`` or pull a chat model.
    """
    available_set = {m.lower(): m for m in available if _is_chat_model(m)}
    if not available_set:
        return None
    for stem in _FALLBACK_PREFERENCE:
        # Match the stem as a prefix of the available model name (or
        # the available model name as a prefix of the stem) so e.g.
        # "deepseek-v4-flash:cloud" matches "deepseek-v4-flash".
        for low, original in available_set.items():
            if low.startswith(stem) or stem.startswith(low.split(":")[0]):
                return original
    # No preferred match — fall back to the first chat model
    return next(iter(available_set.values()))


# Module-level cache so the model list is only fetched once per
# process lifetime. Cleared by `clear_model_cache()` (for tests).
_MODEL_CACHE: dict[str, Any] = {
    "configured": None,    # the user-requested model (raw env value)
    "available": None,     # list of model names the endpoint serves
    "resolved": None,      # the model chat() will actually use
    "fallback_used": None, # True if resolved != configured
    "endpoint_reachable": None,
}


def clear_model_cache() -> None:
    """Reset the model-resolution cache. Test-only; not used in production paths."""
    _MODEL_CACHE.update({
        "configured": None, "available": None, "resolved": None,
        "fallback_used": None, "endpoint_reachable": None,
    })


def resolve_model() -> str:
    """Return the model name chat() should use, with auto-fallback.

    Logic:
      1. Read the configured model from ``LLM_DECOMPILE_MODEL`` (or
         the default ``llm4decompile``).
      2. Try to fetch the endpoint's ``/v1/models``. If the configured
         model is in the list, use it as-is. If the endpoint is
         unreachable, fall back to the configured model name
         (the chat path will 404 and surface a clear error).
      3. If the configured model is NOT in the list, OR the
         configured model looks like an embedding model, pick the
         first non-embed model from the list.
      4. If no model resolves (no available models, all are embeds),
         return the configured model name verbatim so the chat
         path can produce a structured ``ERROR`` rather than a
         silent failure.

    The result is cached for the process lifetime. Override
    ``LLM_DECOMPILE_MODEL`` at process start to switch models.
    """
    if _MODEL_CACHE["resolved"] is not None:
        return _MODEL_CACHE["resolved"]

    configured = get_model()
    _MODEL_CACHE["configured"] = configured

    available = list_models()
    _MODEL_CACHE["available"] = available
    _MODEL_CACHE["endpoint_reachable"] = bool(available) or _get_reachable()

    if not available:
        # Endpoint unreachable (e.g. Ollama not running) — fall back to
        # the configured name. The chat() call will 404 / refuse and
        # surface a clear error to the agent.
        _MODEL_CACHE["resolved"] = configured
        _MODEL_CACHE["fallback_used"] = False
        return configured

    if configured in available and _is_chat_model(configured):
        # A4 fix (v2.8.0): even if the configured model is in the
        # registry, skip it when cloud is disabled and the model is
        # cloud-backed (almost guaranteed to 403).
        if cloud_disabled() and is_cloud_model(configured):
            fallback = _pick_fallback_model(
                [m for m in available if not is_cloud_model(m)]
            )
            if fallback is not None:
                _MODEL_CACHE["resolved"] = fallback
                _MODEL_CACHE["fallback_used"] = True
                return fallback
        _MODEL_CACHE["resolved"] = configured
        _MODEL_CACHE["fallback_used"] = False
        return configured

    # When cloud is disabled, filter cloud models out of the candidate set.
    candidate_pool = available
    if cloud_disabled():
        candidate_pool = [m for m in available if not is_cloud_model(m)]
    fallback = _pick_fallback_model(candidate_pool)
    if fallback is not None:
        _MODEL_CACHE["resolved"] = fallback
        _MODEL_CACHE["fallback_used"] = True
        return fallback

    # No chat model available — return configured so chat() surfaces
    # the explicit error rather than silently using a non-chat model.
    _MODEL_CACHE["resolved"] = configured
    _MODEL_CACHE["fallback_used"] = True
    return configured


def get_resolution_state() -> dict[str, Any]:
    """Return the cached model-resolution state for diagnostic surfaces.

    Always populated (calls ``resolve_model()`` if not yet cached).
    """
    if _MODEL_CACHE["resolved"] is None:
        resolve_model()
    return dict(_MODEL_CACHE)


def _get_reachable() -> bool:
    """Return True if the configured endpoint is reachable, False otherwise.

    Best-effort: tries a short GET against ``/models`` and reports
    whether it returned 2xx. Used by ``resolve_model()`` to populate
    the ``endpoint_reachable`` field; never raises.
    """
    try:
        url = f"{get_endpoint()}/models"
        with httpx.Client(timeout=3.0) as c:
            r = c.get(url)
            return 200 <= r.status_code < 300
    except Exception:  # noqa: BLE001
        return False


def chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Send a chat completion request and return the assistant text.

    Uses the OpenAI-compatible ``/v1/chat/completions`` schema.

    If *model* is None, resolves the model via :func:`resolve_model`
    (configured + auto-fallback). If *model* is explicitly set, that
    value is used verbatim (no auto-fallback).
    """
    chosen = model or resolve_model()
    body = {
        "model": chosen,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = _post("/chat/completions", body, timeout=timeout)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"no choices in response: {data}")
    msg = choices[0].get("message", {})
    return msg.get("content", "").strip()
