"""Send direct HTTP requests to the configured LLM provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import ssl
from typing import TYPE_CHECKING
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REQUEST_TIMEOUT_S = 120.0
# OpenAI Responses does not consume a temperature knob in our current request path,
# so keep Anthropic's setting internal instead of exposing it on the search API.
DEFAULT_ANTHROPIC_TEMPERATURE = 0.3
# Legacy `budget_tokens` presets (1024 = Anthropic's hard minimum). Newer models
# self-pick via adaptive thinking — see `_supports_anthropic_adaptive`.
_ANTHROPIC_THINKING_BUDGET_BY_EFFORT = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "max": 24000,
}
_VALID_EFFORT_LEVELS = frozenset({"none", "low", "medium", "high", "max"})
# Per-family minimum (major, minor) for adaptive thinking. Required on Opus 4.7+
# (manual budget_tokens returns HTTP 400). Below-minimum / unlisted → legacy path.
_ANTHROPIC_ADAPTIVE_MIN_VERSIONS: dict[str, tuple[int, int]] = {
    "opus": (4, 5),
    "sonnet": (4, 6),
}
# Per-family minimum (major, minor) at/after which the `temperature` knob is
# deprecated and rejected (HTTP 400 on the API, ValidationException on Bedrock).
# Opus 4.7+ rejects it. Below-minimum / unlisted families still accept it.
_ANTHROPIC_TEMPERATURE_DEPRECATED_MIN_VERSIONS: dict[str, tuple[int, int]] = {
    "opus": (4, 7),
}
# Minor capped at 2 digits + non-digit lookahead so 8-digit date suffixes (e.g.
# `claude-opus-4-20250514`) don't get mis-parsed as the minor version.
_ANTHROPIC_MODEL_VERSION_RE = re.compile(
    r"^claude-([a-z]+)-(\d+)(?:-(\d{1,2})(?=\D|$))?"
)
# Models that accept OpenAI's `xhigh` effort. Others reject it, so "max" only
# maps to "xhigh" here; elsewhere "max" → "high".
_OPENAI_XHIGH_MODELS = frozenset({"gpt-5.1-codex-max", "gpt-5.4", "gpt-5.5"})

_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "openai": "openai_responses",
    "openai_responses": "openai_responses",
    "openai-responses": "openai_responses",
    "bedrock": "bedrock",
    "aws_bedrock": "bedrock",
    "aws-bedrock": "bedrock",
    "vertex": "vertex",
    "vertex_anthropic": "vertex",
    "vertex-anthropic": "vertex",
    "google_vertex": "vertex",
}


def normalize_provider(provider: str) -> str:
    """Canonicalize user-facing provider names to internal transport IDs."""
    normalized = provider.strip().lower()
    if resolved := _PROVIDER_ALIASES.get(normalized):
        return resolved
    raise ValueError(
        f"Unsupported LLM provider {provider!r}. "
        "Valid providers are: anthropic, openai, openai_responses, bedrock, vertex."
    )


def infer_provider(model: str, provider: str | None = None) -> str:
    """Guess the transport from the model name unless the caller overrides it."""
    if provider is not None:
        return normalize_provider(provider)
    normalized = model.lower()
    # Bedrock and Vertex serve the same Anthropic model ids as the direct API, so
    # they must be opted into explicitly with a `bedrock/` / `vertex/` prefix
    # (e.g. `vertex/claude-sonnet-4-5`) or via HELION_LLM_PROVIDER.
    if normalized.startswith("bedrock/"):
        return "bedrock"
    if normalized.startswith("vertex/"):
        return "vertex"
    if normalized.startswith(("claude", "anthropic/")):
        return "anthropic"
    if normalized.startswith(
        ("gpt-", "chatgpt-", "codex", "o1", "o3", "o4", "openai/")
    ):
        return "openai_responses"
    return "unsupported"


def strip_provider_prefix(model: str) -> str:
    """Remove a provider prefix before sending the model name to the API."""
    for prefix in ("anthropic/", "openai/", "bedrock/", "vertex/"):
        if model.startswith(prefix):
            return model.removeprefix(prefix)
    return model


def split_system_messages(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Hoist system prompts into the format expected by provider adapters."""
    system_messages = [
        message["content"] for message in messages if message["role"] == "system"
    ]
    non_system = [message for message in messages if message["role"] != "system"]
    return "\n\n".join(system_messages), non_system


def responses_input_from_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Convert chat history into the OpenAI Responses input schema."""
    payload: list[dict[str, object]] = []
    for message in messages:
        role = "developer" if message["role"] == "system" else message["role"]
        content_type = "output_text" if role == "assistant" else "input_text"
        payload.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": message["content"]}],
            }
        )
    return payload


def anthropic_messages_from_history(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Convert chat history into the Anthropic Messages schema."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]


def normalize_effort_level(effort_level: str | None) -> str | None:
    """Normalize the optional model effort-level knob."""
    from ...runtime.settings import _FALSE_LITERALS

    if effort_level is None:
        return None
    normalized = effort_level.strip().lower()
    if normalized in _FALSE_LITERALS:
        return "none"
    if normalized not in _VALID_EFFORT_LEVELS:
        raise ValueError(
            f"Unsupported LLM effort level {effort_level!r}. "
            "Valid values are: none, low, medium, high, max."
        )
    return normalized


def _openai_effort_level(effort_level: str | None, model: str) -> str | None:
    normalized = normalize_effort_level(effort_level)
    if normalized in {None, "none"}:
        return None
    if normalized == "max":
        return (
            "xhigh" if strip_provider_prefix(model) in _OPENAI_XHIGH_MODELS else "high"
        )
    return normalized


def _anthropic_thinking_budget_tokens(effort_level: str | None) -> int | None:
    normalized = normalize_effort_level(effort_level)
    if normalized in {None, "none"}:
        return None
    return _ANTHROPIC_THINKING_BUDGET_BY_EFFORT[normalized]


def _anthropic_version_at_least(
    model: str, minimums: dict[str, tuple[int, int]]
) -> bool:
    """Return True if `model`'s (family, version) meets the per-family minimum.

    Tolerates Bedrock-style IDs (e.g. `us.anthropic.claude-opus-4-8`) by matching
    the `claude-...` segment anywhere in the string, not just at the start.
    """
    normalized = model.lower()
    # Bedrock prefixes the model with a region/partition + `anthropic.`; drop
    # everything up to and including that so the `claude-...` regex can match.
    if "anthropic." in normalized:
        normalized = normalized.rsplit("anthropic.", 1)[1]
    match = _ANTHROPIC_MODEL_VERSION_RE.match(normalized)
    if match is None:
        return False
    family, major_str, minor_str = match.groups()
    minimum = minimums.get(family)
    if minimum is None:
        return False
    return (int(major_str), int(minor_str) if minor_str else 0) >= minimum


def _supports_anthropic_adaptive(model: str) -> bool:
    return _anthropic_version_at_least(model, _ANTHROPIC_ADAPTIVE_MIN_VERSIONS)


def _temperature_deprecated(model: str) -> bool:
    """Whether the model rejects the `temperature` knob (Opus 4.7+)."""
    return _anthropic_version_at_least(
        model, _ANTHROPIC_TEMPERATURE_DEPRECATED_MIN_VERSIONS
    )


def _anthropic_max_tokens(
    max_output_tokens: int,
    thinking_budget_tokens: int | None,
) -> int:
    if thinking_budget_tokens is None:
        return max_output_tokens
    return thinking_budget_tokens + max_output_tokens


def _extract_text_content_items(content: object) -> list[str]:
    """Collect plain-text content blocks from a provider response payload."""
    if not isinstance(content, list):
        return []
    return [
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") in {"text", "output_text"}
        and isinstance(item.get("text"), str)
    ]


def extract_openai_response_text(response: dict[str, object]) -> str:
    """Extract concatenated text from an OpenAI Responses payload."""
    output = response.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            texts.extend(_extract_text_content_items(item.get("content")))
        if texts:
            return "".join(texts)
    raise exc.InvalidResponseSchema(f"Unexpected OpenAI responses payload: {response}")


def extract_anthropic_text(response: dict[str, object]) -> str:
    """Extract concatenated text from an Anthropic Messages payload."""
    if texts := _extract_text_content_items(response.get("content")):
        return "".join(texts)
    raise exc.InvalidResponseSchema(f"Unexpected Anthropic payload: {response}")


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token counts (§4, §6.3); ``None`` marks a missing field.

    Missing fields are never substituted with a tokenizer estimate in the primary
    token claim.
    """

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ProviderResult:
    """One provider response with non-secret replay and cache metadata."""

    text: str
    usage: TokenUsage
    request_id: str | None = None
    response_id: str | None = None
    cache_state: str = "unknown"


@dataclass(frozen=True)
class ProviderMetadata:
    """Non-secret identity and provider-reported cache state for one request."""

    request_id: str | None
    response_id: str | None
    cache_state: str


@dataclass(frozen=True)
class ProviderReplayRecord:
    """Credential-free provider request and response data for frozen replay."""

    provider: str
    model: str
    api_base: str | None
    messages: tuple[dict[str, str], ...]
    max_output_tokens: int
    request_timeout_s: float
    effort_level: str | None
    fast_mode: bool
    request_id: str
    response_id: str | None
    cache_state: str | None
    response_text: str | None
    usage: TokenUsage
    error_type: str | None


def provider_replay_request_id(
    provider: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    request_timeout_s: float,
    effort_level: str | None,
    fast_mode: bool,
    api_base: str | None = None,
    extra_headers_sha256: str | None = None,
) -> str:
    """Return the stable logical identity used when a provider returns no ID."""
    return _canonical_sha256(
        {
            "provider": provider,
            "model": model,
            "api_base": api_base,
            "messages": messages,
            "max_output_tokens": max_output_tokens,
            "request_timeout_s": request_timeout_s,
            "effort_level": effort_level,
            "fast_mode": fast_mode,
            "extra_headers_sha256": extra_headers_sha256,
        }
    )


def provider_replay_response_hash(response_text: str) -> str:
    """Return the stable content identity for a successful replay response."""
    return _canonical_sha256({"response_text": response_text})


def provider_replay_extra_headers_hash() -> str | None:
    """Return the credential-free identity of configured extra headers."""
    extra_headers = _extra_headers()
    if not extra_headers:
        return None
    return hashlib.sha256(
        json.dumps(extra_headers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def append_provider_replay_record(record: ProviderReplayRecord) -> Path | None:
    """Append one canonical provider replay row when a log path is configured."""
    raw_path = os.environ.get("HELION_RAG_PROVIDER_REPLAY_LOG")
    if raw_path is None:
        return None
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_headers_sha256 = provider_replay_extra_headers_hash()
    request_hash = provider_replay_request_id(
        record.provider,
        model=record.model,
        api_base=record.api_base,
        messages=list(record.messages),
        max_output_tokens=record.max_output_tokens,
        request_timeout_s=record.request_timeout_s,
        effort_level=record.effort_level,
        fast_mode=record.fast_mode,
        extra_headers_sha256=extra_headers_sha256,
    )
    payload = {
        "provider": record.provider,
        "model": record.model,
        "api_base": record.api_base,
        "extra_headers_sha256": extra_headers_sha256,
        "messages": list(record.messages),
        "max_output_tokens": record.max_output_tokens,
        "request_timeout_s": record.request_timeout_s,
        "effort_level": record.effort_level,
        "fast_mode": record.fast_mode,
        "request_id": record.request_id,
        "request_hash": request_hash,
        "response_id": record.response_id,
        "response_hash": (
            provider_replay_response_hash(record.response_text)
            if record.response_text is not None
            else None
        ),
        "cache_state": record.cache_state,
        "response_text": record.response_text,
        "usage": {
            "input_tokens": record.usage.input_tokens,
            "cached_input_tokens": record.usage.cached_input_tokens,
            "output_tokens": record.usage.output_tokens,
            "reasoning_tokens": record.usage.reasoning_tokens,
        },
        "error_type": record.error_type,
    }
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = memoryview(line + b"\n")
        while data:
            written = os.write(fd, data)
            if written == 0:
                raise OSError("provider replay log write made no progress")
            data = data[written:]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return path


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def extract_openai_usage(response: dict[str, object]) -> TokenUsage:
    """Read provider-reported usage from an OpenAI Responses payload."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    return TokenUsage(
        input_tokens=_int_or_none(usage.get("input_tokens")),
        cached_input_tokens=_int_or_none(
            input_details.get("cached_tokens")
            if isinstance(input_details, dict)
            else None
        ),
        output_tokens=_int_or_none(usage.get("output_tokens")),
        reasoning_tokens=_int_or_none(
            output_details.get("reasoning_tokens")
            if isinstance(output_details, dict)
            else None
        ),
    )


def extract_anthropic_usage(response: dict[str, object]) -> TokenUsage:
    """Read provider-reported usage from an Anthropic Messages payload.

    Anthropic reports cached input as ``cache_read_input_tokens`` and folds
    thinking into ``output_tokens`` (no separate reasoning field).
    """
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=_int_or_none(usage.get("input_tokens")),
        cached_input_tokens=_int_or_none(usage.get("cache_read_input_tokens")),
        output_tokens=_int_or_none(usage.get("output_tokens")),
        reasoning_tokens=None,
    )


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _provider_metadata(
    request_payload: dict[str, Any],
    response: dict[str, object],
    usage: TokenUsage,
) -> ProviderMetadata:
    provider_request_id = response.get("request_id")
    provider_response_id = response.get("id", response.get("response_id"))
    cached_input_tokens = usage.cached_input_tokens
    cache_state = (
        "hit"
        if cached_input_tokens is not None and cached_input_tokens > 0
        else "miss"
        if cached_input_tokens == 0
        else "unknown"
    )
    return ProviderMetadata(
        request_id=(
            provider_request_id
            if isinstance(provider_request_id, str) and provider_request_id
            else _canonical_sha256(request_payload)
        ),
        response_id=(
            provider_response_id
            if isinstance(provider_response_id, str) and provider_response_id
            else _canonical_sha256(response)
        ),
        cache_state=cache_state,
    )


def _identity_request_payload(
    provider: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    effort_level: str | None,
    fast_mode: bool,
) -> dict[str, Any]:
    if provider in {"bedrock", "vertex"}:
        payload = _anthropic_payload(
            model, messages, max_output_tokens, effort_level, fast_mode
        )
        # Keep the logical model in the identity even though these transports put
        # it in the API call rather than the wire body.
        payload["anthropic_version"] = (
            _BEDROCK_ANTHROPIC_VERSION
            if provider == "bedrock"
            else _VERTEX_ANTHROPIC_VERSION
        )
        return payload
    return _build_provider_payload(
        provider,
        model=model,
        messages=messages,
        max_output_tokens=max_output_tokens,
        effort_level=effort_level,
        fast_mode=fast_mode,
    )


def _openai_payload(
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    effort_level: str | None,
    fast_mode: bool,
) -> dict[str, Any]:
    """Build an OpenAI Responses request payload."""
    # fast_mode is Anthropic-only; the kwarg is accepted for dispatch parity.
    system_prompt, input_messages = split_system_messages(messages)
    payload: dict[str, Any] = {
        "model": strip_provider_prefix(model),
        "input": responses_input_from_messages(input_messages),
        "max_output_tokens": max_output_tokens,
    }
    if (effort := _openai_effort_level(effort_level, model)) is not None:
        payload["reasoning"] = {"effort": effort}
    if system_prompt:
        payload["instructions"] = system_prompt
    return payload


def _anthropic_payload(
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    effort_level: str | None,
    fast_mode: bool,
) -> dict[str, Any]:
    """Build an Anthropic Messages request payload."""
    system_prompt, input_messages = split_system_messages(messages)
    normalized_model = strip_provider_prefix(model)
    normalized_effort = normalize_effort_level(effort_level)
    enable_thinking = normalized_effort not in {None, "none"}
    use_adaptive = enable_thinking and _supports_anthropic_adaptive(normalized_model)
    # Reserve max_tokens for both visible output AND thinking. Anthropic counts
    # thinking tokens against `max_tokens`; without this, adaptive thinking can
    # consume the entire budget on the encrypted CoT and produce no text.
    thinking_token_budget = (
        _anthropic_thinking_budget_tokens(effort_level) if enable_thinking else None
    )
    payload: dict[str, Any] = {
        "model": normalized_model,
        "messages": anthropic_messages_from_history(input_messages),
        "max_tokens": _anthropic_max_tokens(max_output_tokens, thinking_token_budget),
    }
    # Fast mode and extended thinking are orthogonal on the wire — Anthropic
    # accepts both — so we forward whichever knobs the user opted into.
    if fast_mode:
        payload["speed"] = "fast"
    if use_adaptive:
        # Adaptive thinking lets the model self-pick its budget within Anthropic's
        # cap for the chosen effort. Required on Opus 4.7 (manual budget_tokens 400s).
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": normalized_effort}
    elif thinking_token_budget is not None:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_token_budget,
        }
    # Claude Opus 4.7+, extended thinking, and fast mode each reject `temperature`.
    if (
        not enable_thinking
        and not fast_mode
        and not _temperature_deprecated(normalized_model)
    ):
        payload["temperature"] = DEFAULT_ANTHROPIC_TEMPERATURE
    if system_prompt:
        payload["system"] = system_prompt
    return payload


def _openai_headers(api_key: str, fast_mode: bool) -> dict[str, str]:
    """Build OpenAI-compatible auth headers."""
    # fast_mode is Anthropic-only; the kwarg is accepted for dispatch parity.
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _anthropic_headers(api_key: str, fast_mode: bool) -> dict[str, str]:
    """Build Anthropic Messages auth headers."""
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": api_key,
    }
    if fast_mode:
        # Opus 4.6/4.7 fast-mode beta. Direct Anthropic API only — Vertex strips
        # this header. Paired with the `speed: "fast"` body field in _anthropic_payload.
        headers["anthropic-beta"] = "fast-mode-2026-02-01"
    return headers


@dataclass(frozen=True)
class _ProviderConfig:
    """Provider-specific transport configuration."""

    endpoint: str
    default_api_base: str
    api_base_env_names: tuple[str, ...]
    api_key_env_names: tuple[str, ...]
    missing_api_key_error: str
    build_payload: Callable[
        [str, list[dict[str, str]], int, str | None, bool],
        dict[str, Any],
    ]
    build_headers: Callable[[str, bool], dict[str, str]]
    extract_text: Callable[[dict[str, object]], str]
    extract_usage: Callable[[dict[str, object]], TokenUsage]


_PROVIDER_CONFIGS = {
    "openai_responses": _ProviderConfig(
        endpoint="responses",
        default_api_base="https://api.openai.com",
        api_base_env_names=("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        api_key_env_names=("OPENAI_API_KEY",),
        missing_api_key_error=(
            "OpenAI-compatible model requested but no api_key, HELION_LLM_API_KEY, "
            "or OPENAI_API_KEY is set"
        ),
        build_payload=_openai_payload,
        build_headers=_openai_headers,
        extract_text=extract_openai_response_text,
        extract_usage=extract_openai_usage,
    ),
    "anthropic": _ProviderConfig(
        endpoint="messages",
        default_api_base="https://api.anthropic.com",
        api_base_env_names=("ANTHROPIC_BASE_URL",),
        api_key_env_names=("ANTHROPIC_API_KEY",),
        missing_api_key_error=(
            "Anthropic model requested but no api_key, HELION_LLM_API_KEY, "
            "or ANTHROPIC_API_KEY is set"
        ),
        build_payload=_anthropic_payload,
        build_headers=_anthropic_headers,
        extract_text=extract_anthropic_text,
        extract_usage=extract_anthropic_usage,
    ),
}


def _provider_config(provider: str) -> _ProviderConfig:
    """Return the provider-specific transport configuration."""
    normalized_provider = normalize_provider(provider)
    return _PROVIDER_CONFIGS[normalized_provider]


def _first_set_env(*names: str) -> str | None:
    """Return the first env var in the list that is present."""
    for name in names:
        if (value := os.environ.get(name)) is not None:
            return value
    return None


def _first_existing_path(*names: str) -> str | None:
    """Return the first configured path that exists on disk."""
    if (path := _first_set_env(*names)) is not None and os.path.exists(path):
        return path
    return None


def _resolve_api_base(provider: str, api_base: str | None) -> str:
    """Resolve the base URL from args, env vars, or provider defaults."""
    if api_base is not None:
        return api_base
    if (generic_api_base := os.environ.get("HELION_LLM_API_BASE")) is not None:
        return generic_api_base
    config = _provider_config(provider)
    return _first_set_env(*config.api_base_env_names) or config.default_api_base


def _resolve_api_key(provider: str, api_key: str | None) -> str:
    """Resolve the API key from args, env vars, or provider defaults."""
    if api_key is not None:
        return api_key
    if (generic_api_key := os.environ.get("HELION_LLM_API_KEY")) is not None:
        return generic_api_key
    config = _provider_config(provider)
    if resolved_api_key := _first_set_env(*config.api_key_env_names):
        return resolved_api_key
    raise exc.ProviderAuthError(config.missing_api_key_error)


def _resolve_v1_endpoint(api_base: str, endpoint: str) -> str:
    """Append the provider endpoint while tolerating bases that already include it."""
    base = api_base.rstrip("/")
    if base.endswith((f"/v1/{endpoint}", f"/{endpoint}")):
        return base
    if base.endswith("/v1"):
        return f"{base}/{endpoint}"
    return f"{base}/v1/{endpoint}"


def _build_ssl_context() -> ssl.SSLContext | None:
    """Build an optional SSL context for custom CA bundles or client certs."""
    ca_bundle = _first_existing_path(
        "HELION_LLM_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "CURL_CA_BUNDLE"
    )
    # Fall back to common mTLS client-cert env conventions used by HTTPS gateways
    # (in addition to helion's own knob) so requests work out-of-the-box when an
    # identity is already configured by another tool.
    cert = _first_existing_path(
        "HELION_LLM_CLIENT_CERT",
        "CLAUDE_CODE_CLIENT_CERT",
        "THRIFT_TLS_CL_CERT_PATH",
    )
    if ca_bundle is None and cert is None:
        return None

    context = (
        ssl.create_default_context(cafile=ca_bundle)
        if ca_bundle is not None
        else ssl.create_default_context()
    )
    if cert is not None:
        key = (
            _first_existing_path(
                "HELION_LLM_CLIENT_KEY",
                "CLAUDE_CODE_CLIENT_KEY",
                "THRIFT_TLS_CL_KEY_PATH",
            )
            or cert
        )
        context.load_cert_chain(certfile=cert, keyfile=key)
    return context


def _build_provider_payload(
    provider: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    effort_level: str | None,
    fast_mode: bool,
) -> dict[str, Any]:
    """Build the JSON request body for the selected provider."""
    return _provider_config(provider).build_payload(
        model,
        messages,
        max_output_tokens,
        effort_level,
        fast_mode,
    )


def _build_provider_headers(
    provider: str, api_key: str, fast_mode: bool
) -> dict[str, str]:
    """Build auth and content headers for the selected provider."""
    return _provider_config(provider).build_headers(api_key, fast_mode)


def _load_json_response(
    request: urllib_request.Request,
    *,
    request_timeout_s: float,
    ssl_context: ssl.SSLContext | None,
) -> object:
    """Load one JSON response body, optionally using a custom SSL context."""
    if ssl_context is None:
        with urllib_request.urlopen(request, timeout=request_timeout_s) as response:
            return json.load(response)
    with urllib_request.urlopen(
        request,
        timeout=request_timeout_s,
        context=ssl_context,
    ) as response:
        return json.load(response)


def _extra_headers() -> dict[str, str]:
    """Optional extra HTTP headers for gateways/proxies that require them.

    Read from ``HELION_LLM_EXTRA_HEADERS`` as either a JSON object
    (``{"X-Header": "value"}``) or newline-separated ``Header: value`` lines.
    Useful when an API-compatible gateway needs custom routing/identity headers
    on top of the standard auth headers.
    """
    raw = os.environ.get("HELION_LLM_EXTRA_HEADERS")
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        parsed = json.loads(raw)
        return {str(k): str(v) for k, v in parsed.items()}
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip()] = value.strip()
    return headers


def _http_status_error(code: int, url: str, body: str) -> exc.BaseError:
    """Map an HTTP status to a typed transport exception (§3.1).

    401/403 and other 4xx (except 408/429) are auth/config/programming errors that
    propagate; 408/429/5xx are catchable provider failures eligible for fallback.
    """
    detail = f"HTTP {code} from {url}: {body}"
    if code in (401, 403):
        return exc.ProviderAuthError(detail)
    if code == 408:
        return exc.ProviderTimeout(detail)
    if code == 429:
        return exc.ProviderRateLimited(detail)
    if 500 <= code < 600:
        return exc.ProviderServerError(detail)
    return exc.ProviderRequestError(detail)


def _bedrock_error(error: Exception, model_id: str) -> exc.BaseError | None:
    """Map botocore failures to the same catchability contract as HTTP."""
    detail = f"Bedrock invoke_model failed for {model_id!r}: {error}"
    if isinstance(error, (TimeoutError, socket.timeout)):
        return exc.ProviderTimeout(detail)

    response = getattr(error, "response", None)
    error_payload = response.get("Error") if isinstance(response, Mapping) else None
    metadata = (
        response.get("ResponseMetadata") if isinstance(response, Mapping) else None
    )
    code = error_payload.get("Code") if isinstance(error_payload, Mapping) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    error_name = type(error).__name__
    if (
        status in (401, 403)
        or code
        in {
            "AccessDeniedException",
            "IncompleteSignature",
            "InvalidClientTokenId",
            "NotAuthorizedException",
            "UnrecognizedClientException",
        }
        or error_name
        in {
            "CredentialRetrievalError",
            "NoCredentialsError",
            "PartialCredentialsError",
            "TokenRetrievalError",
            "UnauthorizedSSOTokenError",
        }
    ):
        return exc.ProviderAuthError(detail)
    if error_name in {
        "ConfigNotFound",
        "ConfigParseError",
        "InvalidConfigError",
        "NoRegionError",
        "ProfileNotFound",
    }:
        return exc.ProviderRequestError(detail)
    if status == 408:
        return exc.ProviderTimeout(detail)
    if status == 429 or code in {
        "LimitExceededException",
        "ServiceQuotaExceededException",
        "ThrottlingException",
        "TooManyRequestsException",
    }:
        return exc.ProviderRateLimited(detail)
    if isinstance(status, int) and 500 <= status < 600:
        return exc.ProviderServerError(detail)
    if isinstance(status, int) and 400 <= status < 500:
        return exc.ProviderRequestError(detail)
    if isinstance(error, OSError) or error_name in {
        "ConnectionClosedError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "HTTPClientError",
        "ProxyConnectionError",
        "ReadTimeoutError",
        "ResponseStreamingError",
        "SSLError",
    }:
        return exc.ProviderTransportError(detail)
    return None


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    request_timeout_s: float,
) -> dict[str, object]:
    """Send one JSON POST and normalize HTTP and payload errors into typed exc."""
    request = urllib_request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, **_extra_headers()},
        method="POST",
    )
    try:
        body = _load_json_response(
            request,
            request_timeout_s=request_timeout_s,
            ssl_context=_build_ssl_context(),
        )
    except urllib_error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise _http_status_error(e.code, url, detail) from e
    except TimeoutError as e:
        # A read-phase timeout surfaces raw (it is not a URLError subclass).
        raise exc.ProviderTimeout(f"Request to {url} timed out") from e
    except urllib_error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise exc.ProviderTimeout(f"Request to {url} timed out: {e.reason}") from e
        raise exc.ProviderTransportError(f"Request to {url} failed: {e.reason}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise exc.InvalidResponseSchema(f"Malformed JSON payload from {url}") from e

    if isinstance(body, dict):
        return body
    raise exc.InvalidResponseSchema(
        f"Unexpected JSON payload from {url}: {type(body).__name__}"
    )


# Anthropic-on-Bedrock requires this version string in the request body and
# forbids the `model` field (the model is named by `modelId` on the API call).
_BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"


def _resolve_bedrock_region(api_base: str | None) -> str | None:
    """Pick the AWS region for Bedrock from the api_base override or env."""
    # Allow `api_base` to carry the region (e.g. "us-east-1") for parity with
    # the other providers' single knob; otherwise fall back to the standard
    # AWS env vars, then let boto3's own config/role resolution take over.
    return (
        api_base or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    )


def _call_bedrock(
    *,
    model: str,
    api_base: str | None,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    request_timeout_s: float,
    effort_level: str | None,
    fast_mode: bool,
) -> dict[str, object]:
    """Invoke Anthropic-on-Bedrock via boto3, reusing the Anthropic codecs.

    Auth is handled by boto3's default credential chain (IAM role, env creds,
    or profile) -- no API key is read. The request body is the same Anthropic
    Messages payload used by the direct API, minus `model` and plus the Bedrock
    `anthropic_version`.

    Request/response format and the boto3 ``invoke_model`` usage follow the AWS
    reference example:
    https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AnthropicClaude_section.html
    """
    try:
        # boto3 is an optional dependency: only required when the Bedrock
        # provider is actually used, so it isn't a Helion install requirement.
        import boto3  # pyrefly: ignore [missing-import]
        import botocore.config  # pyrefly: ignore [missing-import]
    except ImportError as e:
        raise exc.ProviderRequestError(
            "Bedrock provider requested but boto3 is not installed. "
            "Install it with `pip install boto3`."
        ) from e

    payload = _anthropic_payload(
        model,
        messages,
        max_output_tokens,
        effort_level,
        fast_mode,
    )
    # On Bedrock the model is named by `modelId`, not in the body.
    model_id = payload.pop("model")
    payload["anthropic_version"] = _BEDROCK_ANTHROPIC_VERSION

    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=_resolve_bedrock_region(api_base),
            config=botocore.config.Config(
                read_timeout=request_timeout_s,
                connect_timeout=request_timeout_s,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
    except Exception as e:  # botocore configuration and credential errors
        if mapped := _bedrock_error(e, model_id):
            raise mapped from e
        raise
    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(payload))
    except Exception as e:  # botocore.exceptions.ClientError and friends
        if mapped := _bedrock_error(e, model_id):
            raise mapped from e
        raise

    try:
        raw_body = response["body"].read()
    except Exception as e:
        if mapped := _bedrock_error(e, model_id):
            raise mapped from e
        raise
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise exc.InvalidResponseSchema(
            f"Malformed Bedrock payload from {model_id!r}"
        ) from e
    if not isinstance(body, dict):
        raise exc.InvalidResponseSchema(
            f"Unexpected Bedrock payload from {model_id!r}: {type(body).__name__}"
        )
    return body


# Anthropic-on-Vertex-AI names the model in the URL (not the body) and requires
# this version string in the request body, mirroring the Bedrock convention.
_VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"
_DEFAULT_VERTEX_LOCATION = "global"


def _call_vertex(
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    request_timeout_s: float,
    effort_level: str | None,
    fast_mode: bool,
) -> dict[str, object]:
    """Invoke Anthropic-on-Vertex-AI over HTTPS, reusing the Anthropic codecs.

    Vertex AI hosts Anthropic models under the publisher ``rawPredict`` endpoint.
    The request body is the same Anthropic Messages payload used by the direct
    API, but the model is named in the URL (not the body) and a Vertex
    ``anthropic_version`` replaces it. Auth is carried by the SSL context (mTLS)
    and/or an ``Authorization: Bearer`` token, so an API key is optional --
    useful behind a gateway that authenticates the caller by client identity.

    Configure with ``HELION_LLM_PROVIDER=vertex``, ``HELION_LLM_API_BASE`` (the
    endpoint base), ``HELION_LLM_VERTEX_PROJECT`` (project id), and optionally
    ``HELION_LLM_VERTEX_LOCATION`` (default ``global``). Each falls back to the
    standard Anthropic-on-Vertex SDK variable when its helion-specific knob is
    unset: ``ANTHROPIC_VERTEX_BASE_URL``, ``ANTHROPIC_VERTEX_PROJECT_ID``, and
    ``CLOUD_ML_REGION`` respectively -- so an environment already set up for
    Anthropic-on-Vertex needs only ``HELION_LLM_PROVIDER=vertex`` and a model.

    Request/response format follows the Anthropic-on-Vertex reference:
    https://docs.anthropic.com/en/api/claude-on-vertex-ai
    """
    # Resolve endpoint/project/location from helion's own knobs first, then fall
    # back to the standard Anthropic-on-Vertex SDK variables so an environment
    # already configured for Anthropic-on-Vertex works without extra setup.
    base = (
        api_base
        or os.environ.get("HELION_LLM_API_BASE")
        or os.environ.get("ANTHROPIC_VERTEX_BASE_URL")
    )
    if not base:
        raise exc.ProviderRequestError(
            "Vertex provider requires the endpoint base URL via api_base, "
            "HELION_LLM_API_BASE, or ANTHROPIC_VERTEX_BASE_URL."
        )
    project = os.environ.get("HELION_LLM_VERTEX_PROJECT") or os.environ.get(
        "ANTHROPIC_VERTEX_PROJECT_ID"
    )
    if not project:
        raise exc.ProviderRequestError(
            "Vertex provider requires the project id via "
            "HELION_LLM_VERTEX_PROJECT or ANTHROPIC_VERTEX_PROJECT_ID."
        )
    location = (
        os.environ.get("HELION_LLM_VERTEX_LOCATION")
        or os.environ.get("CLOUD_ML_REGION")
        or _DEFAULT_VERTEX_LOCATION
    )
    named_model = strip_provider_prefix(model)
    url = (
        f"{base.rstrip('/')}/projects/{project}/locations/{location}"
        f"/publishers/anthropic/models/{named_model}:rawPredict"
    )
    payload = _anthropic_payload(
        model, messages, max_output_tokens, effort_level, fast_mode
    )
    # The model is in the URL on Vertex; the body carries the version instead.
    payload.pop("model", None)
    payload["anthropic_version"] = _VERTEX_ANTHROPIC_VERSION
    headers = {"content-type": "application/json"}
    key = api_key or os.environ.get("HELION_LLM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return _post_json(url, payload, headers, request_timeout_s=request_timeout_s)


def call_provider_with_usage(
    provider: str,
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    request_timeout_s: float,
    effort_level: str | None = None,
    fast_mode: bool = False,
) -> ProviderResult:
    """Send one request; return extracted text plus provider-reported token usage.

    Usage is populated only from fields present in the provider response; missing
    fields remain empty rather than being estimated.
    """
    normalized_provider = normalize_provider(provider)
    request_payload = _identity_request_payload(
        normalized_provider,
        model=model,
        messages=messages,
        max_output_tokens=max_output_tokens,
        effort_level=effort_level,
        fast_mode=fast_mode,
    )
    if normalized_provider == "vertex":
        # Vertex names the model in the URL and authenticates by SSL/bearer, so
        # it bypasses the api-key-required _ProviderConfig path (like Bedrock).
        response = _call_vertex(
            model=model,
            api_base=api_base,
            api_key=api_key,
            messages=messages,
            max_output_tokens=max_output_tokens,
            request_timeout_s=request_timeout_s,
            effort_level=effort_level,
            fast_mode=fast_mode,
        )
        usage = extract_anthropic_usage(response)
        metadata = _provider_metadata(request_payload, response, usage)
        return ProviderResult(
            text=extract_anthropic_text(response),
            usage=usage,
            request_id=metadata.request_id,
            response_id=metadata.response_id,
            cache_state=metadata.cache_state,
        )
    if normalized_provider == "bedrock":
        # Bedrock uses boto3/SigV4 instead of an HTTP+api-key transport, so it
        # bypasses the _ProviderConfig path entirely.
        response = _call_bedrock(
            model=model,
            api_base=api_base,
            messages=messages,
            max_output_tokens=max_output_tokens,
            request_timeout_s=request_timeout_s,
            effort_level=effort_level,
            fast_mode=fast_mode,
        )
        usage = extract_anthropic_usage(response)
        metadata = _provider_metadata(request_payload, response, usage)
        return ProviderResult(
            text=extract_anthropic_text(response),
            usage=usage,
            request_id=metadata.request_id,
            response_id=metadata.response_id,
            cache_state=metadata.cache_state,
        )
    config = _provider_config(normalized_provider)
    resolved_api_key = _resolve_api_key(normalized_provider, api_key)
    response = _post_json(
        _resolve_v1_endpoint(
            _resolve_api_base(normalized_provider, api_base),
            config.endpoint,
        ),
        request_payload,
        _build_provider_headers(normalized_provider, resolved_api_key, fast_mode),
        request_timeout_s=request_timeout_s,
    )
    usage = config.extract_usage(response)
    metadata = _provider_metadata(request_payload, response, usage)
    return ProviderResult(
        text=config.extract_text(response),
        usage=usage,
        request_id=metadata.request_id,
        response_id=metadata.response_id,
        cache_state=metadata.cache_state,
    )


def call_provider(
    provider: str,
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    request_timeout_s: float,
    effort_level: str | None = None,
    fast_mode: bool = False,
) -> str:
    """Resolve credentials, send one request, and extract text from the response."""
    return call_provider_with_usage(
        provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        messages=messages,
        max_output_tokens=max_output_tokens,
        request_timeout_s=request_timeout_s,
        effort_level=effort_level,
        fast_mode=fast_mode,
    ).text
