"""Send direct HTTP requests to the configured LLM provider."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import ssl
from typing import TYPE_CHECKING
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REQUEST_TIMEOUT_S = 120.0
# OpenAI Responses does not consume a temperature knob in our current request path,
# so keep Anthropic's setting internal instead of exposing it on the search API.
DEFAULT_ANTHROPIC_TEMPERATURE = 0.3

_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "openai": "openai_responses",
    "openai_responses": "openai_responses",
    "openai-responses": "openai_responses",
}


def normalize_provider(provider: str) -> str:
    """Canonicalize user-facing provider names to internal transport IDs."""
    normalized = provider.strip().lower()
    if resolved := _PROVIDER_ALIASES.get(normalized):
        return resolved
    raise ValueError(
        f"Unsupported LLM provider {provider!r}. "
        "Valid providers are: anthropic, openai, openai_responses."
    )


def infer_provider(model: str, provider: str | None = None) -> str:
    """Guess the transport from the model name unless the caller overrides it."""
    if provider is not None:
        return normalize_provider(provider)
    normalized = model.lower()
    if normalized.startswith(("claude", "anthropic/")):
        return "anthropic"
    if normalized.startswith(
        ("gpt-", "chatgpt-", "codex", "o1", "o3", "o4", "openai/")
    ):
        return "openai_responses"
    return "unsupported"


def strip_provider_prefix(model: str) -> str:
    """Remove a provider prefix before sending the model name to the API."""
    for prefix in ("anthropic/", "openai/"):
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
    raise RuntimeError(f"Unexpected OpenAI responses payload: {response}")


def extract_anthropic_text(response: dict[str, object]) -> str:
    """Extract concatenated text from an Anthropic Messages payload."""
    if texts := _extract_text_content_items(response.get("content")):
        return "".join(texts)
    raise RuntimeError(f"Unexpected Anthropic payload: {response}")


def _openai_payload(
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build an OpenAI Responses request payload."""
    system_prompt, input_messages = split_system_messages(messages)
    payload: dict[str, Any] = {
        "model": strip_provider_prefix(model),
        "input": responses_input_from_messages(input_messages),
        "max_output_tokens": max_output_tokens,
    }
    if system_prompt:
        payload["instructions"] = system_prompt
    return payload


def _anthropic_payload(
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build an Anthropic Messages request payload."""
    system_prompt, input_messages = split_system_messages(messages)
    payload: dict[str, Any] = {
        "model": strip_provider_prefix(model),
        "messages": anthropic_messages_from_history(input_messages),
        "max_tokens": max_output_tokens,
        "temperature": DEFAULT_ANTHROPIC_TEMPERATURE,
    }
    if system_prompt:
        payload["system"] = system_prompt
    return payload


def _openai_headers(api_key: str) -> dict[str, str]:
    """Build OpenAI-compatible auth headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _anthropic_headers(api_key: str) -> dict[str, str]:
    """Build Anthropic Messages auth headers."""
    return {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": api_key,
    }


@dataclass(frozen=True)
class _ProviderConfig:
    """Provider-specific transport configuration."""

    endpoint: str
    default_api_base: str
    api_base_env_names: tuple[str, ...]
    api_key_env_names: tuple[str, ...]
    missing_api_key_error: str
    build_payload: Callable[[str, list[dict[str, str]], int], dict[str, Any]]
    build_headers: Callable[[str], dict[str, str]]
    extract_text: Callable[[dict[str, object]], str]


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
    raise RuntimeError(config.missing_api_key_error)


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
    ca_bundle = _first_existing_path("HELION_LLM_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")
    cert = _first_existing_path("HELION_LLM_CLIENT_CERT")
    if ca_bundle is None and cert is None:
        return None

    context = (
        ssl.create_default_context(cafile=ca_bundle)
        if ca_bundle is not None
        else ssl.create_default_context()
    )
    if cert is not None:
        key = _first_existing_path("HELION_LLM_CLIENT_KEY") or cert
        context.load_cert_chain(certfile=cert, keyfile=key)
    return context


def _build_provider_payload(
    provider: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build the JSON request body for the selected provider."""
    return _provider_config(provider).build_payload(model, messages, max_output_tokens)


def _build_provider_headers(provider: str, api_key: str) -> dict[str, str]:
    """Build auth and content headers for the selected provider."""
    return _provider_config(provider).build_headers(api_key)


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


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    request_timeout_s: float,
) -> dict[str, object]:
    """Send one JSON POST and normalize HTTP and payload errors."""
    request = urllib_request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        body = _load_json_response(
            request,
            request_timeout_s=request_timeout_s,
            ssl_context=_build_ssl_context(),
        )
    except urllib_error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib_error.URLError as e:
        raise RuntimeError(f"Request to {url} failed: {e.reason}") from e

    if isinstance(body, dict):
        return body
    raise RuntimeError(f"Unexpected JSON payload from {url}: {type(body).__name__}")


def call_provider(
    provider: str,
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    request_timeout_s: float,
) -> str:
    """Resolve credentials, send one request, and extract text from the response."""
    normalized_provider = normalize_provider(provider)
    config = _provider_config(normalized_provider)
    resolved_api_key = _resolve_api_key(normalized_provider, api_key)
    response = _post_json(
        _resolve_v1_endpoint(
            _resolve_api_base(normalized_provider, api_base),
            config.endpoint,
        ),
        _build_provider_payload(
            normalized_provider,
            model=model,
            messages=messages,
            max_output_tokens=max_output_tokens,
        ),
        _build_provider_headers(normalized_provider, resolved_api_key),
        request_timeout_s=request_timeout_s,
    )
    return config.extract_text(response)
