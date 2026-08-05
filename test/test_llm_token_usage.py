from __future__ import annotations

from unittest import mock

from helion.autotuner.llm import transport
from helion.autotuner.llm.transport import TokenUsage
from helion.autotuner.llm.transport import extract_anthropic_usage
from helion.autotuner.llm.transport import extract_openai_usage

_ANTHROPIC_RESPONSE = {
    "id": "msg-anthropic-1",
    "content": [{"type": "text", "text": "hello"}],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 3,
    },
}
_OPENAI_RESPONSE = {
    "id": "resp-openai-1",
    "output": [{"content": [{"type": "output_text", "text": "hi"}]}],
    "usage": {
        "input_tokens": 20,
        "output_tokens": 8,
        "input_tokens_details": {"cached_tokens": 6},
        "output_tokens_details": {"reasoning_tokens": 4},
    },
}


def _call(response, provider="anthropic"):
    with mock.patch.object(transport, "_post_json", return_value=response):
        return transport.call_provider_with_usage(
            provider,
            model="claude-x" if provider == "anthropic" else "gpt-x",
            api_base=None,
            api_key="test-key",
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=100,
            request_timeout_s=1.0,
        )


# --- usage extraction ------------------------------------------------------
def test_extract_anthropic_usage():
    usage = extract_anthropic_usage(_ANTHROPIC_RESPONSE)
    assert usage == TokenUsage(
        input_tokens=10, cached_input_tokens=3, output_tokens=5, reasoning_tokens=None
    )


def test_extract_openai_usage():
    usage = extract_openai_usage(_OPENAI_RESPONSE)
    assert usage == TokenUsage(
        input_tokens=20, cached_input_tokens=6, output_tokens=8, reasoning_tokens=4
    )


def test_missing_usage_is_all_none_not_estimated():
    assert extract_anthropic_usage({"content": []}) == TokenUsage()
    assert extract_openai_usage({"output": []}) == TokenUsage()


def test_missing_subfields_are_none():
    usage = extract_openai_usage({"usage": {"input_tokens": 5}})
    assert usage.input_tokens == 5
    assert usage.cached_input_tokens is None
    assert usage.reasoning_tokens is None


# --- call_provider_with_usage + backward-compat call_provider --------------
def test_call_provider_with_usage_surfaces_tokens():
    result = _call(_ANTHROPIC_RESPONSE)
    assert result.text == "hello"
    assert result.response_id == "msg-anthropic-1"
    assert result.cache_state == "hit"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.cached_input_tokens == 3


def test_provider_result_uses_provider_response_id_and_hashed_request_identity():
    result = _call(_OPENAI_RESPONSE, provider="openai")

    assert result.request_id.startswith("sha256:")
    assert len(result.request_id) == len("sha256:") + 64
    assert result.response_id == "resp-openai-1"
    assert result.cache_state == "hit"


def test_anthropic_zero_cached_tokens_is_provider_reported_miss():
    response = {
        **_ANTHROPIC_RESPONSE,
        "usage": {
            **_ANTHROPIC_RESPONSE["usage"],
            "cache_read_input_tokens": 0,
        },
    }

    assert _call(response).cache_state == "miss"


def test_missing_usage_yields_unknown_cache_state_and_hashed_response_identity():
    response = {"content": [{"type": "text", "text": "hello"}]}

    first = _call(response)
    second = _call(
        {"content": [{"text": "hello", "type": "text"}]},
    )

    assert first.cache_state == "unknown"
    assert first.response_id.startswith("sha256:")
    assert first.response_id == second.response_id
    assert first.request_id == second.request_id


def test_provider_reported_request_id_is_preferred_over_payload_hash():
    response = {**_ANTHROPIC_RESPONSE, "request_id": "request-provider-1"}

    assert _call(response).request_id == "request-provider-1"


def test_vertex_anthropic_shape_carries_identity_usage_and_cache_state():
    response = {
        "id": "msg-vertex-1",
        "content": [{"type": "text", "text": "vertex"}],
        "usage": {
            "input_tokens": 12,
            "cache_read_input_tokens": 4,
            "output_tokens": 3,
        },
    }
    with mock.patch.object(transport, "_call_vertex", return_value=response):
        result = transport.call_provider_with_usage(
            "vertex",
            model="vertex/claude-sonnet-4-5",
            api_base="https://vertex.invalid",
            api_key=None,
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=100,
            request_timeout_s=1.0,
        )
        other_model = transport.call_provider_with_usage(
            "vertex",
            model="vertex/claude-opus-4-5",
            api_base="https://vertex.invalid",
            api_key=None,
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=100,
            request_timeout_s=1.0,
        )

    assert result.request_id.startswith("sha256:")
    assert result.request_id != other_model.request_id
    assert result.response_id == "msg-vertex-1"
    assert result.cache_state == "hit"
    assert result.usage.cached_input_tokens == 4


def test_call_provider_returns_text_only():
    with mock.patch.object(transport, "_post_json", return_value=_ANTHROPIC_RESPONSE):
        text = transport.call_provider(
            "anthropic",
            model="claude-x",
            api_base=None,
            api_key="test-key",
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=100,
            request_timeout_s=1.0,
        )
    assert text == "hello"
