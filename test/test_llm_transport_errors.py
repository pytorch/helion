from __future__ import annotations

from email.message import Message
import io
import json
from types import SimpleNamespace
from unittest import mock
from urllib import error as urllib_error

import pytest

from helion import exc
from helion.autotuner.llm import transport

_CATCHABLE = [
    exc.ProviderTimeout,
    exc.ProviderTransportError,
    exc.ProviderRateLimited,
    exc.ProviderServerError,
    exc.RetrieverUnavailable,
    exc.InvalidResponseSchema,
    exc.ZeroValidCandidates,
]
_PROPAGATE = [exc.ProviderAuthError, exc.ProviderRequestError]


def _http_error(code: int, body: bytes = b"err") -> urllib_error.HTTPError:
    return urllib_error.HTTPError("http://x", code, "msg", Message(), io.BytesIO(body))


def _post_json_raising(side_effect=None, return_value=None):
    with mock.patch.object(
        transport,
        "_load_json_response",
        side_effect=side_effect,
        return_value=return_value,
    ):
        return transport._post_json("http://x", {}, {}, request_timeout_s=1.0)


# --- taxonomy shape (no message parsing) -----------------------------------
def test_catchable_are_retrieval_provider_errors():
    for e in _CATCHABLE:
        assert issubclass(e, exc.RetrievalProviderError)


def test_propagate_errors_are_not_catchable():
    for e in _PROPAGATE:
        assert not issubclass(e, exc.RetrievalProviderError)


def test_all_transport_errors_are_runtimeerror():
    # Broad ``except RuntimeError`` in existing callers still catches these.
    for e in _CATCHABLE + _PROPAGATE:
        assert issubclass(e, RuntimeError)


# --- HTTP status classification --------------------------------------------
@pytest.mark.parametrize(
    "code,expected",
    [
        (401, exc.ProviderAuthError),
        (403, exc.ProviderAuthError),
        (408, exc.ProviderTimeout),
        (429, exc.ProviderRateLimited),
        (500, exc.ProviderServerError),
        (503, exc.ProviderServerError),
        (400, exc.ProviderRequestError),
        (404, exc.ProviderRequestError),
    ],
)
def test_http_status_mapping(code, expected):
    assert isinstance(transport._http_status_error(code, "http://x", "b"), expected)


@pytest.mark.parametrize(
    "code,expected",
    [
        (429, exc.ProviderRateLimited),
        (500, exc.ProviderServerError),
        (401, exc.ProviderAuthError),
    ],
)
def test_post_json_classifies_http_errors(code, expected):
    with pytest.raises(expected):
        _post_json_raising(side_effect=_http_error(code))


# --- transport / timeout classification ------------------------------------
def test_post_json_raw_timeout_is_provider_timeout():
    with pytest.raises(exc.ProviderTimeout):
        _post_json_raising(side_effect=TimeoutError("read timed out"))


def test_post_json_urlerror_timeout_is_provider_timeout():
    with pytest.raises(exc.ProviderTimeout):
        _post_json_raising(side_effect=urllib_error.URLError(TimeoutError()))


def test_post_json_urlerror_transport_failure():
    with pytest.raises(exc.ProviderTransportError):
        _post_json_raising(side_effect=urllib_error.URLError("connection refused"))


def test_post_json_non_dict_is_invalid_schema():
    with pytest.raises(exc.InvalidResponseSchema):
        _post_json_raising(return_value=["not", "a", "dict"])


def test_post_json_malformed_json_is_invalid_schema():
    malformed = json.JSONDecodeError("bad provider JSON", "{", 1)

    with pytest.raises(exc.InvalidResponseSchema):
        _post_json_raising(side_effect=malformed)


def test_post_json_invalid_encoding_is_invalid_schema():
    malformed = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(exc.InvalidResponseSchema):
        _post_json_raising(side_effect=malformed)


class _BedrockError(Exception):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code, "Message": "provider detail"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


def _call_bedrock_raising(monkeypatch, error: Exception):
    class Client:
        def invoke_model(self, **kwargs):
            raise error

    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        SimpleNamespace(client=lambda *args, **kwargs: Client()),
    )
    config_module = SimpleNamespace(Config=lambda **kwargs: kwargs)
    monkeypatch.setitem(
        __import__("sys").modules,
        "botocore",
        SimpleNamespace(config=config_module),
    )
    monkeypatch.setitem(__import__("sys").modules, "botocore.config", config_module)
    return transport._call_bedrock(
        model="bedrock/claude-sonnet-4-6",
        api_base="us-west-2",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=64,
        request_timeout_s=1.0,
        effort_level=None,
        fast_mode=False,
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_BedrockError("AccessDeniedException", 403), exc.ProviderAuthError),
        (_BedrockError("ValidationException", 400), exc.ProviderRequestError),
        (_BedrockError("ThrottlingException", 429), exc.ProviderRateLimited),
        (_BedrockError("InternalServerException", 500), exc.ProviderServerError),
        (TimeoutError("deadline"), exc.ProviderTimeout),
        (OSError("connection reset"), exc.ProviderTransportError),
    ],
)
def test_bedrock_classifies_provider_errors(monkeypatch, error, expected):
    with pytest.raises(expected):
        _call_bedrock_raising(monkeypatch, error)


@pytest.mark.parametrize(
    "error_name",
    ["ProxyConnectionError", "SSLError", "ResponseStreamingError"],
)
def test_bedrock_classifies_transient_botocore_errors(monkeypatch, error_name):
    error_type = type(error_name, (Exception,), {})

    with pytest.raises(exc.ProviderTransportError):
        _call_bedrock_raising(monkeypatch, error_type("transient transport failure"))


def test_bedrock_classifies_streaming_body_read_failure(monkeypatch):
    class ResponseStreamingError(Exception):
        pass

    class Body:
        def read(self):
            raise ResponseStreamingError("stream closed")

    class Client:
        def invoke_model(self, **kwargs):
            return {"body": Body()}

    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        SimpleNamespace(client=lambda *args, **kwargs: Client()),
    )
    config_module = SimpleNamespace(Config=lambda **kwargs: kwargs)
    monkeypatch.setitem(
        __import__("sys").modules,
        "botocore",
        SimpleNamespace(config=config_module),
    )
    monkeypatch.setitem(__import__("sys").modules, "botocore.config", config_module)

    with pytest.raises(exc.ProviderTransportError):
        transport._call_bedrock(
            model="bedrock/claude-sonnet-4-6",
            api_base="us-west-2",
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=64,
            request_timeout_s=1.0,
            effort_level=None,
            fast_mode=False,
        )


def test_bedrock_client_configuration_error_is_typed(monkeypatch):
    class NoRegionError(Exception):
        pass

    def raise_no_region(*args, **kwargs):
        raise NoRegionError("region is required")

    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        SimpleNamespace(client=raise_no_region),
    )
    config_module = SimpleNamespace(Config=lambda **kwargs: kwargs)
    monkeypatch.setitem(
        __import__("sys").modules,
        "botocore",
        SimpleNamespace(config=config_module),
    )
    monkeypatch.setitem(__import__("sys").modules, "botocore.config", config_module)

    with pytest.raises(exc.ProviderRequestError):
        transport._call_bedrock(
            model="bedrock/claude-sonnet-4-6",
            api_base=None,
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=64,
            request_timeout_s=1.0,
            effort_level=None,
            fast_mode=False,
        )


@pytest.mark.parametrize("payload", [b"{", b"\xff"])
def test_bedrock_malformed_json_is_invalid_schema(monkeypatch, payload):
    class Body:
        def read(self):
            return payload

    class Client:
        def invoke_model(self, **kwargs):
            return {"body": Body()}

    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        SimpleNamespace(client=lambda *args, **kwargs: Client()),
    )
    config_module = SimpleNamespace(Config=lambda **kwargs: kwargs)
    monkeypatch.setitem(
        __import__("sys").modules,
        "botocore",
        SimpleNamespace(config=config_module),
    )
    monkeypatch.setitem(__import__("sys").modules, "botocore.config", config_module)

    with pytest.raises(exc.InvalidResponseSchema):
        transport._call_bedrock(
            model="bedrock/claude-sonnet-4-6",
            api_base="us-west-2",
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=64,
            request_timeout_s=1.0,
            effort_level=None,
            fast_mode=False,
        )


# --- response schema extraction --------------------------------------------
def test_extract_openai_bad_payload():
    with pytest.raises(exc.InvalidResponseSchema):
        transport.extract_openai_response_text({"output": "nope"})


def test_extract_anthropic_bad_payload():
    with pytest.raises(exc.InvalidResponseSchema):
        transport.extract_anthropic_text({"content": "nope"})


# --- auth + programming-error propagation ----------------------------------
def test_missing_api_key_raises_auth(monkeypatch):
    monkeypatch.delenv("HELION_LLM_API_KEY", raising=False)
    with (
        mock.patch.object(transport, "_first_set_env", return_value=None),
        pytest.raises(exc.ProviderAuthError),
    ):
        transport._resolve_api_key("anthropic", None)


def test_unsupported_provider_propagates_valueerror():
    with pytest.raises(ValueError):
        transport.normalize_provider("definitely-not-a-provider")


def test_cancellation_propagates():
    with pytest.raises(KeyboardInterrupt):
        _post_json_raising(side_effect=KeyboardInterrupt())
