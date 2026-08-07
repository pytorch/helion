from __future__ import annotations

import pytest

from helion import exc
from helion.autotuner.rag import FallbackReason
from helion.autotuner.rag import classify_fallback
from helion.autotuner.rag import execute_with_fallback


# --- classification by type (no message parsing) ---------------------------
@pytest.mark.parametrize(
    "exception,expected",
    [
        (exc.ProviderTimeout("t"), FallbackReason.PROVIDER_FAILURE),
        (exc.ProviderTransportError("t"), FallbackReason.PROVIDER_FAILURE),
        (exc.ProviderRateLimited("t"), FallbackReason.PROVIDER_FAILURE),
        (exc.ProviderServerError("t"), FallbackReason.PROVIDER_FAILURE),
        (exc.InvalidResponseSchema("t"), FallbackReason.PROVIDER_FAILURE),
        (exc.ZeroValidCandidates("t"), FallbackReason.PROVIDER_FAILURE),
        (exc.RetrieverUnavailable("t"), FallbackReason.RETRIEVAL_FAILURE),
    ],
)
def test_classify_catchable_provider_errors(exception, expected):
    assert classify_fallback(exception) == expected


@pytest.mark.parametrize(
    "exception",
    [
        exc.ProviderAuthError("nope"),
        exc.ProviderRequestError("nope"),
        ValueError("programming"),
        KeyError("internal"),
    ],
)
def test_classify_propagating_errors_return_none(exception):
    assert classify_fallback(exception) is None


def test_classify_artifact_errors():
    signing = pytest.importorskip("helion_rag.signing")
    assert (
        classify_fallback(signing.SignatureError("x"))
        == FallbackReason.SIGNATURE_FAILURE
    )
    assert (
        classify_fallback(signing.MissingArtifactError("x"))
        == FallbackReason.MISSING_ARTIFACT
    )
    assert (
        classify_fallback(signing.HashMismatchError("x"))
        == FallbackReason.INDEX_CORRUPTION
    )
    assert (
        classify_fallback(signing.VersionMismatchError("x"))
        == FallbackReason.VERSION_MISMATCH
    )


# --- execute_with_fallback -------------------------------------------------
def test_primary_success_no_fallback():
    result, reason = execute_with_fallback(lambda: "primary", lambda: "baseline")
    assert result == "primary"
    assert reason is None


def test_catchable_failure_runs_baseline():
    def primary():
        raise exc.ProviderTimeout("timed out")

    result, reason = execute_with_fallback(primary, lambda: "baseline")
    assert result == "baseline"
    assert reason == FallbackReason.PROVIDER_FAILURE


def test_non_catchable_failure_propagates():
    def primary():
        raise exc.ProviderAuthError("401")

    with pytest.raises(exc.ProviderAuthError):
        execute_with_fallback(primary, lambda: "baseline")


def test_cancellation_propagates():
    def primary():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_with_fallback(primary, lambda: "baseline")
