"""The typed fallback boundary (§3.1).

Failures are classified by exception *type*, never by parsing a message. Only the
closed catchable taxonomy maps to a stable :class:`FallbackReason`; authentication,
configuration, parser-invariant, internal, programming, and cancellation errors
return ``None`` and propagate. ``execute_with_fallback`` runs a primary operation
and, on a catchable failure only, records the reason and runs the frozen baseline.
"""

from __future__ import annotations

from typing import Callable
from typing import TypeVar

from ... import exc
from .types import FallbackReason

R = TypeVar("R")


def _classify_artifact_error(exception: Exception) -> FallbackReason | None:
    """Map helion_rag artifact-verification failures, if that package is present."""
    try:
        from helion_rag import signing  # pyrefly: ignore [missing-import]
    except ImportError:
        return None
    if isinstance(exception, signing.SignatureError):
        return FallbackReason.SIGNATURE_FAILURE
    if isinstance(exception, signing.MissingArtifactError):
        return FallbackReason.MISSING_ARTIFACT
    if isinstance(exception, signing.VersionMismatchError):
        return FallbackReason.VERSION_MISMATCH
    if isinstance(exception, signing.HashMismatchError):
        return FallbackReason.INDEX_CORRUPTION
    if isinstance(exception, signing.ArtifactVerificationError):
        return FallbackReason.INDEX_CORRUPTION
    return None


def classify_fallback(exception: Exception) -> FallbackReason | None:
    """Return a stable :class:`FallbackReason`, or ``None`` if it must propagate.

    Catchable (§3.1): retriever unavailability, provider timeout/transport/rate
    limit/server error, invalid response schema, zero valid candidates, and every
    artifact-verification failure. Everything else propagates.
    """
    if isinstance(exception, exc.RetrieverUnavailable):
        return FallbackReason.RETRIEVAL_FAILURE
    if isinstance(exception, exc.RetrievalProviderError):
        return FallbackReason.PROVIDER_FAILURE
    return _classify_artifact_error(exception)


def execute_with_fallback(
    primary: Callable[[], R], baseline: Callable[[], R]
) -> tuple[R, FallbackReason | None]:
    """Run ``primary``; on a catchable failure, record the reason and run ``baseline``.

    Returns ``(result, reason)`` where ``reason`` is ``None`` on the primary path.
    Non-catchable exceptions (auth/config/parser/internal/programming) and
    cancellation (``KeyboardInterrupt``/``SystemExit``/``CancelledError`` are not
    ``Exception`` subclasses) propagate unchanged.
    """
    try:
        return primary(), None
    except Exception as exception:
        reason = classify_fallback(exception)
        if reason is None:
            raise
        return baseline(), reason
