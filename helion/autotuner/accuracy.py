from __future__ import annotations

import math

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map_only

_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def assert_close(
    actual: object,
    expected: object,
    atol: float,
    rtol: float,
    *,
    scale_atol_by_expected_rms: bool = False,
) -> None:
    """Like torch.testing.assert_close, with fp8 and large tensor handling.

    With ``scale_atol_by_expected_rms`` (used when the caller did not specify
    an explicit atol), the absolute-tolerance floor for each tensor leaf is
    ``atol * max(1, rms(expected))``: a near-zero element of a large reduction
    legitimately differs across accumulation orders by an amount proportional
    to the global output scale, not to the element's own magnitude, so a fixed
    elementwise atol falsely rejects every config that restructures the
    reduction (e.g. all split_k > 1 matmul configs). The scaling never
    tightens the check.
    """

    def convert(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.uint8) if t.dtype in _FP8_DTYPES else t

    actual_flat, actual_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, actual)
    )
    expected_flat, expected_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, expected)
    )

    if actual_spec != expected_spec:
        raise AssertionError(
            f"Output tree structure mismatch during autotuner accuracy check:\n"
            f"  actual:   {actual_spec} ({len(actual_flat)} leaves)\n"
            f"  expected: {expected_spec} ({len(expected_flat)} leaves)"
        )

    for actual_leaf, expected_leaf in zip(actual_flat, expected_flat, strict=True):
        if isinstance(actual_leaf, torch.Tensor):
            if not isinstance(expected_leaf, torch.Tensor):
                raise AssertionError(
                    "Output leaf type mismatch during autotuner accuracy check: "
                    f"actual is Tensor, expected is {type(expected_leaf).__name__}"
                )
            _chunked_assert_close(
                actual_leaf,
                expected_leaf,
                atol=atol,
                rtol=rtol,
                scale_atol_by_expected_rms=scale_atol_by_expected_rms,
            )
        elif isinstance(actual_leaf, str):
            if not isinstance(expected_leaf, str):
                raise AssertionError(f"Type mismatch {actual_leaf} vs {expected_leaf}")
            if actual_leaf != expected_leaf:
                raise AssertionError(
                    f"string mismatch {actual_leaf} vs {expected_leaf}"
                )
        else:
            torch.testing.assert_close(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


def _assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    assert_close(actual, expected, atol=atol, rtol=rtol)


_RMS_CACHE_ATTR = "_helion_accuracy_rms"


def _chunked_rms(t: torch.Tensor, chunk_size: int) -> float:
    # The autotuner compares hundreds of candidates against one fixed
    # baseline per run; memoize the RMS on the tensor (keyed by its mutation
    # version) instead of re-reading the whole output for every candidate.
    cached = getattr(t, _RMS_CACHE_ATTR, None)
    if cached is not None and cached[0] == t._version:
        return cached[1]
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return 0.0
    total = 0.0
    for start in range(0, flat.numel(), chunk_size):
        # Accumulate in float64 so large-magnitude outputs cannot overflow
        # the sum of squares.
        total += float(flat[start : start + chunk_size].double().square().sum())
    rms = math.sqrt(total / flat.numel())
    setattr(t, _RMS_CACHE_ATTR, (t._version, rms))
    return rms


def _chunked_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,
    scale_atol_by_expected_rms: bool = False,
) -> None:
    """Memory-efficient assert_close for large tensors."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Tensor shape mismatch during autotuner accuracy check: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if scale_atol_by_expected_rms and expected.dtype.is_floating_point:
        rms = _chunked_rms(expected, chunk_size)
        # A non-finite RMS (inf in the baseline output) must not disable the
        # gate; keep the unscaled floor in that case.
        if math.isfinite(rms):
            atol = atol * max(1.0, rms)
    if actual.numel() <= chunk_size:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    for start in range(0, actual_flat.numel(), chunk_size):
        actual_chunk = actual_flat[start : start + chunk_size]
        expected_chunk = expected_flat[start : start + chunk_size]
        torch.testing.assert_close(actual_chunk, expected_chunk, atol=atol, rtol=rtol)
