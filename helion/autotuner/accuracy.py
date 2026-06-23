from __future__ import annotations

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


def assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    """Like torch.testing.assert_close, with fp8 and large tensor handling."""

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
            _chunked_assert_close(actual_leaf, expected_leaf, atol=atol, rtol=rtol)
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


def _chunked_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,
) -> None:
    """Memory-efficient assert_close for large tensors."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Tensor shape mismatch during autotuner accuracy check: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.numel() <= chunk_size:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    for start in range(0, actual_flat.numel(), chunk_size):
        actual_chunk = actual_flat[start : start + chunk_size]
        expected_chunk = expected_flat[start : start + chunk_size]
        torch.testing.assert_close(actual_chunk, expected_chunk, atol=atol, rtol=rtol)
