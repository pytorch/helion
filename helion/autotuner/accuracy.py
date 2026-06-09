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


def _assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    """Like torch.testing.assert_close but handles fp8 and uses chunked comparison for large tensors."""

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

    for a, e in zip(actual_flat, expected_flat, strict=True):
        if isinstance(a, torch.Tensor):
            _chunked_assert_close(a, e, atol=atol, rtol=rtol)
        elif isinstance(a, str):
            if not isinstance(e, str):
                raise AssertionError(f"Type mismatch {a} vs {e}")
            if a != e:
                raise AssertionError(f"string mismatch {a} vs {e}")
        else:
            torch.testing.assert_close(a, e, atol=atol, rtol=rtol)


def _chunked_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,  # ~4M elements per chunk
) -> None:
    """Memory-efficient assert_close for large tensors.

    Processes the comparison in chunks to avoid allocating multiple
    full-size temporary tensors.  Uses torch.testing.assert_close on
    each chunk so error messages retain full detail.
    """
    if actual.numel() <= chunk_size:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    a_flat = actual.reshape(-1)
    e_flat = expected.reshape(-1)
    for i in range(0, a_flat.numel(), chunk_size):
        a_chunk = a_flat[i : i + chunk_size]
        e_chunk = e_flat[i : i + chunk_size]
        torch.testing.assert_close(a_chunk, e_chunk, atol=atol, rtol=rtol)
