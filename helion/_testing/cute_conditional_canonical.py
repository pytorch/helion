"""Canonical host adapter for the fixed N32 conditional-attention diagnostic.

The source remains an ordinary two-stage Helion kernel. This adapter only
checks the public ABI, allocates its LSE destination and selects its default
configuration through Helion's guarded bind/compile caches. All partial work
and both device launches remain inside the source's complete call.
"""

from __future__ import annotations

from typing import cast

import torch

from .cute_paged_conditional_semantics import written_conditional_attention


def canonical_arguments(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_kv_len: int,
    window_left: int,
    out: torch.Tensor,
) -> tuple[object, ...]:
    """Validate metadata without reading tensor contents or changing storage."""
    expected = (
        (q, torch.bfloat16, (320, 4, 16, 128), (18432, 4608, 128, 1), 0),
        (k, torch.int8, (7132, 128, 2, 128), (33792, 256, 128, 1), 0),
        (v, torch.int8, (7132, 128, 2, 128), (33792, 256, 128, 1), 0),
        (k_scale, torch.uint8, (7132, 128, 2, 4), (33792, 4, 512, 1), 32768),
        (v_scale, torch.uint8, (7132, 128, 2, 4), (33792, 4, 512, 1), 32768),
        (page_table, torch.int32, (320, 8192), (8192, 1), 0),
        (lengths, torch.int32, (320,), (1,), 0),
        (out, torch.bfloat16, (320, 4, 16, 128), (8192, 2048, 128, 1), 0),
    )
    if max_kv_len != 28027 or window_left != 2047:
        raise ValueError("This diagnostic requires the draft SWA g1023 case")
    for tensor, dtype, shape, stride, offset in expected:
        if (
            tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or tensor.stride() != stride
            or tensor.storage_offset() != offset
            or tensor.device != q.device
        ):
            raise ValueError("Canonical tensor metadata differs from the fixed case")
        span = (
            offset
            + 1
            + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))
        )
        if span * tensor.element_size() > tensor.untyped_storage().nbytes():
            raise ValueError("Canonical tensor view exceeds its backing storage")
        if tensor.data_ptr() % 16:
            raise ValueError("Canonical tensor pointer is not 16-byte aligned")
    for raw, scale in ((k, k_scale), (v, v_scale)):
        if raw.untyped_storage().data_ptr() != scale.untyped_storage().data_ptr():
            raise ValueError("Raw values and scales must share their packed backing")
    roots = [
        tensor.untyped_storage().data_ptr()
        for tensor in (q, k, v, page_table, lengths, out)
    ]
    if len(set(roots)) != len(roots):
        raise ValueError("Canonical independent tensors must have disjoint storage")
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    return q, k, v, k_scale, v_scale, page_table, lengths, out, lse, 148, 18, 2047


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_kv_len: int,
    window_left: int,
    return_lse: bool,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Submit both stages; draft timing still includes source-authored LSE work."""
    if type(return_lse) is not bool:
        raise ValueError("return_lse must be a Boolean")
    args = canonical_arguments(
        q,
        k,
        v,
        k_scale,
        v_scale,
        page_table,
        lengths,
        max_kv_len=max_kv_len,
        window_left=window_left,
        out=out,
    )
    bound = written_conditional_attention.bind(args)
    compiled = bound.compile_config(bound.config_spec.default_config())
    result, lse = compiled(*args)
    if result is not out or lse is not args[8]:
        raise AssertionError(
            "The authored source must return the supplied destinations"
        )
    return result, cast("torch.Tensor", lse) if return_lse else None


def canonical_cpu_fixture() -> tuple[torch.Tensor, ...]:
    """Uninitialized CPU metadata only; GPU gates use the frozen factory values."""
    q_storage = torch.empty((320, 4, 4608), dtype=torch.bfloat16)
    q = q_storage[..., :2048].view(320, 4, 16, 128)
    packed = [torch.empty(7132 * 33792, dtype=torch.int8) for _ in range(2)]
    k, v = [
        storage.as_strided((7132, 128, 2, 128), (33792, 256, 128, 1))
        for storage in packed
    ]
    ks, vs = [
        storage.view(torch.uint8).as_strided(
            (7132, 128, 2, 4), (33792, 4, 512, 1), 32768
        )
        for storage in packed
    ]
    table = torch.empty((320, 8192), dtype=torch.int32)
    lengths = torch.empty((320,), dtype=torch.int32)
    output = torch.empty((320, 4, 16, 128), dtype=torch.bfloat16)
    return q, k, v, ks, vs, table, lengths, output
