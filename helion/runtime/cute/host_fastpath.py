"""Current-call selection for proven device memory fastpaths; no pointer cache."""

from __future__ import annotations

from itertools import starmap
from typing import cast

import torch

from ... import exc

Contract = tuple[tuple[int, ...], tuple[int, ...], str, int]


def validate_trace() -> None:
    if (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.jit.is_scripting()
    ):
        raise exc.BackendUnsupported(
            "cute", "host-selected fastpath requires eager current-call dispatch"
        )


def _is_cuda(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda"


def _span(value: object, contract: Contract) -> tuple[torch.device, int, int] | None:
    if type(value) is not torch.Tensor:
        return None
    tensor = cast("torch.Tensor", value)
    if (
        tensor.layout != torch.strided
        or tensor.is_conj()
        or tensor.is_neg()
        or tensor.is_quantized
        or tensor.requires_grad
        or not _is_cuda(tensor)
    ):
        return None
    shape, stride, dtype, element = contract
    if (
        tuple(tensor.shape) != shape
        or tuple(tensor.stride()) != stride
        or str(tensor.dtype) != dtype
        or tensor.element_size() != element
        or any(n <= 0 for n in shape)
        or any(n < 0 for n in stride)
    ):
        return None
    pointer = tensor.data_ptr()
    storage = tensor.untyped_storage()
    start, size = storage.data_ptr(), storage.nbytes()
    offset = tensor.storage_offset()
    limit = pointer + element * (
        1 + sum((n - 1) * d for n, d in zip(shape, stride, strict=True))
    )
    if (
        pointer % 16
        or offset < 0
        or pointer != start + offset * element
        or not 0 < start <= pointer < limit <= start + size < 2**63
    ):
        return None
    return tensor.device, start, start + size


@torch.compiler.disable
def select_kernel(
    selected: object,
    original: object,
    fast: object,
    values: tuple[object, ...],
    contracts: tuple[Contract, ...],
    grid: tuple[int, ...],
    expected_grid: tuple[int, int, int],
    block: tuple[int, int, int],
    expected_block: tuple[int, int, int],
) -> object:
    """Never override another selector's fallback; check fresh locals every call."""
    if selected is not original or len(values) != len(contracts):
        return selected
    if any(type(n) is not int or n <= 0 for n in (*grid, *block)):
        return selected
    if (
        not 1 <= len(grid) <= 3
        or (*grid, *((1,) * (3 - len(grid)))) != expected_grid
        or block != expected_block
    ):
        return selected
    spans = list(starmap(_span, zip(values, contracts, strict=True)))
    if any(span is None for span in spans):
        return selected
    complete = cast("list[tuple[torch.device, int, int]]", spans)
    if len({span[0] for span in complete}) != 1:
        return selected
    # Conservative whole-storage nonaliasing, including read/read aliases.
    # No write-set inference or previous bind-time disjointness is trusted.
    for i, left in enumerate(complete):
        for right in complete[i + 1 :]:
            if not (left[2] <= right[1] or right[2] <= left[1]):
                return selected
    return fast
