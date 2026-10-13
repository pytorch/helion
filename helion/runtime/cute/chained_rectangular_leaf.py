"""Current-pointer and all-write guards for rectangular raw-leaf TensorMaps."""

from __future__ import annotations

from typing import cast

import torch

from ... import exc
from .tma_tile import append_tma_tile

KIND = "chained_rectangular_leaf_tma"
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def validate_plan(plan: dict[str, object], args: tuple[object, ...]) -> None:
    """Validate each call before caches; retain no tensors, pointers or streams.

    The compiler supplies the exhaustive set of graph-written arguments. Empty
    non-descriptor tensors are valid for zero-trip loops. Storage exclusion is
    deliberately conservative: two disjoint views of one written storage are
    rejected, while read/read aliasing does not affect the transport contract.
    """
    source_index = plan["source_idx"]
    writes = plan["write_indices"]
    if (
        type(source_index) is not int
        or not 0 <= source_index < len(args)
        or not isinstance(writes, (list, tuple))
        or not writes
        or any(type(index) is not int or not 0 <= index < len(args) for index in writes)
        or len(set(writes)) != len(writes)
        or source_index in writes
    ):
        raise exc.BackendUnsupported(
            "cute", "rectangular leaf argument indices mismatch"
        )
    source = args[source_index]
    if type(source) is not torch.Tensor or any(
        type(args[index]) is not torch.Tensor for index in writes
    ):
        raise exc.BackendUnsupported(
            "cute", "rectangular leaf requires ordinary tensors"
        )
    dtype_name = cast("str", plan["dtype"])
    rows, columns = plan["rows"], plan["columns"]
    if (
        type(rows) is not int
        or type(columns) is not int
        or rows <= 0
        or columns <= 0
        or rows * columns != source.numel()
    ):
        raise exc.BackendUnsupported(
            "cute", "rectangular leaf descriptor extent mismatch"
        )
    if (
        dtype_name not in _DTYPES
        or source.device.type != "cuda"
        or source.layout != torch.strided
        or source.is_conj()
        or source.is_neg()
        or source.dtype != _DTYPES[dtype_name]
        or tuple(source.shape) != tuple(cast("tuple[int, ...]", plan["shape"]))
        or source.stride() != tuple(cast("tuple[int, ...]", plan["strides"]))
        or source.data_ptr() % 16
        or source.numel() <= 0
        or any(size <= 0 for size in source.shape)
        or any(stride <= 0 for stride in source.stride())
    ):
        raise exc.BackendUnsupported(
            "cute", "rectangular leaf current layout/alignment mismatch"
        )
    compact = 1
    for stride, size in sorted(zip(source.stride(), source.shape, strict=True)):
        if size != 1:
            if stride != compact:
                raise exc.BackendUnsupported(
                    "cute", "rectangular leaf requires compact storage"
                )
            compact *= size
    spans: dict[int, tuple[int, int]] = {}
    for index, tensor in enumerate(args):
        if not isinstance(tensor, torch.Tensor):
            continue
        if (
            type(tensor) is not torch.Tensor
            or tensor.device != source.device
            or tensor.layout != torch.strided
            or tensor.is_conj()
            or tensor.is_neg()
            or any(size < 0 for size in tensor.shape)
            or any(stride < 0 for stride in tensor.stride())
        ):
            raise exc.BackendUnsupported(
                "cute", "rectangular leaf unsupported current tensor"
            )
        storage = tensor.untyped_storage()
        begin, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        if not (0 <= begin <= end < 2**64) or (storage.nbytes() and not begin):
            raise exc.BackendUnsupported(
                "cute", "rectangular leaf storage span mismatch"
            )
        if tensor.numel():
            pointer = tensor.data_ptr()
            reachable_end = pointer + tensor.element_size() * (
                1
                + sum(
                    (size - 1) * stride
                    for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
                )
            )
            if not 0 < begin <= pointer < reachable_end <= end:
                raise exc.BackendUnsupported(
                    "cute", "rectangular leaf tensor exceeds current storage"
                )
        spans[index] = begin, end
    for output_index in writes:
        begin, end = spans[output_index]
        for index, (other_begin, other_end) in spans.items():
            if index != output_index and max(begin, other_begin) < min(end, other_end):
                raise exc.BackendUnsupported(
                    "cute", "rectangular leaf write aliases another argument"
                )


def append_wrapper(
    body: list[str], call_args: list[str], plan: dict[str, object]
) -> None:
    atom, tensor = cast("list[str]", plan["kernel_args"])
    append_tma_tile(
        body,
        call_args,
        source_index=cast("int", plan["source_idx"]),
        atom=atom,
        tensor=tensor,
        rows=cast("int", plan["rows"]),
        columns=cast("int", plan["columns"]),
        tile=cast("tuple[int, int]", plan["tile"]),
        dtype=_DTYPES[cast("str", plan["dtype"])],
    )
