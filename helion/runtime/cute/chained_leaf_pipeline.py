"""Current-call guards and descriptor construction for paired FP32 leaves."""

from __future__ import annotations

from typing import cast

import torch

from ... import exc
from .tma_tile import append_tma_tile

KIND = "chained_paired_leaf_tma"


def validate_plan(plan: dict[str, object], args: tuple[object, ...]) -> None:
    """No pointer or tensor snapshot: validate before every launcher cache."""
    source_index, output_index = (
        cast("int", plan["lhs_idx"]),
        cast("int", plan["out_idx"]),
    )
    source, output = args[source_index], args[output_index]
    if type(source) is not torch.Tensor or type(output) is not torch.Tensor:
        raise exc.BackendUnsupported("cute", "paired leaf requires ordinary tensors")
    if (
        source.device.type != "cuda"
        or source.device != output.device
        or source.layout != torch.strided
        or output.layout != torch.strided
        or output.is_conj()
        or output.is_neg()
        or source.is_conj()
        or source.is_neg()
        or source.dtype != torch.float32
        or tuple(source.shape) != tuple(cast("tuple[int, ...]", plan["shape"]))
        or source.stride() != tuple(cast("tuple[int, ...]", plan["strides"]))
        or source.data_ptr() % 16
    ):
        raise exc.BackendUnsupported(
            "cute", "paired leaf current layout/alignment mismatch"
        )
    storage = source.untyped_storage()
    start, end = source.data_ptr(), source.data_ptr() + source.numel() * 4
    if not (
        0
        < storage.data_ptr()
        <= start
        < end
        <= storage.data_ptr() + storage.nbytes()
        < 2**64
    ):
        raise exc.BackendUnsupported("cute", "paired leaf source storage span mismatch")
    output_storage = output.untyped_storage()
    lo, hi = (
        output_storage.data_ptr(),
        output_storage.data_ptr() + output_storage.nbytes(),
    )
    if not (0 < lo < hi < 2**64):
        raise exc.BackendUnsupported("cute", "paired leaf output storage span mismatch")
    if any(size <= 0 for size in output.shape) or any(
        stride < 0 for stride in output.stride()
    ):
        raise exc.BackendUnsupported("cute", "paired leaf output extent mismatch")
    output_end = output.data_ptr() + output.element_size() * (
        1
        + sum(
            (size - 1) * stride
            for size, stride in zip(output.shape, output.stride(), strict=True)
        )
    )
    if not lo <= output.data_ptr() < output_end <= hi:
        raise exc.BackendUnsupported(
            "cute", "paired leaf output exceeds current storage"
        )
    # Whole-storage exclusion covers every read dependency, including scalar
    # coefficient tensors and cross-dtype views. Read/read aliasing is harmless.
    for index, arg in enumerate(args):
        if index == output_index or not isinstance(arg, torch.Tensor):
            continue
        if (
            type(arg) is not torch.Tensor
            or arg.layout != torch.strided
            or arg.is_conj()
            or arg.is_neg()
            or arg.device != source.device
        ):
            raise exc.BackendUnsupported(
                "cute", "paired leaf unsupported current tensor"
            )
        other = arg.untyped_storage()
        if max(lo, other.data_ptr()) < min(hi, other.data_ptr() + other.nbytes()):
            raise exc.BackendUnsupported(
                "cute", "paired leaf output aliases a read argument"
            )


def append_wrapper(
    body: list[str], call_args: list[str], plan: dict[str, object]
) -> None:
    atom, tensor = cast("list[str]", plan["kernel_args"])
    append_tma_tile(
        body,
        call_args,
        source_index=cast("int", plan["lhs_idx"]),
        atom=atom,
        tensor=tensor,
        rows=cast("int", plan["rows"]),
        columns=cast("int", plan["columns"]),
        tile=(128, 32),
        dtype=torch.float32,
    )
