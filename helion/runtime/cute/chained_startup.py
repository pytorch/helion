"""Current-pointer validation and wrapper construction for startup TMA."""

from __future__ import annotations

from typing import cast

import torch

from ... import exc
from .tma_tile import append_tma_tile

KIND = "chained_startup_tma"


def validate_arguments(kernel: object, args: tuple[object, ...]) -> None:
    """Run before every cache lookup; never retain tensors or CUDA streams."""
    plans = getattr(kernel, "_helion_cute_wrapper_plans", ())
    for plan in plans:
        if plan.get("kind") != KIND:
            continue
        source, output = args[plan["lhs_idx"]], args[plan["out_idx"]]
        if not isinstance(source, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise exc.BackendUnsupported(
                "cute", "startup TMA requires tensor arguments"
            )
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[plan["dtype"]]
        if (
            source.device.type != "cuda"
            or source.device != output.device
            or source.layout != torch.strided
            or source.is_conj()
            or source.is_neg()
            or source.dtype != dtype
            or tuple(source.shape) != tuple(plan["shape"])
            or source.stride() != tuple(plan["strides"])
            or source.data_ptr() % 16
        ):
            raise exc.BackendUnsupported(
                "cute", "startup TMA current layout/alignment mismatch"
            )
        storage = source.untyped_storage()
        start = source.data_ptr()
        end = start + source.numel() * source.element_size()
        if not (
            0
            < storage.data_ptr()
            <= start
            < end
            <= storage.data_ptr() + storage.nbytes()
            < 2**64
        ):
            raise exc.BackendUnsupported(
                "cute", "startup TMA input storage span mismatch"
            )
        out_storage = output.untyped_storage()
        out_start, out_end = (
            out_storage.data_ptr(),
            out_storage.data_ptr() + out_storage.nbytes(),
        )
        for index, tensor in enumerate(args):
            if index == plan["out_idx"] or not isinstance(tensor, torch.Tensor):
                continue
            other = tensor.untyped_storage()
            if max(out_start, other.data_ptr()) < min(
                out_end, other.data_ptr() + other.nbytes()
            ):
                raise exc.BackendUnsupported(
                    "cute", "startup TMA output aliases a read argument"
                )


def append_wrapper(
    body: list[str], call_args: list[str], plan: dict[str, object]
) -> None:
    """Build descriptors from this call's pointer; shapes are metadata-guarded."""
    atom, tensor = cast("list[str]", plan["kernel_args"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        cast("str", plan["dtype"])
    ]
    append_tma_tile(
        body,
        call_args,
        source_index=cast("int", plan["lhs_idx"]),
        atom=atom,
        tensor=tensor,
        rows=cast("int", plan["rows"]),
        columns=cast("int", plan["columns"]),
        tile=cast("tuple[int, int]", plan["tile"]),
        dtype=dtype,
    )
