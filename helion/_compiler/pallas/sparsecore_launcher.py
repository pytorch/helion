"""Build launcher input and output transforms for SparseCore."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .sc_base import _CAST_STORE_DTYPES
from .sc_base import SC_DMA_GRANULE_BYTES
from .sc_base import SC_LANES
from .sc_base import _reject
from .sparsecore_plan import DirectLoadPlan
from .sparsecore_plan import DirectStorePlan
from .sparsecore_plan import IndirectLoadPlan
from .sparsecore_plan import IndirectStorePlan
from .sparsecore_plan import SparseCoreMemoryPlan

if TYPE_CHECKING:
    from collections.abc import Callable

    from .sparsecore_program import SparseCoreProgram


def _shape(tensor: torch.Tensor) -> list[int]:
    from ..compile_environment import CompileEnvironment

    result = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, torch.SymInt):
            result.append(CompileEnvironment.current().size_hint(dim))
        else:
            _reject("launcher", f"output has dynamic shape {tuple(tensor.shape)}")
    return result


def _index_padding(program: SparseCoreProgram) -> dict[torch.fx.Node, int]:
    fills: dict[torch.fx.Node, int] = {}
    for plan in program.memory_plans:
        if not isinstance(plan, (IndirectLoadPlan, IndirectStorePlan)):
            continue
        fill = (
            0
            if isinstance(plan, IndirectLoadPlan)
            else int(plan.access.tensor.shape[0])
        )
        prior = fills.setdefault(plan.index_node, fill)
        if prior != fill:
            _reject(
                "launcher",
                "one index needs different gather and scatter padding; "
                "load it separately for each use",
                node=plan.access.node,
            )
    return fills


def build_sparsecore_launcher_spec(
    program: SparseCoreProgram,
    arg_position: Callable[[torch.Tensor], int | None],
) -> dict[str, object]:
    index_padding = _index_padding(program)
    pad_inputs: list[list[object]] = []
    stream_inputs: list[list[object]] = []
    seen_inputs: dict[int, tuple[object, ...]] = {}
    emitted_inputs: set[int] = set()

    def record_input(
        position: int,
        transform: tuple[object, ...],
        plan: SparseCoreMemoryPlan,
    ) -> bool:
        prior = seen_inputs.setdefault(position, transform)
        if prior != transform:
            _reject(
                "launcher",
                "one input needs incompatible launcher transforms",
                node=plan.access.node,
            )
        if position in emitted_inputs:
            return False
        emitted_inputs.add(position)
        return True

    for plan in program.loads:
        position = arg_position(plan.access.tensor)
        if position is None:
            continue
        if isinstance(plan, DirectLoadPlan):
            stream = plan.stream
            assert stream is not None
            if plan.access.node in index_padding:
                fill = index_padding.get(plan.access.node, 0)
                transform = (
                    "flat",
                    stream.group_count,
                    program.padded_items * stream.elements_per_item,
                    fill,
                )
                if record_input(position, transform, plan):
                    pad_inputs.append(
                        [
                            position,
                            stream.group_count,
                            program.padded_items * stream.elements_per_item,
                            True,
                            fill,
                        ]
                    )
            else:
                value_size = stream.elements_per_item
                stored_size = plan.layout.storage_shape[1]
                transform = (
                    "window",
                    stream.group_count,
                    program.padded_items,
                    value_size,
                    stored_size,
                )
                if record_input(position, transform, plan):
                    stream_inputs.append(
                        [
                            position,
                            stream.group_count,
                            program.padded_items,
                            value_size,
                            stored_size,
                        ]
                    )
        else:
            record_input(position, ("raw",), plan)

    pad_outputs: list[list[object]] = []
    reshape_outputs: list[list[object]] = []
    scalar_outputs: list[list[object]] = []
    kernel_output_dtypes: list[list[object]] = []
    for plan in program.stores:
        position = arg_position(plan.access.tensor)
        if position is None:
            raise AssertionError("SparseCore output is absent from launcher arguments")
        logical = _shape(plan.access.tensor)
        value_size = math.prod(logical[1:]) if len(logical) > 1 else 1
        if plan.layout.logical_dtype in _CAST_STORE_DTYPES:
            kernel_output_dtypes.append([position, "int32"])
        if isinstance(plan, DirectStorePlan):
            if value_size == 1:
                pad_outputs.append([position, [program.padded_items, SC_LANES]])
                scalar_outputs.append([position, logical])
            else:
                pad_outputs.append([position, [program.padded_items, value_size]])
                reshape_outputs.append([position, logical])
        elif isinstance(plan, IndirectStorePlan):
            pad_outputs.append([position, [logical[0] + 1, *logical[1:]]])

    return {
        "pad_inputs": pad_inputs,
        "stream_inputs": stream_inputs,
        "pad_outputs": pad_outputs,
        "reshape_outputs": reshape_outputs,
        "scalar_outputs": scalar_outputs,
        "kernel_output_dtypes": kernel_output_dtypes,
        "num_cores": program.num_cores,
        "num_subcores": program.num_subcores,
        "dma_granule": SC_DMA_GRANULE_BYTES,
    }
