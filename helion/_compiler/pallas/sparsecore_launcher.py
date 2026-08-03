"""Build launcher input and output transforms for SparseCore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...runtime.pallas.sparsecore_launcher import SparseCoreLauncherSpec
from .sparsecore_base import SC_DMA_GRANULE_BYTES
from .sparsecore_base import SC_LANES
from .sparsecore_base import _reject
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
) -> SparseCoreLauncherSpec:
    index_padding = _index_padding(program)
    index_inputs: list[tuple[int, int, int, int]] = []
    value_inputs: list[tuple[int, int, int, int, int]] = []
    seen_inputs: dict[int, tuple[object, ...]] = {}

    def record_input(
        position: int,
        transform: tuple[object, ...],
        plan: SparseCoreMemoryPlan,
    ) -> bool:
        prior = seen_inputs.get(position)
        if prior is None:
            seen_inputs[position] = transform
            return True
        if prior != transform:
            _reject(
                "launcher",
                "one input needs incompatible launcher transforms",
                node=plan.access.node,
            )
        return False

    for plan in program.loads:
        position = arg_position(plan.access.tensor)
        if position is None:
            continue
        if isinstance(plan, DirectLoadPlan):
            transfer = plan.transfer
            if plan.access.node in index_padding:
                fill = index_padding[plan.access.node]
                transform = (
                    "flat",
                    transfer.prefix_count,
                    program.padded_items * transfer.elements_per_item,
                    fill,
                )
                if record_input(position, transform, plan):
                    index_inputs.append(
                        (
                            position,
                            transfer.prefix_count,
                            program.padded_items * transfer.elements_per_item,
                            fill,
                        )
                    )
            else:
                value_size = transfer.elements_per_item
                stored_size = plan.layout.storage_shape[1]
                transform = (
                    "window",
                    transfer.prefix_count,
                    program.padded_items,
                    value_size,
                    stored_size,
                )
                if record_input(position, transform, plan):
                    value_inputs.append(
                        (
                            position,
                            transfer.prefix_count,
                            program.padded_items,
                            value_size,
                            stored_size,
                        )
                    )
        else:
            record_input(position, ("raw",), plan)

    output_shapes: list[tuple[int, tuple[int, ...]]] = []
    reshape_outputs: list[int] = []
    scalar_outputs: list[int] = []
    int32_outputs: list[int] = []
    for plan in program.stores:
        position = arg_position(plan.access.tensor)
        if position is None:
            raise AssertionError("SparseCore output is absent from launcher arguments")
        logical = _shape(plan.access.tensor)
        value_size = plan.layout.elements_per_item
        if plan.layout.storage_dtype is torch.int32:
            int32_outputs.append(position)
        if isinstance(plan, DirectStorePlan):
            if value_size == 1:
                output_shapes.append((position, (program.padded_items, SC_LANES)))
                scalar_outputs.append(position)
            else:
                output_shapes.append((position, (program.padded_items, value_size)))
                reshape_outputs.append(position)
        elif isinstance(plan, IndirectStorePlan):
            output_shapes.append((position, (logical[0] + 1, *logical[1:])))

    return SparseCoreLauncherSpec(
        index_inputs=index_inputs,
        value_inputs=value_inputs,
        output_shapes=output_shapes,
        reshape_outputs=reshape_outputs,
        scalar_outputs=scalar_outputs,
        int32_outputs=int32_outputs,
        num_cores=program.num_cores,
        num_subcores=program.num_subcores,
        dma_granule=SC_DMA_GRANULE_BYTES,
    )
