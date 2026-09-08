"""Local HBM/VMEM DMA planning and code generation helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

import torch

from ...exc import InvalidConfig
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState
    from .tensorcore_plan import DmaAccessPlan


DmaDirection = Literal["load", "store"]


def is_tpu_dma_aligned_shape(shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    """Whether a concrete VMEM shape satisfies TPU local-DMA alignment."""
    if len(shape) >= 2:
        return shape[-1] % 128 == 0 and shape[-2] % 8 == 0
    if len(shape) == 1:
        bitwidth = min(dtype.itemsize * 8, 32)
        return shape[0] % (128 * (32 // bitwidth)) == 0
    return True


@dataclass(frozen=True, eq=False)
class DmaTransfer:
    """One local HBM/VMEM transfer before resources are allocated."""

    tensor: torch.Tensor
    subscript: tuple[object, ...]
    direction: DmaDirection


@dataclass(frozen=True, eq=False)
class IndirectDmaTransfer(DmaTransfer):
    """A local DMA transfer associated with an indirect access plan."""

    plan: DmaAccessPlan


@dataclass(frozen=True)
class DmaResources:
    """Compile-time scratch and semaphore resources for one transfer."""

    scratch: str
    semaphore: str
    buffer_count: int

    def scratch_ref(self, stage: str | None) -> str:
        if stage is None:
            return self.scratch
        assert self.buffer_count > 1
        return f"{self.scratch}.at[{stage}]"

    def semaphore_ref(self, stage: str | None) -> str:
        if stage is None:
            return self.semaphore
        assert self.buffer_count > 1
        return f"{self.semaphore}.at[{stage}]"


@dataclass(frozen=True, eq=False)
class ScheduledDmaTransfer:
    """A transfer paired with resources allocated for one compiled config."""

    transfer: DmaTransfer
    resources: DmaResources


def allocate_dma_resources(
    device_function: DeviceFunction,
    transfer: DmaTransfer,
    *,
    vmem_shape: tuple[int, ...],
    buffer_count: int,
    scratch_hint: str,
    semaphore_hint: str,
    shape_sources: tuple[tuple[torch.Tensor, int] | None, ...] | None = None,
) -> DmaResources:
    """Allocate the scratch and semaphore used by a local DMA transfer."""
    assert buffer_count in (1, 2)
    scratch_shape = (buffer_count, *vmem_shape) if buffer_count > 1 else vmem_shape
    if buffer_count > 1 and shape_sources is not None:
        shape_sources = (None, *shape_sources)
    scratch = device_function.register_scratch(
        scratch_shape,
        transfer.tensor.dtype,
        name_hint=scratch_hint,
        shape_sources=shape_sources,
    )
    semaphore = device_function.register_dma_semaphore(
        name_hint=semaphore_hint,
        shape=(buffer_count,) if buffer_count > 1 else (),
    )
    return DmaResources(scratch, semaphore, buffer_count)


def allocate_indirect_dma_resources(
    device_function: DeviceFunction,
    transfer: IndirectDmaTransfer,
    *,
    buffer_count: int,
    load_resources: DmaResources | None = None,
) -> DmaResources:
    """Allocate a gather stage or reuse it for its paired writeback."""
    base = device_function.tensor_arg(transfer.tensor).name.replace("_hbm", "")
    if transfer.direction == "load":
        return allocate_dma_resources(
            device_function,
            transfer,
            vmem_shape=transfer.plan.transfer_shape,
            buffer_count=buffer_count,
            scratch_hint=f"{base}_gather_buf",
            semaphore_hint=f"{base}_gather_sem",
        )
    if load_resources is None:
        raise InvalidConfig("indirect DMA writeback has no paired load resources")
    assert buffer_count == load_resources.buffer_count == 1
    return DmaResources(
        load_resources.scratch,
        device_function.register_dma_semaphore(name_hint=f"{base}_scatter_sem"),
        1,
    )


def async_copy_statements(
    state: CodegenState,
    source: str,
    destination: str,
    semaphore: str,
    methods: tuple[str, ...],
    name_hint: str,
) -> list[ast.stmt]:
    """Create an async-copy handle and emit its requested operations."""
    copy_name = state.device_function.new_var(name_hint)
    return [
        statement_from_string(
            f"{copy_name} = pltpu.make_async_copy({source}, {destination}, {semaphore})"
        ),
        *(statement_from_string(f"{copy_name}.{method}()") for method in methods),
    ]


def indirect_group_statements(
    state: CodegenState,
    *,
    group_count: int,
    index_name: str,
    member_hbm: str,
    aggregate_hbm: str,
    scratch_ref: str,
    semaphore_ref: str,
    direction: DmaDirection,
    methods: tuple[str, ...],
) -> list[ast.stmt]:
    """Emit per-member starts and an aggregate wait for an indirect group."""
    result: list[ast.stmt] = []
    if "start" in methods:
        # Duplicate hl.store destinations have undefined ordering in compiled
        # mode, so a scatter group need not serialize runtime index collisions.
        lane_name = state.device_function.new_var("_dma_member")
        member_hbm = member_hbm.replace("{index}", f"{index_name}[{lane_name}]")
        member_vmem = f"{scratch_ref}.at[{lane_name}]"
        loop = statement_from_string(
            f"for {lane_name} in range({group_count}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        loop.body = async_copy_statements(
            state,
            member_vmem if direction == "store" else member_hbm,
            member_hbm if direction == "store" else member_vmem,
            semaphore_ref,
            ("start",),
            "_scatter_copy" if direction == "store" else "_gather_copy",
        )
        result.append(loop)
    if "wait" in methods:
        # Every member start contributes to the same semaphore. The aggregate
        # handle describes the total byte count, so one wait covers the group.
        result.extend(
            async_copy_statements(
                state,
                scratch_ref if direction == "store" else aggregate_hbm,
                aggregate_hbm if direction == "store" else scratch_ref,
                semaphore_ref,
                ("wait",),
                "_scatter_wait" if direction == "store" else "_gather_wait",
            )
        )
    return result


def emit_immediate_indirect_transfer(
    state: CodegenState,
    plan: DmaAccessPlan,
    name: str,
) -> None:
    """Emit a start/wait transfer at an indirect load or store site."""
    from . import codegen as pallas_codegen
    from .memory_access import MemoryAccessKind

    resources = pallas_codegen.grid_memory_op_dma_binding(state)
    if resources is None and plan.spec.index_access is None:
        resources = pallas_codegen.fori_memory_op_dma_binding(state)
    if resources is None:
        return
    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_index = ast_subscripts[0]
    assert isinstance(ast_index, ast.AST)
    index_name = state.codegen.lift(ast_index, dce=False, prefix="index").id
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    member_parts = [*parts]
    member_parts[0] = "{index}"
    aggregate_parts = [*parts]
    aggregate_parts[0] = f"pl.ds(0, {plan.group_count})"
    member_hbm = f"{name}.at[{', '.join(member_parts)}]"
    aggregate_hbm = f"{name}.at[{', '.join(aggregate_parts)}]"
    direction: DmaDirection = (
        "load" if plan.access.kind is MemoryAccessKind.LOAD else "store"
    )
    for statement in indirect_group_statements(
        state,
        group_count=plan.group_count,
        index_name=index_name,
        member_hbm=member_hbm,
        aggregate_hbm=aggregate_hbm,
        scratch_ref=resources.scratch,
        semaphore_ref=resources.semaphore,
        direction=direction,
        methods=("start", "wait"),
    ):
        state.codegen.add_statement(statement)
