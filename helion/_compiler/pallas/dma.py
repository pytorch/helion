"""Shared local HBM/VMEM DMA planning and code generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    import ast

    import torch

    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState


DmaDirection = Literal["load", "store"]


@dataclass(frozen=True, eq=False)
class DmaTransfer:
    """One local HBM/VMEM transfer before resources are allocated."""

    tensor: torch.Tensor
    subscript: tuple[object, ...]
    direction: DmaDirection
    vmem_shape: tuple[int, ...]


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
    buffer_count: int,
    scratch_hint: str,
    semaphore_hint: str,
    shape_sources: tuple[tuple[torch.Tensor, int] | None, ...] | None = None,
) -> DmaResources:
    """Allocate the scratch and semaphore used by a local DMA transfer."""
    assert buffer_count in (1, 2)
    scratch_shape = (
        (buffer_count, *transfer.vmem_shape)
        if buffer_count > 1
        else transfer.vmem_shape
    )
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
