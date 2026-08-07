"""SparseCore backend: hardware constants, rejection machinery, helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import NoReturn

import torch
import torch.fx

from ... import exc

SC_LANES = 16
# Per-subcore VMEM budget on v7x, minus a safety margin for compiler-managed
# allocations (measured: W1 bf16 at 128 items/subcore sat at the limit).
SC_VMEM_BYTES = 512 * 1024
SC_VMEM_MARGIN = 32 * 1024
# 1-D 32-bit VMEM memref slice offsets must be multiples of 8 (observed
# Mosaic-SC constraint; hit as a late MLO error otherwise).
SC_SLICE_MULTIPLE = 8
# Plans are built for this DMA granule; the launcher verifies the active
# target does not require a larger one.
SC_DMA_GRANULE_BYTES = 32


def sc_hardware_info() -> tuple[int, int] | None:
    """(num_cores, num_subcores), or None when the target has no SparseCore.

    The mesh is baked into the generated kernel grid, so the launcher checks
    the active mesh against the serialized spec instead of re-querying.  Set
    ``HELION_SC_ASSUME_MESH=<cores>x<subcores>`` to compile SC code without the
    hardware, which codegen-text tests use.
    """
    forced = os.environ.get("HELION_SC_ASSUME_MESH")
    if forced:
        cores, _, subcores = forced.partition("x")
        return int(cores), int(subcores)
    try:
        from jax.experimental.pallas import tpu as pltpu

        sc = pltpu.get_tpu_info().sparse_core
    except Exception:
        return None
    if sc is None:
        return None
    return int(sc.num_cores), int(sc.num_subcores)


@dataclass(frozen=True)
class SparseCoreRejection:
    """One stable reason why a config cannot lower on the SparseCore."""

    code: str
    reason: str
    operation: str | None = None
    source_location: str | None = None

    def format(self) -> str:
        lines = ["sparsecore config rejected:", f"  code: {self.code}"]
        if self.operation is not None:
            lines.append(f"  operation: {self.operation}")
        lines.append(f"  reason: {self.reason}")
        if self.source_location is not None:
            lines.append(f"  source: {self.source_location}")
        return "\n".join(lines)


def _node_source(node: torch.fx.Node | None) -> str | None:
    if node is None:
        return None
    stack_trace = node.meta.get("stack_trace")
    if not isinstance(stack_trace, str):
        return None
    for line in reversed(stack_trace.splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def _reject(
    code: str,
    reason: str,
    *,
    node: torch.fx.Node | None = None,
    operation: str | None = None,
) -> NoReturn:
    raise exc.InvalidConfig(
        SparseCoreRejection(
            code=code,
            reason=reason,
            operation=operation,
            source_location=_node_source(node),
        ).format()
    )


_INDIRECT_DTYPES = (torch.float32, torch.int32, torch.bfloat16)
_CAST_STORE_DTYPES = (torch.int8, torch.int32, torch.bool)
