"""SparseCore target properties and errors."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NoReturn

from ... import exc

if TYPE_CHECKING:
    import torch.fx

# TODO(tcombes): Make SparseCore target properties configurable by TPU generation.
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
    """Return the SparseCore mesh, or None when the target has no SparseCore."""
    try:
        from jax.experimental.pallas import tpu as pltpu

        sc = pltpu.get_tpu_info().sparse_core
    except Exception:
        return None
    if sc is None:
        return None
    return int(sc.num_cores), int(sc.num_subcores)


def _reject(
    code: str,
    reason: str,
    *,
    node: torch.fx.Node | None = None,
    operation: str | None = None,
) -> NoReturn:
    lines = ["sparsecore config rejected:", f"  code: {code}"]
    if operation is not None:
        lines.append(f"  operation: {operation}")
    lines.append(f"  reason: {reason}")
    location = node.meta.get("location") if node is not None else None
    if location:
        lines.append(f"  source: {location}")
    raise exc.InvalidConfig("\n".join(lines))
