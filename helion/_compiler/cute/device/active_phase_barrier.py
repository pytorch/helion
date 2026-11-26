"""Publish one local phase's effects to the blocks that can execute the next."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .ptx import ld_acquire_gpu_u32


@cute.jit
def join_active(
    ptr: cute.Pointer, producers: cutlass.Constexpr, consumers: cutlass.Constexpr
) -> None:
    block = cute.arch.block_idx()[0]
    x, y, z = cute.arch.thread_idx()
    leader = (x == 0) and (y == 0) and (z == 0)
    if block < producers:
        cute.arch.sync_threads()
        if leader:
            # Each acq_rel RMW acquires its predecessor. The final count thus
            # publishes every contributing CTA's pre-barrier stores.
            cute.arch.atomic_add(ptr, cutlass.Uint32(1), sem="acq_rel", scope="gpu")
    if block < consumers:
        if leader:
            address = cutlass.Int64(ptr.toint())
            while ld_acquire_gpu_u32(address) != cutlass.Uint32(producers):
                pass
        cute.arch.sync_threads()
