# pyrefly: ignore-errors
"""Grid-wide barrier helper for the CuTe backend.

Helion's persistent multi-phase loop machinery (``program_id.py``
``_emit_phase_loops``) emits a barrier statement between device-loop phases
when a kernel uses ``hl.barrier()`` (e.g. the barrier-based split-K matmul
example).  The Triton backend lowers that to
``triton_helpers.x_grid_barrier(sem)``; this module provides the equivalent
for the CuTe backend.

The semantics mirror inductor's ``x_grid_barrier``
(``torch/_inductor/runtime/triton_helpers.py``): every CTA in the persistent
grid arrives at a shared, zero-initialized ``uint32`` semaphore via an atomic
release add, then spin-waits (atomic acquire add of 0) on the high "flip" bit
toggling.  CTA 0 contributes ``0x80000000 - (expected - 1)`` and every other
CTA contributes ``1``, so once all ``expected`` CTAs have arrived the
accumulated value rolls the flip bit, releasing every spinner.

The CuTe wrinkle is that the kernel runs many threads per CTA (Triton runs a
single program per CTA), so a naive port would issue one atomic per thread and
over-count.  We therefore:

  1. ``sync_threads()`` to fence this CTA's stores and synchronize its threads,
  2. elect a single leader thread (``thread_idx`` all zero) to perform the
     global atomics and the spin-wait,
  3. ``sync_threads()`` again so all threads in the CTA wait for the leader.

The expected CTA count is ``grid_dim()[0]`` (the persistent grid is launched as
``(_NUM_SM,)``), and the CTA-0-vs-others contribution is keyed on
``block_idx()[0] == 0``.

This is decorated with ``@cute.jit`` (not ``@dsl_user_op``) so the spin loop's
Python ``while`` over a dynamic value traces into an ``scf.WhileOp``, exactly
like CUTLASS's own ``cutlass.utils.distributed`` spin-lock helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
import cutlass.cute as cute

if TYPE_CHECKING:
    from cutlass._mlir import ir

_FLIP_BIT = 0x80000000


@cute.jit
def grid_barrier(
    sem_ptr: cute.Pointer,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Wait for all CTAs in the (1-D persistent) grid to reach this barrier.

    :param sem_ptr: pointer to a single, zero-initialized ``uint32`` semaphore
        shared by every CTA in the grid (Helion allocates it as
        ``torch.zeros((1,), dtype=torch.uint32)`` and passes the CuTe tensor's
        ``.iterator`` pointer).
    """
    # Ensure stores before this barrier are visible and synchronize the block.
    cute.arch.sync_threads()

    tidx, tidy, tidz = cute.arch.thread_idx()
    is_leader = (tidx == 0) and (tidy == 0) and (tidz == 0)

    # Only the leader thread touches the global semaphore so the per-CTA arrival
    # is counted exactly once.
    if is_leader:
        expected = cutlass.Uint32(cute.arch.grid_dim()[0])
        bidx = cute.arch.block_idx()[0]
        nb = (
            (cutlass.Uint32(_FLIP_BIT) - (expected - cutlass.Uint32(1)))
            if bidx == 0
            else cutlass.Uint32(1)
        )

        old_arrive = cute.arch.atomic_add(
            sem_ptr, nb, sem="release", scope="gpu", loc=loc, ip=ip
        )

        # Spin until the flip bit toggles relative to our arrival snapshot.
        current_arrive = old_arrive
        while ((old_arrive ^ current_arrive) & cutlass.Uint32(_FLIP_BIT)) == 0:
            current_arrive = cute.arch.atomic_add(
                sem_ptr,
                cutlass.Uint32(0),
                sem="acquire",
                scope="gpu",
                loc=loc,
                ip=ip,
            )

    # All threads wait for the leader to clear the barrier.
    cute.arch.sync_threads()
