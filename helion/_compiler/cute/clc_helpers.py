"""CLC (clusterlaunchcontrol) helper for the tcgen05 CLC-persistent path.

G2-H (cute_plan.md): wrap ``nvvm.clusterlaunchcontrol_try_cancel``
in a ``@dsl_user_op`` so the generated kernel can issue the CLC query
from the dedicated scheduler warp. The wrapper mirrors Quack's
``issue_clc_query_nomulticast`` in ``quack/quack/utils.py``; the only
difference is the import path for the helper symbol so Helion-generated
code does not depend on ``quack`` being installed.

The instruction itself requests atomically cancelling the launch of a
cluster that has not started running yet. It asynchronously writes an
opaque response to shared memory indicating success or failure; on
success the response contains the ctaid of the first CTA of the
cancelled cluster, which the scheduler-warp body unpacks via
``cute.arch.clc_response`` to drive the next persistent-loop
iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from cutlass._mlir import ir
    import cutlass.cute as cute


@dsl_user_op
def issue_clc_query_nomulticast(
    mbar_ptr: cute.Pointer,
    clc_response_ptr: cute.Pointer,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Issue ``nvvm.clusterlaunchcontrol_try_cancel`` (no multicast).

    Helion's CLC scheduler-warp body uses the leader-CTA-only Quack
    pattern for cluster_m>1: only the cluster leader CTA's scheduler
    warp issues the query and then broadcasts the result to peer
    CTAs via ``_cute_store_shared_remote_x4``. A single CLC query
    per cluster keeps the cancellation atomicity scoped correctly,
    so the nomulticast variant is the right choice; a hypothetical
    multicast variant would land here when an alternative topology
    needs cluster-wide query delivery without the SMEM broadcast.

    :param mbar_ptr: A pointer to the mbarrier address in SMEM. The
        instruction signals this barrier with a 16-byte transaction
        when the response is fully written.
    :param clc_response_ptr: A pointer to the CLC response buffer in
        SMEM. The hardware writes 16 bytes (4 × Int32) decoding to
        ``(bidx, bidy, bidz, valid)`` via ``cute.arch.clc_response``.
    """
    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    clc_response_llvm_ptr = clc_response_ptr.llvm_ptr
    nvvm.clusterlaunchcontrol_try_cancel(
        clc_response_llvm_ptr,
        mbar_llvm_ptr,
        loc=loc,
        ip=ip,
    )
