"""Device helpers for mbarrier waits, ring advance and cluster (DSM) exchange.

Device helper (see the package doc).  The mbarrier idioms every vendored
schedule spelled for itself -- a ``try_wait`` parity spin, the same spin
returning the flipped phase, an elected single-thread arrival, the
division-free ring advance and the cluster-acquire and suspend-hinted
``try_wait`` spins -- are declared once here; the distributed
shared-memory primitives of the cluster reductions (``mapa`` + ``st.async``
/ ``ld.shared::cluster`` with ``mbarrier::complete_tx``) follow.  The
helpers keep the import paths generated code names (the ``cluster_helpers``
re-exports, ``_helion_mma_dag_tc.st_async_remote_f32``, ...), so rendered
kernels are unchanged and their disk-cache fingerprints follow this file
through the helpers' ``__module__``.

The asm strings, constraints and ``~{memory}`` clobbers are each site's
original text: the cluster_helpers variants take ``cute.Pointer`` operands
and map them with a separate ``mapa``; the mma_dag variants take integer
shared addresses and fuse the ``mapa`` into the asm block.
"""

# The CuTe DSL leaves its traced value types implicit, as in the other
# device modules.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import cutlass
from cutlass import Float32
from cutlass import Int32
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.experimental.primitives as prims

if TYPE_CHECKING:
    from .._mlir_compat import ir

# The inline-asm flags of the integer-address variants.  ``device.ptx`` imports
# them from here rather than the other way round: this module is in the common
# fingerprint set, ``ptx.py`` is keyed per family (the ptx family-isolation
# cases), and the schedules' import-closure tests (``_helion_imports`` in
# test_cute_gdn_recurrence.py) require every Helion module a fingerprinted
# schedule reaches to be fingerprinted too.
ASM_KW = {
    "is_align_stack": False,
    "asm_dialect": llvm.AsmDialect.AD_ATT,
}


# --------------------------------------------------------------- mbarriers


@cute.jit
def mbar_wait(mbar, parity) -> None:
    """Spin until the mbarrier phase with ``parity`` has completed."""
    while not prims.mbarrier_wait_parity(mbar, parity, prims.MBarrierWait.TRY):
        pass


@cute.jit
def mbar_wait_flip(mbar, phase):
    """Spin until the mbarrier phase ``phase`` has completed; the next phase."""
    while not prims.mbarrier_wait_parity(mbar, phase, prims.MBarrierWait.TRY):
        pass
    return phase ^ cutlass.Int32(1)


@cute.jit
def mbar_arrive_elected(mbar) -> None:
    """One arrival on ``mbar`` from the warp's elected lane."""
    if prims.elect_sync():
        prims.mbarrier_arrive(mbar)


@cute.jit
def ring_advance(stage, phase, STAGES: cutlass.Constexpr):
    """Next ring slot and its parity without a division."""
    wrapped = stage == STAGES - 1
    next_stage = cutlass.Int32(0) if wrapped else stage + 1
    next_phase = phase ^ 1 if wrapped else phase
    return next_stage, next_phase


# ------------------------------------------ distributed shared memory (DSM)


@dsl_user_op
def _set_block_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Int32:
    """Map an SMEM pointer to the address at another CTA rank in the cluster."""

    smem_ptr_i32 = cast("Any", smem_ptr).toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def load_shared_remote_f32(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Float32:
    """Read a published FP32 word; the caller owns both cluster rendezvous."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cast("Any", smem_ptr).toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank_in_cluster.ir_value(),
            ],
            "{ .reg .u32 peer_addr; mapa.shared::cluster.u32 peer_addr, $1, $2; "
            "ld.shared::cluster.f32 $0, [peer_addr]; }",
            "=f,r,r",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


@dsl_user_op
def store_shared_remote_x4(
    val0: Float32 | Int32,
    val1: Float32 | Int32,
    val2: Float32 | Int32,
    val3: Float32 | Int32,
    *,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store four scalars into another CTA's SMEM and complete the async tx."""

    remote_smem_ptr_i32 = _set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = _set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    assert isinstance(val0, (Float32, Int32)), "val must be Float32 or Int32"
    dtype = Float32 if isinstance(val0, Float32) else Int32
    suffix = {Float32: "f32", Int32: "s32"}[dtype]
    constraint = {Float32: "f", Int32: "r"}[dtype]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            dtype(val0).ir_value(loc=loc, ip=ip),
            dtype(val1).ir_value(loc=loc, ip=ip),
            dtype(val2).ir_value(loc=loc, ip=ip),
            dtype(val3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        f".reg .v4 .{suffix} abcd;\n\t"
        f"mov.{suffix} abcd.x, $2;\n\t"
        f"mov.{suffix} abcd.y, $3;\n\t"
        f"mov.{suffix} abcd.z, $4;\n\t"
        f"mov.{suffix} abcd.w, $5;\n\t"
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.{suffix} [$0], abcd, [$1];\n\t"
        "}\n",
        f"r,r,{constraint},{constraint},{constraint},{constraint}",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def store_shared_remote_f32(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store one f32 into another CTA's SMEM and complete the mbarrier tx."""

    remote_smem_ptr_i32 = _set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = _set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            Float32(val).ir_value(loc=loc, ip=ip),
            remote_mbar_ptr_i32,
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def store_shared_remote_f32x2(
    val0: Float32,
    val1: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store an (f32, f32) pair into another CTA's SMEM and complete the
    mbarrier tx.  ``smem_ptr`` must be 8-byte aligned (allocate the receive
    buffer as ``Int64`` slots)."""

    remote_smem_ptr_i32 = _set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = _set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            Float32(val0).ir_value(loc=loc, ip=ip),
            Float32(val1).ir_value(loc=loc, ip=ip),
            remote_mbar_ptr_i32,
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [$0], {$1, $2}, [$3];",
        "r,f,f,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def st_async_remote_f32(
    smem_address,
    mbar_address,
    peer_rank,
    value,
    *,
    loc=None,
    ip=None,
) -> None:
    """``st.async`` of one fp32 into the shared memory of cluster peer
    ``peer_rank`` at this CTA's shared address ``smem_address``, completing
    four transaction bytes on the peer's mbarrier at ``mbar_address``."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(mbar_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(peer_rank).ir_value(loc=loc, ip=ip),
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 remote, rbar;\n\tmapa.shared::cluster.u32 remote, $0, $2;"
        "\n\tmapa.shared::cluster.u32 rbar, $1, $2;"
        "\n\tst.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [remote], $3, [rbar];\n\t}",
        "r,r,r,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **ASM_KW,
    )


@dsl_user_op
def _try_wait_cluster(addr, parity, *, loc=None, ip=None) -> object:
    """``mbarrier.try_wait.parity.acquire.cluster``: the peers' shared
    memory writes released by their arrivals are visible afterwards."""
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32(0).ir_value(loc=loc, ip=ip).type,
            [
                cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(parity).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred P1;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [$1], $2;"
            "\n\tselp.b32 $0, 1, 0, P1;\n\t}",
            "=r,r,r,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **ASM_KW,
        )
    )


@cute.jit
def mbar_wait_cluster(mbar, parity) -> None:
    """Spin until the cluster-scope mbarrier phase ``parity`` completed."""
    addr = cutlass.Int32(mbar.toint())
    while _try_wait_cluster(addr, parity) == 0:
        pass


@dsl_user_op
def _try_wait_suspend(addr, parity, *, loc=None, ip=None) -> object:
    """``mbarrier.try_wait.parity`` with a suspend-time hint: the thread
    sleeps in the barrier unit instead of polling shared memory."""
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32(0).ir_value(loc=loc, ip=ip).type,
            [
                cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
                cutlass.Int32(parity).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred P1;\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [$1], $2, 10000000;"
            "\n\tselp.b32 $0, 1, 0, P1;\n\t}",
            "=r,r,r",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **ASM_KW,
        )
    )


@cute.jit
def mbar_wait_sleep(mbar, parity) -> None:
    """``mbar_wait`` for warps with slack (the pass warps between tiles):
    suspended waits, so their polling does not load the shared-memory
    pipe the chain's warps use."""
    addr = cutlass.Int32(mbar.toint())
    while _try_wait_suspend(addr, parity) == 0:
        pass
