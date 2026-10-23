# ruff: noqa: ANN001
# Runtime support for the ``cute_mma_dag`` tcgen05 compute mode (see
# ``cute_mma_dag_tcgen05.py``).  The generated kernel imports this module as
# ``_helion_mma_dag_tc`` and reaches the CUTLASS primitives (``prims``), the
# TMA descriptor type (``cuda``) and the few helpers below through it; the
# host wrapper builds the tensor maps with :func:`make_tensor_map`.
from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from ._flash_row_mma_runtime import _ASM_KW
from ._flash_row_mma_runtime import lds_v4_f32  # noqa: F401
from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401

__all__ = [
    "atom_add_release_gpu_u32",
    "cuda",
    "fence_acq_rel_gpu",
    "ld_acquire_gpu_u32",
    "ldcg_f32",
    "lds_cluster_f32",
    "make_tensor_map",
    "mbar_arrive_cluster",
    "mbar_wait",
    "mbar_wait_cluster",
    "mbar_wait_sleep",
    "prims",
    "spin_until_reached_acquire_gpu_u32",
    "st_async_remote_f32",
    "sts_v4_f32",
    "tma_swizzle",
]

TMEM_SPACE = 6


@cute.jit
def mbar_wait(mbar, parity) -> None:
    """Spin until the mbarrier phase with ``parity`` has completed."""
    while not prims.mbarrier_wait_parity(mbar, parity, prims.MBarrierWait.TRY):
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
            **_ASM_KW,
        )
    )


@dsl_user_op
def mbar_arrive_cluster(mbar_address, peer_rank, *, loc=None, ip=None) -> None:
    """One ``release.cluster`` arrival on the mbarrier at this CTA's shared
    address ``mbar_address`` in cluster peer ``peer_rank`` (the same
    offset in the peer's shared memory)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(mbar_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 remote;\n\tmapa.shared::cluster.u32 remote, $0, $1;"
        "\n\tmbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote];\n\t}",
        "r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def st_async_remote_f32(
    smem_address, mbar_address, peer_rank, value, *, loc=None, ip=None
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
        **_ASM_KW,
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
            **_ASM_KW,
        )
    )


@cute.jit
def mbar_wait_cluster(mbar, parity) -> None:
    """Spin until the cluster-scope mbarrier phase ``parity`` completed."""
    addr = cutlass.Int32(mbar.toint())
    while _try_wait_cluster(addr, parity) == 0:
        pass


@dsl_user_op
def lds_cluster_f32(smem_address, peer_rank, *, loc=None, ip=None) -> object:
    """The fp32 word at this CTA's shared address ``smem_address`` in
    cluster peer ``peer_rank`` (distributed shared memory)."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32(0.0).ir_value(loc=loc, ip=ip).type,
            [
                cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
                cutlass.Int32(peer_rank).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 remote;\n\tmapa.shared::cluster.u32 remote, $1, $2;"
            "\n\tld.shared::cluster.f32 $0, [remote];\n\t}",
            "=f,r,r,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **_ASM_KW,
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


@dsl_user_op
def sts_v4_f32(
    smem_address: object,
    x0: object,
    x1: object,
    x2: object,
    x3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v4.f32``: one 16-byte packet of four fp32 values."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3)],
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


def tma_swizzle(name: str) -> cuda.TensorMapSwizzle:
    return {
        "none": cuda.TensorMapSwizzle.none,
        "s32b": cuda.TensorMapSwizzle.s32b,
        "s64b": cuda.TensorMapSwizzle.s64b,
        "s128b": cuda.TensorMapSwizzle.s128b,
        "s128b_atom_32b": cuda.TensorMapSwizzle.s128b_atom_32b,
    }[name]


@cute.jit
def make_tensor_map(
    tensor: cute.Tensor,
    dims: cutlass.Constexpr,
    box: cutlass.Constexpr,
    swizzle: cutlass.Constexpr,
) -> cuda.TensorMap:
    """A tiled TMA descriptor over the tensor dims ``dims`` (innermost first)
    with the box ``box`` (same order) and the named swizzle.

    Runs in the host wrapper: ``tensor`` is the wrapper's view of the kernel
    argument, ``dims`` the positions of the tensor's modes in TMA order.
    """
    shape = tuple(tensor.shape[d] for d in dims)  # pyrefly: ignore[bad-index]
    # The innermost dim is the operand's contiguous dim (proven at compile
    # time); the descriptor needs that unit stride as a static value.
    stride = (1, *(tensor.stride[d] for d in dims[1:]))  # pyrefly: ignore[bad-index]
    view = cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))
    return cuda.create_tensor_map_tiled_from_view(
        view,
        box_dims=tuple(box),
        stride_order=tuple(range(len(dims))),
        swizzle=tma_swizzle(swizzle),
    )


# -- the fixup split's global exchange (``emit_fixup_allreduce``) ---------------


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    """``fence.acq_rel.gpu``: this thread's (and, after a CTA barrier, the
    CTA's) prior global writes are ordered before its later releases."""
    llvm.inline_asm(
        None,
        [],
        "fence.acq_rel.gpu;",
        "~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def atom_add_release_gpu_u32(gmem_address, value, *, loc=None, ip=None) -> object:
    """``atom.add.release.gpu`` on the u32 at ``gmem_address``: the value
    before the add (this CTA's arrival ticket), unsigned."""
    return cutlass.Uint32(
        llvm.inline_asm(
            cutlass.Uint32(0).ir_value(loc=loc, ip=ip).type,
            [
                cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.add.release.gpu.global.u32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


@dsl_user_op
def ld_acquire_gpu_u32(gmem_address, *, loc=None, ip=None) -> object:
    """``ld.acquire.gpu`` of the u32 at ``gmem_address`` (the peers' writes
    released before their arrivals are visible afterwards), unsigned."""
    return cutlass.Uint32(
        llvm.inline_asm(
            cutlass.Uint32(0).ir_value(loc=loc, ip=ip).type,
            [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


@cute.jit
def spin_until_reached_acquire_gpu_u32(gmem_address, target) -> None:
    """Spin until the u32 counter at ``gmem_address`` reached ``target`` in
    modular arithmetic: the counter only grows and wraps, so "reached" is
    ``counter - target`` (mod 2^32) below 2^31."""
    while (ld_acquire_gpu_u32(gmem_address) - target) >= cutlass.Uint32(2147483648):
        pass


@dsl_user_op
def ldcg_f32(gmem_address, *, loc=None, ip=None) -> object:
    """``ld.global.cg.f32``: the fp32 at ``gmem_address`` from L2 (another
    CTA's partial; never a stale L1 line)."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32(0.0).ir_value(loc=loc, ip=ip).type,
            [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )
