"""Device helpers of the K-split cluster's partial exchange.

``tcgen05_split_k=2`` on a CtaGroup.TWO ``tcgen05_cluster_n=2`` kernel
(``tcgen05_cluster_split_k``) runs one output tile's two K halves on the two
CTA pairs of a 4-CTA cluster.  After the K loop every epilogue thread stores
the fp32 partials of the subtiles its partner pair owns straight from
registers into the partner CTA's receive buffer with ``st.async``, each store
completing its bytes on the partner's receive mbarrier
(``split_k_exchange.render_split_k_cluster_exchange``); with
``tcgen05_split_k_exchange="output"`` it packs them into 16-bit pairs first
(``pack_bf16x2`` / ``pack_f16x2``, stored with ``st_async_b32x4``).  The
generated kernel imports this module as ``_helion_skc``.
"""

from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ._flash_row_mma_runtime import pack_bf16x2
from ._flash_row_mma_runtime import pack_f16x2
from ._flash_runtime import set_block_rank

__all__ = [
    "pack_bf16x2",
    "pack_f16x2",
    "peer_smem_address",
    "st_async_b32x4",
    "st_async_f32x4",
]

# ``mapa``: the shared::cluster address of ``smem_ptr``'s offset in the CTA of
# cluster rank ``peer``.
peer_smem_address = set_block_rank


@dsl_user_op
def st_async_f32x4(
    remote_address: object,
    remote_mbar: object,
    v0: object,
    v1: object,
    v2: object,
    v3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """Store four fp32 values (16 bytes) into another cluster CTA's shared
    memory at ``remote_address`` and complete 16 transaction bytes on its
    mbarrier ``remote_mbar`` (both ``peer_smem_address`` results).

    The store is asynchronous: the issuing thread neither waits for it nor
    orders it with its own later accesses; the receiver's mbarrier phase
    completes once every expected byte has landed."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(remote_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(remote_mbar).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[$0], {$2, $3, $4, $5}, [$1];",
        "r,r,f,f,f,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_async_b32x4(
    remote_address: object,
    remote_mbar: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st_async_f32x4`` of four 32-bit words: eight 16-bit partials packed
    in pairs (``pack_bf16x2`` / ``pack_f16x2``), the low half first."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(remote_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(remote_mbar).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w2).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$2, $3, $4, $5}, [$1];",
        "r,r,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
