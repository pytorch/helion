# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# The CuTe DSL intentionally leaves its compile-time value types implicit.
# ruff: noqa: ANN001, ANN202

"""Shared low-level primitives for exact KDA CuTe schedules."""

from __future__ import annotations

from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op

make_rmem_tensor = cute.make_rmem_tensor
_LLVM_STRUCT_TYPE = cast("Any", llvm).StructType


def _ldmatrix(count: str, trans: str, smem_ptr, num: int, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    outs = ", ".join(f"${i}" for i in range(num))
    struct = llvm.inline_asm(
        _LLVM_STRUCT_TYPE.get_literal([_T.IntegerType.get_signless(32)] * num),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8{count}{trans}.shared.b16 {{{outs}}}, [${num}];",
        ",".join(["=r"] * num) + ",r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(
            llvm.extractvalue(
                _T.IntegerType.get_signless(32), struct, [i], loc=loc, ip=ip
            )
        )
        for i in range(num)
    )


@dsl_user_op
def ldmatrix_x2(smem_ptr, *, loc=None, ip=None):
    """Load two 8x8 b16 matrices into registers."""
    return _ldmatrix(".x2", "", smem_ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x2_trans(smem_ptr, *, loc=None, ip=None):
    """Load and transpose two 8x8 b16 matrices into registers."""
    return _ldmatrix(".x2", ".trans", smem_ptr, 2, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4_trans(smem_ptr, *, loc=None, ip=None):
    """Load and transpose four 8x8 b16 matrices into registers."""
    return _ldmatrix(".x4", ".trans", smem_ptr, 4, loc=loc, ip=ip)


def _stmatrix(count: str, trans: str, smem_ptr, regs, *, loc=None, ip=None):
    ins = ", ".join(f"${i + 1}" for i in range(len(regs)))
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            *[cutlass.Int32(r).ir_value(loc=loc, ip=ip) for r in regs],
        ],
        f"stmatrix.sync.aligned.m8n8{count}{trans}.shared.b16 [$0], {{{ins}}};",
        ",".join(["r"] * (len(regs) + 1)),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmatrix_x2(smem_ptr, r0, r1, *, loc=None, ip=None):
    """Store two 8x8 b16 register matrices to shared memory."""
    _stmatrix(".x2", "", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def stmatrix_x2_trans(smem_ptr, r0, r1, *, loc=None, ip=None):
    """Transpose and store two 8x8 b16 register matrices."""
    _stmatrix(".x2", ".trans", smem_ptr, (r0, r1), loc=loc, ip=ip)


@dsl_user_op
def movmatrix_b16(value, *, loc=None, ip=None):
    """Transpose one 8x8 b16 register matrix."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _mma_m16n8k16(
    kind: str, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None
):
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        _LLVM_STRUCT_TYPE.get_literal([_T.F32Type.get()] * 4),
        [
            cutlass.Int32(a0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a3).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c3).ir_value(loc=loc, ip=ip),
        ],
        f"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(
            llvm.extractvalue(_T.F32Type.get(), struct, [i], loc=loc, ip=ip)
        )
        for i in range(4)
    )


@dsl_user_op
def mma_m16n8k16_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """Accumulate one m16n8k16 BF16 MMA into four FP32 registers."""
    return _mma_m16n8k16("bf16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def pack_bf16x2(lo: cutlass.Float32, hi: cutlass.Float32, *, loc=None, ip=None):
    """Round two FP32 values to BF16 and pack them into one b32 register."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def tma_load_3d(smem_ptr, desc_addr, mbar_ptr, c0, c1, c2, *, loc=None, ip=None):
    """Issue one global-to-shared 3D TMA load."""
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c2).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [$0], [$1, {$3, $4, $5}], [$2];",
        "r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_3d(desc_addr, smem_ptr, c0, c1, c2, *, loc=None, ip=None):
    """Issue one shared-to-global 3D TMA store."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip),
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c2).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [$0, {$2, $3, $4}], [$1];",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_commit_group(*, loc=None, ip=None):
    """Close the current bulk-store group."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.commit_group;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_wait_read(keep: int, *, loc=None, ip=None):
    """Wait until at most ``keep`` stores still hold their source memory."""
    llvm.inline_asm(
        None,
        [],
        f"cp.async.bulk.wait_group.read {keep};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def warp_arrive(mbar, lane):
    """Synchronize a warp, then elect one lane to arrive on an mbarrier."""
    cute.arch.sync_warp()
    if lane == 0:
        cute.arch.mbarrier_arrive(mbar)


@cute.jit
def vec_at(ptr, idx, elems):
    """Build an aligned ``elems``-long tensor at a dynamic pointer offset."""
    return cute.make_tensor(
        ptr + cute.assume(cutlass.Int32(idx), divby=elems), cute.make_layout(elems)
    )


@cute.jit
def vec8_bf16(ptr, idx):
    """Load eight contiguous BF16 values into a register fragment."""
    frag = make_rmem_tensor(8, cutlass.BFloat16)
    cute.autovec_copy(vec_at(ptr, idx, 8), frag)
    return frag


@cute.jit
def vec4_f32(ptr, idx):
    """Load four contiguous FP32 values into a register fragment."""
    frag = make_rmem_tensor(4, cutlass.Float32)
    cute.autovec_copy(vec_at(ptr, idx, 4), frag)
    return frag


@cute.jit
def store_vec8_bf16(ptr, idx, frag):
    """Store an eight-element BF16 fragment as one 16-byte access."""
    cute.autovec_copy(frag, vec_at(ptr, idx, 8))


__all__ = [
    "_mma_m16n8k16",
    "ldmatrix_x2",
    "ldmatrix_x2_trans",
    "ldmatrix_x4_trans",
    "mma_m16n8k16_bf16",
    "movmatrix_b16",
    "pack_bf16x2",
    "stmatrix_x2",
    "stmatrix_x2_trans",
    "store_vec8_bf16",
    "tma_load_3d",
    "tma_store_3d",
    "tma_store_commit_group",
    "tma_store_wait_read",
    "vec4_f32",
    "vec8_bf16",
    "vec_at",
    "warp_arrive",
]
