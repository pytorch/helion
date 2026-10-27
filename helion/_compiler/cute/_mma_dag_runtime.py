# Runtime support for the ``cute_mma_dag`` warp-MMA dataflow owner (see
# ``cute_mma_dag.py`` / ``cute_mma_dag_codegen.py``).
#
# These helpers are imported by the generated module and traced inline into
# the kernel body.  Every helper is one PTX instruction, so the emitted body
# is a straight line of SSA values that ptxas cannot reorder around the
# cp.async wait that guards the operand data.  The MMA, ldmatrix, cp.async,
# pack and global load / store helpers are the row_mma / warp_mma ones,
# re-exported so the generated code names one runtime module.  (Unlike
# _flash_runtime.py there is no ``@cute.struct`` here, so the stringified
# annotations of the future import are harmless.)
from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from ._flash_row_mma_runtime import _ASM_KW
from ._flash_row_mma_runtime import _floats
from ._flash_row_mma_runtime import _words
from ._flash_row_mma_runtime import cp_async_16  # noqa: F401
from ._flash_row_mma_runtime import lane_group_sum  # noqa: F401
from ._flash_row_mma_runtime import ldmatrix_x4  # noqa: F401
from ._flash_row_mma_runtime import ldmatrix_x4_trans  # noqa: F401
from ._flash_row_mma_runtime import lds_f32  # noqa: F401
from ._flash_row_mma_runtime import lds_v4_f32  # noqa: F401
from ._flash_row_mma_runtime import mma_bf16  # noqa: F401
from ._flash_row_mma_runtime import mma_f16  # noqa: F401
from ._flash_row_mma_runtime import movmatrix_trans  # noqa: F401
from ._flash_row_mma_runtime import octet_sum  # noqa: F401
from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401
from ._flash_row_mma_runtime import stg_f32  # noqa: F401
from ._flash_row_mma_runtime import stg_v4_b32  # noqa: F401
from ._flash_row_mma_runtime import sts_f32  # noqa: F401
from ._warp_mma_runtime import _NC
from ._warp_mma_runtime import cp_async_4  # noqa: F401
from ._warp_mma_runtime import cp_async_16_ca  # noqa: F401
from ._warp_mma_runtime import ldg_b32  # noqa: F401
from ._warp_mma_runtime import ldg_bf16_as_f32  # noqa: F401
from ._warp_mma_runtime import ldg_f16_as_f32  # noqa: F401
from ._warp_mma_runtime import ldg_f32  # noqa: F401
from ._warp_mma_runtime import ldg_v2_f32  # noqa: F401
from ._warp_mma_runtime import ldmatrix_x2  # noqa: F401
from ._warp_mma_runtime import ldmatrix_x2_trans  # noqa: F401
from ._warp_mma_runtime import lds_bf16_as_f32  # noqa: F401
from ._warp_mma_runtime import lds_f16_as_f32  # noqa: F401
from ._warp_mma_runtime import lds_s32  # noqa: F401
from ._warp_mma_runtime import mma_tf32  # noqa: F401
from ._warp_mma_runtime import round_bf16_f32  # noqa: F401
from ._warp_mma_runtime import stg_b32  # noqa: F401
from ._warp_mma_runtime import stg_bf16  # noqa: F401
from ._warp_mma_runtime import stg_f16  # noqa: F401
from ._warp_mma_runtime import stg_v2_f32  # noqa: F401
from ._warp_mma_runtime import unpack_bf16x2  # noqa: F401
from ._warp_mma_runtime import unpack_f16x2  # noqa: F401


@dsl_user_op
def sts_b32(
    smem_address: object, word: object, *, loc: object = None, ip: object = None
) -> None:
    """``st.shared.b32``: one 4-byte word (a packed 16-bit element pair of a
    cooperatively computed operand tile)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def ldg_s32(
    gmem_address: object, *, nc: bool = False, loc: object = None, ip: object = None
) -> cutlass.Int32:
    """``ld.global.s32``: one int32 value (a side vector of window bounds)."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        f"ld.global{_NC[nc]}.s32 $0, [$1];",
        "=r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Int32(result)


@dsl_user_op
def ldg_s64(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> cutlass.Int64:
    """``ld.global.s64``: one int64 value (a loop bound read from a table of
    chunk offsets)."""
    result = llvm.inline_asm(
        T.i64(),
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        "ld.global.s64 $0, [$1];",
        "=l,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Int64(result)


# -- predicated global accesses of a ragged root tile ---------------------------
#
# A root tile larger than its axis extent (``M = 8`` rows under a 16-row MMA
# atom) reads and writes rows that do not exist.  The operand staging zero-
# fills them (``cp_async_16_zfill`` with ``src_bytes`` 0: nothing is read),
# the side loads read zero under a false predicate (the frontend's masked-
# value model: a padded element loads as 0) and the stores are guarded by
# the emitter.  ``pred`` is an Int32 (0: out of bounds).


@dsl_user_op
def cp_async_16_zfill(
    smem_address: object,
    gmem_address: object,
    src_bytes: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``cp.async.cg.shared.global [dst], [src], 16, src_bytes``: one 16-byte
    packet whose bytes past ``src_bytes`` (0 or 16 here) are zero-filled; at
    0 the source is not read."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def ldg_f32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """``ld.global.f32`` under ``pred`` (0.0 when it is 0)."""
    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.f32 $0, 0f00000000; "
        "@p ld.global.f32 $0, [$1]; }",
        "=f,l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def ldg_s32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> cutlass.Int32:
    """``ld.global.s32`` under ``pred`` (0 when it is 0)."""
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.s32 $0, 0; "
        "@p ld.global.s32 $0, [$1]; }",
        "=r,l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Int32(result)


@dsl_user_op
def ldg_b32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``ld.global.b32`` under ``pred`` (0 when it is 0: a pair of 16-bit
    zeros)."""
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0; "
        "@p ld.global.b32 $0, [$1]; }",
        "=r,l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def ldg_v2_f32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v2.f32`` under ``pred`` (two zeros when it is 0)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.s32 p, $3, 0; mov.f32 $0, 0f00000000; "
        "mov.f32 $1, 0f00000000; @p ld.global.v2.f32 {$0, $1}, [$2]; }",
        "=f,=f,l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 2, loc=loc, ip=ip)


def _ldg_b16_as_f32_pred(
    kind: str, gmem_address: object, pred: object, *, loc: object, ip: object
) -> cutlass.Float32:
    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; .reg .b16 h; setp.ne.s32 p, $2, 0; mov.b16 h, 0; "
        f"@p ld.global.b16 h, [$1]; cvt.f32.{kind} $0, h; }}",
        "=f,l,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def ldg_f16_as_f32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """One fp16 element widened to fp32 under ``pred`` (0.0 when it is 0)."""
    return _ldg_b16_as_f32_pred("f16", gmem_address, pred, loc=loc, ip=ip)


@dsl_user_op
def ldg_bf16_as_f32_pred(
    gmem_address: object, pred: object, *, loc: object = None, ip: object = None
) -> cutlass.Float32:
    """One bf16 element widened to fp32 under ``pred`` (0.0 when it is 0)."""
    return _ldg_b16_as_f32_pred("bf16", gmem_address, pred, loc=loc, ip=ip)


@dsl_user_op
def f32_bits(value: object, *, loc: object = None, ip: object = None) -> object:
    """``mov.b32``: the bit pattern of an fp32 value (an operand word the
    tensor core truncates to tf32)."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def bits_f32(word: object, *, loc: object = None, ip: object = None) -> object:
    """``mov.b32``: the fp32 value of a 32-bit word (a packet's element)."""
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def shfl_up_add_f32(
    value: object, delta: object, *, loc: object = None, ip: object = None
) -> object:
    """``value + value_of(lane - delta)``, the lanes below ``delta`` keeping
    ``value`` (``shfl.sync.up`` and an add under its in-range predicate):
    one step of an inclusive prefix sum over the lanes ``delta`` apart.
    Every lane must take part."""
    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            cutlass.Int32(delta).ir_value(loc=loc, ip=ip),
        ],
        "{.reg .f32 t; .reg .pred p; "
        "shfl.sync.up.b32 t|p, $1, $2, 0x0, 0xffffffff; "
        "@p add.f32 $0, $1, t; @!p mov.f32 $0, $1;}",
        "=f,f,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def shfl_down_add_f32(
    value: object, delta: object, *, loc: object = None, ip: object = None
) -> object:
    """``value + value_of(lane + delta)``, the lanes within ``delta`` of the
    top keeping ``value``: one step of a reversed inclusive prefix sum."""
    result = llvm.inline_asm(
        T.f32(),
        [
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            cutlass.Int32(delta).ir_value(loc=loc, ip=ip),
        ],
        "{.reg .f32 t; .reg .pred p; "
        "shfl.sync.down.b32 t|p, $1, $2, 0x1f, 0xffffffff; "
        "@p add.f32 $0, $1, t; @!p mov.f32 $0, $1;}",
        "=f,f,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Float32(result)


@dsl_user_op
def shfl_pick_b32(
    lane: object,
    flag: object,
    x: object,
    y: object,
    *,
    loc: object = None,
    ip: object = None,
) -> object:
    """``flag ? y : x`` read from lane ``lane`` (two ``shfl.sync.idx`` and a
    ``selp``): one fp32 operand word picked from a producer fragment's
    register pair in another lane of the warp.  Every lane must take part."""
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Int32(lane).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(x).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(y).ir_value(loc=loc, ip=ip),
            cutlass.Int32(flag).ir_value(loc=loc, ip=ip),
        ],
        "{.reg .b32 t0, t1; .reg .pred p; "
        "shfl.sync.idx.b32 t0, $2, $1, 0x1f, 0xffffffff; "
        "shfl.sync.idx.b32 t1, $3, $1, 0x1f, 0xffffffff; "
        "setp.ne.s32 p, $4, 0; selp.b32 $0, t1, t0, p;}",
        "=r,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def lds_b32(smem_address: object, *, loc: object = None, ip: object = None) -> object:
    """``ld.shared.b32``: one 4-byte word (a packed 16-bit element pair of a
    staged tile read at the lane's fragment coordinates)."""
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.b32 $0, [$1];",
        "=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def lds_v2_f32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.shared.v2.f32``: two consecutive fp32 values (8-byte aligned; an
    fp32 tile pair read at the lane's fragment coordinates)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _floats(result, 2, loc=loc, ip=ip)


@dsl_user_op
def lds_v4_b32(
    smem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.shared.v4.b32``: one 16-byte packet (eight 16-bit elements)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def ldg_v4_b32(
    gmem_address: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """``ld.global.v4.b32``: one 16-byte packet (eight 16-bit or four
    32-bit elements) at a 16-byte aligned address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),  # pyrefly: ignore[missing-attribute]
        [cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return _words(result, 4, loc=loc, ip=ip)


@dsl_user_op
def sts_v4_b32(
    smem_address: object,
    w0: object,
    w1: object,
    w2: object,
    w3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v4.b32``: one 16-byte packet."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            *[cutlass.Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


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
    """``st.shared.v4.f32``: one 16-byte packet of four fp32 values (a
    widened tf32 operand row chunk)."""
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


@dsl_user_op
def sts_v2_f32(
    smem_address: object,
    x0: object,
    x1: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v2.f32``: two consecutive fp32 values (8-byte aligned)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Float32(x0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(x1).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def sts_v2_b32(
    smem_address: object,
    w0: object,
    w1: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``st.shared.v2.b32``: two packed 16-bit pairs (8-byte aligned)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def l2_policy_evict_last(*, loc: object = None, ip: object = None) -> cutlass.Int64:
    """``createpolicy.fractional.L2::evict_last``: the cache-policy word of
    the hinted operand packets (``cp_async_16_l2``), created once per
    kernel."""
    result = llvm.inline_asm(
        T.i64(),
        [],
        "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
        "=l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Int64(result)


@dsl_user_op
def l2_policy_evict_first(*, loc: object = None, ip: object = None) -> cutlass.Int64:
    """``createpolicy.fractional.L2::evict_first``: the cache-policy word of
    streamed operand packets read once."""
    result = llvm.inline_asm(
        T.i64(),
        [],
        "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
        "=l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
    return cutlass.Int64(result)


# The hinted packet forms below are the ones ``helion.runtime.cute.sass_check``
# verifies after assembly: the CuTe DSL's assembler (cutlass-dsl 4.7.0) has
# encoded a hinted LDGSTS whose shared address it split into the
# ``[R + UR + imm]`` form with the descriptor and address-offset fields
# colliding (an illegal instruction at launch), so a kernel that emits them
# declares the check and the launcher refuses the kernel when its SASS shows
# that form (see ``emit_issue_tile``).


@dsl_user_op
def cp_async_16_l2(
    smem_address: object,
    gmem_address: object,
    policy: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``cp.async.cg.shared.global.L2::cache_hint`` of one 16-byte packet
    under the L2 cache policy ``policy`` (``l2_policy_evict_last`` /
    ``l2_policy_evict_first``)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int64(policy).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
        "r,l,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )


@dsl_user_op
def cp_async_16_l2_zfill(
    smem_address: object,
    gmem_address: object,
    src_bytes: object,
    policy: object,
    *,
    loc: object = None,
    ip: object = None,
) -> None:
    """``cp_async_16_zfill`` under the L2 cache policy ``policy``: the bytes
    past ``src_bytes`` (0 or 16) are zero-filled; at 0 nothing is read."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(smem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_address).ir_value(loc=loc, ip=ip),
            cutlass.Int32(src_bytes).ir_value(loc=loc, ip=ip),
            cutlass.Int64(policy).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2, $3;",
        "r,l,r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
        **_ASM_KW,
    )
