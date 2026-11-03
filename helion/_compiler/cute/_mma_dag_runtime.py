# Runtime support for the ``cute_mma_dag`` warp-MMA dataflow owner (see
# ``cute_mma_dag.py`` / ``cute_mma_dag_codegen.py``).
#
# These helpers are imported by the generated module and traced inline into
# the kernel body.  Every helper is one PTX instruction, so the emitted body
# is a straight line of SSA values that ptxas cannot reorder around the
# cp.async wait that guards the operand data.  The single-instruction helpers
# are ``device/ptx.py`` rows (the ragged root tile's predicated loads and the
# L2-hinted ``cp.async`` forms among them), the MMA, pack and ``nc`` loads
# the row_mma / warp_mma ones, re-exported so the generated code names one
# runtime module.  (Unlike
# _flash_runtime.py there is no ``@cute.struct`` here, so the stringified
# annotations of the future import are harmless.)
from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from ._flash_row_mma_runtime import lane_group_sum  # noqa: F401
from ._flash_row_mma_runtime import mma_bf16  # noqa: F401
from ._flash_row_mma_runtime import mma_f16  # noqa: F401
from ._flash_row_mma_runtime import octet_sum  # noqa: F401
from ._flash_row_mma_runtime import pack_bf16x2  # noqa: F401
from ._flash_row_mma_runtime import pack_f16x2  # noqa: F401
from ._warp_mma_runtime import _NC
from ._warp_mma_runtime import ldg_b32  # noqa: F401
from ._warp_mma_runtime import ldg_bf16_as_f32  # noqa: F401
from ._warp_mma_runtime import ldg_f16_as_f32  # noqa: F401
from ._warp_mma_runtime import ldg_f32  # noqa: F401
from ._warp_mma_runtime import ldg_v2_f32  # noqa: F401
from ._warp_mma_runtime import unpack_bf16x2  # noqa: F401
from .device.ptx import ASM_KW
from .device.ptx import bits_f32  # noqa: F401
from .device.ptx import cp_async_4  # noqa: F401
from .device.ptx import cp_async_16  # noqa: F401
from .device.ptx import cp_async_16_ca  # noqa: F401
from .device.ptx import cp_async_16_l2  # noqa: F401
from .device.ptx import cp_async_16_l2_zfill  # noqa: F401
from .device.ptx import cp_async_16_zfill  # noqa: F401
from .device.ptx import f32_bits  # noqa: F401
from .device.ptx import l2_policy_evict_first  # noqa: F401
from .device.ptx import l2_policy_evict_last  # noqa: F401
from .device.ptx import ldg_b32_pred  # noqa: F401
from .device.ptx import ldg_bf16_as_f32_pred  # noqa: F401
from .device.ptx import ldg_f16_as_f32_pred  # noqa: F401
from .device.ptx import ldg_f32_pred  # noqa: F401
from .device.ptx import ldg_s32_pred  # noqa: F401
from .device.ptx import ldg_s64  # noqa: F401
from .device.ptx import ldg_v2_f32_pred  # noqa: F401
from .device.ptx import ldmatrix_x2  # noqa: F401
from .device.ptx import ldmatrix_x2_trans  # noqa: F401
from .device.ptx import ldmatrix_x4  # noqa: F401
from .device.ptx import ldmatrix_x4_trans  # noqa: F401
from .device.ptx import lds_b32  # noqa: F401
from .device.ptx import lds_bf16_as_f32  # noqa: F401
from .device.ptx import lds_f16_as_f32  # noqa: F401
from .device.ptx import lds_f32  # noqa: F401
from .device.ptx import lds_s32  # noqa: F401
from .device.ptx import lds_v2_f32  # noqa: F401
from .device.ptx import lds_v4_b32  # noqa: F401
from .device.ptx import lds_v4_f32  # noqa: F401
from .device.ptx import mma_tf32  # noqa: F401
from .device.ptx import movmatrix_trans  # noqa: F401
from .device.ptx import round_bf16_f32  # noqa: F401
from .device.ptx import shfl_down_add_f32  # noqa: F401
from .device.ptx import shfl_up_add_f32  # noqa: F401
from .device.ptx import stg_b32  # noqa: F401
from .device.ptx import stg_bf16  # noqa: F401
from .device.ptx import stg_f16  # noqa: F401
from .device.ptx import stg_f32  # noqa: F401
from .device.ptx import stg_v2_f32  # noqa: F401
from .device.ptx import stg_v4_b32  # noqa: F401
from .device.ptx import sts_b32  # noqa: F401
from .device.ptx import sts_f32  # noqa: F401
from .device.ptx import sts_v2_b32  # noqa: F401
from .device.ptx import sts_v2_f32  # noqa: F401
from .device.ptx import sts_v4_b32  # noqa: F401
from .device.ptx import sts_v4_f32  # noqa: F401
from .device.ptx import unpack_f16x2  # noqa: F401
from .device.ptx import words


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
        **ASM_KW,
    )
    return cutlass.Int32(result)


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
        **ASM_KW,
    )
    return cutlass.Uint32(result)


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
        **ASM_KW,
    )
    return words(result, 4, loc=loc, ip=ip)
