# Runtime support for the register-MMA row-program flash family (``row_mma``).
#
# NOTE: deliberately NO ``from __future__ import annotations`` (see
# _flash_runtime.py): these helpers are imported by the generated module and
# traced inline into the kernel body.  Every helper is one PTX instruction (or
# a short warp-shuffle chain) with explicit register operands, so the emitted
# body is a straight line of SSA values that ptxas cannot reorder around the
# cp.async wait that guards the operand data.  The single-instruction helpers
# are ``device/ptx.py`` rows re-exported under the names the emitters render.
import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from .device.ptx import ASM_KW
from .device.ptx import bulk_reduce_add_f32  # noqa: F401
from .device.ptx import cp_async_16  # noqa: F401
from .device.ptx import cp_async_16_zfill  # noqa: F401
from .device.ptx import finite_or_zero  # noqa: F401
from .device.ptx import floats
from .device.ptx import ldmatrix_x2_trans  # noqa: F401
from .device.ptx import ldmatrix_x4  # noqa: F401
from .device.ptx import ldmatrix_x4_trans  # noqa: F401
from .device.ptx import lds_f32  # noqa: F401
from .device.ptx import lds_v2_f32  # noqa: F401
from .device.ptx import lds_v4_f32  # noqa: F401
from .device.ptx import mask_key_after  # noqa: F401
from .device.ptx import movmatrix_trans  # noqa: F401
from .device.ptx import stg_b32  # noqa: F401
from .device.ptx import stg_f32  # noqa: F401
from .device.ptx import stg_v4_b32  # noqa: F401
from .device.ptx import sts_f32  # noqa: F401
from .device.ptx import sts_v2_f32  # noqa: F401
from .device.ptx import unpack_f16x2  # noqa: F401


@dsl_user_op
def unpack_bf16x2(
    word: object, *, loc: object = None, ip: object = None
) -> tuple[object, ...]:
    """The two bf16 halves of ``word`` (low half first) widened to fp32."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),  # pyrefly: ignore[missing-attribute]
        [cutlass.Uint32(word).ir_value(loc=loc, ip=ip)],
        "{.reg .b16 lo, hi; mov.b32 {lo, hi}, $2; "
        "cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi;}",
        "=f,=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **ASM_KW,
    )
    return floats(result, 2, loc=loc, ip=ip)


def _mma_m16n8k16(
    kind: str,
    a0: object,
    a1: object,
    a2: object,
    a3: object,
    b0: object,
    b1: object,
    c0: object,
    c1: object,
    c2: object,
    c3: object,
    *,
    loc: object,
    ip: object,
) -> tuple[object, ...]:
    values = [
        *[cutlass.Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)],
        *[cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)],
    ]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),  # pyrefly: ignore[missing-attribute]
        values,
        f"mma.sync.aligned.m16n8k16.row.col.f32.{kind}.{kind}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **ASM_KW,
    )
    return floats(result, 4, loc=loc, ip=ip)


@dsl_user_op
def mma_bf16(
    a0: object,
    a1: object,
    a2: object,
    a3: object,
    b0: object,
    b1: object,
    c0: object,
    c1: object,
    c2: object,
    c3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> tuple[object, ...]:
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``."""
    return _mma_m16n8k16("bf16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def mma_f16(
    a0: object,
    a1: object,
    a2: object,
    a3: object,
    b0: object,
    b1: object,
    c0: object,
    c1: object,
    c2: object,
    c3: object,
    *,
    loc: object = None,
    ip: object = None,
) -> tuple[object, ...]:
    """``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``."""
    return _mma_m16n8k16("f16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


def _pack_x2(
    kind: str, low: object, high: object, *, loc: object, ip: object
) -> cutlass.Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Float32(high).ir_value(loc=loc, ip=ip),
            cutlass.Float32(low).ir_value(loc=loc, ip=ip),
        ],
        f"cvt.rn.{kind}x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
        **ASM_KW,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def pack_bf16x2(
    low: object, high: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``cvt.rn.bf16x2.f32``: ``low`` in the low half, ``high`` in the high half."""
    return _pack_x2("bf16", low, high, loc=loc, ip=ip)


@dsl_user_op
def pack_f16x2(
    low: object, high: object, *, loc: object = None, ip: object = None
) -> cutlass.Uint32:
    """``cvt.rn.f16x2.f32``: ``low`` in the low half, ``high`` in the high half."""
    return _pack_x2("f16", low, high, loc=loc, ip=ip)


def octet_max(value: object) -> object:
    """Max over the eight lanes that share ``lane % 4`` (xor 4, 8, 16): the
    lanes holding one query's scores in the transposed S^T fragment."""
    for offset in (4, 8, 16):
        value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset))
    return value


def octet_sum(value: object) -> object:
    """Sum over the eight lanes that share ``lane % 4``, in a fixed order."""
    for offset in (4, 8, 16):
        value = value + cute.arch.shuffle_sync_bfly(value, offset)
    return value


def lane_group_sum(value: object, width: int) -> object:
    """Sum over the ``width`` consecutive lanes (a power of two dividing 32)
    that share one output row in the combine epilogue.

    A butterfly over the offsets 1, 2, ..., width/2 in a fixed order: fp32
    addition is commutative, so every lane of the group ends with the same
    value and repeated launches are bit-identical."""
    offset = 1
    while offset < width:
        value = value + cute.arch.shuffle_sync_bfly(value, offset)
        offset *= 2
    return value


def lane_group_max(value: object, width: int) -> object:
    """NaN-propagating max over the ``width`` consecutive lanes of a row."""
    offset = 1
    while offset < width:
        value = cute.math.max(
            value, cute.arch.shuffle_sync_bfly(value, offset), propagate_nan=True
        )
        offset *= 2
    return value


def lane_group_min(value: object, width: int) -> object:
    """NaN-propagating min over the ``width`` consecutive lanes of a row."""
    offset = 1
    while offset < width:
        value = cute.math.min(
            value, cute.arch.shuffle_sync_bfly(value, offset), propagate_nan=True
        )
        offset *= 2
    return value
