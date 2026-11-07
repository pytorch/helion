"""General-purpose PTX primitive emitter: one table row per inline-asm wrapper.

Device helper.  ``ptx_op`` builds the ``@dsl_user_op`` wrapper of one PTX
instruction from a table row -- the asm text, the DSL coercion of each
operand (which fixes the constraint letter) and the result kinds -- so the
primitives several runtime modules used to carry as byte-identical copies
are declared once.  Each runtime module re-exports the rows it renders under
the old name, so generated kernels (``_helion_warp_mma.stg_b32``,
``_helion_mma_dag.sts_v4_f32``, ``_helion_flash_rowmma.lds_v2_f32``, ...)
are unchanged, and a kernel's disk-cache fingerprint follows this file
through the rows' ``__module__`` (``helion.runtime.cute.source_dependencies``).

The traced MLIR of a row is a function of its asm string, constraint string,
side-effect flag, operand coercions and result wrappers only, so a row and the
hand-written wrapper it replaces lower identically.  Rows are declared as
plain ``name = ptx_op(...)`` assignments (no loops or ``globals()``) so the
standalone exporter can copy them by name.

Not every wrapper is a row: the ones that bake a Python argument into the asm
text (``ldg_f32(nc=...)``, ``tma_store_wait_read(keep)``), pass their operands
in another order than the asm (``pack_bf16x2``, ``shfl_pick_b32``) or map a
pointer through ``mapa`` first (``device.sync``) stay hand-written.
"""

from __future__ import annotations

# Evaluable at runtime: the standalone exporter copies these definitions into
# a module without the future import.
from collections.abc import Callable  # noqa: TC003
from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

from .sync import ASM_KW

# The inline-asm flags, defined once in ``device.sync`` (the common
# fingerprint set); the rows and the runtime modules read them from here.


class _PointerAddress:
    """Operand coercion of a ``cute.Pointer``: the asm operand is the
    pointer's ``toint`` address (32-bit in the shared window, 64-bit
    generic), traced the way the hand-written wrappers traced it."""

    def __init__(self, pointer: object) -> None:
        self.pointer = pointer

    def ir_value(self, *, loc: object = None, ip: object = None) -> object:
        address = cast("Any", self.pointer).toint(loc=loc, ip=ip)
        return address.ir_value(loc=loc, ip=ip)


class SharedPointer(_PointerAddress):
    """A ``cute.Pointer`` into shared memory (an ``r`` operand)."""


class GlobalPointer(_PointerAddress):
    """A ``cute.Pointer`` into global memory (an ``l`` operand)."""


# DSL coercion of an operand -> its inline-asm constraint letter.
_CONSTRAINT: dict[type, str] = {
    cutlass.Int32: "r",
    cutlass.Uint32: "r",
    cutlass.Int64: "l",
    cutlass.Float32: "f",
    SharedPointer: "r",
    GlobalPointer: "l",
}

# Result wrapper -> MLIR element type.
_RESULT: dict[type, Callable[[], object]] = {
    cutlass.Int32: T.i32,
    cutlass.Uint32: T.i32,
    cutlass.Int64: T.i64,
    cutlass.Float32: T.f32,
}


def words(result: object, count: int, *, loc: object, ip: object) -> tuple[object, ...]:
    """The ``count`` i32 members of an inline-asm result struct as Uint32."""
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(count)
    )


def floats(
    result: object, count: int, *, loc: object, ip: object
) -> tuple[object, ...]:
    """The ``count`` f32 members of an inline-asm result struct as Float32."""
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(count)
    )


def ptx_op(
    name: str,
    asm: str,
    ins: tuple[type, ...],
    outs: tuple[type, ...] = (),
    *,
    side_effects: bool,
    memory: bool,
    doc: str,
) -> Callable[..., object]:
    """Build the ``@dsl_user_op`` wrapper of one PTX instruction.

    ``ins`` are the DSL types the positional operands are coerced to (in asm
    operand order after the results); ``outs`` the result wrappers: none (the
    row returns None), one (the row returns that scalar) or several (returned
    as a tuple, unpacked from the asm's result struct); ``memory`` appends the
    ``~{memory}`` clobber.  A row takes its operands positionally:
    ``inspect.signature`` of a row is ``(*operands, loc=None, ip=None)``, not
    the parameter names of the hand-written wrapper it replaced.
    """
    constraints = [f"={_CONSTRAINT[kind]}" for kind in outs]
    constraints.extend(_CONSTRAINT[kind] for kind in ins)
    if memory:
        constraints.append("~{memory}")
    constraint_string = ",".join(constraints)
    element_types = [_RESULT[kind] for kind in outs]

    def op(*operands: object, loc: object = None, ip: object = None) -> object:
        values = [
            kind(operand).ir_value(loc=loc, ip=ip)
            for kind, operand in zip(ins, operands, strict=True)
        ]
        if not outs:
            result_type = None
        elif len(outs) == 1:
            result_type = element_types[0]()
        else:
            result_type = llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
                [element_type() for element_type in element_types]
            )
        result = llvm.inline_asm(
            result_type,
            values,
            asm,
            constraint_string,
            has_side_effects=side_effects,
            loc=loc,
            ip=ip,
            **ASM_KW,
        )
        if not outs:
            return None
        if len(outs) == 1:
            return outs[0](result)
        return tuple(
            kind(llvm.extractvalue(element_type(), result, [i], loc=loc, ip=ip))
            for i, (kind, element_type) in enumerate(
                zip(outs, element_types, strict=True)
            )
        )

    op.__name__ = op.__qualname__ = name
    op.__doc__ = doc
    return dsl_user_op(op)


# ---- shared memory ---------------------------------------------------------

lds_f32 = ptx_op(
    "lds_f32",
    "ld.shared.f32 $0, [$1];",
    (cutlass.Int32,),
    (cutlass.Float32,),
    side_effects=True,
    memory=True,
    doc="``ld.shared.f32``: one fp32 value.",
)

lds_s32 = ptx_op(
    "lds_s32",
    "ld.shared.s32 $0, [$1];",
    (cutlass.Int32,),
    (cutlass.Int32,),
    side_effects=True,
    memory=True,
    doc="``ld.shared.s32``: one int32 value.",
)

lds_b32 = ptx_op(
    "lds_b32",
    "ld.shared.b32 $0, [$1];",
    (cutlass.Int32,),
    (cutlass.Uint32,),
    side_effects=True,
    memory=True,
    doc=(
        "``ld.shared.b32``: one 4-byte word (a packed 16-bit element pair of "
        "a staged tile read at the lane's fragment coordinates)."
    ),
)

lds_v2_f32 = ptx_op(
    "lds_v2_f32",
    "ld.shared.v2.f32 {$0, $1}, [$2];",
    (cutlass.Int32,),
    (cutlass.Float32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``ld.shared.v2.f32``: two consecutive fp32 words (8-byte aligned).",
)

lds_v4_f32 = ptx_op(
    "lds_v4_f32",
    "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
    (cutlass.Int32,),
    (cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``ld.shared.v4.f32``: one 16-byte packet of four fp32 values.",
)

lds_v4_b32 = ptx_op(
    "lds_v4_b32",
    "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
    (cutlass.Int32,),
    (cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc=(
        "``ld.shared.v4.b32``: one 16-byte packet (eight 16-bit elements, an "
        "MMA operand packet half) as four words."
    ),
)

lds_f16_as_f32 = ptx_op(
    "lds_f16_as_f32",
    "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.f16 $0, h; }",
    (cutlass.Int32,),
    (cutlass.Float32,),
    side_effects=True,
    memory=True,
    doc="``ld.shared.b16`` of an fp16 element, widened to fp32 (exact).",
)

lds_bf16_as_f32 = ptx_op(
    "lds_bf16_as_f32",
    "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
    (cutlass.Int32,),
    (cutlass.Float32,),
    side_effects=True,
    memory=True,
    doc="``ld.shared.b16`` of a bf16 element, widened to fp32 (exact).",
)

sts_f32 = ptx_op(
    "sts_f32",
    "st.shared.f32 [$0], $1;",
    (cutlass.Int32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``st.shared.f32`` of one fp32 value.",
)

sts_b32 = ptx_op(
    "sts_b32",
    "st.shared.b32 [$0], $1;",
    (cutlass.Int32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc=(
        "``st.shared.b32``: one 4-byte word (a packed 16-bit element pair of "
        "a cooperatively computed operand tile)."
    ),
)

sts_v2_f32 = ptx_op(
    "sts_v2_f32",
    "st.shared.v2.f32 [$0], {$1, $2};",
    (cutlass.Int32, cutlass.Float32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``st.shared.v2.f32`` of two consecutive fp32 words (8-byte aligned).",
)

sts_v4_f32 = ptx_op(
    "sts_v4_f32",
    "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
    (cutlass.Int32, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``st.shared.v4.f32``: one 16-byte packet of four fp32 values.",
)

sts_v2_b32 = ptx_op(
    "sts_v2_b32",
    "st.shared.v2.b32 [$0], {$1, $2};",
    (cutlass.Int32, cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc="``st.shared.v2.b32``: two 4-byte words (8-byte aligned).",
)

sts_v4_b32 = ptx_op(
    "sts_v4_b32",
    "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
    (cutlass.Int32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc="``st.shared.v4.b32``: four 32-bit words (one 16-byte packet).",
)

ldmatrix_x2 = ptx_op(
    "ldmatrix_x2",
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
    (cutlass.Int32,),
    (cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc=(
        "``ldmatrix ... .x2``: two 8x8 b16 tiles whose row addresses lanes "
        "0-15 supply (the other lanes' addresses are ignored)."
    ),
)

ldmatrix_x2_trans = ptx_op(
    "ldmatrix_x2_trans",
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
    (cutlass.Int32,),
    (cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc=(
        "``ldmatrix ... .x2.trans``: two 8x8 b16 tiles whose row addresses "
        "lanes 0-15 supply (the other lanes' addresses are ignored); lane "
        "(g, c) receives rows 2c, 2c+1 of column g of each tile."
    ),
)

ldmatrix_x4 = ptx_op(
    "ldmatrix_x4",
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
    (cutlass.Int32,),
    (cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc=(
        "``ldmatrix ... .x4``: four 8x8 b16 tiles, one row address per lane "
        "(lanes 8t..8t+7 address tile t); lane (g, c) receives columns 2c, "
        "2c+1 of row g of each tile."
    ),
)

ldmatrix_x4_trans = ptx_op(
    "ldmatrix_x4_trans",
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
    (cutlass.Int32,),
    (cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc=(
        "``ldmatrix ... .x4.trans``: four 8x8 b16 tiles transposed on the "
        "way into the registers; lane (g, c) receives rows 2c, 2c+1 of "
        "column g of each tile."
    ),
)


def _movmatrix_row(name: str, kind: type, doc: str) -> Callable[..., object]:
    """One ``movmatrix.sync.aligned.m8n8.trans.b16`` row per register word type."""
    return ptx_op(
        name,
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        (kind,),
        (kind,),
        side_effects=False,
        memory=False,
        doc=doc,
    )


movmatrix_trans = _movmatrix_row(
    "movmatrix_trans",
    cutlass.Uint32,
    "``movmatrix.sync.aligned.m8n8.trans.b16``: transpose one 8x8 16-bit "
    "tile held in the mma C/A fragment layout (lane (g, c) holds row g, "
    "columns 2c, 2c+1), yielding the mma B fragment of the transposed tile.",
)

movmatrix_b16 = _movmatrix_row(
    "movmatrix_b16",
    cutlass.Int32,
    "Transpose one 8x8 b16 register matrix (the recurrences' signed word).",
)

# ---- global memory ---------------------------------------------------------

stg_b32 = ptx_op(
    "stg_b32",
    "st.global.b32 [$0], $1;",
    (cutlass.Int64, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc="``st.global.b32``: one 4-byte word (a packed 16-bit pair).",
)

stg_f32 = ptx_op(
    "stg_f32",
    "st.global.f32 [$0], $1;",
    (cutlass.Int64, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``st.global.f32`` of one fp32 value.",
)

stg_f16 = ptx_op(
    "stg_f16",
    "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; st.global.b16 [$0], h; }",
    (cutlass.Int64, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="One fp32 value rounded (RN) to fp16 and stored (2-byte aligned).",
)

stg_bf16 = ptx_op(
    "stg_bf16",
    "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }",
    (cutlass.Int64, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="One fp32 value rounded (RN) to bf16 and stored (2-byte aligned).",
)

stg_v2_f32 = ptx_op(
    "stg_v2_f32",
    "st.global.v2.f32 [$0], {$1, $2};",
    (cutlass.Int64, cutlass.Float32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``st.global.v2.f32``: two consecutive fp32 values (8-byte aligned).",
)

stg_v4_f32 = ptx_op(
    "stg_v4_f32",
    "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
    (cutlass.Int64, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``st.global.v4.f32``: four consecutive fp32 values (16-byte aligned).",
)

stg_v2_b32 = ptx_op(
    "stg_v2_b32",
    "st.global.v2.b32 [$0], {$1, $2};",
    (cutlass.Int64, cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc="``st.global.v2.b32``: two 4-byte words (8-byte aligned; four 16-bit values).",
)

stg_v4_b32 = ptx_op(
    "stg_v4_b32",
    "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
    (cutlass.Int64, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32),
    side_effects=True,
    memory=True,
    doc="``st.global.v4.b32``: one 16-byte packet of four words (eight 16-bit values).",
)

ldg_s64 = ptx_op(
    "ldg_s64",
    "ld.global.s64 $0, [$1];",
    (cutlass.Int64,),
    (cutlass.Int64,),
    side_effects=True,
    memory=True,
    doc=(
        "``ld.global.s64``: one int64 value (a loop bound read from a table "
        "of chunk offsets)."
    ),
)

ldcg_f32 = ptx_op(
    "ldcg_f32",
    "ld.global.cg.f32 $0, [$1];",
    (cutlass.Int64,),
    (cutlass.Float32,),
    side_effects=True,
    memory=True,
    doc=(
        "``ld.global.cg.f32``: the fp32 at the address from L2 (another "
        "CTA's partial; never a stale L1 line)."
    ),
)

# The predicated loads of a ragged root tile: a false ``pred`` (an Int32, 0
# out of bounds) reads nothing and yields zero (the frontend's masked-value
# model: a padded element loads as 0).

ldg_f32_pred = ptx_op(
    "ldg_f32_pred",
    "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.f32 $0, 0f00000000; "
    "@p ld.global.f32 $0, [$1]; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Float32,),
    side_effects=True,
    memory=True,
    doc="``ld.global.f32`` under ``pred`` (0.0 when it is 0).",
)

ldg_s32_pred = ptx_op(
    "ldg_s32_pred",
    "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.s32 $0, 0; @p ld.global.s32 $0, [$1]; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Int32,),
    side_effects=True,
    memory=True,
    doc="``ld.global.s32`` under ``pred`` (0 when it is 0).",
)

ldg_s8_pred = ptx_op(
    "ldg_s8_pred",
    "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.s32 $0, 0; @p ld.global.s8 $0, [$1]; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Int32,),
    side_effects=True,
    memory=True,
    doc="One signed byte widened to int32 under pred (zero when false).",
)

ldg_u8_pred = ptx_op(
    "ldg_u8_pred",
    "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.u32 $0, 0; @p ld.global.u8 $0, [$1]; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Uint32,),
    side_effects=True,
    memory=True,
    doc="One unsigned byte widened to uint32 under pred (zero when false).",
)

ldg_b32_pred = ptx_op(
    "ldg_b32_pred",
    "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0; @p ld.global.b32 $0, [$1]; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Uint32,),
    side_effects=True,
    memory=True,
    doc="``ld.global.b32`` under ``pred`` (0 when it is 0: a pair of 16-bit zeros).",
)

ldg_v2_f32_pred = ptx_op(
    "ldg_v2_f32_pred",
    "{ .reg .pred p; setp.ne.s32 p, $3, 0; mov.f32 $0, 0f00000000; "
    "mov.f32 $1, 0f00000000; @p ld.global.v2.f32 {$0, $1}, [$2]; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Float32, cutlass.Float32),
    side_effects=True,
    memory=True,
    doc="``ld.global.v2.f32`` under ``pred`` (two zeros when it is 0).",
)

ldg_f16_as_f32_pred = ptx_op(
    "ldg_f16_as_f32_pred",
    "{ .reg .pred p; .reg .b16 h; setp.ne.s32 p, $2, 0; mov.b16 h, 0; "
    "@p ld.global.b16 h, [$1]; cvt.f32.f16 $0, h; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Float32,),
    side_effects=True,
    memory=True,
    doc="One fp16 element widened to fp32 under ``pred`` (0.0 when it is 0).",
)

ldg_bf16_as_f32_pred = ptx_op(
    "ldg_bf16_as_f32_pred",
    "{ .reg .pred p; .reg .b16 h; setp.ne.s32 p, $2, 0; mov.b16 h, 0; "
    "@p ld.global.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
    (cutlass.Int64, cutlass.Int32),
    (cutlass.Float32,),
    side_effects=True,
    memory=True,
    doc="One bf16 element widened to fp32 under ``pred`` (0.0 when it is 0).",
)

prefetch_global_l2 = ptx_op(
    "prefetch_global_l2",
    "prefetch.global.L2 [$0];",
    (GlobalPointer,),
    side_effects=True,
    memory=False,
    doc="``prefetch.global.L2`` of the line holding the global pointer.",
)

store_u32x4_if_valid = ptx_op(
    "store_u32x4_if_valid",
    "{ .reg .pred valid; setp.lt.u64 valid, $5, $6; "
    "@valid st.global.L1::no_allocate.v4.u32 [$0], {$1, $2, $3, $4}; }",
    (
        GlobalPointer,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Int64,
        cutlass.Int64,
    ),
    side_effects=True,
    memory=False,
    doc=(
        "Store four packed words at the global pointer only when their slot "
        "``state_index`` is below ``state_size``."
    ),
)

# ---- cp.async --------------------------------------------------------------

cp_async_4 = ptx_op(
    "cp_async_4",
    "cp.async.ca.shared.global [$0], [$1], 4;",
    (cutlass.Int32, cutlass.Int64),
    side_effects=True,
    memory=True,
    doc=(
        "``cp.async.ca.shared.global`` of one 4-byte word (a side-load "
        "element or a pair of 16-bit ones) into a per-thread shared slot."
    ),
)

cp_async_16_ca = ptx_op(
    "cp_async_16_ca",
    "cp.async.ca.shared.global [$0], [$1], 16;",
    (cutlass.Int32, cutlass.Int64),
    side_effects=True,
    memory=True,
    doc="``cp.async.ca.shared.global`` of one 16-byte packet (L1-allocating).",
)

cp_async_16 = ptx_op(
    "cp_async_16",
    "cp.async.cg.shared.global [$0], [$1], 16;",
    (cutlass.Int32, cutlass.Int64),
    side_effects=True,
    memory=True,
    doc="``cp.async.cg.shared.global`` of one 16-byte packet (L2 only).",
)

cp_async_16_zfill = ptx_op(
    "cp_async_16_zfill",
    "cp.async.cg.shared.global [$0], [$1], 16, $2;",
    (cutlass.Int32, cutlass.Int64, cutlass.Int32),
    side_effects=True,
    memory=True,
    doc=(
        "``cp.async.cg.shared.global`` of one 16-byte packet reading "
        "``src_bytes`` (16 or 0) bytes from global memory and zero-filling "
        "the rest: a masked row costs no global read."
    ),
)

# The hinted packet forms are the ones ``helion.runtime.cute.sass_check``
# verifies after assembly: the CuTe DSL's assembler (cutlass-dsl 4.7.0) has
# encoded a hinted LDGSTS whose shared address it split into the
# ``[R + UR + imm]`` form with the descriptor and address-offset fields
# colliding (an illegal instruction at launch), so a kernel that emits them
# declares the check and the launcher refuses the kernel when its SASS shows
# that form (see ``cute_mma_dag_codegen.emit_issue_tile``).

cp_async_16_l2 = ptx_op(
    "cp_async_16_l2",
    "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
    (cutlass.Int32, cutlass.Int64, cutlass.Int64),
    side_effects=True,
    memory=True,
    doc=(
        "``cp.async.cg.shared.global.L2::cache_hint`` of one 16-byte packet "
        "under the L2 cache policy word (``l2_policy_evict_last`` / "
        "``l2_policy_evict_first``)."
    ),
)

cp_async_16_l2_zfill = ptx_op(
    "cp_async_16_l2_zfill",
    "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2, $3;",
    (cutlass.Int32, cutlass.Int64, cutlass.Int32, cutlass.Int64),
    side_effects=True,
    memory=True,
    doc=(
        "``cp_async_16_zfill`` under an L2 cache policy word: the bytes past "
        "``src_bytes`` (0 or 16) are zero-filled; at 0 nothing is read."
    ),
)

l2_policy_evict_last = ptx_op(
    "l2_policy_evict_last",
    "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
    (),
    (cutlass.Int64,),
    side_effects=True,
    memory=False,
    doc=(
        "``createpolicy.fractional.L2::evict_last``: the cache-policy word of "
        "the hinted operand packets (``cp_async_16_l2``), created once per "
        "kernel."
    ),
)

l2_policy_evict_first = ptx_op(
    "l2_policy_evict_first",
    "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
    (),
    (cutlass.Int64,),
    side_effects=True,
    memory=False,
    doc=(
        "``createpolicy.fractional.L2::evict_first``: the cache-policy word of "
        "streamed operand packets read once."
    ),
)

bulk_reduce_add_f32 = ptx_op(
    "bulk_reduce_add_f32",
    "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
    (cutlass.Int64, cutlass.Int32, cutlass.Int32),
    side_effects=True,
    memory=True,
    doc=(
        "``cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32``: one "
        "bulk-group reduce-add of ``size`` contiguous bytes (a multiple of 16) "
        "of fp32 from shared into global memory, issued by one thread; "
        "completion through ``cp.async.bulk.commit_group`` / ``wait_group``."
    ),
)

# ---- tensor maps (TMA) -----------------------------------------------------

fence_tensormap_acquire = ptx_op(
    "fence_tensormap_acquire",
    "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
    (cutlass.Int64,),
    side_effects=True,
    memory=False,
    doc=(
        "Publish a host-written tensor map to the tensormap proxy: the "
        "descriptor is written by the host and read by the TMA unit through "
        "a different proxy, so the kernel acquires it before first use."
    ),
)

tma_load_3d = ptx_op(
    "tma_load_3d",
    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
    " [$0], [$1, {$3, $4, $5}], [$2];",
    (
        SharedPointer,
        cutlass.Int64,
        SharedPointer,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
    ),
    side_effects=True,
    memory=False,
    doc="Issue one global-to-shared 3D TMA load (smem, descriptor, mbarrier, c0..c2).",
)

tma_load_4d = ptx_op(
    "tma_load_4d",
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
    " [$0], [$1, {$3, $4, $5, $6}], [$2];",
    (
        SharedPointer,
        cutlass.Int64,
        SharedPointer,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
    ),
    side_effects=True,
    memory=False,
    doc="One ``cp.async.bulk.tensor.4d`` box, global to shared (smem, descriptor, mbarrier, c0..c3).",
)

tma_store_3d = ptx_op(
    "tma_store_3d",
    "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
    " [$0, {$2, $3, $4}], [$1];",
    (cutlass.Int64, SharedPointer, cutlass.Int32, cutlass.Int32, cutlass.Int32),
    side_effects=True,
    memory=False,
    doc="Issue one shared-to-global 3D TMA store (descriptor, smem, c0..c2).",
)

tma_store_commit_group = ptx_op(
    "tma_store_commit_group",
    "cp.async.bulk.commit_group;",
    (),
    side_effects=True,
    memory=False,
    doc="Close the current bulk-store group.",
)

# ---- conversions -----------------------------------------------------------

# The asm text is the row-MMA runtime's spelling; the warp-MMA copy it also
# replaces differed by whitespace only (the fp16 warp_mma GEMM and mma_dag
# kernels compile to byte-identical cubins with either spelling).
unpack_f16x2 = ptx_op(
    "unpack_f16x2",
    "{.reg .b16 lo, hi; mov.b32 {lo, hi}, $2; cvt.f32.f16 $0, lo; cvt.f32.f16 $1, hi;}",
    (cutlass.Uint32,),
    (cutlass.Float32, cutlass.Float32),
    side_effects=False,
    memory=False,
    doc="The two f16 halves of a word (low half first) widened to fp32 (exact).",
)

f32_bits = ptx_op(
    "f32_bits",
    "mov.b32 $0, $1;",
    (cutlass.Float32,),
    (cutlass.Uint32,),
    side_effects=False,
    memory=False,
    doc=(
        "``mov.b32``: the bit pattern of an fp32 value (an operand word the "
        "tensor core truncates to tf32)."
    ),
)

bits_f32 = ptx_op(
    "bits_f32",
    "mov.b32 $0, $1;",
    (cutlass.Uint32,),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``mov.b32``: the fp32 value of a 32-bit word (a packet's element).",
)

round_bf16_f32 = ptx_op(
    "round_bf16_f32",
    "{ .reg .b32 w; cvt.rn.bf16x2.f32 w, $1, $1; and.b32 w, w, 0xffff0000; mov.b32 $0, w; }",
    (cutlass.Float32,),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc=(
        "The value rounded to bf16 (round to nearest even) and widened back "
        "to fp32: the bits of ``float32(bfloat16(value))``.  "
        "``cvt.rn.bf16x2.f32`` of the value with itself puts the rounded "
        "half-word in the high half (``F2FP.BF16.F32.PACK_AB``, an FMA-pipe "
        "instruction) and the mask is the exact widening; the scalar "
        "``cvt.rn.bf16.f32`` the DSL's cast emits runs on the conversion "
        "unit (``F2F``, the MIO queue) and its widening is a shift behind it."
    ),
)

mul_bf16x2 = ptx_op(
    "mul_bf16x2",
    "mul.rn.bf16x2 $0, $1, $2;",
    (cutlass.Int32, cutlass.Int32),
    (cutlass.Int32,),
    side_effects=False,
    memory=False,
    doc="Packed BF16 multiply of two b32 registers, rounding each product.",
)

sub_bf16x2 = ptx_op(
    "sub_bf16x2",
    "sub.rn.bf16x2 $0, $1, $2;",
    (cutlass.Int32, cutlass.Int32),
    (cutlass.Int32,),
    side_effects=False,
    memory=False,
    doc=(
        "Packed BF16 subtract of two b32 registers, rounding each difference: "
        "both operands already BF16, the exact difference is representable, "
        "so one ``sub.rn`` reproduces the ``BF16(V - X)`` boundary for two "
        "values at once."
    ),
)

# ---- predicated selects ----------------------------------------------------

mask_key_after = ptx_op(
    "mask_key_after",
    "{.reg .pred p; setp.gt.s32 p, $2, $3; selp.f32 $0, 0fFF800000, $1, p;}",
    (cutlass.Float32, cutlass.Int32, cutlass.Int32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``-inf if key > query else value``: the causal mask of one score.",
)

finite_or_zero = ptx_op(
    "finite_or_zero",
    "{.reg .pred p; setp.eq.f32 p, $1, 0fFF800000; selp.f32 $0, 0f00000000, $1, p;}",
    (cutlass.Float32,),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``0.0 if value == -inf else value`` (a fully masked row's maximum).",
)

sel_ge_f32 = ptx_op(
    "sel_ge_f32",
    "{ .reg .pred p; setp.ge.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
    (cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``x if a >= b else y`` (``setp`` + ``selp``): the inclusive triangular mask.",
)

sel_gt_f32 = ptx_op(
    "sel_gt_f32",
    "{ .reg .pred p; setp.gt.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
    (cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``x if a > b else y``: the strict triangular mask.",
)

sel_le_f32 = ptx_op(
    "sel_le_f32",
    "{ .reg .pred p; setp.le.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
    (cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``x if a <= b else y``.",
)

sel_lt_f32 = ptx_op(
    "sel_lt_f32",
    "{ .reg .pred p; setp.lt.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
    (cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``x if a < b else y``.",
)

sel_eq_f32 = ptx_op(
    "sel_eq_f32",
    "{ .reg .pred p; setp.eq.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
    (cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``x if a == b else y``.",
)

sel_ne_f32 = ptx_op(
    "sel_ne_f32",
    "{ .reg .pred p; setp.ne.s32 p, $1, $2; selp.f32 $0, $3, $4, p; }",
    (cutlass.Int32, cutlass.Int32, cutlass.Float32, cutlass.Float32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="``x if a != b else y``.",
)

# ---- warp shuffles ---------------------------------------------------------

shfl_up_add_f32 = ptx_op(
    "shfl_up_add_f32",
    "{.reg .f32 t; .reg .pred p; "
    "shfl.sync.up.b32 t|p, $1, $2, 0x0, 0xffffffff; "
    "@p add.f32 $0, $1, t; @!p mov.f32 $0, $1;}",
    (cutlass.Float32, cutlass.Int32),
    (cutlass.Float32,),
    side_effects=True,
    memory=False,
    doc=(
        "``value + value_of(lane - delta)``, the lanes below ``delta`` keeping "
        "``value`` (``shfl.sync.up`` and an add under its in-range predicate): "
        "one step of an inclusive prefix sum over the lanes ``delta`` apart.  "
        "Every lane must take part."
    ),
)

shfl_down_add_f32 = ptx_op(
    "shfl_down_add_f32",
    "{.reg .f32 t; .reg .pred p; "
    "shfl.sync.down.b32 t|p, $1, $2, 0x1f, 0xffffffff; "
    "@p add.f32 $0, $1, t; @!p mov.f32 $0, $1;}",
    (cutlass.Float32, cutlass.Int32),
    (cutlass.Float32,),
    side_effects=True,
    memory=False,
    doc=(
        "``value + value_of(lane + delta)``, the lanes within ``delta`` of the "
        "top keeping ``value``: one step of a reversed inclusive prefix sum."
    ),
)

# ---- mma.sync --------------------------------------------------------------

mma_e4m3 = ptx_op(
    "mma_e4m3",
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
    (
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ),
    (cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32),
    side_effects=False,
    memory=False,
    doc=(
        "``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32``: lane (g, c) "
        "supplies four e4m3 of rows g / g+8 at k 4c..4c+3 (a0 / a1) and "
        "16+4c.. (a2 / a3), column g of B at the same k (b0 / b1)."
    ),
)

mma_tf32 = ptx_op(
    "mma_tf32",
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
    (
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Uint32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ),
    (cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32),
    side_effects=False,
    memory=False,
    doc=(
        "``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``: lane (g, c) "
        "supplies A rows g / g+8 at k c (a0 / a1) and k c+4 (a2 / a3), column "
        "g of B at the same k (b0 / b1), as fp32 bit patterns the tensor core "
        "truncates to tf32 (the ``input_precision='tf32'`` numerics)."
    ),
)

# ---- fences and global atomics ---------------------------------------------

fence_acq_rel_gpu = ptx_op(
    "fence_acq_rel_gpu",
    "fence.acq_rel.gpu;",
    (),
    side_effects=True,
    memory=True,
    doc=(
        "``fence.acq_rel.gpu``: this thread's (and, after a CTA barrier, the "
        "CTA's) prior global writes are ordered before its later releases."
    ),
)

atom_add_release_gpu_u32 = ptx_op(
    "atom_add_release_gpu_u32",
    "atom.add.release.gpu.global.u32 $0, [$1], $2;",
    (cutlass.Int64, cutlass.Uint32),
    (cutlass.Uint32,),
    side_effects=True,
    memory=True,
    doc=(
        "``atom.add.release.gpu`` on the u32 at the address: the value before "
        "the add (this CTA's arrival ticket), unsigned."
    ),
)

ld_acquire_gpu_u32 = ptx_op(
    "ld_acquire_gpu_u32",
    "ld.acquire.gpu.global.u32 $0, [$1];",
    (cutlass.Int64,),
    (cutlass.Uint32,),
    side_effects=True,
    memory=True,
    doc=(
        "``ld.acquire.gpu`` of the u32 at the address (the peers' writes "
        "released before their arrivals are visible afterwards), unsigned."
    ),
)
