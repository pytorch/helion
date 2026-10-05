from __future__ import annotations

from typing import Any
from typing import cast

from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ._mlir_compat import ir


def _bit_width(mlir_type: ir.Type) -> int:
    if ir.IntegerType.isinstance(mlir_type):
        return ir.IntegerType(mlir_type).width
    return ir.FloatType(mlir_type).width


def _i32_constant(
    value: int, loc: ir.Location | None, ip: ir.InsertionPoint | None
) -> ir.Value:
    return llvm.mlir_constant(
        ir.IntegerAttr.get(Int32.mlir_type, value), loc=loc, ip=ip
    )


def _asm_kwargs(
    is_pure: bool, loc: ir.Location | None, ip: ir.InsertionPoint | None
) -> dict[str, Any]:
    return {
        "has_side_effects": not is_pure,
        "is_align_stack": False,
        "asm_dialect": llvm.AsmDialect.AD_ATT,
        "loc": loc,
        "ip": ip,
    }


@dsl_user_op
def inline_asm_elementwise(
    args: tuple[object, ...],
    *,
    asm: str = "",
    constraints: str = "",
    dtype: object = None,
    is_pure: bool = True,
    pack: int = 1,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> object:
    """CuTe scalar helper for ``hl.inline_asm_elementwise``.

    Helion's CuTe SIMT lowering operates on one scalar per thread.  The public
    API's tensor elementwise semantics are therefore implemented by calling this
    helper once per generated scalar lane.

    ``pack > 1`` keeps Triton's register ABI for the asm block (operands grouped
    per argument, sub-32-bit lanes packed into one ``pack * width``-bit
    register, ``pack`` results per dtype) but feeds the asm the same lane value
    in every pack slot and keeps slot 0 of every output group.  The language
    contract leaves "which set of inputs a block receives" unspecified, so this
    per-lane lowering is a valid instance of it.  On cute ``pack > 1`` is
    therefore correctness-only: the asm still runs once per element, so it
    brings no throughput benefit over ``pack == 1``.
    """

    if pack == 1:
        return _inline_asm_scalar(
            args,
            asm=asm,
            constraints=constraints,
            dtype=dtype,
            is_pure=is_pure,
            loc=loc,
            ip=ip,
        )
    return _inline_asm_packed(
        args,
        asm=asm,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
        loc=loc,
        ip=ip,
    )


def _inline_asm_scalar(
    args: tuple[object, ...],
    *,
    asm: str,
    constraints: str,
    dtype: object,
    is_pure: bool,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> object:
    operands = []
    for arg in args:
        operand = cast("Any", arg).ir_value(loc=loc, ip=ip)
        if str(operand.type) in {"i1", "i8", "i16"}:
            operand = llvm.zext(Int32.mlir_type, operand, loc=loc, ip=ip)
        operands.append(operand)
    kwargs = _asm_kwargs(is_pure, loc, ip)
    if isinstance(dtype, tuple):
        result = llvm.inline_asm(
            llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
                [cast("Any", dt).mlir_type for dt in dtype]
            ),
            operands,
            asm,
            constraints,
            **kwargs,
        )
        return tuple(
            cast("Any", dt)(
                llvm.extractvalue(
                    cast("Any", dt).mlir_type, result, [i], loc=loc, ip=ip
                )
            )
            for i, dt in enumerate(dtype)
        )
    dtype = cast("Any", dtype)
    return dtype(llvm.inline_asm(dtype.mlir_type, operands, asm, constraints, **kwargs))


def _inline_asm_packed(
    args: tuple[object, ...],
    *,
    asm: str,
    constraints: str,
    dtype: object,
    is_pure: bool,
    pack: int,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> object:
    multiple = isinstance(dtype, (tuple, list))
    dtypes = tuple(dtype) if multiple else (dtype,)

    operands: list[ir.Value] = []
    for arg in args:
        operand = cast("Any", arg).ir_value(loc=loc, ip=ip)
        if str(operand.type) == "i1":
            # Same as the scalar path: a predicate is handed over as a 32-bit
            # register (there is no meaningful ``pack x i1`` register).
            operand = llvm.zext(Int32.mlir_type, operand, loc=loc, ip=ip)
        width = _bit_width(operand.type)
        if width < 32:
            # Triton packs ``pack`` sub-32-bit lanes into one register.  libNVVM
            # rejects vector-typed asm operands, so build the vector and hand the
            # asm the equivalent integer.
            vector = llvm.mlir_undef(
                ir.VectorType.get([pack], operand.type), loc=loc, ip=ip
            )
            for i in range(pack):
                vector = llvm.insertelement(
                    vector, operand, _i32_constant(i, loc, ip), loc=loc, ip=ip
                )
            operand = llvm.bitcast(
                ir.IntegerType.get_signless(pack * width), vector, loc=loc, ip=ip
            )
            operands.append(operand)
        else:
            operands.extend([operand] * pack)

    # Result slot layout mirrors the operand layout: one ``pack * width``-bit
    # register per sub-32-bit dtype, ``pack`` separate results otherwise.
    result_types: list[ir.Type] = []
    first_slot: list[int] = []
    for dt in dtypes:
        mlir_type = cast("Any", dt).mlir_type
        first_slot.append(len(result_types))
        width = _bit_width(mlir_type)
        if width < 32:
            result_types.append(ir.IntegerType.get_signless(pack * width))
        else:
            result_types.extend([mlir_type] * pack)

    kwargs = _asm_kwargs(is_pure, loc, ip)
    if len(result_types) == 1:
        values = [
            llvm.inline_asm(result_types[0], operands, asm, constraints, **kwargs)
        ]
    else:
        struct = llvm.inline_asm(
            llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
                result_types
            ),
            operands,
            asm,
            constraints,
            **kwargs,
        )
        values = [
            llvm.extractvalue(result_type, struct, [i], loc=loc, ip=ip)
            for i, result_type in enumerate(result_types)
        ]

    outputs = []
    for dt, slot in zip(dtypes, first_slot, strict=True):
        mlir_type = cast("Any", dt).mlir_type
        value = values[slot]
        if _bit_width(mlir_type) < 32:
            vector = llvm.bitcast(
                ir.VectorType.get([pack], mlir_type), value, loc=loc, ip=ip
            )
            value = llvm.extractelement(
                vector, _i32_constant(0, loc, ip), loc=loc, ip=ip
            )
        outputs.append(cast("Any", dt)(value))
    return tuple(outputs) if multiple else outputs[0]
