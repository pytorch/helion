"""Predicated packet loads: one ``@p ld.global`` that leaves zeros when ``p`` is false.

A masked vector load spelled as ``load(ptr if mask else anchor)`` followed by
per-lane ``value if mask else 0`` selects forces every loaded register to be
consumed by its select right after the load, so ptxas reuses the destination
registers across the loads of a row strip and the loads issue one or two at
a time (a WAW scoreboard wait on each reuse).  The PTX form Triton's masked
``tl.load`` lowers to, ``mov 0; @p ld.global.v4``, writes each packet exactly
once with no consumer until the arithmetic that needs it, so all the loads of
the strip stay in flight.  These helpers emit that form for 16-, 8- and
4-byte packets, reinterpreted as the requested vector type, with the L1
eviction priority of the load site, and for 32- and 64-byte lanes (two or
four consecutive packets, one load each under the same predicate).

The helpers are ``_predicated_load`` rows, declared as plain ``name =
_predicated_load(...)`` assignments so the standalone exporter copies each
by name; ``cache_policy_loads.PACKET_LOAD_HELPERS`` lists the rows generated
code binds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector as _vector_dialect
from cutlass.cutlass_dsl import dsl_user_op

from ._mlir_compat import ir

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "load_v4b_if",
    "load_v8b_if",
    "load_v16b_if",
    "load_v32b_if",
    "load_v64b_if",
]

# Load-site L1 eviction priority -> PTX qualifier.
_PRIORITY_QUALIFIERS = {
    "": "",
    "evict_first": ".L1::evict_first",
    "evict_last": ".L1::evict_last",
    "cs": ".cs",
}


def _assembly(words: int, priority: str) -> str:
    """``mov 0`` every word, then predicated loads of ``words`` 32-bit words:
    one load of up to four words, or one 16-byte load per four words at
    consecutive addresses (a 32- or 64-byte lane: two or four packets)."""
    qualifier = _PRIORITY_QUALIFIERS[priority]
    zeros = " ".join(f"mov.b32 ${index}, 0;" for index in range(words))
    loads = []
    for first in range(0, words, 4):
        count = min(4, words - first)
        suffix = f".v{count}.b32" if count > 1 else ".b32"
        registers = ",".join(f"${index}" for index in range(first, first + count))
        operand = f"{{{registers}}}" if count > 1 else registers
        offset = f"+{first * 4}" if first else ""
        loads.append(f"@p ld.global{qualifier}{suffix} {operand}, [${words}{offset}];")
    return (
        f"{{ .reg .pred p; setp.ne.b32 p, ${words + 1}, 0; {zeros} {' '.join(loads)} }}"
    )


@dsl_user_op
def _load_vector_if(
    ptr: object,
    predicate: object,
    vec_type: ir.VectorType,
    *,
    words: int,
    priority: str = "",
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> ir.Value:
    """Load ``words`` aligned 32-bit words when ``predicate`` holds, zeros
    otherwise, bitcast to ``vec_type``."""
    address = cast("Any", ptr).toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    flag = cutlass.Boolean(predicate).ir_value(loc=loc, ip=ip)
    if str(flag.type) != "i32":
        flag = llvm.zext(cutlass.Int32.mlir_type, flag, loc=loc, ip=ip)
    u32 = cutlass.Uint32.mlir_type
    result_type = (
        llvm.StructType.get_literal([u32] * words)  # pyrefly: ignore[missing-attribute]
        if words > 1
        else u32
    )
    result = llvm.inline_asm(
        result_type,
        [address, flag],
        _assembly(words, priority),
        ",".join(["=r"] * words + ["l", "r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    values = (
        [
            llvm.extractvalue(u32, result, [index], loc=loc, ip=ip)  # pyrefly: ignore
            for index in range(words)
        ]
        if words > 1
        else [result]
    )
    packet_type = ir.VectorType.get([words], u32)
    packet = _vector_dialect.from_elements(packet_type, values, loc=loc, ip=ip)
    if str(vec_type) == str(packet_type):
        return packet
    return _vector_dialect.bitcast(vec_type, packet, loc=loc, ip=ip)


def _predicated_load(name: str, words: int) -> Callable[..., ir.Value]:
    """The ``@dsl_user_op`` loading ``words`` 32-bit words (one packet of up
    to 16 bytes, or two / four consecutive 16-byte packets) when its
    predicate holds and zeros otherwise, bitcast to the requested vector
    type, with the L1 eviction priority of the load site as ``priority``."""

    def load(
        ptr: object,
        predicate: object,
        vec_type: ir.VectorType,
        priority: str = "",
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> ir.Value:
        return _load_vector_if(
            ptr, predicate, vec_type, words=words, priority=priority, loc=loc, ip=ip
        )

    load.__name__ = load.__qualname__ = name
    span = (
        f"{4 * words}-byte packet"
        if words <= 4
        else f"{words // 4} consecutive 16-byte packets"
    )
    load.__doc__ = f"{span} when ``predicate`` holds, zeros otherwise."
    return dsl_user_op(load)


load_v64b_if = _predicated_load("load_v64b_if", 16)
load_v32b_if = _predicated_load("load_v32b_if", 8)
load_v16b_if = _predicated_load("load_v16b_if", 4)
load_v8b_if = _predicated_load("load_v8b_if", 2)
load_v4b_if = _predicated_load("load_v4b_if", 1)
