# pyrefly: ignore-errors
"""Element loads with a cache policy, under ``cute_scalar_load_hints`` and
``cute_narrow_load_hints``, and predicated element loads with a masked-off
value of their own, under ``cute_predicated_element_loads``.

``l2_policy.scalar_policy_load`` writes an L1 priority or the streaming
operator on a 4- or 8-byte element, which is every hinted element load of a
config without the knobs.  With the scalar knob a 4- or 8-byte element load
also takes the three L2 policies, and a hinted load under a mask is one
predicated instruction; with the narrow knob a 2-byte element (bf16, fp16,
int16) takes all six policies in the same forms.  These helpers write those
forms; ``scalar_policy_loads.hint_element_loads`` places them, after every
other pass, and leaves the older helper where a config without the knobs
has it.

``element_fill_load_if`` is the predicated form with the bits the element
reads as where its predicate is false (``mov fill; @p ld.global``, what
Triton's ``tl.load(ptr, mask, other=...)`` lowers to), with or without a
cache policy; ``predicated_element_loads.predicate_element_loads`` places it.

Called during ``@cute.kernel`` tracing (plain Python building MLIR), like
``l2_policy`` and ``masked_packet_load``; ``cache_policy_loads.
PACKET_LOAD_HELPERS`` lists the rows generated code binds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .cache_hint_forms import L1_PRIORITY_QUALIFIERS
from .cache_hint_forms import L2_POLICY_FORMS

if TYPE_CHECKING:
    from cutlass._mlir import ir

__all__ = ["element_fill_load_if", "element_policy_load", "element_policy_load_if"]

# Policy -> the qualifier of its ``ld.global`` and the ``createpolicy``
# eviction policy it passes ("" for none): the L1 priorities
# (``L1_PRIORITY_QUALIFIERS``) and the streaming operator, and the three L2
# families in the form their packet helpers carry (``L2_POLICY_FORMS``).
_POLICY_FORMS: dict[str, tuple[str, str]] = {
    "first": (L1_PRIORITY_QUALIFIERS["evict_first"], ""),
    "last": (L1_PRIORITY_QUALIFIERS["evict_last"], ""),
    "streaming": (".cs", ""),
    **{
        family: (f"{level1}.L2::cache_hint", policy)
        for family, (level1, policy) in L2_POLICY_FORMS.items()
        if family != "nc_l2_first"
    },
}
# Element byte width -> the unsigned carrier the loaded bits arrive in and
# its inline-asm register constraint.  A 2-byte element loads zero-extended
# into a 32-bit register (PTX lets ``ld`` write a register wider than its
# type) and is truncated afterwards: a 16-bit destination makes ptxas merge
# every load into its register's upper half (one ``PRMT`` per load).
_CARRIERS: dict[int, tuple[type[cutlass.Numeric], str]] = {
    2: (cutlass.Uint32, "r"),
    4: (cutlass.Uint32, "r"),
    8: (cutlass.Uint64, "l"),
}
_FLOATS = (cutlass.BFloat16, cutlass.Float16, cutlass.Float32, cutlass.Float64)
_ELEMENT_TYPES = (
    *_FLOATS,
    cutlass.Int16,
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Uint32,
    cutlass.Uint64,
)


def _assembly(policy: str, bits: int, *, predicated: bool, fill: int = 0) -> str:
    """One ``ld.global`` of a ``bits``-wide element with the cache hint of
    ``policy`` (``_POLICY_FORMS``; none for the empty string); an L2 family
    creates its policy descriptor first.  ``predicated`` sets the result to
    the bits ``fill`` (zero unless given) and loads under ``@p``
    (``$2 != 0``), the form of ``masked_packet_load``: one predicated
    instruction, where a load inside a conditional would compile to a branch
    around it."""
    qualifier, l2_policy = _POLICY_FORMS[policy] if policy else ("", "")
    load = f"ld.global{qualifier}.b{bits} $0, [$1]"
    setup = ""
    if l2_policy:
        setup = f".reg .b64 pol; createpolicy.fractional.L2::{l2_policy}.b64 pol, 1.0; "
        load += ", pol"
    if not predicated:
        return f"{{ {setup}{load}; }}"
    # A zero fill is spelled as before the fill existed.
    masked_off = f"{fill:#x}" if fill else "0"
    return (
        f"{{ .reg .pred p; {setup}setp.ne.b32 p, $2, 0; "
        f"mov.b{max(bits, 32)} $0, {masked_off}; @p {load}; }}"
    )


def _load_bits(
    ptr: object,
    predicate: object | None,
    dtype: type[cutlass.Numeric],
    policy: str,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
    fill: int = 0,
) -> cutlass.Numeric:
    """The element at ``ptr`` as ``dtype``, loaded as raw bits (no numeric
    conversion of floating point values, signed zero, NaNs or signed integer
    high bits); the bits ``fill`` where ``predicate`` is false.  Side
    effects and the memory clobber keep the load ordered with surrounding
    memory operations."""
    assert dtype in _ELEMENT_TYPES, dtype
    assert 0 <= fill < 1 << dtype.width, (dtype, fill)
    carrier, register = _CARRIERS[dtype.width // 8]
    operands = [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
    constraints = f"={register},l"
    if predicate is not None:
        flag = cutlass.Boolean(predicate).ir_value(loc=loc, ip=ip)
        if str(flag.type) != "i32":
            flag = llvm.zext(cutlass.Int32.mlir_type, flag, loc=loc, ip=ip)
        operands.append(flag)
        constraints += ",r"
    bits = llvm.inline_asm(
        carrier.mlir_type,
        operands,
        _assembly(policy, dtype.width, predicated=predicate is not None, fill=fill),
        f"{constraints},~{{memory}}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if dtype.width == 16:
        # Narrow through the DSL's own conversion (the raw ``llvm.trunc``
        # takes a flags operand that differs between DSL releases).
        return cutlass.Uint16(cutlass.Uint32(bits)).bitcast(dtype)
    if dtype in _FLOATS:
        bits = llvm.bitcast(dtype.mlir_type, bits, loc=loc, ip=ip)
    return dtype(bits)


@dsl_user_op
def element_policy_load(
    ptr: object,
    dtype: type[cutlass.Numeric],
    policy: str,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Numeric:
    """Load one element with the selected cache policy: an L1 priority
    (``first`` / ``last``), the ``streaming`` operator or an L2 family
    (``l2_last`` / ``l1_l2_first`` / ``l1_l2_last``), on a 2-, 4- or 8-byte
    element."""
    return _load_bits(ptr, None, dtype, policy, loc, ip)


@dsl_user_op
def element_policy_load_if(
    ptr: object,
    predicate: object,
    dtype: type[cutlass.Numeric],
    policy: str,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Numeric:
    """``element_policy_load`` when ``predicate`` holds, zero bits otherwise,
    as one predicated load: the value of ``load if predicate else dtype(0)``
    without reading ``ptr`` when the predicate is false."""
    return _load_bits(ptr, predicate, dtype, policy, loc, ip)


@dsl_user_op
def element_fill_load_if(
    ptr: object,
    predicate: object,
    dtype: type[cutlass.Numeric],
    fill: int,
    policy: str = "",
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cutlass.Numeric:
    """The element at ``ptr`` when ``predicate`` holds and the ``dtype``
    whose bits are ``fill`` otherwise, as one predicated load (``mov fill;
    @p ld.global``) that does not read ``ptr`` when the predicate is false;
    with the cache hint of ``policy`` (``element_policy_load``'s, or none
    for the empty string)."""
    return _load_bits(ptr, predicate, dtype, policy, loc, ip, fill)
