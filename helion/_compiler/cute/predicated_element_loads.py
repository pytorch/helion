"""Masked element loads as single predicated loads that leave their select's
fill (``cute_predicated_element_loads``).

Role: general-purpose kernel-definition pass, placed after ``guard_lanes``.

The load lowering emits one element under a mask as ``NAME = (PTR).load()
if M else T(0)``: a load inside a conditional expression, which NVVM
compiles to a branch around the load.  A window over a padded image masks
each of its taps and then selects the padding value, ``torch.where(m,
hl.load(x, [...], extra_mask=m), -inf)``, so every tap is a branch, a load,
a widening and a select; ptxas merges some of the branches into predicated
loads and leaves the others, which splits a thread's loads over several
basic blocks (the loads of a block issue together, the blocks one after the
other) and keeps the selects between the loads and the arithmetic, where
ptxas would otherwise narrow a maximum of widened bf16 values to its
half-precision form.

``predicate_element_loads`` rewrites such a statement to
``_cute_element_fill_load_if(PTR, M, T, fill)``
(``element_policy_loads.element_fill_load_if``): ``mov fill; @p ld.global``,
the form Triton's ``tl.load(ptr, mask, other=...)`` lowers to.  The element
is not read where ``M`` is false, as before, and the statement stays where
it was.  ``fill`` is zero, the masked-off value of the conditional, unless
the load's value has one reader, a select under the same mask whose other
operand is a constant: ``W = casts(NAME) if M else C``, directly or through
single-use casts.  The load then leaves the bits of ``C`` and the select
becomes ``W = casts(NAME)``.

What the pass proves, statement by statement; a load it cannot prove keeps
its conditional, and a select it cannot prove keeps its place behind a
zero-filled predicated load:

* The load is the whole right-hand side of ``NAME = ...``, a plain
  ``(PTR).load()`` whose masked-off value is ``T(0)`` of a 2-, 4- or 8-byte
  float or signed integer type, and ``PTR`` is ``X.iterator`` plus offsets
  for a global tensor parameter ``X`` of dtype ``T`` that the body never
  rebinds, so the helper reads the bytes of one element (the lowering
  writes ``T(0)`` in the tensor's dtype; the pass checks it against the
  parameter's).  The predicated form evaluates the address where the mask
  is false too, so the address is plain arithmetic
  (``scalar_policy_loads._is_plain_address``).  A test the DSL decides
  while tracing (a literal, a name bound once to one, a
  ``cutlass.const_expr``) is no mask: such a statement is a plain load, or
  none, in the traced kernel and stays as it is.
* For the fill: ``NAME`` and every cast of it on the way are bound once in
  the kernel and read once, by the next link, all of them statements of the
  load's own statement list, so nothing else sees the masked-off value.  The
  select's test is the load's mask, expression for expression, built from
  names, constants, comparisons and integer arithmetic only, and no
  statement between the two binds a name it reads: both evaluate it to the
  same value.  Every cast on the way represents each value of its operand's
  type exactly (bf16 or fp16 to fp32 or fp64, fp32 to fp64, a signed
  integer to a wider one), and the select's constant, as the select's own
  casts round it, is a value of the element type: the element type's bits
  of it, cast the way the loaded element is cast, are the constant, zero
  signs and infinities included.  The constant is a bare literal (a float
  one only beside a float value) or a cast to the type of the select's
  value operand, so the select's result has that operand's type with and
  without the select.  A NaN constant keeps the select, and so does one
  that a cast to bf16 or fp16 could round to two values: the DSL rounds a
  Python float to the type once, a conversion of an fp32 value (and eager
  PyTorch) rounds the nearest fp32, and the two differ for a constant
  within an fp32 rounding of a tie of the half type (``_rounded``).

The pass changes no value.  It runs after ``guard_lanes``, which leaves a
tap's mask without the terms its store's guard implies: a load that still
carries tile masks or bounds beside the select's test keeps the select.  It
runs before ``hint_element_loads`` and passes on the cache policy that pass
would give the tensor's element loads (``element_load_hints``), so a hinted
masked element that takes a fill keeps its hint in the same instruction.
A hinted one whose fill stays zero is left as it is: ``hint_element_loads``
makes it the same predicated load with that policy
(``_cute_element_policy_load_if``) with or without this knob.  The two
configs are one source only where the knob drops no bound from that load's
mask either: where the ``extra_mask`` states the gather's bound and neither
``cute_guarded_lanes`` nor ``cute_vloop_sink`` has left the ``index < size``
term out already (``gathered_bounds.proven_gather_positions``), the lowering
leaves it out under the knob (``gathered_bounds.mask_bounded_gather_positions``),
so the knob's source is a second one and a search times both.
"""

from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import math
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from ...language.memory_ops import load
from ..indexing_strategy import subscript_index_scale
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import is_trace_time_test
from ._ast_pass_utils import loop_target_names
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import same_expr
from ._ast_pass_utils import statement_lists
from .analysis.match_common import node_tensor
from .dtypes import CUTLASS_INTEGER_TYPE_NAMES
from .emitters.lane_packets import _CUTE_VECTOR_UNROLL_DTYPES
from .flat_gather_lane import match_flat_gather_lane
from .scalar_policy_loads import _is_plain_address
from .scalar_policy_loads import _is_zero_of
from .scalar_policy_loads import _pointer_tensor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

PREDICATED_ELEMENT_LOAD = "_cute_element_fill_load_if"


def has_masked_gather_sites(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Whether the kernel has an ``hl.load`` under an ``extra_mask`` that
    gathers single elements along its tensor's last dim at every config: its
    last subscript is a tensor that is neither a tile's index shifted by a
    constant (``indexing_strategy.subscript_index_scale``) nor, for a dtype
    a vector lane loads in packets, a row offset plus a tile's index
    (``flat_gather_lane.match_flat_gather_lane``).  That leaves a scaled
    coordinate, ``2 * tile.index - 1``, and a value that is no function of
    one tile with unit coefficient, such as an index loaded from a tensor:
    such a load takes no packets, so its lanes are single elements under
    that mask, the loads ``predicate_element_loads`` rewrites.  Structure
    only, so ``cute_predicated_element_loads`` is offered at every shape of
    such a kernel.  Only a row offset and a tile index of the subscript's
    rank take the offer away (``broadcast_terms=False``): a 1-D tile index
    added to a row of higher rank (helion.inductor's flat subscripts) keeps
    it, and with it the search space such kernels had, although
    ``cute_vloop_sink`` loads that gather in packets as well; the offer
    costs the search configs to time there, never a value.  A masked load
    along a tile or a shifted tile (``tile.index - 24``: a padded packet) is
    left to ``cute_guarded_lanes``, and one that is contiguous along a tile
    of a flattened gather (an unfold, ``x_flat[tile_n.index[:, None] + C *
    tile_k.index[None, :]]``) to the vector loop that loads it in predicated
    packets (``cute_vloop_sink``): the single-element form only reaches the
    configs of such a kernel that take no packets, which lose to the ones
    that do (ConvBert's unfold, better-benchmark's pointwise_732ce9dc68ae:
    19.6 us at its packet tiles against 23.1 us for the best of the configs
    with predicated elements)."""
    for info in device_ir.graphs:
        for node in info.graph.find_nodes(op="call_function", target=load):
            if len(node.args) <= 2 or node.args[2] is None:
                continue
            index = node.args[1]
            if not isinstance(index, (list, tuple)) or not index:
                continue
            last = index[-1]
            if not (
                isinstance(last, torch.fx.Node)
                and isinstance(last.meta.get("val"), torch.Tensor)
            ):
                continue
            tile, scale = subscript_index_scale(env, last)
            if tile is not None and scale == 1:
                continue
            tensor = node_tensor(node.args[0])
            if (
                tensor is not None
                and tensor.dtype in _CUTE_VECTOR_UNROLL_DTYPES
                and match_flat_gather_lane(last, env, broadcast_terms=False) is not None
            ):
                continue
            return True
    return False


# The element types ``element_fill_load_if`` loads that a constant can be
# rounded to at codegen time, and the integer carrier of each one's bits.
_ELEMENT_DTYPES: dict[str, tuple[torch.dtype, torch.dtype]] = {
    "cutlass.BFloat16": (torch.bfloat16, torch.int16),
    "cutlass.Float16": (torch.float16, torch.int16),
    "cutlass.Float32": (torch.float32, torch.int32),
    "cutlass.Float64": (torch.float64, torch.int64),
    "cutlass.Int16": (torch.int16, torch.int16),
    "cutlass.Int32": (torch.int32, torch.int32),
    "cutlass.Int64": (torch.int64, torch.int64),
}
# Float type -> (significand bits, exponent bits): a cast is exact when its
# target has at least as many of both.
_FLOAT_FORMATS: dict[torch.dtype, tuple[int, int]] = {
    torch.bfloat16: (8, 8),
    torch.float16: (11, 5),
    torch.float32: (24, 8),
    torch.float64: (53, 11),
}


def _cast_is_exact(source: torch.dtype, target: torch.dtype) -> bool:
    """Whether ``target`` represents every value of ``source`` (so a cast
    between them changes no value): a float to one with at least its
    significand and exponent bits, a signed integer to one at least as
    wide."""
    if source in _FLOAT_FORMATS and target in _FLOAT_FORMATS:
        return all(
            ours <= theirs
            for ours, theirs in zip(
                _FLOAT_FORMATS[source], _FLOAT_FORMATS[target], strict=True
            )
        )
    if source.is_floating_point or target.is_floating_point:
        return False
    return target.itemsize >= source.itemsize


def _nearest(value: float, dtype: torch.dtype) -> float:
    """``value`` rounded once to the float type ``dtype``, to nearest even,
    in exact arithmetic: an infinity past the type's range, a zero of
    ``value``'s sign below its smallest subnormal."""
    if value == 0.0 or not math.isfinite(value):
        return value
    significand, exponent_bits = _FLOAT_FORMATS[dtype]
    # The type's largest exponent plus one, and its smallest normal one.
    past = 1 << (exponent_bits - 1)
    smallest = 2 - past
    # ``abs(value)`` is in ``[2**(exponent - 1), 2**exponent)``: the unit of
    # its last place in the type, or the subnormals' below the normal range.
    exponent = math.frexp(value)[1]
    unit = max(exponent - significand, smallest - (significand - 1))
    # ``round`` of a fraction is to nearest, ties to even.
    steps = round(Fraction(value) / Fraction(2) ** unit)
    if steps == 0:
        return math.copysign(0.0, value)
    if abs(steps) * Fraction(2) ** unit >= Fraction(2) ** past:
        return math.copysign(math.inf, value)
    return float(steps * Fraction(2) ** unit)


def _rounded(value: float, dtype: torch.dtype) -> float | None:
    """``value`` as a cast to ``dtype`` holds it; None where that is not one
    known value: an integer type that does not hold it, and a half-precision
    type for which rounding ``value`` once and rounding it through fp32
    differ.  The DSL rounds a Python float once (``cutlass.BFloat16(1 +
    2**-8 + 2**-40)`` is ``1 + 2**-7``); a conversion of an fp32 value, and
    eager PyTorch's of a Python float, round the fp32 nearest it (``1.0``
    there, the tie going to even).  A constant on which the two agree is the
    same value whichever the cast does."""
    if not dtype.is_floating_point:
        info = torch.iinfo(dtype)
        if (
            not math.isfinite(value)
            or value != int(value)
            or not info.min <= value <= info.max
        ):
            return None
        return value
    once = _nearest(value, dtype)
    if _FLOAT_FORMATS[dtype][0] < _FLOAT_FORMATS[torch.float32][0] and once != _nearest(
        _nearest(value, torch.float32), dtype
    ):
        return None
    return once


def _bits(value: float, dtype: torch.dtype, carrier: torch.dtype) -> int:
    """The bits of ``value`` as ``dtype`` holds it, as an unsigned integer."""
    if not dtype.is_floating_point:
        value = int(value)
    with unset_fake_temporarily():
        signed = int(
            torch.tensor(value, dtype=dtype, device="cpu").view(carrier).item()
        )
    return signed & ((1 << (8 * dtype.itemsize)) - 1)


def _is_pure_test(node: ast.expr) -> bool:
    """Whether ``node`` is built from names, constants, comparisons, Boolean
    and integer arithmetic and integer casts only, so two evaluations
    between which none of its names is bound agree: no call that could read
    memory, no subscript, no attribute."""
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.BoolOp):
        return all(_is_pure_test(value) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        return _is_pure_test(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_pure_test(node.left) and _is_pure_test(node.right)
    if isinstance(node, ast.Compare):
        return _is_pure_test(node.left) and all(
            _is_pure_test(comparator) for comparator in node.comparators
        )
    if isinstance(node, ast.Call):
        return (
            call_path_str(node.func) in CUTLASS_INTEGER_TYPE_NAMES
            and not node.keywords
            and all(_is_pure_test(argument) for argument in node.args)
        )
    return False


class _MaskedLoad(NamedTuple):
    """``NAME = (PTR).load() if M else T(0)``."""

    name: str
    pointer: ast.expr
    mask: ast.expr
    dtype: str


class _Casts(NamedTuple):
    """``T_n(...T_1(NAME))``: the name and the types from the inside out."""

    name: str
    dtypes: tuple[torch.dtype, ...]


def _casts(node: ast.expr) -> _Casts | None:
    """``node`` as casts of one name to element types (none: the name)."""
    dtypes: list[torch.dtype] = []
    while isinstance(node, ast.Call):
        entry = _ELEMENT_DTYPES.get(call_path_str(node.func) or "")
        if entry is None or len(node.args) != 1 or node.keywords:
            return None
        dtypes.append(entry[0])
        node = node.args[0]
    if not isinstance(node, ast.Name):
        return None
    return _Casts(node.id, tuple(reversed(dtypes)))


class _Constant(NamedTuple):
    """A constant operand: its value, and the element type a cast gives it
    or the Python type (``int`` / ``float``) of a bare literal."""

    value: float
    kind: torch.dtype | type


class _Kernel:
    """What the proofs read of the whole kernel: how often each name is
    bound and read, and the constants bound to names."""

    def __init__(self, definition: ast.FunctionDef) -> None:
        self.parameters = {argument.arg for argument in definition.args.args}
        self.bound: Counter[str] = Counter()
        self.read: Counter[str] = Counter()
        self.values: dict[str, ast.expr] = {}
        body = ast.Module(body=definition.body, type_ignores=[])
        for node in ast.walk(body):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    self.read[node.id] += 1
                else:
                    self.bound[node.id] += 1
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                self.values[node.targets[0].id] = node.value
        for name in loop_target_names(body):
            if isinstance(name.ctx, ast.Load):
                # A loop target built without a Store context: a binding,
                # counted above as a read.
                self.bound[name.id] += 1
                self.read[name.id] -= 1

    def single_use(self, name: str) -> bool:
        """Bound once in the kernel's body and read once.  (For the target
        of an assignment whose one reader follows it in its statement list,
        which is what ``_folded_fill`` asks about: that reader then sees
        this assignment's value and nothing else does.)"""
        return self.bound[name] == 1 and self.read[name] == 1

    def decided_while_tracing(self, test: ast.expr) -> bool:
        """Whether the DSL evaluates ``test`` as Python while it traces the
        kernel: a literal, a name whose only binding is a literal, a
        ``cutlass.const_expr`` call, or ``and`` / ``or`` / ``not`` of
        those."""
        if isinstance(test, ast.Constant) or is_trace_time_test(test):
            return True
        if isinstance(test, ast.Name):
            return (
                self.bound[test.id] == 1
                and test.id not in self.parameters
                and isinstance(self.values.get(test.id), ast.Constant)
            )
        if isinstance(test, ast.BoolOp):
            return all(self.decided_while_tracing(value) for value in test.values)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self.decided_while_tracing(test.operand)
        return False

    def constant(self, node: ast.expr, depth: int = 0) -> _Constant | None:
        """The number ``node`` evaluates to wherever it is evaluated, and
        its type: a literal, its negation, ``float('inf')`` /
        ``float('-inf')`` (a Python ``int`` or ``float``), a cast of one to
        an element type (rounded as the cast rounds; that type), or a name
        bound once to one.  None for anything else, a cast whose rounding
        is not one known value (``_rounded``) among them.  A NaN literal is
        returned as it is: it equals no value, so no fill is taken from it
        (``_folded_fill``)."""
        if depth > 8:
            return None
        if isinstance(node, ast.Constant):
            literal = node.value
            if isinstance(literal, bool) or not isinstance(literal, (int, float)):
                return None
            return _Constant(literal, type(literal))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            found = self.constant(node.operand, depth + 1)
            return None if found is None else _Constant(-found.value, found.kind)
        if isinstance(node, ast.Name):
            if self.bound[node.id] != 1 or node.id in self.parameters:
                return None
            value_node = self.values.get(node.id)
            if value_node is None:
                return None
            return self.constant(value_node, depth + 1)
        if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
            return None
        path = call_path_str(node.func)
        argument = node.args[0]
        if path == "float":
            if isinstance(argument, ast.Constant) and argument.value in ("inf", "-inf"):
                return _Constant(float(argument.value), float)
            return None
        entry = _ELEMENT_DTYPES.get(path or "")
        if entry is None:
            return None
        found = self.constant(argument, depth + 1)
        if found is None:
            return None
        value = _rounded(found.value, entry[0])
        return None if value is None else _Constant(value, entry[0])


def _masked_load(
    stmt: ast.stmt, kernel: _Kernel, tensors: Mapping[str, torch.dtype]
) -> _MaskedLoad | None:
    """``stmt`` as a masked element load the predicated form may replace
    (module docstring, first item)."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.IfExp)
    ):
        return None
    load = stmt.value.body
    if not (
        isinstance(load, ast.Call)
        and isinstance(load.func, ast.Attribute)
        and load.func.attr == "load"
        and not load.args
        and not load.keywords
    ):
        return None
    pointer = load.func.value
    tensor = _pointer_tensor(pointer)
    if tensor is None or kernel.bound[tensor] or not _is_plain_address(pointer):
        return None
    orelse = stmt.value.orelse
    if not isinstance(orelse, ast.Call) or kernel.decided_while_tracing(
        stmt.value.test
    ):
        return None
    dtype = call_path_str(orelse.func)
    if (
        dtype not in _ELEMENT_DTYPES
        or not _is_zero_of(orelse, dtype)
        # The helper reads as many bytes as ``T`` has: the tensor's own.
        # ``tensors`` names the global tensor parameters and nothing else.
        or tensors.get(tensor) != _ELEMENT_DTYPES[dtype][0]
    ):
        return None
    return _MaskedLoad(stmt.targets[0].id, pointer, stmt.value.test, dtype)


def _folded_fill(
    statements: list[ast.stmt], index: int, load: _MaskedLoad, kernel: _Kernel
) -> tuple[int, ast.stmt] | None:
    """The fill a select behind the load at ``statements[index]`` gives it:
    ``(bits, select)``, the select statement to become ``W = casts(NAME)``;
    None where the load's masked-off value stays zero (module docstring,
    second item)."""
    if not _is_pure_test(load.mask):
        return None
    mask_names = names_read(load.mask)
    element, carrier = _ELEMENT_DTYPES[load.dtype]
    name = load.name
    dtype = element
    position = index
    while True:
        if not kernel.single_use(name):
            return None
        reader = None
        for later in range(position + 1, len(statements)):
            stmt = statements[later]
            if name in names_read(stmt):
                reader = later
                break
            if _binds(stmt) & mask_names:
                return None
        if reader is None:
            return None
        stmt = statements[reader]
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return None
        select = stmt.value if isinstance(stmt.value, ast.IfExp) else None
        casts = _casts(stmt.value if select is None else select.body)
        # (The checks of the select's test and constant leave no other name
        # to cast in a statement that reads ``name``; this states it.)
        if casts is None or casts.name != name:
            return None
        for target in casts.dtypes:
            if not _cast_is_exact(dtype, target):
                return None
            dtype = target
        if select is None:
            # A cast on the way: its value is the next link.
            name = stmt.targets[0].id
            position = reader
            continue
        if not same_expr(select.test, load.mask):
            return None
        found = kernel.constant(select.orelse)
        if found is None:
            return None
        # The select has the type of its value operand, so the value alone
        # keeps it: a constant cast to another type, or a float literal
        # beside an integer, could widen the select's result.
        if isinstance(found.kind, torch.dtype):
            if found.kind != dtype:
                return None
        elif found.kind is float and not dtype.is_floating_point:
            return None
        # The constant is a value of the element type, so the exact casts
        # behind the load give it back.  (Rounding keeps the sign, of a zero
        # too, so equal values are one value; a NaN equals nothing.)
        held = _rounded(found.value, element)
        if held is None or held != found.value:
            return None
        return _bits(held, element, carrier), stmt


def _binds(stmt: ast.stmt) -> set[str]:
    """The names ``stmt`` binds, anywhere inside it, loop targets built
    without a Store context among them."""
    return names_stored_or_deleted(stmt) | {name.id for name in loop_target_names(stmt)}


def predicate_element_loads(
    definition: ast.FunctionDef,
    tensors: Mapping[str, torch.dtype],
    hints: Mapping[str, tuple[str, str] | None],
) -> list[ast.stmt]:
    """The body of the kernel ``definition`` with its masked element loads
    as predicated loads (module docstring).

    ``tensors`` maps the kernel's global tensor parameters to their dtypes:
    a load is taken only where its masked-off ``T(0)`` is of its tensor's
    dtype.  ``hints`` is the device state's ``element_load_hints``: a tensor
    with a policy recorded for its elements of the load's type passes it on
    to its predicated loads that take a fill, and keeps its zero-filled
    masked loads for ``hint_element_loads``.
    """
    kernel = _Kernel(definition)
    for statements in statement_lists(definition.body):
        for index, stmt in enumerate(statements):
            load = _masked_load(stmt, kernel, tensors)
            if load is None:
                continue
            assert isinstance(stmt, ast.Assign)
            hint = hints.get(_pointer_tensor(load.pointer) or "")
            policy = hint[1] if hint is not None and hint[0] == load.dtype else None
            fill = 0
            folded = _folded_fill(statements, index, load, kernel)
            if folded is not None:
                fill, select = folded
                assert isinstance(select, ast.Assign)
                assert isinstance(select.value, ast.IfExp)
                select.value = select.value.body
            elif policy is not None:
                # Zero bits under a cache policy: the hinted predicated load
                # that ``hint_element_loads`` writes for this statement.
                continue
            arguments: list[ast.expr] = [
                load.pointer,
                load.mask,
                ast.parse(load.dtype, mode="eval").body,
                ast.Constant(value=fill),
            ]
            if policy is not None:
                arguments.append(ast.Constant(value=policy))
            stmt.value = ast.copy_location(
                ast.Call(
                    func=ast.Name(id=PREDICATED_ELEMENT_LOAD, ctx=ast.Load()),
                    args=arguments,
                    keywords=[],
                ),
                stmt.value,
            )
            ast.fix_missing_locations(stmt)
    return definition.body
