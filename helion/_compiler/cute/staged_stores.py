"""Stage a CTA's scalar stores in shared memory and flush them as packets.

Role: kernel-definition pass (``cute_staged_store``), last.

A pointwise kernel whose loads are contiguous along one tile and whose
stores are contiguous along another (an unfold: ``out[n, k] = x_flat[n + C *
(k - pad)]``) has no thread layout that makes both sides packets.  With the
loads in 16-byte packets (thread = V consecutive ``n`` x every ``k``) a
thread's stores are V scalars ``stride`` elements apart per ``k``, and each
warp-level store instruction touches 32 x V sectors with two bytes each: every
32-byte sector of the output goes to L2 once per element.  Yet the CTA as a
whole writes one contiguous span of the output.  With the knob on, the
threads write their elements into a shared-memory image of that span, meet at
one barrier, and flush the image with 16-byte shared-memory loads and 16-byte
global stores, thread ``t`` taking packets ``t``, ``t + T``, ...: the 32
threads of a warp then write 32 consecutive packets, 512 contiguous bytes, per
store instruction, and each sector goes to L2 once.  (The flush order is the
gain: the same image flushed by each thread's own packets is no faster than
direct stores.)

Soundness.  The pass replaces the kernel's stores into one tensor ``T`` only
when it proves, for the generated body:

* Effects.  Every write of the kernel is a scalar ``(T.iterator +
  ...).store(value)`` statement into the one tensor argument ``T``, bare or
  alone under an ``if``, and every other call is one the relocation proofs
  call pure or a load (``_is_proven_relocatable_call``), appends to a
  trace-time list or is a dependent-launch marker: no atomic, barrier,
  shared memory, packet store or early ``return``.  ``T`` is named nowhere
  but in those store addresses, and every other tensor argument the body
  names is proven not to overlap ``T``'s storage: no load can observe that
  the stores now land at the end of the CTA.
* Footprint.  The body is run symbolically over the thread block
  (``_Interpreter``): thread indices are arrays over the launch's threads,
  loops with literal trip counts run trip by trip, and a value the threads of
  a CTA share but the CTAs do not (a top-level definition over the block
  index, the tile origins built from it) is an opaque symbol with an integer
  coefficient.  Every store's element offset must be the same combination of
  those symbols plus a per-thread number, and every store mask a conjunction
  of comparisons of such values.  Every integer of the run must stay within
  signed 32 bits, for every thread of every CTA: there neither the kernel's
  32- or 64-bit arithmetic nor a cast between them wraps, so the proof's
  exact integers are the kernel's values.  An unsigned cast is followed
  only as an operand of the division the lowering emits for non-negative
  integers (``Int32(Uint32(a) % Uint32(b))``); any other unsigned value is
  unknown.  The same range holds for what the staged code computes: the
  span's base, the staging test and the flush addresses.  The stores whose
  masks hold in the launch's
  first CTA must write each element of one span ``[lo, lo + L)`` exactly
  once, ``L`` elements a multiple of the 16-byte packet and within the
  planners' static shared-memory budget.
* CTAs.  The symbols are evaluated for every CTA of the launch grid, which
  must be static (``PassContext.exact_grid``): a CTA stages when each of its
  masks has the truth it has in the first CTA (a comparison per symbol
  combination), so the same stores fill the same image, and every staging
  CTA's span must start on a 16-byte boundary of ``T``, whose base pointer is
  proven aligned (``cute_tensor_base_is_aligned``).  When every CTA stages,
  the stores that never run are dropped and the kernel has no direct stores
  left; otherwise each store keeps its direct form in the ``else`` of the
  CTA-uniform test (a partial last tile).

Anything else declines: dynamic shapes (no static grid or strides), an
empty tensor (a grid without a CTA), an output view at an unaligned offset,
an output that may alias an input, a rolled loop the proof cannot count,
stores into several tensors, an index or a mask operand past 32 bits (an
output of 2**31 elements).  The body is then left as it was, so the knob
is inert where the proof fails.

The staged kernel computes the same values and writes the same elements;
only the instruction that writes them changes.  The flush's packets are
plain ``cute.arch.store`` calls emitted after ``lower_packet_store_policy``
has run, so ``cute_store_policy`` does not reach them: they keep the default
cache policy.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from ..ast_extension import statement_from_string
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import statement_lists
from .emitters.lane_packets import _CUTE_VECTOR_UNROLL_CARRIER
from .literal_offset_loads import _stride_reference
from .packet_access import is_store_call
from .packet_access import match_packet_access
from .packet_access import vector_type_source
from .scalar_integer import _INTEGER_CASTS

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch

# The flush moves 16-byte packets: the widest store every target takes in one
# instruction, and half a 32-byte sector.
PACKET_BYTES = 16
# Statements the proof runs (each over arrays of the launch's threads) and
# CTAs it evaluates the CTA-uniform symbols for, before it gives up.
_MAX_STEPS = 1 << 16
_MAX_BLOCKS = 1 << 22
_LOOP_CALLS = frozenset({"range", "cutlass.range", "cutlass.range_constexpr"})
# The programmatic-dependent-launch markers (``cute_pdl``): they order this
# grid against the grids before and after it, whose own waits return only
# once this grid has completed, flush included.
_PDL_CALLS = frozenset(
    {"cute.arch.griddepcontrol_wait", "cute.arch.griddepcontrol_launch_dependents"}
)
_COMPARISONS: dict[type[ast.cmpop], str] = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
_OPERATOR_COMPARISONS = {f"operator.{name}": name for name in _COMPARISONS.values()}
# The casts the proof follows: the signed ones.  Arithmetic on an unsigned
# value wraps where the proof's integers go negative (``Uint32(0) - 1``), so
# an unsigned cast is an unknown value, but for the operands of the division
# the lowering emits for non-negative integers (``_exact_quotient``).
_SIGNED_CASTS = frozenset({"cutlass.Int32", "cutlass.Int64"})
assert _SIGNED_CASTS < _INTEGER_CASTS
# The proof computes exact integers; the kernel computes in 32 or 64 bits.
# Inside this range neither width's arithmetic nor a cast between them wraps,
# so every value the proof computes must stay in it (and no tensor of
# ``INDEX_LIMIT`` elements can be staged or gathered from).
INDEX_LIMIT = 1 << 31
_INDEX_RANGE = (-INDEX_LIMIT, INDEX_LIMIT)

_Number = np.ndarray | int


class _Decline(Exception):
    """The proof failed: the kernel keeps its direct stores."""


@dataclasses.dataclass(frozen=True)
class _Affine:
    """``const + sum(coefficient * symbol)``: ``const`` an integer per thread
    of the launch (an ``int`` when every thread holds the same) and the
    symbols the CTA-uniform values (``_Interpreter.symbols``) by name, with
    nonzero integer coefficients, sorted."""

    const: _Number
    symbols: tuple[tuple[str, int], ...] = ()

    def scaled(self, factor: int) -> _Affine:
        return _Affine(
            self.const * factor,
            tuple((name, value * factor) for name, value in self.symbols if factor),
        )

    def plus(self, other: _Affine, sign: int = 1) -> _Affine:
        symbols = dict(self.symbols)
        for name, value in other.symbols:
            symbols[name] = symbols.get(name, 0) + sign * value
        return _Affine(
            self.const + sign * other.const,
            tuple(sorted((name, value) for name, value in symbols.items() if value)),
        )


@dataclasses.dataclass(frozen=True)
class _Bound:
    """One comparison, ``value < 0`` (``less``) or ``value >= 0``, whose
    ``value`` holds a CTA-uniform symbol."""

    value: _Affine
    less: bool


# A mask: the conjunction of its items, each a ``_Bound`` or a truth value
# per thread (a ``bool`` when the threads agree).
_Mask = tuple[object, ...]
_Value = _Affine | _Mask | None


@dataclasses.dataclass(frozen=True)
class _Symbol:
    """A top-level definition ``name = definition`` the threads of a CTA
    share: pure integer arithmetic over the block index, static constants and
    earlier such values (``inputs``: what each name it reads held)."""

    name: str
    definition: ast.expr
    inputs: Mapping[str, _Affine]
    position: int


@dataclasses.dataclass(frozen=True)
class _Instance:
    """One execution of a store statement by every thread: the element offset
    from the tensor's base and the mask it ran under."""

    site: ast.Expr
    offset: _Affine
    mask: _Mask


def _index_axis(node: ast.AST, call: str) -> int | None:
    """``k`` of ``call()[k]``, ``call`` one of ``cute.arch.block_idx`` and
    ``cute.arch.thread_idx``."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and call_path_str(node.value.func) == call
        and not node.value.args
        and not node.value.keywords
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is int
        and 0 <= node.slice.value < 3
    ):
        return node.slice.value
    return None


def _axis_indices(dims: tuple[int, int, int]) -> list[np.ndarray]:
    """The index along each axis of an ``(x, y, z)`` launch shape, as arrays
    that broadcast to ``[z, y, x]``: flattened, the linear index."""
    x, y, z = dims
    return [
        np.arange(x, dtype=np.int64).reshape(1, 1, x),
        np.arange(y, dtype=np.int64).reshape(1, y, 1),
        np.arange(z, dtype=np.int64).reshape(z, 1, 1),
    ]


def _cast_operand(node: ast.AST, casts: Collection[str]) -> ast.expr | None:
    """The argument of ``cutlass.T(argument)`` for a ``T`` of ``casts``."""
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) in casts
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]
    return None


def _cast_argument(node: ast.AST) -> ast.expr | None:
    """What a signed integer cast ``cutlass.Int32(argument)`` converts: its
    argument, an unsigned division in it as the exact one
    (``_exact_quotient``)."""
    argument = _cast_operand(node, _SIGNED_CASTS)
    return None if argument is None else _exact_quotient(argument)


def _exact_quotient(node: ast.expr) -> ast.expr:
    """``a // b`` for ``cutlass.Uint32(a) // cutlass.Uint32(b)`` (and ``%``)
    under a signed cast, any other ``node`` as it is: the unsigned division
    the lowering emits for non-negative integers, which is the exact one for
    the operands the proof divides (non-negative by positive, within
    ``_INDEX_RANGE``).  An unsigned value used any other way stays unknown."""
    if not (
        isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod))
    ):
        return node
    unsigned = _INTEGER_CASTS - _SIGNED_CASTS
    left = _cast_operand(node.left, unsigned)
    right = _cast_operand(node.right, unsigned)
    return ast.BinOp(
        left=node.left if left is None else left,
        op=node.op,
        right=node.right if right is None else right,
    )


def _in_range(value: _Number, bounds: tuple[int, int]) -> bool:
    low, high = bounds
    return bool(np.all(np.asarray(value) >= low) and np.all(np.asarray(value) < high))


def _floor_divide(left: _Number, right: _Number, op: ast.operator) -> _Number | None:
    """``left // right`` or ``left % right`` of non-negative integers by
    positive ones, where the floored and the truncated forms agree."""
    if not (np.all(np.asarray(right) > 0) and np.all(np.asarray(left) >= 0)):
        return None
    return left // right if isinstance(op, ast.FloorDiv) else left % right


def _pointer_terms(pointer: ast.expr, tensor: str) -> list[ast.expr] | None:
    """The addends of a scalar address ``tensor.iterator + a + b ...`` (none
    for a rank-0 tensor's bare ``tensor.iterator``)."""
    terms: list[ast.expr] = []
    node = pointer
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        terms.append(node.right)
        node = node.left
    if call_path_str(node) != f"{tensor}.iterator":
        return None
    return terms[::-1]


class _Interpreter:
    """Run a kernel body for every thread of one CTA at once (the module
    docstring's footprint proof), collecting each execution of a store
    statement of ``sites``."""

    def __init__(
        self,
        *,
        tensor: str,
        sites: Collection[int],
        strides: Mapping[tuple[str, int], int],
        constexpr_values: Mapping[str, int],
        thread_block_dims: tuple[int, int, int],
    ) -> None:
        self.tensor = tensor
        self.sites = sites
        self.strides = strides
        self.constexpr_values = constexpr_values
        # Thread arrays are ``[z, y, x]``: flattened, the linear thread id.
        self.shape = thread_block_dims[::-1]
        self.thread_index = _axis_indices(thread_block_dims)
        self.env: dict[str, _Value] = {}
        self.symbols: dict[str, _Symbol] = {}
        # Symbols whose name a later statement assigned again: the name no
        # longer holds the symbol's value.
        self.rebound: set[str] = set()
        self.instances: list[_Instance] = []
        # Per combination of symbols, the least and the largest per-thread
        # part of the values computed over it: with the symbols' values at
        # every CTA, the range those values span (``_prove`` checks it).
        self.spans: dict[tuple[tuple[str, int], ...], list[int]] = {}
        self.steps = 0
        self.depth = 0
        self.position = 0

    # ---------------------------------------------------------------- values

    def eval(self, node: ast.AST) -> _Value:
        """The value of ``node``, an integer only while it cannot wrap: one
        that leaves ``_INDEX_RANGE`` is unknown, and one over CTA-uniform
        symbols is held to the range at every CTA (``self.spans``)."""
        value = self.evaluate(node)
        return self.held(value) if isinstance(value, _Affine) else value

    def held(self, value: _Affine) -> _Affine | None:
        """``value`` while it stays in ``_INDEX_RANGE``: checked here without
        symbols, recorded in ``self.spans`` for ``_prove`` with them."""
        low, high = int(np.min(value.const)), int(np.max(value.const))
        if not value.symbols:
            return value if _INDEX_RANGE[0] <= low and high < _INDEX_RANGE[1] else None
        span = self.spans.setdefault(value.symbols, [low, high])
        span[0] = min(span[0], low)
        span[1] = max(span[1], high)
        return value

    def evaluate(self, node: ast.AST) -> _Value:
        if isinstance(node, ast.Constant):
            if type(node.value) is bool:
                return (node.value,)
            return _Affine(node.value) if type(node.value) is int else None
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            value = self.constexpr_values.get(node.id)
            return None if value is None else _Affine(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self.eval(node.operand)
            return operand.scaled(-1) if isinstance(operand, _Affine) else None
        if isinstance(node, ast.BinOp):
            return self.binary(node)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return self.conjunction([self.eval(value) for value in node.values])
        if isinstance(node, ast.Compare):
            name = next(
                (
                    name
                    for kind, name in _COMPARISONS.items()
                    if len(node.ops) == 1 and isinstance(node.ops[0], kind)
                ),
                None,
            )
            if name is None:
                return None
            return self.compare(name, node.left, node.comparators[0])
        if isinstance(node, ast.Subscript):
            axis = _index_axis(node, "cute.arch.thread_idx")
            if axis is not None:
                return _Affine(self.thread_index[axis])
            stride = _stride_reference(node)
            if stride is not None and stride in self.strides:
                return _Affine(self.strides[stride])
            return None
        if isinstance(node, ast.Call):
            converted = _cast_argument(node)
            if converted is not None:
                # A signed conversion keeps a value of ``_INDEX_RANGE``,
                # which ``eval`` holds every value to.
                value = self.eval(converted)
                return value if isinstance(value, _Affine) else None
            name = call_path_str(node.func)
            if name in _OPERATOR_COMPARISONS and len(node.args) == 2:
                return self.compare(_OPERATOR_COMPARISONS[name], *node.args)
        return None

    def binary(self, node: ast.BinOp) -> _Value:
        left = self.eval(node.left)
        right = self.eval(node.right)
        if isinstance(node.op, ast.BitAnd) and not (
            isinstance(left, _Affine) and isinstance(right, _Affine)
        ):
            return self.conjunction([left, right])
        if not isinstance(left, _Affine) or not isinstance(right, _Affine):
            return None
        if isinstance(node.op, ast.Add):
            return left.plus(right)
        if isinstance(node.op, ast.Sub):
            return left.plus(right, -1)
        if isinstance(node.op, ast.Mult):
            if right.symbols:
                left, right = right, left
            if right.symbols or (left.symbols and not isinstance(right.const, int)):
                return None
            if left.symbols:
                return left.scaled(cast("int", right.const))
            return _Affine(left.const * right.const)
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            if left.symbols or right.symbols:
                return None
            value = _floor_divide(left.const, right.const, node.op)
            return None if value is None else _Affine(value)
        return None

    def conjunction(self, values: Sequence[_Value]) -> _Value:
        items: list[object] = []
        for value in values:
            if not isinstance(value, tuple):
                return None
            items.extend(value)
        return tuple(items)

    def compare(self, name: str, left: ast.AST, right: ast.AST) -> _Value:
        lhs = self.eval(left)
        rhs = self.eval(right)
        if not isinstance(lhs, _Affine) or not isinstance(rhs, _Affine):
            return None
        # ``a < b`` and ``a >= b`` are tests of ``a - b``; ``a > b`` and
        # ``a <= b`` of ``b - a``.
        difference = lhs.plus(rhs, -1) if name in ("lt", "ge") else rhs.plus(lhs, -1)
        less = name in ("lt", "gt")
        if difference.symbols:
            return (_Bound(difference, less),)
        truth = difference.const < 0 if less else difference.const >= 0
        return (bool(truth) if isinstance(truth, (bool, np.bool_)) else truth,)

    def uniform(self, node: ast.AST) -> bool:
        """Whether ``node`` is integer arithmetic every thread of a CTA
        evaluates alike: over the block index, static constants and names
        holding such values."""
        if isinstance(node, ast.Constant):
            return type(node.value) is int
        if isinstance(node, ast.Name):
            if node.id in self.env:
                value = self.env[node.id]
                return isinstance(value, _Affine) and isinstance(value.const, int)
            return node.id in self.constexpr_values
        if isinstance(node, ast.BinOp):
            return (
                isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
                and self.uniform(node.left)
                and self.uniform(node.right)
            )
        converted = _cast_argument(node)
        if converted is not None:
            return self.uniform(converted)
        return _index_axis(node, "cute.arch.block_idx") is not None

    # ------------------------------------------------------------ statements

    def run(self, body: Sequence[ast.stmt]) -> None:
        for position, statement in enumerate(body):
            self.position = position
            self.execute(statement)

    def bind(self, name: str, value: _Value) -> None:
        if name in self.symbols:
            self.rebound.add(name)
        self.env[name] = value

    def forget(self, statement: ast.AST) -> None:
        for name in names_stored_or_deleted(statement):
            self.bind(name, None)

    def holds_site(self, statement: ast.AST) -> bool:
        return any(id(node) in self.sites for node in ast.walk(statement))

    def execute(self, statement: ast.stmt) -> None:
        self.steps += 1
        if self.steps > _MAX_STEPS:
            raise _Decline
        if isinstance(statement, ast.Pass):
            return
        if isinstance(statement, ast.Expr):
            if id(statement) in self.sites:
                self.record(statement, ())
            return
        if isinstance(statement, ast.Assign):
            target = statement.targets[0] if len(statement.targets) == 1 else None
            if not isinstance(target, ast.Name):
                self.forget(statement)
                return
            value = self.eval(statement.value)
            if (
                value is None
                and self.depth == 0
                and target.id not in self.symbols
                and self.uniform(statement.value)
            ):
                # Shared by the CTA's threads, different between CTAs: a
                # symbol by its name (a name defined twice stays unknown).
                inputs = {
                    node.id: cast("_Affine", self.env[node.id])
                    for node in ast.walk(statement.value)
                    if isinstance(node, ast.Name) and node.id in self.env
                }
                self.env[target.id] = _Affine(0, ((target.id, 1),))
                self.symbols[target.id] = _Symbol(
                    target.id, statement.value, inputs, self.position
                )
                return
            self.bind(target.id, value)
            return
        if isinstance(statement, ast.For):
            self.loop(statement)
            return
        if isinstance(statement, ast.If):
            self.branch(statement)
            return
        if isinstance(statement, ast.AugAssign):
            self.forget(statement)
            return
        raise _Decline

    def trips(self, loop: ast.For) -> range | None:
        """The values ``loop`` counts through when its bounds are static."""
        call = loop.iter
        if (
            loop.orelse
            or not isinstance(loop.target, ast.Name)
            or not isinstance(call, ast.Call)
            or call_path_str(call.func) not in _LOOP_CALLS
            or call.keywords
            or not 1 <= len(call.args) <= 3
        ):
            return None
        bounds: list[int] = []
        for argument in call.args:
            value = self.eval(argument)
            if not (
                isinstance(value, _Affine)
                and not value.symbols
                and isinstance(value.const, int)
            ):
                return None
            bounds.append(value.const)
        if len(bounds) == 3 and bounds[2] == 0:
            return None
        return range(*bounds)

    def loop(self, loop: ast.For) -> None:
        trips = self.trips(loop)
        if trips is None:
            if self.holds_site(loop):
                raise _Decline
            self.forget(loop)
            return
        assert isinstance(loop.target, ast.Name)
        self.depth += 1
        for value in trips:
            self.bind(loop.target.id, _Affine(value))
            for statement in loop.body:
                self.execute(statement)
        self.depth -= 1

    def branch(self, branch: ast.If) -> None:
        mask = self.eval(branch.test)
        taken = self.decided(mask)
        if taken is not None:
            self.depth += 1
            for statement in branch.body if taken else branch.orelse:
                self.execute(statement)
            self.depth -= 1
            return
        stores = [stmt for stmt in branch.body if not isinstance(stmt, ast.Pass)]
        if (
            isinstance(mask, tuple)
            and stores
            and all(id(stmt) in self.sites for stmt in stores)
            and all(isinstance(stmt, ast.Pass) for stmt in branch.orelse)
        ):
            for store in stores:
                self.record(cast("ast.Expr", store), mask)
            return
        if self.holds_site(branch):
            raise _Decline
        self.forget(branch)

    def decided(self, mask: _Value) -> bool | None:
        """The truth of ``mask`` when every thread of every CTA agrees on it."""
        if not isinstance(mask, tuple):
            return None
        truths = [item for item in mask if not isinstance(item, _Bound)]
        if any(not np.any(item) for item in truths):
            return False
        if len(truths) == len(mask) and all(np.all(item) for item in truths):
            return True
        return None

    def record(self, site: ast.Expr, mask: _Mask) -> None:
        call = cast("ast.Call", site.value)
        terms = _pointer_terms(cast("ast.Attribute", call.func).value, self.tensor)
        if terms is None:
            raise _Decline
        offset = _Affine(0)
        for term in terms:
            value = self.eval(term)
            if not isinstance(value, _Affine):
                raise _Decline
            offset = offset.plus(value)
        self.instances.append(_Instance(site, offset, mask))


class _Grid:
    """The values of the CTA-uniform symbols at every CTA of a static grid."""

    def __init__(
        self,
        symbols: Mapping[str, _Symbol],
        constexpr_values: Mapping[str, int],
        block_grid_dims: tuple[int, int, int],
    ) -> None:
        self.symbols = symbols
        self.constexpr_values = constexpr_values
        self.shape = block_grid_dims[::-1]
        self.block_index = _axis_indices(block_grid_dims)
        self.values: dict[str, np.ndarray] = {}
        self.combinations: dict[tuple[tuple[str, int], ...], np.ndarray] = {}

    def symbol(self, name: str) -> np.ndarray:
        if name not in self.values:
            symbol = self.symbols[name]
            value = self.eval(symbol.definition, symbol)
            if value is None:
                raise _Decline
            self.values[name] = np.broadcast_to(
                np.asarray(value, dtype=np.int64), self.shape
            )
        return self.values[name]

    def combination(self, symbols: tuple[tuple[str, int], ...]) -> np.ndarray:
        """``sum(coefficient * symbol)`` at every CTA (computed once)."""
        if symbols not in self.combinations:
            total = np.zeros(self.shape, dtype=np.int64)
            for name, coefficient in symbols:
                total = total + coefficient * self.symbol(name)
            self.combinations[symbols] = total
        return self.combinations[symbols]

    def eval(self, node: ast.AST, symbol: _Symbol) -> _Number | None:
        """``node`` of ``symbol``'s definition at every CTA, None where a
        value leaves ``_INDEX_RANGE`` (as ``_Interpreter.eval``)."""
        value = self.evaluate(node, symbol)
        return value if value is not None and _in_range(value, _INDEX_RANGE) else None

    def evaluate(self, node: ast.AST, symbol: _Symbol) -> _Number | None:
        if isinstance(node, ast.Constant):
            # ``_Interpreter.uniform`` admitted integer constants only.
            return cast("int", node.value)
        if isinstance(node, ast.Name):
            if node.id in symbol.inputs:
                held = symbol.inputs[node.id]
                return held.const + self.combination(held.symbols)
            return self.constexpr_values[node.id]
        axis = _index_axis(node, "cute.arch.block_idx")
        if axis is not None:
            return self.block_index[axis]
        converted = _cast_argument(node)
        if converted is not None:
            return self.eval(converted, symbol)
        assert isinstance(node, ast.BinOp)
        left = self.eval(node.left, symbol)
        right = self.eval(node.right, symbol)
        if left is None or right is None:
            return None
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            return _floor_divide(left, right, node.op)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right


@dataclasses.dataclass(frozen=True)
class _Plan:
    """What the proof found: the store statements to stage, the span they
    fill (``lo``: its first element less the CTA-uniform part ``symbols`` of
    every offset; ``length`` elements), the position in the body of the
    first statement that holds a staged store (``first_site``), and the
    CTA-uniform test of a CTA that stages (``guard``; none when every CTA
    does)."""

    staged: tuple[ast.Expr, ...]
    skipped: tuple[ast.Expr, ...]
    first_site: int
    symbols: tuple[tuple[str, int], ...]
    lo: int
    length: int
    guard: tuple[tuple[tuple[tuple[str, int], ...], int | None, int | None], ...]


def _store_sites(
    body: Sequence[ast.stmt], tensors: Collection[str]
) -> tuple[str, list[ast.Expr]]:
    """The tensor the kernel writes and its store statements (the effects
    proof of the module docstring), else ``_Decline``."""
    sites: list[ast.Expr] = []
    pointers: set[int] = set()
    loop_calls: set[int] = set()
    target: str | None = None
    for statement in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if isinstance(
            statement,
            (ast.Return, ast.While, ast.With, ast.Try, ast.Break, ast.Continue),
        ):
            raise _Decline
        if isinstance(statement, ast.For) and isinstance(statement.iter, ast.Call):
            if call_path_str(statement.iter.func) in _LOOP_CALLS:
                loop_calls.add(id(statement.iter))
        if not isinstance(statement, ast.Expr):
            continue
        access = match_packet_access(statement.value)
        if access is None or access.kind != "store":
            continue
        root = access.pointer
        while isinstance(root, ast.BinOp) and isinstance(root.op, ast.Add):
            root = root.left
        name = call_path_str(root)
        if name is None or not name.endswith(".iterator"):
            raise _Decline
        name = name.removesuffix(".iterator")
        if name not in tensors or target not in (None, name):
            raise _Decline
        target = name
        sites.append(statement)
        pointers.update(id(node) for node in ast.walk(access.pointer))
    if target is None:
        raise _Decline
    stores = {id(site.value) for site in sites}
    # Trace-time lists (the loads a lane unroll issues first): appending a
    # value to one touches no memory.
    trace_lists = {
        node.targets[0].id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.List)
        and not node.value.elts
    } - set(tensors)
    exempt = stores | loop_calls
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and id(node) not in exempt:
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in trace_lists
                    and len(node.args) == 1
                    and not node.keywords
                ):
                    continue
                if call_path_str(node.func) in _PDL_CALLS:
                    continue
                if is_store_call(node) or not _is_proven_relocatable_call(
                    node, allow_load=True
                ):
                    raise _Decline
            elif isinstance(node, ast.Name) and node.id == target:
                if id(node) not in pointers:
                    raise _Decline
            elif isinstance(node, ast.Attribute) and node.attr == "iterator":
                if not (isinstance(node.value, ast.Name) and node.value.id in tensors):
                    raise _Decline
            elif isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name) and node.value.id in tensors:
                    raise _Decline
    return target, sites


def _prove(
    body: Sequence[ast.stmt],
    *,
    tensor: str,
    sites: Sequence[ast.Expr],
    itemsize: int,
    strides: Mapping[tuple[str, int], int],
    constexpr_values: Mapping[str, int],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int, int, int],
    smem_budget: int,
) -> _Plan:
    interpreter = _Interpreter(
        tensor=tensor,
        sites={id(site) for site in sites},
        strides=strides,
        constexpr_values=constexpr_values,
        thread_block_dims=thread_block_dims,
    )
    interpreter.run(body)
    instances = interpreter.instances
    if not instances:
        raise _Decline
    symbols = instances[0].offset.symbols
    if any(instance.offset.symbols != symbols for instance in instances):
        raise _Decline
    grid = _Grid(interpreter.symbols, constexpr_values, block_grid_dims)
    for combination, (low, high) in interpreter.spans.items():
        values = grid.combination(combination)
        if not _in_range(
            np.asarray([int(values.min()) + low, int(values.max()) + high]),
            _INDEX_RANGE,
        ):
            raise _Decline
    shape = interpreter.shape

    def at_first_cta(bound: _Bound) -> np.ndarray:
        value = bound.value.const + int(grid.combination(bound.value.symbols).flat[0])
        return np.broadcast_to(value < 0 if bound.less else value >= 0, shape)

    # The stores of the launch's first CTA: the image every staging CTA fills.
    offsets: list[np.ndarray] = []
    staged: dict[int, ast.Expr] = {}
    # Per combination of symbols ``s``, the range ``low <= s < high`` in which
    # every mask has its truth of the first CTA.
    ranges: dict[tuple[tuple[str, int], ...], list[int | None]] = {}
    for instance in instances:
        # Threads a symbol-free item masks off store in no CTA.
        live = np.ones(shape, dtype=bool)
        for item in instance.mask:
            if not isinstance(item, _Bound):
                live = live & np.broadcast_to(np.asarray(item, dtype=bool), shape)
        stores = live
        for item in instance.mask:
            if not isinstance(item, _Bound):
                continue
            truth = at_first_cta(item)
            stores = stores & truth
            # ``value = const + s``: ``const + s < 0`` iff ``s < -const``.
            threshold = np.broadcast_to(-np.asarray(item.value.const), shape)
            low, high = ranges.setdefault(item.value.symbols, [None, None])
            for holds in (True, False):
                selected = threshold[live & (truth == holds)]
                if not selected.size:
                    continue
                if item.less == holds:
                    # ``s < threshold`` for each of these threads.
                    bound = int(selected.min())
                    high = bound if high is None else min(high, bound)
                else:
                    bound = int(selected.max())
                    low = bound if low is None else max(low, bound)
            ranges[item.value.symbols] = [low, high]
        if stores.any():
            staged[id(instance.site)] = instance.site
            offsets.append(
                np.broadcast_to(np.asarray(instance.offset.const), shape)[stores]
            )
    if not offsets:
        raise _Decline
    elements = np.concatenate(offsets)
    lo = int(elements.min())
    length = int(elements.size)
    if (
        int(elements.max()) - lo + 1 != length
        or np.unique(elements).size != length
        or (length * itemsize) % PACKET_BYTES
        or length * itemsize > smem_budget
    ):
        raise _Decline
    # The CTAs that stage, and their spans' alignment.
    stages = np.ones(grid.shape, dtype=bool)
    for combination, (low, high) in ranges.items():
        value = grid.combination(combination)
        if low is not None:
            stages = stages & (value >= low)
        if high is not None:
            stages = stages & (value < high)
    if not stages.flat[0]:
        raise _Decline
    starts = (lo + grid.combination(symbols)) * itemsize
    if np.any(starts[stages] % PACKET_BYTES):
        raise _Decline
    # The staged code computes the span's base and the staging test from the
    # symbols in the kernel's index type, product by product and sum by sum:
    # none may wrap at any CTA, nor may a threshold or an address of the
    # span (a flush address; a store's terms, which the image's store sums
    # in the index type where the address added them to a pointer).  Each
    # value is checked as it is computed: they are arrays over the grid.

    def generated(value: _Number) -> None:
        if not _in_range(value, _INDEX_RANGE):
            raise _Decline

    generated(length)
    for combination in {symbols, *ranges}:
        total: _Number = 0
        for name, coefficient in combination:
            product = coefficient * grid.symbol(name)
            total = total + product
            generated(product)
            generated(total)
    base = lo + grid.combination(symbols)
    generated(base)
    generated(base + length)
    for bounds in ranges.values():
        for bound in bounds:
            if bound is not None:
                generated(bound)
    # The names the staged code reads must still hold the symbols' values
    # where it reads them: ahead of the first store.
    first_site = min(
        position
        for position, statement in enumerate(body)
        if any(id(node) in staged for node in ast.walk(statement))
    )
    used = {name for name, _ in symbols}
    used.update(name for combination in ranges for name, _ in combination)
    if any(
        name in interpreter.rebound or interpreter.symbols[name].position >= first_site
        for name in used
    ):
        raise _Decline
    return _Plan(
        staged=tuple(staged.values()),
        skipped=tuple(site for site in sites if id(site) not in staged),
        first_site=first_site,
        symbols=symbols,
        lo=lo,
        length=length,
        guard=()
        if stages.all()
        else tuple(
            (combination, low, high)
            for combination, (low, high) in ranges.items()
            if low is not None or high is not None
        ),
    )


def _combination_source(
    symbols: Sequence[tuple[str, int]], index_type: str, constant: int = 0
) -> str:
    terms = [f"{index_type}({name}) * {value}" for name, value in symbols]
    if constant or not terms:
        terms.append(str(constant))
    return " + ".join(terms)


def stage_cta_stores(
    body: list[ast.stmt],
    *,
    tensors: Mapping[str, torch.Tensor],
    base_aligned: Callable[[torch.Tensor], bool],
    proven_disjoint_tensor_pairs: Collection[frozenset[str]],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    constexpr_values: Mapping[str, int],
    thread_block_dims: tuple[int, int, int] | None,
    block_grid_dims: tuple[int, int, int] | None,
    index_type: str,
    dtype_str: Callable[[torch.dtype], str],
    smem_budget: int,
    new_var: Callable[[str], str],
) -> bool:
    """Stage the stores of the kernel ``body`` (in place) when the proofs of
    the module docstring hold; False, with ``body`` as it was, when one does
    not.

    ``tensors`` are the kernel's tensor arguments by name, ``base_aligned``
    whether one's base pointer is proven a multiple of ``PACKET_BYTES``,
    ``thread_block_dims`` the proven launch shape and ``block_grid_dims`` the
    proven static block grid (each None when unproven), ``index_type`` and
    ``dtype_str`` the backend's spellings of its index type and of a
    tensor's element type.
    """
    if thread_block_dims is None or block_grid_dims is None:
        return False
    # A grid without a CTA (an empty tensor) has no first CTA to take the
    # image from, and nothing to stage.
    if not 0 < int(np.prod(block_grid_dims)) <= _MAX_BLOCKS:
        return False
    try:
        tensor, sites = _store_sites(body, tensors)
        dtype = tensors[tensor].dtype
        if dtype not in _CUTE_VECTOR_UNROLL_CARRIER or not base_aligned(
            tensors[tensor]
        ):
            return False
        named = {
            node.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id in tensors
        }
        if any(
            frozenset((tensor, other)) not in proven_disjoint_tensor_pairs
            for other in named - {tensor}
        ):
            return False
        plan = _prove(
            body,
            tensor=tensor,
            sites=sites,
            itemsize=dtype.itemsize,
            strides=proven_tensor_stride_values,
            constexpr_values=constexpr_values,
            thread_block_dims=thread_block_dims,
            block_grid_dims=block_grid_dims,
            smem_budget=smem_budget,
        )
    except _Decline:
        return False
    stage = new_var("_cta_stage")
    base = new_var("_cta_stage_base")
    full = new_var("_cta_stage_full") if plan.guard else None
    carrier = _CUTE_VECTOR_UNROLL_CARRIER[dtype]
    width = PACKET_BYTES // dtype.itemsize
    element = dtype_str(dtype)

    # The store statements: into the image at the element's offset in the
    # span, and (a CTA that does not stage) through the address as before.
    replacements: dict[int, ast.stmt] = {}
    for site in plan.staged:
        call = cast("ast.Call", site.value)
        terms = _pointer_terms(cast("ast.Attribute", call.func).value, tensor)
        assert terms is not None
        offset = " + ".join(
            ast.unparse(_with_static_strides(term, proven_tensor_stride_values))
            for term in terms
        )
        staged = statement_from_string(
            f"({stage} + (({offset or 0}) - {base})).store({{value}})",
            value=deep_clone(call.args[0]),
        )
        replacements[id(site)] = (
            staged if full is None else _if(full, [staged], [deep_clone(site)])
        )
    for site in plan.skipped if full is None else ():
        # No CTA runs this store: its mask never holds.
        replacements[id(site)] = ast.Pass()
    for statements in statement_lists(body):
        statements[:] = [replacements.get(id(stmt), stmt) for stmt in statements]

    allocation = (
        f"cute.arch.alloc_smem({element}, {plan.length}, alignment={PACKET_BYTES})"
    )
    setup = [
        f"{stage} = {allocation}",
        f"{base} = {_combination_source(plan.symbols, index_type, plan.lo)}",
    ]
    if full is not None:
        tests = []
        for combination, low, high in plan.guard:
            value = _combination_source(combination, index_type)
            if low is not None:
                tests.append(f"({value} >= {low})")
            if high is not None:
                tests.append(f"({value} < {high})")
        setup.append(f"{full} = {' & '.join(tests)}")
    body[plan.first_site : plan.first_site] = [
        statement_from_string(line) for line in setup
    ]

    # The flush: thread ``t`` moves packets ``t``, ``t + T``, ..., so a warp
    # writes 32 consecutive packets per store instruction.
    x, y, _ = thread_block_dims
    threads = int(np.prod(thread_block_dims))
    packets = plan.length // width
    tid = new_var("_cta_stage_thread")
    linear = " + ".join(
        f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {scale}"
        for axis, scale in enumerate((1, x, x * y))
        if thread_block_dims[axis] > 1
    )
    flush: list[ast.stmt] = [
        statement_from_string("cute.arch.sync_threads()"),
        statement_from_string(f"{tid} = {linear or 'cutlass.Int32(0)'}"),
    ]
    vector = vector_type_source(width, carrier)
    for first in range(0, packets, threads):
        packet = new_var("_cta_stage_packet")
        moves = [
            statement_from_string(
                f"{packet} = cute.arch.load("
                f"{stage} + ({tid} + {first}) * {width}, {vector})"
            ),
            statement_from_string(
                f"cute.arch.store({tensor}.iterator + ({base} + "
                f"{index_type}(({tid} + {first}) * {width})), {packet})"
            ),
        ]
        if first + threads > packets:
            # The last round has packets for the first threads only.
            flush.append(_if(f"{tid} < {packets - first}", moves, []))
        else:
            flush.extend(moves)
    body.extend(flush if full is None else [_if(full, flush, [])])
    return True


def _with_static_strides(
    term: ast.expr, strides: Mapping[tuple[str, int], int]
) -> ast.expr:
    """``term`` with each ``T.layout.stride[d]`` as its proven value, which
    lets the compiler see neighbouring elements of the image as neighbours."""

    class Strides(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            stride = _stride_reference(node)
            if stride is not None and stride in strides:
                return ast.Constant(value=strides[stride])
            return self.generic_visit(node)

    result = Strides().visit(deep_clone(term))
    assert isinstance(result, ast.expr)
    return result


def _if(test: str, body: list[ast.stmt], orelse: list[ast.stmt]) -> ast.If:
    branch = statement_from_string(f"if {test}:\n    pass")
    assert isinstance(branch, ast.If)
    branch.body = body
    branch.orelse = orelse
    return branch
