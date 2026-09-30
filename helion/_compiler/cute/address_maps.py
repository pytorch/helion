"""The thread-to-element mapping of the addresses an emitted body computes.

``lane_loop_distribution.add_thread_barriers`` separates with barriers the
accesses of one tensor that the threads of a CTA could race on.  Whether two
accesses can touch one element from different threads is a question about
their address mappings, not about the names they read: ``x[t + 1]`` beside
``x[t]`` reads the thread's coordinate and yet names the neighbour's
element, and so do ``x[n - 1 - t]`` and ``x[(t + 1) % n]``.  This module
translates an address term into a symbolic expression over the thread
coordinates (``cute.arch.thread_idx()[axis]``), the loop variables of the
lane loops, V-loops and the loops inside a statement, and uniform values
(everything else: a tile offset, a kernel argument, a shape query), by
inlining the definitions reaching the body in program order, and proves for
a pair of accesses which coordinates the equations "a term of the first
equals the same term of the second" force to be equal: along those thread
axes the two accesses can only meet on one thread, whose accesses stay in
program order.  A lane variable forced equal confines the pair to its lane;
the iteration of the device loop around the body forced equal (its
variable modelled as ``start + step * iteration`` from the loop's header)
confines it to one iteration.

The proof is a mixed-radix argument over the difference of the two sides,
a sum of terms ``c_i * y_i`` over bounded integer quantities: a variable's
difference between the accesses (``|y| <= r - 1`` for a range ``r``), a
variable or a ``p % m`` of one access alone (``0 <= y <= r - 1``), the
constant.  Sorted by coefficient, the terms fall into groups summing to
zero: a group closes when the interval of its sum lies strictly between
the negated and the positive gcd of the later coefficients (and of the
modulus, for an equation ``p % m == q % m`` taken modulo ``m``), and inside
a closed group a term is zero when the sum of the terms before it is
smaller than its coefficient.  This proves the tile's per-thread index
``offset + tid + lane * threads`` with ``tid < threads`` one thread's, a
contiguous ``offset + tid * lanes + lane`` with ``lane < lanes``, a
flattened ``tid_x + tid_y * 32`` with ``tid_x < 32``, and a swizzled
shared-memory address whose dimensions ``a % m`` and ``a // m`` together
determine ``a``.  The ranges come from the launch thread counts (refined by
a conjunct ``thread_idx()[axis] < c`` of the branches enclosing both
accesses), the lane loops' trip counts and the literal bounds of the loops
(a block-size constant included).  A conjunct bounding an index by a
literal (a tile mask's ``indices < end`` with the end a literal) bounds
that index as one quantity for an access under it (``bounds``,
``_BOUNDED``): the padded elements past the tile's end, which the program
never touches, do not count, so a flattened ``n * row + column`` with
``column = 64 * iteration + thread < n`` is one row's whatever the column
tile's padding; the quantity is one of the argument's terms when both
accesses run under the bound (``pair_ranges``), its value's parts
otherwise, and two quantities forced equal force their values' parts
equal in turn.  A conjunct bounding the index by a uniform value instead
(the end a kernel argument: a dynamic shape's size, a segment's end every
thread loads alike) makes that value a radix of the argument: the terms
it multiplies and the bounded index's difference between the two
accesses are zero separately (``K * row + column`` with ``column < K``
equal on both sides forces the rows equal and the columns equal, whatever
value ``K`` has, since it exceeds a nonnegative column), while the index
stays the sum of its parts in the terms for the rest of the argument.  A
conjunct is read where its branch stands: one around a compound statement
over the definitions reaching the body, so it bounds none of the names the
statement defines for itself (a device loop's own index), one inside it
over the statement's own definitions.  A packet (a
vector load, a store flush) covers one element per iteration of its
wrapper's V-loop along the dimension its per-thread base indexes: its term
there is the base's plus the V-loop's variable (``packet``), so the
packet's later elements meet the other access as much as its first, and
two accesses whose terms can never be equal never meet.  A value the
translation does not know (a loaded index, a name a call defines) is an
unknown integer of its own, differing between the two accesses like any
variable: the part of a term it stands for proves nothing, while the rest
still does (``base + tid`` with a loaded ``base`` times the block size
confines the pair to the thread).  A name a definition around the body
loads from a kernel argument nothing in view writes (a tensor made in
view, a thread's register array, is no such source), at an address over
uniform values alone, is a uniform value itself (``uniform_load``): every
thread and lane reads the same element before the body runs (a segment's
start loaded by the block's index), so the rows it offsets are one lane's
or one thread's when the rest of the term says so.
Whatever the argument cannot prove counts as another thread's element: a
gathered index, a modular wrap, a reversal, an offset differing between the
two accesses, a loop variable of unknown range before a larger coefficient.

The variable of a loop around the body is one value on every thread within
an iteration and ``start + step * iteration`` between two of them
(``iteration_symbol``); it is bound by the loop's own header, whatever
earlier statement (a finished sibling loop over the same block, reusing the
variable) defined the name before the body.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import re
from typing import TYPE_CHECKING
from typing import cast

import sympy

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

_FLOOR_DIVIDE = sympy.Function("_helion_floor_divide")
_MODULO = sympy.Function("_helion_modulo")
# An index a tile mask's conjunct bounds, kept as one quantity of the proof
# (``AddressMaps.bounds``); its argument is the index's value.
_BOUNDED = sympy.Function("_helion_bounded")
_Equation = tuple[sympy.Expr, sympy.Expr, int]
# A bound on a quantity of the proof (``AddressMaps.bounds``): a literal, or a
# uniform value the argument takes as a radix (``AddressMaps._forced_equal``).
Bound = int | sympy.Symbol
_UNIFORM_PREFIX = "_helion_uniform_"
_INTEGER_CAST = re.compile(r"^(cutlass\.)?(Int|Uint)(8|16|32|64)$")
_THREAD_INDEX = "cute.arch.thread_idx"
# Calls whose value is the same on every thread of the CTA and in every
# iteration: a shape query of a tensor, the CTA's own coordinates.
_UNIFORM_CALLS = (
    "cute.size",
    "cute.arch.block_idx",
    "cute.arch.block_dim",
    "cute.arch.grid_dim",
    "cute.arch.cluster_idx",
    "cute.arch.cluster_dim",
    "cute.arch.block_in_cluster_idx",
)
_OPERATORS: dict[str, type[ast.operator | ast.unaryop]] = {
    "operator.add": ast.Add,
    "operator.sub": ast.Sub,
    "operator.mul": ast.Mult,
    "operator.floordiv": ast.FloorDiv,
    "operator.mod": ast.Mod,
    "operator.neg": ast.USub,
}
_LOOP_CALLS = ("range", "cutlass.range", "cutlass.range_constexpr")


def thread_symbol(axis: int) -> sympy.Symbol:
    """The coordinate of a thread along launch axis ``axis``."""
    return sympy.Symbol(f"_helion_thread_{axis}", integer=True, nonnegative=True)


def loop_symbol(name: str) -> sympy.Symbol:
    """The iteration of the loop with variable ``name`` (its trip index)."""
    return sympy.Symbol(f"_helion_loop_{name}", integer=True, nonnegative=True)


def uniform_symbol(name: str) -> sympy.Symbol:
    """The value of a name the body does not define (uniform on every thread)."""
    return sympy.Symbol(f"{_UNIFORM_PREFIX}{name}", integer=True)


def iteration_symbol(name: str) -> sympy.Symbol:
    """The iteration of the device loop with variable ``name`` around the body."""
    return sympy.Symbol(f"_helion_iteration_{name}", integer=True, nonnegative=True)


def _other_side(symbol: sympy.Symbol) -> sympy.Symbol:
    """``symbol`` as evaluated by the second access of a pair."""
    return sympy.Symbol(f"{symbol.name}__other", **symbol.assumptions0)


def _bounded(value: sympy.Expr) -> sympy.Expr:
    """``value`` as one bounded quantity of the proof (``AddressMaps.bounds``)."""
    return cast("sympy.Expr", _BOUNDED(sympy.expand(value)))


def _tighter(first: Bound, second: Bound) -> Bound:
    """The tighter of two bounds on one quantity: the smaller literal, a literal over a uniform value, else the first."""
    if isinstance(first, int) and isinstance(second, int):
        return min(first, second)
    if isinstance(second, int):
        return second
    return first


def whole_quantities(bounds: Mapping[sympy.Expr, Bound]) -> frozenset[sympy.Expr]:
    """The quantities among ``bounds`` a term keeps whole (``AddressMaps.translate``).

    An index bounded by a literal, which the mixed-radix argument ranges
    as one term; one bounded by a uniform value stays the sum of its
    parts, which the argument's symbolic radix step matches
    (``AddressMaps._forced_equal``).
    """
    return frozenset(
        quantity for quantity, bound in bounds.items() if isinstance(bound, int)
    )


@dataclasses.dataclass(frozen=True)
class Load:
    """A load among the definitions around a body, defining one name.

    ``tensor`` is the tensor loaded, ``terms`` the index term of each of its
    dimensions, ``guards`` the mask and the fill of a masked load (``...
    if mask else fill``; empty for a plain one), ``value`` the defining
    statement's value node and ``statement`` the statement
    (``AddressMaps.uniform_load``).
    """

    tensor: str
    terms: tuple[ast.AST, ...]
    guards: tuple[ast.AST, ...]
    value: ast.expr
    statement: ast.AST


def _flatten(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Tuple):
        return [term for element in node.elts for term in _flatten(element)]
    return [node]


def flat_terms(terms: Iterable[ast.AST]) -> list[ast.AST]:
    """The per-dimension terms of an address, a nested coordinate tuple flattened."""
    return [term for node in terms for term in _flatten(node)]


class _Local:
    """The names one statement defines and the loops it runs, in source order."""

    def __init__(self, node: ast.AST, renames: Mapping[str, str]) -> None:
        self.renames = renames
        self.definitions: dict[str, ast.expr | None] = {}
        self.loops: list[ast.For] = []
        self._read: set[str] = set()
        self._visit(node)

    def _canonical(self, name: str) -> str:
        return self.renames.get(name, name)

    def _define(self, target: ast.expr, value: ast.expr | None) -> None:
        if isinstance(target, ast.Name):
            name = self._canonical(target.id)
            # A second definition, or one after a read (a loop-carried
            # value), leaves the name's value unknown.
            self.definitions[name] = (
                None if name in self.definitions or name in self._read else value
            )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._define(element, None)
        elif isinstance(target, ast.Starred):
            self._define(target.value, None)
        else:
            # A subscript or attribute target stores into the object; the
            # name keeps its value, and the index expressions are reads.
            self._visit(target)

    def _visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Assign):
            self._visit(node.value)
            for target in node.targets:
                self._define(target, node.value)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self._visit(node.value)
            self._define(node.target, node.value)
        elif isinstance(node, ast.AugAssign):
            self._visit(node.value)
            self._define(node.target, None)
        elif isinstance(node, ast.For):
            self._visit(node.iter)
            self.loops.append(node)
            self._define(node.target, None)
            for child in [*node.body, *node.orelse]:
                self._visit(child)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = self._canonical(node.id)
            if name not in self.definitions:
                self._read.add(name)
        else:
            for child in ast.iter_child_nodes(node):
                self._visit(child)


class AddressMaps:
    """The symbolic address terms of a body's statements and the pair proofs over them.

    ``leaves`` are the statements the barrier pass pairs (a compound
    statement defines names and runs loops of its own), ``definitions`` the
    per-thread index definitions around the body, ``loop_counts`` the trip
    counts of the lane loops and V-loops by variable, ``loaded`` the names a
    load, an atomic or a call handed a tensor defines (their values are
    unknown integers, ``_unknown``, unless every thread loads the same
    element, ``uniform_load``; a name defined from one is translated
    through its definition), ``loads`` the loads among the definitions by
    the name each defines, ``written`` the tensors any statement in view
    writes or may write, ``constants`` the values of the block-size
    constants and ``axis_sizes`` the launch thread count along each axis.
    """

    def __init__(
        self,
        *,
        axis_sizes: Mapping[int, int],
        loop_counts: Mapping[str, int],
        leaves: Sequence[ast.AST],
        definitions: Sequence[ast.AST],
        renames: Mapping[str, str],
        loaded: Iterable[str],
        constants: Mapping[str, int],
        outer_loops: Sequence[ast.For] = (),
        loads: Mapping[str, Load] | None = None,
        written: Iterable[str] = (),
    ) -> None:
        self.axis_sizes = dict(axis_sizes)
        self.renames = renames
        self.loaded = frozenset(loaded)
        self.loads = dict(loads or {})
        self.written = frozenset(written)
        # Whether a loaded name is one value on every thread and lane
        # (``uniform_load``), by name.
        self._uniform: dict[str, bool] = {}
        # The names the body's own statements define (``_collect``).
        self.body_defined: set[str] = set()
        self.constants = {
            name: sympy.Integer(value) for name, value in constants.items()
        }
        # The variables: the thread coordinates and the loop iterations, with
        # the range of each (None when unknown).
        self.ranges: dict[sympy.Symbol, int | None] = {}
        for name, count in loop_counts.items():
            self.ranges[loop_symbol(name)] = count
        # A loop variable inside a statement stands for ``start + step *
        # iteration``; the variable of a loop around the body (``outer_loops``,
        # the device loop whose body this is) for the same between two of
        # its iterations, and for itself within one.
        self.loop_values: dict[str, sympy.Expr] = {}
        self.iteration_values: dict[sympy.Symbol, sympy.Expr] = {}
        # The variables of the loops inside the body's own statements.
        self.body_loops: set[str] = set()
        # The values the translation does not know, one variable each
        # (``_unknown``), by the node standing for them.
        self.unknowns: set[sympy.Symbol] = set()
        self._unknowns_by_node: dict[int, sympy.Symbol] = {}
        self.locals: dict[int, _Local] = {}
        # A statement's surroundings: no definitions of its own, every name
        # as the definitions reaching the body left it (``nest``); the
        # conjuncts of the branches around a statement read there (``bounds``).
        self.outside = _Local(ast.Pass(), renames)
        self.nest: dict[str, ast.expr | None] = {}
        self._translations: dict[
            tuple[int, int, frozenset[sympy.Expr]], sympy.Expr | None
        ] = {}
        self._proofs: dict[object, frozenset[sympy.Symbol]] = {}
        self._collect(leaves, definitions)
        # A loop around the body binds its variable for every statement of
        # the body, over any loop among the definitions (a finished sibling
        # over the same block) that defined the name before it.  A loop of
        # the body binding the variable again leaves it, for both, the plain
        # iteration of unknown range.
        for loop in outer_loops:
            if not isinstance(loop.target, ast.Name):
                continue
            name = renames.get(loop.target.id, loop.target.id)
            if name in self.body_loops:
                self.loop_values[name] = loop_symbol(name)
                self.ranges[loop_symbol(name)] = None
                continue
            self.loop_values.pop(name, None)
            self.ranges.pop(loop_symbol(name), None)
            symbol = iteration_symbol(name)
            value, count = self._model_loop(loop, _Local(loop.iter, renames), symbol)
            self.iteration_values[uniform_symbol(name)] = value
            self.ranges[symbol] = count

    # -- definitions -------------------------------------------------------

    def _collect(
        self, leaves: Sequence[ast.AST], definitions: Sequence[ast.AST]
    ) -> None:
        loops: list[tuple[ast.For, _Local]] = []
        # The definitions in program order: a later definition of a name
        # reaches the body over an earlier one, and a name assigned inside a
        # compound statement (an earlier loop's index) is unknown from there
        # on unless defined again.
        blocked: set[str] = set()
        for node in definitions:
            local = self._local(node, loops)
            simple = self._simple_definition(node, local)
            if simple is not None:
                self.nest[simple[0]] = simple[1]
                blocked.discard(simple[0])
            else:
                for name in local.definitions:
                    self.nest[name] = None
                    blocked.add(name)
        # The body's own statements: a name defined by one of them is the
        # same for every statement reading it, unless the body defines it
        # twice or a definition reaching the body defined it already (which
        # definition a given access reads is not tracked); a compound
        # statement's definitions reach its own accesses only.  A vec
        # wrapper's attached statement is among the definitions already.
        defined: set[str] = set()
        preceding = len(loops)
        for node in leaves:
            if id(node) in self.locals:
                continue
            local = self._local(node, loops)
            simple = self._simple_definition(node, local)
            if simple is not None:
                name, value = simple
                live = name in self.nest and name not in blocked
                self.nest[name] = None if live or name in defined else value
                defined.add(name)
            else:
                for name in local.definitions:
                    self.nest[name] = None
                    defined.add(name)
        self.body_defined = defined
        for position, (loop, local) in enumerate(loops):
            if isinstance(loop.target, ast.Name):
                name = self.renames.get(loop.target.id, loop.target.id)
                if position >= preceding:
                    self.body_loops.add(name)
                symbol = loop_symbol(name)
                value, count = self._model_loop(loop, local, symbol)
                if name in self.loop_values and self.loop_values[name] != value:
                    # Two loops share the variable with different bounds: the
                    # plain iteration, its range the wider one.
                    value = symbol
                    count = None
                self.loop_values[name] = value
                if symbol in self.ranges:
                    known = self.ranges[symbol]
                    count = (
                        None if known is None or count is None else max(known, count)
                    )
                self.ranges[symbol] = count

    def _local(self, node: ast.AST, loops: list[tuple[ast.For, _Local]]) -> _Local:
        local = _Local(node, self.renames)
        self.locals[id(node)] = local
        loops.extend((loop, local) for loop in local.loops)
        return local

    def _simple_definition(
        self, node: ast.AST, local: _Local
    ) -> tuple[str, ast.expr] | None:
        """``(name, value)`` of a statement defining one name from an expression."""
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or local.loops:
            return None
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            return None
        if node.value is None:
            return None
        return self.renames.get(targets[0].id, targets[0].id), node.value

    def _model_loop(
        self, loop: ast.For, local: _Local, symbol: sympy.Symbol
    ) -> tuple[sympy.Expr, int | None]:
        """A loop's variable as ``start + step * iteration``: ``(value, trip count)``.

        ``symbol`` is the iteration; the count is None unless the bounds are
        literal (a block-size constant included).  A symbolic start or step
        leaves the variable itself as the unknown.
        """
        iterator = loop.iter
        start: sympy.Expr = sympy.Integer(0)
        step: sympy.Expr = sympy.Integer(1)
        count: int | None = None
        stop: sympy.Expr | None = None
        if (
            isinstance(iterator, ast.Call)
            and ast.unparse(iterator.func) in _LOOP_CALLS
            and not iterator.keywords
            and 1 <= len(iterator.args) <= 3
        ):
            arguments = [self.translate(argument, local) for argument in iterator.args]
            if len(arguments) == 1:
                (stop,) = arguments
            elif len(arguments) == 2:
                first, stop = arguments
                start = start if first is None else first
            else:
                first, stop, third = arguments
                start = start if first is None else first
                step = step if third is None else third
        if not (start.is_Integer and step.is_Integer and int(step) > 0):
            # A symbolic start or step: the variable itself is the unknown.
            start, step = sympy.Integer(0), sympy.Integer(1)
        elif stop is not None and stop.is_Integer:
            count = max(-((int(start) - int(stop)) // int(step)), 0)
        return sympy.Add(start, sympy.Mul(step, symbol)), count

    # -- translation ---------------------------------------------------------

    def translate(
        self,
        node: ast.AST,
        statement: ast.AST | _Local,
        bounded: frozenset[sympy.Expr] = frozenset(),
    ) -> sympy.Expr | None:
        """``node``, an address term of ``statement``, over the coordinates, loop variables, uniform and unknown values.

        A part of the term whose value the translation does not know (a
        loaded value, a name defined more than once or by a call it does
        not know, an operation it does not model) is an unknown integer of
        its own (``_unknown``, among ``unknowns``).  A part whose value is
        one of the quantities ``bounded`` (the indices the conjuncts around
        the access bound, as ``bounds`` names them) stays that quantity
        rather than the sum of its parts.
        """
        local = (
            statement if isinstance(statement, _Local) else self.locals[id(statement)]
        )
        key = (id(local), id(node), bounded)
        if key not in self._translations:
            self._translations[key] = self._translate(node, local, frozenset(), bounded)
        return self._translations[key]

    def _translate(
        self,
        node: ast.AST,
        local: _Local,
        active: frozenset[str],
        bounded: frozenset[sympy.Expr],
    ) -> sympy.Expr | None:
        value = self._value(node, local, active, bounded)
        if value is None:
            return self._unknown(node)
        if bounded:
            quantity = _bounded(value)
            if quantity in bounded:
                return quantity
        return value

    def _unknown(self, node: ast.AST) -> sympy.Symbol:
        """The value of ``node``, which the translation does not know: an integer of its own.

        A loaded value, a name defined more than once or by a call the
        translation does not know, an operation it does not model.  The
        variable ranges over every integer and, like any other, differs
        between the two accesses of a pair, so the part of a term it stands
        for proves nothing by itself; the rest of the term still does (the
        lane's columns beside a gathered row, a per-thread offset added to a
        loaded base times the block size).
        """
        key = id(node)
        if key not in self._unknowns_by_node:
            symbol = sympy.Symbol(
                f"_helion_unknown_{len(self._unknowns_by_node)}", integer=True
            )
            self._unknowns_by_node[key] = symbol
            self.unknowns.add(symbol)
            self.ranges[symbol] = None
        return self._unknowns_by_node[key]

    def _value(
        self,
        node: ast.AST,
        local: _Local,
        active: frozenset[str],
        bounded: frozenset[sympy.Expr],
    ) -> sympy.Expr | None:
        """``node`` over the coordinates, loop variables, uniform and unknown values; None when its value is unknown."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                return None
            return sympy.Integer(node.value)
        if isinstance(node, ast.Name):
            return self._name(node.id, local, active, bounded)
        if isinstance(node, ast.UnaryOp):
            operand = self._translate(node.operand, local, active, bounded)
            if operand is None:
                return None
            if isinstance(node.op, ast.USub):
                return sympy.Mul(-1, operand)
            return operand if isinstance(node.op, ast.UAdd) else None
        if isinstance(node, ast.BinOp):
            left = self._translate(node.left, local, active, bounded)
            right = self._translate(node.right, local, active, bounded)
            if left is None or right is None:
                return None
            return self._arithmetic(node.op, left, right)
        if isinstance(node, ast.Call):
            return self._call(node, local, active, bounded)
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == _THREAD_INDEX
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                axis = node.slice.value
                symbol = thread_symbol(axis)
                self.ranges.setdefault(symbol, self.axis_sizes.get(axis))
                return symbol
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) in _UNIFORM_CALLS
                and isinstance(node.slice, ast.Constant)
            ):
                return uniform_symbol(ast.unparse(node))
            return None
        return None

    def _arithmetic(
        self, op: ast.AST, left: sympy.Expr, right: sympy.Expr
    ) -> sympy.Expr | None:
        if isinstance(op, ast.Add):
            return sympy.Add(left, right)
        if isinstance(op, ast.Sub):
            return sympy.Add(left, sympy.Mul(-1, right))
        if isinstance(op, ast.Mult):
            return sympy.Mul(left, right)
        if isinstance(op, (ast.FloorDiv, ast.Mod)):
            if left.is_Integer and right.is_Integer and right != 0:
                return sympy.Integer(
                    int(left) // int(right)
                    if isinstance(op, ast.FloorDiv)
                    else int(left) % int(right)
                )
            function = _FLOOR_DIVIDE if isinstance(op, ast.FloorDiv) else _MODULO
            return cast("sympy.Expr", function(sympy.expand(left), sympy.expand(right)))
        return None

    def _call(
        self,
        node: ast.Call,
        local: _Local,
        active: frozenset[str],
        bounded: frozenset[sympy.Expr],
    ) -> sympy.Expr | None:
        function = ast.unparse(node.func)
        if node.keywords:
            return None
        if (_INTEGER_CAST.match(function) or function == "int") and len(node.args) == 1:
            return self._translate(node.args[0], local, active, bounded)
        if function in _OPERATORS:
            op = _OPERATORS[function]()
            operands = [
                self._translate(argument, local, active, bounded)
                for argument in node.args
            ]
            if any(operand is None for operand in operands):
                return None
            if isinstance(op, ast.unaryop) and len(operands) == 1:
                return self._translate(
                    ast.UnaryOp(op=op, operand=node.args[0]), local, active, bounded
                )
            if isinstance(op, ast.operator) and len(operands) == 2:
                left, right = operands
                assert left is not None and right is not None
                return self._arithmetic(op, left, right)
            return None
        if function in _UNIFORM_CALLS:
            return uniform_symbol(ast.unparse(node))
        return None

    def _name(
        self,
        name: str,
        local: _Local,
        active: frozenset[str],
        bounded: frozenset[sympy.Expr],
    ) -> sympy.Expr | None:
        name = self.renames.get(name, name)
        if name in self.loop_values:
            return self.loop_values[name]
        symbol = loop_symbol(name)
        if symbol in self.ranges:
            # A lane loop's or V-loop's variable.
            return symbol
        uniform = uniform_symbol(name)
        if uniform in self.iteration_values:
            # The variable of a loop around the body: one value within an
            # iteration, whatever defined the name before the loop.
            return uniform
        if name in active:
            return None
        if name in self.loaded:
            # A loaded value: an unknown, unless every thread and lane loads
            # the same element before the body runs.
            return uniform_symbol(name) if self.uniform_load(name) else None
        if name in local.definitions:
            definition, scope = local.definitions[name], local
        elif name in self.nest:
            # Read as the definitions reaching the body left it, over their
            # own scope: a name the statement defines again is another
            # value inside it, of which the outer definition knows nothing.
            definition, scope = self.nest[name], self.outside
        elif name in self.constants:
            return self.constants[name]
        else:
            return uniform_symbol(name)
        if definition is None:
            return None
        return self._translate(definition, scope, active | {name}, bounded)

    def uniform_load(self, name: str) -> bool:
        """Whether ``name`` is defined around the body by a load every thread and lane reads alike.

        The definition is one of the definitions around the body, not one
        of the body's own nor one the body redefines: a plain load, masked
        or not (``loads``), of a kernel argument (a name nothing in view
        defines: not a tensor made in view, whose elements some thread
        fills, nor a thread's register array, each thread's own) that no
        statement in view writes (``written``), whose index terms, mask and
        fill are over uniform values and constants alone (no thread
        coordinate, no lane variable, no loaded value but another such
        load's).  Every thread then loads the same element before the body
        runs (a segment's start by the block's index), and the value is one
        uniform integer for the body: ``uniform_symbol(name)`` in the
        terms, rather than an unknown differing between the two accesses of
        a pair.
        """
        if name not in self._uniform:
            # Fail closed while deciding: a load indexed by itself is none.
            self._uniform[name] = False
            load = self.loads.get(name)
            local = None if load is None else self.locals.get(id(load.statement))
            if (
                load is not None
                and local is not None
                and self.nest.get(name) is load.value
                and name not in self.body_defined
                and load.tensor not in self.nest
                and load.tensor not in self.written
            ):
                self._uniform[name] = all(
                    self._uniform_value(node, local, frozenset({name}))
                    for node in (*load.terms, *load.guards)
                )
        return self._uniform[name]

    def _uniform_value(
        self, node: ast.AST, local: _Local, active: frozenset[str]
    ) -> bool:
        """Whether ``node``, a term or a condition, is over uniform values and constants alone."""
        if isinstance(node, ast.BoolOp):
            return all(
                self._uniform_value(value, local, active) for value in node.values
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._uniform_value(node.operand, local, active)
        if isinstance(node, ast.Compare):
            return all(
                self._uniform_value(operand, local, active)
                for operand in (node.left, *node.comparators)
            )
        value = self._value(node, local, active, frozenset())
        return value is not None and all(
            isinstance(symbol, sympy.Symbol) and symbol.name.startswith(_UNIFORM_PREFIX)
            for symbol in value.free_symbols
        )

    def packet(self, term: sympy.Expr | None, variable: str) -> sympy.Expr | None:
        """``term``, the address of a packet's first element along its dimension, moved along its elements.

        The packet holds one element per iteration of the V-loop
        ``variable``, so the term of any of them is the first's plus the
        loop's variable.  None when ``term`` is unknown or the V-loop is no
        variable of the proof (a packet of unknown extent).
        """
        symbol = loop_symbol(variable)
        if term is None or symbol not in self.ranges:
            return None
        return sympy.Add(term, symbol)

    # -- bounds ----------------------------------------------------------------

    def bounds(
        self,
        statement: ast.AST,
        around: Iterable[str],
        inside: Iterable[str] = (),
    ) -> dict[sympy.Expr, Bound]:
        """The quantities the branch conditions ``around`` ``statement`` and ``inside`` it bound, with their bounds.

        A thread coordinate compared with a literal (``tid < c``, the
        leader's ``tid == 0``), by its symbol; an index compared with a
        literal (a tile mask's ``indices < end``, with the end a literal)
        whose value is a sum the proof knows to be nonnegative, by the
        quantity ``_BOUNDED(value)`` that ``translate`` keeps whole for an
        access under the conjunct: the padded elements past the tile's end
        do not count for the access (``n * row + column`` with ``column =
        64 * iteration + thread < n`` is one row's), while the parts of a
        quantity forced equal are forced equal in turn (``_prove``).  An
        index compared with a uniform value (the end a kernel argument: a
        dynamic shape's size), by the same quantity with that value for
        its bound: the terms keep the index as the sum of its parts
        (``whole_quantities``), and the argument takes the value as a
        radix (``_forced_equal``).

        A conjunct ``around`` the statement (of a branch enclosing it) is
        read before the statement runs, over the definitions reaching the
        body: a name the statement defines for itself (a device loop's
        per-iteration index) is another value there, of which the conjunct
        says nothing.  A conjunct ``inside`` it (of a branch around one of
        its own statements) reads the statement's own definitions, and a
        name among them the statement does not define as the definitions
        reaching the body left it, over their own scope: a mask the nest
        defines over an index the statement redefines bounds the outer
        index, not the statement's.
        """
        local = self.locals[id(statement)]
        result: dict[sympy.Expr, Bound] = {}
        for conjuncts, scope in ((around, self.outside), (inside, local)):
            for conjunct in conjuncts:
                expression = ast.parse(conjunct, mode="eval").body
                for symbol, bound in self._bounds(
                    expression, scope, frozenset()
                ).items():
                    result[symbol] = _tighter(result.get(symbol, bound), bound)
        return result

    def _bounds(
        self, node: ast.expr, local: _Local, active: frozenset[str]
    ) -> dict[sympy.Expr, Bound]:
        if isinstance(node, ast.Name):
            name = self.renames.get(node.id, node.id)
            if name in active:
                return {}
            # A name the statement does not define is read as the
            # definitions reaching the body left it, over their own scope:
            # a mask ``indices < n`` defined around a loop that redefines
            # ``indices`` bounds the outer value, not the loop's.
            if name in local.definitions:
                definition, scope = local.definitions[name], local
            else:
                definition, scope = self.nest.get(name), self.outside
            if definition is None:
                return {}
            return self._bounds(definition, scope, active | {name})
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            result: dict[sympy.Expr, Bound] = {}
            for value in node.values:
                for symbol, bound in self._bounds(value, local, active).items():
                    result[symbol] = _tighter(result.get(symbol, bound), bound)
            return result
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
            return {}
        left = self._translate(node.left, local, active, frozenset())
        right = self._translate(node.comparators[0], local, active, frozenset())
        if left is None or right is None:
            return {}
        (op,) = node.ops
        if left.is_Integer and not right.is_Integer:
            # ``c > t`` is ``t < c``.
            left, right = right, left
            if isinstance(op, ast.Gt):
                op = ast.Lt()
            elif isinstance(op, ast.GtE):
                op = ast.LtE()
            elif isinstance(op, (ast.Lt, ast.LtE)):
                return {}
        radix: sympy.Symbol | None = None
        if not right.is_Integer:
            # A uniform value for the end (a kernel argument: a dynamic
            # shape's size, a segment's end every thread loads alike)
            # bounds an index below it; a coordinate is left to its axis.
            if (
                not isinstance(right, sympy.Symbol)
                or not right.name.startswith(_UNIFORM_PREFIX)
                or not isinstance(op, ast.Lt)
            ):
                return {}
            radix = right
        elif (
            isinstance(left, sympy.Symbol)
            and left in self.ranges
            and left.name.startswith("_helion_thread_")
        ):
            if isinstance(op, ast.Lt):
                bound = int(right)
            elif isinstance(op, ast.LtE):
                bound = int(right) + 1
            elif isinstance(op, ast.Eq) and int(right) == 0:
                # The leader's coordinate: one value, the range's first.  A
                # coordinate equal to another value is not the range
                # ``[0, 1)`` (two accesses under ``tid == 3`` and ``tid ==
                # 0`` would pass as one thread's), so it is left to the
                # launch axis.
                bound = 1
            else:
                return {}
            return {left: bound} if bound >= 1 else {}
        if (
            isinstance(left, sympy.Symbol)
            or left.is_Integer
            or left.free_symbols & self.unknowns
            or left.xreplace(self.iteration_values).is_nonnegative is not True  # pyrefly: ignore[missing-attribute]
        ):
            # A lone variable is bounded by its range; a value the proof
            # does not know, or one it cannot tell from a negative (a
            # uniform offset among its parts), is left to the parts.
            return {}
        if radix is not None:
            return {_bounded(left): radix}
        if isinstance(op, ast.Lt):
            bound = int(right)
        elif isinstance(op, ast.LtE):
            bound = int(right) + 1
        else:
            return {}
        return {_bounded(left): bound} if bound >= 1 else {}

    def pair_ranges(
        self, first: Mapping[sympy.Expr, Bound], second: Mapping[sympy.Expr, Bound]
    ) -> dict[sympy.Expr, Bound | None]:
        """The variables' ranges for a pair of accesses bounded by ``first`` and ``second``.

        A coordinate both accesses bound (a branch condition enclosing each)
        ranges over the wider of the two bounds; one bound on one side only
        ranges over the launch axis.  A bounded index both accesses run
        under ranges over the wider bound too; one bounded on one side only
        is left out, and the proof takes it as the sum of its parts.  An
        index both accesses bound by one uniform value keeps that value
        for its bound (the argument's radix); bounded by a literal on one
        side and a uniform value on the other, or by two uniform values,
        it is left out.
        """
        ranges = cast("dict[sympy.Expr, Bound | None]", dict(self.ranges))
        for symbol, bound in first.items():
            if symbol not in second:
                continue
            other = second[symbol]
            if isinstance(bound, int) and isinstance(other, int):
                size = ranges.get(symbol)
                widest = max(bound, other)
                ranges[symbol] = min(size, widest) if isinstance(size, int) else widest
            elif bound == other:
                ranges[symbol] = bound
        return ranges

    # -- proofs ----------------------------------------------------------------

    def same_thread(
        self,
        first: Sequence[sympy.Expr | None],
        second: Sequence[sympy.Expr | None],
        *,
        ranges: Mapping[sympy.Expr, Bound | None],
        varying: Iterable[str] = (),
        exact: bool = True,
    ) -> frozenset[sympy.Symbol]:
        """The variables the equations ``first[d] == second[d]`` force equal for the two accesses.

        ``first`` and ``second`` are the terms of one access each, per
        dimension (None for a term left out); ``varying`` names the
        uniform values that differ between the two accesses (a device loop's
        variable between two of its iterations: ``start + step * iteration``
        when the loop is among the outer loops, else an unknown other
        value).  A coordinate among the result can only be the same thread's
        in both accesses; a loop variable among it the same iteration's.  Two
        accesses whose elements the terms model exactly (``exact``: one
        element each, or a packet's along its V-loop's variable) never meet
        when their terms can never be equal, and every variable is returned;
        an access standing in for more elements than its terms name decides
        nothing when the equations are unsolvable.
        """
        differing = frozenset(uniform_symbol(name) for name in varying)
        key = (
            tuple(first),
            tuple(second),
            tuple(sorted(ranges.items(), key=str)),
            differing,
            exact,
        )
        if key not in self._proofs:
            self._proofs[key] = self._prove(first, second, ranges, differing, exact)
        return self._proofs[key]

    def _prove(
        self,
        first: Sequence[sympy.Expr | None],
        second: Sequence[sympy.Expr | None],
        ranges: Mapping[sympy.Expr, Bound | None],
        differing: frozenset[sympy.Symbol],
        exact: bool,
    ) -> frozenset[sympy.Symbol]:
        variables = set(self.ranges)
        modelled = {
            symbol: self.iteration_values[symbol]
            for symbol in differing
            if symbol in self.iteration_values
        }
        substitution = {
            symbol: _other_side(symbol)
            for symbol in variables | (differing - set(modelled))
        }
        # A bounded index (``bounds``) is one quantity of the argument when
        # both accesses run under its bound (``pair_ranges``) and both
        # terms name it (a packet's term is its base's plus the V-loop's
        # variable: the index's parts), else the sum of its parts.
        sides = [
            {atom for term in side if term is not None for atom in term.atoms(_BOUNDED)}
            for side in (first, second)
        ]
        quantities = sides[0] | sides[1]
        shared = sides[0] & sides[1]
        unwrap = {
            atom: atom.args[0]
            for atom in quantities
            if ranges.get(atom) is None or atom not in shared
        }
        # An index bounded by a uniform value is never among the terms'
        # quantities: the symbolic radix step matches its value against
        # the terms as modelled (``_forced_equal``).
        ranges = {
            **{
                (
                    quantity.xreplace(modelled)
                    if isinstance(bound, sympy.Symbol)
                    else quantity
                ): bound
                for quantity, bound in ranges.items()
            },
            **{
                atom.xreplace(modelled): ranges[atom]
                for atom in quantities
                if atom not in unwrap
            },
        }
        # (a term of the first access, the same term of the second, the
        # modulus the equation holds under; zero for equality).
        equations: list[_Equation] = [
            (
                a.xreplace(unwrap).xreplace(modelled),
                b.xreplace(unwrap).xreplace(modelled),
                0,
            )
            for a, b in zip(first, second, strict=True)
            if a is not None and b is not None
        ]
        known = set(equations)
        proven: set[sympy.Symbol] = set()
        while equations:
            a, b, modulus = equations.pop()
            derived: list[_Equation] = []
            if a.func is _BOUNDED and b.func is _BOUNDED:
                # Two bounded indices equal: their values are.
                derived.append(
                    (
                        cast("sympy.Expr", a.args[0]),
                        cast("sympy.Expr", b.args[0]),
                        modulus,
                    )
                )
            if (
                a.func in (_FLOOR_DIVIDE, _MODULO)
                and b.func is a.func
                and a.args[1] == b.args[1]
                and not a.args[1].free_symbols & set(substitution)
            ):
                dividend = cast("sympy.Expr", a.args[0])
                other_dividend = cast("sympy.Expr", b.args[0])
                divisor = cast("sympy.Expr", a.args[1])
                # ``p % m == q % m`` and ``p // m == q // m`` together give
                # ``p == q``.
                partner = _MODULO if a.func is _FLOOR_DIVIDE else _FLOOR_DIVIDE
                if (
                    modulus == 0
                    and (
                        partner(dividend, divisor),
                        partner(other_dividend, divisor),
                        0,
                    )
                    in known
                ):
                    derived.append((dividend, other_dividend, 0))
                # ``p % m == q % m`` alone gives ``p == q (mod m)``.
                if (
                    modulus == 0
                    and a.func is _MODULO
                    and divisor.is_Integer
                    and int(divisor) > 0
                ):
                    derived.append((dividend, other_dividend, int(divisor)))
            outcome = self._forced_equal(a, b, substitution, ranges, variables, modulus)
            if outcome is None:
                # The two terms are never equal: the accesses never meet,
                # when the terms model all of their elements.
                if exact:
                    return frozenset(variables)
                continue
            equal, atoms = outcome
            proven |= equal
            derived.extend((atom, atom, 0) for atom in atoms)
            for equation in derived:
                if equation not in known:
                    known.add(equation)
                    equations.append(equation)
        return frozenset(proven)

    @staticmethod
    def _forced_equal(
        first: sympy.Expr,
        second: sympy.Expr,
        substitution: Mapping[sympy.Symbol, sympy.Symbol],
        ranges: Mapping[sympy.Expr, Bound | None],
        variables: set[sympy.Symbol],
        modulus: int,
    ) -> tuple[set[sympy.Symbol], list[sympy.Expr]] | None:
        """What ``first == second`` (modulo ``modulus`` when nonzero) forces equal between the two accesses.

        The variables forced equal, and the other quantities of both sides
        (a ``p % m`` of a swizzled address, a bounded index) forced equal,
        for the caller to take further; None when the equation has no
        solution (the two accesses never meet).  The mixed-radix argument:
        the difference of the two sides is a sum of terms ``c_i * y_i``
        over bounded integer quantities (a variable's difference between
        the accesses, a variable, a ``p % m`` or a bounded index of one
        access, the constant), which in the order of their coefficients
        fall into groups summing to zero: a group closes when its sum lies
        strictly between the negated and the positive gcd of the later
        coefficients and the modulus, and inside a closed group a term is
        zero when the sum of the terms before it is smaller than its
        coefficient.

        A uniform value among the factors of some terms (a dynamic shape's
        size multiplying the row) is a radix of its own when the rest of
        the difference is the difference of one index the conjuncts
        around both accesses bound by that value (``bounds``,
        ``_symbolic_radix``): ``R * quotient == q' - q`` with ``0 <= q,
        q' < R`` forces the quotient to zero and the indices equal,
        whatever value ``R`` has, and the quotient is an equation of its
        own.
        """
        difference = sympy.expand(
            sympy.Add(first, sympy.Mul(-1, second.xreplace(substitution)))
        )
        if not modulus:
            split = AddressMaps._symbolic_radix(difference, substitution, ranges)
            if split is not None:
                quotient, quantity = split
                outcome = AddressMaps._forced_equal(
                    quotient, sympy.Integer(0), substitution, ranges, variables, 0
                )
                if outcome is None:
                    return None
                equal, atoms = outcome
                if quantity is not None:
                    atoms.append(quantity)
                return equal, atoms
        coefficients: dict[sympy.Expr, int] = {}
        constant = 0
        for term in sympy.Add.make_args(difference):
            coefficient, rest = cast("sympy.Expr", term).as_coeff_Mul()
            if not coefficient.is_Integer:
                # A symbolic coefficient (a block size times a variable).
                return set(), []
            if rest == 1:
                constant += int(coefficient)
                continue
            coefficients[rest] = coefficients.get(rest, 0) + int(coefficient)
        if modulus:
            constant %= modulus
        marked = set(substitution) | set(substitution.values())
        unmarked = {other: symbol for symbol, other in substitution.items()}
        # (signed coefficient, the quantity's low and high, the quantity as
        # the first access has it; None for the constant)
        terms: list[tuple[int, float, float, sympy.Expr | None]] = []
        both_sides: set[sympy.Expr] = set()
        seen: set[sympy.Expr] = set()
        for atom, coefficient in coefficients.items():
            if coefficient == 0 or atom in seen:
                continue
            if not atom.free_symbols & marked:
                # A uniform value left over: the two accesses read
                # different ones.
                return set(), []
            plain = atom.xreplace(unmarked)
            other = plain.xreplace(substitution)
            seen |= {plain, other}
            here = coefficients.get(plain, 0)
            there = coefficients.get(other, 0)
            bound = (
                ranges.get(plain)
                if isinstance(plain, sympy.Symbol) or plain.func is _BOUNDED
                else None
            )
            if not isinstance(bound, int):
                # A bound by a uniform value is no range for the integer
                # argument (the symbolic radix step's business).
                bound = None
            if (
                bound is None
                and plain.func is _MODULO
                and plain.args[1].is_Integer
                and int(plain.args[1]) > 0  # pyrefly: ignore [bad-argument-type]
            ):
                bound = int(plain.args[1])  # pyrefly: ignore [bad-argument-type]
            if here and there:
                both_sides.add(plain)
            if bound == 1:
                # The same value in both accesses.
                continue
            high = math.inf if bound is None else bound - 1
            if here and here == -there:
                # The difference of the two accesses' values.
                terms.append((abs(here), -high, high, plain))
            else:
                terms.extend((c, 0, high, plain) for c in (here, there) if c)
        if constant:
            terms.append((constant, 1, 1, None))
        terms.sort(key=lambda term: (abs(term[0]), term[2]))
        zero: set[sympy.Expr] = set()
        unresolved: set[sympy.Expr] = set()
        group: list[tuple[int, float, float, sympy.Expr | None]] = []
        low = high = 0.0
        for index, (coefficient, y_low, y_high, quantity) in enumerate(terms):
            a, b = coefficient * y_low, coefficient * y_high
            group.append((coefficient, min(a, b), max(a, b), quantity))
            low += min(a, b)
            high += max(a, b)
            later = [abs(term[0]) for term in terms[index + 1 :]]
            divisor = math.gcd(*later, modulus) if later or modulus else 0
            if divisor and not (low > -divisor and high < divisor):
                continue
            # The group's sum is a multiple of ``divisor`` strictly inside
            # (-divisor, divisor): zero.
            if low > 0 or high < 0:
                return None
            before_low = before_high = 0.0
            prefixes: list[tuple[float, float]] = []
            for _c, term_low, term_high, _q in group:
                prefixes.append((before_low, before_high))
                before_low += term_low
                before_high += term_high
            for position in reversed(range(len(group))):
                c, _low, _high, member = group[position]
                before_low, before_high = prefixes[position]
                if not (before_low > -abs(c) and before_high < abs(c)):
                    unresolved.update(
                        item[3] for item in group[: position + 1] if item[3] is not None
                    )
                    break
                # ``c * y`` equals the negated sum before it, smaller than
                # ``c``: the term is zero (the constant cannot be).
                if member is None:
                    return None
                zero.add(member)
            group = []
            low = high = 0.0
        unresolved.update(item[3] for item in group if item[3] is not None)
        forced = {
            quantity
            for quantity in both_sides
            if quantity not in unresolved
            and (quantity in zero or all(term[3] != quantity for term in terms))
        }
        equal = {q for q in forced if isinstance(q, sympy.Symbol)}
        atoms = [q for q in forced if not isinstance(q, sympy.Symbol)]
        return equal, atoms

    @staticmethod
    def _symbolic_radix(
        difference: sympy.Expr,
        substitution: Mapping[sympy.Symbol, sympy.Symbol],
        ranges: Mapping[sympy.Expr, Bound | None],
    ) -> tuple[sympy.Expr, sympy.Expr | None] | None:
        """``difference`` as ``R * quotient + q - q'`` for a uniform value ``R`` and an index ``q`` bounded by it: ``(quotient, the index)``.

        The terms with the uniform value among their factors make up ``R *
        quotient``; the rest must be the difference of one index the
        conjuncts around both accesses bound by ``R`` (``bounds``: ``0 <=
        q < R`` for each access), or that index alone when one access
        reads it (forced zero, so the index returned is None).  None when
        the difference has another form: no such terms, two uniform
        values among the factors, a uniform value differing between the
        accesses, a constant or a part of the rest the bound does not
        cover.
        """
        unmarked = {other: symbol for symbol, other in substitution.items()}
        quotients: dict[sympy.Symbol, list[sympy.Expr]] = {}
        remainder: list[sympy.Expr] = []
        for term in sympy.Add.make_args(difference):
            coefficient, rest = cast("sympy.Expr", term).as_coeff_Mul()
            if not coefficient.is_Integer:
                return None
            factors = sympy.Mul.make_args(rest)
            uniform = [
                factor
                for factor in factors
                if isinstance(factor, sympy.Symbol)
                and factor.name.startswith(_UNIFORM_PREFIX)
            ]
            if not uniform:
                remainder.append(cast("sympy.Expr", term))
            elif len(uniform) == 1:
                (radix,) = uniform
                quotients.setdefault(radix, []).append(
                    sympy.Mul(coefficient, *[f for f in factors if f != radix])
                )
            else:
                return None
        if len(quotients) != 1 or not remainder:
            return None
        ((radix, parts),) = quotients.items()
        if radix in substitution or radix in unmarked:
            return None
        rest = sympy.Add(*remainder)
        for quantity, bound in ranges.items():
            if bound != radix or quantity.func is not _BOUNDED:
                continue
            value = cast("sympy.Expr", quantity.args[0])
            other = value.xreplace(substitution)
            negated = sympy.expand(sympy.Mul(-1, value))
            negated_other = sympy.expand(sympy.Mul(-1, other))
            if rest in (
                sympy.expand(sympy.Add(value, negated_other)),
                sympy.expand(sympy.Add(other, negated)),
            ):
                return sympy.Add(*parts), quantity
            if rest in (value, negated, other, negated_other):
                return sympy.Add(*parts), None
        return None
