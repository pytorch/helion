"""Divide a vector's base index instead of each of its lanes.

A per-channel parameter of an NCHW tensor read through a flat tile,
``w[(tile.index // (H * W)) % C]``, lowers inside the unrolled vector loop
to one ``(lane_base + vec_lane) // (H * W)`` per lane, so each of the V
lanes repeats the index math and every gather.  When ``lane_base`` is a
multiple of V and the divisor is too, all V lanes have the same quotient:
``lane_base`` sits ``0 mod V`` below a multiple of the divisor, so adding a
lane offset below V never crosses one (floor division, any sign).  The pass
rewrites such a quotient to ``lane_base // divisor``; the V unrolled copies
then compute one value and load one parameter, which LLVM's CSE merges.  A
Python remainder by such a divisor (``_cute_python_mod``, the ``e % d`` of
a de-linearized ``x[e // d, e % d]``) becomes the base's remainder plus the
lane: the base's is a multiple of V below the divisor.

Under ``cute_uniform_lane_compares`` (``compare_bases``), a comparison of a
lane offset with an integer bound that is a multiple of V (``offset <
bound`` or ``offset >= bound``, either way round) compares the base
instead: no lane offset reaches the next multiple of V.  A chunk's tile
mask ``index < extent`` and the bounds of its de-linearized coordinates
then cost one compare per vector instead of one per lane: a Python
remainder by a positive constant also has a known factor there (the gcd
of its dividend's and the divisor), so the in-row bound ``r % d < d``
compares the base's remainder.  The remainder gets it on its own, under
integer casts or through a name bound to it, not inside a larger sum or
product (``row * 192 + (r % 192)`` has no known factor).  Without the knob
a remainder has factor 1, as before, and the code is the knob-off code.

Facts are proved syntactically on the generated kernel body.  A name's known
factor follows integer constants, constexpr values, sums (gcd), products and
integer casts of integer expressions (``scalar_integer``), and a staged
``range`` loop's counter has the gcd of its start and step; anything else
has factor 1.  V is a power of two, so a factor of V survives wrapping
arithmetic and narrowing casts, and ``lane_base + vec_lane`` cannot wrap.
Only literal ``range_constexpr`` vector loops that leave their lane alone
are recognized, with lane offsets spelled ``base + lane`` or ``base +
cutlass.Int32(lane)`` (the cast form only for a traced base, whose type the
sum keeps), and sums of a lane offset and an integer multiple of V
(``row * d + offset``, whose base is ``row * d + base``).  The offset's
name is replaced by its base under its casts, so the quotient, remainder or
comparison keeps its type.  Facts follow straight-line code: a loop forgets
the names it assigns before its body, except the integers it carries in and
preserves (the next iteration's lane base of a software-pipelined loop),
whose facts are weakened over passes over the body (a vector loop's on a
copy) until they hold at every iteration; an ``if`` forgets the names
either branch assigns, and a body with statements
``simplify_proven_bounds`` cannot model is left alone.

The rewrites are integer identities: every integer and boolean keeps its
value.  The comparisons' only numeric effect is FMA contraction where
per-lane selects kept products unfused: once a whole mask is one value per
vector, LLVM may merge the lanes' masked branches and contract a product
the mask guarded into the add that reads it, where per-lane masks keep the
two in separate basic blocks.  Those elements can differ by about one ulp,
as under ``cute_lane_unroll``, so the comparison rewrite is a knob, off by
default.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING

from ._ast_pass_utils import call_path
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import is_staged_loop
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import merged_names
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import reparse_expr
from ._ast_pass_utils import with_merged_names
from .scalar_integer import _INTEGER_CASTS
from .scalar_integer import _integer_expression
from .simplify_proven_bounds import _python_mod_operands
from .simplify_proven_bounds import _supported_body

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

# A constant this size reads the same through every integer cast.
_INT32_MAX = (1 << 31) - 1
# The operand of a comparison, operator or call, that a lane offset may hold
# for the comparison to keep its answer once the offset becomes its base:
# the first in ``offset < bound`` and ``offset >= bound``, the second in
# ``bound > offset`` and ``bound <= offset``.
_OFFSET_OPERAND: tuple[tuple[type[ast.cmpop], int], ...] = (
    (ast.Lt, 0),
    (ast.GtE, 0),
    (ast.Gt, 1),
    (ast.LtE, 1),
)
_COMPARISON_CALL_OPERAND: dict[str, int] = {
    "operator.lt": 0,
    "operator.ge": 0,
    "operator.gt": 1,
    "operator.le": 1,
}


def _cast_argument(node: ast.expr) -> ast.expr | None:
    """``argument`` of an integer cast ``cutlass.Int32(argument)`` (or Int64,
    Uint32, Uint64), else None."""
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) in _INTEGER_CASTS
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]
    return None


def _strip_casts(node: ast.expr) -> ast.expr:
    while (argument := _cast_argument(node)) is not None:
        node = argument
    return node


def _is_python_mod(node: ast.expr) -> bool:
    return _python_mod_operands(node) is not None


@dataclasses.dataclass
class _Facts:
    """What the straight-line code so far proves about names: which hold
    integers, which of those are traced DSL values, their known factors and
    their trace-time constant values."""

    integer: set[str]
    traced: set[str]
    factor: dict[str, int]
    constant: dict[str, int]
    # Each name ``ast_rename`` merges with others (a loop-carried value and
    # its aliases), with all of them: once renamed they are one variable, so
    # a write to any is a write to all.
    merged: Mapping[str, frozenset[str]]
    # ``cute_uniform_lane_compares`` (``compare_bases``): a Python remainder
    # by a positive constant is an integer with a known factor, and a
    # comparison of a lane offset with a multiple of the width compares the
    # offset's base.
    uniform_compares: bool

    def copy(self) -> _Facts:
        return _Facts(
            set(self.integer),
            set(self.traced),
            dict(self.factor),
            dict(self.constant),
            self.merged,
            self.uniform_compares,
        )

    def empty_like(self) -> _Facts:
        """No facts about any name, under this one's merged names and
        comparison mode."""
        return _Facts(set(), set(), {}, {}, self.merged, self.uniform_compares)

    def with_merged(self, names: Iterable[str]) -> set[str]:
        return with_merged_names(names, self.merged)

    def is_integer(self, node: ast.expr) -> bool:
        remainder = _strip_casts(node)
        if self.uniform_compares and _is_python_mod(remainder):
            assert isinstance(remainder, ast.Call)
            return all(self.is_integer(argument) for argument in remainder.args)
        return _integer_expression(node, "", self.integer)

    def is_traced(self, node: ast.expr) -> bool:
        """Whether the integer expression ``node`` is a traced DSL value: it
        reads a launch coordinate, an integer cast or a traced name."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in self.traced:
                return True
            if isinstance(sub, ast.Call) and (
                _cast_argument(sub) is not None
                or (call_path(sub.func) or ())[:2] == ("cute", "arch")
            ):
                return True
        return False

    def constant_of(self, node: ast.expr) -> int | None:
        stripped = _strip_casts(node)
        if isinstance(stripped, ast.Constant) and type(stripped.value) is int:
            value = stripped.value
        elif isinstance(stripped, ast.Name):
            value = self.constant.get(stripped.id)
        else:
            return None
        if value is None or (stripped is not node and not 0 <= value <= _INT32_MAX):
            # A cast may wrap a value outside the range all casts share.
            return None
        return value

    def factor_of(self, node: ast.expr) -> int:
        """A positive integer dividing every value ``node`` takes (1 when
        nothing is known or ``node`` may not be an integer; 0 is the factor
        of the constant 0)."""
        if not self.is_integer(node):
            return 1
        return self._factor(node)

    def _factor(self, node: ast.expr) -> int:
        value = self.constant_of(node)
        if value is not None:
            return abs(value)
        node = _strip_casts(node)
        if isinstance(node, ast.Name):
            return self.factor.get(node.id, 1)
        if self.uniform_compares and _is_python_mod(node):
            assert isinstance(node, ast.Call)
            # ``a - d * floor(a / d)`` for a positive divisor ``d``.
            divisor = self.constant_of(node.args[1])
            if divisor is None or divisor <= 0:
                return 1
            return math.gcd(self._factor(node.args[0]), divisor)
        if isinstance(node, ast.BinOp):
            left = self._factor(node.left)
            right = self._factor(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return math.gcd(left, right)
            if isinstance(node.op, ast.Mult):
                return left * right
        return 1

    def bind(self, name: str, value: ast.expr) -> None:
        integer = self.is_integer(value)
        traced = integer and self.is_traced(value)
        constant = self.constant_of(value) if integer else None
        factor = self.factor_of(value)
        self.forget((name,))
        if integer:
            self.integer.add(name)
        if traced:
            self.traced.add(name)
        if constant is not None:
            self.constant[name] = constant
        if factor != 1:
            self.factor[name] = factor

    def forget(self, names: Iterable[str]) -> None:
        for name in self.with_merged(names):
            self.integer.discard(name)
            self.traced.discard(name)
            self.factor.pop(name, None)
            self.constant.pop(name, None)

    def restricted(self, names: AbstractSet[str]) -> _Facts:
        return _Facts(
            self.integer & names,
            self.traced & names,
            {name: factor for name, factor in self.factor.items() if name in names},
            {name: value for name, value in self.constant.items() if name in names},
            self.merged,
            self.uniform_compares,
        )

    def meet(self, other: _Facts) -> _Facts:
        """What both hold: common integers with the gcd of their factors."""
        integer = self.integer & other.integer
        factor = {
            name: math.gcd(self.factor.get(name, 1), other.factor.get(name, 1))
            for name in integer
        }
        return _Facts(
            integer,
            self.traced & other.traced & integer,
            {name: value for name, value in factor.items() if value != 1},
            {
                name: value
                for name, value in self.constant.items()
                if name in integer and other.constant.get(name) == value
            },
            self.merged,
            self.uniform_compares,
        )

    def update(self, other: _Facts) -> None:
        self.integer |= other.integer
        self.traced |= other.traced
        self.factor.update(other.factor)
        self.constant.update(other.constant)

    def key(self) -> tuple[frozenset[object], ...]:
        """The facts as a hashable value (within one pass, whose merged
        names and ``uniform_compares`` are fixed)."""
        return (
            frozenset(self.integer),
            frozenset(self.traced),
            frozenset(self.factor.items()),
            frozenset(self.constant.items()),
        )


# Passes over a loop body before giving up on the facts it carries.
_CARRY_ROUNDS = 4


class _Rewriter:
    def __init__(
        self,
        facts: _Facts,
        *,
        in_trial: bool = False,
        carried_cache: dict[tuple[object, ...], _Facts] | None = None,
    ) -> None:
        self.facts = facts
        # Set on the rewriters of ``carried_facts``'s passes over a copy of a
        # loop body, which need only the facts it leaves: a loop nested there
        # takes the facts it carries without a pass over its own body.
        self.in_trial = in_trial
        # The facts ``carried_facts`` found, by loop and facts at its entry,
        # shared by every rewriter of one pass: the passes over an outer
        # loop's body meet each nested loop again, which would otherwise
        # repeat its own passes at every level, exponential in the nesting
        # depth.  A loop is rewritten only after its last lookup.
        self.carried_cache: dict[tuple[object, ...], _Facts] = (
            {} if carried_cache is None else carried_cache
        )

    def body(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.statement(statement)

    def statement(self, statement: ast.stmt) -> None:
        # ``_supported_body`` leaves single-target assignments, ``if``,
        # ``for`` over a name, expression statements and ``pass``.
        if isinstance(statement, ast.Assign):
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                self.facts.bind(target.id, statement.value)
        elif isinstance(statement, ast.For):
            self.loop(statement)
        elif isinstance(statement, ast.If):
            entry = self.facts
            for branch in (statement.body, statement.orelse):
                self.facts = entry.copy()
                self.body(branch)
            self.facts = entry
            self.facts.forget(names_stored_or_deleted(statement))

    def loop(self, loop: ast.For) -> None:
        # The body may run again with the values it leaves, so it starts
        # without facts about the names it assigns, as does the code after,
        # except those of the integers it carries and preserves (a
        # software-pipelined lane base).
        assigned = names_stored_or_deleted(loop)
        counter = self.counter_facts(loop)
        carried = self.carried_facts(loop, assigned, counter)
        self.facts.forget(assigned)
        self.facts.update(carried)
        if self.in_trial:
            if loop.orelse:
                self.facts.forget(assigned)
            return
        self.iteration(loop, counter)
        self.facts.forget(assigned)
        self.facts.update(carried)
        if loop.orelse:
            self.body(loop.orelse)
            self.facts.forget(assigned)

    def counter_facts(self, loop: ast.For) -> _Facts:
        """Facts of ``loop``'s counter, from its bounds at loop entry."""
        assert isinstance(loop.target, ast.Name)
        target = loop.target.id
        facts = self.facts.empty_like()
        if is_staged_loop(loop):
            facts.integer.add(target)
            facts.traced.add(target)
            factor = self.range_factor(loop.iter)
            if factor != 1:
                facts.factor[target] = factor
        elif isinstance(loop.iter, ast.Call) and (
            call_path_str(loop.iter.func) == "cutlass.range_constexpr"
        ):
            facts.integer.add(target)
        return facts

    def carried_facts(
        self, loop: ast.For, assigned: AbstractSet[str], counter: _Facts
    ) -> _Facts:
        """``find_carried_facts``, once per loop and facts at entry."""
        assert isinstance(loop.target, ast.Name)
        if not (self.facts.integer & assigned) - self.facts.with_merged(
            (loop.target.id,)
        ):
            # Nothing carried: ``find_carried_facts`` returns at once.
            return self.find_carried_facts(loop, assigned, counter)
        key = (loop, self.facts.key())
        if key not in self.carried_cache:
            self.carried_cache[key] = self.find_carried_facts(loop, assigned, counter)
        return self.carried_cache[key]

    def find_carried_facts(
        self, loop: ast.For, assigned: AbstractSet[str], counter: _Facts
    ) -> _Facts:
        """Facts of the integers ``loop`` assigns that hold at the start of
        every iteration: their facts at entry, weakened until a pass over the
        body leaves them holding (rewriting a copy of a vector loop's
        statements, the only ones such a pass rewrites)."""
        assert isinstance(loop.target, ast.Name)
        # The counter, and a name a rename merges with it, starts every
        # iteration with the counter's value, not with what it carried.
        names = (self.facts.integer & assigned) - self.facts.with_merged(
            (loop.target.id,)
        )
        carried = self.facts.restricted(names)
        if not names:
            return carried
        for _ in range(_CARRY_ROUNDS):
            trial = _Rewriter(
                self.facts.copy(), in_trial=True, carried_cache=self.carried_cache
            )
            trial.facts.forget(assigned)
            trial.facts.update(carried)
            vector = loop_extent(loop, kinds=("constexpr",), minimum=2) is not None
            trial.iteration(deep_clone(loop) if vector else loop, counter)
            weakened = carried.meet(trial.facts.restricted(names))
            if weakened == carried:
                return carried
            carried = weakened
        return self.facts.empty_like()

    def iteration(self, loop: ast.For, counter: _Facts) -> None:
        """Rewrite ``loop``'s body under the current facts."""
        assert isinstance(loop.target, ast.Name)
        target = loop.target.id
        self.facts.update(counter)
        body_assigned = names_stored_or_deleted(
            ast.Module(body=loop.body, type_ignores=[])
        )
        width = loop_extent(loop, kinds=("constexpr",), minimum=2)
        # A vector loop counts its lane through 0..V-1, V a power of two.
        if (
            width is not None
            and width & (width - 1) == 0
            and target not in self.facts.with_merged(body_assigned)
        ):
            self.vector_body(loop.body, target, width)
        else:
            self.body(loop.body)

    def range_factor(self, iterator: ast.expr) -> int:
        """A factor of every counter value of ``range(start, stop, step)``:
        the gcd of the start's and the step's (1 when unknown)."""
        if (
            not isinstance(iterator, ast.Call)
            or iterator.keywords
            or len(iterator.args) not in (2, 3)
        ):
            return 1
        start = iterator.args[0]
        step = iterator.args[2] if len(iterator.args) == 3 else ast.Constant(1)
        if not (self.facts.is_integer(start) and self.facts.is_integer(step)):
            return 1
        return math.gcd(self.facts.factor_of(start), self.facts.factor_of(step))

    def lane_offset_base(self, value: ast.expr, lane: str) -> ast.expr | None:
        """``base`` of ``base + lane`` or ``base + cutlass.Int32(lane)`` (either
        order), where ``value`` has the type of ``base``: a Python int lane
        takes the base's type, an Int32 one that of a traced base (every
        traced integer here is 32 or 64 bits wide)."""
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
            return None
        for base, offset in ((value.left, value.right), (value.right, value.left)):
            if isinstance(offset, ast.Name) and offset.id == lane:
                return base
            if (
                isinstance(offset, ast.Call)
                and call_path_str(offset.func) == "cutlass.Int32"
                and isinstance(argument := _cast_argument(offset), ast.Name)
                and argument.id == lane
                and self.facts.is_traced(base)
            ):
                return base
        return None

    def vector_body(self, statements: list[ast.stmt], lane: str, width: int) -> None:
        # Lane-offset names: ``name = base + lane`` with ``base`` a multiple
        # of the vector width, and the names ``offset_base`` resolves (aliases
        # and casts of them, and their shifts by integer multiples of the
        # width).
        bases: dict[str, ast.expr] = {}
        for statement in statements:
            if isinstance(statement, ast.Assign) and isinstance(
                statement.targets[0], ast.Name
            ):
                target = statement.targets[0].id
                statement.value = self.divide_bases(statement.value, lane, bases, width)
                # A recorded base reading the rebound name no longer holds.
                rebound = self.facts.with_merged((target,))
                for name, base in list(bases.items()):
                    if rebound & names_read(base) or (
                        name != target and name in rebound
                    ):
                        del bases[name]
                value = statement.value
                base = self.lane_offset_base(value, lane)
                if base is None:
                    base = self.offset_base(value, lane, bases, width)
                if (
                    base is not None
                    and self.facts.factor_of(base) % width == 0
                    and not rebound & names_read(base)
                ):
                    bases[target] = base
                else:
                    bases.pop(target, None)
                self.facts.bind(target, value)
                continue
            # Any other statement may rebind names: forget the lane offsets
            # it touches and recurse with the facts it leaves.
            stored = self.facts.with_merged(names_stored_or_deleted(statement))
            for name, base in list(bases.items()):
                if name in stored or stored & names_read(base):
                    del bases[name]
            self.statement(statement)

    def offset_base(
        self, node: ast.expr, lane: str, bases: dict[str, ast.expr], width: int
    ) -> ast.expr | None:
        """The base of a lane-offset expression, or None: a lane-offset name,
        under integer casts (which the base keeps: ``base + lane`` does not
        wrap), or shifted by an integer multiple of the width that reads
        neither the lane nor a lane offset (``row * d + tile.index``, whose
        base is ``row * d + base``: the offset has its base's type)."""
        if isinstance(node, ast.Name):
            return reparse_expr(bases[node.id]) if node.id in bases else None
        argument = _cast_argument(node)
        if argument is not None:
            base = self.offset_base(argument, lane, bases, width)
            if base is None:
                return None
            assert isinstance(node, ast.Call)
            return ast.Call(func=reparse_expr(node.func), args=[base], keywords=[])
        if not isinstance(node, ast.BinOp) or not isinstance(
            node.op, (ast.Add, ast.Sub)
        ):
            return None
        add = isinstance(node.op, ast.Add)
        pairs = [(node.left, node.right, False)]
        if add:
            pairs.append((node.right, node.left, True))
        for offset, shift, shift_first in pairs:
            if (
                names_read(shift) & {lane, *bases}
                or not self.facts.is_integer(shift)
                or self.facts.factor_of(shift) % width
                or (base := self.offset_base(offset, lane, bases, width)) is None
            ):
                continue
            left, right = (
                (reparse_expr(shift), base)
                if shift_first
                else (base, reparse_expr(shift))
            )
            return ast.BinOp(left=left, op=ast.Add() if add else ast.Sub(), right=right)
        return None

    def divide_bases(
        self, value: ast.expr, lane: str, bases: dict[str, ast.expr], width: int
    ) -> ast.expr:
        """``value`` with each ``offset // d`` (``offset`` a lane-offset
        expression, ``d`` a positive multiple of the width) dividing the
        offset's base instead, each ``_cute_python_mod(offset, d)`` taking
        the base's remainder plus the lane, and, with ``compare_bases``, each
        comparison of an offset with an integer multiple of the width
        comparing the base."""
        facts = self.facts

        def base_of(node: ast.expr) -> ast.expr | None:
            return self.offset_base(node, lane, bases, width)

        def compared_base(offset: ast.expr, bound: ast.expr) -> ast.expr | None:
            # ``base + lane < bound`` iff ``base < bound`` when both are
            # multiples of the width: the lane stays below the next one.
            # (``factor_of`` is 1 for a bound that may not be an integer.)
            if not facts.uniform_compares or facts.factor_of(bound) % width:
                return None
            if (base := base_of(offset)) is not None:
                return base
            # An offset spelled in place, ``base + lane`` (a remainder this
            # rewrite turned into the base's remainder plus the lane).
            base = self.lane_offset_base(offset, lane)
            if base is None or facts.factor_of(base) % width:
                return None
            return reparse_expr(base)

        def compare_offset(operands: list[ast.expr], side: int | None) -> None:
            """Replace ``operands[side]``, a lane offset compared with the
            other operand, with its base where that keeps the answer."""
            if side is not None:
                base = compared_base(operands[side], operands[1 - side])
                if base is not None:
                    operands[side] = base

        class BaseTerms(ast.NodeTransformer):
            def visit_Compare(self, node: ast.Compare) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.Compare)
                if len(visited.ops) != 1:
                    return visited
                operands = [visited.left, *visited.comparators]
                side = next(
                    (
                        side
                        for op, side in _OFFSET_OPERAND
                        if isinstance(visited.ops[0], op)
                    ),
                    None,
                )
                compare_offset(operands, side)
                visited.left, visited.comparators = operands[0], operands[1:]
                return visited

            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.BinOp)
                divisor = facts.constant_of(visited.right)
                if (
                    isinstance(visited.op, ast.FloorDiv)
                    and divisor is not None
                    and divisor > 0
                    and divisor % width == 0
                    and (dividend := base_of(visited.left)) is not None
                ):
                    visited.left = dividend
                return visited

            def visit_Call(self, node: ast.Call) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.Call)
                side = _COMPARISON_CALL_OPERAND.get(call_path_str(visited.func) or "")
                if side is not None:
                    if len(visited.args) == 2 and not visited.keywords:
                        compare_offset(visited.args, side)
                    return visited
                if not _is_python_mod(visited):
                    return visited
                divisor = facts.constant_of(visited.args[1])
                if (
                    divisor is None
                    or divisor <= 0
                    or divisor % width
                    or (dividend := base_of(visited.args[0])) is None
                ):
                    return visited
                # The lane is a trace-time int, which keeps the remainder's type.
                visited.args[0] = dividend
                return ast.BinOp(
                    left=visited, op=ast.Add(), right=ast.Name(id=lane, ctx=ast.Load())
                )

        result = BaseTerms().visit(value)
        assert isinstance(result, ast.expr)
        return result


def divide_vector_bases(
    body: list[ast.stmt],
    *,
    constexpr_values: Mapping[str, int],
    rename_groups: Mapping[str, str] | None = None,
    compare_bases: bool = False,
) -> list[ast.stmt]:
    """Rewrite each lane quotient of a vector loop to its base's quotient,
    each lane remainder to its base's remainder plus the lane, and, with
    ``compare_bases``, each comparison of a lane offset with a multiple of
    the width to compare its base, where the identity is proved (see the
    module docstring).

    ``rename_groups`` maps the names ``ast_rename`` will merge to their
    shared name.  The body still spells a loop-carried value and its aliases
    apart, so a name counts as assigned wherever a name merged with it is.
    """
    if _supported_body(body):
        facts = _Facts(
            set(constexpr_values),
            set(),
            {},
            dict(constexpr_values),
            merged_names(rename_groups or {}),
            compare_bases,
        )
        _Rewriter(facts).body(body)
    return body
