"""Divide a vector's base index instead of each of its lanes.

A per-channel parameter of an NCHW tensor read through a flat tile,
``w[(tile.index // (H * W)) % C]``, lowers inside the unrolled vector loop
to one ``(lane_base + vec_lane) // (H * W)`` per lane, so each of the V
lanes repeats the index math and every gather.  When ``lane_base`` is a
multiple of V and the divisor is too, all V lanes have the same quotient:
``lane_base`` sits ``0 mod V`` below a multiple of the divisor, so adding a
lane offset below V never crosses one (floor division, any sign).  The pass
rewrites such a quotient to ``lane_base // divisor``; the V unrolled copies
then compute one value and load one parameter, which LLVM's CSE merges.

Facts are proved syntactically on the generated kernel body.  A name's known
factor follows integer constants, constexpr values, sums (gcd), products and
integer casts of integer expressions (``scalar_integer``); anything else has
factor 1.  V is a power of two, so a factor of V survives wrapping arithmetic
and narrowing casts, and ``lane_base + vec_lane`` cannot wrap.  Only literal
``range_constexpr`` vector loops that leave their lane alone are recognized,
with lane offsets spelled ``base + lane`` or ``base + cutlass.Int32(lane)``
(the cast form only for a traced base, whose type the sum keeps).  The
quotient's dividend name is replaced under its casts, so the division keeps
its type.  Facts follow straight-line code: a loop forgets the names it
assigns before its body, an ``if`` the names either branch assigns, and a
body with statements ``simplify_proven_bounds`` cannot model is left alone.
The rewrite is an integer identity, so it changes no result.
"""

from __future__ import annotations

import ast
import contextvars
import dataclasses
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

from ._ast_pass_utils import call_path
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import is_staged_loop
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import merged_names
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import reparse_expr
from ._ast_pass_utils import with_merged_names
from .scalar_integer import _INTEGER_CASTS
from .scalar_integer import _integer_expression
from .simplify_proven_bounds import _supported_body

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

# A constant this size reads the same through every integer cast.
_INT32_MAX = (1 << 31) - 1
# For one ``divide_vector_bases`` call, each name ``ast_rename`` merges with
# others (a loop-carried value and its aliases), with all of them: once
# renamed they are one variable, so a write to any is a write to all.
_MERGED: contextvars.ContextVar[Mapping[str, frozenset[str]]] = contextvars.ContextVar(
    "lane_invariant_division_merged", default=MappingProxyType({})
)


def _with_merged(names: Iterable[str]) -> set[str]:
    return with_merged_names(names, _MERGED.get())


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


@dataclasses.dataclass
class _Facts:
    """What the straight-line code so far proves about names: which hold
    integers, which of those are traced DSL values, their known factors and
    their trace-time constant values."""

    integer: set[str]
    traced: set[str]
    factor: dict[str, int]
    constant: dict[str, int]

    def copy(self) -> _Facts:
        return _Facts(
            set(self.integer), set(self.traced), dict(self.factor), dict(self.constant)
        )

    def is_integer(self, node: ast.expr) -> bool:
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
        for name in _with_merged(names):
            self.integer.discard(name)
            self.traced.discard(name)
            self.factor.pop(name, None)
            self.constant.pop(name, None)


class _Rewriter:
    def __init__(self, constexpr_values: Mapping[str, int]) -> None:
        self.facts = _Facts(set(constexpr_values), set(), {}, dict(constexpr_values))

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
        # without facts about the names it assigns, as does the code after.
        assigned = names_stored_or_deleted(loop)
        self.facts.forget(assigned)
        assert isinstance(loop.target, ast.Name)
        target = loop.target.id
        if is_staged_loop(loop):
            self.facts.integer.add(target)
            self.facts.traced.add(target)
        elif isinstance(loop.iter, ast.Call) and (
            call_path_str(loop.iter.func) == "cutlass.range_constexpr"
        ):
            self.facts.integer.add(target)
        body_assigned = names_stored_or_deleted(
            ast.Module(body=loop.body, type_ignores=[])
        )
        width = loop_extent(loop, kinds=("constexpr",), minimum=2)
        # A vector loop counts its lane through 0..V-1, V a power of two.
        if (
            width is not None
            and width & (width - 1) == 0
            and target not in _with_merged(body_assigned)
        ):
            self.vector_body(loop.body, target, width)
        else:
            self.body(loop.body)
        self.facts.forget(assigned)
        self.body(loop.orelse)
        self.facts.forget(assigned)

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
        # of the vector width, and plain aliases of them.
        bases: dict[str, ast.expr] = {}
        for statement in statements:
            if isinstance(statement, ast.Assign) and isinstance(
                statement.targets[0], ast.Name
            ):
                target = statement.targets[0].id
                statement.value = self.divide_bases(statement.value, bases, width)
                # A recorded base reading the rebound name no longer holds.
                rebound = _with_merged((target,))
                for name, base in list(bases.items()):
                    if rebound & names_read(base) or (
                        name != target and name in rebound
                    ):
                        del bases[name]
                value = statement.value
                base = self.lane_offset_base(value, lane)
                if (
                    base is not None
                    and self.facts.factor_of(base) % width == 0
                    and not rebound & names_read(base)
                ):
                    bases[target] = base
                elif isinstance(value, ast.Name) and value.id in bases:
                    bases[target] = bases[value.id]
                else:
                    bases.pop(target, None)
                self.facts.bind(target, value)
                continue
            # Any other statement may rebind names: forget the lane offsets
            # it touches and recurse with the facts it leaves.
            stored = _with_merged(names_stored_or_deleted(statement))
            for name, base in list(bases.items()):
                if name in stored or stored & names_read(base):
                    del bases[name]
            self.statement(statement)

    def divide_bases(
        self, value: ast.expr, bases: dict[str, ast.expr], width: int
    ) -> ast.expr:
        """``value`` with each ``offset // d`` (``offset`` a lane-offset name
        under any integer casts, ``d`` a positive multiple of the width)
        dividing the offset's base instead, under the same casts."""
        facts = self.facts

        def base_under_casts(node: ast.expr) -> ast.expr | None:
            argument = _cast_argument(node)
            if argument is not None:
                replaced = base_under_casts(argument)
                if replaced is None:
                    return None
                assert isinstance(node, ast.Call)
                node.args[0] = replaced
                return node
            if isinstance(node, ast.Name) and node.id in bases:
                return reparse_expr(bases[node.id])
            return None

        class Quotients(ast.NodeTransformer):
            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                visited = self.generic_visit(node)
                assert isinstance(visited, ast.BinOp)
                divisor = facts.constant_of(visited.right)
                if (
                    isinstance(visited.op, ast.FloorDiv)
                    and divisor is not None
                    and divisor > 0
                    and divisor % width == 0
                    and (dividend := base_under_casts(visited.left)) is not None
                ):
                    visited.left = dividend
                return visited

        result = Quotients().visit(value)
        assert isinstance(result, ast.expr)
        return result


def _divide_vector_bases(
    body: list[ast.stmt], *, constexpr_values: Mapping[str, int]
) -> list[ast.stmt]:
    """Rewrite each lane quotient of a vector loop to its base's quotient
    where the identity is proved (see the module docstring)."""
    if _supported_body(body):
        _Rewriter(constexpr_values).body(body)
    return body


def divide_vector_bases(
    body: list[ast.stmt],
    *,
    constexpr_values: Mapping[str, int],
    rename_groups: Mapping[str, str] | None = None,
) -> list[ast.stmt]:
    """Rewrite each lane quotient of a vector loop to its base's quotient
    where the identity is proved (see the module docstring).

    ``rename_groups`` maps the names ``ast_rename`` will merge to their
    shared name.  The body still spells a loop-carried value and its aliases
    apart, so a name counts as assigned wherever a name merged with it is.
    """
    token = _MERGED.set(merged_names(rename_groups or {}))
    try:
        return _divide_vector_bases(body, constexpr_values=constexpr_values)
    finally:
        _MERGED.reset(token)
