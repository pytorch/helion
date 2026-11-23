"""One branch around the statements of a loop step that a lane predicate masks.

A loop over a reduction block masks its last step's idle lanes: 416 packs
in steps of 256 leave the second step's last 96 lanes idle.  Each load of
the step then reads element 0 under ``mask ? offset : 0`` and selects its
fill after it (``select_masked_loads``), and each accumulation adds
``mask ? value : 0``, on every step, full ones included.  Where the
statements after the predicate's definition in a loop body read under it,
write only under it, and leave every value they carry out of the step as
it was when the predicate is false, ``if (mask) { ... }`` around them
computes the same: a lane the branch skips loaded nothing that counted and
added nothing.  The branch takes the selects off every step and the idle
lanes' work off the last one.

The analysis keeps the masked form wherever it can't prove that:

- the statements after the predicate are straight-line assignments to
  names, loads and stores, without control flow;
- their calls compute per lane, so a divergent branch can't break them:
  no SIMD-group, quad or threadgroup operation, atomic or barrier;
- every store and the predicate's every other use carries it as a
  conjunct (of ``&&``, or of ``&`` between booleans) of a load's or
  store's mask or of a select's condition, and the statements don't
  reassign it;
- the statements accumulate: they assign a name that holds a value from
  before them (an accumulator), and each such name keeps that value when
  the predicate is false, through a select whose condition the predicate
  falsifies, a load it masks off (its fill), a cast of zero, a product of
  zero with zero or a positive finite literal, and an addition of zero
  (``metal::fma`` of such a product included).  A step that only loads
  and stores, a pointwise kernel's lane loop, keeps its selected loads,
  which Metal issues together across the unrolled lanes;
- a lane the branch skips adds ``+0`` to each accumulator an addition of
  zero kept (``else { acc = acc + 0; }``), as the masked form did: that
  turns a carried ``-0.0`` into ``+0.0``, so the branch changes no bits.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
import math

from torch._inductor.codegen.mps import DTYPE_TO_METAL

from .lane_fold import BF16_CAST
from .lane_fold import load_mask_other
from .lane_fold import tl_load
from .lane_fold import tl_store
from .mpp_matmul import ROUND
from .msl_expr import PURE_CALLS

#: The MSL scalar types, whose constructors convert (``float(0.0)``).
_SCALAR_TYPES = frozenset(DTYPE_TO_METAL.values())
#: The calls by name that compute a register from registers: the casts, the
#: scalar constructors and the preamble's rounding helpers.
_LANE_CALLS = PURE_CALLS | _SCALAR_TYPES | {BF16_CAST, ROUND}
#: The C++ namespaces whose functions compute per lane, but those
#: ``_collective`` names.
_LANE_NAMESPACES = frozenset({"metal", "c10"})

#: A value when the predicate is false: ``("in", name)``, a name's value
#: where the statements start, ``_ZERO``, or None when unknown.
_Value = tuple[str, ...] | None
_ZERO: tuple[str, ...] = ("zero",)


def branch_masked_steps(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """*stmts* with each loop body's statements after a predicate that masks
    them behind one ``if`` on it (the module docstring)."""
    return _Pass(stmts).block(stmts)


def _assignments(stmts: list[ast.stmt]) -> Counter[str]:
    """How many times *stmts* assign each name, loop targets included."""
    return Counter(
        target.id
        for stmt in stmts
        for node in ast.walk(stmt)
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, (ast.AugAssign, ast.For))
            else []
        )
        if isinstance(target, ast.Name)
    )


def _store(node: ast.AST) -> ast.Call | None:
    """The ``tl.store(ptr, value, mask)`` call *node* is or makes, if any."""
    store = tl_store(node.value if isinstance(node, ast.Expr) else node)
    return store if store is not None and len(store.args) == 3 else None


def _store_mask(store: ast.Call) -> ast.expr | None:
    mask = store.args[2]
    return None if isinstance(mask, ast.Constant) and mask.value is None else mask


def _collective(name: str) -> bool:
    return name.startswith(("simd", "quad")) or any(
        word in name for word in ("threadgroup", "barrier", "atomic")
    )


def _lane_call(call: ast.Call) -> bool:
    """Whether *call* computes per lane: a load, a store, a cast or helper of
    ``_LANE_CALLS``, or a function of a ``_LANE_NAMESPACES`` namespace that
    isn't a ``_collective`` one."""
    func = call.func
    if tl_load(call) is not None or tl_store(call) is not None:
        return True
    if isinstance(func, ast.Name):
        return func.id in _LANE_CALLS
    if not isinstance(func, ast.Attribute) or _collective(func.attr):
        return False
    root = func.value
    while isinstance(root, ast.Attribute):
        root = root.value
    return isinstance(root, ast.Name) and root.id in _LANE_NAMESPACES


def _cast_operand(expr: ast.expr) -> ast.expr | None:
    """The value *expr* casts, ``static_cast<float>(x)`` or ``float(x)``."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        name, args = expr.func.id, expr.args
        if name == "static_cast" and len(args) == 2:
            return args[1]
        if name in _SCALAR_TYPES and len(args) == 1:
            return args[0]
    return None


def _number(expr: ast.expr) -> float | None:
    """*expr*'s value if it is a number literal."""
    value = expr.value if isinstance(expr, ast.Constant) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _positive_zero(expr: ast.expr) -> bool:
    """Whether *expr* is ``+0``: a zero literal, or a cast of one."""
    if (operand := _cast_operand(expr)) is not None:
        return _positive_zero(operand)
    value = _number(expr)
    return value is not None and value == 0 and math.copysign(1.0, value) > 0


def _positive_finite(expr: ast.expr) -> bool:
    """Whether *expr* is a positive finite literal, by which ``+0``
    multiplies or divides to ``+0``."""
    value = _number(expr)
    return value is not None and 0 < value < math.inf


def _fma(expr: ast.expr) -> ast.Call | None:
    """*expr* if it is ``metal::fma(a, b, c)`` (``fuse_multiply_adds``)."""
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id == "metal"
        and expr.func.attr == "fma"
        and len(expr.args) == 3
    ):
        return expr
    return None


class _Pass:
    def __init__(self, stmts: list[ast.stmt]) -> None:
        self.everywhere = _assignments(stmts)
        # The names every assignment sets to a boolean.
        values: dict[str, list[ast.expr]] = {}
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            values.setdefault(target.id, []).append(node.value)
        self.bools: set[str] = set()
        while more := {
            name
            for name, exprs in values.items()
            if name not in self.bools and all(map(self._is_bool, exprs))
        }:
            self.bools |= more

    def _is_bool(self, expr: ast.expr) -> bool:
        if isinstance(expr, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(expr, ast.UnaryOp):
            return isinstance(expr.op, ast.Not)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.BitAnd, ast.BitOr)):
            return self._is_bool(expr.left) and self._is_bool(expr.right)
        if isinstance(expr, ast.Constant):
            return type(expr.value) is bool
        return isinstance(expr, ast.Name) and expr.id in self.bools

    def conjuncts(self, test: ast.expr) -> list[ast.expr]:
        """The terms of *test*'s ``&&`` chain, and of ``&`` between booleans,
        which Helion joins a row mask to a step's with."""
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            return [c for v in test.values for c in self.conjuncts(v)]
        if (
            isinstance(test, ast.BinOp)
            and isinstance(test.op, ast.BitAnd)
            and self._is_bool(test.left)
            and self._is_bool(test.right)
        ):
            return [*self.conjuncts(test.left), *self.conjuncts(test.right)]
        return [test]

    def has(self, test: ast.expr, mask: str) -> bool:
        return any(
            isinstance(c, ast.Name) and c.id == mask for c in self.conjuncts(test)
        )

    def tests(self, stmts: list[ast.stmt]) -> list[ast.expr]:
        """The masks of *stmts*' loads and stores and their selects'
        conditions."""
        tests: list[ast.expr] = []
        for stmt in stmts:
            for node in ast.walk(stmt):
                if (load := tl_load(node)) is not None and (
                    masked := load_mask_other(load)
                ) is not None:
                    tests.append(masked[0])
                elif isinstance(node, ast.IfExp):
                    tests.append(node.test)
            if (store := _store(stmt)) is not None and (
                test := _store_mask(store)
            ) is not None:
                tests.append(test)
        return tests

    def block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        return [self.visit(stmt) for stmt in stmts]

    def visit(self, stmt: ast.stmt) -> ast.stmt:
        """*stmt* with its loop bodies' masked tails branched."""
        if not isinstance(stmt, (ast.For, ast.If)):
            return stmt
        new = copy.copy(stmt)
        new.body = self.block(stmt.body)
        new.orelse = self.block(stmt.orelse)
        if isinstance(stmt, ast.For):
            inside = _assignments(stmt.body)
            # Names assigned outside the body hold values across it.
            outside = {n for n, count in self.everywhere.items() if count > inside[n]}
            new.body = self.branch(new.body, outside)
        return new

    def branch(self, body: list[ast.stmt], outside: set[str]) -> list[ast.stmt]:
        # The predicates to try: the names a conjunct of a mask or a select's
        # condition reads, at the statements that assign them.
        placed = {
            c.id
            for test in self.tests(body)
            for c in self.conjuncts(test)
            if isinstance(c, ast.Name)
        }
        for i, stmt in enumerate(body):
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in placed
            ):
                continue
            mask = stmt.targets[0].id
            head, tail = body[: i + 1], body[i + 1 :]
            added = self.qualifies(head, tail, mask, outside)
            if added is None:
                continue
            assume = _Assume(self, mask)
            branch = ast.If(
                test=ast.Name(id=mask, ctx=ast.Load()),
                body=[assume.rewrite_stmt(s) for s in tail],
                orelse=[ast.copy_location(_add_zero(name), stmt) for name in added],
            )
            return [*head, ast.copy_location(branch, stmt)]
        return body

    def qualifies(
        self,
        head: list[ast.stmt],
        tail: list[ast.stmt],
        mask: str,
        outside: set[str],
    ) -> list[str] | None:
        """The accumulators an addition of zero kept, if *tail* can go behind
        an ``if`` on *mask*; None if it can't."""
        for stmt in tail:
            if (store := _store(stmt)) is not None:
                test = _store_mask(store)
                if test is None or not self.has(test, mask):
                    return None
            elif not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id != mask
            ):
                return None
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call) and not _lane_call(node):
                    return None
        reads = sum(
            isinstance(node, ast.Name) and node.id == mask
            for stmt in tail
            for node in ast.walk(stmt)
        )
        placed = sum(
            isinstance(c, ast.Name) and c.id == mask
            for test in self.tests(tail)
            for c in self.conjuncts(test)
        )
        if not placed or placed != reads:
            return None
        return self.carries(head, tail, mask, outside)

    def carries(
        self,
        head: list[ast.stmt],
        tail: list[ast.stmt],
        mask: str,
        outside: set[str],
    ) -> list[str] | None:
        """The names *tail* assigns that hold a value across the loop body
        (assigned *outside* it, or read in it before it assigns it) and that
        an addition of zero keeps, if *tail* assigns such names and each ends
        *tail* with the value it had where *tail* starts when *mask* is
        false; None otherwise."""
        assigned: set[str] = set()
        carried = set(outside)
        for stmt in [*head, *tail]:
            reads = stmt.value if isinstance(stmt, ast.Assign) else stmt
            carried.update(
                node.id
                for node in ast.walk(reads)
                if isinstance(node, ast.Name) and node.id not in assigned
            )
            if isinstance(stmt, ast.Assign):
                assigned.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        env: dict[str, tuple[_Value, bool]] = {}
        for stmt in tail:
            if isinstance(stmt, ast.Assign):
                (target,) = stmt.targets
                assert isinstance(target, ast.Name)
                env[target.id] = self.when_false(stmt.value, env, mask)
        kept = {name: v for name, v in env.items() if name in carried}
        if not kept or any(value != ("in", name) for name, (value, _) in kept.items()):
            return None
        return [name for name, (_, added) in kept.items() if added]

    def when_false(
        self, expr: ast.expr, env: dict[str, tuple[_Value, bool]], mask: str
    ) -> tuple[_Value, bool]:
        """*expr*'s value when *mask* is false, and whether keeping a value
        took an addition of zero."""
        if isinstance(expr, ast.Name):
            return env.get(expr.id, (("in", expr.id), False))
        if _number(expr) is not None:
            return (_ZERO if _positive_zero(expr) else None), False
        if (
            (load := tl_load(expr)) is not None
            and (masked := load_mask_other(load)) is not None
            and self.has(masked[0], mask)
            and masked[1] is not None
        ):
            # A load the mask masks off gives its fill.
            return self.when_false(masked[1], env, mask)
        if (operand := _cast_operand(expr)) is not None:
            zero = self.when_false(operand, env, mask)[0] == _ZERO
            return (_ZERO if zero else None), False
        if isinstance(expr, ast.IfExp) and self.has(expr.test, mask):
            return self.when_false(expr.orelse, env, mask)
        if (fma := _fma(expr)) is not None:
            a, b, c = fma.args
            if self.product_zero(a, b, env, mask):
                return self.when_false(c, env, mask)[0], True
            return None, False
        if isinstance(expr, ast.BinOp):
            if isinstance(expr.op, (ast.Add, ast.Sub)):
                lhs, lhs_added = self.when_false(expr.left, env, mask)
                rhs, rhs_added = self.when_false(expr.right, env, mask)
                # x + +0 turns -0.0 into +0.0; x - +0 keeps it, and keeps a
                # value an earlier addition of zero kept.
                if rhs == _ZERO:
                    return lhs, lhs_added or isinstance(expr.op, ast.Add)
                if lhs == _ZERO and isinstance(expr.op, ast.Add):
                    return rhs, True
            elif (
                isinstance(expr.op, ast.Mult)
                and self.product_zero(expr.left, expr.right, env, mask)
            ) or (
                # +0 over a positive finite literal.
                isinstance(expr.op, ast.Div)
                and self.when_false(expr.left, env, mask)[0] == _ZERO
                and _positive_finite(expr.right)
            ):
                return _ZERO, False
        return None, False

    def product_zero(
        self,
        a: ast.expr,
        b: ast.expr,
        env: dict[str, tuple[_Value, bool]],
        mask: str,
    ) -> bool:
        """Whether ``a * b`` is ``+0`` when *mask* is false: both factors
        are, or one is and the other a positive finite literal."""
        za = self.when_false(a, env, mask)[0] == _ZERO
        zb = self.when_false(b, env, mask)[0] == _ZERO
        return (
            (za and zb) or (za and _positive_finite(b)) or (zb and _positive_finite(a))
        )


def _add_zero(name: str) -> ast.stmt:
    """``name = name + static_cast<decltype(name)>(0)``."""
    zero = ast.Call(
        func=ast.Name(id="static_cast", ctx=ast.Load()),
        args=[
            ast.Call(
                func=ast.Name(id="decltype", ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load())],
                keywords=[],
            ),
            ast.Constant(0),
        ],
        keywords=[],
    )
    return ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.BinOp(
            left=ast.Name(id=name, ctx=ast.Load()), op=ast.Add(), right=zero
        ),
    )


class _Assume:
    """Rewrites statements under ``if (mask)``: drops *mask* from the masks
    of loads and stores and from the conditions of selects, copying only
    the nodes on the way to a change."""

    def __init__(self, analysis: _Pass, mask: str) -> None:
        self.analysis = analysis
        self.mask = mask

    def rewrite_stmt(self, stmt: ast.stmt) -> ast.stmt:
        new = self.rewrite(stmt)
        assert isinstance(new, ast.stmt)
        return new

    def rewrite(self, node: ast.AST) -> ast.AST:
        fields: dict[str, object] = {}
        changed = False
        for name, value in ast.iter_fields(node):
            if isinstance(value, list):
                new = [self.rewrite(v) if isinstance(v, ast.AST) else v for v in value]
                changed |= any(a is not b for a, b in zip(new, value, strict=True))
            elif isinstance(value, ast.AST):
                new = self.rewrite(value)
                changed |= new is not value
            else:
                new = value
            fields[name] = new
        if changed:
            node = ast.copy_location(type(node)(**fields), node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.IfExp):
            return self._select(node)
        return node

    def _rest(self, test: ast.expr) -> ast.expr | None:
        rest = [
            c
            for c in self.analysis.conjuncts(test)
            if not (isinstance(c, ast.Name) and c.id == self.mask)
        ]
        if not rest:
            return None
        if len(rest) == 1:
            return rest[0]
        return ast.BoolOp(op=ast.And(), values=rest)

    def _call(self, node: ast.Call) -> ast.AST:
        if (load := tl_load(node)) is not None and (
            masked := load_mask_other(load)
        ) is not None:
            test, _ = masked
            if not self.analysis.has(test, self.mask):
                return node
            rest = self._rest(test)
            if rest is None:
                return ast.Call(func=node.func, args=node.args[:1], keywords=[])
            return ast.Call(
                func=node.func,
                args=[node.args[0], rest, *node.args[2:]],
                keywords=node.keywords,
            )
        if (store := _store(node)) is not None and (
            test := _store_mask(store)
        ) is not None:
            rest = self._rest(test)
            return ast.Call(
                func=node.func,
                args=[*node.args[:2], ast.Constant(None) if rest is None else rest],
                keywords=node.keywords,
            )
        return node

    def _select(self, node: ast.IfExp) -> ast.AST:
        if not self.analysis.has(node.test, self.mask):
            return node
        rest = self._rest(node.test)
        if rest is None:
            return node.body
        return ast.IfExp(test=rest, body=node.body, orelse=node.orelse)
