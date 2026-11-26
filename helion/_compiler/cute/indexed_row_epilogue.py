"""Move a proved pointwise phase to the two writers of its staged BF16 row."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import exposed_reads
from ._ast_pass_utils import name_assignment
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written
from ._ast_pass_utils import phase_barrier_call
from ._ast_pass_utils import rename_names
from ._ast_pass_utils import same_expr
from ._ast_pass_utils import scalar_index_cast_arg
from ._ast_pass_utils import scalar_index_offsets
from ._ast_pass_utils import scalar_index_ptr
from ._ast_pass_utils import scalar_tensor_access

if TYPE_CHECKING:
    from .analysis.indexed_row_epilogue import RowEpilogue


_ARITHMETIC_ARITY = {
    "cutlass.Float32": 1,
    "cutlass.BFloat16": 1,
    "cute.math.fma": 3,
    "cute.math.tanh": 1,
}


@dataclass(frozen=True)
class _Slice:
    low: str
    high: str
    math: list[ast.stmt]
    result: ast.expr


def _shift(
    expr: ast.expr, low: ast.expr, delta: int, defs: dict[str, ast.expr]
) -> bool:
    """Prove low+delta without general address simplification or an implicit cast."""

    def get(value: ast.expr, seen: set[str]) -> tuple[int, int] | None:
        if same_expr(value, low):
            return 1, 0
        if isinstance(value, ast.Constant) and type(value.value) is int:
            return 0, value.value
        if isinstance(value, ast.Name) and value.id in defs and value.id not in seen:
            return get(defs[value.id], seen | {value.id})
        if (v := scalar_index_cast_arg(value)) is not None:
            return get(v, seen)
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            a, b = get(value.left, seen), get(value.right, seen)
            if a is not None and b is not None:
                return a[0] + b[0], a[1] + b[1]
        return None

    return get(expr, set()) == (1, delta)


def _address(call: ast.Call, tensor: str) -> list[ast.expr] | None:
    if isinstance(call.func, ast.Attribute):
        return scalar_index_offsets(call.func.value, tensor, 2)
    return None


def _has_dominated_value(block: list[ast.stmt], expression: ast.expr) -> bool:
    # Reused names under an address branch would require an SSA join.
    return not any(names_written(stmt) & names_read(expression) for stmt in block)


def _arithmetic(expr: ast.expr, available: set[str]) -> bool:
    if isinstance(expr, ast.Constant):
        return type(expr.value) in (int, float)
    if isinstance(expr, ast.UnaryOp):
        # Negative constants in FX reappear as signed literals in the AST.
        return (
            isinstance(expr.op, (ast.USub, ast.UAdd))
            and isinstance(expr.operand, ast.Constant)
            and type(expr.operand.value) in (int, float)
        )
    if isinstance(expr, ast.Name):
        return expr.id in available
    if isinstance(expr, ast.BinOp):
        return isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult)) and (
            _arithmetic(expr.left, available) and _arithmetic(expr.right, available)
        )
    if isinstance(expr, ast.Call):
        if expr.keywords and not (
            call_path_str(expr.func) == "cute.math.tanh"
            and len(expr.keywords) == 1
            and expr.keywords[0].arg == "fastmath"
            and isinstance(expr.keywords[0].value, ast.Constant)
            and expr.keywords[0].value.value is True
        ):
            return False
        arity = _ARITHMETIC_ARITY.get(call_path_str(expr.func) or "")
        return arity == len(expr.args) and all(
            _arithmetic(x, available) for x in expr.args
        )
    return False


def _slice(consumer: ast.For, p: RowEpilogue) -> _Slice | None:
    if not consumer.body or not isinstance(guard := consumer.body[-1], ast.If):
        return None
    if any(not isinstance(n, ast.Pass) for n in guard.orelse):
        return None
    ordinary_calls = {
        "range",
        "cutlass.Int32",
        "cutlass.Int64",
        "cute.arch.block_idx",
        *_ARITHMETIC_ARITY,
    }
    if (
        consumer.orelse
        or any(not isinstance(stmt, ast.Assign) for stmt in consumer.body[:-1])
        or any(
            isinstance(n, ast.Call)
            and scalar_tensor_access(n) is None
            and call_path_str(n.func) not in ordinary_calls
            for n in ast.walk(consumer)
        )
    ):
        return None
    mem = [
        scalar_tensor_access(n)
        for n in ast.walk(consumer)
        if isinstance(n, ast.Call) and scalar_tensor_access(n)
    ]
    if (
        mem.count((p.input, "load")) != 2
        or mem.count((p.output, "store")) != 1
        or len(mem) != 3
    ):
        return None
    body = guard.body
    if (
        not body
        or not isinstance(end := body[-1], ast.Expr)
        or not isinstance(st := end.value, ast.Call)
    ):
        return None
    target = _address(st, p.output)
    if scalar_tensor_access(st) != (p.output, "store") or target is None:
        return None
    # Use the order and rounding of the final scalar CuTe operations, including
    # contracted multiply-add; algebraically re-emitting FX may change both.
    defs: dict[str, ast.expr] = {}
    reads: dict[str, str] = {}
    for line in body[:-1]:
        assignment = name_assignment(line)
        if assignment is None:
            return None
        name, value = assignment
        # Prologue can reassign the persistent PID. Scalar arithmetic must be SSA.
        if name in defs and (reads or name in names_read(st.args[0])):
            return None
        defs[name] = value
        if isinstance(value, ast.IfExp):
            test = value.test
            if (
                not isinstance(test, ast.Compare)
                or len(test.ops) != 1
                or not isinstance(test.ops[0], ast.Lt)
                or len(test.comparators) != 1
                or not isinstance(test.comparators[0], ast.Constant)
                or test.comparators[0].value != 2 * p.columns
            ):
                return None
            if not _shift(test.left, target[1], p.columns, defs):
                return None
            value = value.body
        if not isinstance(value, ast.Call) or scalar_tensor_access(value) != (
            p.input,
            "load",
        ):
            continue
        addr = _address(value, p.input)
        if addr is None or not same_expr(target[0], addr[0]):
            return None
        delta = next(
            (i for i in (0, p.columns) if _shift(addr[1], target[1], i, defs)), None
        )
        if delta is None:
            return None
        role = "high" if delta else "low"
        if role in reads:
            return None
        reads[role] = name
    if set(reads) != {"low", "high"}:
        return None
    live = set(reads.values())
    recipe: list[ast.stmt] = []
    required = names_read(st.args[0]) - {"cutlass", "cute"}
    for line in reversed(body[:-1]):
        assn = name_assignment(line)
        assert assn is not None
        name, value = assn
        if name in required and name not in live:
            required.remove(name)
            required.update(names_read(value) - {"cutlass", "cute"})
            recipe.append(line)
    if required != live:
        return None
    available = live.copy()
    for line in reversed(recipe):
        assn = name_assignment(line)
        assert assn is not None
        name, value = assn
        if not _arithmetic(value, available):
            return None
        available.add(name)
    if not _arithmetic(st.args[0], available):
        return None
    return _Slice(reads["low"], reads["high"], list(reversed(recipe)), st.args[0])


def _stores(branch: list[ast.stmt], tensor: str) -> list[ast.Expr]:
    return [
        stmt
        for st in branch
        for stmt in ast.walk(st)
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and scalar_tensor_access(stmt.value) == (tensor, "store")
    ]


def _append(branch: list[ast.stmt], p: RowEpilogue, s: _Slice, scalar: bool) -> bool:
    stores = _stores(branch, p.input)
    if len(stores) != 2:
        return False
    first, second = stores
    a, b = cast("ast.Call", first.value), cast("ast.Call", second.value)
    lo, hi = _address(a, p.input), _address(b, p.input)
    if (
        lo is None
        or hi is None
        or not same_expr(lo[0], hi[0])
        or not all(
            isinstance(v.args[0], ast.Call)
            and call_path_str(v.args[0].func) == "cutlass.BFloat16"
            for v in [a, b]
        )
    ):
        return False
    parents = {
        c: n for st in branch for n in ast.walk(st) for c in ast.iter_child_nodes(n)
    }
    low_owner, high_owner = parents[first], parents[second]
    if (
        not isinstance(low_owner, ast.If)
        or not isinstance(high_owner, ast.If)
        or first not in low_owner.body
        or second not in high_owner.body
    ):
        return False
    defs = {}
    if scalar:
        common = parents.get(low_owner)
        if (
            not isinstance(common, ast.For)
            or parents.get(high_owner) is not common
            or low_owner is high_owner
            or not same_expr(low_owner.test, high_owner.test)
            or low_owner not in common.body
            or high_owner not in common.body
        ):
            return False
        i, j = common.body.index(low_owner), common.body.index(high_owner)
        if (
            i >= j
            or any(
                not isinstance(x, ast.Pass)
                for x in [*low_owner.orelse, *high_owner.orelse]
            )
            or low_owner.body != [first]
            or high_owner.body != [second]
        ):
            return False
        gap = common.body[i + 1 : j]
        if any(name_assignment(x) is None for x in gap) or not all(
            _has_dominated_value(gap, x)
            for x in [a.args[0], lo[0], lo[1], low_owner.test]
        ):
            return False
        defs = dict(cast("tuple[str,ast.expr]", name_assignment(x)) for x in gap)
    elif (
        low_owner is not high_owner
        or high_owner.body.index(second) != high_owner.body.index(first) + 1
    ):
        return False
    if not _shift(hi[1], lo[1], p.columns, defs):
        return False
    renamed = {
        name: "_stage_epilogue_" + name
        for name in [
            s.low,
            s.high,
            *[cast("tuple[str,ast.expr]", name_assignment(i))[0] for i in s.math],
        ]
    }
    code: list[ast.stmt] = [
        ast.copy_location(
            ast.Assign([ast.Name(renamed[n], ast.Store())], deep_clone(v.args[0])),
            original,
        )
        for n, v, original in ((s.low, a, first), (s.high, b, second))
    ]
    code.extend(rename_names(deep_clone(i), renamed) for i in s.math)
    dst = ast.parse(
        f"({scalar_index_ptr(p.output, *map(ast.unparse, lo))}).store(0)"
    ).body[0]
    cast("ast.Call", cast("ast.Expr", dst).value).args[0] = rename_names(
        deep_clone(s.result), renamed
    )
    code.append(dst)
    if scalar:
        # The scalar lane index is x*lanes+local (checked in _recipes); the
        # row-sum broadcasts across y. Leave exactly one new writer per cell.
        code = [
            ast.If(
                ast.parse("cute.arch.thread_idx()[1] == 0", mode="eval").body, code, []
            )
        ]
    j = high_owner.body.index(second)
    high_owner.body[j + 1 : j + 1] = code
    return True


def fuse_row_epilogue(
    definition: ast.FunctionDef,
    producer: ast.For,
    consumer: ast.For,
    staged: ast.If,
    p: RowEpilogue,
) -> bool:
    """Rewrite both address branches or neither, then remove one whole join."""
    pi, ci = definition.body.index(producer), definition.body.index(consumer)
    if (
        ci != pi + 2
        or ci + 1 >= len(definition.body)
        or not same_expr(definition.body[ci - 1], definition.body[ci + 1])
    ):
        return False
    if names_written(consumer) & exposed_reads(definition.body[ci + 1 :]):
        return False
    barrier = phase_barrier_call(definition.body[ci - 1])
    if barrier is None or call_path_str(barrier.func) != "_cute_grid_barrier":
        return False
    if (
        any(
            n.id.startswith("_stage_epilogue_")
            for n in ast.walk(definition)
            if isinstance(n, ast.Name)
        )
        or (s := _slice(consumer, p)) is None
    ):
        return False
    copy = deep_clone(staged)
    if not (_append(copy.body, p, s, False) and _append(copy.orelse, p, s, True)):
        return False
    staged.body = copy.body
    staged.orelse = copy.orelse
    del definition.body[ci - 1 : ci + 1]
    return True
