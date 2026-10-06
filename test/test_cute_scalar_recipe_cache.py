"""CPU semantics for invariant load caching across a reduction loop."""

from __future__ import annotations

import ast
import dataclasses
import itertools
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.scalar_recipe_cache import plan_scalar_recipe_cache

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.cute.scalar_recipe_cache import ScalarRecipeCache


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _assignments(source: str) -> list[ast.Assign]:
    return cast("list[ast.Assign]", ast.parse(source).body)


def _fresh() -> Callable[[str], str]:
    counter = itertools.count()
    return lambda hint: f"prefetch_{next(counter)}_{hint}"


_PRELUDE = "row_position = tid + slot * 16"
_PREFIX = """
count = counts.iterator.load()
valid = row_position < count
selected = (indices.iterator + row_position).load() if valid else cutlass.Int32(0)
row = selected
offset = cutlass.Int64(row) * cutlass.Int64(32) + cutlass.Int64(k_step)
"""
_BODY = "observed.append((slot, offset))"


def _plan(
    *,
    prefix: str = _PREFIX,
    body: str = _BODY,
    slot_count: int = 4,
    slot_guard: str = "row_position < total",
    readonly: tuple[str, ...] = ("counts", "indices"),
    max_slots: int = 8,
    max_cached_values: int = 16,
) -> ScalarRecipeCache | None:
    return plan_scalar_recipe_cache(
        _assignments(prefix),
        ast.parse(body).body,
        slot_name="slot",
        slot_count=slot_count,
        slot_prelude=_assignments(_PRELUDE),
        slot_guard=_expr(slot_guard),
        boundary_names={"tid", "total", "counts", "indices"},
        varying_names={"k_step"},
        tensor_dtypes={"counts": "cutlass.Int32", "indices": "cutlass.Int32"},
        readonly_tensors=readonly,
        max_slots=max_slots,
        max_cached_values=max_cached_values,
    )


@dataclasses.dataclass
class _Pointer:
    values: list[int]
    reads: list[int]
    offset: int = 0

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.values, self.reads, self.offset + offset)

    def load(self) -> int:
        assert 0 <= self.offset < len(self.values), "out-of-bounds prefill load"
        self.reads.append(self.offset)
        return self.values[self.offset]


def _i32(value: int) -> int:
    return (int(value) + 2**31) % 2**32 - 2**31


class _Cache:
    def __init__(self, shape: tuple[int, ...], dtype: Callable[[int], int]) -> None:
        self.values = [0] * shape[0]
        self.dtype = dtype

    def fill(self, value: int) -> None:
        self.values[:] = [self.dtype(value)] * len(self.values)

    def __setitem__(self, index: int, value: int) -> None:
        self.values[index] = self.dtype(value)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


def _execute(
    *,
    cached: bool,
    steps: int,
    slots: int,
    tid: int,
    total: int,
    count: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    counts = SimpleNamespace(iterator=_Pointer([count], []))
    indices = SimpleNamespace(iterator=_Pointer(list(reversed(range(256))), []))
    prefix = _assignments(_PREFIX)
    before: list[ast.stmt] = []
    iterator = _expr(f"range({slots})")
    if cached:
        plan = _plan(slot_count=slots, max_slots=max(8, slots))
        assert plan is not None
        emitted = plan.emit(_fresh(), execution_guard=_expr("steps > 0"))
        assert emitted.cached_names == ("row",)
        before = emitted.prologue
        prefix = emitted.prefix
        iterator = emitted.slot_iterator
    slot_loop = ast.For(
        target=ast.Name(id="slot", ctx=ast.Store()),
        iter=iterator,
        body=[
            *_assignments(_PRELUDE),
            ast.If(
                test=_expr("row_position < total"),
                body=[*prefix, *ast.parse(_BODY).body],
                orelse=[],
            ),
        ],
        orelse=[],
    )
    loop = ast.For(
        target=ast.Name(id="k_step", ctx=ast.Store()),
        iter=_expr("range(steps)"),
        body=[slot_loop],
        orelse=[],
    )
    observed: list[tuple[int, int]] = []
    namespace = {
        "counts": counts,
        "indices": indices,
        "steps": steps,
        "tid": tid,
        "total": total,
        "observed": observed,
        "operator": operator,
        "cutlass": SimpleNamespace(
            Int32=_i32, Int64=int, Boolean=bool, range_constexpr=range
        ),
        "cute": SimpleNamespace(make_rmem_tensor=_Cache),
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[*before, loop], type_ignores=[])
            ),
            "<recipe-cache-test>",
            "exec",
        ),
        namespace,
    )
    return observed, counts.iterator.reads, indices.iterator.reads


@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize(
    "slots,tid,total,count",
    [
        (1, 0, 16, 16),
        (2, 3, 32, 31),
        (4, 0, 63, 61),
        (4, 15, 51, 48),
        (8, 1, 115, 113),
        (16, 0, 256, 253),
    ],
)
def test_caches_gather_once_per_valid_slot_and_preserves_every_output(
    steps: int, slots: int, tid: int, total: int, count: int
) -> None:
    original, old_counts, old_indices = _execute(
        cached=False, steps=steps, slots=slots, tid=tid, total=total, count=count
    )
    actual, counts, indices = _execute(
        cached=True, steps=steps, slots=slots, tid=tid, total=total, count=count
    )
    assert actual == original
    assert len(counts) * steps == len(old_counts)
    assert len(indices) * steps == len(old_indices)
    if steps == 0:
        assert counts == indices == []
    else:
        assert indices == [
            tid + slot * 16
            for slot in range(slots)
            if tid + slot * 16 < min(total, count)
        ]


def test_consumption_and_prefill_both_use_static_slots() -> None:
    plan = _plan()
    assert plan is not None
    emitted = plan.emit(_fresh(), execution_guard=_expr("K > 0"))
    assert ast.unparse(emitted.slot_iterator) == "cutlass.range_constexpr(4)"
    loops = [
        node
        for stmt in emitted.prologue
        for node in ast.walk(stmt)
        if isinstance(node, ast.For)
    ]
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "cutlass.range_constexpr(4)"
    retained_names = [
        stmt.targets[0].id
        for stmt in emitted.prefix
        if isinstance(stmt.targets[0], ast.Name)
    ]
    assert retained_names == ["row", "offset"]
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        for stmt in emitted.prefix
        for node in ast.walk(stmt)
    )


@pytest.mark.parametrize(
    "prefix,body",
    [
        ("row = (indices.iterator + k_step).load()", "observed.append(row)"),
        ("row = tid + slot", "observed.append(row)"),
        (
            "row = unknown(indices)\nvalue = indices.iterator.load()",
            "observed.append(value)",
        ),
        (
            "indices.iterator.store(0)\nrow = indices.iterator.load()",
            "observed.append(row)",
        ),
        ("row = indices.iterator.load()\nrow = row + 1", "observed.append(row)"),
        ("row = missing + indices.iterator.load()", "observed.append(row)"),
        ("row = indices.iterator.load()", "observed.append(tid)"),
        ("tid = indices.iterator.load()", "observed.append(tid)"),
        (
            "row = tid\ntid = indices.iterator.load()",
            "observed.append(row)",
        ),
        (
            "row = cutlass.Int32(indices.iterator.load() + counts[0])",
            "observed.append(row)",
        ),
    ],
)
def test_rejects_varying_effectful_unavailable_or_unneeded_values(
    prefix: str, body: str
) -> None:
    assert _plan(prefix=prefix, body=body) is None


def test_rejects_unproven_readonly_tensor() -> None:
    assert (
        _plan(
            prefix="row = indices.iterator.load()",
            body="observed.append(row)",
            readonly=(),
        )
        is None
    )


@pytest.mark.parametrize(
    "guard", ["row_position + k_step < total", "counts.iterator.load() > 0"]
)
def test_rejects_varying_or_memory_dependent_slot_guard(guard: str) -> None:
    assert _plan(slot_guard=guard) is None


@pytest.mark.parametrize("slots,maximum", [(0, 16), (9, 16), (4, 3)])
def test_respects_static_slot_and_register_cache_budgets(
    slots: int, maximum: int
) -> None:
    assert _plan(slot_count=slots, max_cached_values=maximum) is None


def test_sixteen_slots_are_an_explicit_bounded_opt_in() -> None:
    assert _plan(slot_count=16) is None
    assert _plan(slot_count=16, max_slots=16) is not None
    assert _plan(slot_count=16, max_slots=16, max_cached_values=15) is None


def test_shared_dependency_dag_is_not_expanded_exponentially() -> None:
    prefix = ["value_0 = indices.iterator.load()"]
    prefix.extend(f"value_{i} = value_{i - 1} + value_{i - 1}" for i in range(1, 61))
    plan = _plan(prefix="\n".join(prefix), body="observed.append(value_60)")
    assert plan is not None
    emitted = plan.emit(_fresh(), execution_guard=_expr("True"))
    assert emitted.cached_names == ("value_60",)


def test_preserves_typed_cast_in_cached_value() -> None:
    plan = _plan(
        prefix="row = cutlass.Int64(indices.iterator.load())",
        body="observed.append(row)",
    )
    assert plan is not None
    emitted = plan.emit(_fresh(), execution_guard=_expr("True"))
    source = ast.unparse(ast.Module(body=emitted.prologue, type_ignores=[]))
    assert "cutlass.Int64" in source
    assert "make_rmem_tensor((4,), cutlass.Int64)" in source


def test_keeps_nonreadonly_load_fresh_while_caching_readonly_dependency() -> None:
    plan = _plan(
        prefix="index = counts.iterator.load()\nrow = (indices.iterator + index).load()",
        body="observed.append(row)",
        readonly=("counts",),
    )
    assert plan is not None
    emitted = plan.emit(_fresh(), execution_guard=_expr("True"))
    assert emitted.cached_names == ("index",)
    assert "indices.iterator" in ast.unparse(
        ast.Module(body=[*emitted.prefix], type_ignores=[])
    )


def test_unknown_result_dtype_caches_typed_source_and_retains_expression() -> None:
    plan = _plan(
        prefix="index = indices.iterator.load()\nrow = index + 1",
        body="observed.append(row)",
    )
    assert plan is not None
    emitted = plan.emit(_fresh(), execution_guard=_expr("True"))
    assert emitted.cached_names == ("index",)
    assert ast.unparse(emitted.prefix[-1].value) == "index + 1"


@pytest.mark.parametrize("op", ["-", "+", "~"])
def test_does_not_assume_boolean_arithmetic_preserves_boolean_dtype(op: str) -> None:
    assert (
        _plan(
            prefix=f"row = {op}(indices.iterator.load() < 0)",
            body="observed.append(row)",
        )
        is None
    )


def test_unproven_subscript_remains_fresh_with_readonly_pointer_ancestor() -> None:
    plan = _plan(
        prefix="index = indices.iterator.load()\nrow = cutlass.Int32(index + counts[0])",
        body="observed.append(row)",
        readonly=("indices",),
    )
    assert plan is not None
    emitted = plan.emit(_fresh(), execution_guard=_expr("True"))
    assert emitted.cached_names == ("index",)
    assert "counts[0]" in ast.unparse(emitted.prefix[-1].value)


def test_extended_ast_inputs_are_supported_and_not_mutated() -> None:
    prefix = [
        cast(
            "ast.Assign",
            statement_from_string(
                "row = (indices.iterator + cutlass.Int32(slot) * cutlass.Int32(indices.layout.stride[0])).load()"
            ),
        )
    ]
    snapshot = ast.dump(prefix[0])
    plan = plan_scalar_recipe_cache(
        prefix,
        ast.parse("observed.append(row)").body,
        slot_name="slot",
        slot_count=4,
        slot_prelude=[],
        slot_guard=cast("ast.expr", expr_from_string("slot < 3")),
        boundary_names={"indices"},
        varying_names={"k_step"},
        tensor_dtypes={"indices": "cutlass.Int32"},
        readonly_tensors={"indices"},
    )
    assert plan is not None
    emitted = plan.emit(
        _fresh(), execution_guard=cast("ast.expr", expr_from_string("K > 0"))
    )
    assert emitted.cached_names == ("row",)
    assert ast.dump(prefix[0]) == snapshot
    assert "make_rmem_tensor((4,), cutlass.Int32)" in ast.unparse(
        ast.Module(body=emitted.prologue, type_ignores=[])
    )
