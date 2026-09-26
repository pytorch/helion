from __future__ import annotations

import ast
from itertools import count
import math
import struct
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from test.test_cute_collective_vector_recipe import _Tensor

from helion._compiler.cute.collective_tcgen05 import CollectiveOperandSmem
from helion._compiler.cute.collective_tcgen05 import CollectiveTcgen05Plan
from helion._compiler.cute.collective_tcgen05 import CollectiveTmemResource
from helion._compiler.cute.collective_tmem_operand import CollectiveTmemOperand


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _half(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def _bf16(value: float) -> float:
    bits = struct.unpack("I", struct.pack("f", value))[0]
    if bits & 0x7FFFFFFF > 0x7F800000:
        bits = (bits & 0xFFFF0000) | 0x00400000
    else:
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits))[0]


@pytest.mark.parametrize("bm", [64, 128])
@pytest.mark.parametrize("bk", [16, 32, 64, 128])
def test_store_ownership_covers_exactly_one_full_a_tile(bm: int, bk: int) -> None:
    plan = CollectiveTmemOperand("tile", "tid", bm, bk, "cutlass.BFloat16", 64)
    m, k = (
        compile(ast.Expression(expr), "<coordinate>", "eval")
        for expr in plan.coordinates("v")
    )
    actual = {
        (eval(m, {"tid": tid, "v": v}), eval(k, {"tid": tid, "v": v}))
        for tid in range(128)
        for v in range(plan.values_per_thread)
    }
    assert len(actual) == bm * bk
    assert actual == {(row, column) for row in range(bm) for column in range(bk)}
    assert plan.columns == bk // 2


@pytest.mark.parametrize("bm", [64, 128])
@pytest.mark.parametrize("bk", [16, 64, 128])
@pytest.mark.parametrize("dtype", ["Float16", "BFloat16"])
@pytest.mark.parametrize("mode", ["scalar", "vector", "synthetic_tail"])
def test_recipe_preserves_gathers_masks_casts_and_inactive_reads(
    bm: int, bk: int, dtype: str, mode: str
) -> None:
    # Duplicate gathered rows, a row tail, a K tail, exact signed zero and
    # nonfinite operands are all observed through the original scalar recipe.
    cast_value = _half if dtype == "Float16" else _bf16
    count_rows = bm - 3
    limit = bk - 3
    width = bk + 5
    scalars = [-0.0, 0.0, math.inf, -math.inf, math.nan, 0.33325, -1.25]
    a = _Tensor([cast_value(scalars[i % 7]) for i in range(5 * width)], (5, width))
    rows = _Tensor([i % 5 for i in range(count_rows)], (count_rows,))
    statements = cast(
        "list[ast.Assign]",
        ast.parse(f"""
valid_row = m < count_rows
row = (rows.iterator + m).load() if valid_row else 0
column = cutlass.Int32(origin + k)
active = valid_row and column < limit
loaded = (a.iterator + row * width + column).load() if active else cutlass.{dtype}(0.0)
product = cutlass.Float32(cutlass.Float32(loaded) * cutlass.Float32(0.5))
value = cutlass.{dtype}(product)
""").body,
    )
    counter = count()
    plan = CollectiveTmemOperand("tile", "tid", bm, bk, f"cutlass.{dtype}", 64)
    loop = plan.emit_recipe(
        statements,
        _expr("value"),
        local_m="m",
        local_k="k",
        k_offset="origin",
        static_k_extent=limit if mode == "synthetic_tail" else None,
        vectorize=mode == "vector",
        tensors={},
        aligned_names={},
        fresh_name=lambda name: f"{name}_{next(counter)}",
    )
    code = compile(
        ast.fix_missing_locations(ast.Module([loop], [])), "<recipe>", "exec"
    )
    environment: dict[str, object] = {
        "a": a,
        "rows": rows,
        "count_rows": count_rows,
        "limit": limit,
        "width": width,
        "origin": 0,
        "cutlass": SimpleNamespace(
            range_constexpr=range,
            Int32=int,
            Float32=_f32,
            Float16=_half,
            BFloat16=_bf16,
        ),
    }
    coords = [
        compile(ast.Expression(expr), "<coordinate>", "eval")
        for expr in plan.coordinates("v")
    ]
    result = {}
    for tid in range(128):
        environment.update(tid=tid, tile_a_values=[None] * plan.values_per_thread)
        exec(code, environment)
        values = cast("list[float]", environment["tile_a_values"])
        assert all(value is not None for value in values)
        for v, value in enumerate(values):
            coordinate = tuple(eval(expr, {"tid": tid, "v": v}) for expr in coords)
            assert coordinate not in result
            result[coordinate] = value
    actual_reads = (set(a.loads), set(rows.loads))
    a.loads.clear()
    rows.loads.clear()
    original = compile(
        ast.fix_missing_locations(ast.Module([*statements], [])), "<scalar>", "exec"
    )
    for m in range(bm):
        for k in range(bk):
            environment.update(m=m, k=k)
            if mode == "synthetic_tail" and k >= limit:
                expected = cast_value(0.0)
            else:
                exec(original, environment)
                expected = cast_value(cast("float", environment["value"]))
            assert struct.pack("f", result[m, k]) == struct.pack("f", expected)
    assert actual_reads == (set(a.loads), set(rows.loads))


@pytest.mark.parametrize("bm", [64, 128])
@pytest.mark.parametrize("bk", [16, 128])
@pytest.mark.parametrize("vectorize", [False, True])
def test_recipe_coordinates_retain_int32_overflow_before_cast(
    bm: int, bk: int, vectorize: bool
) -> None:
    plan = CollectiveTmemOperand("tile", "tid", bm, bk, "cutlass.Float16", 64)
    statements = cast(
        "list[ast.Assign]",
        ast.parse("""
value = 1.0 if k * 1073741824 < 0 else -0.0
value2 = 2.0 if m * 1073741824 < 0 else value
""").body,
    )
    counter = count()
    loop = plan.emit_recipe(
        statements,
        _expr("value2"),
        local_m="m",
        local_k="k",
        k_offset="origin",
        static_k_extent=None,
        vectorize=vectorize,
        tensors={},
        aligned_names={},
        fresh_name=lambda name: f"{name}_{next(counter)}",
    )
    code = compile(
        ast.fix_missing_locations(ast.Module([loop], [])), "<typed-recipe>", "exec"
    )
    coords = [
        compile(ast.Expression(expr), "<coordinate>", "eval")
        for expr in plan.coordinates("v")
    ]
    namespace: dict[str, object] = {
        "cutlass": SimpleNamespace(Int32=np.int32, Float16=float, range_constexpr=range)
    }
    with np.errstate(over="ignore"):
        for tid in range(128):
            namespace.update(
                tid=np.int32(tid), tile_a_values=[None] * plan.values_per_thread
            )
            exec(code, namespace)
            for v, actual in enumerate(cast("list[float]", namespace["tile_a_values"])):
                m, k = (np.int32(eval(expr, {"tid": tid, "v": v})) for expr in coords)
                expected = (
                    2.0
                    if m * np.int32(1073741824) < 0
                    else (1.0 if k * np.int32(1073741824) < 0 else -0.0)
                )
                assert struct.pack("f", actual) == struct.pack("f", expected)


@pytest.mark.parametrize("bm", [64, 128])
@pytest.mark.parametrize("bn", [32, 64])
def test_tmem_a_reservation_is_disjoint_and_pooled_after_accumulators(
    bm: int, bn: int
) -> None:
    resource = CollectiveTmemResource("resource", bn)
    operand_pool = CollectiveOperandSmem("pool")
    plans = [
        CollectiveTcgen05Plan(
            f"stage{bk}",
            "tid",
            bm,
            bn,
            bk,
            "cutlass.Float16",
            resource,
            operands=operand_pool,
            a_in_tmem=True,
        )
        for bk in (16, 128, 32)
    ]
    assert resource.operand_columns == 64
    assert resource.allocation_columns == 128
    for plan in plans:
        assert plan.tmem_a is not None
        assert plan.tmem_a.column_offset >= plan.bn
        assert (
            plan.tmem_a.column_offset + plan.tmem_a.columns
            <= resource.allocation_columns
        )
        code = ast.unparse(ast.Module(plan.setup("physical_tid"), []))
        assert "OperandSource.TMEM" in code
        assert "pool_ap" not in code
        assert f"resource_ptr + {bn}" in code
        assert f"({bm}, {plan.bk})" in code
        publish = ast.unparse(ast.Module(plan.tmem_a.publish(), []))
        assert publish.index("cute.copy(") < publish.index(
            "fence_view_async_tmem_store"
        )
    assert operand_pool.a_bytes == 0
    assert "pool_ap" not in ast.unparse(ast.Module(operand_pool.prologue(), []))
    prologue = ast.unparse(ast.Module(resource.prologue(), []))
    assert ".allocate(128)" in prologue
    assert prologue.count(".allocate(") == 1
    assert ".relinquish_alloc_permit()" in prologue


@pytest.mark.parametrize(
    "bm,bk,dtype,offset",
    [
        (32, 64, "cutlass.Float16", 64),
        (64, 256, "cutlass.Float16", 64),
        (64, 64, "cutlass.TFloat32", 64),
        (64, 64, "cutlass.Float32", 64),
        (64, 64, "cutlass.Float16", 16),
    ],
)
def test_unsupported_tmem_a_layout_is_rejected(
    bm: int, bk: int, dtype: str, offset: int
) -> None:
    with pytest.raises(ValueError):
        CollectiveTmemOperand("tile", "tid", bm, bk, dtype, offset)
