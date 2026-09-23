from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from .test_cute_bounded_loop_cache import _cuda_range
from .test_cute_bounded_loop_cache import _Fragment
from .test_cute_bounded_loop_cache import _i32
from .test_cute_bounded_loop_cache import _Tensor
from helion._compiler.cute.bounded_loop_cache import cache_single_tile_loads
from helion._compiler.cute.bounded_loop_cache import specialize_single_tile_sweeps
from helion._compiler.cute.duplicate_tile_maxima import deduplicate_bounded_maxima

SOURCE = """
def kernel(x, out, n, reset):
    for tile in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(8)):
        a = cutlass.Float32(float('-inf'))
        b = cutlass.Float32(float('-inf'))
        for lane in range(2):
            base = tile + cutlass.Int32(lane) * 4
            for vec in cutlass.range_constexpr(4):
                column = base + cutlass.Int32(vec)
                valid = column < n
                raw = (x.iterator + column).load() if valid else cutlass.Float32(0)
                value = cutlass.Float32(raw)
                a = cute.arch.fmax(a, value if valid else cutlass.Float32(float('-inf')))
            first = tile == 0
            first_bool = cutlass.Boolean(first)
            for vec in cutlass.range_constexpr(4):
                column = base + cutlass.Int32(vec)
                valid = column < n
                raw = (x.iterator + column).load() if valid else cutlass.Float32(0)
                value = cutlass.Float32(raw)
                selected = cutlass.Float32(value) if first_bool else cutlass.Float32(float('-inf'))
                b = cute.arch.fmax(b, selected if valid else cutlass.Float32(float('-inf')))
        left = _cute_grouped_reduce_shared_two_stage(a, 'max', cutlass.Float32(float('-inf')), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), pre=1, group_span=128, group_count=1)
        right = _cute_grouped_reduce_shared_two_stage(b, 'max', cutlass.Float32(float('-inf')), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), pre=1, group_span=128, group_count=1)
    for tile in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(8)):
        for lane in range(2):
            base = tile + cutlass.Int32(lane) * 4
            for vec in cutlass.range_constexpr(4):
                column = base + cutlass.Int32(vec)
                valid = column < n
                raw = (x.iterator + column).load() if valid else cutlass.Float32(0)
                if valid:
                    (out.iterator + column).store(cutlass.Float32(raw + left + right))
"""


def _prepare(source=SOURCE):
    function = ast.parse(source).body[0]
    prepared = specialize_single_tile_sweeps(
        function.body,
        integer_arguments={"n"},
        constexpr_values={},
        launch_block=(128, 1, 1),
    )
    assert prepared is not None
    cached = cache_single_tile_loads(
        prepared.body,
        prepared.plan,
        argument_names={"x", "out", "n", "reset"},
        constexpr_values={},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        proven_disjoint_tensor_pairs={frozenset({"x", "out"})},
        rename_groups={},
    )
    assert cached is not None
    return function, cached


def _rewrite(cached):
    return deduplicate_bounded_maxima(
        cached,
        argument_names={"x", "out", "n", "reset"},
        constexpr_values={},
        rename_groups={},
    )


def _count(body):
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_grouped_reduce_shared_two_stage"
        for statement in body
        for node in ast.walk(statement)
    )


def _run(function, values):
    calls = []

    def reduce(value, *args, **kwargs):
        calls.append(value)
        return np.float32(value)

    namespace = {
        "range": _cuda_range,
        "_cute_grouped_reduce_shared_two_stage": reduce,
        "cutlass": SimpleNamespace(
            Float32=np.float32,
            Int32=_i32,
            Boolean=bool,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=_Fragment,
            arch=SimpleNamespace(fmax=lambda a, b: np.float32(np.maximum(a, b))),
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "<duplicate-max>",
            "exec",
        ),
        namespace,
    )
    x = _Tensor(enumerate(values), (1,))
    out = _Tensor(enumerate(np.full(len(values), -17, dtype=np.float32)), (1,))
    with np.errstate(all="ignore"):
        namespace[function.name](x, out, len(values), True)
    return np.array(list(out.values.values()), dtype=np.float32).tobytes(), len(calls)


@pytest.mark.parametrize("n", [1, 5, 8])
@pytest.mark.parametrize(
    "pattern",
    [
        [1.0, 2.0, 3.0],
        [0.0, -0.0],
        [-float("inf")],
        [float("inf"), 1.0],
        [float("nan"), -1.0],
    ],
)
def test_equal_reductions_preserve_values_nan_and_signed_zero(n, pattern):
    function, cached = _prepare()
    assert _count(cached.body) == 2
    function.body = _rewrite(cached)
    assert _count(function.body) == 1
    original = ast.parse(SOURCE).body[0]
    values = np.array((pattern * n)[:n], dtype=np.float32)
    before, before_calls = _run(original, values)
    after, after_calls = _run(function, values)
    assert after == before
    assert before_calls == 2 and after_calls == 1


@pytest.mark.parametrize(
    "old,new",
    [
        ("first = tile == 0", "first = tile == 1"),
        ("first = tile == 0", "first = reset"),
        (
            "b = cute.arch.fmax(b, selected",
            "b = cute.arch.fmax(b, cutlass.Float32(selected + 1)",
        ),
        ("b = cutlass.Float32(float('-inf'))", "b = cutlass.Float32(0)"),
        (
            "right = _cute_grouped",
            "left = cutlass.Float32(0)\n        right = _cute_grouped",
        ),
        (
            "first = tile == 0",
            "if reset:\n                b = cutlass.Float32(0)\n            first = tile == 0",
        ),
        ("group_count=1)\n    for tile", "group_count=2)\n    for tile"),
    ],
)
def test_changed_inputs_carries_geometry_and_conditional_resets_keep_collectives(
    old, new
):
    _, cached = _prepare(SOURCE.replace(old, new))
    before = ast.dump(ast.Module(body=cached.body, type_ignores=[]))
    result = _rewrite(cached)
    assert _count(result) == 2
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == before


def test_modified_producer_cannot_reuse_a_stale_load_fact():
    _, cached = _prepare()
    assert cached.origins
    cached.origins[0].statement.value = ast.parse(
        "cutlass.Float32(9)", mode="eval"
    ).body
    before = ast.dump(ast.Module(body=cached.body, type_ignores=[]))
    assert ast.dump(ast.Module(body=_rewrite(cached), type_ignores=[])) == before
