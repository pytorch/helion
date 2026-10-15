"""Semantic and refusal proofs for online packet-to-chunk folding."""

from __future__ import annotations

import ast
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from helion._compiler.cute.online_chunk_reduction import lower_online_chunk_reduction
from helion._compiler.cute.uniform_packet_mask import uniform_packet_mask

SOURCE = """
maximum = cutlass.Float32(float('-inf'))
denominator = cutlass.Float32(0)
for block in range(0, extent, 8):
    for packet in range(2):
        base = block + packet * 4
        lane = cutlass.Int32(0)
        max_acc = cutlass.Float32(float('-inf'))
        cache = cute.make_rmem_tensor(4, cutlass.Float32)
        for element in cutlass.range_constexpr(4):
            index = base + cutlass.Int32(element)
            mask = index < extent
            loaded = (x.iterator + index).load() if mask else cutlass.Float32(0)
            value = cutlass.Float32(loaded) if mask else cutlass.Float32(float('-inf'))
            cache[element] = cutlass.Float32(value)
            max_acc = cute.arch.fmax(max_acc, cutlass.Float32(value) if mask else cutlass.Float32(float('-inf')), nan=True)
        max_result = _cute_grouped_reduce_warp(max_acc, 'max', cutlass.Float32(float('-inf')), lane, pre=1, group_span=32)
        chunk_maximum = cutlass.Float32(max_result)
        next_maximum = cute.math.max(cutlass.Float32(maximum), cutlass.Float32(chunk_maximum), propagate_nan=True)
        shift = maximum - next_maximum
        alpha = cute.math.exp2(cutlass.Float32(shift) * 1.4426950408889634)
        other_lane = cutlass.Int32(0)
        sum_acc = cutlass.Float32(0)
        for element in cutlass.range_constexpr(4):
            index = base + cutlass.Int32(element)
            mask = index < extent
            value = cache[element]
            shifted = value - next_maximum
            exponential = cute.math.exp2(cutlass.Float32(shifted) * 1.4426950408889634)
            sum_acc = sum_acc + (cutlass.Float32(exponential) if mask else cutlass.Float32(0))
        sum_result = _cute_grouped_reduce_warp(sum_acc, 'sum', cutlass.Float32(0), other_lane, pre=1, group_span=32)
        chunk_sum = cutlass.Float32(sum_result)
        denominator = cute.math.fma(denominator, alpha, chunk_sum)
        maximum = next_maximum
"""


class _Values:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def __sub__(self, value):
        return _Values(self.values - value)

    def __mul__(self, value):
        return _Values(self.values * value)

    def reduce(self, kind, init_val, reduction_profile):
        assert kind == "add"
        values = self.values.tolist()
        while len(values) > 1:
            values = [
                np.float32(values[index] + values[index + 1])
                for index in range(0, len(values), 2)
            ]
        return np.float32(init_val + values[0])


class _Register:
    def __init__(self, size, dtype):
        self.values = [dtype(0)] * size

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = np.float32(value)

    def load(self):
        return _Values(self.values)


class _Pointer:
    def __init__(self, data, index=0):
        self.data = data
        self.index = index

    def __add__(self, index):
        return _Pointer(self.data, self.index + index)

    def load(self):
        return self.data[self.index]


def _exp2(value, fastmath=False):
    if isinstance(value, _Values):
        result = np.exp2(value.values, dtype=np.float32)
        if fastmath:
            result[np.abs(result) < np.finfo(np.float32).tiny] = 0
        return _Values(result)
    return np.exp2(np.float32(value), dtype=np.float32)


def _fma(a, b, c):
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def _execute(source: str, data):
    module = ast.parse(source)
    namespace: dict[str, Any] = {
        "x": SimpleNamespace(iterator=_Pointer(data)),
        "extent": len(data),
        "cutlass": SimpleNamespace(
            Float32=np.float32,
            Int32=int,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=_Register,
            arch=SimpleNamespace(
                fmax=lambda a, b, nan=True: np.maximum(a, b),
            ),
            math=SimpleNamespace(
                max=lambda a, b, propagate_nan=True: np.maximum(a, b),
                exp2=_exp2,
                fma=_fma,
            ),
            ReductionOp=SimpleNamespace(ADD="add"),
        ),
        # Other lanes carry the reduction identity in this scalar simulation.
        "_cute_grouped_reduce_warp": lambda value, op, identity, lane, **kwargs: value,
    }
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exec(compile(module, "<online-chunk-test>", "exec"), namespace)
    return namespace["maximum"], namespace["denominator"]


def _changed(source: str):
    body = ast.parse(source).body
    changed = lower_online_chunk_reduction(body)
    return ast.unparse(ast.Module(body=changed, type_ignores=[]))


def test_complete_chunk_has_one_max_and_sum_reduce() -> None:
    result = _changed(SOURCE)
    assert "cute.make_rmem_tensor(8, cutlass.Float32)" in result
    assert "fastmath=True" in result
    assert result.count("_cute_grouped_reduce_warp(") == 2
    outer = next(node for node in ast.parse(result).body if isinstance(node, ast.For))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_grouped_reduce_warp"
        for statement in outer.body
        if isinstance(statement, ast.For)
        for node in ast.walk(statement)
    )


@pytest.mark.parametrize("length", [1, 3, 7, 8, 9, 17, 33, 129])
@pytest.mark.parametrize("spread", [0.5, 20.0, 150.0])
def test_max_rescaling_and_tail_match_fp64_logsumexp(length, spread) -> None:
    generator = np.random.default_rng(7316 + length)
    data = generator.normal(size=length).astype(np.float32) * spread
    original_max, original_denominator = _execute(SOURCE, data)
    maximum, denominator = _execute(_changed(SOURCE), data)
    expected_max = data.astype(np.float64).max()
    expected = expected_max + np.log(
        np.exp(data.astype(np.float64) - expected_max).sum()
    )
    assert maximum == original_max == expected_max
    np.testing.assert_allclose(
        maximum + np.log(denominator), expected, rtol=3e-6, atol=1e-5
    )
    np.testing.assert_allclose(
        maximum + np.log(denominator),
        original_max + np.log(original_denominator),
        rtol=3e-6,
        atol=1e-5,
    )


def test_nan_propagation_is_retained() -> None:
    data = np.array([2.0, -3.0, math.nan, 6.0, 1.0], dtype=np.float32)
    for source in (SOURCE, _changed(SOURCE)):
        maximum, denominator = _execute(source, data)
        assert math.isnan(maximum)
        assert math.isnan(denominator)


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace(
            "mask = index < extent\n            value = cache",
            "mask = index < extent - 1\n            value = cache",
        ),
        SOURCE.replace(
            "cache[element] = cutlass.Float32(value)",
            "cache[element] = cutlass.Float16(value)",
        ),
        SOURCE.replace("sum_acc = sum_acc +", "sum_acc = sum_acc -"),
        SOURCE.replace(
            "other_lane = cutlass.Int32(0)", "other_lane = cutlass.Int32(packet)"
        ),
        SOURCE.replace(
            "denominator = cute.math.fma(denominator, alpha, chunk_sum)",
            "denominator = cute.math.fma(denominator, alpha, chunk_sum + 1)",
        ),
        SOURCE.replace("maximum = next_maximum", "maximum = next_maximum + 1"),
        SOURCE.replace("for packet in range(2):", "for packet in range(3):"),
        SOURCE.replace(
            "chunk_sum = cutlass.Float32(sum_result)",
            "chunk_sum = cutlass.Float32(sum_result)\n        (x.iterator + base).store(chunk_sum)",
        ),
        SOURCE + "escaped = value\n",
        SOURCE.replace(
            "max_acc = cutlass.Float32(float('-inf'))", "max_acc = cutlass.Float32(100)"
        ),
        SOURCE.replace(
            "max_acc = cutlass.Float32(float('-inf'))",
            "max_acc = cutlass.Float32(float('-inf'))\n        max_acc = runtime_bound",
        ),
        SOURCE.replace(
            "sum_acc = cutlass.Float32(0)", "sum_acc = cutlass.Float32(100)"
        ),
        SOURCE.replace("loaded = (x.iterator", "loaded = maximum + (x.iterator"),
        SOURCE.replace(
            "value = cutlass.Float32(loaded)",
            "value = cutlass.Float32(max_acc + loaded)",
        ),
    ],
)
def test_unproved_changes_keep_original_ast(source) -> None:
    original = ast.parse(source)
    changed = ast.parse(_changed(source))
    assert ast.dump(changed, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def test_unproved_initial_denominator_keeps_exp2_fixup() -> None:
    source = SOURCE.replace(
        "maximum = cutlass.Float32(float('-inf'))", "maximum = cutlass.Float32(200)"
    )
    result = _changed(source)
    assert "cute.make_rmem_tensor(8, cutlass.Float32)" in result
    assert "fastmath=False" in result


@pytest.mark.parametrize("length", [4, 8, 12, 20, 36])
def test_aligned_packet_bounds_move_to_one_mask_branch(length) -> None:
    source = SOURCE.replace("extent", str(length))
    result = _changed(source)
    assert "if mask:" in result
    assert "mask = base <" in result
    data = np.random.default_rng(length).normal(size=length).astype(np.float32)
    before = _execute(source, data)
    after = _execute(result, data)
    np.testing.assert_allclose(after, before, rtol=3e-6, atol=1e-5)


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("extent", "19"),
        SOURCE.replace("extent", "20").replace("range(0, 20, 8)", "range(3, 20, 8)"),
        SOURCE.replace("extent", "20").replace(
            "base = block + packet * 4",
            "base = cutlass.Int32((block + packet * 4) / 1.0)",
        ),
        SOURCE.replace("extent", "20").replace("index < 20", "index % 2 == 0"),
    ],
)
def test_unaligned_or_element_varying_bounds_keep_mask_in_element_loop(source) -> None:
    result = _changed(source)
    assert "if mask:" not in result


@pytest.mark.parametrize(
    "prefix_source",
    [
        "base = 16\nbound = 32\nbound = runtime_bound",
        "base = 16\nbound = 32\nif active:\n    bound = runtime_bound",
        "base = 16\nbound = 32\nfor bound in range(runtime_bound, runtime_bound + 1):\n    pass",
    ],
)
def test_rebound_prefix_bound_keeps_executable_partial_packet(prefix_source) -> None:
    prefix = ast.parse(prefix_source).body
    loop = ast.parse(
        "for lane in cutlass.range_constexpr(4):\n"
        "    index = base + cutlass.Int32(lane)\n"
        "    mask = index < bound\n"
        "    row[lane] = cutlass.Float32(1.0) if mask else cutlass.Float32(float('-inf'))"
    ).body[0]
    assert isinstance(loop, ast.For)
    changed = uniform_packet_mask(prefix, loop, 4, "row", {})
    assert changed is None
    namespace = {
        "runtime_bound": 18,
        "active": True,
        "row": _Register(4, np.float32),
        "cutlass": SimpleNamespace(
            Float32=np.float32, Int32=int, range_constexpr=range
        ),
    }
    program = ast.Module(body=[*prefix, loop], type_ignores=[])
    exec(compile(program, "<rebound-prefix-witness>", "exec"), namespace)
    assert namespace["row"].values == [1.0, 1.0, -math.inf, -math.inf]


@pytest.mark.parametrize(
    "write",
    [
        "if active:\n    base = runtime_base",
        "for base in range(runtime_base, runtime_base + 1):\n    pass",
    ],
)
def test_branch_or_loop_prefix_rebind_invalidates_base_alignment(write) -> None:
    prefix = ast.parse("base = 16\nbound = 32\n" + write).body
    loop = ast.parse(
        "for lane in cutlass.range_constexpr(4):\n"
        "    index = base + cutlass.Int32(lane)\n"
        "    mask = index < bound\n"
        "    row[lane] = cutlass.Float32(1.0) if mask else cutlass.Float32(float('-inf'))"
    ).body[0]
    assert isinstance(loop, ast.For)
    assert uniform_packet_mask(prefix, loop, 4, "row", {}) is None
