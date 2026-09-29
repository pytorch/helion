from __future__ import annotations

import ast
import dataclasses
import itertools
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from helion._compiler.cute.resident_reductions import ResidentReductionLayout
from helion._compiler.cute.resident_reductions import feature_index_expression
from helion._compiler.cute.resident_reductions import materialize_resident_reductions

_FEATURES = 16
_TENSORS = ("x", "dy", "weight", "mean", "scale", "out", "partial")
_LAYOUT = ResidentReductionLayout(0, _FEATURES, 1, _FEATURES, 1, "column")


def _source(*, rows: int = 3, width: int = 1, bounded: bool = False) -> str:
    index = feature_index_expression(
        dataclasses.replace(_LAYOUT, vector_width=width), "lane"
    )
    end = (
        "cutlass.Int32(cutlass.Int32(8) if cutlass.Int32(8) < cutlass.Int32(row_limit) else cutlass.Int32(row_limit))"
        if bounded
        else f"cutlass.Int32({rows})"
    )
    return f"""
for lane in range({_FEATURES}):
    column = {index}
    carry = cutlass.Float32(initial)
    w = (weight.iterator + cutlass.Int32(column)).load()
    for row in range(cutlass.Int32(0), {end}):
        old_carry = carry
        x_value = (x.iterator + cutlass.Int32(row) * {_FEATURES} + cutlass.Int32(column)).load()
        gradient = (dy.iterator + cutlass.Int32(row) * {_FEATURES} + cutlass.Int32(column)).load()
        mean_value = (mean.iterator + cutlass.Int32(row)).load()
        scale_value = (scale.iterator + cutlass.Int32(row)).load()
        centered = x_value - mean_value
        normalized = centered * scale_value
        increment = gradient * normalized
        carry = old_carry + increment
        weighted = gradient * w
        product = weighted * normalized
        reduced_product = cutlass.Float32(_helion_lane_reduce(product, 'sum', cutlass.Float32(0), 1, 1, 0, '', 1, 1))
        reduced_gradient = cutlass.Float32(_helion_lane_reduce(weighted, 'sum', cutlass.Float32(0), 1, 1, 0, '', 1, 1))
        coefficient = reduced_product / {_FEATURES}
        average = reduced_gradient / {_FEATURES}
        first = normalized * coefficient
        both = first + average
        centered_gradient = weighted - both
        result = centered_gradient * scale_value
        if column < limit:
            (out.iterator + cutlass.Int32(row) * {_FEATURES} + cutlass.Int32(column)).store(result)
    (partial.iterator + cutlass.Int32(column)).store(carry)
"""


def _rewrite(
    source: str,
    *,
    width: int = 1,
    disjoint: bool = True,
    alignment: int = 16,
    group_rows: int = 1,
    pipelined: bool = False,
    pipeline_depth: int = 2,
    local_tree: bool = False,
    row_schedule: str = "batched",
    pack_output: bool = False,
    shared_memory_budget: int = 232448,
) -> str:
    counter = itertools.count()
    result = materialize_resident_reductions(
        ast.parse(source).body,
        layouts={"lane": dataclasses.replace(_LAYOUT, vector_width=width)},
        tensor_dtypes=dict.fromkeys(_TENSORS, "cutlass.Float32"),
        tensor_strides={},
        tensor_alignments=dict.fromkeys(_TENSORS, alignment),
        group_rows=group_rows,
        pipelined=pipelined,
        pipeline_depth=pipeline_depth,
        local_tree=local_tree,
        row_schedule=row_schedule,
        pack_output=pack_output,
        shared_memory_budget=shared_memory_budget,
        disjoint_pairs={frozenset(pair) for pair in itertools.combinations(_TENSORS, 2)}
        if disjoint
        else set(),
        rename_groups={},
        new_var=lambda hint: f"{hint}_{next(counter)}",
        constexpr_values={},
        uniform_names={*_TENSORS, "initial", "limit", "row_limit"},
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


class _Pointer:
    def __init__(self, tensor: _Tensor, offset: int = 0) -> None:
        self.tensor = tensor
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.tensor, self.offset + int(offset))

    def load(self) -> np.float32:
        assert 0 <= self.offset < self.tensor.values.size
        self.tensor.reads += 1
        return self.tensor.values.flat[self.offset]

    def store(self, value: np.float32) -> None:
        assert 0 <= self.offset < self.tensor.values.size
        self.tensor.values.flat[self.offset] = value


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.reads = 0

    @property
    def iterator(self) -> _Pointer:
        return _Pointer(self)


def _namespace() -> dict[str, object]:
    pending: list[list[tuple[_Tensor, _Tensor, int, int, int]]] = []
    current: list[tuple[_Tensor, _Tensor, int, int, int]] = []

    def copy_async(
        source: _Tensor,
        destination: _Tensor,
        source_offset: int,
        destination_offset: int,
        width: int,
    ) -> None:
        current.append((source, destination, source_offset, destination_offset, width))

    def commit_group() -> None:
        pending.append(current.copy())
        current.clear()

    def wait_group(remaining: int) -> None:
        while len(pending) > remaining:
            for (
                source,
                destination,
                source_offset,
                destination_offset,
                width,
            ) in pending.pop(0):
                for lane in range(width):
                    (destination.iterator + destination_offset + lane).store(
                        (source.iterator + source_offset + lane).load()
                    )

    def load_vector(tensor: _Tensor, offset: int, width: int) -> np.ndarray:
        return np.array(
            [(tensor.iterator + offset + lane).load() for lane in range(width)],
            dtype=tensor.values.dtype,
        )

    def store_vector(
        tensor: _Tensor, offset: int, fragment: np.ndarray, width: int
    ) -> None:
        for lane in range(width):
            (tensor.iterator + offset + lane).store(fragment[lane])

    return {
        "_cute_resident_sums": lambda values, threads: values.copy(),
        "_cute_resident_copy_async": copy_async,
        "_cute_resident_load_vector": load_vector,
        "_cute_resident_store_vector": store_vector,
        "cutlass": SimpleNamespace(
            Int32=np.int32,
            Int64=np.int64,
            Float32=np.float32,
            range_constexpr=range,
            const_expr=lambda value: value,
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: (0, 0, 0),
                alloc_smem=lambda dtype, size, alignment: (
                    _Tensor(np.empty(size, dtype=dtype)).iterator
                ),
                cp_async_commit_group=commit_group,
                cp_async_wait_group=wait_group,
                sync_threads=lambda: None,
            ),
            make_rmem_tensor=lambda extent, dtype: np.empty(extent, dtype=dtype),
            make_tensor=lambda pointer, layout: pointer.tensor,
            make_layout=lambda size: size,
        ),
        "pending_copies": pending,
        "uncommitted_copies": current,
    }


@pytest.mark.parametrize(
    "rows,group_rows",
    [
        (0, 1),
        (1, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (3, 2),
        (4, 2),
        (0, 4),
        (1, 4),
        (3, 4),
        (4, 4),
    ],
)
@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("initial", [0.0, -0.0, 0.75])
@pytest.mark.parametrize("pipelined", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("reciprocal", [False, True])
def test_resident_keeps_carries_and_masked_output_semantics(
    rows: int,
    group_rows: int,
    width: int,
    initial: float,
    pipelined: bool,
    bounded: bool,
    reciprocal: bool,
) -> None:
    original = _source(rows=rows, width=width, bounded=bounded)
    if reciprocal:
        original = original.replace(
            "coefficient = reduced_product / 16",
            "reciprocal_product = 0.0625\n        coefficient = reduced_product * reciprocal_product",
        ).replace(
            "average = reduced_gradient / 16",
            "reciprocal_gradient = 0.0625\n        average = reduced_gradient * reciprocal_gradient",
        )
    source = _rewrite(
        original,
        width=width,
        group_rows=group_rows,
        pipelined=pipelined,
    )
    assert "resident_state" in source
    assert "_helion_lane_reduce" not in source
    if width > 1:
        if not pipelined:
            assert "_cute_resident_load_vector(x," in source
            assert "_cute_resident_load_vector(dy," in source
        assert "_cute_resident_store_vector(partial," in source
        assert "_cute_resident_store_vector(out," not in source
    rng = np.random.default_rng(2813)
    for _iteration in range(3):
        # Every invocation changes all inputs, including the row statistics.
        x = rng.normal(size=(rows, _FEATURES)).astype(np.float32)
        dy = rng.normal(size=x.shape).astype(np.float32)
        weight = rng.normal(size=_FEATURES).astype(np.float32)
        mean = rng.normal(size=rows).astype(np.float32)
        scale = rng.uniform(0.5, 2, size=rows).astype(np.float32)
        out = np.full(x.shape, np.float32(17))
        partial = np.full(_FEATURES, np.float32(19))
        tensors = {
            name: _Tensor(value)
            for name, value in zip(
                _TENSORS, (x, dy, weight, mean, scale, out, partial), strict=True
            )
        }
        namespace = (
            _namespace()
            | tensors
            | {"initial": initial, "limit": 11, "row_limit": rows}
        )
        exec(compile(source, "<resident reduction>", "exec"), namespace)
        normalized = (x - mean[:, None]) * scale[:, None]
        weighted = dy * weight
        expected = (
            weighted
            - (
                normalized * (normalized * weighted).sum(-1, keepdims=True) / _FEATURES
                + weighted.sum(-1, keepdims=True) / _FEATURES
            )
        ) * scale[:, None]
        expected[:, 11:] = np.float32(17)
        expected_partial = np.full(_FEATURES, np.float32(initial))
        for row in range(rows):
            expected_partial = expected_partial + dy[row] * normalized[row]
        np.testing.assert_allclose(out, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(partial, expected_partial)
        if rows == 0:
            np.testing.assert_array_equal(
                np.signbit(partial), np.signbit(expected_partial)
            )
        assert tensors["x"].reads == rows * _FEATURES
        assert tensors["dy"].reads == rows * _FEATURES
        assert tensors["weight"].reads == _FEATURES
        assert tensors["mean"].reads == rows
        assert tensors["scale"].reads == rows
        assert not namespace["pending_copies"]
        assert not namespace["uncommitted_copies"]


@pytest.mark.parametrize(
    "begin,end",
    [
        (2**31 - 2, 2**31 - 1),
        (2**31 - 5, 2**31 - 1),
        (2**31 - 6, 2**31 - 1),
        (-(2**31), -(2**31) + 1),
        (-(2**31), -(2**31) + 5),
        (2**31 - 1, 2**31 - 1),
    ],
)
@pytest.mark.parametrize("group_rows", [1, 2, 4])
@pytest.mark.parametrize("pipelined", [False, True])
def test_grouped_rows_keep_int32_endpoint_semantics(
    begin: int, end: int, group_rows: int, pipelined: bool
) -> None:
    original = _source(width=4).replace(
        "range(cutlass.Int32(0), cutlass.Int32(3))",
        f"range(cutlass.Int32({begin}), cutlass.Int32({end}))",
    )
    # Rebase valid original accesses into small buffers while retaining the
    # exact Int32 row domain that grouped scheduling must preserve.
    original = original.replace(
        "cutlass.Int32(row)", f"(cutlass.Int32(row) - cutlass.Int32({begin}))"
    )
    rewritten = _rewrite(original, width=4, group_rows=group_rows, pipelined=pipelined)
    assert "resident_state" in rewritten
    results = []
    rows = end - begin
    for source in (original, rewritten):
        shape = (rows + group_rows, _FEATURES)
        tensors = dict(
            zip(
                _TENSORS,
                (
                    _Tensor(np.ones(shape, dtype=np.float32)),
                    _Tensor(np.ones(shape, dtype=np.float32)),
                    _Tensor(np.zeros(_FEATURES, dtype=np.float32)),
                    _Tensor(np.zeros(shape[0], dtype=np.float32)),
                    _Tensor(np.ones(shape[0], dtype=np.float32)),
                    _Tensor(np.full(shape, 17, dtype=np.float32)),
                    _Tensor(np.full(_FEATURES, 19, dtype=np.float32)),
                ),
                strict=True,
            )
        )
        namespace = (
            _namespace()
            | tensors
            | {
                "initial": -0.0,
                "limit": _FEATURES,
                # The zero weight makes both reduction inputs zero, so this
                # reference executes the original serial schedule exactly.
                "_helion_lane_reduce": lambda value, *args: np.float32(0),
            }
        )
        with np.errstate(over="raise", invalid="raise"):
            exec(compile(source, "<resident Int32 endpoints>", "exec"), namespace)
        assert not namespace["pending_copies"]
        assert not namespace["uncommitted_copies"]
        results.append(tensors)
    for name in _TENSORS:
        before, after = (result[name] for result in results)
        np.testing.assert_array_equal(after.values, before.values)
        np.testing.assert_array_equal(
            np.signbit(after.values), np.signbit(before.values)
        )
        if name in {"mean", "scale"}:
            assert before.reads == rows * _FEATURES
            assert after.reads == rows
        else:
            assert after.reads == before.reads


@pytest.mark.parametrize(
    "before,after",
    [
        ("old_carry = carry", "old_carry = carry\n        opaque(x)"),
        ("old_carry = carry", "old_carry = carry\n        cute.arch.sync_threads()"),
        ("w = (weight.iterator + cutlass.Int32(column)).load()", "w = weight[column]"),
        ("carry = cutlass.Float32(initial)", "carry = cutlass.Float64(initial)"),
        ("carry = old_carry + increment", "carry = old_carry + reduced_product"),
        (
            "result = centered_gradient * scale_value",
            "result = centered_gradient * carry",
        ),
        (
            "(partial.iterator + cutlass.Int32(column)).store(carry)",
            "(partial.iterator + cutlass.Int32(column)).store(result)",
        ),
        (
            "for row in range(cutlass.Int32(0), cutlass.Int32(3)):",
            "for row in range(cutlass.Int32(0), cutlass.Int32(3), 2):",
        ),
        (
            "for row in range(cutlass.Int32(0), cutlass.Int32(3)):",
            "for row in range(cutlass.Int32(column), cutlass.Int32(column + 3)):",
        ),
        (
            "for row in range(cutlass.Int32(0), cutlass.Int32(3)):",
            "for row in range(cutlass.Int32(cute.arch.thread_idx()[0]), cutlass.Int32(cute.arch.thread_idx()[0] + 3)):",
        ),
        (
            "for row in range(cutlass.Int32(0), cutlass.Int32(3)):",
            "for row in range(cutlass.Int32(mean[0]), cutlass.Int32(mean[0] + 3)):",
        ),
        (
            "for row in range(cutlass.Int32(0), cutlass.Int32(3)):",
            "for row in range(2147483648, 2147483651):",
        ),
        (
            "for row in range(cutlass.Int32(0), cutlass.Int32(3)):",
            "for row in range(cutlass.Int64(row_limit), cutlass.Int64(row_limit) + 3):",
        ),
        (
            "for row in range(cutlass.Int32(0), cutlass.Int32(3)):",
            "for row in range(cutlass.Int32(row_limit) + 2147483648, cutlass.Int32(row_limit) + 2147483651):",
        ),
        ("old_carry = carry", "old_carry = stale\n        stale = carry"),
        ("old_carry = carry", "old_carry = carry\n        old_carry = old_carry + 1"),
        (
            "cutlass.Int32(row) * 16 + cutlass.Int32(column)).store",
            "cutlass.Int32(row) * 8 + cutlass.Int32(column)).store",
        ),
        (
            "cutlass.Int32(row) * 16 + cutlass.Int32(column)).store",
            "cutlass.Int32(row) * 2147483648 + cutlass.Int32(column)).store",
        ),
        (
            "cutlass.Int32(row) * 16 + cutlass.Int32(column)).store",
            "cutlass.Int64(row) * 16 + cutlass.Int32(column)).store",
        ),
        (
            "cutlass.Int32(row) * 16 + cutlass.Int32(column)).store",
            "cutlass.Int32(row) * 16 + cutlass.Int32(column) + cutlass.Int32(cute.arch.thread_idx()[0])).store",
        ),
        (
            "normalized = centered * scale_value",
            "normalized = centered * scale_value\n        flag = column < limit",
        ),
    ],
)
def test_unknown_effects_aliases_and_loop_carried_values_decline(
    before: str, after: str
) -> None:
    source = _source()
    assert before in source
    source = source.replace(before, after)
    if "flag =" in source:
        source = source.replace("if column < limit:", "if flag:")
    rewritten = _rewrite(source)
    assert "resident_state" not in rewritten
    assert ast.dump(ast.parse(rewritten)) == ast.dump(ast.parse(source))


def test_resident_requires_nonaliasing_outputs() -> None:
    assert "resident_state" not in _rewrite(_source(), disjoint=False)


def test_resident_declines_alias_rebinding() -> None:
    source = _source().replace("    carry =", "    out = x\n    carry =", 1)
    assert "resident_state" not in _rewrite(source)


def test_resident_handles_partial_row_groups() -> None:
    assert "resident_state" in _rewrite(_source(rows=3), group_rows=2)


def test_resident_pipeline_declines_when_shared_memory_exceeds_budget() -> None:
    assert "resident_state" not in _rewrite(
        _source(), pipelined=True, shared_memory_budget=1
    )


@pytest.mark.parametrize("threads", [32, 64, 128, 256])
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_partial_vector_feature_layout_covers_every_element_once(
    threads: int, width: int
) -> None:
    layout = ResidentReductionLayout(0, 4096, threads, 4096 // threads, width, "column")
    expression = compile(feature_index_expression(layout, "lane"), "<layout>", "eval")
    elements = []
    for thread in range(threads):
        namespace = _namespace()
        cast("SimpleNamespace", namespace["cute"]).arch.thread_idx = (
            lambda thread=thread: (
                thread,
                0,
                0,
            )
        )
        for lane in range(layout.lane_extent):
            elements.append(int(eval(expression, namespace | {"lane": lane})))
    assert sorted(elements) == list(range(layout.feature_extent))


def test_resident_vector_copy_requires_base_alignment() -> None:
    result = _rewrite(_source(width=4), width=4, alignment=4)
    assert "resident_state" in result
    assert "_cute_resident_load_vector" not in result
    assert "_cute_resident_store_vector" not in result


def test_resident_vector_copy_does_not_hoist_conditional_load() -> None:
    source = _source(width=4).replace(
        "x_value = (x.iterator + cutlass.Int32(row) * 16 + cutlass.Int32(column)).load()",
        "x_value = (x.iterator + cutlass.Int32(row) * 16 + cutlass.Int32(column)).load() if column < limit else cutlass.Float32(0)",
    )
    result = _rewrite(source, width=4)
    assert "resident_state" in result
    assert "_cute_resident_load_vector(x," not in result
    assert "_cute_resident_load_vector(dy," in result
