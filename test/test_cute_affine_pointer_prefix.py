from __future__ import annotations

import ast
from unittest.mock import patch

from examples.aot_compile_example import add_2d
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_affine_vector_io import _SOURCE
from test.test_cute_affine_vector_io import _execute
from test.test_cute_affine_vector_io import _rewrite

import helion
from helion._testing import skipUnlessBackends


def _prefix_source(prefix: str) -> str:
    source = _SOURCE.replace(
        "address = cutlass.Int32(start) + indices_1", "address = indices_1"
    )
    for name in ("src", "out"):
        source = source.replace(f"{name}.iterator + ", f"{name}.iterator + {prefix} + ")
    return source


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 15, 16, 17, 32])
@pytest.mark.parametrize("row", range(3))
@pytest.mark.parametrize(
    "prefix",
    [
        "cutlass.Int32(start) * cutlass.Int32(16)",
        "cutlass.Int32(start) * cutlass.Int32(8) + cutlass.Int32(start) * cutlass.Int32(8)",
    ],
)
def test_row_and_batch_prefixes_preserve_values_and_tails(
    length: int, row: int, prefix: str
) -> None:
    source = _prefix_source(prefix)
    transformed, changed = _rewrite(source)
    assert changed == 1
    expected, before = _execute(ast.parse(source), row, length)
    actual, after = _execute(transformed, row, length)
    assert actual == expected
    assert before["scalar_loads"] == before["scalar_stores"] == length
    assert after["vector_loads"] == after["vector_stores"] == length // 4
    assert after["scalar_loads"] == after["scalar_stores"] == length % 4


@pytest.mark.parametrize(
    "prefix",
    [
        "cutlass.Int32(start * 4)",
        "cutlass.Int64(start) * cutlass.Int64(16)",
        "cutlass.Int32(vec_lane_1) * cutlass.Int32(16)",
        "cutlass.Int32(2)",
    ],
)
def test_unproved_or_lane_varying_prefix_keeps_original_ast(prefix: str) -> None:
    source = _prefix_source(prefix)
    transformed, changed = _rewrite(source)
    assert changed == 0
    assert ast.dump(transformed, include_attributes=False) == ast.dump(
        ast.parse(source), include_attributes=False
    )


def test_float_before_prefix_cast_is_not_assumed_divisible() -> None:
    source = _prefix_source("cutlass.Int32(start * 4)")
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), 1.25, 16)
    actual, after = _execute(transformed, 1.25, 16)
    assert changed == 0
    assert actual == expected
    assert after["vector_loads"] == 0


@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 8])
def test_dynamic_prefix_alignment_uses_scalar_fallback(start: int) -> None:
    source = _prefix_source("cutlass.Int32(start)")
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), start, 16)
    actual, after = _execute(transformed, start, 16)
    assert changed == 1
    assert actual == expected
    assert after["vector_loads"] == (4 if start % 4 == 0 else 0)


def test_dynamic_prefix_guard_follows_original_masks() -> None:
    source = _prefix_source("cutlass.Int32(8 // cutlass.Int32(start))").replace(
        "indices_1 < cutlass.Int32(length)",
        "indices_1 < cutlass.Int32(length) and start != 0",
    )
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), 0, 16)
    actual, after = _execute(transformed, 0, 16)
    assert changed == 1
    assert actual == expected
    assert not any(after.values())


def test_inactive_prefix_float_conversion_is_not_speculated() -> None:
    source = _prefix_source("cutlass.Int32(start) * cutlass.Int32(16)").replace(
        "indices_1 < cutlass.Int32(length)",
        "indices_1 < cutlass.Int32(length) and start == start",
    )
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), float("nan"), 16)
    actual, after = _execute(transformed, float("nan"), 16)
    assert changed == 1
    assert actual == expected
    assert not any(after.values())


@pytest.mark.parametrize("prefix", ["cutlass.Int32(4)", "cutlass.Int32(2147483644)"])
def test_prefix_pointer_additions_are_not_reassociated_to_int32(prefix: str) -> None:
    source = _prefix_source(prefix).replace(
        "address = indices_1", f"address = indices_1 - {prefix}"
    )
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), 0, 16)
    actual, after = _execute(transformed, 0, 16)
    assert changed == 1
    assert actual == expected
    assert after["vector_loads"] == 4
    for node in ast.walk(transformed):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and node.args
        ):
            pointer = node.args[0]
            assert isinstance(pointer, ast.BinOp)
            assert isinstance(pointer.left, ast.BinOp)
            assert isinstance(pointer.left.left, ast.Attribute)
            assert pointer.left.left.attr == "iterator"


@pytest.mark.parametrize("shape", [(256, 256), (1024, 1024), (4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@skipUnlessBackends(["cute"])
def test_original_dynamic_aot_add_uses_vector_memory(
    shape: tuple[int, int], dtype: torch.dtype
) -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            _target(),
            _mock_cuda_unavailable(),
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        ):
            kernel = helion.kernel(
                add_2d.fn, backend="cute", static_shapes=False, autotune_effort="none"
            )
            bound = kernel._bind_isolated(
                (
                    torch.empty(shape, dtype=dtype),
                    torch.empty(shape, dtype=dtype),
                )
            )
            width = 4 if dtype is torch.float32 else 8
            code = bound.to_code(
                helion.Config(
                    block_sizes=[1, 4096],
                    num_threads=[0, 256],
                    cute_vector_widths=[1, width],
                    cute_lane_layouts=["blocked", "blocked"],
                    cute_cluster_n=1,
                )
            )
            assert "cute.arch.load(" in code
            assert f"ir.VectorType.get([{width}]" in code
            assert "_cute_store_u" in code
    finally:
        torch.set_num_threads(previous)
