from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from helion._compiler.cute.merge_sibling_v_loops import merge_sibling_v_loops


def _run(source: str, value: float) -> tuple[np.floating[Any], list[str]]:
    calls: list[str] = []

    def convert(dtype: str, number: float) -> np.floating[Any]:
        calls.append(dtype)
        with np.errstate(all="ignore"):
            if dtype == "BFloat16":
                return np.float32(torch.tensor(number, dtype=torch.bfloat16).item())
            return {
                "Float16": np.float16,
                "Float32": np.float32,
                "Float64": np.float64,
            }[dtype](number)

    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            **{
                name: lambda number, name=name: convert(name, number)
                for name in ("Float16", "Float32", "Float64", "BFloat16")
            }
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                **{
                    name: lambda number, **kwargs: number
                    for name in (
                        "warp_reduction_sum",
                        "warp_reduction_max",
                        "warp_reduction_min",
                    )
                }
            )
        ),
        "value": value,
    }
    exec(source, namespace)
    return namespace["result"], calls


def _rewrite(source: str) -> str:
    body = merge_sibling_v_loops(ast.parse(source).body)
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _same_value(actual: np.floating[Any], expected: np.floating[Any]) -> None:
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if not np.isnan(expected):
        assert np.signbit(actual) == np.signbit(expected)


@pytest.mark.parametrize("dtype", ["Float16", "BFloat16", "Float64"])
@pytest.mark.parametrize("operation", ["sum", "max", "min"])
@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        1 + 2**-11,
        1 + 2**-8,
        70000.0,
        2**-25,
    ],
)
def test_warp_result_narrowing_then_widening_is_preserved(
    dtype: str, operation: str, value: float
) -> None:
    source = f"""rounded = cutlass.{dtype}(cute.arch.warp_reduction_{operation}(value, threads_in_group=32))
result = cutlass.Float32(rounded)
"""
    expected, original_calls = _run(source, np.float32(value))
    rewritten = _rewrite(source)
    actual, calls = _run(rewritten, np.float32(value))
    _same_value(actual, expected)
    assert calls == original_calls == [dtype, "Float32"]


@pytest.mark.parametrize("dtype", ["Float16", "BFloat16", "Float32", "Float64"])
@pytest.mark.parametrize("value", [1 + 2**-30, -0.0, 70000.0, float("nan")])
def test_identical_second_cast_is_removed_but_first_cast_remains(
    dtype: str, value: float
) -> None:
    source = f"""rounded = cutlass.{dtype}(cute.arch.warp_reduction_sum(value, threads_in_group=32))
result = cutlass.{dtype}(rounded)
"""
    expected, original_calls = _run(source, np.float64(value))
    rewritten = _rewrite(source)
    actual, calls = _run(rewritten, np.float64(value))
    _same_value(actual, expected)
    assert original_calls == [dtype, dtype]
    assert calls == [dtype]
    assert "result = rounded" in rewritten


@pytest.mark.parametrize("container", ["loop", "if", "else"])
def test_narrowing_inside_control_flow_is_preserved(container: str) -> None:
    body = "    rounded = cutlass.Float16(cute.arch.warp_reduction_sum(value))\n    result = cutlass.Float32(rounded)\n"
    source = {
        "loop": "for _iteration in range(2):\n" + body,
        "if": "if True:\n" + body,
        "else": "if False:\n    result = value\nelse:\n" + body,
    }[container]
    expected, original_calls = _run(source, np.float32(1 + 2**-11))
    actual, calls = _run(_rewrite(source), np.float32(1 + 2**-11))
    _same_value(actual, expected)
    assert actual == 1
    assert calls == original_calls


def test_kept_first_cast_preserves_other_live_uses() -> None:
    source = """rounded = cutlass.Float32(cute.arch.warp_reduction_sum(value))
result = cutlass.Float32(rounded)
result = result + rounded
"""
    expected, _ = _run(source, np.float64(1 + 2**-30))
    actual, calls = _run(_rewrite(source), np.float64(1 + 2**-30))
    _same_value(actual, expected)
    assert calls == ["Float32"]
