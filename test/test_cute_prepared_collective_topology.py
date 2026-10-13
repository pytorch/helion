from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from helion._compiler.cute.producer_phase import PointStore
from helion._compiler.cute.producer_phase import PointValue
from helion._compiler.cute.producer_phase import RowSumPoint
from helion._compiler.cute.producer_phase import RowSumTopology
from helion._compiler.cute.producer_phase import emit_point_actions
from helion._compiler.cute.producer_phase import row_sum_actions
from helion._compiler.cute.producer_phase import serial_prefix_actions


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def test_grouped_prefix_captures_values_before_ordered_publication() -> None:
    events = []
    increments = [np.float32(-1e20), np.float32(3), np.float32(-3), np.float32(1)]

    def evaluate(index: int) -> np.float32:
        events.append(("evaluate", index))
        return increments[index]

    class Destination(dict):
        def __setitem__(self, key: int, value: np.float32) -> None:
            events.append(("store", key))
            super().__setitem__(key, value)

    points = tuple(
        PointValue(
            tuple(ast.parse(f"value = evaluate({index})").body),
            _expr("value"),
            PointStore(_expr(f"result[{index}]")),
        )
        for index in range(4)
    )
    originals = tuple(ast.dump(point.statements[0]) for point in points)
    body = emit_point_actions(
        serial_prefix_actions("carry", points, evaluate_first=True)
    )
    assert body is not None
    destination = Destination()
    namespace = {"carry": np.float32(1e20), "evaluate": evaluate, "result": destination}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=list(body), type_ignores=[])),
            "<prefix>",
            "exec",
        ),
        namespace,
    )
    assert list(destination.values()) == [0, 3, 0, 1]
    assert namespace["carry"] == 1
    assert events == [
        (kind, index) for kind in ("evaluate", "store") for index in range(4)
    ]
    assert originals == tuple(ast.dump(point.statements[0]) for point in points)


def test_paired_row_sum_preserves_original_fp32_tree() -> None:
    rng = np.random.default_rng(9271)
    values = (
        rng.standard_normal((8, 16)) * 10 ** rng.integers(-4, 5, (8, 16)).astype(float)
    ).astype(np.float32)
    point = RowSumPoint(
        "sum", (), "cutlass.Float32(values[element]) * cutlass.Float32(values[element])"
    )
    actions = row_sum_actions(
        (point,),
        RowSumTopology(16, 8, True, 2, "xor"),
        element="element",
        position="column",
        lane="lane",
        extent=128,
        fast_math=True,
        target_device_capability=(10, 3),
    )
    body = emit_point_actions(actions)
    assert body is not None

    def fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return (
            a.astype(np.float64) * b.astype(np.float64)
            + np.asarray(c, dtype=np.float64)
        ).astype(np.float32)

    def packed_fma(a: tuple, b: tuple, c: tuple) -> tuple:
        return fma(a[0], b[0], c[0]), fma(a[1], b[1], c[1])

    cute = SimpleNamespace(
        arch=SimpleNamespace(
            fma_packed_f32x2=packed_fma,
            shuffle_sync_bfly=lambda value, offset: value[np.arange(16) ^ offset],
        )
    )
    namespace = {
        "values": values,
        "lane": np.arange(16),
        "cute": cute,
        "cutlass": SimpleNamespace(Float32=np.float32, range_constexpr=range),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=list(body), type_ignores=[])),
            "<row>",
            "exec",
        ),
        namespace,
    )
    even = np.zeros(16, dtype=np.float32)
    odd = np.zeros(16, dtype=np.float32)
    for pair in range(4):
        even = fma(values[2 * pair], values[2 * pair], even)
        odd = fma(values[2 * pair + 1], values[2 * pair + 1], odd)
    expected = even + odd
    for offset in (8, 4, 2, 1):
        expected = expected + expected[np.arange(16) ^ offset]
    np.testing.assert_array_equal(
        namespace["sum_acc"].view(np.uint32), expected.view(np.uint32)
    )


@pytest.mark.parametrize("fast_math,extent", ((False, 128), (True, 127)))
def test_paired_tree_requires_its_math_and_complete_row(
    fast_math: bool, extent: int
) -> None:
    with pytest.raises(AssertionError):
        row_sum_actions(
            (RowSumPoint("sum", (), "cutlass.Float32(values[element])"),),
            RowSumTopology(16, 8, True, 2, "xor"),
            element="element",
            position="column",
            lane="lane",
            extent=extent,
            fast_math=fast_math,
            target_device_capability=(10, 3),
        )
