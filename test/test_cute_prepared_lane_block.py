from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any

import numpy as np

from helion._compiler.cute import producer_phase as current


def test_two_half_coordinates_and_fp32_addition_tree() -> None:
    topology = current.RowSumTopology(
        lanes=8, elements_per_lane=16, lane_block=8, shuffle="xor"
    )
    actions = current.row_sum_actions(
        (current.RowSumPoint("sum", ("visited.append(column)",), "values[column]"),),
        topology,
        element="element",
        position="column",
        lane="lane",
        extent=128,
    )
    body = current.emit_point_actions(actions)
    assert body is not None
    split = next(
        i
        for i, node in enumerate(body)
        if "shuffle_sync_bfly" in ast.unparse(ast.fix_missing_locations(node))
    )
    local = compile(
        ast.fix_missing_locations(ast.Module(body=list(body[:split]), type_ignores=[])),
        "<local>",
        "exec",
    )
    shuffle = compile(
        ast.fix_missing_locations(ast.Module(body=list(body[split:]), type_ignores=[])),
        "<shuffle>",
        "exec",
    )
    rng = np.random.default_rng(219)
    values = (rng.standard_normal(128) * 10.0 ** rng.integers(-4, 5, 128)).astype(
        np.float32
    )
    lane_sums = []
    all_coordinates = []
    expected_lanes: list[np.float32] = []
    for lane in range(8):
        coordinates = [
            half * 64 + lane * 8 + offset for half in range(2) for offset in range(8)
        ]
        namespace: dict[str, Any] = {
            "values": values,
            "lane": lane,
            "visited": [],
            "cutlass": SimpleNamespace(Float32=np.float32, range_constexpr=range),
        }
        exec(local, namespace)
        assert namespace["visited"] == coordinates
        all_coordinates.extend(coordinates)
        lane_sums.append(namespace["sum_acc"])
        carry = np.float32(0)
        for coordinate in coordinates:
            carry = np.float32(carry + values[coordinate])
        expected_lanes.append(carry)
    assert sorted(all_coordinates) == list(range(128))
    offsets = []

    def butterfly(value: np.ndarray, offset: int) -> np.ndarray:
        offsets.append(offset)
        return value[np.arange(8) ^ offset]

    namespace = {
        "sum_acc": np.array(lane_sums, dtype=np.float32),
        "cute": SimpleNamespace(arch=SimpleNamespace(shuffle_sync_bfly=butterfly)),
    }
    exec(shuffle, namespace)
    expected = np.array(expected_lanes, dtype=np.float32)
    for offset in (4, 2, 1):
        expected = expected + expected[np.arange(8) ^ offset]
    assert offsets == [4, 2, 1]
    np.testing.assert_array_equal(
        namespace["sum_acc"].view(np.uint32), expected.view(np.uint32)
    )


def test_explicit_prefix_rounding_keeps_completed_point_value() -> None:
    # A fused multiply-add gives -2**-46, while the original completed FP32
    # product followed by a separately rounded scan add gives +0.
    left = np.float32(1 + 2**-23)
    right = np.float32(1 - 2**-23)
    point = np.float32(left * right)
    carry = np.float32(-1)
    fused = np.float32(np.float64(left) * np.float64(right) + np.float64(carry))
    assert fused != 0
    actions = current.serial_prefix_actions(
        "carry",
        (
            current.PointValue(
                (),
                ast.Name(id="point", ctx=ast.Load()),
                current.PointStore(ast.parse("result[0]", mode="eval").body),
            ),
            current.PointValue(
                (),
                ast.Constant(value=1.0),
                current.PointStore(ast.parse("result[1]", mode="eval").body),
            ),
        ),
        round_each_add=True,
    )
    body = current.emit_point_actions(actions)
    assert body is not None
    calls = []

    def rounded_add(a: np.float32, b: np.float32) -> np.float32:
        calls.append((a, b))
        return np.float32(a + b)

    namespace = {
        "carry": carry,
        "point": point,
        "result": [None, None],
        "_helion_add_fp32_rn": rounded_add,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=list(body), type_ignores=[])),
            "<prefix>",
            "exec",
        ),
        namespace,
    )
    assert namespace["result"] == [np.float32(0), np.float32(1)]
    assert len(calls) == 2
