from __future__ import annotations

import ast
from typing import cast
from unittest.mock import patch

import pytest

from helion._compiler import tile_strategy as lanes
from helion._compiler.cute import nested_lane_reductions as nested


def _normalize(body: list[ast.AST]) -> list[ast.AST]:
    return nested.normalize_nested_lane_reductions(
        body,
        uniform_names={"tid"},
        proven_disjoint_tensor_pairs=set(),
        proven_tensor_stride_values={},
        rename_groups={},
    )


@pytest.mark.parametrize(
    "program",
    [
        "x = 1\ny = x + 2",
        "for i in range(16):\n    x = i + 1\n    if i > 2:\n        y = x * 3",
        "if ready:\n    for i in range(8):\n        tensor[i] = i\nelse:\n    done()",
        "try:\n    unknown()\nfinally:\n    for i in range(4):\n        x = i",
        "def body():\n    for i in range(32):\n        x = f(i)\n    return x",
        "for i in range(8):\n    x = _helion_lane_reduce(value, dtype, 0, 32)",
    ],
)
def test_no_lane_avoids_definition_expansion(program: str) -> None:
    body: list[ast.AST] = list(ast.parse(program).body)
    before = ast.dump(ast.parse(program), include_attributes=True)
    with patch.object(
        lanes,
        "_update_scalar_definitions",
        side_effect=AssertionError("unneeded definition expansion"),
    ):
        assert _normalize(body) is body
    assert (
        ast.dump(
            ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[]),
            include_attributes=True,
        )
        == before
    )


@pytest.mark.parametrize("owner", [None, "different", 0])
def test_nonmatching_lane_metadata_preserves_body(owner: str | int | None) -> None:
    body: list[ast.AST] = list(ast.parse("for i in range(4):\n    x = i + 1").body)
    setattr(body[0], lanes.HELION_LANE_LOOP_VAR_ATTR, owner)
    assert _normalize(body) is body
    assert getattr(body[0], lanes.HELION_LANE_LOOP_VAR_ATTR) == owner


@pytest.mark.parametrize("nested_loop", [False, True])
def test_real_lane_retains_complete_analysis(nested_loop: bool) -> None:
    marker = lanes._lane_reduce_marker_expr(
        "x", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    body: list[ast.AST] = list(
        ast.parse(
            f"for lane in range(4):\n    x = lane + 1\n    reduced = {marker}"
        ).body
    )
    setattr(body[0], lanes.HELION_LANE_LOOP_VAR_ATTR, "lane")
    if nested_loop:
        wrapper = ast.If(
            test=ast.Name(id="ready", ctx=ast.Load()),
            body=cast("list[ast.stmt]", body),
            orelse=[],
        )
        body = [wrapper]
    with (
        patch.object(
            nested, "_normalize_one", wraps=nested._normalize_one
        ) as normalize,
        patch.object(
            lanes, "_update_scalar_definitions", wraps=lanes._update_scalar_definitions
        ) as definitions,
    ):
        _normalize(body)
    assert normalize.call_count == 1
    assert definitions.call_count > 0


@pytest.mark.parametrize("nested_loop", [False, True])
def test_real_lane_without_marker_preserves_upstream_noop(nested_loop: bool) -> None:
    body: list[ast.AST] = list(
        ast.parse("for lane in range(4):\n    x = lane + 1").body
    )
    setattr(body[0], lanes.HELION_LANE_LOOP_VAR_ATTR, "lane")
    if nested_loop:
        body = [
            ast.If(
                test=ast.Name(id="ready", ctx=ast.Load()),
                body=cast("list[ast.stmt]", body),
                orelse=[],
            )
        ]
    before = ast.dump(
        ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[]),
        include_attributes=True,
    )
    with (
        patch.object(
            nested,
            "_normalize_one",
            side_effect=AssertionError("no marker to normalize"),
        ),
        patch.object(
            lanes,
            "_update_scalar_definitions",
            side_effect=AssertionError("unneeded definition expansion"),
        ),
    ):
        assert _normalize(body) is body
    assert (
        ast.dump(
            ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[]),
            include_attributes=True,
        )
        == before
    )
