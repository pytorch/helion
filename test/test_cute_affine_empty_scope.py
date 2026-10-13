from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from .test_cute_affine_vector_io import _SOURCE
from helion._compiler.cute import affine_vector_io as affine


class _OriginalVectorizer(affine._Vectorizer):
    """The inherited body path before the exact-class eligibility gate."""


def vectorizer(body, cls=affine._Vectorizer):
    return cls(
        body,
        strides={("src", 0): 1, ("out", 0): 1},
        alignments={"src": 16, "out": 16},
        dtypes={"src": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset(("src", "out"))},
        constexpr={},
    )


@pytest.mark.parametrize(
    "header",
    [
        "for vec_lane_0 in cutlass.range_constexpr(1):",
        "for vec_lane_0 in cutlass.range_constexpr(16):",
        "for vec_lane_0 in cutlass.range_constexpr(width):",
        "for vec_lane_0 in range(4):",
        "for scalar_lane_0 in cutlass.range_constexpr(4):",
        "for vec_lane_0, other in cutlass.range_constexpr(4):",
    ],
)
def test_ineligible_scope_keeps_original_nodes_without_snapshots(header):
    body = ast.parse(header + "\n    value = unknown + 1\n").body
    original = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=True)
    rewrite = vectorizer(body)
    with patch.object(
        rewrite, "snapshots", side_effect=AssertionError("unneeded proof")
    ):
        result = rewrite.body(body)
    assert result is body
    assert (
        ast.dump(ast.Module(body=result, type_ignores=[]), include_attributes=True)
        == original
    )
    assert rewrite.changed == 0


@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("mutation", ["", "                start = 0\n"])
def test_eligible_nested_loop_preserves_exact_existing_proof(width, mutation):
    source = _SOURCE.replace("range_constexpr(4)", f"range_constexpr({width})")
    source = source.replace(
        "                result = value * 1.25 + 2\n",
        mutation + "                result = value * 1.25 + 2\n",
    )
    results = []
    for cls in (affine._Vectorizer, _OriginalVectorizer):
        module = ast.parse(source)
        function = module.body[0]
        assert isinstance(function, ast.FunctionDef)
        rewrite = vectorizer(function.body, cls)
        function.body = rewrite.body(function.body)
        results.append((ast.dump(module, include_attributes=True), rewrite.changed))
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "wrapper",
    [
        "while ready:\n{loop}",
        "def nested():\n{loop}",
        "for item in range(1):\n    pass\nelse:\n{loop}",
    ],
)
def test_inventory_does_not_visit_scopes_the_original_walker_ignores(wrapper):
    loop = "    for vec_lane_7 in cutlass.range_constexpr(4):\n        pass\n"
    body = ast.parse(wrapper.format(loop=loop)).body
    assert not affine._affine_lane_scopes(body)
    rewrite = vectorizer(body)
    with patch.object(
        rewrite, "snapshots", side_effect=AssertionError("unreachable proof")
    ):
        assert rewrite.body(body) is body


def test_inventory_is_shared_only_within_one_body_traversal():
    module = ast.parse(_SOURCE)
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    rewrite = vectorizer(function.body)
    original = affine._affine_lane_scopes
    with patch.object(affine, "_affine_lane_scopes", wraps=original) as inventory:
        rewrite.body(function.body)
    assert inventory.call_count == 1

    empty = ast.parse("pass").body
    same_rewrite = vectorizer(empty)
    assert same_rewrite.body(empty) is empty
    empty.extend(
        ast.parse("for vec_lane_1 in cutlass.range_constexpr(4):\n    pass\n").body
    )
    with patch.object(
        same_rewrite, "snapshots", wraps=same_rewrite.snapshots
    ) as snapshots:
        same_rewrite.body(empty)
    assert snapshots.call_count > 0


def test_oversized_inert_sibling_does_not_skip_live_scope_proofs():
    module = ast.parse(_SOURCE)
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    statements = ["value_0 = seed"] + [
        f"value_{index} = value_{index - 1} + value_{index - 1}"
        for index in range(1, 20)
    ]
    inert = ast.parse(
        "for item in range(32):\n" + "\n".join("    " + line for line in statements)
    ).body[0]
    assert isinstance(inert, ast.For)
    function.body.append(inert)
    rewrite = vectorizer(function.body)
    seen = []
    original = rewrite.snapshots

    def snapshots(body):
        seen.append(body)
        return original(body)

    with patch.object(rewrite, "snapshots", snapshots):
        rewrite.body(function.body)
    assert rewrite.changed == 1
    assert all(body is not inert.body for body in seen)


def test_subclass_retains_its_original_snapshot_admission():
    body = ast.parse(
        "for packet in cutlass.range_constexpr(4):\n    value = source\n"
    ).body
    rewrite = vectorizer(body, _OriginalVectorizer)
    with patch.object(rewrite, "snapshots", wraps=rewrite.snapshots) as snapshots:
        rewrite.body(body)
    assert snapshots.call_count == 2
