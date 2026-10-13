from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.fx import Node

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_register_islands import _fixture
from .test_cute_chained_register_islands import _region
import helion
from helion._compiler.cute import chained_register_emission as emission
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_register_binding import bind_preparation_island
from helion._compiler.cute.chained_register_islands import RegisterImage
from helion._compiler.cute.chained_register_islands import plan_register_islands
import helion.language as hl


@helion.kernel(
    backend="cute", static_shapes=True, fast_math=True, autotune_effort="none"
)
def _two_image_loop(a, b, initial):
    steps, size, _ = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=32):
        state = initial[rows, :]
        for step in hl.tile(steps, block_size=1):
            i, j = hl.arange(size), hl.arange(size)
            dense = hl.dot(
                a[step.id, rows, j], b[step.id, i, j], out_dtype=torch.float32
            )
            base = hl.dot(b[step.id, i, j], b[step.id, i, j], out_dtype=torch.float32)
            half = base.to(a.dtype)
            diagonal = torch.where(i[:, None] // 16 == j[None, :] // 16, half, 0)
            lower = torch.where((i[:, None] >= 16) & (j[None, :] < 16), half, 0)
            if size == 64:
                lower = torch.where(
                    (i[:, None] % 32 >= 16)
                    & (j[None, :] % 32 < 16)
                    & (i[:, None] // 32 == j[None, :] // 32),
                    half,
                    0,
                )
            first = hl.dot(diagonal, lower, out_dtype=torch.float32)
            rounded = (-first).to(a.dtype)
            second = hl.dot(rounded, diagonal, out_dtype=torch.float32)
            state = hl.dot(
                state.to(a.dtype),
                (second + base).to(a.dtype),
                acc=state + dense,
                out_dtype=torch.float32,
            )
            history[step.id, rows, j] = state
        final[rows, :] = state
    return history, final


def _capture(dtype, check=None, *, size=32):
    args = (
        torch.zeros((3, size, size), dtype=dtype),
        torch.zeros((3, size, size), dtype=dtype),
        torch.zeros((size, size)),
    )
    config = _config(8, pipeline=True)
    config.config["cute_chained_register_islands"] = True
    config.config["cute_chained_warp_mma_rows"] = size
    original = emission.emit_register_island
    observed = []

    def emit(cg, plan, bound, boundaries, **kwargs):
        if any(
            value.additional_images for value in bound.island.values
        ) and bound.matches(plan, bound.frame, bound.execution, boundaries):
            assert kwargs.get("publication") is None
            if check is not None:
                check(cg, plan, bound, boundaries)
            result = original(cg, plan, bound, boundaries, **kwargs)
            assert result is not None
            observed.append((bound, tuple(result)))
            return result
        return original(cg, plan, bound, boundaries, **kwargs)

    with patch.object(emission, "emit_register_island", emit):
        source = _source(_two_image_loop, args, config)
    assert len(observed) == 1
    return observed[0], source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_actual_two_image_loop_uses_original_typed_register_executor(dtype):
    (bound, body), source = _capture(dtype)
    assert [issue.origins for issue in bound.island.components[0].issues] == [
        (16, 0, 16),
        (16, 0, 0),
    ]
    assert any(len(value.images) == 2 for value in bound.island.values)
    assert any(len(value.images) == 3 for value in bound.island.values)
    assert (
        sum(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.gemm"
            for node in ast.walk(ast.parse("\n".join(body)))
        )
        == 2
    )
    assert "chain_register_island" in source
    for component in bound.island.components:
        for issue in component.issues:
            assert (
                len(
                    next(
                        value
                        for value in bound.island.values
                        if value.node is issue.node
                    ).images
                )
                == 1
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_actual_two_component_emission_preserves_complete_origin_relation(dtype):
    (bound, body), source = _capture(dtype, size=64)
    assert [
        [issue.origins for issue in component.issues]
        for component in bound.island.components
    ] == [
        [(16, 0, 16), (16, 0, 0)],
        [(48, 32, 48), (48, 32, 32)],
    ]
    assert any(len(value.images) == 3 for value in bound.island.values)
    for value in bound.island.values:
        for image in value.images:
            assert tuple(component for component, _ in image.tiles) == (0, 1)
    assert "chain_prep_thread < 64" in "\n".join(body)
    assert (
        sum(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.gemm"
            for node in ast.walk(ast.parse("\n".join(body)))
        )
        == 2
    )
    assert "chain_register_island" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_two_components_pair_images_by_exact_uses_not_sorted_origins(dtype):
    plan, groups, shapes, entries, values = _fixture(dtype, permuted=True)
    first, _, second, _, _ = values
    assert isinstance(second.args[1], Node) and isinstance(first.args[1], Node)
    second.replace_input_with(second.args[1], first.args[1])
    region = _region(plan)
    candidates = plan_register_islands(
        region,
        groups,
        shapes,
        fast_math=True,
        entry_boundaries=entries,
        multi_image=True,
    )
    assert len(candidates) == 1
    island = candidates[0]
    reused = next(value for value in island.values if value.node is first.args[1])
    assert len(reused.images) == 2
    expected = {
        tuple(
            (
                index,
                (
                    component.issues[ordinal].origins[2],
                    component.issues[ordinal].origins[1],
                ),
            )
            for index, component in enumerate(island.components)
        )
        for ordinal in (0, 1)
    }
    assert {image.tiles for image in reused.images} == expected
    assert all(image.node is reused.node for image in reused.images)
    old = plan_register_islands(
        region, groups, shapes, fast_math=True, entry_boundaries=entries
    )
    assert not any(candidate.groups == groups for candidate in old)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_actual_binding_rejects_malformed_stale_or_missing_images(dtype):
    def check(cg, plan, bound, boundaries):
        island = bound.island
        value = next(value for value in island.values if value.additional_images)
        image = value.additional_images[0]
        for invalid in (
            replace(value, additional_images=()),
            replace(value, additional_images=(image, image)),
            replace(
                value, additional_images=(replace(image, node=island.values[-1].node),)
            ),
            replace(value, additional_images=(replace(image, tiles=()),)),
            replace(
                value,
                additional_images=(replace(image, tiles=((0, (0, 0)), (0, (16, 16)))),),
            ),
        ):
            mutated = replace(
                island,
                values=tuple(
                    invalid if item is value else item for item in island.values
                ),
            )
            assert (
                bind_preparation_island(
                    cg, plan, bound.frame, mutated, bound.execution, boundaries
                )
                is None
            )
        with pytest.raises(_UnsupportedChain, match="ambiguous"):
            bound.coordinates(value.node, "probe")
        with pytest.raises(_UnsupportedChain, match="missing"):
            bound.coordinates(RegisterImage(value.node, ()), "probe")
        with pytest.raises(_UnsupportedChain, match="invalid"):
            bound.operand_image(-1, 0)
        with pytest.raises(_UnsupportedChain, match="invalid"):
            bound.operand_image(island.components[0].issues[0].stage, 2)
        old_tiles = image.tiles
        try:
            object.__setattr__(image, "tiles", ())
            assert not bound.matches(plan, bound.frame, bound.execution, boundaries)
            assert emission.emit_register_island(cg, plan, bound, boundaries) is None
        finally:
            object.__setattr__(image, "tiles", old_tiles)
        assert bound.matches(plan, bound.frame, bound.execution, boundaries)

    _capture(dtype, check)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_original_expression_requires_exact_image_without_boundary_fallback(dtype):
    def check(cg, plan, bound, boundaries):
        value = next(value for value in bound.island.values if value.additional_images)
        coords = [bound.coordinates(image, "probe") for image in value.images]
        expression = _Expression(
            cg, plan, {**boundaries, value.node: "unapproved_reload"}
        )
        for index, coordinate in enumerate(coords):
            expression.bind_fragment_image(value.node, coordinate, f"image_{index}")
            assert expression.value(value.node, coordinate) == f"image_{index}"
        with pytest.raises(_UnsupportedChain, match="missing"):
            expression.value(value.node, ("wrong_row", "wrong_column"))
        with pytest.raises(_UnsupportedChain, match="duplicate"):
            expression.bind_fragment_image(value.node, coords[0], "overwrite")
        expression.fragments[value.node] = (coords[0], "mixed")
        with pytest.raises(_UnsupportedChain, match="mixed"):
            expression.value(value.node, coords[0])

    _capture(dtype, check)
