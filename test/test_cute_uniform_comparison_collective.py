from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_collective_matmul import _cpu_bind
from test.test_cute_collective_native_seeded import _config

import helion
from helion._compiler.cute import uniform_comparison
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


def _compare_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    seed: torch.Tensor,
    words: torch.Tensor,
    limit: int,
    p: float,
) -> torch.Tensor:
    out = torch.empty_like(seed)
    for row, column in hl.tile([a.size(0), b.size(0)]):
        accumulator = seed[row, column]
        for reduction in hl.tile(limit):
            word = words[row, reduction]
            signed = ((word + 2147483648) & 4294967295) - 2147483648
            magnitude = torch.where(signed < 0, -signed - 1, signed)
            uniform = magnitude.to(torch.float32) * 4.6566127342e-10
            lhs = torch.where(uniform > p, a[row, reduction], 0.0)
            accumulator = hl.dot(lhs, b[column, reduction].T, acc=accumulator)
        out[row, column] = accumulator
    return out


def _code(*, static_shapes: bool, structural_flags: bool, config: helion.Config) -> str:
    kernel = helion.kernel(
        _compare_matmul,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        cute_full_slice_matmul_tiling=structural_flags,
        cute_segmented_matmul_tiling=structural_flags,
        cute_flatten_nested_reductions=structural_flags,
    )
    # Keep storage alive through codegen, including the nonzero FP32 carry.
    arguments = (
        torch.empty((130, 78), dtype=torch.float16),
        torch.empty((70, 78), dtype=torch.float16),
        torch.empty((130, 70), dtype=torch.float32),
        torch.empty((130, 78), dtype=torch.int64),
        78,
        0.25,
    )
    return _cpu_bind(kernel, arguments).to_code(config)


def _without_local_names(source: str) -> str:
    # Typed and ordinary lowerings choose different temporary names. Compare
    # the complete operation/control-flow AST after alpha-renaming locals.
    module = ast.parse(source)
    for function in module.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        names = dict.fromkeys(
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        replacements = {name: f"_local_{index}" for index, name in enumerate(names)}
        for node in ast.walk(function):
            if isinstance(node, ast.Name):
                node.id = replacements.get(node.id, node.id)
            elif isinstance(node, ast.arg):
                node.arg = replacements.get(node.arg, node.arg)
    return ast.dump(module)


@pytest.mark.parametrize("compute", ("warp", "tcgen05"))
@pytest.mark.parametrize("static_shapes", (False, True))
@pytest.mark.parametrize("structural_flags", (False, True))
def test_collective_random_operand_keeps_original_scalar_recipe(
    compute: str, static_shapes: bool, structural_flags: bool
) -> None:
    config = _config(compute=compute)
    with patch.object(uniform_comparison, "match_uniform_float_gt", return_value=None):
        control = _code(
            static_shapes=static_shapes,
            structural_flags=structural_flags,
            config=config,
        )
    candidate = _code(
        static_shapes=static_shapes, structural_flags=structural_flags, config=config
    )
    assert _without_local_names(candidate) == _without_local_names(control)
    assert "uniform_cutoff" not in candidate
    assert "4.6566127342e-10" in candidate
    assert "collective_mma" in candidate
    assert ("tcgen05.CtaGroup.ONE" in candidate) is (compute == "tcgen05")


@pytest.mark.parametrize("static_shapes", (False, True))
def test_simt_random_operand_still_uses_exact_cutoff(static_shapes: bool) -> None:
    config = helion.Config.from_dict(
        {**_config(compute="warp").config, "cute_collective_mma": False}
    )
    source = _code(static_shapes=static_shapes, structural_flags=False, config=config)
    assert "uniform_cutoff(p)" in source
    assert "_helion_uniform_gt(" not in source
    assert "collective_mma" not in source
