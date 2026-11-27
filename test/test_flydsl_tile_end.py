from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "fixed_end", [False, True], ids=["dynamic-stop", "folded-stop"]
)
def test_shifted_tile_end_casts_integer_bounds(fixed_end: bool) -> None:
    @helion.kernel(backend="flydsl", static_shapes=False)
    def shifted(out, fixed_end: hl.constexpr):
        start = out.size(0) - 8 if fixed_end else 1
        stop = 2 if fixed_end else out.size(0)
        for tile in hl.tile(start, stop):
            out[tile, :] = out[tile, :] + tile.end

    # Rendering only checks that the SDK can be imported; no ROCm device is needed.
    with cpu_render_target(), patch.dict(sys.modules, flydsl=ModuleType("flydsl")):
        bound = cpu_bind(shifted, (torch.empty(9, 256), fixed_end))
        source = bound.to_code(helion.Config(block_sizes=[4]))
    # Dynamic stops clamp against zero; a fixed stop folds stop - block_size.
    assert f"fx.Int64({-2 if fixed_end else 0})" in source
