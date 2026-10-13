from __future__ import annotations

import ast
from dataclasses import replace

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_row_parallel import _fixture
from .test_cute_chained_row_registers import _config
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_row_parallel import plan_row_parallel
from helion.exc import InvalidConfig
import helion.language as hl

CACHE = "cute_chained_row_panel_cache"
PACKING = "cute_chained_row_subwarp_packing"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows(a, b, rhs, gather: hl.constexpr):
    output = torch.empty((a.size(0), 16, 32), device=a.device, dtype=a.dtype)
    for batch in hl.tile(a.size(0), block_size=1):
        base = hl.dot(a[batch.begin, :, :], b[batch.begin, :, :])
        index = hl.arange(16)
        total = hl.full([16, 32], 0, torch.float32)
        for island in hl.static_range(4):
            original = base * (1 + island / 8)
            state = original
            for position in hl.static_range(1, 3):
                row = torch.where(index[:, None] == position, state, 0.0).sum(0)
                if gather:
                    column = hl.load(original, [index, position])
                else:
                    column = torch.where(index[None, :] == position, original, 0.0).sum(
                        1
                    )
                state = state * 0.5 + column[:, None] * row[None, :] + 0.0625
            total = hl.dot(state.to(a.dtype), rhs[batch.begin, :, :], acc=total)
        output[batch.begin, :, :] = total.to(a.dtype)
    return output


def _args(gather=True):
    return (
        *[
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in ((2, 16, 16), (2, 16, 16), (2, 16, 32))
        ],
        gather,
    )


@pytest.mark.parametrize("key", [CACHE, PACKING])
def test_row_options_are_strict_and_default_to_absent(key):
    with _cpu_codegen():
        bound = _rows._bind_isolated(_args())
        spec = bound.config_spec
        parent = spec.normalized_config(_config("warp_rows_parallel"))
        assert spec.normalized_config(parent.config | {key: False}) == parent
        for value in (0, 1, None, "true"):
            with pytest.raises(InvalidConfig, match="must be bool"):
                spec.normalized_config(parent.config | {key: value})
        for choices in (
            {"cute_chained_pointwise_schedule": "shared"},
            {"cute_chained_mma_schedule": "tcgen05_tmem"},
            {"cute_cluster_n": 2},
            {"pid_type": "persistent_blocked"},
        ):
            with pytest.raises(InvalidConfig):
                spec.normalized_config(parent.config | choices | {key: True})
        if key == PACKING:
            with pytest.raises(InvalidConfig):
                spec.normalized_config(
                    parent.config
                    | {"cute_chained_pointwise_schedule": "warp_rows", key: True}
                )
        generation = spec.create_config_generation()
        chosen = spec.normalized_config(parent.config | {key: True})
        assert generation.flatten(
            generation.unflatten(generation.flatten(chosen))
        ) == generation.flatten(chosen)


def test_public_cache_and_packing_codegen_restore_original():
    with _cpu_codegen():
        bound = _rows._bind_isolated(_args())
        original = bound.to_code(_config("warp_rows_parallel"))
        cached = bound.to_code(_config("warp_rows_parallel", **{CACHE: True}))
        packed = bound.to_code(
            _config("warp_rows_parallel", **{CACHE: True, PACKING: True})
        )
        assert cached != original and packed != cached
        assert "mask_and_clamp=4127" in packed
        assert (
            bound.to_code(
                _config("warp_rows_parallel", **{CACHE: False, PACKING: False})
            )
            == original
        )
        assert packed.count("cute.arch.sync_threads()") == original.count(
            "cute.arch.sync_threads()"
        )
        ast.parse(packed)
        # Selecting an optimization that cannot produce an image must decline.
        no_gather = _rows._bind_isolated(_args(False))
        with pytest.raises(helion.exc.BackendUnsupported):
            no_gather.to_code(_config("warp_rows_parallel", **{CACHE: True}))
        no_cache_packed = no_gather.to_code(
            _config("warp_rows_parallel", **{PACKING: True})
        )
        assert "mask_and_clamp=4127" in no_cache_packed


@pytest.mark.parametrize("option", ["row_panel_cache", "row_subwarp_packing"])
def test_row_options_are_bound_to_movement_proof(option):
    plan = _fixture()
    parallel = plan_row_parallel(plan)
    assert parallel is not None
    parallel.check(plan)
    changed = (
        replace(plan, row_panel_cache=True)
        if option == "row_panel_cache"
        else replace(plan, row_subwarp_packing=True)
    )
    with pytest.raises(chain._UnsupportedChain, match="changed"):
        parallel.check(changed)
