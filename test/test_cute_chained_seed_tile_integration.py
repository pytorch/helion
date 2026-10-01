from __future__ import annotations

import ast

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _panel_sequence(a, b, initial, steps: int):
    steps = hl.specialize(steps)
    batches, _, rows, reduction = a.shape
    columns = b.size(-1)
    history = torch.empty(
        (batches, steps, rows, columns), device=a.device, dtype=torch.float32
    )
    final = torch.empty_like(initial)
    for batch, rr in hl.tile(
        [batches, rows], block_size=[1, 64 if rows == 64 else 128]
    ):
        state = initial[batch.id, rr, :]
        for step in hl.tile(steps, block_size=1):
            cc, kk = hl.arange(columns), hl.arange(reduction)
            first = hl.dot(a[batch.id, step.id, rr, kk], b[batch.id, step.id, kk, cc])
            state = hl.dot(
                a[batch.id, step.id, rr, kk],
                b[batch.id, step.id, kk, cc],
                acc=state * 0.5 + torch.sigmoid(state) * 0.03125 + first,
            )
            history[batch.id, step.id, rr, cc] = state
        final[batch.id, rr, :] = state
    return history, final


def _args(device, dtype, shape, steps):
    torch.manual_seed(9973)
    rows, columns = shape
    return (
        torch.randn((2, max(steps, 1), rows, 16), device=device, dtype=dtype) * 0.125,
        torch.randn((2, max(steps, 1), 16, columns), device=device, dtype=dtype)
        * 0.125,
        torch.randn((2, rows, columns), device=device) * 0.125,
        steps,
    )


def _config(columns, *, pipeline=False, value_tile=128):
    settings = {
        "num_warps": 16 if pipeline else 8,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_seed_tile_columns": columns,
    }
    if pipeline:
        settings.update(
            block_sizes=[value_tile],
            cute_chained_group_contractions=True,
            cute_chained_warp_mma_rows=32,
            cute_chained_pointwise_cache_bytes=4096,
            cute_chained_scan_schedule="warp",
            cute_chained_scratch_layout="xor",
            cute_chained_pointwise_vectorize=True,
            cute_chained_pointwise_unroll=8,
            cute_chained_preparation_pipeline=True,
        )
    return helion.Config.from_dict(settings)


@pytest.mark.parametrize(
    "shape,columns", [((128, 96), 32), ((128, 96), 64), ((64, 128), 32)]
)
def test_seed_tiles_keep_original_logical_coordinates_cpu(shape, columns):
    source = _source(
        _panel_sequence,
        _args("cpu", torch.bfloat16, shape, 3),
        _config(columns),
    )
    # hl.arange pads the 96-column logical domain to a physical128-column tile.
    width = 64 if shape[0] == 64 else 128
    assert source.count("chain_1_seed_1_segment =") == (width + columns - 1) // columns
    assert "0.03125" in source
    assert "cute.arch.fence_view_async_tmem_store()" in source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "chain_thread < 128":
            assert "cute.arch.sync_threads()" not in ast.unparse(node)


@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("columns", [32, 64])
def test_seed_tiles_preserve_resident_group_neighbor_cpu(value_tile, columns):
    kernel, args = _kda_fixture()
    source = _source(
        kernel, args, _config(columns, pipeline=True, value_tile=value_tile)
    )
    assert source.count("chain_13_seed_14_segment =") == 128 // columns
    if value_tile == 128:
        # Only the full-domain fixture admits the resident carry proof.
        # Its preserved output segment must never be zeroed or seeded again.
        assert "chain_13_seed_13_segment" not in source
        assert source.count("chain_13_seed_14_load =") == 128 // columns
        assert "chain_14_c[" not in source
        assert "chain_resident_carry_" in source
    else:
        assert "chain_13_seed_13_segment" in source
        assert "chain_resident_carry_" not in source


def test_seed_tiles_reject_an_ineffective_positive_request_cpu():
    with pytest.raises(helion.exc.BackendUnsupported, match="multi-panel"):
        _source(
            _panel_sequence,
            _args("cpu", torch.bfloat16, (64, 128), 3),
            _config(64),
        )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape,steps,columns", [((128, 96), 0, 32), ((128, 96), 1, 64), ((64, 128), 5, 32)]
)
def test_seed_tiles_gpu_preserve_fp32_expression_replay_and_inputs(
    dtype, shape, steps, columns
):
    args = _args(DEVICE, dtype, shape, steps)
    saved = tuple(arg.clone() for arg in args if isinstance(arg, torch.Tensor))
    compiled = []
    for width in (0, columns):
        bound = _panel_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_panel_sequence, args)):
            compiled.append(bound.compile_config(_config(width)))
    expected, actual = compiled[0](*args), compiled[1](*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(compiled[1](*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(
        tuple(arg for arg in args if isinstance(arg, torch.Tensor)),
        saved,
        atol=0,
        rtol=0,
    )
