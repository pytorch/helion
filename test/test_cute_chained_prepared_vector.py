from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._compiler.cute import chained_prepared_values
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _prepared_vector_loop(a, b, c, initial, fp32: hl.constexpr):
    steps, rows, reduction = a.shape
    columns = c.shape[-1]
    history = torch.empty((steps, rows, columns), device=a.device)
    final = torch.empty_like(initial)
    for rr, cc in hl.tile([rows, columns], block_size=[32, 128]):
        state = initial[rr, cc]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(32)
            prepared = hl.dot(a[step.id, rr, kk] * 0.5, b[step.id, kk, jj]).to(a.dtype)
            if fp32:
                image_rows = rr.index
            else:
                image_rows = jj
            image = torch.sigmoid(
                hl.load(
                    c, [step.id, image_rows, cc], extra_mask=(image_rows < 17)[:, None]
                ).float()
            )
            if fp32:
                state = hl.dot(prepared, c[step.id, jj, cc], acc=state * image)
            else:
                state = hl.dot(prepared, image.to(a.dtype), acc=state * 0.5)
            history[step.id, rr, cc] = state
        final[rr, cc] = state
    return history, final


def _inputs(device, dtype, fp32, steps=3):
    torch.manual_seed(9739)
    return (
        *(
            torch.randn(shape, device=device, dtype=dtype) * 0.125
            for shape in (
                (steps, 32, 16),
                (steps, 16, 32),
                (steps, 32, 128),
            )
        ),
        torch.randn((32, 128), device=device) * 0.125,
        fp32,
    )


def _vector_config(consumer_warps):
    config = _config(16, pipeline=True, consumer_warps=consumer_warps)
    config.config.update(
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=8,
        cute_chained_scratch_layout="xor",
    )
    return config


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_frontier_vector_preserves_masked_pointwise_expression_cpu(dtype, fp32):
    args = _inputs("cpu", dtype, fp32)
    source = _source(_prepared_vector_loop, args, _vector_config(8))
    assert "chain_prepared_1_vector_leaf_" in source
    if fp32:
        assert "chain_prepared_1_vector_target" not in source
        assert "chain_prepared_1[chain_prepared_1_vector_row" in source
    else:
        assert "chain_prepared_1_vector_values" in source
    with patch.object(
        chained_prepared_values, "emit_vector_expression", return_value=None
    ):
        ordinary = _source(_prepared_vector_loop, args, _vector_config(8))
    assert "chain_prepared_1_vector" not in ordinary
    assert source != ordinary


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_frontier_vector_gpu_matches_scalar_typed_publication(
    fp32, dtype, steps, consumer_warps
):
    args = _inputs(DEVICE, dtype, fp32, steps)
    saved = tuple(value.clone() for value in args[:4])
    config = _vector_config(consumer_warps)
    bound = _prepared_vector_loop._bind_isolated(args)
    with (
        bound.env.use_runtime_arg_values(_runtime_values(_prepared_vector_loop, args)),
        patch.object(
            chained_prepared_values, "emit_vector_expression", return_value=None
        ),
    ):
        ordinary = bound.compile_config(config)
    vector_bound = _prepared_vector_loop._bind_isolated(args)
    with vector_bound.env.use_runtime_arg_values(
        _runtime_values(_prepared_vector_loop, args)
    ):
        assert "chain_prepared_1_vector_leaf_" in vector_bound.to_code(config)
        vector = vector_bound.compile_config(config)
    actual = vector(*args)
    torch.testing.assert_close(actual, ordinary(*args), atol=0, rtol=0)
    torch.testing.assert_close(vector(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args[:4], saved, atol=0, rtol=0)
