from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_register_emission import _register_config
import helion
from helion._compiler.cute import chained_frontier_groups as groups
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shared_frontier_sequence(a, b, initial, narrow_shared: hl.constexpr):
    steps, size, _ = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=32):
        state = initial[rows, :]
        for step in hl.tile(steps, block_size=1):
            cols = hl.arange(size)
            kk = hl.arange(size)
            product = hl.dot(
                a[step.id, cols, kk], b[step.id, kk, cols], out_dtype=torch.float32
            )
            shared = product * torch.exp(a[step.id, kk, cols].float() * 0.125)
            first = shared.to(a.dtype)
            if narrow_shared:
                shared = first.float()
            second = (shared + 0.125).to(a.dtype)
            projected = hl.dot(state.to(a.dtype), first, out_dtype=torch.float32)
            state = hl.dot(
                state.to(a.dtype), second, acc=projected, out_dtype=torch.float32
            )
            history[step.id, rows, cols] = state
        final[rows, :] = state
    return history, final


def _sequence_inputs(dtype=torch.bfloat16, steps=3, device="cpu", narrow_shared=False):
    torch.manual_seed(9451)
    return (
        torch.randn((steps, 32, 32), dtype=dtype, device=device) * 0.125,
        torch.randn((steps, 32, 32), dtype=dtype, device=device) * 0.125,
        torch.randn((32, 32), dtype=torch.float32, device=device) * 0.125,
        narrow_shared,
    )


def _sequence_config(cohorts=1, consumer_warps=4):
    config = _config(16, pipeline=True, consumer_warps=consumer_warps)
    config.config.update(
        cute_chained_pointwise_vectorize=True,
        cute_chained_vector_group=True,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
    )
    return config


def _group_loops(source):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("chain_prepared_")
        and node.target.id.endswith("_group_step")
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cohorts", [1, 3])
@pytest.mark.parametrize("narrow_shared", [False, True])
def test_generic_frontier_group_uses_shared_coordinate_expression_cpu(
    dtype, cohorts, narrow_shared
):
    args = _sequence_inputs(dtype, narrow_shared=narrow_shared)
    source = _source(_shared_frontier_sequence, args, _sequence_config(cohorts))
    loops = _group_loops(source)
    assert len(loops) == 1
    body = ast.unparse(loops[0])
    assert body.count("cute.math.exp2(") == 1
    assert "_output_0_values" in body and "_output_1_values" in body
    assert "arrive_and_wait" not in body


@pytest.mark.parametrize("tma", [False, True])
@pytest.mark.parametrize("cohorts", [1, 3])
def test_frontier_group_keeps_final_frame_native_views_and_register_island_cpu(
    tma, cohorts
):
    kernel, args = _kda_fixture()
    config = _register_config(
        cohorts,
        cute_chained_vector_group=True,
        cute_chained_leaf_pipeline="rectangular_tma" if tma else "legacy",
        cute_chained_register_islands=True,
    )
    source = _source(kernel, args, config)
    loops = _group_loops(source)
    assert len(loops) == 1
    body = ast.unparse(loops[0])
    assert "chain_prepared_5_group_output_2_values" in body
    assert "chain_prepared_7_vector_step" in source
    assert "chain_register_island" in source
    assert ("mbarrier_arrive_and_expect_tx" in source) is tma


def test_disabled_frontier_group_never_plans_and_preserves_source_cpu():
    args = _sequence_inputs()
    config = _sequence_config()
    config.config["cute_chained_vector_group"] = False
    with patch.object(groups, "emit_frontier_group", return_value=None):
        old = _source(_shared_frontier_sequence, args, config)
    with patch.object(
        groups, "plan_frontier_group", side_effect=AssertionError("discovery ran")
    ):
        assert _source(_shared_frontier_sequence, args, config) == old
