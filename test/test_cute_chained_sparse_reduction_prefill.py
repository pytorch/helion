from __future__ import annotations

import ast

import pytest

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from .test_cute_chained_seed_tile_integration import _config
from helion._testing import skipUnlessBackends


@pytest.mark.parametrize("consumer_warps", [4, 8])
@pytest.mark.parametrize("value_tile", [64, 128])
def test_sparse_reduction_prefill_cpu_preserves_selected_fp32_path(
    value_tile, consumer_warps
):
    kernel, args = _kda_fixture()
    config = _config(32, pipeline=True, value_tile=value_tile)
    config.config.update(
        cute_chained_vector_group=True,
        cute_chained_pipeline_consumer_warps=consumer_warps,
        cute_chained_pointwise_cache_layout="xor",
    )
    source = _source(kernel, args, config)
    assert "chain_collective_0[31, chain_collective_1_vector]" in source
    assert "chain_collective_1_position" not in source
    assert "chain_0_vector_group_output_2" in source
    updates = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "chain_collective_1_acc"
            for target in node.targets
        )
    ]
    assert len(updates) == 6
    assert all(
        isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
        for value in updates
    )
    assert all(ast.unparse(value.left) == "cutlass.Float32(0)" for value in updates)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("consumer_warps", [4, 8])
@pytest.mark.parametrize("rectangular_leaf", [False, True])
def test_sparse_reduction_prefill_gpu_preserves_ragged_fp32_state(
    bt32, consumer_warps, rectangular_leaf
):
    _run_preparation_prefill(
        bt32,
        128,
        consumer_warps,
        rectangular_leaf=rectangular_leaf,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
    )
