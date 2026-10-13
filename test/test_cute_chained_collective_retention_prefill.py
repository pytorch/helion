from __future__ import annotations

from unittest.mock import patch

import pytest

from .test_cute_chained_leaf_set_prefill import _ragged_cpu_fixture
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from .test_cute_chained_seed_tile_integration import _config
from helion._compiler.cute import chained_row_collective_emission as row_module
from helion._testing import skipUnlessBackends


def _preflight(leaf_count, cohorts, *, bt32=True):
    kernel, args = _ragged_cpu_fixture(bt32)
    config = _config(32, pipeline=True, value_tile=128)
    config.config.update(
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=leaf_count,
        cute_chained_pointwise_cache_layout="xor",
        cute_chained_vector_group=True,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
        cute_chained_pipeline_consumer_warps=4,
        cute_chained_collective_retention=True,
    )
    if cohorts > 1:
        config.config["cute_chained_register_islands"] = True
    accepted = []
    original = row_module.emit_row_collective_group

    def observe(cg, plan, frame, candidate, boundaries, operands, **kwargs):
        result = original(cg, plan, frame, candidate, boundaries, operands, **kwargs)
        if result is not None:
            accepted.append(candidate)
        return result

    with patch.object(row_module, "emit_row_collective_group", observe):
        source = _source(kernel, args, config)
    assert len(accepted) == 1
    # With repeated raw leaves, the second descriptor load separates the two
    # sums. Only the adjacent suffix is fused; no transaction is moved.
    assert len(accepted[0].collectives) == (2 if leaf_count == 1 else 1)
    assert "chain_row_collective_" in source
    assert "chain_0_vector_group_output_2" not in source
    assert "chained_rectangular_leaf_tma" in source
    assert ("chain_register_island" in source) is (cohorts > 1)
    assert "chain_prepared_group_" in source


@pytest.mark.parametrize("leaf_count", [1, 4])
@pytest.mark.parametrize("cohorts", [1, 3])
def test_ragged_row_retention_and_leaf_set_preflight_cpu(leaf_count, cohorts):
    _preflight(leaf_count, cohorts)


def test_bt64_serial_row_retention_preflight_cpu():
    _preflight(1, 1, bt32=False)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("leaf_count", [1, 4])
@pytest.mark.parametrize("cohorts", [1, 3])
@pytest.mark.parametrize("register_islands", [False, True])
def test_row_retention_prefill_original_oracle_replay_and_inputs_gpu(
    leaf_count, cohorts, register_islands
):
    _run_preparation_prefill(
        True,
        128,
        4,
        rectangular_leaf=True,
        leaf_count=leaf_count,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
        cohorts=cohorts,
        preparation_unroll=1,
        register_islands=register_islands,
        collective_retention=True,
    )


@skipUnlessBackends(["cute"])
def test_bt64_serial_row_retention_original_oracle_gpu():
    _run_preparation_prefill(
        False,
        128,
        4,
        rectangular_leaf=True,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
        preparation_unroll=1,
        collective_retention=True,
    )
