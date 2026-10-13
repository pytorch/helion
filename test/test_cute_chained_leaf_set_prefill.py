from __future__ import annotations

import ast
from unittest.mock import patch

from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _safe
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from .test_cute_chained_seed_tile_integration import _config
import helion
from helion._compiler.cute import chained_pipeline_storage as storage_module
from helion._testing import skipUnlessBackends


def _ragged_cpu_fixture(bt32=True):
    kernel, small = _kda_fixture()
    if not bt32:
        kernel = helion.kernel(
            kda_prefill_native_math.fn,
            backend="cute",
            static_shapes=True,
            fast_math=True,
            autotune_config_overrides={
                "cute_chained_mma_schedule": "tcgen05_tmem",
                "cute_chained_group_contractions": True,
            },
        )
    # Exact tensor metadata/sequence boundaries of the unchanged GPU oracle
    # helper, including empty, single-token and repeatedly recycled slots.
    shape = (1, 259, 2, 128)
    q, k, v, gate = (torch.empty(shape, dtype=torch.bfloat16) for _ in range(4))
    initial = torch.empty((4, 2, 128, 128), dtype=torch.float32)
    return kernel, (
        q,
        k,
        v,
        gate,
        torch.empty(shape[:-1], dtype=torch.bfloat16),
        small[5],
        small[6],
        initial,
        torch.empty_like(v),
        torch.empty_like(initial),
        torch.tensor((0, 0, 1, 98, 259), dtype=torch.int64),
        *small[11:],
    )


def _preflight(leaf_count, cohorts, register_islands, *, bt32=True):
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
    )
    if register_islands:
        config.config["cute_chained_register_islands"] = True
    records = []
    original = storage_module.finalize_pipeline_storage

    def observe(plan, pipeline, transports, **kwargs):
        storage = original(plan, pipeline, transports, **kwargs)
        records.append((pipeline, storage, kwargs["capacity_bytes"]))
        return storage

    with patch.object(storage_module, "finalize_pipeline_storage", observe):
        source = _source(kernel, args, config)
    assert len(records) == 1
    pipeline, storage, capacity = records[0]
    assert storage is not None and storage.charged_bytes <= capacity
    assert len(pipeline.prepared_leaves) == min(leaf_count, 3)
    assert pipeline.preparation_threads == 384
    assert pipeline.recurrence_threads == 128
    if cohorts > 1:
        assert pipeline.cohorts is not None
        assert pipeline.cohorts.cohort_threads == 128
    _safe(pipeline.frame)
    assert "chain_0_vector_group_output_2" in source
    assert ("chain_register_island" in source) is register_islands
    assert "chained_rectangular_leaf_tma" in source
    tree = ast.parse(source)
    waits = [
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.Call)
        and ast.unparse(item.func) == "cute.arch.mbarrier_wait"
    ]
    phase = "chain_iteration & 1" if cohorts == 1 else "chain_generation & 1"
    for ordinal, leaf in enumerate(pipeline.prepared_leaves):
        pointer = f"chain_slot_bars + {pipeline.slots * 2 + ordinal * (pipeline.slots if cohorts > 1 else 1)}"
        if cohorts > 1:
            pointer += " + chain_slot"
        matching = [item for item in waits if ast.unparse(item.args[0]) == pointer]
        assert len(matching) == 1 and ast.unparse(matching[0].args[1]) == phase
        assert all(name in source for name in leaf.wrapper["kernel_args"])
    return source, pipeline, storage


@pytest.mark.parametrize("leaf_count", [2, 4])
@pytest.mark.parametrize("cohorts", [1, 3])
def test_ragged_leaf_set_prefill_complete_quota_and_phase_preflight_cpu(
    leaf_count, cohorts
):
    _preflight(leaf_count, cohorts, register_islands=cohorts > 1)


def test_bt64_serial_leaf_set_complete_quota_preflight_cpu():
    _preflight(4, 1, False, bt32=False)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("leaf_count", [2, 4])
@pytest.mark.parametrize("cohorts", [1, 3])
@pytest.mark.parametrize("register_islands", [False, True])
def test_leaf_set_prefill_original_oracle_replay_and_immutable_inputs_gpu(
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
    )


@skipUnlessBackends(["cute"])
def test_bt64_serial_leaf_set_original_oracle_gpu():
    _run_preparation_prefill(
        False,
        128,
        4,
        rectangular_leaf=True,
        leaf_count=4,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
        preparation_unroll=1,
    )
