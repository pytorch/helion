from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_population_contracts import checked_initial_population
from test.test_autotuner_heuristics import _grouped_worklist_kernel_body
from test.test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test.test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel
from test.test_cute_shared_rhs_grouped import _plans

import helion
from helion._compiler.autotuner_heuristics.cute import (
    _TCGEN05_GROUPED_OUTPUT_RING_SMEM_HEADROOM,
)
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05GroupedSource64Heuristic,
)
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05GroupedWorklistHeuristic,
)
from helion._compiler.autotuner_heuristics.cute import _block_size_value_reachable
from helion._compiler.autotuner_heuristics.cute import (
    _tcgen05_grouped_output_ring_seeds,
)
from helion._compiler.autotuner_heuristics.cute import _tcgen05_grouped_worklist_config
from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_cluster4_carrier,
)
from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_paired_clc_carrier,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_CONSUMER_REGS_DEFAULT
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion._testing import skipUnlessBackends
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from typing import Any

    from helion.runtime.kernel import BoundKernel


_HELPER = (
    "helion._compiler.autotuner_heuristics.cute._tcgen05_grouped_output_ring_seeds"
)


@pytest.fixture(scope="module", autouse=True)
def _cpu_target() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(threads)


def _bind(
    values: tuple[torch.Tensor, ...],
    *,
    static_shapes: bool = False,
    function: Callable[..., object] = grouped_gemm_jagged.fn,
) -> BoundKernel[Any]:
    return helion.kernel(
        function,
        backend="cute",
        autotune_effort="none",
        static_shapes=static_shapes,
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        cute_materialize_transformed_operands=True,
    )._bind_isolated(values)


def _values(shape: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    groups, rows, k, n = (
        (4, 640, 128, 128),
        (8, 2560, 512, 512),
        (16, 10240, 1024, 1024),
    )[shape]
    return (
        torch.empty((rows, k), dtype=dtype),
        torch.empty((k, n), dtype=dtype),
        torch.empty((groups + 1,), dtype=torch.int32),
    )


def _population(
    bound: BoundKernel[Any],
    values: tuple[torch.Tensor, ...],
    seeds: list[helion.Config],
) -> tuple[list[helion.Config], list[helion.Config]]:
    state = random.getstate()
    try:
        random.seed(691)
        with bound.env, patch.object(bound.config_spec, "compiler_seed_configs", seeds):
            search = LFBOTreeSearch(
                bound,
                values,
                **cast(
                    "dict[str, Any]",
                    LFBOTreeSearch.get_kwargs_from_profile(
                        get_effort_profile("full"), bound.settings
                    ),
                ),
            )
            assert (
                search.initial_population_strategy
                is InitialPopulationStrategy.FROM_RANDOM
            )
            flats = checked_initial_population(search)
            return (
                [search.config_gen.unflatten(flat) for flat in flats],
                [config for _, config in search.config_gen.seed_flat_config_pairs()],
            )
    finally:
        random.setstate(state)


@pytest.mark.parametrize("shape", range(3))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_original_shapes_cover_output_ring_and_preserve_seed_prefix(
    shape: int, dtype: torch.dtype, static_shapes: bool
) -> None:
    values = _values(shape, dtype)
    with patch(_HELPER, return_value=[]):
        old_bound = _bind(values, static_shapes=static_shapes)
    previous = old_bound.config_spec.compiler_seed_configs
    bound = _bind(values, static_shapes=static_shapes)
    seeds = bound.config_spec.compiler_seed_configs
    assert bound.host_function is not None
    with bound.env:
        source64 = CuteTcgen05GroupedSource64Heuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        row_union = grouped_row_union_cluster4_carrier(
            bound.env, bound.host_function.device_ir
        )
        paired = grouped_row_union_paired_clc_carrier(
            bound.env, bound.host_function.device_ir
        )
    # Independent carriers follow the worklist family. Check each exact suffix
    # in reverse order before checking the old prefix and output-ring append.
    previous_prefix, seed_prefix = previous, seeds
    if paired is not None:
        startup = copy.deepcopy(paired)
        startup.config["tcgen05_ab_startup_prefill"] = True
        suffix = [paired, startup]
        assert previous_prefix[-2:] == seed_prefix[-2:] == suffix
        previous_prefix, seed_prefix = previous_prefix[:-2], seed_prefix[:-2]
        assert all(seed not in suffix for seed in (*previous_prefix, *seed_prefix))
    if row_union is not None:
        assert previous_prefix[-1] == seed_prefix[-1] == row_union
        previous_prefix, seed_prefix = previous_prefix[:-1], seed_prefix[:-1]
        assert row_union not in previous_prefix and row_union not in seed_prefix
    if source64 is not None:
        assert previous_prefix[-1] == seed_prefix[-1] == source64
        previous_prefix, seed_prefix = previous_prefix[:-1], seed_prefix[:-1]
        assert all(
            seed.get("tcgen05_grouped_worklist_source_m_tile") != 64
            for seed in (*previous_prefix, *seed_prefix)
        )
    assert seed_prefix[: len(previous_prefix)] == previous_prefix
    assert (
        bound.config_spec.compiler_default_config
        == old_bound.config_spec.compiler_default_config
    )
    if static_shapes and shape == 0:
        # Static small-M planning already limits this axis below M256. Seed
        # coverage must not expand the existing native layout admission.
        assert not _block_size_value_reachable(bound.config_spec, 0, 256)
        assert seeds == previous == []
        return
    grouped_previous = [
        seed
        for seed in previous_prefix
        if seed.get("tcgen05_grouped_mode") == "worklist_nm"
    ]
    assert len(grouped_previous) == 8
    assert len(seeds) == len(previous) + 4
    added = seed_prefix[len(previous_prefix) :]
    assert {
        (
            seed["tcgen05_grouped_worklist_source_m_tile"],
            cast("list[int]", seed["block_sizes"])[2],
        )
        for seed in added
    } == {(224, 64), (256, 64), (224, 128), (256, 128)}
    assert bound.host_function is not None
    with bound.env:
        assert (
            CuteTcgen05GroupedWorklistHeuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
            == grouped_previous[0]
        )
    population, normalized = _population(bound, values, list(seeds))
    old_population, _ = _population(bound, values, previous)
    # checked_initial_population checks the exact 100-row base and each addition.
    assert population[0] == old_population[0]
    normalized_added = [
        seed for seed in normalized if seed.get("tcgen05_c_stages") == 4
    ]
    assert len(normalized_added) == 4
    assert all(seed in population for seed in normalized_added)
    for seed in added:
        assert seed["tcgen05_ab_stages"] == 3
        assert seed["tcgen05_c_stages"] == 4
        assert seed["tcgen05_consumer_regs"] == TCGEN05_CONSUMER_REGS_DEFAULT
        assert seed.get("tcgen05_grouped_static_reserved_sms", 0) == 0
        source = bound.to_code(seed)
        plans = _plans(source)
        assert plans[0]["shared_rhs"] is True
        assert plans[0]["scheduler_mode"] == "device_group_search"
        assert plans[0]["device_layout_kind"] == "offsets"
        assert (
            next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")[
                "c_stage_count"
            ]
            == 4
        )
        assert "cute.gemm(" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offsets", [False, True])
@skipUnlessBackends(["cute"])
def test_rank3_device_metadata_reuses_the_same_output_ring_family(
    dtype: torch.dtype, offsets: bool
) -> None:
    values = (
        torch.empty((2560, 128), dtype=dtype),
        torch.empty((8, 256, 128), dtype=dtype),
        torch.empty((9 if offsets else 8,), dtype=torch.int32),
    )
    kernel = _device_offsets_kernel if offsets else _device_split_sizes_kernel
    bound = _bind(values, static_shapes=True, function=kernel.fn)
    added = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.get("tcgen05_grouped_mode") == "worklist_nm"
        and seed.get("tcgen05_c_stages") == 4
    ]
    # These source programs deliberately fix BK64; do not add an unreachable BK.
    assert len(added) == 2
    for seed in added:
        assert cast("list[int]", seed["block_sizes"])[2] == 64
        source = bound.to_code(seed)
        plans = _plans(source)
        assert plans[0].get("shared_rhs", False) is False
        assert plans[0]["scheduler_mode"] == "device_group_search"
        assert (
            next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")[
                "c_stage_count"
            ]
            == 4
        )


@skipUnlessBackends(["cute"])
def test_external_worklist_keeps_its_existing_policy() -> None:
    values = (
        torch.empty((448, 128), dtype=torch.bfloat16),
        torch.empty((2, 256, 128), dtype=torch.bfloat16),
        torch.tensor([[0, 0, 224, 224], [1, 224, 224, 224]], dtype=torch.int32),
    )
    with patch(_HELPER, side_effect=AssertionError("external worklist is unchanged")):
        bound = _bind(
            values, static_shapes=True, function=_grouped_worklist_kernel_body
        )
    assert bound.config_spec.compiler_seed_configs
    assert all(
        config.get("tcgen05_c_stages") == 2
        for config in bound.config_spec.compiler_seed_configs
    )


@pytest.mark.parametrize(
    "rows,k,n,groups,dtype",
    [
        (640, 128, 127, 4, torch.bfloat16),
        (640, 130, 128, 4, torch.bfloat16),
        (640, 128, 128, 4, torch.float32),
        (640, 0, 128, 4, torch.bfloat16),
        (0, 128, 128, 4, torch.bfloat16),
        (640, 128, 0, 4, torch.bfloat16),
        (640, 128, 128, 0, torch.bfloat16),
    ],
)
@skipUnlessBackends(["cute"])
def test_unproven_or_empty_inputs_do_not_gain_grouped_ring_seeds(
    rows: int, k: int, n: int, groups: int, dtype: torch.dtype
) -> None:
    bound = _bind(
        (
            torch.empty((rows, k), dtype=dtype),
            torch.empty((k, n), dtype=dtype),
            torch.empty((groups + 1,), dtype=torch.int32),
        )
    )
    assert not any(
        config.get("tcgen05_grouped_mode") == "worklist_nm"
        and config.get("tcgen05_c_stages") == 4
        for config in bound.config_spec.compiler_seed_configs
    )


@pytest.mark.parametrize("capacity", [0, 32768, 200000, 232448])
def test_seed_capacity_uses_the_physical_worklist_shape(capacity: int) -> None:
    seed = _tcgen05_grouped_worklist_config(256, 128, 3, 240, runtime_direct=False)
    original = copy.deepcopy(seed.config)
    result = _tcgen05_grouped_output_ring_seeds(
        [seed], group_count=8, dtype_bytes=2, capacity_bytes=capacity
    )
    assert seed.config == original
    if capacity <= 32768:
        assert result == []
        return
    assert len(result) == 1
    expected_stages = 2 if capacity == 200000 else 3
    assert result[0]["tcgen05_ab_stages"] == expected_stages
    required = tcgen05_grouped_worklist_smem_bytes(
        group_count=8,
        device_split_sizes=True,
        sched_stage_count=1,
        bm=256,
        bn=256,
        bk=128,
        dtype_bytes=2,
        ab_stages=expected_stages,
        acc_stages=2,
        c_stages=4,
        cluster_m=2,
    )
    assert required + _TCGEN05_GROUPED_OUTPUT_RING_SMEM_HEADROOM <= capacity


@pytest.mark.parametrize("groups,dtype_bytes", [(0, 2), (8, 1), (8, 4)])
def test_missing_shape_or_precision_facts_decline_ring_seeds(
    groups: int, dtype_bytes: int
) -> None:
    seed = _tcgen05_grouped_worklist_config(224, 128, 3, 240, runtime_direct=False)
    assert (
        _tcgen05_grouped_output_ring_seeds(
            [seed],
            group_count=groups,
            dtype_bytes=dtype_bytes,
            capacity_bytes=232448,
        )
        == []
    )
