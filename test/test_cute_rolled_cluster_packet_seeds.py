from __future__ import annotations

import ast
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target
from test.test_cute_full_chunk_row_seeds import bind
from test.test_cute_full_chunk_row_seeds import column_sum
from test.test_cute_full_chunk_row_seeds import normalize_rows

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_specialization_facts
from helion._compiler.autotuner_heuristics.cute_rolled_cluster_packet import (
    CuteRolledClusterPacketHeuristic,
)
from helion._compiler.cute.prefetch_reduction_packets import _containers
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def cpu_only() -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    with _cpu_target(), _forbid_native_compile() as native:
        yield
        assert all(mock.call_count == 0 for mock in native)
    assert torch.cuda.is_initialized() == initialized


def _seeds(bound: BoundKernel) -> list[helion.Config]:
    with bound.env:
        return CuteRolledClusterPacketHeuristic._cluster_seed_configs(
            bound.env, bound.host_function.device_ir
        )


def _fused_rows(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    output = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    for rows in hl.tile(m):
        value = x[rows, :].to(torch.float32) + residual[rows, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(value * value, -1) + 1e-6)
        output[rows, :] = (value * scale[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
        residual_out[rows, :] = value.to(x.dtype)
    return output, residual_out


def _fixed_unit_jagged_row_sum(
    data: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    rows, columns = offsets.size(0) - 1, offsets.size(1)
    output = torch.empty((rows, columns), dtype=data.dtype, device=data.device)
    for row, column in hl.tile([rows, columns], block_size=[1, 1]):
        starts = offsets[row, column]
        ends = offsets[row.index + 1, column]
        lengths = ends - starts
        acc = hl.zeros([row, column], dtype=data.dtype)
        for jagged in hl.jagged_tile(lengths):
            indices = starts[:, :, None] + jagged.index[None, None, :]
            acc += data[indices].sum(dim=2)
        output[row, column] = acc
    return output


def test_fixed_unit_jagged_row_declines_before_scalar_extent_specialization() -> None:
    kernel = helion.kernel(
        _fixed_unit_jagged_row_sum, backend="cute", config={"block_sizes": [4]}
    )
    args = (
        torch.arange(9, dtype=torch.float32),
        torch.tensor([[0, 1], [3, 1], [4, 6], [8, 9]]),
    )
    with patch.object(
        CuteRolledClusterPacketHeuristic, "get_seed_configs", return_value=[]
    ):
        control = kernel._bind_isolated(args)
    bound = kernel._bind_isolated(args)
    spec = bound.config_spec
    row = spec.block_sizes[0].block_id
    assert bound.env.is_jagged_tile(row)
    assert bound.env.block_sizes[row].size is None
    assert not _seeds(bound)
    assert spec.compiler_seed_configs == control.config_spec.compiler_seed_configs
    assert spec.compiler_default_config == control.config_spec.compiler_default_config
    config = helion.Config(block_sizes=[4])
    assert bound.to_code(config) == control.to_code(config)


@pytest.mark.parametrize(
    "dtype,size", [(torch.bfloat16, 7168), (torch.float16, 6144), (torch.float32, 8196)]
)
def test_masked_packet_seeds_strictly_roundtrip_and_emit_cluster_prefetch(
    dtype: torch.dtype, size: int
) -> None:
    bound = bind(4, size, dtype)
    seeds = _seeds(bound)
    assert 0 < len(seeds) <= 16
    prefetch_sources = 0
    for seed in seeds:
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            flat, effective = generation.strict_config_pair(seed)
            assert generation.strict_unflatten(flat) == effective == seed
        assert seed.config["cute_vector_widths"][0] == (
            4 if dtype == torch.float32 else 8
        )
        assert seed.config["cute_reduction_reloads"] == ["register_prefetch"]
        source = bound.to_code(seed)
        assert "_cute_grouped_reduce_cluster(" in source
        assert f"< {size}" in source
        prefetch_sources += "_consume_prefetch_" in source
        assert "_helion_cute_cluster_shape" in source
        kernel = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and any(
                ast.unparse(decorator) == "cute.kernel"
                for decorator in node.decorator_list
            )
        )
        classified = _containers(kernel.body, {})
        assert classified is not None and classified[2] <= 128
    assert prefetch_sources > 0


def test_exact_cluster_packet_slices_stay_unmasked() -> None:
    bound = bind(4, 8192, torch.bfloat16)
    seeds = _seeds(bound)
    assert seeds
    assert all("_cute_grouped_reduce_cluster(" in bound.to_code(seed) for seed in seeds)
    assert all("mask_1" not in bound.to_code(seed) for seed in seeds)


def test_full_fused_population_contains_both_packet_profiles() -> None:
    kernel = helion.kernel(
        _fused_rows,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        autotune_initial_population_strategy="from_random",
        autotune_seed_configs=None,
    )
    args = (
        torch.empty((16, 7168), dtype=torch.bfloat16),
        torch.empty((16, 7168), dtype=torch.bfloat16),
        torch.empty(7168, dtype=torch.bfloat16),
    )
    bound = kernel._bind_isolated(args)
    spec = bound.config_spec
    seeds = _seeds(bound)
    assert len(seeds) == 10
    neutral = next(
        seed
        for seed in seeds
        if seed.num_threads == [1, 128]
        and seed.config["cute_cluster_n"] == 8
        and seed.config["cute_pdl"]
    )
    retained = next(
        seed
        for seed in seeds
        if seed.num_threads == [1, 256]
        and seed.config["cute_cluster_n"] == 4
        and not seed.config["cute_pdl"]
    )
    assert (
        neutral.reduction_loops == [1024]
        and neutral.config["cute_min_blocks_per_mp"] == 2
    )
    assert neutral.config["load_eviction_policies"] == [""] * 8
    assert (
        retained.reduction_loops == [2048]
        and retained.config["cute_min_blocks_per_mp"] == 1
    )
    assert retained.config["load_eviction_policies"] == ["first", *["l2_last"] * 7]
    profile = get_effort_profile("full").pattern_search
    assert profile is not None and profile.initial_population_strategy == "from_random"
    with bound.env:
        generation = ConfigGeneration(spec)
        random.seed(152100)
        population = [
            generation.unflatten(flat)
            for flat in generation.random_population_flat(profile.initial_population)
        ]
        assert all(seed in population for seed in seeds)
    for seed in (neutral, retained):
        source = bound.to_code(seed)
        assert (
            "_consume_prefetch_" in source and "_cute_grouped_reduce_cluster(" in source
        )


def test_legacy_seed_prefix_and_default_are_unchanged() -> None:
    with patch.object(
        CuteRolledClusterPacketHeuristic, "get_seed_configs", return_value=[]
    ):
        previous = bind(4, 7168, torch.bfloat16)
    current = bind(4, 7168, torch.bfloat16)
    old = previous.config_spec.compiler_seed_configs
    assert current.config_spec.compiler_seed_configs[: len(old)] == old
    assert len(current.config_spec.compiler_seed_configs) > len(old)
    with previous.env:
        default = previous.config_spec.default_config()
        reference = previous.config_spec.autotune_reference_config()
    with current.env:
        assert current.config_spec.default_config() == default
        assert current.config_spec.autotune_reference_config() == reference
    assert current.to_code(default) == previous.to_code(default)
    assert all(current.to_code(seed) == previous.to_code(seed) for seed in old)


@pytest.mark.parametrize("layout", ["strided", "row_stride"])
def test_non_packet_input_layouts_decline(layout: str) -> None:
    assert not _seeds(bind(4, 7168, torch.bfloat16, layout=layout))


def test_real_pointer_offsets_use_existing_alignment_cache_key() -> None:
    kernel = helion.kernel(normalize_rows, backend="cute", static_shapes=True)
    aligned = torch.empty((4, 7168), dtype=torch.bfloat16)
    offset = torch.empty(4 * 7168 + 1, dtype=torch.bfloat16)[1:].view(4, 7168)
    weight = torch.empty(7168, dtype=torch.bfloat16)
    aligned_bound = kernel.bind((aligned, weight))
    offset_bound = kernel.bind((offset, weight))
    assert aligned_bound is not offset_bound
    # Tracing may normalize FakeTensor storage offsets to zero. The existing
    # pointer-residue key owns real address admission, separately from these
    # geometry hints and their shape/stride specialization facts.
    assert _seeds(aligned_bound) and _seeds(offset_bound)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float64])
def test_unsupported_dtypes_decline(dtype: torch.dtype) -> None:
    assert not _seeds(bind(4, 7168, dtype))


def test_dynamic_or_high_grid_rows_decline() -> None:
    assert not _seeds(bind(4, 7168, torch.bfloat16, static=False))
    assert not _seeds(bind(149, 7168, torch.bfloat16))


@pytest.mark.parametrize("reason", ["atomic", "indexed", "warp", "thread_domain"])
def test_side_effect_and_unreachable_thread_guards(reason: str) -> None:
    bound = bind(4, 7168, torch.bfloat16)
    spec = bound.config_spec
    ir = bound.host_function.device_ir
    if reason == "atomic":
        with patch.object(ir, "has_atomic_ops", return_value=True):
            assert not _seeds(bound)
    elif reason == "indexed":
        with patch.object(
            spec, "cute_indexed_reduction_block_ids", {spec.reduction_loops[0].block_id}
        ):
            assert not _seeds(bound)
    elif reason == "warp":
        with patch.object(spec, "max_reduction_threads", 32):
            assert not _seeds(bound)
    else:
        item = spec.num_threads.block_id_lookup(spec.reduction_loops[0].block_id)
        with patch.object(item, "_fragment", return_value=EnumFragment((0, 32, 64))):
            assert not _seeds(bound)


def test_noncanonical_and_non_full_slice_loads_decline() -> None:
    bound = helion.kernel(
        column_sum, backend="cute", static_shapes=True
    )._bind_isolated((torch.empty((4, 7168)),))
    assert not _seeds(bound)
    bound = bind(4, 7168, torch.bfloat16)
    ir = bound.host_function.device_ir
    graph = ir.graphs[ir.rolled_reductions[0].original_graph_id].graph
    node = next(
        node for node in graph.nodes if node.target is helion.language.memory_ops.load
    )
    original = node.args
    node.args = (node.args[0], [*node.args[1][:-1], slice(1, None)])
    try:
        assert not _seeds(bound)
    finally:
        node.args = original


def test_cache_budget_charges_all_original_loads_and_reserve() -> None:
    bound = bind(4, 7168, torch.bfloat16)
    with patch.object(
        CuteRolledClusterPacketHeuristic,
        "_load_metadata",
        return_value=(torch.bfloat16, 100, ()),
    ):
        assert not _seeds(bound)


def test_seed_cache_facts_are_local_to_the_fired_family() -> None:
    assert compiler_seed_specialization_facts("cute", ()) == frozenset()
    assert compiler_seed_specialization_facts(
        "cute", (CuteRolledClusterPacketHeuristic.name,)
    ) == frozenset({"device_num_sm", "input_tensor_metadata"})
