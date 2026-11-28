from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test.cute_population_contracts import with_flat_min_blocks_default

import helion
from helion._compiler.cute.autotune.heuristics.cute_materialized import (
    CuteMaterializedMmaHeuristic,
)
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing.cute_codegen import bind_full_slice_matmul as _bind
from helion._testing.cute_codegen import stage_sources as _sources
from helion._testing.cute_structural_kernels import _se_args
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = pytest.mark.cpu_only

squeeze_and_excitation_net_fwd = import_example(
    "squeeze_and_excitation_net"
).squeeze_and_excitation_net_fwd
if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _native_cpu_target() -> Generator[None, None, None]:
    support = CuteMmaSupport(universal=True, warp_f16bf16=True, tcgen05_f16bf16=True)
    with cpu_render_target(mma_support=support, smem_budget_bytes=232448):
        yield


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(128, 256, 64), (73, 88, 56)])
def test_fission_native_seeds_emit_both_matrix_products(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn, _se_args(shape, dtype), fission=True
    )
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    assert spec.tcgen05.search_enabled
    assert spec.tcgen05.matmul_block_ids is None
    assert spec._cute_tcgen05_config.materialized_matmul_block_ids == (
        (0, 1, 2),
        (3, 4, 5),
    )
    with bound.env, host:
        seeds = CuteMaterializedMmaHeuristic.get_seed_configs(bound.env, host.device_ir)
        assert seeds
        generation = ConfigGeneration(spec)
        for seed in seeds:
            normalized = bound._normalized_config_copy(
                helion.Config.from_dict(spec.default_config().config | seed.config)
            )
            assert generation.unflatten(
                generation.flatten(normalized)
            ) == with_flat_min_blocks_default(spec, normalized)
            assert len(normalized.l2_groupings) == 2
    stages = _sources(bound.to_code(seeds[0]))
    assert len(stages) == 2
    assert all("cute.gemm(" in stage and "tcgen05_" in stage for stage in stages)
    assert (
        "tcgen05_warp_spec_ab_load_warps" in bound.config_spec.default_config().config
    )
    if shape == (73, 88, 56):
        assert all(seed.pid_type == "flat" for seed in seeds)


def test_each_materialized_matrix_keeps_its_own_tile_slots() -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn, _se_args((256, 256, 64)), fission=True
    )
    config = bound.config_spec.default_config()
    first = helion.Config.from_dict(
        config.config | {"block_sizes": [64, 32, 64, 128, 128, 32]}
    )
    second = helion.Config.from_dict(
        config.config | {"block_sizes": [128, 64, 32, 128, 128, 32]}
    )
    first_sources = _sources(bound.to_code(first))
    second_sources = _sources(bound.to_code(second))
    assert first_sources[0] != second_sources[0]
    assert first_sources[1] == second_sources[1]


def test_unfissioned_roots_do_not_get_materialized_mma_search() -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn, _se_args((128, 256, 64)), fission=False
    )
    assert not bound.config_spec._cute_tcgen05_config.materialized_matmul_block_ids


def test_fp32_materialized_regions_keep_existing_precision_policy() -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn,
        _se_args((128, 256, 64), torch.float32),
        fission=True,
    )
    assert not bound.config_spec._cute_tcgen05_config.materialized_matmul_block_ids


@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
@pytest.mark.parametrize("grouping", [1, 4])
def test_auxiliary_producer_uses_second_region_matrix_coordinates(
    order: list[int], grouping: int
) -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn,
        _se_args((256, 256, 64)),
        fission=True,
    )
    config = helion.Config.from_dict(
        bound.config_spec.default_config().config
        | {
            "block_sizes": [128, 16, 16, 128, 16, 16],
            "loop_orders": [[0, 1], order],
            "l2_groupings": [1, grouping],
            "pid_type": "persistent_interleaved",
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_strategy": "role_local_with_scheduler",
            "tcgen05_warp_spec_scheduler_warps": 1,
            "tcgen05_warp_spec_c_input_warps": 1,
        }
    )
    source = _sources(bound.to_code(config))[1]
    producer = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.Name)
        and node.test.id.startswith("tcgen05_c_input_warp_valid")
    )
    assignments = {
        node.targets[0].id: node.value
        for node in producer.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    # The second region's M/N axes are 3/4, even if launch order swaps them.
    # Both coordinates must be decomposed before the auxiliary input is read;
    # using raw scheduler coordinates reads unrelated rows/columns.
    for suffix, block_id in (("m", 3), ("n", 4)):
        offset_name = f"tile_offset_{block_id}"
        assert offset_name in assignments
        value = assignments[f"tcgen05_aux_tile_{suffix}"]
        assert offset_name in {
            node.id for node in ast.walk(value) if isinstance(node, ast.Name)
        }
        assert "tcgen05_work_tile_smem" not in ast.unparse(value)


@pytest.mark.parametrize("first,second", [(1, 2), (2, 1)])
def test_second_region_scheduler_uses_its_actual_group(first: int, second: int) -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn, _se_args((640, 512, 256)), fission=True
    )
    config = helion.Config.from_dict(
        bound.config_spec.default_config().config
        | {
            "block_sizes": [256, 256, 128, 256, 256, 128],
            "loop_orders": [[0, 1], [0, 1]],
            "l2_groupings": [first, second],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_strategy": "role_local_with_scheduler",
            "tcgen05_warp_spec_scheduler_warps": 1,
            "tcgen05_warp_spec_c_input_warps": 1,
        }
    )
    source = _sources(bound.to_code(config))[1]
    vote = next(
        line.value
        for line in ast.walk(ast.parse(source))
        if isinstance(line, ast.Assign)
        and isinstance(line.targets[0], ast.Name)
        and line.targets[0].id == "tcgen05_scheduler_warp_full_tile"
    )
    decide = compile(ast.Expression(vote), "<second-region-tile>", "eval")
    # 640 rows give two full tiles and one fringe for each of two N tiles.
    # Group size two visits both columns of the first two M tiles first.
    rows = (0, 1, 0, 1, 2, 2) if second == 2 else (0, 1, 2, 0, 1, 2)
    for pid, expected_row in enumerate(rows):
        scope = {
            "_BLOCK_SIZE_3": 256,
            "_BLOCK_SIZE_4": 256,
            "cutlass": SimpleNamespace(Int32=int, Int64=int),
            "tcgen05_scheduler_warp_work_tile": SimpleNamespace(
                tile_idx=(2 * (pid % 3), pid // 3, 0)
            ),
            "min": min,
        }
        assert eval(decide, {"__builtins__": {}}, scope) == (expected_row < 2)
