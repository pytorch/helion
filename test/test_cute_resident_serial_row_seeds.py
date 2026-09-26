from __future__ import annotations

import ast
from copy import deepcopy
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.layer_norm import layer_norm_bwd
from examples.rms_norm import rms_norm_bwd
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics.cute_resident_reductions import (
    CuteResidentReductionHeuristic,
)
from helion._testing import skipUnlessBackends
from helion.autotuner import LFBOTreeSearch
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl
from helion.runtime.settings import default_autotuner_fn

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from helion.autotuner.base_search import PopulationMember
    from helion.runtime.kernel import BoundKernel


KEY = "cute_host_paired_sum"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "HELION_AUTOTUNE_EFFORT",
        "HELION_AUTOTUNER",
        "HELION_AUTOTUNER_INITIAL_POPULATION",
        "HELION_AUTOTUNE_CONFIG_OVERRIDES",
        "HELION_CUTE_REGION_FISSION",
        "HELION_CUTE_FULL_SLICE_MATMUL_TILING",
        "HELION_CUTE_SEGMENTED_MATMUL_TILING",
        "HELION_CUTE_FLATTEN_NESTED_REDUCTIONS",
        "HELION_CUTE_MATERIALIZE_TRANSFORMED_OPERANDS",
    ):
        monkeypatch.delenv(name, raising=False)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind(
    rows: int = 256,
    columns: int = 1024,
    *,
    kind: str = "rms",
    static: bool = True,
    dtype: torch.dtype = torch.float16,
    **settings: object,
) -> tuple[BoundKernel, tuple[object, ...]]:
    x = torch.empty((rows, columns), dtype=dtype, requires_grad=True)
    grad = torch.empty((rows, columns), dtype=dtype)
    weight = torch.empty(columns, dtype=dtype, requires_grad=True)
    if kind == "rms":
        function = rms_norm_bwd.fn
        args = (grad, x, weight, torch.empty((rows, 1), dtype=torch.float32))
    else:
        function = layer_norm_bwd.fn
        args = (
            grad,
            x,
            torch.empty(rows, dtype=torch.float32),
            torch.empty(rows, dtype=torch.float32),
            weight,
            True,
        )
    kernel = helion.kernel(
        function,
        backend="cute",
        static_shapes=static,
        autotune_effort="full",
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        **settings,
    )
    return _cpu_bind(kernel, args), args


def _family(bound: BoundKernel) -> list[helion.Config]:
    assert bound.host_function is not None
    with bound.env:
        return CuteResidentReductionHeuristic.serial_row_seed_configs(
            bound.env, bound.host_function.device_ir
        )


def _search(bound: BoundKernel, args: tuple[object, ...]) -> LFBOTreeSearch:
    with bound.env, patch.object(LocalAutotuneCache, "_generate_key"):
        cache = default_autotuner_fn(bound, args)
    assert isinstance(cache, LocalAutotuneCache)
    search = cache.autotuner
    assert type(search) is LFBOTreeSearch and search.initial_population == 100
    return search


@pytest.mark.parametrize("kind", ("rms", "layer"))
@pytest.mark.parametrize("static", (False, True))
def test_entire_old_seed_prefix_and_default_are_preserved(
    kind: str, static: bool
) -> None:
    bound, _args = _bind(
        kind=kind,
        static=static,
        dtype=torch.bfloat16 if kind == "layer" else torch.float16,
    )
    spec = bound.config_spec
    current = deepcopy(spec.compiler_seed_configs)
    promoted = deepcopy(spec.compiler_default_config)
    with bound.env:
        default = spec.default_config()
        assert bound.host_function is not None
        with (
            patch.object(
                CuteResidentReductionHeuristic,
                "serial_row_seed_configs",
                return_value=[],
            ),
            patch.object(
                CuteResidentReductionHeuristic,
                "pipeline_depth_seed_configs",
                return_value=[],
            ),
        ):
            old = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        assert spec.compiler_default_config == promoted
        assert spec.default_config() == default
    assert bound.host_function is not None
    with bound.env:
        deeper = CuteResidentReductionHeuristic.pipeline_depth_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    # Local-tree counterparts are appended after this entire legacy prefix.
    current = [
        seed
        for seed in current
        if not seed.config.get("cute_reduction_local_tree")
        and not seed.config.get("cute_reduction_row_schedule")
    ]
    old = [
        seed
        for seed in old
        if not seed.config.get("cute_reduction_local_tree")
        and not seed.config.get("cute_reduction_row_schedule")
    ]
    assert current == [*old, *_family(bound), *deeper]
    assert len(current) == len(old) + 3 + len(deeper)
    assert [group.mechanism for group in spec.compiler_coverage_groups] == [
        "cute.resident_row_consumption",
        "cute.resident_terminal_product",
    ]


@pytest.mark.parametrize("columns", (32, 64, 128, 256, 512, 1024))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_typed_feature_width_and_real_slot_mapping_survive_normalization(
    columns: int, dtype: torch.dtype
) -> None:
    bound, _args = _bind(17, columns, dtype=dtype)
    family = _family(bound)
    assert [seed[KEY] for seed in family] == ["off", "mapped", "narrow"]
    assert bound.host_function is not None
    spec = bound.config_spec
    with bound.env:
        plan = CuteResidentReductionHeuristic._plan(
            bound.env, bound.host_function.device_ir
        )
        assert plan is not None
        generation = ConfigGeneration(spec)
        for seed in family:
            assert dict(
                zip(
                    (item.block_id for item in spec.num_threads),
                    seed.num_threads,
                    strict=True,
                )
            ) == {
                plan.coarse_block: 1,
                plan.row_block: 1,
                plan.reduction_block: columns,
            }
            flat, normalized = generation.strict_config_pair(seed)
            assert generation.unflatten(flat) == normalized
            assert generation.unflatten(generation.flatten(seed)) == normalized
            assert normalized.block_sizes == seed.block_sizes == [4, 4]
            assert normalized.num_threads == seed.num_threads
            assert normalized["cute_cluster_n"] == 1
            assert normalized["cute_min_blocks_per_mp"] == 0
            assert normalized["cute_reduction_reloads"] == ["auto"]
            assert normalized["cute_reduction_group_rows"] == 1
            assert normalized[KEY] == seed[KEY]
    code = bound.to_code(family[-1])
    assert f"block=({columns}, 1, 1)" in code
    assert "m_block = 4" in code
    assert "_cute_try_single_sum_cast(" in code


@pytest.mark.parametrize("kind", ("rms", "layer"))
@pytest.mark.parametrize("static", (False, True))
def test_family_reaches_actual_first_full_lfbo_benchmark(
    kind: str, static: bool
) -> None:
    bound, args = _bind(
        kind=kind,
        static=static,
        dtype=torch.bfloat16 if kind == "layer" else torch.float16,
    )
    search = _search(bound, args)
    search._autotune_metrics = AutotuneMetrics()
    delivered = []

    class HeldBenchmark(Exception):
        pass

    def hold(members: Sequence[PopulationMember], *, desc: str) -> None:
        assert desc == "Initial population"
        delivered.extend(deepcopy(member.config) for member in members)
        raise HeldBenchmark

    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
        patch.object(search, "benchmark_population", side_effect=hold),
        pytest.raises(HeldBenchmark),
    ):
        random.seed(2026092701)
        search._autotune()
    assert len(delivered) == 100
    for seed in _family(bound):
        flat, normalized = search.config_gen.strict_config_pair(seed)
        assert search.config_gen.unflatten(flat) == normalized
        assert normalized in delivered
    assert {config[KEY] for config in delivered} == {"off", "mapped", "narrow"}


@pytest.mark.parametrize("columns", (1, 16, 31, 33, 1025, 2048, 4096))
def test_outside_full_thread_lattice_does_not_add_the_family(columns: int) -> None:
    bound, _args = _bind(64, columns)
    assert not _family(bound)


@pytest.mark.parametrize("field", ("coarse", "row", "threads"))
def test_inaccessible_tile_or_thread_fragment_is_not_silently_repaired(
    field: str,
) -> None:
    bound, _args = _bind()
    spec = bound.config_spec
    assert bound.host_function is not None
    with bound.env:
        plan = CuteResidentReductionHeuristic._plan(
            bound.env, bound.host_function.device_ir
        )
    assert plan is not None
    if field == "threads":
        item = spec.num_threads[
            spec.num_threads.block_id_to_index(plan.reduction_block)
        ]
        key, value = "size_hint", 512
    else:
        block_id = plan.coarse_block if field == "coarse" else plan.row_block
        item = spec.block_sizes[spec.block_sizes.block_id_to_index(block_id)]
        key, value = "max_size", 2
    with patch.object(item, key, value):
        assert not _family(bound)


def test_disabled_heuristics_keep_explicit_config_support() -> None:
    with patch.object(
        CuteResidentReductionHeuristic,
        "serial_row_seed_configs",
        wraps=CuteResidentReductionHeuristic.serial_row_seed_configs,
    ) as hook:
        bound, _args = _bind(disable_autotuner_heuristics=True)
    hook.assert_not_called()
    assert not bound.config_spec.compiler_seed_configs
    assert bound.config_spec.cute_host_paired_sum_available
    assert "_cute_try_single_sum_cast(" in bound.to_code(_family(bound)[-1])


@pytest.mark.parametrize("mode", ("off", "mapped", "narrow"))
def test_explicit_host_mode_remains_authoritative(mode: str) -> None:
    bound, args = _bind(autotune_config_overrides={KEY: mode})
    search = _search(bound, args)
    with bound.env:
        random.seed(2026092701)
        rows = search._generate_initial_population_flat()
        configs = [search.config_gen.unflatten(row) for row in rows]
    assert all(config[KEY] == mode for config in configs)
    assert any(config.block_sizes == [4, 4] for config in configs)


def test_explicit_geometry_wins_over_the_new_seed() -> None:
    bound, args = _bind(autotune_config_overrides={"block_sizes": [2, 1]})
    search = _search(bound, args)
    with bound.env:
        random.seed(2026092701)
        rows = search._generate_initial_population_flat()
        configs = [search.config_gen.unflatten(row) for row in rows]
    assert all(config.block_sizes == [2, 1] for config in configs)


def test_family_owns_nested_lists_and_helper_does_not_change_producer() -> None:
    bound, _args = _bind()
    family = _family(bound)
    original = deepcopy(family)
    family[1].config["num_threads"][0] = 2
    assert family[0] == original[0] and family[2] == original[2]
    producers = []
    for seed in original:
        tree = ast.parse(bound.to_code(seed))
        producers.append(
            [
                ast.dump(node)
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.decorator_list
            ]
        )
    assert producers[0] == producers[1] == producers[2]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _unreturned_feature_partial(x: torch.Tensor) -> torch.Tensor:
    coarse = hl.register_block_size(x.size(0))
    out = torch.empty_like(x)
    for group in hl.tile(x.size(0), block_size=coarse):
        for rows in hl.tile(group.begin, group.end):
            values = x[rows, :].to(torch.float32)
            out[rows, :] = (values - values.mean(-1)[:, None]).to(x.dtype)
    return out


def test_structural_family_without_host_sum_has_only_ordinary_producer() -> None:
    bound = _cpu_bind(
        _unreturned_feature_partial, (torch.empty((17, 128), dtype=torch.float16),)
    )
    assert not bound.config_spec.cute_host_paired_sum_available
    family = _family(bound)
    assert len(family) == 1 and KEY not in family[0]
    assert "_cute_try_single_sum_cast(" not in bound.to_code(family[0])
