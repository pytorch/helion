from __future__ import annotations

import ast
from collections import Counter
import copy
import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_inputs
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_plain_root import _ordinary_root_options
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY as KEY
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR


def _context(*, count=2, initialized=False, **flags):
    graph = torch.fx.Graph()
    operand = graph.placeholder("operand")
    for _ in range(count):
        graph.call_function(dot, (operand, operand, operand if initialized else None))
    spec = SimpleNamespace(
        cute_chained_matmul_search_enabled=True,
        cute_chained_tcgen05_search_enabled=True,
        cute_chained_loop_search_enabled=False,
    )
    vars(spec).update(flags)
    return (
        cast("CompileEnvironment", SimpleNamespace(config_spec=spec)),
        cast("DeviceIR", SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])),
    )


def _seed(width=64, **options):
    return helion.Config.from_dict(
        {
            "block_sizes": [128, width],
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_pointwise_vectorize": False,
            "cute_chained_auxiliary_cache": False,
            **options,
        }
    )


def _insert(seeds, **options):
    env, ir = _context(**options)
    return CuteChainedMatmulHeuristic._with_root_snapshot_seeds(env, ir, seeds)


def test_two_adjacent_one_key_siblings_preserve_all_old_objects_and_order():
    first, second, third = (_seed(width) for width in (48, 96, 192))
    other = _seed(cute_chained_mma_schedule="coalesced")
    seeds = [other, first, first, second, other, third]
    original = copy.deepcopy([seed.config for seed in seeds])
    result = _insert(seeds)
    assert result == [
        other,
        first,
        _seed(48, **{KEY: 32}),
        first,
        second,
        _seed(96, **{KEY: 32}),
        other,
        third,
    ]
    retained = [seed for seed in result if not seed.config.get(KEY)]
    assert all(a is b for a, b in zip(seeds, retained, strict=True))
    assert Counter(map(id, retained)) == Counter(map(id, seeds))
    assert result[0] is seeds[0]
    assert [seed.config for seed in seeds] == original
    assert _insert(result) is result


def test_existing_sibling_is_not_duplicated_or_relocated():
    first, existing, second = _seed(), _seed(**{KEY: 32}), _seed(128)
    seeds = [first, second, existing]
    result = _insert(seeds)
    assert result == [first, second, _seed(128, **{KEY: 32}), existing]
    assert result[0] is first and result[1] is second and result[-1] is existing


@pytest.mark.parametrize(
    "options",
    [
        {"count": 0},
        {"count": 1},
        {"count": 3},
        {"initialized": True},
        {"cute_chained_matmul_search_enabled": False},
        {"cute_chained_tcgen05_search_enabled": False},
        {"cute_chained_loop_search_enabled": True},
    ],
)
def test_unsupported_regions_leave_the_exact_pool(options):
    seeds = [_seed()]
    assert _insert(seeds, **options) is seeds


@pytest.mark.parametrize(
    "override",
    [
        {"num_warps": 8},
        {"cute_chained_mma_schedule": "coalesced"},
        {"cute_chained_pointwise_vectorize": True},
        {"cute_chained_vector_group": True},
        {"cute_chained_pointwise_unroll": 2},
        {"cute_chained_pointwise_read_cache": True},
        {"cute_chained_pointwise_inplace_async": True},
        {"cute_chained_startup_transfer": "tma"},
        {"cute_chained_leaf_pipeline": "paired_tma"},
        {"cute_chained_tmem_free": "last_read"},
        {"cute_chained_seed_tile_columns": 32},
        {"cute_chained_pointwise_cache_layout": "xor"},
        {"cute_chained_scratch_layout": "xor"},
        {"cute_chained_auxiliary_cache": True},
        {"cute_chained_initialized_accumulator": True},
        {"cute_chained_late_rhs_reuse": True},
        {"cute_chained_direct_output": True},
        {"cute_chained_tmem_early_release": True},
        {"cute_chained_k_schedule": "serial64"},
        {"cute_chained_pointwise_cache_bytes": 4096},
        {"cute_chained_preparation_pipeline": True},
        {KEY: 32},
    ],
)
def test_incompatible_parent_is_not_repaired(override):
    seeds = [_seed(**override)]
    before = copy.deepcopy(seeds[0].config)
    assert _insert(seeds) is seeds
    assert seeds[0].config == before


def test_seed_option_filter_covers_original_ordinary_root_contract():
    tree = ast.parse(inspect.getsource(_ordinary_root_options))
    defaults = next(
        ast.literal_eval(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Tuple)
        and node.elts
        and all(
            isinstance(item, ast.Tuple)
            and len(item.elts) == 2
            and isinstance(item.elts[0], ast.Constant)
            and isinstance(item.elts[0].value, str)
            for item in node.elts
        )
    )
    for key, default in defaults:
        incompatible = not default if type(default) is bool else object()
        # Non-boolean values need not be valid configs: the seed filter must
        # never erase a caller's conflicting option before normal admission.
        seeds = [_seed(**{key: incompatible})]
        assert _insert(seeds) is seeds


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["plain", "scan"])
def test_real_root_seed_population_and_default_source(dtype, mode):
    args = (*_tcgen_inputs("cpu", dtype), mode)
    heuristic = CuteChainedMatmulHeuristic
    with _cpu_codegen():
        with patch.object(
            heuristic,
            "_with_root_snapshot_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            previous = _tcgen_chain._bind_isolated(args)
            old_configs = [
                dict(seed.config) for seed in previous.config_spec.compiler_seed_configs
            ]
            default = previous.config_spec.default_config()
            old_source = previous.to_code(default)
        current = _tcgen_chain._bind_isolated(args)
        spec = current.config_spec
        assert spec.default_config() == default
        assert current.to_code(default) == old_source
        seeds = spec.compiler_seed_configs
        assert [
            seed.config for seed in seeds if not seed.config.get(KEY)
        ] == old_configs
        generation = spec.create_config_generation()
        population = generation.random_population_flat(100)
        siblings = [seed for seed in seeds if seed.config.get(KEY) == 32]
        assert 0 < len(siblings) <= 2
        for seed in siblings:
            flat, normalized = generation.canonicalize_flat(generation.flatten(seed))
            parent = seeds[seeds.index(seed) - 1]
            _, parent_normalized = generation.canonicalize_flat(
                generation.flatten(parent)
            )
            assert normalized.config == parent_normalized.config | {KEY: 32}
            assert flat in population
            assert generation.unflatten(flat) == normalized
            code = current.to_code(normalized)
            assert "for chain_1_bridge_panel in cutlass.range(" in code
