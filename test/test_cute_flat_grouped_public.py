from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from types import FunctionType
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import checked_initial_population
from test.cute_population_contracts import with_flat_min_blocks_default
from test.test_cute_flat_grouped_host import rewritten_binding
from test.test_cute_flat_grouped_ir import _host
from test.test_cute_flat_grouped_ir import cuda_binding

import helion
from helion._compiler.autotuner_heuristics.cute_grouped_rna import (
    CuteGroupedRnaHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_grouped_rna import (
    interleave_grouped_rna_seeds,
)
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.tcgen05_flat_grouped_config import CONFIG_KEYS
from helion._compiler.cute.tcgen05_flat_grouped_config import K_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import WARPS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import (
    normalize_grouped_rna_config,
)
from helion._compiler.cute.tcgen05_tma_rn import CONVERSION_KEY
from helion._compiler.cute.tcgen05_tma_rn import TMA_RN
from helion._compiler.cute.tcgen05_tma_rn import WARP_RAW
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from pathlib import Path

    from helion.runtime.kernel import BoundKernel


def _config(bound: BoundKernel, warps: int = 4, block_k: int = 32) -> helion.Config:
    return helion.Config.from_dict(
        bound.config_spec._base_default_config().config
        | {WARPS_KEY: warps, K_KEY: block_k}
    )


def _source(bound: BoundKernel, config: helion.Config) -> str:
    # This is the public compiler entry. There is no emitter/launcher patch.
    with bound.env.suspend():
        return bound.to_code(config)


def _runtime_arguments(bound: BoundKernel) -> tuple[torch.Tensor, ...]:
    values = tuple(
        bound._runtime_tensor_refs_by_name[name]()
        for name in bound.kernel.signature.parameters
    )
    assert all(isinstance(value, torch.Tensor) for value in values)
    return tuple(value for value in values if isinstance(value, torch.Tensor))


def _native_source(source: str) -> str:
    candidates = [
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and "def _helion_native_grouped_rna(" in node.args[0].value
    ]
    assert len(candidates) == 1
    return candidates[0]


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
@pytest.mark.parametrize("block_k", [32, 64])
def test_public_choice_is_typed_serializable_and_emits_complete_schedule(
    warps: int, block_k: int
) -> None:
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        spec = bound.config_spec
        assert spec.cute_grouped_rna_k_choices == (32, 64)
        original = _config(bound, warps, block_k)
        config = spec.normalized_config(original)
        assert config[WARPS_KEY] == warps and config[K_KEY] == block_k
        restored = helion.Config.from_dict(json.loads(json.dumps(config.config)))
        assert spec.normalized_config(restored) == config
        generator = ConfigGeneration(spec)
        assert generator.unflatten(
            generator.flatten(config)
        ) == with_flat_min_blocks_default(spec, config)
        source = _source(bound, restored)
        native = _native_source(source)
        assert "'converter_warps': " + str(warps) in native
        assert "'ab_stage_count': " + str({32: 6, 64: 3}[block_k]) in native
        assert "_helion_cute_native_grouped_plan" in source
        assert "seq_offsets, jagged.view((30856, 128))" in source
        assert "dense.transpose(1, 2)" in source
        assert "quack" not in native.lower()


@pytest.mark.parametrize("warps", [-1, 3, 16, True, False, 4.0, "4", None])
def test_invalid_converter_counts_do_not_coerce_or_change_an_old_config(
    warps: object,
) -> None:
    config = {WARPS_KEY: warps, K_KEY: 32, "ordinary": [1, 2]}
    with pytest.raises(InvalidConfig, match=WARPS_KEY):
        normalize_grouped_rna_config(
            config.copy(), available_k=(32, 64), fix_invalid=False
        )
    normalize_grouped_rna_config(config, available_k=(32, 64), fix_invalid=True)
    assert config == {"ordinary": [1, 2]}


@pytest.mark.parametrize("block_k", [0, 16, 128, True, 32.0, "32", None])
def test_invalid_k_choices_keep_the_exact_stage_pair(block_k: object) -> None:
    config = {WARPS_KEY: 4, K_KEY: block_k}
    with pytest.raises(InvalidConfig, match=K_KEY):
        normalize_grouped_rna_config(
            config.copy(), available_k=(32, 64), fix_invalid=False
        )
    normalize_grouped_rna_config(config, available_k=(32, 64), fix_invalid=True)
    assert config == {WARPS_KEY: 4, K_KEY: 32}


@pytest.mark.parametrize("available", [(), (32,), (32, 64)])
def test_off_is_exactly_the_old_config_identity(available: tuple[int, ...]) -> None:
    config: dict[str, object] = {WARPS_KEY: 0, K_KEY: "irrelevant", "ordinary": [1, 2]}
    normalize_grouped_rna_config(config, available_k=available, fix_invalid=False)
    assert config == {"ordinary": [1, 2]}
    assert not (CONFIG_KEYS & config.keys())


@pytest.mark.parametrize(
    "case",
    [
        "dynamic",
        "misalign_a",
        "misalign_b",
        "misalign_bias",
        "offset_stride",
        "n_tail",
        "k_tail",
        "prefix_capacity",
        "ieee",
        "tf32x3",
        "wrong_stride",
        "shift_bias",
        "extra_math",
    ],
)
def test_ineligible_bindings_do_not_offer_or_silently_admit_the_choice(
    case: str,
) -> None:
    with cuda_binding(
        static=case != "dynamic",
        groups=1024 if case == "prefix_capacity" else 3,
        k=72 if case == "k_tail" else 128,
        n=130 if case == "n_tail" else 128,
        misalign={"misalign_a": 1, "misalign_b": 2, "misalign_bias": 3}.get(case),
        offsets_stride=2 if case == "offset_stride" else 1,
        dot_precision=case if case in ("ieee", "tf32x3") else "tf32",
        mode=case
        if case in ("wrong_stride", "shift_bias", "extra_math")
        else "ordinary",
    ) as bound:
        spec = bound.config_spec
        assert spec.cute_grouped_rna_k_choices == ()
        assert not any(seed.get(WARPS_KEY, 0) for seed in spec.compiler_seed_configs)
        assert not (CONFIG_KEYS & spec.default_config().config.keys())
        with pytest.raises(InvalidConfig, match="native grouped RNA requires"):
            spec.normalized_config(_config(bound))
        fixed = _config(bound)
        spec.normalize(fixed, _fix_invalid=True)
        # Public normalization preserves the absent optional zero; only a flat
        # config round-trip adds the explicit search coordinate.
        assert fixed == spec._base_default_config()


def test_renamed_generic_static_contract_gets_the_same_general_choice() -> None:
    with cuda_binding(static=True, groups=5, rows=131, k=96, n=132) as bound:
        assert bound.config_spec.cute_grouped_rna_k_choices == (32,)
        source = _source(bound, _config(bound, warps=2))
        assert "_helion_cute_native_grouped_plan" in source
        assert "routing, left.view((131, 96))" in source
        assert "right.transpose(1, 2)" in source
        assert "'flat_provider'" in _native_source(source)
        with pytest.raises(InvalidConfig, match=K_KEY):
            bound.config_spec.normalized_config(_config(bound, block_k=64))


def test_all_old_seeds_and_default_survive_in_their_original_order() -> None:
    with (
        patch.object(
            CuteGroupedRnaHeuristic, "register_facts", return_value=frozenset()
        ),
        cuda_binding(
            static=True, original=True, groups=256, rows=30856, k=128, n=128
        ) as old,
    ):
        old_seeds = [config.config for config in old.config_spec.compiler_seed_configs]
        old_default = old.config_spec.default_config().config
        old_source = _source(old, old.config_spec.default_config())
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        spec = bound.config_spec
        seeds = spec.compiler_seed_configs
        assert len(old_seeds) > 100
        assert [
            config.config
            for config in seeds
            if not config.get(WARPS_KEY, 0)
            and config.get(CONVERSION_KEY) not in (TMA_RN, WARP_RAW)
        ] == old_seeds
        assert seeds[0].config == old_seeds[0]
        assert spec.default_config().config == old_default
        assert _source(bound, spec.default_config()) == old_source
        assert [
            (config[WARPS_KEY], config[K_KEY])
            for config in seeds
            if config.get(WARPS_KEY, 0)
            and not config.get("cute_grouped_descriptor_policy")
        ] == [(warps, block_k) for block_k in (32, 64) for warps in (4, 8, 2, 1)]


@pytest.mark.parametrize("search_type", [PatternSearch, LFBOPatternSearch])
def test_real_initial_100_reaches_every_new_schedule_and_retains_old_first(
    search_type,
) -> None:
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        profile = get_effort_profile("full").lfbo_pattern_search
        assert profile is not None
        bound.settings.autotune_random_seed = 2026091719
        search = search_type(
            bound,
            _runtime_arguments(bound),
            initial_population=profile.initial_population,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        flat = checked_initial_population(search)
        configs = [search.config_gen.unflatten(row) for row in flat]
        # The 100-row base and the separate coverage suffix were checked above.
        assert [search.config_gen.flatten(config) for config in configs] == flat
        assert configs[0] == with_flat_min_blocks_default(
            bound.config_spec, bound.config_spec.default_config()
        )
        first = search.config_gen.unflatten(
            search.config_gen.flatten(bound.config_spec.compiler_seed_configs[0])
        )
        assert configs[1] == first
        choices = [config for config in configs if config.get(WARPS_KEY, 0)]
        assert {(config[WARPS_KEY], config[K_KEY]) for config in choices} == {
            (warps, block_k) for block_k in (32, 64) for warps in (1, 2, 4, 8)
        }
        # Check the actual selected modules, not merely membership in full seeds.
        for config in choices:
            assert "_helion_cute_native_grouped_plan" in _source(bound, config)
        assert any(
            config.get("cute_collective_mma")
            for config in configs
            if not config.get(WARPS_KEY, 0)
        )


def test_family_interleave_preserves_first_and_each_family_order() -> None:
    ordinary = [helion.Config(block_sizes=[index + 1]) for index in range(5)]
    native = [helion.Config.from_dict({WARPS_KEY: value}) for value in (4, 8, 2)]
    merged = interleave_grouped_rna_seeds([*ordinary, *native])
    assert merged == [
        ordinary[0],
        native[0],
        ordinary[1],
        native[1],
        ordinary[2],
        native[2],
        *ordinary[3:],
    ]
    assert interleave_grouped_rna_seeds(ordinary) is ordinary
    assert interleave_grouped_rna_seeds(native) is native


def test_explicit_user_seed_keeps_priority_over_the_new_compiler_family() -> None:
    with cuda_binding(static=True, original=True, n=128, k=128) as bound:
        spec = bound.config_spec
        generator = ConfigGeneration(spec)
        ordinary = [
            seed for seed in spec.compiler_seed_configs if not seed.get(WARPS_KEY, 0)
        ]
        explicit = ordinary[-1]
        transferred = generator.unflatten(generator.flatten(explicit))
        population = generator.random_population_flat(100, user_seed_configs=[explicit])
        configs = [generator.unflatten(row) for row in population]
        assert configs[0] == with_flat_min_blocks_default(spec, spec.default_config())
        assert configs[1] == transferred
        assert not configs[1].get(WARPS_KEY, 0)
        assert {
            (config[WARPS_KEY], config[K_KEY])
            for config in configs
            if config.get(WARPS_KEY, 0)
        } == {(warps, block_k) for block_k in (32, 64) for warps in (1, 2, 4, 8)}


def test_public_binding_cache_distinguishes_alignment_and_retains_metadata_facts() -> (
    None
):
    with cuda_binding(
        static=True, original=True, groups=3, rows=273, k=128, n=128
    ) as bound:
        assert bound.config_spec.cute_grouped_rna_k_choices == (32, 64)
        original_facts = bound.env.bound_runtime_input_specialization_results.copy()
        original_values = _runtime_arguments(bound)
        with bound.env.suspend():
            fresh = tuple(
                value.clone().requires_grad_(value.requires_grad)
                for value in original_values
            )
            assert bound.kernel.bind(fresh) is bound
            for index in (1, 2, 3):
                tensor = fresh[index]
                shifted = torch.empty(tensor.numel() + 1, dtype=tensor.dtype)[1:].view(
                    tensor.shape
                )
                shifted.requires_grad_(tensor.requires_grad)
                values = (*fresh[:index], shifted, *fresh[index + 1 :])
                other = bound.kernel.bind(values)
                assert other is not bound
                assert other.config_spec.cute_grouped_rna_k_choices == ()
                with (
                    other.env,
                    _host(other),
                    pytest.raises(InvalidConfig, match="native grouped RNA requires"),
                ):
                    other.config_spec.normalized_config(_config(other))
            assert bound.kernel.bind(fresh) is bound
        assert bound.env.bound_runtime_input_specialization_results == original_facts
        assert "_helion_cute_native_grouped_plan" in _source(bound, _config(bound))


def test_seed_fact_probe_restores_the_exact_prior_storage_snapshot() -> None:
    with cuda_binding(static=True) as bound:
        previous = bound.env.bound_runtime_input_specialization_results
        with bound.env.use_runtime_arg_values(
            dict(zip(bound.kernel.signature.parameters, bound.fake_args, strict=True))
        ):
            CuteGroupedRnaHeuristic.register_facts(bound.env, _host(bound).device_ir)
        assert bound.env.bound_runtime_input_specialization_results is previous
        # FakeTensor pointers cannot be used as real alignment evidence.
        assert bound.config_spec.cute_grouped_rna_k_choices == ()
        with bound.env.use_runtime_arg_values({}):
            CuteGroupedRnaHeuristic.register_facts(bound.env, _host(bound).device_ir)
        assert bound.config_spec.cute_grouped_rna_k_choices == (32, 64)
        assert bound.env.bound_runtime_input_specialization_results is previous


@pytest.mark.parametrize(
    "change", ["output_set", "alias_then_set", "metadata_mutation"]
)
def test_mutating_host_preludes_never_receive_native_public_seeds(
    tmp_path: Path, change: str
) -> None:
    with rewritten_binding(tmp_path, change) as bound:
        assert bound.config_spec.cute_grouped_rna_k_choices == ()
        assert not any(
            config.get(WARPS_KEY, 0)
            for config in bound.config_spec.compiler_seed_configs
        )
        with pytest.raises(InvalidConfig, match="native grouped RNA requires"):
            bound.config_spec.normalized_config(_config(bound))


def test_public_compile_cache_retains_original_inputs_and_fresh_host_allocations() -> (
    None
):
    initial_cuda_state = torch.cuda.is_initialized()
    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        config = helion.Config.from_dict(
            {
                "block_sizes": bound.config_spec._base_default_config().block_sizes,
                WARPS_KEY: 4,
                K_KEY: 32,
            }
        )
        source = _source(bound, config)
        arguments = _runtime_arguments(bound)
        with bound.env.suspend():
            compiled = bound.compile_config(config, allow_print=False)
            assert isinstance(compiled, FunctionType)
            assert bound.compile_config(config, allow_print=False) is compiled
            assert (
                bound.env.backend.generated_source_hash(compiled)
                == hashlib.sha256(source.encode()).hexdigest()
            )
            kernels = CuteBackend._compiled_kernels(compiled, bound.kernel.name)
            assert len(kernels) == 1
            sm_names = [
                alias.asname
                for node in ast.parse(source).body
                if isinstance(node, ast.ImportFrom) and node.module == "helion.runtime"
                for alias in node.names
                if alias.name == "get_num_sm" and alias.asname is not None
            ]
            assert len(sm_names) == 1
            launches = []

            def capture(kernel, grid, *args, **kwargs):
                launches.append((kernel, grid, args, kwargs))

            with (
                torch.no_grad(),
                patch.dict(compiled.__globals__, {sm_names[0]: lambda device: 148}),
            ):
                outputs = [compiled(*arguments, _launcher=capture) for _ in range(2)]
            assert len(launches) == 2
            for output, (kernel, grid, args, kwargs) in zip(
                outputs, launches, strict=True
            ):
                assert kernel is kernels[0]
                assert args[0] is arguments[0] and args[0].dtype is torch.int64
                assert args[1].data_ptr() == arguments[1].data_ptr()
                assert args[2].data_ptr() == arguments[2].data_ptr()
                assert args[2].stride() == (128 * 128, 1, 128)
                assert args[3] is arguments[3]
                assert args[4].data_ptr() == output.data_ptr()
                assert args[5].shape == (grid[2], 3, 16)
                assert kwargs["block"] == (32, 12, 1)
            assert outputs[0].data_ptr() != outputs[1].data_ptr()
            assert launches[0][2][5].data_ptr() != launches[1][2][5].data_ptr()
            assert torch.cuda.is_initialized() == initial_cuda_state


@pytest.mark.parametrize("scope", ["direct", "registered"])
@pytest.mark.parametrize(
    "field",
    [
        "block_sizes",
        "loop_orders",
        "load_eviction_policies",
        "cute_vector_widths",
        "cute_lane_layouts",
    ],
)
def test_native_seeds_own_their_nested_mutable_configuration(
    scope: str, field: str
) -> None:
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        spec = bound.config_spec
        base = spec._base_default_config()
        base_snapshot = deepcopy(base.config)
        with patch.object(spec, "_base_default_config", return_value=base):
            direct = CuteGroupedRnaHeuristic.get_seed_configs(
                bound.env, _host(bound).device_ir
            )
            fresh = CuteGroupedRnaHeuristic.get_seed_configs(
                bound.env, _host(bound).device_ir
            )
        registered = [
            seed for seed in spec.compiler_seed_configs if seed.get(WARPS_KEY, 0)
        ]
        # This regression covers the unchanged RNA family; TMA RN owns no
        # converter coordinate and has separate alias/seed coverage.
        direct = [seed for seed in direct if seed.get(WARPS_KEY, 0)]
        fresh = [seed for seed in fresh if seed.get(WARPS_KEY, 0)]
        seeds = direct if scope == "direct" else registered
        assert len(seeds) == len(fresh) == 16
        snapshots = [deepcopy(seed.config) for seed in seeds]
        hashes = [hash(seed) for seed in seeds]
        fresh_snapshots = [deepcopy(seed.config) for seed in fresh]
        ordinary = [
            seed for seed in spec.compiler_seed_configs if not seed.get(WARPS_KEY, 0)
        ]
        ordinary_snapshots = [deepcopy(seed.config) for seed in ordinary]
        value = seeds[0].config[field]
        assert isinstance(value, list) and value
        if field == "loop_orders":
            inner = value[0]
            assert isinstance(inner, list) and isinstance(inner[0], int)
            inner[0] += 1
        elif field in ("load_eviction_policies", "cute_lane_layouts"):
            value[0] = "changed"
        else:
            assert isinstance(value[0], int)
            value[0] += 1
        assert [
            index for index, seed in enumerate(seeds) if seed.config != snapshots[index]
        ] == [0]
        assert [
            index for index, seed in enumerate(seeds) if hash(seed) != hashes[index]
        ] == [0]
        assert [seed.config for seed in fresh] == fresh_snapshots
        assert [seed.config for seed in ordinary] == ordinary_snapshots
        assert base.config == base_snapshot
