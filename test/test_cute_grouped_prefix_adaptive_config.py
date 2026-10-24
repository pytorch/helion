from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _forbid_cuda_work
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .cute_population_contracts import checked_initial_population
from .test_cute_grouped_fp32_output import _grouped_fp32_output
from .test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel
from .test_cute_grouped_large_device_seeds import _kernel
import helion
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion._compiler.cute import cute_mma
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.exc import InvalidConfig
import helion.language as hl

KEY = "tcgen05_grouped_prefix_adaptive_n"
BALANCED = "tcgen05_grouped_balanced_workers"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _not_pure_fresh(
    a: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor, variant: int
) -> torch.Tensor:
    hl.specialize(variant)
    m, k = a.shape
    groups, n, _ = weights.shape
    bm = hl.register_block_size(256)
    bn = hl.register_block_size(128)
    bk = hl.register_block_size(64)
    out = (
        a[:, :n]
        if variant == 1
        else torch.empty((m, n), device=a.device, dtype=a.dtype)
    )
    for group_tile, rows, columns in hl.tile([groups, m, n], block_size=[1, bm, bn]):
        group = group_tile.index.sum()
        start = offsets[group]
        extent = offsets[group + 1] - offsets[group]
        row_index = start + rows.index
        valid = rows.index < extent
        acc = hl.full(
            [rows, columns], 1.0 if variant == 0 else 0.0, dtype=torch.float32
        )
        for inner in hl.tile(k, block_size=bk):
            left = hl.load(a, [row_index, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, weights[group, columns, inner].T)
        hl.store(
            out, [row_index, columns], acc.to(out.dtype), extra_mask=valid[:, None]
        )
    return out


def _args(
    *,
    groups: int = 16,
    m: int = 2048,
    n: int = 1536,
    k: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    offset_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=dtype),
        torch.empty((groups, n, k), dtype=dtype),
        torch.arange(groups + 1, dtype=offset_dtype) * (m // groups),
    )


def _recipes(spec, source_m_tile: int | None = 128) -> list[helion.Config]:
    return [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get(KEY) is True
        and (
            source_m_tile is None
            or seed.get("tcgen05_grouped_worklist_source_m_tile") == source_m_tile
        )
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("reserved", [0, 20])
def test_both_worker_policies_and_existing_reservations_survive_strict_transfer(
    dtype: torch.dtype, offset_dtype: torch.dtype, balanced: bool, reserved: int
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args(dtype=dtype, offset_dtype=offset_dtype))
        spec = bound.config_spec
        recipes = _recipes(spec)
        assert len(recipes) == 3
        recipe = deepcopy(
            next(seed for seed in recipes if seed.get(BALANCED, False) is balanced)
        )
        recipe.config["tcgen05_grouped_static_reserved_sms"] = reserved
        normalized = spec.normalized_config(recipe)
        flat, effective = spec.create_config_generation().strict_config_pair(recipe)
        assert flat
        assert normalized[KEY] is effective[KEY] is True
        assert effective.get(BALANCED, False) is balanced
        assert effective.get("tcgen05_grouped_static_reserved_sms", 0) == reserved
        profile = spec._cute_tcgen05_config.grouped_prefix_adaptive_n_profile
        assert profile is not None
        assert effective["tcgen05_ab_stages"] == profile.primary.ab_stages == 10
        assert effective["tcgen05_c_stages"] == profile.primary.c_stages == 3
        assert profile.alternate.ab_stages == 8 and profile.alternate.c_stages == 4
        assert profile.compiler_smem_bytes <= 232448
        assert profile.tmem_columns == 256 and profile.common_d_tile == (64, 64)
        assert [
            seed.get("tcgen05_grouped_static_reserved_sms", 0) for seed in recipes
        ] == [0, 0, 10]
        assert not any(
            seed.get("tcgen05_grouped_static_reserved_sms") == 20 for seed in recipes
        )


@pytest.mark.parametrize(
    "change",
    [
        {"cute_min_blocks_per_mp": 0},
        {"cute_min_blocks_per_mp": 2},
        {"tcgen05_cta_group": "one"},
        {"tcgen05_cluster_m2_one_cta_role_local": True},
        {"tcgen05_cluster_m": 1},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_grouped_worklist_source_m_tile": 256},
        {"tcgen05_grouped_worklist_orientation": "nm"},
        {"tcgen05_grouped_worklist_mma_n": 256},
        {"tcgen05_ab_stages": 9},
        {"tcgen05_c_stages": 4},
        {"tcgen05_acc_stages": 1},
        {"tcgen05_num_epi_warps": 8},
        {"tcgen05_grouped_ab_tile_k": 128},
        {"tcgen05_grouped_full_allocation_a": False},
        {"tcgen05_grouped_d_shift": False},
        {"tcgen05_l2_swizzle_size": 4},
        {
            "tcgen05_grouped_l2_swizzle_alternate": 2,
            "tcgen05_grouped_l2_swizzle_mn_ratio": 1,
        },
        {"tcgen05_layout_strategy": "explicit_epi_tile"},
        {"tcgen05_layout_overrides_epi_tile_m": 32},
        {"tcgen05_layout_overrides_epi_tile_n": 32},
        {"tcgen05_layout_overrides_smem_swizzle_a": 64},
        {"tcgen05_layout_overrides_smem_swizzle_b": 64},
        {"tcgen05_layout_overrides_d_store_box_n": 32},
        {"tcgen05_c_store_mode": "direct"},
        {"tcgen05_warp_spec_c_input_warps": 1},
        {"tcgen05_warp_spec_scheduler_warps": 1},
        {"tcgen05_consumer_regs": 224},
        {"pid_type": "flat"},
        {"num_sm_multiplier": 2},
    ],
)
def test_incompatible_explicit_recipes_fail_instead_of_measuring_a_default(
    change: dict,
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args())
        spec = bound.config_spec
        recipe = deepcopy(_recipes(spec)[0])
        recipe.config.update(change)
        with pytest.raises(InvalidConfig):
            spec.normalized_config(recipe)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_prefix_key_is_an_actual_boolean(value) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args())
        recipe = deepcopy(_recipes(bound.config_spec)[0])
        recipe.config[KEY] = value
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(recipe)


@pytest.mark.parametrize("groups,n,k", [(32, 1536, 2048), (16, 768, 768)])
def test_recipe_domain_is_resource_and_tensor_metadata_based(
    groups: int, n: int, k: int
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), _args(groups=groups, n=n, k=k)).config_spec
        assert spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported
        assert len(_recipes(spec)) == 3


@pytest.mark.parametrize(
    "groups,n,k", [(33, 1536, 2048), (16, 2304, 2048), (16, 1024, 2048)]
)
def test_outside_one_panel_or_prefix_domain_has_no_recipe(
    groups: int, n: int, k: int
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), _args(groups=groups, n=n, k=k)).config_spec
        assert spec._cute_tcgen05_config.grouped_prefix_adaptive_n_profile is None
        assert not _recipes(spec)
        if not spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported:
            assert KEY not in spec._flat_fields()


def test_unaligned_operands_cannot_claim_the_native_input_proof() -> None:
    args = list(_args())
    args[0] = torch.empty(2048 * 2048 + 1, dtype=torch.bfloat16)[1:].view(2048, 2048)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), tuple(args)).config_spec
        assert not spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported
        assert not _recipes(spec)


@pytest.mark.parametrize("variant", [0, 1])
def test_nonzero_accumulator_and_output_alias_do_not_claim_pure_fresh_output(
    variant: int,
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_not_pure_fresh, (*_args(), variant)).config_spec
        assert not spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported
        assert not _recipes(spec)


def test_split_sizes_and_different_output_atom_do_not_enter_prefix_recipe() -> None:
    split_args = (
        torch.empty((2048, 768), dtype=torch.bfloat16),
        torch.empty((8, 768, 768), dtype=torch.bfloat16),
        torch.full((8,), 256, dtype=torch.int64),
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        split = _cpu_bind(_device_split_sizes_kernel, split_args).config_spec
        output32 = _cpu_bind(_grouped_fp32_output, _args()).config_spec
        for spec in (split, output32):
            assert not spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported
            assert not _recipes(spec)


@pytest.mark.parametrize("num_sm,expected", [(152, 10), (128, 8), (8, 0)])
def test_one_sixteenth_reservation_uses_the_compile_architecture(
    num_sm: int, expected: int
) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), _args()).config_spec
        with patch.object(spec, "num_sm", num_sm):
            recipes = heuristics._tcgen05_grouped_prefix_adaptive_n_m128_seeds(spec)
        assert [
            recipe.get("tcgen05_grouped_static_reserved_sms", 0) for recipe in recipes
        ] == ([0, 0, expected] if expected else [0, 0])
        assert all(
            recipe.get("tcgen05_grouped_static_reserved_sms", 0) < num_sm
            for recipe in recipes
        )


def test_seed_integer_hints_do_not_override_the_actual_tensor_domain() -> None:
    original = cute_mma.analyze_tcgen05_grouped_worklist

    def invented_hints(*args, **kwargs):
        analysis = original(*args, **kwargs)
        if analysis is None:
            return None
        return replace(
            analysis,
            seed_facts=analysis.seed_facts._replace(n_hint=1536, groups_hint=16),
        )

    with (
        _forbid_cuda_work(),
        _target(),
        _mock_cuda_unavailable(),
        patch.object(
            cute_mma, "analyze_tcgen05_grouped_worklist", side_effect=invented_hints
        ),
    ):
        spec = _cpu_bind(_kernel(), _args(n=2304)).config_spec
        assert not spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported
        assert not _recipes(spec)


def test_unregistered_native_matmul_axes_do_not_claim_a_recipe() -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(
            _kernel(), _args(groups=4, m=257, n=768, k=256, dtype=torch.float16)
        ).config_spec
        assert not spec.cute_tcgen05_search_enabled
        assert spec._cute_tcgen05_config._matmul_block_indices() is None
        assert not spec._cute_tcgen05_config.grouped_prefix_adaptive_n_supported
        assert not _recipes(spec)


def test_optional_recipes_survive_legacy_empty_palette_without_default_promotion() -> (
    None
):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), _args()).config_spec
        env = SimpleNamespace(config_spec=spec)
        cls = heuristics.CuteTcgen05GroupedWorklistHeuristic
        with patch.object(cls, "_eligible_inputs", return_value=None):
            assert cls.is_eligible(env, object())
            assert cls._seed_configs(env, object()) == _recipes(spec, None)
            assert cls.get_seed_config(env, object()) is None


def test_existing_default_and_entire_v6_seed_prefix_are_unchanged() -> None:
    args = _args()
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        with patch.object(
            heuristics, "_tcgen05_grouped_prefix_adaptive_n_seeds", return_value=[]
        ):
            previous = _cpu_bind(_kernel(), args)
            old_seeds = deepcopy(previous.config_spec.compiler_seed_configs)
            old_default = deepcopy(previous.config_spec.compiler_default_config)
        bound = _cpu_bind(_kernel(), args)
        spec = bound.config_spec
        assert spec.compiler_seed_configs[: len(old_seeds)] == old_seeds
        assert spec.compiler_default_config == old_default
        assert spec.compiler_seed_configs[len(old_seeds) :] == _recipes(spec, None)
        assert len(_recipes(spec)) == 3
        control = deepcopy(_recipes(spec)[0])
        control.config.pop(KEY)
        disabled = deepcopy(control)
        disabled.config[KEY] = False
        assert spec.normalized_config(disabled).get(KEY) is None
        assert bound.to_code(disabled) == bound.to_code(control)


def test_native_full_population_contains_and_pins_both_effective_recipes() -> None:
    args = _args()
    saved = random.getstate()
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        kernel = _kernel()
        kernel.settings.autotune_effort = "full"
        bound = _cpu_bind(kernel, args)
        try:
            random.seed(309)
            kwargs = LFBOTreeSearch.get_kwargs_from_profile(
                get_effort_profile("full"), bound.settings
            )
            search = LFBOTreeSearch(
                bound,
                args,
                **kwargs,
            )
            rows = checked_initial_population(search)
            effective = {search.config_gen.unflatten(row) for row in rows}
            assert search.initial_population == kwargs["initial_population"]
            for raw in _recipes(bound.config_spec):
                expected = search.config_gen.strict_config_pair(raw)[1]
                assert expected[KEY] is True
                assert expected in effective
                assert expected in search._pinned_finalist_configs
        finally:
            random.setstate(saved)
