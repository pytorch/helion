from __future__ import annotations

from copy import deepcopy
import random
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _forbid_cuda_work
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .cute_population_contracts import checked_initial_population
from .test_cute_grouped_large_device_seeds import _kernel
from .test_cute_grouped_prefix_adaptive_config import _args
from .test_cute_grouped_prefix_adaptive_config import _recipes
from helion._compiler.autotuner_heuristics import cute as heuristics
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.exc import InvalidConfig

KEY = "tcgen05_grouped_prefix_adaptive_n"
SOURCE = "tcgen05_grouped_worklist_source_m_tile"
BALANCED = "tcgen05_grouped_balanced_workers"
RESERVED = "tcgen05_grouped_static_reserved_sms"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("balanced", [False, True])
def test_m256_pair_survives_strict_transfer(dtype, offset_dtype, balanced):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(
            _kernel(), _args(m=8192, dtype=dtype, offset_dtype=offset_dtype)
        )
        spec = bound.config_spec
        recipes = _recipes(spec, 256)
        assert len(recipes) == 2
        assert [seed.get(BALANCED, False) for seed in recipes] == [False, True]
        assert all(seed.get(RESERVED, 0) == 0 for seed in recipes)
        raw = next(seed for seed in recipes if seed.get(BALANCED, False) is balanced)
        flat, effective = spec.create_config_generation().strict_config_pair(raw)
        assert flat and effective[KEY] is True
        assert effective[SOURCE] == 256
        assert effective["tcgen05_grouped_worklist_mma_n"] == 224
        assert (
            effective["tcgen05_ab_stages"] == 7 and effective["tcgen05_c_stages"] == 2
        )
        assert effective.get(BALANCED, False) is balanced
        assert effective.get(RESERVED, 0) == 0
        assert effective["tcgen05_grouped_l2_swizzle_alternate"] == 4
        assert effective["tcgen05_grouped_l2_swizzle_mn_ratio"] == 1
        profile = spec._cute_tcgen05_config.get_grouped_prefix_adaptive_n_profile(
            effective
        )
        assert profile is not None
        assert (
            profile.primary.n,
            profile.primary.ab_stages,
            profile.primary.c_stages,
        ) == (224, 7, 2)
        assert (
            profile.alternate.n,
            profile.alternate.ab_stages,
            profile.alternate.c_stages,
        ) == (256, 6, 3)
        assert profile.row_tile == 256 and profile.tmem_columns == 512
        assert profile.common_d_tile == (128, 32)
        assert profile.layout_kind == "operand_payload"
        assert profile.arena_bytes == 231424 and profile.compiler_smem_bytes == 232448
        assert (
            profile.primary.packed_d_offset,
            profile.primary.packed_a_offset,
            profile.primary.packed_b_offset,
        ) == (0, 16384, 131072)
        assert (
            profile.alternate.packed_d_offset,
            profile.alternate.packed_a_offset,
            profile.alternate.packed_b_offset,
        ) == (0, 24576, 122880)


@pytest.mark.parametrize(
    "change",
    [
        {"tcgen05_grouped_worklist_mma_n": 256},
        {"tcgen05_ab_stages": 6},
        {"tcgen05_c_stages": 3},
        {"tcgen05_acc_stages": 1},
        {"cute_min_blocks_per_mp": 2},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_cta_group": "one"},
        {"tcgen05_cluster_m2_one_cta_role_local": True},
        {"tcgen05_grouped_l2_swizzle_alternate": None},
        {"tcgen05_grouped_l2_swizzle_alternate": 2},
        {"tcgen05_grouped_l2_swizzle_mn_ratio": 2},
        {"tcgen05_grouped_full_allocation_a": False},
        {"tcgen05_grouped_d_shift": False},
        {"tcgen05_layout_overrides_epi_tile_m": 128},
        {"tcgen05_layout_overrides_epi_tile_n": 32},
        {"tcgen05_layout_overrides_d_store_box_n": 32},
        {"tcgen05_c_store_mode": "simt"},
        {"tcgen05_grouped_ab_tile_k": 128},
        {"tcgen05_warp_spec_c_input_warps": 1},
        {"tcgen05_consumer_regs": 224},
    ],
)
def test_incoherent_m256_recipe_is_rejected(change):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), _args(m=8192)).config_spec
        raw = deepcopy(_recipes(spec, 256)[0])
        raw.config.update(change)
        with pytest.raises(InvalidConfig):
            spec.normalized_config(raw)


@pytest.mark.parametrize(
    "groups,m,n,k",
    [(2, 1024, 1024, 256), (16, 8192, 1536, 2048), (32, 4096, 1792, 512)],
)
def test_m256_domain_uses_real_metadata_and_supports_complete_box_tails(
    groups, m, n, k
):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), _args(groups=groups, m=m, n=n, k=k)).config_spec
        assert (
            spec._cute_tcgen05_config.get_grouped_prefix_adaptive_n_profile(
                {SOURCE: 256}
            )
            is not None
        )
        assert len(_recipes(spec, 256)) == 2


@pytest.mark.parametrize(
    "groups,n,k", [(33, 1536, 2048), (16, 2048, 2048), (16, 1544, 2048), (16, 1536, 96)]
)
def test_m256_outside_domain_has_no_profile(groups, n, k):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        spec = _cpu_bind(_kernel(), _args(groups=groups, m=8192, n=n, k=k)).config_spec
        assert (
            spec._cute_tcgen05_config.get_grouped_prefix_adaptive_n_profile(
                {SOURCE: 256}
            )
            is None
        )
        assert not _recipes(spec, 256)


def test_all_old_seeds_and_default_precede_m256_append():
    args = _args(m=8192)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        with patch.object(
            heuristics, "_tcgen05_grouped_prefix_adaptive_n_m256_seeds", return_value=[]
        ):
            before = _cpu_bind(_kernel(), args).config_spec
            old = deepcopy(before.compiler_seed_configs)
            default = deepcopy(before.compiler_default_config)
        spec = _cpu_bind(_kernel(), args).config_spec
        assert spec.compiler_seed_configs[: len(old)] == old
        assert spec.compiler_default_config == default
        assert spec.compiler_seed_configs[len(old) :] == _recipes(spec, 256)
        assert [seed.get(SOURCE) for seed in _recipes(spec, None)] == [
            128,
            128,
            128,
            256,
            256,
        ]
        assert [seed.get(RESERVED, 0) for seed in _recipes(spec, None)] == [
            0,
            0,
            10,
            0,
            0,
        ]


def test_full_population_preserves_prefix_rng_and_appends_all_effective_m256_seeds():
    args = _args(m=8192)
    saved = random.getstate()
    try:
        with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
            old_kernel = _kernel()
            old_kernel.settings.autotune_effort = "full"
            with patch.object(
                heuristics,
                "_tcgen05_grouped_prefix_adaptive_n_m256_seeds",
                return_value=[],
            ):
                before = _cpu_bind(old_kernel, args)
            kernel = _kernel()
            kernel.settings.autotune_effort = "full"
            after = _cpu_bind(kernel, args)
            kwargs = LFBOTreeSearch.get_kwargs_from_profile(
                get_effort_profile("full"), after.settings
            )
            random.seed(823)
            first = LFBOTreeSearch(before, args, **kwargs)
            old_rows = checked_initial_population(first)
            old_rng = random.getstate()
            random.seed(823)
            second = LFBOTreeSearch(after, args, **kwargs)
            new_rows = checked_initial_population(second)
            assert new_rows[: len(old_rows)] == old_rows
            assert random.getstate() == old_rng
            assert (
                first.initial_population
                == second.initial_population
                == kwargs["initial_population"]
            )
            effective = {second.config_gen.unflatten(row) for row in new_rows}
            for raw in _recipes(after.config_spec, 256):
                normalized = second.config_gen.strict_config_pair(raw)[1]
                assert normalized in effective
                assert normalized in second._pinned_finalist_configs
    finally:
        random.setstate(saved)


def test_m256_disabled_key_uses_exact_native_source():
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_kernel(), _args(m=8192))
        raw = deepcopy(_recipes(bound.config_spec, 256)[0])
        raw.config.pop(KEY)
        disabled = deepcopy(raw)
        disabled.config[KEY] = False
        assert bound.to_code(raw) == bound.to_code(disabled)
