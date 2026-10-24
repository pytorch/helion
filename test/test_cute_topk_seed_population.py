"""Compound small-row top-k hints reach the real full search population."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_population_contracts import checked_initial_population
from test.cute_population_contracts import with_flat_min_blocks_default

import helion
from helion.autotuner import LFBOTreeSearch
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _generic_weights(logits: torch.Tensor, k: hl.constexpr, renorm: hl.constexpr):
    weights = torch.empty((logits.size(0), k), device=logits.device)
    indices = torch.empty((logits.size(0), k), device=logits.device, dtype=torch.int32)
    for row in hl.tile(logits.size(0)):
        scores = logits[row, :]
        selected, ids = torch.topk(scores, k, dim=-1, sorted=True)
        normalizer_scores = selected.float() if renorm else scores.float()
        maximum = normalizer_scores.amax(dim=-1, keepdim=True)
        denominator = (normalizer_scores - maximum).exp().sum(dim=-1, keepdim=True)
        weights[row, :] = (selected.float() - maximum).exp() / denominator
        indices[row, :] = ids.to(torch.int32)
    return weights, indices


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renorm", [False, True])
def test_compound_small_row_seed_reaches_full_random_population(dtype, renorm):
    cuda_initialized_before = torch.cuda.is_initialized()
    args = (torch.empty((4, 512), dtype=dtype), 10, renorm)
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile() as forbidden,
        patch(
            "torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")
        ) as lazy_init,
    ):
        bound = _cpu_bind(_generic_weights, args)
        spec = bound.config_spec
        assert spec.cute_topk_search_enabled
        assert spec.compiler_default_config is None
        before_default = spec.default_config()
        assert before_default["cute_topk_lanes_per_row"] == 16
        key_dtype = "int64" if dtype == torch.float32 else "int32"
        expected = {
            "cute_topk_lanes_per_row": 32,
            "cute_topk_rows_per_block": 1,
            "cute_topk_vector_width": 4,
            "cute_topk_output_vector_width": 1,
            "cute_topk_value_mode": "decode",
            "cute_topk_key_dtype": key_dtype,
            "cute_topk_rank_mode": "ordinal",
            "cute_topk_selection_layout": "distributed",
            "cute_topk_sort_network": "compact_pruned",
            "cute_topk_key_encoder": "dsl",
            "cute_topk_defer_value_gathers": False,
            "cute_topk_merge_schedule": "balanced",
        }
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if all(seed.config.get(key) == value for key, value in expected.items())
        ]
        assert len(seeds) == 1
        seed = seeds[0]
        normalized = spec.normalized_config(deepcopy(seed))
        assert normalized == seed
        compatibility_seed = with_flat_min_blocks_default(spec, before_default)
        bound.settings.autotune_seed_configs = [compatibility_seed]
        bound.settings.autotune_best_available_max_configs = 0
        kwargs = LFBOTreeSearch.get_kwargs_from_profile(
            get_effort_profile("full"), bound.settings
        )
        kwargs["initial_population_strategy"] = InitialPopulationStrategy.FROM_RANDOM
        search = LFBOTreeSearch(bound, args, **kwargs)
        assert search.initial_population == 100
        errors = []
        pairs = search.config_gen.seed_flat_config_pairs(errors.append)
        assert not errors
        assert any(config == normalized for flat, config in pairs)
        population = checked_initial_population(search)
        configs = [search.config_gen.unflatten(flat) for flat in population]
        assert normalized in configs
        assert spec.default_config() == before_default
        assert not any(mock.call_count for mock in forbidden)
        assert lazy_init.call_count == 0
        assert torch.cuda.is_initialized() == cuda_initialized_before
