from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.cute.autotune.flash_surface import enable_cute_flash_search
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.flash_knobs import FLASH_AUTOTUNE_CONFIG_KEYS
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none", disable_autotuner_heuristics=True)
def _copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile]
    return out


def test_disable_heuristics_keeps_structural_coverage() -> None:
    spec = ConfigSpec(backend=CuteBackend())
    assert spec.compiler_seed_timeout_retry_repetitions is None
    for block_id, size_hint in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=size_hint))
    enable_cute_flash_search(
        spec,
        head_dim=64,
        num_kv=64,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=True,
    )
    assert spec.compiler_seed_timeout_retry_repetitions is None
    # Fact registration still runs with heuristic seeds disabled. Give it a
    # real bound graph and its contexts, just as compilation does.
    with cpu_render_target():
        bound = cpu_bind(_copy, (torch.empty(1, 128, 128),))
        env = bound.env
        with env, bound.host_function, patch.object(env, "config_spec", spec):
            assert compiler_seed_configs(env, bound.host_function.device_ir) == []
    assert spec.compiler_seed_timeout_retry_repetitions is None
    assert spec.compiler_seed_configs == []
    assert spec.autotuner_heuristics == []

    config_gen = spec.create_config_generation()
    assert config_gen.seed_flat_config_pairs() == []
    assert not set(FLASH_AUTOTUNE_CONFIG_KEYS) & set(config_gen._config_value_priors)
    coverage = config_gen.extension.deterministic_population_configs()
    assert coverage
    population = config_gen.random_population(100)
    assert len(population) >= 100
    assert set(coverage) <= set(population)
