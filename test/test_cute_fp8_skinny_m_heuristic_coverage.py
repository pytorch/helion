from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.cute.autotune.heuristics.cute import CuteFp8GemmSkinnyMHeuristic
from helion._compiler.cute.backend import CuteBackend
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CuteVectorWidthSpec
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.config_spec import NumThreadsSpec
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def _copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile]
    return out


def _registry_seeds(static_m: int) -> tuple[list[helion.Config], ConfigSpec]:
    spec = ConfigSpec(backend=CuteBackend())
    for block_id in (0, 1):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=4096))
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=4096))
        spec.cute_vector_widths.append(
            CuteVectorWidthSpec(block_id=block_id, size_hint=4096)
        )
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=None,
        n_block_id=0,
        k_block_id=1,
        static_m=static_m,
        static_n=4096,
        static_k=4096,
        lhs_dtype=torch.float8_e4m3fn,
        rhs_dtype=torch.float8_e4m3fn,
    )
    spec.matmul_facts.append(fact)
    # Retain the original synthetic matmul fact and tunable axes. Registry
    # fact registration also needs real graph, environment and host contexts.
    with cpu_render_target():
        bound = cpu_bind(_copy, (torch.empty(1, 1),))
        env = bound.env
        with env, bound.host_function, patch.object(env, "config_spec", spec):
            configs = compiler_seed_configs(env, bound.host_function.device_ir)
    assert spec.matmul_facts == [fact]
    return configs, spec


def test_compiler_seed_configs_records_heuristic() -> None:
    configs, spec = _registry_seeds(static_m=1)
    assert [1, 256] in [config.config["block_sizes"] for config in configs]
    assert CuteFp8GemmSkinnyMHeuristic.name in spec.autotuner_heuristics
    assert any(
        config.config["block_sizes"] == [1, 256]
        and config.config["num_threads"] == [0, 32]
        and config.config["cute_vector_widths"] == [4, 8]
        for config in configs
    )


def test_compiler_seed_configs_skips_large_m() -> None:
    configs, spec = _registry_seeds(static_m=1024)
    assert CuteFp8GemmSkinnyMHeuristic.name not in spec.autotuner_heuristics
    assert [1, 256] not in [config.config["block_sizes"] for config in configs]
