"""CPU checks of the ragged edge family's 128-row CTA pairs on the search surface.

``tcgen05_cta_group="two"`` on ``block_m=128`` is the two-CTA tile cuBLAS-class
kernels run on ragged GEMMs below the 256-row pair's reach. The seeds and the
sampled pairs must survive search normalization (the cta_group validator and
the cluster_m=2 fixer used to project both onto the 256x256 pair), checked
here on a fake SM100 target with 148 SMs.
"""

from __future__ import annotations

import copy

import pytest
import torch

from test.test_cute_tcgen05_split_k_search import _bind
from test.test_cute_tcgen05_split_k_search import _gemm_fp8
from test.test_cute_tcgen05_split_k_search import _samples
from test.test_cute_tcgen05_split_k_search import _sm100_target

import helion
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("shape", ((1000, 1000, 1000), (777, 999, 555)))
def test_ragged_edge_family_keeps_its_128_row_pairs(shape) -> None:
    with _sm100_target():
        bound = _bind(*shape)
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert tcgen05.plain_edge_pair128_admitted()
        gen = ConfigGeneration(spec)
        pair_seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.get("tcgen05_cta_group") == "two"
        ]
        assert [seed.config["block_sizes"][:3] for seed in pair_seeds] == [
            [128, 128, 64],
            [128, 128, 128],
        ]
        for seed in pair_seeds:
            _flat, normalized = gen.strict_config_pair(seed)
            assert normalized.config["block_sizes"] == seed.config["block_sizes"]
            assert normalized.config["tcgen05_cta_group"] == "two"
            assert normalized.config["tcgen05_cluster_m"] == 2
        # A sampled explicit pair on a tile the pair cannot run (an odd N
        # and K tile, one CTA) is projected onto the 128x128 pair instead of
        # being handed to the 256x256 projection.
        sample = copy.deepcopy(pair_seeds[0].config)
        sample.update(
            {
                "block_sizes": [128, 32, 32],
                "tcgen05_cluster_m": 1,
                "tcgen05_ab_stages": 2,
            }
        )
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        assert normalized["tcgen05_cta_group"] == "two"
        assert normalized["tcgen05_cluster_m"] == 2
        assert normalized["block_sizes"][:2] == [128, 128]
        assert normalized["block_sizes"][2] in (64, 128)
        assert normalized["pid_type"] == "persistent_interleaved"
        assert any(
            config.get("tcgen05_cta_group") == "two"
            and config["block_sizes"][:2] == [128, 128]
            for config in _samples(gen, 400)
        )


def test_a_grid_of_256_row_pairs_keeps_the_pair_family_as_is() -> None:
    # 4096^3 fills the machine with 256-row pairs: the ragged 128-row pair
    # family is not admitted (its full-tile paired family has its own seeds).
    with _sm100_target():
        bound = _bind(4096, 4096, 4096)
        tcgen05 = bound.config_spec._cute_tcgen05_config
        assert not tcgen05.plain_edge_pair128_admitted()
        assert tcgen05._plain_edge_pair128_seed_configs(helion.Config()) == []


@helion.kernel(backend="cute", static_shapes=True)
def _bias_gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n]).to(out.dtype)
    return out


def test_a_half_machine_of_256_row_pairs_keeps_the_edge_family_on_them() -> None:
    # 5000^3 is ragged (the edge family runs), but its 400 pairs of 256x256
    # put a CTA on every SM: the 128-row pair brings nothing and is not
    # admitted.
    with _sm100_target():
        bound = _bind(5000, 5000, 5000)
        tcgen05 = bound.config_spec._cute_tcgen05_config
        assert 2 in tcgen05.cluster_m_search_choices
        assert not tcgen05.plain_edge_pair128_admitted()


def test_fp8_and_auxiliary_epilogues_keep_the_256_row_pair() -> None:
    # The 128-row pair is validated for plain 16-bit GEMMs only. fp8 keeps
    # the 256x256 pair count in the planner's work gate too (16 pairs at
    # 1000^3, under a quarter wave: no two-CTA search at all), while an
    # auxiliary epilogue is detected after planning, so the bf16 bias GEMM
    # is admitted to the edge family on the 128x128 count and runs the
    # family's 256x256 pairs.
    with _sm100_target():
        a = torch.empty(1000, 1000, dtype=torch.float8_e4m3fn)
        b = torch.empty(1000, 1000, dtype=torch.float8_e4m3fn).T
        fp8 = _gemm_fp8.bind((a, b)).config_spec._cute_tcgen05_config
        assert not fp8.plain_edge_pair128_admitted()
        assert fp8.cluster_m_search_choices == (1,)
        a = torch.empty(1000, 1000, dtype=torch.bfloat16)
        b = torch.empty(1000, 1000, dtype=torch.bfloat16)
        bias = torch.empty(1000, dtype=torch.bfloat16)
        aux = _bias_gemm.bind((a, b, bias)).config_spec
        assert aux.cute_tcgen05_aux_kernel_detected
        assert not aux._cute_tcgen05_config.plain_edge_pair128_admitted()
        assert 2 in aux._cute_tcgen05_config.cluster_m_search_choices


def test_a_sampled_pair_on_a_wide_tile_is_reset_to_auto() -> None:
    # The ragged edge family's explicit pair is the 128-row tile; a sampled
    # "two" on a 256-row tile is reset to ``auto`` (the family's own
    # 256x256 encoding) and repaired by the fixer body, as before the
    # 128-row pair was admitted.
    with _sm100_target():
        bound = _bind(1000, 1000, 1000)
        gen = ConfigGeneration(bound.config_spec)
        sample = dict(gen.unflatten(gen.random_flat()).config)
        sample.update(
            {
                "block_sizes": [256, 256, 64],
                "tcgen05_cta_group": "two",
                "tcgen05_cluster_m": 2,
                "pid_type": "persistent_interleaved",
            }
        )
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        assert normalized["tcgen05_cta_group"] == "auto"
        assert normalized["tcgen05_cluster_m"] == 2
        assert normalized["block_sizes"][:2] == [256, 256]
