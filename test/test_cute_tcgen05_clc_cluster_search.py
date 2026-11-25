"""CPU checks of the CLC-persistent 4-CTA cluster on the search surface.

The plain CLC seed's 2x2 twin (``tcgen05_cluster_n=2`` on the
``role_local_with_scheduler`` pipeline, Quack's dynamic-persistent 2x2) must
reach the search as CLC: the search's CLC candidate gate used to take the
full-tile family's 2x1 aux-TMA gate, which projected every 4-CTA CLC config
onto the static schedule.
"""

from __future__ import annotations

import pytest

from helion._compiler.cute.tcgen05_config import repair
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.mark.parametrize(
    "shape",
    [
        (8192, 8192, 8192),
        # (the transposed lowering of the 4096x50264x2048 RR product: an M
        # edge on 256-row pairs)
        (50264, 4096, 2048),
    ],
)
def test_the_clc_2x2_seed_reaches_the_search_as_clc(
    shape: tuple[int, int, int],
) -> None:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(*shape, rhs_k_major=False))
        spec = bound.config_spec
        with bound.env:
            gen = ConfigGeneration(spec)
            projected = [
                dict(config.config) for _, config in gen.seed_flat_config_pairs()
            ]
        clc_2x2 = [
            config
            for config in projected
            if config.get("tcgen05_persistence_model") == "clc_persistent"
            and config.get("tcgen05_cluster_n") == 2
        ]
        assert clc_2x2
        # a sampled CLC 2x2 config keeps its schedule through the search repair
        sample = dict(clc_2x2[0])
        repair.fix_search_config(spec._cute_tcgen05_config, sample)
        assert sample["tcgen05_persistence_model"] == "clc_persistent"
        assert sample["tcgen05_cluster_n"] == 2


@pytest.mark.parametrize(
    "shape",
    [
        (8192, 8192, 8192),
        # a K tail (8152 = 127 x 64 + 24): the edge K-tail family
        (8192, 8192, 8152),
    ],
)
def test_the_m_paired_2x2_seed_keeps_its_static_projection(
    shape: tuple[int, int, int],
) -> None:
    # The 4-CTA CLC allowance covers the 256-row pair tile: the M-paired
    # 512-row 2x2 tile (two 256-row subtiles per work tile) has no CLC run on
    # a GPU, so its CLC-shaped seed still reaches the search static, over a
    # whole K and over a K tail alike.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(*shape, rhs_k_major=False))
        spec = bound.config_spec
        with bound.env:
            gen = ConfigGeneration(spec)
            projected = [
                dict(config.config) for _, config in gen.seed_flat_config_pairs()
            ]
        m_paired_2x2 = [
            config
            for config in projected
            if config["block_sizes"][0] == 512 and config.get("tcgen05_cluster_n") == 2
        ]
        assert m_paired_2x2
        assert all(
            config["tcgen05_persistence_model"] == "static_persistent"
            for config in m_paired_2x2
        )
