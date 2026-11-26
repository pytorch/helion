"""CPU checks of the two-warp A/B TMA producer (``tcgen05_warp_spec_ab_load_warps``).

One warp issues a CTA pair's A/B stages serially (``cute_mma``'s unrolled ring
fill, then a K loop issuing k-block ``k + stages``); with two, both A/B-load
warps walk one rolled K loop and each issues the k-blocks it owns
(``cute_mma._build_shared_ring_producer``). Checked on the GB300 target.
"""

from __future__ import annotations

import pytest

import helion
from helion._compiler.cute.tcgen05_config import repair
from helion._compiler.cute.tcgen05_config import surface
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

# bf16 1000x512x2880 on 64 pairs of 128x64 (the one-wave family's tile and
# cuBLAS's), sixteen 12 KB AB stages per CTA.
_PAIR = {
    "block_sizes": [128, 64, 64],
    "pid_type": "persistent_interleaved",
    "tcgen05_cta_group": "two",
    "tcgen05_cluster_m": 2,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 16,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
    "l2_groupings": [1],
}


def test_two_ab_load_warps_share_one_rolled_producer() -> None:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1000, 512, 2880))
        one = bound.to_code(helion.Config(**_PAIR))
        two = bound.to_code(helion.Config(**_PAIR, tcgen05_warp_spec_ab_load_warps=2))
    # One warp: the unrolled ring fill (sixteen stage blocks) and warp 5's
    # role, in the six-warp launch.
    assert "tcgen05_tma_initial_stage_1_full_tile" in one
    assert "tcgen05_ab_load_rank" not in one
    assert "block=(32, 6, 1)" in one
    # Two: warps 5 and 6 run the role, both walk the 45 k-blocks and each
    # issues every other slot of the ring's running count; no unrolled fill
    # and no lookahead K loop.
    assert (
        "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) >= cutlass.Int32(5) "
        "and cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(7):"
    ) in two
    assert "tcgen05_ab_load_rank = tcgen05_warp_idx - cutlass.Int32(5)" in two
    assert (
        "for tcgen05_ab_load_k_tile in cutlass.range(cutlass.Int32(0), "
        "cutlass.Int32(45), cutlass.Int32(1), unroll=1):"
    ) in two
    assert (
        "if tcgen05_ab_producer_state.count % cutlass.Int32(2) == tcgen05_ab_load_rank:"
    ) in two
    assert "tma_gA[None, tcgen05_ab_load_k_tile]" in two
    assert "tcgen05_tma_initial_stage_1_full_tile" not in two
    assert "tcgen05_tma_next_full_tile" not in two
    assert "block=(32, 7, 1)" in two
    # The ring's tail (its last stages' empty-barrier waits) stays on warp 5.
    assert two.count("tcgen05_ab_pipeline.producer_tail(") == 1
    assert "tcgen05_tma_warp = tcgen05_warp_idx == cutlass.Int32(5)" in two


def test_two_ab_load_warps_join_the_hoisted_init_rendezvous() -> None:
    # bf16 1024x1024x2048 runs one tile per CTA (64 pairs of 128x128), so
    # the TMA-load role is hoisted ahead of the pipeline-init rendezvous and
    # joins it itself: the prefix's wait leaves out both A/B-load warps.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1024, 1024, 2048))
        values = dict(bound.config_spec.default_config().config)
        values["tcgen05_warp_spec_ab_load_warps"] = 2
        values["tcgen05_flat_role_coordinates"] = False
        code = bound.to_code(helion.Config(**values))
    assert "while tcgen05_role_local_0_work_tile" not in code
    assert (
        "tcgen05_ab_load_warp = tcgen05_warp_idx >= cutlass.Int32(5) "
        "and tcgen05_warp_idx < cutlass.Int32(7)"
    ) in code
    assert (
        "if not tcgen05_ab_load_warp:\n        cutlass.pipeline.pipeline_init_wait("
    ) in code
    assert (
        "if tcgen05_ab_producer_state.count % cutlass.Int32(2) == "
        "tcgen05_ab_load_rank:" in code
    )


def test_plain_gemm_searches_one_or_two_ab_load_warps() -> None:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1000, 512, 2880))
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert surface.shared_ring_producer_search_enabled(tcgen05)
        fragment = surface.strategy_autotune_fragments(tcgen05)[
            "tcgen05_warp_spec_ab_load_warps"
        ]
        assert fragment.choices == (1, 2)
        # Search samples cover both counts and survive normalization.
        gen = ConfigGeneration(spec)
        counts = {
            gen.unflatten(gen.random_flat()).config.get(
                "tcgen05_warp_spec_ab_load_warps", 1
            )
            for _ in range(256)
        }
        assert counts == {1, 2}
        # A sample outside the persistent role-local producer runs one warp.
        flat = dict(_PAIR, pid_type="flat", tcgen05_warp_spec_ab_load_warps=2)
        repair._fix_ab_load_warps_search_config(tcgen05, flat)
        assert flat["tcgen05_warp_spec_ab_load_warps"] == 1
        paired = dict(_PAIR, tcgen05_warp_spec_ab_load_warps=2)
        repair._fix_ab_load_warps_search_config(tcgen05, paired)
        assert paired["tcgen05_warp_spec_ab_load_warps"] == 2
        _flat, normalized = gen.strict_config_pair(helion.Config(**paired))
        assert normalized.config["tcgen05_warp_spec_ab_load_warps"] == 2


def test_two_ab_load_warps_need_a_ring_as_deep_as_the_warps() -> None:
    # Each warp waits only on the ring slots it issues, every other one: one
    # stage would put two uses of a slot between one warp's waits, whose
    # parity test then passes early (``cute_mma``'s admission). Codegen
    # refuses the config; the search runs it with one warp.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1000, 512, 2880))
        shallow = dict(_PAIR, tcgen05_ab_stages=1, tcgen05_warp_spec_ab_load_warps=2)
        with pytest.raises(InvalidConfig, match="at least as many"):
            bound.to_code(helion.Config(**shallow))
        tcgen05 = bound.config_spec._cute_tcgen05_config
        repaired = dict(shallow)
        repair._fix_ab_load_warps_search_config(tcgen05, repaired)
        assert repaired["tcgen05_warp_spec_ab_load_warps"] == 1
        two = dict(_PAIR, tcgen05_ab_stages=2, tcgen05_warp_spec_ab_load_warps=2)
        repair._fix_ab_load_warps_search_config(tcgen05, two)
        assert two["tcgen05_warp_spec_ab_load_warps"] == 2


def test_m_paired_tiles_run_one_ab_load_warp() -> None:
    # A 512-row M-paired tile (bf16 4096^3, ``repair._m_pair_block_m_is_valid``)
    # runs two 256-row MMA subtiles per tile, which the two-warp producer's
    # admission leaves out: codegen refuses two A/B-load warps on it, and the
    # search runs such a sample with one. The 256-row pair keeps two.
    m_paired = {
        "block_sizes": [512, 256, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 4,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "l2_groupings": [1],
        "tcgen05_warp_spec_ab_load_warps": 2,
    }
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(4096, 4096, 4096))
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert repair._m_pair_block_m_is_valid(tcgen05)
        with pytest.raises(BackendUnsupported, match="ab_load_warps > 1"):
            bound.to_code(helion.Config(**m_paired))
        repaired = dict(m_paired)
        repair._fix_ab_load_warps_search_config(tcgen05, repaired)
        assert repaired["tcgen05_warp_spec_ab_load_warps"] == 1
        gen = ConfigGeneration(spec)
        _flat, normalized = gen.canonicalize_flat(
            gen.flatten(helion.Config(**m_paired))
        )
        assert normalized.config["block_sizes"][0] == 512
        assert normalized.config["tcgen05_warp_spec_ab_load_warps"] == 1
        pair = dict(m_paired, block_sizes=[256, 256, 64])
        repair._fix_ab_load_warps_search_config(tcgen05, pair)
        assert pair["tcgen05_warp_spec_ab_load_warps"] == 2
