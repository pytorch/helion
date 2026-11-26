"""CPU checks of the 128-row CTA pairs (edge and one-wave families) on the search surface.

``tcgen05_cta_group="two"`` on ``block_m=128`` is the two-CTA tile cuBLAS-class
kernels run on ragged GEMMs below the 256-row pair's reach. The seeds and the
sampled pairs must survive search normalization (the cta_group validator and
the cluster_m=2 fixer used to project both onto the 256x256 pair), checked
here on fake SM100 targets (148 SMs; GB300's 152 for the one-wave family).
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
from helion._compiler.cute.tcgen05_config import repair
from helion._compiler.cute.tcgen05_config import seeds
from helion._testing import skipUnlessBackends
from helion._testing.cute_aux import rank_two_aux
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion._testing.cute_dense_tcgen05 import user_bound_mm
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.mark.parametrize("shape", ((1000, 1000, 1000), (777, 999, 555)))
def test_ragged_edge_family_keeps_its_128_row_pairs(shape) -> None:
    with _sm100_target():
        bound = _bind(*shape)
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert repair.plain_pair128_admitted(tcgen05)
        gen = ConfigGeneration(spec)
        # The seeds themselves, ahead of the small-grid default's promoted
        # copy of one of them.
        pair_seeds = [
            seed
            for seed in seeds.autotune_seed_configs(tcgen05)
            if seed.get("tcgen05_cta_group") == "two"
        ]
        assert [seed.config["block_sizes"][:3] for seed in pair_seeds] == [
            [128, 128, 64],
            [128, 128, 128],
        ]
        for seed in pair_seeds:
            # and they reach the autotuner's population
            assert seed in spec.compiler_seed_configs
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


def test_a_sampled_edge_pair_takes_the_proofs_ring_depth() -> None:
    # The edge family's projection clamps a sampled 128x128 pair's ring to
    # its allocation proof: 13 stages for bf16 1000^3 (the coarse per-stage
    # model stops at 12).
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1000, 1000, 1000))
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        assert tcgen05.cluster_m2_search_constraints.allow_edge_k_tail_family
        (seed,) = [
            seed
            for seed in seeds.autotune_seed_configs(tcgen05)
            if seed.get("tcgen05_cta_group") == "two"
            and seed.config["block_sizes"][:3] == [128, 128, 64]
        ][:1]
        gen = ConfigGeneration(spec)
        sample = {**copy.deepcopy(seed.config), "tcgen05_ab_stages": 16}
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        assert normalized["block_sizes"][:3] == [128, 128, 64]
        assert normalized["tcgen05_ab_stages"] == 13


def test_a_grid_of_256_row_pairs_keeps_the_pair_family_as_is() -> None:
    # 4096^3 fills the machine with 256-row pairs: the ragged 128-row pair
    # family is not admitted (its full-tile paired family has its own seeds).
    with _sm100_target():
        bound = _bind(4096, 4096, 4096)
        tcgen05 = bound.config_spec._cute_tcgen05_config
        assert not repair.plain_pair128_admitted(tcgen05)
        assert seeds._plain_edge_pair128_seed_configs(tcgen05, helion.Config()) == []


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
        assert not repair.plain_pair128_admitted(tcgen05)


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
        assert not repair.plain_pair128_admitted(fp8)
        assert fp8.cluster_m_search_choices == (1,)
        a = torch.empty(1000, 1000, dtype=torch.bfloat16)
        b = torch.empty(1000, 1000, dtype=torch.bfloat16)
        bias = torch.empty(1000, dtype=torch.bfloat16)
        aux = _bias_gemm.bind((a, b, bias)).config_spec
        assert aux.tcgen05.aux_kernel_detected
        assert not repair.plain_pair128_admitted(aux._cute_tcgen05_config)
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


def test_one_wave_family_runs_128_row_pairs_at_its_deepest_ring() -> None:
    # bf16 1000x1024x4096 (M ragged, N and K whole) is a one-wave shape: 16
    # pairs of 256x256 against GB300's 76 SM pairs. Its 256x64 tiles and the
    # 128x128 pair both put 128 CTAs on the machine, but each CTA of the
    # 256x64 tile streams 128 + 32 operand rows per K step against 64 + 64
    # for the pair (10.2 us against 8.3, cuBLAS 7.75). The pair is seeded at
    # the deepest ring its allocation proof admits (13 stages: the coarse
    # per-stage model stops at 12, measured 8.5 us), and a sampled pair keeps
    # its 128-row tile, a K tile it runs and that depth.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1000, 1024, 4096))
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        constraints = tcgen05.cluster_m2_search_constraints
        assert constraints is not None and constraints.one_wave_only
        assert not constraints.allow_edge_k_tail_family
        assert repair.plain_pair128_admitted(tcgen05)
        pair_seeds = [
            seed
            for seed in seeds.autotune_seed_configs(tcgen05)
            if seed.get("tcgen05_cta_group") == "two"
        ]
        assert [
            (seed.config["block_sizes"][:3], seed.config["tcgen05_ab_stages"])
            for seed in pair_seeds
        ] == [([128, 128, 64], 13)]
        gen = ConfigGeneration(spec)
        (seed,) = pair_seeds
        assert seed in spec.compiler_seed_configs
        _flat, normalized = gen.strict_config_pair(seed)
        assert normalized.config["block_sizes"][:3] == [128, 128, 64]
        assert normalized.config["tcgen05_ab_stages"] == 13
        sample = copy.deepcopy(seed.config)
        sample.update({"block_sizes": [128, 32, 32], "tcgen05_ab_stages": 16})
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        # (a 32-deep K tile the pair does not run: its 64-deep ring)
        assert normalized["tcgen05_cta_group"] == "two"
        assert normalized["block_sizes"][:3] == [128, 128, 64]
        assert normalized["tcgen05_ab_stages"] == 13
        # The autotune-off default promotes the seed
        # (``_small_grid_pair_default``); it ran the flat base default's
        # non-persistent 128x16x16 tile before (83 us against 8.3).
        default = spec.default_config().config
        assert default["tcgen05_cta_group"] == "two"
        assert default["block_sizes"][:3] == [128, 128, 64]
        assert default["tcgen05_ab_stages"] == 13
        # An N that 128 does not divide keeps the one-wave family's 256-row
        # tiles (the pair would clip a partial last N tile, which only the
        # edge family's pairs run).
        uneven = plain_mm_into._bind_isolated(bf16_mm_args(1000, 960, 4096))
        assert not repair.plain_pair128_admitted(
            uneven.config_spec._cute_tcgen05_config
        )
        # A sampled pair on a deeper K tile keeps its ring's K depth at the
        # pair's 64-deep tile (normalization sized it for the sample's own
        # tile: 128x128x256 asking for 10 stages arrives with 3).
        sample.update({"block_sizes": [128, 128, 256], "tcgen05_ab_stages": 10})
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        assert normalized["block_sizes"][:3] == [128, 128, 64]
        assert normalized["tcgen05_ab_stages"] == 12
        # A K block size the user bounds below the pair's K tiles
        # (hl.register_block_size(32, 32)) keeps the family off the pair.
        x, y, _out = bf16_mm_args(1000, 1024, 4096)
        bounded = user_bound_mm._bind_isolated((x, y, 32))
        assert not repair.plain_pair128_admitted(
            bounded.config_spec._cute_tcgen05_config
        )
        # A whole-tile M leaves the 128x128 pair to the paired-pipeline
        # search, which seeds it; the one-wave family does not take it.
        whole = plain_mm_into._bind_isolated(bf16_mm_args(1024, 1024, 4096))
        whole_tcgen05 = whole.config_spec._cute_tcgen05_config
        assert not repair.plain_pair128_admitted(whole_tcgen05)
        assert any(
            seed.get("tcgen05_cta_group") == "two"
            and seed.config["block_sizes"][:3] == [128, 128, 64]
            for seed in seeds.autotune_seed_configs(whole_tcgen05)
        )


def test_one_wave_family_runs_the_64_wide_pair_on_a_half_empty_grid() -> None:
    # bf16 1500x384x384 (M and N ragged for the 256-row pair, whole K): 36
    # tiles of 256x64 sit under the quarter-wave gate (38 on GB300's 152
    # SMs), so the work is also counted in 128x64 pair tiles (72). The
    # 128x128 pair would put 72 CTAs on the machine, the 128x64 pair 144:
    # the seed and the autotune-off default take the 64-wide pair (2.69 us
    # against the flat one-CTA winner's 3.12; cuBLAS runs 72 pairs of the
    # swapped 128x64 tile in 2.55).
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1500, 384, 384))
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        constraints = tcgen05.cluster_m2_search_constraints
        assert constraints is not None and constraints.one_wave_only
        assert repair.plain_pair128_widths(tcgen05) == (128, 64)
        pair_seeds = [
            seed
            for seed in seeds.autotune_seed_configs(tcgen05)
            if seed.get("tcgen05_cta_group") == "two"
        ]
        assert [
            (seed.config["block_sizes"][:3], seed.config["tcgen05_ab_stages"])
            for seed in pair_seeds
        ] == [([128, 64, 64], 6)]
        default = spec.default_config().config
        assert default["tcgen05_cta_group"] == "two"
        assert default["block_sizes"][:3] == [128, 64, 64]
        # A sampled pair keeps a width the family runs, and an N tile it
        # does not run takes the widest one.
        gen = ConfigGeneration(spec)
        (seed,) = pair_seeds
        normalized = gen.unflatten(gen.flatten(seed)).config
        assert normalized["block_sizes"][:3] == [128, 64, 64]
        sample = {**copy.deepcopy(seed.config), "block_sizes": [128, 32, 64]}
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        assert normalized["tcgen05_cta_group"] == "two"
        assert normalized["block_sizes"][:3] == [128, 128, 64]
        # The 64-wide pair only where its grid fits one wave: bf16
        # 1000x1024x4096 has 64 pairs of 128x128 and would have 128 of
        # 128x64.
        wide = plain_mm_into._bind_isolated(bf16_mm_args(1000, 1024, 4096))
        assert repair.plain_pair128_widths(wide.config_spec._cute_tcgen05_config) == (
            128,
        )


def test_a_one_tile_grid_keeps_its_ab_ring_within_the_k_loop() -> None:
    # bf16 1500x384x384 on 72 pairs of 128x64x128: every CTA runs one tile
    # of three k-blocks, so a sampled seven-stage ring (168 KB, one CTA per
    # SM, no dependent-launch overlap) is the three-stage kernel (72 KB)
    # with four stages never written.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1500, 384, 384))
        spec = bound.config_spec
        gen = ConfigGeneration(spec)
        (seed,) = [
            seed
            for seed in seeds.autotune_seed_configs(spec._cute_tcgen05_config)
            if seed.get("tcgen05_cta_group") == "two"
        ]
        sample = {
            **copy.deepcopy(seed.config),
            "block_sizes": [128, 64, 128],
            "tcgen05_ab_stages": 7,
        }
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        assert normalized["block_sizes"][:3] == [128, 64, 128]
        assert normalized["tcgen05_ab_stages"] == 3
        # An L2 swizzle that pads the scheduler's layout keeps the ring: at
        # swizzle 4 the six N tiles take eight slots (96 for 72 clusters),
        # so 12 clusters walk to a second tile. Swizzle 2 pads no slot.
        for swizzle, stages in ((4, 7), (2, 3)):
            sample["tcgen05_l2_swizzle_size"] = swizzle
            normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
            assert normalized["tcgen05_l2_swizzle_size"] == swizzle
            assert normalized["tcgen05_ab_stages"] == stages
        # A grid with more tiles than CTAs keeps the ring: bf16
        # 4096x4096x128 on 256x256x64 pairs (256 tiles over GB300's 76
        # pairs, two k-blocks each) keeps four stages; the producer runs
        # ahead into the next tile's K loop.
        many = plain_mm_into._bind_isolated(bf16_mm_args(4096, 4096, 128))
        many_gen = ConfigGeneration(many.config_spec)
        sample = {
            **copy.deepcopy(seed.config),
            "block_sizes": [256, 256, 64],
            "tcgen05_cta_group": "auto",
            "tcgen05_ab_stages": 4,
        }
        normalized = many_gen.unflatten(
            many_gen.flatten(helion.Config(**sample))
        ).config
        assert normalized["tcgen05_ab_stages"] == 4


def test_reserved_sms_keep_the_ring_of_a_grid_that_loops() -> None:
    # With 32 SMs reserved (``persistent_reserved_sms``) the persistent
    # launch keeps 60 of GB300's 76 SM pairs: bf16 1500x384x384's 72 pairs
    # of 128x64x128 then run a second tile on 12 of them, whose producer
    # runs ahead into it, so a sampled seven-stage ring is not clamped.
    kernel = helion.kernel(
        plain_mm_into.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        persistent_reserved_sms=32,
    )
    with gb300_target(), forbid_cuda_work():
        bound = kernel._bind_isolated(bf16_mm_args(1500, 384, 384))
        spec = bound.config_spec
        gen = ConfigGeneration(spec)
        (seed,) = [
            seed
            for seed in seeds.autotune_seed_configs(spec._cute_tcgen05_config)
            if seed.get("tcgen05_cta_group") == "two"
        ]
        sample = {
            **copy.deepcopy(seed.config),
            "block_sizes": [128, 64, 128],
            "tcgen05_ab_stages": 7,
        }
        normalized = gen.unflatten(gen.flatten(helion.Config(**sample))).config
        assert normalized["block_sizes"][:3] == [128, 64, 128]
        assert normalized["tcgen05_ab_stages"] == 7


def test_a_one_tile_epi_tma_default_keeps_the_256_row_pair() -> None:
    # bf16 2048x2048x192 + residual: 64 pairs of 256x256x64, one tile per
    # CTA, three k-blocks. The default stages the residual through the
    # epilogue-issued ring beside the AB ring the one-tile cap leaves (the
    # whole K loop, under the ring's usual four-stage floor) instead of
    # moving to 128 pairs of 256x128, two waves.
    with gb300_target(), forbid_cuda_work():
        lhs, rhs, residual = (
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in ((2048, 192), (192, 2048), (2048, 2048))
        )
        bound = rank_two_aux._bind_isolated((lhs, rhs, residual))
        default = bound.config_spec.default_config().config
    assert default["block_sizes"][:3] == [256, 256, 64]
    assert default["tcgen05_aux_load_mode"] == "epi_tma"
    assert default["tcgen05_ab_stages"] == 3


def test_a_normalized_ring_off_the_searched_range_still_encodes() -> None:
    # bf16 1000x512x2880 searches tcgen05_ab_stages over 1..12 plus the
    # 64-wide pair seed's 16. That seed with a sampled N tile of 128
    # normalizes to the 128x128 pair at its deepest ring, 13 stages: off
    # the range and not a seed. The pattern search walks its neighbors and
    # the surrogate encodes it (a full-effort autotune of this GEMM died on
    # "13 is not a compiler-seed integer value").
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(1000, 512, 2880))
        spec = bound.config_spec
        gen = ConfigGeneration(spec)
        (ab_index,), _ = gen._key_to_flat_indices["tcgen05_ab_stages"]
        fragment = gen.flat_spec[ab_index]
        assert (fragment.low, fragment.high) == (1, 12)
        (seed,) = [
            seed
            for seed in seeds.autotune_seed_configs(spec._cute_tcgen05_config)
            if seed.get("tcgen05_cta_group") == "two"
        ]
        assert seed.config["block_sizes"][:3] == [128, 64, 64]
        assert seed.config["tcgen05_ab_stages"] == 16
        sample = {**copy.deepcopy(seed.config), "block_sizes": [128, 128, 64]}
        flat, config = gen.canonicalize_flat(gen.flatten(helion.Config(**sample)))
        assert config.config["tcgen05_cta_group"] == "two"
        assert config.config["tcgen05_ab_stages"] == 13
        gen.encode_config(flat)
        assert fragment.encode(13) == [13.0]
        assert fragment.pattern_neighbors(13) == [12, 11]
