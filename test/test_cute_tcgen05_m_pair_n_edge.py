"""The M-paired tile (``block_m=512``: two 256-row subtiles of a CTA pair that
share each B stage) over a partial last N tile, and at the aligned-N family's
widths (``tcgen05_tile_n``).

The role-local N-edge producer issues the clipped tile's B box whole, which
TMA zero-fills past N, beside both A subtiles, and each subtile's TMA store
clips the tile at N. The source checks render on the GB300 target; the CUDA
checks write NaN-filled outputs and compare against ``torch.mm`` over N tails
of 8 to 248 columns, both majors of each operand (an N-major B's rows need a
16-byte stride, so its N tails are multiples of 8), whole K and a K tail, on
256-wide tiles and on 224- and 160-wide ones.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.tcgen05_config import seeds
from helion._testing import DEVICE
from helion._testing import patch_target_facts
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_M_PAIR = {
    "block_sizes": [512, 256, 64],
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 2,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 3,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
}


def _bind(m: int, n: int, k: int, *, lhs_k_major: bool, rhs_k_major: bool):
    return plain_mm_into._bind_isolated(
        bf16_mm_args(m, n, k, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major)
    )


def test_the_m_paired_tile_issues_the_clipped_n_tile() -> None:
    # The weight gradient's layout: an M-major A and an N-major B; 1000 = 3 x
    # 256 + 232 columns.
    with gb300_target(), forbid_cuda_work():
        code = _bind(1024, 1000, 1024, lhs_k_major=False, rhs_k_major=False).to_code(
            helion.Config(**_M_PAIR)
        )
    # The producer issues every tile that starts inside N, the clipped last one
    # too (its B box zero-filled past N), both A subtiles beside B.
    loop = code[code.index("for tile_offset_2 in range(") :]
    loop = loop[: loop.index("producer_commit")]
    assert "tile_offset_1 < cutlass.Int32(1000)" in loop
    assert "cute.copy(tma_atom_a, tma_gA[None," in loop
    assert "cute.copy(tma_atom_a, tcgen05_msub_tma_gA[None," in loop
    assert "cute.copy(tma_atom_b, tma_gB[None," in loop
    # Each subtile stores its 256 x 256 tile through TMA, which clips it at N.
    assert (
        "cute.local_tile(tcgen05_tma_store_tensor, (256, 256), (tile_offset_0 "
        "// cutlass.Int32(256) + cutlass.Int32(_tcgen05_msub)"
    ) in code


def test_an_n_edge_seeds_the_m_paired_tile_on_one_pair_per_cluster() -> None:
    # 512x30528x32768 (the weight gradient of a 30528-row vocabulary
    # projection): 119.25 tiles of 256 columns.
    with gb300_target(), patch_target_facts(num_sm=152), forbid_cuda_work():
        bound = _bind(512, 30528, 32768, lhs_k_major=False, rhs_k_major=False)
        spec = bound.config_spec
        m_paired = []
        for seed in spec.autotune_seed_configs():
            fixed = helion.Config.from_dict(dict(seed.config))
            spec.normalize(fixed, _fix_invalid=True)
            if fixed.config["block_sizes"][0] == 512:
                m_paired.append(fixed.config)
        assert m_paired
        # A partial-N 2x2 cluster takes the 256-row pair's K-major operands
        # only; the M-paired tile's 2x2 seeds keep one pair per cluster here.
        assert {seed["tcgen05_cluster_n"] for seed in m_paired} == {1}
        bound.to_code(helion.Config.from_dict(m_paired[0]))


def test_the_m_paired_tile_runs_the_aligned_width_the_model_ranks_first() -> None:
    # 512x30528: 120 tiles of 512x256 fill 1.58 waves of 76 pairs, 137 of
    # 512x224 1.80; the search samples the clipped 224 (32-byte swizzle atoms
    # of the N-major B, which the 256-row pair is not offered) and seeds it on
    # its deepest ring, which renders two 256x224 MMAs a stage.
    with gb300_target(), patch_target_facts(num_sm=152), forbid_cuda_work():
        bound = _bind(512, 30528, 32768, lhs_k_major=False, rhs_k_major=False)
        spec = bound.config_spec
        assert 224 in spec._flat_fields()["tcgen05_tile_n"].choices
        state = spec._cute_tcgen05_config
        m_paired = [
            dict(seed.config)
            for seed in seeds.dense_aligned_n_search_seed_configs(state)
            if seed.config["block_sizes"][0] == 512
        ]
        assert [
            (seed["tcgen05_tile_n"], seed["tcgen05_ab_stages"]) for seed in m_paired
        ] == [(224, 4)]
        code = bound.to_code(helion.Config.from_dict(m_paired[0]))
    assert code.count("CtaGroup.TWO, (256, 224)") == 2
    assert "make_tma_umma_pipeline(num_stages=4" in code
    assert "tile_offset_1 < cutlass.Int32(30528)" in code


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


# (N tail, K tail, A K-major, B K-major): N tails from one 8-column vector to
# one short of a tile on each operand layout (an N-major B's rows take
# 16-byte multiples), and a K tail with an N tail on the weight gradient's
# layout and on the K-major one.
_N_TAILS = (
    (8, 0, False, False),
    (64, 0, False, False),
    (136, 0, True, False),
    (248, 0, False, True),
    (96, 40, False, False),
    (232, 24, True, True),
)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("n_tail", "k_tail", "lhs_k_major", "rhs_k_major"),
    _N_TAILS,
    ids=[
        f"ntail{n}_ktail{k}_A{'K' if a else 'M'}_B{'K' if b else 'N'}"
        for n, k, a, b in _N_TAILS
    ],
)
def test_n_edges_match_torch_mm(
    n_tail: int, k_tail: int, lhs_k_major: bool, rhs_k_major: bool
) -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    # Two 512-row tiles by 4 N tiles, the last one clipped: eight tiles on as
    # many pairs, each K loop ending in the K tail where there is one.
    a, b, out = bf16_mm_args(
        1024,
        3 * 256 + n_tail,
        6 * 64 + k_tail,
        lhs_k_major=lhs_k_major,
        rhs_k_major=rhs_k_major,
        device=DEVICE,
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(helion.Config(**_M_PAIR))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_an_n_edge_over_several_tiles_per_pair_matches_torch_mm() -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    # 512 x 40000 (157 N tiles, the last holding 64 columns) on 4-stage rings:
    # more tiles than pairs, so a pair's second tile may be the clipped one.
    a, b, out = bf16_mm_args(
        512, 40000, 1024, lhs_k_major=False, rhs_k_major=False, device=DEVICE
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(helion.Config(**{**_M_PAIR, "tcgen05_ab_stages": 4}))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=3e-2, rtol=2e-2)


# (N, tcgen05_tile_n, A K-major, B K-major): the aligned widths over a partial
# last tile of each (160: 75.5 tiles; 224: 53.6), on an N-major B and a
# K-major one; 1024 rows give the grid more tiles than pairs (a one-wave grid
# keeps the 256-row pairs: ``repair._m_pair_block_m_is_valid``).
_ALIGNED = (
    (12080, 160, False, False),
    (12000, 224, False, False),
    (12000, 224, True, True),
)


def _aligned_config(width: int) -> helion.Config:
    return helion.Config(
        **{**_M_PAIR, "tcgen05_ab_stages": 4, "tcgen05_tile_n": width},
        tcgen05_cta_group="two",
        tcgen05_persistence_model="static_persistent",
        tcgen05_strategy="role_local_monolithic",
    )


def test_aligned_widths_render_two_subtile_mmas() -> None:
    with gb300_target(), forbid_cuda_work():
        for n, width, lhs_k_major, rhs_k_major in _ALIGNED:
            code = _bind(
                1024, n, 1024, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major
            ).to_code(_aligned_config(width))
            assert code.count(f"CtaGroup.TWO, (256, {width})") == 2, (n, width)
            assert f"tile_offset_1 < cutlass.Int32({n})" in code


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("n", "width", "lhs_k_major", "rhs_k_major"),
    _ALIGNED,
    ids=[
        f"n{n}_w{w}_A{'K' if a else 'M'}_B{'K' if b else 'N'}"
        for n, w, a, b in _ALIGNED
    ],
)
def test_aligned_widths_match_torch_mm(
    n: int, width: int, lhs_k_major: bool, rhs_k_major: bool
) -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    a, b, out = bf16_mm_args(
        1024, n, 1024, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major, device=DEVICE
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(_aligned_config(width))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)


def test_an_explicit_aligned_width_is_valid_on_a_one_wave_grid() -> None:
    # 1024x4000 holds 32 tiles of 512x256, fewer than the 76 pairs: the search
    # neither samples nor seeds the M-paired widths there
    # (``facts._dense_aligned_n_m_paired``), but an explicit config is valid
    # wherever the M-paired envelope holds, whatever the GPU's SM count.
    with gb300_target(), patch_target_facts(num_sm=152), forbid_cuda_work():
        bound = _bind(1024, 4000, 1024, lhs_k_major=False, rhs_k_major=False)
        tile_n = bound.config_spec._flat_fields().get("tcgen05_tile_n")
        assert tile_n is None or 224 not in tile_n.choices
        code = bound.to_code(_aligned_config(224))
    assert code.count("CtaGroup.TWO, (256, 224)") == 2


# The schedules the search reaches beside the static one-pair tile above
# (``_M_PAIR``): the CLC-scheduled 512x256x128 seed over a partial last N
# tile, the 2x2 cluster at an aligned width over an even count of whole N
# tiles (K-major B), and a width's deepest ring as its seed takes it (160
# columns: five AB stages beside two C stages, the plan's price).
_SCHEDULES = {
    "clc_n_edge": (
        (1024, 3000, 1024, False, False),
        {
            **_M_PAIR,
            "block_sizes": [512, 256, 128],
            "tcgen05_ab_stages": 2,
            "l2_groupings": [4],
            "tcgen05_persistence_model": "clc_persistent",
            "tcgen05_strategy": "role_local_with_scheduler",
            "tcgen05_warp_spec_scheduler_warps": 1,
        },
    ),
    "aligned_2x2": (
        (1024, 4480, 1024, False, True),
        {
            **_M_PAIR,
            "tcgen05_tile_n": 224,
            "tcgen05_cluster_n": 2,
            "tcgen05_cta_group": "two",
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_strategy": "role_local_monolithic",
        },
    ),
    "w160_ab5": (
        (1024, 12080, 1024, False, False),
        {
            **_M_PAIR,
            "tcgen05_ab_stages": 5,
            "tcgen05_tile_n": 160,
            "tcgen05_cta_group": "two",
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_strategy": "role_local_monolithic",
        },
    ),
}


def test_the_searched_schedules_render() -> None:
    markers = {
        "clc_n_edge": "_clc_",
        "aligned_2x2": "CtaGroup.TWO, (256, 224)",
        "w160_ab5": "make_tma_umma_pipeline(num_stages=5",
    }
    with gb300_target(), forbid_cuda_work():
        for name, ((m, n, k, lhs_k_major, rhs_k_major), config) in _SCHEDULES.items():
            code = _bind(
                m, n, k, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major
            ).to_code(helion.Config(**config))
            assert "tcgen05_msub_tiled_mma" in code, name
            assert markers[name] in code, name


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name", list(_SCHEDULES))
def test_the_searched_schedules_match_torch_mm(name: str) -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    (m, n, k, lhs_k_major, rhs_k_major), config = _SCHEDULES[name]
    a, b, out = bf16_mm_args(
        m, n, k, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major, device=DEVICE
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(helion.Config(**config))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)
