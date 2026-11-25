"""The M-paired tile (``block_m=512``: two 256-row subtiles of a CTA pair that
share each B stage) over a K tail.

The TMA producer issues the partial last K tile of both A subtiles and of B as
whole boxes, which TMA zero-fills past K, as it does for the 256-row pair. The
source checks render on the GB300 target; the CUDA checks write NaN-filled
outputs and compare against ``torch.mm`` over K tails of 1 to 63 elements and
both majors of each operand (a K-major operand's rows need a 16-byte stride,
so its K tails are multiples of 8).
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion.exc import BackendUnsupported

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_M_PAIR = {
    "block_sizes": [512, 256, 64],
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 2,
    "tcgen05_ab_stages": 3,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
}


def _config(cluster_n: int) -> helion.Config:
    return helion.Config(**_M_PAIR, tcgen05_cluster_n=cluster_n)


def _bind(m: int, n: int, k: int, *, lhs_k_major: bool, rhs_k_major: bool):
    return plain_mm_into._bind_isolated(
        bf16_mm_args(m, n, k, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major)
    )


def test_the_m_paired_tile_issues_the_partial_k_tile() -> None:
    k = 12 * 64 + 24
    with gb300_target(), forbid_cuda_work():
        code = _bind(2048, 4096, k, lhs_k_major=False, rhs_k_major=True).to_code(
            _config(cluster_n=2)
        )
    # Both A subtiles and B load each K tile, the partial last one too (its
    # box zero-filled past K): the K loop's guard reads "starts inside K".
    loop = code[code.index("for tile_offset_2 in range(") :]
    loop = loop[: loop.index("producer_commit")]
    assert f"(tile_offset_2 + cutlass.Int32(192) < cutlass.Int32({k}))" in loop
    assert "cute.copy(tma_atom_a, tma_gA[None," in loop
    assert "cute.copy(tma_atom_a, tcgen05_msub_tma_gA[None," in loop
    assert "cute.copy(tma_atom_b, tma_gB[None," in loop
    assert f"tile_offset_2 + cutlass.Int32(64) < cutlass.Int32({k})" in code


# The CLC-scheduled M-paired pair over a K tail (its 2x2 keeps the static
# schedule, ``test_the_m_paired_2x2_seed_keeps_its_static_projection``): the
# 512x256x128 seed's schedule, on a 4096x4096 grid whose 128 tiles outnumber
# the CTA pairs, so the scheduler hands most pairs a second tile.
_CLC_M_PAIR = {
    "block_sizes": [512, 256, 128],
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 2,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 2,
    "tcgen05_acc_stages": 2,
    "tcgen05_strategy": "role_local_with_scheduler",
    "tcgen05_persistence_model": "clc_persistent",
    "tcgen05_warp_spec_scheduler_warps": 1,
}
_CLC_K_TAILS = ((24, True), (7, False))


def test_the_clc_scheduled_pair_renders_over_a_k_tail() -> None:
    with gb300_target(), forbid_cuda_work():
        for tail, rhs_k_major in _CLC_K_TAILS:
            code = _bind(
                4096, 4096, 3 * 128 + tail, lhs_k_major=False, rhs_k_major=rhs_k_major
            ).to_code(helion.Config(**_CLC_M_PAIR))
            assert "_clc_next_work_tile" in code
            assert "tcgen05_msub_tma_gA" in code


def test_an_output_edge_keeps_256_row_pairs() -> None:
    # 1792 = 3.5 x 512 rows: the M-paired tile would run half a tile past M.
    with gb300_target(), forbid_cuda_work():
        bound = _bind(1792, 4096, 12 * 64 + 24, lhs_k_major=False, rhs_k_major=True)
        seeds = bound.config_spec.autotune_seed_configs()
        assert all(seed.config["block_sizes"][0] != 512 for seed in seeds)
        with pytest.raises(BackendUnsupported, match="block_m=512"):
            bound.to_code(_config(cluster_n=1))


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


# (K tail, A K-major, B K-major, cluster_n): every tail an M-major A and an
# N-major B take (any stride), the 16-byte-aligned tails of the K-major
# operands, both cluster shapes on each layout.
_K_TAILS = (
    (1, False, False, 2),
    (7, False, False, 1),
    (33, False, False, 2),
    (63, False, False, 1),
    (8, False, True, 2),
    (24, False, True, 1),
    (56, False, True, 2),
    (16, True, True, 1),
    (40, True, True, 2),
    (32, True, False, 2),
    (48, True, False, 1),
)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("tail", "lhs_k_major", "rhs_k_major", "cluster_n"),
    _K_TAILS,
    ids=[
        f"tail{t}_A{'K' if a else 'M'}_B{'K' if b else 'N'}_c2x{c}"
        for t, a, b, c in _K_TAILS
    ],
)
def test_k_tails_match_torch_mm(
    tail: int, lhs_k_major: bool, rhs_k_major: bool, cluster_n: int
) -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    k = 3 * 64 + tail
    a, b, out = bf16_mm_args(
        512, 512, k, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major, device=DEVICE
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(_config(cluster_n))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_k_tail_over_several_tiles_per_cta_matches_torch_mm() -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    # 4096x4096 on 512x256 M-paired 2x2 tiles: 128 tiles, more than one per
    # persistent cluster slot, each K loop ending in the 40-element tail.
    a, b, out = bf16_mm_args(
        4096, 4096, 12 * 64 + 40, lhs_k_major=False, rhs_k_major=True, device=DEVICE
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(_config(cluster_n=2))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=3e-2, rtol=2e-2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("tail", "rhs_k_major"), _CLC_K_TAILS, ids=["tail24_AM_BK", "tail7_AM_BN"]
)
def test_the_clc_scheduled_pair_over_a_k_tail_matches_torch_mm(
    tail: int, rhs_k_major: bool
) -> None:
    if not _sm100():
        pytest.skip("tcgen05 needs an SM100-family GPU")
    a, b, out = bf16_mm_args(
        4096,
        4096,
        3 * 128 + tail,
        lhs_k_major=False,
        rhs_k_major=rhs_k_major,
        device=DEVICE,
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(helion.Config(**_CLC_M_PAIR))
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)
