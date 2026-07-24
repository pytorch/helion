"""Emission tests for the CuTe tcgen05 formula matmul seed heuristic.

These exercise the pure formula logic (regime classification -> collective -> depth-fill)
against the hill-climbed / pretuned answer keys, with a stubbed MatmulFact/ConfigSpec so
no GPU is required. B200 sm100 geometry: num_sm=148, per-CTA AB-SMEM budget 203776 bytes.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from helion._compiler.autotuner_heuristics import cute_matmul_formula as F

_NUM_SM = 148
_BUDGET = 232448 - 28 * 1024  # 203776 (B200 per-CTA AB-SMEM budget)

_FP8 = torch.float8_e4m3fn
_BF16 = torch.bfloat16
_FP16 = torch.float16


def _fact(m, k, n, dtype):
    return SimpleNamespace(
        static_m=m,
        static_k=k,
        static_n=n,
        lhs_dtype=dtype,
        rhs_dtype=dtype,
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=2,
    )


def _spec(indexing_length=3):
    return SimpleNamespace(indexing=SimpleNamespace(length=indexing_length))


def _seed(m, k, n, dtype, indexing_length=3, aux_rank=0):
    # aux_rank (0 transparent / 2 source-C residual) is computed from the live env by
    # _epilogue_aux_rank in the heuristic; the pure formula takes it as a parameter, so the
    # tests pass it directly (aux-rank detection itself is covered by the compile/run tests).
    return F._formula_seed(
        _fact(m, k, n, dtype), _spec(indexing_length), _NUM_SM, _BUDGET, aux_rank
    )


def _knobs(seed):
    return (
        tuple(seed["block_sizes"]),
        seed["tcgen05_cluster_m"],
        seed["tcgen05_ab_stages"],
        seed["pid_type"],
    )


def test_fp8_decode_key():
    # R1 M1 answer key: [64,64,128] cluster_m=1 ab=12 persistent_blocked.
    assert _knobs(_seed(64, 8192, 8192, _FP8)) == (
        (64, 64, 128),
        1,
        12,
        "persistent_blocked",
    )


def test_fp8_medium_m_key():
    # Climbed key (§1.2): [256,128,128] cluster_m=2 ab=8 persistent_interleaved.
    assert _knobs(_seed(512, 2048, 4096, _FP8)) == (
        (256, 128, 128),
        2,
        8,
        "persistent_interleaved",
    )


def test_bf16_compute_key():
    # R2 #11: bf16 compute fills the isobar to the deep bk64/ab6 pipeline on the DEFAULT path
    # (the bf16-deep-AB prerequisite removed the old ab<=3 cap; measured +1-4% over ab3).
    seed = _seed(4096, 8192, 8192, _BF16)
    assert _knobs(seed) == ((256, 256, 64), 2, 6, "persistent_interleaved")
    assert seed["l2_groupings"] == [1]  # many-wave


def test_fp8_compute_key():
    # R2 #3 / #4: fp8 compute square/flip-point -> [256,256,128] cluster_m=2 ab=6.
    assert _knobs(_seed(4096, 4096, 4096, _FP8)) == (
        (256, 256, 128),
        2,
        6,
        "persistent_interleaved",
    )
    assert _knobs(_seed(2048, 4096, 4096, _FP8)) == (
        (256, 256, 128),
        2,
        6,
        "persistent_interleaved",
    )


def test_fp8_decode_pretuned_rows():
    # Pretuned AOT table decode rows: bn=32/bk=256/ab=8 (medium K/N) and bn=64/bk=128/ab=12.
    assert _knobs(_seed(64, 2048, 4096, _FP8)) == (
        (64, 32, 256),
        1,
        8,
        "persistent_blocked",
    )
    assert _knobs(_seed(64, 4096, 4096, _FP8)) == (
        (64, 32, 256),
        1,
        8,
        "persistent_blocked",
    )
    assert _knobs(_seed(64, 5120, 5120, _FP8)) == (
        (64, 64, 128),
        1,
        12,
        "persistent_blocked",
    )


def test_bf16_decode_deep_ab_key():
    # R3 #8 key: cluster_m=1 bm=64 bn=32 bk=256 ab=4 (deep-AB is the lever; needs the
    # bf16-deep-AB prerequisite so 16-bit cluster_m=1 fills the isobar like fp8).
    assert _knobs(_seed(64, 4096, 4096, _BF16)) == (
        (64, 32, 256),
        1,
        4,
        "persistent_blocked",
    )


def test_isobar_invariant():
    # Every Bucket-A key lands at the ~196608-byte AB-SMEM isobar.
    for m, k, n, dt in [
        (64, 8192, 8192, _FP8),
        (512, 2048, 4096, _FP8),
        (4096, 8192, 8192, _BF16),
        (4096, 4096, 4096, _FP8),
        (64, 4096, 4096, _BF16),
    ]:
        seed = _seed(m, k, n, dt)
        bm, bn, bk = seed["block_sizes"]
        b = F.tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=F._itemsize(dt),
            ab_stages=seed["tcgen05_ab_stages"],
            cluster_m=seed["tcgen05_cluster_m"],
        )
        assert b == 196608, (m, k, n, dt, b)


def test_transparent_epilogue_keeps_deep_ab():
    # A transparent epilogue (aux_rank=0: unary act, rank-1 rowvec bias, OR an fp8/16-bit
    # rowwise [M,1]/[1,N] scale) must NOT clamp the pipeline: fp8 decode ab stays 12.
    # (Distinguishing an [M,1] scale from an [M,N] residual is done dtype-AGNOSTICALLY by
    # the graph detector in _epilogue_aux_rank — covered end-to-end by the compile/run test.)
    assert _seed(64, 8192, 8192, _FP8, aux_rank=0)["tcgen05_ab_stages"] == 12


def test_rank2_residual_caps_ab_at_2():
    # A rank-2 exact-shape [M,N] source-C residual (residual_add / bias_residual_gelu) caps
    # ab at 2 — the C2/C5 aux-TMA ceiling. Dtype-agnostic (the cap is physical, not fp8-vs-16bit).
    assert _seed(8192, 8192, 8192, _BF16, aux_rank=2)["tcgen05_ab_stages"] == 2
    assert _seed(4096, 4096, 4096, _FP8, aux_rank=2)["tcgen05_ab_stages"] == 2


def test_fp16_shares_16bit_path():
    # fp16 shares the bf16 16-bit path (compute cluster_m=2, deep bk64/ab6 default path).
    assert _knobs(_seed(4096, 8192, 8192, _FP16)) == (
        (256, 256, 64),
        2,
        6,
        "persistent_interleaved",
    )


def test_decode_regime_boundary():
    # M<=128 is decode (cluster_m=1); M>=256 that fills a wave is compute (cluster_m=2).
    assert _seed(128, 4096, 4096, _FP8)["tcgen05_cluster_m"] == 1
    assert _seed(4096, 4096, 4096, _FP8)["tcgen05_cluster_m"] == 2


def test_fp8_medium_m_small_grid_vs_rect():
    # fp8 medium-M: the small-grid [128,128,128]/ab12 wins when it fills ~one wave
    # (clusters <= num_sm//2 = 74); else the rectangular tile is kept. Measured head-to-head.
    # 512x2048x2048: 4*16=64 clusters <= 74 -> small-grid
    assert _knobs(_seed(512, 2048, 2048, _FP8))[:3] == ((128, 128, 128), 2, 12)
    # 256x4096x4096: 2*32=64 clusters <= 74 -> small-grid
    assert _knobs(_seed(256, 4096, 4096, _FP8))[:3] == ((128, 128, 128), 2, 12)
    # 512x8192x2048: 4*16=64 clusters <= 74 -> small-grid
    assert _knobs(_seed(512, 8192, 2048, _FP8))[:3] == ((128, 128, 128), 2, 12)
    # 512x2048x4096: 4*32=128 clusters > 74 -> rectangular [256,128,128]/ab8 (the climbed key)
    assert _knobs(_seed(512, 2048, 4096, _FP8))[:3] == ((256, 128, 128), 2, 8)


def test_regime_declines_outside_static_tile_envelope():
    # Non-pow2 decode M (96) and below-floor M (32) decline (return None tile).
    assert F._regime_tile(96, 4096, _NUM_SM, 1) is None
    assert F._regime_tile(32, 4096, _NUM_SM, 1) is None
    # M not a multiple of 256 for cluster_m=2 (e.g. 384) declines.
    assert F._regime_tile(384, 4096, _NUM_SM, 1) is None
    # Valid decode M and compute M return a tile.
    assert F._regime_tile(64, 4096, _NUM_SM, 1) is not None
    assert F._regime_tile(4096, 4096, _NUM_SM, 1) is not None


def test_compute_branch_n_guard_never_emits_bn_gt_n():
    # A tall+narrow shape (huge M, N<256) can clear the 256² occupancy bar on M-tiles alone, but
    # must NOT emit a bn=256 tile over a narrow-N output — the n>=256 guard routes it to the
    # rectangular medium-M tile, which caps bn<=N. (Regression for the review-found safety hole.)
    for m, n in [(16384, 64), (16384, 128), (32768, 192)]:
        cluster_m, bm, bn, _pid = F._regime_tile(m, n, _NUM_SM, 2)  # bf16
        assert bn <= n, (m, n, bn)
        assert (cluster_m, bm) == (2, 256)


def test_decode_fallback_picks_narrowest_bn():
    # In the fallback regime (tiny N: even bn=16 < 0.5 waves at M=64) the decode bn picker must
    # return the NARROWEST bn (max occupancy), not the old bn=64 mid-point. Measured: bn=16
    # ties-or-beats bn=64 on every probed tiny-N shape. Consistent with the loop's shrink premise.
    for n in (256, 512, 1024):
        assert _knobs(_seed(64, 4096, n, _FP8))[0][1] == 16, n
    # And the answer-key decode shapes are unchanged by the fallback edit.
    assert _knobs(_seed(64, 8192, 8192, _FP8))[0][1] == 64
    assert _knobs(_seed(64, 4096, 4096, _FP8))[0][1] == 32
