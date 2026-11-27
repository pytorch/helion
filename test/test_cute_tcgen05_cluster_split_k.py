"""The K-split cluster: ``tcgen05_split_k=2`` on a CtaGroup.TWO kernel with a
4-CTA cluster (``tcgen05_cluster_m=2, tcgen05_cluster_n=2``).

The cluster's two CTA pairs take one output tile's two K halves instead of two
N tiles: the scheduler counts two virtual N tiles per tile and the virtual pid
collapses them, each pair loads its own A (no multicast) and K half, and the
epilogue reduce-scatters the fp32 partials through shared memory
(``split_k_exchange.render_split_k_cluster_exchange``).  The source checks
render on the GB300 target; the CUDA check compares against ``torch.mm``.
"""

from __future__ import annotations

import functools

import pytest
import torch

import helion
from helion._compiler.cute.tcgen05_config import smem
from helion._compiler.cute.tcgen05_constants import tcgen05_cluster_split_k
from helion._compiler.cute.tcgen05_constants import (
    tcgen05_cluster_split_k_exchange_smem_bytes,
)
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


def _config(**overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128, 128, 64],
            "tcgen05_cta_group": "two",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 2,
            "tcgen05_split_k": 2,
            "pid_type": "persistent_blocked",
            "tcgen05_ab_stages": 12,
        }
        | overrides
    )


def _cr_args(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    # (A M-major, B N-major: the CR layout of the 1024x512x4096 target)
    return bf16_mm_args(m, n, k, lhs_k_major=False, rhs_k_major=False)


@functools.cache
def _code(m: int, n: int, k: int, **overrides: object) -> str:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(m, n, k))
        return bound.to_code(_config(**overrides))


def test_the_knob_names_the_cluster_split() -> None:
    plain = {"tcgen05_cluster_m": 2, "tcgen05_cluster_n": 2}
    assert tcgen05_cluster_split_k(plain | {"tcgen05_split_k": 2}) == 2
    # the cluster's lanes multicast A over two N tiles without the split
    assert tcgen05_cluster_split_k(plain) == 1
    # the flat grid's workspace split
    assert tcgen05_cluster_split_k({"tcgen05_split_k": 2}) == 1
    assert tcgen05_cluster_split_k(plain | {"tcgen05_split_k": 4}) == 1


def test_the_pairs_of_a_cluster_take_the_tile_s_k_halves() -> None:
    # bf16 1024x512x4096 on 128x128 pairs: 32 output tiles, one (2, 2, 1)
    # cluster each (128 CTAs).
    code = _code(1024, 512, 4096)
    # The scheduler walks two virtual N tiles per output tile; the virtual
    # pid drops the split lane, so both pairs decode the same tile.
    assert (
        "cutlass.utils.PersistentTileSchedulerParams(((1024 + _BLOCK_SIZE_0 - 1) "
        "// _BLOCK_SIZE_0 * 2, (512 + _BLOCK_SIZE_1 - 1) // _BLOCK_SIZE_1 * 2, 1), "
        "(2, 2, 1))"
    ) in code
    assert "(tcgen05_role_local_0_work_tile.tile_idx[1] >> cutlass.Int32(1))" in code
    # One cluster per tile: the grid is every tile's cluster, not the
    # persistent capacity.
    assert (
        "_launcher(_helion_plain_mm_into, (2, 2, ((1024 + _BLOCK_SIZE_0 - 1) // _BLOCK_SIZE_0 "
        "* 2 + 2 - 1) // 2 * (((512 + _BLOCK_SIZE_1 - 1) // _BLOCK_SIZE_1 * 2 + 2 - 1) "
        "// 2) * 1)"
    ) in code
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in code
    # Each pair's K half, from its N-lane cluster coordinate.
    for line in (
        "tcgen05_split_k_idx = tcgen05_block_in_cluster_coord_vmnk[2]",
        "tcgen05_split_k_tile_begin = tcgen05_split_k_idx * cutlass.Int32(32)",
        "tcgen05_split_k_begin = tcgen05_split_k_tile_begin * cutlass.Int32(64)",
        "tcgen05_split_k_end = tcgen05_split_k_begin + cutlass.Int32(2048)",
    ):
        assert line in code
    # the TMA-load and MMA roles' K loops
    assert (
        code.count(
            "for tile_offset_2 in range(tcgen05_split_k_begin, tcgen05_split_k_end, "
            "cutlass.Int32(_BLOCK_SIZE_2)):"
        )
        == 2
    )
    assert "tma_gA[None, tcgen05_split_k_tile_begin + cutlass.Int32(0)]" in code
    # A is loaded per pair like B (no multicast across the N lanes), and each
    # pair's MMA releases its own pair's ring stages alone.
    assert (
        "tma_a_cta_layout = cute.make_layout(cute.slice_(tcgen05_cluster_layout_vmnk, "
        "(0, None, 0, 0)).shape)"
    ) in code
    assert "tma_a_cta_coord = tcgen05_block_in_cluster_coord_vmnk[1]" in code
    assert (
        "tcgen05_a_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask("
        "tcgen05_cluster_layout_vmnk, tcgen05_block_in_cluster_coord_vmnk, mcast_mode=1)"
    ) in code
    assert (
        "tcgen05_ab_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup("
        "cutlass.pipeline.Agent.Thread, cutlass.Int32(1))"
    ) in code
    assert "mcast_mode_mn=(0, 1)" in code
    # The host builds A atoms without multicast.
    assert (
        "'kind': 'tcgen05_ab_tma', 'bm': 128, 'bn': 128, 'bk': 64, "
        "'cluster_m': 2, 'cluster_n': 1,"
    ) in code


def test_the_pairs_exchange_half_their_partials() -> None:
    code = _code(1024, 512, 4096)
    assert (
        "tcgen05_split_k_recv = cute.arch.alloc_smem(cutlass.Float32, 4096, alignment=128)"
        in code
    )
    assert "cute.arch.mbarrier_init(tcgen05_split_k_recv_bar, 1)" in code
    # Armed before the accumulator wait; the partner's subtile goes out with
    # st.async before this pair's own subtile is loaded; the add waits for
    # the partner's bytes; each pair stores its own half of the subtiles.
    order = [
        code.index(
            "cute.arch.mbarrier_arrive_and_expect_tx(tcgen05_split_k_recv_bar, "
            "1 * skc_frag * 128 * 4)"
        ),
        code.index(
            "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)",
            code.index("skc_frag = cute.size(tcgen05_tTR_rAcc)"),
        ),
        code.index("_helion_skc.st_async_f32x4(skc_remote_recv + "),
        code.index("skc_own[skc_s])"),
        code.index("tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)"),
        code.index("cute.arch.mbarrier_wait(tcgen05_split_k_recv_bar, 0)"),
        code.index("tcgen05_bSG_gD[None, skc_mine + cutlass.Int32(0)]"),
    ]
    assert order == sorted(order)
    assert (
        "skc_peer = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) ^ cutlass.Int32(2)"
        in code
    )
    assert (
        "import helion._compiler.cute._split_k_cluster_runtime as _helion_skc" in code
    )


def test_the_grouped_decode_walks_whole_tiles() -> None:
    # The cluster box is one output tile (its N lanes are K splits), so the
    # L2-grouped decode runs the plain grouping over the tile grid.
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(1024, 512, 4096))
        code = bound.to_code(_config(l2_groupings=[4]))
    assert "num_pid_in_group = 4 * num_pid_n" in code
    assert "cluster_lane_n" not in code


def test_an_odd_n_tile_count_splits() -> None:
    # 1024x384: three N tiles. A cluster_n=2 multicast pair needs an even
    # count; a K-split cluster covers one tile.
    assert "tcgen05_split_k_idx" in _code(1024, 384, 4096)
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(1024, 384, 4096))
        with pytest.raises(helion.exc.InvalidConfig, match="tiles is odd"):
            bound.to_code(_config(tcgen05_split_k=1))


@pytest.mark.parametrize(
    ("shape", "overrides"),
    [
        # an odd K-tile count (65 k-blocks) has no whole halves
        ((1024, 512, 4160), {}),
        # the scheduler swizzle, a second A/B-load warp and the flat grid
        ((1024, 512, 4096), {"tcgen05_l2_swizzle_size": 2}),
        ((1024, 512, 4096), {"tcgen05_warp_spec_ab_load_warps": 2}),
        ((1024, 512, 4096), {"pid_type": "flat"}),
    ],
)
def test_the_split_keeps_its_envelope(
    shape: tuple[int, int, int], overrides: dict[str, object]
) -> None:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(*shape))
        with pytest.raises(
            (helion.exc.InvalidConfig, helion.exc.BackendUnsupported),
            match="tcgen05_split_k|ab_load_warps|cta_group",
        ):
            bound.to_code(_config(**overrides))


def test_reserved_sms_refuse_the_split() -> None:
    # The split's grid runs one cluster per tile, so it cannot leave the SMs
    # persistent_reserved_sms keeps free: refused, and not seeded.
    kernel = helion.kernel(
        plain_mm_into.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        persistent_reserved_sms=8,
    )
    assert not _split_seeds(1024, 512, 4096, kernel=kernel)
    with gb300_target(), forbid_cuda_work():
        bound = kernel._bind_isolated(_cr_args(1024, 512, 4096))
        with pytest.raises(helion.exc.InvalidConfig, match="persistent_reserved_sms"):
            bound.to_code(_config())


def test_the_split_needs_the_tma_store_epilogue() -> None:
    # The pairs exchange their partials in the TMA-store epilogue only. The
    # register-direct store (searched on every plain tcgen05 kernel) and an
    # output no TensorMap addresses (rows 1026 bytes apart) store through
    # other bodies, where each pair would store its own half-K sum: the
    # config layer refuses the split, the search's repair drops it and no
    # seed offers it.
    direct = _config(tcgen05_c_store_mode="direct")
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(1024, 512, 4096))
        with pytest.raises(helion.exc.InvalidConfig, match="tcgen05_c_store_mode"):
            bound.to_code(direct)
        repaired = dict(direct.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert tcgen05_cluster_split_k(repaired) == 1
    a, b, _ = _cr_args(1024, 512, 4096)
    out = torch.empty((1024, 513), dtype=torch.bfloat16)[:, :512]
    assert not _split_seeds(1024, 512, 4096, args=(a, b, out))
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated((a, b, out))
        with pytest.raises(helion.exc.InvalidConfig, match="TensorMap"):
            bound.to_code(_config())


def test_the_exchange_buffer_costs_one_ab_stage() -> None:
    # The 16 KiB receive buffer of the 128x128 pair is one A/B stage of it.
    assert tcgen05_cluster_split_k_exchange_smem_bytes(bm=128, bn=128) == 16384 + 128
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(1024, 512, 4096))
        with bound.env:
            state = bound.config_spec._cute_tcgen05_config
            split = smem._paired_pipeline_stage_limit(
                state, dict(_config(tcgen05_ab_stages=2).config)
            )
            plain = smem._paired_pipeline_stage_limit(
                state, dict(_config(tcgen05_ab_stages=2, tcgen05_split_k=1).config)
            )
    assert split is not None and plain is not None
    assert split == plain - 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape",
    [
        # one wave of clusters; more tiles than co-resident 4-CTA clusters
        # (later hardware waves); an odd N tile count
        (1024, 512, 4096),
        (1536, 1536, 1024),
        (1024, 384, 2048),
    ],
)
def test_the_split_matches_torch_mm(shape: tuple[int, int, int]) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs an SM100-family GPU")
    m, n, k = shape
    a, b, out = bf16_mm_args(
        m, n, k, lhs_k_major=False, rhs_k_major=False, device=DEVICE
    )
    bound = plain_mm_into.bind((a, b, out))
    bound.set_config(_config())
    out.fill_(float("nan"))
    bound(a, b, out)
    torch.testing.assert_close(out, torch.mm(a, b), atol=2e-2, rtol=2e-2)


def _split_seeds(
    m: int,
    n: int,
    k: int,
    kernel: helion.Kernel | None = None,
    args: tuple[torch.Tensor, ...] | None = None,
) -> list[dict[str, object]]:
    """The seeds the autotuner receives running the K-split cluster: the
    search's projection and canonical form of the compiler's seeds
    (``ConfigGeneration.seed_flat_config_pairs``), the flat grid's own
    split-K seeds (``tcgen05_split_k`` on one-CTA tiles) aside."""
    kernel = plain_mm_into if kernel is None else kernel
    args = _cr_args(m, n, k) if args is None else args
    with gb300_target(), forbid_cuda_work():
        bound = kernel._bind_isolated(args)
        with bound.env:
            gen = ConfigGeneration(bound.config_spec)
            seeds = [dict(config.config) for _, config in gen.seed_flat_config_pairs()]
    return [s for s in seeds if tcgen05_cluster_split_k(s) == 2]


def test_the_one_wave_seeds_split_a_deep_k() -> None:
    # 1024x512x4096: 32 tiles of 128x128 (128 CTAs in one wave of 4-CTA
    # clusters), 64 K tiles: the K-split cluster is seeded, its A/B ring one
    # stage short of the unsplit 128x128 pair's for the 16 KiB receive
    # buffer.
    (seed,) = _split_seeds(1024, 512, 4096)
    assert seed["block_sizes"] == [128, 128, 64]
    assert (seed["tcgen05_cluster_m"], seed["tcgen05_cluster_n"]) == (2, 2)
    assert seed["tcgen05_cta_group"] == "two"
    assert seed["tcgen05_ab_stages"] == 12
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(1024, 512, 4096))
        assert "tcgen05_split_k_idx" in bound.to_code(helion.Config(**seed))
    # 768x768x2048 runs 32 K tiles (the exchange costs more than the split
    # saves) and 2048x1024x4096's 128 tiles do not fit one wave: no split.
    for shape in ((768, 768, 2048), (2048, 1024, 4096)):
        assert not _split_seeds(*shape)
    # An odd N tile count (1024x384: three tiles) keeps the seed through the
    # search's projection: the split cluster covers one N tile, where a
    # cluster_n=2 multicast pair needs two.
    (odd,) = _split_seeds(1024, 384, 4096)
    assert (odd["tcgen05_cluster_n"], odd["tcgen05_split_k"]) == (2, 2)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _relu_mm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = torch.relu(acc * 0.5).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bias_mm_into(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, out: torch.Tensor
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        out[rows, cols] = (acc + bias[cols]).to(out.dtype)


def test_an_epilogue_over_the_accumulator_runs_on_the_sum() -> None:
    # relu(acc * 0.5): the chain runs on the summed fragment of each pair's
    # own subtiles, after the partner's partials land; the seed renders.
    (seed,) = _split_seeds(1024, 512, 4096, kernel=_relu_mm_into)
    with gb300_target(), forbid_cuda_work():
        bound = _relu_mm_into._bind_isolated(_cr_args(1024, 512, 4096))
        code = bound.to_code(helion.Config(**seed))
    wait = code.index("cute.arch.mbarrier_wait(tcgen05_split_k_recv_bar, 0)")
    add = code.index("+ skc_recv_view[0, skc_j, tcgen05_epi_tidx, skc_e]", wait)
    assert "cute.where" in code[add:] or "max(" in code[add:]
    # An auxiliary tensor (a bias row) has per-subtile loads the exchange
    # has no place for: no seed, and the config layer refuses the split.
    a, b, out = _cr_args(1024, 512, 4096)
    bias = torch.empty(512, dtype=torch.bfloat16)
    assert not _split_seeds(
        1024, 512, 4096, kernel=_bias_mm_into, args=(a, b, bias, out)
    )
    with gb300_target(), forbid_cuda_work():
        bound = _bias_mm_into._bind_isolated((a, b, bias, out))
        with pytest.raises(helion.exc.InvalidConfig, match="accumulator alone"):
            bound.to_code(_config())


def test_the_grid_cap_is_inert_on_the_split() -> None:
    # One cluster per tile whatever tcgen05_persistent_grid_cap says: the
    # knob is refused on the split (and dropped from search samples).
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(_cr_args(1024, 512, 4096))
        with pytest.raises(
            helion.exc.InvalidConfig, match="tcgen05_persistent_grid_cap"
        ):
            bound.to_code(_config(tcgen05_persistent_grid_cap="resident_clusters"))


# torch's default reduced-precision flags (True, True), through helion.inductor:
# the exchange may carry 16-bit partials (``tcgen05_split_k_exchange``).
_plain_mm_into_output = helion.kernel(
    plain_mm_into.fn,
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    split_k_reduction="output",
)


def _args(m: int, n: int, k: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    # (``Tensor.to`` keeps the CR strides)
    return tuple(t.to(dtype) for t in _cr_args(m, n, k))


@pytest.mark.parametrize(
    ("dtype", "pack", "element"),
    [
        (torch.bfloat16, "pack_bf16x2", "cutlass.BFloat16"),
        (torch.float16, "pack_f16x2", "cutlass.Float16"),
    ],
)
def test_a_16_bit_exchange_sends_half_the_bytes(dtype, pack, element) -> None:
    # tcgen05_split_k_exchange="output": eight partials per 16-byte store,
    # rounded to the output dtype in pairs, into an 8 KiB receive buffer of
    # that dtype; the receiver adds them to its own fp32 partials.
    with gb300_target(), forbid_cuda_work():
        bound = _plain_mm_into_output._bind_isolated(_args(768, 768, 2048, dtype))
        code = bound.to_code(_config(tcgen05_split_k_exchange="output"))
    assert (
        f"tcgen05_split_k_recv = cute.arch.alloc_smem({element}, 4096, alignment=128)"
        in code
    )
    assert "assert cutlass.const_expr(skc_frag % 8 == 0)" in code
    assert (
        "cute.arch.mbarrier_arrive_and_expect_tx(tcgen05_split_k_recv_bar, "
        "1 * skc_frag * 128 * 2)" in code
    )
    assert (
        "_helion_skc.st_async_b32x4(skc_remote_recv + ((cutlass.Int32(skc_s) * "
        "cutlass.Int32(skc_frag // 8) + cutlass.Int32(skc_j)) * cutlass.Int32(128) "
        "+ tcgen05_epi_tidx) * cutlass.Int32(16), skc_remote_bar, "
        f"_helion_skc.{pack}(tcgen05_tTR_rAcc[8 * skc_j + 0], "
        "tcgen05_tTR_rAcc[8 * skc_j + 1]), " in code
    )
    assert (
        "tcgen05_tTR_rAcc[8 * skc_j + skc_e] = skc_own[0][8 * skc_j + skc_e] + "
        "cutlass.Float32(skc_recv_view[0, skc_j, tcgen05_epi_tidx, skc_e])" in code
    )
    assert "st_async_f32x4" not in code
    # The fp32 exchange of the same kernel is the default.
    with gb300_target(), forbid_cuda_work():
        fp32 = bound.to_code(_config())
    assert "_helion_skc.st_async_f32x4(" in fp32
    assert "st_async_b32x4" not in fp32


def test_a_16_bit_exchange_needs_split_k_reduction_output() -> None:
    # Offered and accepted only under split_k_reduction="output", on a
    # K-split cluster: torch's (False, True) flags (the default
    # split_k_reduction="accumulator") keep the partials fp32.
    with gb300_target(), forbid_cuda_work():
        accumulator = plain_mm_into._bind_isolated(_cr_args(768, 768, 2048))
        with pytest.raises(helion.exc.InvalidConfig, match="split_k_reduction"):
            accumulator.to_code(_config(tcgen05_split_k_exchange="output"))
        output = _plain_mm_into_output._bind_isolated(_cr_args(768, 768, 2048))
        unsplit = _config(tcgen05_split_k_exchange="output", tcgen05_split_k=1)
        with pytest.raises(helion.exc.InvalidConfig, match="applies to"):
            output.to_code(unsplit)
        # The search's repair drops it where it cannot apply.
        repaired = dict(unsplit.config)
        output.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired.get("tcgen05_split_k_exchange", "accumulator") == "accumulator"


def test_the_16_bit_exchange_seed_splits_a_shallower_k() -> None:
    # 768x768x2048 (32 K tiles) is too shallow for the fp32 exchange's seed;
    # under split_k_reduction="output" its 16-bit twin is seeded (8 KiB of
    # receive buffer leaves the A/B ring at the fp32 split's 12 stages), and
    # 1024x512x4096 takes both.
    assert (
        tcgen05_cluster_split_k_exchange_smem_bytes(bm=128, bn=128, itemsize=2)
        == 8192 + 128
    )
    kernel = _plain_mm_into_output
    (seed,) = _split_seeds(768, 768, 2048, kernel=kernel)
    assert seed["tcgen05_split_k_exchange"] == "output"
    assert seed["tcgen05_ab_stages"] == 12
    exchanges = sorted(
        s["tcgen05_split_k_exchange"]
        for s in _split_seeds(1024, 512, 4096, kernel=kernel)
    )
    assert exchanges == ["accumulator", "output"]
    with gb300_target(), forbid_cuda_work():
        bound = kernel._bind_isolated(_cr_args(768, 768, 2048))
        assert "st_async_b32x4" in bound.to_code(helion.Config(**seed))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(768, 768, 2048), (1024, 384, 2048)])
def test_the_16_bit_exchange_matches_torch_mm(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    # The partner pair's half of the sum is rounded to the output dtype before
    # the fp32 add: an error of up to half a unit of that partial, which
    # cancellation leaves on small outputs (0.25 on an output of 7e-4 for
    # bf16 at K = 2048), so the error is judged against the output's scale:
    # a few units of the dtype's roundoff (bf16 2^-8, fp16 2^-11). A missing
    # or misplaced partial errs by a whole partial, ~0.2 of that scale.
    # (An odd N tile count on the second shape.)
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs an SM100-family GPU")
    m, n, k = shape
    a, b, out = (
        t.to(dtype)
        for t in bf16_mm_args(
            m, n, k, lhs_k_major=False, rhs_k_major=False, device=DEVICE
        )
    )
    bound = _plain_mm_into_output.bind((a, b, out))
    bound.set_config(_config(tcgen05_split_k_exchange="output"))
    out.fill_(float("nan"))
    bound(a, b, out)
    assert not out.isnan().any()
    expected = a.double() @ b.double()
    error = (out.double() - expected).abs().max() / expected.abs().max()
    assert error <= {torch.bfloat16: 2**-6, torch.float16: 2**-9}[dtype], error
