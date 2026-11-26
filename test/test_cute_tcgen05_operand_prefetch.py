"""CPU checks of the operand L2 prefetches ahead of the dependent-launch wait.

A one-tile CTA whose A and B rows hold at most
``TCGEN05_OPERAND_L2_PREFETCH_MAX_LINES`` 128-byte lines (a one-CTA tile up to
``tcgen05_one_cta_l2_prefetch_max_lines``) prefetches all of them into L2 from
its A/B-load warps before ``griddepcontrol_wait``
(``cute_mma._build_operand_l2_prefetch``), and a flat one-CTA kernel's
epilogue warps prefetch its CTA's footprint there
(``cute_mma._build_flat_operand_l2_prefetch``). A grid of larger footprints
that leaves at least ``TCGEN05_PDL_L2_WARMUP_CTAS`` SMs idle, in its one wave
or in the previous call's last wave, has its first CTAs warm the first
k-blocks of every A and B row instead (``cute_mma._build_pdl_l2_warmup``).
A grid of 2x2 clusters of at most one CTA per SM takes both as the pair grid
does. Checked on the GB300 target (whose CPU model counts 148 SMs for the
grid).
"""

from __future__ import annotations

import functools
import types

import pytest
import torch

import helion
from helion._compiler.cute.cute_mma import _tcgen05_pdl_l2_warmup_kblocks
from helion._compiler.cute.tcgen05_constants import tcgen05_last_wave_idle_sms
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import gb300_target
from helion._testing.cute_dense_tcgen05 import plain_mm_into
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@functools.cache
def _default_code(m: int, n: int, k: int, **overrides: object) -> str:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(bf16_mm_args(m, n, k))
        values = {**bound.config_spec.default_config().config, **overrides}
        return bound.to_code(helion.Config(**values))


def test_a_small_one_tile_footprint_lands_in_l2_before_the_wait() -> None:
    # bf16 1500x384x384 on 72 pairs of 128x64: each CTA's 64 A rows and 32 B
    # rows hold 6 lines each (576 in all), prefetched by the TMA-load warp
    # ahead of its dependent-launch wait.
    code = _default_code(1500, 384, 384)
    pf = code.index("tcgen05_l2_pf_rank = ")
    wait = code.index("cute.arch.griddepcontrol_wait()", pf)
    assert (
        code.index(
            "tcgen05_ab_pipeline.producer_acquire(",
            code.index("tcgen05_tma_initial_full_tile = "),
        )
        < pf
    )
    block = code[pf:wait]
    assert "for tcgen05_l2_pf_i in cutlass.range_constexpr(2):" in block
    assert (
        "tcgen05_l2_pf_index = tile_offset_0 + tcgen05_l2_pf_rank * cutlass.Int32(64)"
        " + tcgen05_l2_pf_row"
    ) in block
    # (each row at the operand's own row stride: A's stride[0], B's
    # stride[1] for its K-major [K, N])
    assert (
        "cute.arch.prefetch((a.iterator + tcgen05_l2_pf_index * "
        "cutlass.Int64(a.layout.stride[0]) + cutlass.Int32(64) * "
        "cutlass.Int32(tcgen05_l2_pf_k)).llvm_ptr, cache_level='L2')"
    ) in block
    assert "for tcgen05_l2_pf_k_1 in cutlass.range_constexpr(6):" in block
    assert (
        "b.iterator + tcgen05_l2_pf_index_1 * cutlass.Int64(b.layout.stride[1])"
    ) in block
    # Two A/B-load warps share the rows (64 lanes).
    two = _default_code(1500, 384, 384, tcgen05_warp_spec_ab_load_warps=2)
    assert (
        "tcgen05_l2_pf_lane = cute.arch.lane_idx() + cutlass.Int32(32) * "
        "tcgen05_ab_load_rank"
    ) in two
    assert "for tcgen05_l2_pf_i in cutlass.range_constexpr(1):" in two


def test_a_prefetching_one_tile_kernel_keeps_the_prefix_join() -> None:
    # 1500x384x384 (an M edge keeps it off the one-shot form) prefetches its
    # operands ahead of the dependent-launch wait, so its TMA-load role stays
    # behind the prefix's cluster join; 1500x384x1536 (2304 lines per CTA,
    # past the prefetch window) hoists the role and joins at its first tile
    # (``tcgen05_hoist_persistent_tma_role``).
    guarded_join = (
        "if tcgen05_ab_producer_state.count == cutlass.Int32(0):\n"
        "                cutlass.pipeline.pipeline_init_wait("
    )
    prefetching = _default_code(1500, 384, 384)
    assert "tcgen05_l2_pf_" in prefetching
    assert guarded_join not in prefetching
    assert "\n    cutlass.pipeline.pipeline_init_wait(" in prefetching
    hoisted = _default_code(1500, 384, 1536)
    assert "tcgen05_l2_pf_" not in hoisted
    assert guarded_join in hoisted
    assert "\n    cutlass.pipeline.pipeline_init_wait(" not in hoisted


@pytest.mark.parametrize(
    "shape",
    (
        # 128 rows of 16 lines per CTA (2048): past the budget
        (1000, 1024, 1024),
        # a K loop longer than the prefetch window (1000x512x2880)
        (1000, 512, 2880),
        # 256 tiles of 256x256 on 76 SM pairs: CTAs walk several tiles
        (4096, 4096, 256),
    ),
)
def test_a_large_footprint_or_a_looping_grid_prefetches_no_tile(shape) -> None:
    # (1000x1024x1024 warms the grid's first k-blocks from idle SMs instead:
    # _build_pdl_l2_warmup)
    code = _default_code(*shape, tcgen05_flat_role_coordinates=False)
    assert "tcgen05_l2_pf_" not in code


def test_a_padded_operand_prefetches_its_own_rows() -> None:
    # A K-major operand whose rows are padded (a[:, :384] of a 1500x512
    # tensor, b likewise) keeps its row stride: with K in its place the
    # prefetch would pull other rows and the padding.
    a = torch.randn((1500, 512), dtype=torch.bfloat16)[:, :384]
    b = torch.randn((384, 512), dtype=torch.bfloat16)[:, :384].T
    out = torch.empty((1500, 384), dtype=torch.bfloat16)
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated((a, b, out))
        code = bound.to_code(bound.config_spec.default_config())
    assert "tcgen05_l2_pf_index * cutlass.Int64(a.layout.stride[0])" in code
    assert "tcgen05_l2_pf_index_1 * cutlass.Int64(b.layout.stride[1])" in code


# The 128x128 CTA pair, two AB stages: the full-effort pick of bf16
# 1000x768x128 with an M-major A.
PAIR_128x128 = {
    "block_sizes": [128, 128, 64],
    "tcgen05_cluster_m": 2,
    "tcgen05_cta_group": "two",
    "pid_type": "persistent_interleaved",
    "tcgen05_ab_stages": 2,
}


def _code(
    m: int, n: int, k: int, *, lhs_k_major: bool, rhs_k_major: bool, **overrides
) -> str:
    with gb300_target(), forbid_cuda_work():
        bound = plain_mm_into._bind_isolated(
            bf16_mm_args(m, n, k, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major)
        )
        values = {**bound.config_spec.default_config().config, **overrides}
        return bound.to_code(helion.Config(**values))


def test_a_one_cta_tile_prefetches_its_own_rows() -> None:
    # The persistent one-CTA pipeline takes the prefetch as the pair does,
    # each CTA its whole tile's rows (rank 0): bf16 1536x384x256 on 72 one-CTA
    # tiles of 128x64, whose 128 A rows and 64 B rows hold four lines each
    # (768 in all), between the TMA-load warp's first acquire and its
    # dependent-launch wait.
    code = _code(
        1536,
        384,
        256,
        lhs_k_major=True,
        rhs_k_major=True,
        block_sizes=[128, 64, 64],
        tcgen05_cluster_m=1,
        tcgen05_cta_group="auto",
        pid_type="persistent_interleaved",
        tcgen05_persistence_model="static_persistent",
    )
    assert "CtaGroup.ONE" in code
    pf = code.index("tcgen05_l2_pf_rank = cutlass.Int32(0)")
    wait = code.index("cute.arch.griddepcontrol_wait()", pf)
    assert (
        code.index(
            "tcgen05_ab_pipeline.producer_acquire(",
            code.index("tcgen05_tma_initial_full_tile = "),
        )
        < pf
    )
    block = code[pf:wait]
    assert block.count("cute.arch.prefetch(") == 2
    assert "for tcgen05_l2_pf_i in cutlass.range_constexpr(4):" in block
    assert (
        "tcgen05_l2_pf_index = tile_offset_0 + tcgen05_l2_pf_rank * cutlass.Int32(128)"
        " + tcgen05_l2_pf_row"
    ) in block
    assert "for tcgen05_l2_pf_k in cutlass.range_constexpr(4):" in block
    assert "for tcgen05_l2_pf_i_1 in cutlass.range_constexpr(2):" in block


def test_a_swizzle_that_pads_the_schedule_takes_no_one_tile_prefetch() -> None:
    # 1500x384x384 on 72 pairs of 128x64 at swizzle 4: the scheduler rounds
    # its 6 N clusters up to 8, so its 96 slots go round the 72 clusters and
    # 12 of them run two tiles.  None prefetches (a second tile would, after
    # the dependency resolved), and the TMA-load role joins at its first tile
    # as on a multi-tile grid.  Swizzle 2 divides the 6 clusters: one tile per
    # CTA, prefetched.
    guarded_join = "if tcgen05_ab_producer_state.count == cutlass.Int32(0):"
    padded = _default_code(1500, 384, 384, tcgen05_l2_swizzle_size=4)
    assert "swizzle_size=4" in padded
    assert "tcgen05_l2_pf_" not in padded
    assert guarded_join in padded
    unpadded = _default_code(1500, 384, 384, tcgen05_l2_swizzle_size=2)
    assert "swizzle_size=2" in unpadded
    assert "tcgen05_l2_pf_" in unpadded
    assert guarded_join not in unpadded
    # The padded dim is the scheduler's dim 1, whichever PID axis sits there:
    # 1536x400x256 on 84 one-CTA tiles of 128x64 (an N edge) whose loop order
    # puts M's 12 tiles on dim 1 is not padded at swizzle 4 (N's 7 would be).
    swapped = _code(
        1536,
        400,
        256,
        lhs_k_major=True,
        rhs_k_major=True,
        block_sizes=[128, 64, 64],
        loop_orders=[[1, 0]],
        tcgen05_cluster_m=1,
        tcgen05_cta_group="auto",
        pid_type="persistent_interleaved",
        tcgen05_persistence_model="static_persistent",
        cute_pdl=True,
        tcgen05_l2_swizzle_size=4,
    )
    m_on_dim1 = (
        "(1536 + _BLOCK_SIZE_0 - 1) // _BLOCK_SIZE_0, 1), (1, 1, 1), swizzle_size=4"
    )
    assert m_on_dim1 in swapped
    assert "tcgen05_l2_pf_" in swapped


def test_an_mn_major_footprint_is_prefetched_along_k() -> None:
    # bf16 1000x768x128, A M-major (x.T of a weight gradient) and B N-major,
    # on 48 pairs of 128x128: each CTA's 64 A rows of every K column
    # (2000-byte columns: two lines) and 64 B columns of every K row
    # (1536-byte rows: one line), 384 lines in all, between the TMA-load
    # warp's first acquire and its dependent-launch wait.
    code = _code(1000, 768, 128, lhs_k_major=False, rhs_k_major=False, **PAIR_128x128)
    pf = code.index("tcgen05_l2_pf_rank = ")
    wait = code.index("cute.arch.griddepcontrol_wait()", pf)
    assert (
        code.index(
            "tcgen05_ab_pipeline.producer_acquire(",
            code.index("tcgen05_tma_initial_full_tile = "),
        )
        < pf
    )
    block = code[pf:wait]
    assert block.count("cute.arch.prefetch(") == 3
    assert (
        "tcgen05_l2_pf_base = tile_offset_0 + tcgen05_l2_pf_rank * cutlass.Int32(64)"
    ) in block
    # The segment's last row, clamped to the last row of A.
    assert "tcgen05_l2_pf_point0 = tcgen05_l2_pf_base + cutlass.Int32(63)" in block
    assert "tcgen05_l2_pf_point0 = cutlass.Int32(999)" in block
    assert "for tcgen05_l2_pf_i in cutlass.range_constexpr(4):" in block
    # A K index steps the operand's own K stride.
    assert (
        "cute.arch.prefetch((a.iterator + tcgen05_l2_pf_k * "
        "cutlass.Int64(a.layout.stride[1]) + tcgen05_l2_pf_point0).llvm_ptr, "
        "cache_level='L2')"
    ) in block
    assert (
        "cute.arch.prefetch((b.iterator + tcgen05_l2_pf_k_1 * "
        "cutlass.Int64(b.layout.stride[0]) + tcgen05_l2_pf_base_1).llvm_ptr, "
        "cache_level='L2')"
    ) in block
    # Two A/B-load warps share the K indices (64 lanes).
    two = _code(
        1000,
        768,
        128,
        lhs_k_major=False,
        rhs_k_major=False,
        **PAIR_128x128,
        tcgen05_warp_spec_ab_load_warps=2,
    )
    assert (
        "tcgen05_l2_pf_lane = cute.arch.lane_idx() + cutlass.Int32(32) * "
        "tcgen05_ab_load_rank"
    ) in two
    assert "for tcgen05_l2_pf_i in cutlass.range_constexpr(2):" in two


CLUSTER_2X2_256x64 = {
    "block_sizes": [256, 64, 64],
    "tcgen05_cluster_m": 2,
    "tcgen05_cluster_n": 2,
    "pid_type": "persistent_interleaved",
    "tcgen05_ab_stages": 2,
    "tcgen05_flat_role_coordinates": False,
}


def test_a_one_wave_2x2_cluster_prefetches_its_pairs_tiles() -> None:
    # bf16 1000x768x128 (A M-major, B N-major) on 256x64 tiles in 2x2
    # clusters: 24 clusters, 96 CTAs, one tile each.  Each pair prefetches its
    # own tile's rows as a pair alone does: its CTA's 128 A rows of every K
    # column (rank in the pair = cluster rank & 1) and 32 B columns.
    code = _code(
        1000, 768, 128, lhs_k_major=False, rhs_k_major=False, **CLUSTER_2X2_256x64
    )
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in code
    pf = code.index("tcgen05_l2_pf_rank = ")
    block = code[pf : code.index("cute.arch.griddepcontrol_wait()", pf)]
    assert (
        "tcgen05_l2_pf_rank = cute.arch.make_warp_uniform("
        "cute.arch.block_idx_in_cluster()) & cutlass.Int32(1)"
    ) in block
    assert (
        "tcgen05_l2_pf_base = tile_offset_0 + tcgen05_l2_pf_rank * cutlass.Int32(128)"
    ) in block
    assert (
        "tcgen05_l2_pf_base_1 = tile_offset_1 + tcgen05_l2_pf_rank * cutlass.Int32(32)"
    ) in block
    # A grid of more 2x2 tiles than SMs runs its pairs over several tiles: no
    # one-tile prefetch.
    many = _code(
        8192, 4096, 128, lhs_k_major=False, rhs_k_major=False, **CLUSTER_2X2_256x64
    )
    assert "tcgen05_l2_pf_" not in many


def test_a_one_wave_2x2_cluster_warms_like_a_pair() -> None:
    # bf16 4096x1024x1024 (K-major) on 256x256 tiles in 2x2 clusters: 64
    # tiles, 128 CTAs, leave 20 of the target's SMs idle; the grid's first
    # eight CTAs (two clusters) warm the first k-blocks of every row, as the
    # 2x1 pair grid's do.
    tile = {
        "block_sizes": [256, 256, 64],
        "tcgen05_cluster_m": 2,
        "pid_type": "persistent_interleaved",
        "tcgen05_ab_stages": 6,
        "tcgen05_flat_role_coordinates": False,
    }
    two_by_two = _code(
        4096,
        1024,
        1024,
        lhs_k_major=True,
        rhs_k_major=True,
        **tile,
        tcgen05_cluster_n=2,
    )
    pair = _code(4096, 1024, 1024, lhs_k_major=True, rhs_k_major=True, **tile)
    for code in (two_by_two, pair):
        assert "if tcgen05_l2_warm_cta < cutlass.Int32(8):" in code
        assert "for tcgen05_l2_warm_line in cutlass.range_constexpr(6):" in code
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in two_by_two
    # The one-wave gate counts SMs, not co-resident clusters: 34 clusters
    # (bf16 4352x1024x1024, 136 CTAs) warm too, though a B200 co-schedules 33
    # four-CTA clusters. The surplus cluster runs as a later wave, and the
    # next call's first clusters start on the SMs it leaves idle.
    band = _code(
        4352,
        1024,
        1024,
        lhs_k_major=True,
        rhs_k_major=True,
        **tile,
        tcgen05_cluster_n=2,
    )
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in band
    assert "for tcgen05_l2_warm_line in cutlass.range_constexpr(6):" in band
    # Several waves (bf16 1000x9216x2304: 144 tiles, 288 CTAs): the pair grid
    # warms from the SMs its last wave leaves idle, the 2x2 grid, whose last
    # wave is not the pair grid's, does not.
    two_by_two = _code(
        1000,
        9216,
        2304,
        lhs_k_major=True,
        rhs_k_major=True,
        **tile,
        tcgen05_cluster_n=2,
    )
    pair = _code(1000, 9216, 2304, lhs_k_major=True, rhs_k_major=True, **tile)
    assert "tcgen05_l2_warm_line" in pair
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in two_by_two
    assert "tcgen05_l2_warm_" not in two_by_two


def test_line_aligned_mn_major_columns_prefetch_one_line_each() -> None:
    # 1024-row A: 2048-byte columns, so each CTA's 64 rows are one line.
    code = _code(1024, 768, 128, lhs_k_major=False, rhs_k_major=False, **PAIR_128x128)
    assert code.count("cute.arch.prefetch(") == 2
    assert "tcgen05_l2_pf_point0" not in code


@pytest.mark.parametrize(
    ("shape", "majors", "overrides"),
    (
        # 64 A rows and 32 B columns per CTA over 2048 K indices: past the
        # window
        ((768, 768, 2048), (False, False), {}),
        # 1024 tiles of 256x256 on 76 SM pairs: CTAs walk several tiles
        ((4096, 4096, 128), (False, False), {}),
        # a K-major A with an N-major B: mixed majors take no prefetch
        ((1000, 768, 128), (True, False), PAIR_128x128),
    ),
)
def test_an_mn_major_footprint_past_the_window_or_mixed_majors_prefetch_nothing(
    shape, majors, overrides
) -> None:
    lhs_k_major, rhs_k_major = majors
    code = _code(*shape, lhs_k_major=lhs_k_major, rhs_k_major=rhs_k_major, **overrides)
    assert "cute.arch.prefetch(" not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_lhs_mm_into(
    x: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """``x.T @ b`` with the transpose inside the kernel: A is M-major logically
    while the argument ``x`` [K, M] keeps its source-order strides."""
    k, m = x.shape
    n = b.shape[1]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[inner, rows].T, b[inner, cols])
        out[rows, cols] = acc.to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_k_major_mm_into(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> None:
    """``x.T @ y.T`` with both transposes inside the kernel (the form
    helion.inductor emits for a transposed product): A and B are K-major
    logically while ``x`` [K, M] and ``y`` [N, K] keep their source-order
    strides."""
    k, m = x.shape
    n = y.shape[0]
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, x[inner, rows].T, y[cols, inner].T)
        out[rows, cols] = acc.to(out.dtype)


def test_a_k_major_operand_transposed_inside_the_kernel_prefetches_its_rows() -> None:
    # bf16 1500x384x384 on the 128x64 pair (576 lines per CTA, as in
    # test_a_small_one_tile_footprint_lands_in_l2_before_the_wait), read as
    # x[inner, rows].T and y[cols, inner].T: A's rows are x's source dim 1
    # (stride 384) and B's rows y's source dim 0 (stride 384).  The source
    # order's stride[0] / stride[1] are 1: every row would prefetch the first
    # row's lines.
    x = torch.randn((1500, 384), dtype=torch.bfloat16).T
    y = torch.randn((384, 384), dtype=torch.bfloat16)
    out = torch.empty((1500, 384), dtype=torch.bfloat16)
    with gb300_target(), forbid_cuda_work():
        bound = _transposed_k_major_mm_into._bind_isolated((x, y, out))
        code = bound.to_code(bound.config_spec.default_config())
    assert "tcgen05_l2_pf_index * cutlass.Int64(x.layout.stride[1])" in code
    assert "tcgen05_l2_pf_index_1 * cutlass.Int64(y.layout.stride[0])" in code


def test_an_operand_transposed_inside_the_kernel_prefetches_nothing() -> None:
    # x [128, 1000] row-major read as x[inner, rows].T: the argument's layout
    # strides are in source order (``x.layout.stride[1]`` is 1, not 1000), so
    # the MN-major addressing would step K one element at a time.
    x = torch.randn((128, 1000), dtype=torch.bfloat16)
    b = torch.randn((128, 768), dtype=torch.bfloat16)
    out = torch.empty((1000, 768), dtype=torch.bfloat16)
    with gb300_target(), forbid_cuda_work():
        bound = _transposed_lhs_mm_into._bind_isolated((x, b, out))
        values = {**bound.config_spec.default_config().config, **PAIR_128x128}
        code = bound.to_code(helion.Config(**values))
    assert "cute.arch.prefetch(" not in code


def test_the_first_ctas_warm_every_rows_first_k_blocks() -> None:
    # bf16 1000x512x1280 on 64 pairs of 128x64 (17 stages): 128 CTAs leave
    # 20 of the target's 148 SMs idle, so the grid's first eight CTAs (in
    # launch order) sit there while the previous call runs. Their epilogue
    # warps prefetch the first 17 k-blocks of all 1000 A rows and 512 B rows
    # (one line per row and k-block; 189 rows per CTA), k-block 0 of every
    # row first, each row at its operand's own row stride.
    code = _default_code(1000, 512, 1280)
    warm = code.index(
        "    if tcgen05_epi_active:\n"
        "        tcgen05_l2_warm_cta = cutlass.Int32(cute.arch.block_idx()[0])"
    )
    # After the TMEM-allocation barrier (the MMA warp's ``wait_for_alloc``
    # never waits for the warming warps), ahead of the epilogue role and its
    # dependent-launch wait.
    assert code.index("tcgen05_tmem_allocator.wait_for_alloc()") < warm
    assert warm < code.index(
        "tcgen05_epi_acc_tmem_ptr = tcgen05_tmem_allocator.retrieve_ptr("
    )
    block = code[warm : code.index("tcgen05_exec_acc_tmem_ptr = ", warm)]
    assert "if tcgen05_l2_warm_cta < cutlass.Int32(8):" in block
    assert (
        "tcgen05_l2_warm_tid = cute.arch.lane_idx() + cutlass.Int32(32) * "
        "tcgen05_warp_idx"
    ) in block
    assert "for tcgen05_l2_warm_line in cutlass.range_constexpr(17):" in block
    # 189 rows over 128 threads: two passes, the second one partial
    assert "for tcgen05_l2_warm_j in cutlass.range_constexpr(2):" in block
    assert (
        "tcgen05_l2_warm_row = tcgen05_l2_warm_cta * cutlass.Int32(189) + "
        "tcgen05_l2_warm_local"
    ) in block
    assert (
        "if tcgen05_l2_warm_local < cutlass.Int32(189) and tcgen05_l2_warm_row "
        "< cutlass.Int32(1000):"
    ) in block
    assert (
        "cute.arch.prefetch((a.iterator + tcgen05_l2_warm_row * "
        "cutlass.Int64(a.layout.stride[0]) + cutlass.Int32(64) * "
        "cutlass.Int32(tcgen05_l2_warm_line)).llvm_ptr, cache_level='L2')"
    ) in block
    assert (
        "b.iterator + (tcgen05_l2_warm_row - cutlass.Int32(1000)) * "
        "cutlass.Int64(b.layout.stride[1])"
    ) in block
    # Each k-block only while the dependency has not released: the TMA-load
    # warp arrives on the release barrier right after its dependent-launch
    # wait (once, at its first tile), so a CTA that starts after the release
    # (no previous kernel still running) warms nothing.
    assert (
        "if not cute.arch.mbarrier_test_wait(tcgen05_l2_warm_release, "
        "cutlass.Int32(0)):"
    ) in block
    init = code.index("cute.arch.mbarrier_init(tcgen05_l2_warm_release, 1)")
    assert init < code.index("cutlass.pipeline.pipeline_init_arrive(")
    tma_wait = code.index("cute.arch.griddepcontrol_wait()")
    arrive = code.index(
        "if tcgen05_ab_producer_state.count == cutlass.Int32(0) and tcgen05_tma_warp:\n"
    )
    # (ahead of the first stage's TMA copies)
    assert (
        tma_wait
        < arrive
        < code.index("cute.arch.mbarrier_arrive(tcgen05_l2_warm_release)")
        < code.index("cute.copy(tma_atom_a")
    )
    assert code.count("cute.arch.mbarrier_arrive(tcgen05_l2_warm_release)") == 1
    # One warm-up per kernel, not per tile.
    assert code.count("tcgen05_l2_warm_cta = ") == 1
    # Two A/B-load warps share the ring producer: the TMA-load warp (one of
    # them) arrives once, right after the shared producer's wait.
    two = _default_code(1000, 512, 1280, tcgen05_warp_spec_ab_load_warps=2)
    assert "tcgen05_ab_load_rank = " in two
    assert two.count("cute.arch.mbarrier_arrive(tcgen05_l2_warm_release)") == 1
    two_wait = two.index("cute.arch.griddepcontrol_wait()")
    assert (
        two_wait
        < two.index(
            "if tcgen05_ab_producer_state.count == cutlass.Int32(0) and tcgen05_tma_warp:\n"
        )
        < two.index("cute.arch.mbarrier_arrive(tcgen05_l2_warm_release)")
    )
    assert "if not cute.arch.mbarrier_test_wait(tcgen05_l2_warm_release" in two


@pytest.mark.parametrize(
    "shape",
    (
        # 144 CTAs leave 4 of the 148 SMs idle: too few warming CTAs
        (1500, 384, 1536),
        # 20 idle SMs and rows of 45 lines over K: a long K loop on a
        # smaller grid
        (1000, 512, 2880),
        # each CTA prefetches its whole footprint itself
        (1500, 384, 384),
        # 4 CTAs: fewer than the warming CTAs, whose row blocks would leave
        # B unwarmed
        (128, 128, 1024),
    ),
)
def test_a_grid_without_idle_sms_or_a_prefetching_one_takes_no_warm_up(
    shape,
) -> None:
    assert "tcgen05_l2_warm_" not in _default_code(*shape)


def test_mn_major_operands_take_no_warm_up() -> None:
    # 1024x512x1024 on 64 pairs of 128x64 (20 idle SMs) warms with K-major
    # operands; with an M-major A and an N-major B (K rows of lines) it gains
    # nothing measurable on GB300 and takes none.
    pair = {
        "block_sizes": [128, 64, 64],
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "pid_type": "persistent_interleaved",
    }
    assert "tcgen05_l2_warm_" in _code(
        1024, 512, 1024, lhs_k_major=True, rhs_k_major=True, **pair
    )
    assert "tcgen05_l2_warm_" not in _code(
        1024, 512, 1024, lhs_k_major=False, rhs_k_major=False, **pair
    )
    # (a one-wave grid launches its tiles whatever ``num_sm_multiplier``)
    assert "tcgen05_l2_warm_" in _code(
        1024, 512, 1024, lhs_k_major=True, rhs_k_major=True, num_sm_multiplier=2, **pair
    )


def test_a_k_major_operand_transposed_inside_the_kernel_warms_its_source_rows() -> None:
    # 1000x512x1280 read as x[inner, rows].T and y[cols, inner].T: A's rows
    # are x's source dim 1 and B's rows y's source dim 0 (stride 1280); the
    # source order's stride[0] / stride[1] are 1.
    x = torch.randn((1000, 1280), dtype=torch.bfloat16).T
    y = torch.randn((512, 1280), dtype=torch.bfloat16)
    out = torch.empty((1000, 512), dtype=torch.bfloat16)
    with gb300_target(), forbid_cuda_work():
        bound = _transposed_k_major_mm_into._bind_isolated((x, y, out))
        code = bound.to_code(bound.config_spec.default_config())
    assert "tcgen05_l2_warm_row * cutlass.Int64(x.layout.stride[1])" in code
    assert (
        "(tcgen05_l2_warm_row - cutlass.Int32(1000)) * cutlass.Int64(y.layout.stride[0])"
    ) in code


def _warmup_plan(
    m: int,
    n: int,
    k: int,
    *,
    stages: int,
    k_major: bool = True,
    pair_m: int = 128,
    pair_n: int = 128,
    bk: int = 64,
):
    return types.SimpleNamespace(
        tcgen05_mma_bm=pair_m,
        tcgen05_mma_bn=pair_n,
        tcgen05_mma_a_k_major=k_major,
        tcgen05_mma_b_k_major=k_major,
        input_dtype=torch.bfloat16,
        bk=bk,
        tcgen05_ab_stage_count_value=stages,
        k_total_size=k,
        m_size=m,
        n_size=n,
    )


def test_the_warm_up_depth_follows_the_ring_the_k_extent_and_the_line_budget() -> None:
    def warm(plan, idle_sms: int) -> int:
        # a one-wave grid of 152 - idle_sms CTAs on GB300's 152 SMs
        return _tcgen05_pdl_l2_warmup_kblocks(
            plan, work_ctas=152 - idle_sms, num_sm=152, one_tile_per_cta=True
        )

    # the ring the TMA warps issue right after the release
    assert warm(_warmup_plan(1000, 1024, 2048, stages=14), idle_sms=8) == 14
    # no more k-blocks than K holds (a partial last one is left out)
    assert warm(_warmup_plan(1000, 1024, 600, stages=14), idle_sms=8) == 9
    # 1277 rows for the busiest of 8 CTAs (10216 in all), at most 8192
    # lines each: 6 of the ring's 8 k-blocks (144 tiles of 256x256 over 76
    # pairs leave 16 SMs idle in the second wave)
    assert (
        _tcgen05_pdl_l2_warmup_kblocks(
            _warmup_plan(1000, 9216, 2304, stages=8, pair_m=256, pair_n=256),
            work_ctas=288,
            num_sm=152,
            one_tile_per_cta=False,
        )
        == 6
    )
    # under four lines per row: none
    assert warm(_warmup_plan(8000, 9000, 2048, stages=8), idle_sms=8) == 0
    # the lines, not the k-blocks: three 128-deep k-blocks warm six lines per
    # row, as six 64-deep ones (1500x1536x384 on the 256x128 pair, 3 stages),
    # one warms two
    for stages, kblocks in ((3, 3), (1, 0)):
        plan = _warmup_plan(1500, 1536, 384, stages=stages, pair_m=256, bk=128)
        assert warm(plan, idle_sms=8) == kblocks
    # 16 idle SMs or more: a 128-row pair warms rows of at most 24 lines over
    # K, a 256-row pair any K
    assert warm(_warmup_plan(1000, 512, 1280, stages=17), idle_sms=24) == 17
    assert warm(_warmup_plan(1000, 512, 1600, stages=17), idle_sms=24) == 0
    assert warm(_warmup_plan(1000, 512, 1600, stages=17), idle_sms=8) == 17
    assert warm(_warmup_plan(1000, 512, 1600, stages=8, pair_m=256), idle_sms=24) == 8
    # fewer idle SMs than warming CTAs, or MN-major operands: none
    assert warm(_warmup_plan(1000, 1024, 2048, stages=14), idle_sms=7) == 0
    assert (
        warm(_warmup_plan(1000, 1024, 2048, stages=14, k_major=False), idle_sms=8) == 0
    )
    # fewer work CTAs than warming CTAs (each takes its own block of rows)
    assert (
        _tcgen05_pdl_l2_warmup_kblocks(
            _warmup_plan(256, 512, 2048, stages=14),
            work_ctas=6,
            num_sm=152,
            one_tile_per_cta=True,
        )
        == 0
    )
    # a one-tile CTA whose 64 + 32 rows of 6 lines (1500x384x384 on the 128x64
    # pair) the operand prefetch covers prefetches them itself
    assert warm(_warmup_plan(1500, 384, 384, stages=6, pair_n=64), idle_sms=8) == 0


def test_the_idle_sms_are_those_the_previous_calls_last_wave_leaves() -> None:
    idle = tcgen05_last_wave_idle_sms
    # one wave: the SMs the grid leaves idle
    assert idle(144, num_sm=152) == 8
    assert idle(152, num_sm=152) == 0
    # 144 tiles of the 256x256 pair (288 CTAs) over 76 pairs: 68 in the
    # second wave, 8 pairs (16 SMs) idle
    assert idle(288, num_sm=152) == 16
    # three waves, 72 pairs in the last
    assert idle(448, num_sm=152) == 8


def test_a_multi_wave_grid_warms_from_its_last_wave_idle_sms() -> None:
    # bf16 2048x3072x768 on the 256x256 pair: 96 tiles over the target's 74
    # pairs, 22 in the second wave, so the next call's first CTAs start on
    # the SMs the previous call's last wave leaves idle; its 8 warming CTAs
    # take 640 rows each over all six k-blocks of the ring.
    code = _code(
        2048,
        3072,
        768,
        lhs_k_major=True,
        rhs_k_major=True,
        block_sizes=[256, 256, 64],
        tcgen05_tile_n=0,
        tcgen05_cluster_m=2,
        tcgen05_cta_group="two",
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=6,
    )
    assert "_BLOCK_SIZE_1 = 256" in code
    assert "if tcgen05_l2_warm_cta < cutlass.Int32(8):" in code
    assert "for tcgen05_l2_warm_line in cutlass.range_constexpr(6):" in code
    assert (
        "tcgen05_l2_warm_row = tcgen05_l2_warm_cta * cutlass.Int32(640) + "
        "tcgen05_l2_warm_local"
    ) in code
    # (several tiles per CTA: the release barrier's arrive stays at the
    # first one)
    assert "advance_to_next_work()" in code
    assert code.count("cute.arch.mbarrier_arrive(tcgen05_l2_warm_release)") == 1


FLAT_64x8 = {
    "block_sizes": [64, 8, 128],
    "pid_type": "flat",
    "tcgen05_cluster_m": 1,
    "tcgen05_cta_group": "auto",
    "tcgen05_ab_stages": 8,
    "cute_pdl": True,
}


def test_a_flat_one_cta_kernel_prefetches_its_footprint_before_the_wait() -> None:
    # The transposed product of bf16 128x192x1000 with b stored [K, N] (in0 =
    # b.T [192, 1000] M-major, in1 = a.T [1000, 128] K-major) on flat
    # [64, 8, 128] one-CTA tiles: 48 CTAs, each reading one 128-byte line of
    # in0 per K index (1000) and 8 rows of in1 of 2000 bytes, 17 prefetches
    # each (every 64th element and the last: the rows start 0, 80, 32, ...
    # bytes into a line, and those more than 48 bytes in reach a 17th line):
    # 1136 lines, under the small grid's cap. Every warp of the flat kernel passes the dependent-launch
    # wait; its four epilogue warps prefetch ahead of it.
    code = _code(192, 128, 1000, lhs_k_major=False, rhs_k_major=True, **FLAT_64x8)
    pf = code.index(
        "if tcgen05_warp_idx < cutlass.Int32(4):\n        tcgen05_l2_pf_lane"
    )
    wait = code.index("cute.arch.griddepcontrol_wait()", pf)
    block = code[pf:wait]
    assert "for tcgen05_l2_pf_i in cutlass.range_constexpr(8):" in block
    assert (
        "cute.arch.prefetch((a.iterator + tcgen05_l2_pf_k * "
        "cutlass.Int64(a.layout.stride[1]) + tcgen05_l2_pf_point).llvm_ptr, "
        "cache_level='L2')"
    ) in block
    assert (
        block.count(
            "b.iterator + tcgen05_l2_pf_index * cutlass.Int64(b.layout.stride[1])"
        )
        == 17
    )
    assert "cutlass.Int32(999)).llvm_ptr" in block
    # the TMA loads follow the wait
    assert code.index("cute.copy(tma_atom_a", wait) > wait


@pytest.mark.parametrize(
    ("shape", "overrides"),
    (
        # 2000-deep K: 2000 + 8 x 32 lines, past the small grid's 2048
        ((192, 128, 2000), FLAT_64x8),
        # 128 CTAs (fewer than half of them start early) over 1136 lines:
        # past 768
        ((1024, 64, 1000), FLAT_64x8),
        # a flat split-K kernel's CTAs read a K slice each
        ((192, 128, 1000), {**FLAT_64x8, "tcgen05_split_k": 2}),
        # a 12-k-block K loop: under TCGEN05_FLAT_L2_PREFETCH_MIN_K_BLOCKS
        ((192, 128, 768), FLAT_64x8),
    ),
)
def test_a_flat_kernel_past_its_window_or_split_prefetches_nothing(
    shape, overrides
) -> None:
    code = _code(*shape, lhs_k_major=False, rhs_k_major=True, **overrides)
    assert "cute.arch.prefetch(" not in code


PERSISTENT_ONE_CTA = {
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 1,
    "tcgen05_cta_group": "auto",
    "tcgen05_persistence_model": "static_persistent",
    "cute_pdl": True,
}


def test_a_one_cta_tile_of_mixed_majors_prefetches_its_footprint() -> None:
    # The transposed product of bf16 128x768x768 with b stored [K, N]: in0 =
    # b.T [768, 768] M-major, in1 = a.T [768, 128] K-major, on persistent
    # one-CTA [64, 32, 64] tiles, 48 CTAs (each one tile): 768 lines of in0
    # (one per K index) and 32 rows of in1 of 12 lines, 1152 in all, under
    # the 2048 a grid of at most two thirds of the SMs takes. The TMA-load
    # warp prefetches them ahead of its dependent-launch wait.
    code = _code(
        768,
        128,
        768,
        lhs_k_major=False,
        rhs_k_major=True,
        block_sizes=[64, 32, 64],
        tcgen05_ab_stages=12,
        **PERSISTENT_ONE_CTA,
    )
    pf = code.index("tcgen05_l2_pf_lane = cute.arch.lane_idx()")
    wait = code.index("cute.arch.griddepcontrol_wait()", pf)
    block = code[pf:wait]
    assert "for tcgen05_l2_pf_i in cutlass.range_constexpr(24):" in block
    assert "a.iterator + tcgen05_l2_pf_k * cutlass.Int64(a.layout.stride[1])" in block
    assert (
        block.count(
            "b.iterator + tcgen05_l2_pf_index * cutlass.Int64(b.layout.stride[1])"
        )
        == 12
    )


@pytest.mark.parametrize(
    ("shape", "majors", "overrides", "prefetches"),
    (
        # K-major a and b, 96 CTAs of [64, 16, 128] (two thirds of the SMs
        # start early) over 960 lines: past 768, under 2048
        ((128, 768, 768), (True, True), {"block_sizes": [64, 16, 128]}, True),
        # 128 CTAs of [64, 16, 128] over 1280 lines (the transposed 256 x 512
        # x 1024 with b stored [K, N]): past the 768 of a fuller grid
        ((512, 256, 1024), (False, True), {"block_sizes": [64, 16, 128]}, False),
    ),
)
def test_a_one_cta_tile_prefetches_up_to_its_grids_window(
    shape, majors, overrides, prefetches
) -> None:
    code = _code(
        *shape,
        lhs_k_major=majors[0],
        rhs_k_major=majors[1],
        tcgen05_ab_stages=6,
        **PERSISTENT_ONE_CTA,
        **overrides,
    )
    assert ("cute.arch.prefetch(" in code) is prefetches
