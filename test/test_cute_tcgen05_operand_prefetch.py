"""CPU checks of the one-tile CTA's operand L2 prefetch ahead of its dependent-launch wait.

A one-tile CTA whose A and B rows hold at most
``TCGEN05_OPERAND_L2_PREFETCH_MAX_LINES`` 128-byte lines prefetches all of them
into L2 from its A/B-load warps before ``griddepcontrol_wait``
(``cute_mma._build_operand_l2_prefetch``). Checked on the GB300 target.
"""

from __future__ import annotations

import functools

import pytest
import torch

import helion
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
    assert "cute.arch.prefetch(" in prefetching
    assert guarded_join not in prefetching
    assert "\n    cutlass.pipeline.pipeline_init_wait(" in prefetching
    hoisted = _default_code(1500, 384, 1536)
    assert "cute.arch.prefetch(" not in hoisted
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
def test_a_large_footprint_or_a_looping_grid_prefetches_nothing(shape) -> None:
    code = _default_code(*shape, tcgen05_flat_role_coordinates=False)
    assert "cute.arch.prefetch(" not in code


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
