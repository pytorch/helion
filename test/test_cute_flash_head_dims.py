"""Symmetric head dims beyond 64 and 128 on the fused CuTe flash path (CPU).

Every multiple of 16 from 16 to 128 is a fused head dim: the SMEM tiles take
the swizzle the width admits (the CuTe DSL picks 128 B at multiples of 64,
64 B at multiples of 32, else 32 B), the TMA boxes and the MMA extents are
exact, and the two TMEM output accumulators take ``head_dim`` columns each.
The schedule's byte model, the resolver's caps, the search surface and the
rendered body all follow the width; the 64 and 128 surfaces and device
bodies are unchanged (the journal below pins the flex renders: their device
code is byte-identical to the commit before the widening, their host lines
follow the example's V-width generalization).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import math
import os
import re
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from test._cute_aux import _cpu_codegen

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import FLASH_ASYMMETRIC_HEAD_DIM_PAIRS
from helion._compiler.cute.attention_plan import flash_head_dim_supported
from helion._compiler.cute.attention_plan import flash_head_dims_supported
from helion._compiler.cute.attention_plan import flash_pair_head_dims_supported
from helion._compiler.cute.flash_schedule import FlashScheduleError
from helion._compiler.cute.flash_schedule import FlashScheduleSpec
from helion._compiler.cute.flash_schedule import build_fa4_schedule
from helion._compiler.cute.flash_schedule import fa4_header_bytes
from helion._compiler.cute.flash_schedule import max_fa4_kv_depth
from helion._compiler.cute.flash_schedule import verify_flash_schedule
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import import_path
import helion.language as hl
from helion.runtime.kernel import Kernel

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

NEW_HEAD_DIMS = (16, 32, 48, 80, 96, 112)
# The flex list-walk structure pinned by ``test_cute_flex_attention_gpu.py``.
FLEX_CONFIG: dict[str, object] = {
    "block_sizes": [128, 128],
    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
    cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
    cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/8",
    cute_flash.FLASH_S_LOAD_REP_KEY: 16,
    cute_flash.FLASH_P_STORE_REP_KEY: 16,
    cute_flash.FLASH_DISC_PIPE_KEY: 2,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    cute_flash.FLASH_EPI_TMA_KEY: True,
    cute_flash.FLASH_KV_SPLIT_KEY: 1,
}
SMEM_BUDGET = 232448


def _header_bytes(kv_depth: int, *, rings: int = 1, aux: int = 0) -> int:
    """The runtime SharedStorage header: 2308 fixed bytes plus two 8-byte
    mbarriers per ring stage, 16-aligned, the flex regions, then rounded to
    ``sQ``'s 1024-byte alignment (independent of the schedule module)."""
    barriers = -(-(2308 + 16 * rings * kv_depth) // 16) * 16
    return -(-(barriers + aux) // 1024) * 1024


def _expected_cap(head_dim: int, *, stage_output: bool, separate: bool = False) -> int:
    tile = 128 * head_dim * 2
    fixed = 2 * tile + (2 * tile if stage_output else 0)
    per_stage = 2 * tile if separate else tile
    rings = 2 if separate else 1
    depth = (SMEM_BUDGET - fixed - 3072) // per_stage
    while fixed + depth * per_stage + _header_bytes(depth, rings=rings) > SMEM_BUDGET:
        depth -= 1
    return depth


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def test_symmetric_head_dims_are_every_multiple_of_16_up_to_256() -> None:
    assert [d for d in range(300) if flash_head_dim_supported(d)] == list(
        range(16, 257, 16)
    )
    for head_dim in range(16, 257, 16):
        assert flash_head_dims_supported(head_dim, head_dim)
        # The FA4 pair body (two query tiles per CTA) stops at 128: two score
        # buffers beside two accumulators in the 512 TMEM columns.
        assert flash_pair_head_dims_supported(head_dim, head_dim) == (head_dim <= 128)
    assert not flash_head_dims_supported(8, 8)
    assert not flash_head_dims_supported(72, 72)
    assert not flash_head_dims_supported(272, 272)
    assert not flash_head_dims_supported(512, 512)
    # Asymmetric widths stay the explicit MLA pair.
    assert {(192, 128)} == FLASH_ASYMMETRIC_HEAD_DIM_PAIRS
    assert flash_head_dims_supported(192, 128)
    assert not flash_head_dims_supported(128, 64)
    assert not flash_head_dims_supported(96, 80)


# --------------------------------------------------------------------------
# the schedule's byte model and the runtime storage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", NEW_HEAD_DIMS)
def test_schedule_byte_model_follows_the_width(head_dim: int) -> None:
    cutlass = importlib.import_module("cutlass")
    flash_fa4_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_fa4_shared_storage

    tile = 128 * head_dim * 2
    for stage_output in (False, True):
        spec = FlashScheduleSpec(head_dim, 2, stage_output=stage_output, causal=True)
        schedule = verify_flash_schedule(build_fa4_schedule(spec)).schedule
        expected = 2 * tile + 2 * tile + (2 * tile if stage_output else 0) + 3072
        assert schedule.shared_memory_bytes == expected
        assert fa4_header_bytes(spec) == 3072
        assert schedule.tmem_columns == 256 + 2 * head_dim
        regions = {region.name: region for region in schedule.memory_regions}
        assert (regions["O_r0_q0"].offset, regions["O_r0_q0"].extent) == (
            256,
            head_dim,
        )
        assert (regions["O_r0_q1"].offset, regions["O_r0_q1"].extent) == (
            256 + head_dim,
            head_dim,
        )
        assert max_fa4_kv_depth(spec) == _expected_cap(
            head_dim, stage_output=stage_output
        )
        storage = flash_fa4_shared_storage(
            head_dim, 2, 2, 2, cutlass.BFloat16, stage_output, False, 1, False, 128, 1
        )
        assert storage.size_in_bytes() == schedule.shared_memory_bytes
    with pytest.raises(FlashScheduleError, match="multiple of 16"):
        build_fa4_schedule(FlashScheduleSpec(head_dim + 8, 2))


def test_schedule_rejects_widths_off_the_step() -> None:
    for head_dim in (8, 24, 72, 136, 144, 160, 256):
        with pytest.raises(FlashScheduleError, match="multiple of 16"):
            build_fa4_schedule(FlashScheduleSpec(head_dim, 2))


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------


def _resolve(
    head_dim: int,
    config: dict[str, object] | None = None,
    *,
    num_kv: int = 32,
    is_causal: bool = True,
    head_dim_v: int | None = None,
    has_list_walk: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> cute_flash.FlashAttentionConfig:
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.resolve_flash_config(
            head_dim,
            num_kv,
            config,
            head_dim_v=head_dim_v,
            dtype=dtype,
            num_bh=64,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
            has_list_walk=has_list_walk,
        )


def _fragments(
    head_dim: int,
    *,
    num_kv: int = 32,
    is_causal: bool = True,
    head_dim_v: int | None = None,
    has_list_walk: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            head_dim,
            num_kv,
            head_dim_v=head_dim_v,
            dtype=dtype,
            num_bh=64,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
            has_list_walk=has_list_walk,
        )


def _active(fragment: Any) -> tuple[object, ...]:
    choices = tuple(fragment.choices)
    search = fragment.search_choices
    return choices if search is None else tuple(search)


@pytest.mark.parametrize("head_dim", NEW_HEAD_DIMS)
def test_resolver_caps_the_ring_by_the_byte_model(head_dim: int) -> None:
    staged_cap = _expected_cap(head_dim, stage_output=True)
    direct_cap = _expected_cap(head_dim, stage_output=False)
    staged = _resolve(
        head_dim,
        {cute_flash.FLASH_KV_STAGE_KEY: 64, cute_flash.FLASH_EPI_TMA_KEY: True},
    )
    assert staged.epi_tma and staged.kv_stage == staged_cap
    direct = _resolve(
        head_dim,
        {
            cute_flash.FLASH_KV_STAGE_KEY: 64,
            cute_flash.FLASH_EPI_TMA_KEY: False,
            cute_flash.FLASH_EPI_STG_KEY: False,
        },
    )
    assert not direct.epi_tma and not direct.epi_stg
    assert direct.kv_stage == direct_cap


def test_deep_one_cta_ring_cap_bands() -> None:
    assert cute_flash._flash_deep_1cta_kv_stage_cap(64) == 4
    assert cute_flash._flash_deep_1cta_kv_stage_cap(128) == 2
    for head_dim in (16, 32, 48):
        assert cute_flash._flash_deep_1cta_kv_stage_cap(head_dim) == 4
    for head_dim in (80, 96, 112):
        assert cute_flash._flash_deep_1cta_kv_stage_cap(head_dim) == 2
    for head_dim in (8, 144, 192, 256):
        assert cute_flash._flash_deep_1cta_kv_stage_cap(head_dim) == 0


@pytest.mark.parametrize("head_dim", (16, 48, 80, 112))
def test_correction_tile_widths_that_do_not_divide_resolve_to_the_default(
    head_dim: int,
) -> None:
    resolved = _resolve(head_dim, {cute_flash.FLASH_CORR_TILE_SIZE_KEY: 32})
    assert resolved.corr_tile_size == 16
    for width in (8, 16):
        assert (
            _resolve(
                head_dim, {cute_flash.FLASH_CORR_TILE_SIZE_KEY: width}
            ).corr_tile_size
            == width
        )


def test_plain_exp2_packets_resolve_at_every_symmetric_width() -> None:
    for head_dim in NEW_HEAD_DIMS:
        for packet in ("4x1", "4x2", "8x1", "8x2"):
            resolved = _resolve(
                head_dim,
                {
                    cute_flash.FLASH_EXP2_PACKET_KEY: packet,
                    cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/8",
                },
            )
            assert resolved.exp2_packet == packet, (head_dim, packet)


# --------------------------------------------------------------------------
# the search surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("head_dim", NEW_HEAD_DIMS)
def test_surface_follows_the_width(head_dim: int, is_causal: bool) -> None:
    fragments = _fragments(head_dim, is_causal=is_causal)
    # The list walk / causal surfaces search the direct-output depths, the
    # dense surface too (its default output is the direct store below 128).
    direct_cap = _expected_cap(head_dim, stage_output=False)
    assert max(_active(fragments[cute_flash.FLASH_KV_STAGE_KEY])) == direct_cap
    assert set(_active(fragments[cute_flash.FLASH_EXP2_PACKET_KEY])) == {
        "1x1",
        "4x1",
        "4x2",
        "8x1",
        "8x2",
    }
    for key, widths in (
        (cute_flash.FLASH_CORR_TILE_SIZE_KEY, (8, 16, 32)),
        (cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY, (8, 16, 32)),
    ):
        assert set(_active(fragments[key])) == {
            width for width in widths if head_dim % width == 0
        }
    families = fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    assert "fa4_deep_1cta" in families.choices
    if not is_causal:
        assert "fa4_deep_1cta" in _active(families)


def test_surfaces_at_64_and_128_are_unchanged() -> None:
    causal64 = _fragments(64)
    assert _active(causal64[cute_flash.FLASH_KV_STAGE_KEY]) == tuple(range(2, 13))
    assert _active(causal64[cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY]) == (
        32,
        8,
        16,
        64,
    )
    assert _active(causal64[cute_flash.FLASH_CORR_TILE_SIZE_KEY]) == (16, 8, 32)
    assert set(_active(causal64[cute_flash.FLASH_EXP2_PACKET_KEY])) == {
        "1x1",
        "4x1",
        "4x2",
        "8x1",
        "8x2",
    }
    causal128 = _fragments(128)
    assert _active(causal128[cute_flash.FLASH_KV_STAGE_KEY]) == (2, 3, 4, 5)
    assert _active(causal128[cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY]) == (16, 8, 32)
    assert set(_active(causal128[cute_flash.FLASH_EXP2_PACKET_KEY])) == {
        "1x1",
        "causal_hd128_resident3_013_prefetch2_xu_early_acquire",
    }
    dense64 = _fragments(64, is_causal=False)
    assert _active(dense64[cute_flash.FLASH_KV_STAGE_KEY]) == (3, 2, *range(4, 25))
    dense128 = _fragments(128, is_causal=False)
    assert _active(dense128[cute_flash.FLASH_KV_STAGE_KEY]) == (3, 2, *range(4, 11))
    assert set(_active(dense128[cute_flash.FLASH_EXP2_PACKET_KEY])) == {
        "1x1",
        "4x1",
        "4x2",
        "8x1",
        "8x2",
    }


def test_staged_store_thread_layout_lines() -> None:
    lines = cute_flash._flash_o_store_thread_layout_lines
    # Widths whose packet count divides the thread count keep the whole-row
    # form (the historical text); the others take the common divisor.
    assert lines(64, 32, dim_name="d", elems_name="e").startswith("d = 64 // e\n")
    assert lines(128, 32, dim_name="d", elems_name="e").startswith("d = 128 // e\n")
    assert lines(32, 32, dim_name="d", elems_name="e").startswith("d = 32 // e\n")
    assert lines(16, 32, dim_name="d", elems_name="e").startswith("d = 16 // e\n")
    assert lines(80, 32, dim_name="d", elems_name="e").startswith("d = 2\n")
    assert lines(96, 32, dim_name="d", elems_name="e").startswith("d = 4\n")
    assert lines(112, 32, dim_name="d", elems_name="e").startswith("d = 2\n")
    assert lines(48, 32, dim_name="d", elems_name="e").startswith("d = 2\n")
    assert lines(96, 128, dim_name="d", elems_name="e").startswith("d = 4\n")
    assert lines(64, 128, dim_name="d", elems_name="e").startswith("d = 64 // e\n")
    for head_dim in range(16, 129, 16):
        for threads in (32, 128):
            text = lines(head_dim, threads, dim_name="d", elems_name="e")
            assert f"({threads} // d, d), order=(1, 0))" in text


# --------------------------------------------------------------------------
# the (192, 128) pair on the flex list walk
# --------------------------------------------------------------------------


def test_pair_list_walk_stages_its_output_in_the_dead_q_stages() -> None:
    """Two 128 x 128 staged output tiles do not fit beside two 192-wide Q
    tiles and a two-deep ring; the dense pair drops the staged output, the
    list walk (whose stores are row guarded through it) keeps it in the dead
    Q stages of the one-item-per-CTA body."""
    dense = _resolve(192, head_dim_v=128, is_causal=False)
    assert not dense.epi_tma and not dense.epi_stg and not dense.o_over_q
    for config in (None, {cute_flash.FLASH_PERSISTENT_KEY: True}):
        walk = _resolve(
            192, config, head_dim_v=128, has_list_walk=True, is_causal=False
        )
        assert walk.epi_tma or walk.epi_stg
        assert walk.o_over_q
        assert not walk.persistent
        assert walk.persistent_ctas_per_sm == 1
        assert walk.kv_stage == 2
    fragments = _fragments(192, head_dim_v=128, has_list_walk=True, is_causal=False)
    assert _active(fragments[cute_flash.FLASH_PERSISTENT_KEY]) == (False,)
    assert _active(fragments[cute_flash.FLASH_O_OVER_Q_KEY]) == (True,)
    # The symmetric list walk keeps its persistent search and separate ``sO``.
    walk128 = _resolve(128, has_list_walk=True, is_causal=False)
    assert walk128.persistent and not walk128.o_over_q
    fragments128 = _fragments(128, has_list_walk=True, is_causal=False)
    assert set(_active(fragments128[cute_flash.FLASH_PERSISTENT_KEY])) == {False, True}


# --------------------------------------------------------------------------
# renders
# --------------------------------------------------------------------------


def _identity(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return score


def _causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return q_idx >= kv_idx


def _render_flex(
    head_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    seq: int = 1024,
    config: dict[str, object] | None = None,
) -> str:
    """The flex example kernel at ``FLEX_CONFIG`` on CPU tensors and a real
    causal BlockMask."""
    module = import_path(EXAMPLES_DIR / "flex_attention.py")
    kernel = Kernel(
        module.helion_flex_attention_kernel.fn,
        settings=module.helion_flex_attention_kernel.settings.copy(backend="cute"),
    )
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, seq, head_dim, dtype=dtype) for _ in range(3))
    # A CPU render: the BlockMask tables live beside the CPU tensors.
    block_mask = create_block_mask(_causal, 2, 4, seq, seq, device=torch.device("cpu"))
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    assert full_num is not None and full_idx is not None
    args = (
        q,
        k,
        v,
        _identity,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        block_mask.mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        1.0 / math.sqrt(head_dim),
        False,
    )
    values = dict(FLEX_CONFIG)
    if config:
        values.update(config)
    with _cpu_codegen():
        bound = kernel.bind(args)
        return bound.to_code(bound._normalized_config_copy(helion.Config(**values)))


@pytest.mark.parametrize(
    ("head_dim", "dtype"),
    (
        pytest.param(32, torch.bfloat16, id="d32"),
        pytest.param(80, torch.bfloat16, id="d80"),
        pytest.param(96, torch.float16, id="d96"),
    ),
)
def test_flex_render_takes_the_exact_extents(head_dim: int, dtype: torch.dtype) -> None:
    code = _render_flex(head_dim, dtype=dtype)
    assert "'kind': 'helion_flash'" in code
    assert f"'head_dim': {head_dim}," in code
    # The Q/K and V tilers and the TMEM output partition carry the width.
    assert f"cute.select((128, 128, {head_dim}), mode=[0, 2])" in code
    assert f"cute.select((128, {head_dim}, 128), mode=[1, 2])" in code
    assert f"flash_pvt.partition_shape_C((128, {head_dim}))" in code
    assert f"_helion_flash_rt.flash_fa4_shared_storage({head_dim}, " in code
    assert f"rescale_o_tmem(tOtO0, flash_a0, flash_local_tidx, {head_dim}, 16)" in code
    assert "_flash_tma_o" in code
    # No padded width leaks into the body.
    padded = 128 if head_dim > 64 else 64
    if head_dim != padded:
        assert f"(128, 128, {padded})" not in code
        assert f"flash_pvt.partition_shape_C((128, {padded}))" not in code


@pytest.mark.parametrize(
    ("head_dim", "threads_per_row"),
    (
        pytest.param(80, 2, id="d80"),
        pytest.param(96, 4, id="d96"),
        pytest.param(32, None, id="d32-whole-rows"),
    ),
)
def test_flex_render_staged_store_covers_the_tile(
    head_dim: int, threads_per_row: int | None
) -> None:
    code = _render_flex(head_dim, config={cute_flash.FLASH_EPI_TMA_KEY: False})
    assert "_flash_tma_o" not in code
    assert "fa4_store_o_smem_to_gmem(" in code
    if threads_per_row is None:
        assert (
            f"flash_o_tO_shape_dim_1 = {head_dim} // flash_o_async_copy_elems" in code
        )
    else:
        assert f"flash_o_tO_shape_dim_1 = {threads_per_row}\n" in code
    assert (
        "cute.make_ordered_layout((32 // flash_o_tO_shape_dim_1, "
        "flash_o_tO_shape_dim_1), order=(1, 0))"
    ) in code


class TestRenderPins(TestCase):
    """The 64 and 128 renders of the flex list walk: the device body is
    byte-identical to the commit before the head-dim widening except for two
    later, intended changes (the journal was first written there; it was
    re-pinned when the example kernel took V's head dim for its accumulator
    and output, which changed host lines, and again when the list walk's
    PASS2 call was emitted once instead of once per tile kind, which
    collapsed four ``if flash_flex_needs_mask`` branches)
    only)."""

    def test_render_d64_unchanged(self) -> None:
        self.assertExpectedJournal(_render_flex(64))

    def test_render_d128_unchanged(self) -> None:
        self.assertExpectedJournal(_render_flex(128, dtype=torch.float16))

    def test_render_d256_flex_cooperative_pair(self) -> None:
        """The flex example at 256 with a causal BlockMask: the list walk on
        the cooperative pair (the pair's two table rows, per-stream cursors,
        the per-tile status, the mask hook per score chunk, the zero P tile)."""
        self.assertExpectedJournal(_render_flex(256))

    def test_render_d64_flex_cooperative_pair_on_request(self) -> None:
        """Within the pair body's value width the cooperative pair is legal
        on request for the table-backed walk (``_flash_ws_two_cta_list_walk``;
        not searched: it measured 26-52% behind the pair body): the flex
        example at 64 with ``ws_2cta`` renders the pair's list walk (the
        cluster decode, the per-tile status) instead of the fa4 pair body,
        which the default request keeps."""
        request = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
        pair = _render_flex(64, config=request)
        assert "cute.arch.block_idx_in_cluster()" in pair
        assert "'topology': 'ws_overlap'" in pair
        assert "flash_flex_status" in pair
        assert "_helion_flash_flex_item_words(" in pair
        body = _render_flex(64)
        assert "cute.arch.block_idx_in_cluster()" not in body
        assert "'topology': 'fa4'" in body

    def test_render_d192_single_tile_body(self) -> None:
        """The dense causal kernel at 192 on the single-tile body alone
        (``ws_overlap``, one CTA per 128-row tile): the flat grid, the
        one-deep K/V ring beside the staged output tile, the direct TMEM
        load repetition from gcd(192, 64)."""
        self.assertExpectedJournal(
            _render_dense_causal(
                192, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
            )
        )

    def test_render_d256_cooperative_pair(self) -> None:
        """The dense causal kernel at 256 on the single-tile body's
        cooperative pair (``ws_2cta``): the cluster decode, the leader-only
        MMA issue beside the follower's half loads, the pair's TMEM dealloc
        handshake and the base-tile addressed output and masks."""
        self.assertExpectedJournal(
            _render_dense_causal(
                256, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
            )
        )

    def test_render_d256_cooperative_pair_issue_knobs(self) -> None:
        """The d256 causal pair with its three issue-stream knobs on: the
        loads in warp 2 of both CTAs (the leader's warp 0 only waits and
        issues), the consumer loop split at the diagonal, the three-deep K
        ring."""
        self.assertExpectedJournal(
            _render_dense_causal(
                256,
                config={
                    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
                    cute_flash.FLASH_WS_LOADER_WARP_KEY: True,
                    cute_flash.FLASH_WS_MASK_PHASES_KEY: True,
                    cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY: 1,
                },
            )
        )

    def test_render_d256_cooperative_pair_dense(self) -> None:
        """The plain dense kernel at 256 on the cooperative pair: the
        head-major pair decode (no launch group), every KV tile live for both
        ranks (no diagonal skip), the direct output store."""
        self.assertExpectedJournal(
            _render_dense(
                256,
                causal=False,
                config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"},
            )
        )

    def test_render_d256_single_cta_dense(self) -> None:
        """The plain dense kernel at 256 on the single CTA's ``ws_overlap``
        body: the one-deep K and V rings beside the 128-row Q tile, the direct
        output store (the staged tile does not fit), one tile per CTA."""
        self.assertExpectedJournal(
            _render_dense(
                256,
                causal=False,
                config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
            )
        )


def test_render_pins_cover_the_families_the_widening_touched() -> None:
    """The pinned 64 render exercises the sites the widening rewrote: the
    staged-store layout (with the TMA store off) and the ring cap."""
    code = _render_flex(64, config={cute_flash.FLASH_EPI_TMA_KEY: False})
    assert "flash_o_tO_shape_dim_1 = 64 // flash_o_async_copy_elems" in code
    assert re.search(r"flash_fa4_shared_storage\(64, 3, 2, 2, ", code)


# --------------------------------------------------------------------------
# the dense surface
# --------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True)
def _dense_causal_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * math.log2(math.e)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _dense_attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * math.log2(math.e)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view(q_in.size())


def _render_dense(
    head_dim: int,
    *,
    causal: bool = True,
    seq: int = 2048,
    batch: int = 2,
    heads: int = 4,
    config: dict[str, object] | None = None,
) -> str:
    """The dense (causal or plain) kernel's render at its default config plus
    ``config`` on CPU tensors."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16) for _ in range(3)
    )
    kernel = _dense_causal_attention if causal else _dense_attention
    kernel.reset()
    with _cpu_codegen():
        bound = kernel.bind((q, k, v))
        values = dict(bound.config_spec.default_config().config)
        if config:
            values.update(config)
        return bound.to_code(bound._normalized_config_copy(helion.Config(**values)))


def _render_dense_causal(
    head_dim: int,
    *,
    seq: int = 2048,
    batch: int = 2,
    heads: int = 4,
    config: dict[str, object] | None = None,
) -> str:
    """The dense causal kernel's render at its default config plus
    ``config`` on CPU tensors."""
    return _render_dense(
        head_dim, causal=True, seq=seq, batch=batch, heads=heads, config=config
    )


@pytest.mark.parametrize("head_dim", (32, 80, 96, 112))
def test_dense_surface_opens_at_the_new_widths(head_dim: int) -> None:
    """The dense matcher reads the V/O width from V's host tensor when the
    loop graph's PV output carries it as an unbacked symbol (it does at 96
    and 80, not at 64 and 128): the flash search opens, the compiler
    default is the promoted fused seed and the pinned fa4 body renders."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(2, 4, 2048, head_dim, dtype=torch.bfloat16) for _ in range(3)
    )
    _dense_causal_attention.reset()
    with _cpu_codegen():
        bound = _dense_causal_attention.bind((q, k, v))
        spec = bound.config_spec
        assert spec.cute_flash_search_enabled
        assert spec._cute_flash_head_dim == head_dim
        assert spec._cute_flash_head_dim_v is None
        default = spec.default_config()
        assert default.config["block_sizes"] == [1, 128, 128]
        assert default.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"
        code = bound.to_code(default)
    assert "'kind': 'helion_flash'" in code
    assert f"'head_dim': {head_dim}," in code
    assert f"flash_pvt.partition_shape_C((128, {head_dim}))" in code


# --------------------------------------------------------------------------
# the byte model against the runtime storage at the depth caps
# --------------------------------------------------------------------------


def test_header_bytes_follow_the_ring_depth() -> None:
    """The flat 3072-byte header holds up to 47 aliased (23 separate) ring
    stages; the per-stage mbarrier pair pushes it to the next kilobyte past
    that (the narrow widths' caps reach it: hd16 direct 53, staged 51)."""
    for depth in (1, 2, 24, 47):
        assert fa4_header_bytes(FlashScheduleSpec(16, depth)) == 3072
    assert fa4_header_bytes(FlashScheduleSpec(16, 48)) == 4096
    assert fa4_header_bytes(FlashScheduleSpec(16, 23, separate_kv=True)) == 3072
    assert fa4_header_bytes(FlashScheduleSpec(16, 24, separate_kv=True)) == 4096
    assert fa4_header_bytes(FlashScheduleSpec(16, 56, separate_kv=True)) == 5120
    assert fa4_header_bytes(FlashScheduleSpec(64, 2, aux_smem_bytes=4096)) == 7168
    assert fa4_header_bytes(FlashScheduleSpec(64, 2, aux_smem_bytes=2048)) == 5120
    assert max_fa4_kv_depth(FlashScheduleSpec(16, 2, stage_output=False)) == 53
    assert max_fa4_kv_depth(FlashScheduleSpec(16, 2, stage_output=True)) == 51
    assert (
        max_fa4_kv_depth(FlashScheduleSpec(16, 2, stage_output=False, separate_kv=True))
        == 26
    )
    # The old flat header over-counted none of the widths from 32 up.
    assert max_fa4_kv_depth(FlashScheduleSpec(32, 2, stage_output=False)) == 26
    assert max_fa4_kv_depth(FlashScheduleSpec(32, 2, stage_output=True)) == 24


@pytest.mark.parametrize("separate", (False, True), ids=("aliased", "separate"))
@pytest.mark.parametrize("stage_output", (False, True), ids=("direct", "staged"))
@pytest.mark.parametrize("head_dim", tuple(range(16, 129, 16)))
def test_byte_model_matches_the_runtime_storage_at_the_cap(
    head_dim: int, stage_output: bool, separate: bool
) -> None:
    cutlass = importlib.import_module("cutlass")
    flash_fa4_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_fa4_shared_storage

    spec = FlashScheduleSpec(
        head_dim, 2, stage_output=stage_output, separate_kv=separate
    )
    cap = max_fa4_kv_depth(spec)
    assert cap == _expected_cap(head_dim, stage_output=stage_output, separate=separate)

    def storage_bytes(depth: int) -> int:
        return flash_fa4_shared_storage(
            head_dim,
            depth,
            2,
            2,
            cutlass.BFloat16,
            stage_output,
            False,
            1,
            separate,
            128,
            1,
        ).size_in_bytes()

    from helion._compiler.cute.flash_schedule import _shared_memory_bytes

    at_cap = _shared_memory_bytes(dataclasses.replace(spec, kv_depth=cap))
    assert storage_bytes(cap) == at_cap <= SMEM_BUDGET
    assert storage_bytes(cap + 1) > SMEM_BUDGET


def test_uneven_pair_ring_charges_the_flex_regions() -> None:
    """The pair's uneven three-slot ring fills the budget exactly (232448 B):
    any flex aux / cooperative staging has to decline it."""
    cutlass = importlib.import_module("cutlass")
    flash_fa4_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_fa4_shared_storage

    spec = FlashScheduleSpec(
        192, 3, stage_output=False, head_dim_v=128, uneven_kv_ring=True
    )
    from helion._compiler.cute.flash_schedule import _shared_memory_bytes

    assert _shared_memory_bytes(spec) == SMEM_BUDGET
    storage = flash_fa4_shared_storage(
        192,
        3,
        2,
        2,
        cutlass.BFloat16,
        False,
        False,
        1,
        False,
        128,
        1,
        head_dim_v=128,
        uneven_kv_ring=True,
    )
    assert storage.size_in_bytes() == SMEM_BUDGET
    supported = cute_flash._flash_uneven_kv_ring_supported
    common = {
        "head_dim_v": 128,
        "stage_output": False,
        "kv_tile_n": 128,
        "topology": "fa4",
        "separate_kv_rings": False,
        "mma_ptx": True,
    }
    assert supported(192, **common)
    # A region that fits in the alignment padding below ``sQ`` is free; the
    # staged tables of the knob's budgets are not.
    assert supported(192, aux_smem_bytes=16, **common)
    assert not supported(192, aux_smem_bytes=1024, **common)
    assert not supported(192, aux_smem_bytes=4096, **common)
    assert fa4_header_bytes(dataclasses.replace(spec, aux_smem_bytes=16)) == 3072
    assert fa4_header_bytes(dataclasses.replace(spec, aux_smem_bytes=4096)) == 7168
    assert (
        flash_fa4_shared_storage(
            192,
            3,
            2,
            2,
            cutlass.BFloat16,
            False,
            False,
            1,
            False,
            128,
            1,
            head_dim_v=128,
            uneven_kv_ring=True,
            flex_aux_bytes=4096,
        ).size_in_bytes()
        == SMEM_BUDGET + 4096
    )
    # The resolver: the list walk's table staging declines the uneven ring
    # (back to the two-deep ring) instead of exceeding the budget.
    with patch.dict(os.environ, {}, clear=True):
        kwargs = {
            "head_dim_v": 128,
            "dtype": torch.bfloat16,
            "num_bh": 64,
            "is_causal": False,
            "standard_dense_output": False,
            "standard_causal_output": False,
            "has_list_walk": True,
            "flex_aux_smem_candidates": (2048,),
        }
        fitting = cute_flash.resolve_flash_config(
            192,
            8,
            {
                cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
                cute_flash.FLASH_FLEX_AUX_SMEM_KEY: 0,
            },
            **kwargs,
        )
        declined = cute_flash.resolve_flash_config(
            192,
            8,
            {
                cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
                cute_flash.FLASH_FLEX_AUX_SMEM_KEY: 4096,
            },
            **kwargs,
        )
    assert fitting.uneven_kv_ring and fitting.kv_stage == 3
    assert fitting.flex_aux_smem == 0
    assert not declined.uneven_kv_ring and declined.kv_stage == 2
    assert declined.flex_aux_smem == 4096


def test_rescale_chunk_widths_that_do_not_divide_resolve_to_the_default() -> None:
    for head_dim in (16, 48, 80, 112):
        resolved = _resolve(head_dim, {cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY: 32})
        assert resolved.rescale_chunk_cols == 16
    for head_dim in (32, 64, 96, 128):
        resolved = _resolve(head_dim, {cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY: 32})
        assert resolved.rescale_chunk_cols == 32


def test_pair_list_walk_searches_the_staged_epilogue() -> None:
    """With the output staged in the dead Q stages, the pair's list walk
    searches the TMA store and the staged vector store like any list walk."""
    fragments = _fragments(192, head_dim_v=128, has_list_walk=True, is_causal=False)
    assert set(_active(fragments[cute_flash.FLASH_EPI_TMA_KEY])) == {False, True}
    assert set(_active(fragments[cute_flash.FLASH_EPI_STG_KEY])) == {False, True}
    assert set(_active(fragments[cute_flash.FLASH_EPI_STG_STORE_KEY])) == {
        "slice",
        "whole",
    }
    assert set(_active(fragments[cute_flash.FLASH_EPI_STG_GMEM_KEY])) == {
        "stage",
        "pair",
    }
    for epi_tma in (False, True):
        walk = _resolve(
            192,
            {cute_flash.FLASH_EPI_TMA_KEY: epi_tma},
            head_dim_v=128,
            has_list_walk=True,
            is_causal=False,
        )
        assert walk.o_over_q and not walk.persistent
        assert walk.epi_tma == epi_tma and walk.epi_stg == (not epi_tma)
        assert not walk.recompute_tile_coords


# --------------------------------------------------------------------------
# widths above 128: the single-query-tile body
# --------------------------------------------------------------------------


def test_single_tile_byte_model_matches_the_runtime_storage() -> None:
    cutlass = importlib.import_module("cutlass")
    flash_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_shared_storage
    for head_dim in (64, 128, 144, 192, 256):
        for kv_stage in (1, 2, 3, 4):
            for s_stage in (1, 2):
                for staged in (False, True):
                    for q_rows in (128, 64):
                        runtime = flash_shared_storage(
                            head_dim,
                            kv_stage,
                            s_stage,
                            cutlass.BFloat16,
                            o_stage=1 if staged else 0,
                            q_rows=q_rows,
                        ).size_in_bytes()
                        assert runtime == cute_flash._flash_ws_storage_bytes(
                            head_dim,
                            kv_stage,
                            q_tile_m=q_rows,
                            s_stage=s_stage,
                            epi_stg=staged,
                        )


def test_single_tile_ring_caps() -> None:
    """One 64 KB K and V stage beside the 64 KB Q tile at 256 (no staged
    output at 128 rows); the measured widths keep the deep-ring table."""
    cap = cute_flash._flash_ws_kv_stage_cap
    assert cap(256, q_tile_m=128, s_stage=2, epi_stg=False) == 1
    assert cap(256, q_tile_m=128, s_stage=2, epi_stg=True) == 0
    assert cap(256, q_tile_m=64, s_stage=2, epi_stg=True) == 1
    assert cap(192, q_tile_m=128, s_stage=2, epi_stg=False) == 1
    assert cap(192, q_tile_m=128, s_stage=2, epi_stg=True) == 1
    assert cap(144, q_tile_m=128, s_stage=2, epi_stg=True) == 2
    assert cap(128, q_tile_m=128, s_stage=2, epi_stg=False) == 3
    assert cute_flash._flash_deep_1cta_kv_stage_cap(128) == 2
    assert cute_flash._flash_deep_1cta_kv_stage_cap(144) == 0
    assert cute_flash._flash_deep_1cta_kv_stage_cap(256) == 0


@pytest.mark.parametrize(
    ("head_dim", "kv_stage", "staged_searched"),
    (
        pytest.param(144, 2, True, id="d144"),
        pytest.param(192, 1, True, id="d192"),
        pytest.param(256, 1, False, id="d256"),
    ),
)
def test_widths_above_128_resolve_to_the_single_tile_body(
    head_dim: int, kv_stage: int, staged_searched: bool
) -> None:
    """Two score buffers beside two accumulators do not fit the pair body's
    TMEM above 128; a pair-body request resolves to ws_overlap with the ring
    depth the byte model allows (a one-deep ring: K and V have their own
    rings), the staged output only where it leaves room for that ring. Left
    to the resolver, an even tile count takes the body's cooperative pair
    (``ws_2cta``); an odd one the single CTA."""
    for is_causal in (False, True):
        for requested in (
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
        ):
            resolved = _resolve(head_dim, dict(requested), is_causal=is_causal)
            assert resolved.topology == "ws_overlap"
            assert resolved.pipeline_family == "ws_overlap"
            assert resolved.kv_stage == kv_stage
            assert resolved.s_stage == 2
            assert resolved.q_tile_m == 128
            assert not resolved.epi_tma
            if head_dim == 256:
                assert not resolved.epi_stg
        default = _resolve(head_dim, None, is_causal=is_causal)
        assert default.pipeline_family == "ws_2cta"
        assert default.use_2cta_instrs
        odd_default = _resolve(head_dim, None, is_causal=is_causal, num_kv=9)
        assert odd_default.pipeline_family == "ws_overlap"
        assert odd_default.kv_stage == kv_stage
        fragments = _fragments(head_dim, is_causal=is_causal)
        # The single-tile body and its cooperative pair (an even tile count).
        assert set(_active(fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY])) == {
            "ws_overlap",
            "ws_2cta",
        }
        assert fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY].choices[0] == "ws_2cta"
        single = _fragments(head_dim, is_causal=is_causal, num_kv=9)
        assert set(_active(single[cute_flash.FLASH_KV_STAGE_KEY])) <= set(
            range(1, kv_stage + 1)
        )
        assert max(_active(single[cute_flash.FLASH_KV_STAGE_KEY])) == kv_stage
        assert set(_active(single[cute_flash.FLASH_EPI_STG_KEY])) == (
            {False, True} if staged_searched else {False}
        )
    # A deeper request clamps; the staged store yields to the ring at 256.
    deep = _resolve(
        head_dim,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_KV_STAGE_KEY: 4,
            cute_flash.FLASH_EPI_STG_KEY: True,
        },
        is_causal=True,
    )
    assert deep.kv_stage == kv_stage
    assert deep.epi_stg == (head_dim != 256)
    staged64 = _resolve(
        head_dim,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_EPI_STG_KEY: True,
            cute_flash.FLASH_Q_TILE_M_KEY: 64,
        },
        is_causal=False,
    )
    assert staged64.kv_stage >= 1


@pytest.mark.parametrize("head_dim", (192, 256))
def test_dense_surface_opens_above_128_on_the_single_tile_body(
    head_dim: int,
) -> None:
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(2, 4, 2048, head_dim, dtype=torch.bfloat16) for _ in range(3)
    )
    _dense_causal_attention.reset()
    with _cpu_codegen():
        bound = _dense_causal_attention.bind((q, k, v))
        spec = bound.config_spec
        assert spec.cute_flash_search_enabled
        assert spec._cute_flash_head_dim == head_dim
        default = spec.default_config()
        assert default.config["block_sizes"] == [1, 128, 128]
        # The default is the body's cooperative pair (an even tile count).
        assert default.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "ws_2cta"
        assert default.config[cute_flash.FLASH_KV_STAGE_KEY] == (
            2 if head_dim == 256 else 3
        )
        code = bound.to_code(default)
        # The single CTA's body at its own defaults: the one-deep ring, the
        # staged output tile where it fits beside that ring.
        single = bound.to_code(
            bound._normalized_config_copy(
                helion.Config(
                    **{
                        **default.config,
                        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                        cute_flash.FLASH_KV_STAGE_KEY: 1,
                        cute_flash.FLASH_EPI_STG_KEY: head_dim != 256,
                    }
                )
            )
        )
    assert "'kind': 'helion_flash'" in code
    assert f"'head_dim': {head_dim}," in code
    assert "'topology': 'ws_overlap'" in code
    assert "'use_2cta_instrs': True" in code
    assert "'use_2cta_instrs': False" in single
    # The staged output tile fits beside the one-deep ring at 192, not at 256.
    o_stage = 0 if head_dim == 256 else 1
    assert (
        f"flash_shared_storage({head_dim}, 1, 2, cutlass.BFloat16, o_stage={o_stage}"
        in single
    )


@pytest.mark.parametrize("head_dim", (192, 256))
def test_pinned_pair_surface_searches_only_the_pair_knobs(head_dim: int) -> None:
    """A surface pinned to the cooperative pair searches neither the score
    stage count nor the staged epilogue (the resolver pins them: two score
    stages, the direct TMEM store), and the resolver re-canonicalises the
    staged store form after un-staging so one render has one identity."""
    with patch.dict(os.environ, {}, clear=True):
        fragments = cute_flash.flash_autotune_fragments(
            head_dim,
            32,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
            pipeline_family_override="ws_2cta",
        )
    for key, value in (
        (cute_flash.FLASH_S_STAGE_KEY, 2),
        (cute_flash.FLASH_EPI_TMA_KEY, False),
        (cute_flash.FLASH_EPI_STG_KEY, False),
        (cute_flash.FLASH_EPI_STG_STORE_KEY, "slice"),
        (cute_flash.FLASH_EPI_STG_GMEM_KEY, "stage"),
    ):
        assert _active(fragments[key]) == (value,), key
    # The ring is searched over the pair's half-tile byte model.
    pair_cap = cute_flash._flash_ws_kv_stage_cap(
        head_dim, q_tile_m=128, s_stage=2, epi_stg=False, kv_rows=64, two_cta=True
    )
    assert set(_active(fragments[cute_flash.FLASH_KV_STAGE_KEY])) == set(
        range(2, pair_cap + 1)
    )
    for store in ("slice", "whole"):
        resolved = _resolve(
            head_dim,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
                cute_flash.FLASH_EPI_STG_KEY: True,
                cute_flash.FLASH_EPI_STG_STORE_KEY: store,
            },
        )
        assert resolved.use_2cta_instrs and not resolved.epi_stg
        assert resolved.epi_stg_store == "slice"
        assert resolved.epi_stg_gmem == "stage"


def test_flex_list_walk_runs_on_the_cooperative_pair_above_128() -> None:
    """Above value width 128 the BlockMask list walk runs on the single-tile
    body's cooperative pair (the fa4 pair body's TMEM cannot host the width):
    the matcher admits the symmetric widths with an even query tile count,
    the resolver pins ``ws_2cta`` whatever the request and the device body
    reads the pair's two table rows into per-stage bitmasks, walks their
    union on per-stream cursors and applies the mask hook on partial tiles."""
    module = import_path(EXAMPLES_DIR / "flex_attention.py")
    kernel = Kernel(
        module.helion_flex_attention_kernel.fn,
        settings=module.helion_flex_attention_kernel.settings.copy(backend="cute"),
    )
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 512, 256, dtype=torch.bfloat16) for _ in range(3))
    block_mask = create_block_mask(_causal, 2, 4, 512, 512, device=torch.device("cpu"))
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    assert full_num is not None and full_idx is not None
    args = (
        q,
        k,
        v,
        _identity,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        block_mask.mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        1.0 / math.sqrt(256),
        False,
    )
    with _cpu_codegen():
        bound = kernel.bind(args)
        spec = bound.config_spec
        assert spec.cute_flash_search_enabled
        default = spec.default_config()
        assert default.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "ws_2cta"
        code = bound.to_code(default)
        requested = bound.to_code(
            bound._normalized_config_copy(
                helion.Config(
                    **{**default.config, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
                )
            )
        )
        # The pair's own surface: the knobs its body realizes are searched
        # (the KV order and launch group, the packed reduce, the mask form);
        # the fa4 pair body's walk knobs, the second pair per cluster and the
        # single score stage are not (the resolver pins them away), so the
        # structural coverage design has a witness for every searched value
        # (the cell's autotune aborted on the uncovered ones).
        from helion.autotuner.config_generation import ConfigGeneration

        fragments = spec._cute_flash_autotune_fragments()
        assert ConfigGeneration(spec).flash_structural_coverage_uncovered_values() == []
    searched = {
        key: set(_active(fragment))
        for key, fragment in fragments.items()
        if len(_active(fragment)) > 1
    }
    assert searched == {
        cute_flash.FLASH_PACKED_REDUCE_KEY: {False, True},
        cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: set(
            cute_flash._flash_causal_lpt_candidates(8)[1]
        ),
        cute_flash.FLASH_CAUSAL_LPT_L2_SHIFT_KEY: {0, 1},
        cute_flash.FLASH_FLEX_MASK_FORM_KEY: {"bitmask", "element"},
        # The pair's separate load warp: the walk's union-list loads move
        # with their cursors.
        cute_flash.FLASH_WS_LOADER_WARP_KEY: {False, True},
        # The extra K ring slot fits beside the pair's two-deep V ring at
        # 256; the mask phases run the item's leading full tiles without
        # the hook.
        cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY: {0, 1},
        cute_flash.FLASH_WS_MASK_PHASES_KEY: {False, True},
    }
    assert _active(fragments[cute_flash.FLASH_S_STAGE_KEY]) == (2,)
    assert _active(fragments[cute_flash.FLASH_KV_STAGE_KEY]) == (2,)
    assert _active(fragments[cute_flash.FLASH_PAIR_CLUSTER_KEY]) == (1,)
    assert _active(fragments[cute_flash.FLASH_FLEX_ITEM_WORDS_KEY]) == (True,)
    for text in (code, requested):
        assert "'kind': 'helion_flash'" in text
        assert "'use_2cta_instrs': True" in text
        assert "'persistent': False" in text
        # The item's bitmask words come from the host table (one word per
        # list here), the item order from the host-sorted launch groups.
        assert "_flash_mFlexWords = flash_flex_words" in text
        assert "flash_item = cutlass.Int32(_flash_mFlexItems[flash_pid])" in text
        assert "_flash_mFlexPartCnt" not in text
        # The pair's two table rows (their words from the host table), the
        # CTA's own stage lists, the union count.
        assert "flash_m_tile1 = flash_m_tile0 + cutlass.Int32(1)" in text
        assert "flash_flex_p1_0 = cutlass.Int32(_flash_mFlexWords[" in text
        assert "flash_flex_pm_0 = cutlass.Int32(cutlass.select_(" in text
        assert "flash_active_count = flash_num_active_kv" in text
        # Per-stream cursors: the K and V loads and the softmax each step
        # their own copy of the union.
        assert "flash_flex_remk0 = flash_flex_u0" in text
        assert "flash_flex_remv0 = flash_flex_u0" in text
        assert "flash_flex_rems0 = flash_flex_u0" in text
        assert "tKgK[None, flash_kv_k]" in text and "tVgV[None, flash_kv_v]" in text
        # The per-tile status, the mask hook on partial tiles, the zero P
        # tile on absent ones.
        assert (
            "flash_flex_status = cutlass.Int32(cutlass.select_(flash_flex_pm_0 & "
            "flash_flex_kvbit != 0" in text
        )
        assert "flash_flex_needs_mask = flash_flex_status == 2" in text
        assert "if flash_flex_needs_mask:" in text
        assert "for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):" in text
        assert "flash_zero_p = cute.make_rmem_tensor(tSTcS.shape" in text


def test_flex_list_walk_above_128_declines_what_the_pair_cannot_host() -> None:
    """The single-tile pair takes the list walk's envelopes above 128 --
    ragged lengths, a dead second tile, the score region, the cursor walk
    past the register bitmask, strided operands -- except one KV tile."""
    from helion._compiler.cute.flex_match import single_tile_pair_walk_decline

    module = import_path(EXAMPLES_DIR / "flex_attention.py")
    kernel = Kernel(
        module.helion_flex_attention_kernel.fn,
        settings=module.helion_flex_attention_kernel.settings.copy(backend="cute"),
    )

    def declines(
        q_seq: int, kv_seq: int, score_mod: Any = _identity, *, transposed: bool = False
    ) -> bool:
        torch.manual_seed(0)
        q = torch.randn(1, 2, q_seq, 256, dtype=torch.bfloat16)
        k = torch.randn(1, 2, kv_seq, 256, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        if transposed:
            # ``[B, S, H, D].transpose(1, 2)`` views: rank-4 TMA maps.
            q, k, v = (
                torch.empty(
                    t.shape[0], t.shape[2], t.shape[1], t.shape[3], dtype=t.dtype
                )
                .transpose(1, 2)
                .copy_(t)
                for t in (q, k, v)
            )
        block_mask = create_block_mask(
            _causal, 1, 2, q_seq, kv_seq, device=torch.device("cpu")
        )
        full_num = block_mask.full_kv_num_blocks
        full_idx = block_mask.full_kv_indices
        assert full_num is not None and full_idx is not None
        args = (
            q,
            k,
            v,
            score_mod,
            block_mask.kv_num_blocks,
            block_mask.kv_indices,
            full_num,
            full_idx,
            block_mask.mask_mod,
            block_mask.BLOCK_SIZE[0],
            block_mask.BLOCK_SIZE[1],
            1.0 / math.sqrt(256),
            False,
        )
        kernel.reset()
        with _cpu_codegen():
            bound = kernel.bind(args)
            return not bound.config_spec.cute_flash_search_enabled

    assert not declines(512, 512)
    assert not declines(384, 512)  # an odd query tile count (a dead second tile)
    assert not declines(500, 576)  # ragged lengths (half-live rows, a KV tail)
    assert declines(512, 128)  # one KV tile
    assert not declines(256, 8192 + 128)  # past the register bitmask: the cursor
    assert not declines(512, 512, transposed=True)  # strided operands: rank-4 maps

    def _bias(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + (q_idx - kv_idx).to(score.dtype)

    assert not declines(512, 512, _bias)  # a score region
    assert single_tile_pair_walk_decline.__doc__ is not None


# --------------------------------------------------------------------------
# the cooperative pair of the single-tile body (``ws_2cta``)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("is_causal", (False, True), ids=("dense", "causal"))
@pytest.mark.parametrize(
    ("head_dim", "kv_stage"),
    (
        pytest.param(192, 3, id="d192"),
        pytest.param(256, 2, id="d256"),
    ),
)
def test_cooperative_pair_resolves_on_the_single_tile_body(
    head_dim: int, kv_stage: int, is_causal: bool
) -> None:
    """``ws_2cta`` is the single-tile body's cooperative pair: two adjacent
    128-row query tiles of one batch head share each K/V tile (64 rows per
    CTA), one cluster per pair (no persistent stride), the direct output
    store; the half tiles let the byte model admit a deeper ring than the
    single CTA's."""
    resolved = _resolve(
        head_dim,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_KV_STAGE_KEY: 4,
            cute_flash.FLASH_EPI_STG_KEY: True,
            cute_flash.FLASH_PERSISTENT_KEY: True,
        },
        is_causal=is_causal,
    )
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.topology == "ws_overlap"
    assert resolved.use_2cta_instrs
    assert not resolved.causal_two_cta
    # The request above asked for the pair's persistent cluster loop; the
    # flat cluster grid is the default.
    assert resolved.persistent
    assert resolved.persistent_ctas_per_sm == 1
    flat = _resolve(
        head_dim, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}, is_causal=is_causal
    )
    assert flat.pipeline_family == "ws_2cta" and not flat.persistent
    assert resolved.q_tile_m == 128
    assert resolved.s_stage == 2
    # (The pair's body is the double-buffered-S overlap: one score stage
    # resolves to two.)
    one_stage = _resolve(
        head_dim,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_S_STAGE_KEY: 1,
        },
        is_causal=is_causal,
    )
    assert one_stage.pipeline_family == "ws_2cta" and one_stage.s_stage == 2
    assert not resolved.epi_stg
    assert not resolved.epi_tma
    assert resolved.kv_stage == kv_stage
    if head_dim > 128:
        # Above the deep-ring table the half tiles' byte model bounds the ring.
        assert resolved.kv_stage == cute_flash._flash_ws_kv_stage_cap(
            head_dim,
            q_tile_m=128,
            s_stage=2,
            epi_stg=False,
            kv_rows=64,
            two_cta=True,
        )
    else:
        # The measured widths keep the single-tile body's table band.
        assert resolved.kv_stage == cute_flash._flash_deep_1cta_kv_stage_cap(head_dim)
    single = _resolve(
        head_dim, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}, is_causal=True
    )
    assert single.pipeline_family == "ws_overlap"
    assert not single.use_2cta_instrs
    if head_dim == 256:
        assert single.kv_stage == 1


def test_cooperative_pair_falls_back_to_the_single_cta_body() -> None:
    """The pair needs an even tile count and the plain dense / causal walk:
    an odd count or the list walk resolve the request to the single CTA's
    body (and the search surface drops the family); the asymmetric pair
    keeps its own body."""
    request = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    for kwargs in ({"num_kv": 9}, {"num_kv": 1}):
        resolved = _resolve(192, dict(request), is_causal=True, **kwargs)
        assert resolved.pipeline_family == "ws_overlap", kwargs
        assert not resolved.use_2cta_instrs, kwargs
    # The pair is the family of the value widths above the fa4 pair body's:
    # at and below 128 the request resolves to the single CTA (the fa4 pair
    # body hosts those widths), and the surface never offers it there.
    for head_dim in (64, 128):
        below = _resolve(head_dim, dict(request), is_causal=False)
        assert below.pipeline_family == "ws_overlap" and not below.use_2cta_instrs
        family = _fragments(head_dim, is_causal=False)[
            cute_flash.FLASH_PIPELINE_FAMILY_KEY
        ]
        assert "ws_2cta" not in _active(family)


def test_cooperative_pair_byte_model_matches_the_runtime_storage() -> None:
    cutlass = importlib.import_module("cutlass")
    flash_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_shared_storage
    for head_dim in (128, 192, 256):
        for kv_stage in (1, 2, 3, 4):
            for staged in (False, True):
                runtime = flash_shared_storage(
                    head_dim,
                    kv_stage,
                    2,
                    cutlass.BFloat16,
                    o_stage=1 if staged else 0,
                    q_rows=128,
                    kv_rows=64,
                    two_cta=True,
                ).size_in_bytes()
                assert runtime == cute_flash._flash_ws_storage_bytes(
                    head_dim,
                    kv_stage,
                    q_tile_m=128,
                    s_stage=2,
                    epi_stg=staged,
                    kv_rows=64,
                    two_cta=True,
                )
    # The pair's dealloc mbarrier rides in the header's alignment padding.
    assert cute_flash._flash_ws_storage_bytes(
        256, 2, q_tile_m=128, s_stage=2, epi_stg=False, kv_rows=64, two_cta=True
    ) == cute_flash._flash_ws_storage_bytes(
        256, 2, q_tile_m=128, s_stage=2, epi_stg=False, kv_rows=64
    )
    cap = cute_flash._flash_ws_kv_stage_cap
    assert (
        cap(256, q_tile_m=128, s_stage=2, epi_stg=False, kv_rows=64, two_cta=True) == 2
    )
    assert (
        cap(192, q_tile_m=128, s_stage=2, epi_stg=False, kv_rows=64, two_cta=True) == 3
    )
    assert (
        cap(256, q_tile_m=128, s_stage=2, epi_stg=True, kv_rows=64, two_cta=True) == 1
    )


def test_cooperative_pair_render_structure() -> None:
    """The pair's device body: cluster rank decode, the pair's Q tile index,
    leader-only MMA issue, the follower's half loads, the base-tile
    addressed mask and output, the two-CTA TMEM allocator."""
    code = _render_dense_causal(
        256, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    )
    assert "'use_2cta_instrs': True" in code
    assert "_helion_cute_cluster_shape = (2, 1, 1)" in code
    # Causal pairs are launched inside FA4's L2-sized group of heads (all 8
    # batch heads at seq 2048), a group's pairs longest first (8 pairs).
    assert "flash_lpt_group = flash_pid // cutlass.Int32(64)" in code
    assert "flash_bh = flash_lpt_group * 8 + flash_lpt_mod % 8" in code
    assert "flash_q_mma_tile = cutlass.Int32(7) - flash_lpt_mod // 8" in code
    assert "flash_m_tile = flash_q_mma_tile * 2 + flash_mma_tile_coord_v" in code
    assert "flash_m_tile_hi = flash_m_tile | cutlass.Int32(1)" in code
    assert "kv_rows=64, two_cta=True)" in code
    assert (
        "is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr="
        "storage.tmem_dealloc_mbar.data_ptr()" in code
    )
    assert "tQgQ[None, flash_q_mma_tile]" in code
    assert "gQ = cute.flat_divide(_flash_mQt, cute.select((256, 128, 256)" in code
    assert "if flash_is_leader_cta:" in code
    assert "causal_mask_t2r(tLDrS, tLDcS, flash_q_mma_tile * 2, flash_kv)" in code
    assert "[None, None, None, flash_q_mma_tile * 2, 0, flash_bh]" in code
    assert "cutlass_pipeline_flash.pipeline_init_arrive(cluster_shape_mn=" in code
    assert "PipelineAsyncUmma.create" in code
    assert "flash_active_count = flash_m_tile_hi" in code
    # The flat cluster grid (the default) decodes the cluster id once; the
    # work item is the pair.
    assert "flash_pid = cutlass.Int32(cute.arch.cluster_idx()[0])" in code
    assert "flash_tile_id" not in code
    assert "'persistent': False," in code
    assert "'total_tiles': 64," in code
    # The persistent cluster loop on request: every role strides the
    # cluster ids by the cluster count, decodes its pair at the head of each
    # tile and drains its pipelines once after its tiles; no per-tile CTA
    # barrier (the carried pipelines order the roles across tiles).
    loop = _render_dense_causal(
        256,
        config={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PERSISTENT_KEY: True,
        },
    )
    assert "flash_num_clusters = cutlass.Int32(cute.arch.grid_dim()[0]) // 2" in loop
    assert loop.count("while flash_tile_id < _flash_total_tiles:") == 3
    assert loop.count("flash_pid = flash_tile_id") == 3
    assert loop.count("flash_tile_id = flash_tile_id + flash_num_clusters") == 3
    assert "        cute.arch.barrier()\n" not in loop.split("if warp_idx >= 4:")[1]
    assert loop.count(".tail()") == 6
    assert "'persistent': True," in loop
    # The single CTA's render keeps its text.
    single = _render_dense_causal(
        256, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    )
    assert "'use_2cta_instrs': False" in single
    assert "flash_mma_tile_coord_v" not in single
    assert "tmem_dealloc_mbar" not in single
    assert (
        "flash_shared_storage(256, 1, 2, cutlass.BFloat16, o_stage=0, q_rows=128)"
        in single
    )


def test_cooperative_pair_two_pair_cluster() -> None:
    """``cute_flash_pair_cluster`` = 2 puts two MMA pairs (four consecutive
    query tiles) in one cluster that shares every K/V tile by TMA multicast
    across the pairs: the launcher's multicast K/V atoms over the four-CTA
    cluster layout, quarter-tile partitions, K/V slots released by both
    pairs' MMAs (two commits), Q per pair; the item is four tiles. Legal
    only where four tiles divide the sequence; the pair stays the default."""
    resolved = _resolve(
        256,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PAIR_CLUSTER_KEY: 2,
        },
        is_causal=True,
    )
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.pair_cluster == 2
    assert (
        _resolve(256, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}).pair_cluster
        == 1
    )
    odd = _resolve(
        256,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PAIR_CLUSTER_KEY: 2,
        },
        num_kv=6,
    )
    assert odd.pipeline_family == "ws_2cta" and odd.pair_cluster == 1
    single = _resolve(
        256,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_PAIR_CLUSTER_KEY: 2,
        },
    )
    assert single.pair_cluster == 1
    fragments = _fragments(256, is_causal=True)
    assert _active(fragments[cute_flash.FLASH_PAIR_CLUSTER_KEY]) == (1, 2)
    assert _active(_fragments(256, num_kv=6)[cute_flash.FLASH_PAIR_CLUSTER_KEY]) == (1,)
    code = _render_dense_causal(
        256,
        config={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_PAIR_CLUSTER_KEY: 2,
        },
    )
    # (The launcher builds the wrapper, with its multicast K/V atoms over
    # the four-CTA cluster layout, from the plan's ``pair_cluster`` at launch.)
    assert "_helion_cute_cluster_shape = (4, 1, 1)" in code
    assert "'pair_cluster': 2" in code
    assert "'total_tiles': 32," in code
    assert "cute.make_layout((4, 1, 1)), (_flash_qk_mma.thr_id.shape,))" in code
    assert "flash_pair_in_cluster = flash_cta_rank // 2" in code
    assert "flash_m_tile_hi = flash_m_tile | cutlass.Int32(3)" in code
    assert (
        "flash_q_mma_tile = (cutlass.Int32(3) - flash_lpt_mod // 8) * 2 + flash_pair_in_cluster"
        in code
    )
    assert "mcast_mode=1)" in code
    assert "_flash_tma_k, flash_cta_coord_vmnk[1], cute.make_layout(2)" in code
    assert "_flash_tma_v, flash_cta_coord_vmnk[1], cute.make_layout(2)" in code
    assert code.count("mcast_mask=flash_kv_mcast_mask)") == code.count(
        "cute.copy(_flash_tma_k"
    ) + code.count("cute.copy(_flash_tma_v")
    assert code.count("Agent.Thread, 2), tx_count=flash_") == 2
    assert "mcast_mode_mn=(0, 1)" in code and "mcast_mode_mn=(1, 0)" in code
    # The pair's own render has none of it.
    pair = _render_dense_causal(
        256, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    )
    assert "mcast" not in pair and "'pair_cluster': 1" in pair


def test_cooperative_pair_launch_group_knobs() -> None:
    """The pair keeps FA4's launch-group knobs through normalization (the
    single CTA's body canonicalizes them to 0) and decodes an explicit
    group, FA4's residual group included (six heads in groups of four:
    the second group is two heads wide)."""
    kept = _resolve(
        256,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 4,
            cute_flash.FLASH_CAUSAL_LPT_L2_SHIFT_KEY: 1,
        },
        is_causal=True,
    )
    assert kept.causal_lpt_swizzle == 4
    # (An explicit group carries no budget shift, as on the fa4 body.)
    assert kept.causal_lpt_l2_shift == 0
    shifted = _resolve(
        256,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_CAUSAL_LPT_L2_SHIFT_KEY: 1,
        },
        is_causal=True,
    )
    assert shifted.causal_lpt_swizzle == 0 and shifted.causal_lpt_l2_shift == 1
    single = _resolve(
        256,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 4,
        },
        is_causal=True,
    )
    assert single.causal_lpt_swizzle == 0
    fragments = _fragments(256, is_causal=True)
    assert set(_active(fragments[cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY])) >= {0, 1}
    assert set(_active(fragments[cute_flash.FLASH_CAUSAL_LPT_L2_SHIFT_KEY])) == {0, 1}
    code = _render_dense_causal(
        256,
        batch=1,
        heads=6,
        config={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 4,
        },
    )
    assert "flash_lpt_group = flash_pid // cutlass.Int32(32)" in code
    assert "flash_lpt_width = cutlass.Int32(4)" in code
    assert "if flash_lpt_group >= 1:" in code
    assert "flash_lpt_width = cutlass.Int32(2)" in code
    assert "flash_bh = flash_lpt_group * 4 + flash_lpt_mod % flash_lpt_width" in code
    assert (
        "flash_q_mma_tile = cutlass.Int32(7) - flash_lpt_mod // flash_lpt_width" in code
    )


def test_cooperative_pair_cell_space_enumerates_under_the_qualification_floor() -> None:
    """The d256 pair cell's surface (two families, eight searched knobs) is a
    small space: its 384 raw combinations normalize to a few dozen configs,
    all of which the autotuner's initial population measures. The structural
    qualification enumerates it under its raw floor so the lanes are provably
    exhausted (an unenumerated space left the lanes "incomplete" and aborted
    the autotune of the cell)."""
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.surrogate_pattern_search import _FLASH_EXACT_SPACE_RAW_FLOOR

    torch.manual_seed(0)
    q, k, v = (torch.randn(4, 8, 4096, 256, dtype=torch.bfloat16) for _ in range(3))
    _dense_causal_attention.reset()
    with _cpu_codegen():
        bound = _dense_causal_attention.bind((q, k, v))
        gen = ConfigGeneration(bound.config_spec)
        assert gen.flash_structural_coverage_uncovered_values() == []
        space = gen.flash_exact_effective_search_space_configs(
            _FLASH_EXACT_SPACE_RAW_FLOOR
        )
    assert space is not None
    families = {c.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] for c in space}
    assert families == {"ws_2cta", "ws_overlap"}
    assert 30 <= len(space) <= _FLASH_EXACT_SPACE_RAW_FLOOR


def test_mixed_pair_surface_pins_the_score_stage() -> None:
    """Where the cooperative pair is on the surface its body is the
    double-buffered-S overlap: one score stage leaves the search (the
    resolver pins the pair's to two, so a searched 1 was the pair's own
    render under another name, half of the cell's raw points); the single
    CTA's surface (an odd tile count) keeps both, and both stay legal
    explicit requests everywhere."""
    for head_dim in (144, 192, 256):
        for is_causal in (False, True):
            paired = _fragments(head_dim, is_causal=is_causal)
            assert "ws_2cta" in _active(paired[cute_flash.FLASH_PIPELINE_FAMILY_KEY])
            assert _active(paired[cute_flash.FLASH_S_STAGE_KEY]) == (2,)
            assert set(paired[cute_flash.FLASH_S_STAGE_KEY].choices) == {1, 2}
            single = _fragments(head_dim, is_causal=is_causal, num_kv=9)
            assert set(_active(single[cute_flash.FLASH_S_STAGE_KEY])) == {1, 2}
    # The single-warpgroup body stays an explicit request beside the pair.
    one_stage = _resolve(
        256,
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_S_STAGE_KEY: 1,
        },
    )
    assert one_stage.s_stage == 1 and not one_stage.use_2cta_instrs


def test_ws_only_kv_stage_legal_set_keeps_the_table_band() -> None:
    """The single-tile body's searched ring depths at the measured widths are
    the table band (the one-deep ring is not searched there); the legal set
    keeps the historical (2, 3) and the one-deep ring at every width, so a
    config carried over from another width or shape (an autotune cache, a
    saved campaign config: 1 is the resolved value above the deep-ring table
    and at one KV tile) normalizes instead of failing -- the resolver clamps
    it to the width's band."""
    for head_dim in (96, 128):
        single = _fragments(head_dim, num_kv=9)
        choices = tuple(single[cute_flash.FLASH_KV_STAGE_KEY].choices)
        assert {1, 2, 3} <= set(choices), (head_dim, choices)
        assert 1 not in _active(single[cute_flash.FLASH_KV_STAGE_KEY])
        clamped = _resolve(
            head_dim,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_KV_STAGE_KEY: 1,
            },
            num_kv=9,
        )
        assert clamped.kv_stage == 2
    searched128 = _fragments(128, num_kv=9)[cute_flash.FLASH_KV_STAGE_KEY]
    assert _active(searched128) == (2,)
    assert tuple(searched128.choices) == (2, 1, 3)
    # Above the table the one-deep ring is the single CTA's resolved depth
    # where the byte model allows no more (192, 256; 144 holds two).
    for head_dim, depth in ((144, 2), (192, 1), (256, 1)):
        for num_kv in (9, 32):
            fragment = _fragments(head_dim, num_kv=num_kv)[
                cute_flash.FLASH_KV_STAGE_KEY
            ]
            assert {1, 2, 3} <= set(fragment.choices), (head_dim, fragment.choices)
            assert (1 in _active(fragment)) == (depth == 1), (head_dim, num_kv)
        one_deep = _resolve(
            head_dim,
            {
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_KV_STAGE_KEY: 1,
            },
            num_kv=9,
        )
        assert one_deep.kv_stage == depth, head_dim


def test_fa4_pair_plans_keep_their_host_line() -> None:
    """The ``pair_cluster`` wrapper-plan key is the single-tile body's
    cooperative pair's alone: the fa4 2-CTA family's host line carries no
    cluster size (it is byte-identical to the line before the pair family)."""
    fa4_pair = _render_dense(
        64, causal=False, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"}
    )
    assert "'use_2cta_instrs': True" in fa4_pair
    assert "'topology': 'fa4'" in fa4_pair
    assert "'pair_cluster'" not in fa4_pair
    pair = _render_dense(
        256, causal=False, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    )
    assert "'pair_cluster': 1" in pair


def test_direct_store_pins_re_canonicalize_the_staged_store_children() -> None:
    """The pair's direct output store and the single CTA's staged store
    yielding to the ring at 256 are decided after the staged store's
    children were canonicalized (while the ws_overlap default still staged
    the output): the children re-canonicalize there, so the resolved config
    is a fixed point of normalization and ``slice`` / ``whole`` are no longer
    two identities of one direct-store render."""
    for request in (
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"},
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_EPI_STG_STORE_KEY: "whole",
        },
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
        {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            cute_flash.FLASH_EPI_STG_KEY: True,
            cute_flash.FLASH_EPI_STG_STORE_KEY: "whole",
        },
    ):
        for is_causal in (False, True):
            resolved = _resolve(256, dict(request), is_causal=is_causal)
            assert not resolved.epi_stg and not resolved.epi_tma, request
            assert resolved.epi_stg_store == "slice", request
            effective = cute_flash.flash_effective_config_values(resolved)
            again = _resolve(256, dict(effective), is_causal=is_causal)
            assert again.epi_stg_store == "slice" and not again.epi_stg
    # The staged store that fits (192, the single CTA) keeps its children.
    staged = _resolve(192, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"})
    assert staged.epi_stg and staged.epi_stg_store == "whole"
    staged_pair = _resolve(192, {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"})
    assert not staged_pair.epi_stg and staged_pair.epi_stg_store == "slice"


# Rendering the entire normalized space takes nearly a minute even without
# other xdist workers contending for the CPU.
@pytest.mark.timeout(180)
def test_d256_causal_pair_cell_identities_render_distinct_texts() -> None:
    """Every normalized config of the d256 causal cell's exact space renders a
    distinct text: no two identities are one measurement. (Before the
    direct-store pins re-canonicalized the staged store's children, the
    pair's ``slice`` and ``whole`` were two identities of one render: half
    of the cell's measured pair configs were duplicates.) At b4 h8 s4096 the
    L2-sized launch group is eight heads, apart from the serial and
    whole-grid groups, so the documented ``0`` / ``num_bh`` alias of the
    group's strategy names does not occur."""
    import collections

    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.surrogate_pattern_search import _FLASH_EXACT_SPACE_RAW_FLOOR

    torch.manual_seed(0)
    q, k, v = (torch.randn(4, 8, 4096, 256, dtype=torch.bfloat16) for _ in range(3))
    _dense_causal_attention.reset()
    with _cpu_codegen():
        bound = _dense_causal_attention.bind((q, k, v))
        gen = ConfigGeneration(bound.config_spec)
        space = gen.flash_exact_effective_search_space_configs(
            _FLASH_EXACT_SPACE_RAW_FLOOR
        )
        assert space is not None
        identities: dict[tuple[tuple[str, str], ...], helion.Config] = {}
        for config in space:
            normalized = bound._normalized_config_copy(config)
            key = tuple(sorted((k, repr(v)) for k, v in normalized.config.items()))
            identities.setdefault(key, normalized)
        texts = {key: bound.to_code(config) for key, config in identities.items()}
    # The exact space is the normalized space: no two of its rows share an
    # identity, and no two identities share a render.
    assert len(identities) == len(space)
    by_text: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for key, text in texts.items():
        by_text[text].append(dict(key))
    duplicates = [group for group in by_text.values() if len(group) > 1]
    assert duplicates == [], [
        {
            field: sorted({row[field] for row in group})
            for field in group[0]
            if any(row[field] != group[0][field] for row in group)
        }
        for group in duplicates
    ]
    families = collections.Counter(
        dict(key)[cute_flash.FLASH_PIPELINE_FAMILY_KEY] for key in identities
    )
    assert set(families) == {"'ws_2cta'", "'ws_overlap'"}
    for key in identities:
        row = dict(key)
        if row[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "'ws_2cta'":
            # The pair's identities carry neither the single score stage nor
            # the staged store's children.
            assert row[cute_flash.FLASH_S_STAGE_KEY] == "2"
            assert row[cute_flash.FLASH_EPI_STG_STORE_KEY] == "'slice'"
            assert row[cute_flash.FLASH_EPI_STG_KEY] == "False"


# -- (y) the single-tile bodies' issue-stream knobs: the pair's load warp -----------


def test_load_warp_resolves_on_the_cooperative_pair() -> None:
    """The separate load warp exists on the two-stage single-tile bodies: the
    cooperative pair (dense, causal, persistent and the list walk alike) and
    the single CTA; the fa4 bodies normalize the request away so one render
    keeps one identity."""
    pair = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
        cute_flash.FLASH_WS_LOADER_WARP_KEY: True,
    }
    for head_dim in (144, 192, 256):
        assert _resolve(head_dim, pair).ws_loader_warp, head_dim
    assert _resolve(256, pair, is_causal=False).ws_loader_warp
    assert _resolve(256, {**pair, cute_flash.FLASH_PERSISTENT_KEY: True}).ws_loader_warp
    walk = _resolve(256, pair, has_list_walk=True)
    assert walk.use_2cta_instrs and walk.ws_loader_warp
    single = {**pair, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    for head_dim in (64, 128, 192, 256):
        resolved = _resolve(head_dim, single)
        assert not resolved.use_2cta_instrs and resolved.ws_loader_warp, head_dim
    assert _resolve(128, single, is_causal=False).ws_loader_warp
    fa4 = _resolve(128, {**pair, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"})
    assert fa4.topology == "fa4" and not fa4.ws_loader_warp
    # The resolved value is a fixed point of normalization.
    resolved = _resolve(256, pair)
    again = _resolve(256, dict(cute_flash.flash_effective_config_values(resolved)))
    assert again.ws_loader_warp


def test_load_warp_is_searched_on_the_pinned_pair_surface() -> None:
    """Every surface whose rows are all two-stage single-tile bodies searches
    the load warp (the pinned pair, the mixed d256 causal surface of pair
    and single CTA, a ws-only d64 surface); the fa4-eligible surfaces keep
    it at rest (a value their fa4 rows would normalize away is an
    unreachable column of the coverage design)."""
    with patch.dict(os.environ, {}, clear=True):
        pinned = cute_flash.flash_autotune_fragments(
            256,
            32,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
            pipeline_family_override="ws_2cta",
        )
    assert _active(pinned[cute_flash.FLASH_WS_LOADER_WARP_KEY]) == (False, True)
    assert _active(_fragments(256)[cute_flash.FLASH_WS_LOADER_WARP_KEY]) == (
        False,
        True,
    )
    assert _active(_fragments(128)[cute_flash.FLASH_WS_LOADER_WARP_KEY]) == (False,)
    with patch.dict(os.environ, {}, clear=True):
        ws_only = cute_flash.flash_autotune_fragments(
            64,
            48,
            num_bh=64,
            is_causal=False,
            standard_dense_output=True,
            pipeline_family_override="ws_overlap",
        )
    assert _active(ws_only[cute_flash.FLASH_WS_LOADER_WARP_KEY]) == (False, True)
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            256,
            32,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
        )
    pair_seeds = [
        seed.config
        for seed in seeds
        if seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == "ws_2cta"
    ]
    assert {seed.get(cute_flash.FLASH_WS_LOADER_WARP_KEY) for seed in pair_seeds} == {
        False,
        True,
    }


def _leader_and_loader(code: str) -> tuple[str, str]:
    leader = code.index("if flash_is_leader_cta:")
    loader = code.index("if warp_idx == 2:")
    consumer = code.index("if warp_idx >= 4:", loader)
    return code[leader:loader], code[loader:consumer]


def test_render_d256_cooperative_pair_load_warp() -> None:
    """With the load warp on, warp 2 of both CTAs runs every TMA load and the
    ring tails while the leader's warp 0 keeps the MMA stream alone; off, the
    render is the pair's unchanged body. The flex list walk renders it the
    same way (its union-list loads move with their cursors)."""
    base = _render_dense_causal(
        256, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    )
    assert "if warp_idx == 2:" not in base
    for code in (
        _render_dense_causal(
            256,
            config={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
                cute_flash.FLASH_WS_LOADER_WARP_KEY: True,
            },
        ),
        _render_dense_causal(
            256,
            config={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
                cute_flash.FLASH_WS_LOADER_WARP_KEY: True,
                cute_flash.FLASH_PERSISTENT_KEY: True,
            },
        ),
        _render_flex(
            256,
            config={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
                cute_flash.FLASH_WS_LOADER_WARP_KEY: True,
            },
        ),
    ):
        leader, loader = _leader_and_loader(code)
        assert "cute.copy(_flash_tma_" not in leader
        assert (
            "cute.gemm(_flash_qk_mma" in leader and "cute.gemm(_flash_pv_mma" in leader
        )
        assert loader.count("cute.copy(_flash_tma_k") >= 3
        assert "cute.gemm(" not in loader
        for tail in (
            "flash_q_prod.tail()",
            "flash_k_prod.tail()",
            "flash_v_prod.tail()",
        ):
            assert tail in loader and tail not in leader


# -- (y) the single-tile bodies' issue-stream knobs: the causal mask phases ----------


def test_mask_phases_resolve_on_the_causal_single_tile_bodies() -> None:
    """The diagonal split exists on the causal two-stage single-tile bodies
    (the pair and the single CTA) and, as the leading run of full tiles, on
    the list walk; dense plans and the fa4 bodies normalize it away."""
    pair = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
        cute_flash.FLASH_WS_MASK_PHASES_KEY: True,
    }
    for head_dim in (144, 192, 256):
        assert _resolve(head_dim, pair).ws_mask_phases, head_dim
    assert _resolve(256, {**pair, cute_flash.FLASH_PERSISTENT_KEY: True}).ws_mask_phases
    assert not _resolve(256, pair, is_causal=False).ws_mask_phases
    # The list walk: the item's leading run of full tiles is the phase.
    assert _resolve(256, pair, has_list_walk=True).ws_mask_phases
    assert _resolve(256, pair, is_causal=False, has_list_walk=True).ws_mask_phases
    single = {**pair, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    for head_dim in (64, 128, 192, 256):
        resolved = _resolve(head_dim, single)
        assert not resolved.use_2cta_instrs and resolved.ws_mask_phases, head_dim
    assert not _resolve(128, single, is_causal=False).ws_mask_phases
    fa4 = _resolve(128, {**pair, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"})
    assert fa4.topology == "fa4" and not fa4.ws_mask_phases
    resolved = _resolve(128, single)
    again = _resolve(128, dict(cute_flash.flash_effective_config_values(resolved)))
    assert again.ws_mask_phases


def test_mask_phases_are_searched_on_causal_single_tile_surfaces() -> None:
    """Every causal surface whose rows are all single-tile bodies searches the
    split (the pinned pair, the mixed d256 surface); dense pair and fa4
    surfaces keep it at rest."""
    with patch.dict(os.environ, {}, clear=True):
        pinned = cute_flash.flash_autotune_fragments(
            256,
            32,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
            pipeline_family_override="ws_2cta",
        )
        dense_pinned = cute_flash.flash_autotune_fragments(
            256,
            32,
            num_bh=64,
            is_causal=False,
            standard_dense_output=True,
            pipeline_family_override="ws_2cta",
        )
    key = cute_flash.FLASH_WS_MASK_PHASES_KEY
    assert _active(pinned[key]) == (False, True)
    assert _active(dense_pinned[key]) == (False,)
    assert _active(_fragments(256)[key]) == (False, True)
    assert _active(_fragments(128)[key]) == (False,)
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            256,
            32,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
        )
    assert any(
        seed.config.get(key) and seed.config.get(cute_flash.FLASH_WS_LOADER_WARP_KEY)
        for seed in seeds
    )


def test_render_mask_phases_split_the_diagonal_out() -> None:
    """The split loop: the tiles below the diagonal without the causal mask,
    then the diagonal tiles with it (one mask call in the render), on the
    pair and on the single CTA; off, the single loop with its guarded mask."""
    for family, head_dim in (("ws_2cta", 256), ("ws_overlap", 192)):
        base = _render_dense_causal(
            head_dim, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: family}
        )
        code = _render_dense_causal(
            head_dim,
            config={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: family,
                cute_flash.FLASH_WS_MASK_PHASES_KEY: True,
            },
        )
        assert (
            base.count("causal_mask_t2r(") == 1 and code.count("causal_mask_t2r(") == 1
        )
        assert "cutlass.range(flash_m_tile - cutlass.Int32(0), unroll=1)" in code
        assert "flash_kv = flash_m_tile + flash_active_kv" in code
        assert "flash_kv = flash_m_tile + flash_active_kv" not in base
        below = code.index("cutlass.range(flash_m_tile - cutlass.Int32(0), unroll=1)")
        diagonal = code.index("flash_kv = flash_m_tile + flash_active_kv")
        assert "causal_mask_t2r(" not in code[below:diagonal]
        assert "causal_mask_t2r(" in code[diagonal:]


# -- (y) the single-tile bodies' issue-stream knobs: the extra K ring slot ----------


def test_extra_k_slot_resolves_where_it_fits() -> None:
    """One more K ring slot where the byte model fits it beside the V ring:
    the pair at 144-256 (dense, causal, the list walk), the single CTA by
    bytes (not at 256: its full-height ring leaves no room); the fa4 bodies
    keep their own rings."""
    pair = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
        cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY: 1,
    }
    for head_dim in (144, 192, 256):
        assert _resolve(head_dim, pair).ws_k_stage_extra == 1, head_dim
    assert _resolve(256, pair, is_causal=False).ws_k_stage_extra == 1
    assert _resolve(256, pair, has_list_walk=True).ws_k_stage_extra == 1
    assert (
        _resolve(256, {**pair, cute_flash.FLASH_PERSISTENT_KEY: True}).ws_k_stage_extra
        == 1
    )
    single = {**pair, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    limit = cute_flash.FlashScheduleLimits().shared_memory_bytes
    for head_dim in (64, 128, 192, 256):
        resolved = _resolve(head_dim, single)
        fits = (
            cute_flash._flash_ws_storage_bytes(
                head_dim,
                resolved.kv_stage,
                q_tile_m=128,
                s_stage=2,
                epi_stg=resolved.epi_stg,
                k_stage_extra=1,
            )
            <= limit
        )
        assert resolved.ws_k_stage_extra == int(fits), head_dim
    assert _resolve(128, single).ws_k_stage_extra == 1
    assert _resolve(256, single).ws_k_stage_extra == 0
    fa4 = _resolve(128, {**pair, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"})
    assert fa4.topology == "fa4" and fa4.ws_k_stage_extra == 0
    resolved = _resolve(256, pair)
    again = _resolve(256, dict(cute_flash.flash_effective_config_values(resolved)))
    assert again.ws_k_stage_extra == 1


def test_extra_k_slot_byte_model_matches_the_runtime_storage() -> None:
    cutlass = importlib.import_module("cutlass")
    flash_shared_storage = importlib.import_module(
        "helion._compiler.cute._flash_runtime"
    ).flash_shared_storage
    for head_dim in (64, 128, 192, 256):
        for kv_stage in (1, 2, 3):
            for two_cta in (False, True):
                for staged in (False, True):
                    kv_rows = 64 if two_cta else 128
                    runtime = flash_shared_storage(
                        head_dim,
                        kv_stage,
                        2,
                        cutlass.BFloat16,
                        o_stage=1 if staged else 0,
                        q_rows=128,
                        kv_rows=kv_rows,
                        two_cta=two_cta,
                        k_stage=kv_stage + 1,
                    ).size_in_bytes()
                    assert runtime == cute_flash._flash_ws_storage_bytes(
                        head_dim,
                        kv_stage,
                        q_tile_m=128,
                        s_stage=2,
                        epi_stg=staged,
                        kv_rows=kv_rows,
                        two_cta=two_cta,
                        k_stage_extra=1,
                    ), (head_dim, kv_stage, two_cta, staged)
    limit = cute_flash.FlashScheduleLimits().shared_memory_bytes
    pair = {
        "q_tile_m": 128,
        "s_stage": 2,
        "epi_stg": False,
        "kv_rows": 64,
        "two_cta": True,
    }
    assert cute_flash._flash_ws_storage_bytes(256, 2, k_stage_extra=1, **pair) <= limit
    assert cute_flash._flash_ws_storage_bytes(256, 2, k_stage_extra=2, **pair) > limit


def test_extra_k_slot_is_searched_where_every_row_fits_it() -> None:
    """A pinned pair surface searches the slot where it fits beside every
    searched ring depth (192 and 256: rings of at most three); the d144 and
    d176 pair surfaces search five- and four-deep rings whose deepest rows
    have no room for it, so they keep it at rest (a searched value those
    rows normalize away would be an alias of their slot-off row). The mixed
    d256 causal surface does not search it either (the single CTA's
    full-height ring has no room), nor the fa4-eligible d128 surface."""
    key = cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY
    for head_dim, searched in ((144, (0,)), (176, (0,)), (192, (0, 1)), (256, (0, 1))):
        with patch.dict(os.environ, {}, clear=True):
            pinned = cute_flash.flash_autotune_fragments(
                head_dim,
                32,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
                pipeline_family_override="ws_2cta",
            )
        assert _active(pinned[key]) == searched, head_dim
        depths = _active(pinned[cute_flash.FLASH_KV_STAGE_KEY])
        assert (max(depths) <= 3) == (searched == (0, 1)), (head_dim, depths)  # type: ignore[type-var]
    assert _active(_fragments(256)[key]) == (0,)
    assert _active(_fragments(128)[key]) == (0,)
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            256,
            32,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
        )
    assert any(
        seed.config.get(key) == 1
        and seed.config.get(cute_flash.FLASH_WS_LOADER_WARP_KEY)
        and seed.config.get(cute_flash.FLASH_WS_MASK_PHASES_KEY)
        for seed in seeds
    )


def test_render_extra_k_slot_deepens_the_k_ring_alone() -> None:
    code = _render_dense_causal(
        256,
        config={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY: 1,
        },
    )
    base = _render_dense_causal(
        256, config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    )
    assert "k_stage=3)" in code and "k_stage=" not in base
    k_pipe = code.index("flash_k_prod, flash_k_cons = ")
    v_pipe = code.index("flash_v_prod, flash_v_cons = ")
    assert "num_stages=3" in code[k_pipe : k_pipe + 200]
    assert "num_stages=2" in code[v_pipe : v_pipe + 200]
    flex = _render_flex(
        256,
        config={
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta",
            cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY: 1,
        },
    )
    assert "k_stage=3)" in flex


def test_render_flex_walk_mask_phases_run_the_leading_full_tiles_first() -> None:
    """On the list walk the mask phases split the softmax loop at the item's
    leading run of full tiles (``flash_flex_lead``, counted from the words
    at item start): the first loop has no mask hook, the second keeps it;
    the register walk (one word) counts with a select chain, the cursor walk
    (16K keys, four words) with a loop over the item's words. The tile order
    and the handshakes are the single loop's."""
    base = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    on = {**base, cute_flash.FLASH_WS_MASK_PHASES_KEY: True}
    for seq, cursor in ((1024, False), (16384, True)):
        off_code = _render_flex(256, seq=seq, config=base)
        code = _render_flex(256, seq=seq, config=on)
        assert "flash_flex_lead" not in off_code
        assert (
            "for flash_active_kv in cutlass.range(flash_flex_lead, unroll=1):" in code
        )
        assert (
            "for flash_active_kv in cutlass.range(flash_active_count - flash_flex_lead, unroll=1):"
            in code
        )
        lead = code.index("cutlass.range(flash_flex_lead, unroll=1)")
        rest = code.index(
            "cutlass.range(flash_active_count - flash_flex_lead, unroll=1)"
        )
        assert "if flash_flex_needs_mask:" not in code[lead:rest]
        assert "if flash_flex_needs_mask:" in code[rest:]
        assert code.count("flash_s_full = flash_mma_s_cons.wait_and_advance()") == (
            off_code.count("flash_s_full = flash_mma_s_cons.wait_and_advance()") + 1
        )
        assert ("for flash_lw in cutlass.range(" in code) == cursor
        assert (
            "_flash_mFlexWords[flash_flex_wbase"
            in code[: code.index("flash_flex_lead = cutlass.Int32(0)") + 4000]
        ) == cursor


def test_render_single_cta_load_warp_carries_the_look_ahead_loads() -> None:
    """With the load warp on, warp 2 carries the single CTA's whole ring (the
    flat body's early prologue issue included: the ring's producer states
    must be one warp's) and warp 0 the MMA stream alone; off, the render is
    the body's unchanged text with its prologue loads ahead of the roles."""
    for head_dim, extra in ((192, {}), (128, {"cute_flash_mma_interleave": False})):
        base = _render_dense_causal(
            head_dim,
            config={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap", **extra},
        )
        assert "if warp_idx == 2:" not in base
        code = _render_dense_causal(
            head_dim,
            config={
                cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                cute_flash.FLASH_WS_LOADER_WARP_KEY: True,
                **extra,
            },
        )
        loader = code.index("    if warp_idx == 2:")
        role0 = code.rindex("    if warp_idx == 0:", 0, loader)
        consumer = code.index("    if warp_idx >= 4:", loader)
        mma, loads = code[role0:loader], code[loader:consumer]
        assert "cute.copy(_flash_tma_" not in mma and "cute.gemm(" in mma
        assert loads.count("cute.copy(_flash_tma_") >= 2 and "cute.gemm(" not in loads
        # The flat body's early prologue issue (ahead of the roles, from warp
        # 0) moves into the load warp's role: the ring's producer states must
        # be one warp's.
        assert "cute.copy(_flash_tma_q" in base[: base.index("    if warp_idx >= 4:")]
        assert "cute.copy(_flash_tma_" not in code[:role0]
        assert "cute.copy(_flash_tma_q" in loads


def test_extra_k_slot_sizes_the_host_k_layout() -> None:
    """The host K smem layout (the TMA destination and the MMA operand) takes
    the K ring's own depth: a one-stage layout has no stage stride (its slot 1
    is slot 0), so a single-tile body at kv_stage 1 with the extra slot carries
    ``k_stage`` 2 in its plan and the wrapper builds its K layout two deep
    while V's stays one; with the slot off the plan has no such key."""
    from helion.runtime.cute.launcher import _append_cute_wrapper_plan

    def plan_of(code: str) -> dict[str, object]:
        marker = "._helion_cute_wrapper_plans = "
        payload = next(line for line in code.splitlines() if marker in line)
        (plan,) = ast.literal_eval(payload.split(marker, 1)[1])
        assert isinstance(plan, dict)
        return plan

    def host_layout_depths(plan: dict[str, object]) -> tuple[str, str]:
        body: list[str] = []
        _append_cute_wrapper_plan(body, [], plan, num_sm=148)
        source = "\n".join(body)
        ksl = re.search(r"_flash_ksl = .*make_smem_layout_b\(.*, (\d+)\)", source)
        vsl = re.search(r"_flash_vsl = .*make_smem_layout_b\(.*, (\d+)\)", source)
        assert ksl is not None and vsl is not None
        return ksl.group(1), vsl.group(1)

    ws = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    for head_dim, kv_stage, extra in (
        (192, 1, {}),
        (128, 2, {"cute_flash_mma_interleave": False}),
    ):
        base = plan_of(_render_dense_causal(head_dim, config={**ws, **extra}))
        assert base["kv_stage"] == kv_stage and "k_stage" not in base
        assert host_layout_depths(base) == (str(kv_stage), str(kv_stage))
        plan = plan_of(
            _render_dense_causal(
                head_dim,
                config={**ws, **extra, cute_flash.FLASH_WS_K_STAGE_EXTRA_KEY: 1},
            )
        )
        assert plan["kv_stage"] == kv_stage and plan["k_stage"] == kv_stage + 1
        assert host_layout_depths(plan) == (str(kv_stage + 1), str(kv_stage))
