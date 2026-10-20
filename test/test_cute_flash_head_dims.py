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


def test_symmetric_head_dims_are_every_multiple_of_16_up_to_128() -> None:
    assert [d for d in range(300) if flash_head_dim_supported(d)] == list(
        range(16, 129, 16)
    )
    for head_dim in range(16, 129, 16):
        assert flash_head_dims_supported(head_dim, head_dim)
    assert not flash_head_dims_supported(8, 8)
    assert not flash_head_dims_supported(144, 144)
    assert not flash_head_dims_supported(192, 192)
    assert not flash_head_dims_supported(256, 256)
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
