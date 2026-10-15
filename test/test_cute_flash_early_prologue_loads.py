"""``cute_flash_early_prologue_loads``: the fa4 pair body's first loads before
the prologue sync.

The plain fa4 prologue initializes its raw mbarriers, fences, syncs the CTA,
then creates the two TMA pipelines (each with its own init + fence + CTA
sync) and only then lets the load warp issue the first work item's Q0/Q1/K/V
copies. With the knob both pipelines are created ``defer_sync`` ahead of the
one fence + sync, and warp 0 (whose elected lane initialized the pipeline
mbarriers) issues the four copies against the pipelines' full barriers right
after the fence, before the sync. The load warp keeps its four acquires (they
post the arrive.expect_tx; an mbarrier tx-count may run negative until then)
and drops the copies, so the ring state and the outputs are unchanged.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.attention_plan import AttentionScorePlan
    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_EARLY_PROLOGUE_LOADS_KEY
HEAD_PAIR_KEY = cute_flash.FLASH_CAUSAL_HEAD_PAIR_KEY
STAGGER_KEY = cute_flash.FLASH_CAUSAL_STAGGER_KEY
FENCE = "cute.arch.mbarrier_init_fence()"
SYNC = "cute.arch.sync_threads()"
EARLY_GUARD = "if warp_idx == 0:"

# The causal hd64 s2048 campaign body: one-CTA fa4 pair, chunked softmax with
# paired 16-column score loads, 4x1 packet at 8/2, ring2 statistics, two-deep
# K/V ring, Q0/Q1/K/V first-load order (2).
CAUSAL_HD64_BODY: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_kv_tile_n": 128,
    "cute_flash_s_stage": 2,
    "cute_flash_kv_stage": 2,
    "cute_flash_persistent": False,
    "cute_flash_e2e_schedule": "8/2",
    "cute_flash_masked_e2e_schedule": "xu",
    "cute_flash_e2e_offset": 4,
    "cute_flash_e2e_offset0": 4,
    "cute_flash_exp2_packet": "4x1",
    "cute_flash_mma_interleave": True,
    "cute_flash_stat_transport": "ring2",
    "cute_flash_pipeline_family": "fa4",
    "cute_flash_softmax_disc": True,
    "cute_flash_disc_pipe": 1,
    "cute_flash_split_p_arrive": True,
    "cute_flash_p_store_rep": 16,
    "cute_flash_s_load_rep": 16,
    "cute_flash_precompute_qk_desc": True,
    "cute_flash_first_load_order": 2,
    "cute_flash_epi_tma": True,
    "cute_flash_epi_stg": False,
    "cute_flash_rescale_threshold": 32.0,
    "cute_flash_rescale_chunk_cols": 32,
    "cute_flash_softmax_regs": 168,
    "cute_flash_corr_regs": 104,
    "cute_flash_other_regs": 56,
    "cute_flash_corr_tile_size": 32,
    "cute_flash_causal_kv_order": "descending",
    "cute_flash_causal_loop_split": True,
    "cute_flash_causal_skip_dead_tile": False,
}
# The dense hd128 campaign winner's structure, non-persistent, K-first order.
DENSE_HD128_BODY: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_kv_tile_n": 128,
    "cute_flash_s_stage": 2,
    "cute_flash_kv_stage": 3,
    "cute_flash_persistent": False,
    "cute_flash_e2e_schedule": "16/6",
    "cute_flash_e2e_offset": 3,
    "cute_flash_e2e_offset0": 11,
    "cute_flash_exp2_packet": "8x2",
    "cute_flash_mma_interleave": True,
    "cute_flash_stat_transport": "ring2",
    "cute_flash_pipeline_family": "fa4",
    "cute_flash_softmax_disc": True,
    "cute_flash_disc_pipe": 2,
    "cute_flash_split_p_arrive": True,
    "cute_flash_p_store_rep": 16,
    "cute_flash_s_load_rep": 32,
    "cute_flash_first_load_order": 4,
    "cute_flash_epi_tma": True,
    "cute_flash_epi_stg": False,
    "cute_flash_rescale_threshold": 12.0,
    "cute_flash_softmax_regs": 192,
    "cute_flash_corr_regs": 64,
    "cute_flash_other_regs": 48,
    "cute_flash_corr_tile_size": 8,
}


def _resolve(
    config: dict[str, object],
    *,
    is_causal: bool,
    head_dim: int,
    **kwargs: object,
) -> cute_flash.FlashAttentionConfig:
    options: dict[str, object] = {
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
        "plain_row_body": True,
        **kwargs,
    }
    with patch.dict(os.environ, {}, clear=True):
        return cast(
            "cute_flash.FlashAttentionConfig",
            cute_flash.resolve_flash_config(
                head_dim,
                16,
                config,
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=is_causal,
                **options,  # pyrefly: ignore[bad-argument-type]
            ),
        )


def _fragments(
    *, is_causal: bool, head_dim: int, **kwargs: object
) -> dict[str, ConfigSpecFragment]:
    options: dict[str, object] = {
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
        "plain_row_body": True,
        **kwargs,
    }
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            head_dim,
            16,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=is_causal,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


def _render(
    cfg: cute_flash.FlashAttentionConfig,
    *,
    is_causal: bool,
    head_dim: int,
    kv_group: int = 1,
    score_plan: AttentionScorePlan | None = None,
) -> str:
    if score_plan is None:
        score_plan = (
            causal_score_plan(head_dim) if is_causal else dense_score_plan(head_dim)
        )
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=16,
        sequence_extent=16 * 128,
        num_bh=64,
        total_tiles=64 * 8,
        cfg=cfg,
        has_lse=False,
        io_dtype="cutlass.BFloat16",
        score_plan=dataclasses.replace(score_plan, kv_group=kv_group),
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _early_block(code: str) -> str:
    """Warp 0's pre-sync load block, between the init fence and the CTA sync."""
    fence = code.index(FENCE)
    sync = code.index(SYNC)
    early = code.index(EARLY_GUARD, fence)
    assert fence < early < sync
    return code[early:sync]


def _copies(block: str) -> list[str]:
    return [line for line in block.splitlines() if "cute.copy(" in line]


# A sliding-window plan: one KV-tile-pruned mask, no causal modifier.
WINDOW_PLAN = dataclasses.replace(
    causal_score_plan(64),
    modifiers=(AttentionScoreModifier(SLIDING_WINDOW_MASK_KIND, window_size=512),),
)
PRUNED_OPTIONS: dict[str, object] = {
    "has_kv_tile_pruning": True,
    "standard_dense_output": False,
    "standard_causal_output": True,
    "has_score_modifiers": True,
    "plain_row_body": False,
    "score_modifier_kinds": (SLIDING_WINDOW_MASK_KIND,),
}


def _load_warp_prologue(code: str) -> str:
    """The load warp's role block up to its K/V loop."""
    start = code.index("if warp_idx == 14:")
    end = code.index("for flash_kv_load_iter in", start)
    return code[start:end]


# --------------------------------------------------------------------------
# schema, surface and resolver
# --------------------------------------------------------------------------


def test_knob_is_an_appended_schema_axis() -> None:
    # Appended once behind the axes it was introduced after; knobs added later
    # (the flex list walk's) append behind it in turn, so the tail itself is
    # not pinned.
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(cute_flash.FLASH_CAUSAL_HEAD_PAIR_KEY)
    assert (
        cute_flash.FlashAttentionConfig.__dataclass_fields__[
            KEY.removeprefix("cute_flash_")
        ].default
        is False
    )


@pytest.mark.parametrize(
    ("is_causal", "head_dim"),
    ((True, 64), (False, 128), (False, 64)),
    ids=("causal_hd64", "dense_hd128", "dense_hd64"),
)
def test_fa4_surfaces_search_the_knob(is_causal: bool, head_dim: int) -> None:
    fragment = _fragments(is_causal=is_causal, head_dim=head_dim)[KEY]
    assert tuple(fragment.choices) == (False, True)  # pyrefly: ignore[missing-attribute]
    assert _active(fragment) == (False, True)
    assert fragment.default() is False


@pytest.mark.parametrize(
    ("family", "is_causal"),
    (("fa4_deep_1cta", True), ("fa4_2cta", False), ("fa4_clc", False)),
)
def test_fixed_fa4_families_do_not_search_the_knob(
    family: str, is_causal: bool
) -> None:
    # The knob is never legal on the separate-ring / 2-CTA / CLC rings, and its
    # dependency witness (the plain fa4 family) must not reach a surface whose
    # family is fixed to one of them.
    fragment = _fragments(
        is_causal=is_causal, head_dim=64, pipeline_family_override=family
    )[KEY]
    assert _active(fragment) == (False,)
    assert fragment.default() is False


def test_asymmetric_head_dims_do_not_search_the_knob() -> None:
    # The asymmetric pair (K and V of different widths) runs two K/V rings; the
    # resolver clears the knob there, so its surface must not offer the value.
    for is_causal in (True, False):
        fragment = _fragments(is_causal=is_causal, head_dim=192, head_dim_v=128)[KEY]
        assert _active(fragment) == (False,), is_causal
        assert fragment.default() is False


def test_pruned_kv_intervals_clear_the_knob() -> None:
    # A pruned interval (sliding-window / prefix-LM / document masks) starts the
    # load warp at the work item's own last active tile, not the ring's: the
    # surface does not search the knob, the resolver clears a request, and the
    # emitter rejects a config built past the resolver.
    fragment = _fragments(is_causal=False, head_dim=64, **PRUNED_OPTIONS)[KEY]
    assert _active(fragment) == (False,)
    cfg = _resolve(
        {**CAUSAL_HD64_BODY, KEY: True},
        is_causal=False,
        head_dim=64,
        **PRUNED_OPTIONS,
    )
    assert cfg.pipeline_family == "fa4" and not cfg.persistent
    assert not cfg.early_prologue_loads
    assert cute_flash.flash_effective_config_values(cfg)[KEY] is False
    plain = _render(cfg, is_causal=False, head_dim=64, score_plan=WINDOW_PLAN)
    assert "defer_sync" not in plain
    assert "tKgK[None, flash_kv_last]" in _load_warp_prologue(plain)
    with pytest.raises(AssertionError, match="pre-sync prologue loads"):
        _render(
            dataclasses.replace(cfg, early_prologue_loads=True),
            is_causal=False,
            head_dim=64,
            score_plan=WINDOW_PLAN,
        )


def test_resolver_keeps_the_knob_on_the_plain_fa4_bodies() -> None:
    assert not _resolve(
        CAUSAL_HD64_BODY, is_causal=True, head_dim=64
    ).early_prologue_loads
    assert _resolve(
        {**CAUSAL_HD64_BODY, KEY: True}, is_causal=True, head_dim=64
    ).early_prologue_loads
    dense = _resolve({**DENSE_HD128_BODY, KEY: True}, is_causal=False, head_dim=128)
    assert dense.early_prologue_loads
    assert dense.first_load_order == 4 and not dense.persistent


@pytest.mark.parametrize(
    ("override", "is_causal", "head_dim"),
    (
        ({"cute_flash_persistent": True}, False, 128),
        ({"cute_flash_pipeline_family": "fa4_2cta"}, False, 128),
        ({"cute_flash_pipeline_family": "fa4_deep_1cta"}, True, 64),
        ({"cute_flash_pipeline_family": "fa4_alt"}, False, 128),
        ({"cute_flash_pipeline_family": "ws_overlap"}, True, 64),
        (
            {"cute_flash_first_load_order": cute_flash.FLASH_FIRST_LOAD_ORDER_STAGED},
            True,
            64,
        ),
    ),
    ids=(
        "persistent",
        "two_cta",
        "separate_rings",
        "alt",
        "ws_overlap",
        "staged_order",
    ),
)
def test_resolver_clears_the_knob_off_the_plain_one_cta_ring(
    override: dict[str, object], is_causal: bool, head_dim: int
) -> None:
    base = CAUSAL_HD64_BODY if is_causal else DENSE_HD128_BODY
    cfg = _resolve(
        {**base, KEY: True, **override}, is_causal=is_causal, head_dim=head_dim
    )
    assert not cfg.early_prologue_loads
    assert cute_flash.flash_effective_config_values(cfg)[KEY] is False


# --------------------------------------------------------------------------
# generated code
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "is_causal", "head_dim", "order"),
    (
        (CAUSAL_HD64_BODY, True, 64, ("q", "q", "k", "v")),
        (DENSE_HD128_BODY, False, 128, ("k", "q", "q", "v")),
    ),
    ids=("causal_hd64_order2", "dense_hd128_order4"),
)
def test_codegen_issues_the_first_loads_before_the_prologue_sync(
    base: dict[str, object], is_causal: bool, head_dim: int, order: tuple[str, ...]
) -> None:
    cfg = _resolve({**base, KEY: True}, is_causal=is_causal, head_dim=head_dim)
    code = _render(cfg, is_causal=is_causal, head_dim=head_dim)
    # Both pipelines defer their sync to the one prologue fence + sync.
    assert code.count("defer_sync=True") == 2
    fence = code.index(FENCE)
    sync = code.index(SYNC)
    assert code.index("PipelineTmaUmma.create(") < fence < sync
    # Warp 0's copies sit between the fence and the sync, against the full
    # barriers (stage 0/1 of each pipeline), in the body's first-load order.
    early = code.index(EARLY_GUARD, fence)
    assert fence < early < sync
    block = code[early:sync]
    copies = [line for line in block.splitlines() if "cute.copy(" in line]
    assert [line.split("_flash_tma_")[1][0] for line in copies] == list(order)
    assert "tQsQ[None, 0], tma_bar_ptr=storage.q_mbar_ptr.data_ptr() + 0" in block
    assert "tQsQ[None, 1], tma_bar_ptr=storage.q_mbar_ptr.data_ptr() + 1" in block
    assert "tKsK[None, 0], tma_bar_ptr=storage.kv_mbar_ptr.data_ptr() + 0" in block
    assert "tVsV[None, 1], tma_bar_ptr=storage.kv_mbar_ptr.data_ptr() + 1" in block
    assert "acquire_and_advance" not in block
    # The load warp only arms the barriers before its K/V loop.
    prologue = _load_warp_prologue(code)
    assert prologue.count("acquire_and_advance()") == 4
    assert "cute.copy(" not in prologue


@pytest.mark.parametrize(
    ("base", "is_causal", "head_dim"),
    ((CAUSAL_HD64_BODY, True, 64), (DENSE_HD128_BODY, False, 128)),
    ids=("causal_hd64", "dense_hd128"),
)
def test_codegen_without_the_knob_is_unchanged(
    base: dict[str, object], is_causal: bool, head_dim: int
) -> None:
    plain = _render(
        _resolve(base, is_causal=is_causal, head_dim=head_dim),
        is_causal=is_causal,
        head_dim=head_dim,
    )
    off = _render(
        _resolve({**base, KEY: False}, is_causal=is_causal, head_dim=head_dim),
        is_causal=is_causal,
        head_dim=head_dim,
    )
    assert plain == off
    assert "defer_sync" not in plain
    assert code_has_late_pipelines(plain)
    assert _load_warp_prologue(plain).count("cute.copy(") == 4


def code_has_late_pipelines(code: str) -> bool:
    return code.index(SYNC) < code.index("PipelineTmaUmma.create(")


def test_emitter_rejects_the_knob_outside_the_resolver_legality() -> None:
    cfg = _resolve(
        {**DENSE_HD128_BODY, "cute_flash_persistent": True},
        is_causal=False,
        head_dim=128,
    )
    assert cfg.persistent and not cfg.early_prologue_loads
    with pytest.raises(AssertionError, match="pre-sync prologue loads"):
        _render(
            dataclasses.replace(cfg, early_prologue_loads=True),
            is_causal=False,
            head_dim=128,
        )


def test_head_pair_reads_each_stage_head_in_the_early_block() -> None:
    # The head pair's two Q stages are two heads at one tile: the pre-sync
    # copies read each stage's tile from its own head's Q view, as the load
    # warp's (dropped) copies did; K/V keep the shared K/V head's views.
    cfg = _resolve(
        {**CAUSAL_HD64_BODY, HEAD_PAIR_KEY: True, KEY: True},
        is_causal=True,
        head_dim=64,
        kv_group=4,
    )
    assert cfg.causal_head_pair and cfg.early_prologue_loads
    code = _render(cfg, is_causal=True, head_dim=64, kv_group=4)
    assert "flash_bh1 = flash_bh + 1" in code
    assert "tQgQ[" not in code
    block = _early_block(code)
    assert [line.split("_flash_tma_")[1][0] for line in _copies(block)] == [
        "q",
        "q",
        "k",
        "v",
    ]
    assert (
        "cute.copy(_flash_tma_q, tQgQ0[None, flash_q_mma_tile0], tQsQ[None, 0], "
        "tma_bar_ptr=storage.q_mbar_ptr.data_ptr() + 0)" in block
    )
    assert (
        "cute.copy(_flash_tma_q, tQgQ1[None, flash_q_mma_tile1], tQsQ[None, 1], "
        "tma_bar_ptr=storage.q_mbar_ptr.data_ptr() + 1)" in block
    )
    assert "tKgK[None, flash_num_active_kv - cutlass.Int32(1)], tKsK[None, 0]" in block
    assert "tVgV[None, flash_num_active_kv - cutlass.Int32(1)], tVsV[None, 1]" in block
    prologue = _load_warp_prologue(code)
    assert prologue.count("acquire_and_advance()") == 4
    assert "cute.copy(" not in prologue
    # The one-head body at the same group keeps the one-head view.
    one_head = _render(
        _resolve(
            {**CAUSAL_HD64_BODY, HEAD_PAIR_KEY: False, KEY: True},
            is_causal=True,
            head_dim=64,
            kv_group=4,
        ),
        is_causal=True,
        head_dim=64,
        kv_group=4,
    )
    assert "tQgQ[None, flash_q_mma_tile0], tQsQ[None, 0]" in _early_block(one_head)
    assert "tQgQ0" not in one_head


def test_stagger_keeps_the_one_head_early_block() -> None:
    # The staggered walk changes only the MMA side (stage 0 starts at its own
    # diagonal); the load warp's first four loads, and so the pre-sync block,
    # are the plain descending body's.
    body = {**CAUSAL_HD64_BODY, "cute_flash_kv_stage": 4, KEY: True}
    staggered = _resolve({**body, STAGGER_KEY: True}, is_causal=True, head_dim=64)
    plain = _resolve({**body, STAGGER_KEY: False}, is_causal=True, head_dim=64)
    assert staggered.causal_stagger and staggered.early_prologue_loads
    assert not plain.causal_stagger and plain.early_prologue_loads
    code = _render(staggered, is_causal=True, head_dim=64)
    assert "flash_num_active_kv0 = flash_num_active_kv - cutlass.Int32(1)" in code
    block = _early_block(code)
    assert block == _early_block(_render(plain, is_causal=True, head_dim=64))
    assert [line.split("_flash_tma_")[1][0] for line in _copies(block)] == [
        "q",
        "q",
        "k",
        "v",
    ]
    assert "tQgQ[None, flash_q_mma_tile0], tQsQ[None, 0]" in block
    assert "tKgK[None, flash_num_active_kv - cutlass.Int32(1)], tKsK[None, 0]" in block
    prologue = _load_warp_prologue(code)
    assert prologue.count("acquire_and_advance()") == 4
    assert "cute.copy(" not in prologue


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True)
def _causal_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
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
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def _dense_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q * qk_scale, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def _run(
    kernel: object, config: dict[str, object], *args: torch.Tensor
) -> torch.Tensor:
    bound = helion.kernel(
        kernel.fn,  # pyrefly: ignore[missing-attribute]
        backend="cute",
        static_shapes=True,
        config=helion.Config(**config),
    )
    out = bound(*args)
    torch.cuda.synchronize()
    return out


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("kernel", "base", "shape", "is_causal"),
    (
        (_causal_attention, CAUSAL_HD64_BODY, (1, 4, 1024, 64), True),
        (_dense_attention, DENSE_HD128_BODY, (1, 2, 1024, 128), False),
    ),
    ids=("causal_hd64_order2", "dense_hd128_order4"),
)
def test_early_prologue_loads_are_bitwise_identical_on_device(
    kernel: object, base: dict[str, object], shape: tuple[int, ...], is_causal: bool
) -> None:
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal
    )
    plain = _run(kernel, base, q, k, v)
    early = _run(kernel, {**base, KEY: True}, q, k, v)
    torch.testing.assert_close(plain, reference, atol=2e-2, rtol=2e-2)
    assert torch.equal(early, plain)
