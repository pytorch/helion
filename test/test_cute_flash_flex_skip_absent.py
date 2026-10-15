"""``cute_flash_flex_skip_absent``: on the FlexAttention list walk the fa4 MMA
warp issues a query stage's QK and PV only on the steps where the stage's
BlockMask row has the tile. Each CTA's two 128-row stages walk the union of
their rows' block lists; on a tile its own row lacks a stage is *absent*: its
softmax takes the zero-P path (a zero P adds nothing to O, so the kernel stays
bitwise the unskipped one) and, with the knob on, the MMA warp skips the dead
QK and PV through a warp-uniform issue predicate inside the PTX gemm streams,
keeping every barrier arrival of the dense protocol.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import os
import re
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import CAUSAL_MASK_KIND
from helion._compiler.cute.attention_plan import FLEX_SCORE_KIND
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.attention_plan import FlexKvWalk
from helion._compiler.cute.attention_plan import dense_score_plan

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY
ENV = "HELION_CUTE_FLASH_FLEX_SKIP_ABSENT"
HEAD_DIM = 64
NUM_BH = 64
NUM_HEADS = 4
QK_HELPER = "_helion_flash_ptx.gemm_ptx_precomputed_qk("
PV_HELPER = "_helion_flash_ptx.gemm_ptx_precomputed_pv_ts("
PRESENT_MASK = re.compile(
    r"^flash_flex_n[01]_\d+ = flash_flex_p[01]_\d+ \| flash_flex_f[01]_\d+\n",
    re.MULTILINE,
)

# The pruned-fa4 campaign structure the flex seeds carry.
BODY: dict[str, object] = {
    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
    cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
    cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/8",
    cute_flash.FLASH_S_LOAD_REP_KEY: 16,
    cute_flash.FLASH_P_STORE_REP_KEY: 16,
    cute_flash.FLASH_DISC_PIPE_KEY: 2,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    cute_flash.FLASH_EPI_TMA_KEY: True,
}


def _plan(*, q_seq: int = 2048, kv_seq: int = 2048) -> AttentionScorePlan:
    """A flex plan without regions (every listed tile is dense)."""
    return AttentionScorePlan(
        head_dim=HEAD_DIM,
        qk_scale_log2=math.log2(math.e) / math.sqrt(HEAD_DIM),
        q_seq=q_seq,
        kv_seq=kv_seq,
        kv_walk=FlexKvWalk(
            part_cnt_name="flex_part_cnt",
            part_idx_name="flex_part_idx",
            full_cnt_name="flex_full_cnt",
            full_idx_name="flex_full_idx",
            q_block=128,
            kv_block=128,
            table_batch=NUM_BH // NUM_HEADS,
            table_heads=NUM_HEADS,
        ),
    )


def _resolve(
    config: dict[str, object] | None,
    *,
    has_list_walk: bool = True,
    num_kv: int = 16,
    environ: dict[str, str] | None = None,
    **kwargs: object,
) -> cute_flash.FlashAttentionConfig:
    options: dict[str, object] = {
        "standard_dense_output": not has_list_walk,
        "standard_causal_output": False,
        "score_modifier_kinds": (FLEX_SCORE_KIND,) if has_list_walk else (),
        **kwargs,
    }
    with patch.dict(os.environ, environ or {}, clear=True):
        return cute_flash.resolve_flash_config(
            HEAD_DIM,
            num_kv,
            config,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=has_list_walk,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _fragments(
    *,
    has_list_walk: bool = True,
    environ: dict[str, str] | None = None,
    **kwargs: object,
) -> dict[str, ConfigSpecFragment]:
    options: dict[str, object] = {
        "standard_dense_output": not has_list_walk and not kwargs.get("is_causal"),
        "score_modifier_kinds": (FLEX_SCORE_KIND,) if has_list_walk else (),
        **kwargs,
    }
    with patch.dict(os.environ, environ or {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=has_list_walk,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


def _render(
    cfg: cute_flash.FlashAttentionConfig, plan: AttentionScorePlan | None = None
) -> str:
    plan = _plan() if plan is None else plan
    q_seq = plan.q_seq if plan.q_seq is not None else 2048
    kv_seq = plan.kv_seq if plan.kv_seq is not None else 2048
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=HEAD_DIM,
        num_kv=-(-kv_seq // 128),
        sequence_extent=kv_seq,
        num_bh=NUM_BH,
        total_tiles=NUM_BH * (-(-q_seq // 256)),
        cfg=cfg,
        has_lse=True,
        io_dtype="cutlass.BFloat16",
        score_plan=plan,
        num_heads=NUM_HEADS if plan.is_flex else None,
        flex_item_table="flex_items" if plan.is_flex else None,
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _mma_role_block(code: str) -> tuple[int, int]:
    """Return the span of the MMA warp's role block (``if warp_idx == N:``)."""
    alloc = code.index("flash_tmem.allocate(512)")
    head = code.rindex("\nif warp_idx", 0, alloc)
    following = code.find("\nif ", alloc)
    return head, len(code) if following < 0 else following


def test_resolver_defaults_the_knob_off_and_takes_it_on_list_walks_only() -> None:
    assert _resolve(BODY).flex_skip_absent is False
    assert _resolve(None).flex_skip_absent is False
    assert _resolve({**BODY, KEY: True}).flex_skip_absent is True
    assert _resolve({**BODY, KEY: False}).flex_skip_absent is False
    # The env switch moves the default only; an explicit key wins.
    assert _resolve(BODY, environ={ENV: "1"}).flex_skip_absent is True
    assert _resolve({**BODY, KEY: False}, environ={ENV: "1"}).flex_skip_absent is False
    assert _resolve({**BODY, KEY: True}, environ={ENV: "0"}).flex_skip_absent is True
    # The list walk is pinned to the fa4 pair body: a request for another
    # family resolves to fa4 and keeps the knob.
    other = _resolve(
        {**BODY, KEY: True, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
    )
    assert other.pipeline_family == "fa4" and other.flex_skip_absent is True
    # No per-stage status exists outside the list walk: dense, causal, pruned
    # and modifier plans pin the knob off whatever the request.
    assert _resolve({**BODY, KEY: True}, has_list_walk=False).flex_skip_absent is False
    assert (
        _resolve(
            {**BODY, KEY: True},
            has_list_walk=False,
            is_causal=True,
            standard_dense_output=False,
            standard_causal_output=True,
        ).flex_skip_absent
        is False
    )
    assert (
        _resolve(
            {**BODY, KEY: True},
            has_list_walk=False,
            has_kv_tile_pruning=True,
            standard_dense_output=False,
        ).flex_skip_absent
        is False
    )
    assert (
        _resolve(
            {**BODY, KEY: True},
            has_list_walk=False,
            has_score_modifiers=True,
            score_modifier_kinds=(CAUSAL_MASK_KIND,),
            standard_dense_output=False,
        ).flex_skip_absent
        is False
    )
    # The skip is an issue predicate inside the PTX gemm streams.
    assert (
        _resolve(
            {**BODY, KEY: True}, environ={"HELION_CUTE_FLASH_MMA_PTX": "0"}
        ).flex_skip_absent
        is False
    )


def test_knob_is_an_appended_schema_axis_with_its_fa4_witness() -> None:
    # Appended behind the axes it was introduced after (an inserted axis
    # would reshuffle every later axis's coverage rows).
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(cute_flash.FLASH_EARLY_PROLOGUE_LOADS_KEY)
    witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, True)]
    assert witness == {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    # The witness context is sufficient on the list walk: it resolves to the
    # knob's value, and the projection round-trips both values.
    assert _resolve({**witness, KEY: True}).flex_skip_absent is True
    for value in (False, True):
        resolved = _resolve({**BODY, KEY: value})
        assert cute_flash.flash_effective_config_values(resolved)[KEY] is value
    assert KEY in cute_flash.flash_effective_config_values(
        _resolve(BODY, has_list_walk=False)
    )


def test_fragments_search_the_knob_on_list_walk_plans_only() -> None:
    flex = _fragments()[KEY]
    assert set(_active(flex)) == {False, True}
    assert flex.default() is False
    dense = _fragments(has_list_walk=False)[KEY]
    assert _active(dense) == (False,)
    assert dense.default() is False
    assert _active(
        _fragments(
            has_list_walk=False,
            is_causal=True,
            standard_causal_output=True,
            score_modifier_kinds=(CAUSAL_MASK_KIND,),
        )[KEY]
    ) == (False,)
    assert _active(_fragments(has_list_walk=False, has_kv_tile_pruning=True)[KEY]) == (
        False,
    )


@pytest.mark.parametrize("env", (None, "0", "1"))
def test_fragment_default_is_the_list_walk_resolver_default(env: str | None) -> None:
    environ = {} if env is None else {ENV: env}
    fragment = _fragments(environ=environ)[KEY]
    # The fragment default is what a config without the key resolves to on
    # the walk's body (the value dependency's witness context): the env
    # switch reaches the seeded and coverage configs through it.
    pinned = _resolve({cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}, environ=environ)
    assert fragment.default() is pinned.flex_skip_absent
    assert fragment.default() is (env == "1")
    # The env moves the default only; the surface keeps measuring both.
    assert set(_active(fragment)) == {False, True}
    dense = _fragments(has_list_walk=False, environ=environ)[KEY]
    assert dense.default() is False and _active(dense) == (False,)


@pytest.mark.parametrize("num_kv", (16, 17, 64))
def test_resolved_value_is_length_invariant(num_kv: int) -> None:
    on = _resolve({**BODY, KEY: True}, num_kv=num_kv)
    assert on.flex_skip_absent is True
    assert cute_flash.flash_effective_config_values(on) == (
        cute_flash.flash_effective_config_values(_resolve({**BODY, KEY: True}))
    )


@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("static_desc", [True, False])
def test_render_differs_only_in_the_present_masks_and_the_mma_warp(
    interleave: bool, static_desc: bool
) -> None:
    body = {
        **BODY,
        cute_flash.FLASH_MMA_INTERLEAVE_KEY: interleave,
        cute_flash.FLASH_PRECOMPUTE_QK_DESC_KEY: static_desc,
    }
    off = _render(_resolve({**body, KEY: False}))
    on = _render(_resolve({**body, KEY: True}))
    assert on == _render(_resolve(body, environ={ENV: "1"}))
    assert off == _render(_resolve(body))
    qk_helper = (
        "_helion_flash_ptx.gemm_ptx_precomputed_qk_static("
        if static_desc
        else QK_HELPER
    )
    # Off: no per-stage present masks, unpredicated streams.
    assert "flash_flex_n0_0" not in off
    assert "issue=" not in off
    assert off.count(qk_helper) == 4
    assert off.count(PV_HELPER) == 4
    # On: the CTA-wide setup adds one "sees the tile" mask per stage; the
    # MMA warp predicates its two QKs and two PVs per stage, never skips a
    # stage's first PV (the accumulator's zero-init) and keeps every commit
    # and arrival of the dense protocol.
    assert len(PRESENT_MASK.findall(on)) == 2
    assert on.count(qk_helper) == 4
    assert on.count(PV_HELPER) == 4
    for stage in ("0", "1"):
        assert on.count(f"issue=flash_flex_qk_live{stage})") == 2
        assert on.count(f"issue=flash_flex_pv_issue{stage})") == 2
        assert (
            on.count(
                f"flash_flex_pv_issue{stage} = flash_flex_pv_live{stage} | "
                f"cutlass.Int32(flash_o_zero{stage})"
            )
            == 2
        )
    for arrival in (
        "cute_tcgen05_flash.commit(flash_s_full_ptr + 0)",
        "cute_tcgen05_flash.commit(flash_s_full_ptr + 1)",
        "cute_tcgen05_flash.commit(flash_o_full_ptr + ",
        "mbar_spin_wait(flash_pfor_ptr + ",
        "flash_pfor_phase ^= 1",
    ):
        assert on.count(arrival) == off.count(arrival), arrival
    # The softmax, correction, load and epilogue roles are untouched: the
    # text before the MMA role block equals the off render's plus the two
    # mask lines, and the text from the next role guard on is identical.
    head_off, next_off = _mma_role_block(off)
    head_on, next_on = _mma_role_block(on)
    assert PRESENT_MASK.sub("", on[:head_on]) == off[:head_off]
    assert on[next_on:] == off[next_off:]
    assert on[head_on:next_on] != off[head_off:next_off]
    for helper in (
        "fa4_disc_zero_store(",
        "rescale_o_tmem(",
        "flash_flex_status = ",
    ):
        assert on.count(helper) == off.count(helper), helper


def test_emitter_rejects_the_knob_without_the_ptx_mma_warp() -> None:
    on = _resolve({**BODY, KEY: True})
    assert on.mma_ptx and on.flex_skip_absent
    with pytest.raises(AssertionError):
        _render(dataclasses.replace(on, mma_ptx=False))


def test_dense_plans_render_without_the_skip() -> None:
    off = _resolve({**BODY, KEY: False}, has_list_walk=False)
    on = _resolve({**BODY, KEY: True}, has_list_walk=False)
    assert on == off
    dense = dense_score_plan(HEAD_DIM)
    assert _render(on, dense) == _render(off, dense)
    assert "issue=" not in _render(on, dense)
    # Forcing the field past the resolver on a dense plan is a no-op too: the
    # emitter keys the skip on the plan's list walk.
    forced = dataclasses.replace(off, flex_skip_absent=True)
    assert _render(forced, dense) == _render(off, dense)
