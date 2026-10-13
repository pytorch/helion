"""The FlexAttention envelope's BlockMask list walk on the CuTe fa4 pair body.

CPU render tests (no GPU, no kernel bind): the walk is rendered through the
direct emitter call with a hand-built ``AttentionScorePlan`` whose
``kv_walk`` is a ``FlexKvWalk``; the resolver's structural pins for the plan
category are checked through ``resolve_flash_config`` with
``has_list_walk=True``.
"""

from __future__ import annotations

import ast
from collections import Counter
import math
import os
import re
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import FLEX_SCORE_KIND
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.attention_plan import FlexKvWalk
from helion._compiler.cute.attention_plan import dense_score_plan
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.cute.flash_score_mod import FlexHookCode
    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


def _stub_render_flex_hook(
    plan: AttentionScorePlan,
    *,
    stage: int,
    score_tensor: str,
    coord_tensor: str,
    m_tile_expr: str,
    kv_tile_expr: str,
    chunk_expr: str,
    chunk_cols: int,
    indent: int,
    with_score: bool,
    with_mask: bool,
    tail_mask: bool,
    **_: object,
) -> cute_flash.FlexHookCode:
    """A marker hook: these tests check the emitter's slots and walk, not the
    renderer (whose own tests use real regions); the plans here carry
    stand-in regions. The marker is a bare string literal (it survives
    ``ast.unparse``; a comment would not) plus the KV tail mask."""
    pad = " " * indent
    marker = (
        f"flex hook stage={stage} score={with_score} mask={with_mask} "
        f"tail={tail_mask} m={m_tile_expr} kv={kv_tile_expr} "
        f"chunk={chunk_expr}x{chunk_cols} coord={coord_tensor}"
    )
    lines = [f"{pad}{marker!r}\n"]
    if tail_mask:
        lines.append(
            f"{pad}_helion_flash_rt.mask_r2p_sm100_rank1({score_tensor}, "
            f"_flash_kv_tail_cols - {chunk_expr} * {chunk_cols})\n"
        )
    return cute_flash.FlexHookCode(
        item_prelude="", tile_prologue="", chunk_body="".join(lines)
    )


@pytest.fixture(autouse=True)
def _stub_hook() -> Iterator[None]:
    with patch.object(cute_flash, "render_flex_hook", _stub_render_flex_hook):
        yield


HEAD_DIM = 64
NUM_BH = 64
NUM_HEADS = 4
# The pruned-fa4 campaign structure (report 06 section 2.2): 8x2 packets at
# the 16/8 cadence, the chunked body with a two-deep PASS2 pipe, split P
# arrivals, the descending walk and the TMA output epilogue.
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
    # The dense comparison body: the pins the resolver applies to the list
    # walk, requested explicitly for the dense plan.
    cute_flash.FLASH_PERSISTENT_KEY: False,
    cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2",
    cute_flash.FLASH_KV_ORDER_KEY: "descending",
}
RT_CALL = re.compile(r"_helion_flash_rt\.([A-Za-z0-9_]+)\(")


def _walk(
    *, q_block: int = 128, table_batch: int = 16, table_heads: int = NUM_HEADS
) -> FlexKvWalk:
    return FlexKvWalk(
        part_cnt_name="flex_part_cnt",
        part_idx_name="flex_part_idx",
        full_cnt_name="flex_full_cnt",
        full_idx_name="flex_full_idx",
        q_block=q_block,
        kv_block=128,
        table_batch=table_batch,
        table_heads=table_heads,
    )


def _flex_plan(
    *,
    q_seq: int = 2048,
    kv_seq: int = 2048,
    q_block: int = 128,
    mask_region: object = None,
    score_region: object = None,
    scale_peeled: bool = True,
) -> AttentionScorePlan:
    """A flex plan; the regions are stand-ins (the stub hook renders a marker
    and the tail mask only), so any non-None object marks a present region."""
    return AttentionScorePlan(
        head_dim=HEAD_DIM,
        qk_scale_log2=math.log2(math.e) / math.sqrt(HEAD_DIM),
        q_seq=q_seq,
        kv_seq=kv_seq,
        kv_walk=_walk(q_block=q_block),
        mask_region=mask_region,  # pyrefly: ignore[bad-argument-type]
        score_region=score_region,  # pyrefly: ignore[bad-argument-type]
        scale_peeled=scale_peeled,
    )


def _resolve(
    config: dict[str, object] | None,
    *,
    has_list_walk: bool = True,
    num_kv: int = 16,
    **kwargs: object,
) -> cute_flash.FlashAttentionConfig:
    options: dict[str, object] = {
        "standard_dense_output": not has_list_walk,
        "standard_causal_output": False,
        **kwargs,
    }
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.resolve_flash_config(
            HEAD_DIM,
            num_kv,
            config,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=has_list_walk,
            score_modifier_kinds=(FLEX_SCORE_KIND,) if has_list_walk else (),
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _render(
    cfg: cute_flash.FlashAttentionConfig,
    plan: AttentionScorePlan,
    *,
    has_lse: bool = True,
) -> str:
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
        has_lse=has_lse,
        io_dtype="cutlass.BFloat16",
        score_plan=plan,
        num_heads=NUM_HEADS if plan.is_flex else None,
        flex_item_table="flex_items" if plan.is_flex else None,
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _roles(src: str) -> dict[str, str]:
    """Split a render into its warp-role blocks (keyed by the guard line)."""
    roles: dict[str, list[str]] = {"setup": []}
    current = "setup"
    for line in src.splitlines():
        if re.match(r"^(el)?if .*warp_idx.*:$", line):
            current = line.strip()
            roles[current] = []
        roles[current].append(line)
    return {name: "\n".join(lines) for name, lines in roles.items()}


def _rt_calls(block: str) -> Counter[str]:
    return Counter(RT_CALL.findall(block))


def _softmax_loop(role: str) -> str:
    """The lines of the online-softmax KV loop of one softmax role block."""
    lines = role.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*for flash_kv(_iter)? in cutlass\.range\(", line)
    ]
    assert len(starts) == 1, starts
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent
    ):
        end += 1
    return "\n".join(lines[start:end])


# --------------------------------------------------------------------------
# Resolver pins
# --------------------------------------------------------------------------


def test_resolver_pins_the_list_walk_to_the_plain_fa4_pair_body() -> None:
    for num_kv in (16, 17, 1):
        cfg = _resolve(BODY, num_kv=num_kv)
        assert cfg.pipeline_family == "fa4"
        assert cfg.topology == "fa4"
        assert not cfg.persistent
        assert not cfg.use_2cta_instrs
        assert not cfg.use_cga2_local_cta
        assert not cfg.use_clc_scheduler
        assert cfg.softmax_disc
        assert cfg.stat_transport == "ring2"
        assert cfg.kv_tile_n == 128
        assert not cfg.early_prologue_loads
        assert not cfg.causal_loop_split
        assert not cfg.causal_stagger
        assert not cfg.causal_skip_dead_tile
        assert not cfg.causal_head_pair
        assert not cfg.alternating_warpgroups
        assert not cfg.tensor_4d_tma
        assert cfg.epi_tma or cfg.epi_stg
        assert cfg.masked_e2e_schedule == "inherit"
        assert cfg.kv_order == "ascending"


def test_resolver_defaults_the_list_walk_to_the_descending_order() -> None:
    assert _resolve(None).causal_kv_order == "descending"
    assert (
        _resolve({cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending"}).causal_kv_order
        == "ascending"
    )


@pytest.mark.parametrize(
    "request_",
    [
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc"},
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_alt"},
        {cute_flash.FLASH_USE_2CTA_KEY: True},
        {cute_flash.FLASH_CGA2_LOCAL_KEY: True},
        {cute_flash.FLASH_CLC_KEY: True},
        {cute_flash.FLASH_EARLY_PROLOGUE_LOADS_KEY: True},
        {cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True},
        {cute_flash.FLASH_KV_TILE_N_KEY: 256},
        {cute_flash.FLASH_SOFTMAX_DISC_KEY: False},
        {cute_flash.FLASH_STAT_TRANSPORT_KEY: "single"},
        {cute_flash.FLASH_TENSOR_4D_TMA_KEY: True},
        {cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "xu"},
    ],
    ids=lambda request_: "+".join(f"{key}={value}" for key, value in request_.items()),
)
def test_explicit_requests_outside_the_pins_resolve_away(
    request_: dict[str, object],
) -> None:
    pinned = cute_flash.flash_effective_config_values(_resolve({**BODY, **request_}))
    assert pinned == cute_flash.flash_effective_config_values(_resolve(BODY))


def test_direct_store_request_resolves_to_the_staged_epilogue() -> None:
    # The query length need not be a multiple of 256 (and the config cannot
    # depend on it), so the row-guarded staged store is pinned: a request
    # for the direct correction-warp store takes the vector store.
    cfg = _resolve(
        {
            **BODY,
            cute_flash.FLASH_EPI_TMA_KEY: False,
            cute_flash.FLASH_EPI_STG_KEY: False,
        }
    )
    assert not cfg.epi_tma and cfg.epi_stg
    assert _resolve({**BODY, cute_flash.FLASH_EPI_TMA_KEY: False}).epi_stg
    assert _resolve(BODY).epi_tma


def test_fragments_offer_only_the_fa4_family_and_pinned_structure() -> None:
    with patch.dict(os.environ, {}, clear=True):
        fragments = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
        )

    def active(key: str) -> tuple[object, ...]:
        fragment = fragments[key]
        choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
        search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
        return choices if search is None else tuple(search)

    assert active(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == ("fa4",)
    # v1.5: static strided persistence is searched for list walks (default off).
    assert set(active(cute_flash.FLASH_PERSISTENT_KEY)) == {False, True}
    # v1.5: the MMA warp's absent-stage skip is searched (default off).
    assert set(active(cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY)) == {False, True}
    assert fragments[cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY].default() is False
    # The split-KV walk is searched over every split (default 1 without a
    # device to seed it from).
    assert active(cute_flash.FLASH_KV_SPLIT_KEY) == (1, 2, 4)
    assert fragments[cute_flash.FLASH_KV_SPLIT_KEY].default() == 1
    assert active(cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY) == (False,)
    assert active(cute_flash.FLASH_EARLY_PROLOGUE_LOADS_KEY) == (False,)
    assert active(cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY) == ("inherit",)
    assert active(cute_flash.FLASH_KV_TILE_N_KEY) == (128,)
    assert active(cute_flash.FLASH_SOFTMAX_DISC_KEY) == (True,)
    assert active(cute_flash.FLASH_KV_ORDER_KEY) == ("ascending",)
    assert set(active(cute_flash.FLASH_CAUSAL_KV_ORDER_KEY)) == {
        "ascending",
        "descending",
    }
    assert fragments[cute_flash.FLASH_CAUSAL_KV_ORDER_KEY].default() == "descending"


def test_flex_mask_form_knob_is_searched_only_with_a_mask_region() -> None:
    """``cute_flash_flex_mask_form`` (the keep word vs. the per-element
    predicate): an appended schema axis, default ``"bitmask"``, searched on
    flex plans with a mask region, pinned to the default elsewhere, and
    resolved away on non-flex plans."""
    key = cute_flash.FLASH_FLEX_MASK_FORM_KEY
    # Appended behind the axes it was introduced after (an inserted axis
    # would reshuffle every later axis's coverage rows).
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(key) == 1
    assert keys.index(key) > keys.index(cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY)
    assert (
        cute_flash.FlashAttentionConfig.__dataclass_fields__["flex_mask_form"].default
        == "bitmask"
    )
    assert (key, "element") in cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES

    def active(fragments: dict[str, object]) -> tuple[object, ...]:
        fragment = fragments[key]
        choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
        search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
        return choices if search is None else tuple(search)

    with patch.dict(os.environ, {}, clear=True):
        flex_mask = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            flex_has_mask_region=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
        )
        flex_score_only = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
        )
        dense = cute_flash.flash_autotune_fragments(
            HEAD_DIM, 16, dtype=torch.bfloat16, num_bh=NUM_BH
        )
    assert active(flex_mask) == ("bitmask", "element")
    assert active(flex_score_only) == ("bitmask",)
    assert active(dense) == ("bitmask",)
    assert flex_mask[key].default() == "bitmask"  # pyrefly: ignore[missing-attribute]
    assert _resolve({**BODY, key: "element"}).flex_mask_form == "element"
    assert _resolve(BODY).flex_mask_form == "bitmask"
    assert _resolve({key: "element"}, has_list_walk=False).flex_mask_form == "bitmask"
    with pytest.raises(InvalidConfig):
        _resolve({**BODY, key: "r2p"})


def test_flex_aux_smem_knob_is_searched_only_with_a_candidate_table() -> None:
    """``cute_flash_flex_aux_smem`` (the shared-memory aux staging budget):
    an appended schema axis, default 0, whose search choices resolve against
    the plan's candidate table sizes (pinned without candidates; the budgets
    that stage distinct table sets with them; a requested budget
    canonicalizes to the smallest one staging the same tables); resolved to
    0 off flex plans."""
    key = cute_flash.FLASH_FLEX_AUX_SMEM_KEY
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    # Appended behind the axes it was introduced after (later knobs follow).
    assert keys.count(key) == 1
    assert keys.index(key) > keys.index(cute_flash.FLASH_FLEX_MASK_FORM_KEY)
    assert (
        cute_flash.FlashAttentionConfig.__dataclass_fields__["flex_aux_smem"].default
        == 0
    )
    for value in (4096, 16384):
        assert (key, value) in cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES

    def active(fragments: dict[str, object]) -> tuple[object, ...]:
        fragment = fragments[key]
        choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
        search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
        return choices if search is None else tuple(search)

    def fragments(candidates: tuple[int, ...]) -> dict[str, object]:
        with patch.dict(os.environ, {}, clear=True):
            return cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                16,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=True,
                flex_aux_smem_candidates=candidates,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
            )

    # One small table: 4096 and 16384 stage the same set, so one of them.
    small = fragments((1552,))
    assert active(small) == (0, 4096)
    assert small[key].default() == 0  # pyrefly: ignore[missing-attribute]
    # Two tables where only the larger budget stages both.
    two = fragments((1552, 8192))
    assert set(active(two)) == {0, 4096, 16384}
    assert two[key].default() == 0  # pyrefly: ignore[missing-attribute]
    # A first table past 4096: staged by 16384 alone.
    big = fragments((4112,))
    assert set(active(big)) == {0, 16384}
    assert big[key].default() == 0  # pyrefly: ignore[missing-attribute]
    none = fragments(())
    assert active(none) == (0,)
    with patch.dict(os.environ, {}, clear=True):
        dense = cute_flash.flash_autotune_fragments(
            HEAD_DIM, 16, dtype=torch.bfloat16, num_bh=NUM_BH
        )
    assert active(dense) == (0,)

    candidates = (1552, 8192)
    assert _resolve(BODY, flex_aux_smem_candidates=candidates).flex_aux_smem == 0
    assert (
        _resolve({**BODY, key: 4096}, flex_aux_smem_candidates=candidates).flex_aux_smem
        == 4096
    )
    assert (
        _resolve(
            {**BODY, key: 16384}, flex_aux_smem_candidates=candidates
        ).flex_aux_smem
        == 16384
    )
    # Canonical: 16384 stages the single small table exactly as 4096 does.
    assert (
        _resolve({**BODY, key: 16384}, flex_aux_smem_candidates=(1552,)).flex_aux_smem
        == 4096
    )
    assert _resolve({**BODY, key: 16384}).flex_aux_smem == 0
    assert _resolve(BODY).flex_aux_smem == 0
    assert (
        _resolve(
            {key: 4096}, has_list_walk=False, flex_aux_smem_candidates=(1552,)
        ).flex_aux_smem
        == 4096
    )
    with pytest.raises(InvalidConfig):
        _resolve({**BODY, key: 1024}, flex_aux_smem_candidates=candidates)


def test_flex_prefetch_knob_is_searched_only_with_per_chunk_table_loads() -> None:
    """``cute_flash_flex_prefetch`` (the table prefetch register budget): an
    appended schema axis, default 64 canonicalized against the plan's
    per-chunk table loads (0 without them), search choices = the budgets that prefetch distinct
    load sets or depths at some chunk width, a requested budget canonical
    to the smallest one with the same loads and depth at the config's chunk
    width; resolved to 0 off flex plans."""
    key = cute_flash.FLASH_FLEX_PREFETCH_KEY
    assert cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS[-3] == key
    assert (
        cute_flash.FlashAttentionConfig.__dataclass_fields__["flex_prefetch"].default
        == 0
    )
    for value in (32, 64, 128):
        assert (key, value) in cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES

    def active(fragments: dict[str, object]) -> tuple[object, ...]:
        fragment = fragments[key]
        choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
        search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
        return choices if search is None else tuple(search)

    def fragments(regs: tuple[int, ...]) -> dict[str, object]:
        with patch.dict(os.environ, {}, clear=True):
            return cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                16,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=True,
                flex_prefetch_regs=regs,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
            )

    # One 32-register vector: 32 / 64 / 128 take it one / two / three chunks
    # ahead at 32 columns (two / four / seven at 16).
    one = fragments((32,))
    assert set(active(one)) == {0, 32, 64, 128}
    assert one[key].default() == 64  # pyrefly: ignore[missing-attribute]
    # The graphormer-edge pair (32 + 96): 32 takes the vector one ahead, 64
    # two ahead, 128 both one ahead.
    pair = fragments((32, 96))
    assert set(active(pair)) == {0, 32, 64, 128}
    assert pair[key].default() == 64  # pyrefly: ignore[missing-attribute]
    # A 34-register window: nothing fits 32 at 32 columns (18 at 16: one
    # ahead), 64 one ahead, 128 three ahead.
    window = fragments((34,))
    assert set(active(window)) == {0, 32, 64, 128}
    # One 8-register byte vector (a bool table over kv, global-sw /
    # jetspec-tree): 32 already reaches the depth cap at both chunk widths,
    # so 64 and 128 resolve to 32 and the default follows (a default of 64
    # would be a searched value no config can realize: the coverage design
    # rejected the global-sw search with "uncovered values [(prefetch, 64)]").
    byte = fragments((8,))
    assert set(active(byte)) == {0, 32}
    assert byte[key].default() == 32  # pyrefly: ignore[missing-attribute]
    assert _resolve(BODY, flex_prefetch_regs=(8,)).flex_prefetch == 32
    assert _resolve({**BODY, key: 64}, flex_prefetch_regs=(8,)).flex_prefetch == 32
    none = fragments(())
    assert active(none) == (0,)
    assert none[key].default() == 0  # pyrefly: ignore[missing-attribute]
    with patch.dict(os.environ, {}, clear=True):
        dense = cute_flash.flash_autotune_fragments(
            HEAD_DIM, 16, dtype=torch.bfloat16, num_bh=NUM_BH
        )
    assert active(dense) == (0,)

    regs = (32, 96)
    assert _resolve(BODY, flex_prefetch_regs=regs).flex_prefetch == 64
    assert _resolve({**BODY, key: 0}, flex_prefetch_regs=regs).flex_prefetch == 0
    # BODY has 16-column chunks (16 + 48 registers, 8 chunks): 64 takes both
    # one ahead, 128 both two ahead; at 32 columns 128 takes both one ahead.
    assert _resolve({**BODY, key: 128}, flex_prefetch_regs=regs).flex_prefetch == 128
    assert (
        _resolve(
            {**BODY, key: 128, cute_flash.FLASH_S_LOAD_REP_KEY: 32},
            flex_prefetch_regs=regs,
        ).flex_prefetch
        == 128
    )
    # A single 32-register vector at 32 columns (4 chunks): 64 is two ahead,
    # 128 three ahead (the lead is capped at chunks - 1).
    assert (
        _resolve(
            {**BODY, key: 64, cute_flash.FLASH_S_LOAD_REP_KEY: 32},
            flex_prefetch_regs=(32,),
        ).flex_prefetch
        == 64
    )
    assert _resolve({**BODY, key: 64}).flex_prefetch == 0
    assert _resolve(BODY).flex_prefetch == 0
    assert (
        _resolve({key: 64}, has_list_walk=False, flex_prefetch_regs=(34,)).flex_prefetch
        == 64
    )
    with pytest.raises(InvalidConfig):
        _resolve({**BODY, key: 48}, flex_prefetch_regs=regs)


def test_flex_coalesce_knob_is_searched_only_with_coalescible_runs() -> None:
    """``cute_flash_flex_coalesce`` (bytes per row of one warp-cooperative
    table request): an appended schema axis, default the widest width whose
    staging fits the surface's bodies (the stream on wherever it fits),
    search choices = the widths giving distinct per-run request widths at
    some chunk width whose staging fits, a requested width canonical to
    the smallest one with the same per-run widths at the config's chunk
    width (0 where its staging leaves no two-deep ring); resolved to 0 off
    flex plans."""
    key = cute_flash.FLASH_FLEX_COALESCE_KEY
    assert cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS[-2] == key
    assert (
        cute_flash.FlashAttentionConfig.__dataclass_fields__["flex_coalesce"].default
        == 0
    )
    for value in (32, 64, 128):
        assert (key, value) in cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES

    def active(fragments: dict[str, object]) -> tuple[object, ...]:
        fragment = fragments[key]
        choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
        search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
        return choices if search is None else tuple(search)

    def fragments(steps: tuple[int, ...]) -> dict[str, object]:
        with patch.dict(os.environ, {}, clear=True):
            return cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                16,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=True,
                flex_coalesce_steps=steps,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
            )

    # The graphormer-edge pair (4 and 12 bytes per column): 128-byte spans
    # at 32 columns would realize every width, but their 136 KB staging
    # fits no searched hd64 body, so 128 is not offered; at 16 columns the
    # spans cap both runs at 64 (74 KB, fits).
    pair = fragments((4, 12))
    assert set(active(pair)) == {0, 32, 64}
    assert pair[key].default() == 64  # pyrefly: ignore[missing-attribute]
    assert cute_flash.flash_flex_coalesce_choices((4, 12), 128) == (0,)
    assert cute_flash.flash_flex_coalesce_default((4, 12), 128) == 0
    # A bf16 unit-stride run: 64-byte spans at 32 columns (32 at 16), so
    # 128 is 64 everywhere.
    half = fragments((2,))
    assert set(active(half)) == {0, 32, 64}
    assert half[key].default() == 64  # pyrefly: ignore[missing-attribute]
    # A byte run: 32-byte spans at 32 columns only.
    byte = fragments((1,))
    assert set(active(byte)) == {0, 32}
    assert byte[key].default() == 32  # pyrefly: ignore[missing-attribute]
    # One int32 vector (graphormer-spatial): 36 KB at 32 columns fits.
    assert cute_flash.flash_flex_coalesce_choices((4,), 64) == (0, 32, 64, 128)
    assert cute_flash.flash_flex_coalesce_default((4,), 64) == 128
    none = fragments(())
    assert active(none) == (0,)
    assert none[key].default() == 0  # pyrefly: ignore[missing-attribute]
    with patch.dict(os.environ, {}, clear=True):
        dense = cute_flash.flash_autotune_fragments(
            HEAD_DIM, 16, dtype=torch.bfloat16, num_bh=NUM_BH
        )
    assert active(dense) == (0,)

    steps = (4, 12)
    # The default (64) at BODY's 16 columns: one slot.
    plain = _resolve(BODY, flex_coalesce_steps=steps)
    assert (plain.flex_coalesce, plain.flex_coalesce_slots) == (64, 1)
    assert (
        _resolve({**BODY, key: 0}, flex_coalesce_steps=steps).flex_coalesce_slots == 0
    )
    # BODY has 16-column chunks: 64-byte spans, so 128 resolves to 64.  The
    # pair's staging is 72 KB per ring slot there: one slot fits beside the
    # two Q tiles, the two staged O tiles and a three-deep ring, two do not.
    resolved = _resolve({**BODY, key: 128}, flex_coalesce_steps=steps)
    assert resolved.flex_coalesce == 64
    assert resolved.flex_coalesce_slots == 1
    assert resolved.kv_stage == 3
    # One int32 table (graphormer-spatial): 20 KB per slot at 16 columns
    # (four slots, 80 KB), 36 KB at 32 (two slots).
    assert (
        _resolve({**BODY, key: 128}, flex_coalesce_steps=(4,)).flex_coalesce_slots == 4
    )
    assert (
        _resolve(
            {**BODY, key: 128, cute_flash.FLASH_S_LOAD_REP_KEY: 32},
            flex_coalesce_steps=(4,),
        ).flex_coalesce_slots
        == 2
    )
    assert _resolve({**BODY, key: 128}, flex_coalesce_steps=steps).flex_coalesce == 64
    assert _resolve({**BODY, key: 32}, flex_coalesce_steps=steps).flex_coalesce == 32
    assert _resolve({**BODY, key: 0}, flex_coalesce_steps=steps).flex_coalesce == 0
    # 32 columns: 136 KB of staging.  Beside the pair's two staged O tiles
    # no two-deep ring is left, so the width resolves to 0 (the ring keeps
    # its depth); with the output staged in the dead Q stages it fits.
    wide = {**BODY, cute_flash.FLASH_S_LOAD_REP_KEY: 32}
    staged = _resolve({**wide, key: 128}, flex_coalesce_steps=steps)
    assert staged.flex_coalesce == 0
    assert staged.kv_stage == 3
    # ... and so does the default there (the campaign's edge pin renders as
    # before).
    assert _resolve(wide, flex_coalesce_steps=steps).flex_coalesce == 0
    over_q = {**wide, cute_flash.FLASH_O_OVER_Q_KEY: True}
    fitted = _resolve({**over_q, key: 128}, flex_coalesce_steps=steps)
    assert (fitted.flex_coalesce, fitted.flex_coalesce_slots, fitted.kv_stage) == (
        128,
        1,
        3,
    )
    assert _resolve({**over_q, key: 64}, flex_coalesce_steps=steps).flex_coalesce == 64
    # A byte run at 16 columns has no request width: every choice is 0.
    assert _resolve({**BODY, key: 128}, flex_coalesce_steps=(1,)).flex_coalesce == 0
    assert _resolve({**wide, key: 128}, flex_coalesce_steps=(1,)).flex_coalesce == 32
    assert _resolve({**BODY, key: 128}).flex_coalesce == 0
    assert _resolve(BODY).flex_coalesce == 0
    assert (
        _resolve(
            {**wide, key: 128}, has_list_walk=False, flex_coalesce_steps=(4,)
        ).flex_coalesce
        == 128
    )
    with pytest.raises(InvalidConfig):
        _resolve({**BODY, key: 48}, flex_coalesce_steps=steps)


def test_flex_coverage_design_covers_the_cooperative_widths() -> None:
    """The structural coverage design finds a witness for every searched
    cooperative width of the graphormer-edge surface (64 and 32 through the
    16-column chunk axis; the cold autotune of edge s2048 failed with
    "uncovered values [(cute_flash_flex_coalesce, 128)]" while 128 was
    offered) and of the one-table surface (128 at 32 columns)."""
    from helion._compiler.backend import CuteBackend
    from helion.autotuner.config_spec import BlockSizeSpec
    from helion.autotuner.config_spec import ConfigSpec

    for steps, hook_ops, expected in (
        ((4, 12), 23, (0, 32, 64)),
        ((4,), 3, (0, 32, 64, 128)),
    ):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True),
            patch(
                "helion.autotuner.config_spec.get_target_device_capability",
                return_value=(10, 0),
            ),
        ):
            spec = ConfigSpec(backend=CuteBackend())
            for block_id, target in enumerate((1, 128, 128)):
                spec.block_sizes.append(
                    BlockSizeSpec(block_id=block_id, size_hint=target)
                )
            spec.enable_cute_flash_search(
                head_dim=HEAD_DIM,
                num_kv=16,
                num_bh=NUM_BH,
                dtype=torch.bfloat16,
                block_size_targets={0: 1, 1: 128, 2: 128},
                standard_dense_output=False,
                standard_causal_output=False,
                plain_row_body=False,
                has_score_modifiers=True,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
                has_list_walk=True,
                flex_has_score_region=True,
                q_seq=2048,
                kv_seq=2048,
                flex_aux_smem_candidates=(1552,),
                flex_prefetch_regs=(32, 96),
                flex_coalesce_steps=steps,
                flex_hook_ops=hook_ops,
            )
            fragments = spec._cute_flash_autotune_fragments()
            fragment = fragments[cute_flash.FLASH_FLEX_COALESCE_KEY]
            assert set(fragment.search_choices) == set(expected)  # pyrefly: ignore[missing-attribute]
            assert fragment.default() == max(expected)  # pyrefly: ignore[missing-attribute]
            rolled = fragments[cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY]
            assert rolled.default() is (hook_ops >= 16)  # pyrefly: ignore[missing-attribute]
            generation = spec.create_config_generation()
            assert generation.flash_structural_coverage_uncovered_values() == []
            # The seeds carry the fact-based defaults (canonicalized per
            # body): the stream at every seed whose body holds it, the roll
            # for the large hook.
            seeds = spec.autotune_seed_configs()
            assert seeds
            widths = {
                seed.config.get(cute_flash.FLASH_FLEX_COALESCE_KEY) for seed in seeds
            }
            assert max(expected) in widths and None not in widths
            rolls = {
                seed.config.get(cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY)
                for seed in seeds
            }
            assert rolls == {hook_ops >= 16}


def test_codegen_binds_the_cooperative_staging_buffer() -> None:
    """With a cooperative width the fa4 storage grows by the eight softmax
    warps' staging buffers (``flex_coop_bytes``: 32 rows at the widest
    request's padded pitch, 36864 at 128 / 20480 at 64), the setup binds
    ``_flash_flex_coop_smem`` and the width reaches both hooks; without one
    nothing of that exists."""

    plan = _gather_plan()
    assert cute_flash._flash_flex_coalesce_steps(plan, NUM_BH, NUM_HEADS, 16) == (4,)
    seen: list[int] = []
    stub = cute_flash.render_flex_hook

    def recording(*args: object, **kwargs: object) -> FlexHookCode:
        seen.append(int(kwargs.get("coalesce", -1)))  # pyrefly: ignore[bad-argument-type]
        return stub(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    wide = {**BODY, cute_flash.FLASH_S_LOAD_REP_KEY: 32}
    with patch.object(cute_flash, "render_flex_hook", recording):
        src = _render(
            _resolve(
                {**wide, cute_flash.FLASH_FLEX_COALESCE_KEY: 128},
                flex_coalesce_steps=(4,),
            ),
            plan,
        )
        assert set(seen) == {128}
        seen.clear()
        narrow = _render(
            _resolve(
                {**BODY, cute_flash.FLASH_FLEX_COALESCE_KEY: 128},
                flex_coalesce_steps=(4,),
            ),
            plan,
        )
        assert set(seen) == {64}
        seen.clear()
        plain = _render(
            _resolve(
                {**wide, cute_flash.FLASH_FLEX_COALESCE_KEY: 0},
                flex_coalesce_steps=(4,),
            ),
            plan,
        )
        assert set(seen) == {0}
    # Two ring slots of 36 KB at 32 columns, four of 20 KB at 16.
    storage = next(
        line for line in src.splitlines() if "flash_fa4_shared_storage(" in line
    )
    assert "flex_coop_bytes=73728" in storage
    assert "_flash_flex_coop_smem = storage.sFlexCoop.data_ptr()" in src
    assert src.index("storage = smem.allocate(") < src.index("_flash_flex_coop_smem = ")
    narrow_storage = next(
        line for line in narrow.splitlines() if "flash_fa4_shared_storage(" in line
    )
    assert "flex_coop_bytes=81920" in narrow_storage
    assert "flex_coop_bytes" not in plain
    assert "sFlexCoop" not in plain


def test_flex_rolled_chunks_knob_rolls_pass1_and_pins_the_register_ring() -> None:
    """``cute_flash_flex_rolled_chunks``: an appended schema axis, default
    on where the score hook renders 16 or more per-element operations
    (``flex_hook_ops``: graphormer-edge 23) and off below, searched on
    list-walk plans (pinned off elsewhere); on, PASS1's chunk loop is a
    dynamic ``cutlass.range`` (one chunk body per stage), the cooperative
    stream's next-tile choice a select, and the register prefetch budget
    resolves to 0 (its slot lists cannot take a runtime chunk index)."""
    key = cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY
    assert cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS[-1] == key
    assert (
        cute_flash.FlashAttentionConfig.__dataclass_fields__[
            "flex_rolled_chunks"
        ].default
        is False
    )
    assert (key, True) in cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES
    with patch.dict(os.environ, {}, clear=True):
        flex = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
        )
        dense = cute_flash.flash_autotune_fragments(
            HEAD_DIM, 16, dtype=torch.bfloat16, num_bh=NUM_BH
        )
    assert set(flex[key].search_choices) == {False, True}  # pyrefly: ignore[missing-attribute]
    assert flex[key].default() is False  # pyrefly: ignore[missing-attribute]
    assert tuple(dense[key].search_choices) == (False,)  # pyrefly: ignore[missing-attribute]
    with patch.dict(os.environ, {}, clear=True):
        large = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
            flex_hook_ops=23,
        )
    assert large[key].default() is True  # pyrefly: ignore[missing-attribute]
    assert cute_flash.flash_flex_rolled_chunks_default(15) is False
    assert cute_flash.flash_flex_rolled_chunks_default(16) is True
    regs = (32, 96)
    plain = _resolve(BODY, flex_prefetch_regs=regs)
    assert (plain.flex_rolled_chunks, plain.flex_prefetch) == (False, 64)
    # graphormer-edge's hook (23 operations) rolls by default, which pins
    # the register ring off; a config may still say otherwise.
    by_fact = _resolve(BODY, flex_prefetch_regs=regs, flex_hook_ops=23)
    assert (by_fact.flex_rolled_chunks, by_fact.flex_prefetch) == (True, 0)
    assert (
        _resolve(
            {**BODY, key: False}, flex_prefetch_regs=regs, flex_hook_ops=23
        ).flex_rolled_chunks
        is False
    )
    rolled = _resolve({**BODY, key: True}, flex_prefetch_regs=regs)
    assert (rolled.flex_rolled_chunks, rolled.flex_prefetch) == (True, 0)
    rolled = _resolve(
        {**BODY, key: True, cute_flash.FLASH_FLEX_PREFETCH_KEY: 128},
        flex_prefetch_regs=regs,
    )
    assert (rolled.flex_rolled_chunks, rolled.flex_prefetch) == (True, 0)
    assert _resolve({key: True}, has_list_walk=False).flex_rolled_chunks is False

    from helion._compiler.cute.flash_score_mod import FlexHookCode

    plan = _gather_plan()
    stub = cute_flash.render_flex_hook

    def streaming(*args: object, **kwargs: object) -> FlexHookCode:
        code = stub(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        if kwargs.get("coalesce"):
            return FlexHookCode(
                code.item_prelude,
                code.tile_prologue,
                code.chunk_body,
                chunk_issue=" " * 16 + "'flex issue'\n",
                chunk_consume=" " * 16 + "'flex consume'\n",
                coop_slots=int(kwargs["coop_slots"]),  # pyrefly: ignore[bad-argument-type]
            )
        return code

    with patch.object(cute_flash, "render_flex_hook", streaming):
        src = _render(
            _resolve(
                {**BODY, key: True, cute_flash.FLASH_FLEX_COALESCE_KEY: 128},
                flex_coalesce_steps=(4,),
            ),
            plan,
        )
    assert src.count("for flash_ci in cutlass.range(flash_LD_CHUNKS):") == 2
    assert "for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):" not in src
    assert (
        "flash_flex_pkv = cutlass.Int32(cutlass.select_(flash_ci + 4 < flash_LD_CHUNKS, "
        "flash_kv, flash_kv_peek))"
    ) in src
    assert "flash_flex_ps = flash_ci % 4" in src
    unrolled = _render(_resolve({**BODY, key: False}, flex_coalesce_steps=(4,)), plan)
    assert "for flash_ci in cutlass.range(flash_LD_CHUNKS):" not in unrolled
    assert (
        unrolled.count("for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):") == 2
    )


def test_codegen_splices_the_cooperative_stream() -> None:
    """With a streamed run the softmax loop (a) issues chunks ``0 .. slots
    - 1`` of the walk's first tile before the KV loop after a full wait and
    a warp sync, (b) at the top of every PASS1 iteration runs the consume
    block ahead of the t2r copy and then issues chunk ``(ci + slots) %
    chunks`` of the current tile, or of the peeked tile past the chunk
    count, into slot ``ci % slots`` (both branches), and (c) refills the
    ring for the peeked tile on an absent step (status 0) after PASS1,
    ahead of PASS2; the slot count reaches the hook.  Without a width none
    of that exists."""
    from helion._compiler.cute.flash_score_mod import FlexHookCode

    plan = _gather_plan()
    seen: list[tuple[int, int]] = []
    stub = cute_flash.render_flex_hook

    def recording(*args: object, **kwargs: object) -> FlexHookCode:
        seen.append(
            (int(kwargs.get("coalesce", -1)), int(kwargs.get("coop_slots", -1)))
        )  # pyrefly: ignore[bad-argument-type]
        code = stub(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        if seen[-1][0]:
            return FlexHookCode(
                code.item_prelude,
                code.tile_prologue,
                code.chunk_body,
                chunk_issue=" " * 16
                + "'flex issue chunk=flash_flex_pc slot=flash_flex_ps'\n",
                chunk_consume=" " * 16 + "'flex consume chunk=flash_ci'\n",
                coop_slots=seen[-1][1],
            )
        return code

    with patch.object(cute_flash, "render_flex_hook", recording):
        src = _render(
            _resolve(
                {**BODY, cute_flash.FLASH_FLEX_COALESCE_KEY: 128},
                flex_coalesce_steps=(4,),
            ),
            plan,
        )
        assert set(seen) == {(64, 4)}
        seen.clear()
        plain = _render(
            _resolve(
                {**BODY, cute_flash.FLASH_FLEX_COALESCE_KEY: 0},
                flex_coalesce_steps=(4,),
            ),
            plan,
        )
        assert set(seen) == {(0, 1)}
    issue = "'flex issue chunk=flash_flex_pc slot=flash_flex_ps'"
    consume = "'flex consume chunk=flash_ci'"
    for loop in re.finditer(
        r"flash_ld_shape = tLDcS\[None, 0, None, None\]\.shape\n(.*?)cute\.arch\.fence_view_async_tmem_load",
        src,
        re.DOTALL,
    ):
        body = loop.group(1).partition(
            "for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):"
        )[2]
        # (b): consume, the bindings, the issue, all ahead of the t2r copy.
        at = body.index(consume)
        assert at < body.index("flash_flex_pc = (flash_ci + 4) % flash_LD_CHUNKS")
        assert "flash_flex_ps = flash_ci % 4" in body
        assert (
            "flash_flex_pkv = flash_kv if flash_ci + 4 < flash_LD_CHUNKS else flash_kv_peek"
            in body
        )
        assert at < body.index(issue) < body.index("cute.copy(")
    # One softmax loop per stage: (a) before the ``for`` (four issues after
    # the wait and the sync), (c) under the absent-step guard after PASS1.
    stages = src.split("flash_row_sum = cutlass.Float32(0.0)")[1:]
    assert len(stages) == 2
    for stage_src in stages:
        loop_at = stage_src.index("for flash_kv_iter in cutlass.range(")
        pre = stage_src[:loop_at]
        assert "flash_kv_peek = cutlass.Int32(0)" in pre
        assert pre.index("cute.arch.cp_async_wait_group(0)") < pre.index(
            "flash_flex_pkv = flash_kv_peek"
        )
        assert pre.count(issue) == 4
        assert "flash_flex_ps = 3" in pre
        body = stage_src[loop_at:]
        refill = body.index("if flash_flex_status == 0:")
        assert refill < body.index("flash_p_sum = cutlass.Float32(0.0)")
        assert body[refill:].count(issue) == 4
        assert "cute.arch.cp_async_wait_group(0)" in body[refill:]
    for text in (
        "flash_flex_pc",
        "flash_kv_peek",
        "flash_flex_pkv",
        "cute.arch.cp_async_wait_group",
        "cute.arch.cp_async_commit_group",
    ):
        assert text not in plain, text


def _gather_plan() -> AttentionScorePlan:
    """A flex plan whose score region gathers ``bias[h, idx[b, q, kv]]``:
    ``idx`` is a vector-form table, ``bias`` (16 x 7 fp32) a staging
    candidate."""
    import helion
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.cute.flash_score_mod import build_flex_region

    def score_mod(score, b, h, q_idx, kv_idx, idx, bias):
        return score + hl.load(bias, [h, hl.load(idx, [b, q_idx, kv_idx])])

    settings = helion.Settings(backend="cute")
    with (
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        CompileEnvironment(torch.device("cpu"), settings),
    ):
        region, aux, scalars = build_flex_region(
            score_mod,
            aux_specs=[
                ("idx", torch.int32, (16, 2048, 2048)),
                ("bias", torch.float32, (16, 7)),
            ],
        )
    return AttentionScorePlan(
        head_dim=HEAD_DIM,
        qk_scale_log2=math.log2(math.e) / math.sqrt(HEAD_DIM),
        q_seq=2048,
        kv_seq=2048,
        kv_walk=_walk(),
        score_region=region,
        aux_buffers=aux,
        host_scalars=scalars,
        scale_peeled=False,
    )


def test_codegen_stages_candidate_tables_in_shared_memory() -> None:
    """With a budget the fa4 storage grows by the padded table (``sFlexAux``),
    the CTA-wide setup binds the typed pointer and fills the copy (every
    thread, one zero sentinel) before the setup sync, and the hook is
    rendered with the budget; without one nothing of that exists."""
    from helion._compiler.cute.flash_score_mod import flex_aux_smem_candidates

    plan = _gather_plan()
    assert flex_aux_smem_candidates(plan) == (464,)
    seen: list[int] = []
    stub = cute_flash.render_flex_hook

    def recording(*args: object, **kwargs: object) -> cute_flash.FlexHookCode:
        seen.append(int(kwargs.pop("aux_smem_budget", -1)))  # pyrefly: ignore[bad-argument-type]
        return stub(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    with patch.object(cute_flash, "render_flex_hook", recording):
        staged = _render(
            _resolve(
                {**BODY, cute_flash.FLASH_FLEX_AUX_SMEM_KEY: 4096},
                flex_aux_smem_candidates=(464,),
            ),
            plan,
        )
        assert set(seen) == {4096}
        seen.clear()
        plain = _render(
            _resolve(
                {**BODY, cute_flash.FLASH_FLEX_AUX_SMEM_KEY: 0},
                flex_aux_smem_candidates=(464,),
            ),
            plan,
        )
        assert set(seen) == {0}
    # The storage allocation follows the descriptor prefetch (an
    # ``if warp_idx == 0`` block), so the fill is checked on the whole text.
    assert "flex_aux_bytes=464)" in staged
    assert "flash_flex_aux_smem = storage.sFlexAux.data_ptr()" in staged
    assert (
        "_flash_aux1_smem = cute.recast_ptr(flash_flex_aux_smem + 0, "
        "dtype=cutlass.Float32)" in staged
    )
    fill = staged.index("for flash_fa in cutlass.range_constexpr(1):")
    sync = staged.index("cute.arch.sync_threads()")
    assert "flash_fa_i = tidx + flash_fa * 512" in staged[fill:sync]
    assert (
        "(_flash_aux1_smem + flash_fa_i).store((_flash_aux1.iterator + "
        "flash_fa_i).load())" in staged[fill:sync]
    )
    assert "if flash_fa_i == 112:" in staged[fill:sync]
    assert (
        "(_flash_aux1_smem + flash_fa_i).store(cutlass.Float32(0))" in staged[fill:sync]
    )
    # Every thread fills, after the allocation and before the setup-wide
    # sync the softmax warps pass; no warp-role guard in between.
    allocation = staged.index("storage = smem.allocate(_flash_storage_cls)")
    assert allocation < fill < sync
    assert "warp_idx" not in staged[allocation:sync]
    for text in ("sFlexAux", "flex_aux_bytes", "_flash_aux1_smem", "flash_fa_i"):
        assert text not in plain, text


def test_codegen_splices_the_next_chunk_prefetch_around_the_t2r_copy() -> None:
    """With a prefetch budget the hook's prefetch block runs (a) before the
    KV loop for the walk's first tile (a peek of the initial words, the
    slots created: chunks ``0 .. depth - 1`` plus chunk 0 again into slot
    ``depth``), (b) at the top of every PASS1 iteration for chunk ``(ci +
    depth) % chunks`` of the current tile ahead of the t2r copy (both
    branches bind ``flash_flex_pkv = flash_kv``), and (c) after PASS1 on
    every step, outside the status guard, for the peeked next tile ahead
    of PASS2; the budget reaches the hook.  Without one nothing of that
    exists."""
    from helion._compiler.cute.flash_score_mod import FlexHookCode

    plan = _gather_plan()
    assert cute_flash._flash_flex_prefetch_regs(plan, NUM_BH, NUM_HEADS, 16) == (32,)
    seen: list[int] = []
    stub = cute_flash.render_flex_hook

    def recording(*args: object, **kwargs: object) -> FlexHookCode:
        seen.append(int(kwargs.pop("prefetch_regs", -1)))  # pyrefly: ignore[bad-argument-type]
        code = stub(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        if seen[-1]:
            return FlexHookCode(
                code.item_prelude,
                code.tile_prologue,
                code.chunk_body,
                " " * 16 + "'flex prefetch chunk=flash_flex_pc'\n",
                1,
            )
        return code

    with patch.object(cute_flash, "render_flex_hook", recording):
        src = _render(
            _resolve(
                {**BODY, cute_flash.FLASH_FLEX_PREFETCH_KEY: 32},
                flex_prefetch_regs=(32,),
            ),
            plan,
        )
        assert set(seen) == {32}
        seen.clear()
        plain = _render(
            _resolve(
                {**BODY, cute_flash.FLASH_FLEX_PREFETCH_KEY: 0},
                flex_prefetch_regs=(32,),
            ),
            plan,
        )
        assert set(seen) == {0}
    marker = "'flex prefetch chunk=flash_flex_pc'"
    for loop in re.finditer(
        r"flash_ld_shape = tLDcS\[None, 0, None, None\]\.shape\n(.*?)cute\.arch\.fence_view_async_tmem_load",
        src,
        re.DOTALL,
    ):
        block = loop.group(1)
        head, _, body = block.partition(
            "for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):"
        )
        # (b): the current tile bound, no fill inside PASS1's head.
        assert "flash_flex_pkv = flash_kv" in head and marker not in head
        step = body.index("flash_flex_pc = (flash_ci + 1) % flash_LD_CHUNKS")
        assert "flash_flex_ps = (flash_ci + 1) % 2" in body[step:]
        assert step < body.index(marker) < body.index("cute.copy(")
        assert "if flash_ci" not in body
    # One softmax loop per stage: (a) before the ``for``, (c) after PASS1
    # ahead of PASS2's ``flash_p_sum`` reset, both for ``flash_kv_peek``.
    stages = src.split("flash_row_sum = cutlass.Float32(0.0)")[1:]
    assert len(stages) == 2
    for stage_src in stages:
        loop_at = stage_src.index("for flash_kv_iter in cutlass.range(")
        pre = stage_src[:loop_at]
        assert "flash_kv_peek = cutlass.Int32(0)" in pre
        assert "flash_flex_pkv = flash_kv_peek" in pre
        assert pre.count(marker) == 2  # chunk 0 into slots 0 and 1 (depth 1)
        assert "flash_flex_pf_init = True" in pre and "flash_flex_ps = 1" in pre
        body = stage_src[loop_at:]
        # The in-loop peek follows the walk step (after the bit is cleared).
        assert body.index(
            "flash_flex_rem0 = flash_flex_rem0 & ~flash_flex_kvbit"
        ) < body.index("flash_kv_peek = cutlass.Int32(0)")
        fill = body.index("flash_flex_pkv = flash_kv_peek")
        assert fill < body.index("flash_p_sum = cutlass.Float32(0.0)")
        assert "flash_flex_pf_init = False" in body[fill:]
        # Outside the status guard: one level below the ``for`` (the render
        # is ``ast.unparse``d, so compare relative indentation).
        for_line = next(
            line
            for line in stage_src.splitlines()
            if line.lstrip().startswith("for flash_kv_iter in cutlass.range(")
        )
        for_indent = len(for_line) - len(for_line.lstrip())
        fill_line = next(
            line
            for line in body.splitlines()
            if line.strip() == "flash_flex_pkv = flash_kv_peek"
        )
        assert len(fill_line) - len(fill_line.lstrip()) == for_indent + 4
    for text in ("flash_flex_pc", "flash_kv_peek", "flash_flex_pkv"):
        assert text not in plain, text


def test_fragment_surface_is_length_invariant_for_the_list_walk() -> None:
    def surface(num_kv: int) -> dict[str, tuple[object, ...]]:
        with patch.dict(os.environ, {}, clear=True):
            fragments = cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                num_kv,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=True,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
            )
        return {
            key: (
                fragment.default(),
                tuple(fragment.choices),  # pyrefly: ignore[missing-attribute]
                None
                if fragment.search_choices is None  # pyrefly: ignore[missing-attribute]
                else tuple(fragment.search_choices),  # pyrefly: ignore[missing-attribute]
            )
            for key, fragment in fragments.items()
        }

    # Odd and even KV tile counts (and the ragged query lengths they go
    # with) share one surface: the family is keyed on the query geometry.
    assert surface(16) == surface(17) == surface(33)


# --------------------------------------------------------------------------
# M-0: the walk alone (no regions) against the dense body
# --------------------------------------------------------------------------


def test_list_walk_render_matches_the_dense_body_up_to_the_walk() -> None:
    cfg = _resolve(BODY)
    dense_cfg = _resolve(BODY, has_list_walk=False)
    flex = _roles(_render(cfg, _flex_plan(), has_lse=False))
    dense = _roles(_render(dense_cfg, dense_score_plan(HEAD_DIM), has_lse=False))
    assert flex.keys() == dense.keys()
    for role, flex_block in flex.items():
        dense_block = dense[role]
        flex_calls = _rt_calls(flex_block)
        dense_calls = _rt_calls(dense_block)
        if "warp_idx < 4" in role or "warp_idx >= 4" in role:
            # Softmax warpgroups: the dense passes under the per-step status
            # plus the zero-P step of an absent tile; the row sum publication
            # of a row with no visible key.
            dense_calls["fa4_disc_zero_store"] += 1
            dense_calls["row_kv_begin_row_sum"] += 1
        elif "warp_idx >= 8" in role:
            # Correction warps: the normalizer of a row with no visible key.
            dense_calls["rcp_approx_ftz_or_zero"] += dense_calls.pop("rcp_approx_ftz")
        assert flex_calls == dense_calls, role
    # Every role's KV loop runs the per-item trip count of the walk.
    for role, block in flex.items():
        if role == "setup":
            continue
        assert "_flash_num_kv_tiles" not in block, role
    assert flex["setup"].count("flash_num_active_kv = ") == 1
    headers = re.findall(
        r"for \w+ in cutlass\.range\((flash_num_active_kv[^,]*),",
        "\n".join(flex.values()),
    )
    # load, MMA steady, softmax0, softmax1, correction.
    assert sorted(headers) == sorted(
        [
            "flash_num_active_kv - 1",
            "flash_num_active_kv - 1",
            "flash_num_active_kv",
            "flash_num_active_kv",
            "flash_num_active_kv - 1",
        ]
    )


def test_softmax_stages_branch_on_their_own_status_and_publish_by_counter() -> None:
    src = _render(_resolve(BODY), _flex_plan())
    roles = _roles(src)
    for stage, guard in (("0", "warp_idx < 4"), ("1", "warp_idx >= 4")):
        role = next(block for name, block in roles.items() if guard in name)
        loop = _softmax_loop(role)
        # The merged tile and this stage's status from its own lists.
        assert "flash_kv = cutlass.Int32(cutlass.select_(flash_flex_rem0 != 0" in loop
        assert f"flash_flex_p{stage}_0 & flash_flex_kvbit" in loop
        assert f"flash_flex_f{stage}_0 & flash_flex_kvbit" in loop
        # Absent -> zero P; present -> the passes.
        assert loop.count("if flash_flex_status != 0:") == 2
        assert "fa4_disc_zero_store(" in loop
        assert "fa4_disc_rowmax_balanced(" in loop
        # The alpha publish guard is the plain loop counter.
        assert "if flash_kv_iter != 0:" in loop
        assert "flash_kv != " not in loop
        # Both stages re-derive the tile ahead of the S wait.
        assert loop.index("flash_kv = cutlass.Int32(0)") < loop.index("mbar_spin_wait(")
    # The load warp walks the same union: first tile before the prologue,
    # the next at the top of each step; no interval arithmetic anywhere.
    load = next(
        block for block in roles.values() if "tKgK[None, flash_kv_first]" in block
    )
    assert "flash_kv_first = cutlass.Int32(cutlass.select_(flash_flex_rem0 != 0" in load
    assert "flash_kv_next = cutlass.Int32(cutlass.select_(flash_flex_rem0 != 0" in load
    assert "tKgK[None, flash_kv_first]" in load
    assert "tKgK[None, flash_kv_next]" in load
    assert "flash_kv_last" not in src and "flash_kv_first +" not in src


def test_setup_reads_both_rows_of_the_pair_and_counts_the_union() -> None:
    setup = _roles(_render(_resolve(BODY), _flex_plan()))["setup"]
    for stage, row in (("0", "flash_m_tile0"), ("1", "flash_m_tile1")):
        assert f"flash_flex_row{stage} = {row}" in setup
        assert (
            f"flash_flex_cp{stage} = cutlass.Int32(_flash_mFlexPartCnt["
            f"flash_flex_b, flash_flex_h, flash_flex_row{stage}])"
        ) in setup
        assert (
            f"flash_flex_cf{stage} = cutlass.Int32(_flash_mFlexFullCnt["
            f"flash_flex_b, flash_flex_h, flash_flex_row{stage}])"
        ) in setup
        assert (
            f"for flash_flex_i in cutlass.range(flash_flex_cp{stage}, unroll=1):"
            in setup
        )
        assert (
            f"for flash_flex_i in cutlass.range(flash_flex_cf{stage}, unroll=1):"
            in setup
        )
    assert (
        "flash_flex_u0 = flash_flex_p0_0 | flash_flex_f0_0 | flash_flex_p1_0 | flash_flex_f1_0"
        in setup
    )
    # An empty union walks one fake step (tile 0, both stages absent).
    assert (
        "flash_num_active_kv = cutlass.max(cutlass.Int32(1), "
        "cutlass.Int32(cute.arch.popc(flash_flex_u0)))"
    ) in setup
    # The tables are indexed by the uncollapsed batch and head.
    assert "_flash_heads = 4" in setup
    assert "flash_flex_b = flash_bh // _flash_heads" in setup
    assert "flash_flex_h = flash_bh % _flash_heads" in setup
    # Constants for the hooks: the last tile and the column limit (no tail).
    assert "_flash_kv_tail_cols = cutlass.Int32(128)" in setup
    assert "_flash_last_tile = 15" in setup


def test_broadcast_tables_index_zero() -> None:
    plan = _flex_plan()
    assert plan.kv_walk is not None
    plan = AttentionScorePlan(
        **{
            **plan.__dict__,
            "kv_walk": _walk(table_batch=1, table_heads=1),
        }
    )
    setup = _roles(_render(_resolve(BODY), plan))["setup"]
    assert "flash_flex_b = cutlass.Int32(0)" in setup
    assert "flash_flex_h = cutlass.Int32(0)" in setup


def test_ascending_walk_takes_the_lowest_set_bit() -> None:
    cfg = _resolve({**BODY, cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending"})
    src = _render(cfg, _flex_plan())
    # (``ast.unparse`` drops the parentheses: ``-`` binds tighter than ``&``.)
    assert "cute.arch.clz(flash_flex_rem0 & cutlass.Int32(0) - flash_flex_rem0)" in src
    desc = _render(_resolve(BODY), _flex_plan())
    assert "cutlass.Int32(0) - flash_flex_rem0" not in desc


def test_wide_kv_counts_use_several_bitmask_words() -> None:
    # 65 KV tiles (svg-temporal s8320): three 32-bit words per list.
    plan = _flex_plan(kv_seq=65 * 128)
    src = _render(_resolve(BODY, num_kv=65), plan)
    setup = _roles(src)["setup"]
    for word in range(3):
        assert f"flash_flex_u{word} = " in setup
        assert f"cute.arch.popc(flash_flex_u{word})" in setup
    assert "flash_flex_u3" not in src
    assert "flash_flex_word = flash_flex_t >> cutlass.Int32(5)" in setup
    # The walk picks the highest word first (descending) ...
    loop = _softmax_loop(next(b for n, b in _roles(src).items() if "warp_idx < 4" in n))
    assert loop.index("flash_flex_rem0 != 0") < loop.index("flash_flex_rem2 != 0")
    # ... and the status reads the tile's word.
    assert "flash_flex_kvword = flash_kv >> cutlass.Int32(5)" in loop
    assert "cutlass.select_(flash_flex_kvword == 2, flash_flex_p0_2" in loop


# --------------------------------------------------------------------------
# M-1: ragged M and N, the 256-row BlockMask, the hook slot
# --------------------------------------------------------------------------


def _softmax_roles(roles: dict[str, str]) -> dict[str, str]:
    return {
        stage: next(block for name, block in roles.items() if guard in name)
        for stage, guard in (("0", "warp_idx < 4"), ("1", "warp_idx >= 4"))
    }


@pytest.mark.parametrize("q_seq", [2048 + 128, 64, 3 * 128 + 40])
def test_ragged_query_length_guards_the_staged_store_and_the_lse(q_seq: int) -> None:
    plan = _flex_plan(q_seq=q_seq)
    staged = _render(_resolve({**BODY, cute_flash.FLASH_EPI_TMA_KEY: False}), plan)
    tma = _render(_resolve(BODY), plan)
    rows = -(-q_seq // 128)
    for src in (staged, tma):
        setup = _roles(src)["setup"]
        for stage in ("0", "1"):
            assert (
                f"flash_live_rows{stage} = cutlass.min(cutlass.Int32(128), cutlass.max("
                f"cutlass.Int32(0), cutlass.Int32({q_seq}) - flash_m_tile{stage} * "
                "cutlass.Int32(128)))"
            ) in setup
        # The second row of the last pair exists only when the row count is
        # even; otherwise its counts are read under a guard.
        assert (f"if flash_flex_row1 < {rows}:" in setup) == (rows % 2 == 1)
        # Both stages' LSE rows are guarded by their live rows.
        assert src.count("if flash_local_tidx < flash_live_rows0:") == 1
        assert src.count("if flash_local_tidx < flash_live_rows1:") == 1
        assert "flash_m_tile0 * 128 + flash_local_tidx, flash_bh]" in src
    # The staged vector store takes the row-guarded variant. The TMA store
    # does not clip at the query length (the descriptor's sequence and batch
    # modes coalesce, so a box past the length lands in the next head's
    # rows): its epilogue is a hybrid, a full tile through TMA, a half-live
    # tile through the row-guarded vector store over the plain output view,
    # a dead tile stores nothing.
    for src in (staged, tma):
        assert src.count("fa4_store_o_smem_to_gmem_rows(") == 2
        assert "tOcO_epi, flash_live_rows0, cutlass.BFloat16)" in src
        assert "tOcO_epi, flash_live_rows1, cutlass.BFloat16)" in src
    for stage in ("0", "1"):
        assert f"if flash_live_rows{stage} == cutlass.Int32(128):" in tma
        assert f"elif flash_live_rows{stage} > cutlass.Int32(0):" in tma
        assert f"cute.copy(_flash_tma_o, tOsO_tma[None, {stage}]" in tma
    assert "flash_mO_stg_cur = _flash_mO[None, None, flash_bh]" in tma
    assert "_flash_mO[" not in staged
    assert "cp_async_bulk_wait_group(1, read=True)" not in tma


def test_aligned_query_length_keeps_the_unguarded_stores() -> None:
    src = _render(_resolve({**BODY, cute_flash.FLASH_EPI_TMA_KEY: False}), _flex_plan())
    assert "flash_live_rows" not in src
    assert "_rows(" not in src
    assert src.count("fa4_store_o_smem_to_gmem(") == 2
    assert "if flash_local_tidx <" not in src
    # The aligned TMA epilogue keeps the pipelined two-stage TMA store.
    tma = _render(_resolve(BODY), _flex_plan())
    assert "flash_live_rows" not in tma
    assert "_rows(" not in tma and "_flash_mO[" not in tma
    assert "cp_async_bulk_wait_group(1, read=True)" in tma


@pytest.mark.parametrize("with_mask", [False, True])
def test_ragged_kv_length_masks_the_last_tile_in_whichever_list(
    with_mask: bool,
) -> None:
    kv_seq = 2048 + 64
    plan = _flex_plan(kv_seq=kv_seq, mask_region=object() if with_mask else None)
    src = _render(_resolve(BODY, num_kv=17), plan)
    setup = _roles(src)["setup"]
    assert "_flash_kv_tail_cols = cutlass.Int32(128)" in setup
    assert "_flash_last_tile = 16" in setup
    needs_mask = (
        "flash_flex_needs_mask = (flash_flex_status == 2) | (flash_kv == _flash_last_tile)"
        if with_mask
        else "flash_flex_needs_mask = flash_kv == _flash_last_tile"
    )
    for stage, role in _softmax_roles(_roles(src)).items():
        loop = _softmax_loop(role)
        assert needs_mask in loop
        # PASS1 and PASS2 both branch on it; the masked PASS1 runs the tail
        # mask ahead of the mask hook and stores the chunk back; the dense
        # branch keeps the balanced row max without a store.
        assert loop.count("if flash_flex_needs_mask:") == 2
        masked = loop.split("else:")[0]
        tail = "_helion_flash_rt.mask_r2p_sm100_rank1(flash_disc_frg, _flash_kv_tail_cols - flash_ci * 16)"
        assert loop.count(tail) == 1
        # The column limit is re-bound per step: the masked body also runs
        # on partial tiles that are not the last one.
        limit = (
            "_flash_kv_tail_cols = cutlass.Int32(cutlass.select_(flash_kv == "
            "_flash_last_tile, cutlass.Int32(64), cutlass.Int32(128)))"
        )
        assert loop.count(limit) == 1
        assert loop.index(limit) < loop.index("mbar_spin_wait(flash_s_full_ptr")
        hook = f"'flex hook stage={stage} score=False mask={with_mask} tail=True"
        assert loop.index(hook) < loop.index(tail)
        store = f"cute.copy(flash_tiled_score_st{stage}, flash_disc_frg, tScoreSTtS{stage}[None, flash_ci, None, None])"
        assert loop.index(tail) < loop.index(store)
        assert loop.count(store) == 1
        assert loop.count("fa4_disc_rowmax_balanced(") == 1
        assert loop.count("_fmax_reduce_chunk_balanced(") == 1
        assert masked.count("mask_r2p_sm100_rank1") == 1


def test_mask_region_alone_selects_the_masked_body_on_partial_tiles() -> None:
    src = _render(_resolve(BODY), _flex_plan(mask_region=object()))
    for stage, role in _softmax_roles(_roles(src)).items():
        loop = _softmax_loop(role)
        assert "flash_flex_needs_mask = flash_flex_status == 2" in loop
        assert loop.count("if flash_flex_needs_mask:") == 2
        assert f"'flex hook stage={stage} score=False mask=True tail=False" in loop
        assert "mask_r2p_sm100_rank1" not in loop
        # Full tiles: the dense passes, no hook, no store-back.
        assert loop.count("fa4_disc_rowmax_balanced(") == 1
        assert loop.count("cute.copy(flash_tiled_score_st") == 1


def test_score_region_hooks_every_present_tile_without_a_mask() -> None:
    src = _render(_resolve(BODY), _flex_plan(score_region=object()))
    for stage, role in _softmax_roles(_roles(src)).items():
        loop = _softmax_loop(role)
        assert "flash_flex_needs_mask" not in loop
        assert f"'flex hook stage={stage} score=True mask=False tail=False" in loop
        assert "fa4_disc_rowmax_balanced(" not in loop
        assert loop.count("cute.copy(flash_tiled_score_st") == 1
        assert loop.count("fa4_disc_zero_store(") == 1
    # The unpeeled scale is a score region too.
    src = _render(_resolve(BODY), _flex_plan(scale_peeled=False))
    assert "'flex hook stage=0 score=True mask=False tail=False" in src


def test_score_and_mask_regions_render_both_hooks() -> None:
    src = _render(
        _resolve(BODY), _flex_plan(mask_region=object(), score_region=object())
    )
    for stage, role in _softmax_roles(_roles(src)).items():
        loop = _softmax_loop(role)
        assert f"'flex hook stage={stage} score=True mask=True tail=False" in loop
        assert f"'flex hook stage={stage} score=True mask=False tail=False" in loop
        assert "fa4_disc_rowmax_balanced(" not in loop
        assert loop.count("cute.copy(flash_tiled_score_st") == 2


def test_hook_pieces_land_in_their_slots_once() -> None:
    real = _stub_render_flex_hook

    def marked(plan: AttentionScorePlan, **kwargs: object) -> cute_flash.FlexHookCode:
        code = real(plan, **kwargs)  # pyrefly: ignore[bad-argument-type]
        stage = kwargs["stage"]
        return cute_flash.FlexHookCode(
            item_prelude=f"flash_hook_item{stage} = cutlass.Int32(7)\n",
            tile_prologue=f"    flash_hook_tile{stage} = flash_kv * 3\n",
            chunk_body=code.chunk_body,
        )

    plan = _flex_plan(mask_region=object(), score_region=object())
    with patch.object(cute_flash, "render_flex_hook", marked):
        src = _render(_resolve(BODY), plan)
    for stage, role in _softmax_roles(_roles(src)).items():
        item = f"flash_hook_item{stage} = cutlass.Int32(7)"
        tile = f"flash_hook_tile{stage} = flash_kv * 3"
        # Rendered once each although two hooks (masked and unmasked) were
        # rendered for the stage.
        assert role.count(item) == 1
        assert role.count(tile) == 1
        # The item prelude sits after the walk state and before the row
        # statistics, outside the KV loop.
        assert role.index("flash_flex_rem0 = flash_flex_u0") < role.index(item)
        assert role.index(item) < role.index(
            "flash_row_max = cutlass.Float32(-cutlass.Float32.inf)"
        )
        loop = _softmax_loop(role)
        assert item not in loop
        # The tile prologue follows the tile derivation and precedes the
        # S wait; the chunk body follows the chunk's TMEM load and precedes
        # the row-max fold.
        assert loop.index("flash_flex_status = ") < loop.index(tile)
        assert loop.index(tile) < loop.index("mbar_spin_wait(flash_s_full_ptr")
        load = f"cute.copy(flash_tiled_ld{stage}, tLDtS{stage}[None, flash_ci, None, None], flash_disc_frg)"
        hook = f"'flex hook stage={stage} score=True mask=True tail=False"
        assert loop.index(load) < loop.index(hook)
        assert loop.index(hook) < loop.index(
            "_fmax_reduce_chunk_balanced(flash_disc_frg, flash_row_max)"
        )


def test_256_row_blockmask_walks_one_row_for_both_stages() -> None:
    plan = _flex_plan(q_block=256, q_seq=2048 + 128)
    src = _render(_resolve(BODY), plan)
    setup = _roles(src)["setup"]
    assert "flash_flex_row0 = flash_m_pair" in setup
    assert "flash_flex_row1" not in src
    assert "flash_flex_u0 = flash_flex_p0_0 | flash_flex_f0_0" in setup
    assert setup.count("for flash_flex_i in cutlass.range(") == 2
    for role in _softmax_roles(_roles(src)).values():
        loop = _softmax_loop(role)
        assert "flash_flex_p0_0 & flash_flex_kvbit" in loop
        assert "flash_flex_p1_0" not in loop
    # The ragged pair's guards are still per 128-row stage.
    assert "flash_live_rows1" in setup


# --------------------------------------------------------------------------
# M-2: seeds
# --------------------------------------------------------------------------


def _flex_seeds(num_kv: int) -> list[cute_flash.FlashAttentionConfig]:
    plan: dict[str, object] = {
        "dtype": torch.bfloat16,
        "num_bh": NUM_BH,
        "has_list_walk": True,
        "score_modifier_kinds": (FLEX_SCORE_KIND,),
        "plain_row_body": False,
    }
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            HEAD_DIM,
            num_kv,
            **plan,  # pyrefly: ignore[bad-argument-type]
        )
        return [
            cute_flash.resolve_flash_config(
                HEAD_DIM,
                num_kv,
                seed.config,
                **plan,  # pyrefly: ignore[bad-argument-type]
            )
            for seed in seeds
        ]


@pytest.mark.parametrize("num_kv", [16, 17])
def test_flex_seeds_carry_the_fa4_defaults_and_the_pruned_campaign_structure(
    num_kv: int,
) -> None:
    resolved = _flex_seeds(num_kv)
    assert resolved
    for cfg in resolved:
        # Every seed resolves to the pinned plan category.
        assert cfg.pipeline_family == "fa4"
        assert not cfg.persistent
        assert cfg.softmax_disc
        assert cfg.stat_transport == "ring2"
        assert cfg.kv_tile_n == 128
        assert not cfg.causal_loop_split
        assert not cfg.early_prologue_loads
        assert cfg.masked_e2e_schedule == "inherit"
        assert cfg.epi_tma or cfg.epi_stg
    structures = {
        (cfg.exp2_packet, cfg.e2e_schedule)
        for cfg in resolved
        if cfg.exp2_packet == "8x2"
        and cfg.disc_pipe_depth == 2
        and cfg.split_p_arrive
        and cfg.causal_kv_order == "descending"
        and cfg.epi_tma
        and cfg.first_load_order == 3
        and cfg.mma_interleave
        and cfg.p_store_repetition == 16
        and cfg.s_load_repetition == 32
    }
    assert structures == {("8x2", "16/8"), ("8x2", "8/2")}
    # The fa4 defaults (the family seed) are there as well.
    with patch.dict(os.environ, {}, clear=True):
        defaults = cute_flash.resolve_flash_config(
            HEAD_DIM,
            num_kv,
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"},
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
            plain_row_body=False,
        )
    assert cute_flash.flash_effective_config_values(defaults) in [
        cute_flash.flash_effective_config_values(cfg) for cfg in resolved
    ]


def test_flex_seeds_are_length_invariant() -> None:
    def fingerprints(num_kv: int) -> list[dict[str, object]]:
        return [
            cute_flash.flash_effective_config_values(cfg) for cfg in _flex_seeds(num_kv)
        ]

    assert fingerprints(16) == fingerprints(17) == fingerprints(64)


# --------------------------------------------------------------------------
# Codegen: the plan category reaches the body through codegen_attention_flash
# --------------------------------------------------------------------------


def _probe_attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dense attention source with the four BlockMask tables as extra
    (source-unused) arguments; the test injects the flex plan at codegen."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = 1.0 / math.sqrt(head_dim) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m] = m_i + torch.log2(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


def _codegen_with_flex_plan(seq: int, config: dict[str, object]) -> str:
    """Render the probe kernel on CPU with the flex plan injected at codegen
    (what the matcher will do): the tables become ``kv_walk_args`` and the
    score plan carries a 128-row BlockMask walk with a mask region."""
    import dataclasses

    from test._cute_aux import _cpu_codegen

    import helion
    from helion._compiler.host_function import HostFunction

    batch, heads = 2, 4
    rows = -(-seq // 128)
    num_kv = -(-seq // 128)
    q, k, v = (
        torch.empty(batch, heads, seq, HEAD_DIM, dtype=torch.bfloat16) for _ in range(3)
    )
    part_cnt = torch.zeros(batch, heads, rows, dtype=torch.int32)
    part_idx = torch.zeros(batch, heads, rows, num_kv, dtype=torch.int32)
    full_cnt = torch.zeros(batch, heads, rows, dtype=torch.int32)
    full_idx = torch.zeros(batch, heads, rows, num_kv, dtype=torch.int32)
    real = cute_flash.flash_attention_tensor_plan

    def patched(df: object) -> object:
        plan = real(df)  # pyrefly: ignore[bad-argument-type]
        assert plan is not None
        by_name = HostFunction.current().params.arguments
        args = tuple(
            df.tensor_arg(by_name[name])  # pyrefly: ignore[missing-attribute]
            for name in ("part_cnt", "part_idx", "full_cnt", "full_idx")
        )
        state = df.cute_state  # pyrefly: ignore[missing-attribute]
        state.attention_flash_score_plan = dataclasses.replace(
            state.attention_flash_score_plan,
            q_seq=seq,
            kv_seq=seq,
            kv_walk=FlexKvWalk(*(arg.name for arg in args), 128, 128, batch, heads),
            mask_region=object(),  # pyrefly: ignore[bad-argument-type]
        )
        return plan._replace(
            kv_walk_args=args, tensor_4d_batch=batch, tensor_4d_heads=heads
        )

    kernel = helion.kernel(
        _probe_attention, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with (
        _cpu_codegen(),
        patch.object(cute_flash, "flash_attention_tensor_plan", patched),
    ):
        bound = kernel._bind_isolated((q, k, v, part_cnt, part_idx, full_cnt, full_idx))
        bound.config_spec._cute_flash_has_list_walk = True
        return bound.to_code(bound._normalized_config_copy(helion.Config(**config)))


@pytest.mark.parametrize("seq", [2048, 2048 + 128])
def test_codegen_emits_the_fa4_list_walk_at_every_kv_tile_count(seq: int) -> None:
    code = _codegen_with_flex_plan(
        seq,
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        },
    )
    pairs = -(-seq // 256)
    # The fa4 pair body (512 threads, 2 * 4 * pairs work items), not the
    # dense ws_overlap body an odd KV tile count used to fall back to.
    assert "'topology': 'fa4'" in code
    assert f"'total_tiles': {8 * pairs}" in code
    assert "block=(512, 1, 1)" in code
    assert "flash_flex_status" in code
    assert "flash_num_active_kv = cutlass.max(cutlass.Int32(1)" in code
    # The tables are plain kernel params the body binds by name.
    assert "_flash_mFlexPartCnt = part_cnt" in code
    assert "_flash_mFlexFullIdx = full_idx" in code
    assert (
        "def _helion__probe_attention(q_view, k_view, v_view, out, lse, part_cnt, part_idx, full_cnt, full_idx"
        in code
    )
    # Q and K/V share one length here: no separate K/V length in the plan.
    assert "'kv_seq'" not in code
    assert ("flash_live_rows0" in code) == (seq % 256 != 0)


# --------------------------------------------------------------------------
# v1.5: the MMA warp skips the QK / PV of an absent stage
# --------------------------------------------------------------------------


def _mma_block(src: str) -> str:
    roles = _roles(src)
    return next(
        block for block in roles.values() if "flash_tmem.allocate(512)" in block
    )


def _resolve_with_skip(config: dict[str, object]) -> cute_flash.FlashAttentionConfig:
    return _resolve({**config, cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: True})


def test_mma_warp_skips_the_absent_stage_and_keeps_the_protocol() -> None:
    cfg = _resolve_with_skip(BODY)
    assert cfg.flex_skip_absent
    src = _render(cfg, _flex_plan(mask_region=object()))
    setup = _roles(src)["setup"]
    # The per-stage "sees the tile" masks come from the CTA-wide setup.
    assert "flash_flex_n0_0 = flash_flex_p0_0 | flash_flex_f0_0" in setup
    assert "flash_flex_n1_0 = flash_flex_p1_0 | flash_flex_f1_0" in setup
    mma = _mma_block(src)
    # The MMA warp walks the same union and tests each stage's bit.
    assert (
        mma.count("flash_kv_mma = cutlass.Int32(cutlass.select_(flash_flex_rem0 != 0")
        == 2
    )
    assert (
        "flash_flex_pv_live0 = cute.arch.make_warp_uniform("
        "cutlass.Int32(flash_flex_n0_0 & flash_flex_kvbit != 0))"
    ) in mma
    assert (
        "flash_flex_qk_live1 = cute.arch.make_warp_uniform(cutlass.Int32(flash_flex_n1_0"
        in mma
    )
    # QK: prologue + steady per stage, predicated inside the PTX stream (no
    # Python branch: the shared descriptor walk stays unconditional).
    assert "if flash_flex" not in mma
    assert "shift_ptx_smem_desc" not in mma
    for stage in ("0", "1"):
        assert mma.count(f"issue=flash_flex_qk_live{stage})") == 2
        # PV: steady + epilogue per stage; the first PV is never skipped (it
        # zero-initializes the accumulator); the predicated stream keeps its
        # folded P-last wait.
        issue = (
            f"flash_flex_pv_issue{stage} = flash_flex_pv_live{stage} | "
            f"cutlass.Int32(flash_o_zero{stage})"
        )
        assert mma.count(issue) == 2
        assert mma.count(f"issue=flash_flex_pv_issue{stage})") == 2
        cleared = f"flash_o_zero{stage} = flash_o_zero{stage} & (flash_flex_pv_issue{stage} == 0)"
        assert mma.count(cleared) == 2
    # Every barrier arrival of the dense protocol is kept.
    assert mma.count("cute_tcgen05_flash.commit(flash_s_full_ptr + 0)") == 2
    assert mma.count("cute_tcgen05_flash.commit(flash_s_full_ptr + 1)") == 2
    assert mma.count("cute_tcgen05_flash.commit(flash_o_full_ptr + ") == 2
    assert mma.count("mbar_spin_wait(flash_pfor_ptr + ") == 4
    assert "mbar_spin_wait(flash_pfor2_ptr" not in mma
    assert mma.count("flash_pfor_phase ^= 1") == 2
    assert "flash_flex_pv_live0 = flash_flex_qk_live0" in mma


def test_mma_skip_is_a_searched_knob_off_by_default() -> None:
    # The knob's contract (schema position, witness, fragment, env default)
    # is in ``test_cute_flash_flex_skip_absent.py``.
    cfg = _resolve(BODY)
    assert not cfg.flex_skip_absent
    src = _render(cfg, _flex_plan(mask_region=object()))
    mma = _mma_block(src)
    assert "flash_flex" not in mma
    assert "flash_flex_n0_0" not in src
    assert mma.count("gemm_ptx_precomputed_qk(") == 4
    assert mma.count("gemm_ptx_precomputed_pv_ts(") == 4
    # The dense plan never carries the skip.
    assert not _resolve(BODY, has_list_walk=False).flex_skip_absent


def test_mma_skip_with_static_descriptors_predicates_the_static_issue() -> None:
    cfg = _resolve_with_skip({**BODY, cute_flash.FLASH_PRECOMPUTE_QK_DESC_KEY: True})
    assert cfg.precompute_qk_desc and cfg.flex_skip_absent
    mma = _mma_block(_render(cfg, _flex_plan(mask_region=object())))
    assert mma.count("gemm_ptx_precomputed_qk_static(") == 4
    assert mma.count("issue=flash_flex_qk_live0)") == 2


# --------------------------------------------------------------------------
# The trivial walk: one KV block covers the sequence, tables never read
# --------------------------------------------------------------------------


def test_trivial_walk_renders_every_tile_through_the_flex_body() -> None:
    plan = AttentionScorePlan(
        head_dim=HEAD_DIM,
        qk_scale_log2=math.log2(math.e) / math.sqrt(HEAD_DIM),
        q_seq=512,
        kv_seq=128,
        kv_walk=FlexKvWalk(
            "flex_part_cnt",
            "flex_part_idx",
            "flex_full_cnt",
            "flex_full_idx",
            512,
            128,
            1,
            1,
            trivial=True,
        ),
        mask_region=object(),  # pyrefly: ignore[bad-argument-type]
    )
    assert plan.is_flex and not plan.has_list_walk
    # The plan category is the flex kind, not the list walk: the surface may
    # report ``has_list_walk=False`` and still take every pin (one KV tile
    # would otherwise fall to the ws_overlap body, which has no hook).
    with patch.dict(os.environ, {}, clear=True):
        cfg = cute_flash.resolve_flash_config(
            HEAD_DIM,
            1,
            BODY,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=False,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
        )
    assert cfg.pipeline_family == "fa4"
    assert not cfg.persistent and cfg.kv_tile_n == 128 and cfg.stat_transport == "ring2"
    assert cfg.causal_kv_order == "descending"
    src = _render(cfg, plan)
    setup = _roles(src)["setup"]
    assert "flash_num_active_kv = _flash_num_kv_tiles" in setup
    assert "_flash_mFlexPartCnt" not in src
    assert "_flash_heads = 4" in setup
    assert "_flash_last_tile = 0" in setup
    for stage, role in _softmax_roles(_roles(src)).items():
        loop = _softmax_loop(role)
        assert (
            "flash_kv = flash_num_active_kv - cutlass.Int32(1) - flash_kv_iter" in loop
        )
        assert "flash_flex_status = cutlass.Int32(2)" in loop
        assert f"'flex hook stage={stage} score=False mask=True tail=False" in loop
        assert "flash_flex_rem" not in loop
        assert loop.count("fa4_disc_zero_store(") == 1
    # The normalizer pair still applies.
    assert "rcp_approx_ftz_or_zero" in src and "row_kv_begin_row_sum" in src


# --------------------------------------------------------------------------
# v1.5: static strided persistence with a per-role count prelude
# --------------------------------------------------------------------------


def test_persistent_list_walk_rebuilds_the_count_in_every_role() -> None:
    cfg = _resolve({**BODY, cute_flash.FLASH_PERSISTENT_KEY: True})
    assert cfg.persistent and cfg.persistent_loop == "while"
    assert not cfg.o_over_q and not cfg.use_clc_scheduler
    plan = _flex_plan(q_seq=2048 + 128, mask_region=object())
    src = _render(cfg, plan)
    setup = _roles(src)["setup"]
    # The CTA-wide setup keeps only the item-independent bindings (the tables
    # are plain kernel params, read per item) plus the host-sorted item order.
    assert "_flash_heads = 4" in setup
    assert "_flash_mFlexItems = flex_items" in setup
    assert "_flash_mFlexPartCnt[" not in setup
    assert "flash_num_active_kv" not in setup
    assert "flash_live_rows0" not in setup
    # Every walking role loops over its work items and rebuilds the masks,
    # the count and the live rows at the top of each (load, MMA, softmax0,
    # softmax1, correction); the epilogue warp only the live rows. Each of
    # them looks its work item up in the item table (longest union first)
    # and decodes the pair from it.
    assert src.count("while flash_tile_id <") >= 6
    assert src.count("flash_num_active_kv = cutlass.max(cutlass.Int32(1)") == 5
    assert src.count("flash_live_rows0 = cutlass.min(") == 6
    item_lookup = "flash_item = cutlass.Int32(_flash_mFlexItems[flash_tile_id])"
    assert src.count(item_lookup) == 6
    assert "flash_bh = flash_item // 9" in src  # 9 pairs per batch head
    for name, block in _roles(src).items():
        if name == "setup":
            continue
        if "flash_num_active_kv = cutlass.max(" in block:
            # The item lookup and decode precede the table reads.
            assert (
                block.index(item_lookup)
                < block.index("flash_m_tile0 = flash_m_pair * 2")
                < block.index(
                    "_flash_mFlexPartCnt[flash_flex_b, flash_flex_h, flash_flex_row0]"
                )
            )
    # Non-persistent: one CTA-wide count, no item loops, no item table.
    plain = _render(_resolve(BODY), plan)
    assert plain.count("flash_num_active_kv = cutlass.max(cutlass.Int32(1)") == 1
    assert "while flash_tile_id <" not in plain
    assert "_flash_mFlexItems" not in plain


def _item_table_reference(
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    *,
    q_block: int,
    batch: int,
    heads: int,
    num_pairs: int,
) -> list[int]:
    table_batch, table_heads, rows = part_cnt.shape
    counts = []
    for b in range(batch):
        for h in range(heads):
            tb, th = b % table_batch, h % table_heads
            for pair in range(num_pairs):
                pair_rows = [pair] if q_block == 256 else [2 * pair, 2 * pair + 1]
                union: set[int] = set()
                for row in pair_rows:
                    if row < rows:
                        union |= set(
                            part_idx[tb, th, row, : int(part_cnt[tb, th, row])].tolist()
                        )
                        union |= set(
                            full_idx[tb, th, row, : int(full_cnt[tb, th, row])].tolist()
                        )
                counts.append(len(union))
    return sorted(range(len(counts)), key=lambda item: (-counts[item], item))


@pytest.mark.parametrize(
    ("q_block", "rows", "num_pairs", "table_batch", "table_heads"),
    [(128, 17, 9, 2, 4), (128, 16, 8, 1, 1), (256, 9, 9, 1, 4), (128, 18, 9, 2, 1)],
)
def test_flex_item_table_orders_pairs_by_their_union_longest_first(
    q_block: int, rows: int, num_pairs: int, table_batch: int, table_heads: int
) -> None:
    from helion.runtime.cute.flash_flex import flex_item_table

    batch, heads, num_kv = 2, 4, 37
    generator = torch.Generator().manual_seed(q_block + rows)
    partial = (
        torch.rand(table_batch, table_heads, rows, num_kv, generator=generator) < 0.3
    )
    full = torch.rand(table_batch, table_heads, rows, num_kv, generator=generator) < 0.2
    full &= ~partial
    partial[0, 0, 0] = False
    full[0, 0, 0] = False
    tables = []
    for member in (partial, full):
        order = torch.argsort(
            member.to(torch.int8), dim=-1, descending=True, stable=True
        )
        tables.extend(
            (member.sum(-1).to(torch.int32), order.to(torch.int32).contiguous())
        )
    part_cnt, part_idx, full_cnt, full_idx = tables
    kwargs = {
        "q_block": q_block,
        "batch": batch,
        "heads": heads,
        "num_pairs": num_pairs,
    }
    table = flex_item_table(
        part_cnt, part_idx, full_cnt, full_idx, num_kv=num_kv, **kwargs
    )
    assert table.dtype == torch.int32 and table.shape == (batch * heads * num_pairs,)
    assert table.tolist() == _item_table_reference(
        part_cnt, part_idx, full_cnt, full_idx, **kwargs
    )
    # Cached behind the tables' version counters: the same tensors return
    # the same table; an in-place write rebuilds it.
    again = flex_item_table(
        part_cnt, part_idx, full_cnt, full_idx, num_kv=num_kv, **kwargs
    )
    assert again is table
    part_cnt[0, 0, 0] = 0
    rebuilt = flex_item_table(
        part_cnt, part_idx, full_cnt, full_idx, num_kv=num_kv, **kwargs
    )
    assert rebuilt is not table
    assert rebuilt.tolist() == _item_table_reference(
        part_cnt, part_idx, full_cnt, full_idx, **kwargs
    )


def test_codegen_binds_the_host_sorted_item_table_for_the_persistent_walk() -> None:
    seq = 2048 + 128
    code = _codegen_with_flex_plan(
        seq,
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        },
    )
    assert "'persistent': True" in code
    # The wrapper builds the order per call from the tables (cached by the
    # runtime behind their version counters) and passes it as a kernel param.
    assert (
        "_helion_flash_flex_item_table(part_cnt, part_idx, full_cnt, full_idx, "
        f"q_block=128, batch=2, heads=4, num_pairs={-(-seq // 256)}, "
        f"num_kv={-(-seq // 128)})"
    ) in code
    assert "_flash_mFlexItems = flash_flex_items" in code
    assert "flash_item = cutlass.Int32(_flash_mFlexItems[flash_tile_id])" in code
    assert (
        "flash_flex_items"
        in code.split("def _helion__probe_attention(")[1].split(")")[0]
    )
    plain = _codegen_with_flex_plan(
        seq,
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        },
    )
    assert "_helion_flash_flex_item_table" not in plain
    assert "_flash_mFlexItems" not in plain


@pytest.mark.parametrize("seq", [2048, 2048 + 128])
def test_ragged_tma_epilogue_passes_the_plain_output_view(seq: int) -> None:
    """A ragged query length under the TMA epilogue stores the last pair's
    partial tiles through the row-guarded vector store, which addresses the
    plain (S, D, B) output view: the launcher passes it after the TMA params
    (``epi_tma_ragged``); aligned lengths keep the TMA-only signature."""
    code = _codegen_with_flex_plan(
        seq,
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        },
    )
    ragged = seq % 256 != 0
    assert "'epi_tma': True" in code
    assert ("'epi_tma_ragged': True" in code) == ragged
    signature = code.split("def _helion__probe_attention(")[1].split(")")[0]
    params = [param.strip() for param in signature.split(",")]
    assert "_flash_mOt" in params and "_flash_tma_o" in params
    assert ("_flash_mO" in params) == ragged
    assert ("_flash_tma_o, _flash_osl, _flash_mO" in code) == ragged
    assert ("fa4_store_o_smem_to_gmem_rows(" in code) == ragged


def test_persistence_stays_pinned_off_for_interval_plans_and_is_searched_for_flex() -> (
    None
):
    assert (
        not _resolve(
            {**BODY, cute_flash.FLASH_PERSISTENT_KEY: True}, has_list_walk=False
        ).persistent
        or True
    )  # the dense plan may be persistent
    with patch.dict(os.environ, {}, clear=True):
        causal = cute_flash.resolve_flash_config(
            HEAD_DIM,
            16,
            {**BODY, cute_flash.FLASH_PERSISTENT_KEY: True},
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            is_causal=True,
            standard_causal_output=True,
        )
        fragments = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
        )
    assert not causal.persistent
    fragment = fragments[cute_flash.FLASH_PERSISTENT_KEY]
    assert fragment.default() is False
    assert set(fragment.search_choices) == {False, True}  # pyrefly: ignore[missing-attribute]
