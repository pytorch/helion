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
import dataclasses
import itertools
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
    fast_full: bool = False,
    **_: object,
) -> cute_flash.FlexHookCode:
    """A marker hook: these tests check the emitter's slots and walk, not the
    renderer (whose own tests use real regions); the plans here carry
    stand-in regions. The marker is a bare string literal (it survives
    ``ast.unparse``; a comment would not) plus the KV tail mask.  With
    ``fast_full`` on a mask-only hook the stub also hands back a stand-in
    keep word (one chunk-class line and its name), as the renderer does for
    a bitmask-form mask region without chunk loads."""
    pad = " " * indent
    chunk_words = mask_word = ""
    if fast_full and with_mask and not with_score:
        mask_word = f"flash_flex{stage}m_w0"
        chunk_words = f"{pad}{mask_word} = cutlass.Int32(stub_word({chunk_expr}))\n"
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
        item_prelude="",
        tile_prologue="",
        chunk_body="".join(lines),
        chunk_words=chunk_words,
        mask_word=mask_word,
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
    words_table: bool = False,
    kv_split: int | None = None,
    tail_split: int = 0,
) -> str:
    """``words_table``: bind the host words table (``cute_flash_flex_item_words``)
    like codegen does, which selects the cursor walk from two words on;
    ``tail_split`` binds the tail split's workspace like codegen does."""
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
        flex_words_table="flex_words" if words_table else None,
        flex_words_pairs=-(-q_seq // 256) if words_table else None,
        kv_split=kv_split,
        flex_split_arg="flash_split_ws" if (kv_split or tail_split) else None,
        flex_tail_split=tail_split,
        flex_tail_items_ws=min(NUM_BH * (-(-q_seq // 256)), 256) if tail_split else 0,
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


def test_trivial_flex_walk_is_never_persistent() -> None:
    """Torch's empty BlockMask (``kv_walk.trivial``: one KV block over the
    sequence, no tables) resolves non-persistent, requested or not, and the
    search surface pins the knob: the persistent list walk decodes its work
    items from the host-built item order those tables define."""
    key = cute_flash.FLASH_PERSISTENT_KEY
    assert _resolve({**BODY, key: True}).persistent is True
    assert _resolve({**BODY, key: True}, flex_trivial_walk=True).persistent is False
    assert _resolve(BODY, flex_trivial_walk=True).persistent is False
    with patch.dict(os.environ, {}, clear=True):
        fragment = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            flex_trivial_walk=True,
        )[key]
    assert fragment.default() is False  # pyrefly: ignore[missing-attribute]
    assert tuple(fragment.search_choices) == (False,)  # pyrefly: ignore[missing-attribute]


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


def test_cooperative_pair_request_is_honoured_on_the_table_backed_walk() -> None:
    """``ws_2cta`` within the pair body's value width: the single-tile
    body's cooperative pair hosts the symmetric table-backed walk on
    request (it measured well behind the pair body, 25% on the d128 cross
    walk and 52% on whisper's d64 ragged walk, 2026-10-17, so the
    unattended surface keeps the pair body alone); the trivial walk and
    the other families stay pinned to the pair body."""
    request = {**BODY, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_2cta"}
    resolved = _resolve(request)
    assert resolved.pipeline_family == "ws_2cta"
    assert resolved.topology == "ws_overlap" and resolved.use_2cta_instrs
    assert _resolve(request, flex_trivial_walk=True).pipeline_family == "fa4"
    surface = {
        "num_bh": NUM_BH,
        "dtype": torch.bfloat16,
        "is_causal": False,
        "has_kv_tile_pruning": False,
        "requires_ws_overlap": False,
        "small_biased_candidate": False,
        "standard_dense_output": False,
        "standard_causal_output": False,
        "output_requires_tma": False,
        "supports_tensor_4d_tma": False,
        "plain_row_body": False,
        "has_score_modifiers": True,
        "score_modifier_kinds": (FLEX_SCORE_KIND,),
        "has_list_walk": True,
    }
    with patch.dict(os.environ, {}, clear=True):
        for requested in (None, "ws_2cta"):
            assert cute_flash._flash_legal_autotune_pipeline_families(
                HEAD_DIM,
                16,
                requested_family=requested,
                **surface,  # pyrefly: ignore[bad-argument-type]
            ) == ("fa4",)


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
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys[keys.index(key) + 1] == cute_flash.FLASH_FLEX_COALESCE_KEY
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
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys[keys.index(key) + 1] == cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY
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
            # for the large hook; the pass-2c walk seeds and the mid-length
            # grouped-query walk roll explicitly.
            seeds = spec.autotune_seed_configs()
            assert seeds
            widths = {
                seed.config.get(cute_flash.FLASH_FLEX_COALESCE_KEY) for seed in seeds
            }
            assert max(expected) in widths and None not in widths
            p2c = [
                seed
                for seed in seeds
                if seed.config.get(cute_flash.FLASH_E2E_SCHEDULE_KEY) == "8/2"
                and seed.config.get(cute_flash.FLASH_SOFTMAX_REGION_KEY) == "shared"
            ]
            assert len(p2c) == 3
            assert {
                seed.config.get(cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY) for seed in p2c
            } == {True}
            rolls = {
                seed.config.get(cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY)
                for seed in seeds
                if seed not in p2c
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
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys[keys.index(key) + 1] == cute_flash.FLASH_FLEX_ITEM_WORDS_KEY
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


def test_resident_tiles_keep_the_prefetch_fill_in_every_fused_branch() -> None:
    """``cute_flash_s_resident`` with the hook's table prefetch: every fused
    tile branch (the masked reload arm, the resident arm and the zero-P
    arm) runs the cross-tile prefetch fill after PASS1 and ahead of PASS2's
    ``flash_p_sum`` reset; the pre-loop peek is the reload render's."""
    from helion._compiler.cute.flash_score_mod import FlexHookCode

    stub = cute_flash.render_flex_hook

    def prefetching(*args: object, **kwargs: object) -> FlexHookCode:
        kwargs.pop("prefetch_regs", None)
        code = stub(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        return FlexHookCode(
            code.item_prelude,
            code.tile_prologue,
            code.chunk_body,
            " " * 16 + "'flex prefetch chunk=flash_flex_pc'\n",
            1,
        )

    key = cute_flash.FLASH_S_RESIDENT_KEY
    with patch.object(cute_flash, "render_flex_hook", prefetching):
        src = _render(
            _resolve(
                {**RESIDENT_BODY, key: True, cute_flash.FLASH_FLEX_PREFETCH_KEY: 32},
                flex_prefetch_regs=(32,),
            ),
            _flex_plan(mask_region=object()),
        )
    marker = "'flex prefetch chunk=flash_flex_pc'"
    stages = src.split("flash_row_sum = cutlass.Float32(0.0)")[1:]
    assert len(stages) == 2
    for stage_src in stages:
        loop_at = stage_src.index("for flash_kv_iter in cutlass.range(")
        assert stage_src[:loop_at].count(marker) == 2
        body = stage_src[loop_at:]
        assert body.count("fa4_disc_exp_convert_store_resident_rep32(") == 1
        # Three fused arms (masked reload, resident, zero P), each with its
        # own cross-tile fill ahead of PASS2's reset.
        fills = [m.start() for m in re.finditer("flash_flex_pkv = flash_kv_peek", body)]
        resets = [
            m.start()
            for m in re.finditer(r"flash_p_sum = cutlass\.Float32\(0\.0\)", body)
        ]
        assert len(fills) == 3
        for fill, following in zip(fills, [*fills[1:], len(body)], strict=True):
            assert any(fill < reset < following for reset in resets)
            assert marker in body[fill:following]


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
    # Both stages' stores are issued before either stage is released (the
    # pipelined TMA path's order): no drain inside a stage's branch; the
    # release drains the first stage's bulk group behind the second's TMA
    # store when there is one, alone otherwise, then the second's.
    epilogue = next(
        block
        for block in _roles(tma).values()
        if "cute.copy(_flash_tma_o, tOsO_tma[None, 0]" in block
    )
    assert epilogue.count("cp_async_bulk_commit_group()") == 2
    stripped = "\n".join(line.strip() for line in epilogue.splitlines())
    assert (
        "cute.arch.cp_async_bulk_commit_group()\n"
        "cute.arch.cp_async_bulk_wait_group(0, read=True)"
    ) not in stripped
    assert (
        "with cute.arch.elect_one():\n"
        "if flash_live_rows1 == cutlass.Int32(128):\n"
        "cute.arch.cp_async_bulk_wait_group(1, read=True)\n"
        "else:\n"
        "cute.arch.cp_async_bulk_wait_group(0, read=True)\n"
        "cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)\n"
        "cute.arch.cp_async_bulk_wait_group(0, read=True)\n"
        "cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)"
    ) in stripped
    assert epilogue.count("mbarrier_arrive(flash_corr_epi_empty_ptr + 0)") == 2
    assert epilogue.count("mbarrier_arrive(flash_corr_epi_empty_ptr + 1)") == 2
    assert "cp_async_bulk_wait_group(1, read=True)" not in staged


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
        # PASS1 branches on it (the masked PASS1 runs the tail mask ahead of
        # the mask hook and stores the chunk back; the dense branch keeps the
        # balanced row max without a store); PASS2 is one call for both.
        assert loop.count("if flash_flex_needs_mask:") == 1
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
        assert loop.count("if flash_flex_needs_mask:") == 1
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
        and cfg.persistent
        and cfg.kv_split == 1
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
    # The two campaign-winner bodies (short and medium list walks), persistent
    # with their deeper rings and the 160-register softmax warps.
    cap = cute_flash._flash_aliased_kv_stage_cap(HEAD_DIM, stage_output=True)
    winners = {
        (
            cfg.e2e_schedule,
            cfg.exp2_packet,
            cfg.kv_stage,
            cfg.softmax_regs,
            cfg.corr_regs,
        )
        for cfg in resolved
        if cfg.exp2_packet == "4x2" and cfg.persistent and cfg.kv_split == 1
    }
    assert {
        ("16/4", "4x2", min(7, cap), 160, 128),
        ("8/2", "4x2", min(8, cap), 160, 112),
    } <= winners
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


@pytest.mark.parametrize("item_words", [True, False])
@pytest.mark.parametrize("seq", [2048, 2048 + 128])
def test_codegen_emits_the_fa4_list_walk_at_every_kv_tile_count(
    seq: int, item_words: bool
) -> None:
    code = _codegen_with_flex_plan(
        seq,
        {
            "block_sizes": [1, 128, 128],
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_EPI_TMA_KEY: True,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
            cute_flash.FLASH_FLEX_ITEM_WORDS_KEY: item_words,
        },
    )
    pairs = -(-seq // 256)
    num_kv = -(-seq // 128)
    # The fa4 pair body (512 threads, 2 * 4 * pairs work items), not the
    # dense ws_overlap body an odd KV tile count used to fall back to.
    assert "'topology': 'fa4'" in code
    assert f"'total_tiles': {8 * pairs}" in code
    assert "block=(512, 1, 1)" in code
    assert "flash_flex_status" in code
    assert "flash_num_active_kv = cutlass.max(cutlass.Int32(1)" in code
    # The tables are plain kernel params.
    assert (
        "def _helion__probe_attention(q_view, k_view, v_view, out, lse, part_cnt, part_idx, full_cnt, full_idx"
        in code
    )
    if item_words:
        # The default: every role loads the pair's [partial, full] words per
        # stage from the host-built table (one word per 32 KV tiles) instead
        # of rebuilding the masks from the counts and index lists.
        assert (
            "_helion_flash_flex_item_words(part_cnt, part_idx, full_cnt, full_idx, "
            f"q_block=128, batch=2, heads=4, num_pairs={pairs}, num_kv={num_kv})"
        ) in code
        assert "_flash_mFlexWords = flash_flex_words" in code
        assert (
            f"flash_flex_wrow = (flash_bh * cutlass.Int32({pairs}) + flash_m_pair) "
            "* cutlass.Int32(4)"
        ) in code
        assert (
            "flash_flex_f1_0 = cutlass.Int32(_flash_mFlexWords[flash_flex_wrow + cutlass.Int32(3)])"
            in code
        )
        assert "for flash_flex_i in cutlass.range(flash_flex_c" not in code
        assert "_flash_mFlexPartCnt" not in code
        # The tables' host flags (``flex_item_flags``, a runtime scalar
        # argument after the words): the words are bound to their all-full
        # constants (the second stage's row of the last pair is missing at
        # an odd row count: no tiles there) and loaded only while the
        # all-full bit is clear.
        assert (
            "_helion_flash_flex_item_flags(part_cnt, part_idx, full_cnt, full_idx, "
            f"q_block=128, batch=2, heads=4, num_pairs={pairs}, num_kv={num_kv})"
        ) in code
        assert "_flash_flex_hflags = cutlass.Int32(flash_flex_host_flags)" in code
        full_word = (1 << num_kv) - 1
        rows = -(-seq // 128)
        f1_word = (
            f"cutlass.Int32({full_word})"
            if rows % 2 == 0
            else f"cutlass.Int32(cutlass.select_(flash_m_tile1 < {rows}, "
            f"cutlass.Int32({full_word}), cutlass.Int32(0)))"
        )
        assert (
            f"flash_flex_p0_0 = cutlass.Int32(0)\n"
            f"            flash_flex_f0_0 = cutlass.Int32({full_word})\n"
            f"            flash_flex_p1_0 = cutlass.Int32(0)\n"
            f"            flash_flex_f1_0 = {f1_word}\n"
            "            if _flash_flex_hflags & cutlass.Int32(1) == cutlass.Int32(0):\n"
            "                flash_flex_p0_0 = cutlass.Int32(_flash_mFlexWords[flash_flex_wrow + cutlass.Int32(0)])"
        ) in code
    else:
        # The device walk: the body binds the tables by name and builds the
        # masks from the counts and lists.
        assert "_flash_mFlexPartCnt = part_cnt" in code
        assert "_flash_mFlexFullIdx = full_idx" in code
        assert "for flash_flex_i in cutlass.range(flash_flex_cp0, unroll=1):" in code
        assert "_helion_flash_flex_item_words" not in code
        assert "_flash_mFlexWords" not in code
    # Q and K/V share one length here: no separate K/V length in the plan.
    assert "'kv_seq'" not in code
    assert ("flash_live_rows0" in code) == (seq % 256 != 0)


def test_flex_item_words_knob_is_an_appended_axis_on_by_default_for_list_walks() -> (
    None
):
    """``cute_flash_flex_item_words``: the appended schema axis behind the
    rolled chunk loop, resolved on for list-walk plans without the key and off
    elsewhere, searched (both values) only with a list walk, carried as True
    by every list-walk seed."""
    key = cute_flash.FLASH_FLEX_ITEM_WORDS_KEY
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(key) == 1
    assert keys.index(key) == keys.index(cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY) + 1
    assert cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(key, True)] == {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
    }
    assert (
        cute_flash.FlashAttentionConfig.__dataclass_fields__["flex_item_words"].default
        is False
    )
    assert _resolve(BODY).flex_item_words is True
    assert _resolve({**BODY, key: False}).flex_item_words is False
    assert _resolve(BODY, has_list_walk=False).flex_item_words is False
    assert _resolve({**BODY, key: True}, has_list_walk=False).flex_item_words is False
    for resolved in (_resolve(BODY), _resolve({**BODY, key: False})):
        assert cute_flash.flash_effective_config_values(resolved)[key] is (
            resolved.flex_item_words
        )

    def search(has_list_walk: bool) -> tuple[object, ...]:
        with patch.dict(os.environ, {}, clear=True):
            fragment = cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                16,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=has_list_walk,
            )[key]
        choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
        picked = fragment.search_choices  # pyrefly: ignore[missing-attribute]
        return choices if picked is None else tuple(picked)

    # (The default leads the search choices.)
    assert search(True) == (True, False)
    assert search(False) == (False,)
    assert all(seed.flex_item_words is True for seed in _flex_seeds(16))


RESIDENT_BODY: dict[str, object] = {
    **BODY,
    cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    cute_flash.FLASH_P_STORE_REP_KEY: 32,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: False,
    cute_flash.FLASH_SOFTMAX_REGS_KEY: 200,
    cute_flash.FLASH_CORR_REGS_KEY: 64,
    cute_flash.FLASH_OTHER_REGS_KEY: 48,
}


def test_s_resident_knob_is_legal_on_the_rep32_chunked_body_only() -> None:
    """``cute_flash_s_resident``: appended behind the item words; legal on the
    list walk's fa4 chunked softmax with Rep32 score loads (any P store
    form), off causal bodies, score-region plans and the two-CTA bodies;
    searched only with a list walk without a score region; three resident
    seeds, none with a score region."""
    key = cute_flash.FLASH_S_RESIDENT_KEY
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(key) == 1
    assert keys.index(key) == keys.index(cute_flash.FLASH_FLEX_ITEM_WORDS_KEY) + 1
    assert cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(key, True)] == {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
        cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    }
    assert _resolve(RESIDENT_BODY).s_resident is False
    assert _resolve({**RESIDENT_BODY, key: True}).s_resident is True
    # Pinned off by Rep16 score loads (the resident fragments are the Rep32
    # load chunks; the resolver may canonicalize the repetitions together,
    # the knob follows the resolved body), by a score region (every tile
    # runs the hook's reload path) and off the list walk.
    rep16_loads = _resolve(
        {**RESIDENT_BODY, cute_flash.FLASH_S_LOAD_REP_KEY: 16, key: True}
    )
    assert rep16_loads.s_resident is (rep16_loads.s_load_repetition == 32)
    assert (
        _resolve({**RESIDENT_BODY, key: True}, has_score_modifiers=True).s_resident
        is False
    )
    assert (
        _resolve({**RESIDENT_BODY, key: True}, has_list_walk=False).s_resident is False
    )
    # Rep32 stores with the split P arrival (FA4's mixed Rep32/Rep16
    # publication) take the resident form as well.
    assert (
        _resolve(
            {**RESIDENT_BODY, cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True, key: True}
        ).s_resident
        is True
    )
    # Rep16 P stores keep the split P arrival (one store per score chunk).
    rep16 = _resolve(
        {
            **RESIDENT_BODY,
            cute_flash.FLASH_P_STORE_REP_KEY: 16,
            cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
            key: True,
        }
    )
    assert rep16.s_resident and rep16.split_p_arrive and rep16.p_store_repetition == 16
    with patch.dict(os.environ, {}, clear=True):
        causal = cute_flash.resolve_flash_config(
            HEAD_DIM,
            16,
            {**RESIDENT_BODY, key: True},
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            is_causal=True,
            standard_causal_output=True,
        )
    assert causal.s_resident is False
    for resolved in (_resolve(RESIDENT_BODY), _resolve({**RESIDENT_BODY, key: True})):
        assert (
            cute_flash.flash_effective_config_values(resolved)[key]
            is resolved.s_resident
        )

    def search(has_list_walk: bool, score_region: bool = False) -> tuple[object, ...]:
        with patch.dict(os.environ, {}, clear=True):
            fragment = cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                16,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=has_list_walk,
                flex_has_score_region=score_region,
            )[key]
        choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
        picked = fragment.search_choices  # pyrefly: ignore[missing-attribute]
        return choices if picked is None else tuple(picked)

    assert search(True) == (False, True)
    assert search(True, score_region=True) == (False,)
    assert search(False) == (False,)
    with patch.dict(os.environ, {}, clear=True):
        scored_seeds = cute_flash.flash_attention_seed_configs(
            HEAD_DIM,
            16,
            dtype=torch.bfloat16,
            num_bh=NUM_BH,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
            plain_row_body=False,
            has_score_modifiers=True,
        )
    assert not any(seed.config.get(key) for seed in scored_seeds)
    # Three resident seeds (short walk, medium walk, FA4 template), all with
    # the mixed Rep32/Rep16 store (split P arrival), FA4's register split and
    # 8-column correction tiles (the 32-column correction epilogue spills
    # beside a resident softmax).
    resident_seeds = [cfg for cfg in _flex_seeds(16) if cfg.s_resident]
    assert len(resident_seeds) == 3
    assert {
        (
            cfg.e2e_schedule,
            cfg.softmax_regs,
            cfg.corr_regs,
            cfg.other_regs,
            cfg.split_p_arrive,
            cfg.p_store_repetition,
            cfg.corr_tile_size,
            cfg.rescale_chunk_cols,
        )
        for cfg in resident_seeds
    } == {
        ("16/4", 200, 64, 48, True, 32, 8, 8),
        ("8/2", 200, 64, 48, True, 32, 8, 8),
    }


def test_tmem_rowmax_is_legal_for_list_walks_and_reduces_the_full_tiles() -> None:
    """``cute_flash_rowmax="tmem"`` resolves for list-walk plans on hardware
    with the TMEM reduction (GB300; the B200 policy has none): the full tiles
    fold the hardware per-load maxima, the masked tiles keep the hook's
    software row max, and the resident PASS1 reads the chunk and its maximum
    in one load."""
    key = cute_flash.FLASH_ROWMAX_KEY
    body = {
        **BODY,
        cute_flash.FLASH_S_LOAD_REP_KEY: 32,
        cute_flash.FLASH_P_STORE_REP_KEY: 32,
        cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: False,
        key: "tmem",
    }
    gb300 = {"target_device_capability": (10, 3)}
    assert _resolve(body, **gb300).rowmax == "tmem"
    assert _resolve(body, target_device_capability=(10, 0)).rowmax == "software"
    # (The dense plan was compatible already; Rep16 loads are not.)
    assert _resolve(body, has_list_walk=False, **gb300).rowmax == "tmem"
    assert (
        _resolve(
            {
                **body,
                cute_flash.FLASH_S_LOAD_REP_KEY: 16,
                cute_flash.FLASH_P_STORE_REP_KEY: 16,
            },
            **gb300,
        ).rowmax
        == "software"
    )
    plan = _flex_plan(mask_region=object())

    def render(cfg: cute_flash.FlashAttentionConfig) -> str:
        body_stmts = cute_flash.emit_flash_fa4_device_body(
            cast("DeviceFunction", None),
            head_dim=HEAD_DIM,
            num_kv=16,
            sequence_extent=2048,
            num_bh=NUM_BH,
            total_tiles=NUM_BH * 8,
            cfg=cfg,
            has_lse=True,
            io_dtype="cutlass.BFloat16",
            score_plan=plan,
            num_heads=NUM_HEADS,
            flex_item_table="flex_items",
            target_device_capability=(10, 3),
        )
        return "\n".join(ast.unparse(stmt) for stmt in body_stmts)

    code = render(_resolve(body, **gb300))
    assert code.count("_helion_flash_rt.disc_rowmax_ldred(") == 2
    assert "fa4_disc_rowmax_balanced(" not in code
    resident = render(
        _resolve(
            {
                **body,
                cute_flash.FLASH_S_RESIDENT_KEY: True,
                cute_flash.FLASH_SOFTMAX_REGS_KEY: 200,
                cute_flash.FLASH_CORR_REGS_KEY: 64,
                cute_flash.FLASH_OTHER_REGS_KEY: 48,
            },
            **gb300,
        )
    )
    assert (
        resident.count(
            "cute.copy(flash_tiled_ldred0, tLDRedtS0[None, 3, None, None], (flash_res_frg3, flash_res_red3))"
        )
        == 1
    )
    assert (
        resident.count(
            "flash_row_max = cute.arch.fmax(flash_row_max, flash_res_red3[flash_red_i])"
        )
        == 2
    )
    assert "disc_rowmax_ldred(" not in resident
    assert resident.count("fa4_disc_exp_convert_store_resident_rep32(") == 2


@pytest.mark.parametrize("seq", [2048, 2048 + 128])
def test_codegen_s_resident_keeps_the_score_row_in_registers_for_unmasked_tiles(
    seq: int,
) -> None:
    config = {
        "block_sizes": [1, 128, 128],
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_EPI_TMA_KEY: True,
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_S_LOAD_REP_KEY: 32,
        cute_flash.FLASH_P_STORE_REP_KEY: 32,
        cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: False,
        cute_flash.FLASH_SOFTMAX_REGS_KEY: 200,
        cute_flash.FLASH_CORR_REGS_KEY: 64,
        cute_flash.FLASH_OTHER_REGS_KEY: 48,
    }
    plain = _codegen_with_flex_plan(seq, config)
    resident = _codegen_with_flex_plan(
        seq, {**config, cute_flash.FLASH_S_RESIDENT_KEY: True}
    )
    assert "fa4_disc_exp_convert_store_resident_rep32(" not in plain
    assert "flash_res_frg3" not in plain
    # Both softmax stages: PASS1 loads the four 32-column chunks into resident
    # fragments (one trailing fence), PASS2 consumes them without a TMEM reload.
    assert (
        resident.count(
            "fa4_disc_exp_convert_store_resident_rep32((flash_res_frg0, flash_res_frg1, flash_res_frg2, flash_res_frg3), "
        )
        == 2
    )
    assert (
        resident.count(
            "flash_res_frg3 = cute.make_rmem_tensor(flash_res_ld_shape, cutlass.Float32)"
        )
        == 2
    )
    assert "fa4_disc_rowmax_balanced(" not in resident
    # The masked tiles (the probe plan has a mask region) keep the reload path.
    assert "fa4_disc_exp_convert_store_rep32" in resident
    assert (
        "flash_pfor_ptr + 0, None, flash_P_STORE_SPLIT, flash_P_STORE_CHUNKS"
        in resident
    )
    # Rep16 P stores: the resident PASS2 keeps the split P arrival.
    rep16 = _codegen_with_flex_plan(
        seq,
        {
            **config,
            cute_flash.FLASH_S_RESIDENT_KEY: True,
            cute_flash.FLASH_P_STORE_REP_KEY: 16,
            cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
        },
    )
    # The Rep16 resident PASS2: one call per resident chunk, each in a
    # region of its own (the staged-P stores keep their chunk order), and
    # the last release once, per stage.
    assert rep16.count("fa4_disc_resident_chunk16(flash_res_frg") == 8
    assert rep16.count("if flash_kv >= cutlass.Int32(0):") == 8
    assert rep16.count("fa4_disc_resident_finish16(") == 2
    assert "fa4_disc_exp_convert_store_resident_rep16(" not in rep16
    # Rep32 stores with the split P arrival: the mixed Rep32/Rep16
    # publication over the resident chunks (FA4's 3/4 release).
    mixed = _codegen_with_flex_plan(
        seq,
        {
            **config,
            cute_flash.FLASH_S_RESIDENT_KEY: True,
            cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
        },
    )
    # The mixed Rep32/Rep16 resident PASS2 per stage: one Rep32 pair below
    # the 3/4 split, the odd chunk and the last chunk through Rep16 stores,
    # each in a region of its own, and the two releases.
    assert (
        mixed.count(
            "fa4_disc_resident_pair32(flash_p_sum, flash_res_frg0, flash_res_frg1, "
        )
        == 2
    )
    assert mixed.count("fa4_disc_resident_chunk16_store(flash_res_frg2, ") == 2
    assert mixed.count("fa4_disc_resident_chunk16_store(flash_res_frg3, ") == 2
    assert mixed.count("fa4_disc_resident_release(flash_pfor_ptr + ") == 2
    assert mixed.count("fa4_disc_resident_release(flash_pfor2_ptr + ") == 2
    assert mixed.count("if flash_kv >= cutlass.Int32(0):") == 6
    # The pair folds into the carried row sum in the reload order; the odd
    # and the last chunk add their sums; stage 1 stores through its views
    # and releases its barriers, the split release before the last chunk.
    assert (
        "flash_p_sum = _helion_flash_rt.fa4_disc_resident_pair32(flash_p_sum, " in mixed
    )
    assert "flash_p_sum + _helion_flash_rt.fa4_disc_resident_chunk16_store(" in mixed
    assert (
        "fa4_disc_resident_pair32(flash_p_sum, flash_res_frg0, flash_res_frg1, "
        "flash_tiled_st32_1, tST32tS1, tST32cS, " in mixed
    )
    assert (
        "fa4_disc_resident_chunk16_store(flash_res_frg3, flash_tiled_st1, tSTtS1, tSTcS, "
        in mixed
    )
    stage1 = mixed[mixed.index("(warp_idx >= 4) & (warp_idx < 8):") :]
    order = [
        stage1.index("fa4_disc_resident_pair32("),
        stage1.index("fa4_disc_resident_chunk16_store(flash_res_frg2, "),
        stage1.index("fa4_disc_resident_release(flash_pfor_ptr + 1)"),
        stage1.index("fa4_disc_resident_chunk16_store(flash_res_frg3, "),
        stage1.index("fa4_disc_resident_release(flash_pfor2_ptr + 1)"),
    ]
    assert order == sorted(order), order
    # The Rep16 form: the four chunk calls in chunk order per stage, each
    # with its stage's views and barriers, then the final release; the
    # template's zero row sum is the only one (no second initialisation).
    for stage in ("0", "1"):
        calls = [
            f"fa4_disc_resident_chunk16(flash_res_frg{chunk}, flash_tiled_st{stage}, "
            f"tSTtS{stage}, tSTcS, "
            for chunk in range(4)
        ]
        positions = [rep16.index(call) for call in calls]
        assert positions == sorted(positions), positions
        for chunk in range(4):
            assert (
                f"{chunk}, flash_P_STORE_SPLIT, flash_P_STORE_CHUNKS, "
                f"flash_pfor_ptr + {stage}, flash_pfor2_ptr + {stage}, " in rep16
            )
        assert rep16.index(calls[3]) < rep16.index(
            f"fa4_disc_resident_finish16(flash_pfor_ptr + {stage}, flash_pfor2_ptr + {stage})"
        )
    for src in (rep16, mixed):
        lines = [line.strip() for line in src.splitlines()]
        assert not any(
            a == b == "flash_p_sum = cutlass.Float32(0.0)"
            for a, b in itertools.pairwise(lines)
        )


@pytest.mark.parametrize(
    ("q_block", "rows", "num_pairs", "table_batch", "table_heads", "num_kv"),
    [
        (128, 17, 9, 2, 4, 37),
        (128, 16, 8, 1, 1, 70),
        (256, 9, 9, 1, 4, 32),
        (128, 18, 9, 2, 1, 33),
    ],
)
def test_flex_item_words_pack_the_pair_rows_into_tile_bitmasks(
    q_block: int,
    rows: int,
    num_pairs: int,
    table_batch: int,
    table_heads: int,
    num_kv: int,
) -> None:
    from helion.runtime.cute.flash_flex import flex_item_table
    from helion.runtime.cute.flash_flex import flex_item_words
    from helion.runtime.cute.flash_flex import flex_words_header

    batch, heads = 2, 4
    generator = torch.Generator().manual_seed(q_block + rows + num_kv)
    partial = (
        torch.rand(table_batch, table_heads, rows, num_kv, generator=generator) < 0.3
    )
    full = torch.rand(table_batch, table_heads, rows, num_kv, generator=generator) < 0.2
    full &= ~partial
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
        "num_kv": num_kv,
    }
    flat = flex_item_words(part_cnt, part_idx, full_cnt, full_idx, **kwargs)
    nwords = -(-num_kv // 32)
    lists = 2 if q_block == 256 else 4
    # From two words on, the cursor walk's per-item header leads the row:
    # [union popcount, highest non-empty union word, lowest one, stage flags].
    header = flex_words_header(num_kv)
    assert header == (4 if nwords >= 2 else 0)
    assert flat.dtype == torch.int32
    assert flat.shape == (batch * heads * num_pairs * (header + lists * nwords),)
    assert flat.is_contiguous()
    words = flat.view(batch * heads * num_pairs, header + lists * nwords)

    def bit(word: int, tile: int) -> int:
        return ((word & 0xFFFFFFFF) >> (tile & 31)) & 1

    for b in range(batch):
        for h in range(heads):
            for pair in range(num_pairs):
                row = words[(b * heads + h) * num_pairs + pair].tolist()
                if q_block == 256:
                    members = [(partial, pair), (full, pair)]
                else:
                    members = [
                        (partial, 2 * pair),
                        (full, 2 * pair),
                        (partial, 2 * pair + 1),
                        (full, 2 * pair + 1),
                    ]
                union = [0] * num_kv
                stage_any = [0, 0]
                for index, (member, table_row) in enumerate(members):
                    for tile in range(num_kv):
                        expect = (
                            int(
                                member[
                                    b % table_batch, h % table_heads, table_row, tile
                                ]
                            )
                            if table_row < rows
                            else 0
                        )
                        assert (
                            bit(row[header + index * nwords + tile // 32], tile)
                            == expect
                        ), (b, h, pair, index, tile)
                        union[tile] |= expect
                        stage_any[index // 2 if lists == 4 else 0] |= expect
                if header:
                    present = [
                        w for w in range(nwords) if any(union[w * 32 : w * 32 + 32])
                    ]
                    expect_header = [
                        sum(union),
                        max(present) if present else -1,
                        min(present) if present else 0,
                        stage_any[0] | (stage_any[1] << 1),
                    ]
                    assert row[:header] == expect_header, (b, h, pair, row[:header])
    # Built together with the item order and cached with it: the same
    # tensors return the same tables; an in-place write rebuilds both.
    order = flex_item_table(part_cnt, part_idx, full_cnt, full_idx, **kwargs)
    assert flex_item_words(part_cnt, part_idx, full_cnt, full_idx, **kwargs) is flat
    assert flex_item_table(part_cnt, part_idx, full_cnt, full_idx, **kwargs) is order
    full_cnt[0, 0, 0] = 0
    rebuilt = flex_item_words(part_cnt, part_idx, full_cnt, full_idx, **kwargs)
    assert rebuilt is not flat
    # The last lookup is memoized on the tables' identities and versions and
    # the geometry (the launch's three host calls hit it): the same call
    # returns the rebuilt tables; another geometry gets its own entry and
    # the first one comes back from the keyed cache; a copy of a table is
    # another object (its own entry, served again on its own repeat); an
    # in-place write misses again.
    assert flex_item_words(part_cnt, part_idx, full_cnt, full_idx, **kwargs) is rebuilt
    grouped = flex_item_table(part_cnt, part_idx, full_cnt, full_idx, **kwargs, group=1)
    assert grouped is not flex_item_table(
        part_cnt, part_idx, full_cnt, full_idx, **kwargs
    )
    assert flex_item_words(part_cnt, part_idx, full_cnt, full_idx, **kwargs) is rebuilt
    assert (
        flex_item_table(part_cnt, part_idx, full_cnt, full_idx, **kwargs, group=1)
        is grouped
    )
    copy_cnt = full_cnt.clone()
    copied = flex_item_words(part_cnt, part_idx, copy_cnt, full_idx, **kwargs)
    assert copied is not rebuilt
    assert flex_item_words(part_cnt, part_idx, copy_cnt, full_idx, **kwargs) is copied
    part_cnt[0, 0, 0] = part_cnt[0, 0, 0]  # a write bumps the version counter
    assert (
        flex_item_words(part_cnt, part_idx, full_cnt, full_idx, **kwargs) is not rebuilt
    )
    # Inference tensors are rebuilt on every call and never memoized.
    from helion.runtime.cute.flash_flex import flex_item_flags

    with torch.inference_mode():
        inf = (part_cnt.clone(), part_idx.clone(), full_cnt.clone(), full_idx.clone())
    assert all(t.is_inference() for t in inf)
    first = flex_item_words(*inf, **kwargs)
    assert flex_item_words(*inf, **kwargs) is not first
    assert flex_item_flags(*inf, **kwargs) == 0


def _flex_tables_from_dense(
    partial: torch.Tensor, full: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The BlockMask's count and index lists of two dense memberships."""
    tables = []
    for member in (partial, full):
        order = torch.argsort(
            member.to(torch.int8), dim=-1, descending=True, stable=True
        )
        tables.extend(
            (member.sum(-1).to(torch.int32), order.to(torch.int32).contiguous())
        )
    return tables[0], tables[1], tables[2], tables[3]


@pytest.mark.parametrize(("q_block", "rows"), [(128, 16), (128, 17), (256, 8)])
def test_flex_item_flags_name_the_all_full_tables_and_the_identity_order(
    q_block: int, rows: int
) -> None:
    """``flex_item_flags``: bit 0 when no row has a partial tile and every
    row's full list is every tile (a ragged query length's missing row does
    not count), bit 1 when the longest-first item order is the identity;
    one host read per build, cached with the tables, 0 for inference
    tensors (the tables are rebuilt per call there, no sync)."""
    from helion.runtime.cute.flash_flex import FLEX_FLAG_ALL_FULL
    from helion.runtime.cute.flash_flex import FLEX_FLAG_IDENTITY_ORDER
    from helion.runtime.cute.flash_flex import flex_item_flags
    from helion.runtime.cute.flash_flex import flex_item_table

    batch, heads, num_kv = 2, 4, 12
    num_pairs = -(-rows // 2) if q_block == 128 else rows
    kwargs = {
        "q_block": q_block,
        "batch": batch,
        "heads": heads,
        "num_pairs": num_pairs,
        "num_kv": num_kv,
    }
    shape = (batch, heads, rows, num_kv)
    none = torch.zeros(shape, dtype=torch.bool)
    every = torch.ones(shape, dtype=torch.bool)
    # torch's noop / full BlockMask: every tile in every row's full list.
    assert (
        flex_item_flags(*_flex_tables_from_dense(none, every), **kwargs)
        == FLEX_FLAG_ALL_FULL | FLEX_FLAG_IDENTITY_ORDER
    )
    # Every row sees the same half of the tiles, as full tiles: the
    # counts are equal (identity order) but the lists are not all full.
    half = every.clone()
    half[..., num_kv // 2 :] = False
    assert (
        flex_item_flags(*_flex_tables_from_dense(none, half), **kwargs)
        == FLEX_FLAG_IDENTITY_ORDER
    )
    # The same tiles as partial tiles: equal counts still, not all full.
    assert (
        flex_item_flags(*_flex_tables_from_dense(half, none), **kwargs)
        == FLEX_FLAG_IDENTITY_ORDER
    )
    # A causal-like staircase (unequal counts) over full tiles: neither.
    stair = every.clone()
    for row in range(rows):
        stair[:, :, row, min(num_kv, row + 1) :] = False
    assert flex_item_flags(*_flex_tables_from_dense(none, stair), **kwargs) == 0
    # One tile short in the last item's row flips bit 0 off and keeps the
    # order (the items' unions stay non-increasing); short in the first
    # pair (both of its rows), the longest-first sort moves it.
    short = every.clone()
    short[batch - 1, heads - 1, rows - 1, 3] = False
    assert (
        flex_item_flags(*_flex_tables_from_dense(none, short), **kwargs)
        == FLEX_FLAG_IDENTITY_ORDER
    )
    short[0, 0, : (2 if q_block == 128 else 1), 3] = False
    assert flex_item_flags(*_flex_tables_from_dense(none, short), **kwargs) == 0
    # Cached with the tables: one build; the host read happens once, lazily,
    # only when the flags are asked for (the words and order arguments of a
    # render that binds no flags never sync). Inference tensors take the
    # uncached path and return 0 without building anything.
    from helion.runtime.cute import flash_flex
    from helion.runtime.cute.flash_flex import flex_item_words

    tables = _flex_tables_from_dense(none, every)
    with (
        patch.object(
            flash_flex,
            "_build_flex_item_tables",
            wraps=flash_flex._build_flex_item_tables,
        ) as build,
        patch.object(
            flash_flex, "_flex_flags_from_bits", wraps=flash_flex._flex_flags_from_bits
        ) as read,
    ):
        words = flex_item_words(*tables, **kwargs)
        order = flex_item_table(*tables, **kwargs)
        assert build.call_count == 1 and read.call_count == 0
        flags = flex_item_flags(*tables, **kwargs)
        assert flags == FLEX_FLAG_ALL_FULL | FLEX_FLAG_IDENTITY_ORDER
        assert flex_item_flags(*tables, **kwargs) == flags
        assert flex_item_words(*tables, **kwargs) is words
        assert flex_item_table(*tables, **kwargs) is order
        assert build.call_count == 1 and read.call_count == 1
        with torch.inference_mode():
            inference = tuple(table.clone() for table in tables)
        assert all(table.is_inference() for table in inference)
        assert flex_item_flags(*inference, **kwargs) == 0
        assert build.call_count == 1 and read.call_count == 1


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
        # folded P-last wait. The last PV1 issues on the item's snapshot of
        # its stage-1 flag (the persistent loop's early next-item head
        # rewrites the walk's flag between the two last PVs).
        issue = (
            f"flash_flex_pv_issue{stage} = flash_flex_pv_live{stage} | "
            f"cutlass.Int32(flash_o_zero{stage})"
        )
        assert mma.count(issue) == (2 if stage == "0" else 1)
        assert mma.count(f"issue=flash_flex_pv_issue{stage})") == 2
        cleared = f"flash_o_zero{stage} = flash_o_zero{stage} & (flash_flex_pv_issue{stage} == 0)"
        assert mma.count(cleared) == 2
    assert mma.count("flash_flex_pv_last1 = flash_flex_pv_live1") == 1
    assert (
        mma.count(
            "flash_flex_pv_issue1 = flash_flex_pv_last1 | cutlass.Int32(flash_o_zero1)"
        )
        == 1
    )
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
    # (The MMA warp's early next-item QK0, the default, re-emits its item
    # setup inside the loop: ``test_persistent_list_walk_early_qk...`` below.)
    cfg = _resolve(
        {
            **BODY,
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_FLEX_EARLY_QK_KEY: False,
        }
    )
    assert cfg.persistent and cfg.persistent_loop == "while"
    assert not cfg.o_over_q and not cfg.use_clc_scheduler
    assert not cfg.flex_early_qk
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


def test_tail_split_renders_the_last_waves_items_as_split_units() -> None:
    """``cute_flash_flex_tail_split``: on the one-item grid the units past
    the launcher's ``_flash_tail_base`` decode as split units (split id,
    the unit's slice shift, the workspace row), the walk's slice takes the
    unit's own shift, and the correction / epilogue / softmax roles branch
    per unit between the staged epilogue and the split-KV partial store and
    merge; off, the render is the plain grid's.  The resolver pins it off
    on the persistent loop, under a split KV and on the trivial walk (the
    cursor walk keeps it); the fragment is searched on every list walk with
    a seed that fills the last partial wave's idle SMs."""
    cfg = _resolve({**BODY, cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4})
    assert cfg.flex_tail_split == 4 and not cfg.persistent and cfg.kv_split == 1
    plan = _flex_plan(q_seq=2048 + 128, mask_region=object())
    src = _render(cfg, plan, tail_split=4)
    setup = _roles(src)["setup"]
    # The unit decode: the tail's units are four per item, adjacent.
    assert "if flash_pid >= _flash_tail_base:" in setup
    assert "flash_tail_item = flash_tail_unit >> cutlass.Int32(2)" in setup
    assert "flash_kv_split_id = flash_tail_unit & cutlass.Int32(3)" in setup
    assert "flash_kv_split_shift = cutlass.Int32(2)" in setup
    assert "flash_pid = _flash_tail_base + flash_tail_item" in setup
    # The slice: ``per = ceil(total / 2**shift)`` with the unit's shift, the
    # walk-order loops over the word in both directions.
    assert (
        "flash_flex_per = flash_flex_total + ((cutlass.Int32(1) << flash_kv_split_shift)"
        " - cutlass.Int32(1)) >> flash_kv_split_shift"
    ) in setup
    assert "for flash_flex_i in cutlass.range(flash_flex_cnt, unroll=1):" in setup
    assert (
        "flash_num_active_kv = cutlass.max(cutlass.Int32(1), flash_flex_cnt)" in setup
    )
    cfg_asc = _resolve(
        {
            **BODY,
            cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending",
        }
    )
    assert cfg_asc.flex_tail_split == 4 and cfg_asc.causal_kv_order == "ascending"
    setup_asc = _roles(_render(cfg_asc, plan, tail_split=4))["setup"]
    assert "for flash_flex_i in cutlass.range(flash_flex_cnt, unroll=1):" in setup_asc
    assert (
        "cute.arch.clz(flash_flex_u0 & cutlass.Int32(0) - flash_flex_u0)" in setup_asc
    )
    # The correction warps branch per unit between the merge and the staged
    # epilogue (both stages), the epilogue warp between the handshake and
    # its stores, the softmax warps store the LSE of the unsplit units only.
    roles = _roles(src)
    corr = next(
        block for block in roles.values() if "fa4_split_partial_store(" in block
    )
    # The correction warps: the unit from the CTA id against the base (no
    # value kept live from the setup), the unsplit path first (the branch
    # tests ``< 0``), the partial stores, and per stage one arrival per warp
    # whose last arriver resets the counter and bumps the generation word,
    # then every warp's rank for its softmax warp.
    assert (
        "flash_corr_tail_unit = cutlass.Int32(cute.arch.block_idx()[0]) - _flash_tail_base"
        in corr
    )
    assert corr.count("if flash_corr_tail_unit < cutlass.Int32(0):") == 2
    assert corr.index("fa4_correction_epilogue_to_smem_scoped(") < corr.index(
        "fa4_split_partial_store("
    )
    assert "flash_split_sid = flash_corr_tail_unit & cutlass.Int32(3)" in corr
    assert "flash_split_item = flash_corr_tail_unit >> cutlass.Int32(2)" in corr
    assert corr.count("fa4_split_partial_store(") == 2
    assert corr.count("fa4_correction_epilogue_to_smem_scoped(") == 2
    # One arrival after both stores; the generations read ahead of it; the
    # bumps and the ranks plain stores behind one gpu-scope fence.
    assert corr.count("= cutlass.Int32(cute.arch.atomic_add(") == 3  # 2 gens, 1 arrival
    assert corr.index(
        "fa4_split_partial_store(", corr.index("fa4_split_partial_store(") + 1
    ) < corr.index("flash_split_old = cutlass.Int32(cute.arch.atomic_add(")
    assert corr.count("cute.arch.fence_acq_rel_gpu()") == 1
    assert corr.count("if flash_split_old == cutlass.Int32(3):") == 3
    assert (
        "= flash_corr_gen0_0 + cutlass.Int32(1) << cutlass.Int32(12) | flash_split_sid"
        " << cutlass.Int32(8)"
    ) in corr
    assert (
        "= flash_corr_gen0_1 + cutlass.Int32(1) << cutlass.Int32(12) | flash_split_sid"
        " << cutlass.Int32(8)"
    ) in corr
    assert corr.count("] = flash_split_old + cutlass.Int32(1)") == 2
    assert "atomic_exch(" not in corr
    assert corr.count("flash_corr_gen0_0 = cutlass.Int32(cute.arch.atomic_add(") == 1
    assert (
        "if (flash_corr_tail_unit >= cutlass.Int32(0)) & (flash_local_tidx"
        " & cutlass.Int32(31) == 0):"
    ) in corr
    assert "fa4_split_merge_rows(" not in corr and "fa4_split_merge_store(" not in corr
    assert "while" not in corr
    # The MMA warp fences its P reads (``tcgen05.fence::after_thread_sync``)
    # after every wait on the softmax warps' P-ready barriers.
    mma = next(
        block
        for block in roles.values()
        if "flash_pfor_ptr + 1, flash_pfor_phase" in block
    )
    assert mma.count("mbar_spin_wait(flash_pfor") > 0
    assert mma.count("mbar_spin_wait(flash_pfor") == mma.count(
        "_helion_flash_rt.tcgen05_fence_after_thread_sync()"
    )
    # The softmax warps of each stage: the generation word read before the
    # final publish's barrier arrive; on a tail unit the wait for the unit's
    # own rank (the one wait, on the same CTA), the bounded poll of the
    # bump, the claim of the unit's slice and the last arriver's sweep of
    # the unclaimed ones; the LSE store on the other units.
    softmax = [block for block in roles.values() if "fa4_split_merge_rows(" in block]
    assert len(softmax) == 2
    for stage, block in enumerate(softmax):
        gen_read = "flash_split_gen0 = cutlass.Int32(cute.arch.atomic_add("
        assert block.count(gen_read) == 1
        assert block.index(gen_read) < block.rindex("named_barrier_arrive_unaligned(")
        assert f"* 2 + {stage}) * 4 + (flash_local_tidx >> cutlass.Int32(5))" in block
        # The generation read is lane 0's alone (an atomic per thread would
        # serialize at the L2 slice); the merge protocol follows the LSE.
        assert (
            "if (flash_sm_tail_unit >= cutlass.Int32(0)) & (flash_local_tidx"
            " & cutlass.Int32(31) == 0):"
        ) in block
        assert block.count("if flash_sm_tail_unit < cutlass.Int32(0):") == 1
        assert block.index("_flash_mLSE[") < block.index("fa4_split_merge_rows(")
        assert block.count("while flash_split_rank == 0:") == 1
        assert (
            "flash_split_last = cute.arch.vote_ballot_sync(flash_split_lastu != 0) != 0"
            in block
        )
        assert (
            "flash_split_word >> cutlass.Int32(8) & cutlass.Int32(15) == flash_split_sid"
            in block
        )
        assert (
            "while (flash_split_word >> cutlass.Int32(12) == flash_split_gen0)"
            " & (flash_split_polls < cutlass.Int32(1024)):"
        ) in block
        assert "if flash_split_rank == 4:" in block
        assert block.count("cute.arch.atomic_or(") == 3
        assert block.count("sem='acq_rel'") == 2
        assert block.count("sem='acquire'") == 3  # the rank reads and the sweep
        assert block.index("while (flash_split_word") < block.index(
            "while flash_split_rank == 0:"
        )
        assert "sem='relaxed', scope='gpu')) >> cutlass.Int32(12)" in block
        assert "cutlass.Int32(15)," in block
        assert "while flash_split_free != 0:" in block
        assert (
            "cute.arch.shuffle_sync(~flash_split_left & cutlass.Int32(15), 0)" in block
        )
        assert block.count("fa4_split_merge_rows(") == 2
        assert (
            f"flash_split_gO, flash_m_tile{stage} * 128, flash_live_rows{stage}),)"
            in block
        )
        assert block.count("_flash_mLSE[") == 1
    epi = next(block for block in roles.values() if "cute.copy(_flash_tma_o" in block)
    branch = epi.index("if cutlass.Int32(cute.arch.block_idx()[0]) < _flash_tail_base:")
    indent = branch - (epi.rfind("\n", 0, branch) + 1)
    assert epi.index("cute.copy(_flash_tma_o", branch) < epi.index(
        "\n" + " " * indent + "else:", branch
    )
    # The unit test reads the CTA id alone: the knob is pinned off wherever a
    # CTA walks more than one item or the launch is clustered.
    assert cute_flash._FLASH_TAIL_UNIT_PREDICATE == (
        "cutlass.Int32(cute.arch.block_idx()[0]) >= _flash_tail_base"
    )
    assert not cfg.persistent and not cfg.use_clc_scheduler and not cfg.use_2cta_instrs
    # The workspace holds one wave of items at most, with the arrival
    # counters, the generation words, the ranks and the slice claims behind
    # the partials and the statistics.
    from helion.runtime.cute.flash_flex import flex_split_workspace_elems

    items_ws = min(NUM_BH * 9, 256)
    partial, stats, counters = flex_split_workspace_elems(
        items_ws, 4, HEAD_DIM, cute_flash.flash_flex_tail_banks(4)
    )
    assert counters == items_ws * 2 * 4 * 6
    assert f"_flash_mSplitWs.iterator + {partial + stats}" in corr
    assert f"_flash_mSplitWs.iterator + {partial + stats}" in softmax[0]
    assert f"{items_ws * 2 * 4} + (flash_split_item * 2 + 1) * 4" in corr
    assert f"{items_ws * 2 * 4} + (flash_split_item * 2 + 1) * 4" in softmax[1]
    assert f"{2 * items_ws * 2 * 4} + ((flash_split_item * 2 + 0) * 4" in softmax[0]
    # Off: the plain grid, byte for byte.
    off = _render(_resolve(BODY), plan)
    assert "flash_unit_split" not in off and "_flash_tail_base" not in off
    # Pins.
    assert (
        _resolve(
            {
                **BODY,
                cute_flash.FLASH_PERSISTENT_KEY: True,
                cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4,
            }
        ).flex_tail_split
        == 0
    )
    assert (
        _resolve(
            {
                **BODY,
                cute_flash.FLASH_KV_SPLIT_KEY: 2,
                cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4,
            }
        ).flex_tail_split
        == 0
    )
    # The cursor walk (two words) keeps the knob: its roles consume the slice.
    assert (
        _resolve(
            {**BODY, cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4}, num_kv=40
        ).flex_tail_split
        == 4
    )
    assert (
        _resolve(
            {**BODY, cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4}, flex_trivial_walk=True
        ).flex_tail_split
        == 0
    )
    with pytest.raises(InvalidConfig):
        _resolve({**BODY, cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 3})
    # The fragment and its seed: 160 items on 148 SMs leave a tail of 12,
    # eight splits of which fill 96 SMs; 128 items (less than one wave) or
    # 296 (two full waves) seed nothing.
    with patch.dict(os.environ, {}, clear=True):
        fragments = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            num_bh=20,
            q_seq=2048,
            device_sm_count=148,
            dtype=torch.bfloat16,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
            plain_row_body=False,
        )
    fragment = fragments[cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY]
    assert fragment.default() == 8  # pyrefly: ignore[missing-attribute]
    assert set(fragment.search_choices) == {0, 2, 4, 8}  # pyrefly: ignore[missing-attribute]
    # The cursor walk (a words header): the same surface, seed and coverage
    # (its roles consume the slice's leading tiles through the cursor).
    with patch.dict(os.environ, {}, clear=True):
        cursor_fragments = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            40,
            num_bh=20,
            q_seq=2048,
            device_sm_count=148,
            dtype=torch.bfloat16,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
            plain_row_body=False,
        )
    cursor_fragment = cursor_fragments[cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY]
    assert cursor_fragment.default() == 8  # pyrefly: ignore[missing-attribute]
    assert set(cursor_fragment.search_choices) == {0, 2, 4, 8}  # pyrefly: ignore[missing-attribute]
    assert cursor_fragment.coverage_choices is None  # pyrefly: ignore[missing-attribute]
    assert fragment.coverage_choices is None  # pyrefly: ignore[missing-attribute]
    seed = cute_flash._flash_flex_tail_split_seed
    assert seed(device_sm_count=148, num_bh=16, q_seq=2048) == 0  # 128 items
    assert seed(device_sm_count=148, num_bh=37, q_seq=2048) == 0  # 296 items
    assert (
        seed(device_sm_count=148, num_bh=160, q_seq=448) == 4
    )  # whisper: 320 items, tail 24
    assert seed(device_sm_count=0, num_bh=160, q_seq=448) == 0


def test_tail_split_launch_follows_the_devices_sm_count() -> None:
    """The launcher derives the first tail unit from the SM count of the
    device it launches on (``flex_tail_base``, a runtime argument of the
    compiled kernel): a different count re-derives it, and a grid the SMs
    cannot hold split, a grid of one wave or less, a tail past the workspace
    or an unknown device keep every unit unsplit."""
    from helion.runtime.cute.flash_flex import flex_tail_base

    # whisper: 320 items, tail 24 on 148 SMs (4 units each fit), 304 on 152
    # SMs (16 x 4 = 64 units), one wave of 320 on a 320-SM device.
    assert flex_tail_base(320, 148, 4, 256) == 296
    assert flex_tail_base(320, 152, 4, 256) == 304
    assert flex_tail_base(320, 320, 4, 256) == 320
    assert flex_tail_base(320, 400, 4, 256) == 320
    # 100 tail items on 148 SMs cannot hold two units each.
    assert flex_tail_base(4096, 148, 2, 256) == 4096
    assert flex_tail_base(4096, 148, 1, 256) == 4096
    # A tail past the workspace's items, an unknown device.
    assert flex_tail_base(320, 148, 4, 16) == 320
    assert flex_tail_base(320, 0, 4, 256) == 320
    # Exactly divisible grids have no tail.
    assert flex_tail_base(296, 148, 4, 256) == 296


def test_tail_split_renders_on_every_multi_word_walk() -> None:
    """The knob is legal on a walk of more than one word too: the cursor
    walk (the host words table) consumes the slice through its cursor and
    the two-word register walk trims its words like the one-word walk, so
    the resolver keeps the value and the render carries the unit decode
    with either words form; on a one-word walk as before."""
    for num_kv in (33, 40, 64):
        plan = _flex_plan(q_seq=2048 + 128, kv_seq=num_kv * 128, mask_region=object())
        for item_words in (None, False, True):
            overrides = {**BODY, cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4}
            if item_words is not None:
                overrides[cute_flash.FLASH_FLEX_ITEM_WORDS_KEY] = item_words
            cfg = _resolve(overrides, num_kv=num_kv)
            assert cfg.flex_tail_split == 4, (num_kv, item_words)
            for words_table in (False, True):
                src = _render(cfg, plan, tail_split=4, words_table=words_table)
                assert "if flash_pid >= _flash_tail_base:" in src
                if words_table:
                    # The cursor walk: the roles consume the slice.
                    assert "flash_flex_left = flash_flex_lo" in src
                else:
                    # The register walk: the per-word trim.
                    assert "flash_flex_k1 = cutlass.Int32(0)" in src
    for num_kv in (16, 32):
        plan = _flex_plan(q_seq=2048 + 128, kv_seq=num_kv * 128, mask_region=object())
        for item_words in (False, True):
            cfg = _resolve(
                {
                    **BODY,
                    cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4,
                    cute_flash.FLASH_FLEX_ITEM_WORDS_KEY: item_words,
                },
                num_kv=num_kv,
            )
            assert cfg.flex_tail_split == 4
            for words_table in (False, True):
                src = _render(cfg, plan, tail_split=4, words_table=words_table)
                assert "if flash_pid >= _flash_tail_base:" in src


def test_persistent_list_walk_early_qk_hoists_the_next_items_first_qk() -> None:
    """``cute_flash_flex_early_qk`` (the persistent default): the MMA warp
    decodes the first item and issues its QK0 ahead of its loop, and inside
    the loop decodes the next item and issues its QK0 (after waiting for its
    Q0 / K0) between this item's last PV0 and last PV1, under ``if
    flash_tile_next < total``; the other roles keep the plain item loop.
    The resolver pins it off on the grid, the split-KV walk and when the
    config says so; the fragment is searched both ways on list walks."""
    cfg = _resolve({**BODY, cute_flash.FLASH_PERSISTENT_KEY: True})
    assert cfg.persistent and cfg.flex_early_qk
    plan = _flex_plan(q_seq=2048 + 128, mask_region=object())
    src = _render(cfg, plan)
    mma = _mma_block(src)
    lines = mma.splitlines()
    loop = next(i for i, line in enumerate(lines) if "while flash_tile_id <" in line)
    before, inside = "\n".join(lines[:loop]), "\n".join(lines[loop:])
    # The first item's decode, setup and QK0 ahead of the loop.
    assert "flash_item = cutlass.Int32(_flash_mFlexItems[flash_tile_id])" in before
    assert "flash_num_active_kv = cutlass.max(cutlass.Int32(1)" in before
    assert "flash_q0_full = flash_q_cons.wait_and_advance()" in before
    assert "flash_k0_full = flash_kv_cons.wait_and_advance()" in before
    assert "cute_tcgen05_flash.commit(flash_s_full_ptr + 0)" in before
    # Inside: Q1 first (the head already ran), the next item's head between
    # the last PV0's O commit and the last PV1's P wait.
    assert inside.index(
        "flash_q1_full = flash_q_cons.wait_and_advance()"
    ) < inside.index("flash_tile_next = flash_tile_id + flash_grid_dim")
    guard = "if flash_tile_next < 576:"  # 64 batch-heads x 9 pairs
    assert guard in inside
    o0 = inside.index("cute_tcgen05_flash.commit(flash_o_full_ptr + 0)")
    nxt = inside.index(guard)
    p1 = inside.index("mbar_spin_wait(flash_pfor_ptr + 1, flash_pfor_phase", o0)
    o1 = inside.index("cute_tcgen05_flash.commit(flash_o_full_ptr + 1)")
    assert o0 < nxt < p1 < o1
    next_block = inside[nxt:p1]
    assert (
        "flash_item = cutlass.Int32(_flash_mFlexItems[flash_tile_next])" in next_block
    )
    assert "flash_num_active_kv = cutlass.max(cutlass.Int32(1)" in next_block
    assert "flash_q0_full = flash_q_cons.wait_and_advance()" in next_block
    assert "cute_tcgen05_flash.commit(flash_s_full_ptr + 0)" in next_block
    assert inside.rstrip().endswith("flash_tile_id = flash_tile_next") or (
        "flash_tile_id = flash_tile_next" in inside
    )
    # The other roles keep the plain per-item loop (one setup per item).
    assert src.count("flash_num_active_kv = cutlass.max(cutlass.Int32(1)") == 6
    assert src.count("flash_tile_next") == mma.count("flash_tile_next")
    # Off: the previous item loop, byte for byte.
    off = _render(
        _resolve(
            {
                **BODY,
                cute_flash.FLASH_PERSISTENT_KEY: True,
                cute_flash.FLASH_FLEX_EARLY_QK_KEY: False,
            }
        ),
        plan,
    )
    assert "flash_tile_next" not in off
    # Pinned off on the grid and under the split-KV walk.
    assert not _resolve(
        {**BODY, cute_flash.FLASH_FLEX_EARLY_QK_KEY: True}
    ).flex_early_qk
    assert not _resolve(
        {
            **BODY,
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_KV_SPLIT_KEY: 2,
            cute_flash.FLASH_FLEX_EARLY_QK_KEY: True,
        }
    ).flex_early_qk
    with patch.dict(os.environ, {}, clear=True):
        fragments = cute_flash.flash_autotune_fragments(
            HEAD_DIM,
            16,
            num_bh=NUM_BH,
            dtype=torch.bfloat16,
            has_list_walk=True,
            score_modifier_kinds=(FLEX_SCORE_KIND,),
            plain_row_body=False,
        )
    fragment = fragments[cute_flash.FLASH_FLEX_EARLY_QK_KEY]
    assert fragment.default() is True  # pyrefly: ignore[missing-attribute]
    assert set(fragment.search_choices) == {False, True}  # pyrefly: ignore[missing-attribute]
    # Both values are witnessed on the persistent loop by the coverage
    # design (the grid pins the knob off, so a grid row carrying either
    # value renders the same kernel and says nothing about the loop).
    for value in (True, False):
        witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[
            (cute_flash.FLASH_FLEX_EARLY_QK_KEY, value)
        ]
        assert witness == {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            cute_flash.FLASH_PERSISTENT_KEY: True,
        }
        resolved = _resolve({**witness, cute_flash.FLASH_FLEX_EARLY_QK_KEY: value})
        assert resolved.persistent and resolved.flex_early_qk is value


def test_persistent_list_walk_early_qk_keeps_the_last_pv1_liveness_under_the_skip() -> (
    None
):
    """With ``cute_flash_flex_skip_absent`` the hoisted next-item head sets
    ``flash_flex_pv_live0 / 1`` for the NEXT item's first tile between this
    item's last PV0 and PV1. The last PV1's issue flag reads the item's own
    stage-1 liveness, snapshotted before the head: read off the walk's flag
    it was the next item's, and a last tile live for stage 1 followed by an
    item whose first tile is not lost its PV1 (O1 short of that tile)."""
    plan = _flex_plan(q_seq=2048 + 128, mask_region=object())
    cfg = _resolve_with_skip({**BODY, cute_flash.FLASH_PERSISTENT_KEY: True})
    assert cfg.persistent and cfg.flex_early_qk and cfg.flex_skip_absent
    src = _render(cfg, plan)
    mma = _mma_block(src)
    lines = mma.splitlines()
    loop = next(i for i, line in enumerate(lines) if "while flash_tile_id <" in line)
    inside = "\n".join(lines[loop:])
    snap = inside.index("flash_flex_pv_last1 = flash_flex_pv_live1")
    o0 = inside.index("cute_tcgen05_flash.commit(flash_o_full_ptr + 0)")
    nxt = inside.index("if flash_tile_next < 576:")
    live1 = inside.index("flash_flex_pv_live1 = cute.arch.make_warp_uniform(", nxt)
    issue1 = inside.index(
        "flash_flex_pv_issue1 = flash_flex_pv_last1 | cutlass.Int32(flash_o_zero1)"
    )
    o1 = inside.index("cute_tcgen05_flash.commit(flash_o_full_ptr + 1)")
    assert snap < o0 < nxt < live1 < issue1 < o1
    # Past the hoisted head's own carry, nothing before the O1 commit reads
    # the walk's flag.
    between = inside[live1 + 1 : o1].replace(
        "flash_flex_qk_live1 = flash_flex_pv_live1", ""
    )
    assert "flash_flex_pv_live1" not in between
    # The steady loop's PVs and the last PV0 keep the walk's flags.
    assert (
        inside.count(
            "flash_flex_pv_issue0 = flash_flex_pv_live0 | cutlass.Int32(flash_o_zero0)"
        )
        == 2
    )
    assert (
        inside.count(
            "flash_flex_pv_issue1 = flash_flex_pv_live1 | cutlass.Int32(flash_o_zero1)"
        )
        == 1
    )
    assert inside.count("flash_flex_pv_issue1 = flash_flex_pv_last1 |") == 1
    # The plain item loop takes the same snapshot (inert: nothing runs
    # between the two PVs there).
    off = _render(
        _resolve_with_skip(
            {
                **BODY,
                cute_flash.FLASH_PERSISTENT_KEY: True,
                cute_flash.FLASH_FLEX_EARLY_QK_KEY: False,
            }
        ),
        plan,
    )
    assert "flash_tile_next" not in off
    assert "flash_flex_pv_issue1 = flash_flex_pv_last1 |" in off


def _load_block(src: str) -> str:
    """The load warp's role block (the only role issuing the TMA Q loads)."""
    return next(
        block for block in _roles(src).values() if "cute.copy(_flash_tma_q" in block
    )


def test_persistent_list_walk_early_qk_loader_issues_q0_before_v0() -> None:
    """The early head waits for the next item's Q0 and K0 while the MMA
    warp's last PV1 still holds this item's last V slot, so the loader's
    per-item prologue must issue Q0 before any acquire that can block behind
    that slot: with the two-stage aliased K/V ring the next item's V0 takes
    it, and order 1 (K0, V0, Q0, Q1) deadlocked (the loader waiting on the
    MMA's PV1, the MMA on the loader's Q0). The resolver gives the early
    head order 3 (K0, Q0, V0, Q1) for every ring depth; the plain item loop
    and the grid keep order 1; the other orders already lead with Q0 or put
    it right behind K0."""
    key = cute_flash.FLASH_FIRST_LOAD_ORDER_KEY
    plan = _flex_plan(q_seq=2048 + 128, mask_region=object())
    early = _resolve({**BODY, cute_flash.FLASH_PERSISTENT_KEY: True, key: 1})
    assert early.persistent and early.flex_early_qk
    assert early.first_load_order == 3
    assert cute_flash.flash_effective_config_values(early)[key] == 3
    for kv_stage in (2, 3, 4):
        deep = _resolve(
            {
                **BODY,
                cute_flash.FLASH_PERSISTENT_KEY: True,
                cute_flash.FLASH_KV_STAGE_KEY: kv_stage,
                key: 1,
            }
        )
        assert deep.flex_early_qk and deep.first_load_order == 3
    plain = _resolve(
        {
            **BODY,
            cute_flash.FLASH_PERSISTENT_KEY: True,
            cute_flash.FLASH_FLEX_EARLY_QK_KEY: False,
            key: 1,
        }
    )
    assert not plain.flex_early_qk and plain.first_load_order == 1
    assert _resolve({**BODY, key: 1}).first_load_order == 1
    for order in (0, 2, 3, 4):
        kept = _resolve({**BODY, cute_flash.FLASH_PERSISTENT_KEY: True, key: order})
        assert kept.flex_early_qk and kept.first_load_order == order
    # The render: under the early head the loader's first Q copy precedes
    # its first V copy (K0 still first); the plain loop's order 1 issues V0
    # before Q0.
    for cfg in (
        early,
        *(
            _resolve({**BODY, cute_flash.FLASH_PERSISTENT_KEY: True, key: o})
            for o in (0, 2, 3, 4)
        ),
    ):
        load = _load_block(_render(cfg, plan))
        q0 = load.index("cute.copy(_flash_tma_q")
        v0 = load.index("cute.copy(_flash_tma_v")
        assert q0 < v0, cfg.first_load_order
    load = _load_block(_render(early, plan))
    assert load.index("cute.copy(_flash_tma_k") < load.index("cute.copy(_flash_tma_q")
    load_plain = _load_block(_render(plain, plan))
    assert load_plain.index("cute.copy(_flash_tma_v") < load_plain.index(
        "cute.copy(_flash_tma_q"
    )


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
            cute_flash.FLASH_PERSISTENT_KEY: False,
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


def test_persistence_stays_pinned_off_for_interval_plans_and_is_on_by_default_for_flex() -> (
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
    # The list walk defaults to the persistent strided loop over the
    # longest-first item order (every 2026-10-12 campaign winner) and still
    # searches both.
    fragment = fragments[cute_flash.FLASH_PERSISTENT_KEY]
    assert fragment.default() is True
    assert set(fragment.search_choices) == {False, True}  # pyrefly: ignore[missing-attribute]
    assert _resolve(BODY).persistent is False  # BODY pins it off
    assert (
        _resolve(
            {k: v for k, v in BODY.items() if k != cute_flash.FLASH_PERSISTENT_KEY}
        ).persistent
        is True
    )


# --------------------------------------------------------------------------
# The cursor walk over the host words table (two words or more)
# --------------------------------------------------------------------------


def _digits_normalised(src: str) -> str:
    return re.sub(r"\d+", "N", src)


def _load_role(roles: dict[str, str]) -> str:
    return next(block for block in roles.values() if "flash_kv_first" in block)


def _cursor_render(
    num_kv: int, config: dict[str, object] | None = None, **plan_kwargs: object
) -> str:
    cfg = _resolve({**BODY, **(config or {})}, num_kv=num_kv)
    plan = _flex_plan(q_seq=2048, kv_seq=num_kv * 128, **plan_kwargs)  # pyrefly: ignore[bad-argument-type]
    return _render(cfg, plan, words_table=True, kv_split=cfg.kv_split)


def test_one_word_masks_keep_the_register_walk_over_the_words_table() -> None:
    """Up to 32 KV tiles the words table is read into registers as before:
    the header-less rows, ``flash_flex_u0``, no cursor."""
    src = _cursor_render(16)
    setup = _roles(src)["setup"]
    assert "_flash_mFlexWords[flash_flex_wrow + cutlass.Int32(3)]" in setup
    assert (
        "flash_flex_u0 = flash_flex_p0_0 | flash_flex_f0_0 | flash_flex_p1_0 | flash_flex_f1_0"
        in setup
    )
    assert "flash_flex_cw" not in src and "flash_flex_total" not in src


def test_cursor_walk_reads_the_item_header_and_scans_the_words() -> None:
    """From two words on every walking role moves a cursor over the item's
    words: the setup reads the header (population, highest and lowest word),
    the roles load the four list words of the current word when the union
    word is exhausted, take the highest set bit and read the status off the
    current words; no per-word registers or select chains."""
    src = _cursor_render(65)  # three words (svg-temporal s8320)
    roles = _roles(src)
    setup = roles["setup"]
    assert (
        "flash_flex_wrow = (flash_bh * cutlass.Int32(8) + flash_m_pair) * cutlass.Int32(16)"
        in setup
    )
    assert (
        "flash_flex_total = cutlass.Int32(_flash_mFlexWords[flash_flex_wrow])" in setup
    )
    assert (
        "flash_flex_hiw = cutlass.Int32(_flash_mFlexWords[flash_flex_wrow + cutlass.Int32(1)])"
        in setup
    )
    assert (
        "flash_flex_low = cutlass.Int32(_flash_mFlexWords[flash_flex_wrow + cutlass.Int32(2)])"
        in setup
    )
    assert "flash_flex_wbase = flash_flex_wrow + cutlass.Int32(4)" in setup
    assert (
        "flash_num_active_kv = cutlass.max(cutlass.Int32(1), flash_flex_total)" in setup
    )
    for marker in (
        "flash_flex_u0",
        "flash_flex_p0_0",
        "flash_flex_rem0",
        "flash_flex_kvword",
        "popc(",
    ):
        assert marker not in src, marker
    softmax0 = next(b for n, b in roles.items() if "warp_idx < 4" in n)
    loop = _softmax_loop(softmax0)
    assert "flash_flex_cw = flash_flex_hiw + cutlass.Int32(1)" in softmax0
    assert "while flash_flex_rem == 0 and flash_flex_cw > flash_flex_low:" in loop
    assert "flash_flex_cw = flash_flex_cw - cutlass.Int32(1)" in loop
    for index in range(4):
        assert (
            f"_flash_mFlexWords[flash_flex_wbase + cutlass.Int32({index * 3}) + flash_flex_cw]"
            in loop
        )
    assert (
        "flash_flex_rem = flash_flex_wp0 | flash_flex_wf0 | flash_flex_wp1 | flash_flex_wf1"
        in loop
    )
    assert (
        "flash_kv = cutlass.Int32(cutlass.select_(flash_flex_rem != 0, flash_flex_cw * "
        "cutlass.Int32(32) + cutlass.Int32(31) - cutlass.Int32(cute.arch.clz(flash_flex_rem)), "
        "cutlass.Int32(0)))"
    ) in loop
    # A fake step (nothing left) carries a zero bit: absent for every test.
    assert (
        "flash_flex_kvbit = cutlass.Int32(cutlass.select_(flash_flex_rem != 0, cutlass.Int32(1) << (flash_kv & cutlass.Int32(31)), cutlass.Int32(0)))"
        in loop
    )
    assert (
        "flash_flex_status = cutlass.Int32(cutlass.select_(flash_flex_wp0 & flash_flex_kvbit != 0, cutlass.Int32(2)"
        in loop
    )
    softmax1 = next(b for n, b in roles.items() if "warp_idx < 8" in n)
    assert (
        "cutlass.select_(flash_flex_wp1 & flash_flex_kvbit != 0, cutlass.Int32(2)"
        in _softmax_loop(softmax1)
    )
    load = _load_role(roles)
    assert "flash_kv_first = cutlass.Int32(cutlass.select_(flash_flex_rem != 0" in load
    assert "flash_kv_next = cutlass.Int32(cutlass.select_(flash_flex_rem != 0" in load
    # The MMA warp walks too when it skips absent stages.
    skip = _cursor_render(65, {cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: True})
    mma = _mma_block(skip)
    assert (
        mma.count("flash_kv_mma = cutlass.Int32(cutlass.select_(flash_flex_rem != 0")
        == 2
    )
    assert (
        "flash_flex_pv_live0 = cute.arch.make_warp_uniform(cutlass.Int32("
        "(flash_flex_wp0 | flash_flex_wf0) & flash_flex_kvbit != 0))"
    ) in mma


def test_cursor_walk_code_does_not_grow_with_the_kv_length() -> None:
    """16K and 128K keys (4 and 32 words) render the same code up to the
    numeric literals (strides, counts): no register set or select chain
    scales with the length."""
    short = _cursor_render(128)
    long = _cursor_render(1024)
    assert short.count("\n") == long.count("\n")
    assert _digits_normalised(short) == _digits_normalised(long)
    # The mask markers of a dense 128K causal item fit the same budget.
    assert "flash_flex_rem1" not in long and "flash_flex_u1" not in long


def test_cursor_walk_ascending_order_scans_up_and_takes_the_lowest_bit() -> None:
    src = _cursor_render(65, {cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending"})
    loop = _softmax_loop(next(b for n, b in _roles(src).items() if "warp_idx < 4" in n))
    assert "flash_flex_cw = flash_flex_low - cutlass.Int32(1)" in src
    assert "while flash_flex_rem == 0 and flash_flex_cw < flash_flex_hiw:" in loop
    assert "flash_flex_cw = flash_flex_cw + cutlass.Int32(1)" in loop
    assert "cute.arch.clz(flash_flex_rem & cutlass.Int32(0) - flash_flex_rem)" in loop


def test_cursor_walk_peeks_the_next_tile_on_temporaries() -> None:
    """The hook prefetch's peek scans ahead on copies of the cursor (the
    stub hook here prefetches nothing, so the lines are checked directly)."""
    lines = cute_flash._flash_flex_cursor_peek_lines(
        "", "flash_kv_peek", words=3, lists=4, descending=True
    )
    assert lines[0] == "flash_flex_pkw = flash_flex_cw"
    assert lines[1] == "flash_flex_prem = flash_flex_rem"
    assert (
        lines[2]
        == "while (flash_flex_prem == 0) and (flash_flex_pkw > flash_flex_low):"
    )
    assert lines[3] == "    flash_flex_pkw = flash_flex_pkw - cutlass.Int32(1)"
    assert lines[4].startswith(
        "    flash_flex_prem = cutlass.Int32(_flash_mFlexWords[flash_flex_wbase + cutlass.Int32(0) + flash_flex_pkw]) | "
    )
    assert lines[4].count("_flash_mFlexWords[") == 4
    assert lines[5].startswith(
        "flash_kv_peek = cutlass.Int32(cutlass.select_(flash_flex_prem != 0, flash_flex_pkw * cutlass.Int32(32)"
    )
    assert "flash_flex_cw" not in "".join(lines[1:])
    ascending = cute_flash._flash_flex_cursor_peek_lines(
        "", "flash_kv_peek", words=3, lists=2, descending=False
    )
    assert "flash_flex_pkw < flash_flex_hiw" in ascending[2]
    assert ascending[4].count("_flash_mFlexWords[") == 2


def test_cursor_walk_split_kv_drops_the_leading_tiles() -> None:
    """The split-KV walk slices the item by rank: the setup derives the
    slice from the header's population and every role's cursor consumes
    the ``lo`` tiles ahead of it at the top of the item."""
    src = _cursor_render(65, {cute_flash.FLASH_KV_SPLIT_KEY: 2})
    roles = _roles(src)
    setup = roles["setup"]
    # (``ast.unparse`` drops the parentheses: ``+`` binds tighter than ``>>``.)
    assert (
        "flash_flex_per = flash_flex_total + cutlass.Int32(1) >> cutlass.Int32(1)"
        in setup
    )
    assert (
        "flash_flex_lo = cutlass.min(flash_flex_total, flash_kv_split_id * flash_flex_per)"
        in setup
    )
    assert (
        "flash_flex_cnt = cutlass.min(flash_flex_total - flash_flex_lo, flash_flex_per)"
        in setup
    )
    assert (
        "flash_num_active_kv = cutlass.max(cutlass.Int32(1), flash_flex_cnt)" in setup
    )
    assert "flash_flex_k0" not in src and "flash_flex_keep" not in src
    softmax = [
        b for n, b in roles.items() if "warp_idx < 4" in n or "warp_idx < 8" in n
    ]
    assert len(softmax) == 2
    for block in (_load_role(roles), *softmax):
        # The leading tiles go whole words at a time by population, the
        # last word bit by bit (O(words + 32), not O(lo)).
        assert "flash_flex_left = flash_flex_lo" in block
        assert "while flash_flex_left > 0:" in block
        assert "flash_flex_wc = cutlass.Int32(cute.arch.popc(flash_flex_rem))" in block
        assert "if flash_flex_wc <= flash_flex_left:" in block
        assert "for flash_flex_i in cutlass.range(flash_flex_left, unroll=1):" in block
        assert "cutlass.range(flash_flex_lo" not in block


def test_tail_split_slices_the_cursor_walk_at_run_time() -> None:
    """``cute_flash_flex_tail_split`` on the cursor walk: the setup derives
    the slice from the header's population with the unit's own shift (0 on
    an unsplit unit: the whole walk), every walking role's cursor consumes
    the ``lo`` leading tiles at the top of the item (whole words by
    population, the last word bit by bit; nothing on an unsplit unit), the
    unit decode and the per-unit merge are the register walk's; the
    resolver no longer pins the knob off past one word."""
    cfg = _resolve({**BODY, cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4}, num_kv=65)
    assert cfg.flex_tail_split == 4 and cfg.kv_split == 1
    plan = _flex_plan(q_seq=2048 + 128, kv_seq=65 * 128, mask_region=object())
    src = _render(cfg, plan, words_table=True, kv_split=cfg.kv_split, tail_split=4)
    roles = _roles(src)
    setup = roles["setup"]
    assert "flash_flex_cw" in src and "flash_flex_total = cutlass.Int32(" in setup
    assert "if flash_pid >= _flash_tail_base:" in setup
    assert "flash_kv_split_shift = cutlass.Int32(2)" in setup
    # (``ast.unparse`` drops the parentheses: ``+`` binds tighter than ``>>``.)
    assert (
        "flash_flex_per = flash_flex_total + ((cutlass.Int32(1) << flash_kv_split_shift)"
        " - cutlass.Int32(1)) >> flash_kv_split_shift"
    ) in setup
    assert (
        "flash_flex_lo = cutlass.min(flash_flex_total, flash_kv_split_id * flash_flex_per)"
        in setup
    )
    assert (
        "flash_num_active_kv = cutlass.max(cutlass.Int32(1), flash_flex_cnt)" in setup
    )
    # No register-walk trim: the cursor consumes the slice instead.
    assert "flash_flex_k0" not in src and "flash_flex_keep" not in src
    softmax = [
        b for n, b in roles.items() if "warp_idx < 4" in n or "warp_idx < 8" in n
    ]
    assert len(softmax) == 2
    for block in (_load_role(roles), *softmax):
        assert "flash_flex_left = flash_flex_lo" in block
        assert "while flash_flex_left > 0:" in block
        assert "flash_flex_wc = cutlass.Int32(cute.arch.popc(flash_flex_rem))" in block
    # Off, the cursor render is the plain one.
    plain_cfg = _resolve(BODY, num_kv=65)
    assert plain_cfg.flex_tail_split == 0
    plain = _render(plain_cfg, plan, words_table=True, kv_split=1)
    assert "flash_flex_left" not in plain and "_flash_tail_base" not in plain


def test_cursor_walk_dead_split_reads_the_stage_flags() -> None:
    """The dead-stage split decides from the header's stage flags (one CTA
    per item) or from a pass of the cursor over the slice (split-KV); the
    MMA warp's per-stage presence comes off the cursor's words."""
    dead = {
        cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: True,
        cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY: True,
    }
    src = _cursor_render(65, dead, mask_region=object())
    setup = _roles(src)["setup"]
    assert (
        "flash_flex_flags = cutlass.Int32(_flash_mFlexWords[flash_flex_wrow + cutlass.Int32(3)])"
        in setup
    )
    assert (
        "flash_flex_split = cutlass.Int32(flash_flex_flags & cutlass.Int32(2) == 0) & "
        "cutlass.Int32(flash_num_active_kv >= 2)"
    ) in setup
    mma = _mma_block(src)
    assert (
        "flash_flex_n0b = cutlass.Int32((flash_flex_wp0 | flash_flex_wf0) & flash_flex_kvbit != 0)"
        in mma
    )
    assert (
        "flash_flex_n1b = cutlass.Int32((flash_flex_wp1 | flash_flex_wf1) & flash_flex_kvbit != 0)"
        in mma
    )
    assert (
        "flash_flex_status_alt = cutlass.Int32(cutlass.select_(flash_flex_wp0 & flash_flex_kvbit != 0"
        in src
    )
    split = _cursor_render(
        65, {**dead, cute_flash.FLASH_KV_SPLIT_KEY: 2}, mask_region=object()
    )
    setup = _roles(split)["setup"]
    assert "flash_flex_flags" not in setup
    # The slice is scanned whole words at a time (the stage's bits against
    # the word's remaining union), the last word bit by bit.
    assert "flash_flex_left = flash_flex_cnt" in setup
    assert "cutlass.range(flash_flex_cnt" not in setup
    assert (
        "flash_flex_any1 = flash_flex_any1 | cutlass.Int32((flash_flex_wp1 | flash_flex_wf1) & flash_flex_rem != 0)"
        in setup
    )
    assert (
        "flash_flex_any1 = flash_flex_any1 | cutlass.Int32((flash_flex_wp1 | flash_flex_wf1) & flash_flex_kvbit != 0)"
        in setup
    )
    assert (
        "flash_flex_split = cutlass.Int32(flash_flex_any1 == 0) & cutlass.Int32(flash_num_active_kv >= 2)"
        in setup
    )


def test_register_words_are_pinned_off_past_two_words() -> None:
    """``cute_flash_flex_item_words`` is searched (both values) up to two
    words and pinned on beyond: the device-built register masks would grow
    with the KV length."""
    key = cute_flash.FLASH_FLEX_ITEM_WORDS_KEY
    for num_kv, choices in (
        (16, (False, True)),
        (64, (False, True)),
        (65, (True,)),
        (1024, (True,)),
    ):
        with patch.dict(os.environ, {}, clear=True):
            fragments = cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                num_kv,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=True,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
                standard_dense_output=False,
                standard_causal_output=False,
            )
        fragment = fragments[key]
        active = (
            fragment.choices
            if fragment.search_choices is None
            else fragment.search_choices
        )  # pyrefly: ignore[missing-attribute]
        assert set(active) == set(choices), (num_kv, fragment)
    assert _resolve({**BODY, key: False}, num_kv=64).flex_item_words is False
    assert _resolve({**BODY, key: False}, num_kv=65).flex_item_words is True


# --------------------------------------------------------------------------
# One PASS2 per step when the masked cadence inherits; the long-walk seed
# --------------------------------------------------------------------------


def test_masked_and_full_tiles_share_one_pass2_when_the_cadence_inherits() -> None:
    """The masked PASS1 stores the masked scores back, so PASS2 is the same
    call for both kinds of tile: it is emitted once after the PASS1 branch
    (two inlined copies doubled the softmax loop's hot code; ~1% on causal
    128K hd128). The resolver pins the masked cadence to ``inherit`` on the
    list walk, so the two calls are always equal there; the dedupe compares
    the rendered calls so a future distinct masked cadence keeps both."""
    src = _render(_resolve(BODY), _flex_plan(mask_region=object()))
    for name in ("warp_idx < 4", "warp_idx < 8"):
        loop = _softmax_loop(next(b for n, b in _roles(src).items() if name in n))
        assert (
            loop.count("flash_p_sum = _helion_flash_rt.fa4_disc_exp_convert_store") == 1
        )
        assert loop.count("if flash_flex_needs_mask:") == 1  # PASS1 only
        # The masked branch still runs its own PASS1 (the hook + store-back).
        assert "flash_tiled_score_st" in loop


def test_long_walk_seed_is_in_the_list_walk_population() -> None:
    """The causal 128K hd128 winner's structure (one CTA per pair item, a
    two-slot ring, the 16/2 cadence, Rep16 loads/stores, 168/80/32
    registers, ascending order) is a seed of every list-walk search."""
    seeds = _flex_seeds(1024)
    long = [
        cfg
        for cfg in seeds
        if not cfg.persistent
        and cfg.kv_stage == 2
        and cfg.e2e_schedule == "16/2"
        and cfg.s_load_repetition == 16
        and cfg.p_store_repetition == 16
        and cfg.causal_kv_order == "ascending"
        and (cfg.softmax_regs, cfg.corr_regs, cfg.other_regs) == (168, 80, 32)
    ]
    # Once with the per-stage softmax regions, once with the shared region
    # (whose one exp cadence ties stage 0's offset to stage 1's).
    assert sorted(cfg.softmax_region for cfg in long) == ["per_stage", "shared"], [
        (cfg.persistent, cfg.kv_stage, cfg.e2e_schedule) for cfg in seeds
    ]
    for cfg in long:
        assert cfg.disc_pipe_depth == 1 and cfg.split_p_arrive
        assert cfg.rescale_threshold == 8.0 and cfg.exp2_packet == "4x2"
    per_stage, shared = sorted(long, key=lambda cfg: cfg.softmax_region)
    assert (per_stage.e2e_offset0, per_stage.e2e_offset) == (9, 2)
    assert (shared.e2e_offset0, shared.e2e_offset) == (2, 2)


def test_pass2c_walk_seed_is_in_the_list_walk_population() -> None:
    """The pass-2c causal 128K hd128 winner's structure (a three-slot ring,
    the 8/2 cadence, Rep16 loads/stores, the shared softmax region, the
    rolled chunk loop, 184/88/32 registers, 16-row correction tiles) is a
    seed of every list-walk search, once per pair item and once persistent
    (the sliding-window winner's walk form)."""
    seeds = _flex_seeds(1024)
    p2c = [
        cfg
        for cfg in seeds
        if cfg.kv_stage == 3
        and cfg.e2e_schedule == "8/2"
        and cfg.s_load_repetition == 16
        and cfg.p_store_repetition == 16
        and cfg.softmax_region == "shared"
        and cfg.flex_rolled_chunks
        and (cfg.softmax_regs, cfg.corr_regs, cfg.other_regs) == (184, 88, 32)
    ]
    assert sorted(cfg.persistent for cfg in p2c) == [False, True], [
        (cfg.persistent, cfg.kv_stage, cfg.e2e_schedule, cfg.softmax_region)
        for cfg in seeds
    ]
    for cfg in p2c:
        assert cfg.disc_pipe_depth == 1 and cfg.split_p_arrive
        assert cfg.rescale_threshold == 8.0 and cfg.exp2_packet == "4x2"
        assert cfg.corr_tile_size == 16 and cfg.causal_kv_order == "ascending"
        # One region runs one exp cadence: stage 0's offset follows stage 1's.
        assert (cfg.e2e_offset0, cfg.e2e_offset) == (1, 1)


def test_w_walk_seed_is_in_the_list_walk_population() -> None:
    """The shared-region autotune's causal 128K hd128 winner (a three-slot
    ring, the 16/6 cadence with 4x1 packets, Rep16 loads/stores, the shared
    softmax region, 152/88/56 registers, the descending order, a 32 rescale
    threshold) is a seed of every list-walk search, once per pair item."""
    seeds = _flex_seeds(1024)
    w = [
        cfg
        for cfg in seeds
        if cfg.kv_stage == 3
        and cfg.e2e_schedule == "16/6"
        and cfg.s_load_repetition == 16
        and cfg.p_store_repetition == 16
        and cfg.softmax_region == "shared"
        and (cfg.softmax_regs, cfg.corr_regs, cfg.other_regs) == (152, 88, 56)
    ]
    assert [cfg.persistent for cfg in w] == [False], [
        (cfg.persistent, cfg.kv_stage, cfg.e2e_schedule, cfg.softmax_region)
        for cfg in seeds
    ]
    (cfg,) = w
    assert cfg.exp2_packet == "4x1" and cfg.disc_pipe_depth == 1
    assert cfg.split_p_arrive and cfg.rescale_threshold == 32.0
    assert cfg.causal_kv_order == "descending" and cfg.corr_tile_size == 8
    assert (cfg.e2e_offset0, cfg.e2e_offset) == (13, 13)


# --------------------------------------------------------------------------
# One softmax region for both query stages (``cute_flash_softmax_region``)
# --------------------------------------------------------------------------

SHARED_REGION: dict[str, object] = {
    **BODY,
    cute_flash.FLASH_SOFTMAX_REGION_KEY: "shared",
}


def test_softmax_region_knob_is_legal_on_the_list_walk_only() -> None:
    """``cute_flash_softmax_region``: appended behind the resident knob;
    ``shared`` on the list walk's chunked fa4 body (with or without a score
    region: the hook's staging ring is indexed by the runtime stage), one
    CTA per pair (any P store form); per stage elsewhere; an unknown value
    is declined. One region runs one exp cadence: stage 0's offset follows
    stage 1's."""
    key = cute_flash.FLASH_SOFTMAX_REGION_KEY
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    # Appended right behind the resident knob (axes added later sit behind it).
    assert keys.index(key) == keys.index(cute_flash.FLASH_S_RESIDENT_KEY) + 1
    assert _resolve(BODY).softmax_region == "per_stage"
    shared = _resolve(SHARED_REGION)
    assert shared.softmax_region == "shared"
    tied = _resolve(
        {
            **SHARED_REGION,
            cute_flash.FLASH_E2E_OFFSET0_KEY: 5,
            cute_flash.FLASH_E2E_OFFSET_KEY: 1,
        }
    )
    assert (tied.e2e_offset0, tied.e2e_offset) == (1, 1)
    apart = _resolve(
        {
            **BODY,
            cute_flash.FLASH_E2E_OFFSET0_KEY: 5,
            cute_flash.FLASH_E2E_OFFSET_KEY: 1,
        }
    )
    assert (apart.e2e_offset0, apart.e2e_offset) == (5, 1)
    assert _resolve({**SHARED_REGION}, has_score_modifiers=True).softmax_region == (
        "shared"
    )
    assert _resolve(SHARED_REGION, has_list_walk=False).softmax_region == "per_stage"
    mixed = _resolve(
        {
            **SHARED_REGION,
            cute_flash.FLASH_P_STORE_REP_KEY: 32,
            cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
        }
    )
    assert mixed.softmax_region == "shared" and mixed.p_store_repetition == 32
    with pytest.raises(InvalidConfig, match="per_stage"):
        _resolve({**SHARED_REGION, key: "bogus"})


def _shared_softmax_roles(src: str) -> dict[str, str]:
    """The softmax role blocks of a render (per stage, or the shared one)."""
    return {
        name: block
        for name, block in _roles(src).items()
        if "warp_idx < 4" in name or "warp_idx < 8" in name
    }


def test_shared_softmax_region_renders_one_role_for_both_stages() -> None:
    """The shared region: one ``warp_idx < 8`` softmax role whose stage is a
    warp-uniform runtime value selecting the TMEM columns, the s_corr
    barrier pair, the P/S-full barriers, the named barrier ids and the
    query tile; the list walk reads the stage's status words through a
    select of the two stages' words. The per-stage render keeps its two
    roles and none of the runtime-stage bindings."""
    plan = _flex_plan(mask_region=object())
    per_stage = _render(_resolve(BODY), plan)
    assert len(_shared_softmax_roles(per_stage)) == 2
    assert "flash_stage" not in per_stage

    src = _render(_resolve(SHARED_REGION), plan)
    roles = _shared_softmax_roles(src)
    assert list(roles) == ["if warp_idx < 8:"], list(roles)
    role = roles["if warp_idx < 8:"]
    for line in (
        "flash_stage = cute.arch.make_warp_uniform(warp_idx // 4)",
        "tStSs = cute.make_tensor(flash_tmem_ptr + flash_stage * 128, tStS.layout)",
        "tLDtSs = flash_thr_lds.partition_S(tStSs)",
        "tStSs_P = cute.make_tensor(tStSs.iterator, flash_P_layout)",
        "tSTtSs = flash_thr_sts.partition_D(tStSs_P)",
        "flash_ss_corr_full_ptr = flash_s0_corr_full_ptr + flash_stage * 4",
        "flash_ss_corr_empty_ptr = flash_ss_corr_full_ptr + 2",
        "flash_m_tiles = flash_m_tile0 + flash_stage",
    ):
        assert line in role, line
    loop = _softmax_loop(role)
    for text in (
        "flash_s_full_ptr + flash_stage, flash_s_full_phase",
        "3 + flash_stage * 4 + warp_idx % 4, 64)",
        "flash_pfor_ptr + flash_stage",
        "flash_pfor2_ptr + flash_stage",
        "flash_ss_corr_empty_ptr + flash_s_corr_prod_index",
        "flash_flex_ps_0 & flash_flex_kvbit",
        "flash_tiled_lds, tLDtSs, tLDcS, flash_tiled_sts, tSTtSs, tSTcS",
    ):
        assert text in loop, text
    # The register words are the item's: their stage select sits in the
    # item prelude, before the loop, not per step.
    head = role[: role.index(loop)]
    for text in (
        (
            "flash_flex_ps_0 = cutlass.Int32(cutlass.select_(flash_stage == 0, "
            "flash_flex_p0_0, flash_flex_p1_0))"
        ),
        (
            "flash_flex_fs_0 = cutlass.Int32(cutlass.select_(flash_stage == 0, "
            "flash_flex_f0_0, flash_flex_f1_0))"
        ),
    ):
        assert text in head and text not in loop, text
    assert "flash_q_mma_tiles" not in role
    # The whole kernel carries one PASS2 call; the loop and the tail name no
    # per-stage TMEM view, barrier or query tile (the role head keeps the
    # shared descriptor setup's static views, unused).
    assert src.count("flash_p_sum = _helion_flash_rt.fa4_disc_exp_convert_store") == 1
    tail = role[role.index(loop) + len(loop) :]
    for name in (
        "tLDtS0",
        "tLDtS1",
        "tSTtS0",
        "tSTtS1",
        "flash_s0_corr",
        "flash_s1_corr",
    ):
        assert name not in loop, name
        assert name not in tail, name
    assert "flash_m_tiles * 128 + flash_local_tidx" in tail  # the LSE rows
    assert "flash_ss_corr_empty_ptr + " in tail


def test_shared_softmax_region_selects_the_cursor_words_per_step() -> None:
    """On the host words table (the cursor walk) the shared region picks its
    stage's current partial / full words off the cursor's four registers at
    every step, and the tail wait takes the stage's s_corr barrier."""
    cfg = _resolve(SHARED_REGION, num_kv=128)
    assert cfg.softmax_region == "shared" and cfg.flex_item_words
    src = _render(cfg, _flex_plan(kv_seq=16384, mask_region=object()), words_table=True)
    roles = _shared_softmax_roles(src)
    assert list(roles) == ["if warp_idx < 8:"]
    role = roles["if warp_idx < 8:"]
    loop = _softmax_loop(role)
    assert (
        "flash_flex_wps = cutlass.Int32(cutlass.select_(flash_stage == 0, "
        "flash_flex_wp0, flash_flex_wp1))" in loop
    )
    assert (
        "flash_flex_wfs = cutlass.Int32(cutlass.select_(flash_stage == 0, "
        "flash_flex_wf0, flash_flex_wf1))" in loop
    )
    assert "flash_flex_wps & flash_flex_kvbit" in loop
    tail = role[role.index(loop) + len(loop) :]
    assert "flash_ss_corr_empty_ptr + " in tail
    assert "flash_s0_corr_empty_ptr" not in tail and "flash_s1_corr" not in tail


def test_shared_softmax_region_renders_the_trivial_walk_without_stage_words() -> None:
    """Torch's empty BlockMask (``kv_walk.trivial``: one KV block over the
    sequence) builds no bitmask words, so the shared item prelude binds the
    stage's query tile only: a select of the two stages' words would name
    registers the kernel never defines (a trace-time NameError)."""
    cfg = _resolve(SHARED_REGION, flex_trivial_walk=True)
    assert cfg.softmax_region == "shared"
    for mask_region, score_region in ((object(), None), (None, object())):
        plan = _flex_plan(mask_region=mask_region, score_region=score_region)
        assert plan.kv_walk is not None
        plan = dataclasses.replace(
            plan, kv_walk=dataclasses.replace(plan.kv_walk, trivial=True)
        )
        assert not plan.has_list_walk
        src = _render(cfg, plan)
        roles = _shared_softmax_roles(src)
        assert list(roles) == ["if warp_idx < 8:"], list(roles)
        role = roles["if warp_idx < 8:"]
        assert "flash_m_tiles = flash_m_tile0 + flash_stage" in role
        assert "cutlass.select_(flash_stage == 0, flash_flex_" not in src
        names = [
            node for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Name)
        ]
        stored = {node.id for node in names if isinstance(node.ctx, ast.Store)}
        loaded = {node.id for node in names if isinstance(node.ctx, ast.Load)}
        assert not {name for name in loaded - stored if name.startswith("flash_flex_")}


# --------------------------------------------------------------------------
# The hooked PASS1's all-kept fast path (``cute_flash_flex_hook_fast_full``)
# --------------------------------------------------------------------------


def test_flex_hook_fast_full_knob_is_legal_on_the_list_walk() -> None:
    """``cute_flash_flex_hook_fast_full``: appended last; on the list walk
    (any walk form), off by default and off the list walk."""
    key = cute_flash.FLASH_FLEX_HOOK_FAST_FULL_KEY
    # Appended after the flex axes (new axes go last; later knobs follow it).
    assert key in cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert _resolve(BODY).flex_hook_fast_full is False
    assert _resolve({**BODY, key: True}).flex_hook_fast_full is True
    assert (
        _resolve({**BODY, key: True}, has_list_walk=False).flex_hook_fast_full is False
    )


def _assert_fast_full_only_inserts(
    base: str, fast: str, blocks: list[list[str]]
) -> None:
    """``fast`` is ``base`` with each block (stripped lines) inserted whole,
    once, in order; nothing else differs."""
    fast_lines = [line.strip() for line in fast.splitlines()]
    base_lines = [line.strip() for line in base.splitlines()]
    cursor = 0
    for block in blocks:
        size = len(block)
        position = next(
            i
            for i in range(cursor, len(fast_lines) - size + 1)
            if fast_lines[i : i + size] == block
        )
        del fast_lines[position : position + size]
        cursor = position
    assert fast_lines == base_lines


def _fast_full_vote_block(stage: int, lost: str, vote: str) -> list[str]:
    return [
        "if flash_flex_needs_mask:",
        "flash_flex_masked = cutlass.Boolean(False)",
        "for flash_ci in cutlass.range_constexpr(flash_LD_CHUNKS):",
        f"flash_flex{stage}m_w0 = cutlass.Int32(stub_word(flash_ci))",
        f"flash_flex_masked = flash_flex_masked | ({lost.format(stage=stage)})",
        f"flash_flex_needs_mask = {vote}",
    ]


_FAST_FULL_VOTE = "cutlass.Boolean(cute.arch.vote_any_sync(flash_flex_masked))"


def test_flex_hook_fast_full_narrows_needs_mask_by_one_warp_vote_per_tile() -> None:
    """With the fast path a masked tile's prologue builds every chunk's keep
    word (index math, no score reads), ORs the chunks a lane loses a column
    of, and narrows ``flash_flex_needs_mask`` by one warp vote, so a warp
    whose rows keep the whole tile takes the step's dense path (PASS1 and
    PASS2 as rendered); nothing else of the render changes (pure insertions,
    one block per stage, right after the S wait and ahead of the loads), in
    the two-pass and the resident PASS2 forms; a ragged KV length keeps
    its last tile on the masked path; a score region keeps the plain hook;
    the default render is unchanged."""
    key = cute_flash.FLASH_FLEX_HOOK_FAST_FULL_KEY
    plan = _flex_plan(mask_region=object())
    base = _render(_resolve(BODY), plan)
    assert "flash_flex_masked" not in base
    assert _render(_resolve({**BODY, key: False}), plan) == base
    fast = _render(_resolve({**BODY, key: True}), plan)
    # (``ast.unparse`` drops the parentheses: ``&`` binds tighter than ``!=``.)
    lost16 = "flash_flex{stage}m_w0 & cutlass.Int32(65535) != cutlass.Int32(65535)"
    _assert_fast_full_only_inserts(
        base,
        fast,
        [
            _fast_full_vote_block(0, lost16, _FAST_FULL_VOTE),
            _fast_full_vote_block(1, lost16, _FAST_FULL_VOTE),
        ],
    )
    for name in ("warp_idx < 4", "warp_idx < 8"):
        loop = _softmax_loop(next(b for n, b in _roles(fast).items() if name in n))
        lines = [line.strip() for line in loop.splitlines()]
        status = lines.index("flash_flex_needs_mask = flash_flex_status == 2")
        wait = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("_helion_flash_rt.mbar_spin_wait(flash_s_full_ptr")
        )
        vote = lines.index(f"flash_flex_needs_mask = {_FAST_FULL_VOTE}")
        first_load = next(
            i for i, line in enumerate(lines) if line.startswith("flash_ld_shape = ")
        )
        assert status < wait < vote < first_load
    # The resident PASS2 form (fused PASS1/PASS2 branches, 32-column chunks).
    resident_body = {
        **BODY,
        cute_flash.FLASH_S_LOAD_REP_KEY: 32,
        cute_flash.FLASH_S_RESIDENT_KEY: True,
    }
    resident = _render(_resolve(resident_body), plan)
    assert "resident" in resident
    lost32 = "flash_flex{stage}m_w0 != cutlass.Int32(-1)"
    _assert_fast_full_only_inserts(
        resident,
        _render(_resolve({**resident_body, key: True}), plan),
        [
            _fast_full_vote_block(0, lost32, _FAST_FULL_VOTE),
            _fast_full_vote_block(1, lost32, _FAST_FULL_VOTE),
        ],
    )
    # A ragged KV length: the last tile always takes the masked path.
    tail_plan = _flex_plan(mask_region=object(), kv_seq=2000)
    tail_vote = f"{_FAST_FULL_VOTE} | (flash_kv == _flash_last_tile)"
    _assert_fast_full_only_inserts(
        _render(_resolve(BODY), tail_plan),
        _render(_resolve({**BODY, key: True}), tail_plan),
        [
            _fast_full_vote_block(0, lost16, tail_vote),
            _fast_full_vote_block(1, lost16, tail_vote),
        ],
    )
    # A score region: no fast path (the hook rewrites every element).
    score_plan = _flex_plan(mask_region=object(), score_region=object())
    assert _render(_resolve({**BODY, key: True}), score_plan) == _render(
        _resolve(BODY), score_plan
    )


def test_shared_staged_output_tile_renders_the_serialized_epilogue() -> None:
    """``cute_flash_o_stage`` = 1: the fa4 pair's two query stages share one
    staged output tile.  The storage takes one tile, the epilogue warp
    pre-arrives only "stage 1 left the tile", TMA-stores both stages from
    tile 0 and drains each store's smem reads before releasing the tile
    (stage 0's release gates stage 1's staging, stage 1's the next item's
    stage 0), and the correction warps wait the other stage's release before
    staging into tile 0.  Two tiles render byte-identically to the key
    absent; the resolver pins the shared tile off under a split-KV walk, a
    tail split and without the TMA epilogue."""
    base = {**BODY, cute_flash.FLASH_PERSISTENT_KEY: True}
    cfg = _resolve({**base, cute_flash.FLASH_O_STAGE_KEY: 1})
    assert cfg.o_stage == 1 and cfg.epi_tma and cfg.persistent
    plan = _flex_plan(mask_region=object())
    src = _render(cfg, plan)
    assert ", o_stage=1)" in src
    assert "sO[None, None, 1]" not in src and "tOsO_tma[None, 1]" not in src
    assert src.count("cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)") == 1
    assert src.count("cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)") == 2
    assert src.count("cute.arch.cp_async_bulk_wait_group(0, read=True)") == 2
    assert "cp_async_bulk_wait_group(1, read=True)" not in src
    # The epilogue warp: store 0, drain, release 0; store 1, drain, release 1.
    epi = re.search(
        r"tOsO_tma\[None, 0\], tOgO_tma\[None, flash_m_tile0, 0, flash_bh\]\)\n"
        r"\s*cute\.arch\.cp_async_bulk_commit_group\(\)\n"
        r"\s*cute\.arch\.cp_async_bulk_wait_group\(0, read=True\)\n"
        r"\s*cute\.arch\.mbarrier_arrive\(flash_corr_epi_empty_ptr \+ 0\)\n"
        r"(.|\n)*?tOsO_tma\[None, 0\], tOgO_tma\[None, flash_m_tile1, 0, flash_bh\]\)\n"
        r"\s*cute\.arch\.cp_async_bulk_commit_group\(\)\n"
        r"\s*cute\.arch\.cp_async_bulk_wait_group\(0, read=True\)\n"
        r"\s*cute\.arch\.mbarrier_arrive\(flash_corr_epi_empty_ptr \+ 1\)",
        src,
    )
    assert epi is not None
    # The correction warps: stage 0 behind the previous stage 1, stage 1
    # behind stage 0, both into tile 0.
    for stage, waits in (("0", "1"), ("1", "0")):
        assert re.search(
            rf"mbar_spin_wait\(flash_corr_epi_empty_ptr \+ {waits}, "
            r"flash_corr_epi_empty_phase, \d+\)\n"
            rf"\s*_helion_flash_rt\.fa4_correction_epilogue_to_smem_scoped\("
            rf"flash_pvt, tOtO{stage}, sO\[None, None, 0\]",
            src,
        ), stage
    # Off: byte-identical to the key absent.
    assert _render(
        _resolve({**base, cute_flash.FLASH_O_STAGE_KEY: 2}), plan
    ) == _render(_resolve(base), plan)
    assert (
        _resolve(
            {**base, cute_flash.FLASH_O_STAGE_KEY: 1, cute_flash.FLASH_KV_SPLIT_KEY: 2}
        ).o_stage
        == 2
    )
    assert (
        _resolve(
            {
                **BODY,
                cute_flash.FLASH_O_STAGE_KEY: 1,
                cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: 4,
            }
        ).o_stage
        == 2
    )
    assert (
        _resolve(
            {
                **base,
                cute_flash.FLASH_O_STAGE_KEY: 1,
                cute_flash.FLASH_EPI_TMA_KEY: False,
            }
        ).o_stage
        == 2
    )
    with pytest.raises(InvalidConfig):
        _resolve({**base, cute_flash.FLASH_O_STAGE_KEY: 3})
    # The asymmetric pair's uneven three-slot ring keeps a tile per stage.
    legal: dict[str, object] = {
        "topology": "fa4",
        "epi_tma": True,
        "epi_stg": False,
        "o_over_q": False,
        "s_stage": 2,
        "use_2cta_instrs": False,
        "separate_kv_rings": False,
        "has_row_epilogue": False,
        "row_bound_tiling": "global",
        "uneven_kv_ring": False,
        "kv_split": 1,
        "flex_tail_split": 0,
    }
    assert cute_flash._flash_o_stage_legal(**legal)
    assert not cute_flash._flash_o_stage_legal(**{**legal, "uneven_kv_ring": True})
    # The ragged query length's hybrid epilogue: each slot's issuing lane
    # drains its own TMA store and releases the tile in the same block (a
    # vector-stored or dead stage releases after its rows).
    ragged = _render(cfg, _flex_plan(q_seq=2048 + 128, mask_region=object()))
    assert ragged.count("cute.arch.cp_async_bulk_wait_group(0, read=True)") == 2
    assert "cp_async_bulk_wait_group(1, read=True)" not in ragged
    for stage in ("0", "1"):
        assert re.search(
            rf"if flash_live_rows{stage} == cutlass\.Int32\(128\):\n"
            r"\s*with cute\.arch\.elect_one\(\):\n"
            r"\s*cute\.copy\(_flash_tma_o, tOsO_tma\[None, 0\], [^\n]*\)\n"
            r"\s*cute\.arch\.cp_async_bulk_commit_group\(\)\n"
            r"\s*cute\.arch\.cp_async_bulk_wait_group\(0, read=True\)\n"
            rf"\s*cute\.arch\.mbarrier_arrive\(flash_corr_epi_empty_ptr \+ {stage}\)",
            ragged,
        ), stage
    assert ragged.count("cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 0)") == 2
    assert ragged.count("cute.arch.mbarrier_arrive(flash_corr_epi_empty_ptr + 1)") == 3


def test_shared_staged_output_tile_deepens_the_kv_ring_at_head_dim_128() -> None:
    """At head dim 128 the pair's TMA epilogue sits at the shared-memory cap
    (Q 64 KB + 3 K/V slots + two staged tiles = 227 KB): the shared tile
    lets the resolver keep a requested depth of 4, which two tiles clamp to
    3; the seeds' persistent structure carries the shared tile and the
    deeper ring there."""

    def resolve(config: dict[str, object]) -> cute_flash.FlashAttentionConfig:
        with patch.dict(os.environ, {}, clear=True):
            return cute_flash.resolve_flash_config(
                128,
                16,
                config,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=True,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
                standard_dense_output=False,
                standard_causal_output=False,
            )

    deep = {
        **BODY,
        cute_flash.FLASH_PERSISTENT_KEY: True,
        cute_flash.FLASH_KV_STAGE_KEY: 4,
    }
    shared = resolve({**deep, cute_flash.FLASH_O_STAGE_KEY: 1})
    assert (shared.o_stage, shared.kv_stage, shared.epi_tma) == (1, 4, True)
    two = resolve(deep)
    assert (two.o_stage, two.kv_stage) == (2, 3)
    with patch.dict(os.environ, {}, clear=True):
        seeds = [
            seed.config
            for seed in cute_flash.flash_attention_seed_configs(
                128,
                16,
                dtype=torch.bfloat16,
                num_bh=NUM_BH,
                has_list_walk=True,
                has_score_modifiers=True,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
            )
        ]
    persistent_seeds = [
        seed
        for seed in seeds
        if seed.get(cute_flash.FLASH_PERSISTENT_KEY)
        and seed.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == "fa4"
        and cute_flash.FLASH_O_STAGE_KEY in seed
    ]
    assert any(
        seed[cute_flash.FLASH_O_STAGE_KEY] == 1
        and seed[cute_flash.FLASH_KV_STAGE_KEY] == 4
        for seed in persistent_seeds
    )
    # No seed asks the shared tile for more than the slot it frees.
    assert all(
        int(seed[cute_flash.FLASH_KV_STAGE_KEY])  # type: ignore[call-overload]
        <= (4 if seed[cute_flash.FLASH_O_STAGE_KEY] == 1 else 3)
        for seed in persistent_seeds
    )
