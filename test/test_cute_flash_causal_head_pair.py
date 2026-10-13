"""``cute_flash_causal_head_pair``: the head-pair schedule of the grouped-query
causal fa4 body. The one-head pair CTA walks two adjacent 128-row query tiles
(2p, 2p+1) of one head, so stage 0 has 2p+1 KV tiles and stage 1 2p+2: every
CTA ends with a half step, the grid holds 128 CTAs per even length and the
K/V head is streamed once per pair of tiles. Paired by head, the two stages are
two consecutive query heads (2j, 2j+1) of one K/V head at the same tile t:
both walk t+1 tiles (no half step, no dead tile, both diagonals in step 0),
the K/V stream is shared like FA4's pack_gqa, the grid holds one CTA per
(head pair, tile) -- FA4-pack's length distribution -- and Q/O/LSE keep their
plain per-head layouts. Per row the arithmetic is unchanged.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import ALIBI_BIAS_KIND
from helion._compiler.cute.attention_plan import CAUSAL_MASK_KIND
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_CAUSAL_HEAD_PAIR_KEY
SKIP_KEY = cute_flash.FLASH_CAUSAL_SKIP_DEAD_TILE_KEY
STAGGER_KEY = cute_flash.FLASH_CAUSAL_STAGGER_KEY
ZERO_STORE = "_helion_flash_rt.fa4_disc_zero_store("
STAGE0_BOUND = "flash_num_active_kv0 = flash_num_active_kv - cutlass.Int32(1)"
MMA_FULL_LOOP = "for flash_i in cutlass.range(flash_num_active_kv - 1, unroll=1)"
CORR_FULL_LOOP = "for flash_kv in cutlass.range(flash_num_active_kv - 1, unroll=1)"

# The hd128 grouped-query winners' body (resident3 packet, rep32 loads) with
# the TMA output staged in the dead Q stages and the five-deep ring.
DESC_BODY: dict[str, object] = {
    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
    cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
    cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/6",
    cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/4",
    cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    cute_flash.FLASH_P_STORE_REP_KEY: 16,
    cute_flash.FLASH_DISC_PIPE_KEY: 2,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    cute_flash.FLASH_EPI_TMA_KEY: True,
    cute_flash.FLASH_O_OVER_Q_KEY: True,
    cute_flash.FLASH_KV_STAGE_KEY: 5,
}
# The same body with the campaign winners' three-slot ring (no stagger).
SHALLOW_BODY: dict[str, object] = {
    **DESC_BODY,
    cute_flash.FLASH_O_OVER_Q_KEY: False,
    cute_flash.FLASH_KV_STAGE_KEY: 3,
}
# The knob defaults off (it wins at s2048 and loses at s4096 / s8192 under the
# power cap); the paired bodies request it.
HP_BODY: dict[str, object] = {**DESC_BODY, KEY: True}
HP_SHALLOW_BODY: dict[str, object] = {**SHALLOW_BODY, KEY: True}


def _resolve(
    config: dict[str, object] | None,
    *,
    kv_group: int = 4,
    is_causal: bool = True,
    head_dim: int = 128,
    num_bh: int = 64,
    **kwargs: object,
) -> cute_flash.FlashAttentionConfig:
    options: dict[str, object] = {
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
        **kwargs,
    }
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.resolve_flash_config(
            head_dim,
            16,
            config,
            dtype=torch.bfloat16,
            num_bh=num_bh,
            is_causal=is_causal,
            kv_group=kv_group,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _fragments(
    *, kv_group: int = 4, is_causal: bool = True, head_dim: int = 128, **kwargs: object
) -> dict[str, ConfigSpecFragment]:
    options: dict[str, object] = {
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
        **kwargs,
    }
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            head_dim,
            16,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=is_causal,
            kv_group=kv_group,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


def _render(
    cfg: cute_flash.FlashAttentionConfig,
    *,
    kv_group: int = 4,
    causal: bool = True,
    head_dim: int = 128,
    num_bh: int = 64,
    has_lse: bool = True,
) -> str:
    plan = causal_score_plan(head_dim) if causal else dense_score_plan(head_dim)
    plan = dataclasses.replace(plan, kv_group=kv_group)
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=16,
        sequence_extent=16 * 128,
        num_bh=num_bh,
        total_tiles=num_bh * 8,
        cfg=cfg,
        has_lse=has_lse,
        io_dtype="cutlass.BFloat16",
        score_plan=plan,
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _role_blocks(code: str) -> dict[str, str]:
    """Split a render at its ``if warp_idx`` role guards (text before the
    first guard under ``"setup"``)."""
    blocks: dict[str, str] = {}
    lines = code.split("\n")
    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith(("if warp_idx", "if (warp_idx"))
    ]
    blocks["setup"] = "\n".join(lines[: starts[0]])
    for begin, end in zip(starts, [*starts[1:], len(lines)], strict=True):
        blocks[lines[begin]] = "\n".join(lines[begin:end])
    return blocks


def test_resolver_takes_the_knob_on_even_grouped_causal_fa4_bodies() -> None:
    for kv_group in (2, 4, 8):
        for body in (HP_BODY, HP_SHALLOW_BODY):
            on = _resolve(body, kv_group=kv_group)
            assert on.causal_head_pair is True, kv_group
            # Both stages sit at one tile: no dead tile to skip and nothing
            # to stagger, so both are pinned off (the identity carries all
            # three) even when requested.
            assert on.causal_skip_dead_tile is False
            assert on.causal_stagger is False
            forced = _resolve(
                {**body, SKIP_KEY: True, STAGGER_KEY: True}, kv_group=kv_group
            )
            assert forced == on
            values = cute_flash.flash_effective_config_values(forced)
            assert values[KEY] is True
            assert values[SKIP_KEY] is False
            assert values[STAGGER_KEY] is False
    # Off (the default, and explicitly), the deep-ring body staggers and the
    # shallow one skips, as before.
    for off in (_resolve(DESC_BODY), _resolve({**DESC_BODY, KEY: False})):
        assert off.causal_head_pair is False
        assert off.causal_stagger is True
        assert off.causal_skip_dead_tile is False
    for shallow_off in (
        _resolve(SHALLOW_BODY),
        _resolve({**SHALLOW_BODY, KEY: False}),
    ):
        assert shallow_off.causal_head_pair is False
        assert shallow_off.causal_stagger is False
        assert shallow_off.causal_skip_dead_tile is True
    # Two consecutive heads share a K/V head only for an even group: plain
    # attention and a group of three keep the one-head pair (and its knobs).
    for kv_group in (1, 3):
        plain = _resolve({**DESC_BODY, KEY: True}, kv_group=kv_group)
        assert plain.causal_head_pair is False, kv_group
        assert plain.causal_stagger is True
        assert cute_flash.flash_effective_config_values(plain)[KEY] is False
        assert plain == _resolve(DESC_BODY, kv_group=kv_group)

    # ``HELION_CUTE_FLASH_CAUSAL_HEAD_PAIR=1`` turns the default on where
    # legal (a plain plan stays off); the explicit key still wins.
    def resolve_env(kv_group: int, config: dict[str, object]) -> bool:
        return cute_flash.resolve_flash_config(
            128,
            16,
            config,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
            kv_group=kv_group,
        ).causal_head_pair

    with patch.dict(
        os.environ, {"HELION_CUTE_FLASH_CAUSAL_HEAD_PAIR": "1"}, clear=True
    ):
        assert resolve_env(4, DESC_BODY) is True
        assert resolve_env(1, DESC_BODY) is False
        assert resolve_env(3, DESC_BODY) is False
        assert resolve_env(4, {**DESC_BODY, KEY: False}) is False


def test_resolver_pins_the_knob_off_outside_its_body() -> None:
    # The TMA output store is what the per-stage O coordinates are written
    # for; the ascending walk, the dense rows, the pruned masks, the other
    # score transforms and the other families have no head pair.
    staged = {
        **DESC_BODY,
        cute_flash.FLASH_EPI_TMA_KEY: False,
        cute_flash.FLASH_EPI_STG_KEY: True,
        cute_flash.FLASH_O_OVER_Q_KEY: False,
        cute_flash.FLASH_KV_STAGE_KEY: 3,
        KEY: True,
    }
    stg = _resolve(staged)
    assert stg.epi_stg is True and stg.epi_tma is False
    assert stg.causal_head_pair is False
    assert stg.causal_skip_dead_tile is True
    ascending = _resolve(
        {
            **DESC_BODY,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending",
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: False,
            KEY: True,
        }
    )
    assert ascending.causal_kv_order == "ascending"
    assert ascending.causal_head_pair is False
    dense = _resolve({**DESC_BODY, KEY: True}, is_causal=False)
    assert dense.causal_head_pair is False
    pruned = _resolve(
        {**DESC_BODY, KEY: True},
        has_kv_tile_pruning=True,
        standard_causal_output=False,
    )
    assert pruned.causal_head_pair is False
    alibi = _resolve(
        {**DESC_BODY, KEY: True},
        has_score_modifiers=True,
        score_modifier_kinds=(CAUSAL_MASK_KIND, ALIBI_BIAS_KIND),
        standard_causal_output=False,
    )
    assert alibi.causal_head_pair is False
    ws = _resolve(
        {**DESC_BODY, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap", KEY: True}
    )
    assert ws.pipeline_family == "ws_overlap"
    assert ws.causal_head_pair is False
    for resolved in (stg, ascending, dense, pruned, alibi, ws):
        assert cute_flash.flash_effective_config_values(resolved)[KEY] is False


def test_resolver_pins_the_knob_off_on_the_resident_native_route() -> None:
    """The SM103 tmem-row-reduce target routes its promoted causal seed to the
    resident-native softmax, which the head pair is not measured on: the
    resolver pins the knob off there (one identity for one render), as it does
    the dead-tile skip and the stagger, and the emitter asserts it."""
    with patch.dict(os.environ, {}, clear=True):
        # The resident-native route is keyed on the degree-2 packet, which
        # only the fast_math surface offers.
        seed = cute_flash.flash_attention_seed_config(
            64,
            512,
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 3),
            kv_group=4,
            fast_math=True,
        )
        assert seed is not None
        assert seed.config[KEY] is False
        body = {
            key: value
            for key, value in seed.config.items()
            if key not in (KEY, SKIP_KEY, STAGGER_KEY)
        }
        body[KEY] = True
        # The head pair needs the TMA output store; the first target template
        # of the shape carries the direct store, so request the store the
        # pair is legal on (the resident route does not read it).
        body[cute_flash.FLASH_EPI_TMA_KEY] = True
        body[cute_flash.FLASH_EPI_STG_KEY] = False

        def resolve(
            capability: tuple[int, int] | None, **overrides: object
        ) -> cute_flash.FlashAttentionConfig:
            return cute_flash.resolve_flash_config(
                64,
                512,
                {**body, **overrides},
                dtype=torch.float16,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=capability,
                kv_group=4,
                fast_math=True,
            )

        native = resolve((10, 3))
        assert native.exp2_packet == cute_flash._FLASH_DEG2_EXP2_PACKET
        assert native.causal_head_pair is False
        for value in (False, True):
            resolved = resolve((10, 3), **{KEY: value})
            assert resolved.causal_head_pair is False
            assert cute_flash.flash_effective_config_values(resolved)[KEY] is False
            assert resolved == native
        # Without the resident route the same request pairs the heads.
        assert resolve((10, 0)).causal_head_pair is True
        assert resolve(None).causal_head_pair is True
        with pytest.raises(AssertionError):
            cute_flash.emit_flash_fa4_device_body(
                cast("DeviceFunction", None),
                head_dim=64,
                num_kv=512,
                sequence_extent=512 * 128,
                num_bh=4,
                total_tiles=1024,
                cfg=dataclasses.replace(native, causal_head_pair=True),
                has_lse=False,
                io_dtype="cutlass.Float16",
                score_plan=dataclasses.replace(causal_score_plan(64), kv_group=4),
                target_device_capability=(10, 3),
            )


def test_knob_is_an_appended_schema_axis_with_its_witness() -> None:
    # Appended once behind the axes it was introduced after; knobs added later
    # append behind it in turn, so the tail itself is not pinned.
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(STAGGER_KEY)
    witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, True)]
    assert witness == {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_EPI_TMA_KEY: True,
        cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
    }
    # The witness context is sufficient on both head sizes for a grouped
    # plan: it resolves to the knob's value.
    for head_dim in (64, 128):
        resolved = _resolve({**witness, KEY: True}, head_dim=head_dim)
        assert resolved.causal_head_pair is True, head_dim
        # Without the key the witness context takes the off default.
        assert _resolve(dict(witness), head_dim=head_dim).causal_head_pair is False
    for value in (False, True):
        resolved = _resolve({**DESC_BODY, KEY: value})
        assert cute_flash.flash_effective_config_values(resolved)[KEY] is value


def test_fragments_search_the_knob_on_even_grouped_causal_fa4_plans_only() -> None:
    for kv_group in (2, 4, 8):
        for head_dim in (64, 128):
            grouped = _fragments(kv_group=kv_group, head_dim=head_dim)[KEY]
            assert set(_active(grouped)) == {False, True}, (kv_group, head_dim)
            assert grouped.default() is False
    for kv_group in (1, 3):
        plain = _fragments(kv_group=kv_group)[KEY]
        assert _active(plain) == (False,), kv_group
        assert plain.default() is False
    dense = _fragments(is_causal=False)[KEY]
    assert _active(dense) == (False,)
    assert dense.default() is False
    assert _active(
        _fragments(
            has_score_modifiers=True,
            score_modifier_kinds=(CAUSAL_MASK_KIND, ALIBI_BIAS_KIND),
            standard_causal_output=False,
        )[KEY]
    ) == (False,)
    assert _active(
        _fragments(has_kv_tile_pruning=True, standard_causal_output=False)[KEY]
    ) == (False,)
    assert _active(_fragments(requires_ws_overlap=True)[KEY]) == (False,)


@pytest.mark.parametrize("env", (None, "0", "1"))
def test_fragment_default_is_the_resolver_default(env: str | None) -> None:
    environ = {} if env is None else {"HELION_CUTE_FLASH_CAUSAL_HEAD_PAIR": env}
    with patch.dict(os.environ, environ, clear=True):
        fragments = cute_flash.flash_autotune_fragments(
            128,
            16,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
            kv_group=4,
        )
        resolved = cute_flash.resolve_flash_config(
            128,
            16,
            DESC_BODY,
            dtype=torch.bfloat16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
            kv_group=4,
        )
    expected = env == "1"
    assert fragments[KEY].default() is expected
    assert resolved.causal_head_pair is expected


def test_decode_pairs_two_consecutive_heads_at_one_tile() -> None:
    on = _resolve(HP_BODY)
    code = _render(on)
    setup = _role_blocks(code)["setup"]
    # 64 batch heads of a group of four at 16 tiles: the L2-sized launch
    # group covers the grid (64 heads = 32 pairs), so one wave holds every
    # pair at one tile length (FA4-pack's distribution): pair fastest, tile
    # descending (LPT), and the work-item count equals the one-head grid's.
    assert "flash_lpt_group = flash_pid // 512" in setup
    assert "flash_lpt_mod = flash_pid % 512" in setup
    assert "flash_m_tile_raw = flash_lpt_mod // 32" in setup
    assert "flash_bh = (flash_lpt_group * 32 + flash_lpt_mod % 32) * 2" in setup
    assert "flash_m_tile0 = 15 - flash_m_tile_raw" in setup
    assert "flash_m_tile1 = flash_m_tile0" in setup
    assert "flash_bh0 = flash_bh" in setup
    assert "flash_bh1 = flash_bh + 1" in setup
    assert "flash_num_active_kv = flash_m_tile1 + cutlass.Int32(1)" in setup
    assert "flash_m_pair" not in code
    # K/V are the shared head's; each stage's Q is its own head's.
    assert "tKgK = tKgK_kdl[None, None, 0, flash_bh // 4]" in code
    assert "tVgV = tVgV_dkl[None, 0, None, flash_bh // 4]" in code
    assert "tQgQ0 = tQgQ_qdl[None, None, 0, flash_bh0]" in code
    assert "tQgQ1 = tQgQ_qdl[None, None, 0, flash_bh1]" in code
    assert "tQgQ = " not in code


def test_launch_group_is_converted_to_head_pairs() -> None:
    # An explicit group of S batch heads is S // 2 pairs; the serial head
    # order is one pair per group; a group the pairs do not divide into
    # takes FA4's residual branch.
    serial = _render(_resolve({**HP_BODY, cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 1}))
    assert "flash_lpt_group = flash_pid // 16" in serial
    assert "flash_m_tile_raw = flash_lpt_mod // 1" in serial
    assert "flash_bh = (flash_lpt_group * 1 + flash_lpt_mod % 1) * 2" in serial
    eight = _render(_resolve({**HP_BODY, cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 8}))
    assert "flash_lpt_group = flash_pid // 64" in eight
    assert "flash_m_tile_raw = flash_lpt_mod // 4" in eight
    assert "if flash_lpt_group <" not in eight
    residual = _render(
        _resolve({**HP_BODY, cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY: 32}, num_bh=48),
        num_bh=48,
    )
    # 24 pairs in groups of 16: one whole group, then a group of 8.
    assert "flash_lpt_group = flash_pid // 256" in residual
    assert "if flash_lpt_group < 1:" in residual
    assert "flash_m_tile_raw = flash_lpt_mod // 16" in residual
    assert "flash_m_tile_raw = flash_lpt_mod // 8" in residual
    assert "flash_bh = (flash_lpt_group * 16 + flash_lpt_mod % 8) * 2" in residual


def test_loads_stores_and_lse_rows_address_each_stage_head() -> None:
    code = _render(_resolve(HP_BODY))
    assert "tQgQ0[None, flash_q_mma_tile0]" in code
    assert "tQgQ1[None, flash_q_mma_tile1]" in code
    assert "tQgQ[" not in code
    assert "tOgO_tma[None, flash_m_tile0, 0, flash_bh0]" in code
    assert "tOgO_tma[None, flash_m_tile1, 0, flash_bh1]" in code
    assert "0, flash_bh]" not in code
    assert "_flash_mLSE[flash_m_tile0 * 128 + flash_local_tidx, flash_bh0]" in code
    assert "_flash_mLSE[flash_m_tile1 * 128 + flash_local_tidx, flash_bh1]" in code
    assert "flash_local_tidx, flash_bh]" not in code


def test_body_is_the_plain_walk_without_the_zero_store() -> None:
    on = _resolve(HP_BODY)
    code = _render(on)
    # Stage 0's first tile is its own diagonal (masked like stage 1's), not
    # the dead tile the zero store serves; the proven split gives both stages
    # one masked tile followed by t unmasked ones.
    assert ZERO_STORE not in code
    assert STAGE0_BOUND not in code
    assert "flash_v_prev" not in code
    assert MMA_FULL_LOOP in code
    assert CORR_FULL_LOOP in code
    for stage in ("0", "1"):
        assert (
            f"for flash_kv_mask_iter in cutlass.range(flash_num_active_kv - flash_m_tile{stage}, unroll=1)"
            in code
        )
        assert (
            f"for flash_kv_unmask_iter in cutlass.range(flash_m_tile{stage}, unroll=1)"
            in code
        )
    # Apart from the decode and the per-stage head coordinates the render is
    # the one-head body's with its skip and stagger off.
    plain = _render(
        _resolve({**DESC_BODY, KEY: False, STAGGER_KEY: False, SKIP_KEY: False})
    )
    assert ZERO_STORE in plain

    def _strip(text: str) -> str:
        return (
            text.replace("flash_bh0", "flash_bh")
            .replace("flash_bh1", "flash_bh")
            .replace("tQgQ0", "tQgQ")
            .replace("tQgQ1", "tQgQ")
        )

    on_lines = _strip(code).split("\n")
    plain_lines = plain.split("\n")
    # Lines only the one-head body has: its decode and stage 0's zero-store
    # guard (the guarded calls themselves reappear unguarded).
    plain_only = (
        "flash_m_pair",
        "flash_bh = flash_lpt_group",
        "flash_m_tile0 = flash_m_pair",
        "flash_m_tile1 = flash_m_pair",
        "if flash_kv_mask_iter != 0:",
        "else:",
        "flash_p_sum = _helion_flash_rt.fa4_disc_zero_store(",
    )
    # Lines only the head pair has: its decode and the second Q slice.
    pair_only = (
        "flash_m_tile_raw = ",
        "flash_bh = ",
        "flash_m_tile0 = 15 - flash_m_tile_raw",
        "flash_m_tile1 = flash_m_tile0",
        "tQgQ = tQgQ_qdl[None, None, 0, flash_bh]",
    )
    unguarded = (
        "flash_row_max = _helion_flash_rt.fa4_disc_rowmax_causal",
        "flash_p_sum = _helion_flash_rt.fa4_disc_exp_convert_store",
    )
    matcher = difflib.SequenceMatcher(None, plain_lines, on_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for line in plain_lines[i1:i2]:
            text = line.strip()
            assert text.startswith((*plain_only, *unguarded)), line
            if text.startswith(unguarded):
                assert text in {other.strip() for other in on_lines[j1:j2]}, line
        for line in on_lines[j1:j2]:
            text = line.strip()
            assert text.startswith((*pair_only, *unguarded)), line


def test_renders_without_the_knob_are_unchanged() -> None:
    # Plans the knob does not apply to render byte-identically whether or not
    # the key is requested: plain attention (both head sizes), a group of
    # three, dense grouped rows and the staged (STG) grouped causal body.
    cases: list[tuple[dict[str, object], dict[str, object]]] = [
        ({"kv_group": 1}, {}),
        ({"kv_group": 1, "head_dim": 64}, {}),
        ({"kv_group": 3}, {}),
        ({"kv_group": 4, "is_causal": False}, {}),
        (
            {"kv_group": 4},
            {
                cute_flash.FLASH_EPI_TMA_KEY: False,
                cute_flash.FLASH_EPI_STG_KEY: True,
                cute_flash.FLASH_O_OVER_Q_KEY: False,
                cute_flash.FLASH_KV_STAGE_KEY: 3,
            },
        ),
    ]
    for options, overrides in cases:
        body = {**DESC_BODY, **overrides}
        base = _resolve(body, **options)  # pyrefly: ignore[bad-argument-type]
        requested = _resolve({**body, KEY: True}, **options)  # pyrefly: ignore[bad-argument-type]
        assert base.causal_head_pair is False, options
        assert requested == base, options
        render_options = {
            "kv_group": options["kv_group"],
            "causal": options.get("is_causal", True),
            "head_dim": options.get("head_dim", 128),
        }
        rendered = _render(requested, **render_options)  # pyrefly: ignore[bad-argument-type]
        assert rendered == _render(base, **render_options)  # pyrefly: ignore[bad-argument-type]
        assert "flash_bh0" not in rendered


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epi_tma", False),
        ("causal_skip_dead_tile", True),
        ("causal_stagger", True),
        ("p_chunk_arrive", True),
        ("mma_ptx", False),
        ("q_tile_m", 64),
        ("alternating_warpgroups", True),
        ("pipeline_family", "fa4_deep_1cta"),
    ],
)
def test_emitter_rejects_the_knob_outside_the_resolver_legality(
    field: str, value: object
) -> None:
    on = _resolve(HP_BODY)
    assert on.causal_head_pair is True
    assert dataclasses.asdict(on)[field] != value
    with pytest.raises(AssertionError):
        _render(dataclasses.replace(on, **{field: value}))


def test_emitter_rejects_the_knob_on_an_ungrouped_plan() -> None:
    on = _resolve(HP_BODY)
    for kv_group in (1, 3):
        with pytest.raises(AssertionError):
            _render(on, kv_group=kv_group)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_grouped_causal_plans_seed_the_head_pair_structure(head_dim: int) -> None:
    # The target seed carries the resolved value (the off default), and
    # the head-pair seeds (per parent, with the shape's ring and with the
    # staggered seeds' deep ring) carry the descending split walk, the TMA
    # output store and the knob with the skip and the stagger off.
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            head_dim,
            16,
            num_bh=64,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 0),
            kv_group=4,
        )
        target = cute_flash.flash_attention_seed_config(
            head_dim,
            16,
            num_bh=64,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 0),
            kv_group=4,
        )
        assert target is not None
        assert target.config[KEY] is False
        # Seeds built from the fragment defaults carry the key's default
        # (True) and normalize to the body they describe, as the skip's and
        # the stagger's do; the head-pair seeds proper carry its structure.
        paired = [
            seed
            for seed in seeds
            if seed.config.get(KEY)
            and seed.config[cute_flash.FLASH_CAUSAL_KV_ORDER_KEY] == "descending"
            and seed.config[cute_flash.FLASH_EPI_TMA_KEY] is True
            and seed.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"
            and seed.config[cute_flash.FLASH_SOFTMAX_DISC_KEY] is True
        ]
        assert len(paired) >= 2
        depths = set()
        for seed in paired:
            resolved = cute_flash.resolve_flash_config(
                head_dim,
                16,
                {k: v for k, v in seed.config.items() if k != "block_sizes"},
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=(10, 0),
                kv_group=4,
            )
            assert resolved.causal_head_pair is True
            assert resolved.causal_skip_dead_tile is False
            assert resolved.causal_stagger is False
            assert resolved.causal_kv_order == "descending"
            assert resolved.causal_loop_split is True
            assert resolved.epi_tma is True
            depths.add(resolved.kv_stage)
        assert cute_flash._FLASH_CAUSAL_STAGGER_SEED_KV_STAGE[head_dim] in depths
        # Plain and odd-group plans seed no head pair, nor do dense ones.
        for kv_group in (1, 3):
            plain = cute_flash.flash_attention_seed_configs(
                head_dim,
                16,
                num_bh=64,
                dtype=torch.bfloat16,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=(10, 0),
                kv_group=kv_group,
            )
            assert not any(seed.config.get(KEY) for seed in plain), kv_group
        dense = cute_flash.flash_attention_seed_configs(
            head_dim,
            16,
            num_bh=64,
            dtype=torch.bfloat16,
            is_causal=False,
            standard_dense_output=True,
            target_device_capability=(10, 0),
            kv_group=4,
        )
        assert not any(seed.config.get(KEY) for seed in dense)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_opt_in_default_keeps_the_one_head_seed_bodies(head_dim: int) -> None:
    # ``HELION_CUTE_FLASH_CAUSAL_HEAD_PAIR=1`` turns the default on, which the
    # seeds built from the fragment defaults follow like the skip's and the
    # stagger's; the seeds that describe a one-head body (the staggered walk,
    # the packet seeds with the dead-tile skip, hd64's measured bodies and
    # hybrid seed) pin the knob off, so a search under the opt-in still
    # starts from them and no two seeds collapse onto one body.
    def bodies(environ: dict[str, str]) -> tuple[int, int, int, int, int]:
        identities = set()
        stagger = skip = pair = 0
        with patch.dict(os.environ, environ, clear=True):
            seeds = cute_flash.flash_attention_seed_configs(
                head_dim,
                16,
                num_bh=64,
                dtype=torch.bfloat16,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=(10, 0),
                kv_group=4,
            )
            for seed in seeds:
                resolved = cute_flash.resolve_flash_config(
                    head_dim,
                    16,
                    {k: v for k, v in seed.config.items() if k != "block_sizes"},
                    dtype=torch.bfloat16,
                    num_bh=64,
                    is_causal=True,
                    standard_causal_output=True,
                    target_device_capability=(10, 0),
                    kv_group=4,
                )
                identities.add(
                    tuple(
                        sorted(
                            cute_flash.flash_effective_config_values(resolved).items()
                        )
                    )
                )
                stagger += resolved.causal_stagger
                skip += resolved.causal_skip_dead_tile
                pair += resolved.causal_head_pair
        return len(seeds), len(identities), stagger, skip, pair

    off = bodies({})
    on = bodies({"HELION_CUTE_FLASH_CAUSAL_HEAD_PAIR": "1"})
    # One body per seed, the staggered and skip bodies and both head-pair
    # rings with the default off ...
    assert off[1] == off[0]
    assert off[2] >= 1 and off[3] >= 1 and off[4] >= 2
    # ... and the same seed count, the same one-head bodies and no collapse
    # with the opt-in default.
    assert on[:4] == off[:4]
    assert on[4] >= off[4]
