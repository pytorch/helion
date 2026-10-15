"""``cute_flash_causal_mask_storeback``: the fa4 chunked softmax applies the
causal mask once. PASS1 of a masked tile writes the masked fp32 scores back to
TMEM (the modifier body's mechanism) and PASS2 runs the plain exp helper on
them; the diagonal tile is latency-bound, so dropping PASS2's per-chunk
R2P/select chain shortens its critical path at bitwise identical output.
"""

from __future__ import annotations

import ast
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

KEY = cute_flash.FLASH_CAUSAL_MASK_STOREBACK_KEY
STORE_HELPER = "fa4_disc_rowmax_causal_store_balanced("
MASK_HELPER = "fa4_disc_rowmax_causal_balanced("

# The causal hd64 s2048 winner: the paired 16-column score-load body.
PAIR_BODY: dict[str, object] = {
    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
    cute_flash.FLASH_EXP2_PACKET_KEY: "4x1",
    cute_flash.FLASH_E2E_SCHEDULE_KEY: "8/2",
    cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "xu",
    cute_flash.FLASH_S_LOAD_REP_KEY: 16,
    cute_flash.FLASH_P_STORE_REP_KEY: 16,
    cute_flash.FLASH_DISC_PIPE_KEY: 1,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    cute_flash.FLASH_EPI_TMA_KEY: True,
}
# The rep32 software-pipelined body (the s4096 / hd128 winners' masked pass).
REP32_BODY: dict[str, object] = {
    **PAIR_BODY,
    cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
    cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "16/4",
    cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    cute_flash.FLASH_DISC_PIPE_KEY: 2,
}


def _resolve(
    config: dict[str, object] | None,
    *,
    is_causal: bool = True,
    head_dim: int = 64,
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
            num_bh=64,
            is_causal=is_causal,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _fragments(
    *,
    is_causal: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    environ: dict[str, str] | None = None,
    **kwargs: object,
) -> dict[str, ConfigSpecFragment]:
    options: dict[str, object] = {
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
        **kwargs,
    }
    with patch.dict(os.environ, environ or {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            64,
            16,
            dtype=dtype,
            num_bh=64,
            is_causal=is_causal,
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


def _render(cfg: cute_flash.FlashAttentionConfig, *, causal: bool = True) -> str:
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=64,
        num_kv=16,
        sequence_extent=16 * 128,
        num_bh=64,
        total_tiles=64 * 8,
        cfg=cfg,
        has_lse=False,
        io_dtype="cutlass.BFloat16",
        score_plan=causal_score_plan(64) if causal else dense_score_plan(64),
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_resolver_defaults_the_knob_on_for_the_plain_causal_fa4_disc_body() -> None:
    assert _resolve(PAIR_BODY).causal_mask_storeback is True
    assert _resolve(REP32_BODY).causal_mask_storeback is True
    assert _resolve({**PAIR_BODY, KEY: False}).causal_mask_storeback is False
    with patch.dict(
        os.environ, {"HELION_CUTE_FLASH_CAUSAL_MASK_STOREBACK": "0"}, clear=True
    ):
        assert (
            cute_flash.resolve_flash_config(
                64,
                16,
                PAIR_BODY,
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
            ).causal_mask_storeback
            is False
        )
    # Dense plans never mask; the modifier and pruned bodies already take the
    # generic store-back path; the whole-row ws_overlap body has no second pass.
    assert (
        _resolve({**PAIR_BODY, KEY: True}, is_causal=False).causal_mask_storeback
        is False
    )
    assert (
        _resolve(
            {**PAIR_BODY, KEY: True},
            has_kv_tile_pruning=True,
            standard_causal_output=False,
        ).causal_mask_storeback
        is False
    )
    assert (
        _resolve(
            {**PAIR_BODY, KEY: True},
            has_score_modifiers=True,
            score_modifier_kinds=(CAUSAL_MASK_KIND, ALIBI_BIAS_KIND),
            standard_causal_output=False,
        ).causal_mask_storeback
        is False
    )
    assert (
        _resolve(
            {**PAIR_BODY, KEY: True, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
        ).causal_mask_storeback
        is False
    )


def test_knob_is_an_appended_schema_axis_with_its_fa4_disc_dependency() -> None:
    # Appended once behind the axes it was introduced after (an inserted axis
    # would reshuffle every later axis's coverage rows); knobs added later
    # append behind it in turn, so the tail itself is not pinned.
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(cute_flash.FLASH_O_OVER_Q_KEY)
    witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, True)]
    assert witness == {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
    }
    # The witness context is sufficient: it resolves to the knob's value.
    assert _resolve({**witness, KEY: True}).causal_mask_storeback is True
    for value in (False, True):
        resolved = _resolve({**PAIR_BODY, KEY: value})
        assert cute_flash.flash_effective_config_values(resolved)[KEY] is value


def test_fragments_search_the_knob_on_plain_causal_fa4_plans_only() -> None:
    causal = _fragments()[KEY]
    assert set(_active(causal)) == {False, True}
    assert causal.default() is True
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


def test_fragments_pin_the_knob_when_the_searched_family_cannot_store_back() -> None:
    # A family override fixes the family axis: the 2-CTA causal pair and the
    # separate-ring body pin the knob (the resolver would normalize a True
    # proposal to False), the plain family searches it.
    for family, dtype in (
        ("fa4_2cta_causal", torch.float16),
        ("fa4_deep_1cta", torch.bfloat16),
        ("ws_overlap", torch.bfloat16),
    ):
        pinned = _fragments(pipeline_family_override=family, dtype=dtype)[KEY]
        assert _active(pinned) == (False,), family
        assert pinned.default() is False, family
    searched = _fragments(pipeline_family_override="fa4")[KEY]
    assert set(_active(searched)) == {False, True}
    assert searched.default() is True
    # Without an override the family axis is searched too and only the plain
    # fa4 family realizes the knob: the value dependency witnesses that pair.
    fragments = _fragments(dtype=torch.float16)
    families = _active(fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY])
    assert "fa4" in families and "fa4_2cta_causal" in families
    for family in families:
        resolved = cute_flash.resolve_flash_config(
            64,
            16,
            {**PAIR_BODY, cute_flash.FLASH_PIPELINE_FAMILY_KEY: family, KEY: True},
            dtype=torch.float16,
            num_bh=64,
            is_causal=True,
            standard_causal_output=True,
        )
        assert resolved.pipeline_family == family
        assert resolved.causal_mask_storeback is (family == "fa4"), family


@pytest.mark.parametrize("env", (None, "0", "1"))
def test_fragment_default_is_the_fa4_family_resolver_default(env: str | None) -> None:
    environ = {} if env is None else {"HELION_CUTE_FLASH_CAUSAL_MASK_STOREBACK": env}
    # (head_dim, dtype, plan options, searched on the plain causal fa4 body)
    surfaces: tuple[tuple[int, torch.dtype, dict[str, object], bool], ...] = (
        (64, torch.bfloat16, {"is_causal": True, "standard_causal_output": True}, True),
        (64, torch.float16, {"is_causal": True, "standard_causal_output": True}, True),
        (
            128,
            torch.bfloat16,
            {"is_causal": True, "standard_causal_output": True},
            True,
        ),
        (
            64,
            torch.bfloat16,
            {"is_causal": False, "standard_dense_output": True},
            False,
        ),
        (64, torch.bfloat16, {"is_causal": True, "has_kv_tile_pruning": True}, False),
        (
            64,
            torch.bfloat16,
            {
                "is_causal": True,
                "has_score_modifiers": True,
                "score_modifier_kinds": (CAUSAL_MASK_KIND, ALIBI_BIAS_KIND),
            },
            False,
        ),
    )
    with patch.dict(os.environ, environ, clear=True):
        for head_dim, dtype, options, searched in surfaces:
            fragment = cute_flash.flash_autotune_fragments(
                head_dim,
                16,
                dtype=dtype,
                num_bh=64,
                **options,  # pyrefly: ignore[bad-argument-type]
            )[KEY]
            # The fragment default is what a config without the key resolves
            # to on the plain fa4 family: the env kill switch reaches the
            # seeded and coverage configs through it.
            pinned_fa4 = cute_flash.resolve_flash_config(
                head_dim,
                16,
                {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"},
                dtype=dtype,
                num_bh=64,
                **options,  # pyrefly: ignore[bad-argument-type]
            )
            assert pinned_fa4.pipeline_family == "fa4"
            assert fragment.default() is pinned_fa4.causal_mask_storeback, options
            assert fragment.default() is (searched and env != "0"), options
            # The env moves the default only; the plain causal surface keeps
            # measuring both values.
            assert set(_active(fragment)) == ({False, True} if searched else {False}), (
                options
            )


@pytest.mark.parametrize(
    ("body", "store_rep", "plain_pass2", "masked_pass2"),
    [
        (
            PAIR_BODY,
            16,
            "fa4_disc_exp_convert_store_sload16_pair_pipe(",
            "fa4_disc_exp_convert_store_sload16_pair_pipe_causal(",
        ),
        (
            REP32_BODY,
            32,
            "fa4_disc_exp_convert_store_pipe(",
            "fa4_disc_exp_convert_store_pipe_causal(",
        ),
    ],
)
def test_split_body_masks_once_and_stores_the_scores_back(
    body: dict[str, object], store_rep: int, plain_pass2: str, masked_pass2: str
) -> None:
    off = _render(_resolve({**body, KEY: False}))
    on = _render(_resolve(body))
    assert on == _render(_resolve({**body, KEY: True}))
    # With the knob off both passes mask: PASS1 reduces in place and PASS2
    # runs the causal helper; no score store atom exists.
    assert off.count(MASK_HELPER) == 2
    assert off.count(masked_pass2) == 2
    assert STORE_HELPER not in off
    assert "flash_score_st_atom" not in off
    # With the knob PASS1 of both query slots masks, reduces and stores back
    # with a store atom matching the load chunking; PASS2 of the masked
    # segment is the plain helper (the unmasked suffix keeps its own).
    assert on.count(STORE_HELPER) == 2
    assert MASK_HELPER not in on
    assert masked_pass2 not in on
    assert on.count(plain_pass2) == off.count(plain_pass2) + 2
    assert (
        f"St32x32bOp(cute_tcgen05_flash.Repetition({store_rep})), cutlass.Float32)"
        in on
    )
    assert "flash_tiled_score_st0, tScoreSTtS0)" in on
    assert "flash_tiled_score_st1, tScoreSTtS1)" in on
    # Exactly the per-slot store setup (seven lines, emitted for both query
    # slots) and the two masked passes per slot change.
    changed = [line for line in on.splitlines() if line not in off.splitlines()]
    assert len(changed) == 2 * 7 + 2 + 2, changed


def test_unsplit_ascending_body_masks_once_behind_its_tile_predicate() -> None:
    body = {
        **REP32_BODY,
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending",
        cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: False,
    }
    on = _render(_resolve(body))
    assert on.count(STORE_HELPER) == 2
    assert MASK_HELPER not in on
    assert "fa4_disc_exp_convert_store_pipe_causal(" not in on
    # The masked arm of the per-tile predicate stores back; the dense arm
    # reduces in place.
    masked_arm = on.index("if flash_kv >= flash_m_tile0:")
    assert on.index(STORE_HELPER) > masked_arm
    assert "fa4_disc_rowmax_balanced(" in on


def test_dense_render_ignores_the_requested_knob() -> None:
    off = _render(_resolve({**PAIR_BODY, KEY: False}, is_causal=False), causal=False)
    on = _render(_resolve({**PAIR_BODY, KEY: True}, is_causal=False), causal=False)
    assert on == off
    assert on == _render(_resolve(PAIR_BODY, is_causal=False), causal=False)
    assert "flash_score_st_atom" not in on
