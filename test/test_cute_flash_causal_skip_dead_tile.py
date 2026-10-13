"""``cute_flash_causal_skip_dead_tile``: the fa4 MMA warp skips the dead
stage-0 tile. With two 128-row query tiles (2p, 2p+1) per CTA and 128-wide KV
tiles, both stages walk KV tiles 0..2p+1, but tile 2p+1 lies entirely above
stage 0's diagonal and the softmax zero-stores its P; the prologue QK0 and the
first steady-loop PV0 are dead MMAs. The knob drops both issues, keeps the
barrier protocol and moves O0's zero-init to the first live PV.
"""

from __future__ import annotations

import ast
import dataclasses
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

KEY = cute_flash.FLASH_CAUSAL_SKIP_DEAD_TILE_KEY
SHIFT_HELPER = "_helion_flash_ptx.shift_ptx_smem_desc("
QK_HELPER = "_helion_flash_ptx.gemm_ptx_precomputed_qk("
PV_HELPER = "_helion_flash_ptx.gemm_ptx_precomputed_pv_ts("
DEAD_GUARD = "if flash_i != 0:"  # the rejected per-step branch form

# The hd128 grouped-query winners' body (resident3 packet, rep32 loads).
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
}
ASC_BODY: dict[str, object] = {
    **DESC_BODY,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending",
    cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: False,
}


def _resolve(
    config: dict[str, object] | None,
    *,
    is_causal: bool = True,
    head_dim: int = 128,
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
    *, is_causal: bool = True, **kwargs: object
) -> dict[str, ConfigSpecFragment]:
    options: dict[str, object] = {
        "standard_dense_output": not is_causal,
        "standard_causal_output": is_causal,
        **kwargs,
    }
    with patch.dict(os.environ, {}, clear=True):
        return cute_flash.flash_autotune_fragments(
            128,
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


def _render(cfg: cute_flash.FlashAttentionConfig, *, causal: bool = True) -> str:
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=128,
        num_kv=16,
        sequence_extent=16 * 128,
        num_bh=64,
        total_tiles=64 * 8,
        cfg=cfg,
        has_lse=False,
        io_dtype="cutlass.BFloat16",
        score_plan=causal_score_plan(128) if causal else dense_score_plan(128),
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_resolver_defaults_the_knob_on_for_the_descending_causal_fa4_body() -> None:
    assert _resolve(DESC_BODY).causal_skip_dead_tile is True
    assert _resolve(DESC_BODY, head_dim=64).causal_skip_dead_tile is True
    assert _resolve({**DESC_BODY, KEY: False}).causal_skip_dead_tile is False
    with patch.dict(
        os.environ, {"HELION_CUTE_FLASH_CAUSAL_SKIP_DEAD_TILE": "0"}, clear=True
    ):
        assert (
            cute_flash.resolve_flash_config(
                128,
                16,
                DESC_BODY,
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
            ).causal_skip_dead_tile
            is False
        )
    # The ascending walk meets the dead tile last (no zero-store prologue);
    # dense plans have no dead tile; the pruned, modifier and non-fa4 bodies
    # keep their own first steps.
    assert _resolve({**ASC_BODY, KEY: True}).causal_skip_dead_tile is False
    assert (
        _resolve({**DESC_BODY, KEY: True}, is_causal=False).causal_skip_dead_tile
        is False
    )
    assert (
        _resolve(
            {**DESC_BODY, KEY: True},
            has_kv_tile_pruning=True,
            standard_causal_output=False,
        ).causal_skip_dead_tile
        is False
    )
    assert (
        _resolve(
            {**DESC_BODY, KEY: True},
            has_score_modifiers=True,
            score_modifier_kinds=(CAUSAL_MASK_KIND, ALIBI_BIAS_KIND),
            standard_causal_output=False,
        ).causal_skip_dead_tile
        is False
    )
    assert (
        _resolve(
            {**DESC_BODY, KEY: True, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
        ).causal_skip_dead_tile
        is False
    )
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_MMA_PTX": "0"}, clear=True):
        assert (
            cute_flash.resolve_flash_config(
                128,
                16,
                {**DESC_BODY, KEY: True},
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
            ).causal_skip_dead_tile
            is False
        )


def test_knob_is_an_appended_schema_axis_with_its_descending_fa4_dependency() -> None:
    # Appended once behind the axes it was introduced after (an inserted axis
    # would reshuffle every later axis's coverage rows); knobs added later
    # append behind it in turn, so the tail itself is not pinned.
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(cute_flash.FLASH_ROW_BOUND_TILING_KEY)
    witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, True)]
    assert witness == {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    }
    # The witness context is sufficient: it resolves to the knob's value.
    assert _resolve({**witness, KEY: True}).causal_skip_dead_tile is True
    for value in (False, True):
        resolved = _resolve({**DESC_BODY, KEY: value})
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


@pytest.mark.parametrize("env", (None, "0", "1"))
def test_fragment_default_is_the_fa4_descending_resolver_default(
    env: str | None,
) -> None:
    environ = {} if env is None else {"HELION_CUTE_FLASH_CAUSAL_SKIP_DEAD_TILE": env}
    # (head_dim, dtype, plan options, searched on the descending plain causal
    # fa4 body)
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
            # to on the descending plain fa4 body (the value dependency's
            # witness context): the env kill switch reaches the seeded and
            # coverage configs through it.
            pinned_fa4 = cute_flash.resolve_flash_config(
                head_dim,
                16,
                {
                    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
                },
                dtype=dtype,
                num_bh=64,
                **options,  # pyrefly: ignore[bad-argument-type]
            )
            assert pinned_fa4.pipeline_family == "fa4"
            assert fragment.default() is pinned_fa4.causal_skip_dead_tile, options
            assert fragment.default() is (searched and env != "0"), options
            # The env moves the default only; the plain causal surface keeps
            # measuring both values.
            assert set(_active(fragment)) == ({False, True} if searched else {False}), (
                options
            )


@pytest.mark.parametrize("requested", (None, True))
def test_resolver_pins_the_knob_off_on_the_explicit_stateful_lowering(
    requested: bool | None,
) -> None:
    """The explicit ``resident_stateful`` lowering reuses the resident-native
    body, whose first step is a full masked step rather than the zero store:
    the resolver pins the skip off there too, on every target, and the
    render asserts the pin."""
    overrides: dict[str, object] = {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
    }
    if requested is not None:
        overrides[KEY] = requested
    with patch.dict(os.environ, {}, clear=True):
        for capability in (None, (10, 0), (10, 3)):
            cfg = cute_flash.resolve_flash_config(
                64,
                32,
                overrides,
                dtype=torch.float16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
                causal_resident_compatible=True,
                target_device_capability=capability,
            )
            assert cfg.softmax_lowering == "resident_stateful"
            assert cfg.causal_skip_dead_tile is False
            body = cute_flash.emit_flash_fa4_device_body(
                cast("DeviceFunction", None),
                head_dim=64,
                num_kv=32,
                sequence_extent=32 * 128,
                num_bh=64,
                total_tiles=64 * 16,
                cfg=cfg,
                has_lse=False,
                io_dtype="cutlass.Float16",
                score_plan=causal_score_plan(64),
                target_device_capability=capability,
            )
            assert body


def test_resolver_pins_the_knob_off_on_the_resident_native_route() -> None:
    """The SM103 tmem-row-reduce target routes its promoted causal seed to the
    resident-native softmax, whose first step is a full masked step rather
    than the zero store: no dead tile to skip, so the resolver pins the knob
    off there (one identity for one render) and the emitter asserts it; the
    same seed on SM100, without a target, or at a KV size without a resident
    policy row keeps the skip."""
    with patch.dict(os.environ, {}, clear=True):
        seed = cute_flash.flash_attention_seed_config(
            64,
            512,
            dtype=torch.float16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 3),
        )
        assert seed is not None
        # The target seed carries the pinned value (its fragment default is
        # the descending fa4 body's), so it equals its normalized form; the
        # resolutions below drop the key to exercise each target's default.
        assert seed.config[KEY] is False
        body = {key: value for key, value in seed.config.items() if key != KEY}

        def resolve(
            capability: tuple[int, int] | None, num_kv: int = 512, **overrides: object
        ) -> cute_flash.FlashAttentionConfig:
            return cute_flash.resolve_flash_config(
                64,
                num_kv,
                {**body, **overrides},
                dtype=torch.float16,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=capability,
            )

        def render(
            cfg: cute_flash.FlashAttentionConfig, capability: tuple[int, int] | None
        ) -> str:
            body = cute_flash.emit_flash_fa4_device_body(
                cast("DeviceFunction", None),
                head_dim=64,
                num_kv=512,
                sequence_extent=512 * 128,
                num_bh=1,
                total_tiles=256,
                cfg=cfg,
                has_lse=False,
                io_dtype="cutlass.Float16",
                score_plan=causal_score_plan(64),
                target_device_capability=capability,
            )
            return "\n".join(ast.unparse(stmt) for stmt in body)

        native = resolve((10, 3))
        assert native.exp2_packet == cute_flash._FLASH_DEG2_EXP2_PACKET
        assert native.causal_skip_dead_tile is False
        for value in (False, True):
            resolved = resolve((10, 3), **{KEY: value})
            assert resolved.causal_skip_dead_tile is False
            assert cute_flash.flash_effective_config_values(resolved)[KEY] is False
            assert resolved == native
        # Without the resident route the same seed keeps the skip: another
        # target, no target, or a KV size the policy table has no row for.
        assert resolve((10, 0)).causal_skip_dead_tile is True
        assert resolve(None).causal_skip_dead_tile is True
        assert resolve((10, 3), num_kv=256).causal_skip_dead_tile is True
        # The resident-native render is the full walk; forcing the knob past
        # the resolver trips the emitter's guard instead of peeling a step
        # whose P-last arrive the softmax never makes.
        source = render(native, (10, 3))
        assert SHIFT_HELPER not in source
        assert (
            "for flash_i in cutlass.range(flash_num_active_kv - 1, unroll=1)" in source
        )
        with pytest.raises(AssertionError):
            render(dataclasses.replace(native, causal_skip_dead_tile=True), (10, 3))
        generic = render(resolve(None), None)
        assert SHIFT_HELPER in generic
        assert (
            "for flash_i in cutlass.range(flash_num_active_kv - 2, unroll=1)" in generic
        )


@pytest.mark.parametrize("interleave", [True, False])
def test_mma_warp_peels_the_dead_tile_step_and_keeps_the_protocol(
    interleave: bool,
) -> None:
    body = {**DESC_BODY, cute_flash.FLASH_MMA_INTERLEAVE_KEY: interleave}
    off = _render(_resolve({**body, KEY: False}))
    on = _render(_resolve(body))
    assert on == _render(_resolve({**body, KEY: True}))
    # Off: the prologue issues QK0 and QK1, the steady loop PV0/QK0/PV1/QK1 and
    # the epilogue PV0/PV1; no descriptor shift, no peeled step.
    assert off.count(QK_HELPER) == 4
    assert off.count(PV_HELPER) == 4
    assert SHIFT_HELPER not in off
    assert "for flash_i in cutlass.range(flash_num_active_kv - 1, unroll=1)" in off
    assert "for flash_i in cutlass.range(flash_num_active_kv - 2, unroll=1)" not in off
    # On: the prologue QK0 becomes the descriptor shift (the shared-register
    # descriptors still move by the stage stride), the dead tile's step is
    # peeled before the steady loop with PV0 replaced by the P-last wait, and
    # the loop runs one step fewer. O0's zero-init stays armed through the
    # peeled step (no assignment there); O1's is cleared by its live PV1.
    assert on.count(QK_HELPER) == 5
    assert on.count(PV_HELPER) == 5
    assert on.count(SHIFT_HELPER) == 1
    assert on.count(DEAD_GUARD) == 0
    shift_at = on.index(SHIFT_HELPER)
    assert shift_at < on.index(QK_HELPER)
    assert on.index("cute_tcgen05_flash.commit(flash_s_full_ptr + 0)") > shift_at
    assert "for flash_i in cutlass.range(flash_num_active_kv - 2, unroll=1)" in on
    assert "for flash_i in cutlass.range(flash_num_active_kv - 1, unroll=1)" not in on
    pfor2_wait = "mbar_spin_wait(flash_pfor2_ptr + 0"
    assert on.count(pfor2_wait) == off.count(pfor2_wait) + 1
    assert on.index(pfor2_wait) < on.index(
        "for flash_i in cutlass.range(flash_num_active_kv - 2, unroll=1)"
    )
    assert on.count("flash_o_zero0 = cutlass.Boolean(False)") == 1
    assert on.count("flash_o_zero1 = cutlass.Boolean(False)") == 2
    # The softmax, correction, load and epilogue roles are untouched.
    for helper in (
        "fa4_disc_zero_store(",
        "fa4_disc_rowmax_causal_store_balanced(",
        "rescale_o_tmem(",
        "fa4_correction_epilogue_to_smem_scoped(",
    ):
        assert on.count(helper) == off.count(helper)
    # Only the MMA warp's role block changes: the text before its role guard
    # and the text from the next role guard on are byte-identical.
    head_off, next_off = _mma_role_block(off)
    head_on, next_on = _mma_role_block(on)
    assert off[:head_off] == on[:head_on]
    assert off[next_off:] == on[next_on:]


def _mma_role_block(code: str) -> tuple[int, int]:
    """Return the span of the MMA warp's role block (``if warp_idx == N:``)."""
    alloc = code.index("flash_tmem.allocate(512)")
    head = code.rindex("\nif warp_idx", 0, alloc)
    following = code.find("\nif ", alloc)
    return head, len(code) if following < 0 else following


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("p_chunk_arrive", True),
        ("mma_ptx", False),
        ("kv_tile_n", 256),
        ("q_tile_m", 64),
        ("alternating_warpgroups", True),
        ("pipeline_family", "fa4_deep_1cta"),
    ],
)
def test_emitter_rejects_the_knob_outside_the_resolver_legality(
    field: str, value: object
) -> None:
    # The peeled step assumes the resolver's legality (whole-P arrives above
    # all: the dead step consumes the zero store's single P-last arrive); a
    # config built past the resolver with the knob on cannot render.
    on = _resolve(DESC_BODY)
    assert on.causal_skip_dead_tile is True
    assert dataclasses.asdict(on)[field] != value
    with pytest.raises(AssertionError):
        _render(dataclasses.replace(on, **{field: value}))


def test_static_descriptors_need_no_shift() -> None:
    body = {**DESC_BODY, cute_flash.FLASH_PRECOMPUTE_QK_DESC_KEY: True}
    on = _render(_resolve(body))
    assert SHIFT_HELPER not in on
    assert on.count("_helion_flash_ptx.gemm_ptx_precomputed_qk_static(") == 5
    assert "for flash_i in cutlass.range(flash_num_active_kv - 2, unroll=1)" in on


def test_ascending_and_dense_renders_ignore_the_requested_knob() -> None:
    asc_off = _render(_resolve({**ASC_BODY, KEY: False}))
    asc_on = _render(_resolve({**ASC_BODY, KEY: True}))
    assert asc_on == asc_off
    assert (
        "for flash_i in cutlass.range(flash_num_active_kv - 2, unroll=1)" not in asc_on
    )
    dense_off = _render(
        _resolve({**DESC_BODY, KEY: False}, is_causal=False), causal=False
    )
    dense_on = _render(
        _resolve({**DESC_BODY, KEY: True}, is_causal=False), causal=False
    )
    assert dense_on == dense_off
    assert SHIFT_HELPER not in dense_on
