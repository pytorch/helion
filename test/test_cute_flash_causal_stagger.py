"""``cute_flash_causal_stagger``: the fully staggered one-head fa4 schedule.
With two 128-row query tiles (2p, 2p+1) per CTA the descending walk puts the
two diagonal KV tiles in different steps (step 0 runs stage 1's diagonal
beside stage 0's dead tile, step 1 stage 0's diagonal beside a full tile).
Staggered, stage 0 starts at its own diagonal: at step i stage 1 processes
KV tile 2p+1-i and stage 0 tile 2p-i, so both diagonals share step 0, the dead
tile disappears and stage 0 finishes one step early. Each K/V slot is then
read at two consecutive steps, which needs an aliased ring of at least four
slots; the output is bitwise identical per row.
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
    import helion
    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import ConfigSpecFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_CAUSAL_STAGGER_KEY
SKIP_KEY = cute_flash.FLASH_CAUSAL_SKIP_DEAD_TILE_KEY
SHIFT_HELPER = "_helion_flash_ptx.shift_ptx_smem_desc("
QK_HELPER = "_helion_flash_ptx.gemm_ptx_precomputed_qk("
PV_HELPER = "_helion_flash_ptx.gemm_ptx_precomputed_pv_ts("
STEADY_LOOP = "for flash_i in cutlass.range(flash_num_active_kv - 2, unroll=1)"
FULL_LOOP = "for flash_i in cutlass.range(flash_num_active_kv - 1, unroll=1)"
STAGE0_BOUND = "flash_num_active_kv0 = flash_num_active_kv - cutlass.Int32(1)"
CORR_LOOP = "for flash_kv in cutlass.range(flash_num_active_kv - 2, unroll=1)"
CORR_FULL_LOOP = "for flash_kv in cutlass.range(flash_num_active_kv - 1, unroll=1)"

# The hd128 grouped-query winners' body (resident3 packet, rep32 loads) with
# the output staged in the dead Q stages, which is what lets the hd128 ring
# reach the stagger's depth.
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
    *, is_causal: bool = True, head_dim: int = 128, **kwargs: object
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
            **options,  # pyrefly: ignore[bad-argument-type]
        )


def _active(fragment: ConfigSpecFragment) -> tuple[object, ...]:
    choices = tuple(fragment.choices)  # pyrefly: ignore[missing-attribute]
    search = fragment.search_choices  # pyrefly: ignore[missing-attribute]
    return choices if search is None else tuple(search)


def _render(
    cfg: cute_flash.FlashAttentionConfig, *, causal: bool = True, head_dim: int = 128
) -> str:
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
        score_plan=causal_score_plan(head_dim)
        if causal
        else dense_score_plan(head_dim),
    )
    return "\n".join(ast.unparse(stmt) for stmt in body)


def _role_blocks(code: str) -> dict[str, str]:
    """Split a render at its ``if warp_idx`` role guards (text before the
    first guard under ``"setup"``)."""
    blocks: dict[str, str] = {}
    starts = [
        index
        for index, line in enumerate(code.split("\n"))
        if line.startswith(("if warp_idx", "if (warp_idx"))
    ]
    lines = code.split("\n")
    blocks["setup"] = "\n".join(lines[: starts[0]])
    for begin, end in zip(starts, [*starts[1:], len(lines)], strict=True):
        blocks[lines[begin]] = "\n".join(lines[begin:end])
    return blocks


def test_resolver_defaults_the_knob_on_for_the_deep_ring_descending_causal_fa4_body() -> (
    None
):
    on = _resolve(DESC_BODY)
    assert on.causal_stagger is True
    assert on.kv_stage == 5
    # The stagger has no dead tile: the skip is pinned off under it even when
    # requested, and the identity carries both.
    assert on.causal_skip_dead_tile is False
    forced = _resolve({**DESC_BODY, SKIP_KEY: True})
    assert forced.causal_stagger is True
    assert forced.causal_skip_dead_tile is False
    assert cute_flash.flash_effective_config_values(forced)[SKIP_KEY] is False
    off = _resolve({**DESC_BODY, KEY: False})
    assert off.causal_stagger is False
    assert off.causal_skip_dead_tile is True
    # hd64 needs no staged output in the Q stages to reach the depth; the
    # default is the same shape-agnostic on-where-legal there (the hd64
    # campaign seeds carry their preference for the plain walk, which
    # measured ahead at s4096 / s8192 on causal b4h16d64, as an explicit
    # False -- see ``test_hd64_campaign_seeds_pin_the_knob_off``).
    hd64_body = {
        **DESC_BODY,
        cute_flash.FLASH_O_OVER_Q_KEY: False,
        cute_flash.FLASH_KV_STAGE_KEY: 4,
    }
    hd64 = _resolve(hd64_body, head_dim=64)
    assert hd64.causal_stagger is True
    assert hd64.causal_skip_dead_tile is False
    assert hd64.kv_stage == 4
    hd64_off = _resolve({**hd64_body, KEY: False}, head_dim=64)
    assert hd64_off.causal_stagger is False
    assert hd64_off.causal_skip_dead_tile is True
    # The env kill switch turns the default off on every head size; an
    # explicit True is still honoured.
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_CAUSAL_STAGGER": "0"}, clear=True):
        for head_dim, body, expected in (
            (128, DESC_BODY, False),
            (64, hd64_body, False),
            (64, {**hd64_body, KEY: True}, True),
        ):
            killed = cute_flash.resolve_flash_config(
                head_dim,
                16,
                body,
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
            )
            assert killed.causal_stagger is expected, (head_dim, body.get(KEY))


def test_resolver_pins_the_knob_off_below_the_ring_depth() -> None:
    # Three slots cannot hold a K/V tile across two steps next to the next
    # tile's loads; the hd128 pair with a separate staged-output buffer is
    # clamped to three however deep the request.
    shallow = _resolve({**DESC_BODY, cute_flash.FLASH_KV_STAGE_KEY: 3, KEY: True})
    assert shallow.kv_stage == 3
    assert shallow.causal_stagger is False
    assert shallow.causal_skip_dead_tile is True
    clamped = _resolve({**DESC_BODY, cute_flash.FLASH_O_OVER_Q_KEY: False, KEY: True})
    assert clamped.kv_stage == 3
    assert clamped.causal_stagger is False
    assert cute_flash.flash_effective_config_values(clamped)[KEY] is False
    # FA4's uneven three-slot ring of the asymmetric pair is too shallow too.
    pair = _resolve(
        {
            **DESC_BODY,
            cute_flash.FLASH_UNEVEN_KV_RING_KEY: True,
            cute_flash.FLASH_O_OVER_Q_KEY: False,
            KEY: True,
        },
        head_dim=192,
        head_dim_v=128,
    )
    assert pair.uneven_kv_ring is True
    assert pair.causal_stagger is False


def test_resolver_pins_the_knob_off_outside_the_dead_tile_skips_body() -> None:
    # The ascending walk, dense plans, pruned and modifier plans, the other
    # families and the cute.gemm MMA warp keep their own schedules.
    assert _resolve({**ASC_BODY, KEY: True}).causal_stagger is False
    assert _resolve({**DESC_BODY, KEY: True}, is_causal=False).causal_stagger is False
    assert (
        _resolve(
            {**DESC_BODY, KEY: True},
            has_kv_tile_pruning=True,
            standard_causal_output=False,
        ).causal_stagger
        is False
    )
    assert (
        _resolve(
            {**DESC_BODY, KEY: True},
            has_score_modifiers=True,
            score_modifier_kinds=(CAUSAL_MASK_KIND, ALIBI_BIAS_KIND),
            standard_causal_output=False,
        ).causal_stagger
        is False
    )
    assert (
        _resolve(
            {**DESC_BODY, KEY: True, cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"}
        ).causal_stagger
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
            ).causal_stagger
            is False
        )


def test_knob_is_an_appended_schema_axis_with_its_deep_ring_dependency() -> None:
    # Appended once behind the axes it was introduced after; knobs added later
    # append behind it in turn, so the tail itself is not pinned.
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(SKIP_KEY)
    witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, True)]
    assert witness == {
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        cute_flash.FLASH_KV_STAGE_KEY: cute_flash.FLASH_CAUSAL_STAGGER_MIN_KV_STAGE,
        cute_flash.FLASH_EPI_TMA_KEY: True,
        cute_flash.FLASH_O_OVER_Q_KEY: True,
    }
    # The witness context is sufficient on both head sizes: it resolves to
    # the knob's value (the hd128 pair reaches the depth only through the
    # output staged in the dead Q stages).
    for head_dim in (64, 128):
        resolved = _resolve({**witness, KEY: True}, head_dim=head_dim)
        assert resolved.causal_stagger is True, head_dim
        assert resolved.kv_stage == cute_flash.FLASH_CAUSAL_STAGGER_MIN_KV_STAGE
    # Without the key the witness context takes the on-where-legal default
    # on both head sizes.
    for head_dim in (64, 128):
        assert _resolve(dict(witness), head_dim=head_dim).causal_stagger is True
    for value in (False, True):
        resolved = _resolve({**DESC_BODY, KEY: value})
        assert cute_flash.flash_effective_config_values(resolved)[KEY] is value


def test_fragments_search_the_knob_on_plain_causal_fa4_plans_only() -> None:
    for head_dim in (64, 128):
        causal = _fragments(head_dim=head_dim)[KEY]
        assert set(_active(causal)) == {False, True}, head_dim
        assert causal.default() is True, head_dim
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


def test_output_over_q_is_searched_where_the_hd128_ring_needs_it() -> None:
    # The hd128 pair's separate staged-output buffer caps the ring at three
    # slots: the staged-in-Q output is searched there so the search can reach
    # the stagger's depth; hd64 reaches it either way and dense hd128 plans
    # have no stagger to feed.
    hd128 = _fragments(head_dim=128)
    assert set(_active(hd128[cute_flash.FLASH_O_OVER_Q_KEY])) == {False, True}
    assert hd128[cute_flash.FLASH_O_OVER_Q_KEY].default() is False
    assert (
        max(
            depth
            for depth in _active(hd128[cute_flash.FLASH_KV_STAGE_KEY])
            if isinstance(depth, int)
        )
        >= cute_flash.FLASH_CAUSAL_STAGGER_MIN_KV_STAGE
    )
    hd64 = _fragments(head_dim=64)
    assert _active(hd64[cute_flash.FLASH_O_OVER_Q_KEY]) == (False,)
    dense = _fragments(is_causal=False, head_dim=128)
    assert _active(dense[cute_flash.FLASH_O_OVER_Q_KEY]) == (False,)


@pytest.mark.parametrize("env", (None, "0", "1"))
def test_fragment_default_is_the_deep_ring_fa4_resolver_default(
    env: str | None,
) -> None:
    environ = {} if env is None else {"HELION_CUTE_FLASH_CAUSAL_STAGGER": env}
    # (head_dim, dtype, plan options, searched on the deep-ring descending
    # plain causal fa4 body)
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
    witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, True)]
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
            # to on the deep-ring descending fa4 body (the value dependency's
            # witness context): the env kill switch reaches the seeded and
            # coverage configs through it.
            pinned_fa4 = cute_flash.resolve_flash_config(
                head_dim,
                16,
                dict(witness),
                dtype=dtype,
                num_bh=64,
                **options,  # pyrefly: ignore[bad-argument-type]
            )
            assert pinned_fa4.pipeline_family == "fa4"
            assert fragment.default() is pinned_fa4.causal_stagger, options
            assert fragment.default() is (searched and env != "0"), options
            # The env moves the default only; the plain causal surface keeps
            # measuring both values.
            assert set(_active(fragment)) == ({False, True} if searched else {False}), (
                options
            )


def test_resolver_pins_the_knob_off_on_the_resident_native_route() -> None:
    """The SM103 tmem-row-reduce target routes its promoted causal seed to the
    resident-native softmax, whose segments the staggered walk is not written
    for: the resolver pins the knob off there (one identity for one render),
    as it does the dead-tile skip, and the emitter asserts it."""
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
            fast_math=True,
        )
        assert seed is not None
        body = {
            key: value
            for key, value in seed.config.items()
            if key not in (KEY, SKIP_KEY)
        }
        body[cute_flash.FLASH_KV_STAGE_KEY] = 6
        # Request the knob explicitly: the resident-native pin must still win.
        body[KEY] = True

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
                fast_math=True,
            )

        native = resolve((10, 3))
        assert native.exp2_packet == cute_flash._FLASH_DEG2_EXP2_PACKET
        assert native.kv_stage == 6
        assert native.causal_stagger is False
        for value in (False, True):
            resolved = resolve((10, 3), **{KEY: value})
            assert resolved.causal_stagger is False
            assert cute_flash.flash_effective_config_values(resolved)[KEY] is False
            assert resolved == native
        # Without the resident route the same request staggers.
        assert resolve((10, 0)).causal_stagger is True
        assert resolve(None).causal_stagger is True
        with pytest.raises(AssertionError):
            cute_flash.emit_flash_fa4_device_body(
                cast("DeviceFunction", None),
                head_dim=64,
                num_kv=512,
                sequence_extent=512 * 128,
                num_bh=1,
                total_tiles=256,
                cfg=dataclasses.replace(native, causal_stagger=True),
                has_lse=False,
                io_dtype="cutlass.Float16",
                score_plan=causal_score_plan(64),
                target_device_capability=(10, 3),
            )


@pytest.mark.parametrize("interleave", [True, False])
def test_mma_warp_runs_the_staggered_walk_and_keeps_the_protocol(
    interleave: bool,
) -> None:
    body = {**DESC_BODY, cute_flash.FLASH_MMA_INTERLEAVE_KEY: interleave}
    off = _render(_resolve({**body, KEY: False}))
    on = _render(_resolve(body))
    assert on == _render(_resolve({**body, KEY: True}))
    mma_off = _role_blocks(off)["if warp_idx == 12:"]
    mma_on = _role_blocks(on)["if warp_idx == 12:"]
    # Off is the dead-tile skip's peeled form.
    assert STEADY_LOOP in mma_off
    assert "flash_v_prev" not in mma_off
    # On: the prologue scores stage 1's diagonal from the first K tile before
    # stage 0's from the second (the S1 commit precedes the S0 commit), holds
    # that V and K slot for stage 1's step-0 reads, runs 2p steady steps that
    # serve the leading stage 1 first and release each slot after its second
    # reader, then the stage-0 tail step (PV1/QK1 of stage 1, then PV0(0)
    # committing O0) and stage 1's own tail.
    assert STEADY_LOOP in mma_on
    assert FULL_LOOP not in mma_on
    assert mma_on.count(QK_HELPER) == 5
    assert mma_on.count(PV_HELPER) == 5
    commit_s1 = "cute_tcgen05_flash.commit(flash_s_full_ptr + 1)"
    commit_s0 = "cute_tcgen05_flash.commit(flash_s_full_ptr + 0)"
    commit_o0 = "cute_tcgen05_flash.commit(flash_o_full_ptr + 0)"
    commit_o1 = "cute_tcgen05_flash.commit(flash_o_full_ptr + 1)"
    assert mma_on.index(commit_s1) < mma_on.index(commit_s0)
    assert mma_on.count("flash_v_prev = flash_kv_cons.wait_and_advance()") == 1
    assert mma_on.count("flash_k_prev = flash_kv_cons.wait_and_advance()") == 1
    assert mma_on.count("flash_v_prev.release()") == 2
    assert mma_on.count("flash_k_prev.release()") == 2
    assert mma_on.count("flash_v_prev = flash_v_full") == 1
    assert mma_on.count("flash_k_prev = flash_k_full") == 1
    assert mma_on.rindex(commit_s1) < mma_on.index(commit_o0) < mma_on.index(commit_o1)
    assert mma_on.count("flash_o_zero0 = cutlass.Boolean(False)") == 1
    assert mma_on.count("flash_o_zero1 = cutlass.Boolean(False)") == 2
    # Stage 1's PV reads the slot stage 0 read last step.
    assert "sV[None, None, None, flash_v_prev.index]" in mma_on
    assert "sK[None, None, None, flash_k_prev.index]" in mma_on
    # Stage 1 is served first in every step: its P wait, PV and QK precede
    # stage 0's, in the steady loop and the stage-0 tail alike.
    steady = mma_on[
        mma_on.index(STEADY_LOOP) : mma_on.index("flash_v_prev = flash_v_full")
    ]
    assert steady.index("mbar_spin_wait(flash_pfor_ptr + 1") < steady.index(
        "mbar_spin_wait(flash_pfor_ptr + 0"
    )
    assert steady.index("gemm_ptx_precomputed_pv_ts(flash_o1_addr") < steady.index(
        "gemm_ptx_precomputed_pv_ts(flash_o0_addr"
    )
    assert steady.index(commit_s1) < steady.index(commit_s0)
    tail = mma_on[mma_on.index("flash_q0_full.release()") :]
    assert tail.index("gemm_ptx_precomputed_pv_ts(flash_o1_addr") < tail.index(
        "gemm_ptx_precomputed_pv_ts(flash_o0_addr"
    )
    if not interleave:
        # Grouped issue: both PVs before either next QK.
        assert steady.index("gemm_ptx_precomputed_pv_ts(flash_o0_addr") < steady.index(
            "flash_k_full ="
        )
    # The shared-register Q descriptors are declared at stage 0 for this
    # walk, so the QK1-before-QK0 order alternates +/- without any shift.
    assert SHIFT_HELPER not in mma_on
    assert mma_on.count("smem_offset=-2048,") == 2
    assert mma_on.count("smem_offset=2048,") == 3
    assert "sQ[None, None, None, 0].iterator" in mma_on
    assert "sQ[None, None, None, 1].iterator" in mma_off
    # Every P-ready wait of a PV precedes its issue; stage 0 waits one fewer.
    assert mma_on.count("mbar_spin_wait(flash_pfor_ptr + 0") == 2
    assert mma_on.count("mbar_spin_wait(flash_pfor_ptr + 1") == 3


def test_softmax_stage0_starts_at_its_diagonal_and_stage1_is_unchanged() -> None:
    off_blocks = _role_blocks(_render(_resolve({**DESC_BODY, KEY: False})))
    on_blocks = _role_blocks(_render(_resolve(DESC_BODY)))
    s0_off = off_blocks["if warp_idx < 4:"]
    s0_on = on_blocks["if warp_idx < 4:"]
    # Stage 0: the walk is one tile shorter, its masked segment opens at its
    # own diagonal (no zero store, the first tile publishes no alpha) and the
    # unmasked suffix is unchanged.
    assert STAGE0_BOUND in s0_on
    assert STAGE0_BOUND not in s0_off
    assert "fa4_disc_zero_store(" in s0_off
    assert "fa4_disc_zero_store(" not in s0_on
    assert (
        "for flash_kv_mask_iter in cutlass.range(flash_num_active_kv0 - flash_m_tile0, unroll=1)"
        in s0_on
    )
    assert (
        "flash_kv = flash_num_active_kv0 - cutlass.Int32(1) - flash_kv_mask_iter"
        in s0_on
    )
    assert "for flash_kv_unmask_iter in cutlass.range(flash_m_tile0, unroll=1)" in s0_on
    assert s0_on.count("fa4_disc_rowmax_causal_store_balanced(") == 1
    # The masked tile's work is the same as the unstaggered diagonal's: one
    # masked PASS1/PASS2 pair and the alpha publish guarded for the first tile.
    assert s0_on.count("if flash_kv_mask_iter != 0:") == 1
    assert s0_on.count("fa4_disc_exp_convert_store_pipe(") == s0_off.count(
        "fa4_disc_exp_convert_store_pipe("
    )
    # Stage 1, the load warp and the epilogue warp are byte-identical.
    assert (
        on_blocks["if (warp_idx >= 4) & (warp_idx < 8):"]
        == off_blocks["if (warp_idx >= 4) & (warp_idx < 8):"]
    )
    assert on_blocks["if warp_idx == 14:"] == off_blocks["if warp_idx == 14:"]
    assert on_blocks["if warp_idx == 13:"] == off_blocks["if warp_idx == 13:"]
    assert on_blocks["setup"] == off_blocks["setup"]


def test_correction_retires_stage0_one_step_early() -> None:
    off = _role_blocks(_render(_resolve({**DESC_BODY, KEY: False})))[
        "if (warp_idx >= 8) & (warp_idx < 12):"
    ]
    on = _role_blocks(_render(_resolve(DESC_BODY)))[
        "if (warp_idx >= 8) & (warp_idx < 12):"
    ]
    assert CORR_FULL_LOOP in off
    assert CORR_LOOP in on
    assert CORR_FULL_LOOP not in on
    # The loop serves the leading stage 1 first; its last alpha follows the
    # loop alone, then stage 0's row sum is retired from the slot the loop
    # stopped at before the consumer index advances for stage 1's; the
    # epilogue count is unchanged.
    loop_body = on[on.index(CORR_LOOP) : on.index("flash_s_corr_cons_index ^= 1")]
    assert loop_body.index("flash_a1 = ") < loop_body.index("flash_a0 = ")
    off_body = off[
        off.index(CORR_FULL_LOOP) : off.index("flash_s_corr_cons_index ^= 1")
    ]
    assert off_body.index("flash_a0 = ") < off_body.index("flash_a1 = ")
    assert on.count("rescale_o_tmem(") == 3
    assert (
        on.count("rescale_o_tmem(\n                tOtO1")
        + on.count("rescale_o_tmem(tOtO1")
        == 2
    )
    loop_end = on.index("flash_s_corr_cons_index ^= 1")
    extra_alpha = on.index("flash_a1 = flash_scale_t", loop_end)
    inv_sum0 = on.index("flash_inv_sum0 = ")
    toggle_after = on.index("flash_s_corr_cons_index ^= 1", loop_end + 1)
    inv_sum1 = on.index("flash_inv_sum1 = ")
    assert loop_end < extra_alpha < inv_sum0 < toggle_after < inv_sum1
    assert on.count("flash_s_corr_cons_index ^= 1") == 2
    assert on.count("fa4_correction_epilogue_to_smem_scoped(") == 2
    assert on.count("mbar_spin_wait(flash_o_full_ptr + 0") == 1
    assert on.index("mbar_spin_wait(flash_o_full_ptr + 0") < on.index(
        "mbar_spin_wait(flash_o_full_ptr + 1"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kv_stage", 3),
        ("causal_skip_dead_tile", True),
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
    on = _resolve(DESC_BODY)
    assert on.causal_stagger is True
    assert dataclasses.asdict(on)[field] != value
    with pytest.raises(AssertionError):
        _render(dataclasses.replace(on, **{field: value}))


def test_static_descriptors_need_no_shift() -> None:
    body = {**DESC_BODY, cute_flash.FLASH_PRECOMPUTE_QK_DESC_KEY: True}
    on = _render(_resolve(body))
    assert SHIFT_HELPER not in on
    assert on.count("_helion_flash_ptx.gemm_ptx_precomputed_qk_static(") == 5
    assert STEADY_LOOP in on
    assert STAGE0_BOUND in on


def test_hd64_paired_body_staggers_without_the_q_staged_output() -> None:
    body = {
        **DESC_BODY,
        cute_flash.FLASH_O_OVER_Q_KEY: False,
        cute_flash.FLASH_KV_STAGE_KEY: 6,
        cute_flash.FLASH_S_LOAD_REP_KEY: 16,
        cute_flash.FLASH_DISC_PIPE_KEY: 1,
        cute_flash.FLASH_EXP2_PACKET_KEY: "4x1",
        cute_flash.FLASH_E2E_SCHEDULE_KEY: "8/2",
        cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY: "xu",
    }
    # The six-deep ring is legal, so the key-less body takes the default; the
    # explicit request below pins the render for determinism.
    assert _resolve(body, head_dim=64).causal_stagger is True
    resolved = _resolve({**body, KEY: True}, head_dim=64)
    assert resolved.causal_stagger is True
    assert resolved.kv_stage == 6
    assert resolved.o_over_q is False
    on = _render(resolved, head_dim=64)
    assert STEADY_LOOP in on
    assert STAGE0_BOUND in on
    assert CORR_LOOP in on
    assert "fa4_disc_zero_store(" not in on


def test_unsplit_descending_body_staggers_its_runtime_masked_loop() -> None:
    body = {**DESC_BODY, cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: False}
    off = _render(_resolve({**body, KEY: False}))
    on = _render(_resolve(body))
    assert "fa4_disc_zero_store(" in off
    assert "fa4_disc_zero_store(" not in on
    s0 = _role_blocks(on)["if warp_idx < 4:"]
    assert STAGE0_BOUND in s0
    assert "for flash_kv_iter in cutlass.range(flash_num_active_kv0, unroll=1)" in s0
    assert "flash_kv = flash_num_active_kv0 - cutlass.Int32(1) - flash_kv_iter" in s0
    assert "if flash_kv >= flash_m_tile0:" in s0
    s1 = _role_blocks(on)["if (warp_idx >= 4) & (warp_idx < 8):"]
    assert s1 == _role_blocks(off)["if (warp_idx >= 4) & (warp_idx < 8):"]


def test_ascending_and_dense_renders_ignore_the_requested_knob() -> None:
    asc_off = _render(_resolve({**ASC_BODY, KEY: False}))
    asc_on = _render(_resolve({**ASC_BODY, KEY: True}))
    assert asc_on == asc_off
    assert STAGE0_BOUND not in asc_on
    dense_off = _render(
        _resolve({**DESC_BODY, KEY: False}, is_causal=False), causal=False
    )
    dense_on = _render(
        _resolve({**DESC_BODY, KEY: True}, is_causal=False), causal=False
    )
    assert dense_on == dense_off
    assert STAGE0_BOUND not in dense_on


@pytest.mark.parametrize("head_dim", [64, 128])
def test_plain_causal_plans_seed_the_staggered_structure(head_dim: int) -> None:
    # One seed per parent (the shape's target seed and the fragment
    # defaults) carries the staggered walk: the descending split, the ring
    # depth the stagger needs and, where a separate staged-output buffer
    # would cap the ring below it (hd128), the output staged in the Q stages.
    with patch.dict(os.environ, {}, clear=True):
        seeds = cute_flash.flash_attention_seed_configs(
            head_dim,
            16,
            num_bh=64,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 0),
        )
        staggered = [
            seed
            for seed in seeds
            if seed.config.get(KEY)
            and seed.config[cute_flash.FLASH_KV_STAGE_KEY]
            >= cute_flash.FLASH_CAUSAL_STAGGER_MIN_KV_STAGE
        ]
        assert staggered
        for seed in staggered:
            resolved = cute_flash.resolve_flash_config(
                head_dim,
                16,
                {k: v for k, v in seed.config.items() if k != "block_sizes"},
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=(10, 0),
            )
            assert resolved.causal_stagger is True
            assert resolved.causal_skip_dead_tile is False
            assert resolved.causal_kv_order == "descending"
            assert resolved.causal_loop_split is True
            assert resolved.kv_stage == seed.config[cute_flash.FLASH_KV_STAGE_KEY]
            assert (
                resolved.kv_stage
                == cute_flash._FLASH_CAUSAL_STAGGER_SEED_KV_STAGE[head_dim]
            )
            assert resolved.o_over_q is (head_dim == 128)
        # Dense plans seed no staggered structure.
        dense = cute_flash.flash_attention_seed_configs(
            head_dim,
            16,
            num_bh=64,
            dtype=torch.bfloat16,
            is_causal=False,
            standard_dense_output=True,
            target_device_capability=(10, 0),
        )
        assert not any(seed.config.get(KEY) for seed in dense)


def test_hd64_campaign_seeds_pin_the_knob_off() -> None:
    # The default is shape-agnostic (on wherever the ring is deep enough);
    # the hd64 preference for the plain walk (at the winner at s2048, 0.5-1%
    # behind at s4096 / s8192) lives in the three causal hd64 campaign seeds
    # as an explicit False, exactly like their dead-tile skip and L2-shift
    # pins. The s8192 body's seven-stage ring would otherwise stagger.
    with patch.dict(os.environ, {}, clear=True):
        # The hybrid body's packet is fast_math-only; without the setting the
        # same body is seeded on the surface's default packet.
        seeds = cute_flash.flash_attention_seed_configs(
            64,
            64,
            num_bh=64,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 0),
            fast_math=True,
        )
        default_seeds = cute_flash.flash_attention_seed_configs(
            64,
            64,
            num_bh=64,
            dtype=torch.bfloat16,
            is_causal=True,
            standard_causal_output=True,
            target_device_capability=(10, 0),
        )

        def pinned_bodies(
            seeds: tuple[helion.Config, ...],
        ) -> list[dict[str, object]]:
            return [
                seed.config
                for seed in seeds
                if seed.config.get(cute_flash.FLASH_PIPELINE_FAMILY_KEY) == "fa4"
                and seed.config.get(SKIP_KEY) is False
                and seed.config.get(cute_flash.FLASH_CAUSAL_LPT_L2_SHIFT_KEY) == 0
            ]

        pinned = pinned_bodies(seeds)
        assert sorted(
            (
                str(config[cute_flash.FLASH_EXP2_PACKET_KEY]),
                int(cast("int", config[cute_flash.FLASH_KV_STAGE_KEY])),
            )
            for config in pinned
        ) == [("4x1", 2), ("8x2", 7), ("hybrid_deg1_16x8", 2)]
        assert sorted(
            (
                str(config[cute_flash.FLASH_EXP2_PACKET_KEY]),
                str(config[cute_flash.FLASH_E2E_SCHEDULE_KEY]),
                int(cast("int", config[cute_flash.FLASH_KV_STAGE_KEY])),
            )
            for config in pinned_bodies(default_seeds)
        ) == [("1x1", "16/8", 2), ("4x1", "8/2", 2), ("8x2", "16/6", 7)]
        pinned.extend(pinned_bodies(default_seeds))
        for config in pinned:
            assert config[KEY] is False
            body = {k: v for k, v in config.items() if k != "block_sizes"}
            resolved = cute_flash.resolve_flash_config(
                64,
                64,
                body,
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=(10, 0),
                fast_math=True,
            )
            assert resolved.pipeline_family == "fa4"
            assert resolved.causal_stagger is False
            assert resolved.causal_skip_dead_tile is False
            # Without the pin the deep-ring body would take the default.
            keyless = {k: v for k, v in body.items() if k != KEY}
            unpinned = cute_flash.resolve_flash_config(
                64,
                64,
                keyless,
                dtype=torch.bfloat16,
                num_bh=64,
                is_causal=True,
                standard_causal_output=True,
                target_device_capability=(10, 0),
                fast_math=True,
            )
            assert unpinned.causal_stagger is (
                resolved.kv_stage >= cute_flash.FLASH_CAUSAL_STAGGER_MIN_KV_STAGE
            )
