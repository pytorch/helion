"""``cute_flash_flex_dead_stage_split``: on the FlexAttention list walk, a pair
whose second 128-row stage sees no KV tile (the ragged last pair of an odd
128-row query count, every pair of a 128-row query, an empty BlockMask row)
splits the first stage's walk over both softmax warpgroups. The second stage
loads the first stage's Q rows, the two stages take the walk's tiles
alternately (step parity; the absent-stage MMA skip issues each tile once) and
the correction warps merge the two partial online softmaxes in the epilogue
(``O = (O0 c0 + O1 c1) / (l0 c0 + l1 c1)``, ``c_s = 2^(m_s - max(m0, m1))``)
and store the merged LSE. Decided once per work item in the CTA-wide setup
(``flash_flex_split``); items with two live stages run the unsplit protocol.
"""

from __future__ import annotations

import dataclasses

import pytest

from test.test_cute_flash_flex_skip_absent import BODY
from test.test_cute_flash_flex_skip_absent import _active
from test.test_cute_flash_flex_skip_absent import _fragments
from test.test_cute_flash_flex_skip_absent import _mma_role_block
from test.test_cute_flash_flex_skip_absent import _plan
from test.test_cute_flash_flex_skip_absent import _render
from test.test_cute_flash_flex_skip_absent import _resolve

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import dense_score_plan

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY
SKIP = cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY
ENV = "HELION_CUTE_FLASH_FLEX_DEAD_STAGE_SPLIT"
MERGE_HELPER = "_helion_flash_rt.fa4_correction_epilogue_merge_to_smem_scoped("
SPLIT_FLAG = "flash_flex_split = cutlass.Int32("


def test_resolver_defaults_the_knob_off_and_pins_it_to_the_skip() -> None:
    assert _resolve(BODY).flex_dead_stage_split is False
    on = _resolve({**BODY, SKIP: True, KEY: True})
    assert on.flex_skip_absent and on.flex_dead_stage_split
    # No skip, no split: the split issues each tile for one stage only.
    assert _resolve({**BODY, SKIP: False, KEY: True}).flex_dead_stage_split is False
    assert _resolve({**BODY, KEY: True}).flex_dead_stage_split is False
    # The env switch moves the default only, under the skip.
    assert _resolve({**BODY, SKIP: True}, environ={ENV: "1"}).flex_dead_stage_split
    assert not _resolve(BODY, environ={ENV: "1"}).flex_dead_stage_split
    assert (
        _resolve(
            {**BODY, SKIP: True, KEY: False}, environ={ENV: "1"}
        ).flex_dead_stage_split
        is False
    )
    # Outside the list walk nothing is split.
    assert (
        _resolve(
            {**BODY, SKIP: True, KEY: True}, has_list_walk=False
        ).flex_dead_stage_split
        is False
    )
    assert (
        _resolve(
            {**BODY, SKIP: True, KEY: True}, environ={"HELION_CUTE_FLASH_MMA_PTX": "0"}
        ).flex_dead_stage_split
        is False
    )


def test_knob_is_an_appended_schema_axis_with_the_skip_as_its_witness() -> None:
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(SKIP)
    witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, True)]
    assert witness == {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4", SKIP: True}
    assert _resolve({**witness, KEY: True}).flex_dead_stage_split is True
    for value in (False, True):
        resolved = _resolve({**BODY, SKIP: True, KEY: value})
        assert cute_flash.flash_effective_config_values(resolved)[KEY] is value


def test_fragments_search_the_knob_on_list_walk_plans_only() -> None:
    flex = _fragments()[KEY]
    assert set(_active(flex)) == {False, True}
    assert flex.default() is False
    dense = _fragments(has_list_walk=False)[KEY]
    assert _active(dense) == (False,) and dense.default() is False


@pytest.mark.parametrize("env", (None, "0", "1"))
def test_fragment_default_is_the_skip_body_resolver_default(env: str | None) -> None:
    environ = {} if env is None else {ENV: env}
    fragment = _fragments(environ=environ)[KEY]
    pinned = _resolve(
        {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4", SKIP: True}, environ=environ
    )
    assert fragment.default() is pinned.flex_dead_stage_split
    assert fragment.default() is (env == "1")
    assert set(_active(fragment)) == {False, True}


@pytest.mark.parametrize("num_kv", (16, 17, 64))
def test_resolved_value_is_length_invariant(num_kv: int) -> None:
    on = _resolve({**BODY, SKIP: True, KEY: True}, num_kv=num_kv)
    assert on.flex_dead_stage_split
    assert cute_flash.flash_effective_config_values(on) == (
        cute_flash.flash_effective_config_values(
            _resolve({**BODY, SKIP: True, KEY: True})
        )
    )


@pytest.mark.parametrize("ragged", [True, False])
def test_render_splits_every_role_on_the_per_item_flag(ragged: bool) -> None:
    plan = _plan(q_seq=128 + 256 * 3 if ragged else 2048, kv_seq=2176)
    skip = _render(_resolve({**BODY, SKIP: True, KEY: False}), plan)
    split = _render(_resolve({**BODY, SKIP: True, KEY: True}), plan)
    assert SPLIT_FLAG not in skip and MERGE_HELPER not in skip
    # Setup: the per-item flag (the second stage sees no tile, two or more
    # tiles) and the second stage's Q row.
    assert split.count(SPLIT_FLAG) == 1
    assert (
        "flash_flex_qrow1 = cutlass.Int32(cutlass.select_("
        "flash_flex_split != 0, flash_m_tile0, flash_m_tile1))"
    ) in split
    # Load warp: the second Q stage takes that row.
    assert "tQgQ[None, flash_flex_qrow1]" in split
    assert "tQgQ[None, flash_q_mma_tile1]" not in split
    # MMA warp: the per-stage issue flags follow the step parity on a split
    # item (prologue step 0, steady step i + 1).
    mma = split[slice(*_mma_role_block(split))]
    assert mma.count("flash_flex_odd = cutlass.Int32(0) & cutlass.Int32(1)") == 1
    assert (
        mma.count("flash_flex_odd = flash_i + cutlass.Int32(1) & cutlass.Int32(1)") == 1
    )
    assert mma.count("flash_flex_n0b & flash_flex_odd, flash_flex_n1b") == 2
    assert mma.count("issue=flash_flex_qk_live0)") == 2
    assert mma.count("issue=flash_flex_pv_issue1)") == 2
    # Softmax warps: stage 0 is absent on odd steps of a split item; stage 1
    # takes stage 0's lists on them.
    assert (
        split.count(
            "(flash_flex_split != 0) & (flash_kv_iter & cutlass.Int32(1) != 0), "
            "cutlass.Int32(0), flash_flex_status"
        )
        == 1
    )
    assert split.count("flash_flex_status_alt = ") == 1
    # Both stages publish their row max beside the row sum; stage 0's LSE
    # store is the unsplit item's, stage 1's rows get -inf on a split item.
    assert (
        "flash_scale_t[flash_s_corr_prod_index ^ 1, 0, flash_local_tidx] = flash_row_max"
        in split
    )
    assert (
        "flash_scale_t[flash_s_corr_prod_index ^ 1, 1, flash_local_tidx] = flash_row_max"
        in split
    )
    assert split.count("if flash_flex_split == 0:") == 1
    assert (
        "-cutlass.Float32.inf * _flash_scale_log2 + cute.math.log2(cutlass.Float32(0.0))"
        in split
    )
    # Correction warps: both publishes first, the merge on a split item (after
    # the second accumulator's O_full), the plain epilogue otherwise; the
    # second stage's epilogue writes zeros on a split item and never waits
    # its publish barrier again.
    assert split.count(MERGE_HELPER) == 1
    assert split.count("_helion_flash_rt.fa4_correction_epilogue_to_smem_scoped(") == 2
    assert split.count(
        "named_barrier_wait_unaligned(7 + warp_idx % 4, 64)"
    ) == skip.count("named_barrier_wait_unaligned(7 + warp_idx % 4, 64)")
    assert (
        "flash_flex_L = flash_flex_l0 * flash_flex_c0 + flash_flex_l1 * flash_flex_c1"
        in split
    )
    assert (
        "flash_inv_sum1 = cutlass.Float32(cutlass.select_(flash_flex_split != 0, "
        "cutlass.Float32(0.0), _helion_flash_rt.rcp_approx_ftz_or_zero(flash_flex_l1)))"
    ) in split
    assert (
        "_flash_mLSE[flash_m_tile0 * 128 + flash_local_tidx, flash_bh] = "
        "flash_flex_mm * _flash_scale_log2 + cute.math.log2(flash_flex_L)"
    ) in split
    assert ("if flash_local_tidx < flash_live_rows0:" in split) == ragged
    # Every barrier arrival of the protocol is kept.
    for arrival in (
        "cute_tcgen05_flash.commit(flash_s_full_ptr + 0)",
        "cute_tcgen05_flash.commit(flash_o_full_ptr + ",
        "mbar_spin_wait(flash_pfor_ptr + ",
        "mbarrier_arrive(flash_pfor_ptr + ",
        "named_barrier_arrive_unaligned(7 + warp_idx % 4, 64)",
        "fa4_disc_zero_store(",
    ):
        assert split.count(arrival) == skip.count(arrival), arrival


def test_dense_and_256_row_plans_render_without_the_split() -> None:
    on = _resolve({**BODY, SKIP: True, KEY: True})
    off = _resolve({**BODY, SKIP: True, KEY: False})
    # A 256-row BlockMask gives both stages one row: no dead second stage.
    plan = _plan()
    plan = dataclasses.replace(
        plan, kv_walk=dataclasses.replace(plan.kv_walk, q_block=256)
    )
    assert _render(on, plan) == _render(off, plan)
    dense = dense_score_plan(64)
    on_dense = _resolve({**BODY, SKIP: True, KEY: True}, has_list_walk=False)
    assert _render(on_dense, dense) == _render(
        _resolve({**BODY, KEY: False}, has_list_walk=False), dense
    )
    assert SPLIT_FLAG not in _render(on, plan)
