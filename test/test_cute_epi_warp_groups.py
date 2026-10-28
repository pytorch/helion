"""Epilogue warp groups, the configured epilogue subtile and the auxiliary
prefetch ring of the tcgen05 role-local TMA-store epilogue (CPU, fake sm100).

``tcgen05_epi_warp_groups`` splits the subtile walk over G groups of four
epilogue warps (own named barrier, C stages and bulk-store groups each),
``tcgen05_epi_subtile_n`` pins the C subtile through the explicit-tile path
and ``tcgen05_aux_prefetch_subtiles`` reads the exact-shape auxiliary
fragment ``d`` subtiles ahead through a register ring. Every variant renders
the same per-element chain in the same K order (bitwise-equal outputs on the
GPU, ``test_cute_epi_warp_groups_cuda.py``); these tests pin the generated
structure, the admission rules, the SMEM pricing and the coverage group.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from test.test_cute_tcgen05_aux_fragment_hoist import HOIST
from test.test_cute_tcgen05_aux_fragment_hoist import ON
from test.test_cute_tcgen05_aux_fragment_hoist import PLACEMENT
from test.test_cute_tcgen05_aux_fragment_hoist import PRE
from test.test_cute_tcgen05_aux_fragment_hoist import SHARED
from test.test_cute_tcgen05_aux_fragment_hoist import _carriers
from test.test_cute_tcgen05_aux_fragment_hoist import _epilogue_role
from test.test_cute_tcgen05_aux_fragment_hoist import _group
from test.test_cute_tcgen05_aux_fragment_hoist import _pair
from test.test_cute_tcgen05_aux_fragment_hoist import base_config
from test.test_cute_tcgen05_aux_fragment_hoist import bind
from test.test_cute_tcgen05_aux_fragment_hoist import cuda_trace  # noqa: F401
from test.test_cute_tcgen05_aux_fragment_hoist import gated_pair
from test.test_cute_tcgen05_aux_fragment_hoist import normalized
from test.test_cute_tcgen05_aux_fragment_hoist import pair_no_aux
from test.test_cute_tcgen05_aux_fragment_hoist import plain_single
from test.test_cute_tcgen05_aux_fragment_hoist import render
from test.test_cute_tcgen05_aux_fragment_hoist import residual_single
from test.test_cute_tcgen05_aux_fragment_hoist import state_of

from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_GROUP_BARRIER_BASE_ID
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_WARP_GROUP_CHOICES
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    tcgen05_aux_prefetch_bytes_per_thread,
)
from helion._compiler.cute.tcgen05_constants import tcgen05_role_launch_n_width
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig

pytestmark = skipUnlessBackends(["cute"])

GROUPS = TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
SUBTILE = TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
PREFETCH = TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
N32 = {SUBTILE: 32}
NARROW_TILE = "(cute.make_layout(128), cute.make_layout(32))"


def pair(*parts: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """The 256x256x64 CTA-pair fanout at the (128, 32) subtile."""
    config = _pair(256, 64, 5, N32, *parts)
    config.update(overrides)
    return config


def two_groups(*parts: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """The 256x256x64 CTA-pair fanout with two epilogue warp groups."""
    return pair({GROUPS: 2}, *parts, **overrides)


def test_two_group_hoist_renders_the_grouped_epilogue() -> None:
    bound = bind(gated_pair)
    code = render(bound, two_groups(ON))
    # Roles: eight epilogue warps, the MMA and TMA warps after them, and the
    # per-warp group / leader predicates; warps w and w + 4 share TMEM lanes.
    assert "tcgen05_epi_active = tcgen05_warp_idx < cutlass.Int32(8)" in code
    assert "tcgen05_exec_active = tcgen05_warp_idx == cutlass.Int32(8)" in code
    assert "tcgen05_tma_warp = tcgen05_warp_idx == cutlass.Int32(9)" in code
    assert "tcgen05_epi_group = tcgen05_warp_idx // cutlass.Int32(4)" in code
    assert (
        "tcgen05_epi_leader = tcgen05_warp_idx % cutlass.Int32(4) == cutlass.Int32(0)"
        in code
    )
    assert (
        "tcgen05_epi_tidx = tcgen05_lane_idx + tcgen05_warp_idx % cutlass.Int32(4) "
        "* cutlass.Int32(32) if tcgen05_epi_active else cutlass.Int32(0)"
    ) in code
    # No setmaxnreg split; the launch pins one CTA per SM instead.
    assert "setmaxregister" not in code
    assert "_helion_cute_min_blocks_per_mp = 1" in code
    assert "block=(32, 10, 1)" in code
    # Barriers and pipelines count every group's warps.
    assert "NamedBarrier(barrier_id=1, num_threads=288)" in code
    assert "CooperativeGroup(cutlass.pipeline.Agent.Thread, cutlass.Int32(16))" in code
    assert (
        code.count(
            "NamedBarrier(barrier_id=cutlass.Int32(6) + tcgen05_epi_group, "
            "num_threads=cutlass.Int32(128))"
        )
        == 2
    )
    # Each output's ring: two stages per group, four in total; each group's
    # bulk-store accounting is two deep.
    assert code.count("'num_c_stage': cutlass.Int32(4)") == 2
    assert code.count("tcgen05_store_epi_tile, 4)") == 1
    assert code.count("tcgen05_store_epi_tile_1, 4)") == 1
    assert code.count("PipelineTmaStore.create(num_stages=2, ") == 2
    assert code.count("'c_stage_count': 4") == 2
    assert code.count("'epi_tile_m': 128, 'epi_tile_n': 32, 'd_store_box_n': 32") == 2
    assert code.count(NARROW_TILE) == 3
    role = _epilogue_role(code)
    # The group's whole fragment (four of the eight subtiles) is read ahead of
    # the accumulator wait, which precedes the loop; the loop walks every
    # second subtile and releases the accumulator once after it.
    ring = [
        role.index(
            f"cute.autovec_copy(tcgen05_tTR_gAux_grouped_0[None, None, None, "
            f"cutlass.Int32({2 * j}) + tcgen05_epi_group], tcgen05_aux_ring_0_{j})"
        )
        for j in range(4)
    ]
    wait = role.index("tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)")
    loop = role.index(
        "for _tcgen05_local in cutlass.range_constexpr(tcgen05_local_count):"
    )
    assert max(ring) < wait < loop
    assert "tcgen05_local_count = (tcgen05_subtile_count + 1) // 2" in role
    assert "assert cutlass.const_expr(tcgen05_subtile_count == 8)" in role
    assert (
        "_tcgen05_subtile = cutlass.Int32(2 * _tcgen05_local) + tcgen05_epi_group"
        in role
    )
    assert (
        "tcgen05_aux_loaded_0 = tcgen05_aux_ring_0[_tcgen05_local % 4].load()" in role
    )
    assert (
        "if cutlass.const_expr(_tcgen05_local != 0) and tcgen05_epi_leader:\n"
        "                    tcgen05_c_pipeline.producer_acquire()"
    ) in role
    assert (
        "tcgen05_c_buffer = tcgen05_epi_group * cutlass.Int32(2) + "
        "(tcgen05_tma_store_role_tile * cutlass.Int32(4) + "
        "cutlass.Int32(_tcgen05_local)) % cutlass.Int32(2)"
    ) in role
    # The first acquire (ahead of the role text), the store issue and the
    # tail drain belong to each group's leader warp.
    assert code.count("if tcgen05_epi_leader:") == 3
    assert "tcgen05_warp_idx == cutlass.Int32(0)" not in role
    release = role.index("cute.arch.fence_view_async_tmem_load()")
    commit = role.index("tcgen05_c_pipeline.producer_commit()")
    assert commit < release
    assert (
        "            cute.arch.fence_view_async_tmem_load()\n"
        "            with cute.arch.elect_one():\n"
        "                tcgen05_acc_pipeline.consumer_release("
        "tcgen05_acc_consumer_state)\n"
        "            if tcgen05_epi_active:\n"
        "                tcgen05_acc_consumer_state.advance()"
    ) in role
    # The per-subtile read, the one-group hoist and the K loop's unroll are
    # untouched by the knobs.
    assert "tcgen05_aux_rmem_full_0" not in code
    assert "cute.autovec_copy(tcgen05_tTR_gAux_subtile_0" not in code
    assert "unroll=1" not in code
    # One stage per group: the ring shrinks to two stages in total.
    one_stage = render(bound, two_groups(ON, tcgen05_c_stages=1))
    assert one_stage.count("'num_c_stage': cutlass.Int32(2)") == 2
    assert one_stage.count("PipelineTmaStore.create(num_stages=1, ") == 2
    assert (
        "tcgen05_c_buffer = tcgen05_epi_group * cutlass.Int32(1) + "
        "(tcgen05_tma_store_role_tile * cutlass.Int32(4) + "
        "cutlass.Int32(_tcgen05_local)) % cutlass.Int32(1)"
    ) in one_stage


def test_defaults_render_byte_identical_to_the_knobless_config() -> None:
    bound = bind(gated_pair)
    for config in (_pair(256, 64, 5), base_config(PRE, SHARED, ON)):
        plain = render(bound, config)
        explicit = render(bound, config | {GROUPS: 1, SUBTILE: None, PREFETCH: 0})
        assert explicit == plain
        assert "range_constexpr" not in plain
        assert "tcgen05_epi_group" not in plain
        assert "tcgen05_aux_ring" not in plain
        assert "_helion_cute_min_blocks_per_mp" not in plain


def test_prefetch_ring_renders_the_subtile_reads_ahead() -> None:
    bound = bind(gated_pair)
    code = render(bound, _pair(256, 64, 5, N32, {PREFETCH: 3}))
    role = _epilogue_role(code)
    # Four buffers; the first three fragments ahead of the accumulator wait,
    # which stays at subtile 0 inside the (trace-time) loop.
    for j in range(4):
        assert (
            f"tcgen05_aux_ring_0_{j} = cute.make_rmem_tensor("
            "cute.make_layout(tcgen05_tRS_rAcc_1.shape), cutlass.Float16)"
        ) in role
    for j in range(3):
        assert (
            f"cute.autovec_copy(tcgen05_tTR_gAux_grouped_0[None, None, None, "
            f"cutlass.Int32({j})], tcgen05_aux_ring_0_{j})"
        ) in role
    loop = role.index(
        "for _tcgen05_subtile in cutlass.range_constexpr(tcgen05_subtile_count):"
    )
    assert role.index("tcgen05_aux_ring_0_2)") < loop
    assert (
        "                if cutlass.const_expr(_tcgen05_subtile + 3 < 8):\n"
        "                    cute.autovec_copy(tcgen05_tTR_gAux_grouped_0"
        "[None, None, None, cutlass.Int32(_tcgen05_subtile + 3)], "
        "tcgen05_aux_ring_0[(_tcgen05_subtile + 3) % 4])\n"
        "                tcgen05_aux_loaded_0 = tcgen05_aux_ring_0"
        "[_tcgen05_subtile % 4].load()"
    ) in role
    assert "if cutlass.const_expr(_tcgen05_subtile == 0):" in role
    assert (
        "if cutlass.const_expr(_tcgen05_subtile != 0) and "
        "tcgen05_warp_idx == cutlass.Int32(0):"
    ) in role
    assert (
        "if cutlass.const_expr(_tcgen05_subtile == tcgen05_subtile_count_1 - 1):"
        in role
    )
    # One group: four warps, two-stage rings, warp 0 issues the stores.
    assert "block=(32, 6, 1)" in code
    assert "tcgen05_epi_group" not in code
    assert code.count("'num_c_stage': cutlass.Int32(2)") == 2
    assert "setmaxregister_increase(256)" in code
    # A depth that reaches the whole tile is the hoist's job: refused.
    with pytest.raises(InvalidConfig, match="deepest fitting depth 3"):
        normalized(bound, _pair(256, 64, 5, N32, {PREFETCH: 4}))
    assert (
        normalized(bound, _pair(256, 64, 5, N32, {PREFETCH: 4}), fix_invalid=True)[
            PREFETCH
        ]
        == 3
    )


def test_ragged_group_split_guards_the_missing_subtile() -> None:
    bound = bind(gated_pair)
    code = render(
        bound,
        pair({GROUPS: 3, PREFETCH: 2}, tcgen05_c_stages=1),
    )
    role = _epilogue_role(code)
    assert "block=(32, 14, 1)" in code
    assert "NamedBarrier(barrier_id=1, num_threads=416)" in code
    assert "CooperativeGroup(cutlass.pipeline.Agent.Thread, cutlass.Int32(24))" in code
    assert "tcgen05_local_count = (tcgen05_subtile_count + 2) // 3" in role
    assert (
        "_tcgen05_subtile = cutlass.Int32(3 * _tcgen05_local) + tcgen05_epi_group"
        in role
    )
    # The ring index strides by the group's own subtile count per tile
    # (3, 3, 2 for eight subtiles over three groups), so consecutive commits
    # of a group cycle its stages and the bulk-store drain
    # (``wait_group.read(c_stages - 1)``) covers every stage's reuse.
    ragged_index = (
        "(tcgen05_tma_store_role_tile * ((cutlass.Int32(10) - tcgen05_epi_group) "
        "// cutlass.Int32(3)) + cutlass.Int32(_tcgen05_local)) % cutlass.Int32("
    )
    assert (
        f"tcgen05_c_buffer = tcgen05_epi_group * cutlass.Int32(1) + {ragged_index}1)"
        in role
    )
    two_stages = pair({GROUPS: 3, PREFETCH: 2}, tcgen05_c_stages=2, tcgen05_ab_stages=4)
    assert normalized(bound, two_stages).items() >= two_stages.items()
    two_stage_role = _epilogue_role(render(bound, two_stages))
    assert (
        f"tcgen05_c_buffer = tcgen05_epi_group * cutlass.Int32(2) + {ragged_index}2)"
        in two_stage_role
    )
    # The grouped hoist is priced at the long groups' three subtile
    # fragments (192 bytes per thread), like the lowering's ring budget.
    ragged_hoist = pair(
        {GROUPS: 3}, {HOIST: True}, tcgen05_c_stages=2, tcgen05_ab_stages=4
    )
    assert state_of(bound).aux_fragment_hoist_admissible(ragged_hoist)
    assert normalized(bound, ragged_hoist).items() >= ragged_hoist.items()
    assert "tcgen05_aux_ring_0 = (tcgen05_aux_ring_0_0, tcgen05_aux_ring_0_1, " in (
        render(bound, ragged_hoist)
    )
    # Eight subtiles over three groups: the third local subtile exists for
    # groups 0 and 1 only, so the body and the ring's read of it are guarded.
    assert (
        "                if _tcgen05_subtile < cutlass.Int32(tcgen05_subtile_count):\n"
        "                    if cutlass.const_expr(_tcgen05_local + 2 < 3):\n"
        "                        if _tcgen05_subtile + cutlass.Int32(6) < "
        "cutlass.Int32(tcgen05_subtile_count_1):\n"
        "                            cute.autovec_copy(tcgen05_tTR_gAux_grouped_0"
        "[None, None, None, _tcgen05_subtile + cutlass.Int32(6)], "
        "tcgen05_aux_ring_0[(_tcgen05_local + 2) % 3])\n"
        "                    tcgen05_aux_loaded_0 = tcgen05_aux_ring_0"
        "[_tcgen05_local % 3].load()"
    ) in role
    # The two fragments read ahead of the wait (subtiles g and g + 3) exist
    # for every group: no guard.
    assert (
        "cute.autovec_copy(tcgen05_tTR_gAux_grouped_0[None, None, None, "
        "cutlass.Int32(3) + tcgen05_epi_group], tcgen05_aux_ring_0_1)"
    ) in role
    assert "cutlass.Int32(6) + tcgen05_epi_group" not in role
    # The accumulator release after the loop is unguarded: every group's
    # warps arrive.
    assert (
        "            cute.arch.fence_view_async_tmem_load()\n"
        "            with cute.arch.elect_one():\n"
    ) in role


def test_group_barrier_ids_skip_the_pipeline_init_barrier() -> None:
    # The one-CTA merged-init path declares the 96-thread pipeline-init
    # barrier (id 3), and the row-union / CLC kernels own ids 4 and 5; the
    # groups' 128-thread barriers take ids from 6 so no id carries two
    # differently-counted uses.
    assert TCGEN05_EPI_GROUP_BARRIER_BASE_ID == 6
    bound = bind(rows=1024, columns=1024)
    code = render(
        bound,
        normalized(
            bound,
            base_config(
                PRE,
                SHARED,
                {GROUPS: 2, HOIST: True},
                tcgen05_c_stages=1,
                tcgen05_ab_stages=4,
            ),
        ),
    )
    assert (
        "tcgen05_pipeline_init_barrier = cutlass.pipeline.NamedBarrier(barrier_id=3, num_threads=96)"
        in code
    )
    static_ids = {
        int(match) for match in re.findall(r"NamedBarrier\(barrier_id=(\d+),", code)
    }
    assert static_ids == {1, 3}
    group_ids = re.findall(
        r"NamedBarrier\(barrier_id=cutlass\.Int32\((\d+)\) \+ tcgen05_epi_group,", code
    )
    assert group_ids == ["6", "6"]
    assert not static_ids & set(
        range(
            TCGEN05_EPI_GROUP_BARRIER_BASE_ID,
            TCGEN05_EPI_GROUP_BARRIER_BASE_ID + max(TCGEN05_EPI_WARP_GROUP_CHOICES),
        )
    )
    # The one-group render keeps its own id.
    one_group = render(
        bound,
        normalized(
            bound,
            base_config(PRE, SHARED, {HOIST: True}, tcgen05_ab_stages=4),
        ),
    )
    assert (
        "NamedBarrier(barrier_id=cutlass.Int32(2), num_threads=cutlass.Int32(128))"
        in one_group
    )


def test_groups_render_single_output_and_aux_free_epilogues() -> None:
    # A single-output aux epilogue needs no fanout: the wait precedes and the
    # release follows the one grouped loop, the ring prefetches the operand.
    bound = bind(residual_single)
    code = render(
        bound,
        base_config(
            PRE,
            N32,
            {GROUPS: 2, PREFETCH: 1},
            block_sizes=[256, 256, 64],
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=5,
            tcgen05_c_stages=2,
        ),
    )
    role = _epilogue_role(code)
    assert "block=(32, 10, 1)" in code
    assert role.index("tcgen05_acc_pipeline.consumer_wait(") < role.index(
        "for _tcgen05_local in cutlass.range_constexpr("
    )
    assert "tcgen05_aux_ring_0 = (tcgen05_aux_ring_0_0, tcgen05_aux_ring_0_1)" in role
    assert role.count("tcgen05_c_pipeline.producer_commit()") == 1
    # Two outputs without an auxiliary operand: the groups alone.
    no_aux = two_groups()
    no_aux.pop(PLACEMENT)
    no_aux.pop(HOIST)
    code = render(bind(pair_no_aux), no_aux)
    assert "tcgen05_aux_ring" not in code
    assert "tcgen05_epi_group = tcgen05_warp_idx // cutlass.Int32(4)" in code
    assert code.count("if tcgen05_epi_leader:") == 3
    # A plain store at its own narrow subtile takes the groups too.
    code = render(
        bind(plain_single),
        base_config(
            {GROUPS: 2},
            block_sizes=[256, 256, 64],
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=6,
            tcgen05_c_stages=2,
        ),
    )
    assert NARROW_TILE in code and "block=(32, 10, 1)" in code
    assert "tcgen05_local_count = (tcgen05_subtile_count + 1) // 2" in code


def test_subtile_knob_is_admitted_and_priced() -> None:
    bound = bind(gated_pair)
    state = state_of(bound)
    for width in (32, 64):
        strict = normalized(bound, _pair(256, 64, 5, {SUBTILE: width}))
        assert strict[SUBTILE] == width
    with pytest.raises(InvalidConfig, match="must be one of"):
        normalized(bound, _pair(256, 64, 5, {SUBTILE: 16}))
    with pytest.raises(InvalidConfig, match="layout_strategy='default'"):
        normalized(
            bound,
            _pair(
                256,
                64,
                5,
                N32,
                {
                    "tcgen05_layout_strategy": "explicit_epi_tile",
                    "tcgen05_layout_overrides_epi_tile_m": 128,
                    "tcgen05_layout_overrides_epi_tile_n": 32,
                    "tcgen05_layout_overrides_d_store_box_n": 32,
                },
            ),
        )
    # The (128, 32) subtile halves each output's ring (32 -> 16 KiB at two
    # stages): six AB stages fit beside two rings at bk=64; two groups double
    # the stages back (five), one stage per group frees them again (six).
    kwargs: dict[str, Any] = {
        "bm": 256,
        "bn": 256,
        "bk": 64,
        "cluster_m": 2,
        "c_stages": 2,
        "stage_rows": False,
    }

    def deepest(**extra: Any) -> int:
        return max(
            ab
            for ab in range(1, 9)
            if state.default_layout_smem_fits(**(kwargs | extra), ab_stages=ab)
        )

    assert deepest() == 5
    assert deepest(epi_subtile_n=32) == 6
    assert deepest(epi_subtile_n=32, epi_warp_groups=2) == 5
    assert deepest(epi_subtile_n=32, epi_warp_groups=2, c_stages=1) == 6
    assert deepest(epi_subtile_n=64, epi_warp_groups=2) == 3
    assert state.default_layout_smem_bytes(
        **kwargs, ab_stages=1
    ) - state.default_layout_smem_bytes(**kwargs, ab_stages=1, epi_subtile_n=32) == (
        2 * 2 * 128 * 32 * 2
    )
    # The pair admission and the ring judge read the knobs: ab6 is admitted
    # at the narrow subtile and refused at the plan's.
    assert normalized(bound, _pair(256, 64, 6, N32))["tcgen05_ab_stages"] == 6
    with pytest.raises(InvalidConfig, match="SMEM budget"):
        normalized(bound, _pair(256, 64, 6))
    with pytest.raises(InvalidConfig, match="SMEM budget"):
        normalized(bound, two_groups(tcgen05_ab_stages=6))
    assert (
        normalized(bound, two_groups(tcgen05_ab_stages=6, tcgen05_c_stages=1))[
            "tcgen05_ab_stages"
        ]
        == 6
    )
    assert (
        normalized(bound, two_groups(tcgen05_ab_stages=6), fix_invalid=True)[
            "tcgen05_ab_stages"
        ]
        == 5
    )
    # Rings that fit at no AB depth (three groups at the one-CTA 128x256 tile:
    # 192 KiB of two-stage rings) shed a stage per group, then the groups.
    wide = base_config(
        PRE,
        SHARED,
        {GROUPS: 3},
        block_sizes=[128, 256, 64],
        tcgen05_ab_stages=4,
        tcgen05_c_stages=2,
    )
    with pytest.raises(InvalidConfig, match="SMEM budget"):
        normalized(bound, wide)
    shed = normalized(bound, wide, fix_invalid=True)
    assert (shed[GROUPS], shed["tcgen05_c_stages"]) in ((3, 1), (1, 2))
    assert shed["tcgen05_ab_stages"] >= 1
    # The repair pass keeps the subtile and the groups through the search's
    # geometry projections.
    repaired = normalized(bound, two_groups(ON), fix_invalid=True)
    assert repaired[SUBTILE] == 32 and repaired[GROUPS] == 2 and repaired[HOIST] is True
    assert repaired["block_sizes"] == [256, 256, 64]
    assert repaired[FANOUT_CONFIG_KEY] == "shared"


def test_groups_admission_is_strict_and_repaired() -> None:
    bound = bind(gated_pair)
    assert normalized(bound, two_groups())[GROUPS] == 2
    assert normalized(bound, two_groups(tcgen05_c_stages=1))["tcgen05_c_stages"] == 1
    # Several output stores need the shared fanout.
    no_fanout = two_groups()
    no_fanout.pop(FANOUT_CONFIG_KEY)
    no_fanout.pop(PLACEMENT)
    with pytest.raises(InvalidConfig, match="shared fanout"):
        normalized(bound, no_fanout)
    assert normalized(bound, no_fanout, fix_invalid=True)[GROUPS] == 1
    # One stage per group needs two or more groups.
    with pytest.raises(InvalidConfig, match="two or more epilogue warp groups"):
        normalized(bound, pair(tcgen05_c_stages=1))
    assert (
        normalized(bound, pair(tcgen05_c_stages=1), fix_invalid=True)[
            "tcgen05_c_stages"
        ]
        == 2
    )
    # The groups wait ahead of their own loop; the diagnostic placement and
    # the staged aux rings are refused.
    with pytest.raises(InvalidConfig, match="subtile_loop"):
        normalized(bound, two_groups(tcgen05_acc_wait_placement="before_subtile_loop"))
    with pytest.raises(InvalidConfig, match="simt"):
        normalized(bound, two_groups(tcgen05_aux_load_mode="epi_tma"))
    with pytest.raises(InvalidConfig, match="must be one of"):
        normalized(bound, two_groups(**{GROUPS: 5}))
    # The role launch's N width (16 rows for two groups) must divide the tile.
    assert tcgen05_role_launch_n_width(1) == 8
    assert tcgen05_role_launch_n_width(2) == 16
    assert tcgen05_role_launch_n_width(4) == 32
    # Admitted with the one-CTA tile and with the before_store acquire.
    one_cta = base_config(PRE, SHARED, {GROUPS: 2}, tcgen05_c_stages=2)
    assert normalized(bound, one_cta)[GROUPS] == 2
    assert (
        normalized(bound, two_groups(tcgen05_c_acquire_placement="before_store"))[
            GROUPS
        ]
        == 2
    )
    # Three and four groups are admitted (measured slower, unseeded).
    for groups in (3, 4):
        config = pair({GROUPS: groups}, tcgen05_c_stages=1)
        assert normalized(bound, config)[GROUPS] == groups


def test_prefetch_admission_follows_the_register_budget() -> None:
    bound = bind(gated_pair)
    state = state_of(bound)
    # 64 B per (128, 32) fp16 subtile: four fragments fit the 256 B budget.
    assert (
        tcgen05_aux_prefetch_bytes_per_thread(
            epi_tile_m=128, epi_tile_n=32, itemsize=2, depth=3
        )
        == TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD
    )
    assert state.aux_prefetch_depth_limit(_pair(256, 64, 5, N32)) == 3
    # At the plan's (128, 64) subtile two fragments fill the budget.
    assert state.aux_prefetch_depth_limit(_pair(256, 64, 5)) == 1
    assert normalized(bound, _pair(256, 64, 5, {PREFETCH: 1}))[PREFETCH] == 1
    with pytest.raises(InvalidConfig, match="deepest fitting depth 1"):
        normalized(bound, _pair(256, 64, 5, {PREFETCH: 2}))
    assert (
        normalized(bound, _pair(256, 64, 5, {PREFETCH: 2}), fix_invalid=True)[PREFETCH]
        == 1
    )
    # Two groups walk four subtiles each: the ring stops one short of the
    # group's last subtile; the whole-group hoist takes over from there.
    assert state.aux_prefetch_depth_limit(two_groups()) == 3
    assert state.aux_prefetch_depth_limit(_pair(256, 64, 5, N32, {GROUPS: 4})) == 1
    # The hoist supersedes the ring: refused together, the hoist wins in search.
    with pytest.raises(InvalidConfig, match="superseded"):
        normalized(bound, two_groups(ON, **{PREFETCH: 2}))
    both = normalized(bound, two_groups(ON, **{PREFETCH: 2}), fix_invalid=True)
    assert both[HOIST] is True and both[PREFETCH] == 0
    # The one-group hoist of the 256-wide pair is over budget (512 B); the
    # per-group hoist of two groups fits (256 B).
    with pytest.raises(InvalidConfig, match="aux_fragment_hoist=True"):
        normalized(bound, _pair(256, 64, 5, N32, ON))
    assert normalized(bound, two_groups(ON))[HOIST] is True
    # In search the over-budget hoist falls back to the deepest ring.
    repaired = normalized(bound, pair(ON, **{PREFETCH: 4}), fix_invalid=True)
    assert repaired[HOIST] is False and repaired[PREFETCH] == 3
    # Without the pre-wait placement the ring has nothing to run ahead of.
    post = _pair(256, 64, 5, N32, {PREFETCH: 2})
    post[PLACEMENT] = "post_acc_wait"
    with pytest.raises(InvalidConfig, match="pre_acc_wait"):
        normalized(bound, post)
    assert normalized(bound, post, fix_invalid=True)[PREFETCH] == 0
    # Typed.
    with pytest.raises(InvalidConfig, match="must be"):
        normalized(bound, _pair(256, 64, 5, N32, {PREFETCH: -1}))


def test_epi_warp_groups_coverage_group() -> None:
    bound = bind(gated_pair)
    group = _group(bound, GROUPS)
    assert group is not None
    assert group.mechanism == "cute.epi_warp_groups"
    assert group.domain == (1, 2, 3, 4) and group.legacy == 1
    assert [(d.mechanism, d.key, d.value) for d in group.dependencies] == [
        ("cute.epilogue_fanout", FANOUT_CONFIG_KEY, "shared")
    ]
    assert [w.value for w in group.witnesses] == [1, 2, 1, 2]
    carriers = _carriers(group)
    # The widest CTA-pair geometry at the (128, 32) subtile with the deepest
    # ring both group counts admit, and the fanout group's one-CTA carrier.
    pair, one = carriers[0], carriers[2]
    assert carriers[1] == pair and carriers[3] == one
    assert pair["block_sizes"] == [256, 256, 64]
    assert pair["tcgen05_cluster_m"] == 2
    assert pair[SUBTILE] == 32 and pair[PREFETCH] == 3
    assert pair[FANOUT_CONFIG_KEY] == "shared"
    assert pair[PLACEMENT] == "pre_acc_wait" and pair[HOIST] is False
    assert pair["tcgen05_ab_stages"] == 5 and pair["tcgen05_c_stages"] == 2
    assert one["block_sizes"] == [128, 128, 64] and one["tcgen05_cluster_m"] == 1
    # The one-CTA 128x128 tile's (128, 32) subtiles: two per group, so the
    # ring both group counts admit is one deep.
    assert one.get(SUBTILE) is None and one[PREFETCH] == 1
    for witness in group.witnesses:
        config = dict(witness.carrier.config) | {GROUPS: witness.value}
        strict = normalized(bound, config)
        assert strict[GROUPS] == witness.value
        assert {k: v for k, v in strict.items() if k != GROUPS} == {
            k: v
            for k, v in normalized(bound, dict(witness.carrier.config)).items()
            if k != GROUPS
        }
    # The best measured recipe is one hoist flip from the two-group witness.
    best = normalized(bound, dict(pair) | {GROUPS: 2, HOIST: True}, fix_invalid=True)
    assert best[HOIST] is True and best[PREFETCH] == 0 and best[GROUPS] == 2
    # The hoist group's own carriers are unchanged by the new group.
    hoist = _group(bound, HOIST)
    assert hoist is not None
    assert [
        (tuple(c["block_sizes"]), c["tcgen05_cluster_m"]) for c in _carriers(hoist)
    ] == [((256, 128, 128), 2)] * 2 + [((128, 128, 64), 1)] * 2
    assert all(c.get(GROUPS, 1) == 1 for c in _carriers(hoist))
    # A plain store has no heavy epilogue: no group, the knob off the search.
    assert _group(bind(plain_single), GROUPS) is None
    assert not state_of(bind(plain_single)).epilogue_group_search_enabled()
