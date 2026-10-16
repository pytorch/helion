"""``cute_flash_kv_split``: the FlexAttention list walk's KV walk split over
two or four CTAs per work item. The grid becomes items x splits (the split
id is the low bits of the CTA id, or of the persistent work unit); the
CTA-wide setup trims the union bitmask to the tiles of the split's rank
range in walk order and every role runs the unchanged item protocol over the
slice. Each split's correction warps store the unnormalized fp32 partial
output and the row statistics of both stages to a host-allocated workspace
(``flex_split_workspace``), count their arrival on a per (item, stage, 32-row
warp group) counter and the last-arriving warp merges the partials in split
order (``O = sum_s O_s c_s / sum_s l_s c_s``, ``c_s = 2^(m_s - max m)``),
writes O and the LSE (zeros / ``-inf`` for empty rows) and resets the
counter; the epilogue warp stores nothing. Composes with the dead-stage split
(decided per split CTA on its slice). The fragment default (the seeds' value)
follows ``min(4, ceil(SMs / items))`` so a small grid starts split; a config
without the key renders byte-identically to the unsplit body.
"""

from __future__ import annotations

import ast
import os
import re
from unittest.mock import patch

import pytest
import torch

from test.test_cute_flash_flex_skip_absent import BODY
from test.test_cute_flash_flex_skip_absent import HEAD_DIM
from test.test_cute_flash_flex_skip_absent import NUM_BH
from test.test_cute_flash_flex_skip_absent import _active
from test.test_cute_flash_flex_skip_absent import _fragments
from test.test_cute_flash_flex_skip_absent import _plan
from test.test_cute_flash_flex_skip_absent import _render
from test.test_cute_flash_flex_skip_absent import _resolve
from test.test_cute_flash_flex_walk import _codegen_with_flex_plan
from test.test_cute_flash_flex_walk import _stub_hook  # noqa: F401  (autouse)

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import FLEX_SCORE_KIND
from helion._compiler.cute.attention_plan import dense_score_plan
from helion.exc import InvalidConfig
from helion.runtime.cute import flash_flex

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

KEY = cute_flash.FLASH_KV_SPLIT_KEY
SKIP = cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY
DEAD = cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY
PERSISTENT = cute_flash.FLASH_PERSISTENT_KEY
PARTIAL_STORE = "_helion_flash_rt.fa4_split_partial_store("
MERGE_STORE = "_helion_flash_rt.fa4_split_merge_store("
ATOMIC = "cute.arch.atomic_add(_flash_mSplitCnt.iterator + "
SPLIT_MARKERS = (
    "flash_kv_split_id",
    "flash_flex_total",
    "flash_split_item",
    "flash_split_sid",
    "_flash_mSplitWs",
    "_flash_mSplitCnt",
    PARTIAL_STORE,
    MERGE_STORE,
    ATOMIC,
)
# The protocol's barrier arrivals an unsplit render makes per step and item.
ARRIVALS = (
    "cute_tcgen05_flash.commit(flash_s_full_ptr + 0)",
    "cute_tcgen05_flash.commit(flash_o_full_ptr + ",
    "mbar_spin_wait(flash_pfor_ptr + ",
    "mbarrier_arrive(flash_pfor_ptr + ",
    "named_barrier_arrive_unaligned(3 + warp_idx % 4, 64)",
    "named_barrier_arrive_unaligned(7 + warp_idx % 4, 64)",
    "mbarrier_arrive(flash_corr_epi_full_ptr + ",
    "mbarrier_arrive(flash_corr_epi_empty_ptr + ",
    "mbar_spin_wait(flash_corr_epi_full_ptr + ",
    "mbar_spin_wait(flash_corr_epi_empty_ptr + ",
    "fa4_disc_zero_store(",
)


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


def _corr_block(src: str) -> str:
    return next(
        block for guard, block in _roles(src).items() if "warp_idx >= 8" in guard
    )


def _epi_block(src: str) -> str:
    return next(
        block
        for guard, block in _roles(src).items()
        if guard != "setup" and "flash_corr_epi_full_phase = cutlass.Int32(0)" in block
    )


def test_resolver_defaults_to_one_and_pins_it_off_the_list_walk() -> None:
    assert _resolve(BODY).kv_split == 1
    assert _resolve(None).kv_split == 1
    for value in cute_flash.FLASH_KV_SPLITS:
        assert _resolve({**BODY, KEY: value}).kv_split == value
    with pytest.raises(InvalidConfig, match="cute_flash_kv_split must be one of"):
        _resolve({**BODY, KEY: 3})
    # Only the list walk's fa4 pair body splits: dense, causal, pruned and
    # modifier plans pin the knob to 1 whatever the request.
    assert _resolve({**BODY, KEY: 4}, has_list_walk=False).kv_split == 1
    assert (
        _resolve(
            {**BODY, KEY: 4},
            has_list_walk=False,
            is_causal=True,
            standard_dense_output=False,
            standard_causal_output=True,
        ).kv_split
        == 1
    )
    assert (
        _resolve(
            {**BODY, KEY: 2},
            has_list_walk=False,
            has_kv_tile_pruning=True,
            standard_dense_output=False,
        ).kv_split
        == 1
    )
    # The merge reads each stage's row max from the second statistics slot:
    # the single-slot transports pin the knob off.
    single = _resolve({**BODY, KEY: 2, cute_flash.FLASH_STAT_TRANSPORT_KEY: "single"})
    assert single.stat_transport in ("single", "ring2")
    if single.stat_transport == "single":
        assert single.kv_split == 1
    # The scoped staged correction epilogue carries the merge: the debug
    # routes that turn it off pin the knob to 1.
    assert (
        _resolve(
            {**BODY, KEY: 2}, environ={"HELION_CUTE_FLASH_SCOPED_CORR_EPI": "0"}
        ).kv_split
        == 1
    )


def test_knob_is_an_appended_schema_axis_with_fa4_witnesses() -> None:
    keys = cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS
    assert keys.count(KEY) == 1
    assert keys.index(KEY) > keys.index(cute_flash.FLASH_FLEX_MASK_FORM_KEY)
    for value in (2, 4):
        witness = cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES[(KEY, value)]
        assert witness == {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
        assert _resolve({**witness, KEY: value}).kv_split == value
    assert (KEY, 1) not in cute_flash.FLASH_AUTOTUNE_VALUE_DEPENDENCIES
    for value in cute_flash.FLASH_KV_SPLITS:
        resolved = _resolve({**BODY, KEY: value})
        assert cute_flash.flash_effective_config_values(resolved)[KEY] == value


def test_fragments_search_every_split_on_list_walk_plans_only() -> None:
    flex = _fragments()[KEY]
    assert set(_active(flex)) == set(cute_flash.FLASH_KV_SPLITS)
    # No device: nothing seeds a split.
    assert flex.default() == 1
    dense = _fragments(has_list_walk=False)[KEY]
    assert _active(dense) == (1,) and dense.default() == 1
    causal = _fragments(
        has_list_walk=False,
        is_causal=True,
        standard_causal_output=True,
    )[KEY]
    assert _active(causal) == (1,) and causal.default() == 1


@pytest.mark.parametrize(
    ("device_sm_count", "q_seq", "num_bh", "expected"),
    [
        # 64 items on 148 SMs: 128 CTAs fit one wave, 256 would take two.
        (148, 128, 64, 2),
        # 37 items: 148 CTAs are exactly one wave.
        (148, 128, 37, 4),
        # 128 items: already most of a wave, a split would start a second.
        (148, 512, 64, 1),
        # 512 items cover the device.
        (148, 2048, 64, 1),
        # Exactly one item per SM.
        (148, 128, 148, 1),
        # One item, any device: the largest split.
        (148, 64, 1, 4),
        # Unknown device or query length: unsplit.
        (0, 128, 64, 1),
        (148, None, 64, 1),
    ],
)
def test_fragment_default_seeds_the_split_from_the_sm_count(
    device_sm_count: int, q_seq: int | None, num_bh: int, expected: int
) -> None:
    def fragments_at(num_kv: int) -> object:
        with patch.dict(os.environ, {}, clear=True):
            return cute_flash.flash_autotune_fragments(
                HEAD_DIM,
                num_kv,
                dtype=torch.bfloat16,
                num_bh=num_bh,
                has_list_walk=True,
                score_modifier_kinds=(FLEX_SCORE_KIND,),
                device_sm_count=device_sm_count,
                q_seq=q_seq,
            )[KEY]

    fragment = fragments_at(16)
    assert fragment.default() == expected  # pyrefly: ignore[missing-attribute]
    assert set(_active(fragment)) == set(cute_flash.FLASH_KV_SPLITS)  # pyrefly: ignore[bad-argument-type]
    assert (
        cute_flash._flash_kv_split_seed(
            device_sm_count=device_sm_count, num_bh=num_bh, q_seq=q_seq
        )
        == expected
    )
    # The KV length never enters the seed.
    assert fragments_at(65).default() == expected  # pyrefly: ignore[missing-attribute]


def test_every_flex_seed_carries_the_seeded_split() -> None:
    plan: dict[str, object] = {
        "dtype": torch.bfloat16,
        "num_bh": NUM_BH,
        "has_list_walk": True,
        "score_modifier_kinds": (FLEX_SCORE_KIND,),
        "plain_row_body": False,
    }
    with patch.dict(os.environ, {}, clear=True):
        # 64 items on 296 SMs: four splits fill one wave.
        seeds = cute_flash.flash_attention_seed_configs(
            HEAD_DIM,
            17,
            device_sm_count=296,
            q_seq=128,
            **plan,  # pyrefly: ignore[bad-argument-type]
        )
        assert len(seeds) >= 3
        # Every fragment-derived seed (the family defaults, the pruned
        # campaign structures) carries the seeded split; the target policy's
        # measured seed, when there is one, keeps its own table values.
        target = cute_flash.flash_attention_seed_config(
            HEAD_DIM,
            17,
            **plan,  # pyrefly: ignore[bad-argument-type]
        )
        derived = [seed for seed in seeds if seed != target]
        assert len(derived) >= len(seeds) - 1
        assert all(seed.config[KEY] == 4 for seed in derived)
        resolved = [
            cute_flash.resolve_flash_config(
                HEAD_DIM,
                17,
                seed.config,
                **plan,  # pyrefly: ignore[bad-argument-type]
            )
            for seed in derived
        ]
        assert all(cfg.kv_split == 4 for cfg in resolved)
        # Without a device every seed stays unsplit.
        plain = cute_flash.flash_attention_seed_configs(
            HEAD_DIM,
            17,
            **plan,  # pyrefly: ignore[bad-argument-type]
        )
        assert all(seed.config[KEY] == 1 for seed in plain)


@pytest.mark.parametrize("num_kv", (16, 17, 64))
def test_resolved_value_is_length_invariant(num_kv: int) -> None:
    for value in cute_flash.FLASH_KV_SPLITS:
        resolved = _resolve({**BODY, KEY: value}, num_kv=num_kv)
        assert resolved.kv_split == value
        assert cute_flash.flash_effective_config_values(resolved) == (
            cute_flash.flash_effective_config_values(_resolve({**BODY, KEY: value}))
        )


def test_unsplit_render_carries_no_split_code() -> None:
    plain = _render(_resolve(BODY), _plan(q_seq=2048, kv_seq=2176))
    one = _render(_resolve({**BODY, KEY: 1}), _plan(q_seq=2048, kv_seq=2176))
    assert plain == one
    for marker in SPLIT_MARKERS:
        assert marker not in plain, marker
    dense = dense_score_plan(HEAD_DIM)
    assert _render(_resolve({**BODY, KEY: 4}, has_list_walk=False), dense) == _render(
        _resolve(BODY, has_list_walk=False), dense
    )


@pytest.mark.parametrize("ragged", [False, True], ids=["aligned", "ragged"])
@pytest.mark.parametrize("dead_split", [False, True], ids=["plain", "dead-split"])
@pytest.mark.parametrize("kv_split", [2, 4])
def test_render_trims_the_walk_and_merges_in_the_correction_warps(
    kv_split: int, dead_split: bool, ragged: bool
) -> None:
    plan = _plan(q_seq=128 + 256 * 3 if ragged else 2048, kv_seq=8320)
    words = 3
    base = {**BODY, SKIP: dead_split, DEAD: dead_split}
    unsplit = _render(_resolve(base), plan)
    cfg = _resolve({**base, KEY: kv_split})
    assert cfg.kv_split == kv_split and cfg.flex_dead_stage_split is dead_split
    split = _render(cfg, plan)
    roles = _roles(split)
    setup = roles["setup"]
    shift = kv_split.bit_length() - 1
    # The CTA id's low bits are the split id; the LPT decode keeps the rest.
    assert (
        f"flash_kv_split_id = flash_pid & cutlass.Int32({kv_split - 1})\n"
        f"flash_pid = flash_pid >> cutlass.Int32({shift})\n"
    ) in setup
    assert "flash_lpt_group = flash_pid // " in setup
    # The union is trimmed to the slice's rank range in walk order: the
    # drop loop, the keep loop (one select per word), the rebuilt union, the
    # slice's count and the item's workspace index.
    assert (
        f"flash_flex_per = flash_flex_total + cutlass.Int32({kv_split - 1}) "
        f">> cutlass.Int32({shift})"
    ) in setup
    assert "for flash_flex_i in cutlass.range(flash_flex_lo, unroll=1):" in setup
    assert "for flash_flex_i in cutlass.range(flash_flex_cnt, unroll=1):" in setup
    assert setup.count("flash_flex_drop = cutlass.Int32(cutlass.select_(") == words
    assert setup.count("flash_flex_keep = cutlass.Int32(cutlass.select_(") == words
    for word in range(words):
        assert f"flash_flex_u{word} = flash_flex_k{word}" in setup
    assert (
        "flash_num_active_kv = cutlass.max(cutlass.Int32(1), flash_flex_cnt)" in setup
    )
    # The split id and the item index are role-local in the correction warps
    # (two values live across every role spilled to local memory).
    assert "flash_split_item" not in setup
    # The descending walk takes the highest set bit in the drop loop as well.
    assert "cutlass.Int32(95) - cutlass.Int32(cute.arch.clz(flash_flex_u2))" in setup
    # The per-stage lists (and the presence masks under the skip) follow the
    # slice, so an empty slice's fake step finds both stages absent.
    for stage in ("0", "1"):
        for kind in ("p", "f") + (("n",) if dead_split else ()):
            for word in range(words):
                assert (
                    f"flash_flex_{kind}{stage}_{word} = flash_flex_{kind}{stage}_{word} "
                    f"& flash_flex_u{word}"
                ) in setup
    assert ("flash_flex_n0_0 = flash_flex_n0_0 & flash_flex_u0" in setup) is dead_split
    if dead_split:
        # The dead-stage decision reads the (sliced) second-stage presence.
        assert (
            "flash_flex_split = cutlass.Int32(flash_flex_n1_0 | flash_flex_n1_1 | "
            "flash_flex_n1_2 == 0)"
        ) in setup
    # Both softmax warpgroups publish their row max; neither stores the LSE.
    for stage in ("0", "1"):
        assert (
            f"flash_scale_t[flash_s_corr_prod_index ^ 1, {stage}, flash_local_tidx] "
            "= flash_row_max"
        ) in split
    corr = _corr_block(split)
    assert split.count("_flash_mLSE[") == 2 == corr.count("_flash_mLSE[")
    assert (
        corr.count(
            f"flash_split_sid = cutlass.Int32(cute.arch.block_idx()[0]) & "
            f"cutlass.Int32({kv_split - 1})"
        )
        == 1
    )
    assert (
        corr.count(
            f"flash_split_item = flash_bh * {-(-plan.q_seq // 256)} + flash_m_pair"
        )
        == 1
    )
    assert "flash_kv_split_id" not in corr
    # Correction warps: one partial store per stage (the dead-stage split
    # adds the two-accumulator form on a split item), the row statistics,
    # the acquire-release arrival count per warp, the merge of every split
    # in fixed order by the last arriving warp, the counter reset.
    assert corr.count(PARTIAL_STORE) == (3 if dead_split else 2)
    assert corr.count("flash_pvt, tOtO0, tOtO1, flash_split_gP") == int(dead_split)
    assert corr.count(MERGE_STORE) == 2
    assert corr.count(ATOMIC) == 2
    assert corr.count("sem='acq_rel', scope='gpu'") == 2
    # Release/acquire ride on the atomic and the warp barriers alone.
    assert "fence_acq_rel_gpu" not in corr
    assert corr.count("cute.arch.sync_warp()") == 4
    assert (
        corr.count(
            f"if cute.arch.vote_ballot_sync(flash_split_old == cutlass.Int32({kv_split - 1})) != 0:"
        )
        == 2
    )
    assert corr.count("] = cutlass.Int32(0)") == 2
    for stage in ("0", "1"):
        for index in range(kv_split):
            assert f"flash_split_c{index} = cute.math.exp2(" in corr
            assert (
                f"flash_split_gP{index} = cute.make_tensor(_flash_mSplitWs.iterator + "
                f"cute.assume(((flash_split_item * {kv_split} + {index}) * 2 + {stage}) * 8192, divby=4)"
            ) in corr
        assert (
            f"_flash_mLSE[flash_m_tile{stage} * 128 + flash_local_tidx, flash_bh] = "
            "flash_split_mm * _flash_scale_log2 + cute.math.log2(flash_split_L)"
        ) in corr
    weights = ", ".join(
        f"flash_split_c{index} * flash_split_invL" for index in range(kv_split)
    )
    assert corr.count(f"({weights}), flash_split_gO") == 2
    # The output tile is addressed over the plain (S, D, B) view.
    assert corr.count("flash_split_gO = cute.make_tensor(_flash_mO.iterator + ") == 2
    # Empty-row statistics of the dead-stage split's second stage.
    assert (
        "flash_split_pm1 = cutlass.Float32(cutlass.select_(flash_flex_split != 0, "
        "-cutlass.Float32.inf, flash_split_m1))" in corr
    ) is dead_split
    # Ragged query lengths guard the partial store and the arrival (a stage
    # without live rows has no output) and the merged stores.
    for stage in ("0", "1"):
        assert (
            corr.count(f"if flash_live_rows{stage} > cutlass.Int32(0):") == 2
        ) is ragged
        assert (f"if flash_local_tidx < flash_live_rows{stage}:" in corr) is ragged
    # The epilogue warp keeps the staged handshake and stores nothing.
    epi = _epi_block(split)
    assert "cute.copy(_flash_tma_o" not in epi
    assert "fa4_store_o_smem_to_gmem" not in epi
    assert epi.count("mbar_spin_wait(flash_corr_epi_full_ptr + ") == 2
    assert epi.count("mbarrier_arrive(flash_corr_epi_empty_ptr + ") == 4
    # Every barrier arrival of the protocol is kept.
    for arrival in ARRIVALS:
        assert split.count(arrival) == unsplit.count(arrival), arrival


def test_row_sum_credit_wait_is_unconditional_on_the_list_walk() -> None:
    """The ring2 final row-sum publish waits for the last alpha's slot on
    every list-walk item, including a one-step (fake) item: the split-KV and
    dead-stage merges publish the row max into that other slot and the
    persistent walk carries the ring across items, so the dense body's
    ``if num_kv > 1`` guard would leave the slot unacquired (the persistent
    split-KV walk hung on its first launch)."""
    plan = _plan(q_seq=2048 + 128, kv_seq=2048)
    for config in (BODY, {**BODY, KEY: 2}, {**BODY, KEY: 2, PERSISTENT: True}):
        src = _render(_resolve(config), plan)
        assert "if flash_num_active_kv > 1:" not in src
        for stage in ("0", "1"):
            assert (
                f"_helion_flash_rt.mbar_spin_wait(flash_s{stage}_corr_empty_ptr + "
                "(flash_s_corr_prod_index ^ 1), flash_s_corr_prod_phase ^ "
                "flash_s_corr_prod_index"
            ) in src


def test_persistent_render_decodes_the_split_from_the_work_unit() -> None:
    plan = _plan(q_seq=2048 + 128, kv_seq=2048)
    cfg = _resolve({**BODY, KEY: 2, PERSISTENT: True})
    assert cfg.persistent and cfg.kv_split == 2
    split = _render(cfg, plan)
    units = NUM_BH * 9 * 2
    # Every role's item loop runs over items x splits work units (the
    # epilogue and empty warps too: their per-item handshakes pace the
    # walking roles), the split id in the low bit and the host-sorted item
    # in the rest.
    assert split.count(f"while flash_tile_id < {units}:") == 7
    assert f"while flash_tile_id < {NUM_BH * 9}:" not in split
    assert split.count("flash_kv_split_id = flash_tile_id & cutlass.Int32(1)") == 5
    # The correction warps take the split id from the work unit.
    assert split.count("flash_split_sid = flash_tile_id & cutlass.Int32(1)") == 1
    assert (
        split.count(
            "flash_item = cutlass.Int32(_flash_mFlexItems[flash_tile_id >> cutlass.Int32(1)])"
        )
        == 5
    )
    assert "flash_pid" not in split
    # The per-role prelude trims the slice in every walking role.
    assert (
        split.count(
            "flash_num_active_kv = cutlass.max(cutlass.Int32(1), flash_flex_cnt)"
        )
        >= 4
    )


def _plan_dict(code: str) -> dict[str, object]:
    marker = "._helion_cute_wrapper_plans = "
    payload = next(line for line in code.splitlines() if marker in line).split(
        marker, 1
    )[1]
    (plan,) = ast.literal_eval(payload)
    assert isinstance(plan, dict)
    return plan


def _wrapper_lines(plan: dict[str, object]) -> tuple[list[str], list[str]]:
    from helion.runtime.cute.launcher import _append_cute_wrapper_plan

    body: list[str] = []
    call_args: list[str] = []
    _append_cute_wrapper_plan(body, call_args, plan, num_sm=148)
    return body, call_args


@pytest.mark.parametrize("epi_tma", [True, False], ids=["tma", "staged"])
@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
def test_codegen_binds_the_workspace_and_scales_the_grid(
    epi_tma: bool, persistent: bool
) -> None:
    seq = 2048 + 128
    items = 8 * -(-seq // 256)
    config = {
        "block_sizes": [1, 128, 128],
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_EPI_TMA_KEY: epi_tma,
        cute_flash.FLASH_EPI_STG_KEY: not epi_tma,
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        PERSISTENT: persistent,
        KEY: 2,
    }
    code = _codegen_with_flex_plan(seq, config)
    plan = _plan_dict(code)
    assert plan["kv_split"] == 2 and plan["total_tiles"] == items
    # The wrapper allocates (and caches) the workspace per call and passes
    # it as one plain kernel param; the body binds the counters as the int32
    # tail of the fp32 buffer.
    assert (
        code.count(
            f"_helion_flash_flex_split_workspace(q_view, items={items}, splits=2, "
            "head_dim_v=64)"
        )
        == 1
    )
    assert "from helion.runtime.cute.flash_flex import flex_split_workspace" in code
    signature = code.split("def _helion__probe_attention(")[1].split(")")[0]
    params = [param.strip() for param in signature.split(",")]
    assert "flash_split_ws" in params and "flash_split_cnt" not in params
    assert "_flash_mSplitWs = flash_split_ws" in code
    partial_elems, stat_elems, counter_elems = flash_flex.flex_split_workspace_elems(
        items, 2, 64
    )
    assert (
        "_flash_mSplitCnt = cute.make_tensor(cute.recast_ptr(_flash_mSplitWs.iterator + "
        f"{partial_elems + stat_elems}, dtype=cutlass.Int32), "
        f"cute.make_layout(({counter_elems},), stride=(1,)))"
    ) in code
    assert MERGE_STORE in code
    # The TMA epilogue passes the plain output view for the merged store.
    assert ("_flash_mO" in params) is epi_tma
    assert plan.get("epi_tma_ragged", False) is epi_tma
    body, call_args = _wrapper_lines(plan)
    source = "\n".join(body)
    if persistent:
        assert f"grid_x = cutlass.Int32({min(items * 2, 148)})" in source
    else:
        assert f"grid_x = cutlass.Int32({items * 2})" in source
    assert ("_flash_mO" in call_args) is epi_tma
    if epi_tma:
        assert call_args[call_args.index("_flash_osl") + 1] == "_flash_mO"
    # Unsplit: no workspace, the plain grid.
    plain = _codegen_with_flex_plan(seq, {**config, KEY: 1})
    plain_plan = _plan_dict(plain)
    assert "kv_split" not in plain_plan
    assert "_helion_flash_flex_split_workspace" not in plain
    assert MERGE_STORE not in plain
    plain_source = "\n".join(_wrapper_lines(plain_plan)[0])
    if persistent:
        assert f"grid_x = cutlass.Int32({min(items, 148)})" in plain_source
    else:
        assert f"grid_x = cutlass.Int32({items})" in plain_source


def test_codegen_declines_the_split_past_the_workspace_budget() -> None:
    config = {
        "block_sizes": [1, 128, 128],
        cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        cute_flash.FLASH_EPI_TMA_KEY: True,
        cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        KEY: 4,
    }
    # 2048 rows, 8 heads: 64 items x 4 splits x 2 stages x 128 x 66 fp32 =
    # 17 MB of workspace; a 1 MB budget declines the split (an unsplit
    # render, the knob stays in the config).
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_KV_SPLIT_WORKSPACE_MB": "1"}):
        code = _codegen_with_flex_plan(2048, config)
    assert "kv_split" not in _plan_dict(code)
    assert "_helion_flash_flex_split_workspace" not in code
    assert MERGE_STORE not in code
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_KV_SPLIT_WORKSPACE_MB": "64"}):
        code = _codegen_with_flex_plan(2048, config)
    assert _plan_dict(code)["kv_split"] == 4
    assert MERGE_STORE in code


def test_workspace_helper_caches_per_device_and_zeroes_the_counters() -> None:
    like = torch.empty(4, dtype=torch.bfloat16)
    workspace = flash_flex.flex_split_workspace(like, items=3, splits=2, head_dim_v=64)
    partial_elems, stat_elems, counter_elems = flash_flex.flex_split_workspace_elems(
        3, 2, 64
    )
    assert (partial_elems, stat_elems, counter_elems) == (
        3 * 2 * 2 * 128 * 64,
        3 * 2 * 2 * 128 * 2,
        3 * 2 * 4,
    )
    assert workspace.dtype is torch.float32
    assert workspace.numel() == partial_elems + stat_elems + counter_elems
    counters = workspace[partial_elems + stat_elems :].view(torch.int32)
    assert torch.equal(counters, torch.zeros_like(counters))
    again = flash_flex.flex_split_workspace(like, items=3, splits=2, head_dim_v=64)
    assert again is workspace
    other = flash_flex.flex_split_workspace(like, items=3, splits=4, head_dim_v=64)
    assert other is not workspace and other.numel() > workspace.numel()
