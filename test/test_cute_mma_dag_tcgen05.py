"""Codegen, admission and search-surface pins for the ``cute_mma_dag``
tcgen05 compute mode (the TMEM-resident serial state recurrence).

CPU-only (fake sm100 target) like ``test_cute_mma_dag.py``, whose binders
and fixture this file reuses.  Runtime coverage lives in
``test_cute_mma_dag_tcgen05_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import os
import re

import pytest
import torch

from test.test_cute_mma_dag import _bind_bwd
from test.test_cute_mma_dag import _bind_dqk
from test.test_cute_mma_dag import _bind_dv
from test.test_cute_mma_dag import _bind_fwd
from test.test_cute_mma_dag import _bind_fwd_o
from test.test_cute_mma_dag import _kernel_body
from test.test_cute_mma_dag import _source
from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute import cute_mma_dag_tcgen05 as tc
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="codegen pins and search-surface checks need the compiled path",
    ),
]

RT = "_helion_mma_dag_tc"
P = f"{RT}.prims"
WARP_RT = "_helion_mma_dag"


def _config(
    *block_sizes: int, warps: int = 8, stages: int = 4, **extra: object
) -> helion.Config:
    return helion.Config(
        block_sizes=list(block_sizes),
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=warps,
        cute_mma_dag_stages=stages,
        **extra,
    )


def _loop(body: str) -> str:
    return body[body.index("for md_i in cutlass.range(") :]


def _assert_in_order(text: str, items: list[str]) -> None:
    """Every item occurs after the previous one (repeated items are the
    next occurrence)."""
    position = 0
    for item in items:
        position = text.index(item, position) + 1


# --------------------------------------------------------------------------
# Admission and the search surface
# --------------------------------------------------------------------------


def test_state_kernels_admit_the_mode_and_seed_it_first() -> None:
    for bound in (_bind_fwd(), _bind_bwd("scalar"), _bind_fwd("diag", has_h0=True)):
        spec = bound.config_spec
        assert spec.cute_mma_dag is not None
        assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        # Two loop operand tiles, no epilogue tiles and no fragment operands.
        assert [t.region for t in facts.tcgen05.tiles] == ["loop", "loop"]
        assert facts.tcgen05.exchanges == ()
        seeds = spec.compiler_seed_configs
        first = seeds[0].config
        assert first["cute_mma_dag"] == "tcgen05"
        assert first["cute_mma_dag_warps"] == 4 and first["cute_mma_dag_stages"] == 8
        assert first["block_sizes"] == [64, 64]
        second = seeds[1].config
        assert second["cute_mma_dag"] == "tcgen05"
        assert second["cute_mma_dag_warps"] == 8 and second["block_sizes"] == [64, 64]
        modes = [c.config.get("cute_mma_dag") for c in seeds]
        assert modes.index("warp") > modes.count("tcgen05") - 1
        assert {
            c.config["cute_mma_dag_stages"]
            for c in seeds
            if c.config.get("cute_mma_dag") == "tcgen05"
        } == {8, 4}
        assert spec.default_config().config["cute_mma_dag"] == "tcgen05"


def test_other_targets_and_k_loop_programs_do_not_admit_the_mode() -> None:
    bound = _bind_fwd()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        assert tc.tcgen05_facts(program) is not None
        spec.target_device_capability = (9, 0)
        assert tc.tcgen05_facts(program) is None
        spec.target_device_capability = (12, 0)
        assert tc.tcgen05_facts(program) is None
        spec.target_device_capability = (10, 3)
        assert tc.tcgen05_facts(program) is not None
    # Without the facts the knob never offers the mode.
    spec.enable_cute_mma_dag_search(dataclasses.replace(facts, tcgen05=None))
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    with bound.env, pytest.raises(InvalidConfig, match="cute_mma_dag must be one of"):
        spec.normalize(_config(64, 64).config)


def test_k_loop_kernels_admit_the_mode_with_their_tiles() -> None:
    # fwd_o: three loop tiles (q shared by both dots, h, k), the epilogue v
    # tile and the masked attention fragment read as the epilogue A operand.
    facts = _bind_fwd_o().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    tc_facts = facts.tcgen05
    assert [t.region for t in tc_facts.tiles] == ["loop", "loop", "loop", "epilogue"]
    assert [(x.slot, x.elem_bytes) for x in tc_facts.exchanges] == [(1, 2)]
    assert tc_facts.loop_stores == () and tc_facts.epilogue_stores == ((0, 2),)
    # dqk: the dh tile is converted from fp32, dA feeds both epilogue dots
    # from one exchange tile, two epilogue stores.
    facts = _bind_dqk().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    tc_facts = facts.tcgen05
    assert [t.source_bytes for t in tc_facts.tiles if t.region == "loop"] == [
        2,
        2,
        2,
        4,
    ]
    # dA read straight and transposed shares one exchange tile (the
    # transpose is a property of the read).
    assert [x.slot for x in tc_facts.exchanges] == [0]
    assert tc_facts.epilogue_stores == ((1, 2), (2, 2))
    for bound in (_bind_fwd_o(), _bind_dqk(), _bind_dv("scalar")):
        spec = bound.config_spec
        assert spec.cute_mma_dag is not None
        assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
        first = spec.compiler_seed_configs[0].config
        assert first["cute_mma_dag"] == "tcgen05"
        assert spec.default_config().config["cute_mma_dag"] == "tcgen05"


def test_k_loop_config_reasons() -> None:
    facts = _bind_dqk().config_spec.cute_mma_dag_facts
    assert facts is not None
    # Block ids: lead 0, root D 1, loop DV 2; the rdim C is fixed at 64.
    assert (
        dag.mma_dag_config_reason(facts, {0: 1, 1: 64, 2: 64}, 4, 2, "tcgen05") is None
    )
    # Three accumulators of 128 columns fit TMEM; the stage ring does not
    # at eight stages of 128-wide converted tiles.
    reason = dag.mma_dag_config_reason(facts, {0: 1, 1: 128, 2: 128}, 4, 8, "tcgen05")
    assert reason is not None and "shared memory" in reason
    assert tc.tcgen05_slot_columns(facts, [128, 1, 128, 64]) == 64 + 128 + 128
    # Wider than 256 columns per accumulator or more than 512 in total.
    reason = tc.tcgen05_config_reason(facts, [64, 1, 512, 64], 4, 1)
    assert reason is not None and "multiples of 8 up to 256" in reason
    reason = tc.tcgen05_config_reason(facts, [64, 1, 256, 64], 4, 1)
    assert reason is not None and "TMEM columns" in reason
    # A fragment operand needs a swizzled row: the [C, C] exchange always has
    # one; a 24-column operand tile has none.
    reason = tc.tcgen05_config_reason(facts, [24, 1, 64, 64], 4, 1)
    assert reason is not None and "no swizzled UMMA layout" in reason


def test_chained_loop_accumulators_stay_on_the_warp_mode() -> None:
    # The loop issues an iteration's dots together: a dot accumulating onto
    # an earlier loop dot's result needs the warp mode's program-order body.
    from test.test_cute_mma_dag import _bind_fn
    from test.test_cute_mma_dag import _tensors
    from test.test_cute_mma_dag import _two_dots_same_carry

    t = _tensors()
    spec = _bind_fn(_two_dots_same_carry, (t["k"], t["v"])).config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    assert spec.default_config().config["cute_mma_dag"] == "warp"


def _facts() -> dag.CuteMmaDagFacts:
    facts = _bind_fwd(d=128, dv=128).config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    return facts


def _sizes(bm: int, bn: int) -> dict[int, object]:
    return {0: 1, 1: bm, 2: bn}


# Block ids of the fwd kernel: lead 0, root M 1, root N 2, rdim K 4.
def _resolved(bm: int, bn: int) -> dict[int, int]:
    return {0: 1, 1: bm, 2: bn, 4: 64}


@pytest.mark.parametrize(
    ("bm", "bn", "warps", "stages", "reason"),
    (
        (64, 64, 2, 4, "must be one of \\(4, 8\\) for tcgen05"),
        (32, 64, 8, 4, "needs \\(64, 128\\) rows"),
        (64, 8, 8, 4, "has no swizzled UMMA layout"),
        (64, 128, 8, 7, "cute_mma_dag_stages 7 must be one of"),
        (128, 128, 8, 8, "need .* B of shared memory"),
        (64, 64, 8, 4, None),
        (64, 16, 4, 2, None),
        (128, 128, 8, 4, None),
    ),
)
def test_config_reasons(
    bm: int, bn: int, warps: int, stages: int, reason: str | None
) -> None:
    result = dag.mma_dag_config_reason(
        _facts(), _sizes(bm, bn), warps, stages, "tcgen05"
    )
    if reason is None:
        assert result is None
    else:
        assert result is not None and re.search(reason, result), result


def test_geometry_helpers() -> None:
    assert tc.tcgen05_atom_bytes(32) == 32 and tc.tcgen05_atom_bytes(64) == 64
    assert tc.tcgen05_atom_bytes(128) == 128 and tc.tcgen05_atom_bytes(256) == 128
    assert tc.tcgen05_atom_bytes(48) is None and tc.tcgen05_atom_bytes(16) is None
    assert [tc.tcgen05_tmem_cols(n) for n in (8, 32, 64, 96, 128, 256)] == [
        32,
        32,
        64,
        128,
        128,
        256,
    ]
    facts = _facts()
    assert facts.tcgen05 is not None
    # 64 x 64 bf16 operand tiles: two 8 KB slabs per stage.
    assert tc.tcgen05_stage_bytes(facts.tcgen05, [1, 64, 64, 64]) == 16384
    assert tc.tcgen05_stage_bytes(facts.tcgen05, [1, 128, 128, 64]) == 32768
    assert tc.tcgen05_smem_bytes(facts, [1, 64, 64, 64], 8, True) == (
        8 * 16384 + 2 * 64 * 64 * 2 + 8 * 11 + 16
    )
    assert tc.tcgen05_slot_columns(facts, [1, 64, 8, 64]) == 32
    assert tc.tcgen05_slot_columns(facts, [1, 64, 96, 64]) == 96


def test_repair_keeps_the_mode_where_a_geometry_exists() -> None:
    facts = _facts()
    assert dag.mma_dag_repair(facts, _sizes(64, 64), 2, 4, "tcgen05") == (
        _resolved(64, 64),
        4,
        4,
    )
    # Too much shared memory: fewer stages before other tiles.
    assert dag.mma_dag_repair(facts, _sizes(128, 128), 8, 8, "tcgen05") == (
        _resolved(128, 128),
        8,
        4,
    )
    repaired = dag.mma_dag_repair(facts, _sizes(32, 24), 8, 4, "tcgen05")
    assert repaired is not None and repaired[0][1] in (64, 128)
    assert repaired[0][2] in (64, 128, 32, 16)


def test_normalize_repairs_or_refuses_tcgen05_configs() -> None:
    bound = _bind_fwd()
    spec = bound.config_spec
    with bound.env:
        with pytest.raises(InvalidConfig, match="cute_mma_dag='tcgen05'"):
            spec.normalize(_config(64, 64, warps=2).config)
        repaired = _config(64, 64, warps=2).config
        spec.normalize(repaired, _fix_invalid=True)
        assert repaired["cute_mma_dag"] == "tcgen05"
        assert repaired["cute_mma_dag_warps"] == 4
        repaired = _config(32, 64, warps=8).config
        spec.normalize(repaired, _fix_invalid=True)
        assert repaired["cute_mma_dag"] == "tcgen05"
        assert repaired["block_sizes"][0] == 64
        assert repaired["cute_mma_dag_warps"] == 8


def test_viability_prescreen_follows_the_mode() -> None:
    bound = _bind_fwd()
    backend = bound.env.backend
    spec = bound.config_spec
    assert backend.autotune_config_is_viable(spec, _config(64, 64, warps=8))
    assert not backend.autotune_config_is_viable(spec, _config(64, 64, warps=2))
    assert not backend.autotune_config_is_viable(spec, _config(32, 64, warps=8))
    assert backend.autotune_config_is_viable(
        spec,
        helion.Config(
            block_sizes=[32, 64],
            cute_mma_dag="warp",
            cute_mma_dag_warps=2,
            cute_mma_dag_stages=2,
        ),
    )


def test_an_illegal_tcgen05_config_fails_closed() -> None:
    bound = _bind_fwd()
    with pytest.raises((InvalidConfig, BackendUnsupported), match="cute_mma_dag"):
        bound.to_code(_config(64, 64, warps=2))
    # The planner refuses what normalize cannot see: a config that names
    # the mode on a program whose facts do not admit it.
    bound.config_spec.target_device_capability = (9, 0)
    with pytest.raises(BackendUnsupported, match="cute_mma_dag"):
        bound.to_code(_config(64, 64))


# --------------------------------------------------------------------------
# Generated code
# --------------------------------------------------------------------------


def test_renders_the_vanilla_forward_recurrence() -> None:
    source = _source(_bind_fwd(), _config(64, 64, warps=8, stages=4))
    body = _kernel_body(source)
    assert "block=(256, 1, 1)" in source
    assert WARP_RT + ".mma_bf16(" not in body and "ldmatrix" not in body
    # Two tensor maps (k, v) built by the host wrapper: [BH, N, C, D] in TMA
    # order (D, C, N, BH), a box of one 64-row swizzle atom per chunk row.
    assert "'kind': 'helion_mma_dag_tma'" in source and "'tma_count': 2" in source
    assert (
        "'tma0_dims': [3, 2, 1, 0], 'tma0_box': [64, 64, 1, 1], 'tma0_swizzle': 's128b'"
        in source
    )
    assert f"md_tma0: cutlass.GridConstant[{RT}.cuda.TensorMap]" in body
    # TMEM: 64 columns, allocated and freed by warp 0 once.
    assert body.count(f"{P}.tcgen05_alloc(md_tslot, 64,") == 1
    assert body.count(f"{P}.tcgen05_dealloc(md_acc0, 64,") == 1
    assert body.count(f"{P}.mbarrier_init(md_full") == 4
    assert body.count(f"{P}.mbarrier_init(md_accb") == 1
    # No epilogue: no epilogue tiles, no exchange, no epilogue commit.
    assert f"{P}.tcgen05_commit(md_eacc" not in body
    assert f"{RT}.mbar_wait(md_efull" not in body and "md_row64" in body
    # Four K steps per iteration, one commit; the prologue issues three
    # iterations' tiles and the loop refills one per iteration.
    loop = _loop(body)
    assert loop.count(f"{P}.tcgen05_mma(") == 4
    assert loop.count(f"{P}.tcgen05_commit(md_accb") == 1
    assert body.count(f"{P}.cp_async_bulk_tensor_shared_cta_global(") == 8
    assert body.count(f"{P}.mbarrier_arrive_expect_tx(") == 4
    assert "if md_i + 3 < 8:" in loop
    assert "md_stage = md_i % 4" in loop and "md_round = md_i // 4 & 1" in loop
    # MN-major descriptors: slab stride 8 KB, eight rows of 128 B per atom.
    assert "leading_byte_offset=8192, stride_byte_offset=1024" in loop
    assert f"layout={P}.Tcgen05SmemSwizzle.SWIZZLE_128B" in loop
    assert (
        "advance_start_address(2048)" in loop and "advance_start_address(6144)" in loop
    )
    assert "a_major=1, b_major=1" in body
    # Ordering: acc wait, tcgen05.ld, staging write, fence, barrier, full
    # wait, MMAs, commit, refill, then the store warps drain the staging tile.
    order = [
        f"{RT}.mbar_wait(md_accb, md_parity ^ 1)",
        f"{P}.tcgen05_ld(",
        f"{P}.tcgen05_wait(kind={P}.Tcgen05Wait.LOAD)",
        f"{WARP_RT}.sts_v4_b32(",
        f"{P}.tcgen05_fence({P}.Tcgen05Fence.BEFORE_THREAD_SYNC)",
        "cute.arch.barrier()",
        f"{RT}.mbar_wait(md_full + md_stage, md_round)",
        f"{P}.tcgen05_mma(",
        f"{P}.tcgen05_commit(md_accb",
        f"{P}.cp_async_bulk_tensor_shared_cta_global(",
        "if md_warp >= 4:",
        f"{WARP_RT}.lds_v4_b32(",
        f"{WARP_RT}.stg_v4_b32(",
    ]
    positions = [loop.index(item) for item in order]
    assert positions == sorted(positions), positions
    # No rescale round trip without a decay.
    assert f"{P}.tcgen05_st(" not in loop
    # The staging tile follows the ring (4 x 16 KB) and is double buffered.
    assert "md_s0 + 65536 + md_parity * 8192" in loop
    assert "alloc_smem(cutlass.Uint8, 81920, alignment=1024)" in body


def test_four_warps_drain_the_staging_tile_themselves() -> None:
    body = _kernel_body(_source(_bind_fwd(), _config(64, 64, warps=4, stages=2)))
    loop = _loop(body)
    assert "if md_warp >= 4:" not in loop
    assert loop.count(f"{WARP_RT}.stg_v4_b32(") == 4
    assert "if md_i + 1 < 8:" in loop
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=1024)" in body


def test_renders_the_seeded_row_rescaled_forward() -> None:
    body = _kernel_body(
        _source(_bind_fwd("diag", has_h0=True), _config(64, 64, warps=8, stages=2))
    )
    seed, loop = body.split("for md_i in cutlass.range(", 1)
    # The seed tile is read in pairs and stored to TMEM once.
    assert f"{WARP_RT}.ldg_b32(" in seed and seed.count(f"{P}.tcgen05_st(") == 1
    # The row decay is one exp per thread per iteration, applied to every
    # column and written back before the MMA.
    assert loop.count(f"{P}.tcgen05_st(") == 1
    assert loop.count("cute.math.exp") == 1
    st = loop.index(f"{P}.tcgen05_st(")
    assert loop.index(f"{WARP_RT}.sts_v4_b32(") < st < loop.index("cute.arch.barrier()")
    assert loop.index(f"{P}.tcgen05_wait(kind={P}.Tcgen05Wait.STORE)") < loop.index(
        "cute.arch.barrier()"
    )


def test_renders_the_scalar_decay_forward_with_an_in_place_pass() -> None:
    body = _kernel_body(
        _source(_bind_fwd("scalar"), _config(64, 64, warps=8, stages=4))
    )
    prologue, loop = body.split("for md_i in cutlass.range(", 1)
    # Both tiles arrive by TMA; the v tile is rewritten in place (exp2 of
    # the K-vector decay, one load and exp2 per packet row) by every thread
    # after the slot's full barrier and before this iteration's acc wait
    # (its previous reader, MMA(i - 4), completed one iteration ago), then
    # published to the async proxy.
    assert "'tma_count': 2" in body
    assert f"{WARP_RT}.lds_v4_b32(" not in prologue
    wait = loop.index(f"{RT}.mbar_wait(md_full + md_stage, md_round)")
    lds = loop.index(f"{WARP_RT}.lds_v4_b32(")
    fence = loop.index("cute.arch.fence_proxy(kind='async.shared', space='cta')")
    acc = loop.index(f"{RT}.mbar_wait(md_accb, md_parity ^ 1)")
    assert acc < wait < lds < fence < loop.index("cute.arch.barrier()")
    assert loop[wait:fence].count("exp2(") == 2
    # The decay scalar and the two K-vector rows of this thread come from
    # registers loaded one iteration ahead (after the MMA issue, guarded by
    # the trip count); the pass itself issues no global loads.
    assert loop[wait:fence].count(f"{WARP_RT}.ldg_f32(") == 0
    assert prologue.count("= cutlass.Float32(0.0)") >= 3
    refill = loop.index("md_ax0 = ")
    assert loop.index(f"{P}.tcgen05_commit(") < refill
    assert "if md_i + 1 < 8:" in loop[:refill]
    assert loop.count(f"{WARP_RT}.ldg_f32(") == 3
    # 512 packets over 256 threads, plus the staging packets.
    assert loop.count(f"{WARP_RT}.sts_v4_b32(") == 2 + 8
    # The leaves stay fp32 and the pack rounds once: no per-element casts
    # (the per-row decay keeps its own rounding, once per packet).
    assert loop[wait:fence].count("cutlass.BFloat16(") <= 2
    # The scalar decay rescales the carry.
    assert loop.count(f"{P}.tcgen05_st(") == 1
    # Warp 0 does not wait the full barrier twice.
    assert loop.count(f"{RT}.mbar_wait(md_full") == 1
    # The passes do not change the vanilla program.
    vanilla = _loop(
        _kernel_body(_source(_bind_fwd(), _config(64, 64, warps=4, stages=2)))
    )
    assert (
        f"{P}.prefetch_l1(" not in vanilla
        and f"{WARP_RT}.lds_v4_b32("
        not in vanilla[: vanilla.index("cute.arch.barrier()")]
    )


def test_renders_the_reversed_scaled_backward_with_fp32_stores() -> None:
    body = _kernel_body(_source(_bind_bwd(), _config(64, 64, warps=8, stages=4)))
    loop = _loop(body)
    # Reversed index in the TMA coordinates (the refill runs three
    # iterations ahead) and the store base.
    assert "cutlass.Int32(7 + -1 * (md_i + 3))" in loop
    assert "cutlass.Int64(7 + -1 * md_i)" in loop
    # fp32 snapshots: 16 packets per row into a 16 KB staging buffer.
    assert loop.count(f"{RT}.sts_v4_f32(") == 16
    assert "md_s0 + 65536 + md_parity * 16384" in loop
    assert loop.count(f"{WARP_RT}.stg_v4_b32(") == 8
    # q * scale is an in-place rewrite of the landed tile (two packets per
    # thread) with the rounding folded into the pack.
    assert "* scale" in loop and loop.count(f"{WARP_RT}.sts_v4_b32(") == 2
    assert "cutlass.BFloat16(" not in loop
    # The scaled operand is read transposed: the K-vector decay of the
    # scalar mode indexes the packet row, not its columns.
    scalar = _loop(
        _kernel_body(_source(_bind_bwd("scalar"), _config(64, 64, warps=4, stages=4)))
    )
    pass_body = scalar[: scalar.index("cute.arch.fence_proxy(")]
    # Four packets per thread over 128 threads: one exp2 each of a row value
    # loaded one iteration ahead (four rows plus the decay scalar).
    assert pass_body.count("exp2(") == 4 and "md_pch * 8" not in pass_body
    assert pass_body.count(f"{WARP_RT}.ldg_f32(") == 0
    assert scalar.count("md_ax4 = ") == 1 and "md_ax5" not in scalar


def test_narrow_tiles_use_the_narrow_swizzles() -> None:
    body = _kernel_body(_source(_bind_fwd(), _config(64, 32, warps=4, stages=2)))
    assert "'tma1_box': [32, 64, 1, 1], 'tma1_swizzle': 's64b'" in body
    assert f"layout={P}.Tcgen05SmemSwizzle.SWIZZLE_64B" in body
    assert "n_dim=32, m_dim=64" in body
    assert f"{P}.tcgen05_alloc(md_tslot, 32," in body
    wide = _kernel_body(
        _source(_bind_fwd(d=128, dv=128), _config(128, 128, warps=8, stages=2))
    )
    # 128-wide tiles are two swizzle atoms side by side: two boxes per
    # operand per stage, the MN-major slab stride of one box.
    assert wide.count(f"{P}.cp_async_bulk_tensor_shared_cta_global(") == 8
    assert "cutlass.Int32(md_a1 + 64)" in wide
    assert "leading_byte_offset=8192, stride_byte_offset=1024" in wide
    assert f"{P}.tcgen05_alloc(md_tslot, 128," in wide
    assert "md_rowok" not in wide and "md_row128 = md_warp % 4 * 32 + md_lane" in wide


def test_renders_the_k_loop_epilogue_through_an_exchange_tile() -> None:
    body = _kernel_body(_source(_bind_fwd_o(), _config(1, 128, 64, warps=4, stages=4)))
    prologue, rest = body.split("for md_i in cutlass.range(", 1)
    loop, epilogue = rest.split(f"{RT}.mbar_wait(md_accb, 2 - 0 - 1 & 1)", 1)
    # Three loop tiles (q shared by both dots, h, k) and the epilogue v tile
    # arrive by TMA; the epilogue tile has its own barrier.
    assert "'tma_count': 4" in body
    assert f"{P}.mbarrier_arrive_expect_tx(md_efull, 16384)" in prologue
    assert f"{P}.mbarrier_init(md_efull, 1)" in prologue
    # Two fragments: o_cross [64, 128] at column 0, attn [64, 64] at 128.
    assert f"{P}.tcgen05_alloc(md_tslot, 256," in prologue
    assert f"md_acc1 = cutlass.inttoptr(md_tmem + 128, {RT}.TMEM_SPACE" in prologue
    # Loop: q K-major A for both dots, h MN-major B, k^T K-major B.
    assert "n_dim=128, m_dim=64, a_major=0, b_major=1" in prologue
    assert "n_dim=64, m_dim=64, a_major=0, b_major=0" in prologue
    assert loop.count(f"{P}.tcgen05_mma(") == 8
    assert loop.count(f"{P}.tcgen05_commit(md_accb") == 1
    # No epilogue-warp work in the loop: no TMEM round trip, no snapshot.
    assert loop.count(f"{P}.tcgen05_ld(") == 0 and "if md_epi" not in loop
    assert loop.count("cute.arch.barrier()") == 1
    # Epilogue: the masked attention row (lanes 0-15 of each quadrant own a
    # row of the 64-row fragment) is packed into the UMMA exchange tile
    # (eight 16-byte packets per row), published to the async proxy and
    # read as the K-major A operand of the tensor-core dot accumulating onto
    # o_cross; one commit, then the store through the staging tile.
    order = [
        f"{RT}.mbar_wait(md_efull, 0)",
        "if md_epi:",
        f"{P}.tcgen05_ld(",
        "if md_rowok:",
        f"{WARP_RT}.sts_v4_b32(md_s0 + 131072 + md_row64 * 128",
        "cute.arch.fence_proxy(kind='async.shared', space='cta')",
        f"{P}.tcgen05_fence({P}.Tcgen05Fence.BEFORE_THREAD_SYNC)",
        "cute.arch.barrier()",
        "if md_warp == 0:",
        f"{P}.Tcgen05SmemDesc.build(md_s0 + 131072, leading_byte_offset=16",
        f"{P}.Tcgen05SmemDesc.build(md_s0 + 139264, leading_byte_offset=8192",
        f"{P}.tcgen05_mma(",
        f"{P}.tcgen05_commit(md_eacc",
        f"{RT}.mbar_wait(md_eacc, 0)",
        f"{P}.tcgen05_ld(",
        "md_srow = md_s0 + 0 + md_row64 * 256",
        f"{WARP_RT}.stg_v4_b32(",
        f"{P}.tcgen05_dealloc(md_acc0, 256,",
    ]
    _assert_in_order(epilogue, order)
    exchange = epilogue[: epilogue.index("cute.arch.fence_proxy(")]
    assert exchange.count(f"{WARP_RT}.sts_v4_b32(") == 8
    assert epilogue.count(f"{P}.tcgen05_mma(") == 4
    assert "md_idesc2, True)" in epilogue
    assert epilogue.count(f"{P}.tcgen05_commit(md_eacc") == 1
    # The store evaluates ``(o * scale)`` per element with the rounding in
    # the pack: 1024 packets over 128 threads.
    assert epilogue.count(f"{WARP_RT}.stg_v4_b32(") == 8
    assert "cutlass.BFloat16(" not in epilogue
    assert f"{P}.tcgen05_st(" not in epilogue


def test_renders_the_transposed_and_shared_exchange_tiles() -> None:
    # dv: attn^T is the MN-major A operand of the epilogue dot read from the
    # exchange tile written at the fragment's own coordinates; dh arrives as
    # a raw fp32 tile converted in place by every thread.
    body = _kernel_body(_source(_bind_dv(), _config(1, 128, 64, warps=4, stages=2)))
    prologue, rest = body.split("for md_i in cutlass.range(", 1)
    loop, epilogue = rest.split(f"{RT}.mbar_wait(md_accb, 2 - 0 - 1 & 1)", 1)
    assert "n_dim=128, m_dim=64, a_major=1, b_major=1" in prologue
    assert loop.index(f"{RT}.mbar_wait(md_full + md_stage, md_round)") < loop.index(
        f"{RT}.lds_v4_f32("
    )
    assert loop.count(f"{WARP_RT}.sts_v2_b32(") == 16
    assert loop.count(f"{RT}.mbar_wait(md_full") == 1
    a_desc = epilogue.index(
        f"{P}.Tcgen05SmemDesc.build(md_s0 + 147456, leading_byte_offset=8192"
    )
    assert (
        epilogue.index(f"{WARP_RT}.sts_v4_b32(md_s0 + 147456 + md_row64 * 128") < a_desc
    )
    # dqk: one dA exchange tile read K-major (dq) and MN-major (dk) by the
    # two epilogue dots, issued back to back under one commit; the two
    # stores drain the one staging tile in turn.
    body = _kernel_body(
        _source(_bind_dqk(gated=True), _config(1, 64, 64, warps=4, stages=2))
    )
    prologue, rest = body.split("for md_i in cutlass.range(", 1)
    loop, epilogue = rest.split(f"{RT}.mbar_wait(md_accb, 2 - 0 - 1 & 1)", 1)
    assert f"{P}.tcgen05_alloc(md_tslot, 256," in prologue
    assert "md_acc2 = cutlass.inttoptr(md_tmem + 128," in prologue
    assert loop.count(f"{P}.tcgen05_mma(") == 12
    descs = [
        line for line in epilogue.splitlines() if f"{P}.Tcgen05SmemDesc.build(" in line
    ]
    assert len(descs) == 4
    bases = [desc.split("build(")[1].split(",")[0] for desc in descs]
    assert bases[0] == bases[2] and len(set(bases)) == 3
    assert "leading_byte_offset=16" in descs[0]
    assert "leading_byte_offset=8192" in descs[2]
    # The exchange tile is written once (eight packets per row) before the
    # first dot; the two stores stage eight packets each.
    first_mma = epilogue.index(f"{P}.tcgen05_mma(")
    assert epilogue[:first_mma].count(f"{WARP_RT}.sts_v4_b32(") == 8
    second_dot = epilogue.index("md_idesc4")
    assert epilogue[first_mma:second_dot].count(f"{WARP_RT}.sts_v4_b32(") == 0
    assert epilogue.count(f"{WARP_RT}.sts_v4_b32(") == 8 + 2 * 8
    assert epilogue.count(f"{P}.tcgen05_mma(") == 8
    assert epilogue.count(f"{P}.tcgen05_commit(md_eacc") == 1
    # The gated accumulators are rescaled in TMEM before the issue (one
    # tcgen05.st per 64 columns each) and the mask / decay are evaluated per
    # exchange element with exp through the backend's lowering.
    assert epilogue.count(f"{P}.tcgen05_st(") == 2
    assert epilogue.index(f"{P}.tcgen05_st(") < epilogue.index(f"{P}.tcgen05_mma(")
    assert epilogue.count("= md_s0 + 0 + md_row64 * 128") == 2
    assert epilogue.count(f"{WARP_RT}.stg_v4_b32(") == 2 * 4


def test_direct_stores_skip_the_staging_tile() -> None:
    body = _kernel_body(
        _source(
            _bind_fwd(), _config(64, 64, warps=8, stages=2, cute_mma_dag_store="direct")
        )
    )
    loop = _loop(body)
    assert f"{WARP_RT}.stg_v4_b32(" not in loop and f"{WARP_RT}.lds_v4_b32(" not in loop
    assert loop.count(f"{WARP_RT}.stg_b32(") == 32
    assert "alloc_smem(cutlass.Uint8, 32768, alignment=1024)" in body


def test_unaligned_sources_take_the_synchronous_pass() -> None:
    bf16 = torch.bfloat16
    k_wide = torch.empty((2, 8, 64, 80), dtype=bf16, device=CPU_DEVICE)
    k = k_wide[..., 4:68]
    v = torch.empty((2, 8, 64, 64), dtype=bf16, device=CPU_DEVICE)
    from test.test_cute_mma_dag import _bind

    bound = _bind(
        "chunk_fwd_h_diag_fused", (k, v, None, None, None, False, False, False, False)
    )
    spec = bound.config_spec
    assert spec.cute_mma_dag_facts is not None
    assert spec.cute_mma_dag_facts.tcgen05 is not None
    body = _kernel_body(_source(bound, _config(64, 64, warps=8, stages=2)))
    assert "'tma_count': 1" in body
    loop = _loop(body)
    # 2048 pairs over 256 threads, before the acc wait and the barrier.
    assert f"{WARP_RT}.ldg_b32(" in loop and loop.count(f"{WARP_RT}.sts_b32(") == 8
    assert loop.index(f"{WARP_RT}.sts_b32(") < loop.index(
        "cute.arch.fence_proxy(kind='async.shared', space='cta')"
    )
    assert loop.index("cute.arch.fence_proxy(") < loop.index("cute.arch.barrier()")
    # Only warp 0 waits the (one) TMA operand's full barrier.
    assert loop.count(f"{RT}.mbar_wait(md_full") == 1
