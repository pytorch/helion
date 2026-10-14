"""Codegen, admission and search-surface pins for the ``cute_mma_dag``
tcgen05 compute mode (the TMEM-resident serial state recurrence).

CPU-only (fake sm100 target) like ``test_cute_mma_dag.py``, whose binders
and fixture this file reuses.  Runtime coverage lives in
``test_cute_mma_dag_tcgen05_gpu.py``.
"""

from __future__ import annotations

import collections
import dataclasses
import os
import re
from typing import TYPE_CHECKING

import pytest
import torch

from test.test_cute_mma_dag import _bind
from test.test_cute_mma_dag import _bind_bwd
from test.test_cute_mma_dag import _bind_dk
from test.test_cute_mma_dag import _bind_dqk
from test.test_cute_mma_dag import _bind_dqkw
from test.test_cute_mma_dag import _bind_dstate
from test.test_cute_mma_dag import _bind_dv
from test.test_cute_mma_dag import _bind_fn
from test.test_cute_mma_dag import _bind_fwd
from test.test_cute_mma_dag import _bind_fwd_h_delta
from test.test_cute_mma_dag import _bind_fwd_o
from test.test_cute_mma_dag import _bind_wy
from test.test_cute_mma_dag import _bind_wy_dl
from test.test_cute_mma_dag import _gen_epi
from test.test_cute_mma_dag import _gen_epi_stored
from test.test_cute_mma_dag import _kernel_body
from test.test_cute_mma_dag import _program
from test.test_cute_mma_dag import _root_rescaled_carry
from test.test_cute_mma_dag import _single_ring
from test.test_cute_mma_dag import _single_ring_bare
from test.test_cute_mma_dag import _single_ring_noacc
from test.test_cute_mma_dag import _source
from test.test_cute_mma_dag import _tensors
from test.test_cute_mma_dag import _tensors3
from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute import cute_mma_dag_tcgen05 as tc
from helion._compiler.cute import cute_mma_dag_tcgen05_regions as regions
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from helion.runtime.kernel import BoundKernel

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


def test_chained_loop_accumulators_split_the_iteration_into_batches() -> None:
    # ``x = dot(acc=h); x_all[...] = x; h = dot(acc=h * 0.5)``: x copies the
    # carry into its own slot before the first batch, its store and the
    # rescale of the carry wait for that batch's commit, the second batch
    # commits to acc_ready.
    from test.test_cute_mma_dag import _loop_chain
    from test.test_cute_mma_dag import _two_dots_same_carry

    t = _tensors()
    bound = _bind_fn(_two_dots_same_carry, (t["k"], t["v"]))
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    loop = _loop(body)
    assert loop.count(f"{P}.tcgen05_commit(md_stepb") == 1
    assert loop.count(f"{P}.tcgen05_commit(md_accb") == 1
    _assert_in_order(
        loop,
        [
            f"{P}.tcgen05_ld(",
            f"{P}.tcgen05_st(",
            "cute.arch.barrier()",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_stepb",
            f"{RT}.mbar_wait(md_stepb, md_i * 1 + 0 & 1)",
            f"{P}.tcgen05_ld(",
            f"{WARP_RT}.sts_v4_b32(md_srow",
            f"{P}.tcgen05_st(",
            "cute.arch.barrier()",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_accb",
            f"{WARP_RT}.stg_v4_b32(",
        ],
    )
    # Without the store (``h = dot(acc=dot(acc=h) * 0.5)``, x in the carry's
    # slot) the rescale still sits between the two batches: the simple loop
    # would rescale the carry before both dots.
    bound = _bind_fn(_loop_chain, (t["k"], t["v"]))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.general_loop
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    loop = _loop(body)
    assert f"{P}.mbarrier_init(md_stepb, 1)" in body
    assert loop.count(f"{P}.tcgen05_commit(md_stepb") == 1
    assert loop.count(f"{P}.tcgen05_commit(md_accb") == 1
    assert loop.count(f"{P}.tcgen05_st(") == 1 and "md_tmem + 0" in loop
    _assert_in_order(
        loop,
        [
            f"{RT}.mbar_wait(md_accb, md_parity ^ 1)",
            "cute.arch.barrier()",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_stepb",
            f"{RT}.mbar_wait(md_stepb, md_i * 1 + 0 & 1)",
            f"{P}.tcgen05_ld(",
            f"{P}.tcgen05_st(",
            "cute.arch.barrier()",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_accb",
        ],
    )
    assert loop.count(f"{P}.tcgen05_mma(") == 4 + 4


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
    assert facts.tcgen05 is not None and not facts.tcgen05.general_loop
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
    # Ordering: the epilogue warps' branch -- acc wait, tcgen05.ld, fence,
    # barrier, full wait, MMAs, commit, then the snapshot pack into the
    # staging tile (under the MMA) and the hand-off barrier; the store
    # warps' branch -- barrier, the ring refill (warp 4, off the MMA warp),
    # hand-off barrier, drain.  Both branches run two barriers.
    order = [
        f"{RT}.mbar_wait(md_accb, md_parity ^ 1)",
        "if md_epi:",
        f"{P}.tcgen05_ld(",
        f"{P}.tcgen05_wait(kind={P}.Tcgen05Wait.LOAD)",
        f"{P}.tcgen05_fence({P}.Tcgen05Fence.BEFORE_THREAD_SYNC)",
        "cute.arch.barrier()",
        f"{RT}.mbar_wait(md_full + md_stage, md_round)",
        f"{P}.tcgen05_mma(",
        f"{P}.tcgen05_commit(md_accb",
        f"{WARP_RT}.sts_v4_b32(",
        "cute.arch.barrier()",
        "else:",
        "cute.arch.barrier()",
        "if md_warp == 4:",
        f"{P}.cp_async_bulk_tensor_shared_cta_global(",
        "cute.arch.barrier()",
        f"{WARP_RT}.lds_v4_b32(",
        f"{WARP_RT}.stg_v4_b32(",
    ]
    _assert_in_order(loop, order)
    iteration = loop.split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)[0]
    assert iteration.count("cute.arch.barrier()") == 4
    assert "if md_warp >= 4:" not in loop
    # The pack is not ahead of the barrier that gates the MMA issue.
    assert loop.index(f"{WARP_RT}.sts_v4_b32(") > loop.index(f"{P}.tcgen05_mma(")
    # The MMA warp issues no refill.
    warp0 = loop[loop.index("if md_warp == 0:") : loop.index("else:")]
    assert f"{P}.cp_async_bulk_tensor_shared_cta_global(" not in warp0
    # No rescale round trip without a decay.
    assert f"{P}.tcgen05_st(" not in loop
    # The staging tile follows the ring (4 x 16 KB) and is double buffered.
    assert "md_s0 + 65536 + md_parity * 8192" in loop
    assert "alloc_smem(cutlass.Uint8, 81920, alignment=1024)" in body


def test_four_warps_store_the_snapshot_by_tma() -> None:
    # Without store warps the epilogue warps drained the staging tile
    # themselves, on the chain (59 of 259 us on fwd_h_delta at s2): thread 0
    # now stores it by TMA (the staging swizzle is TMA's).  Every warp is
    # an epilogue warp, so the loop body needs no ``md_epi`` branch: the
    # TMEM load alone precedes the barrier that gates the MMA issue; the
    # pack, the staging writes and their proxy fence run under the MMA,
    # and a second barrier hands the tile to the store thread, which waits
    # for the previous iteration's group before the next barrier (the
    # stage-in reusing that buffer follows it).  The ring refill stays with
    # the MMA warp after its commit, under the MMA (130 -> 122 us on the s2
    # vanilla forward, 177 -> 162 us on the gated one).
    source = _source(_bind_fwd(), _config(64, 64, warps=4, stages=2))
    body = _kernel_body(source)
    assert "'tma_count': 3" in source
    assert (
        "'tma2_dims': [3, 2, 1, 0], 'tma2_box': [64, 64, 1, 1], 'tma2_swizzle': 's128b'"
        in source
    )
    loop = _loop(body)
    assert "if md_warp >= 4:" not in loop and "if md_epi:" not in loop
    assert f"{WARP_RT}.stg_v4_b32(" not in loop and f"{WARP_RT}.lds_v4_b32(" not in loop
    _assert_in_order(
        loop,
        [
            f"{RT}.mbar_wait(md_accb, md_parity ^ 1)",
            f"{P}.tcgen05_ld(",
            f"{P}.tcgen05_fence({P}.Tcgen05Fence.BEFORE_THREAD_SYNC)",
            "cute.arch.barrier()",
            "if md_warp == 0:",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_accb",
            "if md_i + 1 < 8:",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(",
            f"{WARP_RT}.sts_v4_b32(",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            "cute.arch.barrier()",
            "if md_tidx == 0:",
            (
                f"{P}.cp_async_bulk_tensor_global_shared_cta(md_tma2.get_ptr(), "
                "md_smem + (32768 + md_parity * 8192), ["
            ),
            f"{P}.cp_async_bulk_commit_group()",
            f"{P}.cp_async_bulk_wait_group(1, read=True)",
        ],
    )
    iteration = loop.split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)[0]
    assert iteration.count("cute.arch.barrier()") == 2
    assert "if md_warp == 1:" not in iteration
    assert loop.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1
    # The issue with its group wait, and the final drain after the loop.
    assert loop.count("if md_tidx == 0:") == 2
    assert f"{P}.cp_async_bulk_wait_group(0)" in loop
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=1024)" in body
    assert "md_s0 + 32768 + md_parity * 8192" in loop


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
    assert st < loop.index("cute.arch.barrier()") < loop.index(f"{P}.tcgen05_mma(")
    assert loop.index(f"{P}.tcgen05_wait(kind={P}.Tcgen05Wait.STORE)") < loop.index(
        "cute.arch.barrier()"
    )
    # The snapshot is packed from the pre-rescale registers after the MMA
    # issue (the rescale wrote its own arrays to tensor memory).
    assert loop.index(f"{WARP_RT}.sts_v4_b32(") > loop.index(f"{P}.tcgen05_mma(")
    assert loop.index(f"{WARP_RT}.sts_v4_b32(md_srow") > st
    assert (
        "md_rs"
        not in loop[loop.index(f"{WARP_RT}.sts_v4_b32(") :].split(
            "cute.arch.barrier()"
        )[0]
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
    # Once per branch of the split iteration (the epilogue warps' after the
    # MMA issue, the store warps' after the barrier): three per thread.
    epi, store = loop.split("else:", 1)
    assert epi.count(f"{WARP_RT}.ldg_f32(") == 3
    assert store.count(f"{WARP_RT}.ldg_f32(") == 3
    # 512 packets over 256 threads, plus the staging packets.
    assert loop.count(f"{WARP_RT}.sts_v4_b32(") == 2 + 4
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
    assert loop.count(f"{RT}.sts_v4_f32(") == 8
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
        "md_srow = md_s0 + 0 + md_row64 * 128",
        f"{WARP_RT}.stg_v4_b32(",
        f"{P}.tcgen05_dealloc(md_acc0, 256,",
    ]
    _assert_in_order(epilogue, order)
    # Every lane holds half a row (16x32bx2): four packets per thread.
    exchange = epilogue[: epilogue.index("cute.arch.fence_proxy(")]
    assert exchange.count(f"{WARP_RT}.sts_v4_b32(") == 4
    assert epilogue.count(f"{P}.tcgen05_mma(") == 4
    assert "md_idesc2, True)" in epilogue
    assert epilogue.count(f"{P}.tcgen05_commit(md_eacc") == 1
    # The store evaluates ``(o * scale)`` per element with the rounding in
    # the pack: 1024 packets over 128 threads.
    assert epilogue.count(f"{WARP_RT}.stg_v4_b32(") == 8
    assert "cutlass.BFloat16(" not in epilogue
    assert f"{P}.tcgen05_st(" not in epilogue


def test_gated_epilogue_side_loads_go_out_under_the_last_mmas() -> None:
    # fwd_o with decay: the exchange's ``exp2((gc_i - gc_j) * RCP_LN2)`` and
    # the o_cross rescale's ``exp2(gc_i * RCP_LN2)`` read the chunk's decay
    # row per element.  The thread's row value and its 32 half-row column
    # values take one register each, defined ahead of the loop and loaded
    # in the last iteration between the MMA issue and the accumulator wait
    # (the drain after a one-trip loop is immediate: loads issued there
    # stall the epilogue's first subtraction for a memory round trip); the
    # epilogue issues no global load of its own and the exchange and the
    # rescale share the row register.
    body = _kernel_body(
        _source(_bind_fwd_o(True), _config(1, 128, 128, warps=4, stages=1))
    )
    prologue, rest = body.split("for md_i in cutlass.range(0, 1):", 1)
    loop, epilogue = rest.split(f"{RT}.mbar_wait(md_accb, 1 - 0 - 1 & 1)", 1)
    assert [line.strip() for line in prologue.splitlines() if "md_ex" in line] == [
        f"md_ex{i} = cutlass.Float32(0.0)" for i in range(33)
    ]
    issue, wait = loop.split(f"{RT}.mbar_wait(md_accb, md_parity)", 1)
    assert f"{P}.tcgen05_commit(md_accb" in issue and "md_ex" not in wait
    guarded = issue.split("if md_i + 1 == 1:", 1)[1]
    loads = [line.strip() for line in guarded.splitlines() if line.strip()]
    assert len(loads) == 33
    assert re.fullmatch(
        rf"md_ex0 = {WARP_RT}\.ldg_f32\(md_base_\d+ \+ cutlass\.Int64\(md_row64\) \* 4\)",
        loads[0],
    )
    assert re.fullmatch(
        rf"md_ex1 = {WARP_RT}\.ldg_f32\(md_base_\d+ \+ cutlass\.Int64\(md_cb1 \+ 0\) \* 4\)",
        loads[1],
    )
    assert loads[32].startswith("md_ex32 = ") and "(md_cb1 + 31) * 4)" in loads[32]
    assert "ldg_" not in epilogue
    reads = collections.Counter(re.findall(r"= (md_ex\d+)$", epilogue, re.MULTILINE))
    assert reads["md_ex0"] == 2
    assert all(reads[f"md_ex{i}"] == 1 for i in range(1, 33))
    # Without decay the epilogue reads no vector: nothing is hoisted.
    plain = _kernel_body(
        _source(_bind_fwd_o(), _config(1, 128, 128, warps=4, stages=1))
    )
    assert "md_ex" not in plain and "ldg_" not in plain


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
    assert epilogue[:first_mma].count(f"{WARP_RT}.sts_v4_b32(") == 4
    second_dot = epilogue.index("md_idesc4")
    assert epilogue[first_mma:second_dot].count(f"{WARP_RT}.sts_v4_b32(") == 0
    assert epilogue.count(f"{WARP_RT}.sts_v4_b32(") == 4 + 2 * 4
    assert epilogue.count(f"{P}.tcgen05_mma(") == 8
    assert epilogue.count(f"{P}.tcgen05_commit(md_eacc") == 1
    # The gated accumulators are rescaled in TMEM before the issue (one
    # tcgen05.st per 64 columns each) and the mask / decay are evaluated per
    # exchange element with exp through the backend's lowering.
    assert epilogue.count(f"{P}.tcgen05_st(") == 2
    assert epilogue.index(f"{P}.tcgen05_st(") < epilogue.index(f"{P}.tcgen05_mma(")
    assert epilogue.count("= md_s0 + 0 + md_row64 * 128") == 2
    assert epilogue.count(f"{WARP_RT}.stg_v4_b32(") == 2 * 4


def test_delta_state_kernels_admit_the_mode_with_loop_exchanges() -> None:
    # fwd_h_delta: the carry feeds the first dot and the derived value the
    # second, both through exchange tiles rewritten every iteration; the
    # derived value holds one row of its input (the dot) and its own.
    facts = _bind_fwd_h_delta().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    tc_facts = facts.tcgen05
    assert [(x.slot, x.region) for x in tc_facts.exchanges] == [
        (0, "loop"),
        (1, "loop"),
    ]
    assert [t.region for t in tc_facts.tiles] == ["loop", "loop"]
    assert tc_facts.derived_cols == ((1, 1),)
    assert tc_facts.loop_stores == ((0, 2), (1, 2))
    assert tc_facts.general_loop
    # dstate: attn, the carry and dvni as loop operands; two derived values
    # (the dvni sum in the gated form, the carry update).
    facts = _bind_dstate(gated=True).config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert [x.slot for x in facts.tcgen05.exchanges] == [1, 0, 2]
    assert len(facts.tcgen05.derived_cols) == 2
    # Both admit the mode and seed it first (its snapshots staged), then
    # the warp ladder.
    for bind in (
        _bind_fwd_h_delta,
        lambda: _bind_fwd_h_delta(gated=True),
        _bind_dstate,
        lambda: _bind_dstate(gated=True),
    ):
        spec = bind().config_spec
        assert spec.cute_mma_dag is not None
        assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
        first = spec.compiler_seed_configs[0].config
        assert first["cute_mma_dag"] == "tcgen05"
        assert first["cute_mma_dag_store"] == "staged"
        assert spec.default_config().config["cute_mma_dag"] == "tcgen05"
        modes = [c.config["cute_mma_dag"] for c in spec.compiler_seed_configs]
        assert "tcgen05" in modes and "warp" in modes
    # A derived value over wide rows is refused: fwd_h_delta at bdv 128
    # holds two 128-column rows per thread.
    facts = _bind_fwd_h_delta().config_spec.cute_mma_dag_facts
    assert facts is not None
    reason = tc.tcgen05_config_reason(facts, [1, 128, 128, 64], 4, 1)
    assert reason is not None and "derived value holds" in reason


def test_renders_the_delta_loop_in_program_order() -> None:
    body = _kernel_body(_source(_bind_fwd_h_delta(), _config(32, warps=4, stages=4)))
    prologue, rest = body.split("for md_i in cutlass.range(", 1)
    loop, epilogue = rest.split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)
    # One intermediate commit per iteration (dot1 -> vnew -> dot2).
    assert f"{P}.mbarrier_init(md_stepb, 1)" in prologue
    assert loop.count(f"{P}.tcgen05_commit(md_stepb") == 1
    assert loop.count(f"{P}.tcgen05_commit(md_accb") == 1
    assert f"{RT}.mbar_wait(md_stepb, md_i * 1 + 0 & 1)" in loop
    assert loop.count("cute.arch.barrier()") == 2
    # Block 0: the carry snapshot and the carry's exchange tile (bf16 pack
    # of the 128-row carry into the fixed region, MN-major B of dot1);
    # block 1: the derived value from the dot result and the u pairs, its
    # body store, the exchange of its cast for dot2, and the TMEM write.
    _assert_in_order(
        loop,
        [
            f"{RT}.mbar_wait(md_accb, md_parity ^ 1)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(",
            "if md_epi:",
            f"{P}.tcgen05_ld(",
            f"{WARP_RT}.sts_v4_b32(md_srow",
            f"{WARP_RT}.sts_v4_b32(md_s0 + 131072 + md_row128 * 64",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            "cute.arch.barrier()",
            "if md_warp == 0:",
            f"{RT}.mbar_wait(md_full + md_stage, md_round)",
            f"{P}.Tcgen05SmemDesc.build(md_s0 + 131072, leading_byte_offset=8192",
            f"{P}.tcgen05_mma(",
            "if md_tidx == 0:",
            f"{P}.cp_async_bulk_tensor_global_shared_cta(",
            f"{P}.cp_async_bulk_commit_group()",
            f"{P}.cp_async_bulk_wait_group(2, read=True)",
            f"{P}.tcgen05_commit(md_stepb",
            f"{RT}.mbar_wait(md_stepb, md_i * 1 + 0 & 1)",
            "if md_epi:",
            f"{P}.tcgen05_ld(",
            "md_dr_0 = cutlass.Array(cutlass.Float32, 16, alignment=16)",
            f"{WARP_RT}.unpack_bf16x2(md_u",
            f"{WARP_RT}.sts_v4_b32(md_srow_1",
            f"{WARP_RT}.sts_v4_b32(md_s0 + 139264 + md_row64 * 64",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            "cute.arch.barrier()",
            f"{P}.Tcgen05SmemDesc.build(md_s0 + 139264, leading_byte_offset=4096",
            f"{P}.tcgen05_mma(",
            "if md_tidx == 0:",
            f"{P}.cp_async_bulk_tensor_global_shared_cta(",
            f"{P}.cp_async_bulk_commit_group()",
            f"{P}.cp_async_bulk_wait_group(2, read=True)",
            f"{P}.tcgen05_commit(md_accb",
        ],
    )
    # The snapshot and the v_new rows go out by TMA from their staging
    # tiles: two bulk groups per iteration, each issue waiting until at most
    # two are pending (the previous iteration's are read out before the
    # commit that gates the stage-ins reusing their buffers).
    assert loop.count(f"{WARP_RT}.stg_b32(") == 0
    assert loop.count(f"{WARP_RT}.stg_v4_b32(") == 0
    assert loop.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 2
    assert loop.count(f"{P}.cp_async_bulk_commit_group()") == 2
    assert loop.count(f"{P}.cp_async_bulk_wait_group(2, read=True)") == 2
    # dot1: w K-major A, the carry MN-major B into slot 1 from zero; dot2:
    # k^T MN-major A, vnew MN-major B accumulating into the carry.
    assert "n_dim=32, m_dim=64, a_major=0, b_major=1" in prologue
    assert "n_dim=32, m_dim=128, a_major=1, b_major=1" in prologue
    assert "md_idesc0, False)" in loop and "md_idesc1, True)" in loop
    # The derived value is read only in its block: no tensor-memory write;
    # the exchange packs the derived arrays; the u pairs are loaded at the
    # iteration top, before the acc wait.
    assert loop.count(f"{P}.tcgen05_st(") == 0 and loop.count(f"{P}.tcgen05_ld(") == 2
    assert "= md_dr_0[0]" in loop and "= md_dr_0[15]" in loop
    assert loop.index(f"{WARP_RT}.ldg_b32(") < loop.index(
        f"{RT}.mbar_wait(md_accb, md_parity ^ 1)"
    )
    assert loop.count(f"{WARP_RT}.ldg_b32(") == 8
    # dstate: four blocks (attn | dv_i + dv_i_1 | dS + dot_4 | dS - dot_4),
    # two step commits, the trailing derived value after the acc wait.
    body = _kernel_body(_source(_bind_dstate(), _config(32, warps=4, stages=2)))
    loop = body.split("for md_i in cutlass.range(", 1)[1].split(
        f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1
    )[0]
    assert loop.count(f"{P}.tcgen05_commit(md_stepb") == 2
    assert loop.count("cute.arch.barrier()") == 3
    assert f"{RT}.mbar_wait(md_stepb, md_i * 2 + 0 & 1)" in loop
    assert f"{RT}.mbar_wait(md_stepb, md_i * 2 + 1 & 1)" in loop
    assert loop.count(f"{RT}.mbar_wait(md_accb, md_parity)") == 1
    assert loop.index(f"{RT}.mbar_wait(md_accb, md_parity)") > loop.rindex(
        f"{P}.tcgen05_commit(md_accb"
    )
    # attn: q and k K-major; the transposed attn (k rows, q rows) is read
    # straight as the K-major A of dv_i with do_i MN-major; dv_i_1
    # accumulates onto it in the same batch.
    assert re.search(r"md_idesc0 = .*n_dim=64, m_dim=64, a_major=0, b_major=0", body)
    assert re.search(r"md_idesc1 = .*n_dim=32, m_dim=64, a_major=0, b_major=1", body)
    # attn K = 128, dv_i K = 64, dv_i_1 K = 128, dS K = 64, dot_4 K = 64.
    assert loop.count(f"{P}.tcgen05_mma(") == 8 + 4 + 8 + 4 + 4


def _store_first(
    k: torch.Tensor, v: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A body store in the first block of a single-batch iteration."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    d_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    h_all = torch.empty([BH, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            d = h_acc - u[idx, i_t, tile_d, tile_dv].float()
            d_all[idx, i_t, tile_d, tile_dv] = d.to(d_all.dtype)
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
        h_all[idx, tile_d, tile_dv] = h_acc.to(h_all.dtype)
    return d_all, h_all


def test_body_stores_of_the_first_block_are_double_buffered() -> None:
    # The body store's staging buffer is written before the iteration's only
    # barrier and drained after the batch issue; only the acc wait separates
    # that drain from the next iteration's stage-in, so the buffer alternates
    # by parity like the snapshots (two 8 KB buffers after the 32 KB ring).
    t = _tensors()
    u = torch.empty((2, 8, 64, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_store_first, (t["k"], t["v"], u))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.general_loop
    # The derived value (slot 1) is the stored one.
    assert facts.tcgen05.loop_stores == ((1, 2),)
    # Ring, two staging buffers, the epilogue staging tile (the plan folds
    # it into the ring), the barriers with the step barrier, the TMEM slot.
    assert tc.tcgen05_smem_bytes(facts, [1, 64, 64, 64, 64], 2, True) == (
        2 * 16384 + 2 * 64 * 64 * 2 + 64 * 64 * 2 + 8 * (2 + 3 + 1) + 16
    )
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=1024)" in body
    assert f"{P}.mbarrier_init(md_stepb, 1)" in body
    prologue, rest = body.split("for md_i in cutlass.range(", 1)
    loop, epilogue = rest.split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)
    assert loop.count("cute.arch.barrier()") == 1
    assert loop.count(f"{P}.tcgen05_commit(") == 1
    assert "md_srow = md_s0 + 32768 + md_parity * 8192 + md_row64 * 128" in loop
    # The body store goes out by TMA from its parity buffer, issued with the
    # batch (one bulk group per iteration).
    assert f"{WARP_RT}.lds_v4_b32(" not in loop
    assert f"{WARP_RT}.stg_v4_b32(" not in loop and f"{WARP_RT}.stg_b32(" not in loop
    _assert_in_order(
        loop,
        [
            f"{RT}.mbar_wait(md_accb, md_parity ^ 1)",
            "if md_epi:",
            f"{P}.tcgen05_ld(",
            f"{WARP_RT}.sts_v4_b32(md_srow + ",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            "cute.arch.barrier()",
            "if md_warp == 0:",
            f"{P}.tcgen05_mma(",
            "if md_tidx == 0:",
            f"{P}.cp_async_bulk_tensor_global_shared_cta(",
            "md_smem + (32768 + md_parity * 8192), [",
            f"{P}.cp_async_bulk_commit_group()",
            f"{P}.cp_async_bulk_wait_group(1, read=True)",
            f"{P}.tcgen05_commit(md_accb",
        ],
    )
    # The epilogue store reuses the ring.
    assert "md_srow_1 = md_s0 + 0 + md_row64 * 128" in epilogue


def test_direct_stores_skip_the_staging_tile() -> None:
    body = _kernel_body(
        _source(
            _bind_fwd(), _config(64, 64, warps=8, stages=2, cute_mma_dag_store="direct")
        )
    )
    loop = _loop(body)
    assert f"{WARP_RT}.stg_v4_b32(" not in loop and f"{WARP_RT}.lds_v4_b32(" not in loop
    assert loop.count(f"{WARP_RT}.stg_b32(") == 16
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


# --------------------------------------------------------------------------
# Loop outputs in another slot, the epilogue prefetch's bounds
# --------------------------------------------------------------------------


def _ring_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    bf16 = torch.bfloat16
    return (
        torch.empty((16, 4, 64, 64), dtype=bf16, device=CPU_DEVICE),
        torch.empty((16, 64, 64), dtype=bf16, device=CPU_DEVICE),
    )


@pytest.mark.parametrize("fn", (_single_ring, _single_ring_noacc))
def test_a_loop_output_in_another_slot_declines_the_mode(fn: object) -> None:
    # ``T = dot(a_i, T, acc=T * 0.5)`` (the fresh slot of the rescaled-acc
    # fix) and ``T = dot(a_i, T)``: the loop output lives in slot 1 while
    # the carry is slot 0.  The warp emitter copies it back at the
    # iteration end; TMEM has no such copy, so the next iteration's
    # exchange and the epilogue store would read the initial T.  The mode
    # declines the program (the knob never offers it); the bare acc
    # accumulates in place and keeps it.
    args = _ring_inputs()
    bound = _bind_fn(fn, args)
    program = _program(bound)
    (carry,) = program.carries
    assert program.frag_slots[program.resolve(carry.output)] == 1 != carry.slot
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is None
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    with bound.env, bound.host_function:
        assert tc.tcgen05_facts(program) is None
    with bound.env, pytest.raises(InvalidConfig, match="cute_mma_dag must be one of"):
        spec.normalize(_config(warps=4, stages=2).config)
    bare = _bind_fn(_single_ring_bare, args).config_spec
    assert bare.cute_mma_dag is not None
    assert bare.cute_mma_dag.choices == ("off", "warp", "tcgen05")


def test_a_root_dot_rescaling_its_loops_carry_takes_a_fresh_tmem_slot() -> None:
    # ``dot(kk, kk, acc=kk * 0.5)`` after the K loop: the exchange tiles
    # read the carry (column 0), the rescaled acc is stored into slot 1
    # (column 64), the MMAs accumulate there and the store reads it back.
    bound = _bind_fn(_root_rescaled_carry, (_tensors3()["q"],))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=2)))
    assert "md_acc1 = cutlass.inttoptr(md_tmem + 64, " in body
    epilogue = body.rsplit(f"{RT}.mbar_wait(md_accb, ", 1)[1]
    tmem = r"cutlass\.inttoptr\(md_warp % 4 \* 32 << 16 \| md_tmem \+ (\d+), "
    assert re.findall(rf"tcgen05_ld\('16x32bx2', {tmem}", epilogue) == ["0", "64"]
    assert re.findall(rf"tcgen05_st\('16x32bx2', {tmem}", epilogue) == ["64"]
    mmas = re.findall(
        rf"{re.escape(P)}\.tcgen05_mma\({re.escape(P)}\.Tcgen05MMAKind\.F16, "
        rf"{re.escape(P)}\.CTAGroup\.CTA_1, (md_acc\d), ",
        epilogue,
    )
    assert mmas == ["md_acc1"] * 4
    assert epilogue.count(f"{WARP_RT}.sts_v4_b32(") == 8


def test_epilogue_prefetch_keeps_the_rows_and_a_half_row_of_columns() -> None:
    # Gated fwd_o at c = 128 (tcgen05 takes blocks 32 / 64): the attention
    # exchange is a 128-row fragment whose thread holds all 128 columns of
    # its row, so the column sites would carry 128 registers through the
    # loop (defined ahead of it, assigned in its last iteration).  Only the
    # row site goes out ahead of the drain; the 128 column loads stay in
    # the epilogue.  At c = 64 the half-row's 32 columns ride along (33
    # registers, ``test_gated_epilogue_side_loads_go_out_under_the_last_mmas``).
    assert tc.TCGEN05_EPILOGUE_PREFETCH_COLS == 32
    body = _kernel_body(
        _source(_bind_fwd_o(True, c=128), _config(1, 64, 64, warps=4, stages=2))
    )
    prologue, rest = body.split("for md_i in cutlass.range(0, 2):", 1)
    assert [line.strip() for line in prologue.splitlines() if "md_ex" in line] == [
        "md_ex0 = cutlass.Float32(0.0)"
    ]
    loop, epilogue = rest.rsplit(f"{RT}.mbar_wait(md_accb, ", 1)
    guarded = loop.split("if md_i + 1 == 2:", 1)[1]
    loads = [line.strip() for line in guarded.splitlines() if line.strip()]
    assert len(loads) == 1 and "md_ex1" not in body
    assert re.fullmatch(
        rf"md_ex0 = {WARP_RT}\.ldg_f32\(md_base_\d+ \+ cutlass\.Int64\(md_row128\) \* 4\)",
        loads[0],
    )
    assert epilogue.count(f"{WARP_RT}.ldg_f32(") == 128
    columns = re.findall(
        rf"{WARP_RT}\.ldg_f32\(md_base_\d+ \+ cutlass\.Int64\((\d+)\) \* 4\)", epilogue
    )
    assert columns == [str(j) for j in range(1, 128)]
    reads = collections.Counter(re.findall(r"= (md_ex\d+)$", epilogue, re.MULTILINE))
    assert reads == {"md_ex0": 2}


def _epi_inputs(stored: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, t = _ring_inputs()
    shape = (16, 64, 64) if stored else (16, 64)
    return a, t, torch.empty(shape, dtype=torch.float32, device=CPU_DEVICE)


def test_the_program_order_loop_prefetches_the_epilogue_row_after_the_loop() -> None:
    # ``_gen_epi``: the carry is the B operand, so the loop body is emitted
    # in program order and has no last-iteration slot between the MMA issue
    # and the acc wait; the decay row goes out after the loop, ahead of the
    # drain (its register still defined before the loop), and the epilogue
    # issues no global load.
    bound = _bind_fn(_gen_epi, _epi_inputs(stored=False))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.general_loop
    body = _kernel_body(_source(bound, _config(warps=4, stages=2)))
    prologue, rest = body.split("for md_i in cutlass.range(0, 4):", 1)
    assert [line.strip() for line in prologue.splitlines() if "md_ex" in line] == [
        "md_ex0 = cutlass.Float32(0.0)"
    ]
    loop, epilogue = rest.split(f"{RT}.mbar_wait(md_accb, 4 - 0 - 1 & 1)", 1)
    assert "md_ex" not in loop.rstrip().rsplit("\n", 1)[0]
    assert re.fullmatch(
        rf"    md_ex0 = {WARP_RT}\.ldg_f32\(md_base_\d+ \+ cutlass\.Int64\(md_row64\) \* 4\)",
        loop.rstrip().rsplit("\n", 1)[1],
    )
    assert "ldg_" not in epilogue
    assert re.findall(r"= (md_ex\d+)$", epilogue, re.MULTILINE) == ["md_ex0"]


def test_a_program_that_loads_a_tensor_it_stores_is_not_mine_either() -> None:
    # ``g[b] = T; out[b] = T * exp2(g[b, 0, :])``: the decay row is a row of
    # a tensor the epilogue stores first.  The lowering used to keep the
    # load in the epilogue behind the store's drain; now the matcher
    # declines any program whose load shares storage with a store (both
    # modes, one rule), so no dag knob is offered at all.
    bound = _bind_fn(_gen_epi_stored, _epi_inputs(stored=True))
    assert bound.config_spec.cute_mma_dag is None
    assert bound.config_spec.cute_mma_dag_facts is None


# --------------------------------------------------------------------------
# Region programs
# --------------------------------------------------------------------------


def _split_loop(body: str, var: str) -> tuple[str, str]:
    """``(before, loop and after)`` at the first ``for var in`` line."""
    index = body.index(f"for {var} in cutlass.range(")
    return body[:index], body[index:]


def test_region_programs_admit_the_mode_behind_the_warp_seeds() -> None:
    # dk, dqkw and wy_dL (both decay modes) admit the tensor-memory mode
    # through the region renderer; the warp mode stays the default and the
    # first seed, a tensor-memory seed follows.
    for bind in (
        _bind_dk,
        _bind_dqkw,
        lambda: _bind_dqkw(gated=True),
        _bind_wy_dl,
        lambda: _bind_wy_dl(gated=True),
    ):
        spec = bind().config_spec
        assert spec.cute_mma_dag is not None
        assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        assert facts.tcgen05.regions
        assert spec.default_config().config["cute_mma_dag"] == "warp"
        modes = [c.config["cute_mma_dag"] for c in spec.compiler_seed_configs]
        assert modes[0] == "warp" and "tcgen05" in modes
    # dk: both orientations of dL (plain casts of one load) share a hoisted
    # tile (fp32 source -> a raw tile beside it), k rides loop 0's ring,
    # the store stages through two buffers in loop 0's frame; one slot
    # lives in TMEM.
    facts = _bind_dk().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    items = facts.tcgen05.items
    assert [(i.kind, i.frame, i.copies) for i in items] == [
        ("tile", -1, 1),
        ("raw", -1, 1),
        ("tile", 0, 0),
        ("staging", 0, 2),
    ]
    assert facts.tcgen05.tmem_slots == (0,)
    assert facts.tcgen05.ring_loops == 1 and facts.tcgen05.staged_regions == 1
    # Block ids of dk: loop bd 2, lead 0, rdim C 1 -> axis indices 0..2.
    resolved = [64, 1, 64]
    layout = regions.region_layout(items, facts.loops, resolved, 2)
    assert layout.ring_base == {0: 0} and layout.ring_bytes == {0: 8192}
    # Loop 0's frame first (the ring, the two staging buffers), then the
    # root's hoisted tile with its raw tile.
    assert layout.offsets == (32768, 40960, 0, 16384)
    assert layout.total == 57344
    assert tc.tcgen05_smem_bytes(facts, resolved, 2, True) == 57344 + 8 * 4 + 16
    assert tc.tcgen05_slot_columns(facts, resolved) == 64
    # wy_dL: the vector scratch (four warps of 128 entries) sits in the
    # root frame; a derived value over the dot result cannot hold two
    # 128-column rows; dqkw's five slots exceed the TMEM at bd = 128.
    facts = _bind_wy_dl(gated=True).config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    scratch = [i for i in facts.tcgen05.items if i.kind == "scratch"]
    assert scratch == [tc.Tcgen05SmemItem("scratch", -1, -1, -1, 4, 4)]
    facts = _bind_wy_dl().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    reason = dag.mma_dag_config_reason(
        facts, {0: 1, 2: 128, 3: 64, 4: 64}, 4, 2, "tcgen05"
    )
    assert reason is not None and "derived value holds 256" in reason
    facts = _bind_dqkw().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.tmem_slots == (0, 1, 2, 3, 4)
    reason = dag.mma_dag_config_reason(
        facts, {0: 1, 1: 64, 3: 128, 4: 64}, 4, 2, "tcgen05"
    )
    assert reason is not None and "576 TMEM columns" in reason
    assert (
        dag.mma_dag_config_reason(facts, {0: 1, 1: 64, 3: 64, 4: 64}, 4, 2, "tcgen05")
        is None
    )


def test_renders_dk_with_hoisted_tiles_and_a_seeded_accumulator() -> None:
    body = _kernel_body(_source(_bind_dk(), _config(64, warps=4, stages=2)))
    prologue, loop = _split_loop(body, "md_i")
    # One commit barrier, loop 0's ring (two slots), the root's staged
    # tiles' barrier; the running counters.
    assert "md_cb = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)" in prologue
    assert "md_full0 = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)" in prologue
    assert "md_sfr = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)" in prologue
    assert (
        "md_cn = cutlass.Int32(0)" in prologue
        and "md_rk0 = cutlass.Int32(0)" in prologue
    )
    # The hoisted dL tile lands by TMA (raw fp32) while the ring prologue
    # issues the first iteration's k tile, converts into the bf16 UMMA
    # tile (eight packets per thread) behind one wait, then a barrier.
    _assert_in_order(
        prologue,
        [
            f"{P}.mbarrier_arrive_expect_tx(md_sfr, 16384)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + 40960)",
            f"{P}.mbarrier_arrive_expect_tx(md_full0 + md_rk0 % 2, 8192)",
            f"{RT}.mbar_wait(md_sfr, 0)",
            f"{RT}.lds_v4_f32(md_s0 + 40960",
            f"{WARP_RT}.sts_v2_b32(md_s0 + 32768",
            "cute.arch.barrier()",
        ],
    )
    assert prologue.count(f"{RT}.lds_v4_f32(") == 8
    assert prologue.count(f"{P}.cp_async_bulk_tensor_shared_cta_global(") == 2
    # Per iteration: the slot / phase / parity from the counter, the refill
    # one iteration ahead, the seed (two fp32 tiles at the fragment
    # coordinates) stored to TMEM, both dots in one batch (the ring wait by
    # warp 0), one commit, the wait, the staged store, the counter.
    _assert_in_order(
        loop,
        [
            "md_st0 = md_rk0 % 2",
            "md_rd0 = md_rk0 // 2 & 1",
            "md_pt0 = md_rk0 % 2",
            f"{P}.mbarrier_arrive_expect_tx(md_full0 + (md_rk0 + 1) % 2, 8192)",
            "if md_epi:",
            "md_dr_0 = cutlass.Array(cutlass.Float32, 32, alignment=16)",
            f"{P}.tcgen05_st(",
            "cute.arch.barrier()",
            "if md_warp == 0:",
            f"{RT}.mbar_wait(md_full0 + md_st0, md_rd0)",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_cb",
            "md_cn = md_cn + 1",
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            f"{P}.tcgen05_ld(",
            "md_srow = md_s0 + 16384 + md_pt0 * 8192 + md_row64 * 128",
            f"{WARP_RT}.sts_v4_b32(",
            "cute.arch.barrier()",
            f"{WARP_RT}.lds_v4_b32(md_s0 + 16384 + md_pt0 * 8192",
            f"{WARP_RT}.stg_v4_b32(",
            "md_rk0 = md_rk0 + 1",
        ],
    )
    assert loop.count(f"{P}.tcgen05_mma(") == 8
    assert loop.count(f"{P}.tcgen05_commit(") == 1
    assert loop.count(f"{P}.tcgen05_st(") == 1 and loop.count(f"{P}.tcgen05_ld(") == 1
    # dL as A: K-major straight, MN-major transposed; k is MN-major B.
    assert "n_dim=64, m_dim=64, a_major=0, b_major=1" in prologue
    assert "n_dim=64, m_dim=64, a_major=1, b_major=1" in prologue
    assert "alloc_smem(cutlass.Uint8, 57344, alignment=1024)" in body


def test_renders_wy_dl_sequential_loops_and_vector_carries() -> None:
    body = _kernel_body(_source(_bind_wy_dl(), _config(64, 64, 64, warps=4, stages=2)))
    # Three rings, no staged region (A_inv's tile is hoisted by the
    # synchronous pass... through TMA: the root barrier), four vector
    # registers.
    assert body.count("cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)") == 3
    for slot in range(4):
        assert f"md_vv{slot} = cutlass.Float32(0.0)" in body
    first, rest = _split_loop(body, "md_i")
    loop0, rest = _split_loop(rest, "md_i1")
    loop1, rest = _split_loop(rest, "md_i2")
    # Loop 1 continues dAinv in place (no seed), its vector carry copies
    # nothing (same slot); kk's loop seeds zeros.
    assert loop0.count(f"{P}.tcgen05_st(") == 0
    assert "md_seed" in first and "md_seed" not in loop0
    assert loop1.count("md_seed") > 0
    # Row sums: the thread's local columns, then the other half.
    assert loop0.count("cute.arch.shuffle_sync_bfly(") == 1
    assert loop1.count("cute.arch.shuffle_sync_bfly(") == 1
    assert rest.count("cute.arch.shuffle_sync_bfly(") == 1
    assert "barrier_id=1" not in body
    # The root chain after loop 1: dAinv's exchange tile feeds dot
    # (MN-major B), dot's exchange tile feeds dot_1 (K-major A) after its
    # own commit wait, kk's loop seeds zeros; after that loop the dL store,
    # the row sums (one half-row shuffle) and the vector store (the lanes
    # owning a row), then the staged store drains.
    root = loop1[loop1.index("md_rk1 = md_rk1 + 1") :]
    _assert_in_order(
        root,
        [
            f"{P}.tcgen05_ld(",
            f"{WARP_RT}.sts_v4_b32(",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_cb",
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            f"{P}.tcgen05_ld(",
            f"{WARP_RT}.sts_v4_b32(",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_cb",
            "md_seed",
        ],
    )
    after = rest[rest.index("md_rk2 = md_rk2 + 1") :]
    _assert_in_order(
        after,
        [
            f"{P}.tcgen05_ld(",
            "cute.arch.shuffle_sync_bfly(",
            "if md_lane < 16:",
            f"{WARP_RT}.stg_f32(",
            "cute.arch.barrier()",
            f"{WARP_RT}.stg_v4_b32(",
        ],
    )


def test_renders_gated_wy_dl_column_sums_through_the_scratch() -> None:
    body = _kernel_body(
        _source(_bind_wy_dl(gated=True), _config(64, 64, 64, warps=4, stages=2))
    )
    _first, rest = _split_loop(body, "md_i2")
    # dLL.sum(-2): the 16 rows of each warp by shuffles per local column,
    # the four epilogue warps through the scratch (their own barrier),
    # read back at the thread's row.
    assert rest.count(f"{WARP_RT}.lane_group_sum(") == 32
    assert rest.count("cute.arch.barrier(barrier_id=1, number_of_threads=128)") == 2
    assert rest.count(f"{WARP_RT}.sts_f32(") == 32
    assert rest.count(f"{WARP_RT}.lds_f32(") == 4
    assert "if md_lane % 16 == 0:" in rest
    assert "alloc_smem(cutlass.Uint8, 149504, alignment=1024)" in body


def test_renders_dqkw_nested_loops_with_staged_tiles() -> None:
    body = _kernel_body(_source(_bind_dqkw(), _config(64, 64, 64, warps=4, stages=2)))
    # Rings for loop 0 and the nested loop 2, a staged region for loop 1
    # (k_d and q_d once per outer iteration).
    assert "md_full0 = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)" in body
    assert "md_full2 = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)" in body
    assert "md_sf1 = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)" in body
    assert "md_full1" not in body
    _first, rest = _split_loop(body, "md_i1")
    outer, inner = _split_loop(rest, "md_i2")
    # The outer iteration: its parity, the staged tiles' TMA, the three
    # zero seeds of the inner carries, the inner ring prologue.
    _assert_in_order(
        outer,
        [
            "md_pt1 = md_rk1 % 2",
            f"{P}.mbarrier_arrive_expect_tx(md_sf1,",
            "md_seed",
            f"{P}.mbarrier_arrive_expect_tx(md_full2 + md_rk2 % 2,",
        ],
    )
    assert outer.count(f"{P}.tcgen05_st(") == 3
    # The inner iteration: the no-acc dot, its commit, the subtraction
    # (ld both, store dW), the two accumulating dots in a second batch.
    # The inner iteration: the no-acc dot and the two accumulating dots in
    # one batch (the subtraction reads the first's result, so it follows
    # the commit: ld both, store dW).
    inner_body = inner[: inner.index("md_rk2 = md_rk2 + 1")]
    assert inner_body.count(f"{P}.tcgen05_commit(md_cb") == 1
    assert inner_body.count(f"{P}.tcgen05_mma(") == 12
    assert inner_body.count(f"{P}.tcgen05_st(") == 1
    assert inner_body.index(f"{P}.tcgen05_st(") > inner_body.index(
        f"{P}.tcgen05_commit(md_cb"
    )
    # After the inner loop: the staged tiles are waited, dW is stored, dq
    # and dk_as accumulate onto the carries in one batch with dattn's
    # exchange tile (both orientations from one tile), then their stores.
    after = inner[inner.index("md_rk2 = md_rk2 + 1") :]
    assert f"{RT}.mbar_wait(md_sf1, md_rk1 & 1)" in after
    assert after.count(f"{P}.tcgen05_commit(md_cb") == 1
    # dW_out fp32 (eight packets per thread), dq_out bf16 (four), dk_as_out
    # fp32 (eight).
    assert after.count(f"{WARP_RT}.stg_v4_b32(") == 8 + 4 + 8
    assert "md_rk1 = md_rk1 + 1" in after
    assert "md_idesc3 = " in body
    assert "n_dim=64, m_dim=64, a_major=0, b_major=1" in body
    assert "n_dim=64, m_dim=64, a_major=1, b_major=1" in body


def _bind_fwd_o_anch_f32(bhn: int = 16, d: int = 128, dv: int = 128) -> BoundKernel:
    """chunk_fwd_o_diag_anchored over fp32 inputs (the GLA example)."""
    f32 = torch.float32
    return _bind(
        "chunk_fwd_o_diag_anchored_helion",
        (
            _cpu(bhn, 64, d, dtype=f32),
            _cpu(bhn, 64, dv, dtype=f32),
            _cpu(bhn, 64, d, dtype=f32),
            _cpu(bhn, d, dv, dtype=f32),
            _cpu(bhn, 64, 64, dtype=f32),
            0.125,
        ),
    )


def test_mn_major_tf32_tiles_are_b_operands_only() -> None:
    # The GLA example's ``h[bhn, tile_d, tile_dv]`` B operand is an fp32
    # source read with K along its rows: an MN-major tf32 tile, read under
    # the 32-byte-unit swizzle straight from TMA (box of 32 columns per
    # slab, ``s128b_atom_32b``), so the kernel admits the tcgen05 mode.
    bound = _bind_fwd_o_anch_f32()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        assert regions.region_program_reason(program) is None
    source = _source(bound, _config(1, 64, 64, warps=4, stages=2))
    body = _kernel_body(source)
    assert "'tma0_box': [32, 64, 1], 'tma0_swizzle': 's128b_atom_32b'" in source
    assert (
        "leading_byte_offset=8192, stride_byte_offset=512, "
        f"layout={P}.Tcgen05SmemSwizzle.SWIZZLE_128B_ATOM_32B"
    ) in body
    assert "a_major=0, b_major=1" in body
    # An A operand with K along its rows stays declined (the layout is
    # verified for B operands): the state recurrence over fp32 chunks reads
    # ``k_i^T`` as A.
    f32 = torch.float32
    bound = _bind_fn(
        _store_after, (_cpu(2, 8, 64, 64, dtype=f32), _cpu(2, 8, 64, 64, dtype=f32))
    )
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        assert regions.region_program_reason(program) == (
            "permute: an MN-major tf32 A operand"
        )


def test_wy_admits_the_tf32_chain_and_seeds_it_first() -> None:
    # chunk_fwd_wy_delta: a bf16 dot (kk), the five-step tf32 Neumann chain
    # with carries seeded from a pointwise value over kk (Apow in a fresh
    # slot each step, copied back) and the tf32 w / u dots over widened
    # bf16 tiles.  fp32 operands are admitted (tf32 kind) and, unlike the
    # 16-bit region programs, the tensor-memory seed comes first.
    for gated in (False, True):
        spec = _bind_wy(gated=gated).config_spec
        assert spec.cute_mma_dag is not None
        assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        assert facts.tcgen05.regions and facts.tcgen05.tf32
        assert spec.default_config().config["cute_mma_dag"] == "tcgen05"
        modes = [c.config["cute_mma_dag"] for c in spec.compiler_seed_configs]
        assert modes[0] == "tcgen05" and "warp" in modes
    facts = _bind_wy().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    tc_facts = facts.tcgen05
    # The chain loop's frame holds Apow straight (K-major) and transposed
    # (its own tile), the squared Apow straight and T transposed; T's
    # natural tile feeds the w and u loops from the root frame; the kt / vt
    # operands widen bf16 raw tiles into transposed fp32 tiles in their
    # rings.
    exchanges = [
        (i.frame, i.elem_bytes) for i in tc_facts.items if i.kind == "exchange"
    ]
    assert exchanges == [(1, 4), (1, 4), (1, 4), (1, 4), (-1, 4)]
    assert [(i.frame, i.elem_bytes) for i in tc_facts.items if i.kind == "raw"] == [
        (2, 2),
        (3, 2),
    ]
    # TMEM: the fresh Apow slot and the w / u results share kk's columns
    # (never live together), 192 of the 384 a flat layout would take.
    resolved = [64, 64, 64, 1, 64]
    columns, total = regions.region_tmem_layout(facts, resolved)
    assert columns == {0: 0, 1: 64, 2: 128, 3: 0, 4: 0, 5: 0} and total == 192


def test_renders_the_wy_chain_with_tf32_mmas() -> None:
    body = _kernel_body(_source(_bind_wy(), _config(64, 64, 64, warps=4, stages=2)))
    assert "alloc_smem(cutlass.Uint8, 98304, alignment=1024)" in body
    assert f"{P}.tcgen05_alloc(md_tslot, 256," in body
    for slot in (3, 4, 5):
        assert f"md_acc{slot} = cutlass.inttoptr(md_tmem + 0," in body
    assert body.count(f"{P}.Tcgen05MMAKind.F16") == 4
    assert body.count(f"{P}.Tcgen05MMAKind.TF32") == 4 * 8
    assert body.count("a_dtype=cutlass.Float32, b_dtype=cutlass.Float32") == 4
    assert "a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16" in body
    first, rest = _split_loop(body, "md_i")
    kk_loop, rest = _split_loop(rest, "md_i1")
    chain, rest = _split_loop(rest, "md_i2")
    w_loop, u_loop = _split_loop(rest, "md_i3")
    assert "for md_i1 in cutlass.range(0, 5):" in chain
    # The chain's carries are seeded from kk's row: both arrays evaluate the
    # pointwise seed before the loop; each step runs two tf32 batches
    # (Apow = Apow @ Apow, then T += Apow @ T), the B tiles written as
    # natural four-wide packets under the 32-byte-unit swizzle (the A tiles
    # under SW128), and copies Apow back to its slot.
    init = kk_loop[kk_loop.index("md_rk0 = md_rk0 + 1") :]
    assert init.count(f"{P}.tcgen05_st(") == 2 and "md_seed" in init
    assert chain.count(f"{P}.tcgen05_commit(md_cb") == 2
    assert chain.count(f"{WARP_RT}.sts_f32(") == 0
    assert chain.count(f"{RT}.sts_v4_f32(") == 40
    assert chain.count("% 8 ^ md_row64 % 8) * 16") == 24
    assert chain.count("% 8 ^ md_row64 % 4 * 2) * 16") == 16
    assert chain.count(f"{P}.tcgen05_st(") == 1
    _assert_in_order(
        chain,
        [
            f"{P}.tcgen05_ld(",
            f"{RT}.sts_v4_f32(",
            "cute.arch.barrier()",
            f"{P}.Tcgen05MMAKind.TF32",
            f"{P}.tcgen05_commit(md_cb",
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            f"{P}.tcgen05_ld(",
            f"{RT}.sts_v4_f32(",
            "cute.arch.barrier()",
            f"{P}.Tcgen05MMAKind.TF32",
            f"{P}.tcgen05_commit(md_cb",
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            f"{P}.tcgen05_ld(",
            f"{P}.tcgen05_st(",
        ],
    )
    # Descriptors: A K-major SW128 (LBO 16, SBO 1024), B MN-major under the
    # ATOM_32B swizzle (LBO = the 32-column slab stride K * 128, SBO = the
    # four-row swizzle period 512); the start advances 1024 B per K step.
    assert body.count("leading_byte_offset=16, stride_byte_offset=1024") == 6
    assert (
        body.count(
            "leading_byte_offset=8192, stride_byte_offset=512, "
            f"layout={P}.Tcgen05SmemSwizzle.SWIZZLE_128B_ATOM_32B"
        )
        == 4
    )
    assert body.count("a_major=0, b_major=1") == 4
    assert body.count("advance_start_address(1024)") == 4
    # w: the bf16 k tile lands raw by TMA, the widen pass writes beta * k
    # as natural fp32 packets into the MN-major B tile (four raw packets
    # per thread, two fp32 packets each), T's natural tile is the K-major A,
    # eight tf32 MMAs, the bf16 result stages out.
    assert w_loop.count(f"{WARP_RT}.lds_v4_b32(") == 8
    assert w_loop.count(f"{WARP_RT}.unpack_bf16x2(") == 16
    assert w_loop.count(f"{WARP_RT}.sts_f32(") == 0
    assert w_loop.count(f"{RT}.sts_v4_f32(") == 16
    assert len(re.findall(r"\^ md_pr_t\d+_\d+ % 4 \* 2\) \* 16", w_loop)) == 8
    assert w_loop.count(f"{P}.Tcgen05MMAKind.TF32") == 8
    assert w_loop.count(f"{WARP_RT}.stg_v4_b32(") == 4
    assert u_loop.count(f"{P}.Tcgen05MMAKind.TF32") == 8


# --------------------------------------------------------------------------
# Region programs: the kda / anchored kernels
# --------------------------------------------------------------------------


def _cpu(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=CPU_DEVICE)


def _bind_bwd_o_kda(bhn: int = 16, d: int = 64, dv: int = 64) -> BoundKernel:
    c, f32 = 64, torch.float32
    return _bind(
        "chunk_bwd_o_kda_helion",
        (
            _cpu(bhn, c, d),
            _cpu(bhn, c, dv),
            _cpu(bhn, d, dv),
            _cpu(bhn, c, c),
            _cpu(bhn, c, d, dtype=f32),
            _cpu(bhn, c, dv),
            0.125,
        ),
    )


def _bind_state_du_kda(
    bh: int = 2, n: int = 8, d: int = 64, dv: int = 64
) -> BoundKernel:
    c, f32 = 64, torch.float32
    return _bind(
        "chunk_bwd_state_du_kda_helion",
        (
            _cpu(bh, n, c, d),
            _cpu(bh, n, c, d),
            _cpu(bh, n, c, d, dtype=f32),
            _cpu(bh, n, c, dv, dtype=f32),
            _cpu(bh, n, d, dv, dtype=f32),
            _cpu(bh, d, dv, dtype=f32),
        ),
    )


def _bind_fwd_o_anch(bhn: int = 16, d: int = 64, dv: int = 64) -> BoundKernel:
    c, f32 = 64, torch.float32
    return _bind(
        "chunk_fwd_o_diag_anchored_helion",
        (
            _cpu(bhn, c, d),
            _cpu(bhn, c, dv),
            _cpu(bhn, c, d, dtype=f32),
            _cpu(bhn, d, dv),
            _cpu(bhn, c, c),
            0.125,
        ),
    )


_REGION_BINDERS: dict[str, Callable[[], BoundKernel]] = {
    "dk": _bind_dk,
    "dqkw": _bind_dqkw,
    "dqkw_gated": lambda: _bind_dqkw(gated=True),
    "wy_dL": _bind_wy_dl,
    "wy_dL_gated": lambda: _bind_wy_dl(gated=True),
    "wy": _bind_wy,
    "wy_gated": lambda: _bind_wy(gated=True),
    "bwd_o_kda": _bind_bwd_o_kda,
    "state_du_kda": _bind_state_du_kda,
    "fwd_o_anch": _bind_fwd_o_anch,
    "probe_h1": lambda: _bind_h1(),
    "probe_h3": lambda: _bind_h3(),
    "probe_h4": lambda: _bind_h4(),
}


def _match(bound: BoundKernel) -> dag.MmaDagProgram:
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return program


def _commits_while_one_is_outstanding(body: str) -> list[int]:
    """Abstract interpretation of the kernel body's commit / wait order
    (loops run 0..n times, branches are taken): the line numbers of
    ``tcgen05_commit(md_cb`` issued while a commit is still outstanding
    -- the single barrier's parity tells one phase apart."""
    root: list[object] = []
    stack: list[tuple[int, list[object]]] = [(-1, root)]
    for number, line in enumerate(body.split("\n")):
        text = line.strip()
        if not text:
            continue
        indent = len(line) - len(line.lstrip())
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if text.startswith("for ") and text.endswith(":"):
            children: list[object] = []
            current.append(("loop", number, children))
            stack.append((indent, children))
        elif "tcgen05_commit(md_cb" in text:
            current.append(("commit", number))
        elif "mbar_wait(md_cb" in text:
            current.append(("wait", number))
        elif text.startswith(("if ", "elif ", "else", "with ")) and text.endswith(":"):
            stack.append((indent, current))
    errors: set[int] = set()

    def run(nodes: list[object], outstanding: int) -> int:
        for node in nodes:
            kind = node[0]  # type: ignore[index]
            if kind == "commit":
                if outstanding:
                    errors.add(node[1])  # type: ignore[index]
                outstanding = min(outstanding + 1, 2)
            elif kind == "wait":
                outstanding = 0
            else:
                worst = state = outstanding
                for _trip in range(4):
                    state = run(node[2], state)  # type: ignore[index]
                    worst = max(worst, state)
                    state = max(state, outstanding)
                outstanding = worst
        return outstanding

    run(root, 0)
    return sorted(errors)


def _after_loop(body: str, var: str) -> str:
    """The body from the first line after the ``for var in`` loop (the
    next line at the loop's indentation or less)."""
    lines = body.split("\n")
    start = next(
        i for i, line in enumerate(lines) if f"for {var} in cutlass.range(" in line
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        i
        for i, line in enumerate(lines[start + 1 :], start + 1)
        if line.strip() and len(line) - len(line.lstrip()) <= indent
    )
    return "\n".join(lines[end:])


def _tmem_columns(body: str) -> dict[int, int]:
    return {
        int(slot): int(column)
        for slot, column in re.findall(
            r"md_acc(\d+) = cutlass\.inttoptr\(md_tmem \+ (\d+),", body
        )
    }


def _resolved_axes(
    program: dag.MmaDagProgram, bound: BoundKernel, block_sizes: list[int]
) -> list[int]:
    by_block = {
        spec.block_id: size
        for spec, size in zip(bound.config_spec.block_sizes, block_sizes, strict=True)
    }
    return [
        by_block.get(axis.block_id, axis.extent if isinstance(axis.extent, int) else 0)
        for axis in program.axes
    ]


@pytest.mark.parametrize("kernel", sorted(_REGION_BINDERS))
def test_region_programs_keep_one_commit_in_flight_and_disjoint_live_slots(
    kernel: str,
) -> None:
    # Three rules over every region program the examples admit
    # (the kda and anchored kernels included), at its tensor-memory seeds:
    # a tile-only derived value that a dot reads, a loop carries or another
    # region reads holds its own tensor-memory columns (``derived_crosses``;
    # the rest stay in registers, so the renderer never loads or stores
    # them -- it asserts); slots live together never share columns; and no
    # batch commits while the previous commit is outstanding (a dots-only
    # batch behind a loop used to).
    bound = _REGION_BINDERS[kernel]()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    tc_facts = facts.tcgen05
    assert tc_facts.regions
    program = _match(bound)
    for derived in program.derived:
        assert (derived.slot in tc_facts.tmem_slots) == (
            bool(derived.frags) or regions.derived_crosses(program, derived)
        )
    for dot in program.dots:
        for operand in (dot.a, dot.b):
            if operand.frag is not None:
                assert program.frag_slots[operand.frag] in tc_facts.tmem_slots
    seeds = [
        config
        for config in spec.compiler_seed_configs
        if config.config["cute_mma_dag"] == "tcgen05"
    ]
    assert seeds
    for seed in seeds[:2]:
        with bound.env:
            normalized = spec.normalized_config(seed)
        body = _kernel_body(bound.to_code(normalized))
        assert _commits_while_one_is_outstanding(body) == []
        _assert_live_slots_disjoint(
            bound, program, body, normalized.config["block_sizes"]
        )
        columns = _tmem_columns(body)
        resolved = _resolved_axes(program, bound, normalized.config["block_sizes"])
        groups = [*facts.live_groups, *tc_facts.tmem_conflicts]
        for group in groups:
            for a in group:
                for b in group:
                    if a < b and a in tc_facts.tmem_slots and b in tc_facts.tmem_slots:
                        wa = tc._align(
                            resolved[facts.frags[a][1]], tc.TCGEN05_SLOT_ALIGN
                        )
                        wb = tc._align(
                            resolved[facts.frags[b][1]], tc.TCGEN05_SLOT_ALIGN
                        )
                        assert (
                            columns[a] + wa <= columns[b]
                            or columns[b] + wb <= columns[a]
                        ), (kernel, a, b, columns)


def _probe_h1(
    q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor, do: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A root carry (dattn) read only by an inner loop's dot, and a fresh
    dot (z) of the enclosing loop after it."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = hl.specialize(do.size(2))
    out_a = torch.empty([BHN, C, DV], dtype=torch.float32, device=q.device)
    out_b = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        dattn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            dattn = hl.dot(
                do[tile_bhn, :, tile_dv],
                v_new[tile_bhn, :, tile_dv].transpose(-2, -1),
                acc=dattn,
            )
        dattn = dattn.to(torch.bfloat16)
        for tile_d in hl.tile(D):
            for tile_dv in hl.tile(DV):
                y = hl.dot(dattn, v_new[tile_bhn, :, tile_dv])
                out_a[tile_bhn, :, tile_dv] = y
            z = hl.dot(q[tile_bhn, :, tile_d], k[tile_bhn, :, tile_d].transpose(-2, -1))
            out_b[tile_bhn, :, :] = z
    return out_a, out_b


def _probe_h3(
    do: torch.Tensor, h: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A tile-only derived value computed before a nested dot loop and
    reduced after it, in the same loop."""
    BHN = do.size(0)
    C = hl.specialize(do.size(1))
    DV = hl.specialize(do.size(2))
    D = h.size(1)
    dq_out = torch.empty([BHN, C, D], dtype=torch.float32, device=do.device)
    hd_out = torch.empty([BHN], dtype=torch.float32, device=do.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        hd = hl.zeros([tile_bhn], dtype=torch.float32)
        for tile_d in hl.tile(D):
            ht2 = h[tile_bhn, tile_d, :].to(torch.float32) * g[tile_bhn, tile_d, :]
            dq = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            for tile_dv in hl.tile(DV):
                dq = hl.dot(
                    do[tile_bhn, :, tile_dv],
                    h[tile_bhn, tile_d, tile_dv].transpose(-2, -1),
                    acc=dq,
                )
            dq_out[tile_bhn, :, tile_d] = dq
            hd = hd + ht2.sum(-1).sum(-1)
        hd_out[tile_bhn] = hd
    return dq_out, hd_out


def _probe_h4(
    a32: torch.Tensor, b32: torch.Tensor, q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A tf32 dot ahead of a bf16 dot whose operand is rewritten in
    place."""
    BHN = a32.size(0)
    C = hl.specialize(a32.size(1))
    DV = a32.size(2)
    D = q.size(2)
    out_x = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_y = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        x = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_dv in hl.tile(DV):
            x = hl.dot(
                a32[tile_bhn, :, tile_dv],
                b32[tile_bhn, :, tile_dv].transpose(-2, -1),
                acc=x,
            )
        out_x[tile_bhn, :, :] = x
        y = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qs = (q[tile_bhn, :, tile_d].to(torch.float32) * 2.0).to(torch.bfloat16)
            y = hl.dot(qs, k[tile_bhn, :, tile_d].transpose(-2, -1), acc=y)
        out_y[tile_bhn, :, :] = y
    return out_x, out_y


def _bind_probe(
    fn: object, *shapes: tuple[tuple[int, ...], torch.dtype]
) -> BoundKernel:
    return _bind_fn(fn, tuple(_cpu(*shape, dtype=dtype) for shape, dtype in shapes))


def _bind_h1(bhn: int = 16, d: int = 128, dv: int = 64) -> BoundKernel:
    bf16, c = torch.bfloat16, 64
    return _bind_probe(
        _probe_h1,
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, dv), bf16),
        ((bhn, c, dv), bf16),
    )


def _bind_h3(bhn: int = 16, d: int = 128, dv: int = 64) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_h3, ((bhn, c, dv), bf16), ((bhn, d, dv), bf16), ((bhn, d, dv), f32)
    )


def _bind_h4(bhn: int = 16, d: int = 128, dv: int = 64) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_h4,
        ((bhn, c, dv), f32),
        ((bhn, c, dv), f32),
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
    )


def _unrolled_live_pairs(program: dag.MmaDagProgram) -> set[tuple[int, int]]:
    """Slot pairs live together in an execution with every loop run twice
    (an independent, loop-aware liveness).  A dot / derived result is live
    from each of its definitions to the last read before the next one (a
    value redefined by every iteration is not live across iterations); a
    carry is live from its loop's entry (where it reads the slot seeding
    it) over the whole loop and on to its phi's last reader.  A step reads
    its inputs before it writes its result, so a definition at another
    slot's last read does not overlap it."""
    flat: list[dag.Step] = []
    spans: dict[int, tuple[int, int]] = {}

    def walk(steps: Sequence[dag.Step]) -> None:
        for step in steps:
            if isinstance(step, dag.Loop):
                start = len(flat)
                for _trip in range(2):
                    walk(step.steps)
                spans[step.index] = (start, len(flat))
            else:
                flat.append(step)

    walk(program.body)
    carries: dict[int, int] = {}
    for loop in program.loops:
        for carry in loop.carries:
            carries[carry.slot] = loop.index
    # (slot, first, last, defined) intervals.
    intervals: list[tuple[int, int, int, bool]] = []
    open_at: dict[int, int] = {}
    last_use: dict[int, int] = {}

    def use(slot: int, index: int) -> None:
        if slot in carries:
            last_use[slot] = max(last_use.get(slot, index), index)
            return
        if slot not in open_at:
            open_at[slot] = index
        last_use[slot] = index

    def define(slot: int, index: int) -> None:
        if slot in carries:
            return
        if slot in open_at:
            intervals.append((slot, open_at[slot], last_use[slot], True))
        open_at[slot] = index
        last_use[slot] = index

    seeds: dict[int, int] = {}
    for loop in program.loops:
        start, _end = spans[loop.index]
        for carry in loop.carries:
            if carry.init_frag is not None:
                seeds[start] = program.frag_slots[program.resolve(carry.init_frag)]
    for index, step in enumerate(flat):
        if index in seeds:
            use(seeds[index], index)
        if isinstance(step, dag.Dot):
            for operand in (step.a, step.b):
                if operand.frag is not None:
                    use(program.frag_slots[program.resolve(operand.frag)], index)
            if step.acc_slot is not None:
                use(step.acc_slot, index)
            define(step.slot, index)
        elif isinstance(step, dag.Derived):
            for frag in step.frags:
                use(program.frag_slots[frag], index)
            define(step.slot, index)
        elif isinstance(step, (dag.Store, dag.Reduce)) and step.frag is not None:
            use(program.frag_slots[step.frag], index)
    for slot, start in open_at.items():
        intervals.append((slot, start, last_use[slot], True))
    for slot, loop in carries.items():
        start, end = spans[loop]
        intervals.append((slot, start, max(end, last_use.get(slot, end)), False))

    def overlap(a: tuple[int, int, int, bool], b: tuple[int, int, int, bool]) -> bool:
        _sa, fa, la, da = a
        _sb, fb, lb, db = b
        if fa > lb or fb > la:
            return False
        if fa == lb and da:
            return False
        return not (fb == la and db)

    pairs: set[tuple[int, int]] = set()
    for a in intervals:
        for b in intervals:
            if a[0] < b[0] and overlap(a, b):
                pairs.add((a[0], b[0]))
    return pairs


def _assert_live_slots_disjoint(
    bound: BoundKernel, program: dag.MmaDagProgram, body: str, block_sizes: list[int]
) -> None:
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    columns = _tmem_columns(body)
    resolved = _resolved_axes(program, bound, block_sizes)
    for a, b in _unrolled_live_pairs(program):
        if a in facts.tcgen05.tmem_slots and b in facts.tcgen05.tmem_slots:
            wa = tc._align(resolved[facts.frags[a][1]], tc.TCGEN05_SLOT_ALIGN)
            wb = tc._align(resolved[facts.frags[b][1]], tc.TCGEN05_SLOT_ALIGN)
            assert columns[a] + wa <= columns[b] or columns[b] + wb <= columns[a], (
                a,
                b,
                columns,
            )


def test_values_read_inside_a_nested_loop_stay_live_across_it() -> None:
    # dattn (the root carry, cast for the inner loop's dot) is read by every
    # iteration of the tile_d loop, but the planner's linear liveness ended it
    # at that read, so z (a fresh dot of the tile_d loop after the inner loop)
    # shared its columns -- from the second tile_d iteration on, the inner loop
    # read z as dattn.  The outermost loop containing a read but not the
    # definition now holds the value live: dattn conflicts with everything that
    # loop touches.
    bound = _bind_h1()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    (dattn,) = [
        carry.slot
        for loop in program.loops
        if loop.parent < 0
        for carry in loop.carries
    ]
    (z,) = [
        dot.slot
        for dot in program.dots
        if dot.loop >= 0
        and program.loops[dot.loop].parent < 0
        and dot.a.kind == "load"
        and dot.acc_slot is None
        and dot.loop != 0
    ]
    assert (min(dattn, z), max(dattn, z)) in facts.tcgen05.tmem_conflicts
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    columns = _tmem_columns(body)
    assert columns[dattn] != columns[z]
    _assert_live_slots_disjoint(bound, program, body, [64, 64, 64])


def test_mixed_dtype_passes_use_each_tiles_dtype() -> None:
    # With a tf32 dot first, the program-wide kind picked unpack_f16x2 for the
    # bf16 tile rewritten in place, and the exchange packs defaulted to the
    # program dtype.  The passes now use the tile's own dtype.
    bound = _bind_h4()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None and facts.tcgen05.tf32
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    assert f"{WARP_RT}.unpack_bf16x2(" in body
    assert f"{WARP_RT}.unpack_f16x2(" not in body
    assert f"{WARP_RT}.pack_bf16x2(" in body and f"{WARP_RT}.pack_f16x2(" not in body
    assert f"{P}.Tcgen05MMAKind.TF32" in body and f"{P}.Tcgen05MMAKind.F16" in body


def test_a_derived_value_read_behind_a_nested_loop_crosses_segments() -> None:
    # ht2 is computed before the inner dot loop and reduced after it, in the
    # same tile_d loop; the blocks are built per segment, so its registers do
    # not reach the reduction -- the value takes tensor-memory columns (the
    # render used to hit the emit_ld assertion).
    bound = _bind_h3()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    (ht2,) = [step for step in program.derived if not step.frags]
    assert regions.derived_crosses(program, ht2)
    assert ht2.slot in facts.tcgen05.tmem_slots
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    assert body.count(f"{P}.tcgen05_st(") >= 1
    _assert_live_slots_disjoint(bound, program, body, [64, 64])


def test_region_programs_own_the_carry_slot_rule() -> None:
    # The single-loop emitter declines a loop output computed in another
    # slot (no TMEM carry copy); the region renderer
    # copies the output into the carry's slot (wy's Apow), so region
    # programs dispatch before that rule.
    spec = _bind_wy().config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
    program = _match(_bind_wy())
    assert any(
        program.frag_slots[program.resolve(carry.output)] != carry.slot
        for loop in program.loops
        for carry in loop.carries
    )


def test_tile_only_derived_operands_take_tensor_memory_slots() -> None:
    # bwd_o_kda's ``qg = (q * exp2(gc)).to(bf16)`` is a derived value over
    # tiles alone that the inner loop's dot reads as an operand (through the
    # exchange tile) -- it needs its own columns: at the first seed it took
    # dqg's, and the inner loop read dqg as qg.  Its columns are now shared
    # only with dA (the first loop's carry, dead by then).
    bound = _bind_bwd_o_kda()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    (dh_dot,) = [dot for dot in program.dots if dot.a.frag is not None]
    assert dh_dot.a.frag is not None
    qg = program.frag_slots[dh_dot.a.frag]
    (derived,) = [step for step in program.derived if step.slot == qg]
    assert not derived.frags and regions.derived_crosses(program, derived)
    assert qg in facts.tcgen05.tmem_slots
    (dqg,) = [
        carry.slot
        for loop in program.loops
        if loop.parent >= 0
        for carry in loop.carries
    ]
    (da,) = [
        carry.slot
        for loop in program.loops
        if loop.parent < 0
        for carry in loop.carries
    ]
    body = _kernel_body(_source(bound, _config(64, 64, 16, warps=4, stages=4)))
    columns = _tmem_columns(body)
    assert columns[qg] == columns[da] == 0
    assert columns[dqg] == 64 and columns[dh_dot.slot] == 128
    assert f"{P}.tcgen05_alloc(md_tslot, 256," in body
    # state_du_kda's ``k_scaled`` and the anchored output's ``qg`` likewise.
    for bind in (_bind_state_du_kda, _bind_fwd_o_anch):
        bound = bind()
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        program = _match(bound)
        operands = {
            program.frag_slots[operand.frag]
            for dot in program.dots
            for operand in (dot.a, dot.b)
            if operand.frag is not None
        }
        tile_only = {step.slot for step in program.derived if not step.frags}
        assert operands & tile_only
        assert (operands & tile_only) <= set(facts.tcgen05.tmem_slots)


def test_a_dots_only_batch_waits_for_the_previous_commit() -> None:
    # The anchored output's ``o_intra = dot(A, v)`` follows the ``o_cross``
    # loop as a batch without epilogue work ahead of it; it used to commit
    # while the loop's last commit was outstanding, and the parity wait passed
    # on the stale phase.
    bound = _bind_fwd_o_anch()
    body = _kernel_body(_source(bound, _config(1, 16, 16, warps=4, stages=4)))
    assert _commits_while_one_is_outstanding(body) == []
    epilogue = _after_loop(body, "md_i")
    assert epilogue.count(f"{P}.tcgen05_commit(md_cb") == 1
    _assert_in_order(
        epilogue,
        [
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            "cute.arch.barrier()",
            f"{P}.tcgen05_mma(",
            f"{P}.tcgen05_commit(md_cb",
            "md_cn = md_cn + 1",
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
        ],
    )


def test_the_outer_loop_staging_waits_for_the_sibling_loop() -> None:
    # dqkw's tile_d loop stages k / q tiles into a frame that overlaps the
    # dattn loop's ring; the first iteration's TMA used to be issued while the
    # dattn loop's last batch could still be reading it.
    bound = _bind_dqkw()
    body = _kernel_body(_source(bound, _config(64, 64, 16, warps=4, stages=2)))
    assert _commits_while_one_is_outstanding(body) == []
    _before, outer = _split_loop(body, "md_i1")
    head = outer[: outer.index(f"{P}.mbarrier_arrive_expect_tx(md_sf1, ")]
    assert head.startswith("for md_i1 in cutlass.range(0, 2):")
    assert f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)" in head
    assert f"{P}.tcgen05_commit(" not in head
    assert "cp_async_bulk_tensor" not in head


def test_gated_dstate_computes_its_decay_tables_one_iteration_ahead() -> None:
    # The [C, C] ``exp2((gc[:, None] - gc[None, :]) * RCP_LN2)`` of the attn
    # exchange (dv_i's A operand) and the dv_kh row factor depend on the
    # iteration's gc row alone: the entries this thread evaluates (32 at
    # its row, swapped, plus the row factor) are registers computed for the
    # next iteration under this iteration's MMAs, in the windows after the
    # batch whose block consumed them (the exchange sits in block 1: after
    # the second and third commits, before their waits), from gc elements
    # loaded two iterations ahead.
    body = _kernel_body(
        _source(_bind_dstate(gated=True), _config(32, warps=4, stages=2))
    )
    prologue, loop = body.split("for md_i in cutlass.range(", 1)
    loop = loop.split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)[0]
    table = re.compile(r"md_tb0_\d+ = cutlass\.Float32\(md_v")
    row_factor = re.compile(r"md_tb1_\d+ = cutlass\.Float32\(md_v")
    load = re.compile(r"md_tl\d+ = _helion_mma_dag\.ldg_f32\(")
    # Prologue: the first iteration's tables from its loads, then the
    # second iteration's loads (dl, the row's gc, 32 column gc values).
    assert len(table.findall(prologue)) == 32
    assert len(row_factor.findall(prologue)) == 1
    assert len(load.findall(prologue)) == 2 * 34
    assert prologue.rindex("md_tl0 = ") > prologue.rindex("md_tb0_31 = ")
    segments = loop.split(f"{P}.tcgen05_commit(")
    assert len(segments) == 4
    assert [len(table.findall(s)) for s in segments] == [0, 0, 16, 16]
    assert [len(row_factor.findall(s)) for s in segments] == [0, 0, 0, 1]
    assert [len(load.findall(s)) for s in segments] == [0, 0, 0, 34]
    # Under the MMAs: before the wait that follows each commit.
    assert segments[2].index("md_tb0_") < segments[2].index(
        f"{RT}.mbar_wait(md_stepb, md_i * 2 + 1 & 1)"
    )
    assert segments[3].rindex("md_tl33 = ") < segments[3].index(
        f"{RT}.mbar_wait(md_accb, md_parity)"
    )
    assert "if md_i + 2 < 8:" in segments[3]
    assert segments[3].index("if md_i + 2 < 8:") > segments[3].rindex("md_tb1_")
    # The exchange multiplies by the registers and evaluates no exp2 (the
    # two in its segment are the dog pass's row factors, run there since
    # the pass moved under the first batch).
    assert re.search(r"\* md_tb0_0\b", segments[1])
    assert re.search(r"\* md_tb0_31\b", segments[1])
    assert segments[1].count("exp2(") == 2
    assert segments[1].index("exp2(") < segments[1].index(f"{RT}.mbar_wait(md_stepb")
    # 33 entries, plus the dS rescale's scalar in block 2.
    assert sum(s.count("exp2(") for s in segments[2:]) == 33 + 1
    # Nothing else in the loop reads the vectors the tables hold.
    for other in (_bind_dstate(), _bind_fwd_h_delta(gated=True)):
        assert "md_tb" not in _source(other, _config(32, warps=4, stages=2))


def _store_after(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A body store after the iteration's only dot: the trailing block."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
    return h_all


def _snap_reread(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Snapshots per iteration; the epilogue re-reads row 0 of the last one
    (written by the last iteration's store) as a side vector."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=torch.float32, device=k.device)
    out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    ht = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
        ht[idx, tile_d, tile_dv] = h_acc
        row = h_all[idx, N - 1, 0, tile_dv]
        out[idx, tile_d, tile_dv] = h_acc + row[None, :]
    return h_all, out, ht


def _snap_same_dst(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A loop-invariant snapshot (the same tile every iteration), then the
    epilogue overwrites that tile with the final state: its write must be
    the last."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    ht = torch.empty([BH, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            ht[idx, tile_d, tile_dv] = h_acc.to(ht.dtype)
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
        ht[idx, tile_d, tile_dv] = h_acc.to(ht.dtype)
    return ht


def test_tma_stores_decline_tensors_the_program_stores_elsewhere() -> None:
    # A bulk store completes at the kernel's end only (the in-loop waits order
    # its staging reads, not the global writes).  A program that also LOADS the
    # snapshot tensor is not the lowering's at all (``_snap_reread``); a second
    # store to the tensor -- the epilogue storing the same tile
    # (``_snap_same_dst``) -- is admitted and keeps the packet drain.
    t = _tensors()
    assert _bind_fn(_snap_reread, (t["k"], t["v"])).config_spec.cute_mma_dag is None
    bound = _bind_fn(_snap_same_dst, (t["k"], t["v"]))
    source = _source(bound, _config(64, 16, warps=4, stages=2))
    body = _kernel_body(source)
    assert f"{P}.cp_async_bulk_tensor_global_shared_cta(" not in body
    assert f"{P}.cp_async_bulk_wait_group(" not in body
    assert f"{WARP_RT}.stg_v4_b32(" in _loop(body)
    assert "'tma_count': 2" in source
    # The snapshot of a tensor nothing else touches still goes by TMA.
    body = _kernel_body(_source(_bind_fwd(), _config(64, 16, warps=4, stages=2)))
    assert body.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1


def test_trailing_tma_stores_wait_in_the_loop() -> None:
    # A body store behind the iteration's last dot issues its group behind the
    # trailing barrier; when no batch of the loop issues a group there was no
    # wait before the stage-in two iterations later rewrote the buffer.  Thread
    # 0 now waits for the previous iteration's group right after its commit
    # (ahead of the next iteration's accb commit that the stage-in waits on).
    t = _tensors()
    bound = _bind_fn(_store_after, (t["k"], t["v"]))
    for cols in (16, 32, 64):
        body = _kernel_body(_source(bound, _config(64, cols, warps=4, stages=2)))
        loop = _loop(body)
        assert loop.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1
        _assert_in_order(
            loop,
            [
                f"{P}.tcgen05_commit(md_accb",
                "cute.arch.barrier()",
                "if md_tidx == 0:",
                f"{P}.cp_async_bulk_tensor_global_shared_cta(",
                f"{P}.cp_async_bulk_commit_group()",
                f"{P}.cp_async_bulk_wait_group(1, read=True)",
                f"{P}.cp_async_bulk_wait_group(0)",
            ],
        )
        assert loop.count(f"{P}.cp_async_bulk_wait_group(1, read=True)") == 1


def test_passes_no_first_batch_dot_reads_run_under_the_first_batch() -> None:
    # The gated fwd_h_delta k pass feeds dot2 only: it runs after the first
    # batch's issue (under those MMAs), before its wait, published by the
    # next block's proxy fence; warp 0 then waits the ring slot itself
    # before the first batch (the pass no longer did).  The ungated kernel
    # has no pass and is unchanged; dstate's do_i and dog passes (dv_i's B,
    # dS's B: batches 1 and 2) run together after the first batch.
    body = _kernel_body(
        _source(_bind_fwd_h_delta(gated=True), _config(32, warps=4, stages=4))
    )
    loop = _loop(body).split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)[0]
    segments = loop.split(f"{P}.tcgen05_commit(")
    assert len(segments) == 3
    assert [s.count(f"{WARP_RT}.lds_v4_b32(") for s in segments] == [0, 8, 0]
    assert [s.count(f"{WARP_RT}.sts_v4_b32(md_pa") for s in segments] == [0, 8, 0]
    assert [
        s.count(f"{RT}.mbar_wait(md_full + md_stage, md_round)") for s in segments
    ] == [1, 1, 0]
    _assert_in_order(
        segments[0],
        [
            "if md_warp == 0:",
            f"{RT}.mbar_wait(md_full + md_stage, md_round)",
            f"{P}.tcgen05_mma(",
        ],
    )
    _assert_in_order(
        segments[1],
        [
            f"{RT}.mbar_wait(md_full + md_stage, md_round)",
            f"{WARP_RT}.lds_v4_b32(",
            f"{WARP_RT}.sts_v4_b32(md_pa",
            f"{RT}.mbar_wait(md_stepb, md_i * 1 + 0 & 1)",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            "cute.arch.barrier()",
            f"{P}.tcgen05_mma(",
        ],
    )
    # One proxy fence per block (the exchange tiles' and the pass's are the
    # same fence): none between the pass and the step wait.
    pass_part = segments[1].split(f"{RT}.mbar_wait(md_stepb", 1)[0]
    assert "fence_proxy" not in pass_part
    body = _kernel_body(
        _source(_bind_dstate(gated=True), _config(32, warps=4, stages=2))
    )
    loop = _loop(body).split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)[0]
    segments = loop.split(f"{P}.tcgen05_commit(")
    assert [s.count(f"{WARP_RT}.lds_v4_b32(") for s in segments] == [0, 4, 0, 0]
    assert [
        s.count(f"{RT}.mbar_wait(md_full + md_stage, md_round)") for s in segments
    ] == [1, 1, 0, 0]


def _snap_reread_view(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """snap_reread through a host-side view of the snapshot tensor (another
    argument, the same storage)."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=torch.float32, device=k.device)
    h_last = h_all[:, N - 1]
    out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    ht = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
        ht[idx, tile_d, tile_dv] = h_acc
        row = h_last[idx, 0, tile_dv]
        out[idx, tile_d, tile_dv] = h_acc + row[None, :]
    return h_all, out, ht


def _snap_reread2(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The epilogue adds the whole last snapshot tile (a 2-D tile input of
    an epilogue derived value) to the final state."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=torch.float32, device=k.device)
    out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    ht = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
        ht[idx, tile_d, tile_dv] = h_acc
        out[idx, tile_d, tile_dv] = h_acc + h_all[idx, N - 1, tile_d, tile_dv]
    return h_all, out, ht


def test_alias_checks_see_through_host_views_and_epilogue_derived_values_decline() -> (
    None
):
    # A host-side view of the snapshot tensor (``h_last = h_all[:, N - 1]``)
    # slipped through the name-keyed alias checks, and an epilogue derived
    # value's tile input was prefetched ahead of the last snapshot store.  Both
    # probes load a tensor the loop stores, so the matcher declines them
    # outright (the storage-keyed checks and the epilogue-derived decline stay
    # for the programs it admits).  The loop's derived values keep their top
    # prefetch.
    t = _tensors()
    for fn in (_snap_reread_view, _snap_reread2):
        assert _bind_fn(fn, (t["k"], t["v"])).config_spec.cute_mma_dag is None
    body = _kernel_body(_source(_bind_fwd_h_delta(), _config(32, warps=4, stages=4)))
    loop, epilogue = _loop(body).split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)
    assert "md_u = " in loop and f"{WARP_RT}.ldg_b32(" in loop


def test_pass_warps_rewrite_the_next_tile_off_the_chain() -> None:
    # ``cute_mma_dag_pass_warps='on'`` (eight warps, stages >= 2, loop
    # operand passes): warps 4-7 run their own loop over the tiles, each
    # rewritten in its ring slot once the slot's previous MMAs completed
    # (the slot's free barrier, committed by those MMAs and arrived once per
    # slot in the prologue) and its TMA landed (a suspended wait), published
    # to the async proxy and signalled on the slot's pass barrier (128
    # arrivals); the epilogue warps run the chain on a named barrier,
    # without passes, and warp 0 waits the pass barrier ahead of the first
    # batch reading a passed tile.  The stores take the four-warp path (TMA
    # for the admitted snapshots).
    bound = _bind_fwd_h_delta(gated=True)
    source = _source(
        bound, _config(32, warps=8, stages=4, cute_mma_dag_pass_warps="on")
    )
    body = _kernel_body(source)
    assert body.count(f"{P}.mbarrier_init(md_pass + ") == 4
    assert f"{P}.mbarrier_init(md_pass + 3, 128)" in body
    assert body.count(f"{P}.mbarrier_init(md_free + ") == 4
    assert f"{P}.mbarrier_init(md_free + 3, 1)" in body
    prologue, rest = body.split("if md_warp >= 4:", 1)
    assert prologue.count(f"{P}.mbarrier_arrive(md_free + ") == 4
    assert prologue.index("fence_mbarrier_init()") < prologue.index(
        "mbarrier_arrive(md_free + 0)"
    )
    pass_loop, epilogue_loop = rest.split("\n    else:\n", 1)
    assert pass_loop.count("for md_j in cutlass.range(0, 8):") == 1
    _assert_in_order(
        pass_loop,
        [
            "md_stage = md_j % 4",
            f"{RT}.mbar_wait_sleep(md_free + md_stage, md_round)",
            f"{RT}.mbar_wait_sleep(md_full + md_stage, md_round)",
            f"{WARP_RT}.lds_v4_b32(",
            f"{WARP_RT}.sts_v4_b32(md_pa",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            f"{P}.mbarrier_arrive(md_pass + md_stage)",
        ],
    )
    assert pass_loop.count(f"{WARP_RT}.lds_v4_b32(") == 8
    assert "md_pr_t" in pass_loop and "(md_tidx - 128)" in pass_loop
    assert "cute.arch.barrier" not in pass_loop
    loop = epilogue_loop.split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)[0]
    assert loop.count("for md_i in cutlass.range(0, 8):") == 1
    assert f"{WARP_RT}.lds_v4_b32(" not in loop and "md_pr_t" not in loop
    assert loop.count("cute.arch.barrier(barrier_id=1, number_of_threads=128)") == 2
    assert "cute.arch.barrier()" not in loop
    # The k pass feeds dot2: warp 0 waits the pass before the second batch,
    # the ring slot itself before the first (no pass waited it); the
    # iteration's last commit (acc_ready) is followed by the slot's free
    # commit under the same elect.
    segments = loop.split(f"{P}.tcgen05_commit(")
    assert len(segments) == 4
    assert segments[2].startswith("md_accb") and segments[3].startswith(
        "md_free + md_stage"
    )
    assert [
        s.count(f"{RT}.mbar_wait(md_pass + md_stage, md_round)") for s in segments
    ] == [0, 1, 0, 0]
    assert [
        s.count(f"{RT}.mbar_wait(md_full + md_stage, md_round)") for s in segments
    ] == [1, 0, 0, 0]
    assert loop.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 2
    assert "if md_warp >= 4:" not in loop
    # After the loop every warp meets at the CTA barriers again.
    epilogue = epilogue_loop.split(f"{RT}.mbar_wait(md_accb, 8 - 0 - 1 & 1)", 1)[1]
    assert "cute.arch.barrier()" in epilogue
    # Without the knob the eight-warp render keeps the store-warp drains and
    # the passes on the chain.
    body = _kernel_body(_source(bound, _config(32, warps=8, stages=4)))
    assert "md_pass" not in body and "md_free" not in body
    assert "barrier_id=1" not in body
    assert "for md_j in" not in body and "if md_warp >= 4:" in body


def test_pass_warps_need_a_subject() -> None:
    # Pinned off when there is no loop pass (delta fwd_h_delta), fewer than
    # eight warps, one ring slot, the warp mode or a region program.
    cases = (
        (lambda: _bind_fwd_h_delta(), _config(32, warps=8, stages=4)),
        (lambda: _bind_fwd_h_delta(gated=True), _config(32, warps=4, stages=4)),
        (lambda: _bind_fwd_h_delta(gated=True), _config(32, warps=8, stages=1)),
        (lambda: _bind_wy(gated=True), _config(64, 64, 64, warps=8, stages=2)),
    )
    for bind, config in cases:
        bound = bind()
        config = dict(config.config, cute_mma_dag_pass_warps="on")
        with bound.env:
            bound.config_spec.normalize(config)
        assert config["cute_mma_dag_pass_warps"] == "off", config
    bound = _bind_fwd_h_delta(gated=True)
    config = dict(
        _config(32, warps=8, stages=4).config,
        cute_mma_dag="warp",
        cute_mma_dag_pass_warps="on",
    )
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_pass_warps"] == "off"
    config = dict(_config(32, warps=8, stages=2).config, cute_mma_dag_pass_warps="on")
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_pass_warps"] == "on"
    # dstate's do_i / dog passes and the scalar bwd's q pass qualify too.
    for bind, config in (
        (lambda: _bind_dstate(gated=True), _config(32, warps=8, stages=2)),
        (lambda: _bind_bwd("scalar"), _config(64, 64, warps=8, stages=2)),
    ):
        bound = bind()
        config = dict(config.config, cute_mma_dag_pass_warps="on")
        with bound.env:
            bound.config_spec.normalize(config)
        assert config["cute_mma_dag_pass_warps"] == "on", config
        body = _kernel_body(_source(bound, helion.Config(**config)))
        assert body.count("for md_j in") == 1 and "mbarrier_arrive(md_pass" in body


def test_the_pass_warp_seed_leads_where_its_passes_pay() -> None:
    # Programs with loop operand passes get the first tensor-memory seed
    # again with warps 4-7 as pass warps and a ring of at least three
    # slots.  It leads -- the promoted default -- for a one-wave grid whose
    # passes are worth MMA_DAG_PASS_WARP_PACKETS per thread and iteration
    # (gated fwd_h_delta's k pass: 8 packets); the small passes of gated
    # dstate (2 + 2) and the scalar bwd (4), and the CTA-rich dqk / dv,
    # keep the four-warp default with the pass-warp seed second.  Programs
    # without passes (delta fwd_h_delta) and region programs (wy) seed none.
    spec = _bind_fwd_h_delta(gated=True).config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    default = spec.default_config().config
    sizes = {
        axis.block_id: default["block_sizes"][
            spec.block_sizes.block_id_to_index(axis.block_id)
        ]
        for axis in facts.axes
        if axis.block_id in spec.block_sizes.valid_block_ids()
    }
    resolved = [facts.block_size(i, sizes) for i in range(len(facts.axes))]
    assert dag.mma_dag_pass_packets(facts, resolved) == 8
    assert dag.mma_dag_pass_warps_lead(facts, resolved)
    seeds = [c.config for c in spec.compiler_seed_configs]
    assert default["cute_mma_dag"] == "tcgen05"
    assert default["cute_mma_dag_pass_warps"] == "on"
    assert default["cute_mma_dag_warps"] == 8 and default["cute_mma_dag_stages"] >= 3
    assert {key: default[key] for key in seeds[0]} == seeds[0]
    assert seeds[1]["cute_mma_dag_pass_warps"] == "off"
    assert seeds[1]["block_sizes"] == default["block_sizes"]
    assert sum(c["cute_mma_dag_pass_warps"] == "on" for c in seeds) == 1
    # The one-wave condition counts the root tiles' CTAs too: the same
    # kernel over BH = 64 heads x DV / 64 = 2 tiles (128 CTAs) leads, over
    # BH = 128 (256 CTAs, two waves on 148 SMs) it does not.
    for bh, leads in ((64, True), (128, False)):
        spec = _bind_fwd_h_delta(gated=True, bh=bh, dv=128).config_spec
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.grid_blocks == bh
        default = spec.default_config().config
        sizes = {
            axis.block_id: default["block_sizes"][
                spec.block_sizes.block_id_to_index(axis.block_id)
            ]
            for axis in facts.axes
            if axis.block_id in spec.block_sizes.valid_block_ids()
        }
        resolved = [facts.block_size(i, sizes) for i in range(len(facts.axes))]
        assert default["block_sizes"] == [64]
        assert dag.mma_dag_cta_count(facts, resolved) == bh * 2
        assert dag.mma_dag_pass_packets(facts, resolved) == 8
        assert dag.mma_dag_pass_warps_lead(facts, resolved) is leads
        assert (default["cute_mma_dag_pass_warps"] == "on") is leads
    spec = _bind_fwd_h_delta().config_spec
    assert all(
        c.config["cute_mma_dag_pass_warps"] == "off" for c in spec.compiler_seed_configs
    )
    for bind, packets in (
        (lambda: _bind_dstate(gated=True), 2),
        (lambda: _bind_bwd("scalar"), 4),
        (_bind_dqk, 4),
        (_bind_dv, 4),
    ):
        bound = bind()
        spec = bound.config_spec
        facts = spec.cute_mma_dag_facts
        assert facts is not None
        default = spec.default_config().config
        sizes = {
            axis.block_id: default["block_sizes"][
                spec.block_sizes.block_id_to_index(axis.block_id)
            ]
            for axis in facts.axes
            if axis.block_id in spec.block_sizes.valid_block_ids()
        }
        resolved = [facts.block_size(i, sizes) for i in range(len(facts.axes))]
        assert dag.mma_dag_pass_packets(facts, resolved) == packets
        assert not dag.mma_dag_pass_warps_lead(facts, resolved)
        assert default["cute_mma_dag_pass_warps"] == "off"
        seeds = [c.config for c in spec.compiler_seed_configs]
        assert seeds[1]["cute_mma_dag_pass_warps"] == "on", seeds[1]
        assert seeds[1]["cute_mma_dag_warps"] == 8
        # The seed renders (its ring fits the shared memory).
        assert "for md_j in" in _kernel_body(_source(bound, helion.Config(**seeds[1])))
    # Region programs (wy's tf32 chain) seed none.
    assert all(
        c.config["cute_mma_dag_pass_warps"] == "off"
        for c in _bind_wy().config_spec.compiler_seed_configs
    )


def test_atom_32b_exchange_writes_rotate_the_packet_order() -> None:
    # Under the 32-byte-unit swizzle a packet column of a warp's 16 rows
    # lands on four of the eight bank groups (the XOR runs over the row mod
    # 4): the lanes of rows 4-7 of each eight write the next packet instead
    # (adjacent packets differ in parity, hence in bank-group half), chosen
    # by selects over the row's values -- same bytes at the same addresses,
    # every store instruction over all eight groups.  The SW128 A tiles are
    # conflict-free as they are and keep the plain order.
    body = _kernel_body(_source(_bind_wy(), _config(64, 64, 64, warps=4, stages=2)))
    first, rest = _split_loop(body, "md_i")
    kk_loop, rest = _split_loop(rest, "md_i1")
    chain, rest = _split_loop(rest, "md_i2")
    assert chain.count("md_rot") > 0
    assert re.search(r"md_rot(_\d+)? = md_lane // 4 % 2", chain)
    # Per chain step: two B tiles (Apow, T) of 8 packets, 4 selects each.
    assert len(re.findall(r"if md_rot(_\d+)? else", chain)) == 2 * 8 * 4
    rotated = re.findall(
        r"md_cb\d // 4 \+ \((\d) \+ md_rot(?:_\d+)?\) % 8\) % 8 \^ md_row64 % 4 \* 2",
        chain,
    )
    assert sorted(rotated) == sorted([str(j) for j in range(8)] * 2)
    # The A tiles (SW128) keep the plain packet order.
    assert len(re.findall(r"md_cb\d // 4 \+ \d\) % 8 \^ md_row64 % 8\)", chain)) == 24
    assert not re.search(r"md_rot(_\d+)?\) % 8\) % 8 \^ md_row64 % 8", chain)


def _rmw_scalar_first(
    scale: float,
    k: torch.Tensor,
    v: torch.Tensor,
    s_in: torch.Tensor,
    s_out: torch.Tensor,
) -> torch.Tensor:
    """A vector store with its reduction and side load inside the
    recurrence."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            s_out[idx, tile_d] = s_in[idx, tile_d] + h_acc.sum(-1) * scale
        h_out[idx, tile_d, tile_dv] = h_acc
    return h_out


def test_vector_values_in_the_loop_decline_the_single_loop_emitter() -> None:
    # The single-loop emitter rendered ``s_out[idx, tile_d] = s_in[...] +
    # h_acc.sum(-1) * scale`` inside the recurrence without the reduction, the
    # load or the store (s_out stayed zero while h_out was right) and tcgen05
    # was the kernel's default.  Vector values are the region renderer's: such
    # programs decline the mode (the warp mode renders them).
    t = _tensors()
    s_in = torch.empty((2, 64), dtype=torch.float32, device=CPU_DEVICE)
    s_out = torch.empty((2, 64), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind_fn(_rmw_scalar_first, (0.5, t["k"], t["v"], s_in, s_out))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is None
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None and program.simple and program.vec_stores
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    assert spec.default_config().config["cute_mma_dag"] == "warp"
    body = _kernel_body(
        _source(bound, helion.Config(block_sizes=[64, 64], cute_mma_dag="warp"))
    )
    assert "s_out" in body and "stg_f32" in body or "s_out" in body


def test_mn_major_tf32_tiles_need_32_columns_at_config_time() -> None:
    # The 128-byte-row rule for MN-major tf32 tiles lived in the planner only,
    # so the fp32 GLA kernel's default and every tcgen05 seed ([1, 16, 16])
    # raised in to_code.  ``region_config_reason`` carries it now:
    # normalization refuses or repairs such tiles, the seeds and the default
    # keep 32 fp32 columns at least, and they render.
    bound = _bind_fwd_o_anch_f32()
    spec = bound.config_spec
    with bound.env, pytest.raises(InvalidConfig, match="128-byte rows .32 columns."):
        spec.normalize(_config(1, 16, 16, warps=4, stages=2).config)
    with bound.env:
        spec.normalize(_config(1, 32, 32, warps=4, stages=2).config)
    # block_sizes = [bhn, dv (the B tile's N), d (the K loop tile)].
    default = spec.default_config().config
    assert default["cute_mma_dag"] == "tcgen05" and default["block_sizes"][1] >= 32
    assert "tcgen05_mma(" in _source(bound, helion.Config(**default))
    seeds = [
        c.config
        for c in spec.compiler_seed_configs
        if c.config["cute_mma_dag"] == "tcgen05"
    ]
    assert seeds and all(c["block_sizes"][1] >= 32 for c in seeds)
    config = _config(1, 16, 16, warps=4, stages=2).config
    with bound.env:
        spec.normalize(config, _fix_invalid=True)
    assert config["cute_mma_dag"] != "tcgen05" or config["block_sizes"][1] >= 32


_SIMPLE_BINDERS: dict[str, Callable[[], BoundKernel]] = {
    "fwd": _bind_fwd,
    "fwd_scalar": lambda: _bind_fwd("scalar"),
    "fwd_diag_h0": lambda: _bind_fwd("diag", has_h0=True),
    "bwd": _bind_bwd,
    "bwd_scalar": lambda: _bind_bwd("scalar"),
    "fwd_o": _bind_fwd_o,
    "fwd_o_gated": lambda: _bind_fwd_o(gated=True),
    "dqk": _bind_dqk,
    "dqk_gated": lambda: _bind_dqk(gated=True),
    "dv": _bind_dv,
    "dv_scalar": lambda: _bind_dv("scalar"),
    "fwd_h_delta": _bind_fwd_h_delta,
    "fwd_h_delta_gated": lambda: _bind_fwd_h_delta(gated=True),
    "dstate": _bind_dstate,
    "dstate_gated": lambda: _bind_dstate(gated=True),
    "fwd_o_anch_f32": _bind_fwd_o_anch_f32,
}


_RENDER_BINDERS: dict[str, Callable[[], BoundKernel]] = {
    **_SIMPLE_BINDERS,
    **_REGION_BINDERS,
}


@pytest.mark.parametrize("which", ("default", "first_seed"))
@pytest.mark.parametrize("name", sorted(_RENDER_BINDERS))
def test_every_admitted_default_and_first_seed_renders(name: str, which: str) -> None:
    # A config normalization admits must compile: the default config and
    # the first tcgen05 seed of every binder that admits the mode render
    # (the fp32 GLA default did not): one case per (binder, config), so a
    # failure names the binder and no case is long.
    bound = _RENDER_BINDERS[name]()
    spec = bound.config_spec
    if spec.cute_mma_dag is None or "tcgen05" not in spec.cute_mma_dag.choices:
        pytest.skip(f"{name} does not admit tcgen05")
    default = spec.default_config().config
    if which == "default":
        config = default
    else:
        seeds = [
            c.config
            for c in spec.compiler_seed_configs
            if c.config["cute_mma_dag"] == "tcgen05"
        ]
        if not seeds:
            pytest.skip(f"{name} has no tcgen05 seed")
        if seeds[0] == default:
            pytest.skip(f"{name}: the first tcgen05 seed is the default config")
        config = seeds[0]
    with bound.env:
        spec.normalize(config)
    source = _source(bound, helion.Config(**config))
    assert "@cute.kernel" in source, (name, which)


def _bind_fwd_scalar_unaligned() -> BoundKernel:
    """The scalar-decay forward with k and v sliced [..., 4:68] out of
    80-wide rows: their passes copy from global memory (the sync route)."""
    t = _tensors()
    k = torch.empty((2, 8, 64, 80), dtype=torch.bfloat16, device=CPU_DEVICE)[..., 4:68]
    v = torch.empty((2, 8, 64, 80), dtype=torch.bfloat16, device=CPU_DEVICE)[..., 4:68]
    return _bind(
        "chunk_fwd_h_diag_fused",
        (k, v, t["g_last_s"], t["h0"], t["gc"], True, True, False, True),
    )


def test_pass_warps_pace_sync_route_pass_tiles_on_the_free_barrier() -> None:
    # The pass warps paced themselves on the ring's full barrier alone; a pass
    # tile copied from global memory (an unaligned source) never arms it, so
    # they overran the slot and the pass barrier's phases (0/10 launches
    # right).  Every slot now has a free barrier the slot's MMAs commit to
    # (pre-arrived once per slot in the prologue): the pass warps wait it
    # before every tile, whatever fills the slot, and the full barrier only
    # when a TMA tile does.
    bound = _bind_fwd_scalar_unaligned()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None and facts.loop_passes > 0
    config = dict(
        _config(64, 64, warps=8, stages=3).config, cute_mma_dag_pass_warps="on"
    )
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_pass_warps"] == "on"
    body = _kernel_body(_source(bound, helion.Config(**config)))
    pass_loop = body.split("if md_warp >= 4:", 1)[1].split("\n    else:\n", 1)[0]
    assert pass_loop.count("for md_j in cutlass.range(0, 8):") == 1
    assert f"{RT}.mbar_wait_sleep(md_free + md_stage, md_round)" in pass_loop
    assert "md_full" not in pass_loop
    assert pass_loop.count(f"{WARP_RT}.ldg_") > 0 and "md_tma" not in pass_loop
    assert pass_loop.index("mbar_wait_sleep(md_free") < pass_loop.index(
        f"{WARP_RT}.ldg_"
    )
    assert pass_loop.count(f"{P}.mbarrier_arrive(md_pass + md_stage)") == 1
    # The ring barriers are never armed: warp 0 waits the pass barrier only.
    loop = body.split("\n    else:\n", 1)[1]
    assert f"{RT}.mbar_wait(md_pass + md_stage, md_round)" in loop
    assert "mbar_wait(md_full" not in loop
    assert loop.count(f"{P}.tcgen05_commit(md_free + md_stage") == 1
