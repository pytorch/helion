"""Codegen, admission and search-surface pins for the ``cute_mma_dag``
tcgen05 compute mode (the TMEM-resident serial state recurrence).

CPU-only (fake sm100 target) like ``test_cute_mma_dag.py``, whose binders
and fixture this file reuses.  Runtime coverage lives in
``test_cute_mma_dag_tcgen05_gpu.py``.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import os
import random
import re
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

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
    from collections.abc import Iterator
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


def _chunk_state_inputs(seqlen: int = 512) -> tuple[torch.Tensor, ...]:
    # examples/mamba2_chunk_state.py: batch 2, 8 heads in 2 groups, chunk
    # 128, headdim 64, dstate 64.
    bf16 = torch.bfloat16
    nchunks = (seqlen + 127) // 128
    return (
        torch.empty((2, seqlen, 2, 64), dtype=bf16, device=CPU_DEVICE),
        torch.empty((2, seqlen, 8, 64), dtype=bf16, device=CPU_DEVICE),
        torch.empty((2, 8, nchunks, 128), dtype=bf16, device=CPU_DEVICE),
        torch.empty((2, 8, nchunks, 128), dtype=bf16, device=CPU_DEVICE),
    )


def test_chunk_offset_tiles_and_lead_tile_begins_are_admitted() -> None:
    # ``x[tile_b.begin, tile_k.index + tile_c.begin * chunk, tile_h.begin,
    # tile_m]``: the leading tiles' begins are scalar coordinates and the
    # chunk offset rides the loop tile, in the TMA coordinates and in the
    # byte bases of the side loads.
    from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel

    bound = _bind_fn(helion_mamba2_chunk_state_kernel.fn, _chunk_state_inputs())
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.default_config().config["cute_mma_dag"] == "tcgen05"
    body = _kernel_body(_source(bound, _config(64, 64, 128, warps=4, stages=2)))
    # The x box at (m, h, k + 128 * c, b) and the B box at (n, k + 128 * c, b).
    assert re.search(
        r"\[cutlass\.Int32\(md_a\d\), cutlass\.Int32\(tile_offset_\d\), "
        r"cutlass\.Int32\(0 \* 128 \+ 128 \* tile_offset_\d\)",
        body,
    ), body
    assert body.count("128 * tile_offset_") >= 4, body
    warp = _kernel_body(
        _source(
            bound,
            helion.Config(
                block_sizes=[64, 64, 64],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            ),
        )
    )
    assert re.search(r"md_base = x\.iterator\.toint\(\) .*128 \* tile_offset_", warp)


def test_an_offset_tile_that_can_leave_the_tensor_is_not_mine() -> None:
    # seqlen 500: the last chunk's tile reads rows 384..511 of a 500-row x,
    # which the masked scalar lowering zero-fills.
    from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel

    bound = _bind_fn(
        helion_mamba2_chunk_state_kernel.fn, _chunk_state_inputs(seqlen=500)
    )
    assert bound.config_spec.cute_mma_dag is None


# --------------------------------------------------------------------------
# Region programs
# --------------------------------------------------------------------------


def _split_loop(body: str, var: str) -> tuple[str, str]:
    """``(before, loop and after)`` at the first ``for var in`` line."""
    index = body.index(f"for {var} in cutlass.range(")
    return body[:index], body[index:]


def test_region_programs_admit_the_mode_and_lay_out_their_items() -> None:
    # dk, dqkw and wy_dL (both decay modes) admit the tensor-memory mode
    # through the region renderer and lead with a tensor-memory seed (the
    # promoted default: streaming programs, dqkw's loop fragment dattn
    # being a hoisted root exchange); the warp seeds follow.
    for bind, first in (
        (_bind_dk, "tcgen05"),
        (_bind_dqkw, "tcgen05"),
        (lambda: _bind_dqkw(gated=True), "tcgen05"),
        (_bind_wy_dl, "tcgen05"),
        (lambda: _bind_wy_dl(gated=True), "tcgen05"),
    ):
        spec = bind().config_spec
        assert spec.cute_mma_dag is not None
        assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        assert facts.tcgen05.regions
        assert spec.default_config().config["cute_mma_dag"] == first
        modes = [c.config["cute_mma_dag"] for c in spec.compiler_seed_configs]
        assert modes[0] == first and {"warp", "tcgen05"} <= set(modes)
    # dk: both orientations of dL (plain casts of one load) share a hoisted
    # tile (fp32 source -> a raw tile beside it), k rides loop 0's ring with
    # the seed's two tile inputs (dk_as, d_ba: aux tiles), the store stages
    # through two buffers in loop 0's frame; one slot lives in TMEM.
    facts = _bind_dk().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    items = facts.tcgen05.items
    assert [(i.kind, i.frame, i.copies) for i in items] == [
        ("tile", -1, 1),
        ("raw", -1, 1),
        ("tile", 0, 0),
        ("aux", 0, 0),
        ("aux", 0, 0),
        ("staging", 0, 2),
    ]
    assert facts.tcgen05.tmem_slots == (0,)
    assert facts.tcgen05.ring_loops == 1 and facts.tcgen05.staged_regions == 1
    assert facts.tcgen05.aux_items == 2
    # Block ids of dk: loop bd 2, lead 0, rdim C 1 -> axis indices 0..2.
    resolved = [64, 1, 64]
    layout = regions.region_layout(items, facts.loops, resolved, 2)
    assert layout.ring_base == {0: 0} and layout.ring_bytes == {0: 40960}
    # Loop 0's frame first (the ring: k's tile and the two fp32 aux tiles
    # per slot, then the two staging buffers), then the root's hoisted
    # tile; its raw tile (converted in the prologue) overlays the top of
    # the frame.
    assert [i.segment for i in items] == [-1, 0, -1, -1, -1, -1]
    assert layout.offsets == (98304, 81920, 0, 8192, 24576, 81920)
    assert layout.total == 106496
    assert tc.tcgen05_smem_bytes(facts, resolved, 2, True) == 106496 + 8 * 4 + 16
    # Under the ``lane`` loads mode the aux items take no space.
    assert regions.region_smem_bytes(facts, resolved, 2, "lane") == 40960 + 8 * 4 + 16
    lane = regions.region_layout(items, facts.loops, resolved, 2, frozenset({3, 4}))
    assert lane.offsets == (32768, 16384, 0, -1, -1, 16384) and lane.total == 40960
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
    # dW (slot 1) is a register carry (``register_carries``) at every
    # config (no other derived value shares its loop), so it holds no
    # columns in the layout though the facts list its slot.
    assert facts.tcgen05.tmem_slots == (0, 1, 2, 3, 4)
    assert [c.slot for c in facts.tcgen05.register_carries] == [1]
    assert facts.tcgen05.register_carries[0].live == ()
    # At bd = 128 the three tensor-memory slots of the tile_dv loop take
    # 384 columns (dattn's exchange is hoisted ahead of the tile_d loop, so
    # its columns are shared) and the dW derived value (dW, the dot result
    # and its own row) overruns the register budget.
    assert facts.tcgen05.live_groups is not None
    resolved = _resolved_axes(_match(_bind_dqkw()), _bind_dqkw(), [64, 128, 64])
    assert regions.region_tmem_columns(facts, resolved) == 384
    reason = dag.mma_dag_config_reason(
        facts, {0: 1, 1: 64, 3: 128, 4: 64}, 4, 2, "tcgen05"
    )
    assert reason is not None and "derived value holds 384" in reason
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
    # issues the first iteration's k tile and the seed's two fp32 tiles (two
    # 128-byte slabs each), converts into the bf16 UMMA tile (eight packets
    # per thread) behind one wait, then a barrier.
    _assert_in_order(
        prologue,
        [
            f"{P}.mbarrier_arrive_expect_tx(md_sfr, 16384)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + 81920)",
            f"{P}.mbarrier_arrive_expect_tx(md_full0 + md_rk0 % 2, 40960)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + md_rk0 % 2 * 40960 + 0)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + md_rk0 % 2 * 40960 + 8192)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + md_rk0 % 2 * 40960 + 16384)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + md_rk0 % 2 * 40960 + 24576)",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + md_rk0 % 2 * 40960 + 32768)",
            f"{RT}.mbar_wait(md_sfr, 0)",
            f"{RT}.lds_v4_f32(md_s0 + 81920",
            f"{WARP_RT}.sts_v2_b32(md_s0 + 98304",
            "cute.arch.barrier()",
        ],
    )
    assert prologue.count(f"{RT}.lds_v4_f32(") == 8
    assert prologue.count(f"{P}.cp_async_bulk_tensor_shared_cta_global(") == 6
    # Two loads, two aux tiles, the store's map.
    assert "'tma_count': 5" in body
    # Per iteration: the slot / phase / parity from the counter, every
    # thread's wait for the slot, this thread's row of the two seed tiles
    # (eight 16-byte chunks each, swizzled), the refill one iteration ahead,
    # the seed stored to TMEM, both dots in one batch (warp 0 waited the
    # ring already), one commit, the wait, the staged store, the counter.
    _assert_in_order(
        loop,
        [
            "md_st0 = md_rk0 % 2",
            "md_rd0 = md_rk0 // 2 & 1",
            "md_pt0 = md_rk0 % 2",
            f"{RT}.mbar_wait(md_full0 + md_st0, md_rd0)",
            (
                f"{RT}.lds_v4_f32(md_s0 + 0 + md_st0 * 40960 + 8192 + md_row64 * 128 + "
                "((md_cb0 // 4 + 0) % 8 ^ md_row64 % 8) * 16 + (md_cb0 // 4 + 0) // 8 * 8192)"
            ),
            f"{RT}.lds_v4_f32(md_s0 + 0 + md_st0 * 40960 + 24576 + md_row64 * 128",
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            f"{P}.mbarrier_arrive_expect_tx(md_full0 + (md_rk0 + 1) % 2, 40960)",
            "if md_epi:",
            "md_dr_0 = cutlass.Array(cutlass.Float32, 32, alignment=16)",
            f"{P}.tcgen05_st(",
            "cute.arch.barrier()",
            "if md_warp == 0:",
            f"{P}.tcgen05_mma(",
            f"{P}.cp_async_bulk_wait_group(0, read=True)",
            f"{P}.tcgen05_commit(md_cb",
            "md_cn = md_cn + 1",
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            f"{P}.tcgen05_ld(",
            "md_srow = md_s0 + 81920 + md_pt0 * 8192 + md_row64 * 128",
            f"{WARP_RT}.sts_v4_b32(",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            "cute.arch.barrier()",
            f"if md_tidx == {tc.TMA_STORE_THREAD}:",
            (
                f"{P}.cp_async_bulk_tensor_global_shared_cta(md_tma4.get_ptr(), "
                "md_smem + (81920 + md_pt0 * 8192), [cutlass.Int32(md_i * 64), "
                "cutlass.Int32(0), cutlass.Int32(md_a1)])"
            ),
            f"{P}.cp_async_bulk_commit_group()",
            "md_rk0 = md_rk0 + 1",
        ],
    )
    assert f"{WARP_RT}.stg_v4_b32(" not in loop
    assert loop.count(f"{RT}.mbar_wait(md_full0") == 1
    assert loop.count(f"{RT}.lds_v4_f32(") == 16
    assert f"{WARP_RT}.ldg_v2_f32(" not in loop and f"{P}.prefetch_l2(" not in loop
    assert loop.count(f"{P}.tcgen05_mma(") == 8
    assert loop.count(f"{P}.tcgen05_commit(") == 1
    assert loop.count(f"{P}.tcgen05_st(") == 1 and loop.count(f"{P}.tcgen05_ld(") == 1
    # dL as A: K-major straight, MN-major transposed; k is MN-major B.
    assert "n_dim=64, m_dim=64, a_major=0, b_major=1" in prologue
    assert "n_dim=64, m_dim=64, a_major=1, b_major=1" in prologue
    assert "alloc_smem(cutlass.Uint8, 106496, alignment=1024)" in body


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
            f"{P}.cp_async_bulk_tensor_global_shared_cta(",
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
    # 149504 B of operand, exchange, staging and scratch tiles less the 40
    # KB of root items overlaying the loop frames (A_inv's raw tile, the two
    # exchange tiles of the root chain, dL's staging); the reductions read
    # v and k from the operand tiles (no aux tiles).
    assert "alloc_smem(cutlass.Uint8, 108544, alignment=1024)" in body


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
    # dq_cross and dk_state are stored to tensor memory; dW is a register
    # carry (``register_carries``): its zero seed lands in its scalars.
    assert outer.count(f"{P}.tcgen05_st(") == 2
    assert "md_rc1_0_0 = md_seed" in outer
    # The inner iteration: the no-acc dot and the two accumulating dots in
    # one batch (the subtraction reads the first's result, so it follows
    # the commit: ld the result, the difference into dW's scalars; no
    # tensor-memory store, no store wait).
    inner_body = inner[: inner.index("md_rk2 = md_rk2 + 1")]
    # The next iteration's ring refill goes out ahead of this iteration's
    # operand pass (do * scale, in place) -- behind the previous batch's wait,
    # before the pass's own wait, pass, fence and batch barrier -- so the TMA
    # has the whole pass and batch to land in.
    _assert_in_order(
        inner_body,
        [
            f"{RT}.mbar_wait(md_cb, md_cn + 1 & 1)",
            f"{P}.mbarrier_arrive_expect_tx(md_full2 + (md_rk2 + 1) % 2,",
            f"{P}.cp_async_bulk_tensor_shared_cta_global(",
            f"{RT}.mbar_wait(md_full2 + md_st2, md_rd2)",
            f"{WARP_RT}.lds_v4_b32(",
            f"{WARP_RT}.sts_v4_b32(",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            "cute.arch.barrier()",
            f"{P}.tcgen05_mma(",
        ],
    )
    assert inner_body.index(f"{P}.mbarrier_arrive_expect_tx(") < inner_body.index(
        f"{WARP_RT}.lds_v4_b32("
    )
    assert inner_body.count(f"{P}.tcgen05_commit(md_cb") == 1
    assert inner_body.count(f"{P}.tcgen05_mma(") == 12
    assert inner_body.count(f"{P}.tcgen05_st(") == 0
    assert inner_body.count(f"{P}.tcgen05_ld(") == 1
    assert inner_body.index(f"{P}.tcgen05_ld(") > inner_body.index(
        f"{P}.tcgen05_commit(md_cb"
    )
    assert re.search(r"md_rc1_0_0 = md_dr\w*_0\[0\]", inner_body)
    assert f"{P}.Tcgen05Wait.STORE" not in inner_body
    # After the inner loop: the staged tiles are waited, dW is stored, dq
    # and dk_as accumulate onto the carries in one batch with dattn's
    # exchange tile (both orientations from one tile), then their stores.
    after = inner[inner.index("md_rk2 = md_rk2 + 1") :]
    assert f"{RT}.mbar_wait(md_sf1, md_rk1 & 1)" in after
    assert after.count(f"{P}.tcgen05_commit(md_cb") == 1
    # dW_out fp32 (eight packets per thread), dq_out bf16 (four), dk_as_out
    # fp32 (eight).
    # dW_out and dk_as_out (fp32, 256-byte rows) drain as packets, eight
    # per thread each; dq_out (bf16, one 128-byte atom per row) by TMA.
    assert after.count(f"{WARP_RT}.stg_v4_b32(") == 8 + 8
    assert after.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1
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
    # rings -- the raw tiles land on the fp32 tiles' own bytes, so they
    # take no item.
    exchanges = [
        (i.frame, i.elem_bytes) for i in tc_facts.items if i.kind == "exchange"
    ]
    assert exchanges == [(1, 4), (1, 4), (1, 4), (1, 4), (-1, 4)]
    assert [i for i in tc_facts.items if i.kind == "raw"] == []
    assert all(regions.raw_overlays_tile(dot.b) for dot in _match(_bind_wy()).dots[-2:])
    # TMEM: Apow's carry shares kk's columns (its seed, read before the
    # carries are written), the squared Apow takes the carry's own columns
    # (the copy vanishes) and the w / u results reuse them: 128 of the 384
    # a flat layout would take.
    resolved = [64, 64, 64, 1, 64]
    columns, total = regions.region_tmem_layout(facts, resolved)
    assert columns == {0: 0, 1: 0, 2: 64, 3: 0, 4: 0, 5: 0} and total == 128


def test_renders_the_wy_chain_with_tf32_mmas() -> None:
    body = _kernel_body(_source(_bind_wy(), _config(64, 64, 64, warps=4, stages=2)))
    # Loop 1's two exchange phases (32 KB) under the w / u frames (two
    # 16 KB fp32 slots carrying their own raw tiles, 16 KB of staging),
    # T's root tile on top; A_inv's staging overlays the frames.
    assert "alloc_smem(cutlass.Uint8, 65536, alignment=1024)" in body
    assert f"{P}.tcgen05_alloc(md_tslot, 128," in body
    for slot in (1, 3, 4, 5):
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
    # under SW128); the squared Apow lands on the carry's own columns, so
    # nothing copies it back (no tcgen05.st in the chain).
    init = kk_loop[kk_loop.index("md_rk0 = md_rk0 + 1") :]
    assert init.count(f"{P}.tcgen05_st(") == 2 and "md_seed" in init
    assert chain.count(f"{P}.tcgen05_commit(md_cb") == 2
    assert chain.count(f"{WARP_RT}.sts_f32(") == 0
    # 32 exchange packets per chain iteration plus the 16 of T's exchange
    # tiles for loops 2 / 3, written once in the root segment ahead of
    # loop 2 (``_split_loop`` keeps that segment with ``chain``).
    assert chain.count(f"{RT}.sts_v4_f32(") == 48
    assert chain.count("% 8 ^ md_row64 % 8) * 16") == 32
    assert chain.count("% 8 ^ md_row64 % 4 * 2) * 16") == 16
    assert chain.count(f"{P}.tcgen05_st(") == 0
    # Three tensor-memory reads per step (Apow for its squaring, the
    # squared Apow and T for the accumulate); the root segment behind the
    # loop reads T twice more (A_inv's store, the hoisted exchange).
    loop1 = body[body.index("for md_i1 in") : body.index(_after_loop(body, "md_i1"))]
    assert loop1.count(f"{P}.tcgen05_ld(") == 3
    assert chain.count(f"{P}.tcgen05_ld(") == 5
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
    # (The four raw packets per thread; the bf16 result's drain is a TMA
    # store, no packet reads.)
    assert w_loop.count(f"{WARP_RT}.lds_v4_b32(") == 4
    assert w_loop.count(f"{WARP_RT}.unpack_bf16x2(") == 16
    assert w_loop.count(f"{WARP_RT}.sts_f32(") == 0
    # (T's exchange tile is written ahead of the loop: the eight packets are
    # the widen pass alone.)
    assert w_loop.count(f"{RT}.sts_v4_f32(") == 8
    assert len(re.findall(r"\^ md_pr_t\d+_\d+ % 4 \* 2\) \* 16", w_loop)) == 8
    assert w_loop.count(f"{P}.Tcgen05MMAKind.TF32") == 8
    assert w_loop.count(f"{WARP_RT}.stg_v4_b32(") == 0
    assert w_loop.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1
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
    "probe_hoist": lambda: _bind_probe_hoist(),
    "probe_hoist_loop": lambda: _bind_probe_hoist_loop(),
    "probe_rawlate": lambda: _bind_probe_rawlate(),
    "probe_carry2": lambda: _bind_probe_carry2(),
    "probe_kk": lambda: _bind_probe_kk(),
    "probe_recompute": lambda: _bind_probe_recompute(),
    "probe_drainrace": lambda: _bind_probe_drainrace(),
    "probe_dotless_store": lambda: _bind_probe_dotless_store(),
    "probe_register_carry": lambda: _bind_probe_register_carry(),
    "probe_register_carry_budget": lambda: _bind_probe_register_carry_budget(),
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
    # them -- it asserts); the hardware-order check finds no tensor-memory
    # hazard in the render; and no batch commits while the previous commit
    # is outstanding (a dots-only batch behind a loop used to).
    bound = _REGION_BINDERS[kernel]()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    tc_facts = facts.tcgen05
    assert tc_facts.regions
    program = _match(bound)
    dot_slots = {dot.slot for dot in program.dots}
    dot_slots |= {dot.acc_slot for dot in program.dots if dot.acc_slot is not None}
    dot_slots |= {
        program.frag_slots[operand.frag]
        for dot in program.dots
        for operand in (dot.a, dot.b)
        if operand.frag is not None
    }
    # A register carry (``register_carries``): no dot touches its slot; at
    # a config whose register budget admits it (``resident_slots``) it holds
    # no tensor-memory columns and the render never loads or stores it.
    candidates = set(regions.register_carries(program))
    assert candidates.isdisjoint(dot_slots)
    assert candidates <= set(tc_facts.tmem_slots)
    assert [carry.slot for carry in tc_facts.register_carries] == sorted(candidates)
    owned = dot_slots | {carry.slot for loop in program.loops for carry in loop.carries}
    for derived in program.derived:
        if derived.slot in owned:
            assert derived.slot in tc_facts.tmem_slots
        else:
            assert (derived.slot in tc_facts.tmem_slots) == regions.derived_crosses(
                program, derived
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
        with _instrumented_renderer():
            body = _kernel_body(bound.to_code(normalized))
        assert _commits_while_one_is_outstanding(body) == []
        _assert_tmem_hardware_order(
            bound, program, body, normalized.config["block_sizes"]
        )
        columns = _tmem_columns(body)
        resolved = _resolved_axes(program, bound, normalized.config["block_sizes"])
        resident = set(tc.resident_slots(tc_facts, resolved))
        assert resident <= candidates
        for slot in resident:
            assert not re.search(rf"{_TMEM_MARK} = .(ld|st) {slot} ", body), kernel
        # The region live groups: a hoisted exchange reads its fragment
        # ahead of the loop (the planner's groups keep the dot's position).
        live_groups = tc_facts.live_groups or facts.live_groups
        groups = [*live_groups, *tc_facts.tmem_conflicts]
        for group in groups:
            for a in group:
                for b in group:
                    if (
                        a < b
                        and a in tc_facts.tmem_slots
                        and b in tc_facts.tmem_slots
                        and a not in resident
                        and b not in resident
                    ):
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


def _probe_hoist(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A carry (s0) dying at the batch dot ``a``; the derived value ``w``,
    independent of ``a``, is hoisted ahead of a's operand exchange and
    needs tensor memory."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = hl.specialize(v.size(2))
    out_a = torch.empty([BHN, C, DV], dtype=torch.float32, device=q.device)
    out_c = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_m = torch.empty([BHN, C, DV], dtype=torch.float32, device=q.device)
    out_k = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        s0 = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s0 = hl.dot(
                q[tile_bhn, :, tile_d], k[tile_bhn, :, tile_d].transpose(-2, -1), acc=s0
            )
        m1 = hl.dot(s0.to(torch.bfloat16), v[tile_bhn, :, :])
        a = hl.dot((s0 * 2.0).to(torch.bfloat16), m1.to(torch.bfloat16))
        w = m1 * x[tile_bhn, :, :]
        c = hl.dot(w.to(torch.bfloat16), v[tile_bhn, :, :].transpose(-2, -1))
        out_a[tile_bhn, :, :] = a
        out_c[tile_bhn, :, :] = c
        out_m[tile_bhn, :, :] = m1
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kk = hl.dot(
                q[tile_bhn, :, tile_d], q[tile_bhn, :, tile_d].transpose(-2, -1), acc=kk
            )
        out_k[tile_bhn, :, :] = kk
    return out_a, out_c, out_m, out_k


def _probe_hoist_loop(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_probe_hoist`` inside a loop body: ``s`` dies at the batch dot
    ``acc_a``, the derived ``w`` (independent of it) is hoisted ahead of
    its exchange."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    E = k.size(1)
    DV = hl.specialize(v.size(2))
    out_a = torch.empty([BHN, C, DV], dtype=torch.float32, device=q.device)
    out_c = torch.empty([BHN, C, DV], dtype=torch.float32, device=q.device)
    out_k = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_m = torch.empty([BHN, C, E], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        acc_a = hl.zeros([tile_bhn, C, DV], dtype=torch.float32)
        acc_c = hl.zeros([tile_bhn, C, DV], dtype=torch.float32)
        for tile_e in hl.tile(E):
            s = hl.dot(q[tile_bhn, :, :], k[tile_bhn, tile_e, :].transpose(-2, -1))
            m = hl.dot(
                q[tile_bhn, :, :] * 1.0,
                k[tile_bhn, tile_e, :].transpose(-2, -1) * 1.0,
            )
            acc_a = hl.dot(
                (s * 2.0).to(torch.bfloat16), v[tile_bhn, tile_e, :], acc=acc_a
            )
            w = m * x[tile_bhn, :, tile_e]
            acc_c = hl.dot(w.to(torch.bfloat16), v[tile_bhn, tile_e, :], acc=acc_c)
            out_m[tile_bhn, :, tile_e] = m
        out_a[tile_bhn, :, :] = acc_a
        out_c[tile_bhn, :, :] = acc_c
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(q.size(2)):
            kk = hl.dot(
                q[tile_bhn, :, tile_d], q[tile_bhn, :, tile_d].transpose(-2, -1), acc=kk
            )
        out_k[tile_bhn, :, :] = kk
    return out_a, out_c, out_k, out_m


def _bind_probe_hoist(bhn: int = 4, d: int = 128, dv: int = 64) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_hoist,
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, dv), bf16),
        ((bhn, c, dv), f32),
    )


def _bind_probe_hoist_loop(bhn: int = 4, d: int = 64, e: int = 128) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_hoist_loop,
        ((bhn, c, d), bf16),
        ((bhn, e, d), bf16),
        ((bhn, e, d), bf16),
        ((bhn, c, e), f32),
    )


def _probe_rawlate(
    q: torch.Tensor, k: torch.Tensor, A: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A root fp32 tile converted to bf16 (a raw tile and its pass) read
    only by a root dot behind the first loop."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    out_r = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_k = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        s0 = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s0 = hl.dot(
                q[tile_bhn, :, tile_d], k[tile_bhn, :, tile_d].transpose(-2, -1), acc=s0
            )
        T = A[tile_bhn, :, :].to(torch.bfloat16)
        r = hl.dot(T, s0.to(torch.bfloat16))
        out_r[tile_bhn, :, :] = r
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_e in hl.tile(DV):
            kk = hl.dot(
                v[tile_bhn, :, tile_e], v[tile_bhn, :, tile_e].transpose(-2, -1), acc=kk
            )
        out_k[tile_bhn, :, :] = kk
    return out_r, out_k


def _probe_carry2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """An accumulator carried from loop 0 into loop 1 past two independent
    root dots and a derived value; loop 1 adds a fresh dot."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    E = v.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_z = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_y = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_f = torch.empty([BHN, C, E], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        acc = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            acc = hl.dot(
                q[tile_bhn, :, tile_d],
                k[tile_bhn, :, tile_d].transpose(-2, -1),
                acc=acc,
            )
        z = hl.dot(v[tile_bhn, :, :], v[tile_bhn, :, :].transpose(-2, -1))
        y = hl.dot(k[tile_bhn, :, :], k[tile_bhn, :, :].transpose(-2, -1))
        zz = z * x[tile_bhn, :, :]
        out_z[tile_bhn, :, :] = zz
        out_y[tile_bhn, :, :] = y
        for tile_e in hl.tile(E):
            acc = hl.dot(
                v[tile_bhn, :, tile_e],
                v[tile_bhn, :, tile_e].transpose(-2, -1),
                acc=acc,
            )
            f = hl.dot(y.to(torch.bfloat16), v[tile_bhn, :, tile_e])
            out_f[tile_bhn, :, tile_e] = f
        out[tile_bhn, :, :] = acc
    return out, out_z, out_y, out_f


def _probe_register_carry(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A carry read by epilogue work only -- acc = acc - p * 2 over loop
    0's fresh dot, its store behind loop 1 and the seed of loop 1's carry
    (``register_carries``); a second derived value in loop 0, p * x, the
    operand of a second dot (the register budget counts the carry's row);
    loop 1 accumulates onto its carry seeded from acc (a dot's
    accumulator: tensor memory)."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    E = v.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_u = torch.empty([BHN, C, D], dtype=torch.float32, device=q.device)
    out_t = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        acc = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            p = hl.dot(q[tile_bhn, :, tile_d], k[tile_bhn, :, tile_d].transpose(-2, -1))
            acc = acc - p * 2.0
            px = (p * x[tile_bhn, :, :]).to(q.dtype)
            out_u[tile_bhn, :, tile_d] = hl.dot(px, q[tile_bhn, :, tile_d])
        t = acc
        for tile_e in hl.tile(E):
            t = hl.dot(
                v[tile_bhn, :, tile_e], v[tile_bhn, :, tile_e].transpose(-2, -1), acc=t
            )
        out_t[tile_bhn, :, :] = t
        out[tile_bhn, :, :] = acc
    return out, out_u, out_t


def _bind_probe_register_carry(bhn: int = 4, d: int = 128, e: int = 64) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_register_carry,
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, e), bf16),
        ((bhn, c, c), f32),
    )


def _probe_register_carry_budget(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """T = T + Apow * An over a three-step squaring chain -- T is a
    register-carry candidate whose loop holds a derived value (Apow * An)
    over two other rows."""
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    D = k.size(2)
    out_t = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    out_a = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_kk in hl.tile(D):
            kt = k[tile_bhn, :, tile_kk]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        idx = hl.arange(C)
        strict_lower = idx[:, None] > idx[None, :]
        A = torch.where(strict_lower, -(kk * 0.05), 0.0)
        T = A * 0.0
        Apow = A
        for _ in range(3):
            An = hl.dot(Apow, Apow)
            T = T + Apow * An
            Apow = An
        out_t[tile_bhn, :, :] = T
        out_a[tile_bhn, :, :] = Apow
    return out_t, out_a


def _bind_probe_register_carry_budget(bhn: int = 4, d: int = 128) -> BoundKernel:
    return _bind_probe(_probe_register_carry_budget, ((bhn, 64, d), torch.bfloat16))


def _probe_kk(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two sequential loops; the second reads one q tile as both operands
    (a 128-column tile sharing the 64-column tiles' tensor map)."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    out_s = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_k = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        s0 = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s0 = hl.dot(
                q[tile_bhn, :, tile_d], k[tile_bhn, :, tile_d].transpose(-2, -1), acc=s0
            )
        out_s[tile_bhn, :, :] = s0
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kk = hl.dot(
                q[tile_bhn, :, tile_d], q[tile_bhn, :, tile_d].transpose(-2, -1), acc=kk
            )
        out_k[tile_bhn, :, :] = kk
    return out_s, out_k


def _bind_probe_rawlate(bhn: int = 4, d: int = 128, dv: int = 128) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_rawlate,
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, c), f32),
        ((bhn, c, dv), bf16),
    )


def _bind_probe_carry2(bhn: int = 4, d: int = 64, e: int = 128) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_carry2,
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, e), bf16),
        ((bhn, c, c), f32),
    )


def _bind_probe_kk(bhn: int = 4, d: int = 128) -> BoundKernel:
    bf16, c = torch.bfloat16, 64
    return _bind_probe(_probe_kk, ((bhn, c, d), bf16), ((bhn, c, d), bf16))


def _probe_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    q2: torch.Tensor,
    k2: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """d = s0 * x has no dot behind it before its store, yet xa = a * y
    defers the store behind the dot a's batch and sv = s0 * z takes s0's
    slot in between."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    E = q.size(2)
    out_d = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_c = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_c2 = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_k = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        s0 = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_e in hl.tile(E):
            s0 = hl.dot(
                q[tile_bhn, :, tile_e], k[tile_bhn, :, tile_e].transpose(-2, -1), acc=s0
            )
        a = hl.dot(q2[tile_bhn, :, :], k2[tile_bhn, :, :].transpose(-2, -1))
        d = s0 * x[tile_bhn, :, :]
        xa = a * y[tile_bhn, :, :]
        sv = s0 * z[tile_bhn, :, :]
        out_d[tile_bhn, :, :] = d
        c = hl.dot(sv.to(torch.bfloat16), v[tile_bhn, :, :])
        c2 = hl.dot(xa.to(torch.bfloat16), v[tile_bhn, :, :])
        out_c[tile_bhn, :, :] = c
        out_c2[tile_bhn, :, :] = c2
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(E):
            kk = hl.dot(
                q[tile_bhn, :, tile_d], q[tile_bhn, :, tile_d].transpose(-2, -1), acc=kk
            )
        out_k[tile_bhn, :, :] = kk
    return out_d, out_c, out_c2, out_k


def _probe_drainrace(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A root store drained by TMA in a block that ends with a dot batch,
    right before a loop whose frame overlays the root staging."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    E = k.size(2)
    D = q.size(2)
    DV = hl.specialize(v.size(2))
    out1 = torch.empty([BHN, C, DV], dtype=torch.bfloat16, device=q.device)
    out2 = torch.empty([BHN, C, DV], dtype=torch.float32, device=q.device)
    out_k = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        s0 = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_e in hl.tile(E):
            s0 = hl.dot(
                q[tile_bhn, :, tile_e], k[tile_bhn, :, tile_e].transpose(-2, -1), acc=s0
            )
        d1 = hl.dot(s0.to(torch.bfloat16), v[tile_bhn, :, :])
        out1[tile_bhn, :, :] = d1.to(torch.bfloat16)
        d2 = hl.dot(d1.to(torch.bfloat16), w[tile_bhn, :, :])
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kk = hl.dot(
                q[tile_bhn, :, tile_d], q[tile_bhn, :, tile_d].transpose(-2, -1), acc=kk
            )
        out_k[tile_bhn, :, :] = kk
        out2[tile_bhn, :, :] = d2
    return out1, out2, out_k


def _probe_dotless_store(
    q: torch.Tensor, k: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A dot-less loop whose body stores a derived tile through a staged,
    TMA-drained buffer: no commit in the loop."""
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    E = q.size(2)
    D = x.size(2)
    out_s = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out_z = torch.empty([BHN, C, D], dtype=torch.bfloat16, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        s0 = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_e in hl.tile(E):
            s0 = hl.dot(
                q[tile_bhn, :, tile_e], k[tile_bhn, :, tile_e].transpose(-2, -1), acc=s0
            )
        out_s[tile_bhn, :, :] = s0
        for tile_d in hl.tile(D):
            z = x[tile_bhn, :, tile_d] * y[tile_bhn, :, tile_d]
            out_z[tile_bhn, :, tile_d] = z.to(torch.bfloat16)
    return out_s, out_z


def _bind_probe_drainrace(bhn: int = 4, d: int = 128, dv: int = 64) -> BoundKernel:
    bf16, c = torch.bfloat16, 64
    return _bind_probe(
        _probe_drainrace,
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, dv), bf16),
        ((bhn, c, dv), bf16),
    )


def _bind_probe_dotless_store(bhn: int = 4, e: int = 128, d: int = 512) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_dotless_store,
        ((bhn, c, e), bf16),
        ((bhn, c, e), bf16),
        ((bhn, c, d), f32),
        ((bhn, c, d), f32),
    )


def _bind_probe_recompute(bhn: int = 4, d: int = 64, e: int = 128) -> BoundKernel:
    bf16, f32, c = torch.bfloat16, torch.float32, 64
    return _bind_probe(
        _probe_recompute,
        ((bhn, c, e), bf16),
        ((bhn, c, e), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, d), bf16),
        ((bhn, c, c), f32),
        ((bhn, c, c), f32),
        ((bhn, c, c), f32),
        ((bhn, c, d), bf16),
    )


# --------------------------------------------------------------------------
# The hardware-order tensor-memory checker.  The region renderer is
# instrumented to mark every tcgen05.ld / tcgen05.st / MMA it emits; the
# rendered body is executed abstractly -- loops unrolled, every branch taken,
# MMAs in flight from their issue to the next commit wait -- and a
# tensor-memory read of columns last written by another slot, or a read / store
# / issue on columns an in-flight MMA of another slot writes, is a hazard.  It
# is independent of the liveness rules it guards (the live groups and
# ``tmem_conflicts``): it only sees the render.
# --------------------------------------------------------------------------

_TMEM_MARK = "md_tm_mark"


def _value_ids(program: dag.MmaDagProgram) -> Callable[[object], int]:
    """A small integer per distinct fragment value: nodes through
    ``program.resolve``, with a loop carry's placeholder, its phi, its body
    output and (when the carry starts as the seed itself) its seed in one
    class -- the values the emitter may legitimately find in one slot."""
    from helion._compiler.cute.matmul_utils import strip_new_vars

    parent: dict[object, object] = {}

    def find(node: object) -> object:
        while parent.setdefault(node, node) is not node:
            node = parent[node]
        return node

    def union(a: object, b: object) -> None:
        parent[find(a)] = find(b)

    for loop in program.loops:
        for carry in loop.carries:
            union(carry.phi, carry.placeholder)
            union(program.resolve(carry.output), carry.placeholder)
            if carry.init_frag is not None and carry.init_value is None:
                union(program.resolve(carry.init_frag), carry.placeholder)
    ids: dict[object, int] = {}

    def value_id(node: object) -> int:
        assert node is not None
        canonical = find(program.resolve(strip_new_vars(node)))  # type: ignore[arg-type]
        return ids.setdefault(canonical, len(ids))

    return value_id


@contextlib.contextmanager
def _instrumented_renderer() -> Iterator[None]:
    """The region renderer marks its tensor-memory accesses (statements
    ``md_tm_mark = "<op> <slot> ..."`` the render carries; the kernel is
    never compiled here): ``ld`` / ``st`` / ``mma`` at the access, and the
    values the accesses mean -- ``use <slot> <value>`` where a step consumes
    a slot (an exchange, a store, a reduction, a derived value's input, a
    seed, a copy), ``def <slot> <value>`` where a step wrote one (a derived
    value, a carry init, a copy, a rescale), the MMA's result value and
    the value its accumulate expects -- so a slot holding another value
    than the reader's (two values sharing one logical slot) is a hazard the
    column owners alone cannot see."""
    emitter = regions._Tcgen05RegionEmitter
    names = (
        "emit_ld",
        "emit_st_array",
        "emit_dot_mmas",
        "emit_exchange",
        "emit_store_step",
        "emit_reduce",
        "emit_derived",
        "emit_loop_carry_init",
        "emit_carry_copies",
        "emit_hoisted_exchanges",
        "emit_rescale",
    )
    orig = {name: getattr(emitter, name) for name in names}

    def value_id(self: regions._Tcgen05RegionEmitter, node: object) -> int:
        ids = getattr(self, "_tm_value_ids", None)
        if ids is None:
            ids = _value_ids(self.program)
            self._tm_value_ids = ids  # type: ignore[attr-defined]
        return ids(node)

    def mark(self: regions._Tcgen05RegionEmitter, text: str) -> None:
        self.emit(f'{_TMEM_MARK} = "{text}"')

    def in_registers(self: regions._Tcgen05RegionEmitter, frag: object) -> bool:
        # A register-only derived value (no tensor-memory columns): its
        # readers take the block's registers, not the slot.
        from helion._compiler.cute.matmul_utils import strip_new_vars

        node = self.program.resolve(strip_new_vars(frag))  # type: ignore[arg-type]
        for step in self.program.derived:
            if step.node is node:
                return not self.derived_needs_tmem(step)
        return False

    def use(self: regions._Tcgen05RegionEmitter, frag: object) -> None:
        if not in_registers(self, frag):
            slot = self.program.frag_slots[frag]  # type: ignore[index]
            mark(self, f"use {slot} {value_id(self, frag)}")

    def emit_ld(self: regions._Tcgen05RegionEmitter, slot: int) -> None:
        mark(self, f"ld {slot} 0")
        orig["emit_ld"](self, slot)

    def emit_st_array(
        self: regions._Tcgen05RegionEmitter, name: str, slot: int, chunk: int
    ) -> None:
        mark(self, f"st {slot} {chunk}")
        orig["emit_st_array"](self, name, slot, chunk)

    def emit_dot_mmas(
        self: regions._Tcgen05RegionEmitter, dot: dag.Dot, stage: str, accumulate: bool
    ) -> None:
        expect = value_id(self, dot.acc) if accumulate and dot.acc is not None else -1
        mark(
            self,
            f"mma {dot.slot} {int(bool(accumulate))} {value_id(self, dot.node)} {expect}",
        )
        orig["emit_dot_mmas"](self, dot, stage, accumulate)

    def emit_exchange(
        self: regions._Tcgen05RegionEmitter, tile: tc.Tcgen05Tile
    ) -> None:
        assert tile.source_slot is not None and tile.operand.frag is not None
        mark(self, f"use {tile.source_slot} {value_id(self, tile.operand.frag)}")
        orig["emit_exchange"](self, tile)

    def emit_store_step(
        self: regions._Tcgen05RegionEmitter,
        store: dag.Store,
        loaded: set[int],
        parity: str,
    ) -> None:
        use(self, store.frag)
        orig["emit_store_step"](self, store, loaded, parity)

    def emit_reduce(
        self: regions._Tcgen05RegionEmitter, step: dag.Reduce, loaded: set[int]
    ) -> None:
        if step.frag is not None:
            use(self, step.frag)
        orig["emit_reduce"](self, step, loaded)

    def emit_derived(
        self: regions._Tcgen05RegionEmitter, step: dag.Derived, loaded: set[int]
    ) -> None:
        for frag in step.frags:
            use(self, frag)
        orig["emit_derived"](self, step, loaded)
        if self.derived_needs_tmem(step):
            mark(self, f"def {step.slot} {value_id(self, step.node)}")

    def emit_loop_carry_init(
        self: regions._Tcgen05RegionEmitter, loop: dag.Loop
    ) -> None:
        for carry in loop.carries:
            if carry.init_frag is not None:
                slot = self.program.frag_slots[self.program.resolve(carry.init_frag)]
                mark(self, f"use {slot} {value_id(self, carry.init_frag)}")
        orig["emit_loop_carry_init"](self, loop)
        for carry in loop.carries:
            mark(self, f"def {carry.slot} {value_id(self, carry.placeholder)}")

    def emit_carry_copies(self: regions._Tcgen05RegionEmitter, loop: dag.Loop) -> None:
        copies = [
            (carry, source)
            for other, carry, source in regions.copied_carries(self.program)
            if other.index == loop.index
        ]
        for carry, source in copies:
            mark(self, f"use {source} {value_id(self, carry.output)}")
        orig["emit_carry_copies"](self, loop)
        for carry, _slot in copies:
            mark(self, f"def {carry.slot} {value_id(self, carry.placeholder)}")

    def emit_hoisted_exchanges(
        self: regions._Tcgen05RegionEmitter, loop: dag.Loop
    ) -> None:
        program = self.program
        for dot in program.dots:
            if loop.index not in program.enclosing_loops(dot.loop):
                continue
            for operand in (dot.a, dot.b):
                if regions.hoisted_exchange_loop(program, dot, operand) == loop.index:
                    assert operand.frag is not None
                    slot = program.frag_slots[operand.frag]
                    mark(self, f"use {slot} {value_id(self, operand.frag)}")
        orig["emit_hoisted_exchanges"](self, loop)

    def emit_rescale(
        self: regions._Tcgen05RegionEmitter,
        dot: dag.Dot,
        loaded: set[int],
        prefix: str | None = None,
    ) -> None:
        assert dot.acc_frag is not None and dot.acc is not None
        slot = self.program.frag_slots[dot.acc_frag]
        mark(self, f"use {slot} {value_id(self, dot.acc_frag)}")
        orig["emit_rescale"](self, dot, loaded, prefix)
        # The prepared accumulator lands in the dot's own slot (the MMA
        # accumulates onto its result columns).
        mark(self, f"def {dot.slot} {value_id(self, dot.acc)}")

    def emit_carry_alias(
        self: regions._Tcgen05RegionEmitter, source: int, target: int
    ) -> None:
        # The elided copy: the output's columns hold the carry from here.
        mark(self, f"alias {source} {target}")

    hooks = {
        "emit_ld": emit_ld,
        "emit_st_array": emit_st_array,
        "emit_dot_mmas": emit_dot_mmas,
        "emit_exchange": emit_exchange,
        "emit_store_step": emit_store_step,
        "emit_reduce": emit_reduce,
        "emit_derived": emit_derived,
        "emit_loop_carry_init": emit_loop_carry_init,
        "emit_carry_copies": emit_carry_copies,
        "emit_hoisted_exchanges": emit_hoisted_exchanges,
        "emit_rescale": emit_rescale,
        "emit_carry_alias": emit_carry_alias,
    }
    with contextlib.ExitStack() as stack:
        for name, hook in hooks.items():
            stack.enter_context(patch.object(emitter, name, hook))
        yield


# --------------------------------------------------------------------------
# The value-version checker.  At emission time it tracks which value (FX node)
# every fragment slot holds -- in tensor memory and in the current block's
# registers -- updated on every MMA, derived value, rescale, tensor-memory
# store, carry copy and alias, and checks every read against the value the step
# expects (derived inputs, stores, reductions, exchanges, rescale sources, MMA
# accumulators).  At an iteration's end every carry slot must hold the carry's
# output, and every value read in the body but defined before it must still sit
# in its slot (the next iteration reads it again).  It catches same-slot
# clobbers that the column checker cannot see (a register-only derived value
# recomputed after its input slot was reused).
# --------------------------------------------------------------------------


class _VersionState:
    def __init__(self) -> None:
        self.tmem: dict[int, object] = {}
        self.regs: dict[int, object] = {}
        self.writers: list[object] = []
        self.frames: list[dict[str, object]] = []
        self.last_ld: object = None


def _version_name(value: object) -> str:
    if isinstance(value, tuple):
        return f"{value[0]}({_version_name(value[1])})"
    return getattr(value, "name", str(value))


@contextlib.contextmanager
def _version_checked() -> Iterator[list[str]]:
    """Render under the value-version checker; yields its error list."""
    emitter = regions._Tcgen05RegionEmitter
    names = (
        "emit_ld",
        "emit_st_array",
        "emit_dot_mmas",
        "emit_derived",
        "emit_store_step",
        "emit_reduce",
        "emit_exchange",
        "emit_rescale",
        "emit_loop_carry_init",
        "emit_carry_copies",
        "emit_carry_alias",
        "emit_ring_iteration",
        "emit_outer_iteration",
        "emit_region_loop",
    )
    orig = {name: getattr(emitter, name) for name in names}
    errors: list[str] = []

    def state(self: regions._Tcgen05RegionEmitter) -> _VersionState:
        if "_vc" not in self.__dict__:
            self.__dict__["_vc"] = _VersionState()
        return cast("_VersionState", self.__dict__["_vc"])

    def resolve(self: regions._Tcgen05RegionEmitter, node: object) -> object:
        return self.program.resolve(node) if node is not None else None  # type: ignore[arg-type]

    def current(self: regions._Tcgen05RegionEmitter, slot: int) -> object:
        s = state(self)
        if slot in self.resident:
            # A register carry: its scalars are the slot.
            return s.regs.get(slot, "unset")
        if slot in getattr(self, "slot_regs", {}) and slot in s.regs:
            return s.regs[slot]
        return s.tmem.get(slot, "unset")

    def held(self: regions._Tcgen05RegionEmitter, slot: int) -> object:
        """The value the slot keeps across blocks: its registers for a
        register carry, its tensor-memory columns otherwise."""
        s = state(self)
        if slot in self.resident:
            return s.regs.get(slot, "unset")
        return s.tmem.get(slot, "unset")

    def set_held(self: regions._Tcgen05RegionEmitter, slot: int, value: object) -> None:
        s = state(self)
        if slot in self.resident:
            s.regs[slot] = value
        else:
            s.tmem[slot] = value
            s.regs.pop(slot, None)

    def write_tmem(
        self: regions._Tcgen05RegionEmitter, slot: int, value: object
    ) -> None:
        s = state(self)
        s.tmem[slot] = value
        s.regs.pop(slot, None)
        for frame in s.frames:
            cast("set[int]", frame["written"]).add(slot)

    def check(
        self: regions._Tcgen05RegionEmitter,
        slot: int,
        expected: object,
        what: str,
        tmem_only: bool = False,
    ) -> None:
        s = state(self)
        got = s.tmem.get(slot, "unset") if tmem_only else current(self, slot)
        if got != expected:
            errors.append(
                f"{what}: slot {slot} holds {_version_name(got)}, expected "
                f"{_version_name(expected)}"
            )
        for frame in s.frames:
            if slot not in cast("set[int]", frame["written"]) and slot not in cast(
                "set[int]", frame["carries"]
            ):
                cast("list[object]", frame["reads"]).append((slot, got, what))

    def emit_ld(self: regions._Tcgen05RegionEmitter, slot: int) -> None:
        if slot in self.resident:
            errors.append(f"tcgen05.ld of register carry slot {slot}")
        orig["emit_ld"](self, slot)
        s = state(self)
        s.regs[slot] = s.tmem.get(slot, "unset")
        s.last_ld = s.regs[slot]

    def emit_st_array(
        self: regions._Tcgen05RegionEmitter, name: str, slot: int, chunk: int
    ) -> None:
        if slot in self.resident:
            errors.append(f"tcgen05.st of register carry slot {slot}")
        orig["emit_st_array"](self, name, slot, chunk)
        s = state(self)
        writer = s.writers[-1] if s.writers else "unknown"
        if writer == "copy":
            writer = s.last_ld
        write_tmem(self, slot, writer)

    def emit_dot_mmas(
        self: regions._Tcgen05RegionEmitter, dot: dag.Dot, stage: str, accumulate: bool
    ) -> None:
        if accumulate and dot.acc_frag is not None:
            expected = (
                ("acc", dot.node) if self.prepares(dot) else resolve(self, dot.acc_frag)
            )
            check(self, dot.slot, expected, f"MMA {dot.node.name} accumulate", True)
        orig["emit_dot_mmas"](self, dot, stage, accumulate)
        write_tmem(self, dot.slot, resolve(self, dot.node))

    def emit_derived(
        self: regions._Tcgen05RegionEmitter, step: dag.Derived, loaded: set[int]
    ) -> None:
        for frag in step.frags:
            check(
                self,
                self.program.frag_slots[frag],
                resolve(self, frag),
                f"derived {step.node.name} input",
            )
        s = state(self)
        s.writers.append(resolve(self, step.node))
        try:
            orig["emit_derived"](self, step, loaded)
        finally:
            s.writers.pop()
        s.regs[step.slot] = resolve(self, step.node)
        for frame in s.frames:
            cast("set[int]", frame["written"]).add(step.slot)

    def emit_store_step(
        self: regions._Tcgen05RegionEmitter,
        store: dag.Store,
        loaded: set[int],
        parity: str,
    ) -> None:
        check(
            self,
            self.program.frag_slots[store.frag],
            resolve(self, store.frag),
            f"store {store.access.node.name}",
        )
        orig["emit_store_step"](self, store, loaded, parity)

    def emit_reduce(
        self: regions._Tcgen05RegionEmitter, step: dag.Reduce, loaded: set[int]
    ) -> None:
        if step.frag is not None:
            check(
                self,
                self.program.frag_slots[step.frag],
                resolve(self, step.frag),
                "reduce",
            )
        orig["emit_reduce"](self, step, loaded)

    def emit_exchange(
        self: regions._Tcgen05RegionEmitter, tile: tc.Tcgen05Tile
    ) -> None:
        operand = tile.operand
        if operand.frag is not None and tile.source_slot is not None:
            check(
                self,
                tile.source_slot,
                resolve(self, operand.frag),
                f"exchange {tile.tag}",
            )
        orig["emit_exchange"](self, tile)

    def emit_rescale(
        self: regions._Tcgen05RegionEmitter,
        dot: dag.Dot,
        loaded: set[int],
        prefix: str | None = None,
    ) -> None:
        assert dot.acc_slot is not None
        check(
            self, dot.acc_slot, resolve(self, dot.acc_frag), f"rescale {dot.node.name}"
        )
        s = state(self)
        s.writers.append(("acc", dot.node))
        try:
            orig["emit_rescale"](self, dot, loaded, prefix)
        finally:
            s.writers.pop()
        if prefix is not None:
            s.regs[dot.slot] = ("acc", dot.node)
            for frame in s.frames:
                cast("set[int]", frame["written"]).add(dot.slot)

    def emit_loop_carry_init(
        self: regions._Tcgen05RegionEmitter, loop: dag.Loop
    ) -> None:
        s = state(self)
        for carry in loop.carries:
            if carry.init_frag is not None:
                check(
                    self,
                    self.program.frag_slots[self.program.resolve(carry.init_frag)],
                    resolve(self, carry.init_frag),
                    f"carry seed of loop {loop.index}",
                )
        s.writers.append(("init", loop.index))
        try:
            orig["emit_loop_carry_init"](self, loop)
        finally:
            s.writers.pop()
        for carry in loop.carries:
            if carry.slot in self.resident:
                # The seed went into the carry's scalars.
                s.regs[carry.slot] = ("init", loop.index)
                for frame in s.frames:
                    cast("set[int]", frame["written"]).add(carry.slot)

    def emit_carry_copies(self: regions._Tcgen05RegionEmitter, loop: dag.Loop) -> None:
        s = state(self)
        s.writers.append("copy")
        try:
            orig["emit_carry_copies"](self, loop)
        finally:
            s.writers.pop()

    def emit_carry_alias(
        self: regions._Tcgen05RegionEmitter, source: int, target: int
    ) -> None:
        orig["emit_carry_alias"](self, source, target)
        s = state(self)
        write_tmem(self, target, s.tmem.get(source, "unset"))

    def iteration(name: str) -> Callable[..., None]:
        def wrapper(
            self: regions._Tcgen05RegionEmitter, loop: dag.Loop, *args: object
        ) -> None:
            s = state(self)
            for carry in loop.carries:
                set_held(self, carry.slot, resolve(self, carry.placeholder))
            frame: dict[str, object] = {
                "loop": loop,
                "written": set(),
                "carries": {carry.slot for carry in loop.carries},
                "reads": [],
            }
            s.frames.append(frame)
            try:
                orig[name](self, loop, *args)
            finally:
                s.frames.pop()
            for carry in loop.carries:
                expected = resolve(self, carry.output)
                got = held(self, carry.slot)
                if got != expected:
                    errors.append(
                        f"loop {loop.index} carry slot {carry.slot} ends the iteration "
                        f"holding {_version_name(got)}, expected {_version_name(expected)}"
                    )
            for slot, value, what in cast(
                "list[tuple[int, object, str]]", frame["reads"]
            ):
                if held(self, slot) != value:
                    errors.append(
                        f"loop {loop.index}: {what} reads slot {slot} = "
                        f"{_version_name(value)} defined before the body, but the body "
                        f"leaves {_version_name(held(self, slot))} for the next "
                        "iteration"
                    )

        return wrapper

    def emit_region_loop(
        self: regions._Tcgen05RegionEmitter, loop: dag.Loop, region: int
    ) -> None:
        orig["emit_region_loop"](self, loop, region)
        for carry in loop.carries:
            set_held(self, carry.slot, resolve(self, carry.phi))

    hooks: dict[str, object] = {
        "emit_ld": emit_ld,
        "emit_st_array": emit_st_array,
        "emit_dot_mmas": emit_dot_mmas,
        "emit_derived": emit_derived,
        "emit_store_step": emit_store_step,
        "emit_reduce": emit_reduce,
        "emit_exchange": emit_exchange,
        "emit_rescale": emit_rescale,
        "emit_loop_carry_init": emit_loop_carry_init,
        "emit_carry_copies": emit_carry_copies,
        "emit_carry_alias": emit_carry_alias,
        "emit_ring_iteration": iteration("emit_ring_iteration"),
        "emit_outer_iteration": iteration("emit_outer_iteration"),
        "emit_region_loop": emit_region_loop,
    }
    with contextlib.ExitStack() as stack:
        for name, hook in hooks.items():
            stack.enter_context(patch.object(emitter, name, hook))
        yield errors


def _version_errors(bound: BoundKernel, config: helion.Config) -> list[str]:
    with _version_checked() as errors:
        _source(bound, config)
    return list(errors)


def _random_tcgen05_configs(
    bound: BoundKernel, seed: str, count: int
) -> list[helion.Config]:
    """``count`` random legal tensor-memory configs (power-of-two block
    sizes, 4 / 8 warps, 1-4 stages, both loads modes), normalized."""
    spec = bound.config_spec
    rng = random.Random(seed)
    found: list[helion.Config] = []
    for _attempt in range(6 * count):
        if len(found) == count:
            break
        sizes = []
        for block in spec.block_sizes:
            low, high = max(block.min_size, 16), block.max_size
            choices = [
                1 << i for i in range(4, high.bit_length()) if low <= 1 << i <= high
            ]
            sizes.append(rng.choice(choices or [low]))
        config = helion.Config(
            block_sizes=sizes,
            cute_mma_dag="tcgen05",
            cute_mma_dag_warps=rng.choice([4, 8]),
            cute_mma_dag_stages=rng.choice([1, 2, 3, 4]),
            cute_mma_dag_loads=rng.choice(["tma", "lane"]),
        )
        try:
            with bound.env:
                normalized = spec.normalized_config(config)
        except (InvalidConfig, BackendUnsupported):
            continue
        if normalized.config["cute_mma_dag"] == "tcgen05":
            found.append(normalized)
    return found


def _instrumented_body(bound: BoundKernel, config: helion.Config) -> str:
    with _instrumented_renderer():
        return _kernel_body(_source(bound, config))


def _tmem_trace(body: str) -> list[tuple[object, ...]]:
    """The body's tensor-memory operations as a tree: ``("loop", line,
    trips, children)`` nodes (literal trip counts, at least one and at most
    three) and ``(op, line, slot, *fields)`` leaves for the marks (``ld``,
    ``st``, ``mma``, ``use``, ``def``; ``alias``: an elided carry copy, the
    output slot's columns hold the carry from there) and the commit waits
    (``mbar_wait(md_cb``); branches are transparent."""
    root: list[tuple[object, ...]] = []
    stack: list[tuple[int, list[tuple[object, ...]]]] = [(-1, root)]
    for number, line in enumerate(body.split("\n")):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        mark = re.match(rf"{_TMEM_MARK} = ['\"](\w+) (\d+)((?: -?\d+)+)['\"]", text)
        if text.startswith("for ") and text.endswith(":"):
            trips = 2
            bounds = re.match(r"for \w+ in cutlass\.range\((\d+), (\d+)\):", text)
            if bounds:
                trips = int(bounds.group(2)) - int(bounds.group(1))
            children: list[tuple[object, ...]] = []
            current.append(("loop", number, max(min(trips, 3), 1), children))
            stack.append((indent, children))
        elif mark:
            fields = [int(field) for field in mark.group(3).split()]
            current.append((mark.group(1), number, int(mark.group(2)), *fields))
        elif "mbar_wait(md_cb" in text:
            current.append(("wait", number, 0, 0))
        elif text.endswith(":") and text.startswith(("if ", "elif ", "else", "with ")):
            stack.append((indent, current))
    return root


def _tmem_hazards(body: str, spans: dict[int, tuple[int, int]]) -> list[str]:
    """The hazards of the body's abstract execution over the slots' column
    spans (``slot -> (first column, width)``): a read of columns last
    written by another slot; a read / store / MMA issue on columns an
    in-flight MMA writes (a tcgen05.ld or tcgen05.st under the slot's own
    MMA included; an accumulate chains onto its own); and a value read
    (``use``, an accumulate) that is not the value last defined there
    (``def``, an MMA's result) -- two values sharing one logical slot."""
    owner: dict[int, int] = {}
    value: dict[int, int | None] = {}
    inflight: list[tuple[int, int]] = []
    hazards: set[str] = set()

    def columns(slot: int) -> set[int]:
        start, width = spans[slot]
        return set(range(start, start + width))

    def under_mma(slot: int, line: int, what: str, own: bool = True) -> None:
        for other, issued in inflight:
            if (own or other != slot) and columns(other) & columns(slot):
                hazards.add(
                    f"line {line}: {what} of slot {slot} under the MMA into slot "
                    f"{other} (line {issued})"
                )

    def read(slot: int, line: int, what: str) -> None:
        others = sorted({owner.get(c) for c in columns(slot)} - {slot, None})  # type: ignore[arg-type]
        if others:
            hazards.add(
                f"line {line}: {what} of slot {slot} reads columns last written by "
                f"slot(s) {others}"
            )
        under_mma(slot, line, what, own=what != "the MMA accumulate")

    def expect(slot: int, line: int, what: str, wanted: int) -> None:
        found = {value.get(c) for c in columns(slot)} - {None}
        if found - {wanted}:
            hazards.add(
                f"line {line}: {what} of slot {slot} wants value {wanted} but the "
                f"columns hold value(s) {sorted(found - {wanted})}"  # type: ignore[type-var]
            )

    def define(slot: int, held: int | None) -> None:
        for c in columns(slot):
            value[c] = held

    def write(slot: int) -> None:
        for c in columns(slot):
            owner[c] = slot

    def run(nodes: list[tuple[object, ...]]) -> None:
        for node in nodes:
            kind, line = node[0], node[1]
            if kind == "loop":
                for _trip in range(node[2]):  # type: ignore[arg-type]
                    run(node[3])  # type: ignore[arg-type]
                continue
            slot = node[2]
            assert isinstance(slot, int) and isinstance(line, int)
            if kind == "wait":
                inflight.clear()
                continue
            if slot not in spans:
                # A register-only slot (no tensor-memory columns): its
                # values never meet another slot's.
                continue
            if kind == "ld":
                read(slot, line, "tcgen05.ld")
            elif kind == "st":
                under_mma(slot, line, "tcgen05.st")
                write(slot)
                define(slot, None)
            elif kind == "mma":
                if node[3]:
                    read(slot, line, "the MMA accumulate")
                    if len(node) > 5 and node[5] != -1:
                        expect(slot, line, "the MMA accumulate", node[5])  # type: ignore[arg-type]
                else:
                    under_mma(slot, line, "the MMA", own=False)
                write(slot)
                if len(node) > 4:
                    define(slot, node[4])  # type: ignore[arg-type]
                inflight.append((slot, line))
            elif kind == "use":
                expect(slot, line, "the read", node[3])  # type: ignore[arg-type]
            elif kind == "def":
                define(slot, node[3])  # type: ignore[arg-type]
            elif kind == "alias":
                # The elided copy renames the output's columns to the
                # carry's: a read of the carry before this point in the
                # iteration still sees the output's write (a hazard).
                target = node[3]
                assert isinstance(target, int)
                for c in columns(slot):
                    if owner.get(c) == slot:
                        owner[c] = target

    run(_tmem_trace(body))
    return sorted(hazards)


def _slot_spans(
    body: str, bound: BoundKernel, program: dag.MmaDagProgram, block_sizes: list[int]
) -> dict[int, tuple[int, int]]:
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    resolved = _resolved_axes(program, bound, block_sizes)
    # A register carry resident at this config holds no columns.
    resident = tc.resident_slots(facts.tcgen05, resolved)
    return {
        slot: (
            column,
            tc._align(resolved[facts.frags[slot][1]], tc.TCGEN05_SLOT_ALIGN),
        )
        for slot, column in _tmem_columns(body).items()
        if slot in facts.tcgen05.tmem_slots and slot not in resident
    }


def _assert_tmem_hardware_order(
    bound: BoundKernel, program: dag.MmaDagProgram, body: str, block_sizes: list[int]
) -> None:
    """``body`` (rendered under ``_instrumented_renderer``) has no
    tensor-memory hazard."""
    assert _TMEM_MARK in body, "render the body under _instrumented_renderer"
    hazards = _tmem_hazards(body, _slot_spans(body, bound, program, block_sizes))
    assert not hazards, hazards


def test_a_root_fragment_read_in_a_nested_loop_is_exchanged_once_ahead_of_it() -> None:
    # dattn (the root carry, cast for the inner loop's dot) was re-read from
    # tensor memory and re-exchanged by every iteration of the tile_d loop, so
    # it had to stay live throughout that loop, apart from z (a fresh dot of
    # the tile_d loop after the inner loop; from the second tile_d iteration
    # on, the inner loop once read z as dattn).  A root fragment's operand has
    # one value for every iteration (the matcher keeps the chain's aux loads
    # invariant below the fragment's frame), so its exchange tile is written
    # once ahead of the tile_d loop (``hoisted_exchange_loop``) and its columns
    # are free during it: z shares them, the region live groups say so, and the
    # hardware-order checker agrees.
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
    (y,) = [dot for dot in program.dots if program.loops[dot.loop].parent >= 0]
    (z,) = [
        dot.slot
        for dot in program.dots
        if dot.loop >= 0
        and program.loops[dot.loop].parent < 0
        and dot.a.kind == "load"
        and dot.acc_slot is None
        and dot.loop != 0
    ]
    tile_d = program.loops[y.loop].parent
    assert y.a.frag is not None
    assert regions.hoisted_exchange_loop(program, y, y.a) == tile_d
    assert regions.hoisted_exchange_loop(program, y, y.b) is None
    assert (min(dattn, z), max(dattn, z)) not in facts.tcgen05.tmem_conflicts
    assert facts.tcgen05.live_groups is not None
    assert not any(dattn in group and z in group for group in facts.tcgen05.live_groups)
    body = _instrumented_body(bound, _config(64, 64, 64, warps=4, stages=2))
    columns = _tmem_columns(body)
    assert columns[dattn] == columns[z] == 0
    assert f"{P}.tcgen05_alloc(md_tslot, 64," in body
    _assert_tmem_hardware_order(bound, program, body, [64, 64, 64])
    # The exchange (four packets of the [64, 64] bf16 tile per thread, the
    # tensor-memory read ahead of it) sits between loop 0's exit and the
    # tile_d loop; the loops read no tensor memory but y's and z's results.
    before, rest = _split_loop(body, "md_i1")
    between = before[before.rindex("md_rk0 = md_rk0 + 1") :]
    read = f"{P}.tcgen05_ld('16x32bx2', cutlass.inttoptr(md_warp % 4 * 32 << 16 | md_tmem + 0,"
    assert between.count(read) == 1 and between.count(f"{WARP_RT}.sts_v4_b32(") == 4
    _assert_in_order(
        between,
        [
            read,
            f"{WARP_RT}.sts_v4_b32(",
            "cute.arch.fence_proxy(kind='async.shared', space='cta')",
            f"{P}.tcgen05_fence({P}.Tcgen05Fence.BEFORE_THREAD_SYNC)",
            "cute.arch.barrier()",
        ],
    )
    assert rest.count(f"{WARP_RT}.sts_v4_b32(") == 0
    assert rest.count(f"{P}.tcgen05_ld(") == 2


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
    body = _instrumented_body(bound, _config(64, 64, warps=4, stages=2))
    assert body.count(f"{P}.tcgen05_st(") >= 1
    _assert_tmem_hardware_order(bound, program, body, [64, 64])


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
    # 16 rotated packets per chain iteration and the 16 of T's hoisted
    # exchange tiles ahead of loop 2.
    assert len(re.findall(r"md_cb\d // 4 \+ \d\) % 8 \^ md_row64 % 8\)", chain)) == 32
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


# --------------------------------------------------------------------------
# dag-regions: the derived values' tile inputs ride the loop ring
# --------------------------------------------------------------------------


def test_the_loads_knob_routes_derived_tile_inputs_through_the_ring() -> None:
    # ``cute_mma_dag_loads``: ``tma`` (first choice, the default) fetches
    # dk's seed tiles into the ring; ``lane`` keeps the per-lane global pair
    # loads (and the shared memory); the knob is inert off the tensor-memory
    # region programs with such inputs (pinned to ``tma``).
    bound = _bind_dk()
    spec = bound.config_spec
    assert spec.cute_mma_dag_loads is not None
    assert spec.cute_mma_dag_loads.choices == ("tma", "lane")
    assert spec.default_config().config["cute_mma_dag_loads"] == "tma"
    with bound.env:
        lane = spec.normalized_config(
            _config(64, warps=4, stages=2, cute_mma_dag_loads="lane")
        )
        warp = spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_loads="lane",
            )
        )
    assert lane.config["cute_mma_dag_loads"] == "lane"
    assert warp.config["cute_mma_dag_loads"] == "tma"
    body = _kernel_body(bound.to_code(lane))
    loop = _loop(body)
    assert loop.count(f"{WARP_RT}.ldg_v2_f32(") == 32
    assert f"{RT}.lds_v4_f32(" not in loop and f"{P}.prefetch_l2(" in loop
    assert "'tma_count': 3" in body
    assert "alloc_smem(cutlass.Uint8, 40960, alignment=1024)" in body
    # The simple (single-loop) tensor-memory programs have no aux tiles.
    fwd = _bind_fwd()
    with fwd.env:
        pinned = fwd.config_spec.normalized_config(
            _config(64, 64, warps=4, stages=2, cute_mma_dag_loads="lane")
        )
    assert pinned.config["cute_mma_dag"] == "tcgen05"
    assert pinned.config["cute_mma_dag_loads"] == "tma"


def test_aux_tiles_that_do_not_fit_stay_lane_loads() -> None:
    # state_du_kda's serial loop reads four tiles per iteration through
    # derived values (k and gc for k_scaled, dv_new_out, dh_all); under
    # ``tma`` the largest ring footprints are left out until the layout
    # fits the capacity, so a config is legal under either loads mode.
    facts = _bind_state_du_kda().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    items = facts.tcgen05.items
    assert [i.kind for i in items].count("aux") == 4
    assert facts.tcgen05.aux_items == 4
    resolved = [1, 64, 64]
    capacity = facts.smem_capacity
    assert capacity == 232448
    skip = regions.region_aux_skip(items, facts.loops, resolved, 2, capacity)
    assert skip == frozenset()
    assert regions.region_smem_bytes(facts, resolved, 2) == 213032
    skip = regions.region_aux_skip(items, facts.loops, resolved, 4, capacity)
    assert skip == frozenset({6, 7})
    assert items[6].kind == items[7].kind == "aux"
    assert regions.region_smem_bytes(facts, resolved, 4) == 213048 <= capacity
    assert regions.region_smem_bytes(facts, resolved, 4, "lane") == 114744
    assert regions.region_aux_skip(
        items, facts.loops, resolved, 4, capacity, "lane"
    ) == frozenset({4, 5, 6, 7})
    for stages in (1, 2, 4):
        assert (
            dag.mma_dag_config_reason(facts, {1: 64, 0: 1}, 4, stages, "tcgen05")
            is None
        )


def test_a_load_only_ring_loop_syncs_before_its_refill() -> None:
    # Gated dqkw's ``hd`` loop reads two tiles per iteration through a
    # derived value and runs no dot: its iterations have no batch barrier,
    # so a CTA barrier orders every thread's chunk reads of the previous
    # iteration ahead of the refill of that slot.  dk's loop (a dot batch
    # per iteration) needs none between its reads and the refill.
    body = _kernel_body(
        _source(_bind_dqkw(gated=True), _config(64, 32, 16, 64, 16, warps=4, stages=2))
    )
    _before, hd = _split_loop(body, "md_i3")
    hd = hd[: hd.index("md_rk3 = md_rk3 + 1")]
    _assert_in_order(
        hd,
        [
            f"{RT}.mbar_wait(md_full3 + md_st3, md_rd3)",
            f"{WARP_RT}.lds_v4_b32(md_s0 + 0 + md_st3 * 4096 + 0 + md_row64 * 32",
            f"{WARP_RT}.lds_v4_b32(md_s0 + 0 + md_st3 * 4096 + 2048 + md_row64 * 32",
            "cute.arch.barrier()",
            f"{P}.mbarrier_arrive_expect_tx(md_full3 + (md_rk3 + 1) % 2, 4096)",
        ],
    )
    assert f"{P}.tcgen05_mma(" not in hd
    assert hd.index("cute.arch.barrier()") < hd.index(
        f"{P}.mbarrier_arrive_expect_tx(md_full3"
    )
    assert "'tma_count': 14" in body
    loop = _loop(_kernel_body(_source(_bind_dk(), _config(64, warps=4, stages=2))))
    assert loop.index(f"{P}.mbarrier_arrive_expect_tx(md_full0") < loop.index(
        "cute.arch.barrier()"
    )


def test_an_unaligned_derived_source_keeps_the_lane_loads() -> None:
    # dk_as with a 520-byte row stride cannot be a TMA box: its aux item
    # keeps its place in the layout but the lanes load their pairs from
    # global memory; d_ba still rides the ring.
    t = _tensors3()
    bhn, c, d = t["q"].shape
    f32 = torch.float32
    dk_as = torch.empty((bhn, c, d + 2), dtype=f32, device=CPU_DEVICE)[..., :d]
    bound = _bind(
        "chunk_bwd_dk_delta_helion",
        (
            t["q"],
            torch.empty((bhn, c, c), dtype=f32, device=CPU_DEVICE),
            torch.empty((bhn, c, d), dtype=f32, device=CPU_DEVICE),
            torch.empty((bhn, c), dtype=f32, device=CPU_DEVICE),
            dk_as,
        ),
    )
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=2)))
    loop = _loop(body)
    assert loop.count(f"{WARP_RT}.ldg_v2_f32(") == 16
    assert loop.count(f"{RT}.lds_v4_f32(") == 8
    assert loop.count(f"{P}.prefetch_l2(") == 2
    assert "'tma_count': 4" in body
    assert "alloc_smem(cutlass.Uint8, 106496, alignment=1024)" in body


def _first_tcgen05_seeds(
    bound: BoundKernel, count: int = 4
) -> list[tuple[list[int], int, int]]:
    seeds = [
        c.config
        for c in bound.config_spec.compiler_seed_configs
        if c.config["cute_mma_dag"] == "tcgen05"
    ]
    return [
        (c["block_sizes"], c["cute_mma_dag_warps"], c["cute_mma_dag_stages"])
        for c in seeds[:count]
    ]


def test_region_seeds_rank_two_per_sm_a_two_slot_ring_then_iterations_per_resident_cta() -> (
    None
):
    # The region programs' tensor-memory seeds: two CTAs per SM (shared
    # memory with the aux tiles, TMEM) first, a two-slot ring, then the
    # fewest innermost-loop iterations per resident CTA (every iteration a
    # commit round trip, overlapped by up to three CTAs sharing the SM),
    # the stage depth, four warps before eight.  The stage ladder is capped
    # by the deepest loop, not the last one ([32, 32, 128] has a one-trip
    # third loop beside two four-trip ones).
    # dk: [64] s2 (104 KB) fits two CTAs per SM with two iterations (1 per
    # resident CTA) ahead of [32] s2 (56 KB with its two 8 KB aux tiles per
    # slot) at three with four (1.33); [64] s1 ranks behind every two-slot
    # ring.
    assert _first_tcgen05_seeds(_bind_dk()) == [
        ([64], 4, 2),
        ([64], 8, 2),
        ([32], 4, 2),
        ([32], 8, 2),
    ]
    # wy_dL (the CPU binder's fp32 dvni adds a raw ring tile, so [64, 32,
    # 128] s3 fits two CTAs per SM here, three on the engine's bf16 dvni):
    # 2 + 2 + 1 iterations at two per SM (2.5) before 4 + 4 + 1 at three
    # (3), in both decay modes (gated wy_dL's TMEM is 128 columns too once
    # dLL stays in registers).
    for gated in (False, True):
        assert _first_tcgen05_seeds(_bind_wy_dl(gated=gated)) == [
            ([64, 64, 128], 4, 2),
            ([64, 64, 128], 8, 2),
            ([32, 32, 128], 4, 3),
            ([32, 32, 128], 8, 3),
        ]
    # dqkw: with the tile_d loop's three stores single-buffered (one copy
    # each: a nested loop runs between consecutive stage-ins), [128, 64, 32]
    # s2 fits two CTAs per SM (106560 B) with 1 + 2 * 4 iterations (4.5 per
    # resident CTA) ahead of [128, 32, 32] s3 with 1 + 4 * 4 (8.5); the
    # tile_dv-wide twin [128, 32, 64] s2 ranks next at the same count.
    # (B200, device us at s2 / s1: [128, 64, 32] s2 452 / 129 against the
    # previous first seed [128, 32, 32] s3 585 / 155; gated 590 / 158
    # against 703 / 189.)
    assert _first_tcgen05_seeds(_bind_dqkw(), 3) == [
        ([128, 64, 32], 4, 2),
        ([128, 32, 64], 4, 2),
        ([128, 64, 32], 8, 2),
    ]
    # wy has a chain loop (the Neumann doubling: no tile axis, no ring), so
    # its rank puts the CTAs per SM up to four ahead of the ring depth:
    # [128, 32, 32] s2 (48 KB, 128 TMEM columns: four per SM, 1 + 1 + 4 + 4
    # iterations) before [64, 32, 32] (five more iterations) and far before
    # [128, 64, 64] s2 (64 KB: three per SM), in both decay modes.  (B200
    # at s2: 332 us against 407.)
    for gated in (False, True):
        assert _first_tcgen05_seeds(_bind_wy(gated=gated), 3) == [
            ([128, 32, 32], 4, 2),
            ([128, 32, 32], 8, 2),
            ([64, 32, 32], 4, 3),
        ]
    facts = _bind_wy().config_spec.cute_mma_dag_facts
    assert facts is not None
    loop_axes = {axis.loop for axis in facts.axes if axis.kind == dag.AXIS_LOOP}
    assert [index in loop_axes for index in range(len(facts.loops))] == [
        True,
        False,
        True,
        True,
    ]
    for binder in (_bind_dk, _bind_wy_dl, _bind_dqkw):
        facts = binder().config_spec.cute_mma_dag_facts
        assert facts is not None
        loop_axes = {axis.loop for axis in facts.axes if axis.kind == dag.AXIS_LOOP}
        assert all(index in loop_axes for index in range(len(facts.loops)))
    # The iteration count: dqkw's dattn loop (bdv) beside the tile_d loop
    # nesting the tile_dv loop.
    facts = _bind_dqkw().config_spec.cute_mma_dag_facts
    assert facts is not None
    tunable = [
        (index, axis)
        for index, axis in enumerate(facts.axes)
        if axis.kind in (dag.AXIS_ROOT, dag.AXIS_LOOP)
    ]
    assert [axis.loop for _index, axis in tunable] == [0, 1, 2]
    assert [(loop.parent, loop.innermost) for loop in facts.loops] == [
        (-1, True),
        (-1, False),
        (1, True),
    ]
    assert tc.region_iterations(facts, tunable, (128, 32, 32)) == 1 + 4 * 4
    assert tc.region_iterations(facts, tunable, (32, 32, 32)) == 4 + 4 * 4
    assert tc.region_iterations(facts, tunable, (64, 16, 16)) == 2 + 8 * 8
    # Every two-per-SM seed precedes every one-per-SM seed; within a tier
    # and ring depth class the iteration counts do not decrease.
    with _bind_wy_dl().env:
        seeds = tc.tcgen05_seed_configs(facts)
    assert len(seeds) > 100
    facts_wy = _bind_wy_dl().config_spec.cute_mma_dag_facts
    assert facts_wy is not None
    with _bind_wy_dl().env:
        seeds = tc.tcgen05_seed_configs(facts_wy)
    tunable = [
        (index, axis)
        for index, axis in enumerate(facts_wy.axes)
        if axis.kind in (dag.AXIS_ROOT, dag.AXIS_LOOP)
    ]
    keys = []
    for sizes, _warps, stages in seeds:
        resolved = [facts_wy.block_size(i, sizes) for i in range(len(facts_wy.axes))]
        smem = tc.tcgen05_smem_bytes(facts_wy, resolved, stages, True)
        per_sm = min(
            (facts_wy.smem_capacity + 1024) // (smem + 1024),
            512 // tc.tcgen05_tmem_cols(tc.tcgen05_slot_columns(facts_wy, resolved)),
        )
        choice = tuple(sizes[axis.block_id] for _index, axis in tunable)
        resident = max(min(per_sm, tc.TCGEN05_REGION_SEED_CTAS), 1)
        keys.append(
            (
                int(per_sm >= 2),
                int(stages >= 2),
                -(tc.region_iterations(facts_wy, tunable, choice) * 6) // resident,
            )
        )
    assert keys == sorted(keys, reverse=True)
    assert keys[0][0] == 1 and keys[-1][0] == 0


@helion.kernel(backend="cute", static_shapes=True)
def _two_loops_store_between(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    bhn, rows, k = a.shape
    n = b.shape[2]
    out1 = torch.empty([bhn, rows, n], dtype=a.dtype, device=a.device)
    out2 = torch.empty([bhn, rows, n], dtype=a.dtype, device=a.device)
    for tile in hl.tile(bhn, block_size=1):
        acc = hl.zeros([tile, rows, n], dtype=torch.float32)
        for i in hl.tile(k):
            acc = hl.dot(a[tile, :, i], b[tile, i, :], acc=acc)
        out1[tile, :, :] = acc.to(a.dtype)
        acc2 = hl.zeros([tile, rows, n], dtype=torch.float32)
        for j in hl.tile(k):
            acc2 = hl.dot(c[tile, :, j], d[tile, j, :], acc=acc2)
        out2[tile, :, :] = acc2.to(a.dtype)
    return out1, out2


@helion.kernel(backend="cute", static_shapes=True)
def _hoisted_raw_over_a_narrow_ring(a32: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bhn, rows, _k = a32.shape
    n = b.shape[2]
    out = torch.empty([bhn, rows, n], dtype=b.dtype, device=b.device)
    for tile in hl.tile(bhn, block_size=1):
        a16 = a32[tile, :, :].to(b.dtype)
        for i in hl.tile(n):
            out[tile, :, i] = hl.dot(a16, b[tile, :, i]).to(b.dtype)
    return out


def test_root_items_within_one_segment_overlay_the_loop_frames() -> None:
    # Root items whose lifetime is one root segment overlay the loop frames
    # from the top of their extent: wy_dL's raw A_inv tile (converted in
    # the prologue: segment 0), the two exchange tiles of the chain between
    # loops 1 and 2 (segment 2) and dL's staging tile after loop 2 (segment
    # 3); the hoisted bf16 tile and the scratch stay live across the loops.
    bound = _bind_wy_dl()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    items = facts.tcgen05.items
    assert [(i.kind, i.frame, i.segment) for i in items if i.segment >= 0] == [
        ("raw", -1, 0),
        ("exchange", -1, 2),
        ("exchange", -1, 2),
        ("staging", -1, 3),
    ]
    assert regions.root_segments(_program(bound)) is not None
    # Block ids: loops 2, 3, 4 -> axes 0..2, lead 3, rdim 4.
    resolved = [32, 32, 128, 1, 64]
    layout = regions.region_layout(items, facts.loops, resolved, 2)
    assert layout.ring_bytes == {0: 16384, 1: 16384, 2: 16384}
    # Loop 1's frame (two 16 KB slots + 16 KB of staging) sets the 48 KB
    # extent; the raw tile sits at 32 KB, the exchange pair at 40 / 32 KB,
    # dL's staging at 40 KB; then the bf16 tile and the scratch.
    assert layout.offsets[4] == 32768
    assert (layout.offsets[8], layout.offsets[9]) == (40960, 32768)
    assert layout.offsets[13] == 40960
    assert layout.offsets[3] == 49152 and layout.total == 59392
    body = _kernel_body(_source(bound, _config(32, 32, 128, warps=4, stages=2)))
    assert "alloc_smem(cutlass.Uint8, 59392, alignment=1024)" in body
    prologue = body[: body.index("for md_i in")]
    # The raw tile lies above loop 0's ring (slots at 0 .. 32 KB), so the
    # ring prologue still goes out ahead of the conversion.
    assert prologue.index(f"{P}.mbarrier_arrive_expect_tx(md_full0") < prologue.index(
        f"{RT}.mbar_wait(md_sfr, 0)"
    )
    assert (
        f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + 32768)" in prologue
    )
    assert sorted(set(re.findall(r"sts_v4_b32\(md_s0 \+ (\d+)", body))) == [
        "32768",
        "40960",
    ]


def test_a_root_drain_syncs_before_the_next_loops_prologue() -> None:
    # A root store between two loops stages through a buffer overlaying
    # the frames: its drain (every thread's lds / stg) completes behind a
    # CTA barrier before the next loop's prologue TMA lands on the frame.
    bf16 = torch.bfloat16
    a = torch.empty((16, 64, 128), dtype=bf16, device=CPU_DEVICE)
    b = torch.empty((16, 128, 64), dtype=bf16, device=CPU_DEVICE)
    bound = _two_loops_store_between.bind((a, b, a, b))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert [(i.kind, i.frame, i.copies, i.segment) for i in facts.tcgen05.items] == [
        ("tile", 0, 0, -1),
        ("tile", 0, 0, -1),
        ("tile", 1, 0, -1),
        ("tile", 1, 0, -1),
        ("staging", -1, 1, 1),
        ("staging", -1, 1, 2),
    ]
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    # Two 16 KB ring slots per loop (sibling frames overlap); both 8 KB
    # staging buffers overlay them.
    assert "alloc_smem(cutlass.Uint8, 32768, alignment=1024)" in body
    between = body[body.index("md_rk0 = md_rk0 + 1") : body.index("for md_i1 in")]
    _assert_in_order(
        between,
        [
            f"{P}.cp_async_bulk_tensor_global_shared_cta(",
            f"{P}.cp_async_bulk_commit_group()",
            f"{P}.cp_async_bulk_wait_group(0, read=True)",
            "cute.arch.barrier()",
            f"{P}.mbarrier_arrive_expect_tx(md_full1 + md_rk1 % 2",
        ],
    )
    assert between.index(
        f"{P}.cp_async_bulk_wait_group(0, read=True)"
    ) < between.rindex("cute.arch.barrier()")


def test_a_raw_tile_over_the_ring_settles_before_the_prologue() -> None:
    # A hoisted fp32 operand whose raw tile (16 KB) is larger than the loop
    # frame (a 2 KB ring slot and 2 KB of staging per copy): the raw tile
    # overlays the ring, so its conversion (wait, pass, barrier) precedes
    # the ring prologue issue.
    a32 = torch.empty((16, 64, 64), dtype=torch.float32, device=CPU_DEVICE)
    b = torch.empty((16, 64, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _hoisted_raw_over_a_narrow_ring.bind((a32, b))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    items = facts.tcgen05.items
    assert [(i.kind, i.frame, i.copies, i.segment) for i in items] == [
        ("tile", -1, 1, -1),
        ("raw", -1, 1, 0),
        ("tile", 0, 0, -1),
        ("staging", 0, 2, -1),
    ]
    resolved = [16, 1, 64]
    layout = regions.region_layout(items, facts.loops, resolved, 2)
    assert layout.ring_bytes == {0: 2048}
    assert layout.offsets == (16384, 0, 0, 4096) and layout.total == 24576
    body = _kernel_body(_source(bound, _config(16, warps=4, stages=2)))
    prologue = body[: body.index("for md_i in")]
    _assert_in_order(
        prologue,
        [
            f"{P}.mbarrier_arrive_expect_tx(md_sfr, 16384)",
            f"{RT}.mbar_wait(md_sfr, 0)",
            f"{RT}.lds_v4_f32(md_s0 + 0",
            f"{WARP_RT}.sts_v2_b32(md_s0 + 16384",
            "cute.arch.barrier()",
            f"{P}.mbarrier_arrive_expect_tx(md_full0 + md_rk0 % 2",
        ],
    )
    assert prologue.count(f"{P}.mbarrier_arrive_expect_tx(md_full0") == 1


def test_tmem_conflicts_follow_the_batches_not_the_segments() -> None:
    # Beyond the live groups, a slot conflicts only with the result of a
    # dot that may join an earlier dot's batch (independent of it) over
    # everything live between the two: wy_dL's root chain dot(T, dAinv) ->
    # dA depends step by step, so dAinv, the chain's results and kk share
    # columns -- 128 columns instead of 256 (four CTAs per SM by TMEM);
    # gated wy_dL 128 as well (dLL and the loop-1 product stay in
    # registers).  bwd_o_kda's dA (slot 0, read by the
    # work between its dots) and dv_new (slot 2, independent of dot 1)
    # still conflict: that batch reorder was the 50 %-wrong hazard.
    bound = _bind_wy_dl()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.tmem_conflicts == ((0, 1), (0, 2))
    body = _kernel_body(_source(bound, _config(64, 32, 128, warps=4, stages=2)))
    assert f"{P}.tcgen05_alloc(md_tslot, 128," in body
    assert _tmem_columns(body) == {0: 0, 1: 64, 2: 64, 3: 0, 4: 0, 5: 64}
    body = _kernel_body(
        _source(_bind_wy_dl(gated=True), _config(64, 32, 128, warps=4, stages=2))
    )
    assert f"{P}.tcgen05_alloc(md_tslot, 128," in body
    # ((0, 1) and (3, 4) come from the cross-loop rule: dA is read in the
    # inner loop that does not define it.)
    facts = _bind_bwd_o_kda().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.tmem_conflicts == (
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
    )
    program = _match(_bind_bwd_o_kda())
    dots = [dot for dot in program.dots if dot.loop == 0]
    assert [dot.slot for dot in dots] == [1, 2]
    assert [step.slot for step in program.derived if step.loop == 0] == [0]


def test_flat_streaming_region_programs_lead_with_tensor_memory() -> None:
    # The promoted class: loops, no root tile axis, and every fragment
    # operand inside a loop a hoisted root exchange (a root fragment whose
    # tile is written once ahead of the loop).  dk, wy_dL (both decay
    # modes) and dqkw (both: dattn is hoisted ahead of the tile_d loop) are
    # in it; bwd_o_kda's dA is exchanged every iteration, state_du_kda
    # reads its carry as an operand, fwd_o_anch has a root tile.  Measured
    # (B200, warp default vs first tensor-memory seed): gated dqkw s1 199.8
    # vs 186.7 us, s2 729.5 vs 695.3; plain dqkw 147.3 vs 152.7, 558.2 vs
    # 572.3 (the warp seeds stay in the search).
    from helion._compiler.autotuner_heuristics.cute_mma_dag import CuteMmaDagHeuristic

    # The predicate is exactly the three clauses: probe_carry2's loop-1 dot
    # reads the root fragment y, exchanged once ahead of the loop (a
    # hoisted edge) -- streaming; probe_hoist_loop's loop dots read the
    # loop's own fragments (per-iteration edges) -- not; a program without
    # loops is not streaming either (no matched program is loop-free
    # today).
    expected = {
        "dk": True,
        "wy_dL": True,
        "wy_dL_gated": True,
        "dqkw": True,
        "dqkw_gated": True,
        "bwd_o_kda": False,
        "state_du_kda": False,
        "fwd_o_anch": False,
        "probe_hoist": True,
        "probe_kk": True,
        "probe_rawlate": True,
        "probe_h4": True,
        "probe_carry2": True,
        "probe_h1": True,
        "probe_h3": True,
        "probe_hoist_loop": False,
    }
    for name, streaming in expected.items():
        spec = _REGION_BINDERS[name]().config_spec
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        assert CuteMmaDagHeuristic.streaming_regions(facts) == streaming, name
        first = "tcgen05" if streaming else "warp"
        assert spec.default_config().config["cute_mma_dag"] == first, name
        modes = [c.config["cute_mma_dag"] for c in spec.compiler_seed_configs]
        assert modes[0] == first and "tcgen05" in modes, name
    facts = _REGION_BINDERS["probe_carry2"]().config_spec.cute_mma_dag_facts
    assert facts is not None
    assert [(edge.region, edge.in_loop, edge.hoisted) for edge in facts.edges] == [
        ("epilogue", True, True)
    ]
    facts = _REGION_BINDERS["dqkw_gated"]().config_spec.cute_mma_dag_facts
    assert facts is not None
    assert [(edge.in_loop, edge.hoisted) for edge in facts.edges] == [(True, True)] * 2
    assert any(loop.parent >= 0 for loop in facts.loops)
    facts = _REGION_BINDERS["bwd_o_kda"]().config_spec.cute_mma_dag_facts
    assert facts is not None
    assert [(edge.region, edge.in_loop, edge.hoisted) for edge in facts.edges] == [
        ("loop", True, False)
    ]
    facts = _REGION_BINDERS["state_du_kda"]().config_spec.cute_mma_dag_facts
    assert facts is not None
    assert facts.edges and not any(edge.hoisted for edge in facts.edges)
    facts = _REGION_BINDERS["dk"]().config_spec.cute_mma_dag_facts
    assert facts is not None
    assert not CuteMmaDagHeuristic.streaming_regions(
        dataclasses.replace(facts, loops=())
    )
    facts = _REGION_BINDERS["fwd_o_anch"]().config_spec.cute_mma_dag_facts
    assert facts is not None
    assert any(axis.kind == dag.AXIS_ROOT for axis in facts.axes)
    facts = _REGION_BINDERS["state_du_kda"]().config_spec.cute_mma_dag_facts
    assert facts is not None
    assert all(loop.parent < 0 for loop in facts.loops)
    assert any(edge.region == "loop" for edge in facts.edges)


def test_the_hardware_order_checker_catches_shared_columns() -> None:
    # The checker is independent of the rules it guards, so a wrong rule
    # shows up as a hazard in the render.  (a) bwd_o_kda without the (1, 2)
    # batch conflict: dv_new (slot 2) joins the dA dot's batch and takes
    # slot 1's columns, which the deferred work then reads -- the 50 %-wrong
    # hazard.  (b) probe_hoist with w placed on s0's column: the hoisted
    # tcgen05.st lands before a's tcgen05.ld of s0, which the previous
    # live-pairs checker (dot results at the batch heads, derived results at
    # their steps) missed.
    # (c) wy_dL with d_bv (slot 1) on dAinv's (slot 0) columns: loop 0's
    # accumulate reads d_bv's writes.  Each program is clean unmutated.
    original_conflicts = regions.tmem_conflicts

    def dropped(program: dag.MmaDagProgram) -> tuple[tuple[int, int], ...]:
        return tuple(pair for pair in original_conflicts(program) if pair != (1, 2))

    bound = _bind_bwd_o_kda()
    program = _match(bound)
    sizes = [64, 64, 32]
    body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
    assert _tmem_hazards(body, _slot_spans(body, bound, program, sizes)) == []
    with patch.object(regions, "tmem_conflicts", dropped):
        bound = _bind_bwd_o_kda()
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        assert (1, 2) not in facts.tcgen05.tmem_conflicts
        body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
    hazards = _tmem_hazards(body, _slot_spans(body, bound, program, sizes))
    assert any(
        "tcgen05.ld of slot 1 reads columns last written by slot(s) [2]" in hazard
        for hazard in hazards
    ), hazards
    original_layout = regions.region_tmem_layout
    for binder, sizes, moved, onto, expected in (
        (_bind_probe_hoist, [64, 128], 3, 0, "tcgen05.ld of slot 0 reads"),
        (_bind_wy_dl, [64, 32, 128], 1, 0, "the MMA accumulate of slot 0 reads"),
    ):
        bound = binder()
        program = _match(bound)
        body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
        assert _tmem_hazards(body, _slot_spans(body, bound, program, sizes)) == []

        def collapsed(
            facts: dag.CuteMmaDagFacts,
            resolved: Sequence[int],
            moved: int = moved,
            onto: int = onto,
        ) -> tuple[dict[int, int], int]:
            columns, total = original_layout(facts, resolved)
            columns = dict(columns)
            columns[moved] = columns[onto]
            return columns, total

        with patch.object(regions, "region_tmem_layout", collapsed):
            body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
        columns = _tmem_columns(body)
        assert columns[moved] == columns[onto]
        hazards = _tmem_hazards(body, _slot_spans(body, bound, program, sizes))
        assert any(expected in hazard for hazard in hazards), hazards
    # The trace: loops unroll at least once and at most three times, marks
    # and commit waits are leaves, branches are transparent.
    trace = _tmem_trace(
        "x = 1\n"
        "for md_i in cutlass.range(0, 8):\n"
        "    if md_epi:\n"
        "        md_tm_mark = 'ld 2 0'\n"
        "    md_tm_mark = 'mma 1 1'\n"
        "    _rt.mbar_wait(md_cb, md_cn + 1 & 1)\n"
        "for md_j in cutlass.range(0, md_n):\n"
        "    md_tm_mark = 'st 0 1'\n"
    )
    assert trace == [
        ("loop", 1, 3, [("ld", 3, 2, 0), ("mma", 4, 1, 1), ("wait", 5, 0, 0)]),
        ("loop", 6, 2, [("st", 7, 0, 1)]),
    ]
    spans = {0: (0, 64), 1: (0, 64), 2: (64, 64)}
    assert _tmem_hazards("md_tm_mark = 'mma 1 0'\nmd_tm_mark = 'ld 2 0'\n", spans) == []
    # Values: a read wanting the value a later definition replaced (one
    # logical slot, two values: a derived value recomputed from columns
    # another value took) and an accumulate onto another value are
    # hazards; a read of the slot's own in-flight MMA is one too (an
    # accumulate chains).
    assert _tmem_hazards(
        "md_tm_mark = 'def 1 7'\nmd_tm_mark = 'use 1 7'\nmd_tm_mark = 'def 1 8'\n"
        "md_tm_mark = 'use 1 7'\nmd_tm_mark = 'mma 1 1 9 7'\n",
        spans,
    ) == [
        "line 3: the read of slot 1 wants value 7 but the columns hold value(s) [8]",
        (
            "line 4: the MMA accumulate of slot 1 wants value 7 but the columns hold "
            "value(s) [8]"
        ),
    ]
    assert _tmem_hazards(
        "md_tm_mark = 'mma 1 0 5 -1'\nmd_tm_mark = 'mma 1 1 5 5'\nmd_tm_mark = 'ld 1 0'\n",
        spans,
    ) == [
        "line 2: tcgen05.ld of slot 1 under the MMA into slot 1 (line 0)",
        "line 2: tcgen05.ld of slot 1 under the MMA into slot 1 (line 1)",
    ]
    assert _tmem_hazards(
        "md_tm_mark = 'mma 1 0'\nmd_tm_mark = 'st 0 0'\n"
        "_rt.mbar_wait(md_cb, 0)\nmd_tm_mark = 'ld 1 0'\n",
        spans,
    ) == [
        "line 1: tcgen05.st of slot 0 under the MMA into slot 1 (line 0)",
        "line 3: tcgen05.ld of slot 1 reads columns last written by slot(s) [0]",
    ]


@pytest.mark.parametrize("kernel", sorted(_REGION_BINDERS))
def test_region_programs_pass_the_hardware_order_check_at_random_configs(
    kernel: str,
) -> None:
    # Beyond the seeds (checked with the commit order): four random legal
    # tensor-memory configs per region program (power-of-two block sizes,
    # 4 / 8 warps, 1-4 stages, both loads modes) pass the hardware-order
    # check.
    bound = _REGION_BINDERS[kernel]()
    spec = bound.config_spec
    program = _match(bound)
    rng = random.Random(kernel)
    rendered = 0
    for _attempt in range(24):
        if rendered == 4:
            break
        sizes = []
        for block in spec.block_sizes:
            low, high = max(block.min_size, 16), block.max_size
            choices = [
                1 << i for i in range(4, high.bit_length()) if low <= 1 << i <= high
            ]
            sizes.append(rng.choice(choices or [low]))
        config = helion.Config(
            block_sizes=sizes,
            cute_mma_dag="tcgen05",
            cute_mma_dag_warps=rng.choice([4, 8]),
            cute_mma_dag_stages=rng.choice([1, 2, 3, 4]),
            cute_mma_dag_loads=rng.choice(["tma", "lane"]),
        )
        try:
            with bound.env:
                normalized = spec.normalized_config(config)
            if normalized.config["cute_mma_dag"] != "tcgen05":
                continue
            body = _instrumented_body(bound, normalized)
        except (InvalidConfig, BackendUnsupported):
            continue
        rendered += 1
        _assert_tmem_hardware_order(
            bound, program, body, normalized.config["block_sizes"]
        )
    assert rendered >= 1, kernel


def test_ctas_per_sm_follows_shared_and_tensor_memory() -> None:
    capacity = 232448
    assert dag.mma_dag_ctas_per_sm(75856, capacity) == 3
    assert dag.mma_dag_ctas_per_sm(76800, capacity) == 3
    assert dag.mma_dag_ctas_per_sm(76801, capacity) == 2
    assert dag.mma_dag_ctas_per_sm(106544, capacity) == 2
    assert dag.mma_dag_ctas_per_sm(57344, capacity) == 4
    assert dag.mma_dag_ctas_per_sm(57344, capacity, tmem_cols=256) == 2
    assert dag.mma_dag_ctas_per_sm(232448, capacity, tmem_cols=128) == 1
    assert dag.mma_dag_ctas_per_sm(200000, 0, tmem_cols=64) == 8
    assert dag.mma_dag_ctas_per_sm(8192, 0) == 32


def test_block_local_derived_values_over_fragments_stay_in_registers() -> None:
    # A derived value over fragments read only within its block keeps no
    # TMEM columns: gated wy_dL's dLL (read by its two reductions after
    # loop 2) and the loop-1 product (read by its reduction in the same
    # block); its peak live set is dA + kk -> 128 columns.  A reader behind
    # a dot of the segment (deferred into a later block) or in another
    # region keeps the value in tensor memory.
    bound = _bind_wy_dl(gated=True)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    local = [
        step.node.name
        for step in program.derived
        if step.slot not in facts.tcgen05.tmem_slots
    ]
    assert len(local) == 2 and all(
        step.frags for step in program.derived if step.node.name in local
    )
    assert facts.tcgen05.tmem_slots == (0, 1, 2, 4, 5, 6)
    body = _kernel_body(_source(bound, _config(64, 32, 128, warps=4, stages=2)))
    assert f"{P}.tcgen05_alloc(md_tslot, 128," in body
    # The reductions read the register rows: the unplaced slots' pointers
    # are defined (every slot gets one) but never loaded or stored.
    assert body.count("md_acc7") == 1 and body.count("md_acc3") == 1
    # dqkw's gated row sums over dq * q read dq (a dot result) through a
    # register-only product as well; wy's A = where(...) over kk seeds a
    # carry and stays in tensor memory.
    facts = _bind_dqkw(gated=True).config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.tmem_slots == (0, 1, 2, 3, 4, 5)
    facts = _bind_wy().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(_bind_wy())
    assert all(step.slot in facts.tcgen05.tmem_slots for step in program.derived)


def test_aux_reads_share_a_ring_operands_tile() -> None:
    # wy_dL's reductions read v (loop 0) and k (loop 1), the very tiles the
    # beta-scaled B operands of dAinv land by TMA and rewrite in place: the
    # reads take the operand tile (no aux item, no extra TMA, 8 KB less per
    # ring slot) and a CTA barrier parts them from the rewrite.  dk's seed
    # tiles are nobody's operands and keep their aux items.
    bound = _bind_wy_dl()
    program = _match(bound)
    shared = [
        (step.node.name, access.name)
        for step in program.derived
        for access in step.loads
        if regions.aux_shared_operand(program, step, access) is not None
    ]
    assert shared == [("mul_2", "v"), ("mul_1", "k")]
    assert regions.aux_tile_loads(program) == []
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.aux_items == 0
    assert "aux" not in {item.kind for item in facts.tcgen05.items}
    body = _kernel_body(_source(bound, _config(32, 32, 128, warps=4, stages=2)))
    assert "'tma_count': 9" in body
    loop0 = body[body.index("for md_i in") : body.index("for md_i1 in")]
    # v's operand tile sits 12 KB into loop 0's 16 KB slot; the reads
    # precede a barrier that precedes the in-place pass's first store.
    _assert_in_order(
        loop0,
        [
            f"{RT}.mbar_wait(md_full0 + md_st0, md_rd0)",
            f"{WARP_RT}.lds_v4_b32(md_s0 + 0 + md_st0 * 16384 + 12288 + md_row64 * 64",
            "cute.arch.barrier()",
            f"{WARP_RT}.sts_v4_b32(",
        ],
    )
    assert f"{WARP_RT}.ldg_b32(" not in loop0
    facts = _bind_dk().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.aux_items == 2


def test_staged_stores_drain_by_tma() -> None:
    # A staging tile whose rows are one swizzle atom holds TMA's layout: the
    # writers fence their packets to the async proxy, and behind the CTA
    # barrier one thread stores the tile by TMA as one bulk group instead
    # of every thread draining 16-byte packets.  The store thread waits for
    # its groups' reads ahead of every commit (the stage-in reusing a
    # buffer two iterations later waits for a later commit), ahead of a
    # loop's exit barrier (the root segment's overlaid items) and, fully,
    # before the kernel ends.  Rows wider than an atom keep the packet drain.
    body = _kernel_body(_source(_bind_dk(), _config(64, warps=4, stages=2)))
    loop = _loop(body)
    iteration = loop[: loop.index("md_rk0 = md_rk0 + 1")]
    assert iteration.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1
    assert iteration.count(f"{P}.cp_async_bulk_wait_group(0, read=True)") == 1
    assert f"{WARP_RT}.stg_v4_b32(" not in body
    tail = body[body.rindex("md_rk0 = md_rk0 + 1") :]
    _assert_in_order(
        tail,
        [
            "cute.arch.barrier()",
            f"if md_tidx == {tc.TMA_STORE_THREAD}:",
            f"{P}.cp_async_bulk_wait_group(0)",
            f"{P}.tcgen05_dealloc(",
        ],
    )
    # wy_dL at [64, 64, 64]: dv_out (bf16, 128-byte rows) and dL_out by TMA,
    # d_ba_out (fp32, 256-byte rows) as packets; the loop-exit wait.
    body = _kernel_body(_source(_bind_wy_dl(), _config(64, 64, 64, warps=4, stages=2)))
    loop0, rest = _split_loop(_split_loop(body, "md_i")[1], "md_i1")
    loop1, rest = _split_loop(rest, "md_i2")
    assert loop0.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1
    assert f"{WARP_RT}.stg_v4_b32(" not in loop0
    assert loop1.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 0
    assert loop1.count(f"{WARP_RT}.stg_v4_b32(") == 8
    exit0 = loop0[loop0.rindex("md_rk0 = md_rk0 + 1") :]
    _assert_in_order(
        exit0,
        [f"{P}.cp_async_bulk_wait_group(0, read=True)", "cute.arch.barrier()"],
    )
    assert rest.count(f"{P}.cp_async_bulk_tensor_global_shared_cta(") == 1


def test_aux_reads_share_a_staged_operands_tile() -> None:
    # Gated dqkw's row sums over dq * q and dk_as * k (the tile_d loop's
    # tail, behind the nested loop) read q_d and k_d, the staged B operands
    # of that tail's dq and dk_as dots: the reads take the staged tile (no
    # per-lane global loads of the tile, no aux item) once the frame's
    # barrier has landed it, which the dots' settle waited for already;
    # the end-of-iteration barrier keeps the next stage-in behind the
    # reads.  The ``hd`` loop's two tiles are no operand's and keep their
    # aux items; plain dqkw and dk share nothing.
    bound = _bind_dqkw(gated=True)
    program = _match(bound)
    shared = [
        (step.node.name, access.name, found[1].load.node.name)  # type: ignore[union-attr]
        for step in program.derived
        for access in step.loads
        if (found := regions.aux_shared_operand(program, step, access)) is not None
    ]
    assert shared == [
        ("mul_2", "q", "q_d"),
        ("mul_3", "k", "k_d"),
        ("mul_4", "k", "k_d"),
    ]
    assert [(d.node.name, a.name) for d, a in regions.aux_tile_loads(program)] == [
        ("mul", "h"),
        ("mul", "dS_future"),
    ]
    for bind in (_bind_dqkw, _bind_dk):
        other = _match(bind())
        assert not any(
            regions.aux_shared_operand(other, step, access) is not None
            for step in other.derived
            for access in step.loads
        )
    body = _kernel_body(_source(bound, _config(64, 32, 16, 64, 16, warps=4, stages=2)))
    # No bf16 tile is loaded per lane anywhere; the tail reads q_d (two
    # chunks) and k_d (four) at the staged tiles' offsets, behind the
    # frame's wait and the dots' commit wait, ahead of the exit barrier.
    assert f"{WARP_RT}.ldg_b32(" not in body
    tail = body[body.index("md_rk2 = md_rk2 + 1") : body.index("for md_i3 in")]
    staged = re.findall(
        rf"{re.escape(WARP_RT)}\.lds_v4_b32\(md_s0 \+ (\d+) \+ md_row64", tail
    )
    assert staged == ["28672", "28672", "24576", "24576", "24576", "24576"]
    for offset in ("24576", "28672"):
        assert (
            f"{P}.cp_async_bulk_tensor_shared_cta_global(md_smem + (0 + {offset}), "
            in body
        )
    assert tail.count(f"{RT}.mbar_wait(md_sf1, md_rk1 & 1)") == 1
    _assert_in_order(
        tail,
        [
            f"{RT}.mbar_wait(md_sf1, md_rk1 & 1)",
            f"{P}.tcgen05_commit(md_cb",
            f"{RT}.mbar_wait(md_cb,",
            f"{WARP_RT}.lds_v4_b32(md_s0 + 28672 + md_row64",
            f"{WARP_RT}.lds_v4_b32(md_s0 + 24576 + md_row64",
            "cute.arch.barrier()",
            "md_rk1 = md_rk1 + 1",
        ],
    )


def test_root_fragment_exchanges_are_hoisted_ahead_of_the_loop() -> None:
    # dqkw's dq and dk_as (the tile_d loop's tail) read where(causal, dattn,
    # 0).to(bf16) -- gated: * L, L = exp2 of the pairwise g_cs differences
    # -- a root value; the exchange tile is written once between loop 0 and
    # the tile_d loop (``emit_hoisted_exchanges``: the tensor-memory read,
    # L's 32 exp2 per thread, four packets, fence, barrier), never inside
    # it, so dattn's columns are free for the tile_d loop's slots (TMEM 128
    # instead of 256 columns at bd = 32).  wy's T (tf32) feeds the dots of
    # loops 2 and 3 through one fp32 tile written ahead of loop 2.
    for gated, exp2s in ((False, 0), (True, 32)):
        bound = _bind_dqkw(gated=gated)
        program = _match(bound)
        hoisted = [
            (
                dot.node.name,
                operand.role,
                regions.hoisted_exchange_loop(program, dot, operand),
            )
            for dot in program.dots
            for operand in (dot.a, dot.b)
            if operand.frag is not None
        ]
        assert hoisted == [("dq", "a", 1), ("dk_as", "a", 1)]
        sizes = [64, 32, 16, 64, 16] if gated else [64, 32, 16]
        body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
        assert f"{P}.tcgen05_alloc(md_tslot, 128," in body
        before, rest = _split_loop(body, "md_i1")
        between = before[before.rindex("md_rk0 = md_rk0 + 1") :]
        assert between.count("cute.math.exp2(") == exp2s
        assert between.count(f"{P}.tcgen05_ld(") == 1
        assert between.count(f"{WARP_RT}.sts_v4_b32(") == 4
        loop1 = rest[: rest.index("for md_i3 in")] if gated else rest
        assert "cute.math.exp2(" not in loop1.split("for md_i2 in")[0]
        assert f"{P}.tcgen05_ld(" not in loop1.split("for md_i2 in")[0]
        _assert_tmem_hardware_order(bound, program, body, sizes)
    bound = _bind_wy()
    program = _match(bound)
    (t_slot,) = {
        program.frag_slots[operand.frag]
        for dot in program.dots
        for operand in (dot.a, dot.b)
        if operand.frag is not None
        and regions.hoisted_exchange_loop(program, dot, operand) is not None
    }
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    columns = _tmem_columns(body)
    before, rest = _split_loop(body, "md_i2")
    read = f"md_tmem + {columns[t_slot]},"
    assert before.count(read) >= 1 and read not in rest
    # No exchange store (row-indexed) in loops 2 / 3: their fp32 packet
    # stores are the widen pass's (pass-indexed) alone.
    assert not re.search(rf"{RT}\.sts_v4_f32\([^\n]*md_row64", rest)
    assert f"{P}.tcgen05_alloc(md_tslot, 128," in body and max(columns.values()) == 64


def _tmem_ptr(column: int) -> str:
    return f"cutlass.inttoptr(md_warp % 4 * 32 << 16 | md_tmem + {column},"


def test_a_hoisted_derived_value_keeps_off_the_batch_dots_dying_operands() -> None:
    # The block builder sinks a batch to the end of its block and hoists
    # the independent work after its first dot ahead of it, so a derived
    # value written there lands in tensor memory before the batch's
    # exchanges and rescales read their sources.  probe_hoist:
    # s0 (the loop-0 carry) dies at a = dot((s0 * 2).bf16, m1.bf16); w = m1
    # * x is independent of a, so its tcgen05.st precedes a's tcgen05.ld of
    # s0 in one block.  The batch rule now moves derived values like later
    # dots: w conflicts with s0 (the pair once shared column 0 and a was
    # fully wrong on the GPU).  The same inside a loop body
    # (probe_hoist_loop: s dies at acc_a's exchange, w = m * x is hoisted).
    bound = _bind_probe_hoist()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    (w,) = program.derived
    (s0,) = [carry.slot for carry in program.loops[0].carries]
    (a,) = [
        dot for dot in program.dots if dot.a.frag is not None and dot.b.frag is not None
    ]
    assert w.slot not in (s0, a.slot)
    assert (s0, w.slot) in facts.tcgen05.tmem_conflicts
    assert (min(w.slot, a.slot), max(w.slot, a.slot)) in facts.tcgen05.tmem_conflicts
    body = _kernel_body(_source(bound, _config(64, 128, warps=4, stages=2)))
    columns = _tmem_columns(body)
    assert columns[w.slot] not in (columns[s0], columns[a.slot])
    assert f"{P}.tcgen05_alloc(md_tslot, 256," in body
    before, _rest = _split_loop(body, "md_i1")
    between = before[before.rindex("md_rk0 = md_rk0 + 1") :]
    # The hazard's shape in the render: w's store ahead of a's read of s0,
    # inside one epilogue block (no commit wait between them).
    store = f"{P}.tcgen05_st('16x32bx2', {_tmem_ptr(columns[w.slot])}"
    read = f"{P}.tcgen05_ld('16x32bx2', {_tmem_ptr(columns[s0])}"
    assert between.count(store) == 1
    block = between[between.index(store) :]
    block = block[: block.index(f"{RT}.mbar_wait(md_cb")]
    assert read in block
    bound = _bind_probe_hoist_loop()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    (w,) = program.derived
    (s,) = [
        dot.slot
        for dot in program.dots
        if dot.loop == 0
        and dot.acc_slot is None
        and dot.a.kind == "load"
        and not dot.a.pointwise
    ]
    assert (min(s, w.slot), max(s, w.slot)) in facts.tcgen05.tmem_conflicts
    body = _kernel_body(_source(bound, _config(32, 64, warps=4, stages=2)))
    columns = _tmem_columns(body)
    assert columns[w.slot] != columns[s]


def test_every_root_raw_tile_converts_before_the_first_loop_body() -> None:
    # The layout overlays a root raw tile on the loop frames (segment 0:
    # "converted in the prologue"), but the tile settled lazily at its first
    # consumer, and the loop guard converted only the tiles the loop's own dots
    # read. probe_rawlate: A's fp32 raw tile (16 KB at 16 KB) is read by a root
    # dot behind loop 0 only, whose ring slot 1 (16 KB slots at 0 / 16 KB)
    # refilled its bytes before the conversion (r fully wrong on the GPU).
    # Every pending root raw tile now converts ahead of the first top-level
    # loop's body: before the prologue issue when it overlays that ring, else
    # behind the issue (the prologue in flight) and ahead of the ``for``.
    bound = _bind_probe_rawlate()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    (r,) = [dot for dot in program.dots if dot.loop < 0]
    assert r.a.load is not None and r.a.load.dtype == torch.float32
    items = facts.tcgen05.items
    assert [(i.kind, i.frame, i.segment) for i in items if i.kind == "raw"] == [
        ("raw", -1, 0)
    ]
    convert = f"{RT}.lds_v4_f32(md_s0 + 16384"
    for sizes, overlays in (([64, 64], True), ([16, 128], False)):
        resolved = _resolved_axes(program, bound, sizes)
        layout = regions.region_layout(items, facts.loops, resolved, 2)
        raw = next(i for i, item in enumerate(items) if item.kind == "raw")
        assert layout.offsets[raw] == 16384
        assert (layout.ring_bytes[0] * 2 > 16384) == overlays
        body = _kernel_body(_source(bound, _config(*sizes, warps=4, stages=2)))
        prologue, _rest = _split_loop(body, "md_i")
        assert convert in prologue
        issue = f"{P}.mbarrier_arrive_expect_tx(md_full0"
        assert prologue.count(issue) == 1
        order = [convert, "cute.arch.barrier()", issue]
        _assert_in_order(prologue, order if overlays else [issue, *order[:2]])
        # No conversion in a loop body or behind loop 0.
        assert body.count(convert) == prologue.count(convert)


def test_a_shared_tensor_map_keeps_each_tiles_slab_count() -> None:
    # A TMA box is one swizzle atom wide, so a 64- and a 128-column tile of one
    # tensor share a tensor map and the shared spec copied the first tile's
    # slab count: the 128-column tile loaded one 64-column slab (wy_dL at [..,
    # 64, 128] failed its own repeat-launch check; a cold autotune rejected
    # twelve configs).  Each tile's spec now carries its own slabs: loop 2's
    # ring issue of the 128-column tile expects both slabs' bytes and issues
    # two boxes through the map loop 1's 64-column tile shares.
    bound = _bind_wy_dl()
    body = _kernel_body(_source(bound, _config(32, 64, 128, warps=4, stages=2)))
    loop2 = body[body.index("for md_i2 in") :]
    assert f"{P}.mbarrier_arrive_expect_tx(md_full2 + (md_rk2 + 1) % 2, 16384)" in loop2
    shared = re.findall(
        r"cp_async_bulk_tensor_shared_cta_global\(md_smem \+ \(0 \+ \(md_rk2 \+ 1\) % 2 "
        r"\* 16384 \+ (\d+)\), (md_tma\d+)\.get_ptr\(\), \[cutlass\.Int32\(\(md_i2 \+ 1\) "
        r"\* 128( \+ 64)?\)",
        loop2,
    )
    assert shared == [("0", "md_tma4", ""), ("8192", "md_tma4", " + 64")]
    loop1 = body[body.index("for md_i1 in") : body.index("for md_i2 in")]
    assert "md_tma4.get_ptr(), [cutlass.Int32((md_i1 + 1) * 64)" in loop1
    # probe_kk's second loop reads one [64, 128] q tile as both operands
    # through the first loop's 64-column map.
    body = _kernel_body(_source(_bind_probe_kk(), _config(64, 128, warps=4, stages=2)))
    loop1 = body[body.index("for md_i1 in") :]
    assert f"{P}.mbarrier_arrive_expect_tx(md_full1 + (md_rk1 + 1) % 2, 16384)" in loop1
    assert loop1.count("cp_async_bulk_tensor_shared_cta_global(") == 2


def test_a_loop_output_in_another_slot_takes_the_carrys_columns() -> None:
    # The TMEM lever of wy's occupancy: Apow = dot(Apow, Apow) lands in a
    # fresh slot that the emitter copied into the carry's slot at every
    # iteration's end, and the carry held its columns throughout the loop.
    # The region liveness (``region_live_slots``) ends such a carry at its
    # last read in the body (the exchange ahead of the squaring MMA) and
    # starts it again at the exit, where the copy would run, so the output
    # takes the carry's columns (``region_tmem_layout`` prefers them) and
    # the copy vanishes; the carries' seed kk is read before either carry is
    # written and shares Apow's columns.  wy: 128 columns instead of 192
    # (256 allocated), both decay modes; the hardware-order checker models
    # the elided copy as a renaming at its point and agrees.
    for gated in (False, True):
        bound = _bind_wy(gated=gated)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        program = _match(bound)
        [(loop, carry, output)] = regions.copied_carries(program)
        assert facts.tcgen05.tmem_aliases == ((output, carry.slot),)
        assert (min(output, carry.slot), max(output, carry.slot)) not in (
            facts.tcgen05.tmem_conflicts
        )
        (kk,) = [c.slot for c in program.loops[0].carries]
        assert (kk, carry.slot) not in facts.tcgen05.tmem_conflicts
        sizes = [64, 64, 64]
        body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
        columns = _tmem_columns(body)
        assert columns[output] == columns[carry.slot] == columns[kk] == 0
        assert f"{P}.tcgen05_alloc(md_tslot, 128," in body
        chain = _split_loop(_split_loop(body, "md_i1")[1], "md_i2")[0]
        assert f"{P}.tcgen05_st(" not in chain
        marks = re.findall(rf"{_TMEM_MARK} = ['\"]alias (\d+) (\d+)['\"]", chain)
        assert marks == [(str(output), str(carry.slot))]
        _assert_tmem_hardware_order(bound, program, body, sizes)
        # The seeds are loaded before any carry is written.
        init = _split_loop(body, "md_i1")[0]
        init = init[init.rindex("md_rk0 = md_rk0 + 1") :]
        assert init.index(f"{P}.tcgen05_ld(") < init.index(f"{P}.tcgen05_st(")
        assert init.count(f"{P}.tcgen05_ld(") == 1
    # A carry whose output is in another slot but which is read after that
    # output's definition keeps its own columns: the checker would flag the
    # alias (the carry's read would see the output's write).
    assert _tmem_hazards(
        "md_tm_mark = 'st 1 0'\n"
        "for md_i in cutlass.range(0, 2):\n"
        "    md_tm_mark = 'ld 1 0'\n"
        "    md_tm_mark = 'mma 3 0'\n"
        "    _rt.mbar_wait(md_cb, 0)\n"
        "    md_tm_mark = 'ld 1 0'\n"
        "    md_tm_mark = 'alias 3 1'\n",
        {1: (0, 64), 3: (0, 64)},
    ) == ["line 5: tcgen05.ld of slot 1 reads columns last written by slot(s) [3]"]


def test_exchange_tiles_of_dependent_batches_share_their_bytes() -> None:
    # The shared-memory lever of wy's occupancy: the chain's four fp32
    # exchange tiles (64 KB) were laid out side by side although the
    # squared Apow's and T's tiles (the accumulate MMA's operands) are
    # written behind the commit wait for the squaring MMA, the last reader
    # of Apow's two tiles.  ``exchange_phases`` groups a loop's exchange
    # tiles by batch phase -- a tile whose every consumer depends on every
    # consumer of the earlier tiles opens the next phase -- and the layout
    # packs every phase from the same offset: loop 1's frame is 32 KB; with
    # the w / u rings at 16 KB per slot (the raw tiles on their fp32 tiles)
    # and 16 KB of staging the kernel takes 64 KB at two stages (three CTAs
    # per SM) and 48 KB at one (four).  Tiles read by several loops (wy's
    # T) or exchanged
    # once ahead of the loop (dqkw's dattn) take no phase; state_du_kda's
    # dv_new exchange (behind its own dot) is a second phase too.
    bound = _bind_wy()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    items = facts.tcgen05.items
    exchanges = [(i.frame, i.segment) for i in items if i.kind == "exchange"]
    assert exchanges == [(1, 0), (1, 0), (1, 1), (1, 1), (-1, -1)]
    phases = regions.exchange_phases(program)
    (apow, t) = [dot for dot in program.dots if dot.loop == 1]
    assert {phases[regions.tile_key(program, o)] for o in (apow.a, apow.b)} == {0}
    assert {phases[regions.tile_key(program, o)] for o in (t.a, t.b)} == {1}
    assert len(phases) == 4
    resolved = _resolved_axes(program, bound, [64, 64, 64])
    layout = regions.region_layout(items, facts.loops, resolved, 2)
    assert layout.offsets[1:5] == (0, 16384, 0, 16384)
    assert layout.total == 65536
    assert regions.region_layout(items, facts.loops, resolved, 1).total == 49152
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=1)))
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=1024)" in body
    assert f"{P}.tcgen05_alloc(md_tslot, 128," in body
    # Both phases' straight tiles are written at offset 0 in one iteration
    # (Apow for the squaring, then the squared Apow for the accumulate);
    # the MN-major B tiles (natural four-wide packets under the 32-byte-unit
    # swizzle) share offset 16384 the same way.
    loop1 = body[body.index("for md_i1 in") : body.index(_after_loop(body, "md_i1"))]
    assert loop1.count(f"{RT}.sts_v4_f32(md_s0 + 0 + md_row64") == 16
    assert loop1.count(f"{RT}.sts_v4_f32(md_s0 + 16384 + md_row64") == 16
    assert f"{RT}.sts_v4_f32(md_s0 + 32768" not in loop1
    # The w loop's widening pass: the four raw loads of a thread, the
    # barrier, then the packet stores (the raw tile lands on the fp32 tile).
    w_loop = _split_loop(_split_loop(body, "md_i2")[1], "md_i3")[0]
    loads = [m.start() for m in re.finditer(rf"{WARP_RT}\.lds_v4_b32\(", w_loop)]
    stores = [m.start() for m in re.finditer(rf"{RT}\.sts_v4_f32\(", w_loop)]
    barrier = w_loop.index("cute.arch.barrier()", loads[0])
    assert len(loads) == 4 and len(stores) == 8
    assert max(loads) < barrier < min(stores)
    assert f"{WARP_RT}.lds_v4_b32(md_s0 + 0 + md_st2 * 16384 + 0 +" in w_loop
    assert regions.exchange_phases(_match(_bind_dqkw(gated=True))) == {}
    bound = _bind_state_du_kda()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert [
        (i.frame, i.segment) for i in facts.tcgen05.items if i.kind == "exchange"
    ] == [
        (0, 0),
        (0, 0),
        (0, 1),
    ]


def test_a_derived_value_whose_reader_is_deferred_crosses_the_block() -> None:
    # d = s0 * x has no dot between it and its store, so it stayed in the
    # registers of its block and was recomputed where read; but xa = a * y
    # reads the pending batch's result a, so the block builder defers xa and,
    # in program order, every step behind it -- sv = s0 * z and the store of d
    # -- into the next block, where sv (which takes s0's slot: s0 dies there)
    # is written before d is recomputed from that slot (d came out as (s0 * z)
    # * x on the GPU).  A reader behind a step reading a result of the
    # segment's dots (or writing a slot the batch reads or produces)
    # crosses the block: d takes tensor-memory columns and is stored in
    # its own block.
    bound = _bind_probe_recompute()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    by_name = {step.node.name: step for step in program.derived}
    d, xa, sv = by_name["d"], by_name["xa"], by_name["sv"]
    assert regions.derived_crosses(program, d)
    assert regions.derived_crosses(program, sv) and regions.derived_crosses(program, xa)
    assert d.slot in facts.tcgen05.tmem_slots
    assert d.slot != sv.slot
    s0 = program.loops[0].carries[0].slot
    assert sv.slot == s0 and d.slot != s0
    sizes = [64, 128]
    body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
    columns = _tmem_columns(body)
    assert columns[d.slot] != columns[sv.slot]
    _assert_tmem_hardware_order(bound, program, body, sizes)
    # d is stored to its columns before sv is (no recomputation).
    between = _split_loop(body, "md_i1")[0]
    between = between[between.rindex("md_rk0 = md_rk0 + 1") :]
    store_d = f"{P}.tcgen05_st('16x32bx2', {_tmem_ptr(columns[d.slot])}"
    store_sv = f"{P}.tcgen05_st('16x32bx2', {_tmem_ptr(columns[sv.slot])}"
    assert between.index(store_d) < between.index(store_sv)


def test_a_carry_read_by_epilogue_work_only_lives_in_registers() -> None:
    # dqkw's dW = dW - dot(dvni, S) (both decay modes) and bwd_o_kda's dA = dA
    # + dot(dob, vb) are carries no dot reads or writes (their readers are the
    # derived value, a store and -- for dA -- the where behind the loop), so
    # the round trip through tensor memory per iteration (tcgen05.ld, the
    # subtraction, tcgen05.st, two waits) buys nothing: the carry lives in the
    # epilogue warps' registers (``register_carries``;
    # ``md_rc{slot}_{chunk}_{j}`` scalars defined at the kernel's start, seeded
    # in the carry init, rewritten by the derived value, read by the store),
    # its slot takes no tensor-memory columns and the render never loads or
    # stores it.  A carry a dot accumulates (wy's kk, state_du_kda's dS) or
    # reads as an operand, or one carried by two loops, stays in tensor memory.
    # The carry's row is live throughout its loop, so a config where another
    # derived value of the loop would overrun the register budget with it keeps
    # the carry in tensor memory (``resident_slots``): the earlier render.
    for binder, slots in (
        (_bind_dqkw, (1,)),
        (lambda: _bind_dqkw(gated=True), (1,)),
        (_bind_bwd_o_kda, (0,)),
        (_bind_wy, ()),
        (_bind_wy_dl, ()),
        (_bind_dk, ()),
        (_bind_state_du_kda, ()),
    ):
        bound = binder()
        program = _match(bound)
        assert regions.register_carries(program) == slots
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        assert [c.slot for c in facts.tcgen05.register_carries] == list(slots)
        assert all(slot in facts.tcgen05.tmem_slots for slot in slots)
    facts = _bind_dqkw().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    resolved = _resolved_axes(_match(_bind_dqkw()), _bind_dqkw(), [64, 64, 64])
    assert tc.resident_slots(facts.tcgen05, resolved) == (1,)
    body = _kernel_body(_source(_bind_dqkw(), _config(64, 64, 64, warps=4, stages=2)))
    # dW's 32 scalars (a 64-row fragment: 16x32bx2, half the 64 columns per
    # thread), defined ahead of the loops; slot 1's pointer is defined (every
    # slot gets one) but never used.
    assert len(set(re.findall(r"md_rc1_\d+_\d+", body))) == 32
    assert body.index("md_rc1_0_0 = cutlass.Float32(0.0)") < body.index("for md_i")
    assert body.count("md_acc1") == 1
    assert regions.region_tmem_columns(facts, resolved) == 192
    # The store behind the inner loop reads the scalars.
    after = _after_loop(body, "md_i2")
    assert re.search(r"sts_v4_f32\(md_srow[^\n]*md_rc1_0_0, md_rc1_0_1", after)
    # The probe: acc (loop 0) is the register carry; t (loop 1, a dot's
    # accumulator) is seeded from acc's scalars into its own columns; the
    # derived value p * x of loop 0 counts acc's row in its register budget.
    bound = _bind_probe_register_carry()
    program = _match(bound)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    (acc,) = program.loops[0].carries
    (t,) = program.loops[1].carries
    assert regions.register_carries(program) == (acc.slot,)
    assert t.slot != acc.slot and t.slot in facts.tcgen05.tmem_slots
    assert t.init_frag is not None
    by_name = {step.node.name: step for step in program.derived}
    px = [step for step in program.derived if step.loop == 0 and step.slot != acc.slot]
    assert len(px) == 1
    # (Every row here is a C-column row: the budget tuples count rows.)
    # p * x's live set with acc's row: three rows, 192 columns at C = 64,
    # within the budget, so acc is resident.
    (carry_facts,) = facts.tcgen05.register_carries
    assert carry_facts.slot == acc.slot
    assert len(carry_facts.live) == 1 and len(carry_facts.live[0]) == 3
    assert program.resolve(acc.output).name in by_name
    sizes = [64, 64]
    assert tc.resident_slots(facts.tcgen05, _resolved_axes(program, bound, sizes))
    body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
    _assert_tmem_hardware_order(bound, program, body, sizes)
    assert _version_errors(bound, _config(*sizes, warps=4, stages=2)) == []
    before1 = _split_loop(body, "md_i1")[0]
    seed = before1[before1.rindex("md_rk0 = md_rk0 + 1") :]
    # t's seed: acc's scalars into a row array, stored to t's columns.
    _assert_in_order(seed, [f"] = md_rc{acc.slot}_0_0", f"{P}.tcgen05_st("])
    assert not re.search(rf"{_TMEM_MARK} = .(ld|st) {acc.slot} ", body)
    for config in _random_tcgen05_configs(bound, "register_carry", 4):
        try:
            errors = _version_errors(bound, config)
        except (InvalidConfig, BackendUnsupported):
            continue
        assert errors == [], (config.config["block_sizes"], errors[:3])
    # The budget probe: T = T + Apow * An is a register-carry candidate, but
    # Apow * An (Apow, An, the product and T's row: four 64-column rows)
    # overruns the budget at the fixed C = 64, so T keeps its columns at every
    # config -- the render has no scalars, stores T, and the config stays legal
    # as before.
    bound = _bind_probe_register_carry_budget()
    program = _match(bound)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    (t_slot,) = regions.register_carries(program)
    (carry_facts,) = facts.tcgen05.register_carries
    assert carry_facts.slot == t_slot and len(carry_facts.live) == 1
    assert len(carry_facts.live[0]) == 4
    for sizes in ([64], [32]):
        resolved = _resolved_axes(program, bound, sizes)
        assert tc.resident_slots(facts.tcgen05, resolved) == ()
        assert dag.mma_dag_config_reason(facts, {1: sizes[0]}, 4, 2, "tcgen05") is None
    body = _instrumented_body(bound, _config(64, warps=4, stages=2))
    assert "md_rc" not in body
    assert re.search(rf"{_TMEM_MARK} = .st {t_slot} ", body)
    _assert_tmem_hardware_order(bound, program, body, [64])
    assert _version_errors(bound, _config(64, warps=4, stages=2)) == []


def test_the_checker_sees_a_value_recomputed_from_a_rewritten_slot() -> None:
    # The value marks: with ``derived_crosses`` cut back to its old rule
    # (a reader behind a dot), probe_recompute's d stays in registers and
    # is recomputed in the deferred block from s0's slot, which sv (same
    # slot) has rewritten -- the column owners agree (one logical slot), the
    # values do not.  The exact placement rule keeps d in tensor memory
    # and the render is clean.
    original = regions.derived_crosses

    def behind_a_dot(program: dag.MmaDagProgram, derived: dag.Derived) -> bool:
        if derived.node.name != "d":
            return original(program, derived)
        return False

    def recomputing_blocks(
        self: regions._Tcgen05RegionEmitter, steps: Sequence[dag.Step]
    ) -> list[regions.Block]:
        # The pre-fix block builder: a register-only value is recomputed
        # in every block that reads it, whatever its inputs (the current one
        # asserts that such a value has no fragment inputs).
        program = self.program
        blocks: list[regions.Block] = []
        register_only: dict[int, dag.Derived] = {}
        for work, dots in regions.region_block_steps(program, steps):
            placed: list[dag.Step] = []
            for step in work:
                reads: set[int] = set()
                if isinstance(step, dag.Derived):
                    reads = {program.frag_slots[frag] for frag in step.frags}
                elif (
                    isinstance(step, (dag.Store, dag.Reduce)) and step.frag is not None
                ):
                    reads = {program.frag_slots[step.frag]}
                for slot in sorted(reads):
                    source = register_only.get(slot)
                    if source is not None and source not in placed:
                        placed.append(source)
                placed.append(step)
                if (
                    isinstance(step, dag.Derived)
                    and step.slot not in self.tc.tmem_slots
                ):
                    register_only[step.slot] = step
            blocks.append((placed, dots))
        return blocks

    with (
        patch.object(regions, "derived_crosses", behind_a_dot),
        patch.object(
            regions._Tcgen05RegionEmitter, "region_blocks", recomputing_blocks
        ),
    ):
        bound = _bind_probe_recompute()
        program = _match(bound)
        (d,) = [step for step in program.derived if step.node.name == "d"]
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        assert d.slot not in facts.tcgen05.tmem_slots
        sizes = [64, 128]
        body = _instrumented_body(bound, _config(*sizes, warps=4, stages=2))
    hazards = _tmem_hazards(body, _slot_spans(body, bound, program, sizes))
    assert any("the read of slot" in h and "wants value" in h for h in hazards), hazards


def test_tma_store_groups_finish_reading_before_a_loop_and_a_buffers_reuse() -> None:
    # Latent races of the TMA drains: (a) a root store flushed behind a
    # segment's last batch, then a loop whose prologue TMA lands on the root
    # staging buffer -- the pre-loop wait used to be emitted only while drains
    # were pending; now every loop entry of a plan with TMA stores waits for
    # the store groups' reads (and a barrier); (b) a dot-less loop with a
    # TMA-drained body store has no commit to wait behind, so its iteration
    # lets all but the previous group finish (wait_group(1, read)) and syncs
    # before the stage-in that reuses the buffer of two iterations ago.
    bound = _bind_probe_drainrace()
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    before, loop1 = _split_loop(body, "md_i1")
    segment = before[before.rindex("md_rk0 = md_rk0 + 1") :]
    # The segment's last batch commits, its drain goes out by TMA behind
    # the commit, and -- although no drain is pending any more -- the loop
    # entry waits for the group's reads and syncs ahead of its prologue.
    _assert_in_order(
        segment,
        [
            f"{P}.tcgen05_commit(md_cb",
            f"{P}.cp_async_bulk_tensor_global_shared_cta(",
            f"{P}.cp_async_bulk_commit_group()",
            f"{P}.cp_async_bulk_wait_group(0, read=True)",
            "cute.arch.barrier()",
        ],
    )
    issue = segment.index(f"{P}.cp_async_bulk_tensor_global_shared_cta(")
    tail = segment[issue:]
    # The wait is the last thing before the loop's prologue issue.
    prologue = tail.index(f"{P}.mbarrier_arrive_expect_tx(md_full1")
    assert tail.rindex(f"{P}.cp_async_bulk_wait_group(0, read=True)") < prologue
    assert tail[:prologue].count("cute.arch.barrier()") >= 1
    assert tail.rindex("cute.arch.barrier()", 0, prologue) > tail.rindex(
        f"{P}.cp_async_bulk_wait_group(0, read=True)", 0, prologue
    )
    bound = _bind_probe_dotless_store()
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    loop1 = body[body.index("for md_i1 in") : body.index(_after_loop(body, "md_i1"))]
    assert f"{P}.tcgen05_commit(md_cb" not in loop1
    _assert_in_order(
        loop1,
        [
            f"{P}.cp_async_bulk_wait_group(1, read=True)",
            "cute.arch.barrier()",
            f"{WARP_RT}.sts_v4_b32(",
            "cute.arch.barrier()",
            f"{P}.cp_async_bulk_tensor_global_shared_cta(",
            f"{P}.cp_async_bulk_commit_group()",
        ],
    )
    assert loop1.count(f"{P}.cp_async_bulk_wait_group(1, read=True)") == 1
    assert loop1.index(f"{P}.cp_async_bulk_wait_group(1, read=True)") < loop1.index(
        f"{WARP_RT}.sts_v4_b32("
    )


def test_region_seeds_skip_the_eight_warp_twin_of_an_unchanged_render() -> None:
    # Only the per-packet work (operand passes, packet drains of staging
    # tiles whose rows are not one swizzle atom) spreads over every warp;
    # a config without it renders the same body at four and eight warps,
    # so its eight-warp seed is a duplicate: state_du_kda's [16] tiles are
    # TMA-direct with TMA-drained 64-byte rows, wy's widen passes make the
    # warp count matter.
    facts = _bind_state_du_kda().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.pass_tiles == 0
    assert not tc.region_warps_matter(facts, {0: 1, 1: 16})
    seeds = _first_tcgen05_seeds(_bind_state_du_kda(), 6)
    assert ([16], 8, 2) not in seeds and ([16], 4, 2) in seeds
    facts = _bind_wy().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.pass_tiles == 2
    assert tc.region_warps_matter(facts, {0: 1, 2: 128, 4: 32, 5: 32})
    assert ([128, 32, 32], 8, 2) in _first_tcgen05_seeds(_bind_wy(), 2)


@pytest.mark.parametrize("kernel", sorted(_REGION_BINDERS))
def test_region_programs_pass_the_value_version_check(kernel: str) -> None:
    # The value-version checker over the column checker's sweep: every region
    # binder at its first two tensor-memory seeds and four random legal configs
    # renders with every read finding the value it expects, every carry slot
    # holding its output at the iteration's end and every body read of an outer
    # value surviving the iteration.
    bound = _REGION_BINDERS[kernel]()
    spec = bound.config_spec
    seeds = [
        config
        for config in spec.compiler_seed_configs
        if config.config["cute_mma_dag"] == "tcgen05"
    ][:2]
    with bound.env:
        configs = [spec.normalized_config(seed) for seed in seeds]
    configs += _random_tcgen05_configs(bound, kernel + "/versions", 4)
    assert configs, kernel
    for config in configs:
        try:
            errors = _version_errors(bound, config)
        except (InvalidConfig, BackendUnsupported):
            continue
        assert errors == [], (kernel, config.config["block_sizes"], errors[:3])


def test_the_value_version_checker_sees_the_recompute() -> None:
    # Independence from the rules: probe_recompute with the old rule and the
    # old recomputing block builder (as in the column checker's mutation
    # test) -- d's recomputation reads s0's slot, which holds sv by then;
    # the current rules render it clean.
    original = regions.derived_crosses

    def behind_a_dot(program: dag.MmaDagProgram, derived: dag.Derived) -> bool:
        if derived.node.name != "d":
            return original(program, derived)
        return False

    def recomputing_blocks(
        self: regions._Tcgen05RegionEmitter, steps: Sequence[dag.Step]
    ) -> list[regions.Block]:
        program = self.program
        blocks: list[regions.Block] = []
        register_only: dict[int, dag.Derived] = {}
        for work, dots in regions.region_block_steps(program, steps):
            placed: list[dag.Step] = []
            for step in work:
                reads: set[int] = set()
                if isinstance(step, dag.Derived):
                    reads = {program.frag_slots[frag] for frag in step.frags}
                elif (
                    isinstance(step, (dag.Store, dag.Reduce)) and step.frag is not None
                ):
                    reads = {program.frag_slots[step.frag]}
                for slot in sorted(reads):
                    source = register_only.get(slot)
                    if source is not None and source not in placed:
                        placed.append(source)
                placed.append(step)
                if (
                    isinstance(step, dag.Derived)
                    and step.slot not in self.tc.tmem_slots
                ):
                    register_only[step.slot] = step
            blocks.append((placed, dots))
        return blocks

    with (
        patch.object(regions, "derived_crosses", behind_a_dot),
        patch.object(
            regions._Tcgen05RegionEmitter, "region_blocks", recomputing_blocks
        ),
    ):
        bound = _bind_probe_recompute()
        errors = _version_errors(bound, _config(64, 128, warps=4, stages=2))
    assert any(
        "derived d input" in error and "holds sv" in error for error in errors
    ), errors
    assert (
        _version_errors(_bind_probe_recompute(), _config(64, 128, warps=4, stages=2))
        == []
    )


def test_stores_of_a_loop_with_nested_loops_stage_through_one_buffer() -> None:
    # A store in a loop with nested loops stages once per outer iteration with
    # a whole inner loop between the drain and the next stage-in, so one buffer
    # serves (``staging_copies``); a store in an innermost loop keeps its two
    # (the next iteration stages in while this one's TMA store reads).  Gated
    # dqkw's dW / dq / dk_as (loop 1, nesting the tile_dv loop): 40 -> 20 KB of
    # staging at [128, 32, 32, 64, 128] s3 (108544 -> 100352 B: loops 0 and 3
    # set the extent at three stages), which lets [128, 64, 32, 64, 128] s2 fit
    # two CTAs per SM and lead the seeds.  Ahead of a block staging a
    # single-buffered, TMA-drained store the store thread waits for its groups'
    # reads and the CTA syncs; wy_dL's innermost-loop stores are unchanged.
    bound = _bind_dqkw(gated=True)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    program = _match(bound)
    staging = [item for item in facts.tcgen05.items if item.kind == "staging"]
    assert [(item.frame, item.copies) for item in staging] == [(1, 1), (1, 1), (1, 1)]
    assert [regions.staging_copies(program, store) for store in program.stores] == [
        1,
        1,
        1,
    ]
    assert not program.loops[1].innermost
    resolved = _resolved_axes(program, bound, [128, 32, 32, 64, 128])
    assert (
        regions.region_layout(facts.tcgen05.items, facts.loops, resolved, 3).total
        == 100352
    )
    body = _kernel_body(
        _source(bound, _config(128, 32, 32, 64, 128, warps=4, stages=3))
    )
    assert "alloc_smem(cutlass.Uint8, 100352, alignment=1024)" in body
    # No staging address flips with the parity any more.
    assert "md_pt1 * " not in body and "(md_pt1) * " not in body
    # The dq store (bf16 [64, 32]: a 64-byte row, TMA-drained) stages in
    # behind the store thread's read wait and a barrier.
    loop1 = body[body.index("for md_i1 in") : body.index(_after_loop(body, "md_i1"))]
    stage = loop1.index(f"{WARP_RT}.sts_v4_b32(md_srow")
    block = loop1[:stage]
    wait = block.rindex(f"{P}.cp_async_bulk_wait_group(0, read=True)")
    assert block.rindex("cute.arch.barrier()") > wait
    assert block.rindex("if md_epi:") > block.rindex("cute.arch.barrier()")
    for gated in (False, True):
        bound = _bind_wy_dl(gated=gated)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.tcgen05 is not None
        program = _match(bound)
        assert [
            regions.staging_copies(program, store)
            for store in program.stores
            if store.loop >= 0
        ] == [2, 2]
