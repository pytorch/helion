"""Codegen, admission and search-surface pins for the ``cute_mma_dag`` warp
mode (the warp-level MMA owner of MMA dataflow programs: the serial state
recurrences and the K-loop + epilogue kernels).

CPU-only (fake CUDA target), like ``test_cute_tcgen05_fixed_chain.py``
whose fixture this file reuses: every test binds a kernel, pins a config and
inspects the generated CuTe source, the config surface or the pure fragment
coordinate model.  Runtime coverage lives in ``test_cute_mma_dag_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import itertools
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401

import helion
from helion._compiler.autotuner_heuristics.cute_mma_dag import CuteMmaDagHeuristic
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute import cutedsl_compat
from helion._compiler.cute.cute_mma_dag_codegen import _float_literal
from helion._compiler.cute.cute_mma_dag_codegen import _Layout
from helion._compiler.cute.cute_mma_dag_tcgen05 import tcgen05_corner_seeds
from helion._compiler.cute.cute_mma_dag_tcgen05 import tcgen05_seed_configs
from helion._compiler.cute.cute_warp_mma_gemm import _swizzled_chunk
from helion._compiler.cute.cute_warp_mma_gemm import warp_mma_warp_layout
from helion._compiler.host_function import SymbolOrigin
from helion._compiler.variable_origin import BlockSizeOrigin
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.cute.launcher import _MmaDagAliasGuard

# The standalone launcher imports cuda-python (its wrapper annotation) and
# cutlass at module level; an environment without them (a CPU-only or a
# Triton-only CI runner) skips this module at collection.
pytest.importorskip("cuda.bindings.driver")
pytest.importorskip("cutlass")

from helion.autotuner.config_fragment import EnumFragment
from helion.runtime.cute.standalone_launcher import _standalone_validate_mma_dag_aliases

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="codegen pins and search-surface checks need the compiled path",
    ),
]

RT = "_helion_mma_dag"
SIMT_FALLBACK = "_cute_grouped_reduce_shared_two_stage"


# --------------------------------------------------------------------------
# Fragment coordinate model
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bm", "bn", "warps"),
    ((16, 8, 1), (64, 64, 4), (64, 64, 8), (128, 128, 4), (32, 16, 2), (16, 64, 2)),
)
def test_accumulator_coordinates_are_a_bijection(bm: int, bn: int, warps: int) -> None:
    layout = warp_mma_warp_layout(bm, bn, warps)
    assert layout is not None
    wm, wn = layout
    wtm, wtn = bm // wm, bn // wn
    seen: list[tuple[int, int]] = []
    for warp in range(warps):
        wmi, wni = warp // wn, warp % wn
        for lane in range(32):
            for am in range(wtm // 16):
                for an in range(wtn // 8):
                    for j in range(4):
                        seen.append(
                            dag.accumulator_coordinates(
                                lane=lane,
                                am=am,
                                an=an,
                                j=j,
                                wmi=wmi,
                                wni=wni,
                                wtm=wtm,
                                wtn=wtn,
                            )
                        )
    assert len(seen) == bm * bn
    assert set(seen) == {(r, c) for r in range(bm) for c in range(bn)}
    # The register budget counts exactly these words per thread.
    assert dag.mma_dag_frag_regs(bm, bn, warps) == len(seen) // (32 * warps)


def _a_fragment(a: list[list[int]], lane: int) -> list[list[int]]:
    """The m16n8k16 A fragment of ``a[m][k]`` (16 x 16): (row g, k 2c..),
    (row g + 8, k 2c..), (row g, k 8 + 2c..), (row g + 8, k 8 + 2c..)."""
    g, c = lane // 4, lane % 4
    return [
        [a[g][2 * c], a[g][2 * c + 1]],
        [a[g + 8][2 * c], a[g + 8][2 * c + 1]],
        [a[g][8 + 2 * c], a[g][8 + 2 * c + 1]],
        [a[g + 8][8 + 2 * c], a[g + 8][8 + 2 * c + 1]],
    ]


def _b_fragment(b: list[list[int]], lane: int) -> list[list[int]]:
    """The B fragments of two adjacent n-atoms of ``b[k][n]`` (16 x 16):
    (k 2c.., n g), (k 8 + 2c.., n g), then the same at n 8 + g."""
    g, c = lane // 4, lane % 4
    return [
        [b[2 * c][g], b[2 * c + 1][g]],
        [b[8 + 2 * c][g], b[8 + 2 * c + 1][g]],
        [b[2 * c][8 + g], b[2 * c + 1][8 + g]],
        [b[8 + 2 * c][8 + g], b[8 + 2 * c + 1][8 + g]],
    ]


@pytest.mark.parametrize("role", ("a", "b"))
@pytest.mark.parametrize("trans", (False, True))
@pytest.mark.parametrize(("mn_base", "k_step"), ((0, 0), (16, 1), (48, 3)))
def test_ldmatrix_addresses_yield_the_mma_fragments(
    role: str, trans: bool, mn_base: int, k_step: int
) -> None:
    mn_extent, k_extent = 64, 64
    # Element identity = mn * 1000 + k; the shared tile stores K along the
    # rows (columns along M / N) when ``trans``.
    if trans:
        tile = [[mn * 1000 + k for mn in range(mn_extent)] for k in range(k_extent)]
    else:
        tile = [[mn * 1000 + k for k in range(k_extent)] for mn in range(mn_extent)]
    addresses = [
        dag.ldmatrix_lane_address(
            role=role, trans=trans, lane=lane, mn_base=mn_base, k_step=k_step
        )
        for lane in range(32)
    ]
    rows = {row for row, _chunk in addresses}
    chunks = {chunk for _row, chunk in addresses}
    tile_rows, tile_cols = (k_extent, mn_extent) if trans else (mn_extent, k_extent)
    assert rows <= set(range(tile_rows)) and chunks <= set(range(tile_cols // 8))
    got = dag.emulate_ldmatrix(tile, trans=trans, addresses=addresses)
    block = [
        [(mn_base + i) * 1000 + 16 * k_step + k for k in range(16)] for i in range(16)
    ]
    for lane in range(32):
        if role == "a":
            expected = _a_fragment(block, lane)
        else:
            b = [[block[n][k] for n in range(16)] for k in range(16)]
            expected = _b_fragment(b, lane)
        assert got[lane] == expected, (role, trans, lane)


@pytest.mark.parametrize("chunks_per_row", (1, 2, 4, 8, 16))
def test_swizzle_permutes_the_chunks_of_every_row(chunks_per_row: int) -> None:
    for row in range(64):
        chunks = {
            dag.swizzled_chunk(row, chunk, chunks_per_row)
            for chunk in range(chunks_per_row)
        }
        assert chunks == set(range(chunks_per_row))
    # The emitted expression is the same function.
    namespace = {"row": 13, "chunk": 5}
    assert eval(_swizzled_chunk("row", "chunk", chunks_per_row), {}, namespace) == (
        dag.swizzled_chunk(13, 5, chunks_per_row)
    )


def test_ldmatrix_rows_of_one_matrix_share_a_bank_group() -> None:
    # Eight consecutive rows at one logical chunk land on eight distinct
    # physical chunks (128-byte rows): the swizzle keeps ldmatrix conflict-free.
    for chunk in range(8):
        physical = {dag.swizzled_chunk(row, chunk, 8) for row in range(8)}
        assert len(physical) == 8


# --------------------------------------------------------------------------
# Config legality on the geometry facts (no graphs)
# --------------------------------------------------------------------------


def _state_facts(
    m: int = 128, n: int = 128, k: int = 64, *, heads: int = 32, capacity: int = 0
) -> dag.CuteMmaDagFacts:
    """The facts of ``chunk_fwd_h_diag_fused``: axes lead (block 0), root M
    (block 1), root N (block 2), rdim K (block 4); one [M, N] carry; two
    ring tiles [K, M] and [K, N]; a staged bf16 snapshot."""
    return dag.CuteMmaDagFacts(
        axes=(
            dag.AxisFacts(0, dag.AXIS_LEAD, heads),
            dag.AxisFacts(1, dag.AXIS_ROOT, m),
            dag.AxisFacts(2, dag.AXIS_ROOT, n),
            dag.AxisFacts(4, dag.AXIS_RDIM, k),
        ),
        frags=((1, 2),),
        tiles=(dag.TileFacts(3, 1, 2, True), dag.TileFacts(3, 2, 2, True)),
        staging=((1, 2, 2),),
        k_axes=(3,),
        smem_capacity=capacity,
        grid_blocks=heads,
    )


def _sizes(bm: int, bn: int) -> dict[int, object]:
    return {0: 1, 1: bm, 2: bn}


@pytest.mark.parametrize(
    ("bm", "bn", "warps", "stages", "reason"),
    (
        (48, 64, 4, 1, "power of two"),
        (64, 64, 3, 1, "cute_mma_dag_warps"),
        (64, 64, 4, 5, "cute_mma_dag_stages"),
        (8, 64, 1, 1, "multiples of 16"),
        (256, 64, 4, 1, "must divide"),
        (128, 128, 1, 1, "registers per thread"),
        (16, 8, 2, 1, "cannot tile"),
    ),
)
def test_config_reasons(bm: int, bn: int, warps: int, stages: int, reason: str) -> None:
    facts = _state_facts()
    message = dag.mma_dag_config_reason(facts, _sizes(bm, bn), warps, stages)
    assert message is not None and reason in message, message
    assert dag.mma_dag_config_reason(facts, _sizes(64, 64), 4, 2) is None


def test_shared_memory_budget_and_k_step() -> None:
    tight = _state_facts(capacity=48 * 1024)
    assert dag.mma_dag_config_reason(tight, _sizes(64, 64), 4, 2) is None
    message = dag.mma_dag_config_reason(tight, _sizes(64, 64), 4, 4)
    assert message is not None and "shared memory" in message
    # A loop (K) tile below one mma.sync step is refused as such.
    loop = dag.CuteMmaDagFacts(
        axes=(
            dag.AxisFacts(0, dag.AXIS_LEAD, 16),
            dag.AxisFacts(1, dag.AXIS_ROOT, 128),
            dag.AxisFacts(2, dag.AXIS_LOOP, 128),
            dag.AxisFacts(3, dag.AXIS_RDIM, 64),
        ),
        frags=((3, 1),),
        tiles=(dag.TileFacts(3, 2, 2, True), dag.TileFacts(2, 1, 2, True)),
        staging=(),
        k_axes=(2,),
        smem_capacity=0,
        grid_blocks=16,
    )
    message = dag.mma_dag_config_reason(loop, {0: 1, 1: 64, 2: 8}, 4, 1)
    assert message is not None and "K extent" in message
    assert dag.mma_dag_config_reason(loop, {0: 1, 1: 64, 2: 16}, 4, 1) is None


def test_repair_reduces_stages_then_warps_then_tiles() -> None:
    facts = _state_facts()
    assert dag.mma_dag_repair(facts, _sizes(64, 64), 4, 2) == (
        {0: 1, 4: 64, 1: 64, 2: 64},
        4,
        2,
    )
    # Over the register budget with one warp: more warps keep the tile.
    sizes, warps, stages = dag.mma_dag_repair(facts, _sizes(128, 128), 1, 1)  # type: ignore[misc]
    assert dag.mma_dag_config_reason(facts, sizes, warps, stages) is None
    assert (sizes[1], sizes[2], warps) == (128, 128, 8)
    # A tile no warp count can hold is halved on its larger side.
    sizes, warps, stages = dag.mma_dag_repair(
        _state_facts(256, 256), {0: 1, 1: 256, 2: 256}, 8, 1
    )  # type: ignore[misc]
    assert (
        dag.mma_dag_config_reason(_state_facts(256, 256), sizes, warps, stages) is None
    )
    assert sizes[1] * sizes[2] < 256 * 256
    # Over the shared-memory capacity: fewer stages first.
    tight = _state_facts(capacity=48 * 1024)
    sizes, warps, stages = dag.mma_dag_repair(tight, _sizes(64, 64), 4, 4)  # type: ignore[misc]
    assert (sizes[1], sizes[2], warps, stages) == (64, 64, 4, 2)
    # An illegal warp count is repaired to the most legal warps.
    sizes, warps, stages = dag.mma_dag_repair(facts, _sizes(16, 8), 2, 1)  # type: ignore[misc]
    assert (sizes[1], sizes[2], warps) == (16, 8, 1)


def test_seeds_rank_cta_count_then_the_squarest_largest_tile() -> None:
    facts = _state_facts()
    seeds = dag.mma_dag_seed_configs(facts)
    assert seeds, seeds
    for sizes, warps, stages in seeds:
        assert dag.mma_dag_config_reason(facts, sizes, warps, stages) is None
    # 32 heads x 2 x 2 tiles of 64x64 reach 128 CTAs; 128x128 does not.
    # Without a capacity every stage depth fits two CTAs per SM: the
    # deepest ring ranks first.
    sizes, warps, stages = seeds[0]
    assert (sizes[1], sizes[2], warps, stages) == (64, 64, 8, 8)
    assert (seeds[1][0][1], seeds[1][0][2], seeds[1][2]) == (64, 64, 4)
    # Two heads: only 16x16 tiles reach 128 CTAs and the parallelism wins.
    sizes, warps, _stages = dag.mma_dag_seed_configs(_state_facts(heads=2))[0]
    assert (sizes[1], sizes[2], warps) == (16, 16, 2)
    # One head: the largest tile wins once nothing reaches 128 CTAs.
    few = _state_facts(heads=1)
    sizes, warps, _stages = dag.mma_dag_seed_configs(few)[0]
    assert (sizes[1], sizes[2], warps) == (128, 128, 8)
    assert dag.mma_dag_seed_warps(16, 8) == 1
    assert dag.mma_dag_seed_warps(32, 16) == 4
    assert dag.mma_dag_seed_warps(16, 64) == 8


# --------------------------------------------------------------------------
# Binders (the linear-attention engine kernels on the fake CUDA target)
# --------------------------------------------------------------------------


def _engine() -> object:
    return importlib.import_module("examples.linear.linear_attention_engine")


def _tensors(
    *, bh: int = 2, n: int = 8, c: int = 64, d: int = 64, dv: int = 64
) -> dict[str, torch.Tensor]:
    bf16, f32 = torch.bfloat16, torch.float32
    return {
        "k": torch.empty((bh, n, c, d), dtype=bf16, device=CPU_DEVICE),
        "v": torch.empty((bh, n, c, dv), dtype=bf16, device=CPU_DEVICE),
        "g_last": torch.empty((bh, n, d), dtype=f32, device=CPU_DEVICE),
        "g_last_s": torch.empty((bh, n), dtype=f32, device=CPU_DEVICE),
        "gc": torch.empty((bh, n, c), dtype=f32, device=CPU_DEVICE),
        # The per-channel decay cumsum of the anchored (full_gla) mode.
        "gc4": torch.empty((bh, n, c, d), dtype=f32, device=CPU_DEVICE),
        "h0": torch.empty((bh, d, dv), dtype=bf16, device=CPU_DEVICE),
        "dh_init": torch.empty((bh, d, dv), dtype=f32, device=CPU_DEVICE),
    }


def _tensors3(
    *, bhn: int = 16, c: int = 64, d: int = 128, dv: int = 128
) -> dict[str, torch.Tensor]:
    bf16, f32 = torch.bfloat16, torch.float32
    return {
        "q": torch.empty((bhn, c, d), dtype=bf16, device=CPU_DEVICE),
        "v": torch.empty((bhn, c, dv), dtype=bf16, device=CPU_DEVICE),
        "h": torch.empty((bhn, d, dv), dtype=bf16, device=CPU_DEVICE),
        "dh": torch.empty((bhn, d, dv), dtype=f32, device=CPU_DEVICE),
        "gcs": torch.empty((bhn, c), dtype=f32, device=CPU_DEVICE),
        "gl": torch.empty((bhn,), dtype=f32, device=CPU_DEVICE),
    }


def _bind(name: str, args: tuple[object, ...], *, dag: bool = True) -> BoundKernel:
    """Bind an engine kernel; ``dag=False`` binds it with the dag heuristic
    declining, i.e. the search surface of the ordinary lowering."""
    fn = getattr(_engine(), name).fn
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    if dag:
        return kernel.bind(args)
    with patch.object(CuteMmaDagHeuristic, "_facts", return_value=None):
        return kernel.bind(args)


def _block_bounds(spec: ConfigSpec) -> list[tuple[int, int, int]]:
    return [(b.block_id, b.min_size, b.max_size) for b in spec.block_sizes]


def _bind_fwd(
    mode: str = "vanilla", *, has_h0: bool = False, dag: bool = True, **shape: int
) -> BoundKernel:
    t = _tensors(**shape)
    h0 = t["h0"] if has_h0 else None
    if mode == "vanilla":
        args = (t["k"], t["v"], None, h0, None, False, False, False, has_h0)
    elif mode == "diag":
        args = (t["k"], t["v"], t["g_last"], h0, None, True, False, False, has_h0)
    elif mode == "anchored":
        # full_gla's forward state kernel: the decay rides the key per
        # channel; ``gc_last`` is the last row of the staged ``gc`` tile.
        args = (t["k"], t["v"], t["g_last"], h0, t["gc4"], True, False, True, has_h0)
    else:
        args = (t["k"], t["v"], t["g_last_s"], h0, t["gc"], True, True, False, has_h0)
    return _bind("chunk_fwd_h_diag_fused", args, dag=dag)


def _bind_bwd(mode: str = "vanilla", **shape: int) -> BoundKernel:
    t = _tensors(**shape)
    if mode == "vanilla":
        args = (t["k"], t["v"], None, t["dh_init"], None, False, False, False, 0.125)
    elif mode == "diag":
        args = (
            t["k"],
            t["v"],
            t["g_last"],
            t["dh_init"],
            None,
            True,
            False,
            False,
            1.0,
        )
    else:
        args = (
            t["k"],
            t["v"],
            t["g_last_s"],
            t["dh_init"],
            t["gc"],
            True,
            True,
            False,
            0.125,
        )
    return _bind("chunk_bwd_dh_diag_fused", args)


def _bind_fwd_o(gated: bool = False, *, dag: bool = True, **shape: int) -> BoundKernel:
    t = _tensors3(**shape)
    gcs = t["gcs"] if gated else None
    return _bind(
        "chunk_fwd_o_helion",
        (t["q"], t["q"], t["v"], gcs, t["h"], gated, 0.125),
        dag=dag,
    )


def _bind_dqk(gated: bool = False, **shape: int) -> BoundKernel:
    t = _tensors3(**shape)
    gcs, gl = (t["gcs"], t["gl"]) if gated else (None, None)
    return _bind(
        "chunk_bwd_dqk_helion",
        (t["q"], t["q"], t["v"], gcs, gl, t["h"], t["v"], t["dh"], gated, 0.125),
    )


def _bind_dqkg(
    *, diag: bool = False, compute_dg: bool = True, **shape: int
) -> BoundKernel:
    """``chunk_bwd_dqkg_scalar_helion`` in its scalar-decay or anchored
    (per-channel) mode, with or without the gate gradient."""
    t = _tensors3(**shape)
    bhn, c, d = t["q"].shape
    k = torch.empty_like(t["q"])
    do = torch.empty_like(t["v"])
    if diag:
        gcs = torch.empty((bhn, c, d), dtype=torch.float32, device=CPU_DEVICE)
        gl = None
    else:
        gcs, gl = t["gcs"], t["gl"]
    return _bind(
        "chunk_bwd_dqkg_scalar_helion",
        (t["q"], k, t["v"], gcs, t["h"], do, t["dh"], gl, diag, compute_dg, 0.125),
    )


def _bind_dv(mode: str = "vanilla", **shape: int) -> BoundKernel:
    t = _tensors3(**shape)
    if mode == "vanilla":
        args = (
            t["q"],
            t["q"],
            t["q"],
            None,
            t["v"],
            t["dh"],
            None,
            None,
            False,
            False,
            False,
            0.125,
        )
    elif mode == "diag":
        args = (
            t["q"],
            t["q"],
            t["q"],
            t["gcs"],
            t["v"],
            t["dh"],
            None,
            None,
            True,
            False,
            False,
            0.125,
        )
    else:
        args = (
            t["q"],
            t["q"],
            t["q"],
            t["gcs"],
            t["v"],
            t["dh"],
            t["gl"],
            None,
            True,
            True,
            False,
            0.125,
        )
    return _bind("chunk_bwd_dv_helion", args)


def _tensors4(
    *, bh: int = 2, n: int = 8, c: int = 64, d: int = 128, dv: int = 64
) -> dict[str, torch.Tensor]:
    """The delta-rule state kernels' inputs: D is the whole ``:`` extent."""
    bf16, f32 = torch.bfloat16, torch.float32
    return {
        "k": torch.empty((bh, n, c, d), dtype=bf16, device=CPU_DEVICE),
        "w": torch.empty((bh, n, c, d), dtype=bf16, device=CPU_DEVICE),
        "u": torch.empty((bh, n, c, dv), dtype=bf16, device=CPU_DEVICE),
        "h0": torch.empty((bh, d, dv), dtype=bf16, device=CPU_DEVICE),
        "g_cs": torch.empty((bh, n, c), dtype=f32, device=CPU_DEVICE),
        "dl": torch.empty((bh, n), dtype=f32, device=CPU_DEVICE),
    }


def _bind_fwd_h_delta(gated: bool = False, **shape: int) -> BoundKernel:
    t = _tensors4(**shape)
    g_cs, dl = (t["g_cs"], t["dl"]) if gated else (None, None)
    return _bind(
        "chunk_fwd_h_delta_helion",
        (t["k"], t["w"], t["u"], t["h0"], g_cs, dl, gated, False, False),
    )


def _bind_dstate(gated: bool = False, **shape: int) -> BoundKernel:
    t = _tensors4(**shape)
    g_cs, dl = (t["g_cs"], t["dl"]) if gated else (None, None)
    return _bind(
        "chunk_bwd_dstate_delta_helion",
        (t["k"], t["k"], t["w"], t["u"], t["h0"], g_cs, dl, gated, 0.125),
    )


def _varlen_tables(
    chunks: tuple[int, ...], c: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """The varlen kernels' chunk tables for sequences of ``chunks`` chunks
    each, the last chunk of every sequence one row short: ``(chunk_offsets
    [N + 1], token_base [NT], valid_len [NT], NT, T_total)``."""
    offsets = [0, *itertools.accumulate(chunks)]
    lens = [count * c - 1 for count in chunks]
    cu = [0, *itertools.accumulate(lens)]
    token_base = [cu[n] + i * c for n, count in enumerate(chunks) for i in range(count)]
    valid_len = [
        min(lens[n] - i * c, c) for n, count in enumerate(chunks) for i in range(count)
    ]
    i64 = torch.int64
    return (
        torch.tensor(offsets, dtype=i64, device=CPU_DEVICE),
        torch.tensor(token_base, dtype=i64, device=CPU_DEVICE),
        torch.tensor(valid_len, dtype=i64, device=CPU_DEVICE),
        offsets[-1],
        cu[-1],
    )


def _bind_fwd_h_delta_varlen(
    *,
    h: int = 2,
    chunks: tuple[int, ...] = (4, 1, 3),
    c: int = 64,
    d: int = 128,
    dv: int = 64,
) -> BoundKernel:
    """The varlen state pass: one program per (head, sequence), the chunk
    walk's bounds read from ``chunk_offsets``."""
    bf16, f32 = torch.bfloat16, torch.float32
    chunk_offsets, _base, _valid, nt, _t = _varlen_tables(chunks, c)
    n = len(chunks)

    def e(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=CPU_DEVICE)

    return _bind(
        "chunk_fwd_h_delta_varlen_helion",
        (
            e(h * nt, c, d),
            e(h * nt, c, d),
            e(h * nt, c, dv),
            e(h * n, d, dv, dtype=f32),
            e(h * nt, d, dtype=f32),
            chunk_offsets,
            nt,
            h,
        ),
    )


def _program(bound: BoundKernel) -> dag.MmaDagProgram:
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return program


def _config(
    *block_sizes: int, warps: int, stages: int, **extra: object
) -> helion.Config:
    # The warp mode's ring tiles ride per-thread packets unless a test asks
    # for the TMA ring (``cute_mma_dag_loads="tma"``).
    if extra.get("cute_mma_dag", "warp") == "warp":
        extra.setdefault("cute_mma_dag_loads", "lane")
    return helion.Config(
        block_sizes=list(block_sizes),
        cute_mma_dag="warp",
        cute_mma_dag_warps=warps,
        cute_mma_dag_stages=stages,
        **extra,
    )


def _source(bound: BoundKernel, config: helion.Config) -> str:
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    return bound.to_code(normalized)


def _kernel_body(source: str) -> str:
    start = source.index("@cute.kernel")
    first_def = source.index("\ndef ", start)
    end = source.index("\ndef ", first_def + 1)
    return source[start:end]


def _loop_and_after(body: str) -> str:
    return body[body.index("for md_i in cutlass.range(") :]


# --------------------------------------------------------------------------
# Admission mirrors the knob surface
# --------------------------------------------------------------------------


def test_admitted_kernels_expose_the_knobs_and_the_seeds() -> None:
    bound = _bind_fwd()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None and spec.cute_mma_dag_facts is not None
    facts = spec.cute_mma_dag_facts
    assert [(a.kind, a.extent) for a in facts.axes] == [
        ("lead", 2),
        ("root", 64),
        ("root", 64),
        ("rdim", 64),
    ]
    assert facts.frags == ((1, 2),) and facts.k_axes == (3,)
    assert "cute_mma_dag" in spec.flat_config(lambda f: f.default())  # type: ignore[attr-defined]
    seeds = [
        c for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag") == "warp"
    ]
    assert seeds, spec.compiler_seed_configs
    assert any(c.config["block_sizes"] == [64, 64] for c in seeds)
    assert {c.config["cute_mma_dag_stages"] for c in seeds} == set(
        dag.MMA_DAG_SEED_STAGES
    )
    # The sm100 fake target promotes the first seed to the autotune-off
    # default: the tensor-memory mode where the kernel admits it.
    assert spec.compiler_default_config is not None
    assert spec.compiler_default_config.config["cute_mma_dag"] == "tcgen05"
    assert bound.config_spec.default_config().config["cute_mma_dag"] == "tcgen05"
    # Admission leaves the block-size fragments alone (normalize and the
    # seeds keep the mode on legal tiles).
    assert _block_bounds(spec) == _block_bounds(_bind_fwd(dag=False).config_spec)


def test_k_loop_kernels_are_admitted_with_their_axes() -> None:
    bound = _bind_fwd_o()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    # tile_d is the loop (K) axis, tile_bhn the lead, tile_dv the root
    # tile, C the literal-64 reduction dim behind the [C, C] attention.
    assert sorted((a.kind, a.extent) for a in facts.axes) == [
        ("lead", 16),
        ("loop", 128),
        ("rdim", 64),
        ("root", 128),
    ]
    assert len(facts.frags) == 2 and len(facts.k_axes) == 2
    spec = bound.config_spec
    assert _block_bounds(spec) == _block_bounds(_bind_fwd_o(dag=False).config_spec)
    default = spec.default_config().config
    assert default["cute_mma_dag"] == "tcgen05" and default["block_sizes"][0] == 1
    for bind in (
        lambda: _bind_fwd_o(gated=True),
        _bind_dqk,
        lambda: _bind_dqk(gated=True),
        _bind_dv,
        lambda: _bind_dv("diag"),
        lambda: _bind_dv("scalar"),
    ):
        assert bind().config_spec.cute_mma_dag is not None


@pytest.mark.parametrize("mode", ("vanilla", "diag", "scalar"))
@pytest.mark.parametrize("kernel", ("fwd", "bwd"))
def test_every_decay_mode_is_admitted(mode: str, kernel: str) -> None:
    bound = (
        _bind_fwd(mode, has_h0=mode != "vanilla")
        if kernel == "fwd"
        else _bind_bwd(mode)
    )
    assert bound.config_spec.cute_mma_dag is not None
    source = _source(bound, _config(64, 64, warps=4, stages=2))
    assert f"{RT}.mma_bf16(" in source and SIMT_FALLBACK not in source


def test_non_admitted_kernels_never_see_the_knobs() -> None:
    bound = _bind_gemm((256, 256, 256), torch.bfloat16)
    assert bound.config_spec.cute_mma_dag is None
    requested = _config(64, 64, 64, warps=4, stages=2).config
    with bound.env:
        with pytest.raises(InvalidConfig, match="cute_mma_dag"):
            bound.config_spec.normalize(dict(requested))
        fixed = dict(requested)
        bound.config_spec.normalize(fixed, _fix_invalid=True)
    assert "cute_mma_dag" not in fixed
    # A recurrence with fixed fragment tiles is not mine either.
    t = _tensors()
    other = _bind_fn(_fixed_tiles_recurrence, (t["k"], t["v"]))
    assert other.config_spec.cute_mma_dag is None
    assert "cute_mma_dag" not in other.config_spec.default_config().config


def test_a_carry_read_as_an_operand_inside_the_loop_is_admitted() -> None:
    # The carry cast to bf16 feeds a B operand of the same iteration that
    # accumulates into it; the dot result feeds the next B operand.  With four
    # warps both [64, 64] fragments take the N-first grid and both B operands
    # come from registers (movmatrix per 8x8 block), no exchange tile.
    @helion.kernel(backend="cute", static_shapes=True)
    def carry_operand(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bh, n, c, d = k.shape
        c = hl.specialize(c)
        out = torch.empty((bh, n, d, d), dtype=k.dtype, device=k.device)
        for tile_bh, tile_d, tile_dv in hl.tile([bh, d, d], block_size=[1, None, None]):
            idx = tile_bh.id
            acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
            for i in hl.grid(n):
                out[idx, i, tile_d, tile_dv] = acc.to(out.dtype)
                k_i = k[idx, i, :, tile_d]
                proj = hl.dot(k_i, acc.to(k.dtype))
                acc = hl.dot(k_i.transpose(-2, -1), proj.to(k.dtype), acc=acc)
        return out

    t = _tensors()
    bound = carry_operand.bind((t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    assert "md_L0_wmi = md_warp // 4" in body and "md_L0_wni = md_warp % 4" in body
    assert "md_x" not in body
    # Each warp owns 64 rows x 16 columns of both fragments: 8 K blocks x 2
    # n-atoms of movmatrix words per operand, 4 x 2 atoms x 4 K steps per dot.
    assert body.count(f"{RT}.movmatrix_trans(") == 16 + 16
    assert body.count(f"{RT}.mma_bf16(") == 32 + 32


def _stale_carry(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The carry is read (as an operand) after the dot that replaces it."""
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    out = torch.empty([BH, D, DV], dtype=k.dtype, device=k.device)
    out2 = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i in hl.grid(N):
            k_i = k[idx, i, :, tile_d]
            v_i = v[idx, i, :, tile_dv]
            new = hl.dot(k_i.transpose(-2, -1), v_i, acc=acc)
            proj = hl.dot(k_i, acc.to(k.dtype))
            out2[idx, i, :, tile_dv] = proj.to(out2.dtype)
            acc = new
        out[idx, tile_d, tile_dv] = acc.to(out.dtype)
    return out, out2


def test_a_carry_read_after_its_in_place_update_gets_a_fresh_slot_and_a_copy() -> None:
    # The carry ``acc`` feeds ``proj`` after ``new`` would have overwritten it:
    # ``new`` takes a fresh slot (slot 1) and moves into the carry's registers
    # at the iteration end, after ``proj`` read them.
    t = _tensors()
    bound = _bind_fn(_stale_carry, (t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is not None
    program = _program(bound)
    (carry,) = program.loops[0].carries
    new, proj = program.dots
    assert new.acc_slot == carry.slot and new.slot != carry.slot
    assert proj.b.frag is carry.placeholder
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    loop = _loop_and_after(body)
    mmas = [m.start() for m in re.finditer(rf"{RT}\.mma_bf16\(", loop)]
    copies = re.findall(r"md_c0_\d+_\d+_\d+ = md_c1_\d+_\d+_\d+", loop)
    assert len(copies) == 32
    assert mmas[-1] < loop.index(copies[0])


def test_an_explicit_mode_the_planner_cannot_match_fails_closed() -> None:
    # The knob is on an admitted kernel; if planning cannot re-match the
    # codegen graphs, the lowering refuses instead of falling back to SIMT.
    bound = _bind_fwd()
    config = _config(64, 64, warps=4, stages=2)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    with (
        patch("helion._compiler.cute.cute_mma_dag.match_mma_dag", return_value=None),
        pytest.raises(BackendUnsupported, match="not an MMA dataflow program"),
    ):
        bound.to_code(normalized)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_normalize_pins_the_inert_knobs_when_on() -> None:
    bound = _bind_fwd()
    config = _config(
        64,
        64,
        warps=4,
        stages=2,
        num_threads=[0, 4, 0],
        cute_vector_widths=[2, 1, 1, 1, 1],
        cute_lane_layouts=["strided"] * 5,
        pid_type="persistent_blocked",
        cute_min_blocks_per_mp=2,
    ).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["pid_type"] == "flat"
    assert all(t == 0 for t in config["num_threads"])
    assert all(w == 1 for w in config["cute_vector_widths"])
    assert all(layout == "blocked" for layout in config["cute_lane_layouts"])
    assert config["cute_min_blocks_per_mp"] == 2
    assert config.get("loop_orders", [[0, 1, 2]]) == [[0, 1, 2]]
    assert config["cute_mma_dag_exchange"] == "auto"


def test_normalize_pins_the_lead_tile_to_one_row() -> None:
    bound = _bind_fwd_o()
    config = _config(4, 64, 64, warps=4, stages=2).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["block_sizes"] == [1, 64, 64]


def test_normalize_defaults_the_inert_warps_and_stages_when_off() -> None:
    bound = _bind_fwd()
    config = helion.Config(
        block_sizes=[32, 32],
        cute_mma_dag="off",
        cute_mma_dag_warps=8,
        cute_mma_dag_stages=1,
    ).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag"] == "off"
    assert config["cute_mma_dag_warps"] == dag.MMA_DAG_DEFAULT_WARPS
    assert config["cute_mma_dag_stages"] == dag.MMA_DAG_DEFAULT_STAGES
    # ``off`` keeps the ordinary lowering.
    source = bound.to_code(helion.Config(**config))
    assert SIMT_FALLBACK in source and f"{RT}." not in source


@pytest.mark.parametrize(
    ("bm", "bn", "warps", "stages"),
    ((16, 8, 2, 1), (64, 64, 4, 5), (64, 64, 16, 1), (64, 64, 3, 1), (128, 128, 1, 1)),
)
def test_normalize_refuses_or_repairs_illegal_configs(
    bm: int, bn: int, warps: int, stages: int
) -> None:
    bound = _bind_fwd(d=128, dv=128)
    config = _config(bm, bn, warps=warps, stages=stages).config
    with bound.env:
        with pytest.raises(InvalidConfig, match="cute_mma_dag"):
            bound.config_spec.normalize(dict(config))
        bound.config_spec.normalize(config, _fix_invalid=True)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None
        reason = dag.mma_dag_config_reason(
            facts,
            {0: 1, 1: config["block_sizes"][0], 2: config["block_sizes"][1]},
            config["cute_mma_dag_warps"],
            config["cute_mma_dag_stages"],
        )
    assert config["cute_mma_dag"] == "off" or reason is None


def test_register_budget_overrun_is_not_viable_and_repaired() -> None:
    bound = _bind_dqk()
    spec = bound.config_spec
    # dqk carries dA [64, 64] plus two [64, bd] accumulators: 160 registers
    # per thread at bd = 128 over 4 warps, 80 over 8.
    config = _config(1, 128, 32, warps=4, stages=1)
    assert spec.backend.autotune_config_is_viable(spec, config) is False
    assert (
        spec.backend.autotune_config_is_viable(
            spec, _config(1, 128, 32, warps=8, stages=1)
        )
        is True
    )
    with bound.env:
        with pytest.raises(InvalidConfig, match="registers per thread"):
            spec.normalize(dict(config.config))
        repaired = dict(config.config)
        spec.normalize(repaired, _fix_invalid=True)
    assert repaired["cute_mma_dag"] == "warp"
    assert repaired["cute_mma_dag_warps"] == 8 and repaired["block_sizes"][1] == 128


# --------------------------------------------------------------------------
# Generated code: the serial state recurrences
# --------------------------------------------------------------------------


def test_renders_the_vanilla_forward_recurrence() -> None:
    source = _source(_bind_fwd(), _config(64, 64, warps=4, stages=2))
    body = _kernel_body(source)
    assert SIMT_FALLBACK not in source
    assert "block=(128, 1, 1)" in source
    # The carry is 32 loop-carried fp32 registers per thread (slot 0).
    assert "make_rmem_tensor" not in body
    assert "md_c0_0_7_3 = cutlass.Float32(0.0)" in body
    assert "for md_i in cutlass.range(0, 8):" in body
    # 4 warps along M: one 16-row atom by eight 8-column atoms per warp,
    # four K steps -> 32 mma.sync, 4 A + 16 B ldmatrix.trans per iteration.
    assert body.count(f"{RT}.mma_bf16(") == 32
    assert body.count(f"{RT}.ldmatrix_x4_trans(") == 20
    assert f"{RT}.ldmatrix_x4(" not in body
    # 64x64 bf16 tiles are 512 packets each: four per thread per operand,
    # issued once in the prologue and once per iteration (two stages).
    assert body.count(f"{RT}.cp_async_16(") == 16
    assert body.count("cute.arch.cp_async_commit_group()") == 2
    # Two stages: the group of this iteration was committed one iteration
    # ago and nothing younger exists before the refill, so the wait drains.
    assert "cute.arch.cp_async_wait_group(0)" in body
    assert "if md_i + 1 < 8:" in body
    # The carry snapshot goes through the staging tile: packed bf16 pairs in
    # before the barrier, 16-byte packets out after it (two buffers, 8 KB each).
    assert body.count(f"{RT}.sts_b32(") == 16
    assert body.count(f"{RT}.lds_v4_b32(") == 4
    assert body.count(f"{RT}.stg_v4_b32(") == 4
    assert f"{RT}.stg_b32(" not in body
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=128)" in body
    assert "md_parity = md_i % 2" in body
    # One CTA barrier per iteration: the refill (the next issue) sits past it.
    assert body.count("cute.arch.barrier()") == 1
    wait = body.index("cute.arch.cp_async_wait_group(0)")
    barrier = body.index("cute.arch.barrier()")
    assert body.index(f"{RT}.sts_b32(") < wait < barrier
    assert barrier < body.index("if md_i + 1 < 8:") < body.index(f"{RT}.mma_bf16(")
    assert barrier < body.index(f"{RT}.lds_v4_b32(") < body.index(f"{RT}.stg_v4_b32(")
    assert "tcgen05" not in body
    assert "thread_idx()[1]" not in body


def test_stage_depth_drives_the_wait_depth_and_the_prologue() -> None:
    one = _kernel_body(_source(_bind_fwd(), _config(64, 64, warps=4, stages=1)))
    assert "cute.arch.cp_async_wait_group(0)" in one
    assert one.count("cute.arch.cp_async_commit_group()") == 1
    assert "md_stage = 0" in one
    four = _kernel_body(_source(_bind_fwd(n=16), _config(64, 64, warps=4, stages=4)))
    # Four stages: two younger groups may stay in flight at the wait.
    assert "cute.arch.cp_async_wait_group(2)" in four
    # Three prologue groups plus one per iteration.
    assert four.count("cute.arch.cp_async_commit_group()") == 4
    assert "md_stage = md_i % 4" in four
    assert "if md_i + 3 < 16:" in four
    assert "alloc_smem(cutlass.Uint8, 81920, alignment=128)" in four
    assert (
        one.count("cute.arch.barrier()") == 2 and four.count("cute.arch.barrier()") == 1
    )


def test_warp_layout_and_single_atom_columns_use_ldmatrix_x2() -> None:
    source = _source(_bind_fwd(), _config(16, 8, warps=1, stages=1))
    body = _kernel_body(source)
    assert "block=(32, 1, 1)" in source
    assert "cute.arch.sync_warp()" in body and "cute.arch.barrier()" not in body
    assert body.count(f"{RT}.mma_bf16(") == 4
    assert body.count(f"{RT}.ldmatrix_x2_trans(") == 4
    assert "md_c0_0_0_3 = cutlass.Float32(0.0)" in body and "md_c0_0_1_0" not in body
    eight = _kernel_body(_source(_bind_fwd(), _config(64, 64, warps=8, stages=2)))
    assert "md_L0_wmi = md_warp // 2" in eight and "md_L0_wni = md_warp % 2" in eight
    assert eight.count(f"{RT}.mma_bf16(") == 16


def test_renders_the_seeded_row_rescaled_forward() -> None:
    body = _kernel_body(
        _source(_bind_fwd("diag", has_h0=True), _config(64, 64, warps=4, stages=2))
    )
    # The h0 seed is loaded per lane at the fragment coordinates (bf16 pairs).
    seed = body.index("for md_i in cutlass.range(")
    assert body[:seed].count(f"{RT}.unpack_bf16x2({RT}.ldg_b32(") == 16
    # Rows g / g + 8 of the warp's atom: two scalar loads of the decay row
    # per iteration, through the ordinary exp lowering.
    loop = body[seed:]
    assert loop.count(f"{RT}.ldg_f32(") == 2
    assert "exp" in loop
    assert loop.index(f"{RT}.ldg_f32(") < loop.index(f"{RT}.mma_bf16(")


def test_renders_the_scalar_decay_forward_with_an_inplace_operand_pass() -> None:
    body = _kernel_body(
        _source(_bind_fwd("scalar", has_h0=True), _config(64, 64, warps=4, stages=2))
    )
    # v is staged raw by cp.async and rewritten packet by packet after the
    # thread's own wait: unpack, the K-vector exp2 scale, repack.
    # (4 of the 8 packet reads are the staged carry copy-out.)
    assert body.count(f"{RT}.lds_v4_b32(") == 8
    assert body.count(f"{RT}.sts_v4_b32(") == 4
    assert "exp2" in body
    assert body.count(f"{RT}.cp_async_16(") == 16
    wait = body.index("cute.arch.cp_async_wait_group(0)")
    assert wait < body.index(f"{RT}.sts_v4_b32(") < body.index("cute.arch.barrier()")
    # The K-vector decay is loaded and exponentiated once per packet row,
    # the scalar decay once per iteration: 1 + 2 + 2 loads.
    loop = _loop_and_after(body)
    assert loop.count(f"{RT}.ldg_f32(") == 5
    # The scalar decay multiplies every carried register once.
    assert len(re.findall(r"md_c0_0_0_0 = md_v\w*\n", loop)) == 1


def test_renders_the_reversed_scaled_backward_with_fp32_stores() -> None:
    body = _kernel_body(_source(_bind_bwd(), _config(64, 64, warps=4, stages=2)))
    assert "7 + -1 * md_i" in body
    # fp32 snapshots: 8-byte pair writes into the staging tile (XOR 2 * row
    # swizzle), 16 KB per buffer, eight 16-byte packets out per thread.
    assert body.count(f"{RT}.sts_v2_f32(") == 16
    assert "^ 2 * md_L0_rowl % 8" in body
    assert body.count(f"{RT}.stg_v4_b32(") == 8
    assert "alloc_smem(cutlass.Uint8, 65536, alignment=128)" in body
    assert f"{RT}.stg_b32(" not in body and f"{RT}.stg_v2_f32(" not in body
    # q * scale is the in-place pass on the staged q tile.
    assert body.count(f"{RT}.sts_v4_b32(") == 4
    assert "scale" in body
    assert body.count(f"{RT}.ldmatrix_x4_trans(") == 20
    direct = _kernel_body(
        _source(
            _bind_bwd(), _config(64, 64, warps=4, stages=2, cute_mma_dag_store="direct")
        )
    )
    assert direct.count(f"{RT}.stg_v2_f32(") == 16
    assert f"{RT}.stg_v4_b32(" not in direct
    assert "alloc_smem(cutlass.Uint8, 32768, alignment=128)" in direct


def test_fp32_seed_loads_as_pairs() -> None:
    body = _kernel_body(_source(_bind_bwd("diag"), _config(32, 32, warps=2, stages=1)))
    seed = body.index("for md_i in cutlass.range(")
    assert body[:seed].count(f"{RT}.ldg_v2_f32(") == 8
    assert "md_c0_0_3_3 = md_hi" in body


def test_unaligned_sources_route_through_the_synchronous_pass() -> None:
    t = _tensors()
    k = torch.empty((2, 8, 64, 80), dtype=torch.bfloat16, device=CPU_DEVICE)[..., 4:68]
    bound = _bind(
        "chunk_fwd_h_diag_fused",
        (k, t["v"], None, None, None, False, False, False, False),
    )
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    # k: pair loads straight from global memory into its own ring slot
    # (rewriting a single slot would race with the previous iteration's
    # reads); v keeps its two-stage cp.async ring.
    assert body.count(f"{RT}.ldg_b32(") == 16
    assert body.count(f"{RT}.sts_b32(") == 16 + 16
    assert body.count(f"{RT}.cp_async_16(") == 8
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=128)" in body
    assert body.index(f"{RT}.ldg_b32(") < body.index("cute.arch.cp_async_wait_group(")


# --------------------------------------------------------------------------
# Generated code: K loops with epilogue dots
# --------------------------------------------------------------------------


def test_fwd_o_shares_the_q_tile_and_repacks_attn_from_registers() -> None:
    source = _source(_bind_fwd_o(), _config(1, 64, 64, warps=4, stages=2))
    body = _kernel_body(source)
    assert "block=(128, 1, 1)" in source
    assert SIMT_FALLBACK not in source
    # Two carries (o_cross [64, 64] and attn [64, 64]) share one warp grid.
    assert "md_c0_0_7_3 = cutlass.Float32(0.0)" in body
    assert "md_c1_0_7_3 = cutlass.Float32(0.0)" in body
    assert body.count("md_L0_rowl =") == 1 and "md_L1_rowl" not in body
    # q feeds both loop dots from one ring tile: q, h, k rings (2 x 8 KB
    # each) plus the epilogue's v tile, no exchange tile.
    assert re.findall(r"^\s*(md_[tx]\d+) = md_s0", body, re.MULTILINE) == [
        "md_t0",
        "md_t1",
        "md_t2",
        "md_t3",
    ]
    assert "alloc_smem(cutlass.Uint8, 57344, alignment=128)" in body
    # The v tile is issued at kernel entry (its own group) before the ring
    # prologue, then waited at the epilogue.
    pre = body[: body.index("for md_i in cutlass.range(")]
    assert pre.count("cute.arch.cp_async_commit_group()") == 2
    assert pre.index("md_t3 +") < pre.index("md_t0 +")
    assert "for md_i in cutlass.range(0, 2):" in body
    assert "md_a0 = md_i * 64" in body
    loop = _loop_and_after(body)
    # Per iteration: two dots x 4 K steps x 8 atoms; the epilogue dot adds
    # 32 more with its A fragments repacked from attn's registers (4
    # warps give both fragments whole rows: the identity route).
    assert body.count(f"{RT}.mma_bf16(") == 96
    assert loop.count("md_aw") == 32
    assert f"{RT}.sts_b32(" not in body
    # The causal mask is evaluated per register from the fragment
    # coordinates and lowered as the SIMT where expressions; the bf16 cast
    # feeding the pack is the pack's own ``cvt.rn.bf16x2`` (no scalar cvt).
    assert "md_L0_rowl" in loop and " if " in loop and "cutlass.Int32(" in loop
    assert "cutlass.BFloat16(" not in loop
    assert re.search(rf"{RT}\.pack_bf16x2\(cutlass\.Float32\(md_v\w*\)", loop)
    # The epilogue waits for every group, then one barrier, then the dot and
    # the pair stores of (o * scale).to(bf16).
    epilogue = loop[loop.rindex("cute.arch.cp_async_wait_group(0)") :]
    assert epilogue.count("cute.arch.barrier()") == 1
    assert epilogue.count(f"{RT}.stg_b32(") == 16
    assert "scale" in epilogue


def test_fwd_o_exchange_knob_and_wider_tiles_use_the_shared_tile() -> None:
    smem = _kernel_body(
        _source(
            _bind_fwd_o(),
            _config(1, 64, 64, warps=4, stages=2, cute_mma_dag_exchange="smem"),
        )
    )
    assert "md_aw" not in smem
    # The attention exchange tile is read by the epilogue dot alone, after
    # the K loop: it overlays the loop's ring frame (behind a barrier).
    assert "md_x3 = md_s0 + 0\n" in smem
    assert smem.count(f"{RT}.sts_b32(") == 16
    assert "alloc_smem(cutlass.Uint8, 57344, alignment=128)" in smem
    assert smem.rindex(
        "cute.arch.barrier()", 0, smem.index(f"{RT}.sts_b32(")
    ) > smem.index("for md_i in cutlass.range(")
    assert (
        smem.index(f"{RT}.sts_b32(")
        < smem.rindex("cute.arch.barrier()")
        < smem.rindex(f"{RT}.ldmatrix_x4(")
    )
    wide = _kernel_body(_source(_bind_fwd_o(), _config(1, 128, 128, warps=8, stages=2)))
    # o [64, 128] and attn [64, 64] get different warp grids (4 x 2 each,
    # but different atom counts); attn's wn = 2 forbids the identity route.
    assert "md_L1_rowl =" in wide
    assert "md_x3 = md_s0 + 0\n" in wide and "md_aw" not in wide
    assert wide.count(f"{RT}.sts_b32(") == 8
    assert "alloc_smem(cutlass.Uint8, 147456, alignment=128)" in wide


def test_gated_fwd_o_rescales_the_finished_accumulator_and_masks_with_exp2() -> None:
    body = _kernel_body(
        _source(_bind_fwd_o(gated=True), _config(1, 128, 64, warps=8, stages=4))
    )
    loop = _loop_and_after(body)
    assert "exp2" in loop
    # gc is read at the fragment's rows (o_cross rescale, 2 per atom row) and
    # at both rows and columns for the [C, C] outer difference.
    assert loop.count(f"{RT}.ldg_f32(") == 12
    assert loop.count("cute.arch.cp_async_wait_group(2)") == 1
    # o_cross * exp2(gc)[:, :, None] lands in the slot the epilogue dot
    # accumulates into.
    assert re.search(r"md_c0_0_0_0 = md_v\w*\n", loop) is not None


def test_dqk_reads_one_exchange_tile_straight_and_transposed() -> None:
    body = _kernel_body(_source(_bind_dqk(), _config(1, 64, 64, warps=4, stages=2)))
    tiles = re.findall(r"^\s*(md_[tx]\d+r?) = md_s0", body, re.MULTILINE)
    # do, v, h and the fp32 dh (staged raw beside its bf16 tile, both ringed)
    # in the loop; k and q * scale for the epilogue; one bf16 exchange tile of
    # dA feeds dk's transposed A (dq's A is the identity repack).
    assert tiles == [
        "md_t0",
        "md_t1",
        "md_t2",
        "md_t3",
        "md_t3r",
        "md_t4",
        "md_x5",
        "md_t6",
    ]
    loop = _loop_and_after(body)
    assert loop.count("md_aw") == 32
    assert loop.count(f"{RT}.sts_b32(") == 16
    # dh (fp32) arrives by cp.async in the raw tile; each thread converts its
    # own two packets (after its wait, before the barrier) into the bf16 tile.
    assert f"{RT}.ldg_v2_f32(" not in loop
    assert loop.count(f"{RT}.lds_v4_f32(") == 8
    assert loop.count(f"{RT}.sts_v2_b32(") == 8
    assert (
        loop.index("cute.arch.cp_async_wait_group(")
        < loop.index(f"{RT}.lds_v4_f32(")
        < loop.index(f"{RT}.sts_v2_b32(")
        < loop.index("cute.arch.barrier()")
    )
    # dk's A reads the exchange tile with ldmatrix.trans (4 steps); the two
    # N-contiguous B tiles (k, q * scale) read transposed as well.
    epilogue = loop[loop.rindex("cute.arch.cp_async_wait_group(0)") :]
    assert epilogue.count(f"{RT}.ldmatrix_x4_trans(") == 4 + 16 + 16
    assert epilogue.count(f"{RT}.stg_b32(") == 32
    # Two 16 KB bf16 operand rings, one 32 KB fp32 raw ring of dh (one slot:
    # the pass consumes it ahead of the barrier the refill follows) and the
    # 8 KB attention tile, twice by frame; the 16 KB exchange tile of dA,
    # read by the two epilogue dots alone, overlays the ring frame.
    assert "alloc_smem(cutlass.Uint8, 98304, alignment=128)" in body
    assert "md_x5 = md_s0 + 0\n" in body


def test_dv_transposes_the_masked_attention_through_the_exchange_tile() -> None:
    body = _kernel_body(_source(_bind_dv(), _config(1, 64, 64, warps=4, stages=2)))
    loop = _loop_and_after(body)
    assert "md_x4 = md_s0 + " in body
    assert loop.count(f"{RT}.sts_b32(") == 16
    assert loop.count(f"{RT}.sts_v2_b32(") == 8
    assert "md_aw" not in loop
    epilogue = loop[loop.rindex("cute.arch.cp_async_wait_group(0)") :]
    assert epilogue.count(f"{RT}.ldmatrix_x4_trans(") == 4 + 16
    assert epilogue.count(f"{RT}.stg_b32(") == 16
    scalar = _kernel_body(
        _source(_bind_dv("scalar"), _config(1, 64, 64, warps=8, stages=2))
    )
    assert "exp2" in _loop_and_after(scalar)


def test_unaligned_fp32_sources_keep_the_synchronous_pair_pass() -> None:
    t = _tensors3()
    dh = torch.empty((16, 128, 132), dtype=torch.float32, device=CPU_DEVICE)
    dh = dh[..., 2:130]
    bound = _bind(
        "chunk_bwd_dv_helion",
        (
            t["q"],
            t["q"],
            t["q"],
            None,
            t["v"],
            dh,
            None,
            None,
            False,
            False,
            False,
            0.125,
        ),
    )
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(1, 64, 64, warps=4, stages=2)))
    tiles = re.findall(r"^\s*(md_[tx]\d+r?) = md_s0", body, re.MULTILINE)
    # No raw tile: dh is read in pairs straight from global memory.
    assert tiles == ["md_t0", "md_t1", "md_t2", "md_t3", "md_x4", "md_t5"]
    loop = _loop_and_after(body)
    assert loop.count(f"{RT}.ldg_v2_f32(") == 16
    assert f"{RT}.lds_v4_f32(" not in loop
    assert loop.count(f"{RT}.sts_b32(") == 16 + 16


def test_engine_kernels_pin_their_admission() -> None:
    # Every chunked engine kernel (varlen variants aside) with the dag's
    # verdict: the state, K-loop, delta-rule and region kernels and the kda /
    # anchored kernels whose GPU numerics are covered are admitted; the rest
    # decline.
    bf16, f32 = torch.bfloat16, torch.float32

    def e(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=CPU_DEVICE)

    bhn, bh, n, c, d, dv = 16, 2, 8, 64, 64, 64
    gc = e(bhn, c, d, dtype=f32)
    other = {
        "chunk_bwd_o_kda_helion": (
            e(bhn, c, d),
            e(bhn, c, dv),
            e(bhn, d, dv),
            e(bhn, c, c),
            gc,
            e(bhn, c, dv),
            0.125,
        ),
        "chunk_bwd_state_du_kda_helion": (
            e(bh, n, c, d),
            e(bh, n, c, d),
            e(bh, n, c, d, dtype=f32),
            e(bh, n, c, dv, dtype=f32),
            e(bh, n, d, dv, dtype=f32),
            e(bh, d, dv, dtype=f32),
        ),
        "chunk_bwd_wu_kda_helion": (
            e(bhn, c, c, dtype=f32),
            e(bhn, c, d),
            e(bhn, c, dv),
            e(bhn, c, dtype=f32),
            gc,
            e(bhn, c, c),
            e(bhn, c, d, dtype=f32),
            e(bhn, c, dv, dtype=f32),
        ),
        "chunk_fwd_o_diag_anchored_helion": (
            e(bhn, c, d),
            e(bhn, c, dv),
            gc,
            e(bhn, d, dv),
            e(bhn, c, c),
            0.125,
        ),
        "chunk_bwd_state_dwk_kda_helion": (
            e(bh, n, c, d),
            e(bh, n, c, d, dtype=f32),
            e(bh, n, d, dv),
            e(bh, n, c, dv),
            e(bh, n, c, dv, dtype=f32),
            e(bh, n, d, dv, dtype=f32),
        ),
        "chunk_bwd_gram2_kda_helion": (
            e(bhn, c, d),
            e(bhn, c, d),
            gc,
            e(bhn, c, c, dtype=f32),
            e(bhn, c, c, dtype=f32),
            1.0,
        ),
        "chunk_fwd_A_diag_anchored_helion": (
            e(bhn, c, d),
            e(bhn, c, d),
            gc,
            0.125,
            True,
        ),
        "chunk_bwd_dqkg_scalar_helion": (
            e(bhn, c, d),
            e(bhn, c, d),
            e(bhn, c, dv),
            e(bhn, c, dtype=f32),
            e(bhn, d, dv),
            e(bhn, c, dv),
            e(bhn, d, dv, dtype=f32),
            e(bhn, dtype=f32),
            False,
            True,
            0.125,
        ),
        "chunk_cumsum_gc_helion": (gc,),
    }
    binders: dict[str, Callable[[], BoundKernel]] = {
        "chunk_fwd_h_diag_fused": _bind_fwd,
        "chunk_bwd_dh_diag_fused": _bind_bwd,
        "chunk_fwd_o_helion": _bind_fwd_o,
        "chunk_bwd_dqk_helion": _bind_dqk,
        "chunk_bwd_dv_helion": _bind_dv,
        "chunk_fwd_h_delta_helion": _bind_fwd_h_delta,
        "chunk_bwd_dstate_delta_helion": _bind_dstate,
        "chunk_bwd_dqkw_delta_helion": _bind_dqkw,
        "chunk_bwd_wy_dL_delta_helion": _bind_wy_dl,
        "chunk_bwd_dk_delta_helion": _bind_dk,
        "chunk_fwd_wy_delta_helion": _bind_wy,
        "chunk_fwd_wy_delta_helion[diag_anchored]": _bind_wy_anchored,
        **{name: functools.partial(_bind, name, args) for name, args in other.items()},
    }
    # (dwk's epilogue reads the column sum of its own fragment beside the
    # fragment -- ``dgc_pos + is_last * dgc_last[:, None, :]`` -- admitted
    # since the vector reads; its GPU numerics are covered by
    # ``test_kda_kernels_match_the_triton_backend``.  dqkg with the gate
    # gradient ends in ``hl.cumsum(dg_raw + is_last[None, :, None] *
    # dg_last[:, None, :], reverse=True)``: a scan whose value reads a
    # vector, admitted since the scan step collects them -- see
    # ``test_dqkg_with_the_gate_gradient_is_a_scan_program``.)
    declined = {
        "chunk_bwd_gram2_kda_helion",
        "chunk_fwd_A_diag_anchored_helion",
        "chunk_cumsum_gc_helion",
    }
    verdicts = {}
    for name, binder in binders.items():
        spec = binder().config_spec
        verdicts[name] = spec.cute_mma_dag is not None
        assert ("cute_mma_dag" in spec.default_config().config) == verdicts[name], name
    assert verdicts == {name: name not in declined for name in binders}


def test_wy_is_admitted_under_tf32_only() -> None:
    # The Neumann chain runs fp32 x fp32 dots: tf32 MMAs under the default
    # dot precision; "ieee" keeps the exact lowering (not mine).
    engine = _engine()
    t = _tensors3()
    beta = torch.empty((16, 64), dtype=torch.float32, device=CPU_DEVICE)
    for gated in (False, True):
        args = (
            t["q"],
            t["v"],
            beta,
            t["gcs"] if gated else None,
            None,
            None,
            gated,
            False,
        )
        bound = helion.kernel(
            engine.chunk_fwd_wy_delta_helion.fn, backend="cute", static_shapes=True
        ).bind(args)  # type: ignore[attr-defined]
        assert bound.config_spec.cute_mma_dag is not None, gated
        exact = helion.kernel(
            engine.chunk_fwd_wy_delta_helion.fn,
            backend="cute",
            static_shapes=True,
            dot_precision="ieee",
        ).bind(args)  # type: ignore[attr-defined]
        assert exact.config_spec.cute_mma_dag is None, gated
        assert "cute_mma_dag" not in exact.config_spec.default_config().config


# --------------------------------------------------------------------------
# Generated code: the delta-rule state recurrences
# --------------------------------------------------------------------------


def test_delta_state_kernels_are_admitted_with_their_axes() -> None:
    bound = _bind_fwd_h_delta()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    # tile_bh is the lead, tile_dv the root (N) tile; D (128) and C (64) are
    # the ``:`` reduction dims: M of the carry and K of the second dot.
    assert [(a.kind, a.extent) for a in facts.axes] == [
        ("lead", 2),
        ("root", 64),
        ("rdim", 128),
        ("rdim", 64),
    ]
    # The carry [D, bdv] and the no-acc dot's result [C, bdv] (the derived
    # ``u - dot`` takes its slot); both reduction dims serve as K.
    assert facts.frags == ((2, 1), (3, 1)) and facts.k_axes == (2, 3)
    assert max(facts.live_groups, key=len) == (0, 1)
    assert facts.staging == ((2, 1, 2),)
    # The carry's exchange tile (if a config needs it) is double-buffered;
    # the derived value's u tile rides the ring like the operands.
    assert [(t.ring, t.copies) for t in facts.tiles] == [
        (True, 1),
        (False, 2),
        (True, 1),
        (False, 1),
        (True, 1),
    ]
    # tile_bh is fixed to one row by the kernel (not a tunable block); the
    # dv block's bounds are left alone (normalize refuses / repairs tiles
    # below one atom, the seeds emit legal ones).
    assert 0 not in spec.block_sizes.valid_block_ids()
    assert spec.block_sizes.block_id_lookup(1).min_size == 1
    with bound.env, pytest.raises(InvalidConfig, match="cute_mma_dag"):
        spec.normalize(dict(_config(4, warps=4, stages=2).config))
    assert spec.default_config().config["cute_mma_dag"] == "tcgen05"
    dstate = _bind_dstate().config_spec.cute_mma_dag_facts
    assert dstate is not None
    assert [(a.kind, a.extent) for a in dstate.axes] == [
        ("lead", 2),
        ("root", 64),
        ("rdim", 128),
        ("rdim", 64),
    ]
    # dS [D, bdv], attn [C, C], dv_i [C, bdv], the subtracted dot [D, bdv].
    assert dstate.frags == ((2, 1), (3, 3), (3, 1), (2, 1))
    for bind in (
        lambda: _bind_fwd_h_delta(gated=True),
        lambda: _bind_dstate(gated=True),
    ):
        assert bind().config_spec.cute_mma_dag is not None


def test_delta_state_kernels_are_admitted_when_d_equals_c() -> None:
    # Reduction dims are deduplicated by size, so at D == C == 64 the two
    # ``:`` dims of ``w[idx, i, :, :]`` share one axis: tile strides and the
    # contiguous dim are positional (the s0 shape of the campaign).
    for bind in (_bind_fwd_h_delta, _bind_dstate):
        bound = bind(d=64, dv=64)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None
        assert [(a.kind, a.extent) for a in facts.axes] == [
            ("lead", 2),
            ("root", 64),
            ("rdim", 64),
        ]
        assert bound.config_spec.default_config().config["cute_mma_dag"] == "tcgen05"
    bound = _bind_fwd_h_delta(d=64, dv=64)
    program = _program(bound)
    dot1 = program.steps[1]
    assert isinstance(dot1, dag.Dot) and dot1.a.load is not None
    assert dot1.a.load.name == "w" and dot1.a.load.tile_dim_strides == (64, 1)
    assert dot1.a.rows is dot1.a.cols and dot1.a.rows_pos == 0 and not dot1.a.trans
    body = _kernel_body(_source(bound, _config(32, warps=4, stages=2)))
    loop = _loop_and_after(body)
    # [64, 64] w read straight (4 K steps), [64, 64] k^T read transposed.
    assert loop.count(f"{RT}.ldmatrix_x4(") == 16
    assert loop.count(f"{RT}.ldmatrix_x4_trans(") == 16
    assert loop.count(f"{RT}.mma_bf16(") == 16 + 16


def test_anchored_fwd_h_reads_the_decay_row_from_the_ring_slot() -> None:
    # full_gla's forward state kernel (``diag_anchored``): ``gc_last =
    # gc[idx, i_t, C - 1, tile_d]`` is row 63 of the ``gc[idx, i_t, :,
    # tile_d]`` tile the ring stages for the key's decay chain, so the carry
    # rescale and the chain read it from the slot (``lds_f32`` at the
    # swizzled row address) behind the iteration barrier instead of
    # re-fetching the row from global memory on the chain -- the rescale is
    # no longer hoisted to the iteration top.  The key's ``.to(bf16)`` under
    # its transpose is folded into the exchange's pack: one rounding, no
    # scalar cast widened back.  Both derived routes; the same bytes.
    bound = _bind_fwd("anchored", has_h0=True, d=128, dv=128)
    for derived, row_reads in (("exchange", 2 + 4), ("pass", 2 + 8)):
        source = _source(
            bound, _config(32, 128, warps=8, stages=4, cute_mma_dag_derived=derived)
        )
        assert SIMT_FALLBACK not in source
        loop = _loop_and_after(_kernel_body(source))
        reads = re.findall(
            rf"{RT}\.lds_f32\((md_d\d+) \+ md_stage \* 8192 \+ 63 \* 128 \+ ", loop
        )
        assert len(reads) == row_reads and len(set(reads)) == 1, (derived, reads)
        assert f"{RT}.ldg_f32(" not in loop and f"{RT}.ldg_v2_f32(" not in loop
        barrier = loop.index("cute.arch.barrier()")
        assert loop.index(f"{RT}.lds_f32(") > barrier
        rescale = re.search(r"\n\s+md_c0_0_0_0 = md_v", loop)
        assert rescale is not None and rescale.start() > barrier
        assert loop.count("cute.arch.barrier()") == 2
        # (``pack_bf16x2(cutlass.Float32(<fp32 register>), ...)``: the pack
        # rounds the fp32 products itself; the loaded bf16 halves are
        # re-typed exactly, their fp32 twins feed the chain.)
        assert not _bf16_casts_of_computed_values(loop)
        assert loop.count(f"{RT}.mma_bf16(") == 16


def test_decay_rows_without_a_staged_tile_keep_their_global_loads() -> None:
    # ``diag`` mode: ``g_last[idx, i_t, tile_d]`` has no tile of ``g_last``
    # in the ring, so the rescale's row stays a global load (fail closed: an
    # alias needs the same tensor staged) -- the register prefetch's, loaded
    # one iteration ahead into a register the loop refills.
    loop = _loop_and_after(
        _kernel_body(_source(_bind_fwd("diag"), _config(64, 64, warps=4, stages=2)))
    )
    assert f"{RT}.ldg_f32(" in loop and f"{RT}.lds_f32(" not in loop
    assert re.search(rf"^ *md_px\w* = {RT}\.ldg_f32\(", loop, re.MULTILINE)


def test_fwd_h_delta_feeds_the_carry_and_the_derived_value_from_registers() -> None:
    bound = _bind_fwd_h_delta()
    program = _program(bound)
    dot1, derived, store, dot2 = program.steps[1:]
    assert isinstance(dot1, dag.Dot) and dot1.acc is None
    assert dot1.b.kind == "frag" and dot1.b.frag in program.carry_placeholders
    assert isinstance(derived, dag.Derived)
    assert derived.frags == (dot1.node,) and [a.name for a in derived.loads] == ["u"]
    assert derived.slot == dot1.slot
    assert isinstance(store, dag.Store) and store.source == "body"
    assert isinstance(dot2, dag.Dot) and dot2.b.frag is derived.node
    assert dot2.slot == dot2.acc_slot == 0
    source = _source(bound, _config(32, warps=4, stages=2))
    body = _kernel_body(source)
    assert "block=(128, 1, 1)" in source and SIMT_FALLBACK not in source
    # Four warps split N: every warp owns whole K rows of the [128, 32] carry
    # and of the [64, 32] result, so both B operands are transposed in
    # registers (8x8 movmatrix per K block) and no exchange tile exists.
    assert "md_L0_wmi = md_warp // 4" in body and "md_L1_wmi = md_warp // 4" in body
    # w and k operand rings, the u tile's ring (read per lane), the staging
    # buffers: 32 + 32 + 8 + 16 KB.
    assert re.findall(r"^\s*(md_[txd]\d+r?) = md_s0", body, re.MULTILINE) == [
        "md_t0",
        "md_t1",
        "md_d0",
    ]
    assert "md_d0 = md_s0 + 65536" in body
    assert "alloc_smem(cutlass.Uint8, 90112, alignment=128)" in body
    loop = _loop_and_after(body)
    assert loop.count(f"{RT}.movmatrix_trans(") == 16 + 8
    assert loop.count("md_bw") == 2 * (16 + 8) and "md_aw" not in loop
    # dot1: w [64, 128] read straight (8 K steps x 4 atom rows); dot2: k^T
    # read transposed (4 K steps x 8 atom rows); 32 + 32 mma.sync.
    assert loop.count(f"{RT}.ldmatrix_x4(") == 32
    assert loop.count(f"{RT}.ldmatrix_x4_trans(") == 32
    assert loop.count(f"{RT}.mma_bf16(") == 64
    # u rides the cp.async ring (two packets per thread per issue) and is
    # read as bf16 pairs from the slot at the fragment coordinates after
    # dot1; the derived ``u - dot`` lands in dot1's slot and is stored
    # mid-iteration (pair stores) before dot2 consumes it.
    assert loop.count(f"{RT}.cp_async_16(md_d0") == 2
    assert f"{RT}.ldg_b32(" not in loop
    assert loop.count(f"{RT}.lds_b32(md_d0 + md_stage * 4096 + ") == 8
    first_mma = loop.index(f"{RT}.mma_bf16(")
    assert first_mma < loop.index(f"{RT}.lds_b32(")
    sub = re.search(r"md_c1_0_0_0 = (md_v\w*)\n", loop)
    assert sub is not None and sub.start() > loop.index(f"{RT}.lds_b32(")
    assert loop.count(f"{RT}.stg_b32(") == 8
    assert sub.start() < loop.index(f"{RT}.stg_b32(") < loop.rindex(f"{RT}.mma_bf16(")
    # One barrier per iteration; the staged snapshot of the carry.
    assert loop.count("cute.arch.barrier()") == 1
    assert loop.count(f"{RT}.sts_b32(") == 16 and loop.count(f"{RT}.stg_v4_b32(") == 4


def test_fwd_h_delta_exchanges_through_parity_buffers_at_eight_warps() -> None:
    # Eight warps cannot give every warp whole K rows of a 32-wide fragment
    # (four atom columns), so both B operands go through shared bf16 tiles:
    # the carry's before the iteration barrier (double-buffered by parity),
    # the derived value's behind its own barrier mid-iteration.
    body = _kernel_body(_source(_bind_fwd_h_delta(), _config(32, warps=8, stages=2)))
    assert re.findall(r"^\s*(md_[txd]\d+r?) = md_s0", body, re.MULTILINE) == [
        "md_t0",
        "md_x1",
        "md_t2",
        "md_x3",
        "md_d0",
    ]
    assert "md_x1 = md_s0 + 32768" in body and "md_t2 = md_s0 + 49152" in body
    loop = _loop_and_after(body)
    assert "md_parity = md_i % 2" in loop
    assert f"{RT}.movmatrix_trans(" not in loop and "md_bw" not in loop
    assert loop.count("cute.arch.barrier()") == 2
    assert loop.count(f"{RT}.sts_b32(md_x1 + md_parity * 8192 + ") == 8
    assert loop.count(f"{RT}.sts_b32(md_x3 + ") == 4
    wait = loop.index("cute.arch.cp_async_wait_group(")
    first_barrier = loop.index("cute.arch.barrier()")
    second_barrier = loop.rindex("cute.arch.barrier()")
    assert loop.rindex(f"{RT}.sts_b32(md_x1") < wait < first_barrier
    assert (
        first_barrier
        < loop.index(f"{RT}.mma_bf16(")
        < loop.index(f"{RT}.sts_b32(md_x3")
    )
    assert (
        loop.rindex(f"{RT}.sts_b32(md_x3")
        < second_barrier
        < loop.rindex(f"{RT}.mma_bf16(")
    )
    assert loop.count(f"{RT}.ldmatrix_x4_trans(md_x1 + md_parity * 8192 + ") == 8
    assert "alloc_smem(cutlass.Uint8, 110592, alignment=128)" in body
    # The exchange knob forces the same routes at four warps.
    smem = _kernel_body(
        _source(
            _bind_fwd_h_delta(),
            _config(32, warps=4, stages=2, cute_mma_dag_exchange="smem"),
        )
    )
    assert f"{RT}.movmatrix_trans(" not in smem and "md_x1 = md_s0 + " in smem


def test_gated_fwd_h_delta_rescales_the_carry_after_the_first_dot_reads_it() -> None:
    bound = _bind_fwd_h_delta(gated=True)
    body = _kernel_body(_source(bound, _config(32, warps=4, stages=2)))
    loop = _loop_and_after(body)
    assert "exp2" in loop
    # k * exp2(dl - gc)[:, None] is the in-place pass on the staged k tile
    # (1024 packets over 128 threads).
    assert loop.count(f"{RT}.sts_v4_b32(") == 8
    # The carry feeds dot1 as it was at loop entry; its scalar rescale (the
    # accumulator of dot2) is emitted between the two dots.
    rescale = re.search(r"md_c0_0_0_0 = (md_v\w*)\n", loop)
    assert rescale is not None
    assert (
        loop.index(f"{RT}.mma_bf16(") < rescale.start() < loop.rindex(f"{RT}.mma_bf16(")
    )
    assert loop.rindex(f"{RT}.movmatrix_trans(") > rescale.start()
    mmas = [m.start() for m in re.finditer(rf"{RT}\.mma_bf16\(", loop)]
    assert mmas[31] < rescale.start() < mmas[32]


def test_dstate_emits_the_attention_transposed_and_exchanges_it_once() -> None:
    bound = _bind_dstate()
    program = _program(bound)
    attn, dv_i, dv_i_1, store, ds, dot_4, sub = program.steps[1:]
    # The orientation rule: attn is read only as attn^T, so the product is
    # emitted transposed (k feeds A, q feeds B) and dv_i reads it straight.
    assert isinstance(attn, dag.Dot) and attn.transposed and attn.acc is None
    assert attn.a.load is not None and attn.a.load.name == "k" and attn.a.flip
    assert attn.b.load is not None and attn.b.load.name == "q" and attn.b.flip
    assert isinstance(dv_i, dag.Dot) and dv_i.a.frag is attn.node
    assert not dv_i.a.swap and not dv_i.a.trans
    # k @ dS accumulates into dv_i's slot (the earlier result of the same
    # iteration); dvni is stored mid-iteration; dS = dot(acc=dS) - dot(w^T,
    # dvni) is a derived value in the carry's slot.
    assert isinstance(dv_i_1, dag.Dot) and dv_i_1.acc_slot == dv_i.slot
    assert dv_i_1.b.frag in program.carry_placeholders
    assert isinstance(store, dag.Store) and store.source == "body"
    assert isinstance(ds, dag.Dot) and ds.slot == 0 and ds.acc_slot == 0
    assert isinstance(dot_4, dag.Dot) and dot_4.acc is None
    assert dot_4.b.frag is dv_i_1.node
    assert isinstance(sub, dag.Derived) and sub.slot == 0
    assert sub.frags == (ds.node, dot_4.node) and sub.node is program.carries[0].output
    body = _kernel_body(_source(bound, _config(32, warps=4, stages=2)))
    assert "7 + -1 * md_i" in body
    # Four warps: dS -> B and dvni -> B from registers (N-first grids for
    # [128, 32] and [64, 32]); the [64, 64] attention is the one exchange.
    tiles = re.findall(r"^\s*(md_[tx]\d+r?) = md_s0", body, re.MULTILINE)
    assert tiles == ["md_t0", "md_t1", "md_x2", "md_t3", "md_t4"]
    loop = _loop_and_after(body)
    assert loop.count(f"{RT}.movmatrix_trans(") == 16 + 8
    assert loop.count("cute.arch.barrier()") == 2
    # attn^T is written at its own coordinates and read as A with the
    # straight ldmatrix (M rows, K contiguous); the exchange is single
    # buffered (written behind the iteration barrier).
    assert loop.count(f"{RT}.sts_b32(md_x2 + ") == 16
    assert loop.count(f"{RT}.ldmatrix_x4(md_x2 + ") == 16
    assert f"{RT}.ldmatrix_x4_trans(md_x2" not in loop
    # 64 + 16 + 32 + 32 + 32 mma.sync; the dvni pair stores; the final
    # subtraction into the carry registers.
    assert loop.count(f"{RT}.mma_bf16(") == 176
    assert loop.count(f"{RT}.stg_b32(") == 8
    assert re.search(r"md_c0_0_0_0 = md_v\w*\n", loop) is not None
    assert loop.rindex(f"{RT}.mma_bf16(") < loop.rindex("md_c0_0_0_0 = md_v")
    assert "alloc_smem(cutlass.Uint8, 131072, alignment=128)" in body


def test_dstate_at_eight_warps_exchanges_every_fragment_operand() -> None:
    body = _kernel_body(_source(_bind_dstate(), _config(32, warps=8, stages=2)))
    tiles = re.findall(r"^\s*(md_[tx]\d+r?) = md_s0", body, re.MULTILINE)
    assert tiles == ["md_t0", "md_t1", "md_x2", "md_t3", "md_x4", "md_t5", "md_x6"]
    loop = _loop_and_after(body)
    assert f"{RT}.movmatrix_trans(" not in loop
    # The carry's tile before the iteration barrier (parity buffers), attn
    # and dvni behind their own: three barriers per iteration.
    assert loop.count("cute.arch.barrier()") == 3
    assert loop.count(f"{RT}.sts_b32(md_x4 + md_parity * 8192 + ") == 8
    assert "alloc_smem(cutlass.Uint8, 151552, alignment=128)" in body
    assert loop.count(f"{RT}.mma_bf16(") == 88


def test_gated_dstate_adds_the_scaled_extra_dot_into_the_derived_value() -> None:
    bound = _bind_dstate(gated=True)
    program = _program(bound)
    kinds = [type(step).__name__ for step in program.steps]
    assert kinds == [
        "Store",
        "Dot",
        "Dot",
        "Dot",
        "Derived",
        "Store",
        "Dot",
        "Dot",
        "Derived",
    ]
    dv_kh, dv_i_1 = program.steps[3], program.steps[4]
    assert isinstance(dv_kh, dag.Dot) and dv_kh.acc is None
    assert isinstance(dv_i_1, dag.Derived) and dv_i_1.frags[1] is dv_kh.node
    assert dv_i_1.slot == program.frag_slots[dv_i_1.frags[0]]
    ds = program.steps[6]
    assert isinstance(ds, dag.Dot) and ds.acc is not None and ds.acc_frag is not None
    assert ds.acc is not ds.acc_frag
    body = _kernel_body(_source(bound, _config(32, warps=8, stages=2)))
    tiles = re.findall(r"^\s*(md_[tx]\d+r?) = md_s0", body, re.MULTILINE)
    # do * scale and (do * scale * exp2(gc)[:, None]) are two tiles of do.
    assert tiles == [
        "md_t0",
        "md_t1",
        "md_x2",
        "md_t3",
        "md_x4",
        "md_t5",
        "md_t6",
        "md_x7",
    ]
    loop = _loop_and_after(body)
    assert "exp2" in loop
    mmas = [m.start() for m in re.finditer(rf"{RT}\.mma_bf16\(", loop)]
    # dv_i + dv_kh * exp2(dl - gc)[:, None] lands in dv_i's slot after the
    # third dot (32 + 8 + 16 mma.sync); the scalar rescale of the carry
    # (the accumulator of q^T @ dog) follows it and precedes that dot.
    add = re.search(r"md_c2_0_0_0 = md_v\w*\n", loop)
    rescale = re.search(r"md_c0_0_0_0 = md_v\w*\n", loop)
    assert add is not None and rescale is not None
    assert mmas[55] < add.start() < rescale.start() < mmas[56]
    assert loop.count(f"{RT}.mma_bf16(") == 88


def test_the_register_budget_follows_the_live_fragments() -> None:
    # dstate at bdv = 32: dS (4096) + dv_i (2048) + the larger of attn (4096,
    # dead after its consumer) and the subtracted dot [D, bdv] (4096): 10240
    # words, 40 registers per thread over eight warps, 80 over four.  At bdv
    # = 128 the subtracted dot is as large as the carry: 40960 words are 320
    # registers over four warps and 160 over eight, refused either way; the
    # repair halves the tile.
    bound = _bind_dstate(dv=128)
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    # One live set per loop step (the epilogue has no steps of its own).
    assert len(facts.live_groups) == len(_program(bound).steps)
    resolved = [1, 32, 128, 64]
    peak = max(
        sum(resolved[r] * resolved[c] for r, c in (facts.frags[s] for s in group))
        for group in facts.live_groups
    )
    assert (
        peak == 10240 and sum(resolved[r] * resolved[c] for r, c in facts.frags) > peak
    )
    assert dag.mma_dag_config_reason(facts, {0: 1, 1: 32}, 8, 2) is None
    assert dag.mma_dag_config_reason(facts, {0: 1, 1: 32}, 4, 2) is None
    reason = dag.mma_dag_config_reason(facts, {0: 1, 1: 128}, 4, 1)
    assert reason is not None and "320 registers" in reason
    reason = dag.mma_dag_config_reason(facts, {0: 1, 1: 128}, 8, 1)
    assert reason is not None and "160 registers" in reason
    assert dag.mma_dag_config_reason(facts, {0: 1, 1: 64}, 8, 1) is None
    config = _config(128, warps=4, stages=1)
    assert spec.backend.autotune_config_is_viable(spec, config) is False
    with bound.env:
        with pytest.raises(InvalidConfig, match="registers per thread"):
            spec.normalize(dict(config.config))
        repaired = dict(config.config)
        spec.normalize(repaired, _fix_invalid=True)
    assert repaired["cute_mma_dag_warps"] == 8 and repaired["block_sizes"] == [64]


def test_delta_seeds_prefer_exchange_free_warps_then_the_budget_warps() -> None:
    # fwd_h_delta: four warps feed both B operands from registers (no
    # exchange tile), eight warps need two tiles and a second barrier per
    # iteration; both are seeded, the exchange-free count first and
    # promoted.  dstate exchanges its attention tile at every warp count, so
    # only the most warps within the budget are seeded.  At s2 (32 heads,
    # DV 128) the 32-wide tile reaches 128 CTAs and ranks first.
    bound = _bind_fwd_h_delta(bh=32, dv=128)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    assert [(e.role, e.region) for e in facts.edges] == [("b", "loop"), ("b", "loop")]
    resolved = [1, 32, 128, 64]
    layouts, routes, loop_bytes = dag.mma_dag_layout_solution(facts, resolved, 4)
    assert routes == (dag.ROUTE_MOVMATRIX, dag.ROUTE_MOVMATRIX) and loop_bytes == 0
    assert layouts == {(64, 32): (1, 4), (128, 32): (1, 4)}
    _layouts, routes, loop_bytes = dag.mma_dag_layout_solution(facts, resolved, 8)
    assert routes == (dag.ROUTE_EXCHANGE, dag.ROUTE_EXCHANGE)
    assert loop_bytes == 128 * 32 * 2 + 64 * 32 * 2
    # 128 CTAs on 148 SMs: the deeper ring ranks above two CTAs per SM
    # (eight stages do not fit beside the staging tile).
    seeds = [(s[0][1], s[1], s[2]) for s in dag.mma_dag_seed_configs(facts)]
    assert seeds[:4] == [(32, 4, 4), (32, 4, 2), (32, 8, 4), (32, 8, 2)]
    default = bound.config_spec.default_config().config
    assert default["block_sizes"] == [32]
    assert (default["cute_mma_dag_warps"], default["cute_mma_dag_stages"]) == (4, 4)
    assert default["cute_mma_dag_store"] == "staged"
    # The gated variant rescales k in place: that per-element pass splits
    # across the threads, so the budget warps lead and the exchange-free
    # four follow.
    gated = _bind_fwd_h_delta(gated=True, bh=32, dv=128)
    facts_gated = gated.config_spec.cute_mma_dag_facts
    assert facts_gated is not None and facts_gated.loop_passes == 1
    seeds = [(s[0][1], s[1], s[2]) for s in dag.mma_dag_seed_configs(facts_gated)]
    assert seeds[:4] == [(32, 8, 4), (32, 8, 2), (32, 4, 4), (32, 4, 2)]
    first_warp = next(
        c.config
        for c in gated.config_spec.compiler_seed_configs
        if c.config["cute_mma_dag"] == "warp"
    )
    assert first_warp["cute_mma_dag_warps"] == 8
    # The route-aware shared-memory estimate: at bdv 128 both B operands
    # come from registers at eight warps, so no exchange tile is charged
    # and two stages fit (the tiles alone would not).
    wide = [1, 128, 128, 64]
    assert dag.mma_dag_config_reason(facts, {0: 1, 1: 128}, 8, 2) is None
    assert dag.mma_dag_smem_estimate(facts, wide, 2, 8) == 2 * (16384 * 3) + 2 * 32768
    assert dag.mma_dag_smem_estimate(facts, wide, 2) > facts.smem_capacity
    dstate = _bind_dstate(bh=32, dv=128)
    facts = dstate.config_spec.cute_mma_dag_facts
    assert facts is not None
    assert [e.role for e in facts.edges] == ["a", "b", "b"]
    _layouts, routes, loop_bytes = dag.mma_dag_layout_solution(facts, resolved, 4)
    assert routes == (dag.ROUTE_EXCHANGE, dag.ROUTE_MOVMATRIX, dag.ROUTE_MOVMATRIX)
    assert loop_bytes == 64 * 64 * 2
    seeds = [(s[0][1], s[1], s[2]) for s in dag.mma_dag_seed_configs(facts)]
    assert seeds[0] == (32, 8, 2)
    assert all(warps == 8 for _bdv, warps, _stages in seeds)
    # dstate's promoted default is the tensor-memory mode; the first warp
    # seed carries the budget warps.
    default = dstate.config_spec.default_config().config
    assert default["block_sizes"] == [32] and default["cute_mma_dag"] == "tcgen05"
    first_warp = next(
        c.config
        for c in dstate.config_spec.compiler_seed_configs
        if c.config["cute_mma_dag"] == "warp"
    )
    assert first_warp["block_sizes"] == [32] and first_warp["cute_mma_dag_warps"] == 8


def test_fp32_snapshots_seed_the_direct_store_path() -> None:
    # bwd_dh snapshots the fp32 carry: 8-byte pairs either way, so the
    # direct pair stores skip the staging round trip; fwd_h's bf16
    # snapshot keeps the 16-byte packets of the staging tile.
    # The tensor-memory default stages its snapshots (the TMEM rows are
    # packed into the staging tile either way).
    bwd = _bind_bwd().config_spec
    assert bwd.default_config().config["cute_mma_dag_store"] == "staged"
    warp_seeds = [
        c.config
        for c in bwd.compiler_seed_configs
        if c.config.get("cute_mma_dag") == "warp"
    ]
    assert warp_seeds and all(c["cute_mma_dag_store"] == "direct" for c in warp_seeds)
    fwd = _bind_fwd().config_spec
    assert fwd.default_config().config["cute_mma_dag_store"] == "staged"
    body = _kernel_body(_source(_bind_bwd(), helion.Config(**warp_seeds[0])))
    assert f"{RT}.stg_v2_f32(" in body and f"{RT}.stg_v4_b32(" not in body


def test_a_loop_dot_accumulating_onto_an_earlier_dot_result_runs_in_order() -> None:
    # Both dots accumulate in place into the carry's slot; the second dot's
    # accumulator (``x * 0.5``) is materialized after the first dot's
    # mma.sync and before its own, never hoisted above the first dot (the
    # dstate chain ``dv_i = dot(attn^T, do); dv_i = dot(k, Cb(dS), acc=dv_i)``
    # has the same shape).
    t = _tensors()
    bound = _bind_fn(_loop_chain, (t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is not None
    program = _program(bound)
    first, second = (step for step in program.steps if isinstance(step, dag.Dot))
    assert first.slot == second.slot == second.acc_slot == 0
    assert second.acc_frag is first.node
    loop = _loop_and_after(
        _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    )
    mmas = [m.start() for m in re.finditer(rf"{RT}\.mma_bf16\(", loop)]
    assert len(mmas) == 32 + 32
    rescale = re.search(r"md_c0_0_0_0 = md_v\w*\n", loop)
    assert rescale is not None
    assert mmas[31] < rescale.start() < mmas[32]
    assert "md_c1_" not in loop


def _emulate_movmatrix_trans(words: list[list[int]]) -> list[list[int]]:
    """``movmatrix.trans`` on one 8x8 block held in the C layout (lane
    ``(g, c)`` holds row ``g``, columns ``2c``, ``2c + 1``): the lane then
    holds column ``g``, rows ``2c``, ``2c + 1``."""
    block = [[0] * 8 for _ in range(8)]
    for lane in range(32):
        g, c = lane // 4, lane % 4
        block[g][2 * c], block[g][2 * c + 1] = words[lane]
    return [
        [block[2 * (lane % 4)][lane // 4], block[2 * (lane % 4) + 1][lane // 4]]
        for lane in range(32)
    ]


@pytest.mark.parametrize(
    ("role", "route", "producer", "consumer", "warps", "n_first"),
    (
        ("a", dag.ROUTE_IDENTITY, (64, 64), (64, 64), 4, (False, False)),
        ("a", dag.ROUTE_IDENTITY, (32, 128), (32, 16), 2, (False, False)),
        ("b", dag.ROUTE_MOVMATRIX, (128, 32), (64, 32), 4, (True, True)),
        ("b", dag.ROUTE_MOVMATRIX, (64, 64), (128, 64), 8, (True, True)),
        ("b", dag.ROUTE_IDENTITY, (64, 64), (16, 64), 4, (False, True)),
    ),
)
def test_register_routed_operand_words_yield_the_mma_fragments(
    role: str,
    route: str,
    producer: tuple[int, int],
    consumer: tuple[int, int],
    warps: int,
    n_first: tuple[bool, bool],
) -> None:
    # The producer fragment holds element identities in the mma C layout;
    # packing its registers per ``frag_word_blocks`` (transposing each 8x8
    # block for a [K, N] B operand) must give every consumer lane the A / B
    # fragment of the logical operand for every (atom, K step).
    rows_p, cols_p = producer
    rows_c, cols_c = consumer
    layout_p = dag.mma_dag_warp_layout(rows_p, cols_p, warps, n_first=n_first[0])
    layout_c = dag.mma_dag_warp_layout(rows_c, cols_c, warps, n_first=n_first[1])
    assert layout_p is not None and layout_c is not None
    wm_p, wn_p = layout_p
    wm_c, wn_c = layout_c
    wtm_p, wtn_p = rows_p // wm_p, cols_p // wn_p
    wtm_c, wtn_c = rows_c // wm_c, cols_c // wn_c
    if role == "a":
        k = cols_p
        a = [[r * 1000 + c for c in range(k)] for r in range(rows_p)]
    elif route == dag.ROUTE_MOVMATRIX:
        k = rows_p  # [K, N]: B[k][n] = P[k][n]
        b = [[r * 1000 + c for c in range(cols_p)] for r in range(rows_p)]
    else:
        k = cols_p  # [N, K]: B[k][n] = P[n][k]
        b = [[n * 1000 + kk for n in range(rows_p)] for kk in range(k)]
    for warp in range(warps):
        wmi_p, wni_p = warp // wn_p, warp % wn_p
        wmi_c, wni_c = warp // wn_c, warp % wn_c
        atoms = wtm_c // 16 if role == "a" else wtn_c // 8
        for atom in range(atoms):
            for st in range(k // 16):
                blocks = dag.frag_word_blocks(role, route, atom, st)
                words_by_lane: list[list[list[int]]] = []
                for lane in range(32):
                    words = []
                    for am, an, half in blocks:
                        pair = [
                            dag.accumulator_coordinates(
                                lane=lane,
                                am=am,
                                an=an,
                                j=2 * half + bit,
                                wmi=wmi_p,
                                wni=wni_p,
                                wtm=wtm_p,
                                wtn=wtn_p,
                            )
                            for bit in range(2)
                        ]
                        words.append([r * 1000 + c for r, c in pair])
                    words_by_lane.append(words)
                if route == dag.ROUTE_MOVMATRIX:
                    for w in range(len(blocks)):
                        transposed = _emulate_movmatrix_trans(
                            [words_by_lane[lane][w] for lane in range(32)]
                        )
                        for lane in range(32):
                            words_by_lane[lane][w] = transposed[lane]
                for lane in range(32):
                    if role == "a":
                        m0 = wmi_c * wtm_c + 16 * atom
                        block = [
                            [a[m0 + i][16 * st + kk] for kk in range(16)]
                            for i in range(16)
                        ]
                        assert words_by_lane[lane] == _a_fragment(block, lane), (
                            warp,
                            lane,
                            atom,
                            st,
                        )
                    else:
                        n0 = wni_c * wtn_c + 8 * atom
                        block_b = [
                            [b[16 * st + kk][n0 + i] for i in range(8)]
                            for kk in range(16)
                        ]
                        expected = _b_fragment(
                            [row + [0] * 8 for row in block_b], lane
                        )[:2]
                        assert words_by_lane[lane] == expected, (warp, lane, atom, st)


# --------------------------------------------------------------------------
# Regression pins for the state recurrences
# --------------------------------------------------------------------------


def _fixed_tiles_recurrence(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, 32, 32]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
    return h_all


def _scalar_decay_recurrence(
    k: torch.Tensor, v: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = h_acc * s[idx, i_t].float()
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
    return h_all


def _masked_recurrence(
    k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d]
            m = valid[idx, i_t, :]
            v_i = torch.where(m[:, None], v[idx, i_t, :, tile_dv], 0.0)
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
    return h_all


def _lossy_cast_recurrence(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            h_all[idx, i_t, tile_d, tile_dv] = h_acc.to(h_all.dtype)
            k_i = k[idx, i_t, :, tile_d].to(torch.float16).to(torch.bfloat16)
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
    return h_all


def _bind_fn(fn: object, args: tuple[object, ...]) -> BoundKernel:
    return helion.kernel(fn, backend="cute", static_shapes=True).bind(args)  # type: ignore[arg-type]


def test_sync_tiles_ring_with_the_stages_or_add_a_trailing_barrier() -> None:
    # An unaligned k (synchronous pass) beside v's cp.async ring.  With two
    # stages the k tile has two 8 KB slots, so the pass of iteration i + 1
    # (before any barrier) writes the slot the lanes of iteration i did not
    # read; v's ring follows at 16 KB.
    t = _tensors()
    k = torch.empty((2, 8, 64, 80), dtype=torch.bfloat16, device=CPU_DEVICE)[..., 4:68]
    bound = _bind(
        "chunk_fwd_h_diag_fused",
        (k, t["v"], None, None, None, False, False, False, False),
    )
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    loop = _loop_and_after(body)
    assert "md_t1 = md_s0 + 16384" in body
    assert loop.count(f"{RT}.sts_b32(md_t0 + md_stage * 8192 + ") == 16
    assert loop.count("cute.arch.barrier()") == 1
    barrier = loop.index("cute.arch.barrier()")
    assert loop.rindex(f"{RT}.sts_b32(md_t0") < barrier < loop.index(f"{RT}.ldmatrix")
    # One stage: a single slot, so a barrier after the reads separates them
    # from the next iteration's rewrite.
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=1)))
    loop = _loop_and_after(body)
    assert "md_t1 = md_s0 + 8192" in body
    assert loop.count("cute.arch.barrier()") == 2
    trailing = loop.rindex("cute.arch.barrier()")
    assert trailing > loop.rindex(f"{RT}.mma_bf16(")
    assert trailing > loop.rindex(f"{RT}.ldmatrix")


def test_fixed_fragment_block_sizes_are_not_mine() -> None:
    # The M / N tiles must be tunable (the knobs resize them).
    t = _tensors()
    bound = _bind_fn(_fixed_tiles_recurrence, (t["k"], t["v"]))
    spec = bound.config_spec
    assert spec.cute_mma_dag is None
    assert "cute_mma_dag" not in spec.default_config().config
    source = bound.to_code(spec.default_config())
    assert SIMT_FALLBACK in source and f"{RT}." not in source
    with bound.env, pytest.raises(InvalidConfig, match="cute_mma_dag"):
        spec.normalized_config(
            helion.Config(
                cute_mma_dag="warp", cute_mma_dag_warps=2, cute_mma_dag_stages=2
            )
        )


def _plain_tiled_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def test_contractions_the_tensor_core_families_claim_are_not_mine() -> None:
    # A plain tiled GEMM is a tcgen05 / warp-MMA candidate: the dag neither
    # adds knobs nor touches the block-size fragments, so the config and the
    # code of the ordinary lowering are what they were before the owner.
    x = torch.empty((64, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    y = torch.empty((128, 2), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_plain_tiled_gemm, (x, y))
    spec = bound.config_spec
    assert spec.cute_tcgen05_search_enabled
    assert spec.cute_mma_dag is None and spec.cute_mma_dag_facts is None
    assert not any("mma_dag" in key for key in spec.default_config().config)
    with patch.object(CuteMmaDagHeuristic, "_facts", return_value=None):
        plain = _bind_fn(_plain_tiled_gemm, (x, y)).config_spec
    assert _block_bounds(spec) == _block_bounds(plain)
    config = helion.Config(block_sizes=[64, 2, 16])
    with bound.env:
        normalized = spec.normalized_config(config)
    assert normalized.config["block_sizes"] == [64, 2, 16]
    source = bound.to_code(normalized)
    assert f"{RT}." not in source
    assert "make_trivial_tiled_mma" not in source
    # The engine kernels have no such candidate and stay admitted.
    assert _bind_fwd().config_spec.cute_mma_dag is not None
    assert _bind_fwd_o().config_spec.cute_mma_dag is not None


@pytest.mark.parametrize(
    ("dtype", "admitted"),
    (
        (torch.float32, True),
        (torch.bfloat16, True),
        (torch.float64, False),
        (torch.int32, True),
    ),
)
def test_side_loads_need_a_dtype_the_lane_loads_read(
    dtype: torch.dtype, admitted: bool
) -> None:
    # The per-lane scalar / vector loads read f32 / f16 / bf16 (widened to
    # fp32) and int32 (window bounds; the ``.float()`` converts the loaded
    # integer); anything else (here a float64 decay) is not mine rather than
    # a mistyped load.
    t = _tensors()
    s = torch.empty((2, 8), dtype=dtype, device=CPU_DEVICE)
    bound = _bind_fn(_scalar_decay_recurrence, (t["k"], t["v"], s))
    assert (bound.config_spec.cute_mma_dag is not None) is admitted
    if admitted:
        body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
        load = {
            torch.float32: "ldg_f32",
            torch.bfloat16: "ldg_bf16_as_f32",
            torch.int32: "ldg_s32",
        }[dtype]
        assert f"{RT}.{load}(" in _loop_and_after(body)


def test_a_boolean_mask_load_is_not_mine() -> None:
    t = _tensors()
    valid = torch.empty((2, 8, 64), dtype=torch.bool, device=CPU_DEVICE)
    bound = _bind_fn(_masked_recurrence, (t["k"], t["v"], valid))
    assert bound.config_spec.cute_mma_dag is None


def test_one_budget_screens_the_staging_buffers_for_seeds_and_planner() -> None:
    # chunk_bwd_dh at C = D = DV = 128 with fp32 dh.  The 64x128 tiles at four
    # stages take 196608 B of operand rings plus 65536 B of fp32 staging; a
    # 232448 B device refuses that config, so the seeds fall back to two
    # stages, and the planner allocates exactly the estimate.
    bound = _bind_bwd(bh=64, c=128, d=128, dv=128)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    tight = dataclasses.replace(facts, smem_capacity=232448)
    roots = [axis for axis in tight.axes if axis.kind == dag.AXIS_ROOT]
    sizes: dict[int, object] = {
        axis.block_id: 1 for axis in tight.axes if axis.kind == dag.AXIS_LEAD
    }
    sizes[roots[0].block_id], sizes[roots[1].block_id] = 64, 128
    resolved = [tight.block_size(i, sizes) for i in range(len(tight.axes))]
    assert all(size is not None for size in resolved)
    resolved_sizes = [size for size in resolved if size is not None]
    assert dag.mma_dag_smem_estimate(tight, resolved_sizes, 4) == 196608 + 65536
    assert dag.mma_dag_smem_estimate(tight, resolved_sizes, 2) == 163840
    reason = dag.mma_dag_config_reason(tight, sizes, 8, 4)
    assert reason is not None and "shared memory" in reason
    assert dag.mma_dag_config_reason(tight, sizes, 8, 2) is None
    seeds = dag.mma_dag_seed_configs(tight)
    assert seeds
    for seed_sizes, warps, stages in seeds:
        assert dag.mma_dag_config_reason(tight, seed_sizes, warps, stages) is None
    first_sizes, first_warps, first_stages = seeds[0]
    assert (first_sizes[roots[0].block_id] * first_sizes[roots[1].block_id]) == 8192
    assert (first_warps, first_stages) == (8, 2)
    # The planner's allocation is the estimate (routes selected, store knob
    # staged): the same number normalize and the viability screen use.
    source = _source(bound, _config(64, 128, warps=8, stages=2))
    assert "alloc_smem(cutlass.Uint8, 163840, alignment=128)" in source


def test_a_lossy_cast_chain_on_an_operand_is_evaluated_per_element() -> None:
    # k -> f16 -> bf16 is not an identity; the operand takes the in-place pass
    # and every element is rounded through f16.
    t = _tensors()
    bound = _bind_fn(_lossy_cast_recurrence, (t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    loop = _loop_and_after(body)
    assert loop.count(f"{RT}.sts_v4_b32(") == 4
    assert "Float16(" in loop


def test_emitted_lane_coordinates_and_ldmatrix_addresses_follow_the_model() -> None:
    # The strings the body emits, evaluated for every thread, are the
    # coordinate model the proofs above cover.
    body = _kernel_body(_source(_bind_fwd(), _config(64, 64, warps=4, stages=2)))
    loop = _loop_and_after(body)
    prelude = re.findall(
        r"^\s*(md_(?:lane|g|c|l8|lq|L0_\w+)) = (.*)$", body, re.MULTILINE
    )
    binds = re.findall(
        r"^\s*(md_(?:arow|achunk|brow|bchunk)) = (.*)$", loop, re.MULTILINE
    )
    calls = re.findall(
        rf"^\s*(md_[ab])_\d+_\d+(?:, md_\w+)* = {RT}\.ldmatrix_x4_trans\((.*)\)$",
        loop,
        re.MULTILINE,
    )
    assert len(prelude) == 9 and len(binds) == 4 and len(calls) == 20
    wm, wn = cast("tuple[int, int]", warp_mma_warp_layout(64, 64, 4))
    wtm, wtn = 64 // wm, 64 // wn
    layout = _Layout(0, 64, 64, 4)
    # Per K step the loop reads the A atom, then the B atom pairs.
    order = []
    for st in range(64 // 16):
        order.append(("a", st, 0))
        order.extend(("b", st, an) for an in range(0, wtn // 8, 2))
    for warp in range(4):
        for lane in range(32):
            ns: dict[str, int] = {
                "md_tidx": 32 * warp + lane,
                "md_warp": warp,
                "md_stage": 0,
                "md_t0": 0,
                "md_t1": 0,
            }
            for name, expr in prelude + binds:
                ns[name] = eval(expr, {}, ns)
            wmi, wni = ns["md_L0_wmi"], ns["md_L0_wni"]
            for am in range(wtm // 16):
                for an in range(wtn // 8):
                    for j in range(4):
                        got = (
                            eval(layout.row(am, j // 2), {}, ns),
                            eval(layout.col(an, j & 1), {}, ns),
                        )
                        assert got == dag.accumulator_coordinates(
                            lane=lane,
                            am=am,
                            an=an,
                            j=j,
                            wmi=wmi,
                            wni=wni,
                            wtm=wtm,
                            wtn=wtn,
                        )
            for (role, st, atom), (prefix, expr) in zip(order, calls, strict=True):
                assert prefix == f"md_{role}"
                offset = eval(expr, {}, ns)
                row, physical = offset // 128, (offset % 128) // 16
                chunk = next(
                    c for c in range(8) if dag.swizzled_chunk(row, c, 8) == physical
                )
                mn_base = wmi * wtm + 16 * atom if role == "a" else wni * wtn + 8 * atom
                assert (row, chunk) == dag.ldmatrix_lane_address(
                    role=role, trans=True, lane=lane, mn_base=mn_base, k_step=st
                ), (warp, lane, role, st, atom)


def test_normalize_pins_the_collective_and_gathered_knobs_when_on() -> None:
    # Knobs of other owners are inert once the body is replaced; pinning them
    # keeps equal kernels on one autotune identity.
    bound = _bind_fwd()
    config = _config(
        64,
        64,
        warps=4,
        stages=2,
        cute_collective_copy="async",
        cute_collective_stages=3,
        cute_gathered_mma_n=128,
        cute_proven_bounds=True,
    ).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_collective_copy"] == "scalar"
    assert config["cute_collective_stages"] == 1
    assert config["cute_gathered_mma_n"] == 256
    assert config["cute_proven_bounds"] is False


# --------------------------------------------------------------------------
# Regression pins for the K-loop kernels: accumulator slots, exchange order,
# [C, C] stores, sharing keys, budgets and inert knobs
# --------------------------------------------------------------------------


def _acc_frag_stored(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    out2 = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
        vt = v[tile_bhn, :, tile_dv]
        out2[tile_bhn, :, tile_dv] = o_cross.to(out2.dtype)
        o = hl.dot(attn.to(vt.dtype), vt, acc=o_cross)
        out[tile_bhn, :, tile_dv] = o.to(out.dtype)
    return out, out2


def _acc_dag_stored(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    out2 = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
        t = o_cross * 2.0
        vt = v[tile_bhn, :, tile_dv]
        o = hl.dot(attn.to(vt.dtype), vt, acc=t)
        out[tile_bhn, :, tile_dv] = o.to(out.dtype)
        out2[tile_bhn, :, tile_dv] = t.to(out2.dtype)
    return out, out2


def _epi_chain(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    q2: torch.Tensor,
    k2: torch.Tensor,
) -> torch.Tensor:
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    hl.specialize(q2.size(2))
    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
        attn2 = hl.dot(
            q2[tile_bhn, :, :], k2[tile_bhn, :, :].transpose(-2, -1), acc=attn
        )
        vt = v[tile_bhn, :, tile_dv]
        o = hl.dot(attn2.to(vt.dtype), vt, acc=o_cross)
        out[tile_bhn, :, tile_dv] = o.to(out.dtype)
    return out


def _cc_store(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    a_out = torch.empty([BHN, C, C], dtype=q.dtype, device=q.device)
    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
        idx = hl.arange(C)
        causal = idx[:, None] >= idx[None, :]
        attn = torch.where(causal, attn, 0.0)
        a_out[tile_bhn, :, :] = attn.to(a_out.dtype)
        vt = v[tile_bhn, :, tile_dv]
        o = hl.dot(attn.to(vt.dtype), vt, acc=o_cross)
        out[tile_bhn, :, tile_dv] = o.to(out.dtype)
    return out, a_out


def _cc_load_operand(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    a_in: torch.Tensor,
) -> torch.Tensor:
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
        vt = v[tile_bhn, :, tile_dv]
        o = hl.dot(a_in[tile_bhn, :, :], vt, acc=o_cross)
        out[tile_bhn, :, tile_dv] = o.to(out.dtype)
    return out


def _chain_collide(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
        gc = g[tile_bhn, :]
        col = gc[:, :, None]
        x1 = attn - col
        x2 = col - attn
        vt = v[tile_bhn, :, tile_dv]
        o = hl.dot(x1.to(vt.dtype), vt, acc=o_cross)
        o = hl.dot(x2.to(vt.dtype), vt, acc=o)
        out[tile_bhn, :, tile_dv] = o.to(out.dtype)
    return out


def _two_dots_same_carry(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    x_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    h_all = torch.empty([BH, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            x = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            x_all[idx, i_t, tile_d, tile_dv] = x.to(x_all.dtype)
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc * 0.5)
        h_all[idx, tile_d, tile_dv] = h_acc.to(h_all.dtype)
    return x_all, h_all


def _two_dots_carry_read_after(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    x_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    h_all = torch.empty([BH, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_new = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc * 0.5)
            x = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            x_all[idx, i_t, tile_d, tile_dv] = x.to(x_all.dtype)
            h_acc = h_new
        h_all[idx, tile_d, tile_dv] = h_acc.to(h_all.dtype)
    return x_all, h_all


def _epilogue_of(body: str) -> str:
    loop = _loop_and_after(body)
    return loop[loop.rindex("cute.arch.cp_async_wait_group(0)") :]


def test_an_accumulator_with_other_readers_gets_a_fresh_slot() -> None:
    # ``o_cross`` is stored and then accumulated into.  The store reads slot 0
    # at its program point; a store after the dot makes the dot write a fresh
    # slot.
    t = _tensors3()
    args = (t["q"], t["q"], t["v"], t["h"])
    for fn in (_acc_frag_stored, _acc_dag_stored):
        bound = _bind_fn(fn, args)
        assert bound.config_spec.cute_mma_dag is not None, fn.__name__
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None
        epilogue = _epilogue_of(
            _kernel_body(_source(bound, _config(1, 64, 64, warps=4, stages=2)))
        )
        mma = epilogue.index(f"{RT}.mma_bf16(")
        stores = [m.start() for m in re.finditer(rf"{RT}\.stg_b32\(", epilogue)]
        assert len(stores) == 32
        packed = re.findall(
            rf"{RT}\.stg_b32\(.*?pack_bf16x2\((md_\w+), (md_\w+)\)\)", epilogue
        )
        definitions = dict(
            re.findall(r"^\s*(md_v\w*|v_\w+) = (.*)$", epilogue, re.MULTILINE)
        )
        if fn is _acc_frag_stored:
            # The out2 store precedes the dot in program order and is
            # emitted there, so the dot may accumulate in place (two slots).
            assert len(facts.frags) == 2
            assert stores[15] < mma < stores[16]
            assert "md_c2_" not in epilogue
            out2 = packed[:16]
        else:
            # ``t = o_cross * 2`` is stored after the dot: the dot writes a
            # third slot and the store still derives from slot 0.
            assert len(facts.frags) == 3
            assert mma < stores[0]
            assert "md_c2_0_0_0, md_c2_0_0_1, md_c2_0_0_2, md_c2_0_0_3 = " in epilogue
            assert f"md_c0_0_0_0 = {RT}" not in epilogue
            out2 = packed[16:]
        for names in out2:
            registers = _register_sources(definitions, names)
            assert registers and all(r.startswith("md_c0_") for r in registers), (
                registers
            )


def _register_sources(definitions: dict[str, str], names: tuple[str, ...]) -> set[str]:
    """The fragment registers the values ``names`` are computed from."""
    registers: set[str] = set()
    stack = list(names)
    seen: set[str] = set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name.startswith("md_c"):
            registers.add(name)
            continue
        for ident in re.findall(r"\b(?:md_|v_)\w+", definitions.get(name, "")):
            if ident.startswith("md_c"):
                registers.add(ident)
            elif ident.startswith(("md_v", "v_")):
                stack.append(ident)
    return registers


def test_two_loop_dots_on_one_carry_copy_before_the_rescale() -> None:
    # ``x`` copies the carry into slot 1 before the producer rescales slot 0 in
    # place; the reversed program order would read a rescaled carry, so it is
    # not mine.
    t = _tensors()
    bound = _bind_fn(_two_dots_same_carry, (t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    loop = _loop_and_after(body)
    # The copy overlaps the copies in flight (before the wait); the x store
    # follows x's MMAs at its program point; the producer's rescale of slot
    # 0 waits for x (which read slot 0) and precedes the producer's MMAs.
    copy = loop.index("md_c1_0_0_0 = md_c0_0_0_0")
    x_mmas = [
        m.start() for m in re.finditer(r"md_c1_0_0_0, \w+, \w+, \w+ = " + RT, loop)
    ]
    h_mmas = [
        m.start() for m in re.finditer(r"md_c0_0_0_0, \w+, \w+, \w+ = " + RT, loop)
    ]
    rescale = loop.index("md_c0_0_0_0 = ", copy + 1)
    store = loop.index(f"{RT}.stg_b32(")
    assert copy < loop.index("cute.arch.cp_async_wait_group(") < x_mmas[0]
    assert x_mmas[-1] < store < rescale < h_mmas[0]
    # With the carry read after the rescaled dot, that dot takes a fresh
    # slot and its result moves into the carry at the iteration end.
    after = _bind_fn(_two_dots_carry_read_after, (t["k"], t["v"]))
    assert after.config_spec.cute_mma_dag is not None
    loop = _loop_and_after(
        _kernel_body(_source(after, _config(64, 64, warps=4, stages=2)))
    )
    copies = re.findall(r"md_c0_\d+_\d+_\d+ = md_c1_\d+_\d+_\d+", loop)
    assert len(copies) == 32
    assert loop.rindex(f"{RT}.mma_bf16(") < loop.index(copies[0])


def test_an_exchange_of_an_epilogue_dot_result_follows_that_dot() -> None:
    # ``attn2`` is produced by the first epilogue dot and read through the
    # exchange tile by the second; the tile is written (and fenced) between the
    # two.
    t = _tensors3()
    q2 = torch.empty((16, 64, 32), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_epi_chain, (t["q"], t["q"], t["v"], t["h"], q2, q2))
    assert bound.config_spec.cute_mma_dag is not None
    for exchange in ("smem", "auto"):
        epilogue = _epilogue_of(
            _kernel_body(
                _source(
                    bound,
                    _config(
                        1, 64, 64, warps=4, stages=2, cute_mma_dag_exchange=exchange
                    ),
                )
            )
        )
        # attn2 accumulates in place into attn's slot (1), o into o_cross's (0).
        attn2_last = epilogue.rindex(
            "md_c1_0_0_0, md_c1_0_0_1, md_c1_0_0_2, md_c1_0_0_3 = "
        )
        o_first = epilogue.index(
            "md_c0_0_0_0, md_c0_0_0_1, md_c0_0_0_2, md_c0_0_0_3 = "
        )
        assert attn2_last < o_first
        if exchange == "smem":
            writes = [m.start() for m in re.finditer(rf"{RT}\.sts_b32\(md_x", epilogue)]
            assert len(writes) == 16
            barrier = epilogue.index("cute.arch.barrier()", writes[-1])
            assert attn2_last < writes[0] < writes[-1] < barrier < o_first
        else:
            assert "md_aw" in epilogue and "sts_b32(md_x" not in epilogue


def test_a_cc_store_over_one_rdim_uses_positional_strides() -> None:
    # ``a_out[tile_bhn, :, :]`` indexes both dims by the same rdim; the row
    # stride is C elements, the column stride one.
    t = _tensors3()
    bound = _bind_fn(_cc_store, (t["q"], t["q"], t["v"], t["h"]))
    assert bound.config_spec.cute_mma_dag is not None
    epilogue = _epilogue_of(
        _kernel_body(_source(bound, _config(1, 64, 64, warps=4, stages=2)))
    )
    # (One lane base per store: the fragment's row and column terms with the
    # positional strides, the registers' offsets folded as immediates.)
    assert re.search(
        r"^\s*md_lb\w* = md_\w+ \+ cutlass\.Int64\(md_L0_rowl\) \* 128 \+ cutlass\.Int64\(md_L0_coll\) \* 2$",
        epilogue,
        re.MULTILINE,
    )
    assert "* 128 + cutlass.Int64(md_L0_coll) * 128" not in epilogue
    assert epilogue.count(f"{RT}.stg_b32(") == 16 + 16
    # A [C, C] load as an operand: the positional strides find its
    # contiguous dim, so it is staged with the row stride of C elements.
    a_in = torch.empty((16, 64, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_cc_load_operand, (t["q"], t["q"], t["v"], t["h"], a_in))
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(1, 64, 64, warps=4, stages=2)))
    assert f"{RT}.cp_async_16(md_t2 + md_pr8 * 128" in body
    assert "cutlass.Int64(md_pr8) * 128 + cutlass.Int64(md_pc8 * 16)" in body


def test_operands_with_different_values_do_not_share_a_tile() -> None:
    # ``attn - col`` and ``col - attn`` differ only in where the fragment sits
    # among the arguments.
    t = _tensors3()
    g = torch.empty((16, 64), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind_fn(_chain_collide, (t["q"], t["q"], t["v"], t["h"], g))
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(
        _source(
            bound, _config(1, 64, 64, warps=4, stages=2, cute_mma_dag_exchange="smem")
        )
    )
    tiles = re.findall(r"^\s*(md_x\d+) = md_s0", body, re.MULTILINE)
    assert len(tiles) == 2
    epilogue = _epilogue_of(body)
    assert epilogue.count(f"{RT}.sts_b32(md_x") == 32
    # dqk's dA read straight and transposed still shares one tile (a
    # transpose commutes with the uniform-only chain).
    dqk = _kernel_body(_source(_bind_dqk(), _config(1, 64, 64, warps=4, stages=2)))
    assert len(re.findall(r"^\s*(md_x\d+) = md_s0", dqk, re.MULTILINE)) == 1


def test_register_budget_counts_the_repacked_a_words_and_b_fragments() -> None:
    # A dot repacking a 128x128 fragment as its A operand over K = 128 with one
    # warp column holds 64 more words than the fragments; with four warps the
    # 128 fragment words plus 96 operand words exceed the live budget, with
    # eight (64 + 64) they fit.
    facts = dataclasses.replace(
        _state_facts(128, 128, 128, heads=1),
        dots=(dag.DotFacts(0, 3, True, False),),
        identity_candidates=True,
    )
    resolved = [1, 128, 128, 128]
    # Four warps split M: 32 rows x 128 K of A words (64) plus one K step of
    # B fragments for 16 n atoms (32).
    assert dag.mma_dag_transient_regs(facts, resolved, 4) == 64 + 32
    message = dag.mma_dag_config_reason(facts, _sizes(128, 128), 4, 1)
    assert message is not None and "registers per thread" in message
    assert dag.mma_dag_config_reason(facts, _sizes(128, 128), 8, 1) is None
    # The engine kernels keep their 8-warp defaults within the budget.
    dqk = _bind_dqk().config_spec.cute_mma_dag_facts
    assert dqk is not None and dqk.identity_candidates and len(dqk.dots) == 5
    assert dag.mma_dag_config_reason(dqk, {0: 1, 1: 32, 2: 128}, 8, 2) is None


def test_knobs_without_a_subject_are_pinned() -> None:
    # fwd_h has no fragment operand (exchange is inert) and fwd_o no loop
    # snapshot (the store path is inert).
    bound = _bind_fwd()
    config = _config(64, 64, warps=4, stages=2, cute_mma_dag_exchange="smem").config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_exchange"] == "auto"
    assert config["cute_mma_dag_store"] == "staged"
    bound = _bind_fwd_o()
    config = _config(1, 64, 64, warps=4, stages=2, cute_mma_dag_store="direct").config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_store"] == "staged"
    assert config["cute_mma_dag_exchange"] == "auto"


# --------------------------------------------------------------------------
# Chained loop dots, shared transposed reads, inert exchange on dv, [C, C]
# carry inits
# --------------------------------------------------------------------------


def _loop_chain(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            x = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=x * 0.5)
        h_all[idx, tile_d, tile_dv] = h_acc.to(h_all.dtype)
    return h_all


def _loop_chain_store(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    BH, N, D, DV = k.size(0), k.size(1), k.size(3), v.size(3)
    h_all = torch.empty([BH, D, DV], dtype=k.dtype, device=k.device)
    x_all = torch.empty([BH, N, D, DV], dtype=k.dtype, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            x = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            x_all[idx, i_t, tile_d, tile_dv] = x.to(x_all.dtype)
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=x * 0.5)
        h_all[idx, tile_d, tile_dv] = h_acc.to(h_all.dtype)
    return h_all, x_all


def _cc_init(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    a_in: torch.Tensor,
) -> torch.Tensor:
    BHN = q.size(0)
    C = hl.specialize(q.size(1))
    D = q.size(2)
    DV = v.size(2)
    out = torch.empty([BHN, C, DV], dtype=q.dtype, device=q.device)
    for tile_bhn, tile_dv in hl.tile([BHN, DV]):
        o_cross = hl.zeros([tile_bhn, C, tile_dv], dtype=torch.float32)
        attn = a_in[tile_bhn, :, :].to(torch.float32)
        for tile_d in hl.tile(D):
            qt = q[tile_bhn, :, tile_d]
            kt = k[tile_bhn, :, tile_d]
            ht = h[tile_bhn, tile_d, tile_dv]
            o_cross = hl.dot(qt, ht, acc=o_cross)
            attn = hl.dot(qt, kt.transpose(-2, -1), acc=attn)
        vt = v[tile_bhn, :, tile_dv]
        o = hl.dot(attn.to(vt.dtype), vt, acc=o_cross)
        out[tile_bhn, :, tile_dv] = o.to(out.dtype)
    return out


def test_a_chained_loop_dot_accumulates_the_result_of_the_dot_before_it() -> None:
    # ``x = dot(acc=h); h = dot(acc=x * 0.5)``: both in place in slot 0, the
    # rescale materialized after x's MMAs (0.5 (h + kv) + kv).
    t = _tensors()
    bound = _bind_fn(_loop_chain, (t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is not None
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and len(facts.frags) == 1
    loop = _loop_and_after(
        _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    )
    mmas = [m.start() for m in re.finditer(rf"{RT}\.mma_bf16\(", loop)]
    assert len(mmas) == 2 * 4 * 8
    rescale = loop.index("md_c0_0_0_0 = ")
    assert mmas[31] < rescale < mmas[32]
    assert "0.5" in loop[mmas[31] : rescale]
    # With a store of x in between, x's store sits at its program point
    # (after x's MMAs, before the rescale) and the producer stays in place.
    bound = _bind_fn(_loop_chain_store, (t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is not None
    loop = _loop_and_after(
        _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    )
    mmas = [m.start() for m in re.finditer(rf"{RT}\.mma_bf16\(", loop)]
    store = loop.index(f"{RT}.stg_b32(")
    rescale = loop.index("md_c0_0_0_0 = ")
    assert mmas[31] < store < rescale < mmas[32]
    assert "md_c1_" not in loop


@pytest.mark.parametrize("gated", (False, True))
def test_dqk_shares_the_da_tile_under_every_exchange_config(gated: bool) -> None:
    # dA read straight (dq) and transposed (dk) is one tile whether the
    # straight read repacks from registers (4 warps, auto) or goes through
    # the tile too (smem, or 8 warps with two warp columns).
    bound = _bind_dqk(gated)
    for warps, exchange in ((4, "auto"), (4, "smem"), (8, "auto"), (8, "smem")):
        body = _kernel_body(
            _source(
                bound,
                _config(
                    1, 64, 64, warps=warps, stages=2, cute_mma_dag_exchange=exchange
                ),
            )
        )
        found = re.findall(r"^\s*(md_x\d+) = md_s0", body, re.MULTILINE)
        assert len(found) == 1, (warps, exchange, found)


def test_the_exchange_knob_is_inert_on_dv() -> None:
    # dv's only fragment operand is a transposed A, which only the shared
    # tile can feed: the knob is pinned and both settings render the same
    # body.
    bound = _bind_dv()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and not facts.identity_candidates
    for warps in (4, 8):
        sources = []
        for exchange in ("auto", "smem"):
            config = _config(
                1, 64, 64, warps=warps, stages=2, cute_mma_dag_exchange=exchange
            ).config
            with bound.env:
                bound.config_spec.normalize(config)
            assert config["cute_mma_dag_exchange"] == "auto"
            sources.append(_source(bound, helion.Config(**config)))
        assert sources[0] == sources[1]
    # The register estimate counts a ldmatrix step, not a repack, for it.
    resolved = [
        facts.block_size(i, {0: 1, 1: 64, 2: 64}) for i in range(len(facts.axes))
    ]
    sizes = [size for size in resolved if size is not None]
    assert dag.mma_dag_transient_regs(facts, sizes, 4) == 16 // 4 + 64 // 4


def test_a_cc_carry_init_over_one_rdim_uses_positional_strides() -> None:
    # ``attn = a_in[tile_bhn, :, :]`` seeds the [C, C] accumulator with row
    # stride C and column stride one.
    t = _tensors3()
    a_in = torch.empty((16, 64, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_cc_init, (t["q"], t["q"], t["v"], t["h"], a_in))
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(1, 64, 64, warps=4, stages=2)))
    prologue = body[: body.index("for md_i in cutlass.range(")]
    loads = re.findall(r"\* (\d+) \+ cutlass\.Int64\(md_L\d_coll\) \* (\d+)", prologue)
    assert loads and set(loads) == {("128", "2")}
    assert prologue.count(f"{RT}.ldg_b32(") == 16


# --------------------------------------------------------------------------
# Generated code: inner tile loops and nested loops
# --------------------------------------------------------------------------


def _bind_dk(**shape: int) -> BoundKernel:
    t = _tensors3(**shape)
    bhn, c, d = t["q"].shape
    f32 = torch.float32
    return _bind(
        "chunk_bwd_dk_delta_helion",
        (
            t["q"],
            torch.empty((bhn, c, c), dtype=f32, device=CPU_DEVICE),
            torch.empty((bhn, c, d), dtype=f32, device=CPU_DEVICE),
            torch.empty((bhn, c), dtype=f32, device=CPU_DEVICE),
            torch.empty((bhn, c, d), dtype=f32, device=CPU_DEVICE),
        ),
    )


def _bind_dqkw(gated: bool = False, **shape: int) -> BoundKernel:
    t = _tensors3(**shape)
    gcs, gl = (t["gcs"], t["gl"]) if gated else (None, None)
    return _bind(
        "chunk_bwd_dqkw_delta_helion",
        (t["q"], t["q"], t["h"], t["v"], t["v"], t["v"], t["h"], gcs, gl, gated, 0.125),
    )


def test_dk_is_admitted_with_hoisted_operands_and_a_loaded_accumulator() -> None:
    # A loop with no carry: the two [C, C] operands are loaded (fp32 -> bf16)
    # once per CTA, the accumulator is seeded from two fp32 tiles at the
    # fragment coordinates and both dots accumulate into that slot.
    bound = _bind_dk()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
    assert [(b, hi) for b, _lo, hi in _block_bounds(spec)] == [(2, 128)]
    program = _program(bound)
    assert len(program.loops) == 1 and program.loops[0].innermost
    assert not program.loops[0].carries and not program.simple
    assert [axis.kind for axis in program.axes] == ["loop", "lead", "rdim"]
    assert len(program.slots) == 1
    (derived,) = program.derived
    assert not derived.frags and len(derived.loads) == 2
    assert all(dot.slot == 0 and dot.acc_slot == 0 for dot in program.dots)
    assert all(not program.is_ring(dot, dot.a) for dot in program.dots)
    assert all(program.is_ring(dot, dot.b) for dot in program.dots)
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.regions
    # dL is loaded twice (straight, and transposed for the second dot): the
    # same whole tile of the root region, so one staged tile serves both.
    assert [tile.frame for tile in facts.tiles] == [-1, 0, 0, 0]
    assert [tile.ring for tile in facts.tiles] == [False, True, True, True]
    assert program.dots[0].a.chain_key[0] is program.dots[1].a.chain_key[0]
    assert (program.dots[0].a.trans, program.dots[1].a.trans) == (False, True)
    body = _kernel_body(_source(bound, _config(32, warps=4, stages=2)))
    prologue = body[: body.index("for md_i in cutlass.range(0, 4):")]
    loop = _loop_and_after(body)
    # The staged operands: cp.async, one wait, two convert passes (fp32 ->
    # bf16 packets), a barrier, all before the ring prologue.
    assert prologue.count("cute.arch.cp_async_wait_group(0)") == 1
    assert prologue.count(f"{RT}.lds_v4_f32(") == 8
    assert prologue.count(f"{RT}.sts_v2_b32(") == 8
    assert prologue.count("cute.arch.barrier()") == 1
    assert prologue.index(f"{RT}.sts_v2_b32(") < prologue.index("cute.arch.barrier()")
    assert prologue.count("cute.arch.cp_async_commit_group()") == 2
    # The ring: k (4 KB) and the two fp32 seed tiles (8 KB each) at two
    # stages (40 KB), then the one 64x64 bf16 dL tile (8 KB); its 64x64
    # fp32 source (16 KB) is converted before the loop starts and overlays
    # the ring.  ldmatrix reads it straight for the first dot and
    # transposed for the second.
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=128)" in body
    assert "md_t0r = md_s0 + 0\n" in body and "md_t2" not in body
    assert (
        loop.count(f"{RT}.ldmatrix_x4(md_t0 ")
        == loop.count(f"{RT}.ldmatrix_x4_trans(md_t0 ")
        > 0
    )
    # The seed: fp32 pairs of both tiles read from the ring per register.
    assert loop.count(f"{RT}.lds_v2_f32(") == 16
    assert "md_c1_" not in body
    # dL straight (ldmatrix), dL^T through the same layout with .trans; the
    # k tile is the B operand of both dots (one ring tile).
    assert loop.count(f"{RT}.ldmatrix_x4(md_t0") == 4
    assert loop.count(f"{RT}.ldmatrix_x4_trans(md_t0") == 4
    assert loop.count(f"{RT}.ldmatrix_x4_trans(md_t1") == 16
    assert loop.count(f"{RT}.mma_bf16(") == 32
    assert loop.count(f"{RT}.stg_b32(") == 8
    # No fragment operand: the exchange knob has no subject.
    assert not facts.identity_candidates


def test_dqkw_is_admitted_as_a_three_level_nest() -> None:
    # dattn's loop, then the tile_d loop with the DV loop nested in it.
    bound = _bind_dqkw()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert [(b, hi) for b, _lo, hi in _block_bounds(spec)] == [
        (1, 128),
        (3, 128),
        (4, 128),
    ]
    program = _program(bound)
    assert [(loop.parent, loop.innermost) for loop in program.loops] == [
        (-1, True),
        (-1, False),
        (1, True),
    ]
    assert [len(loop.carries) for loop in program.loops] == [1, 0, 3]
    # dg_p / dg_last_dk pass through the tile_d loop untouched.
    assert [s.identity for s in program.loops[1].small_carries] == [True, True]
    # Slots: dattn [C, C], the three nested carries and the subtracted dot.
    assert len(program.slots) == 5
    # ``dW = dW - dot(...)``: the carry and the no-acc dot feed the derived
    # value, which takes over the carry's slot.
    (derived,) = program.derived
    assert len(derived.frags) == 2 and derived.slot == program.loops[2].carries[0].slot
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.regions
    assert facts.loops == (
        dag.LoopFacts(-1, True),
        dag.LoopFacts(-1, False),
        dag.LoopFacts(1, True),
    )
    # bd = 32: dattn (4096) + three [C, bd] carries + the dot (4 x 2048) are
    # 96 registers over four warps; two warps overflow; bd = 128 overflows
    # at eight.
    sizes = {1: 32, 3: 32, 4: 32}
    assert dag.mma_dag_config_reason(facts, sizes, 4, 2) is None
    reason = dag.mma_dag_config_reason(facts, sizes, 2, 2)
    assert reason is not None and "192 registers" in reason
    reason = dag.mma_dag_config_reason(facts, {1: 32, 3: 128, 4: 32}, 8, 2)
    assert reason is not None and "144 registers" in reason
    body = _kernel_body(_source(bound, _config(32, 32, 32, warps=4, stages=2)))
    first = body.index("for md_i in cutlass.range(0, 4):")
    outer = body.index("for md_i1 in cutlass.range(0, 4):")
    inner = body.index("for md_i2 in cutlass.range(0, 4):")
    assert first < outer < inner
    # Sibling frames overlap: both rings start at the frame base, the outer
    # loop's q / k tiles follow the nested ring, the exchange tile is root.
    assert "md_t0 = md_s0 + 0" in body and "md_t2 = md_s0 + 0" in body
    assert "md_t7 = md_s0 + 32768" in body and "md_x8 = md_s0 + 40960" in body
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=128)" in body
    between = body[first:outer]
    # The dattn loop releases its frame, then the masked bf16 dattn is
    # written once into the exchange tile behind its own barrier.
    assert between.count("cute.arch.barrier()") == 3
    assert between.count(f"{RT}.sts_b32(md_x8") == 16
    assert between.rindex(f"{RT}.sts_b32(md_x8") < between.rindex("cute.arch.barrier()")
    outer_body = body[outer:inner]
    # q and k land once per outer iteration (their own group); the nested
    # ring's prologue is not here: the first iteration's is peeled ahead of
    # the outer ``for`` and each iteration's epilogue issues the next one's.
    assert outer_body.count(f"{RT}.cp_async_16(md_t7") == 2
    assert outer_body.count(f"{RT}.cp_async_16(md_t9") == 2
    assert outer_body.count("cute.arch.cp_async_commit_group()") == 1
    assert outer_body.count(f"{RT}.cp_async_16(md_t2") == 0
    peeled = between[between.rindex("cute.arch.barrier()") :]
    assert peeled.count(f"{RT}.cp_async_16(md_t2") == 2
    assert peeled.count("cute.arch.cp_async_commit_group()") == 1
    after = body[inner:]
    # After the nested loop: wait, barrier, the next iteration's nested
    # prologue behind the barrier (bases at md_i1 + 1, one group), the dW
    # store, dq from the register repack of dattn, dk_as from the exchange
    # tile transposed, and the trailing barrier that guards the single-slot
    # q / k tiles.
    tail = after[after.rindex("cute.arch.cp_async_wait_group(0)") :]
    assert tail.index("cute.arch.barrier()") < tail.index(f"{RT}.stg_v2_f32(")
    early = tail.index("if md_i1 + 1 < 4:")
    assert tail.index("cute.arch.barrier()") < early < tail.index(f"{RT}.stg_v2_f32(")
    assert "cutlass.Int64(md_i1 + 1) * " in tail[early:]
    assert tail[early:].count(f"{RT}.cp_async_16(md_t2") == 2
    assert (
        tail[early : tail.index(f"{RT}.stg_v2_f32(")].count(
            "cute.arch.cp_async_commit_group()"
        )
        == 1
    )
    assert tail.count(f"{RT}.stg_v2_f32(") == 16
    assert tail.count(f"{RT}.stg_b32(") == 8
    assert "md_aw" in tail and tail.count(f"{RT}.ldmatrix_x4_trans(md_x8") == 4
    # (up to the first column-0 line: the body ends with host metadata)
    kernel_tail = re.split(r"\n(?=\S)", tail, maxsplit=1)[0]
    assert kernel_tail.rstrip().endswith("cute.arch.barrier()")
    assert tail.count(f"{RT}.mma_bf16(") == 32


def test_dqkw_exchange_knob_sends_the_hoisted_operand_through_the_tile() -> None:
    bound = _bind_dqkw()
    auto = _kernel_body(_source(bound, _config(32, 32, 32, warps=4, stages=2)))
    smem = _kernel_body(
        _source(
            bound, _config(32, 32, 32, warps=4, stages=2, cute_mma_dag_exchange="smem")
        )
    )
    assert "md_aw" in auto and "md_aw" not in smem
    assert len(re.findall(rf"{RT}\.ldmatrix_x4\(md_x\d+", smem)) == 4
    assert len(re.findall(rf"{RT}\.ldmatrix_x4_trans\(md_x\d+", smem)) == 4
    assert len(re.findall(rf"{RT}\.sts_b32\(md_x\d+", smem)) == 16


def _bind_wy_dl(
    gated: bool = False,
    beta_dtype: torch.dtype = torch.float32,
    dvni_dtype: torch.dtype = torch.float32,
    **shape: int,
) -> BoundKernel:
    t = _tensors3(**shape)
    bhn, c, d = t["q"].shape
    dv = t["v"].shape[2]
    f32 = torch.float32
    return _bind(
        "chunk_bwd_wy_dL_delta_helion",
        (
            t["q"],
            t["v"],
            torch.empty((bhn, c), dtype=beta_dtype, device=CPU_DEVICE),
            torch.empty((bhn, c, c), dtype=f32, device=CPU_DEVICE),
            torch.empty((bhn, c, d), dtype=f32, device=CPU_DEVICE),
            torch.empty((bhn, c, dv), dtype=dvni_dtype, device=CPU_DEVICE),
            t["gcs"] if gated else None,
            gated,
        ),
    )


def test_wy_dl_is_admitted_with_vector_carries_and_row_reductions() -> None:
    # Three sequential loops: dAinv and dbeta carried from the first into
    # the second (their own slots), chained root dots, the kk loop, then the
    # root reduction into dbeta and its vector store.
    bound = _bind_wy_dl()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    program = _program(bound)
    assert [(loop.parent, loop.innermost) for loop in program.loops] == [(-1, True)] * 3
    assert [type(step).__name__ for step in program.body] == [
        "Loop",
        "Loop",
        "Dot",
        "Dot",
        "Loop",
        "Store",
        "Derived",
        "Reduce",
        "VecOp",
        "VecStore",
    ]
    first, second, third = program.loops
    assert second.carries[0].init_frag is first.carries[0].phi
    assert second.carries[0].slot == first.carries[0].slot
    assert [c.identity for c in first.vec_carries] == [False]
    assert [c.identity for c in second.vec_carries] == [False, True]
    assert second.vec_carries[0].init_vec is first.vec_carries[0].phi
    assert second.vec_carries[0].slot == first.vec_carries[0].slot == 0
    assert not third.vec_carries
    # One vector family: the carry, the two loop reductions and the root one,
    # following the three reduced fragments' row split.
    assert len(program.vec_slots) == 4
    assert all(slot.orient == "rows" for slot in program.vec_slots)
    assert len({program.vec_frag_slots[s] for s in range(4)}) == 1
    assert [op.slot for op in program.vec_ops] == [0, 0, 0]
    # ``d_bv * v`` / ``d_ba * k`` take over the dot result's slot; the root
    # ``dA * kk`` takes over dA's.
    assert [d.slot for d in program.derived] == [1, 2, 4]
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.regions
    assert len(facts.vecs) == 4 and facts.vecs[0].frags == (1, 2, 4)
    # Register budget: the vector words count (two per lane at four warps).
    resolved = [32, 32, 128, 1, 64]
    layouts, _routes, _bytes = dag.mma_dag_layout_solution(facts, resolved, 4)
    assert dag.mma_dag_vec_regs(facts, resolved, layouts, 4) == 8
    # The vector scratch holds one vector (a relayout) at one warp column,
    # one per warp column otherwise.
    assert dag.mma_dag_vec_scratch(facts, resolved, layouts, 4) == 64 * 4
    layouts8, _routes, _bytes = dag.mma_dag_layout_solution(facts, resolved, 8)
    assert dag.mma_dag_vec_scratch(facts, resolved, layouts8, 8) == 2 * 64 * 4
    body = _kernel_body(_source(bound, _config(32, 32, 128, warps=4, stages=2)))
    # The vector carry starts at zero; each of the three reductions sums the
    # lane's registers then the quad (xor 1, 2); the final value is stored by
    # the lanes with c == 0 (one warp column: nothing goes through scratch).
    assert "md_r0_0_0 = cutlass.Float32(0.0)" in body
    assert body.count(f"{RT}.lane_group_sum(") == 6
    assert f"{RT}.sts_f32(" not in body
    assert body.count("if md_c == 0:") == 1
    assert body.count(f"{RT}.stg_f32(") == 2
    # The three rings share one frame base, as does the fp32 source of T^T
    # (one staged tile, fp32 -> bf16 once per CTA before the loops, read by
    # four dots straight and transposed).
    assert body.count("= md_s0 + 0\n") == 4
    assert "md_t2r = md_s0 + 0\n" in body
    prologue = body[: body.index("for md_i in cutlass.range(")]
    assert prologue.count(f"{RT}.sts_v2_b32(") == 8
    eight = _kernel_body(_source(bound, _config(32, 32, 64, warps=8, stages=2)))
    # Two warp columns: the partial rows meet in the scratch tile behind a
    # barrier (and another before its reuse), per reduction.
    assert "md_rv = md_s0 +" in eight
    assert eight.count(f"{RT}.sts_f32(md_rv") == 6
    assert eight.count(f"{RT}.lds_f32(md_rv") == 12
    assert "if md_c == 0 and md_L" in eight


def test_gated_dqkw_adds_scalar_reductions_and_a_two_dim_loop() -> None:
    # Gated: dg_p accumulates two row reductions per tile_d iteration,
    # dg_last_dk a row reduction summed into a scalar, and a 2-D tile loop
    # over h * dS (loads only) reduces into a scalar carry.
    bound = _bind_dqkw(gated=True)
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    program = _program(bound)
    assert [
        (loop.parent, loop.innermost, len(loop.axes)) for loop in program.loops
    ] == [
        (-1, True, 1),
        (-1, False, 1),
        (1, True, 1),
        (-1, True, 2),
    ]
    tile_d, hd_loop = program.loops[1], program.loops[3]
    assert [c.identity for c in tile_d.vec_carries] == [False, False]
    assert [program.vec_slots[c.slot].orient for c in tile_d.vec_carries] == [
        "rows",
        "scalar",
    ]
    assert [program.vec_slots[c.slot].orient for c in hd_loop.vec_carries] == ["scalar"]
    # The 2-D loop's product is a fresh [bd, bdv] fragment over two tiles.
    (derived,) = [d for d in program.derived if d.loop == hd_loop.index]
    assert not derived.frags and len(derived.loads) == 2
    assert [type(step).__name__ for step in program.body] == [
        "Loop",
        "Loop",
        "VecStore",
        "Loop",
        "VecOp",
        "VecOp",
        "VecStore",
    ]
    reduces = [r.orient for r in program.reduces]
    assert reduces == ["rows", "rows", "rows", "scalar", "rows", "scalar"]
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.regions
    assert [v.orient for v in facts.vecs].count("scalar") == 4
    body = _kernel_body(_source(bound, _config(32, 32, 32, 32, 32, warps=4, stages=2)))
    # The 2-D loop walks 4 x 4 tiles, the last axis fastest.
    assert "for md_i3 in cutlass.range(0, 16):" in body
    assert "md_a3 = md_i3 // 4 * 32" in body and "md_a4 = md_i3 % 4 * 32" in body
    # Vector -> scalar: the octet shuffles (xor 4, 8, 16) sum the rows a warp
    # holds, the warp rows meet in the scratch; thread 0 stores the scalar.
    assert body.count(f"{RT}.octet_sum(") == 2
    assert body.count("if md_tidx == 0:") == 1
    assert body.count("if md_c == 0:") >= 1
    assert f"{RT}.sts_f32(md_rv + md_L" in body


def test_gated_wy_dl_relays_the_column_reduction_into_the_row_vector() -> None:
    # ``dLL.sum(-1) - dLL.sum(-2)``: the column sums (octet shuffles over g)
    # are written to the scratch by the lanes owning each column and read
    # back in the row vector's layout before the subtraction.
    bound = _bind_wy_dl(gated=True)
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    program = _program(bound)
    assert [r.orient for r in program.reduces] == [
        "rows",
        "rows",
        "rows",
        "rows",
        "rows",
        "cols",
    ]
    cols = [s for s in program.vec_slots if s.orient == "cols"]
    assert len(cols) == 1
    body = _kernel_body(_source(bound, _config(32, 32, 128, warps=4, stages=2)))
    assert body.count(f"{RT}.octet_sum(") == 16
    assert "if md_g == 0:" in body
    assert body.count("md_rt") > 0
    assert body.count(f"{RT}.stg_f32(") == 4


def _bind_wy(gated: bool = False, **shape: int) -> BoundKernel:
    t = _tensors3(**shape)
    bhn, c, _d = t["q"].shape
    beta = torch.empty((bhn, c), dtype=torch.float32, device=CPU_DEVICE)
    gcs = t["gcs"] if gated else None
    return _bind(
        "chunk_fwd_wy_delta_helion",
        (t["q"], t["v"], beta, gcs, None, None, gated, False),
    )


def _bind_wy_anchored(**shape: int) -> BoundKernel:
    """The kda (``diag_anchored``) WY transform: the gram ``Akk`` arrives
    from memory, the per-channel decay ``g_cs`` is a ``[C, D]`` tile and the
    state key ``k_state_out`` is written beside ``w``."""
    t = _tensors3(**shape)
    bhn, c, d = t["q"].shape
    f32 = torch.float32
    beta = torch.empty((bhn, c), dtype=f32, device=CPU_DEVICE)
    gc = torch.empty((bhn, c, d), dtype=f32, device=CPU_DEVICE)
    akk = torch.empty((bhn, c, c), dtype=torch.bfloat16, device=CPU_DEVICE)
    return _bind(
        "chunk_fwd_wy_delta_helion",
        (t["q"], t["v"], beta, gc, akk, torch.empty_like(t["q"]), False, True),
    )


def test_wy_runs_the_neumann_chain_on_tf32_mmas() -> None:
    # kk (bf16) feeds the pointwise carry inits A and T = eye + A of the
    # static five-step loop; its dots are fp32 x fp32 (tf32) over fp32
    # exchange tiles; the w / u loops widen the bf16 k / v tiles into fp32
    # operand tiles for dot(T, kt).
    bound = _bind_wy()
    program = _program(bound)
    assert [
        (loop.parent, loop.innermost, len(loop.axes)) for loop in program.loops
    ] == [
        (-1, True, 1),
        (-1, True, 0),
        (-1, True, 1),
        (-1, True, 1),
    ]
    chain = program.loops[1]
    assert (chain.begin, chain.end) == (0, 5)
    kk = program.loops[0].carries[0]
    assert [c.init_frag is kk.phi for c in chain.carries] == [True, True]
    assert all(c.init_value is not None and c.slot != kk.slot for c in chain.carries)
    apow, t_dot = (dot for dot in program.dots if dot.loop == 1)
    assert apow.acc is None and apow.slot not in {c.slot for c in chain.carries}
    assert t_dot.acc_slot == t_dot.slot == chain.carries[1].slot
    assert [dot.a.dtype for dot in program.dots] == [torch.bfloat16] + [
        torch.float32
    ] * 4
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    assert facts.tcgen05.regions and facts.tcgen05.tf32
    # fp32 fragments bind whole-K words only when the producer is the dot's
    # own accumulator (T feeding dot(Apow, T, acc=T)); the exchange knob
    # has a subject (the chain's shuffle routes).
    assert [(d.a_identity, d.b_frag) for d in facts.dots] == [
        (False, False),
        (False, False),
        (False, True),
        (False, False),
        (False, False),
    ]
    assert facts.identity_candidates
    body = _kernel_body(_source(bound, _config(128, 32, 32, warps=4, stages=2)))
    assert body.count(f"{RT}.mma_bf16(") == 64
    # Apow and T: 8 K steps of 8 x 8 atoms each; w and u: 8 x 4 each.
    assert body.count(f"{RT}.mma_tf32(") == 128 + 32 + 32
    # Four warps split M: the chain's A operands (Apow, then its square) are
    # the warp's own registers, picked per K step with lane shuffles (2 dots
    # x 8 steps x 4 words); its B operands are scattered into transposed
    # ([N, K]) fp32 tiles and T feeds the w / u dots through an fp32 pair
    # tile; both are read by ldmatrix (four fp32 per 16-byte chunk), never
    # per lane or with .trans.
    assert body.count(f"{RT}.shfl_pick_b32(") == 64
    assert body.count(f"{RT}.f32_bits(") == 128
    assert body.count("md_qsl + 2") == 32 and "md_tsl" not in body
    assert len(re.findall(rf"{RT}\.sts_v2_f32\(md_x", body)) == 16
    assert len(re.findall(rf"{RT}\.sts_f32\(md_x", body)) == 64
    assert len(re.findall(rf"{RT}\.ldmatrix_x4\(md_x", body)) == 64 + 16
    assert f"{RT}.lds_b32(md_x" not in body
    assert f"{RT}.ldmatrix_x4_trans(md_x" not in body
    # The bf16 k / v tiles widen into transposed fp32 operand tiles.
    assert len(re.findall(rf"{RT}\.sts_f32\(md_t", body)) == 32
    assert f"{RT}.sts_v4_f32(" not in body
    # dot(Apow, Apow) lands in a fresh slot and moves into Apow's carry.
    copies = re.findall(r"md_c1_\d+_\d+_\d+ = md_c3_\d+_\d+_\d+", body)
    assert len(copies) == 32
    # A_inv is stored as fp32 pairs (16 per lane) between the chain and the
    # w loop.
    assert body.count(f"{RT}.stg_v2_f32(") == 16


def test_cta_rich_kernels_seed_four_warps_and_pipelined_loop_tiles() -> None:
    # The per-chunk delta kernels tile their fragments by rdims and loop
    # tiles alone over 8192 leading CTAs: with the device facts of a B200
    # (148 SMs, 227 KB) they seed four warps first and loop tiles of at
    # least two trips, within two CTAs per SM of shared memory (measured
    # at s2: dk [32] 203 us; dqkw [64, 32, 64] 575, the eight-warp /
    # single-trip tiles 616 - 793; wy_dL [64, 64, 64] 429; wy 787).
    def seeds(bound: BoundKernel) -> list[tuple[list[int], int, int]]:
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None
        facts = dataclasses.replace(facts, sm_count=148, smem_capacity=232448)
        tunable = [a.block_id for a in facts.axes if a.kind in ("root", "loop")]
        return [
            ([sizes[b] for b in tunable], warps, stages)
            for sizes, warps, stages in dag.mma_dag_seed_configs(facts)
        ]

    big = {"bhn": 8192, "d": 128, "dv": 128}
    # (dk here narrows an fp32 dL: its two raw source tiles overlay the ring
    # frame, so the 64 tile's 96 KB fit two CTAs per SM.)
    assert seeds(_bind_dk(**big))[0] == ([64], 4, 2)
    assert seeds(_bind_dqkw(**big))[0] == ([64, 32, 64], 4, 2)
    assert seeds(_bind_wy_dl(**big))[0] == ([64, 64, 64], 4, 2)
    assert seeds(_bind_wy(**big))[0] == ([64, 64, 64], 4, 4)
    # Tiles over the register budget at four warps rank behind every
    # four-warp seed; single-trip loops rank behind pipelined ones.
    ranked = seeds(_bind_dqkw(**big))
    first_eight = next(i for i, (_s, warps, _st) in enumerate(ranked) if warps == 8)
    assert all(warps == 4 for _s, warps, _st in ranked[:first_eight])
    assert all(all(128 // tile >= 2 for tile in sizes) for sizes, _w, _st in ranked[:4])
    # The single-loop kernels tile a root axis: their seeds are unchanged by
    # the rule (fwd_h_delta keeps the exchange-free four warps first,
    # dstate its eight).
    delta = _bind_fwd_h_delta(bh=32, dv=128).config_spec.cute_mma_dag_facts
    assert delta is not None
    delta = dataclasses.replace(delta, sm_count=148, smem_capacity=232448)
    assert [(s[1], s[2]) for s in dag.mma_dag_seed_configs(delta)[:2]] == [
        (4, 4),
        (4, 2),
    ]


def test_m4_kernels_pin_their_inert_knobs_and_repair_over_budget_configs() -> None:
    # dk has no fragment operand and no loop snapshot: both the exchange and
    # the store knob are pinned.  dqkw reads the hoisted dattn as an A
    # operand (the exchange knob has a subject) but snapshots nothing; a
    # tile over the register budget at four warps repairs to eight.
    bound = _bind_dk()
    config = _config(
        32, warps=4, stages=2, cute_mma_dag_exchange="smem", cute_mma_dag_store="direct"
    ).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_exchange"] == "auto"
    assert config["cute_mma_dag_store"] == "staged"
    bound = _bind_dqkw()
    config = _config(
        32,
        32,
        32,
        warps=4,
        stages=2,
        cute_mma_dag_exchange="smem",
        cute_mma_dag_store="direct",
    ).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_exchange"] == "smem"
    assert config["cute_mma_dag_store"] == "staged"
    over = _config(32, 64, 32, warps=4, stages=2)
    assert (
        bound.config_spec.backend.autotune_config_is_viable(bound.config_spec, over)
        is False
    )
    with bound.env:
        with pytest.raises(InvalidConfig, match="registers per thread"):
            bound.config_spec.normalize(dict(over.config))
        repaired = dict(over.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
    assert repaired["cute_mma_dag_warps"] == 8
    assert repaired["block_sizes"] == [32, 64, 32]
    # wy: the knob surface exists only under tf32; the chain's fp32
    # fragments take the shuffle route per iteration, so the exchange knob
    # has a register route to switch off.
    bound = _bind_wy()
    config = _config(
        128, 32, 32, warps=4, stages=2, cute_mma_dag_exchange="smem"
    ).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_exchange"] == "smem"
    body = _kernel_body(_source(bound, helion.Config(**config)))
    assert f"{RT}.shfl_pick_b32(" not in body
    assert len(re.findall(rf"{RT}\.sts_v2_f32\(md_x", body)) == 48


# --------------------------------------------------------------------------
# Probes: register takeovers across regions and through views, the barrier
# behind a nested ring, fp32 operand tiles the passes cannot rewrite, admitted
# shapes that could not render
# --------------------------------------------------------------------------


def _acc_from_outer(
    a: torch.Tensor, b: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """A loop dot accumulates onto a fragment of the enclosing region."""
    BH, N = k.size(0), k.size(1)
    C, D = hl.specialize(k.size(2)), hl.specialize(k.size(3))
    out = torch.empty([BH, N, C, D], dtype=torch.float32, device=k.device)
    for tile_b in hl.tile(BH, block_size=1):
        base = a[tile_b, :, :] + b[tile_b, :, :]
        for i in hl.grid(N):
            out[tile_b, i, :, :] = hl.dot(
                k[tile_b, i, :, :], v[tile_b, i, :, :], acc=base
            )
    return out


def _derived_from_outer(
    a: torch.Tensor, b: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """A loop's derived value reads an earlier loop's accumulator."""
    BH, N = k.size(0), k.size(1)
    C, D = hl.specialize(k.size(2)), hl.specialize(k.size(3))
    out = torch.empty([BH, N, C, D], dtype=torch.float32, device=k.device)
    for tile_b in hl.tile(BH, block_size=1):
        base = hl.zeros([tile_b, C, D], dtype=torch.float32)
        for j in hl.grid(N):
            base = hl.dot(k[tile_b, j, :, :], v[tile_b, j, :, :], acc=base)
        for i in hl.grid(N):
            y = hl.dot(k[tile_b, i, :, :], v[tile_b, i, :, :])
            out[tile_b, i, :, :] = base * 2.0 + y
    return out


def _vec_from_outer(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A loop's vector value reads an earlier loop's vector carry."""
    BH, N = k.size(0), k.size(1)
    C, D = hl.specialize(k.size(2)), hl.specialize(k.size(3))
    out = torch.empty([BH, N, C], dtype=torch.float32, device=k.device)
    for tile_b in hl.tile(BH, block_size=1):
        acc = hl.zeros([tile_b, C, D], dtype=torch.float32)
        vb = hl.zeros([tile_b, C], dtype=torch.float32)
        for j in hl.grid(N):
            acc = hl.dot(k[tile_b, j, :, :], v[tile_b, j, :, :], acc=acc)
            vb = vb + acc.sum(-1)
        for i in hl.grid(N):
            y = hl.dot(k[tile_b, i, :, :], v[tile_b, i, :, :])
            out[tile_b, i, :] = vb + y.sum(-1)
    return out


def _view_read_after_takeover(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``y = -kk`` is read after ``d1 = kk * qq`` would take kk's registers."""
    BHN, C, D = q.size(0), hl.specialize(q.size(1)), q.size(2)
    out1 = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    out2 = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        qq = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kd = k[tile_bhn, :, tile_d]
            qd = q[tile_bhn, :, tile_d]
            kk = hl.dot(kd, kd.transpose(-2, -1), acc=kk)
            qq = hl.dot(qd, qd.transpose(-2, -1), acc=qq)
        y = -kk
        d1 = kk * qq
        out1[tile_bhn, :, :] = d1
        out2[tile_bhn, :, :] = y * qq
    return out1, out2


def _reduced_view_after_takeover(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row reduction of ``kk * 2`` after ``d1 = kk * qq``."""
    BHN, C, D = q.size(0), hl.specialize(q.size(1)), q.size(2)
    out1 = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    sout = torch.empty([BHN, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        qq = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kd = k[tile_bhn, :, tile_d]
            qd = q[tile_bhn, :, tile_d]
            kk = hl.dot(kd, kd.transpose(-2, -1), acc=kk)
            qq = hl.dot(qd, qd.transpose(-2, -1), acc=qq)
        y = kk * 2.0
        out1[tile_bhn, :, :] = kk * qq
        sout[tile_bhn, :] = y.sum(-1)
    return out1, sout


def test_a_loop_never_takes_over_the_registers_of_an_enclosing_region() -> None:
    # The next iteration reads them again: a dot accumulating onto the
    # enclosing region's fragment, a derived value over an earlier loop's
    # accumulator and a vector value over an earlier loop's vector each get
    # a fresh slot (the dot copies its seed in every iteration).
    t = _tensors()
    a = torch.empty((2, 64, 64), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind_fn(_acc_from_outer, (a, a, t["k"], t["v"]))
    program = _program(bound)
    (base,) = program.derived
    (dot,) = program.dots
    assert base.loop == -1 and dot.loop == 0
    assert dot.acc_slot == base.slot and dot.slot != base.slot
    body = _kernel_body(_source(bound, _config(warps=4, stages=2)))
    loop = _loop_and_after(body)
    assert (
        len(re.findall(rf"md_c{dot.slot}_\d+_\d+_\d+ = md_c{base.slot}_", loop)) == 32
    )

    bound = _bind_fn(_derived_from_outer, (a, a, t["k"], t["v"]))
    program = _program(bound)
    (derived,) = program.derived
    (carry,) = program.loops[0].carries
    assert derived.loop == 1 and carry.phi in derived.frags
    assert derived.slot != carry.slot
    assert derived.slot == program.dots[1].slot

    bound = _bind_fn(_vec_from_outer, (t["k"], t["v"]))
    program = _program(bound)
    (vb,) = program.loops[0].vec_carries
    later = [op for op in program.vec_ops if op.loop == 1]
    (op,) = later
    assert vb.phi in op.vecs and op.slot != vb.slot


def test_a_takeover_sees_the_readers_of_the_fragments_views() -> None:
    # ``y = -kk`` is evaluated where it is consumed, after ``d1``: ``d1``
    # cannot take kk's registers (nor qq's, read by the second value);
    # the second value, the last reader of both, takes one of them.  With
    # a row reduction of the view instead, ``d1`` is qq's last reader and
    # takes qq's registers, never kk's.
    t = _tensors3()
    bound = _bind_fn(_view_read_after_takeover, (t["q"], t["q"]))
    program = _program(bound)
    kk, qq = (carry.slot for carry in program.loops[0].carries)
    d1, d2 = program.derived
    assert d1.slot not in (kk, qq)
    assert d2.slot in (kk, qq)

    bound = _bind_fn(_reduced_view_after_takeover, (t["q"], t["q"]))
    program = _program(bound)
    kk, qq = (carry.slot for carry in program.loops[0].carries)
    (d1,) = program.derived
    (reduce,) = program.reduces
    assert d1.slot == qq
    assert program.frag_slots[reduce.frag] == kk


def _nested_ring(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """A ringed loop nested in an outer tile loop that stages nothing."""
    BHN, C, DV = x.size(0), hl.specialize(x.size(1)), hl.specialize(x.size(2))
    D = s.size(1)
    out = torch.empty([BHN, C, D], dtype=torch.float32, device=x.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        for tile_d in hl.tile(D):
            acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            for tile_dv in hl.tile(DV):
                xa = x[tile_bhn, :, tile_dv]
                sb = s[tile_bhn, tile_d, tile_dv]
                acc = hl.dot(xa, sb.transpose(-2, -1), acc=acc)
            out[tile_bhn, :, tile_d] = acc
    return out


def _outer_loop_tail(body: str, outer: str, inner: str) -> str:
    """The text of loop ``outer``'s body after its nested loop ``inner``."""
    lines = body.split("\n")
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith(outer))
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1 :], start + 1)
            if line.strip() and len(line) - len(line.lstrip()) <= indent
        ),
        len(lines),
    )
    inner_start = next(
        i for i in range(start, end) if lines[i].lstrip().startswith(inner)
    )
    inner_indent = len(lines[inner_start]) - len(lines[inner_start].lstrip())
    inner_end = next(
        i
        for i in range(inner_start + 1, end)
        if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= inner_indent
    )
    return "\n".join(lines[inner_end:end])


def test_an_outer_iteration_ends_with_a_barrier_behind_a_nested_ring() -> None:
    # The next outer iteration's ring prologue rewrites slot 0 while slower
    # warps may still read it: the outer iteration ends with a CTA barrier
    # even though the outer loop stages no tile of its own.
    t = _tensors3()
    bound = _bind_fn(_nested_ring, (t["q"], t["h"]))
    program = _program(bound)
    outer, inner = program.loops
    assert not outer.innermost and inner.parent == outer.index
    plan_facts = bound.config_spec.cute_mma_dag_facts
    assert plan_facts is not None and not plan_facts.staging_frames
    body = _kernel_body(_source(bound, _config(32, 128, warps=4, stages=2)))
    tail = _outer_loop_tail(body, "for md_i in", f"for md_i{inner.index} in")
    assert "cute.arch.barrier()" in tail
    assert f"{RT}.stg_v2_f32(" in tail
    assert tail.rindex(f"{RT}.stg_v2_f32(") < tail.rindex("cute.arch.barrier()")


def _tf32_inplace_b(
    k: torch.Tensor, x: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """B = fp32 load * beta (a row-dependent pointwise op) into a tf32 dot."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, D], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        beta_i = beta[tile_bhn, :]
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kt = k[tile_bhn, :, tile_d]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        for tile_d in hl.tile(D):
            xt = x[tile_bhn, :, tile_d] * beta_i[:, :, None]
            out[tile_bhn, :, tile_d] = hl.dot(kk, xt)
    return out


def _tf32_inplace_a(
    k: torch.Tensor, x: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """A and B = fp32 load + 1 in a K loop (K along the columns)."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        acc = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            xt = x[tile_bhn, :, tile_d] + 1.0
            acc = hl.dot(xt, xt.transpose(-2, -1), acc=acc)
        out[tile_bhn, :, :] = acc
    return out


def _tf32_lanes_b(k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """B = fp32 load as is (K = C along the tile rows: per-lane B reads)."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, D], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kt = k[tile_bhn, :, tile_d]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        for tile_d in hl.tile(D):
            xt = x[tile_bhn, :, tile_d]
            out[tile_bhn, :, tile_d] = hl.dot(kk, xt)
    return out


def _bind_tf32(fn: object, args: tuple[object, ...]) -> BoundKernel:
    return helion.kernel(
        fn,  # type: ignore[arg-type]
        backend="cute",
        static_shapes=True,
        dot_precision="tf32",
    ).bind(args)


def test_an_fp32_tile_with_a_pointwise_op_is_not_a_tf32_operand() -> None:
    # The in-place pass rewrites 16-bit packets (eight elements, bf16
    # unpack / pack): an fp32 source tile with a pointwise op feeding a
    # tf32 dot declines at admission instead of being repacked as bf16.
    # The same tile read as loaded stays admitted (per-lane fp32 B reads).
    t = _tensors3()
    x = torch.empty((16, 64, 128), dtype=torch.float32, device=CPU_DEVICE)
    beta = torch.empty((16, 64), dtype=torch.float32, device=CPU_DEVICE)
    assert (
        _bind_tf32(_tf32_inplace_b, (t["q"], x, beta)).config_spec.cute_mma_dag is None
    )
    assert (
        _bind_tf32(_tf32_inplace_a, (t["q"], x, beta)).config_spec.cute_mma_dag is None
    )
    bound = _bind_tf32(_tf32_lanes_b, (t["q"], x))
    assert bound.config_spec.cute_mma_dag is not None
    program = _program(bound)
    (_kk_dot, dot) = program.dots
    assert dot.b.kind == "load" and dot.b.dtype == torch.float32 and not dot.b.pointwise
    body = _kernel_body(_source(bound, _config(32, 32, warps=4, stages=2)))
    assert f"{RT}.mma_tf32(" in body and f"{RT}.lds_b32(" in body
    assert f"{RT}.unpack_bf16x2(" not in body


def _staged_aux_vec(a: torch.Tensor, g: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """A loop-invariant operand tile scaled by a per-iteration row vector."""
    BH, C = a.size(0), hl.specialize(a.size(1))
    N, D = k.size(1), hl.specialize(k.size(3))
    out = torch.empty([BH, C, D], dtype=torch.float32, device=a.device)
    for tile_b in hl.tile(BH, block_size=1):
        T = a[tile_b, :, :].to(torch.float32)
        acc = hl.zeros([tile_b, C, D], dtype=torch.float32)
        for i in hl.grid(N):
            gi = g[tile_b, i, :]
            op = (T * gi[:, :, None]).to(torch.bfloat16)
            acc = hl.dot(op, k[tile_b, i, :, :], acc=acc)
        out[tile_b, :, :] = acc
    return out


def _staged_aux_scalar(
    a: torch.Tensor, g: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """The same with a per-iteration scalar."""
    BH, C = a.size(0), hl.specialize(a.size(1))
    N, D = k.size(1), hl.specialize(k.size(3))
    out = torch.empty([BH, C, D], dtype=torch.float32, device=a.device)
    for tile_b in hl.tile(BH, block_size=1):
        T = a[tile_b, :, :]
        acc = hl.zeros([tile_b, C, D], dtype=torch.float32)
        for i in hl.grid(N):
            s = g[tile_b, i, 0]
            op = (T.to(torch.float32) * s[:, None, None]).to(torch.bfloat16)
            acc = hl.dot(op, k[tile_b, i, :, :], acc=acc)
        out[tile_b, :, :] = acc
    return out


def _scalar_broadcast_store(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A scalar broadcast into a vector store."""
    BHN, C, D = q.size(0), hl.specialize(q.size(1)), q.size(2)
    out = torch.empty([BHN, C], dtype=torch.float32, device=q.device)
    out2 = torch.empty([BHN, C, C], dtype=torch.float32, device=q.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_d in hl.tile(D):
            kd = k[tile_bhn, :, tile_d]
            kk = hl.dot(kd, kd.transpose(-2, -1), acc=kk)
        tot = kk.sum(-1).sum(-1)
        out[tile_bhn, :] = tot[:, None].expand([tile_bhn, C])
        out2[tile_bhn, :, :] = kk
    return out, out2


def test_operand_tiles_whose_aux_loads_vary_below_their_frame_decline() -> None:
    # A loop-invariant tile is staged once per CTA; its pointwise chain
    # reading a per-iteration vector or scalar would be evaluated once too.
    t = _tensors()
    a = torch.empty((2, 64, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    g = torch.empty((2, 8, 64), dtype=torch.float32, device=CPU_DEVICE)
    for fn in (_staged_aux_vec, _staged_aux_scalar):
        assert _bind_fn(fn, (a, g, t["k"])).config_spec.cute_mma_dag is None, fn


def test_a_scalar_broadcast_into_a_vector_store_declines() -> None:
    t = _tensors3()
    bound = _bind_fn(_scalar_broadcast_store, (t["q"], t["q"]))
    assert bound.config_spec.cute_mma_dag is None


def test_wy_with_an_unaligned_key_keeps_the_natural_fp32_tiles() -> None:
    # The widen pass alone scatters into the transposed tf32 B tile; a key
    # whose rows are not 16-byte aligned takes the synchronous pass into the
    # natural tile and the per-lane B reads (admitted, every config renders).
    t = _tensors3()
    bhn, c, d = t["q"].shape
    wide = torch.empty((bhn, c, d + 8), dtype=torch.bfloat16, device=CPU_DEVICE)
    k = wide[:, :, 4 : d + 4]
    beta = torch.empty((bhn, c), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind(
        "chunk_fwd_wy_delta_helion", (k, t["v"], beta, None, None, None, False, False)
    )
    assert bound.config_spec.cute_mma_dag is not None
    program = _program(bound)

    def transposed(program: dag.MmaDagProgram, name: str) -> list[bool]:
        return [
            op.transposed_tile
            for dot in program.dots
            for op in (dot.a, dot.b)
            if op.load is not None
            and op.load.name == name
            and op.dtype == torch.float32
        ]

    # k's widened B tile stays natural; v's (aligned) is still transposed.
    assert transposed(program, "k") == [False]
    assert transposed(program, "v") == [True]
    assert transposed(_program(_bind_wy()), "k") == [True]
    for sizes, warps, stages in (((64, 64, 64), 4, 2), ((128, 32, 32), 4, 2)):
        config = _config(*sizes, warps=warps, stages=stages)
        assert bound.config_spec.backend.autotune_config_is_viable(
            bound.config_spec, config
        )
        body = _kernel_body(_source(bound, config))
        assert f"{RT}.mma_tf32(" in body and f"{RT}.lds_b32(" in body


def test_wy_dl_promoted_default_at_s1_renders_on_the_dag() -> None:
    # delta_rule at s1 (BHN 2048, D = DV = 128, bf16 beta and dvni as the
    # pipeline passes them) on B200 facts: the first seed is a four-warp dag
    # config in both decay modes and renders tensor-core MMAs.
    bf16, f32 = torch.bfloat16, torch.float32
    for gated in (False, True):
        bound = _bind(
            "chunk_bwd_wy_dL_delta_helion",
            (
                torch.empty((2048, 64, 128), dtype=bf16, device=CPU_DEVICE),
                torch.empty((2048, 64, 128), dtype=bf16, device=CPU_DEVICE),
                torch.empty((2048, 64), dtype=bf16, device=CPU_DEVICE),
                torch.empty((2048, 64, 64), dtype=f32, device=CPU_DEVICE),
                torch.empty((2048, 64, 128), dtype=f32, device=CPU_DEVICE),
                torch.empty((2048, 64, 128), dtype=bf16, device=CPU_DEVICE),
                torch.empty((2048, 64), dtype=f32, device=CPU_DEVICE)
                if gated
                else None,
                gated,
            ),
        )
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None
        facts = dataclasses.replace(facts, sm_count=148, smem_capacity=232448)
        sizes, warps, stages = dag.mma_dag_seed_configs(facts)[0]
        tunable = [a.block_id for a in facts.axes if a.kind in ("root", "loop")]
        assert ([sizes[b] for b in tunable], warps, stages) == ([64, 64, 64], 4, 2)
        body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
        assert f"{RT}.mma_bf16(" in body and SIMT_FALLBACK not in body


# --------------------------------------------------------------------------
# The tf32 shuffle route and conflict-free transposed fp32 tiles
# --------------------------------------------------------------------------


def _tf32_c_fragment(
    rows: int, cols: int
) -> dict[tuple[int, int, int, int], tuple[int, int]]:
    """Element (row, col) held by ``(am, an, j)`` of each lane of a warp's
    ``rows x cols`` fp32 fragment in the m16n8 C layout."""
    held = {}
    for am in range(rows // 16):
        for an in range(cols // 8):
            for j in range(4):
                for lane in range(32):
                    g, c = lane // 4, lane % 4
                    held[(am, an, j, lane)] = (
                        16 * am + 8 * (j // 2) + g,
                        8 * an + 2 * c + (j % 2),
                    )
    return held


def _shfl_pick(lane: list[int], flag: list[int], x: list, y: list) -> list:
    """Model of ``shfl_pick_b32``: lane r reads ``y`` (``flag``) or ``x`` of
    lane ``lane[r]``."""
    return [y[lane[r]] if flag[r] else x[lane[r]] for r in range(32)]


@pytest.mark.parametrize("shape", ("a", "b_nk", "b_kn"))
def test_shuffle_routed_words_yield_the_tf32_fragments(shape: str) -> None:
    # The emitter's lane terms: an A word (row g, k c) or an [N, K] B word
    # (n g, k c) lives in lane (g, c // 2) at register parity c % 2; a
    # [K, N] B word (k c, n g) in lane (c, g // 2) at parity g % 2.  k + 4
    # is two lanes (A, [N, K]) or sixteen lanes ([K, N]) further.
    g = [lane // 4 for lane in range(32)]
    c = [lane % 4 for lane in range(32)]
    held = _tf32_c_fragment(64, 64)
    for st in range(8):
        for atom in range(2):
            if shape == "a":
                lane = [4 * g[r] + c[r] // 2 for r in range(32)]
                flag = [c[r] % 2 for r in range(32)]
                regs = [[held[(atom, st, j, r)] for r in range(32)] for j in range(4)]
                words = [
                    _shfl_pick(lane, flag, regs[0], regs[1]),
                    _shfl_pick(lane, flag, regs[2], regs[3]),
                    _shfl_pick([x + 2 for x in lane], flag, regs[0], regs[1]),
                    _shfl_pick([x + 2 for x in lane], flag, regs[2], regs[3]),
                ]
                expected = [
                    [(16 * atom + g[r], 8 * st + c[r]) for r in range(32)],
                    [(16 * atom + g[r] + 8, 8 * st + c[r]) for r in range(32)],
                    [(16 * atom + g[r], 8 * st + c[r] + 4) for r in range(32)],
                    [(16 * atom + g[r] + 8, 8 * st + c[r] + 4) for r in range(32)],
                ]
            elif shape == "b_nk":
                # Producer [N, K]: n-atom ``atom`` is block (atom // 2, atom % 2).
                lane = [4 * g[r] + c[r] // 2 for r in range(32)]
                flag = [c[r] % 2 for r in range(32)]
                am, half = atom // 2, atom % 2
                lo = [held[(am, st, 2 * half, r)] for r in range(32)]
                hi = [held[(am, st, 2 * half + 1, r)] for r in range(32)]
                words = [
                    _shfl_pick(lane, flag, lo, hi),
                    _shfl_pick([x + 2 for x in lane], flag, lo, hi),
                ]
                # (n g, k c) and (n g, k c + 4) as (row, col) of the producer.
                expected = [
                    [(8 * atom + g[r], 8 * st + c[r]) for r in range(32)],
                    [(8 * atom + g[r], 8 * st + c[r] + 4) for r in range(32)],
                ]
            else:
                # Producer [K, N]: K rows 8 st .. of n-atom ``atom`` are block
                # (st // 2, st % 2).
                lane = [4 * c[r] + g[r] // 2 for r in range(32)]
                flag = [g[r] % 2 for r in range(32)]
                am, half = st // 2, st % 2
                lo = [held[(am, atom, 2 * half, r)] for r in range(32)]
                hi = [held[(am, atom, 2 * half + 1, r)] for r in range(32)]
                words = [
                    _shfl_pick(lane, flag, lo, hi),
                    _shfl_pick([x + 16 for x in lane], flag, lo, hi),
                ]
                expected = [
                    [(8 * st + c[r], 8 * atom + g[r]) for r in range(32)],
                    [(8 * st + c[r] + 4, 8 * atom + g[r]) for r in range(32)],
                ]
            assert words == expected, (shape, st, atom)


def test_fp32_fragments_take_the_shuffle_route_per_iteration_only() -> None:
    kwargs = {
        "producer": (64, 64),
        "consumer": (64, 64),
        "k": 64,
        "layout_p": (4, 1),
        "layout_c": (4, 1),
    }
    assert dag.frag_register_route(role="a", logical_swap=False, **kwargs) == (
        dag.ROUTE_IDENTITY
    )
    assert (
        dag.frag_register_route(role="a", logical_swap=False, elem_bytes=4, **kwargs)
        == dag.ROUTE_SHUFFLE
    )
    assert (
        dag.frag_register_route(
            role="a", logical_swap=False, elem_bytes=4, per_iteration=False, **kwargs
        )
        is None
    )
    n_first = dict(kwargs, layout_p=(1, 4), layout_c=(1, 4))
    assert (
        dag.frag_register_route(role="b", logical_swap=False, elem_bytes=4, **n_first)
        == dag.ROUTE_SHUFFLE
    )
    assert (
        dag.frag_register_route(role="b", logical_swap=False, elem_bytes=4, **kwargs)
        is None
    )


def test_transposed_fp32_tiles_keep_their_scatter_stores_conflict_free() -> None:
    # A 64 x 64 fp32 tile (256-byte rows, 16 chunks): the widen pass stores
    # the eight elements of a packet into eight rows 8 apart at one column,
    # the exchange scatters a fragment's registers likewise.  The row
    # swizzle alone lands every such lane group in one bank; XOR-ing the
    # chunk by row // 8 as well spreads the warp over all 32 banks, and the
    # eight rows of an ldmatrix (a multiple of 8 apart) keep eight chunks.
    cpr = 16

    def bank(row: int, col: int, swizzle: object) -> int:
        chunk = swizzle(row, col // 4, cpr)  # pyrefly: ignore[not-callable]
        return ((row * 256 + chunk * 16 + (col % 4) * 4) // 4) % 32

    # Widen pass: lane -> packet (k row, chunk pch) of the bf16 source, its
    # element e lands at (row 8 pch + e, column k) of the transposed tile.
    for e in range(8):
        new = {
            bank(8 * (lane % 8) + e, lane // 8, dag.transposed_swizzled_chunk)
            for lane in range(32)
        }
        old = {
            bank(8 * (lane % 8) + e, lane // 8, dag.swizzled_chunk)
            for lane in range(32)
        }
        assert len(new) == 32 and len(old) == 4
    # Exchange scatter from the M-split C layout: register (an, j) of each
    # lane (g, c) is element (row g + 8 (j // 2), col 8 an + 2 c + j % 2),
    # stored transposed.
    for an in range(8):
        for j in range(4):
            banks = {
                bank(
                    8 * an + 2 * (lane % 4) + j % 2,
                    lane // 4 + 8 * (j // 2),
                    dag.transposed_swizzled_chunk,
                )
                for lane in range(32)
            }
            assert len(banks) == 32, (an, j)
    for n0 in range(0, 64, 8):
        for chunk in range(cpr):
            assert (
                len(
                    {
                        dag.transposed_swizzled_chunk(n0 + r, chunk, cpr)
                        for r in range(8)
                    }
                )
                == 8
            )


def test_packet_rounds_add_immediates_to_one_lane_term() -> None:
    # 128 threads cover 16 rows of a 128-byte-row tile per round: the
    # thread's row, chunk and swizzled chunk are bound once in the preamble
    # and every later round adds a constant (16 rows = 2048 bytes) to both
    # the shared and the global address, so ptxas keeps one base register
    # per tile instead of one per round.
    bound = _bind_fwd()
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    preamble = body[: body.index("for md_i in")]
    assert "md_pr8 = md_tidx // 8" in preamble
    assert "md_pc8 = md_tidx % 8" in preamble
    assert "md_pz8 = md_pc8 ^ md_pr8 % 8" in preamble
    assert "md_prow" not in body and "md_pch" not in body
    rounds = re.findall(
        rf"{RT}\.cp_async_16\(md_t0 \+ md_pr8 \* 128( \+ \d+)? \+ md_pz8 \* 16, "
        r"md_\w+ \+ cutlass\.Int64\(md_pr8\) \* \d+ \+ cutlass\.Int64\(md_pc8 \* 16\)( \+ \d+)?\)",
        body,
    )
    assert len(rounds) == 4
    assert {smem.strip() for smem, _g in rounds} == {"", "+ 2048", "+ 4096", "+ 6144"}
    assert all(smem.strip() == gmem.strip() for smem, gmem in rounds)


# --------------------------------------------------------------------------
# A rescaled accumulator whose leaf an operand reads
# --------------------------------------------------------------------------


def _acc_scaled_b(k: torch.Tensor) -> torch.Tensor:
    """A chain step accumulating into T with a pointwise acc over T, B = T."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_kk in hl.tile(D):
            kt = k[tile_bhn, :, tile_kk]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        idx = hl.arange(C)
        A = torch.where(idx[:, None] > idx[None, :], -kk, 0.0)
        T = A + 1.0
        Apow = A
        for _ in range(3):
            Apow = hl.dot(Apow, Apow)
            T = hl.dot(Apow, T, acc=T * 0.5)
        out[tile_bhn, :, :] = T
    return out


def _acc_scaled_a(k: torch.Tensor) -> torch.Tensor:
    """The same with A = T."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_kk in hl.tile(D):
            kt = k[tile_bhn, :, tile_kk]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        idx = hl.arange(C)
        A = torch.where(idx[:, None] > idx[None, :], -kk, 0.0)
        T = A + 1.0
        Apow = A
        for _ in range(3):
            Apow = hl.dot(Apow, Apow)
            T = hl.dot(T, Apow, acc=T * 0.5)
        out[tile_bhn, :, :] = T
    return out


def _acc_scaled_b16(k: torch.Tensor) -> torch.Tensor:
    """The 16-bit chain (identity / movmatrix routes), B = T."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_kk in hl.tile(D):
            kt = k[tile_bhn, :, tile_kk]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        idx = hl.arange(C)
        A = torch.where(idx[:, None] > idx[None, :], -kk, 0.0)
        T = A + 1.0
        Apow = A
        for _ in range(3):
            Apow = hl.dot(Apow.to(torch.bfloat16), Apow.to(torch.bfloat16))
            T = hl.dot(Apow.to(torch.bfloat16), T.to(torch.bfloat16), acc=T * 0.5)
        out[tile_bhn, :, :] = T
    return out


@pytest.mark.parametrize(
    ("fn", "role"), ((_acc_scaled_b, "b"), (_acc_scaled_a, "a"), (_acc_scaled_b16, "b"))
)
def test_a_rescaled_acc_whose_leaf_an_operand_reads_takes_a_fresh_slot(
    fn: object, role: str
) -> None:
    # ``dot(Apow, T, acc=T * 0.5)``: the acc is materialized into its slot
    # before the MMAs (hoisted to the loop top), so an in-place accumulate
    # would hand the B operand's exchange / words / shuffles the rescaled T.
    # The dot takes a fresh slot (32 rescaled registers in, 32 copied back
    # into the carry at the iteration end); the operand reads the carry.
    t = _tensors3()
    bound = _bind_fn(fn, (t["q"],))
    assert bound.config_spec.cute_mma_dag is not None
    program = _program(bound)
    chain = program.loops[1]
    (t_dot,) = [d for d in program.dots if d.loop == chain.index and d.acc is not None]
    operand = t_dot.b if role == "b" else t_dot.a
    assert operand.kind == "frag" and operand.frag is t_dot.acc_frag
    carry = next(c for c in chain.carries if c.placeholder is t_dot.acc_frag)
    assert t_dot.acc_slot == carry.slot and t_dot.slot != carry.slot
    for warps in (4, 8):
        body = _kernel_body(_source(bound, _config(64, warps=warps, stages=2)))
        loop = body[body.index("for md_i1 in") :]
        regs = 32 if warps == 4 else 16
        rescales = re.findall(
            rf"= md_c{carry.slot}_\d+_\d+_\d+ \* v_\d+$", loop, re.MULTILINE
        )
        copies = re.findall(
            rf"^\s*md_c{carry.slot}_\d+_\d+_\d+ = md_c{t_dot.slot}_\d+_\d+_\d+$",
            loop,
            re.MULTILINE,
        )
        assert len(rescales) == regs == len(copies)
        # Nothing writes the carry's registers before the copies behind the
        # dot's last MMA: the operand reads them intact.
        writes = [
            m.start()
            for m in re.finditer(
                rf"^\s*md_c{carry.slot}_\d+_\d+_\d+ = ", loop, re.MULTILINE
            )
        ]
        assert len(writes) == regs and min(writes) > loop.rindex(f"{RT}.mma_")


def test_register_budget_prices_tf32_hoisted_words_at_eight_k_per_step() -> None:
    # wy at eight warps splits N: dot(Apow, T, acc=T) reads its own
    # accumulator as the [K, N] B operand, so its words are bound for the
    # whole K extent -- 8 K per tf32 step, 16 words for K = 64 and 8
    # columns, twice the 16-bit rate -- beside the 16 ldmatrix A words.
    bound = _bind_wy()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    sizes = {
        axis.block_id: 64
        if axis.kind in ("root", "loop")
        else (1 if axis.kind == "lead" else axis.extent)
        for axis in facts.axes
    }
    resolved = [facts.block_size(i, sizes) for i in range(len(facts.axes))]
    assert all(size is not None for size in resolved)
    assert dag.mma_dag_transient_regs(facts, resolved, 8) == 16 + 16  # pyrefly: ignore[bad-argument-type]
    assert {d.elem_bytes for d in facts.dots} == {2, 4}


def _hoisted_a_chain(k: torch.Tensor) -> torch.Tensor:
    """The chain's A operand is the dot's own accumulator (a bare acc): its
    shuffle words are bound before the MMAs."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_kk in hl.tile(D):
            kt = k[tile_bhn, :, tile_kk]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        idx = hl.arange(C)
        A = torch.where(idx[:, None] > idx[None, :], -kk, 0.0)
        T = A + 1.0
        Apow = A
        for _ in range(3):
            Apow = hl.dot(Apow, Apow)
            T = hl.dot(T, Apow, acc=T)
        out[tile_bhn, :, :] = T
    return out


def _nk_b_shuffle(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """A 16-row fp32 tile times the transpose of a per-iteration fragment:
    the B operand is the fragment read as [N, K]."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    R = hl.specialize(x.size(1))
    out = torch.empty([BHN, 3, R, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_kk in hl.tile(D):
            kt = k[tile_bhn, :, tile_kk]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        idx = hl.arange(C)
        P = torch.where(idx[:, None] > idx[None, :], -kk, 0.0)
        xr = x[tile_bhn, :, :]
        for i in range(3):
            P = hl.dot(P, P)
            out[tile_bhn, i, :, :] = hl.dot(xr, P.transpose(-2, -1))
    return out


def test_the_hoisted_a_and_nk_b_shuffle_routes_render() -> None:
    # Four warps split M: ``dot(T, Apow, acc=T)`` reads its own accumulator
    # as A, so the words of all eight K steps are bound before the MMAs
    # (32 ``md_aw`` per iteration); ``dot(xr, P^T)`` on a 16-row consumer
    # splits N while the 64 x 64 producer splits M, the [N, K] shape of the
    # route: B words picked from lanes ``4g + c // 2`` and two further.
    t = _tensors3()
    bound = _bind_fn(_hoisted_a_chain, (t["q"],))
    program = _program(bound)
    chain = program.loops[1]
    (t_dot,) = [d for d in program.dots if d.loop == chain.index and d.acc is not None]
    assert t_dot.a.frag is t_dot.acc_frag and t_dot.slot == t_dot.acc_slot
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=2)))
    loop = body[body.index("for md_i1 in") :]
    assert (
        len(
            re.findall(
                r"^\s*md_aw(?:_\d+)? = .*shfl_pick_b32\(md_qsl", loop, re.MULTILINE
            )
        )
        == 32
    )
    assert "md_tsl" not in body

    x = torch.empty((16, 16, 64), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind_fn(_nk_b_shuffle, (x, t["q"]))
    assert bound.config_spec.cute_mma_dag is not None
    program = _program(bound)
    (y_dot,) = [
        d for d in program.dots if d.a.kind == "load" and d.a.dtype == torch.float32
    ]
    assert y_dot.b.kind == "frag" and y_dot.b.logical_swap
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=2)))
    loop = body[body.index("for md_i1 in") :]
    picks = re.findall(
        rf"^\s*md_b_\d+_0, md_b_\d+_1 = \({RT}\.shfl_pick_b32\(md_qsl, md_qodd, .*"
        rf"{RT}\.shfl_pick_b32\(md_qsl \+ 2, md_qodd, ",
        loop,
        re.MULTILINE,
    )
    assert len(picks) == 16 and "md_tsl" not in body


# --------------------------------------------------------------------------
# Loop outputs in another slot, the root rescale after a loop
# --------------------------------------------------------------------------


def _single_ring(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """One loop: A staged per iteration, B the carry read as an operand,
    ``acc=T * 0.5`` (the dot takes a fresh slot)."""
    BHN, N, C = a.size(0), a.size(1), hl.specialize(a.size(2))
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=a.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        T = t[tile_bhn, :, :].to(torch.float32)
        for i in range(N):
            T = hl.dot(a[tile_bhn, i, :, :], T.to(torch.bfloat16), acc=T * 0.5)
        out[tile_bhn, :, :] = T
    return out


def _single_ring_noacc(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The same without an accumulator (a fresh slot at every version)."""
    BHN, N, C = a.size(0), a.size(1), hl.specialize(a.size(2))
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=a.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        T = t[tile_bhn, :, :].to(torch.float32)
        for i in range(N):
            T = hl.dot(a[tile_bhn, i, :, :], T.to(torch.bfloat16))
        out[tile_bhn, :, :] = T
    return out


def _single_ring_bare(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Control: the bare acc accumulates in place."""
    BHN, N, C = a.size(0), a.size(1), hl.specialize(a.size(2))
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=a.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        T = t[tile_bhn, :, :].to(torch.float32)
        for i in range(N):
            T = hl.dot(a[tile_bhn, i, :, :], T.to(torch.bfloat16), acc=T)
        out[tile_bhn, :, :] = T
    return out


def _root_rescaled_carry(k: torch.Tensor) -> torch.Tensor:
    """A root-region dot over the K loop's own carry with a rescaled acc:
    ``dot(kk, kk, acc=kk * 0.5)``."""
    BHN, C, D = k.size(0), hl.specialize(k.size(1)), k.size(2)
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        kk = hl.zeros([tile_bhn, C, C], dtype=torch.float32)
        for tile_kk in hl.tile(D):
            kt = k[tile_bhn, :, tile_kk]
            kk = hl.dot(kt, kt.transpose(-2, -1), acc=kk)
        T2 = hl.dot(kk.to(torch.bfloat16), kk.to(torch.bfloat16), acc=kk * 0.5)
        out[tile_bhn, :, :] = T2
    return out


def _gen_epi(a: torch.Tensor, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """A program-order loop (the carry is the B operand) and an epilogue
    reading a decay row."""
    BHN, N, C = a.size(0), a.size(1), hl.specialize(a.size(2))
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=a.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        T = t[tile_bhn, :, :].to(torch.float32)
        for i in range(N):
            T = hl.dot(a[tile_bhn, i, :, :], T.to(torch.bfloat16), acc=T)
        gv = g[tile_bhn, :]
        out[tile_bhn, :, :] = T * torch.exp2(gv)[:, :, None]
    return out


def _gen_epi_stored(a: torch.Tensor, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """``_gen_epi`` whose decay row is a row of a tensor the epilogue stores
    first."""
    BHN, N, C = a.size(0), a.size(1), hl.specialize(a.size(2))
    out = torch.empty([BHN, C, C], dtype=torch.float32, device=a.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        T = t[tile_bhn, :, :].to(torch.float32)
        for i in range(N):
            T = hl.dot(a[tile_bhn, i, :, :], T.to(torch.bfloat16), acc=T)
        g[tile_bhn, :, :] = T
        gv = g[tile_bhn, 0, :]
        out[tile_bhn, :, :] = T * torch.exp2(gv)[:, :, None]
    return out


def test_a_root_dot_rescaling_its_loops_carry_takes_a_fresh_slot() -> None:
    # ``dot(kk, kk, acc=kk * 0.5)`` after the K loop: both operands read
    # the carry, so the rescaled acc goes to a fresh slot (32 registers at
    # four warps) and the operand words are packed from the carry's
    # registers, which nothing writes after the loop.
    t = _tensors3()
    bound = _bind_fn(_root_rescaled_carry, (t["q"],))
    program = _program(bound)
    (dot,) = [d for d in program.dots if d.loop == -1]
    (carry,) = program.loops[0].carries
    assert dot.a.frag is dot.b.frag is dot.acc_frag
    assert dot.acc_slot == carry.slot == 0 and dot.slot == 1
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=2)))
    loop = _loop_and_after(body)
    after = loop[re.search(r"\n    \S", loop).start() + 1 :]
    assert not re.search(r"^\s*md_c0_\d+_\d+_\d+ = ", after, re.MULTILINE)
    rescales = re.findall(r"= md_c0_\d+_\d+_\d+ \* v_\d+$", after, re.MULTILINE)
    copies = re.findall(r"^\s*md_c1_\d+_\d+_\d+ = md_v(?:_\d+)?$", after, re.MULTILINE)
    assert len(rescales) == len(copies) == 32
    packs = [line for line in after.splitlines() if "pack_bf16x2(" in line]
    assert packs and all("md_c0_" in line and "md_c1_" not in line for line in packs)
    mmas = re.findall(
        rf"^\s*(md_c\d)_\d+_\d+_\d+, .* = {RT}\.mma_bf16\(", after, re.MULTILINE
    )
    assert mmas == ["md_c1"] * 32
    assert after.index(copies[-1]) < after.index(f"{RT}.mma_bf16(")


# --------------------------------------------------------------------------
# loop-invariant side loads go out ahead of the loop
# --------------------------------------------------------------------------


def _scalar_global_loads(body: str) -> list[tuple[str, str, str]]:
    """``(indent, name, address)`` of every per-thread scalar global load."""
    return re.findall(
        rf"^( *)(\w+) = {RT}\.ldg_(?:f32|f16_as_f32|bf16_as_f32)\((.*?)(?:, nc=True)?\)$",
        body,
        re.MULTILINE,
    )


def _bf16_casts_of_computed_values(body: str) -> list[str]:
    """``cutlass.BFloat16(x)`` casts of anything but an unpacked 16-bit tile
    half (``md_lo_N`` / ``md_hi_N``: the exact re-typing of a loaded element
    whose fp32 twin the chain reads, ``_Emitter.retyped``)."""
    return re.findall(r"cutlass\.BFloat16\((?!md_(?:lo|hi)(?:_\d+)?\))[^)]*\)", body)


def test_loop_invariant_side_loads_go_out_ahead_of_the_loop() -> None:
    # wy_dL with the pipeline's bf16 beta: the betaV convert pass reads beta
    # at the thread's packet rows (four rounds) and the dv_out epilogue at
    # the lane's fragment rows; neither varies with the DV loop, so the six
    # loads go out once ahead of it (in flight under the ring prologue) and
    # the iterations read the registers.  The D loop's four packet-row loads
    # sit between the loops; no scalar global load is left in a loop body.
    bound = _bind_wy_dl(beta_dtype=torch.bfloat16)
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    loads = _scalar_global_loads(body)
    assert len(loads) == 12 and all(indent == "    " for indent, _n, _a in loads)
    hoisted = [(name, a) for _i, name, a in loads if name.startswith("md_sl")]
    # The addresses: one lane base per lane term (the packet row, the
    # fragment row), the rounds' / halves' offsets as immediates.
    assert "md_lb = md_base_2 + cutlass.Int64(md_pr8) * 2\n" in body
    assert "md_lb_1 = md_base_2 + cutlass.Int64(md_L0_rowl) * 2\n" in body
    packet_rows = ["md_lb", "md_lb + 32", "md_lb + 64", "md_lb + 96"]
    assert [a for _n, a in hoisted] == [
        *packet_rows,
        "md_lb_1",
        "md_lb_1 + 16",
        *packet_rows,
    ]
    first, second = body.index("for md_i in"), body.index("for md_i1 in")
    positions = [body.index(f"{n} = {RT}.ldg_") for n, _a in hoisted]
    assert all(p < first for p in positions[:6])
    assert all(first < p < second for p in positions[6:])
    # The root epilogue's two fragment-row reads (dL = dA * beta) are no
    # loop's business and stay where they are, after the last loop.
    third = body.index("for md_i2 in")
    assert [
        body.index(f"{n} = ") > third
        for _i, n, _a in loads
        if n != "md_sl" and not n.startswith("md_sl_")
    ] == [True, True]
    # The registers (the exact fp32 values) feed the pass and the epilogue.
    assert re.search(
        r"^        md_v\w*, md_v\w* = cute\.arch\.mul_packed_f32x2\(\(md_sl, md_sl\), "
        r"\(md_lo_\d+, md_hi_\d+\)\)$",
        body,
        re.MULTILINE,
    )
    assert re.search(
        r"^        md_v\w*, md_v\w* = cute\.arch\.mul_packed_f32x2\(\(md_c1_0_0_0, md_c1_0_0_1\), "
        r"\(md_sl_8, md_sl_8\)\)$",
        body,
        re.MULTILINE,
    )
    # Loads varying with the loop stay at their site: the scalar-decay
    # recurrence reads its decay vector and scalar per iteration.
    scalar = _kernel_body(
        _source(_bind_fwd("scalar", has_h0=True), _config(64, 64, warps=4, stages=2))
    )
    assert "md_sl" not in scalar
    assert len(_scalar_global_loads(_loop_and_after(scalar))) == 5


def test_side_loads_hoist_out_of_the_outermost_loop_they_are_invariant_in() -> None:
    # Gated dqkw: the D loop nests the DV loop; the g_cs reads at the two
    # fragment layouts' rows / columns and the decay_last scalar vary with
    # neither, so they are issued once ahead of the outer ``for`` (none is
    # left inside either loop or re-read per outer iteration).
    body = _kernel_body(
        _source(_bind_dqkw(gated=True), _config(64, 32, 64, 64, 64, warps=4, stages=2))
    )
    outer = body.index("for md_i1 in")
    inner = body.index("for md_i2 in")
    assert outer < inner
    hoisted = [
        (name, address)
        for indent, name, address in _scalar_global_loads(body)
        if name.startswith("md_sl")
    ]
    assert len(hoisted) == 21
    assert all(outer > body.index(f"{name} = ") for name, _a in hoisted)
    # Their lane bases (g_cs at the two layouts' rows and columns) are bound
    # ahead of the loop as well; the scalar decay_last reads its base alone.
    bases = dict(
        re.findall(r"^\s*(md_lb\w*) = (md_base_\d+ \+ .*)$", body, re.MULTILINE)
    )
    addresses = {a.split(" + ")[0] for _n, a in hoisted}
    assert addresses <= set(bases) | {"md_base_10"}
    assert any(b.endswith("cutlass.Int64(md_L1_rowl) * 4") for b in bases.values())
    assert any(
        a.startswith(lb) and a.endswith("+ 228")
        for _n, a in hoisted
        for lb, b in bases.items()
        if b.endswith("cutlass.Int64(md_L0_coll) * 4")
    )
    assert sum(1 for _n, a in hoisted if a.startswith("md_base_")) == 1
    loops = body[outer : body.index("for md_i3 in")]
    assert not _scalar_global_loads(loops)


# --------------------------------------------------------------------------
# The raw source ring of a converted operand is one slot shorter
# --------------------------------------------------------------------------


def test_the_raw_source_ring_of_a_converted_operand_is_one_slot_shorter() -> None:
    # The convert pass consumes iteration i's raw tile ahead of the iteration
    # barrier, and the refill past that barrier (iteration i + stages - 1)
    # may take its slot: stages - 1 raw slots are ever live.  wy_dL with the
    # pipeline's dtypes: dW (fp32) is converted per D iteration; at two
    # stages its raw ring is one 16 KB slot (the operand rings keep two), so
    # the D loop's frame shrinks from 80 KB to 64 KB; the fp32 source of
    # T^T (16 KB, converted once before the loops) and the 8 KB exchange
    # tile of dAinv (written after the D loop, read by the chain dot before
    # the kk loop) overlay the loop frames instead of following them:
    # 114944 -> 73984 bytes, three CTAs per SM of shared memory.
    assert [dag.mma_dag_raw_slots(s) for s in (1, 2, 3, 4)] == [1, 1, 2, 3]
    bf16 = torch.bfloat16
    bound = _bind_wy_dl(beta_dtype=bf16, dvni_dtype=bf16, bhn=2048)
    two = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    assert "alloc_smem(cutlass.Uint8, 73984, alignment=128)" in two
    assert "md_t2r = md_s0 + 0\n" in two and "md_t2 = md_s0 + 65536\n" in two
    assert "md_x5 = md_s0 + 16384\n" in two
    assert "md_rstage" not in two
    # The exchange tile is written behind a barrier (the D loop's reads of
    # the overlaid frame are done) and the kk loop's prologue follows
    # another (the chain dot's reads of it are done).
    second, third = two.index("for md_i1 in"), two.index("for md_i2 in")
    write = two.index(f"{RT}.sts_b32(md_x5", second)
    assert second < two.rindex("cute.arch.barrier()", 0, write) < write
    read = two.rindex(f"{RT}.ldmatrix_x4_trans(md_x5", 0, third)
    assert write < read < two.rindex("cute.arch.barrier()", 0, third) < third
    # T^T's pass (its raw reads) and the barrier precede the first loop's
    # ring prologue, which rewrites the overlaid bytes.
    first = two.index("for md_i in cutlass.range(")
    assert (
        two.rindex(f"{RT}.lds_v4_f32(md_t2r", 0, first)
        < two.rindex("cute.arch.barrier()", 0, first)
        < two.rindex(f"{RT}.cp_async_16(md_t0 ", 0, first)
    )
    # The raw tile is addressed without a slot term, the bf16 tile by stage.
    assert re.search(rf"{RT}\.lds_v4_f32\(md_t3r \+ md_pr16 \* 256 \+", two)
    assert re.search(rf"{RT}\.cp_async_16\(md_t3r \+ md_pr16 \* 256 \+", two)
    assert "md_t3 + md_stage1 * 8192" in two
    # Three stages: the raw ring has two slots walked by the loop's own
    # raw stage (iteration % 2); the prologue fills slots 0 and 1, the pass
    # reads this iteration's slot and the refill rewrites it.
    three = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=3)))
    assert "alloc_smem(cutlass.Uint8, 114944, alignment=128)" in three
    loop = three[three.index("for md_i1 in") :]
    assert "md_rstage = md_i1 % 2" in loop
    assert loop.count(f"{RT}.lds_v4_f32(md_t3r + md_rstage * 16384 + ") == 8
    assert loop.count(f"{RT}.cp_async_16(md_t3r + md_rstage * 16384 + ") == 8
    prologue = three[three.index("md_t3r = ") : three.index("for md_i1 in")]
    assert prologue.count(f"{RT}.cp_async_16(md_t3r + md_pr16 * 256 + ") == 8
    assert (
        prologue.count(f"{RT}.cp_async_16(md_t3r + 1 * 16384 + md_pr16 * 256 + ") == 8
    )
    # The planner's budget prices the raw ring the same way.
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    resolved = [64, 64, 64, 2048, 64, 64]
    assert dag.mma_dag_smem_estimate(facts, resolved, 2, 4) == 73984
    assert dag.mma_dag_smem_estimate(facts, resolved, 3, 4) == 114944


# --------------------------------------------------------------------------
# An exact dtype round trip is the identity, not a pass
# --------------------------------------------------------------------------


def test_an_exact_dtype_round_trip_stages_the_tile_as_loaded() -> None:
    # wy_dL reads dvni as ``dvni[...].float()`` and feeds ``.to(bf16)`` of it
    # to two dots: with a bf16 dvni (the pipeline's) the chain leaves every
    # element's bits as loaded, so the tile takes the direct route (cp.async
    # into the operand ring, no per-packet rewrite) and both dots share it.
    bf16, f32 = torch.bfloat16, torch.float32
    bound = _bind_wy_dl(beta_dtype=bf16, dvni_dtype=bf16)
    program = _program(bound)
    dvni = [
        (dot, operand)
        for dot in program.dots
        for operand in (dot.a, dot.b)
        if operand.load is not None and operand.load.name == "dvni"
    ]
    assert len(dvni) == 2
    assert all(not operand.pointwise for _dot, operand in dvni)
    assert len({operand.chain_key for _dot, operand in dvni}) == 1
    assert dvni[0][1].chain_key[1] == ()
    # An fp32 dvni narrows for real: a convert pass with its raw tile.
    wide = _bind_wy_dl(beta_dtype=bf16, dvni_dtype=f32)
    passes = [
        operand
        for dot in _program(wide).dots
        for operand in (dot.a, dot.b)
        if operand.load is not None and operand.load.name == "dvni"
    ]
    assert all(operand.pointwise for operand in passes)
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    # The DV loop's only pass is betaV's (four packet rounds): one
    # lds_v4 / sts_v4 pair per round, not two.
    loop = body[body.index("for md_i in") : body.index("for md_i1 in")]
    assert loop.count(f"{RT}.sts_v4_b32(") == 4 and loop.count(f"{RT}.lds_v4_b32(") == 4
    # The rule is the bits' identity only: the lossless widenings.
    convert = torch.ops.prims.convert_element_type.default
    chain = lambda *dtypes: tuple((convert, ("source", d), (), ()) for d in dtypes)  # noqa: E731
    assert dag._exact_round_trip(chain(bf16, f32), bf16)
    assert dag._exact_round_trip(
        chain(torch.float16, torch.float64, f32), torch.float16
    )
    assert dag._exact_round_trip(chain(f32, torch.float64), f32)
    assert not dag._exact_round_trip(chain(f32, bf16), f32)
    assert not dag._exact_round_trip(chain(bf16, torch.float16), bf16)
    assert not dag._exact_round_trip(chain(bf16, f32), torch.float16)
    assert dag._exact_round_trip(chain(bf16), bf16)
    assert not dag._exact_round_trip(chain(bf16, torch.int32), bf16)
    assert not dag._exact_round_trip((), bf16)


# --------------------------------------------------------------------------
# The side-load hoist respects stores and loop levels
# --------------------------------------------------------------------------


def _rmw_vec_store(
    k: torch.Tensor, v: torch.Tensor, s_out: torch.Tensor
) -> torch.Tensor:
    """Every iteration accumulates the state's row sums into s_out in global
    memory: s_out[idx, tile_d] is read (index invariant in i_t) and written
    per iteration."""
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)
    h_out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            s_out[idx, tile_d] = s_out[idx, tile_d] + h_acc.sum(-1)
        h_out[idx, tile_d, tile_dv] = h_acc
    return h_out


def _inner_varying_gate(
    do: torch.Tensor, h: torch.Tensor, gdv: torch.Tensor
) -> torch.Tensor:
    """Outer D loop, inner DV loop; the in-place pass of do_dv reads gdv at
    the DV tile: a vector varying with the inner loop only."""
    BHN = do.size(0)
    C = hl.specialize(do.size(1))
    D = h.size(1)
    DV = hl.specialize(do.size(2))
    out = torch.empty([BHN, C, D], dtype=torch.float32, device=do.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        for tile_d in hl.tile(D):
            acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            for tile_dv in hl.tile(DV):
                gv = gdv[tile_bhn, tile_dv].to(torch.float32)  # [1, dv]
                do_dv = (
                    do[tile_bhn, :, tile_dv].to(torch.float32) * gv[:, None, :]
                ).to(torch.bfloat16)
                S_h = h[tile_bhn, tile_d, tile_dv]
                acc = hl.dot(do_dv, S_h.transpose(-2, -1), acc=acc)
            out[tile_bhn, :, tile_d] = acc
    return out


def _inner_varying_row_gate(
    do: torch.Tensor, h: torch.Tensor, gd: torch.Tensor
) -> torch.Tensor:
    """gd varies with the OUTER D loop only (rows of S_h) and is read in the
    inner loop: hoisting it ahead of the inner loop is legal, ahead of the
    outer is not."""
    BHN = do.size(0)
    C = hl.specialize(do.size(1))
    D = h.size(1)
    DV = hl.specialize(do.size(2))
    out = torch.empty([BHN, C, D], dtype=torch.float32, device=do.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        for tile_d in hl.tile(D):
            acc = hl.zeros([tile_bhn, C, tile_d], dtype=torch.float32)
            for tile_dv in hl.tile(DV):
                g = gd[tile_bhn, tile_d].to(torch.float32)  # [1, bd]
                S_h = (
                    h[tile_bhn, tile_d, tile_dv].to(torch.float32) * g[:, :, None]
                ).to(torch.bfloat16)
                do_dv = do[tile_bhn, :, tile_dv]
                acc = hl.dot(do_dv, S_h.transpose(-2, -1), acc=acc)
            out[tile_bhn, :, tile_d] = acc
    return out


def test_a_side_load_hoists_only_out_of_loops_nested_in_the_one_it_varies_with() -> (
    None
):
    # A vector varying with the inner loop was hoisted ahead of the outer loop
    # (its iteration base is bound inside the outer loop: a NameError at
    # compile time), while one varying with the outer loop alone was never
    # hoisted ahead of the inner loop.
    bf16, f32 = torch.bfloat16, torch.float32
    do = torch.empty((4, 64, 128), dtype=bf16, device=CPU_DEVICE)
    h = torch.empty((4, 128, 128), dtype=bf16, device=CPU_DEVICE)
    g = torch.empty((4, 128), dtype=f32, device=CPU_DEVICE)
    inner = _kernel_body(
        _source(
            _bind_fn(_inner_varying_gate, (do, h, g)),
            _config(64, 64, warps=4, stages=2),
        )
    )
    assert "md_sl" not in inner
    # gdv's column reads (one per packet element) are prefetched one inner
    # iteration ahead: the first iteration's loads sit in the outer loop's
    # body ahead of the inner ``for``, the refills inside the inner loop
    # under the trip guard; none is hoisted ahead of the outer loop.
    loads = _scalar_global_loads(inner)
    assert loads and all(name.startswith("md_px") for _i, name, _a in loads)
    ahead = [load for load in loads if load[0] == "        "]
    refills = [load for load in loads if load[0] == "                "]
    assert ahead and len(ahead) == len(refills) == len(loads) // 2
    assert "if md_i1 + 1 < " in inner
    outer = _kernel_body(
        _source(
            _bind_fn(_inner_varying_row_gate, (do, h, g)),
            _config(64, 64, warps=4, stages=2),
        )
    )
    # gd's four packet-row loads (one per round) sit in the outer loop's
    # body ahead of the inner ``for``, and the inner loop reads the registers.
    first, second = outer.index("for md_i in"), outer.index("for md_i1 in")
    hoisted = [
        (indent, name)
        for indent, name, _address in _scalar_global_loads(outer)
        if name.startswith("md_sl")
    ]
    assert len(hoisted) == 4 and all(indent == "        " for indent, _n in hoisted)
    positions = [outer.index(f"{name} = {RT}.ldg_") for _i, name in hoisted]
    assert all(first < p < second for p in positions)
    assert f"{RT}.ldg_f32(" not in outer[second:]
    assert re.search(
        r"^            md_v\w*, md_v\w* = cute\.arch\.mul_packed_f32x2\("
        r"\(md_lo_\d+, md_hi_\d+\), \(md_sl\w*, md_sl\w*\)\)$",
        outer,
        re.MULTILINE,
    )


def _loop_body(body: str, var: str) -> str:
    """The lines of the ``for var in`` loop's body (up to the first line back
    at the ``for``'s indentation)."""
    lines = body.splitlines()
    start = next(i for i, line in enumerate(lines) if f"for {var} in" in line)
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent
    ):
        end += 1
    return "\n".join(lines[start + 1 : end])


def test_the_aux_ring_stages_vectors_along_the_contiguous_dim_once_per_cta() -> None:
    # ``cute_mma_dag_aux="ring"``: gdv, an fp32 vector along the source
    # tile's contiguous dim varying with the inner loop, is copied once per
    # CTA and stage (its 64-element block: 256 B, a 16-byte cp.async per
    # thread of the first 16) with the ring's prologue and refills; every
    # packet reads its 8-element run with two 16-byte shared reads behind
    # the barrier ahead of the passes.  No per-thread global side load (nor
    # its prefetch) remains.
    bf16, f32 = torch.bfloat16, torch.float32
    do = torch.empty((4, 64, 128), dtype=bf16, device=CPU_DEVICE)
    h = torch.empty((4, 128, 128), dtype=bf16, device=CPU_DEVICE)
    g = torch.empty((4, 128), dtype=f32, device=CPU_DEVICE)
    bound = _bind_fn(_inner_varying_gate, (do, h, g))
    config = _config(64, 64, warps=4, stages=2, cute_mma_dag_aux="ring")
    body = _kernel_body(_source(bound, config))
    assert not _scalar_global_loads(body)
    copies = re.findall(
        rf"^( *)if md_tidx < 16:\n\1    {RT}\.cp_async_16\(md_aux1 \+ (.*?) \+ md_tidx \* 16,"
        rf" (md_ar\w*) \+ cutlass\.Int64\(md_tidx\) \* 16\)$",
        body,
        re.MULTILINE,
    )
    # The peeled prologue ahead of the outer loop, the refill in the inner
    # loop, and the early prologue of the outer loop's next iteration (its
    # first stage again) each copy the vector.
    assert [offset for _i, offset, _src in copies] == [
        "0",
        "(md_stage1 + 1) % 2 * 256 + 0",
        "0",
    ]
    inner = _loop_body(body, "md_i1")
    wait = inner.index("cute.arch.cp_async_wait_group(0)")
    barrier = inner.index("cute.arch.barrier()")
    reads = [
        m.start()
        for m in re.finditer(
            rf"{RT}\.lds_v4_f32\(md_aux1 \+ md_stage1 \* 256 \+ 0"
            rf" \+ cutlass\.Int32\(md_pc8\) \* 32( \+ 16)?\)",
            inner,
        )
    ]
    assert len(reads) == 8 and wait < barrier < reads[0]
    plan = _plan_of(bound, config)
    ring = plan.aux_rings[1]
    assert (ring.words, ring.vector_bytes, ring.stage_bytes) == (0, 256, 256)
    assert plan.aux_bytes == 512 and f"md_aux1 = md_s0 + {ring.offset}" in body
    assert f"alloc_smem(cutlass.Uint8, {plan.smem_bytes}," in body


def test_the_aux_ring_gives_per_thread_words_to_scalars_and_row_vectors() -> None:
    # fwd_h's gated chain: the per-chunk decay scalar and gc's two
    # packet-row elements ride the ring as this thread's 4-byte slots
    # (copied and read by the same thread: no barrier), read behind the
    # stage's wait, which moves ahead of the scalar hoists consuming them.
    bound = _bind_fwd("gated")
    config = _config(16, 16, warps=2, stages=8, cute_mma_dag_aux="ring")
    body = _kernel_body(_source(bound, config))
    loop = _loop_body(body, "md_i")
    assert not _scalar_global_loads(loop)
    copies = re.findall(
        rf"^ *{RT}\.cp_async_4\(md_aux0 \+ (.*?) \+ md_tidx \* 4, (.*)\)$",
        body,
        re.MULTILINE,
    )
    # Seven prologue stages and the refill, three slots each.
    assert len(copies) == 8 * 3
    reads = re.findall(
        rf"^ *(\w+) = {RT}\.lds_f32\(md_aux0 \+ md_stage \* 768 \+ (\d+) \+ md_tidx \* 4\)$",
        loop,
        re.MULTILINE,
    )
    assert [int(offset) for _n, offset in reads] == [0, 256, 512]
    first = loop.index(f"{reads[0][0]} = ")
    assert loop.index("cute.arch.cp_async_wait_group(6)") < first
    assert "cute.arch.barrier()" not in loop[:first]
    plan = _plan_of(bound, config)
    ring = plan.aux_rings[0]
    assert (ring.words, ring.vector_bytes, ring.stage_bytes) == (3, 0, 768)
    assert plan.aux_bytes == 8 * 768
    assert f"alloc_smem(cutlass.Uint8, {plan.smem_bytes}," in body


def test_the_aux_ring_stages_16_bit_vectors_as_packed_runs() -> None:
    # mamba2 chunk scan: dt and dA_cumsum (bf16 vectors along the chunk, the
    # cb tile's contiguous dim) are two 128-byte vectors per stage; a packet
    # reads its eight elements as one 16-byte shared read it unpacks.
    bound = _bind_chunk_scan()
    config = _config(16, 64, 64, warps=2, stages=2, cute_mma_dag_aux="ring")
    body = _kernel_body(_source(bound, config))
    loop = _loop_body(body, "md_i")
    assert f"{RT}.ldg_" not in loop
    copies = re.findall(
        rf"^( *)if md_tidx < 8:\n\1    {RT}\.cp_async_16\(md_aux0 \+ (.*?) \+ md_tidx \* 16, ",
        body,
        re.MULTILINE,
    )
    assert [offset for _i, offset in copies] == [
        "0",
        "128",
        "(md_stage + 1) % 2 * 256 + 0",
        "(md_stage + 1) % 2 * 256 + 128",
    ]
    reads = re.findall(
        rf"^ *(\w+), \w+, \w+, \w+ = {RT}\.lds_v4_b32\(md_aux0 \+ md_stage \* 256"
        rf" \+ (\d+) \+ cutlass\.Int32\(md_pc8\) \* 16\)$",
        loop,
        re.MULTILINE,
    )
    assert sorted(int(offset) for _w, offset in reads) == [0, 0, 128, 128]
    for word, _offset in reads:
        assert f"{RT}.unpack_bf16x2({word})" in loop
    plan = _plan_of(bound, config)
    ring = plan.aux_rings[0]
    assert (ring.words, ring.vector_bytes, plan.aux_bytes) == (0, 256, 512)
    assert f"alloc_smem(cutlass.Uint8, {plan.smem_bytes}," in body


def test_the_aux_knob_has_a_subject_only_with_loop_varying_side_loads() -> None:
    gated = _bind_fwd("gated").config_spec
    assert gated.cute_mma_dag_facts is not None
    assert gated.cute_mma_dag_facts.aux_candidates
    assert gated.cute_mma_dag_aux is not None
    assert gated.cute_mma_dag_aux.choices == ("register", "ring")
    plain = _bind_fwd_o()
    spec = plain.config_spec
    assert spec.cute_mma_dag_facts is not None
    assert not spec.cute_mma_dag_facts.aux_candidates
    assert spec.cute_mma_dag_aux is None
    config = _config(1, 64, 64, warps=4, stages=2, cute_mma_dag_aux="ring").config
    with plain.env:
        spec.normalize(config)
    assert config["cute_mma_dag_aux"] == "register"


def test_exact_16_bit_values_widen_without_a_cast_pair() -> None:
    # The pass holds a 16-bit tile element (an unpacked packet half) and a
    # widening side load exactly in fp32; the pointwise lowering's implicit
    # promotion to fp32 (``cb_local *= exp2(...)``: bf16 times fp32) reads
    # that value instead of re-typing it (``cutlass.BFloat16(x)``, a
    # ``cvt`` on the conversion pipe) and widening it back.
    bound = _bind_chunk_scan()
    body = _kernel_body(_source(bound, _config(16, 64, 64, warps=2, stages=2)))
    loop = _loop_body(body, "md_i")
    assert re.search(r"^ *v_\d+ = md_lo \* md_v\w*$", loop, re.MULTILINE)
    assert re.search(r"^ *v_\d+ = md_hi \* md_v\w*$", loop, re.MULTILINE)
    typed = re.findall(
        r"^ *(md_v\w*) = cutlass\.BFloat16\(md_(?:lo|hi)\w*\)$", loop, re.MULTILINE
    )
    assert typed and all(
        f"cutlass.Float32({name})" not in body
        and len(re.findall(rf"\b{name}\b", loop)) == 1
        for name in typed
    )
    # The bf16 side vectors (dt, dA_cumsum) likewise: their fp32 loads feed
    # the arithmetic directly.
    assert re.search(
        rf"^ *(md_v\w*) = {RT}\.ldg_bf16_as_f32\([^\n]*\)\n *md_v\w* = cutlass\.BFloat16\(\1\)\n"
        rf"(?: *v_\d+ = [^\n]*\n)? *v_\d+ = \1 \* ",
        loop,
        re.MULTILINE,
    )


def test_the_tma_ring_fetches_each_ring_tile_as_one_bulk_tensor_copy() -> None:
    # ``cute_mma_dag_loads="tma"`` in the warp mode: the K loop's two ring
    # tiles (cb [16, 64] and x [64, 64], 128-byte rows) arrive by one bulk
    # tensor copy each, issued by thread 0 behind a proxy fence and an
    # expect-tx of the stage's bytes on the slot's mbarrier; every thread
    # waits on the slot's barrier with the parity of the loop's running
    # count where the cp.async wait sat; the root dot's staged tiles keep
    # their packets.  The host builds one tensor map per tile (grid
    # constants after the arguments), the shared allocation is 1024-byte
    # aligned for the address-based swizzle.
    bound = _bind_chunk_scan()
    assert bound.config_spec.cute_mma_dag_facts is not None
    assert bound.config_spec.cute_mma_dag_facts.tma_ring
    config = _config(16, 64, 64, warps=2, stages=2, cute_mma_dag_loads="tma")
    with bound.env:
        assert (
            bound.config_spec.normalized_config(config).config["cute_mma_dag_loads"]
            == "tma"
        )
    source = _source(bound, config)
    body = _kernel_body(source)
    assert "md_tma0: cutlass.GridConstant[_helion_mma_dag_tc.cuda.TensorMap]" in body
    # 30720 of ring and staged tiles + the epilogue's [16, 64] x tile staged
    # ahead of the loops under ``tma``.
    assert "alloc_smem(cutlass.Uint8, 32768, alignment=1024)" in body
    assert "md_full0 = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)" in body
    assert "md_cnt0 = cutlass.Int32(0)" in body
    for stage in range(2):
        assert f"_helion_mma_dag_tc.prims.mbarrier_init(md_full0 + {stage}, 1)" in body
    assert "_helion_mma_dag_tc.prims.fence_mbarrier_init()" in body
    expects = re.findall(
        r"^( *)_helion_mma_dag_tc\.prims\.mbarrier_arrive_expect_tx\(md_full0 \+ (.*), 10240\)$",
        body,
        re.MULTILINE,
    )
    assert [slot for _i, slot in expects] == ["(md_cnt0 + 0) % 2", "(md_stage + 1) % 2"]
    copies = re.findall(
        r"^ *_helion_mma_dag_tc\.prims\.cp_async_bulk_tensor_shared_cta_global\("
        r"md_smem \+ \((\d+) \+ (.*?) \* (\d+)\), (md_tma\d)\.get_ptr\(\), \[(.*)\], md_full0 \+ (.*)\)$",
        body,
        re.MULTILINE,
    )
    assert [(c[0], c[2], c[3]) for c in copies] == [
        ("0", "2048", "md_tma0"),
        ("4096", "8192", "md_tma1"),
    ] * 2
    # cb's coordinates: K (the loop axis) innermost, then the M tile's base.
    assert copies[0][4].startswith("cutlass.Int32(0 * 64), cutlass.Int32(md_a0)")
    assert copies[2][4].startswith(
        "cutlass.Int32((md_i + 1) * 64), cutlass.Int32(md_a0)"
    )
    loop = _loop_body(body, "md_i")
    assert "md_stage = (md_cnt0 + md_i) % 2" in loop
    assert re.search(r"^ *md_par = \(md_cnt0 \+ md_i\) // 2 & 1$", loop, re.MULTILINE)
    wait = loop.index("_helion_mma_dag_tc.mbar_wait(md_full0 + md_stage, md_par)")
    assert (
        loop.index("cute.arch.cp_async_wait_group(0)")
        < wait
        < loop.index("cute.arch.barrier()")
    )
    # Fences precede every issue (thread 0's, ahead of its expect-tx), and
    # every thread fences at the top of the iteration, past its waits and
    # ahead of the in-place pass (its previous pass's stores precede the
    # barrier before the refill of that slot); the ring tiles have no
    # packets left; the root dot's staged tiles (md_t0 / md_t1) still ride
    # cp.async.
    assert body.count("cute.arch.fence_proxy(kind='async.shared', space='cta')") == 4
    writer_fence = loop.index("cute.arch.fence_proxy(kind='async.shared', space='cta')")
    assert wait < writer_fence < loop.index(f"{RT}.lds_v4_b32(")
    assert "if md_tidx == 0:" not in loop[wait:writer_fence]
    assert (
        f"{RT}.cp_async_16(md_t2" not in body and f"{RT}.cp_async_16(md_t3" not in body
    )
    assert f"{RT}.cp_async_16(md_t0" in body
    assert re.search(r"^ {4}md_cnt0 = md_cnt0 \+ \(1 - 0\)$", body, re.MULTILINE)
    assert "'kind': 'helion_mma_dag_tma', 'tma_count': 2" in source
    assert (
        "'tma0_box': [64, 16, 1, 1, 1], 'tma0_swizzle': 's128b', 'tma0_elem': 2"
        in source
    )
    assert "'tma1_box': [64, 1, 64, 1], 'tma1_swizzle': 's128b'" in source
    # The packet ring on request.
    lane = _kernel_body(_source(bound, _config(16, 64, 64, warps=2, stages=2)))
    assert (
        "md_full" not in lane and "mbar_wait" not in lane and "alignment=128)" in lane
    )


def test_a_re_entered_tma_ring_counts_its_iterations_across_the_outer_loop() -> None:
    # The nested ring's slot and barrier parity follow a running count the
    # outer loop never resets: the count advances by the inner trip count
    # after the inner loop, ahead of the next iteration's early prologue.
    bf16, f32 = torch.bfloat16, torch.float32
    do = torch.empty((4, 64, 128), dtype=bf16, device=CPU_DEVICE)
    h = torch.empty((4, 128, 128), dtype=bf16, device=CPU_DEVICE)
    g = torch.empty((4, 128), dtype=f32, device=CPU_DEVICE)
    bound = _bind_fn(_inner_varying_gate, (do, h, g))
    body = _kernel_body(
        _source(bound, _config(64, 64, warps=4, stages=2, cute_mma_dag_loads="tma"))
    )
    outer = _loop_body(body, "md_i")
    inner_end = outer.index("md_cnt1 = md_cnt1 + (2 - 0)")
    assert outer.index("for md_i1 in") < inner_end
    early = outer.index(
        "_helion_mma_dag_tc.prims.mbarrier_arrive_expect_tx(md_full1 + (md_cnt1 + 0) % 2, 16384)",
        inner_end,
    )
    assert early > inner_end
    inner = _loop_body(body, "md_i1")
    assert "md_stage1 = (md_cnt1 + md_i1) % 2" in inner
    assert "_helion_mma_dag_tc.mbar_wait(md_full1 + md_stage1, md_par)" in inner
    # The aux ring shares the slot expression.
    ring = _kernel_body(
        _source(
            bound,
            _config(
                64,
                64,
                warps=4,
                stages=2,
                cute_mma_dag_loads="tma",
                cute_mma_dag_aux="ring",
            ),
        )
    )
    assert re.search(r"cp_async_16\(md_aux1 \+ \(md_cnt1 \+ 0\) % 2 \* 256", ring)


def test_the_region_renderer_stages_epilogue_tile_inputs_in_the_fixed_region() -> None:
    # chunk_scan's epilogue ``acc += x_residual * D`` reads a loop-invariant
    # [64, 64] tile of ``x`` at the fragment coordinates: the region renderer
    # fetches it by TMA into the fixed region with the root's staged tiles
    # (issued ahead of the loops, landing on the root's staged barrier) and
    # the epilogue reads its rows from shared memory, instead of per-lane
    # global pairs issued after the last MMA.  Under the ``lane`` loads mode
    # the tile keeps the per-lane global loads.
    bound = _bind_chunk_scan()
    spec = bound.config_spec
    tcgen05 = {
        "cute_mma_dag": "tcgen05",
        "cute_mma_dag_warps": 4,
        "cute_mma_dag_stages": 1,
        "cute_mma_dag_store": "staged",
        "cute_mma_dag_exchange": "auto",
        "cute_mma_dag_pass_warps": "off",
        "cute_mma_dag_persistent": 0,
    }
    bodies = {}
    for loads in ("tma", "lane"):
        with bound.env:
            config = spec.normalized_config(
                helion.Config(
                    block_sizes=[64, 64, 64], cute_mma_dag_loads=loads, **tcgen05
                )
            )
        bodies[loads] = _kernel_body(bound.to_code(config))
    expects = {
        loads: [
            int(n)
            for n in re.findall(r"mbarrier_arrive_expect_tx\(md_sfr, (\d+)\)", body)
        ]
        for loads, body in bodies.items()
    }
    assert len(expects["tma"]) == 1
    assert expects["tma"][0] == expects["lane"][0] + 64 * 64 * 2
    # One more bulk copy ahead of the loops than the lane render's (the
    # root operands' staged tiles), into the fixed region behind them.
    copies = {
        loads: set(
            re.findall(
                r"cp_async_bulk_tensor_shared_cta_global\(md_smem \+ \((\d+ \+ \d+)\), "
                r"(md_tma\d+)\.get_ptr\(\), \[[^\]]*\], md_sfr\)",
                body[: body.index("for md_i in cutlass.range(")],
            )
        )
        for loads, body in bodies.items()
    }
    assert len(copies["tma"]) == len(copies["lane"]) + 1
    [(fixed_offset, _param)] = copies["tma"] - copies["lane"]
    offset = sum(int(n) for n in fixed_offset.split(" + "))
    assert offset == max(
        sum(int(n) for n in fixed.split(" + ")) for fixed, _ in copies["tma"]
    )
    body = bodies["tma"]
    # The epilogue waits on the root's staged barrier and reads the tile's
    # rows from the fixed region; no per-lane global load remains there.
    epilogue = body[body.rindex("for md_i in cutlass.range(") :]
    wait = epilogue.index("mbar_wait(md_sfr, 0)")
    read = epilogue.index(f"{RT}.lds_v4_b32(md_s0 + {offset} + md_row64 * 128")
    assert wait < read
    assert f"{RT}.ldg_b32(" not in epilogue
    lane = bodies["lane"]
    assert f"{RT}.ldg_b32(" in lane[lane.rindex("for md_i in cutlass.range(") :]


def test_the_warp_mode_stages_epilogue_tile_inputs_ahead_of_the_loops() -> None:
    # The warp mode's sibling of the region renderer's fixed-region tiles:
    # chunk_scan's epilogue ``acc += x_residual * D`` reads a loop-invariant
    # [64, 64] tile of ``x``.  Under ``cute_mma_dag_loads="tma"`` the tile is
    # a shared tile of the root frame (``md_d0``, after the loops' frames),
    # cp.async'd in the root's staged group ahead of the ring prologue, and
    # the epilogue reads its pairs from the swizzled rows behind a drain and
    # a barrier; under ``lane`` the epilogue keeps its per-lane global pairs.
    # The facts name the subject and count the tile once in the root frame.
    bound = _bind_chunk_scan()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.fixed_derived
    assert [(tile.ring, tile.frame, tile.copies) for tile in facts.tiles].count(
        (False, -1, 1)
    ) == 3
    bodies = {}
    for loads in ("tma", "lane"):
        config = _config(64, 64, 64, warps=8, stages=2, cute_mma_dag_loads=loads)
        with bound.env:
            normalized = bound.config_spec.normalized_config(config)
        assert normalized.config["cute_mma_dag_loads"] == loads
        bodies[loads] = _kernel_body(_source(bound, config))
    body = bodies["tma"]
    assert "alloc_smem(cutlass.Uint8, 57344, alignment=1024)" in body
    assert "alloc_smem(cutlass.Uint8, 49152, alignment=128)" in bodies["lane"]
    assert "md_d0 = md_s0 + 49152" in body
    assert "md_d0" not in bodies["lane"]
    prologue = body[: body.index("for md_i in cutlass.range(")]
    issues = re.findall(
        r"^ *_helion_mma_dag\.cp_async_16\(md_d0 \+ md_pr8 \* 128 \+ (?:4096 \+ )?md_pz8 \* 16, "
        r"(md_base_\d+) \+ cutlass\.Int64\(md_pr8\) \* 1024 \+ cutlass\.Int64\(md_pc8 \* 16\)(?: \+ 32768)?\)$",
        prologue,
        re.MULTILINE,
    )
    assert len(issues) == 2 and len(set(issues)) == 1
    # Its group is the root's staged group (committed once, before the
    # ring prologue's commits).
    first_commit = prologue.index("cute.arch.cp_async_commit_group()")
    assert prologue.rindex(f"{RT}.cp_async_16(md_d0") < first_commit
    epilogue = body[body.rindex("for md_i in cutlass.range(") :]
    epilogue = epilogue[epilogue.index("\n    ") :]  # past the loop header
    reads = re.findall(rf"{RT}\.lds_b32\(md_d0 \+ ", epilogue)
    assert len(reads) == 8
    drain = epilogue.index("cute.arch.cp_async_wait_group(0)\n    cute.arch.barrier()")
    assert drain < epilogue.index(f"{RT}.lds_b32(md_d0")
    assert f"{RT}.ldg_b32(" not in epilogue
    lane = bodies["lane"]
    lane_epilogue = lane[lane.rindex("for md_i in cutlass.range(") :]
    assert lane_epilogue.count(f"{RT}.ldg_b32(") == 8


def test_the_root_loops_ring_prologue_goes_out_ahead_of_the_root_dots_wait() -> None:
    # chunk_scan's root: a dot on two staged tiles, then the K loop, then the
    # epilogue.  The K loop's ring prologue (its first ``stages - 1`` stages)
    # used to follow the root dot (issued after the staged tiles' wait and
    # the MMAs: the ring tiles' latency sat in series with the staged
    # tiles').  Now it goes out right behind the root's staged group, and
    # the root dot's settle waits for all but those younger groups
    # (``cp_async_wait_group(stages - 1)``), with either ring form.
    bound = _bind_chunk_scan()
    for loads, stages, issue in (
        ("tma", 2, "_helion_mma_dag_tc.prims.cp_async_bulk_tensor_shared_cta_global("),
        ("lane", 2, f"{RT}.cp_async_16(md_t2"),
        ("lane", 3, f"{RT}.cp_async_16(md_t2"),
    ):
        config = _config(64, 64, 64, warps=4, stages=stages, cute_mma_dag_loads=loads)
        body = _kernel_body(_source(bound, config))
        prologue = body[: body.index("for md_i in cutlass.range(")]
        staged = prologue.index(f"{RT}.cp_async_16(md_t0")
        commits = [
            m.start()
            for m in re.finditer(r"cute\.arch\.cp_async_commit_group\(\)", prologue)
        ]
        assert len(commits) == stages
        wait = prologue.index(f"cute.arch.cp_async_wait_group({stages - 1})")
        assert staged < commits[0] < prologue.index(issue) < commits[-1] < wait
        assert wait < prologue.index("cute.arch.barrier()", wait)
        assert prologue.index(f"{RT}.mma_bf16(") > wait
        # No prologue is left at the loop; its first iteration's wait depth
        # is unchanged.
        # (The K loop's static trip here is 1: only the first stage is
        # issued -- two bulk copies, or four 16-byte packets per thread of
        # the [64, 64] tile at 128 threads; later stages commit empty.)
        assert prologue.count(issue) == (2 if loads == "tma" else 4)
        loop = _loop_body(body, "md_i")
        assert f"cute.arch.cp_async_wait_group({max(stages - 2, 0)})" in loop
    # A root that starts with its loop keeps the prologue at the loop and the
    # plain drain.
    a = torch.empty((256, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((256, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_causal_plain, (a, b))
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    assert "cute.arch.cp_async_wait_group(1)" not in body
    assert body.index(f"{RT}.cp_async_16(md_t0") < body.index(
        "for md_i in cutlass.range("
    )


def _chunk_scan_s2_inputs() -> tuple[torch.Tensor, ...]:
    # The catalogue's s2 shape (chunk 128, 16 heads, 4 groups, seq 2048).
    bf16 = torch.bfloat16
    batch, heads, groups, seq, chunk, d, ds = 2, 16, 4, 2048, 128, 64, 128
    chunks = seq // chunk
    empty = functools.partial(torch.empty, dtype=bf16, device=CPU_DEVICE)
    return (
        empty((batch, chunks, groups, chunk, chunk)),
        empty((batch, seq, heads, d)),
        empty((batch, heads, chunks, chunk)),
        empty((batch, heads, chunks, chunk)),
        empty((batch, seq, groups, ds)),
        empty((batch, chunks, heads, d, ds)),
        empty((heads,)),
    )


def test_region_programs_seed_the_far_corner_of_the_tile_warp_stage_box() -> None:
    # chunk_scan at the s2 shape: the ranked tensor-memory seeds are 64-row
    # tiles at four warps and two stages with the smallest column tiles (the
    # ranking's CTA-count prior); the measured winner is [128, 64, 64] at
    # eight warps and one stage, two knob moves away with both single-knob
    # neighbours slower.  The heuristic now seeds, per fragment-row tile,
    # the largest column tile at the most warps and the fewest stages too.
    fn = importlib.import_module(
        "examples.mamba2_chunk_scan"
    ).helion_mamba2_chunk_scan_kernel.fn
    bound = _bind_fn(fn, _chunk_scan_s2_inputs())
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None and facts.tcgen05.regions
    ranked = tcgen05_seed_configs(facts)
    top = ranked[: CuteMmaDagHeuristic.SEED_TCGEN05]
    corners = tcgen05_corner_seeds(facts, ranked, top)
    forms = [
        (sizes[0], sizes[1], sizes[2], warps, stages)
        for sizes, warps, stages in corners
    ]
    # (A 128-deep K tile at 64 rows is a partial trip for the odd row tiles:
    # not a tcgen05 config, so the 64-row corner keeps K = 64.  The corners
    # follow their row tiles' ranked order.)
    assert sorted(forms) == [(64, 64, 64, 8, 1), (128, 64, 128, 8, 1)]
    seeds = [
        (
            tuple(c.config["block_sizes"]),
            c.config["cute_mma_dag_warps"],
            c.config["cute_mma_dag_stages"],
        )
        for c in bound.config_spec.compiler_seed_configs
        if c.config.get("cute_mma_dag") == "tcgen05"
    ]
    # (The ranked seeds in their order; the top seed's split / lookahead /
    # persistent twins sit between them.)
    remaining = iter(seeds)
    assert all(
        (tuple(sizes[i] for i in range(3)), warps, stages) in remaining
        for sizes, warps, stages in top
    )
    assert ((64, 64, 64), 8, 1) in seeds and ((128, 64, 128), 8, 1) in seeds
    # Outside the region programs the seed list is the ranked one.
    fwd_o = _bind_fwd_o(True, bhn=2048, d=128, dv=128).config_spec
    ofacts = fwd_o.cute_mma_dag_facts
    assert ofacts is not None and ofacts.tcgen05 is not None
    assert not ofacts.tcgen05.regions
    assert tcgen05_corner_seeds(ofacts, tcgen05_seed_configs(ofacts), []) == []


def test_the_top_warp_seed_comes_again_with_the_one_cta_occupancy_hint() -> None:
    # chunk_scan at s1: the same warp tile / warps / stages ran 4.22 us at
    # the default occupancy hint and 3.95 with ``cute_min_blocks_per_mp``
    # 1 (ptxas budgets registers for one CTA per SM instead of the CTAs the
    # shared-memory footprint admits; a single-wave grid loses nothing).
    # The heuristic seeds that twin right behind the top warp seed (behind
    # the aux-ring twin when there is one), so the search starts from it.
    bound = _bind_chunk_scan()
    seeds = [c.config for c in bound.config_spec.compiler_seed_configs]
    top = next(i for i, c in enumerate(seeds) if c.get("cute_mma_dag") == "warp")
    assert seeds[top].get("cute_min_blocks_per_mp", 0) == 0
    # The top warp seed's twins (``_twin_knobs``: the aux ring, the L2
    # hint where the hinted packets can be verified, the occupancy hint)
    # follow it, each the seed with that one knob flipped.
    twins = []
    while top + 1 + len(twins) < len(seeds) and _twin_knobs(
        seeds[top + 1 + len(twins)]
    ):
        twins.append(seeds[top + 1 + len(twins)])
    hinted = [c for c in twins if c.get("cute_min_blocks_per_mp") == 1]
    assert len(hinted) == 1
    assert {k: v for k, v in hinted[0].items() if k != "cute_min_blocks_per_mp"} == {
        k: v for k, v in seeds[top].items() if k != "cute_min_blocks_per_mp"
    }
    # The aux-ring twin of chunk_scan (side loads of the K loop) is another.
    assert [c.get("cute_mma_dag_aux") for c in twins].count("ring") == 1
    assert len({frozenset(_twin_knobs(c)) for c in twins}) == len(twins)
    # The hint is a searched knob of these kernels (the twin normalizes
    # unchanged).
    with bound.env:
        normalized = bound.config_spec.normalized_config(helion.Config(**hinted[0]))
    assert normalized.config["cute_min_blocks_per_mp"] == 1


def _nested_gated_probe(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """An outer tile loop whose innermost ring loop scales the A operand by
    a per-row gate (a loop-variant side load of the pass) and stores the
    accumulator after it: the early-prologue shape with an aux ring."""
    c, d = k.size()
    dv = v.size(1)
    out = torch.empty([d, dv], dtype=k.dtype, device=k.device)
    for tile_dv in hl.tile(dv):
        for tile_d in hl.tile(d):
            acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
            for tile_c in hl.tile(c):
                kt = (
                    k[tile_c, tile_d].to(torch.float32)
                    * g[tile_c].to(torch.float32)[:, None]
                )
                acc = hl.dot(
                    kt.to(k.dtype).transpose(0, 1), v[tile_c, tile_dv], acc=acc
                )
            out[tile_d, tile_dv] = acc.to(out.dtype)
    return out


def _bind_nested_gated(c: int = 512, d: int = 128, dv: int = 128) -> BoundKernel:
    bf16 = torch.bfloat16
    return _bind_fn(
        _nested_gated_probe,
        (
            torch.empty((c, d), dtype=bf16, device=CPU_DEVICE),
            torch.empty((c, dv), dtype=bf16, device=CPU_DEVICE),
            torch.empty((c,), dtype=bf16, device=CPU_DEVICE),
        ),
    )


def test_the_early_nested_prologue_copies_the_aux_ring_words() -> None:
    # The nested ring's prologue for the outer loop's next iteration is
    # issued in the outer iteration's epilogue (``emit_early_prologue``),
    # after the inner loop's body was rendered and its aux anchors filled.
    # Its own anchors must be filled too, or iteration i+1 starts with the
    # operand tiles in flight and no copy of the gate words: the pass then
    # reads the previous iteration's slot (silently wrong values).
    bound = _bind_nested_gated()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.aux_candidates
    config = _config(64, 64, 64, warps=4, stages=2, cute_mma_dag_aux="ring")
    with bound.env:
        normalized = spec.normalized_config(config)
    plan = _plan_of(bound, normalized)
    assert plan.early_prologues() == {0: 1}
    assert plan.aux_rings[1].words == 4
    body = _kernel_body(_source(bound, normalized))
    outer = _loop_body(body, "md_i")
    early = outer[outer.index("if md_i + 1 < 2:") :]
    inner = _loop_body(body, "md_i1")
    # The peeled prologue ahead of the outer loop, the refill inside the
    # inner loop and the early prologue each copy the four gate words.
    peeled = body[: body.index("for md_i in cutlass.range(")]
    assert peeled.count(f"{RT}.cp_async_4(md_aux1 + ") == 4
    assert inner.count(f"{RT}.cp_async_4(md_aux1 + (md_stage1 + 1) % 2 * 2048 + ") == 4
    assert early.count(f"{RT}.cp_async_4(md_aux1 + ") == 4
    # Their copies precede the commit of the early prologue's group.
    assert early.rindex(f"{RT}.cp_async_4(md_aux1") < early.index(
        "cute.arch.cp_async_commit_group()"
    )
    # A 16-bit side load's word: the low two address bits by a mask, not a
    # signed remainder of the truncated address (negative at bit 31).
    assert body.count("- (cutlass.Int32(md_aw") == 12
    assert (
        len(
            re.findall(
                rf"{RT}\.lds_bf16_as_f32\(md_aux1 \+ .*\) & 3\)\)$", body, re.MULTILINE
            )
        )
        == 4
    )
    assert "% 4" not in body.split("md_aux1 =")[1]


def test_a_partial_trip_zeroes_the_other_operands_tail_in_shared_memory() -> None:
    # chunk_scan at a 32-row tile: the causal K loop's bound (32 at the first
    # row tile) leaves a partial 64-deep trip.  cb's pass masks its elements
    # past the bound; x's tile (direct, K along its rows) is zeroed there in
    # shared memory by the threads owning the packets, after the passes
    # and before the iteration barrier, so an Inf or NaN past the bound
    # cannot reach the accumulator through ``0 * x``.
    bound = _bind_chunk_scan()
    for loads in ("lane", "tma"):
        body = _kernel_body(
            _source(
                bound, _config(32, 64, 64, warps=4, stages=2, cute_mma_dag_loads=loads)
            )
        )
        loop = _loop_body(body, "md_i")
        zeros = re.findall(
            r"^ *if md_a1 \+ (?:\(\d+ \+ )?md_pr8\)? >= md_end0:\n"
            rf" *{re.escape(RT)}\.sts_v4_b32\(md_t3 \+ md_stage \* 8192 \+ md_pr8 \* 128"
            r"(?: \+ \d+)? \+ md_pz8 \* 16, cutlass\.Int32\(0\), cutlass\.Int32\(0\), "
            r"cutlass\.Int32\(0\), cutlass\.Int32\(0\)\)$",
            loop,
            re.MULTILINE,
        )
        assert len(zeros) == 4, loads
        # After the pass (its shared reads issue first: zero stores ahead of
        # them cost chunk_scan s0 7 %), before the iteration barrier.
        first_zero = loop.index(">= md_end0:")
        pass_read = loop.index(f"{RT}.lds_v4_b32(md_pa")
        assert pass_read < first_zero < loop.index("cute.arch.barrier()", pass_read)
    # An aligned bound (64-row tiles: every CTA's bound a multiple of the K
    # block) needs none of it.
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    assert ">= md_end0:" not in body and "sts_v4_b32(md_t3" not in body


def _causal_shifted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A causal K bound whose constant term mixes two block sizes
    (``2 * block_m - block_n``): positive at unit block sizes, not at every
    real pair."""
    m, k = a.size()
    _, n = b.size()
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(64, 64)
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[block_m, block_n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile((tile_m.id + 2) * block_m - block_n, block_size=block_k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def test_a_varying_bound_with_a_mixed_block_coefficient_is_not_admitted() -> None:
    # The nondecreasing proof must hold at every block size, not at 1: a
    # coefficient (or constant) that is not a single nonnegative monomial
    # in the block sizes is refused (``2 * block_m - block_n`` is 1 at unit
    # sizes and negative at block_n > 2 * block_m).
    a = torch.empty((256, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((256, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    assert _bind_fn(_causal_plain, (a, b)).config_spec.cute_mma_dag is not None
    assert _bind_fn(_causal_shifted, (a, b)).config_spec.cute_mma_dag is None


def _rmw_view(k: torch.Tensor, v: torch.Tensor, s_out: torch.Tensor) -> torch.Tensor:
    """Stores through an in-kernel view of the tensor it side-loads."""
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)
    s_alias = s_out.view(BH, D)
    h_out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            s_alias[idx, tile_d] = s_out[idx, tile_d] + h_acc.sum(-1)
        h_out[idx, tile_d, tile_dv] = h_acc
    return h_out


def _store_between(k: torch.Tensor, dL: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Region-level loads of x before and after a store to x in the loop."""
    BHN = k.size(0)
    C = hl.specialize(k.size(1))
    D = k.size(2)
    out = torch.empty([BHN, C, D], dtype=torch.float32, device=k.device)
    for tile_bhn in hl.tile(BHN, block_size=1):
        for tile_d in hl.tile(D):
            k_d = k[tile_bhn, :, tile_d]
            a = x[tile_bhn, :, tile_d]
            acc = hl.dot(dL[tile_bhn, :, :], k_d)
            out[tile_bhn, :, tile_d] = acc + a.float()
            x[tile_bhn, :, tile_d] = acc.to(x.dtype)
            b = x[tile_bhn, :, tile_d]
            acc2 = hl.dot(dL[tile_bhn, :, :], b)
            out[tile_bhn, :, tile_d] = acc2
    return out


def _div_scalar(
    scale: float, k: torch.Tensor, v: torch.Tensor, s_out: torch.Tensor
) -> torch.Tensor:
    """Divides a row sum by a float argument: the sign of a zero matters."""
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)
    h_out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
        s_out[idx, tile_d] = h_acc.sum(-1) / scale
        h_out[idx, tile_d, tile_dv] = h_acc
    return h_out


def _rmw_two_names(
    k: torch.Tensor, v: torch.Tensor, s_in: torch.Tensor, s_out: torch.Tensor
) -> torch.Tensor:
    """Reads s_in, writes s_out: distinct storages at compile time, so the
    s_in reads are hoisted -- the caller may still pass one tensor twice
    (the matcher's fakes have one storage per argument, so two argument
    names bound to one tensor are invisible to it; the launcher refuses
    them)."""
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)
    h_out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
            s_out[idx, tile_d] = s_in[idx, tile_d] + h_acc.sum(-1)
        h_out[idx, tile_d, tile_dv] = h_acc
    return h_out


def test_a_program_that_loads_and_stores_one_storage_is_not_mine() -> None:
    # A tile or side load of a tensor the loop stores to is staged ahead of the
    # store, and a store through a view (or another name) of the tensor passed
    # the name checks.  A load that shares storage with any store of the
    # program declines the whole lowering (both modes); two names with distinct
    # storages are admitted (the lowering cannot see a caller passing one
    # tensor twice: the launcher guards that).
    t = _tensors()
    s = torch.empty((2, 64), dtype=torch.float32, device=CPU_DEVICE)
    for fn in (_rmw_vec_store, _rmw_view):
        bound = _bind_fn(fn, (t["k"], t["v"], s))
        assert bound.config_spec.cute_mma_dag is None, fn.__name__
    t3 = _tensors3()
    x = torch.empty((16, 64, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    dl = torch.empty((16, 64, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_store_between, (t3["q"], dl, x))
    assert bound.config_spec.cute_mma_dag is None
    # Two names with distinct storages are admitted and the s_in reads
    # hoisted: the lowering cannot see a caller passing one tensor twice,
    # so the host source names the program's tensor arguments and which
    # it stores to for the launcher's alias guard.
    s_in = torch.empty((2, 64), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind_fn(_rmw_two_names, (t["k"], t["v"], s_in, s))
    assert bound.config_spec.cute_mma_dag is not None
    source = _source(bound, _config(64, 64, warps=4, stages=2))
    assert "md_sl" in _kernel_body(source)
    assert (
        "_helion_cute_wrapper_plans = [{'kind': 'helion_mma_dag_alias', "
        "'tensor_names': ['k', 'v', 'h_out', 's_in', 's_out'], "
        "'stored': [False, False, True, False, True], "
        "'tensor_idx': [0, 1, 2, 3, 4]}]"
    ) in source


def test_the_alias_guard_refuses_a_stored_argument_that_overlaps_another() -> None:
    # The launcher's guard over the plan above: byte spans of the stored
    # arguments against every other tensor argument.  Pointers are the only
    # per-launch input, so CPU tensors exercise it.
    plan = {
        "kind": "helion_mma_dag_alias",
        "tensor_names": ["k", "v", "h_out", "s_in", "s_out"],
        "stored": [False, False, True, False, True],
        "tensor_idx": [0, 1, 2, 3, 4],
    }
    k = torch.empty((2, 8, 64, 64), dtype=torch.bfloat16)
    v = torch.empty((2, 8, 64, 64), dtype=torch.bfloat16)
    h = torch.empty((2, 64, 64), dtype=torch.float32)
    s = torch.empty((2, 64), dtype=torch.float32)
    s2 = torch.empty((2, 64), dtype=torch.float32)
    guard = _MmaDagAliasGuard(plan, (k, v, h, s, s2))
    guard.check((k, v, h, s, s2))
    with pytest.raises(BackendUnsupported, match="'s_in' and 's_out' alias"):
        guard.check((k, v, h, s, s))
    # A view of a stored tensor under another name, and two stored names
    # sharing storage, alias as well.
    with pytest.raises(BackendUnsupported, match="'s_in' and 's_out' alias"):
        guard.check((k, v, h, s2[:, 32:], s2))
    with pytest.raises(BackendUnsupported, match="'h_out' and 's_out' alias"):
        guard.check((k, v, h, s, h.view(-1)[:128].view(2, 64)))
    # Two loads of one tensor are fine (the engine passes k twice to dv),
    # as are disjoint slices of one buffer and interleaved loads.
    guard.check((k, k, h, s, s2))
    wide = torch.empty((4, 64), dtype=torch.float32)
    guard.check((k, v, h, wide[:2], wide[2:]))
    strided = torch.empty((2, 8, 64, 128), dtype=torch.bfloat16)
    guard.check((strided[..., :64], strided[..., 64:], h, s, s2))
    # Spans follow the view: a strided stored view whose span covers a
    # loaded tensor's bytes is refused even where no element coincides.
    both = torch.empty((2, 128), dtype=torch.float32)
    with pytest.raises(BackendUnsupported, match="'s_in' and 's_out' alias"):
        guard.check((k, v, h, both[:, 1::2], both[:, 0::2]))
    # Empty tensors never alias; the pointers of empty allocations collide.
    empty = torch.empty((0, 64), dtype=torch.float32)
    guard0 = _MmaDagAliasGuard(plan, (k, v, h, empty, empty))
    guard0.check((k, v, h, empty, empty))
    assert guard0.indices == (0, 1, 2)


# --------------------------------------------------------------------------
# Element pairs render fp32 mul / add / sub as packed f32x2 ops
# --------------------------------------------------------------------------


def _packed_lines(body: str) -> list[str]:
    return re.findall(r"cute\.arch\.(\w+)_packed_f32x2\(", body)


def test_element_pairs_render_fp32_products_as_packed_f32x2_ops() -> None:
    # wy_dL with the pipeline's dtypes: betaV's pass (beta[row] * v[row, 2q..])
    # and betaK's render their products over the packet's element pairs as
    # one mul_packed_f32x2 each (the row factor on both lanes), as do the
    # dv_out / d_ba_out / dL stores and the chain's derived values; the
    # products feeding the row reductions (dbeta += (d_bv * v).sum(-1))
    # stay scalar -- the compiler may fuse those into FMAs, which a packed
    # product would not match bit for bit.
    bf16 = torch.bfloat16
    bound = _bind_wy_dl(beta_dtype=bf16, dvni_dtype=bf16, bhn=2048)
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    packed = _packed_lines(body)
    assert len(packed) == 80 and set(packed) == {"mul"}
    first = body[: body.index("for md_i1 in")]
    # The DV loop: four packet rounds x four words of betaV, two fragment
    # pairs x eight atoms of the dv_out store... with no scalar product left
    # in the pass.
    assert first.count("mul_packed_f32x2((md_sl, md_sl)") == 4
    assert not re.search(r"^\s*v_\d+ = md_sl \* md_lo_\d+$", first, re.MULTILINE)
    # Reductions keep their scalar products (the FMA rule).
    assert re.search(
        r"^\s*v_\d+ = md_c\d+_\d+_\d+_\d+ \* md_lo_\d+$", body, re.MULTILINE
    )
    # dqkw: do * scale (a float argument on both lanes) and the gated
    # decays' differences pack; nothing scalar remains.
    dqkw = _kernel_body(
        _source(_bind_dqkw(bhn=2048), _config(64, 32, 64, warps=4, stages=2))
    )
    assert sorted(_packed_lines(dqkw)) == ["mul"] * 32 + ["sub"] * 8
    assert "cutlass.Float32(scale), cutlass.Float32(scale)))" in dqkw
    assert not re.search(r"^\s*v_\d+ = [\w.]+ \* [\w.]+$", dqkw, re.MULTILINE)
    # dk's accumulator seed dk_as + d_ba * beta is a product feeding a sum:
    # the FMA pattern stays scalar, and the tcgen05 renderer never pairs.
    dk = _kernel_body(_source(_bind_dk(bhn=2048), _config(64, warps=4, stages=2)))
    assert not _packed_lines(dk)
    assert re.search(r"^\s*v_\d+ = md_(?:lo|hi)_\d+ \* md_sl\w*$", dk, re.MULTILINE)


# --------------------------------------------------------------------------
# One lane base per address family, the coordinates' immediates folded
# --------------------------------------------------------------------------


def test_addresses_share_one_lane_base_and_fold_the_immediates() -> None:
    # wy_dL with the pipeline's dtypes: the stores of a fragment (rows g /
    # g + 8, columns 2c + 8an) address ``lane base + immediate``, the lane
    # base (iteration base + row and column lane terms) bound once per
    # iteration; a loop-invariant base (beta at the thread's packet row, at
    # the fragment row) is bound once ahead of the loops and every round /
    # half adds its immediate.  The tcgen05 renderer keeps spelling the
    # terms out.
    bf16 = torch.bfloat16
    bound = _bind_wy_dl(beta_dtype=bf16, dvni_dtype=bf16, bhn=2048)
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    first = body.index("for md_i in")
    assert "md_lb = md_base_2 + cutlass.Int64(md_pr8) * 2\n" in body[:first]
    assert "md_lb_1 = md_base_2 + cutlass.Int64(md_L0_rowl) * 2\n" in body[:first]
    stores = re.findall(rf"{RT}\.stg_v2_f32\((md_lb\w*)( \+ \d+)?,", body)
    assert len(stores) == 16 and {name for name, _imm in stores} == {"md_lb_3"}
    # d_ba_out: one base per D iteration (rows g / g + 8 x 8 atoms), the 16
    # registers' offsets 0, 4096 (row + 8), 32, 4128, ...
    d_ba = [imm for name, imm in stores if name == "md_lb_3"]
    assert d_ba[:4] == ["", " + 4096", " + 32", " + 4128"]
    assert re.search(
        r"^        md_lb_3 = md_p_\d+ \+ cutlass\.Int64\(md_L0_rowl\) \* 512 "
        r"\+ cutlass\.Int64\(md_L0_coll\) \* 4$",
        body,
        re.MULTILINE,
    )
    # Every address is a lane base plus an immediate: no coordinate is
    # widened to Int64 inside the loops.
    loops = body[first:]
    assert "cutlass.Int64(md_L0_rowl + 8)" not in loops
    assert "cutlass.Int64(md_L0_coll + " not in loops
    assert "cutlass.Int64(16 + md_pr8)" not in body
    # The pass's side loads read the hoisted base with the round's offset.
    assert re.search(
        rf"{RT}\.ldg_bf16_as_f32\(md_lb \+ 32(?:, nc=True)?\)", body[:first]
    )
    assert _split_coord_cases()


def _split_coord_cases() -> bool:
    from helion._compiler.cute.cute_mma_dag_codegen import _split_coord

    assert _split_coord("md_pr8") == ("md_pr8", 0)
    assert _split_coord("(16 + md_pr8)") == ("md_pr8", 16)
    assert _split_coord("(md_L0_coll + 57)") == ("md_L0_coll", 57)
    assert _split_coord("md_L0_rowl + 8") == ("md_L0_rowl", 8)
    assert _split_coord("5") == (None, 5)
    assert _split_coord("(md_pc8 * 8 + 3)") == ("(md_pc8 * 8 + 3)", 0)
    return True


# --------------------------------------------------------------------------
# The nested ring's prologue for the next iteration goes out during this
# iteration's epilogue
# --------------------------------------------------------------------------


def _plan_of(bound: BoundKernel, config: helion.Config) -> dag.MmaDagPlan:
    """The warp plan the render of ``config`` is emitted from."""
    captured: list[dag.MmaDagPlan] = []
    original = dag.MmaDagPlan.smem_layout

    def wrapped(self: dag.MmaDagPlan) -> object:
        captured.append(self)
        return original(self)

    with patch.object(dag.MmaDagPlan, "smem_layout", wrapped):
        _source(bound, config)
    # The last plan laid out is the one rendered (an aux ring replaces the
    # plan after its tiles are laid out).
    return captured[-1]


def test_the_next_iterations_nested_prologue_overlaps_the_epilogue() -> None:
    # dqkw's tile_d loop nests the DV loop and follows it with the dq /
    # dk_as dots and stores: iteration i's epilogue issues the nested ring's
    # first stages - 1 tiles of iteration i + 1 (behind the barrier every
    # lane passed after the nested loop, so the frame is dead), and the
    # nested loop's own prologue is gone (the first iteration's is peeled
    # ahead of the outer ``for``).  wy_dL's loops are siblings: nothing.
    dqkw = _plan_of(_bind_dqkw(bhn=2048), _config(64, 32, 64, warps=4, stages=2))
    assert dqkw.early_prologues() == {1: 2}
    wy_dl = _plan_of(
        _bind_wy_dl(beta_dtype=torch.bfloat16, dvni_dtype=torch.bfloat16, bhn=2048),
        _config(64, 64, 64, warps=4, stages=2),
    )
    assert wy_dl.early_prologues() == {}
    # One stage: no ring prologue to move.
    single = _plan_of(_bind_dqkw(bhn=2048), _config(64, 32, 64, warps=4, stages=1))
    assert single.early_prologues() == {}
    # Four stages: three groups per prologue, the epilogue issues all three.
    body = _kernel_body(
        _source(_bind_dqkw(bhn=2048), _config(64, 32, 64, warps=8, stages=4))
    )
    outer = body.index("for md_i1 in")
    inner = body.index("for md_i2 in", outer)
    assert f"{RT}.cp_async_16(md_t2" not in body[outer:inner]
    early = body.index("if md_i1 + 1 < 4:", inner)
    tail = body[early : body.index("stg_v2_f32", early)]
    assert tail.count("cute.arch.cp_async_commit_group()") == 3
    assert "cutlass.Int64(md_i1 + 1) * " in tail
    assert body[:outer].count("cute.arch.cp_async_commit_group()") >= 3


def _pack_probe(
    k: torch.Tensor,
    v: torch.Tensor,
    sr: torch.Tensor,
    sc: torch.Tensor,
    s2: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    """A state recurrence with a root epilogue exercising the packed pairs:
    row / column vectors, a product viewed before an add, sub then mul, a
    direct FMA."""
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)
    h_out = torch.empty([BH, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        h_acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
        for i_t in hl.grid(N):
            k_i = k[idx, i_t, :, tile_d]
            v_i = v[idx, i_t, :, tile_dv]
            h_acc = hl.dot(k_i.transpose(-2, -1), v_i, acc=h_acc)
        r = sr[idx, tile_d]
        c = sc[idx, tile_dv]
        if mode == 0:  # per-row vector: the same value on both lanes
            h_out[idx, tile_d, tile_dv] = h_acc * r[:, None]
        elif mode == 1:  # per-column vector: the two lanes differ
            h_out[idx, tile_d, tile_dv] = h_acc * c[None, :]
        elif mode == 2:  # a product viewed, then added (an FMA through a view)
            p = r * s2[idx, tile_d]
            h_out[idx, tile_d, tile_dv] = h_acc + p[:, None]
        elif mode == 3:  # sub then mul (both packable)
            h_out[idx, tile_d, tile_dv] = (h_acc - r[:, None]) * c[None, :]
        else:  # a product added directly (an FMA: stays scalar)
            h_out[idx, tile_d, tile_dv] = h_acc * c[None, :] + r[:, None]
    return h_out


def test_the_fma_refusal_looks_through_views() -> None:
    # ``p = r * s2`` viewed as ``p[:, None]`` and added to the state rendered
    # both the product and the sum packed, while the scalar lowering fuses them
    # into one FMA (1468 of 8192 outputs off by up to 9.5e-7).  The user walk
    # and the input strip look through views now: mode 2 keeps the sum (and the
    # product feeding it) scalar, the direct FMA (mode 4) stays scalar,
    # sub-then-mul (mode 3) packs both.
    t = _tensors()
    f32 = torch.float32
    sr = torch.empty((2, 64), dtype=f32, device=CPU_DEVICE)
    sc = torch.empty((2, 64), dtype=f32, device=CPU_DEVICE)
    renders = {}
    for mode in range(5):
        bound = _bind_fn(_pack_probe, (t["k"], t["v"], sr, sc, sr, mode))
        assert bound.config_spec.cute_mma_dag is not None, mode
        renders[mode] = _packed_lines(
            _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
        )
    assert renders[0] and set(renders[0]) == {"mul"}
    assert renders[1] and set(renders[1]) == {"mul"}
    assert "add" not in renders[2] and "mul" not in renders[2]
    assert set(renders[3]) == {"sub", "mul"}
    assert not renders[4]


# --------------------------------------------------------------------------
# The peeled prologue and a loop of no iterations, loop bounds naming a loop
# index, the standalone export's alias check
# --------------------------------------------------------------------------


def _grid_nest(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """An outer grid loop nesting a tile ring loop and followed by a store
    (the early prologue applies); the outer trip count may be zero."""
    BH = a.size(0)
    N = a.size(1)
    C = hl.specialize(a.size(2))
    K = a.size(3)
    V = b.size(3)
    out = torch.zeros([BH, N, C, V], dtype=torch.float32, device=a.device)
    for tile_bh, tile_v in hl.tile([BH, V], block_size=[1, None]):
        idx = tile_bh.id
        for i in hl.grid(N):
            acc = hl.zeros([C, tile_v], dtype=torch.float32)
            for tile_k in hl.tile(K):
                acc = hl.dot(a[idx, i, :, tile_k], b[idx, i, tile_k, tile_v], acc=acc)
            out[idx, i, :, tile_v] = acc * 2.0
    return out


def _grid_nest_steps(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """``_grid_nest`` over the first ``n`` positions: a dynamic outer trip
    count under static shapes."""
    BH = a.size(0)
    N = a.size(1)
    C = hl.specialize(a.size(2))
    K = a.size(3)
    V = b.size(3)
    out = torch.zeros([BH, N, C, V], dtype=torch.float32, device=a.device)
    for tile_bh, tile_v in hl.tile([BH, V], block_size=[1, None]):
        idx = tile_bh.id
        for i in hl.grid(n):
            acc = hl.zeros([C, tile_v], dtype=torch.float32)
            for tile_k in hl.tile(K):
                acc = hl.dot(a[idx, i, :, tile_k], b[idx, i, tile_k, tile_v], acc=acc)
            out[idx, i, :, tile_v] = acc * 2.0
    return out


def _causal_nest(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A nested grid loop whose bound names the outer loop's index."""
    BH = k.size(0)
    N = k.size(1)
    D = k.size(3)
    DV = v.size(3)
    out = torch.empty([BH, N, D, DV], dtype=torch.float32, device=k.device)
    for tile_bh, tile_d, tile_dv in hl.tile([BH, D, DV], block_size=[1, None, None]):
        idx = tile_bh.id
        for i in hl.grid(N):
            acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
            for j in hl.grid(i + 1):
                k_j = k[idx, j, :, tile_d]
                acc = hl.dot(k_j.transpose(-2, -1), v[idx, j, :, tile_dv], acc=acc)
            out[idx, i, tile_d, tile_dv] = acc
    return out


def _nest_tensors(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    bf16 = torch.bfloat16
    return (
        torch.empty((2, n, 64, 128), dtype=bf16, device=CPU_DEVICE),
        torch.empty((2, n, 128, 64), dtype=bf16, device=CPU_DEVICE),
    )


def test_the_peeled_prologue_follows_the_outer_trip_count() -> None:
    # The first nested prologue, peeled ahead of the outer ``for``, read
    # iteration 0's ring tiles unconditionally -- of empty tensors (pointer 0)
    # when the outer loop had no iterations.  A static zero trip count issues
    # nothing, a positive one the plain peel, and a dynamic one (an int
    # argument under static shapes) guards it: the commits sit inside the guard
    # and no wait precedes the loop.
    config = _config(64, 32, warps=4, stages=3)
    body = _kernel_body(_source(_bind_fn(_grid_nest, _nest_tensors(0)), config))
    outer = body.index("for md_i in cutlass.range(0, 0):")
    assert f"{RT}.cp_async_16(" not in body[:outer]
    assert "cp_async_commit_group" not in body[:outer]
    body = _kernel_body(_source(_bind_fn(_grid_nest, _nest_tensors(3)), config))
    outer = body.index("for md_i in cutlass.range(0, 3):")
    assert body[:outer].count(f"{RT}.cp_async_16(") == 8
    assert body[:outer].count("cute.arch.cp_async_commit_group()") == 2
    assert "if 0 < 3:" not in body
    bound = _bind_fn(_grid_nest_steps, (*_nest_tensors(3), 3))
    body = _kernel_body(_source(bound, config))
    outer = body.index("for md_i in cutlass.range(0, n):")
    guard = body.index("\n    if 0 < n:\n")
    assert guard < outer
    peel = body[guard:outer]
    assert peel.count(f"{RT}.cp_async_16(") == 8
    assert peel.count("cute.arch.cp_async_commit_group()") == 2
    # Every peeled line sits inside the guard.
    lines = [line for line in peel.splitlines()[2:] if line.strip()]
    assert lines and all(line.startswith("        ") for line in lines)
    # The epilogue's prologue for the next iteration keeps its own guard.
    assert "if md_i + 1 < n:" in body[outer:]


def test_a_nested_bound_naming_the_outer_index_is_not_mine() -> None:
    # ``hl.grid(i + 1)`` inside the outer grid loop was admitted and crashed
    # the emitter (a bound renders with the host's symbols, which know no loop
    # index); with the early prologue it would also have guarded iteration i +
    # 1 with iteration i's bound.  The matcher declines it (the ordinary
    # lowering runs) and the planner refuses the early prologue for a nested
    # loop whose bounds name any loop's index.
    t = _tensors()
    bound = _bind_fn(_causal_nest, (t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is None
    config = _config(64, 32, warps=4, stages=3)
    plan = _plan_of(_bind_fn(_grid_nest, _nest_tensors(3)), config)
    assert plan.early_prologues() == {0: 1}
    outer, inner = plan.program.loops
    assert outer.symbol is not None
    end = torch.fx.Graph().placeholder("end")
    end.meta["val"] = ShapeEnv().create_symintnode(outer.symbol + 1, hint=4)
    dependent = dataclasses.replace(inner, end=end)
    outer = dataclasses.replace(
        outer,
        steps=tuple(dependent if step is inner else step for step in outer.steps),
    )
    program = dataclasses.replace(plan.program, loops=(outer, dependent))
    assert dataclasses.replace(plan, program=program).early_prologues() == {}
    # A symbol that is no host value varies, whatever its origin -- one minted
    # in device code (``int or int`` on a loop index) as much as a loop index;
    # a host argument (the int trip count) does not.
    a, b = _nest_tensors(3)
    bound = _bind_fn(_grid_nest_steps, (a, b, 3))
    host = bound.host_function
    with bound.env, host:
        host_symbol = next(
            symbol
            for symbol, info in host.expr_to_origin.items()
            if info.origin.is_host()
        )
        device_symbol = sympy.Symbol("u_device")
        host.expr_to_origin[device_symbol] = SymbolOrigin(BlockSizeOrigin(0))
        matcher = dag._Matcher.__new__(dag._Matcher)
        shape_env = ShapeEnv()
        assert matcher.bound_varies(
            shape_env.create_symintnode(device_symbol + 1, hint=4)
        )
        assert not matcher.bound_varies(
            shape_env.create_symintnode(host_symbol + 1, hint=4)
        )


def test_the_standalone_launcher_runs_the_alias_check() -> None:
    # The standalone export allowlisted the alias plan as launcher-only
    # metadata and the exported wrapper ran with no check.  The launch table
    # now carries the plan's positions and store flags and the exported
    # launcher checks the spans like ``_MmaDagAliasGuard``.
    indices = (0, 1, 2, 3, 4)
    stored = (False, False, True, False, True)
    k = torch.empty((2, 8, 64, 64), dtype=torch.bfloat16)
    v = torch.empty((2, 8, 64, 64), dtype=torch.bfloat16)
    h = torch.empty((2, 64, 64), dtype=torch.float32)
    s = torch.empty((2, 64), dtype=torch.float32)
    s2 = torch.empty((2, 64), dtype=torch.float32)
    _standalone_validate_mma_dag_aliases((k, v, h, s, s2), indices, stored)
    with pytest.raises(ValueError, match="alias"):
        _standalone_validate_mma_dag_aliases((k, v, h, s, s), indices, stored)
    with pytest.raises(ValueError, match="alias"):
        _standalone_validate_mma_dag_aliases((k, v, h, s2[:, 32:], s2), indices, stored)
    with pytest.raises(ValueError, match="alias"):
        h_view = h.view(-1)[:128].view(2, 64)
        _standalone_validate_mma_dag_aliases((k, v, h, s, h_view), indices, stored)
    # Two loads of one tensor, disjoint slices of one buffer and empty
    # tensors pass; a strided stored view's span covers the other's bytes.
    _standalone_validate_mma_dag_aliases((k, k, h, s, s2), indices, stored)
    wide = torch.empty((4, 64), dtype=torch.float32)
    _standalone_validate_mma_dag_aliases((k, v, h, wide[:2], wide[2:]), indices, stored)
    empty = torch.empty((0, 64), dtype=torch.float32)
    _standalone_validate_mma_dag_aliases((k, v, h, empty, empty), indices, stored)
    both = torch.empty((2, 128), dtype=torch.float32)
    with pytest.raises(ValueError, match="alias"):
        _standalone_validate_mma_dag_aliases(
            (k, v, h, both[:, 1::2], both[:, 0::2]), indices, stored
        )
    with pytest.raises(ValueError, match="tensor layout"):
        _standalone_validate_mma_dag_aliases((k, v, h, s, 1.0), indices, stored)


def _seed_knobs(config: helion.Config) -> tuple[list[int], str, int, int, str]:
    c = config.config
    return (
        c["block_sizes"],
        c["cute_mma_dag"],
        c["cute_mma_dag_warps"],
        c["cute_mma_dag_stages"],
        c.get("cute_mma_dag_exchange", "auto"),
    )


def _seed_knobs_of(c: dict[str, object]) -> tuple[list[int], str, int, int, str]:
    return _seed_knobs(helion.Config.from_dict(c))


def _twin_knobs(c: dict[str, object]) -> dict[str, object]:
    """The knobs by which a seed is one of the first warp seed's twins: the
    L2-hinted staged slots, the aux ring, the one-CTA occupancy hint."""
    twin: dict[str, object] = {}
    if c.get("load_eviction_policies") and any(
        cast("list[str]", c["load_eviction_policies"])
    ):
        twin["load_eviction_policies"] = tuple(
            cast("list[str]", c["load_eviction_policies"])
        )
    if c.get("cute_mma_dag_aux") == "ring":
        twin["cute_mma_dag_aux"] = "ring"
    if c.get("cute_min_blocks_per_mp") == 1:
        twin["cute_min_blocks_per_mp"] = 1
    return twin


def _unhinted_seed_knobs(
    spec: ConfigSpec,
) -> list[tuple[list[int], str, int, int, str]]:
    """The seeds' knobs without the first warp seed's twins (the L2-hinted
    one -- see ``test_the_first_warp_seed_has_an_l2_evict_last_twin`` --,
    the aux-ring and the occupancy-hint twins)."""
    return [
        _seed_knobs(c) for c in spec.compiler_seed_configs if not _twin_knobs(c.config)
    ]


def test_cta_rich_programs_seed_the_top_tiles_measured_neighbours() -> None:
    # The cold autotune of the gated delta rule at s1 (seed 31337) left the
    # ranked ladder for: dqkw at three stages, wy_dL at the full kk tile
    # with the shared-memory exchange, dk at the 32 tile with two warps and
    # one stage.  Those neighbours of the top warp seed follow the ladder
    # (the first seed stays), each measured against the default; see
    # ``mma_dag_seed_variants``.
    bf16 = torch.bfloat16

    # Gated dqkw's dattn is a hoisted root exchange: the streaming class
    # leads with the four tensor-memory seeds, then the warp ladder.
    # (The tensor-memory seeds carry their persistent twins at a CTA-rich
    # grid, so the warp ladder starts where the first warp seed is.)
    def warp_start(seeds: list[tuple[list[int], str, int, int, str]]) -> int:
        start = next(i for i, s in enumerate(seeds) if s[1] == "warp")
        assert start >= 4
        assert all(mode == "tcgen05" for _s, mode, _w, _st, _e in seeds[:start])
        return start

    dqkw = _bind_dqkw(True, bhn=2048, d=128, dv=128).config_spec
    seeds = _unhinted_seed_knobs(dqkw)
    w = warp_start(seeds)
    top = ([64, 32, 64, 64, 64], "warp", 4, 2, "auto")
    assert seeds[w] == top
    # The top warp seed's twins sit right behind it in the full list, each
    # the same tile with one knob flipped (the L2 hint, the aux ring, the
    # occupancy hint: see ``_twin_knobs``), the L2-hinted one among them.
    full = [c.config for c in dqkw.compiler_seed_configs]
    at = full.index(
        next(c for c in full if _seed_knobs_of(c) == top and not _twin_knobs(c))
    )
    twins = []
    while at + 1 + len(twins) < len(full) and _twin_knobs(full[at + 1 + len(twins)]):
        twins.append(full[at + 1 + len(twins)])
    assert twins and all(_seed_knobs_of(c) == top for c in twins)
    assert sum(bool(c.get("load_eviction_policies")) for c in twins) == 1
    assert len({frozenset(_twin_knobs(c)) for c in twins}) == len(twins)
    assert seeds[w + 8 : w + 13] == [
        ([64, 32, 64, 64, 64], "warp", 4, 3, "auto"),
        ([64, 32, 64, 64, 64], "warp", 4, 3, "smem"),
        ([64, 32, 64, 64, 64], "warp", 4, 2, "smem"),
        # Of the two K loops of extent 128 only the dattn loop keeps two
        # CTAs per SM at its full extent (the nested DV loop's ring would
        # take 148 KB); the gated [D, DV] tile contracts nothing.
        ([128, 32, 64, 64, 64], "warp", 4, 2, "auto"),
        ([128, 32, 64, 64, 64], "warp", 4, 2, "smem"),
    ]
    assert seeds[w + 13 :] == []
    # The flat streaming programs (wy_dL, dk) lead with the four
    # tensor-memory seeds; the warp ladder and its neighbours follow.
    wy_dl = _bind_wy_dl(True, bf16, bf16, bhn=2048, d=128, dv=128).config_spec
    seeds = _unhinted_seed_knobs(wy_dl)
    w = warp_start(seeds)
    assert seeds[w] == ([64, 64, 64], "warp", 4, 2, "auto")
    assert seeds[w + 8 : w + 13] == [
        ([64, 64, 64], "warp", 4, 3, "auto"),
        ([64, 64, 64], "warp", 4, 3, "smem"),
        ([64, 64, 64], "warp", 4, 2, "smem"),
        # The kk loop at its full extent keeps three CTAs per SM; the DV
        # loop at 128 would drop to two.
        ([64, 64, 128], "warp", 4, 2, "auto"),
        ([64, 64, 128], "warp", 4, 2, "smem"),
    ]
    dk = _bind_dk(bhn=2048, d=128, dv=128).config_spec
    seeds = _unhinted_seed_knobs(dk)
    w = warp_start(seeds)
    assert seeds[w] == ([64], "warp", 4, 2, "auto")
    assert seeds[w + 8 : w + 10] == [
        ([64], "warp", 4, 1, "auto"),
        ([32], "warp", 2, 1, "auto"),
    ]
    # No root tile, no CTA-rich grid: the ladder alone (fwd_o tiles its
    # root; the state kernels run a CTA per head).
    fwd_o = _bind_fwd_o(True, bhn=2048, d=128, dv=128).config_spec
    assert all(_seed_knobs(c)[4] == "auto" for c in fwd_o.compiler_seed_configs)
    assert all(
        _seed_knobs(c)[3] in dag.MMA_DAG_SEED_STAGES
        for c in fwd_o.compiler_seed_configs
        if _seed_knobs(c)[1] == "warp"
    )
    facts = fwd_o.cute_mma_dag_facts
    assert facts is not None
    assert dag.mma_dag_seed_variants(facts, dag.mma_dag_seed_configs(facts)) == []
    fwd = _bind_fwd().config_spec
    facts = fwd.cute_mma_dag_facts
    assert facts is not None
    assert dag.mma_dag_seed_variants(facts, dag.mma_dag_seed_configs(facts)) == []


@pytest.mark.parametrize("kernel", ("dqkw", "wy_dL", "dk"), ids=("dqkw", "wy_dL", "dk"))
def test_every_seed_variant_normalizes_and_renders(kernel: str) -> None:
    bf16 = torch.bfloat16
    bound = {
        "dqkw": lambda: _bind_dqkw(True, bhn=2048, d=128, dv=128),
        "wy_dL": lambda: _bind_wy_dl(True, bf16, bf16, bhn=2048, d=128, dv=128),
        "dk": lambda: _bind_dk(bhn=2048, d=128, dv=128),
    }[kernel]()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    ladder = dag.mma_dag_seed_configs(facts)[: CuteMmaDagHeuristic.SEED_TILES]
    variants = dag.mma_dag_seed_variants(facts, ladder)
    assert variants
    warp = [c for c in spec.compiler_seed_configs if c.config["cute_mma_dag"] == "warp"]
    # The top warp seed's twins (L2 hint, aux ring, occupancy hint) sit
    # behind it; the ladder and its measured neighbours are the rest.
    twins = [c for c in warp if _twin_knobs(c.config)]
    plain = [c for c in warp if not _twin_knobs(c.config)]
    assert len([c for c in twins if c.config.get("load_eviction_policies")]) == 1
    assert len(plain) == len(ladder) + len(variants)
    for config in [*plain[len(ladder) :], *twins]:
        with bound.env:
            normalized = spec.normalized_config(config)
        assert normalized.config["cute_mma_dag"] == "warp"
        assert f"{RT}.mma_bf16(" in _kernel_body(_source(bound, normalized))


# --------------------------------------------------------------------------
# torch.matmul dots and tile.index masks
# --------------------------------------------------------------------------


def _head_sum_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """The sparse-attention indexer's logits: a ``torch.matmul`` per head
    (a bf16 score), relu, a per-row weight, summed over the heads and masked
    to the row's window by ``tile.index``."""
    M, H, N = q.size(0), hl.specialize(q.size(1)), k.size(0)
    hl.specialize(q.size(2))
    out = torch.empty([M, N], dtype=torch.float32, device=q.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        kt = k[tile_n, :]
        for h in hl.grid(H):
            score = torch.matmul(q[tile_m, h, :], kt.transpose(0, 1)).to(torch.float32)
            acc = acc + torch.relu(score) * w[tile_m, h][:, None]
        in_window = (tile_n.index[None, :] >= ks[tile_m][:, None]) & (
            tile_n.index[None, :] < ke[tile_m][:, None]
        )
        out[tile_m, tile_n] = torch.where(in_window, acc, float("-inf"))
    return out


def _head_scores_stored(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Control: each head's bf16 ``torch.matmul`` stored as it is (the
    store would read the unrounded fragment)."""
    M, H, N = q.size(0), hl.specialize(q.size(1)), k.size(0)
    hl.specialize(q.size(2))
    out = torch.empty([M, H, N], dtype=q.dtype, device=q.device)
    for tile_m, tile_n in hl.tile([M, N]):
        kt = k[tile_n, :]
        for h in hl.grid(H):
            out[tile_m, h, tile_n] = torch.matmul(q[tile_m, h, :], kt.transpose(0, 1))
    return out


def _head_sum_inputs(m: int = 256, h: int = 4, n: int = 512, d: int = 128) -> tuple:
    return (
        torch.empty((m, h, d), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((n, d), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((m, h), dtype=torch.float32, device=CPU_DEVICE),
        torch.empty((m,), dtype=torch.int32, device=CPU_DEVICE),
        torch.empty((m,), dtype=torch.int32, device=CPU_DEVICE),
    )


def test_a_16_bit_matmul_is_a_dot_rounded_where_it_is_read() -> None:
    # ``torch.matmul`` of two bf16 tiles is ``aten.mm`` with a bf16 result:
    # the fragment holds the fp32 product and its one reader (the
    # ``.to(torch.float32)``) reads it rounded to bf16, as the SIMT lowering
    # rounds it.  The window mask compares ``tile_n.index`` (the CTA's base
    # plus the column) with int32 bounds loaded per row.
    bound = _bind_fn(_head_sum_logits, _head_sum_inputs())
    program = _program(bound)
    (dot,) = program.dots
    assert dot.node.target is torch.ops.aten.mm.default
    assert dag.rounded_dot_dtype(dot.node) is torch.bfloat16
    assert len(program.tile_index_loads) == 2
    body = _kernel_body(_source(bound, _config(64, 128, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_bf16(" in body
    slot = dot.slot
    # (The read's ``.to(float32)`` rounds and widens the register in one
    # FMA-pipe conversion, ``round_bf16_f32``: the bits of the DSL's
    # ``Float32(BFloat16(x))`` casts without the conversion unit.)
    rounds = re.findall(rf"= {RT}\.round_bf16_f32\(md_c{slot}_\d+_\d+_\d\)", body)
    assert len(rounds) == len(set(rounds)) == dag.mma_dag_frag_regs(64, 128, 4)
    assert "cutlass.BFloat16(" not in body
    assert f"{RT}.ldg_s32(" in body
    (column_base,) = (
        axis for axis in program.axes if axis.block_id == dot.cols.block_id
    )
    assert f"cutlass.Int32(md_a{column_base.index} + " in body


def test_a_16_bit_matmul_read_as_it_is_stays_simt() -> None:
    # A store takes the fragment's registers as they are: the unrounded
    # fp32 product, so the program is not the owner's.
    q, k = _head_sum_inputs()[:2]
    bound = _bind_fn(_head_scores_stored, (q, k))
    with bound.env, bound.host_function:
        assert dag.match_mma_dag(bound.host_function.device_ir.graphs) is None
    assert bound.config_spec.cute_mma_dag is None


# --------------------------------------------------------------------------
# Tile loops whose bound varies with the CTA's root tile coordinates
# --------------------------------------------------------------------------


def _chunk_scan_inputs(chunk: int = 64) -> tuple[torch.Tensor, ...]:
    bf16 = torch.bfloat16
    batch, heads, groups, seq, d, ds = 2, 8, 2, 1024, 64, 64
    chunks = seq // chunk
    empty = functools.partial(torch.empty, dtype=bf16, device=CPU_DEVICE)
    return (
        empty((batch, chunks, groups, chunk, chunk)),
        empty((batch, seq, heads, d)),
        empty((batch, heads, chunks, chunk)),
        empty((batch, heads, chunks, chunk)),
        empty((batch, seq, groups, ds)),
        empty((batch, chunks, heads, d, ds)),
        empty((heads,)),
    )


def _bind_chunk_scan(chunk: int = 64) -> BoundKernel:
    fn = importlib.import_module(
        "examples.mamba2_chunk_scan"
    ).helion_mamba2_chunk_scan_kernel.fn
    return _bind_fn(fn, _chunk_scan_inputs(chunk))


def _causal_plain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A causal K loop over operands read as they are: no pass can mask a
    partial last trip."""
    m, k = a.size()
    _, n = b.size()
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(64, 64)
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[block_m, block_n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _axis_sizes(facts: dag.CuteMmaDagFacts, sizes: dict[int, int]) -> list[int]:
    return [cast("int", facts.block_size(i, sizes)) for i in range(len(facts.axes))]


def test_varying_bound_loops_are_admitted_with_their_largest_extent() -> None:
    # mamba2_chunk_scan: the first dot accumulates onto a fresh ``hl.zeros``
    # (no accumulator), the causal K loop walks ``(tile_m.id + 1) * block_m``
    # (the chunk at the last row tile), its mask compares ``tile_m.index``
    # with the loop tile's ``tile_k.index`` and falls back to ``zeros_like``.
    bound = _bind_chunk_scan()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    program = _program(bound)
    root_dot, loop_dot = program.dots
    assert root_dot.acc is None and root_dot.loop == -1
    assert loop_dot.loop == 0 and loop_dot.a.pointwise
    (loop,) = program.loops
    (axis,) = loop.axes
    assert axis.extent == 64
    rows = next(a for a in program.axes if a.kind == dag.AXIS_ROOT and a.block_id == 0)
    assert loop.tile_end == sympy.expand(
        dag.bound_symbol(dag.BOUND_BLOCK, rows.index)
        * (dag.bound_symbol(dag.BOUND_ID, rows.index) + 1)
    )
    assert any(
        program.tile_index_loads[node] is axis for node in program.tile_index_loads
    )
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    (loop_facts,) = facts.loops
    assert loop_facts.end == loop.tile_end
    assert loop_facts.axis == axis.index and loop_facts.maskable
    # Row tiles below the K block leave partial trips (one trip each); a
    # whole-chunk row tile walks exactly one aligned trip.
    for bm, expected in ((16, (1, 1, False)), (32, (1, 1, False)), (64, (1, 1, True))):
        sizes = _axis_sizes(facts, {0: bm, 1: 64, 2: 64})
        assert (
            dag.loop_trip_range(loop.tile_end, facts.axes, sizes, axis.index)
            == expected
        )
        assert dag.mma_dag_config_reason(facts, {0: bm, 1: 64, 2: 64}, 4, 2) is None


def test_varying_bound_loops_render_the_bound_mask_and_dynamic_trips() -> None:
    bound = _bind_chunk_scan()
    body = _kernel_body(_source(bound, _config(16, 64, 64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert "md_end0 = cutlass.Int32(16 + 16 * (md_a0 // 16))" in body
    # Every element of the cb pass is zeroed at K positions past the bound.
    assert "< md_end0 else cutlass.Float32(0.0)" in body
    assert "md_trips0" not in body
    # The kernel's own causal ``where`` over the bf16-cast operand selects the
    # unrounded fp32 value (the pack rounds once): no scalar converts of
    # either branch.
    assert re.search(
        r"cutlass\.Float32\(md_v\w*\) if md_v\w* else cutlass\.Float32\(0\.0\)", body
    )
    assert "if md_v" in body and "else cutlass.BFloat16(0.0)" not in body
    # Side loads of tensors the program never stores take the read-only path
    # (dA_cumsum / dt rows of the pass, the residual x pairs of the epilogue);
    # the stored tensor is never read through it.
    assert re.search(rf"{RT}\.ldg_bf16_as_f32\([^\n]*, nc=True\)", body)
    assert re.search(rf"{RT}\.ldg_b32\([^\n]*, nc=True\)", body)
    assert not re.search(rf"{RT}\.ldg_\w+\([^\n]*md_base_9[^\n]*nc=True", body)
    # A whole-chunk row tile needs no mask (every CTA's bound is the chunk).
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    assert "< md_end0 else" not in body
    assert "for md_i in cutlass.range(0, 1):" in body
    # Trip counts that differ between CTAs are computed per CTA.
    a = torch.empty((256, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((256, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_causal_plain, (a, b))
    body = _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    assert "md_end0 = cutlass.Int32(64 + 64 * (md_a0 // 64))" in body
    assert "md_trips0 = cutlass.Int32((md_end0 + 63) // 64)" in body
    assert "for md_i in cutlass.range(0, md_trips0):" in body
    assert "else cutlass.Float32(0.0)" not in body


def test_varying_bound_programs_take_the_tcgen05_region_renderer() -> None:
    # The single-loop tcgen05 emitter walks a static trip count and drops an
    # epilogue derived value; the region renderer takes both, with the trip
    # count per CTA, as long as no CTA runs a partial last trip.
    bound = _bind_chunk_scan()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None and "tcgen05" in spec.cute_mma_dag.choices
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.tcgen05 is not None
    tcgen05 = {
        "cute_mma_dag": "tcgen05",
        "cute_mma_dag_warps": 4,
        "cute_mma_dag_stages": 1,
        "cute_mma_dag_store": "staged",
        "cute_mma_dag_exchange": "auto",
        "cute_mma_dag_pass_warps": "off",
        "cute_mma_dag_loads": "tma",
        "cute_mma_dag_persistent": 0,
    }
    with bound.env:
        aligned = spec.normalized_config(
            helion.Config(block_sizes=[64, 64, 64], **tcgen05)
        )
    body = _kernel_body(bound.to_code(aligned))
    assert "md_trips0" not in body  # 64-row tiles of a 64-wide chunk: one trip
    # A 128-wide chunk at 64-row tiles: one or two aligned trips per CTA.
    bound = _bind_chunk_scan(chunk=128)
    with bound.env:
        dynamic = bound.config_spec.normalized_config(
            helion.Config(block_sizes=[64, 64, 64], **tcgen05)
        )
    body = _kernel_body(bound.to_code(dynamic))
    assert "md_trips0 = cutlass.Int32((md_end0 + 63) // 64)" in body
    assert "for md_i in cutlass.range(0, md_trips0):" in body
    assert "tcgen05" in body


def test_a_varying_bound_without_an_operand_pass_refuses_partial_trips() -> None:
    a = torch.empty((256, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((256, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_causal_plain, (a, b))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    (loop_facts,) = facts.loops
    assert loop_facts.end is not None and not loop_facts.maskable
    reason = dag.mma_dag_config_reason(facts, {0: 16, 1: 64, 2: 64}, 4, 2)
    assert reason is not None and "no operand pass" in reason
    assert dag.mma_dag_config_reason(facts, {0: 64, 1: 64, 2: 64}, 4, 2) is None
    assert dag.mma_dag_config_reason(facts, {0: 128, 1: 64, 2: 64}, 4, 2) is None
    with bound.env, pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(_config(16, 64, 64, warps=4, stages=2))


# Loop-less programs: one GEMM over staged tiles with a fused epilogue
# --------------------------------------------------------------------------


def _head_sum_logits_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """The indexer's decode form: the heads folded into one ``torch.matmul``
    (an ``aten.bmm`` whose batch dim is the one-row ``tile_m``), relu, the
    per-head weight, summed over the heads (the fragment rows), masked to
    the row's window and stored -- no device loop at all."""
    M, N = q.size(0), k.size(0)
    hl.specialize(q.size(1))
    hl.specialize(q.size(2))
    out = torch.empty([M, N], dtype=torch.float32, device=q.device)
    for tile_m, tile_n in hl.tile([M, N]):
        kt = k[tile_n, :].transpose(0, 1)
        score = torch.matmul(q[tile_m, :, :], kt).to(torch.float32)
        acc = (torch.relu(score) * w[tile_m, :][:, :, None]).sum(dim=1)
        in_window = (tile_n.index[None, :] >= ks[tile_m][:, None]) & (
            tile_n.index[None, :] < ke[tile_m][:, None]
        )
        out[tile_m, tile_n] = torch.where(in_window, acc, float("-inf"))
    return out


def _loop_less_dot_stored(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One ``hl.dot`` over two whole-extent operands of a one-row batch
    tile, its fp32 product stored: a root region of one dot and one store."""
    BH, N = a.size(0), b.size(2)
    hl.specialize(a.size(1))
    hl.specialize(a.size(2))
    out = torch.empty([BH, a.size(1), N], dtype=torch.float32, device=a.device)
    for tile_bh, tile_n in hl.tile([BH, N], block_size=[1, None]):
        out[tile_bh, :, tile_n] = hl.dot(a[tile_bh, :, :], b[tile_bh, :, tile_n])
    return out


def test_a_loop_less_batched_matmul_program_is_admitted_on_the_warp_emitter() -> None:
    # No device loop: the ``aten.bmm`` over ``q[tile_m, :, :]`` (a leading
    # tile and two rdims) and the lead-broadcast ``kt`` is the program's one
    # dot, the head sum a column-vector reduction of it, the masked store a
    # vector store.  Everything runs in the root region, staged once per
    # CTA; the tensor-memory renderers fail closed.
    args = _head_sum_inputs(h=32)
    bound = _bind_fn(_head_sum_logits_batched, args)
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None and spec.cute_mma_dag_facts is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    facts = spec.cute_mma_dag_facts
    assert facts.loops == () and facts.tcgen05 is None
    assert sorted((a.kind, a.extent) for a in facts.axes) == [
        ("lead", 256),
        ("rdim", 32),
        ("rdim", 128),
        ("root", 512),
    ]
    program = _program(bound)
    assert program.loops == () and program.loop_node is None
    (dot,) = program.dots
    assert dot.node.target is torch.ops.aten.bmm.default
    assert dag.rounded_dot_dtype(dot.node) is torch.bfloat16
    assert (dot.rows.kind, dot.rows.extent) == ("rdim", 32)
    assert (dot.cols.kind, dot.cols.extent) == ("root", 512)
    assert (dot.k.kind, dot.k.extent) == ("rdim", 128)
    # The broadcast views are squeezed off the B operand: a plain
    # transposed load (K along the shared tile's columns).
    assert dot.b.kind == "load" and dot.b.swap and not dot.b.trans
    assert dot.b.value.target is torch.ops.aten.permute.default
    assert not dot.a.pointwise and not dot.b.pointwise
    (reduce,) = program.reduces
    assert reduce.orient == "cols" and reduce.frag is dot.node
    assert len(program.vec_stores) == 1 and not program.stores
    # Admission leaves the block-size fragments alone.
    with patch.object(CuteMmaDagHeuristic, "_facts", return_value=None):
        plain = _bind_fn(_head_sum_logits_batched, args).config_spec
    assert _block_bounds(spec) == _block_bounds(plain)
    # The promoted default is the warp seed at the largest root tile.
    default = spec.default_config().config
    assert default["cute_mma_dag"] == "warp" and default["block_sizes"] == [1, 128]
    source = _source(bound, _config(1, 128, warps=8, stages=4))
    body = _kernel_body(source)
    assert SIMT_FALLBACK not in source and "tcgen05" not in body
    assert "block=(256, 1, 1)" in source
    assert "for md_i" not in body and "md_stage" not in body
    # q (32x128 bf16, 8 KB) and k (128x128 bf16, 32 KB) staged once: 2 + 8
    # packets per thread, one group, one wait, then the eight K steps over
    # the 32x128 fragment (eight warps: one m atom by four n atoms each).
    assert body.count(f"{RT}.cp_async_16(") == 10
    assert body.count("cute.arch.cp_async_commit_group()") == 1
    assert body.count("cute.arch.cp_async_wait_group(0)") == 1
    assert body.count(f"{RT}.mma_bf16(") == 32
    assert body.count(f"{RT}.ldmatrix_x4(") == 24
    # The rounded score (``round_bf16_f32`` of the fp32 register), relu, the
    # per-head weight, the head sum across the octet and the two warp rows,
    # the masked vector store.
    assert f"{RT}.round_bf16_f32(md_c0_" in body
    assert "cutlass.BFloat16(md_c0_" not in body
    assert f"{RT}.octet_sum(" in body and f"{RT}.ldg_s32(" in body
    assert f"{RT}.stg_f32(" in body and f"{RT}.stg_v4_b32(" not in body
    # The example kernel is the same program.
    example = importlib.import_module("examples.sparse_attn_indexer")
    example_bound = _bind_fn(example.mqa_logits_decode.fn, args)
    assert example_bound.config_spec.cute_mma_dag is not None
    assert example_bound.config_spec.default_config().config["cute_mma_dag"] == "warp"


def test_loop_less_seeds_and_normalization_pin_the_inert_stage_depth() -> None:
    # Nothing rides a cp.async ring without a loop: one seed per tile at
    # the default depth, and normalization pins any requested depth to it.
    bound = _bind_fn(_head_sum_logits_batched, _head_sum_inputs(h=32))
    spec = bound.config_spec
    seeds = [c for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag")]
    assert seeds and all(c.config["cute_mma_dag"] == "warp" for c in seeds)
    assert {c.config["cute_mma_dag_stages"] for c in seeds} == {
        dag.MMA_DAG_DEFAULT_STAGES
    }
    # One seed per tile (the first warp seed's twins -- the L2-hinted, the
    # aux-ring and the occupancy-hint twin -- repeat its tile).
    plain = [c for c in seeds if not _twin_knobs(c.config)]
    tiles = [cast("list[int]", c.config["block_sizes"]) for c in plain]
    assert len(tiles) == len({tuple(t) for t in tiles})
    assert len(seeds) - len(plain) >= 1
    with bound.env:
        normalized = spec.normalized_config(_config(1, 64, warps=4, stages=2))
    assert normalized.config["cute_mma_dag_stages"] == dag.MMA_DAG_DEFAULT_STAGES
    assert normalized.config["cute_mma_dag"] == "warp"
    with bound.env, pytest.raises(InvalidConfig, match="tcgen05"):
        spec.normalized_config(
            helion.Config(
                block_sizes=[1, 64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )


def _loop_less_dot_scanned(
    a: torch.Tensor,
    b: torch.Tensor,
    reverse: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """The dot's fp32 product summed down (up) its rows: ``hl.cumsum`` of a
    fragment, a scan step in the root region."""
    BH, N = a.size(0), b.size(2)
    hl.specialize(a.size(1))
    hl.specialize(a.size(2))
    out = torch.empty([BH, a.size(1), N], dtype=torch.float32, device=a.device)
    for tile_bh, tile_n in hl.tile([BH, N], block_size=[1, None]):
        acc = hl.dot(a[tile_bh, :, :], b[tile_bh, :, tile_n]) * 0.5
        out[tile_bh, :, tile_n] = hl.cumsum(acc, dim=1, reverse=reverse)
    return out


@pytest.mark.parametrize("reverse", (False, True))
def test_a_cumsum_of_a_fragment_is_a_scan_step(reverse: bool) -> None:
    # The combine function of ``hl.cumsum`` is a HelperFunction graph the
    # structural gate used to refuse; the scan is a prefix sum of the
    # fragment down (up) its rows: octet shuffles per column register, the
    # halves and atoms chained, the two warps along M meeting in the
    # scratch tile.  Tensor memory fails closed.
    a = torch.empty((16, 32, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((16, 128, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_loop_less_dot_scanned, (a, b, reverse))
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    program = _program(bound)
    (dot,) = program.dots
    (scan,) = program.scans
    assert scan.frag is dot.node and scan.reverse is reverse
    assert scan.orient == "rows" and scan.slot == dot.slot
    (store,) = program.stores
    assert store.frag is scan.node
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.scans == ((dot.slot, "rows"),)
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=4)))
    assert SIMT_FALLBACK not in body and "for md_i" not in body
    shuffle = f"{RT}.{'shfl_down_add_f32' if reverse else 'shfl_up_add_f32'}("
    other = f"{RT}.{'shfl_up_add_f32' if reverse else 'shfl_down_add_f32'}("
    # A 32x64 fragment over four warps is two warps along M by two along
    # N: per column register (four n atoms x two) one m atom of two halves,
    # three shuffle adds each, and the cross-warp prefix through the
    # scratch tile (one total per warp per column: 2 x 64 fp32).
    assert body.count(shuffle) == 8 * 2 * 3 and other not in body
    edge = "md_c" if reverse else "28 + md_c"
    assert body.count("cute.arch.shuffle_sync(md_sc") == 8 * 2
    assert f", {edge}, mask=4294967295, mask_and_clamp=31)" in body
    assert body.count(f"{RT}.sts_f32(md_rv + (md_L0_wmi * 64 + ") == 8
    assert body.count(f"{RT}.lds_f32(md_rv + (") == 16
    assert ("if md_L0_wmi < " if reverse else "if md_L0_wmi > ") in body
    assert "md_rv = md_s0 + " in body
    assert body.count("cute.arch.barrier()") == 3
    # The scan overwrote the dot's slot after reading it.
    assert body.rindex(shuffle) < body.index("md_c0_0_0_0 = md_sc")


def test_a_loop_less_dot_with_a_fragment_store_is_admitted() -> None:
    a = torch.empty((16, 32, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((16, 128, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_loop_less_dot_stored, (a, b))
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    program = _program(bound)
    assert program.loops == ()
    (dot,) = program.dots
    assert dot.node.target is not torch.ops.aten.bmm.default
    (store,) = program.stores
    assert store.region == "epilogue" and store.frag is dot.node
    # ``block_size=[1, None]`` pins the batch tile: one tunable block size.
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=4)))
    assert SIMT_FALLBACK not in body and "for md_i" not in body
    # A 32x64 fragment over four warps: one m atom by four n atoms, eight
    # K steps.
    assert body.count(f"{RT}.mma_bf16(") == 32
    assert body.count("cute.arch.cp_async_commit_group()") == 1


def _k_loop_dot_scanned(
    a: torch.Tensor,
    b: torch.Tensor,
    reverse: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """The running prefix sum down (up) the rows of a K-loop accumulator,
    stored every iteration: ``hl.cumsum`` of the carry, a scan step inside
    the loop whose input (the carry) stays live past it."""
    M, K = a.size()
    N = b.size(1)
    hl.specialize(N)
    out = torch.empty([M, N], dtype=torch.float32, device=a.device)
    for tile_m in hl.tile(M):
        acc = hl.zeros([tile_m, N], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, :], acc=acc)
            out[tile_m, :] = hl.cumsum(acc, dim=0, reverse=reverse)
    return out


def _inner_loop_dot_scanned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The loop-less scan's program under an inner tile loop over N: the
    dot, the scan taking over its slot and the store, all in the loop."""
    BH, N = a.size(0), b.size(2)
    hl.specialize(a.size(1))
    hl.specialize(a.size(2))
    out = torch.empty([BH, a.size(1), N], dtype=torch.float32, device=a.device)
    for tile_bh in hl.tile(BH, block_size=1):
        for tile_n in hl.tile(N):
            acc = hl.dot(a[tile_bh, :, :], b[tile_bh, :, tile_n]) * 0.5
            out[tile_bh, :, tile_n] = hl.cumsum(acc, dim=1)
    return out


def _loop_less_dot_scanned_and_stored(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The scanned value is stored too: the scan's input stays live past
    it, so the scan gets a slot of its own."""
    BH, N = a.size(0), b.size(2)
    hl.specialize(a.size(1))
    hl.specialize(a.size(2))
    out = torch.empty([BH, a.size(1), N], dtype=torch.float32, device=a.device)
    out2 = torch.empty([BH, a.size(1), N], dtype=torch.float32, device=a.device)
    for tile_bh, tile_n in hl.tile([BH, N], block_size=[1, None]):
        acc = hl.dot(a[tile_bh, :, :], b[tile_bh, :, tile_n]) * 0.5
        out[tile_bh, :, tile_n] = hl.cumsum(acc, dim=1)
        out2[tile_bh, :, tile_n] = acc
    return out, out2


@pytest.mark.parametrize("reverse", (False, True))
def test_a_cumsum_inside_a_loop_is_a_scan_step_of_the_loop(reverse: bool) -> None:
    # The scan of the K-loop carry is a step of the loop's region (the
    # loop-body dispatch used to have no Scan branch and the compile
    # failed): the carry stays live, so the scan writes a slot of its own
    # (same shape), and its cross-warp exchange sits inside the iteration
    # behind two barriers like a reduction's.
    a = torch.empty((256, 512), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((512, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_k_loop_dot_scanned, (a, b, reverse))
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    program = _program(bound)
    (dot,) = program.dots
    (scan,) = program.scans
    (loop,) = program.loops
    assert scan.loop == loop.index == dot.loop
    assert scan.frag is dot.node and scan.reverse is reverse
    assert scan.slot != dot.slot
    assert program.slots[scan.slot] == program.slots[dot.slot]
    (store,) = program.stores
    assert store.frag is scan.node and store.loop == loop.index
    body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    loop_body = _loop_and_after(body)
    shuffle = f"{RT}.{'shfl_down_add_f32' if reverse else 'shfl_up_add_f32'}("
    other = f"{RT}.{'shfl_up_add_f32' if reverse else 'shfl_down_add_f32'}("
    # A 64x64 fragment over four warps along M (one m atom of two halves by
    # eight n atoms each): per column register three shuffle adds per
    # half, one total per warp per column through the scratch tile, read
    # back for the four warps.
    assert loop_body.count(shuffle) == 16 * 2 * 3 and other not in body
    assert loop_body.count(f"{RT}.sts_f32(md_rv + (md_L0_wmi * 64 + ") == 16
    assert loop_body.count(f"{RT}.lds_f32(md_rv + (") == 64
    # The iteration barrier and the exchange's two.
    assert loop_body.count("cute.arch.barrier()") == 3
    # The accumulator registers survive the scan; the store reads the
    # scan's.
    assert "md_c0_0_0_0 = md_sc" not in body
    assert "md_c1_0_0_0 = md_sc" in loop_body
    assert loop_body.index(f"{RT}.mma_bf16(") < loop_body.index(shuffle)
    assert loop_body.index("md_c1_0_0_0 = md_sc") < loop_body.index(
        f"{RT}.stg_v2_f32(md_lb, md_c1_0_0_0, md_c1_0_0_1)"
    )


def test_a_cumsum_inside_an_inner_tile_loop_takes_over_the_dots_slot() -> None:
    a = torch.empty((16, 32, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((16, 128, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_inner_loop_dot_scanned, (a, b))
    program = _program(bound)
    (dot,) = program.dots
    (scan,) = program.scans
    (loop,) = program.loops
    assert scan.loop == loop.index == dot.loop and scan.slot == dot.slot
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=4)))
    assert SIMT_FALLBACK not in body
    loop_body = _loop_and_after(body)
    # 32x64 over two warps along M by two along N: eight column registers
    # of one m atom each.
    assert loop_body.count(f"{RT}.shfl_up_add_f32(") == 8 * 2 * 3
    assert loop_body.rindex(f"{RT}.shfl_up_add_f32(") < loop_body.index(
        "md_c0_0_0_0 = md_sc"
    )


def test_a_cumsum_of_a_live_fragment_gets_its_own_slot() -> None:
    a = torch.empty((16, 32, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((16, 128, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_loop_less_dot_scanned_and_stored, (a, b))
    program = _program(bound)
    (dot,) = program.dots
    (scan,) = program.scans
    assert scan.slot != dot.slot
    assert program.slots[scan.slot] == program.slots[dot.slot]
    assert {store.frag for store in program.stores} == {dot.node, scan.node}
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=4)))
    assert SIMT_FALLBACK not in body
    assert "md_c0_0_0_0 = md_sc" not in body and "md_c1_0_0_0 = md_sc" in body


def _loop_less_dot_scanned_with_a_vector(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """dqkg's tail in miniature: the column sums of the fragment folded
    into its last row under an iota mask, then the reverse prefix sum --
    the scan's value reads a vector (the reduction) beside the fragment."""
    BH, N = a.size(0), b.size(2)
    M = hl.specialize(a.size(1))
    hl.specialize(a.size(2))
    out = torch.empty([BH, M, N], dtype=torch.float32, device=a.device)
    for tile_bh, tile_n in hl.tile([BH, N], block_size=[1, None]):
        acc = hl.dot(a[tile_bh, :, :], b[tile_bh, :, tile_n])
        last = acc.sum(dim=1)
        idx = hl.arange(M)
        is_last = (idx == M - 1).float()
        out[tile_bh, :, tile_n] = hl.cumsum(
            acc + is_last[None, :, None] * last[:, None, :], dim=1, reverse=True
        )
    return out


def test_a_cumsum_whose_value_reads_a_vector_is_a_scan_step() -> None:
    # The scan step collects the vectors its value reads like a reduction
    # does (``check_vecs``: defined in an enclosing region, varying along
    # one of the slot's dims); the emitter evaluates the value over the
    # input fragment's registers, so a vector of the fragment's own family
    # resolves to its register at the element's column (no relayout).
    a = torch.empty((16, 32, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((16, 128, 256), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_loop_less_dot_scanned_with_a_vector, (a, b))
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    program = _program(bound)
    (dot,) = program.dots
    (reduce,) = program.reduces
    (scan,) = program.scans
    assert reduce.frag is dot.node and reduce.orient == "cols"
    assert scan.frag is dot.node and scan.reverse
    # The reduction read the dot before the scan: the scan's input stays
    # live past the reduction only, so the scan takes over the dot's slot.
    assert scan.slot == dot.slot
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=4)))
    assert SIMT_FALLBACK not in body
    # A 32x64 fragment over two warps along M by two along N: eight column
    # registers of two halves, three shuffle adds each; the column sums go
    # through ``octet_sum`` and the two warps along M meet in the scratch.
    assert body.count(f"{RT}.shfl_down_add_f32(") == 8 * 2 * 3
    assert f"{RT}.octet_sum(" in body
    assert "md_fv" not in body and "md_rt" not in body
    # The scan's value multiplies the mask by the vector's own register
    # (``md_r<slot>_<atom>_<half>``) at the element's column.
    assert re.search(r"\* md_r\d+_\d+_\d+\b", body)


@pytest.mark.parametrize("diag", (False, True))
def test_dqkg_with_the_gate_gradient_is_a_scan_program(diag: bool) -> None:
    # ``chunk_bwd_dqkg_scalar_helion(compute_dg=True)`` (simple_gla /
    # retention / full_gla training backward): the in-loop ``dh_h_acc +=
    # (h * dh).sum(-1)``, the epilogue ``dg_last = exp_gl * dh_h_acc +
    # (dk_state * k).sum(1)`` and the reverse cumsum of ``dg_raw +
    # is_last[None, :, None] * dg_last[:, None, :]`` -- a scan whose value
    # reads a vector of another fragment's family (relaid through the
    # scratch at the scan fragment's columns).  Warp mode only: tensor
    # memory fails closed on scans.
    bound = _bind_dqkg(diag=diag)
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    program = _program(bound)
    (scan,) = program.scans
    assert scan.reverse and scan.orient == "rows" and scan.loop == -1
    assert len(program.dots) == 5 and len(program.reduces) == 2
    assert len(program.vec_ops) == 3 and len(program.stores) == 3
    assert [store.frag for store in program.stores].count(scan.node) == 1
    (dg_store,) = [store for store in program.stores if store.frag is scan.node]
    assert dg_store.access.dtype == torch.float32
    assert spec.cute_mma_dag_facts is not None
    assert spec.cute_mma_dag_facts.scans == ((scan.slot, "rows"),)
    body = _kernel_body(_source(bound, _config(1, 64, 64, warps=8, stages=2)))
    assert SIMT_FALLBACK not in body
    # A [64, 64] fragment over eight warps: four atoms per warp, two halves
    # each, three shuffle adds per column register.
    assert body.count(f"{RT}.shfl_down_add_f32(") == 4 * 4 * 3
    assert f"{RT}.shfl_up_add_f32(" not in body
    # dg_last follows the [tile_d, tile_dv] family of the loop's reduction:
    # relaid through the scratch into the scan fragment's column positions.
    assert re.search(rf"md_fv\d*_0 = {RT}\.lds_f32\(md_rv", body)
    # The default stays off on the CPU stub (no promoted seed there); the
    # scan program's seeds are all warp seeds.
    with bound.env:
        seeds = CuteMmaDagHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds and all(s.config["cute_mma_dag"] == "warp" for s in seeds)


def test_the_se_block_forward_is_a_loop_less_program() -> None:
    # The loop-less gate admits ``x[tile_m, :] @ w[:, :]`` with its sigmoid
    # and scaled-store epilogue (examples/se_block.py's forward before it
    # tiled the reduction): an fp16 ``aten.mm`` over the whole reduction,
    # one derived value, two epilogue stores.
    # The promoted default is warp at the catalog's small shape; at the
    # larger ones no warp tile fits (w alone is 512 KB and up), so there
    # are no seeds and the default stays off.
    from test.test_cute_full_slice_matmul import _se_block_full_slice_fwd

    def bind(m: int, n: int) -> BoundKernel:
        x = torch.empty((m, n), dtype=torch.float16, device=CPU_DEVICE)
        w = torch.empty((n, n), dtype=torch.float16, device=CPU_DEVICE)
        # The structural policy's full-slice stripmine would hand the
        # contraction a K loop (the tcgen05 K-loop family); this pins the
        # un-rewritten loop-less body the mma-dag owner claims.
        return helion.kernel(
            _se_block_full_slice_fwd,
            backend="cute",
            static_shapes=True,
            cute_full_slice_matmul_tiling=False,
        ).bind((x, w))  # type: ignore[arg-type]

    bound = bind(256, 128)
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None and spec.cute_mma_dag_facts is not None
    assert spec.cute_mma_dag.choices == ("off", "warp")
    facts = spec.cute_mma_dag_facts
    assert facts.loops == () and facts.tcgen05 is None
    assert sorted((a.kind, a.extent) for a in facts.axes) == [
        ("rdim", 128),
        ("root", 256),
    ]
    program = _program(bound)
    assert program.loops == ()
    (dot,) = program.dots
    assert dot.node.target is torch.ops.aten.mm.default
    assert dag.rounded_dot_dtype(dot.node) is torch.float16
    (derived,) = program.derived
    assert [store.frag for store in program.stores] == [dot.node, derived.node]
    assert all(store.region == "epilogue" for store in program.stores)
    default = spec.default_config().config
    assert default["cute_mma_dag"] == "warp" and default["block_sizes"] == [128]
    body = _kernel_body(_source(bound, _config(128, warps=8, stages=4)))
    assert SIMT_FALLBACK not in body and "for md_i" not in body
    # x (128x128 fp16) and w (128x128 fp16) staged once; eight warps over
    # the 128x128 fragment, eight K steps.
    assert body.count(f"{RT}.cp_async_16(") == 16
    assert body.count("cute.arch.cp_async_commit_group()") == 1
    assert body.count(f"{RT}.mma_f16(") == 128
    for m, n in ((1024, 512), (4096, 1024)):
        spec = bind(m, n).config_spec
        assert spec.cute_mma_dag is not None
        assert not [
            c for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag")
        ]
        assert spec.default_config().config["cute_mma_dag"] == "off"


# Vectors beside fragments, loaded seeds, 16-bit derived values, fp32 matmuls
# --------------------------------------------------------------------------


def _bind_dwk(*, d: int = 64, dv: int = 64) -> BoundKernel:
    """kda's parallel state adjoint: its epilogue combines the fragment
    ``dgc_pos`` with ``dgc_last``, the column sum of the same fragment plus
    the loop-carried row reduction ``delast``, under the ``is_last`` row
    mask."""
    bh, n, c = 2, 8, 64
    bf16, f32 = torch.bfloat16, torch.float32

    def e(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=CPU_DEVICE)

    return _bind(
        "chunk_bwd_state_dwk_kda_helion",
        (
            e(bh, n, c, d),
            e(bh, n, c, d, dtype=f32),
            e(bh, n, d, dv),
            e(bh, n, c, dv),
            e(bh, n, c, dv, dtype=f32),
            e(bh, n, d, dv, dtype=f32),
        ),
    )


def test_dwk_reads_the_column_sum_of_its_fragment_beside_it() -> None:
    # ``dgc_s = dgc_pos + is_last * dgc_last[:, None, :]``: the store value
    # combines the derived fragment ``dk_scaled * k_scaled`` with a vector
    # (``delast * elast`` relaid from the loop's row vector plus the
    # fragment's column sum).  The vector's family follows the ``[bk, bv]``
    # loop fragment's warp grid, not the ``[C, bk]`` epilogue fragment's, so
    # the store relays it through the scratch tile at the register's
    # column (``md_fv_*``); the vector op before it relays the column sum
    # into the row vector's layout (``md_rt_*``).
    bound = _bind_dwk()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
    program = _program(bound)
    assert [type(step).__name__ for step in program.body] == [
        "Derived",
        "Loop",
        "Store",
        "Derived",
        "Store",
        "Derived",
        "VecOp",
        "Derived",
        "Reduce",
        "VecOp",
        "Store",
    ]
    assert [slot.orient for slot in program.vec_slots] == ["rows", "rows", "cols"]
    assert all(program.vec_frag_slots[s] for s in range(len(program.vec_slots)))
    (loop,) = program.loops
    assert len(loop.vec_carries) == 1 and not loop.vec_carries[0].identity
    body = _kernel_body(_source(bound, _config(1, 64, 32, warps=8, stages=2)))
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_bf16(" in body
    epilogue = body[body.rindex("for md_i in cutlass.range(") :]
    assert "md_rt_0_0 = " in epilogue
    relaid = re.findall(r"(md_fv_\d+) = _helion_mma_dag\.lds_f32\(md_rv", epilogue)
    assert relaid and len(relaid) == len(set(relaid))
    # The relayout's reads come after the owning lanes' writes and a
    # barrier, with another barrier before the scratch is reused; the
    # masked store (``is_last * dgc_last``) consumes the temporaries.
    writes = epilogue.rindex(f"{RT}.sts_f32(md_rv")
    first = epilogue.index("md_fv_0 = ")
    assert writes < first
    assert epilogue.count("cute.arch.barrier()", writes, first) == 1
    assert epilogue.count("cute.arch.barrier()", first) >= 1
    assert re.search(r"= md_v_\d+ \* md_fv_0\b", epilogue)


def test_a_vector_inside_an_operand_chain_is_not_mine() -> None:
    # The operand passes evaluate per tile element, not per fragment
    # register: a dot operand reading a vector (``(acc * rowsum).to(bf16)``)
    # declines, while the same value stored is admitted.
    bf16, f32 = torch.bfloat16, torch.float32
    b, m, k, n, p = 4, 64, 128, 32, 16
    a = torch.empty((b, m, k), dtype=bf16, device=CPU_DEVICE)
    w = torch.empty((b, k, n), dtype=bf16, device=CPU_DEVICE)
    c = torch.empty((b, n, p), dtype=bf16, device=CPU_DEVICE)
    scaled = torch.empty((b, m, n), dtype=f32, device=CPU_DEVICE)
    out = torch.empty((b, m, p), dtype=f32, device=CPU_DEVICE)
    bound = _bind_fn(_row_scaled_stored, (a, w, scaled))
    program = _program(bound)
    assert [type(step).__name__ for step in program.body] == [
        "Loop",
        "Reduce",
        "Store",
    ]
    bound = _bind_fn(_row_scaled_operand, (a, w, c, out))
    with bound.env, bound.host_function:
        assert dag.match_mma_dag(bound.host_function.device_ir.graphs) is None
    assert bound.config_spec.cute_mma_dag is None


def _row_scaled_stored(
    a: torch.Tensor, w: torch.Tensor, scaled: torch.Tensor
) -> torch.Tensor:
    b, m, k = a.shape
    for tile_b in hl.tile(b, block_size=1):
        acc = hl.zeros([tile_b, m, w.size(2)], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_b, :, tile_k], w[tile_b, tile_k, :], acc=acc)
        rowsum = torch.sum(acc, dim=-1)
        scaled[tile_b, :, :] = acc * rowsum[:, :, None]
    return scaled


def _row_scaled_operand(
    a: torch.Tensor, w: torch.Tensor, c: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    b, m, k = a.shape
    for tile_b in hl.tile(b, block_size=1):
        acc = hl.zeros([tile_b, m, w.size(2)], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_b, :, tile_k], w[tile_b, tile_k, :], acc=acc)
        rowsum = torch.sum(acc, dim=-1)
        lhs = (acc * rowsum[:, :, None]).to(torch.bfloat16)
        out[tile_b, :, :] = hl.dot(lhs, c[tile_b, :, :])
    return out


# The config whose kernel died with CUDA_ERROR_ILLEGAL_INSTRUCTION in the
# kda forward's cold autotune (chunk_fwd_wy_delta_helion, diag_anchored, the
# (8, 8, 1024, 64, 64) shape): a warp mma_dag with ``load_eviction_policies``
# naming "l2_last" at the staged k and v slots, whose cp.async packets the
# emitter once hinted ``.L2::cache_hint`` -- a form the CuTe DSL's nvJitLink
# miscompiles (see ``emit_issue_tile``).
_WY_FAULT_CONFIG: dict[str, object] = {
    "block_sizes": [8, 8],
    "cute_collective_compute": "warp",
    "cute_collective_copy": "scalar",
    "cute_collective_mma": False,
    "cute_collective_native_seeded": False,
    "cute_collective_tmem_a": False,
    "cute_collective_tmem_seed": False,
    "cute_lane_layouts": ["blocked"] * 5,
    "cute_min_blocks_per_mp": 0,
    "cute_mma_dag": "warp",
    "cute_mma_dag_derived": "exchange",
    "cute_mma_dag_epilogue": "row",
    "cute_mma_dag_exchange": "auto",
    "cute_mma_dag_loads": "tma",
    "cute_mma_dag_lookahead": "off",
    "cute_mma_dag_pass_warps": "off",
    "cute_mma_dag_persistent": 0,
    "cute_mma_dag_stages": 8,
    "cute_mma_dag_store": "staged",
    "cute_mma_dag_warps": 4,
    "cute_pdl": True,
    "cute_reduction_reloads": ["auto"],
    "cute_register_chain": False,
    "cute_store_policy": "default",
    "cute_vector_widths": [1] * 5,
    "load_eviction_policies": ["", "", "l2_last", "", "", "l2_last"],
    "num_threads": [0, 0, 0],
}


def test_the_autotune_fault_config_renders_hinted_packets_behind_a_sass_check() -> None:
    # The config whose kernel died in the kda autotune renders its staged k
    # and v tiles' packets hinted ``L2::cache_hint`` again (the normalizer
    # keeps the two staged slots at their canonical value and pins the
    # rest), and the module declares the SASS check the launcher runs
    # before the kernel's first launch: the DSL's assembler, not the
    # render, decides whether a hinted LDGSTS is encoded with its shared
    # address in a uniform register beside the descriptor
    # (``sass_check.hinted_cp_async_misencoding``), so a misencoded kernel
    # is refused instead of launched.  The blank policy list renders plain
    # packets and no check, byte for byte as before; the L1-only priorities
    # leave the L1-bypassing packets plain too.
    bound = _bind_wy_anchored(bhn=1024, d=64, dv=64)
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    # k (2), the derived value's ring load (3) and v (5) are the staged tiles.
    assert facts is not None and facts.staged_slots == (2, 3, 5)
    config = helion.Config.from_dict(_WY_FAULT_CONFIG)
    with bound.env:
        normalized = spec.normalized_config(config)
    assert normalized.config["load_eviction_policies"] == [
        "",
        "",
        "l2_last",
        "",
        "",
        "l2_last",
    ]
    hinted = bound.to_code(normalized)
    body = _kernel_body(hinted)
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_tf32(" in body
    # k's and v's packets are hinted (a hinted tile keeps its packets under
    # the config's ``tma`` loads: ``plan_tma``); the derived value's ring
    # load (slot 3, "" in the config) is the one tile a tensor map fetches
    # -- hinted at slot 3 too, it keeps hinted packets and nothing rides
    # the bulk tensor copy.
    hinted_packets = body.count(f"{RT}.cp_async_16_l2(")
    plain_packets = body.count(f"{RT}.cp_async_16(")
    assert hinted_packets and not plain_packets
    assert "cp_async_bulk_tensor_shared_cta_global" in body
    assert body.count("md_pol_evict_last = ") == 1
    every = _kernel_body(
        _source(
            bound,
            helion.Config.from_dict(
                {
                    **_WY_FAULT_CONFIG,
                    "load_eviction_policies": [
                        "",
                        "",
                        "l2_last",
                        "l2_last",
                        "",
                        "l2_last",
                    ],
                }
            ),
        )
    )
    slot3_packets = every.count(f"{RT}.cp_async_16_l2(") - hinted_packets
    assert slot3_packets > 0
    assert f"{RT}.cp_async_16(" not in every
    assert "cp_async_bulk_tensor_shared_cta_global" not in every
    assert (
        f"._helion_cute_sass_checks = ({dag.MMA_DAG_HINTED_CP_ASYNC_CHECK!r},)"
        in hinted
    )
    plain = _source(
        bound,
        helion.Config.from_dict(
            {**_WY_FAULT_CONFIG, "load_eviction_policies": [""] * 6}
        ),
    )
    assert "cp_async_16_l2" not in plain
    assert "md_pol_" not in plain
    assert "_helion_cute_sass_checks" not in plain
    # (v's converted tile keeps its packets -- a tensor map fetches no raw
    # source -- while k's and the derived ring load's tiles ride tensor maps.)
    assert 0 < _kernel_body(plain).count(f"{RT}.cp_async_16(") < hinted_packets
    assert "cp_async_bulk_tensor_shared_cta_global" in _kernel_body(plain)
    # The same hints with the per-thread packets everywhere (``lane``) is
    # the ``every`` kernel: the normalizer pins the route to ``tma``.
    with bound.env:
        lane = spec.normalized_config(
            helion.Config.from_dict(
                {
                    **_WY_FAULT_CONFIG,
                    "cute_mma_dag_loads": "lane",
                    "load_eviction_policies": [
                        "",
                        "",
                        "l2_last",
                        "l2_last",
                        "",
                        "l2_last",
                    ],
                }
            )
        )
    assert lane.config["cute_mma_dag_loads"] == "tma"
    assert (
        _source(
            bound,
            helion.Config.from_dict(
                {**_WY_FAULT_CONFIG, "load_eviction_policies": ["last", "first"] * 3}
            ),
        )
        == plain
    )
    # Read as text: the runtime module imports cutlass, absent on CPU rows.
    runtime = Path(dag.__file__).with_name("_mma_dag_runtime.py").read_text()
    assert "cp.async.cg.shared.global.L2::cache_hint" in runtime
    assert "createpolicy.fractional.L2::evict_last" in runtime


def test_staged_tiles_carry_the_l2_policy_of_their_slot() -> None:
    # The decode program's loads in program order take the tunable slots k
    # (0), q (1), the weights (2), ``tile_n.index`` (3, 5), ks (4), ke (6);
    # the two operand tiles are the ones cp.async stages.  The L2 policies
    # of their slots ride the packets as ``L2::cache_hint`` with one
    # ``createpolicy`` word per policy ahead of them; the L1-only
    # priorities leave the (L1-bypassing) packets untouched.
    bound = _bind_fn(_head_sum_logits_batched, _head_sum_inputs(h=32))
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.staged_slots == (0, 1)
    assert spec.load_eviction_policies.length == 7
    program = _program(bound)
    assert [(a.name, a.eviction_slot) for a in program.accesses] == [
        ("q", 1),
        ("k", 0),
        ("w", 2),
        ("ks", 4),
        ("ke", 6),
        ("out", None),
    ]

    def render(*policies: str) -> str:
        return _source(
            bound,
            _config(1, 64, warps=4, stages=4, load_eviction_policies=list(policies)),
        )

    plain = render(*[""] * 7)
    assert render("last", "first", "last", "last", "last", "last", "last") == plain
    body = _kernel_body(plain)
    # k's 64x128 tile: eight packets per thread; q's 32x128: four.
    assert body.count(f"{RT}.cp_async_16(") == 12 and "md_pol_" not in body
    assert "_helion_cute_sass_checks" not in plain
    source = render("l2_last", "l1_l2_last", "l2_last", "", "", "", "")
    hinted = _kernel_body(source)
    assert hinted.count(f"{RT}.l2_policy_evict_last()") == 1
    assert hinted.count(f"{RT}.cp_async_16_l2(") == 12
    assert f"{RT}.cp_async_16(" not in hinted
    assert hinted.count(", md_pol_evict_last)") == 12
    assert hinted.index("md_pol_evict_last = ") < hinted.index(f"{RT}.cp_async_16_l2(")
    assert "._helion_cute_sass_checks = ('hinted_cp_async',)" in source
    # The weights' side loads keep their plain form whatever their slot says.
    assert hinted.count(f"{RT}.ldg_f32(") == body.count(f"{RT}.ldg_f32(")
    first = _kernel_body(render("streaming", "nc_l2_first", "", "", "", "", ""))
    assert first.count(f"{RT}.l2_policy_evict_first()") == 1
    assert first.count(", md_pol_evict_first)") == 12
    mixed = _kernel_body(render("l2_last", "streaming", "", "", "", "", ""))
    assert mixed.count(", md_pol_evict_last)") == 8
    assert mixed.count(", md_pol_evict_first)") == 4
    assert mixed.count("md_pol_") == 2 + 12


def test_the_normalizer_keeps_the_staged_slots_canonical_and_pins_the_rest() -> None:
    # One value per packet form on the staged slots ("l2_last" for the
    # evict_last hint, "streaming" for evict_first); the side loads' slots,
    # inert in the mode, pin to identity.  The ordinary lowering (mode off)
    # keeps every slot as given.
    bound = _bind_fn(_head_sum_logits_batched, _head_sum_inputs(h=32))
    spec = bound.config_spec

    def normalized(**extra: object) -> list[str]:
        with bound.env:
            config = spec.normalized_config(
                helion.Config.from_dict({"block_sizes": [1, 64], **extra})
            )
        return cast("list[str]", config.config["load_eviction_policies"])

    warp = {"cute_mma_dag": "warp", "cute_mma_dag_warps": 4}
    policies = [
        "l1_l2_last",
        "nc_l2_first",
        "l2_last",
        "last",
        "streaming",
        "",
        "first",
    ]
    assert normalized(load_eviction_policies=policies, **warp) == [
        "l2_last",
        "streaming",
        "",
        "",
        "",
        "",
        "",
    ]
    assert (
        normalized(load_eviction_policies=["last", "first", *[""] * 5], **warp)
        == [""] * 7
    )
    assert (
        normalized(load_eviction_policies=["last"] * 7, cute_mma_dag="off")
        == ["last"] * 7
    )


def test_hinted_packets_need_a_disassembler() -> None:
    # Without ``nvdisasm`` the launcher could not verify a hinted kernel's
    # SASS, so nothing hinted exists: the normalizer pins every slot to ""
    # (the render is the plain one) and the seeds carry no hinted twin.
    hinted = _config(1, 64, warps=4, stages=4, load_eviction_policies=["l2_last"] * 7)
    with patch.object(cutedsl_compat, "nvdisasm_path", return_value=None):
        assert not cutedsl_compat.hinted_cp_async_available()
        # (The seeds are planted at bind time.)
        bound = _bind_fn(_head_sum_logits_batched, _head_sum_inputs(h=32))
        spec = bound.config_spec
        seeds = [
            c.config for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag")
        ]
        with bound.env:
            normalized = spec.normalized_config(hinted)
        assert normalized.config["load_eviction_policies"] == [""] * 7
        source = bound.to_code(normalized)
    assert "cp_async_16_l2" not in source and "_helion_cute_sass_checks" not in source
    assert source == _source(bound, _config(1, 64, warps=4, stages=4))
    assert seeds and not any("load_eviction_policies" in c for c in seeds)
    assert cutedsl_compat.hinted_cp_async_available()


def test_the_first_warp_seed_has_an_l2_evict_last_twin() -> None:
    # Right behind the first warp seed (and its derived-route / side-load
    # twins), the same tile with the staged operands' slots at "l2_last";
    # the plain seed stays the promoted default and no other seed carries
    # a policy.
    bound = _bind_fn(_head_sum_logits_batched, _head_sum_inputs(h=32))
    spec = bound.config_spec
    seeds = [
        c.config for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag")
    ]
    first = next(i for i, c in enumerate(seeds) if c["cute_mma_dag"] == "warp")
    plain, twin = seeds[first], seeds[first + 1]
    assert "load_eviction_policies" not in plain
    assert twin["load_eviction_policies"] == ["l2_last", "l2_last", "", "", "", "", ""]
    assert {k: v for k, v in twin.items() if k != "load_eviction_policies"} == plain
    assert sum("load_eviction_policies" in c for c in seeds) == 1
    with bound.env:
        normalized = spec.normalized_config(helion.Config.from_dict(twin))
    assert normalized.config["load_eviction_policies"] == twin["load_eviction_policies"]
    assert not any(
        cast("list[str]", spec.default_config().config["load_eviction_policies"])
    )
    # Behind the derived-route twin where the first warp seed has one.
    dx = _bind_se_block("se_block_bwd_dx").config_spec
    seeds = [
        c.config
        for c in dx.compiler_seed_configs
        if c.config.get("cute_mma_dag") == "warp"
    ]
    assert seeds[1]["cute_mma_dag_derived"] == "pass"
    assert "load_eviction_policies" not in seeds[1]
    assert seeds[2]["cute_mma_dag_derived"] == "exchange"
    assert seeds[2]["load_eviction_policies"]
    assert seeds[2]["block_sizes"] == seeds[0]["block_sizes"]
    assert sum("load_eviction_policies" in c for c in seeds) == 1
    # Behind the aux-ring twin where the first warp seed has one.
    bwd = _bind_bwd("scalar").config_spec
    seeds = [
        c.config
        for c in bwd.compiler_seed_configs
        if c.config.get("cute_mma_dag") == "warp"
    ]
    assert seeds[1].get("cute_mma_dag_aux") == "ring"
    assert "load_eviction_policies" not in seeds[1]
    assert seeds[2]["load_eviction_policies"]
    assert seeds[2].get("cute_mma_dag_aux", "register") == "register"
    assert seeds[2]["block_sizes"] == seeds[0]["block_sizes"]


def test_ring_and_zero_filled_packets_carry_the_policy_too() -> None:
    # A K-loop recurrence: the ring's packets of every iteration name the
    # one policy word bound ahead of the loop.  A ragged root tile: the
    # zero-filled packets take the policy behind their byte count.
    bound = _bind_fwd()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.staged_slots
    policies = [
        "l2_last" if slot in facts.staged_slots else ""
        for slot in range(spec.load_eviction_policies.length)
    ]
    body = _kernel_body(
        _source(
            bound, _config(64, 64, warps=4, stages=2, load_eviction_policies=policies)
        )
    )
    assert body.count(f"{RT}.cp_async_16_l2(") == 16
    assert f"{RT}.cp_async_16(" not in body
    assert body.count("md_pol_evict_last = ") == 1
    assert body.index("md_pol_evict_last = ") < body.index("for md_i in cutlass.range(")
    ragged = _bind_fn(_ragged_rows_summed, _ragged_inputs(m=8))
    facts = ragged.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.staged_slots == (0, 1)
    body = _kernel_body(
        _source(
            ragged,
            _config(
                16,
                128,
                warps=4,
                stages=2,
                load_eviction_policies=["l2_last", "l2_last"],
            ),
        )
    )
    assert body.count(f"{RT}.cp_async_16_l2_zfill(") == 1
    assert body.count(f"{RT}.cp_async_16_l2(") == 8
    assert "md_zf, md_pol_evict_last)" in body
    assert f"{RT}.cp_async_16_zfill(" not in body and f"{RT}.cp_async_16(" not in body


def test_anchored_wy_seeds_the_neumann_chain_from_the_loaded_gram() -> None:
    # The kda WY transform: ``A = -(beta * Akk[b])`` and ``T = eye + A`` are
    # pointwise chains over the loaded gram (not over the kk loop's
    # accumulator, as in the gdn variant), so both carries are seeded from
    # the tile's pairs loaded at the fragment coordinates and the chain
    # evaluated per register; ``k_state_out = k * exp2(gc[C-1] - gc)`` is a
    # derived value over two loaded tiles stored from the loop body.
    bound = _bind_wy_anchored()
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
    program = _program(bound)
    assert [type(step).__name__ for step in program.body] == [
        "Loop",
        "Store",
        "Loop",
        "Loop",
    ]
    chain = program.loops[0]
    assert len(chain.carries) == 2
    for carry in chain.carries:
        assert carry.init is not None and carry.init.name == "Akk"
        assert carry.init_frag is None and carry.init_value is not None
    assert [len(carry.init_aux) for carry in chain.carries] == [1, 1]
    assert [store.access.name for store in program.stores] == [
        "A_inv",
        "k_state_out",
        "w",
        "u",
    ]
    assert [store.source for store in program.stores] == [
        "epilogue",
        "body",
        "body",
        "body",
    ]
    assert [(len(d.frags), len(d.loads)) for d in program.derived] == [(0, 2), (0, 2)]
    body = _kernel_body(_source(bound, _config(32, 64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_tf32(" in body
    seed = body[: body.index("for md_i in cutlass.range(")]
    # Each carry's 64x64 tile over 128 lanes: 16 bf16 pairs per lane.
    assert seed.count(f"{RT}.unpack_bf16x2({RT}.ldg_b32(") == 32
    assert "cute.arch.add_packed_f32x2(" in seed
    assert f"{RT}.mma_" not in seed


def _tf32_se_block_bwd_dx(
    grad_out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """examples/se_block.py's ``se_block_bwd_dx`` before the example moved its
    matmul operands to the input dtype: an fp32 (tf32) dot whose A operand is
    the derived chain ``2 * g * x * s * (1 - s)`` over widened half tiles and
    whose B is the widened transposed w tile, with the half-typed
    ``acc += 2.0 * grad_out * s`` seeding the carry.  The subject of the tf32
    derived-chain tests here, in the tcgen05 files and on the GPU."""
    m, n = x.size()
    grad_x = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc += 2.0 * grad_out[tile_m, tile_n] * s[tile_m, tile_n]
        for tile_k in hl.tile(n):
            grad_to_w = (
                2.0
                * grad_out[tile_m, tile_k].to(torch.float32)
                * x[tile_m, tile_k].to(torch.float32)
                * s[tile_m, tile_k].to(torch.float32)
                * (1.0 - s[tile_m, tile_k].to(torch.float32))
            )
            acc += grad_to_w @ w[tile_n, tile_k].to(torch.float32).T
        grad_x[tile_m, tile_n] = acc.to(x.dtype)
    return grad_x


def _tf32_se_block_bwd_dw(
    grad_out: torch.Tensor, x: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """``se_block_bwd_dw`` of the same tf32 form: the widened transposed half
    tile of x as A, the derived chain as B."""
    m, n = x.size()
    grad_w = torch.zeros([n, n], dtype=torch.float32, device=x.device)
    for tile_n1, tile_n2 in hl.tile([n, n]):
        acc_w = hl.zeros([tile_n1, tile_n2], dtype=torch.float32)
        for tile_m in hl.tile(m):
            grad_to_w = (
                2.0
                * grad_out[tile_m, tile_n2].to(torch.float32)
                * x[tile_m, tile_n2].to(torch.float32)
                * s[tile_m, tile_n2].to(torch.float32)
                * (1.0 - s[tile_m, tile_n2].to(torch.float32))
            )
            acc_w += x[tile_m, tile_n1].to(torch.float32).T @ grad_to_w
        grad_w[tile_n1, tile_n2] = acc_w.to(x.dtype)
    return grad_w


TF32_SE_BLOCK: dict[str, Callable[..., torch.Tensor]] = {
    "se_block_bwd_dx": _tf32_se_block_bwd_dx,
    "se_block_bwd_dw": _tf32_se_block_bwd_dw,
}


def _bind_se_block(name: str, *, precision: str | None = None) -> BoundKernel:
    m, n = 256, 128
    f16 = torch.float16
    x = torch.empty((m, n), dtype=f16, device=CPU_DEVICE)
    w = torch.empty((n, n), dtype=f16, device=CPU_DEVICE)
    args: tuple[object, ...] = (x, x, w, x) if name == "se_block_bwd_dx" else (x, x, x)
    settings = {} if precision is None else {"dot_precision": precision}
    fn = TF32_SE_BLOCK[name]
    return helion.kernel(fn, backend="cute", static_shapes=True, **settings).bind(args)  # type: ignore[arg-type]


def test_fp32_matmuls_are_tf32_dots_under_the_default_precision() -> None:
    # ``acc += a @ b`` of fp32 tiles is traced as ``aten.addmm``: a tf32 dot
    # with the accumulator leading, like Triton's ``tl.dot`` runs the same
    # source; ``"ieee"`` keeps the exact lowering.  dw's A operand is the
    # widened transposed half tile, its B the derived ``2 * g * x * s *
    # (1 - s)`` fragment; dx's pre-seed ``acc += 2.0 * grad_out * s`` is a
    # half-typed derived value (torch promotion) seeding the carry.
    dw = _bind_se_block("se_block_bwd_dw")
    assert dw.config_spec.cute_mma_dag is not None
    program = _program(dw)
    (dot,) = program.dots
    assert dot.node.target is torch.ops.aten.addmm.default
    with dw.env:
        # (The dtype query reads the dot precision off the environment.)
        assert dag.mm_dot_dtype(dot.node) is torch.float32
        assert dag.rounded_dot_dtype(dot.node) is None
    assert dot.a.dtype == dot.b.dtype == torch.float32
    assert dot.acc_frag is not None and dot.slot == dot.acc_slot
    assert (dot.a.kind, dot.a.pointwise, dot.a.trans) == ("load", True, True)
    assert dot.b.kind == "frag"
    assert (
        _bind_se_block("se_block_bwd_dw", precision="ieee").config_spec.cute_mma_dag
        is None
    )
    dx = _bind_se_block("se_block_bwd_dx")
    assert dx.config_spec.cute_mma_dag is not None
    program = _program(dx)
    (dot,) = program.dots
    assert (dot.a.kind, dot.b.kind, dot.b.pointwise) == ("frag", "load", True)
    (loop,) = program.loops
    (carry,) = loop.carries
    assert carry.init is None and carry.init_value is not None
    seed = carry.init_frag
    assert seed is not None and seed.meta["val"].dtype == torch.float16
    assert program.frag_regions[seed] == -1
    body = _kernel_body(_source(dx, _config(64, 64, 64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_tf32(" in body
    slot = program.frag_slots[seed]
    # The half-typed value's rounded elements are widened into its fp32
    # registers, and the carry's seed reads them re-typed.
    assert re.search(rf"md_c{slot}_\d+_\d+_\d = cutlass\.Float32\(", body)
    assert re.search(rf"= cutlass\.Float16\(md_c{slot}_\d+_\d+_\d\)", body)


# --------------------------------------------------------------------------
# Non-finite constants beside a fragment
# --------------------------------------------------------------------------


def _masked_neg_inf_stored(
    a: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``where(mask, acc, -inf)`` over the finished accumulator: the
    ``hl.full`` fill beside the fragment is a non-finite constant."""
    b, m, k = a.shape
    for tile_b in hl.tile(b, block_size=1):
        acc = hl.zeros([tile_b, m, w.size(2)], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_b, :, tile_k], w[tile_b, tile_k, :], acc=acc)
        rows = hl.arange(m)
        cols = hl.arange(w.size(2))
        mask = rows[:, None] >= cols[None, :]
        out[tile_b, :, :] = torch.where(
            mask[None, :, :],
            acc,
            hl.full([tile_b, m, w.size(2)], float("-inf"), dtype=torch.float32),
        )
    return out


def _masked_neg_inf_inputs() -> tuple[torch.Tensor, ...]:
    bf16, f32 = torch.bfloat16, torch.float32
    b, m, k, n = 4, 64, 128, 64
    a = torch.empty((b, m, k), dtype=bf16, device=CPU_DEVICE)
    w = torch.empty((b, k, n), dtype=bf16, device=CPU_DEVICE)
    out = torch.empty((b, m, n), dtype=f32, device=CPU_DEVICE)
    return a, w, out


@pytest.mark.parametrize(
    ("value", "literal"),
    (
        (float("-inf"), "float('-inf')"),
        (float("inf"), "float('inf')"),
        (float("nan"), "float('nan')"),
        (-2.5, "-2.5"),
        (0.0, "0.0"),
    ),
    ids=("neg_inf", "inf", "nan", "finite", "zero"),
)
def test_float_literals_spell_non_finite_values_as_calls(
    value: float, literal: str
) -> None:
    # ``repr(float("inf"))`` is the bare name ``inf``, undefined in the
    # generated module: the DSL source spells it as a ``float(...)`` call.
    assert _float_literal(value) == literal


def test_a_non_finite_fill_beside_a_fragment_renders_as_a_call() -> None:
    # ``where(mask, acc, hl.full(..., float("-inf")))`` after the K loop:
    # the fill is a uniform constant read beside the fragment (admitted on
    # both modes) and renders as ``cutlass.Float32(float('-inf'))`` -- a
    # bare ``cutlass.Float32(-inf)`` fails the DSL compile with a NameError.
    bound = _bind_fn(_masked_neg_inf_stored, _masked_neg_inf_inputs())
    spec = bound.config_spec
    assert spec.cute_mma_dag is not None
    assert spec.cute_mma_dag.choices == ("off", "warp", "tcgen05")
    program = _program(bound)
    assert [type(step).__name__ for step in program.body] == ["Loop", "Store"]
    body = _kernel_body(_source(bound, _config(64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_bf16(" in body
    assert "cutlass.Float32(float('-inf'))" in body
    assert "Float32(-inf)" not in body
    assert "Float32(inf)" not in body


# --------------------------------------------------------------------------
# Ragged root tiles: a fragment past the axis extent
# --------------------------------------------------------------------------


def _ragged_rows_summed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a @ b + 1).sum(0)`` over the rows of one root tile (the rows fit
    one tile): the sum's input is masked to zero past the row extent by
    the frontend's ``_mask_to`` (the add's masked value is one, not zero),
    the result a vector along N."""
    M, N = a.size(0), b.size(1)
    hl.specialize(a.size(1))
    out = torch.empty([N], dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([M, N]):
        s = hl.dot(a[tile_m, :], b[:, tile_n])
        out[tile_n] = (s + 1.0).sum(dim=0)
    return out


def _ragged_cols_summed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a @ b + 1).sum(1)`` per row over the whole ``b``: a vector along
    the (ragged) rows, stored under the row guard."""
    M = a.size(0)
    hl.specialize(a.size(1))
    hl.specialize(b.size(1))
    out = torch.empty([M], dtype=torch.float32, device=a.device)
    for tile_m in hl.tile(M):
        s = hl.dot(a[tile_m, :], b[:, :])
        out[tile_m] = (s + 1.0).sum(dim=1)
    return out


def _ragged_dot_stored(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out[tile_m, tile_n] = a[tile_m, :] @ b[:, tile_n]`` into a given
    output (a transposed one makes ``tile_m`` its contiguous dim)."""
    M, N = a.size(0), b.size(1)
    hl.specialize(a.size(1))
    for tile_m, tile_n in hl.tile([M, N]):
        out[tile_m, tile_n] = hl.dot(a[tile_m, :], b[:, tile_n])
    return out


def _ragged_inputs(m: int, k: int = 64, n: int = 256) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((k, n), dtype=torch.bfloat16, device=CPU_DEVICE),
    )


def test_a_root_tile_past_a_small_extent_runs_the_warp_path() -> None:
    # The indexer's logits at M = 8: the 16-row MMA atom overshoots the
    # rows.  The row axis may run ragged (it is never a dot K, never the
    # contiguous dim of a 2-D access), the config rule admits the tile on
    # the warp path only, the seeds walk the padded tile, and the block-size
    # search reaches it (the dag owner raises the row block's ceiling from
    # the extent to the 16-row atom).
    bound = _bind_fn(_head_sum_logits, _head_sum_inputs(m=8))
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    assert sorted((a.kind, a.extent, a.ragged) for a in facts.axes) == [
        ("rdim", 128, False),
        ("root", 8, True),
        ("root", 512, False),
    ]
    sizes = {facts.axes[0].block_id: 16, facts.axes[1].block_id: 128}
    assert dag.mma_dag_config_reason(facts, sizes, 4, 2) is None
    assert dag.mma_dag_config_reason(facts, sizes, 4, 2, "tcgen05") == (
        "block size 16 must divide the extent 8 of tile axis 0"
    )
    assert dag.mma_dag_seed_ladder(facts.axes[0]) == [16]
    assert dag.mma_dag_seed_ladder(facts.axes[1]) == [16, 32, 64, 128]
    assert dag.mma_dag_cta_count(facts, [16, 128, 128]) == 4
    assert dag.mma_dag_cta_count(facts, [16, 64, 128]) == 8
    seeds = [
        c.config for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag")
    ]
    assert seeds and {cast("list[int]", c["block_sizes"])[0] for c in seeds} == {16}
    assert spec.block_sizes[0].max_size == 16
    body = _kernel_body(_source(bound, _config(16, 128, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body and f"{RT}.mma_bf16(" in body
    # q's 16-row tile: the packets of the rows past 8 read nothing (both
    # rounds of the staging, in the prologue and in the loop); k's tile
    # along N stays whole.
    assert (
        "md_zf = cutlass.Int32(16) if md_a0 + md_pr16 < 8 else cutlass.Int32(0)" in body
    )
    assert "if md_a0 + (8 + md_pr16) < 8 else cutlass.Int32(0)" in body
    assert body.count(f"{RT}.cp_async_16_zfill(") == 4
    assert body.count(f"{RT}.cp_async_16(") == 16
    # The per-row weight and the window bounds load zero past the extent.
    assert (
        "md_ok = cutlass.Int32(1) if md_a0 + md_L0_rowl < 8 else cutlass.Int32(0)"
        in body
    )
    assert body.count(f"{RT}.ldg_f32_pred(") == 2 and f"{RT}.ldg_f32(" not in body
    assert body.count(f"{RT}.ldg_s32_pred(") == 4 and f"{RT}.ldg_s32(" not in body
    # The pair stores of each fragment row are guarded by its coordinate.
    assert body.count("if md_a0 + md_L0_rowl < 8:") == 4
    assert body.count("if md_a0 + (md_L0_rowl + 8) < 8:") == 4
    assert body.count(f"{RT}.stg_v2_f32(") == 8


def test_a_one_row_tile_is_computed_at_the_mma_atom() -> None:
    # M = 1: the frontend resolves a tile over an extent of one to one row
    # whatever the config says, so the warp path pads the fragment to the
    # 16-row atom itself; the block divides the extent, so the guards test
    # the coordinate against the block alone.
    bound = _bind_fn(_head_sum_logits, _head_sum_inputs(m=1))
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.axes[0].ragged
    assert dag.mma_dag_padded_sizes(facts, [1, 128, 128]) == [16, 128, 128]
    assert dag.mma_dag_padded_sizes(facts, [16, 128, 128]) == [16, 128, 128]
    assert dag.mma_dag_config_reason(facts, {0: 1, 1: 128}, 4, 2) is None
    assert dag.mma_dag_seed_ladder(facts.axes[0]) == [16]
    source = _source(bound, _config(16, 128, warps=4, stages=2))
    body = _kernel_body(source)
    assert "(4,)," in source.split("_launcher(")[-1]
    assert "md_zf = cutlass.Int32(16) if md_pr16 < 1 else cutlass.Int32(0)" in body
    assert "if 8 + md_pr16 < 1 else cutlass.Int32(0)" in body
    assert body.count("if md_L0_rowl < 1:") == 4
    assert body.count("if md_L0_rowl + 8 < 1:") == 4
    assert "md_ok = cutlass.Int32(1) if md_L0_rowl < 1 else cutlass.Int32(0)" in body
    assert "md_a0 + md_L0_rowl" not in body


def test_a_block_below_the_atom_guards_against_the_block_and_the_extent() -> None:
    # M = 24 under blocks of 8: three CTAs each compute the 16-row atom and
    # own eight rows; the guard tests the block, and (24 % 8 == 0) only the
    # block.  Under blocks of 16 the second CTA holds rows 24..31: the
    # guard tests the extent.
    bound = _bind_fn(_head_sum_logits, _head_sum_inputs(m=24))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.axes[0].ragged
    assert dag.mma_dag_seed_ladder(facts.axes[0]) == [32]
    assert dag.mma_dag_padded_sizes(facts, [8, 128, 128]) == [16, 128, 128]
    assert dag.mma_dag_cta_count(facts, [8, 128, 128]) == 12
    assert dag.mma_dag_cta_count(facts, [16, 128, 128]) == 8
    with patch.object(
        importlib.import_module("helion._compat"),
        "_min_dot_size",
        lambda *args: (1, 1, 16),
    ):
        bound = _bind_fn(_head_sum_logits, _head_sum_inputs(m=24))
        body = _kernel_body(_source(bound, _config(8, 128, warps=4, stages=2)))
    assert "md_zf = cutlass.Int32(16) if md_pr16 < 8 else cutlass.Int32(0)" in body
    assert body.count("if md_L0_rowl < 8:") == 4
    assert body.count("if md_L0_rowl + 8 < 8:") == 4
    body = _kernel_body(_source(bound, _config(16, 128, warps=4, stages=2)))
    assert (
        "md_zf = cutlass.Int32(16) if md_a0 + md_pr16 < 24 else cutlass.Int32(0)"
        in body
    )
    assert body.count("if md_a0 + md_L0_rowl < 24:") == 4


def test_a_reduction_over_ragged_rows_masks_them_to_the_fill() -> None:
    # The frontend's ``_mask_to`` ahead of the row sum is no longer an
    # identity: the elements of the rows past the extent take the fill (a
    # select per register), and the vector store along N is unguarded.
    from helion.language import _tracing_ops

    bound = _bind_fn(_ragged_rows_summed, _ragged_inputs(m=8))
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    assert sorted((a.kind, a.extent, a.ragged) for a in facts.axes) == [
        ("rdim", 64, False),
        ("root", 8, True),
        ("root", 256, False),
    ]
    program = _program(bound)
    (reduce,) = program.reduces
    assert reduce.orient == "cols"
    assert reduce.value.target is _tracing_ops._mask_to
    assert reduce.value.args[1] == 0
    body = _kernel_body(_source(bound, _config(16, 128, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    # a's 16x64 tile: eight chunks per row, one round of packets.
    assert (
        "md_zf = cutlass.Int32(16) if md_a0 + md_pr8 < 8 else cutlass.Int32(0)" in body
    )
    assert body.count(f"{RT}.cp_async_16_zfill(") == 1
    selects = re.findall(
        r"md_v(?:_\d+)? = md_v(?:_\d+)? if md_a0 \+ (?:md_L0_rowl|\(md_L0_rowl \+ 8\)) "
        r"< 8 else cutlass\.Float32\(0\.0\)",
        body,
    )
    assert len(selects) == dag.mma_dag_frag_regs(16, 128, 4)
    assert f"{RT}.octet_sum(" in body
    tail = body.split(f"{RT}.octet_sum(")[-1]
    assert tail.count(f"{RT}.stg_f32(") == 8 and "if md_a0" not in tail


def test_a_vector_along_ragged_rows_is_stored_under_the_row_guard() -> None:
    # ``(a @ b + 1).sum(1)`` with M = 24 under 32-row tiles (one CTA) and
    # 16-row tiles (the second CTA's vector has eight rows past the
    # extent): the vector store along the rows is guarded by its row.
    a, b = _ragged_inputs(m=24, n=128)
    bound = helion.kernel(
        _ragged_cols_summed,  # type: ignore[arg-type]
        backend="cute",
        static_shapes=True,
        cute_full_slice_matmul_tiling=False,
    ).bind((a, b))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    assert sorted((a.kind, a.extent, a.ragged) for a in facts.axes) == [
        ("rdim", 64, False),
        ("rdim", 128, False),
        ("root", 24, True),
    ]
    program = _program(bound)
    (reduce,) = program.reduces
    assert reduce.orient == "rows"
    for bm in (32, 16):
        body = _kernel_body(_source(bound, _config(bm, warps=4, stages=2)))
        assert SIMT_FALLBACK not in body
        # a's tile (eight chunks per row): 32 rows take two rounds of 128
        # packets, 16 rows one; b's whole 64x128 tile stays unguarded.
        assert (
            "md_zf = cutlass.Int32(16) if md_a0 + md_pr8 < 24 else cutlass.Int32(0)"
            in body
        )
        assert ("if md_a0 + (16 + md_pr8) < 24 else cutlass.Int32(0)" in body) == (
            bm == 32
        )
        assert body.count(f"{RT}.cp_async_16_zfill(") == bm // 16
        assert body.count(f"{RT}.cp_async_16(") == 64 * 128 * 2 // 16 // 128
        selects = re.findall(
            r" if md_a0 \+ .* < 24 else cutlass\.Float32\(0\.0\)", body
        )
        assert len(selects) == dag.mma_dag_frag_regs(bm, 128, 4)
        assert f"{RT}.lane_group_sum(" in body
        stores = body[body.index(f"{RT}.lane_group_sum(") :]
        assert re.search(
            r"if md_a0 \+ md_L0_rowl < 24:\n\s+_helion_mma_dag\.stg_f32\(md_lb, ",
            stores,
        )
        assert re.search(
            r"if md_a0 \+ \(md_L0_rowl \+ 8\) < 24:\n\s+_helion_mma_dag\.stg_f32\(",
            stores,
        )
        # One store per fragment row the lane owns (rows g / g + 8 of each
        # atom row of its warp's share).
        wm, _wn = cast("tuple[int, int]", warp_mma_warp_layout(bm, 128, 4))
        assert stores.count(f"{RT}.stg_f32(") == 2 * (bm // wm // 16)


def test_a_ragged_axis_on_a_contiguous_dim_keeps_whole_tiles() -> None:
    # The output transposed: ``tile_m`` is its contiguous dim, whose
    # 16-byte packets and pairs would straddle the extent -- the axis may
    # not run ragged, so a 16-row tile over M = 8 is refused as before.
    a, b = _ragged_inputs(m=8)
    out = torch.empty((256, 8), dtype=torch.float32, device=CPU_DEVICE).t()
    bound = _bind_fn(_ragged_dot_stored, (a, b, out))
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    assert sorted((a.kind, a.extent, a.ragged) for a in facts.axes) == [
        ("rdim", 64, False),
        ("root", 8, False),
        ("root", 256, False),
    ]
    assert dag.mma_dag_seed_ladder(facts.axes[0]) == []
    assert dag.mma_dag_config_reason(facts, {0: 16, 2: 128}, 4, 2) == (
        "block size 16 must divide the extent 8 of tile axis 0"
    )
    with bound.env, pytest.raises(InvalidConfig, match="must divide the extent 8"):
        spec.normalized_config(_config(16, 128, warps=4, stages=2))
    # The same store into a row-major output: the rows may run ragged.
    out = torch.empty((8, 256), dtype=torch.float32, device=CPU_DEVICE)
    facts = _bind_fn(_ragged_dot_stored, (a, b, out)).config_spec.cute_mma_dag_facts
    assert facts is not None and facts.axes[0].ragged


def test_the_matmul_block_size_overshoot_is_a_backend_capability() -> None:
    # The frontend asks the backend whether its matmul lowering pads a
    # small output dim instead of testing the backend's name: Triton pads
    # (``tl.dot`` over a masked tile), TileIR never had the overshoot, and
    # cute does not pad wholesale -- its ordinary lowerings read and write
    # exactly the extent, and a backend-wide ceiling moved the seeds of
    # kernels the ragged warp path never covers (the gdn recurrence's
    # dstate tile, the collective tf32 jagged kernels).  The ragged tile is
    # the dag owner's: its facts registration raises the ceiling of a
    # ragged root axis to its MMA atom, and only there.
    from helion._compiler.cute.backend import CuteBackend
    from helion._compiler.triton.backend import TileIRBackend
    from helion._compiler.triton.backend import TritonBackend

    assert not CuteBackend().pads_small_matmul_dims
    assert TritonBackend().pads_small_matmul_dims
    assert not TileIRBackend().pads_small_matmul_dims
    spec = _bind_fn(_head_sum_logits, _head_sum_inputs(m=8)).config_spec
    m_spec, n_spec = spec.block_sizes
    assert (m_spec.dim_max_size, m_spec.max_size) == (8, 16)
    assert n_spec.max_size == 512
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    assert dag.mma_dag_ragged_ceilings(facts) == {m_spec.block_id: 16}
    # The fragment's own default stays clamped to the extent (the promoted
    # warp seed is what carries the padded tile into the default config).
    assert m_spec._fragment(spec).default() == 8
    # Without the dag owner the ceiling is the extent, as on every other
    # cute kernel.
    with patch.object(CuteMmaDagHeuristic, "_facts", return_value=None):
        plain = _bind_fn(_head_sum_logits, _head_sum_inputs(m=8)).config_spec
    assert [b.max_size for b in plain.block_sizes] == [8, 512]
    # A root axis the ragged rules refuse (the contiguous dim of the
    # transposed output) keeps the extent's ceiling under the dag owner.
    a, b = _ragged_inputs(m=8)
    out = torch.empty((256, 8), dtype=torch.float32, device=CPU_DEVICE).t()
    spec = _bind_fn(_ragged_dot_stored, (a, b, out)).config_spec
    assert spec.cute_mma_dag_facts is not None
    assert spec.block_sizes[0].max_size == 8
    # A ragged extent at or above its atom already admits it (24 under 32).
    spec = _bind_fn(_head_sum_logits, _head_sum_inputs(m=24)).config_spec
    assert (spec.block_sizes[0].dim_max_size, spec.block_sizes[0].max_size) == (
        32,
        32,
    )
    # A dim at or above the old overshoot keeps its own ceiling.
    spec = _bind_fn(_head_sum_logits, _head_sum_inputs(m=256)).config_spec
    assert spec.block_sizes[0].max_size == 256


# Pass chains: a pointwise operand over loaded tiles written by a derived pass
# --------------------------------------------------------------------------


def _half_chain_operand(
    g: torch.Tensor, x: torch.Tensor, s: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """A 16-bit A operand that is a pass chain: ``2 * g * x * (1 - s)`` over
    three bf16 tiles stays bf16-typed (torch promotion) and feeds a bf16
    ``hl.dot``; squeeze db's ``grad_d`` chain spelled as an ``hl.dot`` (its
    own ``torch.addmm`` form is ``_accumulating_addmm``).  The B operand's
    scale keeps the tensor-core K-loop family (direct loads only) from
    claiming the contraction."""
    m, k = g.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=g.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            lhs = (
                2.0 * g[tile_m, tile_k] * x[tile_m, tile_k] * (1.0 - s[tile_m, tile_k])
            )
            acc = hl.dot(lhs, w[tile_k, tile_n] * 0.5, acc=acc)
        out[tile_m, tile_n] = acc
    return out


def _half_chain_inputs() -> tuple[torch.Tensor, ...]:
    bf16 = torch.bfloat16
    g = torch.empty((256, 128), dtype=bf16, device=CPU_DEVICE)
    w = torch.empty((128, 128), dtype=bf16, device=CPU_DEVICE)
    return g, g, g, w


def _bind_half_chain() -> BoundKernel:
    return _bind_fn(_half_chain_operand, _half_chain_inputs())


def _accumulating_addmm(
    grad_out: torch.Tensor, x: torch.Tensor, d: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """the pre-split squeeze_and_excitation_net ``bwd_db`` body (grad_d recomputed per K tile): ``acc = torch.addmm(acc,
    c[tile_m, tile_k].T, grad_d)`` -- a half ``addmm`` into an fp32
    accumulator (the ``hl.dot(a, b, acc=acc)`` contract), its B operand the
    half chain ``grad_out * x * d * (1 - d)``, its A operand a transposed
    half tile."""
    m, n = grad_out.size()
    k = c.size(1)
    grad_b = torch.empty([k, n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_k, tile_n in hl.tile([k, n]):
        acc = hl.zeros([tile_k, tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            grad_d = (
                grad_out[tile_m, tile_n]
                * x[tile_m, tile_n]
                * d[tile_m, tile_n]
                * (1.0 - d[tile_m, tile_n])
            )
            acc = torch.addmm(acc, c[tile_m, tile_k].T, grad_d)
        grad_b[tile_k, tile_n] = acc
    return grad_b


def _half_accumulating_addmm(
    grad_out: torch.Tensor, x: torch.Tensor, d: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """The same body accumulating in half precision: every ``addmm`` rounds
    the running sum, which no fp32-accumulating dot models."""
    m, n = grad_out.size()
    k = c.size(1)
    grad_b = torch.empty([k, n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_k, tile_n in hl.tile([k, n]):
        acc = hl.zeros([tile_k, tile_n], dtype=torch.float16)
        for tile_m in hl.tile(m):
            grad_d = grad_out[tile_m, tile_n] * x[tile_m, tile_n] * d[tile_m, tile_n]
            acc = torch.addmm(acc, c[tile_m, tile_k].T, grad_d)
        grad_b[tile_k, tile_n] = acc
    return grad_b


def _accumulating_addmm_inputs() -> tuple[torch.Tensor, ...]:
    f16 = torch.float16
    grad_out = torch.empty((1024, 512), dtype=f16, device=CPU_DEVICE)
    c = torch.empty((1024, 256), dtype=f16, device=CPU_DEVICE)
    return grad_out, grad_out, grad_out, c


def test_a_half_addmm_into_an_fp32_accumulator_is_a_dot() -> None:
    # ``acc = torch.addmm(acc, a16, b16)`` is the tiled-GEMM idiom Triton
    # lowers to ``tl.dot(a, b, acc)``: fp32 products into the fp32
    # accumulator, no rounding -- what ``hl.dot(a16, b16, acc=acc)`` spells.
    # Its operands here are the ones the tensor-core K-loop family declines
    # (a computed B tile), so the kernel ran SIMT or collective warp MMA.
    bound = _bind_fn(_accumulating_addmm, _accumulating_addmm_inputs())
    spec = bound.config_spec
    assert spec.cute_tcgen05_search_enabled is False
    assert spec.cute_mma_dag is not None
    assert "tcgen05" in spec.cute_mma_dag.choices
    program = _program(bound)
    (dot,) = program.dots
    assert dot.node.target is torch.ops.aten.addmm.default
    with bound.env:
        assert dag.accumulating_addmm_dtype(dot.node) is torch.float16
        assert dag.mm_dot_dtype(dot.node) is None
        assert dag.rounded_dot_dtype(dot.node) is None
    assert dot.node.meta["val"].dtype == torch.float32
    assert dot.a.dtype == dot.b.dtype == torch.float16
    # The accumulator leads: the dot accumulates in place into its slot.
    assert dot.acc is not None and dot.acc_frag is not None
    assert dot.slot == dot.acc_slot
    # A: the transposed half tile ``c[tile_m, tile_k].T`` (K along the
    # staged tile's rows); B: the half chain, a pass chain.
    assert (dot.a.kind, dot.a.pointwise, dot.a.swap, dot.a.trans) == (
        "load",
        False,
        True,
        True,
    )
    chain_root = program.chain_root(dot.b)
    assert dot.b.kind == "frag" and chain_root is not None
    assert [access.name for access in program.chain_loads(chain_root)] == [
        "grad_out",
        "x",
        "d",
    ]
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.pass_chains == 1
    body = _kernel_body(_source(bound, _config(32, 32, 64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_f16" in body
    assert "ldmatrix" in body
    # The seeds lead with the owner's modes (the promoted default).
    seeds = [c.config for c in spec.compiler_seed_configs]
    assert seeds and seeds[0].get("cute_mma_dag") in ("warp", "tcgen05")
    # A half accumulator rounds the running sum: not a dot of this owner.
    half = _bind_fn(_half_accumulating_addmm, _accumulating_addmm_inputs())
    assert half.config_spec.cute_mma_dag is None
    host = half.host_function
    assert host is not None
    with half.env, host:
        nodes = [
            node
            for info in host.device_ir.graphs
            for node in info.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        ]
        (node,) = nodes
        assert dag.accumulating_addmm_dtype(node) is None
        assert not dag._is_dot(node)


def test_a_pointwise_operand_over_loaded_tiles_is_a_pass_chain() -> None:
    # se_block dx's A operand ``2 * g * x * s * (1 - s)`` is three derived
    # steps over three loaded tiles (``s`` read twice) that nothing but the
    # dot reads: a pass chain, the subject of ``cute_mma_dag_derived``.  The
    # loaded B operand is not one; the knob's first choice (the default)
    # keeps the register-direct derived fragment and its exchange tile.
    dx = _bind_se_block("se_block_bwd_dx")
    program = _program(dx)
    (dot,) = program.dots
    root = program.chain_root(dot.a)
    assert root is not None and program.chain_root(dot.b) is None
    chain = program.pass_chains[root]
    assert chain[-1].node is root
    assert [len(step.frags) for step in chain] == [0, 1, 1]
    assert [access.name for access in program.chain_loads(root)] == [
        "grad_out",
        "x",
        "s",
    ]
    facts = dx.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.pass_chains == 1
    assert [tile.pass_ring for tile in facts.tiles].count(True) == 1
    spec = dx.config_spec
    assert spec.cute_mma_dag_derived is not None
    assert spec.cute_mma_dag_derived.choices == ("exchange", "pass")
    # dx's K loop streams under the pass route (every loop of the program is
    # a streamed candidate), so the pass twin leads the tensor-memory seeds;
    # at this 256x128 shape the grid is below 128 CTAs at the top
    # tensor-memory tile, so the warp mode leads and the default keeps the
    # exchange route.
    default = spec.default_config().config
    assert (default["cute_mma_dag"], default["cute_mma_dag_derived"]) == (
        "warp",
        "exchange",
    )
    # dw's B operand (the same chain feeding B) and a bf16-typed chain
    # feeding a 16-bit dot are pass chains as well.
    for bound, role in (
        (_bind_se_block("se_block_bwd_dw"), "b"),
        (_bind_half_chain(), "a"),
    ):
        program = _program(bound)
        (dot,) = program.dots
        assert program.chain_root(getattr(dot, role)) is not None
        assert bound.config_spec.cute_mma_dag_facts.pass_chains == 1  # type: ignore[union-attr]
    # The seeds compare the two routes at the first tile of each mode: on
    # tensor memory the pass variant leads for this streamed program, the
    # exchange route right behind it; in the warp mode the pass variant
    # follows the first seed.
    seeds = [
        c.config
        for c in dx.config_spec.compiler_seed_configs
        if c.config.get("cute_mma_dag") in ("warp", "tcgen05")
    ]
    routes = [(c["cute_mma_dag"], c["cute_mma_dag_derived"]) for c in seeds]
    first_tc = routes.index(("tcgen05", "pass"))
    assert routes[first_tc + 1] == ("tcgen05", "exchange")
    assert seeds[first_tc]["block_sizes"] == seeds[first_tc + 1]["block_sizes"]
    first_warp = routes.index(("warp", "exchange"))
    assert routes[first_warp + 1] == ("warp", "pass")
    assert (
        routes.count(("tcgen05", "pass")) == 1 and routes.count(("warp", "pass")) == 1
    )


def test_the_derived_route_has_no_subject_without_a_pass_chain() -> None:
    # fwd_h's derived values read fragments: no pass chain, so the knob is
    # pinned to its first choice (one autotune identity), and a chain whose
    # sources the lanes load (not provably 16-byte aligned) keeps the
    # exchange route at codegen rather than failing.
    bound = _bind_fwd(d=64, dv=64)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.pass_chains == 0
    assert not _program(bound).pass_chains
    with bound.env:
        pinned = bound.config_spec.normalized_config(
            _config(64, 64, warps=4, stages=2, cute_mma_dag_derived="pass")
        )
    assert pinned.config["cute_mma_dag_derived"] == "exchange"
    dx = _bind_se_block("se_block_bwd_dx")
    with dx.env:
        kept = dx.config_spec.normalized_config(
            _config(64, 64, 64, warps=4, stages=2, cute_mma_dag_derived="pass")
        )
        off = dx.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64, 64, 64],
                cute_mma_dag="off",
                cute_mma_dag_derived="pass",
            )
        )
    assert kept.config["cute_mma_dag_derived"] == "pass"
    assert off.config["cute_mma_dag_derived"] == "exchange"
    program = _program(dx)
    (dot,) = program.dots
    root = program.chain_root(dot.a)
    assert root is not None
    with patch.object(dag, "async_tile_reason", return_value="unaligned"):
        plan = _plan_of(dx, kept)
    (planned,) = plan.program.dots
    # (The layout solver's route for the fragment: an exchange tile or a
    # register route, never the pass.)
    assert plan.route(planned, planned.a) != dag.ROUTE_DERIVED
    assert not plan.absorbed
    plan = _plan_of(dx, kept)
    (planned,) = plan.program.dots
    assert plan.route(planned, planned.a) == dag.ROUTE_DERIVED
    assert len(plan.absorbed) == 3


def test_the_derived_pass_writes_the_operand_tile_from_the_staged_sources() -> None:
    # The warp emitter under ``pass``: no chain registers, no exchange tile;
    # behind the iteration barrier every thread loads its eight-element
    # groups of the three staged source tiles as 16-byte packets, evaluates
    # the chain in fp32 (packed f32x2 muls) and writes two fp32 packets per
    # group into the single-copy A tile, then one barrier ahead of the
    # ``ldmatrix`` reads -- the exchange route's barrier count and shared
    # memory, a third of its shared-memory instructions.
    dx = _bind_se_block("se_block_bwd_dx")
    program = _program(dx)
    (dot,) = program.dots
    root = program.chain_root(dot.a)
    assert root is not None
    slot = program.frag_slots[root]
    exchange_body = _kernel_body(_source(dx, _config(64, 64, 64, warps=8, stages=2)))
    pass_body = _kernel_body(
        _source(dx, _config(64, 64, 64, warps=8, stages=2, cute_mma_dag_derived="pass"))
    )
    exchange, passed = _loop_and_after(exchange_body), _loop_and_after(pass_body)
    assert SIMT_FALLBACK not in pass_body
    assert f"{RT}.mma_tf32(" in passed
    # The exchange route: the chain's registers, written into ``md_x0``.
    assert re.search(rf"md_c{slot}_\d+_\d+_\d = ", exchange)
    assert f"{RT}.sts_v2_f32(md_x0" in exchange
    assert exchange.count(f"{RT}.lds_b32(md_d") == 32
    # The pass route: the same shared memory, one 16-byte load per source and
    # group (64x64 / 8 groups over 256 threads: two per thread), two fp32
    # packets per group, no fragment of the chain.
    assert "md_x0" not in pass_body
    assert not re.search(rf"md_c{slot}_", passed)
    assert passed.count(f"{RT}.lds_v4_b32(md_d") == 6
    assert passed.count(f"{RT}.sts_v4_f32(md_t0 + ") == 4
    assert passed.count("cute.arch.barrier()") == exchange.count("cute.arch.barrier()")
    assert _smem_bytes(pass_body) == _smem_bytes(exchange_body)
    assert "md_t0 = md_s0 + 0" in pass_body
    # Order: refill issue, the pass, its barrier, the operand reads.
    refill = passed.index("cute.arch.cp_async_commit_group()")
    first_pass_store = passed.index(f"{RT}.sts_v4_f32(md_t0 + ")
    barrier = passed.index("cute.arch.barrier()", first_pass_store)
    assert (
        refill < first_pass_store < barrier < passed.index(f"{RT}.ldmatrix_x4(md_t0 + ")
    )
    # A chain over fp32 math rounds nowhere: the sources are widened exactly
    # (``unpack_f16x2``) and the operand tile holds fp32 values.
    assert f"{RT}.unpack_f16x2(" in passed
    assert f"{RT}.pack_f16x2(" not in passed


def _smem_bytes(body: str) -> int:
    match = re.search(r"alloc_smem\(cutlass\.Uint8, (\d+)", body)
    assert match is not None
    return int(match.group(1))


def test_a_half_typed_pass_chain_packs_its_operand_tile_once() -> None:
    # A bf16-typed chain ``2 * g * x * (1 - s)`` (torch promotion) feeding
    # a bf16 dot: the pass evaluates it as the SIMT lowering does and packs
    # eight elements into one 16-byte packet of the 16-bit A tile (one
    # rounding, ``pack_bf16x2``); the exchange route rounds the same values
    # into the fragment registers.
    bound = _bind_half_chain()
    program = _program(bound)
    (dot,) = program.dots
    assert dot.a.kind == "frag" and dot.a.dtype == torch.bfloat16
    assert program.chain_root(dot.a) is not None
    body = _kernel_body(
        _source(
            bound, _config(64, 64, 64, warps=4, stages=2, cute_mma_dag_derived="pass")
        )
    )
    loop = _loop_and_after(body)
    assert SIMT_FALLBACK not in body
    assert f"{RT}.mma_bf16(" in loop
    assert loop.count(f"{RT}.sts_v4_b32(md_t0 + ") == 4
    assert f"{RT}.pack_bf16x2(" in loop
    assert f"{RT}.ldmatrix_x4(md_t0 + " in loop
    # (The exchange route feeds this straight bf16 A fragment from its
    # registers -- the identity repack -- so no tile is written for it; the
    # B tile's in-place pass packs its own packets on both routes.)
    exchange = _loop_and_after(
        _kernel_body(_source(bound, _config(64, 64, 64, warps=4, stages=2)))
    )
    root = program.chain_root(dot.a)
    assert root is not None
    slot = program.frag_slots[root]
    assert re.search(rf"md_c{slot}_\d+_\d+_\d = ", exchange)
    assert not re.search(rf"md_c{slot}_", loop)
    assert f"{RT}.sts_v4_b32(md_t0 + " not in exchange


def test_varlen_state_pass_is_admitted_with_tensor_loop_bounds() -> None:
    # chunk_fwd_h_delta_varlen_helion walks ``hl.tile(chunk_offsets[n],
    # chunk_offsets[n + 1], block_size=1)``: a grid loop whose bounds are
    # tensor elements and whose index is the tile's begin; the chunk row
    # ``(hn // N) * NT + tile_j.begin`` mixes the lead coordinate, the int
    # argument NT and the loop index.  Admitted: the bounds are one uniform
    # scalar load each ahead of the loop, the trip count dynamic.
    bound = _bind_fwd_h_delta_varlen()
    assert bound.config_spec.cute_mma_dag is not None
    assert "cute_mma_dag" in bound.config_spec.default_config().config
    program = _program(bound)
    (loop,) = program.loops
    assert loop.static_trip is None and loop.symbol is not None
    assert [access.name for access in program.scalar_loads] == [
        "chunk_offsets",
        "chunk_offsets",
    ]
    assert all(
        access.dtype == torch.int64 and not access.dims
        for access in program.scalar_loads
    )
    assert program.begin is program.scalar_loads[0].node
    assert program.end is program.scalar_loads[1].node
    # The epilogue stores the final state (the carry's phi) per sequence.
    assert [store.region for store in program.stores].count("epilogue") == 1
    body = _kernel_body(_source(bound, _config(32, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert body.count(f"cutlass.Int32({RT}.ldg_s64(") == 2
    assert "for md_i in cutlass.range(md_scalar, md_scalar_1):" in body
    # The ring prologue and the refill are guarded by the dynamic bounds.
    assert "if md_scalar < md_scalar_1:" in body
    assert "if md_i + 1 < md_scalar_1:" in body
    assert "md_stage = (md_i - md_scalar) % 2" in body
    assert f"{RT}.mma_bf16(" in body
    # (The dense form keeps its static trip count.)
    (dense_loop,) = _program(_bind_fwd_h_delta()).loops
    assert dense_loop.static_trip == 8


def _varlen_binders() -> dict[str, Callable[[], BoundKernel]]:
    """The varlen chunked KDA forward's kernels at one small batch: three
    sequences of 2, 1 and 3 chunks, every last chunk one row short."""
    bf16, f32 = torch.bfloat16, torch.float32
    h, c, d, dv = 2, 64, 64, 64
    _offsets, token_base, valid_len, nt, t_total = _varlen_tables((2, 1, 3), c)
    hnt = h * nt

    def e(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=CPU_DEVICE)

    gc = e(hnt, c, d, dtype=f32)
    binders: dict[str, Callable[[], BoundKernel]] = {
        "chunk_fwd_h_delta_varlen_helion": _bind_fwd_h_delta_varlen,
        "chunk_cumsum_gc_varlen_helion": functools.partial(
            _bind,
            "chunk_cumsum_gc_varlen_helion",
            (
                e(t_total, h, d),
                token_base,
                valid_len,
                gc,
                e(h, dtype=f32),
                e(h, d, dtype=f32),
                -5.0,
                True,
                True,
                True,
            ),
        ),
        "chunk_fwd_A_diag_anchored_varlen_helion": functools.partial(
            _bind,
            "chunk_fwd_A_diag_anchored_varlen_helion",
            (
                e(t_total, h, d),
                e(t_total, h, d),
                gc,
                token_base,
                valid_len,
                e(hnt, c, c),
                e(hnt, c, c),
                0.125,
                True,
            ),
        ),
        "chunk_fwd_wy_delta_varlen_helion": functools.partial(
            _bind,
            "chunk_fwd_wy_delta_varlen_helion",
            (
                e(t_total, h, d),
                e(t_total, h, dv),
                e(t_total, h),
                gc,
                e(hnt, c, c),
                token_base,
                valid_len,
                e(hnt, c, d),
                e(hnt, c, dv),
                e(hnt, c, d),
            ),
        ),
        "chunk_fwd_o_diag_anchored_varlen_helion": functools.partial(
            _bind,
            "chunk_fwd_o_diag_anchored_varlen_helion",
            (
                e(t_total, h, d),
                e(hnt, c, dv),
                gc,
                e(hnt, d, dv),
                e(hnt, c, c),
                token_base,
                valid_len,
                e(t_total, h, dv),
                0.125,
                True,
            ),
        ),
    }
    return binders


def test_varlen_engine_kernels_pin_their_admission() -> None:
    # The varlen chunked KDA forward's kernels with the dag's verdict: the
    # state pass (tensor loop bounds) and the token-major wy / o_diag kernels
    # (windows at a runtime token offset) are admitted; the gated cumsum and
    # the anchored gram (hl.arange sub-blocks, a l2norm reduction over a
    # window) decline.
    binders = _varlen_binders()
    admitted = {
        "chunk_fwd_h_delta_varlen_helion",
        "chunk_fwd_wy_delta_varlen_helion",
        "chunk_fwd_o_diag_anchored_varlen_helion",
    }
    verdicts = {}
    for name, binder in binders.items():
        spec = binder().config_spec
        verdicts[name] = spec.cute_mma_dag is not None
        assert ("cute_mma_dag" in spec.default_config().config) == verdicts[name], name
    assert verdicts == {name: name in admitted for name in binders}


def test_a_windowed_tile_keeps_its_packets_under_the_tma_ring() -> None:
    # The varlen wy transform at D=32, DV=64: the key loop's ring holds the
    # k tile -- a window, its rows at token_base[j] and live below
    # valid_len[j] -- and the derived g_cs tile, 64- and 128-byte rows.
    # Under ``cute_mma_dag_loads="tma"`` the g_cs tile rides a tensor map
    # while the windowed k tile keeps its zero-filled packets: a bulk
    # tensor copy lands its box whole, zero only past the tensor's bounds,
    # so it can neither start the rows at the window's runtime offset nor
    # zero those at and past its limit.  (Fetched by a tensor map at row 0
    # of the token axis, k read the batch's first tokens for every chunk:
    # the kda varlen forward's w and k_state were wrong against the
    # recurrent reference, test_linear_kda_varlen.)  The bind-time facts
    # agree: the k slot is not one the loads knob may move onto a tensor
    # map, so hinting it never pins the route.
    bf16, f32 = torch.bfloat16, torch.float32
    h, c, d, dv = 2, 64, 32, 64
    _offsets, token_base, valid_len, nt, t_total = _varlen_tables((2, 1, 3), c)
    hnt = h * nt

    def e(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=CPU_DEVICE)

    bound = _bind(
        "chunk_fwd_wy_delta_varlen_helion",
        (
            e(t_total, h, d),
            e(t_total, h, dv),
            e(t_total, h),
            e(hnt, c, d, dtype=f32),
            e(hnt, c, c),
            token_base,
            valid_len,
            e(hnt, c, d),
            e(hnt, c, dv),
            e(hnt, c, d),
        ),
    )
    program = _program(bound)
    assert {access.name for access in program.accesses if access.windows} == {
        "beta",
        "k",
        "v",
    }
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.tma_ring
    assert {
        access.name
        for access in program.accesses
        if access.eviction_slot in facts.tma_slots
    } == {"g_cs"}
    config = _config(32, 32, warps=4, stages=2, cute_mma_dag_loads="tma")
    with bound.env:
        assert (
            bound.config_spec.normalized_config(config).config["cute_mma_dag_loads"]
            == "tma"
        )
    source = _source(bound, config)
    body = _kernel_body(source)
    assert SIMT_FALLBACK not in body
    assert "md_tma0: cutlass.GridConstant[_helion_mma_dag_tc.cuda.TensorMap]" in body
    assert "md_tma1" not in source
    assert "'tma_count': 1" in source and "'tma0_elem': 4" in source
    copies = re.findall(
        r"cp_async_bulk_tensor_shared_cta_global\(.*?, (md_tma\d)\.get_ptr\(\), \[(.*?)\], ",
        body,
    )
    assert copies and all(name == "md_tma0" for name, _coords in copies)
    # g_cs's coordinates: the d tile, row 0, the chunk row; no window scalar.
    assert all("md_scalar" not in coords for _name, coords in copies)
    # k: packets from a base at the window's offset, zero-filled past the
    # window's limit and the tensor's end.
    assert "k.iterator.toint() + cutlass.Int64(md_scalar_1) * 128" in body
    assert re.search(
        r"md_zf = cutlass\.Int32\(16\) if md_pr\d* < md_scalar and md_scalar_1 \+ md_pr\d* < 381 else cutlass\.Int32\(0\)",
        body,
    )


def test_windowed_accesses_are_predicated_per_row() -> None:
    # The token-major varlen kernels read ``x[token_base[j] + hl.arange(C),
    # h, :]`` under ``extra_mask=(hl.arange(C) < valid_len[j])[:, None]``: a
    # window of the chunk axis at a runtime row offset, live below a runtime
    # limit.  The offset and the limit are uniform scalar loads; every row
    # of the window is predicated: the staged packets zero-fill, the per-lane
    # loads read zero and the stores are dropped past the limit or the
    # tensor's end.  The tcgen05 region renderer takes the windowed loads
    # (test_windowed_tiles_take_the_tcgen05_region_renderer).
    binders = _varlen_binders()
    wy = binders["chunk_fwd_wy_delta_varlen_helion"]()
    program = _program(wy)
    assert program.windowed and not program.windowed_stores
    with wy.env, wy.host_function:
        assert dag.mma_dag_facts(program).tcgen05 is not None
    windows = {
        access.name: [
            (position, window.offset.name, window.limit.name, window.extent)
            for position, window in access.windows
        ]
        for access in program.accesses
        if access.windows
    }
    # beta is a vector (its one dim windowed); k and v are the tile rows.
    assert windows == {
        "beta": [(0, "token_base", "valid_len", 381)],
        "k": [(0, "token_base", "valid_len", 381)],
        "v": [(0, "token_base", "valid_len", 381)],
    }
    # (The mask is decoded ahead of the subscripts: the limit comes first.)
    assert [access.name for access in program.scalar_loads] == [
        "valid_len",
        "token_base",
    ]
    body = _kernel_body(_source(wy, _config(64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert body.count(f"cutlass.Int32({RT}.ldg_s64(") == 2
    # The window offset moves every windowed base by whole rows.
    assert "md_base_3 = beta.iterator.toint() + cutlass.Int64(md_scalar_1) * 4" in body
    assert "k.iterator.toint() + cutlass.Int64(md_scalar_1) * 256" in body
    # k / v tiles: cp.async packets zero-filled past the limit / the end.
    assert f"{RT}.cp_async_16_zfill(" in body
    assert re.search(
        r"md_zf = cutlass\.Int32\(16\) if md_pr\d* < md_scalar and md_scalar_1 \+ md_pr\d* < 381 else cutlass\.Int32\(0\)",
        body,
    )
    # beta: per-lane predicated loads.
    assert f"{RT}.ldg_bf16_as_f32_pred(" in body
    assert re.search(
        r"md_ok = cutlass\.Int32\(1\) if md_L0_rowl < md_scalar and md_scalar_1 \+ md_L0_rowl < 381 else cutlass\.Int32\(0\)",
        body,
    )

    # o_diag: the window's rows are the output tile's rows too -- the store
    # is guarded; the fused q row norm is a keepdim reduction of a derived
    # fragment over the windowed tile (read along the rows: position 0).
    o_diag = binders["chunk_fwd_o_diag_anchored_varlen_helion"]()
    program = _program(o_diag)
    assert program.windowed and not program.loops
    windows = {
        access.name: [(p, w.offset.name, w.limit.name) for p, w in access.windows]
        for access in program.accesses
        if access.windows
    }
    assert windows == {
        "q": [(0, "token_base", "valid_len")],
        "out": [(0, "token_base", "valid_len")],
    }
    (reduce,) = program.reduces
    assert reduce.orient == "rows" and reduce.frag is not None
    assert reduce.frag in {derived.node for derived in program.derived}
    assert set(program.vec_positions.values()) == {0}
    body = _kernel_body(_source(o_diag, _config(warps=4, stages=2)))
    assert SIMT_FALLBACK not in body
    assert body.count(f"cutlass.Int32({RT}.ldg_s64(") == 2
    assert f"{RT}.ldg_b32_pred(" in body
    assert "if md_L0_rowl < md_scalar and md_scalar_1 + md_L0_rowl < 381:" in body
    assert f"{RT}.stg_b32(" in body


def test_windowed_tiles_take_the_tcgen05_region_renderer() -> None:
    # The varlen wy kernel under the tcgen05 mode: the k / v window tiles
    # ride the TMA ring with their boxes at the runtime row offset (the
    # third coordinate, the token dim), the rows at and past the limit (or
    # the tensor's end) read as zero in the convert passes that widen them
    # (``_Evaluator.tile_leaf``: both branches typed alike), and the beta
    # vector keeps its per-lane predicated loads.
    binders = _varlen_binders()
    wy = binders["chunk_fwd_wy_delta_varlen_helion"]()
    body = _kernel_body(
        _source(
            wy,
            helion.Config(
                block_sizes=[64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            ),
        )
    )
    assert "tcgen05_mma(" in body and SIMT_FALLBACK not in body
    assert body.count(f"cutlass.Int32({RT}.ldg_s64(") == 2
    boxes = re.findall(r"cp_async_bulk_tensor_shared_cta_global\(.*?\[(.*?)\]", body)
    windowed = [box for box in boxes if box.endswith("cutlass.Int32(md_scalar_1)")]
    assert len(windowed) >= 2 and all(
        re.fullmatch(r"cutlass\.Int32\(tile_offset_0 // \d+\)", box.split(", ")[1])
        for box in windowed
    )
    assert re.search(
        r"cutlass\.BFloat16\(md_lo_\d+\) if md_pr\w+ < md_scalar and md_scalar_1 \+ md_pr\w+ < 381 else cutlass\.BFloat16\(0\)",
        body,
    )
    assert f"{RT}.ldg_bf16_as_f32_pred(" in body
    # The window's rows never move the tile address: the offset is the
    # box coordinate, the base of every windowed tensor holds it too.
    assert "k.iterator.toint() + cutlass.Int64(md_scalar_1) * 256" in body


def _windowed_store_loop(
    v: torch.Tensor,
    A: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """A windowed store inside a tile loop: ``out[token_base[j] + arange,
    h, tile_dv] = A[r] @ v[r, :, tile_dv]`` under the row mask."""
    NT = token_base.size(0)
    C = hl.specialize(A.size(1))
    for tile_r in hl.tile(A.size(0), block_size=1):
        j = tile_r.begin % NT
        head = tile_r.begin // NT
        base = token_base[j]
        idx = hl.arange(C)
        valid = idx < valid_len[j]
        At = A[tile_r.begin, :, :]
        for tile_dv in hl.tile(v.size(2)):
            vt = v[tile_r.begin, :, tile_dv]
            hl.store(
                out,
                [base + idx, head, tile_dv],
                hl.dot(At, vt).to(out.dtype),
                extra_mask=valid[:, None],
            )


def _windowed_direct_operand(
    k: torch.Tensor,
    A: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """A windowed bf16 tile fed to a bf16 dot as it is (no pointwise DAG
    between the load and the operand)."""
    NT = token_base.size(0)
    C = hl.specialize(A.size(1))
    for tile_r in hl.tile(A.size(0), block_size=1):
        j = tile_r.begin % NT
        head = tile_r.begin // NT
        base = token_base[j]
        idx = hl.arange(C)
        valid = idx < valid_len[j]
        At = A[tile_r.begin, :, :]
        for tile_d in hl.tile(k.size(2)):
            kt = hl.load(k, [base + idx, head, tile_d], extra_mask=valid[:, None])
            out[tile_r.begin, :, tile_d] = hl.dot(At, kt).to(out.dtype)


def _windowed_probe_args(kernel: str) -> tuple[object, ...]:
    bf16 = torch.bfloat16
    h, c, d = 2, 64, 128
    _offsets, token_base, valid_len, nt, t_total = _varlen_tables((2, 1, 3), c)
    hnt = h * nt

    def e(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=CPU_DEVICE)

    if kernel == "store":
        return (e(hnt, c, d), e(hnt, c, c), token_base, valid_len, e(t_total, h, d))
    return (e(t_total, h, d), e(hnt, c, c), token_base, valid_len, e(hnt, c, d))


def test_windowed_stores_and_direct_window_operands_keep_the_warp_mode() -> None:
    # Fail closed: the tcgen05 renderers' staged / TMA stores do not drop a
    # window's rows past the limit, so a program with a windowed store has
    # no tcgen05 facts; neither has one whose windowed bf16 tile would feed
    # the MMA as it lands (no pointwise DAG, no widening: no pass masks its
    # rows).  The warp mode keeps both.
    store = _bind_fn(_windowed_store_loop, _windowed_probe_args("store"))
    program = _program(store)
    assert program.windowed_stores
    with store.env, store.host_function:
        assert dag.mma_dag_facts(program).tcgen05 is None
    with pytest.raises(InvalidConfig), store.env:
        store.config_spec.normalized_config(
            helion.Config(block_sizes=[64], cute_mma_dag="tcgen05")
        )
    body = _kernel_body(_source(store, _config(64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body and f"{RT}.mma_" in body

    direct = _bind_fn(_windowed_direct_operand, _windowed_probe_args("direct"))
    program = _program(direct)
    assert program.windowed and not program.windowed_stores
    (dot,) = program.dots
    assert dot.b.load is not None and dot.b.load.windows and not dot.b.pointwise
    with direct.env, direct.host_function:
        assert dag.mma_dag_facts(program).tcgen05 is None
    with pytest.raises(InvalidConfig), direct.env:
        direct.config_spec.normalized_config(
            helion.Config(block_sizes=[64], cute_mma_dag="tcgen05")
        )
    body = _kernel_body(_source(direct, _config(64, warps=4, stages=2)))
    assert SIMT_FALLBACK not in body and f"{RT}.cp_async_16_zfill(" in body


def _column_window_load(
    x: torch.Tensor,
    hmat: torch.Tensor,
    base_t: torch.Tensor,
    lim_t: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # A 2-D tile windowed along its second tile dim: ``x[r, :, base + idx]``.
    nt = base_t.size(0)
    d = hl.specialize(hmat.size(0))
    for tile_r in hl.tile(out.size(0), block_size=1):
        j = tile_r.begin % nt
        idx = hl.arange(d)
        valid = idx < lim_t[j]
        xt = torch.where(
            valid[None, :],
            hl.load(
                x,
                [tile_r.begin, slice(None), base_t[j] + idx],
                extra_mask=valid[None, :],
            ),
            0,
        )
        out[tile_r.begin, :, :] = hl.dot(xt, hmat[:, :]).to(out.dtype)


def _column_window_store(
    x: torch.Tensor,
    hmat: torch.Tensor,
    base_t: torch.Tensor,
    lim_t: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # A 2-D tile stored at a window along the contiguous dim.
    nt = base_t.size(0)
    d = hl.specialize(hmat.size(1))
    for tile_r in hl.tile(x.size(0), block_size=1):
        j = tile_r.begin % nt
        idx = hl.arange(d)
        valid = idx < lim_t[j]
        o = hl.dot(x[tile_r.begin, :, :], hmat[:, :])
        hl.store(
            out,
            [tile_r.begin, slice(None), base_t[j] + idx],
            o.to(out.dtype),
            extra_mask=valid[None, :],
        )


def _row_window_load(
    x: torch.Tensor,
    hmat: torch.Tensor,
    base_t: torch.Tensor,
    lim_t: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # The varlen layout: token rows of ``x [T, H, D]`` at a runtime offset.
    nt = base_t.size(0)
    c = hl.specialize(hmat.size(0))
    for tile_r in hl.tile(out.size(0), block_size=1):
        j = tile_r.begin % nt
        h = tile_r.begin // nt
        idx = hl.arange(c)
        valid = idx < lim_t[j]
        xt = torch.where(
            valid[:, None],
            hl.load(x, [base_t[j] + idx, h, slice(None)], extra_mask=valid[:, None]),
            0,
        )
        out[tile_r.begin, :, :] = hl.dot(xt, hmat[:, :]).to(out.dtype)


def test_tile_windows_are_rows_at_a_packet_aligned_stride() -> None:
    # A 2-D tile's window must be its first tile dim at a 16-byte multiple
    # of a row stride: the warp emitter moves the tile's packets and pairs
    # by whole rows and predicates them per row, so a window along the
    # contiguous dim (a layout read or written along its columns) would
    # misalign the cp.async packets at an odd offset and guard a pair store
    # by its first column only.  Declined, load and store alike -- the
    # kernels bind to the ordinary lowering without the knob -- and so is a
    # window on the second tile dim of a transposed layout (an aligned
    # stride, the wrong position) and a row window at a row stride that is
    # not a multiple of 16 bytes.  A vector's window is per element at any
    # stride (the wy kernel's beta: see the test above).
    bf16 = torch.bfloat16

    def e(*shape: int) -> torch.Tensor:
        return torch.empty(shape, dtype=bf16, device=CPU_DEVICE)

    base_t = torch.tensor([0, 64, 200, 136, 448], device=CPU_DEVICE)
    lim_t = torch.tensor([64, 40, 64, 1, 17], device=CPU_DEVICE)
    declined = (
        ("column load", _column_window_load, (e(10, 64, 512), e(64, 64))),
        ("column store", _column_window_store, (e(10, 64, 64), e(64, 64))),
        (
            "second tile dim at a 128-byte stride",
            _column_window_load,
            (e(10, 512, 64).transpose(1, 2), e(64, 64)),
        ),
    )
    for name, fn, (x, hmat) in declined:
        out = e(10, 64, 64) if fn is _column_window_load else e(10, 64, 512)
        spec = _bind_fn(fn, (x, hmat, base_t, lim_t, out)).config_spec
        assert spec.cute_mma_dag is None, name
        assert "cute_mma_dag" not in spec.default_config().config, name
    # The row window of the varlen layout is admitted at a 256-byte row
    # stride (2 heads of 64 bf16) ...
    t, h, c, d, dv = 500, 2, 64, 64, 64
    rows = _bind_fn(
        _row_window_load, (e(t, h, d), e(c, dv), base_t, lim_t, e(h * 5, c, dv))
    )
    assert rows.config_spec.cute_mma_dag is not None
    program = _program(rows)
    assert [
        (position, window.offset.name, window.limit.name, window.extent)
        for access in program.accesses
        for position, window in access.windows
    ] == [(0, "base_t", "lim_t", t)]
    body = _kernel_body(_source(rows, _config(warps=4, stages=2)))
    assert f"{RT}.cp_async_16_zfill(" in body
    # ... and declined over a padded view whose row stride is 2 * 66
    # elements = 264 bytes.
    padded = torch.empty((t, h, d + 2), dtype=bf16, device=CPU_DEVICE)[..., :d]
    spec = _bind_fn(
        _row_window_load, (padded, e(c, dv), base_t, lim_t, e(h * 5, c, dv))
    ).config_spec
    assert spec.cute_mma_dag is None
    assert "cute_mma_dag" not in spec.default_config().config


# --------------------------------------------------------------------------
# Side loads riding the ring (``cute_mma_dag_aux='ring'``)
# --------------------------------------------------------------------------


def _aux_ring_calls(body: str) -> tuple[int, int, int]:
    """(4-byte cp.async packets, shared reads, wait groups) of the aux ring."""
    return (
        body.count(f"{RT}.cp_async_4("),
        body.count(f"{RT}.lds_f32(md_aux0"),
        body.count("cute.arch.cp_async_wait_group("),
    )


def test_side_loads_ride_the_aux_ring_with_the_operand_tiles() -> None:
    # bwd_dh with the scalar decay: the gc row (the q pass's gate, read at
    # the thread's two packet rows) and the g_last scalar (the carry
    # rescale) vary with the chunk loop.  Under ``ring`` both are 4-byte
    # cp.async packets into the loop's aux ring -- a word per thread per
    # site (three), per stage -- issued with their iteration's operand
    # tiles: the prologue's ``stages - 1`` groups and the refill; the lanes
    # read this iteration's stage behind its wait, and no per-lane global
    # load is left in the loop.
    bound = _bind_bwd("scalar")
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.aux_candidates
    assert [
        (r.loop, r.kind, facts.axes[r.axis].kind if r.axis >= 0 else None)
        for r in facts.aux_loads
    ] == [(0, "across", dag.AXIS_RDIM), (0, "scalar", None)]
    assert spec.cute_mma_dag_aux is not None
    assert spec.cute_mma_dag_aux.choices == ("register", "ring")
    register = _kernel_body(_source(bound, _config(32, 64, warps=4, stages=4)))
    ring = _kernel_body(
        _source(bound, _config(32, 64, warps=4, stages=4, cute_mma_dag_aux="ring"))
    )
    assert "md_aux0" not in register
    assert _aux_ring_calls(register) == (0, 0, 1)
    # 3 prologue groups x 3 words, plus the refill's 3.
    assert _aux_ring_calls(ring) == (12, 3, 1)
    loop = _loop_and_after(ring)
    assert f"{RT}.lds_f32(md_aux0 + md_stage * 1536 + 0 + md_tidx * 4)" in loop
    assert (
        f"{RT}.cp_async_4(md_aux0 + (md_stage + 3) % 4 * 1536 + 0 + md_tidx * 4, "
        in loop
    )
    assert loop.index("cute.arch.cp_async_wait_group(2)") < loop.index(
        f"{RT}.lds_f32(md_aux0"
    )
    assert not _scalar_global_loads(loop)
    assert _scalar_global_loads(_loop_and_after(register))
    # The aux ring sits behind the operand rings and the staging buffers;
    # the planner allocates exactly the estimate.
    roots = [axis for axis in facts.axes if axis.kind == dag.AXIS_ROOT]
    sizes: dict[int, object] = {
        axis.block_id: 1 for axis in facts.axes if axis.kind == dag.AXIS_LEAD
    }
    sizes[roots[0].block_id], sizes[roots[1].block_id] = 32, 64
    resolved = [cast("int", facts.block_size(i, sizes)) for i in range(len(facts.axes))]
    plain = dag.mma_dag_smem_estimate(facts, resolved, 4, 4)
    staged = dag.mma_dag_smem_estimate(facts, resolved, 4, 4, aux="ring")
    assert staged == plain + 4 * 3 * 128 * 4
    assert f"alloc_smem(cutlass.Uint8, {plain}, alignment=128)" in register
    assert f"alloc_smem(cutlass.Uint8, {staged}, alignment=128)" in ring
    assert f"md_aux0 = md_s0 + {plain}" in ring
    # One stage: the iteration's own group carries its words, waited for in
    # full ahead of the reads.
    one = _kernel_body(
        _source(bound, _config(32, 64, warps=4, stages=1, cute_mma_dag_aux="ring"))
    )
    assert _aux_ring_calls(one) == (3, 3, 1)
    assert "cute.arch.cp_async_wait_group(0)" in _loop_and_after(one)
    assert not _scalar_global_loads(_loop_and_after(one))


def test_the_aux_knob_is_pinned_without_a_subject() -> None:
    # No side load varying with a ring loop (the vanilla recurrences: the
    # scale is a float argument), or the tensor-memory mode (its own
    # register prefetch): the knob is inert and pinned to the register
    # prefetch.
    for bound in (_bind_fwd(), _bind_bwd()):
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and not facts.aux_candidates
        config = _config(64, 64, warps=4, stages=2, cute_mma_dag_aux="ring").config
        with bound.env:
            bound.config_spec.normalize(config)
        assert config["cute_mma_dag_aux"] == "register"
    bound = _bind_bwd("scalar")
    config = helion.Config(
        block_sizes=[64, 32],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=4,
        cute_mma_dag_aux="ring",
    ).config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag"] == "tcgen05"
    assert config["cute_mma_dag_aux"] == "register"
    config = _config(32, 64, warps=4, stages=4, cute_mma_dag_aux="ring").config
    with bound.env:
        bound.config_spec.normalize(config)
    assert config["cute_mma_dag_aux"] == "ring"
    # A side load varying with an outer loop is read inside the inner one
    # (hoisted ahead of it): not a ring load.
    t = _tensors3(bhn=4, d=128, dv=128)
    gd = torch.empty((4, 128), dtype=torch.float32, device=CPU_DEVICE)
    outer = _bind_fn(_inner_varying_row_gate, (t["q"], t["h"], gd))
    facts = outer.config_spec.cute_mma_dag_facts
    assert facts is not None and not facts.aux_candidates
    # One varying with the inner loop is: the 64-element DV tile of the
    # gate vector per iteration, a 16-byte run per packet copied once per
    # CTA in the inner loop's frame (see
    # ``test_the_aux_ring_stages_vectors_along_the_contiguous_dim_once_per_cta``).
    gdv = torch.empty((4, 128), dtype=torch.float32, device=CPU_DEVICE)
    inner = _bind_fn(_inner_varying_gate, (t["q"], t["h"], gdv))
    facts = inner.config_spec.cute_mma_dag_facts
    assert facts is not None and len(facts.aux_loads) == 1
    (record,) = facts.aux_loads
    assert (record.loop, record.kind) == (1, "run")
    assert facts.axes[record.axis].kind == dag.AXIS_LOOP
    body = _kernel_body(
        _source(inner, _config(64, 64, warps=4, stages=2, cute_mma_dag_aux="ring"))
    )
    assert f"{RT}.cp_async_16(md_aux1 + " in body
    inner_loop = body[body.index("for md_i1 in") :]
    assert not _scalar_global_loads(inner_loop)


def test_the_first_warp_seed_has_an_aux_ring_twin() -> None:
    # Right behind the first warp seed: the same tile with the side loads
    # riding the ring; one such seed, and it normalizes as is.
    bound = _bind_bwd("scalar")
    spec = bound.config_spec
    seeds = [
        c.config for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag")
    ]
    first = next(i for i, c in enumerate(seeds) if c["cute_mma_dag"] == "warp")
    plain, twin = seeds[first], seeds[first + 1]
    assert "cute_mma_dag_aux" not in plain
    assert twin["cute_mma_dag_aux"] == "ring"
    assert {k: v for k, v in twin.items() if k != "cute_mma_dag_aux"} == plain
    assert sum(c.get("cute_mma_dag_aux") == "ring" for c in seeds) == 1
    with bound.env:
        normalized = spec.normalized_config(helion.Config.from_dict(twin))
    assert normalized.config["cute_mma_dag_aux"] == "ring"
    assert spec.default_config().config["cute_mma_dag_aux"] == "register"
    # No twin where no side load varies with a ring loop.
    seeds = [
        c.config
        for c in _bind_fwd().config_spec.compiler_seed_configs
        if c.config.get("cute_mma_dag")
    ]
    assert not any(c.get("cute_mma_dag_aux") == "ring" for c in seeds)


def _bind_fwd_scalar_unaligned(*, v_aligned: bool = False) -> BoundKernel:
    """The scalar-decay forward with k (and v, unless ``v_aligned``) sliced
    ``[..., 4:68]`` out of 80-wide rows: an 8-byte-aligned base cp.async
    cannot stage, so the operand's pass copies from global memory (the
    synchronous route)."""
    t = _tensors()
    k = torch.empty((2, 8, 64, 80), dtype=torch.bfloat16, device=CPU_DEVICE)[..., 4:68]
    v = t["v"]
    if not v_aligned:
        v = torch.empty((2, 8, 64, 80), dtype=torch.bfloat16, device=CPU_DEVICE)[
            ..., 4:68
        ]
    return _bind(
        "chunk_fwd_h_diag_fused",
        (k, v, t["g_last_s"], t["h0"], t["gc"], True, True, False, True),
    )


def test_a_loop_whose_ring_stages_nothing_has_no_aux_ring_to_ride() -> None:
    # k and v sliced [..., 4:68] out of 80-wide rows: 8-byte-aligned bases,
    # so both operand tiles take the synchronous lane pass and the chunk
    # loop's cp.async ring is empty.  The aux ring rides the ring (its
    # copies are issued with the ring tiles), so there is nothing for it to
    # ride: the facts carry no aux loads (the program-level records are
    # there), the knob is pinned to the register prefetch, the seed list
    # has no ring twin, and a hand-written ring plan declines instead of
    # failing inside the emitter.
    bound = _bind_fwd_scalar_unaligned()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and not facts.aux_candidates and not facts.aux_loads
    program = _program(bound)
    assert program.loop_has_ring(0)
    with bound.env:
        assert dag.mma_dag_aux_records(bound.env, program, program.loops[0])
        assert dag.mma_dag_ring_loops(bound.env, program) == frozenset()
        assert dag.mma_dag_aux_loads(bound.env, program) == {}
    config = _config(64, 64, warps=4, stages=2, cute_mma_dag_aux="ring").config
    with bound.env:
        spec.normalize(config)
    assert config["cute_mma_dag_aux"] == "register"
    body = _kernel_body(bound.to_code(helion.Config(**config)))
    assert f"{RT}.cp_async_4(" not in body and "md_aux0" not in body
    assert body == _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
    seeds = [
        c.config for c in spec.compiler_seed_configs if c.config.get("cute_mma_dag")
    ]
    assert seeds and not any(c.get("cute_mma_dag_aux") == "ring" for c in seeds)
    # Normalization's pins bypassed (the knob given a surface, the facts a
    # subject, the reason silenced): the planner declines on its own.
    raw = _config(64, 64, warps=4, stages=2, cute_mma_dag_aux="ring")
    spec.cute_mma_dag_aux = EnumFragment(choices=dag.MMA_DAG_AUX_CHOICES)
    spec.cute_mma_dag_facts = dataclasses.replace(facts, aux_candidates=True)
    try:
        with (
            patch(
                "helion.autotuner.config_spec.mma_dag_aux_reason",
                return_value=None,
            ),
            pytest.raises(BackendUnsupported, match="whose ring stages a tile"),
        ):
            bound.to_code(raw)
    finally:
        spec.cute_mma_dag_aux = None
        spec.cute_mma_dag_facts = facts
    # k alone under-aligned: v's tile rides the ring and the aux copies
    # with it (the k pass copies from global memory beside them); the same
    # aux loads as the aligned kernel's.
    bound = _bind_fwd_scalar_unaligned(v_aligned=True)
    facts = bound.config_spec.cute_mma_dag_facts
    aligned = _bind_fwd("scalar").config_spec.cute_mma_dag_facts
    assert facts is not None and aligned is not None
    assert facts.aux_loads == aligned.aux_loads and len(facts.aux_loads) == 2
    program = _program(bound)
    with bound.env:
        assert dag.mma_dag_ring_loops(bound.env, program) == frozenset({0})
    body = _kernel_body(
        _source(bound, _config(64, 64, warps=4, stages=2, cute_mma_dag_aux="ring"))
    )
    loop = _loop_and_after(body)
    assert f"{RT}.cp_async_4(md_aux0" in loop and f"{RT}.lds_f32(md_aux0" in loop
    assert f"{RT}.ldg_b32(" in loop and f"{RT}.cp_async_16(" in loop
    assert not _scalar_global_loads(loop)


def test_aux_rings_that_overflow_shared_memory_pin_the_knob_to_the_registers() -> None:
    # The capacity squeezed between the register and the ring estimates:
    # the repair keeps the tile and the stages and drops the route; strict
    # normalization names the knob.
    bound = _bind_bwd("scalar")
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    roots = [axis for axis in facts.axes if axis.kind == dag.AXIS_ROOT]
    sizes: dict[int, object] = {
        axis.block_id: 1 for axis in facts.axes if axis.kind == dag.AXIS_LEAD
    }
    sizes[roots[0].block_id], sizes[roots[1].block_id] = 32, 64
    resolved = [cast("int", facts.block_size(i, sizes)) for i in range(len(facts.axes))]
    plain = dag.mma_dag_smem_estimate(facts, resolved, 4, 4)
    assert dag.mma_dag_smem_estimate(facts, resolved, 4, 4, aux="ring") > plain + 16
    spec.cute_mma_dag_facts = dataclasses.replace(facts, smem_capacity=plain + 16)
    try:
        config = _config(32, 64, warps=4, stages=4, cute_mma_dag_aux="ring").config
        with bound.env:
            spec.normalize(config, _fix_invalid=True)
        assert config["cute_mma_dag_aux"] == "register"
        assert (config["block_sizes"], config["cute_mma_dag_stages"]) == ([32, 64], 4)
        config = _config(32, 64, warps=4, stages=4, cute_mma_dag_aux="ring").config
        with (
            bound.env,
            pytest.raises(InvalidConfig, match="cute_mma_dag_aux='ring'"),
        ):
            spec.normalize(config)
    finally:
        spec.cute_mma_dag_facts = facts


def test_a_row_the_ring_slot_would_serve_rides_the_aux_ring_under_ring() -> None:
    # full_gla's anchored forward: ``gc_last = gc[idx, i_t, C - 1, tile_d]``
    # is a row of the staged gc tile, so under ``register`` the ring slot
    # serves it behind the iteration barrier (``_ring_aliases``; the carry
    # rescale runs there), while under ``ring`` it is an fp32 vector the aux
    # ring copies once per CTA (``AuxRing.shared``) and the lanes read by
    # coordinate behind the barrier -- never both routes for one load.  No
    # scalar bf16 cast on either route.
    bound = _bind_fwd("anchored", has_h0=True, d=128, dv=128)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    assert [(r.loop, r.kind, facts.axes[r.axis].kind) for r in facts.aux_loads] == [
        (0, "shared", dag.AXIS_ROOT)
    ]
    alias_read = re.compile(
        rf"{RT}\.lds_f32\(md_d\d+ \+ md_stage \* 8192 \+ 63 \* 128 \+ "
    )
    for derived, row_reads in (("exchange", 2 + 4), ("pass", 2 + 8)):
        loops = {}
        for aux in ("register", "ring"):
            source = _source(
                bound,
                _config(
                    32,
                    128,
                    warps=8,
                    stages=4,
                    cute_mma_dag_derived=derived,
                    cute_mma_dag_aux=aux,
                ),
            )
            loops[aux] = _loop_and_after(_kernel_body(source))
        register, ring = loops["register"], loops["ring"]
        assert len(alias_read.findall(register)) == row_reads
        assert f"{RT}.cp_async_4(" not in register and "md_aux0" not in register
        assert not alias_read.findall(ring)
        assert ring.count(f"{RT}.cp_async_4(") == 1
        assert ring.count(f"{RT}.lds_f32(md_aux0 + md_stage * 128 + 0 + ") == row_reads
        assert f"{RT}.ldg_f32(" not in ring
        assert not _bf16_casts_of_computed_values(ring)
        # The rescale reads the shared row: behind the iteration barrier.
        rescale = re.search(r"\n\s+md_c0_0_0_0 = md_v", ring)
        assert rescale is not None
        assert rescale.start() > ring.index("cute.arch.barrier()")
        assert ring.count("cute.arch.barrier()") == 2


def test_the_rounding_root_is_found_through_views() -> None:
    # The q pass's value is ``(q * gate).to(bf16).transpose(-2, -1)``: the
    # pack rounds the fp32 products itself, so no scalar bf16 cast (and no
    # widening back) is left in the loop; the 16-bit intermediate of a
    # lossy chain (k -> f16 -> bf16) stays (see the lossy cast test).
    body = _kernel_body(
        _source(_bind_bwd("scalar"), _config(32, 64, warps=4, stages=4))
    )
    loop = _loop_and_after(body)
    assert not _bf16_casts_of_computed_values(loop)
    assert loop.count(f"{RT}.pack_bf16x2(") == 8
    assert loop.count("cute.arch.mul_packed_f32x2(") == 8
