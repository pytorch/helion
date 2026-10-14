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
import importlib
import os
import re
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401

import helion
from helion._compiler.autotuner_heuristics.cute_mma_dag import CuteMmaDagHeuristic
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute.cute_mma_dag_codegen import _Layout
from helion._compiler.cute.cute_warp_mma_gemm import _swizzled_chunk
from helion._compiler.cute.cute_warp_mma_gemm import warp_mma_warp_layout
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
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


def _program(bound: BoundKernel) -> dag.MmaDagProgram:
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return program


def _config(
    *block_sizes: int, warps: int, stages: int, **extra: object
) -> helion.Config:
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
    # The sm100 fake target promotes the first seed to the autotune-off default.
    assert spec.compiler_default_config is not None
    assert spec.compiler_default_config.config["cute_mma_dag"] == "warp"
    assert bound.config_spec.default_config().config["cute_mma_dag"] == "warp"
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
    assert default["cute_mma_dag"] == "warp" and default["block_sizes"][0] == 1
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
    engine = _engine()
    t = _tensors3()
    beta = torch.empty((16, 64), dtype=torch.float32, device=CPU_DEVICE)
    wy = helion.kernel(
        engine.chunk_fwd_wy_delta_helion.fn, backend="cute", static_shapes=True
    )  # type: ignore[attr-defined]
    other = wy.bind((t["q"], t["v"], beta, None, None, None, False, False))
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


def test_a_carry_read_after_its_in_place_update_is_not_mine() -> None:
    # The dot accumulates into the carry's registers in place, so a later
    # read of the carry-in value would see the updated state: not mine.
    @helion.kernel(backend="cute", static_shapes=True)
    def stale_carry(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, ...]:
        bh, n, c, d = k.shape
        c = hl.specialize(c)
        out = torch.empty((bh, n, d, d), dtype=k.dtype, device=k.device)
        out2 = torch.empty((bh, n, c, d), dtype=k.dtype, device=k.device)
        for tile_bh, tile_d, tile_dv in hl.tile([bh, d, d], block_size=[1, None, None]):
            idx = tile_bh.id
            acc = hl.zeros([tile_d, tile_dv], dtype=torch.float32)
            for i in hl.grid(n):
                out[idx, i, tile_d, tile_dv] = acc.to(out.dtype)
                k_i = k[idx, i, :, tile_d]
                v_i = v[idx, i, :, tile_dv]
                new = hl.dot(k_i.transpose(-2, -1), v_i, acc=acc)
                proj = hl.dot(k_i, acc.to(k.dtype))
                out2[idx, i, :, tile_dv] = proj.to(out2.dtype)
                acc = new
        return out, out2

    t = _tensors()
    bound = stale_carry.bind((t["k"], t["v"]))
    assert bound.config_spec.cute_mma_dag is None
    assert "cute_mma_dag" not in bound.config_spec.default_config().config


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
    assert "md_x3 = md_s0 + " in smem
    assert smem.count(f"{RT}.sts_b32(") == 16
    assert "alloc_smem(cutlass.Uint8, 65536, alignment=128)" in smem
    assert (
        smem.index(f"{RT}.sts_b32(")
        < smem.rindex("cute.arch.barrier()")
        < smem.rindex(f"{RT}.ldmatrix_x4(")
    )
    wide = _kernel_body(_source(_bind_fwd_o(), _config(1, 128, 128, warps=8, stages=2)))
    # o [64, 128] and attn [64, 64] get different warp grids (4 x 2 each,
    # but different atom counts); attn's wn = 2 forbids the identity route.
    assert "md_L1_rowl =" in wide
    assert "md_x3 = md_s0 + " in wide and "md_aw" not in wide
    assert wide.count(f"{RT}.sts_b32(") == 8
    assert "alloc_smem(cutlass.Uint8, 155648, alignment=128)" in wide


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
    assert "alloc_smem(cutlass.Uint8, 122880, alignment=128)" in body


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


def test_other_engine_kernels_stay_untouched() -> None:
    engine = _engine()
    t = _tensors3()
    beta = torch.empty((16, 64), dtype=torch.float32, device=CPU_DEVICE)
    a_inv = torch.empty((16, 64, 64), dtype=torch.float32, device=CPU_DEVICE)
    kernels = {
        "chunk_fwd_wy_delta_helion": (
            t["q"],
            t["v"],
            beta,
            None,
            None,
            None,
            False,
            False,
        ),
        "chunk_bwd_wy_dL_delta_helion": (
            t["q"],
            t["v"],
            beta,
            a_inv,
            t["dh"][:, :64],
            t["dh"][:, :64],
            None,
            False,
        ),
        # Nested tile loops / fragment reductions are not mine.
        "chunk_bwd_dqkw_delta_helion": (
            t["q"],
            t["q"],
            t["h"],
            t["v"],
            t["v"],
            t["v"],
            t["h"],
            None,
            None,
            False,
            0.125,
        ),
        "chunk_bwd_dk_delta_helion": (
            t["q"],
            a_inv,
            t["dh"][:, :64],
            beta,
            t["dh"][:, :64],
        ),
    }
    for name, args in kernels.items():
        fn = getattr(engine, name).fn
        bound = helion.kernel(fn, backend="cute", static_shapes=True).bind(args)
        assert bound.config_spec.cute_mma_dag is None, name
        assert "cute_mma_dag" not in bound.config_spec.default_config().config


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
    assert spec.default_config().config["cute_mma_dag"] == "warp"
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
        assert bound.config_spec.default_config().config["cute_mma_dag"] == "warp"
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
    assert len(facts.live_groups) == len(_program(bound).steps) + 1
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
    assert gated.config_spec.default_config().config["cute_mma_dag_warps"] == 8
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
    default = dstate.config_spec.default_config().config
    assert default["block_sizes"] == [32] and default["cute_mma_dag_warps"] == 8


def test_fp32_snapshots_seed_the_direct_store_path() -> None:
    # bwd_dh snapshots the fp32 carry: 8-byte pairs either way, so the
    # direct pair stores skip the staging round trip; fwd_h's bf16
    # snapshot keeps the 16-byte packets of the staging tile.
    bwd = _bind_bwd().config_spec
    assert bwd.default_config().config["cute_mma_dag_store"] == "direct"
    assert all(
        c.config["cute_mma_dag_store"] == "direct"
        for c in bwd.compiler_seed_configs
        if c.config.get("cute_mma_dag") == "warp"
    )
    fwd = _bind_fwd().config_spec
    assert fwd.default_config().config["cute_mma_dag_store"] == "staged"
    body = _kernel_body(_source(_bind_bwd(), bwd.default_config()))
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
        (torch.int32, False),
    ),
)
def test_side_loads_need_a_floating_dtype_the_lane_loads_read(
    dtype: torch.dtype, admitted: bool
) -> None:
    # The per-lane scalar / vector loads read f32 / f16 / bf16; anything else
    # (here a float64 or int32 decay) is not mine rather than a mistyped 16-bit
    # load.
    t = _tensors()
    s = torch.empty((2, 8), dtype=dtype, device=CPU_DEVICE)
    bound = _bind_fn(_scalar_decay_recurrence, (t["k"], t["v"], s))
    assert (bound.config_spec.cute_mma_dag is not None) is admitted
    if admitted:
        body = _kernel_body(_source(bound, _config(64, 64, warps=4, stages=2)))
        load = "ldg_f32" if dtype == torch.float32 else "ldg_bf16_as_f32"
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
    assert (
        _bind_fn(_two_dots_carry_read_after, (t["k"], t["v"])).config_spec.cute_mma_dag
        is None
    )


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
    assert (
        "cutlass.Int64(md_L0_rowl) * 128 + cutlass.Int64(md_L0_coll) * 2," in epilogue
    )
    assert "* 128 + cutlass.Int64(md_L0_coll) * 128" not in epilogue
    assert epilogue.count(f"{RT}.stg_b32(") == 16 + 16
    # A [C, C] load as an operand: the positional strides find its
    # contiguous dim, so it is staged with the row stride of C elements.
    a_in = torch.empty((16, 64, 64), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind_fn(_cc_load_operand, (t["q"], t["q"], t["v"], t["h"], a_in))
    assert bound.config_spec.cute_mma_dag is not None
    body = _kernel_body(_source(bound, _config(1, 64, 64, warps=4, stages=2)))
    assert f"{RT}.cp_async_16(md_t2 + md_prow" in body
    assert "cutlass.Int64(md_prow_" in body and ") * 128 + cutlass.Int64(md_pch" in body


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
