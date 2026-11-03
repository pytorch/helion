"""Register-MMA row programs for gated (softmax-free) jagged attention.

The fused tcgen05 body of ``cute_flash_gated`` pays a fixed ~5 us per CTA
(TMEM allocation, three TMA descriptor prefetches, six pipelines, the QK ->
gate -> P -> PV mbarrier handoffs and the teardown) that no tile shape
removes; at small sequence lengths that fixed cost is the whole kernel.  This
module lowers the same matched program (``GatedAttentionMatch``: jagged row
loads, ``S = Q K^T``, an elementwise gate program, ``acc += P V``, a masked
store) to the latency-bound structure Triton reaches with 16-row tiles and
``tl.dot``: one CTA per (leading axes, query tile) with ``row_warps`` warps,
Q/K/V streamed through per-warp shared memory with zero-filling ``cp.async``
packets (Q rows past the jagged length the store-skip terms name, the query
extent or the tensor end, and K/V rows past the KV bound cost no read),
``ldmatrix`` operand fragments, ``mma.sync.m16n8k16`` with fp32 accumulators
in registers, and the gate program replayed on the S fragment rows.

* **Tiles.**  The query tile (16 or 32 rows) and KV tile (16 .. 128 keys) are
  the kernel's block sizes; a config with a 16- or 32-row query tile takes
  this body, the 64- and 128-row tiles keep the fused tcgen05 body.
* **Warps.**  ``cute_flash_gated_row_warps`` warps per CTA: up to one per
  16-row m-tile of the query tile (``m_split``), the rest split the KV
  chunks round robin (``k_split``) and combine their fp32 partial outputs
  through shared memory in fixed warp order (bit-identical across launches).
* **Staging.**  ``cute_flash_kv_stage`` K/V chunks per warp are in flight
  (every chunk is one ``cp.async`` group; the first ``kv_stage`` chunks are
  issued at kernel entry, so a KV range of at most ``kv_stage`` chunks is one
  memory round trip).
* **Gate.**  The gate program runs on vectors of at most
  ``GATED_ROW_GATE_WIDTH`` (8) elements holding one lane's share of one S row
  (pairs of adjacent columns eight apart, the ``mma.sync`` C layout;
  ``GatedChunkGeometry.paired``), one pass per 32 keys of the chunk, with the
  frontend's rounding points preserved exactly as in the fused body.  Each
  pass packs its P values straight into the PV A-fragments
  (``cvt.rn.bf16x2``), so the gate's live vectors and the packed P -- not
  the whole chunk's fp32 P -- are what a lane holds besides its fragments.
* **Numerics.**  Unlike the fused body, the per-KV-tile ``P @ V`` product is
  rounded to the frontend's product dtype (``GatedAttentionMatch.pv_dtype``,
  bf16 for ``acc += (P @ V)`` on bf16 rows) before the fp32 add, exactly as
  the Triton backend does.  Outputs are converted once at the store.
"""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from ...language import tile_ops
from .cute_flash import FLASH_KV_STAGE_KEY
from .cute_flash import _flash_graph_host_tensors
from .cute_flash import emit_flash_module_statements
from .cute_flash_gated import _GEOMETRY
from .cute_flash_gated import _GT_TARGETS
from .cute_flash_gated import _LT_TARGETS
from .cute_flash_gated import GATED_ROW_GATE_WIDTH
from .cute_flash_gated import GatedAttentionPlan
from .cute_flash_gated import _col_mask
from .cute_flash_gated import _gated_io_dtype_str
from .cute_flash_gated import _gated_smem_capacity
from .cute_flash_gated import _GatedEmitter
from .cute_flash_gated import _split_load_dependent
from .cute_flash_gated import _to_b
from .cute_flash_gated import _to_i
from .cute_flash_row_mma import _ROW_MMA_RUNTIME
from .cute_flash_row_mma import emit_flash_row_mma_module_statements
from .state import cute_codegen_state

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from .cute_flash_gated import GatedAttentionMatch

GATED_ROW_WRAPPER_KIND = "helion_flash_gated_row_mma"
GATED_ROW_WARPS_KEY = "cute_flash_gated_row_warps"
GATED_ROW_WARP_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
GATED_ROW_WARPS_DEFAULT = 2
# Query tiles the row programs take (the fused body owns 64 and 128).
GATED_ROW_Q_TILE_CHOICES: tuple[int, ...] = (16, 32)
GATED_ROW_KV_TILE_CHOICES: tuple[int, ...] = (16, 32, 64, 128)
GATED_ROW_KV_STAGE_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
GATED_ROW_KV_STAGE_DEFAULT = 4
GATED_ROW_HEAD_DIMS: tuple[int, ...] = (32, 64, 128)
# The row programs are seeded only on small grids: at most this many 16-row
# programs (leading axes times 16-row query tiles of the query extent's size
# hint).  Measured on jagged_hstu_attn (B200): at 64 programs (s0) the row
# program is 1.39x the Triton winner and 1.36x the fused body, at 512 (s1)
# 1.14x / 1.18x, at 2048 (s2: 8 sequences, 512 tokens, 8 heads, head dim 64)
# the fused tcgen05 body wins 1.44x over the best row config (16.1 vs 23.3 us).
# Larger grids keep the row tiles as search choices without a seed.
GATED_ROW_SEED_MAX_PROGRAMS = 1024
GATED_ROW_IO_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
# Registers per lane the program needs beyond the compiler's own temporaries
# (``gated_row_fragment_registers``): per owned m-tile the Q A-fragments
# (head_dim / 4), the fp32 accumulators (head_dim / 2), the chunk's S
# fragments (kv_tile / 2) and its packed P A-fragments (kv_tile / 4), plus one
# eight-register PV partial and the gate pass's live vectors (about six of
# ``GATED_ROW_GATE_WIDTH`` elements).  Measured on sm_100a with
# ``CUTE_DSL_KEEP=cubin`` and ``cuobjdump --dump-resource-usage`` plus the
# SASS STL/LDL count (kv_stage 1, every head dim x tile x 1/2/4 warps):
#   estimate <= 200 (up to head dim 128 x 64 keys, head dim 64 x 128 keys,
#     two head-dim-64 m-tiles x 32 keys): 68-255 registers, 0-12 spill
#     instructions (at most 24 bytes of stack);
#   estimate 248 (head dim 128 x 128 keys at one m-tile per warp, or two
#     head-dim-64 m-tiles x 64 keys): 255 registers and, for the head-dim-128
#     case, 188-206 spill instructions over 360-376 bytes of stack.
# The budget admits the first group and excludes the second.
GATED_ROW_REGISTER_BUDGET = 232
_GATED_ROW_SMEM_RESERVE_BYTES = 1024
_RT = _ROW_MMA_RUNTIME


class GatedRowSplit(NamedTuple):
    """How a CTA's warps share the work: ``m_split`` warps each own
    ``q_tile // m_split`` query rows, ``k_split`` warps per row group walk
    the KV chunks round robin."""

    m_split: int
    k_split: int


def gated_row_split(q_tile: int, warps: int) -> GatedRowSplit | None:
    if q_tile % 16 or warps < 1:
        return None
    m_tiles = q_tile // 16
    m_split = min(warps, m_tiles)
    if warps % m_split:
        return None
    return GatedRowSplit(m_split, warps // m_split)


def gated_row_fragment_registers(
    head_dim: int, q_tile: int, kv_tile: int, warps: int
) -> int:
    """Registers per lane the program holds across a KV chunk: per owned
    m-tile ``head_dim // 4`` for the Q A-fragments, ``head_dim // 2`` for the
    accumulators, ``kv_tile // 2`` for the S fragments and ``kv_tile // 4``
    for the packed P A-fragments, plus one eight-register PV partial and the
    gate pass's live vectors (see ``GATED_ROW_REGISTER_BUDGET``)."""
    split = gated_row_split(q_tile, warps)
    assert split is not None
    m_tiles = q_tile // split.m_split // 16
    gate_width = min(kv_tile // 4, GATED_ROW_GATE_WIDTH)
    return (
        m_tiles * (head_dim // 4 + head_dim // 2 + kv_tile // 2 + kv_tile // 4)
        + 8
        + 6 * gate_width
    )


def gated_row_warp_choices(head_dim: int, q_tile: int, kv_tile: int) -> tuple[int, ...]:
    """Warp counts whose fragment footprint fits the register budget."""
    return tuple(
        warps
        for warps in GATED_ROW_WARP_CHOICES
        if gated_row_split(q_tile, warps) is not None
        and gated_row_fragment_registers(head_dim, q_tile, kv_tile, warps)
        <= GATED_ROW_REGISTER_BUDGET
    )


def gated_row_warps_default(choices: Sequence[int]) -> int:
    if GATED_ROW_WARPS_DEFAULT in choices:
        return GATED_ROW_WARPS_DEFAULT
    return max(choices)


def gated_row_smem_bytes(
    *, head_dim: int, q_tile: int, kv_tile: int, warps: int, kv_stage: int
) -> int:
    """Static shared memory of one CTA: per warp its Q rows and ``kv_stage``
    K/V chunks (16-bit), plus the fp32 partial-output exchange when warps
    split the keys."""
    split = gated_row_split(q_tile, warps)
    assert split is not None
    rows_w = q_tile // split.m_split
    per_warp = (rows_w * head_dim + kv_stage * 2 * kv_tile * head_dim) * 2
    partials = warps * rows_w * (head_dim + 4) * 4 if split.k_split > 1 else 0
    return warps * per_warp + partials + _GATED_ROW_SMEM_RESERVE_BYTES


def gated_row_kv_stage_choices(
    head_dim: int,
    q_tile: int,
    kv_tile: int,
    warps: int,
    *,
    smem_capacity: int,
) -> tuple[int, ...]:
    """Chunk ring depths that fit shared memory at this tile and warp count."""
    return tuple(
        stage
        for stage in GATED_ROW_KV_STAGE_CHOICES
        if gated_row_smem_bytes(
            head_dim=head_dim,
            q_tile=q_tile,
            kv_tile=kv_tile,
            warps=warps,
            kv_stage=stage,
        )
        <= smem_capacity
    )


def gated_row_kv_stage_default(choices: Sequence[int]) -> int:
    below = [stage for stage in choices if stage <= GATED_ROW_KV_STAGE_DEFAULT]
    return max(below) if below else min(choices)


def gated_row_supported(match: GatedAttentionMatch) -> bool:
    """Whether the row programs host this match: the root form (one query
    tile per CTA, scalar row offset), 16-bit rows, head dims 32 .. 128."""
    return (
        match.form == "root"
        and match.io_dtype in GATED_ROW_IO_DTYPES
        and match.head_dim in GATED_ROW_HEAD_DIMS
        and not any(access.lane_all for access in (match.q, match.k, match.v, match.o))
    )


def gated_row_tile_surface(
    match: GatedAttentionMatch, *, smem_capacity: int
) -> tuple[
    dict[tuple[int, int], tuple[int, ...]],
    dict[tuple[int, int], tuple[int, ...]],
    dict[tuple[int, int, int], tuple[int, ...]],
]:
    """Per row-program tile shape: the ring depths legal at its smallest
    warp count (the most permissive, for the shared by-tile table), the warp
    counts, and the ring depths per (tile, warps)."""
    stages: dict[tuple[int, int], tuple[int, ...]] = {}
    warps_by_tile: dict[tuple[int, int], tuple[int, ...]] = {}
    stages_by_warps: dict[tuple[int, int, int], tuple[int, ...]] = {}
    if not gated_row_supported(match):
        return stages, warps_by_tile, stages_by_warps
    for q_tile in GATED_ROW_Q_TILE_CHOICES:
        for kv_tile in GATED_ROW_KV_TILE_CHOICES:
            legal = []
            for warps in gated_row_warp_choices(match.head_dim, q_tile, kv_tile):
                depths = gated_row_kv_stage_choices(
                    match.head_dim, q_tile, kv_tile, warps, smem_capacity=smem_capacity
                )
                if depths:
                    legal.append(warps)
                    stages_by_warps[(q_tile, kv_tile, warps)] = depths
            if not legal:
                continue
            stages[(q_tile, kv_tile)] = stages_by_warps[(q_tile, kv_tile, legal[0])]
            warps_by_tile[(q_tile, kv_tile)] = tuple(legal)
    return stages, warps_by_tile, stages_by_warps


def gated_row_seed_configs(
    block_size_lists: Sequence[Sequence[int]],
    *,
    tiles: Sequence[tuple[int, int]],
    warp_choices_by_tile: Mapping[tuple[int, int], Sequence[int]],
    stage_choices: Mapping[tuple[int, int, int], Sequence[int]],
    programs: int | None = None,
) -> list[Config]:
    """Generation-zero seeds of the row programs.

    Every row tile at four warps with two chunks in flight (the measured
    winner of the small jagged shapes: at 128 tokens the four key groups of a
    16-row tile each own at most two 16-key chunks, and a deeper ring only
    adds entry code), and the 16-row tiles also at the default two warps and
    at one warp with the default ring depth.  Every seed is legal for its
    tile (a depth or count the tile does not admit falls back to the nearest
    legal one).  ``programs`` is the grid's 16-row program count where it is
    known: none are seeded past ``GATED_ROW_SEED_MAX_PROGRAMS``.
    """
    from ...runtime.config import Config

    seeds: list[Config] = []
    if programs is not None and programs > GATED_ROW_SEED_MAX_PROGRAMS:
        return seeds
    for block_sizes, tile in zip(block_size_lists, tiles, strict=True):
        warp_choices = tuple(warp_choices_by_tile.get(tile, ()))
        if not warp_choices:
            continue
        candidates: list[tuple[int, int | None]] = [(4, 2)]
        if tile[0] == 16:
            candidates.extend(
                ((gated_row_warps_default(warp_choices), None), (1, None))
            )
        for warps, stage in candidates:
            if warps not in warp_choices:
                warps = (
                    max(w for w in warp_choices if w <= warps)
                    if any(w <= warps for w in warp_choices)
                    else min(warp_choices)
                )
            stages = tuple(stage_choices.get((*tile, warps), ()))
            if not stages:
                continue
            if stage is None or stage not in stages:
                stage = (
                    gated_row_kv_stage_default(stages)
                    if stage is None
                    else max((s for s in stages if s <= stage), default=min(stages))
                )
            seed = Config.from_dict(
                {
                    "block_sizes": list(block_sizes),
                    FLASH_KV_STAGE_KEY: stage,
                    GATED_ROW_WARPS_KEY: warps,
                }
            )
            if seed not in seeds:
                seeds.append(seed)
    return seeds


def detect_gated_row_plan(
    match: GatedAttentionMatch,
    env: CompileEnvironment,
    config: Config,
    *,
    q_tile: int,
    kv_tile: int,
) -> GatedAttentionPlan | None:
    """The row program of a config at a row-program tile shape, or None when
    the knobs are illegal for it."""
    if (
        q_tile not in GATED_ROW_Q_TILE_CHOICES
        or kv_tile not in GATED_ROW_KV_TILE_CHOICES
        or not gated_row_supported(match)
    ):
        return None
    warp_choices = gated_row_warp_choices(match.head_dim, q_tile, kv_tile)
    if not warp_choices:
        return None
    requested_warps = config.config.get(GATED_ROW_WARPS_KEY)
    if requested_warps is None:
        warps = gated_row_warps_default(warp_choices)
    elif isinstance(requested_warps, int) and requested_warps in warp_choices:
        warps = requested_warps
    else:
        return None
    stage_choices = gated_row_kv_stage_choices(
        match.head_dim,
        q_tile,
        kv_tile,
        warps,
        smem_capacity=_gated_smem_capacity(env),
    )
    if not stage_choices:
        return None
    requested_stage = config.config.get(FLASH_KV_STAGE_KEY)
    if requested_stage is None:
        kv_stage = gated_row_kv_stage_default(stage_choices)
    elif isinstance(requested_stage, int) and requested_stage in stage_choices:
        kv_stage = requested_stage
    else:
        return None
    return GatedAttentionPlan(match, kv_tile, kv_stage, 1, q_tile, row_warps=warps)


# ---------------------------------------------------------------------------
# Device body
# ---------------------------------------------------------------------------


def emit_gated_row_mma_device_body(
    df: DeviceFunction, plan: GatedAttentionPlan, *, names: dict[str, str]
) -> tuple[list[ast.AST], set[str]]:
    """Build the row-program body; returns the statements and the names of
    the tensor args the scalar programs read.  ``names`` maps q/k/v/o to the
    kernel argument names."""
    token = _GEOMETRY.set(plan.geometry)
    try:
        return _emit_gated_row_mma_device_body(df, plan, names=names)
    finally:
        _GEOMETRY.reset(token)


def _emit_gated_row_mma_device_body(
    df: DeviceFunction, plan: GatedAttentionPlan, *, names: dict[str, str]
) -> tuple[list[ast.AST], set[str]]:
    match = plan.match
    assert match.form == "root"
    hd = match.head_dim
    bn = plan.kv_tile
    rows = plan.q_tile
    warps = plan.row_warps
    stages = plan.kv_stage
    split = gated_row_split(rows, warps)
    assert split is not None
    m_split, k_split = split
    rows_w = rows // m_split
    m_tiles = rows_w // 16
    io_dtype = _gated_io_dtype_str(match.io_dtype)
    kind = "bf16" if match.io_dtype is torch.bfloat16 else "f16"
    mma = f"{_RT}.mma_{kind}"
    pack = f"{_RT}.pack_{kind}x2"
    elem = 2
    row_bytes = hd * elem
    cpr = hd // 8  # 16-byte packets per row
    k_steps = hd // 16  # QK^T k-steps over the head dim
    n_tiles = bn // 8  # S n-tiles (8 keys each)
    kk_steps = bn // 16  # PV k-steps (16 keys each)
    d_tiles = hd // 8  # O n-tiles (8 columns each)
    d_pairs = hd // 16  # ldmatrix.x4.trans loads per PV k-step
    stage_bytes = 2 * bn * row_bytes
    width = plan.geometry.chunk_cols  # gate elements per pass
    passes = (bn // 4) // width  # gate passes per (m-tile, row half)
    keys_per_pass = 4 * width
    part_stride = hd + 4  # fp32 words per row of the partial exchange
    # The XOR swizzle of the 16-byte packets inside a row keeps the eight row
    # reads of one ldmatrix phase on distinct bank groups: ``col ^ (row % 8)``
    # for rows of eight or more packets, ``col ^ ((row >> 1) & 3)`` for the
    # four-packet rows of the 32-wide head (rows two apart share banks).
    if cpr >= 8:
        swz_shift, swz_mask = 0, 7
    else:
        swz_shift, swz_mask = {4: (1, 3), 2: (2, 1)}[cpr]

    def swz(row: str, col: str) -> str:
        return (
            f"({row}) * {row_bytes} + (((({col}) ^ (((({row}) >> {swz_shift}) & "
            f"{swz_mask}))) * 16))"
        )

    pv_round: str | None
    if match.pv_dtype is torch.bfloat16:
        pv_round = "_helion_flash_rt._cvt_rn_bf16_hi({x})"
    elif match.pv_dtype is torch.float16:
        pv_round = "cutlass.Float32(cutlass.Float16({x}))"
    else:
        pv_round = None

    def acc_slot(mt: int, dt: int, i: int) -> int:
        return (mt * d_tiles + dt) * 4 + i

    emitter = _GatedEmitter(df, plan)
    root_extents = [
        emitter.extent_expr(block_id) for block_id in match.leading_block_ids
    ]
    decode_lines = ["flash_pid = cutlass.Int32(cute.arch.block_idx()[0])"]
    divisor = "cutlass.Int32(1)"
    for axis, extent in enumerate(root_extents):
        decode_lines.append(
            f"flash_axis{axis} = (flash_pid // {divisor}) % cutlass.Int32({extent})"
        )
        divisor = f"({divisor} * cutlass.Int32({extent}))"

    if match.row_offset is not None:
        row_base_expr = _to_i(emitter.emit(match.row_offset))
    else:
        row_base_expr = "cutlass.Int32(0)"
    q_extent_expr = emitter.q_extent_expr()
    # Query tiles are walked from the last to the first (heaviest first
    # under a causal KV bound), as in the fused body.
    m_tile_expr = (
        f"(flash_q_extent + cutlass.Int32({rows - 1})) // cutlass.Int32({rows})"
        f" - cutlass.Int32(1) - flash_pid // {divisor}"
    )
    kv_end_expr = _to_i(emitter.emit(match.kv_end))
    if match.q.lane_block_id is not None:
        lane_expr = emitter.tile_scalar_expr(
            tile_ops.tile_begin, match.q.lane_block_id
        ).expr
    else:
        assert match.q.lane_constant is not None
        lane_expr = f"cutlass.Int32({match.q.lane_constant})"
    predicate_value = emitter.emit(match.if_predicate)
    row_terms = [_to_b(emitter.emit(term)) for term in match.store_skip_terms]
    # The jagged length those terms compare the row with (``tile_q.index <
    # off[b + 1] - off[b]``, a CTA-level scalar): Q rows at or past it are
    # zero-filled instead of read -- the gate zeroes their P and the store
    # skips them, so nothing downstream sees the row.
    row_bounds: list[str] = []
    for term in match.store_skip_terms:
        if term.target in _LT_TARGETS:
            length = term.args[1]
        elif term.target in _GT_TARGETS:
            length = term.args[0]
        else:
            continue
        row_bounds.append(_to_i(emitter.emit(length)))
    v_end_expr = "flash_kv_end"
    if match.v_row_bound is not None:
        v_end_expr = (
            f"cutlass.min({_to_i(emitter.emit(match.v_row_bound))}, flash_kv_end)"
        )
    p_value = emitter.emit(match.p_node)
    p_expr = emitter.finish_elem_section(p_value)
    kv_end_terms = [kv_end_expr, *emitter.kv_end_bounds()]
    kv_end_min = kv_end_terms[0]
    for term in kv_end_terms[1:]:
        kv_end_min = f"cutlass.min({kv_end_min}, {term})"
    in_tile_mask = _col_mask("<", "(flash_kv_end - flash_chunk_base)")
    early_cta, late_cta = _split_load_dependent(emitter.sections["cta"])
    row_section = list(emitter.sections["row"])
    kviter_section = list(emitter.sections["kviter"])
    elem_section = list(emitter.sections["elem"])

    # -- generated program --------------------------------------------------
    lines: list[str] = []
    emit = lines.append
    lines.extend(
        (
            "tidx, _, _ = cute.arch.thread_idx()",
            "warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
            "grm_lane = tidx % 32",
            "grm_g = grm_lane // 4",
            "grm_c = grm_lane % 4",
            "grm_mi = grm_lane // 8",
            "grm_r = grm_lane % 8",
            *decode_lines,
            f"grm_s_q = cute.arch.alloc_smem({io_dtype}, {warps * rows_w * hd}, alignment=128)",
            f"grm_s_kv = cute.arch.alloc_smem({io_dtype}, {warps * stages * 2 * bn * hd}, alignment=128)",
            *(
                (
                    f"grm_s_part = cute.arch.alloc_smem(cutlass.Float32, {warps * rows_w * part_stride}, alignment=16)",
                    "grm_part = cutlass.Int32(grm_s_part.toint())",
                )
                if k_split > 1
                else ()
            ),
            *early_cta,
            *late_cta,
            f"flash_row_base = {row_base_expr}",
            f"flash_q_extent = {q_extent_expr}",
            "flash_n_items = cutlass.Int32(1)",
            "flash_item = cutlass.Int32(0)",
            f"flash_m_tile = {m_tile_expr}",
            f"flash_q_row0 = flash_m_tile * cutlass.Int32({rows})",
            f"flash_q_row_end = cutlass.min(flash_q_row0 + cutlass.Int32({rows}), flash_q_extent)",
            *emitter.sections["item"],
            f"flash_kv_end = {kv_end_min}",
            (
                "flash_num_kv_active = cutlass.max((flash_kv_end + cutlass.Int32("
                f"{bn - 1})) // cutlass.Int32({bn}), cutlass.Int32(0))"
            ),
            f"flash_lane = {lane_expr}",
            f"flash_body_active = {_to_b(predicate_value)}",
        )
    )

    body: list[str] = []
    b = body.append
    b(f"grm_ms = warp_idx % {m_split}" if m_split > 1 else "grm_ms = cutlass.Int32(0)")
    b(f"grm_ks = warp_idx // {m_split}" if k_split > 1 else "grm_ks = cutlass.Int32(0)")
    b(f"grm_row0 = flash_q_row0 + grm_ms * cutlass.Int32({rows_w})")
    b(f"grm_q_w = cutlass.Int32(grm_s_q.toint()) + warp_idx * {rows_w * row_bytes}")
    b(f"grm_kv_w = cutlass.Int32(grm_s_kv.toint()) + warp_idx * {stages * stage_bytes}")
    accesses = {"q": match.q, "k": match.k, "v": match.v, "o": match.o}
    for key, access in accesses.items():
        name = names[key]
        body.extend(
            (
                f"grm_{key}_rs = cutlass.Int64({name}.layout.stride[{access.row_dim}]) * {elem}",
                f"grm_{key}_ls = cutlass.Int64({name}.layout.stride[{access.lane_dim}]) * {elem}",
                f"grm_{key}_rows = cutlass.Int32({name}.layout.shape[{access.row_dim}])",
                f"grm_{key}_tensor = {name}.iterator.toint()",
            )
        )
    body.extend(
        (
            "grm_q_base = grm_q_tensor + cutlass.Int64(flash_row_base + grm_row0) * grm_q_rs + cutlass.Int64(flash_lane) * grm_q_ls",
            "grm_k_base = grm_k_tensor + cutlass.Int64(flash_row_base) * grm_k_rs + cutlass.Int64(flash_lane) * grm_k_ls",
            "grm_v_base = grm_v_tensor + cutlass.Int64(flash_row_base) * grm_v_rs + cutlass.Int64(flash_lane) * grm_v_ls",
            "grm_o_base = grm_o_tensor + cutlass.Int64(flash_row_base + grm_row0) * grm_o_rs + cutlass.Int64(flash_lane) * grm_o_ls",
            # Rows this warp loads: inside the query extent, the tensor and
            # the jagged length the store-skip terms name.
            "grm_q_lim = "
            + "".join("cutlass.min(" for _ in row_bounds)
            + "cutlass.min(flash_q_extent, grm_q_rows - flash_row_base)"
            + "".join(f", {bound})" for bound in row_bounds)
            + " - grm_row0",
            (
                f"grm_nchunks = cutlass.max((flash_num_kv_active - grm_ks + cutlass.Int32({k_split - 1}))"
                f" // cutlass.Int32({k_split}), cutlass.Int32(0))"
            ),
        )
    )

    def issue_rows(
        out: list[str],
        indent: str,
        *,
        smem_base: str,
        gmem_base: str,
        row_stride: str,
        tile_rows: int,
        limit: str,
        tensor_base: str,
    ) -> None:
        """cp.async the ``tile_rows`` rows (``cpr`` packets each) starting at
        ``gmem_base``: row ``r`` is read when ``r < limit``, else zero-filled
        without a global read (its address is clamped to the tensor base)."""
        packets = tile_rows * cpr
        for i in range(-(-packets // 32)):
            out.append(f"{indent}grm_idx = {i * 32} + grm_lane")
            inner = indent
            if packets - i * 32 < 32:
                out.append(f"{indent}if grm_idx < {packets}:")
                inner = indent + "    "
            out.extend(
                (
                    f"{inner}grm_irow = grm_idx // {cpr}",
                    f"{inner}grm_icol = grm_idx % {cpr}",
                    f"{inner}grm_ok = grm_irow < {limit}",
                    (
                        f"{inner}grm_src = cutlass.select_(grm_ok, {gmem_base} + "
                        f"cutlass.Int64(grm_irow) * {row_stride} + cutlass.Int64(grm_icol * 16), "
                        f"{tensor_base})"
                    ),
                    (
                        f"{inner}{_RT}.cp_async_16_zfill({smem_base} + {swz('grm_irow', 'grm_icol')}, "
                        "grm_src, cutlass.select_(grm_ok, cutlass.Int32(16), cutlass.Int32(0)))"
                    ),
                )
            )

    def issue_chunk(out: list[str], indent: str, *, stage: str, chunk: str) -> None:
        """cp.async the K and V rows of KV chunk ``chunk`` into ring slot ``stage``."""
        out.extend(
            (
                f"{indent}grm_sbase = grm_kv_w + ({stage}) * {stage_bytes}",
                f"{indent}grm_col0 = ({chunk}) * cutlass.Int32({bn})",
                f"{indent}grm_k_lim = cutlass.min(flash_kv_end, grm_k_rows - flash_row_base) - grm_col0",
                f"{indent}grm_v_lim = cutlass.min({v_end_expr}, grm_v_rows - flash_row_base) - grm_col0",
            )
        )
        issue_rows(
            out,
            indent,
            smem_base="grm_sbase",
            gmem_base="(grm_k_base + cutlass.Int64(grm_col0) * grm_k_rs)",
            row_stride="grm_k_rs",
            tile_rows=bn,
            limit="grm_k_lim",
            tensor_base="grm_k_tensor",
        )
        issue_rows(
            out,
            indent,
            smem_base=f"(grm_sbase + {bn * row_bytes})",
            gmem_base="(grm_v_base + cutlass.Int64(grm_col0) * grm_v_rs)",
            row_stride="grm_v_rs",
            tile_rows=bn,
            limit="grm_v_lim",
            tensor_base="grm_v_tensor",
        )

    # The Q tile joins the first chunk's cp.async group; the first ``stages``
    # chunks go out before any MMA (a group is committed per ring slot, empty
    # when the warp has no such chunk, so the wait counts stay uniform).
    issue_rows(
        body,
        "",
        smem_base="grm_q_w",
        gmem_base="grm_q_base",
        row_stride="grm_q_rs",
        tile_rows=rows_w,
        limit="grm_q_lim",
        tensor_base="grm_q_tensor",
    )
    for s in range(stages):
        b(f"if grm_nchunks > cutlass.Int32({s}):")
        issue_chunk(
            body, "    ", stage=str(s), chunk=f"grm_ks + cutlass.Int32({s * k_split})"
        )
        b("cute.arch.cp_async_commit_group()")
    b(f"cute.arch.cp_async_wait_group({stages - 1})")
    b("cute.arch.sync_warp()")
    # Q A-fragments of every owned m-tile and k-step.
    for mt in range(m_tiles):
        for st in range(k_steps):
            b(
                f"grm_qa_{mt}_{st}_0, grm_qa_{mt}_{st}_1, grm_qa_{mt}_{st}_2, grm_qa_{mt}_{st}_3 = "
                f"{_RT}.ldmatrix_x4(grm_q_w + {swz(f'{16 * mt} + (grm_mi % 2) * 8 + grm_r', f'{2 * st} + grm_mi // 2')})"
            )
    body.extend(
        (
            f"grm_acc = cute.make_rmem_tensor(({m_tiles * d_tiles * 4},), cutlass.Float32)",
            f"for grm_j in cutlass.range_constexpr({m_tiles * d_tiles * 4}):",
            "    grm_acc[grm_j] = cutlass.Float32(0.0)",
            f"grm_iota_frag = cute.make_rmem_tensor(({width},), cutlass.Int32)",
            f"grm_bit_frag = cute.make_rmem_tensor(({width},), cutlass.Int32)",
        )
    )
    for e, offset in enumerate(plan.geometry.column_offsets()):
        body.extend(
            (
                f"grm_iota_frag[{e}] = cutlass.Int32({offset})",
                f"grm_bit_frag[{e}] = cutlass.Int32(1) << cutlass.Int32({e})",
            )
        )
    body.extend(
        (
            "flash_col_iota = grm_iota_frag.load()",
            "flash_bit_iota = grm_bit_frag.load()",
            "grm_zero = cutlass.Float32(0.0)",
        )
    )

    def gate_rows(out: list[str], indent: str) -> None:
        """Replay the gate program on each (m-tile, row half) of the S
        fragments, ``width`` elements (``width // 2`` S n-tiles, 32 keys at
        the full width) per pass, and pack each pass's P values straight into
        the PV A-fragments ``grm_pa_{mt}_{kk}_{i}`` (i = 0, 2 from row g;
        1, 3 from row g + 8)."""
        for mt in range(m_tiles):
            for rh in range(2):
                out.extend(
                    (
                        f"{indent}flash_gate_row = grm_row0 + cutlass.Int32({16 * mt + 8 * rh}) + grm_g",
                        f"{indent}flash_row_in_tile = flash_gate_row < flash_q_extent",
                        *(f"{indent}{line}" for line in row_section),
                    )
                )
                for p in range(passes):
                    nt0 = p * (width // 2)
                    out.extend(
                        (
                            (
                                f"{indent}flash_chunk_base = flash_kv_col0 + "
                                f"cutlass.Int32({keys_per_pass * p}) + grm_c * cutlass.Int32(2)"
                            ),
                            f"{indent}grm_sv = cute.make_rmem_tensor(({width},), cutlass.Float32)",
                        )
                    )
                    out.extend(
                        f"{indent}grm_sv[{e}] = grm_s_{mt}_{nt0 + e // 2}_{2 * rh + e % 2}"
                        for e in range(width)
                    )
                    out.extend(
                        (
                            f"{indent}flash_s_raw_v = grm_sv.load()",
                            f"{indent}flash_gate_col_v = flash_col_iota + flash_chunk_base",
                            (
                                f"{indent}flash_elem_in_tile_m = cutlass.select_(flash_row_in_tile, "
                                f"{in_tile_mask}, cutlass.Int32(0))"
                            ),
                            *(f"{indent}{line}" for line in elem_section),
                            f"{indent}flash_p_v = {p_expr}",
                            f"{indent}grm_pv = cute.make_rmem_tensor(({width},), cutlass.Float32)",
                            f"{indent}grm_pv.store(flash_p_v)",
                        )
                    )
                    lo, hi = (0, 2) if rh == 0 else (1, 3)
                    for j in range(width // 4):
                        kk = nt0 // 2 + j
                        out.extend(
                            (
                                f"{indent}grm_pa_{mt}_{kk}_{lo} = {pack}(grm_pv[{4 * j}], grm_pv[{4 * j + 1}])",
                                f"{indent}grm_pa_{mt}_{kk}_{hi} = {pack}(grm_pv[{4 * j + 2}], grm_pv[{4 * j + 3}])",
                            )
                        )

    def compute_chunk(out: list[str], indent: str) -> None:
        # S^T fragments: grm_s_{mt}_{nt}_{i}; lane (g, c) holds i=0: S[row g][key 8nt+2c],
        # i=1: [..][8nt+2c+1], i=2: S[row g+8][8nt+2c], i=3: [..][8nt+2c+1].
        for mt in range(m_tiles):
            for nt in range(n_tiles):
                out.extend(f"{indent}grm_s_{mt}_{nt}_{i} = grm_zero" for i in range(4))
        for nt in range(n_tiles):
            out.append(f"{indent}grm_krow = {8 * nt} + grm_r")
            for stp in range(k_steps // 2):
                out.append(
                    f"{indent}grm_b0, grm_b1, grm_b2, grm_b3 = {_RT}.ldmatrix_x4("
                    f"grm_kb + {swz('grm_krow', f'{4 * stp} + grm_mi')})"
                )
                for mt in range(m_tiles):
                    s = f"grm_s_{mt}_{nt}"
                    for st, (b0, b1) in (
                        (2 * stp, ("grm_b0", "grm_b1")),
                        (2 * stp + 1, ("grm_b2", "grm_b3")),
                    ):
                        q = f"grm_qa_{mt}_{st}"
                        out.append(
                            f"{indent}{s}_0, {s}_1, {s}_2, {s}_3 = {mma}({q}_0, {q}_1, {q}_2, {q}_3, "
                            f"{b0}, {b1}, {s}_0, {s}_1, {s}_2, {s}_3)"
                        )
        gate_rows(out, indent)
        # PV, one pair of O n-tiles at a time; the chunk's partial product is
        # rounded to the frontend's product dtype before the fp32 add.
        for dp in range(d_pairs):
            for mt in range(m_tiles):
                out.extend(
                    f"{indent}grm_o_{mt}_{n}_{i} = grm_zero"
                    for n in range(2)
                    for i in range(4)
                )
            for kk in range(kk_steps):
                out.extend(
                    (
                        f"{indent}grm_vrow = {16 * kk} + (grm_mi % 2) * 8 + grm_r",
                        (
                            f"{indent}grm_b0, grm_b1, grm_b2, grm_b3 = {_RT}.ldmatrix_x4_trans("
                            f"grm_vb + {swz('grm_vrow', f'{2 * dp} + grm_mi // 2')})"
                        ),
                    )
                )
                for mt in range(m_tiles):
                    a = f"grm_pa_{mt}_{kk}"
                    for n, (b0, b1) in (
                        (0, ("grm_b0", "grm_b1")),
                        (1, ("grm_b2", "grm_b3")),
                    ):
                        o = f"grm_o_{mt}_{n}"
                        out.append(
                            f"{indent}{o}_0, {o}_1, {o}_2, {o}_3 = {mma}({a}_0, {a}_1, {a}_2, {a}_3, "
                            f"{b0}, {b1}, {o}_0, {o}_1, {o}_2, {o}_3)"
                        )
            for mt in range(m_tiles):
                for n in range(2):
                    for i in range(4):
                        slot = acc_slot(mt, 2 * dp + n, i)
                        value = f"grm_o_{mt}_{n}_{i}"
                        if pv_round is not None:
                            value = pv_round.format(x=value)
                        out.append(
                            f"{indent}grm_acc[{slot}] = grm_acc[{slot}] + {value}"
                        )

    b("for grm_i in cutlass.range(grm_nchunks, unroll=1):")
    loop: list[str] = []
    ind = "    "
    loop.extend(
        (
            # Chunk i is cp.async group i: with ``stages`` groups committed
            # ahead of the loop and one per iteration, waiting for all but the
            # last ``stages - 1`` groups completes it.
            f"{ind}cute.arch.cp_async_wait_group({stages - 1})",
            f"{ind}cute.arch.sync_warp()",
            f"{ind}grm_stage = grm_i % cutlass.Int32({stages})"
            if stages > 1
            else f"{ind}grm_stage = cutlass.Int32(0)",
            f"{ind}grm_kb = grm_kv_w + grm_stage * {stage_bytes}",
            f"{ind}grm_vb = grm_kb + {bn * row_bytes}",
            f"{ind}flash_kv = grm_ks + grm_i * cutlass.Int32({k_split})",
            f"{ind}flash_kv_col0 = flash_kv * cutlass.Int32({bn})",
            f"{ind}flash_kv_col_end = cutlass.min(flash_kv_col0 + cutlass.Int32({bn}), flash_kv_end)",
            *(f"{ind}{line}" for line in kviter_section),
        )
    )
    compute_chunk(loop, ind)
    # Every lane finished reading the slot before the refill lands in it.
    loop.extend(
        (
            f"{ind}cute.arch.sync_warp()",
            f"{ind}if grm_i + cutlass.Int32({stages}) < grm_nchunks:",
        )
    )
    issue_chunk(
        loop,
        ind + "    ",
        stage="grm_stage",
        chunk=f"grm_ks + (grm_i + cutlass.Int32({stages})) * cutlass.Int32({k_split})",
    )
    loop.append(f"{ind}cute.arch.cp_async_commit_group()")
    body.extend(loop)

    def store_rows(out: list[str], indent: str) -> None:
        """Convert and store this warp's rows: one 4-byte packet per (row,
        n-tile) per lane; rows the frontend masks at the store are skipped."""
        store_terms = [
            "flash_row_in_tile",
            *row_terms,
            "(flash_o_row < grm_o_rows)",
        ]
        for mt in range(m_tiles):
            for rh in range(2):
                out.extend(
                    (
                        f"{indent}flash_gate_row = grm_row0 + cutlass.Int32({16 * mt + 8 * rh}) + grm_g",
                        f"{indent}flash_row_in_tile = flash_gate_row < flash_q_extent",
                        *(f"{indent}{line}" for line in row_section),
                        f"{indent}flash_o_row = flash_row_base + flash_gate_row",
                        f"{indent}grm_store_ok = {' & '.join(store_terms)}",
                        f"{indent}if grm_store_ok:",
                        (
                            f"{indent}    grm_oaddr = grm_o_base + cutlass.Int64({16 * mt + 8 * rh} + grm_g) * grm_o_rs"
                            f" + cutlass.Int64(grm_c * {2 * elem})"
                        ),
                    )
                )
                out.extend(
                    f"{indent}    {_RT}.stg_b32(grm_oaddr + {8 * dt * elem}, "
                    f"{pack}(grm_acc[{acc_slot(mt, dt, 2 * rh)}], grm_acc[{acc_slot(mt, dt, 2 * rh + 1)}]))"
                    for dt in range(d_tiles)
                )

    if k_split > 1:
        # Warps sharing a row group publish fp32 partials; the key-group-0
        # warp sums them in fixed order and stores.
        b(f"grm_part_w = grm_part + warp_idx * {rows_w * part_stride * 4}")
        for mt in range(m_tiles):
            for dt in range(d_tiles):
                for rh in range(2):
                    b(
                        f"{_RT}.sts_v2_f32(grm_part_w + ((cutlass.Int32({16 * mt + 8 * rh}) + grm_g) * {part_stride} "
                        f"+ {8 * dt} + 2 * grm_c) * 4, grm_acc[{acc_slot(mt, dt, 2 * rh)}], grm_acc[{acc_slot(mt, dt, 2 * rh + 1)}])"
                    )
        b("cute.arch.barrier()")
        b("if grm_ks == cutlass.Int32(0):")
        for ks in range(1, k_split):
            b(
                f"    grm_part_p = grm_part + (grm_ms + cutlass.Int32({ks * m_split})) * {rows_w * part_stride * 4}"
            )
            for mt in range(m_tiles):
                for dt in range(d_tiles):
                    for rh in range(2):
                        lo, hi = acc_slot(mt, dt, 2 * rh), acc_slot(mt, dt, 2 * rh + 1)
                        body.extend(
                            (
                                f"    grm_pa, grm_pb = {_RT}.lds_v2_f32(grm_part_p + ((cutlass.Int32({16 * mt + 8 * rh}) + grm_g) * {part_stride} + {8 * dt} + 2 * grm_c) * 4)",
                                f"    grm_acc[{lo}] = grm_acc[{lo}] + grm_pa",
                                f"    grm_acc[{hi}] = grm_acc[{hi}] + grm_pb",
                            )
                        )
        store_rows(body, "    ")
    else:
        store_rows(body, "")

    emit("if flash_body_active:")
    lines.extend(textwrap.indent("\n".join(body), "    ").split("\n"))
    source = "\n".join(lines) + "\n"
    return list(ast.parse(source).body), set(emitter.tensor_names)


# ---------------------------------------------------------------------------
# Codegen entry
# ---------------------------------------------------------------------------


def codegen_gated_row_mma(cg: GenerateAST, plan: GatedAttentionPlan) -> bool:
    """Replace the device body with the gated row program (see
    ``cute_flash_gated.codegen_gated_attention_flash`` for the fused twin)."""
    df = cg.device_function
    match = plan.match
    host_tensors = _flash_graph_host_tensors(df.codegen.codegen_graphs)
    accesses = {"q": match.q, "k": match.k, "v": match.v, "o": match.o}
    names: dict[str, str] = {}
    for key, access in accesses.items():
        fake = host_tensors.get(access.name)
        if fake is None:
            return False
        arg = df.tensor_arg(fake, prefer_name=access.name)
        if (
            arg.fake_value.ndim != 3
            or arg.fake_value.dtype != match.io_dtype
            or not arg.fake_value.is_contiguous()
            or int(arg.fake_value.shape[2]) != match.head_dim
        ):
            return False
        names[key] = arg.name
    body, scalar_tensor_names = emit_gated_row_mma_device_body(df, plan, names=names)
    emit_flash_module_statements(cg)
    emit_flash_row_mma_module_statements(cg)
    codegen_state = cute_codegen_state(cg)
    codegen_state.wrapper_plans.append(
        {
            "kind": GATED_ROW_WRAPPER_KIND,
            "q_name": names["q"],
            "k_name": names["k"],
            "v_name": names["v"],
            "o_name": names["o"],
            "head_dim": match.head_dim,
            "q_tile": plan.q_tile,
            "kv_tile": plan.kv_tile,
            "kv_stage": plan.kv_stage,
            "row_warps": plan.row_warps,
            "dtype": _gated_io_dtype_str(match.io_dtype),
        }
    )
    df.placeholder_args.update(names.values())
    df.placeholder_args.update(scalar_tensor_names)
    codegen_state.declare_matmul()
    df.cute_state.flash.threads = plan.threads
    df.body = body
    df.preamble = []
    return True


__all__ = [
    "GATED_ROW_HEAD_DIMS",
    "GATED_ROW_KV_STAGE_CHOICES",
    "GATED_ROW_KV_TILE_CHOICES",
    "GATED_ROW_Q_TILE_CHOICES",
    "GATED_ROW_SEED_MAX_PROGRAMS",
    "GATED_ROW_WARPS_DEFAULT",
    "GATED_ROW_WARPS_KEY",
    "GATED_ROW_WARP_CHOICES",
    "GATED_ROW_WRAPPER_KIND",
    "GatedRowSplit",
    "codegen_gated_row_mma",
    "detect_gated_row_plan",
    "emit_gated_row_mma_device_body",
    "gated_row_fragment_registers",
    "gated_row_kv_stage_choices",
    "gated_row_kv_stage_default",
    "gated_row_seed_configs",
    "gated_row_smem_bytes",
    "gated_row_split",
    "gated_row_supported",
    "gated_row_tile_surface",
    "gated_row_warp_choices",
    "gated_row_warps_default",
]
