"""The ``row_mma`` flash pipeline family: register-MMA row programs.

One CTA owns ``row_tile_m`` (8 or 16) query rows; its ``row_warps`` warps split
the keys evenly and each warp runs the whole attention of its key range in
registers with warp-level ``mma.sync.m16n8k16`` instructions:

* ``S^T = K Q^T`` with K rows as the A operand and ``Q^T`` as the B operand,
  so the eight query rows of an octet are the MMA's N = 8 and no M padding is
  wasted;
* exact fp32 softmax (full-row max, MUFU exp2, fp32 row sum) on the S^T
  fragments, the row reductions being shuffles over the eight lanes that hold
  one query;
* ``O^T = V^T P^T`` with ``V^T`` read by ``ldmatrix.trans`` and ``P^T`` built
  from the probabilities by ``movmatrix``;
* Q, K and V stream through XOR-swizzled shared memory with ``cp.async``
  packets issued at kernel entry (keys beyond one chunk stream through a
  chunk loop, double buffered where the shared-memory budget allows, with an
  online-softmax rescale between chunks), and every MMA sits behind a
  ``cp.async.wait_group`` so ptxas cannot interleave the loads with the MMAs;
* the warps publish their partial ``(m, l, O)`` through shared memory and the
  CTA combines them in fixed warp order (deterministic, bit-identical across
  launches), writes bf16/fp16 output rows with 16-byte stores and the base-2
  LSE;
* a fused row epilogue (``flash_row_epilogue.py``) runs in that combine
  epilogue: every thread owns one row's 16-byte output packet (eight
  consecutive fp32 columns after the combine), evaluates the program's
  pointwise passes on them, completes each head_dim reduction with a
  fixed-order butterfly over the lanes sharing the row, and stages its aux
  row packets through shared memory with ``cp.async`` issued at kernel entry.

It is the latency-bound small-grid structure measured in round 4 of the
flash-small-shape work: at ``(1, 4, 256, 64)`` it ran 3.10-3.20 us device span
against 3.62-3.68 us for the Triton winner and 4.55-4.62 us for the 64-row
tcgen05 one-pass tile.  It uses no TMA, TMEM or tcgen05.
"""

from __future__ import annotations

import ast
import math
import textwrap
from typing import TYPE_CHECKING

import torch

from .flash_row_epilogue import emit_row_epilogue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .cute_flash import FlashRowEpilogueEmit

ROW_MMA_FAMILY = "row_mma"
ROW_MMA_PLAN_KIND = "helion_flash_row_mma"
ROW_MMA_WARP_CHOICES: tuple[int, ...] = (2, 4, 8)
ROW_MMA_TILE_M_CHOICES: tuple[int, ...] = (8, 16)
ROW_MMA_DEFAULT_WARPS = 4
ROW_MMA_DEFAULT_TILE_M = 8
ROW_MMA_HEAD_DIMS: tuple[int, ...] = (64, 128)
# Keys per shared-memory chunk (one K and one V slice of this many rows per
# warp per stage); the 128-wide head halves it so two stages of K and V stay
# inside the shared-memory budget for the common warp counts.
_ROW_MMA_CHUNK_KEYS = {64: 64, 128: 32}
# Static shared memory the family may claim per CTA (the DSL opt-in maximum is
# 227 KiB; the margin covers the allocator's alignment padding).
ROW_MMA_SMEM_BUDGET = 200 * 1024
# The row programs are searched only on small grids: at most this many row
# programs at the default 8-row tile, about one wave of two CTAs per SM on a
# 148-SM device. Every program re-streams its head's K and V through shared
# memory, so beyond one wave the programs only add traffic the 128-row tcgen05
# tiles already amortize; larger grids leave the family to explicit configs.
ROW_MMA_SEARCH_MAX_ROW_PROGRAMS = 256
_ROW_MMA_RUNTIME = "_helion_flash_rowmma"
_ROW_MMA_PREAMBLE = (
    f"import helion._compiler.cute._flash_row_mma_runtime as {_ROW_MMA_RUNTIME}\n"
)


def row_mma_supported(
    *,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    has_kv_tile_pruning: bool,
    requires_ws_overlap: bool,
    small_biased_candidate: bool,
    plain_row_body: bool,
    has_row_epilogue: bool,
    has_score_modifiers: bool = False,
) -> bool:
    """Whether the row-program family can run this attention.

    Dense rows only (the score plan carries no mask, bias or other modifier:
    ``plain_row_body`` is False for both a modifier and a fused row epilogue,
    so a row epilogue is admitted only when ``has_score_modifiers`` says the
    scores themselves are plain), 16-bit inputs, head dims 64 and 128.  A
    fused row epilogue runs in the combine epilogue (every program the
    detector accepts fits: its values are per-row, its reductions complete
    over the lanes sharing a row).  The sequence length does not participate:
    longer key ranges stream through the chunk loop, so the family is legal
    in every length class.
    """
    return (
        not has_score_modifiers
        and (plain_row_body or has_row_epilogue)
        and not is_causal
        and not has_kv_tile_pruning
        and not requires_ws_overlap
        and not small_biased_candidate
        and dtype in (torch.float16, torch.bfloat16)
        and head_dim in ROW_MMA_HEAD_DIMS
    )


def row_mma_chunk_keys(
    head_dim: int, keys_per_warp: int, cap: int | None = None
) -> int:
    """Keys per shared-memory chunk: the largest multiple of 16 that divides
    the warp's key range without exceeding ``cap`` (the per-head default)."""
    chunk = min(_ROW_MMA_CHUNK_KEYS[head_dim] if cap is None else cap, keys_per_warp)
    while keys_per_warp % chunk != 0:
        chunk -= 16
    assert chunk >= 16
    return chunk


def row_mma_epilogue_reps(*, head_dim: int, row_tile_m: int, row_warps: int) -> int:
    """Passes of the combine epilogue: thread -> (row, 16-byte output packet)
    pairs, ``row_tile_m * head_dim / 8`` of them over ``32 * row_warps``
    threads (threads beyond the tile repeat a row's work)."""
    return max(1, -(-(row_tile_m * (head_dim // 8)) // (32 * row_warps)))


_ROW_MMA_DTYPE_BYTES = {
    "cutlass.Float32": 4,
    "cutlass.Float16": 2,
    "cutlass.BFloat16": 2,
}


def _row_mma_dtype_bytes(dtype: str) -> int:
    """Element size of a ``cutlass.*`` dtype name of a fused-epilogue aux row."""
    return _ROW_MMA_DTYPE_BYTES[dtype]


def row_mma_aux_smem_bytes(
    *,
    head_dim: int,
    row_tile_m: int,
    row_warps: int,
    aux_dtypes: Sequence[str],
) -> int:
    """Shared memory of the fused row epilogue's aux rows (``cutlass.*`` dtype
    names): every epilogue thread stages the eight-column packet(s) of each
    aux row it reads into its own slot (one slot per thread and epilogue
    pass, so a thread only waits for its own ``cp.async`` groups and no
    barrier is needed)."""
    slots = row_mma_epilogue_reps(
        head_dim=head_dim, row_tile_m=row_tile_m, row_warps=row_warps
    ) * (32 * row_warps)
    return sum(slots * 8 * _row_mma_dtype_bytes(dtype) for dtype in aux_dtypes)


def row_mma_smem_bytes(
    *,
    head_dim: int,
    row_tile_m: int,
    row_warps: int,
    chunk: int,
    stages: int,
    aux_bytes: int = 0,
) -> int:
    """Static shared memory of one CTA: per-warp Q tile and K/V stages, the
    partial-O exchange (padded fp32 rows), the row statistics and the fused
    row epilogue's aux slots (``aux_bytes``, see ``row_mma_aux_smem_bytes``)."""
    row_bytes = head_dim * 2
    per_warp = row_tile_m * row_bytes + stages * 2 * chunk * row_bytes
    partials = row_warps * row_tile_m * (head_dim + 4) * 4
    stats = row_warps * 2 * row_tile_m * 4
    return row_warps * per_warp + partials + stats + aux_bytes


def row_mma_plan(
    *,
    seq: int,
    head_dim: int,
    row_warps: int,
    row_tile_m: int,
    aux_bytes: int = 0,
) -> tuple[int, int] | None:
    """``(chunk keys, stages)`` of the K/V staging, or None when the shape
    cannot be split (every warp needs whole 16-key tiles, every CTA whole
    query octets) or no staging fits the shared-memory budget next to the
    ``aux_bytes`` of a fused row epilogue.

    The widest chunk comes first (a key range that fits one chunk needs no
    chunk loop and a single stage: the measured small-grid structure); a
    longer range is double buffered, down to the smallest chunk, and a single
    stage is the last resort for the widest head with many warps and rows.
    """
    if seq % row_tile_m != 0 or seq % (16 * row_warps) != 0:
        return None
    keys_per_warp = seq // row_warps
    chunks: list[int] = []
    cap: int | None = None
    while True:
        chunk = row_mma_chunk_keys(head_dim, keys_per_warp, cap)
        chunks.append(chunk)
        if chunk <= 16:
            break
        cap = chunk - 16

    def fits(chunk: int, stages: int) -> bool:
        return (
            row_mma_smem_bytes(
                head_dim=head_dim,
                row_tile_m=row_tile_m,
                row_warps=row_warps,
                chunk=chunk,
                stages=stages,
                aux_bytes=aux_bytes,
            )
            <= ROW_MMA_SMEM_BUDGET
        )

    for chunk in chunks:
        stages = 1 if keys_per_warp <= chunk else 2
        if fits(chunk, stages):
            return chunk, stages
    for chunk in chunks:
        if keys_per_warp > chunk and fits(chunk, 1):
            return chunk, 1
    return None


def row_mma_search_grid(*, num_bh: int | None, num_kv: int) -> bool:
    """Return whether the grid is small enough to search the row programs.

    ``num_bh`` is the collapsed batch*heads count and ``num_kv`` the number of
    128-key tiles, so ``num_bh * num_kv`` 128-row tiles become
    ``16 * num_bh * num_kv`` programs at the default tile. An unknown batch
    is not searched; explicit configs stay legal on any grid.
    """
    if num_bh is None:
        return False
    row_programs = (num_bh * 128 * num_kv) // ROW_MMA_DEFAULT_TILE_M
    return row_programs <= ROW_MMA_SEARCH_MAX_ROW_PROGRAMS


def row_mma_shape_supported(
    *,
    seq: int,
    head_dim: int,
    row_warps: int,
    row_tile_m: int,
    aux_bytes: int = 0,
) -> bool:
    return (
        row_mma_plan(
            seq=seq,
            head_dim=head_dim,
            row_warps=row_warps,
            row_tile_m=row_tile_m,
            aux_bytes=aux_bytes,
        )
        is not None
    )


def emit_flash_row_mma_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module import of the row-program runtime helpers."""
    if getattr(cg, "_helion_flash_row_mma_module_emitted", False):
        return
    cg._helion_flash_row_mma_module_emitted = True  # type: ignore[attr-defined]
    for stmt in ast.parse(_ROW_MMA_PREAMBLE).body:
        cg.module_statements.append(stmt)


def emit_flash_row_mma_device_body(
    *,
    q_name: str,
    k_name: str,
    v_name: str,
    o_name: str,
    lse_name: str | None,
    num_bh: int,
    seq: int,
    head_dim: int,
    io_dtype: str,
    scale_log2: float,
    lse_scale: float,
    row_warps: int,
    row_tile_m: int,
    relu_output: bool,
    row_epilogue: FlashRowEpilogueEmit | None = None,
    kv_group: int = 1,
) -> list[ast.AST]:
    """Render the row-program kernel body for one (batch*heads, seq, head_dim)
    problem with contiguous ``(B, S, D)`` inputs.

    ``row_epilogue`` carries a fused row program with its aux tensors named
    as kernel parameters of the same ``(B, S, D)`` geometry and its symbolic
    scalar expressions; the program replaces the identity store.
    ``kv_group`` is the grouped-query divisor: K/V carry ``num_bh // kv_group``
    entries and the program reads them at batch head ``rm_bh // kv_group``.
    """
    aux_dtypes = () if row_epilogue is None else tuple(row_epilogue.aux_dtypes)
    aux_elem_bytes = [_row_mma_dtype_bytes(dtype) for dtype in aux_dtypes]
    aux_bytes = row_mma_aux_smem_bytes(
        head_dim=head_dim,
        row_tile_m=row_tile_m,
        row_warps=row_warps,
        aux_dtypes=aux_dtypes,
    )
    plan = row_mma_plan(
        seq=seq,
        head_dim=head_dim,
        row_warps=row_warps,
        row_tile_m=row_tile_m,
        aux_bytes=aux_bytes,
    )
    assert plan is not None
    chunk, stages = plan
    rows = row_tile_m
    octets = rows // 8
    warps = row_warps
    threads = 32 * warps
    kpw = seq // warps  # keys per warp
    nchunk = kpw // chunk
    row_bytes = head_dim * 2
    cpr = head_dim // 8  # 16-byte packets per row
    k_steps = head_dim // 16  # QK^T k-steps over the head dim
    d_tiles = head_dim // 16  # PV d-tiles (M-tiles of O^T)
    m_tiles = chunk // 16  # 16-key tiles per chunk (QK^T M-tiles == PV k-steps)
    part_stride = head_dim + 4  # fp32 row stride of the partial-O exchange
    stage_bytes = 2 * chunk * row_bytes
    # Combine-epilogue passes: thread -> (row, 16-byte column packet).
    reps = row_mma_epilogue_reps(
        head_dim=head_dim, row_tile_m=row_tile_m, row_warps=row_warps
    )
    rt = _ROW_MMA_RUNTIME
    kind = "bf16" if io_dtype == "cutlass.BFloat16" else "f16"
    mma = f"{rt}.mma_{kind}"
    pack = f"{rt}.pack_{kind}x2"

    def swz(row: str, col: str) -> str:
        """Byte offset of 16-byte packet ``col`` of row ``row`` under the
        XOR-by-row swizzle that keeps ldmatrix's eight row reads on
        distinct bank groups."""
        return f"({row}) * {row_bytes} + ((({col}) ^ (({row}) % 8)) * 16)"

    def issue_tile(
        out: list[str],
        indent: str,
        *,
        smem_base: str,
        gmem_base: str,
        tile_rows: int,
    ) -> None:
        for i in range(tile_rows * cpr // 32):
            out.extend(
                (
                    f"{indent}rm_idx = {i * 32} + rm_lane",
                    f"{indent}rm_irow = rm_idx // {cpr}",
                    f"{indent}rm_icol = rm_idx % {cpr}",
                    (
                        f"{indent}{rt}.cp_async_16({smem_base} + {swz('rm_irow', 'rm_icol')}, "
                        f"{gmem_base} + cutlass.Int64(rm_irow * {row_bytes} + rm_icol * 16))"
                    ),
                )
            )

    def issue_kv(out: list[str], indent: str, *, stage: str, chunk_index: str) -> None:
        """cp.async the K and V rows of chunk ``chunk_index`` into ``stage``."""
        out.extend(
            (
                f"{indent}rm_kv_base = rm_kv_w + ({stage}) * {stage_bytes}",
                f"{indent}rm_kv_off = cutlass.Int64({chunk_index}) * {chunk * row_bytes}",
            )
        )
        issue_tile(
            out,
            indent,
            smem_base="rm_kv_base",
            gmem_base="(rm_k_slice + rm_kv_off)",
            tile_rows=chunk,
        )
        issue_tile(
            out,
            indent,
            smem_base=f"(rm_kv_base + {chunk * row_bytes})",
            gmem_base="(rm_v_slice + rm_kv_off)",
            tile_rows=chunk,
        )

    def compute_chunk(out: list[str], indent: str, *, stage: str, first: bool) -> None:
        """QK^T, softmax (online across chunks) and PV of the chunk in ``stage``.

        ``first`` chunks initialise the carried statistics and accumulators as
        SSA values; later chunks read and write the ``rm_m``/``rm_l``/``rm_o``
        register tensors so the carried state survives the dynamic chunk loop.
        """
        out.extend(
            (
                f"{indent}rm_kb = rm_kv_w + ({stage}) * {stage_bytes}",
                f"{indent}rm_vb = rm_kb + {chunk * row_bytes}",
            )
        )
        # S^T tiles: rm_s_{o}_{mt}_{i}; lane (g, c) holds i=0: S[q 2c][key 16mt+g],
        # i=1: S[q 2c+1][key 16mt+g], i=2: S[q 2c][key 16mt+8+g], i=3: S[q 2c+1][..+8+g].
        for mt in range(m_tiles):
            for o in range(octets):
                out.extend(f"{indent}rm_s_{o}_{mt}_{i} = rm_zero" for i in range(4))
            out.append(f"{indent}rm_krow = {16 * mt} + (rm_mi % 2) * 8 + rm_r")
            for st in range(k_steps):
                out.append(
                    f"{indent}rm_a0, rm_a1, rm_a2, rm_a3 = {rt}.ldmatrix_x4("
                    f"rm_kb + {swz('rm_krow', f'{2 * st} + rm_mi // 2')})"
                )
                for o in range(octets):
                    s = f"rm_s_{o}_{mt}"
                    out.append(
                        f"{indent}{s}_0, {s}_1, {s}_2, {s}_3 = {mma}(rm_a0, rm_a1, rm_a2, rm_a3, "
                        f"rm_qb_{o}_{st}_0, rm_qb_{o}_{st}_1, {s}_0, {s}_1, {s}_2, {s}_3)"
                    )
        # Row statistics of the chunk for queries A = 2c and B = 2c+1 of every octet.
        for o in range(octets):
            for half, idx in (("a", (0, 2)), ("b", (1, 3))):
                terms = [f"rm_s_{o}_{mt}_{i}" for mt in range(m_tiles) for i in idx]
                out.append(f"{indent}rm_cm_{o}_{half} = {terms[0]}")
                out.extend(
                    f"{indent}rm_cm_{o}_{half} = cute.arch.fmax(rm_cm_{o}_{half}, {term})"
                    for term in terms[1:]
                )
                out.append(
                    f"{indent}rm_cm_{o}_{half} = {rt}.octet_max(rm_cm_{o}_{half})"
                )
                if first:
                    out.append(f"{indent}rm_m_{o}_{half} = rm_cm_{o}_{half}")
                else:
                    out.extend(
                        (
                            f"{indent}rm_mo_{o}_{half} = rm_m[{2 * o + (half == 'b')}]",
                            f"{indent}rm_m_{o}_{half} = cute.arch.fmax(rm_mo_{o}_{half}, rm_cm_{o}_{half})",
                            (
                                f"{indent}rm_alpha_{o}_{half} = cute.arch.exp2("
                                f"(rm_mo_{o}_{half} - rm_m_{o}_{half}) * rm_scale)"
                            ),
                        )
                    )
                out.append(
                    f"{indent}rm_neg_{o}_{half} = rm_zero - rm_m_{o}_{half} * rm_scale"
                )
        # p = exp2(s * scale - m * scale), summed in a fixed order.
        for o in range(octets):
            for mt in range(m_tiles):
                s = f"rm_s_{o}_{mt}"
                out.extend(
                    (
                        (
                            f"{indent}rm_x0, rm_x1 = cute.arch.fma_packed_f32x2(({s}_0, {s}_1), "
                            f"(rm_scale, rm_scale), (rm_neg_{o}_a, rm_neg_{o}_b))"
                        ),
                        (
                            f"{indent}rm_x2, rm_x3 = cute.arch.fma_packed_f32x2(({s}_2, {s}_3), "
                            f"(rm_scale, rm_scale), (rm_neg_{o}_a, rm_neg_{o}_b))"
                        ),
                    )
                )
                out.extend(
                    f"{indent}rm_p_{o}_{mt}_{i} = cute.arch.exp2(rm_x{i})"
                    for i in range(4)
                )
            for half, idx in (("a", (0, 2)), ("b", (1, 3))):
                terms = [f"rm_p_{o}_{mt}_{i}" for mt in range(m_tiles) for i in idx]
                out.append(f"{indent}rm_cl_{o}_{half} = {terms[0]}")
                out.extend(
                    f"{indent}rm_cl_{o}_{half} = rm_cl_{o}_{half} + {term}"
                    for term in terms[1:]
                )
                out.append(
                    f"{indent}rm_cl_{o}_{half} = {rt}.octet_sum(rm_cl_{o}_{half})"
                )
                if first:
                    out.append(f"{indent}rm_l_{o}_{half} = rm_cl_{o}_{half}")
                else:
                    out.append(
                        f"{indent}rm_l_{o}_{half} = rm_l[{2 * o + (half == 'b')}] * "
                        f"rm_alpha_{o}_{half} + rm_cl_{o}_{half}"
                    )
            # P^T B-fragments of PV k-step mt: transpose the (keys 16mt..+7) and
            # (keys 16mt+8..+15) 8x8 tiles of the octet's probabilities.
            for mt in range(m_tiles):
                p = f"rm_p_{o}_{mt}"
                out.extend(
                    (
                        f"{indent}rm_pb_{o}_{mt}_0 = {rt}.movmatrix_trans({pack}({p}_0, {p}_1))",
                        f"{indent}rm_pb_{o}_{mt}_1 = {rt}.movmatrix_trans({pack}({p}_2, {p}_3))",
                    )
                )
        # O^T accumulators: rm_o_{o}_{dt}_{i}; lane (g, c) holds i=0: O[q 2c][d 16dt+g],
        # i=1: O[q 2c+1][d 16dt+g], i=2: O[q 2c][d 16dt+8+g], i=3: O[q 2c+1][d 16dt+8+g].
        for o in range(octets):
            for dt in range(d_tiles):
                for i in range(4):
                    half = "a" if i in (0, 2) else "b"
                    if first:
                        out.append(f"{indent}rm_o_{o}_{dt}_{i} = rm_zero")
                    else:
                        out.append(
                            f"{indent}rm_o_{o}_{dt}_{i} = rm_o[{(o * d_tiles + dt) * 4 + i}] * rm_alpha_{o}_{half}"
                        )
        for st in range(m_tiles):
            out.append(f"{indent}rm_vrow = {16 * st} + (rm_mi // 2) * 8 + rm_r")
            for dt in range(d_tiles):
                out.append(
                    f"{indent}rm_a0, rm_a1, rm_a2, rm_a3 = {rt}.ldmatrix_x4_trans("
                    f"rm_vb + {swz('rm_vrow', f'{2 * dt} + rm_mi % 2')})"
                )
                for o in range(octets):
                    acc = f"rm_o_{o}_{dt}"
                    out.append(
                        f"{indent}{acc}_0, {acc}_1, {acc}_2, {acc}_3 = {mma}(rm_a0, rm_a1, rm_a2, rm_a3, "
                        f"rm_pb_{o}_{st}_0, rm_pb_{o}_{st}_1, {acc}_0, {acc}_1, {acc}_2, {acc}_3)"
                    )

    def spill_state(out: list[str], indent: str) -> None:
        """Write the SSA statistics and accumulators into the carried tensors."""
        for o in range(octets):
            for half in ("a", "b"):
                slot = 2 * o + (half == "b")
                out.extend(
                    (
                        f"{indent}rm_m[{slot}] = rm_m_{o}_{half}",
                        f"{indent}rm_l[{slot}] = rm_l_{o}_{half}",
                    )
                )
            out.extend(
                f"{indent}rm_o[{(o * d_tiles + dt) * 4 + i}] = rm_o_{o}_{dt}_{i}"
                for dt in range(d_tiles)
                for i in range(4)
            )

    def reload_state(out: list[str], indent: str) -> None:
        for o in range(octets):
            for half in ("a", "b"):
                slot = 2 * o + (half == "b")
                out.extend(
                    (
                        f"{indent}rm_m_{o}_{half} = rm_m[{slot}]",
                        f"{indent}rm_l_{o}_{half} = rm_l[{slot}]",
                    )
                )
            out.extend(
                f"{indent}rm_o_{o}_{dt}_{i} = rm_o[{(o * d_tiles + dt) * 4 + i}]"
                for dt in range(d_tiles)
                for i in range(4)
            )

    lines: list[str] = []
    emit = lines.append
    lines.extend(
        (
            "rm_tidx, _, _ = cute.arch.thread_idx()",
            "rm_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
            "rm_lane = rm_tidx % 32",
            "rm_g = rm_lane // 4",
            "rm_c = rm_lane % 4",
            "rm_mi = rm_lane // 8",
            "rm_r = rm_lane % 8",
            "rm_pid = cutlass.Int32(cute.arch.block_idx()[0])",
            f"rm_bh = rm_pid % {num_bh}",
            f"rm_row0 = (rm_pid // {num_bh}) * {rows}",
            "rm_zero = cutlass.Float32(0.0)",
            f"rm_scale = cutlass.Float32({scale_log2!r})",
            f"rm_s_q = cute.arch.alloc_smem({io_dtype}, {warps * rows * head_dim}, alignment=128)",
            f"rm_s_kv = cute.arch.alloc_smem({io_dtype}, {warps * stages * 2 * chunk * head_dim}, alignment=128)",
            f"rm_s_part = cute.arch.alloc_smem(cutlass.Float32, {warps * rows * part_stride}, alignment=128)",
            f"rm_s_stat = cute.arch.alloc_smem(cutlass.Float32, {warps * 2 * rows}, alignment=16)",
            *(
                f"rm_s_aux{index} = cute.arch.alloc_smem({dtype}, {reps * threads * 8}, alignment=128)"
                for index, dtype in enumerate(aux_dtypes)
            ),
            f"rm_q_w = cutlass.Int32(rm_s_q.toint()) + rm_warp * {rows * row_bytes}",
            f"rm_kv_w = cutlass.Int32(rm_s_kv.toint()) + rm_warp * {stages * stage_bytes}",
            "rm_part = cutlass.Int32(rm_s_part.toint())",
            "rm_stat = cutlass.Int32(rm_s_stat.toint())",
            *(
                f"rm_aux{index} = cutlass.Int32(rm_s_aux{index}.toint())"
                for index in range(len(aux_dtypes))
            ),
            f"rm_key0 = rm_warp * {kpw}",
            # Byte offsets are formed in Int64 from the start: a head's base
            # passes 2**31 bytes well inside the flash surface's element bound.
            f"rm_bh_off = cutlass.Int64(rm_bh) * {seq * row_bytes}",
            f"rm_q_tile = {q_name}.iterator.toint() + rm_bh_off + cutlass.Int64(rm_row0) * {row_bytes}",
            *(
                ()
                if kv_group == 1
                else (
                    f"rm_kv_bh_off = cutlass.Int64(rm_bh // {kv_group}) * {seq * row_bytes}",
                )
            ),
            f"rm_k_slice = {k_name}.iterator.toint() + {'rm_bh_off' if kv_group == 1 else 'rm_kv_bh_off'} + cutlass.Int64(rm_key0) * {row_bytes}",
            f"rm_v_slice = {v_name}.iterator.toint() + {'rm_bh_off' if kv_group == 1 else 'rm_kv_bh_off'} + cutlass.Int64(rm_key0) * {row_bytes}",
        )
    )
    # ---- a fused row epilogue's aux rows: every epilogue thread stages the
    # eight-column packet(s) it will read into its own slot (group 0, with Q
    # and the first K chunk); it alone waits for and reads that slot. Nothing
    # is emitted without aux rows (the plain render is unchanged).
    for rep in range(reps if aux_elem_bytes else 0):
        lines.extend(
            (
                f"rm_ae = {rep * threads} + rm_tidx",
                f"rm_aerow = (rm_ae // {cpr}) % {rows}",
                f"rm_aecol = rm_ae % {cpr}",
            )
        )
        for index, elem_bytes in enumerate(aux_elem_bytes):
            name = row_epilogue.aux_params[index]  # type: ignore[union-attr]
            aux_row_bytes = head_dim * elem_bytes
            lines.append(
                f"rm_aux{index}_g = {name}.iterator.toint() + cutlass.Int64(rm_bh) * {seq * aux_row_bytes} "
                f"+ cutlass.Int64(rm_row0 + rm_aerow) * {aux_row_bytes} + cutlass.Int64(rm_aecol * {8 * elem_bytes})"
            )
            lines.extend(
                f"{rt}.cp_async_16(rm_aux{index} + rm_ae * {8 * elem_bytes} + {packet * 16}, "
                f"rm_aux{index}_g + {packet * 16})"
                for packet in range(8 * elem_bytes // 16)
            )
    # ---- issue the Q tile and the first chunk's K (group 0), its V (group 1)
    # and, when double buffered, the second chunk (group 2) before any MMA.
    issue_tile(lines, "", smem_base="rm_q_w", gmem_base="rm_q_tile", tile_rows=rows)
    issue_tile(lines, "", smem_base="rm_kv_w", gmem_base="rm_k_slice", tile_rows=chunk)
    emit("cute.arch.cp_async_commit_group()")
    issue_tile(
        lines,
        "",
        smem_base=f"(rm_kv_w + {chunk * row_bytes})",
        gmem_base="rm_v_slice",
        tile_rows=chunk,
    )
    emit("cute.arch.cp_async_commit_group()")
    prefetched = stages == 2 and nchunk > 1
    if prefetched:
        issue_kv(lines, "", stage="1", chunk_index="1")
        emit("cute.arch.cp_async_commit_group()")
    emit(f"cute.arch.cp_async_wait_group({1 + int(prefetched)})")
    emit("cute.arch.sync_warp()")
    # ---- Q^T B-fragments: k-steps 2j, 2j+1 from packets 4j..4j+3 (rows = queries).
    for o in range(octets):
        lines.extend(
            f"rm_qb_{o}_{2 * j}_0, rm_qb_{o}_{2 * j}_1, rm_qb_{o}_{2 * j + 1}_0, rm_qb_{o}_{2 * j + 1}_1 = "
            f"{rt}.ldmatrix_x4(rm_q_w + {swz(f'{8 * o} + rm_r', f'{4 * j} + rm_mi')})"
            for j in range(k_steps // 2)
        )
    if nchunk > 1:
        lines.extend(
            (
                f"rm_m = cute.make_rmem_tensor(({2 * octets},), cutlass.Float32)",
                f"rm_l = cute.make_rmem_tensor(({2 * octets},), cutlass.Float32)",
                f"rm_o = cute.make_rmem_tensor(({octets * d_tiles * 4},), cutlass.Float32)",
            )
        )
    # ---- chunk 0: QK^T may start once K arrived; V is waited right before PV.
    first_chunk: list[str] = []
    compute_chunk(first_chunk, "", stage="0", first=True)
    pv_start = next(
        index for index, line in enumerate(first_chunk) if line.startswith("rm_vrow = ")
    )
    first_chunk[pv_start:pv_start] = [
        f"cute.arch.cp_async_wait_group({int(prefetched)})",
        "cute.arch.sync_warp()",
    ]
    lines.extend(first_chunk)
    if nchunk > 1:
        spill_state(lines, "")
        emit(f"for rm_chunk in cutlass.range(1, {nchunk}):")
        body: list[str] = []
        ind = "    "
        # Every lane finished reading the stage about to be refilled (the
        # previous chunk's, or this chunk's own single stage) before the
        # cp.async packets of the next chunk land in it.
        body.append(f"{ind}cute.arch.sync_warp()")
        if stages == 2:
            body.append(f"{ind}if rm_chunk + 1 < {nchunk}:")
            issue_kv(
                body,
                ind + "    ",
                stage="(rm_chunk + 1) % 2",
                chunk_index="rm_chunk + 1",
            )
            body.extend(
                (
                    f"{ind}    cute.arch.cp_async_commit_group()",
                    f"{ind}    cute.arch.cp_async_wait_group(1)",
                    f"{ind}else:",
                    f"{ind}    cute.arch.cp_async_wait_group(0)",
                )
            )
            stage = "rm_chunk % 2"
        else:
            issue_kv(body, ind, stage="0", chunk_index="rm_chunk")
            body.extend(
                (
                    f"{ind}cute.arch.cp_async_commit_group()",
                    f"{ind}cute.arch.cp_async_wait_group(0)",
                )
            )
            stage = "0"
        body.append(f"{ind}cute.arch.sync_warp()")
        compute_chunk(body, ind, stage=stage, first=False)
        spill_state(body, ind)
        lines.extend(body)
        reload_state(lines, "")
    # ---- publish the warp's partial (m, l, O^T) and combine in fixed warp order.
    emit(f"rm_part_w = rm_part + rm_warp * {rows * part_stride * 4}")
    for o in range(octets):
        for dt in range(d_tiles):
            lines.extend(
                f"{rt}.sts_f32(rm_part_w + (({8 * o} + 2 * rm_c + {row_off}) * {part_stride} + "
                f"{16 * dt + col_off} + rm_g) * 4, rm_o_{o}_{dt}_{i})"
                for i, (row_off, col_off) in enumerate(((0, 0), (1, 0), (0, 8), (1, 8)))
            )
    emit(f"rm_stat_w = rm_stat + rm_warp * {2 * rows * 4}")
    # The eight lanes sharing c hold identical statistics; all of them store.
    for o in range(octets):
        for half, off in (("a", 0), ("b", 1)):
            lines.extend(
                (
                    f"{rt}.sts_f32(rm_stat_w + ({8 * o} + 2 * rm_c + {off}) * 4, rm_m_{o}_{half})",
                    f"{rt}.sts_f32(rm_stat_w + ({rows + 8 * o} + 2 * rm_c + {off}) * 4, rm_l_{o}_{half})",
                )
            )
    emit("cute.arch.barrier()")
    # ---- epilogue: thread -> (row, 16-byte column packet); threads beyond the
    # tile repeat a row's work and store identical values (no divergence).
    group_reduce = {"sum": "sum", "amax": "max", "amin": "min"}

    def lane_reduce(kind: str, value: str) -> str:
        """Complete a row reduction over the ``cpr`` lanes sharing the row."""
        return f"{rt}.lane_group_{group_reduce[kind]}({value}, {cpr})"

    def aux_view(index: int, dtype: str, elem_bytes: int) -> str:
        """This thread's staged packet(s) of aux row ``index`` as a smem tensor."""
        return (
            f"cute.make_tensor(cute.make_ptr({dtype}, rm_aux{index} + rm_e * {8 * elem_bytes}, "
            "cute.AddressSpace.smem, assumed_align=16), cute.make_layout((8,)))"
        )

    for rep in range(reps):
        lines.extend(
            (
                f"rm_e = {rep * threads} + rm_tidx",
                f"rm_erow = (rm_e // {cpr}) % {rows}",
                f"rm_ecol = rm_e % {cpr}",
            )
        )
        for w in range(warps):
            lines.extend(
                (
                    f"rm_mw_{w} = {rt}.lds_f32(rm_stat + ({w * 2 * rows} + rm_erow) * 4)",
                    f"rm_lw_{w} = {rt}.lds_f32(rm_stat + ({w * 2 * rows + rows} + rm_erow) * 4)",
                )
            )
        emit("rm_m_all = rm_mw_0")
        lines.extend(
            f"rm_m_all = cute.arch.fmax(rm_m_all, rm_mw_{w})" for w in range(1, warps)
        )
        lines.extend(
            f"rm_al_{w} = cute.arch.exp2((rm_mw_{w} - rm_m_all) * rm_scale)"
            for w in range(warps)
        )
        emit("rm_l_all = rm_al_0 * rm_lw_0")
        lines.extend(
            f"rm_l_all = rm_l_all + rm_al_{w} * rm_lw_{w}" for w in range(1, warps)
        )
        lines.extend(f"rm_acc_{i} = rm_zero" for i in range(8))
        for w in range(warps):
            lines.extend(
                (
                    f"rm_pbase = rm_part + ({w * rows * part_stride} + rm_erow * {part_stride} + rm_ecol * 8) * 4",
                    f"rm_v0, rm_v1, rm_v2, rm_v3 = {rt}.lds_v4_f32(rm_pbase)",
                    f"rm_v4, rm_v5, rm_v6, rm_v7 = {rt}.lds_v4_f32(rm_pbase + 16)",
                )
            )
            lines.extend(
                f"rm_acc_{i} = rm_acc_{i} + rm_al_{w} * rm_v{i}" for i in range(8)
            )
        emit("rm_inv = cutlass.Float32(1.0) / rm_l_all")
        out_address = (
            f"{o_name}.iterator.toint() + rm_bh_off + cutlass.Int64(rm_row0 + rm_erow) * "
            f"{row_bytes} + cutlass.Int64(rm_ecol * 16)"
        )
        if row_epilogue is None:
            for i in range(8):
                value = f"rm_acc_{i} * rm_inv"
                if relu_output:
                    value = f"cute.arch.fmax({value}, rm_zero)"
                emit(f"rm_out_{i} = {value}")
            words = ", ".join(
                f"{pack}(rm_out_{2 * i}, rm_out_{2 * i + 1})" for i in range(4)
            )
            emit(f"{rt}.stg_v4_b32({out_address}, {words})")
        else:
            # The program walks this thread's eight normalized columns as four
            # element pairs; a reduction's thread partial is completed over the
            # ``cpr`` lanes sharing the row in a fixed butterfly order, and the
            # row values (scalars) are recomputed by every lane of the row.
            words = ", ".join(
                f"{pack}({{out}}[{2 * i}], {{out}}[{2 * i + 1}])" for i in range(4)
            )
            _prologue, epilogue = emit_row_epilogue(
                row_epilogue.program,
                chunks=1,
                chunk_width=8,
                elem_var="rm_ep_j",
                o_alloc="{o} = cute.make_rmem_tensor((8,), cutlass.Float32)",
                o_load=[f"{{o}}[{i}] = rm_acc_{i} * rm_inv" for i in range(8)],
                o_elem="{o}[{j}]",
                aux_allocs=[
                    f"{{a}} = cute.make_rmem_tensor((8,), {dtype})"
                    for dtype in aux_dtypes
                ],
                aux_loads=[
                    [f"cute.autovec_copy({aux_view(index, dtype, elem_bytes)}, {{a}})"]
                    for index, (dtype, elem_bytes) in enumerate(
                        zip(aux_dtypes, aux_elem_bytes, strict=True)
                    )
                ],
                aux_elems=["cutlass.Float32({a}[{j}])"] * len(aux_dtypes),
                out_alloc="{out} = cute.make_rmem_tensor((8,), cutlass.Float32)",
                store_elem="{out}[{j}] = {value}",
                store=[f"{rt}.stg_v4_b32({out_address}, {words})"],
                scalar_names=row_epilogue.scalar_names,
                indent="",
                prefix="rm_ep",
                reduce_across=lane_reduce,
            )
            assert not _prologue
            lines.extend(line for line in epilogue.split("\n") if line.strip())
        if lse_name is not None:
            lse_value = "rm_m_all * rm_scale + cute.math.log2(rm_l_all)"
            if not math.isclose(lse_scale, 1.0, rel_tol=1e-6, abs_tol=1e-7):
                lse_value = f"({lse_value}) * cutlass.Float32({lse_scale!r})"
            # Every thread of a row holds the same LSE; the stores coincide.
            emit(
                f"{rt}.stg_f32({lse_name}.iterator.toint() + cutlass.Int64(rm_bh) * {seq * 4} "
                f"+ cutlass.Int64(rm_row0 + rm_erow) * 4, "
                f"{lse_value})"
            )
    source = "\n".join(lines) + "\n"
    return list(ast.parse(textwrap.dedent(source)).body)
