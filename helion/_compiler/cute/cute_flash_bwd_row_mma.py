"""The ``row_mma`` family of the fused flash-attention BACKWARD: register-MMA
row programs for small grids.

The tcgen05 backward (``cute_flash_bwd.py``) owns a whole SM per 128-row KV
tile: 512 threads, TMEM-resident dK/dV, TMA rings and a TMA reduce-add drain
for dQ.  Its per-CTA fixed cost (TMEM allocation, barrier and descriptor
setup, the first loads, the drains and the staged epilogue) measures ~4.7 us
on a B200 against ~1.4 us per 128x128 Q step, so a problem whose KV tiles see
one or two Q steps each -- ``(1, 4, 256, 64)``: eight CTAs, two steps -- runs
at ~7.4 us while the Triton backward's 16x32 register tiles take ~6.1 us.
A 64-row tcgen05 tile keeps the same floor.

This family is the backward analogue of the forward's ``row_mma`` row
programs: one CTA per ``kv_rows`` KV rows of one (batch, head) group, its
warps arranged as ``kv_rows // 16`` row groups x ``q_split`` Q ranges; a
warp owns 16 KV rows and walks its Q range in ``q_chunk``-row chunks, all in
registers with warp-level ``mma.sync.m16n8k16``:

* ``S^T = K Q^T`` and ``dP^T = V dO^T`` with the warp's K / V rows as the A
  operands (``ldmatrix`` from a per-warp shared-memory copy) and ``Q^T`` /
  ``dO^T`` as the B operands (``ldmatrix`` of the staged chunk);
* ``P^T = exp2(S^T * scale2 - lse[q])`` and ``dS^T = P^T (dP^T - delta[q])``
  on the fp32 accumulator fragments (the fused body's formulas), the per-Q
  statistics read from the staged LSE / delta rows;
* ``dV += P^T dO`` and ``dK += dS^T Q``: the accumulator fragments, packed to
  the I/O dtype, are the m16k16 A fragments as they stand (the forward's
  register reuse), ``dO`` / ``Q`` read as B by ``ldmatrix.trans``;
* ``dQ = dS K``: the dS A fragments are ``movmatrix`` transposes of the
  ``dS^T`` words, K is read as B by ``ldmatrix.trans``; the (16 Q rows x
  head_dim) products are scaled and accumulated into the fp32 dQ buffer with
  ``red.global.add.v2.f32`` (no cross-warp reduction, no barrier);
* the Q / dO chunks and their LSE / delta rows stream through per-warp
  ``cp.async`` stages (double buffered where the budget allows) so the loop
  has no block-level synchronization at all; dK / dV are stored straight
  from registers when a row group has one warp, else the ``q_split``
  partials are summed in fixed order through shared memory after one CTA
  barrier (deterministic dK / dV; dQ's reduce-adds are order-free like the
  tcgen05 family's TMA reduce-adds).

The family hosts dense (non-causal) 16-bit plans of head dim 64 or 128,
including grouped queries (the KV group's ``mm_dim`` rows are walked); the
causal plan and every other shape stay on the tcgen05 family (fail closed).
It uses no TMA, TMEM or tcgen05.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from .cute_flash_row_mma import ROW_MMA_SMEM_BUDGET
from .cute_flash_row_mma import emit_flash_row_mma_module_statements

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from ..generate_ast import GenerateAST
    from .cute_flash_bwd import AttentionBwdPlan

FLASH_BWD_FAMILY_KEY = "cute_flash_bwd_family"
FLASH_BWD_ROW_KV_KEY = "cute_flash_bwd_row_kv"
FLASH_BWD_ROW_Q_SPLIT_KEY = "cute_flash_bwd_row_q_split"
FLASH_BWD_ROW_Q_CHUNK_KEY = "cute_flash_bwd_row_q_chunk"
FLASH_BWD_ROW_KEYS: tuple[str, ...] = (
    FLASH_BWD_ROW_KV_KEY,
    FLASH_BWD_ROW_Q_SPLIT_KEY,
    FLASH_BWD_ROW_Q_CHUNK_KEY,
)
FLASH_BWD_FA4_FAMILY = "fa4"
FLASH_BWD_ROW_FAMILY = "row_mma"
FLASH_BWD_FAMILIES: tuple[str, ...] = (FLASH_BWD_FA4_FAMILY, FLASH_BWD_ROW_FAMILY)
FLASH_BWD_ROW_PLAN_KIND = "helion_flash_bwd_row_mma"
# The first choice of each tuple is the knob's default (and the seed).
FLASH_BWD_ROW_KV_CHOICES: tuple[int, ...] = (16, 32, 64, 128)
FLASH_BWD_ROW_Q_SPLIT_CHOICES: tuple[int, ...] = (4, 8, 2, 1)
FLASH_BWD_ROW_Q_CHUNK_CHOICES: tuple[int, ...] = (64, 32, 16)
FLASH_BWD_ROW_HEAD_DIMS: tuple[int, ...] = (64, 128)
FLASH_BWD_ROW_MAX_WARPS = 16
# The family is searched (and seeded) where the tcgen05 family's grid is
# small: at most this many 128-row KV tiles (less than half the SMs of a
# B200); larger grids leave it to explicit configs.
FLASH_BWD_ROW_SEARCH_MAX_TILES = 64
_RT = "_helion_flash_rowmma"


@dataclasses.dataclass(frozen=True)
class FlashBwdRowFacts:
    """The plan facts the autotuner needs to legalize the row knobs."""

    admitted: bool
    head_dim: int
    mm_dim: int
    n_dim: int
    total_kv_rows: int


def flash_bwd_row_admitted(plan: AttentionBwdPlan) -> bool:
    """Dense 16-bit plans of head dim 64 / 128 (the causal walk stays on the
    tcgen05 family)."""
    return (
        not plan.causal
        and plan.io_dtype in (torch.float16, torch.bfloat16)
        and plan.head_dim in FLASH_BWD_ROW_HEAD_DIMS
    )


def flash_bwd_row_facts(plan: AttentionBwdPlan) -> FlashBwdRowFacts:
    return FlashBwdRowFacts(
        admitted=flash_bwd_row_admitted(plan),
        head_dim=plan.head_dim,
        mm_dim=plan.mm_dim,
        n_dim=plan.n_dim,
        total_kv_rows=plan.total_kv_rows,
    )


def flash_bwd_row_search_grid(facts: FlashBwdRowFacts) -> bool:
    """Whether the tcgen05 family's grid is small enough to search the row
    programs (see ``FLASH_BWD_ROW_SEARCH_MAX_TILES``)."""
    return facts.admitted and (
        facts.total_kv_rows // 128 <= FLASH_BWD_ROW_SEARCH_MAX_TILES
    )


def flash_bwd_row_warps(kv_rows: int, q_split: int) -> int:
    return (kv_rows // 16) * q_split


def flash_bwd_row_smem_bytes(
    *, head_dim: int, kv_rows: int, q_split: int, q_chunk: int, stages: int
) -> int:
    """Static shared memory of one CTA: per warp its K and V rows, ``stages``
    Q / dO chunks with their LSE / delta rows, and one region that holds the
    two-slot dQ tile ring during the walk and (``q_split`` > 1) the warp's
    dK / dV partial rows (padded fp32) after it."""
    warps = flash_bwd_row_warps(kv_rows, q_split)
    row_bytes = head_dim * 2
    per_warp = 2 * 16 * row_bytes + stages * (2 * q_chunk * row_bytes + 2 * q_chunk * 4)
    return warps * (per_warp + flash_bwd_row_region_bytes(head_dim, q_split))


def flash_bwd_row_region_bytes(head_dim: int, q_split: int) -> int:
    """Bytes of a warp's dQ ring / dK-dV partial region: two 16-row fp32 dQ
    tiles, or the two 16-row partials at the padded stride when they are
    exchanged (``q_split`` > 1; the larger of the two)."""
    return 2 * 16 * (head_dim + (4 if q_split > 1 else 0)) * 4


def flash_bwd_row_plan(
    facts: FlashBwdRowFacts, *, kv_rows: int, q_split: int, q_chunk: int
) -> int | None:
    """The number of cp.async stages of the Q / dO walk, or None when the
    knobs are not legal for the plan: whole 16-row groups dividing the KV
    rows of a group, whole chunks dividing each warp's Q range, at most
    ``FLASH_BWD_ROW_MAX_WARPS`` warps, and a staging that fits the
    shared-memory budget (two stages where the range has several chunks,
    one as the last resort)."""
    if not facts.admitted:
        return None
    if kv_rows % 16 or q_chunk % 16 or kv_rows <= 0 or q_split <= 0:
        return None
    if facts.n_dim % kv_rows or facts.mm_dim % (q_split * q_chunk):
        return None
    if flash_bwd_row_warps(kv_rows, q_split) > FLASH_BWD_ROW_MAX_WARPS:
        return None
    nchunk = facts.mm_dim // (q_split * q_chunk)
    for stages in (2, 1) if nchunk > 1 else (1,):
        if (
            flash_bwd_row_smem_bytes(
                head_dim=facts.head_dim,
                kv_rows=kv_rows,
                q_split=q_split,
                q_chunk=q_chunk,
                stages=stages,
            )
            <= ROW_MMA_SMEM_BUDGET
        ):
            return stages
    return None


def flash_bwd_row_knobs(config: MutableMapping[str, object]) -> tuple[int, int, int]:
    """``(kv_rows, q_split, q_chunk)`` of a config (defaults where unset)."""
    return (
        int(config.get(FLASH_BWD_ROW_KV_KEY, FLASH_BWD_ROW_KV_CHOICES[0])),  # type: ignore[call-overload]
        int(config.get(FLASH_BWD_ROW_Q_SPLIT_KEY, FLASH_BWD_ROW_Q_SPLIT_CHOICES[0])),  # type: ignore[call-overload]
        int(config.get(FLASH_BWD_ROW_Q_CHUNK_KEY, FLASH_BWD_ROW_Q_CHUNK_CHOICES[0])),  # type: ignore[call-overload]
    )


def flash_bwd_row_legalize(
    facts: FlashBwdRowFacts | None, config: MutableMapping[str, object]
) -> None:
    """Normalize the family knobs to one identity per kernel.

    Off the row family (or where the plan never admits it) the three row
    knobs are inert: they read their defaults.  On it, a knob triple the plan
    cannot run is replaced by the nearest legal one (fewer Q splits, then
    smaller chunks, then fewer KV rows, in that order); when nothing is legal
    the config renders the tcgen05 family.
    """
    family = config.get(FLASH_BWD_FAMILY_KEY, FLASH_BWD_FA4_FAMILY)
    if family == FLASH_BWD_ROW_FAMILY and facts is not None and facts.admitted:
        kv_rows, q_split, q_chunk = flash_bwd_row_knobs(config)
        if (
            flash_bwd_row_plan(facts, kv_rows=kv_rows, q_split=q_split, q_chunk=q_chunk)
            is None
        ):
            legal = _nearest_legal(facts, kv_rows, q_split, q_chunk)
            if legal is None:
                family = FLASH_BWD_FA4_FAMILY
            else:
                kv_rows, q_split, q_chunk = legal
        if family == FLASH_BWD_ROW_FAMILY:
            config[FLASH_BWD_FAMILY_KEY] = family
            config[FLASH_BWD_ROW_KV_KEY] = kv_rows
            config[FLASH_BWD_ROW_Q_SPLIT_KEY] = q_split
            config[FLASH_BWD_ROW_Q_CHUNK_KEY] = q_chunk
            return
    if FLASH_BWD_FAMILY_KEY in config:
        config[FLASH_BWD_FAMILY_KEY] = FLASH_BWD_FA4_FAMILY
    for key, choices in zip(
        FLASH_BWD_ROW_KEYS,
        (
            FLASH_BWD_ROW_KV_CHOICES,
            FLASH_BWD_ROW_Q_SPLIT_CHOICES,
            FLASH_BWD_ROW_Q_CHUNK_CHOICES,
        ),
        strict=True,
    ):
        if key in config:
            config[key] = choices[0]


def _nearest_legal(
    facts: FlashBwdRowFacts, kv_rows: int, q_split: int, q_chunk: int
) -> tuple[int, int, int] | None:
    splits = [s for s in FLASH_BWD_ROW_Q_SPLIT_CHOICES if s <= q_split]
    splits.sort(reverse=True)
    chunks = [c for c in FLASH_BWD_ROW_Q_CHUNK_CHOICES if c <= q_chunk]
    chunks.sort(reverse=True)
    rows = [r for r in FLASH_BWD_ROW_KV_CHOICES if r <= kv_rows]
    rows.sort(reverse=True)
    for r in rows:
        for c in chunks:
            for s in splits:
                if flash_bwd_row_plan(facts, kv_rows=r, q_split=s, q_chunk=c):
                    return r, s, c
    return None


def emit_flash_bwd_row_mma_module_statements(cg: GenerateAST) -> None:
    """The row programs share the forward family's runtime module."""
    emit_flash_row_mma_module_statements(cg)


def emit_flash_bwd_row_mma_device_body(
    *,
    plan: AttentionBwdPlan,
    io_dtype: str,
    scale2_expr: str,
    out_scale_expr: str,
    dk_scale_expr: str,
    kv_rows: int,
    q_split: int,
    q_chunk: int,
    stages: int,
    q_name: str,
    k_name: str,
    v_name: str,
    do_name: str,
    lse_name: str,
    delta_name: str,
    dq_name: str,
    dk_name: str,
    dv_name: str,
) -> str:
    """Render the row-program backward body (see the module docstring).

    One CTA per (KV group, ``kv_rows``-row KV tile); warp ``w`` owns row
    group ``w // q_split`` (16 KV rows) and Q range ``w % q_split`` of the
    group's ``mm_dim`` rows, walked in ``q_chunk``-row chunks through
    ``stages`` cp.async stages.  Every byte offset is formed in Int64 from
    the tensor base.
    """
    hd = plan.head_dim
    groups = kv_rows // 16
    split = q_split
    chunk = q_chunk
    warps = groups * split
    threads = 32 * warps
    rows_per_split = plan.mm_dim // split
    nchunk = rows_per_split // chunk
    tiles_per_group = plan.n_dim // kv_rows
    row_bytes = hd * 2
    cpr = hd // 8  # 16-byte packets per row
    ksteps = hd // 16  # k-steps of S^T / dP^T over the head dim
    dts = hd // 16  # pairs of head-dim octets (N tiles of dV / dK / dQ)
    octs = chunk // 8  # Q octets per chunk (N tiles of S^T / dP^T)
    qks = chunk // 16  # k-steps over the chunk (dV / dK) == dQ m-tiles
    nacc = hd // 8  # octets of the dK / dV accumulators
    kv_bytes = 2 * 16 * row_bytes
    stage_bytes = 2 * chunk * row_bytes
    stat_bytes = 2 * chunk * 4
    part_stride = hd + 4  # fp32 row stride of the dK / dV partial exchange
    part_bytes = 2 * 16 * part_stride * 4
    dq_tile_bytes = 16 * hd * 4  # one 16-row fp32 dQ tile (global layout)
    region_bytes = flash_bwd_row_region_bytes(hd, split)
    assert region_bytes >= 2 * dq_tile_bytes
    stat_packets = chunk // 4
    kind = "bf16" if io_dtype == "cutlass.BFloat16" else "f16"
    mma = f"{_RT}.mma_{kind}"
    pack = f"{_RT}.pack_{kind}x2"
    prefetched = stages == 2 and nchunk > 1
    assert ksteps % 2 == 0 and stat_packets <= 32

    def swz(row: str, col: str) -> str:
        """Byte offset of 16-byte packet ``col`` of row ``row`` under the
        XOR-by-row swizzle (eight ldmatrix row reads on distinct bank groups)."""
        return f"({row}) * {row_bytes} + ((({col}) ^ (({row}) % 8)) * 16)"

    def issue_tile(
        out: list[str], ind: str, *, smem_base: str, gmem_base: str, tile_rows: int
    ) -> None:
        """cp.async the ``tile_rows`` rows at ``gmem_base`` into the swizzled
        tile at ``smem_base`` (whole 32-packet passes: 16-row tiles of 8+
        packets)."""
        assert (tile_rows * cpr) % 32 == 0
        for i in range(tile_rows * cpr // 32):
            out.extend(
                (
                    f"{ind}rb_idx = {i * 32} + rb_lane",
                    f"{ind}rb_irow = rb_idx // {cpr}",
                    f"{ind}rb_icol = rb_idx % {cpr}",
                    (
                        f"{ind}{_RT}.cp_async_16({smem_base} + {swz('rb_irow', 'rb_icol')}, "
                        f"{gmem_base} + cutlass.Int64(rb_irow * {row_bytes} + rb_icol * 16))"
                    ),
                )
            )

    def issue_stat(out: list[str], ind: str, *, smem_base: str, gmem_base: str) -> None:
        """cp.async the chunk's ``chunk`` fp32 statistics (one packet per lane,
        lane-guarded)."""
        guard = ind
        if stat_packets < 32:
            out.append(f"{ind}if rb_lane < {stat_packets}:")
            guard = ind + "    "
        out.append(
            f"{guard}{_RT}.cp_async_16({smem_base} + rb_lane * 16, "
            f"{gmem_base} + cutlass.Int64(rb_lane * 16))"
        )

    def issue_chunk(out: list[str], ind: str, *, stage: str, ci: str) -> None:
        out.extend(
            (
                f"{ind}rb_sb = rb_stage_w + ({stage}) * {stage_bytes}",
                f"{ind}rb_qoff = cutlass.Int64({ci}) * {chunk * row_bytes}",
                f"{ind}rb_tb = rb_stat_w + ({stage}) * {stat_bytes}",
                f"{ind}rb_toff = cutlass.Int64({ci}) * {chunk * 4}",
            )
        )
        issue_tile(
            out, ind, smem_base="rb_sb", gmem_base="(rb_q_g + rb_qoff)", tile_rows=chunk
        )
        issue_tile(
            out,
            ind,
            smem_base=f"(rb_sb + {chunk * row_bytes})",
            gmem_base="(rb_do_g + rb_qoff)",
            tile_rows=chunk,
        )
        issue_stat(out, ind, smem_base="rb_tb", gmem_base="(rb_lse_g + rb_toff)")
        issue_stat(
            out,
            ind,
            smem_base=f"(rb_tb + {chunk * 4})",
            gmem_base="(rb_dlt_g + rb_toff)",
        )

    def compute_chunk(
        out: list[str], ind: str, *, stage: str, ci: str, first: bool
    ) -> None:
        """S^T / dP^T, the softmax recompute, dV / dK accumulation and the dQ
        reduce-adds of the chunk in ``stage`` (chunk index ``ci`` of the
        warp's range).  ``first`` chunks start the accumulators as SSA
        values; later ones read and write the carried register tensors."""
        out.extend(
            (
                f"{ind}rb_qsb = rb_stage_w + ({stage}) * {stage_bytes}",
                f"{ind}rb_dsb = rb_qsb + {chunk * row_bytes}",
                f"{ind}rb_lseb = rb_stat_w + ({stage}) * {stat_bytes}",
                f"{ind}rb_dltb = rb_lseb + {chunk * 4}",
            )
        )
        # K / V rows as m16k16 A fragments (k-step st: packets 2*st and 2*st+1).
        for name, base in (("ka", "rb_kv_w"), ("va", f"(rb_kv_w + {16 * row_bytes})")):
            out.extend(
                f"{ind}rb_{name}_{st}_0, rb_{name}_{st}_1, rb_{name}_{st}_2, rb_{name}_{st}_3 = "
                f"{_RT}.ldmatrix_x4({base} + {swz('(rb_mi % 2) * 8 + rb_r', f'{2 * st} + rb_mi // 2')})"
                for st in range(ksteps)
            )
        # S^T[kv][q] and dP^T[kv][q] per Q octet: lane (g, c) holds i=0:
        # [g][8o+2c], i=1: [g][8o+2c+1], i=2: [g+8][8o+2c], i=3: [g+8][8o+2c+1].
        for o in range(octs):
            out.extend(f"{ind}rb_s_{o}_{i} = rb_zero" for i in range(4))
            out.extend(f"{ind}rb_dp_{o}_{i} = rb_zero" for i in range(4))
            for j in range(ksteps // 2):
                addr = swz(f"{8 * o} + rb_r", f"{4 * j} + rb_mi")
                out.append(
                    f"{ind}rb_b0, rb_b1, rb_b2, rb_b3 = {_RT}.ldmatrix_x4(rb_qsb + {addr})"
                )
                s = f"rb_s_{o}"
                for st, (b0, b1) in (
                    (2 * j, ("rb_b0", "rb_b1")),
                    (2 * j + 1, ("rb_b2", "rb_b3")),
                ):
                    out.append(
                        f"{ind}{s}_0, {s}_1, {s}_2, {s}_3 = {mma}(rb_ka_{st}_0, rb_ka_{st}_1, "
                        f"rb_ka_{st}_2, rb_ka_{st}_3, {b0}, {b1}, {s}_0, {s}_1, {s}_2, {s}_3)"
                    )
                out.append(
                    f"{ind}rb_b0, rb_b1, rb_b2, rb_b3 = {_RT}.ldmatrix_x4(rb_dsb + {addr})"
                )
                s = f"rb_dp_{o}"
                for st, (b0, b1) in (
                    (2 * j, ("rb_b0", "rb_b1")),
                    (2 * j + 1, ("rb_b2", "rb_b3")),
                ):
                    out.append(
                        f"{ind}{s}_0, {s}_1, {s}_2, {s}_3 = {mma}(rb_va_{st}_0, rb_va_{st}_1, "
                        f"rb_va_{st}_2, rb_va_{st}_3, {b0}, {b1}, {s}_0, {s}_1, {s}_2, {s}_3)"
                    )
        # P^T = exp2(S^T * scale2 - lse[q]); dS^T = P^T * (dP^T - delta[q]);
        # both packed to the I/O dtype as A-fragment words (row g, row g+8).
        for o in range(octs):
            s, dp, p, ds = f"rb_s_{o}", f"rb_dp_{o}", f"rb_p_{o}", f"rb_ds_{o}"
            out.extend(
                (
                    f"{ind}rb_lse_a, rb_lse_b = {_RT}.lds_v2_f32(rb_lseb + ({8 * o} + 2 * rb_c) * 4)",
                    f"{ind}rb_dl_a, rb_dl_b = {_RT}.lds_v2_f32(rb_dltb + ({8 * o} + 2 * rb_c) * 4)",
                    f"{ind}rb_nl_a = rb_zero - rb_lse_a",
                    f"{ind}rb_nl_b = rb_zero - rb_lse_b",
                    f"{ind}rb_x0, rb_x1 = cute.arch.fma_packed_f32x2(({s}_0, {s}_1), (rb_scale2, rb_scale2), (rb_nl_a, rb_nl_b))",
                    f"{ind}rb_x2, rb_x3 = cute.arch.fma_packed_f32x2(({s}_2, {s}_3), (rb_scale2, rb_scale2), (rb_nl_a, rb_nl_b))",
                    *(f"{ind}{p}_{i} = cute.arch.exp2(rb_x{i})" for i in range(4)),
                    f"{ind}rb_y0, rb_y1 = cute.arch.sub_packed_f32x2(({dp}_0, {dp}_1), (rb_dl_a, rb_dl_b))",
                    f"{ind}rb_y2, rb_y3 = cute.arch.sub_packed_f32x2(({dp}_2, {dp}_3), (rb_dl_a, rb_dl_b))",
                    f"{ind}{ds}_0, {ds}_1 = cute.arch.mul_packed_f32x2(({p}_0, {p}_1), (rb_y0, rb_y1))",
                    f"{ind}{ds}_2, {ds}_3 = cute.arch.mul_packed_f32x2(({p}_2, {p}_3), (rb_y2, rb_y3))",
                    f"{ind}rb_pw_{o}_0 = {pack}({p}_0, {p}_1)",
                    f"{ind}rb_pw_{o}_1 = {pack}({p}_2, {p}_3)",
                    f"{ind}rb_dw_{o}_0 = {pack}({ds}_0, {ds}_1)",
                    f"{ind}rb_dw_{o}_1 = {pack}({ds}_2, {ds}_3)",
                )
            )
        # dV / dK accumulators: rb_dv_{n}_{i} over head-dim octet n (lane
        # (g, c): rows g / g+8, columns 8n+2c, 8n+2c+1).
        for acc, carry in (("rb_dv", "rb_dvt"), ("rb_dk", "rb_dkt")):
            for n in range(nacc):
                for i in range(4):
                    if first:
                        out.append(f"{ind}{acc}_{n}_{i} = rb_zero")
                    else:
                        out.append(f"{ind}{acc}_{n}_{i} = {carry}[{n * 4 + i}]")
        # dV += P^T dO and dK += dS^T Q: k-step ks of the chunk's Q rows, B
        # from ``ldmatrix.trans`` of rows 16ks..16ks+15 (octet pair dt).
        for ks in range(qks):
            a_p = ", ".join(
                (
                    f"rb_pw_{2 * ks}_0",
                    f"rb_pw_{2 * ks}_1",
                    f"rb_pw_{2 * ks + 1}_0",
                    f"rb_pw_{2 * ks + 1}_1",
                )
            )
            a_d = ", ".join(
                (
                    f"rb_dw_{2 * ks}_0",
                    f"rb_dw_{2 * ks}_1",
                    f"rb_dw_{2 * ks + 1}_0",
                    f"rb_dw_{2 * ks + 1}_1",
                )
            )
            for dt in range(dts):
                addr = swz(
                    f"{16 * ks} + (rb_mi // 2) * 8 + rb_r", f"{2 * dt} + rb_mi % 2"
                )
                out.append(
                    f"{ind}rb_b0, rb_b1, rb_b2, rb_b3 = {_RT}.ldmatrix_x4_trans(rb_dsb + {addr})"
                )
                for n, (b0, b1) in (
                    (2 * dt, ("rb_b0", "rb_b2")),
                    (2 * dt + 1, ("rb_b1", "rb_b3")),
                ):
                    acc = f"rb_dv_{n}"
                    out.append(
                        f"{ind}{acc}_0, {acc}_1, {acc}_2, {acc}_3 = {mma}({a_p}, {b0}, {b1}, "
                        f"{acc}_0, {acc}_1, {acc}_2, {acc}_3)"
                    )
                out.append(
                    f"{ind}rb_b0, rb_b1, rb_b2, rb_b3 = {_RT}.ldmatrix_x4_trans(rb_qsb + {addr})"
                )
                for n, (b0, b1) in (
                    (2 * dt, ("rb_b0", "rb_b2")),
                    (2 * dt + 1, ("rb_b1", "rb_b3")),
                ):
                    acc = f"rb_dk_{n}"
                    out.append(
                        f"{ind}{acc}_0, {acc}_1, {acc}_2, {acc}_3 = {mma}({a_d}, {b0}, {b1}, "
                        f"{acc}_0, {acc}_1, {acc}_2, {acc}_3)"
                    )
        # dQ[q][d] = sum_kv dS[q][kv] K[kv][d] over this warp's 16 KV rows,
        # per 16-row Q tile mt: A = the dS^T words transposed by movmatrix, B
        # = K rows read as (k = kv, n = d) by ``ldmatrix.trans``. The scaled
        # tile is staged in a two-slot shared-memory ring in the global
        # layout (16 consecutive dQ rows are one contiguous span) and
        # reduce-added by one ``cp.reduce.async.bulk`` per tile: the L2
        # reduction units process bulk operations at bandwidth, while the
        # same elements as per-lane ``red.global.add`` cost ~2 us per call at
        # (1, 4, 256, 64). The slot is reused once its previous bulk read
        # retired (one group may stay in flight).
        for mt in range(qks):
            out.extend(
                (
                    f"{ind}rb_a0 = {_RT}.movmatrix_trans(rb_dw_{2 * mt}_0)",
                    f"{ind}rb_a1 = {_RT}.movmatrix_trans(rb_dw_{2 * mt + 1}_0)",
                    f"{ind}rb_a2 = {_RT}.movmatrix_trans(rb_dw_{2 * mt}_1)",
                    f"{ind}rb_a3 = {_RT}.movmatrix_trans(rb_dw_{2 * mt + 1}_1)",
                    f"{ind}rb_slot = rb_dq_w + ((({ci}) * {qks} + {mt}) % 2) * {dq_tile_bytes}",
                    f"{ind}rb_sa = rb_slot + (rb_g * {hd} + 2 * rb_c) * 4",
                    f"{ind}rb_sb2 = rb_sa + {8 * hd * 4}",
                    f"{ind}if rb_lane == 0:",
                    f"{ind}    cute.arch.cp_async_bulk_wait_group(1, read=True)",
                    f"{ind}cute.arch.sync_warp()",
                )
            )
            for dt in range(dts):
                out.append(
                    f"{ind}rb_b0, rb_b1, rb_b2, rb_b3 = {_RT}.ldmatrix_x4_trans(rb_kv_w + "
                    f"{swz('(rb_mi // 2) * 8 + rb_r', f'{2 * dt} + rb_mi % 2')})"
                )
                for n, (b0, b1) in (
                    (2 * dt, ("rb_b0", "rb_b2")),
                    (2 * dt + 1, ("rb_b1", "rb_b3")),
                ):
                    out.extend(
                        (
                            (
                                f"{ind}rb_q0, rb_q1, rb_q2, rb_q3 = {mma}(rb_a0, rb_a1, rb_a2, rb_a3, {b0}, {b1}, "
                                "rb_zero, rb_zero, rb_zero, rb_zero)"
                            ),
                            f"{ind}rb_q0, rb_q1 = cute.arch.mul_packed_f32x2((rb_q0, rb_q1), (rb_oscale, rb_oscale))",
                            f"{ind}rb_q2, rb_q3 = cute.arch.mul_packed_f32x2((rb_q2, rb_q3), (rb_oscale, rb_oscale))",
                            f"{ind}{_RT}.sts_v2_f32(rb_sa + {8 * n * 4}, rb_q0, rb_q1)",
                            f"{ind}{_RT}.sts_v2_f32(rb_sb2 + {8 * n * 4}, rb_q2, rb_q3)",
                        )
                    )
            out.extend(
                (
                    f"{ind}cute.arch.fence_view_async_shared()",
                    f"{ind}cute.arch.sync_warp()",
                    f"{ind}if rb_lane == 0:",
                    f"{ind}    {_RT}.bulk_reduce_add_f32(rb_dq_g + cutlass.Int64(({ci}) * {chunk} + {16 * mt}) * {hd * 4}, rb_slot, {dq_tile_bytes})",
                    f"{ind}    cute.arch.cp_async_bulk_commit_group()",
                )
            )

    def spill_state(out: list[str], ind: str) -> None:
        for acc, carry in (("rb_dv", "rb_dvt"), ("rb_dk", "rb_dkt")):
            out.extend(
                f"{ind}{carry}[{n * 4 + i}] = {acc}_{n}_{i}"
                for n in range(nacc)
                for i in range(4)
            )

    lines: list[str] = []
    emit = lines.append
    lines.extend(
        (
            "rb_tidx, _, _ = cute.arch.thread_idx()",
            "rb_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
            "rb_lane = rb_tidx % 32",
            "rb_g = rb_lane // 4",
            "rb_c = rb_lane % 4",
            "rb_mi = rb_lane // 8",
            "rb_r = rb_lane % 8",
            "rb_pid = cutlass.Int32(cute.arch.block_idx()[0])",
            f"rb_grp = rb_pid // {tiles_per_group}",
            f"rb_tile = rb_pid % {tiles_per_group}",
            f"rb_rg = rb_warp // {split}",
            f"rb_qs = rb_warp % {split}",
            "rb_zero = cutlass.Float32(0.0)",
            f"rb_scale2 = cutlass.Float32({scale2_expr})",
            f"rb_oscale = cutlass.Float32({out_scale_expr})",
            f"rb_dkscale = cutlass.Float32({dk_scale_expr})",
            f"rb_s_kv = cute.arch.alloc_smem({io_dtype}, {warps * 2 * 16 * hd}, alignment=128)",
            f"rb_s_stage = cute.arch.alloc_smem({io_dtype}, {warps * stages * 2 * chunk * hd}, alignment=128)",
            f"rb_s_stat = cute.arch.alloc_smem(cutlass.Float32, {warps * stages * 2 * chunk}, alignment=16)",
            # Per-warp region: the dQ tile ring during the walk, the dK / dV
            # partial rows after it (q_split > 1).
            f"rb_s_part = cute.arch.alloc_smem(cutlass.Float32, {warps * region_bytes // 4}, alignment=128)",
            "rb_part = cutlass.Int32(rb_s_part.toint())",
            f"rb_dq_w = rb_part + rb_warp * {region_bytes}",
            f"rb_kv_w = cutlass.Int32(rb_s_kv.toint()) + rb_warp * {kv_bytes}",
            f"rb_stage_w = cutlass.Int32(rb_s_stage.toint()) + rb_warp * {stages * stage_bytes}",
            f"rb_stat_w = cutlass.Int32(rb_s_stat.toint()) + rb_warp * {stages * stat_bytes}",
            # The warp's first KV row (flattened) and first Q row of its range.
            f"rb_kv_row = rb_grp * {plan.n_dim} + rb_tile * {kv_rows} + rb_rg * 16",
            f"rb_q_row0 = rb_grp * {plan.mm_dim} + rb_qs * {rows_per_split}",
            f"rb_k_g = {k_name}.iterator.toint() + cutlass.Int64(rb_kv_row) * {row_bytes}",
            f"rb_v_g = {v_name}.iterator.toint() + cutlass.Int64(rb_kv_row) * {row_bytes}",
            f"rb_q_g = {q_name}.iterator.toint() + cutlass.Int64(rb_q_row0) * {row_bytes}",
            f"rb_do_g = {do_name}.iterator.toint() + cutlass.Int64(rb_q_row0) * {row_bytes}",
            f"rb_lse_g = {lse_name}.iterator.toint() + cutlass.Int64(rb_q_row0) * 4",
            f"rb_dlt_g = {delta_name}.iterator.toint() + cutlass.Int64(rb_q_row0) * 4",
            f"rb_dq_g = {dq_name}.iterator.toint() + cutlass.Int64(rb_q_row0) * {hd * 4}",
        )
    )
    # ---- K and V rows with chunk 0 (group 0), chunk 1 when double buffered
    # (group 1), before any MMA.
    issue_tile(lines, "", smem_base="rb_kv_w", gmem_base="rb_k_g", tile_rows=16)
    issue_tile(
        lines,
        "",
        smem_base=f"(rb_kv_w + {16 * row_bytes})",
        gmem_base="rb_v_g",
        tile_rows=16,
    )
    issue_chunk(lines, "", stage="0", ci="0")
    emit("cute.arch.cp_async_commit_group()")
    if prefetched:
        issue_chunk(lines, "", stage="1", ci="1")
        emit("cute.arch.cp_async_commit_group()")
    emit(f"cute.arch.cp_async_wait_group({int(prefetched)})")
    emit("cute.arch.sync_warp()")
    if nchunk > 1:
        lines.extend(
            (
                f"rb_dvt = cute.make_rmem_tensor(({nacc * 4},), cutlass.Float32)",
                f"rb_dkt = cute.make_rmem_tensor(({nacc * 4},), cutlass.Float32)",
            )
        )
    compute_chunk(lines, "", stage="0", ci="0", first=True)
    if nchunk > 1:
        spill_state(lines, "")
        emit(f"for rb_ci in cutlass.range(1, {nchunk}):")
        body: list[str] = []
        ind = "    "
        # Every lane finished reading the stage about to be refilled (the
        # previous chunk's, or this chunk's own single stage).
        body.append(f"{ind}cute.arch.sync_warp()")
        if stages == 2:
            body.append(f"{ind}if rb_ci + 1 < {nchunk}:")
            issue_chunk(body, ind + "    ", stage="(rb_ci + 1) % 2", ci="rb_ci + 1")
            body.extend(
                (
                    f"{ind}    cute.arch.cp_async_commit_group()",
                    f"{ind}    cute.arch.cp_async_wait_group(1)",
                    f"{ind}else:",
                    f"{ind}    cute.arch.cp_async_wait_group(0)",
                )
            )
            stage = "rb_ci % 2"
        else:
            issue_chunk(body, ind, stage="0", ci="rb_ci")
            body.extend(
                (
                    f"{ind}cute.arch.cp_async_commit_group()",
                    f"{ind}cute.arch.cp_async_wait_group(0)",
                )
            )
            stage = "0"
        body.append(f"{ind}cute.arch.sync_warp()")
        compute_chunk(body, ind, stage=stage, ci="rb_ci", first=False)
        spill_state(body, ind)
        lines.extend(body)
        for acc, carry in (("rb_dv", "rb_dvt"), ("rb_dk", "rb_dkt")):
            lines.extend(
                f"{acc}_{n}_{i} = {carry}[{n * 4 + i}]"
                for n in range(nacc)
                for i in range(4)
            )
    # ---- epilogue: dK = acc * dk_scale, dV = acc, in the I/O dtype.
    if split == 1:
        # One warp per row group: straight from the fragments (rows g, g+8;
        # columns 8n+2c, 8n+2c+1 as one packed word).
        lines.extend(
            (
                f"rb_dk_g = {dk_name}.iterator.toint() + cutlass.Int64(rb_kv_row) * {row_bytes}",
                f"rb_dv_g = {dv_name}.iterator.toint() + cutlass.Int64(rb_kv_row) * {row_bytes}",
                f"rb_dk_ga = rb_dk_g + cutlass.Int64(rb_g) * {row_bytes}",
                f"rb_dk_gb = rb_dk_ga + {8 * row_bytes}",
                f"rb_dv_ga = rb_dv_g + cutlass.Int64(rb_g) * {row_bytes}",
                f"rb_dv_gb = rb_dv_ga + {8 * row_bytes}",
            )
        )
        for n in range(nacc):
            lines.extend(
                (
                    f"rb_k0, rb_k1 = cute.arch.mul_packed_f32x2((rb_dk_{n}_0, rb_dk_{n}_1), (rb_dkscale, rb_dkscale))",
                    f"rb_k2, rb_k3 = cute.arch.mul_packed_f32x2((rb_dk_{n}_2, rb_dk_{n}_3), (rb_dkscale, rb_dkscale))",
                    f"{_RT}.stg_b32(rb_dk_ga + ({8 * n} + 2 * rb_c) * 2, {pack}(rb_k0, rb_k1))",
                    f"{_RT}.stg_b32(rb_dk_gb + ({8 * n} + 2 * rb_c) * 2, {pack}(rb_k2, rb_k3))",
                    f"{_RT}.stg_b32(rb_dv_ga + ({8 * n} + 2 * rb_c) * 2, {pack}(rb_dv_{n}_0, rb_dv_{n}_1))",
                    f"{_RT}.stg_b32(rb_dv_gb + ({8 * n} + 2 * rb_c) * 2, {pack}(rb_dv_{n}_2, rb_dv_{n}_3))",
                )
            )
    else:
        # The Q-split partials of a row group are summed in fixed split
        # order after one CTA barrier: thread -> (row, 16-byte packet). The
        # partial rows reuse the warp's dQ ring once its bulk reads retired.
        lines.extend(
            (
                "if rb_lane == 0:",
                "    cute.arch.cp_async_bulk_wait_group(0, read=True)",
                "cute.arch.sync_warp()",
                f"rb_part_w = rb_part + rb_warp * {part_bytes}",
            )
        )
        for which, acc in enumerate(("rb_dk", "rb_dv")):
            for n in range(nacc):
                lines.extend(
                    (
                        f"{_RT}.sts_v2_f32(rb_part_w + ({which * 16 * part_stride} + rb_g * {part_stride} + {8 * n} + 2 * rb_c) * 4, {acc}_{n}_0, {acc}_{n}_1)",
                        f"{_RT}.sts_v2_f32(rb_part_w + ({which * 16 * part_stride + 8 * part_stride} + rb_g * {part_stride} + {8 * n} + 2 * rb_c) * 4, {acc}_{n}_2, {acc}_{n}_3)",
                    )
                )
        emit("cute.arch.barrier()")
        packets = kv_rows * cpr
        reps = -(-packets // threads)
        emit(f"rb_out_row = rb_grp * {plan.n_dim} + rb_tile * {kv_rows}")
        for which, (name, scale) in enumerate(
            ((dk_name, "rb_dkscale"), (dv_name, None))
        ):
            for rep in range(reps):
                ind = ""
                emit(f"rb_e = {rep * threads} + rb_tidx")
                if (rep + 1) * threads > packets:
                    emit(f"if rb_e < {packets}:")
                    ind = "    "
                lines.extend(
                    (
                        f"{ind}rb_erow = rb_e // {cpr}",
                        f"{ind}rb_ecol = rb_e % {cpr}",
                        f"{ind}rb_pb = rb_part + ((rb_erow // 16) * {split * part_bytes} + {which * 16 * part_stride * 4} + (rb_erow % 16) * {part_stride * 4} + rb_ecol * 32)",
                        *(f"{ind}rb_acc_{i} = rb_zero" for i in range(8)),
                    )
                )
                for s in range(split):
                    lines.extend(
                        (
                            f"{ind}rb_v0, rb_v1, rb_v2, rb_v3 = {_RT}.lds_v4_f32(rb_pb + {s * part_bytes})",
                            f"{ind}rb_v4, rb_v5, rb_v6, rb_v7 = {_RT}.lds_v4_f32(rb_pb + {s * part_bytes + 16})",
                            *(
                                f"{ind}rb_acc_{i} = rb_acc_{i} + rb_v{i}"
                                for i in range(8)
                            ),
                        )
                    )
                if scale is not None:
                    lines.extend(
                        f"{ind}rb_acc_{i} = rb_acc_{i} * {scale}" for i in range(8)
                    )
                words = ", ".join(
                    f"{pack}(rb_acc_{2 * i}, rb_acc_{2 * i + 1})" for i in range(4)
                )
                emit(
                    f"{ind}{_RT}.stg_v4_b32({name}.iterator.toint() + cutlass.Int64(rb_out_row + rb_erow) * {row_bytes} + cutlass.Int64(rb_ecol * 16), {words})"
                )
    # The issuing lane's bulk reduce-adds complete before the kernel exits.
    lines.extend(
        ("if rb_lane == 0:", "    cute.arch.cp_async_bulk_wait_group(0, read=True)")
    )
    return "".join(f"    {line}\n" for line in lines)
