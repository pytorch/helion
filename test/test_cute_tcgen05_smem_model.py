"""The DEFAULT-layout tcgen05 GEMM's static SMEM model (CPU only).

``CuteTcgen05Config.default_layout_smem_bytes`` reproduces the arena ptxas
lays out for the role-local kernel; the expectations below are the ``SHARED``
bytes ``cuobjdump --dump-resource-usage`` reported for the compiled cubins on
B200 minus the 1 KiB the system reserves per CTA. The fits verdicts follow the
compile outcomes of the same sweep: every geometry ``fits`` admits compiled and
ran, every one it rejects failed in NVVM (``too much shared``).
"""

from __future__ import annotations

import pytest

from helion._compiler.backend import CuteBackend
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES,
)
from helion._compiler.cute.tcgen05_constants import Tcgen05RowvecAuxFacts
from helion._compiler.cute.tcgen05_constants import Tcgen05RowvecAuxRow
from helion._compiler.cute.tcgen05_constants import tcgen05_fixed_smem_overhead_bytes
from helion._compiler.cute.tcgen05_constants import tcgen05_round_up_smem_bytes
from helion.autotuner.config_spec import ConfigSpec

# The arena model sizes the epilogue stage through CUTLASS's own helper, so it
# needs the CuTe DSL like the kernels it describes; most CI runners lack it.
pytest.importorskip("cutlass")

# B200: 227 KiB opt-in per CTA; a probe allocating exactly this compiles and
# launches, 8 B more fails in NVVM.
B200_OPTIN_BYTES = 232_448
RESERVED_PER_CTA_BYTES = 1_024

F32_ROW = Tcgen05RowvecAuxFacts(
    output_itemsize=2, rows=(Tcgen05RowvecAuxRow(itemsize=4, promoted=False),)
)
PROMOTED_ROW = Tcgen05RowvecAuxFacts(
    output_itemsize=2, rows=(Tcgen05RowvecAuxRow(itemsize=2, promoted=True),)
)
NO_ROWS = Tcgen05RowvecAuxFacts(output_itemsize=2, rows=())


def _config(dtype_bytes: int, facts: Tcgen05RowvecAuxFacts | None) -> CuteTcgen05Config:
    config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
    config.ab_stages_three_search_constraints = Tcgen05AbStagesThreeSearchConstraints(
        dtype_bytes=dtype_bytes,
        per_cta_smem_budget_bytes=B200_OPTIN_BYTES
        - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES,
    )
    config.rowvec_aux_facts = facts
    config.search_enabled = True
    return config


# (kernel facts, operand bytes, bm, bn, bk, cluster_m, ab, c, stage rows,
#  acc stages, cuobjdump SHARED of the compiled cubin)
_MEASURED = [
    # fp16 x fp32 row (the 4 KiB / 2 KiB warp-private stage).
    ("f32row", 2, 128, 128, 64, 1, 6, 2, True, 2, 216_228),
    ("f32row", 2, 128, 128, 128, 1, 3, 2, True, 2, 216_228),
    ("f32row", 2, 128, 256, 64, 1, 3, 2, True, 2, 185_508),
    ("f32row", 2, 256, 256, 64, 2, 5, 2, True, 2, 201_900),
    ("f32row", 2, 256, 256, 64, 2, 4, 4, True, 2, 201_900),
    ("f32row", 2, 256, 128, 64, 2, 8, 2, True, 2, 216_236),
    # The seven configs the 3 KiB headroom demoted (F5): 192 KiB AB ring +
    # 32 KiB c=4 ring + 2 KiB fp32 row stage.
    ("f32row", 2, 256, 128, 64, 2, 8, 4, True, 2, 232_620),
    ("f32row", 2, 256, 128, 128, 2, 4, 4, True, 2, 232_620),
    ("f32row", 2, 128, 128, 32, 1, 12, 4, True, 2, 232_740),
    ("f32row", 2, 128, 128, 64, 1, 6, 4, True, 2, 232_612),
    ("f32row", 2, 128, 128, 128, 1, 3, 4, True, 2, 232_612),
    # plain fp16 (no rows; acc_stages 1 and 2). The 256-wide two-CTA tile
    # takes the plan's (128, 32) subtile (8 KiB C stages).
    ("plain", 2, 128, 128, 64, 1, 6, 2, False, 2, 214_180),
    ("plain", 2, 256, 256, 128, 2, 3, 2, False, 2, 214_188),
    ("plain", 2, 256, 256, 64, 2, 6, 2, False, 2, 214_188),
    ("plain", 2, 128, 128, 64, 1, 6, 4, False, 1, 230_548),
    # fp16 bias (promoted 16-bit row -> one CTA-shared FP32 row, (128, 32)
    # subtile; the 256 B / 512 B stages occupy a whole KiB).
    ("promoted", 2, 256, 256, 128, 2, 3, 2, True, 2, 215_212),
    ("promoted", 2, 256, 256, 64, 2, 6, 4, True, 2, 231_596),
    ("promoted", 2, 256, 256, 64, 2, 5, 4, True, 2, 198_828),
    ("promoted", 2, 256, 64, 64, 2, 9, 2, True, 2, 203_052),
    ("promoted", 2, 128, 64, 64, 1, 8, 2, True, 2, 215_204),
    ("promoted", 2, 128, 128, 64, 1, 6, 2, True, 2, 215_204),
    ("promoted", 2, 256, 128, 64, 2, 8, 2, True, 2, 215_212),
    # fp8 x fp32 row (bf16 output; the AB mbarriers grow past 8 stages).
    ("f32row", 1, 256, 256, 64, 2, 11, 2, True, 2, 218_412),
    ("f32row", 1, 256, 256, 128, 2, 5, 2, True, 2, 201_900),
    ("f32row", 1, 256, 128, 64, 2, 12, 2, True, 2, 167_212),
    ("f32row", 1, 128, 128, 64, 1, 12, 2, True, 2, 216_356),
    ("f32row", 1, 128, 256, 64, 1, 7, 2, True, 2, 210_084),
]

_FACTS = {"f32row": F32_ROW, "plain": NO_ROWS, "promoted": PROMOTED_ROW}


@pytest.mark.parametrize(
    "facts_name,dtype_bytes,bm,bn,bk,cluster_m,ab,c,stage_rows,acc,shared",
    _MEASURED,
    ids=[
        f"{row[0]}_{row[2]}x{row[3]}x{row[4]}_cm{row[5]}_ab{row[6]}_c{row[7]}"
        for row in _MEASURED
    ],
)
def test_arena_model_matches_the_compiled_cubin(
    facts_name: str,
    dtype_bytes: int,
    bm: int,
    bn: int,
    bk: int,
    cluster_m: int,
    ab: int,
    c: int,
    stage_rows: bool,
    acc: int,
    shared: int,
) -> None:
    config = _config(dtype_bytes, _FACTS[facts_name])
    arena = config.default_layout_smem_bytes(
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab,
        c_stages=c,
        stage_rows=stage_rows,
        acc_stages=acc,
    )
    assert arena == shared - RESERVED_PER_CTA_BYTES
    assert arena + TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES <= B200_OPTIN_BYTES
    assert config.default_layout_smem_fits(
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab,
        c_stages=c,
        stage_rows=stage_rows,
        acc_stages=acc,
    )


# Geometries NVVM rejected (``too much shared``): a 192 KiB ring next to a
# 32 KiB C ring and the 4 KiB warp-private fp32 stage.
_REJECTED = [
    (2, 256, 256, 128, 2, 3, 2),
    (2, 256, 256, 64, 2, 6, 2),
    (2, 256, 256, 64, 2, 5, 4),
    (2, 128, 256, 64, 1, 4, 2),
    (1, 256, 256, 64, 2, 12, 2),
    (1, 256, 256, 128, 2, 6, 2),
    (1, 128, 256, 64, 1, 8, 2),
]


@pytest.mark.parametrize(
    "dtype_bytes,bm,bn,bk,cluster_m,ab,c",
    _REJECTED,
    ids=[f"{r[1]}x{r[2]}x{r[3]}_cm{r[4]}_ab{r[5]}_c{r[6]}_op{r[0]}" for r in _REJECTED],
)
def test_arena_model_rejects_the_geometries_nvvm_rejected(
    dtype_bytes: int, bm: int, bn: int, bk: int, cluster_m: int, ab: int, c: int
) -> None:
    config = _config(dtype_bytes, F32_ROW)
    arena = config.default_layout_smem_bytes(
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab,
        c_stages=c,
        stage_rows=True,
    )
    assert arena is not None
    assert arena > B200_OPTIN_BYTES
    assert not config.default_layout_smem_fits(
        bm=bm,
        bn=bn,
        bk=bk,
        cluster_m=cluster_m,
        ab_stages=ab,
        c_stages=c,
        stage_rows=True,
    )


def test_fixed_overhead_follows_the_arena_layout() -> None:
    # 128 B of AB mbarriers up to eight stages, 256 B up to sixteen; 16 B per
    # accumulator stage; the 4 B TMEM holding buffer; the CTA pair's 8 B
    # dealloc mbarrier.
    assert (
        tcgen05_fixed_smem_overhead_bytes(ab_stages=3, acc_stages=2, cluster_m=1) == 164
    )
    assert (
        tcgen05_fixed_smem_overhead_bytes(ab_stages=8, acc_stages=2, cluster_m=1) == 164
    )
    assert (
        tcgen05_fixed_smem_overhead_bytes(ab_stages=8, acc_stages=2, cluster_m=2) == 172
    )
    assert (
        tcgen05_fixed_smem_overhead_bytes(ab_stages=9, acc_stages=2, cluster_m=2) == 300
    )
    assert (
        tcgen05_fixed_smem_overhead_bytes(ab_stages=12, acc_stages=2, cluster_m=1)
        == 292
    )
    assert (
        tcgen05_fixed_smem_overhead_bytes(ab_stages=6, acc_stages=1, cluster_m=1) == 148
    )
    assert tcgen05_round_up_smem_bytes(96, 128) == 128
    assert tcgen05_round_up_smem_bytes(128, 128) == 128
    assert tcgen05_round_up_smem_bytes(512, 1024) == 1024


def test_row_stages_occupy_whole_kib_ahead_of_the_c_ring() -> None:
    # A promoted 16-bit row on a 64-wide tile stages 256 B; the C ring's
    # alignment follows it in the arena, so the plain tile plus one KiB is
    # what the compiled kernel holds (215 204 B vs 214 180 B measured).
    promoted = _config(2, PROMOTED_ROW)
    plain = _config(2, NO_ROWS)
    kwargs = {"bm": 128, "bn": 64, "bk": 64, "cluster_m": 1, "ab_stages": 8}
    staged = promoted.default_layout_smem_bytes(c_stages=2, stage_rows=True, **kwargs)
    bare = plain.default_layout_smem_bytes(c_stages=2, stage_rows=False, **kwargs)
    assert staged is not None and bare is not None
    assert staged - bare == 1024
    # The fp32 warp-private stage of a 128-wide tile is 2 KiB already.
    f32 = _config(2, F32_ROW)
    staged = f32.default_layout_smem_bytes(
        bm=128, bn=128, bk=64, cluster_m=1, ab_stages=6, c_stages=2, stage_rows=True
    )
    bare = plain.default_layout_smem_bytes(
        bm=128, bn=128, bk=64, cluster_m=1, ab_stages=6, c_stages=2, stage_rows=False
    )
    assert staged is not None and bare is not None
    assert staged - bare == 2048


def test_unanalyzed_store_chain_is_judged_at_the_widest_output() -> None:
    # Without store facts a staged row cannot be modelled (fails closed), but
    # a ring-only plan is sized for a 32-bit output instead of being refused.
    # The with-source subtile rule halves the subtile's N as the element
    # doubles, so on tiles 32 wide and up that ring is exactly the 16-bit
    # ring the facts would have recorded (128x128x64 ab=6 keeps c=4 either
    # way); on a 16-wide tile it is the larger of the two.
    unknown = _config(2, None)
    assert (
        unknown.default_layout_smem_bytes(
            bm=128, bn=128, bk=64, cluster_m=1, ab_stages=6, c_stages=4, stage_rows=True
        )
        is None
    )
    assert not unknown.default_layout_smem_fits(
        bm=128, bn=128, bk=64, cluster_m=1, ab_stages=6, c_stages=4, stage_rows=True
    )
    known = _config(2, NO_ROWS)
    kwargs = {"bm": 128, "bn": 128, "bk": 64, "cluster_m": 1, "c_stages": 4}
    unknown_bytes = unknown.default_layout_smem_bytes(
        ab_stages=6, stage_rows=False, **kwargs
    )
    known_bytes = known.default_layout_smem_bytes(
        ab_stages=6, stage_rows=False, **kwargs
    )
    assert unknown_bytes is not None and unknown_bytes == known_bytes
    assert unknown.default_layout_smem_fits(ab_stages=6, stage_rows=False, **kwargs)
    assert not unknown.default_layout_smem_fits(ab_stages=7, stage_rows=False, **kwargs)
    narrow = {"bm": 128, "bn": 16, "bk": 64, "cluster_m": 1, "c_stages": 4}
    unknown_bytes = unknown.default_layout_smem_bytes(
        ab_stages=8, stage_rows=False, **narrow
    )
    known_bytes = known.default_layout_smem_bytes(
        ab_stages=8, stage_rows=False, **narrow
    )
    assert unknown_bytes is not None and known_bytes is not None
    # (128, 16) subtile at 4 B vs 2 B per element, four stages.
    assert unknown_bytes - known_bytes == 4 * 128 * 16 * 2


def test_model_needs_a_recorded_budget() -> None:
    config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
    config.rowvec_aux_facts = NO_ROWS
    assert (
        config.default_layout_smem_bytes(
            bm=128,
            bn=128,
            bk=64,
            cluster_m=1,
            ab_stages=2,
            c_stages=2,
            stage_rows=False,
        )
        is None
    )
    assert not config.default_layout_smem_fits(
        bm=128, bn=128, bk=64, cluster_m=1, ab_stages=2, c_stages=2, stage_rows=False
    )
