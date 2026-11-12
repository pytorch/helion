"""The DEFAULT-layout tcgen05 GEMM's static SMEM model, and the sm_100 compile
checks of the fp32-row geometries it admits (at the end of the module).

``tcgen05_config.smem.default_layout_smem_bytes`` reproduces the arena ptxas
lays out for the role-local kernel; the expectations below are the ``SHARED``
bytes ``cuobjdump --dump-resource-usage`` reported for the compiled cubins on
B200 minus the 1 KiB the system reserves per CTA. The fits verdicts follow the
compile outcomes of the same sweep: every geometry ``fits`` admits compiled and
ran, every one it rejects failed in NVVM (``too much shared``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import pytest
import torch

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from helion._compiler.cute.tcgen05_config import smem
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
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_dense_tcgen05 import aligned_n_bf16_config
from helion._testing.cute_dense_tcgen05 import bf16_mm_args
from helion._testing.cute_dense_tcgen05 import column_scale_mm_into
from helion._testing.cute_dense_tcgen05 import plain_mm_into

if TYPE_CHECKING:
    from collections.abc import Callable
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_dense_tcgen05 import plain_mm
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

pytestmark = pytest.mark.cpu_only

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


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
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
    arena = smem.default_layout_smem_bytes(
        config,
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
    assert smem.default_layout_smem_fits(
        config,
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


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
@pytest.mark.parametrize(
    "dtype_bytes,bm,bn,bk,cluster_m,ab,c",
    _REJECTED,
    ids=[f"{r[1]}x{r[2]}x{r[3]}_cm{r[4]}_ab{r[5]}_c{r[6]}_op{r[0]}" for r in _REJECTED],
)
def test_arena_model_rejects_the_geometries_nvvm_rejected(
    dtype_bytes: int, bm: int, bn: int, bk: int, cluster_m: int, ab: int, c: int
) -> None:
    config = _config(dtype_bytes, F32_ROW)
    arena = smem.default_layout_smem_bytes(
        config,
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
    assert not smem.default_layout_smem_fits(
        config,
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


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
def test_row_stages_occupy_whole_kib_ahead_of_the_c_ring() -> None:
    # A promoted 16-bit row on a 64-wide tile stages 256 B; the C ring's
    # alignment follows it in the arena, so the plain tile plus one KiB is
    # what the compiled kernel holds (215 204 B vs 214 180 B measured).
    promoted = _config(2, PROMOTED_ROW)
    plain = _config(2, NO_ROWS)
    kwargs = {"bm": 128, "bn": 64, "bk": 64, "cluster_m": 1, "ab_stages": 8}
    staged = smem.default_layout_smem_bytes(
        promoted, c_stages=2, stage_rows=True, **kwargs
    )
    bare = smem.default_layout_smem_bytes(plain, c_stages=2, stage_rows=False, **kwargs)
    assert staged is not None and bare is not None
    assert staged - bare == 1024
    # The fp32 warp-private stage of a 128-wide tile is 2 KiB already.
    f32 = _config(2, F32_ROW)
    staged = smem.default_layout_smem_bytes(
        f32,
        bm=128,
        bn=128,
        bk=64,
        cluster_m=1,
        ab_stages=6,
        c_stages=2,
        stage_rows=True,
    )
    bare = smem.default_layout_smem_bytes(
        plain,
        bm=128,
        bn=128,
        bk=64,
        cluster_m=1,
        ab_stages=6,
        c_stages=2,
        stage_rows=False,
    )
    assert staged is not None and bare is not None
    assert staged - bare == 2048


@skipUnlessCuteAvailable("requires the supported CuTe DSL")
def test_unanalyzed_store_chain_is_judged_at_the_widest_output() -> None:
    # Without store facts a staged row cannot be modelled (fails closed), but
    # a ring-only plan is sized for a 32-bit output instead of being refused.
    # The with-source subtile rule halves the subtile's N as the element
    # doubles, so on tiles 32 wide and up that ring is exactly the 16-bit
    # ring the facts would have recorded (128x128x64 ab=6 keeps c=4 either
    # way); on a 16-wide tile it is the larger of the two.
    unknown = _config(2, None)
    assert (
        smem.default_layout_smem_bytes(
            unknown,
            bm=128,
            bn=128,
            bk=64,
            cluster_m=1,
            ab_stages=6,
            c_stages=4,
            stage_rows=True,
        )
        is None
    )
    assert not smem.default_layout_smem_fits(
        unknown,
        bm=128,
        bn=128,
        bk=64,
        cluster_m=1,
        ab_stages=6,
        c_stages=4,
        stage_rows=True,
    )
    known = _config(2, NO_ROWS)
    kwargs = {"bm": 128, "bn": 128, "bk": 64, "cluster_m": 1, "c_stages": 4}
    unknown_bytes = smem.default_layout_smem_bytes(
        unknown, ab_stages=6, stage_rows=False, **kwargs
    )
    known_bytes = smem.default_layout_smem_bytes(
        known, ab_stages=6, stage_rows=False, **kwargs
    )
    assert unknown_bytes is not None and unknown_bytes == known_bytes
    assert smem.default_layout_smem_fits(
        unknown, ab_stages=6, stage_rows=False, **kwargs
    )
    assert not smem.default_layout_smem_fits(
        unknown, ab_stages=7, stage_rows=False, **kwargs
    )
    narrow = {"bm": 128, "bn": 16, "bk": 64, "cluster_m": 1, "c_stages": 4}
    unknown_bytes = smem.default_layout_smem_bytes(
        unknown, ab_stages=8, stage_rows=False, **narrow
    )
    known_bytes = smem.default_layout_smem_bytes(
        known, ab_stages=8, stage_rows=False, **narrow
    )
    assert unknown_bytes is not None and known_bytes is not None
    # (128, 16) subtile at 4 B vs 2 B per element, four stages.
    assert unknown_bytes - known_bytes == 4 * 128 * 16 * 2


def test_model_needs_a_recorded_budget() -> None:
    config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
    config.rowvec_aux_facts = NO_ROWS
    assert (
        smem.default_layout_smem_bytes(
            config,
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
    assert not smem.default_layout_smem_fits(
        config,
        bm=128,
        bn=128,
        bk=64,
        cluster_m=1,
        ab_stages=2,
        c_stages=2,
        stage_rows=False,
    )


def test_output_edge_tma_store_admission_charges_the_ring_once() -> None:
    """The edge plan's (128, 64) C ring next to a 256x256x64 ab=6 ring fits."""
    config = _config(2, NO_ROWS)
    tile = {"bm": 256, "bn": 256, "bk": 64, "cluster_m": 2}
    assert smem.output_edge_tma_store_smem_fits(
        config, {}, ab_stages=6, c_stages=2, **tile
    )
    assert not smem.output_edge_tma_store_smem_fits(
        config, {}, ab_stages=7, c_stages=2, **tile
    )
    assert not smem.output_edge_tma_store_smem_fits(
        config, {}, ab_stages=6, c_stages=4, **tile
    )
    # A configured explicit subtile wider than the plan's (128, 32) on a
    # 128-wide tile is charged at its width: 192 KiB AB + 2 x 16 KiB fits, a
    # 4-deep ring of the wider subtile does not.
    narrow = {"bm": 256, "bn": 128, "bk": 64, "cluster_m": 2}
    explicit = {"tcgen05_layout_overrides_epi_tile_n": 64}
    assert smem.output_edge_tma_store_smem_fits(
        config, explicit, ab_stages=8, c_stages=2, **narrow
    )
    assert smem.output_edge_tma_store_smem_fits(
        config, {}, ab_stages=8, c_stages=4, **narrow
    )
    assert not smem.output_edge_tma_store_smem_fits(
        config, explicit, ab_stages=8, c_stages=4, **narrow
    )
    # Staged ``pre_acc_wait`` rows keep the SIMT epilogue on edge shapes.
    staged = _config(2, F32_ROW)
    assert not smem.output_edge_tma_store_smem_fits(
        staged,
        {"tcgen05_aux_load_placement": "pre_acc_wait"},
        ab_stages=6,
        c_stages=2,
        **tile,
    )
    # Without store facts the ring cannot be sized; the admission fails closed.
    assert not smem.output_edge_tma_store_smem_fits(
        _config(2, None), {}, ab_stages=6, c_stages=2, **tile
    )


# Compile checks for the fp32-row geometries the SMEM model admits (sm_100 GPU).
#
# Each case is a ``pre_acc_wait`` row-vector GEMM whose 192 KiB AB ring, 32 KiB
# c=4 ring and 2 KiB fp32 row stage fill the per-CTA opt-in to within a KiB
# (231 588-231 716 B of 232 448 modelled; the cubins report 232 612-232 740 B
# ``SHARED`` with the 1 KiB reservation). The 3 KiB seed headroom demoted every
# one of them to its ``post_acc_wait`` / c=2 twin; the arena model keeps them,
# and they must compile, launch and agree with the reference.


def _sm100_gpu(test: Callable[..., object]) -> Callable[..., object]:
    """Skip ``test`` without a CUDA GPU of capability 10.0 or newer; the
    ``cuda`` mark sends it to the GPU lane out of this ``cpu_only`` module."""
    return pytest.mark.cuda(
        skipIfNotCUDA()(
            skipIfCudaCapabilityLessThan(
                (10, 0), reason="tcgen05 requires an sm_100 GPU"
            )(test)
        )
    )


LAUNCHES = 8


def _f16_scale_gemm(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc * scale[tile_n]).to(x.dtype)
    return out


def _fp8_scale_gemm(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc * scale[tile_n]).to(torch.bfloat16)
    return out


class _Case(NamedTuple):
    operands: str
    bm: int
    bn: int
    bk: int
    cluster_m: int
    ab_stages: int
    values: dict[str, object]
    # Modelled arena bytes (``default_layout_smem_bytes``), exact for these
    # kernels: the compiled cubins report these plus the 1 KiB reservation.
    arena_bytes: int


def _two_cta(operands: str, bk: int, ab_stages: int, cluster_n: int) -> _Case:
    return _Case(
        operands,
        256,
        128,
        bk,
        2,
        ab_stages,
        {
            "block_sizes": [256, 128, bk],
            "pid_type": "persistent_interleaved",
            "l2_groupings": [4],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": cluster_n,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_c_stages": 4,
            "tcgen05_aux_load_placement": "pre_acc_wait",
        },
        231_596,
    )


def _one_cta(operands: str, bk: int, ab_stages: int) -> _Case:
    return _Case(
        operands,
        128,
        128,
        bk,
        1,
        ab_stages,
        {
            "block_sizes": [128, 128, bk],
            "pid_type": "persistent_interleaved",
            "tcgen05_cta_group": "auto",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_c_stages": 4,
            "tcgen05_acc_stages": 2,
            "l2_groupings": [1],
            "tcgen05_l2_swizzle_size": 1,
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_strategy": "role_local_monolithic",
            "tcgen05_layout_strategy": "default",
            "tcgen05_aux_load_placement": "pre_acc_wait",
        },
        # A ring deeper than eight stages takes a second 128 B mbarrier chunk.
        231_716 if ab_stages > 8 else 231_588,
    )


_CASES = {
    "f16_256x128x64_cm2cn1_ab8": _two_cta("f16", 64, 8, 1),
    "f16_256x128x64_cm2cn2_ab8": _two_cta("f16", 64, 8, 2),
    "f16_256x128x128_cm2cn1_ab4": _two_cta("f16", 128, 4, 1),
    "f16_256x128x128_cm2cn2_ab4": _two_cta("f16", 128, 4, 2),
    "f16_128x128x32_ab12": _one_cta("f16", 32, 12),
    "f16_128x128x64_ab6": _one_cta("f16", 64, 6),
    "f16_128x128x128_ab3": _one_cta("f16", 128, 3),
    "fp8_256x128x128_cm2cn1_ab8": _two_cta("fp8", 128, 8, 1),
    "fp8_256x128x128_cm2cn2_ab8": _two_cta("fp8", 128, 8, 2),
    "fp8_128x128x64_ab12": _one_cta("fp8", 64, 12),
    "fp8_128x128x128_ab6": _one_cta("fp8", 128, 6),
}


def _inputs(operands: str) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    torch.manual_seed(0)
    if operands == "f16":
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)
        y = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)
    else:
        x = (torch.randn(1024, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(1024, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
    scale = torch.rand(1024, device=DEVICE, dtype=torch.float32) + 0.5
    expected = (x.float() @ y.float()) * scale
    return (x, y, scale), expected


@pytest.mark.cuda
@_sm100_gpu
@pytest.mark.parametrize("case", sorted(_CASES), ids=sorted(_CASES))
def test_admitted_fp32_row_geometry_compiles_and_runs(case: str) -> None:
    c = _CASES[case]
    args, expected = _inputs(c.operands)
    kernel_fn = _f16_scale_gemm if c.operands == "f16" else _fp8_scale_gemm
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    spec = bound.config_spec
    # The search projection keeps the sampled knobs ...
    projected = dict(c.values)
    spec.normalize(projected, _fix_invalid=True)
    assert projected["tcgen05_ab_stages"] == c.ab_stages
    assert projected["tcgen05_c_stages"] == 4
    assert projected["tcgen05_aux_load_placement"] == "pre_acc_wait"
    # ... because the modelled arena (exact for these kernels) fits the opt-in.
    modelled = smem.default_layout_smem_bytes(
        spec._cute_tcgen05_config,
        bm=c.bm,
        bn=c.bn,
        bk=c.bk,
        cluster_m=c.cluster_m,
        ab_stages=c.ab_stages,
        c_stages=4,
        stage_rows=True,
    )
    assert modelled == c.arena_bytes
    assert modelled + TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES <= B200_OPTIN_BYTES
    bound.set_config(helion.Config(**c.values))
    outputs = [bound(*args) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    for out in outputs:
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=1e-1)


@_sm100_gpu
def test_output_edge_tma_store_beside_six_ab_stages_compiles_and_runs() -> None:
    """bf16 1000x4096x2880 (a ragged M) at 2x1 256x256x64 and six AB stages:
    the arena model fits the TMA store's (128, 64) C ring beside them
    (229 548 of 232 448 B); the kernel compiles and matches torch."""
    torch.manual_seed(0)
    x = torch.randn(1000, 2880, device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn(2880, 4096, device=DEVICE, dtype=torch.bfloat16)
    bound = plain_mm.bind((x, y))
    values = {
        "block_sizes": [256, 256, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_ab_stages": 6,
        "tcgen05_c_stages": 2,
        "tcgen05_acc_stages": 2,
        "l2_groupings": [1],
    }
    projected = dict(values)
    bound.config_spec.normalize(projected, _fix_invalid=True)
    assert projected["tcgen05_ab_stages"] == 6
    assert "tcgen05_tma_store_atom" in bound.to_code(helion.Config(**projected))
    bound.set_config(helion.Config(**projected))
    out = bound(x, y)
    torch.testing.assert_close(
        out, (x.float() @ y.float()).to(out.dtype), rtol=2e-2, atol=1e-1
    )


@_sm100_gpu
def test_sixteen_bit_aligned_n_pair_scales_columns() -> None:
    """A per-column FP32 scale on the 192-wide pair behind the 256 carrier
    reads the physical columns' scale (8192x768x768)."""
    torch.manual_seed(0)
    x, y, out = bf16_mm_args(8192, 768, 768, device=DEVICE)
    scale = torch.randn((768,), dtype=torch.float32, device=DEVICE)
    bound = column_scale_mm_into.bind((x, y, scale, out))
    config = aligned_n_bf16_config(tcgen05_ab_stages=6)
    assert "_BLOCK_SIZE_1 = 192" in bound.to_code(config)
    bound.set_config(config)
    bound(x, y, scale, out)
    torch.testing.assert_close(
        out, (x.float() @ y.float() * scale).to(out.dtype), rtol=2e-2, atol=1e-1
    )


@_sm100_gpu
@pytest.mark.parametrize(
    ("m", "rhs_k_major", "dtype", "changes"),
    (
        (1024, True, torch.bfloat16, {}),
        (1024, False, torch.bfloat16, {}),
        (1152, True, torch.bfloat16, {}),
        (1024, True, torch.float16, {}),
        # The seeded grouped raster with N tiles paired (A multicast).
        (
            1024,
            True,
            torch.bfloat16,
            {
                "tcgen05_cluster_n": 2,
                "pid_type": "persistent_interleaved",
                "l2_groupings": [8],
            },
        ),
    ),
    ids=("nt", "nn", "nt_m_tail", "nt_fp16", "nt_2x2_grouped"),
)
def test_sixteen_bit_aligned_n_pair_compiles_and_runs(
    m: int, rhs_k_major: bool, dtype: torch.dtype, changes: dict[str, object]
) -> None:
    """The 192-wide paired M256 tile behind the 256 carrier: seven 28 KiB AB
    stages beside a (128, 32) C ring fill the opt-in, writing an argument
    output (the Inductor calling convention); a partial last M tile is
    clipped by the TMA store, and an fp16 GEMM keeps its dtype."""
    torch.manual_seed(0)
    x, y, out = (
        t.to(dtype)
        for t in bf16_mm_args(m, 768, 512, rhs_k_major=rhs_k_major, device=DEVICE)
    )
    bound = plain_mm_into.bind((x, y, out))
    config = aligned_n_bf16_config(**changes)
    assert "_BLOCK_SIZE_1 = 192" in bound.to_code(config)
    bound.set_config(config)
    expected = (x.float() @ y.float()).to(dtype)
    for _ in range(LAUNCHES):
        out.zero_()
        bound(x, y, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=1e-1)
