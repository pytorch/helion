from __future__ import annotations

from itertools import starmap
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import sympy
import torch

import helion
from helion._compiler.autotuner_heuristics.triton import MatmulSeedDraft
from helion._compiler.autotuner_heuristics.triton import (
    TritonB200FormulaMatmulHeuristic,
)
from helion._compiler.autotuner_heuristics.triton import (
    TritonB200MultiMatmulHeuristic as _MULTI,
)
from helion._compiler.autotuner_heuristics.triton import TritonH100MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import _batched_static_matmul_fact
from helion._compiler.autotuner_heuristics.triton import _generalized_static_matmul_fact
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_spec import DotAxes
from helion.autotuner.config_spec import DotAxisKind
from helion.autotuner.config_spec import KernelGridFact
from helion.autotuner.config_spec import LiveTile
from helion.autotuner.config_spec import LoopAxisFact
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.config_spec import PipelinedRegion
from helion.autotuner.config_spec import RootGridFact
from helion.autotuner.config_spec import SymbolicLoopBound


def _matmul_fact(
    *,
    static_m: int = 1024,
    static_n: int = 1024,
    static_k: int = 1024,
    lhs_ndim: int = 2,
    rhs_ndim: int = 2,
) -> MatmulFact:
    return MatmulFact(
        lhs_ndim=lhs_ndim,
        rhs_ndim=rhs_ndim,
        m_block_id=0,
        n_block_id=1,
        k_block_id=2,
        static_m=static_m,
        static_n=static_n,
        static_k=static_k,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )


def _matmul_config_spec(
    *,
    matmul_facts: list[MatmulFact] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        matmul_facts=[] if matmul_facts is None else matmul_facts,
        block_sizes=[object(), object(), object()],
        allowed_pid_types=("flat",),
        _base_default_config=lambda: helion.Config(
            block_sizes=[1, 1, 1],
            l2_groupings=[1],
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        ),
        _flat_fields=lambda: {
            "block_sizes": ListOf(IntegerFragment(1, 4096, 1), length=3),
            "l2_groupings": ListOf(IntegerFragment(1, 64, 1), length=1),
            "num_warps": PowerOfTwoFragment(1, 32, 4),
            "num_stages": IntegerFragment(1, 8, 1),
            "pid_type": EnumFragment(("flat",)),
        },
        normalize=lambda raw, _fix_invalid=False: None,
        _shrink_for_numel_constraints=lambda config: None,
    )


def test_b200_formula_front_ends_supply_execution_defaults() -> None:
    assert TritonB200FormulaMatmulHeuristic.promote_seed_to_default is True
    assert _MULTI.promote_seed_to_default is True
    assert TritonB200FormulaMatmulHeuristic.HARDWARE_TARGETS == (("cuda", "sm100"),)
    assert issubclass(TritonB200FormulaMatmulHeuristic, TritonH100MatmulHeuristic)
    from helion._compiler.autotuner_heuristics import get_heuristics

    order = [h.__name__ for h in get_heuristics("triton")]
    assert "TritonB200MatmulHeuristic" not in order
    assert order.index("TritonB200FormulaMatmulHeuristic") < order.index(
        "TritonB200MultiMatmulHeuristic"
    )


def test_h100_base_tile_is_unchanged_by_tmem_budget() -> None:
    # TMEM_BUDGET is None on the sm90 base (no tensor memory), so the TMEM growth step is skipped and
    # the H100 formula stays byte-identical: a big compute-bound cube keeps the register-budget wide-N
    # [128, 256, 64] w8 s4 tile (num_sm=132).
    assert TritonH100MatmulHeuristic.TMEM_BUDGET is None
    assert TritonH100MatmulHeuristic._matmul_tile(4096, 4096, 4096, 2, 132, 1) == (
        128,
        256,
        64,
        8,
        4,
        1,
    )


def test_b200_tile_grows_to_fill_tmem_budget() -> None:
    # On sm100 the tile is grown against the TENSOR-MEMORY budget (the accumulator lives there, not in
    # registers): double whichever axis still fits, N first since it is the coalesced store axis. The
    # reservation includes the A operand at the largest bk the formula can emit, so the [256,256] fp32
    # accumulator -- which alone exactly fills tensor memory -- can never be reached, and the wide
    # [128,256] tile is the largest that fits.
    cls = TritonB200FormulaMatmulHeuristic
    sm = 148
    for m, n, k in ((2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)):
        assert cls._matmul_tile(m, n, k, 2, sm, 1)[:2] == (128, 256), (m, n, k)
    # Non-saturated batched dots grow too -- tensor memory is theirs to use as well.
    assert cls._matmul_tile(4096, 4096, 4096, 2, sm, 4)[:2] == (128, 256)
    # ...but a SATURATED batched dot keeps the small occupancy tile step (2.5) chose for it; growing it
    # back would undo that (measured: it inflates mamba-shaped tiles from [32,64] to [32,1024]).
    bm, bn = cls._matmul_tile(32, 1024, 4096, 2, sm, 1000)[:2]
    assert bm <= cls.SAT_TILE_BM and bn <= cls.SAT_TILE_BN, (bm, bn)


def test_b200_tmem_budget_matches_measured_hardware() -> None:
    # _tmem_bytes is a deliberate OVER-estimate in BYTES, checked against TMEM_BUDGET exactly like
    # _smem_bytes is checked against SMEM_BUDGET. Hardware reports tensor memory in COLUMNS (limit
    # 512), so the ground truth below is measured tmem_size in columns and what must agree is the
    # VERDICT: fits / does not fit. Ground truth is tmem_size read off compiled sm100 kernel metadata
    # in the worst case, i.e. with the A operand promoted into tensor memory.
    cls = TritonB200FormulaMatmulHeuristic
    hw_columns = {  # (bm, bn, bk) bf16 -> measured tmem columns, A promoted; hardware limit is 512
        (64, 64, 16): 128,
        (64, 128, 16): 256,
        (64, 256, 16): 512,
        (64, 512, 16): 512,
        (64, 1024, 16): 520,
        (128, 128, 16): 256,
        (128, 256, 16): 512,
        (128, 512, 16): 520,
        (128, 512, 32): 528,
        (128, 512, 64): 544,
        (128, 1024, 16): 1032,
        (256, 128, 16): 512,
        (256, 256, 16): 528,
        (
            256,
            256,
            32,
        ): 544,  # the CI failure: 512 for the acc + 32 for the promoted A operand
        (256, 256, 64): 576,
        (256, 256, 128): 640,
        (256, 512, 16): 1040,
    }
    for (bm, bn, bk), cols in hw_columns.items():
        fits_model = cls._tmem_bytes(bm, bn, bk, 2) <= cls.TMEM_BUDGET
        fits_hardware = cols <= 512
        assert fits_model == fits_hardware, (bm, bn, bk, cols)
    # Specifically: the [256,256] square is rejected, the wide [128,256] tile accepted.
    assert cls._tmem_bytes(256, 256, 16, 2) > cls.TMEM_BUDGET
    assert cls._tmem_bytes(128, 256, 64, 2) <= cls.TMEM_BUDGET
    # The budget is the real capacity: 128 lanes x 512 columns x 32 bit.
    assert cls.TMEM_BUDGET == 128 * 512 * 4
    # ...and a [256,256] fp32 accumulator alone exactly fills it, which is why no promoted A operand
    # can ever coexist with that square.
    assert cls.TMEM_BUDGET == 256 * 256 * 4
    # A tile too small for tcgen05 uses no tensor memory at all (measured tmem_size == 0), so it must
    # be charged nothing -- otherwise tiny decode tiles get rejected for a resource they never touch.
    assert cls._tmem_bytes(32, 4096, 16, 2) == 0
    assert cls._tmem_bytes(16, 512, 32, 4) == 0


def test_b200_smem_bytes_bounds_measured_hardware() -> None:
    # The epilogue's accumulator staging buffer (bm*bn*4) is invisible to the operand-ring formula and
    # is independent of num_stages, so a tile can fit the ring and still exceed SMEM. Values are
    # measured `shared` from compiled sm100 metadata; the model must be an UPPER bound on each.
    cls = TritonB200FormulaMatmulHeuristic
    # (bm, bn, bk, itemsize, num_stages) -> measured shared bytes (worst case over epilogue variants)
    measured = {
        (64, 64, 64, 2, 6): 32784,
        (128, 128, 64, 2, 6): 65552,
        (128, 256, 64, 2, 6): 131072,
        (256, 128, 32, 2, 6): 131072,
        (256, 256, 32, 2, 6): 262144,
        (
            256,
            256,
            16,
            2,
            6,
        ): 262144,  # ring shrinks with bk, the epilogue term does NOT
        (128, 128, 128, 4, 6): 786448,
    }
    for (bm, bn, bk, itemsize, ns), shared in measured.items():
        assert cls._smem_bytes(bm, bn, bk, itemsize, ns) >= shared, (bm, bn, bk, ns)
    # [256,256] is rejected by the epilogue term alone, at ANY bk -- the ring cannot rescue it.
    for bk in (16, 32, 64):
        assert cls._smem_bytes(256, 256, bk, 2, 1) > cls.SMEM_BUDGET
    # The slack is load-bearing: without it an otherwise-exact bound misses by the 16-byte mbarriers.
    assert cls.SMEM_SLACK >= 16


def test_sm90_conservative_accounting_is_inert() -> None:
    # sm90 must stay byte-identical, but NOT because the epilogue term is Blackwell-specific -- it is
    # not. Measured on an sm90 target, a [128,256,64] bf16 dot with an fp32 output reports
    # shared=131072 == bm*bn*4, exactly as sm100 does, so EPILOGUE_ACC_ITEMSIZE is set on the base.
    # What is sm100-only is ENFORCING the budget by shrinking the tile.
    cls = TritonH100MatmulHeuristic
    assert cls.TMEM_BUDGET is None  # no tensor memory on sm90
    assert cls._tmem_bytes(256, 256, 32, 2) == 0
    assert cls.EPILOGUE_ACC_ITEMSIZE == 4  # the term is arch-independent...
    assert (
        cls.ENFORCE_SMEM_BUDGET is False
    )  # ...but sm90 does not enforce it (model over-estimates)
    assert cls.SMEM_SLACK == 0

    # The epilogue term is a NO-OP on sm90 for an arithmetic reason: the accumulator lives in the
    # register file, so ACC_BUDGET caps the emittable tile at bm*bn == ACC_BUDGET. Binding would need
    # bm*bn * 4 > SMEM_BUDGET -- unreachable. Pin both halves of that argument so a future ACC_BUDGET
    # bump cannot silently make the term start binding on sm90 unnoticed.
    assert cls.ACC_BUDGET * cls.EPILOGUE_ACC_ITEMSIZE <= cls.SMEM_BUDGET
    emitted = {
        cls._matmul_tile(m, n, k, itemsize, 132, pinned_grid)[:2]
        for m in (1, 16, 256, 4096, 65536)
        for n in (1, 16, 256, 4096, 65536)
        for k in (16, 256, 4096)
        for itemsize in (1, 2, 4)
        for pinned_grid in (1, 4, 1000)
    }
    assert max(bm * bn for bm, bn in emitted) <= cls.ACC_BUDGET

    # And the sm90 tile is unchanged: still the register-budget [128, 256].
    assert cls._matmul_tile(4096, 4096, 4096, 2, 132, 1)[:3] == (128, 256, 64)


# ---------------------------------------------------------------------------
# Generalized axis freedom, graded occupancy, and whole-kernel resources.
#
# These pin the mandatory Section-3 capabilities individually, so a poor
# curriculum result is attributable to policy rather than to an implementation
# bug in the fact layer, the projection, the ranking, or the resource model.
# ---------------------------------------------------------------------------

_FML = TritonB200FormulaMatmulHeuristic
_H100 = TritonH100MatmulHeuristic


def _axes(
    m: DotAxisKind = DotAxisKind.TUNABLE_TILED,
    n: DotAxisKind = DotAxisKind.TUNABLE_TILED,
    k: DotAxisKind = DotAxisKind.TUNABLE_TILED,
    *,
    m_extent: int | None = 1024,
    n_extent: int | None = 1024,
    k_extent: int | None = 1024,
) -> DotAxes:
    return DotAxes(m, n, k, m_extent, n_extent, k_extent)


class _BlockSizesStub(list):
    """A ``ConfigSpec.block_sizes`` stand-in: indexable + sized like the real one, plus
    ``valid_block_ids()``."""

    def __init__(self, block_ids: list[int]) -> None:
        super().__init__(
            SimpleNamespace(block_id=b, min_size=1, max_size=4096, autotuner_min=1)
            for b in block_ids
        )
        self._ids = list(block_ids)

    def valid_block_ids(self) -> list[int]:
        return list(self._ids)

    def block_id_to_index(self, block_id: int) -> int:
        return self._ids.index(block_id)


def _block_sizes_stub(block_ids: list[int]) -> _BlockSizesStub:
    return _BlockSizesStub(block_ids)


def _generalized_spec(
    fact: MatmulFact,
    axes: DotAxes,
    *,
    valid_block_ids: list[int],
    grid_block_ids: tuple[int, ...] = (),
) -> SimpleNamespace:
    loop_axes = (
        (LoopAxisFact(fact.k_block_id, fact.static_k),)
        if axes.k_kind is DotAxisKind.TUNABLE_TILED and fact.k_block_id is not None
        else ()
    )
    mm = SimpleNamespace(
        matmuls=(
            SimpleNamespace(
                fact=fact,
                axes=axes,
                site=SimpleNamespace(
                    graph_id=0,
                    updates_carry=False,
                    loop_axes=loop_axes,
                    exact_loop_trips=None,
                    max_loop_trips=max(1, fact.static_k or 1),
                ),
            ),
        ),
        knob_users=(),
        sequential_loop_trips=1,
        live_tile_steps=(),
        pipelined_regions=(),
        resident_regions=(),
        attribution_complete=True,
    )
    spec = SimpleNamespace(
        matmul_facts=[fact],
        kernel_matmul_fact=mm,
        kernel_grid_fact=None,
        block_sizes=_block_sizes_stub(valid_block_ids),
        grid_block_ids=grid_block_ids,
        _base_default_config=lambda: helion.Config(
            block_sizes=[1] * len(valid_block_ids)
        ),
    )
    spec.autotune_reference_config = lambda: spec._base_default_config()
    return spec


def test_generalized_gate_admits_a_fixed_contraction_axis() -> None:
    """A dot whose K is a specialized full extent has no ``block_k`` to set. The incumbent
    gate declines it for that reason alone; the generalized gate admits it, because a fixed
    axis is a smaller set of knobs and not a smaller problem."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=4,  # registered but NOT tunable
        static_m=256,
        static_n=256,
        static_k=64,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fact,
        _axes(k=DotAxisKind.FIXED_FULL_EXTENT, k_extent=64),
        valid_block_ids=[0, 1],
    )
    assert _generalized_static_matmul_fact(spec) is fact
    # ...and the incumbent gate still declines it, which is what made this widening needed.
    assert _batched_static_matmul_fact(spec) is None


def test_generalized_gate_admits_zero_tunable_dot_axes() -> None:
    """A kernel that exposes no tile at all still wants num_warps / num_stages; the
    alternative is the bare fragment default."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=None,
        n_block_id=None,
        k_block_id=None,
        static_m=64,
        static_n=64,
        static_k=64,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fact,
        _axes(
            DotAxisKind.FIXED_FULL_EXTENT,
            DotAxisKind.FIXED_FULL_EXTENT,
            DotAxisKind.FIXED_FULL_EXTENT,
            m_extent=64,
            n_extent=64,
            k_extent=64,
        ),
        valid_block_ids=[],
    )
    assert _generalized_static_matmul_fact(spec) is fact


def test_generalized_gate_declines_two_axes_sharing_one_knob() -> None:
    """Two tunable axes on one block id is a genuine conflict that must be RANKED, which is
    front end 2's job -- front end 1 has no way to arbitrate it."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=1,  # same knob as N
        static_m=256,
        static_n=256,
        static_k=256,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1])
    assert _generalized_static_matmul_fact(spec) is None


def test_generalized_gate_declines_an_unknown_extent() -> None:
    """No static extent means nothing can be sized; a dynamic/jagged dot must not be
    silently configured from a guess."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=2,
        static_m=256,
        static_n=256,
        static_k=None,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fact, _axes(k=DotAxisKind.UNKNOWN, k_extent=None), valid_block_ids=[0, 1, 2]
    )
    assert _generalized_static_matmul_fact(spec) is None


def test_tmem_is_counted_in_columns_and_sums_over_live_accumulators() -> None:
    """tcgen05 tensor memory is allocated as 128 lanes x N columns of 32 bits, so a tile
    costs ``ceil(bm/128) * bn`` COLUMNS and a bm<128 accumulator costs the same as a
    full-lane one. Measured on B200: a kernel's ``tmem_size`` equals its accumulator's N
    extent exactly ([64,64] -> 64, [128,128] -> 128, [128,256] -> 256)."""
    assert _FML._tmem_columns([(64, 64)]) == 64
    assert _FML._tmem_columns([(128, 128)]) == 128
    assert _FML._tmem_columns([(128, 256)]) == 256
    # A byte model divides by the lanes a narrow accumulator does not use; the column model
    # does not, which is the whole point.
    assert _FML._tmem_columns([(64, 256)]) == 256
    # bm past a lane group needs a second one.
    assert _FML._tmem_columns([(256, 256)]) == 512
    # Live accumulators ADD -- this is the measured failure
    # (``tensor memory, Required: 768, limit 512``) reproduced as arithmetic.
    assert _FML._tmem_columns([(128, 256)] * 3) == 768
    assert _FML._tmem_columns([(128, 256)] * 3) > _FML.TMEM_COLUMN_BUDGET
    # ...and one accumulator under the incumbent caps never binds, so the single-GEMM path
    # is unaffected by adding this check.
    assert _FML._tmem_columns([(128, _FML.BASE_BN_CAP)]) <= _FML.TMEM_COLUMN_BUDGET
    # Below the tcgen05 minimum the dot uses no tensor memory at all (measured
    # ``tmem_size == 0``), so charging it would reject a tiny tile for a resource it never
    # touches.
    assert _FML._tmem_columns([(32, 256)]) == 0
    # sm90 has no tensor memory: the check must be completely inert there.
    assert TritonH100MatmulHeuristic._tmem_columns([(128, 256)] * 8) == 0


def test_tmem_hard_budget_includes_all_lhs_promotion_scratch() -> None:
    """Crossing the tcgen05 M threshold also allocates promoted-LHS scratch.

    Five backward-attention accumulators consume exactly 512 columns; the
    compiler required at least another 64 columns. Omitting LHS scratch emitted
    a configuration that failed at launch with ``Required: 576, limit: 512``;
    the hard bound conservatively reserves every dot's allocation.
    """
    tiles = [
        (128, 64, 128, 2),
        (128, 64, 128, 2),
        (128, 128, 64, 2),
        (64, 128, 128, 2),
        (128, 128, 64, 2),
    ]
    accumulators = [(bm, bn) for bm, bn, _bk, _itemsize in tiles]
    lhs_scratch = [(bm, bk, itemsize) for bm, _bn, bk, itemsize in tiles]
    assert _FML._tmem_columns(accumulators) == 512
    assert _FML._tmem_columns(accumulators, lhs_scratch_tiles=lhs_scratch) == 736
    assert (
        _FML._tmem_columns(accumulators, lhs_scratch_tiles=lhs_scratch)
        > _FML.TMEM_COLUMN_BUDGET
    )


def test_optimistic_tmem_uses_peak_live_accumulators_and_transformed_lhs() -> None:
    env, mm = _knob_spec(
        knob_users=(
            (0, ((0, "m"), (1, "m"))),
            (1, ((0, "n"), (1, "n"))),
            (2, ((0, "k"), (1, "k"))),
        ),
        block_ids=[0, 1, 2],
        extents={0: 1024, 1: 1024, 2: 1024},
    )
    fact = _matmul_fact()
    mm.matmuls = (
        SimpleNamespace(fact=fact, axes=_axes()),
        SimpleNamespace(fact=fact, axes=_axes()),
    )
    output = LiveTile((0, 1), (None, None), 4, "dot_out")
    transformed_lhs = LiveTile(
        (0, 2),
        (None, None),
        2,
        "other",
        promoted_lhs=True,
    )
    mm.live_tile_steps = ((output,), (output, transformed_lhs))
    mm.live_dot_outputs = (output,)
    mm.live_promoted_lhs = (transformed_lhs,)
    blocks = [128, 128, 64]

    assert (
        _FML._candidate_tmem_columns(
            env,
            blocks,
            include_lhs_scratch=True,
            resource_policy="strict",
        )
        == 320
    )
    assert (
        _FML._candidate_tmem_columns(
            env,
            blocks,
            include_lhs_scratch=True,
            resource_policy="optimistic",
        )
        == 160
    )

    mm.live_tile_steps = ((output, transformed_lhs, transformed_lhs),)
    mm.live_promoted_lhs = (transformed_lhs, transformed_lhs)
    assert (
        _FML._candidate_tmem_columns(
            env,
            blocks,
            include_lhs_scratch=True,
            resource_policy="optimistic",
        )
        == 192
    )

    mm.live_tile_steps = ((output,), (output,))
    mm.live_promoted_lhs = ()
    assert (
        _FML._candidate_tmem_columns(
            env,
            blocks,
            include_lhs_scratch=True,
            resource_policy="optimistic",
        )
        == 128
    )

    batched_output = LiveTile(
        (None, 0, 1),
        (16, None, None),
        4,
        "dot_out",
    )
    mm.live_tile_steps = ((batched_output,),)
    mm.live_dot_outputs = (batched_output,)
    assert (
        _FML._candidate_tmem_columns(
            env,
            blocks,
            include_lhs_scratch=True,
            resource_policy="optimistic",
        )
        == 0
    )


def test_register_estimate_picks_its_peak_by_resolved_bytes() -> None:
    """The register estimate must select its peak step by RESOLVED BYTES at the candidate
    config, not by the block-size-free rank profile the reduction liveness machinery uses.

    Selecting by rank picked a step holding several rank-2 loads over the step that actually
    holds the accumulators. Measured against post-ptxas spills at one warp, that inverted the
    ordering: a kernel spilling 540 registers estimated 1.33x the one-warp file while one
    spilling none estimated 1.51x, so no threshold could separate them. With the byte-selected
    peak the same 12 cells separate cleanly -- every zero-spill cell at or below 1.51x, every
    spilling cell at or above 2.30x."""
    from helion.autotuner.config_spec import LiveTile

    def tile(kind: str, rows: int, cols: int, itemsize: int = 4) -> LiveTile:
        return LiveTile(
            dim_block_ids=(None, None),
            static_dims=(rows, cols),
            itemsize=itemsize,
            kind=kind,
        )

    def env_with(steps: tuple) -> SimpleNamespace:
        return SimpleNamespace(
            config_spec=SimpleNamespace(
                kernel_matmul_fact=SimpleNamespace(
                    live_tile_steps=steps,
                    matmuls=(),
                ),
                block_sizes=_block_sizes_stub([]),
                _base_default_config=lambda: helion.Config(block_sizes=[]),
            ),
            block_sizes=[],
        )

    # The peak must be the BYTE-heaviest step, not the one with the most tiles: three small
    # values must not outrank one large one.
    many_small = tuple(tile("other", 8, 8) for _ in range(3))
    one_large = (tile("other", 128, 128),)
    env = env_with((many_small, one_large))
    assert _FML._register_live_bytes(env, [], 4) == 128 * 128 * 4

    # Loads are excluded: they are charged to the shared-memory ring, and charging them here
    # would put the same bytes in two budgets.
    assert _FML._register_live_bytes(env_with(((tile("load", 128, 128),),)), [], 4) == 0

    # A value larger than the largest register file a CTA can have is not register-resident --
    # it lives in HBM and the graph merely names it (a varlen packed buffer measured 256 MiB).
    huge = (tile("other", 8192, 64),)
    assert _FML._register_live_bytes(env_with((huge,)), [], 1) == 0

    # A dot output is charged only where tensor memory cannot absorb it. Below a warpgroup
    # there is no tcgen05 path at all (PTX: num_warps 1 or 2 emits zero tcgen05.mma and
    # tmem_size 0), so the accumulator lands in registers; at 4 warps with bm >= the tcgen05
    # minimum it does not.
    acc = (tile("dot_out", 64, 64),)
    assert _FML._register_live_bytes(env_with((acc,)), [], 1) == 64 * 64 * 4
    assert _FML._register_live_bytes(env_with((acc,)), [], 2) == 64 * 64 * 4
    assert _FML._register_live_bytes(env_with((acc,)), [], 4) == 0
    # ...and a dot below the tcgen05 minimum stays register-resident at any warp count.
    narrow = (tile("dot_out", 32, 64),)
    assert _FML._register_live_bytes(env_with((narrow,)), [], 8) == 32 * 64 * 4
    batched = (LiveTile((None, None, None), (2, 64, 64), 4, "dot_out"),)
    assert _FML._register_live_bytes(env_with((batched,)), [], 4) == 2 * 64 * 64 * 4


def test_graded_stage_depth_falls_off_with_outer_parallelism() -> None:
    """Depth is bought with shared memory per CTA, and shared memory per CTA is what limits
    how many CTAs an SM holds. Below one wave there is no co-residency to protect and depth
    is the only latency hiding available; above it, every extra stage evicts a CTA. So the
    depth must fall off GRADUALLY with the grid -- which the incumbent single threshold at
    ``SAT_WAVES * num_sm`` cannot express, and which matches the hand-tuned corpus (outer
    grid 32 -> 8-11 stages, 96 -> 3-4, 256 -> 2-4, >=1024 -> 2)."""
    per_stage = 16 * 1024

    def smem_of(stages: int) -> int:
        return per_stage * stages

    depths = [
        _FML._graded_stage_depth(smem_of, loop_trips=256, grid=grid, num_sm=148)
        for grid in (32, 96, 148, 296, 1024, 16384)
    ]
    assert depths == sorted(depths, reverse=True), depths
    assert depths[0] > depths[-1]
    # The divisor is CLAMPED at GRADED_MAX_CTAS_PER_SM, so the gradient SATURATES rather
    # than running to the floor: a grid far above the machine size does not demand a
    # matching number of simultaneously-resident CTAs (the excess queues), and dividing by
    # the raw wave count instead collapsed every large-grid kernel to a single stage
    # (measured: an outer grid of 8192 on 148 SMs gives a 4 KiB per-CTA share).
    assert depths[-1] == depths[-2]
    assert (
        _FML._graded_stage_depth(smem_of, loop_trips=256, grid=10**6, num_sm=148)
        == depths[-1]
    )
    # An empty machine reaches depths the incumbent MAX_STAGES ceiling cannot express.
    assert depths[0] > _FML.MAX_STAGES or _FML.HW_MAX_STAGES <= _FML.MAX_STAGES
    assert depths[0] <= _FML.HW_MAX_STAGES


def test_graded_stage_depth_is_capped_by_the_loop_it_pipelines() -> None:
    """Real loop trips cap deep lookahead; one trip conditionally admits stage two."""

    def cheap(stages: int) -> int:
        return 1024 * stages

    assert _FML._graded_stage_depth(cheap, loop_trips=3, grid=1, num_sm=148) == 3
    assert _FML._graded_stage_depth(cheap, loop_trips=1, grid=1, num_sm=148) == 2
    assert (
        _FML._graded_stage_depth(
            cheap,
            loop_trips=1,
            grid=1,
            num_sm=148,
            allow_one_trip_stage2=False,
        )
        == 1
    )


def test_warp_transition_penalty_uses_effective_residency() -> None:
    """Queued waves do not increase the penalty after resident capacity is full."""
    penalty = _FML._warp_transition_occupancy_penalty
    assert penalty(1, 1) == 1.0
    assert penalty(4, 4) == 1.0
    assert penalty(4, 2) == 2.0
    assert penalty(32, 1) == _FML.WARP_TRANSITION_OCCUPANCY_PENALTY_MAX


def test_warp_selection_accounts_for_four_to_eight_residency_loss() -> None:
    """Eight warps must earn back any residency lost relative to four."""
    work = SimpleNamespace(
        total=_FML.EIGHT_WARP_DOT_WORK,
        tcgen05_eligible=_FML.EIGHT_WARP_DOT_WORK,
        uncertain=False,
    )

    def select(eight_warp_residency: int) -> int:
        residency = {2: 2, 4: 2, 8: eight_warp_residency}
        with (
            patch.object(_FML, "_candidate_dot_work", return_value=work),
            patch.object(_FML, "_register_live_bytes", return_value=0),
            patch.object(_FML, "_all_dot_acc_tiles", return_value=[]),
            patch.object(
                _FML,
                "_estimated_resident_ctas",
                side_effect=lambda *_args, num_warps, **_kwargs: residency[num_warps],
            ),
        ):
            return _FML._select_num_warps(
                1,
                SimpleNamespace(),
                [],
                grid=148,
                num_sm=148,
                smem_bytes=0,
            )

    assert select(2) == 8
    assert select(1) == 4


def test_multi_matmul_ranking_prefers_a_carried_accumulator_then_work() -> None:
    """A dot feeding a loop-carried accumulator holds that accumulator resident for the whole
    loop, so its tile sets the kernel's whole-loop footprint -- hence ranking priority. But
    it is a PREFERENCE: dimensions and execution count must also matter, and a kernel with no
    carried accumulator has to rank purely on work."""
    big = MatmulFact(
        2, 2, None, None, None, 256, 256, 256, torch.bfloat16, torch.bfloat16
    )
    small = MatmulFact(
        2, 2, None, None, None, 64, 64, 64, torch.bfloat16, torch.bfloat16
    )

    def mm(carry: tuple[bool, ...], trips: tuple[int, ...]) -> SimpleNamespace:
        facts = (big, small)
        sites = tuple(
            SimpleNamespace(
                graph_id=0,
                updates_carry=c,
                loop_axes=(LoopAxisFact(0, 64 * t),),
                exact_loop_trips=None,
                max_loop_trips=t,
            )
            for c, t in zip(carry, trips, strict=True)
        )
        return SimpleNamespace(
            matmuls=tuple(
                SimpleNamespace(fact=fact, axes=axes, site=site)
                for fact, axes, site in zip(
                    facts,
                    (
                        _axes(
                            DotAxisKind.FIXED_FULL_EXTENT,
                            DotAxisKind.FIXED_FULL_EXTENT,
                            DotAxisKind.FIXED_FULL_EXTENT,
                            m_extent=256,
                            n_extent=256,
                            k_extent=256,
                        ),
                        _axes(
                            DotAxisKind.FIXED_FULL_EXTENT,
                            DotAxisKind.FIXED_FULL_EXTENT,
                            DotAxisKind.FIXED_FULL_EXTENT,
                            m_extent=64,
                            n_extent=64,
                            k_extent=64,
                        ),
                    ),
                    sites,
                    strict=True,
                )
            ),
            attribution_complete=True,
        )

    def rank(mm: SimpleNamespace, index: int) -> tuple[int, int, int]:
        spec = SimpleNamespace(
            kernel_matmul_fact=mm,
            block_sizes=_block_sizes_stub([0]),
            _base_default_config=lambda: helion.Config(block_sizes=[64]),
        )
        env = SimpleNamespace(
            config_spec=spec,
            block_sizes=[SimpleNamespace(size=64 * 4096)],
            size_hint=int,
        )
        return _MULTI._rank_key(env, mm, index, [64])

    # The carried dot wins even though it does far less work.
    f = mm((False, True), (1, 1))
    assert rank(f, 1) > rank(f, 0)
    # With no carry anywhere, work decides.
    f = mm((False, False), (1, 1))
    assert rank(f, 0) > rank(f, 1)
    # Execution count is part of work, so a small dot run many times can outrank a big one.
    f = mm((False, False), (1, 4096))
    assert rank(f, 1) > rank(f, 0)
    # Untrusted attribution must collapse the carry term for EVERY dot equally, so ranking
    # degrades to pure work rather than to an arbitrary order.
    f = mm((False, True), (1, 1))
    f.attribution_complete = False
    assert rank(f, 0) > rank(f, 1)


def test_candidate_dot_work_counts_serial_attention_axis_once() -> None:
    """A serial key axis contributes candidate width times candidate trip count."""
    qk = MatmulFact(2, 2, 0, 1, None, 1024, 1024, 64, torch.bfloat16, torch.bfloat16)
    pv = MatmulFact(2, 2, 0, None, 1, 1024, 64, 1024, torch.bfloat16, torch.bfloat16)
    site = SimpleNamespace(
        graph_id=0,
        updates_carry=False,
        loop_axes=(LoopAxisFact(1, 1024),),
        exact_loop_trips=None,
        max_loop_trips=1024,
    )
    mm = SimpleNamespace(
        matmuls=(
            SimpleNamespace(
                fact=qk,
                axes=_axes(
                    k=DotAxisKind.FIXED_FULL_EXTENT,
                    k_extent=64,
                ),
                site=site,
            ),
            SimpleNamespace(
                fact=pv,
                axes=_axes(
                    n=DotAxisKind.FIXED_FULL_EXTENT,
                    n_extent=64,
                ),
                site=site,
            ),
        ),
    )
    spec = SimpleNamespace(
        kernel_matmul_fact=mm,
        block_sizes=_block_sizes_stub([0, 1]),
        _base_default_config=lambda: helion.Config(block_sizes=[128, 64]),
    )
    env = SimpleNamespace(
        config_spec=spec,
        block_sizes=[SimpleNamespace(size=1024), SimpleNamespace(size=1024)],
        size_hint=int,
    )

    work = _FML._candidate_dot_work(env, [128, 64])

    per_dot = 128 * 1024 * 64
    assert work.total == 2 * per_dot
    assert work.uncertain is False


def test_projection_is_a_no_op_for_a_clean_gemm() -> None:
    """With three tunable axes and no extra live accumulator, ``_tile_for_dot`` must return
    the incumbent proposal untouched -- that is what makes the whole widening safe for the
    GEMM/BMM/split-K workloads it was not measured on."""
    fact = _matmul_fact(static_m=4096, static_n=4096, static_k=4096)
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1, 2])
    env = SimpleNamespace(config_spec=spec, block_sizes=[])
    proposal = _FML._matmul_tile(4096, 4096, 4096, 2, 148, 1)
    assert (
        _FML._tile_for_dot(
            env,
            fact,
            _axes(),
            2,
            148,
            site=spec.kernel_matmul_fact.matmuls[0].site,
        )
        == proposal
    )


def _kernel_smem_env(
    facts: tuple[MatmulFact, ...],
    *,
    block_ids: list[int],
    pipelined_regions: tuple[PipelinedRegion, ...] = (),
) -> SimpleNamespace:
    matmuls = tuple(
        SimpleNamespace(fact=fact, axes=_axes(), site=SimpleNamespace(graph_id=0))
        for fact in facts
    )
    spec = SimpleNamespace(
        kernel_matmul_fact=SimpleNamespace(
            matmuls=matmuls,
            pipelined_regions=pipelined_regions,
            resident_regions=(),
        ),
        block_sizes=_block_sizes_stub(block_ids),
        _base_default_config=lambda: helion.Config(block_sizes=[1] * len(block_ids)),
    )
    return SimpleNamespace(config_spec=spec, block_sizes=[], size_hint=int)


def test_kernel_smem_uses_region_peak_and_applies_slack_once() -> None:
    fact = _matmul_fact()
    loads = (
        LiveTile((0, 2), (None, None), 2, "load"),
        LiveTile((2, 1), (None, None), 2, "load"),
    )
    env = _kernel_smem_env(
        (fact,),
        block_ids=[0, 1, 2],
        pipelined_regions=(PipelinedRegion((LoopAxisFact(2, 64),), loads),),
    )
    blocks = [64, 32, 16]
    region_peak = (64 * 16 * 2 + 16 * 32 * 2) * 4

    assert _FML._smem_region_demands(env, blocks, 4) == (region_peak,)
    assert _FML._kernel_smem_bytes(env, blocks, 4) == region_peak + _FML.SMEM_SLACK


def test_kernel_smem_separates_useful_depth_from_hard_allocation() -> None:
    fact = _matmul_fact()
    loads = (
        _live("load", 0, 2),
        _live("load", 2, 1),
    )
    env = _kernel_smem_env(
        (fact,),
        block_ids=[0, 1, 2],
        pipelined_regions=(PipelinedRegion((LoopAxisFact(2, 32),), loads),),
    )
    blocks = [64, 32, 16]

    per_stage = 64 * 16 * 2 + 16 * 32 * 2
    assert _FML._smem_region_demands(env, blocks, 6) == (per_stage * 2,)
    assert _FML._smem_region_demands(
        env,
        blocks,
        6,
        hard_allocation=True,
    ) == (per_stage * 6,)


def test_symbolic_loop_bounds_resolve_candidates_and_preserve_unknowns() -> None:
    outer_block = sympy.Symbol("outer_block", integer=True, positive=True)
    outer_tile_id = sympy.Symbol("outer_tile_id", integer=True, nonnegative=True)
    spec = SimpleNamespace(
        block_sizes=_block_sizes_stub([0, 1]),
        _base_default_config=lambda: helion.Config(block_sizes=[1, 1]),
    )
    env = SimpleNamespace(
        config_spec=spec,
        block_sizes=[
            SimpleNamespace(size=1024),
            SimpleNamespace(size=object()),
        ],
        size_hint=int,
    )
    loop_axes = (
        LoopAxisFact(
            block_id=1,
            extent=None,
            symbolic_bound=SymbolicLoopBound(
                2 * outer_block * (outer_tile_id + 1) + 16,
                block_size_symbols=((outer_block, 0),),
                tile_id_symbols=((outer_tile_id, 0),),
            ),
        ),
    )

    trips = _FML._resolved_loop_trips(
        env,
        [128, 64],
        loop_axes,
    )

    # Eight outer tiles have lower-median ID 3. The arbitrary bound evaluates
    # to 2 * 128 * (3 + 1) + 16 = 1040, or 17 inner 64-element tiles.
    assert trips == 17
    unresolved = sympy.Symbol("unresolved", integer=True, positive=True)
    axis = LoopAxisFact(
        block_id=1,
        extent=None,
        symbolic_bound=SymbolicLoopBound(
            outer_block * (outer_tile_id + 1) + unresolved,
            block_size_symbols=((outer_block, 0),),
            tile_id_symbols=((outer_tile_id, 0),),
        ),
    )

    assert _FML._resolved_loop_trips(env, [128, 64], (axis,)) is None


def test_kernel_smem_uses_latest_complete_map_and_max_dot_epilogue() -> None:
    first = _matmul_fact()
    second = MatmulFact(
        2,
        2,
        3,
        4,
        2,
        1024,
        1024,
        1024,
        torch.bfloat16,
        torch.bfloat16,
    )
    env = _kernel_smem_env((first, second), block_ids=[0, 1, 2, 3, 4])

    assert _FML._kernel_smem_demands(env, [16, 32, 8, 64, 64], 3) == (
        64 * 64 * 4 + _FML.SMEM_SLACK,
    )
    assert _FML._kernel_smem_bytes(env, [16, 32, 8, 32, 32], 3) == (
        32 * 32 * 4 + _FML.SMEM_SLACK
    )


def test_multi_dot_proposal_preconditions_with_complete_kernel_smem_map() -> None:
    fact = _matmul_fact()
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1, 2, 7])
    env = SimpleNamespace(config_spec=spec, block_sizes=[])
    seen: list[tuple[int, ...]] = []

    def record_smem(_env: object, block_sizes: list[int], _stages: int) -> int:
        seen.append(tuple(block_sizes))
        return 0

    with patch.object(
        _MULTI,
        "_kernel_smem_bytes",
        side_effect=record_smem,
    ):
        _MULTI._tile_for_dot(
            env,
            fact,
            _axes(),
            2,
            148,
            site=spec.kernel_matmul_fact.matmuls[0].site,
        )
    assert seen
    assert all(len(block_sizes) == len(spec.block_sizes) for block_sizes in seen)
    assert all(block_sizes[3] == 1 for block_sizes in seen)


def test_sm90_keeps_the_incumbent_gate_and_every_switch_off() -> None:
    """Every measurement behind the generalized machinery is B200, so sm90 must be a
    byte-identical freeze: same eligibility precondition, no graded stages, no work-aware
    warps, no tensor-memory column budget."""
    cls = TritonH100MatmulHeuristic
    assert cls.GENERALIZED_AXES is False
    assert cls.GRADED_STAGES is False
    assert cls.WORK_AWARE_WARPS is False
    assert cls.TMEM_COLUMN_BUDGET is None
    fixed = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=4,
        static_m=256,
        static_n=256,
        static_k=64,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fixed,
        _axes(k=DotAxisKind.FIXED_FULL_EXTENT, k_extent=64),
        valid_block_ids=[0, 1],
    )
    # sm90 routes through the incumbent gate, which declines the fixed-axis dot.
    assert cls._eligible_fact(spec) is None
    # ...while sm100 admits it.
    assert _FML._eligible_fact(spec) is fixed


def test_multi_matmul_front_end_declines_whatever_front_end_one_owns() -> None:
    """Exactly one of the two front ends may own a kernel, so promotion is unambiguous
    without either of them having to know the other's policy."""
    fact = _matmul_fact()
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1, 2])
    env = SimpleNamespace(
        config_spec=spec,
        device=torch.device("cuda"),
        settings=SimpleNamespace(),
    )
    with patch(
        "helion._compiler.autotuner_heuristics.triton.matches_hardware",
        return_value=True,
    ):
        assert _MULTI.is_eligible(env, None) is False
    # Keep single-contraction seeds ahead of the disjoint multi-contraction path.
    from helion._compiler.autotuner_heuristics import HEURISTICS_BY_BACKEND

    triton_order = HEURISTICS_BY_BACKEND["triton"]
    assert triton_order.index(_MULTI) > triton_order.index(_FML)


def _live(kind: str, *block_ids: int | None) -> LiveTile:
    return LiveTile(
        dim_block_ids=tuple(block_ids),
        static_dims=tuple(None for _ in block_ids),
        itemsize=2,
        kind=kind,
    )


def _knob_spec(
    *,
    knob_users: tuple[tuple[int, tuple[tuple[int, str], ...]], ...],
    block_ids: list[int],
    extents: dict[int, object],
    grid_block_ids: tuple[int, ...] = (),
    pipelined_regions: tuple[PipelinedRegion, ...] = (),
) -> SimpleNamespace:
    """An ``env`` stub carrying just what ``_apply_knob_roles`` reads."""
    mm = SimpleNamespace(
        matmuls=(),
        knob_users=knob_users,
        sequential_loop_trips=1,
        live_tile_steps=(),
        pipelined_regions=pipelined_regions,
        resident_regions=(),
        attribution_complete=True,
    )
    spec = SimpleNamespace(
        matmul_facts=[],
        kernel_matmul_fact=mm,
        kernel_grid_fact=None,
        block_sizes=_block_sizes_stub(block_ids),
        grid_block_ids=grid_block_ids,
        _base_default_config=lambda: helion.Config(block_sizes=[1] * len(block_ids)),
    )
    spec.autotune_reference_config = lambda: spec._base_default_config()
    env = SimpleNamespace(
        config_spec=spec,
        block_sizes=[
            SimpleNamespace(
                size=extents.get(b, 1),
                block_size_source=SimpleNamespace(
                    from_config=lambda _config, info: info.size
                ),
            )
            for b in range(max(extents or {0: 0}) + 1)
        ],
        size_hint=lambda v: int(v),
    )
    return env, mm


def test_launch_grid_counts_only_grid_axes() -> None:
    """A knob that is walked by a SEQUENTIAL loop contributes iterations, not programs. The
    budget formula's wave model counts every M/N tile as a program, which reports a saturated
    machine for a kernel that launches a handful of CTAs."""
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "n"),)), (1, ((0, "m"),))),
        block_ids=[0, 1],
        extents={0: 128, 1: 128},
        grid_block_ids=(1,),
    )
    # Only block id 1 is a grid axis: halving the LOOP axis 0 must not change the grid,
    # halving the grid axis must double it.
    assert _MULTI._launch_grid(env, [128, 128]) == 1
    assert _MULTI._launch_grid(env, [32, 128]) == 1
    assert _MULTI._launch_grid(env, [128, 32]) == 4


def test_dot_proposal_uses_only_proven_grid_axes() -> None:
    """The formula callback sees only proven grid axes, never serial M/N tiles."""
    env, mm = _knob_spec(
        knob_users=((0, ((0, "m"),)), (1, ((0, "n"),))),
        block_ids=[0, 1],
        extents={0: 1024, 1: 64},
        grid_block_ids=(1,),
    )
    fact = MatmulFact(
        2,
        2,
        0,
        1,
        None,
        1024,
        64,
        64,
        torch.bfloat16,
        torch.bfloat16,
    )
    axes = _axes(
        k=DotAxisKind.FIXED_FULL_EXTENT,
        m_extent=1024,
        n_extent=64,
        k_extent=64,
    )
    site = SimpleNamespace(graph_id=0, loop_axes=())
    mm.matmuls = (SimpleNamespace(fact=fact, axes=axes, site=site),)
    observed: dict[str, object] = {}

    def capture_formula(
        _m: int,
        _n: int,
        _k: int,
        _itemsize: int,
        _num_sm: int,
        _pinned_grid: int,
        *,
        launch_grid: Any,
        allow_l2_grouping: bool,
    ) -> tuple[int, int, int, int, int, int]:
        observed["grids"] = (launch_grid(32, 64), launch_grid(512, 64))
        observed["allow_l2_grouping"] = allow_l2_grouping
        return 128, 64, 64, 4, 2, 1

    with patch.object(_FML, "_matmul_tile", side_effect=capture_formula):
        _FML._projected_tile_for_dot(
            env,
            fact,
            axes,
            itemsize=2,
            num_sm=148,
            site=site,
        )

    assert observed["grids"] == (1, 1)
    assert observed["allow_l2_grouping"] is False


def test_launch_grid_uses_only_dot_bearing_root_groups() -> None:
    """Independent top-level loops add their grids, while a matmul policy sizes against
    only roots that execute a dot. Nested dot graphs resolve through the generic kernel
    grid fact rather than carrying a copied block-id tuple."""
    env, mm = _knob_spec(
        knob_users=(),
        block_ids=[0, 1, 2, 4, 5],
        extents={0: 128, 1: 128, 2: 128, 4: 256, 5: 256},
        grid_block_ids=(0, 1, 2, 4, 5),
    )
    grid_fact = KernelGridFact(
        roots=(
            RootGridFact(100, (0, 1, 2)),
            RootGridFact(200, (4, 5)),
        ),
        graph_to_root=((100, 100), (101, 100), (200, 200)),
    )
    env.config_spec.kernel_grid_fact = grid_fact
    mm.matmuls = (SimpleNamespace(site=SimpleNamespace(graph_id=101)),)

    # The nested dot belongs to root 100: 2 * 2 * 2 = 8 programs. Root 200's
    # 4 * 4 = 16 programs do not execute the dot and must not inflate its waves.
    assert grid_fact.group_for_graph(101) == (0, 1, 2)
    assert _MULTI._launch_grid(env, [64, 64, 64, 64, 64]) == 8

    # Without a dot-bearing subset, evaluate the complete independent-root grid
    # as a sum, 8 + 16, never as a product of the flattened block-id list.
    mm.matmuls = ()
    assert _MULTI._launch_grid(env, [64, 64, 64, 64, 64]) == 24


def test_knob_amortization_separates_reuse_free_from_reuse_bearing() -> None:
    """The discriminator for whether growing a tile buys anything: does its loop region stage
    a load the knob does NOT span? If every load spans the knob, bytes and MMA work both
    scale with the tile and arithmetic intensity is constant in it."""
    reuse_free = (_live("load", 9, 4), _live("store", 9, 4))
    reuse_bearing = (_live("load", 9, 4), _live("load", 9, 3), _live("store", 9, 4))
    _env, mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=(PipelinedRegion((), reuse_free),),
    )
    assert _MULTI._knob_amortizes(mm, 4) is False
    _env, mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=(PipelinedRegion((), reuse_bearing),),
    )
    # Block id 3 is re-fetched for every iteration of 4, so growing 4 amortizes it.
    assert _MULTI._knob_amortizes(mm, 4) is True
    # ...and symmetrically, 3's own loop re-fetches the 4-spanning load, so 3 amortizes too.
    assert _MULTI._knob_amortizes(mm, 3) is True

    _env, mm = _knob_spec(
        knob_users=((4, ((0, "m"),)),),
        block_ids=[4, 5],
        extents={4: 128, 5: 128},
        pipelined_regions=(
            PipelinedRegion(
                (LoopAxisFact(5, 128), LoopAxisFact(4, 128)),
                (_live("load", 9, 5),),
            ),
        ),
    )
    # The inner loop's load does not span its enclosing outer axis. Growing
    # that outer tile therefore amortizes the inner scan.
    assert _MULTI._knob_amortizes(mm, 4) is True


def test_reuse_free_output_knob_drops_to_the_allocation_floor() -> None:
    """A reuse-free knob that sizes a dot OUTPUT extent buys no arithmetic intensity while
    the fp32 accumulator, the register-resident intermediates and the store staging all scale
    with it, so it goes to the tcgen05 allocation granularity. A reuse-BEARING knob in the
    same position must be left alone."""
    env, _mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=(
            PipelinedRegion((), (_live("load", 9, 4), _live("store", 9, 4))),
        ),
    )
    block_sizes = [128]
    _MULTI._apply_knob_roles(env, env.config_spec.kernel_matmul_fact, block_sizes, 148)
    assert block_sizes == [_MULTI.TMEM_ALLOC_COLUMNS]

    env, _mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=(
            PipelinedRegion((), (_live("load", 9, 4), _live("load", 9, 3))),
        ),
    )
    block_sizes = [128]
    _MULTI._apply_knob_roles(env, env.config_spec.kernel_matmul_fact, block_sizes, 148)
    assert block_sizes == [128]


def test_grid_knob_shrinks_only_when_wave_utilization_improves() -> None:
    """A knob that IS the grid trades tile area against occupancy, so it is sized by the
    machine: shrink while the launch grid is under one wave and wave utilization strictly
    improves, then stop at the legal block minimum."""
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "m"), (1, "n"))),),
        block_ids=[0],
        extents={0: 8192},
        grid_block_ids=(0,),
        pipelined_regions=(
            PipelinedRegion((), (_live("load", 9, 0), _live("load", 9, 7))),
        ),
    )
    block_sizes = [8192]
    _MULTI._apply_knob_roles(env, env.config_spec.kernel_matmul_fact, block_sizes, 148)
    # 8192 / 64 = 128 programs >= 0.8 * 148, so the shrink stops there rather than
    # continuing to the floor.
    assert block_sizes == [64]
    # When even a small tile cannot fill a wave, continue to the legal block
    # minimum. Tensor-memory column granularity is not a launch-grid constraint.
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "m"), (1, "n"))),),
        block_ids=[0],
        extents={0: 1024},
        grid_block_ids=(0,),
        pipelined_regions=(
            PipelinedRegion((), (_live("load", 9, 0), _live("load", 9, 7))),
        ),
    )
    block_sizes = [1024]
    _MULTI._apply_knob_roles(env, env.config_spec.kernel_matmul_fact, block_sizes, 148)
    assert block_sizes == [8]
    # Do not cross into a second partial wave when that leaves utilization unchanged:
    # 11008/128 = 86 programs in one wave, while 11008/64 = 172 in two waves.
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "m"),)),),
        block_ids=[0],
        extents={0: 11008},
        grid_block_ids=(0,),
        pipelined_regions=(
            PipelinedRegion((), (_live("load", 9, 0), _live("load", 9, 7))),
        ),
    )
    block_sizes = [128]
    _MULTI._apply_knob_roles(env, env.config_spec.kernel_matmul_fact, block_sizes, 148)
    assert block_sizes == [128]
    # A grid that already covers the machine is left alone.
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "m"),)),),
        block_ids=[0],
        extents={0: 65536},
        grid_block_ids=(0,),
        pipelined_regions=(
            PipelinedRegion((), (_live("load", 9, 0), _live("load", 9, 7))),
        ),
    )
    block_sizes = [128]
    _MULTI._apply_knob_roles(env, env.config_spec.kernel_matmul_fact, block_sizes, 148)
    assert block_sizes == [128]


def test_multi_seed_recipes_use_the_shared_bounded_fixups() -> None:
    env, mm = _knob_spec(
        knob_users=(
            (0, ((0, "m"),)),
            (1, ((0, "n"),)),
            (2, ((0, "k"),)),
        ),
        block_ids=[0, 1, 2],
        extents={0: 4096, 1: 4096, 2: 4096},
    )
    seen_stage_warps: list[int] = []
    assert _MULTI._seed_axis_ids({0: ("m",), 1: ("n", "k")}) == (
        (0,),
        (1,),
        (1,),
    )

    def solve_stages(
        _env: object,
        _blocks: list[int],
        *,
        num_warps: int,
        **_kwargs: object,
    ) -> int:
        seen_stage_warps.append(num_warps)
        return 3

    with (
        patch.object(_FML, "_select_num_warps", return_value=8),
        patch.object(_FML, "_kernel_smem_bytes", return_value=0),
        patch.object(_FML, "_solve_candidate_stages", side_effect=solve_stages),
        patch.object(
            _FML,
            "_fixup_candidate_resources",
            side_effect=lambda _env, _blocks, stages, warps, **_kwargs: (
                stages,
                warps,
            ),
        ),
    ):
        drafts = _FML._register_mma_seed_drafts(
            env,
            mm,
            MatmulSeedDraft([128, 256, 64], 8, 4),
            num_sm=148,
            axis_ids=((0,), (1,), (2,)),
        )

    assert [draft.num_warps for draft in drafts] == [1, 2]
    assert set(seen_stage_warps) == {1, 2}


def test_b200_single_matmul_emits_the_compact_seed_families() -> None:
    fact = _matmul_fact(static_m=4096, static_n=4096, static_k=4096)
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1, 2])
    spec.kernel_matmul_fact.knob_users = (
        (0, ((0, "m"),)),
        (1, ((0, "n"),)),
        (2, ((0, "k"),)),
    )
    spec.grid_block_ids = (7,)
    spec.kernel_grid_fact = SimpleNamespace(
        group_for_graph=lambda _graph_id: (7,),
        groups_for_graphs=lambda _graph_ids: ((7,),),
        grid_groups=((7,),),
    )
    spec.l2_groupings = _block_sizes_stub([0, 7])
    spec._base_default_config = lambda: helion.Config(
        block_sizes=[16, 16, 16],
        l2_groupings=[2, 3],
        num_warps=4,
        num_stages=1,
    )
    env = SimpleNamespace(
        config_spec=spec,
        block_sizes=[],
        device=None,
        size_hint=int,
    )

    with patch("helion.runtime.get_num_sm", return_value=148):
        ranked = _FML._ranked_configs(env, fact)

    assert ranked[0] == helion.Config(
        block_sizes=[64, 32, 64],
        l2_groupings=[2, 1],
        num_warps=4,
        num_stages=4,
    )
    assert (
        helion.Config(
            block_sizes=[128, 256, 64],
            l2_groupings=[2, 1],
            num_warps=8,
            num_stages=4,
        )
        in ranked
    )
    assert len(ranked) == 9
    assert all(config["l2_groupings"] == [2, 1] for config in ranked[1:])
    register_mma = [
        config
        for config in ranked
        if config["num_warps"] <= 2 and config["block_sizes"][2] <= 32
    ]
    assert len(register_mma) >= 2
    assert any(
        config["num_warps"] >= 4
        and config["block_sizes"][0] >= _FML.TCGEN05_MIN_BM
        and config["block_sizes"][2] == 32
        for config in ranked
    )
    assert ranked[-1]["num_warps"] >= _FML.TCGEN05_WARPGROUP_WARPS
    assert ranked[-1]["block_sizes"][0] >= _FML.TCGEN05_MIN_BM

    spec.kernel_matmul_fact.matmuls[0].site.loop_axes = (
        LoopAxisFact(2, 4096, bounded_by_block_id=0),
    )
    capped = _FML._projected_tile_for_dot(
        env,
        fact,
        _axes(),
        2,
        148,
        site=spec.kernel_matmul_fact.matmuls[0].site,
    )
    with patch("helion.runtime.get_num_sm", return_value=148):
        split_k_ranked = _FML._ranked_configs(env, fact)
    assert capped[:2] == (
        _FML.SAT_PARTITIONED_K_BM,
        _FML.SAT_PARTITIONED_K_BN,
    )
    assert ranked[1] in split_k_ranked


def test_multi_focal_seeds_default_fill_and_canonical_dedupe() -> None:
    fact = _matmul_fact()
    mm = SimpleNamespace(
        matmuls=tuple(
            SimpleNamespace(fact=fact, site=SimpleNamespace(graph_id=index % 2))
            for index in range(15)
        ),
        attribution_complete=True,
    )
    spec = SimpleNamespace(
        kernel_matmul_fact=mm,
        kernel_grid_fact=SimpleNamespace(
            group_for_graph=lambda graph_id: (0,) if graph_id == 0 else (7,)
        ),
        block_sizes=_block_sizes_stub([0, 1, 2, 7]),
        l2_groupings=_block_sizes_stub([0, 7]),
        _base_default_config=lambda: helion.Config(
            block_sizes=[16, 16, 16, 128],
            l2_groupings=[1, 1],
            num_warps=4,
            num_stages=2,
        ),
        _flat_fields=lambda: {
            "block_sizes": None,
            "l2_groupings": None,
            "num_warps": None,
            "num_stages": None,
        },
        allowed_pid_types=(),
    )
    spec.autotune_reference_config = lambda: spec._base_default_config()
    current_limit: list[int | None] = [None]

    def normalize(
        config: helion.Config | dict[str, object],
        *,
        _fix_invalid: bool,
    ) -> None:
        values = config.config if isinstance(config, helion.Config) else config
        values.setdefault("l2_groupings", [1, 1])
        if current_limit[0] is not None:
            blocks = cast("list[int]", values["block_sizes"])
            blocks[0] = min(
                current_limit[0],
                blocks[0],
            )

    spec.normalize = normalize
    spec._shrink_for_numel_constraints = lambda _config: None
    env = SimpleNamespace(config_spec=spec, block_sizes=[])
    invalid_primary = helion.Config(block_sizes=[3])
    alternate = helion.Config(block_sizes=[4])
    with patch.object(
        _MULTI,
        "_canonical_seed_config",
        side_effect=(None, alternate, alternate),
    ):
        assert _MULTI._dedupe_seed_pool(
            env,
            (invalid_primary, alternate, alternate),
        ) == [invalid_primary, alternate]
    proposals = {
        index: (
            16 << (index % 4),
            16 << (index // 4),
            16,
            1,
            2,
            1,
        )
        for index in range(15)
    }

    root_configs = _MULTI._focal_seed_configs(
        env,
        mm,
        dict.fromkeys(range(2), (16, 16, 16, 1, 2, 4)),
    )
    assert {tuple(config["l2_groupings"]) for config in root_configs} == {
        (4, 1),
        (1, 4),
    }

    for m_limit, expected in ((32, 8), (None, 15)):
        current_limit[0] = m_limit
        configs = _MULTI._focal_seed_configs(
            env,
            mm,
            proposals,
        )

        assert len(configs) == expected
        assert all(config["block_sizes"][3] == 128 for config in configs)
        assert all(config["l2_groupings"] == [1, 1] for config in configs)


def test_role_correction_runs_once_per_front_end() -> None:
    """Front end 1 corrects its projected tile; front end 2 corrects only its merged draft."""
    assert _FML.SINGLE_ROLE_AWARE_KNOBS is True
    assert _MULTI.SINGLE_ROLE_AWARE_KNOBS is False


def test_multi_stages_follow_the_emitted_tile_by_default() -> None:
    """Stage facts must describe the role-corrected block sizes the kernel actually uses."""
    assert _MULTI.ROLE_KEEP_STAGES is False


def _steps_env(
    steps: tuple[tuple[tuple[int, int, int, str], ...], ...],
) -> SimpleNamespace:
    """A stub env whose ``live_tile_steps`` are literal (rows, cols, itemsize, kind) tiles."""
    mk = lambda r, c, isz, kind: LiveTile(  # noqa: E731
        dim_block_ids=(None, None), static_dims=(r, c), itemsize=isz, kind=kind
    )
    return SimpleNamespace(
        config_spec=SimpleNamespace(
            kernel_matmul_fact=SimpleNamespace(
                live_tile_steps=tuple(tuple(starmap(mk, step)) for step in steps)
            ),
            block_sizes=_block_sizes_stub([]),
            _base_default_config=lambda: helion.Config(block_sizes=[]),
        ),
        block_sizes=[],
    )


def test_register_estimate_peaks_on_bytes_not_on_rank() -> None:
    """The register estimate must select its peak step by RESOLVED BYTES, not by rank profile.

    Rank profile is block-size-free, which is the right question for a reduction's working set
    (block sizes are unknown there) and the wrong one here (they are known). Selecting by rank
    picked a step holding several rank-2 loads over the step actually holding the accumulators:
    it under-counted a kernel spilling 540 registers at one warp (1.33x the one-warp file) while
    over-counting one spilling none (1.51x), so the ordering inverted and no threshold could
    separate them. This pins the fix: a step with FEWER tiles but more bytes must win."""
    many_small = tuple((8, 8, 4, "other") for _ in range(6))  # 6 tiles, 1536 B
    one_big = ((128, 128, 4, "other"),)  # 1 tile, 65536 B
    env = _steps_env((many_small, one_big))
    assert _FML._register_live_bytes(env, [], 8) == 128 * 128 * 4


def test_register_estimate_excludes_loads_and_unregisterable_values() -> None:
    """Loads are charged to the shared-memory ring, so charging them here would put the same
    bytes in two budgets. And a value larger than the largest register file a CTA can have is
    not register-resident at all -- it lives in HBM and the graph merely names it. Without that
    exclusion a varlen packed [T, C, D] buffer (measured 256 MiB) pinned every warp count to
    the maximum."""
    env = _steps_env((((64, 64, 4, "load"),),))
    assert _FML._register_live_bytes(env, [], 8) == 0
    huge = _FML.MAX_NUM_WARPS * 32 * _FML.REG_BYTES_PER_THREAD
    env = _steps_env((((huge, 4, 4, "other"),),))
    assert _FML._register_live_bytes(env, [], 8) == 0


def test_register_estimate_lets_tensor_memory_absorb_wide_accumulators() -> None:
    """A dot output is charged to the register file only when tensor memory cannot take it.
    tcgen05 MMA issues per warpgroup -- confirmed in PTX, num_warps 1 or 2 emits zero
    tcgen05.mma with tmem_size 0 -- and below TCGEN05_MIN_BM the dot never reaches that path at
    any warp count. So the charge is warp-count dependent, which is why the ladder must re-ask
    the estimate at every rung rather than compute it once."""
    wide = (((128, 128, 4, "dot_out"),),)
    assert _FML._register_live_bytes(_steps_env(wide), [], 1) == 128 * 128 * 4
    assert _FML._register_live_bytes(_steps_env(wide), [], 2) == 128 * 128 * 4
    assert _FML._register_live_bytes(_steps_env(wide), [], 4) == 0
    narrow = (
        ((32, 128, 4, "dot_out"),),
    )  # below TCGEN05_MIN_BM: never in tensor memory
    assert _FML._register_live_bytes(_steps_env(narrow), [], 8) == 32 * 128 * 4


def test_warps_climb_the_ladder_while_the_live_set_overshoots() -> None:
    """Section 3's register fix-up: estimate from the register-resident live set and the
    proposed warp count, then increase num_warps first. A fixed point, because raising the count
    both enlarges the file and can move the accumulators out of it entirely.

    The ladder is 1 -> 2 -> 4 -> 8, not a jump to a warpgroup: losing tcgen05 below one is not
    itself a penalty (at 16 KiB live, one warp on mma.sync beats four on tcgen05) and two warps
    is the hand-tuned answer in 11 of 18 chunk_cumsum_gc cells."""
    one_warp_file = 32 * _FML.REG_BYTES_PER_THREAD
    fits_one = (((64, 64, 4, "other"),),)  # 16 KiB
    assert _FML._register_live_bytes(_steps_env(fits_one), [], 1) <= one_warp_file
    assert _FML._warps_for_live_set(1, _steps_env(fits_one), []) == 1
    two = (((64, 64, 4, "other"), (64, 64, 4, "other")),)  # 32 KiB > 31.9 KiB
    assert _FML._warps_for_live_set(1, _steps_env(two), []) == 2
    four = tuple([tuple((64, 64, 4, "other") for _ in range(4))])
    assert _FML._warps_for_live_set(1, _steps_env(four), []) == 4
    # never lowers what the tile ramp asked for, and no opinion when nothing is recorded
    assert _FML._warps_for_live_set(8, _steps_env(four), []) == 8
    assert _FML._warps_for_live_set(1, _steps_env(()), []) == 1


def test_register_climb_stops_at_a_warpgroup_not_at_max_warps() -> None:
    """Registers are SOFT -- they spill, where tensor and shared memory raise OutOfResources at
    launch -- so relieving pressure past the point where it is relieved has a real cost. Each warp
    doubling halves resident CTAs, and for a grid over batch/head that buys less than the spill
    costs: measured on chunk_fwd_A_diag_anchored_varlen, nw=2 spills 38 B holding 4 CTAs/SM while
    nw=8 spills NOTHING holding 1, and nw=2 is 1.75x faster.

    The stop is a warpgroup because that is where _register_live_bytes hands the accumulators to
    tcgen05, so past it the estimate cannot justify another doubling. Over 75 swept cells scored
    against each cell's measured optimum, climbing to a fit and on to eight scores 0.8959 and
    stopping at a warpgroup scores 0.9591 -- above the hand-tuned key's 0.9363."""
    # A live set far past ANY register file must still stop at a warpgroup, not run to eight.
    huge = tuple([tuple((128, 128, 4, "other") for _ in range(8))])  # 512 KiB
    assert (
        _FML._warps_for_live_set(1, _steps_env(huge), [])
        == _FML.TCGEN05_WARPGROUP_WARPS
    )
    assert _FML.TCGEN05_WARPGROUP_WARPS < _FML.MAX_NUM_WARPS
    # Eight is still reachable -- it just has to come from the work term, which this rule keeps.
    assert _FML._warps_for_live_set(8, _steps_env(huge), []) == 8
    # The stop is a per-arch class attribute, not an arch test inside the ladder: hardware
    # selection already happened in is_eligible via HARDWARE_TARGETS. Pin that the sm100 carrier
    # ties it to the absorption boundary it is derived from, so the two cannot drift apart...
    assert _FML.REG_CLIMB_MAX_WARPS == _FML.TCGEN05_WARPGROUP_WARPS
    # ...and that the sm90 carrier is left where it was, holding that frozen emit still.
    assert _H100.REG_CLIMB_MAX_WARPS == _H100.MAX_NUM_WARPS
    assert _H100._warps_for_live_set(1, _steps_env(huge), []) == _H100.MAX_NUM_WARPS


def test_no_structural_warp_floor_survives() -> None:
    """The two-warp floor for multi-contraction kernels is DELETED, not disabled.

    It was a patch over the rank-selected estimate. Once the peak is chosen by bytes, the
    floor's own predicate became identical to the ladder's first rung, so it could never raise
    anything the ladder did not already raise. Adversarial review had independently shown the
    unconditional form cost +72% on a cell whose hand-tuned num_warps is 1 in all 10 of its
    cases, and that a 4-contraction kernel with one live output is fastest at one warp."""
    import inspect

    src = inspect.getsource(_MULTI)
    assert "MULTI_DOT_MIN_NUM_WARPS" not in src
    assert "_needs_warp_floor" not in src


def test_only_a_fixed_block_size_source_has_a_config_independent_extent() -> None:
    """The per-program extent an axis reports to the fact layer, and which sources have one.

    A ``FixedBlockSizeSource`` ignores the config, so its value IS the per-program extent and
    can be recorded once. Every other source reads a config knob -- ``block_sizes`` for a loop
    axis, ``reduction_loops`` for a reduction one -- so there is no config-independent extent
    and the answer must be ``None``, leaving the axis classified as tunable.

    This replaced a two-config PROBE (call ``from_config`` under block_sizes=16 and =128, and
    call the axis immovable if the answer did not move). The probe perturbed only
    ``block_sizes``, so a source movable solely via ``reduction_loops`` returned the same value
    both times and was misreported as immovable.

    The movable case uses a STUB source that returns a definite value, not a real
    ``LoopSpecBlockSizeSource``. That is deliberate: a real one needs a live
    ``CompileEnvironment`` and a spec with ``block_id_to_index``, and without them it returns
    ``None`` through the exception path -- which made an earlier version of this test pass with
    the ``isinstance`` guard DELETED. The stub is what makes the guard load-bearing here.
    """
    from helion._compiler.compile_environment import BlockSizeSource
    from helion._compiler.compile_environment import FixedBlockSizeSource
    from helion._compiler.device_ir_analysis import _immovable_extent

    spec = _matmul_config_spec()

    class _MovableSource(BlockSizeSource):
        """Not a FixedBlockSizeSource, and yields a value the guard must still refuse."""

        def from_config(self, config: object, block_size_info: object) -> int:
            return 64

    def env_with(source: object) -> SimpleNamespace:
        return SimpleNamespace(
            block_sizes=[SimpleNamespace(block_size_source=source, size=8192)],
            size_hint=lambda v: int(v),
        )

    # hl.tile(seqlen, block_size=64): the axis LENGTH is 8192, one program sees 64.
    assert _immovable_extent(env_with(FixedBlockSizeSource(64)), spec, 0) == 64
    # hl.grid: one scalar index per program, so the extent is 1 -- not the grid length.
    assert _immovable_extent(env_with(FixedBlockSizeSource(1)), spec, 0) == 1
    # Anything the config can move has NO config-independent extent, even though asking it
    # would have produced a perfectly plausible number.
    assert _MovableSource().from_config(None, None) == 64
    assert _immovable_extent(env_with(_MovableSource()), spec, 0) is None
    # Out-of-range ids are not an error, they simply have no extent.
    assert _immovable_extent(env_with(FixedBlockSizeSource(64)), spec, 7) is None
