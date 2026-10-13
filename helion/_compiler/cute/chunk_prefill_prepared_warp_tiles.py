"""Bind one engine's actual prepared interval to checked common warp tiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import chained_matmul as chain
from .chained_execution import ChainedExecution
from .prepared_graph_schedule import ContractionPartition
from .prepared_warp_tiles import CausalWarpPublication
from .prepared_warp_tiles import RuntimeTileAxis
from .prepared_warp_tiles import SharedLease
from .prepared_warp_tiles import SharedTileView
from .prepared_warp_tiles import WarpTile
from .prepared_warp_tiles import WarpTileInterval
from .prepared_warp_tiles import WarpTileSchedule

if TYPE_CHECKING:
    from .chunk_prefill_prepared_inverse import FastInverse
    from .chunk_prefill_prepared_pairwise import FastPairwise


def _fast_interval(pairwise: FastPairwise, inverse: FastInverse) -> WarpTileInterval:
    """Authority for the existing factor dispatch and its two actual cuts."""
    from .chunk_prefill_bt32 import common as cm

    pairwise.check()
    inverse.check()
    owner = pairwise.owner
    if inverse.pairwise is not pairwise:
        raise chain._UnsupportedChain("foreign fixed-work owner")
    role = cm.PIPELINE_PLAN.role("factor")
    qk = pairwise.qk
    inputs = (
        SharedTileView(qk.lhs, cm.QD, (cm.BT, cm.DK)),
        SharedTileView(qk.rhs, cm.KI, (cm.BT, cm.DK)),
    )
    # Gate-prefix reads have completed at the first full-team join. The QK
    # slab reuses those bytes only inside this interval. The other views are
    # still live, including the independent inverse and gamma publications.
    step = owner.region.step
    return WarpTileInterval(
        owner,
        role,
        ChainedExecution(
            128, thread="phase_tid", warp="local_warp", sync="cm.team_sync(team)"
        ),
        cm.STAGES,
        10,
        0,
        1,
        (cm.QD, cm.QD + cm.STAGE_BYTES),
        inputs,
        (
            SharedLease(
                step.gate_scan,
                cm.GATE_PREFIX,
                cm.GATE_PREFIX + cm.BT * cm.DK * 4,
                -1,
                0,
            ),
            SharedLease(pairwise.kk.lhs, cm.KD, cm.KD + cm.BT * cm.DK * 2, 0, 1),
            SharedLease(
                inverse.inverse16_bf16,
                cm.INV_WORK,
                cm.INV_WORK + cm.BT * cm.DK * 2,
                0,
                1,
            ),
            SharedLease(
                step.gate_scan, cm.PREFIX_LAST, cm.PREFIX_LAST + cm.DK * 4, 0, 1
            ),
            SharedLease(
                step.gate_scan,
                cm.RESTORE_FACTOR,
                cm.RESTORE_FACTOR + (cm.DK + 1) * 4,
                0,
                2,
            ),
            SharedLease(step.beta_load, cm.PREP_BETA, cm.PREP_BETA + cm.BT * 4, 0, 1),
            SharedLease(step.gate_scan, cm.GAMMA, cm.GAMMA + cm.DK * 4, 0, 1),
        ),
    )


def _fast_result(pairwise: FastPairwise) -> SharedTileView:
    """Actual output-update operand consumed after the second team cut."""
    from .chunk_prefill_bt32 import common as cm

    owner = pairwise.owner
    consumers = tuple(
        (port, index)
        for port in owner.ports
        for index, placement in enumerate(port.placements)
        if placement.issue.spec is owner.output_update
    )
    if len(consumers) != 1:
        raise chain._UnsupportedChain("ambiguous original output publication consumer")
    ((port, index),) = consumers
    origin = sum(
        placement.member[1] - placement.member[0]
        for placement in port.placements[:index]
    )
    if (
        port.descriptor
        != (
            cm.TMEM_RHS_U,
            cm.FINAL_TRANS,
            cm.TMEM_STATE,
            cm.DV,
            cm.DK + cm.BT,
            4096,
            1024,
            128,
            1,
            (2, 1, 16, 128),
        )
        or origin != cm.DK
        or port.ready != (cm.U2_INP_READY,)
        or port.complete != (cm.FINAL_READY, cm.SMEM_FREE)
    ):
        raise chain._UnsupportedChain("changed original output consumer port")
    offset = port.descriptor[1]
    assert isinstance(offset, int)
    return SharedTileView(
        pairwise.qk_value,
        offset,
        (cm.BT, cm.BT),
        (0, origin),
        transposed=True,
    )


@dataclass(frozen=True)
class FastWarpTileBinding:
    """Engine authority, not a claim that arbitrary descriptors prove the ABI."""

    pairwise: FastPairwise
    inverse: FastInverse

    def check(self, schedule: WarpTileSchedule) -> None:
        expected_interval = _fast_interval(self.pairwise, self.inverse)
        if schedule.interval != expected_interval:
            raise chain._UnsupportedChain("changed actual caller team/cut authority")
        if (
            schedule.publication.spec is not self.pairwise.qk
            or schedule.publication.publication is not self.pairwise.qk_value
            or self.pairwise.qk_value is not self.pairwise.owner.output_update.lhs
            or schedule.result != _fast_result(self.pairwise)
        ):
            raise chain._UnsupportedChain(
                "changed original consumer publication binding"
            )
        if schedule.fixed != (
            self.pairwise.kk,
            *(product.spec for product in self.inverse.products[:6]),
        ):
            raise chain._UnsupportedChain("changed actual fixed interval work")


def bind_fast_warp_tiles(
    pairwise: FastPairwise, inverse: FastInverse, strategy: str = "independent"
) -> WarpTileSchedule:
    """Static physical alternatives; no shape/name dispatch or tuning policy."""
    from .chunk_prefill_bt32 import common as cm

    interval = _fast_interval(pairwise, inverse)
    owner = pairwise.owner
    qk = pairwise.qk
    step = owner.region.step
    shape = (cm.BT, cm.BT)
    publication = CausalWarpPublication(
        qk, pairwise.qk_value, step.token_coordinate, step.token_coordinate, shape
    )
    axes = (RuntimeTileAxis(2, None, 16), RuntimeTileAxis(1, 2, 16))
    if strategy == "original":
        ownership = tuple((warp, 0, 0, 1) for warp in range(4))
    elif strategy == "independent":
        # The fixed packed inverse products bind the diagonal warps. This
        # existing adapter supplies a physical placement, not a semantic name
        # heuristic; the common checker still proves every tile and dependency.
        if any(product.local_warps != (0, 3) for product in inverse.products[:6]):
            raise chain._UnsupportedChain("changed fixed diagonal warp ownership")
        ownership = ((1, 0, 0, 1), (1, 1, 1, 0), (1, 2, 1, 1), (2, 0, 1, 1))
    else:
        raise chain._UnsupportedChain("unknown prepared warp ownership strategy")
    tiles = tuple(
        WarpTile(
            warp,
            ordinal,
            axes[row].value(warp),
            axes[col].value(warp),
            (row, col),
            (0, cm.DK),
            axes[row].value(warp) + 15 < axes[col].value(warp),
        )
        for warp, ordinal, row, col in ownership
    )
    result = _fast_result(pairwise)
    schedule = WarpTileSchedule(
        ContractionPartition(owner.graph, (qk,)),
        publication,
        interval,
        (pairwise.kk, *(product.spec for product in inverse.products[:6])),
        result,
        axes,
        tiles,
        pairwise.payload()[1],
        FastWarpTileBinding(pairwise, inverse),
    )
    schedule.check()
    return schedule
