"""The lane loop of one tile block, as the per-thread strategies and the CuTe
memory emitters see it.

Role: core state of the per-thread tile and reduction strategies
(``tile_strategy.py``, ``reduction_strategy.py``), which the CuTe and Metal
backends share; the vector-packet fields are filled by the CuTe emitters.

A per-thread tile strategy (``PerThreadNDTileStrategy``,
``PerThreadFlattenedTileStrategy``) or a reduction strategy with a vectorized
lane (``LoopedReductionStrategy``, ``PersistentReductionStrategy``) walks the
elements a thread owns in a lane loop, and splits that loop into an outer loop
and a constexpr V-loop when the autotuner picks a vector width.  The emitters
that splice vector transactions around the V-loop (``emitters/lane_packets.py``,
``atomic_ops.py``, ``signed_bitfield.py``) need the loop's artifacts: the lane
body the hoists and flushes are inserted into, the V-loop node that positions
them, the per-thread base index and the V-loop variable the extracts read, and
the packets already placed.  ``LaneLoopContext`` carries them; the strategy
that opens the lane loop fills it in and registers it under its block id in
``DeviceFunction.lane_contexts`` (``DeviceFunction.register_lane``; one strategy
owns a block id, so the map is unambiguous), and an emitter reaches it by block
id through ``lane_context``.

Facts decided before any lane loop exists stay on the strategies: a
reduction's vector width and lane extent (``_cute_reduction_vec_width``,
``_cute_reduction_lane_extent``, read by the execution planners and the
memory-access resolver), the lane layout and the packing of a tile axis.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ast

    import torch

    from .cute.emitters.lane_packets import CuteTileVecStoreSite
    from .inductor_lowering import CodegenState

# Orders the blocks of a strategy by the registration of their first vector
# store: ``CuteBackend.demote_grid_tile_vec_stores`` decides the blocks in
# that order, and a decision can change what the next one observes.
_VEC_STORE_SERIAL = itertools.count()


@dataclasses.dataclass(eq=False)
class LaneLoopContext:
    """One block's lane loop and the vector packets placed around its V-loop.

    ``vec_width`` is the tile lane's vector width (1 = scalar; a reduction's
    is the strategy's ``_cute_reduction_vec_width``).  ``vec_lane_var`` is the
    constexpr V-loop variable, ``base_index_var`` the per-thread base of the
    V-wide chunk, ``lane_body`` the statements inside the outer lane loop
    (hoists go before ``vloop``, flushes after it) -- all None until the
    strategy builds the V-loop.  ``lane_axis_pos`` is the ``index_exprs``
    position of the vectorized axis a load site recorded (the last position
    when unset); ``flat_multi`` marks a flattened multi-block tile whose base
    is a flat offset; ``cluster_n`` the thread-block-cluster split of the
    axis; ``chunk_full_var`` a looped reduction's whole-chunk predicate.
    ``vec_loads`` caches the hoisted packets by ``(tensor, base pointer,
    guard)`` and ``vec_stores`` records the collected stores in source order.
    """

    block_id: int
    vec_width: int = 1
    vec_lane_var: str | None = None
    base_index_var: str | None = None
    lane_body: list[ast.AST] | None = None
    vloop: ast.For | None = None
    lane_axis_pos: int | None = None
    flat_multi: bool = False
    cluster_n: int = 1
    chunk_full_var: str | None = None
    vec_loads: dict[tuple[str, str, str | None], tuple[str, torch.dtype]] = (
        dataclasses.field(default_factory=dict)
    )
    vec_stores: list[CuteTileVecStoreSite] = dataclasses.field(default_factory=list)
    first_vec_store_serial: int | None = None

    def vloop_position(self) -> int:
        """Position of the V-loop in ``lane_body``: where a hoist is inserted.

        The V-loop is not guaranteed to be the last entry once store flushes
        follow it; the last position is the fallback when it is not found.
        """
        lane_body = self.lane_body
        assert lane_body is not None
        for index, statement in enumerate(lane_body):
            if statement is self.vloop:
                return index
        return len(lane_body) - 1

    def lane_axis(self, index_exprs: list[str]) -> int:
        """``index_exprs`` position of the vectorized (stride-1) axis."""
        if self.lane_axis_pos is not None:
            return self.lane_axis_pos
        return len(index_exprs) - 1

    def add_vec_store(self, site: CuteTileVecStoreSite) -> None:
        if self.first_vec_store_serial is None:
            self.first_vec_store_serial = next(_VEC_STORE_SERIAL)
        self.vec_stores.append(site)


def lane_context(state: CodegenState, block_id: int) -> LaneLoopContext | None:
    """The lane-loop context registered for ``block_id``; None when no
    per-thread tile or vectorizable reduction strategy owns the block."""
    return state.device_function.lane_contexts.get(block_id)
