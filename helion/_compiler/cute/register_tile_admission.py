"""Whether a reduction may nest as a per-thread register tile.

The CuTe persistent reduction strategy chooses its lane nesting in
``codegen_preamble``, before the tile body is generated (see
``DeviceGridState.nest_reduction_lane_outside_vector_tiles``).  Two decisions
therefore read the device IR instead of the generated statements:

* ``register_tile_body_admitted`` rejects bodies with a rolled loop, an
  operation whose lowering is neither a load, a store, a staging append nor a
  pure per-element computation, or a reduction over a tile block (owned by
  that tile's own lane, not by the reduction lane the schedule lowers).
  ``DeviceIR`` records the admitted reduction blocks (with a static extent) in
  ``ConfigSpec.cute_register_tile_reduction_blocks`` so config normalization,
  the seed heuristic and codegen agree.
* Bodies the IR admits may still contain a statement the two-pass schedule of
  ``register_tile_reductions.py`` cannot place (a guard inside the element
  loops, a per-element store, a register budget overrun).  The lowering raises
  ``RegisterTileUnsupported`` and ``generate_ast`` regenerates the kernel with
  ``CompileEnvironment.cute_register_tile_disabled`` set, which keeps the
  rolled lane nesting the strategy used before register tiles existed.
"""

from __future__ import annotations

import dataclasses
import types
from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

    from ..device_ir import GraphInfo


class RegisterTileUnsupported(exc.BackendUnsupported):
    """The owner lane's body is outside the register-tile shape.

    ``_split_one_lane_loop`` catches it to lift a lane-invariant guard outside
    the lane loop and retry each branch; ``generate_ast`` catches it to
    regenerate the kernel with the rolled lane nesting.
    """

    def __init__(self, reason: str) -> None:
        super().__init__("cute", f"register-tile lane reduction: {reason}")


@dataclasses.dataclass(frozen=True)
class RegisterTileOptions:
    """Autotuner knobs of the register-tile schedule
    (``register_tile_reductions.lower_register_tile_lane_loop``).

    * ``warp_fold`` (``cute_register_tile_warp_fold``): the shared-memory
      column reduce of a multi-warp group whose threads are laid out along the
      vector axis first (``pre > 1``) folds each warp's reduction coordinates
      with a strided shuffle butterfly before the shared-memory exchange
      (``_cute_grouped_reduce_shared_columns(..., fold='warp')``) instead of
      every thread summing ``group_span // pre`` staged partials per column.
    * ``store_first`` (``cute_register_tile_store_first``): the per-lane
      consume nest (the lane-varying stores) is emitted before the reductions'
      cross-thread combines when no store reads a reduction result, so a
      warp's stores do not wait behind the combines' CTA barriers.

    Both default off, and a knob the tile cannot honour (a one-warp group, a
    store that needs a reduction result) is ignored: the schedule then renders
    byte-identically to the one before the knobs existed.
    """

    warp_fold: bool = False
    store_first: bool = False


# ``hl`` operations whose lowering is neither a load, a store, a staging append
# nor a pure per-element computation, so the register-tile passes could not
# move it between the accumulate and consume nests; the rolled lane nesting
# stays.
_CUTE_REGISTER_TILE_REJECTED_MODULES = frozenset(
    {
        "helion.language.atomic_ops",
        "helion.language.barrier",
        "helion.language.debug_ops",
        "helion.language.device_print",
        "helion.language.distributed_ops",
        "helion.language.inline_asm_ops",
        "helion.language.inline_triton_ops",
        "helion.language.matmul_ops",
        "helion.language.quantized_ops",
        "helion.language.random_ops",
        "helion.language.reduce_ops",
        "helion.language.scan_ops",
        "helion.language.stack_tensor",
    }
)
# ``helion.language._tracing_ops`` targets that lower to a rolled loop inside
# the tile body (``test_control_flow_names_are_tracing_ops`` pins the names).
# ``_if`` is not listed: a lane-invariant guard around the reductions is
# lifted outside the lane loop by the lane splitter, while a guard that lands
# inside the element loops is rejected at split time and the kernel is
# regenerated with the rolled nesting.
_CUTE_REGISTER_TILE_TRACING_OPS = "helion.language._tracing_ops"
_CUTE_REGISTER_TILE_CONTROL_FLOW = frozenset(
    {"_for_loop", "_for_loop_step", "_while_loop"}
)


def register_tile_reduction_levels(graphs: Iterable[GraphInfo]) -> int:
    """How many accumulate nests the register-tile schedule runs for these
    graphs: one more than the longest chain of reductions where one's input
    depends on another's result (1 for independent reductions, 2 for a
    softmax's ``sum(exp(x - max))`` or a two-pass variance).  Every level
    after the first keeps more of the tile stashed, which the seed heuristic
    pays for with fewer lanes per thread."""
    from ..inductor_lowering import ReductionLowering

    levels = 1
    for graph_info in graphs:
        # Level of each node's value: the number of combines it waits for.
        level: dict[torch.fx.Node, int] = {}
        for node in graph_info.graph.nodes:
            value = max(
                (level.get(source, 0) for source in node.all_input_nodes), default=0
            )
            if isinstance(node.meta.get("lowering"), ReductionLowering):
                value += 1
                levels = max(levels, value)
            level[node] = value
    return levels


def register_tile_body_admitted(graphs: Iterable[GraphInfo]) -> bool:
    """Whether every device graph lowers to statements the register-tile passes
    can schedule: no rolled loop inside the tile, no operation with effects
    beyond a load, a store or a pure per-element computation, and every
    reduction over a reduction block.  A reduction fed by another reduction's
    result is admitted: the schedule runs one accumulate nest per dependency
    level (``register_tile_reductions``).

    A reduction over a tile block (the ``grad_weight`` partial of an autodiff
    rms_norm backward, ``sum(grad_out * normalized, 0)`` over the row tile) is
    not: under the register-tile nesting every tile block is a vectorized
    one-vector wrapper, so that reduction's marker is owned by the tile's own
    lane (``tile_strategy._flatten_vector_reduction_lane``), and the schedule
    lowers only the markers the synthetic reduction lane owns
    (``lower_register_tile_lane_loop``).  The rolled nesting stays, as it did
    before chained bodies were admitted.
    """
    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import ReductionLowering

    block_sizes = CompileEnvironment.current().block_sizes
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            lowering = node.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and not block_sizes[lowering.block_index].reduction
            ):
                return False
            target = node.target
            if node.op != "call_function" or not isinstance(target, types.FunctionType):
                continue
            if target.__module__ in _CUTE_REGISTER_TILE_REJECTED_MODULES:
                return False
            if (
                target.__module__ == _CUTE_REGISTER_TILE_TRACING_OPS
                and target.__name__ in _CUTE_REGISTER_TILE_CONTROL_FLOW
            ):
                return False
    return True
