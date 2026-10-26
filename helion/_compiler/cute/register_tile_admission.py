"""Whether a reduction may nest as a per-thread register tile.

The CuTe persistent reduction strategy chooses its lane nesting in
``codegen_preamble``, before the tile body is generated (see
``DeviceGridState.nest_reduction_lane_outside_vector_tiles``).  Two decisions
therefore read the device IR instead of the generated statements:

* ``register_tile_body_admitted`` rejects bodies with a rolled loop, an
  operation whose lowering is neither a load, a store, a staging append nor a
  pure per-element computation, or one reduction feeding another.
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


def register_tile_body_admitted(graphs: Iterable[GraphInfo]) -> bool:
    """Whether every device graph lowers to statements the register-tile passes
    can schedule: no rolled loop inside the tile, no operation with effects
    beyond a load, a store or a pure per-element computation, and no reduction
    fed by another reduction's result."""
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
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
        if _reduction_feeds_reduction(graph_info.graph):
            return False
    return True


def _reduction_feeds_reduction(graph: torch.fx.Graph) -> bool:
    """Whether a reduction's input depends on another reduction's result (the
    sum of ``exp(x - max)`` in a softmax, the squared deviations of a two-pass
    variance).  Such a chain needs a second accumulate pass the register-tile
    schedule does not have, so ``register_tile_reductions`` would reject it
    after codegen; the rolled nesting is kept up front instead."""
    from ..inductor_lowering import ReductionLowering

    # Nodes whose value carries a reduction result, in the graph's topological
    # order.
    tainted: set[torch.fx.Node] = set()
    for node in graph.nodes:
        reduction = isinstance(node.meta.get("lowering"), ReductionLowering)
        if any(source in tainted for source in node.all_input_nodes):
            if reduction:
                return True
            tainted.add(node)
        elif reduction:
            tainted.add(node)
    return False
