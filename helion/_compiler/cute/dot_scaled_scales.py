"""Read ``hl.dot_scaled``'s block scales where each operand element needs them.

``hl.dot_scaled`` takes its e8m0 scales as tiles (``x_scale[tile_m, :]``, one
byte per K group).  The CuTe fallback (``cute/matmul_ops.py``) dequantizes
every operand element by the scale of its own K group, read straight from the
scale tensor at the thread's row and K coordinates, so the scale tile's own
load is dead.  Left in the graph it would still lay its K-group axis out over
threads or synthetic lanes that no value of the contraction uses.  Pass the
scale tensor itself to ``dot_scaled`` instead and drop the load, when the scale
is a full K slice indexed by the same tile as its operand's row (or column).
The caller then recomputes ``DeviceIR.codegen_active_block_ids`` so the scale
tile's K-group axis, now unreferenced, gets no threads or lanes either.
"""

from __future__ import annotations

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language.matmul_ops import dot_scaled


def _tile_load(node: object) -> list[object] | None:
    """The subscript of a plain 2-d tile load (no extra mask or policy)."""
    if not (
        isinstance(node, torch.fx.Node)
        and node.target is memory_ops.load
        and len(node.args) == 4
        and node.args[2] is None
        and node.args[3] is None
        and isinstance(node.args[1], (list, tuple))
        and len(node.args[1]) == 2
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].target is _tracing_ops._host_tensor
    ):
        return None
    return list(node.args[1])


def expose_dot_scaled_scales(graph: torch.fx.Graph) -> bool:
    """Rewire each eligible ``dot_scaled`` scale argument to its scale tensor;
    return whether any was."""
    rewired = False
    for node in graph.find_nodes(op="call_function", target=dot_scaled):
        # (data argument, its row axis, its K axis); the scale follows it.
        for data_position, row_axis, k_axis in ((0, 0, 1), (3, 1, 0)):
            data_index = _tile_load(node.args[data_position])
            scale = node.args[data_position + 1]
            scale_index = _tile_load(scale)
            if (
                not isinstance(scale, torch.fx.Node)
                or data_index is None
                or scale_index is None
                or data_index[k_axis] != slice(None)
                or scale_index[1] != slice(None)
                or scale_index[0] is not data_index[row_axis]
                or len(scale.users) != 1
            ):
                continue
            node.update_arg(data_position + 1, scale.args[0])
            # One scale tile may serve both operands (``x[tile, :]`` and
            # ``y[:, tile]`` scaled by the same ``s[tile, :]``): it is dead
            # once the last of its arguments is rewired.
            if not scale.users:
                graph.erase_node(scale)
            rewired = True
    return rewired
