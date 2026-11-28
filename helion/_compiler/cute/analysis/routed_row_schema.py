"""The FX operations that identify routed row fields, independent of bound storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....language import _tracing_ops as tr
from ....language import memory_ops
from ...inductor_lowering import ReductionLowering
from ..single_row_publication import row_pointwise_witness
from .match_common import node_tensor

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


def register_row_schema(env: CompileEnvironment, ir: DeviceIR) -> None:
    """Keep flat coordinates stable across shapes, strides and overlap facts.

    Registration selects field names only. Each concrete bound kernel proves
    its own eligible roots before seeding, normalizing, or emitting a schedule.
    """
    spec = env.config_spec
    cap = spec.target_device_capability
    for root, grid in zip(ir.root_ids, ir.grid_block_ids, strict=True):
        graph = ir.graphs[root].graph
        nodes = list(graph.nodes)
        stores = [n for n in nodes if n.target is memory_ops.store]
        sums = [
            n
            for n in nodes
            if isinstance(r := n.meta.get("lowering"), ReductionLowering)
            and r.reduction_type == "sum"
            and isinstance(v := n.meta.get("val"), torch.Tensor)
            and v.dtype == torch.float32
        ]
        if len(ir.root_ids) >= 2 and len(stores) == 1 and stores[0] is nodes[-2]:
            store = stores[0]
            axes = store.args[1]
            if (
                len(grid) == 1
                and len(sums) == 1
                and isinstance(axes, (list, tuple))
                and len(axes) == 2
                and axes[-1] == slice(None)
                and row_pointwise_witness(graph)
            ):
                spec.cute_row_publication_structure = True
            if sums and store.args[2] is sums[-1]:
                spec.cute_row_reduction_structure = True
                if (
                    cap is not None
                    and cap[0] == 10
                    and root == ir.root_ids[0]
                    and any(
                        n.target is memory_ops.load
                        and (t := node_tensor(n.args[0])) is not None
                        and t.dtype == torch.bfloat16
                        and t.ndim == 2
                        for n in nodes
                    )
                ):
                    spec.cute_row_prefetch_structure = True
        if (
            len(grid) != 3
            or cap is None
            or cap[0] < 8
            or not env.settings.fast_math
            or env.process_group_name is not None
            or not any(
                n.target is memory_ops.load
                and (t := node_tensor(n)) is not None
                and t.ndim == 0
                and t.dtype in (torch.int32, torch.int64)
                for n in nodes
            )
        ):
            continue
        for loop in graph.find_nodes(op="call_function", target=tr._for_loop):
            gid = loop.args[0]
            if type(gid) is not int or not 0 <= gid < len(ir.graphs):
                continue
            inner = ir.graphs[gid].graph
            if any(
                n.target is memory_ops.load
                and (t := node_tensor(n)) is not None
                and t.ndim == 2
                and t.dtype == torch.bfloat16
                for n in inner.nodes
            ) and any(n.target is torch.ops.aten.sum.dim_IntList for n in inner.nodes):
                spec.cute_indexed_row_structure = True
                if len(ir.root_ids) > 1 and not env.settings.persistent_reserved_sms:
                    spec.cute_staged_grid_structure = True
                if any(len(other) == 2 for other in ir.grid_block_ids):
                    spec.cute_row_epilogue_structure = True
                break
