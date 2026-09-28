"""Plan register-resident top-k for a complete row-wise selection region."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _get_symnode
from ...language._tracing_ops import _host_tensor
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..device_ir import GraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from .memory_ops import runtime_tensors_are_proven_disjoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch


@dataclass(frozen=True)
class CuteTopKPlan:
    root_graph: torch.fx.Graph
    row_block_id: int
    n: int
    k: int
    largest: bool
    x: torch.Tensor
    values: torch.Tensor
    indices: torch.Tensor
    lanes_per_row: int = 16
    rows_per_block: int = 8
    vector_width: int = 8
    output_vector_width: int = 1
    value_mode: str = "gather"
    key_dtype: str = "int32"
    rank_mode: str = "signed"
    selection_layout: str = "replicated"

    @property
    def threads(self) -> int:
        return max(32, self.lanes_per_row * self.rows_per_block)


def _tensor(node: object) -> torch.Tensor | None:
    if not isinstance(node, Node) or node.target is not _host_tensor:
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _row_block(node: object) -> int | None:
    if not isinstance(node, Node) or node.target is not _get_symnode:
        return None
    value = node.meta.get("val")
    if value is None:
        return None
    return CompileEnvironment.current().get_block_id(value)


def match_topk_root(
    graphs: Sequence[GraphInfo], *, noncanonical_block_ids: set[int]
) -> CuteTopKPlan | None:
    """Prove a full-row load, top-k, and direct stores with exact index narrowing.

    Reject additional computation or effects. In particular, neither inputs
    nor outputs may share storage: the output-value gather reads the input.
    Final planning also requires a cache-specialized runtime overlap proof.
    Reduction rolling can wrap the load in a graph-return/getitem pair; this
    is accepted only when that child graph returns exactly the direct load.
    """
    roots = {id(info.graph): info for info in graphs if isinstance(info, RootGraphInfo)}
    if len(roots) != 1:
        return None
    root = next(iter(roots.values()))
    nodes = list(root.graph.nodes)
    if nodes[-1].op != "output" or nodes[-1].args != (None,):
        return None
    selections = [n for n in nodes if n.target is torch.ops.aten.topk.default]
    stores = [n for n in nodes if n.target is store]
    if len(selections) != 1 or len(stores) != 2:
        return None
    env = CompileEnvironment.current()
    selection = selections[0]
    if not 2 <= len(selection.args) <= 5 or selection.kwargs:
        return None
    source, k = selection.args[:2]
    dim = selection.args[2] if len(selection.args) > 2 else -1
    largest = selection.args[3] if len(selection.args) > 3 else True
    if type(k) is not int or type(largest) is not bool or dim not in (-1, 1):
        return None
    consumed = {selection, *stores}
    reduction_end: object | None = None
    reduction_block: int | None = None
    if isinstance(source, Node) and source.target is operator.getitem:
        if len(source.args) != 2 or source.args[1] != 0 or source.kwargs:
            return None
        loop = source.args[0]
        if (
            not isinstance(loop, Node)
            or loop.target is not _for_loop
            or len(loop.args) != 4
            or loop.kwargs
            or loop.args[1] != [0]
            or loop.args[3] != []
        ):
            return None
        child = next((g for g in graphs if g.graph_id == loop.args[0]), None)
        if not isinstance(child, ReductionLoopGraphInfo) or len(child.block_ids) != 1:
            return None
        ends = loop.args[2]
        if not isinstance(ends, (list, tuple)) or len(ends) != 1:
            return None
        reduction_end = (
            ends[0].meta.get("val") if isinstance(ends[0], Node) else ends[0]
        )
        reduction_block = child.block_ids[0]
        child_nodes = list(child.graph.nodes)
        loads = [n for n in child_nodes if n.target is load]
        output = child_nodes[-1]
        if (
            len(loads) != 1
            or output.op != "output"
            or output.args != ([loads[0]],)
            or any(
                n.op != "output" and n.target not in (load, _host_tensor, _get_symnode)
                for n in child_nodes
            )
        ):
            return None
        consumed.update((source, loop))
        source = loads[0]
    if (
        not isinstance(source, Node)
        or source.target is not load
        or len(source.args) != 4
        or source.args[2:] != (None, None)
        or source.kwargs
    ):
        return None
    consumed.add(source)
    x = _tensor(source.args[0])
    subscript = source.args[1]
    if (
        x is None
        or x.ndim != 2
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not isinstance(subscript, (list, tuple))
        or len(subscript) != 2
        or subscript[1] != slice(None)
    ):
        return None
    row_block_id = _row_block(subscript[0])
    if row_block_id is None or env.block_sizes[row_block_id].reduction:
        return None
    if row_block_id in noncanonical_block_ids:
        return None
    m, n = x.shape
    if (
        not isinstance(m, int)
        or m <= 0
        or not isinstance(n, int)
        or not 0 < k <= n <= 32768
    ):
        return None
    row_size = env.block_sizes[row_block_id].size
    if not isinstance(row_size, (int, torch.SymInt)) or not env.known_equal(
        row_size, m
    ):
        return None
    if reduction_block is not None:
        reduction_size = env.block_sizes[reduction_block].size
        if (
            not isinstance(reduction_end, (int, torch.SymInt))
            or not isinstance(reduction_size, (int, torch.SymInt))
            or not env.known_equal(reduction_end, n)
            or not env.known_equal(reduction_size, n)
        ):
            return None
    if not env.known_equal(x.stride(-1), 1):
        return None
    outputs: dict[int, torch.Tensor] = {}
    for effect in stores:
        if len(effect.args) != 4 or effect.args[3] is not None or effect.kwargs:
            return None
        output_tensor = _tensor(effect.args[0])
        output_subscript, value = effect.args[1:3]
        if (
            isinstance(value, Node)
            and value.target is torch.ops.prims.convert_element_type.default
        ):
            if (
                output_tensor is None
                or output_tensor.dtype != torch.int32
                or len(value.args) != 2
                or value.args[1] != torch.int32
                or value.kwargs
                or not isinstance(value.args[0], Node)
                or value.args[0].target is not operator.getitem
                or value.args[0].args != (selection, 1)
            ):
                return None
            consumed.add(value)
            value = value.args[0]
        if (
            output_tensor is None
            or output_tensor.ndim != 2
            or not isinstance(output_subscript, (list, tuple))
            or len(output_subscript) != 2
            or output_subscript[1] != slice(None)
            or _row_block(output_subscript[0]) != row_block_id
            or not isinstance(value, Node)
            or value.target is not operator.getitem
            or len(value.args) != 2
            or value.args[0] is not selection
            or value.args[1] not in (0, 1)
            or value.kwargs
            or not env.known_equal(output_tensor.shape[0], m)
            or not env.known_equal(output_tensor.shape[1], k)
            or not env.known_equal(output_tensor.stride(0), k)
            or not env.known_equal(output_tensor.stride(1), 1)
        ):
            return None
        index = value.args[1]
        if index in outputs:
            return None
        outputs[index] = output_tensor
        consumed.add(value)
    if set(outputs) != {0, 1}:
        return None
    values, indices = outputs[0], outputs[1]
    if values.dtype != x.dtype or indices.dtype not in (torch.int32, torch.int64):
        return None
    # The direct store preserves topk's integer index, with only the store's
    # dtype conversion. The proven N <= 32768 bounds every emitted index to
    # [0, N), so narrowing an Int64 topk result to Int32 is exact.
    tensors = (x, values, indices)
    if any(
        tensors[i].untyped_storage()._cdata == tensors[j].untyped_storage()._cdata
        for i in range(3)
        for j in range(i)
    ):
        return None
    if any(
        node not in consumed
        and node.op != "output"
        and node.target not in (_host_tensor, _get_symnode)
        for node in nodes
    ):
        return None
    return CuteTopKPlan(root.graph, row_block_id, n, k, largest, x, values, indices)


def topk_tensors_are_proven_disjoint(
    candidate: CuteTopKPlan,
    env: CompileEnvironment,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Share the final storage proof with compiler-owned seed selection."""
    tensors = (candidate.x, candidate.values, candidate.indices)
    for i, left in enumerate(tensors):
        for right in tensors[:i]:
            # Fresh wrapper allocations cannot overlap another live storage.
            # Distinct runtime StorageImpls are insufficient: DLPack can wrap
            # the same memory twice. Use the cache-specialized span proof,
            # which becomes available after structural matching and binding.
            if (
                left.untyped_storage() not in env._symbolically_exact_layout_storages
                and right.untyped_storage()
                not in env._symbolically_exact_layout_storages
                and not runtime_tensors_are_proven_disjoint(
                    env, left, right, allow_unbound=allow_unbound
                )
            ):
                return False
    return True


def plan_topk_root(
    graphs: Sequence[GraphInfo], tile_strategy: TileStrategyDispatch
) -> CuteTopKPlan | None:
    if len(HostFunction.current().device_ir.root_ids) != 1:
        return None
    candidate = match_topk_root(
        graphs,
        noncanonical_block_ids=HostFunction.current().device_ir.noncanonical_task_origin_block_ids,
    )
    if candidate is None or not topk_tensors_are_proven_disjoint(
        candidate, CompileEnvironment.current()
    ):
        return None
    config = tile_strategy.strategies[0].fn.config
    return replace(
        candidate,
        lanes_per_row=cast("int", config.get("cute_topk_lanes_per_row", 16)),
        rows_per_block=cast("int", config.get("cute_topk_rows_per_block", 8)),
        vector_width=cast("int", config.get("cute_topk_vector_width", 8)),
        output_vector_width=cast("int", config.get("cute_topk_output_vector_width", 1)),
        value_mode=cast("str", config.get("cute_topk_value_mode", "gather")),
        key_dtype=cast("str", config.get("cute_topk_key_dtype", "int32")),
        rank_mode=cast("str", config.get("cute_topk_rank_mode", "signed")),
        selection_layout=cast(
            "str", config.get("cute_topk_selection_layout", "replicated")
        ),
    )


def codegen_topk_root(cg: GenerateAST, plan: CuteTopKPlan) -> bool:
    from .topk_codegen import codegen_topk_root as emit

    return emit(cg, plan)
