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
from ...language._tracing_ops import _mask_to
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..device_ir import GraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from .memory_ops import runtime_tensors_are_proven_disjoint
from .ordered_selection import is_ordered_selection
from .ordered_selection import selection_args

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch
    from .row_topk import RowTopKGraph


@dataclass(frozen=True)
class CuteTopKPlan:
    root_graph: torch.fx.Graph
    row_block_id: int
    n: int
    k: int
    largest: bool
    x: torch.Tensor
    values: torch.Tensor | None
    indices: torch.Tensor | None
    fragment_graph: RowTopKGraph | None = None
    softmax: bool = False
    lanes_per_row: int = 16
    rows_per_block: int = 8
    vector_width: int = 8
    output_vector_width: int = 1
    value_mode: str = "gather"
    key_dtype: str = "int32"
    rank_mode: str = "signed"
    selection_layout: str = "replicated"
    sort_network: str = "batcher"
    key_encoder: str = "dsl"
    defer_value_gathers: bool = False
    merge_schedule: str = "sequential"
    stable_ties: bool = False

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


def _match_softmax_epilogue(
    value: Node, selection: Node, dtype: torch.dtype
) -> tuple[Node, set[Node]] | None:
    """Recognize stable FP32 softmax of the selected values along their K axis."""
    consumed: set[Node] = set()

    def args(node: object, target: object, count: int) -> tuple[object, ...] | None:
        if (
            not isinstance(node, Node)
            or node.target is not target
            or node.kwargs
            or len(node.args) != count
        ):
            return None
        consumed.add(node)
        return node.args

    cast_args = args(value, torch.ops.prims.convert_element_type.default, 2)
    if cast_args is None or cast_args[1] != dtype:
        return None
    div_args = args(cast_args[0], torch.ops.aten.div.Tensor, 2)
    if div_args is None:
        return None
    exponential, denominator = div_args
    sum_args = args(denominator, torch.ops.aten.sum.dim_IntList, 3)
    if sum_args is None or sum_args[1] not in ([-1], [1]) or sum_args[2] is not True:
        return None
    sum_input = sum_args[0]
    if isinstance(sum_input, Node) and sum_input.target is _mask_to:
        masked = args(sum_input, _mask_to, 2)
        if masked is None or masked[1] != 0:
            return None
        sum_input = masked[0]
    if sum_input is not exponential:
        return None
    exp_args = args(exponential, torch.ops.aten.exp.default, 1)
    if exp_args is None:
        return None
    sub_args = args(exp_args[0], torch.ops.aten.sub.Tensor, 2)
    if sub_args is None:
        return None
    logits, maximum = sub_args
    max_args = args(maximum, torch.ops.aten.amax.default, 3)
    if max_args is None or max_args[1] not in ([-1], [1]) or max_args[2] is not True:
        return None
    max_input = max_args[0]
    if isinstance(max_input, Node) and max_input.target is _mask_to:
        masked = args(max_input, _mask_to, 2)
        if masked is None or masked[1] != float("-inf"):
            return None
        max_input = masked[0]
    if max_input is not logits:
        return None
    float_args = args(logits, torch.ops.prims.convert_element_type.default, 2)
    if float_args is None or float_args[1] != torch.float32:
        return None
    selected = float_args[0]
    selected_args = args(selected, operator.getitem, 2)
    if selected_args != (selection, 0):
        return None
    assert isinstance(selected, Node)
    return selected, consumed


def _match_direct_topk_root(
    graphs: Sequence[GraphInfo], *, noncanonical_block_ids: set[int]
) -> CuteTopKPlan | None:
    """Prove a full-row top-k and direct or stable-softmax value stores.

    Reject computation outside the recognized selection/softmax epilogue or
    additional effects. Neither inputs nor outputs may share storage: the
    output-value gather reads the input.
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
    selections = [n for n in nodes if is_ordered_selection(n)]
    stores = [n for n in nodes if n.target is store]
    if len(selections) != 1 or len(stores) != 2:
        return None
    env = CompileEnvironment.current()
    selection = selections[0]
    arguments = selection_args(selection)
    if arguments is None:
        return None
    source, k, largest, stable_ties = arguments
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
    if k is None and isinstance(n, int):
        k = n
    if (
        not isinstance(m, int)
        or m <= 0
        or not isinstance(n, int)
        or not isinstance(k, int)
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
    softmax = False
    for effect in stores:
        if len(effect.args) != 4 or effect.args[3] is not None or effect.kwargs:
            return None
        output_tensor = _tensor(effect.args[0])
        output_subscript, value = effect.args[1:3]
        if isinstance(value, Node):
            epilogue = _match_softmax_epilogue(value, selection, x.dtype)
            if epilogue is not None:
                value, epilogue_nodes = epilogue
                consumed.update(epilogue_nodes)
                softmax = True
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
    return CuteTopKPlan(
        root.graph,
        row_block_id,
        n,
        k,
        largest,
        x,
        values,
        indices,
        softmax=softmax,
        stable_ties=stable_ties,
    )


def match_topk_root(
    graphs: Sequence[GraphInfo], *, noncanonical_block_ids: set[int]
) -> CuteTopKPlan | None:
    from .row_topk import match_row_topk

    direct = _match_direct_topk_root(
        graphs, noncanonical_block_ids=noncanonical_block_ids
    )
    if direct is not None:
        return direct
    graph = match_row_topk(graphs, noncanonical_block_ids=noncanonical_block_ids)
    if graph is None:
        return None
    return CuteTopKPlan(
        graph.root_graph,
        graph.row_block_id,
        graph.n,
        graph.k,
        graph.largest,
        graph.x,
        None,
        None,
        fragment_graph=graph,
        stable_ties=graph.stable_ties,
    )


def topk_tensors_are_proven_disjoint(
    candidate: CuteTopKPlan,
    env: CompileEnvironment,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Share the final storage proof with compiler-owned seed selection."""
    if candidate.fragment_graph is not None:
        graph = candidate.fragment_graph
        pairs = [
            (output, tensor)
            for index, output in enumerate(graph.write_tensors)
            for tensor in (*graph.read_tensors, *graph.write_tensors[:index])
        ]
    else:
        assert candidate.values is not None and candidate.indices is not None
        tensors = (candidate.x, candidate.values, candidate.indices)
        pairs = [
            (left, right) for i, left in enumerate(tensors) for right in tensors[:i]
        ]
    for left, right in pairs:
        if left.untyped_storage()._cdata == right.untyped_storage()._cdata:
            return False
        # Fresh wrapper allocations cannot overlap another live storage.
        # Distinct runtime StorageImpls are insufficient: DLPack can wrap
        # the same memory twice. Use the cache-specialized span proof,
        # which becomes available after structural matching and binding.
        if (
            left.untyped_storage() not in env._symbolically_exact_layout_storages
            and right.untyped_storage() not in env._symbolically_exact_layout_storages
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
        # Sort preserves the original index order of equal values, including
        # opposite zero signs. Native floating keys deliberately distinguish
        # those signs, so only the integer-based encodings are eligible here.
        key_dtype=(
            "int32"
            if candidate.stable_ties
            and config.get("cute_topk_key_dtype") == "float32_native"
            else cast("str", config.get("cute_topk_key_dtype", "int32"))
        ),
        rank_mode=(
            "signed"
            if candidate.stable_ties
            else cast("str", config.get("cute_topk_rank_mode", "signed"))
        ),
        selection_layout=cast(
            "str", config.get("cute_topk_selection_layout", "replicated")
        ),
        sort_network=cast("str", config.get("cute_topk_sort_network", "batcher")),
        key_encoder=cast("str", config.get("cute_topk_key_encoder", "dsl")),
        defer_value_gathers=cast(
            "bool", config.get("cute_topk_defer_value_gathers", False)
        ),
        merge_schedule=cast(
            "str", config.get("cute_topk_merge_schedule", "sequential")
        ),
    )


def codegen_topk_root(cg: GenerateAST, plan: CuteTopKPlan) -> bool:
    from .topk_codegen import codegen_topk_root as emit

    return emit(cg, plan)
