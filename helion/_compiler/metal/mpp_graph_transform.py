"""DeviceIR rewrite for Metal MPP matmul lowering.

The pass replaces an eligible K-loop matmul result and optional post-K epilogue
with an ``MPPGraphInfo`` region.  That synthetic graph owns cooperative MPP
setup, K iteration, epilogue, and store; the surrounding root graph stays on the
normal scalar Metal lowering path.

Recognition is conservative: the K-loop must return a supported 2D matmul
accumulator, the root graph must have a single owner chain from
``getitem -> _phi`` to one store, and A/B/output views must match the canonical
matmul tile.  Non-fusible same-shape scalar postprocessing is handled by first
materializing the MPP result, then reloading it in scalar code.
"""

from __future__ import annotations

import contextlib
import dataclasses
import operator

import torch
from torch.fx import Graph
from torch.fx.node import Node
from torch.fx.node import map_arg

from ..cute.cute_mma import _trace_to_load_tensor
from ..device_ir import DeviceIR
from ..device_ir import ForLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..inductor_lowering import APIFuncLowering
from .mpp_graph_codegen import MPPGraphInfo
from .mpp_graph_codegen import _mpp_graph
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops


@dataclasses.dataclass(frozen=True)
class _MPPLoadView:
    """Tensor plus 2D indices used by an MPP operand load."""

    tensor: torch.Tensor
    indices: tuple[object, object]


@dataclasses.dataclass(frozen=True)
class _MPPStoreView:
    """Tensor plus 2D indices used by the MPP output store."""

    tensor: torch.Tensor
    indices: tuple[object, object]


@dataclasses.dataclass(frozen=True)
class _Candidate:
    """Matched root/K-loop slice that can become one MPPGraphInfo.

    ``mpp_output_node`` is the value produced by cooperative MPP lowering.
    ``reload_value_node`` is set only for materialized scalar continuation,
    where root scalar code reloads the MPP result from the final store target.
    """

    for_loop_node: Node
    getitem_node: Node
    phi_node: Node
    k_loop_info: ForLoopGraphInfo
    mma_node: Node
    lhs_view: _MPPLoadView
    rhs_view: _MPPLoadView
    bias_view: _MPPLoadView | None
    acc_dtype: torch.dtype
    store_view: _MPPStoreView
    mpp_output_node: Node
    reload_value_node: Node | None
    epilogue_nodes: tuple[Node, ...]
    store_node: Node


def rewrite_mpp_graphs(device_ir: DeviceIR) -> None:
    """Rewrite eligible root/K-loop matmul pairs into ``MPPGraphInfo``.

    The pass scans only root graphs.  For each root ``_for_loop`` call, it
    checks whether one returned value is a completed K reduction whose loop
    body returns a supported matmul accumulator.  The root use must be
    single-owner: ``getitem -> _phi -> optional epilogue -> store``.  This
    ownership check is what lets the rewrite remove the original K-loop result
    from the root graph and replace it with a synthetic ``_mpp_graph`` call
    without duplicating stores or losing side effects.

    The produced ``MPPGraphInfo`` owns code generation for the cooperative
    region.  The root graph remains responsible for any scalar continuation
    that the transform deliberately leaves outside the cooperative region.
    Unsupported or ambiguous shapes return no candidate and leave the graph
    unchanged.
    """
    for graph_info in list(device_ir.graphs):
        if not isinstance(graph_info, RootGraphInfo):
            continue
        for candidate in _find_root_candidates(device_ir, graph_info):
            mpp_graph_id = _append_mpp_graph(device_ir, candidate)
            _rewrite_root_for_mpp_graph(graph_info, candidate, mpp_graph_id)


def _find_root_candidates(
    device_ir: DeviceIR, root_info: RootGraphInfo
) -> list[_Candidate]:
    """Find root/K-loop slices that can be replaced by an MPPGraph."""
    candidates: list[_Candidate] = []
    for node in root_info.graph.nodes:
        if not _is_for_loop_call(node):
            continue
        k_loop_info = _for_loop_graph_info(device_ir, node)
        if k_loop_info is None:
            continue
        for getitem_node in _getitem_users(node):
            mma_node = _mma_output_for_getitem(k_loop_info, getitem_node)
            if mma_node is None:
                continue
            phi_node = _single_phi_user(getitem_node)
            if phi_node is None:
                continue
            candidate = _classify_phi_users(
                for_loop_node=node,
                getitem_node=getitem_node,
                phi_node=phi_node,
                k_loop_info=k_loop_info,
                mma_node=mma_node,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _is_for_loop_call(node: Node) -> bool:
    """Return whether *node* calls a DeviceIR for-loop graph."""
    return (
        node.op == "call_function"
        and _tracing_ops.is_for_loop_target(node.target)
        and bool(node.args)
        and isinstance(node.args[0], int)
    )


def _for_loop_graph_info(device_ir: DeviceIR, node: Node) -> ForLoopGraphInfo | None:
    """Return the ForLoopGraphInfo referenced by a for-loop call node."""
    graph_id = node.args[0]
    assert isinstance(graph_id, int)
    if not (0 <= graph_id < len(device_ir.graphs)):
        return None
    graph_info = device_ir.graphs[graph_id]
    if not isinstance(graph_info, ForLoopGraphInfo):
        return None
    return graph_info


def _getitem_users(for_loop_node: Node) -> list[Node]:
    """Return direct ``operator.getitem`` users of a for-loop result."""
    return [
        user
        for user in for_loop_node.users
        if user.op == "call_function" and user.target is operator.getitem
    ]


def _mma_output_for_getitem(
    k_loop_info: ForLoopGraphInfo, getitem_node: Node
) -> Node | None:
    """Return the K-loop output selected by *getitem_node* if it is matmul."""
    if len(getitem_node.args) < 2 or not isinstance(getitem_node.args[1], int):
        return None
    output_idx = getitem_node.args[1]
    output_nodes = list(k_loop_info.graph.find_nodes(op="output"))
    if len(output_nodes) != 1:
        return None
    output_value = output_nodes[0].args[0]
    if not isinstance(output_value, (tuple, list)):
        return None
    if not (0 <= output_idx < len(output_value)):
        return None
    candidate = output_value[output_idx]
    if not isinstance(candidate, Node) or not _is_matmul_node(candidate):
        return None
    return candidate


def _is_matmul_node(node: Node) -> bool:
    """Return whether *node* is a matmul op supported by MPPGraph matching."""
    return node.op == "call_function" and node.target in {
        hl.dot,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    }


def _is_phi_node(node: Node, value: Node) -> bool:
    """Return whether *node* is a phi that selects *value*."""
    return (
        node.op == "call_function"
        and node.target is _tracing_ops._phi
        and len(node.args) >= 2
        and node.args[1] is value
    )


def _single_phi_user(getitem_node: Node) -> Node | None:
    """Return the unique root phi that consumes a K-loop getitem result."""
    phi_users = [
        user for user in getitem_node.users if _is_phi_node(user, getitem_node)
    ]
    if len(phi_users) != 1:
        return None
    return phi_users[0]


def _classify_phi_users(
    *,
    for_loop_node: Node,
    getitem_node: Node,
    phi_node: Node,
    k_loop_info: ForLoopGraphInfo,
    mma_node: Node,
) -> _Candidate | None:
    """Classify the post-K path as fused epilogue or materialized scalar use."""
    operand_info = _mpp_operand_views(mma_node)
    if operand_info is None:
        return None
    lhs_view, rhs_view, bias_view, acc_dtype = operand_info
    epilogue_nodes: list[Node] = []
    cur = phi_node
    visited: set[Node] = set()
    # Walk the single-owner post-K value chain:
    # phi -> optional cooperative epilogue nodes -> store.  The first
    # non-fusible op switches to the materialized scalar continuation path.
    while True:
        if cur in visited or len(cur.users) != 1:
            return None
        visited.add(cur)
        (user,) = cur.users
        if _is_store_of_value(user, cur):
            store_view = _validated_mpp_store_view(
                user,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                mpp_output_node=cur,
                acc_dtype=acc_dtype,
            )
            if store_view is None:
                return None
            return _Candidate(
                for_loop_node=for_loop_node,
                getitem_node=getitem_node,
                phi_node=phi_node,
                k_loop_info=k_loop_info,
                mma_node=mma_node,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                bias_view=bias_view,
                acc_dtype=acc_dtype,
                store_view=store_view,
                mpp_output_node=cur,
                reload_value_node=None,
                epilogue_nodes=tuple(epilogue_nodes),
                store_node=user,
            )
        if not _is_coop_fusible_epilogue_node(user, cur):
            materialized = _find_materialized_scalar_store(
                start_node=user,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                mpp_output_node=cur,
                acc_dtype=acc_dtype,
            )
            if materialized is None:
                return None
            store_node, store_view = materialized
            return _Candidate(
                for_loop_node=for_loop_node,
                getitem_node=getitem_node,
                phi_node=phi_node,
                k_loop_info=k_loop_info,
                mma_node=mma_node,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                bias_view=bias_view,
                acc_dtype=acc_dtype,
                store_view=store_view,
                mpp_output_node=cur,
                reload_value_node=cur,
                epilogue_nodes=tuple(epilogue_nodes),
                store_node=store_node,
            )
        epilogue_nodes.append(user)
        cur = user


def _is_store_of_value(node: Node, value: Node) -> bool:
    """Return whether *node* stores exactly *value*."""
    return (
        node.op == "call_function"
        and node.target is memory_ops.store
        and len(node.args) >= 3
        and node.args[2] is value
    )


def _find_materialized_scalar_store(
    *,
    start_node: Node,
    lhs_view: _MPPLoadView,
    rhs_view: _MPPLoadView,
    mpp_output_node: Node,
    acc_dtype: torch.dtype,
) -> tuple[Node, _MPPStoreView] | None:
    """Find a scalar continuation store after materializing an MPP result."""
    cur = start_node
    visited: set[Node] = set()
    while True:
        if cur in visited:
            return None
        visited.add(cur)
        if (
            cur.op == "call_function"
            and cur.target is memory_ops.store
            and len(cur.args) >= 3
        ):
            store_view = _validated_mpp_store_view(
                cur,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                mpp_output_node=mpp_output_node,
                acc_dtype=acc_dtype,
            )
            if store_view is None:
                return None
            return cur, store_view
        if len(cur.users) != 1:
            return None
        (cur,) = cur.users


def _mpp_operand_views(
    mma_node: Node,
) -> tuple[_MPPLoadView, _MPPLoadView, _MPPLoadView | None, torch.dtype] | None:
    """Extract validated A/B/bias views and accumulator dtype from matmul."""
    lhs_idx = 1 if mma_node.target is torch.ops.aten.addmm.default else 0
    rhs_idx = 2 if mma_node.target is torch.ops.aten.addmm.default else 1
    acc_idx = 0 if mma_node.target is torch.ops.aten.addmm.default else None
    if len(mma_node.args) <= max(lhs_idx, rhs_idx):
        return None
    lhs_node = mma_node.args[lhs_idx]
    rhs_node = mma_node.args[rhs_idx]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return None
    lhs_view = _trace_to_load_view(lhs_node)
    rhs_view = _trace_to_load_view(rhs_node)
    if lhs_view is None or rhs_view is None:
        return None
    if lhs_view.tensor.ndim != 2 or rhs_view.tensor.ndim != 2:
        return None
    if lhs_view.tensor.dtype != rhs_view.tensor.dtype:
        return None
    acc_dtype = _tensor_dtype_from_meta(mma_node)
    if acc_dtype is None:
        return None
    bias_view = None
    if acc_idx is not None and len(mma_node.args) > acc_idx:
        acc_node = mma_node.args[acc_idx]
        if isinstance(acc_node, Node):
            bias_view = _trace_to_load_view(acc_node)
    return lhs_view, rhs_view, bias_view, acc_dtype


def _trace_to_load_view(node: Node) -> _MPPLoadView | None:
    """Trace a value back to the 2D tensor load that produced it."""
    load_info = _trace_to_load_tensor(node)
    if load_info is None:
        return None
    load_node, _, tensor = load_info
    if len(load_node.args) < 2:
        return None
    indices = _as_2d_indices(load_node.args[1])
    if indices is None:
        return None
    return _MPPLoadView(tensor=tensor, indices=indices)


def _tensor_dtype_from_meta(node: Node) -> torch.dtype | None:
    """Return the tensor dtype recorded in FX metadata, if present."""
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return value.dtype
    return None


def _as_2d_indices(indices: object) -> tuple[object, object] | None:
    """Normalize an index object to two dimensions."""
    if not isinstance(indices, (list, tuple)) or len(indices) != 2:
        return None
    return indices[0], indices[1]


def _validated_mpp_store_view(
    store_node: Node,
    *,
    lhs_view: _MPPLoadView,
    rhs_view: _MPPLoadView,
    mpp_output_node: Node,
    acc_dtype: torch.dtype,
) -> _MPPStoreView | None:
    """Return the store view if it can receive the materialized MPP result."""
    store_view = _store_output_view(store_node)
    if store_view is None:
        return None
    if not _can_store_mpp_output(
        lhs_view, rhs_view, mpp_output_node, store_view, acc_dtype
    ):
        return None
    return store_view


def _is_canonical_mpp_view(
    lhs_view: _MPPLoadView,
    rhs_view: _MPPLoadView,
    store_view: _MPPStoreView,
) -> bool:
    """Return whether A/B/output views match canonical matmul layout."""
    if not (
        _same_dim(lhs_view.tensor.shape[0], store_view.tensor.shape[0])
        and _same_dim(rhs_view.tensor.shape[1], store_view.tensor.shape[1])
        and _same_dim(lhs_view.tensor.shape[1], rhs_view.tensor.shape[0])
    ):
        return False
    lhs_m, lhs_k = lhs_view.indices
    rhs_k, rhs_n = rhs_view.indices
    out_m, out_n = store_view.indices
    # Shape checks prove the tensor extents match.  Integer index checks prove
    # the matched tile axes are wired as A[M,K], B[K,N], C[M,N].
    if not all(
        isinstance(index, int) for index in (lhs_m, lhs_k, rhs_k, rhs_n, out_m, out_n)
    ):
        return True
    return lhs_m == out_m and rhs_n == out_n and lhs_k == rhs_k


def _can_store_mpp_output(
    lhs_view: _MPPLoadView,
    rhs_view: _MPPLoadView,
    mpp_output_node: Node,
    store_view: _MPPStoreView,
    acc_dtype: torch.dtype,
) -> bool:
    """Validate that an MPP result can be stored through *store_view*."""
    expected_dtype = store_view.tensor.dtype
    value = mpp_output_node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return acc_dtype == expected_dtype and _is_canonical_mpp_view(
            lhs_view, rhs_view, store_view
        )
    # Dtype changes must be explicit epilogue cast nodes before the final store;
    # MPP cooperative store does not perform an implicit output cast here.
    if value.dtype != expected_dtype or value.ndim != store_view.tensor.ndim:
        return False
    return _is_canonical_mpp_view(lhs_view, rhs_view, store_view)


def _same_dim(lhs: int | torch.SymInt, rhs: int | torch.SymInt) -> bool:
    """Best-effort dimension equality for symbolic and concrete dims."""
    try:
        return bool(lhs == rhs)
    except TypeError:
        return False


def _is_coop_fusible_epilogue_node(node: Node, input_node: Node) -> bool:
    """Return whether *node* is a single-input cooperative epilogue op."""
    if node.op != "call_function" or not _is_coop_fusible_epilogue_target(node.target):
        return False
    return all(user_input is input_node for user_input in node.all_input_nodes)


def _is_coop_fusible_epilogue_target(target: object) -> bool:
    """Return whether *target* is in the current cooperative epilogue allowlist."""
    return target in {
        torch.ops.aten.relu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.neg.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.prims.convert_element_type.default,
    }


def _append_mpp_graph(device_ir: DeviceIR, candidate: _Candidate) -> int:
    """Append an MPPGraphInfo for *candidate* and return its graph id."""
    # The K-loop itself is not copied.  MPPGraphInfo records its bounds and
    # operands, then emits the MPP setup and K-step loop directly during codegen.
    new_graph = Graph()
    acc = new_graph.placeholder(candidate.mma_node.name)
    acc.meta.update(candidate.mma_node.meta)
    env: dict[Node, Node] = {candidate.phi_node: acc}

    # Only the post-K cooperative epilogue remains as FX.  The original phi is
    # replaced by a placeholder representing the completed MPP accumulator.
    for node in candidate.epilogue_nodes:
        _copy_node_recursive(node, new_graph, env)
    new_graph.output((env[candidate.mpp_output_node],))

    begin = candidate.for_loop_node.args[1]
    end = candidate.for_loop_node.args[2]
    assert isinstance(begin, list)
    assert isinstance(end, list)

    graph_id = len(device_ir.graphs)
    device_ir.graphs.append(
        MPPGraphInfo(
            graph_id=graph_id,
            graph=new_graph,
            node_args=[*candidate.k_loop_info.node_args],
            k_block_id=candidate.k_loop_info.block_ids[0],
            begin=[*begin],
            end=[*end],
            result_name=candidate.mma_node.name,
            lhs_tensor=candidate.lhs_view.tensor,
            rhs_tensor=candidate.rhs_view.tensor,
            bias_tensor=(
                candidate.bias_view.tensor if candidate.bias_view is not None else None
            ),
            acc_dtype=candidate.acc_dtype,
            out_tensor=candidate.store_view.tensor,
            out_dtype=candidate.store_view.tensor.dtype,
            needs_store_barrier=candidate.reload_value_node is not None,
        )
    )
    return graph_id


def _store_output_view(store_node: Node | None) -> _MPPStoreView | None:
    """Extract the output tensor and 2D indices from a store node."""
    if store_node is None or len(store_node.args) < 2:
        return None
    out_arg = store_node.args[0]
    if not isinstance(out_arg, Node):
        return None
    fake = out_arg.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    if fake.ndim != 2:
        return None
    indices = _as_2d_indices(store_node.args[1])
    if indices is None:
        return None
    return _MPPStoreView(
        tensor=fake,
        indices=indices,
    )


def _copy_node_recursive(node: Node, new_graph: Graph, env: dict[Node, Node]) -> Node:
    """Copy *node* and its uncopied inputs into *new_graph*."""
    if node in env:
        return env[node]
    for input_node in node.all_input_nodes:
        if input_node not in env:
            _copy_node_recursive(input_node, new_graph, env)
    new_node = new_graph.node_copy(node, lambda n: env[n])
    env[node] = new_node
    return new_node


def _rewrite_root_for_mpp_graph(
    root_info: RootGraphInfo,
    candidate: _Candidate,
    mpp_graph_id: int,
) -> None:
    """Replace the consumed root slice with a synthetic ``_mpp_graph`` call."""
    consumed = {
        candidate.for_loop_node,
        candidate.getitem_node,
        candidate.phi_node,
        *candidate.epilogue_nodes,
    }
    if candidate.reload_value_node is None:
        consumed.add(candidate.store_node)
    old_graph = root_info.graph
    new_graph = Graph()
    env: dict[Node, Node] = {}
    inserted = False

    def load_arg(n: Node) -> Node:
        return env[n]

    def ensure_arg(arg: Node) -> Node:
        if arg in env:
            return env[arg]
        for input_node in arg.all_input_nodes:
            if input_node not in env:
                ensure_arg(input_node)
        new_node = new_graph.node_copy(arg, load_arg)
        env[arg] = new_node
        return new_node

    for node in old_graph.nodes:
        if node is candidate.for_loop_node:
            mpp_node = new_graph.call_function(
                _mpp_graph,
                args=(
                    mpp_graph_id,
                    node.args[1],
                    node.args[2],
                    [],
                ),
            )
            _prepare_inserted_mpp_node(mpp_node)
            env[node] = mpp_node
            if candidate.reload_value_node is not None:
                load_node = new_graph.call_function(
                    memory_ops.load,
                    args=(
                        map_arg(candidate.store_node.args[0], ensure_arg),
                        map_arg(candidate.store_node.args[1], ensure_arg),
                        None,
                        None,
                    ),
                )
                _prepare_inserted_reload_node(
                    load_node,
                    candidate.reload_value_node,
                    fallback_node=candidate.mma_node,
                )
                env[candidate.reload_value_node] = load_node
            inserted = True
            continue
        if node in consumed:
            continue
        if _is_dead_after_mpp_rewrite(node, consumed):
            continue
        if node.op == "output":
            new_graph.output(map_arg(node.args[0], load_arg))
            continue
        if node in env:
            continue
        new_node = new_graph.node_copy(node, load_arg)
        env[node] = new_node

    assert inserted, "expected to insert _mpp_graph marker"
    root_info.graph = new_graph


def _is_dead_after_mpp_rewrite(node: Node, consumed: set[Node]) -> bool:
    """Return whether a pure node is only used by nodes consumed by rewrite."""
    return (
        node.op not in {"placeholder", "output"}
        and bool(node.users)
        and not node.is_impure()
        and all(user in consumed for user in node.users)
    )


def _prepare_inserted_mpp_node(node: Node) -> None:
    """Attach lowering metadata to an inserted ``_mpp_graph`` node."""
    node.meta["lowering"] = APIFuncLowering(_mpp_graph)
    node.meta["location"] = contextlib.nullcontext()
    node.meta["val"] = []


def _prepare_inserted_reload_node(
    node: Node, value_node: Node, *, fallback_node: Node
) -> None:
    """Attach lowering metadata to a reload inserted after materialization."""
    node.meta["lowering"] = APIFuncLowering(memory_ops.load)
    node.meta["location"] = contextlib.nullcontext()
    node.meta["val"] = value_node.meta.get("val", fallback_node.meta["val"])
