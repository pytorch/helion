"""Epilogue subtiling: split → pointwise → multi-store pattern.

Splits the epilogue along a tile dimension by a power-of-two factor S.

    Before:
        acc = matmul(...)                    # [M, N] in TMEM
        v = pointwise(acc)                   # e.g. bias add, cast
        store(v, [tile_m, tile_n])

    After (S=2):
        acc = matmul(...)                    # [M, N]
        view -> [S, M, N//S]                # reshape to expose split factor
        piece_0, piece_1 = tl.split(...)    # split into S pieces

        v_0 = pointwise(piece_0)            # [M, N//S] -- cloned per piece
        v_1 = pointwise(piece_1)

        store(v_0, [tile_m, tile_n + 0])    # store each piece separately
        store(v_1, [tile_m, tile_n + N//S])
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch
from torch.fx import map_arg

from ..autotuner.config_fragment import integer_power_of_two
from ..language.view_ops import split as hl_split
from .compile_environment import CompileEnvironment
from .inductor_lowering import APIFuncLowering

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


class EpilogueSubtilingCandidate(NamedTuple):
    store_node: torch.fx.Node
    pointwise_nodes: list[torch.fx.Node]
    boundary: torch.fx.Node
    split_dim: int
    block_id: int


def apply_epilogue_subtiling(
    graph: torch.fx.Graph,
    split_factor: int = 2,
    configured_block_sizes: dict[int, int | torch.SymInt] | None = None,
) -> bool:
    """Apply epilogue subtiling to the graph in-place."""
    if split_factor < 2 or not integer_power_of_two(split_factor):
        raise ValueError(
            f"epilogue_subtile expects a power-of-two >= 2, got {split_factor}"
        )
    env = CompileEnvironment.current()
    configured_block_sizes = configured_block_sizes or {}
    changed = False

    for candidate in _iter_eligible_epilogue_chains(graph, env, split_factor):
        _rewrite_chain(
            graph,
            env,
            candidate,
            split_factor,
            configured_block_sizes,
        )
        changed = True

    if changed:
        graph.eliminate_dead_code()
    return changed


def has_epilogue_subtiling_candidate(
    graph: torch.fx.Graph, split_factor: int = 2
) -> bool:
    """Return whether the graph has at least one eligible epilogue chain."""
    if not (integer_power_of_two(split_factor) and split_factor >= 2):
        return False
    env = CompileEnvironment.current()
    return any(True for _ in _iter_eligible_epilogue_chains(graph, env, split_factor))


def _iter_eligible_epilogue_chains(
    graph: torch.fx.Graph,
    env: CompileEnvironment,
    split_factor: int,
) -> Iterator[EpilogueSubtilingCandidate]:
    """Yield eligible store + pointwise chains that can be split."""
    from ..language.memory_ops import store

    for store_node in list(graph.nodes):
        if store_node.op != "call_function" or store_node.target is not store:
            continue

        value_node = store_node.args[2]
        if not isinstance(value_node, torch.fx.Node):
            continue

        chain = _trace_pointwise_chain(value_node)
        if chain is None:
            continue

        boundary, pointwise_nodes = chain
        boundary_val = boundary.meta.get("val")
        if not isinstance(boundary_val, torch.Tensor):
            continue

        split_candidate = _find_split_candidate(boundary_val, env, split_factor)
        if split_candidate is None:
            continue

        split_dim, block_id = split_candidate
        yield EpilogueSubtilingCandidate(
            store_node=store_node,
            pointwise_nodes=pointwise_nodes,
            boundary=boundary,
            split_dim=split_dim,
            block_id=block_id,
        )


def _rewrite_chain(
    graph: torch.fx.Graph,
    env: CompileEnvironment,
    candidate: EpilogueSubtilingCandidate,
    split_factor: int,
    configured_block_sizes: dict[int, int | torch.SymInt],
) -> None:
    boundary = candidate.boundary
    block_id = candidate.block_id
    split_dim = candidate.split_dim
    boundary_val = boundary.meta["val"]
    piece_shape_size = boundary_val.size(split_dim) // split_factor
    piece_block_size = (
        configured_block_sizes.get(block_id, boundary_val.size(split_dim))
        // split_factor
    )

    with graph.inserting_before(candidate.pointwise_nodes[0]):
        pieces = _reshape_and_split(graph, boundary, split_dim, split_factor)

    # Clone pointwise ops for each piece and split broadcasted inputs as needed.
    node_pieces: dict[torch.fx.Node, list[torch.fx.Node]] = {boundary: pieces}
    with graph.inserting_before(candidate.store_node):
        for pw in candidate.pointwise_nodes:
            for arg_node in _tensor_input_nodes(pw):
                if arg_node in node_pieces:
                    continue
                if split_nodes := _split_node_for_block(
                    graph,
                    env,
                    arg_node,
                    block_id,
                    split_factor,
                ):
                    node_pieces[arg_node] = split_nodes

            node_pieces[pw] = [
                _clone_pointwise_piece(
                    graph,
                    env,
                    pw,
                    piece_idx,
                    node_pieces,
                    piece_shape_size,
                    block_id,
                )
                for piece_idx in range(split_factor)
            ]

    pw_results = node_pieces[candidate.pointwise_nodes[-1]]

    _rewrite_store(
        graph,
        candidate.store_node,
        pw_results,
        split_dim,
        piece_shape_size,
        piece_block_size,
        block_id,
    )


def _tensor_input_nodes(node: torch.fx.Node) -> list[torch.fx.Node]:
    return [
        n for n in node.all_input_nodes if isinstance(n.meta.get("val"), torch.Tensor)
    ]


def _clone_pointwise_piece(
    graph: torch.fx.Graph,
    env: CompileEnvironment,
    pointwise_node: torch.fx.Node,
    piece_idx: int,
    node_pieces: dict[torch.fx.Node, list[torch.fx.Node]],
    piece_shape_size: int | torch.SymInt,
    block_id: int,
) -> torch.fx.Node:
    def _remap(arg: torch.fx.Node) -> torch.fx.Node:
        split_nodes = node_pieces.get(arg)
        return arg if split_nodes is None else split_nodes[piece_idx]

    cloned = graph.call_function(
        cast("Callable[..., object]", pointwise_node.target),
        map_arg(pointwise_node.args, _remap),
        map_arg(pointwise_node.kwargs, _remap),
    )
    orig_val = pointwise_node.meta.get("val")
    if isinstance(orig_val, torch.Tensor):
        cloned.meta = {
            **pointwise_node.meta,
            "val": orig_val.new_empty(
                [
                    piece_shape_size if env.get_block_id(size) == block_id else size
                    for size in orig_val.shape
                ]
            ),
        }
    else:
        cloned.meta = {**pointwise_node.meta, "val": orig_val}
    return cloned


def _split_node_for_block(
    graph: torch.fx.Graph,
    env: CompileEnvironment,
    node: torch.fx.Node,
    block_id: int,
    split_factor: int,
) -> list[torch.fx.Node] | None:
    """Split a node along the dim matching block_id, or return None."""
    node_val = node.meta.get("val")
    if not isinstance(node_val, torch.Tensor):
        return None

    split_dim = _find_split_dim_for_block(node_val, env, block_id, split_factor)
    if split_dim is None:
        return None

    return _reshape_and_split(graph, node, split_dim, split_factor)


def _rewrite_store(
    graph: torch.fx.Graph,
    store_node: torch.fx.Node,
    pieces: list[torch.fx.Node],
    split_dim: int,
    piece_shape_size: int | torch.SymInt,
    piece_block_size: int | torch.SymInt,
    block_id: int,
) -> None:
    subscript_arg = store_node.args[1]
    assert isinstance(subscript_arg, (list, tuple))
    subscript = list(cast("list[object] | tuple[object, ...]", subscript_arg))
    base_index_node = subscript[split_dim]
    assert isinstance(base_index_node, torch.fx.Node)
    group_id = store_node.name

    with graph.inserting_before(store_node):
        for piece_idx, piece in enumerate(pieces):
            new_subscript = [*subscript]
            new_subscript[split_dim] = _new_subtile_index_node(
                graph,
                base_index_node,
                piece_idx,
                piece_shape_size,
                piece_block_size,
                block_id,
            )
            new_args = (
                store_node.args[0],
                new_subscript,
                piece,
                *store_node.args[3:],
            )
            new_store = graph.call_function(
                cast("Callable[..., object]", store_node.target),
                # pyrefly: ignore [bad-argument-type]
                new_args,
                store_node.kwargs,
            )
            new_store.meta = {
                **store_node.meta,
                "epilogue_subtile_group_id": group_id,
                "epilogue_subtile_primary_store": piece_idx == 0,
            }

    graph.erase_node(store_node)


def _new_subtile_index_node(
    graph: torch.fx.Graph,
    base_index_node: torch.fx.Node,
    piece_idx: int,
    piece_shape_size: int | torch.SymInt,
    piece_block_size: int | torch.SymInt,
    block_id: int,
) -> torch.fx.Node:
    offset = piece_idx * piece_shape_size
    node = graph.call_function(operator.add, (base_index_node, 0))
    node.meta = {
        **base_index_node.meta,
        "val": base_index_node.meta["val"] + offset,
        "tile_with_offset": {
            "block_id": block_id,
            "offset": offset,
            "block_size": piece_block_size,
        },
    }
    return node


def _reshape_and_split(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    split_dim: int,
    split_factor: int,
) -> list[torch.fx.Node]:
    """Reshape [... N ...] -> [... S, N//S ...] at split_dim and recursively split."""
    val = node.meta["val"]
    piece_size = val.size(split_dim) // split_factor
    reshape_shape = list(val.shape)
    reshape_shape[split_dim] = split_factor
    reshape_shape.insert(split_dim + 1, piece_size)
    reshaped_val = val.reshape(reshape_shape)
    reshape_node = _new_node(
        graph,
        torch.ops.aten.view.default,
        (node, reshape_shape),
        node.meta,
        reshaped_val,
    )
    return _recursive_split(
        graph,
        reshape_node,
        reshaped_val,
        split_dim,
        split_factor,
        node.meta,
    )


def _split_dim_2(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    val: torch.Tensor,
    dim: int,
    base_meta: dict[str, object],
) -> list[torch.fx.Node]:
    """Split a tensor along ``dim`` (which must have size 2) using hl.split."""
    ndim = len(val.shape)
    # hl.split requires the split dim to be last
    if dim != ndim - 1:
        perm = [i for i in range(ndim) if i != dim] + [dim]
        val = val.permute(perm)
        node = _new_node(
            graph,
            torch.ops.aten.permute.default,
            (node, perm),
            base_meta,
            val,
        )

    out_shape = val.shape[:-1]
    half_vals = [val.new_empty(out_shape) for _ in range(2)]
    split_node = graph.call_function(hl_split, (node,))
    split_node.meta = {
        **base_meta,
        "val": tuple(half_vals),
        "lowering": APIFuncLowering(hl_split),
    }

    halves = []
    for i in range(2):
        gi = graph.call_function(operator.getitem, (split_node, i))
        gi.meta = {**base_meta, "val": half_vals[i]}
        _set_lowering(gi)
        halves.append(gi)
    return halves


def _recursive_split(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    val: torch.Tensor,
    dim: int,
    factor: int,
    base_meta: dict[str, object],
) -> list[torch.fx.Node]:
    """Recursively split along ``dim`` into ``factor`` pieces via tl.split."""
    if factor == 1:
        return [node]

    if factor == 2:
        return _split_dim_2(graph, node, val, dim, base_meta)

    # factor > 2: reshape [S] → [2, S//2] at dim, split the 2,
    # then recursively split each half's [S//2].
    new_shape = list(val.shape)
    new_shape[dim] = 2
    new_shape.insert(dim + 1, factor // 2)
    reshaped_val = val.reshape(new_shape)
    reshaped = _new_node(
        graph,
        torch.ops.aten.view.default,
        (node, new_shape),
        base_meta,
        reshaped_val,
    )

    halves = _split_dim_2(graph, reshaped, reshaped_val, dim, base_meta)
    result: list[torch.fx.Node] = []
    for h in halves:
        result.extend(
            _recursive_split(
                graph,
                h,
                h.meta["val"],
                dim,
                factor // 2,
                base_meta,
            )
        )
    return result


def _trace_pointwise_chain(
    value_node: torch.fx.Node,
) -> tuple[torch.fx.Node, list[torch.fx.Node]] | None:
    """Return the tensor-producing boundary and pointwise nodes feeding a store."""
    pointwise_nodes: list[torch.fx.Node] = []
    current: torch.fx.Node | None = value_node
    while (
        isinstance(current, torch.fx.Node)
        and current.op == "call_function"
        and isinstance(current.target, torch._ops.OpOverload)
        and len(current.users) == 1
    ):
        pointwise_nodes.append(current)
        current = next(
            (
                arg
                for arg in current.args
                if isinstance(arg, torch.fx.Node)
                and isinstance(arg.meta.get("val"), torch.Tensor)
            ),
            None,
        )

    if not pointwise_nodes:
        return None

    pointwise_nodes.reverse()
    boundary = current if isinstance(current, torch.fx.Node) else pointwise_nodes[-1]
    return boundary, pointwise_nodes


def _find_split_candidate(
    value: torch.Tensor,
    env: CompileEnvironment,
    split_factor: int,
) -> tuple[int, int] | None:
    """Find a non-reduction, non-jagged tile dim to split. Returns (dim, block_id)."""
    for dim in range(value.dim() - 1, -1, -1):
        size = value.size(dim)
        block_id = env.get_block_id(size)
        if (
            block_id is not None
            and not env.block_sizes[block_id].reduction
            and not env.is_jagged_tile(block_id)
            and size % split_factor == 0
        ):
            return dim, block_id
    return None


def _find_split_dim_for_block(
    value: torch.Tensor,
    env: CompileEnvironment,
    block_id: int,
    split_factor: int,
) -> int | None:
    """Find a dim matching block_id that is divisible by split_factor."""
    for dim in range(value.dim() - 1, -1, -1):
        size = value.size(dim)
        if env.get_block_id(size) == block_id and size % split_factor == 0:
            return dim
    return None


def _new_node(
    graph: torch.fx.Graph,
    target: object,
    args: tuple[object, ...],
    base_meta: dict[str, object],
    val: torch.Tensor,
) -> torch.fx.Node:
    node = graph.call_function(
        cast("Callable[..., object]", target),
        # pyrefly: ignore [bad-argument-type]
        args,
    )
    node.meta = {**base_meta, "val": val}
    _set_lowering(node)
    return node


def _set_lowering(node: torch.fx.Node) -> None:
    from .aten_lowering import aten_lowering_dispatch

    if node.target in aten_lowering_dispatch:
        node.meta["lowering"] = aten_lowering_dispatch[node.target](node)
