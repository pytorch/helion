"""Lower a reduction over several adjacent dims as one reduction over a
flattened view, so the dims share one reduction index, mask and loop.

``chain_reduction_dims`` decomposes ``x.sum(dim=(1, 2))`` into a chain of
single-dim reductions while tracing and tags the chain's result. Flattening
requires static_shapes so a later input cannot change the view's strides.
``merge_reduction_dims`` then rewrites the producers of each tagged chain's
input to read ``x.flatten(1, 2)`` instead of ``x`` and replaces the chain
with a single reduction.  Chains it cannot rewrite are kept as they are.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

import sympy
import torch
from torch._prims_common import canonicalize_dims
from torch.fx import Node
from torch.fx import map_arg
from torch.fx.experimental import proxy_tensor

from .. import exc
from ..language._tracing_ops import _host_tensor
from ..language.memory_ops import load
from .compile_environment import CompileEnvironment
from .compile_environment import _symint_expr
from .host_function import HostFunction
from .loop_dependency_checker import graph_memory_access_storage_ids
from .variable_origin import BlockSizeOrigin
from .variable_origin import FlattenOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from torch.fx import Graph
    from torch.fx.node import Argument

    from .device_ir import GraphInfo


_META_KEY = "helion_multi_dim_reduction"


def chain_reduction_dims(
    op: torch._ops.OpOverload, fallback: Callable[..., object] | None
) -> Callable[..., object]:
    # Lower a reduction over several dims as a chain of single-dim reductions,
    # innermost (highest) dim first so the lower dim numbers stay valid
    # without keepdim.
    def decomp(
        x: torch.Tensor,
        dim: list[int] | None = None,
        keepdim: bool = False,
        **kwargs: object,
    ) -> object:
        if (
            not dim
            or len(dims := set(canonicalize_dims(x.ndim, dim))) != len(dim)
            or len(dims) < 2
        ):
            if fallback is None:
                return NotImplemented
            return fallback(x, dim, keepdim, **kwargs)
        merged = (op, sorted(dims), keepdim, {**kwargs})
        source = x
        mode = proxy_tensor.get_proxy_mode()
        source_node = None
        existing_nodes: set[Node] = set()
        if mode is not None:
            source_node = cast(
                "Any", proxy_tensor.get_proxy_slot(source, mode.tracer)
            ).proxy.node
            existing_nodes = set(source_node.graph.nodes)
        # Like eager, cast the input to a low-precision dtype= first, then
        # accumulate in fp32 across every reduced dim and round once.
        out_dtype = cast("torch.dtype", kwargs.get("dtype") or x.dtype)
        upcast = op in (
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.mean.dim,
        ) and out_dtype in (torch.float16, torch.bfloat16)
        bool_extrema = source.dtype == torch.bool and op in (
            torch.ops.aten.amax.default,
            torch.ops.aten.amin.default,
        )
        # Bool extrema otherwise decompose to any/all, which not every
        # backend lowers. Integer min/max has the same truth table.
        if bool_extrema:
            x = x.to(torch.uint8)
        if upcast:
            x = x.to(out_dtype)
            kwargs["dtype"] = torch.float32
        for d in sorted(dims, reverse=True):
            x = op(x, [d], keepdim, **kwargs)
        result = x.to(out_dtype) if upcast or bool_extrema else x
        # Tag the chain with its source so merge_reduction_dims can replace it.
        # The merged reduction is not decomposed again, so leave the bool
        # inputs amax/amin decompose (to any/all) chained.
        if mode is not None and not (
            source.dtype == torch.bool
            and op in (torch.ops.aten.amax.default, torch.ops.aten.amin.default)
        ):
            node = cast(
                "Any", proxy_tensor.get_proxy_slot(result, mode.tracer)
            ).proxy.node
            chain_nodes = frozenset(node.graph.nodes) - existing_nodes
            node.meta[_META_KEY] = (source_node, *merged, chain_nodes)
        return result

    return decomp


def merge_reduction_dims(graphs: Sequence[GraphInfo]) -> set[int]:
    """Merge the dims of each tagged reduction chain in ``graphs`` that allows
    it; returns the block ids of the rdims the merged dims had."""
    env = CompileEnvironment.current()
    # Other backends' tensor argument handling is untested with views, and
    # without static_shapes kernels aren't specialized on strides, so a later
    # input's flatten() might copy.
    enabled = env.backend.name in ("triton", "cute") and env.settings.static_shapes
    replaced: set[int] = set()
    excluded: set[int] | None = None
    # Shared across graphs, so each flattened view is one kernel argument
    flat_views: dict[tuple[int, int, int], torch.Tensor] = {}
    # a replaced chain's result -> its replacement, which later chains reduce
    replacements: dict[Node, Node] = {}
    for graph_info in graphs:
        graph = graph_info.graph
        flat_nodes: dict[tuple[Node, int, int], Node] = {}
        for node in [*graph.nodes]:
            if _META_KEY not in node.meta:
                continue
            source, op, dims, keepdim, kwargs, chain_nodes = node.meta.pop(_META_KEY)
            source = replacements.get(source, source)
            # A full-slice tensor's size is the original extent, even when
            # its reduction is later rolled in chunks. Keep scalar size reads
            # outside the rolled graph by specializing their static extents.
            if env.settings.static_shapes:
                _specialize_reduced_sizes(source, dims)
            if (
                not enabled
                or source.graph is not graph
                or dims != [*range(dims[0], dims[-1] + 1)]
            ):
                continue
            if excluded is None:
                # Leave tensors the kernel writes unmerged: a merged load reads
                # them in a different layout than they are written, and CuTe
                # may promote an atomic target's dtype after merging, which
                # its flattened view would miss.
                excluded = {
                    storage
                    for i in range(len(graphs))
                    for storage in graph_memory_access_storage_ids([*graphs], i)[1]
                }
            merger = _Merger(source, dims, flat_views, flat_nodes, excluded)
            if (
                merger.kind(source) != "merge"
                or (chain := merger.chain(node, chain_nodes)) is None
            ):
                continue
            merged = merger.clone(source)
            with graph.inserting_before(node.next):
                result = _new_node(
                    # Emit the single-axis ATen op directly: decomposing it
                    # again would recreate the chain being replaced.
                    graph,
                    op,
                    (merged, [dims[0]], keepdim),
                    kwargs,
                    node.meta,
                )
                for d in dims[1:] if keepdim else ():
                    result = _new_node(
                        graph,
                        torch.ops.aten.unsqueeze.default,
                        (result, d),
                        {},
                        node.meta,
                    )
            node.replace_all_uses_with(result)
            replacements[node] = result
            replaced.update(
                b for s in merger.sizes if (b := env.get_block_id(s)) is not None
            )
            # Erase the replaced chain and the producers only it used
            dead = chain | {n for n, k in merger.kinds.items() if k == "merge"}
            while unused := [n for n in dead if not n.users]:
                for n in unused:
                    graph.erase_node(n)
                    dead.remove(n)
    return replaced


def _specialize_reduced_sizes(source: Node, dims: list[int]) -> None:
    """Keep static full-slice size reads outside a later rolled reduction."""
    env = CompileEnvironment.current()
    for user in list(source.users):
        if user.target is not torch.ops.aten.sym_size.int or user.args[1] not in dims:
            continue
        block_id = env.get_block_id(source.meta["val"].size(user.args[1]))
        if block_id is None or not env.block_sizes[block_id].reduction:
            continue
        extent = env.block_sizes[block_id].numel
        if not extent.is_Integer:
            continue
        for consumer in list(user.users):
            consumer.args = map_arg(
                consumer.args,
                lambda n, size=int(extent), source=user: size if n is source else n,
            )
            consumer.kwargs = map_arg(
                consumer.kwargs,
                lambda n, size=int(extent), source=user: size if n is source else n,
            )
        source.graph.erase_node(user)


def _new_node(
    graph: Graph,
    target: Callable[..., object],
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    meta: dict[str, object],
) -> Node:
    """Insert ``target(*args, **kwargs)`` at the current insert point, with
    ``meta`` copied and its fake value recomputed."""
    node = graph.call_function(target, args, kwargs)
    vals, kwvals = map_arg((args, kwargs), lambda a: a.meta["val"])
    node.meta = {**_source_meta(meta), "val": target(*vals, **kwvals)}
    if isinstance(target, torch._ops.OpOverload):
        node.meta["original_aten"] = target
    return node


def _source_meta(meta: dict[str, object]) -> dict[str, object]:
    """The parts of ``meta`` that describe where a node came from."""
    return {k: meta[k] for k in ("location", "stack_trace", "seq_nr") if k in meta}


class _Merger:
    """Rewrites the producers of a reduction's input over the merged dims.

    The merged dims are counted from the right so they line up across
    broadcasting.  A producer covering them either has their full extents
    (a load with ``:`` slices, or a pointwise op of such producers), or size
    1 in each (a broadcast operand, squeezed to one size-1 dim).
    """

    def __init__(
        self,
        source: Node,
        dims: list[int],
        flat_views: dict[tuple[int, int, int], torch.Tensor],
        flat_nodes: dict[tuple[Node, int, int], Node],
        excluded_storages: set[int],
    ) -> None:
        val = source.meta["val"]
        self.env = CompileEnvironment.current()
        self.offsets = [d - val.ndim for d in dims]
        self.sizes = [val.size(d) for d in dims]
        self.kinds: dict[Node, str | None] = {}
        self.flat_loads: dict[Node, tuple[int, int]] = {}
        self.flat_views = flat_views
        # (host tensor node, merged dims) -> its flattened view's node
        self.flat_nodes = flat_nodes
        self.excluded_storages = excluded_storages
        self.cloned: dict[Node, Node] = {}

    def kind(self, node: Node) -> str | None:
        """'merge', 'squeeze' or 'keep' (the dims are left of ``node``'s
        shape), or None if the dims can't be merged."""
        if node not in self.kinds:
            self.kinds[node] = self._kind(node)
        return self.kinds[node]

    def _kind(self, node: Node) -> str | None:
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            return "keep"
        positions = [val.ndim + o for o in self.offsets]
        if all(p < 0 for p in positions):
            return "keep"
        if any(p < 0 for p in positions):
            return None
        sizes = [val.size(p) for p in positions]
        if all(self.env.known_equal(s, 1) for s in sizes):
            return "squeeze"
        if not all(map(self.env.known_equal, sizes, self.sizes)):
            return None
        if (flat := self._flat_load(node)) is not None:
            self.flat_loads[node] = flat
            return "merge"
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.pointwise in node.target.tags
            # cloning an in-place op would leave the original live
            and not node.target._schema.is_mutable
            and all(self.kind(a) is not None for a in node.all_input_nodes)
        ):
            return "merge"
        return None

    def chain(self, result: Node, chain_nodes: frozenset[Node]) -> set[Node] | None:
        """The nodes of the chain ending at ``result``, or None if the merged
        producers have other users, which would then load them twice."""
        chain: set[Node] = set()
        stack = [result]
        while stack:
            n = stack.pop()
            if n in chain_nodes and n not in chain and n not in self.kinds:
                chain.add(n)
                stack.extend(n.all_input_nodes)
        if all(
            user in chain or self.kinds.get(user) == "merge"
            for node, kind in self.kinds.items()
            if kind == "merge"
            for user in node.users
        ):
            return chain
        return None

    def _flat_load(self, node: Node) -> tuple[int, int] | None:
        """For ``load(x, [..., :, :, ...])`` whose ``:`` slices are the merged
        dims of a host tensor ``x`` that can flatten them into a view, the
        positions of the first slice in the index and in ``x``.  How each
        index kind maps to dims follows ``SubscriptIndexing.compute_shape``."""
        if node.target is not load or node.args[2:3] not in ((), (None,)):
            return None  # not a load, or one with an extra_mask
        host = node.args[0]
        if not (isinstance(host, torch.fx.Node) and host.target is _host_tensor):
            return None
        tensor = host.meta["val"]
        if id(tensor.untyped_storage()) in self.excluded_storages:
            return None
        index = cast("list[Argument]", node.args[1])
        first = node.meta["val"].ndim + self.offsets[0]
        merged = range(first, first + len(self.offsets))
        out_dim = tensor_dim = 0
        slices: list[tuple[int, int]] = []
        for i, k in enumerate(index):
            if isinstance(k, torch.fx.Node):
                k = k.meta["val"]
                if isinstance(k, torch.Tensor):
                    return None  # tensor indices may add or broadcast dims
            if isinstance(k, (int, torch.SymInt)) and not self._is_tile(k):
                tensor_dim += 1  # a scalar index drops its dim
                continue
            if out_dim in merged:
                if k != slice(None):
                    return None
                slices.append((i, tensor_dim))
            elif not (k is None or isinstance(k, (slice, torch.SymInt))):
                return None
            out_dim += 1
            tensor_dim += k is not None
        tensor_dims = [d for _, d in slices]
        if len(slices) != len(merged) or any(
            b != a + 1
            or not self.env.known_equal(
                tensor.stride(a), tensor.stride(b) * tensor.size(b)
            )
            for a, b in itertools.pairwise(tensor_dims)
        ):
            return None
        return slices[0]

    def _extent(self, size: int | torch.SymInt) -> int | torch.SymInt:
        """``size``, or the extent of the rdim it is the block size of."""
        block_id = self.env.get_block_id(size)
        if block_id is not None and (info := self.env.block_sizes[block_id]).reduction:
            return cast("int | torch.SymInt", info.size)
        return size

    @staticmethod
    def _is_tile(k: int | torch.SymInt) -> bool:
        """Whether index ``k`` is a tile's block size (which keeps its dim, see
        ``SubscriptIndexing.compute_shape``) rather than a scalar."""
        if not isinstance(k, torch.SymInt) or not isinstance(
            symbol := _symint_expr(k), sympy.Symbol
        ):
            return False
        origin = HostFunction.current().expr_to_origin.get(symbol)
        return origin is not None and isinstance(origin.origin, BlockSizeOrigin)

    def clone(self, node: Node) -> Node:
        """``node`` with the merged dims flattened into one."""
        if node in self.cloned:
            return self.cloned[node]
        kind = self.kinds[node]
        if kind == "keep":
            return node
        result = node
        first = node.meta["val"].ndim + self.offsets[0]
        with node.graph.inserting_before(node.next):
            if kind == "squeeze":
                for p in reversed(range(first + 1, first + len(self.offsets))):
                    result = _new_node(
                        node.graph,
                        torch.ops.aten.squeeze.dim,
                        (result, p),
                        {},
                        node.meta,
                    )
            elif node in self.flat_loads:
                result = self._clone_load(node, *self.flat_loads[node])
            else:
                result = _new_node(
                    node.graph,
                    cast("Callable[..., object]", node.target),
                    map_arg(node.args, self.clone),
                    map_arg(node.kwargs, self.clone),
                    node.meta,
                )
        self.cloned[node] = result
        return result

    def _clone_load(self, node: Node, index_pos: int, tensor_dim: int) -> Node:
        """``load(x.flatten(...), [..., :, ...])``, inserted after ``node``."""
        host = cast("Node", node.args[0])
        end = tensor_dim + len(self.offsets) - 1
        key = (id(host.meta["val"]), tensor_dim, end)
        if (tensor := self.flat_views.get(key)) is None:
            tensor = host.meta["val"].flatten(tensor_dim, end)
            tensor_to_origin = HostFunction.current().tensor_to_origin
            tensor_to_origin[tensor] = FlattenOrigin(
                tensor_to_origin[host.meta["val"]], tensor_dim, end
            )
            self.flat_views[key] = tensor
        if (flat := self.flat_nodes.get((host, tensor_dim, end))) is None:
            origin = HostFunction.current().tensor_to_origin[tensor]
            # right after the host tensor, so before every load of it
            with node.graph.inserting_before(host.next):
                flat = node.graph.call_function(
                    _host_tensor,
                    (origin.host_str(),),
                    name=origin.suggest_var_name(),
                )
                flat.meta = {**_source_meta(host.meta), "val": tensor}
            self.flat_nodes[host, tensor_dim, end] = flat
        index = [*cast("list[Argument]", node.args[1])]
        del index[index_pos + 1 : index_pos + len(self.offsets)]
        result = _new_node(
            node.graph, load, (flat, index, *node.args[2:]), {}, node.meta
        )
        # compute_shape (via load) must agree with how _flat_load mapped dims
        expected = [*node.meta["val"].shape]
        first = len(expected) + self.offsets[0]
        expected[first : first + len(self.offsets)] = [tensor.size(tensor_dim)]
        actual = [*result.meta["val"].shape]
        if len(actual) != len(expected) or not all(
            map(
                self.env.known_equal,
                map(self._extent, actual),
                map(self._extent, expected),
            )
        ):
            raise exc.InternalError(
                RuntimeError(f"merged load shape {actual}, expected {expected}")
            )
        return result
