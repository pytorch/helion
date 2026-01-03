from __future__ import annotations

import operator
import torch
from dataclasses import dataclass
from typing import Callable, Sequence

import sympy
from torch.fx import Node
from torch.utils._sympy.functions import FloorDiv

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import BaseView
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import DtypeView
from torch._inductor.ir import ExpandView
from torch._inductor.ir import GenericView
from torch._inductor.ir import get_stride_order
from torch._inductor.ir import IRNode
from torch._inductor.ir import PermuteView
from torch._inductor.ir import Pointwise
from torch._inductor.ir import ReinterpretView
from torch._inductor.ir import SliceView
from torch._inductor.ir import stride_order2fill_order
from torch._inductor.ir import View
from torch._inductor.ir import fuse_reindexing
from torch._inductor.ir import same_reorder
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V


@dataclass
class HelionIndexingTransform:
    kernel_symbols: list[sympy.Symbol]
    index_exprs: list[sympy.Expr]
    mod_masks: list[sympy.Expr]
    broadcast_dims: list[tuple[int, int]] | None
    unsupported: bool = False


def _identity_reindexer(index: list[sympy.Expr]) -> list[sympy.Expr]:
    return list(index)


def _has_overlapping_strides(layout: object) -> bool:
    if not hasattr(layout, "size") or not hasattr(layout, "stride"):
        return False
    sizes = list(layout.size)
    strides = list(layout.stride)
    if len(sizes) != len(strides):
        return False
    size_ints: list[int] = []
    stride_ints: list[int] = []
    for size, stride in zip(sizes, strides, strict=True):
        if isinstance(size, sympy.Integer):
            size_ints.append(int(size))
        elif isinstance(size, int):
            size_ints.append(size)
        else:
            return False
        if isinstance(stride, sympy.Integer):
            stride_ints.append(int(stride))
        elif isinstance(stride, int):
            stride_ints.append(stride)
        else:
            return False

    seen: dict[int, int] = {}
    for size, stride in zip(size_ints, stride_ints, strict=True):
        if size <= 1 or stride == 0:
            continue
        if stride < 0:
            return True
        if stride in seen and seen[stride] > 1:
            return True
        seen[stride] = size
    return False


def _layout_reindexer_to_view(
    view: ReinterpretView,
) -> Callable[[list[sympy.Expr]], list[sympy.Expr]] | None:
    return _layout_reindexer_from_layout(view.get_layout(), list(view.data.get_size()))


def _layout_reindexer_from_layout(
    layout: object, base_size: list[sympy.Expr]
) -> Callable[[list[sympy.Expr]], list[sympy.Expr]] | None:
    if not hasattr(layout, "size") or not hasattr(layout, "stride"):
        return None
    view_size = list(layout.size)
    try:
        V.graph.sizevars.check_equals(sympy_product(base_size), sympy_product(view_size))
    except Exception:
        return None

    stride_order = list(get_stride_order(V.graph.sizevars.size_hints(layout.stride)))
    fill_order = stride_order2fill_order(stride_order)
    reversed_fill_order = list(reversed(fill_order))
    size_ordered = [view_size[i] for i in reversed_fill_order]
    reshape_reindex = View.dynamic_reshape_indexer(size_ordered, base_size)
    from_stride_ordered_to_view_order = [
        (len(stride_order) - 1) - stride_order[i] for i in range(len(stride_order))
    ]
    stride_reindex = same_reorder(from_stride_ordered_to_view_order)
    return fuse_reindexing(stride_reindex, reshape_reindex)


def _invert_affine_view(
    view: BaseView,
) -> tuple[
    Callable[[list[sympy.Expr]], list[sympy.Expr]],
    Callable[[list[sympy.Expr]], list[sympy.Expr]],
] | None:
    view_size = list(view.get_size())
    view_syms = [sympy.Symbol(f"view_{i}") for i in range(len(view_size))]
    base_exprs = list(view.make_reindexer()(view_syms))
    sizevars = V.graph.sizevars

    mapping: dict[
        int, tuple[int, sympy.Expr, sympy.Expr, list[tuple[int, sympy.Expr, sympy.Expr]]]
    ] = {}
    const_constraints: list[tuple[int, sympy.Expr]] = []
    for base_dim, expr in enumerate(base_exprs):
        expr = sympy.expand(expr)
        used = [i for i, sym in enumerate(view_syms) if sym in expr.free_symbols]
        if not used:
            if not expr.free_symbols:
                const_constraints.append((base_dim, expr))
            continue
        if len(used) > 1:
            return None
        view_dim = used[0]
        coeff = sympy.expand(expr.coeff(view_syms[view_dim]))
        remainder = sympy.expand(expr - coeff * view_syms[view_dim])
        if remainder.free_symbols:
            return None
        if sizevars.statically_known_equals(coeff, 0):
            continue
        if sizevars.statically_known_lt(coeff, 0):
            return None
        if view_dim in mapping:
            existing = mapping[view_dim]
            mapping[view_dim] = (
                existing[0],
                existing[1],
                existing[2],
                [*existing[3], (base_dim, coeff, remainder)],
            )
        else:
            mapping[view_dim] = (base_dim, coeff, remainder, [])

    def _add_alias_symbols(
        expr: sympy.Expr,
        index: list[sympy.Expr],
        aliases: list[tuple[int, sympy.Expr, sympy.Expr]],
    ) -> sympy.Expr:
        out = expr
        for alias_dim, _alias_coeff, _alias_remainder in aliases:
            term = sympy.Mul(sympy.Integer(0), index[alias_dim], evaluate=False)
            out = sympy.Add(out, term, evaluate=False)
        return out

    def inv_reindex(index: list[sympy.Expr]) -> list[sympy.Expr]:
        out: list[sympy.Expr] = []
        for view_dim in range(len(view_size)):
            if view_dim in mapping:
                base_dim, coeff, remainder, aliases = mapping[view_dim]
                expr = FloorDiv(index[base_dim] - remainder, coeff)
                out.append(_add_alias_symbols(expr, index, aliases))
            else:
                out.append(sympy.S.Zero)
        return out

    def mask_builder(input_exprs: list[sympy.Expr]) -> list[sympy.Expr]:
        masks: list[sympy.Expr] = []
        for view_dim, (base_dim, coeff, remainder, aliases) in mapping.items():
            if not sizevars.statically_known_equals(
                coeff, 1
            ) or not sizevars.statically_known_equals(remainder, 0):
                expr = input_exprs[base_dim] - (
                    FloorDiv(input_exprs[base_dim] - remainder, coeff) * coeff + remainder
                )
                masks.append(expr)
            if aliases:
                view_expr = FloorDiv(input_exprs[base_dim] - remainder, coeff)
                for alias_dim, alias_coeff, alias_remainder in aliases:
                    expr = input_exprs[alias_dim] - (
                        alias_coeff * view_expr + alias_remainder
                    )
                    masks.append(expr)
        for base_dim, const in const_constraints:
            masks.append(input_exprs[base_dim] - const)
        return masks

    return inv_reindex, mask_builder


def _invert_reinterpret_view_affine(
    view: ReinterpretView,
) -> tuple[
    Callable[[list[sympy.Expr]], list[sympy.Expr]],
    Callable[[list[sympy.Expr]], list[sympy.Expr]],
] | None:
    base = view.data
    if not hasattr(base, "get_layout"):
        return None
    base_layout = base.get_layout()
    view_layout = view.get_layout()
    if not hasattr(base_layout, "stride") or not hasattr(view_layout, "stride"):
        return None

    base_stride = list(base_layout.stride)
    view_stride = list(view_layout.stride)
    if len(base_stride) != len(view_stride):
        return None

    sizevars = V.graph.sizevars
    base_offset = getattr(base_layout, "offset", sympy.S.Zero)
    view_offset = getattr(view_layout, "offset", sympy.S.Zero)
    offset_delta = sympy.expand(view_offset - base_offset)

    stride_order = list(get_stride_order(sizevars.size_hints(base_stride)))
    if hasattr(base_layout, "is_stride_ordered") and not base_layout.is_stride_ordered(
        stride_order
    ):
        return None

    steps: list[sympy.Expr] = []
    for b_stride, v_stride in zip(base_stride, view_stride, strict=True):
        if sizevars.statically_known_equals(b_stride, 0):
            if not sizevars.statically_known_equals(v_stride, 0):
                return None
            steps.append(sympy.Integer(1))
            continue
        ratio = sympy.simplify(v_stride / b_stride)
        if ratio.free_symbols:
            return None
        if ratio.is_integer is False:
            return None
        if sizevars.statically_known_leq(ratio, 0):
            return None
        steps.append(sympy.Integer(ratio))

    start: list[sympy.Expr] = [sympy.S.Zero for _ in base_stride]
    remaining = offset_delta
    for dim in reversed(stride_order2fill_order(stride_order)):
        stride = base_stride[dim]
        if sizevars.statically_known_equals(stride, 0):
            if not sizevars.statically_known_equals(remaining, 0):
                return None
            continue
        idx = FloorDiv(remaining, stride)
        start[dim] = idx
        remaining = sympy.expand(remaining - idx * stride)

    if not sizevars.statically_known_equals(remaining, 0):
        return None

    def inv_reindex(index: list[sympy.Expr]) -> list[sympy.Expr]:
        out: list[sympy.Expr] = []
        for dim, step in enumerate(steps):
            if sizevars.statically_known_equals(step, 1):
                out.append(index[dim] - start[dim])
            else:
                out.append(FloorDiv(index[dim] - start[dim], step))
        return out

    def mask_builder(input_exprs: list[sympy.Expr]) -> list[sympy.Expr]:
        masks: list[sympy.Expr] = []
        for dim, step in enumerate(steps):
            if sizevars.statically_known_equals(step, 1):
                continue
            expr = input_exprs[dim] - (
                FloorDiv(input_exprs[dim] - start[dim], step) * step + start[dim]
            )
            masks.append(expr)
        return masks

    return inv_reindex, mask_builder


def _inverse_view_reindexer(
    view: BaseView,
) -> tuple[
    Callable[[list[sympy.Expr]], list[sympy.Expr]],
    Callable[[list[sympy.Expr]], list[sympy.Expr]],
] | None:
    if isinstance(view, DtypeView):
        return _identity_reindexer, lambda _idx: []
    if isinstance(view, ReinterpretView):
        base = view.data
        if hasattr(base, "get_layout"):
            base_layout = base.get_layout()
            view_layout = view.get_layout()
            sizevars = V.graph.sizevars
            if hasattr(base_layout, "size") and hasattr(view_layout, "size"):
                base_size = list(base_layout.size)
                view_size = list(view_layout.size)
                offset_delta = sympy.expand(
                    getattr(view_layout, "offset", sympy.S.Zero)
                    - getattr(base_layout, "offset", sympy.S.Zero)
                )
                same_numel = sizevars.statically_known_equals(
                    sympy_product(base_size), sympy_product(view_size)
                )
                same_offset = sizevars.statically_known_equals(offset_delta, 0)
                if same_numel and same_offset:
                    reindex = _layout_reindexer_to_view(view)
                    if reindex is not None:
                        return reindex, lambda _idx: []

                if hasattr(base_layout, "stride") and hasattr(view_layout, "stride"):
                    base_stride = list(base_layout.stride)
                    view_stride = list(view_layout.stride)
                    if len(base_stride) == len(view_stride):
                        perm: list[int] | None = []
                        used: set[int] = set()
                        for v_stride, v_size in zip(
                            view_stride, view_size, strict=True
                        ):
                            match = None
                            for idx, (b_stride, b_size) in enumerate(
                                zip(base_stride, base_size, strict=True)
                            ):
                                if idx in used:
                                    continue
                                if not sizevars.statically_known_equals(
                                    v_stride, b_stride
                                ):
                                    continue
                                if not sizevars.statically_known_leq(v_size, b_size):
                                    continue
                                match = idx
                                break
                            if match is None:
                                perm = None
                                break
                            perm.append(match)
                            used.add(match)

                        if perm is not None and sizevars.statically_known_equals(
                            offset_delta, 0
                        ):
                            stride_order = list(
                                get_stride_order(sizevars.size_hints(base_stride))
                            )
                            start: list[sympy.Expr] = [
                                sympy.S.Zero for _ in base_stride
                            ]
                            remaining = offset_delta
                            for dim in reversed(stride_order2fill_order(stride_order)):
                                stride = base_stride[dim]
                                if sizevars.statically_known_equals(stride, 0):
                                    if not sizevars.statically_known_equals(
                                        remaining, 0
                                    ):
                                        perm = None
                                        break
                                    continue
                                idx = FloorDiv(remaining, stride)
                                start[dim] = idx
                                remaining = sympy.expand(remaining - idx * stride)
                            if perm is not None and sizevars.statically_known_equals(
                                remaining, 0
                            ):
                                def inv_reindex(
                                    index: list[sympy.Expr],
                                ) -> list[sympy.Expr]:
                                    return [
                                        index[perm_dim] - start[perm_dim]
                                        for perm_dim in perm
                                    ]

                                return inv_reindex, lambda _idx: []

        affine = _invert_reinterpret_view_affine(view)
        if affine is not None:
            return affine
        reindex = _layout_reindexer_to_view(view)
        if reindex is None:
            return None
        return reindex, lambda _idx: []
    if isinstance(view, GenericView):
        return _invert_affine_view(view)
    if isinstance(view, SliceView):
        return _invert_affine_view(view)
    if isinstance(view, ExpandView):
        return _invert_affine_view(view)
    if isinstance(view, PermuteView):
        return same_reorder(view.dims), lambda _idx: []
    if isinstance(view, View):
        return (
            View.dynamic_reshape_indexer(view.get_size(), view.data.get_size()),
            lambda _idx: [],
        )
    return None


def _collect_view_chain(buf: IRNode, target_name: str) -> list[BaseView] | None:
    chain: list[BaseView] = []
    cur: IRNode = buf
    while isinstance(cur, BaseView):
        chain.append(cur)
        cur = cur.data
    if isinstance(cur, IRNode) and cur.get_name() == target_name:
        return chain
    return None


def _get_fx_node_input(node: Node) -> Node | None:
    for arg in node.args:
        if isinstance(arg, Node):
            return arg
    return None


def _find_view_chain_from_fx(nodes: Sequence[object]) -> list[BaseView] | None:
    view_map = getattr(V.graph, "helion_view_node_map", None)
    kernel_nodes = getattr(V.graph, "helion_kernel_fx_nodes", None)
    if not view_map or not kernel_nodes:
        return None

    candidates = _collect_candidate_fx_nodes(nodes)
    if not candidates:
        return None

    memo: dict[Node, list[BaseView] | None] = {}
    visiting: set[Node] = set()

    def _choose_best(
        best: list[BaseView] | None, cand: list[BaseView] | None
    ) -> list[BaseView] | None:
        if cand is None:
            return best
        if best is None or len(cand) > len(best):
            return cand
        return best

    def _dfs(node: Node) -> list[BaseView] | None:
        if node in memo:
            return memo[node]
        if node in visiting:
            return None
        visiting.add(node)
        if node in kernel_nodes:
            result: list[BaseView] | None = []
        elif node.target is operator.getitem:
            nxt = _get_fx_node_input(node)
            result = _dfs(nxt) if isinstance(nxt, Node) else None
        elif node in view_map:
            nxt = _get_fx_node_input(node)
            inner = _dfs(nxt) if isinstance(nxt, Node) else None
            result = [*inner, view_map[node]] if inner is not None else None
        else:
            best: list[BaseView] | None = None
            for arg in node.args:
                if isinstance(arg, Node):
                    best = _choose_best(best, _dfs(arg))
            for arg in node.kwargs.values():
                if isinstance(arg, Node):
                    best = _choose_best(best, _dfs(arg))
            result = best
        visiting.remove(node)
        memo[node] = result
        return result

    best_chain: list[BaseView] | None = None
    for cand in candidates:
        best_chain = _choose_best(best_chain, _dfs(cand))

    return best_chain


def _collect_candidate_fx_nodes(nodes: Sequence[object]) -> list[Node]:
    candidates: list[Node] = []
    for snode in nodes:
        n = snode.node if hasattr(snode, "node") else snode
        if not isinstance(n, IRNode):
            continue
        origin = n.get_origin_node()
        if isinstance(origin, Node) and origin not in candidates:
            candidates.append(origin)
        origins = getattr(n, "origins", None)
        if origins:
            for o in origins:
                if isinstance(o, Node) and o not in candidates:
                    candidates.append(o)
    return candidates


def has_unsafe_views_from_fx_nodes(nodes: Sequence[Node]) -> bool:
    view_map = getattr(V.graph, "helion_view_node_map", None) or {}
    kernel_nodes = getattr(V.graph, "helion_kernel_fx_nodes", None)

    visited: set[Node] = set()

    def _dfs(node: Node) -> bool:
        if node in visited:
            return False
        visited.add(node)
        if kernel_nodes and node in kernel_nodes:
            return False
        if node.target in (
            torch.ops.aten.as_strided.default,
            torch.ops.aten.as_strided_scatter.default,
        ):
            return True
        view = view_map.get(node)
        if view is not None:
            layout = view.get_layout() if hasattr(view, "get_layout") else None
            if layout is not None and _has_overlapping_strides(layout):
                return True
        for arg in node.args:
            if isinstance(arg, Node) and _dfs(arg):
                return True
        for arg in node.kwargs.values():
            if isinstance(arg, Node) and _dfs(arg):
                return True
        return False

    return any(_dfs(node) for node in nodes)


def has_unsafe_views(nodes: Sequence[object]) -> bool:
    candidates = _collect_candidate_fx_nodes(nodes)
    if candidates and has_unsafe_views_from_fx_nodes(candidates):
        return True

    for snode in nodes:
        n = snode.node if hasattr(snode, "node") else snode
        if not isinstance(n, ComputedBuffer) or not isinstance(n.data, Pointwise):
            continue
        for dep in n.data.get_reads():
            if not isinstance(dep, MemoryDep):
                continue
            try:
                buf = V.graph.get_buffer(dep.name)
            except Exception:
                continue
            cur = buf
            while isinstance(cur, BaseView):
                layout = cur.get_layout() if hasattr(cur, "get_layout") else None
                if layout is not None and _has_overlapping_strides(layout):
                    return True
                cur = cur.data

    return False


def _find_view_chain(nodes: Sequence[object], acc_name: str) -> list[BaseView] | None:
    chain = _find_view_chain_from_fx(nodes)
    if chain is not None:
        return chain
    for snode in nodes:
        node = snode.node if hasattr(snode, "node") else snode
        if not isinstance(node, ComputedBuffer) or not isinstance(node.data, Pointwise):
            continue
        for dep in node.data.get_reads():
            if not isinstance(dep, MemoryDep):
                continue
            if dep.name == acc_name:
                return []
            try:
                buf = V.graph.get_buffer(dep.name)
            except Exception:
                continue
            if isinstance(buf, BaseView):
                chain = _collect_view_chain(buf, acc_name)
                if chain is not None:
                    return chain
    return None


def compute_helion_transform(
    nodes: Sequence[object],
    acc_name: str,
    kernel_shape: Sequence[sympy.Expr] | None,
    target_shape: Sequence[sympy.Expr] | None,
    target_layout: object | None,
    kernel_strides: Sequence[sympy.Expr] | None,
) -> HelionIndexingTransform:
    kernel_shape = list(kernel_shape or [])
    target_shape = list(target_shape or [])
    kernel_symbols = [sympy.Symbol(f"idx_{i}") for i in range(len(kernel_shape))]

    if not kernel_shape or not target_shape:
        return HelionIndexingTransform(
            kernel_symbols=kernel_symbols,
            index_exprs=[],
            mod_masks=[],
            broadcast_dims=None,
            unsupported=True,
        )

    if target_layout is not None and _has_overlapping_strides(target_layout):
        return HelionIndexingTransform(
            kernel_symbols=kernel_symbols,
            index_exprs=[],
            mod_masks=[],
            broadcast_dims=None,
            unsupported=True,
        )

    mask_mods: list[sympy.Expr] = []
    chain = _find_view_chain(nodes, acc_name)
    if chain and len(chain) > 1:
        for idx in range(len(chain) - 1):
            if getattr(chain[idx], "data", None) is not chain[idx + 1]:
                chain = [chain[-1]]
                break
    if chain:
        for view in chain:
            layout = view.get_layout() if hasattr(view, "get_layout") else None
            if layout is not None and _has_overlapping_strides(layout):
                return HelionIndexingTransform(
                    kernel_symbols=kernel_symbols,
                    index_exprs=[],
                    mod_masks=[],
                    broadcast_dims=None,
                    unsupported=True,
                )
    sizevars = V.graph.sizevars
    if chain == []:
        shape_mismatch = not sizevars.statically_known_list_equals(
            target_shape, kernel_shape
        )
        stride_mismatch = False
        if kernel_strides is not None and target_layout is not None and hasattr(
            target_layout, "stride"
        ):
            stride_mismatch = not sizevars.statically_known_list_equals(
                list(target_layout.stride), list(kernel_strides)
            )
        if shape_mismatch or stride_mismatch:
            chain = None
    reindex: Callable[[list[sympy.Expr]], list[sympy.Expr]] | None = None
    if chain is not None:
        reindex = _identity_reindexer
        for view in reversed(chain):
            inverse = _inverse_view_reindexer(view)
            if inverse is None:
                reindex = None
                break
            inv_reindex, mask_builder = inverse
            input_exprs = list(reindex(kernel_symbols))
            mask_mods.extend(mask_builder(input_exprs))
            reindex = fuse_reindexing(inv_reindex, reindex)

    if chain is None or reindex is None:
        mask_mods = []
        if (
            sizevars.statically_known_equals(
                sympy_product(target_shape), sympy_product(kernel_shape)
            )
            and target_layout is not None
        ):
            reindex = _layout_reindexer_from_layout(target_layout, kernel_shape)
            if reindex is None:
                return HelionIndexingTransform(
                    kernel_symbols=kernel_symbols,
                    index_exprs=[],
                    mod_masks=[],
                    broadcast_dims=None,
                    unsupported=True,
                )
        elif sizevars.statically_known_list_equals(target_shape, kernel_shape):
            reindex = _identity_reindexer
        else:
            return HelionIndexingTransform(
                kernel_symbols=kernel_symbols,
                index_exprs=[],
                mod_masks=[],
                broadcast_dims=None,
                unsupported=True,
            )

    index_exprs = list(reindex(kernel_symbols))
    broadcast_dims: list[tuple[int, int]] = []
    kernel_sym_set = set(kernel_symbols)
    for dim, (expr, size) in enumerate(zip(index_exprs, target_shape, strict=True)):
        if expr.free_symbols.isdisjoint(kernel_sym_set):
            if isinstance(size, sympy.Integer):
                size_int = int(size)
            elif isinstance(size, int):
                size_int = size
            else:
                return HelionIndexingTransform(
                    kernel_symbols=kernel_symbols,
                    index_exprs=[],
                    mod_masks=[],
                    broadcast_dims=None,
                    unsupported=True,
                )
            if size_int > 1:
                broadcast_dims.append((dim, size_int))

    if broadcast_dims:
        return HelionIndexingTransform(
            kernel_symbols=kernel_symbols,
            index_exprs=index_exprs,
            mod_masks=mask_mods,
            broadcast_dims=broadcast_dims,
            unsupported=False,
        )

    return HelionIndexingTransform(
        kernel_symbols=kernel_symbols,
        index_exprs=index_exprs,
        mod_masks=mask_mods,
        broadcast_dims=None,
        unsupported=False,
    )
