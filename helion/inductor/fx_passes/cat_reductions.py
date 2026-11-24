"""The rewrite of reductions and slices over a concatenated dim, which
HelionPostGradPass applies. Softmax with attention sinks (GPT-OSS) appends
one sink to the scores' last dim, the rdim, normalizes rdims of 1001 and
drops the sink:

    x = torch.cat([scores, sinks.expand(64, 1000, 1)], -1)
    x = x - x.amax(-1, keepdim=True)
    probs = torch.softmax(x.float(), -1)[..., :-1].contiguous()

Inductor lowers the cat to a select between masked loads of its parts, so
the kernel that reduces rdims of 1001 reads the scores masked past 1000 and
selects the sink in at r = 1000:

    rdim = hl.zeros([tile_x, 1001]).size(1)
    tmp5 = hl.load(in0, [tile_y, tile_x, slice(None)], extra_mask=tmp4)
    ...
    tmp19 = torch.where(tmp4, tmp11, tmp18)
    tmp20 = torch.amax(tmp19, -1)

Every op between the cat and the reductions is pointwise, so each reduction
over the cat dim combines the reductions of the parts, amax(cat([a, b])) =
maximum(amax(a), amax(b)), and the slice that drops the sinks is the scores'
part of the result. Rewritten so, the kernel reduces the scores' rdims of
1000, read as tiles, and combines one sink per rdim with each reduction:

    tmp0 = in0[tile_y, tile_x, :].to(torch.float32)
    ...
    tmp5 = torch.amax(tmp4, -1)
    tmp8 = torch.maximum(tmp5, tmp7)"""

from __future__ import annotations

import itertools
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch.utils import _pytree as pytree

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Container

    from torch.fx.node import Argument

aten = torch.ops.aten

# The most parts a cat may have: each pointwise op between it and its
# reductions becomes one op per part.
MAX_PARTS = 4


def split_cat_reductions(
    graph: torch.fx.Graph, devices: Container[str] = ("cuda",)
) -> bool:
    """Rewrites the reductions and slices over the dim of each cat of a tensor
    on `devices` (the CUDA tensors Helion's kernels compute) that
    CatRegion.find() accepts; returns whether it rewrote one."""
    changed = False
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target is aten.cat.default
            and isinstance(value := node.meta.get("val"), torch.Tensor)
            and value.device.type in devices
        ):
            region = CatRegion.find(graph, node)
            if region is not None:
                region.rewrite()
                changed = True
    return changed


def target(node: torch.fx.Node) -> Callable[..., Any]:
    """The op of a call_function node."""
    return cast("Callable[..., Any]", node.target)


def needs_cat_storage(node: torch.fx.Node) -> bool:
    """Whether the slice `node` must stay a view of the cat's storage, where a
    part has its own: when what reads it can reach the storage past it,
    as_strided or as_strided_scatter, directly or through views of `node`, as
    in cat([a, b], 1)[:, :32].t().as_strided(...), or the caller, when `node`
    or a view of it is a graph output, which the caller may view past the
    slice as well, out.as_strided((4, 8), (40, 1), 32) (a slice consumed by an
    op that writes a new tensor, as .contiguous() does, is not reached past);
    and when an op writes into it or a view of it, x[:, :32].add_(1), a write
    the cat's other readers must see. The views of an op with several (split,
    unbind, chunk) are the getitems of its result."""
    views = [node]
    while views:
        for user in views.pop().users:
            if (
                user.op == "output"
                or user.target
                in (aten.as_strided.default, aten.as_strided_scatter.default)
                or writes_operand(user)
            ):
                return True
            # aten._unsafe_view aliases its input too, though neither is_view
            # nor its schema says so.
            if (
                user.target in (operator.getitem, aten._unsafe_view.default)
                or isinstance(user.target, torch._ops.OpOverload)
                and user.target.is_view
            ):
                views.append(user)
    return False


def writes_operand(node: torch.fx.Node) -> bool:
    """Whether `node` writes into one of its operands (add_, copy_)."""
    target = node.target
    return isinstance(target, torch._ops.OpOverload) and any(
        argument.alias_info is not None and argument.alias_info.is_write
        for argument in target._schema.arguments
    )


def pointwise(node: torch.fx.Node) -> bool:
    """Whether `node` is a pointwise op that writes a new tensor: not one
    that mutates its input (add_), which split into one op per part would
    mutate the parts, nor one that returns its input (positive), whose slice
    is a view of the cat that the rewrite would make a part's own tensor."""
    target = node.target
    return node.op == "call_function" and (
        target is torch.ops.prims.convert_element_type.default
        or (
            isinstance(target, torch._ops.OpOverload)
            and torch.Tag.pointwise in target.tags
            and not target._schema.is_mutable
            and all(ret.alias_info is None for ret in target._schema.returns)
        )
    )


class CatRegion:
    """A cat along `dim` of `parts`, the pointwise ops of it (`nodes`, each a
    cat of its ops on the parts), and the reductions over `dim` and slices
    along it that read them (`exits`)."""

    def __init__(
        self,
        graph: torch.fx.Graph,
        cat: torch.fx.Node,
        parts: list[torch.fx.Node],
        dim: int,
    ) -> None:
        self.graph = graph
        self.cat = cat
        self.parts = parts
        self.dim = dim
        value = cat.meta["val"]
        self.rank = value.dim()
        self.fake_mode = value.fake_mode
        self.sizes = [part.meta["val"].shape[dim] for part in parts]
        # Where each part starts along the dim, and the cat's end.
        self.offsets = [0, *itertools.accumulate(self.sizes)]
        self.total = value.shape[dim]
        self.nodes: list[torch.fx.Node] = [cat]
        self.exits: list[torch.fx.Node] = []
        self.created: list[torch.fx.Node] = []

    @classmethod
    def find(cls, graph: torch.fx.Graph, cat: torch.fx.Node) -> CatRegion | None:
        """The region of `cat` if every use of it and of its pointwise ops is
        another such op or an exit; None if any is not (as cumsum(x, dim)),
        or a value of the region is a graph output, or the cat has a dynamic
        size or more than MAX_PARTS parts."""
        parts = cat.args[0] if cat.args else cat.kwargs.get("tensors")
        dim = cat.args[1] if len(cat.args) > 1 else cat.kwargs.get("dim", 0)
        value = cat.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(parts, (list, tuple))
            or not 2 <= len(parts) <= MAX_PARTS
            or not all(
                isinstance(part, torch.fx.Node)
                and isinstance(part.meta.get("val"), torch.Tensor)
                and part.meta["val"].dim() == value.dim()
                for part in parts
            )
            or not isinstance(dim, int)
            or not all(isinstance(n, int) for n in value.shape)
        ):
            return None
        region = cls(
            graph, cat, cast("list[torch.fx.Node]", list(parts)), dim % value.dim()
        )
        # A cat none of whose parts is thin, a quarter of the dim or less, keeps
        # its one reduction. Lowering the parts apart pays for a thin part, the
        # sink an attention-sink softmax cats to rdims of 1000 (whose
        # masked loads cost every lane of the rdim), and costs where the parts
        # are alike: cat([a(8, 2100), b(8, 900)]).sum(-1) as sum(a) + sum(b) is
        # two kernels over rdims of different lengths and a combine, for loads
        # the masked cat reads in one. The quarter is a cutoff between the two,
        # not a measured break-even.
        if 0 in region.sizes or 4 * min(region.sizes) > region.total:
            return None
        members = {cat}
        # The uses of the region not yet seen: the scan ends with the last.
        pending = set(cat.users)
        node = cat
        while pending and (node := node.next).op != "root":
            if node not in pending:
                continue
            pending.discard(node)
            uses = [n for n in node.all_input_nodes if n in members]
            if node.op == "output":
                return None
            if region.is_member(node, members):
                members.add(node)
                region.nodes.append(node)
                pending.update(node.users)
            elif node.args and uses == [node.args[0]] and region.is_exit(node):
                region.exits.append(node)
            else:
                return None
        return region if region.exits else None

    def is_member(self, node: torch.fx.Node, members: set[torch.fx.Node]) -> bool:
        """Whether `node` is pointwise, with the cat's shape along `dim`, and
        each tensor it reads is of the region, broadcast along `dim`, or of the
        cat's size there (read as slices, one per part)."""
        value = node.meta.get("val")
        if (
            not pointwise(node)
            or not isinstance(value, torch.Tensor)
            or value.dim() != self.rank
            or value.shape[self.dim] != self.total
        ):
            return False
        return all(
            arg in members or self.broadcast(arg) is not None
            for arg in node.all_input_nodes
        )

    def position(self, node: torch.fx.Node) -> int:
        """The dim of `node` that broadcasting aligns with the cat's, negative
        if its rank does not reach it."""
        return node.meta["val"].dim() - (self.rank - self.dim)

    def broadcast(self, node: torch.fx.Node) -> bool | None:
        """True if `node` broadcasts along the cat's dim (size 1 there, or a
        rank that does not reach it), False if it has the cat's size there,
        None otherwise."""
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return None
        position = self.position(node)
        if position < 0 or value.shape[position] == 1:
            return True
        return False if value.shape[position] == self.total else None

    def is_exit(self, node: torch.fx.Node) -> bool:
        """Whether `node` is an amax, sum or max.dim (of which only the values
        are read) over the cat's dim, or a slice along it that keeps whole
        parts of a contiguous value and need not stay a view of the cat
        (needs_cat_storage()). The slice of a value with another layout, as
        c + cat([a, b], -1) for a transposed c, which takes c's strides, or a
        cat of channels-last parts, has views (its transpose's view(-1)) that
        the contiguous copy replacing it would not have."""
        dim, args = self.dim, node.args
        if node.target in (aten.amax.default, aten.sum.dim_IntList):
            dims = args[1] if len(args) > 1 else node.kwargs.get("dim")
            if node.target is aten.sum.dim_IntList and node.kwargs.get("dtype"):
                return False
            return isinstance(dims, (list, tuple)) and [
                d % self.rank for d in dims
            ] == [dim]
        if node.target is aten.max.dim:
            max_dim = args[1] if len(args) > 1 else node.kwargs.get("dim")
            return (
                isinstance(max_dim, int)
                and max_dim % self.rank == dim
                and all(
                    user.target is operator.getitem and user.args[1] == 0
                    for user in node.users
                )
            )
        if node.target is aten.slice.Tensor:
            source = node.args[0]
            return (
                self.sliced_parts(node) is not None
                and isinstance(source, torch.fx.Node)
                and source.meta["val"].is_contiguous()
                and not needs_cat_storage(node)
            )
        return False

    def sliced_parts(self, node: torch.fx.Node) -> range | None:
        """The parts a slice along the cat's dim keeps, if they are whole."""
        args, kwargs = node.args, node.kwargs
        slice_dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        start = args[2] if len(args) > 2 else kwargs.get("start")
        end = args[3] if len(args) > 3 else kwargs.get("end")
        step = args[4] if len(args) > 4 else kwargs.get("step", 1)
        if (
            not isinstance(slice_dim, int)
            or slice_dim % self.rank != self.dim
            or step != 1
            or not isinstance(start, (int, type(None)))
            or not isinstance(end, (int, type(None)))
        ):
            return None
        start, end, _ = slice(start, end).indices(self.total)
        offsets = self.offsets
        if start not in offsets or end not in offsets or start >= end:
            return None
        return range(offsets.index(start), offsets.index(end))

    def call(
        self,
        before: torch.fx.Node,
        target: Callable[..., Any],
        /,
        *args: Argument,
        **kwargs: Argument,
    ) -> torch.fx.Node:
        with self.graph.inserting_before(before):
            new = self.graph.call_function(target, args, kwargs)
        self.created.append(new)
        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.fx.Node, lambda n: n.meta["val"], (args, kwargs)
        )
        with self.fake_mode:
            new.meta["val"] = target(*fake_args, **fake_kwargs)
        return new

    def rewrite(self) -> None:
        """Each op of the region as one op per part, each exit as the
        combination of its parts, then the dead ops erased."""
        # cat promotes its parts to one dtype, which the parts' ops keep.
        dtype = self.cat.meta["val"].dtype
        parts: dict[torch.fx.Node, list[torch.fx.Node]] = {
            self.cat: [
                part
                if part.meta["val"].dtype == dtype
                else self.call(
                    self.cat, torch.ops.prims.convert_element_type.default, part, dtype
                )
                for part in self.parts
            ]
        }
        offsets = self.offsets
        for node in self.nodes[1:]:

            def part_of(arg: object, i: int, node: torch.fx.Node = node) -> object:
                if not isinstance(arg, torch.fx.Node):
                    return arg
                if arg in parts:
                    return parts[arg][i]
                if self.broadcast(arg):
                    return arg
                # A tensor of the cat's size along the dim: its slice, along
                # its own dim there (a bias of [1001] beside [b, h, q, 1001]).
                return self.call(
                    node,
                    aten.slice.Tensor,
                    arg,
                    self.position(arg),
                    offsets[i],
                    offsets[i + 1],
                )

            parts[node] = [
                self.call(
                    node,
                    target(node),
                    *pytree.tree_map(lambda a, i=i: part_of(a, i), node.args),
                    **pytree.tree_map(lambda a, i=i: part_of(a, i), node.kwargs),
                )
                for i in range(len(self.parts))
            ]
        for node in self.exits:
            source = parts[node.args[0]]  # pyrefly: ignore [bad-index]
            if node.target is aten.slice.Tensor:
                kept = [source[i] for i in self.sliced_parts(node) or ()]
                contiguous = {"memory_format": torch.contiguous_format}
                if len(kept) > 1:
                    result = self.call(node, aten.cat.default, kept, self.dim)
                elif kept[0] in self.created and kept[0].meta["val"].is_contiguous():
                    # An op of the part the rewrite made, new storage as the
                    # slice of the cat is.
                    result = kept[0]
                else:
                    # A part of the cat itself, which can be an input, a view
                    # of one or another output, or an op of a part that is not
                    # contiguous: the slice of the cat is new storage, which a
                    # contiguous copy is (one that Inductor fuses). Any view of
                    # the slice is one of a contiguous tensor (is_exit() takes
                    # only a slice of a contiguous value), where it need not be
                    # one of the part: the slice of
                    # cat([a.permute(1, 0, 2), b], -1) to a's part of the last
                    # dim, reshaped to merge its first two dims, would be no
                    # view of a's permuted copy.
                    result = self.call(node, aten.clone.default, kept[0], **contiguous)
                node.replace_all_uses_with(result)
                continue
            sum_dtype = (
                node.meta["val"].dtype if node.target is aten.sum.dim_IntList else None
            )
            if node.target is aten.max.dim:
                keepdim = (
                    node.args[2]
                    if len(node.args) > 2
                    else node.kwargs.get("keepdim", False)
                )
                reduced = [
                    self.call(node, aten.amax.default, part, [self.dim], keepdim)
                    for part in source
                ]
            elif sum_dtype in (torch.float16, torch.bfloat16):
                # The parts' sums in float32, rounded once, as the sum of the
                # cat accumulates its rdim.
                reduced = [
                    self.call(
                        node,
                        target(node),
                        part,
                        *node.args[1:],
                        **{**node.kwargs, "dtype": torch.float32},
                    )
                    for part in source
                ]
            else:
                reduced = [
                    self.call(node, target(node), part, *node.args[1:], **node.kwargs)
                    for part in source
                ]
            # A sum adds the parts' sums: it reassociates the rdim's sum, as any
            # split of a reduction does (Inductor's split partials, a tree's
            # order), so an rdim of 1e38 and -1e38 in two parts sums to inf - inf,
            # nan, where an order that pairs them gives 0 and a sequential one
            # inf.
            combine = aten.maximum.default if sum_dtype is None else aten.add.Tensor
            result = reduced[0]
            for other in reduced[1:]:
                result = self.call(node, combine, result, other)
            if sum_dtype is not None and result.meta["val"].dtype != sum_dtype:
                result = self.call(
                    node,
                    torch.ops.prims.convert_element_type.default,
                    result,
                    sum_dtype,
                )
            if node.target is aten.max.dim:
                for user in list(node.users):
                    user.replace_all_uses_with(result)
                    self.graph.erase_node(user)
            else:
                node.replace_all_uses_with(result)
        # The ops of the parts that only fed the parts a slice drops, as the
        # sinks of a softmax.
        for node in reversed([*self.nodes, *self.exits, *self.created]):
            if not node.users:
                self.graph.erase_node(node)
