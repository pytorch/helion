"""A pad and its transposes: compute the value into the pad, copy the rest.

The CE backward of a causal LM whose vocabulary is padded to a multiple of 8
(better-benchmark's sum_81b4fd73f8d1: Bart, Pegasus, TrOCR; sum_6119d597d188:
PLBart) returns its bf16 [16384, 50265] gradient z padded twice: as it is,
for the hidden states' gradient, and transposed, for the lm head weight's:

    pad = F.pad(z, (0, 7))            # [16384, 50272]
    pad_t = F.pad(z.t(), (0, 0, 0, 7))  # [50272, 16384], pad.t() as values

Inductor stores z, a pointwise value read by two ops that reads more than
realize_reads_threshold buffers (StorageBox.mark_reuse()), and pads it in a
second kernel that writes both outputs:

    def kernel(in0, ..., in6, out0):             # CE backward
        for tile_x in hl.tile(16384):
            ...
            out0[tile_x, :] = tmp31.to(torch.bfloat16)  # an rdim of 50265

    def kernel(in0, out0, out1):                 # both pads
        for tile_y, tile_x in hl.tile([16384, 50272]):
            tmp5 = hl.load(in0, [tile_y, tile_x], extra_mask=tmp4)
            ...
            out0[tile_y, tile_x] = tmp13.to(torch.bfloat16)  # transposed
            out1[tile_y, tile_x] = tmp13.to(torch.bfloat16)

six passes over 1.65 GB; an rdim of 50265 bf16 is not a multiple of 16 bytes,
which keeps the CE kernel's stores narrow. Lowered
here, the pad computes z itself (Inductor lowers it as a cat with its fill,
whose slice z is computed into, _pad_as_cat()), so the CE kernel writes the
padded rdim of 50272, and the transposed pad copies it:

    def kernel(in0, ..., in6, out0):             # CE backward into the pad
        for tile_x in hl.tile(16384):
            ...
            out0[tile_x, :] = tmp31.to(torch.bfloat16)  # an rdim of 50272

    def kernel(in0, out0):                       # the transposed copy
        for tile_y, tile_x in hl.tile([50272, 16384]):
            tmp0 = in0[tile_y, tile_x].to(torch.float32)  # in0: pad.t()
            out0[tile_y, tile_x] = tmp0.to(torch.bfloat16)

beside a kernel that writes the fill: five passes. Computing both outputs
in the CE kernel instead (four passes) is slower: a kernel that reduces
over r for each x stores the transposed one an element per r. The stored z has no
reader left, and the scheduler drops it.

The values do not change: the pad computes the ops that stored z, in the
same order on the same inputs, and is stored, and the transposes copy it.
Declines (the pads then read the stored value, as before): a value that
another op than the pads reads; pads that are not one pad and its
transposes (other paddings, other fills, a fill of the other sign,
permutations); a value Inductor did not store, or stored for another
reason (a reduction, an extern kernel, a graph input, a pointwise too large
to inline) or written into by its own lowering (scatter); a value an op
writes in place (index_put_), or an op from it to a pad with effects or a
higher-order op (a user Triton kernel's write in place); a value whose
inputs, or a pad, something wrote into before the pads lower, through any
buffer sharing their storage (V.graph.mutated_buffers); and sizes that are
not constants. The group decides once, at whichever pad Inductor lowers
first, so that it never declines at one pad and fires at another (which
would compute the value twice), and its pad is never padded in place
(config.inplace_padding, for a pad only a matmul reads), which would
realize the recomputed value at a padded stride and write the fill into
its buffer, a write each transpose would decline at."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import weakref

import torch
from torch._inductor import config as inductor_config
from torch._inductor import ir
from torch._inductor import lowering as inductor_lowering
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

aten = torch.ops.aten

PAD = aten.constant_pad_nd.default
PERMUTE = aten.permute.default

# The views between the stored value and the padded tensor: each lowers to
# a view of its input (or of the recomputed value) and reads nothing else.
VIEWS = frozenset(
    {
        aten.view.default,
        aten._unsafe_view.default,
        aten.reshape.default,
        aten.squeeze.dim,
        aten.squeeze.dims,
        aten.unsqueeze.default,
    }
)


def _static_view_args(view: torch.fx.Node) -> list[object] | None:
    """The arguments after the input that lower `view` again: its shape
    for a view or reshape (whose size arguments may be graph inputs that
    Inductor has freed by now), its other arguments for the rest; None
    when one is not a constant (dynamic shapes, keyword arguments)."""
    if view.kwargs:
        return None
    if view.target in (
        aten.view.default,
        aten._unsafe_view.default,
        aten.reshape.default,
    ):
        value = view.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return None
        args: list[object] = [list(value.shape)]
    else:
        args = list(view.args[1:])
    flat = [
        a for arg in args for a in (arg if isinstance(arg, (list, tuple)) else [arg])
    ]
    if not all(type(a) is int for a in flat):
        return None
    return args


def _reads_only_input(user: torch.fx.Node, node: torch.fx.Node) -> bool:
    """Whether `user` is a call that takes `node` as its first argument and
    nowhere else."""
    return (
        user.op == "call_function"
        and bool(user.args)
        and user.args[0] is node
        and node not in user.args[1:]
        and node not in user.kwargs.values()
    )


def _pad_args(pad: torch.fx.Node) -> tuple[list[tuple[int, int]], object] | None:
    """The (low, high) padding of each dim of `pad`'s input, outermost first,
    and its fill; None if the padding is not constant."""
    padding = pad.args[1] if len(pad.args) > 1 else pad.kwargs.get("pad")
    value = pad.args[2] if len(pad.args) > 2 else pad.kwargs.get("value", 0)
    source = pad.args[0]
    if not (
        isinstance(padding, (list, tuple))
        and all(type(p) is int for p in padding)
        and len(padding) % 2 == 0
        and isinstance(source, torch.fx.Node)
        and isinstance(meta := source.meta.get("val"), torch.Tensor)
        and len(padding) // 2 <= meta.dim()
    ):
        return None
    ndim = meta.dim()
    ints = cast("list[int]", list(padding))
    pairs = [(0, 0)] * ndim
    for i in range(len(ints) // 2):
        pairs[ndim - 1 - i] = (ints[2 * i], ints[2 * i + 1])
    return pairs, value


class TransposedPads:
    """The pads of one tensor z (`base`): `pad` = constant_pad_nd(z) and
    `transposes`, each constant_pad_nd(permute(z, perm)) with the padding and
    fill of `pad` permuted, i.e. permute(`pad`, perm) as values; `source`,
    the pointwise value z views (`views`, in graph order)."""

    def __init__(
        self,
        source: torch.fx.Node,
        views: list[torch.fx.Node],
        pad: torch.fx.Node,
        transposes: dict[torch.fx.Node, list[int]],
    ) -> None:
        self.source = source
        self.views = views
        self.pad = pad
        self.transposes = transposes

    @staticmethod
    def find(node: torch.fx.Node) -> TransposedPads | None:
        """The TransposedPads that the pad `node` belongs to, when every op
        that reads its value (and the views of it) is that pad or one of its
        transposes, and no op between the value and them writes into a
        tensor; None otherwise."""
        base = node.args[0]
        if not isinstance(base, torch.fx.Node):
            return None
        if base.op == "call_function" and base.target is PERMUTE:
            base = base.args[0]
            if not isinstance(base, torch.fx.Node):
                return None
        pads: list[torch.fx.Node] = []
        permuted: dict[torch.fx.Node, list[int]] = {}
        for user in base.users:
            if not _reads_only_input(user, base):
                return None
            if user.target is PAD:
                pads.append(user)
            elif user.target is PERMUTE and len(user.users) == 1:
                (pad_t,) = user.users
                perm = user.args[1]
                if not (
                    pad_t.target is PAD
                    and _reads_only_input(pad_t, user)
                    and isinstance(perm, (list, tuple))
                    and all(type(d) is int for d in perm)
                ):
                    return None
                permuted[pad_t] = cast("list[int]", list(perm))
            else:
                return None
        if len(pads) != 1 or not permuted:
            return None
        (pad,) = pads
        found = _pad_args(pad)
        if found is None:
            return None
        pairs, value = found
        for pad_t, perm in permuted.items():
            found_t = _pad_args(pad_t)
            if found_t is None or not _same_fill(found_t[1], value):
                return None
            if found_t[0] != [pairs[d % len(pairs)] for d in perm]:
                return None
        # z: the views of one pointwise value that only z reads.
        source, views = base, []
        while source.op == "call_function" and source.target in VIEWS:
            views.append(source)
            parent = source.args[0]
            if not (
                isinstance(parent, torch.fx.Node)
                and len(parent.users) == 1
                and _reads_only_input(source, parent)
            ):
                return None
            source = parent
        if source.op != "call_function":
            return None
        # fx nodes compare by their order in the graph.
        last = max(pad, *permuted)
        if _writes_between(source, last):
            return None
        return TransposedPads(source, views[::-1], pad, permuted)


def _writes_between(source: torch.fx.Node, last: torch.fx.Node) -> bool:
    """Whether `source` or a node after it, up to `last` (in graph order),
    has effects (Node.is_impure()) or calls a higher-order op: writes into
    one of its arguments, as the index_put_ that Inductor's reinplacing makes
    of an index_put, or a user Triton kernel's write in place
    (triton_kernel_wrapper_mutation, which Node.is_impure() does not report),
    whose stored value the lowering would recompute without the write."""
    node = source
    while node is not last and node.op != "root":
        if node.op == "call_function" and (
            isinstance(node.target, torch._ops.HigherOrderOperator)
            or node.is_impure(impure_random=False)
        ):
            return True
        node = node.next
    return node is not last


def _aliased(names: Iterable[str]) -> OrderedSet[str]:
    """`names` and the buffers each one aliases, transitively: the inputs an
    extern kernel's output aliases (a view of another dtype width) and the
    buffer a NonOwningLayout views (a value computed into a cat's part).
    V.graph.mutated_buffers holds the name a write went to, and a write to
    one of these is a write to the others."""
    seen: OrderedSet[str] = OrderedSet()
    stack = list(names)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        buffer = V.graph.try_get_buffer(name)
        if not isinstance(buffer, ir.Buffer):
            continue
        op = buffer.get_defining_op()
        if isinstance(op, ir.ExternKernel):
            stack.extend(op.get_inputs_that_alias_output())
        # A buffer of no layout (an in-place op's MutationOutput, a tuple
        # output's parent) views nothing.
        layout = buffer.maybe_get_layout()
        if isinstance(layout, ir.NonOwningLayout):
            stack.append(layout.view.get_name())
    return seen


def _written(names: Iterable[str]) -> bool:
    """Whether a buffer of `names`, or one it shares storage with, is in
    V.graph.mutated_buffers (_aliased())."""
    return bool(_aliased(names) & _aliased(V.graph.mutated_buffers))


def _same_fill(a: object, b: object) -> bool:
    """Whether two pads' fills are the same value, -0.0 and 0.0 apart."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return type(a) is type(b) and a == b


def _stored_pointwise(x: object, source: torch.fx.Node) -> ir.Pointwise | None:
    """The Pointwise that lowering `source` returned, when Inductor stored it
    and `x`, the lowered input of a pad, views that storage (a view that
    copied, to make its input contiguous, stores a copy instead); None for
    any other storage, a Pointwise too large to inline, which Inductor
    stores lest inlining it overflow Python's recursion
    (GraphLowering.run_node()), or a stored buffer a lowering writes into
    (V.graph.mutated_buffers)."""
    storage: object = x.data if isinstance(x, ir.TensorBox) else None
    while isinstance(storage, ir.BaseView):
        storage = storage.data
    if not (
        isinstance(storage, ir.StorageBox)
        and isinstance(stored := storage.data, ir.ComputedBuffer)
        and type(pointwise := stored.data) is ir.Pointwise
        and pointwise.origin_node is source
        and not pointwise.has_large_inner_fn(threshold=100)
        # A lowering that writes into the buffer it stored: scatter,
        # index_put and the other functional scatters store a copy of their
        # input (the Pointwise) and then write into it, which the Pointwise
        # does not compute.
        and stored.get_name() not in V.graph.mutated_buffers
    ):
        return None
    return pointwise


# The pad each graph lowered for a TransposedPads, by its pad node, so that
# whichever of the pads Inductor lowers first computes it for all of them.
_LOWERED_PADS: weakref.WeakKeyDictionary[object, dict[torch.fx.Node, object]] = (
    weakref.WeakKeyDictionary()
)


def _pad_of_value(
    x: object, pads: TransposedPads, replaced: Callable[..., Any] | None
) -> object:
    """The pad of `pads`, its value recomputed into it and stored, for
    transposed_pad(); None where the lowering declines (_stored_pointwise(),
    a write since to a buffer the value reads, a view of no static sizes)."""
    pointwise = _stored_pointwise(x, pads.source)
    # A buffer the value reads that something wrote since (a user Triton
    # kernel's in-place write, which no FX effect marks): recomputed after
    # it, the value would read the new data.
    if pointwise is None or _written(pointwise.get_read_names()):
        return None
    value: Any = ir.Pointwise.create(
        device=pointwise.device,
        dtype=pointwise.dtype,
        inner_fn=pointwise.inner_fn,
        ranges=pointwise.ranges,
    )
    for view in pads.views:
        view_args = _static_view_args(view)
        if view_args is None:
            return None
        value = inductor_lowering.lowerings[view.target](value, *view_args)
    # Lowered as its own node (the transpose may come first in the graph):
    # cat() asks the current node about its users and inputs. Not padded in
    # place (config.inplace_padding: a pad that only a matmul reads), which
    # would realize the value at a padded stride and write its fill into it,
    # a write each transpose would then decline at.
    with (
        V.graph.set_current_node(pads.pad),
        V.set_current_node(pads.pad),
        inductor_config.patch(inplace_padding=False),
    ):
        args, kwargs = pads.pad.args[1:], pads.pad.kwargs
        padded = replaced(value, *args, **kwargs) if replaced else None
        if padded is None:
            padded = inductor_lowering.lowerings[PAD](value, *args, **kwargs)
        # The transposes copy the stored pad rather than inline the value.
        padded.realize()
    return padded


def transposed_pad(x: object, replaced: Callable[..., Any] | None = None) -> object:
    """The lowering of the pad V.graph.current_node lowers, with
    cuda_backend "helion" on a CUDA device, when it belongs to
    TransposedPads.find() and Inductor stored the value it pads: the pad of
    the value recomputed, through `replaced` (the lowering of
    aten.constant_pad_nd this one replaced, a user's) or, where that returns
    None, Inductor's, and stored (a pad other than one right pad comes back
    unrealized), or the copy of its transpose; None otherwise."""
    node = V.graph.current_node
    if not (
        inductor_config.cuda_backend == "helion"
        and isinstance(x, ir.TensorBox)
        and (device := x.get_device()) is not None
        and device.type == "cuda"
        and node is not None
        and node.target is PAD
        and (pads := TransposedPads.find(node)) is not None
    ):
        return None
    # Decided once per group, at whichever pad Inductor lowers first, a
    # decline included: a group that declines at its plain pad (a value
    # Inductor had not stored, which cat() then computes into the pad) and
    # fired at a transpose would compute the value twice.
    lowered = _LOWERED_PADS.setdefault(V.graph, {})
    if pads.pad not in lowered:
        lowered[pads.pad] = _pad_of_value(x, pads, replaced)
    padded = lowered[pads.pad]
    if padded is None:
        return None
    if node is pads.pad:
        return padded
    if _written([cast("ir.TensorBox", padded).get_name()]):
        # Written since: the transpose pads the stored value itself.
        return None
    perm = pads.transposes[node]
    return inductor_lowering.clone(inductor_lowering.permute(padded, perm))


def lowering_of_pads(replaced: Callable[..., Any] | None) -> Callable[..., Any]:
    """The user lowering of aten.constant_pad_nd: transposed_pad(), else
    the replaced entry (a user's lowering, or None for Inductor's)."""

    def lowering(x: object, *args: object, **kwargs: object) -> object:
        out = transposed_pad(x, replaced)
        if out is None and replaced is not None:
            out = replaced(x, *args, **kwargs)
        return out

    return lowering


def register_transposed_pads_lowering() -> None:
    """Register lowering_of_pads() for aten.constant_pad_nd, once:
    registering again (helion.inductor._register() runs again in tests)
    keeps the one wrapper."""
    current = inductor_lowering.user_lowerings.get(PAD)
    if getattr(current, _TRANSPOSED_PADS_WRAPPER, False):
        return
    lowering = lowering_of_pads(current)
    setattr(lowering, _TRANSPOSED_PADS_WRAPPER, True)
    inductor_lowering.user_lowerings[PAD] = lowering


# Marks the user lowering register_transposed_pads_lowering() installed.
_TRANSPOSED_PADS_WRAPPER = "_helion_transposed_pads_lowering"
