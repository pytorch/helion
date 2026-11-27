"""helion.inductor's lowerings of aten.mm and aten.bmm, which follow the
backend's plan (HelionChoices.transpose_matmul() and the other matmul questions):
a transposed product, copies of operands, and pad_mm's zero pad moved to the
output."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import Any

import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor import ir
from torch._inductor.lowering import lowerings
from torch._inductor.lowering import make_pointwise
from torch._inductor.lowering import user_lowerings
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops

from ..choices import HelionChoices
from ..fx_passes.cat_reductions import STORAGE_OPS
from ..fx_passes.cat_reductions import UNMARKED_VIEWS

if TYPE_CHECKING:
    from collections.abc import Callable

aten = torch.ops.aten


def _addresses_storage(target: torch._ops.OpOverload) -> bool:
    """Whether the ATen or prims op `target` reaches a tensor's storage rather
    than its elements (the cat pass's STORAGE_OPS)."""
    return target.overloadpacket in STORAGE_OPS


def _returns_alias(
    target: torch._ops.OpOverload, user: torch.fx.Node, view: torch.fx.Node
) -> bool:
    """Whether `user`, a call of the ATen or prims op `target`, returns `view`
    or a view of it: the schema marks the argument aliased, or the op is one
    of the cat pass's UNMARKED_VIEWS."""
    if target.overloadpacket in UNMARKED_VIEWS:
        return True
    arguments = target._schema.arguments
    names = [argument.name for argument in arguments]
    bound = dict(zip(names, user.args, strict=False)) | user.kwargs
    return any(
        bound.get(argument.name) is view and argument.alias_info is not None
        for argument in arguments
    )


def _read_past(node: torch.fx.Node) -> bool:
    """Whether something can read the storage behind `node`, a slice of the
    padded product, past it into the pad: the graph's output, an op other
    than ATen's or prims' (a custom or higher-order op), or one that addresses
    storage (an as_strided), reached through `node` or views of it."""
    views = [node]
    while views:
        view = views.pop()
        for user in view.users:
            target = user.target
            if user.op == "output":
                return True
            if target is operator.getitem:
                views.append(user)
            elif not (
                isinstance(target, torch._ops.OpOverload)
                and target.namespace in ("aten", "prims")
            ) or _addresses_storage(target):
                return True
            elif _returns_alias(target, user, view):
                views.append(user)
    return False


def _slices_keep(user: torch.fx.Node, dim: int, keep: int, size: int) -> bool:
    """Whether `user` is an aten.slice.Tensor of a `size`-long `dim` that
    keeps only its first `keep` elements."""
    if user.target is not aten.slice.Tensor:
        return False
    args = [*user.args[1:4], None, None, None][:3]
    sliced = user.kwargs.get("dim", args[0] if args[0] is not None else 0)
    start = user.kwargs.get("start", args[1])
    end = user.kwargs.get("end", args[2])
    start = 0 if start is None else start
    end = size if end is None else end
    if not (
        isinstance(sliced, int) and isinstance(start, int) and isinstance(end, int)
    ):
        return False
    start += size if start < 0 else 0
    end += size if end < 0 else 0
    return sliced % 2 == dim and start >= 0 and min(end, size) <= keep


def _written_in_place(tensor: ir.IRNode) -> bool:
    """Whether memory that `tensor` reads was written in place before the node
    being lowered: its storage's buffer (or, for a value not yet stored, the
    buffers it reads) or a buffer aliasing one of them, as a fallback
    view.dtype to another width does."""
    if not V.graph.mutated_buffers:
        return False
    node = tensor
    while isinstance(node, (ir.MutableBox, ir.BaseView)):
        node = node.data
    names = (
        {node.get_name()} if isinstance(node, ir.Buffer) else {*node.get_read_names()}
    )
    aliases: dict[str, list[str]] = {}
    for buf in V.graph.buffers:
        for name in buf.get_inputs_that_alias_output():
            aliases.setdefault(name, []).append(buf.get_name())
            aliases.setdefault(buf.get_name(), []).append(name)
    pending = [*names]
    while pending:
        for alias in aliases.get(pending.pop(), ()):
            if alias not in names:
                names.add(alias)
                pending.append(alias)
    return any(name in V.graph.mutated_buffers for name in names)


def _zero_pad_source(
    mat: ir.IRNode, operand: int
) -> tuple[ir.IRNode, int, bool] | None:
    """(source, pad, read_past) where operand `operand` (0: mat1, 1: mat2) of
    the aten.mm being lowered is a zero pad of only the end of its M (mat1) or
    N (mat2) by `pad` elements, as pad_mm writes it, mm(a, constant_pad_nd(b,
    [0, 1]))[:, :197951], and every reader of the product slices the pad off;
    read_past says whether something can read past the slices (_read_past()).
    The source must be a graph input or a value still live at the mm (not an
    in-graph view, such as a Linear's w.t(), that only the pad reads), not
    written in place before the mm (_written_in_place())."""
    node = V.graph.current_node
    if node is None or node.target is not aten.mm.default:
        return None
    arg = node.args[operand]
    if not (
        isinstance(arg, torch.fx.Node)
        and arg.target is aten.constant_pad_nd.default
        and V.graph.env.get(arg) is mat
    ):
        return None
    value = arg.args[2] if len(arg.args) > 2 else arg.kwargs.get("value", 0)
    pad_arg = arg.args[1]
    if value != 0 or not isinstance(pad_arg, (list, tuple)):
        return None
    pads = [pad for pad in [*pad_arg, 0, 0, 0, 0][:4] if isinstance(pad, int)]
    if len(pads) != 4:
        return None
    # constant_pad_nd's pads run from the last dim: [N lo, N hi, K lo, K hi]
    # for mat2 [K, N], [K lo, K hi, M lo, M hi] for mat1 [M, K]
    hi = 1 if operand == 1 else 3
    if pads[hi] <= 0 or any(pad for i, pad in enumerate(pads) if i != hi):
        return None
    padded = mat.get_size()[operand]
    if not isinstance(padded, (int, sympy.Integer)):
        return None  # a dynamic M or N, as static_layout() declines
    size = int(padded)
    if not all(
        _slices_keep(user, operand, size - pads[hi], size) for user in node.users
    ):
        return None
    source = arg.args[0]
    if not isinstance(source, torch.fx.Node):
        return None
    # GraphLowering frees a node's value after its last use, which the pad may
    # have been; a graph input's stays in graph_inputs
    tensor = V.graph.env.get(source)
    if tensor is None and source.op == "placeholder":
        tensor = V.graph.graph_inputs.get(str(source.target))
    # reinplace_inplaceable_ops can move a write ahead of the mm, after the pad
    # read the source
    if not isinstance(tensor, ir.TensorBox) or _written_in_place(tensor):
        return None
    return tensor, pads[hi], any(_read_past(user) for user in node.users)


def _computed_in_kernel(node: ir.IRNode) -> bool:
    """Whether a kernel that reads `node` may compute it itself and hold it in
    the compute type rather than as stored: an operation not yet stored, or a
    ComputedBuffer whose kernel the reader's may fuse with. A graph input, a
    constant or an extern kernel's output (or a view of one) is read from
    memory, in its dtype."""
    if not ir.is_storage_and_layout(node):
        return True
    storage, _ = ir.as_storage_and_layout(node, freeze=False)
    return isinstance(storage.data, ir.ComputedBuffer)


def _pad_of_product(other: ir.IRNode, operand: int, size: list[Any]) -> Any:  # noqa: ANN401
    """What eager's padded product holds in each pad column (operand 1, other =
    mat1 [M, K]) or row (operand 0, other = mat2 [K, N]), broadcast to the
    pad's `size`: sum_k |other| * 0, +0 or NaN where `other`'s row (column)
    holds inf or NaN. |x| * 0 keeps every term +0 or NaN, so no reduction
    order gives -0. `other` is read as the dot reads it, rounded to its dtype,
    where the kernel may compute it in fp32 (_computed_in_kernel())."""
    if _computed_in_kernel(other):
        dtype = other.get_dtype()
        other = make_pointwise(
            lambda x: ops.to_dtype(
                ops.to_dtype(x, dtype, use_compute_types=False), dtype
            )
        )(other)
    zeros = lowerings[aten.mul](lowerings[aten.abs](other), 0)
    # K is mat1's dim 1 and mat2's dim 0
    return lowerings[aten.expand](lowerings[aten.sum](zeros, [operand], True), size)


def register_mm() -> None:
    """Lower mat1 @ mat2 as the backend's plan asks (HelionChoices), else as
    before: the user_lowerings entry this replaces, a user's, or Inductor's
    own lowering. The transposed product writes an N-contiguous inner buffer,
    whose transpose is the row-major result itself:

        buf0 = empty_strided_cuda((1024, 16384), (1, 1024), torch.bfloat16)
        helion_fused_mm_0(reinterpret_tensor(arg1_1, (1024, 4096), (4096, 1), 0),
                          reinterpret_tensor(arg0_1, (4096, 16384), (1, 4096), 0),
                          buf0)  # permute(mm(b.T, a.T))
        return (reinterpret_tensor(buf0, (16384, 1024), (1024, 1), 0), )
    """
    previous: Callable[..., Any] | None = user_lowerings.get(aten.mm.default)
    if getattr(previous, "_helion_transposed_mm", False):
        return  # registered already

    def lower_as_given(mat1: ir.IRNode, mat2: ir.IRNode) -> Any:  # noqa: ANN401
        # (a user's lowering that declines would leave Inductor's own lowering
        # the original operands, not these)
        result = None if previous is None else previous(mat1, mat2)
        return lowerings[aten.mm](mat1, mat2) if result is None else result

    def lower_mm(mat1: ir.IRNode, mat2: ir.IRNode) -> Any:  # noqa: ANN401
        choices = V.choices
        if not isinstance(choices, HelionChoices):
            return None if previous is None else previous(mat1, mat2)
        # (a zero pad's FX nodes are read only where the plan may use it)
        for operand in (1, 0) if choices.may_rewrite(mat1) else ():
            unpadded = _zero_pad_source((mat1, mat2)[operand], operand)
            if unpadded is None:
                continue
            source, pad, read_past = unpadded
            pair = (mat1, source) if operand == 1 else (source, mat2)
            if choices.pad_matmul_output(*pair, operand, exact=read_past):
                # the dot writes the source's columns (rows) of the padded
                # output, a small kernel the pad's
                product = lower_as_given(*pair)
                size = list(product.get_size())
                size[operand] = pad
                if not read_past:
                    zeros = lowerings[aten.full](
                        size,
                        0,
                        dtype=product.get_dtype(),
                        device=product.get_device(),
                    )
                    return ir.TensorBox(
                        ir.ConcatKernel.create([product, zeros], operand)
                    )
                pad_value = _pad_of_product(pair[1 - operand], operand, size)
                # what reads past the slices reads eager's layout, whose rows
                # comprehensive padding would widen
                with inductor_config.patch(comprehensive_padding=False):
                    padded = ir.ConcatKernel.create([product, pad_value], operand)
                return ir.TensorBox(padded)
        staged = False
        for operand in (0, 1):
            row_stride = choices.stage_matmul_operand(mat1, mat2, operand)
            if row_stride is None:
                continue
            strides = [row_stride, 1] if operand == 0 else [1, row_stride]
            mat = (mat1, mat2)[operand]
            if [int(stride) for stride in mat.get_stride()] == strides:
                # rows that line up from a base that does not (a view at an odd
                # offset), which require_exact_strides() would keep
                mat = ir.ExternKernel.copy_input(mat)
            mat = ir.ExternKernel.require_exact_strides(mat, strides)
            if operand == 0:
                mat1 = mat
            else:
                mat2 = mat
            staged = True
        if choices.transpose_matmul(mat1, mat2):
            permute = lowerings[aten.permute]
            product = lowerings[aten.mm](permute(mat2, [1, 0]), permute(mat1, [1, 0]))
            product.realize()
            product = ir.ExternKernel.require_stride_order(product, [0, 1])
            return permute(product, [1, 0])
        if choices.copy_matmul_b(mat1, mat2):
            mat2 = ir.ExternKernel.require_stride_order(mat2, [0, 1])
            return lower_as_given(mat1, mat2)
        if staged:
            return lower_as_given(mat1, mat2)
        return None if previous is None else previous(mat1, mat2)

    lower_mm._helion_transposed_mm = True  # pyrefly: ignore [missing-attribute]
    user_lowerings[aten.mm.default] = lower_mm


def register_bmm() -> None:
    """Lower torch.bmm(mat1, mat2) as (mat2.mT @ mat1.mT).mT where
    HelionChoices.transpose_bmm() says so, into an N-contiguous inner buffer as
    register_mm() does, else as before."""
    previous: Callable[..., Any] | None = user_lowerings.get(aten.bmm.default)
    if getattr(previous, "_helion_transposed_bmm", False):
        return  # registered already

    def lower_bmm(mat1: ir.IRNode, mat2: ir.IRNode) -> Any:  # noqa: ANN401
        choices = V.choices
        if isinstance(choices, HelionChoices) and choices.transpose_bmm(mat1, mat2):
            permute = lowerings[aten.permute]
            product = lowerings[aten.bmm](
                permute(mat2, [0, 2, 1]), permute(mat1, [0, 2, 1])
            )
            product.realize()
            product = ir.ExternKernel.require_stride_order(product, [2, 0, 1])
            return permute(product, [0, 2, 1])
        return None if previous is None else previous(mat1, mat2)

    lower_bmm._helion_transposed_bmm = True  # pyrefly: ignore [missing-attribute]
    user_lowerings[aten.bmm.default] = lower_bmm
