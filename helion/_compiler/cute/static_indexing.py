"""Keep bounded tensor indexing distinct from shape-only logical views.

Inductor returns a ReinterpretView for slice/select, which cannot describe the
cross-thread exchange of an already computed tensor. Complete fragment owners
implement its coordinates; scalar lowering must reject non-shape-only indexing.
"""

from __future__ import annotations

import torch
from torch.fx import Node

from ... import exc
from ..aten_lowering import Lowering
from ..aten_lowering import LoweringContext
from ..aten_lowering import passthrough_masked_value
from ..compile_environment import CompileEnvironment
from ..utils import is_full_slice


def static_tensor_index(node: Node) -> tuple[Node, list[object]] | None:
    target = node.target
    if target is torch.ops.aten.slice.Tensor:
        names = ("dim", "start", "end", "step")
        values: dict[str, object] = {"dim": 0, "start": None, "end": None, "step": 1}
    elif target is torch.ops.aten.select.int:
        names = ("dim", "index")
        values = {}
    else:
        return None
    if len(node.args) > len(names) + 1 or any(key not in names for key in node.kwargs):
        return None
    positional = dict(zip(names, node.args[1:], strict=False))
    if positional.keys() & node.kwargs.keys():
        return None
    values.update(positional)
    values.update(node.kwargs)
    source = node.args[0]
    if not isinstance(source, Node):
        return None
    fake = source.meta.get("val")
    dim = values.get("dim")
    if not isinstance(fake, torch.Tensor) or type(dim) is not int:
        return None
    if not -fake.ndim <= dim < fake.ndim:
        return None
    dim %= fake.ndim
    extent = fake.shape[dim]
    indices: list[object] = [slice(None)] * fake.ndim
    if target is torch.ops.aten.select.int:
        if isinstance(extent, torch.SymInt):
            extent = CompileEnvironment.current().static_kernel_tensor_extent(extent)
        position = values.get("index")
        if (
            type(extent) is not int
            or type(position) is not int
            or not -extent <= position < extent
        ):
            return None
        indices[dim] = position % extent
    else:
        start, end, step = (values[key] for key in ("start", "end", "step"))
        if (
            (start is not None and type(start) is not int)
            or (end is not None and type(end) is not int)
            or type(step) is not int
            or step <= 0
        ):
            return None
        # make_fx retains ':' on symbolic leading dimensions. It is already
        # the full logical view, so no knowledge of that extent is needed.
        if start in (None, 0) and end in (None, 2**63 - 1) and step == 1:
            return source, indices
        if isinstance(extent, torch.SymInt):
            extent = CompileEnvironment.current().static_kernel_tensor_extent(extent)
        if type(extent) is not int:
            return None
        bounds = slice(start, end, step).indices(extent)
        if bounds != (0, extent, 1):
            indices[dim] = slice(*bounds)
    return source, indices


def shape_only_indices(source: Node, indices: list[object]) -> bool:
    fake = source.meta.get("val")
    axis = 0
    for index in indices:
        if index is None:
            continue
        if not is_full_slice(index):
            if (
                type(index) is not int
                or index != 0
                or not isinstance(fake, torch.Tensor)
                or axis >= fake.ndim
            ):
                return False
            extent = fake.shape[axis]
            if isinstance(extent, torch.SymInt):
                extent = CompileEnvironment.current().static_kernel_tensor_extent(
                    extent
                )
            if extent != 1:
                return False
        axis += 1
    return True


class StaticIndexLowering(Lowering):
    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        index = static_tensor_index(node)
        assert index is not None
        if not shape_only_indices(*index):
            raise exc.BackendUnsupported(
                "cute", "static tensor indexing requires a complete coordinate owner"
            )
        return ctx.env[index[0]]

    def get_masked_value(self, node: Node) -> float | bool | None:
        return passthrough_masked_value(node)
