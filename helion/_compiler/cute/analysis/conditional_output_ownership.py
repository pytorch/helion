"""Sufficient mixed-radix ownership proof for the three partial stores.

The source scalar root coordinate and exact output maps are retained. Positive
ordered physical strides prove disjoint logical digits and disjoint root-item
tiles; no finite test grid stands in for that proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import PythonMod
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from ...host_function import HostFunction
from ...variable_origin import TileBeginOrigin
from ...variable_origin import TileIdOrigin
from ..cute_mma_dag import _NotMine
from .conditional_index_context import ConditionalIndexContext
from .separable_index_map import prove_separable_index_map

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ....runtime.config import Config
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR
    from .conditional_attention_program import ConditionalAttentionProgram
    from .separable_index_map import SeparableIndexMap


@dataclass(frozen=True)
class ScalarRootLayout:
    root: sympy.Symbol
    root_extent: int
    root_stride: int
    base_offset: int
    item_span: int


@dataclass(frozen=True)
class ConditionalOutputOwnership:
    output_map: SeparableIndexMap
    output_layout: ScalarRootLayout
    metadata_host: Node
    metadata_layout: ScalarRootLayout


def _layout(
    context: ConditionalIndexContext,
    base: sympy.Expr,
    digits: Sequence[tuple[int, int]],
) -> ScalarRootLayout | None:
    symbols = base.free_symbols
    if len(symbols) != 1:
        return None
    root = next(iter(symbols))
    if (
        not isinstance(root, sympy.Symbol)
        or root.assumptions0.get("integer") is not True
    ):
        return None
    origin = HostFunction.current().expr_to_origin.get(root)
    if (
        origin is None
        or not isinstance(origin.origin, (TileBeginOrigin, TileIdOrigin))
        or not context.lead_coordinate(origin.origin)
    ):
        return None
    # Integer quotient/remainder identity; this does not alter emitted source
    # arithmetic. The index-map proof has already checked tensor-width bounds.
    replacements: dict[sympy.Basic, sympy.Basic] = {}
    for part in sympy.preorder_traversal(base):
        if (
            isinstance(part, PythonMod)
            and part.args[0] == root
            and isinstance(part.args[1], sympy.Integer)
            and part.args[1] > 0
        ):
            replacements[part] = sympy.Add(
                root, sympy.Mul(-1, part.args[1], FloorDiv(root, part.args[1]))
            )
    normalized = sympy.expand(base.xreplace(replacements))
    stride = normalized.coeff(root)
    offset = sympy.Add(normalized, sympy.Mul(-1, stride, root))
    if (
        not isinstance(stride, sympy.Integer)
        or stride <= 0
        or not isinstance(offset, sympy.Integer)
        or offset < 0
    ):
        return None
    span = 1
    for count, step in sorted(digits, key=operator.itemgetter(1)):
        if type(count) is not int or type(step) is not int or count <= 0:
            return None
        if count == 1:
            continue
        if step < span:
            return None
        span += (count - 1) * step
    if stride < span:
        return None
    # lead_coordinate admitted this exact typed origin.
    axis = context.axis_of_block(origin.origin.block_id)
    return ScalarRootLayout(root, axis.extent, int(stride), int(offset), span)


def prove_conditional_output_ownership(
    program: ConditionalAttentionProgram,
    ir: DeviceIR,
    env: CompileEnvironment,
    config: Config,
) -> ConditionalOutputOwnership | None:
    """Prove O plus paired m/l stores without changing their address recipes."""
    try:
        context = ConditionalIndexContext(program, ir, env, config)
    except _NotMine:
        return None
    # Every launched root coordinate must be accounted for by the physical
    # output footprint. An unused second grid axis duplicates those stores.
    if len(context.root_blocks) != 1:
        return None
    recurrence = program.recurrence
    by_value = {node.args[2]: node for node in program.writes}
    output = by_value.get(recurrence.loop_outputs[1])
    maximum = by_value.get(recurrence.loop_outputs[0])
    denominator = by_value.get(recurrence.final_denominator)
    if output is None or maximum is None or denominator is None:
        return None
    for store in (output, maximum, denominator):
        if len(store.args) != 4 or store.args[3] is not None or store.kwargs:
            return None
    output_host, output_indices = output.args[:2]
    if not isinstance(output_host, torch.fx.Node):
        return None
    output_fake = output_host.meta.get("val")
    if (
        not isinstance(output_fake, torch.Tensor)
        or output_fake.dtype != torch.float32
        or output_fake.ndim != 1
        or output_fake.stride() != (1,)
        or type(output_fake.shape[0]) is not int
        or not isinstance(output_indices, (tuple, list))
        or len(output_indices) != 1
        or not isinstance(output_indices[0], torch.fx.Node)
    ):
        return None
    index_fake = output_indices[0].meta.get("val")
    value_fake = recurrence.loop_outputs[1].meta.get("val")
    if (
        not isinstance(index_fake, torch.Tensor)
        or not isinstance(value_fake, torch.Tensor)
        or index_fake.ndim != 2
        or index_fake.shape != value_fake.shape
    ):
        return None
    try:
        output_map = prove_separable_index_map(
            context, output_indices[0], output_fake.shape[0]
        )
        if (
            output_map is None
            or len(output_map.axes) != 2
            or output_map.axes[0].axis == output_map.axes[1].axis
            or tuple(axis.axis.extent for axis in output_map.axes) != value_fake.shape
        ):
            return None
        digits = []
        for axis in output_map.axes:
            extent, divisor = axis.axis.extent, axis.divisor
            digits.append(((extent + divisor - 1) // divisor, axis.major_stride))
            if divisor > 1:
                digits.append((min(extent, divisor), axis.minor_stride))
        output_layout = _layout(context, output_map.base, digits)
        if output_layout is None:
            return None
        host, m_indices = maximum.args[:2]
        l_host, l_indices = denominator.args[:2]
        if not isinstance(host, torch.fx.Node) or l_host is not host:
            return None
        fake = host.meta.get("val")
        if (
            not isinstance(fake, torch.Tensor)
            or fake.dtype != torch.float32
            or fake.ndim != 3
            or any(type(size) is not int for size in (*fake.shape, *fake.stride()))
            or fake.shape[2] != 2
            or not isinstance(m_indices, (tuple, list))
            or not isinstance(l_indices, (tuple, list))
            or len(m_indices) != 3
            or len(l_indices) != 3
            or m_indices[:2] != l_indices[:2]
            or type(m_indices[2]) is not int
            or type(l_indices[2]) is not int
            or (m_indices[2], l_indices[2]) != (0, 1)
            or not isinstance(m_indices[0], torch.fx.Node)
            or not isinstance(m_indices[1], torch.fx.Node)
        ):
            return None
        row = context.resolve(m_indices[1])
        axes = context.shape_axes(row)
        if (
            len(axes) != 1
            or axes[0] is None
            or row.target is not torch.ops.prims.iota.default
            or row.args != (fake.shape[1],)
            or row.kwargs.get("start") != 0
            or row.kwargs.get("step") != 1
            or axes[0] != output_map.axes[0].axis
        ):
            return None
        slot, loops = context.uniform_expr(m_indices[0])
        if loops:
            return None
        metadata_layout = _layout(
            context,
            sympy.Mul(slot, fake.stride(0)),
            [(fake.shape[1], fake.stride(1)), (2, fake.stride(2))],
        )
        if metadata_layout is None or metadata_layout.root != output_layout.root:
            return None
        bounds = bound_sympy(
            slot,
            {metadata_layout.root: ValueRanges(0, metadata_layout.root_extent - 1)},
        )
        if bounds.lower < 0 or bounds.upper >= fake.shape[0]:
            return None
        return ConditionalOutputOwnership(
            output_map, output_layout, host, metadata_layout
        )
    except _NotMine:
        return None
