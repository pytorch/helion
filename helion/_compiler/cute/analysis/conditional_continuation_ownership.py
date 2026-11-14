"""Prove independent continuation outputs and bounded imported statistics."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import PythonMod

from ....language import _tracing_ops
from ...host_function import HostFunction
from ...variable_origin import TileBeginOrigin
from ...variable_origin import TileIdOrigin
from ..cute_mma_dag import _NotMine
from .conditional_index_context import ConditionalIndexContext
from .separable_index_map import SeparableIndexMap
from .separable_index_map import prove_strided_index_map

if TYPE_CHECKING:
    from torch.fx import Node

    from ....runtime.config import Config
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR
    from .conditional_attention_program import ConditionalAttentionProgram


@dataclass(frozen=True)
class ConditionalContinuationOwnership:
    root: sympy.Symbol
    output: SeparableIndexMap
    lse: SeparableIndexMap
    imports: tuple[SeparableIndexMap, ...]


def _map(
    context: ConditionalIndexContext,
    node: Node,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> SeparableIndexMap | None:
    host, indices = node.args[:2]
    if not isinstance(host, torch.fx.Node):
        return None
    host = context.resolve(host)
    fake = host.meta.get("val")
    if (
        host.target is not _tracing_ops._host_tensor
        or not isinstance(fake, torch.Tensor)
        or fake.dtype is not dtype
        or not isinstance(indices, (tuple, list))
        or any(type(n) is not int for n in (*fake.shape, *fake.stride()))
    ):
        return None
    return prove_strided_index_map(context, indices, fake.shape, fake.stride(), shape)


def _independent(
    context: ConditionalIndexContext, mapped: SeparableIndexMap
) -> sympy.Symbol | None:
    """Include the folded scalar task coordinate in one mixed-radix proof."""
    if len(mapped.base.free_symbols) != 1:
        return None
    root = next(iter(mapped.base.free_symbols))
    if (
        not isinstance(root, sympy.Symbol)
        or root.assumptions0.get("integer") is not True
    ):
        return None
    info = HostFunction.current().expr_to_origin.get(root)
    if (
        info is None
        or not isinstance(info.origin, (TileBeginOrigin, TileIdOrigin))
        or not context.lead_coordinate(info.origin)
    ):
        return None
    extent = context.axis_of_block(info.origin.block_id).extent
    offset = mapped.base.xreplace({root: sympy.Integer(0)})
    if not isinstance(offset, sympy.Integer) or offset < 0:
        return None
    expression = sympy.expand(sympy.Add(mapped.base, sympy.Mul(-1, offset)))
    step = expression.coeff(root)
    digits = []
    if isinstance(step, sympy.Integer) and expression == step * root and step > 0:
        digits.append((extent, int(step)))
    else:
        divisors = {
            int(part.args[1])
            for part in sympy.preorder_traversal(expression)
            if isinstance(part, (FloorDiv, PythonMod))
            and part.args[0] == root
            and isinstance(part.args[1], sympy.Integer)
        }
        if len(divisors) != 1:
            return None
        divisor = divisors.pop()
        if divisor <= 1:
            return None
        major = expression.coeff(FloorDiv(root, divisor))
        minor = expression.coeff(PythonMod(root, divisor))
        if not all(isinstance(n, sympy.Integer) and n > 0 for n in (major, minor)):
            return None
        if (
            sympy.expand(
                expression
                - major * FloorDiv(root, divisor)
                - minor * PythonMod(root, divisor)
            )
            != 0
        ):
            return None
        digits.extend(
            (
                ((extent + divisor - 1) // divisor, int(major)),
                (min(extent, divisor), int(minor)),
            )
        )
    for axis in mapped.axes:
        extent, divisor = axis.axis.extent, axis.divisor
        digits.append(((extent + divisor - 1) // divisor, axis.major_stride))
        if divisor > 1:
            digits.append((min(extent, divisor), axis.minor_stride))
    # A partial last quotient overapproximates its remainder digit. Proving
    # non-overlap for this superset is sufficient for every launched task.
    span = 1
    for count, stride in sorted(digits, key=operator.itemgetter(1)):
        if count <= 1:
            continue
        if stride < span:
            return None
        span += (count - 1) * stride
    return root


def prove_conditional_continuation_ownership(
    program: ConditionalAttentionProgram,
    ir: DeviceIR,
    env: CompileEnvironment,
    config: Config,
) -> ConditionalContinuationOwnership | None:
    continuation = program.recurrence.continuation
    if continuation is None:
        return None
    try:
        context = ConditionalIndexContext(program, ir, env, config)
        if len(context.root_blocks) != 1:
            return None
        shape = tuple(program.recurrence.pv.meta["val"].shape)
        output = _map(context, continuation.output_store, shape, torch.bfloat16)
        lse = _map(context, continuation.lse_store, shape[:1], torch.float32)
        if output is None or lse is None:
            return None
        root = _independent(context, output)
        if root is None or _independent(context, lse) != root:
            return None
        if tuple(axis.axis for axis in lse.axes) != (output.axes[0].axis,):
            return None
        imported = tuple(
            _map(context, node, shape if index == 2 else shape[:1], torch.float32)
            for index, node in enumerate(continuation.imported_reads)
        )
        if any(item is None for item in imported):
            return None
        imports = tuple(item for item in imported if item is not None)
        for index, item in enumerate(imports):
            expected = output.axes if index == 2 else lse.axes
            if tuple(axis.axis for axis in item.axes) != tuple(
                axis.axis for axis in expected
            ):
                return None
        # Imports are readonly within this phase (program proof), so they need
        # bounds and coordinate correspondence but no inter-task exclusivity.
        return ConditionalContinuationOwnership(root, output, lse, imports)
    except _NotMine:
        return None
