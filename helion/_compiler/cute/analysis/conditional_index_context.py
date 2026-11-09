"""Exact index-map context for an owned conditional attention region.

Iota provenance, rather than equal shape hints, identifies vector axes. The
configured root tile must be scalar; runtime metadata loads and loop-dependent
coordinates are not treated as bounded root symbols by this context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from ....language import view_ops
from ...host_function import HostFunction
from ...variable_origin import TileBeginOrigin
from ...variable_origin import TileIdOrigin
from ..cute_mma_dag import AXIS_LEAD
from ..cute_mma_dag import AXIS_RDIM
from ..cute_mma_dag import Axis
from ..cute_mma_dag import _NotMine
from .conditional_attention_access import _AccessProof

if TYPE_CHECKING:
    from torch.fx import Node

    from ....runtime.config import Config
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR
    from .conditional_attention_program import ConditionalAttentionProgram


class ConditionalIndexContext:
    def __init__(
        self,
        program: ConditionalAttentionProgram,
        ir: DeviceIR,
        env: CompileEnvironment,
        config: Config,
    ) -> None:
        self.env = env
        self.config = config
        self.resolve = _AccessProof(program.recurrence).resolve
        phase = program.recurrence.regions.root.phase_index
        if type(phase) is not int or not 0 <= phase < len(ir.grid_block_ids):
            raise _NotMine
        self.root_blocks = {
            env.canonical_block_id(block) for block in ir.grid_block_ids[phase]
        }
        self.axes: dict[int, Axis] = {}
        self.iotas: dict[Node, Axis] = {}
        self.axis_count = 0
        self.cache: dict[Node, tuple[Axis | None, ...]] = {}

    def axis_of_block(self, block_id: int) -> Axis:
        block_id = self.env.canonical_block_id(block_id)
        if block_id not in self.root_blocks:
            raise _NotMine
        if block_id not in self.axes:
            info = self.env.block_sizes[block_id]
            configured = info.from_config(self.config)
            if not isinstance(info.size, (int, torch.SymInt)):
                raise _NotMine
            extent = self.env.try_concretize_symint(info.size)
            if type(configured) is not int or configured != 1:
                raise _NotMine
            if type(extent) is not int or extent <= 0:
                raise _NotMine
            self.axes[block_id] = Axis(self.axis_count, block_id, AXIS_LEAD, extent)
            self.axis_count += 1
        return self.axes[block_id]

    def lead_coordinate(self, origin: object) -> bool:
        if not isinstance(origin, (TileBeginOrigin, TileIdOrigin)):
            return False
        self.axis_of_block(origin.block_id)
        return True

    def uniform_expr(self, node: Node) -> tuple[sympy.Expr, tuple[int, ...]]:
        value = self.resolve(node).meta.get("val")
        if type(value) is int:
            return sympy.Integer(value), ()
        if not isinstance(value, torch.SymInt):
            raise _NotMine
        expression = value._sympy_()
        if not isinstance(expression, sympy.Expr):
            raise _NotMine
        host = HostFunction.current()
        for symbol in expression.free_symbols:
            info = host.expr_to_origin.get(symbol)
            if info is None or not self.lead_coordinate(info.origin):
                raise _NotMine
        return expression, ()

    def shape_axes(self, node: Node) -> tuple[Axis | None, ...]:
        node = self.resolve(node)
        if node in self.cache:
            return self.cache[node]
        fake = node.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            raise _NotMine
        if any(type(size) is not int for size in fake.shape):
            raise _NotMine
        if not fake.ndim:
            result: tuple[Axis | None, ...] = ()
        elif node.target is torch.ops.prims.iota.default:
            if fake.ndim != 1 or fake.shape[0] <= 0:
                raise _NotMine
            if node not in self.iotas:
                # Negative ids are local complete-iota axes, not configured
                # block ids. Their actual producer nodes retain identity even
                # when two independent axes have the same integer extent.
                self.iotas[node] = Axis(
                    self.axis_count, -self.axis_count - 1, AXIS_RDIM, fake.shape[0]
                )
                self.axis_count += 1
            result = (self.iotas[node],)
        elif node.target is view_ops.subscript:
            source, indices = node.args
            if not isinstance(source, torch.fx.Node) or not isinstance(
                indices, (tuple, list)
            ):
                raise _NotMine
            source_axes = iter(self.shape_axes(source))
            axes: list[Axis | None] = []
            for index in indices:
                if index is None:
                    axes.append(None)
                elif index == slice(None):
                    try:
                        axes.append(next(source_axes))
                    except StopIteration:
                        raise _NotMine from None
                else:
                    raise _NotMine
            if list(source_axes) or len(axes) != fake.ndim:
                raise _NotMine
            result = tuple(axes)
        else:
            axes = [None] * fake.ndim
            for source in node.all_input_nodes:
                value = self.resolve(source).meta.get("val")
                if not isinstance(value, torch.Tensor):
                    continue
                source_axes = self.shape_axes(source)
                if len(source_axes) > fake.ndim:
                    raise _NotMine
                offset = fake.ndim - len(source_axes)
                for index, axis in enumerate(source_axes):
                    if value.shape[index] == 1:
                        continue
                    current = index + offset
                    if value.shape[index] != fake.shape[current] or axis is None:
                        raise _NotMine
                    if axes[current] is not None and axes[current] != axis:
                        raise _NotMine
                    axes[current] = axis
            if any(
                axis is None and size != 1
                for axis, size in zip(axes, fake.shape, strict=True)
            ):
                raise _NotMine
            result = tuple(axes)
        self.cache[node] = result
        return result
