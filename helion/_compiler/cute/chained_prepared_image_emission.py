"""Explicit typed bindings for original raw-load preparation transfers.

No caller is activated here. The enclosing pipeline still owns publication,
whole-frame lifetime and release. Bindings keep the original allocation charged
and replace only the epilogue/recurrence boundary map, never the semantic frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from .chained_preparation_frame import PreparationBuffer
from .chained_prepared_image_transfers import RawWideningTransfer
from .chained_prepared_image_transfers import discover_raw_widening_transfers
from .chained_vector_leaf import DenseVectorSink

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _Expression
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_frame import PreparationFrame
    from .chained_vector_stage import VectorStaging
    from .warp_specialized_plan import SharedBufferRegion


@dataclass(frozen=True)
class RawImageBinding:
    transfer: RawWideningTransfer
    buffer: PreparationBuffer
    region: SharedBufferRegion

    @property
    def vector_sink(self) -> DenseVectorSink:
        dtype = (
            "cutlass.BFloat16"
            if self.buffer.dtype == torch.bfloat16
            else "cutlass.Float16"
        )
        rows, columns = self.buffer.shape
        return DenseVectorSink(self.buffer.name, (rows, columns), dtype)

    def view_lines(self, frame_pointer: str) -> tuple[str, ...]:
        sink = self.vector_sink
        rows, columns = sink.shape
        return (
            (
                f"{self.buffer.name} = cute.make_tensor(cute.recast_ptr({frame_pointer} + "
                f"{self.region.byte_offset}, dtype={sink.dtype}), "
                f"cute.make_layout(({rows}, {columns}), stride=({columns}, 1)))"
            ),
        )


@dataclass(frozen=True)
class BoundRawWidenings:
    plan: ChainedMatmulPlan
    frame: PreparationFrame
    shapes: tuple[tuple[Node, tuple[int, ...]], ...]
    bindings: tuple[RawImageBinding, ...]
    _selection: tuple[object, ...]

    def matches(self, plan: ChainedMatmulPlan) -> bool:
        if (
            self._selection != (self.plan, self.frame, self.shapes, self.bindings)
            or replace(plan, prepared_widenings=None) != self.plan
            or self.plan.prepared_widenings is not None
        ):
            return False
        shapes = dict(self.shapes)
        return all(
            binding.transfer.matches(self.plan, self.frame, shapes)
            and binding.region == self.frame.layout.region(binding.transfer.buffer.name)
            for binding in self.bindings
        )

    def recurrence_boundaries(self, boundaries: Mapping[Node, str]) -> dict[Node, str]:
        """Clone original semantic boundaries after their publication barrier.

        The returned map names the original HALF node, not its Float32 user.
        Only the recurrence plan carrying this proof may consume the map.
        """
        from .chained_matmul import _UnsupportedChain

        if not self.matches(self.plan):
            raise _UnsupportedChain("raw image transfer revision changed")
        result = dict(boundaries)
        for binding in self.bindings:
            transfer = binding.transfer
            if (
                result.get(transfer.widening) != transfer.buffer.name
                or transfer.source in result
            ):
                raise _UnsupportedChain(
                    "raw image does not replace its original boundary"
                )
            result.pop(transfer.widening)
            result[transfer.source] = binding.buffer.name
        return result

    def expression(
        self, expression: _Expression, node: Node, coordinates: tuple[str, ...]
    ) -> str:
        """Emit the original widening, with its original typed-zero read guard."""
        from ..host_function import HostFunction
        from . import chained_half_widening
        from .chained_matmul import _UnsupportedChain

        if not self.matches(expression.plan):
            raise _UnsupportedChain("raw widening transfer revision changed")
        binding = next(item for item in self.bindings if item.transfer.widening is node)
        if expression.boundaries.get(binding.transfer.source) != binding.buffer.name:
            raise _UnsupportedChain("raw widening lacks its typed boundary")
        function = (
            "bfloat16_to_float32"
            if binding.transfer.dtype == torch.bfloat16
            else "float16_to_float32"
        )
        value = expression.value(binding.transfer.source, coordinates)
        origin = HostFunction.current().import_from_module(
            vars(chained_half_widening), function
        )
        return expression.bind(f"{origin.host_str()}({value})")


def bind_raw_widening_transfers(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    shapes: Mapping[Node, tuple[int, ...]],
    transfers: tuple[RawWideningTransfer, ...],
) -> BoundRawWidenings | None:
    """Prepare typed views in unchanged, fully charged original allocations.

    Binding does not shorten any source lease, remove an original allocation,
    mutate a Node dtype, or emit a READY/EMPTY token. Native prepared aliases
    must not also claim these allocations in the eventual pipeline integration.
    """
    available = discover_raw_widening_transfers(plan, frame, shapes)
    if (
        plan.prepared_widenings is not None
        or available is None
        or not transfers
        or transfers != tuple(item for item in available if item in transfers)
    ):
        return None
    names = {buffer.name for buffer in frame.buffers}
    bindings = []
    for transfer in transfers:
        name = f"{transfer.buffer.name}_raw"
        if name in names:
            return None
        names.add(name)
        bindings.append(
            RawImageBinding(
                transfer,
                PreparationBuffer(
                    name, "frontier", transfer.source, transfer.dtype, transfer.shape
                ),
                frame.layout.region(transfer.buffer.name),
            )
        )
    resolved = tuple(shapes.items())
    values = tuple(bindings)
    return BoundRawWidenings(
        plan, frame, resolved, values, (plan, frame, resolved, values)
    )


def emit_raw_image(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    bound: BoundRawWidenings,
    binding: RawImageBinding,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    *,
    vector: VectorStaging,
    producer_unroll: BoundedProducerUnroll,
) -> list[str] | None:
    """Use the original load emitter and its original vector/zero-fill guards."""
    from .chained_prepared_values import emit_prepared_value

    if (
        not bound.matches(plan)
        or binding not in bound.bindings
        or binding.transfer.source in boundaries
        or binding.transfer.widening in boundaries
    ):
        return None
    return emit_prepared_value(
        cg,
        plan,
        binding.buffer,
        boundaries,
        execution,
        vector=vector,
        producer_unroll=producer_unroll,
        shared_sink=binding.vector_sink,
    )
