"""Helion-owned tcgen05 pipeline state helpers.

This mirrors ``cutlass.pipeline.make_pipeline_state`` for the explicit
pure-matmul role-lifecycle path while preserving a Helion-owned state type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import PipelineState
from cutlass.pipeline import PipelineUserType

if TYPE_CHECKING:
    from cutlass._mlir import ir


class HelionPipelineState(PipelineState):
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> HelionPipelineState:
        # Keep this in lockstep with cutlass.pipeline.PipelineState; only the
        # returned Python type differs.
        return HelionPipelineState(
            self.stages,
            Int32(values[0]),
            Int32(values[1]),
            Int32(values[2]),
        )


@dsl_user_op
def make_pipeline_state(
    user_type: PipelineUserType,
    stages: int,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> HelionPipelineState:
    # Match CUTLASS's effective ProducerConsumer behavior: it initializes with
    # producer phase state because the producer branch wins upstream too.
    if user_type in (PipelineUserType.Producer, PipelineUserType.ProducerConsumer):
        phase = 1
    elif user_type is PipelineUserType.Consumer:
        phase = 0
    else:
        raise AssertionError("invalid PipelineUserType")
    return HelionPipelineState(
        stages,
        Int32(0, loc=loc, ip=ip),
        Int32(0, loc=loc, ip=ip),
        Int32(phase, loc=loc, ip=ip),
    )
