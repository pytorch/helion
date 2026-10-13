from __future__ import annotations

from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_loop_tmem_carry import plan_loop_tmem_carry
from helion._compiler.cute.chained_loop_tmem_carry_transport import (
    LoopTmemCarryTransport,
)
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.chained_tmem_accumulator import (
    plan_tmem_accumulator_residency,
)


def _carry(width, offset):
    graph = Graph()
    state = _input(graph, "state", (128, width), torch.float32)
    image = _convert(graph, state, torch.bfloat16)
    projection_width = offset or 16
    weight = _input(graph, "weight", (width, projection_width), torch.bfloat16)
    projected = _dot(graph, image, weight)
    update = _input(graph, "update", (128, 16), torch.bfloat16)
    groups = (
        ContractionGroup((0,), (StageGeometry((128, projection_width, width), False),)),
    )
    final_geometries = []
    if offset:
        side = _input(graph, "side", (16, offset), torch.bfloat16)
        _dot(graph, update, side, projected)
        final_geometries.append(StageGeometry((128, offset, 16), False))
    rhs = _input(graph, "rhs", (16, width), torch.bfloat16)
    accumulator = _call(
        graph, torch.ops.aten.mul.Scalar, (state, 0.5), (128, width), torch.float32
    )
    result = _dot(graph, update, rhs, accumulator)
    final_geometries.append(StageGeometry((128, width, 16), False))
    groups += (
        ContractionGroup(
            tuple(range(1, 1 + len(final_geometries))), tuple(final_geometries)
        ),
    )
    plan = replace(_loop_plan(graph, (state,), (result,)), contraction_groups=groups)
    residency = plan_tmem_accumulator_residency(plan, groups)
    candidate = plan_loop_tmem_carry(plan, groups, residency)
    assert candidate is not None and candidate.member_offset == offset
    arena = 64
    snapshot = (arena + candidate.arena_columns + 31) // 32 * 32
    return LoopTmemCarryTransport(
        candidate,
        arena,
        snapshot,
        snapshot + width // 2,
        "cutlass.BFloat16",
        (),
        "cutlass.BFloat16(0)",
        (),
        "cutlass.Float32(0)",
    )


@pytest.mark.parametrize(
    "width,offset",
    [
        (16, 0),
        (16, 16),
        (32, 32),
        (48, 16),
        (64, 32),
        (96, 48),
        (128, 32),
        (192, 64),
        (240, 16),
    ],
)
def test_actual_cute_carry_seed_load_matches_store_rmem_and_member_coordinates_cpu(
    width, offset
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    carry = _carry(width, offset)
    seed = carry.seed
    total = width + offset
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, total),
                tcgen05.OperandSource.SMEM,
            )
            layout = mma.make_fragment_C(mma.partition_shape_C((128, total))).layout
            arena = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32,
                    carry.arena_offset,
                    cute.AddressSpace.tmem,
                ),
                layout,
            )
            origin, shape = ((0, offset), 0, 0), ((128, width), 1, 1)
            segment = cute.composition(
                cute.domain_offset(origin, arena), cute.make_layout(shape)
            )
            identity = cute.composition(
                cute.domain_offset(
                    origin,
                    mma.get_slice(0).partition_C(
                        cute.make_identity_tensor((128, total))
                    ),
                ),
                cute.make_layout(shape),
            )
            store = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(min(32, width & -width))),
                    cutlass.Float32,
                ),
                segment,
            )
            seen = []
            for thread in range(128):
                writer = store.get_slice(thread)
                target = writer.partition_D(segment)
                coordinates = writer.partition_S(identity)
                values = cute.make_rmem_tensor(coordinates.shape, cutlass.Float32)
                namespace: dict[str, Any] = {
                    "cute": cute,
                    "cutlass": cutlass,
                    "tcgen05": tcgen05,
                    "consumer_thread": thread,
                    f"{seed}_segment": segment,
                    f"{seed}_values": values,
                }
                # Execute the production load directly into the existing
                # store-layout RMEM, before the original accumulator expression.
                lines = carry.seed_values(
                    ChainedExecution(128, thread="consumer_thread")
                )
                exec("\n".join(lines[:5]), namespace)
                reader = namespace[f"{seed}_reader"]
                loaded_coordinates = reader.partition_D(identity)
                loaded_values = cute.make_rmem_tensor(
                    loaded_coordinates.shape, cutlass.Float32
                )
                assert int(cute.size(coordinates)) == int(cute.size(loaded_coordinates))
                for index in range(int(cute.size(coordinates))):
                    coord = tuple(map(int, coordinates[index]))
                    assert coord == tuple(map(int, loaded_coordinates[index]))
                    assert int(values.layout(index)) == int(loaded_values.layout(index))
                    seen.append(coord)
                cute.copy(store, values, target)
            expected = {
                (row, col) for row in range(128) for col in range(offset, total)
            }
            assert len(seen) == len(set(seen)) and set(seen) == expected
        assert module.operation.verify()
    assert torch.cuda.is_initialized() is before
