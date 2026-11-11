from __future__ import annotations

import ast
import dataclasses
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest import mock

import pytest
import torch

import helion
from helion._compat import supports_host_tensor_descriptor
from helion._compiler import cross_loop_codegen
from helion._compiler import cross_loop_scheduler
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cross_loop_codegen import _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
from helion._compiler.cross_loop_codegen import _ast_fingerprint
from helion._compiler.cross_loop_codegen import _clone_opaque_loop_segment
from helion._compiler.cross_loop_codegen import _clone_opaque_statements
from helion._compiler.cross_loop_codegen import (
    _clone_opaque_statements_with_loop_segments,
)
from helion._compiler.cross_loop_codegen import _dry_pass_binds
from helion._compiler.cross_loop_codegen import _dry_pass_rewrite
from helion._compiler.cross_loop_codegen import _outline_cross_loop_region
from helion._compiler.cross_loop_codegen import _scope_wait
from helion._compiler.cross_loop_codegen import _triton_root_requires_kernel_scope
from helion._compiler.device_function import DeviceFunction
from helion._compiler.program_id import PIDInfo
from helion._compiler.tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import DenseTaskOrder
from helion._compiler.tile_dependency import Incidence
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.triton.backend import TritonBackend
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import output_only
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessTensorDescriptor
from helion.autotuner.benchmark_provider import _triton_compile
import helion.language as hl
from helion.runtime.triton.launcher import compile_only_launch_args


def _generated_function(code: str, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one generated function {name!r}")
    return matches[0]


def _call_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def grouped_affine_chain(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    group_size: int,
    reverse_groups: hl.constexpr,
) -> torch.Tensor:
    m, hidden = x.size()
    _, twice_intermediate = w13.size()
    intermediate = twice_intermediate // 2
    _, out_features = w2.size()
    hl.specialize(group_size)
    groups = intermediate // group_size
    gate_up = torch.empty((m, twice_intermediate), dtype=x.dtype, device=x.device)
    activation = torch.empty((m, intermediate), dtype=x.dtype, device=x.device)
    activation_scale = torch.empty((m, groups), dtype=torch.float32, device=x.device)
    out = torch.empty((m, out_features), dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, twice_intermediate]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(hidden, block_size=32):
            acc = torch.addmm(acc, x[tile_m, tile_k], w13[tile_k, tile_n])
        gate_up[tile_m, tile_n] = acc.to(x.dtype)

    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, group_size]):
        if reverse_groups:
            source_group = groups - 1 - tile_i.id
            source_i = source_group * group_size + hl.arange(group_size)
        else:
            source_i = tile_i
        gate = gate_up[tile_m, source_i].to(torch.float32)
        up = gate_up[tile_m, source_i + intermediate].to(torch.float32)
        activated = gate * up
        map_scale = torch.amax(torch.abs(activated), dim=-1) + 1
        activation[tile_m, tile_i] = activated.to(x.dtype)
        activation_scale[tile_m, tile_i.id] = map_scale

    for tile_m, tile_n in hl.tile([m, out_features]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate, block_size=group_size):
            values = activation[tile_m, tile_k].to(torch.float32)
            consumer_scale = activation_scale[tile_m, tile_k.id].to(torch.float32)
            acc = torch.addmm(
                acc,
                values * consumer_scale[:, None],
                w2[tile_k, tile_n].to(torch.float32),
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def cartesian_affine_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for tile_batch, tile_width in hl.tile([batch, width]):
        tmp[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        out[tile_batch, tile_width] = tmp[tile_batch, tile_width] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def size_one_view_chain(x: torch.Tensor) -> torch.Tensor:
    heads, width = x.size()
    tmp = torch.empty_like(x)
    viewed = tmp.unsqueeze(0)
    out = torch.empty_like(viewed)

    for tile_head in hl.tile(heads):
        tmp[tile_head, :] = x[tile_head, :] + 1
    for tile_batch, tile_head, tile_width in hl.tile([1, heads, width]):
        out[tile_batch, tile_head, tile_width] = (
            viewed[tile_batch, tile_head, tile_width] * 2
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def computed_index_view_chain(x: torch.Tensor) -> torch.Tensor:
    rows, width = x.size()
    tmp = torch.empty_like(x)
    tmp_groups = tmp.view(rows, width // 16, 16)
    x_groups = x.view(rows, width // 16, 16)
    out = torch.empty_like(x)

    # A computed (non-affine) group index, then a read through another view.
    for tile_row, tile_quarter in hl.tile([rows, width // 4], block_size=[1, 4]):
        group = tile_quarter.begin // 4
        tmp_groups[tile_row, group, :] = x_groups[tile_row, group, :] + 1
    for tile_row in hl.tile(rows, block_size=1):
        out[tile_row, :] = tmp[tile_row, :] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def overshooting_tile_chain(x: torch.Tensor) -> torch.Tensor:
    rows, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for tile_row, tile_col in hl.tile([rows, width]):
        tmp[tile_row, tile_col] = x[tile_row, tile_col] + 1
    for tile_row, tile_col in hl.tile([rows, width]):
        out[tile_row, tile_col] = tmp[tile_row, tile_col] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def three_way_affine_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    output_width = width // 3
    tmp = torch.empty_like(x)
    out = torch.empty((batch, output_width), dtype=x.dtype, device=x.device)

    for tile_batch, tile_width in hl.tile([batch, width]):
        tmp[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, output_width]):
        out[tile_batch, tile_width] = (
            tmp[tile_batch, tile_width]
            + tmp[tile_batch, tile_width + output_width]
            + tmp[tile_batch, tile_width + 2 * output_width]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def readiness_counter_chain(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.size()
    assert rows == 8
    assert columns == 4
    tmp = torch.empty_like(x)
    partial = torch.empty((rows // 2, columns), dtype=x.dtype, device=x.device)
    reduced = torch.empty((columns,), dtype=x.dtype, device=x.device)
    out = torch.empty((1,), dtype=x.dtype, device=x.device)

    for producer_row, producer_column in hl.tile([rows, columns], block_size=[1, 1]):
        tmp[producer_row, producer_column] = x[producer_row, producer_column] + 1
    for partial_row, partial_column in hl.tile([rows, columns], block_size=[2, 1]):
        partial[partial_row.id, partial_column] = torch.sum(
            tmp[partial_row, partial_column], dim=0
        )
    for final_row, final_column in hl.tile(
        [rows // 2, columns], block_size=[rows // 2, 1]
    ):
        reduced[final_column] = torch.sum(partial[final_row, final_column], dim=0)
    for output_index in hl.tile(1, block_size=1):
        out[output_index] = torch.sum(reduced[:], dim=-1)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def cartesian_affine_join(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    out = torch.empty_like(x)

    for tile_batch, tile_width in hl.tile([batch, width]):
        left[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        right[tile_batch, tile_width] = x[tile_batch, tile_width] - 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        out[tile_batch, tile_width] = (
            left[tile_batch, tile_width] + right[tile_batch, tile_width]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def singleton_root_join(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    out = torch.empty((batch,), dtype=torch.float32, device=x.device)

    for tile_batch, tile_width in hl.tile([batch, width]):
        left[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for tile_batch, tile_width in hl.tile([batch, width]):
        right[tile_batch, tile_width] = x[tile_batch, tile_width] - 1
    for tile_batch in hl.tile(batch, block_size=1):
        out[tile_batch] = torch.sum(
            left[tile_batch, :].to(torch.float32)
            + right[tile_batch, :].to(torch.float32),
            dim=-1,
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def streamed_singleton_reduction(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty((batch,), dtype=torch.float32, device=x.device)

    for producer_batch, producer_width in hl.tile([batch, width]):
        tmp[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch in hl.tile(batch, block_size=1):
        acc = hl.zeros([consumer_batch], dtype=torch.float32)
        for reduction_width in hl.tile(width, block_size=16):
            acc = acc + torch.sum(
                tmp[consumer_batch, reduction_width].to(torch.float32), dim=-1
            )
        out[consumer_batch] = acc + tmp[consumer_batch, 0].to(torch.float32)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def prewait_singleton_reduction(x: torch.Tensor) -> torch.Tensor:
    """Keep the scalar read before the nested waits as an ordering adversary."""
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty((batch,), dtype=torch.float32, device=x.device)

    for producer_batch, producer_width in hl.tile([batch, width]):
        tmp[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch in hl.tile(batch, block_size=1):
        first = tmp[consumer_batch, 0].to(torch.float32)
        acc = hl.zeros([consumer_batch], dtype=torch.float32)
        for reduction_width in hl.tile(width, block_size=16):
            acc = acc + torch.sum(
                tmp[consumer_batch, reduction_width].to(torch.float32), dim=-1
            )
        out[consumer_batch] = acc + first
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def singleton_gate_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    gate = torch.empty((batch,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)

    for tile_batch, tile_width in hl.tile([batch, width]):
        tmp[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    for gate_batch in hl.tile(batch, block_size=1):
        gate[gate_batch] = torch.sum(tmp[gate_batch, :].to(torch.float32), dim=-1)
    for tile_batch, tile_width in hl.tile([batch, width]):
        out[tile_batch, tile_width] = tmp[tile_batch, tile_width] * gate[tile_batch]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def fixed_block_dense_span_chain(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.size()
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(n, block_size=16):
        y[tile] = x[tile] + 1
    for row in hl.tile(n // 16, block_size=1):
        columns = row.begin * 16 + hl.arange(16)
        out[columns] = y[columns] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def streamed_sibling_reductions(x: torch.Tensor) -> torch.Tensor:
    """Exercise two independently ready nested sites in one consumer root task."""
    batch, width = x.size()
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    out = torch.empty((batch,), dtype=torch.float32, device=x.device)

    for producer_batch, producer_width in hl.tile([batch, width]):
        left[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for producer_batch, producer_width in hl.tile([batch, width]):
        right[producer_batch, producer_width] = x[producer_batch, producer_width] * 2
    for consumer_batch in hl.tile(batch, block_size=1):
        left_acc = hl.zeros([consumer_batch], dtype=torch.float32)
        for reduction_width in hl.tile(width, block_size=16):
            left_acc = left_acc + torch.sum(
                left[consumer_batch, reduction_width].to(torch.float32), dim=-1
            )
        right_acc = hl.zeros([consumer_batch], dtype=torch.float32)
        for reduction_width in hl.tile(width, block_size=16):
            right_acc = right_acc + torch.sum(
                right[consumer_batch, reduction_width].to(torch.float32), dim=-1
            )
        out[consumer_batch] = left_acc + right_acc
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def nested_store_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_batch in hl.tile(batch, block_size=1):
        for producer_width in hl.tile(width, block_size=16):
            tmp[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch, consumer_width in hl.tile([batch, width], block_size=[1, 16]):
        out[consumer_batch, consumer_width] = tmp[consumer_batch, consumer_width] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def conditional_payload_completion_chain(
    x: torch.Tensor,
    active_children: torch.Tensor,
) -> torch.Tensor:
    """Use an unconditional completion value to cover conditional payloads."""
    groups, fan_in, width = x.size()
    payload = torch.empty_like(x)
    completion = torch.empty((groups, fan_in), dtype=torch.int32, device=x.device)
    out = torch.empty((groups, width), dtype=torch.float32, device=x.device)

    for producer_group, producer_child, producer_width in hl.tile(
        [groups, fan_in, width], block_size=[1, 1, 32]
    ):
        active = hl.load(active_children, [producer_group.begin])
        if producer_child.begin < active:
            payload[producer_group, producer_child, producer_width] = (
                x[producer_group, producer_child, producer_width] + 1
            )
        completion[producer_group, producer_child] = 1

    for consumer_group, consumer_width in hl.tile([groups, width], block_size=[1, 32]):
        completed = torch.zeros([], dtype=torch.int32, device=x.device)
        for ready_child in hl.tile(fan_in, block_size=1):
            completed = completed + completion[consumer_group.begin, ready_child.begin]
        active = hl.load(active_children, [consumer_group.begin])
        acc = hl.zeros([consumer_width], dtype=torch.float32)
        child = torch.zeros([], dtype=torch.int32, device=x.device)
        while (child < active) & (completed == fan_in):
            acc = acc + payload[consumer_group.begin, child, consumer_width]
            child = child + 1
        out[consumer_group.begin, consumer_width] = acc
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def nested_early_completion_chain(x: torch.Tensor) -> torch.Tensor:
    """Do not let nested completion cover a later access in its parent."""
    batch, width = x.size()
    payload = torch.empty_like(x)
    completion = torch.empty_like(x, dtype=torch.int32)
    out = torch.empty_like(x)

    for producer_batch, producer_width in hl.tile([batch, width], block_size=[1, 32]):
        for _marker in hl.tile(1, block_size=1):
            completion[producer_batch.begin, producer_width] = 1
        payload[producer_batch.begin, producer_width] = (
            x[producer_batch.begin, producer_width] + 1
        )

    for consumer_batch, consumer_width in hl.tile([batch, width], block_size=[1, 32]):
        completed = torch.zeros([], dtype=torch.int32, device=x.device)
        for _ready in hl.tile(1, block_size=1):
            completed = completed + torch.sum(
                completion[consumer_batch.begin, consumer_width]
            ).to(torch.int32)
        if completed == consumer_width.block_size:
            out[consumer_batch.begin, consumer_width] = payload[
                consumer_batch.begin, consumer_width
            ]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def nested_load_store_chain(x: torch.Tensor) -> torch.Tensor:
    """Make one nested scope both a readiness consumer and a producer."""
    batch, width = x.size()
    first = torch.empty_like(x)
    second = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_batch, producer_width in hl.tile([batch, width]):
        first[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for middle_batch in hl.tile(batch, block_size=1):
        for middle_width in hl.tile(width, block_size=16):
            second[middle_batch, middle_width] = first[middle_batch, middle_width] * 2
    for consumer_batch, consumer_width in hl.tile([batch, width], block_size=[1, 16]):
        out[consumer_batch, consumer_width] = second[consumer_batch, consumer_width] + 3
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def pdl_store_chain(x: torch.Tensor) -> torch.Tensor:
    """Launch a cross-loop chain with PDL."""
    batch, width = x.size()
    first = torch.empty_like(x)
    out = torch.empty_like(x)

    hl.pdl_wait()
    for producer_batch, producer_width in hl.tile([batch, width]):
        first[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch, consumer_width in hl.tile([batch, width], block_size=[1, 16]):
        out[consumer_batch, consumer_width] = first[consumer_batch, consumer_width] * 2
    hl.pdl_launch_dependents()
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def pdl_unwaited_chain(x: torch.Tensor) -> torch.Tensor:
    """Trigger PDL without waiting, which races the cross-loop state."""
    first = torch.empty_like(x)
    out = torch.empty_like(x)

    hl.pdl_launch_dependents()
    for producer_batch, producer_width in hl.tile(x.size()):
        first[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch, consumer_width in hl.tile(x.size(), block_size=[1, 16]):
        out[consumer_batch, consumer_width] = first[consumer_batch, consumer_width] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def nested_two_axis_consumer(x: torch.Tensor) -> torch.Tensor:
    """Exercise conservative fallback for an unrendered two-axis action scope."""
    rows, columns = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_row, producer_column in hl.tile([rows, columns]):
        tmp[producer_row, producer_column] = x[producer_row, producer_column] + 1
    for _consumer in hl.tile(1, block_size=1):
        for consumer_row, consumer_column in hl.tile(
            [rows, columns], block_size=[8, 8]
        ):
            out[consumer_row, consumer_column] = tmp[consumer_row, consumer_column] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def offset_affine_chain(x: torch.Tensor) -> torch.Tensor:
    width = x.size(0)
    tmp = torch.empty_like(x)
    out = torch.empty((width - 32,), dtype=x.dtype, device=x.device)

    for producer_tile in hl.tile(32, width):
        tmp[producer_tile] = x[producer_tile] + 1
    for consumer_tile in hl.tile(width - 32):
        out[consumer_tile] = tmp[consumer_tile + 32] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def partial_prefix_continuation(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = x.size(0)
    tmp = torch.empty_like(x)
    out = torch.empty((width - 32,), dtype=x.dtype, device=x.device)

    for producer_tile in hl.tile(width):
        tmp[producer_tile] = x[producer_tile] + 1
    for consumer_tile in hl.tile(width - 32):
        out[consumer_tile] = tmp[consumer_tile] * 2
    return tmp, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def partial_prefix_in_place_chain(x: torch.Tensor) -> torch.Tensor:
    width = x.size(0)
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_tile in hl.tile(width):
        tmp[producer_tile] = x[producer_tile] + 1
    for prefix_tile in hl.tile(width - 32):
        tmp[prefix_tile] = tmp[prefix_tile] * 2
    for output_tile in hl.tile(width):
        out[output_tile] = tmp[output_tile]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def multi_producer_join(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    left = torch.empty_like(x)
    right = torch.empty_like(y)
    out = torch.empty_like(x)

    for tile in hl.tile(x.size(0)):
        left[tile] = x[tile] + 1
    for tile in hl.tile(y.size(0)):
        right[tile] = y[tile] * 2
    for tile in hl.tile(x.size(0)):
        out[tile] = left[tile] + right[tile]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def coalesced_multi_producer_join(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    heads, width = x.shape
    splits = 4
    left = torch.empty_like(x)
    right = torch.empty_like(y)
    out = torch.empty((splits, heads, width), dtype=x.dtype, device=x.device)

    for tile_head, tile_width in hl.tile([heads, width], block_size=[1, 1]):
        left[tile_head, tile_width] = x[tile_head, tile_width] + 1
    for tile_head in hl.tile(heads, block_size=1):
        right[tile_head] = y[tile_head] * 2
    for tile_split, tile_head, tile_width in hl.tile(
        [splits, heads, width], block_size=[1, 1, width]
    ):
        out[tile_split, tile_head, tile_width] = (
            left[tile_head, tile_width]
            + right[tile_head][:, None]
            + tile_split.index[:, None, None]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def coalesced_single_producer_fanout(x: torch.Tensor) -> torch.Tensor:
    heads, width = x.shape
    splits = 4
    tmp = torch.empty_like(x)
    out = torch.empty((splits, heads, width), dtype=x.dtype, device=x.device)

    for tile_head, tile_width in hl.tile([heads, width], block_size=[1, 1]):
        tmp[tile_head, tile_width] = x[tile_head, tile_width] + 1
    for tile_split, tile_head, tile_width in hl.tile(
        [splits, heads, width], block_size=[1, 1, width]
    ):
        out[tile_split, tile_head, tile_width] = (
            tmp[tile_head, tile_width] + tile_split.index[:, None, None]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def inline_then_two_rows(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    c = torch.empty((rows // 2, cols), dtype=x.dtype, device=x.device)
    d = torch.empty_like(x)
    for t in hl.tile(rows, block_size=1):
        a[t, :] = x[t, :] + 1
    for t in hl.tile(rows, block_size=1):
        v = a[t, :]
        b[t, :] = torch.sin(v) * torch.cos(v) + torch.exp(torch.sin(v * 3)) * 2
    for t in hl.tile(rows, block_size=2):
        c[t.id, :] = torch.sum(b[t, :], dim=0)
    for t in hl.tile(rows, block_size=4):
        d[t, :] = b[t, :] + 1
    return d


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def split_norm_then_scale(x: torch.Tensor) -> torch.Tensor:
    rows, splits, width = x.shape
    sums = torch.empty((splits, rows), dtype=x.dtype, device=x.device)
    doubled = torch.empty_like(x)
    scale = torch.empty((rows,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)

    for tile_row, tile_split in hl.tile([rows, splits], block_size=[1, 1]):
        values = x[tile_row.begin, tile_split.begin, :] * 2
        doubled[tile_row.begin, tile_split.begin, :] = values
        sums[tile_split.begin, tile_row.begin] = torch.sum(values)
    for tile_row in hl.tile(rows, block_size=1):
        scale[tile_row] = torch.sum(sums[:, tile_row], dim=0) + 1
    for tile_row, tile_split in hl.tile([rows, splits], block_size=[1, 1]):
        out[tile_row.begin, tile_split.begin, :] = (
            doubled[tile_row.begin, tile_split.begin, :] * scale[tile_row.begin]
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def shard_of_whole_rows(x: torch.Tensor, shard: hl.constexpr) -> torch.Tensor:
    rows, splits, width = x.shape
    own = splits // 2
    y = torch.empty((2 * rows, splits, width), dtype=x.dtype, device=x.device)
    out = torch.empty((own, width), dtype=x.dtype, device=x.device)

    for tile_row, tile_split in hl.tile([rows, splits], block_size=[1, 1]):
        values = x[tile_row.begin, tile_split.begin, :]
        y[tile_row.begin, tile_split.begin, :] = values * 2
        y[tile_row.begin + rows, tile_split.begin, :] = values * 3
    for tile_split, tile_col in hl.tile([own, width], block_size=[1, 16]):
        out[tile_split.begin, tile_col] = torch.sum(
            y[:, shard * own + tile_split.begin, tile_col], dim=0
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def keyed_rows_beside_whole_read(x: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    totals = torch.empty((2, width), dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)
    out = torch.empty_like(x)

    for tile_half in hl.tile(2, block_size=1):
        totals[tile_half.begin, :] = x[tile_half.begin, :] + 1
    for tile_row in hl.tile(rows, block_size=1):
        y[tile_row.begin, :] = x[tile_row.begin, :] * 2
    for tile_row in hl.tile(rows, block_size=1):
        # Every task reads all of the first root, and its own row of the second.
        out[tile_row.begin, :] = y[tile_row.begin, :] + torch.sum(totals[:, :], dim=0)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def join_of_whole_reads(x: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows, block_size=1):
        a[tile_r.begin, :] = x[tile_r.begin, :] + 1
    for tile_r in hl.tile(rows, block_size=1):
        b[tile_r.begin, :] = x[tile_r.begin, :] * 2
    for tile_r in hl.tile(rows, block_size=1):
        # Every task reads both producers whole: one wait serves them all.
        whole = torch.sum(a[:, :], dim=0) + torch.sum(b[:, :], dim=0)
        out[tile_r.begin, :] = whole + x[tile_r.begin, :]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def guarded_dot_chain(
    x: torch.Tensor, w: torch.Tensor, w2: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    rows, k = x.size()
    staged = torch.zeros([rows, w.size(1)], dtype=torch.float16, device=x.device)
    out = torch.empty([rows, w2.size(1)], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows, block_size=16):
        # Every write of the task is under its trailing ``if``.
        if counts[tile_r.begin] > 0:
            acc = hl.zeros([tile_r, w.size(1)], dtype=torch.float32)
            for tile_k in hl.tile(k, block_size=32):
                acc = hl.dot(x[tile_r, tile_k], w[tile_k, :], acc=acc)
            staged[tile_r, :] = acc.to(torch.float16)
    for tile_o in hl.tile(rows, block_size=32):
        out[tile_o, :] = hl.dot(staged[tile_o, :], w2[:, :])
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def zeroed_then_guarded_chain(
    x: torch.Tensor, w: torch.Tensor, w2: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    rows, k = x.size()
    staged = torch.empty([rows, w.size(1)], dtype=torch.float16, device=x.device)
    out = torch.empty([rows, w2.size(1)], dtype=torch.float32, device=x.device)
    for tile_z in hl.tile(rows, block_size=32):
        staged[tile_z, :] = hl.zeros([tile_z, w.size(1)], dtype=torch.float16)
    for tile_r in hl.tile(rows, block_size=16):
        # Writes the zeroed rows again, but only where counts allow.
        if counts[tile_r.begin] > 0:
            acc = hl.zeros([tile_r, w.size(1)], dtype=torch.float32)
            for tile_k in hl.tile(k, block_size=32):
                acc = hl.dot(x[tile_r, tile_k], w[tile_k, :], acc=acc)
            staged[tile_r, :] = acc.to(torch.float16)
    for tile_o in hl.tile(rows, block_size=32):
        out[tile_o, :] = hl.dot(staged[tile_o, :], w2[:, :])
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def scaled_then_guarded_chain(
    x: torch.Tensor, w: torch.Tensor, w2: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    rows, k = x.size()
    scaled = torch.empty_like(x)
    staged = torch.zeros([rows, w.size(1)], dtype=torch.float16, device=x.device)
    out = torch.empty([rows, w2.size(1)], dtype=torch.float32, device=x.device)
    for tile_p in hl.tile(rows, block_size=16):
        scaled[tile_p, :] = x[tile_p, :] * 2
    for tile_r in hl.tile(rows, block_size=16):
        if counts[tile_r.begin] > 0:
            acc = hl.zeros([tile_r, w.size(1)], dtype=torch.float32)
            for tile_k in hl.tile(k, block_size=32):
                acc = hl.dot(scaled[tile_r, tile_k], w[tile_k, :], acc=acc)
            staged[tile_r, :] = acc.to(torch.float16)
    for tile_o in hl.tile(rows, block_size=32):
        out[tile_o, :] = hl.dot(staged[tile_o, :], w2[:, :])
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def scaled_then_self_guarded_chain(
    x: torch.Tensor, w: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    rows, k = x.size()
    scaled = torch.empty_like(x)
    staged = torch.zeros([rows, w.size(1)], dtype=torch.float16, device=x.device)
    out = torch.empty([rows, w2.size(1)], dtype=torch.float32, device=x.device)
    for tile_p in hl.tile(rows, block_size=16):
        scaled[tile_p, :] = x[tile_p, :] * 2
    for tile_r in hl.tile(rows, block_size=16):
        # The guard reads the producer's output.
        if scaled[tile_r.begin, 0] > 0:
            acc = hl.zeros([tile_r, w.size(1)], dtype=torch.float32)
            for tile_k in hl.tile(k, block_size=32):
                acc = hl.dot(scaled[tile_r, tile_k], w[tile_k, :], acc=acc)
            staged[tile_r, :] = acc.to(torch.float16)
    for tile_o in hl.tile(rows, block_size=32):
        out[tile_o, :] = hl.dot(staged[tile_o, :], w2[:, :])
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def masked_rows_chain(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    staged = torch.zeros_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.store(staged, [tile], x[tile] + 1, extra_mask=keep[tile] != 0)
    for tile in hl.tile(x.size(0)):
        out[tile] = staged[tile] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def flat_load_after_store(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    staged = torch.empty_like(x)
    out = torch.empty((offsets.size(0), 16), dtype=torch.float16, device=x.device)
    for tile in hl.tile(x.size(0)):
        staged[tile] = x[tile] + 1
    for tile in hl.tile(offsets.size(0)):
        # Reads ``staged`` at flat offsets, not through a subscript.
        lanes = hl.load_bfloat16_x16_to_float16(staged, offsets[tile])
        for i in hl.static_range(16):
            out[tile, i] = lanes[i]
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def regrouped_view_of_whole_rows(x: torch.Tensor, shard: hl.constexpr) -> torch.Tensor:
    rows, splits, width = x.shape
    y = torch.empty((2 * rows, splits, width), dtype=x.dtype, device=x.device)
    # The consumer's view regroups the producer's (split, column) axes.
    regrouped = y.view(2 * rows, 2, splits // 2, width)
    out = torch.empty((splits // 2, width), dtype=x.dtype, device=x.device)

    for tile_row, tile_split in hl.tile([rows, splits], block_size=[1, 1]):
        values = x[tile_row.begin, tile_split.begin, :]
        y[tile_row.begin, tile_split.begin, :] = values * 2
        y[tile_row.begin + rows, tile_split.begin, :] = values * 3
    for tile_split in hl.tile(splits // 2, block_size=1):
        out[tile_split.begin, :] = torch.sum(
            regrouped[:, shard, tile_split.begin, :], dim=0
        )
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def direct_nested_continuation(x: torch.Tensor) -> torch.Tensor:
    width = x.size(0)
    tmp = torch.empty_like(x)
    reduced = torch.empty((width // 2,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(reduced)

    for producer_tile in hl.tile(width, block_size=1):
        tmp[producer_tile] = x[producer_tile] + 1
    for reduced_tile in hl.tile(width, block_size=2):
        reduced[reduced_tile.id] = torch.sum(tmp[reduced_tile], dim=-1)
    for output_tile in hl.tile(width // 2, block_size=1):
        out[output_tile] = reduced[output_tile] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def mixed_radix_continuation(x: torch.Tensor) -> torch.Tensor:
    slots, gate_up_size = x.size()
    intermediate = gate_up_size // 2
    hl.specialize(slots)
    hl.specialize(gate_up_size)
    hl.specialize(intermediate)

    gate_up = torch.empty_like(x)
    activation = torch.empty(
        (slots, intermediate),
        dtype=x.dtype,
        device=x.device,
    )
    flat_x = x.view(slots * gate_up_size)
    flat_gate_up = gate_up.view(slots * gate_up_size)

    for producer_tile in hl.tile(slots * gate_up_size, block_size=16):
        flat_gate_up[producer_tile] = flat_x[producer_tile] + 1.0

    for slot, activation_block in hl.tile(
        [slots, intermediate],
        block_size=[1, 256],
    ):
        gate = gate_up[slot, activation_block].to(torch.float32)
        up = gate_up[slot, activation_block + intermediate].to(torch.float32)
        activation[slot, activation_block] = (gate * torch.sigmoid(gate) * up).to(
            x.dtype
        )
    return activation


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def runtime_bound_store_chain(
    x: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    rows, columns = x.size()
    tmp = torch.zeros_like(x)
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    for producer_tile in hl.tile(rows, block_size=1):
        row = producer_tile.begin
        for column in hl.grid(starts[row], ends[row]):
            tmp[row, column] = x[row, column] * 2
    for consumer_tile in hl.tile(rows, block_size=1):
        out[consumer_tile] = torch.sum(tmp[consumer_tile, :], dim=-1)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def runtime_bound_loop_chain(
    x: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    rows, columns = x.size()
    tmp = torch.empty([rows], dtype=x.dtype, device=x.device)
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    for producer_tile in hl.tile(rows, block_size=1):
        row = producer_tile.begin
        acc = hl.zeros([1], dtype=torch.float32)
        for column in hl.grid(starts[row], ends[row]):
            acc = acc + x[row, column]
        tmp[producer_tile] = acc
    for consumer_tile in hl.tile(rows, block_size=1):
        out[consumer_tile] = tmp[consumer_tile] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def guarded_prefix_chain(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, columns = x.size()
    tmp = torch.zeros_like(x)
    out = torch.empty_like(x)
    side = torch.zeros_like(x)
    for tile_c, tile_r in hl.tile([columns, rows], block_size=[64, 1]):
        live = torch.sum((flags[:] > 0).to(torch.int32))
        if tile_r.begin < live:
            tmp[tile_r, tile_c] = x[tile_r, tile_c] + 1
    for tile_r in hl.tile(rows, block_size=4):
        out[tile_r, :] = tmp[tile_r, :] * 2 + tmp[0, :][None, :]
    for tile_s in hl.tile(rows, block_size=2):
        side_live = torch.sum((flags[:] > 0).to(torch.int32))
        if tile_s.begin < side_live:
            side[tile_s, :] = x[tile_s, :] * 3
    return tmp, out, side


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def computed_bound_chain(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.size()
    marks = torch.empty_like(flags)
    out = torch.zeros_like(x)
    for tile_f in hl.tile(flags.size(0), block_size=4):
        marks[tile_f] = (flags[tile_f] > 0).to(torch.int32)
    # The bound is this kernel's own result: a whole-root dependency.
    for tile_c, tile_r in hl.tile([columns, rows], block_size=[64, 1]):
        live = torch.sum(marks[:])
        if tile_r.begin < live:
            out[tile_r, tile_c] = x[tile_r, tile_c] + 1
    return marks, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def overwritten_bound_chain(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.size()
    marks = torch.empty_like(flags)
    out = torch.zeros_like(x)
    for tile_f in hl.tile(flags.size(0), block_size=4):
        marks[tile_f] = (flags[tile_f] > 0).to(torch.int32)
    for tile_c, tile_r in hl.tile([columns, rows], block_size=[64, 1]):
        live = torch.sum(marks[:])
        if tile_r.begin < live:
            out[tile_r, tile_c] = x[tile_r, tile_c] + 1
    # A later root overwrites the bound.
    for tile_f in hl.tile(flags.size(0), block_size=4):
        marks[tile_f] = marks[tile_f] * 2
    return marks, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def guarded_readiness_fanout(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.size()
    splits = 4
    tmp = torch.zeros_like(x)
    out = torch.empty((splits, rows, columns), dtype=x.dtype, device=x.device)
    for tile_c, tile_r in hl.tile([columns, rows], block_size=[64, 1]):
        live = torch.sum((flags[:] > 0).to(torch.int32))
        if tile_r.begin < live:
            tmp[tile_r, tile_c] = x[tile_r, tile_c] + 1
    for tile_s, tile_r in hl.tile([splits, rows], block_size=[1, 1]):
        out[tile_s.begin, tile_r, :] = tmp[tile_r, :] * 2 + tile_s.begin
    return tmp, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def barrier_after_weight_load(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.size()
    act = torch.empty_like(x)
    out = torch.empty((n,), dtype=x.dtype, device=x.device)
    for tile_k in hl.tile(k, block_size=64):
        act[tile_k] = x[tile_k] + 1
    for tile_n in hl.tile(n, block_size=16):
        weight = w[tile_n, :]
        out[tile_n] = torch.sum(weight * act[None, :], dim=1)
    return act, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def guarded_barrier_consumer(
    x: torch.Tensor, w: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = w.size()
    act = torch.empty_like(x)
    out = torch.zeros((n,), dtype=x.dtype, device=x.device)
    res = torch.empty_like(out)
    for tile_k in hl.tile(k, block_size=64):
        act[tile_k] = x[tile_k] + 1
    for tile_n in hl.tile(n, block_size=1):
        live = torch.sum((flags[:] > 0).to(torch.int32))
        if tile_n.begin < live:
            weight = w[tile_n, :]
            out[tile_n] = torch.sum(weight * act[None, :], dim=1)
    for tile_m in hl.tile(n, block_size=1):
        res[tile_m] = out[tile_m] * 2
    return act, out, res


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def guarded_scan_gather(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.size()
    picked = torch.zeros_like(x)
    out = torch.empty_like(x)
    for tile_c, tile_r in hl.tile([columns, rows], block_size=[64, 1]):
        member = (flags[:] > 0).to(torch.int32)
        rank = hl.cumsum(member, dim=0) - 1
        live = torch.sum(member)
        if tile_r.begin < live:
            hit = (member == 1) & (rank == tile_r.begin)
            src = torch.sum(torch.where(hit, hl.arange(rows), 0))
            picked[tile_r, tile_c] = x[src, tile_c][None, :]
    for tile_r in hl.tile(rows, block_size=4):
        out[tile_r, :] = picked[tile_r, :] * 2 + picked[0, :][None, :]
    return picked, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def split_merge_chain(x: torch.Tensor) -> torch.Tensor:
    keys, splits = x.size()
    partial = torch.empty_like(x)
    out = torch.empty([keys], dtype=x.dtype, device=x.device)
    for key_tile, split_tile in hl.tile([keys, splits], block_size=[1, 1]):
        partial[key_tile, split_tile] = x[key_tile, split_tile] * 2
    for key_tile in hl.tile(keys, block_size=1):
        out[key_tile] = torch.sum(partial[key_tile, :], dim=-1)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def conditional_split_merge_chain(
    x: torch.Tensor, active: torch.Tensor, early: hl.constexpr
) -> torch.Tensor:
    keys, rows, splits = x.size()
    partial = torch.empty_like(x)
    lse = torch.empty_like(x)
    out = torch.empty([keys, rows], dtype=x.dtype, device=x.device)
    for key_tile, split_tile in hl.tile([keys, splits], block_size=[1, 1]):
        key = key_tile.begin
        split = split_tile.begin
        partial[key, :, split] = x[key, :, split] * 2
        lse[key, :, split] = x[key, :, split] + 1
    for merge_tile in hl.tile(keys, block_size=1):
        merge_key = merge_tile.begin
        top = lse[merge_key, :, 0]
        for index in hl.static_range(1, splits):
            top = torch.maximum(top, lse[merge_key, :, index])
        acc = hl.zeros([rows], dtype=torch.float32)
        for rank in hl.static_range(early):
            acc = acc + partial[merge_key, :, rank] * (lse[merge_key, :, rank] - top)
        # Every split is read twice; the late ones only under a runtime branch.
        if active[0] > early:
            for late in hl.static_range(early, splits):
                weight = lse[merge_key, :, late] - top
                acc = acc + partial[merge_key, :, late] * weight
        out[merge_key, :] = acc
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def split_merge_then_independent(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys, splits = x.size()
    partial = torch.empty_like(x)
    out = torch.empty([keys], dtype=x.dtype, device=x.device)
    side = torch.empty_like(y)
    other = torch.empty_like(z)
    for key_tile, split_tile in hl.tile([keys, splits], block_size=[1, 1]):
        partial[key_tile, split_tile] = x[key_tile, split_tile] * 2
    for key_tile in hl.tile(keys, block_size=1):
        out[key_tile] = torch.sum(partial[key_tile, :], dim=-1)
    for side_tile in hl.tile(y.size(0), block_size=1):
        side[side_tile] = y[side_tile] + 1
    for other_tile in hl.tile(z.size(0), block_size=1):
        other[other_tile] = z[other_tile] - 1
    return out, side, other


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def split_independent_merge(
    x: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    keys, splits = x.size()
    partial = torch.empty_like(x)
    out = torch.empty([keys], dtype=x.dtype, device=x.device)
    other = torch.empty_like(z)
    for key_tile, split_tile in hl.tile([keys, splits], block_size=[1, 1]):
        partial[key_tile, split_tile] = x[key_tile, split_tile] * 2
    for other_tile in hl.tile(z.size(0), block_size=1):
        other[other_tile] = z[other_tile] - 1
    for key_tile in hl.tile(keys, block_size=1):
        out[key_tile] = torch.sum(partial[key_tile, :], dim=-1)
    return out, other


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def specialized_quotient_chain(
    x: torch.Tensor,
    numerator: int,
    denominator: int,
) -> torch.Tensor:
    hl.specialize(numerator)
    hl.specialize(denominator)
    width = numerator // denominator
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_tile in hl.tile(width, block_size=1):
        tmp[producer_tile] = x[producer_tile] + 1
    for consumer_tile in hl.tile(width, block_size=1):
        out[consumer_tile] = tmp[consumer_tile] * 2
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def split_keyed_and_fan_in_consumers(
    x: torch.Tensor, rank: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, splits, width = x.size()
    own = splits // 2
    act = torch.empty([2 * tokens, splits, width], dtype=x.dtype, device=x.device)
    act2 = act.view(2, tokens, splits, width)
    branch = torch.empty_like(x)
    sums = torch.empty([splits, tokens], dtype=torch.float32, device=x.device)
    rows = torch.empty(
        [own, width // 128, 2 * tokens], dtype=torch.float32, device=x.device
    )
    out = torch.empty_like(x)
    for t, s in hl.tile([tokens, splits], block_size=[2, 1]):
        v = x[t, s.begin, :] * 2.0
        branch[t, s.begin, :] = v
        act2[0, t, s.begin, :] = v + 1.0
        act2[1, t, s.begin, :] = v - 1.0
        sums[s.begin, t] = torch.sum(v, dim=-1)
    for s1, k in hl.tile([own, width], block_size=[1, 128]):
        rows[s1.begin, k.id, :] = torch.sum(act[:, rank * own + s1.begin, k], dim=-1)
    for t2, s2 in hl.tile([tokens, splits], block_size=[1, 1]):
        total = torch.sum(sums[:, t2.begin], dim=0)
        out[t2, s2.begin, :] = branch[t2, s2.begin, :] * total
    return rows, out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def fan_in_and_split_keyed_consumers(
    x: torch.Tensor, rank: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, splits, width = x.size()
    own = splits // 2
    act = torch.empty([2 * tokens, splits, width], dtype=x.dtype, device=x.device)
    act2 = act.view(2, tokens, splits, width)
    branch = torch.empty_like(x)
    sums = torch.empty([splits, tokens], dtype=torch.float32, device=x.device)
    rows = torch.empty(
        [own, width // 128, 2 * tokens], dtype=torch.float32, device=x.device
    )
    out = torch.empty_like(x)
    for t, s in hl.tile([tokens, splits], block_size=[2, 1]):
        v = x[t, s.begin, :] * 2.0
        branch[t, s.begin, :] = v
        act2[0, t, s.begin, :] = v + 1.0
        act2[1, t, s.begin, :] = v - 1.0
        sums[s.begin, t] = torch.sum(v, dim=-1)
    for t2, s2 in hl.tile([tokens, splits], block_size=[1, 1]):
        total = torch.sum(sums[:, t2.begin], dim=0)
        out[t2, s2.begin, :] = branch[t2, s2.begin, :] * total
    for s1, k in hl.tile([own, width], block_size=[1, 128]):
        rows[s1.begin, k.id, :] = torch.sum(act[:, rank * own + s1.begin, k], dim=-1)
    return rows, out


class TestCrossLoopCodegenHelpers(TestCase):
    def test_scope_wait_passes_only_independent_loads(self) -> None:
        wait = ast.parse(
            "w = tl.inline_asm_elementwise(asm='x', constraints='=r,l', "
            "args=[state + key], dtype=tl.uint32, is_pure=False, pack=1)"
        ).body
        allocations = {"a": 1, "a_view": 1, "b": 2}

        def placed(source: str) -> int:
            body = ast.parse(source).body
            _scope_wait(body, wait, frozenset({1}), frozenset({"state"}), allocations)
            return next(i for i, s in enumerate(body) if s is wait[0])

        # Index math and loads of uncovered tensors pass the wait; a covered
        # tensor, under any of its argument names, stops it.
        self.assertEqual(
            placed("i = pid * 4\nx = tl.load(b + i)\ny = tl.load(a + i)"), 2
        )
        self.assertEqual(placed("x = tl.load(b + 1)\ny = a_view.shape"), 1)
        for stop in (
            "tl.store(b + i, x)",
            "tl.atomic_add(b + i, 1)",
            "helper(b)",
            "d.load([0])",
            "x = tl.load(q + 1)",  # an address naming no tensor argument
            "x = tl.load(tl.load(b + 0).to(tl.pointer_type(tl.float32)) + 1)",
            "x = tl.load(state + 1)",
            "key = 3",  # reassigns what the wait reads
            (
                "z = tl.inline_asm_elementwise(asm='y', constraints='=r', args=[], "
                "dtype=tl.int32, is_pure=False, pack=1)"
            ),
        ):
            self.assertEqual(
                placed(f"i = pid * 4\n{stop}\ny = tl.load(a + i)"), 1, stop
            )
        # Nothing touches what it orders: the wait still runs, last.
        self.assertEqual(placed("i = pid * 4"), 1)

    def test_batch_waits_polls_adjacent_spins_together(self) -> None:
        """Adjacent spins issue every first load before any spin loop, and
        adjacent wait-only guards of one test merge; anything else splits."""

        def spin(name: str, counter: str, fence: bool = True) -> str:
            load = (
                f"tl.inline_asm_elementwise(asm='ld.acquire.gpu.global.u32 $0, [$1];', "
                f"constraints='=r,l', args=[{counter}], dtype=tl.uint32, "
                "is_pure=False, pack=1)"
            )
            asm = "tl.inline_asm_elementwise(asm='{} mov.u32 $0, $1;', constraints='=r,r', args=[v], dtype=tl.uint32, is_pure=False, pack=1)"
            lines = [
                f"{name} = {load}",
                f"while {name} != t:\n    {name} = {load}",
                f"{name}_sync = " + asm.format("bar.warp.sync 0xffffffff;"),
            ]
            if fence:
                lines.append(
                    f"{name}_fence = " + asm.format("fence.proxy.async.global;")
                )
            return "\n".join(lines)

        names = {
            f"{w}{suffix}"
            for w in ("a", "b", "c")
            for suffix in ("", "_sync", "_fence")
        }
        df = SimpleNamespace(sync_names=lambda *kinds: names)

        def batched(source: str) -> list[str]:
            body = ast.parse(source).body
            out = cross_loop_codegen._batch_waits(cast("Any", df), body)
            return [ast.unparse(s).split(" = ")[0].split("\n")[0] for s in out]

        self.assertEqual(
            batched(
                f"{spin('a', 's + 0')}\n{spin('b', 's + 32', False)}\nx = tl.load(p)"
            ),
            [
                "a",
                "b",
                "while a != t:",
                "while b != t:",
                "a_sync",
                "a_fence",
                "b_sync",
                "x",
            ],
        )
        # A statement between two spins keeps them apart.
        self.assertEqual(
            batched(f"{spin('a', 's')}\ny = 1\n{spin('b', 's')}")[:3],
            ["a", "while a != t:", "a_sync"],
        )
        guarded = [
            "if pid == 0:\n    " + spin(w, "s").replace("\n", "\n    ") for w in "ab"
        ]
        out = cross_loop_codegen._batch_waits(
            cast("Any", df), ast.parse("\n".join(guarded)).body
        )
        self.assertEqual(len(out), 1)
        assert isinstance(out[0], ast.If)
        self.assertEqual(
            [ast.unparse(s).split(" = ")[0].split("\n")[0] for s in out[0].body][:4],
            ["a", "b", "while a != t:", "while b != t:"],
        )
        # A guard whose test reads what the first body assigns does not merge.
        feeding = [guarded[0], guarded[1].replace("pid == 0", "a == 0")]
        out = cross_loop_codegen._batch_waits(
            cast("Any", df), ast.parse("\n".join(feeding)).body
        )
        self.assertEqual(len(out), 2)
        # Merged guards evaluate their test once: only a pure test merges (a
        # load after the first spin's acquire may read a newer value).
        for test, merges in (
            ("tl.program_id(0) == 0", True),
            ("tl.load(f) == 0", False),
        ):
            same = [g.replace("pid == 0", test) for g in guarded]
            out = cross_loop_codegen._batch_waits(
                cast("Any", df), ast.parse("\n".join(same)).body
            )
            self.assertEqual(len(out), 1 if merges else 2, test)

    def test_guard_hoisting_preserves_dtype_checks(self) -> None:
        indexing = "pid_0 = virtual_pid\noffset_0 = pid_0\n"
        invariant = (
            "indices = tl.arange(0, 16)\n"
            "member = tl.load(flags + indices)\n"
            "member = tl.cast(member, tl.int32)\n"
            "tl.static_assert(member.dtype == tl.int32)\n"
            "live = tl.sum(member, 0)\n"
            "live = tl.cast(live, tl.int64)\n"
            "tl.static_assert(live.dtype == tl.int64)\n"
        )
        task = (
            "value = tl.load(x + offset_0)\n"
            "value = tl.cast(value, tl.float32)\n"
            "tl.static_assert(value.dtype == tl.float32)\n"
            "tl.store(out + offset_0, value)\n"
        )
        guarded = (
            "predicate = live > offset_0\n"
            "if predicate:\n"
            + "".join(f"    {line}\n" for line in task.splitlines())
            + "else:\n    pass\n"
        )

        def extent(
            source: str, read_only: bool = True
        ) -> cross_loop_codegen._GuardedExtent | None:
            return cross_loop_codegen._guarded_extent(
                ast.parse(source).body,
                slowest=PIDInfo("pid_0", "1", "16", 0),
                block_size=1,
                inner_tasks=1,
                variant_names=frozenset({"virtual_pid"}),
                read_only_tensors=frozenset({"flags"}) if read_only else frozenset(),
                tensor_names=frozenset({"flags", "x", "out"}),
            )

        result = extent(indexing + invariant + guarded)
        assert result is not None
        self.assertEqual(result.live_tasks, "live")
        self.assertEqual(result.hoisted_names, ("indices", "member", "live"))
        self.assertEqual(
            _ast_fingerprint(result.hoisted),
            _ast_fingerprint(ast.parse(invariant).body),
        )
        self.assertEqual(
            _ast_fingerprint(result.body),
            _ast_fingerprint(ast.parse(indexing + task).body),
        )
        # Bounds from mutable memory or task-varying loads cannot move.
        self.assertIsNone(extent(indexing + invariant + guarded, read_only=False))
        self.assertIsNone(
            extent(
                indexing
                + invariant.replace("flags + indices", "flags + offset_0")
                + guarded
            )
        )
        # Narrowing the offset can wrap it, so it no longer denotes the tile begin.
        self.assertIsNone(
            extent(
                indexing
                + "offset_0 = tl.cast(offset_0, tl.int8)\n"
                + "tl.static_assert(offset_0.dtype == tl.int8)\n"
                + invariant
                + guarded
            )
        )
        # Ordinary updates and intervening uses follow the same dataflow rules.
        for reassignment in (
            "live = live + 1\n",
            "other = live\nlive = tl.cast(live, tl.int64)\n",
            "tl.static_assert(live.dtype == tl.int64, 'expected int64')\n",
        ):
            with self.subTest(reassignment=reassignment):
                result = extent(indexing + invariant + reassignment + guarded)
                assert result is not None
                self.assertEqual(
                    _ast_fingerprint(result.hoisted),
                    _ast_fingerprint(ast.parse(invariant + reassignment).body),
                )

    def test_guard_hoisting_preserves_reaching_definitions(self) -> None:
        indexing = "pid_0 = virtual_pid\noffset_0 = pid_0\n"
        invariant = (
            "live = 2\n"
            "earlier = live\n"
            "live = live + 1\n"
            "bound = live\n"
            "live = live + 2\n"
            "tl.static_assert(bound == 3, 'bound keeps the earlier value')\n"
        )
        guarded = (
            "predicate = bound > offset_0\n"
            "if predicate:\n"
            "    out[offset_0] = earlier + live\n"
            "else:\n"
            "    pass\n"
        )

        def extent(source: str) -> cross_loop_codegen._GuardedExtent | None:
            return cross_loop_codegen._guarded_extent(
                ast.parse(source).body,
                slowest=PIDInfo("pid_0", "1", "16", 0),
                block_size=1,
                inner_tasks=1,
                variant_names=frozenset({"virtual_pid"}),
                read_only_tensors=frozenset(),
                tensor_names=frozenset(),
            )

        def execute(statements: list[ast.stmt], namespace: dict[str, Any]) -> None:
            module = ast.fix_missing_locations(ast.Module(statements, type_ignores=[]))
            exec(compile(module, "<guard-hoisting-test>", "exec"), namespace)

        source = indexing + invariant + guarded
        result = extent(source)
        assert result is not None
        self.assertEqual(result.live_tasks, "bound")
        self.assertEqual(result.hoisted_names, ("live", "earlier", "bound"))
        reference: dict[str, Any] = {
            "out": [0] * 16,
            "tl": SimpleNamespace(static_assert=self.assertTrue),
        }
        transformed: dict[str, Any] = {
            "out": [0] * 16,
            "tl": reference["tl"],
        }
        for pid in range(16):
            reference["virtual_pid"] = pid
            execute(ast.parse(source).body, reference)
        execute(result.hoisted, transformed)
        for pid in range(transformed[result.live_tasks]):
            transformed["virtual_pid"] = pid
            execute(result.body, transformed)
        self.assertEqual(reference["out"], [7] * 3 + [0] * 13)
        self.assertEqual(transformed["out"], reference["out"])

        # A task-dependent reader of an older value cannot move past its update.
        self.assertIsNone(
            extent(source.replace("earlier = live", "earlier = live + offset_0"))
        )
        self.assertIsNone(
            extent(source.replace("live = live + 1", "live = live + offset_0"))
        )
        self.assertIsNone(
            extent(
                source.replace("bound = live", "tl.store(output, live)\nbound = live")
            )
        )

    def test_blackwell_dot_root_stays_in_kernel_scope(self) -> None:
        dot_body = ast.parse("acc = tl.dot(lhs, rhs, acc=acc)\n").body
        scaled_body = ast.parse(
            "acc = tl.dot_scaled(lhs, lhs_scale, 'e4m3', rhs, rhs_scale, 'e4m3')\n"
        ).body
        ordinary_body = ast.parse("value = tl.load(pointer)\n").body

        self.assertTrue(_triton_root_requires_kernel_scope(dot_body, (10, 0)))
        self.assertTrue(_triton_root_requires_kernel_scope(scaled_body, (10, 0)))
        self.assertFalse(_triton_root_requires_kernel_scope(dot_body, (9, 0)))
        self.assertFalse(_triton_root_requires_kernel_scope(ordinary_body, (10, 0)))

    def test_dry_pass_masks_memory_and_guards_waits(self) -> None:
        acquire = (
            "tl.inline_asm_elementwise(asm='ld.acquire.gpu.global.u32 $0, [$1];', "
            "constraints='=r,l', args=[state], dtype=tl.uint32, is_pure=False, pack=1)"
        )
        reciprocal = (
            "tl.inline_asm_elementwise(asm='rcp.approx.ftz.f32 $0, $1;', "
            "constraints='=f,f', args=[value], dtype=tl.float32, is_pure=True, pack=1)"
        )
        body = ast.parse(
            f"flag = {acquire}\n"
            "while flag != 1:\n"
            f"    flag = {acquire}\n"
            "value = tl.load(x + offsets, None).to(tl.float32)\n"
            f"scale = {reciprocal}\n"
            "tl.store(out + offsets, value * scale, mask)\n"
            "tl.atomic_add(state, 1, sem='release')\n"
        ).body
        guards: list[ast.If] = []
        rewritten = _dry_pass_rewrite(body, "live", frozenset(), guards)

        self.assertTrue(_dry_pass_binds(rewritten, guards, set()))
        self.assertEqual(len(guards), 1)
        self.assertEqual(
            ast.unparse(rewritten[1:]),
            "value = tl.load(x + offsets, live, other=0.0).to(tl.float32)\n"
            f"scale = {reciprocal}\n"
            "tl.store(out + offsets, value * scale, mask & live)\n"
            "tl.atomic_add(state, 1, sem='release', mask=live)",
        )
        # The dry pass takes one trip through each loop.
        body = ast.parse(
            "for k in tl.range(0, 2048, BLOCK, num_stages=4):\n"
            "    acc += tl.load(x + k, None)\n"
        ).body
        self.assertEqual(
            ast.unparse(_dry_pass_rewrite(body, "live", frozenset(), [])),
            "for k in tl.range(0, tl.where(live, 2048, tl.minimum(2048, 0 + BLOCK)), "
            "BLOCK, num_stages=4):\n    acc += tl.load(x + k, live, other=0.0)",
        )
        # A value only a skipped descriptor load produces disables the pass.
        body = ast.parse("word = desc.load([0])\ntl.store(out, word)\n").body
        guards = []
        rewritten = _dry_pass_rewrite(body, "live", frozenset(), guards)
        self.assertFalse(_dry_pass_binds(rewritten, guards, set()))

    def test_dry_pass_runs_helion_reduction_helpers(self) -> None:
        # max/min reductions lower to Helion's NaN-propagating helpers; they only
        # compute, so a root that reduces still gets its warm-up pass.
        body = ast.parse(
            "amax = helion_triton_helpers.max_propagate_nan(value, 0)\n"
            "amin = helion_triton_helpers.min_propagate_nan(value, 0)\n"
            "tl.store(out, amax - amin)\n"
        ).body
        guards: list[ast.If] = []
        rewritten = _dry_pass_rewrite(body, "live", frozenset(), guards)
        self.assertEqual(guards, [])
        self.assertTrue(_dry_pass_binds(rewritten, guards, set()))
        # The signal wait still spins, so it stays skipped.
        body = ast.parse(
            "flag = helion_triton_helpers.triton_wait_signal(addr, 1)\n"
            "tl.store(out, flag)\n"
        ).body
        guards = []
        rewritten = _dry_pass_rewrite(body, "live", frozenset(), guards)
        self.assertFalse(_dry_pass_binds(rewritten, guards, set()))

    def test_opaque_tile_body_clone_is_structurally_identical(self) -> None:
        body = ast.parse("value = value * 2\nout[index] = value\n").body
        cloned = _clone_opaque_statements(body)
        self.assertEqual(_ast_fingerprint(cloned), _ast_fingerprint(body))
        self.assertIsNot(cloned[0], body[0])

    def test_tile_dependency_loop_staging_preserves_computation(self) -> None:
        loop = ast.parse(
            "for k in tl.range(0, 128, 16):\n"
            "    partial = tl.load(pointer + k)\n"
            "    accumulator = accumulator + partial\n"
        ).body[0]
        assert isinstance(loop, ast.For)
        computation = _ast_fingerprint(loop.body)

        first = _clone_opaque_loop_segment(loop, end=ast.parse("64", mode="eval").body)
        second = _clone_opaque_loop_segment(
            loop, begin=ast.parse("64", mode="eval").body
        )
        setattr(loop, TILE_DEPENDENCY_SITE_ID_ATTR, 7)
        staged = _clone_opaque_statements_with_loop_segments(
            [loop],
            site_id=7,
            split_iteration_offsets=(4,),
            segment_waits=(
                tuple(ast.parse("first_ready = tl.load(counter)\n").body),
                tuple(ast.parse("second_ready = tl.load(counter + 1)\n").body),
            ),
        )

        self.assertEqual(_ast_fingerprint(first.body), computation)
        self.assertEqual(_ast_fingerprint(second.body), computation)
        self.assertIsInstance(staged[1], ast.For)
        self.assertIsInstance(staged[3], ast.For)
        self.assertEqual(_ast_fingerprint(staged[1].body), computation)
        self.assertEqual(_ast_fingerprint(staged[3].body), computation)
        self.assertEqual(ast.unparse(staged[0]), "first_ready = tl.load(counter)")
        self.assertEqual(ast.unparse(staged[2]), "second_ready = tl.load(counter + 1)")

    def test_opaque_tile_body_can_be_outlined_without_rewriting(self) -> None:
        device_function = object.__new__(DeviceFunction)
        device_function.ring = None
        device_function.arguments = []
        device_function.wrapper_only_params = []
        device_function.preamble = []
        cast("Any", device_function).namespace = SimpleNamespace(
            create_name=lambda name, _value: name
        )
        device_function.triton_outlined_helpers = []
        device_function.triton_outlined_helper_constexprs = {}
        device_function._variable_renames = {}
        device_function.dce_vars = []
        cast("Any", device_function).codegen = SimpleNamespace(module_statements=[])
        cast("Any", device_function).helper_manager = SimpleNamespace(
            codegen_helper_functions=list
        )
        body = ast.parse("value = tl.load(pointer)\ntl.store(output, value)\n").body
        computation = _ast_fingerprint(body)
        environment = SimpleNamespace(backend_name="triton", backend=TritonBackend())
        with mock.patch.object(CompileEnvironment, "current", return_value=environment):
            for scheduler in (False, True):
                for warp_specialize in (False, True):
                    with self.subTest(
                        scheduler=scheduler, warp_specialize=warp_specialize
                    ):
                        device_function.triton_outlined_helpers = []
                        device_function.config = helion.Config(
                            range_warp_specializes=[None, warp_specialize]
                        )
                        if scheduler:
                            call = _outline_cross_loop_region(
                                device_function,
                                name_hint="opaque_tile",
                                body=body,
                                noinline=True,
                            )
                            self.assertEqual(ast.unparse(call), "opaque_tile()")
                        else:
                            helper_name, arguments = (
                                device_function.register_triton_outlined_helper(
                                    "opaque_tile", body, noinline=True
                                )
                            )
                            self.assertEqual(helper_name, "opaque_tile")
                            self.assertEqual(arguments, ())
                        helper = device_function.codegen_helper_functions()[0]
                        self.assertIsInstance(helper, ast.FunctionDef)
                        assert isinstance(helper, ast.FunctionDef)
                        self.assertEqual(_ast_fingerprint(helper.body), computation)
                        self.assertEqual(helper.args.args, [])
                        self.assertEqual(
                            ast.unparse(helper.decorator_list[0]),
                            "triton.jit"
                            if scheduler and warp_specialize
                            else "triton.jit(noinline=True)",
                        )

    def test_outlined_tile_body_captures_compiler_preamble_values(self) -> None:
        device_function = object.__new__(DeviceFunction)
        device_function.ring = None
        device_function.arguments = []
        device_function.wrapper_only_params = []
        device_function.preamble = cast(
            "list[ast.AST]",
            ast.parse(
                "weight_desc = tl.make_tensor_descriptor(weight, [size], [1], [16])\n"
            ).body,
        )
        cast("Any", device_function).namespace = SimpleNamespace(
            create_name=lambda name, _value: name
        )
        device_function.triton_outlined_helpers = []
        device_function.triton_outlined_helper_constexprs = {}
        device_function._variable_renames = {}
        device_function.dce_vars = []
        cast("Any", device_function).codegen = SimpleNamespace(module_statements=[])
        cast("Any", device_function).helper_manager = SimpleNamespace(
            codegen_helper_functions=list
        )
        body = ast.parse("value = weight_desc.load([offset])\n").body
        environment = SimpleNamespace(backend_name="triton", backend=TritonBackend())

        with mock.patch.object(CompileEnvironment, "current", return_value=environment):
            helper_name, arguments = device_function.register_triton_outlined_helper(
                "descriptor_tile", body
            )
            helper = device_function.codegen_helper_functions()[0]

        self.assertEqual(helper_name, "descriptor_tile")
        self.assertEqual(arguments, ("weight_desc",))
        self.assertIsInstance(helper, ast.FunctionDef)
        assert isinstance(helper, ast.FunctionDef)
        self.assertEqual(
            [argument.arg for argument in helper.args.args], ["weight_desc"]
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def guarded_row_pair(
    x: torch.Tensor, flags1: torch.Tensor, flags2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = x.size()
    out1 = torch.zeros_like(x)
    out2 = torch.zeros_like(x)
    total = torch.zeros([cols], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows, block_size=1):
        live = torch.sum((flags1[:] > 0).to(torch.int32))
        if tile_r.begin < live:
            out1[tile_r, :] = x[tile_r, :] + 1
    for tile_r in hl.tile(rows, block_size=1):
        live2 = torch.sum((flags2[:] > 0).to(torch.int32))
        if tile_r.begin < live2:
            out2[tile_r, :] = x[tile_r, :] * 2
    for tile_c in hl.tile(cols, block_size=1):
        total[tile_c] = torch.sum(out2[:, tile_c], dim=0)
    return out1, out2, total


@onlyBackends(["triton", "cute"])
@skipIfTileIR("implicit tile-dependency lowering is unavailable on TileIR")
class TestCrossLoopBarrierPipeline(RefEagerTestBase, TestCase):
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_default_pipeline_orders_roots_with_a_grid_barrier(self) -> None:
        x = torch.arange(2 * 96, device=DEVICE, dtype=torch.float32).reshape(2, 96)
        row = torch.arange(64, device=DEVICE, dtype=torch.float32).reshape(1, 64)
        wide = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        flat = torch.arange(128, device=DEVICE, dtype=torch.float32)
        views = torch.arange(32 * 128, device=DEVICE, dtype=torch.float32)
        views = views.reshape(32, 128)
        cases = (
            (nested_store_chain, (x[:, :64],), [], (x[:, :64] + 1) * 2),
            (nested_load_store_chain, (wide,), [1, 16], (wide + 1) * 2 + 3),
            (cartesian_affine_chain, (x[:, :64],), [1, 16, 1, 32], (x[:, :64] + 1) * 2),
            (
                three_way_affine_chain,
                (x,),
                [1, 16, 1, 16],
                (x[:, :32] + x[:, 32:64] + x[:, 64:]) + 3,
            ),
            (
                multi_producer_join,
                (flat, flat + 3),
                [16, 16, 16],
                flat + 1 + (flat + 3) * 2,
            ),
            (singleton_root_join, (row,), [1, 16, 1, 16], torch.sum(row * 2, dim=-1)),
            (
                streamed_singleton_reduction,
                (wide,),
                [1, 16],
                torch.sum(wide + 1, dim=-1) + wide[:, 0] + 1,
            ),
            (size_one_view_chain, (views,), [4, 1, 4, 32], ((views + 1) * 2)[None]),
        )
        for kernel, args, block_sizes, expected in cases:
            with self.subTest(kernel=kernel.name):
                code, out = code_and_output(
                    kernel,
                    args,
                    block_sizes=block_sizes,
                    pid_type="persistent_blocked",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
                torch.testing.assert_close(out, expected)
                self.assertIn("grid_barrier(", code)
                self.assertNotIn("tile_dependency_", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_zero_task_roots_do_not_allocate_task_events(self) -> None:
        x = torch.empty((0, 64), device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
            num_warps=1,
        )

        self.assertEqual(out.shape, x.shape)
        self.assertNotIn("tile_dependency_", code)


@skipIfNotTriton("static/dynamic cross-loop pipelines are Triton-only codegen")
class TestCrossLoopCodegen(RefEagerTestBase, TestCase):
    def assertUsesExactReadiness(self, code: str) -> None:
        self.assertTrue(
            "tile_dependency_continuation_previous" in code
            or "tile_dependency_readiness_wait" in code,
            "expected a final-arrival continuation or an exact readiness wait",
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nested_producer_iterations_publish_readiness(self) -> None:
        x = torch.arange(2 * 64, device=DEVICE, dtype=torch.float32).reshape(2, 64)
        for name, extra_config, expected_range_option in (
            ("default", {"num_warps": 1}, None),
            (
                "pipelined",
                {"num_warps": 4, "range_num_stages": [0, 4, 0]},
                "num_stages=4",
            ),
            (
                "unrolled",
                {"num_warps": 4, "range_unroll_factors": [0, 2, 0]},
                "loop_unroll_factor=2",
            ),
        ):
            with self.subTest(name=name):
                code, out = code_and_output(
                    nested_store_chain,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    **extra_config,
                )

                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertIn("tile_dependency_readiness_wait", code)
                self.assertNotIn("tile_dependency_root_barrier", code)
                if expected_range_option is not None:
                    self.assertIn(expected_range_option, code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dynamic_pipeline_preserves_nested_readiness(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        code, out = code_and_output(
            nested_load_store_chain,
            (x,),
            block_sizes=[1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2 + 3)
        self.assertIn("tile_dependency_raw_dispatch_ticket", code)
        self.assertIn("tile_dependency_nested_loop_wait", code)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("_minimum_resident_programs=", code)
        self.assertNotIn("launch_pdl", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_pdl_launch_waits_before_state_reads(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        config = {
            "block_sizes": [1, 16],
            "pid_type": "persistent_blocked",
            "cross_loop_pipeline": "dynamic",
            "num_sm_multiplier": 1,
            "num_warps": 1,
        }
        code, out = code_and_output(pdl_store_chain, (x,), **config)

        torch.testing.assert_close(out, (x + 1) * 2)
        torch.testing.assert_close(pdl_store_chain(x), (x + 1) * 2)
        self.assertIn("launch_pdl=True", code)
        kernel = code[code.index("def _helion_") : code.index("\ndef pdl_store")]
        statements = [
            line.strip()
            for line in kernel.splitlines()[1:]
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertEqual(statements[0], "tl.extra.cuda.gdc_wait()")
        self.assertEqual(statements[-1], "tl.extra.cuda.gdc_launch_dependents()")
        with self.assertRaises(helion.exc.PdlStateWithoutWait):
            code_and_output(pdl_unwaited_chain, (x,), **config)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dynamic_pipeline_precompiles_without_launcher_state(self) -> None:
        x = torch.zeros(1, 4096, device=DEVICE)
        bound = nested_load_store_chain.bind((x,))
        config = helion.Config(
            block_sizes=[1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=1,
        )
        compiled = bound.compile_config(config)
        # The precompiler gets placeholders for the launcher-owned state.
        self.assertTrue(_triton_compile(compiled, (x,), config, bound))

    @skipIfRefEager("compile-only launcher arguments have no eager reference")
    def test_compile_only_launch_args_mirror_launcher_state(self) -> None:
        x = torch.zeros(4)
        self.assertEqual(
            compile_only_launch_args(3, num_warps=4, _minimum_resident_programs=2),
            ((3,), {"num_warps": 4}),
        )
        args, kwargs = compile_only_launch_args(
            x,
            num_warps=4,
            _remote_barrier_signal_slots_per_program=1,
            _remote_copy_scratch_specs=((x.half(), 8),),
            _persistent_state_specs=(
                (x, 2, torch.uint32, False),
                (x, 3, torch.uint64, True),
            ),
            _persistent_state_process_group_name="group",
        )
        self.assertEqual(
            [arg.dtype for arg in args[1:]],
            [torch.int64, torch.float16, torch.uint32, torch.uint64, torch.int64],
        )
        self.assertEqual(kwargs, {"num_warps": 4})
        # Every tensor passed as a descriptor: the state specs name the device.
        args, _ = compile_only_launch_args(
            "descriptor", _persistent_state_specs=((x, 2, torch.uint32, False),)
        )
        self.assertEqual(args[1].device, x.device)

    @skipIfNotCUDA()
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dynamic_pipeline_with_host_tensor_descriptors(self) -> None:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def two_stage(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.empty_like(x)
            for producer_m in hl.tile(x.size(0), block_size=32):
                for producer_n in hl.tile(x.size(1), block_size=32):
                    tmp[producer_m, producer_n] = x[producer_m, producer_n] + 1
            for consumer_m, consumer_n in hl.tile(x.size(), block_size=[32, 32]):
                out[consumer_m, consumer_n] = tmp[consumer_m, consumer_n] * 2
            return out

        x = torch.arange(64 * 128, device=DEVICE, dtype=torch.float32).reshape(64, 128)
        for host_descriptors in (False, True):
            if host_descriptors and not supports_host_tensor_descriptor():
                continue
            with self.subTest(host_descriptors=host_descriptors):
                code, out = code_and_output(
                    two_stage,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="dynamic",
                    num_sm_multiplier=1,
                    num_warps=1,
                    range_num_stages=[0, 4, 0],
                    indexing="tensor_descriptor",
                    host_tensor_descriptors=host_descriptors,
                )

                torch.testing.assert_close(out, (x + 1) * 2)
                if host_descriptors:
                    self.assertIn("_helion_tensor_descriptor(", code)
                    self.assertNotIn("tl.make_tensor_descriptor", code)
                    self.assertIn("num_stages=4", code)
                else:
                    self.assertIn("tl.make_tensor_descriptor", code)
                    self.assertNotIn("num_stages=4", code)
                self.assertIn("tile_dependency_raw_dispatch_ticket", code)
                self.assertIn("tile_dependency_root_0_scheduled_task", code)
                self.assertNotIn("tile_dependency_root_barrier", code)
                # The TMA store of tmp must complete before its release, and the
                # consumer's TMA load of tmp must follow a proxy fence.
                self.assertIn(
                    "cp.async.bulk.wait_group 0; fence.proxy.async.global", code
                )
                self.assertIn("async_load_fence", code)

    @skipIfNotCUDA()
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_middle_root_tma_store_drains_every_release(self) -> None:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def three_stage(x: torch.Tensor) -> torch.Tensor:
            a = torch.empty_like(x)
            b = torch.empty_like(x)
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size(), block_size=[32, 32]):
                a[tile_m, tile_n] = x[tile_m, tile_n] + 1
            for tile_m, tile_n in hl.tile(x.size(), block_size=[32, 32]):
                b[tile_m, tile_n] = a[tile_m, tile_n] * 2
            for tile_m, tile_n in hl.tile(x.size(), block_size=[32, 32]):
                out[tile_m, tile_n] = b[tile_m, tile_n] + 3
            return out

        x = torch.arange(64 * 128, device=DEVICE, dtype=torch.float32).reshape(64, 128)
        # Continuations with fan-in 1 publish without atomics; keep every edge
        # on the release path this test inspects.
        with mock.patch.object(
            cross_loop_scheduler,
            "choose_final_arrival_continuations",
            return_value=(),
        ):
            code, out = code_and_output(
                three_stage,
                (x,),
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                num_sm_multiplier=1,
                num_warps=4,
                # Only the store of b is a TMA store; every load uses pointers.
                indexing=[
                    "pointer",
                    "pointer",
                    "pointer",
                    "tensor_descriptor",
                    "pointer",
                    "pointer",
                ],
            )

        torch.testing.assert_close(out, (x + 1) * 2 + 3)
        self.assertIn(".store(", code)
        lines = code.splitlines()
        releases = [
            i
            for i, line in enumerate(lines)
            if "sem='release'" in line or "sem='acq_rel'" in line
        ]
        self.assertTrue(releases)
        # Walking back from each release must reach a drain before any store.
        for i in releases:
            for line in reversed(lines[:i]):
                if "cp.async.bulk.wait_group 0;" in line:
                    break
                self.assertFalse(
                    ".store(" in line or line.lstrip().startswith("def "),
                    msg=f"release without a TMA drain: {lines[i].strip()}",
                )

        # Without a TMA access there is nothing to drain or fence.
        code, out = code_and_output(
            three_stage,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=4,
            indexing="pointer",
        )
        torch.testing.assert_close(out, (x + 1) * 2 + 3)
        self.assertNotIn("fence.proxy.async", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_completion_relation_covers_conditional_indirect_payload(self) -> None:
        x = torch.arange(
            2 * 4 * 32,
            device=DEVICE,
            dtype=torch.float32,
        ).reshape(2, 4, 32)
        active_children = torch.tensor([4, 2], device=DEVICE, dtype=torch.int32)
        code, out = code_and_output(
            conditional_payload_completion_chain,
            (x, active_children),
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=1,
        )

        expected = torch.stack(
            (torch.sum(x[0, :4], dim=0) + 4, torch.sum(x[1, :2], dim=0) + 2)
        )
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_nested_loop_wait", code)
        self.assertNotIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nested_completion_does_not_cover_later_parent_payload(self) -> None:
        x = torch.arange(2 * 32, device=DEVICE, dtype=torch.float32).reshape(2, 32)
        code, out = code_and_output(
            nested_early_completion_chain,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, x + 1)
        self.assertIn("tile_dependency_nested_loop_wait", code)
        # The later parent payload needs its own root-task readiness event; it
        # cannot be covered by the earlier nested completion publication.
        self.assertIn("tile_dependency_readiness_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dynamic_dispatch_uses_the_selected_local_order(self) -> None:
        x = torch.arange(4 * 64, device=DEVICE, dtype=torch.float32).reshape(4, 64)
        original_build = cross_loop_codegen.build_static_pipeline_plan

        def build_with_transposed_producer_order(**kwargs: Any):
            plan = original_build(**kwargs)
            orders = list(plan.execution_orders)
            domain = orders[0].tasks_by_ordinal.target_domain
            replacement = DenseTaskOrder.from_pid(
                domain, tuple(reversed(domain.axis_order))
            )
            assert replacement is not None
            orders[0] = replacement
            return dataclasses.replace(plan, execution_orders=tuple(orders))

        with mock.patch.object(
            cross_loop_codegen,
            "build_static_pipeline_plan",
            side_effect=build_with_transposed_producer_order,
        ):
            code, out = code_and_output(
                cartesian_affine_chain,
                (x,),
                block_sizes=[1, 16, 1, 32],
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                num_sm_multiplier=1,
                num_warps=1,
            )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_dispatch_ticket", code)
        self.assertIn("tile_dependency_scheduled_pid_task", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dynamic_kernel_scope_prefix_outlines_safe_suffix(self) -> None:
        x = torch.arange(4 * 64, device=DEVICE, dtype=torch.float32).reshape(4, 64)
        call_index = 0

        def first_root_requires_kernel_scope(*_args: object) -> bool:
            nonlocal call_index
            result = call_index % 2 == 0
            call_index += 1
            return result

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "choose_final_arrival_continuations",
                return_value=(),
            ),
            mock.patch.object(
                cross_loop_codegen,
                "_triton_root_requires_kernel_scope",
                side_effect=first_root_requires_kernel_scope,
            ),
        ):
            code, out = code_and_output(
                cartesian_affine_chain,
                (x,),
                block_sizes=[1, 16, 1, 32],
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                num_sm_multiplier=1,
                num_warps=1,
            )

        torch.testing.assert_close(out, (x + 1) * 2)
        marker = "@triton.jit(noinline=True)\ndef tile_dependency_packet_dispatch"
        self.assertIn(marker, code)
        helper = code[
            code.index(marker) : code.index(
                "\n@triton.jit", code.index(marker) + len(marker)
            )
        ]
        self.assertIn("tile_dependency_root_1_scheduled_task", helper)
        self.assertNotIn("tile_dependency_root_0_scheduled_task", helper)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dynamic_waiting_root_runs_a_dry_pass(self) -> None:
        width = 32 * torch.cuda.get_device_properties(DEVICE).multi_processor_count
        x = torch.arange(width, device=DEVICE, dtype=torch.float32).reshape(1, width)
        code, out = code_and_output(
            singleton_gate_chain,
            (x,),
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * (x + 1).sum(-1, keepdim=True))
        # The waiting gate first runs once with memory masked and zero-filled.
        self.assertIn(
            "for tile_dependency_dry_pass in "
            "tl.range(tile_dependency_dispatch_ticket_1 // 1073741824, 2, 1, "
            "num_stages=1):",
            code,
        )
        self.assertIn("other=0.0", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_exact_nested_keys_wait_inside_each_loop_iteration(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)

        with mock.patch.object(
            cross_loop_scheduler,
            "_compact_nested_loop_counters_for_schedule",
            side_effect=lambda _graph, _schedule, exact, *_args, **_kwargs: exact,
        ):
            code, out = code_and_output(
                nested_load_store_chain,
                (x,),
                block_sizes=[1, 16],
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )

        torch.testing.assert_close(out, (x + 1) * 2 + 3)
        tree = ast.parse(code)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        wait = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id.startswith("tile_dependency_nested_loop_wait")
                for target in node.targets
            )
        )
        parent = parents.get(wait)
        while parent is not None and not isinstance(parent, ast.For):
            parent = parents.get(parent)
        self.assertIsInstance(parent, ast.For)
        assert isinstance(parent, ast.For) and isinstance(parent.target, ast.Name)
        self.assertIn(
            parent.target.id,
            {node.id for node in ast.walk(wait.value) if isinstance(node, ast.Name)},
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_constant_nested_key_uses_membership_guard(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        original_build = cross_loop_codegen.build_static_pipeline_plan

        def build_with_partial_constant_nested_key(**kwargs: Any):
            plan = original_build(**kwargs)
            replacement = None
            replaced_counters = []
            for counter in plan.readiness_counters:
                consumers = []
                for consumer in counter.consumers:
                    relation = consumer.keys_by_consumer
                    nested_axes = cross_loop_scheduler.nested_logical_axes(
                        plan.root_domains[consumer.consumer_root],
                        relation.source_domain,
                    )
                    if replacement is not None or len(nested_axes) != 1:
                        consumers.append(consumer)
                        continue
                    (nested_axis,) = nested_axes
                    source_bounds = tuple(
                        (
                            axis,
                            1 if axis == nested_axis else 0,
                            relation.source_domain.axis_count_expressions[axis],
                            1,
                        )
                        for axis in relation.source_domain.axis_order
                    )
                    replacement = CoordinateRelation(
                        relation.source_domain,
                        relation.target_domain,
                        (
                            _CoordinateRelationPiece(
                                source_bounds,
                                tuple(
                                    (axis, 0, 1, 1)
                                    for axis in relation.target_domain.axis_order
                                ),
                            ),
                        ),
                    )
                    consumers.append(
                        dataclasses.replace(
                            consumer,
                            incidence=Incidence._from_constructed(
                                consumer.incidence.items_by_key,
                                keys_by_item=replacement,
                                count_by_key=consumer.incidence.count_by_key,
                                grouped_items=consumer.incidence.grouped_items,
                            ),
                        )
                    )
                replaced_counters.append(
                    dataclasses.replace(counter, consumers=tuple(consumers))
                )
            self.assertIsNotNone(replacement)
            return dataclasses.replace(
                plan,
                readiness_counters=tuple(replaced_counters),
            )

        with mock.patch.object(
            cross_loop_codegen,
            "build_static_pipeline_plan",
            side_effect=build_with_partial_constant_nested_key,
        ):
            bound = nested_load_store_chain.bind((x,))
            code = bound.to_triton_code(
                helion.Config(
                    block_sizes=[1, 16],
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
            )

        tree = ast.parse(code)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        waits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id.startswith("tile_dependency_nested_loop_wait")
                for target in node.targets
            )
        ]
        self.assertTrue(waits)

        def has_if_ancestor(node: ast.AST) -> bool:
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.If):
                    return True
            return False

        self.assertTrue(all(has_if_ancestor(wait) for wait in waits))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_consumers_reading_different_axes_whole_each_keep_a_counter(
        self,
    ) -> None:
        # Two later roots read root 0's tasks through one producer-keyed event:
        # one every token of a split, the other every split of a token.  No key
        # serves both, so each gets its own coarse counter (in either source
        # order) instead of both falling back to root barriers.
        x = torch.randn(8, 16, 512, device=DEVICE)
        v = x * 2.0
        act = torch.cat([v + 1.0, v - 1.0])
        for kernel in (
            split_keyed_and_fan_in_consumers,
            fan_in_and_split_keyed_consumers,
        ):
            with self.subTest(kernel=kernel.name):
                code, (rows, out) = code_and_output(
                    kernel,
                    (x, 1),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_warps=4,
                    num_stages=1,
                )
                torch.testing.assert_close(
                    rows, act[:, 8:].reshape(16, 8, 4, 128).sum(-1).permute(1, 2, 0)
                )
                torch.testing.assert_close(
                    out, v * v.sum((1, 2))[:, None, None], rtol=1e-4, atol=1e-3
                )
                self.assertIn("tile_dependency_readiness_wait_1", code)
                self.assertNotIn("tile_dependency_root_barrier_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_root_arrival_rides_last_task_arrivals(self) -> None:
        # Root 0 publishes per-task counters (to root 1) and, forced here, a
        # root barrier (to root 2): a worker's barrier arrival joins its last
        # task's shared atomic instead of a second release after the loop.
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        original_build = cross_loop_codegen.build_static_pipeline_plan

        def build_with_extra_root_barrier(**kwargs: Any):
            plan = original_build(**kwargs)
            return dataclasses.replace(
                plan,
                root_barrier_edges=plan.root_barrier_edges | frozenset(((0, 2),)),
            )

        with mock.patch.object(
            cross_loop_codegen,
            "build_static_pipeline_plan",
            side_effect=build_with_extra_root_barrier,
        ):
            config = {
                "block_sizes": [1, 16],
                "pid_type": "persistent_blocked",
                "cross_loop_pipeline": "static",
                "num_sm_multiplier": 1,
                "num_warps": 1,
            }
            code, out = code_and_output(nested_load_store_chain, (x,), **config)
            # An extra or missing arrival leaves the counter off for later launches.
            for _ in range(20):
                out = output_only(nested_load_store_chain, (x,), **config)
                torch.testing.assert_close(out, (x + 1) * 2 + 3)

        self.assertIn("tile_dependency_root_barrier_wait", code)

        def release_atomics(function: ast.FunctionDef) -> list[ast.Call]:
            return [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("atomic_")
            ]

        task = release_atomics(
            _generated_function(code, "tile_dependency_root_0_scheduled_task")
        )
        self.assertEqual(len(task), 1)
        self.assertIn("tl.where", ast.unparse(task[0].args[0]))
        # Only a worker's last task arrives (256 tasks, one per 16 columns).
        workers = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        self.assertIn(f"virtual_pid + {workers} >= 256", ast.unparse(task[0]))
        self.assertEqual(
            release_atomics(
                _generated_function(code, "_helion_nested_load_store_chain")
            ),
            [],
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_single_trip_root_barrier_publisher_can_inline(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        original_build = cross_loop_codegen.build_static_pipeline_plan

        def build_with_middle_root_barrier(**kwargs: Any):
            plan = original_build(**kwargs)
            return dataclasses.replace(
                plan,
                readiness_counters=tuple(
                    counter
                    for counter in plan.readiness_counters
                    if all(
                        producer.producer_root != 1 for producer in counter.producers
                    )
                ),
                root_barrier_edges=plan.root_barrier_edges | frozenset(((1, 2),)),
            )

        with mock.patch.object(
            cross_loop_codegen,
            "build_static_pipeline_plan",
            side_effect=build_with_middle_root_barrier,
        ):
            code, out = code_and_output(
                nested_load_store_chain,
                (x,),
                block_sizes=[1, 16],
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )

        torch.testing.assert_close(out, (x + 1) * 2 + 3)
        self.assertIn("tile_dependency_root_barrier_wait", code)
        wrapper = _generated_function(code, "tile_dependency_root_1_scheduled_task")
        self.assertTrue(
            all(
                not isinstance(decorator, ast.Call)
                for decorator in wrapper.decorator_list
            )
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Name)
                and node.id.startswith("tile_dependency_root_barrier")
                for node in ast.walk(wrapper)
            )
        )
        kernel = _generated_function(code, "_helion_nested_load_store_chain")
        dispatch = next(
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and _call_name(node) == "tile_dependency_root_1_scheduled_task"
        )
        publication = next(
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("atomic_")
            and any(
                keyword.arg == "sem"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "release"
                for keyword in node.keywords
            )
        )
        # Root 2 waits on root 1's barrier inside each of its tasks (its waits
        # are scoped to the first access of what they order).
        consumer = _generated_function(code, "tile_dependency_root_2")
        self.assertTrue(
            any(
                isinstance(node, ast.Name)
                and node.id.startswith("tile_dependency_root_barrier_wait")
                for node in ast.walk(consumer)
            )
        )
        consumer_dispatch = next(
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and _call_name(node) == "tile_dependency_root_2"
        )
        self.assertLess(dispatch.lineno, publication.lineno)
        self.assertLess(publication.lineno, consumer_dispatch.lineno)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_two_axis_nested_loop_falls_back_to_root_barrier(self) -> None:
        x = torch.arange(32 * 32, device=DEVICE, dtype=torch.float32).reshape(32, 32)
        for pipeline in ("static", "dynamic"):
            with self.subTest(pipeline=pipeline):
                for launch in range(2):
                    code, out = code_and_output(
                        nested_two_axis_consumer,
                        (x + launch,),
                        block_sizes=[8, 8],
                        pid_type="persistent_blocked",
                        cross_loop_pipeline=pipeline,
                        num_sm_multiplier=1,
                        num_warps=1,
                    )
                    torch.testing.assert_close(out, ((x + launch) + 1) * 2)
                self.assertNotIn("tile_dependency_nested_loop_wait", code)
                self.assertIn("tile_dependency_root_barrier_wait", code)
                self.assertEqual(
                    "tile_dependency_raw_dispatch_ticket" in code,
                    pipeline == "dynamic",
                )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_cartesian_unequal_tiles_choose_proven_synchronization(self) -> None:
        for batch, width, producer_width, consumer_width in (
            (2, 64, 16, 32),
            (4, 64, 16, 32),
            (2, 64, 32, 16),
        ):
            with self.subTest(
                batch=batch,
                width=width,
                producer_width=producer_width,
                consumer_width=consumer_width,
            ):
                x = torch.arange(
                    batch * width,
                    device=DEVICE,
                    dtype=torch.float32,
                ).reshape(batch, width)
                for launch in range(2):
                    code, out = code_and_output(
                        cartesian_affine_chain,
                        (x + launch,),
                        block_sizes=[1, producer_width, 1, consumer_width],
                        pid_type="persistent_blocked",
                        cross_loop_pipeline="static",
                        num_sm_multiplier=1,
                        num_warps=1,
                    )
                    torch.testing.assert_close(out, ((x + launch) + 1) * 2)
                self.assertNotIn("tile_dependency_task_wait", code)
                if producer_width < consumer_width:
                    self.assertNotIn("tile_dependency_root_barrier", code)
                    self.assertUsesExactReadiness(code)
                else:
                    self.assertIn("tile_dependency_root_barrier_wait", code)
                    self.assertNotIn("tile_dependency_readiness_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_continuation_accepts_non_power_of_two_fanin(self) -> None:
        x = torch.arange(2 * 96, device=DEVICE, dtype=torch.float32).reshape(2, 96)
        for launch in range(2):
            code, out = code_and_output(
                three_way_affine_chain,
                (x + launch,),
                block_sizes=[1, 16, 1, 16],
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )
            expected_input = x + launch + 1
            expected = (
                expected_input[:, :32]
                + expected_input[:, 32:64]
                + expected_input[:, 64:]
            )
            torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("* tl.cast(3, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_readiness_counter_supports_chaining(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        for pipeline in ("static", "dynamic"):
            with self.subTest(pipeline=pipeline):
                for launch in range(2):
                    code, out = code_and_output(
                        readiness_counter_chain,
                        (x + launch,),
                        pid_type="persistent_blocked",
                        cross_loop_pipeline=pipeline,
                        num_sm_multiplier=1,
                        num_warps=1,
                    )
                    torch.testing.assert_close(
                        out, torch.sum(x + launch + 1).reshape(1)
                    )
                continuation_lines = [
                    line
                    for line in code.splitlines()
                    if "tile_dependency_continuation_previous" in line
                    and "tl.atomic_add" in line
                ]
                self.assertGreaterEqual(len(continuation_lines), 1)
                self.assertLessEqual(len(continuation_lines), 2)
                for line in continuation_lines:
                    self.assertIn(f"* {_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}", line)
                if len(continuation_lines) == 2:
                    self.assertIn(
                        f"+ {16 * _CROSS_LOOP_COUNTER_ALIGNMENT_WORDS} +",
                        continuation_lines[1],
                    )
                else:
                    self.assertIn("tile_dependency_readiness_wait", code)
                self.assertNotIn("tile_dependency_task_wait", code)
                self.assertIn("tile_dependency_root_barrier_wait", code)
                self.assertIn("ld.acquire.gpu.global.u32", code)
                self.assertNotIn("ld.acquire.gpu.global.u64", code)
                self.assertNotIn("tl.atomic_max", code)
                self.assertEqual(
                    "tile_dependency_raw_dispatch_ticket" in code,
                    pipeline == "dynamic",
                )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_tiles_keep_exact_task_readiness(self) -> None:
        x = torch.arange(140, device=DEVICE, dtype=torch.float32).reshape(2, 70)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_root_barrier_wait", code)
        self.assertNotIn("tile_dependency_readiness_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_root_loops_use_range_config(self) -> None:
        x = torch.arange(140, device=DEVICE, dtype=torch.float32).reshape(2, 70)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
            range_unroll_factors=[2, 3],
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        roots = [line for line in code.splitlines() if "for virtual_pid in" in line]
        self.assertEqual(len(roots), 2)
        self.assertIn("loop_unroll_factor=2", roots[0])
        self.assertIn("loop_unroll_factor=3", roots[1])

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    @skipIfCudaCapabilityLessThan((10, 0), reason="warp specialization needs sm100")
    def test_static_root_loops_warp_specialize_each_root(self) -> None:
        x = torch.arange(140, device=DEVICE, dtype=torch.float32).reshape(2, 70)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=4,
            range_warp_specializes=[True, True],
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        roots = [line for line in code.splitlines() if "for virtual_pid in" in line]
        self.assertEqual(len(roots), 2)
        self.assertTrue(all("warp_specialize=True" in root for root in roots))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_roots_start_after_earlier_roots(self) -> None:
        static = {
            "pid_type": "persistent_blocked",
            "cross_loop_pipeline": "static",
            "num_sm_multiplier": 1,
        }
        x = torch.arange(140, device=DEVICE, dtype=torch.float32).reshape(2, 70)
        for launch in range(2):
            code, out = code_and_output(
                cartesian_affine_chain,
                (x + launch,),
                block_sizes=[1, 16, 1, 32],
                num_warps=1,
                **static,
            )
            torch.testing.assert_close(out, (x + launch + 1) * 2)
        # The second root's lanes start after the first root's 2 x 5 tasks.
        roots = [line for line in code.splitlines() if "for virtual_pid in" in line]
        self.assertEqual(len(roots), 2)
        self.assertNotIn("%", roots[0])
        workers = helion.runtime.get_num_sm(DEVICE)
        self.assertIn(f"(tl.program_id(0) + {workers - 10}) % {workers}", roots[1])
        # Guarded roots move later roots by their runtime live counts.
        x = torch.randn((32, 256), device=DEVICE, dtype=torch.float32)
        for live in (0, 1, 5, 16):
            flags = (torch.arange(16, device=DEVICE) < live).to(torch.int32)
            _, (tmp, out, side) = code_and_output(
                guarded_prefix_chain, (x, flags), num_warps=4, **static
            )
            expected = torch.zeros_like(x)
            expected[:live] = x[:live] + 1
            torch.testing.assert_close(tmp, expected)
            torch.testing.assert_close(out, expected * 2 + expected[0][None, :])
            side_rows = (live + 1) // 2 * 2
            torch.testing.assert_close(side[:side_rows], x[:side_rows] * 3)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_prefix_uses_exact_readiness(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, (tmp, out) = code_and_output(
                partial_prefix_continuation,
                (x + launch,),
                block_sizes=[16, 32],
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(tmp, x + launch + 1)
            torch.testing.assert_close(out, (x[:64] + launch + 1) * 2)
        self.assertUsesExactReadiness(code)
        self.assertIn("tl.cast(2, tl.uint32)", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_mixed_radix_dependency_uses_counter_continuation(self) -> None:
        # Keep the producer within one resident wave so final-arrival
        # continuation dominance is exact.
        x = torch.randn((2, 512), device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, out = code_and_output(
                mixed_radix_continuation,
                (x + launch,),
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )
            gate_up = x + launch + 1
            gate, up = gate_up.chunk(2, dim=1)
            torch.testing.assert_close(out, gate * torch.sigmoid(gate) * up)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tl.cast(32, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertLessEqual(code.count("tl.where"), 2)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_runtime_bound_inner_loop_store_is_ordered(self) -> None:
        x = torch.randn((8, 16), device=DEVICE, dtype=torch.float32)
        starts = torch.tensor([0, 1, 5, 0, 3, 8, 2, 4], device=DEVICE)
        ends = torch.tensor([0, 2, 9, 16, 3, 15, 16, 11], device=DEVICE)
        code, out = code_and_output(
            runtime_bound_store_chain,
            (x, starts, ends),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )
        columns = torch.arange(16, device=DEVICE)
        mask = (columns[None, :] >= starts[:, None]) & (
            columns[None, :] < ends[:, None]
        )
        torch.testing.assert_close(out, torch.sum(x * mask, dim=1) * 2)
        # The store's site has no geometry, so the edge falls back to a barrier.
        self.assertIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_runtime_bound_inner_loop_keeps_exact_readiness(self) -> None:
        x = torch.randn((8, 16), device=DEVICE, dtype=torch.float32)
        starts = torch.tensor([0, 1, 5, 0, 3, 8, 2, 4], device=DEVICE)
        ends = torch.tensor([0, 2, 9, 16, 3, 15, 16, 11], device=DEVICE)
        code, out = code_and_output(
            runtime_bound_loop_chain,
            (x, starts, ends),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )
        columns = torch.arange(16, device=DEVICE)
        mask = (columns[None, :] >= starts[:, None]) & (
            columns[None, :] < ends[:, None]
        )
        torch.testing.assert_close(out, torch.sum(x * mask, dim=1) * 2)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_invariant_guard_bounds_the_static_task_loop(self) -> None:
        x = torch.randn((32, 256), device=DEVICE, dtype=torch.float32)
        for live in (0, 1, 5, 16):
            flags = (torch.arange(16, device=DEVICE) < live).to(torch.int32)
            code, (tmp, out, side) = code_and_output(
                guarded_prefix_chain,
                (x, flags),
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=4,
            )
            expected = torch.zeros_like(x)
            expected[:live] = x[:live] + 1
            torch.testing.assert_close(tmp, expected)
            torch.testing.assert_close(out, expected * 2 + expected[0][None, :])
            side_rows = (live + 1) // 2 * 2
            expected_side = torch.zeros_like(x)
            expected_side[:side_rows] = x[:side_rows] * 3
            torch.testing.assert_close(side, expected_side)
        kernel = _generated_function(code, "_helion_guarded_prefix_chain")
        # The invariant bound is hoisted and caps the loop; tasks keep no guard.
        # The last root rebases its PID first and tiles the bound by 2.  (With
        # fewer SMs than the first root's tasks, a later root's start rotates by
        # the live count too: only the loops' stops count.)
        prologue = ast.unparse(kernel).split("for virtual_pid")[0]
        bounded = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.For)
            and ast.unparse(node.target) == "virtual_pid"
            and isinstance(node.iter, ast.Call)
            and "tile_dependency_live_tasks" in ast.unparse(node.iter.args[1])
        ]
        self.assertEqual(len(bounded), 2)
        self.assertIn("tl.cdiv(side_live, 2)", ast.unparse(kernel))
        self.assertIn("redux.sync.min.s32", prologue)
        self.assertIn("tl.sum(", prologue)
        root = _generated_function(code, "tile_dependency_root_0")
        self.assertFalse(any(isinstance(node, ast.If) for node in ast.walk(root)))
        self.assertNotIn("flags", ast.unparse(root))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_computed_bound_waits_before_its_hoisted_load(self) -> None:
        x = torch.randn((16, 256), device=DEVICE, dtype=torch.float32)
        for kernel in (computed_bound_chain, overwritten_bound_chain):
            for live in (0, 5, 16):
                flags = (torch.arange(16, device=DEVICE) < live).to(torch.int32)
                code, (marks, out) = code_and_output(
                    kernel,
                    (x, flags),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=4,
                )
                expected = torch.zeros_like(x)
                expected[:live] = x[:live] + 1
                torch.testing.assert_close(out, expected)
            launch = ast.unparse(_generated_function(code, f"_helion_{kernel.name}"))
            bounded = "tile_dependency_live_tasks" in launch
            if kernel is overwritten_bound_chain:
                # A later root writes the bound: each task reads it.
                self.assertFalse(bounded)
                continue
            # The bound caps the loop, read after the wait on its writers.
            self.assertTrue(bounded)
            wait = min(
                launch.find(name)
                for name in (
                    "tile_dependency_root_barrier_wait",
                    "tile_dependency_readiness_wait",
                )
                if name in launch
            )
            self.assertLess(wait, launch.index("redux.sync.min.s32"))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_guard_skipped_tasks_publish_local_readiness(self) -> None:
        x = torch.randn((16, 256), device=DEVICE, dtype=torch.float32)
        for live in (0, 1, 5, 16):
            flags = (torch.arange(16, device=DEVICE) < live).to(torch.int32)
            code, (tmp, out) = code_and_output(
                guarded_readiness_fanout,
                (x, flags),
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=4,
            )
            expected = torch.zeros_like(x)
            expected[:live] = x[:live] + 1
            splits = torch.arange(4, device=DEVICE, dtype=x.dtype)[:, None, None]
            torch.testing.assert_close(tmp, expected)
            torch.testing.assert_close(out, expected[None] * 2 + splits)
        # Each row's four split tasks wait on its four column tiles, live or not.
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertIn("sem='relaxed'", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    @skipIfCudaCapabilityLessThan((10, 0), reason="warp specialization needs sm100")
    def test_warp_specialized_consumer_waits_on_a_readiness_token(self) -> None:
        x = torch.randn((16, 256), device=DEVICE, dtype=torch.float32)
        for live in (0, 5, 16):
            flags = (torch.arange(16, device=DEVICE) < live).to(torch.int32)
            code, (tmp, out) = code_and_output(
                guarded_readiness_fanout,
                (x, flags),
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=4,
                range_warp_specializes=[None, True],
            )
            expected = torch.zeros_like(x)
            expected[:live] = x[:live] + 1
            splits = torch.arange(4, device=DEVICE, dtype=x.dtype)[:, None, None]
            torch.testing.assert_close(out, expected[None] * 2 + splits)
        # The wait is one asm spin, not a while loop, and its zero token feeds
        # the task PID so the producer partition's loads wait too.
        task = ast.unparse(
            _generated_function(code, "tile_dependency_root_1_scheduled_task")
        )
        root = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
        self.assertIn("READY_WAIT", task)
        self.assertNotIn("while ", task)
        self.assertIn("+ tile_dependency_readiness_token", root)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_guarded_consumer_scopes_root_barrier(self) -> None:
        # Fewer guarded tasks than workers: members run the barrier wait in
        # each task, a member with no live task waits before its skipped-task
        # arrivals, and non-members never wait.
        x = torch.randn((4096,), device=DEVICE, dtype=torch.float32)
        w = torch.randn((32, 4096), device=DEVICE, dtype=torch.float32)
        for live in (0, 3, 32):
            with self.subTest(live=live):
                flags = (torch.arange(32, device=DEVICE) < live).to(torch.int32)
                code, (act, out, res) = code_and_output(
                    guarded_barrier_consumer,
                    (x, w, flags),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=4,
                )
                expected = (w * (x + 1)[None, :]).sum(1)
                expected[live:] = 0
                torch.testing.assert_close(act, x + 1)
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
                torch.testing.assert_close(res, expected * 2, rtol=1e-4, atol=1e-3)
        task = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
        self.assertIn("tile_dependency_root_barrier_wait", task)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_root_barrier_wait_follows_independent_loads(self) -> None:
        x = torch.randn((256,), device=DEVICE, dtype=torch.float32)
        w = torch.randn((64, 256), device=DEVICE, dtype=torch.float32)
        code, (act, out) = code_and_output(
            barrier_after_weight_load,
            (x, w),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=4,
        )
        torch.testing.assert_close(act, x + 1)
        torch.testing.assert_close(out, (w * (x + 1)[None, :]).sum(1))
        # The consumer's tasks load the weight, then wait on the producer's
        # barrier, then read what it wrote.
        task = "\n".join(
            ast.unparse(fn)
            for fn in ast.parse(code).body
            if isinstance(fn, ast.FunctionDef)
            and fn.name.startswith("tile_dependency_root_1")
        ) or ast.unparse(_generated_function(code, "_helion_barrier_after_weight_load"))
        weight, wait, read = (
            task.index("tl.load(w "),
            task.index("tile_dependency_root_barrier_wait"),
            task.index("tl.load(act "),
        )
        self.assertLess(weight, wait)
        self.assertLess(wait, read)
        # The barrier stays complete for the launch: a worker's first task waits.
        guards = [
            node
            for node in ast.walk(ast.parse(task))
            if isinstance(node, ast.If)
            and "virtual_pid ==" in ast.unparse(node.test)
            and "tile_dependency_root_barrier_wait" in ast.unparse(node)
        ]
        self.assertEqual(len(guards), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_invariant_scan_is_hoisted_with_the_guard(self) -> None:
        x = torch.randn((16, 256), device=DEVICE, dtype=torch.float32)
        for live in (0, 3, 16):
            flags = (torch.randperm(16, device=DEVICE) < live).to(torch.int32)
            code, (picked, out) = code_and_output(
                guarded_scan_gather,
                (x, flags),
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=4,
            )
            expected = torch.zeros_like(x)
            expected[:live] = x[flags.nonzero().flatten()]
            torch.testing.assert_close(picked, expected)
            torch.testing.assert_close(out, expected * 2 + expected[0][None, :])
        kernel = _generated_function(code, "_helion_guarded_scan_gather")
        # The scan feeds the bound, so it is hoisted with it and tasks keep no copy.
        prologue = ast.unparse(kernel).split("for virtual_pid")[0]
        self.assertIn("tl.associative_scan(", prologue)
        root = _generated_function(code, "tile_dependency_root_0")
        self.assertNotIn("tl.associative_scan", ast.unparse(root))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_one_wave_producer_keeps_configured_order(self) -> None:
        workers = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        if workers % 2:
            self.skipTest("needs an even SM count")
        for splits, reordered in ((workers // 2, False), (workers, True)):
            with self.subTest(splits=splits):
                x = torch.randn((2, splits), device=DEVICE, dtype=torch.float32)
                code, out = code_and_output(
                    split_merge_chain,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
                torch.testing.assert_close(out, torch.sum(x * 2, dim=1))
                self.assertIn("tile_dependency_continuation_previous", code)
                self.assertEqual(
                    "tile_dependency_scheduled_pid_task" in code, reordered
                )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_conditional_rereads_keep_final_arrival_continuation(self) -> None:
        x = torch.randn((4, 8, 8), device=DEVICE, dtype=torch.float32)
        lse = x + 1
        terms = x * 2 * (lse - lse.amax(dim=-1, keepdim=True))
        for active in (8, 5):
            with self.subTest(active=active):
                code, out = code_and_output(
                    conditional_split_merge_chain,
                    (x, torch.tensor([active], device=DEVICE), 5),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
                live = terms if active > 5 else terms[..., :5]
                torch.testing.assert_close(out, live.sum(dim=-1))
                self.assertIn("tile_dependency_continuation_previous", code)
                self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_trailing_root_starts_after_the_live_tasks_before_it(self) -> None:
        workers = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        rows = 2 * workers
        x = torch.randn((rows, 64), device=DEVICE, dtype=torch.float32)
        index = torch.arange(rows, device=DEVICE)
        for live, live2 in ((0, 0), (5, rows), (rows, 3), (workers + 7, workers - 1)):
            with self.subTest(live=live, live2=live2):
                code, (out1, out2, total) = code_and_output(
                    guarded_row_pair,
                    (
                        x,
                        (index < live).to(torch.int32),
                        (index < live2).to(torch.int32),
                    ),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
                torch.testing.assert_close(
                    out1, torch.where(index[:, None] < live, x + 1, 0)
                )
                expected2 = torch.where(index[:, None] < live2, x * 2, 0)
                torch.testing.assert_close(out2, expected2)
                torch.testing.assert_close(total, expected2.sum(0))
        # The second root's lanes start past the first root's live tasks.
        self.assertIn(
            f"({rows} - tl.minimum(tile_dependency_live_tasks, {rows}))", code
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_trailing_independent_roots_pack_behind_previous_root(self) -> None:
        workers = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        x = torch.randn((2, 3), device=DEVICE, dtype=torch.float32)
        y = torch.randn((workers - 1,), device=DEVICE, dtype=torch.float32)
        z = torch.randn((5,), device=DEVICE, dtype=torch.float32)
        config = {
            "pid_type": "persistent_blocked",
            "cross_loop_pipeline": "static",
            "num_sm_multiplier": 1,
            "num_warps": 1,
        }
        code, (out, side, other) = code_and_output(
            split_merge_then_independent, (x, y, z), **config
        )
        torch.testing.assert_close(out, torch.sum(x * 2, dim=1))
        torch.testing.assert_close(side, y + 1)
        torch.testing.assert_close(other, z - 1)
        self.assertIn("tile_dependency_continuation_previous", code)
        # y starts after the six producer tasks and wraps; z packs behind y.
        self.assertIn(f"% {workers} - 0 + 6, {workers + 5}, {workers})", code)
        end = workers + 10
        self.assertIn(f"% {workers} - 0 + {workers + 5}, {end}, {workers})", code)
        # An edge-free root before a synchronized one keeps its own wave; its
        # lanes start after the first root's six tasks.
        code, (out, other) = code_and_output(split_independent_merge, (x, z), **config)
        torch.testing.assert_close(out, torch.sum(x * 2, dim=1))
        torch.testing.assert_close(other, z - 1)
        lanes = f"(tl.program_id(0) + {workers - 6}) % {workers}"
        self.assertIn(f"{lanes} - 0 + {workers}, {workers + 5},", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partial_in_place_preserves_unowned_reaching_definition(self) -> None:
        x = torch.arange(96, device=DEVICE, dtype=torch.float32)
        for launch in range(2):
            code, out = code_and_output(
                partial_prefix_in_place_chain,
                (x + launch,),
                block_sizes=[16, 32, 16],
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )
            expected = x + launch + 1
            expected = torch.cat((expected[:64] * 2, expected[64:]))
            torch.testing.assert_close(out, expected)
        self.assertUsesExactReadiness(code)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_multi_producer_join_uses_one_readiness_counter(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32)
        y = torch.arange(128, device=DEVICE, dtype=torch.float32) + 3
        for launch in range(2):
            code, out = code_and_output(
                multi_producer_join,
                (x + launch, y + launch),
                block_sizes=[16, 16, 16],
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, x + launch + 1 + (y + launch) * 2)
        self.assertIn("tile_dependency_continuation_previous", code)
        self.assertIn("tl.cast(2, tl.uint32) - 1", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_repeated_join_waits_once_on_a_coalesced_key(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        y = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            coalesced_multi_producer_join,
            (x, y),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        expected = torch.stack([x + 1 + (y * 2)[:, None] + split for split in range(4)])
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertIn("tl.cast(5, tl.uint32)", code)
        wait_lines = [
            line
            for line in code.splitlines()
            if "tile_dependency_readiness_wait =" in line
        ]
        publication_lines = [
            line
            for line in code.splitlines()
            if "tl.atomic_add(tile_dependency_state" in line
        ]
        self.assertTrue(wait_lines)
        self.assertTrue(publication_lines)
        for line in wait_lines:
            self.assertIn(f"* {_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}]", line)
        for line in publication_lines:
            self.assertIn(f"* {_CROSS_LOOP_COUNTER_ALIGNMENT_WORDS}, 1", line)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_single_producer_fanout_waits_once_per_ready_group(self) -> None:
        x = torch.arange(8 * 4, device=DEVICE, dtype=torch.float32).reshape(8, 4)
        code, out = code_and_output(
            coalesced_single_producer_fanout,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        expected = torch.stack([x + 1 + split for split in range(4)])
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertIn("tl.cast(4, tl.uint32)", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_arrivals_of_one_task_share_one_atomic(self) -> None:
        # Each split task arrives on its row's scale counter and on the scaled
        # root's counter: one atomic over both counters, not a release atomic
        # each (the second would wait for the first one's acknowledgment).
        x = torch.randn(8, 4, 64, device=DEVICE, dtype=torch.float32)
        expected = x * 2 * ((x * 2).sum(dim=(1, 2)) + 1)[:, None, None]
        for pipeline in ("static", "dynamic"):
            code, out = code_and_output(
                split_norm_then_scale,
                (x,),
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
            self.assertIn(
                "tile_dependency_arrival_previous = tl.atomic_add(tl.where(",
                code,
                pipeline,
            )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_inline_continuation_publishes_after_its_task(self) -> None:
        # Each row task of the second loop runs inline on its producer's arrival
        # and then arrives on two counters (the pair and quad consumers): its
        # shared atomic must follow its own task, not the producer's arrivals.
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.float32)
        v = x + 1
        expected = torch.sin(v) * torch.cos(v) + torch.exp(torch.sin(v * 3)) * 2 + 1
        for _ in range(5):
            code, out = code_and_output(
                inline_then_two_rows,
                (x,),
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=4,
            )
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        scheduled = code[code.index("def tile_dependency_root_0_scheduled_task") :]
        self.assertLess(
            scheduled.index("tile_dependency_root_1("),
            scheduled.index("tile_dependency_arrival_previous = tl.atomic_add("),
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_fan_in_one_nested_continuation_needs_no_counter(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            direct_nested_continuation,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1).reshape(4, 2).sum(dim=-1) * 2)
        self.assertIn("tl.cast(2, tl.uint32) - 1", code)
        self.assertNotIn("tl.cast(1, tl.uint32) - 1", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_l2_remapped_roots_use_logical_task_readiness(self) -> None:
        for batch in (3, 4):
            with self.subTest(batch=batch):
                x = torch.arange(
                    batch * 64, device=DEVICE, dtype=torch.float32
                ).reshape(batch, 64)
                code, out = code_and_output(
                    cartesian_affine_chain,
                    (x,),
                    block_sizes=[1, 16, 1, 32],
                    l2_groupings=[2, 2],
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=1,
                )

                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertNotIn("tile_dependency_task_wait", code)
                self.assertUsesExactReadiness(code)
                self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_size_one_view_uses_task_readiness(self) -> None:
        x = torch.arange(32 * 128, device=DEVICE, dtype=torch.float32).reshape(32, 128)
        code, out = code_and_output(
            size_one_view_chain,
            (x,),
            block_sizes=[4, 1, 4, 32],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, ((x + 1) * 2).unsqueeze(0))
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_computed_index_keeps_other_dims_exact(self) -> None:
        x = torch.arange(8 * 64, device=DEVICE, dtype=torch.float32).reshape(8, 64)
        code, out = code_and_output(
            computed_index_view_chain,
            (x,),
            block_sizes=[],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        # The computed group index covers its whole dimension; the rows stay
        # exact through the other view, so each row's reader follows its four
        # writers alone.
        self.assertIn("tile_dependency_continuation", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_overshooting_tile_keeps_exact_readiness(self) -> None:
        x = torch.arange(3 * 128, device=DEVICE, dtype=torch.float32).reshape(3, 128)
        code, out = code_and_output(
            overshooting_tile_chain,
            (x,),
            # The writer's single row tile overshoots the three rows.
            block_sizes=[4, 32, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("tile_dependency_readiness_wait", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_whole_axis_read_of_a_shard_waits_per_split(self) -> None:
        # The consumer reads every row of one of its shard's splits: no single
        # (row, split) key serves it, so it waits on its split's key, which all
        # of the split's rows publish, instead of a whole-root barrier.
        x = torch.arange(8 * 4 * 64, device=DEVICE, dtype=torch.float32).reshape(
            8, 4, 64
        )
        for pipeline in ("static", "dynamic"):
            code, out = code_and_output(
                shard_of_whole_rows,
                (x, 1),
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, (x * 5).sum(dim=0)[2:])
            self.assertIn("tl.cast(8, tl.uint32)", code, pipeline)
            self.assertNotIn("tile_dependency_root_barrier", code, pipeline)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_whole_read_root_keeps_other_producers_keyed(self) -> None:
        # Every consumer task waits on all of the first root (a barrier), but
        # that must not coarsen its per-row wait on the second root.
        x = torch.arange(8 * 64, device=DEVICE, dtype=torch.float32).reshape(8, 64)
        for pipeline in ("static", "dynamic"):
            code, out = code_and_output(
                keyed_rows_beside_whole_read,
                (x,),
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, x * 2 + (x[:2] + 1).sum(dim=0))
            self.assertIn("tile_dependency_readiness_wait", code, pipeline)
            barrier_waits = [
                line
                for line in code.splitlines()
                if "root_barrier_wait" in line and " != " in line
            ]
            self.assertTrue(barrier_waits, pipeline)
            # Only the two-task root is waited on whole.
            for line in barrier_waits:
                self.assertIn("tl.cast(2, tl.uint32)", line, pipeline)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_task_invariant_wait_runs_once_per_worker(self) -> None:
        x = torch.randn((8, 64), device=DEVICE, dtype=torch.float32)
        expected = (x + 1).sum(dim=0) + (x * 2).sum(dim=0) + x
        for pipeline in ("static", "dynamic"):
            code, out = code_and_output(
                join_of_whole_reads,
                (x,),
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
            self.assertNotIn("tile_dependency_root_barrier", code, pipeline)
            task = ast.unparse(
                _generated_function(code, "tile_dependency_root_2_scheduled_task")
            )
            kernel = code[code.index("def _helion_join_of_whole_reads") :]
            if pipeline == "static":
                # The join's wait runs once, before the worker's task loop.
                self.assertNotIn("tile_dependency_readiness_wait", task)
                self.assertLess(
                    kernel.index("tile_dependency_readiness_wait ="),
                    kernel.rindex("for virtual_pid in"),
                )
            else:
                # A program running one packet waits in it (at the task's
                # first read of what the wait orders).
                root = ast.unparse(_generated_function(code, "tile_dependency_root_2"))
                self.assertIn("tile_dependency_readiness_wait", task + root)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0), reason="dot roots stay in kernel scope")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_guarded_task_publishes_relaxed_when_it_writes_nothing(self) -> None:
        x = torch.randn((128, 64), device=DEVICE, dtype=torch.float16)
        w = torch.randn((64, 32), device=DEVICE, dtype=torch.float16)
        w2 = torch.randn((32, 32), device=DEVICE, dtype=torch.float16)
        for pattern in ([1, 0, 2, 0, 1, 1, 0, 0], [0] * 8, [1] * 8):
            counts = torch.tensor(pattern, device=DEVICE, dtype=torch.int32)
            counts = counts.repeat_interleave(16)
            staged = torch.where(counts[:, None] > 0, (x.float() @ w.float()).half(), 0)
            for pipeline in ("static", "dynamic"):
                code, out = code_and_output(
                    guarded_dot_chain,
                    (x, w, w2, counts),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=4,
                )
                torch.testing.assert_close(
                    out, staged.float() @ w2.float(), rtol=1e-2, atol=1e-2
                )
                arrivals = [
                    line.strip()
                    for line in code.splitlines()
                    if "tl.atomic_add(tile_dependency_state" in line
                ]
                # A task that writes publishes with release; one that skips
                # its writes publishes relaxed (no fence on its writes).
                self.assertTrue(any("sem='release'" in a for a in arrivals))
                self.assertTrue(any("sem='relaxed'" in a for a in arrivals))

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0), reason="dot roots stay in kernel scope")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_guarded_task_releases_when_it_supersedes_a_write(self) -> None:
        # The guarded root's stores supersede the zero fill: its consumer relies
        # on the guarded task's release for the zero fill too, so a task that
        # skips its writes still publishes with release.
        x = torch.randn((128, 64), device=DEVICE, dtype=torch.float16)
        w = torch.randn((64, 32), device=DEVICE, dtype=torch.float16)
        w2 = torch.randn((32, 32), device=DEVICE, dtype=torch.float16)
        counts = torch.tensor([1, 0, 2, 0, 1, 1, 0, 0], device=DEVICE)
        counts = counts.to(torch.int32).repeat_interleave(16)
        staged = torch.where(counts[:, None] > 0, (x.float() @ w.float()).half(), 0)
        for pipeline in ("static", "dynamic"):
            code, out = code_and_output(
                zeroed_then_guarded_chain,
                (x, w, w2, counts),
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=4,
            )
            torch.testing.assert_close(
                out, staged.float() @ w2.float(), rtol=1e-2, atol=1e-2
            )
            arrivals = [
                line
                for line in code.splitlines()
                if "tl.atomic_add(tile_dependency_state" in line
            ]
            self.assertTrue(arrivals, pipeline)
            self.assertFalse(any("sem='relaxed'" in a for a in arrivals), pipeline)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((10, 0), reason="dot roots stay in kernel scope")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_guarded_task_waits_after_its_guard(self) -> None:
        # A task whose guard fails writes nothing and reads nothing the wait
        # on ``scaled`` protects, so the wait follows the guard; a guard that
        # reads ``scaled`` keeps the wait before it, as does a guarded root
        # whose writes supersede the zero fill (its consumer relies on it).
        x = torch.randn((128, 64), device=DEVICE, dtype=torch.float16)
        w = torch.randn((64, 32), device=DEVICE, dtype=torch.float16)
        w2 = torch.randn((32, 32), device=DEVICE, dtype=torch.float16)
        counts = torch.tensor([1, 0, 2, 0, 1, 1, 0, 0], device=DEVICE)
        counts = counts.to(torch.int32).repeat_interleave(16)
        rows_live = (x[::16, 0] * 2 > 0).repeat_interleave(16)
        for kernel, args, live, scale, after_guard in (
            (scaled_then_guarded_chain, (x, w, w2, counts), counts > 0, 2, True),
            (scaled_then_self_guarded_chain, (x, w, w2), rows_live, 2, False),
            (zeroed_then_guarded_chain, (x, w, w2, counts), counts > 0, 1, False),
        ):
            product = (x.float() * scale @ w.float()).half()
            staged = torch.where(live[:, None], product, 0)
            for pipeline in ("static", "dynamic"):
                code, out = code_and_output(
                    kernel,
                    args,
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=4,
                )
                torch.testing.assert_close(
                    out, staged.float() @ w2.float(), rtol=1e-2, atol=1e-2
                )
                (guard,) = [
                    node
                    for node in ast.walk(ast.parse(code))
                    if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and not node.test.id.startswith("tile_dependency")
                    and "acc =" in ast.unparse(node)
                ]
                waits = "tile_dependency_readiness_wait" in ast.unparse(guard)
                self.assertEqual(waits, after_guard, (kernel.name, pipeline))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_masked_store_keeps_task_dependencies(self) -> None:
        # A masked store writes a subset of its tile: the consumer still waits
        # per task, not for the whole producer root.
        x = torch.arange(1024, device=DEVICE, dtype=torch.float32)
        keep = torch.randint(0, 2, (1024,), device=DEVICE, dtype=torch.int32)
        for pipeline in ("static", "dynamic"):
            code, out = code_and_output(
                masked_rows_chain,
                (x, keep),
                block_sizes=[64, 16],
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, torch.where(keep != 0, (x + 1) * 2, 0))
            self.assertIn("tile_dependency_readiness_wait", code, pipeline)
            self.assertNotIn("tile_dependency_root_barrier", code, pipeline)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="Packed BF16 conversion requires Blackwell"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_flat_load_waits_for_its_writer(self) -> None:
        # A flat load's footprint is its whole tensor: it waits for every task
        # of the root that writes it.
        x = torch.randn(4096, device=DEVICE, dtype=torch.bfloat16)
        offsets = torch.randperm(256, device=DEVICE)
        code, out = code_and_output(
            flat_load_after_store,
            (x, offsets),
            block_sizes=[64, 4],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )
        expected = (x + 1).view(256, 16)[offsets].to(torch.float16)
        torch.testing.assert_close(out, expected)
        self.assertIn("tile_dependency_root_barrier_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_regrouped_view_read_waits_per_split(self) -> None:
        # Producer and consumer index different views of one buffer, and each
        # consumer task reads a strided box (every row of one split): on the
        # axes both views share, it still reads one producer split.
        x = torch.arange(8 * 8 * 64, device=DEVICE, dtype=torch.float32).reshape(
            8, 8, 64
        )
        for pipeline in ("static", "dynamic"):
            code, out = code_and_output(
                regrouped_view_of_whole_rows,
                (x, 1),
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, (x * 5).sum(dim=0)[4:])
            self.assertIn("tl.cast(8, tl.uint32)", code, pipeline)
            self.assertNotIn("tile_dependency_root_barrier", code, pipeline)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nonzero_grid_start_uses_root_barrier(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32)
        bound = offset_affine_chain.bind((x,))
        assert bound.host_function is not None
        dependency_graph = bound.host_function.device_ir.tile_dependency_graph
        assert dependency_graph is not None

        code, out = code_and_output(
            offset_affine_chain,
            (x,),
            block_sizes=[16, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x[32:] + 1) * 2)
        self.assertIn("tile_dependency_root_barrier", code)
        worker_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        padded_second_root_base = -(-254 // worker_count) * worker_count
        root = _generated_function(code, "tile_dependency_root_1")
        pid_assignment = next(
            node
            for node in root.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pid_shared"
                for target in node.targets
            )
        )
        self.assertIn(
            "virtual_pid",
            {
                node.id
                for node in ast.walk(pid_assignment.value)
                if isinstance(node, ast.Name)
            },
        )
        constants = {
            node.value
            for node in ast.walk(pid_assignment.value)
            if isinstance(node, ast.Constant) and isinstance(node.value, int)
        }
        self.assertIn(254, constants)
        self.assertIn(padded_second_root_base, constants)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_specialized_quotient_retains_static_task_geometry(self) -> None:
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        codes: list[str] = []
        cache_keys: list[str | None] = []
        cubin_hashes: set[str] = set()
        bounds = []
        for numerator, denominator in ((8, 2), (12, 3)):
            code, out = code_and_output(
                specialized_quotient_chain,
                (x, numerator, denominator),
                pid_type="persistent_blocked",
                cross_loop_pipeline="static",
                num_sm_multiplier=1,
                num_warps=1,
            )
            bound = specialized_quotient_chain.bind((x, numerator, denominator))
            config, compiled = next(iter(bound._compile_cache.items()))
            triton_kernel = compiled.__globals__[f"_helion_{bound.kernel.name}"]
            cubin_hashes.update(
                compiled_kernel.hash
                for cache_tuple in triton_kernel.device_caches.values()
                for compiled_kernel in cache_tuple[0].values()
                if getattr(compiled_kernel, "hash", None) is not None
            )
            codes.append(code)
            cache_keys.append(bound.backend_cache_key(config))
            bounds.append(bound)
            torch.testing.assert_close(out, (x + 1) * 2)
            self.assertIn("tile_dependency_continuation_task", code)
            self.assertNotIn("tile_dependency_root_barrier", code)

        self.assertIsNot(bounds[0], bounds[1])
        self.assertEqual(codes[0], codes[1])
        self.assertIsNotNone(cache_keys[0])
        self.assertEqual(cache_keys[0], cache_keys[1])
        self.assertEqual(len(cubin_hashes), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_exact_readiness_follows_each_roots_pid_order(self) -> None:
        # Exercise exact readiness across distinct configured PID orders. The
        # worker count may select either continuation or counter lowering.
        x = torch.arange(4 * 256, device=DEVICE, dtype=torch.float32).reshape(4, 256)
        code, out = code_and_output(
            cartesian_affine_chain,
            (x,),
            block_sizes=[1, 16, 1, 32],
            loop_orders=[[1, 0], [0, 1]],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertUsesExactReadiness(code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_cartesian_join_combines_both_producers(self) -> None:
        x = torch.arange(128, device=DEVICE, dtype=torch.float32).reshape(2, 64)
        code, out = code_and_output(
            cartesian_affine_join,
            (x,),
            block_sizes=[1, 16, 1, 16, 1, 32],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, x * 2)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertUsesExactReadiness(code)
        self.assertIn("tl.cast(4, tl.uint32)", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_singleton_root_waits_for_multiple_producers(self) -> None:
        x = torch.arange(64, device=DEVICE, dtype=torch.float32).reshape(1, 64)
        code, out = code_and_output(
            singleton_root_join,
            (x,),
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, torch.sum(x * 2, dim=-1))
        self.assertUsesExactReadiness(code)
        self.assertIn("tl.cast(8, tl.uint32)", code)
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_singleton_stream_uses_nested_split_nested_loop_at_readiness(self) -> None:
        for batch in (1, 2):
            with self.subTest(batch=batch):
                x = torch.arange(
                    batch * 4096, device=DEVICE, dtype=torch.float32
                ).reshape(batch, 4096)
                code, out = code_and_output(
                    streamed_singleton_reduction,
                    (x,),
                    block_sizes=[1, 16],
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=1,
                )

                torch.testing.assert_close(out, torch.sum(x + 1, dim=-1) + x[:, 0] + 1)
                self.assertIn("tile_dependency_nested_loop_wait", code)
                self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_real_planner_declines_unproved_resident_nested_interleave(self) -> None:
        x = torch.arange(8192, device=DEVICE, dtype=torch.float32).reshape(1, 8192)
        code, out = code_and_output(
            streamed_singleton_reduction,
            (x,),
            block_sizes=[1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=2,
            num_warps=1,
        )

        torch.testing.assert_close(out, torch.sum(x + 1, dim=-1) + x[:, 0] + 1)
        kernel = _generated_function(code, "_helion_streamed_singleton_reduction")
        scheduled_calls = [
            _call_name(call)
            for call in sorted(
                (
                    node
                    for node in ast.walk(kernel)
                    if isinstance(node, ast.Call)
                    and (_call_name(node) or "").endswith("_scheduled_task")
                ),
                key=lambda node: (node.lineno, node.col_offset),
            )
        ]
        # A CTA-level rank cannot justify admitting a resident nested waiter
        # before a later producer wave.  Until the scheduler models internal
        # checkpoints for arbitrary resident roots, retain the baseline order.
        self.assertEqual(
            scheduled_calls,
            [
                "tile_dependency_root_0_scheduled_task",
                "tile_dependency_root_1_scheduled_task",
            ],
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_fixed_block_size_dense_span_is_task_ready(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            fixed_block_dense_span_chain,
            (x,),
            block_sizes=[],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        # arange(16) traces as a constant; block_size=1 keeps a symbolic var.
        self.assertNotIn("tile_dependency_root_barrier", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_nested_wait_does_not_cover_an_earlier_access(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        code, out = code_and_output(
            prewait_singleton_reduction,
            (x,),
            block_sizes=[1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, torch.sum(x + 1, dim=-1) + x[:, 0] + 1)
        # The early read keeps its own entry wait (one key: once per worker,
        # before the root's task loop); the nested wait follows it.
        self.assertNotIn("tile_dependency_root_barrier", code)
        kernel = ast.unparse(
            _generated_function(code, "_helion_prewait_singleton_reduction")
        )
        self.assertLess(
            kernel.index("tile_dependency_readiness_wait"),
            kernel.index("tile_dependency_root_1_scheduled_task("),
        )
        root = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
        self.assertLess(
            root.index("first = "), root.index("tile_dependency_nested_loop_wait")
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_multiple_nested_loops_share_one_scheduled_root_task(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        code, out = code_and_output(
            streamed_sibling_reductions,
            (x,),
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(
            out,
            torch.sum(x + 1, dim=-1) + torch.sum(x * 2, dim=-1),
        )
        self.assertGreaterEqual(code.count("tile_dependency_nested_loop_wait"), 2)
        self.assertNotIn("tile_dependency_root_barrier_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_task_events_are_capture_safe(self) -> None:
        for pipeline in ("static", "dynamic"):
            with self.subTest(pipeline=pipeline):
                x = torch.arange(128, device=DEVICE, dtype=torch.float32).reshape(2, 64)
                bound = cartesian_affine_chain.bind((x,))
                config = helion.Config(
                    block_sizes=[1, 16, 1, 32],
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=4,
                    num_warps=8,
                )
                code = bound.to_triton_code(config)
                self.assertNotIn("launch_cooperative_grid=True", code)
                compiled = bound.compile_config(config)
                compiled(x)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = compiled(x)
                for value in (3.0, 7.0, -2.0):
                    x.fill_(value)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(captured, (x + 1) * 2)

    # Six static-pipeline schedules, each proved and generated twice by
    # code_and_output, spend ~30s in host codegen on an idle machine; CI's
    # four workers per GPU push that past the default 60s timeout, and
    # pytest-timeout then kills the worker.
    @pytest.mark.timeout(300)
    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_grouped_schedule_requires_the_proven_access_order(self) -> None:
        torch.manual_seed(0)
        block_sizes = [1, 16, 1, 16]

        for batch, intermediate, group_size, reverse_groups in (
            (1, 128, 32, False),
            (1, 128, 32, True),
            (2, 128, 32, False),
            (4, 128, 32, False),
            (2, 96, 32, False),
            (2, 128, 64, False),
        ):
            with self.subTest(
                batch=batch,
                intermediate=intermediate,
                group_size=group_size,
                reverse_groups=reverse_groups,
            ):
                # Positive inputs avoid cancellation-dominated relative error;
                # this test is intended to catch readiness failures.
                x = torch.rand((batch, 64), device=DEVICE, dtype=torch.float16)
                w13 = torch.rand(
                    (64, 2 * intermediate), device=DEVICE, dtype=torch.float16
                )
                w2 = torch.rand((intermediate, 64), device=DEVICE, dtype=torch.float16)
                kernel_args = (
                    x,
                    w13,
                    w2,
                    group_size,
                    hl.constexpr(reverse_groups),
                )
                if batch == 1 and not reverse_groups:
                    bound = grouped_affine_chain.bind(kernel_args)
                    assert bound.host_function is not None
                    dependency_graph = (
                        bound.host_function.device_ir.tile_dependency_graph
                    )
                    assert dependency_graph is not None
                    self.assertTrue(
                        all(
                            access.root in (0, 1, 2)
                            for access in dependency_graph.accesses
                        )
                    )
                    downstream_edges = tuple(
                        edge
                        for edge in dependency_graph.edges
                        if edge.producer_root == 1 and edge.consumer_root == 2
                    )
                    self.assertEqual(len(downstream_edges), 2)
                    nested_site_ids = {
                        site.site_id
                        for edge in downstream_edges
                        for dependency in edge.access_dependencies
                        for site_id in dependency_graph.site_ids_by_access[
                            dependency.consumer_access_id
                        ]
                        for site in (dependency_graph.execution_sites[site_id],)
                        if not site.is_root
                    }
                    self.assertEqual(len(nested_site_ids), 1)
                code, out = code_and_output(
                    grouped_affine_chain,
                    kernel_args,
                    block_sizes=block_sizes,
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="static",
                    num_sm_multiplier=1,
                    num_warps=4,
                    num_stages=2,
                )

                gate_up = (x.float() @ w13.float()).half()
                gate, up = gate_up.chunk(2, dim=-1)
                groups = intermediate // group_size
                if reverse_groups:
                    gate = (
                        gate.reshape(batch, groups, group_size)
                        .flip(1)
                        .reshape(batch, intermediate)
                    )
                    up = (
                        up.reshape(batch, groups, group_size)
                        .flip(1)
                        .reshape(batch, intermediate)
                    )
                activated = gate.float() * up.float()
                scale = (
                    activated.abs().reshape(batch, groups, group_size).amax(dim=-1) + 1
                )
                activation = activated.half()
                expected = (
                    activation.float().reshape(batch, groups, group_size)
                    * scale[:, :, None]
                ).reshape(batch, intermediate) @ w2.float()
                torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)

                if reverse_groups:
                    self.assertIn("tile_dependency_root_barrier", code)
                else:
                    self.assertNotIn("tile_dependency_root_barrier", code)
                    self.assertTrue(
                        "tile_dependency_continuation_previous" in code
                        or "tile_dependency_readiness_wait" in code
                    )
                    self.assertIn("tile_dependency_nested_loop_wait", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_pipeline_uses_exact_nested_loop_wait(self) -> None:
        torch.manual_seed(0)
        x = torch.rand((1, 64), device=DEVICE, dtype=torch.float16)
        w13 = torch.rand((64, 256), device=DEVICE, dtype=torch.float16)
        w2 = torch.rand((128, 64), device=DEVICE, dtype=torch.float16)
        kernel_args = (x, w13, w2, 32, hl.constexpr(False))
        bound = grouped_affine_chain.bind(kernel_args)
        self.assertNotIn(
            "cross_loop_num_workers",
            bound.config_spec.user_defined_tunables,
        )
        invalid_config = dict(bound.config_spec.default_config())
        invalid_config["cross_loop_num_workers"] = 3
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "Invalid config keys"):
            bound.config_spec.normalize(invalid_config)

        code, out = code_and_output(
            grouped_affine_chain,
            kernel_args,
            block_sizes=[1, 16, 1, 16],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
            num_warps=4,
            num_stages=2,
        )

        gate_up = (x.float() @ w13.float()).half()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = gate.float() * up.float()
        scale = activated.abs().reshape(1, 4, 32).amax(dim=-1) + 1
        expected = (
            activated.half().float().reshape(1, 4, 32) * scale[:, :, None]
        ).reshape(1, 128) @ w2.float()
        torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)
        self.assertNotIn("tile_dependency_root_barrier", code)
        self.assertUsesExactReadiness(code)
        self.assertIn("tile_dependency_nested_loop_wait", code)
