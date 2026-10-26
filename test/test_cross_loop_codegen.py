from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
from functools import partial
import itertools
import re
from types import SimpleNamespace
from typing import Any
from typing import Callable
from typing import Literal
from typing import cast
from unittest import mock

import pytest
import sympy
import torch
import triton
import triton.language as tl

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
from helion._compiler.cross_loop_codegen import _insert_before_first_use
from helion._compiler.cross_loop_codegen import _insert_producer_waits
from helion._compiler.cross_loop_codegen import _triton_root_requires_kernel_scope
from helion._compiler.device_function import DeviceFunction
from helion._compiler.program_id import PIDInfo
from helion._compiler.tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import DenseTaskOrder
from helion._compiler.tile_dependency import Incidence
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
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
def pdl_exit_trigger_chain(x: torch.Tensor) -> torch.Tensor:
    """Wait per task, not at entry, and trigger after the cross-loop state."""
    first = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_batch, producer_width in hl.tile(x.size()):
        hl.inline_triton("tl.extra.cuda.gdc_wait()\n0", args=(), output_like=None)
        first[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch, consumer_width in hl.tile(x.size(), block_size=[1, 16]):
        out[consumer_batch, consumer_width] = first[consumer_batch, consumer_width] * 2
    hl.pdl_launch_dependents()
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
def fanout_and_keyed_join(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A small root every consumer task reads whole (a routing table) and a keyed
    # root with several tasks per key (FC1 units per expert slot): the consumer
    # (FC2) must keep its keyed wait on the keyed root.
    n = x.size(0)
    rows, width = y.shape
    quarters = 4
    small = torch.empty_like(x)
    keyed = torch.empty_like(y)
    out = torch.empty_like(y)
    for tile in hl.tile(n, block_size=1):
        small[tile] = x[tile] + 1
    for tile_q, tile_r in hl.tile([quarters, rows], block_size=[1, 1]):
        cols = tile_q.begin * (width // quarters) + hl.arange(width // quarters)
        keyed[tile_r, cols] = y[tile_r, cols] * 2 + torch.sum(small[:], dim=0)
    for tile_r in hl.tile(rows, block_size=1):
        out[tile_r, :] = keyed[tile_r, :] + torch.sum(small[:], dim=0)
    return out


def _short_then_long_sinks_impl(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # A feeds two sinks written short-task first: B's tasks load 9 floats, C's a
    # 512-float row (+ all of A).  Nothing waits on B or C.
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    c = torch.empty_like(y)
    for tile in hl.tile(x.size(0), block_size=1):
        a[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0), block_size=1):
        b[tile] = x[tile] + torch.sum(a[:], dim=0)
    for tile in hl.tile(y.size(0), block_size=1):
        c[tile, :] = y[tile, :] * 2 + torch.sum(a[:], dim=0)
    return b, c


short_then_long_sinks = helion.kernel(
    _short_then_long_sinks_impl, static_shapes=True, autotune_effort="none"
)
short_then_long_sinks_source_order = helion.kernel(
    _short_then_long_sinks_impl,
    static_shapes=True,
    autotune_effort="none",
    cross_loop_sink_lpt_order=False,
)

_GLOBALTIMER_SPIN = (
    "{ .reg .u64 t0, t1; .reg .pred p; mov.u64 t0, %globaltimer; "
    "SPIN: mov.u64 t1, %globaltimer; sub.u64 t1, t1, t0; "
    "setp.lt.u64 p, t1, $1; @p bra SPIN; mov.u32 $0, 0; }"
)
# The same spin in every warp but the first: warp 0 runs ahead to its stores
# (and to a publication right behind them) while warps 1.. still spin.
_WARP_SKEWED_SPIN = (
    "{ .reg .u64 t0, t1; .reg .pred p, w; .reg .u32 tid; mov.u32 tid, %tid.x; "
    "setp.lt.u32 w, tid, 32; @w bra SKEW_DONE; mov.u64 t0, %globaltimer; "
    "SKEW: mov.u64 t1, %globaltimer; sub.u64 t1, t1, t0; "
    "setp.lt.u64 p, t1, $1; @p bra SKEW; SKEW_DONE: mov.u32 $0, 0; }"
)


def _warp_skewed_publisher_impl(
    x: torch.Tensor, y: torch.Tensor, delay: torch.Tensor
) -> torch.Tensor:
    # P: one element per thread (128-element tiles, 4 warps), stored with no
    # layout conversion (no CTA barrier of Triton's own) right after the
    # skewed spin.  C reads all of y, so it waits for P's root barrier, whose
    # release thread 0 issues: only a CTA barrier in front of that release
    # orders the stores of warps 1-3 before it.
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=128):
        spun = hl.inline_asm_elementwise(
            _WARP_SKEWED_SPIN,
            "=r,l",
            [delay[0]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        y[tile] = x[tile] + 1 + spun.to(torch.float32)
    for tile in hl.tile(x.size(0), block_size=128):
        out[tile] = y[tile] + torch.sum(y[:], dim=0) * 0.0
    return out


@triton.jit
def _tma_store_rows(y_ptr, row0, value):
    descriptor = tl.make_tensor_descriptor(
        y_ptr,
        shape=[65, 128],  # pyrefly: ignore[bad-argument-type]
        strides=[128, 1],  # pyrefly: ignore[bad-argument-type]
        block_shape=[32, 128],  # pyrefly: ignore[bad-argument-type]
    )
    descriptor.store([row0, 0], value)  # pyrefly: ignore[bad-argument-type]


def _callee_tma_store_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # P writes rows of y only through a TMA store inside a user Triton
    # function (an async-proxy write the root body does not spell) and the pad
    # row generically; C reads all of y.
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0), block_size=32):
        value = x[tile, :] + 1.0
        hl.triton_kernel(_tma_store_rows, args=(y, tile.begin, value), output_like=None)
        y[x.size(0), :] = hl.zeros([x.size(1)], dtype=torch.float32) + 7.0
    for tile in hl.tile(x.size(0), block_size=32):
        out[tile, :] = (
            y[tile, :]
            + torch.sum(y[0 : x.size(0), :], dim=0)[None, :] * 0.0
            + y[x.size(0), :][None, :] * 0.0
        )
    return out


def _grouped_source_impl(
    w: torch.Tensor, x: torch.Tensor, parts: torch.Tensor, delay: torch.Tensor
) -> torch.Tensor:
    # S (64 tasks, a 64 KiB fp32 tile + the 8-byte delay scalar each, no
    # producer) writes one partial per task after spinning `delay` ns; C reads
    # every partial.  Its first wave puts 64 x 65544 B (just over 4 MiB) in
    # flight, so under a 1 MiB + 4 KiB source budget S claims 4 consecutive
    # tasks per ticket.
    out = torch.empty_like(x)
    for tile in hl.tile(w.size(0), block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [delay[0]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        parts[tile] = torch.sum(torch.sum(w[tile, :, :], dim=-1), dim=-1) + spun.to(
            torch.float32
        )
    for tile in hl.tile(x.size(0), block_size=1):
        out[tile] = x[tile] + torch.sum(parts[:], dim=0)
    return out


grouped_source = helion.kernel(
    _grouped_source_impl,
    static_shapes=True,
    autotune_effort="none",
    cross_loop_source_inflight_budget=(1 << 20) + (1 << 12),
    cross_loop_source_max_group=4,
)
capped_source = helion.kernel(
    _grouped_source_impl,
    static_shapes=True,
    autotune_effort="none",
    cross_loop_source_inflight_budget=(1 << 20) + (1 << 12),
)
ungrouped_source = helion.kernel(
    _grouped_source_impl,
    static_shapes=True,
    autotune_effort="none",
    cross_loop_source_inflight_budget=0,
)


def _transitive_chain_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    small: torch.Tensor,
    delay: torch.Tensor,
    middle_reads_small: hl.constexpr,
    small_first: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    # S writes `small` late (spins `delay` ns first); M reads all of it (when
    # middle_reads_small) and publishes keyed rows; C reads its row of M and all
    # of `small`.  C's keyed wait on M certifies S transitively.  small_first:
    # C reads `small` before M's rows (certifying S through the M wait would
    # move a sunk wait up).
    n = x.size(0)
    rows, width = y.shape
    quarters = 4
    keyed = torch.empty_like(y)
    out = torch.empty_like(y)
    for tile in hl.tile(n, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [delay[0]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        small[tile] = x[tile] + 1 + spun.to(torch.float32)
    for tile_q, tile_r in hl.tile([quarters, rows], block_size=[1, 1]):
        cols = tile_q.begin * (width // quarters) + hl.arange(width // quarters)
        if middle_reads_small:
            keyed[tile_r, cols] = y[tile_r, cols] * 2 + torch.sum(small[:], dim=0)
        else:
            keyed[tile_r, cols] = y[tile_r, cols] * 2
    # Two C tasks per M key: each first reads data no root writes, then M's row
    # half and all of `small` (certified only through M), in either order.
    for tile_h, tile_c in hl.tile([2, rows], block_size=[1, 1]):
        half = tile_h.begin * (width // 2) + hl.arange(width // 2)
        base = y[tile_c, half] * 0
        if small_first:
            small_sum = torch.sum(small[:], dim=0)
            kept = keyed[tile_c, half]
        else:
            kept = keyed[tile_c, half]
            small_sum = torch.sum(small[:], dim=0)
        out[tile_c, half] = base + small_sum + kept
    return out


def _war_after_producer_read_impl(
    x: torch.Tensor,
    x_store: torch.Tensor,
    y: torch.Tensor,
    delay: torch.Tensor,
) -> torch.Tensor:
    # P reads x late (a spin feeds the load address) and writes y = x + 1.  C
    # first overwrites its tile of x through `x_store` (x itself, or a view of
    # its storage: a write-after-read on P's input), then reads all of y (C's
    # root barrier on P).  Sequential semantics: y = x0 + 1 everywhere and x
    # ends -1; a C store that wins the race against a late P read corrupts y.
    n = x.size(0)
    out = torch.empty_like(y)
    for tile in hl.tile(n, block_size=64):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [delay[0]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        y[tile] = x[tile.index + spun] + 1.0
    for tile_c in hl.tile(n, block_size=64):
        x_store[tile_c] = hl.zeros([tile_c], dtype=torch.float32) - 1.0
        out[tile_c] = hl.zeros([tile_c], dtype=torch.float32) + torch.sum(y[:], dim=0)
    return out


def _dot_branch_transitive_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    small: torch.Tensor,
    flags: torch.Tensor,
    delay: torch.Tensor,
    keyed: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    # S writes `small` late.  M (a dot root: kernel scope on sm100) reads all
    # of `small` only under a runtime branch and publishes keyed rows; C reads
    # its rows of M and all of `small`.  A transitive cover of S through M's
    # keyed counter is unsound when M's barrier wait on S lands in the branch
    # (static first-use sinking) and the branch is not taken.
    n = x.size(0)
    rows, width = y.shape
    out = torch.empty_like(y)
    for tile in hl.tile(n, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [delay[0]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        small[tile] = x[tile] + 1 + spun.to(torch.float32)
    for tile_r in hl.tile(rows, block_size=16):
        on = torch.sum(flags[:])
        base = torch.matmul(y[tile_r, :], w[:, :])
        if on > 0:
            keyed[tile_r, :] = base + torch.sum(small[:], dim=0)
        else:
            keyed[tile_r, :] = base
    for tile_h, tile_c in hl.tile([2, rows], block_size=[1, 16]):
        half = tile_h.begin * (width // 2) + hl.arange(width // 2)
        kept = keyed[tile_c, half]
        small_sum = torch.sum(small[:], dim=0)
        out[tile_c, half] = small_sum + kept
    return out


def _guarded_transitive_chain_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    small: torch.Tensor,
    flags: torch.Tensor,
    delay: torch.Tensor,
    mid: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    # 32 x 512.  S late -> K per row (keyed counter to M) -> M guarded by a
    # runtime live-row count -> C (16x16 dot tiles) reads M's rows and all of
    # `small`.  Guard-skipped M tasks of a static pipeline publish their
    # arrivals without running K's wait, so a cover of S through M's counter
    # is unsound there.
    n = x.size(0)
    rows, width = y.shape
    kk = torch.empty_like(y)
    out = torch.empty_like(y)
    for tile in hl.tile(n, block_size=1):
        spun = hl.inline_asm_elementwise(
            _GLOBALTIMER_SPIN,
            "=r,l",
            [delay[0]],
            dtype=torch.int32,
            is_pure=False,
            pack=1,
        )
        small[tile] = x[tile] + 1 + spun.to(torch.float32)
    for tile_k in hl.tile(rows, block_size=1):
        kk[tile_k, :] = y[tile_k, :] + torch.sum(small[:], dim=0)
    for tile_c, tile_r in hl.tile([width, rows], block_size=[128, 1]):
        live = torch.sum((flags[:] > 0).to(torch.int32))
        if tile_r.begin < live:
            mid[tile_r, tile_c] = kk[tile_r, tile_c] * 2
    for tile_o, tile_n in hl.tile([rows, width], block_size=[16, 16]):
        prod = torch.matmul(mid[tile_o, :], w[:, tile_n])
        out[tile_o, tile_n] = prod + torch.sum(small[:], dim=0)
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
def partial_row_tile_absorb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, rows, width = x.size()
    _, used_rows, cols = w.size()
    q = torch.empty_like(x)
    out = torch.empty([heads, cols, width], dtype=x.dtype, device=x.device)
    for head_tile, row_tile in hl.tile([heads, rows], block_size=[1, None]):
        q[head_tile.begin, row_tile, :] = x[head_tile.begin, row_tile, :] * 2
    # Each consumer task reads only the leading ``used_rows`` of its head.
    for head_tile, col_tile in hl.tile([heads, cols], block_size=[1, 64]):
        acc = hl.zeros([col_tile, width], dtype=torch.float32)
        for used_tile in hl.tile(used_rows, block_size=32):
            acc = hl.dot(
                w[head_tile.begin, used_tile, col_tile].permute(1, 0),
                q[head_tile.begin, used_tile, :],
                acc=acc,
            )
        out[head_tile.begin, col_tile, :] = acc.to(out.dtype)
    return out


def _domain_points(domain: CoordinateDomain) -> list[dict[int, int]]:
    counts = domain.axis_counts
    return [
        dict(zip(domain.axis_order, values, strict=True))
        for values in itertools.product(
            *(range(counts[axis]) for axis in domain.axis_order)
        )
    ]


def _related_points(
    relation: CoordinateRelation, point: dict[int, int]
) -> set[tuple[int, ...]]:
    """Enumerate one source point's targets, clipped to the target domain."""
    substitutions = {
        coordinate_axis_symbol(axis): sympy.Integer(value)
        for axis, value in point.items()
    }

    def value(expression: object) -> int:
        return int(sympy.sympify(expression).xreplace(substitutions))

    counts = relation.target_domain.axis_counts
    related: set[tuple[int, ...]] = set()
    for piece in relation.pieces:
        if all(
            value(begin) <= point[axis] < value(end)
            and (point[axis] - value(begin)) % step == 0
            for axis, begin, end, step in piece.source_bounds_items
        ):
            related.update(
                itertools.product(
                    *(
                        [
                            target
                            for target in range(value(begin), value(end), step)
                            if 0 <= target < counts[axis]
                        ]
                        for axis, begin, end, step in piece.target_ranges
                    )
                )
            )
    return related


def _readiness_arrival_mismatches(
    plan: cross_loop_scheduler.StaticPipelinePlan,
) -> tuple[list[str], int]:
    """Compare every producer arm's publications with its exact-wait count.

    Also returns the number of arms in which some producer items publish
    nothing.
    """
    mismatches: list[str] = []
    partial_arms = 0
    for counter in plan.readiness_counters:
        key_domain = counter.readiness_key_domain
        for producer in counter.producers:
            incidence = producer.incidence
            assert incidence.keys_by_item is not None
            assert incidence.count_by_key is not None
            arrivals: collections.Counter[tuple[int, ...]] = collections.Counter()
            items = _domain_points(incidence.items_by_key.target_domain)
            silent = 0
            for item in items:
                keys = _related_points(incidence.keys_by_item, item)
                if len(keys) > 1:
                    mismatches.append(f"item {item} publishes to {sorted(keys)}")
                silent += not keys
                arrivals.update(keys)
            partial_arms += silent > 0
            for key in _domain_points(key_domain):
                key_index = tuple(key[axis] for axis in key_domain.axis_order)
                ((expected,),) = _related_points(incidence.count_by_key, key)
                fiber = len(_related_points(incidence.items_by_key, key))
                if not arrivals[key_index] == expected == fiber:
                    mismatches.append(
                        f"root {producer.producer_root} key {key_index}: "
                        f"{arrivals[key_index]} arrivals, waits for {expected}, "
                        f"fiber has {fiber} items"
                    )
    return mismatches, partial_arms


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


_GLOBALTIMER_SPIN_20US = """
t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
t1 = t0
while t1 - t0 < 20000:
    t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
0
"""


def late_producer_chain(
    x: torch.Tensor,
    w: torch.Tensor,
    flag: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Two producers (q written 20 us late) feed a Blackwell dot root that
    reads ``a`` first and ``q`` only where ``flag`` is set; the last root reads
    ``q`` through the dot root's publication alone (no barrier of its own)."""
    tokens, k = x.size()
    n = w.size(0)
    mid = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        q[t, :] = x[t, :] * 2
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
        if flag[0] > 0:
            vq = hl.load(q, [rows, hl.arange(k)], extra_mask=live[:, None])
            acc = hl.dot(wt, vq.permute(1, 0), acc=acc)
        mid[n_t, :] = acc
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        vq = hl.load(q, [rows, n_t.begin + hl.arange(16)], extra_mask=live[:, None])
        out[n_t, :] = mid[(n - 16 - n_t.begin) + hl.arange(16), :] + vq.permute(
            1, 0
        ).to(torch.float32)
    return out


def late_producer_branches(
    x: torch.Tensor,
    w: torch.Tensor,
    flag: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """Like :func:`late_producer_chain`, but both branches of the dot root
    read ``q`` (adding or subtracting its product)."""
    tokens, k = x.size()
    n = w.size(0)
    mid = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        q[t, :] = x[t, :] * 2
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
        if flag[0] > 0:
            vq = hl.load(q, [rows, hl.arange(k)], extra_mask=live[:, None])
            acc = hl.dot(wt, vq.permute(1, 0), acc=acc)
        else:
            vq = hl.load(q, [rows, hl.arange(k)], extra_mask=live[:, None])
            acc = acc - hl.dot(wt, vq.permute(1, 0), out_dtype=torch.float32)
        mid[n_t, :] = acc
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        vq = hl.load(q, [rows, n_t.begin + hl.arange(16)], extra_mask=live[:, None])
        out[n_t, :] = mid[(n - 16 - n_t.begin) + hl.arange(16), :] + vq.permute(
            1, 0
        ).to(torch.float32)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def column_half_producers(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    tokens, k = x.size()
    n = w.size(0)
    half = k // 2
    c = torch.empty_like(x)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        c[t, :half] = x[t, :half] + 1
    for t in hl.tile(tokens, block_size=1):
        c[t, half:] = x[t, half:] * 2
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        vc = hl.load(c, [rows, hl.arange(k)], extra_mask=live[:, None])
        out[n_t, :] = hl.dot(w[n_t, :], vc.permute(1, 0), out_dtype=torch.float32)
    return out


def late_pair_producers(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Two independent producers (``b`` written 20 us late) feed a root whose
    outlined task body first reads both: one combined poll."""
    tokens, k = x.size()
    out = torch.empty([k, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        b[t, :] = x[t, :] * 2
    for n_t in hl.tile(k, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        va = hl.load(a, [rows, n_t.begin + hl.arange(16)], extra_mask=live[:, None])
        vb = hl.load(b, [rows, n_t.begin + hl.arange(16)], extra_mask=live[:, None])
        out[n_t, :] = (va.to(torch.float32) + vb.to(torch.float32)).permute(1, 0)
    return out


def transitive_pre_chain(
    x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """``a`` (written 20 us late) reaches the dot root only through the root
    that reads all of it: the dot root's edge from its writer is pruned, and
    it reads ``a`` before ``b``."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] + torch.sum(a[:, :], dim=0)
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
        vb = hl.load(b, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, vb.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def transitive_split_chain(
    x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Like :func:`transitive_pre_chain`, plus a fast producer of ``c`` that
    the dot root reads first: it reads ``c``, then ``a``, then ``b``."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] + torch.sum(a[:, :], dim=0)
    for t in hl.tile(tokens, block_size=1):
        c[t, :] = x[t, :] * 3
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        vc = hl.load(c, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, vc.permute(1, 0), out_dtype=torch.float32)
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, va.permute(1, 0), acc=acc)
        vb = hl.load(b, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, vb.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def war_pre_chain(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """The dot root overwrites ``s``, which its producer (20 us late) reads,
    before it reads that producer's output."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        a[t, :] = s[t, :] + 1
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        wt = w[n_t, :]
        hl.store(
            s,
            [rows, n_t.begin + hl.arange(16)],
            hl.full([16, 16], 100.0, dtype=s.dtype),
            extra_mask=live[:, None],
        )
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        out[n_t, :] = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
    return out


def war_split_chain(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Like :func:`war_pre_chain`, but the dot root first reads ``b`` of a
    fast producer: it reads ``b``, overwrites ``s``, then reads ``a``."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        a[t, :] = s[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] * 2
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        vb = hl.load(b, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, vb.permute(1, 0), out_dtype=torch.float32)
        hl.store(
            s,
            [rows, n_t.begin + hl.arange(16)],
            hl.full([16, 16], 100.0, dtype=s.dtype),
            extra_mask=live[:, None],
        )
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, va.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def branch_then_later_chain(
    x: torch.Tensor,
    w: torch.Tensor,
    flag: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """The dot root reads ``q`` (written 20 us late) inside a branch and
    again after it."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        q[t, :] = x[t, :] * 2
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        va = hl.load(a, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, va.permute(1, 0), out_dtype=torch.float32)
        if flag[0] > 0:
            vq = hl.load(q, [rows, hl.arange(k)], extra_mask=live[:, None])
            acc = hl.dot(wt, vq.permute(1, 0), acc=acc)
        vq2 = hl.load(q, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, vq2.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def alias_view_chain(
    x: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    a_v: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """The dot root reads ``b``, then the output of the producer of ``a``
    (20 us late) through ``a_v``, which the caller may pass as a view of
    ``a``."""
    tokens, k = x.size()
    n = w.size(0)
    out = torch.empty([n, 16], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        b[t, :] = x[t, :] * 2
    for n_t in hl.tile(n, block_size=16):
        rows = hl.arange(16)
        live = rows < tokens
        vb = hl.load(b, [rows, hl.arange(k)], extra_mask=live[:, None])
        wt = w[n_t, :]
        acc = hl.dot(wt, vb.permute(1, 0), out_dtype=torch.float32)
        va = hl.load(a_v, [rows, hl.arange(k)], extra_mask=live[:, None])
        acc = hl.dot(wt, va.permute(1, 0), acc=acc)
        out[n_t, :] = acc
    return out


def offset_alias_chain(
    x: torch.Tensor, a: torch.Tensor, big: torch.Tensor
) -> torch.Tensor:
    """Writes ``a`` (20 us late); the next root reads rows ``tokens`` and up of
    ``big``, of which the caller may pass ``a`` as the tail."""
    tokens, k = x.size()
    out = torch.empty([tokens, k], dtype=torch.float32, device=x.device)
    for t in hl.tile(tokens, block_size=1):
        hl.inline_triton(_GLOBALTIMER_SPIN_20US, args={}, output_like=None)
        a[t, :] = x[t, :] + 1
    for t in hl.tile(tokens, block_size=1):
        out[t, :] = big[t.index + tokens, :].to(torch.float32)
    return out


def _padded_rows(t: torch.Tensor) -> torch.Tensor:
    rows = torch.zeros(16, t.size(1), device=t.device, dtype=torch.float32)
    rows[: t.size(0)] = t.float()
    return rows


def _bad_launches(
    launch: Callable[[], torch.Tensor],
    poison: Callable[[], None],
    good: Callable[[torch.Tensor], bool],
    launches: int = 20,
) -> int:
    """Launches whose result shows an access that ran before its producer
    finished (``poison`` resets the producers' memory before each)."""
    bad = 0
    for _ in range(launches):
        poison()
        out = launch()
        torch.cuda.synchronize()
        bad += not (not torch.isnan(out).any() and good(out))
    return bad


def _without_names(
    dropped: str,
) -> Callable[..., tuple[int, ...]]:
    """``_insert_producer_waits`` that ignores ``dropped`` as a use of any
    producer's memory (a negative control)."""
    original = cross_loop_codegen._insert_producer_waits

    def insert(
        statements: list[ast.stmt],
        names: tuple[frozenset[str] | None, ...],
        make_wait: Any,
    ) -> tuple[int, ...]:
        return original(
            statements,
            tuple(None if n is None else (n - {dropped}) or None for n in names),
            make_wait,
        )

    return insert


_SINGLE_TRIP_DOT_CONFIG: dict[str, Any] = {
    "block_sizes": [],
    "cross_loop_pipeline": "static",
    "pid_type": "persistent_blocked",
    "num_warps": 4,
    "num_stages": 1,
    "num_sm_multiplier": 1,
    "indexing": "pointer",
}


def _initial_polls(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line, indent, counter) of every root-barrier poll outside its spin."""
    pattern = re.compile(
        r"^(\s*)tile_dependency_root_barrier_wait(?:_\d+)? = tl\.inline_asm_elementwise"
        r"\(asm='ld\.(?:acquire|relaxed)\.gpu\.global\.u32 \$0, \[\$1\];', "
        r"constraints='=r,l', "
        r"args=\[([^\]]+)\]"
    )
    polls = []
    spin: int | None = None
    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        if spin is not None and indent > spin:
            continue
        spin = indent if line.lstrip().startswith("while ") else None
        match = pattern.match(line)
        if match:
            polls.append((index, indent, match.group(2)))
    return polls


class TestCrossLoopCodegenHelpers(TestCase):
    def test_async_proxy_write_detector_counts_unknown_callees(self) -> None:
        # Lean publication drops a root's bulk drain only when nothing in its
        # body may write through the async proxy; a callee that is neither a
        # Triton builtin nor a known Helion helper may issue TMA stores.
        writes = cross_loop_codegen._body_writes_async_proxy

        def body(*lines: str) -> list[ast.stmt]:
            return [statement for line in lines for statement in ast.parse(line).body]

        lean = (
            "tl.store(y + offsets, v)",
            "tl.atomic_add(counter, 1, sem='release', scope='gpu')",
            "v = tl.load(x + offsets).to(tl.float32)",
            "w = x_desc.load([0, 0]).reshape([32, 4]).sum(1)",
            "e = libdevice.exp(v) + tl_math.abs(v)",
            "m = helion_triton_helpers.max_propagate_nan(v, 0)",
            "q = triton_helpers.div_floor_integer(a, 4)",
            "tl.extra.cuda.gdc_wait()",
            "n = min(a, b) + len(s)",
            (
                "t = tl.inline_asm_elementwise(asm='ld.acquire.gpu.global.u32 $0, "
                "[$1];', constraints='=r,l', args=[c], dtype=tl.uint32, "
                "is_pure=False, pack=1)"
            ),
        )
        for line in lean:
            with self.subTest(line=line):
                self.assertFalse(writes(body(line)))
        async_writes = (
            "y_desc.store([0, 0], v)",
            "y_desc.atomic_add([0, 0], v)",
            "y_desc.scatter(v, rows, 0)",
            "tl.store_tensor_descriptor(y_desc, [0, 0], v)",
            (
                "t = tl.inline_asm_elementwise(asm='cp.reduce.async.bulk.global"
                ".shared::cta.add.f32 [$1], [$2], 4;', constraints='=r,l,r', "
                "args=[a, b], dtype=tl.int32, is_pure=False, pack=1)"
            ),
            (
                "t = tl.inline_asm_elementwise(asm=template, constraints='=r', "
                "args=[], dtype=tl.int32, is_pure=False, pack=1)"
            ),
            # Callees that may store through TMA themselves.
            "user_triton_function(y, row, v)",
            "_source_module.helper(y, v)",
            "v.frobnicate()",
        )
        for line in async_writes:
            with self.subTest(line=line):
                self.assertTrue(writes(body(line)))
        # A compiler-outlined helper is judged by its own body.
        helpers = {
            "pure_helper": tuple(body("tl.store(y + offsets, v)")),
            "tma_helper": tuple(body("y_desc.store([0, 0], v)")),
            "nested_helper": tuple(body("tma_helper(y_desc, v)")),
            "recursive_helper": tuple(body("recursive_helper(v)")),
        }
        self.assertFalse(writes(body("pure_helper(y, v)"), helpers))
        self.assertTrue(writes(body("tma_helper(y_desc, v)"), helpers))
        self.assertTrue(writes(body("nested_helper(y_desc, v)"), helpers))
        self.assertFalse(writes(body("recursive_helper(v)"), helpers))
        self.assertTrue(writes(body("pure_helper(y, v)")))

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
            "for k in tl.range(0, tl.where(live, 2048, 0 + BLOCK), BLOCK, "
            "num_stages=4):\n    acc += tl.load(x + k, live, other=0.0)",
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

    def test_sunk_wait_dominates_every_use(self) -> None:
        def place(source: str) -> str:
            body = ast.parse(source).body
            wait = ast.parse("wait()").body
            if not _insert_before_first_use(body, frozenset({"z", "z_desc"}), wait):
                return "unused"
            return ast.unparse(body)

        # Independent work runs first; the wait enters the one branch that uses z.
        self.assertEqual(
            place(
                "n = tl.load(routes)\n"
                "if n > 0:\n    e = n + 1\n    a = tl.load(z + e)\n"
                "else:\n    b = n\n"
            ),
            "n = tl.load(routes)\n"
            "if n > 0:\n    e = n + 1\n    wait()\n    a = tl.load(z + e)\n"
            "else:\n    b = n",
        )
        # A use in the test, in both branches or after the if keeps it outside.
        for source in (
            "if tl.load(z) > 0:\n    a = 1\n",
            "if c:\n    a = tl.load(z)\nelse:\n    b = z_desc.load([0])\n",
            "if c:\n    a = tl.load(z)\nb = tl.load(z + 1)\n",
        ):
            with self.subTest(source=source):
                self.assertTrue(place(source).startswith("wait()\n"))
        # Loops are never entered.
        self.assertEqual(
            place("for i in tl.range(0, 4):\n    a = tl.load(z + i)\n"),
            "wait()\nfor i in tl.range(0, 4):\n    a = tl.load(z + i)",
        )
        # Every name counts, including a descriptor of the same storage.
        self.assertEqual(
            place("a = z_desc.load([0])\nb = tl.load(z)\n"),
            "wait()\na = z_desc.load([0])\nb = tl.load(z)",
        )
        self.assertEqual(place("a = tl.load(x)\n"), "unused")

    def test_each_producer_waits_before_its_own_first_use(self) -> None:
        def place(source: str, *names: frozenset[str] | None) -> str:
            body = ast.parse(source).body
            self.top = _insert_producer_waits(
                body,
                names,
                lambda producers: (
                    ast.parse(f"wait({', '.join(map(str, producers))})").body
                ),
            )
            return ast.unparse(ast.Module(body=body, type_ignores=[]))

        a, q, kv = frozenset({"a"}), frozenset({"q", "q_desc"}), frozenset({"kv"})
        # The read of another producer's rows runs before q's wait.
        self.assertEqual(
            place(
                "i = t + 1\nx = tl.load(a + i)\nj = i * 2\ny = tl.load(q + j)\n", a, q
            ),
            "i = t + 1\nwait(0)\nx = tl.load(a + i)\nj = i * 2\nwait(1)\n"
            "y = tl.load(q + j)",
        )
        self.assertEqual(self.top, (0, 1))
        # Outputs first used together, or with only latency-free work between
        # the uses, share one poll.
        for source in (
            "x = tl.load(a + q_desc.load([0]))\n",
            "x = a + 1\ni = tl.where(c, t, 0).to(tl.int32)\ny = tl.load(q + i)\n",
        ):
            with self.subTest(source=source):
                text = place(source, a, q)
                self.assertEqual(text.count("wait("), 1)
                self.assertIn("wait(0, 1)", text)
                self.assertEqual(self.top, (0, 1))
        # A producer used in both branches waits in each, before that branch's
        # first use (after the branch's own reads): every path passes a wait.
        self.assertEqual(
            place(
                "r = tl.load(kv + t)\n"
                "if c:\n    s = t + 1\n    y = tl.load(q + s)\n"
                "else:\n    n = tl.load(kv + s)\n    z = tl.load(q + n)\n",
                kv,
                q,
                q,
            ),
            "wait(0)\nr = tl.load(kv + t)\n"
            "if c:\n    s = t + 1\n    wait(1, 2)\n    y = tl.load(q + s)\n"
            "else:\n    n = tl.load(kv + s)\n    wait(1, 2)\n    z = tl.load(q + n)",
        )
        self.assertEqual(self.top, (0, 1, 2))
        # One using branch: the other branch skips the wait.
        self.assertEqual(
            place("x = tl.load(a + t)\nif c:\n    y = tl.load(q + x)\nz = x\n", a, q),
            "wait(0)\nx = tl.load(a + t)\nif c:\n    wait(1)\n    y = tl.load(q + x)\n"
            "z = x",
        )
        self.assertEqual(self.top, (0,))
        # A use in a branch and another after the `if`: the wait stays before
        # the `if`, so the read after it is covered on the path that skips
        # the branch.
        self.assertEqual(
            place(
                "x = tl.load(a + t)\nif c:\n    y = tl.load(q + x)\n"
                "z = tl.load(q + t)\n",
                a,
                q,
            ),
            "wait(0)\nx = tl.load(a + t)\nwait(1)\nif c:\n    y = tl.load(q + x)\n"
            "z = tl.load(q + t)",
        )
        self.assertEqual(self.top, (0, 1))
        # A wait right before the `if` absorbs a branch wait that follows only
        # latency-free work; a load in between keeps them apart.
        self.assertEqual(
            place("i = t + 1\nif tl.load(a + i) > 0:\n    y = tl.load(q + i)\n", a, q),
            "i = t + 1\nwait(0, 1)\nif tl.load(a + i) > 0:\n    y = tl.load(q + i)",
        )
        self.assertEqual(self.top, (0, 1))
        self.assertEqual(
            place(
                "if tl.load(a + t) > 0:\n    j = tl.load(r + t)\n    y = tl.load(q + j)\n",
                a,
                q,
            ),
            "wait(0)\nif tl.load(a + t) > 0:\n    j = tl.load(r + t)\n    wait(1)\n"
            "    y = tl.load(q + j)",
        )
        self.assertEqual(self.top, (0,))
        # Unknown outputs and unused producers wait at the top; loops are not
        # entered.
        self.assertEqual(
            place("for i in tl.range(0, 4):\n    y = tl.load(q + i)\n", None, a, q),
            "wait(0, 1, 2)\nfor i in tl.range(0, 4):\n    y = tl.load(q + i)",
        )
        self.assertEqual(self.top, (0, 1, 2))
        self.assertEqual(
            place("x = tl.load(r + t)\ny = tl.load(q + x)\n", None, q),
            "wait(0)\nx = tl.load(r + t)\nwait(1)\ny = tl.load(q + x)",
        )
        self.assertEqual(self.top, (0, 1))

    def test_multi_counter_poll_uses_an_acquire_fence_from_sm90(self) -> None:
        names = itertools.count()
        device_function = SimpleNamespace(
            new_var=lambda prefix, dce=False: f"{prefix}_{next(names)}",
            async_load_fence=list,
        )
        # fence.acquire.gpu assembles for sm_80 too, as an illegal instruction:
        # older targets poll with acquire loads.
        for capability, relaxed in (
            ((8, 0), False),
            ((8, 9), False),
            ((9, 0), True),
            ((10, 0), True),
            (None, False),
        ):
            with (
                self.subTest(capability=capability),
                mock.patch.object(
                    CompileEnvironment,
                    "current",
                    return_value=SimpleNamespace(ptx_capability=capability),
                ),
            ):
                text = "\n".join(
                    ast.unparse(statement)
                    for statement in cross_loop_codegen._wait_for_dependencies(
                        device_function=cast("DeviceFunction", device_function),
                        dependencies=(("c0", "t0"), ("c1", "t1")),
                        prefix="w",
                    )
                )
                self.assertEqual(text.count("ld.relaxed.gpu"), 4 if relaxed else 0)
                self.assertEqual(text.count("ld.acquire.gpu"), 0 if relaxed else 4)
                self.assertEqual("fence.acquire.gpu" in text, relaxed)

    def test_launcher_refuses_storage_sharing_the_compiler_did_not_see(
        self,
    ) -> None:
        from helion.runtime.triton import launcher

        x = torch.zeros(8, 4)
        y = torch.zeros(8, 4)
        args = (
            x,
            3,
            x[2:],
            y,
            SimpleNamespace(base=x),  # a host tensor descriptor of x
            torch.empty(0),
            torch.empty(0),
        )
        self.assertEqual(launcher._shared_storages(args), ((0, 2, 4),))
        self.assertFalse(launcher._distinct_storages(args))
        self.assertTrue(launcher._distinct_storages((x, y, torch.empty(0))))
        kernel = SimpleNamespace(arg_names=["x", "n", "x_tail", "y", "x_desc"])
        launcher._check_compiled_storage_sharing(kernel, ((0, 2, 4),), ((0, 2, 4),))
        for compiled in ((), ((0, 4),), ((0, 2), (4, 5))):
            with (
                self.subTest(compiled=compiled),
                self.assertRaisesRegex(
                    RuntimeError,
                    "Cross-loop launch aliasing: arguments 'x', 'x_tail', 'x_desc'",
                ),
            ):
                launcher._check_compiled_storage_sharing(kernel, ((0, 2, 4),), compiled)
        # The in-process launcher checks before it touches launch state.
        with self.assertRaises(helion.exc.CrossLoopAliasedLaunch):
            helion.runtime.default_launcher(
                kernel,
                (1,),
                x,
                x[1:],
                num_warps=4,
                num_stages=1,
                _persistent_state_specs=((x, 1, torch.int32, False),),
            )

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
        environment = SimpleNamespace(backend_name="triton")
        with mock.patch.object(CompileEnvironment, "current", return_value=environment):
            helper_name, arguments = device_function.register_triton_outlined_helper(
                "opaque_tile", body, noinline=True
            )
            helper = device_function.codegen_helper_functions()[0]

        self.assertEqual(helper_name, "opaque_tile")
        self.assertEqual(arguments, ())
        self.assertIsInstance(helper, ast.FunctionDef)
        assert isinstance(helper, ast.FunctionDef)
        self.assertEqual(_ast_fingerprint(helper.body), computation)
        self.assertEqual(
            ast.unparse(helper.decorator_list[0]), "triton.jit(noinline=True)"
        )

    def test_outlined_tile_body_captures_compiler_preamble_values(self) -> None:
        device_function = object.__new__(DeviceFunction)
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
        environment = SimpleNamespace(backend_name="triton")

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
        self.assertNotIn("fence.acq_rel.gpu", code)
        with self.assertRaises(helion.exc.PdlStateWithoutWait):
            code_and_output(pdl_unwaited_chain, (x,), **config)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_pdl_exit_trigger_orders_state_without_wait(self) -> None:
        x = torch.arange(4096, device=DEVICE, dtype=torch.float32).reshape(1, 4096)
        for pipeline in ("static", "dynamic"):
            config = {
                "block_sizes": [1, 16],
                "pid_type": "persistent_blocked",
                "cross_loop_pipeline": pipeline,
                "num_sm_multiplier": 1,
                "num_warps": 4,
            }
            with self.subTest(pipeline=pipeline):
                code, out = code_and_output(pdl_exit_trigger_chain, (x,), **config)

                torch.testing.assert_close(out, (x + 1) * 2)
                self.assertIn("launch_pdl=True", code)
                kernel = code[
                    code.index("def _helion_") : code.index("\ndef pdl_exit_trigger")
                ]
                statements = [
                    line.strip()
                    for line in kernel.splitlines()[1:]
                    if line.strip() and not line.strip().startswith("#")
                ]
                # Every warp is done with the state and its writes are released
                # before the CTA lets the next launch start.
                self.assertEqual(statements[-3], "tl.debug_barrier()")
                self.assertIn("fence.acq_rel.gpu", statements[-2])
                self.assertEqual(
                    statements[-1], "tl.extra.cuda.gdc_launch_dependents()"
                )
                self.assertEqual(code.count("gdc_launch_dependents"), 1)
                if pipeline == "static":
                    # The previous launch's epoch store is read gpu-scope strong.
                    epoch = next(
                        s for s in statements if s.startswith("tile_dependency_epoch =")
                    )
                    self.assertIn("ld.acquire.gpu.global.u32", epoch)
                # Back to back, each launch starts once the previous one triggered.
                compiled = pdl_exit_trigger_chain.bind((x,)).compile_config(
                    helion.Config(**config)
                )
                outs = [compiled(x) for _ in range(8)]
                for out in outs:
                    torch.testing.assert_close(out, (x + 1) * 2)

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
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_sunk_readiness_wait_runs_where_the_task_first_uses(self) -> None:
        def gated(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.zeros_like(x)
            for producer_m in hl.tile(x.size(0), block_size=1):
                for producer_n in hl.tile(x.size(1), block_size=32):
                    tmp[producer_m, producer_n] = x[producer_m, producer_n] + 1
            for consumer_m, consumer_n in hl.tile(x.size(), block_size=[1, 32]):
                if keep[consumer_m.begin] > 0:
                    out[consumer_m, consumer_n] = tmp[consumer_m, consumer_n] * 2
            return out

        x = torch.randn(8, 128, device=DEVICE)
        keep = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], device=DEVICE, dtype=torch.int32)
        expected = torch.where(keep[:, None] > 0, (x + 1) * 2, 0)
        kernel = helion.kernel(gated, static_shapes=True, autotune_effort="none")
        for pipeline in ("static", "dynamic"):
            with self.subTest(pipeline=pipeline):
                code, out = code_and_output(
                    kernel,
                    (x, keep),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=1,
                )

                torch.testing.assert_close(out, expected)
                root = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
                # Only a dynamic consumer waits inside its root, in the branch
                # that reads tmp, after the keep load.
                sunk = "tile_dependency_readiness_wait" in root
                self.assertEqual(sunk, pipeline == "dynamic")
                if sunk:
                    branch = root.index("if v_")
                    self.assertLess(root.index("tl.load(keep"), branch)
                    self.assertLess(
                        branch, root.index("tile_dependency_readiness_wait")
                    )
                    self.assertLess(
                        root.rindex("tile_dependency_readiness_wait"),
                        root.index("tl.load(tmp"),
                    )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_sunk_wait_precedes_every_read_it_certifies(self) -> None:
        # W writes w (20 us late, into NaN-poisoned storage); Q reads all of w;
        # A first computes from its own input, then reads all of q and all of
        # w.  A (publishing nothing) waits once -- on a counter both W and Q
        # arrive on, or on Q's root barrier that implies W's (W -> Q -> A) --
        # sunk past its own work but in front of its first read of q or w.
        def chain_join(
            x: torch.Tensor,
            y: torch.Tensor,
            w: torch.Tensor,
            q: torch.Tensor,
            delay: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty_like(y)
            for w_tile in hl.tile(x.size(0), block_size=1):
                spun = hl.inline_asm_elementwise(
                    _GLOBALTIMER_SPIN,
                    "=r,l",
                    [delay[0]],
                    dtype=torch.int32,
                    is_pure=False,
                    pack=1,
                )
                w[w_tile] = x[w_tile] + spun.to(torch.float32)
            for q_tile in hl.tile(q.size(0), block_size=1):
                q[q_tile] = torch.sum(w[:], dim=0) + q_tile.begin
            for a_tile in hl.tile(y.size(0), block_size=1):
                own = y[a_tile] * 2
                out[a_tile] = own + torch.sum(w[:], dim=0) + torch.sum(q[:], dim=0)
            return out

        x = torch.randn(16, device=DEVICE)
        y = torch.randn(8, device=DEVICE)
        w = torch.empty_like(x)
        q = torch.empty(4, device=DEVICE)
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        kernel = helion.kernel(chain_join, static_shapes=True, autotune_effort="none")
        for launch in range(3):
            w.fill_(float("nan"))
            q.fill_(float("nan"))
            code, out = code_and_output(
                kernel,
                (x + launch, y, w, q, delay),
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.cuda.synchronize()
            total = (x + launch).sum()
            expected = (
                y * 2 + total + (total * 4 + torch.arange(4, device=DEVICE).sum())
            )
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
        root = ast.unparse(_generated_function(code, "tile_dependency_root_2"))
        kinds = [
            kind
            for kind in (
                "tile_dependency_readiness_wait",
                "tile_dependency_root_barrier_wait",
            )
            if kind in root
        ]
        self.assertEqual(len(kinds), 1, root)
        wait = root.index(kinds[0])
        self.assertLess(root.index("tl.load(y"), wait)
        self.assertLess(wait, root.index("tl.load(w"))
        self.assertLess(wait, root.index("tl.load(q"))
        # One poll: W's dependency is carried by the same wait as Q's.
        self.assertEqual(root.count(f"{kinds[0]}_sync"), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_sunk_readiness_wait_in_a_root_barrier_publisher(self) -> None:
        # The gated consumer also feeds a join whose every task reads all of
        # `out` (a root barrier).  Barrier consumers order other roots by
        # barrier edges only, so the consumer keeps its sunk wait (after the
        # keep load, in the branch that reads tmp) although it publishes.  The
        # producer writes 20 us late into a NaN-poisoned tmp, so a missing or
        # skipped wait on a kept row shows up.
        def gated_join(
            x: torch.Tensor, keep: torch.Tensor, tmp: torch.Tensor, delay: torch.Tensor
        ) -> torch.Tensor:
            out = torch.zeros_like(x)
            total = torch.empty([x.size(0)], device=x.device, dtype=x.dtype)
            for producer_m in hl.tile(x.size(0), block_size=1):
                for producer_n in hl.tile(x.size(1), block_size=32):
                    spun = hl.inline_asm_elementwise(
                        _GLOBALTIMER_SPIN,
                        "=r,l",
                        [delay[0]],
                        dtype=torch.int32,
                        is_pure=False,
                        pack=1,
                    )
                    tmp[producer_m, producer_n] = (
                        x[producer_m, producer_n] + 1 + spun.to(torch.float32)
                    )
            for consumer_m, consumer_n in hl.tile(x.size(), block_size=[1, 32]):
                if keep[consumer_m.begin] > 0:
                    out[consumer_m, consumer_n] = tmp[consumer_m, consumer_n] * 2
            for join_m in hl.tile(x.size(0), block_size=1):
                total[join_m] = (
                    torch.sum(torch.sum(out[:, :], dim=-1), dim=0) + 0 * join_m.begin
                )
            return total

        x = torch.randn(8, 128, device=DEVICE)
        keep = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], device=DEVICE, dtype=torch.int32)
        tmp = torch.empty_like(x)
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        expected = torch.where(keep[:, None] > 0, (x + 1) * 2, 0).sum().expand(8)
        kernel = helion.kernel(gated_join, static_shapes=True, autotune_effort="none")
        for launch in range(3):
            tmp.fill_(float("nan"))
            code, total = code_and_output(
                kernel,
                (x + launch, keep, tmp, delay),
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                total,
                torch.where(keep[:, None] > 0, (x + launch + 1) * 2, 0).sum().expand(8),
                rtol=1e-5,
                atol=1e-3,
            )
        del expected
        root = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
        branch = root.index("if v_")
        self.assertLess(root.index("tl.load(keep"), branch)
        self.assertLess(branch, root.index("tile_dependency_readiness_wait"))
        self.assertLess(
            root.rindex("tile_dependency_readiness_wait"), root.index("tl.load(tmp")
        )
        # The consumer publishes the join's root barrier and the join waits on it.
        self.assertIn("tile_dependency_root_barrier_wait", code)
        self.assertGreaterEqual(code.count("sem='release'"), 2)

    @skipIfNotCUDA()
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_sunk_readiness_wait_precedes_device_descriptor_reads(self) -> None:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def read_twice(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.empty_like(x)
            for producer_m in hl.tile(x.size(0), block_size=32):
                for producer_n in hl.tile(x.size(1), block_size=32):
                    tmp[producer_m, producer_n] = x[producer_m, producer_n] + 1
            for consumer_m, consumer_n in hl.tile(x.size(), block_size=[32, 32]):
                out[consumer_m, consumer_n] = (
                    tmp[consumer_m, consumer_n] * 2 + tmp[consumer_m, consumer_n]
                )
            return out

        x = torch.randn(64, 128, device=DEVICE)
        code, out = code_and_output(
            read_twice,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=1,
            range_num_stages=[0, 0, 0],
            # The first read of tmp goes through a descriptor built in the
            # kernel preamble, which is not a kernel argument.
            indexing=["pointer", "pointer", "tensor_descriptor", "pointer", "pointer"],
            host_tensor_descriptors=False,
        )

        torch.testing.assert_close(out, (x + 1) * 3)
        root = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
        self.assertLess(
            root.rindex("tile_dependency_readiness_wait"), root.index("tmp_desc.load")
        )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="FP4 conversion instructions require Blackwell"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_sunk_readiness_wait_precedes_unrecorded_reads(self) -> None:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def scaled_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            q = torch.empty_like(packed)
            s = torch.empty_like(scale)
            out = torch.empty(
                [packed.size(0), 16], dtype=torch.float32, device=packed.device
            )
            for producer in hl.tile(packed.size(0), block_size=32):
                busy = scale[producer]
                for _ in hl.static_range(64):
                    busy = busy * 0.5 + 0.25
                q[producer, :] = packed[producer, :]
                s[producer] = busy * 0.0 + scale[producer]
            # Two consumer tasks per producer task: a readiness wait, not a
            # final-arrival continuation.
            for consumer in hl.tile(packed.size(0), block_size=16):
                # The dependency analysis does not record this read of q, so
                # the readiness edge covers only s.
                lanes = hl.load_float4_e2m1fn_x16_to_float16(q, consumer.index)
                factor = s[consumer]
                for lane in hl.static_range(16):
                    out[consumer, lane] = lanes[lane].to(torch.float32) * factor
            return out

        packed = torch.randint(0, 256, (4096, 8), dtype=torch.uint8, device=DEVICE)
        scale = torch.rand(4096, device=DEVICE) + 0.5
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=DEVICE,
        )
        packed_i64 = packed.long()
        lanes = torch.stack((lut[packed_i64 & 0xF], lut[packed_i64 >> 4]), dim=-1)
        expected = lanes.reshape(4096, 16) * scale[:, None]
        code, out = code_and_output(
            scaled_fp4,
            (packed, scale),
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=4,
        )

        torch.testing.assert_close(out, expected)
        root = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
        # Producer storage counts as used even where the read is unrecorded:
        # the wait still sinks past the index math, but not past the q read.
        self.assertLess(
            root.index("indices_"), root.index("tile_dependency_readiness_wait")
        )
        self.assertLess(
            root.rindex("tile_dependency_readiness_wait"), root.index("tl.load(q")
        )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_publishing_root_keeps_readiness_wait_at_task_entry(self) -> None:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def chain(x: torch.Tensor) -> torch.Tensor:
            first = torch.empty_like(x)
            second = torch.empty_like(x)
            out = torch.empty_like(x)
            for producer_m in hl.tile(x.size(0), block_size=1):
                for producer_n in hl.tile(x.size(1), block_size=32):
                    first[producer_m, producer_n] = x[producer_m, producer_n] + 1
            for middle_m, middle_n in hl.tile(x.size(), block_size=[1, 32]):
                second[middle_m, middle_n] = first[middle_m, middle_n] * 2
            for consumer_m, consumer_n in hl.tile(x.size(), block_size=[1, 32]):
                out[consumer_m, consumer_n] = second[consumer_m, consumer_n] + 3
            return out

        x = torch.randn(8, 128, device=DEVICE)
        code, out = code_and_output(
            chain,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
            num_sm_multiplier=1,
            num_warps=1,
        )

        torch.testing.assert_close(out, (x + 1) * 2 + 3)
        # The middle root publishes to the last one, which relies on the middle
        # root having waited for the first: its wait stays at task entry.
        task = _generated_function(code, "tile_dependency_root_1_scheduled_task")
        root = _generated_function(code, "tile_dependency_root_1")
        self.assertIn("tile_dependency_readiness_wait", ast.unparse(task))
        self.assertNotIn("tile_dependency_readiness_wait", ast.unparse(root))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_single_trip_source_publisher_wrapper_inlines(self) -> None:
        def chain_impl(x: torch.Tensor) -> torch.Tensor:
            first = torch.empty_like(x)
            second = torch.empty_like(x)
            out = torch.empty_like(x)
            for producer_m, producer_n in hl.tile(x.size(), block_size=[1, 32]):
                first[producer_m, producer_n] = x[producer_m, producer_n] + 1
            for middle_m, middle_n in hl.tile(x.size(), block_size=[1, 32]):
                second[middle_m, middle_n] = first[middle_m, middle_n] * 2
            for consumer_m, consumer_n in hl.tile(x.size(), block_size=[1, 32]):
                out[consumer_m, consumer_n] = second[consumer_m, consumer_n] + 3
            return out

        def wrapper_decorator(code: str, root: int) -> str:
            wrapper = _generated_function(
                code, f"tile_dependency_root_{root}_scheduled_task"
            )
            self.assertIn("sem='release'", ast.unparse(wrapper))
            return ast.unparse(wrapper.decorator_list[0])

        # The first two roots publish readiness counters (no continuation:
        # fan-in 1 would run the next root inside its producer).  With 8 x 4
        # tasks they run in one wave: the source root's wrapper inlines unless
        # the setting is off; the middle root waits on the first, so it keeps
        # the call boundary.  With 2048 x 4 tasks they loop: both stay noinline.
        noinline = "triton.jit(noinline=True)"
        for rows, inline, expected in (
            (8, True, ("triton.jit", noinline)),
            (8, False, (noinline, noinline)),
            (2048, True, (noinline, noinline)),
        ):
            x = torch.randn(rows, 128, device=DEVICE)
            chain = helion.kernel(
                chain_impl,
                static_shapes=True,
                autotune_effort="none",
                cross_loop_inline_single_trip_publishers=inline,
            )
            with mock.patch.object(
                cross_loop_scheduler,
                "choose_final_arrival_continuations",
                return_value=(),
            ):
                code, out = code_and_output(
                    chain,
                    (x,),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="dynamic",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
            torch.testing.assert_close(out, (x + 1) * 2 + 3)
            self.assertEqual(
                (wrapper_decorator(code, 0), wrapper_decorator(code, 1)),
                expected,
                (rows, inline),
            )

    @skipIfNotCUDA()
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_middle_root_tma_store_drains_every_release(self) -> None:
        def three_stage_impl(x: torch.Tensor) -> torch.Tensor:
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

        def release_drains(lean_publication: bool) -> dict[str, bool]:
            three_stage = helion.kernel(
                three_stage_impl,
                static_shapes=True,
                autotune_effort="none",
                cross_loop_lean_publication=lean_publication,
            )
            # Continuations with fan-in 1 publish without atomics; keep every
            # edge on the release path this test inspects.
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
            # Walking back from a release must reach a drain before any store;
            # a release that reaches its function's start publishes undrained.
            drained: dict[str, bool] = {}
            for i in releases:
                function = next(
                    line.split("(")[0].split()[-1]
                    for line in reversed(lines[:i])
                    if line.lstrip().startswith("def ")
                )
                has_drain = False
                for line in reversed(lines[:i]):
                    if "cp.async.bulk.wait_group 0;" in line:
                        has_drain = True
                        break
                    self.assertNotIn(
                        ".store(",
                        line,
                        msg=f"store between drain and release: {lines[i].strip()}",
                    )
                    if line.lstrip().startswith("def "):
                        has_drain = False
                        break
                drained[function] = drained.get(function, True) and has_drain
            return drained

        # Only the root with the TMA store drains before publishing; the root
        # that stores through pointers publishes without the drain.
        drained = release_drains(lean_publication=True)
        self.assertEqual(
            drained,
            {
                "tile_dependency_root_0_scheduled_task": False,
                "tile_dependency_root_1_scheduled_task": True,
            },
        )
        # Without lean publication every release drains.
        drained = release_drains(lean_publication=False)
        self.assertEqual(
            set(drained),
            {
                "tile_dependency_root_0_scheduled_task",
                "tile_dependency_root_1_scheduled_task",
            },
        )
        self.assertTrue(all(drained.values()), drained)

        # Without a TMA access there is nothing to drain or fence.
        code, out = code_and_output(
            helion.kernel(three_stage_impl, static_shapes=True, autotune_effort="none"),
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
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_lean_publication_drains_after_a_callee_tma_store(self) -> None:
        # P's rows of y land only through a TMA store inside a user Triton
        # function: its publication must complete that bulk store first even
        # with lean publication, since its body does not spell the store.
        x = torch.randn(64, 128, device=DEVICE)
        y = torch.empty(65, 128, device=DEVICE)
        pipelines: tuple[Literal["dynamic", "static"], ...] = ("dynamic", "static")
        for pipeline, lean in itertools.product(pipelines, (True, False)):
            with self.subTest(pipeline=pipeline, lean=lean):
                config = helion.Config(
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=4,
                    # x through TMA, so the kernel uses the async proxy.
                    indexing=["tensor_descriptor", *["pointer"] * 5],
                )
                kernel = helion.kernel(
                    _callee_tma_store_impl,
                    config=config,
                    static_shapes=True,
                    cross_loop_lean_publication=lean,
                )
                code = kernel.bind((x, y)).to_triton_code(config)
                self.assertIn("_tma_store_rows(", code)
                self.assertIn("cp.async.bulk.wait_group 0;", code)
                for _ in range(4):
                    y.fill_(float("nan"))
                    out = kernel(x, y)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(out, x + 1.0))

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_lean_publication_keeps_the_cta_barrier_before_its_release(self) -> None:
        # Lean publication drops only the bulk drain: thread 0's release must
        # still follow a CTA barrier, or it publishes while warps 1-3 have not
        # stored yet (here they spin 20 us first) and C reads NaN.
        x = torch.randn(1024, device=DEVICE)
        y = torch.empty_like(x)
        delay = torch.tensor([20_000], device=DEVICE, dtype=torch.int64)
        for pipeline in ("dynamic", "static"):
            with self.subTest(pipeline=pipeline):
                config = helion.Config(
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=4,
                )
                kernel = helion.kernel(
                    _warp_skewed_publisher_impl, config=config, static_shapes=True
                )
                # Runtime first: without the barrier the race shows on every
                # launch (C reads warps 1-3's NaN).
                bad = 0
                for _ in range(20):
                    y.fill_(float("nan"))
                    out = kernel(x, y, delay)
                    torch.cuda.synchronize()
                    bad += not torch.equal(out, x + 1.0)
                self.assertEqual(bad, 0)
                # Lean: nothing to drain.  Walking back from each release
                # meets the CTA barrier before any store.
                code = kernel.bind((x, y, delay)).to_triton_code(config)
                self.assertNotIn("cp.async.bulk.wait_group", code)
                lines = code.splitlines()
                releases = [
                    i
                    for i, line in enumerate(lines)
                    if "sem='release'" in line or "sem='acq_rel'" in line
                ]
                self.assertTrue(releases)
                for i in releases:
                    for line in reversed(lines[:i]):
                        if "tl.debug_barrier()" in line:
                            break
                        self.assertNotIn("tl.store(", line, lines[i])
                        self.assertFalse(line.lstrip().startswith("def "), lines[i])

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
        wait = next(
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Name)
            and node.id.startswith("tile_dependency_root_barrier_wait")
        )
        self.assertLess(dispatch.lineno, publication.lineno)
        self.assertLess(publication.lineno, wait.lineno)

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
        # The last root rebases its PID first and tiles the bound by 2.
        prologue = ast.unparse(kernel).split("for virtual_pid")[0]
        bounded = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.For)
            and ast.unparse(node.target) == "virtual_pid"
            and "tile_dependency_live_tasks" in ast.unparse(node.iter)
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
        # An edge-free root before a synchronized one keeps its own wave, and
        # each root's lanes start on the worker after the previous root's
        # last task (packed placement): z on workers 6-10, the merge on 11-12.
        code, (out, other) = code_and_output(split_independent_merge, (x, z), **config)
        torch.testing.assert_close(out, torch.sum(x * 2, dim=1))
        torch.testing.assert_close(other, z - 1)
        self.assertIn(
            f"(tl.program_id(0) + {workers - 6}) % {workers} - 0 + {workers}, "
            f"{workers + 5},",
            code,
        )
        self.assertIn(
            f"(tl.program_id(0) + {workers - 11}) % {workers} - 0 + "
            f"{2 * workers}, {2 * workers + 2},",
            code,
        )

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
    def test_fanout_producer_keeps_keyed_readiness_of_other_producer(self) -> None:
        # Regression: merging the fan-out producer (every consumer key reads all of
        # it) into the keyed producer's readiness event made the event unlowerable,
        # so the consumer fell back to whole-root barriers on BOTH producers.
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        y = torch.arange(32 * 64, device=DEVICE, dtype=torch.float32).reshape(32, 64)
        for pipeline in ("static", "dynamic"):
            for launch in range(2):
                code, out = code_and_output(
                    fanout_and_keyed_join,
                    (x + launch, y + launch),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=1,
                )
                small_sum = (x + launch + 1).sum()
                torch.testing.assert_close(out, (y + launch) * 2 + 2 * small_sum)
            # The consumer stays keyed on M: a keyed wait, or (once M's keyed
            # wait certifies S transitively and no S -> C barrier remains) a
            # final-arrival continuation of M's key.
            self.assertTrue(
                "tile_dependency_readiness_wait" in code
                or "tile_dependency_continuation_previous" in code,
                pipeline,
            )
            barrier_targets = {
                line.split("args=[", 1)[1].split("]", 1)[0]
                for line in code.splitlines()
                if "tile_dependency_root_barrier_wait" in line and "ld.acquire" in line
            }
            # Only the fan-out producer keeps a root barrier.
            self.assertEqual(len(barrier_targets), 1, barrier_targets)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_dynamic_tickets_claim_long_task_sinks_first(self) -> None:
        # Sink LPT order (default on): C's long tasks are claimed before B's
        # short ones; with the setting off the stream keeps source order.
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        y = torch.arange(16 * 512, device=DEVICE, dtype=torch.float32).reshape(16, 512)
        expected_bounds = {
            short_then_long_sinks: [(0, 8), (8, 24), (24, 32)],
            short_then_long_sinks_source_order: [(0, 8), (8, 16), (16, 32)],
        }
        for kernel, bounds_expected in expected_bounds.items():
            for launch in range(3):
                code, (b, c) = code_and_output(
                    kernel,
                    (x + launch, y + launch),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="dynamic",
                    num_sm_multiplier=1,
                    num_warps=1,
                )
                a_sum = (x + launch + 1).sum()
                torch.testing.assert_close(b, x + launch + a_sum)
                torch.testing.assert_close(c, (y + launch) * 2 + a_sum)
            ranges = [
                line.split("if ", 1)[1]
                for line in code.splitlines()
                if "tile_dependency_dispatch_ticket_1 >= " in line
                and " and tile_dependency_dispatch_ticket_1 < " in line
            ]
            bounds = [
                tuple(int(part.split()[-1].rstrip(":")) for part in item.split(" and "))
                for item in ranges
            ]
            # A has 8 tasks, B 8, C 16: the 16-ticket packet is C's.
            self.assertEqual(bounds, bounds_expected, ranges)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_source_root_claims_grouped_tasks_under_inflight_budget(self) -> None:
        # S's 64 tasks are claimed 4 per ticket (16 packets) and run in order;
        # each ticket publishes its 4 tasks at once, so C waits for all 64.  S
        # writes 20 us late into NaN-poisoned partials: a short count would read
        # NaN.  With the default group cap (2) the same budget claims 2 per
        # ticket (32 packets).
        w = torch.randn(64, 128, 128, device=DEVICE, dtype=torch.float32)
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        parts = torch.empty(64, device=DEVICE, dtype=torch.float32)
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        expected_packets = {grouped_source: 16, capped_source: 32, ungrouped_source: 64}
        for kernel, packets in expected_packets.items():
            for launch in range(4):
                parts.fill_(float("nan"))
                code, out = code_and_output(
                    kernel,
                    (w + launch, x, parts, delay),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="dynamic",
                    num_sm_multiplier=1,
                    num_warps=4,
                )
                torch.cuda.synchronize()
                reference = (w + launch).sum(dim=(1, 2))
                torch.testing.assert_close(parts, reference, rtol=1e-4, atol=1e-2)
                torch.testing.assert_close(out, x + parts.sum(), rtol=1e-5, atol=1e-3)
            ranges = [
                line.split("if ", 1)[1]
                for line in code.splitlines()
                if "tile_dependency_dispatch_ticket_1 >= " in line
                and " and tile_dependency_dispatch_ticket_1 < " in line
            ]
            bounds = [
                tuple(int(part.split()[-1].rstrip(":")) for part in item.split(" and "))
                for item in ranges
            ]
            self.assertEqual(bounds[0], (0, packets), ranges)
            self.assertEqual("tile_dependency_group_task" in code, packets != 64)
            if packets != 64:
                # A plain group loop and one publication of all of a ticket's
                # tasks.
                group = 64 // packets
                self.assertIn(f"in tl.range(0, {group}, 1):", code)
                self.assertNotIn(f"tl.range(0, {group}, 1, num_stages=1)", code)
                self.assertIn(f", {group}, sem='release', scope='gpu')", code)
            # C's root barrier waits for all 64 S tasks either way.
            self.assertIn("tl.cast(64, tl.uint32)", code)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_transitive_keyed_wait_replaces_root_barrier(self) -> None:
        # C waits only on M's keyed readiness: M waited on all of S, so S keeps a
        # root barrier for M alone; C's wait precedes its read of S; S writes
        # 20 us late into a NaN-poisoned buffer, so a missing wait shows up.
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        y = torch.arange(32 * 64, device=DEVICE, dtype=torch.float32).reshape(32, 64)
        small = torch.empty_like(x)
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        no_continuations = mock.patch.object(
            cross_loop_scheduler,
            "choose_final_arrival_continuations",
            lambda *args, **kwargs: (),
        )
        for pipeline, continuations in itertools.product(
            ("static", "dynamic"), (True, False)
        ):
            kernel = helion.kernel(
                _transitive_chain_impl, static_shapes=True, autotune_effort="none"
            )
            for launch in range(3):
                small.fill_(float("nan"))
                with contextlib.nullcontext() if continuations else no_continuations:
                    code, out = code_and_output(
                        kernel,
                        (x + launch, y + launch, small, delay, True),
                        pid_type="persistent_blocked",
                        cross_loop_pipeline=pipeline,
                        num_sm_multiplier=1,
                        num_warps=1,
                    )
                small_sum = (x + launch + 1).sum()
                torch.testing.assert_close(out, (y + launch) * 2 + 2 * small_sum)
            spins = [
                line
                for line in code.splitlines()
                if line.lstrip().startswith("while tile_dependency_root_barrier_wait")
            ]
            consumer = ast.unparse(_generated_function(code, "tile_dependency_root_2"))
            if pipeline == "static":
                # Transitive covers are formed for dynamic pipelines only: C
                # keeps its own S barrier (see
                # test_static_pipelines_form_no_transitive_covers).
                self.assertEqual(len(spins), 2, (pipeline, spins))
                continue
            self.assertEqual(len(spins), 1, (pipeline, spins))
            self.assertNotIn("tile_dependency_root_barrier_wait", consumer)
            if not continuations:
                # The task's wait on M is emitted before C's first read of S,
                # wherever the task places it (task entry, or sunk to first use
                # in a dynamic root, after the read of y).
                scheduled = ast.unparse(
                    _generated_function(code, "tile_dependency_root_2_scheduled_task")
                )
                self.assertIn("tile_dependency_readiness_wait", scheduled + consumer)
                if "tile_dependency_readiness_wait" in consumer:
                    self.assertLess(
                        consumer.index("tile_dependency_readiness_wait"),
                        consumer.index("tl.load(small"),
                        pipeline,
                    )
                    self.assertLess(
                        consumer.index("tl.load(y"),
                        consumer.index("tile_dependency_readiness_wait"),
                        pipeline,
                    )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_transitive_cover_keeps_direct_wait_when_it_would_move(self) -> None:
        # C reads all of S before M's rows.  A dynamic C sinks its wait on M to
        # its first use of M's data; certifying S through that wait would move
        # it above the S read (serializing the work in between), so C keeps its
        # direct S wait there.  Static waits sit at task entry: the cover stays.
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        y = torch.arange(32 * 64, device=DEVICE, dtype=torch.float32).reshape(32, 64)
        small = torch.empty_like(x)
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        no_continuations = mock.patch.object(
            cross_loop_scheduler,
            "choose_final_arrival_continuations",
            lambda *args, **kwargs: (),
        )
        for pipeline in ("static", "dynamic"):
            kernel = helion.kernel(
                _transitive_chain_impl, static_shapes=True, autotune_effort="none"
            )
            for launch in range(2):
                small.fill_(float("nan"))
                with no_continuations:
                    code, out = code_and_output(
                        kernel,
                        (x + launch, y + launch, small, delay, True, True),
                        pid_type="persistent_blocked",
                        cross_loop_pipeline=pipeline,
                        num_sm_multiplier=1,
                        num_warps=1,
                    )
                small_sum = (x + launch + 1).sum()
                torch.testing.assert_close(out, (y + launch) * 2 + 2 * small_sum)
            spins = [
                line
                for line in code.splitlines()
                if line.lstrip().startswith("while tile_dependency_root_barrier_wait")
            ]
            consumer = ast.unparse(_generated_function(code, "tile_dependency_root_2"))
            # Dynamic: S keeps a barrier for C (emitted at C's dispatch branch)
            # and C's sunk wait on M stays after its read of S.  Static: no
            # transitive cover is formed at all, so C keeps its S barrier too.
            self.assertEqual(len(spins), 2, spins)
            if pipeline == "dynamic":
                self.assertLess(
                    consumer.index("tl.load(small"),
                    consumer.index("tile_dependency_readiness_wait"),
                )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_transitive_wait_race_negative_control(self) -> None:
        # M does not read S, so C needs its own S synchronization.  The sound
        # scheduler keeps it (correct output); an unsound cover that skips the
        # causal proof drops it and C reads the poisoned buffer: the race test
        # must detect that.
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        y = torch.arange(32 * 64, device=DEVICE, dtype=torch.float32).reshape(32, 64)
        small = torch.empty_like(x)
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        expected = y * 2 + (x + 1).sum()

        def unsound(
            readiness_graph, obligations_by_root_pair, counters, charge, *_args
        ):
            covered = cross_loop_scheduler._covered_obligations(counters)
            return tuple(
                dataclasses.replace(
                    plan,
                    consumers=tuple(
                        dataclasses.replace(
                            consumer,
                            covered_obligations=consumer.covered_obligations.union(
                                *(
                                    obligations - covered
                                    for (
                                        _p,
                                        root,
                                    ), obligations in obligations_by_root_pair
                                    if root == consumer.consumer_root
                                )
                            ),
                        )
                        for consumer in plan.consumers
                    ),
                )
                for plan in counters
            )

        for pipeline in ("static", "dynamic"):
            small.fill_(float("nan"))
            sound = helion.kernel(
                _transitive_chain_impl, static_shapes=True, autotune_effort="none"
            )
            _code, out = code_and_output(
                sound,
                (x, y, small, delay, False),
                pid_type="persistent_blocked",
                cross_loop_pipeline=pipeline,
                num_sm_multiplier=1,
                num_warps=1,
            )
            torch.testing.assert_close(out, expected)
            broken = helion.kernel(
                _transitive_chain_impl, static_shapes=True, autotune_effort="none"
            )
            with mock.patch.object(
                cross_loop_scheduler, "_transitively_covered_counters", unsound
            ):
                small.fill_(float("nan"))
                code, out = code_and_output(
                    broken,
                    (x, y, small, delay, False),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=1,
                )
            if pipeline == "static":
                # Static pipelines form no transitive covers, so even the
                # unsound cover function is never consulted.
                torch.testing.assert_close(out, expected)
                continue
            consumer = ast.unparse(_generated_function(code, "tile_dependency_root_2"))
            self.assertNotIn("tile_dependency_root_barrier_wait", consumer)
            self.assertTrue(torch.isnan(out).any(), pipeline)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_sunk_barrier_wait_precedes_write_after_read_store(self) -> None:
        # A dynamic root that publishes nothing sinks its root-barrier wait to
        # the first use of what the wait orders.  That includes a store into a
        # tensor its producer READS (write-after-read), also through a view of
        # the tensor's storage passed as another argument.  P reads x 20 us
        # late; C's store of -1 must not reach x before P's read.
        n = 1024
        x0 = (torch.arange(n, device=DEVICE, dtype=torch.float32) % 13) + 0.5
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        expected = torch.full((n,), float((x0 + 1).sum()), device=DEVICE)
        for alias in (False, True):
            kernel = helion.kernel(
                _war_after_producer_read_impl,
                static_shapes=True,
                autotune_effort="none",
            )
            x = x0.clone()
            y = torch.empty(n, device=DEVICE)
            for _launch in range(6):
                x.copy_(x0)
                y.fill_(float("nan"))
                code, out = code_and_output(
                    kernel,
                    (x, x.view(n) if alias else x, y, delay),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline="dynamic",
                    num_sm_multiplier=1,
                    num_warps=4,
                    num_stages=1,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out, expected), alias)
                self.assertTrue(bool((x == -1).all()), alias)
            consumer = ast.unparse(_generated_function(code, "tile_dependency_root_1"))
            store = consumer.index("tl.store(x_store")
            self.assertIn(
                "tile_dependency_root_barrier_wait", consumer[:store], msg=str(alias)
            )

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_pipelines_form_no_transitive_covers(self) -> None:
        # Two shapes where a transitive cover of S through M's counter was
        # unsound in static pipelines (S writes `small` 20 us late into
        # NaN-poisoned memory): M's barrier wait on S sunk into a branch of a
        # kernel-scope root that publishes keyed rows (flags = 0 skips it), and
        # guard-skipped M tasks publishing their arrivals without K's wait
        # (live < rows).  Static pipelines now keep C's own S wait; dynamic
        # ones (publishers wait at task entry) stay exact with covers.
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        s_sum = (x + 1).sum()
        delay = torch.full((1,), 20000, device=DEVICE, dtype=torch.int64)
        for pipeline in ("static", "dynamic"):
            rows, width = 32, 64
            y = (
                (torch.arange(rows * width, device=DEVICE) % 97)
                .float()
                .reshape(rows, width)
            )
            w = torch.eye(width, device=DEVICE, dtype=torch.float32)
            small = torch.empty(8, device=DEVICE)
            keyed = torch.zeros(rows, width, device=DEVICE)
            flags = torch.zeros(4, device=DEVICE, dtype=torch.int32)
            kernel = helion.kernel(
                _dot_branch_transitive_impl, static_shapes=True, autotune_effort="none"
            )
            for on in (0, 1, 0):
                small.fill_(float("nan"))
                keyed.zero_()
                flags.fill_(on)
                _code, out = code_and_output(
                    kernel,
                    (x, y, small, flags, delay, keyed, w),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=4,
                    num_stages=1,
                )
                torch.cuda.synchronize()
                self.assertFalse(bool(torch.isnan(out).any()), (pipeline, on))
                self.assertTrue(
                    torch.allclose(
                        out,
                        y @ w + (s_sum if on else 0.0) + s_sum,
                        rtol=1e-6,
                        atol=1e-3,
                    ),
                    (pipeline, on),
                )
            rows, width = 32, 512
            y = (
                (torch.arange(rows * width, device=DEVICE) % 97)
                .float()
                .reshape(rows, width)
            )
            w = torch.eye(width, device=DEVICE, dtype=torch.float32)
            mid = torch.zeros(rows, width, device=DEVICE)
            kernel = helion.kernel(
                _guarded_transitive_chain_impl,
                static_shapes=True,
                autotune_effort="none",
            )
            for live in (0, 7, 32):
                flags = (torch.arange(rows, device=DEVICE) < live).to(torch.int32)
                small.fill_(float("nan"))
                mid.fill_(3.0)
                _code, out = code_and_output(
                    kernel,
                    (x, y, small, flags, delay, mid, w),
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                    num_sm_multiplier=1,
                    num_warps=4,
                    num_stages=1,
                )
                torch.cuda.synchronize()
                ref_mid = torch.full((rows, width), 3.0, device=DEVICE)
                ref_mid[:live] = 2 * (y[:live] + s_sum)
                self.assertFalse(bool(torch.isnan(out).any()), (pipeline, live))
                self.assertTrue(
                    torch.allclose(out, ref_mid + s_sum, rtol=1e-6, atol=1e-3),
                    (pipeline, live),
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
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_static_root_waits_for_each_producer_at_its_first_use(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()
        a = torch.empty_like(x)
        q = torch.empty_like(x)
        flags = {
            v: torch.full([1], v, device=DEVICE, dtype=torch.int32) for v in (0, 1)
        }
        config = helion.Config(**_SINGLE_TRIP_DOT_CONFIG)

        def build() -> helion.Kernel:
            return helion.kernel(late_producer_chain, config=config, static_shapes=True)

        lines = build().bind((x, w, flags[1], a, q)).to_triton_code(config).splitlines()
        polls = _initial_polls(lines)
        va = next(
            i for i, s in enumerate(lines) if s.lstrip().startswith("va = tl.load(a ")
        )
        vq = next(
            i
            for i, s in enumerate(lines)
            if i > va and s.lstrip().startswith("vq = tl.load(q ")
        )
        publish = next(
            i for i, s in enumerate(lines) if i > vq and "sem='release'" in s
        )
        a_poll = max(p for p in polls if p[0] < va)
        q_poll = max(p for p in polls if p[0] < vq)
        # The dot root waits for `a` before its first read of it, on every
        # path, and for `q` only inside the branch that reads it (after that
        # read of `a`), each producer on its own counter.
        self.assertNotEqual(a_poll[2], q_poll[2])
        self.assertGreater(q_poll[0], va)
        self.assertGreater(q_poll[1], a_poll[1])
        # Tasks that skip the branch never waited for `q`, and the last root
        # trusts this root's publication: it re-polls `q`, not `a`.
        self.assertEqual([p[2] for p in polls if vq < p[0] < publish], [q_poll[2]])

        def reference(flag: int) -> torch.Tensor:
            rows = [torch.zeros(16, 64, device=DEVICE) for _ in range(2)]
            rows[0][:6] = (x.float() + 1).bfloat16().float()
            rows[1][:6] = (x.float() * 2).bfloat16().float()
            mid = w.float() @ rows[0].T + flag * (w.float() @ rows[1].T)
            return mid.view(4, 16, 16).flip(0).reshape(64, 16) + rows[1].T

        def mismatches(kernel: helion.Kernel, flag: int) -> int:
            # Poisoned producer outputs: a read before its producer finished
            # turns the result into NaN.
            outs = []
            for _ in range(20):
                a.fill_(float("nan"))
                q.fill_(float("nan"))
                outs.append(kernel(x, w, flags[flag], a, q))
            torch.cuda.synchronize()
            if not torch.isnan(outs[0]).any():
                torch.testing.assert_close(
                    outs[0], reference(flag), rtol=2e-2, atol=2e-2
                )
            return sum(
                not torch.equal(out, outs[0]) or bool(torch.isnan(out).any())
                for out in outs
            )

        kernel = build()
        for flag in (1, 0):
            with self.subTest(flag=flag):
                self.assertEqual(mismatches(kernel, flag), 0)

        # Negative controls (must race): claiming that every producer was
        # waited for on every path drops the publication's re-poll; dropping
        # q's branch wait lets the branch read q early.
        original = cross_loop_codegen._insert_producer_waits

        def certify_all(
            statements: list[ast.stmt],
            names: tuple[frozenset[str] | None, ...],
            make_wait: Any,
        ) -> tuple[int, ...]:
            original(statements, names, make_wait)
            return tuple(range(len(names)))

        def skip_q(
            statements: list[ast.stmt],
            names: tuple[frozenset[str] | None, ...],
            make_wait: Any,
        ) -> tuple[int, ...]:
            def without_q(producers: tuple[int, ...]) -> list[ast.stmt]:
                kept = tuple(p for p in producers if "q" not in (names[p] or ()))
                return make_wait(kept) if kept else []

            original(statements, names, without_q)
            return tuple(range(len(names)))

        for mutation, flag in ((certify_all, 0), (skip_q, 1)):
            with (
                self.subTest(mutation=mutation.__name__),
                mock.patch.object(
                    cross_loop_codegen, "_insert_producer_waits", mutation
                ),
            ):
                self.assertGreater(mismatches(build(), flag), 0)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_waits_in_both_branches_certify_the_publication(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()
        a = torch.empty_like(x)
        q = torch.empty_like(x)
        flags = {
            v: torch.full([1], v, device=DEVICE, dtype=torch.int32) for v in (0, 1)
        }
        config = helion.Config(**_SINGLE_TRIP_DOT_CONFIG)

        def build() -> helion.Kernel:
            return helion.kernel(
                late_producer_branches, config=config, static_shapes=True
            )

        lines = build().bind((x, w, flags[1], a, q)).to_triton_code(config).splitlines()
        polls = _initial_polls(lines)
        va = next(
            i for i, s in enumerate(lines) if s.lstrip().startswith("va = tl.load(a ")
        )
        vqs = [
            i
            for i, s in enumerate(lines)
            if i > va and s.lstrip().startswith("vq = tl.load(q ")
        ][:2]
        publish = next(
            i for i, s in enumerate(lines) if i > vqs[1] and "sem='release'" in s
        )
        a_poll = max(p for p in polls if p[0] < va)
        q_polls = [max(p for p in polls if p[0] < vq) for vq in vqs]
        # Each branch waits for `q` right before its own read, after the read
        # of `a`; every path passes one of them, so the publication certifies
        # `q` without a re-poll.
        self.assertEqual(len({p[2] for p in q_polls}), 1)
        self.assertNotEqual(a_poll[2], q_polls[0][2])
        self.assertGreater(q_polls[0][0], va)
        self.assertGreater(q_polls[1][0], vqs[0])
        self.assertTrue(all(p[1] > a_poll[1] for p in q_polls))
        self.assertEqual([p for p in polls if vqs[1] < p[0] < publish], [])

        def reference(flag: int) -> torch.Tensor:
            rows = [torch.zeros(16, 64, device=DEVICE) for _ in range(2)]
            rows[0][:6] = (x.float() + 1).bfloat16().float()
            rows[1][:6] = (x.float() * 2).bfloat16().float()
            sign = 1 if flag else -1
            mid = w.float() @ rows[0].T + sign * (w.float() @ rows[1].T)
            return mid.view(4, 16, 16).flip(0).reshape(64, 16) + rows[1].T

        def mismatches(kernel: helion.Kernel, flag: int) -> int:
            outs = []
            for _ in range(20):
                a.fill_(float("nan"))
                q.fill_(float("nan"))
                outs.append(kernel(x, w, flags[flag], a, q))
            torch.cuda.synchronize()
            if not torch.isnan(outs[0]).any():
                torch.testing.assert_close(
                    outs[0], reference(flag), rtol=2e-2, atol=2e-2
                )
            return sum(
                not torch.equal(out, outs[0]) or bool(torch.isnan(out).any())
                for out in outs
            )

        kernel = build()
        for flag in (1, 0):
            with self.subTest(flag=flag):
                self.assertEqual(mismatches(kernel, flag), 0)

        # Negative control (must race): without the else branch's wait the
        # certification is wrong, and both the branch and the last root read
        # `q` early.
        original = cross_loop_codegen._insert_producer_waits

        def skip_else_q(
            statements: list[ast.stmt],
            names: tuple[frozenset[str] | None, ...],
            make_wait: Any,
        ) -> tuple[int, ...]:
            q_waits = itertools.count()

            def without_second_q(producers: tuple[int, ...]) -> list[ast.stmt]:
                is_q = [p for p in producers if "q" in (names[p] or ())]
                if is_q and next(q_waits) == 1:
                    kept = tuple(p for p in producers if p not in is_q)
                    return make_wait(kept) if kept else []
                return make_wait(producers)

            return original(statements, names, without_second_q)

        with mock.patch.object(
            cross_loop_codegen, "_insert_producer_waits", skip_else_q
        ):
            self.assertGreater(mismatches(build(), 0), 0)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_producers_of_one_tensor_share_one_poll(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()
        code, out = code_and_output(
            column_half_producers, (x, w), **_SINGLE_TRIP_DOT_CONFIG
        )
        lines = code.splitlines()
        vc = next(
            i for i, s in enumerate(lines) if s.lstrip().startswith("vc = tl.load(c ")
        )
        polls = [p for p in _initial_polls(lines) if p[0] < vc]
        # Both producers write `c`: their counters are polled in one spin
        # right before the first read, with relaxed loads that all issue at
        # once and one acquire fence after the spin.
        self.assertEqual(len({p[2] for p in polls[-2:]}), 2)
        self.assertEqual(polls[-1][0], polls[-2][0] + 1)
        self.assertTrue(lines[polls[-1][0] + 1].lstrip().startswith("while ("))
        self.assertTrue(all("ld.relaxed.gpu" in lines[p[0]] for p in polls[-2:]))
        self.assertIn("asm='fence.acquire.gpu; bar.warp.sync", lines[polls[-1][0] + 4])
        rows = torch.zeros(16, 64, device=DEVICE)
        rows[:6, :32] = (x[:, :32].float() + 1).bfloat16().float()
        rows[:6, 32:] = (x[:, 32:].float() * 2).bfloat16().float()
        torch.testing.assert_close(out, w.float() @ rows.T, rtol=2e-2, atol=2e-2)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_combined_relaxed_poll_waits_for_every_producer(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        a = torch.empty_like(x)
        b = torch.empty_like(x)
        config = helion.Config(**_SINGLE_TRIP_DOT_CONFIG)

        def build() -> helion.Kernel:
            return helion.kernel(late_pair_producers, config=config, static_shapes=True)

        lines = build().bind((x, a, b)).to_triton_code(config).splitlines()
        call = next(
            i
            for i, s in enumerate(lines)
            if s.lstrip().startswith("tile_dependency_root_2(")
        )
        polls = [p for p in _initial_polls(lines) if p[0] < call]
        # The consumer's first statement reads both producers' outputs: one
        # spin over both counters, relaxed loads and one acquire fence.
        self.assertEqual(polls[-1][0], polls[-2][0] + 1)
        self.assertNotEqual(polls[-1][2], polls[-2][2])
        self.assertTrue(all("ld.relaxed.gpu" in lines[p[0]] for p in polls[-2:]))
        self.assertIn("asm='fence.acquire.gpu; bar.warp.sync", lines[polls[-1][0] + 4])

        expected = torch.zeros(16, 64, device=DEVICE)
        expected[:6] = (x.float() + 1).bfloat16().float() + (
            x.float() * 2
        ).bfloat16().float()
        reference = expected.T

        def mismatches(kernel: helion.Kernel) -> int:
            # Poisoned producer outputs: a read before its producer finished
            # turns the result into NaN.
            outs = []
            for _ in range(20):
                a.fill_(float("nan"))
                b.fill_(float("nan"))
                outs.append(kernel(x, a, b))
            torch.cuda.synchronize()
            if not torch.isnan(outs[0]).any():
                torch.testing.assert_close(outs[0], reference)
            return sum(
                not torch.equal(out, outs[0]) or bool(torch.isnan(out).any())
                for out in outs
            )

        self.assertEqual(mismatches(build()), 0)

        # Negative control (must race): the spin polls only the first counter,
        # not the late producer's.
        original = cross_loop_codegen._wait_for_dependencies

        def first_counter_only(
            *,
            device_function: DeviceFunction,
            dependencies: tuple[tuple[str, str], ...],
            prefix: str,
        ) -> list[ast.stmt]:
            return original(
                device_function=device_function,
                dependencies=dependencies[:1],
                prefix=prefix,
            )

        with mock.patch.object(
            cross_loop_codegen, "_wait_for_dependencies", first_counter_only
        ):
            self.assertGreater(mismatches(build()), 0)

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_covers_producers_its_producer_certifies(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()
        a, b, c = (torch.empty_like(x) for _ in range(3))
        config = helion.Config(**_SINGLE_TRIP_DOT_CONFIG)
        cases = {
            "transitive_pre": (transitive_pre_chain, (x, w, a, b)),
            "transitive_split": (transitive_split_chain, (x, w, a, b, c)),
        }

        def good(name: str, out: torch.Tensor) -> bool:
            expected = w.float() @ _padded_rows(a).T + w.float() @ _padded_rows(b).T
            if name == "transitive_split":
                expected = expected + w.float() @ _padded_rows(c).T
            return torch.allclose(out, expected, rtol=2e-2, atol=2e-2)

        def poison() -> None:
            for t in (a, b, c):
                t.fill_(float("nan"))

        for name, (fn, args) in cases.items():
            with self.subTest(name=name):
                kernel = helion.kernel(fn, config=config, static_shapes=True)
                lines = kernel.bind(args).to_triton_code(config).splitlines()
                va = next(
                    i
                    for i, line in enumerate(lines)
                    if line.lstrip().startswith("va = tl.load(a ")
                )
                consumer = max(
                    i
                    for i, line in enumerate(lines[:va])
                    if "for virtual_pid in" in line
                )
                # The pruned producer of `a` is certified only by the producer
                # that reads all of `a`: that producer's wait precedes the read.
                self.assertTrue(
                    any(consumer < p[0] < va for p in _initial_polls(lines))
                )
                self.assertEqual(
                    _bad_launches(partial(kernel, *args), poison, partial(good, name)),
                    0,
                )
                # Negative control (must race): the wait ignores the memory of
                # the producer behind it (the producer's own writes only).
                with mock.patch.object(
                    cross_loop_codegen, "_insert_producer_waits", _without_names("a")
                ):
                    mutant = helion.kernel(fn, config=config, static_shapes=True)
                    self.assertGreater(
                        _bad_launches(
                            partial(mutant, *args), poison, partial(good, name)
                        ),
                        0,
                    )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_wait_precedes_overwriting_what_a_producer_reads(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()
        s0 = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        s, a, b = s0.clone(), torch.empty_like(x), torch.empty_like(x)
        config = helion.Config(**_SINGLE_TRIP_DOT_CONFIG)
        cases = {
            "war_pre": (war_pre_chain, (x, w, s, a)),
            "war_split": (war_split_chain, (x, w, s, a, b)),
        }
        a_expected = (s0.float() + 1).bfloat16()

        def good(name: str, out: torch.Tensor) -> bool:
            # The producer read the original `s`, not the consumer's 100s.
            expected = w.float() @ _padded_rows(a).T
            if name == "war_split":
                expected = expected + w.float() @ _padded_rows(b).T
            return torch.equal(a, a_expected) and torch.allclose(
                out, expected, rtol=2e-2, atol=2e-2
            )

        def poison() -> None:
            s.copy_(s0)
            a.fill_(float("nan"))
            b.fill_(float("nan"))

        for name, (fn, args) in cases.items():
            with self.subTest(name=name):
                kernel = helion.kernel(fn, config=config, static_shapes=True)
                lines = kernel.bind(args).to_triton_code(config).splitlines()
                store = next(
                    i
                    for i, line in enumerate(lines)
                    if line.lstrip().startswith("tl.store(s ")
                )
                consumer = max(
                    i
                    for i, line in enumerate(lines[:store])
                    if "for virtual_pid in" in line
                )
                polls = [p for p in _initial_polls(lines) if consumer < p[0] < store]
                self.assertEqual(len({p[2] for p in polls}), 1 + (name == "war_split"))
                self.assertEqual(
                    _bad_launches(partial(kernel, *args), poison, partial(good, name)),
                    0,
                )
                # Negative control (must race): the wait ignores what the
                # producer reads.
                with mock.patch.object(
                    cross_loop_codegen, "_insert_producer_waits", _without_names("s")
                ):
                    mutant = helion.kernel(fn, config=config, static_shapes=True)
                    self.assertGreater(
                        _bad_launches(
                            partial(mutant, *args), poison, partial(good, name)
                        ),
                        0,
                    )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_branch_wait_does_not_cover_a_read_after_the_branch(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()
        a = torch.empty_like(x)
        q = torch.empty_like(x)
        flags = {
            v: torch.full([1], v, device=DEVICE, dtype=torch.int32) for v in (0, 1)
        }
        config = helion.Config(**_SINGLE_TRIP_DOT_CONFIG)
        kernel = helion.kernel(
            branch_then_later_chain, config=config, static_shapes=True
        )
        lines = kernel.bind((x, w, flags[1], a, q)).to_triton_code(config).splitlines()
        va = next(
            i for i, s in enumerate(lines) if s.lstrip().startswith("va = tl.load(a ")
        )
        branch = next(
            i for i, s in enumerate(lines) if i > va and s.lstrip().startswith("if ")
        )
        polls = _initial_polls(lines)
        a_poll = max(p for p in polls if p[0] < va)
        q_poll = max(p for p in polls if p[0] < branch)
        # q is read after the `if` too: its wait precedes the `if`, at the
        # task's top level, not only inside the branch.
        self.assertNotEqual(a_poll[2], q_poll[2])
        self.assertGreater(q_poll[0], va)
        self.assertEqual(q_poll[1], a_poll[1])
        self.assertFalse(any(p[0] > branch for p in polls))

        def good(flag: int, out: torch.Tensor) -> bool:
            expected = w.float() @ _padded_rows(a).T + (flag + 1) * (
                w.float() @ _padded_rows(q).T
            )
            return torch.allclose(out, expected, rtol=2e-2, atol=2e-2)

        def poison() -> None:
            a.fill_(float("nan"))
            q.fill_(float("nan"))

        for flag in (0, 1):
            with self.subTest(flag=flag):
                self.assertEqual(
                    _bad_launches(
                        partial(kernel, x, w, flags[flag], a, q),
                        poison,
                        partial(good, flag),
                    ),
                    0,
                )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="only sm100 keeps dot roots inline in the task loop"
    )
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_view_of_a_written_input_waits_for_its_writer(self) -> None:
        from helion.runtime.triton import launcher

        torch.manual_seed(0)
        x = torch.randn(6, 64, device=DEVICE, dtype=torch.bfloat16)
        w = (0.1 * torch.randn(64, 64, device=DEVICE)).bfloat16()
        a, a2, b = (torch.empty_like(x) for _ in range(3))
        config = helion.Config(**_SINGLE_TRIP_DOT_CONFIG)
        kernel = helion.kernel(alias_view_chain, config=config, static_shapes=True)
        distinct = (x, w, a, a2, b)
        aliased = (x, w, a, a.view(6, 64), b)

        # Inputs sharing a storage are one allocation in the dependency graph:
        # the writer of `a` precedes the reader of `a_v`.
        def edges(args: tuple[torch.Tensor, ...]) -> set[tuple[int, int]]:
            host_function = kernel.bind(args).host_function
            assert host_function is not None
            graph = host_function.device_ir.tile_dependency_graph
            assert graph is not None
            return {(edge.producer_root, edge.consumer_root) for edge in graph.edges}

        self.assertNotIn((0, 2), edges(distinct))
        self.assertIn((0, 2), edges(aliased))
        self.assertIsNot(kernel.bind(aliased), kernel.bind(distinct))
        lines = kernel.bind(aliased).to_triton_code(config).splitlines()
        va = next(
            i for i, s in enumerate(lines) if s.lstrip().startswith("va = tl.load(a_v ")
        )
        consumer = max(
            i for i, line in enumerate(lines[:va]) if "for virtual_pid in" in line
        )
        polls = [p for p in _initial_polls(lines) if consumer < p[0] < va]
        self.assertEqual(len({p[2] for p in polls}), 2)
        self.assertIn("_tensor_storage_aliases=((", "\n".join(lines))

        def good(out: torch.Tensor) -> bool:
            expected = w.float() @ _padded_rows(a).T + w.float() @ _padded_rows(b).T
            return torch.allclose(out, expected, rtol=2e-2, atol=2e-2)

        def poison() -> None:
            a.fill_(float("nan"))
            b.fill_(float("nan"))

        # The kernel's cache key includes the storage aliasing of its inputs:
        # after a call with distinct buffers, the aliasing call compiles for it.
        a2.copy_(x)
        torch.testing.assert_close(
            kernel(*distinct),
            w.float() @ _padded_rows(a2).T + w.float() @ _padded_rows(b).T,
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertEqual(_bad_launches(partial(kernel, *aliased), poison, good), 0)
        # So does calling the binding compiled for distinct buffers.
        bound = kernel.bind(distinct)
        self.assertEqual(_bad_launches(partial(bound, *aliased), poison, good), 0)
        # Its compiled code alone (as exported, no compiler behind it): the
        # launcher refuses the aliasing launch.
        run = bound._run
        assert run is not None
        self.assertRaises(helion.exc.CrossLoopAliasedLaunch, run, *aliased)
        # Negative control (must race): without that check the code's
        # schedule does not order the read of `a_v` after the writer of `a`.
        with mock.patch.object(
            launcher, "_check_compiled_storage_sharing", lambda *args: None
        ):
            self.assertGreater(_bad_launches(partial(run, *aliased), poison, good), 0)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_inputs_sharing_storage_overlap_at_any_offset(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(8, 64, device=DEVICE, dtype=torch.bfloat16)
        buf = torch.empty(16, 64, device=DEVICE, dtype=torch.bfloat16)
        aliased = (x, buf[8:], buf)
        distinct = (x, torch.empty_like(x), torch.empty_like(buf))

        def edges(
            args: tuple[torch.Tensor, ...], static_shapes: bool = True
        ) -> set[tuple[int, int]]:
            kernel = helion.kernel(
                offset_alias_chain, static_shapes=static_shapes, autotune_effort="none"
            )
            host_function = kernel.bind(args).host_function
            assert host_function is not None
            graph = host_function.device_ir.tile_dependency_graph
            assert graph is not None
            return {(edge.producer_root, edge.consumer_root) for edge in graph.edges}

        # The second root reads where the first writes only through the
        # offset of `a` in `buf`, which the cache key does not record (the
        # fakes of inputs sharing a storage all start at offset 0, where these
        # rows are disjoint): their accesses overlap whatever the offsets.
        self.assertEqual(edges(distinct), set())
        for static_shapes in (True, False):
            with self.subTest(static_shapes=static_shapes):
                self.assertEqual(edges(aliased, static_shapes), {(0, 1)})
        # Negative control: trusting the offset-0 layouts loses the edge.
        original = CompileEnvironment._aliased_input_fake

        def trusted_layouts(
            env: CompileEnvironment, tensor: torch.Tensor
        ) -> torch.Tensor | None:
            fake = original(env, tensor)
            env.aliased_input_storages.clear()
            return fake

        with mock.patch.object(
            CompileEnvironment, "_aliased_input_fake", trusted_layouts
        ):
            self.assertEqual(edges(aliased), set())

        config = helion.Config(
            block_sizes=[],
            cross_loop_pipeline="static",
            pid_type="persistent_blocked",
            num_warps=4,
            num_stages=1,
            num_sm_multiplier=1,
            indexing="pointer",
        )
        kernel = helion.kernel(offset_alias_chain, config=config, static_shapes=True)
        expected = (x.float() + 1).bfloat16().float()

        def poison() -> None:
            buf.fill_(float("nan"))

        self.assertEqual(
            _bad_launches(
                partial(kernel, *aliased), poison, partial(torch.equal, expected)
            ),
            0,
        )

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
        # The early read keeps its own entry wait; the nested wait follows it.
        self.assertNotIn("tile_dependency_root_barrier", code)
        scheduled = ast.unparse(
            _generated_function(code, "tile_dependency_root_1_scheduled_task")
        )
        self.assertLess(
            scheduled.index("tile_dependency_readiness_wait"),
            scheduled.index("tile_dependency_root_1("),
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

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_partially_read_producer_tiles_publish_counted_arrivals(self) -> None:
        # Consumers read 2 of each head's 3 producer row tiles.  Exact-equality
        # waits hang unless only the counted producer tasks publish.
        torch.manual_seed(0)
        x = torch.randn([2, 96, 16], device=DEVICE, dtype=torch.float16)
        w = torch.randn([2, 64, 128], device=DEVICE, dtype=torch.float16)
        expected = torch.einsum("hdc,hdt->hct", w.float(), x[:, :64].float() * 2).half()
        original_build = cross_loop_codegen.build_static_pipeline_plan
        plans: list[cross_loop_scheduler.StaticPipelinePlan] = []

        def record(**kwargs: Any) -> cross_loop_scheduler.StaticPipelinePlan:
            plan = original_build(**kwargs)
            plans.append(plan)
            return plan

        for pipeline in ("dynamic", "static"):
            with self.subTest(pipeline=pipeline):
                config = {
                    "block_sizes": [32],
                    "pid_type": "persistent_blocked",
                    "cross_loop_pipeline": pipeline,
                    "num_warps": 4,
                }
                plans.clear()
                with mock.patch.object(
                    cross_loop_codegen, "build_static_pipeline_plan", side_effect=record
                ):
                    partial_row_tile_absorb.bind((x, w)).to_triton_code(
                        helion.Config(**config)
                    )
                (plan,) = plans
                # Check before launching: an inconsistent counter deadlocks.
                mismatches, partial_arms = _readiness_arrival_mismatches(plan)
                self.assertEqual(mismatches, [])
                self.assertGreater(partial_arms, 0)
                self.assertNotIn((0, 1), plan.root_barrier_edges)

                code, out = code_and_output(partial_row_tile_absorb, (x, w), **config)
                torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                self.assertNotIn("tile_dependency_root_barrier", code)
