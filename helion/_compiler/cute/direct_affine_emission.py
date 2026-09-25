# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compose validated direct-affine templates into replacement CuTe AST."""

from __future__ import annotations

import ast
import keyword
from typing import TYPE_CHECKING
from typing import AbstractSet
from typing import cast

from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from .direct_affine_plan import DirectAffineMma
from .direct_affine_plan import DirectAffinePhaseOrder
from .direct_affine_plan import DirectAffinePlan
from .direct_affine_plan import DirectAffineStateIngress
from .direct_affine_replay_core import DirectAffineAsyncEntryReplay
from .direct_affine_replay_core import DirectAffineCoordinates
from .direct_affine_replay_core import DirectAffineEffectReplay
from .direct_affine_replay_core import DirectAffineEmission
from .direct_affine_replay_core import DirectAffineOrdinaryAxis
from .direct_affine_replay_core import DirectAffineReplay
from .direct_affine_replay_core import DirectAffineReplayProof
from .direct_affine_replay_core import DirectAffineResolvedTemplates
from .direct_affine_replay_core import DirectAffineSharedBuffer
from .direct_affine_replay_core import DirectAffineStateStoreReplay
from .direct_affine_replay_core import DirectAffineStepReplay
from .direct_affine_replay_core import DirectAffineValueReplay
from .direct_affine_replay_core import _ast_identity_set
from .direct_affine_replay_core import _AsyncStateEmission
from .direct_affine_replay_core import _bound_names
from .direct_affine_replay_core import _clone_ast
from .direct_affine_replay_core import _dump
from .direct_affine_replay_core import _instantiate_bound_nodes
from .direct_affine_replay_core import _qualified_name
from .direct_affine_replay_core import _result_snapshots_are_intact
from .direct_affine_replay_core import instantiate_direct_affine_value
from .direct_affine_replay_core import resolve_direct_affine_coordinates
from .direct_affine_templates import _has_lane_reduce_marker
from .direct_affine_templates import _template_components
from .direct_affine_templates import _validate_direct_affine_templates
from .direct_affine_templates import resolve_direct_affine_templates
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import allocate_shared_regions

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


_SMEM_ALIGNMENT = 128
_HELPER_MODULE = "helion._compiler.cute.short_affine_scan_mma"


def _name(name: str, ctx: ast.expr_context | None = None) -> ast.Name:
    return ast.Name(id=name, ctx=ctx or ast.Load())


def _attr(value: ast.expr, name: str) -> ast.Attribute:
    return ast.Attribute(value=value, attr=name, ctx=ast.Load())


def _qualified(value: str) -> ast.expr:
    parts = value.split(".")
    result: ast.expr = _name(parts[0])
    for part in parts[1:]:
        result = _attr(result, part)
    return result


def _call(function: ast.expr, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return ast.Call(
        func=function,
        args=list(args),
        keywords=[ast.keyword(arg=name, value=value) for name, value in kwargs.items()],
    )


def _cast_i32(value: ast.expr) -> ast.Call:
    return _call(_qualified("cutlass.Int32"), value)


def _add(left: ast.expr, right: ast.expr) -> ast.BinOp:
    return ast.BinOp(left=left, op=ast.Add(), right=right)


def _mul(left: ast.expr, right: ast.expr) -> ast.BinOp:
    return ast.BinOp(left=left, op=ast.Mult(), right=right)


def _global_coordinate(axis: DirectAffineOrdinaryAxis, local: ast.expr) -> ast.expr:
    if isinstance(axis.tile_offset, ast.Constant) and axis.tile_offset.value == 0:
        return _clone_ast(local)
    return _add(_clone_ast(axis.tile_offset), _clone_ast(local))


def _pointer(name: str) -> ast.Attribute:
    return _attr(_name(name), "iterator")


def _store(pointer: ast.expr, index: ast.expr, value: ast.expr) -> ast.Expr:
    address = _add(pointer, _cast_i32(index))
    return ast.Expr(value=_call(_attr(address, "store"), value))


def _shared_layout(
    plan: DirectAffinePlan, prefix: str
) -> tuple[tuple[DirectAffineSharedBuffer, ...], int]:
    columns = plan.columns.factor_column_extent
    sizes = (
        ("state", "BFloat16", plan.row_extent * plan.feature_extent, 2),
        ("factor", "BFloat16", plan.feature_extent * columns, 2),
        ("coefficient", "Float32", plan.columns.coefficient_element_count, 4),
        ("prediction", "Float32", plan.step_count * plan.feature_extent, 4),
        ("observation", "Float32", plan.step_count * plan.feature_extent, 4),
        ("diagonal", "Float32", plan.step_count * plan.feature_extent, 4),
        ("update", "Float32", plan.step_count * plan.feature_extent, 4),
        ("update_scale", "Float32", plan.step_count, 4),
        ("residual", "Float32", plan.step_count * plan.row_extent, 4),
    )
    layout = allocate_shared_regions(
        tuple(
            SharedBufferRequest(role, count * item_size, _SMEM_ALIGNMENT)
            for role, _dtype_name, count, item_size in sizes
        ),
        final_alignment=_SMEM_ALIGNMENT,
    )
    buffers = tuple(
        DirectAffineSharedBuffer(
            role=role,
            name=f"{prefix}_{role}",
            dtype_name=dtype_name,
            element_count=count,
            byte_offset=layout.region(role).byte_offset,
        )
        for role, dtype_name, count, _item_size in sizes
    )
    return buffers, layout.allocated_bytes


def _buffer_setup(
    buffers: Sequence[DirectAffineSharedBuffer],
    *,
    smem_name: str,
    smem_bytes: int,
) -> tuple[ast.stmt, ...]:
    result: list[ast.stmt] = [
        ast.Assign(
            targets=[_name(smem_name, ast.Store())],
            value=_call(
                _qualified("cute.arch.alloc_smem"),
                _qualified("cutlass.Uint8"),
                ast.Constant(value=smem_bytes),
                alignment=ast.Constant(value=_SMEM_ALIGNMENT),
            ),
        )
    ]
    for buffer in buffers:
        byte_pointer = _add(
            _name(smem_name),
            _cast_i32(ast.Constant(value=buffer.byte_offset)),
        )
        typed_pointer = _call(
            _qualified("cute.recast_ptr"),
            byte_pointer,
            dtype=_qualified(f"cutlass.{buffer.dtype_name}"),
        )
        result.append(
            ast.Assign(
                targets=[_name(buffer.name, ast.Store())],
                value=_call(
                    _qualified("cute.make_tensor"),
                    typed_pointer,
                    ast.Tuple(
                        elts=[ast.Constant(value=buffer.element_count)],
                        ctx=ast.Load(),
                    ),
                ),
            )
        )
    return tuple(result)


def _names_in_program(
    replay: DirectAffineReplay,
    templates: DirectAffineResolvedTemplates,
) -> set[str]:
    trees: list[ast.AST] = [*replay.source_statements]
    for _, component in _template_components(templates):
        trees.extend(component)
    return {
        node.id
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }


def _instantiate_effect(
    replay: DirectAffineEffectReplay,
    *,
    value: ast.expr,
    row: ast.expr,
    feature: ast.expr | None,
    row_local: ast.expr,
    feature_local: ast.expr | None,
    suffix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    result = _instantiate_bound_nodes(
        replay.bindings,
        replay.statements,
        (),
        row=row,
        feature=feature,
        row_local=row_local,
        feature_local=feature_local,
        extra_substitutions=((replay.logical_value, value),),
        suffix=suffix,
        reserved_names=reserved_names,
    )
    return result[0] if result is not None else None


def _instantiate_state_address(
    replay: DirectAffineStateStoreReplay,
    *,
    row: ast.expr,
    feature: ast.expr,
    row_local: ast.expr,
    feature_local: ast.expr,
    suffix: str,
    reserved_names: AbstractSet[str],
) -> tuple[tuple[ast.stmt, ...], ast.expr, ast.expr, ast.expr, ast.expr] | None:
    result = _instantiate_bound_nodes(
        replay.bindings,
        replay.statements,
        (replay.pointer, replay.slot, replay.slot_extent, replay.valid),
        row=row,
        feature=feature,
        row_local=row_local,
        feature_local=feature_local,
        suffix=suffix,
        reserved_names=reserved_names,
    )
    if result is None:
        return None
    statements, outputs = result
    return statements, outputs[0], outputs[1], outputs[2], outputs[3]


def _instantiate_async_entry(
    replay: DirectAffineAsyncEntryReplay,
    *,
    row: ast.expr,
    feature: ast.expr,
    row_local: ast.expr,
    feature_local: ast.expr,
    suffix: str,
    reserved_names: AbstractSet[str],
) -> tuple[tuple[ast.stmt, ...], ast.expr, ast.expr, ast.expr] | None:
    result = _instantiate_bound_nodes(
        replay.bindings,
        replay.statements,
        (replay.pointer, replay.row_stride, replay.valid),
        row=row,
        feature=feature,
        row_local=row_local,
        feature_local=feature_local,
        suffix=suffix,
        reserved_names=reserved_names,
    )
    if result is None:
        return None
    statements, outputs = result
    return statements, outputs[0], outputs[1], outputs[2]


def _sync_threads() -> ast.Expr:
    return ast.Expr(value=_call(_qualified("cute.arch.sync_threads")))


def _emit_state_materialization(
    entry: DirectAffineValueReplay,
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    *,
    helper_alias: str,
    buffers: Mapping[str, DirectAffineSharedBuffer],
    lane_name: str,
    warp_name: str,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    threads = plan.cta_shape[0] * plan.cta_shape[1]
    elements = plan.row_extent * plan.feature_extent
    slots = (elements + threads - 1) // threads
    slot_name = f"{prefix}_state_slot"
    linear_name = f"{prefix}_state_linear"
    local_row_name = f"{prefix}_state_row"
    local_feature_name = f"{prefix}_state_feature"
    linear = _add(
        _add(
            _mul(_name(warp_name), ast.Constant(value=32)),
            _name(lane_name),
        ),
        _mul(_name(slot_name), ast.Constant(value=threads)),
    )
    local_row = ast.BinOp(
        left=_name(linear_name),
        op=ast.FloorDiv(),
        right=ast.Constant(value=plan.feature_extent),
    )
    local_feature = ast.BinOp(
        left=_name(linear_name),
        op=ast.Mod(),
        right=ast.Constant(value=plan.feature_extent),
    )
    instantiated = instantiate_direct_affine_value(
        entry,
        row=_global_coordinate(coordinates.row, _name(local_row_name)),
        feature=_global_coordinate(coordinates.feature, _name(local_feature_name)),
        row_local=_name(local_row_name),
        feature_local=_name(local_feature_name),
        suffix=f"{prefix.strip('_')}_entry",
        reserved_names=reserved_names,
    )
    if instantiated is None:
        return None
    producer_statements, value = instantiated
    state_index = _call(
        _attr(_name(helper_alias), "direct_state_index"),
        _name(local_row_name),
        _name(local_feature_name),
        ast.Constant(value=plan.row_extent),
    )
    body: list[ast.stmt] = [
        ast.Assign(targets=[_name(linear_name, ast.Store())], value=linear),
        ast.If(
            test=ast.Compare(
                left=_name(linear_name),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value=elements)],
            ),
            body=[
                ast.Assign(
                    targets=[_name(local_row_name, ast.Store())], value=local_row
                ),
                ast.Assign(
                    targets=[_name(local_feature_name, ast.Store())],
                    value=local_feature,
                ),
                *producer_statements,
                _store(
                    _pointer(buffers["state"].name),
                    state_index,
                    _call(_qualified("cutlass.BFloat16"), value),
                ),
            ],
            orelse=[],
        ),
    ]
    return (
        ast.For(
            target=_name(slot_name, ast.Store()),
            iter=_call(
                _qualified("cutlass.range_constexpr"), ast.Constant(value=slots)
            ),
            body=body,
            orelse=[],
        ),
        _sync_threads(),
    )


def _emit_async_state_materialization(
    entry: DirectAffineAsyncEntryReplay,
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    *,
    helper_alias: str,
    buffers: Mapping[str, DirectAffineSharedBuffer],
    lane_name: str,
    warp_name: str,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> _AsyncStateEmission | None:
    local_row_name = f"{prefix}_async_state_row"
    local_feature_name = f"{prefix}_async_state_feature"
    local_row = _add(
        _mul(_name(warp_name), ast.Constant(value=16)),
        _mul(
            ast.BinOp(
                left=_name(lane_name),
                op=ast.FloorDiv(),
                right=ast.Constant(value=16),
            ),
            ast.Constant(value=8),
        ),
    )
    local_feature = _mul(
        ast.BinOp(
            left=_name(lane_name),
            op=ast.Mod(),
            right=ast.Constant(value=16),
        ),
        ast.Constant(value=8),
    )
    instantiated = _instantiate_async_entry(
        entry,
        row=_global_coordinate(coordinates.row, _name(local_row_name)),
        feature=_global_coordinate(coordinates.feature, _name(local_feature_name)),
        row_local=_name(local_row_name),
        feature_local=_name(local_feature_name),
        suffix=f"{prefix.strip('_')}_async_entry",
        reserved_names=reserved_names,
    )
    if instantiated is None:
        return None
    support, pointer, row_stride, valid = instantiated
    call = ast.Expr(
        value=_call(
            _attr(_name(helper_alias), "stage_state_tile8x8_async_bf16"),
            pointer,
            ast.Constant(value=0),
            row_stride,
            _pointer(buffers["state"].name),
            _name(local_row_name),
            _name(local_feature_name),
            valid,
            ast.Constant(value=plan.row_extent),
        )
    )
    return _AsyncStateEmission(
        begin=(
            ast.If(
                test=ast.Compare(
                    left=_name(warp_name),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=plan.row_warps)],
                ),
                body=[
                    ast.Assign(
                        targets=[_name(local_row_name, ast.Store())], value=local_row
                    ),
                    ast.Assign(
                        targets=[_name(local_feature_name, ast.Store())],
                        value=local_feature,
                    ),
                    *support,
                    call,
                ],
                orelse=[],
            ),
        ),
        finish=(
            ast.Expr(
                value=_call(
                    _qualified("cute.arch.cp_async_wait_group"),
                    ast.Constant(value=0),
                )
            ),
            _sync_threads(),
        ),
    )


def _materialized_vector(
    replay: DirectAffineValueReplay,
    *,
    step: int,
    feature: ast.expr,
    global_feature: ast.expr,
    global_row_zero: ast.expr,
    destination: DirectAffineSharedBuffer,
    feature_extent: int,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    instantiated = instantiate_direct_affine_value(
        replay,
        row=global_row_zero,
        feature=global_feature,
        row_local=ast.Constant(value=0),
        feature_local=feature,
        suffix=prefix,
        reserved_names=reserved_names,
    )
    if instantiated is None:
        return None
    statements, value = instantiated
    index = _add(
        ast.Constant(value=step * feature_extent),
        _clone_ast(feature),
    )
    return (
        *statements,
        _store(
            _pointer(destination.name),
            index,
            _call(_qualified("cutlass.Float32"), value),
        ),
    )


def _split_feature_reduction_loop(
    loop: ast.For,
    buffers: Mapping[str, DirectAffineSharedBuffer],
    lane_name: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    """Finalize captured feature reductions before their vector consumers."""

    from ... import exc
    from ..tile_strategy import _is_lane_reduce_marker_assign
    from ..tile_strategy import split_lane_loop_reductions

    markers = tuple(
        marker
        for statement in loop.body
        if (marker := _is_lane_reduce_marker_assign(statement)) is not None
    )
    tensor_names = {
        node.value.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }
    shared_names = {buffer.name for buffer in buffers.values()}
    disjoint_pairs = {
        frozenset((shared, other))
        for shared in shared_names
        for other in tensor_names
        if shared != other
    }
    try:
        result = split_lane_loop_reductions(
            [ast.fix_missing_locations(loop)],
            proven_disjoint_tensor_pairs=disjoint_pairs,
            thread_axis_names={lane_name: frozenset((0,))},
        )
    except exc.BackendUnsupported:
        return None
    detached = tuple(_clone_ast(statement) for statement in result)
    original_bound = _bound_names(cast("tuple[ast.stmt, ...]", tuple(loop.body)))
    result_bound = _bound_names(cast("tuple[ast.stmt, ...]", detached))
    if (result_bound - original_bound).intersection(reserved_names):
        return None
    expected_reductions: dict[str, int] = {}
    for marker in markers:
        expected_reductions[marker.reduction_type] = (
            expected_reductions.get(marker.reduction_type, 0) + 1
        )
    actual_reductions: dict[str, int] = {}
    for statement in detached:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            name = _qualified_name(node.func)
            prefix = "cute.arch.warp_reduction_"
            if name is not None and name.startswith(prefix):
                kind = name.removeprefix(prefix)
                actual_reductions[kind] = actual_reductions.get(kind, 0) + 1
    if _has_lane_reduce_marker(detached) or actual_reductions != expected_reductions:
        return None
    return cast("tuple[ast.stmt, ...]", detached)


def _emit_coefficient_materialization(
    steps: Sequence[DirectAffineStepReplay],
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    *,
    helper_alias: str,
    buffers: Mapping[str, DirectAffineSharedBuffer],
    lane_name: str,
    warp_name: str,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    element_name = f"{prefix}_feature_element"
    local_feature = _add(
        _mul(_name(lane_name), ast.Constant(value=4)),
        _name(element_name),
    )
    global_feature = _global_coordinate(coordinates.feature, local_feature)
    global_row_zero = _global_coordinate(coordinates.row, ast.Constant(value=0))
    result: list[ast.stmt] = []
    for step_index, step in enumerate(steps):
        feature_body: list[ast.stmt] = []
        for role, value_replay in (
            ("diagonal", step.diagonal),
            ("prediction", step.prediction_vector),
            ("update", step.update_vector),
            ("observation", step.observation_vector),
        ):
            materialized = _materialized_vector(
                value_replay,
                step=step_index,
                feature=local_feature,
                global_feature=global_feature,
                global_row_zero=global_row_zero,
                destination=buffers[role],
                feature_extent=plan.feature_extent,
                prefix=f"{prefix.strip('_')}_{role}_{step_index}",
                reserved_names=reserved_names,
            )
            if materialized is None:
                return None
            feature_body.extend(materialized)
        scale = instantiate_direct_affine_value(
            step.update_scale,
            row=_clone_ast(global_row_zero),
            feature=_global_coordinate(coordinates.feature, ast.Constant(value=0)),
            row_local=ast.Constant(value=0),
            feature_local=ast.Constant(value=0),
            suffix=f"{prefix.strip('_')}_scale_{step_index}",
            reserved_names=reserved_names,
        )
        if scale is None:
            return None
        scale_statements, scale_value = scale
        feature_loop = ast.For(
            target=_name(element_name, ast.Store()),
            iter=_call(_qualified("cutlass.range_constexpr"), ast.Constant(value=4)),
            body=feature_body,
            orelse=[],
        )
        setattr(feature_loop, HELION_LANE_LOOP_VAR_ATTR, element_name)
        split_feature_loop = _split_feature_reduction_loop(
            feature_loop,
            buffers,
            lane_name,
            reserved_names,
        )
        if split_feature_loop is None:
            return None
        guarded_body: list[ast.stmt] = [
            *split_feature_loop,
            *scale_statements,
            ast.If(
                test=ast.Compare(
                    left=_name(lane_name),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[
                    _store(
                        _pointer(buffers["update_scale"].name),
                        ast.Constant(value=step_index),
                        _call(_qualified("cutlass.Float32"), scale_value),
                    )
                ],
                orelse=[],
            ),
        ]
        result.append(
            ast.If(
                test=ast.Compare(
                    left=_name(warp_name),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=step_index)],
                ),
                body=guarded_body,
                orelse=[],
            )
        )
    result.append(_sync_threads())
    columns = plan.columns
    result.extend(
        (
            ast.If(
                test=ast.Compare(
                    left=_name(warp_name),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=plan.step_count)],
                ),
                body=[
                    ast.Expr(
                        value=_call(
                            _attr(
                                _name(helper_alias),
                                "precompute_affine_from_buffers_bf16",
                            ),
                            _pointer(buffers["prediction"].name),
                            _pointer(buffers["observation"].name),
                            _pointer(buffers["diagonal"].name),
                            _pointer(buffers["update"].name),
                            _pointer(buffers["update_scale"].name),
                            _pointer(buffers["factor"].name),
                            _pointer(buffers["coefficient"].name),
                            _name(lane_name),
                            _name(warp_name),
                            ast.Constant(value=plan.step_count),
                            ast.Constant(value=plan.feature_extent),
                            ast.Constant(value=columns.factor_column_extent),
                            ast.Constant(value=columns.factor_target_stride),
                            ast.Constant(value=columns.factor_role_stride),
                            ast.Constant(value=columns.coefficient_row_stride),
                            ast.Constant(value=columns.coefficient_source_stride),
                            ast.Constant(value=columns.coefficient_role_stride),
                        )
                    )
                ],
                orelse=[],
            ),
            _sync_threads(),
        )
    )
    return tuple(result)


def _subscript(name: str, index: ast.expr) -> ast.Subscript:
    return ast.Subscript(value=_name(name), slice=index, ctx=ast.Load())


def _emit_row_inputs(
    steps: Sequence[DirectAffineStepReplay],
    coordinates: DirectAffineCoordinates,
    *,
    row_low: ast.expr,
    row_high: ast.expr,
    low_name: str,
    high_name: str,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    result: list[ast.stmt] = []
    for step_index, step in enumerate(steps):
        for label, local_row, destination in (
            ("low", row_low, low_name),
            ("high", row_high, high_name),
        ):
            instantiated = instantiate_direct_affine_value(
                step.row_input,
                row=_global_coordinate(coordinates.row, local_row),
                feature=_global_coordinate(
                    coordinates.feature,
                    ast.Constant(value=0),
                ),
                row_local=local_row,
                feature_local=ast.Constant(value=0),
                suffix=f"{prefix.strip('_')}_row_{step_index}_{label}",
                reserved_names=reserved_names,
            )
            if instantiated is None:
                return None
            statements, value = instantiated
            result.extend(statements)
            target = ast.Subscript(
                value=_name(destination),
                slice=ast.Constant(value=step_index),
                ctx=ast.Store(),
            )
            result.append(
                ast.Assign(
                    targets=[target],
                    value=_call(_qualified("cutlass.Float32"), value),
                )
            )
    return tuple(result)


def _emit_output_effects(
    step: DirectAffineStepReplay,
    step_index: int,
    coordinates: DirectAffineCoordinates,
    *,
    row_low: ast.expr,
    row_high: ast.expr,
    output_low_name: str,
    output_high_name: str,
    lane_name: str,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    statements: list[ast.stmt] = []
    for label, local_row, values_name in (
        ("low", row_low, output_low_name),
        ("high", row_high, output_high_name),
    ):
        effect = _instantiate_effect(
            step.output_effect,
            value=_subscript(values_name, ast.Constant(value=step_index)),
            row=_global_coordinate(coordinates.row, local_row),
            feature=None,
            row_local=local_row,
            feature_local=None,
            suffix=f"{prefix.strip('_')}_output_{step_index}_{label}",
            reserved_names=reserved_names,
        )
        if effect is None:
            return None
        statements.extend(effect)
    return (
        ast.If(
            test=ast.Compare(
                left=ast.BinOp(
                    left=_name(lane_name),
                    op=ast.Mod(),
                    right=ast.Constant(value=4),
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=0)],
            ),
            body=statements,
            orelse=[],
        ),
    )


def _emit_state_effects(
    step: DirectAffineStepReplay,
    step_index: int,
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    *,
    helper_alias: str,
    buffers: Mapping[str, DirectAffineSharedBuffer],
    lane_name: str,
    row_base_name: str,
    history_name: str,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    diagonal_name = f"{prefix}_checkpoint_diagonal_{step_index}"
    update_name = f"{prefix}_checkpoint_update_{step_index}"
    row_offset_name = f"{prefix}_checkpoint_row_{step_index}"
    packed_name = f"{prefix}_checkpoint_packed_{step_index}"
    factor_call = _call(
        _attr(_name(helper_alias), "load_checkpoint_factors_bf16"),
        _pointer(buffers["diagonal"].name),
        _pointer(buffers["update"].name),
        _name(lane_name),
        ast.Constant(value=step_index),
        ast.Constant(value=plan.feature_extent),
    )
    local_row = _add(
        _add(
            _name(row_base_name),
            _mul(
                ast.BinOp(
                    left=_name(lane_name),
                    op=ast.FloorDiv(),
                    right=ast.Constant(value=16),
                ),
                ast.Constant(value=8),
            ),
        ),
        _name(row_offset_name),
    )
    local_feature = _mul(
        ast.BinOp(
            left=_name(lane_name),
            op=ast.Mod(),
            right=ast.Constant(value=16),
        ),
        ast.Constant(value=8),
    )
    checkpoint = _call(
        _attr(_name(helper_alias), "checkpoint_affine_row_bf16"),
        _name(history_name),
        _pointer(buffers["residual"].name),
        _name(diagonal_name),
        _name(update_name),
        _pointer(buffers["update_scale"].name),
        _name(lane_name),
        _name(row_base_name),
        ast.Constant(value=step_index),
        _name(row_offset_name),
        ast.Constant(value=plan.row_extent),
    )
    checkpoint_statement = ast.Assign(
        targets=[
            ast.Tuple(
                elts=[
                    _name(history_name, ast.Store()),
                    _name(packed_name, ast.Store()),
                ],
                ctx=ast.Store(),
            )
        ],
        value=checkpoint,
    )
    state_effect = step.state_effect
    address = _instantiate_state_address(
        state_effect,
        row=_global_coordinate(coordinates.row, local_row),
        feature=_global_coordinate(coordinates.feature, local_feature),
        row_local=local_row,
        feature_local=local_feature,
        suffix=f"{prefix.strip('_')}_state_{step_index}",
        reserved_names=reserved_names,
    )
    if address is None:
        return None
    address_statements, pointer, slot, slot_extent, valid = address
    packed_values = tuple(
        _subscript(packed_name, ast.Constant(value=index)) for index in range(4)
    )
    packed_store = ast.Expr(
        value=_call(
            _attr(_name(helper_alias), "store_packed_b16x8_if_valid"),
            pointer,
            *packed_values,
            slot,
            slot_extent,
        )
    )
    guarded_store: ast.stmt = packed_store
    if not (isinstance(valid, ast.Constant) and valid.value is True):
        guarded_store = ast.If(test=valid, body=[packed_store], orelse=[])
    loop_body = [checkpoint_statement, *address_statements, guarded_store]
    return (
        ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[
                        _name(diagonal_name, ast.Store()),
                        _name(update_name, ast.Store()),
                    ],
                    ctx=ast.Store(),
                )
            ],
            value=factor_call,
        ),
        ast.For(
            target=_name(row_offset_name, ast.Store()),
            iter=_call(_qualified("cutlass.range_constexpr"), ast.Constant(value=8)),
            body=loop_body,
            orelse=[],
        ),
    )


def _emit_consume(
    steps: Sequence[DirectAffineStepReplay],
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    *,
    helper_alias: str,
    buffers: Mapping[str, DirectAffineSharedBuffer],
    lane_name: str,
    warp_name: str,
    prefix: str,
    reserved_names: AbstractSet[str],
) -> tuple[ast.stmt, ...] | None:
    row_base_name = f"{prefix}_row_base"
    accumulator_name = f"{prefix}_projection"
    history_name = f"{prefix}_history"
    row_input_low_name = f"{prefix}_row_input_low"
    row_input_high_name = f"{prefix}_row_input_high"
    output_low_name = f"{prefix}_output_low"
    output_high_name = f"{prefix}_output_high"
    row_low = _add(
        _name(row_base_name),
        ast.BinOp(
            left=_name(lane_name),
            op=ast.FloorDiv(),
            right=ast.Constant(value=4),
        ),
    )
    row_high = _add(_clone_ast(row_low), ast.Constant(value=8))
    project_helper = {
        DirectAffineMma.M16N8: "project_retain_affine_m16n8_bf16",
        DirectAffineMma.M16N16: "project_retain_affine_m16n16_bf16",
    }.get(plan.mma)
    if project_helper is None:
        return None
    body: list[ast.stmt] = [
        ast.Assign(
            targets=[_name(row_base_name, ast.Store())],
            value=_mul(_name(warp_name), ast.Constant(value=16)),
        ),
        ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[
                        _name(accumulator_name, ast.Store()),
                        _name(history_name, ast.Store()),
                    ],
                    ctx=ast.Store(),
                )
            ],
            value=_call(
                _attr(_name(helper_alias), project_helper),
                _pointer(buffers["state"].name),
                _pointer(buffers["factor"].name),
                _name(lane_name),
                _name(row_base_name),
                ast.Constant(value=plan.row_extent),
                ast.Constant(value=plan.feature_extent),
            ),
        ),
        ast.Assign(
            targets=[_name(row_input_low_name, ast.Store())],
            value=_call(
                _qualified("cute.make_rmem_tensor"),
                ast.Constant(value=plan.step_count),
                _qualified("cutlass.Float32"),
            ),
        ),
        ast.Assign(
            targets=[_name(row_input_high_name, ast.Store())],
            value=_call(
                _qualified("cute.make_rmem_tensor"),
                ast.Constant(value=plan.step_count),
                _qualified("cutlass.Float32"),
            ),
        ),
    ]
    row_inputs = _emit_row_inputs(
        steps,
        coordinates,
        row_low=row_low,
        row_high=row_high,
        low_name=row_input_low_name,
        high_name=row_input_high_name,
        prefix=prefix,
        reserved_names=reserved_names,
    )
    if row_inputs is None:
        return None
    body.extend(row_inputs)
    columns = plan.columns
    body.append(
        ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[
                        _name(output_low_name, ast.Store()),
                        _name(output_high_name, ast.Store()),
                    ],
                    ctx=ast.Store(),
                )
            ],
            value=_call(
                _attr(_name(helper_alias), "consume_affine_steps"),
                _name(accumulator_name),
                _name(row_input_low_name),
                _name(row_input_high_name),
                _pointer(buffers["residual"].name),
                _pointer(buffers["coefficient"].name),
                _name(lane_name),
                _name(row_base_name),
                ast.Tuple(
                    elts=[
                        ast.Constant(value=value)
                        for value in columns.prediction_projection_columns
                    ],
                    ctx=ast.Load(),
                ),
                ast.Tuple(
                    elts=[
                        ast.Constant(value=value)
                        for value in columns.observation_projection_columns
                    ],
                    ctx=ast.Load(),
                ),
                ast.Constant(value=plan.step_count),
                ast.Constant(value=plan.row_extent),
                ast.Constant(value=columns.coefficient_row_stride),
                ast.Constant(value=columns.coefficient_source_stride),
                ast.Constant(value=columns.coefficient_role_stride),
            ),
        )
    )
    for step_index, step in enumerate(steps):
        output = _emit_output_effects(
            step,
            step_index,
            coordinates,
            row_low=row_low,
            row_high=row_high,
            output_low_name=output_low_name,
            output_high_name=output_high_name,
            lane_name=lane_name,
            prefix=prefix,
            reserved_names=reserved_names,
        )
        state = _emit_state_effects(
            step,
            step_index,
            plan,
            coordinates,
            helper_alias=helper_alias,
            buffers=buffers,
            lane_name=lane_name,
            row_base_name=row_base_name,
            history_name=history_name,
            prefix=prefix,
            reserved_names=reserved_names,
        )
        if output is None or state is None:
            return None
        body.extend(
            (*output, *state) if step.output_before_state else (*state, *output)
        )
    return (
        ast.If(
            test=ast.Compare(
                left=_name(warp_name),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value=plan.row_warps)],
            ),
            body=body,
            orelse=[],
        ),
    )


def _compose_direct_affine_replacement(
    replay: DirectAffineReplay,
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    proof: DirectAffineReplayProof,
    templates: DirectAffineResolvedTemplates,
    *,
    name_prefix: str = "_helion_direct_affine",
) -> DirectAffineEmission | None:
    """Build a complete detached DIRECT replacement.

    ``None`` is a normal fallback.  No input AST or owner table is changed on
    any failure path.  M16N8 and M16N16 are selected solely from ``plan``; the
    only step-capacity rule is the plan's algebraic ``2 * T <= N`` contract.
    """

    try:
        plan.validate()
        region = replay.candidate.region
        entry_state = templates.entry_state
        steps = templates.steps
        async_entry_state = templates.async_entry_state
        if (
            not _result_snapshots_are_intact(replay)
            or proof.source_replay is not replay
            or region.step_count != plan.step_count
            or region.row_extent != plan.row_extent
            or region.feature_extent != plan.feature_extent
            or region.storage_dtype is not plan.storage_dtype
            or len(steps) != plan.step_count
            or (
                plan.state_ingress is DirectAffineStateIngress.ASYNC
                and async_entry_state is None
            )
            or not proof.supports(plan)
            or coordinates.row.extent != plan.row_extent
            or coordinates.feature.extent != plan.feature_extent
            or resolve_direct_affine_coordinates(
                plan, row=coordinates.row, feature=coordinates.feature
            )
            != coordinates
            or not name_prefix.isidentifier()
            or keyword.iskeyword(name_prefix)
        ):
            return None
        existing_names = _names_in_program(replay, templates)
        existing_names.update(replay.reserved_names)
        if any(name.startswith(name_prefix) for name in existing_names):
            return None
        helper_alias = f"{name_prefix}_mma"
        smem_name = f"{name_prefix}_smem"
        lane_name = f"{name_prefix}_lane"
        warp_name = f"{name_prefix}_warp"
        buffers, smem_bytes = _shared_layout(plan, name_prefix)
        buffer_by_role = {buffer.role: buffer for buffer in buffers}
        setup = [
            *_buffer_setup(buffers, smem_name=smem_name, smem_bytes=smem_bytes),
            ast.Assign(
                targets=[_name(lane_name, ast.Store())],
                value=_cast_i32(
                    ast.Subscript(
                        value=_call(_qualified("cute.arch.thread_idx")),
                        slice=ast.Constant(value=0),
                        ctx=ast.Load(),
                    )
                ),
            ),
            ast.Assign(
                targets=[_name(warp_name, ast.Store())],
                value=_call(
                    _qualified("cute.arch.make_warp_uniform"),
                    _cast_i32(
                        ast.Subscript(
                            value=_call(_qualified("cute.arch.thread_idx")),
                            slice=ast.Constant(value=1),
                            ctx=ast.Load(),
                        )
                    ),
                ),
            ),
        ]
        async_state_phase = (
            _emit_async_state_materialization(
                async_entry_state,
                plan,
                coordinates,
                helper_alias=helper_alias,
                buffers=buffer_by_role,
                lane_name=lane_name,
                warp_name=warp_name,
                prefix=name_prefix,
                reserved_names=existing_names,
            )
            if plan.state_ingress is DirectAffineStateIngress.ASYNC
            and async_entry_state is not None
            else None
        )
        sync_state_phase = (
            _emit_state_materialization(
                entry_state,
                plan,
                coordinates,
                helper_alias=helper_alias,
                buffers=buffer_by_role,
                lane_name=lane_name,
                warp_name=warp_name,
                prefix=name_prefix,
                reserved_names=existing_names,
            )
            if plan.state_ingress is DirectAffineStateIngress.SYNC
            else None
        )
        coefficient_phase = _emit_coefficient_materialization(
            steps,
            plan,
            coordinates,
            helper_alias=helper_alias,
            buffers=buffer_by_role,
            lane_name=lane_name,
            warp_name=warp_name,
            prefix=name_prefix,
            reserved_names=existing_names,
        )
        consume_phase = _emit_consume(
            steps,
            plan,
            coordinates,
            helper_alias=helper_alias,
            buffers=buffer_by_role,
            lane_name=lane_name,
            warp_name=warp_name,
            prefix=name_prefix,
            reserved_names=existing_names,
        )
        if (
            coefficient_phase is None
            or consume_phase is None
            or (async_state_phase is None and sync_state_phase is None)
        ):
            return None
        if async_state_phase is not None:
            if len(coefficient_phase) < 3:
                return None
            coefficient_stage = coefficient_phase[:-2]
            coefficient_compute = (coefficient_phase[-2],)
            phases = (
                (
                    *coefficient_stage,
                    *async_state_phase.begin,
                    *coefficient_compute,
                    *async_state_phase.finish,
                )
                if plan.phase_order is DirectAffinePhaseOrder.STATE_FIRST
                else (
                    *coefficient_stage,
                    *coefficient_compute,
                    *async_state_phase.begin,
                    *async_state_phase.finish,
                )
            )
        else:
            assert sync_state_phase is not None
            phases = (
                (*sync_state_phase, *coefficient_phase)
                if plan.phase_order is DirectAffinePhaseOrder.STATE_FIRST
                else (*coefficient_phase, *sync_state_phase)
            )
        replacement = cast(
            "tuple[ast.stmt, ...]",
            tuple(
                ast.fix_missing_locations(statement)
                for statement in (*setup, *phases, *consume_phase)
            ),
        )
        if _ast_identity_set(replacement).intersection(
            _ast_identity_set(replay.source_statements)
        ):
            return None
        module_import = ast.fix_missing_locations(
            ast.Import(
                names=[ast.alias(name=_HELPER_MODULE, asname=helper_alias)],
            )
        )
        return DirectAffineEmission(
            module_statements=(module_import,),
            replacement_statements=replacement,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def compose_direct_affine_replacement(
    replay: DirectAffineReplay,
    plan: DirectAffinePlan,
    coordinates: DirectAffineCoordinates,
    proof: DirectAffineReplayProof,
    *,
    name_prefix: str = "_helion_direct_affine",
) -> DirectAffineEmission | None:
    """Resolve, validate, and compose a detached DIRECT replacement atomically."""

    source_snapshot = tuple(_dump(statement) for statement in replay.source_statements)
    try:
        templates = resolve_direct_affine_templates(replay, coordinates)
        if templates is None:
            return None
        templates = _validate_direct_affine_templates(
            templates,
            replay,
            plan,
            coordinates,
            proof,
        )
        if templates is None:
            return None
        return _compose_direct_affine_replacement(
            replay,
            plan,
            coordinates,
            proof,
            templates,
            name_prefix=name_prefix,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    finally:
        assert tuple(_dump(statement) for statement in replay.source_statements) == (
            source_snapshot
        )
