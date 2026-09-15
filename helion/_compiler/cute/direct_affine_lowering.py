# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed adapter from ordinary CuTe codegen to direct affine replay.

The semantic matcher intentionally runs before layout selection, while the
direct emitter needs facts which only exist after ordinary code generation.
This module is that narrow bridge.  It reads the final generated AST and
cache-key-backed runtime facts, builds a detached replacement, and leaves both
the source body and codegen ownership tables unchanged.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._subclasses import FakeTensor

from ..ast_extension import ExtendedAST
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ..device_function import TensorDescriptorArg
from ..indexing_strategy import subscript_tile_info
from .cutedsl_compat import cp_async_supported
from .direct_affine_plan import STATE_VECTOR_BYTES
from .direct_affine_plan import DirectAffinePlan
from .direct_affine_plan import DirectAffineSchedule
from .direct_affine_plan import DirectAffineStateAccessProof
from .direct_affine_plan import DirectAffineStateIngress
from .direct_affine_plan import decode_direct_affine_schedule
from .direct_affine_plan import resolve_direct_affine_plan
from .direct_affine_replay import DirectAffineCoordinates
from .direct_affine_replay import DirectAffineEmission
from .direct_affine_replay import DirectAffineOrdinaryAxis
from .direct_affine_replay import DirectAffineReplay
from .direct_affine_replay import DirectAffineReplayProof
from .direct_affine_replay import compose_direct_affine_replacement
from .direct_affine_replay import resolve_direct_affine_coordinates
from .direct_affine_replay import resolve_direct_affine_replay

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceGridState
    from .direct_affine_candidate import DirectAffineCandidate
    from .short_affine_scan import ShortAffineScanAccess


_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_A = TypeVar("_A", bound=ast.AST)


@dataclasses.dataclass(frozen=True)
class DirectAffineResolvedLowering:
    """Complete, detached late-lowering result ready for integration."""

    replay: DirectAffineReplay = dataclasses.field(repr=False)
    plan: DirectAffinePlan
    coordinates: DirectAffineCoordinates = dataclasses.field(repr=False)
    proof: DirectAffineReplayProof
    emission: DirectAffineEmission = dataclasses.field(repr=False)
    source_cta_shape: tuple[int, int, int]
    row_block_id: int
    feature_block_id: int


@dataclasses.dataclass(frozen=True)
class _AxisSources:
    row: torch.fx.Node = dataclasses.field(repr=False)
    feature: torch.fx.Node = dataclasses.field(repr=False)
    row_block_id: int
    feature_block_id: int
    feature_is_synthetic: bool = False
    feature_index_name: str | None = None


@dataclasses.dataclass(frozen=True)
class _TensorMetadata:
    fake: torch.Tensor = dataclasses.field(repr=False)
    runtime: torch.Tensor = dataclasses.field(repr=False)
    argument_name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]


def _static_size(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt) and isinstance(value._sympy_(), sympy.Integer):
        return int(value._sympy_())
    return None


def _qualified_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _clone_ast(node: _A) -> _A:
    """Clone generated AST without invoking ``ExtendedAST.__init__`` directly."""

    def clone(value: object) -> object:
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, ast.AST):
            fields = {field: clone(getattr(value, field)) for field in value._fields}
            result = (
                value.copy(**fields)
                if isinstance(value, ExtendedAST)
                else ast.copy_location(type(value)(**fields), value)
            )
            for name, metadata in vars(value).items():
                if name not in value._fields:
                    setattr(result, name, metadata)
            return result
        return value

    result = clone(node)
    assert isinstance(result, ast.AST)
    return cast("_A", result)


def _depends_on_node(value: object, target: torch.fx.Node) -> bool:
    seen: set[torch.fx.Node] = set()

    def visit(item: object) -> bool:
        if item is target:
            return True
        if isinstance(item, torch.fx.Node):
            if item in seen:
                return False
            seen.add(item)
            return any(visit(argument) for argument in item.all_input_nodes)
        if isinstance(item, (tuple, list)):
            return any(visit(child) for child in item)
        if isinstance(item, dict):
            return any(visit(child) for child in item.values())
        if isinstance(item, slice):
            return any(visit(child) for child in (item.start, item.stop, item.step))
        return False

    return visit(value)


def _access_rank_is_exact(access: ShortAffineScanAccess) -> bool:
    value = access.base.meta.get("val")
    return isinstance(value, torch.Tensor) and len(access.indices) == value.ndim


def _access_uses_axes(
    access: ShortAffineScanAccess,
    row: torch.fx.Node,
    feature: torch.fx.Node,
    *,
    feature_required: bool,
) -> bool:
    if (
        not _access_rank_is_exact(access)
        or access.mask is not None
        or access.other is not None
        or access.kwargs
    ):
        return False
    if feature_required:
        if len(access.indices) < 2 or access.indices[-2:] != (row, feature):
            return False
        leading = access.indices[:-2]
    else:
        if not access.indices or access.indices[-1] is not row:
            return False
        leading = access.indices[:-1]
        if _depends_on_node(access.indices, feature):
            return False
    return not _depends_on_node(leading, row) and not _depends_on_node(leading, feature)


def _block_index_name(
    codegen: GenerateAST,
    grid: DeviceGridState,
    block_id: int,
) -> str | None:
    """Return the strategy which owns ``block_id``'s exact coordinate name."""

    try:
        strategy = codegen.device_function.tile_strategy.block_id_to_strategy.get_any(
            block_id
        )
        if strategy is None:
            raise KeyError(block_id)
        name = strategy.index_var(block_id)
    except (AttributeError, KeyError, TypeError, ValueError):
        try:
            name = grid.strategy.index_var(block_id)
        except (KeyError, TypeError, ValueError):
            return None
    return name if isinstance(name, str) and name.isidentifier() else None


def _resolve_axis_sources(
    candidate: DirectAffineCandidate,
    codegen: GenerateAST,
    grid: DeviceGridState,
) -> _AxisSources | None:
    region = candidate.region
    entry = region.entry_access
    if len(entry.indices) < 2:
        return None
    row, feature = entry.indices[-2:]
    if (
        not isinstance(row, torch.fx.Node)
        or not isinstance(feature, torch.fx.Node)
        or row is feature
    ):
        return None
    env = CompileEnvironment.current()
    row_info = subscript_tile_info(env, row)
    feature_info = subscript_tile_info(env, feature)
    if row_info is None or not env.known_equal(row_info.offset, 0):
        return None
    graph = region.entry_load.graph
    row_block_id = env.resolve_codegen_block_id(row_info.block_id, codegen, graph)
    feature_is_synthetic = feature_info is None
    feature_index_name: str | None = None
    if feature_info is not None:
        if not env.known_equal(feature_info.offset, 0):
            return None
        feature_block_id = env.resolve_codegen_block_id(
            feature_info.block_id, codegen, graph
        )
    else:
        # A free ``hl.arange`` has no TileWithOffsetInfo.  Bind its recorded
        # result to the unique block-specific strategy coordinate.  Synthetic
        # blocks need not belong to ``grid.strategy`` itself; the device
        # function's block-to-strategy map is the authoritative owner.
        found, generated = codegen.codegen_result_for_node(feature)
        if not found or not isinstance(generated, ast.Name):
            return None
        matching_blocks = [
            block_id
            for block_id in grid.block_thread_axes
            if _block_index_name(codegen, grid, block_id) == generated.id
        ]
        if len(matching_blocks) != 1:
            return None
        (feature_block_id,) = matching_blocks
        feature_index_name = generated.id
    if row_block_id == feature_block_id:
        return None
    try:
        row_block_size = codegen.device_function.resolved_block_size(row_block_id)
        feature_block_size = codegen.device_function.resolved_block_size(
            feature_block_id
        )
        if row_block_size is None or feature_block_size is None:
            return None
        row_block_size = env.size_hint(row_block_size)
        feature_block_size = env.size_hint(feature_block_size)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    if (
        row_block_size != region.row_extent
        or feature_block_size != region.feature_extent
        or not _access_uses_axes(entry, row, feature, feature_required=True)
        or any(
            not _access_uses_axes(
                step.state_access,
                row,
                feature,
                feature_required=True,
            )
            or not _access_uses_axes(
                step.output_access,
                row,
                feature,
                feature_required=False,
            )
            for step in region.steps
        )
    ):
        return None
    return _AxisSources(
        row=row,
        feature=feature,
        row_block_id=row_block_id,
        feature_block_id=feature_block_id,
        feature_is_synthetic=feature_is_synthetic,
        feature_index_name=feature_index_name,
    )


def _simple_assignments(
    statements: Iterable[ast.AST], name: str
) -> tuple[ast.Assign, ...]:
    return tuple(
        statement
        for statement in statements
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    )


class _ReplaceExactExpression(ast.NodeTransformer):
    def __init__(self, source: ast.expr, replacement: ast.expr) -> None:
        self.source = _dump(source)
        self.replacement = replacement
        self.count = 0

    def visit(self, node: ast.AST) -> ast.AST:  # type: ignore[override]
        if (
            isinstance(node, ast.expr)
            and isinstance(getattr(node, "ctx", ast.Load()), ast.Load)
            and _dump(node) == self.source
        ):
            self.count += 1
            return _clone_ast(self.replacement)
        return super().visit(node)


class _FoldCoordinateZeros(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = cast("ast.BinOp", self.generic_visit(node))
        if isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                return node.left
        if (
            isinstance(node.op, ast.Sub)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 0
        ):
            return node.left
        return node


def _axis_expression(
    source_node: torch.fx.Node,
    block_id: int,
    extent: int,
    codegen: GenerateAST,
    grid: DeviceGridState,
    root_body: Sequence[ast.AST],
    *,
    synthetic: bool,
    offset_required: bool,
    coordinate_name: str | None = None,
) -> DirectAffineOrdinaryAxis | None:
    found, generated = codegen.codegen_result_for_node(source_node)
    if not found:
        return None
    index_name = coordinate_name or _block_index_name(codegen, grid, block_id)
    if index_name is None:
        return None
    # A TileWithOffsetInfo result can be the specialized block-size symbol,
    # not the generated coordinate used by memory pointers.  Conversely, a
    # free arange has no tile metadata, so require its recorded result to be
    # exactly the inferred synthetic strategy coordinate.
    if synthetic and (
        not isinstance(generated, ast.Name) or generated.id != index_name
    ):
        return None
    source = ast.Name(id=index_name, ctx=ast.Load())
    visible = (
        *(grid.hoist_parent_statements or ()),
        *grid.outer_prefix,
        *grid.lane_setup_statements,
        *root_body,
    )
    assignments = _simple_assignments(visible, index_name)
    if len(assignments) != 1:
        return None
    if synthetic:
        local = _clone_ast(assignments[0].value)
        tile_offset: ast.expr = ast.Constant(value=0)
    else:
        try:
            offset_name = grid.strategy.offset_var(block_id)
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(offset_name, str) or not offset_name.isidentifier():
            return None
        offset = ast.Name(id=offset_name, ctx=ast.Load())
        replacement = _ReplaceExactExpression(offset, ast.Constant(value=0))
        local = replacement.visit(_clone_ast(assignments[0].value))
        if (
            not isinstance(local, ast.expr)
            or replacement.count > 1
            or (offset_required and replacement.count != 1)
        ):
            return None
        local = _FoldCoordinateZeros().visit(local)
        if not isinstance(local, ast.expr):
            return None
        tile_offset = offset if replacement.count == 1 else ast.Constant(value=0)

    thread_axis = grid.block_thread_axes.get(block_id)
    if not isinstance(thread_axis, int) or not 0 <= thread_axis < 3:
        return None
    thread_extent = grid.thread_axis_sizes.get(thread_axis, 1)
    if not isinstance(thread_extent, int) or thread_extent <= 0:
        return None
    allowed_lane_blocks = {frozenset((block_id,))}
    if synthetic:
        allowed_lane_blocks.add(frozenset((-1,)))
    read_names = {
        node.id
        for node in ast.walk(local)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    lane_matches = tuple(
        (name, lane_extent)
        for name, lane_extent in grid.lane_loops
        if grid.lane_loop_block_ids.get(name) in allowed_lane_blocks
        and name in read_names
    )
    if len(lane_matches) > 1:
        return None
    if lane_matches:
        lane_name, lane_extent = lane_matches[0]
    else:
        lane_name, lane_extent = None, 1
    if (
        not isinstance(lane_extent, int)
        or lane_extent <= 0
        or thread_extent * lane_extent != extent
    ):
        return None
    return DirectAffineOrdinaryAxis(
        source=source,
        tile_offset=tile_offset,
        local_expression=local,
        extent=extent,
        thread_axis=thread_axis,
        thread_extent=thread_extent,
        lane_name=lane_name,
        lane_extent=lane_extent,
    )


def _source_cta_shape(
    codegen: GenerateAST,
    grid: DeviceGridState,
) -> tuple[int, int, int] | None:
    sizes = [1, 1, 1]
    for mapping in (grid.thread_axis_sizes, codegen.cute_synthetic_arange_axis_sizes):
        for axis, extent in mapping.items():
            if (
                not isinstance(axis, int)
                or not 0 <= axis < 3
                or not isinstance(extent, int)
                or extent <= 0
            ):
                return None
            sizes[axis] = max(sizes[axis], extent)
    return cast("tuple[int, int, int]", tuple(sizes))


def _resolve_coordinates(
    candidate: DirectAffineCandidate,
    sources: _AxisSources,
    plan: DirectAffinePlan,
    codegen: GenerateAST,
    grid: DeviceGridState,
    root_body: Sequence[ast.AST],
) -> tuple[DirectAffineCoordinates, tuple[int, int, int]] | None:
    lane_owner_sets = set(grid.lane_loop_block_ids.values())
    accepted_lane_owner_sets = {
        frozenset(
            (
                frozenset((sources.row_block_id,)),
                frozenset((sources.feature_block_id,)),
            )
        )
    }
    if sources.feature_is_synthetic:
        accepted_lane_owner_sets.add(
            frozenset(
                (
                    frozenset((sources.row_block_id,)),
                    frozenset((-1,)),
                )
            )
        )
    if (
        grid.vec_lane_wrappers
        or grid.outer_suffix
        or set(grid.block_thread_axes)
        != {sources.row_block_id, sources.feature_block_id}
        or grid.block_thread_axes.get(sources.feature_block_id) != 0
        or grid.block_thread_axes.get(sources.row_block_id) != 1
        or frozenset(lane_owner_sets) not in accepted_lane_owner_sets
    ):
        return None
    source_cta_shape = _source_cta_shape(codegen, grid)
    if source_cta_shape is None or source_cta_shape != (32, plan.row_warps, 1):
        return None
    row = _axis_expression(
        sources.row,
        sources.row_block_id,
        candidate.region.row_extent,
        codegen,
        grid,
        root_body,
        synthetic=False,
        offset_required=True,
    )
    feature = _axis_expression(
        sources.feature,
        sources.feature_block_id,
        candidate.region.feature_extent,
        codegen,
        grid,
        root_body,
        synthetic=sources.feature_is_synthetic,
        offset_required=False,
        coordinate_name=sources.feature_index_name,
    )
    if row is None or feature is None:
        return None
    coordinates = resolve_direct_affine_coordinates(plan, row=row, feature=feature)
    if coordinates is None:
        return None
    return coordinates, source_cta_shape


def _tensor_argument_name(
    device_function: DeviceFunction, fake: torch.Tensor
) -> str | None:
    matches = tuple(
        argument.name
        for argument in device_function.arguments
        if isinstance(argument, TensorArg)
        and not isinstance(argument, TensorDescriptorArg)
        and argument.fake_value is fake
    )
    return matches[0] if len(matches) == 1 else None


def _tensor_metadata(
    base: torch.fx.Node,
    codegen: GenerateAST,
    proven_sizes: dict[tuple[str, int], tuple[str, int]],
    proven_strides: dict[tuple[str, int], int],
    *,
    require_layout: bool,
) -> _TensorMetadata | None:
    fake = base.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    env = CompileEnvironment.current()
    runtime = env.runtime_value_for_tensor(fake)
    argument_name = _tensor_argument_name(codegen.device_function, fake)
    if (
        argument_name is None
        or env.tensor_input_source(fake) is None
        or not isinstance(runtime, torch.Tensor)
        or isinstance(runtime, FakeTensor)
        or runtime.ndim != fake.ndim
        or runtime.dtype is not fake.dtype
        or runtime.device != fake.device
    ):
        return None
    shape = tuple(int(value) for value in runtime.shape)
    strides = tuple(int(value) for value in runtime.stride())
    if require_layout and (
        not shape
        or any(value <= 0 for value in shape)
        or any(value <= 0 for value in strides)
        or any(
            proven_strides.get((argument_name, dim)) != stride
            for dim, stride in enumerate(strides)
        )
    ):
        return None
    # ``input_tensor_metadata`` is a cache specialization over the complete
    # runtime shape.  Individual ``*_size_*`` symbols are needed only when a
    # generated guard must be replayed; requiring one for every leading shape
    # dimension would reject otherwise exact, cache-keyed layouts.
    return _TensorMetadata(
        fake=fake,
        runtime=runtime,
        argument_name=argument_name,
        shape=shape,
        strides=strides,
    )


def _offsets_fit_i32(metadata: _TensorMetadata) -> bool:
    return (
        len(metadata.shape) == len(metadata.strides)
        and sum(
            (size - 1) * stride
            for size, stride in zip(
                metadata.shape,
                metadata.strides,
                strict=True,
            )
        )
        <= _INT32_MAX
    )


def _proven_extent_expression(
    metadata: _TensorMetadata,
    proven_sizes: dict[tuple[str, int], tuple[str, int]],
    dim: int,
) -> ast.expr | None:
    dim %= len(metadata.shape)
    fake_size = metadata.fake.shape[dim]
    proof = proven_sizes.get((metadata.argument_name, dim))
    if proof is not None and proof[1] == metadata.shape[dim]:
        try:
            expression = ast.parse(proof[0], mode="eval").body
        except SyntaxError:
            return None
        return expression if isinstance(expression, ast.expr) else None
    return (
        ast.Constant(value=metadata.shape[dim])
        if _static_size(fake_size) == metadata.shape[dim]
        else None
    )


def _state_slot_fits_i32(slot: object) -> bool:
    if type(slot) is int:
        return _INT32_MIN <= slot <= _INT32_MAX
    if not isinstance(slot, torch.fx.Node):
        return False
    value = slot.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        return False
    if value.dtype is torch.int32:
        return True
    return (
        value.dtype is torch.int64
        and slot.op == "call_function"
        and slot.target is torch.ops.prims.convert_element_type.default
        and len(slot.args) == 2
        and slot.args[1] is torch.int64
        and not slot.kwargs
        and isinstance(slot.args[0], torch.fx.Node)
        and isinstance(slot.args[0].meta.get("val"), torch.Tensor)
        and slot.args[0].meta["val"].ndim == 0
        and slot.args[0].meta["val"].dtype is torch.int32
    )


def _resolve_memory_proof(
    candidate: DirectAffineCandidate,
    codegen: GenerateAST,
    state_ingress: DirectAffineStateIngress,
) -> tuple[DirectAffineStateAccessProof, DirectAffineReplayProof] | None:
    env = CompileEnvironment.current()
    region = candidate.region
    if "input_tensor_metadata" not in env.compiler_fact_specialization_facts:
        return None
    try:
        proven_sizes = codegen.device_function.proven_tensor_size_values()
        proven_strides = codegen.device_function.proven_tensor_stride_values()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    bases = tuple(dict.fromkeys((*region.read_bases, *region.write_bases)))
    written = set(region.write_bases)
    metadata = {
        base: _tensor_metadata(
            base,
            codegen,
            proven_sizes,
            proven_strides,
            require_layout=base in written,
        )
        for base in bases
    }
    if any(value is None for value in metadata.values()):
        return None
    resolved = cast("dict[torch.fx.Node, _TensorMetadata]", metadata)
    state = resolved.get(region.state_base)
    if state is None or region.state_base in region.output_bases:
        return None
    if set(region.read_bases).intersection(region.write_bases) - {region.state_base}:
        return None
    if (
        len(state.shape) < 3
        or state.fake.dtype is not region.storage_dtype
        or state.shape[-2] < region.row_extent
        or state.shape[-2] % region.row_extent
        or state.shape[-1] != region.feature_extent
        or state.strides[-1] != 1
        or region.feature_extent % (STATE_VECTOR_BYTES // state.fake.element_size())
        or any(
            stride * state.fake.element_size() % STATE_VECTOR_BYTES
            for stride in state.strides[:-1]
        )
        or not _offsets_fit_i32(state)
        or not all(
            _state_slot_fits_i32(access.indices[0])
            for access in (
                region.entry_access,
                *(step.state_access for step in region.steps),
            )
        )
    ):
        return None
    state_feature_extent_expression = _proven_extent_expression(
        state,
        proven_sizes,
        -1,
    )
    state_row_extent_expression = _proven_extent_expression(
        state,
        proven_sizes,
        -2,
    )
    if state_feature_extent_expression is None:
        return None

    output_row_extents: list[int] = []
    output_argument_names: list[str] = []
    output_ranks: list[int] = []
    for step in region.steps:
        output = resolved.get(step.output_base)
        if output is None or not output.shape:
            return None
        output_row_extents.append(output.shape[-1])
        output_argument_names.append(output.argument_name)
        output_ranks.append(len(output.shape))
    if len(output_row_extents) != region.step_count or any(
        extent < region.row_extent or extent % region.row_extent
        for extent in output_row_extents
    ):
        return None

    from .memory_ops import runtime_tensor_has_specialized_alignment
    from .memory_ops import runtime_tensors_are_proven_disjoint

    if not runtime_tensor_has_specialized_alignment(
        env, state.fake, STATE_VECTOR_BYTES
    ):
        return None
    for left, right in itertools.combinations(bases, 2):
        if left not in written and right not in written:
            continue
        if not runtime_tensors_are_proven_disjoint(
            env,
            resolved[left].fake,
            resolved[right].fake,
        ):
            return None

    async_supported = cp_async_supported(env.config_spec.target_device_capability)
    state_access = DirectAffineStateAccessProof(
        source_vector_alignment_bytes=STATE_VECTOR_BYTES,
        destination_vector_alignment_bytes=128,
        source_feature_stride_one=True,
        full_vector_coverage=True,
        cp_async_supported=async_supported,
    )
    if (
        state_ingress is DirectAffineStateIngress.ASYNC
        and not state_access.supports_async_io()
    ):
        return None
    replay_proof = DirectAffineReplayProof(
        state_output_alias_free=True,
        state_feature_stride_one=True,
        state_vector_aligned=True,
        masks_preserved=True,
        row_tail_free=True,
        feature_tail_free=True,
        runtime_state_shape=cast("tuple[int, int]", state.shape[-2:]),
        runtime_output_row_extents=tuple(output_row_extents),
        state_feature_extent_expression=state_feature_extent_expression,
        state_row_extent_expression=state_row_extent_expression,
        state_argument_name=state.argument_name,
        state_rank=len(state.shape),
        output_argument_names=tuple(output_argument_names),
        output_ranks=tuple(output_ranks),
    )
    return state_access, replay_proof


def _stored_names(statements: Iterable[ast.AST]) -> set[str]:
    return {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _visible_grid_names(grid: DeviceGridState) -> set[str]:
    statements = (
        *(grid.hoist_parent_statements or ()),
        *grid.outer_prefix,
        *grid.lane_setup_statements,
    )
    names = {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    names.update(
        node.arg
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.arg)
    )
    return names


def _prefix_is_lane_independent(
    statements: Sequence[ast.AST],
    grid: DeviceGridState,
) -> bool:
    """Prove a retained pre-region prefix is safe for any added source warp."""

    from ..tile_strategy import _is_proven_relocatable_assignment

    lane_names = {name for name, _extent in grid.lane_loops}
    lane_names.update(_stored_names(grid.lane_setup_statements))
    for statement in statements:
        if not _is_proven_relocatable_assignment(statement, allow_load=True):
            return False
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in lane_names
            ):
                return False
            if not isinstance(node, ast.Call):
                continue
            name = _qualified_name(node.func)
            if name in {
                "cute.arch.lane_idx",
                "cute.arch.thread_idx",
                "cute.arch.warp_idx",
            }:
                return False
    return True


def _emission_has_old_schedule_names(
    emission: DirectAffineEmission,
    coordinates: DirectAffineCoordinates,
    sources: _AxisSources,
    codegen: GenerateAST,
    grid: DeviceGridState,
) -> bool:
    forbidden = {name for name, _extent in grid.lane_loops}
    forbidden.update(_stored_names(grid.lane_setup_statements))
    for wrapper in grid.vec_lane_wrappers.values():
        forbidden.update((wrapper.vec_lane_var, wrapper.base_index_var))
    for source in (coordinates.row.source, coordinates.feature.source):
        if isinstance(source, ast.Name):
            forbidden.add(source.id)
    for block_id in (sources.row_block_id, sources.feature_block_id):
        index_name = _block_index_name(codegen, grid, block_id)
        if index_name is None:
            return True
        forbidden.add(index_name)
    return any(
        (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in forbidden
        )
        or (
            isinstance(node, ast.Call)
            and _qualified_name(node.func) == "cute.arch.lane_idx"
        )
        for statement in emission.replacement_statements
        for node in ast.walk(statement)
    )


def _configured_plan(
    candidate: DirectAffineCandidate,
    schedule: DirectAffineSchedule,
    state_access: DirectAffineStateAccessProof,
) -> DirectAffinePlan | None:
    region = candidate.region
    return resolve_direct_affine_plan(
        step_count=region.step_count,
        row_extent=region.row_extent,
        feature_extent=region.feature_extent,
        storage_dtype=region.storage_dtype,
        mma=schedule.mma,
        coefficient_layout=schedule.coefficient_layout,
        state_ingress=schedule.state_ingress,
        phase_order=schedule.phase_order,
        state_access_proof=state_access,
    )


def resolve_direct_affine_lowering(
    candidate: DirectAffineCandidate,
    graph_info: GraphInfo,
    codegen: GenerateAST,
    grid: DeviceGridState,
    root_body: list[ast.AST],
    *,
    name_prefix: str = "_helion_direct_affine",
) -> DirectAffineResolvedLowering | None:
    """Resolve a direct lowering without mutating codegen or ``root_body``.

    The caller may commit ``result.emission.replacement_statements`` through
    :func:`direct_affine_replay.replace_direct_affine_replay` only after it has
    installed the returned CTA and module-import metadata.
    """

    if not CompileEnvironment.has_current():
        return None
    source_snapshot = ast.dump(
        ast.Module(body=cast("list[ast.stmt]", root_body), type_ignores=[]),
        include_attributes=True,
    )
    try:
        env = CompileEnvironment.current()
        config = codegen.device_function.config
        if (
            not env.settings.fast_math
            or config.pid_type != "flat"
            or grid.hoist_parent_statements is None
        ):
            return None
        try:
            schedule = decode_direct_affine_schedule(config.cute_affine_scan_schedule)
        except ValueError:
            return None
        if schedule is None:
            return None
        replay = resolve_direct_affine_replay(
            candidate,
            graph_info,
            codegen,
            root_body,
        )
        if (
            replay is None
            or not root_body
            or replay.source_span[1] != len(root_body) - 1
            or not _prefix_is_lane_independent(root_body[: replay.source_span[0]], grid)
            or not _prefix_is_lane_independent(grid.outer_prefix, grid)
            or not _prefix_is_lane_independent(
                grid.hoist_parent_statements,
                grid,
            )
        ):
            return None
        replay = dataclasses.replace(
            replay,
            reserved_names=(replay.reserved_names | _visible_grid_names(grid)),
        )
        sources = _resolve_axis_sources(candidate, codegen, grid)
        if sources is None:
            return None
        memory = _resolve_memory_proof(candidate, codegen, schedule.state_ingress)
        if memory is None:
            return None
        state_access, proof = memory
        plan = _configured_plan(candidate, schedule, state_access)
        if plan is None:
            return None
        coordinate_result = _resolve_coordinates(
            candidate,
            sources,
            plan,
            codegen,
            grid,
            root_body,
        )
        if coordinate_result is None:
            return None
        coordinates, source_cta_shape = coordinate_result
        proof = dataclasses.replace(
            proof,
            source_replay=replay,
            output_store_pre_wrap_proven=True,
        )
        emission = compose_direct_affine_replacement(
            replay,
            plan,
            coordinates,
            proof,
            name_prefix=name_prefix,
        )
        if emission is None or _emission_has_old_schedule_names(
            emission,
            coordinates,
            sources,
            codegen,
            grid,
        ):
            return None
        return DirectAffineResolvedLowering(
            replay=replay,
            plan=plan,
            coordinates=coordinates,
            proof=proof,
            emission=emission,
            source_cta_shape=source_cta_shape,
            row_block_id=sources.row_block_id,
            feature_block_id=sources.feature_block_id,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    finally:
        assert (
            ast.dump(
                ast.Module(body=cast("list[ast.stmt]", root_body), type_ignores=[]),
                include_attributes=True,
            )
            == source_snapshot
        )


__all__ = [
    "DirectAffineResolvedLowering",
    "resolve_direct_affine_lowering",
]
