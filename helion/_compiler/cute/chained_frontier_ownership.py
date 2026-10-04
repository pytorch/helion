"""Bounded ownership selection for already admitted native frontier groups.

This policy changes neither expression eligibility nor native image placement.
The caller supplies its final, late-admitted native bindings and records
activation only after the original group emitter succeeds. Layout strings are
checked against the existing native policy; scalar-target names are not proof
of a transposed native image (they can also describe unrelated XOR layouts).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from ... import exc
from .chained_frontier_groups import plan_frontier_group
from .chained_prepared_groups import _member_layout
from .chained_prepared_groups import _native_row_subview
from .chained_prepared_groups import _valid_frame
from .chained_prepared_operands import _matches_contraction
from .chained_tcgen05 import _layout
from .chained_tcgen_stage import stage_geometry
from .chained_vector_ownership import plan_vector_ownership

if TYPE_CHECKING:
    from .chained_frontier_groups import FrontierGroup
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand
    from .chained_tcgen_stage import StageGeometry
    from .chained_vector_ownership import VectorOwnership


def _typed_buffer(frame: PreparationFrame, buffer: PreparationBuffer) -> bool:
    node = buffer.node
    if (
        node is None
        or buffer.kind != "frontier"
        or buffer not in frame.buffers
        or len(buffer.shape) != 2
        or buffer.dtype not in (torch.bfloat16, torch.float16)
        or any(type(extent) is not int or extent <= 0 for extent in buffer.shape)
    ):
        return False
    value = node.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == buffer.dtype
        and value.ndim == 2
        and all(
            type(old) is not int or old == new
            for old, new in zip(value.shape, buffer.shape, strict=True)
        )
        and node.graph is frame.cut.region.graph
    )


def _operand_matches(
    frame: PreparationFrame,
    buffer: PreparationBuffer,
    stage: int,
    geometry: StageGeometry,
    operand_index: int,
    modes: tuple[int, int],
) -> bool:
    specs = frame.cut.region.contractions
    if (
        type(stage) is not int
        or not 0 <= stage < len(specs)
        or not _typed_buffer(frame, buffer)
        or stage_geometry(geometry.logical) is None
    ):
        return False
    spec = specs[stage]
    index, coordinates = geometry.operand("b", "row", "k")
    expected = ("row", "k") if modes == (0, 1) else ("k", "row")
    shape = geometry.physical[1:]
    return (
        modes in ((0, 1), (1, 0))
        and operand_index == index
        and coordinates == expected
        and buffer.node is (spec.lhs, spec.rhs)[index]
        and _matches_contraction(spec)
        and spec.operand_dtypes == (buffer.dtype, buffer.dtype)
        and all(
            type(old) is not int or old == new
            for old, new in zip(spec.shape, geometry.logical, strict=True)
        )
        and buffer.shape == (shape if modes == (0, 1) else shape[::-1])
    )


def _singleton_matches(frame: PreparationFrame, operand: PreparedOperand) -> bool:
    buffer = operand.buffer
    region = next(
        (item for item in frame.layout.regions if item.name == buffer.name), None
    )
    dtype = "cutlass.BFloat16" if buffer.dtype == torch.bfloat16 else "cutlass.Float16"
    return (
        operand.role == "b"
        and _operand_matches(
            frame,
            buffer,
            operand.stage,
            operand.geometry,
            operand.operand_index,
            (0, 1),
        )
        and operand.physical_shape == buffer.shape
        and operand.region == region
        and region is not None
        and region.byte_size == math.prod(buffer.shape) * buffer.dtype.itemsize
        and operand.native_layout
        == tuple(_layout(buffer.name, operand.physical_shape, 1, dtype))
    )


def _group_matches(frame: PreparationFrame, binding: PreparedGroupBinding) -> bool:
    candidate = binding.candidate
    group = candidate.group
    if (
        len(group.stages) < 2
        or len(group.stages) != len(group.geometries)
        or len(group.stages) != len(candidate.members)
        or len(set(group.stages)) != len(group.stages)
        or candidate.dtype not in (torch.bfloat16, torch.float16)
        or candidate.physical_shape != group.physical[1:]
        or candidate.byte_size
        != math.prod(candidate.physical_shape) * candidate.dtype.itemsize
        or binding.byte_offset < 0
        or binding.byte_offset % 128
        or binding.byte_offset + candidate.byte_size > frame.layout.allocated_bytes
        or stage_geometry(group.physical) is None
    ):
        return False
    dtype = (
        "cutlass.BFloat16" if candidate.dtype == torch.bfloat16 else "cutlass.Float16"
    )
    if candidate.native_layout != tuple(
        _layout(candidate.name, candidate.physical_shape, 1, dtype)
    ):
        return False
    regions = {region.name: region for region in frame.layout.regions}
    if len({member.buffer.name for member in candidate.members}) != len(
        candidate.members
    ):
        return False
    lifetimes = []
    for member, stage, geometry, offset in zip(
        candidate.members, group.stages, group.geometries, group.offsets, strict=True
    ):
        region = regions.get(member.buffer.name)
        if (
            member.stage != stage
            or member.geometry != geometry
            or geometry.physical[::2] != group.physical[::2]
            or member.row_offset != offset
            or member.physical_shape != geometry.physical[1:]
            or member.byte_offset
            != offset * candidate.physical_shape[1] * candidate.dtype.itemsize
            or member.buffer.dtype != candidate.dtype
            or not _operand_matches(
                frame,
                member.buffer,
                member.stage,
                member.geometry,
                member.operand_index,
                member.logical_modes,
            )
            or not _native_row_subview(
                candidate.physical_shape, member.physical_shape, member.row_offset
            )
            or member.native_layout
            != _member_layout(
                member.buffer.name, member.physical_shape, dtype, member.logical_modes
            )
            or region is None
            or region.byte_offset != binding.byte_offset + member.byte_offset
            or region.byte_size
            != math.prod(member.physical_shape) * candidate.dtype.itemsize
        ):
            return False
        lifetimes.append((region.live_from, region.live_until))
    return (candidate.live_from, candidate.live_until) == (
        min(begin for begin, _ in lifetimes),
        max(end for _, end in lifetimes),
    )


@dataclass
class FrontierOwnership:
    """Per-codegen selection, with successful-emission activation caller-owned."""

    tile_columns: int
    activated: bool = False

    def __post_init__(self) -> None:
        if type(self.tile_columns) is not int or self.tile_columns < 0:
            raise ValueError("frontier tile columns must be a nonnegative integer")

    def select(
        self,
        frame: PreparationFrame,
        group: FrontierGroup,
        prepared_operands: tuple[PreparedOperand, ...],
        prepared_groups: tuple[PreparedGroupBinding, ...],
        threads: int,
    ) -> VectorOwnership | None:
        """Retile only exact mixed-orientation, fully native half frontiers.

        The supplied binding tuples must be the caller's final accepted sets,
        after all late domain checks and frame repacking. They are not early
        graph-discovery candidates. This function only rechecks their physical
        and typed provenance; it cannot grant missing expression admission.
        """
        if not self.tile_columns:
            return None
        if (
            type(threads) is not int
            or threads < 32
            or threads & (threads - 1)
            or not _valid_frame(frame)
            or frame.cut.region.nodes != tuple(frame.cut.region.graph.nodes)
            or any(not _typed_buffer(frame, buffer) for buffer in group.buffers)
            or plan_frontier_group(frame, group.first_event) != group
        ):
            return None
        orientations = set()
        for buffer in group.buffers:
            modes = []
            for operand in prepared_operands:
                if operand.buffer.name == buffer.name:
                    if operand.buffer != buffer or not _singleton_matches(
                        frame, operand
                    ):
                        return None
                    modes.append((0, 1))
            for binding in prepared_groups:
                for member in binding.candidate.members:
                    if member.buffer.name == buffer.name:
                        if member.buffer != buffer or not _group_matches(
                            frame, binding
                        ):
                            return None
                        modes.append(member.logical_modes)
            if not modes or len(set(modes)) != 1:
                return None
            orientations.add(modes[0])
        if orientations != {(0, 1), (1, 0)}:
            return None
        height, width = group.buffers[0].shape
        ownership = plan_vector_ownership(
            (height, width), threads, tile_columns=self.tile_columns
        )
        return ownership if ownership is not None and ownership.changed else None

    def validate(self) -> None:
        if self.tile_columns and not self.activated:
            raise exc.BackendUnsupported(
                "cute",
                "frontier tile columns require a successfully emitted mixed-native "
                "frontier group with changed ownership",
            )
