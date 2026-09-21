# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure physical planning for a short direct affine scan.

The planner deliberately knows only the dimensions of an affine recurrence
and the physical choices exposed to the CuTe backend.  It does not inspect
tensor names, producer formulas, or application-specific arguments.  A later
lowering is responsible for matching the recurrence, proving memory effects,
and replaying its producer expressions.

``ORDINARY`` is an explicit fallback choice and therefore resolves to no
direct plan.  The two direct choices describe the column capacity of the
native BF16 projection image.  Selection is algebraic: every scan step needs
one prediction and one observation column.
"""

from __future__ import annotations

import dataclasses
import enum

import torch

MMA_M = 16
MMA_N = 8
WARP_SIZE = 32
MAX_CTA_WARPS = 8
SUPPORTED_FEATURE_EXTENT = 128
STATE_VECTOR_BYTES = 16
DIRECT_AFFINE_ORDINARY_SCHEDULE = "ordinary"
DIRECT_AFFINE_M16N8_V1_SCHEDULE = "direct_m16n8_v1"
DIRECT_AFFINE_M16N16_V1_SCHEDULE = "direct_m16n16_v1"
DIRECT_AFFINE_SCHEDULES = (
    DIRECT_AFFINE_ORDINARY_SCHEDULE,
    DIRECT_AFFINE_M16N8_V1_SCHEDULE,
    DIRECT_AFFINE_M16N16_V1_SCHEDULE,
)


class DirectAffineMma(enum.Enum):
    """Projection implementation selected for an affine scan."""

    ORDINARY = "ordinary"
    M16N8 = "m16n8"
    M16N16 = "m16n16"

    @property
    def column_extent(self) -> int | None:
        if self is DirectAffineMma.M16N8:
            return MMA_N
        if self is DirectAffineMma.M16N16:
            return 2 * MMA_N
        return None


class DirectAffineCoefficientLayout(enum.Enum):
    """Physical ordering of prediction and observation values."""

    INTERLEAVED = "interleaved"
    ROLE_MAJOR = "role_major"


class DirectAffineStateIngress(enum.Enum):
    """How the initial state reaches the shared-memory projection image."""

    SYNC = "sync"
    ASYNC = "async"


class DirectAffinePhaseOrder(enum.Enum):
    """Order of the independent state and coefficient work in phase B."""

    STATE_FIRST = "state_first"
    COEFFICIENT_FIRST = "coefficient_first"


@dataclasses.dataclass(frozen=True)
class DirectAffineSchedule:
    """Physical choices named by one public affine-scan schedule."""

    mma: DirectAffineMma
    coefficient_layout: DirectAffineCoefficientLayout
    state_ingress: DirectAffineStateIngress
    phase_order: DirectAffinePhaseOrder


_DIRECT_AFFINE_SCHEDULES = {
    DIRECT_AFFINE_M16N8_V1_SCHEDULE: DirectAffineSchedule(
        mma=DirectAffineMma.M16N8,
        coefficient_layout=DirectAffineCoefficientLayout.INTERLEAVED,
        state_ingress=DirectAffineStateIngress.ASYNC,
        phase_order=DirectAffinePhaseOrder.STATE_FIRST,
    ),
    DIRECT_AFFINE_M16N16_V1_SCHEDULE: DirectAffineSchedule(
        mma=DirectAffineMma.M16N16,
        coefficient_layout=DirectAffineCoefficientLayout.ROLE_MAJOR,
        state_ingress=DirectAffineStateIngress.ASYNC,
        phase_order=DirectAffinePhaseOrder.STATE_FIRST,
    ),
}


def decode_direct_affine_schedule(schedule: str) -> DirectAffineSchedule | None:
    """Decode a public schedule name into its internal physical choices."""

    if schedule == DIRECT_AFFINE_ORDINARY_SCHEDULE:
        return None
    try:
        return _DIRECT_AFFINE_SCHEDULES[schedule]
    except KeyError as error:
        raise ValueError(f"unknown direct-affine schedule: {schedule!r}") from error


def direct_affine_schedule_choices(step_count: int | None) -> tuple[str, ...]:
    """Return schedule profiles whose MMA can hold the requested scan."""

    if step_count is not None and not 2 <= step_count <= 8:
        raise ValueError(f"unsupported affine-scan step count: {step_count}")
    if step_count is not None and step_count > 4:
        return (
            DIRECT_AFFINE_ORDINARY_SCHEDULE,
            DIRECT_AFFINE_M16N16_V1_SCHEDULE,
        )
    return DIRECT_AFFINE_SCHEDULES


def select_direct_affine_schedule(step_count: int) -> str:
    """Select the measured schedule profile for a supported scan length."""

    mma = select_direct_affine_mma(step_count)
    if mma is DirectAffineMma.M16N8:
        return DIRECT_AFFINE_M16N8_V1_SCHEDULE
    if mma is DirectAffineMma.M16N16:
        return DIRECT_AFFINE_M16N16_V1_SCHEDULE
    return DIRECT_AFFINE_ORDINARY_SCHEDULE


@dataclasses.dataclass(frozen=True)
class DirectAffineStateAccessProof:
    """Facts proved by the caller for one 16-byte state-vector schedule.

    Alignment values describe every source and destination address emitted by
    the state staging loop, not merely the underlying allocations.  The
    capability fact is consumed only by asynchronous ingress.
    """

    source_vector_alignment_bytes: int
    destination_vector_alignment_bytes: int
    source_feature_stride_one: bool
    full_vector_coverage: bool
    cp_async_supported: bool

    def supports_vector_io(self) -> bool:
        return (
            _positive_int(self.source_vector_alignment_bytes)
            and self.source_vector_alignment_bytes % STATE_VECTOR_BYTES == 0
            and _positive_int(self.destination_vector_alignment_bytes)
            and self.destination_vector_alignment_bytes % STATE_VECTOR_BYTES == 0
            and self.source_feature_stride_one is True
            and self.full_vector_coverage is True
        )

    def supports_async_io(self) -> bool:
        return self.cp_async_supported is True and self.supports_vector_io()


@dataclasses.dataclass(frozen=True)
class DirectAffineColumnPlan:
    """Strides passed directly to ``short_affine_scan_mma`` helpers."""

    factor_column_extent: int
    factor_target_stride: int
    factor_role_stride: int
    coefficient_row_stride: int
    coefficient_source_stride: int
    coefficient_role_stride: int
    prediction_projection_columns: tuple[int, ...]
    observation_projection_columns: tuple[int, ...]
    coefficient_element_count: int


@dataclasses.dataclass(frozen=True)
class DirectAffinePlan:
    """Canonical physical plan for one direct affine scan."""

    step_count: int
    row_extent: int
    feature_extent: int
    storage_dtype: torch.dtype
    mma: DirectAffineMma
    coefficient_layout: DirectAffineCoefficientLayout
    state_ingress: DirectAffineStateIngress
    phase_order: DirectAffinePhaseOrder
    state_access_proof: DirectAffineStateAccessProof

    @property
    def row_warps(self) -> int:
        return self.row_extent // MMA_M

    @property
    def cta_shape(self) -> tuple[int, int, int]:
        return (WARP_SIZE, max(self.step_count, self.row_warps), 1)

    @property
    def columns(self) -> DirectAffineColumnPlan:
        column_extent = self.mma.column_extent
        columns = (
            _resolve_column_plan(
                self.step_count,
                column_extent,
                self.coefficient_layout,
            )
            if column_extent is not None
            else None
        )
        if columns is None:
            raise ValueError("invalid direct-affine column plan")
        return columns

    def validate(self) -> None:
        """Raise when the plan is not the canonical result of the resolver."""

        if (
            not isinstance(self.mma, DirectAffineMma)
            or not isinstance(self.coefficient_layout, DirectAffineCoefficientLayout)
            or not isinstance(self.state_ingress, DirectAffineStateIngress)
            or not isinstance(self.phase_order, DirectAffinePhaseOrder)
            or not isinstance(self.state_access_proof, DirectAffineStateAccessProof)
            or not _positive_int(self.row_extent)
        ):
            raise ValueError("invalid direct-affine physical plan")
        column_extent = self.mma.column_extent
        expected_columns = (
            _resolve_column_plan(
                self.step_count,
                column_extent,
                self.coefficient_layout,
            )
            if column_extent is not None
            else None
        )
        expected_row_warps = self.row_extent // MMA_M
        if (
            self.mma is DirectAffineMma.ORDINARY
            or column_extent is None
            or not _positive_int(self.step_count)
            or self.step_count < 2
            or 2 * self.step_count > column_extent
            or self.storage_dtype is not torch.bfloat16
            or not _positive_int(self.feature_extent)
            or self.feature_extent != SUPPORTED_FEATURE_EXTENT
            or self.row_extent % MMA_M
            or not 1 <= expected_row_warps <= MAX_CTA_WARPS
            or max(self.step_count, expected_row_warps) > MAX_CTA_WARPS
            or expected_columns is None
            or not self.state_access_proof.supports_vector_io()
            or (
                self.state_ingress is DirectAffineStateIngress.ASYNC
                and not self.state_access_proof.supports_async_io()
            )
        ):
            raise ValueError("invalid direct-affine physical plan")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def select_direct_affine_mma(step_count: int) -> DirectAffineMma:
    """Select the smallest native projection that holds two columns per step."""

    if not _positive_int(step_count) or step_count < 2:
        return DirectAffineMma.ORDINARY
    for mma in (DirectAffineMma.M16N8, DirectAffineMma.M16N16):
        column_extent = mma.column_extent
        assert column_extent is not None
        if 2 * step_count <= column_extent:
            return mma
    return DirectAffineMma.ORDINARY


def _resolve_column_plan(
    step_count: int,
    column_extent: int,
    layout: DirectAffineCoefficientLayout,
) -> DirectAffineColumnPlan | None:
    if (
        not _positive_int(step_count)
        or not _positive_int(column_extent)
        or 2 * step_count > column_extent
    ):
        return None
    if layout is DirectAffineCoefficientLayout.INTERLEAVED:
        factor_target_stride = 2
        factor_role_stride = 1
        coefficient_row_stride = 2 * step_count
        coefficient_source_stride = 2
        coefficient_role_stride = 1
    elif layout is DirectAffineCoefficientLayout.ROLE_MAJOR:
        factor_target_stride = 1
        factor_role_stride = column_extent // 2
        coefficient_row_stride = step_count
        coefficient_source_stride = 1
        coefficient_role_stride = step_count * step_count
    else:
        return None

    prediction_columns = tuple(
        step * factor_target_stride for step in range(step_count)
    )
    observation_columns = tuple(
        step * factor_target_stride + factor_role_stride for step in range(step_count)
    )
    if (
        len({*prediction_columns, *observation_columns}) != 2 * step_count
        or max((*prediction_columns, *observation_columns)) >= column_extent
    ):
        return None
    return DirectAffineColumnPlan(
        factor_column_extent=column_extent,
        factor_target_stride=factor_target_stride,
        factor_role_stride=factor_role_stride,
        coefficient_row_stride=coefficient_row_stride,
        coefficient_source_stride=coefficient_source_stride,
        coefficient_role_stride=coefficient_role_stride,
        prediction_projection_columns=prediction_columns,
        observation_projection_columns=observation_columns,
        coefficient_element_count=2 * step_count * step_count,
    )


def resolve_direct_affine_plan(
    *,
    step_count: int,
    row_extent: int,
    feature_extent: int,
    storage_dtype: torch.dtype,
    mma: DirectAffineMma,
    coefficient_layout: DirectAffineCoefficientLayout,
    state_ingress: DirectAffineStateIngress,
    phase_order: DirectAffinePhaseOrder,
    state_access_proof: DirectAffineStateAccessProof,
) -> DirectAffinePlan | None:
    """Resolve a supported physical choice, returning ``None`` on fallback.

    All inputs are semantic or explicit proof facts.  In particular, no
    benchmark name, tensor name, gate formula, or exact token-count branch is
    consulted here.
    """

    if (
        not isinstance(mma, DirectAffineMma)
        or not isinstance(coefficient_layout, DirectAffineCoefficientLayout)
        or not isinstance(state_ingress, DirectAffineStateIngress)
        or not isinstance(phase_order, DirectAffinePhaseOrder)
        or not isinstance(state_access_proof, DirectAffineStateAccessProof)
        or mma is DirectAffineMma.ORDINARY
    ):
        return None
    column_extent = mma.column_extent
    if column_extent is None:
        return None
    columns = _resolve_column_plan(step_count, column_extent, coefficient_layout)
    if (
        columns is None
        or storage_dtype is not torch.bfloat16
        or not _positive_int(feature_extent)
        or feature_extent != SUPPORTED_FEATURE_EXTENT
        or not _positive_int(row_extent)
        or row_extent % MMA_M
        or not state_access_proof.supports_vector_io()
        or (
            state_ingress is DirectAffineStateIngress.ASYNC
            and not state_access_proof.supports_async_io()
        )
    ):
        return None
    row_warps = row_extent // MMA_M
    cta_warps = max(step_count, row_warps)
    if not 1 <= row_warps <= MAX_CTA_WARPS or cta_warps > MAX_CTA_WARPS:
        return None
    plan = DirectAffinePlan(
        step_count=step_count,
        row_extent=row_extent,
        feature_extent=feature_extent,
        storage_dtype=storage_dtype,
        mma=mma,
        coefficient_layout=coefficient_layout,
        state_ingress=state_ingress,
        phase_order=phase_order,
        state_access_proof=state_access_proof,
    )
    try:
        plan.validate()
    except ValueError:
        return None
    return plan


__all__ = [
    "DIRECT_AFFINE_M16N8_V1_SCHEDULE",
    "DIRECT_AFFINE_M16N16_V1_SCHEDULE",
    "DIRECT_AFFINE_ORDINARY_SCHEDULE",
    "DIRECT_AFFINE_SCHEDULES",
    "MAX_CTA_WARPS",
    "MMA_M",
    "MMA_N",
    "STATE_VECTOR_BYTES",
    "SUPPORTED_FEATURE_EXTENT",
    "WARP_SIZE",
    "DirectAffineCoefficientLayout",
    "DirectAffineColumnPlan",
    "DirectAffineMma",
    "DirectAffinePhaseOrder",
    "DirectAffinePlan",
    "DirectAffineSchedule",
    "DirectAffineStateAccessProof",
    "DirectAffineStateIngress",
    "decode_direct_affine_schedule",
    "direct_affine_schedule_choices",
    "resolve_direct_affine_plan",
    "select_direct_affine_mma",
    "select_direct_affine_schedule",
]
