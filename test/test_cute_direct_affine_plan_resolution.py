from __future__ import annotations

import dataclasses

import pytest
import torch

from helion._compiler.cute.direct_affine_plan import DirectAffineCoefficientLayout
from helion._compiler.cute.direct_affine_plan import DirectAffineMma
from helion._compiler.cute.direct_affine_plan import DirectAffinePhaseOrder
from helion._compiler.cute.direct_affine_plan import DirectAffineStateAccessProof
from helion._compiler.cute.direct_affine_plan import DirectAffineStateIngress
from helion._compiler.cute.direct_affine_plan import decode_direct_affine_schedule
from helion._compiler.cute.direct_affine_plan import direct_affine_schedule_choices
from helion._compiler.cute.direct_affine_plan import resolve_direct_affine_plan
from helion._compiler.cute.direct_affine_plan import select_direct_affine_mma
from helion._compiler.cute.direct_affine_plan import select_direct_affine_schedule


def _state_proof(*, cp_async_supported: bool = True) -> DirectAffineStateAccessProof:
    return DirectAffineStateAccessProof(
        source_vector_alignment_bytes=16,
        destination_vector_alignment_bytes=128,
        source_feature_stride_one=True,
        full_vector_coverage=True,
        cp_async_supported=cp_async_supported,
    )


def test_direct_affine_mma_selection_uses_only_column_capacity() -> None:
    assert select_direct_affine_mma(1) is DirectAffineMma.ORDINARY
    assert select_direct_affine_mma(3) is DirectAffineMma.M16N8
    assert select_direct_affine_mma(4) is DirectAffineMma.M16N8
    assert select_direct_affine_mma(5) is DirectAffineMma.M16N16
    assert select_direct_affine_mma(8) is DirectAffineMma.M16N16
    assert select_direct_affine_mma(9) is DirectAffineMma.ORDINARY


@pytest.mark.parametrize(
    ("name", "mma", "layout"),
    (
        (
            "direct_m16n8_v1",
            DirectAffineMma.M16N8,
            DirectAffineCoefficientLayout.INTERLEAVED,
        ),
        (
            "direct_m16n16_v1",
            DirectAffineMma.M16N16,
            DirectAffineCoefficientLayout.ROLE_MAJOR,
        ),
    ),
)
def test_direct_affine_schedule_decodes_measured_profiles(
    name: str,
    mma: DirectAffineMma,
    layout: DirectAffineCoefficientLayout,
) -> None:
    schedule = decode_direct_affine_schedule(name)

    assert schedule is not None
    assert schedule.mma is mma
    assert schedule.coefficient_layout is layout
    assert schedule.state_ingress is DirectAffineStateIngress.ASYNC
    assert schedule.phase_order is DirectAffinePhaseOrder.STATE_FIRST


def test_direct_affine_schedule_selection_tracks_smallest_mma() -> None:
    assert decode_direct_affine_schedule("ordinary") is None
    assert select_direct_affine_schedule(3) == "direct_m16n8_v1"
    assert select_direct_affine_schedule(5) == "direct_m16n16_v1"
    assert select_direct_affine_schedule(9) == "ordinary"

    with pytest.raises(ValueError, match="unknown direct-affine schedule"):
        decode_direct_affine_schedule("direct_m8n8_v1")


@pytest.mark.parametrize(
    ("step_count", "expected"),
    (
        (None, ("ordinary", "direct_m16n8_v1", "direct_m16n16_v1")),
        (2, ("ordinary", "direct_m16n8_v1", "direct_m16n16_v1")),
        (4, ("ordinary", "direct_m16n8_v1", "direct_m16n16_v1")),
        (5, ("ordinary", "direct_m16n16_v1")),
        (8, ("ordinary", "direct_m16n16_v1")),
    ),
)
def test_direct_affine_schedule_choices_cover_supported_boundaries(
    step_count: int | None, expected: tuple[str, ...]
) -> None:
    assert direct_affine_schedule_choices(step_count) == expected


@pytest.mark.parametrize("step_count", (1, 9))
def test_direct_affine_schedule_choices_reject_unsupported_counts(
    step_count: int,
) -> None:
    with pytest.raises(ValueError, match="unsupported affine-scan step count"):
        direct_affine_schedule_choices(step_count)


def test_three_step_interleaved_m16n8_plan() -> None:
    plan = resolve_direct_affine_plan(
        step_count=3,
        row_extent=64,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        mma=DirectAffineMma.M16N8,
        coefficient_layout=DirectAffineCoefficientLayout.INTERLEAVED,
        state_ingress=DirectAffineStateIngress.ASYNC,
        phase_order=DirectAffinePhaseOrder.STATE_FIRST,
        state_access_proof=_state_proof(),
    )

    assert plan is not None
    assert plan.cta_shape == (32, 4, 1)
    assert plan.columns.factor_column_extent == 8
    assert plan.columns.prediction_projection_columns == (0, 2, 4)
    assert plan.columns.observation_projection_columns == (1, 3, 5)
    assert (
        plan.columns.coefficient_row_stride,
        plan.columns.coefficient_source_stride,
        plan.columns.coefficient_role_stride,
    ) == (6, 2, 1)
    plan.validate()


def test_five_step_role_major_m16n16_plan() -> None:
    plan = resolve_direct_affine_plan(
        step_count=5,
        row_extent=128,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        mma=DirectAffineMma.M16N16,
        coefficient_layout=DirectAffineCoefficientLayout.ROLE_MAJOR,
        state_ingress=DirectAffineStateIngress.ASYNC,
        phase_order=DirectAffinePhaseOrder.STATE_FIRST,
        state_access_proof=_state_proof(),
    )

    assert plan is not None
    assert plan.cta_shape == (32, 8, 1)
    assert plan.columns.factor_column_extent == 16
    assert plan.columns.prediction_projection_columns == (0, 1, 2, 3, 4)
    assert plan.columns.observation_projection_columns == (8, 9, 10, 11, 12)
    assert (
        plan.columns.coefficient_row_stride,
        plan.columns.coefficient_source_stride,
        plan.columns.coefficient_role_stride,
    ) == (5, 1, 25)
    assert plan.columns.coefficient_element_count == 50
    plan.validate()


@pytest.mark.parametrize(
    "proof",
    (
        _state_proof(cp_async_supported=False),
        dataclasses.replace(_state_proof(), source_vector_alignment_bytes=8),
        dataclasses.replace(_state_proof(), destination_vector_alignment_bytes=8),
        dataclasses.replace(_state_proof(), source_feature_stride_one=False),
        dataclasses.replace(_state_proof(), full_vector_coverage=False),
    ),
)
def test_async_state_ingress_requires_explicit_proofs(
    proof: DirectAffineStateAccessProof,
) -> None:
    assert (
        resolve_direct_affine_plan(
            step_count=3,
            row_extent=64,
            feature_extent=128,
            storage_dtype=torch.bfloat16,
            mma=DirectAffineMma.M16N8,
            coefficient_layout=DirectAffineCoefficientLayout.INTERLEAVED,
            state_ingress=DirectAffineStateIngress.ASYNC,
            phase_order=DirectAffinePhaseOrder.STATE_FIRST,
            state_access_proof=proof,
        )
        is None
    )


def test_sync_ingress_does_not_require_cp_async_capability() -> None:
    plan = resolve_direct_affine_plan(
        step_count=3,
        row_extent=64,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        mma=DirectAffineMma.M16N8,
        coefficient_layout=DirectAffineCoefficientLayout.ROLE_MAJOR,
        state_ingress=DirectAffineStateIngress.SYNC,
        phase_order=DirectAffinePhaseOrder.COEFFICIENT_FIRST,
        state_access_proof=_state_proof(cp_async_supported=False),
    )

    assert plan is not None
    assert plan.state_ingress is DirectAffineStateIngress.SYNC
    assert plan.phase_order is DirectAffinePhaseOrder.COEFFICIENT_FIRST
    assert plan.columns.prediction_projection_columns == (0, 1, 2)
    assert plan.columns.observation_projection_columns == (4, 5, 6)


@pytest.mark.parametrize(
    "proof",
    (
        dataclasses.replace(_state_proof(), source_vector_alignment_bytes=8),
        dataclasses.replace(_state_proof(), destination_vector_alignment_bytes=8),
        dataclasses.replace(_state_proof(), source_feature_stride_one=False),
        dataclasses.replace(_state_proof(), full_vector_coverage=False),
    ),
)
def test_sync_state_ingress_still_requires_vector_io(
    proof: DirectAffineStateAccessProof,
) -> None:
    assert (
        resolve_direct_affine_plan(
            step_count=3,
            row_extent=64,
            feature_extent=128,
            storage_dtype=torch.bfloat16,
            mma=DirectAffineMma.M16N8,
            coefficient_layout=DirectAffineCoefficientLayout.INTERLEAVED,
            state_ingress=DirectAffineStateIngress.SYNC,
            phase_order=DirectAffinePhaseOrder.STATE_FIRST,
            state_access_proof=proof,
        )
        is None
    )


@pytest.mark.parametrize("step_count", range(2, 9))
def test_supported_step_range_accepts_non_profile_row_extent(step_count: int) -> None:
    mma = select_direct_affine_mma(step_count)
    layout = (
        DirectAffineCoefficientLayout.INTERLEAVED
        if mma is DirectAffineMma.M16N8
        else DirectAffineCoefficientLayout.ROLE_MAJOR
    )
    plan = resolve_direct_affine_plan(
        step_count=step_count,
        row_extent=80,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        mma=mma,
        coefficient_layout=layout,
        state_ingress=DirectAffineStateIngress.SYNC,
        phase_order=DirectAffinePhaseOrder.COEFFICIENT_FIRST,
        state_access_proof=_state_proof(cp_async_supported=False),
    )

    assert plan is not None
    assert plan.step_count == step_count
    assert plan.row_extent == 80
    assert plan.state_ingress is DirectAffineStateIngress.SYNC
    assert plan.phase_order is DirectAffinePhaseOrder.COEFFICIENT_FIRST
    plan.validate()


@pytest.mark.parametrize("feature_extent", (128.0, True))
def test_feature_extent_requires_an_exact_integer(feature_extent: object) -> None:
    arguments: dict[str, object] = {
        "step_count": 3,
        "row_extent": 64,
        "feature_extent": feature_extent,
        "storage_dtype": torch.bfloat16,
        "mma": DirectAffineMma.M16N8,
        "coefficient_layout": DirectAffineCoefficientLayout.INTERLEAVED,
        "state_ingress": DirectAffineStateIngress.SYNC,
        "phase_order": DirectAffinePhaseOrder.STATE_FIRST,
        "state_access_proof": _state_proof(),
    }
    assert resolve_direct_affine_plan(**arguments) is None  # type: ignore[arg-type]

    arguments["feature_extent"] = 128
    plan = resolve_direct_affine_plan(**arguments)  # type: ignore[arg-type]
    assert plan is not None
    with pytest.raises(ValueError, match="invalid direct-affine physical plan"):
        # pyrefly: ignore [bad-argument-type]
        dataclasses.replace(plan, feature_extent=feature_extent).validate()


@pytest.mark.parametrize(
    ("overrides", "mma"),
    (
        ({"step_count": 5}, DirectAffineMma.M16N8),
        ({"feature_extent": 64}, DirectAffineMma.M16N8),
        ({"row_extent": 63}, DirectAffineMma.M16N8),
        ({"row_extent": 144}, DirectAffineMma.M16N8),
        ({"storage_dtype": torch.float16}, DirectAffineMma.M16N8),
        ({}, DirectAffineMma.ORDINARY),
    ),
)
def test_unsupported_choices_resolve_to_fallback(
    overrides: dict[str, object],
    mma: DirectAffineMma,
) -> None:
    arguments: dict[str, object] = {
        "step_count": 3,
        "row_extent": 64,
        "feature_extent": 128,
        "storage_dtype": torch.bfloat16,
        "mma": mma,
        "coefficient_layout": DirectAffineCoefficientLayout.INTERLEAVED,
        "state_ingress": DirectAffineStateIngress.SYNC,
        "phase_order": DirectAffinePhaseOrder.STATE_FIRST,
        "state_access_proof": _state_proof(cp_async_supported=False),
    }
    arguments.update(overrides)

    assert resolve_direct_affine_plan(**arguments) is None  # type: ignore[arg-type]


def test_plan_derived_fields_follow_inputs() -> None:
    plan = resolve_direct_affine_plan(
        step_count=5,
        row_extent=128,
        feature_extent=128,
        storage_dtype=torch.bfloat16,
        mma=DirectAffineMma.M16N16,
        coefficient_layout=DirectAffineCoefficientLayout.ROLE_MAJOR,
        state_ingress=DirectAffineStateIngress.SYNC,
        phase_order=DirectAffinePhaseOrder.COEFFICIENT_FIRST,
        state_access_proof=_state_proof(cp_async_supported=False),
    )
    assert plan is not None

    narrower = dataclasses.replace(plan, row_extent=64)
    assert narrower.row_warps == 4
    assert narrower.cta_shape == (32, 5, 1)
    narrower.validate()
