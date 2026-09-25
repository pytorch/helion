from __future__ import annotations

import importlib
import inspect

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
mma = importlib.import_module("helion._compiler.cute.short_affine_scan_mma")
primitives = importlib.import_module(
    "helion._compiler.cute.affine_recurrence_primitives"
)
pipeline_primitives = importlib.import_module(
    "helion._compiler.cute.warp_specialized_primitives"
)
rank1 = importlib.import_module("helion._compiler.cute.single_token_rank1_recurrence")


def _factor_b_fragment_coordinates(
    lane: int,
    feature_block: int,
    column_extent: int,
) -> tuple[int, int]:
    column = lane // 16 * 8 if column_extent == 16 else 0
    return feature_block * 16 + lane % 16, column


def _state_a_fragment_coordinates(
    lane: int,
    row_base: int,
    feature_block: int,
) -> tuple[int, int]:
    return row_base + lane % 16, feature_block * 16 + lane // 16 * 8


def _state_a_fragment_value_coordinates(
    lane: int,
    row_base: int,
    feature_block: int,
    register: int,
    packed_half: int,
) -> tuple[int, int]:
    return (
        row_base + lane // 4 + 8 * (register % 2),
        feature_block * 16 + 2 * (lane % 4) + 8 * (register // 2) + packed_half,
    )


def _accumulator_coord(lane: int, slot: int) -> tuple[int, int]:
    n_block = slot // 4
    local_slot = slot - 4 * n_block
    row = lane // 4 + 8 * (local_slot // 2)
    column = 2 * (lane % 4) + local_slot % 2 + n_block * 8
    return row, column


def test_physical_strides_describe_both_winning_layouts() -> None:
    # T3 M16N8: pairs are interleaved in both tables.
    assert [mma.direct_factor_column(step, 0, 2, 1) for step in range(3)] == [
        0,
        2,
        4,
    ]
    assert [mma.direct_factor_column(step, 1, 2, 1) for step in range(3)] == [
        1,
        3,
        5,
    ]
    assert [
        mma.direct_coefficient_index(2, source, role, 6, 2, 1)
        for role in range(2)
        for source in range(3)
    ] == [12, 14, 16, 13, 15, 17]

    # T5 M16N16: factors and coefficients are split into role-major planes.
    assert [mma.direct_factor_column(step, 0, 1, 8) for step in range(5)] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [mma.direct_factor_column(step, 1, 1, 8) for step in range(5)] == [
        8,
        9,
        10,
        11,
        12,
    ]
    assert [
        mma.direct_coefficient_index(2, source, role, 5, 1, 25)
        for role in range(2)
        for source in range(5)
    ] == [10, 11, 12, 13, 14, 35, 36, 37, 38, 39]


@pytest.mark.parametrize("step_count", range(2, 9))
def test_coefficient_layouts_are_bijections(step_count: int) -> None:
    interleaved = {
        mma.direct_coefficient_index(
            target,
            source,
            role,
            2 * step_count,
            2,
            1,
        )
        for role in range(2)
        for target in range(step_count)
        for source in range(step_count)
    }
    role_major = {
        mma.direct_coefficient_index(
            target,
            source,
            role,
            step_count,
            1,
            step_count * step_count,
        )
        for role in range(2)
        for target in range(step_count)
        for source in range(step_count)
    }
    expected = set(range(2 * step_count * step_count))
    assert interleaved == expected
    assert role_major == expected


@pytest.mark.parametrize("column_extent", (8, 16))
def test_factor_swizzle_is_a_dense_permutation(column_extent: int) -> None:
    indices = {
        mma.direct_factor_index(feature, column, column_extent)
        for feature in range(128)
        for column in range(column_extent)
    }
    assert indices == set(range(128 * column_extent))


@pytest.mark.parametrize("row_extent", (64, 128))
def test_state_swizzle_is_a_dense_segmented_permutation(row_extent: int) -> None:
    indices = {
        mma.direct_state_index(row, feature, row_extent)
        for row in range(row_extent)
        for feature in range(128)
    }
    assert indices == set(range(row_extent * 128))


@pytest.mark.parametrize("column_extent", (8, 16))
def test_factor_fragment_starts_match_ldmatrix_geometry(column_extent: int) -> None:
    coordinates = [
        _factor_b_fragment_coordinates(lane, 2, column_extent) for lane in range(32)
    ]
    assert [feature for feature, _column in coordinates[:16]] == list(range(32, 48))
    assert [feature for feature, _column in coordinates[16:]] == list(range(32, 48))
    expected_columns = [0] * 32 if column_extent == 8 else [0] * 16 + [8] * 16
    assert [column for _feature, column in coordinates] == expected_columns
    addresses = [
        mma.direct_factor_b_fragment_index(lane, 2, column_extent) for lane in range(32)
    ]
    assert addresses == [
        mma.direct_factor_index(feature, column, column_extent)
        for feature, column in coordinates
    ]
    assert all(address * 2 % 16 == 0 for address in addresses)


def test_state_fragment_starts_match_ldmatrix_geometry() -> None:
    coordinates = [_state_a_fragment_coordinates(lane, 32, 3) for lane in range(32)]
    addresses = [
        mma.direct_state_a_fragment_index(lane, 32, 3, 128) for lane in range(32)
    ]
    assert addresses == [
        mma.direct_state_index(row, feature, 128) for row, feature in coordinates
    ]
    assert all(address * 2 % 16 == 0 for address in addresses)


def test_fragment_coordinates_cover_one_m16k16_tile() -> None:
    values = {
        _state_a_fragment_value_coordinates(
            lane,
            32,
            3,
            register,
            packed_half,
        )
        for lane in range(32)
        for register in range(4)
        for packed_half in range(2)
    }
    assert values == {
        (row, feature) for row in range(32, 48) for feature in range(48, 64)
    }


@pytest.mark.parametrize(("slots", "columns"), ((4, 8), (8, 16)))
def test_accumulator_coordinates_cover_projection_tile(
    slots: int,
    columns: int,
) -> None:
    coordinates = {
        _accumulator_coord(lane, slot) for lane in range(32) for slot in range(slots)
    }
    assert coordinates == {
        (row, column) for row in range(16) for column in range(columns)
    }


def _sequential_affine_scan(
    initial_state: torch.Tensor,
    diagonal: torch.Tensor,
    update: torch.Tensor,
    prediction: torch.Tensor,
    observation: torch.Tensor,
    row_input: torch.Tensor,
    update_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = initial_state.clone()
    residuals = []
    outputs = []
    states = []
    for step in range(diagonal.shape[0]):
        state = state * diagonal[step]
        residual = row_input[step] - state @ prediction[step]
        state = state + (update_scale[step] * residual)[:, None] * update[step]
        residuals.append(residual)
        outputs.append(state @ observation[step])
        states.append(state.clone())
    return torch.stack(residuals), torch.stack(outputs), torch.stack(states)


def _direct_affine_scan(
    initial_state: torch.Tensor,
    diagonal: torch.Tensor,
    update: torch.Tensor,
    prediction: torch.Tensor,
    observation: torch.Tensor,
    row_input: torch.Tensor,
    update_scale: torch.Tensor,
    *,
    row_stride: int,
    source_stride: int,
    role_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = diagonal.shape[0]
    coefficients = diagonal.new_zeros(2 * steps * steps)
    prefix = torch.cumprod(diagonal, dim=0)
    prediction_projection = initial_state @ (prefix * prediction).T
    observation_projection = initial_state @ (prefix * observation).T

    for target in range(steps):
        suffix = diagonal.new_ones(diagonal.shape[1])
        for source in range(target, -1, -1):
            weighted_update = update[source] * suffix * update_scale[source]
            if source < target:
                coefficients[
                    mma.direct_coefficient_index(
                        target,
                        source,
                        0,
                        row_stride,
                        source_stride,
                        role_stride,
                    )
                ] = prediction[target] @ weighted_update
            coefficients[
                mma.direct_coefficient_index(
                    target,
                    source,
                    1,
                    row_stride,
                    source_stride,
                    role_stride,
                )
            ] = observation[target] @ weighted_update
            if source > 0:
                suffix *= diagonal[source]

    residuals = []
    outputs = []
    for target in range(steps):
        residual = row_input[target] - prediction_projection[:, target]
        for source in range(target):
            coefficient = coefficients[
                mma.direct_coefficient_index(
                    target,
                    source,
                    0,
                    row_stride,
                    source_stride,
                    role_stride,
                )
            ]
            residual -= coefficient * residuals[source]
        residuals.append(residual)

        output = observation_projection[:, target]
        for source in range(target + 1):
            coefficient = coefficients[
                mma.direct_coefficient_index(
                    target,
                    source,
                    1,
                    row_stride,
                    source_stride,
                    role_stride,
                )
            ]
            output += coefficient * residuals[source]
        outputs.append(output)

    states = []
    for target in range(steps):
        state = initial_state * prefix[target]
        for source in range(target + 1):
            suffix = torch.prod(diagonal[source + 1 : target + 1], dim=0)
            state += (
                (update_scale[source] * residuals[source])[:, None]
                * update[source]
                * suffix
            )
        states.append(state)
    return torch.stack(residuals), torch.stack(outputs), torch.stack(states)


@pytest.mark.parametrize("step_count", range(2, 9))
def test_physical_layouts_preserve_affine_scan_algebra(step_count: int) -> None:
    generator = torch.Generator().manual_seed(20260914 + step_count)
    rows = 5
    features = 17
    initial_state = torch.randn(
        rows, features, dtype=torch.float64, generator=generator
    )
    diagonal = 0.75 + 0.2 * torch.rand(
        step_count,
        features,
        dtype=torch.float64,
        generator=generator,
    )
    diagonal[1, ::5] = 0.0
    update = torch.randn(
        step_count,
        features,
        dtype=torch.float64,
        generator=generator,
    )
    prediction = torch.randn(
        update.shape, dtype=update.dtype, device=update.device, generator=generator
    )
    observation = torch.randn(
        update.shape, dtype=update.dtype, device=update.device, generator=generator
    )
    row_input = torch.randn(
        step_count,
        rows,
        dtype=torch.float64,
        generator=generator,
    )
    update_scale = torch.randn(step_count, dtype=torch.float64, generator=generator)
    expected = _sequential_affine_scan(
        initial_state,
        diagonal,
        update,
        prediction,
        observation,
        row_input,
        update_scale,
    )
    if step_count <= 4:
        layout = (2 * step_count, 2, 1)
    else:
        layout = (step_count, 1, step_count * step_count)
    actual = _direct_affine_scan(
        initial_state,
        diagonal,
        update,
        prediction,
        observation,
        row_input,
        update_scale,
        row_stride=layout[0],
        source_stride=layout[1],
        role_stride=layout[2],
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_device_helpers_reuse_shared_primitives() -> None:
    assert mma.ldmatrix_x2_trans is primitives.ldmatrix_x2_trans
    assert mma.ldmatrix_x4_trans is primitives.ldmatrix_x4_trans
    assert mma.mma_m16n8k16_bf16 is primitives.mma_m16n8k16_bf16
    assert mma.pack_bf16x2 is primitives.pack_bf16x2
    assert mma._store_u32x4_if_valid is primitives.store_u32x4_if_valid
    assert mma.vec8_bf16 is primitives.vec8_bf16
    assert rank1.rank1_store_u32x4_if_valid is primitives.store_u32x4_if_valid
    assert mma.copy_b16x8_async is pipeline_primitives.copy_b16x8_async
    assert (
        mma.segmented_swizzle_b16_element_index
        is pipeline_primitives.segmented_swizzle_b16_element_index
    )


def test_async_ingress_leaves_wait_and_barrier_to_caller() -> None:
    source = inspect.getsource(mma.stage_state_tile8x8_async_bf16)
    primitive_source = inspect.getsource(pipeline_primitives.copy_b16x8_async)
    assert "copy_b16x8_async(" in source
    assert "cp_async_commit_group" in source
    assert "cp_async_wait_group" not in source
    assert "sync_threads" not in source
    assert "cp_async_shared_global" in primitive_source
    assert "cp_size=copy_size" in primitive_source
    assert "assumed_align=16" in primitive_source
    assert "cp_async_wait_group" not in primitive_source
    assert "sync_threads" not in primitive_source


def test_packed_store_delegates_to_shared_predicated_primitive() -> None:
    source = inspect.getsource(mma.store_packed_b16x8_if_valid)
    assert "_store_u32x4_if_valid(" in source
    assert "if slot" not in source


def test_module_is_a_physical_primitive_not_a_complete_kda_kernel() -> None:
    source = inspect.getsource(mma)
    for application_name in (
        "accepted_ptr",
        "gate_ptr",
        "k_ptr",
        "output_ptr",
        "q_ptr",
        "query_head",
        "sequence",
        "state_indices_ptr",
        "state_pool",
        "value_head",
    ):
        assert application_name not in source
    assert "@cute.kernel" not in source
    assert len(source.splitlines()) <= 1200
