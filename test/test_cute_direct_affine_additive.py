from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from test.test_cute_affine_scan_heuristic import _fake_cute_context
from test.test_cute_direct_affine_lowering import _real_codegen_tensor_metadata

import helion
from helion._compiler.cute import direct_affine_lowering as lowering
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=False, fast_math=True)
def _diagonal_outer_scan(
    diagonal: torch.Tensor,
    update: torch.Tensor,
    observation: torch.Tensor,
    predictor: torch.Tensor,
    values: torch.Tensor,
    checkpoints: torch.Tensor,
    output: torch.Tensor,
    feedback_mask: hl.constexpr,
) -> torch.Tensor:
    steps = hl.specialize(values.size(0))
    features = hl.specialize(diagonal.size(-1))
    hl.specialize(checkpoints.size(0))
    row_count = hl.specialize(values.size(3))
    hl.specialize(checkpoints.size(3))
    hl.specialize(
        (
            diagonal.stride(),
            update.stride(),
            observation.stride(),
            predictor.stride(),
            values.stride(),
            checkpoints.stride(),
            output.stride(),
        )
    )
    for batch_tile, head_tile, rows in hl.tile(
        [values.size(1), values.size(2), row_count], block_size=[1, 1, None]
    ):
        batch = batch_tile.id
        head = head_tile.id
        feature = hl.arange(features)
        state = checkpoints[0, batch, head, rows.index, feature].float()
        for step in hl.static_range(steps):
            decay = diagonal[step, batch, head, feature].float()
            state = state * decay[None, :]
            row_update = torch.sin(values[step, batch, head, rows].float())
            if feedback_mask & (1 << step):  # pyrefly: ignore[unsupported-operation]
                prediction = (
                    state * predictor[step, batch, head, feature].float()[None, :]
                ).sum(-1)
                row_update = 0.5 * (row_update - prediction)
            state = (
                state
                + row_update[:, None]
                * update[step, batch, head, feature].float()[None, :]
            )
            output[step, batch, head, rows] = (
                state * observation[step, batch, head, feature].float()[None, :]
            ).sum(-1)
            checkpoints[step + 1, batch, head, rows.index, feature] = state
    return output


def _inputs(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(18)
    vectors = [
        torch.randn((3, 1, 2, 128), generator=generator, device=device) * 0.1
        for _ in range(4)
    ]
    vectors[0] = vectors[0] * 0.1 + 0.8
    return (
        *vectors,
        torch.randn((3, 1, 2, 64), generator=generator, device=device) * 0.1,
        (torch.randn((4, 1, 2, 64, 128), generator=generator, device=device) * 0.1).to(
            torch.bfloat16
        ),
        torch.empty((3, 1, 2, 64), device=device, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("feedback_mask", (0, 5, 7))
def test_additive_and_feedback_use_same_real_codegen(feedback_mask: int) -> None:
    with (
        _fake_cute_context(),
        patch.object(lowering, "_tensor_metadata", new=_real_codegen_tensor_metadata),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensor_has_specialized_alignment",
            return_value=True,
        ),
        patch(
            "helion._compiler.cute.memory_ops.runtime_tensors_are_proven_disjoint",
            return_value=True,
        ),
    ):
        bound = _diagonal_outer_scan._bind_isolated((*_inputs(), feedback_mask))
        config = next(
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_affine_scan_schedule") == "direct_m16n8_v1"
            and seed.config.get("block_sizes") == [64]
        )
        source = bound.to_triton_code(config)
    assert "precompute_affine_from_buffers_bf16" in source
    assert "consume_affine_steps" in source
    assert "checkpoint_affine_row_bf16" in source
    if feedback_mask == 7:
        assert "FEEDBACK_MASK=" not in source
        assert "HAS_FEEDBACK=" not in source
    else:
        assert f"FEEDBACK_MASK={feedback_mask}" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("feedback_mask", (0, 5, 7))
def test_additive_and_feedback_runtime(feedback_mask: int) -> None:
    args = _inputs("cuda")
    diagonal, update, observation, predictor, values, checkpoints, output = args
    for step in range(3):
        if not feedback_mask & (1 << step):
            predictor[step].fill_(float("nan"))
    expected_state = checkpoints[0].float()
    expected_outputs = []
    expected_checkpoints = []
    for step in range(3):
        expected_state = expected_state * diagonal[step, :, :, None, :]
        row_update = torch.sin(values[step])
        if feedback_mask & (1 << step):
            row_update = 0.5 * (
                row_update - (expected_state * predictor[step, :, :, None, :]).sum(-1)
            )
        expected_state = (
            expected_state + row_update[:, :, :, None] * update[step, :, :, None, :]
        )
        expected_outputs.append(
            (expected_state * observation[step, :, :, None, :]).sum(-1).to(output.dtype)
        )
        expected_checkpoints.append(expected_state.to(checkpoints.dtype))
    bound = _diagonal_outer_scan._bind_isolated((*args, feedback_mask))
    config = next(
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_affine_scan_schedule") == "direct_m16n8_v1"
        and seed.config.get("block_sizes") == [64]
    )
    actual = bound.compile_config(config, allow_print=False)(*args, feedback_mask)
    torch.testing.assert_close(
        actual, torch.stack(expected_outputs), rtol=0.08, atol=5e-4
    )
    torch.testing.assert_close(
        checkpoints[1:], torch.stack(expected_checkpoints), rtol=0.08, atol=5e-4
    )
