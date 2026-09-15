from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from test.test_cute_fixed_token_rank1_recurrence import _fixed_rank1
from test.test_cute_fixed_token_rank1_recurrence import _inputs
from test.test_cute_fixed_token_rank1_recurrence import _precise_fixed_rank1
from test.test_cute_fixed_token_rank1_recurrence import _to_cuda_fake_tensor

import helion
import helion.language as hl

pytest.importorskip("cutlass")


def _fake_cute_context() -> ExitStack:
    stack = ExitStack()
    stack.enter_context(
        patch("helion.runtime.kernel._find_device", return_value=torch.device("cuda"))
    )
    stack.enter_context(
        patch(
            "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
            new=_to_cuda_fake_tensor,
        )
    )
    stack.enter_context(
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 3))
    )
    stack.enter_context(
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        )
    )
    stack.enter_context(
        patch("helion.language.loops.use_tileir_tunables", return_value=False)
    )
    stack.enter_context(
        patch("helion.language.loops._supports_warp_specialize", return_value=True)
    )
    stack.enter_context(
        patch("helion._compat._supports_tensor_descriptor", return_value=True)
    )
    stack.enter_context(
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16))
    )
    stack.enter_context(patch("helion._compat._is_hip", return_value=False))
    return stack


@pytest.mark.parametrize(
    ("steps", "schedule", "rows", "threads"),
    (
        (2, "direct_m16n8_v1", 64, [4, 32]),
        (4, "direct_m16n8_v1", 64, [4, 32]),
        (5, "direct_m16n16_v1", 128, [8, 32]),
        (6, "direct_m16n16_v1", 128, [8, 32]),
    ),
)
def test_affine_scan_seed_is_algebraic(
    steps: int,
    schedule: str,
    rows: int,
    threads: list[int],
) -> None:
    with _fake_cute_context():
        bound = _fixed_rank1._bind_isolated(
            (*_inputs(steps), steps, 1.0e-6, True, False, False, False)
        )

    assert "cute_affine_scan" in bound.config_spec.autotuner_heuristics
    matching = [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_affine_scan_schedule") == schedule
        and seed.config.get("block_sizes") == [rows]
        and seed.config.get("num_threads") == threads
        and seed.config.get("loop_orders") == [[2, 1, 0]]
    ]
    assert len(matching) == 1
    seed = matching[0]
    assert seed["num_warps"] == rows // 16
    assert seed["num_stages"] == 1
    assert seed["indexing"] == "pointer"
    assert seed["pid_type"] == "flat"
    assert seed["cute_affine_scan_schedule"] == schedule


@helion.kernel(backend="cute", static_shapes=False)
def _pointwise(x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        output[tile] = x[tile] + 1
    return output


def test_affine_scan_schedule_stays_hidden_for_unrelated_kernel() -> None:
    with _fake_cute_context():
        bound = _pointwise._bind_isolated((torch.empty(256), torch.empty(256)))

    assert bound.config_spec.cute_affine_scan_schedule is None
    assert "cute_affine_scan" not in bound.config_spec.autotuner_heuristics


def test_affine_scan_schedule_stays_hidden_without_fast_math() -> None:
    with _fake_cute_context():
        bound = _precise_fixed_rank1._bind_isolated(
            (*_inputs(3), 3, 1.0e-6, True, False, False, False)
        )

    assert bound.config_spec.cute_affine_scan_schedule is None
    assert "cute_affine_scan" not in bound.config_spec.autotuner_heuristics
