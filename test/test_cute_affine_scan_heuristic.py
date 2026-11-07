from __future__ import annotations

import pytest
import torch

import helion
from helion._testing.cute_fixed_token_rank1 import fixed_rank1
from helion._testing.cute_fixed_token_rank1 import fixed_rank1_cpu_target
from helion._testing.cute_fixed_token_rank1 import fixed_rank1_inputs
from helion._testing.cute_fixed_token_rank1 import precise_fixed_rank1
import helion.language as hl

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")


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
    with fixed_rank1_cpu_target():
        bound = fixed_rank1._bind_isolated(
            (*fixed_rank1_inputs(steps), steps, 1.0e-6, True, False, False, False)
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
    with fixed_rank1_cpu_target():
        bound = _pointwise._bind_isolated((torch.empty(256), torch.empty(256)))

    assert bound.config_spec.cute_affine_scan_schedule is None
    assert "cute_affine_scan" not in bound.config_spec.autotuner_heuristics


def test_affine_scan_schedule_stays_hidden_without_fast_math() -> None:
    with fixed_rank1_cpu_target():
        bound = precise_fixed_rank1._bind_isolated(
            (*fixed_rank1_inputs(3), 3, 1.0e-6, True, False, False, False)
        )

    assert bound.config_spec.cute_affine_scan_schedule is None
    assert "cute_affine_scan" not in bound.config_spec.autotuner_heuristics
