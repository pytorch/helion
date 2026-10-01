from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_producer_teams import _args
from .test_cute_chained_producer_teams import _cooperative_loop
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _config(warps: int, vectorize: bool, grouped: bool) -> helion.Config:
    return helion.Config(
        num_warps=warps,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=grouped,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=vectorize,
    )


@pytest.mark.parametrize("warps", [4, 16, 32])
@pytest.mark.parametrize("grouped", [False, True])
def test_vector_stage_preserves_partial_physical_tiles_and_masks(
    warps: int, grouped: bool
) -> None:
    with _cpu_codegen():
        bound = _cooperative_loop._bind_isolated(_args("cpu", 3, True))
        source = bound.to_code(_config(warps, True, grouped))
    assert "_last_pointer =" in source
    assert "_last_address ==" in source
    assert "_vectorized = cutlass.Boolean(False)" in source
    assert "_scalar" not in source  # Fallback is normal expression lowering.
    assert "chain_0_a_0_vector_row < 128" in source
    assert f"block=({32 * warps}, 1, 1)" in source
    assert "chain_loop_index" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "warps,grouped,steps", [(4, False, 3), (16, True, 3), (32, True, 0)]
)
def test_vector_stage_gpu_ragged_mixed_seeded_loop(
    warps: int, grouped: bool, steps: int
) -> None:
    args = _args(DEVICE, steps, True)
    copies = tuple(value.clone() for value in args[:-1])
    bound = _cooperative_loop._bind_isolated(args)
    scalar = bound.compile_config(_config(warps, False, grouped))(*args)
    compiled = bound.compile_config(_config(warps, True, grouped))
    actual = compiled(*args)
    repeated = compiled(*args)
    for position, (expected, result, replay) in enumerate(
        zip(scalar, actual, repeated, strict=True)
    ):
        if steps == 0 and position == 0:
            continue  # The source intentionally leaves zero-trip history unused.
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay, result, rtol=0, atol=0)
    for initial, current in zip(copies, args[:-1], strict=True):
        torch.testing.assert_close(current, initial, rtol=0, atol=0)
