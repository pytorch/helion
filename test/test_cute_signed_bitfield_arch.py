"""Packed signed-byte conversions are available throughout Blackwell SM10x."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _mock_cuda_unavailable
from .test_cute_lane_invariant_atomics import _convert_bytes_and_count
from .test_cute_lane_loop_distribution_cuda import _convert_bytes_then_zero
from .test_cute_lane_loop_distribution_cuda import _copy_zero_copy
import helion


@pytest.mark.parametrize(
    "capability,enabled",
    [
        ((10, 0), True),
        ((10, 3), True),
        ((8, 0), False),
        ((9, 0), False),
        ((12, 0), False),
    ],
)
@pytest.mark.parametrize("columns", [128, 256])
@pytest.mark.parametrize("kind", ["counter", "zero_then_flush", "copy_zero_copy"])
def test_packet_conversion_discovery_and_emission(
    capability: tuple[int, int], enabled: bool, columns: int, kind: str
) -> None:
    packed = torch.empty((8, columns), dtype=torch.int8)
    args = (packed,)
    if kind == "counter":
        kernel = _convert_bytes_and_count
        args = (packed, torch.zeros((1,), dtype=torch.int32))
    elif kind == "zero_then_flush":
        kernel = _convert_bytes_then_zero
    else:
        kernel = _copy_zero_copy
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion.runtime.kernel.target_device_capability", return_value=capability
        ),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=capability,
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
    ):
        bound = kernel._bind_isolated(args)
        assert bound.config_spec.target_device_capability == capability
        assert bound.config_spec.cute_signed_bitfield_bf16_available is enabled
        config = helion.Config(
            block_sizes=[1, columns],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
            cute_signed_bitfield_bf16=True,
        )
        if enabled:
            code = bound.to_code(config)
            if kind == "zero_then_flush":
                assert "_cute_signed_bitfield_to_bf16_packed(" in code
            else:
                # Discovery does not override the emitter's effect checks.
                assert "def _helion_" in code
        else:
            with pytest.raises(helion.exc.InvalidConfig, match="signed-byte BF16"):
                bound.to_code(config)
