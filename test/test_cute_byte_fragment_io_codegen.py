"""Byte packets retain the existing alignment and complete-lane proofs."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_byte_fragment_io import byte_window_gram
from helion._testing.cute_codegen import cpu_bind

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("mode", ["vector", "misaligned"])
def test_byte_window_packet_render(dtype: torch.dtype, mode: str) -> None:
    storage = torch.zeros(3 * 16 * 32 + 1, dtype=dtype)
    x = storage[1:] if mode == "misaligned" else storage[:-1]
    x = x.reshape(3, 16, 32)
    with cpu_render_target():
        bound = cpu_bind(byte_window_gram, (x,))
        key = "cute_fragment_vector_io"
        assert key in bound.config_spec._flat_fields()
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config()) | {key: True}
        )
        code = bound.to_code(config)
    scalar = "cutlass.Int8" if dtype == torch.int8 else "cutlass.Uint8"
    if mode == "vector":
        assert f"{scalar}, num_bits_per_copy=128" in code
        assert f"cute.make_layout((16,)), {scalar}" in code
    else:
        # The output remains aligned and packetized; the offset input falls back.
        assert "fragment_load" in code and ").load()" in code
        assert f"{scalar}, num_bits_per_copy=128" in code
