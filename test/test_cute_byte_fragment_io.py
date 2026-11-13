"""Native byte transport preserves signedness, tails and changed inputs."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.cute_byte_fragment_io import byte_window_gram

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("mode", ["vector", "packet_tail", "misaligned"])
def test_byte_window_packet_values(dtype: torch.dtype, mode: str) -> None:
    cols = 35 if mode == "packet_tail" else 32
    rows = 17 if mode == "packet_tail" else 16
    offset = int(mode == "misaligned")
    count = 3 * rows * cols
    storage = torch.arange(count + offset, device=DEVICE).to(dtype)
    x = storage[offset:].reshape(3, rows, cols)
    args = (x,)
    bound = byte_window_gram.bind(args)
    key = (
        "cute_fragment_packet_loads"
        if mode == "packet_tail"
        else "cute_fragment_vector_io"
    )
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {key: True}
    )
    if mode == "packet_tail":
        code = bound.to_code(config)
        assert "num_bits_per_copy=32" in code
        assert "fragment_packet_valid" in code
    compiled = bound.compile_config(config)
    result = compiled(*args)

    def check() -> None:
        expected = x.float() @ x.float().transpose(-1, -2)
        torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(result[1], x + 1, rtol=0, atol=0)

    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compiled(*args)
    x.add_(17)
    graph.replay()
    check()
