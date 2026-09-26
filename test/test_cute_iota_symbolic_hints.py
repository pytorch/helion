from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@helion.kernel(backend="cute", autotune_effort="none")
def _copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    block = hl.register_block_size(32, 128)
    for pid in hl.grid((x.size(0) + block - 1) // block):
        indices = pid * block + hl.arange(block)
        values = hl.load(x, [indices], extra_mask=indices < x.size(0))
        hl.store(out, [indices], values, extra_mask=indices < x.size(0))
    return out


@pytest.mark.parametrize("sizes", [(32, 128, 64), (128, 32, 64)])
def test_iota_codegen_keeps_tile_sizes_symbolic(sizes: tuple[int, ...]) -> None:
    bound = _copy.bind((torch.empty(257),))
    shape_env = bound.env.shape_env
    symbols = [block.var.node._expr for block in bound.env.block_sizes]
    assert symbols
    for size in sizes:
        bound.to_code(helion.Config(block_sizes=[size]))
        for symbol in symbols:
            assert shape_env.replace(symbol) == symbol


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_iota_repeated_configurations_copy_all_elements() -> None:
    x = torch.arange(257, device=CUDA_DEVICE, dtype=torch.float32)
    bound = _copy.bind((x,))
    for size in (32, 128, 64):
        bound.set_config(helion.Config(block_sizes=[size]))
        torch.testing.assert_close(bound(x), x, rtol=0, atol=0)
