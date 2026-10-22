from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.cute.affine_vector_io import _loaded_vector_register_reads
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, disable_autotuner_heuristics=True)
def _copy_integer_routes(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows, width = data.shape
    output = torch.empty((rows, width), dtype=data.dtype, device=data.device)
    for row, col in hl.tile([rows, width]):
        destination = positions[row]
        hl.store(
            output,
            [destination[:, None], col.index[None, :]],
            data[row, col],
            extra_mask=row.index[:, None] < rows,
        )
    return output


@pytest.mark.parametrize("suffix", ["", "packet = other", "packet[0] = other"])
def test_loaded_vector_proof_rejects_rebound_or_mutated_bindings(suffix: str) -> None:
    tree = ast.parse(
        "packet = cute.arch.load(pointer, "
        "ir.VectorType.get([4], cutlass.Uint32.mlir_type))\n"
        f"{suffix}\n"
        "value = packet[lane]\n"
    )
    reads = _loaded_vector_register_reads(tree.body)
    assert bool(reads) == (not suffix)


def test_named_tensor_extract_is_not_a_loaded_vector_register() -> None:
    tree = ast.parse("packet = tensor\nvalue = packet[lane]\n")
    assert not _loaded_vector_register_reads(tree.body)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint32])
@pytest.mark.parametrize("width, offset", [(384, 0), (385, 0), (384, 1)])
@skipUnlessBackends(["cute"])
def test_integer_route_copy_preserves_bits_and_partial_tiles(
    dtype: torch.dtype, width: int, offset: int
) -> None:
    # Include sign extremes and the integer bit patterns of floating NaNs.
    patterns = torch.tensor(
        [-2147483648, 2147483647, -1, 0, 1, 2143289345, -8388607],
        dtype=torch.int32,
        device=DEVICE,
    )
    count = 5 * width
    storage = patterns.repeat(
        (count + offset + patterns.numel() - 1) // patterns.numel()
    )
    data = storage[offset : offset + count].view(5, width).view(dtype)
    positions = torch.tensor([4, 0, 3, 1, 2], dtype=torch.int32, device=DEVICE)
    bound = _copy_integer_routes.bind((data, positions))
    config = bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[2, 512],
            num_threads=[0, 128],
            cute_vector_widths=[1, 4],
        )
    )
    if width == 384 and offset == 0 and dtype == torch.int32:
        assert "_cute_store_u32_vec(" in bound.to_code(config)
    bound.set_config(config)
    actual = bound(data, positions).view(torch.int32)
    expected = torch.empty_like(actual)
    expected[positions.long()] = data.view(torch.int32)
    assert torch.equal(actual, expected)
    positions.copy_(torch.tensor([2, 4, 0, 3, 1], device=DEVICE))
    actual = bound(data, positions).view(torch.int32)
    expected[positions.long()] = data.view(torch.int32)
    assert torch.equal(actual, expected)
