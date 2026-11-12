"""The tcgen05 TMA store's last check of its destination.

A TensorMap needs a 16-byte base address and 16-byte outer strides
(``_check_tcgen05_tma_store_destination``).  The bound alignment signature
records an input's base modulo 32 bytes, so a destination that starts 16
bytes into a 32-byte block is addressable, while one 8 bytes in is not.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.tcgen05.store_epilogue import (
    _check_tcgen05_tma_store_destination,
)
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        out[tile] = x[tile]
    return out


def _check_destination(offset_bytes: int) -> None:
    """Run the check on an ``out`` that starts ``offset_bytes`` into a
    32-byte-aligned allocation."""
    storage = torch.empty(64 * 64 + 32, dtype=torch.bfloat16)
    assert storage.data_ptr() % 32 == 0
    start = offset_bytes // storage.element_size()
    out = storage[start : start + 64 * 64].view(64, 64)
    bound = cpu_bind(_copy_into, (torch.empty(64, 64, dtype=torch.bfloat16), out))
    with bound.env:
        _check_tcgen05_tma_store_destination(
            bound.fake_args[1],
            "out",
            transposed_store=False,
            output_column_major=False,
        )


@pytest.mark.parametrize("offset_bytes", [0, 16])
def test_a_16_byte_aligned_destination_passes_the_store_check(
    offset_bytes: int,
) -> None:
    _check_destination(offset_bytes)


def test_an_8_byte_aligned_destination_is_refused() -> None:
    with pytest.raises(exc.BackendUnsupported, match="base address residue 8"):
        _check_destination(8)
