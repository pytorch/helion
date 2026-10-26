from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import _has_cute_dsl
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["triton", "cute"])


@pytest.fixture(params=["triton", "cute"])
def backend(request: pytest.FixtureRequest) -> str:
    if request.param == "cute" and not _has_cute_dsl():
        pytest.skip("requires the supported CuTe runtime")
    return request.param


def _jagged_broadcast(
    lengths: torch.Tensor,
    data: torch.Tensor,
    mode: hl.constexpr,
    parent_size: hl.constexpr,
) -> torch.Tensor:
    for parent in hl.tile(lengths.size(0), block_size=parent_size):
        for other in hl.tile(lengths.size(1), block_size=1):
            parent_values = lengths[parent, 0]
            other_values = lengths[0, other]
            if mode == "two_parents" or mode == "missing_one_parent":
                extent = lengths[parent, other]
            else:
                extent = lengths[parent, 0]
            for jagged in hl.jagged_tile(extent):
                if mode == "missing":
                    indices = jagged.index
                elif mode == "literal_unit":
                    indices = jagged.index[None, :]
                elif mode == "unrelated_unit":
                    indices = other_values[:, None] * 0 + jagged.index[None, :]
                elif mode == "parent_with_unit_operand":
                    combined = parent_values[:, None] + other_values[:, None]
                    indices = combined * 0 + jagged.index[None, :]
                elif mode == "unrelated_units":
                    for third in hl.tile(lengths.size(1), block_size=1):
                        third_values = lengths[0, third]
                        combined = other_values[:, None] + third_values[:, None]
                        indices = combined * 0 + jagged.index[None, :]
                elif mode == "missing_one_parent":
                    indices = parent_values[:, None] * 0 + jagged.index[None, :]
                elif mode == "two_parents":
                    indices = (
                        parent_values[:, None, None] * 0
                        + other_values[None, :, None] * 0
                        + jagged.index[None, None, :]
                    )
                else:
                    indices = parent_values[:, None] * 0 + jagged.index[None, :]
                values = hl.load(data, [indices])
                hl.store(data, [indices], values)
    return data


@pytest.mark.parametrize("mode", ["parent", "two_parents", "parent_with_unit_operand"])
@skipIfRefEager("Test checks compiler lowering")
def test_fixed_unit_jagged_parent_survives_broadcast(backend: str, mode: str) -> None:
    kernel = helion.kernel(_jagged_broadcast, backend=backend, autotune_effort="none")
    bound = kernel.bind((torch.ones((3, 2), dtype=torch.int64), torch.ones(8), mode, 1))
    assert bound.host_function is not None


@pytest.mark.parametrize(
    ("mode", "parent_size"),
    [
        ("missing", 1),
        ("literal_unit", 1),
        ("unrelated_unit", 1),
        ("unrelated_unit", 2),
        ("unrelated_units", 1),
        ("missing_one_parent", 1),
    ],
)
@skipIfRefEager("Compiler validation does not run in ref eager mode")
def test_jagged_parent_requires_operand_provenance(
    backend: str, mode: str, parent_size: int
) -> None:
    kernel = helion.kernel(_jagged_broadcast, backend=backend, autotune_effort="none")
    with pytest.raises(exc.InvalidJaggedTileUsage, match="assignment indices"):
        kernel.bind(
            (torch.ones((3, 2), dtype=torch.int64), torch.ones(8), mode, parent_size)
        )


@skipIfNotCUDA()
@skipIfRefEager("jagged_tile is not implemented in ref eager mode")
def test_fixed_unit_jagged_parents_execute(backend: str) -> None:
    @helion.kernel(backend=backend, config={"block_sizes": [4]})
    def row_sum(data: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        rows, columns = offsets.size(0) - 1, offsets.size(1)
        output = torch.empty((rows, columns), dtype=data.dtype, device=data.device)
        for row, column in hl.tile([rows, columns], block_size=[1, 1]):
            starts = offsets[row, column]
            ends = offsets[row.index + 1, column]
            lengths = ends - starts
            acc = hl.zeros([row, column], dtype=data.dtype)
            for jagged in hl.jagged_tile(lengths):
                indices = starts[:, :, None] + jagged.index[None, None, :]
                acc += data[indices].sum(dim=2)
            output[row, column] = acc
        return output

    offsets = torch.tensor([[0, 1], [3, 1], [4, 6], [8, 9]], device=DEVICE)
    data = torch.arange(9, dtype=torch.float32, device=DEVICE)
    expected = torch.tensor(
        [[3, 0], [3, 15], [22, 21]], dtype=torch.float32, device=DEVICE
    )
    torch.testing.assert_close(row_sum(data, offsets), expected)
