from __future__ import annotations

import ast

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _tickets(
    indices: torch.Tensor, increments: torch.Tensor, counters: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(indices)
    second = torch.empty_like(indices)
    for tile in hl.tile(indices.numel()):
        index = indices[tile]
        value = increments[tile].to(torch.int32)
        first[tile] = hl.atomic_add(counters, [index], value)
        second[tile] = hl.atomic_add(counters, [index], value)
    return first, second


@helion.kernel(backend="cute", static_shapes=True)
def _mixed_atomic_orders(
    indices: torch.Tensor, counters: torch.Tensor, floats: torch.Tensor
) -> torch.Tensor:
    result = torch.empty_like(indices)
    for tile in hl.tile(indices.numel()):
        index = indices[tile]
        active = (tile.index < indices.numel()).to(torch.int32)
        a = hl.atomic_add(counters, [index], active)
        b = hl.atomic_add(counters, [index], active, sem="acq_rel")
        c = hl.atomic_add(counters, [index], 2)
        d = hl.atomic_add(floats, [index], 1.0)
        result[tile] = a + b + c + d.to(torch.int32)
    return result


def test_warp_atomic_preserves_strong_and_nonboolean_operations() -> None:
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _mixed_atomic_orders,
            (
                torch.empty((135,), dtype=torch.int32),
                torch.empty((4,), dtype=torch.int32),
                torch.empty((4,), dtype=torch.float32),
            ),
        )
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[128],
                num_threads=[128],
                cute_vector_widths=1,
                cute_warp_aggregated_atomics=True,
            )
        )
        calls = [
            node
            for node in ast.walk(ast.parse(bound.to_code(config)))
            if isinstance(node, ast.Call)
        ]
        assert (
            sum(
                isinstance(node.func, ast.Name)
                and node.func.id == "_cute_warp_fetch_add_int32"
                for node in calls
            )
            == 1
        )
        assert (
            sum(
                isinstance(node.func, ast.Attribute) and node.func.attr == "atomic_add"
                for node in calls
            )
            == 3
        )


@pytest.mark.parametrize("count", [8, 135, 1027])
@pytest.mark.parametrize(
    "block,threads,width", [(128, 128, 1), (256, 128, 1), (256, 128, 2)]
)
@skipUnlessBackends(["cute"])
def test_warp_atomic_tickets(count: int, block: int, threads: int, width: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    indices = (torch.arange(count, device=DEVICE) % 5).to(torch.int32)
    indices[::3] = 0
    increments = torch.arange(count, device=DEVICE) % 7 != 0
    initial = torch.tensor([-100, 10, 1000, -5, 31], device=DEVICE, dtype=torch.int32)
    counters = initial.clone()
    bound = _tickets.bind((indices, increments, counters))
    bound.set_config(
        helion.Config(
            block_sizes=[block],
            num_threads=[threads],
            cute_vector_widths=width,
            cute_warp_aggregated_atomics=True,
        )
    )
    first, second = bound(indices, increments, counters)
    for expert in range(initial.numel()):
        selected = (indices == expert) & increments
        count_for_expert = int(selected.sum().item())
        start = int(initial[expert].item())
        assert int(counters[expert].item()) == start + 2 * count_for_expert
        tickets = torch.cat((first[selected], second[selected])).sort().values
        torch.testing.assert_close(
            tickets,
            torch.arange(
                start, start + 2 * count_for_expert, device=DEVICE, dtype=torch.int32
            ),
            atol=0,
            rtol=0,
        )
        assert bool(torch.all(first[selected] < second[selected]))
        zeros = (indices == expert) & ~increments
        assert bool(torch.all(first[zeros] <= second[zeros]))
        assert bool(torch.all(first[zeros] >= start))
        assert bool(torch.all(second[zeros] <= start + 2 * count_for_expert))
