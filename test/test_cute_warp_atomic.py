from __future__ import annotations

import ast

import pytest
import torch

from .cute_population_contracts import _target
import helion
from helion._compiler.cute import hoist_lane_invariant_reductions as reduction_hoist
from helion._compiler.cute.atomic_ops import _CUTE_FLOAT_ATOMIC_HELPERS
from helion._compiler.cute.atomic_ops import _CUTE_VECTOR_ATOMIC_HELPERS
from helion._compiler.cute.atomic_ops import _CUTE_WARP_FETCH_ADD_HELPER
from helion._compiler.cute.fuse_two_pass_loads import _is_store_call
from helion._compiler.cute.fuse_two_pass_loads import _store_tensor_roots
from helion._compiler.cute.lane_loops import statements as lane_statements
from helion._compiler.cute.lane_loops.owner_guards import _ATOMIC_HELPERS
from helion._compiler.cute.lane_loops.owner_guards import _VECTOR_ATOMIC_HELPERS
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


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


@helion.kernel(backend="cute", static_shapes=True)
def _increment_then_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([1], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.numel()):
        hl.atomic_add(x, [tile], 1, sem="relaxed")
        out[0] = x[tile].sum()
    return out


def test_warp_atomic_preserves_strong_and_nonboolean_operations() -> None:
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(
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
                and node.func.id == _CUTE_WARP_FETCH_ADD_HELPER
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


@pytest.mark.parametrize("warp", [False, True], ids=["cute_arch", "warp"])
def test_the_lane_split_declines_to_move_tickets_past_the_loads_they_feed(
    warp: bool,
) -> None:
    """Four elements per thread: splitting the lane loop into the sum's
    accumulate pass and the store's consume pass would run the tile's
    fetch-adds after the loads that read them (0 instead of 128).  The
    split's write check (``_is_store_call``) knows the warp-aggregated
    fetch-add for a write of ``x``, as it knows ``cute.arch.atomic_add``, and
    declines the split for both."""
    with mock_cuda_unavailable(), _target(), forbid_native_compile():
        bound = cpu_bind(_increment_then_sum, (torch.zeros(128, dtype=torch.int32),))
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[128],
                num_threads=[32],
                cute_vector_widths=1,
                cute_warp_aggregated_atomics=warp,
            )
        )
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match="synthetic-lane reduction would reorder a potentially aliasing write",
        ):
            bound.to_code(config)


def test_every_emitted_atomic_helper_is_a_write_of_its_tensor() -> None:
    """The atomic helpers ``atomic_ops`` emits are the owner guards' atomic
    helpers, and each, like ``cute.arch.atomic_add``, is an atomic for the
    lane statements and the lane-invariant reduction hoist and a write of its
    pointer's tensor for the write check of the lane split and the load
    passes (``_is_store_call``, ``_store_tensor_roots``)."""
    helpers = {
        *_CUTE_FLOAT_ATOMIC_HELPERS.values(),
        *_CUTE_VECTOR_ATOMIC_HELPERS.values(),
        _CUTE_WARP_FETCH_ADD_HELPER,
    }
    assert helpers == _ATOMIC_HELPERS | _VECTOR_ATOMIC_HELPERS
    for callee in [*sorted(helpers), "cute.arch.atomic_add"]:
        call = ast.parse(f"{callee}((x.iterator + i).llvm_ptr, v)", mode="eval").body
        assert isinstance(call, ast.Call)
        assert lane_statements._is_atomic_call(call), callee
        assert reduction_hoist._is_atomic_call(call), callee
        assert _is_store_call(call), callee
        assert _store_tensor_roots(call, {"x"}) == {"x"}, callee


# Nine cases: the counts end in one partial warp (8), in a partial warp
# after full ones inside a block (135) and in a partial block after full ones
# (1027); the configs give each thread one element, two through a lane loop,
# or two as one vector.  Which lanes of a warp are live, and so share a fetch
# and order their tickets, depends on both axes together.
@pytest.mark.cuda
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
