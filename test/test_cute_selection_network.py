"""CPU semantics and Helion tracing for tensor selection decompositions."""

from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.cute.selection_coarse import coarse_rank_full_keys
from helion._compiler.cute.selection_coarse import coarse_rank_guard
from helion._compiler.cute.selection_coarse import coarse_rank_keys
from helion._compiler.cute.selection_coarse import coarse_rank_recover
from helion._compiler.cute.selection_coarse import trace_coarse_rank_selection
from helion._compiler.cute.selection_network import selection_network
from helion._compiler.device_ir import IfGraphInfo
import helion.language as hl


def _expected(
    keys: torch.Tensor, k: int, groups_per_result: int, mode: str = "distributed"
) -> torch.Tensor:
    groups, registers = keys.shape
    rows = groups // groups_per_result
    ordered = (
        keys.reshape(rows, groups_per_result * registers)
        .sort(dim=1, descending=True)
        .values[:, :k]
    )
    if mode == "replicated":
        return ordered[:, None, :].expand(-1, groups_per_result, -1).reshape(groups, k)
    if groups_per_result > k:
        return ordered.repeat(1, groups_per_result // k).reshape(groups, 1)
    return (
        ordered.reshape(rows, k // groups_per_result, groups_per_result)
        .transpose(1, 2)
        .reshape(groups, k // groups_per_result)
    )


@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
@pytest.mark.parametrize("schedule", ["sequential", "balanced"])
@pytest.mark.parametrize(
    "groups,registers,k",
    [(1, 8, 1), (1, 16, 4), (2, 1, 2), (8, 1, 4), (8, 2, 16), (4, 8, 4)],
)
def test_selection_network_families(groups, registers, k, network, schedule):
    generator = torch.Generator().manual_seed(31)
    keys = torch.randint(-100, 100, (groups, registers), generator=generator)
    actual = selection_network(keys, k, network, schedule)
    torch.testing.assert_close(actual, _expected(keys, k, groups))


@pytest.mark.parametrize("groups_per_result", [1, 2, 4, 8])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_selection_network_zero_one_principle(groups_per_result, k):
    # Check all Boolean inputs to the same eight-candidate network together.
    # Each consecutive set of groups is an independent selection problem.
    registers = 8 // groups_per_result
    keys = ((torch.arange(256)[:, None] >> torch.arange(8)) & 1).reshape(
        256 * groups_per_result, registers
    )
    actual = selection_network(
        keys, k, "compact_pruned", "balanced", groups_per_result=groups_per_result
    )
    torch.testing.assert_close(actual, _expected(keys, k, groups_per_result))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("mode", ["replicated", "distributed"])
def test_selection_network_extreme_keys_and_independent_results(dtype, mode):
    keys = torch.arange(256, dtype=dtype).reshape(32, 8) - 128
    if dtype == torch.float32:
        keys[0, 0] = -float("inf")
    else:
        keys[0, 0] = torch.iinfo(dtype).min
        keys[3, 7] = torch.iinfo(dtype).max
    actual = selection_network(
        keys, 4, "compact", "balanced", mode, groups_per_result=4
    )
    torch.testing.assert_close(actual, _expected(keys, 4, 4, mode))


def _positive_ranks(groups: int, registers: int, index_bits: int) -> torch.Tensor:
    index = torch.arange(groups * registers, dtype=torch.int32).reshape(
        groups, registers
    )
    mask = (1 << index_bits) - 1
    return 0x3F800000 + index * (mask + 1) + ((index * 197) & mask)


def _full_keys_reference(
    rank: torch.Tensor, index_bits: int, vector_width: int, subgroup: int
) -> torch.Tensor:
    group = torch.arange(rank.shape[0])[:, None] % subgroup
    slot = torch.arange(rank.shape[1])
    column = ((slot // vector_width) * subgroup + group) * vector_width
    column = column + slot % vector_width
    full = (rank.to(torch.int64) << index_bits) | ((1 << index_bits) - 1 - column)
    return torch.where(rank != -2147483648, full, -9223372036854775808)


@pytest.mark.parametrize(
    "case,full_fallback,payload_fallback",
    [
        ("unique", False, False),
        ("cutoff_collision", True, True),
        ("higher_collision", False, True),
        ("unsupported_other_row", True, True),
        ("insufficient_normal", True, True),
        ("lower_negative_and_subnormal", False, False),
        ("padding", False, False),
    ],
)
def test_coarse_rank_guard_is_exact_and_uniform(case, full_fallback, payload_fallback):
    rank = _positive_ranks(32, 8, 5)
    if case == "cutoff_collision":
        rank[3, 4] = rank[3, 5]
    elif case == "higher_collision":
        rank[3, 7] = rank[3, 6] + 1
    elif case == "unsupported_other_row":
        rank[31, 7] = 0x7F800000
    elif case == "insufficient_normal":
        rank[:4] = 1
    elif case == "lower_negative_and_subnormal":
        rank[0, 0] = -1
        rank[0, 1] = 1
    elif case == "padding":
        rank[0, 0] = -2147483648
    coarse = coarse_rank_keys(rank, 5, 2, 4)
    selected = selection_network(coarse, 4, groups_per_result=4)
    guard = coarse_rank_guard(rank, selected, 4, 5, 4)
    assert guard.ndim == 0
    assert guard.item() is full_fallback
    assert coarse_rank_guard(rank, selected, 4, 5, 4, True).item() is payload_fallback


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("k", [4, 16])
def test_coarse_rank_resident_recovery_preserves_bits(input_dtype, k):
    # Eight-bit residues pack four per Int32, including words whose high bit
    # is set. k=4 also exercises replicated selection with eight groups/row.
    rank = _positive_ranks(32, 32, 8)
    full = _full_keys_reference(rank, 8, 4, 8)
    keys = rank if input_dtype == torch.int32 else full
    torch.testing.assert_close(coarse_rank_full_keys(keys, 8, 4, 8), full)
    coarse = coarse_rank_keys(keys, 8, 4, 8)
    selected = selection_network(coarse, k, groups_per_result=8)
    assert not coarse_rank_guard(keys, selected, k, 8, 8).item()
    direct = coarse_rank_recover(keys, selected, k, 8, 4, 8, "direct")
    packed = coarse_rank_recover(keys, selected, k, 8, 4, 8, "packed")
    torch.testing.assert_close(direct, packed)
    actual = selection_network(packed, k, groups_per_result=8)
    torch.testing.assert_close(actual, _expected(full, k, 8))


@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("payload_only", [False, True])
def test_coarse_rank_conditional_graph_runs_both_branches(recovery, payload_only):
    graph = trace_coarse_rank_selection(
        4, 4, 4, torch.int32, "compact", "balanced", 4, 1, 4, recovery, payload_only
    )
    for fallback in (False, True):
        rank = _positive_ranks(4, 4, 4)
        if fallback:
            rank[3, 3] = 0x7F800000
        expected = _expected(_full_keys_reference(rank, 4, 1, 4), 4, 4)
        actual = graph(rank)
        if payload_only:
            actual, expected = actual & 15, expected & 15
        torch.testing.assert_close(actual, expected)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _direct_selection_expression(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), 4, 2), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        keys = x[row, :, :].reshape(4, 8)
        selected = selection_network(keys, 8, "compact_pruned", "balanced")
        out[row, :, :] = selected
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _direct_coarse_expression(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), 32, 1), dtype=torch.int64, device=x.device)
    for row in hl.grid(x.size(0)):
        keys = x[row, :, :].reshape(32, 4)
        coarse = coarse_rank_keys(keys, 4, 1, 4)
        selected = selection_network(
            coarse, 4, "compact", "balanced", groups_per_result=4
        )
        fallback = coarse_rank_guard(keys, selected, 4, 4, 4)
        if fallback:
            full = coarse_rank_full_keys(keys, 4, 1, 4)
            result = selection_network(
                full, 4, "compact", "balanced", groups_per_result=4
            )
        else:
            recovered = coarse_rank_recover(keys, selected, 4, 4, 1, 4, "packed")
            result = selection_network(
                recovered, 4, "compact", "balanced", groups_per_result=4
            )
        out[row, :, :] = result
    return out


@pytest.mark.parametrize("coarse", [False, True])
def test_selection_tensor_helpers_trace_directly_in_helion(coarse):
    kernel = _direct_coarse_expression if coarse else _direct_selection_expression
    x = (
        torch.ones((3, 32, 4), dtype=torch.int32)
        if coarse
        else torch.ones((3, 4, 8), dtype=torch.int64)
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, (x,))
    graphs = bound.host_function.device_ir.graphs
    targets = {node.target for info in graphs for node in info.graph.nodes}
    assert torch.ops.aten.topk.default not in targets
    assert torch.ops.aten.gather.default in targets
    assert torch.ops.aten.fmax.default in targets
    if coarse:
        assert any(isinstance(info, IfGraphInfo) for info in graphs)
