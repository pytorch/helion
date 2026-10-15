"""Stable ordering through row and complete-fragment CuTe selection."""

from __future__ import annotations

import copy

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_topk import _allocating_topk
from test.test_cute_topk import _has_register_selection
from test.test_cute_topk import _simulate_network_topk_fragment

import helion
from helion._compiler.cute.ordered_selection import ordered_selection_args
from helion._compiler.cute.ordered_selection import selection_args
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_stable_sort(x: torch.Tensor, descending: hl.constexpr):
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    for row in hl.tile(x.size(0)):
        value, index = torch.sort(x[row, :], dim=-1, descending=descending, stable=True)
        values[row, :] = value
        indices[row, :] = index
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_stable_sort(x: torch.Tensor, descending: hl.constexpr):
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    for row in hl.tile(x.size(0)):
        value, index = torch.sort(
            x[row, :, :], dim=-1, descending=descending, stable=True
        )
        values[row, :, :] = value
        indices[row, :, :] = index
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _canonical_stable_sort(probabilities: torch.Tensor, original_ids: torch.Tensor):
    values = torch.empty_like(probabilities)
    indices = torch.empty_like(original_ids)
    for row in hl.tile(probabilities.size(0)):
        ids, id_order = torch.sort(original_ids[row, :, :], dim=-1, stable=True)
        ordered = torch.gather(probabilities[row, :, :], -1, id_order)
        sorted_values, value_order = torch.sort(
            ordered, dim=-1, descending=True, stable=True
        )
        values[row, :, :] = sorted_values
        indices[row, :, :] = torch.gather(ids, -1, value_order)
    return values, indices


@pytest.mark.parametrize("stable", [True, False, None])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_stable_sort_argument_schema(stable: bool | None, dim: int) -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    node = graph.call_function(
        torch.ops.aten.sort.stable,
        (source,),
        {"stable": stable, "dim": dim, "descending": True},
    )
    assert ordered_selection_args(node) == (source, None, dim, True, True)
    assert selection_args(node) == (
        (source, None, True, True) if dim in (-1, 1) else None
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("descending", [False, True])
def test_stable_sort_row_owner(descending: bool) -> None:
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _row_stable_sort, (torch.empty((3, 17), dtype=torch.bfloat16), descending)
        )
        assert bound.config_spec.cute_topk_search_enabled
        config = bound.config_spec.default_config()
        config.config.update(
            cute_topk_rank_mode="ordinal", cute_topk_key_dtype="float32_native"
        )
        source = bound.to_code(config)
    assert _has_register_selection(source)
    assert "topk_native_key" not in source
    assert source.count("@cute.kernel") == 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("network", [False, True])
def test_stable_sort_fragment_values(dtype, descending, network):
    x = ((torch.arange(2 * 3 * 17).reshape(2, 3, 17) * 7) % 13 - 6).to(dtype)
    if dtype == torch.float32:
        x[:, :, :5] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0]
        )
    else:
        x[:, :, 0] = torch.iinfo(torch.int32).min
        x[:, :, 1] = torch.iinfo(torch.int32).max
    original = x.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_stable_sort, (x, descending))
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_topk_network=network, cute_fragment_threads=32
        )
        source = bound.to_code(config)
    values = torch.empty_like(x)
    indices = torch.full_like(x, -99, dtype=torch.int64)
    _simulate_network_topk_fragment(
        source,
        {"x": x},
        {"values": values, "indices": indices},
        (2 + config.block_sizes[0] - 1) // config.block_sizes[0],
        32,
    )
    expected = torch.argsort(x, dim=-1, descending=descending, stable=True)
    assert torch.equal(indices, expected)
    assert torch.equal(
        values.view(torch.int32), original.view(torch.int32).gather(-1, expected)
    )
    assert torch.equal(x.view(torch.int32), original.view(torch.int32))
    assert ("fragment_topk_previous_rank" not in source) == network


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("network", [False, True])
def test_canonical_probability_sort_retains_original_id_ties(network: bool) -> None:
    ids = ((torch.arange(17) * 7) % 17).to(torch.int32).expand(2, 3, 17).clone()
    probabilities = (ids % 5).float()
    original_probabilities, original_ids = probabilities.clone(), ids.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_canonical_stable_sort, (probabilities, ids))
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_topk_network=network, cute_fragment_threads=32
        )
        source = bound.to_code(config)
    values = torch.empty_like(probabilities)
    indices = torch.full_like(ids, -99)
    _simulate_network_topk_fragment(
        source,
        {"probabilities": probabilities, "original_ids": ids},
        {"values": values, "indices": indices},
        (2 + config.block_sizes[0] - 1) // config.block_sizes[0],
        32,
    )
    ordered_ids, id_order = original_ids.sort(dim=-1, stable=True)
    ordered_probabilities = original_probabilities.gather(-1, id_order)
    expected_values, value_order = ordered_probabilities.sort(
        dim=-1, descending=True, stable=True
    )
    expected_ids = ordered_ids.gather(-1, value_order)
    assert torch.equal(indices, expected_ids)
    assert torch.equal(values, expected_values)
    assert torch.equal(probabilities, original_probabilities)
    assert torch.equal(ids, original_ids)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [4, 17])
def test_small_row_topk_seed_remains_a_hint(dtype, rows):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(
            _allocating_topk, (torch.empty((rows, 512), dtype=dtype), 10, True)
        )
    spec = bound.config_spec
    assert spec.compiler_default_config is None
    assert spec.default_config()["cute_topk_lanes_per_row"] == 16
    seeds = [
        seed for seed in spec.compiler_seed_configs if "cute_topk_lanes_per_row" in seed
    ]
    hints = [
        seed
        for seed in seeds
        if seed["cute_topk_lanes_per_row"] == 32
        and seed["cute_topk_rows_per_block"] == 1
        and seed["cute_topk_vector_width"] == 4
        and seed["cute_topk_key_encoder"] == "dsl"
    ]
    assert bool(hints) == (rows == 4)
    if hints:
        assert seeds[-1] == hints[0]
        normalized = copy.deepcopy(hints[0])
        spec.normalize(normalized)
        assert normalized == hints[0]
