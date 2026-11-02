"""Bounded Int64 selection keeps full values when indices are unobserved."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_topk import _simulate_network_topk_fragment

import helion
from helion._compiler.autotuner_heuristics.cute_fragment_topk_network import (
    network_topk_supported,
)
from helion._compiler.cute.ordered_selection import selection_values_only
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with _mock_cuda_unavailable(), _forbid_native_compile():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _value_selection(
    x: torch.Tensor, k: hl.constexpr, largest: hl.constexpr, kind: hl.constexpr
):
    values = torch.empty((*x.shape[:-1], k), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        if kind == "topk":
            chosen = torch.topk(x[row, :, :], k, dim=-1, largest=largest).values
        elif kind == "sort":
            chosen = torch.sort(x[row, :, :], dim=-1, descending=largest).values
        else:
            chosen = torch.sort(
                x[row, :, :], dim=-1, descending=largest, stable=True
            ).values
        values[row, :, :] = chosen
    return values


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_indexed(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty((*x.shape[:-1], 3), dtype=x.dtype, device=x.device)
    control = torch.empty((*y.shape[:-1], 3), dtype=y.dtype, device=y.device)
    indices = torch.empty_like(control, dtype=torch.int64)
    for row in hl.tile(x.size(0)):
        output[row, :, :] = torch.topk(x[row, :, :], 3, dim=-1).values
        selected, selected_indices = torch.topk(y[row, :, :], 3, dim=-1, largest=False)
        control[row, :, :] = selected
        indices[row, :, :] = selected_indices
    return output, control, indices


def _integer_data(n: int) -> torch.Tensor:
    special = [
        torch.iinfo(torch.int64).min,
        torch.iinfo(torch.int64).max,
        0,
        -1,
        1,
        torch.iinfo(torch.int64).min,
        torch.iinfo(torch.int64).max,
        2**62,
        -(2**62),
        0,
        -2,
        2,
    ]
    values = (special * ((n + len(special) - 1) // len(special)))[:n]
    return torch.tensor(values, dtype=torch.int64).repeat(2, 3, 1)


def _source(x: torch.Tensor, k: int, largest: bool, kind: str, enabled: bool):
    with _target():
        bound = _cpu_bind(_value_selection, (x, k, largest, kind))
        config = bound.config_spec.default_config()
        config.config.update(cute_fragment_threads=32)
        config.config["cute_fragment_topk_network"] = enabled
        return bound.to_code(config), config


@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "n,k,kind",
    [(1, 1, "topk"), (17, 7, "topk"), (17, 17, "sort"), (33, 33, "stable_sort")],
)
def test_int64_value_network_preserves_all_bits(
    largest: bool, n: int, k: int, kind: str
) -> None:
    x = _integer_data(n)
    before = x.clone()
    code, config = _source(x, k, largest, kind, True)
    actual = torch.empty((*x.shape[:-1], k), dtype=torch.int64)
    _simulate_network_topk_fragment(
        code,
        {"x": x},
        {"values": actual},
        (x.size(0) + config.block_sizes[0] - 1) // config.block_sizes[0],
        threads=32,
    )
    expected = torch.sort(x, dim=-1, descending=largest, stable=True).values[..., :k]
    assert torch.equal(actual, expected)
    assert torch.equal(x, before)
    assert "fragment_topk_previous_rank" not in code
    assert "fragment_topk_selected_column" not in code
    if not largest:
        assert any(
            isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert)
            for node in ast.walk(ast.parse(code))
        )


def test_int64_value_network_skips_unobserved_index_storage() -> None:
    x = _integer_data(17)
    before, _config = _source(x, 7, True, "topk", False)
    after, _config = _source(x, 7, True, "topk", True)

    def allocations(code: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allocate_tensor"
            and ast.unparse(node.args[0]) == "cutlass.Int64"
            for node in ast.walk(ast.parse(code))
        )

    assert allocations(after) == allocations(before) - 1


@pytest.mark.parametrize(
    "use",
    [
        "value",
        "index",
        "tuple",
        "value_and_tuple",
        "cross_graph",
        "shape",
        "bool_index",
        "none",
    ],
)
def test_value_only_selection_proof(use: str) -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty(17, dtype=torch.int64)
    selection = graph.call_function(
        torch.ops.aten.topk.default, (source, 3, -1, True, True)
    )
    if use in ("value", "value_and_tuple"):
        value = graph.call_function(operator.getitem, (selection, 0))
        graph.output(value if use == "value" else (value, selection))
    elif use == "index":
        graph.output(graph.call_function(operator.getitem, (selection, 1)))
    elif use == "tuple":
        graph.output(selection)
    elif use == "cross_graph":
        other = torch.fx.Graph()
        other.call_function(operator.getitem, (selection, 0))
    elif use == "shape":
        graph.call_function(torch.ops.aten.sym_size.int, (selection, 0))
    elif use == "bool_index":
        graph.call_function(operator.getitem, (selection, False))
    assert selection_values_only(selection) is (use == "value")


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.int32])
@pytest.mark.parametrize(
    "n,values_only,expected",
    [
        (17, False, False),
        (17, True, True),
        (1024, True, True),
        (1025, True, False),
        (None, True, False),
    ],
)
def test_value_network_bounded_dtype_admission(dtype, n, values_only, expected):
    if dtype != torch.int64:
        expected = n is not None and n <= 1024
    assert network_topk_supported(dtype, n, 3, values_only=values_only) is expected


def test_int64_observed_indices_keep_serial_lowering() -> None:
    x = torch.arange(2 * 3 * 17).reshape(2, 3, 17).float()
    y = _integer_data(17)
    with _target():
        bound = _cpu_bind(_mixed_indexed, (x, y))
        config = bound.config_spec.default_config()
        config.config.update(cute_fragment_topk_network=True, cute_fragment_threads=32)
        code = bound.to_code(config)
    actual = {
        "output": torch.empty((2, 3, 3)),
        "control": torch.empty((2, 3, 3), dtype=torch.int64),
        "indices": torch.empty((2, 3, 3), dtype=torch.int64),
    }
    _simulate_network_topk_fragment(
        code,
        {"x": x, "y": y},
        actual,
        (2 + config.block_sizes[0] - 1) // config.block_sizes[0],
        threads=32,
    )
    expected_indices = torch.argsort(y, dim=-1, stable=True)[..., :3]
    assert torch.equal(actual["indices"], expected_indices)
    assert torch.equal(actual["control"], y.gather(-1, expected_indices))
    assert "fragment_topk_previous_rank" in code


@pytest.mark.parametrize("largest", [False, True])
def test_float32_network_retains_indexed_value_recovery(largest: bool) -> None:
    x = (torch.arange(2 * 3 * 17).reshape(2, 3, 17) % 7).float()
    x[..., 0] = -0.0
    x[..., 1] = 0.0
    code, config = _source(x, 7, largest, "topk", True)
    actual = torch.empty((2, 3, 7))
    _simulate_network_topk_fragment(
        code,
        {"x": x},
        {"values": actual},
        (2 + config.block_sizes[0] - 1) // config.block_sizes[0],
        threads=32,
    )
    expected = torch.sort(x, dim=-1, descending=largest, stable=True).values[..., :7]
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert "fragment_topk_selected_column" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _encoded_probability_sort(probabilities: torch.Tensor, original_ids: torch.Tensor):
    values = torch.empty_like(probabilities)
    ids = torch.empty_like(original_ids)
    for row in hl.tile(probabilities.size(0)):
        positive_bits = probabilities[row, :, :].view(torch.int32) & 0x7FFFFFFF
        keys = (positive_bits.to(torch.int64) << 32) | (
            0x7FFFFFFF - original_ids[row, :, :].to(torch.int64)
        )
        sorted_keys = torch.sort(keys, dim=-1, descending=True).values
        values[row, :, :] = (sorted_keys >> 32).to(torch.int32).view(torch.float32)
        ids[row, :, :] = (0x7FFFFFFF - (sorted_keys & 0xFFFFFFFF)).to(torch.int32)
    return values, ids


def test_int64_network_preserves_encoded_probability_and_id_fields() -> None:
    probabilities = (torch.arange(2 * 3 * 17).reshape(2, 3, 17) % 7).float() / 8
    probabilities[..., 0] = -0.0
    original_ids = torch.arange(17, dtype=torch.int32).flip(0).repeat(2, 3, 1)
    with _target():
        bound = _cpu_bind(_encoded_probability_sort, (probabilities, original_ids))
        config = bound.config_spec.default_config()
        config.config.update(cute_fragment_topk_network=True, cute_fragment_threads=32)
        code = bound.to_code(config)
    actual = {
        "values": torch.empty_like(probabilities),
        "ids": torch.empty_like(original_ids),
    }
    _simulate_network_topk_fragment(
        code,
        {"probabilities": probabilities, "original_ids": original_ids},
        actual,
        (2 + config.block_sizes[0] - 1) // config.block_sizes[0],
        threads=32,
    )
    keys = ((probabilities.view(torch.int32) & 0x7FFFFFFF).to(torch.int64) << 32) | (
        0x7FFFFFFF - original_ids.to(torch.int64)
    )
    sorted_keys = keys.sort(dim=-1, descending=True).values
    expected_values = (sorted_keys >> 32).to(torch.int32).view(torch.float32)
    expected_ids = (0x7FFFFFFF - (sorted_keys & 0xFFFFFFFF)).to(torch.int32)
    assert torch.equal(
        actual["values"].view(torch.int32), expected_values.view(torch.int32)
    )
    assert torch.equal(actual["ids"], expected_ids)


def test_int64_network_default_and_population_prefix_stay_legacy() -> None:
    import random

    from test.cute_population_contracts import checked_initial_population

    from helion._compiler.autotuner_heuristics import cute_fragment_topk_network
    from helion.autotuner.pattern_search import PatternSearch

    original_support = cute_fragment_topk_network.network_topk_supported

    def old_support(dtype, n, k, *, values_only=False):
        return original_support(dtype, n, k)

    args = (_integer_data(17), 7, True, "topk")
    with _target():
        with patch.object(
            cute_fragment_topk_network, "network_topk_supported", old_support
        ):
            before = _cpu_bind(_value_selection, args)
        after = _cpu_bind(_value_selection, args)
    assert before.config_spec.default_config() == after.config_spec.default_config()
    assert (
        before.config_spec.compiler_seed_configs
        == after.config_spec.compiler_seed_configs
    )
    assert before.to_code(before.config_spec.default_config()) == after.to_code(
        after.config_spec.default_config()
    )
    populations = []
    states = []
    for bound in (before, after):
        with bound.env:
            random.seed(31)
            search = PatternSearch(bound, args, initial_population=20)
            population = checked_initial_population(search)
            populations.append(
                [
                    dict(search.config_gen.canonicalize_flat(row)[1])
                    for row in population
                ]
            )
            states.append(random.getstate())
    assert states[0] == states[1]
    # New deferred owners append witnesses after the unchanged legacy prefix.
    assert populations[1][: len(populations[0])] == populations[0]
    suffix = populations[1][len(populations[0]) :]
    assert len(suffix) == 2
    assert suffix[0]["cute_fragment_topk_network"] is True
    assert suffix[0].get("cute_fragment_cta_topk", False) is False
    assert suffix[1]["cute_fragment_topk_network"] is True
    assert suffix[1]["cute_fragment_cta_topk"] is True
