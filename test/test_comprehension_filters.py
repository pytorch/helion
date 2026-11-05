from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
import helion.language as hl


@helion.kernel(autotune_effort="none")
def filtered_values(x: torch.Tensor, limit: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i + 1 for i in (0, 1, 2, 3) if i < limit if i % 2 == 0]
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for value in values:
            acc = acc + x[tile] * value
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def short_circuit(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [8 // i for i in (0, 1, 2, 4) if i != 0 and 8 // i >= 4]
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def chained_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if 0 < i <= 8 // i]
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def none_values(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [None for i in (0, 1, 2) if i > 0]
        if values[1] is None:
            out[tile] = x[tile]
    return out


@helion.kernel(autotune_effort="none")
def dynamic_filter(x: torch.Tensor, limit: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if i < limit]
        out[tile] = x[tile] * values[0]
    return out


@helion.kernel(autotune_effort="none")
def shadowed_target(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    i = 7
    for tile in hl.tile(x.numel()):
        values = [i + 1 for i in (0, 1, 2) if i > 0]
        out[tile] = x[tile] * (values[0] + values[1] + i)
    return out


@pytest.mark.parametrize("limit, factors", [(0, ()), (1, (1,)), (4, (1, 3))])
def test_filtered_values_bind(limit: int, factors: tuple[int, ...]) -> None:
    bound = filtered_values.bind((torch.ones(32), limit))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    for factor in factors:
        assert f", {factor})" in graph
    assert graph.count("aten.mul.Tensor") == len(factors)


def test_filter_short_circuits() -> None:
    bound = short_circuit.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert ", 12)" in graph


def test_chained_filter_short_circuits() -> None:
    bound = chained_filter.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert ", 3)" in graph


def test_none_is_an_element() -> None:
    bound = none_values.bind((torch.ones(32),))
    assert any(
        node.target is hl.store
        for node in bound.host_function.device_ir.graphs[0].graph.nodes
    )


def test_runtime_filter_is_rejected() -> None:
    with pytest.raises(exc.StatementNotSupported, match="compile-time constant"):
        dynamic_filter.bind((torch.ones(32), 2))


def test_comprehension_target_does_not_shadow_outer_value() -> None:
    bound = shadowed_target.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert ", 12)" in graph
