from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
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
def membership_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i + 1 for i in (0, 1, 2, 3) if i in (0, 2)]
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def static_range_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i + 1 for i in hl.static_range(4) if i % 2 == 0]
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def filtered_generator(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = tuple(i + 1 for i in (0, 1, 2, 3) if i % 2 == 0)
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def nested_boolean_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if (i != 0 and 8 // i >= 4) == 1]
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def sequence_truth_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (1, 2) if (i,) if not ()]
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
def dynamic_membership(x: torch.Tensor, limit: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if i in (limit, 5)]
        for value in values:
            out[tile] = x[tile] * value
    return out


@helion.kernel(autotune_effort="none")
def tensor_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if x[tile] > 0]
        for value in values:
            out[tile] = x[tile] * value
    return out


@helion.kernel(autotune_effort="none")
def identity_filter(x: torch.Tensor, limit: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (10000, 20000) if i is limit]
        for value in values:
            out[tile] = x[tile] * value
    return out


@helion.kernel(autotune_effort="none")
def shadowed_target(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    i = 7
    for tile in hl.tile(x.numel()):
        values = [i + 1 for i in (0, 1, 2) if i > 0]
        out[tile] = x[tile] * (values[0] + values[1] + i)
    return out


@helion.kernel(autotune_effort="none")
def filtered_tensors(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        pairs = ((0, 1), (1, 2), (2, 3))
        values = [x[tile] * factor for index, factor in pairs if index != 1]
        out[tile] = values[0] + values[1]
    return out


@helion.kernel(autotune_effort="none")
def repeated_comprehension(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for selected in (0, 2):
            values = [i + 1 for i in (0, 1, 2) if i == selected]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def repeated_static_range_comprehension(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for selected in hl.static_range(3):
            values = [i + 1 for i in (0, 1, 2) if i == selected]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def changing_comprehension_length(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for limit in hl.static_range(1, 4):
            values = [i for i in (0, 1, 2) if i < limit]
            for value in values:
                acc = acc + x[tile] * value
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def filtered_multidimensional_tensor(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        values = [x[tile] * i for i in (1, 2, 3) if i != 2]
        out[tile] = values[0] + values[1]
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


def test_membership_filter() -> None:
    bound = membership_filter.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert ", 4)" in graph


@pytest.mark.parametrize("kernel", [static_range_filter, filtered_generator])
def test_filtered_sequence_forms(kernel: helion.Kernel) -> None:
    bound = kernel.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert ", 4)" in graph


@pytest.mark.parametrize("kernel", [nested_boolean_filter, sequence_truth_filter])
def test_nested_constant_conditions(kernel: helion.Kernel) -> None:
    bound = kernel.bind((torch.ones(32),))
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


def test_runtime_membership_is_rejected() -> None:
    with pytest.raises(exc.StatementNotSupported, match="compile-time constant"):
        dynamic_membership.bind((torch.ones(32), 2))


def test_tensor_filter_is_rejected() -> None:
    with pytest.raises(exc.StatementNotSupported, match="compile-time constant"):
        tensor_filter.bind((torch.ones(32),))


def test_identity_filter_is_rejected() -> None:
    with pytest.raises(exc.StatementNotSupported, match="Identity comparisons"):
        identity_filter.bind((torch.ones(32), 10000))


def test_comprehension_target_does_not_shadow_outer_value() -> None:
    bound = shadowed_target.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert ", 12)" in graph


def test_filtered_tensor_bodies_are_traced_only_for_selected_elements() -> None:
    bound = filtered_tensors.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert graph.count("aten.mul.Tensor") == 2
    assert ", 1)" in graph
    assert ", 3)" in graph


def test_repeated_comprehension_uses_each_iteration_scope() -> None:
    bound = repeated_comprehension.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert graph.count("aten.mul.Tensor") == 2
    assert ", 1)" in graph
    assert ", 3)" in graph


def test_static_range_comprehension_uses_each_iteration_scope() -> None:
    bound = repeated_static_range_comprehension.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert graph.count("aten.mul.Tensor") == 3
    assert all(f", {factor})" in graph for factor in (1, 2, 3))


def test_changing_comprehension_length_is_rejected() -> None:
    with pytest.raises(exc.StatementNotSupported, match="changing lengths"):
        changing_comprehension_length.bind((torch.ones(32),))


def test_filtered_body_preserves_subscript_rewrites() -> None:
    bound = filtered_multidimensional_tensor.bind((torch.ones(16, 4),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert graph.count("aten.mul.Tensor") == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("limit, multiplier", [(0, 0), (1, 1), (4, 4)])
def test_filtered_values_cuda(limit: int, multiplier: int) -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(filtered_values(x, limit), x * multiplier)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_filtered_tensor_bodies_cuda() -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(filtered_tensors(x), x * 4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_filtered_multidimensional_tensor_cuda() -> None:
    x = torch.randn(16, 4, device=DEVICE)
    torch.testing.assert_close(filtered_multidimensional_tensor(x), x * 4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_repeated_static_range_comprehension_cuda() -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(repeated_static_range_comprehension(x), x * 6)
