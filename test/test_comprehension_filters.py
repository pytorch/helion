from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion import exc
from helion._compiler.loop_dependency_checker import _holds_no_storage
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
def computed_literal_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for j in (0, 1):
            n = j + 1
            values = [i for i in (1, 2) if i == n]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def computed_static_range_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for j in hl.static_range(2):
            n = j + 1
            values = [i for i in (1, 2) if i == n]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def iteration_dependent_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for j in (1, 2):
            values = [i for i in (j,) if i > 0]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def heterogeneous_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [v for v in (1, None, 2) if v is not None]
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def mixed_literal_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [v for v in (1, "skip", 2) if v != "skip"]
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def heterogeneous_result(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [v for v in (1, "keep", 2) if v != 2]
        if values[1] == "keep":
            out[tile] = x[tile] * values[0]
    return out


@helion.kernel(autotune_effort="none")
def tensor_and_scalar_result(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [v for v in (x, 1) if v is not None]
        out[tile] = values[0][tile] * values[1]
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


@helion.kernel(autotune_effort="none")
def generator_as_filtered_element(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        values = [generator for _ in (1,) if 1]
        out[tile] = x[tile] * (2 if values[0] == (3, 4) else 1)
    return out


@helion.kernel(autotune_effort="none")
def generator_enclosing_filtered_dict(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 2))
        first = {0: len([i for i in (1, 2) if i > 0]) for _ in generator}  # noqa: B035
        second = {0: len([i for i in (1, 2) if i > 0]) for _ in generator}  # noqa: B035
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def generator_enclosing_nested_filtered_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 1))
        first = [len([i for i in (1, 2) if value]) for value in generator]
        second = [len([i for i in (1, 2) if value]) for value in generator]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def generator_enclosing_filtered_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 1))
        acc = x[tile] * 0
        for value in generator:
            selected = [i for i in (1, 2) if value]
            acc = acc + len(selected)
        for value in generator:
            selected = [i for i in (1, 2) if value]
            acc = acc + len(selected)
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def consumed_generator_enclosing_filtered_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 1))
        acc = x[tile] * 0
        for value in tuple(generator):
            selected = [i for i in (1, 2) if value]
            acc = acc + len(selected)
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def consumed_generator_in_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        first = [0 for _ in tuple(generator) if 1]
        second = [0 for _ in tuple(generator) if 1]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def consumed_generator_in_element(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        values = [len(tuple(generator)) for i in (1, 2) if i]
        out[tile] = x[tile] * (values[0] + 10 * values[1])
    return out


@helion.kernel(autotune_effort="none")
def host_generator_as_filtered_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    generator = (j + 0 for j in (3, 4))
    for tile in hl.tile(x.numel()):
        first = [0 for _ in generator if 1]
        second = [0 for _ in generator if 1]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def generator_as_filtered_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        first = [0 for _ in generator if 1]
        second = [0 for _ in generator if 1]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def generator_in_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (1, 2) if (j + 0 for j in (3, 4)) == (3, 4)]
        out[tile] = x[tile] * len(values)
    return out


@helion.kernel(autotune_effort="none")
def assigned_generator_in_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        values = [i for i in (1, 2) if generator == (3, 4)]
        out[tile] = x[tile] * len(values)
    return out


@helion.kernel(autotune_effort="none")
def indexed_generator_in_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        values = [i for i in (1, 2) if generator[0] == 3]
        out[tile] = x[tile] * len(values)
    return out


@helion.kernel(autotune_effort="none")
def nested_generator_in_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        values = [i for i in (1, 2) if [generator] == [(3, 4)]]
        out[tile] = x[tile] * len(values)
    return out


@helion.kernel(autotune_effort="none")
def mutate_comprehension_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        cached = [x[tile] * 0 for _ in (0, 1)]
        cached[0] = x[tile]
        cached[1] = x[tile] * 2
        out[tile] = cached[0] + cached[1]
    return out


@helion.kernel(autotune_effort="none")
def unpack_list_comprehensions(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        a, b = [i + 0 for i in (1, 2)]
        c, d = [i for i in (0, 3, 4) if i > 0]
        out[tile] = x[tile] * (a + b + c + d)
    return out


@helion.kernel(autotune_effort="none")
def mixed_sequence_tuple_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = x[tile] * 0
        for values in ([1, 2], (1, 2)):
            if values == (1, 2):
                acc = acc + x[tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def mixed_sequence_list_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = x[tile] * 0
        for values in ((1, 2), [1, 2]):
            if values == [1, 2]:
                acc = acc + x[tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def generator_tuple_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (i + 0 for i in (1, 2))
        if generator == (1, 2):
            out[tile] = x[tile] * 7
        else:
            out[tile] = x[tile] * 3
    return out


@helion.kernel(autotune_effort="none")
def generator_list_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (i + 0 for i in (1, 2))
        if generator == [1, 2]:
            out[tile] = x[tile] * 7
        else:
            out[tile] = x[tile] * 3
    return out


@helion.kernel(autotune_effort="none")
def filtered_list_tuple_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (1, 2) if i > 0]
        if values == (1, 2):
            out[tile] = x[tile] * 7
        else:
            out[tile] = x[tile] * 3
    return out


@helion.kernel(autotune_effort="none")
def filtered_list_comparisons(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        first = [i for i in (1, 2) if i > 0]
        selected = [i for i in (1, 2) if first == [1, 2]]
        different_type = [i for i in (3, 4) if first == (1, 2)]
        acc = x[tile] * 0
        for value in selected:
            acc = acc + x[tile] * value
        for value in different_type:
            acc = acc + x[tile] * value
        out[tile] = acc
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


def test_filtered_sequence_forms() -> None:
    bound = static_range_filter.bind((torch.ones(32),))
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


@pytest.mark.parametrize(
    "kernel", [computed_literal_filter, computed_static_range_filter]
)
def test_computed_filter_per_iteration(kernel: helion.Kernel) -> None:
    bound = kernel.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert graph.count("aten.mul.Tensor") == 2
    assert all(f", {factor})" in graph for factor in (1, 2))


@pytest.mark.parametrize(
    "kernel", [iteration_dependent_iterable, heterogeneous_filter, mixed_literal_filter]
)
def test_filtered_sequence_types_per_iteration(kernel: helion.Kernel) -> None:
    bound = kernel.bind((torch.ones(32),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    if kernel is iteration_dependent_iterable:
        assert graph.count("aten.mul.Tensor") == 2
    else:
        assert ", 3)" in graph


def test_filtered_heterogeneous_result() -> None:
    bound = heterogeneous_result.bind((torch.ones(32),))
    graph = bound.host_function.device_ir.graphs[0].graph
    assert any(node.target is hl.store for node in graph.nodes)


def test_filtered_mixed_result_preserves_tensor_alias_information() -> None:
    bound = tensor_and_scalar_result.bind((torch.ones(32),))
    comprehension = next(
        node
        for stmt in bound.host_function.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.ListComp)
    )
    assert isinstance(comprehension.elt, ast.Name)
    assert not _holds_no_storage(comprehension.elt)


def test_changing_comprehension_length_is_rejected() -> None:
    with pytest.raises(exc.StatementNotSupported, match="changing lengths"):
        changing_comprehension_length.bind((torch.ones(32),))


def test_filtered_body_preserves_subscript_rewrites() -> None:
    bound = filtered_multidimensional_tensor.bind((torch.ones(16, 4),))
    graph = str(bound.host_function.device_ir.graphs[0].graph)
    assert graph.count("aten.mul.Tensor") == 2


def test_filtered_generator_expressions_are_rejected() -> None:
    with pytest.raises(
        exc.StatementNotSupported,
        match="Filtered generator expressions inside device loops",
    ):
        filtered_generator.bind((torch.ones(32),))


@pytest.mark.parametrize(
    "kernel",
    [generator_enclosing_nested_filtered_list, generator_enclosing_filtered_dict],
)
def test_generator_enclosing_nested_filtered_list_is_rejected(
    kernel: helion.Kernel,
) -> None:
    with pytest.raises(
        exc.StatementNotSupported,
        match="Generator values in comprehensions containing filtered lists",
    ):
        kernel.bind((torch.ones(32),))


@pytest.mark.parametrize(
    "kernel",
    [generator_enclosing_filtered_list, consumed_generator_enclosing_filtered_list],
)
def test_generator_enclosing_filtered_list_is_rejected(kernel: helion.Kernel) -> None:
    with pytest.raises(
        exc.StatementNotSupported,
        match="Generator values in loops containing filtered comprehensions",
    ):
        kernel.bind((torch.ones(32),))


@pytest.mark.parametrize(
    "kernel", [consumed_generator_in_iterable, consumed_generator_in_element]
)
def test_consumed_generator_in_filtered_list_is_rejected(kernel: helion.Kernel) -> None:
    with pytest.raises(
        exc.StatementNotSupported, match="Generator values in filtered comprehensions"
    ):
        kernel.bind((torch.ones(32),))


def test_generator_as_filtered_element_is_rejected() -> None:
    with pytest.raises(
        exc.StatementNotSupported,
        match="Generator values in filtered comprehensions",
    ):
        generator_as_filtered_element.bind((torch.ones(32),))


@pytest.mark.parametrize(
    "kernel", [generator_as_filtered_iterable, host_generator_as_filtered_iterable]
)
def test_generator_as_filtered_iterable_is_rejected(kernel: helion.Kernel) -> None:
    with pytest.raises(
        exc.StatementNotSupported,
        match="Generator values in filtered comprehensions",
    ):
        kernel.bind((torch.ones(32),))


@pytest.mark.parametrize(
    "kernel",
    [
        generator_in_filter,
        assigned_generator_in_filter,
        indexed_generator_in_filter,
        nested_generator_in_filter,
    ],
)
def test_generator_values_in_filter_are_rejected(kernel: helion.Kernel) -> None:
    with pytest.raises(
        exc.StatementNotSupported, match="Generator values in filtered comprehensions"
    ):
        kernel.bind((torch.ones(32),))


def test_mutate_comprehension_list() -> None:
    bound = mutate_comprehension_list.bind((torch.ones(32),))
    assert bound.host_function is not None
    graph = bound.host_function.device_ir.graphs[0].graph
    assert any(node.target is torch.ops.aten.add.Tensor for node in graph.nodes)


def test_unpack_list_comprehensions() -> None:
    bound = unpack_list_comprehensions.bind((torch.ones(32),))
    assert bound.host_function is not None
    graph = bound.host_function.device_ir.graphs[0].graph
    factors = [
        node.args[1] for node in graph.nodes if node.target is torch.ops.aten.mul.Tensor
    ]
    assert factors == [10]


@pytest.mark.parametrize(
    "kernel", [mixed_sequence_tuple_comparison, mixed_sequence_list_comparison]
)
def test_mixed_sequence_comparisons(kernel: helion.Kernel) -> None:
    bound = kernel.bind((torch.ones(32),))
    assert bound.host_function is not None
    graph = bound.host_function.device_ir.graphs[0].graph
    assert sum(node.target is torch.ops.aten.add.Tensor for node in graph.nodes) == 1


@pytest.mark.parametrize(
    "kernel", [generator_tuple_comparison, generator_list_comparison]
)
def test_generator_comparisons_are_rejected(kernel: helion.Kernel) -> None:
    with pytest.raises(
        exc.StatementNotSupported, match="Generator equality comparisons"
    ):
        kernel.bind((torch.ones(32),))


def test_filtered_list_tuple_comparison() -> None:
    bound = filtered_list_tuple_comparison.bind((torch.ones(32),))
    assert bound.host_function is not None
    graph = bound.host_function.device_ir.graphs[0].graph
    factors = [
        node.args[1] for node in graph.nodes if node.target is torch.ops.aten.mul.Tensor
    ]
    assert factors == [3]


def test_filtered_list_comparisons() -> None:
    bound = filtered_list_comparisons.bind((torch.ones(32),))
    assert bound.host_function is not None
    graph = bound.host_function.device_ir.graphs[0].graph
    factors = [
        node.args[1] for node in graph.nodes if node.target is torch.ops.aten.mul.Tensor
    ]
    assert factors == [0, 1, 2]


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
def test_mutate_comprehension_list_cuda() -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(mutate_comprehension_list(x), x * 3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unpack_list_comprehensions_cuda() -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(unpack_list_comprehensions(x), x * 10)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "kernel", [mixed_sequence_tuple_comparison, mixed_sequence_list_comparison]
)
def test_mixed_sequence_comparisons_cuda(kernel: helion.Kernel) -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(kernel(x), x)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_filtered_list_tuple_comparison_cuda() -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(filtered_list_tuple_comparison(x), x * 3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_filtered_list_comparisons_cuda() -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(filtered_list_comparisons(x), x * 3)


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "kernel", [computed_literal_filter, computed_static_range_filter]
)
def test_computed_filter_per_iteration_cuda(kernel: helion.Kernel) -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(kernel(x), x * 3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_and_scalar_result_cuda() -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(tensor_and_scalar_result(x), x)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "kernel", [iteration_dependent_iterable, heterogeneous_filter, mixed_literal_filter]
)
def test_filtered_sequence_types_per_iteration_cuda(kernel: helion.Kernel) -> None:
    x = torch.randn(128, device=DEVICE)
    torch.testing.assert_close(kernel(x), x * 3)
