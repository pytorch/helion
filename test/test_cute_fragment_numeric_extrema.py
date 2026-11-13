from __future__ import annotations

import ast
import operator

import pytest
import torch

from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.computed_fragment import Fragment
from helion._compiler.cute.computed_fragment import FragmentCompiler
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_fragment_numeric_extrema import fragment_numeric_extremum

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("minimum", [False, True])
@forbid_cuda_work()
def test_numeric_extrema_preserve_broadcast_logical_domains(minimum):
    compiler = object.__new__(FragmentCompiler)
    compiler.pending_local_atomics = set()
    compiler.shape = lambda sizes: tuple(sizes)
    graph = torch.fx.Graph()
    left_node, right_node = graph.placeholder("left"), graph.placeholder("right")
    target = torch.ops.aten.fmin.default if minimum else torch.ops.aten.fmax.default
    node = graph.call_function(target, (left_node, right_node))
    node.meta.update(val=torch.empty((4, 8)), lowering=None)
    left = Fragment(
        (4, 8),
        torch.float32,
        lambda coordinates: "left",
        storage="left",
        logical_domain=lambda coordinates: (
            f"{coordinates[0]} < 3",
            f"{coordinates[1]} < 5",
        ),
    )
    right = Fragment(
        (1, 8),
        torch.float32,
        lambda coordinates: "right",
        storage="right",
        logical_domain=lambda coordinates: (f"{coordinates[1]} < 4",),
    )
    result = compiler.node(node, {left_node: left, right_node: right})
    assert isinstance(result, Fragment) and not result.pure
    assert FragmentCompiler.referenced_buffers([result]) == {"left", "right"}
    domain = result.domain(("row", "column"))
    observed = torch.tensor(
        [
            [
                all(
                    eval(term.expression, {}, {"row": row, "column": column})
                    for term in domain
                )
                for column in range(8)
            ]
            for row in range(4)
        ]
    )
    expected = (torch.arange(4)[:, None] < 3) & (torch.arange(8)[None, :] < 4)
    assert torch.equal(observed, expected)
    # Invalid physical lanes contain a large sentinel, so a downstream sum
    # using only the physical shape would produce a visibly different result.
    values = torch.where(expected, 1.0, 10000.0)
    assert torch.where(observed, values, 0.0).sum().item() == 12.0


@forbid_cuda_work()
def test_isnan_override_expression_preserves_ieee_classes():
    bits = torch.tensor(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFFC00042,
            0x3F800000,
            0xBF800000,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    values = bits.view(torch.float32)
    expression = CuteBackend().inductor_op_overrides().isnan("x")
    actual = eval(str(expression), {"operator": operator, "x": values})
    assert torch.equal(actual, torch.isnan(values))


@pytest.mark.parametrize("minimum", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@forbid_cuda_work()
def test_numeric_extrema_keep_complete_fragment_coordinates(minimum, dtype):
    # The broadcast operand and shared vote prevent a scalar-only load recipe
    # or a dead result from standing in for complete fragment emission.
    values = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e-38, -1e-38]
    ).to(dtype)
    x = values.repeat(3, 1)
    y = values.flip(0).clone()
    with cpu_render_target():
        bound = cpu_bind(fragment_numeric_extremum, (x, y, minimum))
        code = bound.to_code(bound.config_spec.default_config())
    calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
    if dtype is torch.float32:
        extrema = [
            node for node in calls if ast.unparse(node.func) == "_cute_numeric_extremum"
        ]
        assert extrema
        assert all(
            next(
                keyword.value.value
                for keyword in call.keywords
                if keyword.arg == "minimum"
            )
            is minimum
            for call in extrema
        )
    else:
        operation = "cute.math.min" if minimum else "cute.math.max"
        extrema = [node for node in calls if ast.unparse(node.func) == operation]
        # The full uint8 vote also lowers to a NaN-propagating max; distinguish
        # that reduction from the numeric, broadcast pointwise operation.
        assert any(
            next(
                keyword.value.value
                for keyword in call.keywords
                if keyword.arg == "propagate_nan"
            )
            is False
            for call in extrema
        )
    assert "fragment" in code
