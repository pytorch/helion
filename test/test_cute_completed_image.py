from __future__ import annotations

import dataclasses
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from helion._compiler.cute.chained_completed_image import CompletedSource
from helion._compiler.cute.chained_completed_image import ImageProvenance
from helion._compiler.cute.chained_completed_image import discover_completed_images
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion.language import view_ops
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from helion._compiler.cute.contraction_region import ContractionRegion


def _fixture(width: int = 8, dtype: torch.dtype = torch.bfloat16):
    graph = Graph()

    def node(target, args, shape, kind=torch.float32, kwargs=None):
        result = graph.call_function(target, args, kwargs or {})
        result.meta["val"] = torch.zeros(shape, dtype=kind)
        return result

    a = node(dot, (None, None, None), (2 * width, width))
    b = node(dot, (None, None, None), (width, width))
    folded = node(
        torch.ops.aten.view.default, (a, [2, width, width]), (2, width, width)
    )
    permuted = node(
        torch.ops.aten.permute.default, (folded, [1, 2, 0]), (width, width, 2)
    )
    split = graph.call_function(view_ops.split, (permuted,))
    split.meta["val"] = (torch.zeros(width, width), torch.zeros(width, width))
    cropped = node(operator.getitem, (split, 0), (width, width))
    zero = node(
        torch.ops.aten.full.default,
        ([width, width], 0.0),
        (width, width),
        kwargs={"dtype": torch.float32},
    )
    left = node(view_ops.join, (cropped, zero), (width, width, 2))
    right = node(view_ops.join, (zero, b), (width, width, 2))
    joined = node(view_ops.join, (left, right), (width, width, 2, 2))
    swapped = node(
        torch.ops.aten.permute.default, (joined, [2, 0, 3, 1]), (2, width, 2, width)
    )
    cloned = node(
        torch.ops.aten.clone.default,
        (swapped,),
        (2, width, 2, width),
        kwargs={"memory_format": torch.contiguous_format},
    )
    matrix = node(
        torch.ops.aten.view.default,
        (cloned, [2 * width, 2 * width]),
        (2 * width, 2 * width),
    )
    typed = node(
        torch.ops.prims.convert_element_type.default,
        (matrix, dtype),
        (2 * width, 2 * width),
        dtype,
    )
    half_a = node(
        torch.ops.prims.convert_element_type.default,
        (cropped, dtype),
        (width, width),
        dtype,
    )
    half_b = node(
        torch.ops.prims.convert_element_type.default, (b, dtype), (width, width), dtype
    )
    doubled = node(view_ops.join, (half_b, half_b), (width, width, 2), dtype)
    doubled_permute = node(
        torch.ops.aten.permute.default, (doubled, [2, 0, 1]), (2, width, width), dtype
    )
    doubled_clone = node(
        torch.ops.aten.clone.default,
        (doubled_permute,),
        (2, width, width),
        dtype,
        kwargs={"memory_format": torch.contiguous_format},
    )
    doubled_matrix = node(
        torch.ops.aten.view.default,
        (doubled_clone, [2 * width, width]),
        (2 * width, width),
        dtype,
    )
    consumer = node(dot, (doubled_matrix, half_a, None), (2 * width, width))
    store = graph.output(typed)
    region = cast("ContractionRegion", SimpleNamespace(graph=graph))
    plan = ChainedMatmulPlan(
        0,
        (a, b, consumer),
        store,
        (),
        ((2 * width, width, width), (width, width, width), (2 * width, width, width)),
        dtype,
        128,
        region=region,
    )
    return plan, a, b, typed, zero, matrix


def _evaluate(node: Node, sources: dict[Node, torch.Tensor]) -> torch.Tensor:
    memo: dict[Node, Any] = dict(sources)

    def run(value: Any) -> Any:
        if isinstance(value, Node):
            if value not in memo:
                args = [run(arg) for arg in value.args]
                if value.target is view_ops.join:
                    memo[value] = torch.stack(args, dim=-1)
                elif value.target is view_ops.split:
                    memo[value] = tuple(args[0].unbind(-1))
                else:
                    assert callable(value.target)
                    memo[value] = value.target(*args, **value.kwargs)
            return memo[value]
        if isinstance(value, (list, tuple)):
            return type(value)(run(item) for item in value)
        return value

    return run(node)


@pytest.mark.parametrize("width", [2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_completed_cast_views_preserve_every_cell(width, dtype):
    plan, a, b, typed, _, _ = _fixture(width, dtype)
    images = discover_completed_images(plan)
    assert len(images) == 1
    image = images[0]
    image.check(plan)
    assert image.node is typed and len(image.operands) == 2
    assert len(image.publications) == 2
    assert image.allocated_bytes == 8 * width * width
    assert image.publications[0].extent == (width, width)
    assert image.publications[0].source.shape == (2 * width, width)
    torch.manual_seed(7)
    values = {a: torch.randn(2 * width, width), b: torch.randn(width, width)}
    # Preserve raw special-value bits and single cast association as well.
    values[a].flatten()[:4] = torch.tensor([0.0, -0.0, float("inf"), float("nan")])
    expected = _evaluate(typed, values)
    actual = torch.zeros(image.shape, dtype=dtype)
    writes = set()
    for publication in image.publications:
        for destination, source in publication.locations(image.shape[1]):
            assert destination not in writes
            writes.add(destination)
            actual.flatten()[destination] = (
                values[publication.source.node].flatten()[source].to(dtype)
            )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    for view in image.operands:
        expected_operand = _evaluate(view.operand, values)
        if view.operand_index == 1:
            expected_operand = expected_operand.T
        actual_operand = actual.flatten()[list(view.addresses)].reshape(
            view.physical_shape
        )
        assert torch.equal(
            actual_operand.view(torch.int16),
            expected_operand.contiguous().view(torch.int16),
        )
        assert max(view.addresses) < actual.numel() and min(view.addresses) >= 0
    assert image.operands[0].period == (width, width)
    assert image.operands[0].physical_shape == (2 * width, width)
    assert image.operands[1].transpose


@pytest.mark.parametrize(
    "mutation",
    [
        "negative_zero",
        "nonzero",
        "cast_twice",
        "arithmetic",
        "shape",
        "view_shape",
        "dtype",
        "effect",
    ],
)
def test_unproved_composition_is_not_an_image(mutation):
    plan, a, b, typed, zero, matrix = _fixture()
    if mutation == "negative_zero":
        zero.args = (zero.args[0], -0.0)
    elif mutation == "nonzero":
        zero.args = (zero.args[0], 1.0)
    elif mutation == "cast_twice":
        matrix.target = torch.ops.prims.convert_element_type.default
        matrix.args = (a, torch.float32)
    elif mutation == "arithmetic":
        matrix.target = torch.ops.aten.add.Tensor
        matrix.args = (a, b)
    elif mutation == "shape":
        matrix.meta["val"] = torch.zeros(8, 8)
    elif mutation == "view_shape":
        matrix.args = (matrix.args[0], [8, 32])
    elif mutation == "dtype":
        typed.args = (matrix, torch.float16)
    else:
        matrix.target = torch.ops.aten.copy_.default
        matrix.args = (a, b)
    assert discover_completed_images(plan) == ()


def test_early_consumer_cannot_read_late_publication():
    plan, a, b, _, _, _ = _fixture()
    consumer = plan.dots[-1]
    reordered = dataclasses.replace(plan, dots=(a, consumer, b))
    (image,) = discover_completed_images(reordered)
    assert [(use.stage, use.operand_index) for use in image.operands] == [(1, 1)]


@pytest.mark.parametrize(
    "mutation",
    [
        "graph_args",
        "graph_shape",
        "aliases",
        "threads",
        "foreign_plan",
        "publication",
        "address",
    ],
)
def test_image_receipt_rejects_changed_graph_plan_or_selection(mutation):
    plan, a, b, _, zero, _ = _fixture()
    (image,) = discover_completed_images(plan)
    if mutation == "graph_args":
        zero.args = (zero.args[0], 1.0)
    elif mutation == "graph_shape":
        a.meta["val"] = torch.zeros(8, 8)
    elif mutation == "aliases":
        plan.tensor_aliases["input"] = "foreign"
    elif mutation == "threads":
        object.__setattr__(plan, "threads", 64)
    elif mutation == "foreign_plan":
        plan = dataclasses.replace(plan)
    elif mutation == "publication":
        image = dataclasses.replace(
            image,
            publications=(
                dataclasses.replace(image.publications[0], extent=(16, 8)),
                *image.publications[1:],
            ),
        )
    else:
        image = dataclasses.replace(
            image,
            operands=(
                dataclasses.replace(image.operands[0], addresses=(4096,)),
                *image.operands[1:],
            ),
        )
    with pytest.raises(ValueError, match="changed"):
        image.check(plan)


def test_capacity_and_negative_indices_fail_closed():
    plan, a, b, typed, _, _ = _fixture()
    assert discover_completed_images(plan, capacity_bytes=64) == ()
    with pytest.raises(ValueError):
        discover_completed_images(plan, capacity_bytes=True)
    provenance = ImageProvenance(
        (
            CompletedSource(a, (16, 8), 1, "contraction"),
            CompletedSource(b, (8, 8), 2, "contraction"),
        ),
        maximum_elements=512,
    )
    matrix = provenance.image(typed)
    assert matrix is not None
    for coordinates in [(-1, 0), (16, 0), (0, 16), (0,)]:
        with pytest.raises(ValueError):
            matrix.at(coordinates)
