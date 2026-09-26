"""Integer range admission before materialization enables native half/BF16 MMA."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_signed_bitfield_bf16 import _graph

import helion
from helion._compiler.cute.materialize_operand import _exact_integer_cast
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from torch._ops import OpOverload


def _cast(
    source_dtype: torch.dtype,
    destination_dtype: torch.dtype,
    target: OpOverload,
) -> Node:
    graph = Graph()
    source = graph.placeholder("source")
    # Even an all-zero sample is not proof of the runtime integer range.
    source.meta["val"] = torch.zeros((2, 8), dtype=source_dtype)
    if target is torch.ops.prims.convert_element_type.default:
        value = graph.call_function(target, (source, destination_dtype))
    else:
        value = graph.call_function(target, (source,), {"dtype": destination_dtype})
    value.meta["val"] = torch.empty((2, 8), dtype=destination_dtype)
    return value


@pytest.mark.parametrize(
    "target",
    (torch.ops.aten._to_copy.default, torch.ops.prims.convert_element_type.default),
)
@pytest.mark.parametrize("destination", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "source,expected",
    (
        (torch.int8, True),
        (torch.uint8, True),
        (torch.int16, False),
        (torch.int32, False),
        (torch.int64, False),
        (torch.bool, True),
        (torch.float32, True),
    ),
)
def test_full_dtype_range_not_sample_values(
    source: torch.dtype,
    destination: torch.dtype,
    target: OpOverload,
    expected: bool,
) -> None:
    assert _exact_integer_cast(_cast(source, destination, target)) is expected


@pytest.mark.parametrize("destination", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("kind", ("low", "high", "byte"))
def test_signed_fields_survive_integer_widening(
    kind: str, destination: torch.dtype
) -> None:
    if kind == "low":
        case = _graph(0, 4)
    elif kind == "high":
        case = _graph(4, 4, shift_only=True)
    else:
        case = _graph(0, 8, identity=True)
    case.value.kwargs = {"dtype": destination}
    case.value.meta["val"] = torch.empty((2, 8), dtype=destination)
    assert case.value.args[0].meta["val"].dtype is torch.int32
    assert _exact_integer_cast(case.value)


@pytest.mark.parametrize(
    "fault",
    (
        "wide_load",
        "wrong_sign_threshold",
        "extra_integer_arithmetic",
        "foreign_graph_operand",
        "missing_input_type",
        "missing_output_type",
        "inconsistent_output_type",
        "extra_cast_keyword",
    ),
)
def test_unproved_or_stale_field_declines(fault: str) -> None:
    case = _graph(0, 4)
    integer = case.value.args[0]
    assert isinstance(integer, Node)
    if fault == "wide_load":
        case.source.meta["val"] = torch.empty((2, 8), dtype=torch.int16)
        case.load.meta["val"] = torch.empty((2, 8), dtype=torch.int16)
    elif fault == "wrong_sign_threshold":
        condition = integer.args[0]
        assert isinstance(condition, Node)
        condition.args = (condition.args[0], 7)
    elif fault == "extra_integer_arithmetic":
        changed = case.graph.call_function(torch.ops.aten.add.Scalar, (integer, 1))
        changed.meta["val"] = torch.empty((2, 8), dtype=torch.int32)
        case.value.args = (changed,)
    elif fault == "foreign_graph_operand":
        case.value.args = (_graph(0, 4).value.args[0],)
    elif fault == "missing_input_type":
        integer.meta.clear()
    elif fault == "missing_output_type":
        case.value.meta.clear()
    elif fault == "inconsistent_output_type":
        case.value.meta["val"] = torch.empty((2, 8), dtype=torch.float16)
    elif fault == "extra_cast_keyword":
        case.value.kwargs = {"dtype": torch.bfloat16, "non_blocking": True}
    else:
        raise AssertionError(fault)
    # A copied semantic annotation cannot rescue an invalid current graph.
    case.value.meta["cute_signed_bitfield_bf16"] = {
        "signed": True,
        "bounds": (-8, 7),
    }
    assert not _exact_integer_cast(case.value)


def test_whole_graph_copy_reproves_current_field() -> None:
    case = _graph(0, 4)
    copied = Graph()
    values: dict[Node, Node] = {}
    for node in case.graph.nodes:
        values[node] = copied.node_copy(node, lambda item: values[item])
    value = values[case.value]
    value.meta["cute_signed_bitfield_bf16"] = {"old_load": case.load}
    assert value.graph is not case.graph
    assert _exact_integer_cast(value)


def test_floating_arithmetic_recipe_is_unchanged() -> None:
    cast = _cast(torch.float32, torch.bfloat16, torch.ops.aten._to_copy.default)
    source = cast.args[0]
    assert isinstance(source, Node)
    product = cast.graph.call_function(torch.ops.aten.mul.Scalar, (source, 0.25))
    product.meta["val"] = torch.empty((2, 8), dtype=torch.float32)
    cast.args = (product,)
    assert _exact_integer_cast(product)
    assert _exact_integer_cast(cast)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _direct_cast_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            operand = b[inner, column].to(a.dtype)
            accumulator = torch.addmm(accumulator, a[row, inner], operand)
        output[row, column] = accumulator.to(a.dtype)
    return output


@pytest.mark.parametrize("destination", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "source,expected",
    (
        (torch.int8, True),
        (torch.uint8, True),
        (torch.int16, False),
        (torch.int32, False),
    ),
)
@skipUnlessBackends(["cute"])
def test_actual_recipe_admission_uses_integer_precision(
    source: torch.dtype, destination: torch.dtype, expected: bool
) -> None:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        kernel = helion.kernel(
            _direct_cast_rhs.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            cute_materialize_transformed_operands=True,
        )
        args = (
            torch.empty((16, 32), dtype=destination),
            torch.empty((32, 16), dtype=source),
        )
        bound = kernel._bind_isolated(args)
        assert (bound.env.cute_fission_plan is not None) is expected
