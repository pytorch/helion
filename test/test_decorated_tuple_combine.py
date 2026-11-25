from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.device_ir import HelperFunctionGraphInfo
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEagerMark
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.triton_codegen import interpret_triton_selection
import helion.language as hl

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu_only


@helion.kernel()
def _pair_combine(left, right):
    return left[0] + right[0], left[1] + right[1]


def _with_options(left, right, *, i=0, j=1):
    return left[i] + right[i], left[j] + right[j]


@helion.kernel()
def _flat_combine(a, b, c, d):
    return a + c, b + d


def _scalar_add(left, right, *, scale=1, **options):
    return left + scale * right


def add_one(left, right):
    return (left[0] + right[0],)


def _singleton_sum(left, right, i=0):
    return (left[i] + right[i],)


def _flat_rest(ax, ay, bx, *rest):
    return ax + bx, ay + rest[0]


def _flat_options(a, b, *right, scale=1):
    return a + scale * right[0], b + scale * right[1]


def _singleton_op(a, b, op=torch.add, alpha=1):
    return (op(a, b, alpha=alpha),)


def _eager_packed_options(left, right, i=0, j=1, *, op=torch.add):
    return op(left[i], right[i]), op(left[j], right[j])


def _packed_three(left, right, i=1, j=2):
    return left[0] + right[0], left[i] + right[i], left[j] + right[j]


@helion.kernel(static_shapes=True)
def _combine(x, out, op: hl.constexpr):
    for tile in hl.tile(x.size(0)):
        values = x[tile]
        if op == "scan":
            a, b = hl.associative_scan(_pair_combine, (values, values + 1), 0)
        elif op == "flat_rest":
            a, b = hl.reduce(_flat_rest, (values, values + 1), 0)
        elif op == "flat_options":
            a, b = hl.reduce(_flat_options, (values, values + 1), 0)
        elif op == "singleton_op":
            (a,) = hl.reduce(_singleton_op, (values,), 0)
            b = 0
        elif op == "packed_three":
            a, b, c = hl.reduce(_packed_three, (values, values + 1, values - 1), 0)
            a = a + c
        elif op == "configured_reduce":
            a, b = hl.reduce(_with_options, (values, values + 1), 0)
        elif op == "scalar_reduce":
            a = hl.reduce(_scalar_add, values, 0)
            b = 0
        elif op == "singleton_reduce":
            (a,) = hl.reduce(_singleton_sum, (values,), 0)
            b = 0
        else:
            a, b = hl.reduce(_flat_combine, (values, values + 1), 0)
        out[tile] = a + b


@skipIfRefEagerMark("inspects device IR, which ref mode does not build")
@skipUnlessBackends(["triton", "cute"])
# Each signature exercises a distinct layout, default or decorated adapter path.
@pytest.mark.parametrize(
    "op, callback, count",
    [
        ("scan", "_pair_combine", 2),
        ("reduce", "_flat_combine", 2),
        ("flat_rest", "_flat_rest", 2),
        ("flat_options", "_flat_options", 2),
        ("singleton_op", "_singleton_op", 1),
        ("packed_three", "_packed_three", 3),
        ("configured_reduce", "_with_options", 2),
        ("singleton_reduce", "_singleton_sum", 1),
        ("scalar_reduce", "_scalar_add", 1),
    ],
)
def test_combine_traces_the_python_callback(op, callback, count):
    x = torch.ones(4)
    bound = cpu_bind(_combine, (x, torch.empty_like(x), op))
    assert bound.host_function is not None
    (helper,) = [
        g
        for g in bound.host_function.device_ir.graphs
        if isinstance(g, HelperFunctionGraphInfo)
    ]
    assert helper.original_function_name == callback
    assert (
        len(
            helper.graph.find_nodes(
                op="call_function", target=torch.ops.aten.add.Tensor
            )
        )
        == count
    )


@helion.kernel(static_shapes=True, ref_mode="eager")
def _eager_combine(x, out, other, op: hl.constexpr):
    for tile in hl.tile(x.size(0), block_size=4):
        if op == "scan":
            a, b = hl.associative_scan(_flat_combine, (x[tile], x[tile] + 1), 0)
            out[tile] = a
            other[tile] = b
        elif op == "variadic":
            a, b = hl.reduce(_flat_rest, (x[tile], x[tile] + 1), 0)
            out[0] = a
            other[0] = b
        elif op == "variadic_options":
            a, b = hl.reduce(_flat_options, (x[tile], x[tile] + 1), 0)
            out[0] = a
            other[0] = b
        elif op == "legacy_options":
            a, b = hl.reduce(_eager_packed_options, (x[tile], x[tile] + 1), 0)
            out[0] = a
            other[0] = b
        elif op == "singleton_op":
            (total,) = hl.reduce(_singleton_op, (x[tile],), 0)
            out[0] = total
        elif op == "defaults":
            a, b, c = hl.reduce(_packed_three, (x[tile], x[tile] + 1, x[tile] - 1), 0)
            out[0] = a + b
            other[0] = c
        else:
            (total,) = hl.reduce(add_one, (x[tile],), 0)
            out[0] = total


@pytest.mark.parametrize(
    "op, expected, extra",
    [
        ("scan", [1, 3, 7, 15], [2, 5, 10, 19]),
        ("variadic", [15, -73, -73, -73], [19, -73, -73, -73]),
        ("variadic_options", [15, -73, -73, -73], [19, -73, -73, -73]),
        ("legacy_options", [15, -73, -73, -73], [19, -73, -73, -73]),
        ("singleton_op", [15, -73, -73, -73], [-73] * 4),
        ("defaults", [34, -73, -73, -73], [11, -73, -73, -73]),
        ("singleton", [15, -73, -73, -73], [-73] * 4),
    ],
)
def test_eager_tuple_combine_preserves_operands(op, expected, extra):
    x = torch.tensor([1, 2, 4, 8])
    out = torch.full_like(x, -73)
    other = torch.full_like(x, -73)
    _eager_combine(x, out, other, op)
    assert out.tolist() == expected
    assert other.tolist() == extra


@helion.kernel()
def _sum_tuples_or_max_elements(a, b, c=None, d=None, add=torch.add):
    if c is None:
        return add(a[0], b[0]), add(a[1], b[1])
    return torch.maximum(a, c), torch.maximum(b, d)


@skipUnlessBackends(["triton"])
@skipIfRefEagerMark("replays compiled Triton and compares with eager")
def test_decorated_ambiguous_reduction_matches_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fold(x, out):
        for t in hl.tile(x.size(0), block_size=4):
            a, b = hl.reduce(_sum_tuples_or_max_elements, (x[t], x[t] + 1), 0)
            out[0] = a
            out[1] = b

    x = torch.tensor([1, 2, 4, 8])
    eager = torch.full((2,), -73)
    helion.kernel(static_shapes=True, ref_mode="eager")(fold)(x, eager)
    assert eager.tolist() == [15, 19]

    out = torch.full_like(eager, -73)
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(static_shapes=True)(fold), (x, out))
        source = bound.to_code(helion.Config(block_sizes=[]))
        interpret_triton_selection(source, (x, out), tmp_path / "fold.py", monkeypatch)
    # Passing four tensors here would select maximum and produce [8, 9].
    torch.testing.assert_close(out, eager)
