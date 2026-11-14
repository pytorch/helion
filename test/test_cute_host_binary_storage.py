from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.runtime_tensor_guards import (
    tensor_pairs_with_proven_disjoint_storage,
)
from helion._compiler.type_info import _register_host_operator_allocation
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _host_binary_output(
    x: torch.Tensor, left: torch.Tensor, select: int, reverse: hl.constexpr
) -> torch.Tensor:
    if reverse:
        first = 1 - left
    else:
        first = left + 1
    alias = first[:]
    if select > 0:
        out = alias
    else:
        out = torch.zeros_like(left) * 2
    for row in hl.tile(x.size(0)):
        out[row] = x[row, :].sum(-1)
    return out


@pytest.mark.parametrize("reverse", [False, True])
def test_binary_allocations_remain_disjoint_through_views_and_host_joins(reverse):
    kernel = helion.kernel(
        _host_binary_output,
        backend="cute",
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.empty(64, 32), torch.empty(64), 1, reverse))
        host = bound.host_function
        with bound.env, host:
            tensors = {
                name: host.local_types[name].fake_value
                for name in ("x", "left", "first", "alias", "out")
            }
            pairs = tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)
            assert frozenset(("left", "out")) in pairs
            assert frozenset(("x", "out")) in pairs
            assert frozenset(("first", "alias")) not in pairs
            assert frozenset(("first", "out")) not in pairs
            assert not bound.env.tensor_storage_is_zero_filled_allocation(
                tensors["first"]
            )
            assert not bound.env.tensor_layout_is_symbolically_exact(tensors["first"])
        ast.parse(bound.to_code(bound.config_spec.default_config()))


def test_binary_allocation_proof_rejects_aliases_and_overloaded_operands():
    kernel = helion.kernel(
        _host_binary_output,
        backend="cute",
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.empty(64, 32), torch.empty(64), 1, False))
        host = bound.host_function
        with bound.env, host:
            left = host.local_types["left"].fake_value
            x = host.local_types["x"].fake_value

            class Overloaded:
                def __radd__(self, other):
                    return left

            assert x + Overloaded() is left
            # An overloaded operator may return a captured input even though
            # that result does not alias either explicit operand.
            _register_host_operator_allocation((x, Overloaded()), left)
            # An aliased result must not turn an input into an allocation.
            _register_host_operator_allocation((left, 0), left)
            assert not bound.env.tensor_storage_is_compiler_allocated(left)


def _host_inplace_output(x: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    out = left
    out += 1
    for row in hl.tile(x.size(0)):
        out[row] = x[row, :].sum(-1)
    return out


def test_augmented_assignment_does_not_gain_out_of_place_allocation_proof():
    kernel = helion.kernel(
        _host_inplace_output,
        backend="cute",
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.empty(64, 32), torch.empty(64)))
        with bound.env, bound.host_function:
            tensors = {
                name: bound.host_function.local_types[name].fake_value
                for name in ("left", "out")
            }
            assert not bound.env.tensor_storage_is_compiler_allocated(tensors["out"])
            assert not tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)
