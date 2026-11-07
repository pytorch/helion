"""Pure host helpers retain allocation proofs used by shared backends."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _helper_lane_output(x):
    _ignored = abs(-2)
    _ignored = helion.cdiv(x.size(0), 16)
    _ignored = helion.next_power_of_2(x.size(1))
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _ignored = float(1)
    _ignored = list((2,))[0]  # noqa: C410, RUF015 - exercise the host callable
    for rows in hl.tile(x.size(0)):
        for cols in hl.tile(x.size(1)):
            value = x[rows, cols]
            out[rows, cols] = value - value.sum(dim=1)[:, None]
    return out


@forbid_cuda_work()
def test_pure_helpers_before_allocation_preserve_metal_lane_proof():
    # Model the target before kernel construction; the outer guard stays active.
    with cpu_render_target(forbid_native=False, forbid_cuda_init=None):
        kernel = helion.kernel(_helper_lane_output, backend="metal", static_shapes=True)
        bound = cpu_bind(kernel, (torch.empty(16, 16),))
        code = bound.to_code(helion.Config(block_sizes=[16, 16], num_threads=[16, 1]))
    loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("lane_")
    ]
    assert len(loops) == 2


@forbid_cuda_work()
def test_pure_helpers_after_allocation_preserve_triton_exact_stride():
    with cpu_render_target(forbid_native=False, forbid_cuda_init=None):
        kernel = helion.kernel(
            _helper_lane_output, backend="triton", static_shapes=False
        )
        bound = cpu_bind(kernel, (torch.empty(16, 16),))
        out = bound.host_function.local_types["out"].fake_value
        assert bound.env.tensor_layout_is_symbolically_exact(out)
        code = bound.to_code(helion.Config(block_sizes=[16, 16]))
    assert "out_stride_1" not in code
    assert "tl.store" in code


def _constexpr_row_output(x):
    out = torch.empty([x.size(0)], device=x.device, dtype=x.dtype)
    block = hl.register_block_size(x.size(0))
    take_max = hl.constexpr(block >= 8)
    for rows in hl.tile(x.size(0), block_size=block):
        if take_max:
            out[rows] = x[rows, :].amax(-1)
        else:
            out[rows] = x[rows, :].sum(-1)
    return out


@forbid_cuda_work()
def test_constexpr_constructor_preserves_existing_allocation_proof():
    with cpu_render_target(
        fake_device_sm_count=148, forbid_native=False, forbid_cuda_init=None
    ):
        kernel = helion.kernel(_constexpr_row_output, static_shapes=True)
        bound = cpu_bind(kernel, (torch.empty(64, 256),))
        out = bound.host_function.local_types["out"].fake_value
        source = bound.host_function.local_types["x"].fake_value
        assert bound.env.tensor_storage_provenance(source)
        assert bound.env.tensor_storage_is_compiler_allocated(out)
        for block in (8, 4):
            ast.parse(bound.to_code(helion.Config(block_sizes=[block])))
