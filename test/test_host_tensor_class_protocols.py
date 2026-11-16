"""Native Tensor-class access preserves proofs; user protocols remain opaque."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.backend_registry import get_backend
from helion._compiler.type_info import LiteralType
from helion._compiler.type_info import _is_known_host_value
from helion._compiler.variable_origin import AttributeOrigin
from helion._compiler.variable_origin import NameOrigin
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.tensor_views import chunk_unbind_axes
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _revisit_tensor_class(x: torch.Tensor, offset):
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        if isinstance(offset, torch.Tensor):
            shift = offset[row, col]
        else:
            shift = offset
        out[row, col] = x[row, col] + shift
    return out


def _native_chunk(x: torch.Tensor):
    out = torch.empty_like(x)
    left, right = torch.Tensor.chunk(x, 2, dim=1)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col]
    return out, left, right


def _bind(fn=_native_chunk, *, tensor_offset=False):
    args = [torch.empty(8, 8)]
    if fn is _revisit_tensor_class:
        args.append(torch.empty(8, 8) if tensor_offset else 0.5)
    return cpu_bind(helion.kernel(fn, static_shapes=True), tuple(args))


@pytest.mark.parametrize("tensor_offset", [False, True])
@forbid_cuda_work()
def test_revisited_tensor_class_preserves_allocation_and_layout(tensor_offset):
    with cpu_render_target(fake_device_sm_count=148):
        bound = _bind(_revisit_tensor_class, tensor_offset=tensor_offset)
        out = bound.host_function.local_types["out"].fake_value
        assert bound.env.tensor_storage_is_compiler_allocated(out)
        assert bound.env.tensor_layout_is_symbolically_exact(out)
        bound.to_code(bound.config_spec.default_config())


@forbid_cuda_work()
def test_native_tensor_descriptor_preserves_input_and_output_origins():
    with cpu_render_target(fake_device_sm_count=148):
        bound = _bind()
        tensors = {
            name: bound.host_function.local_types[name].fake_value
            for name in ("x", "out", "left", "right")
        }
        assert all(
            bound.env.tensor_storage_provenance(tensor) is not None
            for tensor in tensors.values()
        )
        assert tensors["left"].untyped_storage() is tensors["x"].untyped_storage()
        assert tensors["right"].untyped_storage() is tensors["x"].untyped_storage()
        assert bound.env.tensor_storage_is_compiler_allocated(tensors["out"])
        assert not _is_known_host_value(torch.Tensor)


@forbid_cuda_work()
def test_chunk_unbind_codegen_retains_input_and_fresh_output_proofs():
    kernel = helion.kernel(chunk_unbind_axes, autotune_effort="none")
    with cpu_render_target(fake_device_sm_count=148):
        if not get_backend(kernel.settings.backend).lowers_chunk_unbind:
            # Host Tensor.chunk provenance is covered above; its device
            # lowering has a separate, explicit backend capability.
            with pytest.raises(
                helion.exc.BackendUnsupported, match="torch.chunk device lowering"
            ):
                cpu_bind(kernel, (torch.empty(65, 8, 4), True))
            return
        bound = cpu_bind(kernel, (torch.empty(65, 8, 4), True))
        tensors = {
            name: bound.host_function.local_types[name].fake_value
            for name in ("x", "lo", "hi")
        }
        assert bound.env.tensor_storage_provenance(tensors["x"]) is not None
        assert all(
            bound.env.tensor_storage_is_compiler_allocated(tensors[name])
            for name in ("lo", "hi")
        )
        bound.to_code(helion.Config(block_sizes=[32]))


@forbid_cuda_work()
def test_custom_metaclass_equality_still_invalidates_storage_proofs():
    with cpu_render_target(fake_device_sm_count=148):
        bound = _bind(_revisit_tensor_class)
        out = bound.host_function.local_types["out"].fake_value
        x = bound.host_function.local_types["x"].fake_value
        assert bound.env.tensor_storage_is_compiler_allocated(out)
        calls = []

        class Meta(type):
            def __eq__(cls, other):
                calls.append(other)
                out.set_(x)
                return True

        class User(metaclass=Meta):
            pass

        origin = NameOrigin("User")
        literal = LiteralType(origin, User)
        with bound.env, bound.host_function:
            assert literal.merge(LiteralType(origin, User)) is literal
        assert calls == [User]
        assert bound.env.tensor_storage_provenance(out) is None


@forbid_cuda_work()
def test_python_tensor_descriptor_still_executes_under_effect_tracking():
    with cpu_render_target(fake_device_sm_count=148):
        bound = _bind(_revisit_tensor_class)
        out = bound.host_function.local_types["out"].fake_value
        x = bound.host_function.local_types["x"].fake_value
        assert bound.env.tensor_storage_is_compiler_allocated(out)
        calls = []
        native_chunk = torch.Tensor.chunk

        class Descriptor:
            def __get__(self, instance, owner):
                calls.append(owner)
                out.set_(x)
                return native_chunk

        origin = NameOrigin("Tensor")
        with (
            bound.env,
            bound.host_function,
            patch.object(torch.Tensor, "chunk", Descriptor()),
        ):
            result = LiteralType(origin, torch.Tensor).propagate_attribute(
                "chunk", AttributeOrigin(origin, "chunk")
            )
        assert result.proxy() is native_chunk
        assert calls == [torch.Tensor]
        assert bound.env.tensor_storage_provenance(out) is None
