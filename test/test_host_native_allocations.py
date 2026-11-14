"""Positive native host allocation facts without executing device arithmetic."""

from __future__ import annotations

from functools import cache

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only
_GLOBAL_RESULT = torch.empty(8, 8)


def _user_allocation(x):
    return torch.empty_like(x)


def _global_result(x):
    return _GLOBAL_RESULT


# A Python helper does not become native by claiming the torch module name.
_user_allocation.__module__ = "torch"


def _host_result(x: torch.Tensor, mode: hl.constexpr):
    if mode == "binary":
        out = x + 1
    elif mode == "compare":
        out = x > 0
    elif mode == "unary":
        out = -x
    elif mode == "function":
        out = torch.add(x, 1)
    elif mode == "method":
        out = x.add(1)
    elif mode == "cast" or mode == "cast_alias":
        out = x.float()
    elif mode == "view":
        out = x.view_as(x)
    elif mode == "contiguous_copy" or mode == "contiguous_alias":
        out = x.contiguous()
    elif mode == "reshape_copy":
        out = x.reshape(4, 16)
    elif mode == "out":
        out = torch.add(x, 1, out=x)
    elif mode == "augmented":
        out = x
        out += 1
    elif mode == "user_allocation":
        out = _user_allocation(x)
    else:
        out = _global_result(x)
    scratch = torch.empty_like(x)
    for rows, cols in hl.tile(x.shape):
        scratch[rows, cols] = x[rows, cols]
    return out, scratch


@cache
def _case(mode):
    dtype = torch.float32 if mode == "cast_alias" else torch.float16
    x = torch.arange(64, dtype=dtype).reshape(8, 8)
    if mode in ("contiguous_copy", "reshape_copy"):
        x = x.t()
    kernel = helion.kernel(_host_result, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x, mode))
    return bound, x


def _run_host(bound, x, mode):
    with cpu_render_target(fake_device_sm_count=148):
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        result, _scratch = entry(x, mode)
    return result


@pytest.mark.parametrize(
    "mode", ["binary", "compare", "unary", "function", "method", "cast"]
)
@forbid_cuda_work()
def test_native_host_result_has_allocation_proof(mode):
    bound, x = _case(mode)
    out = bound.host_function.local_types["out"].fake_value
    assert bound.env.tensor_storage_is_compiler_allocated(out)
    assert not bound.env.tensor_storage_is_zero_filled_allocation(out)
    assert out.untyped_storage() not in bound.env._symbolically_exact_layout_storages
    result = _run_host(bound, x, mode)
    assert result.untyped_storage() is not x.untyped_storage()
    expected = (
        x > 0
        if mode == "compare"
        else -x
        if mode == "unary"
        else x.float()
        if mode == "cast"
        else x + 1
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "mode, copies",
    [
        ("cast_alias", False),
        ("view", False),
        ("contiguous_alias", False),
        ("contiguous_copy", True),
        ("reshape_copy", True),
    ],
)
@forbid_cuda_work()
def test_conditional_copy_or_view_does_not_prove_allocation(mode, copies):
    bound, x = _case(mode)
    out = bound.host_function.local_types["out"].fake_value
    assert not bound.env.tensor_storage_is_compiler_allocated(out)
    result = _run_host(bound, x, mode)
    assert (result.untyped_storage() is not x.untyped_storage()) is copies


@pytest.mark.parametrize("mode", ["out", "augmented"])
@forbid_cuda_work()
def test_in_place_host_result_does_not_prove_allocation(mode):
    bound, x = _case(mode)
    out = bound.host_function.local_types["out"].fake_value
    assert not bound.env.tensor_storage_is_compiler_allocated(out)
    result = _run_host(bound, x, mode)
    assert result.untyped_storage() is x.untyped_storage()


@pytest.mark.parametrize("mode", ["user_allocation", "global_result"])
@forbid_cuda_work()
def test_python_helper_result_is_not_native_allocation_provenance(mode):
    bound, x = _case(mode)
    out = bound.host_function.local_types["out"].fake_value
    assert not bound.env.tensor_storage_is_compiler_allocated(out)
    if mode == "user_allocation":
        # Its fake result really is distinct; native identity is still required.
        assert (
            out.untyped_storage()
            is not bound.host_function.local_types["x"].fake_value.untyped_storage()
        )
