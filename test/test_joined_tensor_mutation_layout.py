from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _joined_owner_mutation(
    x: torch.Tensor,
    src: torch.Tensor,
    choose: int,
    mutation: hl.constexpr,
    nested: hl.constexpr,
):
    first = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    second = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    third = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    if choose > 0:
        inner = first
    else:
        inner = second
    if nested:
        if choose < 0:
            target = third
        else:
            target = inner
    else:
        target = inner
    if mutation == "set":
        target.set_(src)
    elif mutation == "offset":
        target.as_strided_((3, 3), (4, 1), 1)
    else:
        target.t_()
    for i, j in hl.tile(first.size()):
        first[i, j] = 1.0
    return first, second, third


@pytest.mark.parametrize("mutation", ["set", "transpose", "offset"])
@pytest.mark.parametrize("nested", [False, True])
def test_join_receiver_mutation_refreshes_every_possible_owner(mutation, nested):
    kernel = helion.kernel(_joined_owner_mutation, static_shapes=False)
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    src = torch.empty(4, 4).t()
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x, src, 1, mutation, nested))
        with bound.env, bound.host_function:
            owners = ("first", "second", "third") if nested else ("first", "second")
            for name in owners:
                fake = bound.host_function.local_types[name].fake_value
                if mutation != "transpose":
                    assert bound.env.tensor_storage_provenance(fake) is None
                    assert not bound.env.tensor_storage_is_compiler_allocated(fake)
                    assert not bound.env.tensor_storage_is_zero_filled_allocation(fake)
                else:
                    assert bound.env.tensor_storage_is_compiler_allocated(fake)
                    assert bound.env.tensor_storage_is_zero_filled_allocation(fake)
                assert not bound.env.tensor_layout_is_symbolically_exact(fake)
                assert isinstance(fake.stride(1), torch.SymInt)
        config = bound.config_spec.default_config()
        code = bound.to_code(config)
        if bound.env.backend.name == "triton":
            # Triton passes explicit strides; CuTe obtains them from the
            # runtime tensor metadata in its ordinary launcher ABI.
            assert "first.stride(1)" in code
        entry = load_host_function(bound, config, lambda *args, **kwargs: None)
        for choose in (1, 0, -1, 1):
            result = entry(x, src, choose, mutation, nested)
            selected = 2 if nested and choose < 0 else 0 if choose > 0 else 1
            for index, value in enumerate(result):
                if mutation == "offset":
                    assert value.shape == ((3, 3) if index == selected else (4, 4))
                    assert value.stride() == (4, 1)
                    assert value.storage_offset() == (1 if index == selected else 0)
                else:
                    assert value.stride() == ((1, 4) if index == selected else (4, 1))
                if mutation == "set":
                    assert (value.untyped_storage() is src.untyped_storage()) is (
                        index == selected
                    )


def test_joined_transpose_preserves_each_owners_runtime_shape():
    kernel = helion.kernel(_joined_owner_mutation, static_shapes=False)
    x = torch.empty(3, 5)
    src = torch.empty_like(x)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (x, src, 1, "transpose", True))
        with bound.env, bound.host_function:
            for name in ("first", "second", "third"):
                fake = bound.host_function.local_types[name].fake_value
                for dim, size in enumerate(fake.size()):
                    assert isinstance(size, torch.SymInt)
                    expr = bound.env.shape_env.simplify(size._sympy_())
                    origin = bound.host_function.expr_to_origin[expr].origin
                    assert origin.host_str() == f"{name}.size({dim})"
        config = bound.config_spec.default_config()
        code = bound.to_code(config)
        if bound.env.backend.name == "triton":
            assert "first.size(0)" in code
            assert "first.size(1)" in code
        entry = load_host_function(bound, config, lambda *args, **kwargs: None)
        for choose in (1, 0, -1, 1):
            result = entry(x, src, choose, "transpose", True)
            selected = 2 if choose < 0 else 0 if choose > 0 else 1
            for index, value in enumerate(result):
                assert value.shape == ((5, 3) if index == selected else (3, 5))
                assert value.stride() == ((1, 5) if index == selected else (5, 1))


def test_joined_mutation_cannot_give_an_owner_runtime_dependent_rank():
    kernel = helion.kernel(_joined_owner_mutation, static_shapes=False)
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.TypeInferenceError, match="cannot change its rank"),
    ):
        cpu_bind(kernel, (torch.empty(4, 4), torch.empty(16), 1, "set", False))
