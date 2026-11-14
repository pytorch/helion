from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.tunable_fragments import load_host_function
from helion.exc import TypeInferenceError
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _post_mutation(x, flag: int):
    if flag > 0:
        x.t_()
    width = hl.specialize(x.size(1))
    out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = x[row, :].sum(-1) + width
    return x, out


@pytest.mark.parametrize("flag", [0, 1])
def test_post_mutation_specialization_requires_an_entry_input_source(flag):
    kernel = helion.kernel(_post_mutation, static_shapes=True)
    x = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    with (
        cpu_render_target(),
        pytest.raises(TypeInferenceError, match="entry input.*before.*mutation"),
    ):
        kernel.bind((x, flag))
    assert x.shape == (5, 7)
    assert not kernel._bound_kernels


def _before_mutation(x, flag: hl.constexpr):
    width = hl.specialize(x.size(0) if flag > 0 else x.size(1))
    output_size = hl.specialize(x.size(1) if flag > 0 else x.size(0))
    if flag > 0:
        x.t_()
    out = torch.empty((output_size,), dtype=x.dtype, device=x.device)
    for row in hl.tile(output_size):
        out[row] = width
    return x, out


def test_specializing_input_metadata_before_mutation_preserves_valid_replay():
    kernel = helion.kernel(_before_mutation, static_shapes=True)
    with cpu_render_target():
        bindings = []
        for flag in (1, 0, 1):
            x = torch.arange(35, dtype=torch.float32).reshape(5, 7)
            bound = kernel.bind((x, flag))
            bindings.append(bound)
            entry = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            value, out = entry(x, flag)
            assert value.shape == ((7, 5) if flag else (5, 7))
            assert out.shape == (value.size(0),)
        assert bindings[0] is bindings[2]
        assert bindings[0] is not bindings[1]


def _input_backed(x, scalar: int):
    width = hl.specialize(x.size(1))
    pitch = hl.specialize(x.stride(0))
    value = hl.specialize(scalar)
    out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = width + pitch + value
    return out


def test_input_backed_size_stride_and_scalar_specialization_still_bind():
    kernel = helion.kernel(_input_backed, static_shapes=False)
    x = torch.empty(5, 7)
    padded = torch.empty_strided((5, 7), (8, 1))
    with cpu_render_target():
        original = kernel.bind((x, 3))
        assert kernel.bind((torch.empty_like(x), 3)) is original
        changed_stride = kernel.bind((padded, 3))
        changed_scalar = kernel.bind((x, 4))
        assert changed_stride is not original
        assert changed_scalar is not original
        for bound in (original, changed_stride, changed_scalar):
            ast.parse(bound.to_code(bound.config_spec.default_config()))


@torch.library.custom_op("helion_test::passthrough", mutates_args=())
def _passthrough(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_passthrough.register_fake
def _(x):
    return torch.empty_like(x)


def _after_operator(x, mode: hl.constexpr):
    hl.specialize(x.size())
    hl.specialize(x.stride())
    flat = x.view(-1)
    if mode == "packet":
        # Resolves to the .default overload, not the writing .out one.
        torch.ops.aten.clone(x)
    elif mode == "permute":
        # Changes x's metadata, not flat's: checked like x.t_().
        torch.ops.aten.t_.default(x)
    elif mode == "custom":
        # A torch.library operator with a Python fake kernel.
        torch.ops.helion_test.passthrough(x)
    elif mode == "restride":
        # Changes flat's own metadata, which the tracker sees.
        torch.ops.aten.as_strided_.default(flat, (flat.size(0) // 2,), (2,), 0)
    else:
        # A registered effect may reach any tensor.
        torch.ops.aten._print.default("")
    out = torch.empty_like(flat)
    for tile in hl.tile(flat.size(0)):
        out[tile] = flat[tile]
    return out


# Triton passes a non-literal stride as a kernel argument; CuTe always reads
# strides from the tensor layout, so its code shows no difference.
@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("mode", ["packet", "permute", "custom", "restride", "effect"])
def test_torch_ops_calls_keep_view_strides_unless_they_change(mode):
    # Looking up and calling a torch.ops operator keeps the exact layout of a
    # specialized input's view unless the call restrides it or the operator
    # has a registered effect.
    kernel = helion.kernel(_after_operator, static_shapes=False)
    with cpu_render_target():
        bound = kernel.bind((torch.empty(4, 8), mode))
        code = bound.to_code(bound.config_spec.default_config())
    assert ("flat_stride_0" in code) == (mode in ("restride", "effect"))


def _patched_operator(x, use_alias: hl.constexpr):
    hl.specialize(x.size())
    hl.specialize(x.stride())
    flat = x.view(-1)
    if use_alias:
        torch.ops.aten.helion_test_alias(x)
    else:
        torch.ops.aten.clone(x)
    out = torch.empty_like(flat)
    for tile in hl.tile(flat.size(0)):
        out[tile] = flat[tile]
    return out


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("patch", ["descriptor", "call"])
def test_patched_operator_hooks_stay_opaque(monkeypatch, patch):
    # A Python hook on the registry may run anything on lookup or call.
    if patch == "descriptor":
        monkeypatch.setattr(
            type(torch.ops.aten),
            "helion_test_alias",
            property(lambda self: torch.ops.aten.alias),
            raising=False,
        )
    else:
        call = torch._ops.OpOverloadPacket.__call__
        monkeypatch.setattr(
            torch._ops.OpOverloadPacket,
            "__call__",
            lambda self, *args, **kwargs: call(self, *args, **kwargs),
        )
    kernel = helion.kernel(_patched_operator, static_shapes=False)
    with cpu_render_target():
        bound = kernel.bind((torch.empty(4, 8), patch == "descriptor"))
        code = bound.to_code(bound.config_spec.default_config())
    assert "flat_stride_0" in code
