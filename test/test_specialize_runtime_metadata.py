from __future__ import annotations

import ast
import contextlib
import dataclasses
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
from torch._higher_order_ops.effects import _EffectType
from torch._higher_order_ops.effects import _register_effectful_op
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.overrides import TorchFunctionMode
from torch.utils._device import DeviceContext
from torch.utils._python_dispatch import TorchDispatchMode

import helion
from helion._compiler import type_info
from helion._compiler.tensor_utils import function_modes_trusted
from helion._compiler.tensor_utils import host_modes_trusted
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.tunable_fragments import load_host_function
from helion.exc import TypeInferenceError
import helion.language as hl
from helion.runtime.kernel import _HostModesTrustedExtractor

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


# An operator whose non-default overload alone has a registered effect: a
# packet call's effect lookup follows the overload the dispatcher picks.
_LIB = torch.library.Library("helion_test", "FRAGMENT")
_LIB.define("tagged(Tensor x) -> Tensor")
_LIB.define("tagged.note(str s) -> ()")
_LIB.impl("tagged", lambda x: x.clone(), "CompositeExplicitAutograd")
_LIB.impl("tagged.note", lambda s: None, "CompositeExplicitAutograd")
_register_effectful_op(torch.ops.helion_test.tagged.note, _EffectType.ORDERED)


# An operator of another namespace named like aten.clone.
@torch.library.custom_op("helion_test_ns::clone", mutates_args=())
def _other_clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_other_clone.register_fake
def _(x):
    return torch.empty_like(x)


# A module attribute holding an operator, reached without a torch.ops lookup.
_ALIASES = ModuleType("helion_test_aliases")
_ALIASES.clone = torch.ops.aten.clone


def _after_operator(x, mode: hl.constexpr):
    hl.specialize(x.size())
    hl.specialize(x.stride())
    flat = x.view(-1)
    if mode == "packet":
        torch.ops.aten.clone(x)
    elif mode == "method":
        x.clone()
    elif mode == "module":
        _ALIASES.clone(x)
    elif mode == "permute":
        # Changes a fresh tensor's metadata, not flat's: checked like t_() on
        # it. Not x: torch nightly's meta check rejects restriding an input
        # whose storage offset is symbolic, x.t_() included.
        torch.ops.aten.t_.default(torch.empty_like(x))
    elif mode == "custom":
        # A torch.library operator with a Python fake kernel.
        torch.ops.helion_test.passthrough(x)
    elif mode == "custom_overload":
        # Its overload, which dispatches through its PyObject dispatcher
        # record on torch 2.14+.
        torch.ops.helion_test.passthrough.default(x)
    elif mode == "resolved":
        # A packet call resolving to tagged.default, which has no effect.
        torch.ops.helion_test.tagged(x)
    elif mode == "resolved_effect":
        # A packet call resolving to tagged.note, which has one.
        torch.ops.helion_test.tagged("")
    elif mode == "restride":
        # Changes flat's own metadata, which the tracker sees.
        torch.ops.aten.as_strided_.default(flat, (flat.size(0) // 2,), (2,), 0)
    elif mode == "overload_effect":
        # A registered effect may reach any tensor.
        torch.ops.aten._print.default("")
    elif mode == "packet_effect":
        torch.ops.aten._print("")
    else:
        raise AssertionError(mode)
    out = torch.empty_like(flat)
    for tile in hl.tile(flat.size(0)):
        out[tile] = flat[tile]
    return out


# Triton passes a non-literal stride as a kernel argument; CuTe always reads
# strides from the tensor layout, so its code shows no difference.
@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    "mode",
    [
        "packet",
        "method",
        "module",
        "permute",
        "custom",
        "custom_overload",
        "resolved",
        "restride",
        "overload_effect",
        "packet_effect",
        "resolved_effect",
    ],
)
def test_torch_ops_calls_keep_view_strides_unless_they_change(mode):
    # Looking up and calling a torch.ops operator keeps the exact layout of a
    # specialized input's view unless the call restrides it or the operator
    # has a registered effect.
    # Before torch 2.14 nothing links an overload to its entry: opaque there.
    linked = hasattr(torch.ops.aten.clone.default, "_cpp_dispatch_handle")
    opaque = mode in ("restride", "overload_effect", "packet_effect", "resolved_effect")
    opaque = opaque or (mode in ("permute", "custom_overload") and not linked)
    if type_info._get_effect is None and not opaque:
        pytest.skip("this torch has no effect query: every torch.ops call is opaque")
    kernel = helion.kernel(_after_operator, static_shapes=False)
    with cpu_render_target():
        bound = kernel.bind((torch.empty(4, 8), mode))
        code = bound.to_code(bound.config_spec.default_config())
    assert ("flat_stride_0" in code) == opaque


def _patched_operator(x, call: hl.constexpr):
    hl.specialize(x.size())
    hl.specialize(x.stride())
    flat = x.view(-1)
    if call == "alias":
        torch.ops.aten.helion_test_alias(x)
    elif call == "overload":
        torch.ops.aten.clone.default(x)
    elif call == "custom":
        torch.ops.helion_test.passthrough.default(x)
    elif call == "print":
        torch.ops.aten._print.default("")
    elif call == "module":
        _ALIASES.clone(x)
    else:
        torch.ops.aten.clone(x)
    out = torch.empty_like(flat)
    for tile in hl.tile(flat.size(0)):
        out[tile] = flat[tile]
    return out


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    "patch",
    [
        "descriptor",
        "call",
        "init",
        "dispatch",
        "builtin_dispatch",
        "dispatcher_record",
        "packet_swap",
        "namespace_swap",
        "overload_swap",
        "fake_dispatch_type",
        "foreign_dispatch_type",
        "reschema_overload",
        "no_effect_query",
    ],
)
def test_patched_operator_hooks_stay_opaque(monkeypatch, patch):
    # A Python hook on the registry may run anything on lookup or call; a torch
    # without the effect query cannot prove a call free of effects.
    packet = torch._ops.OpOverloadPacket
    if patch == "no_effect_query":
        monkeypatch.setattr(type_info, "_get_effect", None)
    elif patch == "descriptor":
        monkeypatch.setattr(
            type(torch.ops.aten),
            "helion_test_alias",
            property(lambda self: torch.ops.aten.alias),
            raising=False,
        )
    elif patch == "call":
        call = packet.__call__
        monkeypatch.setattr(
            packet,
            "__call__",
            lambda self, *args, **kwargs: call(self, *args, **kwargs),
        )
    elif patch == "init":
        init = packet.__init__
        monkeypatch.setattr(
            packet,
            "__init__",
            lambda self, *args, **kwargs: init(self, *args, **kwargs),
        )
    elif patch == "builtin_dispatch":
        # Another builtin as the entry (not the operator's native one).
        monkeypatch.setattr(torch.ops.aten.clone, "_op", id)
    elif patch == "packet_swap":
        # Another operator's native entries, which an effect lookup by this
        # operator's name would not cover.
        monkeypatch.setattr(torch.ops.aten.clone, "_op", torch.ops.aten.alias._op)
    elif patch == "namespace_swap":
        # The same name in another namespace.
        other = torch.ops.helion_test_ns.clone._op
        monkeypatch.setattr(torch.ops.aten.clone, "_op", other)
    elif patch == "overload_swap":
        alias = torch.ops.aten.alias.default._op
        monkeypatch.setattr(torch.ops.aten.clone.default, "_op", alias)
    elif patch == "fake_dispatch_type":
        # A Python class posing as torch.library's dispatch callable.
        overload = torch.ops.helion_test.passthrough.default
        if getattr(overload, "_pyobj_dispatcher", None) is None:
            pytest.skip("this torch has no PyObject dispatcher records")

        class _PyObjectDispatchFunc:
            # Named like the static type: only being a heap type gives it away.
            __module__ = "torch._C"
            __qualname__ = "_PyObjectDispatchFunc"

            def __call__(self, *args, **kwargs):
                return overload._cpp_dispatch_handle(*args, **kwargs)

        fake = _PyObjectDispatchFunc()
        monkeypatch.setattr(overload, "_op", fake)
        monkeypatch.setattr(
            overload,
            "_pyobj_dispatcher",
            dataclasses.replace(overload._pyobj_dispatcher, dispatch=fake),
        )
    elif patch == "foreign_dispatch_type":
        # A static type from elsewhere (a builtin) as the dispatch callable.
        overload = torch.ops.helion_test.passthrough.default
        if getattr(overload, "_pyobj_dispatcher", None) is None:
            pytest.skip("this torch has no PyObject dispatcher records")
        monkeypatch.setattr(overload, "_op", id)
        monkeypatch.setattr(
            overload,
            "_pyobj_dispatcher",
            dataclasses.replace(overload._pyobj_dispatcher, dispatch=id),
        )
    elif patch == "reschema_overload":
        # An effectful overload's schema naming an operator without one.
        schema = torch.ops.aten.clone.default._schema
        monkeypatch.setattr(torch.ops.aten._print.default, "_schema", schema)
    elif patch == "dispatch":
        op = torch.ops.aten.clone._op
        shim = lambda *args, **kwargs: op(*args, **kwargs)  # noqa: E731
        monkeypatch.setattr(torch.ops.aten.clone, "_op", shim)
    elif patch == "dispatcher_record":
        # A stand-in for a torch.library overload's dispatcher record, around
        # its real dispatch callable.
        overload = torch.ops.helion_test.passthrough.default
        if getattr(overload, "_pyobj_dispatcher", None) is None:
            pytest.skip("this torch has no PyObject dispatcher records")
        real = overload._pyobj_dispatcher
        stand_in = SimpleNamespace(dispatch=real.dispatch, redispatch=real.redispatch)
        monkeypatch.setattr(overload, "_pyobj_dispatcher", stand_in)
    else:
        raise AssertionError(patch)
    kernel = helion.kernel(_patched_operator, static_shapes=False)
    with cpu_render_target():
        entry = {
            "descriptor": "alias",
            # Also reached through a module attribute, past the lookup check.
            "call": "module",
            "init": "module",
            "overload_swap": "overload",
            "fake_dispatch_type": "custom",
            "foreign_dispatch_type": "custom",
            "dispatcher_record": "custom",
            "reschema_overload": "print",
        }.get(patch, "packet")
        bound = kernel.bind((torch.empty(4, 8), entry))
        code = bound.to_code(bound.config_spec.default_config())
    assert "flat_stride_0" in code


class _PassthroughFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class _PassthroughDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class _PassthroughPreDispatchMode(SchemaCheckMode):
    # Pushed on the pre-dispatch stack, which only a few mode classes may use.
    def __init__(self):
        super().__init__()
        self._dispatch_key = torch._C.DispatchKey.PreDispatch

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("user_mode", ["function", "dispatch", "pre_dispatch"])
def test_user_torch_modes_make_torch_ops_calls_opaque(user_mode):
    # A user's torch mode runs Python inside every torch.ops call, which may
    # restride tensors the call does not take. Whether one is active is part
    # of the binding's cache key, in both directions.
    if type_info._get_effect is None:
        pytest.skip("this torch has no effect query: every torch.ops call is opaque")
    kernel = helion.kernel(_after_operator, static_shapes=False)
    args = (torch.empty(4, 8), "packet")
    active = {
        "function": _PassthroughFunctionMode,
        "dispatch": _PassthroughDispatchMode,
        "pre_dispatch": _PassthroughPreDispatchMode,
    }[user_mode]()
    with cpu_render_target():
        with active:
            under_mode = kernel.bind(args)
        outside = kernel.bind(args)
        with active:
            assert kernel.bind(args) is under_mode
        assert kernel.bind(args) is outside
        codes = [
            bound.to_code(bound.config_spec.default_config())
            for bound in (under_mode, outside)
        ]
    assert ["flat_stride_0" in code for code in codes] == [True, False]


class _IntCloneMode(TorchFunctionMode):
    # Answers aten.clone on an int, which no schema of it matches.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.clone and type(args[0]) is int:
            return torch.zeros(())
        return func(*args, **(kwargs or {}))


def _mode_answered(x):
    torch.ops.aten.clone(3)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile]
    return out


@skipUnlessBackends(["triton"])
def test_torch_ops_call_a_mode_answers_binds_opaque():
    # Under a torch mode the call is opaque before its schemas are matched: the
    # mode may answer arguments no schema takes.
    kernel = helion.kernel(_mode_answered, static_shapes=False)
    with cpu_render_target(), _IntCloneMode():
        bound = kernel.bind((torch.empty(8),))
    assert bound.env.torch_mode_sensitive


def test_function_modes_trusted_matches_under_dynamo(monkeypatch):
    # The cache key's mode check under capture reads Dynamo's symbolic function
    # mode stack, with the same torch.device exemption as in eager, which holds
    # only while torch.device keeps its own __torch_function__.
    def pick(x):
        return x + 1 if function_modes_trusted() else x - 1

    x = torch.zeros(())
    results = []
    for active in (
        contextlib.nullcontext(),
        torch.device("cpu"),
        _PassthroughFunctionMode(),
    ):
        with active:
            compiled = torch.compile(pick, fullgraph=True, backend="eager")
            results.append((compiled(x).item(), host_modes_trusted()))
    hook = DeviceContext.__torch_function__
    monkeypatch.setattr(
        DeviceContext,
        "__torch_function__",
        lambda self, func, types, args=(), kwargs=None: hook(
            self, func, types, args, kwargs
        ),
    )
    for active in (contextlib.nullcontext(), torch.device("cpu")):
        torch._dynamo.reset()
        with active:
            compiled = torch.compile(pick, fullgraph=True, backend="eager")
            results.append((compiled(x).item(), host_modes_trusted()))
    assert results == [
        (1.0, True),
        (1.0, True),
        (-1.0, False),
        (-1.0, False),
        (-1.0, False),
    ]


def test_mode_instances_with_their_own_hooks_are_not_trusted(monkeypatch):
    # Torch looks a mode's hook up on the instance, so one stored there runs
    # inside every call even when the class is trusted. The cache key's check
    # outside Dynamo's trace (torch.compile sessions and torch.export set
    # is_compiling there too) reads the real stacks.
    def forward(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))

    device = DeviceContext("cpu")
    device.__torch_function__ = forward.__get__(device)
    fake = FakeTensorMode()
    fake.__torch_dispatch__ = forward.__get__(fake)
    extractor = _HostModesTrustedExtractor()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    answers = []
    for active in (FakeTensorMode(), device, fake, _PassthroughDispatchMode()):
        with active:
            answers.append((host_modes_trusted(), extractor(())))
    assert answers == [(True, True), (False, False), (False, False), (False, False)]


def test_registered_dispatch_mode_with_a_replaced_class_hook_is_not_trusted(
    monkeypatch,
):
    # A registered mode is trusted for the __torch_dispatch__ it had when
    # registered; a replacement on its class runs inside every host call.
    hook = FakeTensorMode.__dict__["__torch_dispatch__"]

    def forward(self, func, types, args=(), kwargs=None):
        return hook(self, func, types, args, kwargs)

    with FakeTensorMode():
        assert host_modes_trusted()
    monkeypatch.setattr(FakeTensorMode, "__torch_dispatch__", forward)
    with FakeTensorMode():
        assert not host_modes_trusted()
