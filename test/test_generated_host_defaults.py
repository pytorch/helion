from __future__ import annotations

import ast
import inspect
import pickle
import struct
import sys
from types import FunctionType
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._testing import cpu_render_target
from helion._testing import import_path
from helion._testing import matchesBackends
from helion._testing import skipIfMetal
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import pallas_host_observer
from helion.autotuner import EnumFragment
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.kernel_args import make_portable_args
from helion.autotuner.kernel_args import resolve_portable_args
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _unload_compiled_fn
import helion.language as hl
from helion.runtime.kernel import _with_evaluated_defaults

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def load_standalone(tmp_path):
    modules = []

    def load(source):
        # Match the documented standalone contract: import the complete generated
        # module, registering it before class decorators resolve annotations.
        path = tmp_path / f"generated_defaults_{uuid.uuid4().hex}.py"
        path.write_text(source)
        module = import_path(path)
        modules.append(module)
        return module

    yield load
    for module in modules:
        sys.modules.pop(module.__name__, None)


DEFAULT_CALLS = []


def _make_default():
    DEFAULT_CALLS.append(None)
    return [2, 16]


@helion.kernel(autotune_effort="none")
def _fragment_default(x, fragment: hl.constexpr = EnumFragment((2, 16))):  # noqa: B008 -- default AST must retain the constructor
    factor = hl.register_tunable("factor", fragment)
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] * factor
    return x


@helion.kernel(autotune_effort="none")
def _factory_default(x, values=_make_default()):  # noqa: B008 -- detect constructor replay
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + values[0]
    return x


@helion.kernel(autotune_effort="none")
def _alias_defaults(x, left=_make_default(), right=_make_default()):  # noqa: B008 -- detect constructor replay
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + left[0] + right[0]
    return x


@helion.kernel(autotune_effort="none")
def _positional_default(x, scale=2.5, /, offset=3.5):
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] * scale + offset
    return x


@helion.kernel(autotune_effort="none")
def _torch_defaults(
    x,
    out_dtype: hl.constexpr = torch.float64,
    device: hl.constexpr = torch.device("cpu"),  # noqa: B008 -- test captured default
    eps=1e-5,
):
    out = torch.empty_like(x, dtype=out_dtype, device=device)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + eps
    return out


@pytest.mark.parametrize("standalone", [False, True])
def test_generated_torch_defaults_remain_available(standalone, load_standalone):
    if standalone and matchesBackends(["metal"]):
        pytest.skip("Metal does not support dependency-free export")
    dtype = torch.float32 if matchesBackends(["metal"]) else torch.float64
    original = _torch_defaults.fn
    fn = FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        (dtype, *original.__defaults__[1:]),
    )
    fn.__annotations__ = dict(original.__annotations__)
    tensor = torch.ones(7)
    with (
        cpu_render_target(),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
    ):
        kernel = helion.kernel(fn, autotune_effort="none")
        bound = kernel.bind((tensor,))
        source = bound.to_code(
            options=helion.OutputCodeOptions(allow_helion_deps=not standalone)
        )
        module = load_standalone(source) if standalone else PyCodeCache.load(source)
        entry = getattr(module, kernel.name)
        arguments = inspect.signature(entry).bind(tensor)
        arguments.apply_defaults()
        assert arguments.arguments["out_dtype"] is dtype
        assert arguments.arguments["device"] == torch.device("cpu")
        assert type(arguments.arguments["device"]) is torch.device
        assert arguments.arguments["eps"] == 1e-5
        if not standalone:

            def observe(*args, **kwargs):
                return None

            launcher = (
                pallas_host_observer(observe)
                if bound.settings.backend == "pallas"
                else observe
            )
            output = entry(tensor, _launcher=launcher)
            assert output.dtype is dtype
            assert output.device == torch.device("cpu")


@pytest.mark.parametrize("template", [_fragment_default, _factory_default])
def test_generated_module_load_does_not_evaluate_user_default_expression(template):
    tensor = torch.empty(17)
    defaults = template.fn.__defaults__
    assert defaults is not None
    calls = len(DEFAULT_CALLS)
    # The isolated bind avoids the separate old unhashable constexpr-key issue.
    bound = cpu_bind(template, (tensor, defaults[0]))
    module = PyCodeCache.load(bound.to_code())
    assert len(DEFAULT_CALLS) == calls
    entry = getattr(module, template.name)
    assert entry.__defaults__ is None
    assert entry(tensor, defaults[0], _launcher=lambda *args, **kwargs: None) is tensor
    with pytest.raises(TypeError, match="required positional argument"):
        entry(tensor)


def test_compiled_public_calls_keep_evaluated_default_identity():
    kernel = helion.kernel(_factory_default.fn, autotune_effort="none")
    tensor = torch.empty(17)
    (default,) = kernel.fn.__defaults__
    calls = len(DEFAULT_CALLS)
    with cpu_render_target():
        bound = kernel.bind((tensor,))
        compiled = bound.compile_config()
        assert compiled.__defaults__ == (default,)
        assert compiled.__defaults__[0] is default
        bound._run = lambda *args: compiled(*args, _launcher=lambda *a, **kw: None)
        assert kernel(tensor) is tensor
        assert bound(tensor) is tensor
        assert kernel(tensor, values=default) is tensor
        assert kernel(tensor, default) is tensor
        assert len(DEFAULT_CALLS) == calls
        # Direct search construction must fill defaults before the worker
        # reloads source whose nonportable defaults are deliberately omitted.
        search = BaseSearch(bound, (tensor,))
        portable = _load_compiled_fn(_serialize_compiled_fn(compiled))
        try:
            assert portable.__defaults__ is None
            assert portable(*search.args, _launcher=lambda *a, **kw: None) is tensor
        finally:
            _unload_compiled_fn(portable)
        assert search.args[1] is default
        assert len(DEFAULT_CALLS) == calls


def test_shared_code_cache_does_not_share_mutable_default_objects():
    first_default, second_default = [2, 16], [2, 16]
    tensor = torch.empty(17)
    compiled = []
    with cpu_render_target():
        for default in (first_default, second_default):
            original = _alias_defaults.fn
            fn = FunctionType(
                original.__code__,
                original.__globals__,
                original.__name__,
                (default, default),
                original.__closure__,
            )
            kernel = helion.kernel(fn, autotune_effort="none")
            bound = kernel.bind((tensor,))
            entry = bound.compile_config()
            assert entry.__defaults__[0] is entry.__defaults__[1] is default
            compiled.append(entry)
        first, second = compiled
        assert first is not second
        assert first.__globals__ is second.__globals__
        cached = first.__globals__[first.__name__]
        assert cached.__defaults__ is None
        first_default.append(32)
        assert first.__defaults__[0] == [2, 16, 32]
        assert second.__defaults__[0] == [2, 16]


@skipIfMetal("Metal does not support dependency-free export")
def test_standalone_keeps_typed_scalar_and_signed_zero_defaults(load_standalone):
    original = _positional_default.fn
    fn = FunctionType(
        original.__code__, original.__globals__, original.__name__, (-0.0, 3)
    )
    kernel = helion.kernel(fn, autotune_effort="none", static_shapes=True)
    tensor = torch.empty(17)
    with (
        cpu_render_target(),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
    ):
        bound = kernel.bind((tensor,))
        source = bound.to_code(
            options=helion.OutputCodeOptions(allow_helion_deps=False)
        )
        entry = getattr(load_standalone(source), kernel.name)
        assert type(entry.__defaults__[0]) is float
        assert entry.__defaults__[0].hex() == "-0x0.0p+0"
        assert type(entry.__defaults__[1]) is int
        assert entry.__defaults__[1] == 3
        # Exercise the exported signature without launching: CuTe's standalone
        # tensor guard correctly refuses a CPU tensor even with a no-op launcher.
        arguments = inspect.signature(entry).bind(tensor)
        arguments.apply_defaults()
        assert arguments.arguments["scale"].hex() == "-0x0.0p+0"
        assert type(arguments.arguments["offset"]) is int


@pytest.mark.parametrize("mode", ["removed", "extra", "removed_extra"])
def test_generated_parameter_changes_preserve_trailing_default_alignment(mode):
    tensor = torch.empty(17)
    with cpu_render_target():
        bound = _positional_default.bind((tensor,))
        removed = {"scale"} if mode != "extra" else set()
        extra = ["workspace"] if mode != "removed" else []
        with bound.env, bound.host_function:
            definition = bound.host_function.codegen_function_def(
                [ast.Return(value=ast.Constant(value=None))],
                extra_params=extra,
                removed_args=removed,
            )
            namespace = {bound.env.backend.get_launcher_name(): object()}
            # Extended AST nodes are rendered to source before Python loads them.
            source = ast.unparse(ast.fix_missing_locations(definition))
            exec(compile(source, "<generated-signature>", "exec"), namespace)
        entry = namespace[bound.kernel.name]
        restored = _with_evaluated_defaults(entry, bound.kernel.signature)
        if extra:
            assert entry.__defaults__ is restored.__defaults__ is None
            with pytest.raises(TypeError, match="required positional"):
                entry(tensor)
            args = [tensor, *([] if removed else [2.5]), 3.5, object()]
            assert entry(*args) is None
        else:
            assert entry.__defaults__ == restored.__defaults__ == (3.5,)
            assert tuple(inspect.signature(entry).parameters) == (
                "x",
                "offset",
                "_launcher",
            )
            assert entry(tensor) is None


def test_positional_only_literal_defaults_remain_available_in_generated_code():
    tensor = torch.empty(17)
    with cpu_render_target():
        bound = _positional_default.bind((tensor,))
        module = PyCodeCache.load(bound.to_code())
        entry = module._positional_default
        assert entry.__defaults__ == (2.5, 3.5)
        assert (
            inspect.signature(entry).parameters["scale"].kind
            is inspect.Parameter.POSITIONAL_ONLY
        )
        assert entry(tensor, _launcher=lambda *args, **kwargs: None) is tensor
        assert (
            entry(tensor, 4.5, offset=5.5, _launcher=lambda *args, **kwargs: None)
            is tensor
        )
        with pytest.raises(TypeError, match="positional.only"):
            entry(tensor, scale=4.5)


def test_keyword_only_kernel_parameters_remain_unsupported():
    def invalid(x, *, scale=2.5):
        return x

    with pytest.raises(TypeError, match="keyword-only"):
        helion.kernel(invalid)


@helion.kernel(autotune_effort="none")
def _tuple_default(x, shape: hl.constexpr = (2, (3, -0.0))):
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + shape[0] + shape[1][0]
    return x


@helion.kernel(autotune_effort="none")
def _complex_default(x, unused: hl.constexpr = 1 + 2j):
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + 1
    return x


def test_raw_source_preserves_recursive_immutable_tuple_default():
    tensor = torch.empty(17)
    with cpu_render_target():
        bound = _tuple_default.bind((tensor,))
        entry = PyCodeCache.load(bound.to_code())._tuple_default
        assert entry(tensor, _launcher=lambda *args, **kwargs: None) is tensor
        (default,) = entry.__defaults__
        assert default[:1] == (2,) and default[1][:1] == (3,)
        assert type(default[1][1]) is float and default[1][1].hex() == "-0x0.0p+0"


@pytest.mark.parametrize("value", [1 + 2j, complex(-0.0, -0.0)])
def test_raw_source_preserves_complex_literal_default(value):
    tensor = torch.empty(17)
    original = _complex_default.fn
    fn = FunctionType(
        original.__code__, original.__globals__, original.__name__, (value,)
    )
    kernel = helion.kernel(fn, autotune_effort="none")
    with cpu_render_target():
        # Complex runtime input tracing is not added by this test. The explicit
        # scalar is unused; the exported Python signature still has its default.
        bound = kernel.bind((tensor, 0.0))
        entry = PyCodeCache.load(bound.to_code())._complex_default
        assert entry(tensor, _launcher=lambda *args, **kwargs: None) is tensor
        default = inspect.signature(entry).parameters["unused"].default
        assert type(default) is complex
        assert default.real.hex() == value.real.hex()
        assert default.imag.hex() == value.imag.hex()


def test_tuple_with_mutable_element_remains_explicit_in_raw_source():
    tensor = torch.empty(17)
    original = _tuple_default.fn
    default = (2, [3, -0.0])
    fn = FunctionType(
        original.__code__, original.__globals__, original.__name__, (default,)
    )
    kernel = helion.kernel(fn, autotune_effort="none")
    # Isolated binding avoids the preexisting unhashable constexpr key issue.
    bound = cpu_bind(kernel, (tensor, default))
    entry = PyCodeCache.load(bound.to_code())._tuple_default
    assert entry.__defaults__ is None
    assert entry(tensor, default, _launcher=lambda *args, **kwargs: None) is tensor


@helion.kernel(autotune_effort="none")
def _callable_default(x, epilogue=lambda value: value + 1):
    for tile in hl.tile(x.numel()):
        x[tile] = epilogue(x[tile])
    return x


def test_normalized_arguments_keep_callable_default_for_raw_entry():
    tensor = torch.empty(17)
    kernel = helion.kernel(_callable_default.fn, autotune_effort="none")
    # Raw generated code deliberately does not recreate callback defaults.
    # Capture them before binding and use the same arguments for execution.
    args = kernel.normalize_args(tensor)
    assert args[0] is tensor
    assert args[1] is kernel.fn.__defaults__[0]
    assert args[1](2) == 3
    with cpu_render_target():
        bound = kernel.bind(args)
        entry = getattr(PyCodeCache.load(bound.to_code()), kernel.name)
        assert entry.__defaults__ is None
        with pytest.raises(TypeError, match="epilogue"):
            inspect.signature(entry).bind(tensor)
        assert entry(*args, _launcher=lambda *a, **kw: None) is tensor


def test_direct_search_callable_default_reaches_reloaded_host():
    tensor = torch.empty(17)
    kernel = helion.kernel(_callable_default.fn, autotune_effort="none")
    with cpu_render_target():
        bound = kernel.bind((tensor,))
        search = BaseSearch(bound, (tensor,))
        compiled = bound.compile_config()
        portable = _load_compiled_fn(
            pickle.loads(pickle.dumps(_serialize_compiled_fn(compiled)))
        )
        try:
            # Use the same by-code fallback as the worker's argument file.
            packed = make_portable_args(search.args)
            args = resolve_portable_args(
                pickle.loads(pickle.dumps(search.args if packed is None else packed))
            )
            assert portable(*args, _launcher=lambda *a, **kw: None) is args[0]
            assert search.args[1] is kernel.fn.__defaults__[0]
            assert args[1](2) == 3
        finally:
            _unload_compiled_fn(portable)


def test_direct_search_keeps_default_aliases_and_explicit_arguments():
    default = [2, 16]
    original = _alias_defaults.fn
    fn = FunctionType(
        original.__code__, original.__globals__, original.__name__, (default, default)
    )
    kernel = helion.kernel(fn, autotune_effort="none")
    tensor = torch.empty(17)
    with cpu_render_target():
        bound = kernel.bind((tensor,))
        search = BaseSearch(bound, (tensor,))
        assert search.args[1] is search.args[2] is default
        explicit = [3, 16]
        explicit_search = BaseSearch(bound, (tensor, explicit))
        assert explicit_search.args[1] is explicit
        assert explicit_search.args[2] is default
        # External autotunable adapters own their argument signature.
        adapter = SimpleNamespace(
            settings=kernel.settings, config_spec=bound.config_spec
        )
        opaque_args = [object()]
        assert BaseSearch(adapter, opaque_args).args is opaque_args


def test_direct_search_preserves_multi_shape_carrier_and_provider():
    kernel = helion.kernel(_factory_default.fn, autotune_effort="none")
    with cpu_render_target():
        cases = tuple(
            (kernel.bind((tensor,)), kernel.normalize_args(tensor))
            for tensor in (torch.empty(17), torch.empty(9))
        )
        carrier = _MultiShapeAutotuneArgs(
            cases=cases,
            aggregation="max",
            relative_to="default",
            cache_tag="default-arguments",
            workload_key=(17, 9),
            reference_latencies=(1.0, 2.0),
        )
        search = BaseSearch(cases[0][0], carrier)
        assert search.args is carrier
        with patch.object(
            LocalBenchmarkProvider, "_compute_baseline", return_value=(None, [], None)
        ):
            search._prepare()
        try:
            provider = search.benchmark_provider
            assert isinstance(provider, MultiShapeBenchmarkProvider)
            assert provider.args is carrier
            assert len(provider.children) == 2
            assert all(
                child.args is args
                for child, (_, args) in zip(provider.children, cases, strict=True)
            )
        finally:
            search.benchmark_provider.cleanup()


def _float_bits(value):
    return struct.pack("!d", value).hex()


def _default_float_bits(value):
    if type(value) is tuple:
        return tuple(map(_default_float_bits, value))
    if type(value) is complex:
        return complex, _float_bits(value.real), _float_bits(value.imag)
    assert type(value) is float
    return float, _float_bits(value)


@pytest.mark.parametrize(
    "bits",
    [
        "7ff8000000000042",
        "fff8000000000043",
        "7ff0000000000001",
        "fff0000000000001",
        "7ff0000000000000",
        "fff0000000000000",
    ],
)
@pytest.mark.parametrize("form", ["float", "tuple", "complex"])
@pytest.mark.parametrize("standalone", [False, True])
def test_raw_nonfinite_defaults_preserve_bits(bits, form, standalone, load_standalone):
    if standalone and matchesBackends(["metal"]):
        pytest.skip("Metal does not support dependency-free export")
    value = struct.unpack("!d", bytes.fromhex(bits))[0]
    if form == "complex":
        value = complex(value, value)
    elif form == "tuple":
        value = (value, (complex(-0.0, value), value))
    expected = _default_float_bits(value)
    original = _complex_default.fn
    fn = FunctionType(
        original.__code__, original.__globals__, original.__name__, (value,)
    )
    tensor = torch.empty(17)
    with (
        cpu_render_target(),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
    ):
        kernel = helion.kernel(fn, autotune_effort="none")
        # The explicitly bound scalar is unused. This tests the raw exported
        # Python signature, without extending complex input tracing or attaching
        # evaluated defaults through compile_config's private wrapper.
        bound = kernel.bind((tensor, 0.0))
        source = bound.to_code(
            options=helion.OutputCodeOptions(allow_helion_deps=not standalone)
        )
        module = load_standalone(source) if standalone else PyCodeCache.load(source)
        entry = getattr(module, kernel.name)
        arguments = inspect.signature(entry).bind(tensor)
        arguments.apply_defaults()
        assert _default_float_bits(arguments.arguments["unused"]) == expected
        if not standalone:
            assert entry(tensor, _launcher=lambda *args, **kwargs: None) is tensor
