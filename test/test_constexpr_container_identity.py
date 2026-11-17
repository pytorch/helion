from __future__ import annotations

import ast
from dataclasses import dataclass
import gc
import types
from unittest.mock import Mock
from unittest.mock import patch
import weakref

import pytest
import torch

import helion
from helion._compiler.compile_environment import RuntimeInputSpecialization
from helion._compiler.kernel_compiler import KernelCompiler
from helion._compiler.variable_origin import ArgumentOrigin
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import assert_scalar_addition
from helion._testing.binding_observation import observe_compiled_binding
from helion.autotuner import BooleanFragment
from helion.autotuner import LocalAutotuneCache
from helion.autotuner import StrictLocalAutotuneCache
from helion.autotuner.base_cache import _UncachedAutotuneCache
from helion.autotuner.remote_cache import RemoteAutotuneCache
from helion.autotuner.remote_cache import StrictRemoteAutotuneCache
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.settings import default_autotuner_fn

pytestmark = pytest.mark.cpu_only
FRAGMENT = BooleanFragment()
SAVED_TUPLE = (FRAGMENT,)
SAVED_LIST = [FRAGMENT]
SAVED_DICT = {"value": FRAGMENT}
SAVED_PLAIN = (4, 8)
REPEATED_ANCHORS = (SAVED_TUPLE, tuple([FRAGMENT]))


@helion.kernel(autotune_effort="none")
def _captured_identity(x, values: hl.constexpr, kind: hl.constexpr):
    alias = values
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if kind == "tuple":
            same = alias is SAVED_TUPLE
        elif kind == "nested":
            same = alias[0] is SAVED_TUPLE
        else:
            same = not (SAVED_PLAIN is not alias)
        if same:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@helion.kernel(autotune_effort="none")
def _argument_identity(
    x, left: hl.constexpr, right: hl.constexpr, nested: hl.constexpr
):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if nested:
            same = left[0] is right[0]
        else:
            same = left is right
        if same:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


def _check_routes(fn, shared, distinct):
    kernel = helion.kernel(fn, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "_call_with_bound_args", observe_compiled_binding),
    ):
        # Assert both fresh programs before asking dispatch to reuse either one.
        assert_scalar_addition(kernel.bind(shared), 2.0)
        assert_scalar_addition(
            helion.kernel(fn, autotune_effort="none").bind(distinct), 5.0
        )
        first = kernel(*shared)
        assert_scalar_addition(first, 2.0)
        prepared = kernel._prepared_call
        assert prepared is not None
        assert prepared.matches(kernel, shared)
        assert not prepared.matches(kernel, distinct)
        second = kernel(*distinct)
        assert first is not second
        assert_scalar_addition(second, 5.0)
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*shared) is first
            assert kernel(*distinct) is second


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["tuple", "nested", "plain"])
def test_captured_container_identity_preserves_fresh_and_cached_branches(kind):
    anchor = {
        "tuple": SAVED_TUPLE,
        "nested": SAVED_TUPLE,
        "plain": SAVED_PLAIN,
    }[kind]
    other = tuple(list(anchor))  # noqa: C414 -- force a distinct tuple identity
    if kind == "nested":
        anchor, other = (anchor,), (other,)
    tensor = torch.empty(17)
    _check_routes(
        _captured_identity.fn,
        (tensor, hl.constexpr(anchor), kind),
        (tensor, hl.constexpr(other), kind),
    )


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("nested", [False, True])
def test_argument_container_identity_preserves_alias_partition(nested):
    tensor = torch.empty(17)
    left = (FRAGMENT,)
    other = tuple([FRAGMENT])
    if nested:
        left, other = (left,), (other,)
    _check_routes(
        _argument_identity.fn,
        (tensor, left, left, nested),
        (tensor, left, other, nested),
    )


def _last_container(*values):
    return values[-1]


@helion.kernel(autotune_effort="none")
def _filtered_identity(x, left, right, form: hl.constexpr):
    choices = [value for value in (left, right) if value is right]
    if form == "helper":
        selected = _last_container(*choices)
    else:
        selected = choices[0]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if selected is left:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("form", ["index", "helper"])
@pytest.mark.parametrize("distinct_first", [False, True])
def test_filtered_identity_guards_selection_and_both_alias_orders(form, distinct_first):
    left = tuple([4, 8])
    other = tuple([4, 8])
    assert left == other and left is not other
    tensor = torch.empty(17)
    shared = (tensor, hl.constexpr(left), hl.constexpr(left), form)
    distinct = (tensor, hl.constexpr(left), hl.constexpr(other), form)
    ordered = [(shared, 2.0), (distinct, 5.0)]
    if distinct_first:
        ordered.reverse()
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "_call_with_bound_args", observe_compiled_binding),
    ):
        kernel = helion.kernel(_filtered_identity.fn, autotune_effort="none")
        bounds = []
        for args, expected in ordered:
            fresh = helion.kernel(_filtered_identity.fn, autotune_effort="none")
            assert_scalar_addition(fresh.bind(args), expected)
            bound = kernel(*args)
            assert_scalar_addition(bound, expected)
            bounds.append(bound)
        assert bounds[0] is not bounds[1]
        if form == "helper":
            for args, expected in ordered * 2:
                assert_scalar_addition(kernel(*args), expected)
                assert_scalar_addition(kernel.bind(args), expected)
            assert all(bound.env.requires_fresh_binding for bound in bounds)
            assert not kernel._bound_kernels
            assert not kernel._dispatch_cache
            assert kernel._prepared_call is None
            return
        if any(bound.env.requires_fresh_binding for bound in bounds):
            # A filtered host temporary may have no proved input path. Its
            # later anchor cannot be published under the first trace's schema.
            for args, expected in ordered * 2:
                fresh = helion.kernel(_filtered_identity.fn, autotune_effort="none")
                assert_scalar_addition(fresh.bind(args), expected)
                assert_scalar_addition(kernel(*args), expected)
                assert_scalar_addition(kernel.bind(args), expected)
            assert all(
                not bound.env.requires_fresh_binding
                for bound in kernel._bound_kernels.values()
            )
            assert all(
                not bound.env.requires_fresh_binding
                for bound in kernel._dispatch_cache.values()
            )
            return
        # A deeper trace may extend the schema and evict the earlier binding,
        # whose newly required facts were unknown. Revisit each route once to
        # populate the complete schema before checking steady cache reuse.
        bounds = []
        for args, expected in ordered:
            bound = kernel(*args)
            assert_scalar_addition(bound, expected)
            bounds.append(bound)
        assert bounds[0] is not bounds[1]
        prepared = kernel._prepared_call
        assert prepared is not None
        assert prepared.matches(kernel, ordered[-1][0])
        assert not prepared.matches(kernel, ordered[0][0])
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            for (args, expected), bound in zip(ordered * 2, bounds * 2, strict=True):
                assert kernel(*args) is bound
                assert_scalar_addition(bound, expected)


@onlyBackends(["triton", "cute"])
def test_filtered_identity_guards_do_not_own_tensor_inputs():
    tensor = torch.empty(17)
    shared = (tensor,)
    with cpu_render_target():
        kernel = helion.kernel(_filtered_identity.fn, autotune_effort="none")
        bound = kernel.bind((tensor, shared, shared, "helper"))
        assert_scalar_addition(bound, 2.0)
        reference = weakref.ref(tensor)
        del tensor, shared
        gc.collect()
        assert reference() is None
        assert not bound.env._container_identities
        first = bound.to_code(helion.Config(block_sizes=[32]))
        second = bound.to_code(helion.Config(block_sizes=[64]))
        assert first != second
        assert bound.to_code(helion.Config(block_sizes=[32])) == first
        assert reference() is None


def _captured_global_container():
    return SAVED_PLAIN


def _captured_default_container(value=SAVED_PLAIN):
    return value


def _capture_container(value):
    def inner():
        return value

    return inner


@helion.kernel(autotune_effort="none")
def _captured_result_identity(
    x, values: hl.constexpr, fn, form: hl.constexpr, reverse: hl.constexpr = False
):
    if form == "global":
        selected = _captured_global_container()
    else:
        selected = fn()
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if reverse:
            same = values is selected
        else:
            same = selected is values
        if same:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    ("form", "distinct_first", "reverse"),
    # Pairwise capture-kind, first-call order and comparison operand order.
    [
        ("global", False, False),
        ("global", True, True),
        ("default", False, True),
        ("default", True, False),
        ("closure", False, False),
        ("closure", True, True),
    ],
)
def test_helper_captured_containers_guard_both_alias_orders(
    form, distinct_first, reverse
):
    tensor = torch.empty(17)
    other = tuple(list(SAVED_PLAIN))  # noqa: C414 -- require a distinct identity
    fn = {
        "global": None,
        "default": _captured_default_container,
        "closure": _capture_container(SAVED_PLAIN),
    }[form]
    ordered = [
        ((tensor, SAVED_PLAIN, fn, form, reverse), 2.0),
        ((tensor, other, fn, form, reverse), 5.0),
    ]
    if distinct_first:
        ordered.reverse()
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "_call_with_bound_args", observe_compiled_binding),
    ):
        kernel = helion.kernel(_captured_result_identity.fn, autotune_effort="none")
        bounds = []
        for args, expected in ordered:
            fresh = helion.kernel(_captured_result_identity.fn, autotune_effort="none")
            assert_scalar_addition(fresh.bind(args), expected)
            bound = kernel(*args)
            assert_scalar_addition(bound, expected)
            bounds.append(bound)
        assert bounds[0] is not bounds[1]
        for (args, expected), previous in zip(ordered * 2, bounds * 2, strict=True):
            rebound = kernel(*args)
            assert rebound is not previous
            assert_scalar_addition(rebound, expected)
            assert_scalar_addition(previous(*args), expected)
            assert_scalar_addition(kernel.bind(args), expected)
            assert rebound.env.requires_fresh_binding
            assert not rebound.is_cacheable()
        assert not kernel._bound_kernels
        assert not kernel._dispatch_cache
        assert not kernel._specialization_aliases
        assert not kernel._specialize_extra
        assert kernel._prepared_call is None


_CAPTURE_MODULE = types.ModuleType("container_capture")
_CAPTURE_MODULE.anchor = SAVED_PLAIN


class _CaptureClass:
    anchor = SAVED_PLAIN


@dataclass
class _CaptureReceiver:
    anchor: tuple[int, ...]

    def result(self):
        return self.anchor


_CAPTURE_RECEIVER = _CaptureReceiver(SAVED_PLAIN)


def _receiver_container():
    return _CAPTURE_RECEIVER.result()


def _module_container():
    return _CAPTURE_MODULE.anchor


def _class_container():
    return _CaptureClass.anchor


def _indirect_container():
    return _captured_global_container()


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["module", "class", "receiver", "indirect"])
def test_opaque_capture_boundaries_never_reuse_bindings(kind):
    functions = {
        "module": _module_container,
        "class": _class_container,
        "receiver": _receiver_container,
        "indirect": _indirect_container,
    }
    tensor = torch.empty(17)
    other = tuple([4, 8])
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "_call_with_bound_args", observe_compiled_binding),
    ):
        kernel = helion.kernel(_captured_result_identity.fn, autotune_effort="none")
        for value, expected in [(SAVED_PLAIN, 2.0), (other, 5.0)] * 2:
            form = "call"
            args = (tensor, value, functions[kind], form)
            first = kernel(*args)
            fresh = kernel.bind(args)
            assert_scalar_addition(first, expected)
            assert_scalar_addition(fresh, expected)
            assert first is not fresh
            assert first.env.requires_fresh_binding
        assert not kernel._bound_kernels
        assert not kernel._dispatch_cache
        assert kernel._prepared_call is None
        assert not kernel._specialization_aliases


_ITERATED_CAPTURES = (SAVED_PLAIN, tuple([4, 8]))


@helion.kernel(autotune_effort="none")
def _iterated_capture(x, values: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for index in hl.static_range(2):
            selected = _last_container(_ITERATED_CAPTURES[index])
            if selected is values:
                result = result + 2.0
            else:
                result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
def test_actual_unrolled_opaque_calls_refuse_cache_publication():
    tensor = torch.empty(17)
    other = _ITERATED_CAPTURES[1]
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "_call_with_bound_args", observe_compiled_binding),
    ):
        kernel = helion.kernel(_iterated_capture.fn, autotune_effort="none")
        for value, expected in [(SAVED_PLAIN, [2.0, 5.0]), (other, [5.0, 2.0])] * 2:
            bound = kernel(tensor, value)
            additions = [
                node.args[1]
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.target is torch.ops.aten.add.Tensor
            ]
            assert additions == expected
            assert bound.env.requires_fresh_binding
        assert not kernel._bound_kernels
        assert not kernel._dispatch_cache
        assert kernel._prepared_call is None


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("selection", ["set_config", "autotune"])
def test_fresh_identity_bind_preserves_explicit_configuration(selection):
    tensor = torch.empty(17)
    args = (tensor, SAVED_PLAIN, None, "global", False)
    config = helion.Config(block_sizes=[32])
    compiled = []

    def compile_without_device(bound, selected, *, allow_print=True):
        # Exercise the normal config validator and source renderer, replacing
        # only device compilation/launch with an observable CPU callable.
        code = bound.to_code(selected)
        ast.parse(code)
        compiled.append((bound, selected, code))
        return lambda *values: bound

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "compile_config", compile_without_device),
    ):
        kernel = helion.kernel(_captured_result_identity.fn, autotune_effort="none")
        bound = kernel.bind(args)
        if selection == "set_config":
            bound.set_config(config)
        else:
            with patch.object(bound.env.backend, "autotune", return_value=config):
                assert bound.autotune(args) == config
        saved_launcher = bound._run
        assert saved_launcher is not None
        assert saved_launcher(*args) is bound
        for value, expected in [(SAVED_PLAIN, 2.0), (tuple([4, 8]), 5.0)]:
            current = (tensor, value, None, "global", False)
            rebound = bound(*current)
            assert rebound is not bound
            assert rebound._config == config
            assert rebound._run is not saved_launcher
            assert_scalar_addition(rebound, expected)
        # A compile_config-returned function deliberately remains the compiled
        # one-shot program; the public bound invocation performs the rebind.
        direct = bound.compile_config(helion.Config(block_sizes=[64]))
        assert direct(*args) is bound
        assert compiled[-1][1].block_sizes == [64]
        assert not kernel._bound_kernels
        assert not kernel._dispatch_cache
        assert kernel._prepared_call is None


@onlyBackends(["triton", "cute"])
def test_fresh_identity_bind_snapshots_reusable_input_facts():
    original_compile = KernelCompiler.compile
    key = "test_input_alignment"

    def compile_with_input_fact(compiler, *args, **kwargs):
        result = original_compile(compiler, *args, **kwargs)
        compiler.env.register_runtime_input_specialization(
            key,
            RuntimeInputSpecialization(
                sources=(ArgumentOrigin("x").to_source(),),
                classifier_identity=(key,),
                classifier=lambda values: values[0].data_ptr() % 16 == 0,
                reusable_tensor_properties=frozenset({"data_ptr"}),
            ),
        )
        return result

    tensor = torch.empty(17)
    with (
        cpu_render_target(),
        patch.object(KernelCompiler, "compile", compile_with_input_fact),
    ):
        kernel = helion.kernel(_captured_result_identity.fn, autotune_effort="none")
        bound = kernel.bind((tensor, SAVED_PLAIN, None, "global", False))
        assert not bound._cache_managed
        assert bound.env.bound_runtime_input_specialization_results[key] is True
        assert_scalar_addition(bound, 2.0)
        assert bound.env.bound_runtime_input_specialization_results[key] is True


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("trials", [1, 2])
@pytest.mark.parametrize("form", ["default", "bracket", "direct", "strict_direct"])
def test_autotuner_factories_preserve_trials_without_cache_io(trials, form):
    calls = []

    class Search:
        def __init__(self, kernel, args, **kwargs):
            self.kernel = kernel
            self.args = args
            self.settings = kernel.settings
            self.best_perf_so_far = 1.0

        @staticmethod
        def get_kwargs_from_profile(profile, settings):
            return {}

        def log(self, message):
            pass

        def autotune(self, *, skip_cache=False):
            seed = self.settings.autotune_random_seed
            calls.append((self, seed, skip_cache))
            return helion.Config(block_sizes=[32 if seed == 41 else 64])

    tensor = torch.empty(17)
    args = (tensor, SAVED_PLAIN, None, "global", False)
    with cpu_render_target(), patch("helion.autotuner.LFBOTreeSearch", Search):
        kernel = helion.kernel(
            _captured_result_identity.fn,
            autotune_effort="none",
            autotune_random_seed=41,
            autotune_best_of_k=trials,
        )
        bound = kernel.bind(args)
        # Use the actual public default factory and real cache orchestration.
        # This private wrapper's get/put/key/list methods fail if called.

        def direct_factory(kernel, args):
            def make_search():
                return Search(kernel, args)

            if form == "strict_direct":
                return StrictLocalAutotuneCache(
                    autotuner=make_search(), autotuner_factory=make_search
                )
            return LocalAutotuneCache(make_search(), autotuner_factory=make_search)

        factory = (
            default_autotuner_fn
            if form == "default"
            else LocalAutotuneCache[Search]
            if form == "bracket"
            else direct_factory
        )
        autotuner = factory(bound, args)
        assert type(autotuner) is _UncachedAutotuneCache
        original_search = autotuner.autotuner
        with (
            patch.object(
                autotuner, "_run_rebench_round", return_value=[2.0, 1.0]
            ) as rebench,
            patch.object(autotuner, "_release_trial_state") as release,
        ):
            selected = autotuner.autotune()
        assert selected.block_sizes == [32 if trials == 1 else 64]
        assert [seed for _, seed, _ in calls] == list(range(41, 41 + trials))
        assert all(skip for _, _, skip in calls)
        assert len({id(search) for search, _, _ in calls}) == trials
        assert rebench.call_count == int(trials > 1)
        assert release.call_count == (0 if trials == 1 else trials + 1)
        assert autotuner.autotuner is original_search
        assert bound.settings.autotune_random_seed == 41
        assert not kernel._bound_kernels
        # The fallback is specific to real fresh-only bindings; unrelated
        # external kernel protocols keep the existing cache refusal.
        external = types.SimpleNamespace(
            settings=bound.settings, is_cacheable=lambda: False
        )
        with pytest.raises(TypeError, match="External kernels are not cacheable"):
            factory(external, args)


@helion.kernel(autotune_effort="none")
def _disjoint_identity(x, values: hl.constexpr, other: hl.constexpr):
    selected = _last_container(values)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if selected is not other:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("other", [None, 7])
def test_type_disjoint_identity_adds_no_alias_or_opaque_reuse_guards(other):
    tensor = torch.empty(17)
    args = (tensor, SAVED_PLAIN, other)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "_call_with_bound_args", observe_compiled_binding),
    ):
        kernel = helion.kernel(_disjoint_identity.fn, autotune_effort="none")
        first = kernel(*args)
        assert_scalar_addition(first, 2.0)
        assert not first.env.requires_fresh_binding
        # Backends can add their own alignment guards during rendering; the
        # disjoint comparison contributes no identity/alias classifier.
        assert not any(
            key.startswith("container_identity:")
            for key in first.env.runtime_input_specializations
        )
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(tensor, tuple([4, 8]), other) is first


@helion.kernel(autotune_effort="none")
def _global_identity(x, kind: hl.constexpr):
    if kind == "list":
        alias = SAVED_LIST
    else:
        alias = SAVED_DICT
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if kind == "list":
            same = alias is SAVED_LIST
            copied = [FRAGMENT]
            different = copied is not SAVED_LIST
        else:
            same = alias is SAVED_DICT
            copied = {"value": FRAGMENT}
            different = copied is not SAVED_DICT
        if same and different:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["list", "dict"])
def test_global_mutable_container_alias_and_fresh_construction(kind):
    # List/dict constexpr argument keys remain unsupported. Global metadata
    # aliases exercise these existing container types without extending that API.
    with cpu_render_target():
        kernel = helion.kernel(_global_identity.fn, autotune_effort="none")
        assert_scalar_addition(kernel.bind((torch.empty(17), kind)), 2.0)


@helion.kernel(autotune_effort="none")
def _tensor_container(x, values):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        local = (values[0],)
        alias = local
        if local is alias and local is not values:
            out[tile] = values[0][tile] + 2.0
        else:
            out[tile] = values[0][tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_fresh_container_identity_keeps_tensor_proxy_tracing():
    tensor = torch.empty(17)
    with cpu_render_target():
        kernel = helion.kernel(_tensor_container.fn, autotune_effort="none")
        bound = kernel.bind((tensor, (tensor,)))
        assert_scalar_addition(bound, 2.0)
        assert not bound.env._container_identities
        reference = weakref.ref(tensor)
        del tensor
        gc.collect()
        assert reference() is None
        # Keep the compiled bound live while checking the input was released.
        assert bound.host_function.device_ir.graphs
        first = bound.to_code(helion.Config(block_sizes=[32]))
        second = bound.to_code(helion.Config(block_sizes=[64]))
        assert first != second
        assert bound.to_code(helion.Config(block_sizes=[32])) == first
        assert_scalar_addition(bound, 2.0)
        assert reference() is None
        assert not bound.env._container_identities


@onlyBackends(["triton", "cute"])
def test_failed_frontend_releases_container_identity_witnesses():
    tensor = torch.empty(17)
    witness = []
    environments = []

    def fail(compiler, hf):
        environments.append(compiler.env)
        witness.extend(
            entry[1] for entry in compiler.env._container_identities.values()
        )
        raise RuntimeError("stop frontend after argument conversion")

    with cpu_render_target(), patch.object(KernelCompiler, "lower", fail):
        kernel = helion.kernel(_tensor_container.fn, autotune_effort="none")
        with pytest.raises(RuntimeError, match="stop frontend"):
            kernel.bind((tensor, (tensor,)))
    assert witness
    assert all(item.value is None for item in witness)
    assert all(not env._container_identities for env in environments)


@helion.kernel(autotune_effort="none")
def _repeated_identity(x, value: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for index in hl.static_range(2):
            if value is REPEATED_ANCHORS[index]:
                result = result + 2.0
            else:
                result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
def test_repeated_identity_comparisons_keep_distinct_anchors_after_rendering():
    tensor = torch.empty(17)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "_call_with_bound_args", observe_compiled_binding),
    ):
        kernel = helion.kernel(_repeated_identity.fn, autotune_effort="none")
        bounds = [kernel(tensor, anchor) for anchor in REPEATED_ANCHORS]
        assert bounds[0] is not bounds[1]
        for bound, expected in zip(bounds, ([2.0, 5.0], [5.0, 2.0]), strict=True):
            additions = [
                node.args[1]
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.target is torch.ops.aten.add.Tensor
            ]
            assert additions == expected
            before = tuple(bound.env.runtime_input_specializations.values())
            first = bound.to_code(helion.Config(block_sizes=[32]))
            ast.parse(bound.to_code(helion.Config(block_sizes=[64])))
            assert bound.to_code(helion.Config(block_sizes=[32])) == first
            after = tuple(bound.env.runtime_input_specializations.values())
            assert all(a is b for a, b in zip(before, after, strict=True))
            assert not bound.env._container_identities
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            for anchor, bound in zip(REPEATED_ANCHORS, bounds, strict=True):
                assert kernel(tensor, anchor) is bound


@helion.kernel(autotune_effort="none")
def _host_identity(x, values: hl.constexpr):
    same = values is SAVED_TUPLE
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if same:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out, same


@onlyBackends(["triton", "cute"])
def test_host_identity_remains_a_dynamic_wrapper_comparison():
    tensor = torch.empty(17)
    with cpu_render_target():
        kernel = helion.kernel(_host_identity.fn, autotune_effort="none")
        bound = kernel.bind((tensor, SAVED_TUPLE))
        source = ast.parse(bound.to_code())
        assert kernel.bind((tensor, tuple([FRAGMENT]))) is bound
        additions = [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        ]
        assert sorted(additions) == [2.0, 5.0]
    host = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_host_identity"
    )
    namespace = {"torch": torch, "hl": hl}
    namespace.update(
        (value.alias, value.value)
        for value in bound.host_function.global_imports.values()
    )
    launcher = Mock()
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node is not host:
            namespace[node.name] = object()
    for default in host.args.kw_defaults:
        if isinstance(default, ast.Name):
            namespace[default.id] = launcher
    module = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), host], []
    )
    exec(
        compile(ast.fix_missing_locations(module), "<container_identity>", "exec"),
        namespace,
    )
    for value, expected in ((SAVED_TUPLE, True), (tuple([FRAGMENT]), False)):
        _, same = namespace[host.name](tensor, value)
        assert same is expected
    assert launcher.call_count == 2


def test_local_cache_constructor_preserves_ordinary_and_custom_initializers():
    calls = []

    class CustomCache(LocalAutotuneCache):
        def __init__(self, marker, *, search):
            calls.append(marker)
            super().__init__(search)
            self.marker = marker

    tensor = torch.empty(17)
    with cpu_render_target():
        ordinary_args = (tensor, SAVED_PLAIN, SAVED_PLAIN, False)
        ordinary = helion.kernel(_argument_identity.fn).bind(ordinary_args)
        fresh_args = (tensor, SAVED_PLAIN, None, "global", False)
        fresh = helion.kernel(_captured_result_identity.fn).bind(fresh_args)
        assert not ordinary.env.requires_fresh_binding
        assert fresh.env.requires_fresh_binding
        search = types.SimpleNamespace(
            kernel=ordinary, args=ordinary_args, settings=ordinary.settings
        )
        key = object()
        for cache_cls in (LocalAutotuneCache, StrictLocalAutotuneCache):
            with patch.object(cache_cls, "_generate_key", return_value=key) as generate:
                wrapper = cache_cls(search)
                assert type(wrapper) is cache_cls and wrapper.key is key
                generate.assert_called_once_with()
        marker = object()
        with patch.object(LocalAutotuneCache, "_generate_key", return_value=key):
            custom = CustomCache(marker, search=search)
            assert type(custom) is CustomCache and custom.marker is marker
            assert custom.key is key
            fresh_search = types.SimpleNamespace(
                kernel=fresh, args=fresh_args, settings=fresh.settings
            )
            with pytest.raises(TypeError, match="explicit cache constructor"):
                CustomCache(marker, search=fresh_search)
        assert calls == [marker, marker]


@pytest.mark.parametrize("cache_cls", (RemoteAutotuneCache, StrictRemoteAutotuneCache))
def test_fresh_remote_caches_require_factory_before_cache_initialization(
    cache_cls,
):
    class Search:
        def __init__(self, kernel, args):
            self.kernel = kernel
            self.args = args
            self.settings = kernel.settings

    tensor = torch.empty(17)
    args = (tensor, SAVED_PLAIN, None, "global", False)
    with (
        cpu_render_target(),
        patch(
            "helion.autotuner.remote_cache._load_remote_backend",
            side_effect=AssertionError("remote cache initialized"),
        ),
    ):
        bound = helion.kernel(_captured_result_identity.fn).bind(args)
        assert bound.env.requires_fresh_binding
        with pytest.raises(TypeError, match="explicit cache constructor"):
            cache_cls(Search(bound, args))
        wrapper = cache_cls[Search](bound, args)
        assert type(wrapper) is _UncachedAutotuneCache
        assert type(wrapper.autotuner) is Search
        assert wrapper.autotuner.kernel is bound
        assert wrapper.autotuner.args is args
