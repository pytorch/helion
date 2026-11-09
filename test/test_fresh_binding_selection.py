from __future__ import annotations

import ast
from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion.autotuner import EnumFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.settings import default_autotuner_fn

pytestmark = pytest.mark.cpu_only

SAVED = (4, 8)
DOMAIN = (2, 3)


def _saved():
    return SAVED


@helion.kernel()
def _identity_kernel(x, value: hl.constexpr):
    factor = hl.register_tunable("factor", EnumFragment(DOMAIN))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        selected = _saved()
        offset = 2.0 if value is selected else 5.0
        out[tile] = x[tile] * factor + offset
    return out


def _compile(bound, config, *, allow_print=True):
    ast.parse(bound.to_code(config))
    return lambda *args: bound


@pytest.mark.parametrize("explicit", [False, True])
def test_kernel_retains_selected_config_across_fresh_traces(explicit):
    tensor = torch.empty(17)
    args = (tensor, SAVED)
    tuned = []

    def tune(backend, bound, args, **kwargs):
        tuned.append(bound)
        with bound.env:
            config = bound.config_spec.default_config()
        config.config.update(block_sizes=[64], factor=3)
        return config

    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_identity_kernel.fn)
        probe = kernel.bind(args)
        assert probe.env.requires_fresh_binding
        with patch.object(type(probe.env.backend), "autotune", tune):
            if explicit:
                assert kernel.autotune(args).config["factor"] == 3
            outputs = [kernel(*args) for _ in range(3)]
        assert len(tuned) == 1
        assert all(out._config.config["factor"] == 3 for out in outputs)
        assert len({id(out) for out in outputs}) == 3
        assert not kernel._bound_kernels and not kernel._dispatch_cache
        assert kernel._prepared_call is None


@onlyBackends(["triton", "cute", "pallas"])
def test_autotune_multi_retains_selection_for_fresh_bindings():
    tensor = torch.empty(17)
    args = (tensor, SAVED)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "compile_config", _compile),
        patch(
            "helion.runtime.kernel._canonicalize_multi_shape_device",
            return_value=torch.device("cuda", 0),
        ),
        patch("helion.runtime.kernel._current_device_index", return_value=0),
    ):
        kernel = helion.kernel(_identity_kernel.fn)
        probe = kernel.bind(args)
        with probe.env:
            config = probe.config_spec.default_config()
        config.config.update(block_sizes=[64], factor=3)
        with patch.object(
            type(probe.env.backend), "autotune", return_value=config
        ) as tune:
            other = (tensor, tuple([4, 8]))
            assert (
                kernel.autotune_multi([args, other], cache_tag="fresh-selection")
                == config
            )
            assert kernel(*args)._config.config["factor"] == 3
            assert kernel(*other)._config.config["factor"] == 3
        assert tune.call_count == 1


@pytest.mark.parametrize("bypass", [None, "force", "environment"])
def test_assert_cache_hit_refuses_uncached_search(monkeypatch, bypass):
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
            calls.append(skip_cache)
            return helion.Config(block_sizes=[64], factor=3)

    with cpu_render_target(), patch("helion.autotuner.LFBOTreeSearch", Search):
        kernel = helion.kernel(_identity_kernel.fn)
        args = (torch.empty(17), SAVED)
        bound = kernel.bind(args)
        assert bound.env.requires_fresh_binding
        autotuner = default_autotuner_fn(bound, args)
        monkeypatch.setenv("HELION_ASSERT_CACHE_HIT", "1")
        if bypass is None:
            with pytest.raises(helion.exc.CacheAssertionError):
                autotuner.autotune()
            assert not calls
        else:
            if bypass == "environment":
                monkeypatch.setenv("HELION_SKIP_CACHE", "1")
            assert (
                autotuner.autotune(skip_cache=bypass == "force").config["factor"] == 3
            )
            assert calls == [True]


@helion.kernel()
def _conditional_identity(x, value: hl.constexpr):
    factor = hl.register_tunable("factor", EnumFragment(DOMAIN))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if factor == 2:
            out[tile] = x[tile] + 1
        else:
            offset = 2.0 if value is _saved() else 5.0
            out[tile] = x[tile] + offset
    return out


def _choice(bound, factor=3):
    with bound.env:
        config = bound.config_spec.default_config()
    config.config.update(block_sizes=[64], factor=factor)
    return config


@pytest.mark.parametrize("static_shapes", [False, True])
def test_fresh_selection_retains_input_layout(static_shapes):
    tensors = (torch.empty(17), torch.empty(34)[::2], torch.empty(18)[1:])
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_identity_kernel.fn, static_shapes=static_shapes)
        backend = type(kernel.bind((tensors[0], SAVED)).env.backend)
        with patch.object(
            backend, "autotune", side_effect=lambda bound, args, **kw: _choice(bound)
        ) as tune:
            selected = [kernel(tensor, SAVED) for tensor in tensors]
            assert tune.call_count == 3
            assert len({bound._fresh_selection_key for bound in selected}) == 3
            for tensor, previous in zip(tensors, selected, strict=True):
                current = kernel(tensor, SAVED)
                assert current is not previous
                assert current._config == previous._config
            # Allocation identity alone must not invalidate a matching layout.
            assert kernel(torch.empty(17), SAVED)._config == selected[0]._config
            assert tune.call_count == 3


def test_selection_uses_all_frontend_branches_and_changed_domains(monkeypatch):
    import sys

    tensor = torch.empty(17)
    args = (tensor, SAVED)
    calls = []

    def tune(backend, bound, args, **kwargs):
        calls.append(bound)
        return _choice(bound, DOMAIN[-1])

    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_conditional_identity.fn)
        backend = type(kernel.bind(args).env.backend)
        with patch.object(backend, "autotune", tune):
            first = kernel(*args)
            other = kernel(tensor, tuple([4, 8]))
            again = kernel(*args)
            assert len(calls) == 2
            assert first is not again and first._config == again._config
            # The default-factor branch is identical; the unselected branch
            # must still distinguish the complete current frontend program.
            assert first._fresh_selection_key != other._fresh_selection_key
            for bound, expected in ((first, 2.0), (other, 5.0)):
                assert any(
                    node.target is torch.ops.aten.add.Tensor
                    and node.args[1] == expected
                    for graph in bound.host_function.device_ir.graphs
                    for node in graph.graph.nodes
                )
            monkeypatch.setattr(sys.modules[__name__], "DOMAIN", (2, 4))
            changed = kernel(*args)
            assert len(calls) == 3 and changed._config.config["factor"] == 4
            assert changed._fresh_selection_key != first._fresh_selection_key
            assert kernel(*args)._config.config["factor"] == 4
            assert len(calls) == 3


def test_force_skip_reset_and_explicit_bound_selection(monkeypatch):
    tensor = torch.empty(17)
    args = (tensor, SAVED)
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_identity_kernel.fn)
        probe = kernel.bind(args)
        with patch.object(
            type(probe.env.backend),
            "autotune",
            side_effect=lambda bound, args, **kw: _choice(bound),
        ) as tune:
            kernel.autotune(args)
            kernel(*args)
            assert tune.call_count == 1
            kernel.autotune(args, force=True)
            assert tune.call_count == 2
            kernel.autotune(args, force=False)
            assert tune.call_count == 2
            with monkeypatch.context() as context:
                context.setenv("HELION_SKIP_CACHE", "1")
                kernel(*args)
                kernel(*args)
            assert tune.call_count == 4
            kernel(*args)
            assert tune.call_count == 4
            kernel.settings.force_autotune = True
            kernel(*args)
            assert tune.call_count == 5
            kernel.settings.force_autotune = False
            bound = kernel.bind(args)
            bound.set_config(_choice(bound, 2))
            assert bound(*args)._config.config["factor"] == 2
            assert kernel(*args)._config.config["factor"] == 3
            assert tune.call_count == 5
            kernel.reset()
            assert not kernel._fresh_selections
            kernel(*args)
            assert tune.call_count == 6


class CustomDomain(EnumFragment):
    pass


@helion.kernel()
def _custom_identity(x, value: hl.constexpr):
    factor = hl.register_tunable("factor", CustomDomain((2, 3)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        offset = 2.0 if value is _saved() else 5.0
        out[tile] = x[tile] * factor + offset
    return out


def test_custom_domain_retunes_with_one_warning(caplog):
    args = (torch.empty(17), SAVED)
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_custom_identity.fn)
        probe = kernel.bind(args)
        with patch.object(
            type(probe.env.backend),
            "autotune",
            side_effect=lambda bound, args, **kw: _choice(bound),
        ) as tune:
            kernel(*args)
            kernel(*args)
        assert tune.call_count == 2
        assert not kernel._fresh_selections
        assert (
            sum(
                "selected configs cannot be retained" in row.message
                for row in caplog.records
            )
            == 1
        )


def test_custom_domain_fixed_config_does_not_warn_about_retuning(caplog):
    args = (torch.empty(17), SAVED)
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        probe = helion.kernel(_custom_identity.fn).bind(args)
        kernel = helion.kernel(_custom_identity.fn, config=_choice(probe, 2))
        with patch.object(
            type(probe.env.backend),
            "autotune",
            side_effect=AssertionError("fixed config must not tune"),
        ):
            outputs = [kernel(*args) for _ in range(2)]
        assert all(bound._config.config["factor"] == 2 for bound in outputs)
        assert not any(
            "Calls may repeat autotuning" in row.message for row in caplog.records
        )


def test_unconfigured_opaque_eviction_does_not_warn_about_retuning(caplog):
    # The eviction path must not evaluate the selection key before a config
    # exists. Its property can warn about an otherwise irrelevant custom domain.
    with cpu_render_target():
        kernel = helion.kernel(_custom_identity.fn)
        bound = kernel.bind((torch.empty(17), SAVED))
        assert bound._config is None
        assert "_fresh_selection_key" not in bound.__dict__
        bound.env.has_opaque_fragment_inputs = True
        kernel._opaque_bindings = OrderedDict([(bound, None)])
        with patch("helion.runtime.kernel._OPAQUE_BINDING_CACHE_SIZE", 0):
            kernel._retain_opaque_binding(bound)
        assert not kernel._opaque_bindings
        assert "_fresh_selection_key" not in bound.__dict__
        assert not any(
            "Calls may repeat autotuning" in row.message for row in caplog.records
        )


@helion.kernel()
def _optional_identity(x, value: hl.constexpr, opaque: hl.constexpr):
    factor = hl.register_tunable("factor", EnumFragment(DOMAIN))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if opaque:
            offset = 2.0 if value is _saved() else 5.0
        else:
            offset = 2.0 if value is SAVED else 5.0
        out[tile] = x[tile] * factor + offset
    return out


@onlyBackends(["triton", "cute", "pallas"])
def test_mixed_multi_shape_uses_uncached_factory_for_all_cases():
    from helion.autotuner.base_cache import _UncachedAutotuneCache

    class Search:
        def __init__(self, kernel, args, **kwargs):
            self.kernel, self.args, self.settings = kernel, args, kernel.settings

        @staticmethod
        def get_kwargs_from_profile(profile, settings):
            return {}

    tensor = torch.empty(17)
    args = [(tensor, SAVED, False), (tensor, SAVED, True)]
    calls = []

    def tune(backend, bound, multi, **kwargs):
        calls.append(multi)
        assert bound.is_cacheable()
        assert multi.cases[1][0].env.requires_fresh_binding
        # Actual default factory must choose no-I/O even with a reusable anchor.
        wrapper = default_autotuner_fn(bound, multi)
        assert type(wrapper) is _UncachedAutotuneCache
        assert multi.workload_key[-1][1] == ("fresh_case", 1)
        return _choice(bound)

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "compile_config", _compile),
        patch("helion.autotuner.LFBOTreeSearch", Search),
        patch(
            "helion.runtime.kernel._canonicalize_multi_shape_device",
            return_value=torch.device("cuda", 0),
        ),
        patch("helion.runtime.kernel._current_device_index", return_value=0),
    ):
        kernel = helion.kernel(_optional_identity.fn)
        backend = type(kernel.bind(args[0]).env.backend)
        with patch.object(backend, "autotune", tune):
            kernel.autotune_multi(args, cache_tag="mixed-fresh")
            assert kernel(*args[0])._config.config["factor"] == 3
            assert kernel(*args[1])._config.config["factor"] == 3
        assert len(calls) == 1
        assert len(kernel._fresh_selections) == 1


@pytest.mark.parametrize("cached_first", [False, True])
def test_direct_fresh_call_does_not_overwrite_reusable_selection(cached_first):
    tensor = torch.empty(17)
    fresh_args = (tensor, SAVED, True)
    cached_args = (tensor, SAVED, False)
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_optional_identity.fn)
        if cached_first:
            cached = kernel.bind(cached_args)
            fresh = kernel.bind(fresh_args)
        else:
            fresh = kernel.bind(fresh_args)
            cached = kernel.bind(cached_args)
        assert fresh.env.requires_fresh_binding
        assert not cached.env.requires_fresh_binding
        cached.set_config(_choice(cached, 3))
        fresh.set_config(_choice(fresh, 2))
        assert fresh(*cached_args) is cached
        assert cached._config.config["factor"] == 3
        assert kernel(*cached_args) is cached
        assert cached._config.config["factor"] == 3
        rebound = fresh(*fresh_args)
        assert rebound is not fresh and rebound.env.requires_fresh_binding
        assert rebound._config.config["factor"] == 2


@pytest.mark.parametrize("cached_first", [False, True])
@pytest.mark.parametrize("selection", ["set_config", "autotune"])
def test_direct_fresh_call_supplies_unconfigured_shared_binding(
    cached_first, selection
):
    tensor = torch.empty(17)
    fresh_args = (tensor, SAVED, True)
    shared_args = (tensor, SAVED, False)
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_optional_identity.fn)
        shared = kernel.bind(shared_args) if cached_first else None
        fresh = kernel.bind(fresh_args)
        assert fresh.env.requires_fresh_binding
        config = _choice(fresh, 3)
        if selection == "set_config":
            fresh.set_config(config)
        else:
            with patch.object(fresh.env.backend, "autotune", return_value=config):
                assert fresh.autotune(fresh_args) == config
        saved_launcher = fresh._run
        with patch.object(
            fresh.env.backend, "autotune", side_effect=AssertionError("lost config")
        ):
            rebound = fresh(*shared_args)
        assert rebound is not fresh and not rebound.env.requires_fresh_binding
        assert rebound is kernel.bind(shared_args)
        if shared is not None:
            assert rebound is shared
        assert rebound._config.config["factor"] == 3
        assert rebound._run is not saved_launcher
        assert kernel(*shared_args) is rebound


def test_selection_store_is_bounded_copied_and_does_not_retain_tensors():
    import gc
    import weakref

    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_identity_kernel.fn)
        tensor = torch.empty(17)
        backend = type(kernel.bind((tensor, SAVED)).env.backend)
        with patch.object(
            backend, "autotune", side_effect=lambda bound, args, **kw: _choice(bound)
        ) as tune:
            selected = kernel.autotune((tensor, SAVED))
            selected.config["factor"] = 2
            bound = kernel(tensor, SAVED)
            assert bound._config.config["factor"] == 3
            bound._config.config["factor"] = 2
            assert kernel(tensor, SAVED)._config.config["factor"] == 3
            assert tune.call_count == 1
            tensor_ref = weakref.ref(tensor)
            del bound, tensor
            tune.reset_mock()
            gc.collect()
            assert tensor_ref() is None
            # Distinct sizes are distinct real frontend traces; old entries
            # must be evicted rather than accumulating bounds or nonces.
            for size in range(18, 35):
                kernel(torch.empty(size), SAVED)
            assert len(kernel._fresh_selections) == 16
        assert not kernel._bound_kernels and not kernel._dispatch_cache
        assert kernel._prepared_call is None


@helion.kernel()
def _identity_domain_argument(x, value: hl.constexpr, choices: hl.constexpr):
    factor = hl.register_tunable("factor", EnumFragment(choices))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        offset = 2.0 if value is _saved() else 5.0
        out[tile] = x[tile] * factor + offset
    return out


@onlyBackends(["triton", "cute", "pallas"])
def test_multi_shape_rejects_changed_declared_domain_before_search():
    tensor = torch.empty(17)
    with (
        cpu_render_target(),
        patch(
            "helion.runtime.kernel._canonicalize_multi_shape_device",
            return_value=torch.device("cuda", 0),
        ),
        patch("helion.runtime.kernel._current_device_index", return_value=0),
    ):
        kernel = helion.kernel(_identity_domain_argument.fn)
        first = (tensor, SAVED, (2, 3))
        second = (tensor, SAVED, (2, 4))
        probe = kernel.bind(first)
        with (
            patch.object(
                type(probe.env.backend),
                "autotune",
                side_effect=AssertionError("must not search"),
            ),
            pytest.raises(helion.exc.InvalidAPIUsage, match="structurally compatible"),
        ):
            kernel.autotune_multi([first, second], cache_tag="different-domains")
        assert not kernel._fresh_selections


@pytest.mark.parametrize("mixed", [False, True])
def test_fresh_search_never_reads_warm_start_caches(mixed):
    from helion.autotuner.base_search import BaseSearch
    from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs

    class Search(BaseSearch):
        def autotune(self, *, skip_cache=False):
            raise AssertionError("this test only constructs the search")

    with cpu_render_target():
        kernel = helion.kernel(_optional_identity.fn)
        tensor = torch.empty(17)
        args = (tensor, SAVED, True)
        bound = kernel.bind(args)
        if mixed:
            reusable_args = (tensor, SAVED, False)
            reusable = kernel.bind(reusable_args)
            args = _MultiShapeAutotuneArgs(
                cases=((reusable, reusable_args), (bound, args)),
                aggregation="geomean",
                relative_to=None,
                cache_tag="no-io",
                workload_key=("fresh-only-invocation",),
            )
            bound = reusable
        search = Search(bound, args)
        with (
            patch.object(
                search,
                "_get_current_hardware_and_specialization",
                side_effect=AssertionError("no warm key"),
            ),
            patch(
                "helion.autotuner.local_cache.iter_cache_entries",
                side_effect=AssertionError("no local read"),
            ),
            patch(
                "helion.autotuner.remote_cache._load_remote_backend_if_configured",
                side_effect=AssertionError("no remote read"),
            ),
        ):
            assert search._find_similar_cached_configs(4) == []


@onlyBackends(["triton", "cute"])
def test_forced_fresh_selection_overrides_decorator_config():
    args = (torch.empty(17), SAVED)
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        seed_kernel = helion.kernel(_identity_kernel.fn)
        declared = _choice(seed_kernel.bind(args), 2)
        kernel = helion.kernel(_identity_kernel.fn, config=declared)
        first = kernel(*args)
        assert first._config.config["factor"] == 2
        with patch.object(
            type(first.env.backend), "autotune", return_value=_choice(first, 3)
        ) as tune:
            assert kernel.autotune(args, force=True).config["factor"] == 3
            assert kernel(*args)._config.config["factor"] == 3
            assert kernel(*args)._config.config["factor"] == 3
            assert tune.call_count == 1
            kernel.reset()
            assert kernel(*args)._config.config["factor"] == 2


def test_fresh_selection_respects_the_evaluated_custom_key():
    state = {"key": 0, "calls": 0}

    def custom_key(*args):
        state["calls"] += 1
        return state["key"]

    tensor = torch.empty(17)
    with cpu_render_target(), patch.object(BoundKernel, "compile_config", _compile):
        kernel = helion.kernel(_identity_kernel.fn, key=custom_key)
        probe = kernel.bind((tensor, SAVED))
        assert probe.env.requires_fresh_binding
        with patch.object(
            type(probe.env.backend),
            "autotune",
            side_effect=lambda bound, args, **kw: _choice(bound, 2 + state["key"]),
        ) as tune:
            state["calls"] = 0
            first = kernel(tensor, SAVED)
            state["key"] = 1
            second = kernel(tensor, SAVED)
            state["key"] = 0
            restored = kernel(tensor, SAVED)
        assert first._config["factor"] == restored._config["factor"] == 2
        assert second._config["factor"] == 3
        assert tune.call_count == 2
        assert state["calls"] == 3
