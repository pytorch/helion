from __future__ import annotations

from collections import OrderedDict
import math
import struct
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import sympy
import torch

import helion
from helion._compiler._inductor.template_buffer import _FusionAutotuneAdapter
from helion._compiler.cute.runtime_hooks import register_compiled
from helion._compiler.cute.state import cute_env_state
from helion._testing import matchesBackends
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.external import _ExternalKernelAdapter
from helion.autotuner.external import create_user_config_spec
import helion.language as hl
from helion.runtime.config import _config_key
from helion.runtime.config_cache import ConfigCache
from helion.runtime.kernel import _SEARCHING_CANDIDATES

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def _identity_probe(x: torch.Tensor) -> torch.Tensor:
    selected = hl.register_tunable("selected", EnumFragment((-0.0, 0.0, 1, 1.0, True)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + selected
    return out


@helion.kernel(autotune_effort="none")
def _field_order_probe(x: torch.Tensor) -> torch.Tensor:
    a = hl.register_tunable("a", EnumFragment((2, 3)))
    b = hl.register_tunable("b", EnumFragment((4, 5)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * a + b
    return out


@pytest.fixture
def bound():
    result = cpu_bind(_identity_probe, (torch.empty(32),))

    def source(config, **kwargs):
        # Exercise real normalization, compile-cache and PyCodeCache loading;
        # the CPU callable exposes exactly which selected value was compiled.
        return (
            f"def {result.kernel.name}():\n    return {config.config['selected']!r}\n"
        )

    with (
        patch.object(result, "to_triton_code", side_effect=source) as render,
        patch.object(result.env.backend, "setup_compile_cache_dir"),
        patch.object(result.env.backend, "after_compile"),
    ):
        yield result, render


def config(value):
    return helion.Config(block_sizes=[32], selected=value)


def assert_exact(actual, expected):
    assert type(actual) is type(expected)
    assert actual == expected
    if type(expected) is float:
        assert math.copysign(1, actual) == math.copysign(1, expected)


@pytest.mark.parametrize("first,second", [(-0.0, 0.0), (1, 1.0), (1, True)])
def test_real_compile_cache_distinguishes_publicly_equal_configs(bound, first, second):
    kernel, render = bound
    left, right = config(first), config(second)
    assert left == right and hash(left) == hash(right)
    a = kernel.compile_config(left)
    assert kernel.has_compiled_config(left)
    assert not kernel.has_compiled_config(right)
    b = kernel.compile_config(right)
    assert a is not b
    assert_exact(a(), first)
    assert_exact(b(), second)
    assert kernel.compile_config(left) is a
    assert kernel.compile_config(right) is b
    assert render.call_count == 2
    assert kernel.get_cached_path(left) != kernel.get_cached_path(right)
    with patch.object(
        kernel.env.backend, "compiled_cache_key", side_effect=lambda owner, fn: fn()
    ):
        assert_exact(kernel.backend_cache_key(left), first)
        assert_exact(kernel.backend_cache_key(right), second)


def test_primed_render_and_parent_rerender_keep_exact_config_identity(bound):
    kernel, render = bound
    negative, positive = config(-0.0), config(0.0)
    kernel.prime_rendered_code(
        negative, f"def {kernel.kernel.name}():\n    return -0.0\n"
    )
    token = _SEARCHING_CANDIDATES.set(True)
    try:
        positive_fn = kernel.compile_config(positive)
        negative_fn = kernel.compile_config(negative)
    finally:
        _SEARCHING_CANDIDATES.reset(token)
    assert_exact(positive_fn(), 0.0)
    assert_exact(negative_fn(), -0.0)
    assert render.call_count == 1
    assert kernel.compile_config(positive) is positive_fn
    assert render.call_count == 1
    assert_exact(kernel.compile_config(negative)(), -0.0)
    assert render.call_count == 2


def test_compile_cache_ignores_tunable_keyword_order():
    kernel = cpu_bind(_field_order_probe, (torch.empty(32),))

    def source(selected_config, **kwargs):
        a, b = selected_config["a"], selected_config["b"]
        return f"def {kernel.kernel.name}():\n    return {a!r}, {b!r}\n"

    with (
        patch.object(kernel, "to_triton_code", side_effect=source) as render,
        patch.object(kernel.env.backend, "setup_compile_cache_dir"),
        patch.object(kernel.env.backend, "after_compile"),
    ):
        first = helion.Config(block_sizes=[32], a=2, b=4)
        second = helion.Config(block_sizes=[32], b=4, a=2)
        fn = kernel.compile_config(first)
        assert kernel.has_compiled_config(second)
        assert kernel.compile_config(second) is fn and fn() == (2, 4)
        assert kernel.get_cached_path(second) == kernel.get_cached_path(first)
        assert render.call_count == 1


def test_cached_paths_and_eviction_do_not_alias(bound):
    kernel, _ = bound
    negative, positive = config(-0.0), config(0.0)
    kernel.compile_config(negative)
    assert kernel.get_cached_path(positive) is None
    assert kernel.backend_cache_key(positive) is None
    positive_fn = kernel.compile_config(positive)
    normalized_negative = kernel._normalized_config_copy(negative)
    kernel._compile_cache.pop(normalized_negative)
    kernel._cache_path_map.pop(normalized_negative)
    assert not kernel.has_compiled_config(negative)
    assert kernel.compile_config(positive) is positive_fn
    assert kernel.get_cached_path(negative) is None
    assert kernel.get_cached_path(positive) is not None


@onlyBackends(["cute"])
def test_codegen_and_compilation_hooks_share_exact_metadata_keys(bound):
    kernel, _ = bound
    negative, positive = config(-0.0), config(0.0)
    extents = cute_env_state(kernel.env).hint_sized_reduction_extents
    with (
        patch.object(kernel, "_cache_managed", True),
        patch.object(kernel, "_extend_descriptor_specializations") as extend,
    ):
        try:
            for requested, extent in [(negative, 64), (positive, 128)]:
                kernel.env.backend.before_codegen(kernel, requested)
                extents.add(sympy.Integer(extent))
                kernel.env.backend.after_codegen(
                    kernel, kernel._normalized_config_copy(requested)
                )
            extend.reset_mock()
            for requested, extent in [(positive, 128), (negative, 64)]:
                register_compiled(kernel, requested)
                descriptors = extend.call_args.args[1]
                assert len(descriptors) == 1 and descriptors[0].extent == extent
            assert extend.call_count == 2
        finally:
            extents.clear()


def test_nested_cache_keys_snapshot_plain_containers_and_preserve_dict_order():
    cache = ConfigCache[str]()
    nested = {"items": [(-0.0, 1)], "enabled": True}
    original = helion.Config(selected=nested)
    cache[original] = "original"
    equivalent = helion.Config(selected={"items": [(-0.0, 1)], "enabled": True})
    assert helion.Config(selected={"enabled": True, "items": [(-0.0, 1)]}) not in cache
    assert cache[equivalent] == "original"
    # Mutating the caller's container or an iterated key cannot retag an entry.
    nested["items"] = [(0.0, 1.0)]
    assert original not in cache
    iterated = next(iter(cache))
    iterated.config["selected"]["items"][0] = (0.0, True)
    assert cache[equivalent] == "original"
    for changed in [
        {"items": [(0.0, 1)], "enabled": True},
        {"items": [(-0.0, 1.0)], "enabled": True},
        {"items": [(-0.0, 1)], "enabled": 1},
        {"items": ((-0.0, 1),), "enabled": True},
    ]:
        assert helion.Config(selected=changed) not in cache
    assert cache.pop(equivalent) == "original"
    assert not cache


def test_cache_keys_preserve_nan_bits_and_opaque_leaf_equality():
    class Leaf:
        def __hash__(self):
            return 7

        def __eq__(self, other):
            return type(other) is type(self)

        def __deepcopy__(self, memo):
            raise AssertionError("cache must not copy opaque tunables")

    cache = ConfigCache[str]()
    leaf = Leaf()
    cache[helion.Config(selected=[leaf])] = "leaf"
    assert cache[helion.Config(selected=[Leaf()])] == "leaf"
    assert next(iter(cache)).config["selected"][0] is leaf
    first, second = [
        struct.unpack("!d", bits.to_bytes(8, "big"))[0]
        for bits in (0x7FF8000000000001, 0x7FF8000000000002)
    ]
    cache[config(first)] = "first"
    cache[config(second)] = "second"
    assert cache[config(first)] == "first"
    assert cache[config(second)] == "second"
    cache.clear()
    assert not cache


def test_external_compiler_uses_exact_config_identity():
    compiled = []

    def compile_fn(selected_config):
        value = selected_config["selected"]
        compiled.append(value)
        return lambda: value

    kernel = _ExternalKernelAdapter(
        create_user_config_spec({"selected": EnumFragment((-0.0, 0.0, 1, 1.0, True))}),
        compile_fn,
        (),
        device=torch.device("cpu"),
    )
    for value in (-0.0, 0.0, 1, 1.0, True):
        selected = helion.Config(selected=value)
        fn = kernel.compile_config(selected)
        assert_exact(fn(), value)
        assert kernel.compile_config(selected) is fn
    assert len(compiled) == 5


def test_external_compiler_preserves_ordered_mapping_semantics():
    first = OrderedDict((("left", 1), ("right", 2)))
    second = OrderedDict(reversed(tuple(first.items())))
    assert helion.Config(selected=first) != helion.Config(selected=second)

    def compile_fn(selected_config):
        order = tuple(selected_config["selected"])
        return lambda: order

    kernel = _ExternalKernelAdapter(
        create_user_config_spec({"selected": EnumFragment((first, second))}),
        compile_fn,
        (),
        device=torch.device("cpu"),
    )
    assert kernel.compile_config(helion.Config(selected=first))() == ("left", "right")
    assert kernel.compile_config(helion.Config(selected=second))() == ("right", "left")


def test_ordered_mapping_snapshot_survives_caller_and_iterator_mutation():
    value = OrderedDict((("left", [1]), ("right", [2])))
    cache = ConfigCache[str]()
    cache[config(value)] = "before"
    expected = config(OrderedDict((("left", [1]), ("right", [2]))))
    value.move_to_end("left")
    value["left"].append(3)
    assert config(value) not in cache
    copied = next(iter(cache))["selected"]
    assert type(copied) is OrderedDict
    copied.move_to_end("left")
    copied["right"].append(4)
    assert cache[expected] == "before"


def test_tuple_subclass_keeps_its_equality_contract():
    class IdentityTuple(tuple):
        __slots__ = ()
        __hash__ = tuple.__hash__

        def __eq__(self, other):
            return self is other

    first, second = IdentityTuple((1, 2)), IdentityTuple((1, 2))
    cache = ConfigCache[str]()
    cache[config(first)] = "first"
    assert config(second) not in cache
    cache[config(second)] = "second"
    assert cache[config(first)] == "first"


def test_float_subclass_preserves_signed_zero():
    class Float(float):
        pass

    cache = ConfigCache[str]()
    cache[config(Float(-0.0))] = "negative"
    assert config(Float(0.0)) not in cache


@pytest.mark.parametrize("default,selected", [(-0.0, 0.0), (1, 1.0), (1, True)])
def test_minimize_preserves_selected_scalar_identity(default, selected):
    spec = create_user_config_spec({"selected": EnumFragment((default, selected))})
    for value in (default, selected):
        requested = helion.Config(selected=value)
        minimal = requested.minimize(spec)
        # UserConfigSpec.normalize is a no-op: even the declared default must
        # remain explicit when omission would not reconstruct its exact value.
        assert "selected" in minimal
        assert_exact(minimal["selected"], value)
        assert _config_key(spec.normalized_config(minimal)) == _config_key(
            spec.normalized_config(requested)
        )


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "construction", ["minimal", "defaults", "reordered", "unflattened", "partial"]
)
def test_backend_finalization_uses_the_minimized_winner(bound, construction):
    kernel, _ = bound
    default, winner = config(-0.0), config(0.0)
    if construction in ("defaults", "reordered"):
        default = helion.Config.from_dict(
            kernel.config_spec.default_config().config | default.config
        )
        winner = helion.Config.from_dict(
            kernel.config_spec.default_config().config | winner.config
        )
        if construction == "reordered":
            default = helion.Config.from_dict(dict(reversed(default.config.items())))
            winner = helion.Config.from_dict(dict(reversed(winner.config.items())))
    elif construction == "unflattened":
        generation = kernel.config_spec.create_config_generation()
        default = generation.unflatten(generation.flatten(default))
        winner = generation.unflatten(generation.flatten(winner))
        assert_exact(default["selected"], -0.0)
        assert_exact(winner["selected"], 0.0)
    elif construction == "partial":
        field = "num_warps" if matchesBackends(["triton"]) else "cute_vector_widths"
        value = kernel.config_spec.default_config()[field]
        default.config[field] = value
        winner.config[field] = value
    default_fn = kernel.compile_config(default)
    winner_fn = kernel.compile_config(winner)
    with patch.object(
        kernel.env.backend, "compiled_cache_key", side_effect=lambda bound, fn: fn()
    ):
        assert_exact(
            kernel.backend_cache_key(default.minimize(kernel.config_spec)), -0.0
        )
        assert_exact(kernel.backend_cache_key(winner.minimize(kernel.config_spec)), 0.0)
    default_persist, winner_persist = Mock(), Mock()
    default_jit = SimpleNamespace(
        device_caches={"entry": "default"},
        _helion_cute_compiled_launchers={
            "main": SimpleNamespace(persist_compiled=default_persist)
        },
    )
    winner_jit = SimpleNamespace(
        device_caches={"entry": "winner"},
        _helion_cute_compiled_launchers={
            "main": SimpleNamespace(persist_compiled=winner_persist)
        },
    )
    name = f"_helion_{kernel.kernel.name}"
    with (
        patch.dict(default_fn.__globals__, {name: default_jit}),
        patch.dict(winner_fn.__globals__, {name: winner_jit}),
    ):
        kernel.env.backend.finalize_ephemeral_cache(
            kernel, winner.minimize(kernel.config_spec)
        )
    if matchesBackends(["triton"]):
        assert kernel.has_compiled_config(default)
        assert not kernel.has_compiled_config(winner)
        assert default_jit.device_caches == {"entry": "default"}
        assert winner_jit.device_caches == {}
    else:
        assert kernel.settings.backend == "cute"
        winner_persist.assert_called_once_with()
        default_persist.assert_not_called()


@onlyBackends(["cute"])
def test_unlaunched_minimized_wrapper_does_not_hide_compiled_winner(bound):
    kernel, render = bound

    def source(selected, **kwargs):
        # Real generated wrappers include their config in repro metadata. Keep
        # the full and minimized Python modules distinct while using PyCodeCache.
        return (
            f"# {selected.config!r}\n"
            f"def {kernel.kernel.name}():\n    return {selected['selected']!r}\n"
        )

    render.side_effect = source
    default = config(-0.0)
    full = helion.Config.from_dict(
        kernel.config_spec.default_config().config | config(0.0).config
    )
    default_fn = kernel.compile_config(default)
    full_fn = kernel.compile_config(full)
    assert kernel.compile_config(full.minimize(kernel.config_spec)) is full_fn
    # Older minimization omitted these explicit search defaults. Exercise that
    # legacy sparse form deliberately: current minimization preserves the exact
    # normalized compile key and therefore correctly reuses full_fn above.
    minimal = config(0.0)
    minimal_fn = kernel.compile_config(minimal)
    assert full_fn is not minimal_fn
    with patch.object(
        kernel.env.backend,
        "compiled_cache_key",
        side_effect=lambda bound, fn: "winner-key" if fn is full_fn else None,
    ):
        assert kernel.backend_cache_key(minimal) == "winner-key"

    default_persist, full_persist = Mock(), Mock()
    default_jit = SimpleNamespace(
        device_caches={"default": "binary"},
        _helion_cute_compiled_launchers={
            "main": SimpleNamespace(persist_compiled=default_persist)
        },
    )
    full_jit = SimpleNamespace(
        device_caches={"winner": "binary"},
        _helion_cute_compiled_launchers={
            "main": SimpleNamespace(persist_compiled=full_persist)
        },
    )
    unlaunched_jit = SimpleNamespace(
        device_caches={}, _helion_cute_compiled_launchers={}
    )
    name = f"_helion_{kernel.kernel.name}"
    with (
        patch.dict(default_fn.__globals__, {name: default_jit}),
        patch.dict(full_fn.__globals__, {name: full_jit}),
        patch.dict(minimal_fn.__globals__, {name: unlaunched_jit}),
    ):
        kernel.env.backend.finalize_ephemeral_cache(kernel, minimal)
    assert default_jit.device_caches == {"default": "binary"}
    default_persist.assert_not_called()
    full_persist.assert_called_once_with()
    assert kernel.has_compiled_config(full)
    assert kernel.has_compiled_config(minimal)


def test_fusion_compiled_paths_keep_exact_config_identity(bound):
    kernel, _ = bound

    def factory(selected_config):
        value = selected_config["selected"]
        return lambda: value, f"{value!r}.py"

    adapter = _FusionAutotuneAdapter(kernel, factory, "identity-test")
    negative, positive = config(-0.0), config(0.0)
    adapter.bench_compile_config(negative)
    adapter.bench_compile_config(positive)
    assert adapter.get_cached_path(negative) == "-0.0.py"
    assert adapter.get_cached_path(positive) == "0.0.py"


def test_unhashable_container_subclass_is_conservatively_keyed_without_copy_hooks():
    class OpaqueList(list):
        def __deepcopy__(self, memo):
            raise AssertionError("opaque containers must not run copy hooks")

    value = OpaqueList([1])
    cache = ConfigCache[str]()
    cache[config(value)] = "before"
    assert cache[config(value)] == "before"
    assert config(OpaqueList([1])) not in cache
    # Contents mutations cause a cache miss, while traversing/removing existing
    # entries remains well-defined. Arbitrary custom state is caller-owned.
    value[0] = 2
    assert config(value) not in cache
    assert list(cache.values()) == ["before"]
    ((stored, item),) = cache.items()
    assert stored["selected"] is value and item == "before"
    assert "before" in repr(cache)
    assert cache.popitem()[1] == "before"
    assert not cache
