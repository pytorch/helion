from __future__ import annotations

import ast
from contextlib import contextmanager
from types import FunctionType
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler._inductor.template_buffer import _FusionAutotuneAdapter
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner import EnumFragment
import helion.language as hl
from helion.runtime.config import _config_key

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def _selected_scalar(x, choices: hl.constexpr):
    value = hl.register_tunable("value", EnumFragment(choices))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + value
    return out


@pytest.mark.parametrize(
    "default,selected", [(0.0, -0.0), (-0.0, 0.0), (2, 2.0), (True, 1)]
)
def test_minimize_preserves_scalar_type_and_sign(default, selected):
    with cpu_render_target():
        bound = cpu_bind(_selected_scalar, (torch.empty(17), (default, selected)))
        with bound.env:
            full = bound.config_spec.default_config()
        full.config["value"] = selected
        minimized = full.minimize(bound.config_spec)
        assert "value" in minimized.config
        value = minimized.config["value"]
        assert type(value) is type(selected)
        if type(value) is float:
            assert value.hex() == selected.hex()
        restored = helion.Config.from_dict(
            bound.config_spec.default_config().config | minimized.config
        )
        assert bound.to_code(restored) == bound.to_code(full)
        # A true default still disappears; public numeric Config equality stays.
        assert (
            "value"
            not in bound.config_spec.default_config().minimize(bound.config_spec).config
        )
        assert helion.Config(value=default) == helion.Config(value=selected)


def test_minimize_preserves_nested_scalar_metadata():
    with cpu_render_target():
        bound = cpu_bind(_selected_scalar, (torch.empty(17), (0.0,)))
        # Config supports structured values independently of language scalar
        # registration. Exercise its actual default-driven minimization API.
        bound.config_spec.user_defined_tunables["nested"] = EnumFragment(
            ({"parts": [0.0, True]},)
        )
        selected = {"parts": [-0.0, 1]}
        config = helion.Config(block_sizes=[64], nested=selected)
        minimized = config.minimize(bound.config_spec)
        assert minimized.config["nested"]["parts"][0].hex() == "-0x0.0p+0"
        assert type(minimized.config["nested"]["parts"][1]) is int


@pytest.mark.parametrize("reverse", [False, True])
def test_fusion_benchmark_paths_keep_typed_config_identity(reverse):
    with cpu_render_target():
        bound = cpu_bind(
            _selected_scalar, (torch.empty(17), (True, 1, False, 0, -0.0, 0.0))
        )
        values = (True, 1, False, 0, -0.0, 0.0)
        if reverse:
            values = tuple(reversed(values))
        calls = []

        def factory(config):
            ast.parse(bound.to_code(config))
            path = f"path-{len(calls)}"
            calls.append((config, path))
            return lambda: path, path

        adapter = _FusionAutotuneAdapter(bound, factory, "fusion-context")
        configs = [helion.Config(block_sizes=[64], value=value) for value in values]
        for config in configs:
            adapter.bench_compile_config(config)
        for config, path in calls:
            assert adapter.get_cached_path(config) == path
            assert (
                adapter.get_cached_path(helion.Config.from_dict(config.config)) == path
            )
        assert adapter.get_cached_path() is None
        assert adapter.extra_cache_key() == "fusion-context"


@contextmanager
def _compiled_host(bound):
    compiled = []

    def load(source, **kwargs):
        ast.parse(source)
        module = ModuleType("scalar_lifecycle_host")
        module.__file__ = f"scalar_lifecycle_{len(compiled)}.py"
        jit = SimpleNamespace(device_caches={0: {"compiled": object()}})
        fn = FunctionType(
            (lambda: None).__code__, {f"_helion_{bound.kernel.name}": jit}
        )
        setattr(module, bound.kernel.name, fn)
        compiled.append((fn, jit))
        return module

    with (
        patch("torch._inductor.codecache.PyCodeCache.load", side_effect=load),
        patch.object(bound.env.backend, "setup_compile_cache_dir"),
        patch.object(bound.env.backend, "annotate_compiled_module"),
        patch.object(bound.env.backend, "after_compile"),
    ):
        yield compiled


@pytest.mark.parametrize("reverse", [False, True])
def test_scalar_selections_use_distinct_compiled_entries(reverse):
    with cpu_render_target():
        values = (True, 1, False, 0, -0.0, 0.0)
        if reverse:
            values = tuple(reversed(values))
        bound = cpu_bind(_selected_scalar, (torch.empty(17), values))
        configs = [helion.Config(block_sizes=[64], value=value) for value in values]
        with _compiled_host(bound) as compiled:
            functions = [
                bound.compile_config(config, allow_print=False) for config in configs
            ]
            assert len({id(fn) for fn in functions}) == len(values)
            for config, fn in zip(configs, functions, strict=True):
                assert bound.compile_config(config, allow_print=False) is fn
            assert len(compiled) == len(values)


def test_nested_compilation_keys_preserve_scalar_paths():
    configs = [
        helion.Config(meta={"parts": [value], "other": 0})
        for value in (True, 1, False, 0, -0.0, 0.0)
    ]
    assert len({_config_key(config) for config in configs}) == len(configs)
    assert _config_key(
        helion.Config(meta={"parts": [True], "other": 0})
    ) != _config_key(helion.Config(meta={"parts": [1], "other": False}))


@helion.kernel(autotune_effort="none")
def _ordinary_pointwise(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + 2
    return out


@helion.kernel(autotune_effort="none")
def _ordinary_reduction(x):
    out = torch.empty(x.size(0), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = torch.sum(x[row, :], dim=1)
    return out


@pytest.mark.parametrize("reduction", [False, True])
def test_minimize_round_trip_keeps_selected_compiled_entry(reduction, tmp_path):
    with cpu_render_target():
        kernel = _ordinary_reduction if reduction else _ordinary_pointwise
        args = (torch.empty(4, 17),) if reduction else (torch.empty(17),)
        bound = cpu_bind(kernel, args)
        full = bound.config_spec.default_config()
        original = full.to_json()
        bare = helion.Config(block_sizes=full.block_sizes)
        normalized_bare = bound._normalized_config_copy(bare)
        minimal = full.minimize(bound.config_spec)
        path = tmp_path / "selected.json"
        minimal.save(path)
        restored = helion.Config.load(path)
        assert _config_key(bound._normalized_config_copy(restored)) == _config_key(
            bound._normalized_config_copy(full)
        )
        # Ordinary omitted-config interpretation has not changed.
        assert bound._normalized_config_copy(bare) == normalized_bare
        with _compiled_host(bound):
            selected = bound.compile_config(full, allow_print=False)
            omission = bound.compile_config(bare, allow_print=False)
            bound.set_config(restored)
            assert bound._run is selected
            if _config_key(normalized_bare) != _config_key(
                bound._normalized_config_copy(full)
            ):
                assert omission is not selected
            if bound.settings.backend == "triton":
                bound.env.backend.finalize_ephemeral_cache(bound, restored)
                assert not selected.__globals__[
                    f"_helion_{bound.kernel.name}"
                ].device_caches
                assert bound._normalized_config_copy(full) not in bound._compile_cache
                if omission is not selected:
                    assert bound._compile_cache[normalized_bare] is omission
                    assert omission.__globals__[
                        f"_helion_{bound.kernel.name}"
                    ].device_caches
        assert full.to_json() == original


@onlyBackends(["triton"])
def test_minimize_keeps_config_when_omission_invalidates_dependent_fields():
    with cpu_render_target():
        bound = cpu_bind(_ordinary_pointwise, (torch.empty(17),))
        spec = bound.config_spec
        spec.compiler_default_config = helion.Config(
            pid_type="persistent_blocked", num_sm_multiplier=2
        )
        full = spec.default_config()
        full.config["num_sm_multiplier"] = 3
        expected = spec.normalized_config(full)
        # Removing the promoted default pid_type leaves an explicitly selected
        # multiplier on a flat grid, which normalization must still reject.
        with pytest.raises(helion.exc.InvalidConfig, match="persistent"):
            spec.normalized_config(
                helion.Config(block_sizes=full.block_sizes, num_sm_multiplier=3)
            )
        minimal = full.minimize(spec)
        assert _config_key(spec.normalized_config(minimal)) == _config_key(expected)
        assert minimal.config == full.config


def test_minimize_does_not_add_custom_default_calls():
    class CountingEnum(EnumFragment):
        def __init__(self):
            super().__init__((2, 4))
            self.calls = 0

        def default(self):
            self.calls += 1
            return super().default()

    with cpu_render_target():
        bound = cpu_bind(_ordinary_pointwise, (torch.empty(17),))
        fragment = CountingEnum()
        bound.config_spec.user_defined_tunables["counted"] = fragment
        full = bound.config_spec.default_config()
        original = full.to_json()
        before = fragment.calls
        minimal = full.minimize(bound.config_spec)
        # Keep the established single default_config evaluation, but never
        # invoke this custom default again for candidate normalization.
        assert fragment.calls == before + 1
        # Independent schedule defaults can disappear without omitting this
        # custom selection or changing the effective compiled configuration.
        assert type(minimal.config["counted"]) is int
        assert minimal.config["counted"] == full.config["counted"] == 2
        if full.config.keys() - {"block_sizes", "counted"}:
            assert len(minimal.config) < len(full.config)
        else:
            # No independent schedule defaults exist to remove on this backend.
            assert minimal.config == full.config
        assert _config_key(bound.config_spec.normalized_config(minimal)) == _config_key(
            bound.config_spec.normalized_config(full)
        )
        assert bound.to_code(minimal) == bound.to_code(full)
        assert full.to_json() == original
