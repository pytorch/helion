"""Out-of-tree backends add autotuner config keys through their search surface."""

from __future__ import annotations

import pytest
import torch

import helion
from helion import backend_api
from helion import exc
from helion._compiler.backend_registry import _BUILTIN_BACKENDS
from helion._compiler.backend_registry import _check_surface_keys
from helion._compiler.backend_registry import list_backends
from helion._compiler.backend_registry import unregister_compiler_backend
from helion._compiler.cute.backend import CuteBackend
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.autotuner import EnumFragment
from helion.autotuner.config_spec import valid_keys
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_KNOB = "_tk_knob"


class _KnobSurface(backend_api.BackendSearchSurface):
    config_keys = frozenset({_KNOB})


_KNOB_SURFACE = _KnobSurface()


class _KnobBackend(backend_api.TritonBackend):
    @property
    def name(self) -> str:
        return "_test_knob"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def config_spec_extension(self) -> backend_api.BackendSearchSurface:
        return _KNOB_SURFACE

    def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
        return {**super().tunable_fragments(), _KNOB: EnumFragment(choices=(4, 8))}


def _surface_backend(
    name: str, keys: frozenset[str], debug: frozenset[str] = frozenset()
) -> type[backend_api.Backend]:
    class _Surface(backend_api.BackendSearchSurface):
        config_keys = keys
        debug_keys = debug

    surface = _Surface()

    class _Backend(backend_api.TritonBackend):
        @property
        def name(self) -> str:
            return name

        @property
        def codegen_name(self) -> str:
            return "triton"

        def config_spec_extension(self) -> backend_api.BackendSearchSurface:
            return surface

    return _Backend


def _add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + 1
    return out


def _spec(backend: str) -> backend_api.ConfigSpec:
    kernel = helion.kernel(backend=backend)(_add)
    return kernel.bind((torch.randn(64, 64),)).config_spec


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestSurfaceConfigKeys(TestCase):
    def setUp(self) -> None:
        super().setUp()
        backend_api.register_compiler_backend(_KnobBackend)
        self.addCleanup(unregister_compiler_backend, "_test_knob")

    def test_key_is_searched_and_defaulted(self) -> None:
        spec = _spec("_test_knob")
        self.assertIn(_KNOB, spec._flat_fields())
        self.assertEqual(spec.default_config().config[_KNOB], 4)

    def test_key_is_searched_with_a_surface_field_table(self) -> None:
        # A surface supplying its whole field table (as CuTe's does) gets the
        # backend's fragments; they win, matching normalize's defaults.
        class _TableSurface(_KnobSurface):
            config_keys = frozenset({_KNOB, "_tk_table"})

            def fields(
                self,
                config_spec: backend_api.ConfigSpec,
                *,
                flash_pipeline_family: str | None = None,
            ) -> dict[str, object] | None:
                return {
                    "block_sizes": config_spec.block_sizes,
                    "_tk_table": EnumFragment(choices=(1,)),
                }

        table_surface = _TableSurface()

        class _TableBackend(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_knob_table"

            def config_spec_extension(self) -> backend_api.BackendSearchSurface:
                return table_surface

            def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
                return {
                    **super().tunable_fragments(),
                    "_tk_table": EnumFragment(choices=(2,)),
                }

        backend_api.register_compiler_backend(_TableBackend)
        self.addCleanup(unregister_compiler_backend, "_test_knob_table")
        spec = _spec("_test_knob_table")
        fields = spec._flat_fields()
        self.assertEqual(set(fields), {"block_sizes", "_tk_table", _KNOB})
        self.assertEqual(fields["_tk_table"].default(), 2)
        config = helion.Config(block_sizes=spec.default_config().config["block_sizes"])
        spec.normalize(config)
        self.assertEqual(config.config["_tk_table"], 2)

    def test_key_table_follows_registration(self) -> None:
        self.assertIn(_KNOB, valid_keys())
        unregister_compiler_backend("_test_knob")
        self.assertNotIn(_KNOB, valid_keys())  # cached tables are rebuilt
        backend_api.register_compiler_backend(_KnobBackend)
        self.assertIn(_KNOB, valid_keys())

    def test_surface_fingerprint_versions_the_cache_fingerprint(self) -> None:
        # LocalAutotuneCache (local_cache.py) builds the key's config_spec_hash
        # from this fingerprint; it needs GPU hardware info, so test it here.
        class _VersionSurface(backend_api.BackendSearchSurface):
            version: str | None = "1.0"

            def fingerprint_extra(
                self, config_spec: backend_api.ConfigSpec
            ) -> object | None:
                return None if self.version is None else f"pkg=={self.version}"

        surface = _VersionSurface()

        class _VersionedBackend(backend_api.TritonBackend):
            @property
            def name(self) -> str:
                return "_test_versioned"

            @property
            def codegen_name(self) -> str:
                return "triton"

            def config_spec_extension(self) -> backend_api.BackendSearchSurface:
                return surface

        backend_api.register_compiler_backend(_VersionedBackend)
        self.addCleanup(unregister_compiler_backend, "_test_versioned")

        def fingerprint(backend: str) -> str:
            return _spec(backend).cache_fingerprint_hash()

        v1 = fingerprint("_test_versioned")
        self.assertEqual(fingerprint("_test_versioned"), v1)
        surface.version = "2.0"
        v2 = fingerprint("_test_versioned")
        self.assertNotEqual(v1, v2)
        # Returning None adds nothing: the fingerprint matches the same backend
        # with the default surface. (It need not be the structural hash: compiler
        # seeds, e.g. under HELION_BACKEND=pallas, also change it.)
        surface.version = None
        backend_api.register_compiler_backend(
            _surface_backend("_test_default_surface", frozenset())
        )
        self.addCleanup(unregister_compiler_backend, "_test_default_surface")
        self.assertEqual(
            fingerprint("_test_versioned"), fingerprint("_test_default_surface")
        )

    def test_plugin_keys_leave_built_in_tables_unchanged(self) -> None:
        def table(spec: backend_api.ConfigSpec) -> tuple[list[str], str]:
            return list(spec._flat_fields()), spec.cache_fingerprint_hash()

        with_plugin = table(_spec("triton"))
        unregister_compiler_backend("_test_knob")
        self.assertEqual(table(_spec("triton")), with_plugin)

    def test_other_backends_reject_key(self) -> None:
        self.assertTrue(_spec("_test_knob").supports_config_key(_KNOB))
        spec = _spec("triton")
        self.assertFalse(spec.supports_config_key(_KNOB))
        config = spec.default_config()
        config.config[_KNOB] = 4
        with self.assertRaises(exc.InvalidConfig):
            spec.normalize(config)

    def test_generic_keys_are_rejected(self) -> None:
        self.addCleanup(unregister_compiler_backend, "_test_generic")
        for keys, debug in (
            (frozenset({"num_warps"}), frozenset()),
            (frozenset(), frozenset({"num_warps"})),
        ):
            with self.subTest(debug=bool(debug)):
                with self.assertRaisesRegex(
                    ValueError, "generic config keys.*num_warps"
                ):
                    backend_api.register_compiler_backend(
                        _surface_backend("_test_generic", keys, debug)
                    )
                self.assertNotIn("_test_generic", list_backends())
        self.assertTrue(_spec("triton").supports_config_key("num_warps"))

    def test_fragment_needs_a_declared_config_key(self) -> None:
        # Only the surface's own config_keys may get fragments: not an
        # undeclared key, a debug key, or a gated generic key Helion builds.
        none: frozenset[str] = frozenset()
        for key, keys, debug in (
            ("_tk_undeclared", none, none),
            ("_tk_debug", none, frozenset({"_tk_debug"})),
            ("xcd_remap", frozenset({"xcd_remap"}), none),
        ):
            surface_backend = _surface_backend(f"_test{key}", keys, debug)

            class _FragmentBackend(surface_backend):
                fragment_key = key

                def tunable_fragments(
                    self,
                ) -> dict[str, backend_api.ConfigSpecFragment]:
                    fragment = EnumFragment((1, 2))
                    return {**super().tunable_fragments(), self.fragment_key: fragment}

            backend_api.register_compiler_backend(_FragmentBackend)
            self.addCleanup(unregister_compiler_backend, f"_test{key}")
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, f"unknown tunables.*{key}"),
            ):
                _spec(f"_test{key}")

    def test_rejected_surface_key_is_explained(self) -> None:
        # A fixed-list supports_config_key that does not defer to super()
        # rejects the backend's own key.
        class _AllowListBackend(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_allow_list"

            def supports_config_key(self, key: str) -> bool:
                return key in {"block_sizes", "num_warps"}

        backend_api.register_compiler_backend(_AllowListBackend)
        self.addCleanup(unregister_compiler_backend, "_test_allow_list")
        with self.assertRaisesRegex(RuntimeError, f"{_KNOB}.*defer to super"):
            _spec("_test_allow_list")

    def test_built_in_surfaces_declare_no_generic_keys(self) -> None:
        # Built-ins skip the registration check (they register while
        # config_spec is importing), so check them here.
        for cls in _BUILTIN_BACKENDS:
            with self.subTest(backend=cls.__name__):
                _check_surface_keys(cls())  # raises ValueError on a clash

    def test_backend_specific_built_in_keys_are_allowed(self) -> None:
        # Already backend-specific, so declaring one changes no other backend.
        cls = _surface_backend("_test_gated", frozenset({"xcd_remap", "waves_per_eu"}))
        backend_api.register_compiler_backend(cls)
        self.addCleanup(unregister_compiler_backend, "_test_gated")
        self.assertIn("_test_gated", list_backends())

    def test_backends_may_share_keys(self) -> None:
        # Unrelated backends and subclasses may declare the same keys, e.g.
        # CuteBackend subclasses repeating CuTe's (incl. cross_loop_pipeline).
        shared = _surface_backend("_test_shared_a", frozenset({"_tk_shared"}))

        class _VariantB(shared):
            @property
            def name(self) -> str:
                return "_test_shared_b"

        class _CuteChild(CuteBackend):
            @property
            def name(self) -> str:
                return "_test_cute_child"

        for cls in (shared, _VariantB, _CuteChild):
            backend_api.register_compiler_backend(cls)
            self.addCleanup(unregister_compiler_backend, cls().name)
        for name in ("_test_shared_a", "_test_shared_b"):
            self.assertTrue(_spec(name).supports_config_key("_tk_shared"))
