from __future__ import annotations

import contextlib
import importlib.metadata
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest import mock

import pytest
import torch

from helion import exc
from helion._compiler import backend_registry
from helion._compiler.aten_lowering import AtenLowering
from helion._compiler.backend import Backend
from helion._compiler.backend import BackendState
from helion._compiler.backend_registry import _REGISTRY
from helion._compiler.backend_registry import all_reserved_launch_param_names
from helion._compiler.backend_registry import get_backend_class
from helion._compiler.backend_registry import install_hint
from helion._compiler.backend_registry import list_backends
from helion._compiler.backend_registry import register_compiler_backend
from helion._compiler.backend_registry import registered_backends
from helion._compiler.backend_registry import registry_generation
from helion._compiler.backend_registry import repair_backend_codegen
from helion._compiler.backend_registry import unregister_compiler_backend
from helion._compiler.compile_environment import CompileEnvironment
import helion.language as hl
from helion.language._decorators import APIFunc
from helion.language._decorators import BackendDispatchTable
from helion.language._decorators import api
from helion.language._decorators import codegen
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.cpu_only


class TestBackendRegistry(unittest.TestCase):
    def test_list_backends_contains_all_builtins(self) -> None:
        names = list_backends()
        for expected in ("triton", "pallas", "cute", "tileir", "metal"):
            self.assertIn(expected, names)

    @contextlib.contextmanager
    def _moved_out_backend(
        self, *, installed: bool, cached: dict[str, object] | None = None
    ) -> Iterator[None]:
        # Entry points as they are now; ``cached`` is the discovery result.
        eps = []
        if installed:
            eps.append(
                importlib.metadata.EntryPoint(
                    name="flydsl",
                    value="helion_flydsl.backend:FlyDSLBackend",
                    group=backend_registry.PLUGIN_ENTRY_POINT_GROUP,
                )
            )
        with (
            mock.patch.object(backend_registry, "_PLUGIN_ENTRY_POINTS", cached),
            mock.patch.object(importlib.metadata, "entry_points", return_value=eps),
            mock.patch.dict(_REGISTRY),
            mock.patch.dict(os.environ),
        ):
            _REGISTRY.pop("flydsl", None)  # in case the plugin is installed
            os.environ.pop("HELION_DISABLE_BACKEND_PLUGINS", None)
            os.environ.pop("HELION_BACKEND", None)
            yield

    def test_moved_out_backend_points_to_its_package(self) -> None:
        # FlyDSL now ships as the helion-flydsl plugin; without it installed,
        # both ways of selecting it say how to get it.
        with self._moved_out_backend(installed=False, cached={}):
            with self.assertRaisesRegex(ValueError, "pip install helion-flydsl"):
                get_backend_class("flydsl")
            with (
                mock.patch.dict(os.environ, {"HELION_BACKEND": "flydsl"}),
                self.assertRaisesRegex(ValueError, "pip install helion-flydsl"),
            ):
                Settings()
            # Not installed: unsetting the variable would not help.
            with (
                mock.patch.dict(os.environ, {"HELION_DISABLE_BACKEND_PLUGINS": "1"}),
                self.assertRaisesRegex(ValueError, "pip install helion-flydsl"),
            ):
                get_backend_class("flydsl")

    def test_moved_out_backend_hint_when_plugins_disabled(self) -> None:
        with (
            self._moved_out_backend(installed=True),
            mock.patch.dict(os.environ, {"HELION_DISABLE_BACKEND_PLUGINS": "1"}),
        ):
            with self.assertRaisesRegex(ValueError, "HELION_DISABLE_BACKEND_PLUGINS"):
                get_backend_class("flydsl")
            self.assertNotIn("pip install", install_hint("flydsl"))

    def test_moved_out_backend_hint_after_late_enable(self) -> None:
        # Discovered while disabled, then re-enabled: only a restart helps.
        with self._moved_out_backend(installed=True, cached={}):
            with self.assertRaisesRegex(ValueError, "restart the process"):
                get_backend_class("flydsl")
            with (
                mock.patch.dict(os.environ, {"HELION_BACKEND": "flydsl"}),
                self.assertRaisesRegex(ValueError, "restart the process"),
            ):
                Settings()

    def test_install_hint_ignores_invalid_disable_value(self) -> None:
        for installed in (False, True):
            with (
                self.subTest(installed=installed),
                self._moved_out_backend(installed=installed, cached={}),
                mock.patch.dict(
                    os.environ,
                    {
                        "HELION_DISABLE_BACKEND_PLUGINS": "maybe",
                        "HELION_BACKEND": "flydsl",
                    },
                ),
            ):
                self.assertNotIn("is set", install_hint("flydsl"))
                with self.assertRaisesRegex(
                    ValueError, "HELION_BACKEND must be one of"
                ) as ctx:
                    Settings()
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)

    def test_get_backend_class_and_instantiate_all(self) -> None:
        for name in registered_backends():
            cls = get_backend_class(name)
            instance = cls()
            self.assertIsInstance(instance, Backend)
            self.assertEqual(instance.name, name)

    def test_get_backend_class_raises_for_unknown(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown backend: 'nonexistent'"):
            get_backend_class("nonexistent")

    def test_default_fake_subscript_shape(self) -> None:
        backend = get_backend_class("triton")()
        tensor = torch.empty(4, 8)

        self.assertEqual(
            backend.fake_subscript_shape(tensor, [slice(None), None, slice(None)]),
            [4, 1, 8],
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "narrowing kernel-tensor subscripts"
        ):
            backend.fake_subscript_shape(tensor, [0, slice(None)])
        with self.assertRaises(exc.InvalidIndexingType):
            backend.fake_subscript_shape(tensor, [object(), slice(None)])
        with self.assertRaises(exc.InvalidIndexingType):
            backend.fake_subscript_shape(tensor, [slice(0, 4, 0), slice(None)])

    def test_register_custom_backend(self) -> None:
        class _TestBackend(Backend):
            @property
            def name(self) -> str:
                return "_test_custom"

            def dtype_str(self, dtype: object) -> str:
                return ""

            def acc_type(self, dtype: object) -> str:
                return ""

            @property
            def default_launcher_name(self) -> str:
                return ""

            @property
            def constexpr_type(self) -> str:
                return ""

            @property
            def function_decorator(self) -> str:
                return ""

            @property
            def library_imports(self) -> dict[str, str]:
                return {}

        register_compiler_backend(_TestBackend)
        try:
            self.assertIn("_test_custom", list_backends())
            self.assertIs(get_backend_class("_test_custom"), _TestBackend)
            # custom backends default to experimental
            self.assertIs(_TestBackend().state, BackendState.EXPERIMENTAL)
        finally:
            unregister_compiler_backend("_test_custom")

    def test_late_registration_refreshes_keys_and_heuristics(self) -> None:
        """A backend registered after import contributes its config keys and
        heuristics: the derived tables key their caches on the registry."""
        from helion._compiler.autotuner_heuristics import get_heuristics
        from helion._compiler.autotuner_heuristics.registry import AutotunerHeuristic
        from helion.autotuner import config_spec
        from helion.autotuner.search_surface import BackendSearchSurface

        class _LateSurface(BackendSearchSurface):
            config_keys = frozenset({"_test_late_key"})
            debug_keys = frozenset({"_test_late_debug_key"})

        class _LateHeuristic(AutotunerHeuristic):
            name = "_test_late_heuristic"
            backend = "_test_late"

        class _LateBackend(Backend):
            @property
            def name(self) -> str:
                return "_test_late"

            def dtype_str(self, dtype: object) -> str:
                return ""

            def acc_type(self, dtype: object) -> str:
                return ""

            @property
            def default_launcher_name(self) -> str:
                return ""

            @property
            def constexpr_type(self) -> str:
                return ""

            @property
            def function_decorator(self) -> str:
                return ""

            @property
            def library_imports(self) -> dict[str, str]:
                return {}

            def config_spec_extension(self) -> BackendSearchSurface:
                return _LateSurface()

            def autotuner_heuristics(self) -> tuple[type[AutotunerHeuristic], ...]:
                return (_LateHeuristic,)

        from helion.language.tunable_ops import _tunable_name_conflicts

        # The tables are populated before the registration.
        self.assertNotIn("_test_late_key", config_spec.valid_keys())
        self.assertEqual(get_heuristics("_test_late"), ())
        self.assertFalse(_tunable_name_conflicts("_test_late_key"))
        generation = registry_generation()
        register_compiler_backend(_LateBackend)
        try:
            self.assertGreater(registry_generation(), generation)
            self.assertIn("_test_late_key", config_spec.valid_keys())
            self.assertIn("_test_late_key", config_spec.backend_specific_keys())
            self.assertIn("_test_late_debug_key", config_spec.debug_keys())
            self.assertIn("_test_late_key", config_spec.VALID_KEYS)
            # ``hl.register_tunable`` refuses the late backend's key through
            # the same live table.
            self.assertTrue(_tunable_name_conflicts("_test_late_key"))
            self.assertEqual(get_heuristics("_test_late"), (_LateHeuristic,))
            self.assertTrue(_LateBackend().supports_config_key("_test_late_key"))
            self.assertFalse(
                get_backend_class("triton")().supports_config_key("_test_late_key")
            )
        finally:
            unregister_compiler_backend("_test_late")
        self.assertNotIn("_test_late_key", config_spec.valid_keys())
        self.assertNotIn("_test_late_debug_key", config_spec.debug_keys())
        self.assertEqual(get_heuristics("_test_late"), ())
        self.assertFalse(_tunable_name_conflicts("_test_late_key"))

    def test_reregister_same_class_is_noop(self) -> None:
        repaired = frozenset({"triton"})
        with mock.patch.object(backend_registry, "_REPAIRED_CODEGEN_NAMES", repaired):
            register_compiler_backend(get_backend_class("triton"))
            self.assertIs(backend_registry._REPAIRED_CODEGEN_NAMES, repaired)

    def test_all_reserved_launch_param_names_is_union(self) -> None:
        result = all_reserved_launch_param_names()
        self.assertIsInstance(result, frozenset)
        # must contain at least the triton names
        for expected in ("grid", "warmup", "num_warps", "num_stages"):
            self.assertIn(expected, result)
        # must be the union of all backends
        for cls in _REGISTRY.values():
            self.assertTrue(cls.reserved_launch_param_names().issubset(result))


class TestBackendCodegenRepair(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(
            backend_registry, "_REPAIRED_CODEGEN_NAMES", frozenset()
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_repair_runs_once_per_codegen_backend(self) -> None:
        with mock.patch.object(
            backend_registry, "_reload_backend_codegen"
        ) as reload_mock:
            repair_backend_codegen("triton")
            repair_backend_codegen("triton")

        reload_mock.assert_called_once_with("triton")

    def test_failed_repair_can_be_retried(self) -> None:
        with mock.patch.object(
            backend_registry,
            "_reload_backend_codegen",
            side_effect=[RuntimeError("broken import"), None],
        ) as reload_mock:
            with self.assertRaisesRegex(RuntimeError, "broken import"):
                repair_backend_codegen("triton")
            repair_backend_codegen("triton")
            repair_backend_codegen("triton")

        self.assertEqual(reload_mock.call_count, 2)

    def test_controlled_repair_allows_reregistration(self) -> None:
        @api()
        def op() -> None:
            return None

        def original(state: object) -> str:
            return "original"

        def replacement(state: object) -> str:
            return "replacement"

        codegen(op, "triton")(original)

        def reload_backend(codegen_name: str) -> None:
            self.assertEqual(codegen_name, "triton")
            codegen(op, "triton")(replacement)

        with mock.patch.object(
            backend_registry,
            "_reload_backend_codegen",
            side_effect=reload_backend,
        ):
            repair_backend_codegen("triton")

        self.assertIs(
            dict.__getitem__(cast("APIFunc", op)._codegen, "triton"), replacement
        )

    def test_repair_rejects_second_registration_for_same_slot(self) -> None:
        @api()
        def op() -> None:
            return None

        def first(state: object) -> str:
            return "first"

        def second(state: object) -> str:
            return "second"

        def reload_backend(codegen_name: str) -> None:
            codegen(op, codegen_name)(first)
            codegen(op, codegen_name)(second)

        with (
            mock.patch.object(
                backend_registry,
                "_reload_backend_codegen",
                side_effect=reload_backend,
            ),
            self.assertRaisesRegex(
                AssertionError, "codegen already registered for backend 'triton'"
            ),
        ):
            repair_backend_codegen("triton")

    def test_real_reload_restores_missing_handler(self) -> None:
        codegen_impls = cast("APIFunc", hl.load)._codegen
        original = dict.__getitem__(codegen_impls, "triton")
        del codegen_impls["triton"]
        self.addCleanup(codegen_impls.__setitem__, "triton", original)

        restored = codegen_impls["triton"]

        self.assertEqual(restored.__module__, "helion._compiler.triton.memory_ops")
        self.assertIsNot(restored, original)

    def test_resident_plan_metadata_survives_codegen_reload(self) -> None:
        from helion._compiler.pallas import view_ops

        plan = view_ops._ResidentPlan(False, (), None)
        node = mock.Mock(meta={view_ops.RESIDENT_PLAN_META: plan})

        class ReloadedResidentPlan:
            pass

        with mock.patch.object(view_ops, "_ResidentPlan", ReloadedResidentPlan):
            self.assertIs(view_ops._resident_plan(node), plan)


class TestAtenCodegenRepair(unittest.TestCase):
    def test_exact_backend_passes_repair_barrier(self) -> None:
        def triton(ctx: object, node: object) -> str:
            return "triton"

        lowering = AtenLowering(codegen_impls=BackendDispatchTable({"triton": triton}))
        env = SimpleNamespace(codegen_name="triton", backend_name="triton")

        with (
            mock.patch.object(CompileEnvironment, "current", return_value=env),
            mock.patch(
                "helion._compiler.backend_registry.repair_backend_codegen"
            ) as repair_mock,
        ):
            result = lowering.codegen(mock.Mock(), mock.Mock())

        self.assertEqual(result, "triton")
        repair_mock.assert_called_once_with("triton")

    def test_missing_backend_repairs_before_common_fallback(self) -> None:
        def common(ctx: object, node: object) -> str:
            return "common"

        def triton(ctx: object, node: object) -> str:
            return "triton"

        lowering = AtenLowering(codegen_impls=BackendDispatchTable({"common": common}))
        env = SimpleNamespace(codegen_name="triton", backend_name="triton")

        def repair(codegen_name: str) -> None:
            lowering.codegen_impls[codegen_name] = triton

        with (
            mock.patch.object(CompileEnvironment, "current", return_value=env),
            mock.patch(
                "helion._compiler.backend_registry.repair_backend_codegen",
                side_effect=repair,
            ) as repair_mock,
        ):
            result = lowering.codegen(mock.Mock(), mock.Mock())

        self.assertEqual(result, "triton")
        repair_mock.assert_called_once_with("triton")

    def test_rejects_duplicate_registration_outside_repair(self) -> None:
        lowering = AtenLowering()

        def first(ctx: object, node: object) -> None:
            return None

        def second(ctx: object, node: object) -> None:
            return None

        lowering.register_codegen("triton")(first)
        with self.assertRaisesRegex(
            AssertionError, "codegen already registered for backend 'triton'"
        ):
            lowering.register_codegen("triton")(second)


if __name__ == "__main__":
    unittest.main()
