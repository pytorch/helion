from __future__ import annotations

import contextlib
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
import time
from typing import Iterator
import unittest
from unittest import mock
import warnings

import pytest

import helion
from helion._compiler import backend_registry
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.backend_registry import _REGISTRY
from helion._compiler.backend_registry import get_backend_class
from helion._compiler.backend_registry import list_backends
from helion._compiler.backend_registry import registry_generation
from helion._testing import onlyBackends
from helion.backend_api import TritonBackend
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_BACKEND = """
from helion import backend_api

class Plugin(backend_api.TritonBackend):
    HELION_BACKEND_API_VERSION = {version}

    @property
    def name(self):
        return {name!r}

    @property
    def codegen_name(self):
        return {name!r}
"""

_LOAD_HANDLER = """
from helion import backend_api
import helion.language as hl

@backend_api.codegen(hl.load, {codegen_name!r})
def _load(state):
    return None
"""


def _backend(name: str, version: int = 1) -> str:
    return _BACKEND.format(version=version, name=name)


@contextlib.contextmanager
def _isolated_registry_state() -> Iterator[None]:
    """Save and restore the registry, plugin-loading globals and the
    ``hl.load`` handlers (the only op the tests register on)."""
    registry_before = dict(_REGISTRY)
    repaired_before = backend_registry._REPAIRED_CODEGEN_NAMES
    handlers_before = dict(hl.load._codegen)
    try:
        with (
            mock.patch.object(backend_registry, "_PLUGIN_ENTRY_POINTS", None),
            mock.patch.object(backend_registry, "_FAILED_PLUGINS", {}),
            mock.patch.object(backend_registry, "_LOADING", set()),
            # The suite must not depend on the caller's environment.
            mock.patch.dict(os.environ),
        ):
            os.environ.pop("HELION_DISABLE_BACKEND_PLUGINS", None)
            os.environ.pop("HELION_BACKEND", None)  # e.g. the tileir CI lane
            yield
    finally:
        # Change-only restore: never blank the registry under a concurrent
        # reader.
        for key in [k for k in _REGISTRY if k not in registry_before]:
            _REGISTRY.pop(key, None)
        for key, value in registry_before.items():
            if _REGISTRY.get(key) is not value:
                _REGISTRY[key] = value
        backend_registry._GENERATION += 1  # derived tables forget the test's backends
        backend_registry._REPAIRED_CODEGEN_NAMES = repaired_before
        # Raw dict access: BackendDispatchTable lookups dispatch through repair.
        for key in [k for k in hl.load._codegen if k not in handlers_before]:
            dict.pop(hl.load._codegen, key, None)
        for key, handler in handlers_before.items():
            if dict.get(hl.load._codegen, key) is not handler:
                dict.__setitem__(hl.load._codegen, key, handler)


@contextlib.contextmanager
def _plugins(
    *specs: tuple[str, str, str, str | None],
    modules: dict[str, str] | None = None,
    entry_points: dict[str, str] | None = None,
) -> Iterator[mock.MagicMock]:
    """Install plugins given as (package, entry-point name, backend.py source,
    _codegen_modules.py source or None), plus extra ``modules`` (paths like
    "pkg/mod" without .py) and raw ``entry_points`` (name -> value)."""
    with tempfile.TemporaryDirectory() as tmp:
        eps = []
        for package, ep_name, backend_src, codegen_src in specs:
            pkg = Path(tmp) / package
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "backend.py").write_text(textwrap.dedent(backend_src))
            if codegen_src is not None:
                (pkg / "_codegen_modules.py").write_text(textwrap.dedent(codegen_src))
            eps.append(
                importlib.metadata.EntryPoint(
                    name=ep_name,
                    value=f"{package}.backend:Plugin",
                    group=backend_registry.PLUGIN_ENTRY_POINT_GROUP,
                )
            )
        for module, src in (modules or {}).items():
            path = Path(tmp) / f"{module}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.parent != Path(tmp):
                (path.parent / "__init__.py").touch()
            path.write_text(textwrap.dedent(src))
        eps.extend(
            importlib.metadata.EntryPoint(
                name=name, value=value, group=backend_registry.PLUGIN_ENTRY_POINT_GROUP
            )
            for name, value in (entry_points or {}).items()
        )
        sys.path.insert(0, tmp)
        try:
            with (
                _isolated_registry_state(),
                mock.patch.object(
                    importlib.metadata, "entry_points", return_value=eps
                ) as discover,
            ):
                yield discover
        finally:
            sys.path.remove(tmp)
            for name in [m for m in sys.modules if m.startswith(("_hplug", "_hsig"))]:
                del sys.modules[name]


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestLazyPluginLoading(unittest.TestCase):
    def test_listed_but_not_imported_until_requested(self) -> None:
        spec = ("_hplug_good", "_hplug_good", _backend("_hplug_good"), None)
        with _plugins(spec):
            self.assertIn("_hplug_good", list_backends())
            self.assertNotIn("_hplug_good", _REGISTRY)
            self.assertNotIn("_hplug_good.backend", sys.modules)
            cls = get_backend_class("_hplug_good")
            self.assertEqual(cls.__module__, "_hplug_good.backend")
            self.assertIs(_REGISTRY["_hplug_good"], cls)

    def test_heuristics_table_skips_unloaded_plugins(self) -> None:
        # The heuristics table is built at helion import; it must not import
        # plugins, and picks a plugin up once it loads.
        src = _backend("_hplug_heur") + (
            "\nclass Heuristic:\n    pass\n\n"
            "Plugin.autotuner_heuristics = lambda self: (Heuristic,)\n"
        )
        with _plugins(("_hplug_heur", "_hplug_heur", src, None)):
            backend_registry._GENERATION += 1  # force a rebuild, as at import
            self.assertEqual(get_heuristics("_hplug_heur"), ())
            self.assertNotIn("_hplug_heur.backend", sys.modules)
            get_backend_class("_hplug_heur")
            (heuristic,) = get_heuristics("_hplug_heur")
            self.assertEqual(heuristic.__module__, "_hplug_heur.backend")

    def test_codegen_modules_imported_on_load(self) -> None:
        codegen = _LOAD_HANDLER.format(codegen_name="_hplug_cg")
        with _plugins(("_hplug_cg", "_hplug_cg", _backend("_hplug_cg"), codegen)):
            get_backend_class("_hplug_cg")
            self.assertIn("_hplug_cg", hl.load._codegen)

    def test_unrequested_broken_plugin_is_never_imported(self) -> None:
        with (
            _plugins(
                ("_hplug_ok", "_hplug_ok", _backend("_hplug_ok"), None),
                ("_hplug_bad", "_hplug_bad", "raise ImportError('boom')", None),
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            get_backend_class("_hplug_ok")
            self.assertNotIn("_hplug_bad.backend", sys.modules)

    def test_load_backend_plugins_loads_all_and_warns(self) -> None:
        with (
            _plugins(
                ("_hplug_ok", "_hplug_ok", _backend("_hplug_ok"), None),
                ("_hplug_bad", "_hplug_bad", "raise ImportError('boom')", None),
            ),
            self.assertWarnsRegex(UserWarning, "_hplug_bad.*boom"),
        ):
            backend_registry.load_backend_plugins()
            self.assertIn("_hplug_ok", _REGISTRY)

    def test_disable_env(self) -> None:
        spec = ("_hplug_good", "_hplug_good", _backend("_hplug_good"), None)
        with (
            _plugins(spec) as discover,
            mock.patch.dict(os.environ, {"HELION_DISABLE_BACKEND_PLUGINS": "true"}),
        ):
            self.assertNotIn("_hplug_good", list_backends())
            discover.assert_not_called()


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestPluginLoadFailures(unittest.TestCase):
    def _assert_fails(self, name: str, pattern: str) -> None:
        registry_before = dict(_REGISTRY)
        with self.assertRaisesRegex(ValueError, f"{name}.*failed to load.*{pattern}"):
            get_backend_class(name)
        self.assertEqual(_REGISTRY, registry_before)
        # The error is remembered rather than retried.
        with self.assertRaisesRegex(ValueError, pattern):
            get_backend_class(name)

    def test_import_error(self) -> None:
        with _plugins(("_hplug_bad", "_hplug_bad", "raise ImportError('boom')", None)):
            self._assert_fails("_hplug_bad", "boom")

    def test_not_a_backend(self) -> None:
        with _plugins(("_hplug_nb", "_hplug_nb", "class Plugin:\n    pass\n", None)):
            self._assert_fails("_hplug_nb", "not a Backend subclass")

    def test_missing_api_version(self) -> None:
        src = _backend("_hplug_nov").replace("    HELION_BACKEND_API_VERSION = 1\n", "")
        with _plugins(("_hplug_nov", "_hplug_nov", src, None)):
            self._assert_fails("_hplug_nov", "HELION_BACKEND_API_VERSION=None")

    def test_api_version_mismatch(self) -> None:
        src = _backend("_hplug_old", version=999)
        with _plugins(("_hplug_old", "_hplug_old", src, None)):
            self._assert_fails("_hplug_old", "HELION_BACKEND_API_VERSION=999")

    def test_settings_for_a_built_in_does_not_scan_plugins(self) -> None:
        spec = ("_hplug_scan", "_hplug_scan", _backend("_hplug_scan"), None)
        with _plugins(spec) as discover:
            self.assertEqual(helion.Settings().backend, "triton")
            discover.assert_not_called()
            with mock.patch.dict(os.environ, {"HELION_BACKEND": "_hplug_scan"}):
                self.assertEqual(helion.Settings().backend, "_hplug_scan")
            discover.assert_called_once()

    def test_settings_raises_for_a_failing_plugin(self) -> None:
        # The backend is resolved (and the plugin imported) in Settings().
        with (
            _plugins(("_hplug_set", "_hplug_set", "raise ImportError('boom')", None)),
            self.assertRaisesRegex(ValueError, "_hplug_set.*failed to load.*boom"),
        ):
            helion.Settings(backend="_hplug_set")

    def test_api_version_true_rejected(self) -> None:
        # True == 1; the version must be an int, not a truthy stand-in.
        src = _backend("_hplug_bool", version=True)
        with _plugins(("_hplug_bool", "_hplug_bool", src, None)):
            self._assert_fails("_hplug_bool", "HELION_BACKEND_API_VERSION=True")

    def test_invalid_disable_value_raises(self) -> None:
        with (
            mock.patch.object(backend_registry, "_PLUGIN_ENTRY_POINTS", None),
            mock.patch.dict(os.environ, {"HELION_DISABLE_BACKEND_PLUGINS": "maybe"}),
            self.assertRaisesRegex(
                ValueError, "HELION_DISABLE_BACKEND_PLUGINS must be one of"
            ),
        ):
            list_backends()

    def test_entry_point_name_must_match_backend_name(self) -> None:
        with _plugins(("_hplug_x", "_hplug_alias", _backend("_hplug_real"), None)):
            self._assert_fails("_hplug_alias", "named '_hplug_real'")

    def test_replacing_a_registered_backend_is_refused(self) -> None:
        src = textwrap.dedent(
            """
            from helion import backend_api

            class Impostor(backend_api.TritonBackend):
                pass

            backend_api.register_compiler_backend(Impostor)
            """
        ) + _backend("_hplug_imp")
        with _plugins(("_hplug_imp", "_hplug_imp", src, None)):
            self._assert_fails("_hplug_imp", "already registered to")
            self.assertIs(_REGISTRY["triton"], TritonBackend)

    def test_self_registration_is_accepted(self) -> None:
        src = _backend("_hplug_self") + (
            "\nbackend_api.register_compiler_backend(Plugin)\n"
        )
        with _plugins(("_hplug_self", "_hplug_self", src, None)):
            self.assertEqual(get_backend_class("_hplug_self").__name__, "Plugin")

    def test_codegen_module_cannot_replace_a_backend(self) -> None:
        codegen = """
        from helion import backend_api

        class Impostor(backend_api.TritonBackend):
            pass

        backend_api.register_compiler_backend(Impostor)
        """
        with _plugins(
            ("_hplug_cgimp", "_hplug_cgimp", _backend("_hplug_cgimp"), codegen)
        ):
            self._assert_fails("_hplug_cgimp", "already registered to")
            self.assertIs(_REGISTRY["triton"], TritonBackend)

    def test_plugin_looking_itself_up_while_importing(self) -> None:
        src = (
            "from helion._compiler.backend_registry import get_backend_class\n"
            "try:\n"
            "    get_backend_class('_hplug_selfref')\n"
            "except ValueError as e:\n"
            "    ERROR = str(e)\n" + _backend("_hplug_selfref")
        )
        with _plugins(("_hplug_selfref", "_hplug_selfref", src, None)):
            self.assertEqual(get_backend_class("_hplug_selfref").__name__, "Plugin")
            self.assertEqual(backend_registry._FAILED_PLUGINS, {})
            module = sys.modules["_hplug_selfref.backend"]
            self.assertIn("'_hplug_selfref' is still loading", module.ERROR)

    def test_failed_self_registration_under_another_name_is_removed(self) -> None:
        src = _backend("_hplug_real2") + (
            "\nbackend_api.register_compiler_backend(Plugin)\n"
        )
        with _plugins(("_hplug_x2", "_hplug_alias2", src, None)):
            self._assert_fails("_hplug_alias2", "named '_hplug_real2'")
            self.assertNotIn("_hplug_real2", _REGISTRY)

    def test_self_registered_then_failed_import_is_removed(self) -> None:
        src = (
            _backend("_hplug_srf")
            + "\nbackend_api.register_compiler_backend(Plugin)\n"
            + "raise RuntimeError('after registering')\n"
        )
        with _plugins(("_hplug_srf", "_hplug_srf", src, None)):
            self._assert_fails("_hplug_srf", "after registering")
            self.assertNotIn("_hplug_srf", _REGISTRY)

    def test_failed_load_keeps_backends_registered_before_it(self) -> None:
        # An entry point naming an already registered class (here a built-in)
        # fails validation; the cleanup must not unregister that class.
        with _plugins(entry_points={"_hplug_bi": "helion.backend_api:TritonBackend"}):
            self._assert_fails("_hplug_bi", "HELION_BACKEND_API_VERSION=None")
            self.assertIs(_REGISTRY["triton"], TritonBackend)

    def test_failed_load_removes_other_names_it_registered(self) -> None:
        src = (
            _backend("_hplug_extra", version=999)
            + "\nclass Extra(Plugin):\n"
            + "    @property\n    def name(self):\n        return '_hplug_extra2'\n\n"
            + "backend_api.register_compiler_backend(Extra)\n"
        )
        with _plugins(("_hplug_extra", "_hplug_extra", src, None)):
            self._assert_fails("_hplug_extra", "HELION_BACKEND_API_VERSION=999")
            self.assertNotIn("_hplug_extra2", _REGISTRY)

    def test_interrupted_load_is_remembered(self) -> None:
        # A half-imported plugin would not re-import cleanly, so a Ctrl-C
        # during its load is remembered like any other failure.
        with _plugins(("_hplug_int", "_hplug_int", "raise KeyboardInterrupt\n", None)):
            with self.assertRaises(KeyboardInterrupt):
                get_backend_class("_hplug_int")
            self.assertNotIn("_hplug_int", _REGISTRY)
            try:
                with self.assertRaisesRegex(
                    ValueError, r"interrupted \(KeyboardInterrupt\)"
                ):
                    get_backend_class("_hplug_int")
            except KeyboardInterrupt:
                self.fail("the interrupted load was retried")


_WAITING_PLUGIN_SRC = """
import _hsigwait

_hsigwait.STARTED.set()
if not _hsigwait.RELEASE.wait(30):
    raise RuntimeError("release timed out")
"""

_WAIT_MODULE_SRC = """
import threading

STARTED = threading.Event()
RELEASE = threading.Event()
"""


def _start_loading(name: str) -> tuple[threading.Thread, list[object]]:
    return _start_loading_into(name, [])


def _start_loading_into(
    name: str, results: list[object]
) -> tuple[threading.Thread, list[object]]:
    def run() -> None:
        try:
            results.append(get_backend_class(name))
        except BaseException as e:
            results.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, results


def _wait_for_signal(module: str = "_hsigwait") -> object:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        mod = sys.modules.get(module)
        if mod is not None and getattr(mod, "STARTED", None) is not None:
            assert mod.STARTED.wait(10)
            return mod
        time.sleep(0.01)
    raise AssertionError(f"{module} never imported")


class _ContendedLock:
    """Wraps a lock; ``contended`` is set once an acquire has to wait."""

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self.contended = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        self.contended.set()
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestPluginLoadConcurrency(unittest.TestCase):
    def test_registry_lock_free_during_plugin_import(self) -> None:
        # Plugin imports run arbitrary code and must not hold
        # _CODEGEN_REPAIR_LOCK: a concurrent compile takes that lock and can
        # deadlock against an import lock the plugin chain is waiting on.
        src = _WAITING_PLUGIN_SRC + _backend("_hplug_blk")
        with _plugins(
            ("_hplug_blk", "_hplug_blk", src, None),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            thread, results = _start_loading("_hplug_blk")
            try:
                _wait_for_signal()
                acquired = backend_registry._CODEGEN_REPAIR_LOCK.acquire(timeout=10)
                self.assertTrue(acquired, "plugin import holds the codegen repair lock")
                backend_registry._CODEGEN_REPAIR_LOCK.release()
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
                thread.join(30)
        self.assertEqual(
            [getattr(r, "__name__", r) for r in results], ["Plugin"], results
        )

    def test_threads_first_touching_a_plugin_share_one_load(self) -> None:
        # Other threads wait for the one load and get its class.
        src = (
            "import _hsigcount\n_hsigcount.IMPORTS.append(1)\n"
            + _WAITING_PLUGIN_SRC
            + _backend("_hplug_shared")
        )
        lock = _ContendedLock(backend_registry._PLUGIN_LOAD_LOCK)
        with (
            _plugins(
                ("_hplug_shared", "_hplug_shared", src, None),
                modules={"_hsigwait": _WAIT_MODULE_SRC, "_hsigcount": "IMPORTS = []\n"},
            ),
            mock.patch.object(backend_registry, "_PLUGIN_LOAD_LOCK", lock),
        ):
            first, results = _start_loading("_hplug_shared")
            others: list[threading.Thread] = []
            try:
                _wait_for_signal()
                for _ in range(4):
                    thread, _ = _start_loading_into("_hplug_shared", results)
                    others.append(thread)
                self.assertTrue(lock.contended.wait(30))
                self.assertEqual(len(results), 0)
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            for thread in [first, *others]:
                thread.join(30)
            self.assertEqual(
                [getattr(r, "__name__", r) for r in results], ["Plugin"] * 5, results
            )
            self.assertEqual(sys.modules["_hsigcount"].IMPORTS, [1])

    def test_discovery_does_not_wait_for_a_load(self) -> None:
        # list_backends() and Settings() for another backend must not block on
        # another thread's plugin import.
        src = _WAITING_PLUGIN_SRC + _backend("_hplug_slow")
        with _plugins(
            ("_hplug_slow", "_hplug_slow", src, None),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            thread, results = _start_loading("_hplug_slow")
            done: list[object] = []

            def discover() -> None:
                done.append(list_backends())
                done.append(helion.Settings(backend="triton").backend)

            other = threading.Thread(target=discover, daemon=True)
            try:
                _wait_for_signal()
                other.start()
                other.join(30)
                self.assertFalse(other.is_alive(), "discovery waited for the load")
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            thread.join(30)
            self.assertIn("_hplug_slow", done[0])  # pyrefly: ignore
            self.assertEqual(done[1], "triton")
            self.assertEqual([getattr(r, "__name__", r) for r in results], ["Plugin"])

    def test_heuristics_lookup_does_not_wait_for_a_load(self) -> None:
        # A plugin that registered itself is in the registry while its load is
        # still running on another thread; rebuilding the heuristics table must
        # not wait for that load.
        src = (
            _backend("_hplug_hreg")
            + "\nbackend_api.register_compiler_backend(Plugin)\n"
            + _WAITING_PLUGIN_SRC
        )
        with _plugins(
            ("_hplug_hreg", "_hplug_hreg", src, None),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            thread, results = _start_loading("_hplug_hreg")
            done: list[object] = []

            def lookup() -> None:
                backend_registry._GENERATION += 1  # force a rebuild
                done.append(get_heuristics("triton"))

            other = threading.Thread(target=lookup, daemon=True)
            try:
                _wait_for_signal()
                self.assertIn("_hplug_hreg", _REGISTRY)
                other.start()
                other.join(30)
                self.assertFalse(other.is_alive(), "heuristics waited for the load")
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            thread.join(30)
            self.assertEqual(len(done), 1)
            self.assertEqual([getattr(r, "__name__", r) for r in results], ["Plugin"])


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestPluginLoadCleanup(unittest.TestCase):
    def test_failed_load_bumps_registry_generation(self) -> None:
        # Tables derived from the registry (config keys, heuristics) key their
        # caches on registry_generation(); a failed load must invalidate them.
        codegen = (
            "import sys\n"
            "from helion._compiler import backend_registry\n"
            "sys._hplug_gen = backend_registry.registry_generation()\n"
            "raise RuntimeError('codegen boom')\n"
        )
        try:
            with _plugins(
                ("_hplug_gen", "_hplug_gen", _backend("_hplug_gen"), codegen)
            ):
                with self.assertRaisesRegex(ValueError, "codegen boom"):
                    get_backend_class("_hplug_gen")
                self.assertNotIn("_hplug_gen", _REGISTRY)
                # pyrefly: ignore[missing-attribute]
                self.assertGreater(registry_generation(), sys._hplug_gen)
        finally:
            sys.__dict__.pop("_hplug_gen", None)

    def test_cannot_overwrite_builtin_handler(self) -> None:
        builtin = dict.__getitem__(hl.load._codegen, "triton")
        codegen = _LOAD_HANDLER.format(codegen_name="triton")
        with _plugins(("_hplug_ow", "_hplug_ow", _backend("_hplug_ow"), codegen)):
            with self.assertRaisesRegex(ValueError, "already registered"):
                get_backend_class("_hplug_ow")
            # Inside the fixture, which restores the handlers on exit.
            self.assertIs(dict.__getitem__(hl.load._codegen, "triton"), builtin)


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestEntryPointNames(unittest.TestCase):
    def test_builtin_name_ignored(self) -> None:
        with _plugins(("_hplug_t", "triton", _backend("triton"), None)):
            with self.assertWarnsRegex(UserWarning, "'triton' is a built-in backend"):
                list_backends()
            self.assertIs(get_backend_class("triton"), TritonBackend)
            backend_registry.load_backend_plugins()
            self.assertNotIn("_hplug_t.backend", sys.modules)

    def test_builtin_name_warning_as_error_does_not_break_discovery(self) -> None:
        with (
            _plugins(("_hplug_t", "triton", _backend("triton"), None)),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            with self.assertRaisesRegex(UserWarning, "is a built-in backend"):
                list_backends()
            self.assertIn("triton", list_backends())  # discovered once, no re-raise

    def test_duplicate_name_is_an_error(self) -> None:
        src = _backend("_hplug_same")
        with (
            _plugins(
                ("_hplug_b", "_hplug_same", src, None),
                ("_hplug_a", "_hplug_same", src, None),
            ),
            mock.patch.object(
                backend_registry, "_dist_name", side_effect=lambda ep: ep.value
            ),
        ):
            self.assertIn("_hplug_same", list_backends())
            with self.assertRaisesRegex(
                ValueError,
                "'_hplug_same' is provided by more than one installed package: "
                "'_hplug_a.backend:Plugin', '_hplug_b.backend:Plugin'; "
                "uninstall all but one",
            ):
                get_backend_class("_hplug_same")
            with self.assertWarnsRegex(UserWarning, "more than one installed"):
                backend_registry.load_backend_plugins()
            self.assertNotIn("_hplug_a.backend", sys.modules)
            self.assertNotIn("_hplug_b.backend", sys.modules)

    def test_duplicate_name_with_hand_registered_class(self) -> None:
        # An explicit registration wins; neither provider is imported.
        src = _backend("_hplug_same")

        class Manual(TritonBackend):
            @property
            def name(self) -> str:
                return "_hplug_same"

        with _plugins(
            ("_hplug_b", "_hplug_same", src, None),
            ("_hplug_a", "_hplug_same", src, None),
        ):
            backend_registry.register_compiler_backend(Manual)
            self.assertIs(get_backend_class("_hplug_same"), Manual)
            self.assertNotIn("_hplug_a.backend", sys.modules)
            self.assertNotIn("_hplug_b.backend", sys.modules)

    def test_duplicate_resolution_from_real_dist_infos(self) -> None:
        # No mocking of discovery: two on-path .dist-info dirs, so the error
        # uses real distribution attribution via EntryPoint.dist.
        with tempfile.TemporaryDirectory() as tmp:
            for dist, pkg in (
                ("z-legit", "_hplug_zlegit"),
                ("a-shadow", "_hplug_ashadow"),
            ):
                dist_info = Path(tmp) / f"{dist}-1.0.dist-info"
                dist_info.mkdir()
                (dist_info / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {dist}\n"
                )
                (dist_info / "entry_points.txt").write_text(
                    f"[{backend_registry.PLUGIN_ENTRY_POINT_GROUP}]\n"
                    f"_hplug_same = {pkg}.backend:Plugin\n"
                )
                pkg_dir = Path(tmp) / pkg
                pkg_dir.mkdir()
                (pkg_dir / "__init__.py").write_text("")
                (pkg_dir / "backend.py").write_text(
                    textwrap.dedent(_backend("_hplug_same"))
                )
            sys.path.insert(0, tmp)
            try:
                with _isolated_registry_state():
                    self.assertIn("_hplug_same", list_backends())
                    with self.assertRaisesRegex(
                        ValueError,
                        "provided by more than one installed package: "
                        "'a-shadow', 'z-legit'",
                    ):
                        get_backend_class("_hplug_same")
                    self.assertNotIn("_hplug_ashadow.backend", sys.modules)
                    self.assertNotIn("_hplug_zlegit.backend", sys.modules)
            finally:
                sys.path.remove(tmp)
                for name in [m for m in sys.modules if m.startswith("_hplug_")]:
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
