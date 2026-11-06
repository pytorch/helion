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
import types
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
            mock.patch.object(backend_registry, "_LOADED_PLUGINS", set()),
            mock.patch.object(backend_registry, "_LOADING_PLUGINS", {}),
            # The suite must not depend on the caller's environment.
            mock.patch.dict(os.environ),
        ):
            os.environ.pop("HELION_DISABLE_BACKEND_PLUGINS", None)
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
            for name in [
                m for m in sys.modules if m.startswith(("_hplug", "_hside", "_hsig"))
            ]:
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

    def test_replacing_a_registered_backend_is_reverted(self) -> None:
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


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestNestedPluginLoads(unittest.TestCase):
    def test_plugin_importing_another_plugin(self) -> None:
        uses_dep = (
            "from helion._compiler.backend_registry import get_backend_class\n"
            "get_backend_class('_hplug_dep')\n" + _backend("_hplug_user")
        )
        with _plugins(
            ("_hplug_user", "_hplug_user", uses_dep, None),
            ("_hplug_dep", "_hplug_dep", _backend("_hplug_dep"), None),
        ):
            self.assertEqual(get_backend_class("_hplug_user").__name__, "Plugin")
            self.assertIn("_hplug_dep", _REGISTRY)
            self.assertEqual(backend_registry._FAILED_PLUGINS, {})

    def test_inner_plugin_survives_outer_failure_in_same_package(self) -> None:
        # One distribution, two backends: the outer loads the inner, then fails.
        # Rolling back the outer must not strip the inner's handlers.
        inner = _LOAD_HANDLER.format(codegen_name="_hplug_inner") + _backend(
            "_hplug_inner"
        )
        outer = (
            "from helion._compiler.backend_registry import get_backend_class\n"
            "get_backend_class('_hplug_inner')\n"
            "raise RuntimeError('outer broke')\n"
        )
        with _plugins(
            modules={"_hplug_pair/inner": inner, "_hplug_pair/outer": outer},
            entry_points={
                "_hplug_inner": "_hplug_pair.inner:Plugin",
                "_hplug_outer": "_hplug_pair.outer:Plugin",
            },
        ):
            with self.assertRaisesRegex(ValueError, "outer broke"):
                get_backend_class("_hplug_outer")
            self.assertIn("_hplug_inner", _REGISTRY)
            self.assertIn("_hplug_inner", hl.load._codegen)

    def test_plugin_looking_itself_up_while_importing(self) -> None:
        src = (
            "from helion._compiler.backend_registry import get_backend_class\n"
            "try:\n"
            "    get_backend_class('_hplug_selfref')\n"
            "except ValueError:\n"
            "    pass\n" + _backend("_hplug_selfref")
        )
        with _plugins(("_hplug_selfref", "_hplug_selfref", src, None)):
            self.assertEqual(get_backend_class("_hplug_selfref").__name__, "Plugin")
            self.assertEqual(backend_registry._FAILED_PLUGINS, {})


_WAITING_PLUGIN_SRC = """
import _hsigwait

_hsigwait.STARTED.set()
assert _hsigwait.RELEASE.wait(30), "release timed out"
"""

_WAIT_MODULE_SRC = """
import threading

STARTED = threading.Event()
RELEASE = threading.Event()
"""


def _start_loading(name: str) -> tuple[threading.Thread, list[object]]:
    results: list[object] = []

    def run() -> None:
        try:
            results.append(get_backend_class(name))
        except BaseException as e:
            results.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, results


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


def _wait_until_loading(name: str) -> None:
    """Block until some thread has started loading plugin ``name``."""
    deadline = time.monotonic() + 30
    while name not in backend_registry._LOADING_PLUGINS:
        if time.monotonic() > deadline:
            raise AssertionError(f"{name!r} never started loading")
        time.sleep(0.01)
    time.sleep(0.1)  # let it reach the import


def _wait_for_signal(module: str = "_hsigwait") -> object:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        mod = sys.modules.get(module)
        if mod is not None and getattr(mod, "STARTED", None) is not None:
            assert mod.STARTED.wait(10)
            return mod
        time.sleep(0.01)
    raise AssertionError(f"{module} never imported")


def _signal_module(module: str) -> types.ModuleType:
    # Importing (not polling sys.modules) waits until another thread has
    # finished initializing it.
    return importlib.import_module(module)


@contextlib.contextmanager
def _same_thread_repair() -> Iterator[None]:
    from helion.language import _decorators

    _decorators._begin_codegen_repair(frozenset({"_hrepair"}))
    try:
        yield
    finally:
        _decorators._end_codegen_repair()


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

    def test_direct_import_racing_a_lookup_completes(self) -> None:
        # Thread A imports a self-registering plugin module directly; thread B
        # looks the plugin up meanwhile, takes the load lock and waits on A's
        # import.  A's registration must not need the load lock, or this hangs.
        src = (
            _WAITING_PLUGIN_SRC
            + _backend("_hplug_direct")
            + "\nbackend_api.register_compiler_backend(Plugin)\n"
        )
        with _plugins(
            ("_hplug_direct", "_hplug_direct", src, None),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            imported: list[object] = []

            def direct_import() -> None:
                try:
                    imported.append(importlib.import_module("_hplug_direct.backend"))
                except BaseException as e:
                    imported.append(e)

            importer = threading.Thread(target=direct_import, daemon=True)
            importer.start()
            try:
                _wait_for_signal()
                looker, looked = _start_loading("_hplug_direct")
                _wait_until_loading("_hplug_direct")
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            importer.join(30)
            looker.join(30)
            self.assertFalse(importer.is_alive() or looker.is_alive(), "hung")
            self.assertIsInstance(imported[0], types.ModuleType)
            self.assertIs(looked[0], imported[0].Plugin)  # pyrefly: ignore
            self._assert_no_load_state()

    def test_other_thread_waits_for_half_loaded_plugin(self) -> None:
        # Registered but its codegen modules still importing: another thread
        # must wait for the outcome, not get the uncommitted class.
        codegen = (
            "import _hsigwait\n\n_hsigwait.STARTED.set()\n"
            'assert _hsigwait.RELEASE.wait(30), "release timed out"\n'
            "raise RuntimeError('codegen boom')\n"
        )
        lock = _ContendedLock(backend_registry._PLUGIN_LOAD_LOCK)
        with (
            _plugins(
                (
                    "_hplug_halfreg",
                    "_hplug_halfreg",
                    _backend("_hplug_halfreg"),
                    codegen,
                ),
                modules={"_hsigwait": _WAIT_MODULE_SRC},
            ),
            mock.patch.object(backend_registry, "_PLUGIN_LOAD_LOCK", lock),
        ):
            t1, r1 = _start_loading("_hplug_halfreg")
            try:
                _wait_for_signal()
                self.assertIn("_hplug_halfreg", _REGISTRY)
                t2, r2 = _start_loading("_hplug_halfreg")
                self.assertTrue(lock.contended.wait(30))
                self.assertEqual(r2, [])
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            t1.join(30)
            t2.join(30)
            self.assertNotIn("_hplug_halfreg", _REGISTRY)
            for results in (r1, r2):
                self.assertEqual(len(results), 1, results)
                self.assertIsInstance(results[0], ValueError)
                self.assertIn("codegen boom", str(results[0]))

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

    def test_own_lookup_during_direct_import_times_out(self) -> None:
        # Thread A imports the plugin module directly and the module looks up
        # its own backend; thread B holds the load lock, waiting on A's import.
        # A gets the timeout error instead of both hanging.
        src = (
            _WAITING_PLUGIN_SRC
            + "import _hsigout\n"
            + "import helion\n"
            + "try:\n"
            + "    helion.Settings(backend='_hplug_ownlook')\n"
            + "    _hsigout.SEEN.append('found')\n"
            + "except (ValueError, RuntimeError) as e:\n"
            + "    _hsigout.SEEN.append(e)\n"
            + _backend("_hplug_ownlook")
        )
        with (
            _plugins(
                ("_hplug_ownlook", "_hplug_ownlook", src, None),
                modules={"_hsigwait": _WAIT_MODULE_SRC, "_hsigout": "SEEN = []\n"},
            ),
            mock.patch.object(backend_registry, "_PLUGIN_LOAD_TIMEOUT_S", 0.5),
        ):
            importer = threading.Thread(
                target=importlib.import_module, args=("_hplug_ownlook.backend",)
            )
            importer.daemon = True
            importer.start()
            try:
                _wait_for_signal()
                looker, looked = _start_loading("_hplug_ownlook")
                _wait_until_loading("_hplug_ownlook")
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            importer.join(30)
            looker.join(30)
            self.assertFalse(importer.is_alive() or looker.is_alive(), "hung")
            seen = sys.modules["_hsigout"].SEEN
            self.assertIsInstance(seen[0], RuntimeError, seen)
            self.assertRegex(str(seen[0]), "timed out.*'_hplug_ownlook'")
            self.assertEqual([getattr(r, "__name__", r) for r in looked], ["Plugin"])

    def test_repair_skips_a_plugin_mid_load(self) -> None:
        # A plugin sharing triton codegen is in triton's repair scope; a repair
        # on another thread must not import its codegen modules while its
        # loader holds their import lock (and will next need the repair lock).
        backend_src = _backend("_hplug_share").replace(
            "        return '_hplug_share'\n\n    @property\n    def codegen_name(self):\n        return '_hplug_share'",
            "        return '_hplug_share'\n\n    @property\n    def codegen_name(self):\n        return 'triton'",
        )
        self.assertIn("return 'triton'", backend_src)
        codegen = (
            "import _hsigwait\n\n_hsigwait.STARTED.set()\n"
            'assert _hsigwait.RELEASE.wait(30), "release timed out"\n'
        )
        with (
            _plugins(
                ("_hplug_share", "_hplug_share", backend_src, codegen),
                modules={"_hsigwait": _WAIT_MODULE_SRC},
            ),
            mock.patch.object(backend_registry.importlib, "reload"),
        ):
            loader, loaded = _start_loading("_hplug_share")
            repairer = threading.Thread(
                target=backend_registry.repair_backend_codegen,
                args=("triton",),
                daemon=True,
            )
            try:
                _wait_for_signal()
                backend_registry._REPAIRED_CODEGEN_NAMES = frozenset()
                repairer.start()
                repairer.join(30)
                self.assertFalse(repairer.is_alive(), "repair waited on the plugin")
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            loader.join(30)
            self.assertEqual([getattr(r, "__name__", r) for r in loaded], ["Plugin"])

    def test_load_all_warns_on_retryable_refusal(self) -> None:
        specs = tuple(
            (f"_hplug_ra{name}", f"_hplug_ra{name}", _backend(f"_hplug_ra{name}"), None)
            for name in ("a", "b")
        )
        with _plugins(*specs):
            with (
                _same_thread_repair(),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                backend_registry.load_backend_plugins()  # must not raise
            refusals = [
                w for w in caught if "during a codegen repair" in str(w.message)
            ]
            self.assertEqual(len(refusals), 2, [str(w.message) for w in caught])
            self.assertEqual([k for k in _REGISTRY if k.startswith("_hplug_ra")], [])
            self.assertEqual(get_backend_class("_hplug_raa").__name__, "Plugin")

    def test_load_during_codegen_repair_is_retryable(self) -> None:
        # Handler registrations made while a codegen repair is in progress
        # would be dropped by the repair filter; the load is refused without
        # poisoning the plugin, so a later lookup still succeeds.
        spec = ("_hplug_repair", "_hplug_repair", _backend("_hplug_repair"), None)
        with _plugins(spec):
            with _same_thread_repair():
                with self.assertRaisesRegex(RuntimeError, "during a codegen repair"):
                    get_backend_class("_hplug_repair")
                self.assertEqual(backend_registry._FAILED_PLUGINS, {})
            self.assertEqual(get_backend_class("_hplug_repair").__name__, "Plugin")

    def test_load_under_the_repair_lock_is_refused(self) -> None:
        # E.g. a backend constructed by register_compiler_backend (under the
        # repair lock) that looks up a plugin: refused, not a lock inversion.
        spec = ("_hplug_under", "_hplug_under", _backend("_hplug_under"), None)
        with _plugins(spec):
            with (
                backend_registry._repair_lock(),
                self.assertRaisesRegex(RuntimeError, "during a codegen repair"),
            ):
                get_backend_class("_hplug_under")
            self.assertEqual(get_backend_class("_hplug_under").__name__, "Plugin")

    def test_load_waits_for_other_thread_repair(self) -> None:
        # A repair on another thread is waited out, not raised into the
        # caller's compile.
        spec = ("_hplug_xrep", "_hplug_xrep", _backend("_hplug_xrep"), None)
        entered, release = threading.Event(), threading.Event()
        lock = _ContendedLock(backend_registry._CODEGEN_REPAIR_LOCK)

        def blocking_reload(codegen_name: str) -> None:
            entered.set()
            assert release.wait(30), "release timed out"

        with (
            _plugins(spec),
            mock.patch.object(
                backend_registry, "_reload_backend_codegen", blocking_reload
            ),
            mock.patch.object(backend_registry, "_CODEGEN_REPAIR_LOCK", lock),
        ):
            backend_registry._REPAIRED_CODEGEN_NAMES = frozenset()
            repair = threading.Thread(
                target=backend_registry.repair_backend_codegen,
                args=("triton",),
                daemon=True,
            )
            repair.start()
            try:
                self.assertTrue(entered.wait(30))
                thread, results = _start_loading("_hplug_xrep")
                self.assertTrue(lock.contended.wait(30), results)
                self.assertEqual(results, [])
                self.assertNotIn("_hplug_xrep.backend", sys.modules)
            finally:
                release.set()
            repair.join(30)
            thread.join(30)
            self.assertEqual(
                [getattr(r, "__name__", r) for r in results], ["Plugin"], results
            )

    def test_failed_load_keeps_concurrent_public_registration(self) -> None:
        # A backend another thread registers through the public API while a
        # plugin load is in flight must survive that plugin's rollback.
        src = _WAITING_PLUGIN_SRC + "raise RuntimeError('boom')\n"

        class UserBackend(TritonBackend):
            @property
            def name(self) -> str:
                return "_huser_be"

        with _plugins(
            ("_hplug_ureg", "_hplug_ureg", src, None),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            thread, results = _start_loading("_hplug_ureg")
            try:
                _wait_for_signal()
                backend_registry.register_compiler_backend(UserBackend)
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            thread.join(30)
            self.assertIsInstance(results[0], ValueError)
            self.assertIs(_REGISTRY.get("_huser_be"), UserBackend)

    def test_failed_load_keeps_a_namespace_sibling(self) -> None:
        # Two plugins under one parent package: the failed one's rollback must
        # not remove what its sibling registered (matched by entry-point
        # package, not top-level package).
        sibling = _backend("_hplug_ns_ok") + (
            "\nbackend_api.register_compiler_backend(Plugin)\n"
        )
        failing = "import _hplug_ns.ok.backend\nraise RuntimeError('ns boom')\n"
        with _plugins(
            modules={
                "_hplug_ns/ok/backend": sibling,
                "_hplug_ns/bad/backend": failing,
            },
            entry_points={"_hplug_ns_bad": "_hplug_ns.bad.backend:Plugin"},
        ):
            with self.assertRaisesRegex(ValueError, "ns boom"):
                get_backend_class("_hplug_ns_bad")
            self.assertIn("_hplug_ns_ok", _REGISTRY)

    def test_failed_load_removes_class_package_handlers(self) -> None:
        # The class (and its _codegen_modules) live outside the entry point's
        # package; handlers registered there are undone too.
        codegen = _LOAD_HANDLER.format(codegen_name="_hplug_clspkg") + (
            "\nraise RuntimeError('codegen boom')\n"
        )
        with _plugins(
            modules={
                "_hside_cls/backends": _backend("_hplug_clspkg"),
                "_hside_cls/_codegen_modules": codegen,
                "_hplug_front/backend": "from _hside_cls.backends import Plugin\n",
            },
            entry_points={"_hplug_clspkg": "_hplug_front.backend:Plugin"},
        ):
            with self.assertRaisesRegex(ValueError, "codegen boom"):
                get_backend_class("_hplug_clspkg")
            self.assertNotIn("_hplug_clspkg", hl.load._codegen)

    def test_failed_load_removes_a_re_exported_class(self) -> None:
        # The entry point's class lives outside the plugin's package; a failed
        # load must still remove it, or lookups would return a dead backend.
        core = _backend("_hplug_reexp")
        front = (
            "from _hside_core.backends import Plugin\n"
            "from helion import backend_api\n"
            "backend_api.register_compiler_backend(Plugin)\n"
        )
        codegen = "raise RuntimeError('codegen boom')\n"
        with _plugins(
            modules={
                "_hside_core/backends": core,
                "_hside_core/_codegen_modules": codegen,
                "_hplug_reexp/backend": front,
            },
            entry_points={"_hplug_reexp": "_hplug_reexp.backend:Plugin"},
        ):
            with self.assertRaisesRegex(ValueError, "codegen boom"):
                get_backend_class("_hplug_reexp")
            self.assertNotIn("_hplug_reexp", _REGISTRY)
            with self.assertRaisesRegex(ValueError, "codegen boom"):
                get_backend_class("_hplug_reexp")

    def _assert_no_load_state(self) -> None:
        self.assertEqual(backend_registry._LOADING_PLUGINS, {})
        self.assertEqual(backend_registry._FAILED_PLUGINS, {})


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestPluginHandlerRollback(unittest.TestCase):
    def test_rollback_error_does_not_hide_load_error(self) -> None:
        src = "raise RuntimeError('own error')\n"
        with (
            _plugins(("_hplug_rbe", "_hplug_rbe", src, None)),
            mock.patch.object(
                backend_registry,
                "_undo_plugin_handlers",
                side_effect=RuntimeError("rollback boom"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "own error") as ctx:
                get_backend_class("_hplug_rbe")
            self.assertNotIn("rollback boom", str(ctx.exception))
            self.assertIn("own error", backend_registry._FAILED_PLUGINS["_hplug_rbe"])
            cause = ctx.exception.__cause__
            self.assertIn("rollback boom", str(getattr(cause, "__notes__", [])))
            self.assertNotIn("_hplug_rbe", _REGISTRY)

    def test_failed_codegen_module_handlers_removed(self) -> None:
        codegen = _LOAD_HANDLER.format(codegen_name="_hplug_half") + (
            "\nraise RuntimeError('half imported')\n"
        )
        with _plugins(("_hplug_half", "_hplug_half", _backend("_hplug_half"), codegen)):
            with self.assertRaisesRegex(ValueError, "half imported"):
                get_backend_class("_hplug_half")
            self.assertNotIn("_hplug_half", hl.load._codegen)

    def test_rollback_bumps_registry_generation(self) -> None:
        # Tables derived from the registry (config keys, heuristics) key their
        # caches on registry_generation(); a rollback must invalidate them.
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

    def test_skipped_plugin_handlers_removed(self) -> None:
        # Registered at import time by a plugin then rejected on version.
        src = _LOAD_HANDLER.format(codegen_name="_hplug_old") + _backend(
            "_hplug_old", version=999
        )
        with _plugins(("_hplug_old", "_hplug_old", src, None)):
            with self.assertRaises(ValueError):
                get_backend_class("_hplug_old")
            self.assertNotIn("_hplug_old", hl.load._codegen)

    def test_handlers_from_other_packages_kept(self) -> None:
        # A plugin may trigger imports that register handlers it doesn't own
        # (e.g. lazily imported Helion modules); rollback must leave them.
        side = _LOAD_HANDLER.format(codegen_name="_hside")
        src = "import _hside_reg\nraise RuntimeError('after side import')\n"
        with _plugins(
            ("_hplug_side", "_hplug_side", src, None), modules={"_hside_reg": side}
        ):
            with self.assertRaisesRegex(ValueError, "after side import"):
                get_backend_class("_hplug_side")
            self.assertIn("_hside", hl.load._codegen)

    def test_cannot_overwrite_builtin_handler(self) -> None:
        builtin = dict.__getitem__(hl.load._codegen, "triton")
        codegen = _LOAD_HANDLER.format(codegen_name="triton")
        with (
            _plugins(("_hplug_ow", "_hplug_ow", _backend("_hplug_ow"), codegen)),
            self.assertRaisesRegex(ValueError, "already registered"),
        ):
            get_backend_class("_hplug_ow")
        self.assertIs(dict.__getitem__(hl.load._codegen, "triton"), builtin)


@onlyBackends(["triton"])  # backend-independent; run in one lane
class TestEntryPointNames(unittest.TestCase):
    def test_builtin_name_ignored(self) -> None:
        with (
            _plugins(("_hplug_t", "triton", _backend("triton"), None)),
            self.assertWarnsRegex(UserWarning, "'triton' is a built-in backend"),
        ):
            list_backends()
        self.assertIs(get_backend_class("triton"), TritonBackend)

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
