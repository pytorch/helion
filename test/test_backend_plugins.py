from __future__ import annotations

import contextlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from typing import TYPE_CHECKING
from typing import Iterator
import unittest
from unittest import mock
import warnings
import weakref

import helion
from helion._compiler import backend_registry
from helion._compiler.backend import TritonBackend
from helion._compiler.backend_registry import _REGISTRY
from helion._compiler.backend_registry import get_backend_class
from helion._compiler.backend_registry import list_backends
from helion._testing import onlyBackends
import helion.language as hl

if TYPE_CHECKING:
    import types

_REPO_ROOT = Path(helion.__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "test" / "data" / "helion_dummy_backend"

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
            mock.patch.object(backend_registry, "_ROLLED_BACK", weakref.WeakSet()),
            mock.patch.object(backend_registry, "_FAILED_PLUGINS", {}),
            mock.patch.object(backend_registry, "_LOADED_PLUGINS", set()),
            mock.patch.object(backend_registry, "_LOADING_PLUGINS", {}),
            mock.patch.object(backend_registry, "_PLUGIN_LOAD_EVENTS", {}),
            mock.patch.object(backend_registry, "_WAITING_FOR", {}),
            # The suite must not depend on the caller's environment.
            mock.patch.dict(os.environ),
        ):
            os.environ.pop("HELION_DISABLE_BACKEND_PLUGINS", None)
            yield
    finally:
        # Change-only restore: never blank the registry (a concurrent
        # reader must not see it empty -- the invariant _restore_registry
        # exists to protect).
        for key in [k for k in _REGISTRY if k not in registry_before]:
            _REGISTRY.pop(key, None)
        for key, value in registry_before.items():
            if _REGISTRY.get(key) is not value:
                _REGISTRY[key] = value
        backend_registry._REPAIRED_CODEGEN_NAMES = repaired_before
        # Raw dict access: CodegenDict's own methods dispatch through repair.
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

    def test_clashing_config_key(self) -> None:
        src = _backend("_hplug_keys") + (
            "\n    @classmethod\n"
            "    def plugin_config_keys(cls):\n"
            "        return frozenset({'num_warps'})\n"
        )
        with _plugins(("_hplug_keys", "_hplug_keys", src, None)):
            self._assert_fails("_hplug_keys", "config keys that already exist")


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


def _wait_until_waiting(thread: threading.Thread, name: str) -> None:
    """Block until ``thread`` is waiting for another thread's load of ``name``."""
    deadline = time.monotonic() + 30
    while backend_registry._WAITING_FOR.get(thread.ident or -1) != name:
        if time.monotonic() > deadline or not thread.is_alive():
            raise AssertionError(f"{thread.name} never waited for {name!r}")
        time.sleep(0.01)


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

    def test_second_thread_waits_for_same_plugin_load(self) -> None:
        # A concurrent lookup of the same plugin must block on the loader and
        # then succeed, not return early with "Unknown backend".
        src = _WAITING_PLUGIN_SRC + _backend("_hplug_shared")
        with _plugins(
            ("_hplug_shared", "_hplug_shared", src, None),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            t1, r1 = _start_loading("_hplug_shared")
            try:
                _wait_for_signal()
                t2, r2 = _start_loading("_hplug_shared")
                _wait_until_waiting(t2, "_hplug_shared")
                self.assertEqual(r2, [])
                sys.modules["_hsigwait"].RELEASE.set()
                t1.join(30)
                t2.join(30)
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            self.assertEqual(
                [getattr(r, "__name__", r) for r in r1 + r2],
                ["Plugin", "Plugin"],
                (r1, r2),
            )

    def test_failing_load_does_not_poison_concurrent_load(self) -> None:
        # The poison's registry snapshot is taken before the survivor
        # registers (it polls for it), so the survivor's entry shows up as
        # "not in my snapshot" to the poison's rollback; it must be left
        # alone because its load owns it (the pop used to make the survivor
        # fail validation and get permanently marked failed).
        blocked_codegen = (
            "import _hsigwait\n\n_hsigwait.STARTED.set()\n"
            'assert _hsigwait.RELEASE.wait(30), "release timed out"\n'
        )
        poison = (
            "import time\n"
            "import helion._compiler.backend_registry as _br\n"
            "deadline = time.monotonic() + 30\n"
            "while '_hplug_survivor' not in _br._REGISTRY:\n"
            "    assert time.monotonic() < deadline, 'survivor never registered'\n"
            "    time.sleep(0.01)\n"
            "raise RuntimeError('boom')\n"
        )
        with _plugins(
            (
                "_hplug_survivor",
                "_hplug_survivor",
                _backend("_hplug_survivor"),
                blocked_codegen,
            ),
            ("_hplug_poison", "_hplug_poison", poison, None),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            t1, r1 = _start_loading("_hplug_survivor")
            t2, r2 = _start_loading("_hplug_poison")
            try:
                _wait_for_signal()  # survivor registered, mid codegen import
                t2.join(30)
                self.assertIsInstance(r2[0], ValueError)
                sys.modules["_hsigwait"].RELEASE.set()
                t1.join(30)
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
        self.assertEqual(getattr(r1[0], "__name__", r1[0]), "Plugin", r1)
        self.assertEqual(backend_registry._FAILED_PLUGINS, {})

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
            self.assertEqual(
                [str(w.message) for w in caught].count(
                    "cannot load a backend plugin while backend codegen "
                    "repair is in progress: handlers registered now would "
                    "be silently dropped by the repair filter; retry after "
                    "compilation"
                ),
                2,
            )
            self.assertEqual([k for k in _REGISTRY if k.startswith("_hplug_ra")], [])

    def test_load_during_codegen_repair_is_retryable(self) -> None:
        # Handler registrations made while a codegen repair is in progress
        # would be dropped by the repair filter; the load is refused without
        # poisoning the plugin, so a later lookup still succeeds.
        spec = ("_hplug_repair", "_hplug_repair", _backend("_hplug_repair"), None)
        with _plugins(spec):
            with _same_thread_repair():
                with self.assertRaisesRegex(RuntimeError, "repair is in progress"):
                    get_backend_class("_hplug_repair")
                self.assertEqual(backend_registry._FAILED_PLUGINS, {})
            self.assertEqual(get_backend_class("_hplug_repair").__name__, "Plugin")

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

    def test_cross_thread_lookup_cycle_does_not_hang(self) -> None:
        # p1's import looks up p2 while p2's import (on another thread) looks
        # up p1; waiting on each other would hang forever.
        def cyclic(name: str, other: str) -> str:
            return (
                "import _hsigcycle\n"
                "from helion._compiler.backend_registry import get_backend_class\n"
                "_hsigcycle.BARRIER.wait()\n"
                "try:\n"
                f"    get_backend_class({other!r})\n"
                "except ValueError:\n"
                "    pass\n" + _backend(name)
            )

        with _plugins(
            ("_hplug_cyc1", "_hplug_cyc1", cyclic("_hplug_cyc1", "_hplug_cyc2"), None),
            ("_hplug_cyc2", "_hplug_cyc2", cyclic("_hplug_cyc2", "_hplug_cyc1"), None),
            modules={
                "_hsigcycle": "import threading\nBARRIER = threading.Barrier(2, timeout=30)\n"
            },
        ):
            t1, r1 = _start_loading("_hplug_cyc1")
            t2, r2 = _start_loading("_hplug_cyc2")
            t1.join(30)
            t2.join(30)
            self.assertFalse(t1.is_alive() or t2.is_alive(), "lookup cycle hung")
            self.assertEqual(
                [getattr(r, "__name__", r) for r in r1 + r2],
                ["Plugin", "Plugin"],
                (r1, r2),
            )

    def test_other_thread_waits_for_half_loaded_plugin(self) -> None:
        # Registered but its codegen modules still importing: another thread
        # must wait for the outcome, not get the uncommitted class.
        codegen = (
            "import _hsigwait\n\n_hsigwait.STARTED.set()\n"
            'assert _hsigwait.RELEASE.wait(30), "release timed out"\n'
            "raise RuntimeError('codegen boom')\n"
        )
        with _plugins(
            ("_hplug_halfreg", "_hplug_halfreg", _backend("_hplug_halfreg"), codegen),
            modules={"_hsigwait": _WAIT_MODULE_SRC},
        ):
            t1, r1 = _start_loading("_hplug_halfreg")
            try:
                _wait_for_signal()
                self.assertIn("_hplug_halfreg", _REGISTRY)
                t2, r2 = _start_loading("_hplug_halfreg")
                _wait_until_waiting(t2, "_hplug_halfreg")
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

    def test_lookup_cycle_through_repair_lock_times_out(self) -> None:
        # A repair (holding _CODEGEN_REPAIR_LOCK) looks up a plugin another
        # thread is loading, and that load next needs the repair lock.
        src = _WAITING_PLUGIN_SRC + _backend("_hplug_rep7")

        def reload_looks_up(codegen_name: str) -> None:
            get_backend_class("_hplug_rep7")

        with (
            _plugins(
                ("_hplug_rep7", "_hplug_rep7", src, None),
                modules={"_hsigwait": _WAIT_MODULE_SRC},
            ),
            mock.patch.object(backend_registry, "_PLUGIN_WAIT_TIMEOUT_S", 0.5),
            mock.patch.object(
                backend_registry, "_reload_backend_codegen", reload_looks_up
            ),
        ):
            backend_registry._REPAIRED_CODEGEN_NAMES = frozenset()
            loader, loaded = _start_loading("_hplug_rep7")
            repaired: list[BaseException] = []

            def repair() -> None:
                try:
                    backend_registry.repair_backend_codegen("triton")
                except BaseException as e:
                    repaired.append(e)

            repairer = threading.Thread(target=repair, daemon=True)
            try:
                _wait_for_signal()
                repairer.start()
                _wait_until_waiting(repairer, "_hplug_rep7")
            finally:
                mod = sys.modules.get("_hsigwait")
                if mod is not None:
                    mod.RELEASE.set()
            repairer.join(30)
            loader.join(30)
            self.assertFalse(repairer.is_alive() or loader.is_alive(), "hung")
            self.assertEqual(len(repaired), 1, repaired)
            self.assertIsInstance(repaired[0], RuntimeError)
            self.assertRegex(
                str(repaired[0]),
                "'_hplug_rep7'.*likely a circular plugin import or lookup",
            )
            # The repair's deadline is earlier, so it always gives up; the
            # loader may time out too if the repair unwinds slowly.
            self.assertEqual(len(loaded), 1, loaded)
            if not isinstance(loaded[0], type):
                self.assertIsInstance(loaded[0], RuntimeError)
                self.assertRegex(str(loaded[0]), "timed out.*'_hplug_rep7'")
            self._assert_no_load_state()
            self.assertEqual(get_backend_class("_hplug_rep7").__name__, "Plugin")

    def test_lookup_cycle_through_import_lock_times_out(self) -> None:
        # p1's import looks up p2 while p2's import does "import p1.backend":
        # p2 is stuck on p1's import lock, which the cycle check cannot see.
        p1 = (
            "import _hsigic\n"
            "_hsigic.P1_IN.set()\n"
            "assert _hsigic.GO.wait(30)\n"
            "from helion._compiler.backend_registry import get_backend_class\n"
            "try:\n"
            "    get_backend_class('_hplug_ic2')\n"
            "except ValueError:\n"
            "    pass\n" + _backend("_hplug_ic1")
        )
        p2 = (
            "import _hsigic\n"
            "_hsigic.P2_IN.set()\n"
            "import _hplug_ic1.backend\n" + _backend("_hplug_ic2")
        )
        events = "import threading\nP1_IN, P2_IN, GO = (threading.Event() for _ in range(3))\n"
        with (
            _plugins(
                ("_hplug_ic1", "_hplug_ic1", p1, None),
                ("_hplug_ic2", "_hplug_ic2", p2, None),
                modules={"_hsigic": events},
            ),
            mock.patch.object(backend_registry, "_PLUGIN_WAIT_TIMEOUT_S", 0.5),
        ):
            t1, r1 = _start_loading("_hplug_ic1")
            try:
                sig = _signal_module("_hsigic")
                self.assertTrue(sig.P1_IN.wait(30))
                t2, r2 = _start_loading("_hplug_ic2")
                self.assertTrue(sig.P2_IN.wait(30))
            finally:
                mod = sys.modules.get("_hsigic")
                if mod is not None:
                    mod.GO.set()
            t1.join(30)
            t2.join(30)
            self.assertFalse(t1.is_alive() or t2.is_alive(), "hung")
            self.assertEqual(len(r1), 1, r1)
            self.assertIsInstance(r1[0], RuntimeError)
            self.assertRegex(
                str(r1[0]), "'_hplug_ic2'.*likely a circular plugin import or lookup"
            )
            self.assertEqual([getattr(r, "__name__", r) for r in r2], ["Plugin"])
            self._assert_no_load_state()
            # The timeout is not remembered: a later lookup retries.
            self.assertEqual(get_backend_class("_hplug_ic1").__name__, "Plugin")

    def _assert_no_load_state(self) -> None:
        self.assertEqual(backend_registry._LOADING_PLUGINS, {})
        self.assertEqual(backend_registry._PLUGIN_LOAD_EVENTS, {})
        self.assertEqual(backend_registry._WAITING_FOR, {})
        self.assertEqual(backend_registry._FAILED_PLUGINS, {})

    def test_rollback_keeps_later_registration_over_dead_class(self) -> None:
        # X snapshots Y's uncommitted class; Y fails, a user registers its
        # name; X's rollback must not restore Y's dead class over the user's.
        events = "import threading\nY_IN, Y_GO, X_IN, X_GO = (threading.Event() for _ in range(4))\n"
        y_codegen = (
            "import _hsigrr\n_hsigrr.Y_IN.set()\n"
            "assert _hsigrr.Y_GO.wait(30)\nraise RuntimeError('y boom')\n"
        )
        x_src = (
            "import _hsigrr\n_hsigrr.X_IN.set()\n"
            "assert _hsigrr.X_GO.wait(30)\nraise RuntimeError('x boom')\n"
        )

        class UserBackend(TritonBackend):
            @property
            def name(self) -> str:
                return "_hplug_rry"

        with _plugins(
            ("_hplug_rry", "_hplug_rry", _backend("_hplug_rry"), y_codegen),
            ("_hplug_rrx", "_hplug_rrx", x_src, None),
            modules={"_hsigrr": events},
        ):
            ty, ry = _start_loading("_hplug_rry")
            tx: threading.Thread | None = None
            try:
                sig = _signal_module("_hsigrr")
                self.assertTrue(sig.Y_IN.wait(30))
                tx, rx = _start_loading("_hplug_rrx")
                self.assertTrue(sig.X_IN.wait(30))  # X has taken its snapshot
                sig.Y_GO.set()
                ty.join(30)
                self.assertIsInstance(ry[0], ValueError)
                backend_registry.register_compiler_backend(UserBackend)
            finally:
                mod = sys.modules.get("_hsigrr")
                if mod is not None:
                    mod.Y_GO.set()
                    mod.X_GO.set()
            assert tx is not None
            tx.join(30)
            self.assertIsInstance(rx[0], ValueError)
            self.assertIs(_REGISTRY.get("_hplug_rry"), UserBackend)


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


def _run_with_fixture(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    full_env.pop("HELION_DISABLE_BACKEND_PLUGINS", None)
    full_env["PYTHONPATH"] = os.pathsep.join(
        [str(_FIXTURE), str(_REPO_ROOT), full_env.get("PYTHONPATH", "")]
    )
    full_env.update(env)
    # A real file, since Helion reads kernel source with inspect.getsource.
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(textwrap.dedent(code))
        return subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            env=full_env,
            timeout=300,
        )


class TestInstalledPlugin(unittest.TestCase):
    # The fixture is a package plus a .dist-info on PYTHONPATH, which
    # importlib.metadata discovers like a pip-installed plugin.
    def test_not_imported_until_requested(self) -> None:
        result = _run_with_fixture(
            """
            import sys
            import helion
            from helion._compiler.backend_registry import list_backends

            assert "triton_oot" in list_backends(), list_backends()
            assert "helion_dummy_backend.backend" not in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plugin_imported_before_helion(self) -> None:
        result = _run_with_fixture(
            """
            import helion_dummy_backend.backend
            import helion
            from helion._compiler.backend_registry import get_backend_class

            assert get_backend_class("triton_oot").__name__ == "TritonOOTBackend"
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_by_env(self) -> None:
        result = _run_with_fixture(
            """
            import helion
            assert helion.Settings().backend == "triton_oot"
            """,
            HELION_BACKEND="triton_oot",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_spawned_worker_loads_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "pids.log"
            result = _run_with_fixture(
                """
                import multiprocessing
                import os
                from helion._compiler.backend_registry import get_backend_class

                if __name__ == "__main__":
                    with multiprocessing.get_context("spawn").Pool(1) as pool:
                        cls = pool.apply(get_backend_class, ("triton_oot",))
                    assert cls.__name__ == "TritonOOTBackend"
                    print(os.getpid())
                """,
                HELION_DUMMY_PLUGIN_LOG=str(log),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            parent = result.stdout.strip()
            importers = set(log.read_text().split())
            # The worker imported the plugin itself, not just the parent.
            self.assertTrue(importers - {parent}, (importers, parent))


@onlyBackends(["triton"])
class TestInstalledPluginCompile(unittest.TestCase):
    def test_compiles_and_runs(self) -> None:
        result = _run_with_fixture(
            """
            import torch
            import helion
            import helion.language as hl
            from helion._testing import DEVICE

            @helion.kernel(backend="triton_oot")
            def add(x, y):
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            x = torch.randn(64, 64, device=DEVICE)
            bound = add.bind((x, x))
            out = bound.compile_config(bound.config_spec.default_config())(x, x)
            torch.testing.assert_close(out, x + x)
            """,
            HELION_BACKEND="triton",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
