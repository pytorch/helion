"""Centralized registry for Helion codegen backends.

All backend lookup and instantiation should go through this module.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
import re
import sys
import threading
import types
from typing import TYPE_CHECKING
import warnings

from .backend import Backend
from .cute.backend import CuteBackend
from .metal.backend import MetalBackend
from .pallas.backend import PallasBackend
from .triton.backend import TileIRBackend
from .triton.backend import TritonBackend

if TYPE_CHECKING:
    from collections.abc import Iterator
    from importlib.metadata import EntryPoint

    from ..autotuner.search_surface import BackendSearchSurface

# Version of the out-of-tree backend contract in ``helion.backend_api``; covers
# only the names and hooks listed in docs/api/backend_api.md, bumped per the
# version policy there.
BACKEND_API_VERSION = 1

# Out-of-tree backends register an entry point in this group, named after the
# backend, whose value is a ``Backend`` subclass, e.g.
# ``mybackend = "my_pkg.backend:MyBackend"``.  A plugin is imported the first
# time its backend is looked up by name.  A name with more than one provider is
# an error.
PLUGIN_ENTRY_POINT_GROUP = "helion.backends"
_PLUGIN_ENTRY_POINTS: dict[str, EntryPoint] | None = None
_FAILED_PLUGINS: dict[str, str] = {}
# Guards discovery; never held during an import.
_DISCOVERY_LOCK = threading.Lock()
# Held for a plugin's whole load (import, registration, codegen modules), so a
# thread looking the plugin up waits for the finished load.  Nothing that runs
# during an import takes it: register_compiler_backend and @codegen use only
# _CODEGEN_REPAIR_LOCK, which is always taken after this lock, never before.
_PLUGIN_LOAD_LOCK = threading.RLock()
# name -> id of the thread loading it, set under _PLUGIN_LOAD_LOCK.  Lookups of
# a name in here go through the lock; the loading thread itself sees whatever
# is registered so far.
_LOADING_PLUGINS: dict[str, int] = {}
# Plugins that finished loading; an enclosing plugin's rollback keeps them.
_LOADED_PLUGINS: set[str] = set()
# Bound on waiting for another thread's load.  Reached only through a cycle
# with an import lock (a plugin module that looks up its own backend while
# being imported directly on another thread); raises instead of hanging.
_PLUGIN_LOAD_TIMEOUT_S = 60.0

# Backends that moved out of tree: name -> pip package providing it. An entry
# only improves the "unknown backend" error; remove it once users have had
# time to migrate.
_KNOWN_EXTERNAL_BACKENDS: dict[str, str] = {"flydsl": "helion-flydsl"}

_BUILTIN_BACKENDS: list[type[Backend]] = [
    TritonBackend,
    PallasBackend,
    CuteBackend,
    TileIRBackend,
    MetalBackend,
]

_REGISTRY: dict[str, type[Backend]] = {}
# Bumped by every registration and unregistration: the tables derived from
# the registry (``config_spec``'s key tables, ``autotuner_heuristics``'s
# heuristics by backend) key their caches on it, so a backend registered
# after those modules imported is seen on the next lookup.
_GENERATION = 0
_CODEGEN_REPAIR_LOCK = threading.RLock()
# Per-thread count of _repair_lock() holds, so a plugin load can tell that this
# thread holds the repair lock (loading then would invert the lock order).


class _RepairLockHeld(threading.local):
    depth = 0


_REPAIR_LOCK_HELD = _RepairLockHeld()
_REPAIRED_CODEGEN_NAMES: frozenset[str] = frozenset()


@contextlib.contextmanager
def _repair_lock() -> Iterator[None]:
    """``_CODEGEN_REPAIR_LOCK``, counted in ``_REPAIR_LOCK_HELD`` while held."""
    with _CODEGEN_REPAIR_LOCK:
        _REPAIR_LOCK_HELD.depth += 1
        try:
            yield
        finally:
            _REPAIR_LOCK_HELD.depth -= 1


def register_compiler_backend(backend_class: type[Backend]) -> None:
    """Register a compiler backend.

    The backend's ``name`` property is used as the registry key.
    Built-in backends are registered at module load time below.

    Registering the same class twice is a no-op, and the class a reloaded
    module defines under the same name replaces the old one.  Registering a
    *different* class under an already-registered name (including built-in
    names) raises ``ValueError``; ``helion.backend_api`` has no unregister, so
    a name is owned by its first registering class.

    Args:
        backend_class: A :class:`Backend` subclass.

    Raises:
        ValueError: if a non-built-in backend's search surface declares a
            generic config key.
    """
    global _GENERATION, _REPAIRED_CODEGEN_NAMES

    backend = backend_class()
    if backend_class not in _BUILTIN_BACKENDS:
        _check_surface_keys(backend)  # calls plugin code: not under the lock
    with _repair_lock():
        existing = _REGISTRY.get(backend.name)
        if existing is backend_class:
            return
        if existing is not None and not _is_reloaded(existing, backend_class):
            raise ValueError(
                f"Backend name {backend.name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}; "
                f"refusing to overwrite it with "
                f"{backend_class.__module__}.{backend_class.__qualname__}"
            )
        _REGISTRY[backend.name] = backend_class
        _GENERATION += 1
        codegen_names, _ = _codegen_repair_scope(backend.codegen_name)
        _REPAIRED_CODEGEN_NAMES = _REPAIRED_CODEGEN_NAMES.difference(codegen_names)


def _is_reloaded(existing: type[Backend], backend_class: type[Backend]) -> bool:
    # importlib.reload re-runs the module and defines a new class object under
    # the same name; the module then holds the new class.
    name = backend_class.__qualname__
    return (existing.__module__, existing.__qualname__) == (
        backend_class.__module__,
        name,
    ) and getattr(
        sys.modules.get(backend_class.__module__), name, None
    ) is backend_class


def _check_surface_keys(backend: Backend) -> None:
    # Not run for built-ins: they register while config_spec is importing.
    surface = backend.config_spec_extension()
    from ..autotuner.config_spec import GENERIC_KEYS

    # A surface key becomes backend-specific, so a generic one would be
    # rejected by every other backend from then on.
    clashes = (surface.config_keys | surface.debug_keys) & GENERIC_KEYS
    if clashes:
        raise ValueError(
            f"Backend {backend.name!r} declares generic config keys on its "
            f"search surface: {sorted(clashes)}"
        )


def unregister_compiler_backend(name: str) -> None:
    """Remove the backend registered as ``name`` (a no-op for unknown names).

    For tests that register a throwaway backend: the derived tables forget
    it on their next lookup.
    """
    global _GENERATION

    with _repair_lock():
        if _REGISTRY.pop(name, None) is not None:
            _GENERATION += 1


def registry_generation() -> int:
    """A counter bumped by every (un)registration, for caches derived from
    the registry."""
    return _GENERATION


def get_backend_class(name: str) -> type[Backend]:
    """Look up a backend class by name, loading its plugin on first use.

    A plugin another thread is loading is returned only once that load has
    finished.  (Readers of ``_REGISTRY`` itself, e.g. ``registered_backends``,
    can see a plugin that registered itself during its import.)
    """
    if name not in _REGISTRY or name in _LOADING_PLUGINS:
        _load_plugin_by_name(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend: {name!r}. Available backends: {list_backends()}"
            f"{install_hint(name)}"
        )
    return _REGISTRY[name]


def get_backend(name: str) -> Backend:
    """A fresh instance of the backend registered as ``name``.

    Callers that only read a hook (a capability predicate, the heuristics
    tuple, the structural policy type) get their own instance: backends are
    cheap to construct and several (Metal, Pallas, out-of-tree ones) keep per-compile
    state on the instance, so instances are never shared.
    """
    return get_backend_class(name)()


def install_hint(name: str) -> str:
    """Suffix for "unknown backend" errors when ``name`` moved out of tree."""
    package = _KNOWN_EXTERNAL_BACKENDS.get(name)
    if package is None:
        return ""
    if not _entry_point_installed(name):
        return f". The {name!r} backend is provided by `pip install {package}`"
    if _parse_plugins_disabled() is True:
        return (
            f". The {name!r} plugin is ignored because "
            "HELION_DISABLE_BACKEND_PLUGINS is set"
        )
    cached = _PLUGIN_ENTRY_POINTS
    if cached is not None and name not in cached:
        # Discovery ran while plugins were disabled (or before the install).
        return (
            f". The {name!r} plugin is installed but was not discovered when "
            "plugins were first looked up; restart the process to load it"
        )
    return ""


def _entry_point_installed(name: str) -> bool:
    # A fresh look, not the cached discovery, so the hint reflects now.
    entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    return any(ep.name == name for ep in entry_points)


def list_backends() -> list[str]:
    """Return registered backends plus installed, not-yet-loaded plugins."""
    names = list(_REGISTRY)
    names.extend(n for n in _plugin_entry_points() if n not in _REGISTRY)
    return names


def registered_backends() -> list[str]:
    """Names of the backends registered so far, without loading plugins.

    For tables derived from the registry: they refresh on
    ``registry_generation()`` when a plugin loads later.
    """
    return list(_REGISTRY)


def all_reserved_launch_param_names() -> frozenset[str]:
    """Union of reserved launch param names across all registered backends.

    Reserving all names ensures kernel portability. A variable name
    that collides with any backend's launch params is avoided regardless
    of which backend is currently active.
    """
    result: set[str] = set()
    for backend_cls in list(_REGISTRY.values()):
        result.update(backend_cls.reserved_launch_param_names())
    return frozenset(result)


def all_search_surfaces() -> list[BackendSearchSurface]:
    """The distinct search surfaces of the registered backends.

    ``helion.autotuner.config_spec`` derives ``BACKEND_SPECIFIC_KEYS`` /
    ``VALID_KEYS`` / ``DEBUG_KEYS`` from them; backends that share one surface
    object (the no-op default) contribute it once.
    """
    surfaces: list[BackendSearchSurface] = []
    for backend_cls in _REGISTRY.values():
        surface = backend_cls().config_spec_extension()
        if not any(surface is seen for seen in surfaces):
            surfaces.append(surface)
    return surfaces


def import_backend_codegen() -> None:
    """Import every registered backend's per-op codegen modules.

    Each backend lists its own codegen modules in
    ``helion/_compiler/<backend>/_codegen_modules.py``.  Importing that module
    runs the backend's ``@_decorators.codegen(op, "<backend>")`` /
    ``register_codegen("<backend>")`` handlers, wiring them onto the op and
    aten-lowering objects they extend.

    This is called once from ``helion.language`` after all language ops are
    defined (so the eager registration timing matches the old per-file bottom
    imports).  Because it is driven by the registry, adding a backend requires
    no edits to the core ``helion/language`` files -- only registering the
    backend class (below) and adding its ``_codegen_modules`` module.
    """
    seen: set[str] = set()
    for backend_cls in list(_REGISTRY.values()):
        # e.g. "helion._compiler.cute.backend" -> "helion._compiler.cute".
        # Subclasses that share a folder (e.g. TileIRBackend in the triton
        # package) collapse to one import.
        package = backend_cls.__module__.rsplit(".", 1)[0]
        if package in seen:
            continue
        seen.add(package)
        _import_codegen_modules(backend_cls)


def _import_codegen_modules(backend_cls: type[Backend]) -> None:
    module = f"{backend_cls.__module__.rsplit('.', 1)[0]}._codegen_modules"
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:
        # A backend without any per-op codegen modules is allowed; only
        # swallow the missing _codegen_modules module itself, never a
        # broken import inside it.
        if e.name != module:
            raise


def load_backend_plugins() -> None:
    """Load every installed backend plugin now, warning about any that fail.

    Plugins otherwise load on first lookup by name (:func:`get_backend_class`).
    """
    for name in list(_plugin_entry_points()):
        try:
            _load_plugin_by_name(name)
        except (ValueError, RuntimeError) as e:
            # A RuntimeError (refusal, timeout) is not remembered: the lookup
            # on first use retries.
            warnings.warn(str(e), stacklevel=2)


def _plugins_disabled() -> bool:
    from ..runtime.settings import _parse_env_bool  # settings imports this module

    var = "HELION_DISABLE_BACKEND_PLUGINS"
    return _parse_env_bool(var, os.environ.get(var), False)


def _parse_plugins_disabled() -> bool | None:
    """HELION_DISABLE_BACKEND_PLUGINS as a bool, or None if invalid."""
    try:
        return _plugins_disabled()
    except ValueError:
        return None


def _plugin_entry_points() -> dict[str, EntryPoint]:
    """Installed ``helion.backends`` entry points by name (discovered once)."""
    global _PLUGIN_ENTRY_POINTS

    with _DISCOVERY_LOCK:
        if _PLUGIN_ENTRY_POINTS is None:
            found = _discover_plugin_entry_points()
            for name, eps in found.items():
                if len(eps) > 1:
                    # Remembered like a load failure; neither plugin is imported.
                    _FAILED_PLUGINS[name] = (
                        f"Helion backend {name!r} is provided by more than one "
                        "installed package: "
                        f"{', '.join(repr(_dist_name(ep)) for ep in eps)}; "
                        "uninstall all but one"
                    )
            _PLUGIN_ENTRY_POINTS = {name: eps[0] for name, eps in found.items()}
    return _PLUGIN_ENTRY_POINTS


def _discover_plugin_entry_points() -> dict[str, list[EntryPoint]]:
    found: dict[str, list[EntryPoint]] = {}
    if not _plugins_disabled():
        builtins = {cls().name for cls in _BUILTIN_BACKENDS}
        entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        for ep in sorted(entry_points, key=lambda ep: (_dist_name(ep), ep.name)):
            if ep.name in builtins:
                warnings.warn(
                    f"Ignoring {_describe(ep)}: {ep.name!r} is a built-in backend",
                    stacklevel=1,  # call depth to user code varies
                )
            else:
                found.setdefault(ep.name, []).append(ep)
    return found


def _dist_name(ep: EntryPoint) -> str:
    """Normalized distribution name providing ``ep`` (PEP 503)."""
    dist = ep.dist  # None for hand-built EntryPoints
    if dist is None or not dist.name:
        return "<unknown>"
    # PEP 503 normalization, so names compare ignoring case and -/_/. differences.
    return re.sub(r"[-_.]+", "-", dist.name).lower()


def _describe(ep: EntryPoint) -> str:
    return f"Helion backend plugin {ep.name!r} ({ep.value}) from {_dist_name(ep)!r}"


def _load_plugin_by_name(name: str) -> None:
    ep = _plugin_entry_points().get(name)
    if ep is None or _LOADING_PLUGINS.get(name) == threading.get_ident():
        return  # not a plugin, or looked up while it imports: see the registry
    from ..language._decorators import _codegen_repair_in_progress

    if _codegen_repair_in_progress() or _REPAIR_LOCK_HELD.depth:
        # Handlers registered during a repair are dropped by its filter, and
        # taking the load lock under the repair lock would invert the order.
        raise RuntimeError(
            f"cannot load Helion backend plugin {name!r} during a codegen "
            "repair or backend registration on this thread; retry after it "
            "finishes"
        )
    if not _PLUGIN_LOAD_LOCK.acquire(timeout=_PLUGIN_LOAD_TIMEOUT_S):
        raise RuntimeError(
            f"timed out after {_PLUGIN_LOAD_TIMEOUT_S:g}s waiting to load Helion "
            f"backend plugin {name!r}; likely a plugin module that looks up its "
            "own backend while it is imported on another thread"
        )
    try:
        if name in _REGISTRY:
            return  # loaded meanwhile, or registered by hand (which wins)
        if name in _FAILED_PLUGINS:
            raise ValueError(_FAILED_PLUGINS[name])
        with _repair_lock():
            pass  # let a codegen repair on another thread finish first
        _LOADING_PLUGINS[name] = threading.get_ident()
        try:
            _load_backend_plugin(ep)
        except Exception as e:
            message = f"{_describe(ep)} failed to load: {type(e).__name__}: {e}"
            # Remembered for the process lifetime: a fresh process (or a fixed
            # package plus restart) is required to retry.
            _FAILED_PLUGINS[name] = message
            raise ValueError(message) from e
        finally:
            _LOADING_PLUGINS.pop(name, None)  # after any rollback
    finally:
        _PLUGIN_LOAD_LOCK.release()


def _load_backend_plugin(ep: EntryPoint) -> None:
    """Import, validate and register one plugin; undo its registrations on
    failure.  The caller holds ``_PLUGIN_LOAD_LOCK``.

    Rollback removes the registry entries this load added (the entry point's
    name, and other new names whose class lives in the entry point's or
    the backend class's package, unless that plugin finished loading) and the ``@codegen``
    handlers the plugin's package registered on this thread.  Other import
    side effects (threads started, files written, modules cached in
    ``sys.modules``) are not undone.
    """
    from ..language._decorators import _forget_recorded_registrations
    from ..language._decorators import _record_codegen_registrations

    registry_before = dict(_REGISTRY)
    # Packages whose registrations a failed load undoes: the entry point's, and
    # the backend class's (where its _codegen_modules live).
    roots = [ep.module.rpartition(".")[0] or ep.module]
    with _record_codegen_registrations() as recorded:
        try:
            backend_cls = ep.load()
            if not (isinstance(backend_cls, type) and issubclass(backend_cls, Backend)):
                raise TypeError("entry point is not a Backend subclass")
            cls_package = backend_cls.__module__.rpartition(".")[0]
            if not _in_package(cls_package, ["helion"]):  # never Helion's own
                roots.append(cls_package)
            version = getattr(backend_cls, "HELION_BACKEND_API_VERSION", None)
            if type(version) is not int or version != BACKEND_API_VERSION:
                raise RuntimeError(
                    f"it declares HELION_BACKEND_API_VERSION={version!r}, but this "
                    f"Helion provides {BACKEND_API_VERSION}"
                )
            name = backend_cls().name
            if name != ep.name:
                raise RuntimeError(
                    f"its backend is named {name!r}; the entry point must use "
                    "the backend name"
                )
            # A no-op if the plugin registered itself; raises if another class
            # holds the name.
            register_compiler_backend(backend_cls)
            _import_codegen_modules(backend_cls)
        except BaseException as e:
            _rollback_plugin_load(e, ep.name, registry_before, recorded, roots)
            raise
        _LOADED_PLUGINS.add(ep.name)
        # A plugin loading this one must not undo it if that plugin fails.
        _forget_recorded_registrations(recorded)


def _rollback_plugin_load(
    error: BaseException,
    name: str,
    registry_before: dict[str, type[Backend]],
    recorded: list[tuple[dict[str, object], str, object, object]],
    roots: list[str],
) -> None:
    """Undo a failed load; a rollback failure becomes a note on ``error``."""
    global _GENERATION

    try:
        with _repair_lock():
            added = [
                key
                for key, cls in _REGISTRY.items()
                if key not in registry_before
                and key not in _LOADED_PLUGINS
                and (key == name or _in_package(cls.__module__, roots))
            ]
            for key in added:
                del _REGISTRY[key]
            if added:
                _GENERATION += 1  # the tables derived from the registry rebuild
    except Exception as rollback_error:
        _add_note(error, f"registry rollback also failed: {rollback_error!r}")
    try:
        _undo_plugin_handlers(recorded, roots)
    except Exception as rollback_error:
        _add_note(error, f"@codegen handler rollback also failed: {rollback_error!r}")


def _add_note(error: BaseException, note: str) -> None:
    if sys.version_info >= (3, 11):
        error.add_note(note)
    else:
        # pyrefly: ignore[missing-attribute]
        error.__notes__ = [*getattr(error, "__notes__", ()), note]


def _in_package(module: str, packages: list[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in packages if p)


def _undo_plugin_handlers(
    recorded: list[tuple[dict[str, object], str, object, object]], roots: list[str]
) -> None:
    from ..language._decorators import _NO_PREVIOUS_HANDLER

    for implementations, backend, previous, handler in reversed(recorded):
        # Only handlers defined by the plugin's own package; a plugin may
        # trigger lazy imports of Helion modules that register built-in ones.
        module = getattr(handler, "__module__", None) or ""
        if not _in_package(module, roots):
            continue
        # Raw dict access, deliberately: ``implementations`` is usually a
        # BackendDispatchTable whose getitem/get dispatch through
        # repair_backend_codegen, which takes _CODEGEN_REPAIR_LOCK -- and
        # rollback must never re-enter repair (possibly while holding its
        # lock) just to restore a mapping.
        if dict.get(implementations, backend) is not handler:
            continue
        if previous is _NO_PREVIOUS_HANDLER:
            dict.pop(implementations, backend, None)
        else:
            dict.__setitem__(implementations, backend, previous)


def _codegen_repair_scope(
    codegen_name: str,
) -> tuple[frozenset[str], frozenset[str]]:
    entries = [
        (
            backend_class().codegen_name,
            backend_class.__module__.rsplit(".", 1)[0],
        )
        for backend_class in list(_REGISTRY.values())
    ]
    codegen_names = {codegen_name}
    packages: set[str] = set()
    while True:
        next_packages = packages.union(
            package for name, package in entries if name in codegen_names
        )
        next_codegen_names = codegen_names.union(
            name for name, package in entries if package in next_packages
        )
        if next_packages == packages and next_codegen_names == codegen_names:
            return frozenset(codegen_names), frozenset(packages)
        packages = next_packages
        codegen_names = next_codegen_names


def _reload_backend_codegen(codegen_name: str) -> None:
    if not _REGISTRY:
        for _cls in _BUILTIN_BACKENDS:
            register_compiler_backend(_cls)

    _, packages = _codegen_repair_scope(codegen_name)
    preexisting_modules = frozenset(sys.modules)
    seen_packages: set[str] = set()
    for name, backend_cls in list(_REGISTRY.items()):
        package = backend_cls.__module__.rsplit(".", 1)[0]
        if package not in packages or package in seen_packages:
            continue
        if name in _LOADING_PLUGINS:
            # Mid-load: its loader imports these modules itself, and may hold
            # their import locks while waiting for this repair's lock.
            continue
        seen_packages.add(package)
        module_name = f"{package}._codegen_modules"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name != module_name:
                raise
            continue

        # Reload the registry module first so all of its leaf modules are bound,
        # then rerun each leaf module's registration decorators.
        if module_name in preexisting_modules:
            importlib.reload(module)
        reloaded_modules: set[str] = set()
        for value in list(vars(module).values()):
            if (
                isinstance(value, types.ModuleType)
                and value.__name__.startswith(package + ".")
                and value.__name__ in preexisting_modules
                and value.__name__ not in reloaded_modules
            ):
                reloaded_modules.add(value.__name__)
                importlib.reload(value)


def repair_backend_codegen(codegen_name: str) -> None:
    """Force-complete backend codegen registrations skipped due to partial import.

    :func:`import_backend_codegen` relies on ``importlib.import_module`` to run
    each backend's ``@_decorators.codegen`` / ``register_codegen`` handlers. When
    a codegen module is already present in ``sys.modules`` in a partially
    initialized state -- a circular import during package init cached it before
    its module-scope registrations ran -- ``import_module`` returns the
    incomplete module and the registrations are silently skipped, leaving
    ``APIFunc._codegen`` empty for that backend. That surfaces later as
    :class:`~helion.exc.BackendImplementationMissing` at codegen time (notably
    under ``torch.compile`` in a packaged runtime, where the init import order
    differs).

    This reloads the requested backend's codegen leaf modules so their
    registrations run. Repairs are serialized across compilation threads and a
    backend is marked repaired only after a successful reload, allowing a failed
    attempt to be retried. Every backend dispatch crosses this barrier before
    reading a handler, so no compiler thread can use a handler while its module
    is being reloaded.
    """
    global _REPAIRED_CODEGEN_NAMES

    if codegen_name in _REPAIRED_CODEGEN_NAMES:
        return
    with _repair_lock():
        from ..language._decorators import _begin_codegen_repair
        from ..language._decorators import _codegen_repair_in_progress
        from ..language._decorators import _end_codegen_repair

        if codegen_name in _REPAIRED_CODEGEN_NAMES:
            return
        if _codegen_repair_in_progress():
            return
        if not _REGISTRY:
            for _cls in _BUILTIN_BACKENDS:
                register_compiler_backend(_cls)
        codegen_names, _ = _codegen_repair_scope(codegen_name)
        _begin_codegen_repair(codegen_names)
        try:
            _reload_backend_codegen(codegen_name)
            _REPAIRED_CODEGEN_NAMES = _REPAIRED_CODEGEN_NAMES.union(codegen_names)
        finally:
            _end_codegen_repair()


# register built-in backends
for _cls in _BUILTIN_BACKENDS:
    register_compiler_backend(_cls)
