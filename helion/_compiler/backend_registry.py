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
from typing import Iterator
import warnings
import weakref

from .backend import Backend
from .backend import CuteBackend
from .backend import MetalBackend
from .backend import PallasBackend
from .backend import TileIRBackend
from .backend import TritonBackend
from .flydsl.backend import FlyDSLBackend

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

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
# Short critical sections around plugin loading (never held during an
# import: a plugin module may import modules that a concurrent compile is
# mid-import on, and codegen dispatch takes _CODEGEN_REPAIR_LOCK, so holding
# the repair lock across plugin imports could deadlock).
_PLUGIN_LOAD_LOCK = threading.Lock()
# name -> thread id of the loading thread; a plugin looking itself up while
# it imports finds nothing yet instead of recursing or waiting on itself.
_LOADING_PLUGINS: dict[str, int] = {}
# name -> Event that waiters (other threads loading the same plugin) block on.
_PLUGIN_LOAD_EVENTS: dict[str, threading.Event] = {}
# thread id -> name of the plugin that thread is waiting for (cycle detection).
_WAITING_FOR: dict[int, str] = {}
# Plugins that finished loading; an enclosing plugin's rollback keeps them.
_LOADED_PLUGINS: set[str] = set()
# Per-thread stack of active plugin loads' register_compiler_backend records.
_BACKEND_RECORDING = threading.local()
# Classes removed by a failed load's rollback; never restored by another one.
_ROLLED_BACK: weakref.WeakSet[type[Backend]] = weakref.WeakSet()
# Bound on every wait a plugin load makes on another thread (load event,
# _CODEGEN_REPAIR_LOCK): a cycle through a lock the cycle check cannot see
# (an import lock, the repair lock) raises instead of hanging.
_PLUGIN_WAIT_TIMEOUT_S = 60.0

_BUILTIN_BACKENDS: list[type[Backend]] = [
    TritonBackend,
    PallasBackend,
    CuteBackend,
    TileIRBackend,
    MetalBackend,
    FlyDSLBackend,
]

_REGISTRY: dict[str, type[Backend]] = {}
_CODEGEN_REPAIR_LOCK = threading.RLock()
_REPAIRED_CODEGEN_NAMES: frozenset[str] = frozenset()


class _RetryablePluginLoadError(RuntimeError):
    """A transient refusal (a codegen repair in progress on this thread, or a
    wait on another thread that timed out); not memoized in
    ``_FAILED_PLUGINS`` so a later lookup can retry."""


def _plugin_wait_timeout(name: str, what: str) -> _RetryablePluginLoadError:
    return _RetryablePluginLoadError(
        f"timed out after {_PLUGIN_WAIT_TIMEOUT_S:g}s waiting for {what} while "
        f"loading Helion backend plugin {name!r}; likely a circular plugin "
        "import or lookup, or a long codegen repair on another thread"
    )


@contextlib.contextmanager
def _plugin_repair_lock(name: str) -> Iterator[None]:
    """``_CODEGEN_REPAIR_LOCK``, acquired with the plugin wait timeout."""
    if not _CODEGEN_REPAIR_LOCK.acquire(timeout=_PLUGIN_WAIT_TIMEOUT_S):
        raise _plugin_wait_timeout(name, "the codegen repair lock")
    try:
        yield
    finally:
        _CODEGEN_REPAIR_LOCK.release()


def register_compiler_backend(backend_class: type[Backend]) -> None:
    """Register a compiler backend.

    The backend's ``name`` property is used as the registry key.
    Built-in backends are registered at module load time below.

    Registering the same class twice is a no-op.  Registering a *different*
    class under an already-registered name (including built-in names) raises
    ``ValueError``; there is no public unregister, so a name is owned by its
    first registering class for the lifetime of the process.

    Args:
        backend_class: A :class:`Backend` subclass.
    """
    global _REPAIRED_CODEGEN_NAMES

    backend = backend_class()
    in_plugin_load = bool(getattr(_BACKEND_RECORDING, "stack", None))
    with _plugin_repair_lock(backend.name) if in_plugin_load else _CODEGEN_REPAIR_LOCK:
        existing = _REGISTRY.get(backend.name)
        if existing is backend_class:
            return
        if existing is not None:
            raise ValueError(
                f"Backend name {backend.name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}; "
                f"refusing to overwrite it with "
                f"{backend_class.__module__}.{backend_class.__qualname__}"
            )
        for recorded in getattr(_BACKEND_RECORDING, "stack", None) or []:
            recorded.append((backend.name, backend_class))
        _REGISTRY[backend.name] = backend_class
        codegen_names, _ = _codegen_repair_scope(backend.codegen_name)
        _REPAIRED_CODEGEN_NAMES = _REPAIRED_CODEGEN_NAMES.difference(codegen_names)


def get_backend_class(name: str) -> type[Backend]:
    """Look up a backend class by name, loading its plugin on first use.

    A plugin another thread is still loading is returned only once that load
    has committed.
    """
    if name not in _REGISTRY or name in _LOADING_PLUGINS:
        _load_plugin_by_name(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend: {name!r}. Available backends: {list_backends()}"
        )
    return _REGISTRY[name]


def list_backends() -> list[str]:
    """Return registered backends plus installed, not-yet-loaded plugins."""
    names = list(_REGISTRY)
    names.extend(n for n in _plugin_entry_points() if n not in _REGISTRY)
    return names


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
        except (ValueError, _RetryablePluginLoadError) as e:
            # Retryable refusals warn too; the lookup on first use retries.
            warnings.warn(str(e), stacklevel=2)


def _plugins_disabled() -> bool:
    from ..runtime.settings import _parse_env_bool  # settings imports this module

    var = "HELION_DISABLE_BACKEND_PLUGINS"
    return _parse_env_bool(var, os.environ.get(var), False)


def _plugin_entry_points() -> dict[str, EntryPoint]:
    """Installed ``helion.backends`` entry points by name (discovered once)."""
    global _PLUGIN_ENTRY_POINTS

    with _PLUGIN_LOAD_LOCK:
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
    dist = getattr(ep, "dist", None)  # None for hand-built EntryPoints
    if dist is None or not dist.name:
        return "<unknown>"
    return _normalize_dist_name(dist.name)


def _normalize_dist_name(name: str) -> str:
    # PEP 503 normalization, so names compare ignoring case and -/_/. differences.
    return re.sub(r"[-_.]+", "-", name).lower()


def _describe(ep: EntryPoint) -> str:
    return f"Helion backend plugin {ep.name!r} ({ep.value}) from {_dist_name(ep)!r}"


def _load_plugin_by_name(name: str) -> None:
    ep = _plugin_entry_points().get(name)
    if ep is None:
        return
    me = threading.get_ident()
    while True:
        with _PLUGIN_LOAD_LOCK:
            owner = _LOADING_PLUGINS.get(name)
            if owner is None:
                if name in _REGISTRY:
                    return
                if name in _FAILED_PLUGINS:
                    raise ValueError(_FAILED_PLUGINS[name])
                _LOADING_PLUGINS[name] = me
                _PLUGIN_LOAD_EVENTS[name] = threading.Event()
                break
            if _waits_on(owner, me):
                # Self-lookup, or a cross-thread cycle that would deadlock:
                # the caller sees only what is registered so far.
                return
            _WAITING_FOR[me] = name
            waiter = _PLUGIN_LOAD_EVENTS[name]
        # Another thread is importing this plugin; wait for its outcome.
        try:
            finished = waiter.wait(_PLUGIN_WAIT_TIMEOUT_S)
        finally:
            with _PLUGIN_LOAD_LOCK:
                _WAITING_FOR.pop(me, None)
        if not finished:
            raise _plugin_wait_timeout(name, "its load on another thread")
    try:
        _load_backend_plugin(ep)
    except _RetryablePluginLoadError:
        # Transient refusal or wait timeout: retryable, not remembered.
        raise
    except Exception as e:
        message = f"{_describe(ep)} failed to load: {type(e).__name__}: {e}"
        # Remembered for the process lifetime: a fresh process (or a fixed
        # package plus restart) is required to retry.
        _FAILED_PLUGINS[name] = message
        raise ValueError(message) from e
    finally:
        with _PLUGIN_LOAD_LOCK:
            _LOADING_PLUGINS.pop(name, None)
            done = _PLUGIN_LOAD_EVENTS.pop(name, None)
        if done is not None:
            done.set()


def _waits_on(owner: int | None, me: int) -> bool:
    """Whether ``owner``'s load is (transitively) waiting for thread ``me``."""
    seen: set[int] = set()
    while owner is not None and owner not in seen:
        if owner == me:
            return True
        seen.add(owner)
        waited = _WAITING_FOR.get(owner)
        owner = None if waited is None else _LOADING_PLUGINS.get(waited)
    return False


@contextlib.contextmanager
def _record_backend_registrations() -> Iterator[list[tuple[str, type[Backend]]]]:
    """Collect ``(name, class)`` for each new registry entry on this thread.

    Nests like ``_record_codegen_registrations``.
    """
    stack = getattr(_BACKEND_RECORDING, "stack", None)
    if stack is None:
        stack = _BACKEND_RECORDING.stack = []
    recorded: list[tuple[str, type[Backend]]] = []
    stack.append(recorded)
    try:
        yield recorded
    finally:
        stack.pop()


def _load_backend_plugin(ep: EntryPoint) -> None:
    """Import, validate and register one plugin; undo everything on failure.

    The imports (``ep.load()`` and the backend's ``_codegen_modules``) run
    without ``_CODEGEN_REPAIR_LOCK`` held: plugin code is arbitrary and
    importing under the repair lock can deadlock against a concurrent
    compile (which takes the repair lock and the import lock in the other
    order).  Loads of the *same* plugin stay serialized by
    ``_LOADING_PLUGINS``, and other threads looking it up wait for the
    commit.  The post-import check compares against the pre-import
    snapshot, ignoring entries others added or rolled back meanwhile.
    Rollback removes only the registry entries and ``@codegen`` handlers
    this load made on this thread; other import side effects (threads
    started, files written, modules cached in ``sys.modules``) are not
    undone.  A codegen repair running on another thread is waited out; one
    on this thread refuses the load (retryably), since the repair filter
    would drop its handlers.  Both are checked once, up front.  Every wait on
    another thread is bounded by ``_PLUGIN_WAIT_TIMEOUT_S`` and fails the load
    retryably, so a cycle through an import or repair lock raises, not hangs.
    """
    from ..language._decorators import _codegen_repair_in_progress
    from ..language._decorators import _forget_recorded_registrations
    from ..language._decorators import _record_codegen_registrations

    if _codegen_repair_in_progress():
        raise _RetryablePluginLoadError(
            "cannot load a backend plugin while backend codegen repair is "
            "in progress: handlers registered now would be silently dropped "
            "by the repair filter; retry after compilation"
        )
    with _plugin_repair_lock(ep.name):
        pass  # wait for another thread's repair to finish

    registry_before = dict(_REGISTRY)
    plugin_root = ep.module.split(".", 1)[0]
    ok = False
    with (
        _record_codegen_registrations() as recorded,
        _record_backend_registrations() as registered,
    ):
        try:
            backend_cls = ep.load()
            if not (isinstance(backend_cls, type) and issubclass(backend_cls, Backend)):
                raise TypeError("entry point is not a Backend subclass")
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
            if _REGISTRY.get(name) not in (None, backend_cls):
                raise RuntimeError(f"another class registered itself as {name!r}")
            with _plugin_repair_lock(name):
                if name not in _REGISTRY:
                    register_compiler_backend(backend_cls)
            _import_codegen_modules(backend_cls)
            with _plugin_repair_lock(name):
                # Checked last so the codegen modules cannot replace a backend
                # either.
                if any(
                    _REGISTRY.get(k, v) is not v for k, v in registry_before.items()
                ):
                    raise RuntimeError("importing it replaced a registered backend")
                if _REGISTRY.get(name) is not backend_cls:
                    raise RuntimeError(f"another class registered itself as {name!r}")
                ok = True
                _LOADED_PLUGINS.add(name)
                _ROLLED_BACK.discard(backend_cls)  # a retry may succeed
                # A plugin loading this one must not undo it if that plugin
                # fails.
                _forget_recorded_registrations(recorded)
        except BaseException as e:
            if not ok:
                _rollback_plugin_load(
                    e, registry_before, registered, recorded, plugin_root
                )
            raise


def _rollback_plugin_load(
    error: BaseException,
    registry_before: dict[str, type[Backend]],
    registered: list[tuple[str, type[Backend]]],
    recorded: list[tuple[dict[str, object], str, object, object]],
    plugin_root: str,
) -> None:
    """Undo a failed load; a rollback failure becomes a note on ``error``."""
    try:
        # Roll back even without the lock: a dead class must not stay registered.
        locked = _CODEGEN_REPAIR_LOCK.acquire(timeout=_PLUGIN_WAIT_TIMEOUT_S)
        try:
            _restore_registry(registry_before, registered)
        finally:
            if locked:
                _CODEGEN_REPAIR_LOCK.release()
    except Exception as rollback_error:
        _add_note(error, f"registry rollback also failed: {rollback_error!r}")
    try:
        _undo_plugin_handlers(recorded, plugin_root)
    except Exception as rollback_error:
        _add_note(error, f"@codegen handler rollback also failed: {rollback_error!r}")


def _add_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)  # Python 3.11+
    if add_note is not None:
        add_note(note)
    else:
        # pyrefly: ignore[missing-attribute]
        error.__notes__ = [*getattr(error, "__notes__", ()), note]


def _restore_registry(
    before: dict[str, type[Backend]], registered: list[tuple[str, type[Backend]]]
) -> None:
    # Change only what differs, so concurrent readers never see it empty.
    for key, cls in reversed(registered):
        if key not in _LOADED_PLUGINS and _REGISTRY.get(key) is cls:
            _ROLLED_BACK.add(cls)  # before the delete, so no restore re-adds it
            del _REGISTRY[key]
    for key, value in before.items():
        # Absent entries were another load's rollback; a present one may be a
        # later registration over it.  Either way a rolled-back class stays out.
        if value not in _ROLLED_BACK and _REGISTRY.get(key, value) is not value:
            _REGISTRY[key] = value


def _undo_plugin_handlers(
    recorded: list[tuple[dict[str, object], str, object, object]], plugin_root: str
) -> None:
    from ..language._decorators import _NO_PREVIOUS_HANDLER

    for implementations, backend, previous, handler in reversed(recorded):
        # Only handlers defined by the plugin's own package; a plugin may
        # trigger lazy imports of Helion modules that register built-in ones.
        module = getattr(handler, "__module__", None) or ""
        if module.split(".", 1)[0] != plugin_root:
            continue
        # Raw dict access, deliberately: ``implementations`` is usually a
        # CodegenDict whose getitem/__setitem__ dispatch through
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
    for backend_cls in list(_REGISTRY.values()):
        package = backend_cls.__module__.rsplit(".", 1)[0]
        if package not in packages or package in seen_packages:
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
    with _CODEGEN_REPAIR_LOCK:
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
