"""Centralized registry for Helion codegen backends.

All backend lookup and instantiation should go through this module.
"""

from __future__ import annotations

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
from .flydsl.backend import FlyDSLBackend
from .metal.backend import MetalBackend
from .pallas.backend import PallasBackend
from .triton.backend import GluonBackend
from .triton.backend import TileIRBackend
from .triton.backend import TritonBackend

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from ..autotuner.search_surface import BackendSearchSurface

# Version of the out-of-tree backend contract in ``helion.backend_api``; covers
# only the names and hooks listed in docs/api/backend_api.md, bumped per the
# version policy there.
BACKEND_API_VERSION = 1

# The Backend members and strategy methods whose names and signatures
# BACKEND_API_VERSION covers; docs/api/backend_api.md describes them.
COVERED_BACKEND_HOOKS: tuple[str, ...] = (
    "name",
    "dtype_str",
    "acc_type",
    "function_decorator",
    "constexpr_type",
    "default_launcher_name",
    "library_imports",
    "codegen_name",
    "state",
    "validate_environment",
    "pre_inductor_lowering",
    "inductor_op_overrides",
    "customize_ast",
    "pre_codegen",
    "create_loop_strategy",
    "create_reduction_strategy",
    "program_id_expr",
    "grid_index_expr",
    "loop_index_expr",
    "arange_expr",
    "range_str",
    "lane_index_expr",
    "lane_offset_expr",
    "thread_in_tile_mask_expr",
    "scalar_load_expr",
    "inline_constexpr",
    "scalar_arg_preamble",
    "finalize_device_body",
    "cast_expr",
    "cast_scalar_ast",
    "where_expr",
    "minimum_expr",
    "zeros_expr",
    "full_expr",
    "reshape_expr",
    "broadcast_to_expr",
    "expands_broadcast_dims",
    "reduction_expr",
    "reduction_combine_expr",
    "reduction_index_expr",
    "reduction_index_zero_expr",
    "wrap_reduction_accumulator",
    "looped_reduction_thread_count",
    "register_reduction_loop_config_slots",
    "max_reduction_threads",
    "max_reduction_loop",
    "reduction_block_size_is_inlined_constexpr",
    "launches_surplus_tile_threads",
    "launcher_keyword_args",
    "supports_config_key",
    "config_spec_extension",
    "tunable_fragments",
    "supports_precompile",
    "adjust_block_size_constraints",
    "autotune",
    "get_do_bench",
    "get_interleaved_bench",
)
COVERED_STRATEGY_METHODS: tuple[str, ...] = (
    "TileStrategy.user_size",
    "TileStrategy.codegen_grid",
    "TileStrategy.codegen_device_loop",
    "TileStrategy.codegen_preamble",
    "ReductionStrategy.codegen_reduction",
)
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
# Serializes plugin loads.  Taken only to load a name not yet registered, so
# lookups of registered backends never wait on it.
_PLUGIN_LOAD_LOCK = threading.RLock()
# Names being loaded, set under _PLUGIN_LOAD_LOCK: a plugin that looks up its
# own backend while it imports (same thread) sees the registry as it is.
_LOADING: set[str] = set()

_BUILTIN_BACKENDS: list[type[Backend]] = [
    TritonBackend,
    GluonBackend,
    PallasBackend,
    CuteBackend,
    TileIRBackend,
    MetalBackend,
    FlyDSLBackend,
]

_REGISTRY: dict[str, type[Backend]] = {}
# Bumped by every registration and unregistration: the tables derived from
# the registry (``config_spec``'s key tables, ``autotuner_heuristics``'s
# heuristics by backend) key their caches on it, so a backend registered
# after those modules imported is seen on the next lookup.
_GENERATION = 0
_CODEGEN_REPAIR_LOCK = threading.RLock()
_REPAIRED_CODEGEN_NAMES: frozenset[str] = frozenset()


def register_compiler_backend(backend_class: type[Backend]) -> None:
    """Register a compiler backend.

    The backend's ``name`` property is used as the registry key.
    Built-in backends are registered at module load time below.

    Registering the same class twice is a no-op, and a class with the same
    module and qualified name (the one a reloaded module or a re-run factory
    defines) replaces the old one.  Registering any other class under an
    already-registered name (including built-in names) raises ``ValueError``,
    so through ``helion.backend_api`` a name is owned by its first registering
    class.  Helion-internal code that must
    replace a backend calls :func:`unregister_compiler_backend` first.

    Args:
        backend_class: A :class:`Backend` subclass.

    Raises:
        ValueError: if another class is already registered under the name, or
            if a non-built-in backend's search surface declares a generic
            config key.
    """
    global _GENERATION, _REPAIRED_CODEGEN_NAMES

    backend = backend_class()
    if backend_class not in _BUILTIN_BACKENDS:
        _check_surface_keys(backend)  # calls plugin code: not under the lock
    with _CODEGEN_REPAIR_LOCK:
        existing = _REGISTRY.get(backend.name)
        if existing is backend_class:
            return
        if existing is not None and not _is_redefinition(existing, backend_class):
            hint = (
                " (the same class from a second copy of its module, e.g. run as "
                "__main__ and also imported by name?)"
                if existing.__qualname__ == backend_class.__qualname__
                and "__main__" in (existing.__module__, backend_class.__module__)
                else ""
            )
            raise ValueError(
                f"Backend name {backend.name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}; "
                f"refusing to overwrite it with "
                f"{backend_class.__module__}.{backend_class.__qualname__}{hint}"
            )
        _REGISTRY[backend.name] = backend_class
        _GENERATION += 1
        codegen_names, _ = _codegen_repair_scope(backend.codegen_name)
        _REPAIRED_CODEGEN_NAMES = _REPAIRED_CODEGEN_NAMES.difference(codegen_names)


def _is_redefinition(existing: type[Backend], backend_class: type[Backend]) -> bool:
    # importlib.reload, or a factory run again, defines a new class object with
    # the same module and qualified name; it replaces the old one.
    return (existing.__module__, existing.__qualname__) == (
        backend_class.__module__,
        backend_class.__qualname__,
    )


def _check_surface_keys(backend: Backend) -> None:
    # Not run for built-ins: they register while config_spec is importing.
    from ..autotuner.config_spec import GENERIC_KEYS

    surface = backend.config_spec_extension()
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

    with _CODEGEN_REPAIR_LOCK:
        if _REGISTRY.pop(name, None) is not None:
            _GENERATION += 1


def registry_generation() -> int:
    """A counter bumped by every (un)registration, for caches derived from
    the registry."""
    return _GENERATION


def get_backend_class(name: str) -> type[Backend]:
    """Look up a backend class by name, loading its plugin on first use."""
    if name not in _REGISTRY:
        _load_plugin_by_name(name)
    if name not in _REGISTRY:
        if name in _LOADING:  # only ever this thread's own load
            raise ValueError(
                f"Backend {name!r} is still loading: it was looked up while its "
                "plugin is being imported"
            )
        raise ValueError(
            f"Unknown backend: {name!r}. Available backends: {list_backends()}"
        )
    return _REGISTRY[name]


def get_backend(name: str) -> Backend:
    """A fresh instance of the backend registered as ``name``.

    Callers that only read a hook (a capability predicate, the heuristics
    tuple, the structural policy type) get their own instance: backends are
    cheap to construct and several (Metal, Pallas, FlyDSL) keep per-compile
    state on the instance, so instances are never shared.
    """
    return get_backend_class(name)()


def list_backends() -> list[str]:
    """Return registered backends plus installed, not-yet-loaded plugins."""
    names = list(_REGISTRY)
    names.extend(n for n in _plugin_entry_points() if n not in _REGISTRY)
    return names


def registered_backend_classes() -> dict[str, type[Backend]]:
    """A snapshot of the backends registered so far; never loads a plugin.

    For tables derived from the registry: they refresh on
    ``registry_generation()`` when a plugin loads later.
    """
    return dict(_REGISTRY)


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
    for backend_cls in list(_REGISTRY.values()):
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
        except ValueError as e:
            warnings.warn(str(e), stacklevel=2)


def _plugins_disabled() -> bool:
    from ..runtime.settings import _parse_env_bool  # settings imports this module

    var = "HELION_DISABLE_BACKEND_PLUGINS"
    return _parse_env_bool(var, os.environ.get(var), False)


def _plugin_entry_points() -> dict[str, EntryPoint]:
    """Installed ``helion.backends`` entry points by name (discovered once)."""
    global _PLUGIN_ENTRY_POINTS

    entry_points = _PLUGIN_ENTRY_POINTS
    if entry_points is not None:
        return entry_points
    # Imports helion.runtime.settings, so not under the lock.
    disabled = _plugins_disabled()
    ignored: list[EntryPoint] = []
    with _DISCOVERY_LOCK:
        if _PLUGIN_ENTRY_POINTS is None:
            found, ignored = _discover_plugin_entry_points(disabled)
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
        entry_points = _PLUGIN_ENTRY_POINTS
    # After the lock: a warning filter or hook may run arbitrary code.
    for ep in ignored:
        warnings.warn(
            f"Ignoring {_describe(ep)}: {ep.name!r} is a built-in backend",
            stacklevel=2,
        )
    return entry_points


def _discover_plugin_entry_points(
    disabled: bool,
) -> tuple[dict[str, list[EntryPoint]], list[EntryPoint]]:
    """Plugin entry points by name, and those named after a built-in."""
    found: dict[str, list[EntryPoint]] = {}
    ignored: list[EntryPoint] = []
    if not disabled:
        builtins = {cls().name for cls in _BUILTIN_BACKENDS}
        entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        for ep in sorted(entry_points, key=lambda ep: (_dist_name(ep), ep.name)):
            if ep.name in builtins:
                ignored.append(ep)
            else:
                found.setdefault(ep.name, []).append(ep)
    return found, ignored


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
    if ep is None:
        return
    with _PLUGIN_LOAD_LOCK:
        if name in _REGISTRY or name in _LOADING:
            # Loaded meanwhile or registered by hand; or looked up by the
            # plugin itself while it imports.
            return
        if name in _FAILED_PLUGINS:
            raise ValueError(_FAILED_PLUGINS[name])
        _LOADING.add(name)
        try:
            _load_backend_plugin(ep)
        # Remembered for the process lifetime, interrupted loads too: a fresh
        # process is required to retry (a half-imported plugin would not
        # re-import cleanly).
        except Exception as e:
            message = f"{_describe(ep)} failed to load: {type(e).__name__}: {e}"
            _FAILED_PLUGINS[name] = message
            raise ValueError(message) from e
        except BaseException as e:
            _FAILED_PLUGINS[name] = (
                f"{_describe(ep)} failed to load: interrupted ({type(e).__name__})"
            )
            raise
        finally:
            _LOADING.discard(name)


def _load_backend_plugin(ep: EntryPoint) -> None:
    """Import, validate and register one plugin.  The caller holds
    ``_PLUGIN_LOAD_LOCK``.

    On failure every name registered during the load is unregistered; names
    registered before it (built-ins included) stay.  Nothing else is undone:
    ``@codegen`` handlers (keyed by the plugin's own codegen name) and other
    import side effects remain.
    """
    registered_before = set(_REGISTRY)
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
                f"its backend is named {name!r}; the entry point must use the "
                "backend name"
            )
        # A no-op if the plugin registered itself; raises if another class
        # holds the name.
        register_compiler_backend(backend_cls)
        _import_codegen_modules(backend_cls)
    except BaseException:
        for key in [k for k in list(_REGISTRY) if k not in registered_before]:
            unregister_compiler_backend(key)
        raise


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
