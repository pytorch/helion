"""Centralized registry for Helion codegen backends.

All backend lookup and instantiation should go through this module.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
from typing import TYPE_CHECKING

from .cute.backend import CuteBackend
from .flydsl.backend import FlyDSLBackend
from .metal.backend import MetalBackend
from .pallas.backend import PallasBackend
from .triton.backend import TileIRBackend
from .triton.backend import TritonBackend

if TYPE_CHECKING:
    from ..autotuner.search_surface import BackendSearchSurface
    from .backend import Backend

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
    "TileStrategy.compact_shape",
    "TileStrategy.codegen_preamble",
    "ReductionStrategy.codegen_reduction",
)

_BUILTIN_BACKENDS: list[type[Backend]] = [
    TritonBackend,
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
    """
    global _GENERATION, _REPAIRED_CODEGEN_NAMES

    backend = backend_class()
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
    """Look up a registered backend class by name."""
    if name not in _REGISTRY:
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
    """Return the names of all registered backends."""
    return list(_REGISTRY.keys())


def all_reserved_launch_param_names() -> frozenset[str]:
    """Union of reserved launch param names across all registered backends.

    Reserving all names ensures kernel portability. A variable name
    that collides with any backend's launch params is avoided regardless
    of which backend is currently active.
    """
    result: set[str] = set()
    for backend_cls in _REGISTRY.values():
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
    for backend_cls in _REGISTRY.values():
        # e.g. "helion._compiler.cute.backend" -> "helion._compiler.cute".
        # Subclasses that share a folder (e.g. TileIRBackend in the triton
        # package) collapse to one import.
        package = backend_cls.__module__.rsplit(".", 1)[0]
        if package in seen:
            continue
        seen.add(package)
        module = f"{package}._codegen_modules"
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            # A backend without any per-op codegen modules is allowed; only
            # swallow the missing _codegen_modules module itself, never a
            # broken import inside it.
            if e.name != module:
                raise


def _codegen_repair_scope(
    codegen_name: str,
) -> tuple[frozenset[str], frozenset[str]]:
    entries = [
        (
            backend_class().codegen_name,
            backend_class.__module__.rsplit(".", 1)[0],
        )
        for backend_class in _REGISTRY.values()
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
    for backend_cls in _REGISTRY.values():
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
