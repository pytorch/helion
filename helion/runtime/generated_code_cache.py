"""Persistent Triton source and conservative pre-frontend binding caches."""

from __future__ import annotations

import builtins
import dataclasses
import functools
import hashlib
import json
import logging
import math
import operator
import os
from pathlib import Path
import re
import sys
import types
from typing import TYPE_CHECKING
from typing import cast
import uuid

from filelock import FileLock
from filelock import Timeout
import sympy
import torch
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from .._argument_device import _find_argument_device
from .._compat import device_num_sm
from .._compat import get_device_name
from .._compat import supports_torch_compile_fusion
from .._compat import target_device_capability
from .._compiler.cache_dependencies import validate_dependencies
from .._compiler.compile_environment import TensorDescriptorAlignmentGuard
from .._utils import counters
from .._utils import indexing_uses_tensor_descriptor
from ..autotuner.base_cache import helion_key
from ..autotuner.base_cache import should_skip_cache
from ..autotuner.base_cache import torch_key_wrapper
from ..autotuner.base_cache import triton_key_wrapper
from ..autotuner.local_cache import get_helion_cache_dir
from ..language.constexpr import ConstExpr
from .config import Config
from .ref_mode import RefMode
from .ref_mode import is_ref_mode_enabled
from .settings import default_autotuner_fn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Hashable

    from torch._guards import Source

    from .._compiler.compile_environment import TensorDescriptorLayoutGuard
    from .kernel import BoundKernel
    from .kernel import CompiledConfig
    from .kernel import Kernel
    from .kernel import _GuardedInputFact
    from .settings import Settings

    # Specialization key, serialized guards, their results, compiler seed
    # results and the input fingerprint behind generated-code cache keys.
    GeneratedCodeInputKey = tuple[
        tuple[Hashable, ...],
        tuple[list[object] | None, ...],
        tuple[Hashable, ...],
        tuple[Hashable, ...],
        str,
    ]
    # What a ``BoundKernel._specialize_extra`` extractor reads: its source, the
    # descriptor guard and extent cap for descriptor extractors, or None for
    # runtime input classifiers.
    GuardSource = (
        tuple[
            Source,
            TensorDescriptorLayoutGuard | TensorDescriptorAlignmentGuard | None,
            int | None,
        ]
        | None
    )

log = logging.getLogger(__name__)
_UNCACHEABLE = object()
# Running byte total of the JSON entries, guarded by the writers' lock.
_SIZE_INDEX = ".size"
_SIZE_INDEX_RESCAN_WRITES = 1024
_DEFAULT_CACHE_LIMIT = 1 << 30
# Default reprs carry a per-process address, e.g. a fragment passed to
# hl.register_tunable in the frontend debug string, or the code object or
# builtin of a callable argument in the specialization key. The keys identify
# these objects by content elsewhere (kernel source, binding_input_key, the
# config), and binding_input_key rejects arguments it cannot fingerprint, so
# dropping the address loses nothing.
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_HANDLER_SLOTS = (
    "__wrapped__",
    "_type_function",
    "_fake_fn",
    "_prepare_args",
    "_post_create_proxy",
    "_to_device_ir",
    "_get_masked_value",
)
_FROZEN_COMPILER_HANDLERS: dict[int, dict[tuple[object, ...], object]] | None = None
_FROZEN_COMPILER_NAMESPACES: dict[
    str, tuple[types.ModuleType, dict[str, tuple[object, ...]]]
] = {}
_FROZEN_WARNING_TYPES: dict[int, tuple[type, str]] = {}


def _intrinsic_handlers(value: object) -> dict[tuple[object, ...], object]:
    if isinstance(value, type):
        return {
            (base, name): member
            for base in value.__mro__
            for name, member in vars(base).items()
            if isinstance(member, types.FunctionType)
        }
    if isinstance(value, types.FunctionType):
        result: dict[tuple[object, ...], object] = {
            (name,): vars(value).get(name) for name in _HANDLER_SLOTS
        }
        codegen = vars(value).get("_codegen")
        result[("_codegen",)] = codegen
        if isinstance(codegen, dict):
            result.update(
                (("_codegen", name), handler)
                for name, handler in dict.items(codegen)
                if name in ("common", "triton")
            )
        return result
    return {}


def freeze_compiler_cache_contract() -> None:
    """Pin native registrations once language and backend imports have finished."""
    global _FROZEN_COMPILER_HANDLERS
    if _FROZEN_COMPILER_HANDLERS is None:
        from .. import exc
        from .._compiler.backend_registry import repair_backend_codegen
        from ..autotuner.config_spec import valid_keys

        # Triton dispatch normally completes these registrations on its first
        # compile. Capture that native state before user code can replace it.
        repair_backend_codegen("triton")
        # Materialize the native config-key table before pinning module bindings.
        valid_keys()
        _FROZEN_COMPILER_HANDLERS = {
            identifier: _intrinsic_handlers(value)
            for identifier, (value, _) in _cache_intrinsics().items()
        }
        _FROZEN_WARNING_TYPES.update(
            (id(value), (value, f"{exc.__name__}.{name}"))
            for name, value in vars(exc).items()
            if type(value) is type
            and value.__module__ == exc.__name__
            and issubclass(value, exc.BaseWarning)
        )
        _freeze_compiler_namespaces()


def _namespace_binding(
    value: object, module_name: str = "", name: str = ""
) -> tuple[object, ...]:
    if (
        module_name == "helion._compiler.backend_registry"
        and name == "_REPAIRED_CODEGEN_NAMES"
        and type(value) is frozenset
        and all(type(backend) is str for backend in value)
    ):
        # This is a backend-load memo. Other backends may finish loading after
        # Triton's contract is frozen; its actual handlers are guarded separately.
        return frozenset, "triton" in value
    if type(value) is types.FunctionType:
        return value, value.__code__, value.__defaults__, value.__kwdefaults__
    return (value,)


def _freeze_compiler_namespaces() -> None:
    """Pin the native module graph reached by operations and Triton handlers.

    This guards the fixed compiler context, not arbitrary user Python. Imported
    Helion helper functions and modules extend the graph. Module data objects
    retain identity so native mutable caches can continue to populate in place.
    """
    pending: list[str] = []
    for value, _ in _cache_intrinsics().values():
        if isinstance(value, (types.FunctionType, type)):
            pending.append(value.__module__)
        for path, handler in _intrinsic_handlers(value).items():
            if path[0] == "_codegen" and (
                len(path) == 1 or path[1] not in ("common", "triton")
            ):
                continue
            if type(handler) is types.FunctionType:
                pending.append(handler.__module__)
    while pending:
        name = pending.pop()
        if (
            name in _FROZEN_COMPILER_NAMESPACES
            or name != "helion"
            and not name.startswith("helion.")
        ):
            continue
        module = sys.modules.get(name)
        if type(module) is not types.ModuleType:
            continue
        namespace = dict(vars(module))
        _FROZEN_COMPILER_NAMESPACES[name] = (
            module,
            {
                key: _namespace_binding(value, name, key)
                for key, value in namespace.items()
            },
        )
        # The package exports are bindings, not a request to traverse unrelated
        # runtime entry points such as KernelCompiler and its instrumentation.
        if name == "helion":
            continue
        for value in namespace.values():
            if type(value) is types.FunctionType:
                pending.append(value.__module__)
            elif type(value) is types.ModuleType:
                pending.append(value.__name__)


def _compiler_namespaces_valid() -> bool:
    for name, (module, bindings) in _FROZEN_COMPILER_NAMESPACES.items():
        if sys.modules.get(name) is not module:
            return False
        namespace = vars(module)
        for key, expected in bindings.items():
            current = _namespace_binding(namespace.get(key, _UNCACHEABLE), name, key)
            if len(current) != len(expected) or any(
                actual is not saved
                for actual, saved in zip(current, expected, strict=True)
            ):
                return False
    return True


def _compiler_cache_contract_valid() -> bool:
    from .._compiler.type_info import tensor_cache_contract_valid

    if (
        _FROZEN_COMPILER_HANDLERS is None
        or not tensor_cache_contract_valid()
        or not _compiler_namespaces_valid()
    ):
        return False
    for identifier, (value, _) in _cache_intrinsics().items():
        expected = _FROZEN_COMPILER_HANDLERS[identifier]
        current = _intrinsic_handlers(value)
        if current.keys() != expected.keys() or any(
            current[path] is not handler for path, handler in expected.items()
        ):
            return False
    return True


@functools.cache
def _warn_cache_unavailable_once(module: str, name: str, reason: str) -> None:
    log.warning(
        "Generated-code cache unavailable for %s.%s: %s; using normal compilation",
        module,
        name,
        reason,
    )


def _warn_cache_unavailable(kernel: Kernel, reason: str) -> None:
    if kernel.settings.cache_level == "code":
        # Cache only strings: retaining a Kernel here would keep its bindings,
        # compiled modules and captured input objects alive indefinitely.
        _warn_cache_unavailable_once(kernel.fn.__module__, kernel.name, reason)


def _without_addresses(value: object) -> str:
    return _OBJECT_ADDRESS.sub("", value if isinstance(value, str) else repr(value))


@functools.cache
def _cache_limit() -> int:
    """Byte budget shared by every entry, read once from the environment."""
    value = os.environ.get("HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES")
    if value is None or not value.strip():
        return _DEFAULT_CACHE_LIMIT
    try:
        limit = int(value)
    except ValueError:
        limit = -1
    if limit < 0:
        log.warning(
            "HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES must be a nonnegative "
            "integer, got %r; using %d",
            value,
            _DEFAULT_CACHE_LIMIT,
        )
        return _DEFAULT_CACHE_LIMIT
    return limit


def _code_key(code: types.CodeType) -> tuple[object, ...]:
    constants = tuple(
        _code_key(value)
        if isinstance(value, types.CodeType)
        else tuple(sorted(value, key=repr))
        if isinstance(value, frozenset)
        else value
        for value in code.co_consts
    )
    return (
        code.co_code,
        constants,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_filename,
        code.co_firstlineno,
        getattr(code, "co_exceptiontable", b""),
    )


@functools.cache
def _cache_intrinsics() -> dict[int, tuple[object, str]]:
    """The compiler operations whose implementation is keyed by library hashes.

    The frontend records reads and rejects calls without a dependency contract.
    Reflection, I/O and arbitrary user Python helpers cannot publish a certificate.
    Match objects, never a callable's claimed module/name or an API marker.
    """
    from .._utils import cdiv
    from .._utils import next_power_of_2
    from ..autotuner import config_fragment
    from ..autotuner.config_fragment import ConfigSpecFragment

    entries: list[tuple[object, str]] = [
        (cdiv, "helion.cdiv"),
        (next_power_of_2, "helion.next_power_of_2"),
    ]
    # Exact compiler-owned fragment types share the constructor/parameter
    # contract consumed by register_tunable. User subclasses are not admitted.
    entries.extend(
        (cls, f"{cls.__module__}.{cls.__qualname__}")
        for cls in vars(config_fragment).values()
        if isinstance(cls, type)
        and cls.__module__ == config_fragment.__name__
        and issubclass(cls, ConfigSpecFragment)
        and cls is not ConfigSpecFragment
    )
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    ):
        entries.append((vars(builtins)[name], f"builtins.{name}"))
    for module in (math, operator):
        entries.extend(
            (value, f"{module.__name__}.{name}")
            for name, value in vars(module).items()
            if not name.startswith("_") and isinstance(value, types.BuiltinFunctionType)
        )
    from ..language._decorators import _cacheable_apis

    entries.extend(
        (fn, f"{fn.__module__}.{fn.__qualname__}")
        for fn in _cacheable_apis
        if isinstance(fn, types.FunctionType)
    )
    # Tensor operators with value semantics. Overloadability alone is not
    # enough: Torch also exposes state readers and metadata queries whose
    # results are not covered by the binding's saved tensor guards.
    for name in (
        "abs",
        "add",
        "addmm",
        "arange",
        "bmm",
        "cat",
        "clamp",
        "cos",
        "div",
        "empty",
        "empty_like",
        "empty_strided",
        "exp",
        "exp2",
        "full",
        "full_like",
        "log",
        "log2",
        "matmul",
        "maximum",
        "minimum",
        "mm",
        "mul",
        "neg",
        "ones",
        "ones_like",
        "pow",
        "relu",
        "rsqrt",
        "sigmoid",
        "sin",
        "sqrt",
        "stack",
        "sub",
        "sum",
        "tanh",
        "where",
        "zeros",
        "zeros_like",
    ):
        fn = vars(torch)[name]
        if isinstance(fn, types.BuiltinFunctionType) and fn.__self__ is None:
            entries.append((fn, f"torch.{name}"))
    return {id(value): (value, name) for value, name in entries}


def _intrinsic_implementation(value: object) -> str | None:
    """Detect changes to compiler-owned implementations after import as well."""

    def function_key(fn: object) -> object:
        if isinstance(fn, types.FunctionType):
            defaults = _dependency_key((fn.__defaults__, fn.__kwdefaults__))
            if defaults is _UNCACHEABLE:
                return _UNCACHEABLE
            return _code_key(fn.__code__), defaults
        if fn is None:
            return None
        if isinstance(fn, types.BuiltinFunctionType):
            return fn.__module__, fn.__qualname__
        return _UNCACHEABLE

    if isinstance(value, type):
        parts = [
            (base.__qualname__, name, function_key(member))
            for base in value.__mro__
            for name, member in vars(base).items()
            if isinstance(member, types.FunctionType)
        ]
    elif isinstance(value, types.FunctionType):
        parts = [("function", function_key(value))]
        for name in _HANDLER_SLOTS:
            parts.append((name, function_key(vars(value).get(name))))
        codegen = vars(value).get("_codegen", {})
        if isinstance(codegen, dict):
            parts.extend(
                (name, function_key(fn))
                for name, fn in sorted(codegen.items())
                if name in ("common", "triton")
            )
    else:
        parts = []
    if any(part[-1] is _UNCACHEABLE for part in parts):
        return None
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def compiler_abi_key() -> str:
    """Version the registered operation implementations, including transitive helpers."""
    parts = sorted(
        (name, _intrinsic_implementation(value) or "unsupported")
        for value, name in _cache_intrinsics().values()
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def validate_kernel_dependencies(snapshot: object, fn: types.FunctionType) -> bool:
    return (
        isinstance(snapshot, dict)
        and not _has_active_torch_mode()
        and _compiler_cache_contract_valid()
        and snapshot.get("compiler_abi") == compiler_abi_key()
        and validate_dependencies(snapshot, fn, dependency_fingerprint)
    )


def _has_active_torch_mode() -> bool:
    """User dispatch modes can execute Python beyond the captured operations."""
    return bool(
        torch._C._is_torch_function_mode_enabled() or _get_current_dispatch_mode_stack()
    )


def _dependency_snapshot(bound: BoundKernel) -> dict[str, object] | None:
    capture = bound.env.cache_dependencies
    if capture is None:
        return None
    if not _compiler_cache_contract_valid():
        capture.reject("the native compiler implementation context changed")
        return None
    compiler_abi = compiler_abi_key()
    if compiler_abi != bound.env.cache_compiler_abi:
        capture.reject("compiler operations changed during compilation")
        return None
    snapshot = capture.snapshot()
    return None if snapshot is None else {**snapshot, "compiler_abi": compiler_abi}


def dependency_fingerprint(value: object) -> str | None:
    key = _dependency_key(value)
    return None if key is _UNCACHEABLE else repr(key)


def cacheable_call(value: object) -> bool:
    entry = _cache_intrinsics().get(id(value))
    return entry is not None and entry[0] is value


def _dependency_key(value: object, seen: frozenset[int] = frozenset()) -> object:
    """Fingerprint supported Python dependencies without pickling live objects."""
    if value is None or type(value) in (int, float, bool, str, bytes):
        return type(value).__name__, value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if type(value) is RefMode:
        return "ref_mode", value.value
    if type(value) in (tuple, list, dict):
        if id(value) in seen:
            return _UNCACHEABLE
        seen = seen | {id(value)}
        items = (
            value.items()
            if isinstance(value, dict)
            else enumerate(cast("Sequence[object]", value))
        )
        result = []
        for name, item in items:
            key = _dependency_key(item, seen)
            name_key = _dependency_key(name, seen)
            if key is _UNCACHEABLE or name_key is _UNCACHEABLE:
                return _UNCACHEABLE
            result.append((name_key, key))
        return type(value).__name__, tuple(result)
    intrinsic = _cache_intrinsics().get(id(value))
    if intrinsic is not None and intrinsic[0] is value:
        implementation = _intrinsic_implementation(value)
        return (
            _UNCACHEABLE
            if implementation is None
            else ("intrinsic", intrinsic[1], implementation)
        )
    if isinstance(value, (types.ModuleType, types.BuiltinFunctionType, type)):
        return _UNCACHEABLE
    return _UNCACHEABLE


# Settings that cannot change generated code. Search policy selects a config
# already included in the source key, and the cache level does not change the
# generated source. Every other field, including future ones, is keyed.
_NON_CODEGEN_SETTING_PREFIXES = ("autotun",)
_NON_CODEGEN_SETTINGS = frozenset({"cache_level"})


def _compilation_settings_key(settings: Settings) -> object:
    # Reject unknown compilation values instead of hashing an address-bearing
    # repr. Read fields directly: to_dict() deep-copies arbitrary values first.
    values = dict(
        sorted(
            (field.name, getattr(settings, field.name))
            for field in dataclasses.fields(settings)
            if field.name not in _NON_CODEGEN_SETTINGS
            and not field.name.startswith(_NON_CODEGEN_SETTING_PREFIXES)
        )
    )
    warnings = values["ignore_warnings"]
    if type(warnings) not in (list, tuple):
        return _UNCACHEABLE
    warning_names = []
    for warning in warnings:
        native = _FROZEN_WARNING_TYPES.get(id(warning))
        if native is None or native[0] is not warning:
            return _UNCACHEABLE
        warning_names.append(native[1])
    # Warning policy affects diagnostics, but accepting native warning classes
    # here must not admit arbitrary classes or calls in kernel dependencies.
    values["ignore_warnings"] = ("native_warning_types", tuple(warning_names))
    return _dependency_key(values)


def _argument_key(value: object) -> object:
    """Fingerprint argument structure, not values that dynamic bindings share.

    The in-memory specialization key owns shapes, strides and constexpr values.
    Runtime scalars specialize only through the frontend's saved guards.
    """
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        tensor = cast("torch.Tensor", value)
        return (
            "tensor",
            str(tensor.dtype),
            str(tensor.device),
            tensor.dim(),
            tuple(sorted(getattr(tensor, "_dynamo_static_indices", ()))),
        )
    if type(value) in (int, float, bool):
        return type(value).__name__
    if isinstance(value, ConstExpr):
        key = _dependency_key(value.value)
        return _UNCACHEABLE if key is _UNCACHEABLE else ("constexpr", key)
    if type(value) in (tuple, list, dict):
        items = (
            value.items()
            if isinstance(value, dict)
            else enumerate(cast("Sequence[object]", value))
        )
        result = []
        for name, item in items:
            key = _argument_key(item)
            name_key = _dependency_key(name)
            if key is _UNCACHEABLE or name_key is _UNCACHEABLE:
                return _UNCACHEABLE
            result.append((name_key, key))
        return type(value).__name__, tuple(result)
    return _dependency_key(value)


def binding_input_key(kernel: Kernel, args: Sequence[object]) -> str | None:
    """Fingerprint the kernel's dependencies and the arguments' structure."""
    if (
        kernel.settings.cache_level != "code"
        or kernel.settings.backend != "triton"
        or torch.compiler.is_compiling()
        or _has_active_torch_mode()
    ):
        return None
    dependencies = (
        kernel.fn.__module__,
        _code_key(kernel.fn.__code__),
        kernel.kernel_source(),
    )
    inputs = _argument_key(tuple(args))
    if inputs is _UNCACHEABLE:
        _warn_cache_unavailable(kernel, "arguments contain an unsupported value")
        return None
    return hashlib.sha256(repr((dependencies, inputs)).encode("utf-8")).hexdigest()


def compiled_kernel_cache_key(
    kernel: Kernel,
    args: Sequence[object],
    base_spec_key: tuple[Hashable, ...],
    inputs: str | None = None,
) -> str | None:
    """Identify a signature's saved bindings before the frontend runs."""
    if (
        kernel.settings.force_autotune
        or _has_active_torch_mode()
        or kernel.settings.autotune_handoff
        or not supports_torch_compile_fusion()
        or kernel.settings.print_output_code
        or kernel.settings.print_repro
        or is_ref_mode_enabled(kernel.settings)
        or should_skip_cache()
    ):
        return None
    if (
        not kernel.configs
        and kernel.settings.autotune_effort != "none"
        and kernel.settings.autotuner_fn is default_autotuner_fn
    ):
        # Adaptive tuning must consult LocalAutotuneCache. A source manifest is
        # not authoritative when the user deletes or replaces a tuning result.
        return None
    if inputs is None:
        inputs = binding_input_key(kernel, args)
    if inputs is None:
        return None
    settings = _compilation_settings_key(kernel.settings)
    if settings is _UNCACHEABLE:
        _warn_cache_unavailable(
            kernel, "compilation settings contain an unsupported value"
        )
        return None
    configs = _dependency_key(tuple(dict(config) for config in kernel.configs))
    if configs is _UNCACHEABLE:
        _warn_cache_unavailable(
            kernel, "configured candidates contain an unsupported value"
        )
        return None
    device = _find_argument_device(args)
    hardware: tuple[object, ...]
    if device.type == "cuda":
        # The source key bakes in the compute capability and SM count; devices
        # sharing a name can differ in both (partitioned or cloud SKUs).
        capability = target_device_capability(device)
        if capability is None:
            return None
        hardware = (get_device_name(device), capability, device_num_sm(device))
    elif device.type == "xpu":
        hardware = (get_device_name(device),)
    else:
        hardware = (str(device),)
    payload = (
        "helion-bound-source-v6",
        sys.version_info[:3],
        helion_key(),
        torch_key_wrapper(),
        triton_key_wrapper(),
        inputs,
        compiler_abi_key(),
        _without_addresses(base_spec_key),
        hardware,
        settings,
        # The manifest stores a selected winner. Removing or changing pinned
        # configs must not let it override the new selection policy, especially
        # when autotune_effort="none" selects the frontend's default config.
        configs,
        kernel.settings.autotune_effort,
        str(torch.get_default_dtype()),
        str(torch.get_default_device()),
        torch.get_float32_matmul_precision(),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled(device.type),
        str(torch.get_autocast_dtype(device.type)),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def load_compiled_kernel(key: str) -> tuple[Config, Config, str] | None:
    artifact = _load_compiled_kernel_entry(key)
    if artifact is None:
        return None
    config, normalized, _, source = artifact
    counters["generated_code_cache"]["frontend_hit"] += 1
    return config, normalized, source


def _load_compiled_kernel_entry(key: str) -> tuple[Config, Config, str, str] | None:
    path = get_helion_cache_dir() / "generated_code" / "bindings" / f"{key}.json"
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        entry = envelope["payload"]
        checksum = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if envelope["sha256"] != checksum:
            raise ValueError("compiled kernel checksum mismatch")
        config = Config.from_json(entry["config"])
        normalized = Config.from_json(entry["normalized_config"])
        source_key = entry["source_key"]
        if not isinstance(source_key, str) or len(source_key) != 64:
            raise ValueError("invalid source cache key")
        if any(char not in "0123456789abcdef" for char in source_key):
            raise ValueError("invalid source cache key")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:
        log.warning("Ignoring compiled kernel cache entry %s: %s", path, error)
        return None
    source = load_generated_code(source_key)
    if source is None:
        return None
    _mark_used(path)
    return config, normalized, source_key, source


def save_compiled_kernel(
    key: str, config: Config, normalized: Config, source_key: str
) -> None:
    if should_skip_cache():
        return
    path = get_helion_cache_dir() / "generated_code" / "bindings" / f"{key}.json"
    entry = {
        "config": config.to_json(),
        "normalized_config": normalized.to_json(),
        "source_key": source_key,
    }
    _save_entry(
        path,
        {
            "payload": entry,
            "sha256": hashlib.sha256(
                json.dumps(entry, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    )


def source_guard(source: Source, arguments: dict[str, int]) -> list[object] | None:
    """Serialize an argument projection read by ``BoundKernel._specialize_extra``."""
    if (
        isinstance(source, TensorPropertySource)
        and source.prop in (TensorProperty.SIZE, TensorProperty.STRIDE)
        and source.idx is not None
    ):
        base = source_guard(source.base, arguments)
        kind = "size" if source.prop == TensorProperty.SIZE else "stride"
        return None if base is None else [kind, base, source.idx]
    if (
        isinstance(source, GetItemSource)
        and isinstance(source.index, (int, str))
        and not source.index_is_slice
    ):
        base = source_guard(source.base, arguments)
        return None if base is None else ["item", base, source.index]
    if isinstance(source, LocalSource) and source.local_name in arguments:
        return ["arg", arguments[source.local_name]]
    return None


def _serialized_guard(
    guard_source: GuardSource, arguments: dict[str, int], backend: str
) -> list[object] | None:
    """Serialize what one ``BoundKernel._specialize_extra`` extractor reads.

    ``None`` marks extractors the cache cannot rebuild in a later process.
    """
    if guard_source is None:
        return None
    source, guard, extent_cap = guard_source
    tensor_guard = source_guard(source, arguments)
    if tensor_guard is None or guard is None:
        return tensor_guard
    if isinstance(guard, TensorDescriptorAlignmentGuard):
        return ["td_alignment", tensor_guard, guard.requires_zero_storage_offset]
    if backend != "triton":
        return None
    return ["td_layout", tensor_guard, guard.ndim, guard.element_size, extent_cap]


def serialized_fact(
    fact: _GuardedInputFact, arguments: dict[str, int]
) -> list[object] | None:
    """Serialize a guard fact on an input size or stride and a constant or another one.

    ``None`` marks other facts, such as one on a reduction extent expression.
    """
    sources = dict(fact.sources)

    def operand(expr: sympy.Expr) -> list[object] | int | None:
        if isinstance(expr, sympy.Integer):
            return int(expr)
        source = sources.get(cast("sympy.Symbol", expr))
        return None if source is None else source_guard(source, arguments)

    lhs = operand(fact.lhs)
    rhs = operand(fact.rhs)
    if not isinstance(lhs, list) or rhs is None:
        return None
    return ["fact", lhs, rhs, fact.equal, fact.block]


def _valid_fact_operand(operand: object) -> bool:
    match operand:
        case ["size" | "stride", ["arg", int()], int()]:
            return True
    return type(operand) is int


def _valid_guard(guard: object) -> bool:
    match guard:
        case ["arg", int()]:
            return True
        case ["fact", list() as lhs, rhs, bool(), bool()]:
            return _valid_fact_operand(lhs) and _valid_fact_operand(rhs)
        case ["item", base, int() | str()]:
            return _valid_guard(base)
        case ["size" | "stride", base, int()]:
            return _valid_guard(base)
        case ["td_layout", base, int(), int(), int() | None]:
            return _valid_guard(base)
        case ["td_alignment", base, bool()]:
            return _valid_guard(base)
    return False


def _schema_path(key: str) -> Path:
    return get_helion_cache_dir() / "generated_code" / "schemas" / f"{key}.json"


def load_binding_schema(key: str) -> dict[str, object] | None:
    path = _schema_path(key)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        schema = envelope["payload"]
        checksum = hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if envelope["sha256"] != checksum:
            raise ValueError("binding schema checksum mismatch")
        if not all(_valid_guard(guard) for guard in schema["guards"]):
            raise ValueError("invalid binding guard")
        for fact in schema["compiler_seed_facts"]:
            match fact:
                case [
                    "config_num_sm" | "device_num_sm" | "input_tensor_metadata",
                    int(),
                ]:
                    pass
                case _:
                    raise ValueError("invalid compiler seed fact")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:
        log.warning("Ignoring binding schema cache entry %s: %s", path, error)
        return None
    _mark_used(path)
    return schema


def save_binding_schema(key: str, schema: dict[str, object]) -> None:
    """Publish a checked guard schema shared by binding and source lookups."""
    _save_entry(
        _schema_path(key),
        {
            "payload": schema,
            "sha256": hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    )


def binding_variant_key(
    key: str,
    schema: dict[str, object],
    extra_results: tuple[Hashable, ...],
    compiler_seed_results: tuple[Hashable, ...],
) -> str:
    """Identify one binding by the guard results its frontend specialized on."""
    payload = (
        "helion-binding-variant-v1",
        key,
        json.dumps(schema, sort_keys=True),
        extra_results,
        compiler_seed_results,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def save_binding(
    bound: BoundKernel, config: Config, normalized: Config, source_key: str
) -> None:
    """Save a binding with the guards that select it before the frontend runs.

    Kernels whose specialization includes a runtime input classifier, or a
    guard fact the cache cannot serialize, keep the ordinary frontend path.
    """
    state = bound._generated_code
    if state is None or state.artifact_key is None or should_skip_cache():
        return
    if (dependencies := _dependency_snapshot(bound)) is None:
        return
    key = state.artifact_key
    signature, own_guards, guard_results, compiler_seed_results, _ = state.input_key
    guards = bound.kernel._published_guards.get(signature)
    facts = bound._guard_fact_results()
    # The binding is only valid under the schema its signature published in
    # this process: its guards, then the guard facts (see
    # BoundKernel._guarded_input_facts) bindings of the signature fired so far,
    # keyed by their values here. All of it must be serializable (no ``None``).
    if (
        guards is None
        or None in guards
        or facts is None
        or guards[: len(own_guards)] != own_guards
        or len(guards) != len(own_guards) + len(facts)
    ):
        return
    # One schema per signature, and the last writer wins. Facts published
    # later extend it; earlier variants stay reachable (see load_cached_kernel).
    # Should bindings of one signature ever save different guard lists, the
    # earlier variants (keyed by the schema JSON) become unreachable until
    # eviction removes them.
    schema: dict[str, object] = {
        "dependencies": dependencies,
        "guards": list(guards),
        "compiler_seed_facts": [
            [extractor.fact, extractor.reserved_sms]
            for extractor in bound._compiler_seed_specialization_extractors
        ],
    }
    save_binding_schema(key, schema)
    save_compiled_kernel(
        binding_variant_key(
            key,
            schema,
            (*guard_results, *(value for _, value in facts)),
            compiler_seed_results,
        ),
        config,
        normalized,
        source_key,
    )


def generated_code_cache_key(bound: BoundKernel, config: Config) -> str | None:
    """Identify source only after the frontend has established specialization.

    Plain Triton kernels do not need codegen to discover symbolic descriptor
    guards. Other backends, distributed kernels and descriptor
    configs retain their ordinary codegen path.
    """
    return _generated_code_cache_key(bound, config, include_guard_facts=True)


def _generated_code_cache_key(
    bound: BoundKernel,
    config: Config,
    *,
    include_guard_facts: bool,
    guard_facts: tuple[tuple[str, Hashable], ...] | None = None,
) -> str | None:
    state = bound._generated_code
    if (
        bound.settings.cache_level != "code"
        or should_skip_cache()
        or bound.settings.backend != "triton"
        or state is None
    ):
        return None
    if bound.env._is_distributed:
        _warn_cache_unavailable(bound.kernel, "distributed kernels")
        return None
    if bound.env.runtime_input_specializations:
        _warn_cache_unavailable(bound.kernel, "runtime input specialization")
        return None
    if any(
        indexing_uses_tensor_descriptor(indexing)
        for indexing in (config.indexing, config.atomic_indexing)
    ):
        _warn_cache_unavailable(bound.kernel, "tensor-descriptor indexing")
        return None
    if (dependencies := _dependency_snapshot(bound)) is None:
        return None
    fn = bound.kernel.fn
    host_function = bound.host_function
    assert host_function is not None
    with bound.env:
        frontend = _without_addresses(host_function.debug_str())
    settings = _compilation_settings_key(bound.settings)
    facts = bound._guard_fact_results() if guard_facts is None else guard_facts
    if settings is _UNCACHEABLE:
        _warn_cache_unavailable(
            bound.kernel, "compilation settings contain an unsupported value"
        )
        return None
    if facts is None:
        _warn_cache_unavailable(bound.kernel, "unserializable codegen guard facts")
        return None
    payload = (
        "helion-generated-code-v1"
        if include_guard_facts
        else "helion-generated-code-discovery-v1",
        helion_key(),
        torch_key_wrapper(),
        triton_key_wrapper(),
        fn.__module__,
        fn.__code__.co_filename,
        fn.__code__.co_firstlineno,
        bound.kernel.kernel_source(),
        _without_addresses(state.input_key),
        get_device_name(bound.env.device),
        bound.config_spec.target_device_capability,
        bound.config_spec.num_sm,
        str(bound.env.index_dtype),
        frontend,
        dependencies,
        bound.config_spec.cache_fingerprint_hash(),
        config.to_json(),
        settings,
        facts if include_guard_facts else (),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


@dataclasses.dataclass
class GeneratedCodeState:
    """A binding's generated code cache keys and the entries it persisted."""

    input_key: GeneratedCodeInputKey
    # Key of the binding's pre-frontend entries; None when they are not saved.
    artifact_key: str | None
    source_keys: dict[Config, str] = dataclasses.field(default_factory=dict)
    # Disk entries this binding already wrote or loaded. Repeated set_config
    # calls skip the writers' lock and eviction scan.
    persisted_source_keys: set[str] = dataclasses.field(default_factory=set)
    persisted_source_links: set[tuple[str, str]] = dataclasses.field(
        default_factory=set
    )
    persisted_binding: tuple[Config, str] | None = None

    @classmethod
    def create(
        cls,
        bound: BoundKernel,
        args: Sequence[object],
        artifact_key: str | None,
        input_key: str | None,
    ) -> GeneratedCodeState | None:
        """The state of a binding the frontend just compiled, or None if ineligible."""
        capture = bound.env.cache_dependencies
        if _dependency_snapshot(bound) is None:
            if capture is not None:
                _warn_cache_unavailable(
                    bound.kernel, capture.reason or "untracked dependencies"
                )
            return None
        if input_key is None:
            input_key = binding_input_key(bound.kernel, args)
        if input_key is None:
            return None
        guard_sources: list[GuardSource] = []
        extractors = bound._specialize_extra(guard_sources)
        assert len(guard_sources) == len(extractors), (
            "generated-code guards must match specialization extractors"
        )
        arguments = {
            name: index
            for index, name in enumerate(bound.kernel.signature.parameters.keys())
        }
        guards = tuple(
            _serialized_guard(guard_source, arguments, bound.env.backend_name)
            for guard_source in guard_sources
        )
        return cls(
            (
                bound._base_spec_key,
                guards,
                tuple(extractor(args) for extractor in extractors),
                bound._compiler_seed_specialization_results,
                input_key,
            ),
            artifact_key,
        )

    @property
    def guards(self) -> tuple[list[object] | None, ...]:
        """The specialization guards as the cache saves them."""
        return self.input_key[1]

    def load(self, bound: BoundKernel, config: Config) -> str | None:
        """Saved source for a normalized config that ``compile_and_save`` keyed."""
        if _has_active_torch_mode():
            if (capture := bound.env.cache_dependencies) is not None:
                capture.reject("active Torch dispatch modes cannot be certified")
            return None
        if (source_key := self.source_keys.get(config)) is None:
            return None
        source = load_generated_code(source_key)
        if source is not None:
            self.persisted_source_keys.add(source_key)
            return source
        # Code generation can add size/stride facts after the frontend. The
        # source key includes them, so discover the saved variant separately
        # and restore those facts before the generated module can run.
        lookup_key = _generated_code_cache_key(bound, config, include_guard_facts=False)
        if lookup_key is None or (schema := load_binding_schema(lookup_key)) is None:
            return None
        if not validate_kernel_dependencies(
            schema.get("dependencies"), bound.kernel.fn
        ):
            return None
        guards = cast("list[list[object]]", schema["guards"])
        if schema["compiler_seed_facts"] or any(guard[0] != "fact" for guard in guards):
            return None
        from .cached_kernel import _guard_fact
        from .kernel import PreparedMetadataSpecializationExtractor

        try:
            param_names = tuple(bound.kernel.signature.parameters)
            facts = tuple(_guard_fact(guard, param_names) for guard in guards)
            results = tuple(bound._guard_fact_value(fact) for fact in facts)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            log.warning(
                "Ignoring generated source guard schema %s: %s", lookup_key, error
            )
            return None
        if None in results:
            return None
        artifact = _load_compiled_kernel_entry(
            binding_variant_key(lookup_key, schema, results, ())
        )
        if artifact is None:
            return None
        selected, normalized, source_key, source = artifact
        extractors = [PreparedMetadataSpecializationExtractor(fact) for fact in facts]
        expected_key = _generated_code_cache_key(
            bound,
            config,
            include_guard_facts=True,
            guard_facts=tuple(zip(map(repr, extractors), results, strict=True)),
        )
        if selected != config or normalized != config or source_key != expected_key:
            log.warning("Ignoring inconsistent generated source entry %s", lookup_key)
            return None
        bound.kernel._publish_guard_facts(bound, extractors)
        self.source_keys[config] = source_key
        self.persisted_source_keys.add(source_key)
        self.persisted_source_links.add((lookup_key, source_key))
        return source

    def _save_source_link(
        self, bound: BoundKernel, config: Config, source_key: str
    ) -> None:
        """Index source by its codegen facts, without choosing a tuning config."""
        facts = bound._guard_fact_results()
        if not facts:
            return
        guards = bound.kernel._published_guards.get(bound._base_spec_key)
        if guards is None:
            return
        fact_guards = guards[len(self.guards) :]
        if len(fact_guards) != len(facts) or any(
            guard is None or guard[0] != "fact" for guard in fact_guards
        ):
            _warn_cache_unavailable(bound.kernel, "unserializable codegen guard facts")
            return
        lookup_key = _generated_code_cache_key(bound, config, include_guard_facts=False)
        if (
            lookup_key is None
            or (lookup_key, source_key) in self.persisted_source_links
        ):
            return
        if (dependencies := _dependency_snapshot(bound)) is None:
            return
        schema: dict[str, object] = {
            "dependencies": dependencies,
            "guards": list(fact_guards),
            "compiler_seed_facts": [],
        }
        save_binding_schema(lookup_key, schema)
        save_compiled_kernel(
            binding_variant_key(
                lookup_key, schema, tuple(value for _, value in facts), ()
            ),
            config,
            config,
            source_key,
        )
        self.persisted_source_links.add((lookup_key, source_key))

    def compile_and_save(self, bound: BoundKernel, config: Config) -> CompiledConfig:
        """Compile the config ``set_config`` selected and persist its source.

        Only a selected config is saved: autotuning candidates compiled with
        ``compile_config`` stay off disk.
        """
        if _has_active_torch_mode():
            if (capture := bound.env.cache_dependencies) is not None:
                capture.reject("active Torch dispatch modes cannot be certified")
        normalized = bound._normalized_config_copy(config)
        source_key = None
        if not bound.settings.print_output_code:
            source_key = self.source_keys.get(normalized)
            if source_key is None:
                source_key = generated_code_cache_key(bound, normalized)
            if source_key is not None:
                self.source_keys[normalized] = source_key
        published = bound.kernel._specialize_extra.get(bound._base_spec_key)
        run = bound.compile_config(config)
        if _dependency_snapshot(bound) is None:
            return run
        if (
            source_key is not None
            and source_key not in self.persisted_source_keys
            and bound.kernel._specialize_extra.get(bound._base_spec_key)
            is not published
        ):
            # Code generation published guard facts; key them too.
            source_key = generated_code_cache_key(bound, normalized)
            if source_key is None:
                del self.source_keys[normalized]
            else:
                self.source_keys[normalized] = source_key
        # compile_config wrote the source now or, for an autotuning winner,
        # earlier. Persist that file without generating it again.
        if (
            source_key is not None
            and source_key not in self.persisted_source_keys
            and (path := bound._cache_path_map.get(normalized)) is not None
        ):
            # PyCodeCache writes the supplied source unchanged; its extra key
            # changes the module identity, not the stored source text.
            try:
                source = Path(path).read_text(encoding="utf-8")
            except OSError as error:
                log.warning("Could not read generated code %s: %s", path, error)
            else:
                save_generated_code(source_key, source)
                self.persisted_source_keys.add(source_key)
        if source_key is not None and source_key in self.persisted_source_keys:
            self._save_source_link(bound, normalized, source_key)
        if (
            self.artifact_key is not None
            and (source_key := self.source_keys.get(normalized)) is not None
            and self.persisted_binding != (config, source_key)
        ):
            save_binding(bound, config, normalized, source_key)
            self.persisted_binding = (config, source_key)
        return run


def _cache_path(key: str) -> Path:
    return get_helion_cache_dir() / "generated_code" / f"{key}.json"


def load_generated_code(key: str) -> str | None:
    """Read a complete source entry, treating missing or corrupt files as misses."""
    path = _cache_path(key)
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        source = entry["source"]
        if not isinstance(source, str) or not source:
            raise ValueError("invalid generated source")
        if entry["sha256"] != hashlib.sha256(source.encode("utf-8")).hexdigest():
            raise ValueError("generated source checksum mismatch")
    except FileNotFoundError:
        counters["generated_code_cache"]["miss"] += 1
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:
        log.warning("Ignoring generated code cache entry %s: %s", path, error)
        counters["generated_code_cache"]["miss"] += 1
        return None
    _mark_used(path)
    counters["generated_code_cache"]["hit"] += 1
    return source


def _mark_used(path: Path) -> None:
    """Refresh a hit's mtime so eviction removes the least recently used entries.

    Readers take no lock. A concurrent eviction or a read-only cache directory
    only loses recency information.
    """
    try:
        os.utime(path)
    except OSError:
        log.debug("Could not update generated code cache entry %s", path)


def save_generated_code(key: str, source: str) -> None:
    """Atomically publish source so concurrent processes cannot read partial JSON."""
    _save_entry(
        _cache_path(key),
        {
            "source": source,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        },
    )


def _save_entry(path: Path, entry: dict[str, object]) -> None:
    if should_skip_cache():
        return
    temporary: Path | None = None
    try:
        limit = _cache_limit()
        data = json.dumps(entry).encode("utf-8")
        if len(data) > limit:
            return
        if _file_contents(path) == data:
            # Workers starting together write the same entry; a hit's mtime
            # refresh is all the existing entry needs.
            _mark_used(path)
            return
        root = get_helion_cache_dir() / "generated_code"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize publication and eviction across threads/processes. Readers
        # need no lock: an evicted artifact is an ordinary cache miss.
        with FileLock(root / ".lock", timeout=5):
            _reserve_space(root, path, len(data), limit)
            # Like the config cache, use ordinary file permissions filtered by
            # the process umask, so image-build caches can be read by a service
            # user. Exclusive creation prevents following a pre-existing link.
            candidate = path.parent / f".{uuid.uuid4().hex}.json.tmp"
            with candidate.open("xb") as output:
                temporary = candidate
                output.write(data)
            os.replace(temporary, path)
    except Timeout:
        # Another writer holds the lock; the entry is usually already published.
        log.debug("Skipped generated code cache entry %s: lock timeout", path)
    except OSError as error:
        log.warning("Could not write generated code cache entry %s: %s", path, error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                log.debug("Could not remove temporary cache file %s", temporary)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _file_contents(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _reserve_space(root: Path, target: Path, incoming: int, limit: int) -> None:
    """Keep source and binding JSON within the shared byte budget on writes.

    Writers maintain a running total under the lock, so a write scans the cache
    only when the total could exceed the budget, the index is missing or
    corrupt, or a periodic rescan is due. The index is updated before the entry
    is published: an interrupted or failed write leaves an overestimate, which
    forces the next writer to rescan. Files removed manually have the same
    effect, and the periodic rescan corrects files added outside the writers.
    """
    index = root / _SIZE_INDEX
    try:
        total, writes = (
            int(value) for value in index.read_text(encoding="utf-8").split()
        )
    except (OSError, ValueError):
        total, writes = -1, _SIZE_INDEX_RESCAN_WRITES
    total -= _file_size(target)
    if total < 0 or total + incoming > limit or writes >= _SIZE_INDEX_RESCAN_WRITES:
        total = _evict_entries(root, target, incoming, limit)
        writes = 0
    else:
        total += incoming
    index.write_text(f"{total} {writes + 1}\n", encoding="utf-8")


def _evict_entries(root: Path, target: Path, incoming: int, limit: int) -> int:
    """Evict least recently used entries and return the resulting total."""
    entries: list[tuple[int, Path, int]] = []
    total = incoming
    for path in root.rglob("*.json*"):
        if path.name.endswith(".json.tmp"):
            # All writers hold the lock, so these can only be abandoned by an
            # interrupted writer. Its reservation counted the bytes until now.
            path.unlink(missing_ok=True)
            continue
        if path.suffix != ".json" or path == target:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            # Manual cleanup can race a write despite the writers' lock.
            continue
        entries.append((stat.st_mtime_ns, path, stat.st_size))
        total += stat.st_size
    # Hits refresh mtime, so the oldest mtime is the least recently used entry.
    for _, path, size in sorted(entries):
        if total <= limit:
            break
        path.unlink(missing_ok=True)
        total -= size
    return total
