"""Persistent Triton source and conservative pre-frontend binding caches."""

from __future__ import annotations

import dataclasses
import dis
import enum
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import TYPE_CHECKING
from typing import cast

from filelock import FileLock
from filelock import Timeout
import torch
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from .._argument_device import _find_argument_device
from .._compat import get_device_name
from .._compat import supports_torch_compile_fusion
from .._utils import counters
from .._utils import indexing_uses_tensor_descriptor
from ..autotuner.base_cache import helion_key
from ..autotuner.base_cache import should_skip_cache
from ..autotuner.base_cache import torch_key_wrapper
from ..autotuner.base_cache import triton_key_wrapper
from ..autotuner.local_cache import get_helion_cache_dir
from ..language.constexpr import ConstExpr
from .config import Config
from .ref_mode import is_ref_mode_enabled
from .settings import default_autotuner_fn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Hashable

    from torch._guards import Source

    from .kernel import BoundKernel
    from .kernel import Kernel
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

log = logging.getLogger(__name__)
_UNCACHEABLE = object()


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


def _global_names(code: types.CodeType) -> set[str]:
    """Include globals loaded only by nested functions and comprehensions."""
    names = {
        cast("str", instruction.argval)
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
    }
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_global_names(constant))
    return names


def _function_dependencies(fn: types.FunctionType) -> object:
    try:
        closure = inspect.getclosurevars(fn)
    except ValueError:
        # A deleted nonlocal leaves an empty closure cell.
        return _UNCACHEABLE
    dependencies: dict[str, object] = {}
    for name in sorted(_global_names(fn.__code__)):
        if name in fn.__globals__:
            dependencies[name] = fn.__globals__[name]
        elif name in fn.__builtins__:
            dependencies[name] = fn.__builtins__[name]
        else:
            # A missing global could be installed before compilation or resolved
            # by dynamic execution. Its dependency cannot be established here.
            return _UNCACHEABLE
    # Dependency discovery order can vary with Python's hash seed.
    # Variable bindings are unordered; user-provided dict inputs are not.
    return dict(sorted(closure.nonlocals.items())), dict(sorted(dependencies.items()))


def _dependency_key(value: object, seen: frozenset[int] = frozenset()) -> object:
    """Fingerprint supported Python dependencies without pickling live objects."""
    if value is None or type(value) in (int, float, bool, str, bytes):
        return type(value).__name__, value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, enum.Enum):
        item = _dependency_key(value.value, seen)
        if item is _UNCACHEABLE:
            return _UNCACHEABLE
        return "enum", type(value).__module__, type(value).__qualname__, item
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
    if isinstance(value, types.ModuleType):
        name = value.__name__
        if name.split(".")[0] in {"torch", "helion", "math", "operator", "builtins"}:
            return "module", name
        return _UNCACHEABLE
    if isinstance(value, types.BuiltinFunctionType):
        if value.__module__ == "builtins" and value.__name__ in {
            "eval",
            "exec",
            "globals",
        }:
            return _UNCACHEABLE
        return "builtin", value.__module__, value.__qualname__
    if isinstance(value, type) and value.__module__.split(".")[0] in {
        "helion",
        "torch",
        "builtins",
    }:
        return "class", value.__module__, value.__qualname__
    if isinstance(value, types.FunctionType):
        if id(value) in seen:
            return "recursive", value.__module__, value.__qualname__
        seen = seen | {id(value)}
        bindings = _function_dependencies(value)
        if bindings is _UNCACHEABLE:
            return _UNCACHEABLE
        dependencies = _dependency_key(bindings, seen)
        defaults = _dependency_key((value.__defaults__, value.__kwdefaults__), seen)
        if dependencies is _UNCACHEABLE or defaults is _UNCACHEABLE:
            return _UNCACHEABLE
        return (
            "function",
            value.__module__,
            value.__qualname__,
            _code_key(value.__code__),
            defaults,
            dependencies,
        )
    return _UNCACHEABLE


def _compilation_settings_key(settings: Settings) -> object:
    # Search policy selects a config already included in the source key. Reject
    # unknown compilation values instead of hashing an address-bearing repr.
    # Read fields directly: to_dict() deep-copies arbitrary future values first.
    return _dependency_key(
        dict(
            sorted(
                (field.name, getattr(settings, field.name))
                for field in dataclasses.fields(settings)
                if field.repr and not field.name.startswith("autotun")
            )
        )
    )


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
        return "constexpr", _dependency_key(value.value)
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
        not kernel.settings.generated_code_cache
        or kernel.settings.backend != "triton"
        or torch.compiler.is_compiling()
    ):
        return None
    dependencies = _dependency_key(kernel.fn)
    inputs = _argument_key(tuple(args))
    if dependencies is _UNCACHEABLE or inputs is _UNCACHEABLE:
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
        return None
    device = _find_argument_device(args)
    hardware = (
        get_device_name(device)
        if device is not None and device.type in {"cuda", "xpu"}
        else str(device)
    )
    payload = (
        "helion-bound-source-v2",
        sys.version_info[:3],
        helion_key(),
        torch_key_wrapper(),
        triton_key_wrapper(),
        inputs,
        base_spec_key,
        hardware,
        settings,
        kernel.settings.autotune_effort,
        str(torch.get_default_dtype()),
        str(torch.get_default_device()),
        torch.get_float32_matmul_precision(),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        str(torch.get_autocast_dtype("cuda")),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def load_compiled_kernel(key: str) -> tuple[Config, Config, str] | None:
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
    counters["generated_code_cache"]["frontend_hit"] += 1
    return config, normalized, source


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


def _valid_guard(guard: object) -> bool:
    match guard:
        case ["arg", int()]:
            return True
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


def load_binding_schema(key: str) -> dict[str, list[object]] | None:
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
    return schema


def binding_variant_key(
    key: str,
    schema: dict[str, list[object]],
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

    Kernels whose specialization includes a runtime input classifier have no
    serializable guard and keep the ordinary frontend path.
    """
    key = bound._generated_code_artifact_key
    input_key = bound._generated_code_input_key
    if key is None or input_key is None or should_skip_cache():
        return
    _, guards, extra_results, compiler_seed_results, _ = input_key
    if None in guards:
        return
    schema: dict[str, list[object]] = {
        "guards": list(guards),
        "compiler_seed_facts": [
            [extractor.fact, extractor.reserved_sms]
            for extractor in bound._compiler_seed_specialization_extractors
        ],
    }
    _save_entry(
        _schema_path(key),
        {
            "payload": schema,
            "sha256": hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    )
    save_compiled_kernel(
        binding_variant_key(key, schema, extra_results, compiler_seed_results),
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
    if (
        not bound.settings.generated_code_cache
        or should_skip_cache()
        or bound.settings.backend != "triton"
        or bound.env._is_distributed
        or bound.env.runtime_input_specializations
        or bound._generated_code_input_key is None
    ):
        return None
    if any(
        indexing_uses_tensor_descriptor(indexing)
        for indexing in (config.indexing, config.atomic_indexing)
    ):
        return None
    fn = bound.kernel.fn
    host_function = bound.host_function
    assert host_function is not None
    with bound.env:
        frontend = host_function.debug_str()
    settings = _compilation_settings_key(bound.settings)
    if settings is _UNCACHEABLE:
        return None
    payload = (
        "helion-generated-code-v1",
        helion_key(),
        torch_key_wrapper(),
        triton_key_wrapper(),
        fn.__module__,
        fn.__code__.co_filename,
        fn.__code__.co_firstlineno,
        bound.kernel.kernel_source(),
        bound._generated_code_input_key,
        get_device_name(bound.env.device),
        bound.config_spec.target_device_capability,
        bound.config_spec.num_sm,
        str(bound.env.index_dtype),
        frontend,
        bound.config_spec.cache_fingerprint_hash(),
        config.to_json(),
        settings,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


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
    counters["generated_code_cache"]["hit"] += 1
    return source


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
        limit = int(
            os.environ.get("HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES", str(1 << 30))
        )
        if limit < 0:
            raise ValueError("HELION_GENERATED_CODE_CACHE_MAX_SIZE_BYTES must be >= 0")
        data = json.dumps(entry).encode("utf-8")
        if len(data) > limit:
            return
        root = get_helion_cache_dir() / "generated_code"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize publication and eviction across threads/processes. Readers
        # need no lock: an evicted artifact is an ordinary cache miss.
        with FileLock(root / ".lock", timeout=1):
            _evict_entries(root, path, len(data), limit)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(data)
            os.replace(temporary, path)
    except (OSError, Timeout, ValueError) as error:
        log.warning("Could not write generated code cache entry %s: %s", path, error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                log.debug("Could not remove temporary cache file %s", temporary)


def _evict_entries(root: Path, target: Path, incoming: int, limit: int) -> None:
    """Keep source and binding JSON within the shared byte budget on writes."""
    entries: list[tuple[int, Path, int]] = []
    total = incoming
    for path in root.rglob("*.json"):
        if path == target:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            # Manual cleanup can race a write despite the writers' lock.
            continue
        entries.append((stat.st_mtime_ns, path, stat.st_size))
        total += stat.st_size
    for _, path, size in sorted(entries):
        if total <= limit:
            break
        path.unlink(missing_ok=True)
        total -= size
