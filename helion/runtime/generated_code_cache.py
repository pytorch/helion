"""Persistent Triton source and conservative pre-frontend binding caches."""

from __future__ import annotations

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

import torch

from .._argument_device import _find_argument_device
from .._compat import get_device_name
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

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Hashable

    from .kernel import BoundKernel
    from .kernel import Kernel

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


def _dependency_key(value: object, seen: frozenset[int] = frozenset()) -> object:
    """Fingerprint supported Python dependencies without pickling live objects."""
    if value is None or type(value) in (int, float, bool, str, bytes):
        return type(value).__name__, value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
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
        closure = inspect.getclosurevars(value)
        dependencies = _dependency_key({**closure.globals, **closure.nonlocals}, seen)
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


def _argument_key(value: object) -> object:
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        tensor = cast("torch.Tensor", value)
        return (
            "tensor",
            str(tensor.dtype),
            str(tensor.device),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr() % 16,
            tensor.requires_grad,
            tensor.is_inference(),
            tuple(sorted(getattr(tensor, "_dynamo_static_indices", ()))),
        )
    if isinstance(value, ConstExpr):
        return _argument_key(value.value)
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


def exact_input_key(kernel: Kernel, args: Sequence[object]) -> Hashable | None:
    """A conservative guard usable before frontend specialization is discovered."""
    if not kernel.settings.generated_code_cache or torch.compiler.is_compiling():
        return None
    dependencies = _dependency_key(kernel.fn)
    inputs = _argument_key(tuple(args))
    if dependencies is _UNCACHEABLE or inputs is _UNCACHEABLE:
        return None
    return hashlib.sha256(repr((dependencies, inputs)).encode("utf-8")).hexdigest()


def compiled_kernel_cache_key(
    kernel: Kernel, args: Sequence[object], base_spec_key: tuple[Hashable, ...]
) -> str | None:
    if (
        not kernel.settings.generated_code_cache
        or kernel.settings.backend != "triton"
        or kernel.settings.force_autotune
        or kernel.settings.print_output_code
        or kernel.settings.print_repro
        or is_ref_mode_enabled(kernel.settings)
        or should_skip_cache()
        or torch.compiler.is_compiling()
    ):
        return None
    inputs = exact_input_key(kernel, args)
    if inputs is None:
        return None
    settings = {
        name: value
        for name, value in kernel.settings.to_dict().items()
        if not name.startswith("autotun")
    }
    device = _find_argument_device(args)
    hardware = (
        get_device_name(device)
        if device is not None and device.type in {"cuda", "xpu"}
        else str(device)
    )
    payload = (
        "helion-bound-source-v1",
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
    # Search policy determines the selected config, which is already in the key.
    # In particular, the default time-based autotune seed must not prevent reuse.
    settings = {
        name: value
        for name, value in bound.settings.to_dict().items()
        if not name.startswith("autotun")
    }
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
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(entry, output)
        os.replace(temporary, path)
    except OSError as error:
        log.warning("Could not write generated code cache entry %s: %s", path, error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                log.debug("Could not remove temporary cache file %s", temporary)
