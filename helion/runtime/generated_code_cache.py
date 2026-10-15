"""Persistent Triton source and conservative pre-frontend binding caches."""

from __future__ import annotations

import dataclasses
import dis
import enum
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import types
from typing import TYPE_CHECKING
from typing import cast

from filelock import FileLock
from filelock import Timeout
import sympy
import torch
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from .._argument_device import _find_argument_device
from .._compat import device_num_sm
from .._compat import get_device_name
from .._compat import supports_torch_compile_fusion
from .._compat import target_device_capability
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


# Settings that cannot change generated code. Search policy selects a config
# already included in the source key, and the cache toggle is always enabled
# when a key is computed. Every other field, including future ones, is keyed.
_NON_CODEGEN_SETTING_PREFIXES = ("autotun",)
_NON_CODEGEN_SETTINGS = frozenset({"generated_code_cache"})


def _compilation_settings_key(settings: Settings) -> object:
    # Reject unknown compilation values instead of hashing an address-bearing
    # repr. Read fields directly: to_dict() deep-copies arbitrary values first.
    return _dependency_key(
        dict(
            sorted(
                (field.name, getattr(settings, field.name))
                for field in dataclasses.fields(settings)
                if field.name not in _NON_CODEGEN_SETTINGS
                and not field.name.startswith(_NON_CODEGEN_SETTING_PREFIXES)
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
    hardware: tuple[object, ...]
    if device is not None and device.type == "cuda":
        # The source key bakes in the compute capability and SM count; devices
        # sharing a name can differ in both (partitioned or cloud SKUs).
        capability = target_device_capability(device)
        if capability is None:
            return None
        hardware = (get_device_name(device), capability, device_num_sm(device))
    elif device is not None and device.type == "xpu":
        hardware = (get_device_name(device),)
    else:
        hardware = (str(device),)
    payload = (
        "helion-bound-source-v3",
        sys.version_info[:3],
        helion_key(),
        torch_key_wrapper(),
        triton_key_wrapper(),
        inputs,
        _without_addresses(base_spec_key),
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
    _mark_used(path)
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
    _mark_used(path)
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

    Kernels whose specialization includes a runtime input classifier, or a
    guard fact the cache cannot serialize, keep the ordinary frontend path.
    """
    state = bound._generated_code
    if state is None or state.artifact_key is None or should_skip_cache():
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
    state = bound._generated_code
    if (
        not bound.settings.generated_code_cache
        or should_skip_cache()
        or bound.settings.backend != "triton"
        or bound.env._is_distributed
        or bound.env.runtime_input_specializations
        or state is None
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
        frontend = _without_addresses(host_function.debug_str())
    settings = _compilation_settings_key(bound.settings)
    facts = bound._guard_fact_results()
    if settings is _UNCACHEABLE or facts is None:
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
        _without_addresses(state.input_key),
        get_device_name(bound.env.device),
        bound.config_spec.target_device_capability,
        bound.config_spec.num_sm,
        str(bound.env.index_dtype),
        frontend,
        bound.config_spec.cache_fingerprint_hash(),
        config.to_json(),
        settings,
        facts,
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
        if input_key is None:
            input_key = binding_input_key(bound.kernel, args)
        if input_key is None:
            return None
        guard_sources: list[GuardSource] = []
        extractors = bound._specialize_extra(guard_sources)
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

    def load(self, config: Config) -> str | None:
        """Saved source for a normalized config that ``compile_and_save`` keyed."""
        if (source_key := self.source_keys.get(config)) is None:
            return None
        source = load_generated_code(source_key)
        if source is not None:
            self.persisted_source_keys.add(source_key)
        return source

    def compile_and_save(self, bound: BoundKernel, config: Config) -> CompiledConfig:
        """Compile the config ``set_config`` selected and persist its source.

        Only a selected config is saved: autotuning candidates compiled with
        ``compile_config`` stay off disk.
        """
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
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, delete=False
            ) as output:
                temporary = Path(output.name)
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
    # Hits refresh mtime, so the oldest mtime is the least recently used entry.
    for _, path, size in sorted(entries):
        if total <= limit:
            break
        path.unlink(missing_ok=True)
        total -= size
    return total
