from __future__ import annotations

import ast
from collections import OrderedDict
from collections.abc import Sequence
import contextlib
import copy
import dataclasses
import functools
import hashlib
import inspect
import itertools
import logging
import os
import re
import struct
import sys
import textwrap
import threading
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generic
from typing import Hashable
from typing import Literal
from typing import NamedTuple
from typing import TypeVar
from typing import cast
from typing import overload
from typing_extensions import Protocol
import weakref

import sympy
import torch
from torch._dynamo.source import AttrSource
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codecache import compiled_fx_graph_hash
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
import torch.distributed as dist
from torch.utils._pytree import tree_map_only
from torch.utils.weak import WeakIdKeyDictionary

from .. import exc
from .._argument_device import _ArgumentDeviceResolver
from .._argument_device import _canonicalize_argument_device
from .._argument_device import _current_device_index
from .._argument_device import _find_argument_device as _find_device
from .._compat import shape_env_size_hint
from .._compat import target_device_capability
from .._compile_time import measure
from .._compiler.ast_extension import unparse
from .._compiler.autotuner_heuristics import compiler_promotion_specialization_key
from .._compiler.autotuner_heuristics import compiler_seed_configs
from .._compiler.autotuner_heuristics import compiler_seed_specialization_facts
from .._compiler.autotuner_heuristics import register_compiler_coverage_groups
from .._compiler.backend_registry import get_backend
from .._compiler.compile_environment import CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import _concrete_tensor_satisfies_alignment_guard
from .._compiler.compile_environment import _is_supported_tensor_input_source
from .._compiler.compile_environment import _replay_tensor_input_source
from .._compiler.compile_environment import _symint_free_symbols
from .._compiler.compile_environment import shape_env_var_hints
from .._compiler.compile_environment import (
    tensor_descriptor_layout_signature_from_strides,
)
from .._compiler.generate_ast import generate_ast
from .._compiler.inductor_lowering_extra import patch_inductor_lowerings
from .._compiler.kernel_compiler import KernelCompiler
from .._compiler.output_header import assert_no_conflicts
from .._compiler.runtime_tensor_guards import _CanonicalLayoutClassifier
from .._compiler.variable_origin import ArgumentOrigin
from .._compiler.variable_origin import BlockSizeOrigin
from .._dist_utils import _find_process_group_name
from .._dist_utils import check_config_consistancy as dist_check_config_consistancy
from .._dist_utils import kernel_declares_process_group
from .._dist_utils import kernel_uses_symm_mem
from .._logging import LazyString
from .._utils import _read_cell
from .._utils import counters
from .._utils import indexing_uses_tensor_descriptor
from .._utils import next_power_of_2
from ..autotuner.base_search import _AutotunableKernel
from ..autotuner.benchmark_provider import _SEARCHING_CANDIDATES
from ..autotuner.config_fragment import _BUILTIN_TUNABLE_FRAGMENTS
from ..autotuner.config_fragment import _immutable_fragment_key
from ..autotuner.config_fragment import _plain_builtin_fragment_key
from ..language.constexpr import ConstExpr
from .config import Config
from .config import _config_key
from .config_cache import ConfigCache
from .cute.structural_config import DEFAULT_STRUCTURAL_POLICY
from .cute.structural_config import CuteStructuralConfig
from .cute.structural_config import select_structural_policy
from .ref_mode import RefModeContext
from .ref_mode import is_ref_mode_enabled
from .settings import Settings
from .structural_policy import StructuralConfig
from .structural_policy import StructuralPolicyError
from .structural_policy import require_same_structural_policy

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch._guards import Source

    from .._compiler.autotuner_heuristics.registry import (
        CompilerHeuristicSpecializationFact,
    )
    from .._compiler.compile_environment import PreparedGuardFactory
    from .._compiler.compile_environment import RuntimeInputSpecialization
    from .._compiler.compile_environment import TensorDescriptorLayoutGuard
    from .._compiler.host_function import HostFunction
    from ..autotuner import ConfigSpec
    from ..autotuner.base_cache import BoundKernelInMemoryCacheKey
    from ..autotuner.config_fragment import ConfigSpecFragment
    from .cute.structural_config import CuteStructuralPolicyRequest
    from .cute.structural_policy import CuteStructuralPolicy

    ConfigLike = Config | dict[str, object] | CuteStructuralConfig

log: logging.Logger = logging.getLogger(__name__)


def _indexing_config_uses_tensor_descriptor(indexing: object, index: int) -> bool:
    if indexing == "tensor_descriptor":
        return True
    if isinstance(indexing, list):
        return index < len(indexing) and indexing[index] == "tensor_descriptor"
    return False


class _TensorDescriptorOperationGuard(Protocol):
    memory_op_indices: set[int]
    atomic_op_indices: set[int]


def _td_guard_active_for_config(
    guard: _TensorDescriptorOperationGuard,
    config: Config,
    descriptor_memory_ops: frozenset[int],
) -> bool:
    """Whether ``config`` indexes one of ``guard``'s memory ops through a
    tensor descriptor.  ``descriptor_memory_ops`` holds the loads and stores
    its code does, by their facts' ``indexing_index``
    (``BoundKernel._descriptor_memory_ops``)."""
    return not guard.memory_op_indices.isdisjoint(descriptor_memory_ops) or any(
        _indexing_config_uses_tensor_descriptor(config.atomic_indexing, index)
        for index in guard.atomic_op_indices
    )


_R = TypeVar("_R")
CompiledConfig = Callable[..., _R]


def _with_evaluated_defaults(
    function: types.FunctionType, signature: inspect.Signature
) -> types.FunctionType:
    """Attach public defaults without mutating a shared PyCodeCache function."""
    defaults: list[object] = []
    for name in function.__code__.co_varnames[: function.__code__.co_argcount]:
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            defaults.clear()
        else:
            defaults.append(parameter.default)
    if not defaults:
        return function
    result = types.FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        tuple(defaults),
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__dict__.update(function.__dict__)
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    return result


# Opt-in: auto-capture Pallas kernels under torch.compile (see
# pallas._tpu_compile_capture).
# Off by default so the eager dispatch path is unchanged.
_TPU_COMPILE_CAPTURE = os.environ.get("HELION_TPU_COMPILE_CAPTURE", "0") == "1"

# Cache for GraphModule hashes
_graph_module_hash_cache: WeakIdKeyDictionary = WeakIdKeyDictionary()

_INT32_INDEX_LIMIT = torch.iinfo(torch.int32).max
_CUSTOM_KEY_UNSET = object()
_OPAQUE_BINDING_CACHE_SIZE = 16
_FRESH_SELECTION_CACHE_SIZE = 16

_HostSemanticInputNormalization = tuple[tuple[int, str, int, int], ...]


class _FastDispatchEntry(NamedTuple):
    """A dispatch key together with the state used to build it."""

    key: tuple[Hashable, ...]
    extra_guards: tuple[tuple[Callable[[Sequence[object]], Hashable], Hashable], ...]
    specialization_generation: int


class _FastDispatchSchema:
    """What one exact fast-key prefix resolves to.

    The prefix (exact per-argument metadata, ``is_distributed``, the custom
    key and the input-alias partition) of a call the memo admits determines
    the base specialization signature for fixed kernel settings
    (``Kernel._memoized_fast_dispatch_key``), hence which published
    ``_specialize_extra`` list applies and the result of every extractor that
    exact metadata implies (``_implied_by_exact_metadata``).  ``results``
    holds the extras tuple of the call that resolved the prefix; ``per_call``
    names the slots every call re-evaluates (pointer alignment, runtime input
    classifiers, ``hl.specialize``'d scalar values, ...).
    """

    __slots__ = (
        "extra_fns",
        "full_key",
        "generation",
        "per_call",
        "results",
        "settings",
    )

    def __init__(
        self,
        generation: int,
        settings: tuple[Hashable, ...],
        extra_fns: list[Callable[[Sequence[object]], Hashable]],
        results: tuple[Hashable, ...],
        per_call: tuple[tuple[int, Callable[[Sequence[object]], Hashable]], ...],
        prefix: tuple[Hashable, ...],
    ) -> None:
        self.generation = generation
        self.settings = settings
        # Held so the list cannot be collected while this entry is alive.
        self.extra_fns = extra_fns
        self.results = results
        self.per_call = per_call
        # Without per-call slots, the whole fast key: one object shared by
        # every call, so dispatch-cache lookups compare it by identity.
        self.full_key: tuple[Hashable, ...] | None = None
        if not per_call:
            self.full_key = (*prefix, results) if extra_fns else prefix


def _nested_fast_dispatch_key(
    value: object, *, specialize_numeric: bool = False
) -> tuple[Hashable, bool, bool] | None:
    """Return a safe key, tensor presence, and canonicalized fragment presence."""
    value_type = type(value)
    if value_type is torch.Tensor or value_type is torch.nn.Parameter:
        if specialize_numeric:
            # A constexpr-annotated argument specializes on the raw container,
            # so tensor identity (not metadata) drives the full key; a
            # metadata-based fast key would conflate keys the full key
            # distinguishes.
            return None
        tensor = cast("torch.Tensor", value)
        static_indices = getattr(tensor, "_dynamo_static_indices", None)
        return (
            (
                tensor.dtype,
                tensor.shape,
                tensor.stride(),
                tensor.device,
                None if static_indices is None else frozenset(static_indices),
            ),
            True,
            False,
        )
    if value_type in (int, float, bool):
        if specialize_numeric and value_type is float:
            value = struct.pack("!d", cast("float", value))
        return ((value_type, value) if specialize_numeric else value_type), False, False
    if value_type is str:
        return (value_type, value), False, False
    if isinstance(value, ConstExpr):
        constexpr_key = _constexpr_key(value.value)
        try:
            hash(constexpr_key)
        except TypeError:
            return None
        return (
            (value_type, constexpr_key),
            False,
            type(value.value) is not float and constexpr_key is not value.value,
        )
    if specialize_numeric and value_type in _BUILTIN_TUNABLE_FRAGMENTS:
        constexpr_key = _constexpr_key(value)
        try:
            hash(constexpr_key)
        except TypeError:
            return None
        return (value_type, constexpr_key), False, constexpr_key is not value
    if value is None:
        return None, False, False
    if value_type is torch.dtype or value_type is torch.device:
        return cast("Hashable", value), False, False
    if value_type is tuple or value_type is list:
        items: list[Hashable] = []
        has_tensor = False
        has_fragment = False
        for item in cast("Sequence[object]", value):
            nested = _nested_fast_dispatch_key(
                item, specialize_numeric=specialize_numeric
            )
            if nested is None:
                return None
            item_key, item_has_tensor, item_has_fragment = nested
            items.append(item_key)
            has_tensor = has_tensor or item_has_tensor
            has_fragment = has_fragment or item_has_fragment
        return (value_type, tuple(items)), has_tensor, has_fragment
    if value_type is dict:
        entries: list[tuple[str | int, Hashable]] = []
        has_tensor = False
        has_fragment = False
        for item_key, item_value in cast("dict[object, object]", value).items():
            if type(item_key) not in (str, int):
                return None
            nested = _nested_fast_dispatch_key(
                item_value, specialize_numeric=specialize_numeric
            )
            if nested is None:
                return None
            value_key, value_has_tensor, value_has_fragment = nested
            entries.append((cast("str | int", item_key), value_key))
            has_tensor = has_tensor or value_has_tensor
            has_fragment = has_fragment or value_has_fragment
        try:
            entries.sort()
        except TypeError:
            return None
        return (value_type, tuple(entries)), has_tensor, has_fragment
    if value_type is types.FunctionType and not specialize_numeric:
        return _function_fast_dispatch_key(cast("types.FunctionType", value))
    if value_type is types.BuiltinFunctionType:
        # ``_specialization_extractors`` keys builtins by identity.
        return (value_type, value), False, False
    return None


def _function_fast_dispatch_key(
    fn: types.FunctionType,
) -> tuple[Hashable, bool, bool] | None:
    """Refine ``_function_key``: the code object, every closure cell's key and
    every default value's key.

    Callable arguments (score-modifier style hooks) are common in attention
    kernels; keying them by code object keeps repeat calls on the prepared
    fast path even when the caller rebuilds the function object per call.
    Closure cells are re-read on every call because ``nonlocal`` rebinding can
    change their contents without changing the function object.  Default
    values are baked into the compiled kernel when the lifted function is
    traced (``lift_closures`` passes ``__defaults__`` through), so they key by
    value like constexpr arguments; a default the constexpr rules cannot key
    (a tensor, a function) keeps the slow path.  Module globals the function
    reads are not part of the key: like the kernel's own globals they are read
    from the module when the kernel runs and specialized once, on the first
    call's values.
    """
    closure = fn.__closure__
    defaults = _function_defaults_fast_dispatch_key(fn)
    if defaults is None:
        return None
    defaults_key, has_fragment = defaults
    if not closure and not defaults_key:
        return (types.FunctionType, fn.__code__), False, False
    cells: list[Hashable] = []
    has_tensor = False
    for cell in closure or ():
        filled, contents = _read_cell(cell)
        if not filled:
            return None
        nested = _nested_fast_dispatch_key(contents)
        if nested is None:
            return None
        cell_key, cell_has_tensor, cell_has_fragment = nested
        cells.append(cell_key)
        has_tensor = has_tensor or cell_has_tensor
        has_fragment = has_fragment or cell_has_fragment
    if not defaults_key:
        return (types.FunctionType, fn.__code__, tuple(cells)), has_tensor, has_fragment
    return (
        (types.FunctionType, fn.__code__, tuple(cells), defaults_key),
        has_tensor,
        has_fragment,
    )


def _function_defaults_fast_dispatch_key(
    fn: types.FunctionType,
) -> tuple[Hashable, bool] | None:
    """Default keys and fragment presence, or None for an unsupported default."""
    defaults = fn.__defaults__ or ()
    kwdefaults = fn.__kwdefaults__ or {}
    if not defaults and not kwdefaults:
        return (), False
    positional: list[Hashable] = []
    has_fragment = False
    for value in defaults:
        nested = _nested_fast_dispatch_key(value, specialize_numeric=True)
        if nested is None:
            return None
        positional.append(nested[0])
        has_fragment = has_fragment or nested[2]
    keyword: list[tuple[str, Hashable]] = []
    for name, value in kwdefaults.items():
        nested = _nested_fast_dispatch_key(value, specialize_numeric=True)
        if nested is None:
            return None
        keyword.append((name, nested[0]))
        has_fragment = has_fragment or nested[2]
    keyword.sort()
    return (tuple(positional), tuple(keyword)), has_fragment


_TensorMetadataPathStep = (
    tuple[Literal["sequence"], type[object], int]
    | tuple[Literal["mapping"], type[object], type[object], Hashable]
    | tuple[Literal["dataclass"], type[object], str]
)


def _mapping_metadata_sort_key(key: object) -> tuple[str, str, str]:
    key_type = type(key)
    return key_type.__module__, key_type.__qualname__, repr(key)


def _walk_input_tensors(
    value: object,
    path: tuple[_TensorMetadataPathStep, ...] = (),
) -> Generator[tuple[tuple[_TensorMetadataPathStep, ...], torch.Tensor], None, None]:
    """Yield input tensors with deterministic structural roles."""
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, ConstExpr):
        return
    container_type = type(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk_input_tensors(
                getattr(value, field.name),
                (*path, ("dataclass", container_type, field.name)),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_input_tensors(
                item,
                (*path, ("sequence", container_type, index)),
            )
    elif isinstance(value, dict):
        for key, item in sorted(
            value.items(),
            key=lambda entry: _mapping_metadata_sort_key(entry[0]),
        ):
            yield from _walk_input_tensors(
                item,
                (
                    *path,
                    ("mapping", container_type, type(key), cast("Hashable", key)),
                ),
            )


def _input_tensor_metadata(values: Sequence[object]) -> tuple[Hashable, ...]:
    """Return exact tensor metadata keyed by deterministic structural roles."""
    return tuple(
        (path, _hashable_dims(tensor.size()), _hashable_dims(tensor.stride()))
        for path, tensor in _walk_input_tensors(values)
    )


def _input_tensor_aliases(values: Sequence[object]) -> tuple[int, ...] | None:
    """Return a canonical key only when tensor arguments alias."""
    # Eager dispatch normally receives a flat argument list. Only structured
    # inputs need the deterministic recursive walk and its metadata paths.
    tensors: list[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, ConstExpr):
            continue
        elif isinstance(value, (tuple, list, dict)) or (
            dataclasses.is_dataclass(value) and not isinstance(value, type)
        ):
            tensors = [tensor for _path, tensor in _walk_input_tensors(values)]
            break
    return _tensor_aliases(tensors)


def _input_fragment_aliases(values: Sequence[object]) -> tuple[int, ...] | None:
    """Keep observable fragment identity relationships beside their domain keys.

    Fresh declarations may share a binding only if their alias partition also
    agrees. In particular, BooleanFragment uses identity equality, so (f, f)
    and two distinct declarations can generate different code.
    """
    fragments: list[object] = []

    pending: list[tuple[object, frozenset[int]]] = [
        (value, frozenset()) for value in reversed(values)
    ]
    while pending:
        value, ancestors = pending.pop()
        kind = type(value)
        if kind in _BUILTIN_TUNABLE_FRAGMENTS:
            if (
                _plain_builtin_fragment_key(cast("ConfigSpecFragment", value))
                is not None
            ):
                fragments.append(value)
            continue
        if isinstance(value, ConstExpr):
            pending.append((value.value, ancestors))
            continue
        is_dataclass = dataclasses.is_dataclass(value) and not isinstance(value, type)
        is_namedtuple = isinstance(value, tuple) and hasattr(kind, "__match_args__")
        if (
            kind not in (tuple, list, dict, frozenset, types.FunctionType)
            and not is_dataclass
            and not is_namedtuple
        ):
            continue
        identity = id(value)
        if identity in ancestors:
            continue
        ancestors = ancestors | {identity}
        if is_dataclass:
            # pyrefly: ignore [bad-argument-type]
            children = [getattr(value, f.name) for f in dataclasses.fields(value)]
        elif is_namedtuple:
            # Match the named fields traversed by specialization_key.
            # pyrefly: ignore [missing-attribute]
            children = [getattr(value, name) for name in kind.__match_args__]
        elif kind is dict:
            mapping = cast("dict[object, object]", value)
            children = [mapping[key] for key in sorted(mapping, key=repr)]
        elif kind is frozenset:
            # Ordering equal-domain members arbitrarily can cause an extra
            # cache miss, but retaining every occurrence never merges distinct
            # alias graphs. No process-specific identity enters the key.
            children = sorted(
                cast("frozenset[object]", value),
                key=lambda item: repr(_constexpr_key(item)),
            )
        elif kind is types.FunctionType:
            function = cast("types.FunctionType", value)
            children = [
                *(
                    contents
                    for cell in function.__closure__ or ()
                    for filled, contents in (_read_cell(cell),)
                    if filled
                ),
                *(function.__defaults__ or ()),
                function.__kwdefaults__ or {},
            ]
        else:
            children = cast("Sequence[object]", value)
        pending.extend((child, ancestors) for child in reversed(children))
    if len(fragments) < 2:
        return None
    unique: list[object] = []
    partition: list[int] = []
    for fragment in fragments:
        for index, previous in enumerate(unique):
            if fragment is previous:
                partition.append(index)
                break
        else:
            partition.append(len(unique))
            unique.append(fragment)
    return tuple(partition)


def _tensor_aliases(tensors: Sequence[torch.Tensor]) -> tuple[int, ...] | None:
    """Canonical alias partition of ``tensors``; None when all are distinct."""
    if len(tensors) < 2:
        return None
    # Identity comparisons only: ``id()`` fires an audit event per call once
    # any audit hook is installed (importing inductor installs one).  The
    # common all-distinct case returns before building the partition.
    unique_tensors: list[torch.Tensor] = []
    for tensor in tensors:
        for previous_tensor in unique_tensors:
            if tensor is previous_tensor:
                break
        else:
            unique_tensors.append(tensor)
            continue
        break
    else:
        return None
    aliases: list[int] = []
    unique_tensors = []
    for tensor in tensors:
        for index, previous_tensor in enumerate(unique_tensors):
            if tensor is previous_tensor:
                aliases.append(index)
                break
        else:
            aliases.append(len(unique_tensors))
            unique_tensors.append(tensor)
    return tuple(aliases)


@dataclasses.dataclass(frozen=True)
class _CompilerSeedSpecializationExtractor:
    fact: CompilerHeuristicSpecializationFact
    reserved_sms: int

    def for_device(self, device: torch.device) -> Hashable:
        from . import get_num_sm
        from .settings import is_pallas_interpret

        if device.type == "cpu" and not is_pallas_interpret():
            # CPU tensors bound on a GPU backend (e.g. type-propagation debug
            # binds) have no SM count and can never launch; any stable key
            # keeps the specialization cache coherent.
            return self.fact, 0
        if self.fact == "device_num_sm":
            return self.fact, get_num_sm(device)
        assert self.fact == "config_num_sm"
        return self.fact, get_num_sm(device, reserved_sms=self.reserved_sms)

    def __call__(self, values: Sequence[object]) -> Hashable:
        if self.fact == "input_tensor_metadata":
            return self.fact, _input_tensor_metadata(values)

        device = _canonicalize_argument_device(_find_device(tuple(values)))
        return self.for_device(device)


def _compiler_seed_specialization_extractors(
    facts: frozenset[CompilerHeuristicSpecializationFact],
    reserved_sms: int,
    *,
    static_shapes: bool,
) -> tuple[_CompilerSeedSpecializationExtractor, ...]:
    if static_shapes:
        facts = frozenset(fact for fact in facts if fact != "input_tensor_metadata")
    return tuple(
        _CompilerSeedSpecializationExtractor(fact, reserved_sms)
        for fact in sorted(facts)
    )


# (slot index, extractor) pairs of an extras tuple re-evaluated on every call.
_PerCallSlots = tuple[tuple[int, Callable[[Sequence[object]], Hashable]], ...]


class _SpecializationAlias:
    """An omitted-default signature backed by a normalized signature."""

    __slots__ = (
        "schemas",
        "canonical_signature",
        "trailing_defaults",
        "per_call_cache",
    )

    def __init__(
        self,
        schemas: dict[Hashable, list[Callable[[Sequence[object]], Hashable]]],
        canonical_signature: tuple[Hashable, ...],
        trailing_defaults: tuple[object, ...],
    ) -> None:
        self.schemas = schemas
        self.canonical_signature = canonical_signature
        self.trailing_defaults = trailing_defaults
        # The canonical list ``_specialization_alias_per_call`` last split and
        # its result: published lists are replaced, never mutated.
        self.per_call_cache: (
            tuple[
                list[Callable[[Sequence[object]], Hashable]] | None,
                _PerCallSlots | None,
            ]
            | None
        ) = None

    def __call__(self, values: Sequence[object]) -> Hashable:
        normalized = (*values, *self.trailing_defaults)
        return tuple(
            extractor(normalized)
            for extractor in self.schemas[self.canonical_signature]
        )


@dataclasses.dataclass(frozen=True)
class PreparedMetadataSpecializationExtractor:
    """Specialization read from input tensor metadata.

    Exact metadata, which ``_make_prepared_arg_guard`` checks, implies its
    result unless it reads a storage offset (``_implied_by_exact_metadata``).
    """

    extractor: Callable[[Sequence[object]], Hashable]

    def __call__(self, args: Sequence[object]) -> Hashable:
        return self.extractor(args)


@dataclasses.dataclass(frozen=True, eq=False)
class _TensorDescriptorAlignmentExtractor:
    """Whether a host-descriptor tensor's base pointer meets its alignment
    guard.  Prepared metadata guards do not cover base pointers, so this runs
    on every call."""

    extract_tensor: Callable[[Sequence[object]], Hashable]
    requires_zero_storage_offset: bool

    def __call__(self, args: Sequence[object]) -> Hashable:
        tensor = self.extract_tensor(args)
        tensor_type = type(tensor)
        if tensor_type is torch.Tensor or tensor_type is torch.nn.Parameter:
            # ``_concrete_tensor_satisfies_alignment_guard`` for a concrete
            # tensor, inline: this runs on every dispatch.
            tensor = cast("torch.Tensor", tensor)
            return tensor.data_ptr() % 16 == 0 and (
                not self.requires_zero_storage_offset or tensor.storage_offset() == 0
            )
        return _concrete_tensor_satisfies_alignment_guard(
            tensor, self.requires_zero_storage_offset
        )


def _implied_by_exact_metadata(
    extractor: Callable[[Sequence[object]], Hashable],
) -> bool:
    """Whether exact argument metadata implies ``extractor``'s result.

    Both the fast-key prefix and the prepared argument guard
    (``_make_prepared_arg_guard``) pin each tensor's exact
    dtype/shape/stride/device but not its storage offset (symbolic in a
    dynamic-shape compile), so a guard fact over a storage-offset symbol is
    evaluated on every call like pointer alignment and scalar values.
    """
    if not isinstance(extractor, PreparedMetadataSpecializationExtractor):
        return False
    inner = extractor.extractor
    return not (
        isinstance(inner, _GuardedInputFact)
        and any(
            source.prop == TensorProperty.STORAGE_OFFSET
            for _symbol, source in inner.sources
        )
    )


# Omitted defaults an alias can look through: immutable, so the canonical
# extractors exact metadata implies read nothing a later call can change.
_IMMUTABLE_DEFAULT_TYPES: tuple[type[object], ...] = (
    int,
    float,
    bool,
    str,
    type(None),
    torch.dtype,
    torch.device,
)


class _SpecializationAliasDispatch:
    """The per-call part of an omitted-default ``_SpecializationAlias`` slot.

    Produces the alias's result tuple from memoized canonical results,
    re-evaluating only the canonical extractors that exact metadata does not
    imply, on the arguments with the trailing defaults appended.
    """

    __slots__ = ("per_call", "results", "trailing_defaults")

    def __init__(
        self,
        trailing_defaults: tuple[object, ...],
        results: tuple[Hashable, ...],
        per_call: tuple[tuple[int, Callable[[Sequence[object]], Hashable]], ...],
    ) -> None:
        self.trailing_defaults = trailing_defaults
        self.results = results
        self.per_call = per_call

    def __call__(self, args: Sequence[object]) -> Hashable:
        normalized = (*args, *self.trailing_defaults)
        results = list(self.results)
        for index, extractor in self.per_call:
            results[index] = extractor(normalized)
        return tuple(results)


def _evaluate_fast_dispatch_extras(
    extra_fns: Sequence[Callable[[Sequence[object]], Hashable]],
    args: Sequence[object],
) -> tuple[
    tuple[Hashable, ...],
    tuple[tuple[int, Callable[[Sequence[object]], Hashable]], ...],
]:
    """Evaluate a published schema on ``args``, in order, and split off the
    slots the exact fast-key prefix does not imply (with what re-evaluates
    each)."""
    results: list[Hashable] = []
    per_call: list[tuple[int, Callable[[Sequence[object]], Hashable]]] = []
    for index, extractor in enumerate(extra_fns):
        dispatch: Callable[[Sequence[object]], Hashable] | None
        if isinstance(extractor, _SpecializationAlias):
            result, dispatch = _evaluate_specialization_alias(extractor, args)
        else:
            result = extractor(args)
            dispatch = None if _implied_by_exact_metadata(extractor) else extractor
            if isinstance(dispatch, _RuntimeInputSpecializationExtractor):
                dispatch = _fast_canonical_metadata_dispatch(dispatch, result, args)
        results.append(result)
        if dispatch is not None:
            per_call.append((index, dispatch))
    return tuple(results), tuple(per_call)


def _specialization_alias_per_call(
    alias: _SpecializationAlias,
) -> tuple[list[Callable[[Sequence[object]], Hashable]], _PerCallSlots] | None:
    """The canonical schema behind ``alias`` and its slots exact metadata does
    not imply; None when the whole alias stays per call (a mutable trailing
    default, or no canonical schema, which makes the alias raise).

    Cached per canonical list: every re-prepared call looks it up.
    """
    canonical = alias.schemas.get(alias.canonical_signature)
    cached = alias.per_call_cache
    if cached is None or cached[0] is not canonical:
        per_call = None
        if canonical is not None and all(
            type(default) in _IMMUTABLE_DEFAULT_TYPES
            for default in alias.trailing_defaults
        ):
            per_call = tuple(
                (index, extractor)
                for index, extractor in enumerate(canonical)
                if not _implied_by_exact_metadata(extractor)
            )
        cached = alias.per_call_cache = (canonical, per_call)
    canonical, per_call = cached
    if canonical is None or per_call is None:
        return None
    return canonical, per_call


def _evaluate_specialization_alias(
    alias: _SpecializationAlias,
    args: Sequence[object],
) -> tuple[Hashable, Callable[[Sequence[object]], Hashable] | None]:
    """``alias(args)`` and what re-evaluates it per call (None when exact
    metadata implies all of it)."""
    if (split := _specialization_alias_per_call(alias)) is None:
        return alias(args), alias
    canonical, per_call = split
    normalized = (*args, *alias.trailing_defaults)
    # The same tuple ``alias(args)`` builds.
    results = tuple(extractor(normalized) for extractor in canonical)
    if not per_call:
        return results, None
    per_call = tuple(
        (
            index,
            _fast_canonical_metadata_dispatch(extractor, results[index], normalized)
            if isinstance(extractor, _RuntimeInputSpecializationExtractor)
            else extractor,
        )
        for index, extractor in per_call
    )
    return results, _SpecializationAliasDispatch(
        alias.trailing_defaults, results, per_call
    )


def _capture_safe_specialization(
    extractor: Callable[[Sequence[object]], Hashable],
) -> bool:
    if isinstance(extractor, PreparedMetadataSpecializationExtractor):
        return True
    if isinstance(extractor, _RuntimeInputSpecializationExtractor):
        # This classifier reads only projected tensor layout/size/stride
        # metadata. Keep its live call; capture safety is not a promise that
        # eager prepared argument guards imply its result.
        return type(extractor.classifier) is _CanonicalLayoutClassifier
    if isinstance(extractor, _SpecializationAlias):
        return all(
            _capture_safe_specialization(item)
            for item in extractor.schemas[extractor.canonical_signature]
        )
    return False


@dataclasses.dataclass(frozen=True)
class _RuntimeInputSpecializationExtractor:
    """Runtime classifier together with its source projections."""

    source_extractors: tuple[Callable[[Sequence[object]], Hashable], ...]
    classifier: Callable[[Sequence[object]], Hashable]
    reusable_tensor_properties: frozenset[str]
    specialization_key: str | None = None
    prepared_guard_factory: PreparedGuardFactory | None = None
    specialization: RuntimeInputSpecialization | None = dataclasses.field(
        default=None, compare=False, repr=False
    )

    def source_values(self, args: Sequence[object]) -> tuple[object, ...]:
        return tuple(extract(args) for extract in self.source_extractors)

    def __call__(self, args: Sequence[object]) -> Hashable:
        return self.classifier(self.source_values(args))


def _fast_canonical_metadata_dispatch(
    extractor: _RuntimeInputSpecializationExtractor,
    expected: Hashable,
    args: Sequence[object],
) -> Callable[[Sequence[object]], Hashable]:
    # A fast-schema hit already pins the same container/tensor metadata as
    # prepared dispatch. Only the typed canonical classifier supplies this
    # narrow residual-layout predicate; generic classifiers remain live.
    if type(extractor.classifier) is not _CanonicalLayoutClassifier:
        return extractor
    guard = _build_prepared_fact_guard(extractor, expected, tuple(args))
    if guard is None:
        return extractor

    def read(values: Sequence[object]) -> Hashable:
        return expected if guard(values) else extractor(values)

    return read


def _partition_prepared_extra_guards(
    args: tuple[object, ...],
    extra_guards: tuple[tuple[Callable[[Sequence[object]], Hashable], Hashable], ...],
) -> tuple[
    tuple[tuple[Callable[[Sequence[object]], Hashable], Hashable], ...],
    tuple[Callable[[Sequence[object]], bool], ...],
]:
    """Split guards into always-check classifiers and prepared fact guards.

    Tensor type/shape/stride/dtype/device facts are already checked by the
    prepared argument guard, so extractors exact metadata implies are
    dropped, and an omitted-default alias keeps only its per-call part
    (``_prepared_alias_guard``).  A runtime input specialization may supply a
    ``prepared_guard_factory`` that turns its expected result into a direct
    check of the pointer/storage facts those pins leave open, so repeat calls
    with fresh output tensors never re-run the full classifier.  Classifiers
    without one (tensor contents, arbitrary Python state) remain in the
    always-check set, as does any factory guard that rejects the very
    arguments it was prepared from.
    """
    always_check: list[tuple[Callable[[Sequence[object]], Hashable], Hashable]] = []
    fact_guards: list[Callable[[Sequence[object]], bool]] = []
    for extractor, expected in extra_guards:
        if isinstance(extractor, _SpecializationAlias):
            alias_guard = _prepared_alias_guard(extractor, expected, args)
            if alias_guard is not None:
                always_check.append((alias_guard, expected))
            continue
        if _implied_by_exact_metadata(extractor):
            continue
        guard = (
            _build_prepared_fact_guard(extractor, expected, args)
            if isinstance(extractor, _RuntimeInputSpecializationExtractor)
            else None
        )
        if guard is None:
            always_check.append((extractor, expected))
        else:
            fact_guards.append(guard)
    return tuple(always_check), tuple(fact_guards)


def _prepared_alias_guard(
    alias: _SpecializationAlias,
    expected: Hashable,
    args: tuple[object, ...],
) -> Callable[[Sequence[object]], Hashable] | None:
    """What a prepared call re-evaluates for an omitted-default alias slot.

    Looks through to the canonical schema like the fast-key memo
    (``_specialization_alias_per_call``): None when exact metadata, which the
    prepared argument guard pins, implies every canonical extractor, else a
    ``_SpecializationAliasDispatch`` that re-evaluates only the rest and
    takes the implied slots from ``expected``, the fast key's value for these
    arguments (as the guards of non-alias extractors exact metadata implies
    are dropped unevaluated; evaluating them would cost a few microseconds
    each time a call re-prepares).  The whole alias stays when its defaults
    are mutable or its canonical schema is missing, and, failing closed, when
    ``expected`` does not have the canonical shape or the re-evaluated slots
    do not reproduce it.
    """
    split = _specialization_alias_per_call(alias)
    if split is None or type(expected) is not tuple or len(expected) != len(split[0]):
        return alias
    per_call = split[1]
    if not per_call:
        return None
    dispatch = _SpecializationAliasDispatch(alias.trailing_defaults, expected, per_call)
    return dispatch if dispatch(args) == expected else alias


def _build_prepared_fact_guard(
    extractor: _RuntimeInputSpecializationExtractor,
    expected: Hashable,
    args: tuple[object, ...],
) -> Callable[[Sequence[object]], bool] | None:
    factory = extractor.prepared_guard_factory
    if factory is None:
        return None
    try:
        guard = factory(extractor.source_extractors, expected)
        return guard if guard(args) else None
    except Exception:
        return None


def _make_prepared_arg_guard(
    kernel: Kernel,
    args: tuple[object, ...],
) -> Callable[[tuple[object, ...]], bool]:
    namespace: dict[str, object] = {}
    checks = [f"len(args) == {len(args)}"]
    counter = itertools.count()
    tensor_paths: list[tuple[str, weakref.ReferenceType[torch.Tensor]]] = []
    # Plain tensors whose dtype / device / shape / stride one C++ guard
    # (dynamo's ``TensorGuards``) checks in front of the Python checks: ~0.1
    # us per tensor against ~0.5 us for the four attribute comparisons.
    cxx_tensor_paths: list[tuple[str, torch.Tensor]] = []

    def append_checks(
        value: object,
        prefix: str,
        annotation: object | None = None,
    ) -> None:
        token = next(counter)
        value_type = type(value)
        namespace[f"type_{token}"] = value_type
        checks.append(f"type({prefix}) is type_{token}")
        fragment = (
            value.value
            if isinstance(value, ConstExpr)
            else value
            if annotation is ConstExpr
            else None
        )
        if type(fragment) in _BUILTIN_TUNABLE_FRAGMENTS:
            fragment_key = _constexpr_key(fragment)
            if not isinstance(fragment_key, _FragmentValueKey):
                raise TypeError("fragment without an immutable constexpr key")
            namespace[f"value_{token}"] = fragment_key
            namespace["_constexpr_key"] = _constexpr_key
            source = f"{prefix}.value" if isinstance(value, ConstExpr) else prefix
            checks.append(f"_constexpr_key({source}) == value_{token}")
        elif value_type in (torch.Tensor, torch.nn.Parameter):
            assert isinstance(value, torch.Tensor)
            for previous_prefix, previous_tensor_ref in tensor_paths:
                relation = "is" if previous_tensor_ref() is value else "is not"
                checks.append(f"{prefix} {relation} {previous_prefix}")
            tensor_paths.append((prefix, weakref.ref(value)))
            static_indices = getattr(value, "_dynamo_static_indices", None)
            if value_type is torch.Tensor and static_indices is None:
                # (The C++ guard also pins requires_grad and the dispatch
                # keys: a change there falls back to the full dispatch.)
                cxx_tensor_paths.append((prefix, value))
            else:
                namespace[f"dtype_{token}"] = value.dtype
                namespace[f"shape_{token}"] = value.shape
                namespace[f"stride_{token}"] = value.stride()
                namespace[f"device_{token}"] = value.device
                checks.extend(
                    (
                        f"{prefix}.dtype is dtype_{token}",
                        f"{prefix}.shape == shape_{token}",
                        f"{prefix}.stride() == stride_{token}",
                        f"{prefix}.device == device_{token}",
                    )
                )
            if static_indices is None:
                checks.append(
                    f"getattr({prefix}, '_dynamo_static_indices', None) is None"
                )
            else:
                namespace[f"static_indices_{token}"] = frozenset(static_indices)
                checks.extend(
                    (
                        f"getattr({prefix}, '_dynamo_static_indices', None) is not None",
                        f"frozenset({prefix}._dynamo_static_indices) == static_indices_{token}",
                    )
                )
        elif value_type is tuple or value_type is list:
            sequence = cast("Sequence[object]", value)
            checks.append(f"len({prefix}) == {len(sequence)}")
            for item_index, item in enumerate(sequence):
                append_checks(
                    item,
                    f"{prefix}[{item_index}]",
                    annotation,
                )
        elif value_type is dict:
            mapping = cast("dict[str | int, object]", value)
            checks.append(f"len({prefix}) == {len(mapping)}")
            for item_key, item_value in mapping.items():
                key_name = f"key_{token}_{len(namespace)}"
                namespace[key_name] = item_key
                checks.append(f"{key_name} in {prefix}")
                append_checks(
                    item_value,
                    f"{prefix}[{key_name}]",
                    annotation,
                )
        elif value_type in (bool, int, float):
            # Ordinary numeric arguments are runtime values in Helion's base
            # specialization.  A constexpr annotation is the one exception;
            # hl.specialize() constraints are covered by ``_extra_guards``.
            if annotation is ConstExpr:
                if value_type is float and value != value:
                    # The hex-based prepared guard cannot distinguish NaN
                    # payloads. Keep those calls on the regular bit-exact
                    # specialization path.
                    raise TypeError("constexpr NaN cannot use a prepared call")
                namespace[f"value_{token}"] = (
                    cast("float", value).hex() if value_type is float else value
                )
                checks.append(
                    f"{prefix}.hex() == value_{token}"
                    if value_type is float
                    else f"{prefix} == value_{token}"
                )
        elif value_type in (str, type(None), torch.dtype, torch.device):
            namespace[f"value_{token}"] = value
            checks.append(f"{prefix} == value_{token}")
        elif isinstance(value, ConstExpr):
            inner = value.value
            namespace[f"inner_type_{token}"] = type(inner)
            checks.append(f"type({prefix}.value) is inner_type_{token}")
            if type(inner) is float:
                if inner != inner:
                    # Match the constexpr-annotation path above: the hex-based
                    # guard cannot distinguish NaN payloads.
                    raise TypeError("constexpr NaN cannot use a prepared call")
                namespace[f"value_{token}"] = inner.hex()
                checks.append(f"{prefix}.value.hex() == value_{token}")
            else:
                key = _constexpr_key(inner)
                namespace[f"value_{token}"] = key
                if key is inner and type(inner) not in (tuple, frozenset):
                    checks.append(f"{prefix}.value == value_{token}")
                else:
                    # Even an all-integer container needs the canonical key:
                    # Python equality also accepts incoming float/bool members.
                    namespace["_constexpr_key"] = _constexpr_key
                    checks.append(f"_constexpr_key({prefix}.value) == value_{token}")
        elif value_type is types.FunctionType and annotation is not ConstExpr:
            # Mirror ``_function_key``: the code object identifies the
            # function, every closure cell is guarded like an argument of its
            # own type (cells are re-read because ``nonlocal`` can rebind them
            # without replacing the function object), and every default value
            # like a constexpr argument (defaults are baked into the traced
            # kernel).  Module globals the function reads are not guarded, as
            # the kernel's own globals are not.
            function = cast("types.FunctionType", value)
            namespace[f"code_{token}"] = function.__code__
            checks.append(f"{prefix}.__code__ is code_{token}")
            for cell_index, cell in enumerate(function.__closure__ or ()):
                filled, contents = _read_cell(cell)
                if not filled:
                    raise TypeError("empty closure cell cannot use a prepared call")
                cell_prefix = f"{prefix}.__closure__[{cell_index}]"
                namespace["_read_cell"] = _read_cell
                checks.append(f"_read_cell({cell_prefix})[0]")
                append_checks(contents, f"{cell_prefix}.cell_contents")
            if _function_defaults_fast_dispatch_key(function) is None:
                raise TypeError("callable default without a constexpr key")
            defaults = function.__defaults__ or ()
            checks.append(f"len({prefix}.__defaults__ or ()) == {len(defaults)}")
            for default_index, default in enumerate(defaults):
                append_checks(
                    default, f"{prefix}.__defaults__[{default_index}]", ConstExpr
                )
            kwdefaults = function.__kwdefaults__ or {}
            checks.append(f"len({prefix}.__kwdefaults__ or {{}}) == {len(kwdefaults)}")
            for name, default in kwdefaults.items():
                checks.append(f"{name!r} in {prefix}.__kwdefaults__")
                append_checks(default, f"{prefix}.__kwdefaults__[{name!r}]", ConstExpr)
        elif value_type is types.BuiltinFunctionType:
            namespace[f"value_{token}"] = value
            checks.append(f"{prefix} is value_{token}")
        else:
            raise TypeError(f"unsupported prepared-call argument: {value_type!r}")

    try:
        for index, arg in enumerate(args):
            append_checks(arg, f"args[{index}]", kernel._annotations[index])

    finally:
        # Break the recursive closure cycle so its temporary C++ guard inputs
        # are released even when cyclic GC is disabled or traversal raises.
        append_checks = cast("Any", None)
    if (fragment_aliases := _input_fragment_aliases(args)) is not None:
        namespace["_input_fragment_aliases"] = _input_fragment_aliases
        namespace["fragment_aliases"] = fragment_aliases
        checks.append("_input_fragment_aliases(args) == fragment_aliases")
    if cxx_tensor_paths:
        cxx_check = _prepared_cxx_tensor_check(
            [tensor for _prefix, tensor in cxx_tensor_paths]
        )
        if cxx_check is not None:
            namespace["cxx_tensor_check"] = cxx_check
            paths = ", ".join(prefix for prefix, _tensor in cxx_tensor_paths)
            # After every structural check: a path below a container or a
            # closure cell exists only once the checks before it have held
            # (a shorter list, a builtin in a function's place), and the
            # chain short-circuits before reading it.
            checks.append(f"cxx_tensor_check({paths})")
        else:
            for prefix, value in cxx_tensor_paths:
                token = next(counter)
                namespace[f"dtype_{token}"] = value.dtype
                namespace[f"shape_{token}"] = value.shape
                namespace[f"stride_{token}"] = value.stride()
                namespace[f"device_{token}"] = value.device
                checks.extend(
                    (
                        f"{prefix}.dtype is dtype_{token}",
                        f"{prefix}.shape == shape_{token}",
                        f"{prefix}.stride() == stride_{token}",
                        f"{prefix}.device == device_{token}",
                    )
                )
    return eval(f"lambda args: {' and '.join(checks)}", namespace)


def _prepared_cxx_tensor_check(
    tensors: Sequence[torch.Tensor],
) -> Callable[..., bool] | None:
    """One C++ check of every tensor's type, dtype, device, requires_grad,
    dispatch keys, sizes and strides against these tensors' (dynamo's
    ``TensorGuards``), or None when the guard is unavailable or does not
    accept them (the Python checks then stand alone)."""
    try:
        from torch._C._dynamo.guards import TensorGuards
    except ImportError:  # pragma: no cover - older torch
        return None
    try:
        guards = TensorGuards(
            *tensors,
            dynamic_dims_sizes=cast("Any", [list(t.shape) for t in tensors]),
            dynamic_dims_strides=cast("Any", [list(t.stride()) for t in tensors]),
        )
        if not guards.check(*tensors):
            return None
    except Exception:
        return None
    return guards.check


class _PreparedCall:
    """Monomorphic eager-call guard for a compiled ``BoundKernel``.

    ``Kernel._dispatch_cache`` remains the general multi-specialization cache.
    This object only makes its most recently used entry cheap to revisit: it
    compares argument metadata without rebuilding and hashing a nested cache
    key, then calls the already-compiled host wrapper directly.
    """

    __slots__ = (
        "_dist_initialized",
        "_extra_guards",
        "_fact_guards",
        "_is_distributed",
        "_matches_args",
        "_reset_generation",
        "_specialization_generation",
        "bound",
    )

    def __init__(
        self,
        bound: BoundKernel,
        args: tuple[object, ...],
        *,
        dist_initialized: bool,
        extra_guards: tuple[
            tuple[Callable[[Sequence[object]], Hashable], Hashable], ...
        ],
        is_distributed: bool,
    ) -> None:
        self._matches_args = _make_prepared_arg_guard(bound.kernel, args)
        self._dist_initialized = dist_initialized
        self._is_distributed = is_distributed
        self._extra_guards, self._fact_guards = _partition_prepared_extra_guards(
            args, extra_guards
        )
        self._specialization_generation = bound.kernel._specialization_generation
        self._reset_generation = bound.kernel._reset_generation
        self.bound = bound

    @classmethod
    def build(
        cls,
        kernel: Kernel,
        bound: BoundKernel,
        args: tuple[object, ...],
        *,
        dist_initialized: bool,
        extra_guards: tuple[
            tuple[Callable[[Sequence[object]], Hashable], Hashable], ...
        ],
        is_distributed: bool,
    ) -> _PreparedCall | None:
        if (
            not bound.env.backend.supports_eager_prepared_call
            or kernel._key_fn is not None
        ):
            return None
        # The caller already obtained a non-None ``_fast_dispatch_key``, which
        # proves every argument has an exact supported type.
        try:
            return cls(
                bound,
                args,
                dist_initialized=dist_initialized,
                extra_guards=extra_guards,
                is_distributed=is_distributed,
            )
        except Exception:
            # Preparation is optional and must not introduce a new user-visible
            # failure after a kernel has run successfully.
            return None

    def matches(self, kernel: Kernel, args: tuple[object, ...]) -> bool:
        try:
            if (
                self._reset_generation != kernel._reset_generation
                or self._specialization_generation != kernel._specialization_generation
                or not self._matches_args(args)
            ):
                return False
            dist_initialized = dist.is_initialized()
            # ``kernel_uses_symm_mem`` and declared distributed intent are both
            # false before process-group initialization. If that state changes,
            # reject the prepared call before rechecking it.
            if dist_initialized != self._dist_initialized or (
                dist_initialized
                and kernel._compute_is_distributed(
                    args, dist_initialized=dist_initialized
                )
                != self._is_distributed
            ):
                return False
            for extractor, expected in self._extra_guards:
                if extractor(args) != expected:
                    return False
            for guard in self._fact_guards:
                if not guard(args):
                    return False
            return (
                self._reset_generation == kernel._reset_generation
                and self._specialization_generation == kernel._specialization_generation
            )
        except Exception:
            # Guard evaluation is an optional fast path. Falling through lets
            # the normal dispatch machinery preserve its own error semantics.
            return False


def _canonicalize_multi_shape_device(device: torch.device) -> torch.device:
    if device.type in ("cpu", "meta", "mps"):
        raise exc.InvalidAPIUsage(
            f"autotune_multi requires an indexed accelerator device, got {device}"
        )
    if device.index is not None:
        return device
    try:
        index = _current_device_index(device.type)
    except exc.InvalidAPIUsage as error:
        raise exc.InvalidAPIUsage(
            "autotune_multi requires a current indexed accelerator, "
            f"got {device.type!r}"
        ) from error
    return torch.device(device.type, index)


def _has_unspecialized_numeric_value(value: object) -> bool:
    """Return whether normal specialization records only a numeric value's type."""
    if isinstance(value, ConstExpr):
        return False
    if type(value) in (bool, int, float):
        return True
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _has_unspecialized_numeric_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, dict):
        return any(_has_unspecialized_numeric_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_unspecialized_numeric_value(item) for item in value)
    return False


def _resolve_index_dtype(
    settings: Settings,
    args: Sequence[object] | tuple[object, ...],
) -> torch.dtype:
    if (index_dtype := settings.index_dtype) is not None:
        limit = torch.iinfo(index_dtype).max
    else:
        limit = _INT32_INDEX_LIMIT
    over_limit = False

    def _check(tensor: torch.Tensor) -> None:
        nonlocal over_limit
        if over_limit:
            return
        try:
            over_limit = bool(tensor.numel() > limit)
        except RuntimeError:  # unbacked SymInt
            if index_dtype is None:
                over_limit = True

    tree_map_only(torch.Tensor, _check, args)
    # pyrefly: ignore [unbound-name]
    if index_dtype is None:  # Auto-select when not provided
        return torch.int64 if over_limit else torch.int32
    if over_limit:
        # pyrefly: ignore [unbound-name]
        raise exc.InputTensorNumelExceedsIndexType(index_dtype=index_dtype)
    # pyrefly: ignore [unbound-name]
    return index_dtype


def _argument_device_is_fixed(args: Sequence[object]) -> bool:
    """Whether the device ``_device_specialization_key`` keys is the same for
    every insertion order of ``args``' dicts and every current device.

    True when every tensor and ``torch.device`` that ``_find_device`` can
    reach is one device, and that device is indexed (or has no index to
    resolve).  The fast-key prefix sorts dict entries, so with several devices
    a dict's insertion order can change which one is found first.  ``args``
    are arguments ``_fast_dispatch_key`` admits, every value of an exact
    type, so exact type checks walk what ``_find_device``'s ``isinstance``
    checks do (a ``ConstExpr`` is not a plain tuple).
    """
    device: torch.device | None = None
    pending = list(args)
    while pending:
        value = pending.pop()
        value_type = type(value)
        if value_type is torch.Tensor or value_type is torch.nn.Parameter:
            value_device = cast("torch.Tensor", value).device
        elif value_type is torch.device:
            value_device = cast("torch.device", value)
        else:
            if value_type is tuple or value_type is list:
                pending.extend(cast("Sequence[object]", value))
            elif value_type is dict:
                pending.extend(cast("dict[object, object]", value).values())
            continue
        if device is None:
            device = value_device
        elif value_device != device:
            return False
    return (
        device is None
        or device.index is not None
        or device.type in ("cpu", "meta", "mps")
    )


def _device_specialization_key(
    args: Sequence[object],
    *,
    backend: str,
    compiler_heuristics_enabled: bool,
) -> tuple[
    str | None,
    tuple[int, int] | None,
    tuple[tuple[str, str | None], ...],
]:
    """Return the recursive argument device key used by bound-kernel caching.

    `_find_device` intentionally searches tensors and bare `torch.device`
    objects inside supported containers to match binding device selection.
    Capability splits mixed-sm cache entries. Exact hardware identity is added
    only when an active compiler heuristic uses it to gate default promotion.
    Device index remains excluded so otherwise-compatible devices share bound
    kernels, matching prior behavior.
    """
    try:
        device = _canonicalize_argument_device(_find_device(tuple(args)))
    except exc.NoTensorArgs:
        return None, None, ()
    promotion_key = (
        compiler_promotion_specialization_key(backend, device)
        if compiler_heuristics_enabled
        else ()
    )
    return device.type, target_device_capability(device), promotion_key


@dataclasses.dataclass
class OutputCodeOptions:
    """Options for :meth:`BoundKernel.to_code`.

    Passing ``options=None`` (the default) keeps ``to_code``'s original behavior.

    Attributes:
        allow_helion_deps: When ``False``, emit a self-contained module that does
            not import ``helion`` at runtime -- the dependency-free launcher is
            inlined (and any in-kernel runtime helpers are embedded) so the only
            deps are ``torch`` + the backend DSL.
        jax_fn: Pallas only. When ``True``, emit a module whose entrypoint operates
            on ``jax.Array`` inputs instead of TorchTPU tensors. Orthogonal to
            ``allow_helion_deps``: combine with ``allow_helion_deps=False`` for a
            pure-JAX module (launch core inlined), or leave ``allow_helion_deps=True``
            to import the launch core from helion.
    """

    allow_helion_deps: bool = True
    jax_fn: bool = False


class Kernel(Generic[_R]):
    def __init__(
        self,
        fn: Callable[..., _R],
        *,
        configs: Sequence[ConfigLike] | None = None,
        settings: Settings | None,
        key: Callable[..., Hashable] | None = None,
        cute_structural_policy: CuteStructuralPolicyRequest = DEFAULT_STRUCTURAL_POLICY,
    ) -> None:
        """
        Initialize the Kernel object.  This is typically called from the `@helion.kernel` decorator.

        Args:
            fn: The function to be compiled as a Helion kernel.
            configs: A list of configurations to use for the kernel.
            settings: The settings to be used by the Kernel. If None, a new `Settings()` instance is created.
            key: Optional callable that returns an extra hashable component for specialization.
        """
        super().__init__()
        assert isinstance(fn, types.FunctionType)
        assert_no_conflicts(fn)
        self.name: str = fn.__name__
        # pyrefly: ignore [read-only]
        self.fn: types.FunctionType = fn
        self._kernel_source: str | None = None
        self.signature: inspect.Signature = inspect.signature(fn)
        self.settings, selected_configs, self._cute_structural_policy = (
            select_structural_policy(
                settings or Settings(), configs or [], cute_structural_policy
            )
        )
        self._key_fn: Callable[..., Hashable] | None = key
        # Whether the kernel declares distributed intent via an hl.ProcessGroupName
        # argument. Computed once so the per-call is_distributed check stays cheap
        # on the dispatch hot path (avoids re-running inspect.signature). See #3024.
        self._declares_process_group: bool = kernel_declares_process_group(fn)
        self.configs: list[Config] = [
            # pyrefly: ignore [bad-argument-type]
            Config(**config) if isinstance(config, dict) else config
            for config in selected_configs
        ]
        self._bind_lock = threading.RLock()
        self._specialize_extra_lock = threading.Lock()
        self._bound_kernels: dict[BoundKernelInMemoryCacheKey, BoundKernel] = {}
        self._opaque_bindings: OrderedDict[BoundKernel, None] = OrderedDict()
        # Fresh traces cannot share bounds, but compatible frontend programs can
        # keep an explicitly selected configuration. No inputs or launchers live
        # here, and the fixed bound avoids growth from arbitrary Python branches.
        self._fresh_selections: OrderedDict[Hashable, Config] = OrderedDict()
        self._warned_fresh_selection = False
        self._warned_uncovered_specializations: set[str] = set()
        # Fast dispatch cache: maps a cheap, fine-grained argument key (exact
        # dtype/shape/stride/device per tensor) directly to a BoundKernel,
        # skipping the full specialization-key machinery on repeat calls.
        self._dispatch_cache: dict[Hashable, BoundKernel] = {}
        # Fast-key prefix -> the schema it resolves to (``_FastDispatchSchema``),
        # so repeat ``_fast_dispatch_key`` calls skip the base specialization
        # key and the extractors exact metadata implies.  Entries are valid for
        # one ``_specialization_generation``; every generation bump drops them.
        self._fast_dispatch_schemas: dict[
            tuple[Hashable, ...], _FastDispatchSchema
        ] = {}
        self._prepared_call: _PreparedCall | None = None
        # Signature -> its published extractor list.  Writers either fill a
        # missing (None) slot or bump ``_specialization_generation``, and a
        # published list is replaced, never mutated in place: memoized fast
        # dispatch schemas (and any cache keyed by a list's id(), which must
        # also hold the list) rely on both.
        self._specialize_extra: dict[
            Hashable, list[Callable[[Sequence[object]], Hashable]]
        ] = {}
        self._compiler_seed_specialize_extra: dict[
            Hashable, tuple[_CompilerSeedSpecializationExtractor, ...]
        ] = {}
        self._specialization_aliases: dict[
            tuple[Hashable, ...], _SpecializationAlias
        ] = {}
        self._specialization_generation = 0
        self._reset_generation = 0
        self._has_specialization_extras = False
        # Descriptors of the input-value specializations a backend registered
        # from generated code (``Backend.after_codegen`` / ``after_compile``),
        # per signature, so each is keyed once.
        self._cute_grouped_static_tail_extra_descriptors: dict[
            Hashable, set[Hashable]
        ] = {}
        self._cute_reduction_extent_descriptors: dict[Hashable, set[Hashable]] = {}
        if any(
            param.kind
            in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for param in self.signature.parameters.values()
        ):
            raise TypeError(
                f"Kernel({self.name}) cannot have *args, **kwargs, or keyword-only arguments"
            )

        self._annotations: list[object] = []
        for param in self.signature.parameters.values():
            ann = param.annotation
            if isinstance(ann, str) and re.search(r"constexpr", ann, re.IGNORECASE):
                self._annotations.append(ConstExpr)
            else:
                self._annotations.append(ann)

        # Cache the number of parameters to avoid accessing self.signature.parameters
        # during torch.compile tracing.
        self._num_params: int = len(self.signature.parameters)
        self._normalize_args_plans: dict[
            tuple[int, tuple[str, ...]],
            tuple[tuple[object, ...], tuple[int, ...]] | None,
        ] = {}

        # Expose function attributes for compatibility with torch.library.custom_op
        # These are set as instance attributes to allow the Kernel to be used
        # as if it were a regular function for introspection purposes
        functools.update_wrapper(self, fn)
        # Manually add function-specific attributes not copied by update_wrapper
        self.__globals__ = fn.__globals__
        self.__code__ = fn.__code__
        self.__defaults__ = fn.__defaults__
        self.__kwdefaults__ = fn.__kwdefaults__

        # Opt-in: register the torch.compile(backend="tpu") capture op now, so an
        # annotated/functional/benchmark-free Pallas kernel is captured with zero
        # warm-up (like a hand-written custom_op). None if it must use first-call
        # registration (unannotated, mutating, or autotuning among configs=[...]).
        self._capture_op: Callable[..., Any] | None = None
        if self.settings.backend == "pallas" and _TPU_COMPILE_CAPTURE:
            from .pallas._tpu_compile_capture import register_decoration_op

            self._capture_op = register_decoration_op(self)

    def kernel_source(self) -> str:
        """
        Return the kernel's source text.

        This is the stable identifier across processes/runs, suitable for
        grouping telemetry rows by kernel during analysis.
        """
        if self._kernel_source is None:
            self._kernel_source = inspect.getsource(self.fn)
        return self._kernel_source

    def _validate_structural_policy(self) -> None:
        # Legacy kernels retain their existing mutable Settings behavior. An
        # explicitly policy-bound kernel cannot silently change its IR contract.
        if self.cute_structural_policy is not None:
            policy_type = get_backend(self.settings.backend).structural_policy_type()
            if policy_type is None:
                raise StructuralPolicyError(
                    "A policy-bound Kernel requires a backend with a structural "
                    f"policy (the {self.settings.backend} backend has none)"
                )
            require_same_structural_policy(
                policy_type.from_settings(self.settings),
                self.cute_structural_policy,
                context="Kernel settings",
            )

    @property
    def cute_structural_policy(self) -> CuteStructuralPolicy | None:
        """The explicitly selected policy, or None for the unchanged legacy API."""
        return self._cute_structural_policy

    def _get_bound_kernel_cache_key(
        self, args: tuple[object, ...], signature: tuple[Hashable, ...]
    ) -> BoundKernelInMemoryCacheKey | None:
        from ..autotuner.base_cache import BoundKernelInMemoryCacheKey

        extra_results = self._stable_specialization_extra_results(args, signature)
        if extra_results is None:
            return None
        compiler_seed_results = self._stable_compiler_seed_results(args, signature)
        if compiler_seed_results is None:
            return None
        return BoundKernelInMemoryCacheKey(
            signature,
            extra_results,
            compiler_seed_results=compiler_seed_results,
        )

    def _stable_specialization_extra_results(
        self,
        args: Sequence[object],
        signature: tuple[Hashable, ...],
    ) -> tuple[Hashable, ...] | None:
        # Evaluate outside the schema lock because some backend extractors may
        # synchronize device data. Schema changes are finite and retry here.
        while True:
            with self._specialize_extra_lock:
                extra_fns = self._specialize_extra.get(signature)
                generation = self._specialization_generation
            extra_results = (
                None if extra_fns is None else tuple(fn(args) for fn in extra_fns)
            )
            with self._specialize_extra_lock:
                if (
                    self._specialization_generation == generation
                    and self._specialize_extra.get(signature) is extra_fns
                ):
                    return extra_results

    def _stable_compiler_seed_results(
        self,
        args: Sequence[object],
        signature: tuple[Hashable, ...],
    ) -> tuple[Hashable, ...] | None:
        while True:
            with self._specialize_extra_lock:
                extractors = self._compiler_seed_specialize_extra.get(signature)
                generation = self._specialization_generation
            results = (
                None
                if extractors is None
                else tuple(extractor(args) for extractor in extractors)
            )
            with self._specialize_extra_lock:
                if (
                    self._specialization_generation == generation
                    and self._compiler_seed_specialize_extra.get(signature)
                    is extractors
                ):
                    return results

    def _runtime_specialization_extensions(
        self,
        signature: tuple[Hashable, ...],
        specializations: Mapping[str, RuntimeInputSpecialization],
    ) -> set[str] | None:
        """New/stronger guard metadata, or None for incompatible classifiers."""
        with self._specialize_extra_lock:
            published = self._specialize_extra.get(signature)
        if published is None:
            return set()
        descriptors = {
            extractor.specialization_key: extractor.specialization
            for extractor in published
            if isinstance(extractor, _RuntimeInputSpecializationExtractor)
        }
        incompatible = [
            key
            for key, specialization in specializations.items()
            if key in descriptors
            and (
                (previous := descriptors[key]) is None
                or not previous.same_binding_semantics(specialization)
            )
        ]
        if not incompatible:
            # Tensor-property snapshots must still describe this exact trace;
            # unused scalar guards may safely remain in the extended schema.
            if any(
                specialization.reusable_tensor_properties and key not in specializations
                for key, specialization in descriptors.items()
                if specialization is not None
            ):
                return None
            extensions = specializations.keys() - descriptors.keys()
            extensions.update(
                key
                for key, specialization in specializations.items()
                if specialization.retains_input_identity
                and (previous := descriptors.get(key)) is not None
                and not previous.retains_input_identity
            )
            return extensions
        with self._specialize_extra_lock:
            new_warnings = [
                key
                for key in incompatible
                if key not in self._warned_uncovered_specializations
            ]
            self._warned_uncovered_specializations.update(new_warnings)
        for key in new_warnings:
            log.warning(
                "Kernel %s requires an uncovered binding specialization %s; "
                "calls using this guard will be traced again to preserve correctness.",
                self.name,
                key,
            )
        return None

    def _create_bound_kernel_cache_key(
        self,
        bound_kernel: BoundKernel,
        args: tuple[object, ...],
        signature: tuple[Hashable, ...],
        *,
        extra_fns: list[Callable[[Sequence[object]], Hashable]] | None = None,
        snapshot_runtime_results: bool = False,
    ) -> BoundKernelInMemoryCacheKey:
        from ..autotuner.base_cache import BoundKernelInMemoryCacheKey

        if extra_fns is None:
            extra_fns = bound_kernel._specialize_extra()
        compiler_seed_fns = bound_kernel._compiler_seed_specialization_extractors
        if not bound_kernel._cache_managed:
            extra_results = tuple(s(args) for s in extra_fns)
            compiler_seed_results = tuple(s(args) for s in compiler_seed_fns)
            return BoundKernelInMemoryCacheKey(
                signature,
                extra_results,
                compiler_seed_results=compiler_seed_results,
            )

        # Autotune cache keys can be generated outside eager binding, so
        # schema synchronization must not wait for the eager bind lock or hold
        # the schema lock while backend extractors run.
        while True:
            with self._specialize_extra_lock:
                if bound_kernel._reset_generation != self._reset_generation:
                    active_extra_fns = extra_fns
                    active_compiler_seed_fns = compiler_seed_fns
                    generation = None
                else:
                    published_extra_fns = self._specialize_extra.get(signature)
                    if published_extra_fns is None:
                        # Fills an unpublished slot, which no memoized fast
                        # dispatch schema references, without a generation bump.
                        self._specialize_extra[signature] = extra_fns
                        self._compiler_seed_specialize_extra[signature] = (
                            compiler_seed_fns
                        )
                        if extra_fns:
                            self._has_specialization_extras = True
                    else:
                        # Late specialization can extend this schema after the
                        # bound is constructed. The published list is the
                        # authoritative schema.
                        extra_fns = published_extra_fns
                    active_compiler_seed_fns = self._compiler_seed_specialize_extra.get(
                        signature
                    )
                    if active_compiler_seed_fns is None:
                        active_compiler_seed_fns = compiler_seed_fns
                        self._compiler_seed_specialize_extra[signature] = (
                            active_compiler_seed_fns
                        )
                    generation = self._specialization_generation
                    active_extra_fns = extra_fns
            extra_results = tuple(s(args) for s in active_extra_fns)
            compiler_seed_results = tuple(s(args) for s in active_compiler_seed_fns)
            cache_key = BoundKernelInMemoryCacheKey(
                signature,
                extra_results,
                compiler_seed_results=compiler_seed_results,
            )
            if generation is None:
                return cache_key
            with self._specialize_extra_lock:
                if bound_kernel._reset_generation != self._reset_generation:
                    return cache_key
                if (
                    self._specialization_generation == generation
                    and self._specialize_extra.get(signature) is active_extra_fns
                    and self._compiler_seed_specialize_extra.get(signature)
                    is active_compiler_seed_fns
                ):
                    if snapshot_runtime_results:
                        bound_kernel._record_runtime_input_specialization_results(
                            active_extra_fns,
                            extra_results,
                        )
                    return cache_key

    def _extend_bound_kernel_specializations(
        self,
        bound_kernel: BoundKernel,
        signature: tuple[Hashable, ...],
        extractors: list[Callable[[Sequence[object]], Hashable]],
        args: Sequence[object],
    ) -> bool:
        if not extractors:
            return False

        from ..autotuner.base_cache import BoundKernelInMemoryCacheKey

        with self._bind_lock:
            if bound_kernel._reset_generation != self._reset_generation:
                return False
            full_args = tuple(args)
            aliases = {
                alias_signature: alias
                for alias_signature, alias in self._specialization_aliases.items()
                if alias.canonical_signature == signature
            }
            # Keep cache-key generation from snapshotting the old schema while
            # this extension is being validated and published.
            with self._specialize_extra_lock:
                updated_extractors = list(self._specialize_extra.get(signature, []))
                for extractor in extractors:
                    if (
                        isinstance(extractor, _RuntimeInputSpecializationExtractor)
                        and extractor.specialization is not None
                        and extractor.specialization.retains_input_identity
                    ):
                        for index, previous in enumerate(updated_extractors):
                            if (
                                isinstance(
                                    previous, _RuntimeInputSpecializationExtractor
                                )
                                and previous.specialization_key
                                == extractor.specialization_key
                                and previous.specialization is not None
                                and previous.specialization.same_binding_semantics(
                                    extractor.specialization
                                )
                            ):
                                # Lifetime facts only grow. Keep the published
                                # classifier and its captured anchors unchanged.
                                updated_extractors[index] = dataclasses.replace(
                                    previous,
                                    specialization=dataclasses.replace(
                                        previous.specialization,
                                        retains_input_identity=True,
                                    ),
                                )
                                break
                        else:
                            updated_extractors.append(extractor)
                    else:
                        updated_extractors.append(extractor)
                compiler_seed_fns = self._compiler_seed_specialize_extra.get(
                    signature,
                    (),
                )
                with unset_fake_temporarily():
                    current_results = tuple(
                        extractor(full_args) for extractor in updated_extractors
                    )
                    current_compiler_seed_results = tuple(
                        extractor(full_args) for extractor in compiler_seed_fns
                    )
                    updated_cache_key = BoundKernelInMemoryCacheKey(
                        signature,
                        current_results,
                        compiler_seed_results=current_compiler_seed_results,
                    )
                    hash(updated_cache_key)
                    for alias_signature, alias in aliases.items():
                        alias_arg_count = len(full_args) - len(alias.trailing_defaults)
                        alias_args = full_args[:alias_arg_count]
                        alias_normalized_args = (
                            *alias_args,
                            *alias.trailing_defaults,
                        )
                        alias_results = tuple(
                            extractor(alias_normalized_args)
                            for extractor in updated_extractors
                        )
                        alias_compiler_seed_results = tuple(
                            extractor(alias_normalized_args)
                            for extractor in compiler_seed_fns
                        )
                        hash(
                            BoundKernelInMemoryCacheKey(
                                alias_signature,
                                (alias_results,),
                                compiler_seed_results=alias_compiler_seed_results,
                            )
                        )

                # Publish the extended specialization only after every extractor
                # succeeds. Otherwise a failed extension could leave a prepared
                # call guarding an already-mutated extractor list.  Replacing
                # published lists requires the generation bump below.
                self._has_specialization_extras = True
                self._specialize_extra[signature] = updated_extractors
                for alias_signature, alias in aliases.items():
                    self._specialize_extra[alias_signature] = [alias]
                    self._compiler_seed_specialize_extra[alias_signature] = (
                        compiler_seed_fns
                    )
                self._specialization_generation += 1
                # Every memoized fast dispatch schema is stale now.
                self._fast_dispatch_schemas = {}

            affected_signatures = {signature, *aliases}
            stale_keys = [
                key
                for key in self._bound_kernels
                if key.specialization_key in affected_signatures
            ]
            for stale_key in stale_keys:
                self._bound_kernels.pop(stale_key)
            for fast_key, cached_bound in list(self._dispatch_cache.items()):
                if cached_bound._base_spec_key == signature:
                    self._dispatch_cache.pop(fast_key)
            self._prepared_call = None
            if bound_kernel._record_runtime_input_specialization_results(
                updated_extractors,
                current_results,
            ):
                self._bound_kernels[updated_cache_key] = bound_kernel
            else:
                # A BoundKernel's generated programs may already consume its
                # construction-time storage facts.  If those facts changed
                # while a late specialization was discovered, do not migrate
                # the existing programs to the new cache identity.  Retire the
                # bound through the same generation check used by reset(); a
                # later call will bind and compile against the extended schema.
                bound_kernel._reset_generation = self._reset_generation - 1
                bound_kernel._direct_prepared_call = None
                bound_kernel._run = None
                bound_kernel._config = None
                bound_kernel._compile_cache.clear()
                bound_kernel._cache_path_map.clear()
            return True

    def _publish_guard_facts(
        self,
        bound_kernel: BoundKernel,
        facts: Sequence[PreparedMetadataSpecializationExtractor],
    ) -> None:
        """Append guard facts to a signature's dispatch schema.

        A guard fact evaluates at any cached bound's bind-time hints, so every
        bound of the signature (the publishing one included) is re-keyed in
        place rather than evicted, and no runtime arguments are needed: code
        generation can publish too.
        """
        from ..autotuner.base_cache import BoundKernelInMemoryCacheKey

        signature = bound_kernel._base_spec_key
        with self._bind_lock, self._specialize_extra_lock:
            if bound_kernel._reset_generation != self._reset_generation:
                return
            published = self._specialize_extra.get(signature, [])
            new_facts = [fact for fact in facts if fact not in published]
            if not new_facts:
                return
            aliases = {
                alias_signature: alias
                for alias_signature, alias in self._specialization_aliases.items()
                if alias.canonical_signature == signature
            }
            for key, bound in list(self._bound_kernels.items()):
                if key.specialization_key not in (signature, *aliases):
                    continue
                del self._bound_kernels[key]
                # Alias keys pack the canonical results; let them rebind.
                if (
                    key.specialization_key == signature
                    and (results := _bound_guarded_input_facts(bound, new_facts))
                    is not None
                ):
                    self._bound_kernels[
                        BoundKernelInMemoryCacheKey(
                            signature,
                            (*key.extra_results, *results),
                            compiler_seed_results=key.compiler_seed_results,
                        )
                    ] = bound
            # Replacing a published list requires the generation bump below.
            self._specialize_extra[signature] = [*published, *new_facts]
            for alias_signature, alias in aliases.items():
                # An alias published while the canonical schema was empty
                # keys no canonical extractor; let it key the new facts.
                if alias_signature in self._specialize_extra:
                    self._specialize_extra[alias_signature] = [alias]
            self._has_specialization_extras = True
            self._specialization_generation += 1
            self._fast_dispatch_schemas = {}
            for fast_key, cached_bound in list(self._dispatch_cache.items()):
                if cached_bound._base_spec_key == signature:
                    del self._dispatch_cache[fast_key]
            self._prepared_call = None

    def _compute_is_distributed(
        self,
        args: Sequence[object],
        *,
        dist_initialized: bool | None = None,
    ) -> bool:
        """Whether this call should compile as a distributed kernel.

        True when the arguments carry symmetric-memory tensors, or the author
        declared distributed intent (``settings.distributed`` or an
        ``hl.ProcessGroupName`` argument) inside an initialized process group.
        This is folded into the specialization key so a symmetric-memory call
        and an identically-shaped ordinary call never share a compiled kernel.
        See GitHub issue #3024.
        """
        if dist_initialized is None:
            dist_initialized = dist.is_initialized()
        return kernel_uses_symm_mem(tuple(args), dist_initialized=dist_initialized) or (
            dist_initialized
            and (self.settings.distributed or self._declares_process_group)
        )

    def _fast_dispatch_key(
        self,
        args: tuple[object, ...],
        *,
        is_distributed: bool | None = None,
        signature: tuple[Hashable, ...] | None = None,
        _extra_guards: list[tuple[Callable[[Sequence[object]], Hashable], Hashable]]
        | None = None,
    ) -> tuple[Hashable, ...] | None:
        """Build a specialization-safe eager key, optionally collecting guards.

        Without ``signature``, a kernel with specialization extras reads them
        through the schema memoized for the key's prefix
        (``_memoized_fast_dispatch_key``); the key's value is the same.
        """
        key: list[Hashable] = []
        has_tensor = False
        # Top-level tensors, for the alias partition below; a container
        # argument defers to the recursive ``_input_tensor_aliases`` walk.
        tensors: list[torch.Tensor] = []
        has_container = False
        has_fragment = False
        # A constexpr-annotated tensor keys the base signature by identity,
        # which its metadata in the prefix does not pin, and an index-less
        # accelerator device resolves against the current device.
        memoizable = True
        for index, a in enumerate(args):
            t = type(a)
            annotation = self._annotations[index] if index < self._num_params else None
            if t is torch.Tensor or t is torch.nn.Parameter:
                tensor = cast("torch.Tensor", a)
                has_tensor = True
                tensors.append(tensor)
                if annotation is ConstExpr:
                    memoizable = False
                si = getattr(tensor, "_dynamo_static_indices", None)
                key.append(
                    (
                        tensor.dtype,
                        tensor.shape,
                        tensor.stride(),
                        tensor.device,
                        None if si is None else frozenset(si),
                    )
                )
            elif t is int or t is float or t is bool:
                key.append((t, _constexpr_key(a)) if annotation is ConstExpr else t)
            elif t is str:
                key.append((t, a))
            elif isinstance(a, ConstExpr):
                constexpr_key = _constexpr_key(a.value)
                has_fragment = has_fragment or (
                    type(a.value) not in (bool, float) and constexpr_key is not a.value
                )
                try:
                    hash(constexpr_key)
                except TypeError:
                    return None
                key.append((t, constexpr_key))
            elif annotation is ConstExpr and t in _BUILTIN_TUNABLE_FRAGMENTS:
                constexpr_key = _constexpr_key(a)
                has_fragment = has_fragment or constexpr_key is not a
                try:
                    hash(constexpr_key)
                except TypeError:
                    return None
                key.append((t, constexpr_key))
            elif a is None:
                key.append(None)
            elif t is torch.dtype:
                key.append(a)
            elif t is torch.device:
                key.append(a)
                device = cast("torch.device", a)
                # (Even behind a tensor, which ``_find_device`` would find
                # first: rare enough to stay conservative.)
                if device.index is None and device.type not in ("cpu", "meta", "mps"):
                    memoizable = False
            elif (
                t is tuple
                or t is list
                or t is dict
                or t is types.FunctionType
                or t is types.BuiltinFunctionType
            ):
                nested = _nested_fast_dispatch_key(
                    a,
                    specialize_numeric=annotation is ConstExpr,
                )
                if nested is None:
                    return None
                nested_key, nested_has_tensor, nested_has_fragment = nested
                key.append(nested_key)
                has_tensor = has_tensor or nested_has_tensor
                has_fragment = has_fragment or nested_has_fragment
                # Callables are flat to the alias walk and the device lookup.
                if t is tuple or t is list or t is dict:
                    has_container = True
            else:
                return None
        if not has_tensor:
            return None
        if is_distributed is None:
            is_distributed = self._compute_is_distributed(args)
        key.append(is_distributed)
        custom_key: object = _CUSTOM_KEY_UNSET
        if self._key_fn is not None:
            # User key functions execute exactly once per launch.
            custom_key = self._key_fn(*args) if signature is None else signature[-1]
            key.append(custom_key)
        # ``_input_tensor_aliases(args)`` without re-walking flat arguments.
        tensor_aliases = (
            _input_tensor_aliases(args) if has_container else _tensor_aliases(tensors)
        )
        if tensor_aliases is not None:
            key.append(("input_tensor_aliases", tensor_aliases))
        if has_fragment:
            if (fragment_aliases := _input_fragment_aliases(args)) is not None:
                key.append(("input_fragment_aliases", fragment_aliases))
        prefix = tuple(key)
        if signature is None:
            if not self._has_specialization_extras:
                return prefix
            # Dynamo would guard on the memo's contents and replay its writes.
            if memoizable and not torch.compiler.is_compiling():
                return self._memoized_fast_dispatch_key(
                    prefix,
                    args,
                    is_distributed,
                    custom_key,
                    _extra_guards,
                    has_container,
                )
            signature = self._base_specialization_key(
                args, is_distributed=is_distributed, custom_key=custom_key
            )
        extra_fns = self._specialize_extra.get(signature)
        if not extra_fns:
            return prefix
        extra_results: list[Hashable] = []
        for extractor in extra_fns:
            result = extractor(args)
            extra_results.append(result)
            if _extra_guards is not None:
                _extra_guards.append((extractor, result))
        return (*prefix, tuple(extra_results))

    def _memoized_fast_dispatch_key(
        self,
        prefix: tuple[Hashable, ...],
        args: tuple[object, ...],
        is_distributed: bool,
        custom_key: object,
        extra_guards: list[tuple[Callable[[Sequence[object]], Hashable], Hashable]]
        | None,
        has_container: bool,
    ) -> tuple[Hashable, ...]:
        """``_fast_dispatch_key``'s value from the schema memoized for ``prefix``.

        The base signature is a function of the prefix and the kernel settings
        it reads: every per-argument component of ``_base_specialization_key``
        derives from the exact metadata/values the prefix records, and the
        alias partition, custom key and ``is_distributed`` are in it.  Its
        device component comes from the first device-bearing argument, which
        the prefix pins for flat arguments, except an index-less accelerator
        ``torch.device`` resolving against the *current* device (such calls
        never get here).  The prefix sorts dict entries, so with container
        arguments a miss memoizes only when every reachable device is one
        indexed device (``_argument_device_is_fixed``).  Nor is a signature
        without a published schema memoized: its first publication does not
        bump the generation.  A hit skips ``_base_specialization_key`` and
        every extractor exact metadata implies.
        """
        settings = self.settings
        # The settings ``_base_specialization_key`` reads.
        base_key_settings = (
            settings.static_shapes,
            settings.index_dtype,
            settings.backend,
            settings.disable_autotuner_heuristics,
        )
        schema = self._fast_dispatch_schemas.get(prefix)
        if (
            schema is not None
            and schema.generation == self._specialization_generation
            and schema.settings == base_key_settings
        ):
            if (key := schema.full_key) is not None:
                extras = schema.results
            else:
                results = list(schema.results)
                for index, extractor in schema.per_call:
                    results[index] = extractor(args)
                extras = tuple(results)
                key = (*prefix, extras)
            if extra_guards is not None:
                extra_guards.extend(zip(schema.extra_fns, extras, strict=True))
            return key
        # Read before the schema: an extension publishes its list before
        # bumping the generation, so a racing resolve can only memoize an
        # entry that is already stale.
        generation = self._specialization_generation
        signature = self._base_specialization_key(
            args, is_distributed=is_distributed, custom_key=custom_key
        )
        extra_fns = self._specialize_extra.get(signature)
        if extra_fns is None:
            return prefix
        if has_container and not _argument_device_is_fixed(args):
            # The signature is not a function of the prefix: evaluate the
            # schema as for an explicit signature.
            extras = tuple(extractor(args) for extractor in extra_fns)
            if extra_guards is not None:
                extra_guards.extend(zip(extra_fns, extras, strict=True))
            return (*prefix, extras) if extra_fns else prefix
        extras, per_call = _evaluate_fast_dispatch_extras(extra_fns, args)
        if extra_guards is not None:
            extra_guards.extend(zip(extra_fns, extras, strict=True))
        schema = _FastDispatchSchema(
            generation, base_key_settings, extra_fns, extras, per_call, prefix
        )
        self._fast_dispatch_schemas[prefix] = schema
        if schema.full_key is not None:
            return schema.full_key
        return (*prefix, extras)

    def _fast_dispatch_key_and_guards(
        self,
        args: tuple[object, ...],
        *,
        is_distributed: bool | None = None,
        signature: tuple[Hashable, ...] | None = None,
    ) -> _FastDispatchEntry | None:
        """
        Build a cheap dispatch key for the fast-path cache in ``__call__``.

        The key records exact per-argument metadata (dtype/shape/stride/device
        for tensors), exact values for specialized arguments, and types for
        ordinary runtime numeric values. It refines the full specialization
        key: any two argument lists that produce different full keys also
        produce different fast keys. That makes it safe to map a fast key
        directly to the BoundKernel that a full ``bind()`` resolved for the
        same arguments.

        If a base signature has extra specialization extractors, their results
        are appended to preserve the same no-collision invariant for
        value-based specializations.

        Returns None when an argument type is not handled (for example, tensor
        subclasses or unsupported objects nested in a container), or when there
        is no tensor argument to pin down the device; callers must then take the
        regular ``bind()`` path.
        """
        specialization_generation = self._specialization_generation
        extra_guards: list[tuple[Callable[[Sequence[object]], Hashable], Hashable]] = []
        key = self._fast_dispatch_key(
            args,
            is_distributed=is_distributed,
            signature=signature,
            _extra_guards=extra_guards,
        )
        if key is None:
            return None
        return _FastDispatchEntry(
            key,
            tuple(extra_guards),
            specialization_generation,
        )

    def _prepare_dispatch_entry(
        self,
        args: tuple[object, ...],
        bound: BoundKernel[_R],
        fast_entry: _FastDispatchEntry,
    ) -> tuple[_PreparedCall | None, bool] | None:
        """Validate and construct eager fast paths from one runtime snapshot."""
        try:
            if bound.env.requires_fresh_binding:
                return None
            if fast_entry.specialization_generation != self._specialization_generation:
                return None
            if bound._reset_generation != self._reset_generation:
                return None
            dist_initialized = dist.is_initialized()
            is_distributed = self._compute_is_distributed(
                args, dist_initialized=dist_initialized
            )
            if fast_entry.key[len(args)] != is_distributed:
                return None
            if self._key_fn is None:
                raw_signature = self._base_specialization_key(
                    args, is_distributed=is_distributed
                )
            else:
                # Reuse the custom key captured by the dispatch lookup; user
                # key functions must execute exactly once per launch.
                raw_signature = self._base_specialization_key(
                    args,
                    is_distributed=is_distributed,
                    custom_key=fast_entry.key[len(args) + 1],
                )
            alias = self._specialization_aliases.get(raw_signature)
            signature = raw_signature if alias is None else alias.canonical_signature
            if signature != bound._base_spec_key:
                return None

            if self._key_fn is None:
                prepared = _PreparedCall.build(
                    self,
                    bound,
                    args,
                    dist_initialized=dist_initialized,
                    extra_guards=fast_entry.extra_guards,
                    is_distributed=is_distributed,
                )
                keyed_direct_dispatch = False
            else:
                prepared = None
                keyed_direct_dispatch = (
                    not self.settings.distributed
                    and not self._declares_process_group
                    and not bound._env._is_distributed
                )
            # Process-group initialization is external to ``_bind_lock``. Do
            # not publish a key assembled across a state transition.
            if dist.is_initialized() != dist_initialized:
                return None
            return prepared, keyed_direct_dispatch
        except Exception:
            # Fast-path publication is optional and must not add a failure
            # after the compiled kernel has already run successfully.
            return None

    def bind(self, args: tuple[object, ...]) -> BoundKernel[_R]:
        """
        Bind the given arguments to the Kernel and return a BoundKernel object.

        Args:
            args: The arguments to bind to the Kernel.

        Returns:
            BoundKernel: A BoundKernel object with the given arguments bound.
        """
        # Dynamo executes bind while capturing the call but cannot trace an RLock.
        # Capture only needs an independent host function and compile environment.
        if torch.compiler.is_compiling():
            return self._bind_isolated(args)
        with self._bind_lock:
            return self._bind(args)

    def _validate_bind_args(
        self, args: tuple[object, ...] | list[object]
    ) -> tuple[object, ...]:
        if not isinstance(args, tuple):
            assert isinstance(args, list), "args must be a tuple or list"
            args = tuple(args)
        if len(args) > self._num_params:
            raise TypeError(
                f"Too many arguments passed to the kernel, expected: {self._num_params} got: {len(args)}."
            )
        return args

    def _bind_isolated(
        self, args: tuple[object, ...], *, rewrite_ast: bool = True
    ) -> BoundKernel[_R]:
        """Construct a canonical bound without reading or publishing shared caches.

        ``rewrite_ast=False`` traces the kernel as written, without the
        backend's AST customizations.
        """
        self._validate_structural_policy()
        args = self._validate_bind_args(args)
        args = self.normalize_args(*args)
        dist_initialized = dist.is_initialized()
        is_distributed = self._compute_is_distributed(
            args, dist_initialized=dist_initialized
        )
        signature = self._base_specialization_key(args, is_distributed=is_distributed)
        return BoundKernel(
            self,
            args,
            base_spec_key=signature,
            is_distributed=is_distributed,
            cache_managed=False,
            rewrite_ast=rewrite_ast,
        )

    def _bind(self, args: tuple[object, ...]) -> BoundKernel[_R]:
        self._validate_structural_policy()
        with measure("Kernel.bind"):
            args = self._validate_bind_args(args)
            dist_initialized = dist.is_initialized()
            is_distributed = self._compute_is_distributed(
                args, dist_initialized=dist_initialized
            )
            signature = self._base_specialization_key(
                args, is_distributed=is_distributed
            )
            cache_key = self._get_bound_kernel_cache_key(args, signature)
            bound_kernel = (
                None if cache_key is None else self._bound_kernels.get(cache_key, None)
            )
            created = False
            if bound_kernel is None:
                normalized_args: tuple[object, ...] = self.normalize_args(*args)
                extra_fns: list[Callable[[Sequence[object]], Hashable]] | None = None
                if len(normalized_args) != len(args):
                    # we had default args that needed to be applied
                    bound_kernel = self._bind(normalized_args)
                    if bound_kernel.env.requires_fresh_binding:
                        return bound_kernel
                    canonical_signature = bound_kernel._base_spec_key
                    alias = self._specialization_aliases.get(signature)
                    if alias is None:
                        trailing_defaults = normalized_args[len(args) :]
                        alias = _SpecializationAlias(
                            self._specialize_extra,
                            canonical_signature,
                            trailing_defaults,
                        )
                        self._specialization_aliases[signature] = alias
                    extra_fns = (
                        [alias]
                        if self._specialize_extra.get(canonical_signature)
                        else []
                    )
                else:
                    bound_kernel = BoundKernel(
                        self,
                        args,
                        base_spec_key=signature,
                        is_distributed=is_distributed,
                    )
                    created = True
                if bound_kernel.env.requires_fresh_binding:
                    # No nonce or growing cache: opaque identity decisions get
                    # a fresh unmanaged binding for this exact invocation.
                    return bound_kernel
                if created and bound_kernel._pending_runtime_specialization_keys:
                    extensions: list[Callable[[Sequence[object]], Hashable]] = [
                        extractor
                        for extractor in bound_kernel._specialize_extra()
                        if isinstance(extractor, _RuntimeInputSpecializationExtractor)
                        and extractor.specialization_key
                        in bound_kernel._pending_runtime_specialization_keys
                    ]
                    # Publish only after construction succeeds. The existing
                    # extension path invalidates cached/prepared schemas and
                    # evicts earlier bindings whose new facts are unknown.
                    self._extend_bound_kernel_specializations(
                        bound_kernel, signature, extensions, args
                    )
                    cache_key = self._get_bound_kernel_cache_key(args, signature)
                if cache_key is None:
                    cache_key = self._create_bound_kernel_cache_key(
                        bound_kernel,
                        args,
                        signature,
                        extra_fns=extra_fns,
                        snapshot_runtime_results=(
                            signature == bound_kernel._base_spec_key
                        ),
                    )
                elif signature == bound_kernel._base_spec_key:
                    published_extra_fns = self._specialize_extra.get(signature)
                    if published_extra_fns is not None:
                        bound_kernel._record_runtime_input_specialization_results(
                            published_extra_fns,
                            cache_key.extra_results,
                        )
                self._bound_kernels[cache_key] = bound_kernel
                if created:
                    # Key the input-shape guards this compile recorded.
                    bound_kernel._publish_guarded_input_facts()
            self._retain_opaque_binding(bound_kernel)
            return bound_kernel

    def _retain_opaque_binding(self, bound: BoundKernel) -> None:
        """Bound owner retention for identity-sensitive opaque helper calls.

        Called under the bind lock. Keep insertion order: a fast-path hit need
        not acquire a new lock merely to update the cache's eviction order.
        Explicitly retained BoundKernel objects remain usable after eviction.
        """
        if not bound.env.has_opaque_fragment_inputs:
            return
        self._opaque_bindings.setdefault(bound, None)
        while len(self._opaque_bindings) > _OPAQUE_BINDING_CACHE_SIZE:
            evicted, _ = self._opaque_bindings.popitem(last=False)
            # Autotuning publishes its own selection. Eviction must not turn a
            # bound-only set_config override into a winner for other bindings.
            for key, cached in list(self._bound_kernels.items()):
                if cached is evicted:
                    del self._bound_kernels[key]
            for key, cached in list(self._dispatch_cache.items()):
                if cached is evicted:
                    del self._dispatch_cache[key]
            if self._prepared_call is not None and self._prepared_call.bound is evicted:
                self._prepared_call = None
            # Memoized classifier results can themselves own the fragment.
            self._fast_dispatch_schemas.clear()

    def _base_specialization_key(
        self,
        args: Sequence[object],
        *,
        is_distributed: bool | None = None,
        custom_key: object = _CUSTOM_KEY_UNSET,
    ) -> tuple[Hashable, ...]:
        """
        Generate the base specialization key from input argument metadata only,
        using the per-type extractor functions defined in `_specialization_extractors`,
        without any extras discovered during compilation. Used internally for
        _specialize_extra lookups.
        """
        result: list[Hashable] = []
        if self.cute_structural_policy is not None:
            result.append(
                ("cute_structural_policy", self.cute_structural_policy.identity())
            )
        assert len(args) <= len(self._annotations)
        for value, annotation in zip(args, self._annotations, strict=False):
            if isinstance(value, ConstExpr):
                result.append(_constexpr_key(value.value))
            elif annotation is ConstExpr:
                result.append(_constexpr_key(value))
            else:
                result.append(self._specialization_key(value))
        if (tensor_aliases := _input_tensor_aliases(args)) is not None:
            result.append(("input_tensor_aliases", tensor_aliases))
        if (fragment_aliases := _input_fragment_aliases(args)) is not None:
            result.append(("input_fragment_aliases", fragment_aliases))
        device_type, device_capability, promotion_hardware_key = (
            _device_specialization_key(
                args,
                backend=self.settings.backend,
                compiler_heuristics_enabled=(
                    not self.settings.disable_autotuner_heuristics
                ),
            )
        )
        promotion_key = (promotion_hardware_key,) if promotion_hardware_key else ()
        if is_distributed is None:
            is_distributed = self._compute_is_distributed(args)
        if self._key_fn is not None:
            if custom_key is _CUSTOM_KEY_UNSET:
                custom_key = self._key_fn(*args)
            return (
                *result,
                device_type,
                device_capability,
                *promotion_key,
                is_distributed,
                cast("Hashable", custom_key),
            )
        return (
            *result,
            device_type,
            device_capability,
            *promotion_key,
            is_distributed,
        )

    def specialization_key(self, args: Sequence[object]) -> tuple[Hashable, ...]:
        """
        Generate the full specialization key for the given arguments, including
        any additional specialization constraints discovered during compilation
        (e.g. from hl.specialize() calls).

        Before the first compilation, these extras are not yet known and the
        key may be incomplete.

        Args:
            args: The arguments to generate a specialization key for.

        Returns:
            Hashable: A hashable key representing the specialization of the arguments.
        """
        base = self._base_specialization_key(args)
        extra_results = self._stable_specialization_extra_results(args, base)
        compiler_seed_results = self._stable_compiler_seed_results(args, base)
        compiler_seed_key = (
            ()
            if compiler_seed_results is None or not compiler_seed_results
            else (("compiler_seed_results", compiler_seed_results),)
        )
        return (*base, *compiler_seed_key, *(extra_results or ()))

    def _specialization_key(self, obj: object) -> Hashable:
        """
        Helper used to generate a specialization key for the given object.

        This method determines a unique key for the object based on its type
        and the corresponding extractor function defined in `_specialization_extractors`.

        Args:
            obj: The argument to generate a specialization key for.

        Returns:
            Hashable: A hashable key representing the specialization of the object.
        """
        extractor = _specialization_extractors.get(type(obj))
        if extractor is None:
            if isinstance(obj, torch.fx.GraphModule):
                # GraphModule subclasses need special handling
                extractor = _specialization_extractors[torch.fx.GraphModule]
            elif isinstance(obj, torch.Tensor):
                # torch.Tensor subclasses (e.g. the JAX-export adapter)
                # share the standard tensor specialization key. Use the
                # SymInt-safe extractor: unlike exact ``torch.Tensor``,
                # subclasses may carry symbolic sizes/strides.
                extractor = _specialization_extractors["tensor_subclass"]
            elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
                # this is a namedtuple
                extractor = _specialization_extractors["namedtuple"]
            elif dataclasses.is_dataclass(obj):
                extractor = _specialization_extractors["dataclass"]
            else:
                raise TypeError(f"unsupported argument type: {type(obj).__name__}")
        return extractor(self, obj)

    def normalize_args(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        """
        Normalize the given arguments and keyword arguments according to the function signature.

        Args:
            args: The positional arguments to normalize.
            kwargs: The keyword arguments to normalize.

        Returns:
            tuple[object, ...]: A tuple of normalized positional arguments.
        """
        key = (len(args), tuple(kwargs))
        plans = self._normalize_args_plans
        if torch.compiler.is_compiling():
            # Dynamo would guard on the plan cache's contents and replay the
            # write, recompiling once per call shape; trace inspect instead.
            plan = None
        elif key in plans:
            plan = plans[key]
        else:
            plan = plans[key] = self._plan_normalize_args(*key)
        if plan is None:
            # Calls that ``inspect`` must reject (or that need its binding
            # rules) raise its errors.
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return tuple(bound_args.args)
        defaults, positions = plan
        normalized = [*args, *defaults[len(args) :]]
        for position, value in zip(positions, kwargs.values(), strict=True):
            normalized[position] = value
        return tuple(normalized)

    def _plan_normalize_args(
        self, positional_count: int, keyword_names: tuple[str, ...]
    ) -> tuple[tuple[object, ...], tuple[int, ...]] | None:
        """Defaults template and keyword slots for one call shape, if bindable.

        ``inspect.Signature.bind`` costs ~10us per call; a call shape (count of
        positional arguments plus the keyword names) that binds cleanly to a
        positional-or-keyword-only signature can be replayed from this plan.
        """
        parameters = list(self.signature.parameters.values())
        if positional_count > len(parameters) or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        ):
            return None
        names = [parameter.name for parameter in parameters]
        positions: list[int] = []
        for name in keyword_names:
            if name not in names or names.index(name) < positional_count:
                return None
            positions.append(names.index(name))
        defaults = tuple(parameter.default for parameter in parameters)
        if any(
            default is inspect.Parameter.empty
            for index, default in enumerate(defaults)
            if index >= positional_count and index not in positions
        ):
            return None
        return defaults, tuple(positions)

    def autotune(
        self,
        args: Sequence[object],
        *,
        force: bool = True,
        **options: object,
    ) -> Config:
        """
        Perform autotuning to find the optimal configuration for the kernel.  This uses the
        default setting, you can call helion.autotune.* directly for more customization.

        If config= or configs= is provided to helion.kernel(), the search will be restricted to
        the provided configs.  Use force=True to ignore the provided configs.

        Mutates (the bound version of) self so that `__call__` will run the best config found.

        Args:
            args: Example arguments used for benchmarking during autotuning.
            force: If True, force full autotuning even if a config is provided.
            options: Additional keyword options forwarded to the autotuner.

        Returns:
            Config: The best configuration found during autotuning.
        """
        args = self.normalize_args(*args)
        return self.bind(args).autotune(args, force=force, **options)

    def autotune_multi(
        self,
        arg_sets: Sequence[Sequence[object]],
        *,
        aggregation: Literal["geomean", "max"] = "geomean",
        relative_to: Literal["default", "baseline"] | None = None,
        cache_tag: str | None = None,
        force: bool = True,
        **options: object,
    ) -> Config:
        """Find one config using an objective measured across several inputs.

        The first argument set anchors config generation. Each candidate is measured
        on every set, then reduced with a geometric mean or maximum. ``relative_to``
        optionally optimizes per-shape latency relative to each shape's default config
        or custom baseline. Only the supplied bound specializations are configured.

        Every argument set must bind normally to the same exact, current accelerator
        device. Distributed processes are not supported. A non-empty ``cache_tag`` is
        required for custom callbacks, dynamic-shape tuning, and runtime numeric
        arguments; callers own tag invalidation in those cases. Structural differences
        in per-operation fields are allowed when a scalar override broadcasts across
        every operation.

        Args:
            arg_sets: Non-empty sequence of representative kernel argument sequences.
            aggregation: Joint objective, either ``"geomean"`` or ``"max"``.
            relative_to: Optional ``"default"`` or ``"baseline"`` normalization.
            cache_tag: User-managed cache discriminator for dynamic shapes, runtime
                numeric arguments, or callbacks.
            force: If true, ignore pinned configs and cache reads during the search.
            options: Additional options forwarded to the registered autotuner.

        Returns:
            The config selected for all supplied specializations.
        """
        from ..autotuner.benchmark_provider import LocalBenchmarkProvider
        from ..autotuner.benchmark_provider import _has_valid_multi_shape_measurement
        from ..autotuner.benchmark_provider import _materialize_multi_shape_config
        from ..autotuner.benchmark_provider import _MultiShapeAutotuneArgs
        from .settings import default_autotuner_fn

        if aggregation not in ("geomean", "max"):
            raise exc.InvalidAPIUsage(
                "autotune_multi aggregation must be 'geomean' or 'max'"
            )
        if relative_to not in (None, "default", "baseline"):
            raise exc.InvalidAPIUsage(
                "autotune_multi relative_to must be None, 'default', or 'baseline'"
            )
        if cache_tag is not None and (not isinstance(cache_tag, str) or not cache_tag):
            raise exc.InvalidAPIUsage(
                "autotune_multi cache_tag must be a non-empty string"
            )
        if (
            not isinstance(arg_sets, Sequence)
            or isinstance(arg_sets, (str, bytes))
            or not arg_sets
        ):
            raise exc.InvalidAPIUsage(
                "autotune_multi arg_sets must be a non-empty sequence"
            )
        if dist.is_initialized():
            raise exc.InvalidAPIUsage(
                "autotune_multi does not support an initialized distributed process group"
            )
        if not get_backend(self.settings.backend).supports_autotune_multi:
            raise exc.InvalidAPIUsage(
                f"autotune_multi does not support backend {self.settings.backend!r}"
            )
        if self.settings.autotuner_fn is not default_autotuner_fn:
            raise exc.InvalidAPIUsage(
                "autotune_multi requires the default registered autotuner"
            )
        if self.settings.autotune_cache == "AOTAutotuneCache":
            raise exc.InvalidAPIUsage(
                "autotune_multi does not support AOTAutotuneCache"
            )
        if self.settings.autotune_cache not in {
            "LocalAutotuneCache",
            "StrictLocalAutotuneCache",
            "RemoteAutotuneCache",
            "StrictRemoteAutotuneCache",
        }:
            raise exc.InvalidAPIUsage(
                "autotune_multi requires a built-in local or remote best-config cache"
            )
        if "benchmark_provider_cls" in options:
            if options.pop("benchmark_provider_cls") is not LocalBenchmarkProvider:
                raise exc.InvalidAPIUsage(
                    "autotune_multi does not support a custom benchmark provider"
                )
        if relative_to == "baseline" and self.settings.autotune_baseline_fn is None:
            raise exc.InvalidAPIUsage(
                "autotune_multi relative_to='baseline' requires autotune_baseline_fn"
            )
        if self.settings.autotune_benchmark_input_banks > 1:
            raise exc.InvalidAPIUsage(
                "autotune_multi does not support rotating benchmark input banks"
            )
        if self.settings.autotune_benchmark_fn is not None:
            raise exc.InvalidAPIUsage(
                "autotune_multi does not support autotune_benchmark_fn"
            )

        custom_callbacks = (
            self.settings.autotune_baseline_fn,
            self.settings.autotune_baseline_accuracy_check_fn,
            self.settings.autotune_config_filter,
        )
        if cache_tag is None and any(fn is not None for fn in custom_callbacks):
            raise exc.InvalidAPIUsage(
                "autotune_multi requires cache_tag when a custom baseline, "
                "accuracy check, or config filter is configured"
            )
        if cache_tag is None and not self.settings.static_shapes:
            raise exc.InvalidAPIUsage(
                "autotune_multi requires cache_tag when static_shapes=False"
            )

        normalized_arg_sets: list[tuple[object, ...]] = []
        for case_index, arg_set in enumerate(arg_sets):
            if not isinstance(arg_set, Sequence) or isinstance(arg_set, (str, bytes)):
                raise exc.InvalidAPIUsage(
                    f"autotune_multi arg_sets[{case_index}] must be a sequence"
                )
            try:
                normalized_arg_sets.append(self.normalize_args(*arg_set))
            except TypeError as error:
                raise exc.InvalidAPIUsage(
                    f"autotune_multi arg_sets[{case_index}] does not match the "
                    f"kernel signature: {error}"
                ) from error

        if cache_tag is None:
            for case_index, normalized_args in enumerate(normalized_arg_sets):
                if any(
                    annotation is not ConstExpr
                    and _has_unspecialized_numeric_value(value)
                    for value, annotation in zip(
                        normalized_args, self._annotations, strict=True
                    )
                ):
                    raise exc.InvalidAPIUsage(
                        "autotune_multi requires cache_tag when an argument set "
                        "contains a runtime numeric value; "
                        f"arg_sets[{case_index}] does"
                    )

        cases: list[tuple[BoundKernel[_R], tuple[object, ...]]] = []
        case_keys: list[tuple[object, ...]] = []
        fresh_bounds: list[tuple[BoundKernel[_R], Hashable | None]] = []
        for case_index, normalized_args in enumerate(normalized_arg_sets):
            try:
                bound_kernel = self.bind(normalized_args)
            except exc.NoTensorArgs as error:
                raise exc.InvalidAPIUsage(
                    "autotune_multi requires each argument set to have a device "
                    f"discoverable by normal kernel binding; arg_sets[{case_index}] did not"
                ) from error
            signature = self._base_specialization_key(normalized_args)
            case_key = self._get_bound_kernel_cache_key(normalized_args, signature)
            cases.append((bound_kernel, normalized_args))
            if bound_kernel.env.requires_fresh_binding:
                # Snapshot compatibility before code generation mutates any
                # per-config facts. The property retains immutable keys only.
                fresh_bounds.append((bound_kernel, bound_kernel._fresh_selection_key))
                # Invocation-local only: the no-I/O wrapper below must not turn
                # this ordinal into a persistent specialization identity.
                case_keys.append(("fresh_case", case_index))
            else:
                assert case_key is not None
                case_keys.append(
                    (
                        case_key.specialization_key,
                        case_key.extra_results,
                        case_key.compiler_seed_results,
                    )
                )

        anchor = cases[0][0]
        canonical_device = _canonicalize_multi_shape_device(anchor.env.device)
        current_index = _current_device_index(canonical_device.type)
        if canonical_device.index != current_index:
            raise exc.InvalidAPIUsage(
                "autotune_multi requires the indexed accelerator to be current: "
                f"got {canonical_device}, current index is {current_index}"
            )

        unique_bound_kernels: list[BoundKernel[_R]] = []
        seen_bound_kernel_ids: set[int] = set()
        for bound_kernel, _ in cases:
            if id(bound_kernel) not in seen_bound_kernel_ids:
                seen_bound_kernel_ids.add(id(bound_kernel))
                unique_bound_kernels.append(bound_kernel)

        anchor_backend = anchor.env.backend.name
        anchor_capability = target_device_capability(anchor.env.device)
        advanced_controls_files = self.settings.autotune_search_acf or None
        broadcastable_per_operation_fields = frozenset(
            {
                "indexing",
                "atomic_indexing",
                "load_eviction_policies",
            }
        )
        scalar_overridden_config_keys = frozenset(
            key
            for key, value in (self.settings.autotune_config_overrides or {}).items()
            if key in broadcastable_per_operation_fields and isinstance(value, str)
        )

        def effective_fingerprint(
            bound_kernel: BoundKernel[_R],
        ) -> tuple[tuple[str | int, ...], ...]:
            return tuple(
                field
                for field in bound_kernel.config_spec.structural_fingerprint(
                    advanced_controls_files=advanced_controls_files
                )
                if field[0] not in scalar_overridden_config_keys
            )

        anchor_fingerprint = effective_fingerprint(anchor)
        for case_index, (bound_kernel, _) in enumerate(cases):
            if bound_kernel.env.process_group_name is not None:
                raise exc.InvalidAPIUsage(
                    "autotune_multi does not support a bound kernel with a process group"
                )
            bound_device = _canonicalize_multi_shape_device(bound_kernel.env.device)
            if bound_device != canonical_device:
                raise exc.InvalidAPIUsage(
                    "autotune_multi bound a different device for "
                    f"arg_sets[{case_index}]: {bound_device} != {canonical_device}"
                )
            if bound_kernel.env.backend.name != anchor_backend:
                raise exc.InvalidAPIUsage(
                    "autotune_multi requires every case to use the same backend"
                )
            if target_device_capability(bound_kernel.env.device) != anchor_capability:
                raise exc.InvalidAPIUsage(
                    "autotune_multi requires every case to have the same device capability"
                )
            fingerprint = effective_fingerprint(bound_kernel)
            if fingerprint != anchor_fingerprint:
                raise exc.InvalidAPIUsage(
                    "autotune_multi requires structurally compatible ConfigSpec "
                    f"instances; arg_sets[{case_index}] is incompatible with the anchor"
                )
        multi_args = _MultiShapeAutotuneArgs(
            cases=tuple(cases),
            aggregation=aggregation,
            relative_to=relative_to,
            cache_tag=cache_tag,
            workload_key=(
                "multi_shape:v1",
                aggregation,
                relative_to,
                cache_tag,
                tuple(case_keys),
            ),
            reference_latencies=None,
        )

        ephemeral = anchor.env.backend.make_ephemeral_cache()
        ctx = ephemeral if ephemeral is not None else contextlib.nullcontext()
        with ctx:
            config = anchor.env.backend.autotune(
                anchor,
                cast("Sequence[object]", multi_args),
                force=force,
                **options,
            )
        if multi_args.search_started and not multi_args.found_valid_config:
            raise exc.NoConfigFound
        if multi_args.search_started and not _has_valid_multi_shape_measurement(
            multi_args,
            anchor.config_spec,
            config,
        ):
            raise exc.NoConfigFound
        winner = _materialize_multi_shape_config(anchor.config_spec, config)
        if ephemeral is not None:
            for bound_kernel in unique_bound_kernels:
                bound_kernel.env.backend.finalize_ephemeral_cache(bound_kernel, winner)

        for bound_kernel in unique_bound_kernels:
            bound_kernel.compile_config(winner)
        for bound_kernel in unique_bound_kernels:
            bound_kernel.set_config(winner)
        for bound_kernel, selection_key in fresh_bounds:
            bound_kernel._remember_fresh_selection(selection_key)
        return winner

    def __call__(self, *args: object, **kwargs: object) -> _R:
        """
        Call the Kernel with the given arguments and keyword arguments.

        Args:
            args: The positional arguments to pass to the Kernel.
            kwargs: The keyword arguments to pass to the Kernel.

        Returns:
            _R: The result of the Kernel function call.
        """
        self._validate_structural_policy()
        if kwargs:
            args = self.normalize_args(*args, **kwargs)
        is_compiling = torch.compiler.is_compiling()
        if (
            not is_compiling
            and (prepared := self._prepared_call) is not None
            and prepared.matches(self, args)
            and prepared.bound._run is not None
        ):
            return prepared.bound._run(*args)
        # Runtime specialization extractors can inspect pointers/storage offsets.
        # Capture may reuse metadata-only entries; legacy Dynamo integration
        # needs their compiled launchers and cannot trace isolated binding.
        if self._dispatch_cache and (
            not is_compiling
            or all(
                _capture_safe_specialization(extractor)
                for extractors in self._specialize_extra.values()
                for extractor in extractors
            )
        ):
            # Fast path: repeat call with argument metadata seen before. The
            # cache is only populated by calls that already took the slow
            # path below, so hitting it cannot skip autotuning/compilation.
            # (The cache stays empty under TPU compile capture, so this
            # cannot bypass auto_capture_call below.)
            fast_entry: _FastDispatchEntry | None = None
            if (
                self._key_fn is not None
                and not self.settings.distributed
                and not self._declares_process_group
            ):
                # Keyed kernels cannot use a prepared call. Build their legacy
                # exact key without allocating guard metadata on every launch.
                specialization_generation = self._specialization_generation
                dist_initialized = dist.is_initialized()
                is_distributed = self._compute_is_distributed(
                    args, dist_initialized=dist_initialized
                )
                fast_key = self._fast_dispatch_key(args, is_distributed=is_distributed)
                if fast_key is not None:
                    bound = self._dispatch_cache.get(fast_key)
                    if bound is not None and bound._run is not None:
                        run = bound._run
                        if is_compiling:
                            return run(*args)
                        if (
                            bound._dispatch_generation == specialization_generation
                            and bound._run is run
                            and self._specialization_generation
                            == specialization_generation
                            and dist.is_initialized() == dist_initialized
                        ):
                            return run(*args)
                    fast_entry = _FastDispatchEntry(
                        fast_key,
                        (),
                        specialization_generation,
                    )
            if fast_entry is None:
                fast_entry = self._fast_dispatch_key_and_guards(args)
            if fast_entry is not None:
                fast_key = fast_entry.key
                bound = self._dispatch_cache.get(fast_key)
                if bound is not None and bound._run is not None:
                    run = bound._run
                    if is_compiling:
                        return run(*args)
                    # A compilation can discover a late specialization while a
                    # different thread is reading this cache. Revalidate fast-
                    # path state while the mapping is still current; the same
                    # lock protects specialization-driven invalidation.
                    with self._bind_lock:
                        if (
                            self._dispatch_cache.get(fast_key) is bound
                            and bound._run is run
                        ):
                            entry = self._prepare_dispatch_entry(
                                args,
                                bound,
                                fast_entry,
                            )
                            if entry is not None:
                                self._prepared_call = entry[0]
                                if entry[1]:
                                    bound._dispatch_generation = (
                                        fast_entry.specialization_generation
                                    )
                            else:
                                run = None
                        else:
                            run = None
                    if run is not None:
                        return run(*args)
        if self.settings.backend == "pallas" and _TPU_COMPILE_CAPTURE:
            # Local import: _tpu_compile_capture pulls in the dynamo HOP machinery,
            # not ready when kernel.py first loads during ``import helion``.
            from .pallas._tpu_compile_capture import RUN_NORMAL
            from .pallas._tpu_compile_capture import auto_capture_call

            result = auto_capture_call(self, args)
            if result is not RUN_NORMAL:
                return cast("_R", result)
            return self.bind(args)(*args)
        bound = self.bind(args)
        if bound.env.requires_fresh_binding:
            return bound._call_with_bound_args(*args)
        result = bound(*args)
        if is_compiling:
            return result
        # Avoid a second bind for signatures the fast paths cannot support.
        # This cheap key check also captures a custom key exactly once.
        fast_entry = self._fast_dispatch_key_and_guards(args)
        if fast_entry is not None:
            fast_key = fast_entry.key
            # Resolve any late specialization and publish both fast paths as
            # one atomic update with respect to discovery and reset.
            with self._bind_lock:
                if self._has_specialization_extras:
                    bound = self._bind(args)
                if bound._run is not None:
                    entry = self._prepare_dispatch_entry(
                        args,
                        bound,
                        fast_entry,
                    )
                    if entry is not None:
                        self._dispatch_cache[fast_key] = bound
                        self._prepared_call = entry[0]
                        if entry[1]:
                            bound._dispatch_generation = (
                                fast_entry.specialization_generation
                            )
        return result

    def reset(self) -> None:
        """
        Clears the cache of bound kernels, meaning subsequent calls will
        recompile and re-autotune.
        """
        with self._bind_lock:
            self._bound_kernels.clear()
            self._opaque_bindings.clear()
            self._fresh_selections.clear()
            self._dispatch_cache.clear()
            self._prepared_call = None
            # Specialization extractors are discovered by tracing the host
            # function and can change after an explicit reset. Keeping the old
            # schema could hide newly discovered hl.specialize() calls.
            with self._specialize_extra_lock:
                # Replace rather than clear: in-flight omitted-default aliases
                # retain the old schema mapping while new work starts fresh.
                # The generation bump below retires every published list.
                self._specialize_extra = {}
                self._compiler_seed_specialize_extra = {}
                self._specialization_aliases = {}
                self._cute_grouped_static_tail_extra_descriptors = {}
                self._cute_reduction_extent_descriptors = {}
                self._has_specialization_extras = False
                self._specialization_generation += 1
                self._reset_generation += 1
                self._fast_dispatch_schemas = {}

    @property
    def jax_fn(self) -> Callable[..., Any]:
        """A pure-JAX callable view of this Helion kernel.

        Pallas-backend kernels can be called directly with JAX arrays
        or tracers (i.e. inside ``jax.jit``) by going through this
        property.  The kernel's compile/specialize path runs the first
        time the callable is invoked; subsequent calls reuse the
        cached compilation just like ``__call__``.

        This only supports kernels compiled for the Pallas backend.
        """
        from .pallas.jax_export import make_jax_fn

        cached = getattr(self, "_jax_fn_callable", None)
        if cached is not None:
            return cast("Callable[..., Any]", cached)
        rv = make_jax_fn(self)
        # pyrefly: ignore [unsupported-attribute-set]
        self._jax_fn_callable = rv
        return rv


class BoundKernel(_AutotunableKernel, Generic[_R]):
    def __init__(
        self,
        kernel: Kernel[_R],
        args: tuple[object, ...],
        *,
        base_spec_key: tuple[Hashable, ...] | None = None,
        is_distributed: bool | None = None,
        cache_managed: bool = True,
        rewrite_ast: bool = True,
    ) -> None:
        """
        Initialize a BoundKernel object.

        This constructor sets up the environment, compiles the kernel function, and prepares
        the arguments for execution.

        Args:
            kernel: The Kernel object to bind.
            args: A tuple of arguments to bind to the kernel.
            cache_managed: Whether this bound participates in the kernel's shared
                specialization and bound caches.
            rewrite_ast: Whether to apply the backend's AST customizations.
        """
        super().__init__()
        kernel._validate_structural_policy()
        self.kernel = kernel
        self._reset_generation = kernel._reset_generation
        # Extending this bound's schema evicts all of its dispatch mappings.
        self._dispatch_generation: int | None = None
        # ``Kernel.__call__`` owns a shared prepared fast path, but callers may
        # also retain and invoke a BoundKernel directly (benchmark harnesses do
        # this deliberately). Keep the latest validated direct-call guard here.
        self._direct_prepared_call: _PreparedCall | None = None
        self._cache_managed = cache_managed
        self._pending_runtime_specialization_keys: set[str] = set()
        self._has_input_fragment_aliases = _input_fragment_aliases(args) is not None
        if is_distributed is None:
            dist_initialized = dist.is_initialized()
            is_distributed = kernel._compute_is_distributed(
                args, dist_initialized=dist_initialized
            )
        # Base specialization key from the REAL args (the same key bind() uses to
        # distinguish BoundKernels).  Reused in compile_config as the PyCodeCache
        # ``extra`` discriminator for backends that cache shape-specific module state
        # (Backend.requires_shape_specialized_module).  Must be computed on the real
        # args: fake_args turn scalar int/float/bool into SymInt/SymFloat/SymBool,
        # which the specialization extractors reject.
        self._base_spec_key = (
            kernel._base_specialization_key(args, is_distributed=is_distributed)
            if base_spec_key is None
            else base_spec_key
        )
        self._run: Callable[..., _R] | None = None
        self._config: Config | None = None
        self._compiler_seed_specialization_extractors: tuple[
            _CompilerSeedSpecializationExtractor, ...
        ] = ()
        self._compiler_seed_specialization_results: tuple[Hashable, ...] = ()
        # Direct BoundKernel calls bypass Kernel's dispatch cache. Remember the
        # first device-bearing argument path and the immutable SM-derived facts
        # per device so their hot path does not recursively walk args or query
        # device properties on every launch.
        self._compiler_seed_device_resolver: _ArgumentDeviceResolver | None = None
        self._compiler_seed_device_results: dict[
            torch.device,
            tuple[Hashable | None, ...],
        ] = {}
        self._compile_cache: ConfigCache[CompiledConfig] = ConfigCache()
        # Source an autotune render worker produced, keyed like the compile
        # cache and consumed by the next compile_config of that config.
        self._primed_renders: ConfigCache[str] = ConfigCache()
        # Configs whose cached callable was loaded from a worker's source: the
        # facts code generation records on this kernel (see to_code) are
        # missing until the parent renders them itself.
        self._render_only_configs: ConfigCache[None] = ConfigCache()
        # Shape guards recorded past this index come from code generation.
        self._bind_guard_count = 0
        # Backend specialization descriptors recorded per generated config
        # (``Backend.after_codegen``; CuTe: hint-sized reduction extents).
        self._cute_reduction_extents: ConfigCache[tuple[Hashable, ...]] = ConfigCache()
        self._cache_path_map: ConfigCache[str | None] = ConfigCache()
        self._host_semantic_fingerprints: dict[
            tuple[_HostSemanticInputNormalization, tuple[object, ...]], str
        ] = {}
        self._fresh_input_layouts: tuple[Hashable, ...] | None = None
        # Direct to_code() has no call arguments, so keep this bound kernel's
        # construction-time tensor values as its stable weak fallback.
        self._runtime_tensor_refs_by_name = {
            name: weakref.ref(value)
            for name, value in zip(self.kernel.signature.parameters, args, strict=False)
            if isinstance(value, torch.Tensor)
        }
        self._first_compile_lock = threading.RLock()
        self._backward_compiled: (
            tuple[Kernel[object], str, BoundKernel[object]] | None
        ) = None
        # Distributed detection lives on Kernel so the same value feeds both the
        # specialization cache key and CompileEnvironment gating; that keeps a
        # symmetric-memory call from ever aliasing a non-distributed compiled
        # kernel of the same shape. See GitHub issue #3024.
        self._env = CompileEnvironment(
            _canonicalize_argument_device(_find_device(args)),
            self.kernel.settings,
            index_dtype=_resolve_index_dtype(self.kernel.settings, args),
            is_distributed=is_distributed,
        )
        # The structural policy the traced IR will follow (None for backends
        # whose config schema does not depend on Settings).
        policy_type = self._env.backend.structural_policy_type()
        self._structural_policy = (
            None if policy_type is None else policy_type.from_settings(kernel.settings)
        )

        if is_ref_mode_enabled(self.kernel.settings):
            self.fake_args = []  # type: ignore[assignment]
            self.host_function = None  # type: ignore[assignment]
            return

        self._env.backend.on_bind(self, args, rewrite_ast=rewrite_ast)

        with self.env:
            self._env.process_group_name = _find_process_group_name(
                kernel.fn, args, is_distributed
            )
            assert len(args) == len(self.kernel.signature.parameters)
            self.fake_args: list[object] = []
            constexpr_args = {}
            for name, arg, annotation in zip(
                self.kernel.signature.parameters,
                args,
                self.kernel._annotations,
                strict=False,
            ):
                if isinstance(arg, ConstExpr):
                    assert not isinstance(arg.value, torch.Tensor), (
                        "ConstExpr cannot be a tensor"
                    )
                    self.fake_args.append(arg.value)
                    constexpr_args[name] = arg.value
                elif annotation is ConstExpr:
                    assert not isinstance(arg, torch.Tensor), (
                        "ConstExpr cannot be a tensor"
                    )
                    self.fake_args.append(arg)
                    constexpr_args[name] = arg
                else:
                    self.fake_args.append(self.env.to_fake(arg, ArgumentOrigin(name)))

            self._apply_mark_static(args)
            runtime_args = dict(
                zip(self.kernel.signature.parameters, args, strict=False)
            )

            with (
                maybe_skip_dtype_check_in_meta_registrations(),
                patch_inductor_lowerings(),
                measure("BoundKernel.create_host_function"),
            ):
                try:
                    compiler = KernelCompiler(self.env)
                    with self.env.use_runtime_arg_values(runtime_args):
                        self.host_function: HostFunction = compiler.compile(
                            self.kernel.fn,
                            self.fake_args,
                            constexpr_args,
                            rewrite_ast=rewrite_ast,
                        )
                except Exception:
                    config = self.env.config_spec.default_config()
                    self.maybe_log_repro(log.warning, args, config=config)
                    raise

                if self._cache_managed:
                    if any(
                        isinstance(extractor, _RuntimeInputSpecializationExtractor)
                        and extractor.specialization is not None
                        and extractor.specialization.retains_input_identity
                        for extractor in self.kernel._specialize_extra.get(
                            self._base_spec_key, ()
                        )
                    ):
                        self.env.has_opaque_fragment_inputs = True
                    extensions = self.kernel._runtime_specialization_extensions(
                        self._base_spec_key, self.env.runtime_input_specializations
                    )
                    if extensions is None:
                        # A changed classifier or anchor cannot be merged merely
                        # because its source-site key happens to match.
                        self.env.requires_fresh_binding = True
                    else:
                        self._pending_runtime_specialization_keys = extensions
                if self.env.requires_fresh_binding:
                    # Frontend identity folding can disable shared reuse. Set
                    # this before snapshots/seeds, just as for isolated binds.
                    self._cache_managed = False
                if (
                    self.env.requires_fresh_binding
                    or self.env.has_opaque_fragment_inputs
                ):
                    self._fresh_input_layouts = self._snapshot_fresh_input_layouts(
                        runtime_args
                    )
                self.env.restrict_pid_types_for_persistent(args)
                self._bind_guard_count = len(self.env.shape_env.guards)

                self.env.config_spec.configure_epilogue_subtile_autotune(args)
                self.env.snapshot_tensor_descriptor_alignments(runtime_args)

                # Structural facts the compiler seeds and the search surface
                # read off the traced host function (recorded before
                # ``compiler_seed_configs`` runs; independent of the runtime
                # arguments).
                self.env.backend.host_structural_facts(
                    self.host_function, self.env.config_spec
                )
                with self.env.use_runtime_arg_values(runtime_args):
                    self.env.config_spec.compiler_seed_configs = compiler_seed_configs(
                        self.env,
                        self.host_function.device_ir,
                    )
                    if not self._cache_managed:
                        self.env.snapshot_runtime_input_specialization_results(
                            runtime_args
                        )
                self._compiler_seed_specialization_extractors = (
                    _compiler_seed_specialization_extractors(
                        compiler_seed_specialization_facts(
                            self.env.backend_name,
                            self.config_spec.autotuner_heuristics,
                        )
                        | self.env.compiler_fact_specialization_facts,
                        self.settings.persistent_reserved_sms,
                        static_shapes=self.settings.static_shapes,
                    )
                )
                self._compiler_seed_specialization_results = tuple(
                    extractor(args)
                    for extractor in self._compiler_seed_specialization_extractors
                )
                if any(
                    extractor.fact != "input_tensor_metadata"
                    for extractor in self._compiler_seed_specialization_extractors
                ):
                    self._compiler_seed_device_resolver = (
                        _ArgumentDeviceResolver.from_values(args)
                    )
                    self._compiler_seed_device_results[self.env.device] = tuple(
                        (None if extractor.fact == "input_tensor_metadata" else result)
                        for extractor, result in zip(
                            self._compiler_seed_specialization_extractors,
                            self._compiler_seed_specialization_results,
                            strict=True,
                        )
                    )

                if not self.env.settings.disable_autotuner_heuristics:
                    for seed_config in self.env.config_spec.autotune_seed_configs():
                        if (
                            seed_config
                            not in self.env.config_spec.compiler_seed_configs
                        ):
                            self.env.config_spec.compiler_seed_configs.append(
                                seed_config
                            )
                with self.env.use_runtime_arg_values(runtime_args):
                    register_compiler_coverage_groups(
                        self.env, self.host_function.device_ir
                    )

    def _apply_mark_static(self, args: tuple[object, ...]) -> None:
        """
        Apply torch._dynamo.mark_static() markings from input tensors.

        This reads _dynamo_static_indices from each tensor argument and marks
        the corresponding dimensions as specialized (constant) in the kernel.
        """
        for arg, fake_arg in zip(args, self.fake_args, strict=True):
            if isinstance(arg, torch.Tensor) and isinstance(fake_arg, torch.Tensor):
                for dim in getattr(arg, "_dynamo_static_indices", ()):
                    size = fake_arg.size(dim)
                    if isinstance(size, torch.SymInt):
                        self.env.specialized_vars.update(_symint_free_symbols(size))

    @property
    def env(self) -> CompileEnvironment:  # pyrefly: ignore[bad-override]
        return self._env

    @property
    def settings(self) -> Settings:
        """
        Retrieve the settings associated with the kernel.

        Returns:
            Settings: The settings of the kernel.
        """
        return self.kernel.settings

    @property
    def config_spec(self) -> ConfigSpec:
        """
        Retrieve the configuration specification for the kernel.

        Returns:
            ConfigSpec: The configuration specification.
        """
        return self.env.config_spec

    @property
    def configs(self) -> list[Config]:
        """Return the kernel's configured configs (alias for `self.kernel.configs`)."""
        return self.kernel.configs

    def _normalize_config(self, config: ConfigLike) -> Config:
        self.kernel._validate_structural_policy()
        if isinstance(config, StructuralConfig):
            require_same_structural_policy(
                config.policy, self._structural_policy, context="Late config envelope"
            )
            return config.config
        if isinstance(config, Config):
            return config
        # pyrefly: ignore [bad-argument-type]
        return Config(**config)

    def _normalized_config_copy(self, config: ConfigLike) -> Config:
        return self.env.config_spec.normalized_config(self._normalize_config(config))

    def config_envelope(self, config: ConfigLike) -> CuteStructuralConfig:
        """Snapshot a config and this bound's policy for a later pre-binding load."""
        if self._structural_policy is None:
            raise ValueError("Config policy envelopes require backend='cute'")
        return CuteStructuralConfig(
            self._normalize_config(config), self._structural_policy
        )

    def format_kernel_decorator(self, config: ConfigLike, settings: Settings) -> str:
        """Return the @helion.kernel decorator capturing backend codegen settings."""
        if self is not None and self.kernel.cute_structural_policy is not None:
            config = self.config_envelope(config)
        parts = [
            f"config={config.__repr__()}",
            f"static_shapes={settings.static_shapes}",
        ]
        if settings.index_dtype is not None:
            parts.append(f"index_dtype={settings.index_dtype}")
        # Resolved from the settings' backend: callers pass foreign Settings
        # and ``self`` may be None.
        parts.extend(get_backend(settings.backend).decorator_settings(settings))
        return f"@helion.kernel({', '.join(parts)})"

    def to_code(
        self,
        config: ConfigLike | None = None,
        *,
        options: OutputCodeOptions | None = None,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str:
        """
        Generate backend-specific code for the kernel based on the given configuration.

        Args:
            config: The configuration to use for code generation.
            options: Optional :class:`~helion.runtime.kernel.OutputCodeOptions`.
                With ``allow_helion_deps=False`` the returned module is
                self-contained (no ``helion`` import at runtime); ``jax_fn=True``
                (Pallas only) emits a pure-JAX module operating on ``jax.Array``s.
                ``None`` keeps the default behavior.
            emit_repro_caller: Emits a main function to call the kernel with example inputs.

        Generated source preserves already-evaluated literal parameter defaults,
        including complex numbers and recursively immutable tuples.
        Other defaults, such as mutable objects, functions and config fragments,
        must be passed explicitly when calling a raw generated or standalone
        entrypoint. Their original default expressions are never re-executed.
        :meth:`compile_config` retains all original evaluated defaults.

        Returns:
            str: The generated code as a string.
        """
        if config is None:
            config = self._require_implicit_config()
        with self.env, measure("BoundKernel.to_code"):
            # Work on a copy so the caller's Config is not mutated with defaults
            # specific to this BoundKernel's config_spec.
            config = self._normalized_config_copy(config)
            with (
                self._runtime_arg_values_for_codegen(),
                measure("BoundKernel.generate_ast"),
            ):
                self.env.backend.before_codegen(self, config)
                # pyrefly: ignore [bad-argument-type]
                root = generate_ast(self.host_function, config, emit_repro_caller)
                # Code generation can guard on input sizes too.
                self._publish_guarded_input_facts()
                self.env.backend.after_codegen(self, config)
            if output_origin_lines is None:
                output_origin_lines = self.settings.output_origin_lines
            import_lines: list[str] = []
            body_start = 0
            for i, stmt in enumerate(root.body):
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    if not (
                        isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
                    ):
                        import_lines.append(ast.unparse(stmt))
                    continue
                body_start = i
                break
            else:
                body_start = len(root.body)
            body_root = ast.Module(body=root.body[body_start:], type_ignores=[])
            ast.fix_missing_locations(body_root)
        # Export after codegen and outside the fake-tensor environment: backends
        # may capture launch metadata before rewriting the module. Check jax_fn
        # first because it supports both dependency modes.
        if options is not None and options.jax_fn:
            from .._compiler.output_code_utils import build_jax_fn_module
            from .._compiler.output_code_utils import capture_jax_launch_metadata

            jax_meta = capture_jax_launch_metadata(self, config)
            body_root = build_jax_fn_module(
                self, options, import_lines, body_root, jax_meta
            )
        elif options is not None and not options.allow_helion_deps:
            from .._compiler.output_code_utils import build_dependency_free_code

            body_root = build_dependency_free_code(
                self, options, import_lines, body_root
            )
        with measure("BoundKernel.unparse"):
            body = unparse(body_root, output_origin_lines=output_origin_lines)
        imports = "\n".join(import_lines)
        if imports:
            return f"from __future__ import annotations\n\n{imports}\n\n{body}"
        return f"from __future__ import annotations\n\n{body}"

    def to_triton_code(
        self,
        config: ConfigLike | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str:
        """Backward-compatible alias for :meth:`to_code`."""
        return self.to_code(
            config,
            emit_repro_caller=emit_repro_caller,
            output_origin_lines=output_origin_lines,
        )

    def has_compiled_config(self, config: ConfigLike) -> bool:
        """Whether ``compile_config(config)`` would hit this kernel's compile cache.

        A config the strict normalization rejects (a search candidate carrying
        a knob its other knobs do not admit, e.g. a grouped register profile
        on a schedule without one) is not compiled: it answers ``False`` and
        the compile path reports the rejection for that one config, instead
        of the rejection aborting the batch that asked.
        """
        try:
            normalized = self._normalized_config_copy(self._normalize_config(config))
        except exc.InvalidConfig:
            return False
        return normalized in self._compile_cache

    def prime_rendered_code(self, config: ConfigLike, code: str) -> None:
        """Hand :meth:`compile_config` the source a render worker made for ``config``.

        Honoured only while the autotuner measures candidates: the next
        ``compile_config(config)`` loads ``code`` instead of generating it,
        and remembers that the facts code generation records on this kernel
        (guard facts, CuTe specializations; see :meth:`to_code`) are missing
        for that config. The first ``compile_config`` of such a config outside
        the search renders it again so a config that runs for real always
        went through this process's code generation; that render happens when
        the selected config is set on the kernel, after the search returned
        and outside its wall-clock budget. Workers render from the parent's
        state at fork time; with ``static_shapes=False`` the guards the parent
        accumulates meanwhile only widen the dispatch key, not the source. See
        :mod:`helion.autotuner.render_pool`.
        """
        normalized = self._normalized_config_copy(self._normalize_config(config))
        self._primed_renders[normalized] = code

    def compile_config(
        self, config: ConfigLike | None = None, *, allow_print: bool = True
    ) -> CompiledConfig:
        """
        Compile the kernel for a specific configuration.

        Args:
            config: The configuration to compile the kernel with.
            allow_print: Set to suppress printing the output code when autotuning.

        Returns:
            CompiledConfig: A callable object representing the compiled kernel.
                Its omitted arguments use the original evaluated parameter
                defaults, including their object identities.
        """
        if config is None:
            config = self._require_implicit_config()
        requested_config = self._normalize_config(config)
        config = self._normalized_config_copy(requested_config)
        dist_check_config_consistancy(
            config, process_group_name=self._env.process_group_name
        )
        primed = self._primed_renders.pop(config, None)
        if (rv := self._compile_cache.get(config)) is not None:
            if config in self._render_only_configs and not _SEARCHING_CANDIDATES.get():
                # Loaded from a render worker's source for a measurement; a
                # config about to run for real is rendered here so to_code
                # records its guard facts and CuTe specializations.
                self._render_only_configs.pop(config, None)
                del self._compile_cache[config]
            else:
                self.env.backend.after_compile(self, config)
                return rv
        device_index = (
            self._env.device.index if self._env.device.index is not None else 0
        )
        self.env.backend.setup_compile_cache_dir(device_index)
        try:
            if primed is not None and _SEARCHING_CANDIDATES.get():
                triton_code = primed
                self._render_only_configs[config] = None
            else:
                triton_code = self.to_triton_code(
                    config, emit_repro_caller=self.settings.print_output_code
                )
                self._render_only_configs.pop(config, None)
            # static_shapes=True keys a distinct BoundKernel per input shape (see
            # _tensor_key), but PyCodeCache keys compiled modules by SOURCE TEXT, so
            # two shapes that emit byte-identical source (e.g. compact_worklist, whose
            # token dim enters only via runtime offsets + a data-dependent loop) would
            # share one module.  For backends whose generated module caches shape-
            # specific state (Backend.requires_shape_specialized_module -- e.g. Pallas
            # output-meta descriptors proved fixed by the input specialization),
            # that shared module returns the first shape's cached output extent for
            # both. Fold the specialization key into
            # the PyCodeCache key (via ``extra``, leaving the source untouched) so each
            # specialization gets its own module.  Reusing the same key that
            # distinguishes BoundKernels (``self._base_spec_key``, computed from the
            # real args in __init__) guarantees distinct BoundKernel => distinct module
            # and already covers shape/dtype/stride recursively through nested
            # container args.
            cache_extra = ""
            if (
                self.settings.static_shapes
                and self.env.backend.requires_shape_specialized_module
            ):
                cache_extra = repr(self._base_spec_key)
            if self.kernel.cute_structural_policy is not None:
                cache_extra += self.extra_cache_key()
            with measure("BoundKernel.PyCodeCache.load"):
                module = PyCodeCache.load(triton_code, extra=cache_extra)
            self.env.backend.annotate_compiled_module(
                module, triton_code, self.kernel.name
            )
        except Exception:
            log.warning(
                "Helion compiler triton codegen error for %s",
                self.format_kernel_decorator(requested_config, self.settings),
                exc_info=True,
            )
            self.maybe_log_repro(log.warning, self.fake_args, config=requested_config)
            raise
        if allow_print:
            log.info("Output code written to: %s", module.__file__)
            log.debug("Debug string: \n%s", LazyString(lambda: self._debug_str()))

            # for distributed kernel, print rank1 code since rank0
            # code can skip some offset computation.
            if (
                not dist.is_initialized() or dist.get_rank() == 1
            ) and self.settings.print_output_code:
                log.info("Output code: \n%s", triton_code)
                print(f"# Output code written to: {module.__file__}", file=sys.stderr)
                print(triton_code, file=sys.stderr)
        rv = _with_evaluated_defaults(
            getattr(module, self.kernel.name), self.kernel.signature
        )
        self._compile_cache[config] = rv
        self._cache_path_map[config] = module.__file__
        self.env.backend.after_compile(self, config)
        return rv

    def bench_compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., object]:
        return self.compile_config(config, allow_print=allow_print)

    def extra_cache_key(self) -> str:
        """Return extra data folded into the disk-cache key.

        Returns ``""`` by default, leaving the cache key unchanged.
        """
        policy = self.kernel.cute_structural_policy
        return "" if policy is None else f"cute_structural_policy:{policy.identity()}"

    def supports_subprocess_benchmark(self) -> bool:
        return True

    def is_cacheable(self) -> bool:
        return not self.env.requires_fresh_binding

    def get_cached_path(self, config: ConfigLike | None = None) -> str | None:
        """
        Get the file path of the generated Triton code for a specific configuration.

        Args:
            config: The configuration to get the file path for.
        Returns:
            str | None: The file path of the generated Triton code, or None if not found.
        """
        if config is None:
            config = self._require_implicit_config()
        requested_config = self._normalize_config(config)
        if requested_config in self._cache_path_map:
            return self._cache_path_map[requested_config]
        try:
            config = self._normalized_config_copy(requested_config)
        except exc.InvalidConfig:
            return None
        return self._cache_path_map.get(config, None)

    def _debug_str(self) -> str:
        """
        Generate a debug string for the kernel.

        Returns:
            str: A string containing debug information about the kernel.
        """
        if self.host_function is None:
            # In ref mode, host_function is not created
            return f"<BoundKernel {self.kernel.fn.__name__} in ref mode>"
        with self.env:
            return self.host_function.debug_str()

    def _get_host_semantic_input_normalization(
        self,
    ) -> _HostSemanticInputNormalization:
        """Return capture-time equivalence classes for dynamic input symbols."""
        if self.settings.static_shapes:
            return ()

        canonical_by_expr: dict[sympy.Expr, int] = {}
        normalization: list[tuple[int, str, int, int]] = []
        with self.env:
            tensor_args = self._host_semantic_input_tensors()
            for tensor_index, arg in enumerate(tensor_args):
                for property_name, values in (
                    ("size", arg.size()),
                    ("stride", arg.stride()),
                ):
                    for index, value in enumerate(values):
                        if not isinstance(value, torch.SymInt):
                            continue
                        expr = self.env.shape_env.simplify(value._sympy_())
                        if isinstance(expr, sympy.Integer):
                            continue
                        canonical = canonical_by_expr.get(expr)
                        if canonical is None:
                            canonical = len(canonical_by_expr)
                            canonical_by_expr[expr] = canonical
                        normalization.append(
                            (tensor_index, property_name, index, canonical)
                        )
        return tuple(normalization)

    def _host_semantic_debug_renames(
        self,
        normalization: _HostSemanticInputNormalization,
    ) -> tuple[dict[sympy.Basic, sympy.Basic], bool]:
        tensor_args = self._host_semantic_input_tensors()
        renames: dict[sympy.Basic, sympy.Basic] = {}
        class_by_expr: dict[sympy.Basic, int] = {}
        compatible = True
        for tensor_index, property_name, index, canonical in normalization:
            # A missing or concretized property is left unnormalized. If it is
            # semantically used, the rendered host/device trace still differs.
            if tensor_index >= len(tensor_args):
                continue
            tensor = tensor_args[tensor_index]
            values = tensor.size() if property_name == "size" else tensor.stride()
            if index >= len(values):
                continue
            value = values[index]
            if not isinstance(value, torch.SymInt):
                continue
            expr = self.env.shape_env.simplify(value._sympy_())
            existing = class_by_expr.get(expr)
            if existing is not None and existing != canonical:
                compatible = False
                continue
            class_by_expr[expr] = canonical
            renames[expr] = sympy.Symbol(
                f"<helion_input_symbol_{canonical}>",
                integer=True,
            )
        return renames, compatible

    def _host_semantic_input_tensors(self) -> list[torch.Tensor]:
        return [
            tensor
            for tensor, source in self.env.input_sources.items()
            if _is_supported_tensor_input_source(source)
        ]

    def _host_semantic_external_tensor_keys(self) -> tuple[object, ...]:
        return tuple(
            (
                type(source).__name__,
                tuple(self._semantic_dimension_key(dim) for dim in tensor.shape),
                tuple(self._semantic_dimension_key(dim) for dim in tensor.stride()),
                str(tensor.dtype),
                str(tensor.device),
                str(tensor.layout),
                tensor.requires_grad,
            )
            for tensor, source in self.env.input_sources.items()
            if not _is_supported_tensor_input_source(source)
        )

    def _get_host_semantic_fingerprint(
        self,
        outputs: Sequence[object],
        *,
        input_normalization: _HostSemanticInputNormalization | None = None,
    ) -> str:
        assert self.host_function is not None
        if input_normalization is None:
            input_normalization = self._get_host_semantic_input_normalization()

        output_keys: list[object] = []
        for output in outputs:
            if isinstance(output, torch.Tensor):
                if self.settings.static_shapes:
                    shape = tuple(
                        self._semantic_dimension_key(dim) for dim in output.shape
                    )
                else:
                    shape = ("rank", output.ndim)
                output_keys.append(
                    (
                        "tensor",
                        shape,
                        str(output.dtype),
                        str(output.device),
                        str(output.layout),
                    )
                )
            elif isinstance(output, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                output_keys.append((type(output).__name__,))
            elif output is None or type(output) in (bool, float, int, str):
                output_type = type(output)
                output_keys.append(
                    (
                        f"{output_type.__module__}.{output_type.__qualname__}",
                        repr(output),
                    )
                )
            else:
                raise TypeError(
                    f"Unsupported Helion host output in semantic fingerprint: "
                    f"{type(output).__name__}"
                )

        output_key = tuple(output_keys)
        cache_key = (input_normalization, output_key)
        fingerprint = self._host_semantic_fingerprints.get(cache_key)
        if fingerprint is not None:
            return fingerprint

        normalization_compatible = True
        with self.env:
            if input_normalization:
                renames, normalization_compatible = self._host_semantic_debug_renames(
                    input_normalization
                )
                with self.env.use_debug_shape_renames(renames):
                    host_source = self.host_function.semantic_debug_str()
            else:
                host_source = self.host_function.semantic_debug_str()

        payload = (
            input_normalization,
            normalization_compatible,
            self.env.backend_name,
            str(self.env.index_dtype),
            self._host_semantic_external_tensor_keys(),
            host_source,
            output_key,
        )
        fingerprint = hashlib.sha256(repr(payload).encode()).hexdigest()
        self._host_semantic_fingerprints[cache_key] = fingerprint
        return fingerprint

    def _semantic_dimension_key(self, dim: object) -> object:
        if isinstance(dim, torch.SymInt) and dim.node.shape_env is self.env.shape_env:
            return shape_env_size_hint(self.env.shape_env, dim.node.expr)
        return dim

    def _snapshot_fresh_input_layouts(
        self, runtime_args: dict[str, object]
    ) -> tuple[Hashable, ...] | None:
        # Static fake inputs can erase storage offsets. Snapshot real metadata
        # before argument references leave construction, without retaining any
        # input allocation or replaying a custom container conversion.
        assert self.host_function is not None
        layouts: list[Hashable] = []
        with self.host_function, self.env.use_runtime_arg_values(runtime_args):
            for fake in self._host_semantic_input_tensors():
                tensor = self.env.runtime_value_for_tensor(fake)
                if not isinstance(tensor, torch.Tensor):
                    return None
                layouts.append(
                    (
                        _hashable_dims(tensor.size()),
                        _hashable_dims(tensor.stride()),
                        _hashable_dim(tensor.storage_offset()),
                        tensor.dtype,
                        tensor.layout,
                    )
                )
        return tuple(layouts)

    @functools.cached_property
    def _fresh_selection_key(self) -> Hashable | None:
        """Identity of the complete frontend trace and its concrete domains."""
        if not (self.env.requires_fresh_binding or self.env.has_opaque_fragment_inputs):
            return None
        if self._fresh_input_layouts is None:
            return None
        domains = []
        for name, fragment in sorted(self.config_spec.user_defined_tunables.items()):
            key = _plain_builtin_fragment_key(fragment)
            if key is None:
                return None
            domains.append((name, type(fragment), key))
        # This is the existing capture/lowering semantic fingerprint, including
        # *all* FX branch graphs, not code rendered under a default config.
        semantic = self._get_host_semantic_fingerprint((), input_normalization=())
        with self.env:
            domain = self.config_spec.cache_fingerprint_hash(
                advanced_controls_files=self.settings.autotune_search_acf or None
            )
        return (
            semantic,
            domain,
            tuple(domains),
            self._fresh_input_layouts,
            self.env.device,
            target_device_capability(self.env.device),
            # The base key already evaluated the user key once. Preserve its
            # partition even when traces, layouts and config domains match.
            *((self._base_spec_key[-1],) if self.kernel._key_fn is not None else ()),
        )

    def _warn_fresh_selection(self) -> None:
        with self.kernel._bind_lock:
            if not self.kernel._warned_fresh_selection:
                self.kernel._warned_fresh_selection = True
                log.warning(
                    "Kernel %s selected configs cannot be retained for fresh or evicted "
                    "bindings with unproved input layouts or custom or opaque "
                    "config domains. Calls may repeat autotuning. "
                    "Explicit BoundKernel.set_config "
                    "remains available.",
                    self.kernel.name,
                )

    def _remember_fresh_selection(self, key: Hashable | None) -> None:
        from ..autotuner.base_cache import should_skip_cache

        if self._config is None or should_skip_cache():
            return
        if key is None:
            if self.env.requires_fresh_binding or self.env.has_opaque_fragment_inputs:
                self._warn_fresh_selection()
            return
        if _immutable_fragment_key(self._config.config) is None:
            self._warn_fresh_selection()
            return
        with self.kernel._bind_lock:
            if self._reset_generation != self.kernel._reset_generation:
                return
            selections = self.kernel._fresh_selections
            selections[key] = copy.deepcopy(self._config)
            selections.move_to_end(key)
            if len(selections) > _FRESH_SELECTION_CACHE_SIZE:
                selections.popitem(last=False)

    def _retained_fresh_selection(self) -> Config | None:
        from ..autotuner.base_cache import should_skip_cache

        if self.settings.force_autotune or should_skip_cache():
            return None
        key = self._fresh_selection_key
        if key is None:
            return None
        with self.kernel._bind_lock:
            if self._reset_generation != self.kernel._reset_generation:
                return None
            selections = self.kernel._fresh_selections
            config = selections.get(key)
            if config is not None:
                selections.move_to_end(key)
                return copy.deepcopy(config)
        return None

    def autotune(
        self,
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        """
        Perform autotuning to find the optimal configuration for the kernel.  This uses the
        default setting, you can call helion.autotune.* directly for more customization.

        If config= or configs= is provided to helion.kernel(), the search will be restricted to
        the provided configs.  Use force=True to ignore the provided configs.

        Mutates self so that `__call__` will run the best config found.

        Args:
            args: Example arguments used for benchmarking during autotuning.
            force: If True, force full autotuning even if a config is provided.
            kwargs: Additional keyword options forwarded to the autotuner.

        Returns:
            Config: The best configuration found during autotuning.
        """
        normalized_args = self.kernel.normalize_args(*args)
        selection_key = self._fresh_selection_key
        if self._cache_managed:
            rebound = self.kernel.bind(normalized_args)
            if rebound is not self:
                return rebound.autotune(
                    normalized_args,
                    force=force,
                    **kwargs,
                )
        if not force and (config := self._retained_fresh_selection()) is not None:
            self.set_config(config)
            return config
        ephemeral = self.env.backend.make_ephemeral_cache()
        ctx = ephemeral if ephemeral is not None else contextlib.nullcontext()
        with ctx:
            config = self.env.backend.autotune(
                self,
                normalized_args,
                force=force,
                **kwargs,
            )
        if ephemeral is not None:
            self.env.backend.finalize_ephemeral_cache(self, config)
        self.set_config(config)
        self._remember_fresh_selection(selection_key)
        return config

    def set_config(self, config: ConfigLike) -> None:
        """
        Set the configuration for the kernel and compile it.

        Mutates self so that `__call__` will run the provided config.

        Args:
            config: The configuration to set.
        """
        requested_config = config
        config = self._normalize_config(config)
        self._run = self.compile_config(config)
        self._config = config
        repro_config = (
            requested_config
            if isinstance(requested_config, StructuralConfig)
            else config
        )
        counters["best_config_decorator"][
            self.format_kernel_decorator(repro_config, self.settings)
        ] = 1

    def _specialize_extra(self) -> list[Callable[[Sequence[object]], Hashable]]:
        """
        Returns a list of functions that will be called to generate extra specialization keys.
        This is used to specialize on the values hl.specialize()'ed arguments.

        Returns:
            list[Callable[[Sequence[object]], Hashable]]: A list of functions that generate extra specialization keys.
        """
        tensor_descriptor_layout_guards = self.env.tensor_descriptor_layout_guards
        tensor_descriptor_alignment_guards = getattr(
            self.env, "tensor_descriptor_alignment_guards", {}
        )
        if (
            not self.env.specialized_vars
            and not self.env.specialized_strides
            and not tensor_descriptor_layout_guards
            and not tensor_descriptor_alignment_guards
            and not self.env.runtime_input_specializations
        ):
            return []

        def make_extractor(v: Source) -> Callable[[Sequence[object]], Hashable]:
            if isinstance(v, TensorPropertySource):
                index = v.idx
                assert index is not None
                inner = make_extractor(v.base)
                if v.prop == TensorProperty.SIZE:

                    def size_extractor(
                        args: Sequence[object],
                        _inner: Callable[[Sequence[object]], Hashable] = inner,
                        _index: int = index,
                    ) -> Hashable:
                        result = _inner(args)
                        return cast("torch.Tensor", result).size(_index)

                    return size_extractor
                if v.prop == TensorProperty.STRIDE:

                    def stride_extractor(
                        args: Sequence[object],
                        _inner: Callable[[Sequence[object]], Hashable] = inner,
                        _index: int = index,
                    ) -> Hashable:
                        result = _inner(args)
                        return cast("torch.Tensor", result).stride(_index)

                    return stride_extractor
                raise exc.SpecializeArgType(v)
            if isinstance(v, AttrSource):
                if not _is_supported_tensor_input_source(v):
                    raise exc.SpecializeArgType(v)
                inner = make_extractor(v.base)

                def attribute_extractor(
                    args: Sequence[object],
                    _inner: Callable[[Sequence[object]], Hashable] = inner,
                    _member: str = v.member,
                ) -> Hashable:
                    return getattr(_inner(args), _member)

                return attribute_extractor
            if isinstance(v, GetItemSource):
                if not isinstance(v.index, (int, str)) or v.index_is_slice:
                    raise exc.SpecializeArgType(v)
                inner = make_extractor(v.base)

                def getitem_extractor(
                    args: Sequence[object],
                    _inner: Callable[[Sequence[object]], Hashable] = inner,
                    _index: int | str = v.index,
                ) -> Hashable:
                    result: Any = _inner(args)
                    return result[_index]

                return getitem_extractor
            if isinstance(v, LocalSource):
                index = arg_name_to_index[v.local_name]

                def local_extractor(
                    args: Sequence[object],
                    _index: int = index,
                ) -> Hashable:
                    return args[_index]  # pyrefly: ignore[bad-return]

                # The argument index, for generated guards that read the
                # argument directly instead of calling the extractor.
                local_extractor._helion_arg_index = index  # pyrefly: ignore[missing-attribute]
                return local_extractor
            raise exc.SpecializeArgType(v)

        arg_name_to_index: dict[str, int] = {
            n: i for i, n in enumerate(self.kernel.signature.parameters.keys())
        }
        extractors: list[Callable[[Sequence[object]], Hashable]] = []
        extracted_strides: set[TensorPropertySource] = set()
        for v in sorted(self.env.specialized_vars, key=lambda v: v.name):
            sources = self.env.shape_env.var_to_sources.get(v)
            if not sources:
                raise exc.TypeInferenceError(
                    "hl.specialize requires a value that can be guarded from an "
                    "entry input. Specialize the required input metadata before "
                    "host mutation instead of specializing an unbacked result."
                )
            source = sources[0]
            extractor = make_extractor(source)
            if isinstance(source, TensorPropertySource):
                extractor = PreparedMetadataSpecializationExtractor(extractor)
            extractors.append(extractor)
            if (
                isinstance(source, TensorPropertySource)
                and source.prop == TensorProperty.STRIDE
                and source.idx is not None
            ):
                extracted_strides.add(source)
        for source in sorted(self.env.specialized_strides, key=repr):
            if source in extracted_strides:
                continue
            extractors.append(
                PreparedMetadataSpecializationExtractor(make_extractor(source))
            )
        candidate_configs: tuple[Config, ...] | None
        if tensor_descriptor_layout_guards or tensor_descriptor_alignment_guards:
            implicit_config = self._fixed_config_for_td_layout_guards()
            if implicit_config is not None:
                candidate_configs = (implicit_config,)
            elif not self.settings.force_autotune and len(self.kernel.configs) > 1:
                normalized_configs = []
                for config in self.kernel.configs:
                    try:
                        normalized_configs.append(self._normalized_config_copy(config))
                    except exc.InvalidConfig:
                        # Finite-search autotuning deliberately permits invalid
                        # candidates and skips them at compile time. Descriptor
                        # guard discovery must not make those failures eager.
                        continue
                candidate_configs = tuple(normalized_configs)
            else:
                candidate_configs = None
        else:
            candidate_configs = None

        # Each candidate with the memory ops its code indexes through
        # tensor descriptors.
        candidates = (
            None
            if candidate_configs is None
            else [
                (config, self._descriptor_memory_ops(config))
                for config in candidate_configs
            ]
        )

        def guard_is_active(guard: _TensorDescriptorOperationGuard) -> bool:
            return candidates is None or any(
                _td_guard_active_for_config(guard, config, memory_ops)
                for config, memory_ops in candidates
            )

        def descriptor_extent_cap(
            guard: TensorDescriptorLayoutGuard,
        ) -> int | None:
            if guard.has_derived_block_extent:
                return (
                    CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
                    if self.env.device.type == "cuda"
                    else None
                )
            if candidates is None:
                return (
                    CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
                    if self.env.device.type == "cuda"
                    else None
                )
            active_configs = tuple(
                config
                for config, memory_ops in candidates
                if _td_guard_active_for_config(guard, config, memory_ops)
            )
            with self.env:
                resolved_block_sizes = (
                    block_size.from_config(config)
                    for config in active_configs
                    for block_size in self.env.block_sizes
                )
                cap = max(
                    (
                        value
                        for value in resolved_block_sizes
                        if type(value) is int and value > 0 and value & (value - 1) == 0
                    ),
                    default=None,
                )
            if self.env.device.type != "cuda":
                return cap
            if cap is None:
                return CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
            return min(cap, CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE)

        for source, guard in sorted(
            tensor_descriptor_layout_guards.items(),
            key=lambda item: repr(item[0]),
        ):
            if not guard_is_active(guard):
                continue
            extract_tensor = make_extractor(source)

            def td_layout_extractor(
                args: Sequence[object],
                _extract_tensor: Callable[
                    [Sequence[object]], Hashable
                ] = extract_tensor,
                _ndim: int = guard.ndim,
                _element_size: int = guard.element_size,
                _extent_cap: int | None = descriptor_extent_cap(guard),
            ) -> Hashable:
                tensor = cast("torch.Tensor", _extract_tensor(args))
                if tensor.ndim != _ndim:
                    return ("ndim", tensor.ndim)
                layout = tensor_descriptor_layout_signature_from_strides(
                    tensor.stride(),
                    _element_size,
                )
                # Extent classes guard Triton's descriptor block-shape and
                # int32-coordinate legality. Other backends retain the legacy
                # layout-only key; they neither consume host descriptors nor
                # use Triton's descriptor legality checks.
                if self.env.backend_name != "triton":
                    return layout
                return (
                    layout,
                    tuple(
                        _tensor_descriptor_extent_class(int(size), _extent_cap)
                        for size in tensor.size()
                    ),
                    all(int(size) < 2**31 for size in tensor.size()),
                )

            extractors.append(
                PreparedMetadataSpecializationExtractor(td_layout_extractor)
            )

        for source, guard in sorted(
            tensor_descriptor_alignment_guards.items(),
            key=lambda item: repr(item[0]),
        ):
            if not guard_is_active(guard):
                continue
            # Prepared metadata guards do not cover base pointers.
            extractors.append(
                _TensorDescriptorAlignmentExtractor(
                    make_extractor(source), guard.requires_zero_storage_offset
                )
            )

        for key, specialization in sorted(
            self.env.runtime_input_specializations.items(),
        ):
            source_extractors = tuple(
                make_extractor(source) for source in specialization.sources
            )

            extractors.append(
                _RuntimeInputSpecializationExtractor(
                    source_extractors,
                    specialization.classifier,
                    frozenset(specialization.reusable_tensor_properties),
                    key,
                    specialization.prepared_guard_factory,
                    specialization,
                )
            )
        return extractors

    def _publish_guarded_input_facts(self) -> None:
        """Extend the dispatch schema with guard facts it does not check yet."""
        if self._cache_managed:
            self.kernel._publish_guard_facts(self, self._guarded_input_facts())

    def _guarded_input_facts(
        self,
    ) -> list[PreparedMetadataSpecializationExtractor]:
        """Input-shape guards this dynamic-shape compilation baked into its code.

        ``static_shapes=False`` keeps 0/1 sizes symbolic so one bound kernel
        can serve every size, but a compile-time comparison still guards on
        them: ``size == 1`` at a size-1 dim replaces the symbol and drops that
        dim's index, ``size != 1`` keeps a broadcast dim's index, and equal
        sizes merge two reduction dims.  Each replacement of an input
        size/stride (by a constant or other input sizes) and each ``!=``
        against a constant that code generation recorded becomes a cache-key
        fact, as do those on a reduction block symbol (an empty reduction
        guarded 1-wide), keyed by the block its extent's input sizes give.
        Nothing is added when no such guard fired, so the common case keeps
        #2353's single bound kernel and its dispatch cost.
        """
        if self.settings.static_shapes:
            return []
        shape_env = self.env.shape_env
        # Item and field projections identify individual tensor leaves. Replay
        # their shape guards just like those on direct tensor parameters.
        fake_by_name = dict(
            zip(self.kernel.signature.parameters, self.fake_args, strict=False)
        )
        sources: dict[sympy.Symbol, TensorPropertySource] = {}
        for symbol, symbol_sources in shape_env.var_to_sources.items():
            for source in symbol_sources:
                if (
                    isinstance(source, TensorPropertySource)
                    and _is_supported_tensor_input_source(source.base)
                    and isinstance(
                        _replay_tensor_input_source(source.base, fake_by_name),
                        torch.Tensor,
                    )
                ):
                    sources[symbol] = source
                    break

        def guarded_quantity(symbol: sympy.Symbol) -> tuple[sympy.Expr, bool] | None:
            """The input expression a guarded symbol stands for, and if a block."""
            if (source := sources.get(symbol)) is not None:
                # hl.specialize already keys the exact value.
                if (
                    symbol in self.env.specialized_vars
                    and symbol not in self.env.singleton_vars
                ) or source in self.env.specialized_strides:
                    return None
                return symbol, False
            assert self.host_function is not None
            origin = self.host_function.expr_to_origin.get(symbol)
            if (
                origin is not None
                and isinstance(origin.origin, BlockSizeOrigin)
                and (info := self.env.block_sizes[origin.origin.block_id]).reduction
                and isinstance(info.size, torch.SymInt)
            ):
                return cast("sympy.Expr", info.size._sympy_()), True
            return None

        relations: list[tuple[sympy.Symbol, sympy.Expr, bool]] = [
            (symbol, expr, True)
            for symbol, expr in shape_env.replacements.items()
            if expr != symbol
        ]
        relations.extend(
            (symbol, expr, True) for symbol, expr in self.env.keyed_size_equalities
        )
        relations.extend(
            (lhs, rhs, False)
            # Binding records ``!= 1`` for any size it saw >= 2 without code
            # depending on it; code generation's (block pointers keep a
            # broadcast dim's index) does.
            for guard in shape_env.guards[self._bind_guard_count :]
            if isinstance(guard.expr, sympy.Ne)
            and isinstance(lhs := guard.expr.lhs, sympy.Symbol)
            and isinstance(rhs := guard.expr.rhs, sympy.Integer)
        )
        param_names = tuple(self.kernel.signature.parameters)
        facts: dict[_GuardedInputFact, None] = {}
        for symbol, expr, equal in relations:
            if (quantity := guarded_quantity(symbol)) is None:
                continue
            lhs, block = quantity
            free_symbols = sorted(
                {
                    cast("sympy.Symbol", free)
                    for free in (*lhs.free_symbols, *expr.free_symbols)
                },
                key=str,
            )
            if lhs.free_symbols and all(free in sources for free in free_symbols):
                facts.setdefault(
                    _GuardedInputFact(
                        lhs,
                        expr,
                        tuple((free, sources[free]) for free in free_symbols),
                        block,
                        equal,
                        param_names,
                    )
                )
        return [
            PreparedMetadataSpecializationExtractor(fact)
            for fact in sorted(facts, key=repr)
        ]

    def _record_runtime_input_specialization_results(
        self,
        extractors: Sequence[Callable[[Sequence[object]], Hashable]],
        results: Sequence[Hashable],
    ) -> bool:
        """Initialize immutable reusable facts from an exact cache-key evaluation."""
        expected_keys = {
            key
            for key, specialization in self.env.runtime_input_specializations.items()
            if specialization.reusable_tensor_properties
        }
        observed = {
            extractor.specialization_key: result
            for extractor, result in zip(extractors, results, strict=True)
            if isinstance(extractor, _RuntimeInputSpecializationExtractor)
            and extractor.specialization_key is not None
            and extractor.reusable_tensor_properties
        }
        if observed.keys() != expected_keys:
            return False
        previous = self.env.bound_runtime_input_specialization_results
        if previous and previous != observed:
            return False
        if not previous:
            self.env.bound_runtime_input_specialization_results = observed
        return True

    @contextlib.contextmanager
    def _runtime_arg_values_for_codegen(self) -> Generator[None, None, None]:
        values: dict[str, object] = self.env.runtime_arg_values_by_name
        if not values:
            values = {
                name: value
                for name, ref in self._runtime_tensor_refs_by_name.items()
                if (value := ref()) is not None
            }
        with self.env.use_runtime_arg_values(values):
            yield

    def _extend_descriptor_specializations(
        self,
        seen_by_signature: dict[Hashable, set[Hashable]],
        descriptors: Sequence[Hashable],
        make_extractor: Callable[[Any], Callable[[Sequence[object]], Hashable]],
    ) -> None:
        """Key every descriptor that this signature does not key yet.

        Backends call this from their ``after_codegen`` / ``after_compile``
        hooks with a per-kernel ``seen_by_signature`` slot.
        """
        if not descriptors:
            return
        signature = self._base_spec_key
        # Serialize descriptor discovery with schema extension so concurrent
        # code generation cannot publish duplicate equivalent extractors.
        with self.kernel._bind_lock:
            if self._reset_generation != self.kernel._reset_generation:
                return
            seen = seen_by_signature.setdefault(signature, set())
            new_descriptors = [
                descriptor for descriptor in descriptors if descriptor not in seen
            ]
            with self._runtime_arg_values_for_codegen():
                runtime_args = tuple(
                    self.env.runtime_arg_values_by_name.get(name)
                    for name in self.kernel.signature.parameters
                )
            if self.kernel._extend_bound_kernel_specializations(
                self,
                signature,
                [make_extractor(descriptor) for descriptor in new_descriptors],
                runtime_args,
            ):
                seen.update(new_descriptors)

    def _descriptor_memory_ops(self, config: Config) -> frozenset[int]:
        """The facts' ``indexing_index`` (``memory_op_facts``) of the loads
        and stores the code for ``config`` indexes through tensor
        descriptors.  ``indexing`` follows that code's slots, which a config
        that rolls a reduction orders differently from the facts
        (``DeviceIR.config_memory_op_fact_pairs``)."""
        indexing = config.indexing
        if not indexing_uses_tensor_descriptor(indexing):
            return frozenset()
        every_op = frozenset(
            fact.indexing_index for fact in self.config_spec.memory_op_facts
        )
        if indexing == "tensor_descriptor":
            # Every load and store, in any order.
            return every_op
        assert self.host_function is not None
        device_ir = self.host_function.device_ir
        try:
            with self.env:
                graphs = device_ir.config_codegen_graphs(config)
        except Exception:
            # Its code cannot be built, so its compile fails too, which
            # autotuning skips; guard discovery must not make that failure
            # eager.  Count every load and store as a descriptor's.
            log.debug(
                "descriptor guards: the codegen graphs of %s do not build",
                config,
                exc_info=True,
            )
            return every_op
        from .._compiler.device_ir import codegen_memory_op_fact_pairs

        with self.env:
            pairs = codegen_memory_op_fact_pairs(graphs, device_ir.root_ids)
        return frozenset(
            fact.indexing_index
            for fact, config_fact in pairs
            if _indexing_config_uses_tensor_descriptor(
                indexing, config_fact.indexing_index
            )
        )

    def _fixed_config_for_td_layout_guards(self) -> Config | None:
        """Return the fixed config if TD layout guards can be filtered safely."""
        if self._config is not None:
            config = self._config
        elif self.kernel.settings.autotune_effort == "none" and (
            len(self.kernel.configs) == 0 or self.settings.force_autotune
        ):
            config = self.config_spec.default_config()
        elif self.settings.force_autotune:
            return None
        elif len(self.kernel.configs) == 1:
            config = self.kernel.configs[0]
        else:
            return None

        # Decorator configs are intentionally allowed to omit inferred fields
        # such as block_sizes.  Resolve the same effective config that codegen
        # will see before asking BlockSizeSource to read those fields.
        return self._normalized_config_copy(config)

    def _user_provided_config(self) -> Config | None:
        """Return a config if the user explicitly provided one, else None.

        Checks the kernel's config list and settings to determine if
        a config can be resolved without autotuning.
        """
        configs = self.kernel.configs
        if self.kernel.settings.autotune_effort == "none" and (
            len(configs) == 0 or self.settings.force_autotune
        ):
            config = self.config_spec.default_config()
            if not is_ref_mode_enabled(self.kernel.settings):
                kernel_decorator = self.format_kernel_decorator(config, self.settings)
                print(
                    f"Using implicit config: {kernel_decorator}",
                    file=sys.stderr,
                )
            return config
        if self.settings.force_autotune:
            return None
        if len(configs) == 1:
            return configs[0]
        return None

    def _implicit_config(self) -> Config | None:
        """
        Returns a single config that is implicitly used by this kernel, if any.
        """
        if self._config is not None:
            return self._config
        return self._user_provided_config()

    def _require_implicit_config(self) -> Config:
        """
        Returns the implicit config for this kernel, or raises an error if no implicit config is available.
        """
        if (config := self._implicit_config()) is None:
            raise RuntimeError("no config provided and no implicit config available")
        return config

    def ensure_config_exists(self, args: Sequence[object]) -> None:
        """
        Ensure a config is available, triggering autotuning if needed.

        Retained autotune selections precede implicit configs (from a configs
        list or default). Otherwise, autotune with the provided args.
        """
        if self._config is not None:
            return  # Already have a config
        if (config := self._retained_fresh_selection()) is not None:
            self.set_config(config)
        elif (config := self._implicit_config()) is not None:
            with measure("BoundKernel.set_config"):
                self.set_config(config)
        else:
            with measure("BoundKernel.autotune"):
                self.autotune(args, force=False)

    # pyrefly: ignore [bad-return]
    def run_ref(self, *args: object) -> _R:
        # Unwrap ConstExpr arguments
        clean_args = []
        for arg in args:
            if isinstance(arg, ConstExpr):
                clean_args.append(arg.value)
            else:
                clean_args.append(arg)

        # Pass the config to RefModeContext
        with RefModeContext(self.env, self._config):
            result = self.kernel.fn(*clean_args)
            return cast("_R", result)

    def __call__(self, *args: object) -> _R:
        """
        Execute the kernel with the given arguments.

        Args:
            args: The arguments to pass to the kernel.

        Returns:
            _R: The result of the kernel execution.
        """
        if self.env.requires_fresh_binding:
            rebound = self.kernel.bind(args)
            if not rebound.env.requires_fresh_binding:
                # This specialization can be shared with ordinary kernel calls.
                # Its own selected config must survive a call through a fresh bound.
                if rebound._config is None and self._config is not None:
                    rebound.set_config(self._config)
                return rebound(*args)
            if self._config is not None:
                # Preserve explicit set_config/autotune choices, but compile
                # them against the new trace rather than copying its launcher.
                rebound.set_config(self._config)
            return rebound._call_with_bound_args(*args)
        return self._call_with_bound_args(*args)

    def _call_with_bound_args(self, *args: object) -> _R:
        """Execute an invocation whose metadata was just bound or guarded."""
        if (
            self._cache_managed
            and self._reset_generation != self.kernel._reset_generation
        ):
            return self.kernel.bind(args)(*args)
        is_compiling = torch.compiler.is_compiling()
        if (
            not is_compiling
            and self._cache_managed
            and (prepared := self._direct_prepared_call) is not None
            and prepared.bound is self
            and prepared.matches(self.kernel, args)
            and self._run is not None
        ):
            return self._run(*args)

        if self._cache_managed and self._compiler_seed_specialization_extractors:
            device_results: tuple[Hashable | None, ...] | None = None
            new_compiler_seed_device = False
            if self._compiler_seed_device_resolver is not None:
                device = self._compiler_seed_device_resolver(args)
                device_results = self._compiler_seed_device_results.get(device)
                if device_results is None:
                    new_compiler_seed_device = True
                    device_results = tuple(
                        (
                            None
                            if extractor.fact == "input_tensor_metadata"
                            else extractor.for_device(device)
                        )
                        for extractor in self._compiler_seed_specialization_extractors
                    )
                    self._compiler_seed_device_results[device] = device_results
            compiler_seed_results = tuple(
                (
                    extractor(args)
                    if extractor.fact == "input_tensor_metadata"
                    else cast("tuple[Hashable | None, ...]", device_results)[index]
                )
                for index, extractor in enumerate(
                    self._compiler_seed_specialization_extractors
                )
            )
            if (
                new_compiler_seed_device
                or compiler_seed_results != self._compiler_seed_specialization_results
            ):
                rebound = self.kernel.bind(args)
                if rebound is not self:
                    return rebound(*args)
        if self._cache_managed and (
            self.kernel._has_specialization_extras or self._has_input_fragment_aliases
        ):
            rebound = self.kernel.bind(args)
            if rebound is not self:
                return rebound(*args)

        if self._run is None:
            with self._first_compile_lock:
                # Another caller may have discovered a late specialization while
                # this caller waited for the first compile.
                if self._cache_managed and (
                    self.kernel._has_specialization_extras
                    or self._has_input_fragment_aliases
                ):
                    rebound = self.kernel.bind(args)
                    if rebound is not self:
                        return rebound(*args)
                if self._run is None:
                    if is_ref_mode_enabled(self.kernel.settings):
                        if (config := self._implicit_config()) is not None:
                            self._config = config
                        return self.run_ref(*args)
                    runtime_args: dict[str, object] = {
                        name: value
                        for name, value in zip(
                            self.kernel.signature.parameters, args, strict=False
                        )
                        if isinstance(value, torch.Tensor)
                    }
                    with self.env.use_runtime_arg_values(runtime_args):
                        self.ensure_config_exists(args)
                    assert self._run is not None
                    self.maybe_log_repro(log.warning, args)

        result = self._run(*args)
        if not is_compiling and self._cache_managed:
            self._prepare_direct_call(args)
        return result

    def _prepare_direct_call(self, args: tuple[object, ...]) -> None:
        """Publish a monomorphic fast path for repeated BoundKernel calls."""
        run = self._run
        if (
            run is None
            or self.kernel._key_fn is not None
            or not self.env.backend.supports_eager_prepared_call
            or not self.kernel._has_specialization_extras
        ):
            return
        try:
            fast_entry = self.kernel._fast_dispatch_key_and_guards(args)
            if fast_entry is None:
                return
            with self.kernel._bind_lock:
                if (
                    self._reset_generation != self.kernel._reset_generation
                    or self.kernel._bind(args) is not self
                    or self._run is not run
                ):
                    return
                entry = self.kernel._prepare_dispatch_entry(args, self, fast_entry)
                if entry is not None and entry[0] is not None:
                    self._direct_prepared_call = entry[0]
        except Exception:
            # Preparation runs after the real kernel call.  It is optional and
            # must not turn a successful launch into a user-visible failure.
            return

    def _matching_compiled_configs(
        self, config: ConfigLike
    ) -> list[tuple[Config, CompiledConfig]]:
        """Find cached wrappers for a winner, allowing stripped defaults.

        Multiple equivalent wrappers can exist, and a rendered wrapper need not
        have launched yet. Preserve all matches so it cannot hide another
        wrapper's compiled binary. Return a snapshot for callers that evict.
        Compilation itself keeps its exact normalized key.
        """
        requested = self._normalize_config(config)
        normalized = self._normalized_config_copy(requested)
        defaults = self.config_spec.default_config().config

        def expanded(candidate: Config) -> Config:
            return self._normalized_config_copy(
                Config.from_dict(defaults | candidate.config)
            )

        matches = ConfigCache[CompiledConfig]()
        if (compiled_fn := self._compile_cache.get(normalized)) is not None:
            matches[normalized] = compiled_fn
        # Restore omitted search defaults before normalization inserts generic
        # defaults (e.g. num_warps=4, while the search default may be 1).
        try:
            full_config = expanded(requested)
        except exc.InvalidConfig:
            # Defaults may describe another schedule than a valid explicit AOT
            # config. Expansion is optional; keep any direct match.
            return list(matches.items())
        if (compiled_fn := self._compile_cache.get(full_config)) is not None:
            matches[full_config] = compiled_fn
        # A caller can compile a config with only some defaults explicit. Neither
        # the minimized nor fully expanded form is then its stored cache key.
        full_key = _config_key(full_config)
        for cached_config, compiled_fn in self._compile_cache.items():
            if cached_config in matches:
                continue
            try:
                cached_key = _config_key(expanded(cached_config))
            except exc.InvalidConfig:
                continue
            if cached_key == full_key:
                matches[cached_config] = compiled_fn
        return list(matches.items())

    def backend_cache_key(self, config: ConfigLike | None = None) -> str | None:
        """
        Return the backend cache key for the compiled kernel.

        For the Triton backend, this is the base32 encoding of the SHA-256
        hash that Triton uses to cache compiled GPU binaries under
        ``TRITON_CACHE_DIR/<key>/``.  For the CuTe backend, it is the base32
        encoding of the SHA-256 hash of the compiled IR module, which names the
        ``CUTE_DSL_CACHE_DIR/<key>.mlir`` artifact.

        Args:
            config: The configuration to look up. Defaults to the implicit config.

        Returns:
            str | None: The cache key, or None if the kernel hasn't been
            JIT-compiled yet or the backend doesn't support cache keys.
        """
        if config is None:
            config = self._require_implicit_config()
        for _, compiled_fn in self._matching_compiled_configs(config):
            if (
                key := self.env.backend.compiled_cache_key(self, compiled_fn)
            ) is not None:
                return key
        return None

    def maybe_log_repro(
        self,
        log_func: Callable[[str], None],
        args: Sequence[object],
        config: Config | None = None,
    ) -> None:
        if not self.settings.print_repro:
            return

        effective_config = config or self._config
        assert effective_config is not None

        # Get kernel source
        try:
            raw_source = inspect.getsource(self.kernel.fn)
            source_lines = textwrap.dedent(raw_source).splitlines()
            # Skip decorator lines (including multi-line decorators)
            start_idx = 0
            while start_idx < len(source_lines) and not source_lines[
                start_idx
            ].lstrip().startswith("def "):
                start_idx += 1
            kernel_body = "\n".join(source_lines[start_idx:])
        except (OSError, TypeError):
            kernel_body = f"# Source unavailable for {self.kernel.fn.__module__}.{self.kernel.fn.__qualname__}"

        # Format decorator
        decorator = self.format_kernel_decorator(effective_config, self.settings)

        # Build output
        output_lines = [
            "# === HELION KERNEL REPRO ===",
            "import helion",
            "import helion.language as hl",
            "import torch",
            "from torch._dynamo.testing import rand_strided",
            "",
            decorator,
            kernel_body,
        ]

        # Generate caller function
        if args:

            def _render_input_arg_assignment(name: str, value: object) -> list[str]:
                if isinstance(value, torch.Tensor):
                    shape = tuple(int(d) for d in value.shape)
                    stride = tuple(int(s) for s in value.stride())
                    device = str(value.device)
                    dtype = str(value.dtype)

                    lines = [
                        f"{name} = rand_strided({shape!r}, {stride!r}, dtype={dtype}, device={device!r})"
                    ]

                    if value.requires_grad:
                        lines.append(f"{name}.requires_grad_(True)")
                    return lines

                return [f"{name} = {value!r}"]

            sig_param_names = list(self.kernel.signature.parameters.keys())
            assert len(args) == len(sig_param_names)

            output_lines.extend(["", "def helion_repro_caller():"])
            output_lines.append("    torch.manual_seed(0)")
            arg_names: list[str] = []

            for i, value in enumerate(args):
                var_name = sig_param_names[i]
                arg_names.append(var_name)

                # Add assignment lines with indentation
                for line in _render_input_arg_assignment(var_name, value):
                    output_lines.append(f"    {line}")

            # Add return statement
            call_args = ", ".join(arg_names)
            output_lines.append(f"    return {self.kernel.name}({call_args})")
            output_lines.extend(["", "helion_repro_caller()"])

        output_lines.append("# === END HELION KERNEL REPRO ===")
        repro_text = "\n" + "\n".join(output_lines)
        log_func(repro_text)


class _KernelDecorator(Protocol):
    def __call__(
        self,
        fn: Callable[..., _R],
    ) -> Kernel[_R]: ...


@overload
def kernel(
    fn: Callable[..., _R],
    *,
    config: ConfigLike | None = None,
    configs: Sequence[ConfigLike] | None = None,
    key: Callable[..., Hashable] | None = None,
    cute_structural_policy: CuteStructuralPolicyRequest = DEFAULT_STRUCTURAL_POLICY,
    **settings: object,
) -> Kernel[_R]: ...


@overload
def kernel(
    fn: None = None,
    *,
    config: ConfigLike | None = None,
    configs: Sequence[ConfigLike] | None = None,
    key: Callable[..., Hashable] | None = None,
    cute_structural_policy: CuteStructuralPolicyRequest = DEFAULT_STRUCTURAL_POLICY,
    **settings: object,
) -> _KernelDecorator: ...


def kernel(
    fn: Callable[..., _R] | None = None,
    *,
    config: ConfigLike | None = None,
    configs: Sequence[ConfigLike] | None = None,
    key: Callable[..., Hashable] | None = None,
    cute_structural_policy: CuteStructuralPolicyRequest = DEFAULT_STRUCTURAL_POLICY,
    **settings: object,
) -> Kernel[_R] | _KernelDecorator:
    """
    Decorator to create a Kernel object from a Python function.

    Args:
        fn: The function to be wrapped by the Kernel. If None, a decorator is returned.
        config: A single configuration to use for the kernel. Refer to the
            ``helion.Config`` class for details.
        configs: A list of configurations to use for the kernel. Can only specify
            one of config or configs. Refer to the ``helion.Config`` class for
            details.
        key: Optional callable returning a hashable that augments the specialization key.
        cute_structural_policy: Select the CuTe config schema before binding.
            Ordinary CuTe searches without recorded configs automatically enable
            proved structural passes while honoring explicit settings. None
            preserves legacy configs and cache identity; "auto" also permits
            selection for a no-search invocation. A recorded policy selects its
            exact schema. Existing unversioned declarations retain their axes.
        settings: Keyword arguments representing settings for the Kernel.
            Can also use settings=Settings(...) to pass a Settings object
            directly. Refer to the ``helion.Settings`` class for available
            options.

    Returns:
        object: A Kernel object or a decorator that returns a Kernel object.

    See Also:
        - :class:`~helion.Settings`: Controls compilation behavior and debugging options
        - :class:`~helion.Config`: Controls GPU execution parameters and optimization strategies
    """
    if config is not None:
        assert not configs, "Cannot specify both config and configs"
        configs = [config]
    elif configs is None:
        configs = []

    if settings_obj := settings.get("settings"):
        assert len(settings) == 1, "settings must be the only keyword argument"
        assert isinstance(settings_obj, Settings), "settings must be a Settings object"
    else:
        settings_obj = Settings(**settings)

    if fn is None:
        return functools.partial(
            kernel,
            configs=configs,
            settings=settings_obj,
            key=key,
            cute_structural_policy=cute_structural_policy,
        )
    return Kernel(
        fn,
        configs=configs,
        settings=settings_obj,
        key=key,
        cute_structural_policy=cute_structural_policy,
    )


def _hashable_dim(s: int | torch.SymInt) -> Hashable:
    if isinstance(s, torch.SymInt):
        return (id(s.node.shape_env), s.node.expr)
    return s


def _safe_bucket_dim(s: int | torch.SymInt) -> Hashable:
    if isinstance(s, torch.SymInt):
        return (id(s.node.shape_env), s.node.expr)
    # Dynamic-shape kernels should not get separate bound kernels for sizes
    # 0 or 1.  Keep 2 as the canonical "dynamic dimension" bucket that was
    # already used for all concrete sizes >= 2.
    return 2


def _tensor_descriptor_extent_class(size: int, cap: int | None) -> int:
    """Largest power-of-two descriptor block that can fit this dimension."""
    result = 0 if size <= 0 else 1 << (size.bit_length() - 1)
    return result if cap is None else min(result, cap)


_EMPTY_FROZENSET: frozenset[int] = frozenset()


def _bucketed_size(obj: torch.Tensor) -> tuple[Hashable, ...]:
    sz = obj.size()
    n = len(sz)
    if n == 1:
        return (_safe_bucket_dim(sz[0]),)
    if n == 2:
        return (_safe_bucket_dim(sz[0]), _safe_bucket_dim(sz[1]))
    if n == 3:
        return (
            _safe_bucket_dim(sz[0]),
            _safe_bucket_dim(sz[1]),
            _safe_bucket_dim(sz[2]),
        )
    return tuple(_safe_bucket_dim(s) for s in sz)


def _hashable_dims(dims: Sequence[int | torch.SymInt]) -> tuple[Hashable, ...]:
    n = len(dims)
    if n == 1:
        return (_hashable_dim(dims[0]),)
    if n == 2:
        return (_hashable_dim(dims[0]), _hashable_dim(dims[1]))
    if n == 3:
        return (_hashable_dim(dims[0]), _hashable_dim(dims[1]), _hashable_dim(dims[2]))
    return tuple(_hashable_dim(s) for s in dims)


def _concrete_tensor_key(fn: Kernel, obj: torch.Tensor) -> Hashable:
    # Fast extractor for plain ``torch.Tensor`` / ``torch.nn.Parameter``:
    # exact-type dispatch guarantees concrete int sizes/strides, so
    # ``torch.Size`` and the stride tuple can be used directly (both are
    # tuple subclasses that hash/compare identically to plain int tuples).
    # The ``_hashable_dims`` wrap in ``_tensor_key`` exists only to
    # normalize SymInts, which appear on FakeTensors during tracing.
    si = getattr(obj, "_dynamo_static_indices", None)
    static_indices = frozenset(si) if si is not None else _EMPTY_FROZENSET
    if fn.settings.static_shapes:
        return (obj.dtype, obj.size(), obj.stride(), static_indices)
    bucketed = _bucketed_size(obj)
    if fn.settings.index_dtype is None:
        try:
            needs_int64 = bool(obj.numel() > _INT32_INDEX_LIMIT)
        except RuntimeError:
            needs_int64 = True  # unbacked SymInt
        return (
            obj.dtype,
            bucketed,
            needs_int64,
            static_indices,
        )
    return (
        obj.dtype,
        bucketed,
        static_indices,
    )


def _tensor_key(fn: Kernel, obj: torch.Tensor) -> Hashable:
    si = getattr(obj, "_dynamo_static_indices", None)
    static_indices = frozenset(si) if si is not None else _EMPTY_FROZENSET
    if fn.settings.static_shapes:
        return (
            obj.dtype,
            _hashable_dims(obj.size()),
            _hashable_dims(obj.stride()),
            static_indices,
        )
    bucketed = _bucketed_size(obj)
    if fn.settings.index_dtype is None:
        try:
            needs_int64 = bool(obj.numel() > _INT32_INDEX_LIMIT)
        except RuntimeError:
            needs_int64 = True  # unbacked SymInt
        return (
            obj.dtype,
            bucketed,
            needs_int64,
            static_indices,
        )
    return (
        obj.dtype,
        bucketed,
        static_indices,
    )


def _sequence_key(fn: Kernel, obj: Sequence) -> Hashable:
    return type(obj), tuple([fn._specialization_key(item) for item in obj])


def _mapping_key(
    fn: Kernel, obj: dict[str | int, object], real_type: type[object]
) -> Hashable:
    return real_type, tuple(
        sorted((k, fn._specialization_key(v)) for k, v in obj.items())
    )


def _number_key(fn: Kernel, n: float | bool) -> object:
    return type(n)


def _function_key(
    fn: Kernel, obj: types.FunctionType, *, baked: bool = False
) -> object:
    """Specialization key of a callable argument: its code object, the key of
    every closure cell and the key of every default.

    A cell of the argument itself is a runtime input of the compiled kernel
    (``lift_closures`` lifts it, so it is read on each call) and keys like an
    argument of its type. Defaults are baked into the kernel when the lifted
    function is traced, so they key like constexprs through ``_constant_key``,
    and so does everything reachable from a default: with ``baked`` the
    function is itself a default (``lift_closures`` passes ``__defaults__``
    through raw, so a default function is traced with its own cells, which
    are never lifted) and its cells key by value too. Module globals the
    function reads are not part of the key: like the kernel's own globals
    they are read from the module when the kernel runs and specialized once,
    on the first call's values."""
    cells = obj.__closure__ or ()
    defaults = obj.__defaults__ or ()
    kwdefaults = obj.__kwdefaults__ or {}
    if not cells and not defaults and not kwdefaults:
        return obj.__code__
    closure_keys = []
    for cell in cells:
        filled, contents = _read_cell(cell)
        closure_keys.append(
            (_constant_key(fn, contents) if baked else fn._specialization_key(contents))
            if filled
            else (types.CellType,)
        )
    closures = tuple(closure_keys)
    return (
        obj.__code__,
        closures,
        tuple(_constant_key(fn, value) for value in defaults),
        tuple(
            sorted(
                (name, _constant_key(fn, value)) for name, value in kwdefaults.items()
            )
        ),
    )


@dataclasses.dataclass(frozen=True)
class _BoolValueKey:
    # Python equality merges bool with int; their compiled tensor arithmetic
    # can have different dtypes, including inside immutable containers.
    value: bool


@dataclasses.dataclass(frozen=True)
class _FloatValueKey:
    # A tagged binary64 value cannot collide with an ordinary constexpr tuple
    # or bytes argument, and retains signed zero and every NaN payload.
    bits: bytes


@dataclasses.dataclass(frozen=True)
class _FragmentValueKey:
    fragment_type: type[ConfigSpecFragment]
    fields: Hashable


@dataclasses.dataclass(frozen=True)
class _FragmentSetValueKey:
    # Distinct fragment objects with equal domains must retain their count:
    # replacing them directly in a frozenset would change its cardinality.
    entries: frozenset[tuple[Hashable, int]]

    def __repr__(self) -> str:
        return f"_FragmentSetValueKey({', '.join(sorted(map(repr, self.entries)))})"


_TRANSFORMED_CONSTEXPR_TYPES = frozenset(
    (*_BUILTIN_TUNABLE_FRAGMENTS, bool, float, tuple, frozenset)
)


def _constexpr_key(value: object) -> Hashable:
    if type(value) is bool:
        return _BoolValueKey(value)
    if type(value) is float:
        return _FloatValueKey(struct.pack("!d", value))
    # Built-in scalar declarations need value keys, including identity-based
    # Boolean/NumThreads fragments and unhashable integer-family dataclasses.
    # Snapshot their concrete fields into a stable value key: it is also part
    # of persistent autotune serialization, so object identity is unsuitable.
    # Plain immutable containers apply this to nested declarations too.
    # Other constexpr values, including hashable custom fragments and
    # built-ins with extra state and container subclasses, retain their
    # existing equality policy.
    # Exact immutable containers canonicalize nested declarations, booleans and
    # all floats. Other members and subclasses retain their equality policy.
    # Return the original object when no member needed a value key. Fast
    # dispatch uses that identity to decide whether aliases need a walk.
    if type(value) in _BUILTIN_TUNABLE_FRAGMENTS:
        fragment = cast("ConfigSpecFragment", value)
        key = _plain_builtin_fragment_key(fragment)
        if key is not None:
            return _FragmentValueKey(type(fragment), key)
    if type(value) is tuple or type(value) is frozenset:
        keys: list[Hashable] = []
        changed = False
        for item in cast("tuple[object, ...] | frozenset[object]", value):
            key = (
                _constexpr_key(item)
                if type(item) in _TRANSFORMED_CONSTEXPR_TYPES
                else cast("Hashable", item)
            )
            keys.append(key)
            changed |= key is not item
        if changed:
            if type(value) is tuple:
                return tuple(keys)
            counts: dict[Hashable, int] = {}
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
            return _FragmentSetValueKey(frozenset(counts.items()))
    return cast("Hashable", value)


class _IdentityKey:
    """Hashable stand-in for a baked-in value that cannot key by its own
    equality (a tensor or a container holding one, whose ``==`` is not a
    bool; an unhashable object): equal only to a key of the same object. It
    keeps the object alive, so its identity cannot pass to another object
    while a cache holds the key."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return type(other) is _IdentityKey and other.value is self.value

    def __repr__(self) -> str:
        return f"_IdentityKey({self.value!r})"


def _constant_key(fn: Kernel, value: object) -> Hashable:
    """Key of a value baked into the compiled kernel: by value through the
    constexpr rules (numbers, strings, dtypes, containers of them), a
    function through ``_function_key`` with its cells baked too (a lambda
    default rebuilt per call keys like one passed as an argument), any other
    hashable value with no tensor inside by its own equality (a frozenset,
    a namedtuple, an enum member, a frozen dataclass, a bound method), else
    by identity (a tensor or a container holding one, an unhashable
    object).

    Built-in float defaults retain their binary value, including signed zero,
    through nested built-in containers. For other value-keyed types equality
    is Python's: a frozenset, namedtuple or frozen dataclass
    holding ``1``, ``1.0`` or ``True`` (or ``0.0`` and ``-0.0``) shares one
    specialization although the baked constant differs (``1.0`` casts an
    int32 input through float32), the same merge a constexpr argument
    holding that container makes. Only a default that is itself a function
    is read through ``_function_key``: a function inside a container
    default keys by identity with the container, so its cells are not
    re-read (a ``nonlocal`` rebind reuses the old kernel) and a lambda
    rebuilt per call inside a tuple is a new specialization each call."""
    if type(value) is types.FunctionType:
        return cast("Hashable", _function_key(fn, value, baked=True))
    nested = _nested_fast_dispatch_key(value, specialize_numeric=True)
    if nested is not None:
        return nested[0]
    try:
        hash(value)
    except TypeError:
        return _IdentityKey(value)
    if _contains_tensor(value, set()):
        return _IdentityKey(value)
    return (type(value), cast("Hashable", value))


def _contains_tensor(value: object, visited: set[int]) -> bool:
    """Whether a ``torch.Tensor`` is reachable from ``value`` through tuples,
    lists, sets, frozensets, dict keys and values, dataclass fields or a
    bound method's ``__self__``. ``visited`` holds the ids of the objects
    already walked (every object reachable from ``value`` stays alive for
    the walk), so a cycle such as a dataclass holding itself ends."""
    if isinstance(value, torch.Tensor):
        return True
    if id(value) in visited:
        return False
    visited.add(id(value))
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_tensor(item, visited) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_tensor(key, visited) or _contains_tensor(item, visited)
            for key, item in value.items()
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_tensor(getattr(value, field.name), visited)
            for field in dataclasses.fields(value)
        )
    if isinstance(value, types.MethodType):
        return _contains_tensor(value.__self__, visited)
    return False


def tensor_property_value(
    source: TensorPropertySource, root_values: dict[str, object]
) -> int | None:
    tensor = _replay_tensor_input_source(source.base, root_values)
    if not isinstance(tensor, torch.Tensor):
        return None
    if source.prop == TensorProperty.STORAGE_OFFSET:
        return int(tensor.storage_offset())
    assert source.idx is not None
    if source.prop == TensorProperty.SIZE:
        return int(tensor.size(source.idx))
    return int(tensor.stride(source.idx))


def evaluate_sympy_expr(expr: sympy.Expr, values: dict[sympy.Symbol, int]) -> int:
    if isinstance(expr, sympy.Symbol):
        return values[expr]
    if isinstance(expr, sympy.Integer):
        return int(expr)
    return int(expr.xreplace(values))


@dataclasses.dataclass(frozen=True)
class _GuardedInputFact:
    """Whether a guard ``lhs == rhs`` (or ``!=`` without ``equal``) still holds.

    Both sides are expressions of input sizes/strides; ``sources`` names the
    input behind each symbol.  With ``block`` the guarded symbol was a
    reduction block, whose value is ``next_power_of_2`` of the extent ``lhs``.
    """

    lhs: sympy.Expr
    rhs: sympy.Expr
    sources: tuple[tuple[sympy.Symbol, TensorPropertySource], ...]
    block: bool
    equal: bool
    param_names: tuple[str, ...]

    def _holds(self, values: dict[sympy.Symbol, int]) -> bool:
        lhs = evaluate_sympy_expr(self.lhs, values)
        same = (next_power_of_2(lhs) if self.block else lhs) == evaluate_sympy_expr(
            self.rhs, values
        )
        # Branch rather than compare: traced by dynamo the sizes are SymInts,
        # and their SymBool compared with a bool nests an Eq the shape env
        # cannot evaluate, while a branch guards on it.
        return self.equal if same else not self.equal

    def __call__(self, args: Sequence[object]) -> Hashable:
        root_values = dict(zip(self.param_names, args, strict=False))
        values: dict[sympy.Symbol, int] = {}
        for symbol, source in self.sources:
            if (value := tensor_property_value(source, root_values)) is None:
                return None
            values[symbol] = value
        return self._holds(values)

    def bound_value(self, env: CompileEnvironment) -> Hashable:
        """This fact at ``env``'s bind-time hints (None when not recoverable)."""
        symbols = {
            source: symbol
            for symbol, sources in env.shape_env.var_to_sources.items()
            for source in sources
        }
        hints = shape_env_var_hints(env.shape_env)
        values: dict[sympy.Symbol, int] = {}
        for symbol, source in self.sources:
            if (
                bound_symbol := symbols.get(source)
            ) is None or bound_symbol not in hints:
                return None
            values[symbol] = int(hints[bound_symbol])
        return self._holds(values)


def _bound_guarded_input_facts(
    bound: BoundKernel,
    extractors: Sequence[Callable[[Sequence[object]], Hashable]],
) -> tuple[Hashable, ...] | None:
    """Guard-fact results for a cached bound, or None if any is not a guard fact."""
    results: list[Hashable] = []
    for extractor in extractors:
        if not (
            isinstance(extractor, PreparedMetadataSpecializationExtractor)
            and isinstance(fact := extractor.extractor, _GuardedInputFact)
        ):
            return None
        if (result := fact.bound_value(bound.env)) is None:
            return None
        results.append(result)
    return tuple(results)


def _graph_module_key(fn: Kernel, obj: torch.fx.GraphModule) -> Hashable:
    """Generate a specialization key for GraphModule arguments."""
    # Check if already cached
    if obj in _graph_module_hash_cache:
        return _graph_module_hash_cache[obj]

    # Check for unsupported operations
    unsupported_ops = {
        node.op
        for node in itertools.chain(
            obj.graph.find_nodes(op="call_module"),
            obj.graph.find_nodes(op="get_attr"),
        )
    }
    if unsupported_ops:
        raise exc.GraphModuleUnsupportedOps(", ".join(sorted(unsupported_ops)))

    _graph_module_hash_cache[obj] = rv = str(compiled_fx_graph_hash(obj, [], {}, []))
    return rv


_specialization_extractors: dict[
    type[object] | str,
    Callable[[Kernel, object], Hashable],
    # pyrefly: ignore [bad-assignment]
] = {
    # Exact-type dispatch (see ``_specialization_key``): plain tensors and
    # Parameters always have concrete int sizes/strides and take the fast
    # extractor. Subclasses (FakeTensor below, or anything hitting the
    # ``isinstance`` fallback) go through SymInt-safe ``_tensor_key``.
    torch.Tensor: _concrete_tensor_key,
    torch.nn.Parameter: _concrete_tensor_key,
    FakeTensor: _tensor_key,
    # SymInt-safe extractor for torch.Tensor subclasses reached via the
    # isinstance fallback in ``_specialization_key`` (string key so the
    # fallback stays loosely typed, like "namedtuple" / "dataclass").
    "tensor_subclass": _tensor_key,
    torch.dtype: lambda fn, x: x,
    torch.device: lambda fn, x: x,
    int: _number_key,
    float: _number_key,
    bool: _number_key,
    str: lambda fn, x: x,
    list: _sequence_key,
    tuple: _sequence_key,
    # pyrefly: ignore [bad-argument-type]
    dict: lambda fn, x: _mapping_key(fn, x, type(x)),
    # pyrefly: ignore [missing-attribute]
    "namedtuple": lambda fn, x: _mapping_key(fn, x._asdict(), type(x)),
    # pyrefly: ignore [no-matching-overload, bad-argument-type]
    "dataclass": lambda fn, x: _mapping_key(
        fn,
        # pyrefly: ignore [bad-argument-type]
        {f.name: getattr(x, f.name) for f in dataclasses.fields(x)},
        type(x),
    ),
    types.FunctionType: _function_key,
    types.BuiltinFunctionType: lambda fn, x: x,
    torch.fx.GraphModule: _graph_module_key,
    # pyrefly: ignore [missing-attribute]
    ConstExpr: lambda fn, x: _constexpr_key(x.value),
    type(None): lambda fn, x: None,
}


def maybe_skip_dtype_check_in_meta_registrations() -> contextlib.AbstractContextManager[
    None, None
]:
    # pyrefly: ignore [implicit-import]
    if hasattr(torch.fx.experimental._config, "skip_dtype_check_in_meta_registrations"):
        # pyrefly: ignore [implicit-import, missing-attribute]
        return torch.fx.experimental._config.patch(
            skip_dtype_check_in_meta_registrations=True
        )
    return contextlib.nullcontext()
