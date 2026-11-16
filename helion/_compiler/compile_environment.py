from __future__ import annotations

import ast
import builtins
import collections
import contextlib
import contextvars
import dataclasses
import itertools
import logging
import math
import os
import sys
import threading
import traceback
import types
import typing
from typing import TYPE_CHECKING
import warnings

import sympy
import torch
from torch import SymInt
from torch._dynamo.source import AttrSource
from torch._dynamo.source import EphemeralSource
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._inductor.codegen.wrapper import (
    user_defined_triton_kernel_transitive_closure_source_code,
)
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._subclasses import FakeTensor
import torch.distributed as dist
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.symbolic_shapes import StatelessSymbolicContext
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils import _pytree as pytree
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._sympy.value_ranges import bound_sympy

from .. import exc
from .._compat import is_hip
from .._compat import shape_env_size_hint
from .._compat import target_device_capability
from .._compat import tensor_descriptor_supports_dtype
from .._utils import triton_is_available
from ..language.constexpr import ConstExpr
from .backend import BackendState
from .backend_registry import get_backend
from .fake_tensor import create_fake_tensor_mode
from .source_location import SourceLocation
from .source_location import current_location
from .variable_origin import AttributeOrigin
from .variable_origin import BlockSizeOrigin
from .variable_origin import GetItemOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin

log = logging.getLogger(__name__)

TensorDescriptorLayoutSignature = tuple[int | None, tuple[bool, ...]]
# (source projections, expected classifier result) -> predicate over call args
PreparedGuardFactory = typing.Callable[
    [
        typing.Sequence[typing.Callable[[typing.Sequence[object]], typing.Hashable]],
        typing.Hashable,
    ],
    typing.Callable[[typing.Sequence[object]], bool],
]
# CUDA TMA limits each box dimension to 256 elements. Other descriptor
# backends have their own legality checks and must not inherit this cap.
CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE = 256


@dataclasses.dataclass(eq=False)
class _ContainerIdentity:
    # Only the frontend owns the original. TypeInfo retains the emptied token.
    value: object = dataclasses.field(repr=False)
    opaque_result: bool = False


def _supports_container_identity(value: object) -> bool:
    return (
        type(value) in (tuple, list, dict)
        or (dataclasses.is_dataclass(value) and not isinstance(value, type))
        or (isinstance(value, tuple) and hasattr(type(value), "__match_args__"))
    )


@dataclasses.dataclass(frozen=True)
class ConfigValueExpression:
    """Small integer expression whose leaves are emitted config values."""

    operation: str
    arguments: tuple[int | str | ConfigValueExpression, ...]

    def evaluate(self, config: Config) -> int:
        def value(arg: int | str | ConfigValueExpression) -> int:
            if isinstance(arg, ConfigValueExpression):
                return arg.evaluate(config)
            if isinstance(arg, str):
                result = config[arg]
                if not isinstance(result, int):
                    raise TypeError(f"config value {arg!r} is not an integer")
                return result
            return arg

        args = tuple(value(arg) for arg in self.arguments)
        if self.operation == "config":
            assert len(self.arguments) == 1 and isinstance(self.arguments[0], str)
            return value(self.arguments[0])
        if self.operation == "cdiv":
            assert len(args) == 2
            return (args[0] + args[1] - 1) // args[1]
        if self.operation == "next_power_of_2":
            assert len(args) == 1
            return next_power_of_2(args[0])
        if self.operation == "min":
            return min(args)
        if self.operation == "max":
            return max(args)
        if self.operation == "add":
            return sum(args)
        if self.operation == "mul":
            return math.prod(args)
        if self.operation == "floordiv":
            assert len(args) == 2
            return args[0] // args[1]
        raise ValueError(f"unknown config expression operation {self.operation!r}")

    def __str__(self) -> str:
        """The expression as kernel code spells it (``next_power_of_2(foo)``,
        ``(foo + 1)``)."""
        args = [str(arg) for arg in self.arguments]
        if self.operation == "config":
            return args[0]
        infix = {"add": " + ", "mul": " * ", "floordiv": " // "}.get(self.operation)
        if infix is not None:
            return f"({infix.join(args)})"
        return f"{self.operation}({', '.join(args)})"

    def config_keys(self) -> list[str]:
        """The config keys the expression reads, in order: the tunables
        under a value a helper computes from them."""
        keys = []
        for arg in self.arguments:
            if isinstance(arg, ConfigValueExpression):
                keys.extend(arg.config_keys())
            elif isinstance(arg, str):
                keys.append(arg)
        return keys


@dataclasses.dataclass
class TensorDescriptorLayoutGuard:
    ndim: int
    element_size: int
    memory_op_indices: set[int] = dataclasses.field(default_factory=set)
    atomic_op_indices: set[int] = dataclasses.field(default_factory=set)
    has_derived_block_extent: bool = False


@dataclasses.dataclass
class TensorDescriptorAlignmentGuard:
    memory_op_indices: set[int] = dataclasses.field(default_factory=set)
    atomic_op_indices: set[int] = dataclasses.field(default_factory=set)
    requires_zero_storage_offset: bool = False


@dataclasses.dataclass(frozen=True)
class RuntimeInputSpecialization:
    """Internal runtime-input projection used to extend a kernel cache key.

    ``classifier_identity`` distinguishes classifier semantics when compiler
    discovery registers the same named projection more than once.
    ``reusable_tensor_properties`` is an optional promise that the classifier's
    result depends only on the named storage properties (plus tensor metadata
    already covered by eager dispatch guards), never on tensor contents; or,
    for ``int`` argument sources, only on the argument's value (``int_value``).
    It lets code generation consume the result recorded in the bound kernel key.
    ``prepared_guard_factory`` optionally builds, from the source projections
    and an expected classifier result, a cheaper predicate over the raw call
    arguments that holds exactly when the classifier would produce that result
    again. Fast/prepared dispatch evaluates it only after pinning every
    argument's type, dtype, shape, stride and device, so the predicate needs to
    read just the facts those pins leave open, such as pointers, storage spans
    or tensor layout. Factories must also prove that their source paths are
    covered by those pins; an arbitrary attribute is not covered.
    Runtime cache state is deliberately excluded from descriptor equality.
    """

    sources: tuple[Source, ...]
    classifier_identity: typing.Hashable
    classifier: typing.Callable[[typing.Sequence[object]], typing.Hashable] = (
        dataclasses.field(compare=False, repr=False)
    )
    reusable_tensor_properties: frozenset[
        typing.Literal["data_ptr", "storage_span", "int_value"]
    ] = frozenset()
    prepared_guard_factory: PreparedGuardFactory | None = dataclasses.field(
        default=None, compare=False, repr=False
    )
    # Captured objects distinguish otherwise identical classifiers between
    # bindings. Retain them, compare only identity, and never serialize them.
    binding_anchors: tuple[object, ...] = dataclasses.field(
        default=(), compare=False, repr=False
    )
    # Classifier results may retain explicit input owners. This cache-lifetime
    # requirement also applies to later traces inheriting the published guard.
    retains_input_identity: bool = dataclasses.field(default=False, compare=False)

    def same_binding_semantics(self, other: RuntimeInputSpecialization) -> bool:
        return (
            self == other
            and len(self.binding_anchors) == len(other.binding_anchors)
            and all(
                left is right
                for left, right in zip(
                    self.binding_anchors, other.binding_anchors, strict=True
                )
            )
        )


def _is_supported_tensor_input_source(source: Source) -> bool:
    if isinstance(source, LocalSource):
        return True
    if isinstance(source, AttrSource):
        # ClosureOrigin records function.__closure__[index].cell_contents.
        # Captures retain their external specialization policy.
        if (
            source.member == "cell_contents"
            and isinstance(source.base, GetItemSource)
            and isinstance(source.base.base, AttrSource)
            and source.base.base.member == "__closure__"
        ):
            return False
        return _is_supported_tensor_input_source(source.base)
    if isinstance(source, GetItemSource):
        return (
            isinstance(source.index, (int, str))
            and not source.index_is_slice
            and _is_supported_tensor_input_source(source.base)
        )
    return False


# Set by the torch.compile HOP lowering on its example inputs: the dims
# Dynamo's graph keeps static, which the captured trace bound as literals.
HELION_STATIC_DIMS_ATTR = "_helion_static_dims"


def _static_dims_symbolic_context(
    tensor: torch.Tensor,
) -> StatelessSymbolicContext | None:
    """A dynamic-shape fakeification of ``tensor`` that keeps its
    ``HELION_STATIC_DIMS_ATTR`` dims static (None: the default context)."""
    static_dims = getattr(tensor, HELION_STATIC_DIMS_ATTR, None)
    if not static_dims:
        return None
    ndim = tensor.dim()
    return StatelessSymbolicContext(
        dynamic_sizes=[
            DimDynamic.STATIC if i in static_dims else DimDynamic.DUCK
            for i in range(ndim)
        ],
        dynamic_strides=[DimDynamic.INFER_STRIDE] * ndim,
        constraint_sizes=[None] * ndim,
        constraint_strides=[None] * ndim,
    )


def tensor_descriptor_runtime_alignment_signature(
    value: object,
) -> tuple[bool, bool]:
    """Return the runtime alignment predicates used by descriptor codegen."""
    if not isinstance(value, torch.Tensor) or type(value).__name__ in (
        "FakeTensor",
        "FunctionalTensor",
    ):
        return False, False
    offset = value.storage_offset()
    return value.data_ptr() % 16 == 0, isinstance(offset, int) and offset == 0


def _concrete_tensor_base_is_aligned(value: object) -> bool:
    return tensor_descriptor_runtime_alignment_signature(value)[0]


def _concrete_tensor_satisfies_alignment_guard(
    value: object, requires_zero_storage_offset: bool
) -> bool:
    aligned, zero_storage_offset = tensor_descriptor_runtime_alignment_signature(value)
    return aligned and (not requires_zero_storage_offset or zero_storage_offset)


# Wrapper factories whose result is a fresh allocation (new storage starting
# at an allocator-aligned base) unless ``out=`` or an aliasing argument says
# otherwise; ``register_tensor_factory_layout`` records that provenance.  The
# ``Tensor.new_*`` methods are registered with the receiver as the first
# argument.
_FRESH_ALLOCATION_FACTORIES: tuple[object, ...] = (
    torch.empty,
    torch.empty_like,
    torch.empty_strided,
    torch.zeros,
    torch.zeros_like,
    torch.ones,
    torch.ones_like,
    torch.full,
    torch.full_like,
    # Clones allocate storage without proving zero contents or exact layout.
    torch.clone,
    torch.Tensor.clone,
    # Constant padding copies even when every pad is zero or negative.
    # Its contents are not uniformly zero and its layout is proved separately.
    torch.constant_pad_nd,
    torch.Tensor.new_empty,
    torch.Tensor.new_empty_strided,
    torch.Tensor.new_zeros,
    torch.Tensor.new_ones,
    torch.Tensor.new_full,
)
# Factories whose fresh storage starts as positive zeros, and the ``full``
# family with the positional slot of its fill value (``fill_value=`` otherwise).
_ZERO_FILLED_FACTORIES: tuple[object, ...] = (
    torch.zeros,
    torch.zeros_like,
    torch.Tensor.new_zeros,
)
_FILL_VALUE_FACTORIES: dict[object, int] = {
    torch.full: 1,
    torch.full_like: 1,
    torch.Tensor.new_full: 2,
}


def _factory_fills_positive_zero(
    factory: object,
    args: typing.Sequence[object],
    kwargs: typing.Mapping[str, object],
) -> bool:
    """Whether the wrapper factory call fills its allocation with ``+0``.

    Only a literal Python zero counts: ``-0.0`` is a different value for the
    sign-of-zero reasoning this feeds, and a traced scalar is unknown.
    """
    if factory in _ZERO_FILLED_FACTORIES:
        return True
    position = _FILL_VALUE_FACTORIES.get(factory)
    if position is None:
        return False
    fill = kwargs.get("fill_value", args[position] if len(args) > position else None)
    if isinstance(fill, bool) or not isinstance(fill, (int, float)):
        return False
    return fill == 0 and math.copysign(1.0, float(fill)) > 0


def _replay_tensor_input_source(
    source: Source,
    root_values: typing.Mapping[str, object],
) -> object:
    if isinstance(source, LocalSource):
        return root_values.get(source.local_name)
    if isinstance(source, AttrSource):
        base = _replay_tensor_input_source(source.base, root_values)
        return getattr(base, source.member, None) if base is not None else None
    if isinstance(source, GetItemSource):
        if not isinstance(source.index, (int, str)) or source.index_is_slice:
            return None
        base = _replay_tensor_input_source(source.base, root_values)
        if (
            isinstance(source.index, int)
            and isinstance(base, (list, tuple))
            and 0 <= source.index < len(base)
        ):
            return base[source.index]
        if isinstance(base, dict) and source.index in base:
            return base[source.index]
    return None


def _find_tensor_input_source(
    target: torch.Tensor,
    value: object,
    source: Source,
) -> Source | None:
    if value is target:
        return source
    attributes = False
    if isinstance(value, dict):
        items = value.items()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        attributes = True
        items = (
            (field.name, getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        attributes = True
        items = ((name, getattr(value, name)) for name in value._fields)
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return None
    for index, item in items:
        if isinstance(index, (int, str)):
            result = _find_tensor_input_source(
                target,
                item,
                AttrSource(source, typing.cast("str", index))
                if attributes
                else GetItemSource(source, index),
            )
            if result is not None:
                return result
    return None


def _search_tensor_input_source(
    tensor: torch.Tensor, arguments: typing.Mapping[str, object]
) -> Source | None:
    """The first source in ``arguments``, the kernel's arguments by parameter
    name, that replays to ``tensor``."""
    for name, value in arguments.items():
        source = _find_tensor_input_source(
            tensor, value, LocalSource(name, is_input=True)
        )
        if source is not None:
            return source
    return None


def tensor_input_sources(
    arguments: typing.Mapping[str, object], tensors: typing.Collection[torch.Tensor]
) -> frozenset[Source]:
    """The sources ``CompileEnvironment.tensor_input_source`` gives ``tensors``
    in a binding of ``arguments``, the kernel's arguments by parameter name."""
    return frozenset(
        source
        for tensor in tensors
        if (source := _search_tensor_input_source(tensor, arguments)) is not None
    )


def tensor_descriptor_layout_signature_from_strides(
    strides: typing.Sequence[int | torch.SymInt | sympy.Integer],
    element_size: int,
    size_hint: typing.Callable[[int | torch.SymInt], int] | None = None,
) -> TensorDescriptorLayoutSignature:
    """Return the stride layout facts tensor descriptors depend on.

    The signature intentionally records predicates rather than exact strides so
    dynamic-shape kernels can share code across sizes that have the same tensor
    descriptor eligibility.
    """
    stride_one_dim: int | None = None
    has_multiple_stride_one_dims = False
    aligned_dims = []
    for dim, raw_stride in enumerate(strides):
        if isinstance(raw_stride, sympy.Integer):
            stride = int(raw_stride)
        elif isinstance(raw_stride, int):
            stride = raw_stride
        else:
            if size_hint is None:
                raise TypeError(
                    "symbolic tensor descriptor strides require an explicit size_hint"
                )
            stride = size_hint(raw_stride)
        if stride == 1:
            if stride_one_dim is None:
                stride_one_dim = dim
            else:
                has_multiple_stride_one_dims = True
        aligned_dims.append((stride * element_size) % 16 == 0)
    if has_multiple_stride_one_dims:
        stride_one_dim = None
    return stride_one_dim, tuple(aligned_dims)


def _make_numel_check(
    symbols: list[sympy.Basic], expr: sympy.Basic
) -> typing.Callable[..., bool]:
    """Evaluate a sympy constraint with concrete block-size values."""

    def check(*args: int) -> bool:
        return bool(expr.subs(list(zip(symbols, args, strict=True))))

    return check


if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet
    from types import TracebackType
    from typing_extensions import Self

    from torch._guards import Source
    from torch.fx.experimental.symbolic_shapes import ShapeGuard

    from .. import Config
    from ..autotuner.config_fragment import ConfigSpecFragment
    from ..runtime.settings import Settings
    from .ast_extension import ExtendedAST
    from .autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from .backend import Backend
    from .pallas.compact_worklist import CompactWorklistPlan
    from .pallas.compact_worklist import ResidentCacheDecision
    from .pallas.compact_worklist import ResidentPrepHoist


class _TLS(threading.local):
    env: CompileEnvironment | None = None


tls = _TLS()


class HelionKernelSource(EphemeralSource):
    """Ephemeral source that formats as a kernel file location."""

    class _CompatSourceName(str):
        """String that is also callable (for torch<=2.9 which calls `source.name()`)."""

        __slots__ = ()

        def __call__(self) -> str:
            return self

    def __init__(self, location: SourceLocation) -> None:
        super().__init__()
        self.location = location

    @property
    def name(self) -> str:  # type: ignore[override]
        formatted = self.location.format().rstrip("\n")
        if not formatted:
            return ""
        return self._CompatSourceName("\nHelion kernel stack:\n" + formatted)


def _current_symbol_source() -> EphemeralSource | None:
    location = current_location()
    if not location:
        return None
    return HelionKernelSource(location)


def shape_env_var_hints(shape_env: ShapeEnv) -> dict[sympy.Symbol, sympy.Integer]:
    # torch renamed ShapeEnv.var_to_val -> ShapeEnv.backed_var_to_val.
    if (backed_var_to_val := getattr(shape_env, "backed_var_to_val", None)) is not None:
        return typing.cast("dict[sympy.Symbol, sympy.Integer]", backed_var_to_val)
    return shape_env.var_to_val  # pyrefly: ignore [deprecated]


# The configs a test of config values is tried under, one by one, where the
# ranges of the values do not decide it; past them it counts as failing.
_CONFIGS_TRIED = 4096


def _holds_for_sizes(
    expr: sympy.Basic, ranges: dict[sympy.Expr, ValueRanges] | None = None
) -> bool:
    """Whether ``expr`` holds whenever each of its symbols in ``ranges`` lies
    in its range and each other one is an integer of at least 2, as Helion
    takes a symbolic size to be (its hint keeps ``x == 1`` checks false)."""
    ranges = ranges or {}
    shifted = expr.xreplace(
        {
            symbol: sympy.Add(
                sympy.Dummy(integer=True, nonnegative=True), sympy.Integer(2)
            )
            for symbol in expr.free_symbols
            if symbol not in ranges
        }
    )
    if shifted is sympy.true:
        return True
    try:
        bound = bound_sympy(
            typing.cast("sympy.Expr", shifted),
            {
                typing.cast("sympy.Symbol", symbol): ranges.get(
                    typing.cast("sympy.Expr", symbol), ValueRanges(0, int_oo)
                )
                for symbol in shifted.free_symbols
            },
        )
    except (KeyError, AttributeError, NotImplementedError):
        # A function bound_sympy has no rule for proves nothing.
        return False
    return bound.is_singleton() and bound.lower is sympy.true


# Where torch and Helion live: the guards their code records take a symbolic
# size to be at least 2, as Helion does throughout.
_COMPILER_DIRS = tuple(
    os.path.join(path, "")
    for path in (
        *torch.__path__,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
)


def _in_builtin_call(location: object) -> bool:
    """Whether ``location``, a guard's frame, is where Helion calls a
    function the kernel's code calls (``_CheckForIndexCalls.retry_call``): a
    builtin such as ``max`` compares there, in no frame of its own."""
    return (
        isinstance(location, traceback.FrameSummary)
        and location.name == "retry_call"
        and location.filename.endswith(os.path.join("language", "tile_proxy.py"))
    )


def _recorded_by_compiler(guard: ShapeGuard) -> bool:
    """Whether torch's shape machinery or Helion's compiler recorded
    ``guard`` (``Ne(u0, 1)`` when broadcasting a dim, a contiguity test, a
    tile packing's ``extent != 1``), as opposed to a comparison the kernel's
    own code makes: that records its frame (a helper defined outside the
    kernel, say), or Helion's call of a builtin the kernel calls
    (``max(foo, 2)``)."""
    location = guard.sloc.framework_loc
    return (
        location is not None
        and not isinstance(location, str)
        and location.filename.startswith(_COMPILER_DIRS)
        and not _in_builtin_call(location)
    )


# Legacy ShapeEnv unifier retained for Helion.
def _constrain_unify(shape_env: ShapeEnv, a: SymInt, b: SymInt) -> None:
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            if a != b:
                raise AssertionError(f"Expected {a} == {b}")
        else:
            if not isinstance(b.node.expr, sympy.Symbol):
                raise AssertionError("constraining non-Symbols NYI")
            if b.node.shape_env is not shape_env:
                raise AssertionError("b.node.shape_env must be shape_env")
            shape_env.replacements[b.node.expr] = sympy.Integer(a)
    else:
        # NB: We can't actually do "unification" as our operators are not
        # injective
        if not isinstance(a.node.expr, sympy.Symbol):
            raise AssertionError("constraining non-Symbols NYI")
        if a.node.shape_env is not shape_env:
            raise AssertionError("a.node.shape_env must be shape_env")
        if not isinstance(b, SymInt):
            shape_env.replacements[a.node.expr] = sympy.Integer(b)
        else:
            if a.node.shape_env is not b.node.shape_env:
                raise AssertionError("a.node.shape_env must be b.node.shape_env")
            if not isinstance(b.node.expr, sympy.Symbol):
                raise AssertionError("constraining non-Symbols NYI")
            new_var = shape_env._find(a.node.expr)
            shape_env.replacements[b.node.expr] = new_var


class _GuardAxioms(dict):
    """The axioms of a ``_GuardedShapeEnv``: an update that adds axioms (a
    new guard's, in ``ShapeEnv._evaluate_expr``) advances the ShapeEnv's
    version counter."""

    def __init__(self, shape_env: ShapeEnv) -> None:
        super().__init__()
        self._shape_env = shape_env

    def update(self, *args: typing.Any, **kwargs: typing.Any) -> None:  # noqa: ANN401
        axioms = len(self)
        super().update(*args, **kwargs)
        if len(self) != axioms:
            # TODO(jansel): drop this once PyTorch's ShapeEnv advances the
            # counter itself: ``_get_key`` leaves out the guards and their
            # axioms, and ``evaluate_expr`` appends a guard without calling
            # ``_update_version_counter``.
            self._shape_env._version_counter += 1


class _GuardedShapeEnv(ShapeEnv):
    """A ShapeEnv whose cached evaluations are decided by every guard added
    before them.

    ``ShapeEnv`` caches ``_maybe_evaluate_static``, the ``simplify`` it starts
    from (which decides ``Max(0, a - b)`` from the guards), ``get_axioms`` and
    its other evaluations keyed on a version counter that a new guard does not
    advance, although the guard's axioms can decide them: ``Eq(u1, u2)``,
    cached as undecided, stays undecided after the guards that imply it, and
    ``get_axioms`` leaves the guard out.  So adding axioms advances the
    counter (``_GuardAxioms``), which drops every such cache.  It happens in
    the axioms' update, after torch records the guard: a guard records as its
    location the first frame outside torch, which is the code that compared,
    not a frame of this class.
    """

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self.axioms = _GuardAxioms(self)


class CompileEnvironment:
    """
    Global state for the duration of a compilation.
    There is a 1:1 mapping between this and a BoundKernel,
    and a single CompileEnvironment will be used for multiple Configs.
    No config or codegen specific state should be stored here.
    """

    def __init__(
        self,
        device: torch.device,
        settings: Settings,
        *,
        index_dtype: torch.dtype | None = None,
        is_distributed: bool = False,
    ) -> None:
        from ..autotuner.config_spec import ConfigSpec

        super().__init__()
        # pyrefly: ignore [read-only]
        self.device = device
        self.settings = settings
        self.index_dtype: torch.dtype = (
            index_dtype or settings.index_dtype or torch.int32
        )
        self._is_distributed = is_distributed
        self.process_group_name = None
        self._backend = get_backend(settings.backend)
        self._backend.validate_environment()
        rng_streams = self._backend.rng_streams()
        if settings.cute_rng_stream not in rng_streams:
            raise ValueError(
                f"cute_rng_stream={settings.cute_rng_stream!r} is not supported by "
                f"the {settings.backend} backend; expected one of "
                f"{', '.join(rng_streams)}"
            )
        # The backend's per-compile state (``Backend.new_env_state``); read
        # through the backend's typed accessor, never by name from core.
        self.backend_state: object | None = self._backend.new_env_state()
        if (
            settings.debug_dtype_asserts
            and not self._backend.supports_debug_dtype_asserts
        ):
            raise exc.BackendUnsupported(
                self._backend.name,
                "debug_dtype_asserts=True (it emits Triton tl.cast / "
                "tl.static_assert checks)",
            )
        if self._backend.state is BackendState.EXPERIMENTAL:
            from torch._dynamo.utils import warn_once

            warn_once(
                f"The '{self._backend.name}' backend is experimental and may have limited functionality.",
            )
        # For dynamic kernels, keep 0/1 tensor dimensions symbolic so a kernel
        # first seen with size 0 or 1 can be reused for larger sizes.
        self.shape_env = _GuardedShapeEnv(
            specialize_zero_one=settings.static_shapes,
            duck_shape=False,
            assume_static_by_default=settings.static_shapes,
        )
        # TODO(jansel): check for guards in the shapeenv
        self.fake_mode = create_fake_tensor_mode(shape_env=self.shape_env)
        self.input_sources: dict[torch.Tensor, Source] = {}
        self._ambiguous_tensor_input_source_ids: set[int] = set()
        # Positive provenance for host allocations whose layout is fixed by the
        # generated wrapper.  Track storage rather than tensor identity so a
        # deterministic view of such an allocation retains the proof, while a
        # view of a user input cannot acquire it merely because it has no direct
        # replayable input source.
        self._symbolically_exact_layout_storages: set[torch.UntypedStorage] = set()
        # Storage of every fresh wrapper allocation, whatever its layout proof:
        # its base is allocator-aligned, so a static storage offset decides the
        # base alignment of any view of it.
        self._fresh_allocation_storages: set[torch.UntypedStorage] = set()
        self._zero_filled_allocation_storages: set[torch.UntypedStorage] = set()
        # A host control-flow join has synthetic fake storage, not a runtime
        # allocation. Retain original storage separately from watched Tensor
        # owners: a view keeps its storage when its former owner calls set_.
        self._joined_tensor_storage_origins: dict[
            torch.UntypedStorage, tuple[torch.UntypedStorage, ...]
        ] = {}
        self._joined_tensor_storage_owners: dict[
            torch.UntypedStorage,
            tuple[tuple[torch.Tensor, torch.UntypedStorage], ...],
        ] = {}
        # Metadata belongs to Tensor objects, not their shared storage. A
        # selected Tensor aliases an owner; a view of it has independent shape
        # and stride metadata even though its storage origins are identical.
        self._joined_tensor_metadata_owners: dict[
            torch.Tensor, tuple[torch.Tensor, ...]
        ] = {}
        # Branches and zero-trip loop bodies share representative tensors.
        # A storage replacement makes both old and new storage ambiguous even
        # before a join is registered, including for ordinary tensor pairs.
        self._invalidated_storage_provenance: set[torch.UntypedStorage] = set()
        # A host-selected tensor can change layout without changing the input
        # signature. Derived host results must not inherit a representative
        # layout, even when input shapes and strides are otherwise static.
        self._runtime_layout_storages: set[torch.UntypedStorage] = set()
        self._runtime_layout_stride_symbols: dict[
            torch.UntypedStorage, set[sympy.Symbol]
        ] = {}
        self._runtime_arg_values_by_name: contextvars.ContextVar[
            dict[str, object] | None
        ] = contextvars.ContextVar(
            f"helion_runtime_arg_values_{id(self)}",
            default=None,
        )
        self._dispatch_guards_enforced: contextvars.ContextVar[bool] = (
            contextvars.ContextVar(
                f"helion_dispatch_guards_enforced_{id(self)}",
                default=True,
            )
        )
        self._input_alignment_limit: contextvars.ContextVar[int | None] = (
            contextvars.ContextVar(
                f"helion_input_alignment_limit_{id(self)}",
                default=None,
            )
        )
        self._symbolic_stride_sources: contextvars.ContextVar[frozenset[Source]] = (
            contextvars.ContextVar(
                f"helion_symbolic_stride_sources_{id(self)}",
                default=frozenset(),
            )
        )
        # Set when an input is an outer trace's FakeTensor (``_to_fake_tensor``):
        # its hints are the ones that trace chose, recorded here so that code
        # generation refuses them whether or not the FakeTensor is still alive.
        self._binds_outer_trace_inputs = False
        # Host integer helpers such as cdiv/next_power_of_2 deliberately return
        # unbacked SymInts during tracing. Preserve the config expression beside
        # that symbol so a fixed block size derived from a user tunable can still
        # be resolved for each candidate configuration.
        self.config_value_expressions: dict[sympy.Expr, ConfigValueExpression] = {}
        self.block_sizes: list[BlockSizeInfo] = []
        # The reduction dims a subscript's static extent allocated
        # (``new_index_result``) and, for each one a reduction was lowered
        # over, the number of block sizes when the first such reduction was
        # (``record_reduction_over``).  Older versions padded such a subscript
        # to the power of two its extent rounds up to, unless a full slice of
        # the extent had a dim of its own, so that reduction allocated a
        # reduction dim of the power of two then, and the configs they saved
        # have reduction_loops entries for those dims
        # (``DeviceIR._record_legacy_reduction_loops``).
        self.subscript_extent_block_ids: set[int] = set()
        self.subscript_extent_reductions: dict[int, int] = {}
        # The root device loop type propagation is typing
        # (``TypePropagation.visit_For``); for each subscript extent's dim, the
        # root loops whose subscripts typed it while its size was literal; the
        # dims a full slice of their extent joined; and those a full slice
        # joined in one of those loops.  Typed before a full slice of its
        # extent in the same loop, such a subscript kept its padded size in
        # older versions, and so did the reductions over it, also where the
        # slice is inside a branch or an inner loop
        # (``DeviceIR._padded_range_dims``).
        self.typing_root_id: int | None = None
        self.literal_subscript_roots: dict[int, set[int]] = {}
        self.subscript_extents_joined_by_slice: set[int] = set()
        self.subscript_extents_joined_in_loop: set[int] = set()
        self.debug_shape_renames: dict[sympy.Basic, sympy.Basic] = {}
        self._debug_shape_rename_override: contextvars.ContextVar[
            dict[sympy.Basic, sympy.Basic] | None
        ] = contextvars.ContextVar(
            f"helion_debug_shape_renames_{id(self)}",
            default=None,
        )
        # The symbols of ``config_value_expressions`` no host variable names
        # (a host helper written inline, ``block_size=[next_power_of_2(foo)]``):
        # code generation substitutes the value each takes under the config
        # it compiles (``config_constants``).  Recorded once the type pass has
        # given host variables their origins (``propagate_types``).
        self.inline_config_symbols: frozenset[sympy.Symbol] = frozenset()
        try:
            from ..runtime import get_num_sm

            _num_sm = get_num_sm(device, reserved_sms=settings.persistent_reserved_sms)
        except Exception:
            _num_sm = 1
        self.config_spec = ConfigSpec(
            backend=self.backend,
            target_device_capability=target_device_capability(device),
            device=device,
            num_sm=_num_sm,
            log_restrictions_verbose=settings.autotune_log_search_space_verbose,
        )
        # Correctness facts registered by compiler heuristics can depend on
        # dynamic runtime inputs even when seed generation is disabled or later
        # found ineligible. Bound-kernel caching consumes this set separately
        # from specialization requirements of heuristics that emitted seeds.
        self.compiler_fact_specialization_facts: frozenset[
            CompilerHeuristicSpecializationFact
        ] = frozenset()
        # TODO(hinriksnaer): tracing state, not env config. move to CompilerState?
        self.kernel_tensor_sizes: dict[tuple[sympy.Expr, ...], int] = (
            collections.Counter()
        )
        # TODO(hinriksnaer): tracing state, not env config. move to CompilerState?
        self.kernel_min_element_bits: int = 32  # smallest dtype bits across all tensors
        self.specialized_vars: set[sympy.Symbol] = set()
        # The specialized_vars that were 1 at this binding. Unlike
        # hl.specialize, they can differ between bindings of one signature,
        # whose dispatch schema the first binding publishes; each stays a
        # guard fact so it keys every bound kernel.
        self.singleton_vars: set[sympy.Symbol] = set()
        # Input sizes code generation relied on being equal (a block store
        # bounded by its tensor covering the tile's loop) only at this
        # binding's values; like a guard, each keys the bound kernel.
        self.keyed_size_equalities: set[tuple[sympy.Symbol, sympy.Expr]] = set()
        self.specialized_strides: set[TensorPropertySource] = set()
        # Config-backed values created by hl.register_tunable().
        self.tunable_symbols: set[sympy.Symbol] = set()
        # Revisit a declaration with the same symbol after validating its
        # snapshot. Separate call contexts need separate host variable origins.
        self.tunable_values: dict[
            tuple[str, tuple[ExtendedAST, ...]],
            torch.SymInt | torch.SymFloat | torch.SymBool,
        ] = {}
        # Repeated visits reuse exact built-in domain snapshots. Keep originals
        # alive so object IDs cannot be recycled; custom fragments stay shared.
        self.tunable_fragment_snapshots: dict[
            int, tuple[ConfigSpecFragment, ConfigSpecFragment]
        ] = {}
        # Exact primitive globals before normal symbolification. Fragment
        # constructors may snapshot these values without reading the source
        # again. Only operations with fully proved scalar operands extend it;
        # dynamic inputs and untracked helper expressions never enter this map.
        # Retain scalar occurrences, not SymPy expressions: algebraic
        # simplification can erase floating rounding, signed zero and inputs.
        # The strong scalar reference also prevents object-id reuse. These
        # records never own runtime tensors or enter persistent cache keys.
        self.fragment_global_literals: dict[
            int,
            tuple[torch.SymInt | torch.SymFloat | torch.SymBool, int | float | bool],
        ] = {}
        self._symbolic_fragment_domain_warned = False
        self.tensor_descriptor_layout_guards: dict[
            Source, TensorDescriptorLayoutGuard
        ] = {}
        self.tensor_descriptor_alignment_guards: dict[
            Source, TensorDescriptorAlignmentGuard
        ] = {}
        self.bound_tensor_descriptor_alignments: dict[Source, bool] = {}
        self.runtime_input_specializations: dict[str, RuntimeInputSpecialization] = {}
        # Original inputs/proxies are retained only until frontend compilation
        # completes. Identity witnesses never become serialized cache-key data.
        self._container_identities: dict[int, tuple[object, _ContainerIdentity]] = {}
        # Opaque Python container identities cannot be guarded by value metadata.
        self.requires_fresh_binding = False
        # Opaque helper calls can inspect a fragment's Python identity. Their
        # bindings retain explicit constexpr owners and need a bounded cache.
        self.has_opaque_fragment_inputs = False
        # Immutable classifier outputs captured from the arguments that created
        # this BoundKernel.  Codegen may run later and obtain those arguments
        # through weak references, after their storage metadata has changed.
        # Runtime-dependent optimizations must match this snapshot before they
        # consume a live alignment or aliasing fact.
        self.bound_runtime_input_specialization_results: dict[str, typing.Hashable] = {}
        # Relocatable metadata facts are exported without granting the reusable
        # storage-property/cache promises of the mapping above.
        self.bound_metadata_specialization_results: dict[str, typing.Hashable] = {}
        self._tensor_input_source_cache: dict[int, Source | None] = {}
        self.jagged_tile_parent_ids: dict[int, list[int]] = {}
        self.jagged_tile_mask_shapes: dict[int, list[torch.SymInt]] = {}
        # Set by the Pallas backend when worklist grouping is 1 or 2 and
        # detect_compact_worklist_plan succeeds; gates the compact-worklist
        # codegen path (see helion/_compiler/pallas/compact_worklist.py).
        self.compact_worklist_plan: CompactWorklistPlan | None = None
        # Final resident-cache decision for this concrete config.  This includes
        # the cached physical window integer; runtime/codegen consumers must read
        # this instead of recomputing resident-cache eligibility.
        self.compact_worklist_resident_cache_decision: ResidentCacheDecision | None = (
            None
        )
        # Optional prep-cache descriptors admitted for this concrete config.  Kept
        # separate from the correctness-bearing resident-window decision.
        self.compact_worklist_resident_prep_hoists: tuple[ResidentPrepHoist, ...] = ()
        # Static megablocks upper bound (int) for the compact worklist grid /
        # metadata, computed at pre_codegen from static shapes.
        self.compact_worklist_upper: int = 1
        # Compact-axis tile block size (int), resolved from the config at
        # pre_codegen; used by the worklist builder and UPPER (NOT max(block_sizes)).
        self.compact_worklist_block: int = 1
        # Ordered (reduction) tile block size.  May differ from the compact block
        # (e.g. compact_block != ordered_block); resident caching sizes its window to a
        # multiple of THIS so a single ordered tile read always fits the window.
        self.compact_worklist_ordered_block: int = 1
        # Offsets-tensor parameter names the generated _build_worklist takes, in
        # order (set when the builder is emitted); used by the launcher to map
        # them to host-call arg positions.
        self.compact_worklist_offset_params: list[str] = []
        self._symint_cache: dict[object, torch.SymInt] = {}
        self._input_symint_cache: dict[Source, torch.SymInt] = {}
        self._foreign_symint_cache: dict[
            tuple[int, sympy.Expr], int | torch.SymInt
        ] = {}
        # The distributed restriction is deferred to
        # restrict_pid_types_for_persistent() so it can gate on a real per-kernel
        # signal after tracing rather than the process-global dist.is_initialized().
        # force_persistent restricts pid_types unconditionally; the symm-mem
        # signal-pad clamp is symm-mem-specific and left to
        # restrict_pid_types_for_persistent().
        if settings.autotune_force_persistent:
            self._disallow_nonpersistent_pid_types(
                reason="autotune_force_persistent is set"
            )

        # TODO(hinriksnaer): tracing flag, not env config. move to CompilerState?
        self.has_barrier: bool = False
        # hl.pdl_* ops in source order, run at kernel entry or exit.
        self.pdl_entry: list[str] = []
        self.pdl_exit: list[str] = []

    def warn_symbolic_fragment_domain(self) -> None:
        """Emit at most once per binding, subject to Python's warning filters."""
        if not self._symbolic_fragment_domain_warned:
            self._symbolic_fragment_domain_warned = True
            location = current_location()
            warnings.warn_explicit(
                "Symbolic config fragment domains are deprecated. Use hl.constexpr "
                "for domain values, or hl.specialize for integer shapes and arguments. "
                "Legacy fixed configs remain supported; symbolic domains are not "
                "snapshotted and autotuning may fail when it evaluates them.",
                FutureWarning,
                filename=location.filename,
                lineno=location.lineno or 0,
            )

    def _disallow_nonpersistent_pid_types(self, reason: str | None = None) -> None:
        """Restrict the search space to persistent kernels. Idempotent."""
        for pid_type in ("flat", "xyz"):
            self.config_spec.disallow_pid_type(pid_type, reason=reason)

    def require_persistent_blocked(self, reason: str) -> None:
        """Restrict program-ID selection to blocked persistent execution."""
        for pid_type in ("flat", "xyz", "persistent_interleaved"):
            self.config_spec.disallow_pid_type(pid_type, reason=reason)

    def restrict_pid_types_for_persistent(self, args: Sequence[object]) -> None:
        """Restrict to persistent kernels when the kernel needs cross-rank sync.

        Called after tracing so it can gate on a real per-kernel signal (an
        ``hl.barrier()`` or a symmetric-memory tensor argument) rather than the
        process-global ``dist.is_initialized()``, which would needlessly shrink
        the search space for every kernel in a distributed process. A barrier or
        symm-mem tensor forces persistent pid_types; the signal-pad clamp is a
        symm-mem-only constraint, so a barrier-only kernel keeps its full
        ``max_num_sm_multiplier`` range.
        """
        if not dist.is_initialized():
            return

        # Two independent signals: a barrier forces persistent pid_types, while a
        # symmetric-memory kernel additionally needs the signal-pad clamp.
        # ``_is_distributed`` already folds in ``kernel_uses_symm_mem(args)``; scan
        # the args only as the newer-torch fallback when it is unset.
        uses_symm_mem = self._is_distributed
        if not uses_symm_mem:
            from .._dist_utils import is_symm_mem_tensor

            uses_symm_mem = any(
                isinstance(arg, torch.Tensor)
                and is_symm_mem_tensor(arg, self.process_group_name)
                for arg in args
            )

        if not uses_symm_mem and not self.has_barrier:
            return

        self._disallow_nonpersistent_pid_types(
            reason="a distributed process group is initialized (persistent "
            "kernels required)"
        )
        if uses_symm_mem:
            self._clamp_max_num_sm_multiplier_for_symm_mem()

    def _clamp_max_num_sm_multiplier_for_symm_mem(self) -> None:
        """Clamp max_num_sm_multiplier to the symmetric-memory signal-pad budget."""
        # CUDA symmetric-memory persistent-kernel sizing only. Guard on CUDA: the
        # Pallas/TPU backend traces with a cpu-device torch tensor (the torch<->jax
        # bridge), so under a multi-host (dist-initialized) serve this would call
        # get_num_sm(cpu) -> "TODO: implement for other devices" and crash the
        # kernel compile. _SymmetricMemory / SM-multiplier are irrelevant to Pallas.
        if self.device.type != "cuda":
            return

        from torch._C._distributed_c10d import _SymmetricMemory

        from .._dist_utils import max_num_blocks_for_symm_mem
        from ..runtime import get_num_sm

        num_sms = get_num_sm(
            self.device, reserved_sms=self.settings.persistent_reserved_sms
        )
        # Floor to previous power of two since PowerOfTwoFragment requires pow2 bounds
        raw_max = min(
            max_num_blocks_for_symm_mem() // num_sms,
            self.config_spec.max_num_sm_multiplier,
        )
        newmax = 1 << (raw_max.bit_length() - 1) if raw_max > 0 else 1
        if newmax < self.config_spec.max_num_sm_multiplier:
            warnings.warn(
                f"max_num_sm_multipler is reduced from {self.config_spec.max_num_sm_multiplier} to {newmax} due to the restriction of _SymmetricMemory.signal_pad_size={_SymmetricMemory.signal_pad_size}. Increase the signal pad size to allow autotuner to choose among all possible values in the range.",
                stacklevel=1,
            )
        self.config_spec.max_num_sm_multiplier = newmax

    @property
    def runtime_arg_values_by_name(self) -> dict[str, object]:
        return self._runtime_arg_values_by_name.get() or {}

    @contextlib.contextmanager
    def use_runtime_arg_values(
        self, values: dict[str, object]
    ) -> typing.Iterator[None]:
        token = self._runtime_arg_values_by_name.set(values)
        try:
            yield
        finally:
            self._runtime_arg_values_by_name.reset(token)

    @property
    def dispatch_guards_enforced(self) -> bool:
        """Whether every call the code being generated serves passed this
        binding's dispatch guards, so that what its cache key fixes (exact
        input sizes and strides under ``input_tensor_metadata``, the size and
        stride residues of a runtime input specialization) holds for each of
        them.

        True for the bound kernel's own launches: dispatch routes a call to it
        only when the call matches its key.  False while Inductor renders the
        kernel into a dynamic-shape graph (``without_dispatch_guards``): it
        binds the kernel to placeholder inputs (every dynamic size 64, every
        dynamic stride 1) and runs the code at every size the graph accepts.
        """
        return self._dispatch_guards_enforced.get()

    @contextlib.contextmanager
    def without_dispatch_guards(
        self, symbolic_strides: typing.Collection[Source] = ()
    ) -> typing.Iterator[None]:
        """Generate code that runs without this binding's dispatch guards, for
        calls whose inputs in ``symbolic_strides`` (``tensor_input_sources``)
        come at any strides (``traced_strides_hold``)."""
        token = self._dispatch_guards_enforced.set(False)
        strides_token = self._symbolic_stride_sources.set(frozenset(symbolic_strides))
        try:
            yield
        finally:
            self._symbolic_stride_sources.reset(strides_token)
            self._dispatch_guards_enforced.reset(token)

    @property
    def symbolic_stride_sources(self) -> frozenset[Source]:
        """The inputs whose strides vary across the calls the code being
        generated serves (``without_dispatch_guards``); none for the bound
        kernel's own launches."""
        return self._symbolic_stride_sources.get()

    @property
    def input_alignment_limit(self) -> int | None:
        """The modulus, in bytes, of the input base-pointer residues that hold
        for every call the code being generated serves, or None for those
        the binding records.

        None for the bound kernel's own launches: dispatch checks each
        residue its runtime input specializations record.  Inductor renders
        the kernel for its graph's inputs and buffers
        (``limit_input_alignment``), which it aligns only to
        ``GPU_ALIGN_BYTES``; the placeholder inputs it binds the kernel to
        reproduce their residues only modulo that, static graphs included.
        """
        return self._input_alignment_limit.get()

    def input_residues_hold_modulo(self, alignment: int) -> bool:
        """Whether the bound base residues hold modulo ``alignment`` bytes for
        every call the code being generated serves: always, unless an Inductor
        render's ``input_alignment_limit`` is not a multiple of it."""
        limit = self.input_alignment_limit
        return limit is None or limit % alignment == 0

    @contextlib.contextmanager
    def limit_input_alignment(self, limit: int) -> typing.Iterator[None]:
        """Generate code for callers that align inputs only to ``limit`` bytes."""
        token = self._input_alignment_limit.set(limit)
        try:
            yield
        finally:
            self._input_alignment_limit.reset(token)

    @property
    def active_debug_shape_renames(self) -> dict[sympy.Basic, sympy.Basic]:
        return self._debug_shape_rename_override.get() or self.debug_shape_renames

    @property
    def has_debug_shape_rename_override(self) -> bool:
        return self._debug_shape_rename_override.get() is not None

    @contextlib.contextmanager
    def use_debug_shape_renames(
        self, values: dict[sympy.Basic, sympy.Basic]
    ) -> typing.Iterator[None]:
        token = self._debug_shape_rename_override.set(
            {**self.debug_shape_renames, **values}
        )
        try:
            yield
        finally:
            self._debug_shape_rename_override.reset(token)

    def specialize_expr(self, expr: sympy.Expr) -> sympy.Expr:
        """Substitute any specialized vars with their concrete values."""
        if subs := {
            s: sympy.Integer(shape_env_size_hint(self.shape_env, s))
            for s in expr.free_symbols & self.specialized_vars
        }:
            # pyrefly: ignore [bad-assignment]
            expr = expr.xreplace(subs)
        return expr

    def static_kernel_tensor_extent(self, size: int | torch.SymInt) -> int | None:
        """Resolve a logical tensor extent without treating a tile as its full axis."""
        if isinstance(size, int):
            return size
        expr = self.specialize_expr(self.shape_env.simplify(size._sympy_()))
        block_id = self.resolve_block_id(size)
        if not isinstance(expr, sympy.Integer) and block_id is not None:
            block = self.block_sizes[block_id]
            if block.reduction:
                # Full-axis loads carry a reduction block symbol even when
                # the logical extent is constant. Tiled axes do not.
                expr = self.specialize_expr(self.shape_env.simplify(block.numel))
        return int(expr) if isinstance(expr, sympy.Integer) else None

    def config_constants(self, expr: sympy.Expr, config: Config) -> sympy.Expr:
        """``expr`` with each of ``inline_config_symbols`` replaced by the
        value it takes under ``config`` (``config_value_expressions``)."""
        if not self.inline_config_symbols:
            return expr
        free = expr.free_symbols
        symbols = [symbol for symbol in self.inline_config_symbols if symbol in free]
        if not symbols:
            return expr
        # pyrefly: ignore [bad-return]
        return expr.xreplace(
            {
                symbol: sympy.Integer(
                    self.config_value_expressions[symbol].evaluate(config)
                )
                for symbol in symbols
            }
        )

    def register_tensor_descriptor_layout_guard(
        self,
        fake_tensor: torch.Tensor,
        *,
        memory_op_index: int | None = None,
        atomic_op_index: int | None = None,
        has_derived_block_extent: bool = False,
    ) -> None:
        """Specialize kernels on replayable tensor-descriptor predicates."""
        if not tensor_descriptor_supports_dtype(fake_tensor.dtype):
            # No descriptor takes this tensor (``TensorDescriptorIndexingStrategy``
            # declines it), so its layout cannot change the code.
            return
        source = self.tensor_input_source(fake_tensor)
        has_direct_source = source is not None and _is_supported_tensor_input_source(
            source
        )
        # Triton tensor descriptors need a 16-byte aligned base on CUDA (TMA)
        # and on XPU, where a descriptor over a misaligned base reads from
        # the address rounded down.  Other tensor-descriptor backends retain
        # their existing legality checks and must not acquire this runtime
        # specialization.
        alignment_source = (
            self.tensor_descriptor_alignment_source(fake_tensor)
            if self.backend_name == "triton" and self.device.type in ("cuda", "xpu")
            else None
        )
        if alignment_source is not None:
            alignment_guard = self.tensor_descriptor_alignment_guards.setdefault(
                alignment_source, TensorDescriptorAlignmentGuard()
            )
            if memory_op_index is not None:
                alignment_guard.memory_op_indices.add(memory_op_index)
            if atomic_op_index is not None:
                alignment_guard.atomic_op_indices.add(atomic_op_index)
            alignment_guard.requires_zero_storage_offset |= not has_direct_source

        if not has_direct_source:
            return
        assert source is not None
        guard = self.tensor_descriptor_layout_guards.setdefault(
            source,
            TensorDescriptorLayoutGuard(
                ndim=fake_tensor.ndim,
                element_size=fake_tensor.element_size(),
            ),
        )
        if memory_op_index is not None:
            guard.memory_op_indices.add(memory_op_index)
        if atomic_op_index is not None:
            guard.atomic_op_indices.add(atomic_op_index)
        guard.has_derived_block_extent |= has_derived_block_extent

    def has_tensor_descriptor_layout_guard(self, fake_tensor: torch.Tensor) -> bool:
        source = self.tensor_input_source(fake_tensor)
        return (
            source is not None
            and _is_supported_tensor_input_source(source)
            and source in self.tensor_descriptor_layout_guards
        )

    def tensor_descriptor_base_is_aligned(self, fake_tensor: torch.Tensor) -> bool:
        """Whether a tensor descriptor can prove its runtime base is 16B aligned."""
        source = self.tensor_descriptor_alignment_source(fake_tensor)
        if source in self.bound_tensor_descriptor_alignments:
            return self.bound_tensor_descriptor_alignments[source]
        runtime_value = self.runtime_value_for_tensor(fake_tensor)
        if _concrete_tensor_base_is_aligned(runtime_value):
            return True
        if isinstance(runtime_value, torch.Tensor):
            return False
        if (
            fake_tensor.untyped_storage()
            not in self._symbolically_exact_layout_storages
        ):
            return False
        storage_offset = fake_tensor.storage_offset()
        return (
            isinstance(storage_offset, int)
            and (storage_offset * fake_tensor.element_size()) % 16 == 0
        )

    def _tensor_input_storage_owners(
        self, fake_tensor: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, Source], ...]:
        """Trace an exact view to inputs sharing its storage, not disjoint inputs.

        This is provenance only. Different fake storages do not establish any
        runtime alias fact. Static/exact layouts retain the traced view/copy
        decision when a bound kernel is reused.
        """
        if (
            not (
                self.settings.static_shapes
                or self.tensor_layout_is_symbolically_exact(fake_tensor)
            )
            or not all(isinstance(value, int) for value in fake_tensor.size())
            or not all(isinstance(value, int) for value in fake_tensor.stride())
        ):
            return ()
        return tuple(
            (tensor, source)
            for tensor, source in self.input_sources.items()
            if tensor.untyped_storage() == fake_tensor.untyped_storage()
        )

    def tensor_storage_input_source(self, fake_tensor: torch.Tensor) -> Source | None:
        """Find a unique input owner for a whole-storage runtime predicate.

        Unlike pointer alignment, storage-span disjointness also applies to
        nonzero-offset views. The caller must still establish its registered,
        cache-key-backed runtime predicate on the returned input Source.
        """
        source = self.tensor_input_source(fake_tensor)
        if source is not None and _is_supported_tensor_input_source(source):
            return source
        owners = self._tensor_input_storage_owners(fake_tensor)
        if len(owners) != 1:
            return None
        tensor, source = owners[0]
        if (
            id(tensor) in self._ambiguous_tensor_input_source_ids
            or not _is_supported_tensor_input_source(source)
        ):
            return None
        return source

    def tensor_alignment_owner(
        self, fake_tensor: torch.Tensor
    ) -> tuple[Source, int] | None:
        """The input whose runtime base ``fake_tensor`` starts a fixed number of bytes past.

        A direct input is its own owner at offset zero.  A statically exact
        view of the storage of exactly one zero-offset input starts
        ``storage_offset * element_size`` bytes past that input's base, so
        its base residue follows from the owner's bound residue and the
        static offset.  Layout legality remains a separate proof.
        """
        source = self.tensor_input_source(fake_tensor)
        if source is not None and _is_supported_tensor_input_source(source):
            return source, 0
        storage_offset = fake_tensor.storage_offset()
        if (
            not isinstance(storage_offset, int)
            or not (
                self.settings.static_shapes
                or self.tensor_layout_is_symbolically_exact(fake_tensor)
            )
            or not all(isinstance(value, int) for value in fake_tensor.size())
            or not all(isinstance(value, int) for value in fake_tensor.stride())
        ):
            return None

        def is_zero_offset(tensor: torch.Tensor) -> bool:
            offset = tensor.storage_offset()
            return isinstance(offset, int) and offset == 0

        owners = tuple(
            (tensor, candidate)
            for tensor, candidate in self._tensor_input_storage_owners(fake_tensor)
            if is_zero_offset(tensor)
            and _is_supported_tensor_input_source(candidate)
            and id(tensor) not in self._ambiguous_tensor_input_source_ids
        )
        if len(owners) != 1:
            return None
        return owners[0][1], storage_offset * fake_tensor.element_size()

    def tensor_descriptor_alignment_source(
        self, fake_tensor: torch.Tensor
    ) -> Source | None:
        """Find the input whose base-alignment predicate applies to ``fake_tensor``.

        A zero-offset, statically exact view has the same data pointer as its
        unique input storage owner.  Layout legality remains a separate proof;
        this only lets the descriptor reuse that owner's runtime alignment guard.
        """
        owner = self.tensor_alignment_owner(fake_tensor)
        if owner is None or owner[1] != 0:
            return None
        return owner[0]

    def snapshot_tensor_descriptor_alignments(
        self, root_values: typing.Mapping[str, object]
    ) -> None:
        """Capture descriptor base-alignment facts for this bound kernel."""
        self.bound_tensor_descriptor_alignments = {
            source: _concrete_tensor_satisfies_alignment_guard(
                value, guard.requires_zero_storage_offset
            )
            for source, guard in self.tensor_descriptor_alignment_guards.items()
            if (value := _replay_tensor_input_source(source, root_values)) is not None
        }

    def tensor_input_source(self, fake_tensor: torch.Tensor) -> Source | None:
        """Return a replayable source for a direct or container tensor input."""
        cache_key = id(fake_tensor)
        if cache_key in self._tensor_input_source_cache:
            return self._tensor_input_source_cache[cache_key]
        if cache_key in self._ambiguous_tensor_input_source_ids:
            self._tensor_input_source_cache[cache_key] = None
            return None

        source = self.input_sources.get(fake_tensor)
        from .host_function import HostFunction

        root_values = HostFunction.current().params.arguments
        if (
            source is not None
            and _is_supported_tensor_input_source(source)
            and _replay_tensor_input_source(source, root_values) is fake_tensor
        ):
            result = source
        else:
            result = _search_tensor_input_source(fake_tensor, root_values)

        self._tensor_input_source_cache[cache_key] = result
        return result

    def tensor_stride_input_source(self, fake_tensor: torch.Tensor) -> Source | None:
        """Resolve a runtime stride source, including an aliased input leaf.

        Full, fast and prepared dispatch guard the input tensor alias partition,
        so any recorded path to this exact receiver supplies its current stride.
        Keep the unique-owner query conservative for consumers that need storage
        ownership rather than receiver identity.
        """
        source = self.tensor_input_source(fake_tensor)
        if source is not None:
            return source
        if id(fake_tensor) not in self._ambiguous_tensor_input_source_ids:
            return None
        source = self.input_sources.get(fake_tensor)
        from .host_function import HostFunction

        if (
            source is not None
            and _is_supported_tensor_input_source(source)
            and _replay_tensor_input_source(
                source, HostFunction.current().params.arguments
            )
            is fake_tensor
        ):
            return source
        return None

    def tensor_layout_is_symbolically_exact(self, tensor: torch.Tensor) -> bool:
        """Whether the wrapper determines this tensor's layout exactly.

        This is intentionally a positive proof.  Absence from ``input_sources``
        is insufficient: input views and aliases also commonly lack a direct
        source.  Storage identity lets deterministic views of either a proven
        compiler allocation or one fully stride-specialized, unaliased input
        share the proof without separately classifying every view operation.
        """
        storage = tensor.untyped_storage()
        if storage in self._invalidated_storage_provenance:
            return False
        if storage in self._symbolically_exact_layout_storages:
            return True

        # A view does not have its own replayable input source.  Its layout is
        # nevertheless fixed when the sole input owning its storage has every
        # stride explicitly specialized.  Refuse shared input storage: another
        # input alias has independent metadata (including storage offset), so
        # blessing the storage from only one tensor would not be a proof.
        input_aliases = tuple(
            (input_tensor, source)
            for input_tensor, source in self.input_sources.items()
            if input_tensor.untyped_storage() == storage
        )
        if len(input_aliases) != 1:
            return False
        input_tensor, source = input_aliases[0]
        if id(input_tensor) in self._ambiguous_tensor_input_source_ids:
            return False
        return all(
            TensorPropertySource(source, TensorProperty.STRIDE, dim)
            in self.specialized_strides
            for dim in range(input_tensor.ndim)
        )

    def known_stride(self, tensor: torch.Tensor, dim: int) -> int | None:
        """``tensor.stride(dim)`` when every call the kernel serves has it,
        else None (the stride is read at run time).

        A dynamic-shape kernel's traced stride is only the first call's
        unless something fixes it: the wrapper allocating the tensor, or an
        ``hl.specialize`` of the stride. A subsequent host metadata mutation
        voids that entry-stride specialization.
        """
        v = tensor.stride(dim)
        # Only literalize a dynamic-kernel stride with positive provenance that
        # the generated wrapper fixes this layout.  A missing input source is
        # not such a proof: views and aliases of inputs commonly have none.
        if isinstance(v, int) and self.tensor_layout_is_symbolically_exact(tensor):
            return v
        source = self.tensor_input_source(tensor)
        # An explicitly specialized input stride remains known only while host
        # metadata mutations have not made its layout a runtime property.
        if (
            source is not None
            and not self.tensor_has_runtime_layout(tensor)
            and TensorPropertySource(source, TensorProperty.STRIDE, dim)
            in self.specialized_strides
        ):
            return int(v)
        if isinstance(v, int) and self.settings.static_shapes:
            return v
        return None

    def tensor_storage_provenance(
        self, tensor: torch.Tensor
    ) -> tuple[tuple[torch.UntypedStorage, bool], ...] | None:
        """Possible storage origins and positive allocation facts for each.

        An ordinary fake tensor contributes its storage, not proof of freshness.
        Only an independently recorded factory allocation or guarded input
        source can establish disjointness for that origin.
        """
        storage = tensor.untyped_storage()
        origins = self._joined_tensor_storage_origins.get(storage, (storage,))
        if storage in self._invalidated_storage_provenance or any(
            origin in self._invalidated_storage_provenance for origin in origins
        ):
            return None
        return tuple(
            (origin, origin in self._fresh_allocation_storages) for origin in origins
        )

    def fresh_allocation_is_disjoint_from_input(
        self, allocation: torch.Tensor, tensor: torch.Tensor
    ) -> bool:
        """A surviving allocation fact proves independence from older inputs.

        Opaque effects discard every allocation fact already known at the call.
        A later fresh allocation is therefore independent of a direct input even
        when that input's earlier span proof is invalid. A synthetic joined
        storage is never such an input or a direct allocation.
        """
        allocation_storage = allocation.untyped_storage()
        input_storage = tensor.untyped_storage()
        return (
            allocation_storage != input_storage
            and allocation_storage not in self._joined_tensor_storage_origins
            and input_storage not in self._joined_tensor_storage_origins
            and self.tensor_storage_provenance(allocation)
            == ((allocation_storage, True),)
            and any(
                known.untyped_storage() == input_storage for known in self.input_sources
            )
        )

    def _invalidate_tensor_storage_provenance(
        self, storages: typing.Iterable[torch.UntypedStorage]
    ) -> None:
        """Discard facts permanently; later input guards cannot restore them."""
        invalid = self._invalidated_storage_provenance
        invalid.update(storages)
        # A join is an alias of one of its owners at runtime. Propagate both
        # upward from a changed owner and downward from a changed receiver,
        # including through nested joins and owners' current/original storage.
        changed = True
        while changed:
            changed = False
            for storage, origins in self._joined_tensor_storage_origins.items():
                related = {storage, *origins}
                for owner, original in self._joined_tensor_storage_owners[storage]:
                    related.update((original, owner.untyped_storage()))
                if invalid.intersection(related) and not related <= invalid:
                    invalid.update(related)
                    changed = True
        self._fresh_allocation_storages.difference_update(invalid)
        self._zero_filled_allocation_storages.difference_update(invalid)
        self._symbolically_exact_layout_storages.difference_update(invalid)

    def register_tensor_storage_join(
        self, result: torch.Tensor, branches: typing.Sequence[torch.Tensor]
    ) -> None:
        """Flatten host-join alternatives without granting allocation provenance."""
        result_storage = result.untyped_storage()
        origins: dict[torch.UntypedStorage, None] = {}
        owners: dict[int, tuple[torch.Tensor, torch.UntypedStorage]] = {}
        metadata_owners: dict[int, torch.Tensor] = {}
        for branch in branches:
            branch_storage = branch.untyped_storage()
            provenance = self.tensor_storage_provenance(branch)
            if provenance is None:
                self._invalidated_storage_provenance.add(result_storage)
            else:
                origins.update((storage, None) for storage, _fresh in provenance)
            # Keep the intermediate joined Tensor as well as its leaf owners.
            # Tensor objects sharing a storage can later change independently.
            for owner in self._joined_tensor_storage_owners.get(branch_storage, ()):
                owners.setdefault(id(owner[0]), owner)
            owners.setdefault(id(branch), (branch, branch_storage))
            for owner in self._joined_tensor_metadata_owners.get(branch, ()):
                metadata_owners.setdefault(id(owner), owner)
            metadata_owners.setdefault(id(branch), branch)
        self._joined_tensor_storage_origins[result_storage] = tuple(origins)
        self._joined_tensor_storage_owners[result_storage] = tuple(owners.values())
        self._joined_tensor_metadata_owners[result] = tuple(metadata_owners.values())

    def _invalidate_mutated_tensor_storage_joins(self) -> None:
        """A mutation permanently invalidates the original alias alternatives."""
        changed = set()
        for storage, owners in self._joined_tensor_storage_owners.items():
            for tensor, original_storage in owners:
                current = tensor.untyped_storage()
                if current is not original_storage:
                    # Never substitute the current owner: earlier views retain
                    # old storage, and a traced sibling may not run at runtime.
                    changed.update((storage, original_storage, current))
        if changed:
            self._invalidate_tensor_storage_provenance(changed)

    def tensor_storage_is_compiler_allocated(self, fake_tensor: torch.Tensor) -> bool:
        """Whether ``fake_tensor`` views storage a wrapper factory freshly allocated.

        Such storage starts at an allocator-aligned base, so the base alignment
        of a view follows from its static storage offset.  Lacking an input
        source is not enough: input views and dtype-punning aliases lack one
        too while inheriting an arbitrary runtime base.
        """
        return fake_tensor.untyped_storage() in self._fresh_allocation_storages

    def register_compiler_allocation_replacement(
        self, original: torch.Tensor, replacement: torch.Tensor
    ) -> None:
        """Retain allocation facts after rewriting a proved host factory.

        The caller must rewrite the runtime allocation as well as its fake
        value, preserving the factory's initialization. This is not a proof
        for arbitrary tensor conversions or synthetic control-flow joins.
        Layout facts remain separate and are not transferred here.
        """
        original_storage = original.untyped_storage()
        replacement_storage = replacement.untyped_storage()
        if (
            original_storage not in self._fresh_allocation_storages
            or replacement_storage == original_storage
        ):
            return
        self._fresh_allocation_storages.add(replacement_storage)
        if original_storage in self._zero_filled_allocation_storages:
            self._zero_filled_allocation_storages.add(replacement_storage)

    def tensor_has_canonical_contiguous_strides(self, tensor: torch.Tensor) -> bool:
        """Prove the canonical dense formula without ordering by size hints."""
        expected: int | torch.SymInt = 1
        for size, stride in reversed(
            tuple(zip(tensor.size(), tensor.stride(), strict=True))
        ):
            if not self.known_equal(stride, expected):
                return False
            expected = expected * torch.sym_max(1, size)
        return True

    def _prove_empty_like_layout(
        self, tensor: torch.Tensor, result: torch.Tensor
    ) -> bool:
        from .runtime_tensor_guards import _guard_canonical_layout

        # In rank one, compaction always yields stride one for a nondegenerate
        # extent. Empty/singleton tensors instead preserve arbitrary strides.
        if tensor.ndim == 0 or (
            tensor.ndim == 1 and isinstance(tensor.size(0), int) and tensor.size(0) > 1
        ):
            if tensor.ndim:
                with torch.no_grad():
                    result.as_strided_(result.size(), (1,), result.storage_offset())
            return True
        if (
            tensor.ndim == 1
            and isinstance(tensor.size(0), torch.SymInt)
            and not self.tensor_has_canonical_contiguous_strides(tensor)
        ):
            # A noncanonical one-dimensional input still compacts to stride
            # one when its actual extent exceeds one. Zero/singleton inputs
            # share the runtime-stride path without changing their metadata.
            if _guard_canonical_layout(
                self, tensor, require_input=False, coalesce_degenerate=True
            ) == (2,):
                with torch.no_grad():
                    result.as_strided_(result.size(), (1,), result.storage_offset())
                return True
        if self.tensor_has_runtime_layout(tensor):
            return False
        trusted = (
            self.settings.static_shapes
            or self.tensor_layout_is_symbolically_exact(tensor)
        )
        if trusted and all(
            isinstance(value, int) for value in (*tensor.size(), *tensor.stride())
        ):
            return True
        if not (
            self.tensor_has_canonical_contiguous_strides(tensor)
            and self.tensor_has_canonical_contiguous_strides(result)
        ):
            return False
        classes = _guard_canonical_layout(self, tensor, require_input=not trusted)
        if classes is None:
            return False
        # Guarded extent classes make the clamp explicit without asking
        # ShapeEnv to simplify Max(1, size) using its bind-time ranges.
        strides: list[int | torch.SymInt] = []
        expected: int | torch.SymInt = 1
        for size, extent_class in reversed(
            tuple(zip(result.size(), classes, strict=True))
        ):
            strides.append(expected)
            expected = expected * (size if extent_class in (None, 2) else 1)
        shape = tuple(
            size if extent_class in (None, 2) else extent_class
            for size, extent_class in zip(result.size(), classes, strict=True)
        )
        with torch.no_grad():
            result.as_strided_(shape, tuple(reversed(strides)), result.storage_offset())
        return True

    def tensor_storage_is_zero_filled_allocation(
        self, fake_tensor: torch.Tensor
    ) -> bool:
        """Whether ``fake_tensor`` views fresh wrapper storage filled with ``+0``.

        ``torch.zeros`` and friends, or a ``full`` with a literal positive
        zero, recorded by ``register_tensor_factory_layout``.  The zeros are
        the storage's initial contents only; what the host and the kernel do
        to it afterwards is the caller's proof.
        """
        return fake_tensor.untyped_storage() in self._zero_filled_allocation_storages

    def tensor_has_runtime_layout(self, tensor: torch.Tensor) -> bool:
        return tensor.untyped_storage() in self._runtime_layout_storages

    def mark_runtime_tensor_layout(self, tensor: torch.Tensor) -> None:
        storage = tensor.untyped_storage()
        self._runtime_layout_storages.add(storage)
        self._symbolically_exact_layout_storages.discard(storage)

    def _metadata_permutation_preserves_storage(
        self, before: tuple[int | torch.SymInt, ...], tensor: torch.Tensor
    ) -> bool:
        """Prove an unchanged element-address set, including empty dimensions."""
        rank = tensor.ndim
        if len(before) != 2 * rank + 1 or not self.known_equal(
            before[-1], tensor.storage_offset()
        ):
            return False
        remaining = list(zip(before[:rank], before[rank:-1], strict=True))
        for size, stride in zip(tensor.size(), tensor.stride(), strict=True):
            for i, (old_size, old_stride) in enumerate(remaining):
                if self.known_equal(size, old_size) and self.known_equal(
                    stride, old_stride
                ):
                    remaining.pop(i)
                    break
            else:
                return False
        return True

    def _refresh_mutated_tensor_metadata(
        self,
        tensor: torch.Tensor,
        changed_dimensions: set[int],
        mutation_site: tuple[object, ...],
    ) -> None:
        """Read post-call metadata from this Tensor, without reusing old origins."""
        shape = tuple(
            self.cached_create_unbacked_symint(
                ("host_mutation_size", tensor, mutation_site, dim), self.size_hint(size)
            )
            if dim in changed_dimensions
            else size
            for dim, size in enumerate(tensor.size())
        )
        strides = tuple(
            self.cached_create_unbacked_symint(
                ("host_mutation_stride", tensor, mutation_site, dim),
                self.size_hint(stride),
            )
            for dim, stride in enumerate(tensor.stride())
        )
        self.mark_runtime_tensor_layout(tensor)
        symbols = self._runtime_layout_stride_symbols.setdefault(
            tensor.untyped_storage(), set()
        )
        for stride in strides:
            symbols.update(_symint_free_symbols(stride))
        with torch.no_grad():
            tensor.as_strided_(shape, strides, tensor.storage_offset())

    @contextlib.contextmanager
    def track_runtime_layout_mutations(
        self, inputs: object, *, opaque: bool
    ) -> typing.Iterator[None]:
        """Retain exact aliases only when the host call preserves their metadata."""
        from .ast_extension import ExtendedAST

        # A second statement can mutate the same Tensor after an earlier
        # allocation captured its size. Reusing that statement's symbols would
        # rewrite the earlier allocation's extent to the final Tensor state.
        # Repeated inference of the same loop site still needs stable symbols.
        mutation_site = tuple(ExtendedAST.current())
        if opaque:
            # Python can replace .data and restore it without changing the
            # Tensor version. Its captures/record fields can reach owners not
            # present in the arguments, even before any join exists. Factories
            # after this call can still establish independent fresh storage.
            known = (
                self._fresh_allocation_storages
                | self._zero_filled_allocation_storages
                | self._symbolically_exact_layout_storages
                | set(self._joined_tensor_storage_origins)
                | {tensor.untyped_storage() for tensor in self.input_sources}
            )
            for origins in self._joined_tensor_storage_origins.values():
                known.update(origins)
            for owners in self._joined_tensor_storage_owners.values():
                for owner, original in owners:
                    known.update((original, owner.untyped_storage()))
            self._invalidate_tensor_storage_provenance(known)
        tensors = [
            value
            for value in pytree.tree_leaves(inputs)
            if isinstance(value, torch.Tensor)
        ]
        storage_before = [(value, value.untyped_storage()) for value in tensors]
        before = [
            (
                value,
                value.untyped_storage(),
                tuple(value.size()),
                (*value.size(), *value.stride(), value.storage_offset()),
            )
            for value in tensors
        ]
        yield
        changed = set()
        for value, storage in storage_before:
            current_storage = value.untyped_storage()
            if current_storage is not storage:
                changed.update((storage, current_storage))
        if changed:
            self._invalidate_tensor_storage_provenance(changed)
        # Native value writes do not change storage. Opaque calls were already
        # invalidated, so neither version counters nor inference-mode counters
        # are needed for the remaining storage watches.
        self._invalidate_mutated_tensor_storage_joins()
        for value, storage, original_shape, metadata in before:
            current = (*value.size(), *value.stride(), value.storage_offset())
            if (
                value.untyped_storage() is not storage
                or len(metadata) != len(current)
                or any(
                    not self.known_equal(left, right)
                    for left, right in zip(metadata, current, strict=True)
                )
            ):
                if not self._metadata_permutation_preserves_storage(metadata, value):
                    # Entry span/alignment facts do not describe resized or
                    # restrided views. A pure dimension permutation preserves
                    # the actual address set, even for zero/singleton sizes.
                    self._invalidate_tensor_storage_provenance((storage,))
                # Only the selected Tensor itself shares its owners' metadata;
                # an actual view merely shares storage. Upward propagation then
                # refreshes every selection that may name a changed owner,
                # without resizing that selection's other, untouched owners.
                affected = dict.fromkeys(
                    (value, *self._joined_tensor_metadata_owners.get(value, ()))
                )
                while True:
                    dependent = [
                        joined
                        for joined, owners in self._joined_tensor_metadata_owners.items()
                        if joined not in affected
                        and any(owner in affected for owner in owners)
                    ]
                    if not dependent:
                        break
                    affected.update(dict.fromkeys(dependent))
                changed_dimensions = {
                    dim
                    for dim, (old_size, new_size) in enumerate(
                        zip(original_shape, value.size(), strict=False)
                    )
                    if not self.known_equal(old_size, new_size)
                }
                if len(original_shape) != value.ndim:
                    if (
                        len(affected) > 1
                        or any(
                            isinstance(
                                node, (ast.If, ast.IfExp, ast.For, ast.While, ast.Try)
                            )
                            # The first Boolean operand and first comparison pair
                            # run unconditionally. Later operands may short-circuit.
                            or (
                                isinstance(node, ast.BoolOp)
                                and child is not node.values[0]
                            )
                            or (
                                isinstance(node, ast.Compare)
                                and any(
                                    child is value for value in node.comparators[1:]
                                )
                            )
                            # Only a comprehension's outer iterable runs eagerly
                            # once. Its filters and body may run zero or many times.
                            or (
                                isinstance(
                                    node,
                                    (
                                        ast.ListComp,
                                        ast.SetComp,
                                        ast.DictComp,
                                        ast.GeneratorExp,
                                    ),
                                )
                                and child is not node.generators[0].iter
                            )
                            for node, child in itertools.pairwise(mutation_site)
                        )
                    ):
                        raise exc.TypeInferenceError(
                            "Host tensor metadata mutation cannot change its rank across control flow"
                        )
                    changed_dimensions = set(range(value.ndim))
                for tensor in affected:
                    if tensor.ndim != value.ndim:
                        raise exc.TypeInferenceError(
                            "Mutation through a host tensor join cannot change its rank"
                        )
                    self._refresh_mutated_tensor_metadata(
                        tensor, changed_dimensions, mutation_site
                    )

    def propagate_runtime_tensor_layout(self, inputs: object, result: object) -> None:
        """Carry host-selection layout uncertainty through tensor operations."""
        input_storages = {
            value.untyped_storage()
            for value in pytree.tree_leaves(inputs)
            if isinstance(value, torch.Tensor)
        }
        if input_storages & self._runtime_layout_storages:
            for value in pytree.tree_leaves(result):
                if (
                    isinstance(value, torch.Tensor)
                    and value.untyped_storage() not in input_storages
                ):
                    # Views already share their source's marker. The host-call
                    # snapshot separately handles mutations of existing inputs.
                    self.mark_runtime_tensor_layout(value)

    def runtime_layout_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Use actual host strides while preserving the fake storage alias."""
        storage = tensor.untyped_storage()
        self.mark_runtime_tensor_layout(tensor)
        known_symbols = self._runtime_layout_stride_symbols.setdefault(storage, set())
        strides: list[int | torch.SymInt] = []
        changed = False
        for dim, stride in enumerate(tensor.stride()):
            symbols = (
                _symint_free_symbols(stride)
                if isinstance(stride, torch.SymInt)
                else set()
            )
            if symbols and symbols <= known_symbols:
                # A view of this storage already expresses its runtime layout.
                # A fresh allocation has its own symbol set and origin.
                strides.append(stride)
                continue
            new_stride = self.cached_create_unbacked_symint(
                ("runtime_tensor_layout", tensor, dim), self.size_hint(stride)
            )
            known_symbols.update(_symint_free_symbols(new_stride))
            strides.append(new_stride)
            changed = True
        if not changed:
            return tensor
        result = torch.as_strided(
            tensor, tensor.size(), strides, tensor.storage_offset()
        )
        if tensor in self._joined_tensor_metadata_owners:
            # This is an internal representation change, not a runtime view.
            self._joined_tensor_metadata_owners[result] = (
                self._joined_tensor_metadata_owners[tensor]
            )
        return result

    def register_tensor_factory_layout(
        self,
        factory: object,
        args: typing.Sequence[object],
        kwargs: typing.Mapping[str, object],
        result: object,
    ) -> None:
        """Record allocation and layout provenance for wrapper factory calls.

        Every factory in ``_FRESH_ALLOCATION_FACTORIES`` that neither writes
        ``out=`` nor aliases an argument produces fresh storage, recorded for
        base-alignment proofs; for ``Tensor.new_*`` the receiver is passed as
        the first argument.  Exact layout provenance is narrower:
        ``torch.empty/empty_strided/zeros/ones/full`` and ``Tensor.new_*``
        create fresh layouts determined by their shape/stride arguments,
        not the receiver's strides.
        ``torch.empty_like`` can compact a non-dense input. Exactness requires
        a fixed trusted layout or a proved canonical dense layout; ordinary
        inputs also carry a runtime layout guard. Ambiguous dynamic layouts
        retain runtime stride reads.
        Other factories conservatively remain runtime-strided until their
        layout contracts are added here.
        """
        if factory not in _FRESH_ALLOCATION_FACTORIES:
            return
        if not isinstance(result, torch.Tensor) or result.layout != torch.strided:
            return
        # Provenance is granted only to a true fresh allocation.  ``out=`` is
        # an explicit non-fresh contract, and the storage check also covers
        # positional aliases and tensors nested in ordinary pytree containers.
        if kwargs.get("out") is not None:
            return
        result_storage = result.untyped_storage()
        argument_storages = {
            value.untyped_storage()
            for value in pytree.tree_leaves((args, kwargs))
            if isinstance(value, torch.Tensor)
        }
        if result_storage in argument_storages:
            return
        self._fresh_allocation_storages.add(result_storage)
        if _factory_fills_positive_zero(factory, args, kwargs):
            self._zero_filled_allocation_storages.add(result_storage)
        is_exact = False
        if factory in (
            torch.empty,
            torch.empty_strided,
            torch.zeros,
            torch.ones,
            torch.full,
            torch.Tensor.new_empty,
            torch.Tensor.new_empty_strided,
            torch.Tensor.new_zeros,
            torch.Tensor.new_ones,
            torch.Tensor.new_full,
        ):
            is_exact = True
        elif factory is torch.empty_like:
            like_input = args[0] if args else kwargs.get("input")
            is_exact = isinstance(
                like_input, torch.Tensor
            ) and self._prove_empty_like_layout(like_input, result)
            if not is_exact:
                # Both sizes and strides affect preserve-versus-compact and
                # dimension ordering. Bind-time hints cannot fix that choice,
                # even when empty_strided specifies its input layout exactly.
                self.mark_runtime_tensor_layout(result)
        if is_exact:
            # The fresh-storage/alias checks above ensure that this cannot
            # erase uncertainty belonging to an existing marked input/view.
            self._runtime_layout_storages.discard(result_storage)
            self._runtime_layout_stride_symbols.pop(result_storage, None)
            self._symbolically_exact_layout_storages.add(result_storage)

    def register_host_tensor_allocation(
        self, inputs: list[FakeTensor], result: object
    ) -> None:
        """Record a proven native nonaliasing result, without contents/layout facts.

        The caller establishes the native operation and exact argument contract.
        Distinct fake storage alone is insufficient for conditional-copy methods
        such as contiguous/reshape, whose runtime result can alias an input.
        """
        if type(result) is not FakeTensor or result.layout != torch.strided:
            return
        if not inputs or any(
            result.untyped_storage() == value.untyped_storage() for value in inputs
        ):
            return
        self._fresh_allocation_storages.add(result.untyped_storage())

    def runtime_value_for_tensor(self, fake_tensor: torch.Tensor) -> object | None:
        """Replay a traced tensor's input source against the current real arguments."""
        source = self.tensor_input_source(fake_tensor)
        if source is None:
            return None
        return _replay_tensor_input_source(source, self.runtime_arg_values_by_name)

    def live_example_input(self, source: Source) -> torch.Tensor | None:
        """The real tensor the input ``source`` replays to in the current call,
        or None: a direct ``to_code`` no longer finds the example input alive
        (the bound kernel holds them weakly), or the call replays a FakeTensor
        (an outer trace's input)."""
        runtime = _replay_tensor_input_source(source, self.runtime_arg_values_by_name)
        if isinstance(runtime, torch.Tensor) and not isinstance(runtime, FakeTensor):
            return runtime
        return None

    def traced_strides_hold(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor``'s traced strides are its strides in every call
        the code being generated serves, for a layout proof that reads them.

        Always for the bound kernel's own launches.  Inductor renders a
        dynamic-shape graph's kernel (``without_dispatch_guards``) from
        placeholder inputs that give each stride the graph keeps symbolic a
        stride of 1 (``lower_helion_kernel``), and the bind can trace that as
        a literal: a graph that saw ``x[:, :64]`` and then ``x[:, ::2]`` keeps
        the inner stride symbolic, its placeholder reads unit-stride, and the
        code later runs at a stride of 2 (a binding the lowering reuses from
        an earlier call traced that call's strides, which need not hold
        either).  There an input whose graph strides are all static (one not
        in ``symbolic_stride_sources``) holds, as does a view one such input
        owns, and a tensor laid out from any other input (an allocation shaped
        like it, say) holds only when no input of the graph has a symbolic
        stride.
        """
        symbolic = self.symbolic_stride_sources
        if self.dispatch_guards_enforced or not symbolic:
            return True
        owner = self.tensor_alignment_owner(tensor)
        return owner is not None and owner[0] not in symbolic

    def exact_input_metadata_holds(self) -> bool:
        """Whether the binding's exact input sizes and strides hold in every
        call the code being generated serves, so that their trace hints are
        proofs: the binding keys them (the ``input_tensor_metadata`` fact), its
        dispatch guards are enforced, and no input was bound from an outer
        trace's FakeTensor (the conditions ``keyed_hint`` checks)."""
        return (
            "input_tensor_metadata" in self.compiler_fact_specialization_facts
            and self.dispatch_guards_enforced
            and not self._binds_outer_trace_inputs
        )

    def keyed_hint(self, value: int | torch.SymInt | sympy.Expr) -> int | None:
        """The value ``value``, a traced size or stride (or an expression in
        them), has in every call this binding serves, for a value its cache
        key fixes (the caller checks the fact that does), or None.

        The trace hint is that value only while the key's dispatch guards are
        enforced (``dispatch_guards_enforced``), when no input was bound from
        an outer trace's FakeTensor (whose hints that trace chose; recorded
        at binding, so this does not depend on the FakeTensor staying alive)
        and when ``value`` has no unbacked symbol (whose hint is made up).  A
        live example input is not consulted: dispatch never routes one whose
        metadata changed in place after binding to this binding, so the
        binding decides whether or not the caller still holds it.
        """
        if not self.dispatch_guards_enforced or self._binds_outer_trace_inputs:
            return None
        if isinstance(value, int):
            return value
        expr = value._sympy_() if isinstance(value, torch.SymInt) else value
        if isinstance(expr, sympy.Integer):
            return int(expr)
        if _has_unbacked(expr):
            return None
        try:
            return int(shape_env_size_hint(self.shape_env, expr))
        except (RuntimeError, TypeError, ValueError):
            return None

    def register_runtime_input_specialization(
        self,
        key: str,
        specialization: RuntimeInputSpecialization,
    ) -> None:
        """Register an internal projection from runtime inputs to a cache-key fact."""
        previous = self.runtime_input_specializations.setdefault(key, specialization)
        if not previous.same_binding_semantics(specialization):
            raise RuntimeError(f"conflicting runtime input specializations for {key!r}")
        if (
            specialization.retains_input_identity
            and not previous.retains_input_identity
        ):
            self.runtime_input_specializations[key] = dataclasses.replace(
                previous, retains_input_identity=True
            )

    def snapshot_runtime_input_specialization_results(
        self,
        root_values: typing.Mapping[str, object],
    ) -> None:
        """Capture storage facts used by runtime-dependent code generation.

        Content-dependent classifiers are intentionally excluded. The codegen
        consumers need only facts described by ``reusable_tensor_properties``;
        filtering avoids an extra device read for worklist classifiers.
        """
        self.bound_runtime_input_specialization_results = {
            key: specialization.classifier(
                tuple(
                    _replay_tensor_input_source(source, root_values)
                    for source in specialization.sources
                )
            )
            for key, specialization in self.runtime_input_specializations.items()
            if specialization.reusable_tensor_properties
            and all(
                _is_supported_tensor_input_source(source)
                for source in specialization.sources
            )
        }

    def runtime_input_specialization_matches_bound(
        self,
        key: str,
        result: typing.Hashable,
    ) -> bool:
        """Whether a live classifier result matches this bound's cache identity."""
        return (
            key in self.bound_runtime_input_specialization_results
            and self.bound_runtime_input_specialization_results[key] == result
        )

    def tensor_descriptor_layout_signature(
        self, fake_tensor: torch.Tensor
    ) -> TensorDescriptorLayoutSignature | None:
        has_symbolic_stride = False
        for stride in fake_tensor.stride():
            if isinstance(stride, int):
                continue
            expr = _to_sympy(stride)
            expr = self.specialize_expr(self.shape_env.replace(expr))
            if expr.free_symbols:
                has_symbolic_stride = True
                break
        if has_symbolic_stride and not self.has_tensor_descriptor_layout_guard(
            fake_tensor
        ):
            return None
        return tensor_descriptor_layout_signature_from_strides(
            fake_tensor.stride(),
            fake_tensor.element_size(),
            self.size_hint,
        )

    def add_kernel_tensor_size(
        self,
        sizes: Sequence[int | torch.SymInt],
        dtype: torch.dtype | None = None,
    ) -> None:
        for size in sizes:
            if isinstance(size, torch.SymInt):
                block_idx = self.resolve_block_id(size)
                if block_idx is None:
                    value = self.specialize_expr(self.shape_env.replace(size._sympy_()))
                    if value.free_symbols and not self._is_static_kernel_shape_expr(
                        value
                    ):
                        raise exc.ShapeSpecializingAllocation
        self.kernel_tensor_sizes[(*map(_to_sympy, sizes),)] += 1
        if dtype is not None and dtype.is_floating_point:
            bits = {
                torch.float64: 64,
                torch.float32: 32,
                torch.bfloat16: 16,
                torch.float16: 16,
            }.get(dtype, 32)
            self.kernel_min_element_bits = min(self.kernel_min_element_bits, bits)

    def _is_static_kernel_shape_expr(self, expr: sympy.Expr) -> bool:
        from .host_function import HostFunction

        for symbol in expr.free_symbols:
            if not isinstance(symbol, sympy.Symbol):
                return False
            if (
                symbol in self.specialized_vars
                or symbol in self.config_value_expressions
            ):
                continue
            origin_info = HostFunction.current().expr_to_origin.get(symbol)
            if origin_info is None:
                return False
            origin = origin_info.origin
            if isinstance(origin, BlockSizeOrigin):
                continue
            if origin.is_host() and not isinstance(origin, TensorSizeOrigin):
                continue
            return False
        return True

    def finalize_config_spec(self) -> None:
        from .tile_strategy import FlattenedTileStrategy

        for shape in self.kernel_tensor_sizes:
            FlattenedTileStrategy.update_allow_flattened(shape)
        self._pin_range_num_stages_for_aliasing()
        self.config_spec._remove_duplicates()
        self.backend.adjust_block_size_constraints(
            list(self.config_spec.block_sizes),
            len(self.config_spec.block_sizes),
            block_sizes=self.block_sizes,  # pyrefly: ignore[bad-argument-type]
            kernel_tensor_sizes=self.kernel_tensor_sizes,  # pyrefly: ignore[bad-argument-type]
            min_element_bits=self.kernel_min_element_bits,
        )
        self._extract_tensor_numel_constraints()

    def _extract_tensor_numel_constraints(self) -> None:
        """Compile per-tensor numel constraints from kernel_tensor_sizes."""
        from ..autotuner.config_spec import TensorNumelConstraint

        max_numel = self.backend.max_tensor_numel
        if max_numel is None:
            # Backend (e.g. Pallas) has no compile-time per-tile element cap;
            # VMEM byte budget is enforced separately at runtime.
            return None

        uses_triton_codegen = self.codegen_name == "triton"
        cs_block_sizes = self.config_spec.block_sizes
        config_block_ids = set(cs_block_sizes.valid_block_ids())
        block_sym_to_id: dict[sympy.Symbol, int] = {}
        block_sym_to_info: dict[sympy.Symbol, BlockSizeInfo] = {}
        if uses_triton_codegen:
            for info in self.block_sizes:
                symbol = info.symbol()
                # Reused reduction dimensions deliberately share the original
                # tile symbol. Preserve the first origin rather than allowing a
                # later fixed/reduction alias to erase its tunable provenance.
                block_sym_to_info.setdefault(symbol, info)
                if info.block_id in config_block_ids:
                    block_sym_to_id.setdefault(symbol, info.block_id)
        else:
            for bs in self.block_sizes:
                block_sym_to_id[bs.symbol()] = bs.block_id

        seen_exprs: set[str] = set()
        for shape in self.kernel_tensor_sizes:
            if not shape:
                continue
            numel_expr = sympy.Mul(*shape) if len(shape) > 1 else shape[0]
            if uses_triton_codegen:
                substitutions: dict[sympy.Basic, sympy.Basic] = {}
                for symbol in numel_expr.free_symbols:
                    if (
                        not isinstance(symbol, sympy.Symbol)
                        or symbol in block_sym_to_id
                        or (info := block_sym_to_info.get(symbol)) is None
                    ):
                        continue
                    extent = self._search_invariant_extent_for_numel_constraint(info)
                    if extent is not None:
                        substitutions[symbol] = sympy.Integer(extent)
                if substitutions:
                    numel_expr = numel_expr.xreplace(substitutions)

            all_free = numel_expr.free_symbols
            involved_syms = all_free & block_sym_to_id.keys()
            if not involved_syms:
                continue
            # Skip expressions with non-block-size free symbols (e.g.,
            # runtime tensor dimensions) — they can't be evaluated at
            # config generation time. A rollable reduction remains symbolic
            # here and is skipped by the same rule.
            if all_free - block_sym_to_id.keys():
                log.debug(
                    "skipping numel constraint for shape %s: expression has "
                    "non-block-size free symbols %s",
                    shape,
                    all_free - block_sym_to_id.keys(),
                )
                continue
            try:
                sym_to_cs_idx = {
                    # pyrefly: ignore[bad-index]
                    s: cs_block_sizes.block_id_to_index(block_sym_to_id[s])
                    for s in involved_syms
                }
            except KeyError:
                log.debug(
                    "skipping numel constraint for shape %s: block_id removed "
                    "during dedup",
                    shape,
                )
                continue
            ordered = sorted(involved_syms, key=lambda s: sym_to_cs_idx[s])
            indices = tuple(sym_to_cs_idx[s] for s in ordered)
            # pyrefly: ignore[unsupported-operation]
            constraint_expr = numel_expr <= max_numel
            # srepr is more canonical than str() for dedup; a false
            # negative only causes a harmless duplicate, not a missed one.
            dedup_key = sympy.srepr(constraint_expr)
            if dedup_key in seen_exprs:
                continue
            seen_exprs.add(dedup_key)
            expr_str = str(constraint_expr)
            # pyrefly: ignore[bad-argument-type]
            check_fn = _make_numel_check(ordered, constraint_expr)
            self.config_spec.tensor_numel_constraints.append(
                TensorNumelConstraint(
                    check_fn=check_fn,
                    block_indices=indices,
                    expr_str=expr_str,
                )
            )

    def _search_invariant_extent_for_numel_constraint(
        self, block_size: BlockSizeInfo
    ) -> int | None:
        """Return an extent fixed across the generated Triton search choices."""
        source = block_size.block_size_source
        if isinstance(source, FixedBlockSizeSource):
            value = source.value
        elif isinstance(source, ReductionLoopBlockSizeSource):
            reduction_loops = self.config_spec.reduction_loops
            if block_size.block_id in reduction_loops.valid_block_ids():
                loop_spec = reduction_loops.block_id_lookup(block_size.block_id)
                if loop_spec._flat_fragment(self.config_spec).low < loop_spec.size_hint:
                    return None
            value = block_size.size
        else:
            return None
        if not isinstance(value, (int, torch.SymInt)):
            return None
        expr = self.specialize_expr(self.shape_env.replace(_to_sympy(value)))
        if expr.free_symbols or not expr.is_Integer:
            return None
        extent = int(expr)
        if isinstance(source, ReductionLoopBlockSizeSource):
            # Every fragment-generated choice is persistent, so codegen uses
            # the full backend-rounded reduction dimension.
            extent = self.backend.static_rdim_size(extent)
        return extent

    def _pin_range_num_stages_for_aliasing(self) -> None:
        """
        Pin range_num_stages to 0 on loops that read and write the same argument.

        Workaround for https://github.com/triton-lang/triton/issues/8259: such a
        loop gets no per-loop ``num_stages``.  The slot stays in the config
        space (``RangeNumStagesSpec.pinned``), so a saved config keeps its
        shape whatever the alias analysis finds.  A device loop is unsafe when
        a host tensor it reads and a host tensor it writes may share storage
        that comes from a kernel argument, by
        ``collect_host_tensor_roots`` with each argument its own root: any
        view, conversion or unknown op on an argument is assumed to alias it,
        bindings in nested host blocks count, and a tensor of unknown
        provenance may alias anything.
        """

        if not self.config_spec.range_num_stages:
            return

        from .ast_extension import ExtendedAST
        from .ast_read_writes import ReadWrites
        from .host_function import HostFunction
        from .loop_dependency_checker import HOST_UNKNOWN_ROOT
        from .loop_dependency_checker import collect_host_tensor_roots
        from .loop_dependency_checker import is_host_argument_root
        from .type_info import IterType
        from .type_info import SequenceType
        from .type_info import TensorType
        from .type_info import TileIndexType

        host_fn = HostFunction.current()
        module = ast.Module(body=host_fn.body, type_ignores=[])
        roots = collect_host_tensor_roots(
            host_fn.body,
            set(host_fn.params.arguments.keys()),
            distinct_arguments=True,
        )
        device_loops = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.For)
            and isinstance(node, ExtendedAST)
            and isinstance(node.iter, ExtendedAST)
            and isinstance(node.iter._type_info, IterType)
        ]
        # Names bound inside a device loop are tiles, not host tensors.
        device_names = {
            name.id
            for loop in device_loops
            for name in ast.walk(loop)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
        }
        tensor_names = {
            name.id
            for name in ast.walk(module)
            if isinstance(name, ast.Name)
            and isinstance(name, ExtendedAST)
            and isinstance(name._type_info, TensorType)
        } - device_names
        unknown = frozenset({HOST_UNKNOWN_ROOT})

        def argument_storage(names: set[str]) -> frozenset[str]:
            result: frozenset[str] = frozenset()
            for name in names & tensor_names:
                result |= roots.get(name, unknown)
            return result

        unsafe_block_ids: set[int] = set()
        for node in device_loops:
            rw = ReadWrites.from_list(node.body)
            # Name traversal sees the target passed to an in-place store and
            # host-side tensor metadata as reads, but neither reads storage.
            reads = argument_storage(
                {
                    name
                    for name, count in rw.reads.items()
                    if count
                    > rw.inplace_writes.get(name, 0)
                    + rw.tensor_metadata_reads.get(name, 0)
                }
                | set(rw.atomic_reads)
            )
            writes = argument_storage(set(rw.writes))
            if not (
                (reads and HOST_UNKNOWN_ROOT in writes)
                or (writes and HOST_UNKNOWN_ROOT in reads)
                or any(is_host_argument_root(root) for root in reads & writes)
            ):
                continue
            iter_node = node.iter
            assert isinstance(iter_node, ExtendedAST)
            iter_type = iter_node._type_info
            assert isinstance(iter_type, IterType)
            inner = iter_type.inner
            if isinstance(inner, SequenceType):
                unsafe_block_ids.update(
                    item.block_id
                    for item in inner.unpack()
                    if isinstance(item, TileIndexType)
                )
            elif isinstance(inner, TileIndexType):
                unsafe_block_ids.add(inner.block_id)
        for spec in self.config_spec.range_num_stages:
            if unsafe_block_ids.intersection(spec.block_ids):
                spec.pinned = True

    def allocate_block_size(
        self,
        size: int | torch.SymInt | AutoSize | None,
        *,
        reduction: bool = False,
        source: BlockSizeSource,
        hint: int = 64,
        reuse_var: torch.SymInt | None = None,
    ) -> int:
        idx = len(self.block_sizes)
        # Use the provided var or create a new one
        var = (
            reuse_var
            if reuse_var is not None
            else self.create_block_var(
                f"block_size_{idx}" if not reduction else f"rdim_{idx}",
                hint=hint,
            )
        )
        self.block_sizes.append(
            info := BlockSizeInfo(
                block_id=idx,
                size=size,
                var=var,
                reduction=reduction,
                block_size_source=source,
            )
        )
        if isinstance(source, FixedBlockSizeSource):
            if isinstance(source.value, torch.SymInt):
                source_expr = _symint_expr(source.value)
                if isinstance(source_expr, sympy.Symbol):
                    _constrain_unify(self.shape_env, source.value, info.var)
                    # Match the block var's hint to the size it is now unified
                    # with, so both agree once the shared range is narrowed.
                    shape_env_var_hints(self.shape_env)[info.symbol()] = sympy.Integer(
                        self.size_hint(source.value)
                    )
            elif source.value != 1:
                # A fixed integer extent is exact, but keep its block symbol
                # distinct so backend codegen can still track the tile axis.
                # A singleton range gives fake-tensor propagation the same
                # equality fact without replacing the symbol with the integer.
                # An extent of 1 needs no such fact (a literal 1 broadcasts
                # against the symbol anyway), and narrowing the symbol to
                # [1, 1] makes every size-1 check take the tile dim for a
                # broadcast dim and drop its block id from derived shapes
                # (``routing[tile.index + 1]`` traces as shape [1]).
                shape_env_var_hints(self.shape_env)[info.symbol()] = sympy.Integer(
                    source.value
                )
                self.shape_env.constrain_symbol_range(
                    info.symbol(), source.value, source.value
                )

        from .host_function import HostFunction
        from .host_function import SymbolOrigin

        # Only register in expr_to_origin if we created a new var
        # (otherwise the var is already registered under its original block)
        if reuse_var is None:
            HostFunction.current().expr_to_origin[info.symbol()] = SymbolOrigin(
                origin=BlockSizeOrigin(idx),
            )
        return idx

    def allocate_reduction_dimension(
        self, size: torch.SymInt | int, *, guard: bool = True
    ) -> BlockSizeInfo:
        # Check if this size is already a registered block size
        existing_block: BlockSizeInfo | None = None
        if isinstance(size, torch.SymInt):
            from .host_function import HostFunction

            expr = size._sympy_()
            origin_info = HostFunction.current().expr_to_origin.get(expr)
            if origin_info and isinstance(origin_info.origin, BlockSizeOrigin):
                block_idx = origin_info.origin.block_id
                existing_block = self.block_sizes[block_idx]

        def _has_unbacked(x: int | torch.SymInt) -> bool:
            return isinstance(x, torch.SymInt) and bool(
                free_unbacked_symbols(x._sympy_())
            )

        # Check for existing reduction dimensions with the same size. When an
        # unbacked symbol is involved, the comparison must not guard:
        # ``rdim.size == size`` on a mixed unbacked-SymInt/int pair forces a
        # ShapeEnv guard that SPECIALIZES the unbacked block symbol to the
        # rdim's concrete size (e.g. a tile_m block symbol silently becomes
        # head_dim==64 when comparing against an existing rdim), corrupting
        # every downstream shape of that tile. known_equal answers via
        # _maybe_evaluate_static (no new guards) and returns False when the
        # equality is undecidable. Backed sizes keep the guarding ``==`` so
        # distinct input symbols that are equal by hint (e.g. x.size(1) vs
        # weight.size(0)) still unify into a single rdim, and so does a dynamic
        # full slice with the dim of a specialized one (``weight[:]`` beside
        # ``y[tile, :]`` of ``n = hl.specialize(y.size(1))``: the elementwise
        # op between them needs the one dim).  A literal factory dim
        # (``guard=False``, see ``BlockSizeInfo.literal_size``) matches and is
        # matched only by sizes known equal: a padded ``hl.full([tile, 48])``
        # must not specialize a dynamic full slice that is 48 at bind time.
        for rdim in self.block_sizes:
            if not rdim.reduction or not isinstance(rdim.size, (int, torch.SymInt)):
                continue
            if (
                not guard
                or rdim.literal_size
                or _has_unbacked(rdim.size)
                or _has_unbacked(size)
            ):
                if self.known_equal(rdim.size, size):
                    if guard:
                        # A non-literal size joined it: it unifies as usual now.
                        rdim.literal_size = False
                    return rdim
            elif rdim.size == size:
                return rdim

        # Allocate a new reduction dimension
        # If size is already a block var, reuse it to maintain symbol identity
        reuse_var = existing_block.var if existing_block is not None else None
        rdim_idx = self.allocate_block_size(
            size,
            reduction=True,
            source=ReductionLoopBlockSizeSource(),
            # When size==0, next_power_of_2(size_hint(0)) == 1, and a hint of 1
            # causes Inductor to see reduction_numel==1 and skip the reduction
            # instead of generating a masked reduction that yields the identity value.
            # Use hint=2 in that case so the reduction is preserved (and a dynamic
            # empty extent does not guard its block to 1).  Test the hint: a
            # symbolic ``size == 0`` is statically False.
            hint=2
            if self.size_hint(size) == 0
            else next_power_of_2(self.size_hint(size)),
            reuse_var=reuse_var,
        )
        info = self.block_sizes[rdim_idx]
        info.literal_size = not guard
        return info

    def padded_extent_dim(self, size: int) -> int | torch.SymInt:
        """The dim of a static extent ``size`` whose lanes are padded to a
        power of two (a tensor factory's, an ``hl.arange``'s).

        A power of two is its own dim.  Any other extent is named by the
        reduction dim of its size, allocated without a guard
        (``BlockSizeInfo.literal_size``): a full slice of that size shares
        it, so a reduction over it masks the padding lanes, while a dynamic
        full slice that merely has the size at bind time keeps its own dim.
        """
        if size <= 0 or size == next_power_of_2(size):
            return size
        return self.allocate_reduction_dimension(size, guard=False).var

    def create_block_var(self, debug_name: str, hint: int = 64) -> torch.SymInt:
        source = _current_symbol_source()
        with self.shape_env.ignore_fresh_unbacked_symbols():
            sym = self.shape_env.create_unbacked_symint(source=source)
            # self.shape_env.guards.append(
            #     ShapeGuard(
            #         sympy.Ne(sym._sympy_(), 0),
            #         SLoc("create_block_var", current_location().format()),
            #         True,
            #     )
            # )
            # TODO(jansel): I was hoping the above would work, seems like some decomps require concrete values
            #               to determine zeroness.  Figure out a better way to do this.

            shape_env_var_hints(self.shape_env)[_symint_sympy_symbol(sym)] = (
                sympy.Integer(hint)
            )
        assert isinstance(sym._sympy_(), sympy.Symbol)
        self.debug_shape_renames[sym._sympy_()] = sympy.Symbol(debug_name, integer=True)
        return sym

    def create_unbacked_symint(self, hint: int = 8192) -> torch.SymInt:
        source = _current_symbol_source()
        with self.shape_env.ignore_fresh_unbacked_symbols():
            sym = self.shape_env.create_unbacked_symint(source=source)
            # TODO(jansel): this is a hack to get us past some == 1 checks
            #               we should probably have a better way to handle this
            # type: ignore [unsupported-operation]
            shape_env_var_hints(self.shape_env)[sym._sympy_()] = sympy.sympify(hint)
            return sym

    def cached_create_unbacked_symint(
        self, key: Sequence[object], hint: int = 8192
    ) -> torch.SymInt:
        """Create an unbacked symint with caching based on a key.

        This ensures that the same key always returns the same unbacked
        symint, which is crucial to allow simplification of expressions
        for things like tile_begin.

        Args:
            key: The cache key (should be sequence of hashables and unique for the desired symint)
            hint: Hint value for the symint

        Returns:
            A consistent unbacked symint for the given key
        """

        key = tuple([x._sympy_() if hasattr(x, "_sympy_") else x for x in key])
        result = self._symint_cache.get(key)
        if result is None:
            result = self.create_unbacked_symint(hint)
            self._symint_cache[key] = result
        return result

    def input_symint(self, value: int | torch.SymInt, source: Source) -> torch.SymInt:
        """Represent an input property with an independent symbolic value.

        FakeTensor may express a contiguous stride in terms of a size symbol.  A
        dynamic kernel can be reused with a different layout, so exposing that
        expression to user code would incorrectly couple the runtime stride to
        the runtime size.
        """
        result = self._input_symint_cache.get(source)
        if result is None:
            hint = self.size_hint(value)
            expr = self.shape_env.create_symbol(
                hint,
                source,
                dynamic_dim=DimDynamic.DYNAMIC,
            )
            result = typing.cast(
                "torch.SymInt",
                self.shape_env.create_symintnode(
                    expr,
                    hint=hint,
                    source=source,
                ),
            )
            self._input_symint_cache[source] = result
        return result

    def _normalize_shape_to_block_vars(
        self, shape: list[int | torch.SymInt]
    ) -> list[int | torch.SymInt]:
        """Normalize shape dimensions to use canonical block size variables."""
        return [
            self.block_sizes[self.canonical_block_id(bid)].var
            if (bid := self.resolve_block_id(s)) is not None
            else s
            for s in shape
        ]

    def should_broadcast_tensor_indexers(self, index: typing.Sequence[object]) -> bool:
        """Check whether tensor indexers need broadcasting.

        Args:
            index: The full index list (may contain torch.Tensor or TensorType)
        """
        # Import here to avoid circular import
        from .type_info import TensorType

        positions = [
            i for i, k in enumerate(index) if isinstance(k, (torch.Tensor, TensorType))
        ]
        tensors = [
            k.fake_value if isinstance(k, TensorType) else k
            for k in index
            if isinstance(k, (torch.Tensor, TensorType))
        ]

        if not tensors:
            return False
        # 1D tensors with block-size dims don't need broadcasting
        if all(
            t.ndim == 1 and self.get_block_id(t.size(0)) is not None for t in tensors
        ):
            return False
        # Single scalar or 1D tensor doesn't need broadcast handling
        if len(tensors) == 1 and tensors[0].ndim <= 1:
            return False
        # Non-consecutive tensor indexers don't broadcast together
        return len(positions) <= 1 or positions == list(
            range(positions[0], positions[-1] + 1)
        )

    def tensor_indexer_broadcast_shape(
        self, tensors: typing.Sequence[torch.Tensor]
    ) -> list[int | torch.SymInt]:
        """Compute broadcast shape for tensor indexers."""
        result: list[int | torch.SymInt] = [
            tensors[sources[0][0]].size(sources[0][1]) if sources else 1
            for sources in self.tensor_indexer_broadcast_sources(tensors)
        ]
        # Normalize the result to use canonical block size variables
        return self._normalize_shape_to_block_vars(result)

    def tensor_indexer_broadcast_sources(
        self, tensors: typing.Sequence[torch.Tensor]
    ) -> list[list[tuple[int, int]]]:
        """For each dim of ``tensor_indexer_broadcast_shape``, the
        ``(tensor, dim)`` positions of the indexer dims it runs along, the
        one that sizes it first: one dim per tensor for several 1-D indexers
        (the Cartesian case), else the dims of the tensors that align with it
        from the trailing end, leaving out those of size 1."""
        shapes = [list(t.size()) for t in tensors]
        if all(len(s) == 1 for s in shapes) and len(shapes) > 1:  # Cartesian
            return [[(i, 0)] for i in range(len(shapes))]
        max_ndim = max(len(s) for s in shapes)
        return [
            [
                (i, dim - max_ndim + len(s))
                for i, s in enumerate(shapes)
                if dim - max_ndim + len(s) >= 0
                and self.size_hint(s[dim - max_ndim + len(s)]) != 1
            ]
            for dim in range(max_ndim)
        ]

    def tensor_indexer_dims(
        self, indexer_tensor: torch.Tensor
    ) -> list[int | torch.SymInt]:
        """Return dims contributed by a tensor indexer (non-broadcast case)."""
        return [size for size, _ in self._tensor_indexer_dims(indexer_tensor)]

    def tensor_indexer_dim_sources(
        self, indexer_tensor: torch.Tensor
    ) -> list[int | None]:
        """The dim of ``indexer_tensor`` each of its ``tensor_indexer_dims``
        runs along, or None where no one dim does (a block's variable
        standing for several, or the 1 of an indexer of size-1 dims)."""
        return [dim for _, dim in self._tensor_indexer_dims(indexer_tensor)]

    def _tensor_indexer_dims(
        self, indexer_tensor: torch.Tensor
    ) -> list[tuple[int | torch.SymInt, int | None]]:
        """``tensor_indexer_dims`` with ``tensor_indexer_dim_sources``."""
        if indexer_tensor.ndim == 0:
            # Scalar tensor eliminates a dimension, contributes no output dims
            return []
        non_trivial = [
            (dim, size)
            for dim, size in enumerate(indexer_tensor.size())
            if self.size_hint(size) != 1
        ]
        # Use size-based approach to find block_id
        bid = self.resolve_block_id(non_trivial[0][1]) if non_trivial else None
        if bid is not None:
            var = self.block_sizes[self.canonical_block_id(bid)].var
            return [(var, non_trivial[0][0] if len(non_trivial) == 1 else None)]
        return [(size, dim) for dim, size in non_trivial] or [(1, None)]

    def new_index_result(
        self,
        tensor: torch.Tensor,
        output_shape: typing.Sequence[int | torch.SymInt],
        *,
        keep_static_extents: bool = False,
    ) -> torch.Tensor:
        """Create tensor for indexing ops with normalized shapes.

        Uses size-based approach to normalize all dimensions that correspond
        to block sizes to their canonical variables.  ``keep_static_extents``
        keeps each static extent of a shape of only static dims, as a
        subscript or view of a device tensor (``t[None, :]``, a gather of
        one) has its tensor's extents, rather than padding it to a power of
        two.
        """
        specialized_shape: list[int | torch.SymInt] = []
        for dim in output_shape:
            if isinstance(dim, torch.SymInt):
                expr = self.specialize_expr(_symint_sympy_expr(dim))
                if not expr.free_symbols:
                    with contextlib.suppress(TypeError, ValueError):
                        specialized_shape.append(int(expr))
                        continue
            specialized_shape.append(dim)
        # Normalize all dimensions to canonical block size variables
        shape = self._normalize_shape_to_block_vars(specialized_shape)
        static = [dim for dim in shape if isinstance(dim, int)]
        if (
            keep_static_extents
            and self.backend.pad_factory_tensors_to_power_of_2
            and len(static) == len(shape)
        ):
            # ``new_empty`` pads every dim of a shape of only static dims to a
            # power of two, but a view keeps its tensor's extents.  Name each
            # static non-power-of-2 dim (a padded ``hl.arange``'s) as
            # ``tensor_utils`` names the padded dim of a factory beside a tile,
            # by the reduction dim of its size.  Small dims too, unlike there:
            # ``hl.arange(5)`` is emitted over 8 lanes just as ``hl.arange(200)``
            # is over 256, and a reduction over it must mask them.
            count = len(self.block_sizes)
            shape = [self.padded_extent_dim(dim) for dim in static]
            self.subscript_extent_block_ids.update(range(count, len(self.block_sizes)))
            if self.typing_root_id is not None:
                for dim in shape:
                    block_id = self.get_block_id(dim)
                    if (
                        block_id is not None
                        and block_id in self.subscript_extent_block_ids
                        and self.block_sizes[block_id].literal_size
                    ):
                        self.literal_subscript_roots.setdefault(block_id, set()).add(
                            self.typing_root_id
                        )
        return tensor.new_empty(shape)

    def resolve_fragment_global(
        self, value: torch.SymInt | torch.SymFloat | torch.SymBool
    ) -> int | float | bool | None:
        """Resolve only a captured scalar or an occurrence evaluated by Python."""
        record = self.fragment_global_literals.get(id(value))
        if record is not None:
            assert record[0] is value
            return record[1]
        return None

    def record_fragment_global_operation(
        self,
        result: object,
        evaluate: typing.Callable[..., object],
        op: ast.AST,
        operands: tuple[object, ...],
    ) -> object:
        """Keep original scalar operations separate from their simplified symbols."""
        if not self.fragment_global_literals:
            return result
        values: list[int | float | bool] = []
        captured = False
        for operand in operands:
            if isinstance(operand, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                value = self.resolve_fragment_global(operand)
                if value is None:
                    return result
                captured = True
                values.append(value)
            elif type(operand) in (int, float, bool):
                values.append(typing.cast("int | float | bool", operand))
            else:
                return result
        if not captured:
            return result
        # This is optional provenance, including for untaken branches. Bound
        # Python integer work before evaluating it; checking the result after
        # pow/left-shift would be too late to prevent a large allocation.
        limit = 4096
        bits = [value.bit_length() for value in values if type(value) is int]
        if any(width > limit for width in bits):
            return result
        if len(values) == 2 and all(type(value) in (int, bool) for value in values):
            left, right = (int(value) for value in values)
            if isinstance(op, ast.Pow) and abs(left) > 1 and right >= 0:
                if right > limit // left.bit_length():
                    return result
            elif isinstance(op, ast.LShift) and right >= 0:
                if right > limit - left.bit_length():
                    return result
            elif isinstance(op, ast.Mult):
                if left.bit_length() + right.bit_length() > limit:
                    return result
        try:
            value = evaluate(op, *values)
        except (ArithmeticError, ValueError):
            # Symbolic tracing also visits untaken branches. This optional
            # provenance must not evaluate an operation that its guard excludes.
            return result
        if type(value) not in (int, float, bool):
            return result
        if isinstance(result, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            # Even an operator returning an existing wrapper/node must get its
            # own occurrence; e.g. x + 0 and x may have different zero signs.
            result = type(result)(result.node)
            self.fragment_global_literals[id(result)] = (
                result,
                typing.cast("int | float | bool", value),
            )
            return result
        # Some Torch versions fold a symbolic operation to a Python scalar.
        # All operands were proved above; keep the authored Python value.
        return value if type(result) in (int, float, bool) else result

    def container_identity(self, value: object) -> object:
        """Recover a container or declared-field holder during frontend tracing."""
        entry = self._container_identities.get(id(value))
        return entry[1].value if entry is not None else value

    def container_identity_witness(self, value: object) -> _ContainerIdentity | None:
        if not _supports_container_identity(value):
            return None
        entry = self._container_identities.get(id(value))
        if entry is not None:
            return entry[1]
        witness = _ContainerIdentity(value)
        self._container_identities[id(value)] = (value, witness)
        return witness

    def record_container_proxy(
        self, proxy: object, witness: _ContainerIdentity
    ) -> None:
        if witness.value is not None and _supports_container_identity(proxy):
            self._container_identities[id(proxy)] = (proxy, witness)

    def record_container_identity(self, proxy: object, original: object) -> None:
        witness = self.container_identity_witness(original)
        if witness is not None:
            self.record_container_proxy(proxy, witness)

    def clear_container_identities(self) -> None:
        # In particular, do not let a cached BoundKernel retain real input
        # tensors reachable through their tuple/list/dict arguments.
        for _, witness in self._container_identities.values():
            witness.value = None
        self._container_identities.clear()

    def renew_pinned_subscript_extents(self) -> None:
        """Replace each subscript extent's dim that a guard pinned to its
        padded size with a fresh dim of the same extent.

        Type propagation types a load over a range, ``hl.load(x, [i, cols])``
        in a grid kernel, with the padded power of two when no dim of the
        range's extent exists yet.  A subscript of the range typed after it,
        ``cols[None, :]``, then allocates that dim, and an op between the two
        pins the dim's var to the power of two.  Device IR tracing types the
        load again after the dim exists, so the load shares it, as when the
        subscript comes first; a fresh dim lets it do so instead of reading
        the pinned power of two.  No join record moves to the fresh dim
        (``subscript_extents_joined_in_loop``): a full slice joining the dim
        during type propagation clears its literal size, which renewal needs.
        """
        for block_id in sorted(self.subscript_extent_block_ids):
            info = self.block_sizes[block_id]
            size = info.size
            if (
                info.reduction
                and info.literal_size
                and isinstance(size, int)
                and info.var._sympy_() == next_power_of_2(size)
            ):
                info.retire_reduction()
                count = len(self.block_sizes)
                self.padded_extent_dim(size)
                self.subscript_extent_block_ids.update(
                    range(count, len(self.block_sizes))
                )

    def record_full_slice_dim(self, block_id: int) -> None:
        """Note that a full slice's dim is the reduction dim ``block_id``.

        Only a subscript extent's dim (``subscript_extent_block_ids``) is
        recorded, in ``subscript_extents_joined_by_slice``, and in
        ``subscript_extents_joined_in_loop`` when the slice is in a root loop
        whose subscripts typed the dim while its size was literal.  A
        reduction over the extent joins the dim too, but older versions gave
        the range's reads the slice's dim only.
        """
        if block_id not in self.subscript_extent_block_ids:
            return
        self.subscript_extents_joined_by_slice.add(block_id)
        if self.typing_root_id in self.literal_subscript_roots.get(block_id, ()):
            self.subscript_extents_joined_in_loop.add(block_id)

    def record_reduction_over(self, block_id: int) -> None:
        """Note a reduction over the reduction dim ``block_id``.

        Only a subscript extent's dim (``subscript_extent_block_ids``) is
        recorded: older versions could reduce over a dim of its next power
        of two instead, allocated here unless one of that size existed.
        """
        if block_id in self.subscript_extent_block_ids:
            self.subscript_extent_reductions.setdefault(block_id, len(self.block_sizes))

    def to_fake(self, obj: object, origin: Origin) -> object:
        # Importing autotuner at module scope cycles back to CompileEnvironment.
        from ..autotuner.config_fragment import ConfigSpecFragment

        if obj is None:
            return None
        if isinstance(obj, ConfigSpecFragment) and not (
            origin.is_argument() and dataclasses.is_dataclass(obj)
        ):
            if origin.is_argument():
                raise TypeError(
                    f"unsupported config fragment argument ({origin}); "
                    "construct the fragment directly inside the kernel"
                )
            # Fragments describe compile-time choices, not runtime scalars.
            # register_tunable snapshots their metadata for the bound kernel.
            return obj
        if isinstance(obj, torch.Tensor):
            return self._to_fake_tensor(obj, origin.to_source())
        if isinstance(obj, (bool, int, float)):
            if isinstance(obj, bool):
                with self.shape_env.ignore_fresh_unbacked_symbols():
                    numeric = self.shape_env.create_unbacked_symbool()
            elif isinstance(obj, int):
                # Preserve the concrete value as the initial hint so that
                # subsequent hl.specialize() calls can recover the real value
                # rather than falling back to the generic size hint.
                sym = self.create_unbacked_symint(hint=obj)
                try:
                    source = origin.to_source()
                except NotImplementedError:
                    pass
                else:
                    self.shape_env.var_to_sources[_symint_sympy_symbol(sym)] = [source]
                numeric = sym
            else:
                with self.shape_env.ignore_fresh_unbacked_symbols():
                    numeric = self.shape_env.create_unbacked_symfloat()
            if origin.is_global() and type(obj) in (int, float, bool):
                self.fragment_global_literals[id(numeric)] = (numeric, obj)
            return numeric
        if isinstance(
            obj,
            (
                torch.dtype,
                torch.device,
                types.BuiltinFunctionType,
                types.ModuleType,
                type,
            ),
        ):
            return obj
        if triton_is_available():
            from triton import JITFunction

            if isinstance(obj, JITFunction):
                return user_defined_triton_kernel_transitive_closure_source_code(obj)
        # Handle functions and Kernel objects
        from ..runtime.kernel import Kernel

        if isinstance(obj, (types.FunctionType, Kernel)) or hasattr(obj, "fn"):
            from .helper_function import extract_helper_function
            from .lift_closures import lift_closures

            # If Triton JITFunction is passed, try to unwrap to underlying Python function
            if hasattr(obj, "fn") and isinstance(obj.fn, types.FunctionType):
                fn = obj.fn
            else:
                fn = extract_helper_function(obj)
            return lift_closures(fn, origin)
        # Handle GraphModule - treat it like a function
        if isinstance(obj, torch.fx.GraphModule):
            # GraphModule can be treated like a callable function
            # We return it as-is since it will be called during execution
            return obj
        if isinstance(obj, ConstExpr):
            return obj.value
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            result = [
                self.to_fake(e, GetItemOrigin(origin, i)) for i, e in enumerate(obj)
            ]
            if type(obj) is list:
                self.record_container_identity(result, obj)
            return result
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            result = type(obj)(
                **{
                    k: self.to_fake(e, AttributeOrigin(origin, k))
                    # pyrefly: ignore [missing-attribute]
                    for k, e in obj._asdict().items()
                }
            )
            self.record_container_identity(result, obj)
            return result
        if isinstance(obj, tuple):
            result = tuple(
                self.to_fake(e, GetItemOrigin(origin, i)) for i, e in enumerate(obj)
            )
            if type(obj) is tuple:
                self.record_container_identity(result, obj)
            return result
        if isinstance(obj, dict):
            result = {
                k: self.to_fake(e, GetItemOrigin(origin, k)) for k, e in obj.items()
            }
            if type(obj) is dict:
                self.record_container_identity(result, obj)
            return result
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            # Preserve stored fields, including init=False, without rerunning
            # initialization (InitVar inputs need not be stored on the object).
            # Only the declared fields participate in typing; allocate that
            # representation without invoking a user-defined copy hook either.
            result = object.__new__(type(obj))
            for field in dataclasses.fields(obj):
                object.__setattr__(
                    result,
                    field.name,
                    self.to_fake(
                        getattr(obj, field.name), AttributeOrigin(origin, field.name)
                    ),
                )
            self.record_container_identity(result, obj)
            return result

        raise TypeError(f"unsupported argument type {type(obj)} ({origin})")

    def _maybe_recreate_symint(
        self,
        s: int | torch.SymInt,
        source: Source,
    ) -> int | torch.SymInt:
        """Create a fresh SymInt in our ShapeEnv that mirrors a foreign one."""
        if isinstance(s, int):
            return s
        outer_se = s.node.shape_env
        if outer_se is self.shape_env:
            return s
        assert outer_se is not None
        expr = s.node.expr
        assert isinstance(expr, sympy.Expr)
        cache_key = (id(outer_se), expr)
        cached = self._foreign_symint_cache.get(cache_key)
        if cached is not None:
            return cached
        if free_unbacked_symbols(expr):
            result = self.create_unbacked_symint()
        elif isinstance(expr, sympy.Symbol):
            hint = int(shape_env_var_hints(outer_se)[expr])
            new_expr = self.shape_env.create_symbol(
                hint, source, dynamic_dim=DimDynamic.DYNAMIC
            )
            result = self.shape_env.create_symintnode(
                new_expr, hint=hint, source=source
            )
        else:
            # A compound size (``s0 - 1`` for ``x[1:]``): mirror each foreign
            # symbol, then rebuild the expression over the mirrors.  ``source``
            # names this size, not its symbols, which get ephemeral sources.
            hints = shape_env_var_hints(outer_se)
            mirrors: dict[sympy.Basic, sympy.Basic] = {}
            for symbol in expr.free_symbols:
                assert isinstance(symbol, sympy.Symbol)
                mirror = self._maybe_recreate_symint(
                    outer_se.create_symintnode(symbol, hint=int(hints[symbol])),
                    EphemeralSource(f"helion_foreign_{symbol}"),
                )
                assert isinstance(mirror, torch.SymInt)
                mirrors[symbol] = mirror._sympy_()
            result = self.shape_env.create_symintnode(
                expr.xreplace(mirrors), hint=int(expr.xreplace(hints))
            )
        self._foreign_symint_cache[cache_key] = result
        return result

    def _to_fake_tensor(self, tensor: torch.Tensor, source: Source) -> torch.Tensor:
        assert CompileEnvironment.current() is self
        assert not self.fake_mode.is_our_fake(tensor)
        if isinstance(tensor, FakeTensor):
            self._binds_outer_trace_inputs = True
            # FakeTensor from an outer tracing context (e.g. make_fx, Dynamo).
            # Create fresh symbols in our own ShapeEnv to avoid leaking
            # foreign symbols whose var_to_range entries are missing,
            # which causes assertion failures in _maybe_evaluate_static
            # on PyTorch versions without optimization_hint (< 2.12).
            new_sizes = tuple(
                self._maybe_recreate_symint(
                    s,
                    TensorPropertySource(source, TensorProperty.SIZE, i),
                )
                for i, s in enumerate(tensor.size())
            )
            new_strides = tuple(
                self._maybe_recreate_symint(
                    s,
                    TensorPropertySource(source, TensorProperty.STRIDE, i),
                )
                for i, s in enumerate(tensor.stride())
            )
            result = torch.empty_strided(
                new_sizes,
                new_strides,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        elif self.settings.static_shapes:
            result = torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            result = self.fake_mode.fake_tensor_converter.from_real_tensor(
                self.fake_mode,
                tensor,
                shape_env=self.shape_env,
                source=source,
                symbolic_context=_static_dims_symbolic_context(tensor),
            )
        result = self.backend.normalize_input_fake_tensor(result)
        previous_source = self.input_sources.get(result)
        if previous_source is not None and previous_source != source:
            self._ambiguous_tensor_input_source_ids.add(id(result))
            self._tensor_input_source_cache.pop(id(result), None)
        else:
            self.input_sources[result] = source
        if isinstance(source, LocalSource):
            for i, s in enumerate(result.size()):
                if isinstance(s, torch.SymInt) and isinstance(
                    s._sympy_(), sympy.Symbol
                ):
                    self.debug_shape_renames[s._sympy_()] = sympy.Symbol(
                        f"{source.local_name}_size{i}", integer=True
                    )
        return result

    def try_concretize_symint(self, size: int | torch.SymInt) -> int | torch.SymInt:
        """Convert a SymInt to a plain int when the value is provably concrete.

        Backed SymInts (whose values are determined by input shapes) are
        concretized via their size hint.  Unbacked SymInts (e.g. block-size
        variables) are left symbolic.
        """
        if not isinstance(size, torch.SymInt):
            return size
        if _has_unbacked(size._sympy_()):
            return size
        return self.size_hint(size)

    def is_tunable_expr(self, expr: sympy.Basic) -> bool:
        """Whether ``expr`` has free symbols and all of them are tunables."""
        return bool(expr.free_symbols) and expr.free_symbols <= self.tunable_symbols

    def is_tunable_extent(self, size: int | torch.SymInt) -> bool:
        """Whether a tensor dim is sized by ``hl.register_tunable`` values alone.

        Such an extent is fixed per config but is not a block size, so a full
        slice over it needs a reduction dim for reductions, masks and loops.
        """
        return (
            isinstance(size, torch.SymInt)
            and self.is_tunable_expr(size._sympy_())
            and self.resolve_block_id(size) is None
        )

    def preserve_tunable_float_provenance(
        self, result: torch.SymFloat, operands: tuple[object, ...]
    ) -> None:
        """Keep fresh ordered-float symbols constant only for tunable-only inputs."""
        symbols: set[sympy.Basic] = set()
        for value in operands:
            if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                symbols.update(value._sympy_().free_symbols)
            elif type(value) not in (int, float, bool):
                return
        if symbols and symbols <= self.tunable_symbols:
            symbol = result._sympy_()
            assert isinstance(symbol, sympy.Symbol)
            self.tunable_symbols.add(symbol)

    def tunable_default_hint(self, expr: sympy.Basic) -> int | None:
        """``expr`` evaluated with every ``hl.register_tunable`` symbol at its default.

        A tunable-sized extent (a split-K factor sizing a workspace's
        reduction dim) is not an unknown runtime size: the kernel declared
        its range and default, and seeds size their loops over it from that
        default rather than from the generic unbacked hint of 8192. Only
        expressions whose free symbols are all integer tunables qualify; the
        result stays above one, clear of singleton special cases.
        """
        if not self.is_tunable_expr(expr):
            return None
        symbols = {
            symbol for symbol in expr.free_symbols if isinstance(symbol, sympy.Symbol)
        }
        substitutions: dict[sympy.Symbol, sympy.Integer] = {}
        for symbol in symbols:
            value = self.config_value_expressions.get(symbol)
            if (
                value is None
                or value.operation != "config"
                or len(value.arguments) != 1
            ):
                return None
            name = value.arguments[0]
            fragment = (
                self.config_spec.user_defined_tunables.get(name)
                if isinstance(name, str)
                else None
            )
            if fragment is None:
                return None
            default = fragment.default()
            if type(default) is not int:
                return None
            substitutions[symbol] = sympy.Integer(max(default, 2))
        hinted = sympy.sympify(expr.xreplace(substitutions))
        if not isinstance(hinted, sympy.Integer):
            return None
        return max(int(hinted), 2)

    def config_value_symint(self, expression: ConfigValueExpression) -> torch.SymInt:
        """The unbacked symbol of the value ``expression`` computes from the
        config, recorded in ``config_value_expressions``: one per distinct
        expression, so a helper the type pass visits again (a loop body is
        visited until its types settle) or calls twice keeps one symbol."""
        symint = self.cached_create_unbacked_symint(("config_value", expression))
        expr = _symint_expr(symint)
        assert expr is not None
        self.config_value_expressions[expr] = expression
        return symint

    def check_config_values_unspecialized(self) -> None:
        """Raise TunableSpecialized when tracing has evaluated a value the
        config sets (a tunable, or a helper over tunables) at its hint, which
        every config would then compute with: when it replaced the value's
        symbol by an integer, or guarded on it with a test some configs fail,
        as a comparison in a function defined outside the kernel or in a
        builtin the kernel calls does (``min(foo, 128)``).  This catches what
        the type pass does not
        record, such as a helper called with keyword arguments or through a
        wrapper.  A replacement by another symbol (``assert n == foo`` on the
        host) still computes with the config's value."""
        for symbol, value in self.shape_env.replacements.items():
            if symbol in self.config_value_expressions and isinstance(
                value, sympy.Integer
            ):
                raise self.tunable_specialized(
                    symbol, f", where tracing set {self._spelled(symbol)} to {value}"
                )
        for guard in self.shape_env.guards:
            symbols = {
                symbol
                for symbol in self.config_value_expressions
                if symbol in guard.expr.free_symbols
            }
            if symbols and not self._holds_for_every_config(guard, symbols):
                where = (
                    "(in a builtin the kernel calls)"
                    if _in_builtin_call(guard.sloc.framework_loc)
                    else guard.sloc
                )
                raise self.tunable_specialized(
                    min(symbols, key=str),
                    f", by the test {self._spelled(guard.expr)} {where}",
                )

    def _holds_for_every_config(
        self, guard: ShapeGuard, symbols: AbstractSet[sympy.Expr]
    ) -> bool:
        """Whether ``guard`` holds under every config, its config value
        ``symbols`` taking the values the configs give them and its other
        symbols sizes.  A guard torch or Helion records about a size holds
        where the size is at least 2, as Helion takes every symbolic size to
        be, a size a tunable sets included (``Ne(u0, 1)`` beside a dim of
        ``foo``); a comparison the kernel's code makes must hold for every
        value the tunables take (``foo == 1`` with ``foo`` from 1 to 8 does
        not)."""
        if _recorded_by_compiler(guard) and _holds_for_sizes(guard.expr):
            return True
        return self._holds_for_config_values(guard.expr, symbols)

    def _holds_for_config_values(
        self, expr: sympy.Basic, symbols: AbstractSet[sympy.Expr]
    ) -> bool:
        """Whether ``expr`` holds for every value its config value ``symbols``
        take, its other symbols sizes: by the tunables' ranges where they
        decide it, else config by config."""
        ranges = {}
        for symbol in symbols:
            bounds = self._config_value_bounds(self.config_value_expressions[symbol])
            if bounds is None:
                break
            ranges[symbol] = ValueRanges(*bounds)
        else:
            if _holds_for_sizes(expr, ranges):
                return True
        return self._holds_for_each_config(expr, symbols)

    def picked_by_every_config(
        self, values: Sequence[int | torch.SymInt], op: object
    ) -> int | None:
        """The position of the argument of ``min(*values)`` (``max`` for
        ``op``) that every config picks, when each is a config value:
        ``min(foo, 8)`` is 8 where ``foo`` takes 16 to 256.  None when the
        configs pick different ones, or another symbol occurs."""
        exprs = [_to_sympy(value) for value in values]
        if any(self.config_value_expression(expr) is None for expr in exprs):
            return None
        compare = sympy.Le if op is builtins.min else sympy.Ge
        for position, candidate in enumerate(exprs):
            relations = [compare(candidate, other) for other in exprs]
            if sympy.false not in relations and all(
                self._holds_for_config_values(
                    relation,
                    {
                        symbol
                        for symbol in relation.free_symbols
                        if symbol in self.config_value_expressions
                    },
                )
                for relation in relations
            ):
                return position
        return None

    def _config_value_bounds(
        self, expression: int | str | ConfigValueExpression
    ) -> tuple[int, int] | None:
        """The least and greatest values ``expression`` takes over the
        configs (its tunables' ranges), or None where they are not known."""
        from ..autotuner.config_fragment import BaseIntegerFragment

        if isinstance(expression, int):
            return expression, expression
        if isinstance(expression, str):
            fragment = self.config_spec.user_defined_tunables.get(expression)
            if isinstance(fragment, BaseIntegerFragment):
                return fragment.low, fragment.high
            return None
        bounds = [self._config_value_bounds(arg) for arg in expression.arguments]
        if any(bound is None for bound in bounds):
            return None
        lows = [typing.cast("tuple[int, int]", bound)[0] for bound in bounds]
        highs = [typing.cast("tuple[int, int]", bound)[1] for bound in bounds]
        operation = expression.operation
        if operation == "config":
            return lows[0], highs[0]
        if operation in ("min", "max"):
            pick = min if operation == "min" else max
            return pick(lows), pick(highs)
        if operation == "add":
            return sum(lows), sum(highs)
        if min(lows) < 0:
            # The rules below are monotone over nonnegative operands only.
            return None
        if operation == "mul":
            return math.prod(lows), math.prod(highs)
        if operation == "next_power_of_2" and lows[0] >= 1:
            return next_power_of_2(lows[0]), next_power_of_2(highs[0])
        if operation == "cdiv" and lows[1] >= 1:
            return -(-lows[0] // highs[1]), -(-highs[0] // lows[1])
        if operation == "floordiv" and lows[1] >= 1:
            return lows[0] // highs[1], highs[0] // lows[1]
        return None

    def _holds_for_each_config(
        self, expr: sympy.Basic, symbols: AbstractSet[sympy.Expr]
    ) -> bool:
        """Whether ``expr`` holds under each config of the tunables its
        config value ``symbols`` read, tried one by one, with its other
        symbols sizes.  Past ``_CONFIGS_TRIED`` configs, or over a tunable
        whose values are not listed, it does not hold."""
        from ..autotuner.config_fragment import EnumFragment
        from ..runtime.config import Config

        expressions = {
            symbol: self.config_value_expressions[symbol] for symbol in symbols
        }
        names = sorted({key for e in expressions.values() for key in e.config_keys()})
        # Every value a config may hold: an enum's choices include those its
        # search leaves out (``search_choices``), which a config can still name.
        choices = [
            (
                list(fragment.choices)
                if len(fragment.choices) <= _CONFIGS_TRIED
                else None
            )
            if isinstance(fragment, EnumFragment)
            else fragment.search_values(limit=_CONFIGS_TRIED)
            for fragment in (
                self.config_spec.user_defined_tunables[name] for name in names
            )
        ]
        if (
            any(values is None for values in choices)
            or math.prod(len(typing.cast("list[object]", values)) for values in choices)
            > _CONFIGS_TRIED
        ):
            return False
        for values in itertools.product(*typing.cast("list[list[object]]", choices)):
            config = Config.from_dict(dict(zip(names, values, strict=True)))
            try:
                substitutions = {
                    symbol: sympy.Integer(expression.evaluate(config))
                    for symbol, expression in expressions.items()
                }
            except (TypeError, ValueError, ZeroDivisionError):
                return False
            if not _holds_for_sizes(expr.xreplace(substitutions)):
                return False
        return True

    def _spelled(self, expr: sympy.Basic) -> str:
        """``expr`` with each config value spelled as the kernel computes it
        (``Ne(foo, 1)``, ``next_power_of_2(foo)``)."""
        return str(
            expr.xreplace(
                {
                    symbol: sympy.Symbol(str(expression))
                    for symbol, expression in self.config_value_expressions.items()
                }
            )
        )

    def tunable_specialized(
        self, symbol: sympy.Expr, cause: str = ""
    ) -> exc.TunableSpecialized:
        """The refusal of the config value ``symbol`` evaluated at its hint,
        naming the first tunable it reads, that tunable's hint and ``cause``,
        what evaluated it (``", by the test Ne(foo, 1) (helpers.py:3 in f)"``)."""
        name = self.config_value_expressions[symbol].config_keys()[0]
        tunable = next(
            leaf
            for leaf, expression in self.config_value_expressions.items()
            if expression == ConfigValueExpression("config", (name,))
        )
        return exc.TunableSpecialized(
            name, shape_env_size_hint(self.shape_env, tunable), cause
        )

    def config_value_expression(
        self, expr: sympy.Basic
    ) -> int | ConfigValueExpression | None:
        """``expr`` as an expression over config values: an integer, a symbol
        with a recorded ``config_value_expressions`` entry, or sums, products,
        floor divisions, minima and maxima of those (the argument ``foo * 2``
        of a host helper over a tunable ``foo``, or ``min(foo, 100)`` written
        in device code).  None when another symbol occurs."""
        if not isinstance(expr, sympy.Expr):
            return None
        if (expression := self.config_value_expressions.get(expr)) is not None:
            return expression
        if isinstance(expr, sympy.Integer):
            return int(expr)
        if isinstance(expr, sympy.Add):
            operation = "add"
        elif isinstance(expr, sympy.Mul):
            operation = "mul"
        elif isinstance(expr, FloorDiv):
            operation = "floordiv"
        elif isinstance(expr, sympy.Min):
            operation = "min"
        elif isinstance(expr, sympy.Max):
            operation = "max"
        else:
            return None
        operands: list[int | ConfigValueExpression] = []
        for arg in expr.args:
            operand = self.config_value_expression(arg)
            if operand is None:
                return None
            operands.append(operand)
        return ConfigValueExpression(operation, tuple(operands))

    def config_value(self, value: object, config: Config) -> int | None:
        """``value`` as the integer ``config`` gives it, or None when it stays
        dynamic.  Every ``hl.register_tunable`` symbol and host helper over
        them takes its value from ``config`` (``config_value_expressions``),
        and every block-size symbol its block's configured size, itself
        resolved the same way (``block_size=[bs * 2]`` over
        ``bs = hl.register_block_size(...)``); any other symbol leaves the
        value dynamic: a dynamic input size, and a grid index or a tile's
        ``begin``, ``end``, ``count`` or ``id``, which name a block too
        (``get_block_id``) but take a value per tile at run time."""
        return self._config_value(value, config, frozenset())

    def config_may_fix(self, value: object) -> bool:
        """Whether some config gives ``value`` an integer (``config_value``):
        every symbol in it is a tunable or a host helper over them, or a
        block-size symbol whose block's size a config may fix in turn."""
        return self._config_may_fix(value, frozenset())

    def _config_may_fix(self, value: object, resolving: frozenset[int]) -> bool:
        expr = self._config_expr(value)
        if expr is None or isinstance(expr, int):
            return expr is not None
        sources = self._config_sources(expr, resolving)
        if sources is None:
            return False
        for source in sources.values():
            if isinstance(source, ConfigValueExpression):
                continue
            block_size_source = self.block_sizes[source].block_size_source
            if isinstance(
                block_size_source, FixedBlockSizeSource
            ) and not self._config_may_fix(
                block_size_source.value, resolving | {source}
            ):
                return False
        return True

    @staticmethod
    def _config_expr(value: object) -> int | sympy.Basic | None:
        """``value`` as an int or a sympy expression; None for a bool or a
        value of another type."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, torch.SymInt):
            return value._sympy_()
        if isinstance(value, sympy.Basic):
            return value
        return None

    def _config_sources(
        self, expr: sympy.Basic, resolving: frozenset[int]
    ) -> dict[sympy.Symbol, ConfigValueExpression | int] | None:
        """Each symbol in ``expr`` with what gives it a value under a config:
        the expression of a tunable or a host helper over them
        (``config_value_expressions``), or the id of a block-size symbol's
        block not in ``resolving``.  None when another symbol is in it."""
        sources: dict[sympy.Symbol, ConfigValueExpression | int] = {}
        for symbol in expr.free_symbols:
            if not isinstance(symbol, sympy.Symbol):
                return None
            if (expression := self.config_value_expressions.get(symbol)) is not None:
                sources[symbol] = expression
                continue
            block_id = self._block_size_block_id(symbol)
            if block_id is None or block_id in resolving:
                return None
            sources[symbol] = block_id
        return sources

    @staticmethod
    def _block_size_block_id(symbol: sympy.Symbol) -> int | None:
        """The block whose block-size symbol ``symbol`` is (``BlockSizeOrigin``,
        not a grid index or a tile's ``begin`` / ``end`` / ``id``)."""
        from .host_function import HostFunction

        origin_info = HostFunction.current().expr_to_origin.get(symbol)
        if origin_info is not None and isinstance(origin_info.origin, BlockSizeOrigin):
            return origin_info.origin.block_id
        return None

    def _config_value(
        self, value: object, config: Config, resolving: frozenset[int]
    ) -> int | None:
        expr = self._config_expr(value)
        if expr is None or isinstance(expr, int):
            return expr
        sources = self._config_sources(expr, resolving)
        if sources is None:
            return None
        substitutions: dict[sympy.Basic, sympy.Integer] = {}
        for symbol, source in sources.items():
            if isinstance(source, ConfigValueExpression):
                size = source.evaluate(config)
            else:
                size = self._config_value(
                    self.block_sizes[source].from_config(config),
                    config,
                    resolving | {source},
                )
                if size is None:
                    return None
            substitutions[symbol] = sympy.Integer(size)
        resolved = sympy.sympify(expr.xreplace(substitutions))
        if not isinstance(resolved, sympy.Integer):
            resolved = sympy.simplify(resolved)
        return int(resolved) if isinstance(resolved, sympy.Integer) else None

    def size_hint(self, n: int | torch.SymInt) -> int:
        if isinstance(n, torch.SymInt):
            expr = n._sympy_()
            if _has_unbacked(expr):
                var_hints = shape_env_var_hints(self.shape_env)
                # For unbacked symbols, try to use the hint we stored in var_to_val
                # when creating the symint (see create_unbacked_symint).
                # This preserves the original value passed to the kernel.
                if expr in var_hints:
                    return int(var_hints[expr])
                # A min or max over config values and sizes (``min(foo,
                # 32)``) takes the value their hints give it, as Python's min
                # would.
                if isinstance(expr, (sympy.Min, sympy.Max)) and (
                    free_unbacked_symbols(expr) <= self.config_value_expressions.keys()
                ):
                    hinted = expr.xreplace(
                        {
                            symbol: var_hints[symbol]
                            for symbol in expr.free_symbols
                            if symbol in var_hints
                        }
                    )
                    if isinstance(hinted, sympy.Integer):
                        return int(hinted)
                # Fall back to default hint if not found
                return 8192

            return shape_env_size_hint(self.shape_env, n._sympy_())
        assert isinstance(n, int)
        return n

    def is_singleton_size(self, size: int | torch.SymInt) -> bool:
        """Specialize input singletons without fixing configurable tile extents.

        Backed input sizes need a binding-cache guard when collapsed. Allocation
        sizes involving unbacked tile symbols must remain valid across configs,
        even when their tracing hint happens to be one.
        """
        if not isinstance(size, torch.SymInt):
            return size == 1
        symbols = _symint_free_symbols(size)
        if _has_unbacked(size._sympy_()):
            return self.known_equal(size, 1)
        singleton = bool(size == 1)
        if singleton:
            # ShapeEnv equality guards alone are not part of BoundKernel's
            # cache key. Capture symbols before evaluating the equality, which
            # may replace the backed expression with the constant one.
            self.specialized_vars.update(symbols)
            self.singleton_vars.update(symbols)
        return singleton

    def known_equal(self, a: int | torch.SymInt, b: int | torch.SymInt) -> bool:
        if isinstance(a, torch.SymInt) or isinstance(b, torch.SymInt):
            sa = _symint_expr(a) if isinstance(a, torch.SymInt) else sympy.Integer(a)
            sb = _symint_expr(b) if isinstance(b, torch.SymInt) else sympy.Integer(b)
            if sa is None or sb is None:
                return False
            if sa == sb:
                return True
            res = self.shape_env._maybe_evaluate_static(sympy.Eq(sa, sb))
            if res is None:
                return False
            return bool(res)
        return a == b

    def resolve_broadcast_dim(
        self,
        size: int | torch.SymInt,
        block_id: int,
        offset: int | torch.SymInt = 0,
    ) -> None:
        """Settle whether a tensor dim indexed by block ``block_id`` is size 1.

        Indexing a size-1 dim with a tile over a larger extent broadcasts.  A
        dynamic size is compared with 1 under a guard unless it provably
        covers the tile's extent plus ``offset``: at size 1 such a tile is at
        most one wide there, where indexing equals broadcasting, so the
        tile's own dim (``x[tile]`` over ``x.size()``) records no guard.  A
        true comparison replaces the symbol by 1, which later
        ``known_equal(size, 1)`` checks see, and either outcome's guard,
        recorded during code generation, keys the bound kernel.
        """
        if not isinstance(size, torch.SymInt) or self.known_equal(size, 1):
            return
        if any(
            self.get_block_id(symbol) is not None
            for symbol in _to_sympy(size).free_symbols
        ):
            # A size computed from block sizes (``cdiv(n, block)`` partials)
            # is fixed by the config, not by the inputs: a guard would settle
            # it for whichever config compiled first.
            return
        extent = self.block_sizes[self.canonical_block_id(block_id)].size
        if isinstance(extent, (int, torch.SymInt)) and self.known_nonnegative(
            _to_sympy(size - extent - offset)
        ):
            return
        bool(size == 1)

    def known_nonnegative(self, expression: sympy.Expr) -> bool:
        """Prove ``expression >= 0`` without adding a specialization guard."""
        expression = self.shape_env.simplify(sympy.simplify(expression))
        if expression.is_nonnegative is True:
            return True
        if not expression.free_symbols.issubset(self.shape_env.var_to_range):
            return False
        result = self.shape_env._maybe_evaluate_static(sympy.Ge(expression, 0))
        return result is sympy.true

    def known_multiple(self, a: sympy.Expr, b: int | torch.SymInt) -> bool:
        a = self.specialize_expr(_to_sympy(a))
        b_expr = self.specialize_expr(_to_sympy(b))
        if isinstance(a, (int, sympy.Integer)) and isinstance(
            b_expr, (int, sympy.Integer)
        ):
            return (int(a) % int(b_expr)) == 0
        return False

    def specialized_multiple(self, value: object, divisor: int) -> bool:
        """Whether ``value`` (an int, SymInt or sympy expression) is a
        multiple of ``divisor`` for every input this bound kernel may see.

        A static value proves it directly.  A dynamic value is proven through
        the ``input_tensor_metadata`` specialization fact: the bound kernel is
        then keyed on the exact input sizes and strides, so the traced hint
        is the runtime value while that holds (``exact_input_metadata_holds``,
        ``keyed_hint``).  Unbacked symbols have no such hint.
        """
        if divisor <= 1:
            return True
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return value % divisor == 0
        expr: sympy.Expr
        if isinstance(value, torch.SymInt):
            expr = typing.cast("sympy.Expr", value._sympy_())
        elif isinstance(value, sympy.Expr):
            expr = value
        else:
            return False
        if isinstance(expr, sympy.Integer):
            return int(expr) % divisor == 0
        if not self.exact_input_metadata_holds():
            return False
        hint = self.keyed_hint(expr)
        return hint is not None and hint % divisor == 0

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def codegen_name(self) -> str:
        return self._backend.codegen_name

    @property
    def ptx_capability(self) -> tuple[int, int] | None:
        """Compute capability when the kernel lowers to NVIDIA PTX, else None."""
        if self.backend_name != "triton" or is_hip():
            return None
        return self.config_spec.target_device_capability

    def index_type(self) -> str:
        """Backend-specific index type string based on Settings()."""
        return self._backend.index_type_str(self.index_dtype)

    def triton_index_type(self) -> str:
        """Deprecated alias for index_type()."""
        return self.index_type()

    def sympy_debug(self, expr: sympy.Basic) -> str:
        return str(expr.xreplace(self.active_debug_shape_renames))

    def __enter__(self) -> Self:
        assert tls.env is None, "CompileEnvironment already active"
        self.fake_mode.__enter__()
        tls.env = self
        return self

    @contextlib.contextmanager
    def suspend(self) -> typing.Iterator[None]:
        """Temporarily leave this environment while compiling an owned stage."""
        assert tls.env is self
        self.__exit__(None, None, None)
        try:
            yield
        finally:
            self.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        tls.env = None
        self.fake_mode.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def current() -> CompileEnvironment:
        if (env := tls.env) is not None:
            return env
        raise NoCurrentEnvironment from None

    @staticmethod
    def has_current() -> bool:
        return tls.env is not None

    def get_block_id(self, size: int | torch.SymInt | sympy.Basic) -> int | None:
        """
        Get the block ID associated with a given size expression.

        This method determines if a size expression corresponds to a registered block size
        or grid index in the current compilation environment. It looks up the origin information of
        symbolic expressions to find their associated block IDs.

        Args:
            size: The size expression to check. Can be an integer, torch.SymInt, or sympy.Basic.

        Returns:
            The block ID if the size corresponds to a registered block size, None otherwise.
        """
        if isinstance(size, torch.SymInt):
            expr = _symint_expr(size)
            if isinstance(expr, sympy.Symbol):
                return self.get_block_id(expr)
            return None
        if isinstance(size, sympy.Symbol):
            from .host_function import HostFunction

            origin_info = HostFunction.current().expr_to_origin.get(size)
            if origin_info is not None and isinstance(
                origin_info.origin,
                BlockSizeOrigin,
            ):
                return origin_info.origin.block_id
            if origin_info is not None and isinstance(origin_info.origin, GridOrigin):
                return origin_info.origin.block_id
        return None

    def registered_extent_owner(self, block_id: int) -> int | None:
        """Return an untiled registered config axis behind an extent or its alias."""
        symbol = self.block_sizes[block_id].symbol()
        return next(
            (
                info.block_id
                for info in self.block_sizes
                if isinstance(info.size, AutoSize) and info.symbol() == symbol
            ),
            None,
        )

    def pin_unflattened_along(self, size: int | torch.SymInt) -> None:
        """For an op along a dim of extent ``size``: pin the loop of the tile
        it measures unflattened (``BlockSizeInfo.pin_unflattened``)."""
        if (block_id := self.resolve_block_id(size)) is not None:
            self.block_sizes[block_id].pin_unflattened()

    def resolve_block_id(self, size: object) -> int | None:
        """Resolve the block id carried by ``size``'s symbolic provenance."""

        if not isinstance(size, (int, torch.SymInt, sympy.Expr)):
            return None

        if isinstance(size, torch.SymInt):
            if (block_id := self.get_block_id(size)) is not None:
                return block_id
            expr = _symint_expr(size)
            if isinstance(expr, sympy.Symbol):
                from .host_function import HostFunction

                origin_info = HostFunction.current().expr_to_origin.get(expr)
                if origin_info is not None and isinstance(
                    origin_info.origin, GridOrigin
                ):
                    return origin_info.origin.block_id
            if isinstance(expr, sympy.Expr):
                expr = self.specialize_expr(expr)
                if expr is None or getattr(expr, "free_symbols", None):
                    block_id = self.get_block_id(expr)
                    if block_id is not None:
                        return block_id
                    if isinstance(expr, sympy.Symbol):
                        from .host_function import HostFunction

                        origin_info = HostFunction.current().expr_to_origin.get(expr)
                        if origin_info is not None and isinstance(
                            origin_info.origin, GridOrigin
                        ):
                            return origin_info.origin.block_id
                if (
                    expr is not None
                    and not getattr(expr, "free_symbols", None)
                    and hasattr(self, "block_sizes")
                ):
                    for info in reversed(self.block_sizes):
                        if info.reduction and info.size_matches(expr):
                            return info.block_id
            return None

        expr = _to_sympy(size)
        if isinstance(expr, sympy.Expr):
            expr = self.specialize_expr(expr)
        if expr is None or getattr(expr, "free_symbols", None):
            block_id = self.get_block_id(size)
            if block_id is not None:
                return block_id
            if isinstance(expr, sympy.Symbol):
                from .host_function import HostFunction

                origin_info = HostFunction.current().expr_to_origin.get(expr)
                if origin_info is not None and isinstance(
                    origin_info.origin, GridOrigin
                ):
                    return origin_info.origin.block_id
            return None
        if (block_id := self.get_block_id(size)) is not None:
            return block_id
        if hasattr(self, "block_sizes"):
            for info in reversed(self.block_sizes):
                if info.reduction and info.size_matches(expr):
                    return info.block_id
        return None

    def canonical_block_id(self, block_id: int) -> int:
        """Follow block-size aliases back to their canonical symbolic owner.

        Reduction lowering creates a separate output-range block even when
        that range is exactly an already-active tile block.  In that case
        ``allocate_reduction_dimension`` deliberately reuses the tile's
        symbolic variable, so preserve that identity here as well as for the
        explicit ``FixedBlockSizeSource`` aliases.
        """

        seen: set[int] = set()
        current = block_id
        while current not in seen:
            seen.add(current)
            info = self.block_sizes[current]
            source = info.block_size_source
            if isinstance(source, FixedBlockSizeSource):
                value = source.value
            elif self.backend.canonicalizes_reduction_loop_blocks() and isinstance(
                source, ReductionLoopBlockSizeSource
            ):
                value = info.size
            else:
                break
            if not isinstance(value, torch.SymInt):
                break
            value_expr = value._sympy_()
            # ``get_block_id`` intentionally prefers the newest matching
            # reduction block.  Canonicalization needs the opposite: locate
            # the earlier owner whose symbol was reused when this alias was
            # allocated.  This is also usable after HostFunction teardown.
            next_block_id = next(
                (
                    candidate.block_id
                    for candidate in self.block_sizes[:current]
                    if candidate.symbol() == value_expr
                ),
                None,
            )
            if next_block_id is None:
                next_block_id = self.get_block_id(value)
            if next_block_id is None or next_block_id == current:
                break
            current = next_block_id
        return current

    def resolve_codegen_block_id(
        self,
        block_id: int,
        codegen: object,
        graph: object | None = None,
    ) -> int:
        """Map an aliased block-size symbol back to the live loop block for codegen."""

        active_device_loops = getattr(codegen, "active_device_loops", {})
        if active_device_loops.get(block_id):
            return block_id

        canonical = self.canonical_block_id(block_id)

        def active_candidates(block_ids: list[int]) -> list[int]:
            return [
                candidate
                for candidate in block_ids
                if active_device_loops.get(candidate)
                and self.canonical_block_id(candidate) == canonical
            ]

        def pick_candidate(candidates: list[int]) -> int | None:
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            return None

        if graph is not None:
            graph_block_ids = [
                graph_info.block_ids
                for graph_info in getattr(codegen, "codegen_graphs", [])
                if hasattr(graph_info, "block_ids") and graph_info.graph is graph
            ]
            if len(graph_block_ids) == 1:
                if (
                    candidate := pick_candidate(active_candidates(graph_block_ids[0]))
                ) is not None:
                    return candidate

        if (
            candidate := pick_candidate(
                active_candidates(list(active_device_loops.keys()))
            )
        ) is not None:
            return candidate
        return block_id

    def register_jagged_tile(self, block_id: int, parent_ids: list[int]) -> None:
        self.jagged_tile_parent_ids[block_id] = parent_ids

    def is_jagged_tile(self, block_id: int) -> bool:
        return block_id in self.jagged_tile_parent_ids


class NoCurrentEnvironment(RuntimeError):
    pass


class AutoSize:
    """A marker used to delay setting the size of a block until it is known."""


@dataclasses.dataclass
class BlockSizeInfo:
    """
    Information about a block size.
    Used to track the block size for a given dimension.
    """

    block_id: int
    size: torch.SymInt | int | AutoSize | None
    var: torch.SymInt
    reduction: bool
    block_size_source: BlockSizeSource
    debug_names: set[str] = dataclasses.field(default_factory=set)
    # A reduction dim allocated for a literal tensor-factory size
    # (``allocate_reduction_dimension(..., guard=False)``): it unifies with
    # another size only when they are known equal, never by guarding a dynamic
    # input size to the literal.  Cleared once a known-equal non-literal size
    # (a full slice of a specialized dim) joins it.
    literal_size: bool = False
    # Merging can retire an axis while its stable ID must remain addressable.
    retired: bool = False

    @property
    def is_tile(self) -> bool:
        return not self.reduction and not self.retired

    def retire_reduction(self) -> None:
        assert self.reduction
        self.reduction = False
        self.retired = True

    def add_debug_name(self, name: str) -> None:
        if not name:
            return
        self.debug_names.add(name)

    @property
    def numel(self) -> sympy.Expr:
        assert isinstance(self.size, (int, torch.SymInt))
        return _to_sympy(self.size)

    def known_multiple(self, block_size: int | torch.SymInt) -> bool:
        if block_size == 1:
            return True
        if not isinstance(self.size, (int, torch.SymInt)):
            return False
        return CompileEnvironment.current().known_multiple(self.numel, block_size)

    def size_hint(self) -> int:
        size = self.size
        assert isinstance(size, (int, torch.SymInt))
        return CompileEnvironment.current().size_hint(size)

    def size_matches(self, numel: sympy.Expr | None) -> bool:
        """Check if a concrete numel value matches this block's concrete size.

        Both sides must be concrete (no free symbols) for this to return True.
        Used by resolve_block_id to match constant reduction dimensions.
        """
        if numel is None or not isinstance(self.size, (int, torch.SymInt)):
            return False
        return numel == self.numel

    def dim_matches(self, dim_symbol: sympy.Expr | None) -> bool:
        """Check if a symbolic tensor dimension corresponds to this block.

        Compares against the sympy Symbol underlying self.var, which is the
        same object as the symbol in kernel_tensor_sizes shape tuples (both
        originate from the same fake tensor .size() call during tracing).
        Used by adjust_block_size_constraints to map blocks to tensor dims.
        """
        if dim_symbol is None or not isinstance(self.size, (int, torch.SymInt)):
            return False
        return dim_symbol == _to_sympy(self.var)

    def mark_alternate_size(self, size: torch.SymInt | int | None) -> None:
        """If a block size is used with a different size, we need to clear the hint to enable masking."""
        if isinstance(self.size, AutoSize):
            # The block size was created by hl.register_block_size, and we didn't know the size yet.
            self.size = size
            if size is not None:
                env = CompileEnvironment.current()
                # Refresh the var_to_val hint to match the resolved block size
                hint = env.size_hint(size)
                shape_env_var_hints(env.shape_env)[self.symbol()] = sympy.Integer(hint)
                with contextlib.suppress(KeyError):
                    # update the size hint now that we know the size
                    env.config_spec.block_sizes.block_id_lookup(
                        self.block_id
                    ).update_hint(hint)
        elif size is None or self.size is None or self.size != size:
            self.size = None

    def symbol(self) -> sympy.Symbol:
        expr = _symint_expr(self.var)
        if isinstance(expr, sympy.Symbol):
            return expr
        return _symint_sympy_symbol(self.var)

    def from_config(self, config: Config) -> int | torch.SymInt | None:
        value = self.block_size_source.from_config(config, self)
        if isinstance(value, torch.SymInt):
            env = CompileEnvironment.current()
            if (block_id := env.get_block_id(value)) is not None:
                canonical_block_id = env.canonical_block_id(block_id)
                if canonical_block_id != self.block_id:
                    return env.block_sizes[canonical_block_id].from_config(config)
        return value

    def from_config_assert(self, config: Config) -> int | torch.SymInt:
        val = self.from_config(config)
        assert val is not None
        return val

    def is_flattened(self, config: Config) -> bool:
        spec = CompileEnvironment.current().config_spec
        return spec.flatten_loops.config_get(config.flatten_loops, self.block_id, False)

    def pin_unflattened(self) -> None:
        """Pin this tile's loop unflattened for an op along its dim
        (``pin_unflattened_loops``): where a flattened tile's dims share one
        index (``Backend.flattened_tile_shares_one_index``), none of them is a
        dim of its own."""
        env = CompileEnvironment.current()
        if self.is_tile and env.backend.flattened_tile_shares_one_index():
            env.config_spec.pin_flatten_loop(self.block_id)

    def update_min_block(self, value: int, *, allow_flattened: bool = True) -> None:
        spec = CompileEnvironment.current().config_spec
        if not allow_flattened:
            spec.flatten_loops.disable_block_id(self.block_id)
        with contextlib.suppress(KeyError):
            spec.block_sizes.block_id_lookup(self.block_id).update_min(value)

    def update_max_block(self, value: int) -> None:
        spec = CompileEnvironment.current().config_spec
        with contextlib.suppress(KeyError):
            spec.block_sizes.block_id_lookup(self.block_id).update_max(value)


class BlockSizeSource:
    def from_config(
        self, config: Config, block_size_info: BlockSizeInfo
    ) -> int | torch.SymInt | None:
        raise NotImplementedError

    def l2_grouping(self, config: Config) -> int:
        return 1


@dataclasses.dataclass
class FixedBlockSizeSource(BlockSizeSource):
    value: int | torch.SymInt
    # Whether ``value`` reads a symbol that no host variable names, a host
    # helper over tunables written inline (``inline_config_symbols``), so the
    # block varies with the config.  The type pass sets it once it knows
    # those symbols.
    varies_with_config: bool = dataclasses.field(default=False, compare=False)

    def from_config(
        self, config: Config, block_size_info: BlockSizeInfo
    ) -> int | torch.SymInt:
        if (
            self.varies_with_config
            and isinstance(self.value, torch.SymInt)
            and (expr := _symint_expr(self.value)) is not None
        ):
            # A host helper over tunables written inline in ``block_size``
            # (``helion.next_power_of_2(foo)``) has no host variable to pass
            # its value in: the block takes the value under ``config`` and is
            # static, like a literal block size.
            resolved = CompileEnvironment.current().config_constants(expr, config)
            if isinstance(resolved, sympy.Integer):
                return int(resolved)
        return self.value


@dataclasses.dataclass
class LoopSpecBlockSizeSource(BlockSizeSource):
    def from_config(self, config: Config, block_size_info: BlockSizeInfo) -> int:
        env = CompileEnvironment.current()
        size = block_size_info.size
        if isinstance(size, (int, torch.SymInt)) and env.known_equal(size, 1):
            return 1
        index = env.config_spec.block_sizes.block_id_to_index(block_size_info.block_id)
        return config.block_sizes[index]


@dataclasses.dataclass
class ReductionLoopBlockSizeSource(BlockSizeSource):
    def from_config(
        self, config: Config, block_size_info: BlockSizeInfo
    ) -> int | torch.SymInt | None:
        # ``config.reduction_loops`` holds one entry per *registered* rollable
        # dim (``config_spec.reduction_loops``), so look the dim up by block id
        # as the strategy selection does: an unregistered earlier dim would
        # shift a positional index onto another dim's chunk.
        env = CompileEnvironment.current()
        chunk = env.config_spec.reduction_loops.config_get(
            config.reduction_loops, block_size_info.block_id, None
        )
        if chunk is None:
            # Registered extents use their selected count, never the tracing
            # hint used to construct their symbolic full-slice reduction alias.
            if (
                owner := env.registered_extent_owner(block_size_info.block_id)
            ) is not None:
                return env.block_sizes[owner].from_config(config)
            size = max(1, block_size_info.size_hint())
            # Backends override static_rdim_size to control whether the
            # persistent-reduction extent is rounded up to a power of two
            # (Triton/CuTe) or kept exact (Pallas).
            return env.backend.static_rdim_size(size)
        return chunk


def warning(warning: exc.BaseWarning | type[exc.BaseWarning]) -> None:
    """Print a warning to stderr if it's not in the ignore list."""
    env = CompileEnvironment.current()
    if callable(warning):
        warning = warning()

    if not isinstance(warning, exc.BaseWarning):
        raise TypeError(f"expected BaseWarning, got {type(warning)}")

    # Check if this warning type should be ignored
    if isinstance(warning, tuple(env.settings.ignore_warnings)):
        return
    from .ast_extension import ExtendedAST

    if isinstance(warning, exc.TensorOperationInWrapper) and any(
        node._compiler_inserted for node in ExtendedAST.current()
    ):
        # A host tensor operation a backend rewrite inserted (e.g. the CuTe
        # operand staging copy), not one the user wrote.
        return
    print(f"WARNING[{type(warning).__name__}]: {warning.args[0]}", file=sys.stderr)


def _symint_sympy_expr(x: torch.SymInt) -> sympy.Expr:
    """Narrow ``x._sympy_()`` to ``sympy.Expr``.

    A ``torch.SymInt`` is always backed by a ``sympy.Expr`` at runtime.  Newer
    torch annotates ``_sympy_()`` as returning the wider ``sympy.Basic``, so this
    re-establishes the narrower type the rest of the compiler relies on.
    """
    expr = x._sympy_()
    assert isinstance(expr, sympy.Expr)
    return expr


def _symint_sympy_symbol(x: torch.SymInt) -> sympy.Symbol:
    """Narrow ``x._sympy_()`` to ``sympy.Symbol`` for sites keyed by a bare symbol."""
    sym = x._sympy_()
    assert isinstance(sym, sympy.Symbol)
    return sym


def _symint_free_symbols(x: torch.SymInt) -> set[sympy.Symbol]:
    """Return the free symbols of ``x._sympy_()`` narrowed to ``set[sympy.Symbol]``."""
    return {s for s in x._sympy_().free_symbols if isinstance(s, sympy.Symbol)}


def _to_sympy(x: int | torch.SymInt | sympy.Expr) -> sympy.Expr:
    if isinstance(x, torch.SymInt):
        return _symint_sympy_expr(x)
    if isinstance(x, int):
        return sympy.Integer(x)
    if isinstance(x, sympy.Expr):
        return x
    # type: ignore [missing-attribute]
    return sympy.sympify(x)


def compute_symbolic_min_max(
    args: tuple[int | torch.SymInt, ...], op: object
) -> torch.SymInt | int:
    env = CompileEnvironment.current()
    # Over config values, the argument every config picks is the value.
    if (position := env.picked_by_every_config(args, op)) is not None:
        return args[position]
    shape_env = env.shape_env
    sympy_op = sympy.Min if op is builtins.min else sympy.Max
    hint_fn = min if op is builtins.min else max

    expr = _to_sympy(args[0])
    hint = env.size_hint(args[0])

    for arg in args[1:]:
        rhs_expr = _to_sympy(arg)
        rhs_hint = env.size_hint(arg)
        expr = sympy_op(expr, rhs_expr)  # type: ignore[call-arg]
        hint = hint_fn(hint, rhs_hint)  # type: ignore[arg-type]

    return shape_env.create_symintnode(expr, hint=hint)  # type: ignore[return-value]


def _symint_expr(x: torch.SymInt) -> sympy.Expr | None:
    expr = getattr(getattr(x, "node", None), "_expr", None)
    if isinstance(expr, sympy.Expr):
        return expr
    with contextlib.suppress(Exception):
        return _symint_sympy_expr(x)
    return None


def _has_unbacked(expr: sympy.Basic) -> bool:
    # pyrefly: ignore [missing-attribute]
    return any(n.name.startswith("u") for n in expr.free_symbols)


def format_shape(shape: tuple[object, ...]) -> str:
    def _format_dim(dim: object) -> str:
        if isinstance(dim, torch.SymInt):
            env = CompileEnvironment.current()
            block_id = env.get_block_id(dim)
            if block_id is not None and (
                names := sorted(env.block_sizes[block_id].debug_names)
            ):
                return f"{' or '.join(names)} (symbol: {dim})"
        return str(dim)

    return "(" + ", ".join(_format_dim(d) for d in shape) + ")"
