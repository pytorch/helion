from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import functools
from inspect import getattr_static
from itertools import starmap
import math
import re
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn
from typing import TypeVar
from typing import cast
from unittest.mock import patch

import sympy
import torch
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental import proxy_tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils._pytree import tree_map_only
from torch.utils._sympy.functions import FloatTrueDiv
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import IntTrueDiv
from torch.utils._sympy.functions import ToFloat

from .. import exc
from .._utils import cdiv
from .._utils import next_power_of_2
from ..autotuner.config_fragment import _BUILTIN_TUNABLE_FRAGMENTS
from ..autotuner.config_fragment import _FRAGMENT_CHANGED_MESSAGE
from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_fragment import _plain_builtin_fragment_key
from ..autotuner.config_fragment import _same_scalar
from ..autotuner.config_spec import BlockSizeSpec
from ..autotuner.config_spec import NumThreadsSpec
from ..language._decorators import _TENSOR_METHOD_REPLACEMENTS
from ..language._decorators import get_device_func_replacement
from ..language._decorators import is_api_func
from ..language.constexpr import ConstExpr
from ..language.stack_tensor import StackTensor
from ..language.tile_proxy import Tile
from ..language.tile_proxy import _CheckForIndexCalls
from .ast_extension import ExtendedAST
from .compile_environment import AutoSize
from .compile_environment import CompileEnvironment
from .compile_environment import ConfigValueExpression
from .compile_environment import FixedBlockSizeSource
from .compile_environment import LoopSpecBlockSizeSource
from .compile_environment import RuntimeInputSpecialization
from .compile_environment import _ContainerIdentity
from .compile_environment import _symint_expr
from .compile_environment import _symint_free_symbols
from .compile_environment import warning
from .container_identity import record_opaque_container_result
from .device_function import contains_only_block_size_symbols
from .fragment_call import guard_fragment_call_input
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .host_function import SymbolOrigin
from .utils import compute_slice_size
from .utils import host_subscript_geometry
from .utils import is_basic_host_subscript
from .variable_origin import AttributeOrigin
from .variable_origin import GetItemOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import ScalarArithmeticOrigin
from .variable_origin import TensorSizeOrigin
from .variable_origin import TensorStrideOrigin
from .variable_origin import TileBeginOrigin
from .variable_origin import TileEndOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing_extensions import Self

    from torch._ops import OpOverload

    _T = TypeVar("_T")


# Ops not matching this emit a warning if they are used in a host function
regexp_allowed_host_ops: re.Pattern[str] = re.compile(
    r"like|new|broadcast|promote|view|reshape|expand|permute|strided|"
    r"transpose|contiguous|unsqueeze|squeeze|zero|rand|full|fill"
)


def _is_known_host_value(value: object) -> bool:
    """Whether Python protocols can only reach builtin or compiler proxy code.

    Do not flatten arbitrary pytrees or inspect user attributes here: those
    operations can themselves call Python that changes tensor storage.
    """
    active: set[int] = set()

    def known(item: object) -> bool:
        kind = type(item)
        if CompileEnvironment.has_current():
            original = CompileEnvironment.current().container_identity(item)
            if type(original) is not kind and not any(
                type(original) is container
                for container in (tuple, list, dict, set, frozenset, torch.Size, slice)
            ):
                # Declared-field records are represented by plain dictionaries
                # during tracing. Their runtime protocols are not dict's.
                return False
        if (
            any(
                kind is safe
                for safe in (
                    type(None),
                    bool,
                    int,
                    float,
                    complex,
                    str,
                    bytes,
                    range,
                    torch.SymInt,
                    torch.SymFloat,
                    torch.SymBool,
                    torch.dtype,
                    torch.device,
                    torch.layout,
                    torch.memory_format,
                    FakeTensor,
                )
            )
            or item is Ellipsis
        ):
            return True
        if not any(
            kind is container
            for container in (tuple, list, dict, set, frozenset, torch.Size, slice)
        ):
            return False
        identity = id(item)
        if identity in active:
            return False
        active.add(identity)
        try:
            if kind is dict:
                return all(
                    known(key) and known(value)
                    for key, value in cast("dict[object, object]", item).items()
                )
            if kind is slice:
                item = cast("slice", item)
                return all(known(part) for part in (item.start, item.stop, item.step))
            return all(known(part) for part in cast("Iterable[object]", item))
        finally:
            active.remove(identity)

    return known(value)


# Capture Python-library identities before user code can replace module attrs.
_NATIVE_TENSOR_CLASS = torch.Tensor
# The receivers of an operator lookup (``torch.ops.prims.fma``), and with
# the overload it ends in, the values such a lookup yields: torch's own
# classes, whose attribute lookup and equality run no user protocol.
_TORCH_OPS_LOOKUP_TYPES = (
    type(torch.ops),
    torch._ops._OpNamespace,
    torch._ops.OpOverloadPacket,
)
_TORCH_OPS_VALUE_TYPES = (*_TORCH_OPS_LOOKUP_TYPES, torch._ops.OpOverload)
_KNOWN_HOST_CONSTRUCTORS = (
    abs,
    all,
    any,
    bool,
    complex,
    dict,
    divmod,
    float,
    frozenset,
    len,
    list,
    min,
    max,
    int,
    pow,
    range,
    repr,
    round,
    set,
    sorted,
    str,
    sum,
    tuple,
    torch.device,
    torch.finfo,
    torch.iinfo,
    ConstExpr,
    *_BUILTIN_TUNABLE_FRAGMENTS,
)
_KNOWN_HOST_SPLITS = (torch.split, torch.Tensor.split)
_KNOWN_HOST_INT_METHODS = (int.bit_length, int.bit_count)
try:
    import triton as _host_triton
except ImportError:
    _KNOWN_HOST_SIZE_HELPERS = (cdiv, next_power_of_2)
else:
    _KNOWN_HOST_SIZE_HELPERS = (
        cdiv,
        next_power_of_2,
        _host_triton.cdiv,
        _host_triton.next_power_of_2,
    )


def _is_known_host_call(
    fn: object, args: Sequence[object], kwargs: dict[str, object]
) -> bool:
    """Recognize pure Python builtins and native effects the tracker can see."""
    if (
        any(fn is builtin for builtin in (len, bool, list, tuple))
        and len(args) == 1
        and not kwargs
    ):
        value = args[0]
        containers = (list, tuple, dict, set, frozenset, range, torch.Size)
        original = (
            CompileEnvironment.current().container_identity(value)
            if CompileEnvironment.has_current()
            else value
        )
        if any(type(value) is kind for kind in containers) and any(
            type(original) is kind for kind in containers
        ):
            # Native outer iteration/length/truth cannot invoke the elements'
            # protocols. A list containing a record is still a native list.
            return True
    if not _is_known_host_value((args, kwargs)):
        return False
    if any(fn is builtin for builtin in _KNOWN_HOST_CONSTRUCTORS):
        return True
    if any(fn is helper for helper in _KNOWN_HOST_SIZE_HELPERS):
        return True
    if any(fn is split for split in _KNOWN_HOST_SPLITS):
        return bool(args) and type(args[0]) is FakeTensor
    if type(fn) is types.MethodType:
        return fn.__func__ is _KNOWN_HOST_SPLITS[1] and type(fn.__self__) is FakeTensor
    if type(fn) is types.MethodDescriptorType:
        if any(fn is method for method in _KNOWN_HOST_INT_METHODS):
            return bool(args) and type(args[0]) is int
        return (
            bool(args)
            and type(args[0]) is FakeTensor
            and getattr(torch.Tensor, fn.__name__, None) is fn
        )
    if type(fn) is not types.BuiltinFunctionType:
        return False
    receiver = fn.__self__
    if type(receiver) is int:
        # Both sides are native builtin methods: equality cannot invoke a
        # user protocol, unlike comparing an arbitrary callable by name.
        return any(fn == method.__get__(receiver) for method in _KNOWN_HOST_INT_METHODS)
    if type(receiver) is FakeTensor:
        return (
            type(getattr(torch.Tensor, fn.__name__, None)) is types.MethodDescriptorType
        )
    if fn.__module__ == "math" and getattr(math, fn.__name__, None) is fn:
        return True
    if receiver is not None and receiver is not torch._C:
        return False
    return (
        getattr(torch._C._VariableFunctions, fn.__name__, None) is fn
        or getattr(torch._C, fn.__name__, None) is fn
    )


@contextlib.contextmanager
def _track_host_storage_effects(inputs: object, origin: Origin) -> Iterator[None]:
    """Guard existing storage proofs around host protocols and operators."""
    if CompileEnvironment.has_current() and origin.is_host():
        with CompileEnvironment.current().track_runtime_layout_mutations(
            inputs, opaque=not _is_known_host_value(inputs)
        ):
            yield
    else:
        yield


def _is_known_host_format_value(value: TypeInfo) -> bool:
    """Prove formatting cannot invoke user-defined conversion protocols."""
    if type(value) is StringType:
        return True
    if isinstance(value, (LiteralType, NumericType, TensorType)):
        return _is_known_host_value(value.proxy())
    if type(value) not in (SequenceType, DictType, SliceType):
        # ClassType's proxy is a dictionary of fields, but the runtime object
        # can have its own __format__, __str__ or __repr__ implementation.
        return False
    assert isinstance(value, CollectionType)
    if not value.contents_exact:
        return False
    if value._original_identity is not None and not _is_known_host_value(
        value._original_identity.value
    ):
        return False
    elements = value.element_types
    if isinstance(elements, dict):
        children = elements.values()
    elif isinstance(elements, slice):
        children = (elements.start, elements.stop, elements.step)
    else:
        children = elements
    return all(_is_known_host_format_value(child) for child in children)


def _has_untraced_host_protocol(value: TypeInfo, *, nested: bool = True) -> bool:
    """A structural proxy cannot stand in for a record's Python methods."""
    if not isinstance(value, CollectionType):
        return False
    if type(value) not in (SequenceType, DictType, SliceType):
        return True
    identity = value._original_identity
    if (
        identity is not None
        and identity.value is not None
        and not any(
            type(identity.value) is kind
            for kind in (list, tuple, dict, slice, torch.Size)
        )
    ):
        return True
    # An uncertain builtin collection is not a known structural record. Its
    # existing projection checks decide whether it can be read on the device;
    # preserve their error contract instead of classifying it as a custom hook.
    if not nested:
        return False
    elements = value.element_types
    if isinstance(elements, dict):
        children = elements.values()
    elif isinstance(elements, slice):
        children = (elements.start, elements.stop, elements.step)
    else:
        children = elements
    return any(_has_untraced_host_protocol(child) for child in children)


def _reject_untraced_host_protocol(operation: str | None = None) -> NoReturn:
    context = f" ({operation})" if operation is not None else ""
    raise exc.TypeInferenceError(
        f"Host operations with custom record or container protocols are unsupported{context} "
        "because tracing their structural proxies cannot track tensor mutations; "
        "pass declared fields explicitly"
    )


def _native_record_length(value: TypeInfo, *, truth: bool) -> int | None:
    """Preserve the native tuple protocols of ordinary namedtuple records."""
    if type(value) is not ClassType or value._original_identity is None:
        return None
    original = value._original_identity.value
    if (
        isinstance(original, tuple)
        and getattr_static(type(original), "__len__", None) is tuple.__len__
        and (not truth or getattr_static(type(original), "__bool__", None) is None)
    ):
        return tuple.__len__(original)
    return None


def _native_record_call(
    fn: object, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
) -> TypeInfo | None:
    if not args or kwargs or type(args[0]) is not ClassType:
        return None
    value = args[0]
    if value._original_identity is None:
        return None
    original = value._original_identity.value
    kind = type(original)
    if fn is type and len(args) == 1:
        return CallableType(origin, kind)
    if any(fn is constructor for constructor in (tuple, list)) and len(args) == 1:
        if (
            _native_record_length(value, truth=False) is not None
            and getattr_static(kind, "__iter__", None) is tuple.__iter__
            and value.contents_exact
        ):
            assert isinstance(original, tuple)
            items = tuple(tuple.__iter__(original))
            return TypeInfo.from_example(items if fn is tuple else list(items), origin)
        return None
    # Ordinary attribute lookup is needed even for isinstance: Python can
    # consult an object's custom __class__ when the concrete type differs.
    if (
        getattr_static(kind, "__getattribute__", None) is not object.__getattribute__
        or getattr_static(kind, "__getattr__", None) is not None
        or getattr_static(kind, "__class__", None) is not object.__dict__["__class__"]
    ):
        return None
    if fn is isinstance and len(args) == 2 and isinstance(args[1], LiteralType):
        classes = args[1].value
        if type(classes) is type:
            return LiteralType(origin, isinstance(original, classes))
    if (
        fn is hasattr
        and len(args) == 2
        and isinstance(args[1], LiteralType)
        and type(args[1].value) is str
        and args[1].value in value.element_types
    ):
        descriptor = getattr_static(kind, args[1].value, None)
        missing_descriptor = object()
        if (
            descriptor is None
            or type(descriptor) is types.MemberDescriptorType
            or getattr_static(type(descriptor), "__get__", missing_descriptor)
            is missing_descriptor
        ):
            # Declared fields were already read to construct this structural
            # proxy. Arbitrary missing attributes have no such dispatch guard.
            # Ordinary class defaults are values, not attribute lookup hooks.
            return LiteralType(origin, True)
    return None


def _guard_host_membership_protocol(left: TypeInfo, right: TypeInfo) -> None:
    """Membership may invoke container hooks or equality on its elements."""
    native_tuple_length = None
    if _has_untraced_host_protocol(right, nested=False):
        native_tuple_length = _native_record_length(right, truth=False)
        if native_tuple_length is None:
            _reject_untraced_host_protocol()
        assert isinstance(right, ClassType) and right._original_identity is not None
        original = right._original_identity.value
        assert isinstance(original, tuple)
        if (
            getattr_static(type(original), "__contains__", None)
            is not tuple.__contains__
            or not right.contents_exact
            or not all(_is_known_host_value(item) for item in tuple.__iter__(original))
        ):
            _reject_untraced_host_protocol()
    # Empty native sequences do not compare the query. Dictionary membership
    # still hashes it, even when empty; dictionary values are never compared.
    if native_tuple_length == 0 or (
        type(right) is SequenceType and right.contents_exact and not right.element_types
    ):
        return
    if _has_untraced_host_protocol(left) or (
        isinstance(right, SequenceType) and _has_untraced_host_protocol(right)
    ):
        _reject_untraced_host_protocol()


def _register_host_operator_allocation(inputs: object, result: object) -> None:
    """Native operators require exact arguments, without user flatten hooks."""
    if not _is_known_host_value(inputs):
        return
    tensors: list[FakeTensor] = []

    def collect(value: object) -> None:
        if type(value) is FakeTensor:
            tensors.append(value)
        elif type(value) is dict:
            for key, item in value.items():
                collect(key)
                collect(item)
        elif type(value) in (tuple, list, set, frozenset, torch.Size):
            for item in cast("Iterable[object]", value):
                collect(item)
        elif type(value) is slice:
            collect((value.start, value.stop, value.step))

    collect(inputs)
    CompileEnvironment.current().register_host_tensor_allocation(tensors, result)


@functools.cache
def _native_schema_allocates(name: str) -> bool:
    # Exclude conditional-copy/view schemas even when their traced result copied.
    schemas = [
        schema
        for schema in torch._C._jit_get_schemas_for_operator(f"aten::{name}")
        if not any(argument.is_out for argument in schema.arguments)
    ]
    return bool(schemas) and all(
        result.alias_info is None for schema in schemas for result in schema.returns
    )


def _register_native_host_allocation(
    fn: object, args: list[object], kwargs: dict[str, object], result: object
) -> None:
    receiver = None
    if type(fn) is types.MethodDescriptorType:
        if getattr(torch.Tensor, fn.__name__, None) is not fn:
            return
        # Unbound Tensor methods receive their receiver as the first argument.
        receiver = args[0] if args else None
    elif type(fn) is types.BuiltinFunctionType:
        owner = fn.__self__
        if type(owner) is FakeTensor:
            descriptor = getattr(torch.Tensor, fn.__name__, None)
            if type(descriptor) is not types.MethodDescriptorType:
                return
            receiver = owner
        elif getattr(torch._C._VariableFunctions, fn.__name__, None) is not fn:
            return
    else:
        return
    if receiver is not None and type(receiver) is not FakeTensor:
        return
    if not _native_schema_allocates(fn.__name__):
        # These native casts can alias for an unchanged dtype. A changed dtype
        # necessarily allocates; it supplies no exact-layout or zero-fill fact.
        if not (
            type(receiver) is FakeTensor
            and type(result) is FakeTensor
            and receiver.dtype != result.dtype
            and fn.__name__
            in {
                "float",
                "double",
                "half",
                "bfloat16",
                "bool",
                "byte",
                "char",
                "short",
                "int",
                "long",
                "cfloat",
                "cdouble",
                "to",
                "type_as",
            }
        ):
            return
    _register_host_operator_allocation((receiver, args, kwargs), result)


def _is_symbolic_numeric(value: object) -> bool:
    return isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)) and bool(
        value._sympy_().free_symbols
    )


def _is_discarded_host_result(origin: Origin) -> bool:
    nodes = ExtendedAST.current()
    return (
        origin.is_host()
        and len(nodes) > 1
        and isinstance(nodes[-2], ast.Expr)
        and nodes[-2].value is nodes[-1]
    )


def tile_spanned_by_slice(lower: object, upper: object) -> torch.SymInt | None:
    """The tile ``lower:upper`` spans when they are its ``begin`` and ``end``.

    Such a slice indexes what the tile does, with the tile's size; as a range
    its length would be a value only the device knows.
    """
    if not isinstance(lower, torch.SymInt) or not isinstance(upper, torch.SymInt):
        return None
    origins = HostFunction.current().expr_to_origin
    begin = origins.get(lower._sympy_())
    end = origins.get(upper._sympy_())
    if (
        begin is None
        or end is None
        or not isinstance(begin.origin, TileBeginOrigin)
        or not isinstance(end.origin, TileEndOrigin)
        or begin.origin.block_id != end.origin.block_id
    ):
        return None
    block_size = CompileEnvironment.current().block_sizes[begin.origin.block_id].var
    assert isinstance(block_size, torch.SymInt)
    return block_size


class TypeInfo:
    origin: Origin

    def __init__(self, origin: Origin) -> None:
        assert isinstance(origin, Origin)
        self.origin = origin

    @classmethod
    def from_example(cls, value: object, origin: Origin) -> TypeInfo:
        if isinstance(value, ConstExpr):
            # hl.constexpr(...) used inside a kernel body marks its argument as a
            # compile-time constant.  It is transparent to control flow and
            # arithmetic, so represent it by its wrapped value's type.  Without
            # this, ConstExpr (a NamedTuple, incl. subclasses like
            # ProcessGroupName) becomes a ClassType whose truth_value() is
            # bool(element_types) -- always True, since the wrapper always has its
            # single 'value' field -- so an `if hl.constexpr(cond):` ignores cond
            # and always takes the body.
            return cls.from_example(value.value, origin)
        if isinstance(value, torch.Tensor):
            # TODO(jansel): need to wrap this in a fake tensor
            # TODO(jansel): tensor subclass support
            return TensorType(origin, fake_value=value)
        if isinstance(value, torch.SymBool):
            return SymBoolType(origin, value)
        if isinstance(value, torch.SymInt):
            return SymIntType(origin, value)
        if isinstance(value, torch.SymFloat):
            return SymFloatType(origin, value)
        from ..language.distributed_ops import AsyncCopyDescriptor

        if isinstance(value, AsyncCopyDescriptor):
            return AsyncCopyDescriptorType(origin=origin, element_types={})
        if type(value) in (int, float, bool, type(None), range):
            return LiteralType(origin, value)
        if type(value) in (str, torch.dtype, torch.device):
            # TODO(jansel): track specializations
            return LiteralType(origin, value)
        if isinstance(value, types.ModuleType):
            return PythonModuleType(origin, value)
        if callable(value):
            # TODO(jansel): track specializations
            return CallableType(origin, value)
        if type(value) is list:
            # TODO(jansel): track specializations
            return SequenceType(
                origin,
                cls._unpack_example(enumerate(value), origin),
                original_value=value,
            )
        if type(value) is tuple:
            # TODO(jansel): track specializations
            return SequenceType(
                origin,
                tuple(cls._unpack_example(enumerate(value), origin)),
                original_value=value,
            )
        if type(value) is torch.Size:
            return cls.from_example(tuple(value), origin)
        if type(value) is dict:
            # TODO(jansel): track specializations
            if not all(type(key) in (str, int) for key in value):
                raise exc.TypeInferenceError(
                    "Only int/string keys are supported in dict"
                )
            items: list[tuple[int | str, object]] = [*value.items()]
            return DictType(
                origin,
                dict(
                    zip(value.keys(), cls._unpack_example(items, origin), strict=False)
                ),
                original_value=value,
            )
        if isinstance(value, tuple) and hasattr(type(value), "__match_args__"):
            # namedtuple or torch.return_types structseq (e.g., sort, topk)
            # __match_args__ gives field names in positional order, so
            # unpacking `vals, idx = torch.sort(x)` assigns correct types.
            field_names = list(
                type(value).__match_args__  # pyrefly: ignore [missing-attribute]
            )
            return ClassType(
                origin,
                dict(
                    zip(
                        field_names,
                        cls._unpack_example(
                            [(name, getattr(value, name)) for name in field_names],
                            origin,
                            attributes=True,
                        ),
                        strict=False,
                    )
                ),
                original_value=value,
            )
        if isinstance(value, ConfigSpecFragment):
            return ConfigFragmentType(origin, value)
        if dataclasses.is_dataclass(value):
            keys = [field.name for field in dataclasses.fields(value)]
            return ClassType(
                origin,
                dict(
                    zip(
                        keys,
                        cls._unpack_example(
                            tuple((k, getattr(value, k)) for k in keys),
                            origin,
                            attributes=True,
                        ),
                        strict=False,
                    )
                ),
                original_value=value,
            )
        if isinstance(value, zip):
            # Handle zip objects by converting to tuple of tuples
            # This allows zip to work in list comprehensions
            zipped_tuples = tuple(tuple(items) for items in value)
            return cls.from_example(zipped_tuples, origin)
        if isinstance(
            value, (torch.cuda._CudaDeviceProperties, torch.xpu._XpuDeviceProperties)
        ):
            attrs = {}
            env = CompileEnvironment.current()

            compute_unit_literal = (
                "gpu_subslice_count"
                if torch.xpu.is_available()
                else "multi_processor_count"
            )

            # Only `multi_processor_count` attribute is supported for now
            # TODO(yf225): support other torch.cuda._CudaDeviceProperties attributes
            attr_origin = AttributeOrigin(origin, compute_unit_literal)
            # Create a symbolic integer that can be passed as kernel argument
            sym = env.create_unbacked_symint()
            HostFunction.current().expr_to_origin[sym._sympy_()] = SymbolOrigin(
                origin=attr_origin
            )
            attrs[compute_unit_literal] = SymIntType(attr_origin, sym)

            # pyrefly: ignore [bad-argument-type]
            return ClassType(origin, attrs)
        raise exc.UnsupportedPythonType(type(value).__name__)

    @staticmethod
    def _unpack_example(
        values: Sequence[tuple[int | str, object]] | enumerate[object],
        origin: Origin,
        *,
        attributes: bool = False,
    ) -> list[TypeInfo]:
        result = []
        for key, value in values:
            if attributes:
                assert isinstance(key, str)
                item_origin = AttributeOrigin(origin, key)
            else:
                item_origin = GetItemOrigin(origin, key)
            result.append(TypeInfo.from_example(value, item_origin))
        return result

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return str(self)

    def debug_annotations(self) -> list[str]:
        return [f"{self!s} {self.origin!r}"]

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        """Combine two types at a join point in control flow."""
        if isinstance(other, NoType) or self == other:
            return self
        raise exc.TypeInferenceError(
            f"Can't combine types from control flow: {self!s} and {other!s}"
        )

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        raise exc.TypeInferenceError(f"{type(op).__name__} not supported on {self!s}")

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        raise exc.TypeInferenceError(f"Function calls are not supported on {self!s}")

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        raise exc.TypeInferenceError(f"Attributes are not supported on {self!s}")

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        """Should return updated type of self after running `self[key] = value`"""
        raise exc.TypeInferenceError(
            f"Subscript assignment not supported with self={self!s} key={key!s} value={value!s}"
        )

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        """Should return updated type of self after running `self[key] = value`"""
        raise exc.TypeInferenceError(
            f"Subscript not supported with self={self!s} key={key!s}"
        )

    def propagate_iter(self, origin: Origin) -> TypeInfo:
        try:
            values = self.unpack()
        except NotImplementedError:
            pass
        else:
            return functools.reduce(lambda x, y: x.merge(y), values)
        raise exc.TypeInferenceError(f"Iteration over {self!s} is not supported")

    def unpack(self) -> list[TypeInfo]:
        raise NotImplementedError

    def proxy(self) -> object:
        raise NotImplementedError

    def truth_value(self) -> bool:
        return len(self.unpack()) > 0

    def as_literal(self) -> object:
        raise NotImplementedError

    def is_literal(self) -> bool:
        try:
            self.as_literal()
        except NotImplementedError:
            return False
        return True

    def populate_symbol_origins(self, origin: Origin) -> None:
        pass

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> object:
        """Apply fn to all non-Collection TypeInfos in the tree"""
        return fn(self)

    def contains_type(self, cls: type[TypeInfo] | tuple[type[TypeInfo], ...]) -> bool:
        def visit(n: TypeInfo) -> None:
            if isinstance(n, cls):
                nonlocal found
                found = True

        found = False
        self.tree_map(visit)
        return found

    def contains_tensor(self) -> bool:
        return self.contains_type((TensorType, TensorAttributeType))


def _is_scaled_block_extent(expr: sympy.Basic) -> bool:
    """A tile width scaled only by positive integer constants.

    Preserve the relation needed by unpack/reshape/dot without exposing
    partition counts (division by a block size) to singleton specialization.
    The caller additionally requires a bounded slice and block-only symbols.
    """
    if isinstance(expr, sympy.Symbol):
        return True
    if isinstance(expr, sympy.Mul):
        coefficient, extent = expr.as_coeff_Mul()
        return (
            coefficient.is_Integer is True
            and coefficient > 0
            and coefficient != 1
            and _is_scaled_block_extent(extent)
        )
    if isinstance(expr, FloorDiv):
        extent, divisor = expr.args
        return (
            divisor.is_Integer is True
            and divisor > 0
            and _is_scaled_block_extent(extent)
        )
    return False


class TensorType(TypeInfo):
    fake_value: torch.Tensor

    def __init__(self, origin: Origin, fake_value: torch.Tensor) -> None:
        super().__init__(origin)
        if origin.is_host() and CompileEnvironment.has_current():
            env = CompileEnvironment.current()
            if env.tensor_has_runtime_layout(fake_value):
                fake_value = env.runtime_layout_tensor(fake_value)
        self.fake_value = fake_value
        if origin.is_device():
            CompileEnvironment.current().add_kernel_tensor_size(
                fake_value.size(), fake_value.dtype
            )

    def __str__(self) -> str:
        env = CompileEnvironment.current()
        shape: list[str] = []
        for s in self.fake_value.size():
            if isinstance(s, torch.SymInt):
                shape.append(env.sympy_debug(s._sympy_()))
            else:
                shape.append(str(s))
        dtype = self.fake_value.dtype
        return f"{type(self).__name__}([{', '.join(shape)}], {dtype})"

    def proxy(self) -> torch.Tensor:
        return self.fake_value

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        if origin.is_host():
            warning(exc.TensorOperationInWrapper)
        if isinstance(op, ast.Not):
            # Host code runs as Python, where PyTorch raises this itself.
            if origin.is_device() and tensor_may_have_many_elements(self.fake_value):
                raise exc.TypeInferenceError(
                    f"Boolean value of {self!s} with more than one value is ambiguous"
                )
            return SymBoolType.new_unbacked(origin)
        try:
            with _track_host_storage_effects(self.fake_value, origin):
                result = _eval_unary(op, self.fake_value)
            if origin.is_host():
                CompileEnvironment.current().propagate_runtime_tensor_layout(
                    self.fake_value, result
                )
                _register_host_operator_allocation(self.fake_value, result)
            return TypeInfo.from_example(result, origin)
        except exc.Base:
            raise
        except Exception as e:
            raise exc.TorchOpTracingError(e) from e

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        assert origin.key == attr
        if attr in {"dtype", "device", "ndim", "shape", "T"}:
            return TypeInfo.from_example(getattr(self.fake_value, attr), origin)
        return TensorAttributeType(origin, self)

    def _device_indexing_size(self, key: TypeInfo) -> list[int | torch.SymInt]:
        if isinstance(key, SequenceType):
            keys = key.unpack()
        else:
            keys = [key]
        inputs_consumed = 0
        output_sizes: list[int | torch.SymInt] = []
        env = CompileEnvironment.current()
        tensor_indexers = [k.fake_value for k in keys if isinstance(k, TensorType)]
        should_broadcast = env.should_broadcast_tensor_indexers(keys)
        for k in keys:
            if isinstance(k, LiteralType):
                if isinstance(k.value, (int, torch.SymInt)):
                    inputs_consumed += 1
                elif k.value is None:
                    output_sizes.append(1)
                else:
                    raise exc.InvalidIndexingType(k)
            elif isinstance(k, SymIntType):
                inputs_consumed += 1
            elif isinstance(k, SliceType):
                # Handle slices - including those with steps
                slice_obj = k.proxy()
                size = self.fake_value.size(inputs_consumed)
                inputs_consumed += 1

                tile = tile_spanned_by_slice(slice_obj.start, slice_obj.stop)
                if tile is not None and slice_obj.step in (None, 1):
                    output_sizes.append(tile)
                    continue

                # For slices with steps, we need to calculate the output size differently
                output_size = compute_slice_size(slice_obj, size)

                if self.origin.is_device():
                    output_sizes.append(output_size)
                elif not env.is_singleton_size(output_size):
                    # Explicit tile windows must retain their block-size
                    # algebra (e.g. packed loads followed by reshape/dot).
                    # Full/stepped allocation slices and partition counts keep
                    # a reduction variable: their expression can have hint 1
                    # while another config needs several elements, so exposing
                    # it directly can specialize contiguity to a singleton.
                    if isinstance(output_size, torch.SymInt):
                        expr = output_size._sympy_()
                        if (
                            isinstance(expr, sympy.Expr)
                            and expr.free_symbols
                            and contains_only_block_size_symbols(expr)
                            and (
                                isinstance(expr, sympy.Symbol)
                                or (
                                    slice_obj.stop is not None
                                    and _is_scaled_block_extent(expr)
                                )
                            )
                        ):
                            output_sizes.append(output_size)
                            continue
                    # On backends that don't pad factory ops to power-of-2,
                    # concrete dims must stay concrete so subsequent shape
                    # inference can prove equality with other concretely-sized
                    # buffers (e.g. host-allocated accumulators via hl.zeros).
                    # Allocating a reduction-dim block here would introduce a
                    # fresh unbacked symbol that does not unify with the int
                    # even when the hint matches.  A tunable extent has no
                    # int to unify with and needs a reduction dim like below.
                    if (
                        not env.backend.pad_factory_tensors_to_power_of_2
                        and not env.is_tunable_extent(output_size)
                    ):
                        output_sizes.append(env.try_concretize_symint(output_size))
                        continue
                    rdim = CompileEnvironment.current().allocate_reduction_dimension(
                        output_size
                    )
                    output_sizes.append(rdim.var)
                else:
                    output_sizes.append(1)
            elif isinstance(k, TileIndexType):
                inputs_consumed += 1
                output_sizes.append(env.block_sizes[k.block_id].var)
            elif isinstance(k, TensorType) and k.fake_value.dtype == torch.bool:
                raise exc.DataDependentOutputShapeNotSupported(
                    op_desc="Boolean mask indexing (tensor[boolean_mask])"
                )
            elif isinstance(k, TensorType):
                inputs_consumed += 1
                if not should_broadcast:
                    output_sizes.extend(env.tensor_indexer_dims(k.fake_value))
                elif k.fake_value is tensor_indexers[0]:
                    output_sizes.extend(
                        env.tensor_indexer_broadcast_shape(tensor_indexers)
                    )
            elif k.contains_type(TileIndexType):
                # Unwrap single-element containers so hl.tile([m]) works
                # with multi-dim indexing (e.g. x[tile, :]).
                if (
                    isinstance(k, SequenceType)
                    and len(k.element_types) == 1
                    and isinstance(k.element_types[0], TileIndexType)
                ):
                    inputs_consumed += 1
                    output_sizes.append(
                        env.block_sizes[k.element_types[0].block_id].var
                    )
                else:
                    raise exc.OverpackedTile(k)
            else:
                raise exc.InvalidIndexingType(k)
        if inputs_consumed != self.fake_value.ndim:
            raise exc.RankMismatch(
                self.fake_value.ndim,
                inputs_consumed,
                f"tensor shape: {tuple(self.fake_value.shape)}",
            )
        return output_sizes

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if origin.is_host():
            warning(exc.TensorOperationInWrapper)
        else:
            lhs_shape = self._device_indexing_size(key)
            lhs_rank = len(lhs_shape)
            if isinstance(value, TensorType):
                rhs_rank = value.fake_value.ndim
                # Allow scalar tensors (rank 0) to be assigned to any rank (broadcasts)
                if rhs_rank != 0 and lhs_rank != rhs_rank:
                    env = CompileEnvironment.current()
                    can_squeeze = rhs_rank > lhs_rank and all(
                        env.known_equal(value.fake_value.size(d), 1)
                        for d in range(rhs_rank - lhs_rank)
                    )
                    if not can_squeeze:
                        raise exc.RankMismatch(
                            lhs_rank,
                            rhs_rank,
                            f"LHS shape: {tuple(lhs_shape)}, RHS shape: {tuple(value.fake_value.shape)}",
                        )
            elif isinstance(value, (NumericType, LiteralType)):
                # Allow scalar assignment to tensor (broadcasts to tensor shape)
                pass
            else:
                raise exc.RequiresTensorInAssignment(value)
        return self

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        if origin.is_host():
            try:
                # Suppress shape guards to prevent symbolic variables from
                # being specialized to concrete values during type inference.
                env = CompileEnvironment.current()
                with env.shape_env.suppress_guards():
                    index = key.proxy()
                    with _track_host_storage_effects((self.fake_value, index), origin):
                        if (
                            isinstance(index, (tuple, list))
                            and sum(item is Ellipsis for item in index) > 1
                        ):
                            raise exc.TypeInferenceError(
                                "Host tensor indexing supports at most one ellipsis"
                            )

                        def example_index(item: object) -> object:
                            if isinstance(item, slice) and isinstance(
                                item.step, torch.SymInt
                            ):
                                # aten.slice takes a concrete step. Passing the
                                # original SymInt would specialize it through
                                # __index__; only its example execution needs a hint.
                                return slice(
                                    item.start, item.stop, env.size_hint(item.step)
                                )
                            return item

                        example = index
                        if is_basic_host_subscript(index):
                            example = (
                                tuple(map(example_index, index))
                                if isinstance(index, tuple)
                                else example_index(index)
                            )
                        # pyrefly: ignore [bad-index]
                        result = self.fake_value[example]
                        geometry = host_subscript_geometry(self.fake_value, index)
                        if geometry is not None:
                            with torch.no_grad():
                                result = torch.as_strided(result, *geometry)
                        # Advanced indexing can allocate instead of sharing the
                        # source storage, so its result needs the same provenance.
                        env.propagate_runtime_tensor_layout(
                            (self.fake_value, index), result
                        )
                        return TypeInfo.from_example(result, origin)
            except NotImplementedError:
                raise exc.TypeInferenceError(
                    f"Subscript not supported on {self!s} with key={key!s}"
                ) from None
        new_sizes = self._device_indexing_size(key)
        env = CompileEnvironment.current()
        new_fake = env.new_index_result(self.fake_value, new_sizes)

        return TensorType(origin, new_fake)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, TensorType):
            if self.fake_value is other.fake_value:
                return self
            if self.fake_value.device != other.fake_value.device:
                raise exc.ControlFlowTensorMismatch(
                    var=var_name,
                    details=f"device {self.fake_value.device} != {other.fake_value.device}",
                )
            if self.fake_value.dtype != other.fake_value.dtype:
                raise exc.ControlFlowTensorMismatch(
                    var=var_name,
                    details=f"dtype {self.fake_value.dtype} != {other.fake_value.dtype}",
                )
            if self.fake_value.dim() != other.fake_value.dim():
                raise exc.ControlFlowTensorMismatch(
                    var=var_name,
                    details=f"rank {self.fake_value.dim()} != {other.fake_value.dim()}",
                )
            shape = self.fake_value.size()
            same_shape = shape == other.fake_value.size()
            if not same_shape:
                # A static size and the reduction dimension allocated for it
                # later (e.g. by a ``:`` slice) name the same dim; compare the
                # shapes in their block-variable form.
                env = CompileEnvironment.current()
                shape = env._normalize_shape_to_block_vars([*shape])
                if shape != env._normalize_shape_to_block_vars(
                    [*other.fake_value.size()]
                ):
                    raise exc.ControlFlowTensorMismatch(
                        var=var_name,
                        details=f"size {self.fake_value.size()} != {other.fake_value.size()}",
                    )
            host = other.origin.is_host()
            env = CompileEnvironment.current()
            common_strides = False
            if host:
                common_strides = all(
                    starmap(
                        env.known_equal,
                        zip(
                            self.fake_value.stride(),
                            other.fake_value.stride(),
                            strict=True,
                        ),
                    )
                )
            if common_strides:
                # empty_like compacts non-dense layouts. Preserve a proven
                # common stride tuple on synthetic fake storage, without
                # selecting either branch's alias or origin as the join.
                merged = torch.empty_strided(
                    shape,
                    self.fake_value.stride(),
                    dtype=self.fake_value.dtype,
                    device=self.fake_value.device,
                )
            elif same_shape:
                merged = torch.empty_like(self.fake_value)
            else:
                merged = self.fake_value.new_empty(shape)
            if host:
                env.register_tensor_storage_join(
                    merged, (self.fake_value, other.fake_value)
                )
                if not common_strides:
                    # A host branch may select different layouts even when
                    # both inputs have statically known shapes and strides.
                    env.mark_runtime_tensor_layout(merged)
                env.propagate_runtime_tensor_layout(
                    (self.fake_value, other.fake_value), merged
                )
            return TensorType(other.origin, merged)
        return super().merge(other, var_name=var_name)

    def populate_symbol_origins(self, origin: Origin) -> None:
        shape_env = CompileEnvironment.current().shape_env
        expr_to_origin = HostFunction.current().expr_to_origin
        tensor_to_origin = HostFunction.current().tensor_to_origin
        if (
            self.fake_value not in tensor_to_origin
            or origin.depth() < tensor_to_origin[self.fake_value].depth()
        ):
            tensor_to_origin[self.fake_value] = origin
        for i, size in enumerate(self.fake_value.size()):
            if isinstance(size, torch.SymInt):
                expr = shape_env.simplify(size._sympy_())
                sub_origin = TensorSizeOrigin(origin, i)
                if (
                    expr not in expr_to_origin
                    or sub_origin.depth() < expr_to_origin[expr].depth()
                ):
                    expr_to_origin[expr] = SymbolOrigin(
                        sub_origin, fake_value=self.fake_value
                    )
        for i, stride in enumerate(self.fake_value.stride()):
            if isinstance(stride, torch.SymInt):
                expr = shape_env.simplify(stride._sympy_())
                sub_origin = TensorStrideOrigin(origin, i)
                if (
                    expr not in expr_to_origin
                    or sub_origin.depth() < expr_to_origin[expr].depth()
                ):
                    expr_to_origin[expr] = SymbolOrigin(
                        sub_origin, fake_value=self.fake_value
                    )


class TensorAttributeType(TypeInfo):
    # pyrefly: ignore [bad-override]
    origin: AttributeOrigin
    tensor: TensorType

    def __init__(self, origin: AttributeOrigin, tensor: TensorType) -> None:
        super().__init__(origin)
        self.tensor = tensor

    def attr(self) -> str:
        return self.origin.key

    def proxy(self) -> object:
        return getattr(self.tensor.proxy(), self.attr())

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, TensorAttributeType) and self.attr() == other.attr():
            combined_tensor = self.tensor.merge(other.tensor, var_name=var_name)
            if isinstance(combined_tensor, TensorType):
                return TensorAttributeType(self.origin, combined_tensor)
        return super().merge(other, var_name=var_name)

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        attr = self.attr()
        if (
            origin.is_device()
            and attr in _TENSOR_METHOD_REPLACEMENTS
            and (
                replacement := get_device_func_replacement(getattr(torch.Tensor, attr))
            )
        ):
            result = CallableType(origin, replacement).propagate_call(
                (self.tensor, *args), kwargs, origin
            )
            assert result is not None
            return result
        if attr in {"dim", "ndimension"} and not (args or kwargs):
            return TypeInfo.from_example(self.tensor.fake_value.ndim, origin)
        stride_dim_kwarg = attr == "stride" and not args and set(kwargs) == {"dim"}
        if (attr in {"shape", "size", "stride"} and not kwargs) or stride_dim_kwarg:
            fn = getattr(self.tensor.fake_value, attr)
            try:
                literal_args = [x.as_literal() for x in args]
                literal_kwargs = {
                    key: value.as_literal() for key, value in kwargs.items()
                }
                with _track_host_storage_effects(
                    (self.tensor.fake_value, literal_args, literal_kwargs), origin
                ):
                    result = fn(*literal_args, **literal_kwargs)
            except NotImplementedError:
                raise exc.TypeInferenceError(
                    f"Tensor.{attr}() args must be literals"
                ) from None
            if attr == "stride":

                def stride_type(value: int | torch.SymInt, dim: int) -> TypeInfo:
                    stride_origin = TensorStrideOrigin(self.tensor.origin, dim)
                    if not CompileEnvironment.current().settings.static_shapes:
                        try:
                            source = stride_origin.to_source()
                        except NotImplementedError:
                            # Host-created layouts can also vary between
                            # calls. Their stride is replayed by the host,
                            # rather than extracted from a kernel argument.
                            value = CompileEnvironment.current().cached_create_unbacked_symint(
                                ("host_tensor_stride", id(self.tensor.fake_value), dim),
                                CompileEnvironment.current().size_hint(value),
                            )
                        else:
                            value = CompileEnvironment.current().input_symint(
                                value, source
                            )
                    return TypeInfo.from_example(value, stride_origin)

                if literal_args or literal_kwargs:
                    dim = cast(
                        "int",
                        literal_args[0] if literal_args else literal_kwargs["dim"],
                    )
                    if dim < 0:
                        dim += self.tensor.fake_value.ndim
                    return stride_type(result, dim)
                return SequenceType(
                    origin,
                    tuple(
                        stride_type(stride, dim) for dim, stride in enumerate(result)
                    ),
                )
            return TypeInfo.from_example(result, origin)
        if attr == "item" and not (args or kwargs):
            if origin.is_device():
                raise exc.NotAllowedOnDevice("Tensor.item()")
            if self.tensor.fake_value.numel() != 1:
                raise exc.TypeInferenceError("Tensor.item() requires numel() == 1")
            dtype = self.tensor.fake_value.dtype
            if dtype.is_complex:
                raise exc.TypeInferenceError("Complex tensors not supported")
            if dtype.is_floating_point:
                return SymFloatType.new_unbacked(origin)
            if dtype == torch.bool:
                return SymBoolType.new_unbacked(origin)
            return SymIntType.new_unbacked(origin)

        proxy_args = [x.tree_map(_to_proxy) for x in args]
        proxy_kwargs = {k: v.tree_map(_to_proxy) for k, v in kwargs.items()}
        try:
            fn = getattr(self.tensor.fake_value, attr)
            with (
                CompileEnvironment.current().track_runtime_layout_mutations(
                    (self.tensor.fake_value, proxy_args, proxy_kwargs),
                    opaque=not _is_known_host_call(fn, proxy_args, proxy_kwargs),
                )
                if origin.is_host()
                else contextlib.nullcontext()
            ):
                result = _CheckForIndexCalls.retry_call(fn, proxy_args, proxy_kwargs)
            if origin.is_host():
                # ``Tensor.new_*`` allocate like the ``torch.*`` factories;
                # the receiver is the method's first argument.
                CompileEnvironment.current().propagate_runtime_tensor_layout(
                    (self.tensor.fake_value, proxy_args, proxy_kwargs), result
                )
                CompileEnvironment.current().register_tensor_factory_layout(
                    getattr(torch.Tensor, attr, None),
                    (self.tensor.fake_value, *proxy_args),
                    proxy_kwargs,
                    result,
                )
                _register_native_host_allocation(fn, proxy_args, proxy_kwargs, result)
            output_type = TypeInfo.from_example(result, origin)
        except exc.Base:
            raise
        except Exception as e:
            # TODO(jansel): point to other tracing modes
            raise exc.TorchOpTracingError(e) from e
        if origin.is_host() and output_type.contains_tensor():
            if not regexp_allowed_host_ops.search(attr):
                warning(exc.TensorOperationInWrapper(attr))
        return output_type


class LiteralType(TypeInfo):
    value: object

    def __init__(self, origin: Origin, value: object) -> None:
        super().__init__(origin)
        self.value = value

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    @property
    def python_type(self) -> type[object]:
        return type(self.value)

    def proxy(self) -> object:
        return self.value

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        with _track_host_storage_effects(self.value, origin):
            result = _eval_unary(op, self.value)
        return TypeInfo.from_example(result, origin)

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        if type(self.value) is types.ModuleType and attr in vars(self.value):
            # An existing exact-module entry cannot call module __getattr__.
            result = vars(self.value)[attr]
        elif (
            self.value is _NATIVE_TENSOR_CLASS
            and type(descriptor := getattr_static(self.value, attr, None))
            is types.MethodDescriptorType
        ):
            # Native Tensor method lookup does not invoke a user descriptor.
            # Python replacements still go through opaque effect tracking.
            result = descriptor
        elif any(type(self.value) is kind for kind in _TORCH_OPS_LOOKUP_TYPES):
            # ``torch.ops.<namespace>.<op>.<overload>`` resolves a registered
            # operator: torch's own lookup, which runs no user protocol and
            # touches no tensor, so the storage proofs survive it.
            result = getattr(self.value, attr)
        else:
            with _track_host_storage_effects(self.value, origin):
                result = getattr(self.value, attr)
        return TypeInfo.from_example(result, origin)

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        try:
            literal_key = key.as_literal()
            with _track_host_storage_effects((self.value, literal_key), origin):
                # pyrefly: ignore [bad-index]
                result = self.value[literal_key]
            return TypeInfo.from_example(result, origin)
        except NotImplementedError:
            pass
        return super().propagate_getitem(key, origin)

    def truth_value(self) -> bool:
        with _track_host_storage_effects(self.value, self.origin):
            return bool(self.value)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        compared = other.value if isinstance(other, LiteralType) else None
        # Native function/module/Tensor-class equality has no user hook. Keep
        # them forbidden as call arguments, where they could be callbacks.
        inputs = tuple(
            value
            for value in (self.value, compared)
            if value is not _NATIVE_TENSOR_CLASS
            and not any(
                type(value) is kind
                for kind in (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    types.MethodDescriptorType,
                    types.ModuleType,
                    type,
                    *_TORCH_OPS_VALUE_TYPES,
                )
            )
        )
        with _track_host_storage_effects(inputs, self.origin):
            if type(other) is type(self) and (
                self.value is _NATIVE_TENSOR_CLASS
                and other.value is _NATIVE_TENSOR_CLASS
                or self.value == other.value
            ):
                if (
                    type(self.value) in (bool, int, float)
                    and type(other.value) in (bool, int, float)
                    and not _same_scalar(self.value, other.value)
                ):
                    # Python equality alone cannot prove a declared domain across
                    # branches: zero signs and equal int/float/bool values differ.
                    # Keep the runtime branch valid without certifying either arm.
                    kind = (
                        float if float in (self.python_type, other.python_type) else int
                    )
                    return NumericType.subtype(kind).new_unbacked(self.origin)
                return self
            if isinstance(other, (LiteralType, NumericType)):
                if NumericType.known_equal(other.value, self.value):
                    if isinstance(other, NumericType):
                        captured = CompileEnvironment.current().resolve_fragment_global(
                            other.value
                        )
                        if (
                            captured is None
                            and other.python_type is float
                            or captured is not None
                            and not _same_scalar(captured, self.value)
                        ):
                            # A symbolic float can have rounded differently from
                            # its simplified expression. Do not let a literal arm
                            # certify the other runtime branch's domain value.
                            return NumericType.subtype(other.python_type).new_unbacked(
                                self.origin
                            )
                    return self
                if self.python_type == other.python_type and self.python_type in (
                    int,
                    float,
                    bool,
                ):
                    # pyrefly: ignore [bad-argument-type]
                    return NumericType.subtype(self.python_type).new_unbacked(
                        self.origin
                    )
            return super().merge(other, var_name=var_name)

    def unpack(self) -> list[TypeInfo]:
        try:
            with _track_host_storage_effects(self.value, self.origin):
                # pyrefly: ignore [no-matching-overload]
                values = list(iter(self.value))
        except TypeError:
            return super().unpack()
        return [TypeInfo.from_example(x, self.origin) for x in values]

    def as_literal(self) -> object:
        return self.value


class StringType(TypeInfo):
    """TypeInfo for unknown strings (e.g., from f-strings)."""

    def __str__(self) -> str:
        return "str"


def _iter_body_excluding_nested_defs(node: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _iter_body_excluding_nested_defs(child)


class NestedFunctionType(TypeInfo):
    func_node: ast.FunctionDef

    def __init__(
        self,
        origin: Origin,
        func_node: ast.FunctionDef,
    ) -> None:
        super().__init__(origin)
        self.func_node = func_node

    def __str__(self) -> str:
        return f"NestedFunctionType({self.func_node.name})"

    def find_effective_return(self) -> ast.Return | None:
        func_node = self.func_node
        for stmt in func_node.body:
            for desc in _iter_body_excluding_nested_defs(stmt):
                if isinstance(desc, ast.Return):
                    raise exc.StatementNotSupported(
                        f"Early return inside control flow is not supported "
                        f"in nested function '{func_node.name}'"
                    )
        for stmt in func_node.body:
            if isinstance(stmt, ast.Return):
                return stmt
        return None


class ConfigFragmentType(LiteralType):
    """TypeInfo for config fragments are treated as constant literals during compilation."""

    # pyrefly: ignore [bad-override]
    value: ConfigSpecFragment

    def __init__(self, origin: Origin, fragment: ConfigSpecFragment) -> None:
        assert isinstance(fragment, ConfigSpecFragment)
        super().__init__(origin, fragment)
        self._captured_domain = self._revisit_domain()

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        with _track_host_storage_effects(self.value, origin):
            value = getattr(self.value, attr)
        if (
            self.origin.is_global()
            and dataclasses.is_dataclass(self.value)
            and any(field.name == attr for field in dataclasses.fields(self.value))
        ):
            # A captured declaration is literal metadata, but ordinary reads of
            # its dataclass fields retain the original runtime scalar behavior.
            value = CompileEnvironment.current().to_fake(value, origin)
        return TypeInfo.from_example(value, origin)

    def _revisit_domain(self) -> tuple[object, ...] | None:
        # These two built-ins compare by identity, but their complete domain
        # can be snapshotted immutably. The original fields remain mutable and
        # are rechecked on merge. Exclude subclasses and opaque search state.
        value = self.value

        def resolve(field: object) -> object:
            if isinstance(field, torch.SymInt):
                # A helper can store a traced literal global before registration
                # normalizes its snapshot. Use the same exact recorded value for
                # this private key, without mutating the original or using hints.
                return CompileEnvironment.current().resolve_fragment_global(field)
            return field

        key = _plain_builtin_fragment_key(
            value, resolve=resolve, for_identity_equality=True
        )
        return (type(value), key) if key is not None else None

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        for value in (self, other):
            if (
                isinstance(value, ConfigFragmentType)
                and value._captured_domain is not None
                and value._captured_domain != value._revisit_domain()
            ):
                # An identical object may have changed since the first visit.
                # LiteralType's identity/equality fallback must not erase that.
                raise exc.TypeInferenceError(_FRAGMENT_CHANGED_MESSAGE)
        if (
            isinstance(other, ConfigFragmentType)
            and self._captured_domain is not None
            and self._captured_domain == other._captured_domain
        ):
            return self
        return super().merge(other, var_name=var_name)


def _starts_symmetric_allocation(values: Sequence[object]) -> bool:
    """Runtime guard: get_remote_tensors views start at the allocation base."""
    (local,) = values
    if cast("torch.Tensor", local).storage_offset() != 0:
        raise exc.InvalidAPIUsage(
            "get_remote_tensors views start at the symmetric allocation, so the "
            "tensor passed to it must too"
        )
    return True


def _offset_expr(tensor: torch.Tensor) -> object:
    """A tensor's storage offset as an int or a sympy expression."""
    offset = tensor.storage_offset()
    return offset if isinstance(offset, int) else _symint_expr(offset)


class _PeerViewCapture(TorchDispatchMode):
    """Record get_remote_tensors views, also when a host helper calls it."""

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.symm_mem.get_remote_tensors.default:
            local, group_name = args
            env = CompileEnvironment.current()
            storage = local.untyped_storage()
            owners = [t for t in env.input_sources if t.untyped_storage() == storage]
            source = env.tensor_input_source(owners[0]) if len(owners) == 1 else None
            # Peer views start at the allocation base, as does the guarded owner;
            # static fakes drop input offsets, symbolic ones are exact.
            mapped = (
                group_name == env.process_group_name
                and source is not None
                and _offset_expr(local) in (0, _offset_expr(owners[0]))
                and local.is_contiguous()
            )
            if mapped:
                env.register_runtime_input_specialization(
                    f"symmetric_allocation_base:{source!r}",
                    RuntimeInputSpecialization(
                        sources=(source,),
                        classifier_identity="symmetric_allocation_base",
                        classifier=_starts_symmetric_allocation,
                        reusable_tensor_properties=frozenset(
                            ("data_ptr", "storage_span")
                        ),
                    ),
                )
            HostFunction.current().compiler_state.peer_views.update(
                (
                    peer.untyped_storage(),
                    (local.untyped_storage(), rank) if mapped else None,
                )
                for rank, peer in enumerate(result)
            )
        return result


def _config_value_names(value: object) -> list[str]:
    """The tunables the config values ``value`` reads derive from
    (``config_value_expressions``), sorted."""
    if not isinstance(value, torch.SymInt):
        return []
    expressions = CompileEnvironment.current().config_value_expressions
    pending = [
        expressions[symbol]
        for symbol in value._sympy_().free_symbols
        if symbol in expressions
    ]
    names: set[str] = set()
    while pending:
        for argument in pending.pop().arguments:
            if isinstance(argument, ConfigValueExpression):
                pending.append(argument)
            elif isinstance(argument, str):
                names.add(argument)
    return sorted(names)


class CallableType(LiteralType):
    # pyrefly: ignore [bad-override]
    value: Callable[..., object]

    def __init__(self, origin: Origin, value: Callable[..., object]) -> None:
        super().__init__(origin, value)
        self.value = value

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name})"

    @property
    def name(self) -> str:
        try:
            return self.value.__qualname__
        except AttributeError:
            try:
                return self.value.__name__
            except AttributeError:
                # A callable object: name it by its type. ``str()`` can span
                # lines (an fx.GraphModule prints its code), and the name is
                # printed into one-line type annotations.
                return type(self.value).__name__

    # pyrefly: ignore [bad-override]
    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo | None:
        if self.value is breakpoint:
            # special handling to prevent breakpoint() from being called during host-code type propagation
            return LiteralType(origin, None)
        if self.value in (torch.nonzero, torch.Tensor.nonzero) and origin.is_device():
            raise exc.DataDependentOutputShapeNotSupported(op_desc="torch.nonzero")
        if self.value in (torch.split, torch.Tensor.split) and origin.is_device():
            raise exc.UnsupportedSplitOperation(op="torch.split")
        if (
            self.value in (torch.tensor_split, torch.Tensor.tensor_split)
            and origin.is_device()
        ):
            raise exc.UnsupportedSplitOperation(op="torch.tensor_split")
        if origin.is_device() and self._config_value_operation() == "next_power_of_2":
            # Device code would run the helper on the hints of the tunables it
            # reads (8192), not on each config's values; the host records it
            # per config.
            for arg in [*args, *kwargs.values()]:
                if isinstance(arg, SymIntType) and (
                    names := _config_value_names(arg.value)
                ):
                    raise exc.NotAllowedOnDevice(
                        f"next_power_of_2 over the tunable {names[0]!r}, which "
                        "device code would compute at the tunable's hint whatever "
                        "the config (compute it before the loop),"
                    )
        if is_api_func(fn := self.value):
            if fn._is_device_only and origin.is_host():
                raise exc.DeviceAPIOnHost(fn.__qualname__)
            if fn._cache_type and ExtendedAST.current()[-1]._type_info is not None:
                return ExtendedAST.current()[-1]._type_info
            assert fn._type_function is not None
            return fn._type_function(*args, **kwargs, origin=origin)
        if origin.is_host():
            native_record_result = _native_record_call(self.value, args, kwargs, origin)
            if native_record_result is not None:
                return native_record_result
            if (
                any(self.value is fn for fn in (len, bool))
                and len(args) == 1
                and not kwargs
            ):
                length = _native_record_length(args[0], truth=self.value is bool)
                if length is not None:
                    return LiteralType(
                        origin, bool(length) if self.value is bool else length
                    )
            if any(self.value is fn for fn in (str, repr, ascii, format)):
                if any(
                    not _is_known_host_format_value(arg)
                    for arg in (*args, *kwargs.values())
                ):
                    _reject_untraced_host_protocol(
                        self.value.__name__
                        if type(self.value)
                        in (types.BuiltinFunctionType, types.FunctionType)
                        else "host call"
                    )
            else:
                # These operations inspect only the outer collection. In
                # particular, len([record]) need not invoke record.__bool__.
                # Other calls can invoke arbitrary record protocols, including
                # str.format or a Python helper that performs the conversion.
                nested = not any(self.value is fn for fn in (len, bool, list, tuple))
                if any(
                    _has_untraced_host_protocol(arg, nested=nested)
                    for arg in (*args, *kwargs.values())
                ):
                    _reject_untraced_host_protocol(
                        self.value.__name__
                        if type(self.value)
                        in (types.BuiltinFunctionType, types.FunctionType)
                        else "host call"
                    )
        if self.value is slice:
            if kwargs or not (1 <= len(args) <= 3):
                raise exc.TypeInferenceError(
                    "slice() expects 1 to 3 positional arguments"
                )
            none = LiteralType(origin, None)
            if len(args) == 1:
                elements = slice(none, args[0], none)
            elif len(args) == 2:
                elements = slice(args[0], args[1], none)
            else:
                elements = slice(args[0], args[1], args[2])
            return SliceType(origin, elements)
        if (
            any(self.value is builtin for builtin in (len, bool))
            and len(args) == 1
            and not kwargs
            and isinstance(args[0], CollectionType)
            and type(args[0]) in (SequenceType, DictType)
            and args[0].contains_approximate_contents()
        ):
            collection = args[0]
            if not collection.contents_exact:
                return (
                    SymIntType.new_unbacked(origin)
                    if self.value is len
                    else SymBoolType.new_unbacked(origin)
                )
            if collection.length_projection_safe:
                # Filtered children do not change the length of a proved
                # outer builtin collection. Do not inspect their proxies.
                return LiteralType(origin, self.value(collection.element_types))
        # TODO(jansel): add no-tracing mode

        def warn_wrong_device(arg: TypeInfo) -> None:
            if (
                isinstance(arg, TensorType)
                and arg.fake_value.device.type != env.device.type
            ):
                warning(exc.WrongDevice(self.value, arg.fake_value.device, env.device))

        fragment_constructor = isinstance(self.value, type) and issubclass(
            self.value, ConfigSpecFragment
        )

        def to_proxy(arg: TypeInfo) -> object:
            if isinstance(arg, TensorType):
                nonlocal input_contains_tensor
                input_contains_tensor = True
                warn_wrong_device(arg)
            value = _to_proxy(arg)
            if fragment_constructor and isinstance(
                value, (torch.SymInt, torch.SymFloat, torch.SymBool)
            ):
                literal = env.resolve_fragment_global(value)
                if literal is not None:
                    return literal
            return value

        input_contains_tensor: bool = False
        env: CompileEnvironment = CompileEnvironment.current()
        proxy_args = [x.tree_map(to_proxy) for x in args]
        proxy_kwargs = {k: v.tree_map(to_proxy) for k, v in kwargs.items()}

        if (
            origin.is_host()
            and self.value is int
            and len(proxy_args) == 1
            and not proxy_kwargs
            and _is_size_backed_float(proxy_args[0])
        ):
            # Python int(SymFloat) guards and specializes its current hint.
            # Keep truncation symbolic so a reused binding computes its size.
            return TypeInfo.from_example(
                torch.sym_int(cast("torch.SymFloat", proxy_args[0])), origin
            )

        if fragment_constructor and any(
            _is_symbolic_numeric(value)
            for value in tree_leaves((proxy_args, proxy_kwargs))
        ):
            env.warn_symbolic_fragment_domain()

        # special handling for symint arguments
        if any(
            isinstance(x, torch.SymInt) and _is_symbolic_numeric(x) for x in proxy_args
        ):
            if self.value in self._new_symint_on_host_fns() and origin.is_host():
                result = SymIntType.new_unbacked(origin)
                operation = self._config_value_operation()
                derived_args: list[int | str | ConfigValueExpression] = []
                for arg in proxy_args:
                    if isinstance(arg, int):
                        derived_args.append(arg)
                    elif isinstance(arg, torch.SymInt):
                        expr = _symint_expr(arg)
                        if expr is None:
                            break
                        expression = env.config_value_expressions.get(expr)
                        if expression is None:
                            break
                        derived_args.append(expression)
                    else:
                        break
                else:
                    result_expr = _symint_expr(result.value)
                    if (
                        operation is not None
                        and not proxy_kwargs
                        and result_expr is not None
                    ):
                        env.config_value_expressions[result_expr] = (
                            ConfigValueExpression(operation, tuple(derived_args))
                        )
                return result
        try:
            with (
                patch.object(torch.SymInt, "__index__", _raise_shape_specializing),
                _PeerViewCapture()
                if env.process_group_name is not None
                else contextlib.nullcontext(),
            ):
                # Exit before constructing result provenance: a resized input
                # must already have its runtime strides if the result aliases it.
                with (
                    env.track_runtime_layout_mutations(
                        (proxy_args, proxy_kwargs),
                        opaque=not _is_known_host_call(
                            self.value, proxy_args, proxy_kwargs
                        ),
                    )
                    if origin.is_host()
                    else contextlib.nullcontext()
                ):
                    result = _CheckForIndexCalls.retry_call(
                        self.value, proxy_args, proxy_kwargs
                    )
                if origin.is_host():
                    env.propagate_runtime_tensor_layout(
                        (proxy_args, proxy_kwargs), result
                    )
                    env.register_tensor_factory_layout(
                        self.value,
                        proxy_args,
                        proxy_kwargs,
                        result,
                    )
                    _register_native_host_allocation(
                        self.value, proxy_args, proxy_kwargs, result
                    )
                output_origin = origin
                if (
                    self.value is tuple
                    and len(proxy_args) == 1
                    and not proxy_kwargs
                    and type(result) is tuple
                    and result is proxy_args[0]
                ):
                    # tuple(exact_tuple) preserves the operand's identity and
                    # provenance, including a capture aliased by this input.
                    output_origin = args[0].origin
                output_type = TypeInfo.from_example(result, output_origin)
                if any(
                    isinstance(arg, CollectionType)
                    and arg.contains_approximate_contents()
                    for arg in (*args, *kwargs.values())
                ):
                    if isinstance(output_type, CollectionType):
                        # Host calls execute with representative proxy elements;
                        # their result cannot establish exact collection contents.
                        output_type.mark_approximate_contents(
                            length_projection_safe=False
                        )
                        output_type.length_projection_safe = any(
                            self.value is builtin
                            for builtin in (list, tuple, dict, sorted)
                        )
                    else:
                        numeric_type = (
                            type(output_type)
                            if isinstance(output_type, NumericType)
                            else _numeric_types.get(type(result))
                        )
                        if _is_discarded_host_result(origin):
                            # The original host call still executes. Its unused
                            # representative result cannot affect device code.
                            output_type = NoType(origin)
                        elif not origin.is_host() or numeric_type is None:
                            raise exc.TracedArgNotSupported(
                                "a non-numeric result from representative collection contents"
                            )
                        # The host computes this scalar from the actual collection.
                        # A fresh symbol passes that value instead of specializing
                        # the representative result or its symbolic expression.
                        else:
                            output_type = numeric_type.new_unbacked(origin)
                if not any(self.value is cls for cls in _BUILTIN_TUNABLE_FRAGMENTS):
                    if not any(self.value is fn for fn in (tuple, list, dict)):
                        record_opaque_container_result(result)
                    guard_fragment_call_input(self.origin)

                    def guard_input(arg: TypeInfo) -> TypeInfo:
                        if isinstance(arg, (ConfigFragmentType, CallableType)):
                            guard_fragment_call_input(arg.origin)
                        return arg

                    for arg in (*args, *kwargs.values()):
                        arg.tree_map(guard_input)
            output_type.tree_map(warn_wrong_device)
            if (
                origin.is_host()
                and input_contains_tensor
                and output_type.contains_tensor()
            ):
                if getattr(self.value, "__module__", "").startswith(
                    "torch"
                ) and not regexp_allowed_host_ops.search(self.name):
                    warning(exc.TensorOperationInWrapper(self.name))
            return output_type
        except exc.ShapeSpecializingCall:
            if origin.is_host() and not input_contains_tensor:
                proxy_args, proxy_kwargs = tree_map_only(
                    torch.SymInt, env.size_hint, (proxy_args, proxy_kwargs)
                )
                with env.track_runtime_layout_mutations(
                    (proxy_args, proxy_kwargs),
                    opaque=not _is_known_host_call(
                        self.value, proxy_args, proxy_kwargs
                    ),
                ):
                    example = self.value(*proxy_args, **proxy_kwargs)
                # We can handle many functions like math.sqrt by introducing unbacked values
                if isinstance(example, bool):
                    return SymBoolType.new_unbacked(origin)
                if isinstance(example, int):
                    return SymIntType.new_unbacked(origin)
                if isinstance(example, float):
                    return SymFloatType.new_unbacked(origin)
            raise
        except exc.Base:
            raise
        except Exception as e:
            # TODO(jansel): point to other tracing modes
            raise exc.TorchOpTracingError(e) from e

    @staticmethod
    @functools.cache
    def _new_symint_on_host_fns() -> dict[object, None]:
        """Functions that should return a new unbacked symint when called on host with a symint argument."""
        from .._utils import cdiv
        from .._utils import next_power_of_2

        fns: list[object] = [cdiv, next_power_of_2]
        # Also register the triton versions if available so callers using
        # triton.cdiv / triton.next_power_of_2 are handled transparently.
        try:
            import triton as _triton

            fns.extend([_triton.cdiv, _triton.next_power_of_2])
        except ImportError:
            pass
        return cast("dict[object, None]", dict.fromkeys(fns))

    def _config_value_operation(self) -> str | None:
        from .._utils import cdiv
        from .._utils import next_power_of_2

        if self.value in (cdiv, next_power_of_2):
            return self.value.__name__
        try:
            import triton as _triton

            if self.value is _triton.cdiv:
                return "cdiv"
            if self.value is _triton.next_power_of_2:
                return "next_power_of_2"
        except ImportError:
            pass
        return None


def _raise_shape_specializing(*args: object) -> None:
    raise exc.ShapeSpecializingCall


class PythonModuleType(LiteralType):
    # pyrefly: ignore [bad-override]
    value: types.ModuleType

    def __init__(self, origin: Origin, value: types.ModuleType) -> None:
        super().__init__(origin, value)
        self.value = value

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.value.__name__})"


class NumericType(TypeInfo):
    value: torch.SymInt | torch.SymBool | torch.SymFloat

    def __init__(
        self, origin: Origin, value: torch.SymInt | torch.SymBool | torch.SymFloat
    ) -> None:
        super().__init__(origin)
        self.value = value

    def to_sympy(self) -> sympy.Expr:
        # value may be a SymBool whose `_sympy_()` is a boolean rather than Expr.
        # pyrefly: ignore [bad-return]
        return self.value._sympy_()

    @property
    def python_type(self) -> type[float | int | bool]:
        raise NotImplementedError

    def __str__(self) -> str:
        env = CompileEnvironment.current()
        # Preserve the existing debug/golden rendering except while semantic
        # fingerprinting supplies canonical symbolic-shape names.
        value = (
            env.sympy_debug(self.value._sympy_())
            if env.has_debug_shape_rename_override
            else self.value
        )
        return f"{type(self).__name__}({value})"

    def proxy(self) -> torch.SymInt | torch.SymBool | torch.SymFloat | int:
        return self.value

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        return TypeInfo.from_example(_eval_unary(op, self.value), self.origin)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, (LiteralType, NumericType)):
            if NumericType.known_equal(self.value, other.value):
                env = CompileEnvironment.current()
                left = env.resolve_fragment_global(self.value)
                right = (
                    env.resolve_fragment_global(other.value)
                    if isinstance(other, NumericType)
                    else other.value
                )
                if left is not None and not _same_scalar(left, right):
                    # Equal simplified symbols do not prove equal captured
                    # values across a runtime branch (notably -0 versus +0).
                    # Keep the ordinary symbolic type but drop domain proof.
                    return self.new_unbacked(self.origin)
                return self
            if self.python_type == other.python_type:
                return self.new_unbacked(self.origin)
        return super().merge(other, var_name=var_name)

    @staticmethod
    def subtype(
        python_type: type[float | int | bool],
    ) -> type[NumericType]:
        return _numeric_types[python_type]

    @staticmethod
    def known_equal(left: object, right: object) -> bool:
        """Check if two are equal without introducing guards"""
        if isinstance(left, (NumericType, LiteralType)):
            left = left.value
        if isinstance(right, (NumericType, LiteralType)):
            right = right.value

        if isinstance(left, (int, float, bool)) and isinstance(
            right, (int, float, bool)
        ):
            return left == right

        if isinstance(left, (torch.SymInt | torch.SymBool | torch.SymFloat)):
            vleft = left._sympy_()
        elif isinstance(left, int | float | bool):
            vleft = sympy.sympify(left)
        else:
            return False

        if isinstance(right, (torch.SymInt | torch.SymBool | torch.SymFloat)):
            vright = right._sympy_()
        elif isinstance(right, int | float | bool):
            vright = sympy.sympify(right)
        else:
            return False

        try:
            static_expr = CompileEnvironment.current().shape_env._maybe_evaluate_static(
                sympy.Eq(vleft, vright)
            )
            if static_expr is not None:
                return bool(static_expr)
        except TypeError:
            pass
        return False

    @classmethod
    def new_unbacked(cls, origin: Origin) -> Self:
        raise NotImplementedError

    def populate_symbol_origins(self, origin: Origin) -> None:
        expr_to_origin = HostFunction.current().expr_to_origin
        expr = CompileEnvironment.current().shape_env.simplify(self.value._sympy_())
        if expr not in expr_to_origin or origin.depth() < expr_to_origin[expr].depth():
            expr_to_origin[expr] = SymbolOrigin(origin)


class SymIntType(NumericType):
    # pyrefly: ignore [bad-override]
    value: torch.SymInt

    @classmethod
    def new_unbacked(cls, origin: Origin) -> Self:
        return cls(
            origin,
            CompileEnvironment.current().create_unbacked_symint(),
        )

    @property
    def python_type(self) -> type[int]:
        return int

    def proxy(self) -> torch.SymInt | int:
        if isinstance(self.value._sympy_(), sympy.Integer):
            return self.value.__int__()
        return self.value


class SymFloatType(NumericType):
    # pyrefly: ignore [bad-override]
    value: torch.SymFloat

    @classmethod
    def new_unbacked(cls, origin: Origin) -> Self:
        shape_env = CompileEnvironment.current().shape_env
        with shape_env.ignore_fresh_unbacked_symbols():
            return cls(
                origin,
                shape_env.create_unbacked_symfloat(),
            )

    @property
    def python_type(self) -> type[float]:
        return float


class SymBoolType(NumericType):
    # pyrefly: ignore [bad-override]
    value: torch.SymBool

    @classmethod
    def new_unbacked(cls, origin: Origin) -> Self:
        shape_env = CompileEnvironment.current().shape_env
        with shape_env.ignore_fresh_unbacked_symbols():
            return cls(
                origin,
                shape_env.create_unbacked_symbool(),
            )

    @property
    def python_type(self) -> type[bool]:
        return bool


_numeric_types: dict[type[object], type[NumericType]] = {
    int: SymIntType,
    float: SymFloatType,
    bool: SymBoolType,
}


def _get_hint(numel: int | torch.SymInt | AutoSize | None) -> int:
    """Get the size hint for the block size, or 8192 if not specified."""
    if numel is None or isinstance(numel, AutoSize):
        # For data-dependent sizes, use arbitrary hint of 8192
        return 8192
    return CompileEnvironment.current().size_hint(numel)


def _detect_outer_block_bound(
    numel: torch.SymInt, env: CompileEnvironment
) -> int | None:
    """If ``numel`` equals ``tile.end - tile.begin`` for an outer tile, return
    that tile's block_id.  Used to cap inner-tile block sizes to the enclosing
    outer tile's extent (e.g. ``hl.tile(outer.begin, outer.end)``).
    """
    from .variable_origin import TileBeginOrigin
    from .variable_origin import TileEndOrigin

    # Direct match: numel is another block's var.
    direct = env.get_block_id(numel)
    if direct is not None:
        return direct

    expr = _symint_expr(numel)
    if expr is None:
        return None
    host_fn = HostFunction.current()

    # Walk the free symbols: expect exactly one TileBeginOrigin and one
    # TileEndOrigin with the same block_id, then verify the expression is
    # structurally the tile extent.
    begin_bid: int | None = None
    end_bid: int | None = None
    begin_sym: sympy.Symbol | None = None
    end_sym: sympy.Symbol | None = None
    for sym in expr.free_symbols:
        if not isinstance(sym, sympy.Symbol):
            return None
        symbol_origin = host_fn.expr_to_origin.get(sym)
        if symbol_origin is None:
            return None
        origin = symbol_origin.origin
        if isinstance(origin, TileBeginOrigin):
            if begin_bid is not None:
                return None
            begin_bid = origin.block_id
            begin_sym = sym
        elif isinstance(origin, TileEndOrigin):
            if end_bid is not None:
                return None
            end_bid = origin.block_id
            end_sym = sym
        else:
            return None
    if (
        begin_bid is not None
        and end_bid is not None
        and begin_bid == end_bid
        and begin_sym is not None
        and end_sym is not None
        and sympy.expand(expr - (end_sym - begin_sym)) == 0  # pyrefly: ignore[unsupported-operation]
    ):
        return begin_bid
    return None


class TileIndexType(TypeInfo):
    block_id: int

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.block_id})"

    def __init__(self, origin: Origin, block_id: int) -> None:
        super().__init__(origin)
        self.block_id = block_id

    def proxy(self) -> object:
        with proxy_tensor.disable_proxy_modes_tracing():
            fake_mode = torch._C._unset_dispatch_mode(
                torch._C._TorchDispatchModeKey.FAKE
            )
            try:
                with torch._C._DisableTorchDispatch():
                    return Tile(self.block_id)
            finally:
                assert fake_mode is not None
                torch._C._set_dispatch_mode(fake_mode)

    @staticmethod
    def allocate(
        numel: int | torch.SymInt | AutoSize | None,
        origin: Origin,
        block_size: int | torch.SymInt | None = None,
    ) -> TileIndexType:
        env = CompileEnvironment.current()
        if block_size is None:
            block_id = env.allocate_block_size(numel, source=LoopSpecBlockSizeSource())
            outer_max: int | None = None
            bounded_by: int | None = None
            if isinstance(numel, torch.SymInt):
                maybe_bounded_by = _detect_outer_block_bound(numel, env)
                if maybe_bounded_by is not None:
                    bounded_by = maybe_bounded_by
                    try:
                        outer_spec = env.config_spec.block_sizes.block_id_lookup(
                            maybe_bounded_by
                        )
                    except KeyError:
                        # A fixed outer tile has no autotuner BlockSizeSpec, but it
                        # still bounds this inner loop. Retain the relationship and
                        # use the fixed source's value as the search ceiling.
                        if 0 <= maybe_bounded_by < len(env.block_sizes):
                            outer_info = env.block_sizes[maybe_bounded_by]
                            source = outer_info.block_size_source
                            if isinstance(source, FixedBlockSizeSource):
                                try:
                                    outer_max = max(
                                        1,
                                        int(env.size_hint(source.value)),
                                    )
                                except Exception:
                                    outer_max = None
                    else:
                        outer_max = outer_spec.max_size
            env.config_spec.block_sizes.append(
                BlockSizeSpec(
                    block_id=block_id,
                    size_hint=_get_hint(numel),
                    max_size=outer_max,
                    bounded_by_block_id=bounded_by,
                )
            )
            if env.config_spec.supports_config_key("num_threads"):
                env.config_spec.num_threads.append(
                    NumThreadsSpec(
                        block_id=block_id,
                        size_hint=_get_hint(numel),
                    )
                )
        else:
            block_id = env.allocate_block_size(
                numel,
                source=FixedBlockSizeSource(block_size),
            )
        return TileIndexType(origin, block_id)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, TileIndexType):
            if self.block_id == other.block_id:
                return self
            raise exc.TypeInferenceError(
                f"TileIndexType mismatch in control flow: {self.block_id} and {other.block_id}"
            )
        return super().merge(other, var_name=var_name)

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        if isinstance(getattr(Tile, attr, None), property):
            return TypeInfo.from_example(getattr(self.proxy(), attr), origin)
        return super().propagate_attribute(attr, origin)


class JaggedTileIndexType(TileIndexType):
    parent_block_ids: list[int]

    def __init__(
        self, origin: Origin, block_id: int, parent_block_ids: list[int]
    ) -> None:
        super().__init__(origin, block_id)
        self.parent_block_ids = parent_block_ids

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, JaggedTileIndexType):
            if (
                self.block_id == other.block_id
                and self.parent_block_ids == other.parent_block_ids
            ):
                return self
            raise exc.TypeInferenceError(
                f"JaggedTileIndexType mismatch: block/parents {self.block_id}/{self.parent_block_ids} "
                f"vs {other.block_id}/{other.parent_block_ids}"
            )
        return super().merge(other, var_name=var_name)


class BlockSizeType(SymIntType):
    """Type for block sizes registered via register_block_size"""

    block_id: int

    def __init__(self, origin: Origin, value: torch.SymInt, block_id: int) -> None:
        super().__init__(origin, value)
        self.block_id = block_id

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.block_id})"


class GridIndexType(SymIntType):
    block_id: int

    def __init__(
        self,
        origin: Origin,
        sym: torch.SymInt,
        block_id: int,
    ) -> None:
        super().__init__(origin, sym)
        self.block_id = block_id

    def __str__(self) -> str:  # pragma: no cover – debug helper
        return f"{type(self).__name__}({self.block_id})"

    @staticmethod
    def allocate(
        numel: int | torch.SymInt,
        origin: Origin,
        step: int | torch.SymInt = 1,
    ) -> GridIndexType:
        from .._compiler.compile_environment import CompileEnvironment
        from .host_function import HostFunction
        from .host_function import SymbolOrigin

        env = CompileEnvironment.current()
        block_id = env.allocate_block_size(numel, source=FixedBlockSizeSource(step))
        # assign this a new unbacked symbol since this should be treated like a scalar rather than a tile
        sym = env.create_unbacked_symint()
        HostFunction.current().expr_to_origin[sym._sympy_()] = SymbolOrigin(
            origin=GridOrigin(block_id),
        )
        return GridIndexType(origin, sym, block_id)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:  # type: ignore[override]
        if isinstance(other, GridIndexType):
            if self.block_id == other.block_id:
                return self
            raise exc.TypeInferenceError(
                f"GridIndexType mismatch in control flow: {self.block_id} vs {other.block_id}"
            )
        return super().merge(other, var_name=var_name)


class IterType(TypeInfo):
    inner: TypeInfo

    def __init__(self, origin: Origin, inner: TypeInfo) -> None:
        super().__init__(origin)
        self.inner = inner

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.inner!s})"

    def propagate_iter(self, origin: Origin) -> TypeInfo:
        return self.inner


class NoType(TypeInfo):
    """Used for AST nodes like Store() where a type is not applicable."""

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        return other

    def debug_annotations(self) -> list[str]:
        return []


class CollectionType(TypeInfo):
    element_types: (
        list[TypeInfo] | tuple[TypeInfo, ...] | dict[str | int, TypeInfo] | slice
    )

    def __init__(
        self,
        origin: Origin,
        element_types: list[TypeInfo]
        | tuple[TypeInfo, ...]
        | dict[str | int, TypeInfo]
        | slice,
        *,
        contents_exact: bool = True,
        original_value: object = None,
    ) -> None:
        super().__init__(origin)
        self.element_types = element_types
        self._original_identity = (
            original_value
            if isinstance(original_value, _ContainerIdentity)
            else CompileEnvironment.current().container_identity_witness(original_value)
            if original_value is not None
            else None
        )
        # An opaque call can return a different collection subclass for the
        # actual contents. Only proved builtin collections permit a pure
        # host-length projection into device code.
        self.length_projection_safe = self._original_identity is None or type(
            self._original_identity.value
        ) in (list, tuple, dict)
        # Comprehension propagation may retain only representative elements.
        # Those describe element types, not the runtime collection contents.
        self.contents_exact = contents_exact
        if not contents_exact:
            self.mark_approximate_contents()

    def membership_container(self) -> object:
        """Return literal membership operands only when contents are proved."""
        raise NotImplementedError

    def _nested_collections(self) -> list[CollectionType]:
        elements = self.element_types
        # Slice bounds are scalar; only list/tuple/dict values can contain
        # another collection.
        if isinstance(elements, slice):
            return []
        children = elements.values() if isinstance(elements, dict) else elements
        return [child for child in children if isinstance(child, CollectionType)]

    def contains_approximate_contents(self) -> bool:
        return not self.contents_exact or any(
            child.contains_approximate_contents()
            for child in self._nested_collections()
        )

    def mark_approximate_contents(
        self, *, length_projection_safe: bool | None = None
    ) -> None:
        # Representative contents cannot prove a nested collection's values.
        self.contents_exact = False
        if length_projection_safe is not None:
            self.length_projection_safe = length_projection_safe
        self.element_types = tree_map_only(
            CollectionType,
            lambda child: child.with_approximate_contents(
                length_projection_safe=length_projection_safe
            ),
            self.element_types,
        )

    def with_approximate_contents(
        self, *, length_projection_safe: bool | None = None
    ) -> CollectionType:
        # Keep origins and subclass metadata while isolating shared children:
        # extracting an uncertain item must not taint an exact source alias.
        result = copy.copy(self)
        result.mark_approximate_contents(length_projection_safe=length_projection_safe)
        return result

    def _preserve_identity(self, proxy: _T) -> _T:
        # Output-spec consumers reconstruct proxies after binding has left its
        # environment. The identity side table is only needed during tracing.
        if self._original_identity is not None and CompileEnvironment.has_current():
            CompileEnvironment.current().record_container_proxy(
                proxy, self._original_identity
            )
        return proxy

    def _merged_original(self, other: CollectionType) -> object:
        return (
            self._original_identity
            if self._original_identity is other._original_identity
            else None
        )

    @property
    def python_type(self) -> type[object]:
        return type(self.element_types)

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        if isinstance(op, ast.Not):
            if (
                origin.is_host()
                and (length := _native_record_length(self, truth=True)) is not None
            ):
                return LiteralType(origin, not length)
            if origin.is_host() and _has_untraced_host_protocol(self, nested=False):
                _reject_untraced_host_protocol()
            if not self.contents_exact:
                return SymBoolType.new_unbacked(origin)
            return LiteralType(origin, not self.element_types)
        return super().propagate_unary(op, origin)

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if isinstance(key, LiteralType):
            if isinstance(elements := self.element_types, (list, dict)) and isinstance(
                k := key.value, (int, str)
            ):
                if k in elements:
                    # pyrefly: ignore [bad-index, unsupported-operation]
                    elements[k] = elements[k].merge(value)
                else:
                    # pyrefly: ignore [unsupported-operation]
                    elements[k] = value
                return self
        return super().propagate_setitem(key, value, origin)

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        try:
            # A slice with uncertain provenance still supplies representative
            # bounds for element types, but cannot prove the result's contents.
            literal_key = key.as_literal()
        except NotImplementedError:
            pass
        else:
            element_key = (
                0
                if not self.contents_exact
                and isinstance(self, SequenceType)
                and isinstance(literal_key, int)
                else literal_key
            )
            try:
                # pyrefly: ignore [bad-index]
                result = self.element_types[element_key]
            except (KeyError, IndexError) as e:
                raise exc.TypeInferenceError(f"{type(e).__name__}: {e}") from None
            if isinstance(result, TypeInfo):
                if not self.contents_exact and origin.is_host():
                    return self._runtime_item(result, origin)
                return result
            if type(result) is self.python_type:  # sliced!
                output = type(self)(
                    origin=origin,
                    element_types=result,  # pyrefly: ignore [bad-argument-type]
                    contents_exact=self.contents_exact
                    and (not isinstance(key, SliceType) or key.contents_exact),
                )
                output.length_projection_safe = self.length_projection_safe
                return output
        return super().propagate_getitem(key, origin)

    @staticmethod
    def _runtime_item(item: TypeInfo, origin: Origin) -> TypeInfo:
        # Each read selects an actual runtime value, not the representative
        # element's scalar or tensor identity. In particular, positive and
        # negative indices must not share one unbacked symbol.
        if isinstance(item, (NumericType, LiteralType)) and item.python_type in (
            int,
            float,
            bool,
        ):
            return NumericType.subtype(item.python_type).new_unbacked(origin)
        if isinstance(item, TensorType):
            value = item.fake_value
            env = CompileEnvironment.current()
            # Keep this selection independent of the representative's storage.
            # Host allocations derived from it inherit runtime layout metadata.
            selected = torch.empty_strided(
                value.size(), value.stride(), dtype=value.dtype, device=value.device
            )
            env.mark_runtime_tensor_layout(selected)
            return TensorType(origin, selected)
        if isinstance(item, CollectionType):
            # Nested collections already carry approximate contents. Preserve
            # their source alias without treating those contents as constants.
            return item
        raise exc.TracedArgNotSupported(
            "a non-numeric result from representative collection contents"
        )

    def truth_value(self) -> bool:
        if (
            self.origin.is_host()
            and (length := _native_record_length(self, truth=True)) is not None
        ):
            return bool(length)
        if self.origin.is_host() and _has_untraced_host_protocol(self, nested=False):
            _reject_untraced_host_protocol()
        if not self.contents_exact:
            raise NotImplementedError
        return bool(self.element_types)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> object:
        raise NotImplementedError


class SequenceType(CollectionType):
    # pyrefly: ignore [bad-override]
    element_types: list[TypeInfo] | tuple[TypeInfo, ...]

    def __str__(self) -> str:
        start, *_, end = repr(self.element_types)
        if len(self.element_types) == 1 and self.python_type is tuple:
            end = ", )"
        items = ", ".join(map(str, self.element_types))
        return f"{type(self).__name__}({start}{items}{end})"

    def _maybe_tuple(self, x: list[_T]) -> tuple[_T, ...] | list[_T]:
        if isinstance(self.element_types, tuple):
            return tuple(x)
        return x

    def proxy(self) -> list[object] | tuple[object, ...]:
        if self.contains_approximate_contents():
            raise NotImplementedError
        return self._preserve_identity(
            self._maybe_tuple([x.proxy() for x in self.element_types])
        )

    def membership_container(self) -> object:
        return self.as_literal()

    def as_literal(self) -> list[object] | tuple[object, ...]:
        if not self.contents_exact:
            raise NotImplementedError
        return self._preserve_identity(
            self._maybe_tuple([x.as_literal() for x in self.element_types])
        )

    def unpack(self) -> list[TypeInfo]:
        return [*self.element_types]

    def populate_symbol_origins(self, origin: Origin) -> None:
        for i, subtype in enumerate(self.element_types):
            subtype.populate_symbol_origins(GetItemOrigin(origin, i))

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        # Tuple/List indexing with non-literal indices (e.g., from hl.static_range)
        if self.python_type in (tuple, list) and isinstance(key, SymIntType):
            env = CompileEnvironment.current()
            if _symint_free_symbols(key.value) <= env.specialized_vars:
                # hl.specialize already guards these symbols. Retain the actual
                # selected element's value, identity and origin rather than the
                # representative type used for a genuinely dynamic index.
                return super().propagate_getitem(
                    LiteralType(key.origin, int(key.value)), origin
                )
            if not self.element_types:
                raise exc.TypeInferenceError("Cannot index empty sequence")
            first_type = self.element_types[0]
            if not all(type(e) is type(first_type) for e in self.element_types[1:]):
                raise exc.TypeInferenceError(
                    "Sequence indexing with non-literal index requires all elements to have the same type"
                )
            if origin.is_host():
                if not self.contents_exact:
                    return self._runtime_item(first_type, origin)
                # A runtime index can select any element. Merge their values,
                # layouts and storage owners just as for a host control-flow
                # join, instead of borrowing the first element's identity.
                selected = functools.reduce(
                    lambda left, right: left.merge(right), self.element_types
                )
                selected = copy.copy(selected)
                selected.origin = origin
                return selected
            return first_type

        return super().propagate_getitem(key, origin)

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if self.python_type is list and isinstance(key, SymIntType):
            if not self.element_types:
                raise exc.TypeInferenceError("Cannot index empty sequence")
            new_elements = [elem.merge(value) for elem in self.element_types]
            result = SequenceType(
                origin=origin,
                element_types=new_elements,
                contents_exact=self.contents_exact,
                original_value=self._original_identity,
            )
            result.length_projection_safe = self.length_projection_safe
            return result
        return super().propagate_setitem(key, value, origin)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, SequenceType):
            self_elements = self.element_types
            other_elements = other.element_types
            if len(self_elements) == len(other_elements):
                result = SequenceType(
                    origin=other.origin,
                    contents_exact=self.contents_exact and other.contents_exact,
                    original_value=self._merged_original(other),
                    element_types=self._maybe_tuple(
                        [
                            self_elements[i].merge(other_elements[i], var_name=var_name)
                            for i in range(len(self_elements))
                        ]
                    ),
                )
                result.length_projection_safe = (
                    self.length_projection_safe and other.length_projection_safe
                )
                return result
        return super().merge(other, var_name=var_name)

    def tree_map(
        self, fn: Callable[[TypeInfo], object]
    ) -> list[object] | tuple[object, ...]:
        return self._preserve_identity(
            self._maybe_tuple([x.tree_map(fn) for x in self.element_types])
        )


class DictType(CollectionType):
    # pyrefly: ignore [bad-override]
    element_types: dict[str | int, TypeInfo]

    def __str__(self) -> str:
        items = ", ".join(f"{k!r}: {v!s}" for k, v in self.element_types.items())
        return f"{type(self).__name__}({{{items}}})"

    def proxy(self) -> dict[str | int, object]:
        if self.contains_approximate_contents():
            raise NotImplementedError
        return self._preserve_identity(
            {k: v.proxy() for k, v in self.element_types.items()}
        )

    def membership_container(self) -> object:
        if not self.contents_exact:
            raise NotImplementedError
        return self.element_types.keys()

    def as_literal(self) -> dict[str | int, object]:
        if not self.contents_exact:
            raise NotImplementedError
        return self._preserve_identity(
            {k: v.as_literal() for k, v in self.element_types.items()}
        )

    def unpack(self) -> list[TypeInfo]:
        return [TypeInfo.from_example(k, self.origin) for k in self.element_types]

    def populate_symbol_origins(self, origin: Origin) -> None:
        for k, subtype in self.element_types.items():
            subtype.populate_symbol_origins(GetItemOrigin(origin, k))

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if type(self) is type(other):
            assert isinstance(other, DictType)
            self_elements = self.element_types
            other_elements = other.element_types
            if set(self_elements.keys()) == set(other_elements.keys()):
                result = type(self)(
                    origin=other.origin,
                    contents_exact=self.contents_exact and other.contents_exact,
                    original_value=self._merged_original(other),
                    element_types={
                        key: self_elements[key].merge(
                            other_elements[key], var_name=var_name
                        )
                        for key in self_elements
                    },
                )
                result.length_projection_safe = (
                    self.length_projection_safe and other.length_projection_safe
                )
                return result
        return super().merge(other, var_name=var_name)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> dict[str | int, object]:
        return self._preserve_identity(
            {k: v.tree_map(fn) for k, v in self.element_types.items()}
        )


class ClassType(DictType):
    def membership_container(self) -> object:
        # Record fields are not membership keys. Only default tuple containment
        # over the same proved positional values can be evaluated here.
        identity = self._original_identity
        original = identity.value if identity is not None else None
        if (
            not self.contents_exact
            or not isinstance(original, tuple)
            or type(original).__contains__ is not tuple.__contains__
        ):
            raise NotImplementedError
        values = tuple(value.as_literal() for value in self.element_types.values())
        if len(values) != tuple.__len__(original) or any(
            value is not item
            for value, item in zip(values, tuple.__iter__(original), strict=True)
        ):
            # Custom field layouts or reconstructed contents do not establish
            # the tuple's actual positional values or their equality behavior.
            raise NotImplementedError
        return values

    def populate_symbol_origins(self, origin: Origin) -> None:
        for key, subtype in self.element_types.items():
            assert isinstance(key, str)
            subtype.populate_symbol_origins(AttributeOrigin(origin, key))

    def unpack(self) -> list[TypeInfo]:
        """Unpack a ClassType into its values (not field name strings).

        ClassType represents namedtuples and torch.return_types structseqs
        (e.g., torch.sort, torch.topk). In Python, unpacking these yields
        their values, not their field names::

            vals, indices = torch.sort(x)  # vals=tensor, indices=tensor

        DictType.unpack() returns keys (field name strings), which is wrong
        for tuple-like unpacking. This override returns the values instead.
        """
        return list(self.element_types.values())

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        try:
            return self.element_types[attr]
        except KeyError:
            desc = str(
                getattr(origin.value, "location", origin.value.__class__.__name__)
            )
            raise exc.TypeInferenceError(
                f"Attribute '{attr}' is not supported on {desc}"
            ) from None


class StackTensorType(ClassType):
    # pyrefly: ignore [bad-override]
    element_types: dict[str, TypeInfo]

    # pyrefly: ignore [bad-override]
    def proxy(self) -> StackTensor:
        with proxy_tensor.disable_proxy_modes_tracing():
            fake_mode = torch._C._unset_dispatch_mode(
                torch._C._TorchDispatchModeKey.FAKE
            )
            try:
                assert isinstance(self.element_types["tensor_like"], TensorType)
                assert isinstance(self.element_types["dev_ptrs"], TensorType)
                return StackTensor(
                    self.element_types["tensor_like"].proxy(),
                    self.element_types["dev_ptrs"].proxy(),
                )
            finally:
                assert fake_mode is not None
                torch._C._set_dispatch_mode(fake_mode)

    def _device_indexing_size(self, key: TypeInfo) -> list[int | torch.SymInt]:
        tensor_like_type = self.element_types["tensor_like"]
        assert isinstance(tensor_like_type, TensorType)
        size_like = tensor_like_type._device_indexing_size(key)

        dev_ptrs_type = self.element_types["dev_ptrs"]
        assert isinstance(dev_ptrs_type, TensorType)
        stack_size = list(dev_ptrs_type.fake_value.size())

        return stack_size + size_like

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if origin.is_host():
            warning(exc.TensorOperationInWrapper)
        else:
            lhs_shape = self._device_indexing_size(key)
            lhs_rank = len(lhs_shape)
            if isinstance(value, TensorType):
                rhs_rank = value.fake_value.ndim
                if lhs_rank != rhs_rank:
                    raise exc.RankMismatch(
                        lhs_rank,
                        rhs_rank,
                        f"LHS shape: {tuple(lhs_shape)}, RHS shape: {tuple(value.fake_value.shape)}",
                    )
            elif isinstance(value, (NumericType, LiteralType)):
                # Allow scalar assignment to tensor (broadcasts to tensor shape)
                pass
            else:
                raise exc.RequiresTensorInAssignment(value)
        return self

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        if origin.is_host():
            warning(exc.TensorOperationInWrapper)

        assert isinstance(self.element_types["tensor_like"], TensorType)
        return TensorType(
            origin,
            self.element_types["tensor_like"]
            .proxy()
            .new_empty(self._device_indexing_size(key)),
        )


class AsyncCopyDescriptorType(ClassType):
    """Type of a Helion asynchronous remote-copy handle."""

    # pyrefly: ignore [bad-override]
    def proxy(self) -> object:
        from ..language.distributed_ops import AsyncCopyDescriptor

        return AsyncCopyDescriptor()

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        if attr in ("start", "wait", "wait_send", "wait_recv"):
            return _AsyncCopyDescriptorMethodType(origin, self, attr)
        return super().propagate_attribute(attr, origin)


class _AsyncCopyDescriptorMethodType(TypeInfo):
    descriptor_type: AsyncCopyDescriptorType
    method_name: str

    def __init__(
        self,
        origin: Origin,
        descriptor_type: AsyncCopyDescriptorType,
        method_name: str,
    ) -> None:
        super().__init__(origin)
        self.descriptor_type = descriptor_type
        self.method_name = method_name

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        if args or kwargs:
            raise exc.TypeInferenceError(
                f"AsyncCopyDescriptor.{self.method_name}() takes no arguments"
            )
        from ..language.distributed_ops import start_async_remote_copy_descriptor
        from ..language.distributed_ops import wait_async_remote_copy
        from ..language.distributed_ops import wait_recv_async_remote_copy
        from ..language.distributed_ops import wait_send_async_remote_copy

        descriptor_ops = {
            "start": start_async_remote_copy_descriptor,
            "wait": wait_async_remote_copy,
            "wait_send": wait_send_async_remote_copy,
            "wait_recv": wait_recv_async_remote_copy,
        }
        result = CallableType(origin, descriptor_ops[self.method_name]).propagate_call(
            (self.descriptor_type,), {}, origin
        )
        assert result is not None
        return result

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if (
            isinstance(other, _AsyncCopyDescriptorMethodType)
            and self.method_name == other.method_name
        ):
            return self
        return super().merge(other, var_name=var_name)


class SliceType(CollectionType):
    # pyrefly: ignore [bad-override]
    element_types: slice

    @property
    def lower(self) -> TypeInfo:
        return self.element_types.start

    @property
    def upper(self) -> TypeInfo:
        return self.element_types.stop

    @property
    def step(self) -> TypeInfo:
        return self.element_types.step

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.lower!s}:{self.upper!s}:{self.step!s})"

    def proxy(self) -> slice:
        return slice(self.lower.proxy(), self.upper.proxy(), self.step.proxy())

    def as_literal(self) -> slice:
        # These bounds may be representative. Collection slicing separately
        # carries contents_exact; literal Python descriptors retain their
        # existing type-propagation behavior.
        return slice(
            self.lower.as_literal(), self.upper.as_literal(), self.step.as_literal()
        )

    def unpack(self) -> list[TypeInfo]:
        return [self.lower, self.upper, self.step]

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, SliceType):
            self_elements = self.element_types
            other_elements = other.element_types
            return SliceType(
                origin=other.origin,
                contents_exact=self.contents_exact and other.contents_exact,
                element_types=slice(
                    self_elements.start.merge(other_elements.start, var_name=var_name),
                    self_elements.stop.merge(other_elements.stop, var_name=var_name),
                    self_elements.step.merge(other_elements.step, var_name=var_name),
                ),
            )
        return super().merge(other, var_name=var_name)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> slice:
        return slice(
            self.lower.tree_map(fn), self.upper.tree_map(fn), self.step.tree_map(fn)
        )


def tensor_may_have_many_elements(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is not known to hold exactly one element (a tile's
    extent is a block size, which may exceed 1)."""
    env = CompileEnvironment.current()
    return not all(env.known_equal(size, 1) for size in tensor.shape)


def _is_size_backed_float(value: object) -> bool:
    """Retain shape expressions without treating runtime float data as a size."""
    if not isinstance(value, torch.SymFloat) or not CompileEnvironment.has_current():
        return False
    expression = value._sympy_()
    # These nodes preserve Python conversion/division instead of algebraically
    # simplifying ordered IEEE arithmetic. Add/Sub/Mul must retain their host
    # value: even size-derived floats can round before cancellation.
    if not isinstance(expression, (ToFloat, IntTrueDiv, FloatTrueDiv)):
        return False
    symbols = expression.free_symbols
    sources = CompileEnvironment.current().shape_env.var_to_sources
    return bool(symbols) and all(
        isinstance(symbol, sympy.Symbol)
        and any(
            isinstance(source, TensorPropertySource)
            and source.prop is TensorProperty.SIZE
            for source in sources.get(symbol, ())
        )
        for symbol in symbols
    )


def _preserve_host_float_operation(
    result: object, op: ast.AST, operands: tuple[object, ...]
) -> object:
    # These operations cannot raise for exact float operands. Integer literals
    # also need a finite float conversion; symbolic integers have no such proof.
    # Do not hoist division/powers out of a potentially untaken device branch.
    operator = next(
        (
            kind
            for kind in (ast.Add, ast.Sub, ast.Mult, ast.UAdd, ast.USub)
            if isinstance(op, kind)
        ),
        None,
    )
    if (
        operator is None
        or _is_size_backed_float(result)
        or not any(isinstance(value, torch.SymFloat) for value in operands)
        or not CompileEnvironment.has_current()
    ):
        return result
    try:
        host = HostFunction.current()
    except NoCurrentFunction:
        return result
    origins: list[Origin | int | float | bool] = []
    for value in operands:
        if isinstance(value, torch.SymFloat):
            info = host.expr_to_origin.get(value._sympy_())
            if info is None or not info.origin.is_host():
                return result
            origins.append(info.origin)
        elif type(value) in (float, int, bool):
            if type(value) is int:
                try:
                    float(value)
                except OverflowError:
                    return result
            origins.append(cast("int | float | bool", value))
        else:
            return result
    origin = ScalarArithmeticOrigin(operator, tuple(origins))
    result = SymFloatType.new_unbacked(origin).value
    CompileEnvironment.current().preserve_tunable_float_provenance(result, operands)
    host.expr_to_origin[result._sympy_()] = SymbolOrigin(origin)
    return result


def _eval_unary(op: ast.unaryop, value: object) -> object:
    result = _eval_unary_impl(op, value)
    result = _preserve_host_float_operation(result, op, (value,))
    if CompileEnvironment.has_current():
        return CompileEnvironment.current().record_fragment_global_operation(
            result, _eval_unary_impl, op, (value,)
        )
    return result


def _eval_unary_impl(op: ast.unaryop, value: object) -> object:
    if isinstance(op, ast.Not):
        # ``not`` would call bool() and guard a symbolic value on its hint.
        if isinstance(value, torch.SymBool):
            return torch.sym_not(value)
        if isinstance(value, (torch.SymInt, torch.SymFloat)):
            return value == 0
        return not value
    if isinstance(op, ast.UAdd):
        # pyrefly: ignore [unsupported-operation]
        return +value
    if isinstance(op, ast.USub):
        # pyrefly: ignore [unsupported-operation]
        return -value
    if isinstance(op, ast.Invert):
        # pyrefly: ignore [unsupported-operation]
        return ~value
    raise AssertionError(f"{type(op).__name__} unknown unary op")


def _eval_binary(op: ast.operator, left: object, right: object) -> object:
    result = _eval_binary_impl(op, left, right)
    result = _preserve_host_float_operation(result, op, (left, right))
    if CompileEnvironment.has_current():
        return CompileEnvironment.current().record_fragment_global_operation(
            result, _eval_binary_impl, op, (left, right)
        )
    return result


def _eval_binary_impl(op: ast.operator, left: object, right: object) -> object:
    if isinstance(op, ast.Add):
        # pyrefly: ignore [unsupported-operation]
        return left + right
    if isinstance(op, ast.Sub):
        # pyrefly: ignore [unsupported-operation]
        return left - right
    if isinstance(op, ast.Mult):
        # pyrefly: ignore [unsupported-operation]
        return left * right
    if isinstance(op, ast.Div):
        # pyrefly: ignore [unsupported-operation]
        return left / right
    if isinstance(op, ast.FloorDiv):
        # pyrefly: ignore [unsupported-operation]
        return left // right
    if isinstance(op, ast.Mod):
        # pyrefly: ignore [unsupported-operation]
        return left % right
    if isinstance(op, ast.Pow):
        # pyrefly: ignore [unsupported-operation]
        return left**right
    if isinstance(op, ast.LShift):
        # pyrefly: ignore [unsupported-operation]
        return left << right
    if isinstance(op, ast.RShift):
        # pyrefly: ignore [unsupported-operation]
        return left >> right
    if isinstance(op, ast.BitOr):
        # pyrefly: ignore [unsupported-operation]
        return left | right
    if isinstance(op, ast.BitXor):
        # pyrefly: ignore [unsupported-operation]
        return left ^ right
    if isinstance(op, ast.BitAnd):
        # pyrefly: ignore [unsupported-operation]
        return left & right
    if isinstance(op, ast.MatMult):
        # pyrefly: ignore [unsupported-operation]
        return left @ right
    raise AssertionError(f"{type(op).__name__} unknown binary op")


def _eval_compare(op: ast.cmpop, left: object, right: object) -> object:
    result = _eval_compare_impl(op, left, right)
    if CompileEnvironment.has_current() and isinstance(
        op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    ):
        return CompileEnvironment.current().record_fragment_global_operation(
            result, _eval_compare_impl, op, (left, right)
        )
    return result


def _eval_compare_impl(op: ast.cmpop, left: object, right: object) -> object:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        # pyrefly: ignore [unsupported-operation]
        return left < right
    if isinstance(op, ast.LtE):
        # pyrefly: ignore [unsupported-operation]
        return left <= right
    if isinstance(op, ast.Gt):
        # pyrefly: ignore [unsupported-operation]
        return left > right
    if isinstance(op, ast.GtE):
        # pyrefly: ignore [unsupported-operation]
        return left >= right
    if isinstance(op, ast.Is):
        return left is right
    if isinstance(op, ast.IsNot):
        return left is not right
    if isinstance(op, ast.In):
        # pyrefly: ignore [not-iterable]
        return left in right
    if isinstance(op, ast.NotIn):
        # pyrefly: ignore [not-iterable]
        return left not in right
    raise AssertionError(f"{type(op).__name__} unknown compare op")


def _to_proxy(arg: TypeInfo) -> object:
    try:
        return arg.proxy()
    except NotImplementedError:
        raise exc.TracedArgNotSupported(arg) from None


class BarrierResultType(LiteralType):
    """Marker type returned by hl.barrier() to signal a phase boundary."""


class PdlResultType(LiteralType):
    """Marker type returned by the hl.pdl_* ops; the value names the op."""
