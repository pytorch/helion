from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import functools
import re
import types
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import Protocol
from typing import TypeVar
from typing import cast
from unittest.mock import patch

import sympy
import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.convert_frame import compile_lock
from torch.fx.experimental import proxy_tensor
from torch.utils._pytree import tree_map_only

from .. import exc
from .. import language as language_module
from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_spec import BlockSizeSpec
from ..language._decorators import get_device_func_replacement
from ..language._decorators import is_api_func
from ..language.stack_tensor import StackTensor
from ..language.tile_proxy import Tile
from ..language.tile_proxy import _CheckForIndexCalls
from ..runtime.kernel import Kernel
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import create
from .compile_environment import AutoSize
from .compile_environment import CompileEnvironment
from .compile_environment import FixedBlockSizeSource
from .compile_environment import LoopSpecBlockSizeSource
from .compile_environment import warning
from .device_function import contains_only_block_size_symbols
from .host_function import HostFunction
from .host_function import SymbolOrigin
from .output_header import library_imports
from .source_location import current_location
from .tensor_utils import patch_tensor_factories
from .utils import compute_slice_size
from .variable_origin import ArgumentOrigin
from .variable_origin import AttributeOrigin
from .variable_origin import BuiltinOrigin
from .variable_origin import DeviceOrigin
from .variable_origin import GetItemOrigin
from .variable_origin import GlobalOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import SourceOrigin
from .variable_origin import TensorSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing_extensions import Self

    class _VisitMethod(Protocol):
        @staticmethod
        def __call__(self: object, node: ast.AST) -> TypeInfo: ...  # pyright: ignore[reportSelfClsParameterName]

    _T = TypeVar("_T")


class Scope:
    def get(self, name: str) -> TypeInfo:
        raise NotImplementedError

    def set(self, name: str, type_info: TypeInfo) -> NoReturn:
        raise NotImplementedError


@dataclasses.dataclass
class GlobalScope(Scope):
    function: HostFunction
    cache: dict[str, TypeInfo] = dataclasses.field(default_factory=dict)

    def get(self, name: str) -> TypeInfo:
        if name not in self.cache:
            self.cache[name] = self._get(name)
        return self.cache[name]

    def _get(self, name: str) -> TypeInfo:
        try:
            value = self.function.fn.__globals__[name]
        except KeyError:
            if hasattr(builtins, name):
                value = getattr(builtins, name)
                origin = BuiltinOrigin(name=name, function=self.function)
            else:
                raise exc.UndefinedVariable(name) from None
        else:
            if value is language_module:
                origin = GlobalOrigin(name="hl", function=self.function)
                return TypeInfo.from_example(value, origin)
            if name in library_imports:
                origin = GlobalOrigin(name=name, function=self.function)
                return TypeInfo.from_example(value, origin)

            origin = self.function.global_scope_origin(name)
            if isinstance(value, Kernel):
                return TypeInfo.from_example(value, origin)
            if not isinstance(
                value,
                (types.ModuleType, types.FunctionType, types.BuiltinFunctionType),
            ):
                value = self.function.register_fake(value, origin)
        return TypeInfo.from_example(value, origin)

    def set(self, name: str, type_info: TypeInfo) -> NoReturn:
        raise AssertionError("Cannot set in global scope")


@dataclasses.dataclass
class LocalScope(Scope):
    parent: Scope
    variables: dict[str, TypeInfo] = dataclasses.field(default_factory=dict)

    def get(self, name: str) -> TypeInfo:
        if name in self.variables:
            return self.variables[name]
        return self.parent.get(name)

    def maybe_get(self, name: str) -> TypeInfo | None:
        try:
            return self.get(name)
        except exc.UndefinedVariable:
            return None

    def set(self, name: str, type_info: TypeInfo) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.variables[name] = type_info

    def merge(self, other: LocalScope | dict[str, TypeInfo]) -> LocalScope:
        if isinstance(other, LocalScope):
            other = other.variables
        for k, v in other.items():
            if k in self.variables:
                existing = self.variables[k]
                merged = existing.merge(v, var_name=k)
                self.variables[k] = merged
            else:
                self.variables[k] = v
        return self

    def merge_if_else(
        self, true_scope: LocalScope, false_scope: LocalScope
    ) -> LocalScope:
        true = {**true_scope.variables}
        false = {**false_scope.variables}
        both = {}
        for k in [*false]:
            if k in true:
                lhs = true.pop(k)
                rhs = false.pop(k)
                both[k] = lhs.merge(rhs, var_name=k)
        self.merge(true)
        self.merge(false)
        # variables defined in both sides of branch overwrite existing values
        self.variables.update(both)
        return self

    def overwrite(self, other: LocalScope) -> None:
        self.variables.update(other.variables)

    def clone(self) -> LocalScope:
        return LocalScope(parent=self.parent, variables=dict(self.variables))

    def extract_locals(self) -> dict[str, TypeInfo]:
        if isinstance(self.parent, LocalScope):
            return {**self.parent.extract_locals(), **self.variables}
        return {**self.variables}


# Ops not matching this emit a warning if they are used in a host function
regexp_allowed_host_ops: re.Pattern[str] = re.compile(
    r"like|new|broadcast|promote|view|reshape|expand|permute|strided|"
    r"transpose|contiguous|unsqueeze|squeeze|zero|rand|full|fill"
)


class TypeInfo:
    origin: Origin

    def __init__(self, origin: Origin) -> None:
        assert isinstance(origin, Origin)
        self.origin = origin

    @classmethod
    def from_example(cls, value: object, origin: Origin) -> TypeInfo:
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
            return SequenceType(origin, cls._unpack_example(enumerate(value), origin))
        if type(value) is tuple:
            # TODO(jansel): track specializations
            return SequenceType(
                origin,
                tuple(cls._unpack_example(enumerate(value), origin)),
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
            )
        if isinstance(value, tuple) and hasattr(value, "_asdict"):
            # namedtuple
            return ClassType(
                origin,
                dict(
                    zip(
                        value._fields,  # pyright: ignore[reportAttributeAccessIssue]
                        cls._unpack_example(
                            value._asdict().items(),  # pyright: ignore[reportAttributeAccessIssue]
                            origin,
                        ),
                        strict=False,
                    )
                ),
            )
        if isinstance(value, ConfigSpecFragment):
            return ConfigFragmentType(origin, value)
        if dataclasses.is_dataclass(value):
            keys = value.__dataclass_fields__.keys()
            return ClassType(
                origin,
                dict(
                    zip(
                        keys,
                        cls._unpack_example(
                            tuple((k, getattr(value, k)) for k in keys),
                            origin,
                        ),
                        strict=False,
                    )
                ),
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

            return ClassType(origin, attrs)
        raise exc.UnsupportedPythonType(type(value).__name__)

    @staticmethod
    def _unpack_example(
        values: Sequence[tuple[int | str, object]] | enumerate[object],
        origin: Origin,
    ) -> list[TypeInfo]:
        return [
            TypeInfo.from_example(value, GetItemOrigin(origin, key))
            for key, value in values
        ]

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


class TensorType(TypeInfo):
    fake_value: torch.Tensor

    def __init__(self, origin: Origin, fake_value: torch.Tensor) -> None:
        super().__init__(origin)
        self.fake_value = fake_value
        if origin.is_device():
            CompileEnvironment.current().add_kernel_tensor_size(fake_value.size())

    def __str__(self) -> str:
        shape = []
        for s in self.fake_value.size():
            if isinstance(s, torch.SymInt):
                shape.append(
                    str(
                        s._sympy_().xreplace(
                            CompileEnvironment.current().debug_shape_renames
                        )
                    )
                )
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
            return SymBoolType.new_unbacked(origin)
        try:
            return TypeInfo.from_example(_eval_unary(op, self.fake_value), origin)
        except exc.Base:
            raise
        except Exception as e:
            raise exc.TorchOpTracingError(e) from e

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        assert origin.key == attr
        if attr in {"dtype", "device", "ndim", "shape"}:
            return TypeInfo.from_example(getattr(self.fake_value, attr), origin)
        return TensorAttributeType(origin, self)

    def _device_indexing_size(self, key: TypeInfo) -> list[int | torch.SymInt]:
        if isinstance(key, SequenceType):
            keys = key.unpack()
        else:
            keys = [key]
        inputs_consumed = 0
        output_sizes = []
        env = CompileEnvironment.current()
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

                # For slices with steps, we need to calculate the output size differently
                output_size = compute_slice_size(slice_obj, size)

                if self.origin.is_device():
                    output_sizes.append(output_size)
                elif output_size != 1:
                    # If all symbols in output_size are block size symbols, we reuse them
                    if isinstance(output_size, torch.SymInt):
                        expr = output_size._sympy_()
                        if (
                            isinstance(expr, sympy.Expr)
                            and expr.free_symbols
                            and contains_only_block_size_symbols(expr)
                        ):
                            output_sizes.append(output_size)
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
            elif isinstance(k, TensorType) and k.fake_value.ndim == 1:
                inputs_consumed += 1
                output_sizes.append(k.fake_value.size(0))
            elif k.contains_type(TileIndexType):
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
                return TypeInfo.from_example(self.fake_value[key.proxy()], origin)  # pyright: ignore[reportArgumentType]
            except NotImplementedError:
                raise exc.TypeInferenceError(
                    f"Subscript not supported on {self!s} with key={key!s}"
                ) from None
        return TensorType(
            origin, self.fake_value.new_empty(self._device_indexing_size(key))
        )

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
            if self.fake_value.size() != other.fake_value.size():
                raise exc.ControlFlowTensorMismatch(
                    var=var_name,
                    details=f"size {self.fake_value.size()} != {other.fake_value.size()}",
                )
            # TODO(jansel): handle symbolic shapes
            # TODO(jansel): stride check?
            return TensorType(other.origin, torch.empty_like(self.fake_value))
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


class TensorAttributeType(TypeInfo):
    origin: AttributeOrigin  # pyright: ignore[reportIncompatibleVariableOverride]
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
        if attr in {"dim", "ndimension"} and not (args or kwargs):
            return TypeInfo.from_example(self.tensor.fake_value.ndim, origin)
        if attr in {"shape", "size"} and not kwargs:
            fn = getattr(self.tensor.fake_value, attr)
            try:
                return TypeInfo.from_example(
                    fn(*[x.as_literal() for x in args]),
                    origin,
                )
            except NotImplementedError:
                raise exc.TypeInferenceError(
                    f"Tensor.{attr}() args must be literals"
                ) from None
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
            output_type = TypeInfo.from_example(
                _CheckForIndexCalls.retry_call(fn, proxy_args, proxy_kwargs), origin
            )
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
        return TypeInfo.from_example(
            _eval_unary(op, self.value),
            origin,
        )

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        return TypeInfo.from_example(getattr(self.value, attr), origin)

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        try:
            return TypeInfo.from_example(self.value[key.as_literal()], origin)  # pyright: ignore[reportIndexIssue]
        except NotImplementedError:
            pass
        return super().propagate_getitem(key, origin)

    def truth_value(self) -> bool:
        return bool(self.value)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if type(other) is type(self) and self.value == other.value:
            return self
        if isinstance(other, (LiteralType, NumericType)):
            if NumericType.known_equal(other.value, self.value):
                return self
            if self.python_type == other.python_type and self.python_type in (
                int,
                float,
                bool,
            ):
                return NumericType.subtype(self.python_type).new_unbacked(self.origin)
        return super().merge(other, var_name=var_name)

    def unpack(self) -> list[TypeInfo]:
        try:
            it = iter(self.value)  # pyright: ignore[reportArgumentType,reportCallIssue]
        except TypeError:
            return super().unpack()
        return [TypeInfo.from_example(x, self.origin) for x in it]

    def as_literal(self) -> object:
        return self.value


class StringType(TypeInfo):
    """TypeInfo for unknown strings (e.g., from f-strings)."""

    def __str__(self) -> str:
        return "str"


class ConfigFragmentType(LiteralType):
    """TypeInfo for config fragments are treated as constant literals during compilation."""

    value: ConfigSpecFragment  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, origin: Origin, fragment: ConfigSpecFragment) -> None:
        assert isinstance(fragment, ConfigSpecFragment)
        super().__init__(origin, fragment)


class CallableType(LiteralType):
    value: Callable[..., object]

    def __init__(self, origin: Origin, value: Callable[..., object]) -> None:
        super().__init__(origin, value)
        self.value = value  # pyright: ignore[reportIncompatibleVariableOverride]

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
                return str(self.value)

    def propagate_call(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo | None:
        if self.value is breakpoint:
            # special handling to prevent breakpoint() from being called during host-code type propagation
            return LiteralType(origin, None)
        if self.value in (torch.nonzero, torch.Tensor.nonzero) and origin.is_device():
            raise exc.DataDependentOutputShapeNotSupported(op_desc="torch.nonzero")
        if self.value in (torch.chunk, torch.Tensor.chunk) and origin.is_device():
            raise exc.UnsupportedSplitOperation(op="torch.chunk")
        if self.value in (torch.unbind, torch.Tensor.unbind) and origin.is_device():
            raise exc.UnsupportedSplitOperation(op="torch.unbind")
        if is_api_func(fn := self.value):
            if fn._is_device_only and origin.is_host():
                raise exc.DeviceAPIOnHost(fn.__qualname__)
            if fn._cache_type and ExtendedAST.current()[-1]._type_info is not None:
                return ExtendedAST.current()[-1]._type_info
            assert fn._type_function is not None
            return fn._type_function(*args, **kwargs, origin=origin)
        # TODO(jansel): add no-tracing mode

        def warn_wrong_device(arg: TypeInfo) -> None:
            if (
                isinstance(arg, TensorType)
                and arg.fake_value.device.type != env.device.type
            ):
                warning(exc.WrongDevice(self.value, arg.fake_value.device, env.device))

        def to_proxy(arg: TypeInfo) -> object:
            if isinstance(arg, TensorType):
                nonlocal input_contains_tensor
                input_contains_tensor = True
                warn_wrong_device(arg)
            return _to_proxy(arg)

        input_contains_tensor: bool = False
        env: CompileEnvironment = CompileEnvironment.current()
        proxy_args = [x.tree_map(to_proxy) for x in args]
        proxy_kwargs = {k: v.tree_map(to_proxy) for k, v in kwargs.items()}

        # special handling for symint arguments
        if any(
            (isinstance(x, torch.SymInt) and not isinstance(x._sympy_(), sympy.Integer))
            for x in proxy_args
        ):
            if self.value in self._new_symint_on_host_fns() and origin.is_host():
                return SymIntType.new_unbacked(origin)
            if isinstance(self.value, type) and issubclass(
                self.value, ConfigFragmentType
            ):
                raise exc.ConfigSpecFragmentWithSymInt(args)

        try:
            with patch.object(torch.SymInt, "__index__", _raise_shape_specializing):
                output_type = TypeInfo.from_example(
                    _CheckForIndexCalls.retry_call(
                        self.value, proxy_args, proxy_kwargs
                    ),
                    origin,
                )
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
        from triton import cdiv
        from triton import next_power_of_2

        return cast("dict[object, None]", dict.fromkeys([cdiv, next_power_of_2]))


def _raise_shape_specializing(*args: object) -> None:
    raise exc.ShapeSpecializingCall


class PythonModuleType(LiteralType):
    value: types.ModuleType

    def __init__(self, origin: Origin, value: types.ModuleType) -> None:
        super().__init__(origin, value)
        self.value = value  # pyright: ignore[reportIncompatibleVariableOverride]

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
        return self.value._sympy_()

    @property
    def python_type(self) -> type[float | int | bool]:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.value})"

    def proxy(self) -> torch.SymInt | torch.SymBool | torch.SymFloat | int:
        return self.value

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        return TypeInfo.from_example(_eval_unary(op, self.value), self.origin)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, (LiteralType, NumericType)):
            if NumericType.known_equal(self.value, other.value):
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
        elif isinstance(right, int | float | bool):
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
    value: torch.SymInt  # pyright: ignore[reportIncompatibleVariableOverride]

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
    value: torch.SymFloat  # pyright: ignore[reportIncompatibleVariableOverride]

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
    value: torch.SymBool  # pyright: ignore[reportIncompatibleVariableOverride]

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


class TileIndexType(TypeInfo):
    block_id: int

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.block_id})"

    def __init__(self, origin: Origin, block_id: int) -> None:
        super().__init__(origin)
        self.block_id = block_id

    def proxy(self) -> object:
        with proxy_tensor.disable_proxy_modes_tracing():
            fake_mode = torch._C._unset_dispatch_mode(  # pyright: ignore[reportAttributeAccessIssue]
                torch._C._TorchDispatchModeKey.FAKE  # pyright: ignore[reportAttributeAccessIssue]
            )
            try:
                with torch._C._DisableTorchDispatch():  # pyright: ignore[reportAttributeAccessIssue]
                    return Tile(self.block_id)
            finally:
                assert fake_mode is not None
                torch._C._set_dispatch_mode(fake_mode)  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def allocate(
        numel: int | torch.SymInt | AutoSize | None,
        origin: Origin,
        block_size: int | torch.SymInt | None = None,
    ) -> TileIndexType:
        env = CompileEnvironment.current()
        if block_size is None:
            block_id = env.allocate_block_size(numel, source=LoopSpecBlockSizeSource())
            env.config_spec.block_sizes.append(
                BlockSizeSpec(
                    block_id=block_id,
                    size_hint=_get_hint(numel),
                )
            )
        else:
            block_id = env.allocate_block_size(
                numel, source=FixedBlockSizeSource(block_size)
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
    ) -> None:
        super().__init__(origin)
        self.element_types = element_types

    @property
    def python_type(self) -> type[object]:
        return type(self.element_types)

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        if isinstance(op, ast.Not):
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
                    elements[k] = elements[k].merge(value)  # pyright: ignore[reportArgumentType,reportCallIssue]
                else:
                    elements[k] = value  # pyright: ignore[reportArgumentType,reportCallIssue]
                return self
        return super().propagate_setitem(key, value, origin)

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        try:
            literal_key = key.as_literal()
        except NotImplementedError:
            pass
        else:
            try:
                result = self.element_types[literal_key]  # pyright: ignore[reportArgumentType,reportCallIssue,reportIndexIssue]
            except (KeyError, IndexError) as e:
                raise exc.TypeInferenceError(f"{type(e).__name__}: {e}") from None
            if isinstance(result, TypeInfo):
                return result
            if type(result) is self.python_type:  # sliced!
                return type(self)(origin=origin, element_types=result)  # pyright: ignore[reportArgumentType]
        return super().propagate_getitem(key, origin)

    def truth_value(self) -> bool:
        return bool(self.element_types)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> object:
        raise NotImplementedError


class SequenceType(CollectionType):
    element_types: list[TypeInfo] | tuple[TypeInfo, ...]  # pyright: ignore[reportIncompatibleVariableOverride]

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
        return self._maybe_tuple([x.proxy() for x in self.element_types])

    def as_literal(self) -> list[object] | tuple[object, ...]:
        return self._maybe_tuple([x.as_literal() for x in self.element_types])

    def unpack(self) -> list[TypeInfo]:
        return [*self.element_types]

    def populate_symbol_origins(self, origin: Origin) -> None:
        for i, subtype in enumerate(self.element_types):
            subtype.populate_symbol_origins(GetItemOrigin(origin, i))

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        return super().propagate_getitem(key, origin)

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, SequenceType):
            self_elements = self.element_types
            other_elements = other.element_types
            if len(self_elements) == len(other_elements):
                return SequenceType(
                    origin=other.origin,
                    element_types=self._maybe_tuple(
                        [
                            self_elements[i].merge(other_elements[i], var_name=var_name)
                            for i in range(len(self_elements))
                        ]
                    ),
                )
        return super().merge(other, var_name=var_name)

    def tree_map(
        self, fn: Callable[[TypeInfo], object]
    ) -> list[object] | tuple[object, ...]:
        return self._maybe_tuple([x.tree_map(fn) for x in self.element_types])


class DictType(CollectionType):
    element_types: dict[str | int, TypeInfo]  # pyright: ignore[reportIncompatibleVariableOverride]

    def __str__(self) -> str:
        items = ", ".join(f"{k!r}: {v!s}" for k, v in self.element_types.items())
        return f"{type(self).__name__}({{{items}}})"

    def proxy(self) -> dict[str | int, object]:
        return {k: v.proxy() for k, v in self.element_types.items()}

    def as_literal(self) -> dict[str | int, object]:
        return {k: v.as_literal() for k, v in self.element_types.items()}

    def unpack(self) -> list[TypeInfo]:
        return [TypeInfo.from_example(k, self.origin) for k in self.element_types]

    def populate_symbol_origins(self, origin: Origin) -> None:
        for k, subtype in self.element_types.items():
            subtype.populate_symbol_origins(GetItemOrigin(origin, k))

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, DictType):
            self_elements = self.element_types
            other_elements = other.element_types
            if set(self_elements.keys()) == set(other_elements.keys()):
                return DictType(
                    origin=other.origin,
                    element_types={
                        key: self_elements[key].merge(
                            other_elements[key], var_name=var_name
                        )
                        for key in self_elements
                    },
                )
        return super().merge(other, var_name=var_name)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> dict[str | int, object]:
        return {k: v.tree_map(fn) for k, v in self.element_types.items()}


class ClassType(DictType):
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
    element_types: dict[str, TypeInfo]  # pyright: ignore[reportIncompatibleVariableOverride]

    def proxy(self) -> StackTensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        with proxy_tensor.disable_proxy_modes_tracing():
            fake_mode = torch._C._unset_dispatch_mode(  # pyright: ignore[reportAttributeAccessIssue]
                torch._C._TorchDispatchModeKey.FAKE  # pyright: ignore[reportAttributeAccessIssue]
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
                torch._C._set_dispatch_mode(fake_mode)  # pyright: ignore[reportAttributeAccessIssue]

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if isinstance(other, StackTensorType):
            self_elements = self.element_types
            other_elements = other.element_types
            if set(self_elements.keys()) == set(other_elements.keys()):
                return StackTensorType(
                    origin=other.origin,
                    element_types={
                        key: self_elements[key].merge(
                            other_elements[key], var_name=var_name
                        )
                        for key in self_elements
                    },
                )
        return super().merge(other, var_name=var_name)

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


class SliceType(CollectionType):
    element_types: slice  # pyright: ignore[reportIncompatibleVariableOverride]

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


def _eval_unary(op: ast.unaryop, value: object) -> object:
    if isinstance(op, ast.Not):
        return not value
    if isinstance(op, ast.UAdd):
        return +value  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.USub):
        return -value  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Invert):
        return ~value  # pyright: ignore[reportOperatorIssue]
    raise AssertionError(f"{type(op).__name__} unknown unary op")


def _eval_binary(op: ast.operator, left: object, right: object) -> object:
    if isinstance(op, ast.Add):
        return left + right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Sub):
        return left - right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Mult):
        return left * right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Div):
        return left / right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.FloorDiv):
        return left // right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Mod):
        return left % right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Pow):
        return left**right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.LShift):
        return left << right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.RShift):
        return left >> right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.BitOr):
        return left | right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.BitXor):
        return left ^ right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.BitAnd):
        return left & right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.MatMult):
        return left @ right  # pyright: ignore[reportOperatorIssue]
    raise AssertionError(f"{type(op).__name__} unknown binary op")


def _eval_compare(op: ast.cmpop, left: object, right: object) -> object:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.LtE):
        return left <= right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Gt):
        return left > right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.GtE):
        return left >= right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.Is):
        return left is right
    if isinstance(op, ast.IsNot):
        return left is not right
    if isinstance(op, ast.In):
        return left in right  # pyright: ignore[reportOperatorIssue]
    if isinstance(op, ast.NotIn):
        return left not in right  # pyright: ignore[reportOperatorIssue]
    raise AssertionError(f"{type(op).__name__} unknown compare op")


CMP_BASIC = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)
CMP_IS = (
    ast.Is,
    ast.IsNot,
)
CMP_IN = (
    ast.In,
    ast.NotIn,
)
CMP_ALWAYS_BOOL: tuple[type[ast.AST], ...] = (
    *CMP_IS,
    *CMP_IN,
)


def _unsupported(
    python_type: type[object],
) -> Callable[[TypePropagation, ast.AST], NoReturn]:
    def visit(self: TypePropagation, node: ast.AST) -> NoReturn:
        super(TypePropagation, self).generic_visit(node)
        raise exc.UnsupportedPythonType(python_type.__name__)

    return visit


class TypePropagation(ast.NodeVisitor):
    def __init__(self, func: HostFunction, scope: LocalScope) -> None:
        super().__init__()
        self.func = func
        self.scope = scope
        self.device_loop_depth = 0
        self.device_loop_count = 0

    def push_scope(self) -> None:
        self.scope = LocalScope(parent=self.scope)

    def pop_scope_merge(self) -> None:
        parent = self.scope.parent
        assert isinstance(parent, LocalScope)
        parent.merge(self.scope)
        self.scope = parent

    def pop_scope_overwrite(self) -> None:
        parent = self.scope.parent
        assert isinstance(parent, LocalScope)
        parent.overwrite(self.scope)
        self.scope = parent

    def pop_scope(self) -> LocalScope:
        current = self.scope
        parent = current.parent
        assert isinstance(parent, LocalScope)
        self.scope = parent
        return current

    @contextlib.contextmanager
    def swap_scope(self, new_scope: LocalScope) -> Iterator[None]:
        prior = self.scope
        self.scope = new_scope
        try:
            yield
        finally:
            self.scope = prior

    def visit(self, node: ast.AST) -> TypeInfo:
        assert isinstance(node, ExtendedAST)
        with node:
            try:
                visitor = getattr(
                    self,
                    f"visit_{node.__class__.__name__}",
                    self.generic_visit,
                )
                type_info = visitor(node)
                assert isinstance(type_info, TypeInfo), (
                    f"expected TypeInfo, got {type_info!r} from {visitor!r}"
                )
                return node.update_type_info(type_info)
            except exc.Base:
                raise
            except Exception as e:
                raise exc.InternalError(e) from e

    def origin(self) -> Origin:
        if self.device_loop_depth == 0:
            return SourceOrigin(current_location())
        return DeviceOrigin(current_location())

    def generic_visit(self, node: ast.AST) -> TypeInfo:
        super().generic_visit(node)
        raise exc.UnsupportedPythonType(f"ast.{node.__class__.__name__}")

    def _bool_op(self, op: ast.boolop, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        try:
            val = left.truth_value()
            if isinstance(op, ast.Or):
                return left if val is True else right
            if isinstance(op, ast.And):
                return right if val is True else left
        except NotImplementedError:
            pass
        if (
            isinstance(left, (NumericType, LiteralType))
            and isinstance(right, (NumericType, LiteralType))
            and left.python_type == right.python_type
            and (pt := left.python_type) in (int, float, bool)
        ):
            return NumericType.subtype(pt).new_unbacked(self.origin())
        raise exc.TypeInferenceError(
            f"{type(op).__name__} not supported on {left!s} and {right!s}"
        )

    def _compare(self, op: ast.cmpop, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        if isinstance(left, LiteralType) and isinstance(right, LiteralType):
            return LiteralType(
                origin=self.origin(),
                value=_eval_compare(op, left.value, right.value),
            )
        if (
            isinstance(left, LiteralType)
            and isinstance(right, CollectionType)
            and isinstance(op, CMP_IN)
        ):
            return LiteralType(
                origin=self.origin(),
                value=_eval_compare(op, left.value, right.element_types),
            )
        if isinstance(left, (NumericType, LiteralType)) and isinstance(
            right,
            (NumericType, LiteralType),
        ):
            return SymBoolType.new_unbacked(self.origin())
        if isinstance(op, CMP_ALWAYS_BOOL):
            return SymBoolType.new_unbacked(self.origin())
        if isinstance(left, TensorType) or isinstance(right, TensorType):
            try:
                left_example = left.proxy()
                right_example = right.proxy()
            except NotImplementedError:
                pass
            else:
                try:
                    return TypeInfo.from_example(
                        _eval_compare(op, left_example, right_example),
                        self.origin(),
                    )
                except exc.Base:
                    raise
                except Exception as e:
                    raise exc.TorchOpTracingError(e) from e
        if (
            isinstance(left, SequenceType)
            and isinstance(right, SequenceType)
            and isinstance(op, ast.Eq)
        ):
            if len(left.element_types) != len(right.element_types):
                return LiteralType(origin=self.origin(), value=False)

            can_determine_statically = True
            all_elements_equal = True
            for left_elem, right_elem in zip(
                left.element_types, right.element_types, strict=False
            ):
                if isinstance(left_elem, LiteralType) and isinstance(
                    right_elem, LiteralType
                ):
                    if left_elem.value != right_elem.value:
                        all_elements_equal = False
                        break
                elif isinstance(left_elem, (NumericType, LiteralType)) and isinstance(
                    right_elem, (NumericType, LiteralType)
                ):
                    if NumericType.known_equal(left_elem.value, right_elem.value):
                        continue
                    can_determine_statically = False
                    break
                else:
                    can_determine_statically = False
                    break

            if can_determine_statically:
                return LiteralType(origin=self.origin(), value=all_elements_equal)
            return SymBoolType.new_unbacked(self.origin())
        raise exc.TypeInferenceError(
            f"{type(op).__name__} not supported on {left!s} and {right!s}"
        )

    def _assign(self, lhs: ast.AST, rhs: TypeInfo) -> None:
        if isinstance(lhs, ast.Name):
            # Check if we're trying to modify a host variable inside a device loop
            if (
                (existing_type := self.scope.maybe_get(lhs.id)) is not None
                and existing_type.origin.is_host()
                and rhs.origin.is_device()
            ):
                raise exc.CannotModifyHostVariableOnDevice(lhs.id) from None
            if isinstance(rhs, TileIndexType):
                CompileEnvironment.current().block_sizes[rhs.block_id].add_debug_name(
                    lhs.id
                )
            return self.scope.set(lhs.id, rhs)
        if isinstance(lhs, ast.Starred):
            try:
                unpacked = SequenceType(self.origin(), rhs.unpack())
            except NotImplementedError:
                raise exc.TypeInferenceError(
                    f"Failed to unpack starred assignment: {rhs!s}"
                ) from None
            return self._assign(lhs.value, unpacked)
        if isinstance(lhs, (ast.Tuple, ast.List)):
            lhs = lhs.elts  # pyright: ignore[reportAssignmentType]
            elements: list[TypeInfo]
            try:
                elements = rhs.unpack()
            except NotImplementedError:
                if isinstance(rhs, TileIndexType):
                    raise exc.FailedToUnpackTile from None
                raise exc.FailedToUnpackTupleAssign(len(lhs), rhs) from None  # pyright: ignore[reportArgumentType]
            used_star = False
            idx = 0
            for elt in lhs:  # pyright: ignore[reportGeneralTypeIssues]
                if isinstance(elt, ast.Starred):
                    # TODO(jansel): need to test this
                    assert not used_star, "multiple `*` in assignment"
                    used_star = True
                    star_len = len(elements) - len(lhs) + 1  # pyright: ignore[reportArgumentType]
                    assert star_len >= 0, "wrong number of elements to unpack"
                    self._assign(
                        elt.value,
                        SequenceType(self.origin(), elements[idx : idx + star_len]),
                    )
                    idx += star_len
                else:
                    self._assign(elt, elements[idx])
                    idx += 1
            assert idx == len(elements), "wrong number of elements to unpack"
            return None
        if isinstance(lhs, ast.Subscript):
            # TODO(jansel): test different types of subscript
            lhs_base_type = self.visit(lhs.value)
            if isinstance(lhs_base_type, (TensorType, StackTensorType)):
                self.visit(lhs)  # need to populate shape info
            lhs_base_type = lhs_base_type.propagate_setitem(
                self.visit(lhs.slice), rhs, self.origin()
            )
            # update the stored type for the container
            return self._assign(lhs.value, lhs_base_type)
        raise AssertionError(f"unhandled lhs in assignment {type(lhs).__name__}")

    ################################################################
    # Expressions
    ################################################################

    def visit_Constant(self, node: ast.Constant) -> TypeInfo:
        return LiteralType(value=node.value, origin=self.origin())

    def visit_FormattedValue(self, node: ast.FormattedValue) -> TypeInfo:
        # Visit the value expression for type checking, but ignore the result
        self.visit(node.value)
        # Visit the format spec if present
        if node.format_spec is not None:
            self.visit(node.format_spec)
        # Return StringType for unknown strings
        return StringType(origin=self.origin())

    def visit_JoinedStr(self, node: ast.JoinedStr) -> TypeInfo:
        # Visit all values for type checking
        for value in node.values:
            self.visit(value)
        # Check if all values are string constants
        all_constants = True
        constant_parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                constant_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                all_constants = False
                break
            else:
                all_constants = False
                break

        if all_constants:
            # If all parts are string constants, return a LiteralType
            return LiteralType(value="".join(constant_parts), origin=self.origin())
        # Otherwise, return StringType for unknown strings
        return StringType(origin=self.origin())

    def _list_or_tuple(self, node: ast.List | ast.Tuple) -> TypeInfo:
        elements = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                to_unpack = self.visit(elt.value)
                try:
                    elements.extend(to_unpack.unpack())
                except NotImplementedError:
                    raise exc.TypeInferenceError(
                        f"Failed to unpack starred assignment: {to_unpack!s}"
                    ) from None
            else:
                elements.append(self.visit(elt))
        cls = list if isinstance(node, ast.List) else tuple
        return SequenceType(
            self.origin(),
            cls(elements),
        )

    visit_List: _VisitMethod = _list_or_tuple  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Tuple: _VisitMethod = _list_or_tuple  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Set: _VisitMethod = _unsupported(set)  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    def visit_Dict(self, node: ast.Dict) -> TypeInfo:
        assert len(node.keys) == len(node.values)
        element_types = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            value = self.visit(value_node)
            if key_node is not None:
                key = self.visit(key_node)
                if not (
                    isinstance(key, LiteralType) and isinstance(key.value, (str, int))
                ):
                    raise exc.TypeInferenceError(
                        f"Only string/int literals are supported as dict keys, got {key!s}"
                    )
                element_types[key.value] = value
            else:
                if not (isinstance(value, DictType)):
                    raise exc.TypeInferenceError(
                        f"Only collection types are supported as dict** values, got {value!s}"
                    )
                element_types.update(value.element_types)
        return DictType(element_types=element_types, origin=self.origin())

    def visit_Name(self, node: ast.Name) -> TypeInfo:
        result = self.scope.get(node.id)
        if self.device_loop_depth == 0 and result.origin.is_device():
            raise exc.CannotReadDeviceVariableOnHost(node.id)
        return result

    visit_Starred: _VisitMethod = generic_visit  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    def visit_Expr(self, node: ast.Expr) -> TypeInfo:
        return self.visit(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> TypeInfo:
        return self.visit(node.operand).propagate_unary(node.op, self.origin())

    def visit_BinOp(self, node: ast.BinOp) -> TypeInfo:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if (
            isinstance(left, TensorType) or isinstance(right, TensorType)
        ) and self.device_loop_depth == 0:
            warning(exc.TensorOperationInWrapper)

        try:
            left_example = left.proxy()
            right_example = right.proxy()
        except NotImplementedError:
            pass
        else:
            try:
                # Special case: if this is Tile + offset pattern, expand to tile.index + offset
                if (
                    isinstance(node.op, ast.Add)
                    and isinstance(left, TileIndexType)
                    and isinstance(right, (SymIntType, LiteralType, NumericType))
                ):
                    # Expand tile + offset to tile.index + offset
                    tile_index = left.propagate_attribute(
                        "index", AttributeOrigin(self.origin(), "index")
                    )
                    return TypeInfo.from_example(
                        _eval_binary(node.op, tile_index.proxy(), right.proxy()),
                        self.origin(),
                    )

                return TypeInfo.from_example(
                    _eval_binary(node.op, left_example, right_example),
                    self.origin(),
                )
            except exc.Base:
                raise
            except TypeError as e:
                # Re-raise as TorchOpTracingError for proper error handling in visit_AugAssign
                raise exc.TorchOpTracingError(e) from e
            except RuntimeError as e:
                # Re-raise as TorchOpTracingError for proper error handling in visit_AugAssign
                raise exc.TorchOpTracingError(e) from e

        raise exc.TypeInferenceError(
            f"{type(node.op).__name__} not supported on {left!s} and {right!s}"
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> TypeInfo:
        values = [self.visit(node.values[0])]
        for value in node.values[1:]:
            # Everything after first node is conditionally executed
            self.push_scope()
            values.append(self.visit(value))
            self.pop_scope_merge()

        result = values[0]
        for value in values[1:]:
            result = self._bool_op(node.op, result, value)
        return result

    def visit_Compare(self, node: ast.Compare) -> TypeInfo:
        comparators = [
            self.visit(node.left),
            *[self.visit(comparator) for comparator in node.comparators],
        ]
        if (
            any(isinstance(comparator, TensorType) for comparator in comparators)
            and self.device_loop_depth == 0
        ):
            warning(exc.TensorOperationInWrapper)
        result = self._compare(node.ops[0], comparators[0], comparators[1])
        for i in range(2, len(comparators)):
            new_result = self._compare(
                node.ops[i - 1],
                comparators[i - 1],
                comparators[i],
            )
            result = self._bool_op(ast.And(), result, new_result)
        return result

    def visit_Call(self, node: ast.Call) -> TypeInfo:
        # TODO(jansel): test handling if *args and **kwargs
        func = self.visit(node.func)

        # Check for calling a Helion kernel from within another Helion kernel
        if isinstance(func, CallableType) and isinstance(func.value, Kernel):
            raise exc.NestedKernelCallsNotSupported

        if (
            isinstance(func, CallableType)
            and self.origin().is_device()
            and (replacement := get_device_func_replacement(func.value))
        ):
            func = CallableType(func.origin, replacement)

        unhandled = []
        args = []
        kwargs = {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                arg_type = self.visit(arg.value)
                if isinstance(arg_type, SequenceType):
                    args.extend(arg_type.element_types)
                else:
                    unhandled.append(arg_type)
            else:
                args.append(self.visit(arg))
        for kwarg in node.keywords:
            if kwarg.arg is None:
                kwarg_type = self.visit(kwarg.value)
                if isinstance(kwarg_type, DictType):
                    kwargs.update(kwarg_type.element_types)
                else:
                    unhandled.append(kwarg_type)
            else:
                kwargs[kwarg.arg] = self.visit(kwarg.value)
        if unhandled:
            raise exc.TypeInferenceError(
                "Failed to unpack */** args to function, got: "
                + ", ".join(map(str, unhandled))
            )
        return func.propagate_call(tuple(args), kwargs, self.origin())  # pyright: ignore[reportReturnType]

    def visit_IfExp(self, node: ast.IfExp) -> TypeInfo:
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        try:
            truth_val = test.truth_value()
            if truth_val:
                return body
            return orelse
        except NotImplementedError:
            pass
        return body.merge(orelse)

    def visit_Attribute(self, node: ast.Attribute) -> TypeInfo:
        value = self.visit(node.value)
        origin = AttributeOrigin(value.origin, node.attr)
        return value.propagate_attribute(node.attr, origin)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> TypeInfo:
        # x := y
        type_info = self.visit(node.value)
        self._assign(node.target, type_info)
        return type_info

    def visit_Subscript(self, node: ast.Subscript) -> TypeInfo:
        value_type = self.visit(node.value)
        slice_type = self.visit(node.slice)
        return value_type.propagate_getitem(slice_type, self.origin())

    def visit_Slice(self, node: ast.Slice) -> TypeInfo:
        lower = (
            self.visit(node.lower)
            if node.lower is not None
            else LiteralType(self.origin(), None)
        )
        upper = (
            self.visit(node.upper)
            if node.upper is not None
            else LiteralType(self.origin(), None)
        )
        step = (
            self.visit(node.step)
            if node.step is not None
            else LiteralType(self.origin(), None)
        )
        return SliceType(self.origin(), slice(lower, upper, step))

    ################################################################
    # Statements
    ################################################################

    def generic_statement(self, node: ast.AST) -> TypeInfo:
        self.generic_visit(node)
        return NoType(origin=self.origin())

    def visit_Assign(self, node: ast.Assign) -> TypeInfo:
        type_info = self.visit(node.value)
        for target in node.targets:
            self._assign(target, type_info)
        return NoType(origin=self.origin())

    def visit_AnnAssign(self, node: ast.AnnAssign) -> TypeInfo:
        # TODO(jansel): handle constexpr in annotation
        if node.value is not None:
            type_info = self.visit(node.value)
            self._assign(node.target, type_info)
        return NoType(origin=self.origin())

    def visit_AugAssign(self, node: ast.AugAssign) -> TypeInfo:
        assert isinstance(node.target, ExtendedAST)
        try:
            type_info = self.visit(
                create(
                    ast.BinOp,
                    left=node.target.copy(ctx=ast.Load()),
                    op=node.op,
                    right=node.value,
                )
            )
        except exc.TorchOpTracingError as e:
            # Check if this is a shape mismatch when modifying a host variable in device loop
            if (
                isinstance(node.target, ast.Name)
                and (existing_type := self.scope.maybe_get(node.target.id)) is not None
                and existing_type.origin.is_host()
                and self.device_loop_depth > 0
            ):
                raise exc.CannotModifyHostVariableOnDevice(node.target.id) from e
            raise
        self._assign(node.target, type_info)
        return NoType(origin=self.origin())

    def visit_Assert(self, node: ast.Assert) -> TypeInfo:
        # Visit the test expression for type checking, but ignore the result
        self.visit(node.test)
        # Visit the optional message expression if present
        if node.msg is not None:
            self.visit(node.msg)
        return NoType(origin=self.origin())

    visit_Raise: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Delete: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Pass: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_TypeAlias: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    visit_Import: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_ImportFrom: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    def visit_Global(self, node: ast.Global) -> TypeInfo:
        # Global statements don't need child visiting since they only declare names
        return NoType(origin=self.origin())

    # TODO(jansel): support lambda
    visit_Lambda: _VisitMethod = generic_visit  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    ################################################################
    # Control flow
    ################################################################

    def visit_If(self, node: ast.If) -> TypeInfo:
        test = self.visit(node.test)
        try:
            truth_val = test.truth_value()
            has_truth_val = True
        except NotImplementedError:
            truth_val = None
            has_truth_val = False
        if has_truth_val:
            # For constant conditions, only type propagate one branch
            self.scope.merge(self._body(node.body if truth_val else node.orelse))
        else:
            self.scope.merge_if_else(self._body(node.body), self._body(node.orelse))
        return NoType(origin=self.origin())

    def _body(self, stmts: list[ast.stmt]) -> LocalScope:
        self.push_scope()
        for stmt in stmts:
            self.visit(stmt)
        return self.pop_scope()

    def _loop_body(self, stmts: list[ast.stmt]) -> LocalScope:
        self.push_scope()
        exit_scopes = [self.scope]
        for stmt in stmts:
            self.visit(stmt)
            if isinstance(stmt, (ast.Break, ast.Continue)):
                exit_scopes.append(self.scope.clone())
        self.pop_scope()
        return functools.reduce(lambda x, y: x.merge(y), exit_scopes)

    def visit_For(self, node: ast.For) -> TypeInfo:
        parent_scope = self.scope
        self.push_scope()
        iter_type = self.visit(node.iter)
        self._assign(node.target, iter_type.propagate_iter(self.origin()))
        device_loop = (
            isinstance(call_node := node.iter, ast.Call)
            and isinstance(fn_node := call_node.func, ExtendedAST)
            and isinstance(fn_type := fn_node._type_info, CallableType)
            and is_api_func(fn := fn_type.value)
            and fn._is_device_loop
        )

        assert isinstance(node, ExtendedAST)
        node._loop_type = (
            LoopType.HOST if self.device_loop_depth == 0 else LoopType.DEVICE
        )
        if device_loop:
            if node.orelse:
                raise exc.DeviceLoopElseBlock(fn.__qualname__)  # pyright: ignore[reportPossiblyUnboundVariable]

            if self.device_loop_depth == 0:
                self.func.set_local_types(parent_scope.extract_locals())
                node._loop_type = LoopType.GRID
                node._root_id = self.device_loop_count
                self.device_loop_count += 1
                if len(ExtendedAST.current()) != 1:
                    raise exc.NestedGridLoop

        self.device_loop_depth += device_loop
        _maybe_patch_tensor_factories = (
            patch_tensor_factories
            if self.device_loop_depth > 0
            else contextlib.nullcontext
        )
        with _maybe_patch_tensor_factories():
            body = self._loop_body(node.body)
            with self.swap_scope(body):
                # second pass for fixed point
                body.merge(self._loop_body(node.body))
            orelse = self._body(node.orelse)
            self.scope.merge_if_else(body, orelse)
        self.device_loop_depth -= device_loop
        return NoType(origin=self.origin())

    def visit_While(self, node: ast.While) -> TypeInfo:
        self.visit(node.test)
        body = self._loop_body(node.body)
        with self.swap_scope(body):
            # second pass for fixed point
            self.visit(node.test)
            body.merge(self._loop_body(node.body))
        orelse = self._body(node.orelse)
        self.scope.merge_if_else(body, orelse)
        return NoType(origin=self.origin())

    visit_Break: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Continue: _VisitMethod = generic_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    def visit_Try(self, node: ast.Try) -> TypeInfo:
        self.scope.merge(self._body(node.body))
        for handler in node.handlers:
            self.push_scope()
            self.visit(handler)
            self.pop_scope_merge()
        self.scope.merge(self._body(node.orelse))
        self.scope.overwrite(self._body(node.finalbody))
        return NoType(origin=self.origin())

    visit_TryStar: _VisitMethod = visit_Try  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]

    def _not_on_device_statement(self, node: ast.AST) -> TypeInfo:
        if self.device_loop_depth:
            raise exc.NotAllowedOnDevice(type(node).__name__)
        for child_node in ast.iter_child_nodes(node):
            self.visit(child_node)
        return NoType(origin=self.origin())

    visit_ExceptHandler: _VisitMethod = _not_on_device_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_With: _VisitMethod = _not_on_device_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Return: _VisitMethod = _not_on_device_statement  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    def _not_supported(self, node: ast.AST) -> TypeInfo:
        raise exc.StatementNotSupported(type(node).__name__)

    def _evaluate_comprehension(
        self, generator: ast.comprehension, expression: ast.AST
    ) -> TypeInfo:
        """Helper method to evaluate comprehension type propagation."""
        # Visit the iterable to get its type
        iter_type = self.visit(generator.iter)

        # Get element type and evaluate expression in scope
        self.push_scope()
        try:
            element_type = iter_type.propagate_iter(self.origin())
            self._assign(generator.target, element_type)

            # Process conditional filters (basic validation)
            for if_clause in generator.ifs:
                self.visit(if_clause)

            element_result_type = self.visit(expression)
        finally:
            self.pop_scope()

        # Try to determine exact result by unpacking iterable
        try:
            iterable_elements = iter_type.unpack()
            result_elements = []

            for element_type in iterable_elements:
                self.push_scope()
                try:
                    self._assign(generator.target, element_type)
                    # For now, assume all conditions pass
                    for if_clause in generator.ifs:
                        self.visit(if_clause)
                    result_elements.append(self.visit(expression))
                finally:
                    self.pop_scope()

            # If there are conditions, we can't determine exact length
            if generator.ifs and result_elements:
                result_elements = [result_elements[0]]

            return SequenceType(self.origin(), result_elements)

        except NotImplementedError:
            # Fallback to generic list type
            return SequenceType(self.origin(), [element_result_type])

    def visit_ListComp(self, node: ast.ListComp) -> TypeInfo:
        """Type propagation for list comprehensions."""
        if len(node.generators) != 1:
            raise exc.StatementNotSupported(
                "List comprehensions with multiple generators are not supported"
            )

        return self._evaluate_comprehension(node.generators[0], node.elt)

    # TODO(jansel): need to implement these
    visit_SetComp: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_GeneratorExp: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_DictComp: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    # TODO(jansel): support closure functions defined on host
    visit_FunctionDef: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]

    visit_ClassDef: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Yield: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_YieldFrom: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_AsyncFunctionDef: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_AsyncFor: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_AsyncWith: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Await: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_Match: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchValue: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchSingleton: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchSequence: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchStar: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchMapping: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchClass: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchAs: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    visit_MatchOr: _VisitMethod = _not_supported  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]


def _to_proxy(arg: TypeInfo) -> object:
    try:
        return arg.proxy()
    except NotImplementedError:
        raise exc.TracedArgNotSupported(arg) from None


def propagate_types(func: HostFunction) -> None:
    # Lock needed since patch.object(torch.SymInt.__index__, ...) is not thread safe
    with compile_lock, func, enable_python_dispatcher():
        global_scope = GlobalScope(function=func)
        local_scope = LocalScope(parent=global_scope)
        for name, value in func.params.arguments.items():
            # TODO(jansel): handle specializations/constexpr
            type_info = TypeInfo.from_example(
                value,
                ArgumentOrigin(name=name, function=func),
            )
            local_scope.set(name, type_info)
        assert not func.fn.__closure__
        prop = TypePropagation(func, local_scope)

        seen_for_loop = False
        seen_non_for_loop_statement_after_for_loop = False
        for stmt in func.body:
            if isinstance(stmt, ast.For):
                if seen_for_loop and seen_non_for_loop_statement_after_for_loop:
                    # TODO(oulgen): This check is too coarse, refine it.
                    raise exc.TopLevelStatementBetweenLoops
                seen_for_loop = True
            elif seen_for_loop:
                seen_non_for_loop_statement_after_for_loop = True
            prop.visit(stmt)
