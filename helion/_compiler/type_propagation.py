from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import functools
import inspect
import re
import types
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import Protocol
from typing import TypeVar
from unittest.mock import patch

import sympy
import torch
from torch._dynamo.convert_frame import compile_lock
from torch.fx.experimental import proxy_tensor
from torch.utils._pytree import tree_map_only

from .. import exc
from ..autotuner.config_spec import BlockSizeSpec
from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import create
from .compile_environment import AutoSize
from .compile_environment import CompileEnvironment
from .compile_environment import FixedBlockSizeSource
from .compile_environment import LoopSpecBlockSizeSource
from .compile_environment import warning
from .host_function import HostFunction
from .host_function import SymbolOrigin
from .output_header import library_imports
from .source_location import SourceLocation
from .source_location import current_location
from .tile_index_proxy import CheckForIndexCalls
from .tile_index_proxy import TileIndexProxy
from .variable_origin import ArgumentOrigin
from .variable_origin import AttributeOrigin
from .variable_origin import BuiltinOrigin
from .variable_origin import DeviceOrigin
from .variable_origin import GetItemOrigin
from .variable_origin import GlobalOrigin
from .variable_origin import Origin
from .variable_origin import SourceOrigin
from .variable_origin import TensorSizeOrigin
import helion

# pyre-ignore-all-errors[8,15,58]: visit_* overrides
if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from typing_extensions import Self

    class _VisitMethod(Protocol):
        @staticmethod
        def __call__(self: object, node: ast.AST) -> TypeInfo: ...

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
            if value is helion.language:
                origin = GlobalOrigin(name="hl", function=self.function)
                return TypeInfo.from_example(value, origin)
            if name in library_imports:
                origin = GlobalOrigin(name=name, function=self.function)
                return TypeInfo.from_example(value, origin)

            origin = self.function.global_scope_origin(name)
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

    def set(self, name: str, type_info: TypeInfo) -> None:
        self.variables[name] = type_info

    def merge(self, other: LocalScope | dict[str, TypeInfo]) -> LocalScope:
        if isinstance(other, LocalScope):
            other = other.variables
        for k, v in other.items():
            if k in self.variables:
                existing = self.variables[k]
                merged = existing.merge(v)
                if (
                    isinstance(merged, UnknownType)
                    and not isinstance(existing, UnknownType)
                    and not isinstance(v, UnknownType)
                ):
                    # Improve error message
                    merged = UnknownType(
                        merged.origin,
                        f"Variable {k!r} has different types in control flow: {existing!s} and {v!s}",
                    )
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
                both[k] = true.pop(k).merge(false.pop(k))
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
                return UnknownType(
                    debug_msg="Only int/string keys are supported in dict",
                    origin=origin,
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
                        value._fields,  # pyre-ignore[16]
                        cls._unpack_example(
                            value._asdict().items(),  # pyre-ignore[16]
                            origin,
                        ),
                        strict=False,
                    )
                ),
            )
        if dataclasses.is_dataclass(value):
            keys = value.__dataclass_fields__.keys()  # pyre-ignore[16]
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
        return UnknownType(
            debug_msg=f"{type(value).__name__} is not supported",
            origin=origin,
        )

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

    def merge(self, other: TypeInfo) -> TypeInfo:
        """Combine two types at a join point in control flow."""
        if isinstance(other, UnknownType):
            return other
        if isinstance(other, NoType) or self == other:
            return self
        return UnknownType(
            debug_msg=f"Can't combine types from control flow: {self!s} and {other!s}",
            origin=other.origin,
        )

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        return UnknownType(
            debug_msg=f"{type(op).__name__} not supported on {self!s}",
            origin=origin,
        )

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        return UnknownType(
            debug_msg=f"Function calls are not supported on {self!s}",
            origin=origin,
        )

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        return UnknownType(
            debug_msg=f"Attributes are not supported on {self!s}",
            origin=origin,
        )

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        """Should return updated type of self after running `self[key] = value`"""
        return UnknownType(
            debug_msg=f"Subscript assignment not supported with self={self!s} key={key!s} value={value!s}",
            origin=origin,
        )

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        """Should return updated type of self after running `self[key] = value`"""
        return UnknownType(
            debug_msg=f"Subscript not supported with self={self!s} key={key!s}",
            origin=origin,
        )

    def propagate_iter(self, origin: Origin) -> TypeInfo:
        try:
            values = self.unpack()
        except NotImplementedError:
            pass
        else:
            for val in values:
                if isinstance(val, UnknownType):
                    return val.chained(origin)
            return functools.reduce(lambda x, y: x.merge(y), values)
        return UnknownType(
            debug_msg=f"Iteration over {self!s} is not supported",
            origin=origin,
        )

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
                assert str(k.proxy()) == "slice(None, None, None)"
                size = self.fake_value.size(inputs_consumed)
                inputs_consumed += 1
                if self.origin.is_device():
                    output_sizes.append(size)
                elif size != 1:
                    rdim = CompileEnvironment.current().allocate_reduction_dimension(
                        size
                    )
                    output_sizes.append(rdim.var)
                else:
                    output_sizes.append(1)
            elif isinstance(k, TileIndexType):
                inputs_consumed += 1
                output_sizes.append(env.block_sizes[k.block_size_idx].var)
            elif isinstance(k, TypeNotAllowedOnDevice):
                raise exc.TypePropagationError(k)
            elif isinstance(k, TensorType) and k.fake_value.ndim == 1:
                inputs_consumed += 1
                output_sizes.append(k.fake_value.size(0))
            else:
                raise exc.InvalidIndexingType(k)
        if inputs_consumed != self.fake_value.ndim:
            raise exc.RankMismatch(self.fake_value.ndim, inputs_consumed)
        return output_sizes

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if origin.is_host():
            warning(exc.TensorOperationInWrapper)
        else:
            lhs_rank = len(self._device_indexing_size(key))
            if isinstance(value, TensorType):
                rhs_rank = value.fake_value.ndim
                if lhs_rank != rhs_rank:
                    raise exc.RankMismatch(lhs_rank, rhs_rank)
            elif isinstance(value, UnknownType):
                raise exc.TypePropagationError(value)
            else:
                raise exc.RequiresTensorInAssignment(value)
        return self

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        if origin.is_host():
            try:
                # pyre-ignore[6]
                return TypeInfo.from_example(self.fake_value[key.proxy()], origin)
            except NotImplementedError:
                return UnknownType(
                    origin,
                    f"Subscript not supported on {self!s} with key={key!s}",
                )
        return TensorType(
            origin, self.fake_value.new_empty(self._device_indexing_size(key))
        )

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, TensorType):
            if self.fake_value is other.fake_value:
                return self
            if self.fake_value.device != other.fake_value.device:
                return UnknownType(
                    debug_msg=f"device mismatch in control flow: {self.fake_value.device} != {other.fake_value.device}",
                    origin=other.origin,
                )
            if self.fake_value.dtype != other.fake_value.dtype:
                return UnknownType(
                    debug_msg=f"dtype mismatch in control flow: {self.fake_value.dtype} != {other.fake_value.dtype}",
                    origin=other.origin,
                )
            if self.fake_value.dim() != other.fake_value.dim():
                return UnknownType(
                    debug_msg=f"rank mismatch in control flow: {self.fake_value.dim()} != {other.fake_value.dim()}",
                    origin=other.origin,
                )
            if self.fake_value.size() != other.fake_value.size():
                return UnknownType(
                    debug_msg=f"size mismatch in control flow: {self.fake_value.size()} != {other.fake_value.size()}",
                    origin=other.origin,
                )
            # TODO(jansel): handle symbolic shapes
            # TODO(jansel): stride check?
            return TensorType(other.origin, torch.empty_like(self.fake_value))
        return super().merge(other)

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
    origin: AttributeOrigin
    tensor: TensorType

    def __init__(self, origin: AttributeOrigin, tensor: TensorType) -> None:
        super().__init__(origin)
        self.tensor = tensor

    def attr(self) -> str:
        return self.origin.key

    def proxy(self) -> object:
        return getattr(self.tensor.proxy(), self.attr())

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, TensorAttributeType) and self.attr() == other.attr():
            combined_tensor = self.tensor.merge(other.tensor)
            if isinstance(combined_tensor, TensorType):
                return TensorAttributeType(self.origin, combined_tensor)
        return super().merge(other)

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
                return UnknownType(origin, f"Tensor.{attr}() args must be literals")

        proxy_args = [x.tree_map(_to_proxy) for x in args]
        proxy_kwargs = {k: v.tree_map(_to_proxy) for k, v in kwargs.items()}
        try:
            fn = getattr(self.tensor.fake_value, attr)
            output_type = TypeInfo.from_example(
                CheckForIndexCalls.retry_call(fn, proxy_args, proxy_kwargs), origin
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
            # pyre-ignore[16]
            return TypeInfo.from_example(self.value[key.as_literal()], origin)
        except NotImplementedError:
            pass
        return super().propagate_getitem(key, origin)

    def truth_value(self) -> bool:
        return bool(self.value)

    def merge(self, other: TypeInfo) -> TypeInfo:
        if type(other) is type(self) and self.value is other.value:
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
        return super().merge(other)

    def unpack(self) -> list[TypeInfo]:
        try:
            # pyre-ignore[6]
            it = iter(self.value)
        except TypeError:
            return super().unpack()
        return [TypeInfo.from_example(x, self.origin) for x in it]

    def as_literal(self) -> object:
        return self.value


class CallableType(LiteralType):
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
                return str(self.value)

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo | None:
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
        try:
            with patch.object(torch.SymInt, "__index__", _raise_shape_specializing):
                output_type = TypeInfo.from_example(
                    CheckForIndexCalls.retry_call(self.value, proxy_args, proxy_kwargs),
                    origin,
                )
            output_type.tree_map(warn_wrong_device)
            if (
                origin.is_host()
                and input_contains_tensor
                and output_type.contains_tensor()
            ):
                if not regexp_allowed_host_ops.search(self.name):
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


def _raise_shape_specializing(*args: object) -> None:
    raise exc.ShapeSpecializingCall


class PythonModuleType(LiteralType):
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

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, (LiteralType, NumericType)):
            if NumericType.known_equal(self.value, other.value):
                return self
            if self.python_type == other.python_type:
                return self.new_unbacked(self.origin)
        return super().merge(other)

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
    value: torch.SymInt

    @classmethod
    def new_unbacked(cls, origin: Origin) -> Self:
        shape_env = CompileEnvironment.current().shape_env
        with shape_env.ignore_fresh_unbacked_symbols():
            return cls(
                origin,
                shape_env.create_unbacked_symint(),
            )

    @property
    def python_type(self) -> type[int]:
        return int

    def proxy(self) -> torch.SymInt | int:
        if isinstance(self.value._sympy_(), sympy.Integer):
            return self.value.__int__()
        return self.value


class SymFloatType(NumericType):
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


class TileIndexType(TypeInfo):
    block_size_idx: int

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.block_size_idx})"

    def __init__(self, origin: Origin, block_size_idx: int) -> None:
        super().__init__(origin)
        self.block_size_idx = block_size_idx

    def proxy(self) -> object:
        with proxy_tensor.disable_proxy_modes_tracing():
            fake_mode = torch._C._unset_dispatch_mode(
                torch._C._TorchDispatchModeKey.FAKE
            )
            try:
                return TileIndexProxy(self.block_size_idx)
            finally:
                assert fake_mode is not None
                torch._C._set_dispatch_mode(fake_mode)

    @staticmethod
    def allocate(
        numels: list[int | torch.SymInt | AutoSize | None], origin: Origin
    ) -> list[TileIndexType]:
        env = CompileEnvironment.current()
        spec_id = len(env.config_spec.block_size_specs)
        env.config_spec.block_size_specs.append(
            BlockSizeSpec(
                size_hints=[*map(_get_hint, numels)],
                allow_flattened=len(numels) > 1,
                allow_reorder=len(numels) > 1,
                # TOOD(jansel): implement N-D l2 grouping
                allow_l2_grouping=len(numels) == 2
                # TODO(jansel): replace this check with "is outer loop"
                and len(env.config_spec.block_size_specs) == 0,
            )
        )
        return [
            TileIndexType(
                origin,
                env.allocate_block_size(
                    x, source=LoopSpecBlockSizeSource(spec_id, dim)
                ),
            )
            for dim, x in enumerate(numels)
        ]

    @staticmethod
    def allocate_fixed(
        numel: int | torch.SymInt | AutoSize | None,
        block_size: int | torch.SymInt,
        origin: Origin,
    ) -> TileIndexType:
        env = CompileEnvironment.current()
        return TileIndexType(
            origin,
            env.allocate_block_size(numel, source=FixedBlockSizeSource(block_size)),
        )

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, TileIndexType):
            if self.block_size_idx == other.block_size_idx:
                return self
            return UnknownType(
                debug_msg=f"TileIndexType mismatch in control flow: {self.block_size_idx} and {other.block_size_idx}",
                origin=other.origin,
            )
        return super().merge(other)

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        if isinstance(getattr(TileIndexProxy, attr, None), property):
            return TypeInfo.from_example(getattr(self.proxy(), attr), origin)
        return super().propagate_attribute(attr, origin)


class GridIndexType(SymIntType):
    block_size_idx: int

    def __init__(self, origin: Origin, block_size_idx: int) -> None:
        from .._compiler.compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        super().__init__(origin, env.block_sizes[block_size_idx].var)
        self.block_size_idx = block_size_idx

    def __str__(self) -> str:  # pragma: no cover – debug helper
        return f"{type(self).__name__}({self.block_size_idx})"

    @staticmethod
    def allocate(numel: int | torch.SymInt, origin: Origin) -> GridIndexType:
        from .._compiler.compile_environment import CompileEnvironment
        from .._compiler.compile_environment import GridBlockSizeSource

        env = CompileEnvironment.current()
        block_idx = env.allocate_block_size(numel, source=GridBlockSizeSource())
        return GridIndexType(origin, block_idx)

    def merge(self, other: TypeInfo) -> TypeInfo:  # type: ignore[override]
        if isinstance(other, GridIndexType):
            if self.block_size_idx == other.block_size_idx:
                return self
            return UnknownType(
                debug_msg=f"GridIndexType mismatch in control flow: {self.block_size_idx} vs {other.block_size_idx}",
                origin=other.origin,
            )
        return super().merge(other)


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

    def merge(self, other: TypeInfo) -> TypeInfo:
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
                    # pyre-ignore[6]
                    elements[k] = elements[k].merge(value)
                else:
                    # pyre-ignore[6]
                    elements[k] = value
                return self
        return super().propagate_setitem(key, value, origin)

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        try:
            literal_key = key.as_literal()
        except NotImplementedError:
            pass
        else:
            try:
                # pyre-ignore[16]
                result = self.element_types[literal_key]
            except (KeyError, IndexError) as e:
                return UnknownType(origin, f"{type(e).__name__}: {e}")
            if isinstance(result, TypeInfo):
                return result
            if type(result) is self.python_type:  # sliced!
                # pyre-ignore[6]
                return type(self)(origin=origin, element_types=result)
        return super().propagate_getitem(key, origin)

    def truth_value(self) -> bool:
        return bool(self.element_types)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> object:
        raise NotImplementedError


class SequenceType(CollectionType):
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
        return self._maybe_tuple([x.proxy() for x in self.element_types])

    def as_literal(self) -> list[object] | tuple[object, ...]:
        return self._maybe_tuple([x.as_literal() for x in self.element_types])

    def unpack(self) -> list[TypeInfo]:
        return [*self.element_types]

    def populate_symbol_origins(self, origin: Origin) -> None:
        for i, subtype in enumerate(self.element_types):
            subtype.populate_symbol_origins(GetItemOrigin(origin, i))

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, SequenceType):
            self_elements = self.element_types
            other_elements = self.element_types
            if len(self_elements) == len(other_elements):
                return SequenceType(
                    origin=other.origin,
                    element_types=[
                        self_elements[i].merge(other_elements[i])
                        for i in range(len(self_elements))
                    ],
                )
        return super().merge(other)

    def tree_map(
        self, fn: Callable[[TypeInfo], object]
    ) -> list[object] | tuple[object, ...]:
        return self._maybe_tuple([x.tree_map(fn) for x in self.element_types])


class DictType(CollectionType):
    element_types: dict[str | int, TypeInfo]

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

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, DictType):
            self_elements = self.element_types
            other_elements = self.element_types
            if set(self_elements.keys()) == set(other_elements.keys()):
                return DictType(
                    origin=other.origin,
                    element_types={
                        key: self_elements[key].merge(other_elements[key])
                        for key in self_elements
                    },
                )
        return super().merge(other)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> dict[str | int, object]:
        return {k: v.tree_map(fn) for k, v in self.element_types.items()}


class ClassType(DictType):
    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        return self.element_types[attr]


class SliceType(CollectionType):
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
        return slice(
            self.lower.as_literal(), self.upper.as_literal(), self.step.as_literal()
        )

    def unpack(self) -> list[TypeInfo]:
        return [self.lower, self.upper, self.step]

    def merge(self, other: TypeInfo) -> TypeInfo:
        if isinstance(other, SliceType):
            self_elements = self.element_types
            other_elements = self.element_types
            return SliceType(
                origin=other.origin,
                element_types=slice(
                    self_elements.start.merge(other_elements.start),
                    self_elements.stop.merge(other_elements.stop),
                    self_elements.step.merge(other_elements.step),
                ),
            )
        return super().merge(other)

    def tree_map(self, fn: Callable[[TypeInfo], object]) -> slice:
        return slice(
            self.lower.tree_map(fn), self.upper.tree_map(fn), self.step.tree_map(fn)
        )


class TypeNotAllowedOnDevice(TypeInfo):
    locations: list[SourceLocation]

    def __init__(self, origin: Origin) -> None:
        super().__init__(origin)
        self.locations = [current_location()]


class UnknownType(TypeNotAllowedOnDevice):
    def __init__(
        self, origin: Origin, debug_msg: str, *, chained_from: TypeInfo | None = None
    ) -> None:
        super().__init__(origin)
        self.debug_msg: str = debug_msg
        if isinstance(chained_from, UnknownType):
            self.locations: list[SourceLocation] = [
                *chained_from.locations,
                *self.locations,
            ]

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.debug_msg!r})"

    def chained(self, origin: Origin) -> TypeInfo:
        return ChainedUnknownType(origin, self)

    def merge(self, other: TypeInfo) -> TypeInfo:
        return self

    def propagate_unary(self, op: ast.unaryop, origin: Origin) -> TypeInfo:
        return self.chained(origin)

    def propagate_call(
        self, args: tuple[TypeInfo, ...], kwargs: dict[str, TypeInfo], origin: Origin
    ) -> TypeInfo:
        return self.chained(origin)

    def propagate_attribute(self, attr: str, origin: AttributeOrigin) -> TypeInfo:
        return self.chained(origin)


class UnsupportedType(UnknownType):
    def __init__(self, origin: Origin, python_type: type[object]) -> None:
        super().__init__(
            origin=origin,
            debug_msg=f"{python_type.__name__} is not supported",
        )
        self.python_type = python_type


class ChainedUnknownType(UnknownType):
    """Keep track of multiple locations for an operation that is already unknown type."""

    def __init__(self, origin: Origin, prior_type: UnknownType) -> None:
        super().__init__(
            origin=origin, debug_msg=prior_type.debug_msg, chained_from=prior_type
        )


def _eval_unary(op: ast.unaryop, value: object) -> object:
    if isinstance(op, ast.Not):
        return not value
    if isinstance(op, ast.UAdd):
        # pyre-ignore[16]
        return +value
    if isinstance(op, ast.USub):
        # pyre-ignore[16]
        return -value
    if isinstance(op, ast.Invert):
        # pyre-ignore[16]
        return ~value
    raise AssertionError(f"{type(op).__name__} unknown unary op")


def _eval_binary(op: ast.operator, left: object, right: object) -> object:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.FloorDiv):
        return left // right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.Pow):
        return left**right
    if isinstance(op, ast.LShift):
        return left << right
    if isinstance(op, ast.RShift):
        return left >> right
    if isinstance(op, ast.BitOr):
        return left | right
    if isinstance(op, ast.BitXor):
        return left ^ right
    if isinstance(op, ast.BitAnd):
        return left & right
    if isinstance(op, ast.MatMult):
        return left @ right
    raise AssertionError(f"{type(op).__name__} unknown binary op")


def _eval_compare(op: ast.cmpop, left: object, right: object) -> object:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.Is):
        return left is right
    if isinstance(op, ast.IsNot):
        return left is not right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
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
) -> Callable[[TypePropagation, ast.AST], TypeInfo]:
    def visit(self: TypePropagation, node: ast.AST) -> TypeInfo:
        # pyre-ignore[16]
        super(TypePropagation, self).generic_visit(node)
        return UnsupportedType(python_type=python_type, origin=self.origin())

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
                if self.device_loop_depth > 0 and isinstance(
                    type_info, TypeNotAllowedOnDevice
                ):
                    # We could defer this error with:
                    # CompileEnvironment.current().errors.add_type_error(type_info)
                    raise exc.TypePropagationError(type_info)
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
        return UnknownType(
            debug_msg=f"ast.{node.__class__.__name__} is not supported",
            origin=self.origin(),
        )

    def _bool_op(self, op: ast.boolop, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        try:
            val = left.truth_value()
            if isinstance(op, ast.Or) and val is False:
                return left
            if isinstance(op, ast.And) and val is True:
                return right
        except NotImplementedError:
            pass
        if (
            isinstance(left, (NumericType, LiteralType))
            and isinstance(right, (NumericType, LiteralType))
            and left.python_type == right.python_type
            and (pt := left.python_type) in (int, float, bool)
        ):
            return NumericType.subtype(pt).new_unbacked(self.origin())
        if isinstance(left, UnknownType):
            return left.chained(self.origin())
        if isinstance(right, UnknownType):
            return right.chained(self.origin())
        return UnknownType(
            debug_msg=f"{type(op).__name__} not supported on {left!s} and {right!s}",
            origin=self.origin(),
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
        if isinstance(left, UnknownType):
            return left.chained(self.origin())
        if isinstance(right, UnknownType):
            return right.chained(self.origin())
        return UnknownType(
            debug_msg=f"{type(op).__name__} not supported on {left!s} and {right!s}",
            origin=self.origin(),
        )

    def _assign(self, lhs: ast.AST, rhs: TypeInfo) -> None:
        if isinstance(lhs, ast.Name):
            return self.scope.set(lhs.id, rhs)
        if isinstance(lhs, ast.Starred):
            try:
                unpacked = SequenceType(self.origin(), rhs.unpack())
            except NotImplementedError:
                unpacked = UnknownType(
                    self.origin(),
                    f"Failed to unpack starred assignment: {rhs!s}",
                )
            return self._assign(lhs.value, unpacked)
        if isinstance(lhs, (ast.Tuple, ast.List)):
            lhs = lhs.elts
            elements: list[TypeInfo]
            try:
                elements = rhs.unpack()
            except NotImplementedError:
                elements = [
                    UnknownType(
                        self.origin(),
                        f"Failed to unpack assignment: {rhs!s}",
                    )
                    for _ in lhs
                ]
            used_star = False
            idx = 0
            for elt in lhs:
                if isinstance(elt, ast.Starred):
                    # TODO(jansel): need to test this
                    assert not used_star, "multiple `*` in assignment"
                    used_star = True
                    star_len = len(elements) - len(lhs) + 1
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
            if isinstance(lhs_base_type, TensorType):
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

    visit_FormattedValue: _VisitMethod = _unsupported(str)
    visit_JoinedStr: _VisitMethod = _unsupported(str)

    def _list_or_tuple(self, node: ast.List | ast.Tuple) -> TypeInfo:
        errors = []
        elements = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                to_unpack = self.visit(elt.value)
                try:
                    elements.extend(to_unpack.unpack())
                except NotImplementedError:
                    errors.append(
                        UnknownType(
                            self.origin(),
                            f"Failed to unpack starred assignment: {to_unpack!s}",
                        )
                    )
            else:
                elements.append(self.visit(elt))
        if errors:
            return errors[0]
        cls = list if isinstance(node, ast.List) else tuple
        return SequenceType(
            self.origin(),
            cls(elements),
        )

    visit_List: _VisitMethod = _list_or_tuple
    visit_Tuple: _VisitMethod = _list_or_tuple
    visit_Set: _VisitMethod = _unsupported(set)

    def visit_Dict(self, node: ast.Dict) -> TypeInfo:
        assert len(node.keys) == len(node.values)
        errors = []
        element_types = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            value = self.visit(value_node)
            if key_node is not None:
                key = self.visit(key_node)
                if not (
                    isinstance(key, LiteralType) and isinstance(key.value, (str, int))
                ):
                    errors.append(
                        UnknownType(
                            debug_msg=f"Only string/int literals are supported as dict keys, got {key!s}",
                            origin=self.origin(),
                        )
                    )
                    continue
                element_types[key.value] = value
            else:
                if not (isinstance(value, DictType)):
                    errors.append(
                        UnknownType(
                            debug_msg=f"Only collection types are supported as dict** values, got {value!s}",
                            origin=self.origin(),
                        )
                    )
                    continue
                element_types.update(value.element_types)
        if errors:
            return errors[0]
        return DictType(element_types=element_types, origin=self.origin())

    def visit_Name(self, node: ast.Name) -> TypeInfo:
        return self.scope.get(node.id)

    visit_Starred: _VisitMethod = generic_visit

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
                return TypeInfo.from_example(
                    _eval_binary(node.op, left_example, right_example),
                    self.origin(),
                )
            except exc.Base:
                raise
            except Exception as e:
                raise exc.TorchOpTracingError(e) from e

        if isinstance(left, UnknownType):
            return left.chained(self.origin())
        if isinstance(right, UnknownType):
            return right.chained(self.origin())

        return UnknownType(
            debug_msg=f"{type(node.op).__name__} not supported on {left!s} and {right!s}",
            origin=self.origin(),
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
        # TODO(jansel): check for calling a Kernel here
        func = self.visit(node.func)
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
            for arg in unhandled:
                if isinstance(arg, UnknownType):
                    return arg.chained(self.origin())
            return UnknownType(
                debug_msg="Failed to unpack */** args to function, got: "
                + ", ".join(map(str, unhandled)),
                origin=self.origin(),
                chained_from=unhandled[0],
            )
        return func.propagate_call(tuple(args), kwargs, self.origin())

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
        type_info = self.visit(
            create(
                ast.BinOp,
                left=node.target.copy(ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
        )
        self._assign(node.target, type_info)
        return NoType(origin=self.origin())

    visit_Raise: _VisitMethod = generic_statement
    visit_Assert: _VisitMethod = generic_statement
    visit_Delete: _VisitMethod = generic_statement
    visit_Pass: _VisitMethod = generic_statement
    visit_TypeAlias: _VisitMethod = generic_statement
    visit_Import: _VisitMethod = generic_statement
    visit_ImportFrom: _VisitMethod = generic_statement

    # TODO(jansel): support lambda
    visit_Lambda: _VisitMethod = generic_visit

    ################################################################
    # Control flow
    ################################################################

    def visit_If(self, node: ast.If) -> TypeInfo:
        test = self.visit(node.test)
        body = self._body(node.body)
        orelse = self._body(node.orelse)
        try:
            truth_val = test.truth_value()
            if truth_val:
                self.scope.merge(body)
            else:
                self.scope.merge(orelse)
        except NotImplementedError:
            self.scope.merge_if_else(body, orelse)
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
                raise exc.DeviceLoopElseBlock(fn.__qualname__)

            self.device_loop_count += 1
            if self.device_loop_depth == 0:
                self.func.set_local_types(parent_scope.extract_locals())
                node._loop_type = LoopType.GRID
                if self.device_loop_count != 1:
                    raise exc.MultipleDeviceLoops
                if len(ExtendedAST.current()) != 1:
                    raise exc.NestedGridLoop

        self.device_loop_depth += device_loop
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

    visit_Break: _VisitMethod = generic_statement
    visit_Continue: _VisitMethod = generic_statement

    def visit_Try(self, node: ast.Try) -> TypeInfo:
        self.scope.merge(self._body(node.body))
        for handler in node.handlers:
            self.push_scope()
            self.visit(handler)
            self.pop_scope_merge()
        self.scope.merge(self._body(node.orelse))
        self.scope.overwrite(self._body(node.finalbody))
        return NoType(origin=self.origin())

    visit_TryStar: _VisitMethod = visit_Try

    def _not_on_device_statement(self, node: ast.AST) -> TypeInfo:
        if self.device_loop_depth:
            raise exc.NotAllowedOnDevice(type(node).__name__)
        return NoType(origin=self.origin())

    visit_ExceptHandler: _VisitMethod = _not_on_device_statement
    visit_With: _VisitMethod = _not_on_device_statement
    visit_Return: _VisitMethod = _not_on_device_statement

    def _not_supported(self, node: ast.AST) -> TypeInfo:
        raise exc.StatementNotSupported(type(node).__name__)

    # TODO(jansel): need to implement these
    visit_ListComp: _VisitMethod = _not_supported
    visit_SetComp: _VisitMethod = _not_supported
    visit_GeneratorExp: _VisitMethod = _not_supported
    visit_DictComp: _VisitMethod = _not_supported

    # TODO(jansel): support closure functions defined on host
    visit_FunctionDef: _VisitMethod = _not_supported

    visit_ClassDef: _VisitMethod = _not_supported
    visit_Yield: _VisitMethod = _not_supported
    visit_YieldFrom: _VisitMethod = _not_supported
    visit_AsyncFunctionDef: _VisitMethod = _not_supported
    visit_AsyncFor: _VisitMethod = _not_supported
    visit_AsyncWith: _VisitMethod = _not_supported
    visit_Await: _VisitMethod = _not_supported
    visit_Match: _VisitMethod = _not_supported
    visit_MatchValue: _VisitMethod = _not_supported
    visit_MatchSingleton: _VisitMethod = _not_supported
    visit_MatchSequence: _VisitMethod = _not_supported
    visit_MatchStar: _VisitMethod = _not_supported
    visit_MatchMapping: _VisitMethod = _not_supported
    visit_MatchClass: _VisitMethod = _not_supported
    visit_MatchAs: _VisitMethod = _not_supported
    visit_MatchOr: _VisitMethod = _not_supported


def _to_proxy(arg: TypeInfo) -> object:
    if isinstance(arg, UnknownType):
        raise exc.TypePropagationError(arg)
    try:
        return arg.proxy()
    except NotImplementedError:
        raise exc.TracedArgNotSupported(arg) from None


def propagate_types(func: HostFunction, fake_args: list[object]) -> None:
    # Lock needed since patch.object(torch.SymInt.__index__, ...) is not thread safe
    with compile_lock, func:
        global_scope = GlobalScope(function=func)
        local_scope = LocalScope(parent=global_scope)
        params = inspect.signature(func.fn).bind(*fake_args)
        params.apply_defaults()
        for name, value in params.arguments.items():
            # TODO(jansel): handle specializations/constexpr
            type_info = TypeInfo.from_example(
                value,
                ArgumentOrigin(name=name, function=func),
            )
            local_scope.set(name, type_info)
        assert not func.fn.__closure__
        prop = TypePropagation(func, local_scope)
        for stmt in func.body:
            prop.visit(stmt)
