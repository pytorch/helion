from __future__ import annotations

import ast
import dataclasses
import functools
import re
import types
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast
from unittest.mock import patch

import sympy
import torch
from torch.fx.experimental import proxy_tensor
from torch.utils._pytree import tree_map_only

from .. import exc
from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_spec import BlockSizeSpec
from ..autotuner.config_spec import NumThreadsSpec
from ..language._decorators import is_api_func
from ..language.stack_tensor import StackTensor
from ..language.tile_proxy import Tile
from ..language.tile_proxy import _CheckForIndexCalls
from .ast_extension import ExtendedAST
from .compile_environment import AutoSize
from .compile_environment import CompileEnvironment
from .compile_environment import FixedBlockSizeSource
from .compile_environment import LoopSpecBlockSizeSource
from .compile_environment import _symint_expr
from .compile_environment import warning
from .device_function import contains_only_block_size_symbols
from .host_function import HostFunction
from .host_function import SymbolOrigin
from .utils import compute_slice_size
from .variable_origin import AttributeOrigin
from .variable_origin import GetItemOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from typing_extensions import Self

    _T = TypeVar("_T")


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

            # pyrefly: ignore [bad-argument-type]
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
            CompileEnvironment.current().add_kernel_tensor_size(
                fake_value.size(), fake_value.dtype
            )

    def __str__(self) -> str:
        shape: list[str] = []
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
                    # On backends that don't pad factory ops to power-of-2,
                    # concrete int dims must stay concrete so subsequent shape
                    # inference can prove equality with other concretely-sized
                    # buffers (e.g. host-allocated accumulators via new_zeros).
                    # Allocating a reduction-dim block here would introduce a
                    # fresh unbacked symbol that does not unify with the int
                    # even when the hint matches.
                    if (
                        isinstance(output_size, int)
                        and not env.backend.pad_factory_tensors_to_power_of_2
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
                with CompileEnvironment.current().shape_env.suppress_guards():
                    # pyrefly: ignore [bad-index]
                    return TypeInfo.from_example(self.fake_value[key.proxy()], origin)
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
        if attr in {"dim", "ndimension"} and not (args or kwargs):
            return TypeInfo.from_example(self.tensor.fake_value.ndim, origin)
        if attr in {"shape", "size", "stride"} and not kwargs:
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
            # pyrefly: ignore [bad-index]
            return TypeInfo.from_example(self.value[key.as_literal()], origin)
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
                # pyrefly: ignore [bad-argument-type]
                return NumericType.subtype(self.python_type).new_unbacked(self.origin)
        return super().merge(other, var_name=var_name)

    def unpack(self) -> list[TypeInfo]:
        try:
            # pyrefly: ignore [no-matching-overload]
            it = iter(self.value)
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

    # pyrefly: ignore [bad-override]
    value: ConfigSpecFragment

    def __init__(self, origin: Origin, fragment: ConfigSpecFragment) -> None:
        assert isinstance(fragment, ConfigSpecFragment)
        super().__init__(origin, fragment)


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
                return str(self.value)

    # pyrefly: ignore [bad-override]
    def propagate_call(
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
        if self.value in (torch.split, torch.Tensor.split) and origin.is_device():
            raise exc.UnsupportedSplitOperation(op="torch.split")
        if (
            self.value in (torch.tensor_split, torch.Tensor.tensor_split)
            and origin.is_device()
        ):
            raise exc.UnsupportedSplitOperation(op="torch.tensor_split")
        if is_api_func(fn := self.value):
            if fn._is_device_only and origin.is_host():
                raise exc.DeviceAPIOnHost(fn.__qualname__)
            if fn._cache_type and ExtendedAST.current()[-1]._type_info is not None:
                return ExtendedAST.current()[-1]._type_info
            assert fn._type_function is not None
            return fn._type_function(*args, **kwargs, origin=origin)
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
                    try:
                        outer_spec = env.config_spec.block_sizes.block_id_lookup(
                            maybe_bounded_by
                        )
                    except KeyError:
                        pass
                    else:
                        bounded_by = maybe_bounded_by
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
                    # pyrefly: ignore [bad-index, unsupported-operation]
                    elements[k] = elements[k].merge(value)
                else:
                    # pyrefly: ignore [unsupported-operation]
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
                # pyrefly: ignore [bad-index]
                result = self.element_types[literal_key]
            except (KeyError, IndexError) as e:
                raise exc.TypeInferenceError(f"{type(e).__name__}: {e}") from None
            if isinstance(result, TypeInfo):
                return result
            if type(result) is self.python_type:  # sliced!
                # pyrefly: ignore [bad-argument-type]
                return type(self)(origin=origin, element_types=result)
        return super().propagate_getitem(key, origin)

    def truth_value(self) -> bool:
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
        return self._maybe_tuple([x.proxy() for x in self.element_types])

    def as_literal(self) -> list[object] | tuple[object, ...]:
        return self._maybe_tuple([x.as_literal() for x in self.element_types])

    def unpack(self) -> list[TypeInfo]:
        return [*self.element_types]

    def populate_symbol_origins(self, origin: Origin) -> None:
        for i, subtype in enumerate(self.element_types):
            subtype.populate_symbol_origins(GetItemOrigin(origin, i))

    def propagate_getitem(self, key: TypeInfo, origin: Origin) -> TypeInfo:
        # Tuple/List indexing with non-literal indices (e.g., from hl.static_range)
        if self.python_type in (tuple, list) and isinstance(key, SymIntType):
            if not self.element_types:
                raise exc.TypeInferenceError("Cannot index empty sequence")
            first_type = self.element_types[0]
            if not all(type(e) is type(first_type) for e in self.element_types[1:]):
                raise exc.TypeInferenceError(
                    "Sequence indexing with non-literal index requires all elements to have the same type"
                )
            return first_type

        return super().propagate_getitem(key, origin)

    def propagate_setitem(
        self, key: TypeInfo, value: TypeInfo, origin: Origin
    ) -> TypeInfo:
        if self.python_type is list and isinstance(key, SymIntType):
            if not self.element_types:
                raise exc.TypeInferenceError("Cannot index empty sequence")
            new_elements = [elem.merge(value) for elem in self.element_types]
            return SequenceType(origin=origin, element_types=new_elements)
        return super().propagate_setitem(key, value, origin)

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
    # pyrefly: ignore [bad-override]
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

    def merge(self, other: TypeInfo, var_name: str | None = None) -> TypeInfo:
        if type(self) is type(other):
            assert isinstance(other, DictType)
            self_elements = self.element_types
            other_elements = other.element_types
            if set(self_elements.keys()) == set(other_elements.keys()):
                return type(self)(
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
