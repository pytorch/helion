from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import functools
import types
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import Protocol

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.convert_frame import compile_lock

from .. import exc
from .. import language as language_module
from ..language._decorators import get_device_func_replacement
from ..language._decorators import is_api_func
from ..runtime.kernel import Kernel
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import create
from .compile_environment import CompileEnvironment
from .compile_environment import block_size_constexpr_branch
from .compile_environment import warning
from .output_header import library_imports
from .source_location import current_location
from .tensor_utils import patch_tensor_factories
from .type_info import BarrierResultType
from .type_info import CallableType
from .type_info import CollectionType
from .type_info import DictType
from .type_info import LiteralType
from .type_info import NoType
from .type_info import NumericType
from .type_info import SequenceType
from .type_info import SliceType
from .type_info import StackTensorType
from .type_info import StringType
from .type_info import SymBoolType
from .type_info import SymIntType
from .type_info import TensorType
from .type_info import TileIndexType
from .type_info import TypeInfo
from .type_info import _eval_binary
from .type_info import _eval_compare
from .variable_origin import ArgumentOrigin
from .variable_origin import AttributeOrigin
from .variable_origin import BuiltinOrigin
from .variable_origin import DeviceOrigin
from .variable_origin import GlobalOrigin
from .variable_origin import Origin
from .variable_origin import SourceOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from .host_function import HostFunction

    class _VisitMethod(Protocol):
        @staticmethod
        def __call__(self: object, node: ast.AST) -> TypeInfo: ...


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
            # pyrefly: ignore [missing-attribute]
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

    # pyrefly: ignore [bad-override]
    def set(self, name: str, type_info: TypeInfo) -> None:
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


def _is_block_size_constexpr_test(test: TypeInfo) -> bool:
    """Return True if ``test`` is a symbolic bool that depends only on block sizes.

    Such a condition cannot be folded during the (single) frontend pass because
    block sizes are still symbolic, but it *is* a compile-time constant for each
    concrete config and is resolved per-config at codegen.  This is what lets an
    ``if``/``else`` define a variable with divergent shapes across branches.
    """
    import sympy

    from .host_function import HostFunction
    from .variable_origin import BlockSizeOrigin

    if not isinstance(test, SymBoolType):
        return False
    expr = test.value._sympy_()
    # A boolean expression (And/Or/Relational) is a sympy.Basic but not a
    # sympy.Expr, so check free symbols directly instead of find_block_size_symbols.
    if not isinstance(expr, sympy.Basic) or not expr.free_symbols:
        return False
    hf = HostFunction.current()
    for symbol in expr.free_symbols:
        origin_info = hf.expr_to_origin.get(symbol)
        if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
            return False
    return True


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

    @staticmethod
    def _contains_matmul(node: ast.AST | None) -> bool:
        if node is None:
            return False

        matmul_functions = ["torch.matmul", "torch.mm", "torch.bmm", "hl.dot"]

        for sub_node in ast.walk(node):
            # Check for @ operator
            if isinstance(sub_node, ast.BinOp) and isinstance(sub_node.op, ast.MatMult):
                return True

            # Check for function calls
            if not isinstance(sub_node, ast.Call):
                continue

            func = sub_node.func

            # Check for matmul function calls
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                qualified_name = f"{func.value.id}.{func.attr}"
                if qualified_name in matmul_functions:
                    return True

        return False

    def _bool_op(self, op: ast.boolop, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        try:
            val = left.truth_value()
            if isinstance(op, ast.Or):
                return left if val is True else right
            if isinstance(op, ast.And):
                return right if val is True else left
        except NotImplementedError:
            pass
        if isinstance(left, SymBoolType) and isinstance(right, SymBoolType):
            # Preserve the symbolic conjunction/disjunction (e.g.
            # ``block_m < 64 and block_n >= 64``) instead of minting a fresh
            # unbacked bool, so a block-size-only condition stays resolvable
            # per-config at codegen.  ``&``/``|`` on SymBool build the combined
            # symbolic expression without forcing a guard.
            combined = (
                left.value | right.value
                if isinstance(op, ast.Or)
                else left.value & right.value
            )
            if isinstance(combined, torch.SymBool):
                return SymBoolType(self.origin(), combined)
        if (
            isinstance(left, (NumericType, LiteralType))
            and isinstance(right, (NumericType, LiteralType))
            and left.python_type == right.python_type
            and (pt := left.python_type) in (int, float, bool)
        ):
            # pyrefly: ignore [bad-argument-type]
            return NumericType.subtype(pt).new_unbacked(self.origin())
        raise exc.TypeInferenceError(
            f"{type(op).__name__} not supported on {left!s} and {right!s}"
        )

    def _compare(self, op: ast.cmpop, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        # Handle `tensor is None` and `tensor is not None` checks
        # When comparing a TensorType with LiteralType(None), we can determine the result
        if isinstance(op, (ast.Is, ast.IsNot)):
            left_is_none = isinstance(left, LiteralType) and left.value is None
            right_is_none = isinstance(right, LiteralType) and right.value is None
            left_is_tensor = isinstance(left, TensorType)
            right_is_tensor = isinstance(right, TensorType)
            # tensor is None -> False, tensor is not None -> True
            if (left_is_tensor and right_is_none) or (right_is_tensor and left_is_none):
                result = isinstance(op, ast.IsNot)
                return LiteralType(origin=self.origin(), value=result)
            # None is None -> True, None is not None -> False
            if left_is_none and right_is_none:
                result = isinstance(op, ast.Is)
                return LiteralType(origin=self.origin(), value=result)
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
            # Preserve the symbolic comparison expression (e.g. ``block_m < 64``)
            # rather than minting a fresh unbacked bool.  This keeps the block-size
            # dependency intact so a block-size-only condition can be resolved
            # per-config at codegen (see IfGraphInfo.codegen), which is what makes
            # block-size-dependent control flow work on the host.
            if not isinstance(op, CMP_ALWAYS_BOOL):
                try:
                    result = _eval_compare(op, left.proxy(), right.proxy())
                except NotImplementedError:
                    result = None
                if isinstance(result, torch.SymBool):
                    return SymBoolType(self.origin(), result)
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
            if isinstance(rhs, TensorType):
                env = CompileEnvironment.current()
                shape_id = [
                    env.resolve_block_id(size) for size in list(rhs.fake_value.shape)
                ]
                jagged_tile_info = env.jagged_tile_parent_ids
                for jagged_tile_id, parent_block_ids in jagged_tile_info.items():
                    include_jagged = jagged_tile_id in shape_id
                    include_parents = all(p in shape_id for p in parent_block_ids)
                    if include_jagged and not include_parents:
                        raise exc.InvalidJaggedTileUsage(
                            f"jagged_tile alone cannot be used without its parent in assignment {lhs.id}"
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
            # pyrefly: ignore [bad-assignment]
            lhs = lhs.elts
            elements: list[TypeInfo]
            try:
                elements = rhs.unpack()
            except NotImplementedError:
                if isinstance(rhs, TileIndexType):
                    raise exc.FailedToUnpackTile from None
                # pyrefly: ignore [bad-argument-type]
                raise exc.FailedToUnpackTupleAssign(len(lhs), rhs) from None
            used_star = False
            idx = 0
            # pyrefly: ignore [not-iterable]
            for elt in lhs:
                if isinstance(elt, ast.Starred):
                    assert not used_star, "multiple `*` in assignment"
                    used_star = True
                    # pyrefly: ignore [bad-argument-type]
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

    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_List: _VisitMethod = _list_or_tuple
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Tuple: _VisitMethod = _list_or_tuple
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Set: _VisitMethod = _unsupported(set)

    def visit_Dict(self, node: ast.Dict) -> TypeInfo:
        assert len(node.keys) == len(node.values)
        element_types: dict[int | str, TypeInfo] = {}
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

    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
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
        # pyrefly: ignore [bad-argument-type, bad-return]
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
        if (
            self.device_loop_depth > 0
            and isinstance(node.op, ast.Add)
            and self._contains_matmul(node.value)
        ):
            warning(exc.TiledKMatmulAccumulationWarning)
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

    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Raise: _VisitMethod = generic_statement
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Delete: _VisitMethod = generic_statement

    def visit_Pass(self, node: ast.Pass) -> TypeInfo:
        return NoType(origin=self.origin())

    # pyrefly: ignore [bad-assignment]
    visit_TypeAlias: _VisitMethod = generic_statement
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Import: _VisitMethod = generic_statement
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_ImportFrom: _VisitMethod = generic_statement

    def visit_Global(self, node: ast.Global) -> TypeInfo:
        # Global statements don't need child visiting since they only declare names
        return NoType(origin=self.origin())

    # TODO(jansel): support lambda
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Lambda: _VisitMethod = generic_visit

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
        elif _is_block_size_constexpr_test(test):
            # The condition is a compile-time constant that depends on block
            # sizes: it stays symbolic during this single frontend pass but is
            # resolved per-config at codegen, where only the live branch is
            # emitted (see DeviceFunction.evaluate_constexpr_condition and
            # IfGraphInfo.codegen).  Because exactly one branch survives per
            # config, the branches may legally define a variable with the same
            # rank but different shapes (e.g. a transposed accumulator).
            with block_size_constexpr_branch():
                self.scope.merge_if_else(self._body(node.body), self._body(node.orelse))
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
                raise exc.DeviceLoopElseBlock(fn.__qualname__)

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
            if (
                self.device_loop_depth > 0
                and CompileEnvironment.current().backend.pad_factory_tensors_to_power_of_2
            )
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

    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Break: _VisitMethod = generic_statement
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
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

    # pyrefly: ignore [bad-assignment]
    visit_TryStar: _VisitMethod = visit_Try

    def _not_on_device_statement(self, node: ast.AST) -> TypeInfo:
        if self.device_loop_depth:
            raise exc.NotAllowedOnDevice(type(node).__name__)
        for child_node in ast.iter_child_nodes(node):
            self.visit(child_node)
        return NoType(origin=self.origin())

    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_ExceptHandler: _VisitMethod = _not_on_device_statement
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_With: _VisitMethod = _not_on_device_statement
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Return: _VisitMethod = _not_on_device_statement

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

    def _visit_comprehension(
        self, node: ast.ListComp | ast.GeneratorExp, name: str
    ) -> TypeInfo:
        """Type propagation for list comprehensions and generator expressions."""
        if len(node.generators) != 1:
            raise exc.StatementNotSupported(
                f"{name.capitalize()}s with multiple generators are not supported"
            )
        return self._evaluate_comprehension(node.generators[0], node.elt)

    def visit_ListComp(self, node: ast.ListComp) -> TypeInfo:
        return self._visit_comprehension(node, "list comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> TypeInfo:
        return self._visit_comprehension(node, "generator expression")

    def visit_DictComp(self, node: ast.DictComp) -> TypeInfo:
        """Type propagation for dict comprehensions."""
        if len(node.generators) != 1:
            raise exc.StatementNotSupported(
                "Dict comprehensions with multiple generators are not supported"
            )

        generator = node.generators[0]
        iter_type = self.visit(generator.iter)

        # Try to unpack the iterable
        try:
            iterable_elements = iter_type.unpack()
        except NotImplementedError:
            raise exc.StatementNotSupported(
                "Dict comprehensions over non-unpackable iterables are not supported"
            ) from None

        result_elements: dict[str | int, TypeInfo] = {}

        def clear_type_info(n: ast.AST) -> None:
            """Clear _type_info on AST nodes to allow re-visiting with different values."""
            if isinstance(n, ExtendedAST):
                n._type_info = None
            for child in ast.iter_child_nodes(n):
                clear_type_info(child)

        for element_type in iterable_elements:
            self.push_scope()
            try:
                self._assign(generator.target, element_type)
                for if_clause in generator.ifs:
                    self.visit(if_clause)
                # Clear type info before visiting to avoid merging with previous iteration
                clear_type_info(node.key)
                clear_type_info(node.value)
                key_type = self.visit(node.key)
                value_type = self.visit(node.value)
                # Get the literal key value by evaluating with proxy
                try:
                    key = key_type.proxy()
                except (NotImplementedError, TypeError):
                    raise exc.StatementNotSupported(
                        "Dict comprehension keys must evaluate to literals"
                    ) from None
                if not isinstance(key, (str, int)):
                    raise exc.StatementNotSupported(
                        f"Dict comprehension keys must be str or int, got {type(key).__name__}"
                    )
                result_elements[key] = value_type
            finally:
                self.pop_scope()

        return DictType(self.origin(), result_elements)

    # TODO(jansel): need to implement these
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_SetComp: _VisitMethod = _not_supported

    # TODO(jansel): support closure functions defined on host
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_FunctionDef: _VisitMethod = _not_supported

    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_ClassDef: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Yield: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_YieldFrom: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_AsyncFunctionDef: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_AsyncFor: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_AsyncWith: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Await: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_Match: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchValue: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchSingleton: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchSequence: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchStar: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchMapping: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchClass: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchAs: _VisitMethod = _not_supported
    # pyrefly: ignore [bad-assignment, bad-param-name-override, bad-override-mutable-attribute]
    visit_MatchOr: _VisitMethod = _not_supported


def _is_barrier_stmt(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Expr):
        value = statement.value
        type_info = getattr(value, "_type_info", None)
        return isinstance(type_info, BarrierResultType)
    return False


def _create_type_propagation(func: HostFunction) -> TypePropagation:
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
    return TypePropagation(func, local_scope)


def _check_no_stmts_between_loops(body: list[ast.stmt]) -> None:
    """Raise if non-barrier host statements appear between top-level for-loops."""
    seen_loop = False
    host_stmt_after_loop = False
    for stmt in body:
        if isinstance(stmt, ast.For):
            if host_stmt_after_loop:
                # TODO(oulgen): This check is too coarse, refine it.
                raise exc.TopLevelStatementBetweenLoops
            seen_loop = True
        elif seen_loop and not _is_barrier_stmt(stmt):
            host_stmt_after_loop = True


def propagate_types(func: HostFunction) -> None:
    # Lock needed since patch.object(torch.SymInt.__index__, ...) is not thread safe
    with compile_lock, func, enable_python_dispatcher():
        prop = _create_type_propagation(func)
        for stmt in func.body:
            prop.visit(stmt)
        _check_no_stmts_between_loops(func.body)
