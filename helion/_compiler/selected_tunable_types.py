"""Config-local scalar typing for a bounded, source-proved pointwise use."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx.experimental.sym_node import SymNode
from torch.fx.node import map_arg

from .. import exc
from ..autotuner.config_fragment import BooleanFragment
from ..language import _tracing_ops
from ..language.memory_ops import store
from ..language.tunable_ops import _get_tunable
from ..language.tunable_ops import register_tunable
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .aten_lowering import AtenLowering
from .compile_environment import CompileEnvironment
from .inductor_lowering import PointwiseLowering
from .inductor_lowering import SympyExprLowering
from .inductor_lowering import prepare_added_node_lowering
from .type_info import CallableType
from .type_info import LiteralType
from .type_info import NestedFunctionType
from .type_info import NumericType
from .type_info import TensorAttributeType
from .type_info import TensorType
from .variable_origin import NameOrigin
from .variable_origin import ScalarArithmeticOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy
    from torch.fx.node import Argument

    from ..runtime.config import Config
    from .device_ir import GraphInfo
    from .host_function import HostFunction


_POINTWISE = frozenset(
    (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.abs.default,
        torch.ops.aten.where.self,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp_min.default,
        torch.ops.aten.clamp_max.default,
        torch.ops.aten.full.default,
        torch.ops.prims.convert_element_type.default,
    )
)
_SCALARS = (torch.SymInt, torch.SymFloat, torch.SymBool)
_DTYPE_CAST_ALIASES = {
    "bool": torch.bool,
    "byte": torch.uint8,
    "char": torch.int8,
    "short": torch.int16,
    "int": torch.int32,
    "long": torch.int64,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "double": torch.float64,
}
_DTYPE_PRESERVING_UNARY = (torch.abs, torch.absolute, torch.neg, torch.negative)
_DTYPE_PRESERVING_UNARY_METHODS = ("abs", "absolute", "neg", "negative")


def _only_boolean_conditions(func: HostFunction, expression: sympy.Basic) -> bool:
    """Recognize integer 0/1 spellings used solely as direct Boolean conditions.

    BooleanFragment has always accepted these spellings. They have the same
    truth value, but their tensor promotion differs, so arithmetic must still
    take the selected-type path. Do not infer safety from simplified FX uses.
    """
    names = {
        node.id
        for statement in func.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node, ExtendedAST)
        and isinstance(node._type_info, NumericType)
        and node._type_info.value._sympy_() == expression
    }
    if not names:
        return False
    safe = True
    definitions: set[ast.FunctionDef] = set()

    def walk(
        node: ast.AST, parent: ast.AST | None = None, in_definition: bool = False
    ) -> None:
        nonlocal safe
        if isinstance(node, ast.FunctionDef):
            return
        if (
            isinstance(node, ExtendedAST)
            and isinstance(node, (ast.expr, ast.stmt))
            and not in_definition
            and node._type_info is None
        ):
            return
        if (
            isinstance(node, ast.Name)
            and node.id in names
            and isinstance(node.ctx, ast.Load)
        ):
            safe &= isinstance(parent, ast.If) and parent.test is node
        info = node._type_info if isinstance(node, ExtendedAST) else None
        if isinstance(info, NestedFunctionType) and info.func_node not in definitions:
            definitions.add(info.func_node)
            for statement in info.func_node.body:
                walk(statement, in_definition=True)
        for child in ast.iter_child_nodes(node):
            walk(child, node, in_definition)

    for statement in func.body:
        walk(statement)
    return safe


def _refuse(detail: str) -> None:
    raise exc.TypeInferenceError(
        "A selected tunable changes scalar type outside supported direct "
        f"pointwise arithmetic ({detail}). Use a same-type tunable domain or "
        "an explicit scalar conversion before registration."
    )


def _prove_source(
    func: HostFunction, selected: dict[sympy.Basic, type[int | float | bool]]
) -> dict[str, set[type[int | float | bool]]]:
    """Follow scalar and tensor uses before dtype-dependent operations disappear.

    Both ``round(m)`` and ``floor(x * m)`` can disappear for integer defaults.
    An FX-only proof cannot distinguish either from its unrounded input.
    Conservatively union named dependencies across the already typed branches.
    """
    scalar, tensor = 1, 2
    expressions = {symbol for expr in selected for symbol in expr.free_symbols}
    named_types: dict[str, set[type[int | float | bool]]] = {}

    class Prove(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0
            self.names: dict[str, int] = {}
            # AugAssign Store nodes have no type annotation of their own.
            self.tensor_names: set[str] = set()
            self.scalar_names: dict[str, type[int | float | bool]] = {}

        def scalar_type(self, node: ast.AST) -> type[int | float | bool]:
            if isinstance(node, ast.Name) and node.id in self.scalar_names:
                return self.scalar_names[node.id]
            info = node._type_info if isinstance(node, ExtendedAST) else None
            if isinstance(info, NumericType):
                return selected.get(info.value._sympy_(), info.value.node.pytype)
            if isinstance(info, LiteralType) and type(info.value) in (int, float, bool):
                return cast("type[int | float | bool]", type(info.value))
            self.fail(node)
            raise AssertionError("unreachable")

        def record_scalar(self, node: ast.AST, kind: type[int | float | bool]) -> None:
            info = node._type_info if isinstance(node, ExtendedAST) else None
            if isinstance(info, NumericType):
                selected[info.value._sympy_()] = kind

        def visit(self, node: ast.AST) -> int:
            if (
                isinstance(node, ExtendedAST)
                and isinstance(node, (ast.expr, ast.stmt))
                and node._type_info is None
            ):
                # A constexpr branch that was not traced is not emitted.
                return 0
            return super().visit(node)

        def fail(self, node: ast.AST) -> None:
            with node if isinstance(node, ExtendedAST) else func.location:
                _refuse(f"authored {type(node).__name__}")

        def scalar_origin(self, node: ast.AST) -> int:
            info = node._type_info if isinstance(node, ExtendedAST) else None
            return (
                scalar
                if (
                    isinstance(info, NumericType)
                    and bool(info.value._sympy_().free_symbols & expressions)
                )
                else 0
            )

        def visit_Name(self, node: ast.Name) -> int:
            if isinstance(node.ctx, ast.Store):
                return 0
            if node.id in self.scalar_names:
                self.record_scalar(node, self.scalar_names[node.id])
            used = self.scalar_origin(node) | self.names.get(node.id, 0)
            if used == scalar:
                named_types.setdefault(node.id, set()).add(self.scalar_type(node))
            return used

        def visit_Assign(self, node: ast.Assign) -> int:
            used = self.visit(node.value)
            kind = self.scalar_type(node.value) if used == scalar else None
            for target in node.targets:
                self.visit(target)
                if isinstance(target, ast.Name):
                    self.tensor_names.discard(target.id)
                    self.scalar_names.pop(target.id, None)
                    if isinstance(node.value, ExtendedAST) and isinstance(
                        node.value._type_info, TensorType
                    ):
                        self.tensor_names.add(target.id)
            if used:
                if all(isinstance(target, ast.Name) for target in node.targets):
                    for target in node.targets:
                        assert isinstance(target, ast.Name)
                        self.names[target.id] = self.names.get(target.id, 0) | used
                        if kind is not None:
                            self.scalar_names[target.id] = kind
                elif not (
                    self.depth > 0
                    and used == tensor
                    and all(
                        isinstance(target, ast.Subscript) for target in node.targets
                    )
                ):
                    self.fail(node)
            return 0

        def visit_AnnAssign(self, node: ast.AnnAssign) -> int:
            used = self.visit(node.value) if node.value is not None else 0
            kind = (
                self.scalar_type(node.value)
                if used == scalar and node.value is not None
                else None
            )
            if isinstance(node.target, ast.Name):
                self.scalar_names.pop(node.target.id, None)
            if used:
                if not isinstance(node.target, ast.Name):
                    self.fail(node)
                assert isinstance(node.target, ast.Name)
                self.names[node.target.id] = self.names.get(node.target.id, 0) | used
                if kind is not None:
                    self.scalar_names[node.target.id] = kind
            if isinstance(node.target, ast.Name):
                self.tensor_names.discard(node.target.id)
                if isinstance(node.value, ExtendedAST) and isinstance(
                    node.value._type_info, TensorType
                ):
                    self.tensor_names.add(node.target.id)
            return 0

        def visit_AugAssign(self, node: ast.AugAssign) -> int:
            used = self.visit(node.value)
            if isinstance(node.target, ast.Name):
                used |= self.names.get(node.target.id, 0)
            else:
                used |= self.visit(node.target)
            if used:
                if not (
                    self.depth > 0
                    and isinstance(node.target, ast.Name)
                    and node.target.id in self.tensor_names
                    and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
                ):
                    self.fail(node)
                assert isinstance(node.target, ast.Name)
                self.names[node.target.id] = tensor
            return 0

        def visit_For(self, node: ast.For) -> int:
            if self.visit(node.iter):
                self.fail(node)
            device = isinstance(node, ExtendedAST) and node._loop_type in (
                LoopType.DEVICE,
                LoopType.GRID,
            )
            self.depth += device
            for child in (*node.body, *node.orelse):
                self.visit(child)
            self.depth -= device
            return 0

        def visit_Call(self, node: ast.Call) -> int:
            callee = (
                node.func._type_info if isinstance(node.func, ExtendedAST) else None
            )
            if (
                isinstance(callee, TensorAttributeType)
                and (callee.attr() == "to" or callee.attr() in _DTYPE_CAST_ALIASES)
                and isinstance(node.func, ast.Attribute)
            ):
                # Preserve an explicit dtype conversion only when its dtype is
                # independent of the selected value and the default trace kept
                # the conversion. A default no-op cast may disappear before FX;
                # admitting it could omit newly observable rounding/truncation.
                used = self.visit(node.func.value)
                arguments = (*node.args, *(kw.value for kw in node.keywords))
                for argument in arguments:
                    if self.visit(argument):
                        self.fail(node)
                if not used:
                    return 0
                dtype_node = (
                    node.args[0]
                    if len(node.args) == 1 and not node.keywords
                    else node.keywords[0].value
                    if not node.args
                    and len(node.keywords) == 1
                    and node.keywords[0].arg == "dtype"
                    else None
                )
                dtype_info = (
                    dtype_node._type_info
                    if isinstance(dtype_node, ExtendedAST)
                    else None
                )
                if not node.args and not node.keywords:
                    target_dtype = _DTYPE_CAST_ALIASES.get(callee.attr())
                elif (
                    callee.attr() == "to"
                    and isinstance(dtype_info, LiteralType)
                    and type(dtype_info.value) is torch.dtype
                ):
                    target_dtype = dtype_info.value
                else:
                    target_dtype = None
                result = node._type_info if isinstance(node, ExtendedAST) else None
                if not (
                    self.depth > 0
                    and target_dtype is not None
                    and isinstance(result, TensorType)
                    and result.fake_value.dtype == target_dtype
                    and callee.tensor.fake_value.dtype != target_dtype
                ):
                    self.fail(node)
                return tensor
            unary_method = (
                isinstance(callee, TensorAttributeType)
                and callee.attr() in _DTYPE_PRESERVING_UNARY_METHODS
            )
            unary_function = isinstance(callee, CallableType) and any(
                callee.value is fn for fn in _DTYPE_PRESERVING_UNARY
            )
            tensor_method = unary_method or (
                isinstance(callee, TensorAttributeType)
                and callee.attr()
                in ("add", "sub", "mul", "div", "clamp", "clamp_min", "clamp_max")
            )
            if (
                tensor_method
                or unary_function
                or (
                    isinstance(callee, CallableType)
                    and any(
                        callee.value is fn
                        for fn in (torch.where, torch.clamp, torch.full_like)
                    )
                )
            ):
                used = 0
                if tensor_method:
                    used |= self.visit(
                        node.func.value
                        if isinstance(node.func, ast.Attribute)
                        else node.func
                    )
                for argument in (
                    *node.args,
                    *(keyword.value for keyword in node.keywords),
                ):
                    used |= self.visit(argument)
                if used:
                    info = node._type_info if isinstance(node, ExtendedAST) else None
                    if self.depth <= 0 or not isinstance(info, TensorType):
                        self.fail(node)
                    if unary_method or unary_function:
                        # These native operations retain their value operation
                        # for integer defaults. Unlike floor/round, their effect
                        # cannot disappear before selected-type propagation.
                        if (
                            unary_method
                            and isinstance(node.func, ast.Attribute)
                            and not node.args
                            and not node.keywords
                        ):
                            operand = node.func.value
                        elif (
                            unary_function and len(node.args) == 1 and not node.keywords
                        ):
                            operand = node.args[0]
                        elif (
                            unary_function
                            and not node.args
                            and len(node.keywords) == 1
                            and node.keywords[0].arg == "input"
                        ):
                            operand = node.keywords[0].value
                        else:
                            operand = None
                        operand_info = (
                            operand._type_info
                            if isinstance(operand, ExtendedAST)
                            else None
                        )
                        if not isinstance(operand_info, TensorType) or (
                            operand_info.fake_value.dtype
                            != cast("TensorType", info).fake_value.dtype
                        ):
                            self.fail(node)
                    return tensor
                return 0
            if isinstance(callee, NestedFunctionType):
                # Nested definitions are not themselves emitted or visited.
                # Inspect the invoked definition's captures even if its result
                # became a literal, e.g. round(integer) - integer. Follow other
                # resolved nested helpers without executing or retracing them.
                pending = [callee.func_node]
                seen: set[ast.FunctionDef] = set()
                while pending:
                    definition = pending.pop()
                    if definition in seen:
                        continue
                    seen.add(definition)
                    for child in ast.walk(definition):
                        if isinstance(child, ast.Name) and isinstance(
                            child.ctx, ast.Load
                        ):
                            if self.visit_Name(child):
                                self.fail(node)
                            info = (
                                child._type_info
                                if isinstance(child, ExtendedAST)
                                else None
                            )
                            if isinstance(info, NestedFunctionType):
                                pending.append(info.func_node)
            return self.generic_visit(node)

        def generic_visit(self, node: ast.AST) -> int:
            used = 0
            for child in ast.iter_child_nodes(node):
                used |= self.visit(child)
            if not used:
                origin = self.scalar_origin(node)
                if origin:
                    # A symbolic result is not a proof of the operation that
                    # produced it: default typing can erase calls, attributes
                    # or subscripts. Only direct registration seeds the proof.
                    callee = (
                        node.func._type_info
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ExtendedAST)
                        else None
                    )
                    if not (
                        isinstance(callee, CallableType)
                        and callee.value is register_tunable
                    ):
                        self.fail(node)
                return origin
            if isinstance(node, ast.Expr):
                return 0
            if isinstance(node, ast.Return) and used == scalar:
                return 0
            info = node._type_info if isinstance(node, ExtendedAST) else None
            arithmetic = (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
            ) or (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, (ast.UAdd, ast.USub))
            )
            if used == scalar and isinstance(info, NumericType) and arithmetic:
                assert isinstance(node, (ast.BinOp, ast.UnaryOp))
                operands = (
                    (node.left, node.right)
                    if isinstance(node, ast.BinOp)
                    else (node.operand,)
                )
                kinds = [self.scalar_type(operand) for operand in operands]
                kind = float if float in kinds or isinstance(node.op, ast.Div) else int
                self.record_scalar(node, kind)
                return scalar
            if (
                self.depth > 0
                and isinstance(info, TensorType)
                and (
                    (
                        isinstance(node, ast.BinOp)
                        and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
                    )
                    or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub))
                )
            ):
                return tensor
            self.fail(node)
            return 0

    prover = Prove()
    for statement in func.body:
        prover.visit(statement)
    return named_types


def specialize_selected_tunable_types(
    func: HostFunction, graphs: Sequence[GraphInfo], config: Config
) -> None:
    """Retype only copied nodes; never change the bound graph or its search space."""
    from .device_ir import NodeArgsGraphInfo
    from .device_ir import control_flow_child_graph_ids

    env = CompileEnvironment.current()
    selected: dict[sympy.Basic, type[int | float | bool]] = {}
    for (name, _), value in env.tunable_values.items():
        if name not in config.config:
            continue
        chosen = config.config[name]
        if type(chosen) not in (int, float, bool):
            continue
        if type(chosen) is not value.node.pytype:
            if (
                type(env.config_spec.user_defined_tunables[name]) is BooleanFragment
                and type(chosen) is int
                and chosen in (0, 1)
                and _only_boolean_conditions(func, value._sympy_())
            ):
                continue
            selected[value._sympy_()] = cast("type[int | float | bool]", type(chosen))
    if not selected:
        return
    if func.backend_rewrote_ast:
        _refuse("a backend-rewritten source")
    named_types = _prove_source(func, selected)

    def ordered_scalar_type(origin: object) -> type[int | float | bool] | None:
        # Device tracing may create another fresh symbol for the same ordered
        # host arithmetic. Its origin, rather than a default-value hint or
        # algebraic simplification, connects it to the proved authored inputs.
        if type(origin) in (int, float, bool):
            return cast("type[int | float | bool]", type(origin))
        if isinstance(origin, NameOrigin):
            kinds = named_types.get(origin.name, set())
            if len(kinds) > 1:
                _refuse("an ambiguously retyped host name")
            return next(iter(kinds), None)
        if isinstance(origin, ScalarArithmeticOrigin) and origin.op in (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.UAdd,
            ast.USub,
        ):
            kinds = [ordered_scalar_type(operand) for operand in origin.operands]
            if None not in kinds:
                return float if float in kinds else int
        return None

    # NodeArgsGraphInfo keeps links into the original graphs when copied.
    # Rebind those links only in this config-local view.
    copies = {
        old: new
        for original, copied in zip(func.device_ir.graphs, graphs, strict=True)
        for old, new in zip(original.graph.nodes, copied.graph.nodes, strict=True)
    }
    for graph in graphs:
        if isinstance(graph, NodeArgsGraphInfo):
            graph.node_args = [copies.get(node, node) for node in graph.node_args]

    changed: set[torch.fx.Node] = set()
    visited: set[int] = set()
    where_casts: list[tuple[torch.fx.Node, torch.fx.Node]] = []

    def signature(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return type(value), value.dtype, tuple(value.shape), tuple(value.stride())
        if isinstance(value, (tuple, list)):
            return type(value), tuple(map(signature, value))
        return type(value)

    def update(node: torch.fx.Node, value: object) -> None:
        if signature(value) != signature(node.meta.get("val")):
            node.meta["val"] = value
            changed.add(node)

    def value(node: torch.fx.Node) -> Argument:
        return cast("Argument", node.meta.get("val"))

    def walk(graph_id: int, captures: Sequence[torch.fx.Node] = ()) -> bool:
        if graph_id in visited:
            _refuse("a shared control-flow graph")
        visited.add(graph_id)
        graph = graphs[graph_id].graph
        placeholders = list(graph.find_nodes(op="placeholder"))
        if captures:
            if len(placeholders) != len(captures):
                _refuse("a control-flow capture mismatch")
            for placeholder, capture in zip(placeholders, captures, strict=True):
                if capture in changed:
                    update(placeholder, value(capture))
        for node in list(graph.nodes):
            original = node.meta.get("val")
            if isinstance(original, _SCALARS):
                expression = original._sympy_()
                if expression not in selected:
                    origin = func.expr_to_origin.get(expression)
                    if origin is not None:
                        kind = ordered_scalar_type(origin.origin)
                        if kind is not None:
                            selected[expression] = kind
                if expression in selected and node.target in (
                    _get_tunable,
                    _tracing_ops._get_symnode,
                ):
                    scalar_type = selected[expression]
                    wrapper = {
                        int: torch.SymInt,
                        float: torch.SymFloat,
                        bool: torch.SymBool,
                    }[scalar_type]
                    update(
                        node,
                        wrapper(
                            SymNode(expression, env.shape_env, scalar_type, hint=None)
                        ),
                    )
            inputs_changed = any(arg in changed for arg in node.all_input_nodes)
            if node.op == "output":
                return inputs_changed
            if _tracing_ops.is_for_loop_target(node.target):
                child_id, _, _, child_args = node.args[:4]
                if walk(child_id, child_args):
                    _refuse("a loop-carried type change")
                continue
            if control_flow_child_graph_ids(node):
                _refuse("conditional or while-loop device graphs")
            if not inputs_changed:
                continue
            if node.op == "placeholder":
                continue
            if node.target is _tracing_ops._new_var:
                # Scalar SSA copies keep their fresh expression identity, but
                # take the selected input's type before nested captures use it.
                # Tensor copies have storage/alias metadata and are not covered
                # by this scalar transfer or the authored pointwise proof.
                (source,) = map_arg(node.args, value)
                if not isinstance(original, _SCALARS) or not isinstance(
                    source, _SCALARS
                ):
                    _refuse("a non-symbolic scalar SSA copy")
                update(
                    node,
                    type(source)(
                        SymNode(
                            original._sympy_(),
                            env.shape_env,
                            source.node.pytype,
                            hint=None,
                        )
                    ),
                )
                continue
            if (
                node.target
                in (
                    operator.add,
                    operator.sub,
                    operator.mul,
                    operator.truediv,
                    operator.neg,
                    operator.pos,
                )
                and isinstance(original, _SCALARS)
                and isinstance(node.meta.get("lowering"), SympyExprLowering)
            ):
                # Direct authored scalar arithmetic can survive as a symbolic
                # FX operation (e.g. f*f). Retype that operation too, before its
                # tensor consumer decides how the selected value promotes.
                args = map_arg(node.args, value)
                if node.kwargs or not all(
                    isinstance(arg, (*_SCALARS, int, float, bool)) for arg in args
                ):
                    _refuse("an unproved scalar arithmetic operand")
                updated_scalar = node.target(*args)
                if not isinstance(updated_scalar, _SCALARS):
                    _refuse("scalar arithmetic lost its symbolic value")
                update(node, updated_scalar)
                prepare_added_node_lowering(graph, node)
                continue
            if node.target is store:
                changed_address = []

                def check_address(
                    arg: torch.fx.Node,
                    *,
                    changed_address: list[torch.fx.Node] = changed_address,
                ) -> torch.fx.Node:
                    if arg in changed:
                        changed_address.append(arg)
                    return arg

                map_arg((node.args[:2], node.args[3:]), check_address)
                if changed_address:
                    _refuse("a store address or mask")
                continue
            if node.target is torch.ops.aten.scalar_tensor.default:
                # where's decomposition fixes a scalar cast at trace time. It
                # is reusable only when the selected scalar still promotes to
                # that dtype with the other tensor operand. Check after walking
                # all operands, including those retyped later in this graph.
                if not (
                    isinstance(node.meta.get("lowering"), AtenLowering)
                    and len(node.args) == 1
                    and node.users
                    and "dtype" in node.kwargs
                ):
                    _refuse("an unproved scalar tensor conversion")
                for user in node.users:
                    if not (
                        user.target is torch.ops.aten.where.self
                        and len(user.args) == 3
                        and (user.args[1] is node or user.args[2] is node)
                    ):
                        _refuse("a scalar conversion outside where")
                    other = user.args[2] if user.args[1] is node else user.args[1]
                    if not isinstance(other, torch.fx.Node) or not isinstance(
                        value(other), torch.Tensor
                    ):
                        _refuse("where without a tensor operand")
                    where_casts.append((node, other))
                continue
            # Older FX versions deep-copy OpOverload metadata before lowering.
            # Its full schema remains equal even when singleton identity is lost.
            original_aten = node.meta.get("original_aten")
            if (
                node.target not in _POINTWISE
                or (
                    node.target is not torch.ops.prims.convert_element_type.default
                    and original_aten is not None
                    and not (
                        isinstance(original_aten, torch._ops.OpOverload)
                        and (
                            any(
                                original_aten._schema == op._schema for op in _POINTWISE
                            )
                            or (
                                node.target is torch.ops.aten.full.default
                                and original_aten._schema
                                == torch.ops.aten.full_like.default._schema
                            )
                            or (
                                node.target is torch.ops.aten.sub.Tensor
                                and original_aten._schema
                                == torch.ops.aten.rsub.Scalar._schema
                            )
                            or (
                                node.target is torch.ops.aten.where.self
                                and any(
                                    original_aten._schema == op._schema
                                    for op in (
                                        torch.ops.aten.where.ScalarSelf,
                                        torch.ops.aten.where.ScalarOther,
                                        torch.ops.aten.where.Scalar,
                                    )
                                )
                            )
                        )
                    )
                )
                or not isinstance(
                    node.meta.get("lowering"),
                    AtenLowering
                    if node.target
                    in (torch.ops.aten.where.self, torch.ops.aten.full.default)
                    else PointwiseLowering,
                )
                or node.kwargs.get("_extra_args")
                or node.meta.get("output_nodes")
            ):
                _refuse(f"FX operation {node.target}")
            if any(arg is None for arg in node.args):
                _refuse("an operand elided by the original lowering")
            args, kwargs = map_arg((node.args, node.kwargs), value)
            updated = node.target(*args, **kwargs)
            if not isinstance(updated, torch.Tensor) or not isinstance(
                original, torch.Tensor
            ):
                _refuse("a non-tensor pointwise result")
            if tuple(updated.shape) != tuple(original.shape):
                _refuse("a pointwise shape change")
            update(node, updated)
            prepare_added_node_lowering(graph, node)
        return False

    for root_id in func.device_ir.root_ids:
        walk(root_id)
    for scalar_node, tensor_node in where_casts:
        assert isinstance(scalar_node.args[0], torch.fx.Node)
        selected_scalar = value(scalar_node.args[0])
        tensor_value = value(tensor_node)
        assert isinstance(tensor_value, torch.Tensor)
        assert isinstance(
            selected_scalar,
            (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool),
        )
        if (
            torch.result_type(tensor_value, selected_scalar)
            != scalar_node.kwargs["dtype"]
        ):
            _refuse("where changes a traced scalar conversion dtype")
