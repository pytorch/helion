"""Use exact host metadata and launch coverage to prove complete SIMT tiles.

The proof removes redundant predicates and exposes nonnegative index widening.
The original body remains the fallback for dynamic metadata, zero/partial
extents, nonunit strides, unequal tensor extents, and sizes outside the
nonnegative signed-Int32 proof domain.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from .proven_loop_bounds import _Context
from .proven_loop_bounds import _Form
from .proven_loop_bounds import _Proof
from .proven_loop_bounds import _Value
from .simplify_proven_bounds import _assigned_names
from .simplify_proven_bounds import _call_path
from .tensor_layout_relations import TensorLayoutRelation
from .tensor_layout_relations import prove_tensor_layout_relations

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from ..device_function import Argument
    from ..device_function import DeviceFunction


_LIMIT = (1 << 31) - 1
_GLOBALS = frozenset({"cute", "cutlass", "range"})


@dataclass(frozen=True)
class FullTilePlan:
    relation: TensorLayoutRelation
    block_size: int
    threads: tuple[int, int, int]
    tensors: tuple[str, ...]

    def predicate(self) -> ast.expr:
        extent = f"cute.size({self.relation.tensor}, mode=[0])"
        # Test staticness first: zero-size/unbaked schemas contain runtime
        # values, and those must not enter a const_expr comparison.
        guards = [
            condition
            for tensor in self.tensors
            for condition in (
                f"cute.is_static({tensor}.layout.shape)",
                f"cute.is_static({tensor}.layout.stride)",
            )
        ]
        guards.extend(
            (
                f"0 < {extent}",
                f"{extent} <= {_LIMIT}",
                f"{extent} % {self.block_size} == 0",
            )
        )
        for tensor in self.tensors:
            guards.append(f"{tensor}.layout.stride[0] == 1")
            if tensor != self.relation.tensor:
                guards.append(f"cute.size({tensor}, mode=[0]) == {extent}")
        return ast.parse(
            f"cutlass.const_expr({' and '.join(guards)})", mode="eval"
        ).body


def _host_linear(
    node: ast.expr,
    extent: ast.expr | None,
    constants: Mapping[str, int],
    budget: int = 64,
) -> tuple[int, int] | None:
    """An exact Python-integer affine form, without evaluating host code."""
    if budget <= 0:
        return None
    if extent is not None and ast.dump(node) == ast.dump(extent):
        return 1, 0
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return (0, node.value) if abs(node.value) <= _LIMIT else None
    if isinstance(node, ast.Name) and node.id in constants:
        return 0, constants[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _host_linear(node.operand, extent, constants, budget - 1)
        if value is None:
            return None
        sign = -1 if isinstance(node.op, ast.USub) else 1
        return sign * value[0], sign * value[1]
    if not isinstance(node, ast.BinOp):
        return None
    left = _host_linear(node.left, extent, constants, budget - 1)
    right = _host_linear(node.right, extent, constants, budget - 1)
    if left is None or right is None:
        return None
    if isinstance(node.op, (ast.Add, ast.Sub)):
        sign = -1 if isinstance(node.op, ast.Sub) else 1
        result = left[0] + sign * right[0], left[1] + sign * right[1]
    elif isinstance(node.op, ast.Mult) and (left[0] == 0 or right[0] == 0):
        result = left[0] * right[1] + right[0] * left[1], left[1] * right[1]
    elif (
        isinstance(node.op, (ast.FloorDiv, ast.Mod))
        and left[0] == right[0] == 0
        and right[1] != 0
    ):
        result = (
            0,
            left[1] // right[1]
            if isinstance(node.op, ast.FloorDiv)
            else left[1] % right[1],
        )
    else:
        return None
    return result if abs(result[0]) <= 32 and abs(result[1]) <= _LIMIT else None


def prove_full_tile_launch(
    host_body: Sequence[ast.stmt],
    *,
    kernel_name: str,
    parameter_names: Sequence[str],
    tensor_parameters: Mapping[str, int],
    integer_parameters: Collection[str],
    tensor_inputs: Mapping[str, int],
    scalar_inputs: Collection[str],
    constexpr_values: Mapping[str, int],
) -> FullTilePlan | None:
    if (
        not tensor_parameters
        or any(rank != 1 for rank in tensor_parameters.values())
        or set(parameter_names) & _GLOBALS
        or any(
            type(value) is not int or not 0 <= value <= _LIMIT
            for value in constexpr_values.values()
        )
    ):
        return None
    relations = prove_tensor_layout_relations(
        host_body,
        kernel_name=kernel_name,
        parameter_names=parameter_names,
        tensor_parameters=tensor_parameters,
        integer_parameters=integer_parameters,
        tensor_inputs=tensor_inputs,
        scalar_inputs=scalar_inputs,
        allow_flatten=True,
    )
    if not relations:
        return None
    # The relation proof already requires one direct, ordinary launch and a
    # bounded, effect-free host prefix. Reuse that same literal invocation.
    launch = next(
        statement.value
        for statement in host_body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_launcher"
    )
    if len(launch.keywords) != 1 or launch.keywords[0].arg != "block":
        return None
    block = launch.keywords[0].value
    if not (
        isinstance(block, ast.Tuple)
        and len(block.elts) == 3
        and all(
            isinstance(item, ast.Constant)
            and type(item.value) is int
            and 0 < item.value <= 1024
            for item in block.elts
        )
    ):
        return None
    threads = cast(
        "tuple[int, int, int]",
        tuple(item.value for item in block.elts if isinstance(item, ast.Constant)),
    )
    if threads[0] * threads[1] * threads[2] > 1024:
        return None
    constants = {
        name: value
        for name, value in constexpr_values.items()
        if type(value) is int and 0 <= value <= _LIMIT
    }
    for name in set(tensor_inputs) | set(scalar_inputs):
        constants.pop(name, None)
    for statement in host_body:
        if isinstance(statement, ast.Expr) and statement.value is launch:
            break
        value = (
            _host_linear(statement.value, None, constants)
            if isinstance(statement, ast.Assign)
            else None
        )
        for name in _assigned_names(statement):
            constants.pop(name, None)
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and value is not None
            and value[0] == 0
        ):
            constants[statement.targets[0].id] = value[1]
    # Device constexprs must agree with the host values actually used by this
    # invocation, rather than a later assignment or a same-named scalar input.
    if any(constants.get(name) != value for name, value in constexpr_values.items()):
        return None
    grid = launch.args[1]
    if not isinstance(grid, ast.Tuple) or not 1 <= len(grid.elts) <= 3:
        return None
    if any(_host_linear(item, None, constants) != (0, 1) for item in grid.elts[1:]):
        return None
    count = grid.elts[0]
    if not isinstance(count, ast.BinOp) or not isinstance(count.op, ast.FloorDiv):
        return None
    divisor = _host_linear(count.right, None, constants)
    if divisor is None or divisor[0] != 0 or not 0 < divisor[1] <= _LIMIT:
        return None
    for relation in relations:
        if relation.property != "size" or relation.axis != 0:
            continue
        extent = launch.args[parameter_names.index(relation.scalar) + 2]
        numerator = _host_linear(count.left, extent, constants)
        if numerator not in ((1, 0), (1, divisor[1] - 1)):
            continue
        return FullTilePlan(
            relation,
            divisor[1],
            (threads[0], threads[1], threads[2]),
            tuple(tensor_parameters),
        )
    return None


class _FullTileProof(_Proof):
    def __init__(self, plan: FullTilePlan, renames: Mapping[str, str]) -> None:
        self.plan = plan
        self.renames = renames
        self.maximum_blocks = _LIMIT // plan.block_size
        super().__init__(plan.threads, (self.maximum_blocks, 1, 1))

    def canonical(self, name: str) -> str:
        return self.renames.get(name, name)

    def value(
        self, node: ast.expr, context: _Context, budget: int = 128
    ) -> _Value | None:
        if isinstance(node, ast.Name):
            return context.values.get(self.canonical(node.id))
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
            and node.slice.value == 0
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "stride"
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "layout"
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id in self.plan.tensors
        ):
            return _Value(1, 1, _Form(1))
        return super().value(node, context, budget)

    def true_comparison(self, node: ast.Compare, context: _Context) -> bool:
        if len(node.ops) == len(node.comparators) == 1 and isinstance(
            node.ops[0], ast.Eq
        ):
            left = self.value(node.left, context)
            right = self.value(node.comparators[0], context)
            return left is not None and right is not None and left.form == right.form
        return super().true_comparison(node, context)

    def expression(self, expression: ast.expr, context: _Context) -> ast.expr:
        expression = super().expression(expression, context)
        proof = self

        class Widen(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                node = cast("ast.Call", self.generic_visit(node))
                if (
                    _call_path(node.func) != ("cutlass", "Int64")
                    or len(node.args) != 1
                    or node.keywords
                    or not isinstance(node.args[0], ast.Name)
                ):
                    return node
                value = proof.value(node.args[0], context)
                if value is None or value.low == value.high:
                    return node
                # The same checked affine proof rejects negative values and
                # every intermediate that could wrap signed Int32. Exposing
                # the zero high word is therefore exact, while allowing the
                # device compiler to simplify later shifts and bit operations.
                assert 0 <= value.low <= value.high <= _LIMIT
                node.args[0] = ast.copy_location(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="cutlass", ctx=ast.Load()),
                            attr="Uint32",
                            ctx=ast.Load(),
                        ),
                        args=[node.args[0]],
                        keywords=[],
                    ),
                    node.args[0],
                )
                proof.changed += 1
                return node

        return cast("ast.expr", Widen().visit(expression))

    def forget(self, statement: ast.AST, context: _Context) -> None:
        for name in _assigned_names(statement):
            context.values.pop(self.canonical(name), None)

    def block(self, statements: list[ast.stmt], context: _Context) -> list[ast.stmt]:
        result = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                statement.value = self.expression(statement.value, context)
                value = self.value(statement.value, context)
                self.forget(statement, context)
                if isinstance(statement.targets[0], ast.Name) and value is not None:
                    context.values[self.canonical(statement.targets[0].id)] = value
            elif isinstance(statement, ast.For):
                induction = self.loop_value(statement, context)
                local = context.copy()
                self.forget(statement, local)
                if induction is not None:
                    value, symbol, upper = induction
                    assert isinstance(statement.target, ast.Name)
                    local.values[self.canonical(statement.target.id)] = value
                    local.upper[symbol] = upper
                statement.body = self.block(statement.body, local)
                self.forget(statement, context)
                statement.orelse = self.block(statement.orelse, context.copy())
            elif isinstance(statement, ast.If):
                statement.test = self.expression(statement.test, context)
                if (
                    isinstance(statement.test, ast.Constant)
                    and statement.test.value is True
                ):
                    self.changed += 1
                    result.extend(self.block(statement.body, context))
                    continue
                statement.body = self.block(statement.body, context.copy())
                statement.orelse = self.block(statement.orelse, context.copy())
                self.forget(statement, context)
            elif isinstance(statement, ast.Expr):
                statement.value = self.expression(statement.value, context)
            result.append(statement)
        return result


def _supported(body: Sequence[ast.stmt]) -> bool:
    for statement in body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                return False
            target = statement.targets[0]
            if isinstance(target, (ast.Tuple, ast.List)):
                if not all(isinstance(item, ast.Name) for item in target.elts):
                    return False
            elif not isinstance(target, (ast.Name, ast.Subscript)):
                return False
        elif isinstance(statement, (ast.If, ast.For)):
            if isinstance(statement, ast.For) and not isinstance(
                statement.target, ast.Name
            ):
                return False
            if not _supported(statement.body) or not _supported(statement.orelse):
                return False
        elif not isinstance(statement, (ast.Expr, ast.Pass)):
            return False
    return True


def specialize_full_tile_bounds(
    body: list[ast.stmt],
    plan: FullTilePlan,
    *,
    argument_names: Collection[str],
    constexpr_values: Mapping[str, int],
    rename_groups: Mapping[str, str],
    packet_prefetch: int = 0,
    proven_disjoint_tensor_pairs: AbstractSet[frozenset[str]] = frozenset(),
) -> list[ast.stmt]:
    from ..device_function import _clone_extended_ast

    module = ast.Module(body=body, type_ignores=[])
    nodes = list(ast.walk(module))
    if len(nodes) > 32768 or not _supported(body):
        return body
    immutable = {
        rename_groups.get(name, name)
        for name in set(argument_names) | set(constexpr_values) | _GLOBALS
    }
    if any(
        rename_groups.get(name, name) in immutable for name in _assigned_names(module)
    ):
        return body
    if any(
        isinstance(
            node,
            (
                ast.NamedExpr,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for node in nodes
    ):
        return body
    proof = _FullTileProof(plan, rename_groups)
    quotient = proof.symbol(("full_tile_extent",), 1, proof.maximum_blocks)
    block = proof.symbol(("block", 0), 0, proof.maximum_blocks - 1)
    upper = quotient.add(_Form(-1))
    assert upper is not None
    context = _Context(
        {
            proof.canonical(name): _Value(value, value, _Form(value))
            for name, value in constexpr_values.items()
        },
        {block.terms[0][0]: upper},
    )
    context.values[proof.canonical(plan.relation.scalar)] = _Value(
        plan.block_size,
        proof.maximum_blocks * plan.block_size,
        quotient.scale(plan.block_size),
    )
    optimized = proof.block(cast("list[ast.stmt]", _clone_extended_ast(body)), context)
    if proof.changed == 0:
        return body
    if packet_prefetch:
        from .prefetch_pointwise_packets import prefetch_pointwise_packets

        names = {
            node.id
            for statement in optimized
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }
        # The packet matcher reasons about raw names. Decline if final
        # renaming merges distinct bindings and could hide a recurrence.
        canonical_names = {rename_groups.get(name, name) for name in names}
        if len(canonical_names) == len(names):
            optimized = prefetch_pointwise_packets(
                optimized,
                batch_size=packet_prefetch,
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                reserved_names=(
                    set(argument_names)
                    | set(constexpr_values)
                    | set(rename_groups)
                    | set(rename_groups.values())
                ),
            )
    return [
        ast.If(
            test=plan.predicate(),
            body=[plan.relation.assignment(), *optimized],
            orelse=body,
        )
    ]


def lower_full_tile_bounds(
    body: list[ast.stmt],
    function: DeviceFunction,
    arguments: Sequence[Argument],
    constants: Mapping[str, int],
    rename_groups: Mapping[str, str],
) -> list[ast.stmt]:
    if not function.config.config.get("cute_proven_bounds", False):
        return body
    import torch

    from ..compile_environment import CompileEnvironment
    from ..device_function import NumericArgument
    from ..device_function import TensorArg
    from ..device_function import TensorDescriptorArg
    from ..device_function import TensorPropertyArg
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    if (
        not env.config_spec.pointwise_facts
        or function.codegen.cute_wrapper_plans
        or function.codegen.cute_uses_matmul
        or function.cute_state.simt_cluster_n != 1
        or function._scratch_args
        or function.wrapper_only_params
        or function.codegen._extra_params
        or any(isinstance(arg, TensorDescriptorArg) for arg in arguments)
        or any(
            not isinstance(arg, (TensorArg, NumericArgument, TensorPropertyArg))
            for arg in arguments
        )
    ):
        return body
    inputs = HostFunction.current().params.arguments
    names = tuple(arg.name for arg in arguments)
    plan = prove_full_tile_launch(
        cast("list[ast.stmt]", function.codegen.host_statements),
        kernel_name=function.name,
        parameter_names=names,
        tensor_parameters={
            arg.name: arg.fake_value.ndim
            for arg in arguments
            if isinstance(arg, TensorArg)
        },
        integer_parameters={
            arg.name
            for expression, arg in function._expr_args.items()
            if expression.is_integer is True  # pyrefly: ignore[missing-attribute]
        }
        | {arg.name for arg in arguments if isinstance(arg, TensorPropertyArg)},
        tensor_inputs={
            name: value.ndim
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        scalar_inputs={
            name
            for name, value in inputs.items()
            if isinstance(value, (bool, int, float, torch.SymInt, torch.SymFloat))
        },
        constexpr_values=constants,
    )
    if plan is None:
        return body
    packet_prefetch = function.config.cute_packet_prefetch
    return specialize_full_tile_bounds(
        body,
        plan,
        argument_names=names,
        constexpr_values=constants,
        rename_groups=rename_groups,
        packet_prefetch=packet_prefetch,
        proven_disjoint_tensor_pairs=(
            function.proven_disjoint_tensor_pairs() if packet_prefetch else frozenset()
        ),
    )
