from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

import helion
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.inductor_lowering import PointwiseLowering
from helion._testing.tunable_fragments import load_host_function
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.aten_lowering import LoweringContext
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel

_REGISTER_SELECTED = hl.register_tunable


@helion.kernel(autotune_effort="none")
def mixed_selected_scalar_uses(
    x: torch.Tensor, operation: hl.constexpr
) -> torch.Tensor:
    selected = hl.register_tunable("selected", EnumFragment((0.5, 1, 2)))
    if operation == "host_arithmetic":
        selected = selected * 2
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if operation == "clamp":
            out[tile] = x[tile].clamp(max=selected)
        elif operation == "full_like":
            out[tile] = torch.full_like(x[tile], selected)
        elif operation == "device_arithmetic":
            out[tile] = x[tile] * (selected * 2)
        else:
            out[tile] = x[tile] * selected
    return out


def emitted_device_arguments(
    bound: BoundKernel[object], config: Config, source: str, args: tuple[object, ...]
) -> dict[str, object]:
    """Run the generated wrapper and observe its single native call's inputs."""
    device = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == f"_helion_{bound.kernel.fn.__name__}"
    )
    launches: list[tuple[object, ...]] = []

    def observe(
        _kernel: object, _grid: object, *args: object, **kwargs: object
    ) -> None:
        launches.append(args)

    load_host_function(bound, config, observe)(*args)
    (arguments,) = launches
    return dict(zip((arg.arg for arg in device.args.args), arguments, strict=True))


def render_scalar_operands(
    bound: BoundKernel[object], config: Config
) -> tuple[str, list[ast.expr]]:
    """Observe emitted scalars consumed by multiplication, through SSA copies.

    These fixtures have straight-line scalar definitions, possibly captured by
    nested tiles. Resolve actual generated assignments rather than requiring a
    particular symbolic-expression node or assuming copies keep its symbol.
    Ambiguous reassignments and cycles are deliberately not interpreted.
    """
    operands: list[ast.expr] = []
    original = PointwiseLowering.codegen

    def expand(expression: ast.expr, statements: list[ast.AST]) -> ast.expr:
        definitions: dict[str, list[ast.expr]] = {}
        seen: set[int] = set()
        for statement in statements:
            for item in ast.walk(statement):
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and id(item) not in seen
                ):
                    seen.add(id(item))
                    definitions.setdefault(item.targets[0].id, []).append(item.value)

        class Expand(ast.NodeTransformer):
            def __init__(self) -> None:
                self.active: set[str] = set()

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id not in definitions:
                    return node
                assert node.id not in self.active, node.id
                (definition,) = definitions[node.id]
                self.active.add(node.id)
                result = self.visit(
                    ast.parse(ast.unparse(definition), mode="eval").body
                )
                self.active.remove(node.id)
                return result

        return cast(
            "ast.expr",
            Expand().visit(ast.parse(ast.unparse(expression), mode="eval").body),
        )

    def observe(
        self: PointwiseLowering, ctx: LoweringContext, node: torch.fx.Node
    ) -> object:
        if node.target is torch.ops.aten.mul.Tensor:
            assert isinstance(ctx.cg, GenerateAST)
            # The binary operator's positional arguments are its value operands.
            for operand in node.args:
                if isinstance(operand, torch.fx.Node) and isinstance(
                    operand.meta["val"], (torch.SymInt, torch.SymFloat, torch.SymBool)
                ):
                    operands.append(
                        expand(
                            cast("ast.expr", ctx.env[operand]),
                            [
                                stmt
                                for stack in ctx.cg.statements_stack
                                for stmt in stack
                            ],
                        )
                    )
        return original(self, ctx, node)

    with patch.object(PointwiseLowering, "codegen", observe):
        source = bound.to_code(config)
    device = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == f"_helion_{bound.kernel.fn.__name__}"
    )
    assert operands, "No scalar reached the tensor multiplication"
    # Loop prefixes are published after the child is lowered. Resolve those
    # against the finished function after resolving copies that DCE can remove.
    return source, [expand(operand, [device]) for operand in operands]


@helion.kernel(autotune_effort="none", static_shapes=False)
def mixed_selected_nested_arithmetic(
    x: torch.Tensor, choices: hl.constexpr, reverse: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        selected = hl.register_tunable("selected", EnumFragment(choices))
        for column in hl.tile(x.size(1)):
            if reverse:
                out[row, column] = selected - x[row, column]
            else:
                out[row, column] = x[row, column] * selected + 1
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def mixed_selected_tensor_product(
    x: torch.Tensor, nested: hl.constexpr
) -> torch.Tensor:
    selected = _REGISTER_SELECTED("selected", EnumFragment((2, 4.5)))
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        if nested:
            for column in hl.tile(x.size(1)):
                out[row, column] = x[row, column] * selected
        else:
            out[row, :] = x[row, :] * selected
    return out


@helion.kernel(autotune_effort="none", static_shapes=False, fast_math=False)
def nested_selected_float_results(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    direct = torch.empty_like(x)
    divided = torch.empty_like(x)
    negated = torch.empty_like(x)
    runtime = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        value = hl.register_tunable("value", EnumFragment((-0.0, 1.0)))
        half = value / 2
        negative = -value
        scaled = value * x.size(0)
        for column in hl.tile(x.size(1)):
            direct[row, column] = x[row, column] * value
            divided[row, column] = x[row, column] * half
            negated[row, column] = x[row, column] * negative
            runtime[row, column] = x[row, column] * scaled
    return direct, divided, negated, runtime


@helion.kernel(autotune_effort="none")
def mixed_selected_tensor_forms(
    x: torch.Tensor, operation: hl.constexpr
) -> torch.Tensor:
    selected = hl.register_tunable("selected", EnumFragment((1, 0.5)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if operation == "augassign":
            acc = torch.zeros_like(x[tile], dtype=torch.float32)
            acc += x[tile] * selected
            out[tile] = acc
        elif operation == "where":
            out[tile] = torch.where(x[tile] > 0, x[tile], selected)
        elif operation == "method_alias":
            multiply = x[tile].mul
            out[tile] = multiply(selected)
        else:
            out[tile] = x[tile].mul(selected)
    return out


@helion.kernel(autotune_effort="none")
def selected_boolean_forms(x: torch.Tensor, operation: hl.constexpr) -> torch.Tensor:
    flag = hl.register_tunable("flag", BooleanFragment())
    if operation == "arithmetic":
        out = torch.empty_like(x, dtype=torch.int64)
    else:
        out = torch.empty_like(x)

    def hidden() -> bool:
        return flag

    for tile in hl.tile(x.numel()):
        if operation == "branch":
            if flag:
                out[tile] = x[tile] + 2
            else:
                out[tile] = x[tile] - 3
        elif operation == "helper":
            if hidden():
                out[tile] = x[tile] + 2
            else:
                out[tile] = x[tile] - 3
        else:
            out[tile] = x[tile] * flag + flag
    return out


@helion.kernel(autotune_effort="none")
def selected_tensor_cast_chain(x: torch.Tensor, mode: hl.constexpr) -> torch.Tensor:
    selected = hl.register_tunable("selected", EnumFragment((2, 0.5)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = x[tile] * selected
        if mode == "chain":
            out[tile] = value.to(torch.int32).to(x.dtype) + selected * selected
        elif mode == "alias_chain":
            out[tile] = value.int().float() + selected * selected
        elif mode == "erased":
            out[tile] = value.to(torch.int32) + selected * selected
        elif mode == "erased_alias":
            out[tile] = value.int() + selected * selected
        else:
            out[tile] = value.to(value.dtype) + selected * selected
    return out


@helion.kernel(autotune_effort="none")
def selected_tensor_unary(x: torch.Tensor, operation: hl.constexpr) -> torch.Tensor:
    selected = hl.register_tunable("selected", EnumFragment((2, 0.5)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = x[tile] * selected
        if operation == "abs":
            result = torch.abs(value)
        elif operation == "absolute":
            result = torch.absolute(value)
        elif operation == "method_abs":
            result = value.abs()
        elif operation == "method_absolute":
            result = value.absolute()
        elif operation == "neg":
            result = torch.neg(value)
        elif operation == "negative":
            result = torch.negative(value)
        elif operation == "method_negative":
            result = value.negative()
        else:
            result = value.neg()
        out[tile] = result + selected
    return out
