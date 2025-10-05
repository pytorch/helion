from __future__ import annotations

import ast
from collections.abc import Mapping
from collections.abc import Sequence
import textwrap
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import torch
from torch._inductor.utils import triton_type
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import convert
from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

    _T = TypeVar("_T")

__all__ = ["inline_triton"]


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def inline_triton(
    triton_source: str,
    args: Sequence[object] | Mapping[str, object],
    output_like: _T,
) -> _T:
    """Inline a raw Triton snippet inside a Helion kernel.

    Args:
        triton_source: The Triton code snippet. The last statement must be an
            expression representing the return value. The snippet may be
            indented, and common indentation is stripped automatically.
        args: Positional or keyword placeholders that will be substituted via
            ``str.format`` before code generation. Provide a tuple/list for
            positional placeholders (``{0}``, ``{1}``, ...) or a mapping for
            named placeholders (``{x}``, ``{y}``, ...).
        output_like: Example tensors describing the expected outputs. A single
            tensor indicates a single output; a tuple/list of tensors indicates
            multiple outputs.

    Returns:
        The value(s) produced by the snippet. Matches the structure of
        ``output_like``.
    """
    raise exc.NotInsideKernel


def _validate_args(args: object) -> None:
    if isinstance(args, Mapping):
        return
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
        return
    raise exc.InvalidAPIUsage("inline_triton args must be a tuple/list or a mapping")


def _fake_outputs(output_like: object) -> object:
    if output_like is None:
        return None
    if isinstance(output_like, torch.Tensor):
        return torch.empty_like(output_like)
    if isinstance(output_like, Sequence) and not isinstance(output_like, (str, bytes)):
        outputs = []
        for i, item in enumerate(output_like):
            if not isinstance(item, torch.Tensor):
                raise exc.InvalidAPIUsage(
                    f"output_like[{i}] must be a torch.Tensor, got {type(item)}"
                )
            outputs.append(torch.empty_like(item))
        return type(output_like)(outputs) if isinstance(output_like, tuple) else outputs
    raise exc.InvalidAPIUsage(
        "output_like must be a tensor or a sequence of tensors or None"
    )


@_decorators.register_fake(inline_triton)
def _(
    triton_source: str,
    args: object,
    output_like: object,
) -> object:
    if not isinstance(triton_source, str):
        raise exc.InvalidAPIUsage(
            f"triton_source must be a string, got {type(triton_source)}"
        )
    _validate_args(args)
    return _fake_outputs(output_like)


def _ensure_name(state: CodegenState, node: ast.AST) -> str:
    lifted = state.codegen.lift(node)
    assert isinstance(lifted, ast.Name)
    return lifted.id


def _format_triton_source(
    state: CodegenState,
    triton_source: str,
    args_obj: object,
    args_ast: object,
) -> str:
    source = textwrap.dedent(triton_source).strip()
    if not source:
        raise exc.InvalidAPIUsage("triton_source must contain code")

    if isinstance(args_obj, Mapping):
        if not isinstance(args_ast, dict):
            raise exc.InvalidAPIUsage(
                "inline_triton expects a dict literal when args is a mapping"
            )
        assert args_obj.keys() == args_ast.keys()
        format_args: dict[str, str] = {
            key: _ensure_name(state, args_ast[key]) for key in args_ast
        }
        try:
            return source.format(**format_args)
        except (KeyError, IndexError, ValueError) as exc_value:
            raise exc.InvalidAPIUsage(
                f"Failed to format triton_source with mapping args: {exc_value}"
            ) from exc_value

    if isinstance(args_obj, Sequence) and not isinstance(args_obj, (str, bytes)):
        if not isinstance(args_ast, (ast.List, ast.Tuple, list, tuple)):
            raise exc.InvalidAPIUsage(
                "inline_triton expects a list/tuple literal when args is a sequence"
            )
        arg_nodes = (
            args_ast.elts
            if isinstance(args_ast, (ast.List, ast.Tuple))
            else list(args_ast)
        )
        names = [_ensure_name(state, node) for node in arg_nodes]
        try:
            expected_len = len(args_obj)
        except TypeError:  # pragma: no cover - defensive
            expected_len = len(names)
        if expected_len != len(names):
            raise exc.InvalidAPIUsage(
                "inline_triton sequence args must be provided as a literal"
            )
        try:
            return source.format(*names)
        except (IndexError, ValueError) as exc_value:
            raise exc.InvalidAPIUsage(
                f"Failed to format triton_source with positional args: {exc_value}"
            ) from exc_value

    raise exc.InvalidAPIUsage("inline_triton args must be a tuple/list or a mapping")


def _parse_triton_source(source: str) -> tuple[list[ast.stmt], ast.AST]:
    try:
        module = ast.parse(source)
    except SyntaxError as exc_value:
        raise exc.InvalidAPIUsage(
            f"Failed to parse triton_source: {exc_value}"
        ) from exc_value

    if not module.body:
        raise exc.InvalidAPIUsage("triton_source must contain at least one expression")

    *prefix, last = module.body
    if not isinstance(last, ast.Expr):
        raise exc.InvalidAPIUsage(
            "The last line of triton_source must be an expression"
        )

    converted_prefix = [cast("ast.stmt", convert(stmt)) for stmt in prefix]
    return converted_prefix, convert(last.value)


def _normalize_output_ast(output_ast: object) -> list[ast.AST]:
    if isinstance(output_ast, (ast.Tuple, ast.List)):
        return [cast("ast.AST", elem) for elem in output_ast.elts]
    if isinstance(output_ast, (tuple, list)):
        nodes: list[ast.AST] = []
        for elem in output_ast:
            if not isinstance(elem, ast.AST):
                raise exc.InvalidAPIUsage(
                    "output_like literal must reference tensors directly"
                )
            nodes.append(elem)
        return nodes
    if isinstance(output_ast, ast.AST):
        return [output_ast]
    raise exc.InvalidAPIUsage(
        "output_like must be provided as a tensor or tuple/list literal"
    )


def _collect_output_metadata(
    output_like: object,
    output_ast: object,
) -> tuple[list[torch.dtype], list[ast.AST], bool]:
    if output_like is None:
        return [], [], False
    if isinstance(output_like, torch.Tensor):
        return [output_like.dtype], _normalize_output_ast(output_ast), False
    if isinstance(output_like, Sequence) and not isinstance(output_like, (str, bytes)):
        if not output_like:
            raise exc.InvalidAPIUsage("output_like sequence must not be empty")
        ast_nodes = _normalize_output_ast(output_ast)
        dtypes: list[torch.dtype] = []
        for i, item in enumerate(output_like):
            if not isinstance(item, torch.Tensor):
                raise exc.InvalidAPIUsage(
                    f"output_like[{i}] must be a torch.Tensor, got {type(item)}"
                )
            dtypes.append(item.dtype)
        if len(dtypes) != len(ast_nodes):
            raise exc.InvalidAPIUsage(
                "output_like literal must match the structure passed into inline_triton"
            )
        return dtypes, ast_nodes, True
    raise exc.InvalidAPIUsage(
        "output_like must be a tensor or a sequence of tensors or None"
    )


def _emit_output_assertions(
    state: CodegenState,
    result_name: str,
    dtypes: list[torch.dtype],
    output_nodes: list[ast.AST],
    is_multi: bool,
) -> None:
    if not dtypes:
        return

    if not is_multi:
        lhs = expr_from_string(f"{result_name}.dtype")
        rhs = expr_from_string(triton_type(dtypes[0]))
        msg = ast.Constant(
            value=f"inline_triton output dtype mismatch; expected {dtypes[0]}"
        )
        state.add_statement(
            statement_from_string(
                "tl.static_assert({lhs} == {rhs}, {msg})", lhs=lhs, rhs=rhs, msg=msg
            )
        )
        shape_lhs = expr_from_string(f"{result_name}.shape")
        shape_rhs = expr_from_string("{value}.shape", value=output_nodes[0])
        shape_msg = ast.Constant(value="inline_triton output shape mismatch")
        state.add_statement(
            statement_from_string(
                "tl.static_assert({lhs} == {rhs}, {msg})",
                lhs=shape_lhs,
                rhs=shape_rhs,
                msg=shape_msg,
            )
        )
        return

    count_msg = ast.Constant(value=f"inline_triton expected {len(dtypes)} outputs")
    state.add_statement(
        statement_from_string(
            "tl.static_assert(len({result}) == {count}, {msg})",
            result=expr_from_string(result_name),
            count=ast.Constant(value=len(dtypes)),
            msg=count_msg,
        )
    )

    for index, dtype in enumerate(dtypes):
        lhs = expr_from_string(f"{result_name}[{index}].dtype")
        rhs = expr_from_string(triton_type(dtype))
        msg = ast.Constant(
            value=f"inline_triton output {index} dtype mismatch; expected {dtype}"
        )
        state.add_statement(
            statement_from_string(
                "tl.static_assert({lhs} == {rhs}, {msg})",
                lhs=lhs,
                rhs=rhs,
                msg=msg,
            )
        )
        shape_lhs = expr_from_string(f"{result_name}[{index}].shape")
        shape_rhs = expr_from_string("{value}.shape", value=output_nodes[index])
        shape_msg = ast.Constant(value=f"inline_triton output {index} shape mismatch")
        state.add_statement(
            statement_from_string(
                "tl.static_assert({lhs} == {rhs}, {msg})",
                lhs=shape_lhs,
                rhs=shape_rhs,
                msg=shape_msg,
            )
        )


@_decorators.codegen(inline_triton)
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    triton_source = state.proxy_arg(0)
    args_obj = state.proxy_arg(1)
    output_like = state.proxy_arg(2)

    if not isinstance(triton_source, str):  # defensive; validated earlier
        raise exc.InvalidAPIUsage(
            f"triton_source must be a string, got {type(triton_source)}"
        )

    formatted = _format_triton_source(
        state,
        triton_source,
        args_obj,
        state.ast_args[1],
    )

    statements, result_expr = _parse_triton_source(formatted)
    for stmt in statements:
        state.add_statement(stmt)

    if output_like is None:
        state.add_statement(create(ast.Expr, value=result_expr))
        return create(ast.Constant, value=None)

    result_name = state.device_function.new_var("inline_triton_result")
    assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=result_name, ctx=ast.Store())],
        value=result_expr,
    )
    state.add_statement(assign)

    dtypes, output_nodes, is_multi = _collect_output_metadata(
        output_like, state.ast_args[2]
    )
    _emit_output_assertions(state, result_name, dtypes, output_nodes, is_multi)

    if is_multi:
        return [expr_from_string(f"{result_name}[{i}]") for i in range(len(dtypes))]

    return expr_from_string(result_name)
