from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING
from typing import TypeVar

from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from . import _decorators
from .inline_triton_ops import _collect_output_metadata
from .inline_triton_ops import _emit_output_assertions
from .inline_triton_ops import _ensure_name
from .inline_triton_ops import _fake_outputs
from .inline_triton_ops import _format_triton_source
from .inline_triton_ops import _get_or_add_triton_function_preamble
from .inline_triton_ops import _parse_triton_source
from .inline_triton_ops import _validate_args

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .._compiler.inductor_lowering import CodegenState

    _T = TypeVar("_T")

__all__ = ["gluon_kernel", "inline_gluon"]


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def inline_gluon(
    gluon_source: str,
    args: Sequence[object] | Mapping[str, object],
    output_like: _T,
) -> _T:
    """Inline a raw Gluon snippet inside a Helion kernel.

    Gluon is Triton's next-generation dialect for programming Blackwell GPUs,
    exposing features such as TMEM, warp specialization, and TMA descriptors.

    Args:
        gluon_source: The Gluon code snippet. The last statement must be an
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


@_decorators.register_fake(inline_gluon)
def _(
    gluon_source: str,
    args: object,
    output_like: object,
) -> object:
    if not isinstance(gluon_source, str):
        raise exc.InvalidAPIUsage(
            f"gluon_source must be a string, got {type(gluon_source)}"
        )
    _validate_args(args)
    return _fake_outputs(output_like)


@_decorators.codegen(inline_gluon, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    gluon_source = state.proxy_arg(0)
    args_obj = state.proxy_arg(1)
    output_like = state.proxy_arg(2)

    if not isinstance(gluon_source, str):  # defensive; validated earlier
        raise exc.InvalidAPIUsage(
            f"gluon_source must be a string, got {type(gluon_source)}"
        )

    formatted = _format_triton_source(
        state,
        gluon_source,
        args_obj,
        state.ast_args[1],
    )

    statements, result_expr = _parse_triton_source(
        formatted, require_expression=output_like is not None
    )
    for stmt in statements:
        state.add_statement(stmt)

    if output_like is None:
        if result_expr is not None:
            state.add_statement(create(ast.Expr, value=result_expr))
        return create(ast.Constant, value=None)

    result_name = state.device_function.new_var("inline_gluon_result")
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


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def gluon_kernel(
    gluon_source_or_fn: object,
    args: Sequence[object] | Mapping[str, object],
    output_like: _T,
) -> _T:
    """Define (once) and call a @triton.jit function using Gluon primitives.

    Use this instead of ``inline_gluon()`` when you need a complete function
    definition rather than an inline snippet. The function is emitted at module
    scope once and then invoked from the kernel body.

    Gluon functions use ``@triton.jit`` (same as regular Triton) but call
    ``gl.*`` primitives for Blackwell-specific features like TMEM, warp
    specialization, and TMA descriptors.

    Args:
        gluon_source_or_fn: Source for a single @triton.jit function definition
            using Gluon primitives, or a Python function object.
        args: Positional or keyword placeholders that will be substituted via
            name resolution of Helion variables.
        output_like: Example tensor(s) describing the expected outputs for
            shape/dtype checks.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(gluon_kernel)
def _(
    gluon_source_or_fn: object,
    args: object,
    output_like: object,
) -> object:
    if not (
        isinstance(gluon_source_or_fn, str) or inspect.isfunction(gluon_source_or_fn)
    ):
        raise exc.InvalidAPIUsage(
            f"gluon_kernel expects a string source or a function, got {type(gluon_source_or_fn)}"
        )
    _validate_args(args)
    return _fake_outputs(output_like)


@_decorators.codegen(gluon_kernel, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    from typing import cast

    gluon_source_or_fn = state.proxy_arg(0)
    args_obj = state.proxy_arg(1)
    output_like = state.proxy_arg(2)

    if not (
        isinstance(gluon_source_or_fn, str) or inspect.isfunction(gluon_source_or_fn)
    ):
        raise exc.InvalidAPIUsage(
            f"gluon_kernel expects a string source or a function, got {type(gluon_source_or_fn)}"
        )
    _validate_args(args_obj)

    # Install the Gluon function into preamble (once) and get the callable name
    fn_name = _get_or_add_triton_function_preamble(state, gluon_source_or_fn)

    # Resolve argument names similar to inline_gluon formatting
    call_args_src = ""
    if isinstance(state.ast_args[1], dict):
        kw_pairs: list[str] = []
        mapping = cast("Mapping[str, object]", args_obj)
        for key, node in state.ast_args[1].items():
            kw_pairs.append(f"{key}=" + _ensure_name(state, node, mapping[key]))
        call_args_src = ", ".join(kw_pairs)
    else:
        if not isinstance(state.ast_args[1], (ast.List, ast.Tuple, list, tuple)):
            raise exc.InvalidAPIUsage(
                "gluon_kernel expects a literal list/tuple for positional args"
            )
        arg_nodes = (
            state.ast_args[1].elts
            if isinstance(state.ast_args[1], (ast.List, ast.Tuple))
            else list(state.ast_args[1])
        )
        names = [
            _ensure_name(state, node, arg)
            for node, arg in zip(
                arg_nodes, cast("Sequence[object]", args_obj), strict=False
            )
        ]
        call_args_src = ", ".join(names)

    call_expr = expr_from_string(f"{fn_name}({call_args_src})")

    if output_like is None:
        state.add_statement(create(ast.Expr, value=call_expr))
        return create(ast.Constant, value=None)

    result_name = state.device_function.new_var("gluon_kernel_result")
    assign = create(
        ast.Assign,
        targets=[create(ast.Name, id=result_name, ctx=ast.Store())],
        value=call_expr,
    )
    state.add_statement(assign)

    dtypes, output_nodes, is_multi = _collect_output_metadata(
        output_like, state.ast_args[2]
    )
    _emit_output_assertions(state, result_name, dtypes, output_nodes, is_multi)

    if is_multi:
        return [expr_from_string(f"{result_name}[{i}]") for i in range(len(dtypes))]
    return expr_from_string(result_name)
