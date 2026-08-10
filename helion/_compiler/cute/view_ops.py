"""CuTe-backend codegen for ops defined in ``helion.language.view_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``view_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language.view_ops import join
from ...language.view_ops import split
from ...language.view_ops import subscript
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(subscript, "cute")
def _(state: CodegenState) -> ast.AST:
    # CuTe kernels currently execute scalarized pointwise code, so shape-only
    # indexing used for broadcast setup is a no-op.
    return state.ast_arg(0)


@_decorators.codegen(split, "cute")
def _(state: CodegenState) -> list[ast.AST]:
    from ..ast_extension import statement_from_string
    from ..generate_ast import GenerateAST
    from .cute_reshape import _flat_index_from_coords
    from .cute_reshape import _get_node_dim_local_coord
    from .cute_reshape import _get_tile_shape

    fx_node = state.fx_node
    assert fx_node is not None
    input_node = fx_node.args[0]
    assert isinstance(input_node, torch.fx.Node)
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    output_val = input_val.new_empty(input_val.shape[:-1])

    cg = state.codegen
    assert isinstance(cg, GenerateAST)
    df = cg.device_function
    env = CompileEnvironment.current()
    config = df.config

    input_shape = _get_tile_shape(input_val, env, config)
    output_shape = _get_tile_shape(output_val, env, config)

    input_numel = 1
    for s in input_shape:
        input_numel *= s

    dtype_str = env.backend.dtype_str(input_val.dtype)

    smem_ptr = df.new_var("split_smem_ptr")
    smem = df.new_var("split_smem")

    src_coords = [
        _get_node_dim_local_coord(cg, input_node, input_val, i)
        for i in range(len(input_shape))
    ]
    src_flat = _flat_index_from_coords(src_coords, input_shape)

    if output_shape:
        output_coords = [
            _get_node_dim_local_coord(cg, input_node, output_val, i)
            for i in range(len(output_shape))
        ]
        out_flat_base = _flat_index_from_coords(output_coords, output_shape)
    else:
        out_flat_base = "cutlass.Int32(0)"
    lo_flat = f"({out_flat_base}) * cutlass.Int32(2)"
    hi_flat = f"({out_flat_base}) * cutlass.Int32(2) + cutlass.Int32(1)"

    cg.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({dtype_str}, {input_numel})"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"{smem} = cute.make_tensor({smem_ptr}, ({input_numel},))"
        )
    )
    cg.add_statement(
        statement_from_string(f"{smem}[{src_flat}] = {{_inp}}", _inp=state.ast_arg(0))
    )
    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))

    lo_var = df.new_var("split_lo")
    hi_var = df.new_var("split_hi")
    cg.add_statement(statement_from_string(f"{lo_var} = {smem}[{lo_flat}]"))
    cg.add_statement(statement_from_string(f"{hi_var} = {smem}[{hi_flat}]"))

    return [
        expr_from_string(lo_var),
        expr_from_string(hi_var),
    ]


@_decorators.codegen(join, "cute")
def _(state: CodegenState) -> ast.AST:
    from ..generate_ast import GenerateAST
    from .cute_reshape import _get_dim_local_coord

    fx_node = state.fx_node
    assert fx_node is not None
    output_val = fx_node.meta["val"]
    assert isinstance(output_val, torch.Tensor)
    assert isinstance(state.codegen, GenerateAST)

    new_dim = output_val.ndim - 1
    selector = _get_dim_local_coord(state.codegen, output_val, new_dim)

    return expr_from_string(
        f"(({{a}}) if ({selector}) == cutlass.Int32(0) else ({{b}}))",
        a=state.ast_arg(0),
        b=state.ast_arg(1),
    )
