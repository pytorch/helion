"""Emit a certified split reduction with one weight vector per CTA."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import codegen_call_with_graph
from ..tile_strategy import DeviceGridState
from .analysis.match_common import input_closure
from .analysis.split_weighted_merge import prove_split_weighted_merge
from .gathered_launch import gathered_launch_compatible
from .gathered_softmax import emit_gathered_softmax_preamble

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST


def emit_split_weighted_merge(cg: GenerateAST) -> bool:
    ir = cg.host_function.device_ir
    root = cg.current_root_graph_info
    grid = cg.current_grid_state
    df = cg.device_function
    if (
        root is None
        or len(ir.root_ids) != 2
        or root.graph_id != ir.root_ids[-1]
        or len(ir.grid_block_ids[-1]) != 2
        or not isinstance(grid, DeviceGridState)
        or not df.config.get("cute_fragment_half_mma")
        or not gathered_launch_compatible(df.config.config)
    ):
        return False
    env = CompileEnvironment.current()
    row, col = ir.grid_block_ids[-1]
    spec = env.config_spec
    sizes = spec.block_sizes
    if (
        sizes.config_get(df.config.block_sizes, row) != 1
        or sizes.config_get(df.config.block_sizes, col) != 128
    ):
        return False
    if any(
        lane != (128 if item.block_id == col else 1)
        for item, lane in zip(spec.num_threads, df.config.num_threads, strict=True)
    ):
        return False
    proof = prove_split_weighted_merge(root.graph, cg.codegen_graphs, env, row, col)
    if proof is None:
        return False
    out_tensor = cast("torch.Tensor", proof.output.meta["val"])
    for axis, extent in zip((row, col), out_tensor.shape, strict=True):
        dim = grid.block_id_to_info[axis]
        if dim.begin_expr != 0 or dim.end_expr != extent:
            return False
    # Keeping the split width at one preserves the source's ordered FP32
    # denominator and numerator reductions, including the final division.
    if sizes.config_get(df.config.block_sizes, proof.split_axis, 1) != 1:
        return False
    # Only the already certified scalar prefix is projected. Let the ordinary
    # lowering interpret its original signed arithmetic, indexing and divisors.
    prefix = torch.fx.Graph()
    copies = {}
    nodes = input_closure(proof.live)
    for n in root.graph.nodes:
        if n in nodes:
            copies[n] = prefix.node_copy(n, copies.__getitem__)
    prefix.output([copies[proof.live]])
    (live,) = codegen_call_with_graph(cg, prefix, [])
    assert isinstance(live, ast.AST)
    names = [
        df.tensor_arg(
            cast("torch.Tensor", n.meta["val"]), prefer_name=cast("str", n.args[0])
        ).name
        for n in (proof.scores, proof.values, proof.output)
    ]
    emit_gathered_softmax_preamble(cg)
    math_option = ", fast_math=True" if env.settings.fast_math else ""
    cg.add_statement(
        statement_from_string(
            f"_helion_gathered_softmax.split_weighted_merge({', '.join(names)}, "
            f"{grid.strategy.grid_origin_var(row)}, {grid.strategy.grid_origin_var(col)}, {{live}}{math_option})",
            live=live,
        )
    )
    return True
