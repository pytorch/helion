"""CuTe-backend codegen for ops defined in ``helion.language.view_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``view_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language import memory_ops
from ...language._tracing_ops import _host_tensor
from ...language.atomic_ops import ATOMIC_OPS
from ...language.view_ops import join
from ...language.view_ops import split
from ...language.view_ops import subscript
from ..ast_extension import expr_from_string
from ..ast_read_writes import HELION_LANE_ORDERED_ATTR
from ..compile_environment import CompileEnvironment
from ..loop_dependency_checker import canonical_host_tensor_name
from ..loop_dependency_checker import collect_host_tensor_aliases

if TYPE_CHECKING:
    import ast

    from ..aten_lowering import LoweringContext
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState

# Casts keep each thread's element in place; ``hl.split`` looks through them to
# re-read the loaded tile.
_CAST_TARGETS = frozenset(
    {
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    }
)

# Static ``alloc_smem`` without an opt-in carve-out; a larger exchange buffer
# fails in the NVVM backend rather than at this check.
_SPLIT_SMEM_BUDGET_BYTES = 48 * 1024


@_decorators.codegen(subscript, "cute")
def _(state: CodegenState) -> ast.AST:
    # CuTe kernels currently execute scalarized pointwise code, so shape-only
    # indexing used for broadcast setup is a no-op.
    return state.ast_arg(0)


def _host_tensor_name(node: torch.fx.Node) -> str:
    name = node.args[0]
    assert isinstance(name, str)
    return name


def _host_tensor_root_name(debug_name: str) -> str:
    """Leading identifier of a ``_host_tensor`` debug name (``args['x']`` -> ``args``)."""
    match = re.match(r"[A-Za-z_]\w*", debug_name)
    return debug_name if match is None else match.group(0)


def _tensor_written_anywhere(cg: GenerateAST, load_node: torch.fx.Node) -> bool:
    """True unless no store or atomic in the kernel can target the loaded tensor.

    The re-read for ``hl.split`` happens after the load, so any write to the
    same storage (in this graph or a nested loop graph, before or after the
    load) could change what it observes; be conservative and refuse the fold
    whenever the tensor may be written at all.  Host tensor names are resolved
    through the host-side view aliases (``y = x.view(...)``) before they are
    compared, and two distinct kernel arguments are never provably distinct
    storage (the caller may pass one tensor twice), so a loaded argument is
    treated as written when any argument is.  A loaded value that is not a
    host tensor is treated as written.
    """
    tensor = load_node.args[0]
    if not (isinstance(tensor, torch.fx.Node) and tensor.target is _host_tensor):
        return True
    host_fn = cg.host_function
    aliases = collect_host_tensor_aliases(host_fn.body)
    arg_names = set(host_fn.params.arguments.keys())

    def is_argument(name: str) -> bool:
        return _host_tensor_root_name(name) in arg_names

    name = canonical_host_tensor_name(_host_tensor_name(tensor), aliases)
    loaded_argument = is_argument(name)
    for graph_info in cg.codegen_graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or not node.args:
                continue
            if node.target is not memory_ops.store and node.target not in ATOMIC_OPS:
                continue
            target = node.args[0]
            if not (
                isinstance(target, torch.fx.Node) and target.target is _host_tensor
            ):
                continue
            written = canonical_host_tensor_name(_host_tensor_name(target), aliases)
            if written == name or (loaded_argument and is_argument(written)):
                return True
    return False


def _split_leaf_expr(
    ctx: LoweringContext,
    node: torch.fx.Node,
    flat_index: str,
) -> ast.AST | None:
    """Resolve a shape-chain leaf at ``flat_index`` by re-reading its load."""
    from .memory_ops import cute_reindexed_scalar_load_expr

    if node.op != "call_function":
        return None
    if node.target in _CAST_TARGETS:
        source = node.args[0]
        value = node.meta.get("val")
        if not isinstance(source, torch.fx.Node) or not isinstance(value, torch.Tensor):
            return None
        inner = _split_leaf_expr(ctx, source, flat_index)
        if inner is None:
            return None
        return CompileEnvironment.current().backend.cast_ast(inner, value.dtype)
    if node.target is memory_ops.load:
        from ..generate_ast import GenerateAST

        assert isinstance(ctx.cg, GenerateAST)
        if _tensor_written_anywhere(ctx.cg, node):
            return None
        return cute_reindexed_scalar_load_expr(ctx.cg, node, flat_index)
    return None


def _fold_split_halves(
    state: CodegenState,
    input_node: torch.fx.Node,
    input_shape: list[int],
    output_coords: list[str],
) -> list[ast.AST] | None:
    """Resolve both pair elements through the input's shape chain.

    Output coordinate ``c`` of ``hl.split`` is input coordinate ``(c, minor)``;
    folding the chain down to a load re-reads that element directly, matching
    ``tl.split`` semantics with no shared memory or barrier.
    """
    from .cute_reshape import _flat_index_from_coords
    from .cute_reshape import resolve_cute_shape_chain_value_at

    halves: list[ast.AST] = []
    for minor in (0, 1):
        flat_index = _flat_index_from_coords(
            [*output_coords, f"cutlass.Int32({minor})"], input_shape
        )
        half = resolve_cute_shape_chain_value_at(
            state, input_node, flat_index, leaf_resolver=_split_leaf_expr
        )
        if half is None:
            return None
        halves.append(half)
    return halves


def _check_split_smem_exchange(
    cg: GenerateAST,
    nbytes: int,
    numel: int,
    write_index: str,
    read_indices: list[str],
) -> bool:
    """Reject shared-memory exchanges that cannot be lowered correctly.

    The write, the barrier and the reads all run inside the current lane
    iteration, so a pair element that another lane iteration produces is not
    in shared memory yet.  ``verify_split_smem_exchange`` proves per lane
    iteration that every slot read was written in that same iteration (the
    blocked ``tid * EPT + lane`` layout with halves pairs, the synthetic
    ``tid + lane * T`` layout with interleaved pairs, ...) and rejects the
    rest (e.g. halves pairs on a ``:`` dim, where the partner is the next
    synthetic lane iteration).  Returns whether that proof ran: it needs the
    grid's lane loops, so an exchange under another root state is left to
    the barrier pass alone.
    """
    from ..tile_strategy import DeviceGridState
    from .split_exchange import verify_split_smem_exchange

    if nbytes > _SPLIT_SMEM_BUDGET_BYTES:
        raise exc.BackendUnsupported(
            "cute",
            f"hl.split of a non-load tile needs {nbytes} bytes of shared memory "
            f"(limit {_SPLIT_SMEM_BUDGET_BYTES})",
        )
    grid_state = cg.current_grid_state
    if not isinstance(grid_state, DeviceGridState):
        return False
    verify_split_smem_exchange(cg, grid_state, numel, write_index, read_indices)
    return True


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
    output_coords = [
        _get_node_dim_local_coord(cg, input_node, output_val, i, strict=True)
        for i in range(len(output_shape))
    ]

    lo_var = df.new_var("split_lo")
    hi_var = df.new_var("split_hi")
    halves = _fold_split_halves(state, input_node, input_shape, output_coords)
    if halves is not None:
        for var, half in zip((lo_var, hi_var), halves, strict=True):
            cg.add_statement(statement_from_string(f"{var} = {{half}}", half=half))
        return [expr_from_string(lo_var), expr_from_string(hi_var)]

    # Fallback for non-load tiles: stage the whole input in shared memory and
    # read this thread's pair back.
    input_numel = math.prod(input_shape)
    src_coords = [
        _get_node_dim_local_coord(cg, input_node, input_val, i, strict=True)
        for i in range(len(input_shape))
    ]
    src_flat = _flat_index_from_coords(src_coords, input_shape)
    if output_shape:
        out_flat_base = _flat_index_from_coords(output_coords, output_shape)
    else:
        out_flat_base = "cutlass.Int32(0)"
    lo_flat = f"({out_flat_base}) * cutlass.Int32(2)"
    hi_flat = f"({out_flat_base}) * cutlass.Int32(2) + cutlass.Int32(1)"
    verified = _check_split_smem_exchange(
        cg,
        input_numel * input_val.dtype.itemsize,
        input_numel,
        src_flat,
        [lo_flat, hi_flat],
    )
    dtype_str = env.backend.dtype_str(input_val.dtype)
    smem_ptr = df.new_var("split_smem_ptr")
    smem = df.new_var("split_smem")

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
    stage = statement_from_string(
        f"{smem}[{src_flat}] = {{_inp}}", _inp=state.ast_arg(0)
    )
    reads = [
        statement_from_string(f"{lo_var} = {smem}[{lo_flat}]"),
        statement_from_string(f"{hi_var} = {smem}[{hi_flat}]"),
    ]
    if verified:
        # The check proved, over every thread and lane iteration, that each
        # slot read was staged in the same iteration ahead of the barrier;
        # the barrier pass cannot derive that from the index terms and would
        # reject the exchange across the lanes, so it is told the ordering
        # is settled.
        for statement in (stage, *reads):
            setattr(statement, HELION_LANE_ORDERED_ATTR, frozenset({smem}))
    cg.add_statement(stage)
    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    for statement in reads:
        cg.add_statement(statement)

    return [
        expr_from_string(lo_var),
        expr_from_string(hi_var),
    ]


@_decorators.codegen(join, "cute")
def _(state: CodegenState) -> ast.AST:
    from ..generate_ast import GenerateAST
    from .cute_reshape import _get_node_dim_local_coord

    fx_node = state.fx_node
    assert fx_node is not None
    output_val = fx_node.meta["val"]
    assert isinstance(output_val, torch.Tensor)
    assert isinstance(state.codegen, GenerateAST)

    new_dim = output_val.ndim - 1
    # The selector picks data, so an unowned pair dim must fail loudly instead
    # of folding to a constant that silently keeps one operand.
    selector = _get_node_dim_local_coord(
        state.codegen, fx_node, output_val, new_dim, strict=True
    )

    return expr_from_string(
        f"(({{a}}) if ({selector}) == cutlass.Int32(0) else ({{b}}))",
        a=state.ast_arg(0),
        b=state.ast_arg(1),
    )
