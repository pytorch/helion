"""DeviceIR rewrite for Metal MPP matmul lowering.

This module recognizes a narrow, ownership-preserving matmul pattern in the
DeviceIR produced after Helion's normal graph construction and loop planning.
The goal is to replace the reduction loop that computes a single output tile
with an ``MPPGraphInfo`` region while leaving the surrounding root graph in the
normal scalar Metal lowering path.

The expected shape is:

* the root graph calls one K-loop via ``_for_loop``;
* the selected K-loop result is read by one ``operator.getitem``;
* that value is threaded through one ``_phi`` node in the root graph;
* the K-loop returns a supported matmul node (``aten.mm``, ``aten.addmm``, or
  ``hl.dot``), meaning post-K epilogues live outside the reduction loop;
* the post-K value flows through either a cooperative-fusible epilogue chain or
  through scalar work that can safely reload the materialized output tile;
* the final value is stored to a canonical 2D output tile.

When the cooperative path matches, the transform creates a new ``MPPGraphInfo``
containing the K-loop bounds, matmul operands, optional bias, optional fused
epilogue graph, and cooperative store metadata.  The original root graph is
rewritten to call the synthetic ``_mpp_graph`` marker.  During MSL generation,
``MPPGraphInfo`` emits the MPP setup, the internal K loop, optional cooperative
epilogue iteration, and the cooperative store.

If the root graph contains scalar postprocessing after the matmul that is not
cooperative-fusible but still has the same output tile shape and dtype, the MPP
region first materializes the matmul result to the final output tensor.  The
root graph then reloads that tile and continues with scalar Metal code.  A
device-memory threadgroup barrier is recorded when scalar code may observe the
cooperative store in the same kernel.

The transform is intentionally conservative.  Failed recognition is treated as
"leave the DeviceIR alone"; the later Metal lowering path is responsible for
raising ``BackendUnsupported`` if it still sees an unsupported matmul.  This
keeps partial rewrites out of the graph and avoids silently changing semantics
for cases such as batched matmul, non-canonical operand views, or epilogues
inside the K-loop that alter every partial accumulation step.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import Graph
from torch.fx.node import Node
from torch.fx.node import map_arg

from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..cute.cute_mma import _trace_to_load_tensor
from ..device_ir import DeviceIR
from ..device_ir import ForLoopGraphInfo
from ..device_ir import NodeArgsGraphInfo
from ..device_ir import RootGraphInfo
from ..inductor_lowering import APIFuncLowering
from ..inductor_lowering import CodegenState
from ..inductor_lowering import codegen_call_with_graph
from .metal_mma import _MppKStep
from .metal_mma import _MppSetup
from helion.language import _decorators
from helion.language import _tracing_ops
from helion.language import memory_ops

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST


@_decorators.api()
def _mpp_graph(
    graph_id: int,
    begin: list[object],
    end: list[object],
    args: list[object],
) -> list[object]:
    """DeviceIR marker for a Metal MPP-owned graph.

    This marker is introduced by :func:`rewrite_mpp_graphs` and lowered by
    :class:`MPPGraphInfo`.
    """
    raise AssertionError("this should never be called")


@_decorators.codegen(_mpp_graph, "common")
def _(state: CodegenState) -> list[object] | None:
    graph_info = state.get_graph(state.proxy_arg(0))
    return graph_info.codegen(state)


@dataclasses.dataclass(frozen=True)
class _GridAxisInfo:
    block_id: int
    offset_var: str
    block_size: int | None


def _resolve_grid_mn_axes(
    codegen: GenerateAST,
    env: CompileEnvironment,
    m_size: int | torch.SymInt,
    n_size: int | torch.SymInt,
) -> tuple[_GridAxisInfo, _GridAxisInfo] | None:
    grid_state = codegen.current_grid_state
    if grid_state is None:
        return None
    if len(grid_state.block_ids) == 2:
        m_block_id, n_block_id = grid_state.block_ids
        m_bs = codegen.device_function.resolved_block_size(m_block_id)
        n_bs = codegen.device_function.resolved_block_size(n_block_id)
        return (
            _GridAxisInfo(
                block_id=m_block_id,
                offset_var=grid_state.strategy.offset_var(m_block_id),
                block_size=int(m_bs) if isinstance(m_bs, int) else None,
            ),
            _GridAxisInfo(
                block_id=n_block_id,
                offset_var=grid_state.strategy.offset_var(n_block_id),
                block_size=int(n_bs) if isinstance(n_bs, int) else None,
            ),
        )

    m_axis: _GridAxisInfo | None = None
    n_axis: _GridAxisInfo | None = None
    for block_id in grid_state.block_ids:
        block_size_info = env.block_sizes[block_id]
        size = block_size_info.size
        if not isinstance(size, (int, torch.SymInt)):
            continue
        block_size = block_size_info.from_config(codegen.device_function.config)
        axis = _GridAxisInfo(
            block_id=block_id,
            offset_var=grid_state.strategy.offset_var(block_id),
            block_size=int(block_size) if isinstance(block_size, int) else None,
        )
        if m_axis is None and env.known_equal(size, m_size):
            m_axis = axis
        elif n_axis is None and env.known_equal(size, n_size):
            n_axis = axis
    if m_axis is None or n_axis is None:
        return None
    return m_axis, n_axis


@dataclasses.dataclass
class MPPGraphInfo(NodeArgsGraphInfo):
    """GraphInfo for a Metal MPP matmul region."""

    k_block_id: int
    begin: list[object]
    end: list[object]
    result_name: str
    lhs_tensor: torch.Tensor | None = None
    rhs_tensor: torch.Tensor | None = None
    bias_tensor: torch.Tensor | None = None
    acc_dtype: torch.dtype | None = None
    out_tensor: torch.Tensor | None = None
    out_dtype: torch.dtype | None = None
    epilogue_node_names: tuple[str, ...] = ()
    store_node_name: str | None = None
    needs_store_barrier: bool = False

    @property
    def name(self) -> str:
        return f"mpp_graph_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "k_block_id": self.k_block_id,
            "begin": self.begin,
            "end": self.end,
            "result_name": self.result_name,
            "lhs_tensor": self.lhs_tensor,
            "rhs_tensor": self.rhs_tensor,
            "bias_tensor": self.bias_tensor,
            "acc_dtype": self.acc_dtype,
            "out_tensor": self.out_tensor,
            "out_dtype": self.out_dtype,
            "epilogue_node_names": self.epilogue_node_names,
            "store_node_name": self.store_node_name,
            "needs_store_barrier": self.needs_store_barrier,
        }

    def codegen(self, state: CodegenState) -> list[object]:
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and grid_state.has_lane_loops():
            mpp_body: list[ast.AST] = []
            with state.codegen.set_statements(mpp_body):
                self._codegen(state, emit_store_barrier=False)
            grid_state.outer_prefix.extend(mpp_body)
            if self.needs_store_barrier:
                grid_state.outer_prefix.append(_mpp_threadgroup_barrier_stmt())
            return []
        return self._codegen(state, emit_store_barrier=True)

    def _codegen(
        self, state: CodegenState, *, emit_store_barrier: bool
    ) -> list[object]:
        args = state.ast_args[3]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)

        codegen = state.codegen
        setup_var = self._emit_setup_and_k_loop(state)
        epilogue_body: list[ast.AST] = []
        with codegen.set_statements(epilogue_body):
            outputs = codegen_call_with_graph(
                codegen,
                self.graph,
                [expr_from_string(self.result_name)],
                copy_named_args=False,
            )
        assert len(outputs) == 1
        output = outputs[0]
        assert isinstance(output, ast.expr)
        if not (
            isinstance(output, ast.Name)
            and output.id == self.result_name
            and not epilogue_body
        ):
            epilogue_body.append(
                create(
                    ast.Expr,
                    value=create(
                        ast.Call,
                        func=create(ast.Name, id="_coop_writeback", ctx=ast.Load()),
                        args=[output],
                        keywords=[],
                    ),
                )
            )
        if epilogue_body:
            codegen.add_statement(
                create(
                    ast.For,
                    target=create(ast.Name, id="_it", ctx=ast.Store()),
                    iter=create(
                        ast.Call,
                        func=create(ast.Name, id="_coop_iter", ctx=ast.Load()),
                        args=[expr_from_string(setup_var)],
                        keywords=[],
                    ),
                    body=epilogue_body,
                    orelse=[],
                )
            )

        assert self.out_tensor is not None
        assert self.out_dtype is not None
        from ..backend import MetalBackend

        out_name = codegen.device_function.tensor_arg(self.out_tensor).name
        codegen.device_function.placeholder_args.add(out_name)
        out_dtype = MetalBackend._get_dtype_to_metal()[self.out_dtype]
        codegen.add_statement(
            statement_from_string(
                f'_metal_mpp_coop_store({setup_var}, "{out_name}", "{out_dtype}")'
            )
        )
        if emit_store_barrier and self.needs_store_barrier:
            codegen.add_statement(_mpp_threadgroup_barrier_stmt())
        return []

    def _emit_setup_and_k_loop(self, state: CodegenState) -> str:
        assert self.lhs_tensor is not None
        assert self.rhs_tensor is not None

        codegen = state.codegen
        df = codegen.device_function
        env = CompileEnvironment.current()

        from ..backend import MetalBackend

        dtype_map = MetalBackend._get_dtype_to_metal()
        if self.lhs_tensor.dtype not in dtype_map:
            from ... import exc

            raise exc.BackendUnsupported(
                "metal",
                f"unsupported input dtype for MPP matmul: {self.lhs_tensor.dtype}",
            )
        acc_dtype = self.acc_dtype
        if acc_dtype not in dtype_map:
            from ... import exc

            raise exc.BackendUnsupported(
                "metal",
                f"unsupported accumulator dtype for MPP matmul: {acc_dtype}",
            )
        assert acc_dtype is not None
        if self.out_dtype is not None and acc_dtype != self.out_dtype:
            from ... import exc

            raise exc.BackendUnsupported(
                "metal",
                "MPP cooperative store requires accumulator dtype to match output "
                f"dtype; got accumulator {acc_dtype} and output {self.out_dtype}",
            )
        assert self.lhs_tensor.ndim == 2 and self.rhs_tensor.ndim == 2
        assert self.lhs_tensor.dtype == self.rhs_tensor.dtype

        axes = _resolve_grid_mn_axes(
            codegen,
            env,
            self.lhs_tensor.shape[0],
            self.rhs_tensor.shape[1],
        )
        if axes is None:
            from ... import exc

            raise exc.BackendUnsupported("metal", "unable to resolve MPP M/N axes")
        m_axis, n_axis = axes
        if m_axis.block_size is None or n_axis.block_size is None:
            from ... import exc

            raise exc.BackendUnsupported("metal", "dynamic MPP M/N block size")

        bk = env.block_sizes[self.k_block_id].from_config(df.config)
        if not isinstance(bk, int):
            from ... import exc

            raise exc.BackendUnsupported("metal", "dynamic MPP K block size")

        num_sg = df.config.num_warps if df.config.num_warps is not None else 4
        codegen.max_thread_block_dims[0] = max(
            codegen.max_thread_block_dims[0], num_sg * 32
        )

        lhs_arg_name = df.tensor_arg(self.lhs_tensor).name
        rhs_arg_name = df.tensor_arg(self.rhs_tensor).name
        df.placeholder_args.add(lhs_arg_name)
        df.placeholder_args.add(rhs_arg_name)

        acc_arg_name = ""
        acc_metal_dtype = ""
        if self.bias_tensor is not None:
            acc_arg_name = df.tensor_arg(self.bias_tensor).name
            df.placeholder_args.add(acc_arg_name)
            acc_metal_dtype = dtype_map.get(self.bias_tensor.dtype, "")

        setup = _MppSetup(
            lhs=lhs_arg_name,
            rhs=rhs_arg_name,
            M=int(env.size_hint(self.lhs_tensor.shape[0])),
            N=int(env.size_hint(self.rhs_tensor.shape[1])),
            K=int(env.size_hint(self.lhs_tensor.shape[1])),
            TILE_M=m_axis.block_size,
            TILE_N=n_axis.block_size,
            TILE_K=bk,
            NUM_SG=num_sg,
            in_dtype=dtype_map[self.lhs_tensor.dtype],
            acc_dtype=dtype_map[acc_dtype],
            bias=acc_arg_name,
            bias_dtype=acc_metal_dtype,
            fx_name=self.result_name,
        )
        setup_var = df.new_var("_mpp_setup")
        codegen.add_statement(statement_from_string(f"{setup_var} = {setup}"))
        k_offset_var = df.new_var(f"offset_{self.k_block_id}")
        body = [
            statement_from_string(
                str(_MppKStep(setup_var=setup_var, k_offset=k_offset_var))
            )
        ]
        loop = create(
            ast.For,
            target=create(ast.Name, id=k_offset_var, ctx=ast.Store()),
            iter=create(
                ast.Call,
                func=create(
                    ast.Attribute,
                    value=create(ast.Name, id="tl", ctx=ast.Load()),
                    attr="range",
                    ctx=ast.Load(),
                ),
                args=[
                    _bound_expr(self.begin, 0),
                    _bound_expr(self.end, 0),
                    create(ast.Constant, value=bk),
                ],
                keywords=[],
            ),
            body=body,
            orelse=[],
        )
        codegen.add_statement(loop)
        return setup_var


@dataclasses.dataclass(frozen=True)
class MPPGraphRewrite:
    root_graph_id: int
    mpp_graph_id: int
    result_name: str
    epilogue_node_names: tuple[str, ...]
    store_node_name: str | None


@dataclasses.dataclass(frozen=True)
class _MPPLoadView:
    tensor: torch.Tensor
    indices: tuple[object, object]


@dataclasses.dataclass(frozen=True)
class _MPPStoreView:
    tensor: torch.Tensor
    dtype: torch.dtype
    indices: tuple[object, object]


@dataclasses.dataclass(frozen=True)
class _Candidate:
    for_loop_node: Node
    getitem_node: Node
    phi_node: Node
    k_loop_info: ForLoopGraphInfo
    mma_node: Node
    lhs_view: _MPPLoadView
    rhs_view: _MPPLoadView
    bias_view: _MPPLoadView | None
    acc_dtype: torch.dtype
    store_view: _MPPStoreView
    mpp_output_node: Node
    reload_value_node: Node | None
    epilogue_nodes: tuple[Node, ...]
    store_node: Node


def rewrite_mpp_graphs(device_ir: DeviceIR) -> list[MPPGraphRewrite]:
    """Rewrite eligible root/K-loop matmul pairs into ``MPPGraphInfo``.

    The pass scans only root graphs.  For each root ``_for_loop`` call, it
    checks whether one returned value is a completed K reduction whose loop
    body returns a supported matmul accumulator.  The root use must be
    single-owner: ``getitem -> _phi -> optional epilogue -> store``.  This
    ownership check is what lets the rewrite remove the original K-loop result
    from the root graph and replace it with a synthetic ``_mpp_graph`` call
    without duplicating stores or losing side effects.

    The produced ``MPPGraphInfo`` owns code generation for the cooperative
    region.  The root graph remains responsible for any scalar continuation
    that the transform deliberately leaves outside the cooperative region.
    Unsupported or ambiguous shapes return no candidate and leave the graph
    unchanged.
    """
    rewrites: list[MPPGraphRewrite] = []
    for graph_info in list(device_ir.graphs):
        if not isinstance(graph_info, RootGraphInfo):
            continue
        for candidate in _find_root_candidates(device_ir, graph_info):
            mpp_graph_id = _append_mpp_graph(device_ir, candidate)
            _rewrite_root_for_mpp_graph(graph_info, candidate, mpp_graph_id)
            rewrites.append(
                MPPGraphRewrite(
                    root_graph_id=graph_info.graph_id,
                    mpp_graph_id=mpp_graph_id,
                    result_name=candidate.mma_node.name,
                    epilogue_node_names=tuple(n.name for n in candidate.epilogue_nodes),
                    store_node_name=candidate.store_node.name,
                )
            )
    return rewrites


def _find_root_candidates(
    device_ir: DeviceIR, root_info: RootGraphInfo
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for node in root_info.graph.nodes:
        if not _is_for_loop_call(node):
            continue
        k_loop_info = _for_loop_graph_info(device_ir, node)
        if k_loop_info is None:
            continue
        for getitem_node in _single_user_getitems(node):
            mma_node = _mma_output_for_getitem(k_loop_info, getitem_node)
            if mma_node is None:
                continue
            phi_node = _single_phi_user(getitem_node)
            if phi_node is None:
                continue
            candidate = _classify_phi_users(
                for_loop_node=node,
                getitem_node=getitem_node,
                phi_node=phi_node,
                k_loop_info=k_loop_info,
                mma_node=mma_node,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _is_for_loop_call(node: Node) -> bool:
    return (
        node.op == "call_function"
        and _tracing_ops.is_for_loop_target(node.target)
        and bool(node.args)
        and isinstance(node.args[0], int)
    )


def _for_loop_graph_info(device_ir: DeviceIR, node: Node) -> ForLoopGraphInfo | None:
    graph_id = node.args[0]
    assert isinstance(graph_id, int)
    if not (0 <= graph_id < len(device_ir.graphs)):
        return None
    graph_info = device_ir.graphs[graph_id]
    if not isinstance(graph_info, ForLoopGraphInfo):
        return None
    return graph_info


def _single_user_getitems(for_loop_node: Node) -> list[Node]:
    return [
        user
        for user in for_loop_node.users
        if user.op == "call_function" and user.target is operator.getitem
    ]


def _mma_output_for_getitem(
    k_loop_info: ForLoopGraphInfo, getitem_node: Node
) -> Node | None:
    if len(getitem_node.args) < 2 or not isinstance(getitem_node.args[1], int):
        return None
    output_idx = getitem_node.args[1]
    output_nodes = list(k_loop_info.graph.find_nodes(op="output"))
    if len(output_nodes) != 1:
        return None
    output_value = output_nodes[0].args[0]
    if not isinstance(output_value, (tuple, list)):
        return None
    if not (0 <= output_idx < len(output_value)):
        return None
    candidate = output_value[output_idx]
    if not isinstance(candidate, Node) or not _is_mpp_matmul_node(candidate):
        return None
    return candidate


def _is_mpp_matmul_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    }


def _single_phi_user(getitem_node: Node) -> Node | None:
    phi_users = [
        user
        for user in getitem_node.users
        if user.op == "call_function"
        and user.target is _tracing_ops._phi
        and len(user.args) >= 2
        and user.args[1] is getitem_node
    ]
    if len(phi_users) != 1:
        return None
    return phi_users[0]


def _classify_phi_users(
    *,
    for_loop_node: Node,
    getitem_node: Node,
    phi_node: Node,
    k_loop_info: ForLoopGraphInfo,
    mma_node: Node,
) -> _Candidate | None:
    operand_info = _mpp_operand_views(mma_node)
    if operand_info is None:
        return None
    lhs_view, rhs_view, bias_view, acc_dtype = operand_info
    epilogue_nodes: list[Node] = []
    cur = phi_node
    visited: set[Node] = set()
    while True:
        if cur in visited or len(cur.users) != 1:
            return None
        visited.add(cur)
        (user,) = cur.users
        if _is_store_of_value(user, cur):
            store_view = _store_output_view(user)
            if store_view is None or not _can_store_mpp_output(
                lhs_view, rhs_view, cur, store_view, acc_dtype
            ):
                return None
            return _Candidate(
                for_loop_node=for_loop_node,
                getitem_node=getitem_node,
                phi_node=phi_node,
                k_loop_info=k_loop_info,
                mma_node=mma_node,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                bias_view=bias_view,
                acc_dtype=acc_dtype,
                store_view=store_view,
                mpp_output_node=cur,
                reload_value_node=None,
                epilogue_nodes=tuple(epilogue_nodes),
                store_node=user,
            )
        if not _is_coop_fusible_epilogue_node(user, cur):
            materialized = _find_materialized_scalar_store(
                start_node=user,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                mpp_output_node=cur,
                acc_dtype=acc_dtype,
            )
            if materialized is None:
                return None
            store_node, store_view = materialized
            return _Candidate(
                for_loop_node=for_loop_node,
                getitem_node=getitem_node,
                phi_node=phi_node,
                k_loop_info=k_loop_info,
                mma_node=mma_node,
                lhs_view=lhs_view,
                rhs_view=rhs_view,
                bias_view=bias_view,
                acc_dtype=acc_dtype,
                store_view=store_view,
                mpp_output_node=cur,
                reload_value_node=cur,
                epilogue_nodes=tuple(epilogue_nodes),
                store_node=store_node,
            )
        epilogue_nodes.append(user)
        cur = user


def _is_store_of_value(node: Node, value: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is memory_ops.store
        and len(node.args) >= 3
        and node.args[2] is value
    )


def _find_materialized_scalar_store(
    *,
    start_node: Node,
    lhs_view: _MPPLoadView,
    rhs_view: _MPPLoadView,
    mpp_output_node: Node,
    acc_dtype: torch.dtype,
) -> tuple[Node, _MPPStoreView] | None:
    cur = start_node
    visited: set[Node] = set()
    while True:
        if cur in visited:
            return None
        visited.add(cur)
        if _is_store_of_value(cur, cur):
            return None
        if (
            cur.op == "call_function"
            and cur.target is memory_ops.store
            and len(cur.args) >= 3
        ):
            store_view = _store_output_view(cur)
            if store_view is None or not _can_store_mpp_output(
                lhs_view, rhs_view, mpp_output_node, store_view, acc_dtype
            ):
                return None
            return cur, store_view
        if len(cur.users) != 1:
            return None
        (cur,) = cur.users


def _mpp_operand_views(
    mma_node: Node,
) -> tuple[_MPPLoadView, _MPPLoadView, _MPPLoadView | None, torch.dtype] | None:
    lhs_idx = 1 if mma_node.target is torch.ops.aten.addmm.default else 0
    rhs_idx = 2 if mma_node.target is torch.ops.aten.addmm.default else 1
    acc_idx = 0 if mma_node.target is torch.ops.aten.addmm.default else None
    if len(mma_node.args) <= max(lhs_idx, rhs_idx):
        return None
    lhs_node = mma_node.args[lhs_idx]
    rhs_node = mma_node.args[rhs_idx]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return None
    lhs_view = _trace_to_load_view(lhs_node)
    rhs_view = _trace_to_load_view(rhs_node)
    if lhs_view is None or rhs_view is None:
        return None
    if lhs_view.tensor.ndim != 2 or rhs_view.tensor.ndim != 2:
        return None
    if lhs_view.tensor.dtype != rhs_view.tensor.dtype:
        return None
    acc_dtype = _tensor_dtype_from_meta(mma_node)
    if acc_dtype is None:
        return None
    bias_view = None
    if acc_idx is not None and len(mma_node.args) > acc_idx:
        acc_node = mma_node.args[acc_idx]
        if isinstance(acc_node, Node):
            bias_view = _trace_to_load_view(acc_node)
    return lhs_view, rhs_view, bias_view, acc_dtype


def _trace_to_load_view(node: Node) -> _MPPLoadView | None:
    load_info = _trace_to_load_tensor(node)
    if load_info is None:
        return None
    load_node, _, tensor = load_info
    if len(load_node.args) < 2:
        return None
    indices = _as_2d_indices(load_node.args[1])
    if indices is None:
        return None
    return _MPPLoadView(tensor=tensor, indices=indices)


def _tensor_dtype_from_meta(node: Node) -> torch.dtype | None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return value.dtype
    return None


def _as_2d_indices(indices: object) -> tuple[object, object] | None:
    if not isinstance(indices, (list, tuple)) or len(indices) != 2:
        return None
    return indices[0], indices[1]


def _is_same_index(lhs: object, rhs: object) -> bool:
    if lhs is rhs:
        return True
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    return False


def _is_canonical_mpp_view(
    lhs_view: _MPPLoadView,
    rhs_view: _MPPLoadView,
    store_view: _MPPStoreView,
) -> bool:
    if lhs_view.tensor.ndim != 2 or rhs_view.tensor.ndim != 2:
        return False
    if store_view.tensor.ndim != 2:
        return False
    if not (
        _same_dim(lhs_view.tensor.shape[0], store_view.tensor.shape[0])
        and _same_dim(rhs_view.tensor.shape[1], store_view.tensor.shape[1])
        and _same_dim(lhs_view.tensor.shape[1], rhs_view.tensor.shape[0])
    ):
        return False
    lhs_m, lhs_k = lhs_view.indices
    rhs_k, rhs_n = rhs_view.indices
    out_m, out_n = store_view.indices
    if not all(
        isinstance(index, int) for index in (lhs_m, lhs_k, rhs_k, rhs_n, out_m, out_n)
    ):
        return True
    return (
        _is_same_index(lhs_m, out_m)
        and _is_same_index(rhs_n, out_n)
        and _is_same_index(lhs_k, rhs_k)
    )


def _can_store_mpp_output(
    lhs_view: _MPPLoadView,
    rhs_view: _MPPLoadView,
    mpp_output_node: Node,
    store_view: _MPPStoreView,
    acc_dtype: torch.dtype,
) -> bool:
    value = mpp_output_node.meta.get("val")
    if isinstance(value, torch.Tensor) and (
        value.dtype != store_view.dtype or value.ndim != store_view.tensor.ndim
    ):
        return False
    if not isinstance(value, torch.Tensor) and acc_dtype != store_view.dtype:
        return False
    return _is_canonical_mpp_view(lhs_view, rhs_view, store_view)


def _same_dim(lhs: object, rhs: object) -> bool:
    try:
        return bool(lhs == rhs)
    except TypeError:
        return True


def _is_coop_fusible_epilogue_node(node: Node, input_node: Node) -> bool:
    if node.op != "call_function" or not _is_coop_fusible_epilogue_target(node.target):
        return False
    return all(user_input is input_node for user_input in node.all_input_nodes)


def _is_coop_fusible_epilogue_target(target: object) -> bool:
    return target in {
        torch.ops.aten.relu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.neg.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.prims.convert_element_type.default,
    }


def _append_mpp_graph(device_ir: DeviceIR, candidate: _Candidate) -> int:
    new_graph = Graph()
    acc = new_graph.placeholder(candidate.mma_node.name)
    acc.meta.update(candidate.mma_node.meta)
    env: dict[Node, Node] = {candidate.phi_node: acc}

    for node in candidate.epilogue_nodes:
        _copy_node_recursive(node, new_graph, env)
    new_graph.output((env[candidate.mpp_output_node],))

    begin = candidate.for_loop_node.args[1]
    end = candidate.for_loop_node.args[2]
    assert isinstance(begin, list)
    assert isinstance(end, list)

    graph_id = len(device_ir.graphs)
    device_ir.graphs.append(
        MPPGraphInfo(
            graph_id=graph_id,
            graph=new_graph,
            node_args=[*candidate.k_loop_info.node_args],
            k_block_id=candidate.k_loop_info.block_ids[0],
            begin=[*begin],
            end=[*end],
            result_name=candidate.mma_node.name,
            lhs_tensor=candidate.lhs_view.tensor,
            rhs_tensor=candidate.rhs_view.tensor,
            bias_tensor=(
                candidate.bias_view.tensor if candidate.bias_view is not None else None
            ),
            acc_dtype=candidate.acc_dtype,
            out_tensor=candidate.store_view.tensor,
            out_dtype=candidate.store_view.dtype,
            epilogue_node_names=tuple(n.name for n in candidate.epilogue_nodes),
            store_node_name=(
                candidate.store_node.name if candidate.store_node is not None else None
            ),
            needs_store_barrier=candidate.reload_value_node is not None,
        )
    )
    return graph_id


def _store_output_view(store_node: Node | None) -> _MPPStoreView | None:
    if store_node is None or len(store_node.args) < 2:
        return None
    out_arg = store_node.args[0]
    if not isinstance(out_arg, Node):
        return None
    fake = out_arg.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    indices = _as_2d_indices(store_node.args[1])
    if indices is None:
        return None
    return _MPPStoreView(
        tensor=fake,
        dtype=fake.dtype,
        indices=indices,
    )


def _bound_expr(bounds: object, index: int) -> ast.expr:
    assert isinstance(bounds, list)
    bound = bounds[index]
    if isinstance(bound, ast.expr):
        return bound
    assert isinstance(bound, int)
    return create(ast.Constant, value=bound)


def _mpp_threadgroup_barrier_stmt() -> ast.stmt:
    return statement_from_string("_metal_mpp_threadgroup_barrier()")


def _copy_node_recursive(node: Node, new_graph: Graph, env: dict[Node, Node]) -> Node:
    if node in env:
        return env[node]
    for input_node in node.all_input_nodes:
        if input_node not in env:
            _copy_node_recursive(input_node, new_graph, env)
    new_node = new_graph.node_copy(node, lambda n: env[n])
    env[node] = new_node
    return new_node


def _rewrite_root_for_mpp_graph(
    root_info: RootGraphInfo,
    candidate: _Candidate,
    mpp_graph_id: int,
) -> None:
    consumed = {
        candidate.for_loop_node,
        candidate.getitem_node,
        candidate.phi_node,
        *candidate.epilogue_nodes,
    }
    if candidate.reload_value_node is None:
        consumed.add(candidate.store_node)
    old_graph = root_info.graph
    new_graph = Graph()
    env: dict[Node, Node] = {}
    inserted = False

    def load_arg(n: Node) -> Node:
        return env[n]

    def ensure_arg(arg: Node) -> Node:
        if arg in env:
            return env[arg]
        for input_node in arg.all_input_nodes:
            if input_node not in env:
                ensure_arg(input_node)
        new_node = new_graph.node_copy(arg, load_arg)
        env[arg] = new_node
        return new_node

    for node in old_graph.nodes:
        if node is candidate.for_loop_node:
            mpp_node = new_graph.call_function(
                _mpp_graph,
                args=(
                    mpp_graph_id,
                    node.args[1],
                    node.args[2],
                    map_arg(node.args[3], load_arg),
                ),
            )
            _prepare_inserted_mpp_node(mpp_node)
            env[node] = mpp_node
            if candidate.reload_value_node is not None:
                load_node = new_graph.call_function(
                    memory_ops.load,
                    args=(
                        map_arg(candidate.store_node.args[0], ensure_arg),
                        map_arg(candidate.store_node.args[1], ensure_arg),
                        None,
                        None,
                    ),
                )
                _prepare_inserted_reload_node(
                    load_node,
                    candidate.reload_value_node,
                    fallback_node=candidate.mma_node,
                )
                env[candidate.reload_value_node] = load_node
            inserted = True
            continue
        if node in consumed:
            continue
        if node.op == "output":
            new_graph.output(map_arg(node.args[0], load_arg))
            continue
        if node in env:
            continue
        new_node = new_graph.node_copy(node, load_arg)
        env[node] = new_node

    assert inserted, "expected to insert _mpp_graph marker"
    root_info.graph = new_graph


def _prepare_inserted_mpp_node(node: Node) -> None:
    node.meta["lowering"] = APIFuncLowering(_mpp_graph)
    node.meta["location"] = contextlib.nullcontext()
    node.meta["val"] = []


def _prepare_inserted_reload_node(
    node: Node, value_node: Node, *, fallback_node: Node
) -> None:
    node.meta["lowering"] = APIFuncLowering(memory_ops.load)
    node.meta["location"] = contextlib.nullcontext()
    node.meta["val"] = value_node.meta.get("val", fallback_node.meta["val"])
