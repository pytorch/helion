"""Codegen support for DeviceIR MPPGraph nodes."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

import torch

from ... import exc
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import NodeArgsGraphInfo
from ..inductor_lowering import CodegenState
from ..inductor_lowering import codegen_call_with_graph
from helion.language import _decorators

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST


@dataclasses.dataclass(frozen=True)
class MPPSetupParams:
    """Arguments for ``_metal_mpp_setup(...)`` consumed by the MSL walker.

    Field order is mirrored by ``msl_ast_walker._extract_mpp_setup_params``.
    """

    lhs: str
    rhs: str
    M: int
    N: int
    K: int
    TILE_M: int
    TILE_N: int
    TILE_K: int
    NUM_SG: int
    in_dtype: str
    acc_dtype: str
    bias: str | None
    bias_dtype: str | None
    fx_name: str | None

    def __str__(self) -> str:
        return (
            f'_metal_mpp_setup("{self.lhs}", "{self.rhs}", '
            f"{self.M}, {self.N}, {self.K}, "
            f"{self.TILE_M}, {self.TILE_N}, {self.TILE_K}, {self.NUM_SG}, "
            f'"{self.in_dtype}", "{self.acc_dtype}", '
            f'"{self.bias or ""}", "{self.bias_dtype or ""}", '
            f'"{self.fx_name or ""}")'
        )


@dataclasses.dataclass(frozen=True)
class _MppKStep:
    """``_metal_mpp_k_step(setup_var, k_offset)`` payload."""

    setup_var: str
    k_offset: str

    def __str__(self) -> str:
        return f"_metal_mpp_k_step({self.setup_var}, {self.k_offset})"


@_decorators.api()
def _mpp_graph(
    graph_id: int,
    begin: list[object],
    end: list[object],
    args: list[object],
) -> list[object]:
    """DeviceIR marker for a Metal MPP-owned graph."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_mpp_graph, "common")
def _(state: CodegenState) -> list[object] | None:
    """Dispatch the synthetic marker to its owning MPPGraphInfo."""
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
    """Resolve the root grid axes that correspond to matmul M and N."""
    grid_state = codegen.current_grid_state
    if grid_state is None:
        return None

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
    needs_store_barrier: bool = False

    @property
    def name(self) -> str:
        """Return the synthetic graph name used in generated comments."""
        return f"mpp_graph_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        """Serialize fields needed when cloning this graph info."""
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
            "needs_store_barrier": self.needs_store_barrier,
        }

    def codegen(self, state: CodegenState) -> list[object]:
        """Emit the MPP region outside scalar lane loops when needed."""
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and grid_state.has_lane_loops():
            # GenerateAST wraps scalar root statements in per-lane loops for
            # oversized tile axes.  MPP is already cooperative across the whole
            # threadgroup, so it must run once in the root prefix.
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
        """Emit setup, K-loop, optional epilogue, and cooperative store."""
        args = state.ast_args[3]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)

        codegen = state.codegen

        # 1. Declare the MPP tensors/cooperative accumulator and run the
        # reduction loop over the original K tile range.
        setup_var = self._emit_setup_and_k_loop(state)

        # 2. Lower the optional epilogue graph with the cooperative tensor as
        # its input.  codegen_call_with_graph interprets the copied FX graph and
        # emits normal Helion AST statements into epilogue_body, returning the
        # graph output expression.
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
        if not (isinstance(output, ast.Name) and output.id == self.result_name):
            # A transformed output must be written back into the cooperative
            # tensor element before the final MPP store.
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
            # 3. Apply the epilogue once per cooperative tensor element.
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

        # 4. Store the cooperative tensor to HBM.
        out_name = codegen.device_function.tensor_arg(self.out_tensor).name
        codegen.device_function.placeholder_args.add(out_name)
        out_dtype = MetalBackend._get_dtype_to_metal()[self.out_dtype]
        codegen.add_statement(
            statement_from_string(
                f'_metal_mpp_coop_store({setup_var}, "{out_name}", "{out_dtype}")'
            )
        )
        if emit_store_barrier and self.needs_store_barrier:
            # Inline MPP emission needs to emit its own barrier.  When MPP is
            # hoisted to the root prefix, codegen() appends the barrier after
            # extending that prefix so scalar reloads see the cooperative store.
            codegen.add_statement(_mpp_threadgroup_barrier_stmt())
        return []

    def _emit_setup_and_k_loop(self, state: CodegenState) -> str:
        """Emit MPP setup plus the K-loop and return the setup variable name."""
        assert self.lhs_tensor is not None
        assert self.rhs_tensor is not None

        codegen = state.codegen
        df = codegen.device_function
        env = CompileEnvironment.current()

        from ..backend import MetalBackend

        dtype_map = MetalBackend._get_dtype_to_metal()
        if self.lhs_tensor.dtype not in dtype_map:
            raise exc.BackendUnsupported(
                "metal",
                f"unsupported input dtype for MPP matmul: {self.lhs_tensor.dtype}",
            )
        acc_dtype = self.acc_dtype
        if acc_dtype not in dtype_map:
            raise exc.BackendUnsupported(
                "metal",
                f"unsupported accumulator dtype for MPP matmul: {acc_dtype}",
            )
        assert acc_dtype is not None
        if self.out_dtype is not None and acc_dtype != self.out_dtype:
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
            raise exc.BackendUnsupported("metal", "unable to resolve MPP M/N axes")
        m_axis, n_axis = axes
        if m_axis.block_size is None or n_axis.block_size is None:
            raise exc.BackendUnsupported("metal", "dynamic MPP M/N block size")

        bk = env.block_sizes[self.k_block_id].from_config(df.config)
        if not isinstance(bk, int):
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

        setup = MPPSetupParams(
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
            bias=acc_arg_name or None,
            bias_dtype=acc_metal_dtype or None,
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


def _bound_expr(bounds: object, index: int) -> ast.expr:
    """Return one loop bound as a Python AST expression."""
    assert isinstance(bounds, list)
    bound = bounds[index]
    if isinstance(bound, ast.expr):
        return bound
    assert isinstance(bound, int)
    return create(ast.Constant, value=bound)


def _mpp_threadgroup_barrier_stmt() -> ast.stmt:
    """Return the pseudo-call statement for an MPP/scalar memory barrier."""
    return statement_from_string("_metal_mpp_threadgroup_barrier()")
