from __future__ import annotations

import ast
import builtins
from collections.abc import Iterable
import contextlib
import copy
import dataclasses
import functools
import operator
import re
import textwrap
import threading
from typing import TYPE_CHECKING
from typing import Callable
from typing import Iterator
from typing import NamedTuple
from typing import Protocol
from typing import cast
from unittest.mock import patch

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor.decomposition import select_decomp_table
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.fx.experimental import proxy_tensor
from torch.fx.traceback import preserve_node_meta
from torch.utils import _pytree as pytree

from .. import Config
from .. import exc
from .. import language as hl
from ..autotuner.config_spec import ReductionLoopSpec
from ..language import _tracing_ops
from ..language._decorators import args_to_proxies
from ..language._decorators import get_device_func_replacement
from ..language._tracing_ops import _new_var
from ..language.tile_proxy import Tile
from ..language.tile_proxy import _CheckForIndexCalls
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_read_writes import ReadWrites
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .inductor_lowering import APIFuncLowering
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .inductor_lowering import prepare_graph_lowerings
from .loop_dependency_checker import LoopDependencyChecker
from .matmul_utils import tensor_matmul_replacement
from .matmul_utils import torch_matmul_replacement
from .node_masking import remove_unnecessary_masking
from .roll_reduction import ReductionRoller
from .source_location import current_location
from .type_propagation import CallableType
from .type_propagation import DictType
from .type_propagation import GridIndexType
from .type_propagation import IterType
from .type_propagation import JaggedTileIndexType
from .type_propagation import LiteralType
from .type_propagation import NumericType
from .type_propagation import SequenceType
from .type_propagation import StackTensorType
from .type_propagation import TensorType
from .type_propagation import TileIndexType
from .type_propagation import TypeInfo
from .type_propagation import _eval_binary
from .type_propagation import _eval_compare
from .type_propagation import _eval_unary

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence

    class _TLS(Protocol):
        device_irs: list[DeviceIR]


tls: _TLS = cast("_TLS", threading.local())


def _lerp_scalar_decomp(
    start: torch.Tensor, end: torch.Tensor, weight: float
) -> torch.Tensor:
    # PyTorch nightly's inductor _lerp_scalar decomposition branches on
    # `weight >= 0.5` for numerical stability.  Helion traces scalar kernel
    # args as unbacked symfloats, so that comparison raises
    # GuardOnDataDependentSymNode.  Use the simple algebraic form instead.
    return start + weight * (end - start)


def _get_custom_decomp_table() -> dict[torch._ops.OpOverload, Callable[..., object]]:
    decomp_table = select_decomp_table().copy()
    # Normally, aten.stack is decomposed to aten.unsqueeze + aten.cat, but it's difficult to
    # figure out the right Triton implementation for aten.cat. As a workaround, we disable
    # the decomp for aten.stack and implement aten.stack in Triton (codegen_stack) instead.
    decomp_table.pop(torch.ops.aten.stack.default, None)
    # Override lerp.Scalar to avoid data-dependent guard on the weight parameter.
    decomp_table[torch.ops.aten.lerp.Scalar] = _lerp_scalar_decomp
    return decomp_table


def _make_fx(fn: Callable[..., object], *args: object) -> torch.fx.Graph:
    """
    We monkey patch get_proxy_slot to support Tensor/SymInt/SymFloat/SymBool in the
    graph without any origin for them.  We instead insert _host_tensor(), _get_symnode()
    in the graph to originate them.
    """

    def _get_proxy_slot(
        obj: object,
        tracer: proxy_tensor.PythonKeyTracer,
        default: object = proxy_tensor.no_default,
        transform: Callable[[object], object] = lambda x: x,
    ) -> object:
        if isinstance(obj, torch.Tensor) and not isinstance(obj, Tile):
            tracker = tracer.tensor_tracker
            if obj not in tracker:
                host_function = HostFunction.current()
                origin = host_function.tensor_to_origin.get(obj)
                if origin is not None:
                    assert origin.is_host()
                    # pyrefly: ignore [unsupported-operation]
                    tracker[obj] = proxy = tracer.create_proxy(
                        "call_function",
                        _tracing_ops._host_tensor,
                        (origin.host_str(),),
                        {},
                        name=origin.suggest_var_name(),
                    )
                    proxy.node.meta["val"] = obj
                    proxy.node.meta["lowering"] = APIFuncLowering(
                        _tracing_ops._host_tensor
                    )
                elif obj.numel() == 1 and not isinstance(
                    obj, torch._subclasses.FakeTensor
                ):
                    # Handle constant scalar tensors created inside the kernel
                    # (e.g., torch.tensor(val, dtype=...))
                    # These are real tensors (not FakeTensors) that contain constant values
                    from torch.utils._python_dispatch import _disable_current_modes

                    # Need to exit dispatch modes temporarily to access the real tensor value
                    with _disable_current_modes():
                        value = obj.detach().cpu().item()
                    # pyrefly: ignore [unsupported-operation]
                    tracker[obj] = proxy = tracer.create_proxy(
                        "call_function",
                        _tracing_ops._constant_tensor,
                        (value, obj.dtype),
                        {},
                        name="constant",
                    )
                    proxy.node.meta["val"] = obj
                    proxy.node.meta["lowering"] = APIFuncLowering(
                        _tracing_ops._constant_tensor
                    )
                else:
                    raise KeyError(
                        f"Tensor {obj} not found in tensor_to_origin and is not a scalar constant"
                    )
            return transform(tracker[obj])
        if isinstance(obj, proxy_tensor.py_sym_types):
            tracker = tracer.symnode_tracker
            if obj not in tracker:
                debug_name = CompileEnvironment.current().sympy_debug(obj._sympy_())
                # pyrefly: ignore [unsupported-operation]
                tracker[obj] = proxy = tracer.create_proxy(
                    "call_function",
                    _tracing_ops._get_symnode,
                    (debug_name,),
                    {},
                    name=debug_name if debug_name.isidentifier() else "symnode",
                )
                proxy.node.meta["val"] = obj
                proxy.node.meta["lowering"] = APIFuncLowering(_tracing_ops._get_symnode)
                # pyrefly: ignore [missing-attribute]
                proxy.force = lambda: proxy
            return transform(tracker[obj])
        return get_proxy_slot(obj, tracer, default, transform)

    get_proxy_slot: Callable[..., object] = proxy_tensor.get_proxy_slot

    with (
        preserve_node_meta(),
        patch.object(proxy_tensor, "get_proxy_slot", _get_proxy_slot),
        patch.object(
            torch.fx.proxy,
            "_COPY_META_FIELDS",
            [*torch.fx.proxy._COPY_META_FIELDS, "location"],
        ),
        patch.object(torch, "matmul", torch_matmul_replacement),
        patch.object(
            torch.Tensor,
            "matmul",
            tensor_matmul_replacement,
        ),
    ):
        current_location().set_fx_location()
        return proxy_tensor.make_fx(fn, decomposition_table=_get_custom_decomp_table())(
            *args
        ).graph


@dataclasses.dataclass
class GraphInfo:
    graph_id: int
    graph: torch.fx.Graph

    @property
    def name(self) -> str:
        raise NotImplementedError

    def kwargs(self) -> dict[str, object]:
        """Return a dictionary of keyword needed to copy this graph."""
        return {}

    def __str__(self) -> str:
        output = (
            _LazyGraphModule({}, self.graph).print_readable(print_output=False).strip()
        )
        return textwrap.dedent(
            re.sub(
                r"forward\(self,? ?([^)]*)\)",
                rf"{self.name}(\1)",
                # remove `class <lambda>():` from the output
                re.sub("^[^\n]+\n", "", output),
            )
        )

    def copy(self) -> GraphInfo:
        """Deep-copy the graph using node_copy, preserving metadata."""
        new_graph = torch.fx.Graph()
        node_map: dict[torch.fx.Node, torch.fx.Node] = {}
        for node in self.graph.nodes:
            new_node = new_graph.node_copy(node, lambda n: node_map[n])
            node_map[node] = new_node
        return type(self)(graph_id=self.graph_id, graph=new_graph, **self.kwargs())

    def codegen(self, state: CodegenState) -> list[object]:
        raise NotImplementedError


@dataclasses.dataclass
class RootGraphInfo(GraphInfo):
    phase_index: int = 0

    def kwargs(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(type(self))
            if field.name not in {"graph_id", "graph"}
        }

    @property
    def name(self) -> str:
        return f"root_graph_{self.graph_id}"


@dataclasses.dataclass
class NodeArgsGraphInfo(GraphInfo):
    """Common base class for graphs that have arguments from another graph."""

    node_args: list[torch.fx.Node]

    def placeholder_to_outer_arg(self, node: torch.fx.Node) -> torch.fx.Node:
        assert node.op == "placeholder"
        for placeholder, outer_node in zip(
            node.graph.find_nodes(op="placeholder"),
            self.node_args,
            strict=True,
        ):
            if placeholder is node:
                return outer_node
        raise KeyError("Placeholder not found in node_args")

    def kwargs(self) -> dict[str, object]:
        # TODO(jansel): do we need to map these to the new graph in the case of a copy?
        return {
            "node_args": [*self.node_args],
        }


@dataclasses.dataclass
class ForLoopGraphInfo(NodeArgsGraphInfo):
    block_ids: list[int]

    @property
    def name(self) -> str:
        return f"for_loop_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "block_ids": [*self.block_ids],
        }

    def codegen(self, state: CodegenState) -> list[object]:
        args = state.ast_args[3]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)
        with state.codegen.add_device_loop(
            state.device_function.tile_strategy.codegen_device_loop(
                state, self.block_ids
            )
        ):
            return codegen_call_with_graph(
                state.codegen,
                self.graph,
                args,
            )


class ReductionLoopGraphInfo(ForLoopGraphInfo):
    @property
    def name(self) -> str:
        return f"reduction_loop_{self.graph_id}"


@dataclasses.dataclass
class IfGraphInfo(NodeArgsGraphInfo):
    predicate_is_tensor: bool = False
    else_branch: ElseGraphInfo | None = None

    if_arg_names: list[str] | None = None
    else_arg_names: list[str] | None = None

    # list of outputs of the branches,
    # [(if_out_0, else_out_0), (if_out_1, else_out_1), ...]
    # where each output is represented either as an index into the graph output,
    # or as a name of a non-local variable that is written to
    branches_outputs: list[tuple[int | str, ...]] | None = None

    @property
    def name(self) -> str:
        return f"if_graph_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "predicate_is_tensor": self.predicate_is_tensor,
            "else_branch": self.else_branch,
            "if_arg_names": self.if_arg_names,
            "else_arg_names": self.else_arg_names,
            "branches_outputs": self.branches_outputs,
        }

    def codegen(self, state: CodegenState) -> list[object]:
        from .generate_ast import GenerateAST

        if_args = state.ast_args[3]
        assert isinstance(if_args, list)
        assert all(isinstance(x, ast.AST) for x in if_args)
        else_args = state.ast_args[4]
        assert isinstance(else_args, list)
        assert all(isinstance(x, ast.AST) for x in else_args)

        assert isinstance(state.codegen, GenerateAST)

        test = state.ast_arg(0)
        body_stmts: list[ast.AST] = []
        orelse_stmts: list[ast.AST] = []
        if_ast_node = create(ast.If, test=test, body=body_stmts, orelse=orelse_stmts)
        state.add_statement(if_ast_node)

        with state.codegen.set_statements(body_stmts):
            if_outputs = codegen_call_with_graph(state.codegen, self.graph, if_args)

        else_outputs = []
        if self.else_branch is not None:
            else_graph = state.get_graph(self.else_branch)
            assert isinstance(else_graph, ElseGraphInfo)
            with state.codegen.set_statements(orelse_stmts):
                else_outputs = codegen_call_with_graph(
                    state.codegen, else_graph.graph, else_args
                )

        if len(body_stmts) == 0:
            body_stmts.append(ast.Pass())
        if len(orelse_stmts) == 0:
            orelse_stmts.append(ast.Pass())
        return if_outputs + else_outputs


@dataclasses.dataclass
class ElseGraphInfo(NodeArgsGraphInfo):
    @property
    def name(self) -> str:
        return f"else_graph_{self.graph_id}"

    def codegen(self, state: CodegenState) -> list[object]:
        raise exc.InternalError(
            RuntimeError("ElseGraphInfo should not be codegenned directly")
        )


@dataclasses.dataclass
class WhileConditionGraphInfo(NodeArgsGraphInfo):
    @property
    def name(self) -> str:
        return f"while_condition_{self.graph_id}"

    def codegen(self, state: CodegenState) -> list[object]:
        raise exc.InternalError(
            RuntimeError("WhileConditionGraphInfo should not be codegenned directly")
        )


@dataclasses.dataclass
class WhileLoopGraphInfo(NodeArgsGraphInfo):
    cond_graph_id: int

    @property
    def name(self) -> str:
        return f"while_loop_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "cond_graph_id": self.cond_graph_id,
        }

    def codegen(self, state: CodegenState) -> list[object]:
        cond_info = state.get_graph(self.cond_graph_id)

        args = state.ast_args[2]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)

        def emit_condition(
            target_statements: list[ast.AST],
            cond_args: list[ast.AST] | None = None,
        ) -> ast.expr:
            with state.codegen.set_statements(target_statements):
                cond_outputs = codegen_call_with_graph(
                    state.codegen,
                    cond_info.graph,
                    # pyrefly: ignore [bad-argument-type]
                    cond_args or args,
                    copy_named_args=False,
                )
            if len(cond_outputs) != 1:
                raise exc.InternalError(
                    RuntimeError("While loop condition must produce a single value")
                )
            cond_output = cond_outputs[0]
            if isinstance(cond_output, ast.expr):
                return cond_output
            if isinstance(cond_output, ast.AST):
                return cast("ast.expr", cond_output)
            if isinstance(cond_output, (bool, int, float)):
                return cast("ast.expr", expr_from_string(repr(cond_output)))
            raise exc.InternalError(
                RuntimeError(
                    f"While loop condition produced unsupported value: {cond_output!r}"
                )
            )

        condition_statements: list[ast.AST] = []
        cond_expr = emit_condition(condition_statements)
        cond_var = state.device_function.new_var("while_cond")
        for stmt in condition_statements:
            state.codegen.add_statement(stmt)
        state.codegen.add_statement(
            create(
                ast.Assign,
                targets=[create(ast.Name, id=cond_var, ctx=ast.Store())],
                value=cond_expr,
            )
        )

        body_statements: list[ast.AST] = []
        with state.codegen.set_statements(body_statements):
            outputs = codegen_call_with_graph(
                state.codegen,
                self.graph,
                args,
                copy_named_args=False,
            )
        loop_condition_update: list[ast.AST] = []
        cond_expr_loop = emit_condition(loop_condition_update)
        body_statements.extend(loop_condition_update)
        body_statements.append(
            create(
                ast.Assign,
                targets=[create(ast.Name, id=cond_var, ctx=ast.Store())],
                value=cond_expr_loop,
            )
        )

        state.codegen.add_statement(
            create(
                ast.While,
                test=create(ast.Name, id=cond_var, ctx=ast.Load()),
                body=body_statements,
                orelse=[],
            )
        )
        return outputs


class RolledReductionInfo(NamedTuple):
    rolled_block_ids: list[int]
    original_graph_id: int
    used_rdim: bool
    can_be_rolled_by_caller: bool


@dataclasses.dataclass
class KernelPhase:
    roots: list[int]  # store root indices
    root_nodes: list[ast.For]
    loop_dependency_checker: LoopDependencyChecker = dataclasses.field(
        default_factory=LoopDependencyChecker
    )


class DeviceIR:
    def __init__(self) -> None:
        super().__init__()
        self.graphs: list[GraphInfo] = []
        self.root_ids: list[int] = []
        self.rolled_reductions: list[RolledReductionInfo] = []
        self.phases: list[KernelPhase] = []
        self.grid_block_ids: list[list[int]] = []

    def __str__(self) -> str:
        return "\n\n".join(map(str, self.graphs))

    def debug_str(self) -> str:
        result = str(self)
        # Normalize indentation to 4 spaces to handle both PyTorch 2.9 and nightly formatting
        return re.sub(r" *(# File:\s+).*/([^/:]+:\d+)", r"    \1.../\2", result)

    def add_graph(
        self,
        graph: torch.fx.Graph,
        graph_info_cls: type[GraphInfo] = GraphInfo,
        **kwargs: object,
    ) -> int:
        graph.eliminate_dead_code()
        graph_id = len(self.graphs)
        self.graphs.append(graph_info_cls(graph_id=graph_id, graph=graph, **kwargs))
        return graph_id

    def add_reduction_loop_graph(
        self,
        graph: torch.fx.Graph,
        block_index: int,
        node_args: list[torch.fx.Node],
    ) -> int:
        return self.add_graph(
            graph,
            graph_info_cls=ReductionLoopGraphInfo,
            block_ids=[block_index],
            node_args=node_args,
        )

    def add_root_graph(self, graph: torch.fx.Graph) -> None:
        self.root_ids.append(self.add_graph(graph, graph_info_cls=RootGraphInfo))

    def phase_for_root(self, root_id: int) -> int:
        graph_info = self.graphs[self.root_ids[root_id]]
        assert isinstance(graph_info, RootGraphInfo)
        return graph_info.phase_index

    def register_rollable_reductions(self) -> None:
        """Analyze graphs for rollable reductions and register ReductionLoopSpec entries.

        This is analysis-only: it runs the roller to determine which graphs can
        be rolled, records lightweight RolledReductionInfo entries, and registers
        config_spec entries for the autotuner.  Sub-graphs created by the roller
        (e.g. ReductionLoopGraphInfo) are kept so that _count_device_loads_and_stores
        can account for their loads/stores in the indexing config.
        """
        env = CompileEnvironment.current()
        rdims = [bs for bs in env.block_sizes if bs.reduction]
        if not rdims:
            return
        num_original_graphs = len(self.graphs)

        # First pass: run roller analysis for all reduction dims and
        # record which original graphs use each rdim.
        rdim_results = []
        for rdim in rdims:
            graph_to_info: dict[int, RolledReductionInfo] = {}
            allow_loop = False

            # Check if any graph contains matmul or dev_prts stacking with rdim
            can_roll_graphs = True
            for graph_info in self.graphs[:num_original_graphs]:
                roller = ReductionRoller(self, rdim, {})
                if roller.has_matmul_with_rdim(
                    graph_info.graph
                ) or roller.has_stack_tensor_with_rdim(graph_info.graph):
                    can_roll_graphs = False
                    break

            if not can_roll_graphs:
                rdim_results.append((rdim, False, set()))
                continue

            used_graphs: set[int] = set()
            all_graphs_processed = True
            for graph_id in range(num_original_graphs):
                graph_info = self.graphs[graph_id]
                assert graph_id == graph_info.graph_id
                roller = ReductionRoller(self, rdim, graph_to_info)
                try:
                    roller.process(graph_info.graph)
                except NotImplementedError:
                    all_graphs_processed = False
                    break
                reduction_info = RolledReductionInfo(
                    rolled_block_ids=[rdim.block_id],
                    original_graph_id=graph_id,
                    used_rdim=len(roller.graphs_added) > 0,
                    can_be_rolled_by_caller=roller.outer_count == 0
                    and len(roller.graphs_added) == 1,
                )
                allow_loop = allow_loop or reduction_info.used_rdim
                if reduction_info.used_rdim:
                    used_graphs.add(graph_id)
                self.rolled_reductions.append(reduction_info)
                graph_to_info[graph_id] = reduction_info
            if not all_graphs_processed:
                allow_loop = False
            rdim_results.append((rdim, allow_loop, used_graphs))

        # Second pass: register reduction loop specs, ensuring that each
        # original graph is only rolled for one reduction dim at a time.
        graphs_with_rolled_rdim: set[int] = set()
        for rdim, allow_loop, used_graphs in rdim_results:
            if not allow_loop:
                continue
            if used_graphs & graphs_with_rolled_rdim:
                continue
            env.config_spec.reduction_loops.append(
                ReductionLoopSpec(
                    block_id=rdim.block_id,
                    size_hint=rdim.size_hint(),
                )
            )
            graphs_with_rolled_rdim |= used_graphs

    def build_codegen_graphs(self, config: Config) -> list[GraphInfo]:
        """Build and return graph copies with reduction rolling and epilogue subtiling applied.

        Creates a temporary DeviceIR with copied graphs, applies reduction
        rolling and epilogue subtiling based on the config, and returns the
        resulting graphs. The original graphs are never modified.
        """

        temp = copy.copy(self)
        temp.graphs = [g.copy() for g in self.graphs]
        temp._apply_rolling(config)
        temp._apply_epilogue_subtiling(config)
        return temp.graphs

    def _apply_rolling(self, config: Config) -> None:
        """Apply reduction rolling on the graph copies."""
        env = CompileEnvironment.current()
        reduction_loops = config.reduction_loops

        enabled_reduction_blocks = [
            spec.block_id
            for spec in env.config_spec.reduction_loops
            if env.config_spec.reduction_loops.config_get(
                reduction_loops, spec.block_id, None
            )
            is not None
        ]
        if not enabled_reduction_blocks:
            return

        rdims_by_block = {bs.block_id: bs for bs in env.block_sizes if bs.reduction}
        num_original_graphs = len(self.graphs)

        for block_id in enabled_reduction_blocks:
            rdim = rdims_by_block.get(block_id)
            if rdim is None:
                continue

            # Build graph_to_info from rolled_reductions for this block_id
            graph_to_info: dict[int, RolledReductionInfo] = {}
            for info in self.rolled_reductions:
                if info.rolled_block_ids == [block_id]:
                    graph_to_info[info.original_graph_id] = info

            for graph_id in range(num_original_graphs):
                info = graph_to_info.get(graph_id)
                if info is None or not info.used_rdim:
                    continue
                graph_info = self.graphs[graph_id]
                roller = ReductionRoller(self, rdim, graph_to_info)
                new_graph = roller.process(graph_info.graph)
                new_graph_id = self.add_graph(
                    new_graph, type(graph_info), **graph_info.kwargs()
                )
                # Replace only the graph payload to preserve root metadata
                # (e.g., phase_index used for barrier phase splitting).
                graph_info.graph = self.graphs[new_graph_id].graph

    def _apply_epilogue_subtiling(self, config: Config) -> None:
        """Apply epilogue subtiling on the graph copies if enabled."""
        split_factor = config.epilogue_subtile
        if not split_factor:
            return

        from .epilogue_subtiling import apply_epilogue_subtiling

        env = CompileEnvironment.current()
        configured_block_sizes = {
            info.block_id: info.from_config_assert(config)
            for info in env.block_sizes
            if not info.reduction
        }

        for graph_info in self.graphs:
            if isinstance(graph_info, RootGraphInfo):
                apply_epilogue_subtiling(
                    graph_info.graph,
                    split_factor,
                    configured_block_sizes,
                )

    def __enter__(self) -> None:
        try:
            tls.device_irs.append(self)
        except AttributeError:
            tls.device_irs = [self]

    def __exit__(self, *args: object) -> None:
        tls.device_irs.pop()

    @staticmethod
    def current() -> DeviceIR:
        return tls.device_irs[-1]


class WalkDeviceAST(NodeVisitor):
    def __init__(self, device_ir: DeviceIR) -> None:
        super().__init__()
        self.device_ir = device_ir
        self.scope: dict[str, object] = {}

    def generic_visit(self, node: ast.AST) -> None:
        raise exc.StatementNotSupported(type(node).__name__)

    def _assign(self, target: ast.AST, value: object) -> None:
        if isinstance(target, ast.Name):
            if isinstance(value, torch.Tensor):
                # rename the node to match the variable name
                mode = proxy_tensor.get_proxy_mode()
                assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
                tracer = mode.tracer
                slot = proxy_tensor.get_proxy_slot(value, tracer, default=None)
                if isinstance(slot, proxy_tensor._ProxyTensor):
                    node = slot.proxy.node
                    if target.id not in node.name:
                        node.name = node.graph._graph_namespace.create_name(
                            target.id, None
                        )
            self.scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, n in enumerate(target.elts):
                if isinstance(n, ast.Starred):
                    raise exc.StarredArgsNotSupportedOnDevice

                # pyrefly: ignore [bad-index]
                self._assign(n, value[i])
        elif isinstance(target, ast.Subscript):
            dst = self.visit(target.value)
            assert isinstance(value, torch.Tensor)
            assert isinstance(dst, torch.Tensor)
            hl.store(
                dst,
                self._subscript_slice_proxy(target.slice),
                value,
            )
        else:
            raise NotImplementedError(
                f"Unsupported target type {type(target).__name__}"
            )

    def _body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self.visit(stmt)

    def _static_scope(self) -> dict[str, object]:
        return {k: v for k, v in self.scope.items() if not self.should_become_arg(v)}

    def _lift_inputs(self, names: Iterable[str]) -> LiftTensorArgs:
        return LiftTensorArgs(
            {
                name: self.scope[name]
                for name in names
                if name in self.scope and self.should_become_arg(self.scope[name])
            }
        )

    def _collect_outputs(
        self,
        subgraph_scope: dict[str, object],
        writes: dict[str, int],
        include_new: bool = False,
    ) -> LiftTensorArgs:
        return LiftTensorArgs(
            {
                k: v
                for k, v in subgraph_scope.items()
                if k in writes
                and (include_new or k in self.scope)
                and self.scope.get(k) is not v
            }
        )

    @staticmethod
    def _rw_names(rw: ReadWrites) -> tuple[str, ...]:
        ordered = dict.fromkeys([*rw.reads.keys(), *rw.writes.keys()])
        return tuple(ordered)

    def _trace_graph(
        self,
        inputs: LiftTensorArgs,
        build_fn: Callable[[WalkDeviceAST], tuple[object, LiftTensorArgs]],
        *,
        graph_info_cls: type[NodeArgsGraphInfo],
        copy_tensor_args: bool = True,
        **graph_kwargs: object,
    ) -> tuple[int, LiftTensorArgs]:
        outputs_holder: LiftTensorArgs | None = None

        def runner(*args: object) -> object:
            nonlocal outputs_holder
            subgraph_walker = WalkDeviceAST(self.device_ir)
            subgraph_walker.scope.update(self._static_scope())
            subgraph_walker.scope.update(
                inputs.replace_tensor_args(args, copy_tensors=copy_tensor_args)
            )
            result, outputs_holder = build_fn(subgraph_walker)
            return result

        with self.disable_tracing() as tracer:
            graph = proxy_tensor.make_fx(
                runner, decomposition_table=_get_custom_decomp_table()
            )(*inputs.get_tensor_args()).graph
            graph_id = self.device_ir.add_graph(
                graph,
                graph_info_cls=graph_info_cls,
                node_args=inputs.get_node_args(tracer),
                **graph_kwargs,
            )
        assert outputs_holder is not None
        return graph_id, outputs_holder

    def visit_Pass(self, node: ast.Pass) -> None:
        return None

    def visit_BinOp(self, node: ast.BinOp) -> object:
        left = self.visit(node.left)
        right = self.visit(node.right)
        # Special handling for Tile + offset: expand to tile.index + offset
        # and mark with metadata for indexing strategies to recognize
        if (
            isinstance(node.op, ast.Add)
            and isinstance(left, Tile)
            and isinstance(right, (int, torch.SymInt))
        ):
            # Implicitly expand to tile.index + offset
            left = hl.tile_index(left)
        return _eval_binary(node.op, left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> object:
        return _eval_unary(node.op, self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> object:
        lhs = self.visit(node.left)
        results = []
        for op, rhs in zip(node.ops, node.comparators, strict=True):
            rhs = self.visit(rhs)
            results.append(result := _eval_compare(op, lhs, rhs))
            if not isinstance(result, _tracing_ops._symbolic_types) and not result:
                break
            lhs = rhs
        return functools.reduce(_tracing_ops._and, results)

    def visit_BoolOp(self, node: ast.BoolOp) -> object:
        if isinstance(node.op, ast.And):
            combine_op = _tracing_ops._and
            early_exit = operator.not_
        else:
            assert isinstance(node.op, ast.Or)
            combine_op = _tracing_ops._or
            early_exit = operator.truth
        results = []
        for value in node.values:
            results.append(result := self.visit(value))
            if not isinstance(result, _tracing_ops._symbolic_types) and early_exit(
                result
            ):
                break
        return functools.reduce(combine_op, results)

    @staticmethod
    @contextlib.contextmanager
    def disable_tracing() -> Iterator[proxy_tensor.PythonKeyTracer]:
        mode = proxy_tensor.get_proxy_mode()
        assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
        tracer = mode.tracer
        assert isinstance(tracer, proxy_tensor.PythonKeyTracer)
        with proxy_tensor.disable_proxy_modes_tracing():
            yield tracer

    @staticmethod
    def should_become_arg(value: object) -> bool:
        if isinstance(value, (Tile, int, float, bool, type(None), torch.SymInt)):
            return False
        if isinstance(value, torch.Tensor):
            if (
                origin := HostFunction.current().tensor_to_origin.get(value)
            ) is not None:
                return origin.is_device()
        return True

    def _extract_tile_range(
        self, for_node: ast.For, *, supports_step: bool
    ) -> tuple[object, object, object | None]:
        call_node = for_node.iter
        assert isinstance(call_node, ast.Call)
        func_node = call_node.func
        assert isinstance(func_node, ExtendedAST)
        func_type = func_node._type_info
        assert isinstance(func_type, CallableType)
        assert func_type.value in (hl.jagged_tile, hl.tile, hl.grid, builtins.range)
        args = call_node.args
        assert len(args) >= 1
        if len(args) == 1:
            begin = None
            end = self.visit(args[0])
            step = (
                next(
                    (
                        self.visit(keyword.value)
                        for keyword in call_node.keywords
                        if keyword.arg == "step"
                    ),
                    None,
                )
                if supports_step
                else None
            )
        else:
            begin = self.visit(args[0])
            end = self.visit(args[1])
            step = (
                self.visit(args[2])
                if supports_step and len(args) >= 3
                else next(
                    (
                        self.visit(keyword.value)
                        for keyword in call_node.keywords
                        if keyword.arg == "step"
                    ),
                    None,
                )
                if supports_step
                else None
            )
        return begin, end, step

    def _handle_sequence_unrolling(
        self,
        sequence_iter: ast.AST,
        target: ast.AST,
        element_processor: Callable[[], object | None],
        preserve_scope: bool = False,
    ) -> list[object]:
        """Common logic for unrolling sequences in both loops and comprehensions."""
        # Get the sequence of values to iterate over
        sequence_value = self.visit(sequence_iter)
        assert isinstance(sequence_value, (tuple, list)), (
            f"Expected tuple or list, got {type(sequence_value)}"
        )

        results = []
        for element_value in sequence_value:
            if preserve_scope:
                # For loops: don't create new scope, allow state to persist
                self._assign(target, element_value)
                result = element_processor()
                if result is not None:
                    results.append(result)
            else:
                # For comprehensions: create isolated scope for each iteration
                old_scope = self.scope.copy()
                try:
                    self._assign(target, element_value)
                    result = element_processor()
                    if result is not None:
                        results.append(result)
                finally:
                    self.scope = old_scope

        return results

    def _handle_tuple_unrolling(
        self,
        node: ast.For,
    ) -> None:
        """Handle unrolling of loops that iterate over tuples of tensors."""

        def execute_body() -> None:
            self._body(node.body)
            return None  # No result to collect for loops

        self._handle_sequence_unrolling(
            node.iter, node.target, execute_body, preserve_scope=True
        )

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        assert not node.orelse
        assert isinstance(node.iter, ExtendedAST)
        iter_type = node.iter._type_info

        # Check if we're iterating directly over a sequence (tuple unrolling)
        if isinstance(iter_type, SequenceType):
            self._handle_tuple_unrolling(node)
            return

        # Special handling for variables that might contain sequences from list comprehensions
        if isinstance(node.iter, ast.Name) and node.iter.id in self.scope:
            scope_value = self.scope[node.iter.id]
            if isinstance(scope_value, (tuple, list)):
                # This is a sequence in the scope, we should try to unroll it
                # even if the type info doesn't indicate it's a SequenceType
                self._handle_tuple_unrolling(node)
                return

        if not isinstance(iter_type, IterType):
            raise exc.InvalidDeviceForLoop(iter_type)
        inner_type: TypeInfo = iter_type.inner
        if node._loop_type == LoopType.GRID:
            self._assign(node.target, inner_type.proxy())
            self._body(node.body)
        elif node._loop_type == LoopType.DEVICE:
            rw: ReadWrites = ReadWrites.from_ast(node)
            inputs = self._lift_inputs(self._rw_names(rw))
            supports_step = False
            if isinstance(inner_type, SequenceType):
                supports_step = all(
                    isinstance(value, GridIndexType) for value in inner_type.unpack()
                )
            else:
                supports_step = isinstance(inner_type, GridIndexType)
            begin, end, step = self._extract_tile_range(
                node, supports_step=supports_step
            )
            if isinstance(inner_type, SequenceType):
                iter_vars = inner_type.unpack()
                if begin is None:
                    begin = [0] * len(iter_vars)
                if step is None:
                    step = [None] * len(iter_vars)
            else:
                if isinstance(inner_type, JaggedTileIndexType):
                    # hl.jagged_tile takes a 1D parent tensor, not a scalar bound.
                    assert isinstance(end, torch.Tensor)
                    jagged_parent = end

                    # The first lifted loop input must be the jagged parent tensor.
                    # _setup_mask uses that parent tensor to recover each lane's true end.
                    assert inputs.flat_values[0] is jagged_parent

                    end = torch.amax(jagged_parent)

                iter_vars = [inner_type]
                begin = [0] if begin is None else [begin]
                end = [end]
                step = [step]
            assert all(isinstance(x, (TileIndexType, GridIndexType)) for x in iter_vars)

            def build_subgraph(
                subgraph_walker: WalkDeviceAST,
            ) -> tuple[list[object], LiftTensorArgs]:
                subgraph_walker._assign(node.target, inner_type.proxy())
                subgraph_walker._body(node.body)
                loop_outputs = self._collect_outputs(subgraph_walker.scope, rw.writes)
                return loop_outputs.get_tensor_args(), loop_outputs

            block_ids: list[int] = []
            for var in iter_vars:
                assert isinstance(var, (TileIndexType, GridIndexType))
                block_ids.append(var.block_id)

            graph_idx, outputs = self._trace_graph(
                inputs,
                build_subgraph,
                graph_info_cls=ForLoopGraphInfo,
                block_ids=block_ids,
            )
            step_list = step if isinstance(step, list) else None
            if step_list is None or all(s is None for s in step_list):
                args = (
                    graph_idx,
                    begin,
                    end,
                    inputs.get_tensor_args(),
                )
                loop_target = _tracing_ops._for_loop
            else:
                args = (
                    graph_idx,
                    begin,
                    end,
                    inputs.get_tensor_args(),
                    step_list,
                )
                loop_target = _tracing_ops._for_loop_step
            mode = proxy_tensor.get_proxy_mode()
            assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
            tracer = mode.tracer
            proxy_out = tracer.create_proxy(
                "call_function",
                loop_target,
                # pyrefly: ignore [bad-argument-type]
                *args_to_proxies(tracer, args),
            )
            proxy_tensor.track_tensor_tree(
                outputs.get_tensor_args(),
                proxy_out,
                constant=None,
                tracer=tracer,
            )
            for name, value in outputs.unflatten().items():
                if isinstance(value, Tile):
                    continue
                if name in self.scope:
                    try:
                        self.scope[name] = _tracing_ops._phi(self.scope[name], value)
                    except Exception as e:
                        raise exc.CantCombineTypesInControlFlow(
                            name, self.scope[name], value
                        ) from e
                else:
                    self.scope[name] = value
        else:
            raise AssertionError(f"Unexpected loop type {node._loop_type}")

    def visit_While(self, node: ast.While) -> None:
        if node.orelse:
            raise exc.StatementNotSupported("while ... else ...")

        test_rw = ReadWrites.from_ast(node.test)
        body_rw = ReadWrites.from_list(node.body)
        names = tuple(
            dict.fromkeys((*self._rw_names(test_rw), *self._rw_names(body_rw)))
        )

        inputs = self._lift_inputs(names)

        def build_condition(
            subgraph_walker: WalkDeviceAST,
        ) -> tuple[list[object], LiftTensorArgs]:
            result = subgraph_walker.visit(node.test)
            return [result], LiftTensorArgs({})

        cond_graph_id, _ = self._trace_graph(
            inputs,
            build_condition,
            graph_info_cls=WhileConditionGraphInfo,
            copy_tensor_args=False,
        )

        def build_body(
            subgraph_walker: WalkDeviceAST,
        ) -> tuple[list[object], LiftTensorArgs]:
            subgraph_walker._body(node.body)
            loop_outputs = self._collect_outputs(subgraph_walker.scope, body_rw.writes)
            return loop_outputs.get_tensor_args(), loop_outputs

        body_graph_id, outputs = self._trace_graph(
            inputs,
            build_body,
            graph_info_cls=WhileLoopGraphInfo,
            cond_graph_id=cond_graph_id,
            copy_tensor_args=False,
        )

        args = (
            cond_graph_id,
            body_graph_id,
            inputs.get_tensor_args(),
            None,
        )
        mode = proxy_tensor.get_proxy_mode()
        assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
        tracer = mode.tracer
        proxy_out = tracer.create_proxy(
            "call_function",
            _tracing_ops._while_loop,
            # pyrefly: ignore [bad-argument-type]
            *args_to_proxies(tracer, args),
        )
        proxy_tensor.track_tensor_tree(
            outputs.get_tensor_args(),
            proxy_out,
            constant=None,
            tracer=tracer,
        )

        for name, value in outputs.unflatten().items():
            if isinstance(value, Tile):
                continue
            if name in self.scope:
                try:
                    self.scope[name] = _tracing_ops._phi(self.scope[name], value)
                except Exception as e:
                    raise exc.CantCombineTypesInControlFlow(
                        name, self.scope[name], value
                    ) from e
            else:
                self.scope[name] = value

    def visit_If(self, node: ast.If) -> object:
        test_proxy = self.visit(node.test)
        if not isinstance(test_proxy, _tracing_ops._symbolic_types):
            body = node.body if test_proxy else node.orelse
            if body:
                self._body(body)
            return
        self._create_if_subgraph(test_proxy, node.body, node.orelse)

    def _create_if_subgraph(
        self,
        test_proxy: object,
        body: list[ast.stmt],
        orelse: list[ast.stmt],
    ) -> int:
        # Track whether the predicate is tensor-derived (vs truly scalar).
        predicate_is_tensor = isinstance(test_proxy, torch.Tensor)

        if_branch_rw: ReadWrites = ReadWrites.from_list(body)
        else_branch_rw: ReadWrites = ReadWrites.from_list(orelse)

        if_branch_inputs = self._lift_inputs(self._rw_names(if_branch_rw))
        else_branch_inputs = self._lift_inputs(self._rw_names(else_branch_rw))

        def build_body(
            subgraph_walker: WalkDeviceAST,
            stmts: list[ast.stmt],
            rw: ReadWrites,
        ) -> tuple[list[object], LiftTensorArgs]:
            subgraph_walker._body(stmts)
            outputs_local = self._collect_outputs(
                subgraph_walker.scope, rw.writes, include_new=True
            )
            return outputs_local.get_tensor_args(), outputs_local

        else_graph_idx, else_outputs = self._trace_graph(
            else_branch_inputs,
            functools.partial(build_body, stmts=orelse, rw=else_branch_rw),
            graph_info_cls=ElseGraphInfo,
        )

        if_graph_idx, if_outputs = self._trace_graph(
            if_branch_inputs,
            functools.partial(build_body, stmts=body, rw=if_branch_rw),
            graph_info_cls=IfGraphInfo,
            predicate_is_tensor=predicate_is_tensor,
            else_branch=else_graph_idx,
        )
        if_graph = cast("IfGraphInfo", self.device_ir.graphs[if_graph_idx])

        def get_arg_values_and_names(
            inputs: LiftTensorArgs,
        ) -> tuple[list[object], list[str]]:
            input_tensor_arg_values = inputs.get_tensor_args()

            def is_tensor_arg_value(v: object) -> bool:
                return any(v is t for t in input_tensor_arg_values)

            input_tensor_node_names = [
                k for k, v in inputs.values.items() if is_tensor_arg_value(v)
            ]

            return input_tensor_arg_values, input_tensor_node_names

        if_arg_values, if_graph.if_arg_names = get_arg_values_and_names(
            if_branch_inputs
        )
        else_arg_values, if_graph.else_arg_names = get_arg_values_and_names(
            else_branch_inputs
        )

        args = (
            test_proxy,
            if_graph_idx,
            else_graph_idx,
            if_arg_values,
            else_arg_values,
        )
        mode = proxy_tensor.get_proxy_mode()
        assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
        tracer = mode.tracer
        proxy_out = tracer.create_proxy(
            "call_function",
            _tracing_ops._if,
            # pyrefly: ignore [bad-argument-type]
            *args_to_proxies(tracer, args),
        )
        proxy_tensor.track_tensor_tree(
            if_outputs.get_tensor_args() + else_outputs.get_tensor_args(),
            proxy_out,
            constant=None,
            tracer=tracer,
        )

        if_output_values = if_outputs.values
        else_output_values = else_outputs.values
        common_output_names = [n for n in if_output_values if n in else_output_values]

        # branches_outputs:  [(if_out_0, else_out_0), (if_out_1, else_out_1), ...]
        # where each output is either an index if the graph's output values,
        # or a name of a nonlocal variable which the opposite branch writes to
        if_graph.branches_outputs = []

        def get_output_idx(name: str, output_values: dict[str, object]) -> int:
            return next(i for i, n in enumerate(output_values) if n == name)

        for name in common_output_names:
            if_value = if_output_values[name]
            else_value = else_output_values[name]
            self.scope[name] = _tracing_ops._phi(if_value, else_value)
            if_output_index = get_output_idx(name, if_output_values)
            else_output_index = get_output_idx(name, else_output_values)
            if_graph.branches_outputs.append((if_output_index, else_output_index))

        for name in if_output_values:
            if name not in common_output_names and name in self.scope:
                self.scope[name] = _tracing_ops._phi(
                    self.scope[name], if_output_values[name]
                )
                if_output_index = get_output_idx(name, if_output_values)
                if_graph.branches_outputs.append((if_output_index, name))

        for name in else_output_values:
            if name not in common_output_names and name in self.scope:
                self.scope[name] = _tracing_ops._phi(
                    self.scope[name], else_output_values[name]
                )
                else_output_index = get_output_idx(name, else_output_values)
                if_graph.branches_outputs.append((else_output_index, name))

        return if_graph_idx

    def visit_Name(self, node: ast.Name) -> object:
        if node.id in self.scope:
            return self.scope[node.id]
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        assert type_info is not None and type_info.origin.is_host()
        try:
            return type_info.proxy()
        except NotImplementedError:
            raise exc.CantReadOnDevice(type_info) from None

    def _subscript_slice_proxy(self, slice_node: ast.AST) -> list[object]:
        assert isinstance(slice_node, ExtendedAST)
        result = self.visit(slice_node)
        if isinstance(result, (list, tuple)):
            return [*result]
        return [result]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[object, ...]:
        return tuple([self.visit(x) for x in node.elts])

    def visit_List(self, node: ast.List) -> list[object]:
        return [self.visit(x) for x in node.elts]

    def _visit_comprehension(
        self, node: ast.ListComp | ast.GeneratorExp, name: str
    ) -> tuple[object, ...]:
        """Handle list comprehension or generator expression unrolling."""
        assert isinstance(node, ExtendedAST)

        # Only handle simple cases with single generator and no if conditions
        if len(node.generators) != 1 or node.generators[0].ifs:
            raise exc.StatementNotSupported(f"Complex {name}s are not supported")

        generator = node.generators[0]
        assert isinstance(generator.iter, ExtendedAST)
        iter_type = generator.iter._type_info

        # Check if we're iterating over a sequence (similar to tuple unrolling)
        if isinstance(iter_type, SequenceType):
            return self._handle_comprehension_unrolling(node.elt, generator)

        # For non-sequence iterables, we could extend this later
        raise exc.StatementNotSupported(
            f"{name.capitalize()}s over non-sequence types are not supported"
        )

    def visit_ListComp(self, node: ast.ListComp) -> tuple[object, ...]:
        return self._visit_comprehension(node, "list comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> tuple[object, ...]:
        return self._visit_comprehension(node, "generator expression")

    def _handle_comprehension_unrolling(
        self, elt: ast.expr, generator: ast.comprehension
    ) -> tuple[object, ...]:
        """Handle unrolling of comprehensions (list comp or generator exp) over sequences."""

        def evaluate_expression() -> object:
            # Evaluate the comprehension expression
            result = self.visit(elt)
            # If the result is a SymInt that can be evaluated to a concrete value, do so
            if isinstance(result, torch.SymInt):
                try:
                    return int(result)
                except (ValueError, TypeError):
                    return result
            return result

        results = self._handle_sequence_unrolling(
            generator.iter, generator.target, evaluate_expression, preserve_scope=False
        )
        # Return as tuple to match the expected type for tuple unrolling
        return tuple(results)

    def visit_DictComp(self, node: ast.DictComp) -> dict[object, object]:
        """Handle dict comprehension unrolling."""
        assert isinstance(node, ExtendedAST)

        if len(node.generators) != 1 or node.generators[0].ifs:
            raise exc.StatementNotSupported(
                "Complex dict comprehensions are not supported"
            )

        generator = node.generators[0]
        assert isinstance(generator.iter, ExtendedAST)
        iter_type = generator.iter._type_info

        if not isinstance(iter_type, SequenceType):
            raise exc.StatementNotSupported(
                "Dict comprehensions over non-sequence types are not supported"
            )

        result: dict[object, object] = {}

        def evaluate_key_value() -> None:
            key = self.visit(node.key)
            value = self.visit(node.value)
            result[key] = value

        self._handle_sequence_unrolling(
            generator.iter, generator.target, evaluate_key_value, preserve_scope=False
        )
        return result

    def visit_Dict(self, node: ast.Dict) -> dict[object, object]:
        keys = [self.visit(key) if key is not None else None for key in node.keys]
        values = [self.visit(value) for value in node.values]
        return dict(zip(keys, values, strict=False))

    def visit_Slice(self, node: ast.Slice) -> slice | torch.Tensor:
        if node.lower is None:
            lower = None
        else:
            lower = self.visit(node.lower)
        if node.upper is None:
            upper = None
        else:
            upper = self.visit(node.upper)
        if node.step is None:
            step = None
        else:
            step = self.visit(node.step)

        # Convert slice to hl.arange when step is None or 1 and we have both bounds
        # This allows FX tracing to handle slice operations with dynamic bounds
        if lower is not None and upper is not None and (step is None or step == 1):
            # pyrefly: ignore [bad-argument-type]
            return hl.arange(lower, upper)

        return slice(lower, upper, step)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise exc.AssignmentMultipleTargets
        (target,) = node.targets
        if isinstance(target, ast.Name):
            # TODO(jansel): should assert that name is only used on device
            value = self.visit(node.value)
            # For simple variable assignments like `a = b`, we need to create a new
            # variable to avoid phi node issues when the source variable gets mutated
            if isinstance(node.value, ast.Name) and (
                isinstance(value, torch.Tensor) and not isinstance(value, Tile)
            ):
                value = _new_var(value)
            self._assign(target, value)
            return None
        if isinstance(target, ast.Tuple):
            # Handle tuple unpacking
            value = self.visit(node.value)
            if not isinstance(value, tuple):
                raise exc.InvalidAssignment
            if len(target.elts) != len(value):
                raise exc.InvalidAssignment
            for t, v in zip(target.elts, value, strict=True):
                if isinstance(t, ast.Name):
                    self._assign(t, v)
                elif isinstance(t, ast.Subscript):
                    # Handle subscript targets in tuple unpacking (e.g., a[i], b[j] = tuple)
                    self._assign_subscript(t, v)
                else:
                    raise exc.InvalidAssignment
            return None
        if not isinstance(target, ast.Subscript):
            raise exc.InvalidAssignment
        assert isinstance(node.value, ExtendedAST)
        rhs_type = node.value._type_info
        assert isinstance(target, ExtendedAST)
        lhs_type = target._type_info
        if not isinstance(lhs_type, TensorType) or not isinstance(
            rhs_type, (TensorType, NumericType, LiteralType)
        ):
            raise exc.NonTensorSubscriptAssign(lhs_type, rhs_type)
        assert isinstance(target.value, ExtendedAST)
        assert target.value._type_info is not None
        target_origin = target.value._type_info.origin
        if not target_origin.is_host() and not isinstance(
            target.value._type_info, StackTensorType
        ):
            # Get the variable name for the error message
            var_name = (
                target.value.id
                if isinstance(target.value, ast.Name)
                else str(target.value)
            )
            raise exc.DeviceTensorSubscriptAssignmentNotAllowed(var_name)
        val = self.visit(node.value)
        self._assign_subscript(target, val)

    def _assign_subscript(self, target: ast.Subscript, val: object) -> None:
        """Helper method to assign a value to a subscript target."""
        assert isinstance(target, ExtendedAST)
        lhs_type = target._type_info

        # Validate that we're assigning to a tensor subscript
        from .type_propagation import TensorType

        if not isinstance(lhs_type, TensorType):
            raise exc.NonTensorSubscriptAssign(lhs_type, type(val))

        assert isinstance(target.value, ExtendedAST)
        assert target.value._type_info is not None
        target_origin = target.value._type_info.origin
        assert target_origin.is_host() or isinstance(
            target.value._type_info, StackTensorType
        )

        return hl.store(
            # pyrefly: ignore [bad-argument-type]
            self.visit(target.value),
            self._subscript_slice_proxy(target.slice),
            # pyrefly: ignore [bad-argument-type]
            val,
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(
                create(
                    ast.Assign,
                    targets=[node.target],
                    value=node.value,
                )
            )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        assert isinstance(node.target, ExtendedAST)
        self._assign(
            node.target,
            _eval_binary(node.op, self.visit(node.target), self.visit(node.value)),
        )

    def visit_Subscript(self, node: ast.Subscript) -> object:
        value = node.value
        assert isinstance(value, ExtendedAST)
        type_info = value._type_info
        if isinstance(type_info, SequenceType):
            index_value = self.visit(node.slice)
            if isinstance(index_value, int):
                # pyrefly: ignore [bad-index]
                return self.visit(value)[index_value]
            raise exc.InvalidSequenceSubscription(node.slice)
        # Check StackTensorType before DictType since StackTensorType inherits from DictType
        if isinstance(type_info, StackTensorType):
            # pyrefly: ignore [bad-argument-type]
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))
        if isinstance(type_info, DictType):
            key_value = self.visit(node.slice)
            if isinstance(key_value, (str, int)):
                # pyrefly: ignore [bad-index]
                return self.visit(value)[key_value]
            raise exc.TypeInferenceError(
                f"Dict subscript must be a literal str or int, got {type(key_value).__name__}"
            )
        if type_info is not None and type_info.origin.is_host():
            # pyrefly: ignore [bad-argument-type]
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))
        # pyrefly: ignore [bad-argument-type]
        return hl.subscript(self.visit(value), self._subscript_slice_proxy(node.slice))

    def visit_Call(self, node: ast.Call) -> object:
        args = []
        kwargs = {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                # pyrefly: ignore [bad-argument-type]
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))
        for kwarg in node.keywords:
            if kwarg.arg is None:
                # pyrefly: ignore [no-matching-overload]
                kwargs.update(self.visit(kwarg.value))
            else:
                kwargs[kwarg.arg] = self.visit(kwarg.value)

        if isinstance(
            (
                # pyrefly: ignore [missing-attribute]
                func_type_info := node.func._type_info
            ),
            CallableType,
        ) and (replacement := get_device_func_replacement(func_type_info.value)):
            func = replacement
        else:
            func = self.visit(node.func)

        # pyrefly: ignore [bad-argument-type]
        return _CheckForIndexCalls.retry_call(func, args, kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> object:
        return getattr(self.visit(node.value), node.attr)

    def visit_Expr(self, node: ast.Expr) -> object:
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> object:
        return node.value


class LiftTensorArgs:
    values: dict[str, object]
    flat_values: list[object]
    spec: pytree.TreeSpec
    tensor_indices: list[int]

    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.flat_values, self.spec = pytree.tree_flatten(values)
        self.tensor_indices = [
            i
            for i, v in enumerate(self.flat_values)
            if isinstance(v, torch.Tensor) and not isinstance(v, Tile)
        ]

    def unflatten(self) -> dict[str, object]:
        return pytree.tree_unflatten(self.flat_values, self.spec)

    def replace_tensor_args(
        self, args: Sequence[object], *, copy_tensors: bool = True
    ) -> dict[str, object]:
        flat_values = [*self.flat_values]
        assert len(self.tensor_indices) == len(args)
        for i, v in zip(self.tensor_indices, args, strict=False):
            flat_values[i] = _new_var(v) if copy_tensors else v
        return pytree.tree_unflatten(flat_values, self.spec)

    def get_tensor_args(self) -> list[object]:
        return [self.flat_values[i] for i in self.tensor_indices]

    def get_node_args(
        self, tracer: proxy_tensor.PythonKeyTracer
    ) -> list[torch.fx.Node]:
        proxy_args = args_to_proxies(tracer, self.get_tensor_args())[0]
        result = []
        for proxy in proxy_args:
            assert isinstance(proxy, torch.fx.Proxy)
            result.append(proxy.node)
        return result


class WalkHostAST(NodeVisitor):
    def __init__(self, device_ir: DeviceIR) -> None:
        super().__init__()
        self.device_ir = device_ir
        self.root_index = 0
        self.current_phase_roots: list[int] = []
        self.phases: list[KernelPhase] = []
        self.root_nodes: list[ast.For] = []

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            self.device_ir.add_root_graph(
                _make_fx(lambda: WalkDeviceAST(self.device_ir).visit(node))
            )
            # pyrefly: ignore [missing-attribute]
            iter_type = node.iter._type_info
            assert isinstance(iter_type, IterType)
            inner = iter_type.inner
            if isinstance(inner, SequenceType):
                # pyrefly: ignore [missing-attribute]
                block_ids = [x.block_id for x in inner.unpack()]
            else:
                # pyrefly: ignore [missing-attribute]
                block_ids = [inner.block_id]
            self.device_ir.grid_block_ids.append(block_ids)
            # store root index (position) not graph id
            self.root_nodes.append(node)
            self.current_phase_roots.append(len(self.device_ir.root_ids) - 1)
            self.root_index += 1
        else:
            self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # Record barrier placement between top-level loops.
        from .type_propagation import BarrierResultType

        assert isinstance(node, ExtendedAST)
        assert isinstance(node.value, ExtendedAST)
        is_barrier = isinstance(node.value._type_info, BarrierResultType)

        if is_barrier:
            if self.root_index == 0 or not self.current_phase_roots:
                raise exc.BarrierOnlyAllowedAtTopLevel
            self.phases.append(
                KernelPhase(
                    roots=self.current_phase_roots,
                    root_nodes=[self.root_nodes[r] for r in self.current_phase_roots],
                )
            )
            self.current_phase_roots = []
            return
        self.generic_visit(node)

    def flush_phases(self) -> None:
        if self.current_phase_roots:
            self.phases.append(
                KernelPhase(
                    roots=self.current_phase_roots,
                    root_nodes=[self.root_nodes[r] for r in self.current_phase_roots],
                )
            )
            self.current_phase_roots = []


def _count_device_loads_and_stores(device_ir: DeviceIR) -> tuple[int, int, int]:
    """Count the number of load and store operations in device code for autotuning.

    Returns:
        tuple[int, int, int]: (total_load_count, loads_without_eviction_policy, store_count)
            - total_load_count: all loads (for indexing tunable)
            - loads_without_eviction_policy: loads that need eviction policy tuning
            - store_count: all stores (for indexing tunable)
    """
    from ..language import memory_ops

    total_load_count = 0
    loads_without_eviction_policy = 0
    store_count = 0

    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op == "call_function":
                # Check if this is a load operation
                if node.target is memory_ops.load:
                    total_load_count += 1
                    # Check if this load needs eviction policy tuning
                    # (user can still specify eviction_policy to override tuning)
                    eviction_policy_arg = node.kwargs.get("eviction_policy")
                    if eviction_policy_arg is None:
                        # Check if eviction_policy was passed as positional arg (index 3)
                        if len(node.args) >= 4:
                            eviction_policy_arg = node.args[3]
                        if eviction_policy_arg is None:
                            loads_without_eviction_policy += 1
                # Check if this is a store operation
                elif node.target is memory_ops.store:
                    store_count += 1

    return total_load_count, loads_without_eviction_policy, store_count


def _register_load_store_tunables(
    total_load_count: int, loads_without_eviction_policy: int, store_count: int
) -> None:
    """Register list-based tunables (indexing, eviction policies) for all device loads and stores.

    Args:
        total_load_count: Total number of loads (for indexing tunable)
        loads_without_eviction_policy: Number of loads that need eviction policy tuning
        store_count: Total number of stores (for indexing tunable)
    """
    if total_load_count == 0 and store_count == 0:
        return

    from ..autotuner.config_fragment import EnumFragment
    from ..autotuner.config_fragment import ListOf
    from ..autotuner.config_spec import get_valid_eviction_policies

    env = CompileEnvironment.current()

    # Register eviction policies only for loads without explicit eviction_policy
    if loads_without_eviction_policy > 0:
        env.config_spec.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(env.backend_name)),
            length=loads_without_eviction_policy,
        )
        env.device_load_count = loads_without_eviction_policy

    # Indexing applies to ALL loads and stores
    total_count = total_load_count + store_count
    if total_count > 0:
        env.config_spec.indexing = ListOf(
            EnumFragment(choices=env.config_spec.valid_indexing_types()),
            length=total_count,
        )


def lower_to_device_ir(func: HostFunction) -> DeviceIR:
    device_ir = DeviceIR()
    with func, device_ir, compile_lock:
        visitor = WalkHostAST(device_ir)
        for stmt in func.body:
            visitor.visit(stmt)
        visitor.flush_phases()
        device_ir.phases = visitor.phases
        # Run dependency checks once, per phase, so codegen does not redo it per-config.
        for phase in device_ir.phases:
            checker = phase.loop_dependency_checker
            for loop_node in phase.root_nodes:
                checker.register_loop(loop_node)
        for phase_idx, phase in enumerate(device_ir.phases):
            for ridx in phase.roots:
                graph_info = device_ir.graphs[device_ir.root_ids[ridx]]
                assert isinstance(graph_info, RootGraphInfo)
                graph_info.phase_index = phase_idx
        # If there are no top-level device loops, we cannot generate a valid kernel.
        # Raise a friendly error instead of emitting an empty Triton function body.
        if len(device_ir.root_ids) == 0:
            raise exc.NoDeviceLoopsInKernel
        from ..language.random_ops import rewrite_implicit_random_ops

        for graph in device_ir.graphs:
            rewrite_implicit_random_ops(graph.graph)
        for graph in device_ir.graphs:
            prepare_graph_lowerings(graph.graph)
        for graph in device_ir.graphs:
            validate_host_tensor_usage(graph.graph)
            add_tile_with_offset_metadata(graph)
            remove_unnecessary_tile_index(graph.graph)
            remove_unnecessary_masking(graph.graph)

        from .epilogue_subtiling import has_epilogue_subtiling_candidate

        has_epilogue_subtile_candidate = False
        for graph_info in device_ir.graphs:
            if not isinstance(graph_info, RootGraphInfo):
                continue
            if has_epilogue_subtiling_candidate(graph_info.graph):
                has_epilogue_subtile_candidate = True
                break
        config_spec = CompileEnvironment.current().config_spec
        config_spec.epilogue_subtile_candidate_enabled = has_epilogue_subtile_candidate
        config_spec.epilogue_subtile_k_hint = 0
        config_spec.epilogue_subtile_autotune_choices = None

        device_ir.register_rollable_reductions()
        CompileEnvironment.current().config_spec.raise_grid_block_minimums()
        if len(device_ir.root_ids) > 1:
            # xyz not supported with shared program IDs, but persistent kernels are allowed
            CompileEnvironment.current().config_spec.disallow_pid_type("xyz")

        # Count all device loads and stores and register tunables
        total_load_count, loads_without_eviction_policy, store_count = (
            _count_device_loads_and_stores(device_ir)
        )
        _register_load_store_tunables(
            total_load_count, loads_without_eviction_policy, store_count
        )

        return device_ir


@dataclasses.dataclass
class HelperFunctionGraphInfo(NodeArgsGraphInfo):
    """Graph info for helper functions in higher-order operations like associative_scan."""

    _param_names: list[str] = dataclasses.field(default_factory=list)
    original_function_name: str | None = dataclasses.field(default=None)

    @property
    def name(self) -> str:
        # This property should only be used during registration, not for final names
        # Final names are generated in codegen using the namespace below
        if self.original_function_name:
            return f"{self.original_function_name}_{self.graph_id}"
        return f"helper_function_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            **super().kwargs(),
            "_param_names": [*self._param_names],
            "original_function_name": self.original_function_name,
        }

    def find_input_nodes(self) -> list[torch.fx.Node]:
        """Find all placeholder nodes (inputs) in the graph."""
        return self.graph.find_nodes(op="placeholder")

    def codegen(self, state: CodegenState) -> list[object]:
        from .helper_function import codegen_helper_function_graph_info

        return codegen_helper_function_graph_info(self, state)


def validate_host_tensor_usage(graph: torch.fx.Graph) -> None:
    """
    Validate that scalar _host_tensor ops only flow into allowed operations.
    This replaces the AST visitor context detection with cleaner FX graph validation.
    Only checks 0-dimensional tensors (scalars), not regular tensors.
    Uses decorator metadata to determine which operations allow host tensors.
    """
    from ..language._decorators import is_api_func
    from ..language._tracing_ops import _host_tensor

    for node in graph.find_nodes(op="call_function", target=_host_tensor):
        scalar_tensor_name = node.args[0]
        assert isinstance(scalar_tensor_name, str), scalar_tensor_name

        # Check all users of this scalar _host_tensor node
        for user in node.users:
            if user.op == "call_function":
                # Check if this operation allows host tensors via decorator metadata
                if not (
                    is_api_func(user.target)
                    and getattr(user.target, "_allow_host_tensor", False)
                ):
                    op_name = getattr(user.target, "__name__", str(user.target))
                    raise exc.HostTensorDirectUsage(scalar_tensor_name, op_name)


def add_tile_with_offset_metadata(graph_info: GraphInfo) -> None:
    """
    Recognize tile.index + offset patterns and add metadata to enable tensor descriptor indexing.

    This pass identifies FX nodes that represent `tile.index + offset` (where offset is an
    integer or SymInt), and adds the `tile_with_offset` metadata to those nodes so that
    indexing strategies can generate efficient code (e.g., tensor descriptors) for them.
    """
    graph = graph_info.graph
    env = CompileEnvironment.current()
    add_targets = (operator.add, torch.ops.aten.add.Tensor)
    offset_types = (int, torch.SymInt)
    for node in graph.nodes:
        if (
            node.op != "call_function"
            or node.target not in add_targets
            or node.kwargs
            or len(node.args) != 2
        ):
            continue

        block_id: int | None = None
        total_offset: int | torch.SymInt = 0
        valid = True

        for arg in node.args:
            tile_offset_value: int | torch.SymInt | None = None
            arg_block_id: int | None = None

            if isinstance(arg, torch.fx.Node):
                meta_tile = arg.meta.get("tile_with_offset")
                if meta_tile is not None:
                    arg_block_id = meta_tile.get("block_id")
                    if arg_block_id is None:
                        valid = False
                        break
                    tile_offset_value = meta_tile.get("offset", 0)
                elif (
                    arg.op == "call_function"
                    and arg.target == hl.tile_index
                    and arg.args
                    and isinstance(arg.args[0], torch.fx.Node)
                ):
                    tile_val = arg.args[0].meta.get("val")
                    if isinstance(tile_val, torch.SymInt):
                        arg_block_id = env.get_block_id(tile_val)
                        if arg_block_id is None:
                            valid = False
                            break
                        tile_offset_value = 0
                else:
                    val = arg.meta.get("val")
                    if isinstance(val, offset_types):
                        total_offset = total_offset + val
                        continue

                if arg_block_id is not None:
                    if block_id is not None:
                        valid = False
                        break
                    if tile_offset_value is None:
                        tile_offset_value = 0
                    block_id = arg_block_id
                    total_offset = total_offset + tile_offset_value
                    continue

                val = arg.meta.get("val")
                if isinstance(val, offset_types):
                    total_offset = total_offset + val
                    continue

                valid = False
                break

            if isinstance(arg, offset_types):
                total_offset = total_offset + arg
                continue
            valid = False
            break

        if not valid or block_id is None:
            continue

        node.meta["tile_with_offset"] = {
            "block_id": block_id,
            "offset": total_offset,
        }


def remove_unnecessary_tile_index(graph: torch.fx.Graph) -> None:
    """
    Remove unnecessary tile_index nodes from the graph.
    Passing a tile directly results block_ptrs being supported.
    """
    for node in graph.find_nodes(op="call_function", target=hl.tile_index):
        for user in [*node.users]:
            if user.op == "call_function" and user.target in (hl.load, hl.store):
                new_args = [*user.args]
                assert isinstance(new_args[1], (list, tuple))
                new_args[1] = [(node.args[0] if x is node else x) for x in new_args[1]]
                user.args = tuple(new_args)
        if len(node.users) == 0:
            graph.erase_node(node)
