from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import functools
import operator
import re
import textwrap
import threading
from typing import TYPE_CHECKING
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
from .ast_read_writes import ReadWrites
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .inductor_lowering import APIFuncLowering
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .inductor_lowering import prepare_graph_lowerings
from .matmul_utils import tensor_matmul_replacement
from .matmul_utils import torch_matmul_replacement
from .node_masking import remove_unnecessary_masking
from .roll_reduction import ReductionRoller
from .source_location import current_location
from .type_propagation import CallableType
from .type_propagation import GridIndexType
from .type_propagation import IterType
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
    from collections.abc import Sequence

    class _TLS(Protocol):
        device_irs: list[DeviceIR]


tls: _TLS = cast("_TLS", threading.local())


def _get_custom_decomp_table() -> dict[torch._ops.OpOverload, Callable[..., object]]:
    decomp_table = select_decomp_table().copy()
    # Normally, aten.stack is decomposed to aten.unsqueeze + aten.cat, but it's difficult to
    # figure out the right Triton implementation for aten.cat. As a workaround, we disable
    # the decomp for aten.stack and implement aten.stack in Triton (codegen_stack) instead.
    decomp_table.pop(torch.ops.aten.stack.default, None)
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
                origin = HostFunction.current().tensor_to_origin[obj]
                assert origin.is_host()
                tracker[obj] = proxy = tracer.create_proxy(  # pyright: ignore[reportArgumentType]
                    "call_function",
                    _tracing_ops._host_tensor,
                    (origin.host_str(),),
                    {},
                    name=origin.suggest_var_name(),
                )
                proxy.node.meta["val"] = obj
                proxy.node.meta["lowering"] = APIFuncLowering(_tracing_ops._host_tensor)
            return transform(tracker[obj])
        if isinstance(obj, proxy_tensor.py_sym_types):
            tracker = tracer.symnode_tracker
            if obj not in tracker:
                debug_name = CompileEnvironment.current().sympy_debug(obj._sympy_())
                tracker[obj] = proxy = tracer.create_proxy(  # pyright: ignore[reportArgumentType]
                    "call_function",
                    _tracing_ops._get_symnode,
                    (debug_name,),
                    {},
                    name=debug_name if debug_name.isidentifier() else "symnode",
                )
                proxy.node.meta["val"] = obj
                proxy.node.meta["lowering"] = APIFuncLowering(_tracing_ops._get_symnode)
                proxy.force = lambda: proxy  # pyright: ignore[reportAttributeAccessIssue]
            return transform(tracker[obj])
        return get_proxy_slot(obj, tracer, default, transform)

    get_proxy_slot: Callable[..., object] = proxy_tensor.get_proxy_slot
    with (
        preserve_node_meta(),
        patch.object(proxy_tensor, "get_proxy_slot", _get_proxy_slot),
        patch.object(
            torch.fx.proxy,  # pyright: ignore[reportAttributeAccessIssue]
            "_COPY_META_FIELDS",
            [*torch.fx.proxy._COPY_META_FIELDS, "location"],  # pyright: ignore[reportAttributeAccessIssue]
        ),
        patch.object(torch, "matmul", torch_matmul_replacement),
        patch.object(
            torch.Tensor,
            "matmul",
            tensor_matmul_replacement,
        ),
    ):
        current_location().set_fx_location()
        return proxy_tensor.make_fx(fn, decomposition_table=select_decomp_table())(
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

    def codegen(self, state: CodegenState) -> list[object]:
        raise NotImplementedError


class RootGraphInfo(GraphInfo):
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
        args = state.ast_args[-1]
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


class IfGraphInfo(NodeArgsGraphInfo):
    @property
    def name(self) -> str:
        return f"if_else_graph_{self.graph_id}"

    def codegen(self, state: CodegenState) -> list[object]:
        test = state.ast_arg(0)

        args = state.ast_args[2]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)
        state.add_statement(create(ast.If, test=test, body=(body := []), orelse=[]))
        with state.codegen.set_statements(body):
            return codegen_call_with_graph(state.codegen, self.graph, args)


class RolledReductionInfo(NamedTuple):
    rolled_block_ids: list[int]
    original_graph_id: int
    new_graph_id: int | None
    used_rdim: bool
    can_be_rolled_by_caller: bool


class DeviceIR:
    def __init__(self) -> None:
        super().__init__()
        self.graphs: list[GraphInfo] = []
        self.root_ids: list[int] = []
        self.rolled_reductions: list[RolledReductionInfo] = []
        self.grid_block_ids: list[list[int]] = []

    def get_root(self, config: Config, graph_id: int) -> torch.fx.Graph:
        """ " If we are using a rolled reduction, return the rolled reduction graph otherwise
        return the root graph."""
        if graph_id >= len(self.graphs):
            raise AssertionError("Invalid graph id")
        reduction_loops = config.reduction_loops
        if len(reduction_loops) > 1:
            raise NotImplementedError("Multiple reduction loops not implemented")
        if len(reduction_loops) == 0 or reduction_loops[0] is None:
            return self.graphs[graph_id].graph
        for info in reversed(self.rolled_reductions):
            if info.original_graph_id == graph_id:
                assert info.new_graph_id is not None
                return self.graphs[info.new_graph_id].graph
        raise AssertionError("No rolled reduction graph found")

    def __str__(self) -> str:
        return "\n\n".join(map(str, self.graphs))

    def debug_str(self) -> str:
        result = str(self)
        return re.sub(r" ?(# File:\s+).*/([^/:]+:\d+)", r"\1.../\2", result)

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

    def build_rolled_reductions(self) -> None:
        env = CompileEnvironment.current()
        rdims = [bs for bs in env.block_sizes if bs.reduction]
        if not rdims:
            return
        first = True
        for rdim in rdims:
            graph_to_info = {}
            allow_loop = False

            # First, check if any graph contains matmul or dev_prts stacking with rdim
            # If so, we can't roll any graphs in this reduction dimension
            can_roll_graphs = True
            for graph_info in self.graphs:
                roller = ReductionRoller(self, rdim, {})
                if roller.has_matmul_with_rdim(
                    graph_info.graph
                ) or roller.has_stack_tensor_with_rdim(graph_info.graph):
                    can_roll_graphs = False
                    break

            if not can_roll_graphs:
                first = False
                continue

            # Process graphs normally
            for graph_id, graph_info in enumerate([*self.graphs]):
                assert graph_id == graph_info.graph_id
                roller = ReductionRoller(self, rdim, graph_to_info)
                try:
                    new_graph = roller.process(graph_info.graph)
                except NotImplementedError:
                    first = False
                    break
                new_graph_id = self.add_graph(
                    new_graph, type(graph_info), **graph_info.kwargs()
                )
                reduction_info = RolledReductionInfo(
                    rolled_block_ids=[rdim.block_id],
                    original_graph_id=graph_id,
                    new_graph_id=new_graph_id,
                    used_rdim=len(roller.graphs_added) > 0,
                    can_be_rolled_by_caller=roller.outer_count == 0
                    and len(roller.graphs_added) == 1,
                )
                allow_loop = allow_loop or reduction_info.used_rdim
                self.rolled_reductions.append(reduction_info)
                graph_to_info[graph_id] = reduction_info
            if allow_loop and first:
                # TODO(jansel): we should add support for rolling multiple dims at once
                env.config_spec.reduction_loops.append(
                    ReductionLoopSpec(
                        block_id=rdim.block_id,
                        size_hint=rdim.size_hint(),
                    )
                )
            first = False

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

                self._assign(n, value[i])  # pyright: ignore[reportIndexIssue]
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

    def _extract_tile_begin_end(self, for_node: ast.For) -> tuple[object, object]:
        call_node = for_node.iter
        assert isinstance(call_node, ast.Call)
        func_node = call_node.func
        assert isinstance(func_node, ExtendedAST)
        func_type = func_node._type_info
        assert isinstance(func_type, CallableType)
        assert func_type.value in (hl.tile, hl.grid, builtins.range)
        args = call_node.args
        assert len(args) >= 1
        if len(args) == 1:
            begin = None
            end = self.visit(args[0])
        else:
            begin = self.visit(args[0])
            end = self.visit(args[1])
        return begin, end

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
            inputs: LiftTensorArgs = LiftTensorArgs(
                {
                    k: self.scope[k]
                    for k in rw
                    if k in self.scope and self.should_become_arg(self.scope[k])
                }
            )
            outputs: LiftTensorArgs | None = None
            begin, end = self._extract_tile_begin_end(node)
            if isinstance(inner_type, SequenceType):
                iter_vars = inner_type.unpack()
                if begin is None:
                    begin = [0] * len(iter_vars)
            else:
                iter_vars = [inner_type]
                begin = [0] if begin is None else [begin]
                end = [end]
            assert all(isinstance(x, (TileIndexType, GridIndexType)) for x in iter_vars)

            def run_subgraph(*args: object) -> list[object]:
                nonlocal outputs
                subgraph_walker = WalkDeviceAST(self.device_ir)
                subgraph_walker.scope.update(
                    {
                        k: v
                        for k, v in self.scope.items()
                        if not self.should_become_arg(v)
                    }
                )
                subgraph_walker.scope.update(inputs.replace_tensor_args(args))
                subgraph_walker._assign(node.target, inner_type.proxy())
                subgraph_walker._body(node.body)

                outputs = LiftTensorArgs(
                    {
                        k: v
                        for k, v in subgraph_walker.scope.items()
                        if k in rw.writes
                        # Only propagate variables that existed before the loop and have been modified
                        and (k in self.scope and self.scope[k] is not v)
                    }
                )
                return outputs.get_tensor_args()

            with self.disable_tracing() as tracer:
                graph = proxy_tensor.make_fx(
                    run_subgraph, decomposition_table=_get_custom_decomp_table()
                )(*inputs.get_tensor_args()).graph
                graph_idx = self.device_ir.add_graph(
                    graph,
                    ForLoopGraphInfo,
                    block_ids=[x.block_id for x in iter_vars],  # pyright: ignore[reportAttributeAccessIssue]
                    node_args=inputs.get_node_args(tracer),
                )
                args = (
                    graph_idx,
                    begin,
                    end,
                    inputs.get_tensor_args(),
                )
                proxy_out = tracer.create_proxy(
                    "call_function",
                    _tracing_ops._for_loop,
                    *args_to_proxies(tracer, args),
                )
                assert outputs is not None
                proxy_tensor.track_tensor_tree(
                    [*outputs.get_tensor_args()],
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

    def visit_If(self, node: ast.If) -> object:
        test_proxy = self.visit(node.test)
        if not isinstance(test_proxy, _tracing_ops._symbolic_types):
            body = node.body if test_proxy else node.orelse
            if body:
                self._body(body)
            return
        self._create_if_subgraph(test_proxy, node.body)
        if node.orelse:
            self._create_if_subgraph(_tracing_ops._not(test_proxy), node.orelse)

    def _create_if_subgraph(self, test_proxy: object, body: list[ast.stmt]) -> None:
        rw: ReadWrites = ReadWrites.from_list(body)
        inputs: LiftTensorArgs = LiftTensorArgs(
            {
                k: self.scope[k]
                for k in rw
                if k in self.scope and self.should_become_arg(self.scope[k])
            }
        )
        outputs: LiftTensorArgs | None = None

        def run_body(*args: object) -> list[object]:
            nonlocal outputs
            subgraph_walker = WalkDeviceAST(self.device_ir)
            subgraph_walker.scope.update(
                {k: v for k, v in self.scope.items() if not self.should_become_arg(v)}
            )
            subgraph_walker.scope.update(inputs.replace_tensor_args(args))
            subgraph_walker._body(body)
            outputs = LiftTensorArgs(
                {
                    k: v
                    for k, v in subgraph_walker.scope.items()
                    if k in rw.writes
                    and (k not in self.scope or self.scope[k] is not v)
                }
            )
            return outputs.get_tensor_args()

        with self.disable_tracing() as tracer:
            body_graph = proxy_tensor.make_fx(
                run_body, decomposition_table=_get_custom_decomp_table()
            )(*inputs.get_tensor_args()).graph
            assert outputs is not None
            graph_idx = self.device_ir.add_graph(
                body_graph,
                IfGraphInfo,
                node_args=inputs.get_node_args(tracer),
            )
            args = (
                test_proxy,
                graph_idx,
                inputs.get_tensor_args(),
            )
            proxy_out = tracer.create_proxy(
                "call_function",
                _tracing_ops._if,
                *args_to_proxies(tracer, args),
            )
            proxy_tensor.track_tensor_tree(
                [*outputs.get_tensor_args()],
                proxy_out,
                constant=None,
                tracer=tracer,
            )
        for name, value in outputs.unflatten().items():
            if name in self.scope:
                try:
                    self.scope[name] = _tracing_ops._phi(self.scope[name], value)
                except Exception as e:
                    raise exc.CantCombineTypesInControlFlow(
                        name, self.scope[name], value
                    ) from e
            else:
                self.scope[name] = value

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

    def visit_ListComp(self, node: ast.ListComp) -> tuple[object, ...]:
        """Handle list comprehension unrolling similar to tuple unrolling."""
        assert isinstance(node, ExtendedAST)

        # Only handle simple cases with single generator and no if conditions
        if len(node.generators) != 1 or node.generators[0].ifs:
            raise exc.StatementNotSupported(
                "Complex list comprehensions are not supported"
            )

        generator = node.generators[0]
        assert isinstance(generator.iter, ExtendedAST)
        iter_type = generator.iter._type_info

        # Check if we're iterating over a sequence (similar to tuple unrolling)
        if isinstance(iter_type, SequenceType):
            return self._handle_listcomp_unrolling(node)

        # For non-sequence iterables, we could extend this later
        raise exc.StatementNotSupported(
            "List comprehensions over non-sequence types are not supported"
        )

    def _handle_listcomp_unrolling(self, node: ast.ListComp) -> tuple[object, ...]:
        """Handle unrolling of list comprehensions over sequences."""
        generator = node.generators[0]

        def evaluate_expression() -> object:
            # Evaluate the comprehension expression
            result = self.visit(node.elt)
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
            return hl.arange(lower, upper)  # pyright: ignore[reportArgumentType]

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
        target_origin = target.value._type_info.origin  # pyright: ignore[reportOptionalMemberAccess]
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
            self.visit(target.value),  # pyright: ignore[reportArgumentType]
            self._subscript_slice_proxy(target.slice),
            val,  # pyright: ignore[reportArgumentType]
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
            if isinstance(node.slice, ast.Constant):
                return self.visit(value)[self.visit(node.slice)]  # pyright: ignore[reportIndexIssue]
            raise exc.InvalidSequenceSubscription(node.slice)
        if isinstance(type_info, StackTensorType):
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))  # pyright: ignore[reportArgumentType]
        if type_info is not None and type_info.origin.is_host():
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))  # pyright: ignore[reportArgumentType]
        return hl.subscript(self.visit(value), self._subscript_slice_proxy(node.slice))  # pyright: ignore[reportArgumentType]

    def visit_Call(self, node: ast.Call) -> object:
        args = []
        kwargs = {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                args.extend(self.visit(arg.value))  # pyright: ignore[reportArgumentType]
            else:
                args.append(self.visit(arg))
        for kwarg in node.keywords:
            if kwarg.arg is None:
                kwargs.update(self.visit(kwarg.value))  # pyright: ignore[reportArgumentType,reportCallIssue]
            else:
                kwargs[kwarg.arg] = self.visit(kwarg.value)

        if isinstance(
            (
                func_type_info := node.func._type_info  # pyright: ignore[reportAttributeAccessIssue]
            ),
            CallableType,
        ) and (replacement := get_device_func_replacement(func_type_info.value)):
            func = replacement
        else:
            func = self.visit(node.func)

        return _CheckForIndexCalls.retry_call(func, args, kwargs)  # pyright: ignore[reportArgumentType]

    def visit_Attribute(self, node: ast.Attribute) -> object:
        return getattr(self.visit(node.value), node.attr)

    def visit_Expr(self, node: ast.Expr) -> object:
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> object:
        return node.value


class LiftTensorArgs:
    flat_values: list[object]
    spec: pytree.TreeSpec
    tensor_indices: list[int]

    def __init__(self, values: dict[str, object]) -> None:
        self.flat_values, self.spec = pytree.tree_flatten(values)
        self.tensor_indices = [
            i
            for i, v in enumerate(self.flat_values)
            if isinstance(v, torch.Tensor) and not isinstance(v, Tile)
        ]

    def unflatten(self) -> dict[str, object]:
        return pytree.tree_unflatten(self.flat_values, self.spec)

    def replace_tensor_args(self, args: Sequence[object]) -> dict[str, object]:
        flat_values = [*self.flat_values]
        assert len(self.tensor_indices) == len(args)
        for i, v in zip(self.tensor_indices, args, strict=False):
            flat_values[i] = _new_var(v)
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

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            self.device_ir.add_root_graph(
                _make_fx(lambda: WalkDeviceAST(self.device_ir).visit(node))
            )
            iter_type = node.iter._type_info  # pyright: ignore[reportAttributeAccessIssue]
            assert isinstance(iter_type, IterType)
            inner = iter_type.inner
            if isinstance(inner, SequenceType):
                block_ids = [x.block_id for x in inner.unpack()]  # pyright: ignore[reportAttributeAccessIssue]
            else:
                block_ids = [inner.block_id]  # pyright: ignore[reportAttributeAccessIssue]
            self.device_ir.grid_block_ids.append(block_ids)
        else:
            self.generic_visit(node)


def _count_device_loads_and_stores(device_ir: DeviceIR) -> tuple[int, int, int]:
    """Count the number of load and store operations in device code for autotuning.

    Returns:
        tuple[int, int, int]: (total_load_count, loads_without_eviction_policy, store_count)
            - total_load_count: all loads (for indexing tunable)
            - loads_without_eviction_policy: loads that need eviction policy tuning
            - store_count: all stores (for indexing tunable)
    """
    from ..language import memory_ops

    # Build set of rolled graph IDs to exclude (these are duplicates)
    rolled_graph_ids = {
        info.new_graph_id
        for info in device_ir.rolled_reductions
        if info.new_graph_id is not None
    }

    total_load_count = 0
    loads_without_eviction_policy = 0
    store_count = 0

    # Walk all graphs except rolled duplicates
    for graph_info in device_ir.graphs:
        if graph_info.graph_id in rolled_graph_ids:
            continue

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
    from ..autotuner.config_spec import VALID_EVICTION_POLICIES
    from ..autotuner.config_spec import ConfigSpec

    env = CompileEnvironment.current()

    # Register eviction policies only for loads without explicit eviction_policy
    if loads_without_eviction_policy > 0:
        env.config_spec.load_eviction_policies = ListOf(
            EnumFragment(choices=VALID_EVICTION_POLICIES),
            length=loads_without_eviction_policy,
        )
        env.device_load_count = loads_without_eviction_policy

    # Indexing applies to ALL loads and stores
    total_count = total_load_count + store_count
    if total_count > 0:
        env.config_spec.indexing = ListOf(
            EnumFragment(choices=ConfigSpec._valid_indexing_types()), length=total_count
        )


def lower_to_device_ir(func: HostFunction) -> DeviceIR:
    device_ir = DeviceIR()
    with func, device_ir, compile_lock:
        visitor = WalkHostAST(device_ir)
        for stmt in func.body:
            visitor.visit(stmt)
        # If there are no top-level device loops, we cannot generate a valid kernel.
        # Raise a friendly error instead of emitting an empty Triton function body.
        if len(device_ir.root_ids) == 0:
            raise exc.NoDeviceLoopsInKernel
        for graph in device_ir.graphs:
            prepare_graph_lowerings(graph.graph)
        for graph in device_ir.graphs:
            validate_host_tensor_usage(graph.graph)
            add_tile_with_offset_metadata(graph)
            remove_unnecessary_tile_index(graph.graph)
            remove_unnecessary_masking(graph.graph)
        device_ir.build_rolled_reductions()
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
