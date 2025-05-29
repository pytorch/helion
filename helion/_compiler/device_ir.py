from __future__ import annotations

import ast
from collections.abc import Callable
import contextlib
import dataclasses
import functools
import operator
import re
import textwrap
from typing import TYPE_CHECKING
from typing import Iterator
from typing import NamedTuple
from unittest.mock import patch

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental import proxy_tensor
from torch.fx.traceback import preserve_node_meta
from torch.utils import _pytree as pytree

from .. import Config
from .. import exc
from .. import language as hl
from ..autotuner.config_spec import ReductionLoopSpec
from ..language import _tracing_ops
from ..language._decorators import args_to_proxies
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_read_writes import ReadWrites
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .inductor_lowering import prepare_graph_lowerings
from .roll_reduction import ReductionRoller
from .source_location import current_location
from .tile_index_proxy import CheckForIndexCalls
from .tile_index_proxy import TileIndexProxy
from .type_propagation import CallableType
from .type_propagation import GridIndexType
from .type_propagation import IterType
from .type_propagation import SequenceType
from .type_propagation import TensorType
from .type_propagation import TileIndexType
from .type_propagation import TypeInfo
from .type_propagation import _eval_binary
from .type_propagation import _eval_compare
from .type_propagation import _eval_unary

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


def _make_fx(fn: Callable[..., object], *args: object) -> torch.fx.GraphModule:
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
        if isinstance(obj, torch.Tensor) and not isinstance(obj, TileIndexProxy):
            tracker = tracer.tensor_tracker
            if obj not in tracker:
                origin = HostFunction.current().tensor_to_origin[obj]
                assert origin.is_host()
                tracker[obj] = proxy = tracer.create_proxy(
                    "call_function",
                    _tracing_ops._host_tensor,
                    (origin.host_str(),),
                    {},
                    name=origin.suggest_var_name(),
                )
                proxy.node.meta["val"] = obj
            return transform(tracker[obj])
        if isinstance(obj, proxy_tensor.py_sym_types):
            tracker = tracer.symnode_tracker
            if obj not in tracker:
                debug_name = CompileEnvironment.current().sympy_debug(obj._sympy_())
                tracker[obj] = proxy = tracer.create_proxy(
                    "call_function",
                    _tracing_ops._get_symnode,
                    (debug_name,),
                    {},
                    name=debug_name if debug_name.isidentifier() else "symnode",
                )
                proxy.node.meta["val"] = obj
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
    ):
        current_location().set_fx_location()
        return proxy_tensor.make_fx(fn, decomposition_table=select_decomp_table())(
            *args
        )


@dataclasses.dataclass
class GraphInfo:
    graph_id: int
    # TODO(jansel): GraphModule -> Graph to avoid fx compile
    graph: torch.fx.GraphModule

    @property
    def name(self) -> str:
        raise NotImplementedError

    def kwargs(self) -> dict[str, object]:
        """Return a dictionary of keyword needed to copy this graph."""
        return {}

    def __str__(self) -> str:
        output = self.graph.print_readable(print_output=False).strip()
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
class ForLoopGraphInfo(GraphInfo):
    block_indices: list[int]

    @property
    def name(self) -> str:
        return f"for_loop_{self.graph_id}"

    def kwargs(self) -> dict[str, object]:
        return {
            "block_indices": [*self.block_indices],
        }

    def codegen(self, state: CodegenState) -> list[object]:
        args = state.ast_args[-1]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)
        with state.codegen.add_device_loop(
            state.device_function.tile_strategy.codegen_device_loop(
                state, self.block_indices
            )
        ):
            return codegen_call_with_graph(
                state.codegen,
                self.graph,
                args,
            )


@dataclasses.dataclass
class ReductionLoopGraphInfo(ForLoopGraphInfo):
    @property
    def name(self) -> str:
        return f"reduction_loop_{self.graph_id}"


class IfGraphInfo(GraphInfo):
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
    rolled_block_indices: list[int]
    original_graph_id: int
    new_graph_id: int | None
    used_rdim: bool
    can_be_rolled_by_caller: bool


class DeviceIR:
    def __init__(self) -> None:
        super().__init__()
        self.graphs: list[GraphInfo] = []
        self.root_id: int | None = None
        self.rolled_reductions: list[RolledReductionInfo] = []
        self.grid_block_indices: list[list[int]] = []

    def get_root(self, config: Config) -> torch.fx.GraphModule:
        """ " If we are using a rolled reduction, return the rolled reduction graph otherwise
        return the root graph."""
        if (root_id := self.root_id) is None:
            raise AssertionError("No root graph")
        reduction_loops = config.reduction_loops
        if len(reduction_loops) > 1:
            raise NotImplementedError("Multiple reduction loops not implemented")
        if len(reduction_loops) == 0 or reduction_loops[0] is None:
            return self.graphs[root_id].graph
        for info in reversed(self.rolled_reductions):
            if info.original_graph_id == root_id:
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
        graph: torch.fx.GraphModule,
        graph_info_cls: type[GraphInfo] = GraphInfo,
        **kwargs: object,
    ) -> int:
        graph.graph.eliminate_dead_code()
        graph_id = len(self.graphs)
        self.graphs.append(graph_info_cls(graph_id=graph_id, graph=graph, **kwargs))
        return graph_id

    def add_reduction_loop_graph(
        self, graph: torch.fx.GraphModule, block_index: int
    ) -> int:
        return self.add_graph(
            graph,
            graph_info_cls=ReductionLoopGraphInfo,
            block_indices=[block_index],
        )

    def add_root_graph(self, graph: torch.fx.GraphModule) -> None:
        assert self.root_id is None
        self.root_id = self.add_graph(graph, graph_info_cls=RootGraphInfo)

    def build_rolled_reductions(self) -> None:
        env = CompileEnvironment.current()
        rdims = [bs for bs in env.block_sizes if bs.reduction]
        if not rdims:
            return
        first = True
        for rdim in rdims:
            graph_to_info = {}
            allow_loop = False
            for graph_id, graph_info in enumerate([*self.graphs]):
                assert graph_id == graph_info.graph_id
                roller = ReductionRoller(self, rdim, graph_to_info)
                new_graph = torch.fx.GraphModule(
                    {}, roller.process(graph_info.graph.graph)
                )
                new_graph_id = self.add_graph(
                    new_graph, type(graph_info), **graph_info.kwargs()
                )
                reduction_info = RolledReductionInfo(
                    rolled_block_indices=[rdim.block_size_idx],
                    original_graph_id=graph_id,
                    new_graph_id=new_graph_id,
                    used_rdim=len(roller.graphs_added) > 0,
                    can_be_rolled_by_caller=roller.outer_count == 0
                    and len(roller.graphs_added) == 1,
                )
                allow_loop = allow_loop or reduction_info.used_rdim
                self.rolled_reductions.append(reduction_info)
                graph_to_info[graph_id] = reduction_info
            env.config_spec.reduction_loop_specs.append(
                ReductionLoopSpec(
                    size_hint=rdim.size_hint(),
                    # TODO(jansel): we should add support for rolling multiple dims at once
                    allow_loop=allow_loop and first,
                )
            )
            first = False


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
                # pyre-ignore[16]
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

    def _to_proxy(self, node: ast.AST) -> object:
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        if not type_info.contains_tensor():
            return type_info.proxy()
        return self.visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> object:
        return _eval_binary(node.op, self.visit(node.left), self.visit(node.right))

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
        if isinstance(value, (TileIndexProxy, torch.SymInt)):
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
        assert func_type.value in (hl.tile, hl.grid)
        args = call_node.args
        assert len(args) >= 1
        if len(args) == 1:
            begin = None
            end = self.visit(args[0])
        else:
            begin = self.visit(args[0])
            end = self.visit(args[1])
        return begin, end

    def visit_For(self, node: ast.For) -> None:
        assert isinstance(node, ExtendedAST)
        assert not node.orelse
        assert isinstance(node.iter, ExtendedAST)
        iter_type = node.iter._type_info
        assert isinstance(iter_type, IterType)
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
                        and (k not in self.scope or self.scope[k] is not v)
                    }
                )
                return outputs.get_tensor_args()

            with self.disable_tracing() as tracer:
                graph = proxy_tensor.make_fx(
                    run_subgraph, decomposition_table=select_decomp_table()
                )(*inputs.get_tensor_args())
                graph_idx = self.device_ir.add_graph(
                    graph,
                    ForLoopGraphInfo,
                    block_indices=[x.block_size_idx for x in iter_vars],
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
                if isinstance(value, TileIndexProxy):
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
                run_body, decomposition_table=select_decomp_table()
            )(*inputs.get_tensor_args())
            assert outputs is not None
            graph_idx = self.device_ir.add_graph(
                body_graph,
                IfGraphInfo,
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
        assert type_info.origin.is_host()
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

    def visit_Slice(self, node: ast.Slice) -> slice:
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
        return slice(lower, upper, step)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise exc.AssignmentMultipleTargets
        (target,) = node.targets
        if isinstance(target, ast.Name):
            # TODO(jansel): should assert that name is only used on device
            self._assign(target, self.visit(node.value))
            return None
        if not isinstance(target, ast.Subscript):
            raise exc.InvalidAssignment
        assert isinstance(node.value, ExtendedAST)
        rhs_type = node.value._type_info
        assert isinstance(target, ExtendedAST)
        lhs_type = target._type_info
        if not isinstance(lhs_type, TensorType) or not isinstance(rhs_type, TensorType):
            raise exc.NonTensorSubscriptAssign(lhs_type, rhs_type)
        assert isinstance(target.value, ExtendedAST)
        target_origin = target.value._type_info.origin
        assert target_origin.is_host()
        val = self.visit(node.value)
        return hl.store(
            self.visit(target.value), self._subscript_slice_proxy(target.slice), val
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
        if type_info.origin.is_host():
            return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))
        return hl.subscript(self.visit(value), self._subscript_slice_proxy(node.slice))

    def visit_Call(self, node: ast.Call) -> object:
        args = []
        kwargs = {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                # pyre-ignore[6]
                args.extend(self._to_proxy(arg.value))
            else:
                args.append(self._to_proxy(arg))
        for kwarg in node.keywords:
            if kwarg.arg is None:
                # pyre-ignore[6]
                kwargs.update(self._to_proxy(kwarg.value))
            else:
                kwargs[kwarg.arg] = self._to_proxy(kwarg.value)
        # pyre-ignore[6]
        return CheckForIndexCalls.retry_call(self.visit(node.func), args, kwargs)

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
            if isinstance(v, torch.Tensor) and not isinstance(v, TileIndexProxy)
        ]

    def unflatten(self) -> dict[str, object]:
        return pytree.tree_unflatten(self.flat_values, self.spec)

    def replace_tensor_args(self, args: Sequence[object]) -> dict[str, object]:
        flat_values = [*self.flat_values]
        assert len(self.tensor_indices) == len(args)
        for i, v in zip(self.tensor_indices, args, strict=False):
            flat_values[i] = v
        return pytree.tree_unflatten(flat_values, self.spec)

    def get_tensor_args(self) -> list[object]:
        return [self.flat_values[i] for i in self.tensor_indices]


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
            iter_type = node.iter._type_info
            assert isinstance(iter_type, IterType)
            inner = iter_type.inner
            if isinstance(inner, SequenceType):
                block_indices = [x.block_size_idx for x in inner.unpack()]
            else:
                block_indices = [inner.block_size_idx]
            self.device_ir.grid_block_indices.append(block_indices)
        else:
            self.generic_visit(node)


def lower_to_device_ir(func: HostFunction) -> DeviceIR:
    with func, compile_lock:
        device_ir = DeviceIR()
        visitor = WalkHostAST(device_ir)
        for stmt in func.body:
            visitor.visit(stmt)
        CompileEnvironment.current().errors.raise_if_errors()
        for graph in device_ir.graphs:
            prepare_graph_lowerings(graph.graph)
        device_ir.build_rolled_reductions()
        return device_ir
