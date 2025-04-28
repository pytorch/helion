from __future__ import annotations

import ast
import dataclasses
import re
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental import proxy_tensor
from torch.fx.traceback import preserve_node_meta
from torch.utils import _pytree as pytree

from .. import exc
from .. import language as hl
from ..language._decorators import args_to_proxies
from ..language._tracing_ops import _for_loop
from ..language._tracing_ops import _get_symnode
from ..language._tracing_ops import _host_tensor
from ..language._tracing_ops import _phi
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
from .source_location import current_location
from .tile_index_proxy import TileIndexProxy
from .type_propagation import IterType
from .type_propagation import SequenceType
from .type_propagation import TensorType
from .type_propagation import TileIndexType
from .type_propagation import TypeInfo
from .type_propagation import _eval_binary
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
                    _host_tensor,
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
                    _get_symnode,
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
    graph: torch.fx.GraphModule

    @property
    def name(self) -> str:
        return "device_ir" if self.graph_id == -1 else f"subgraph_{self.graph_id}"

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
    pass


@dataclasses.dataclass
class ForLoopGraphInfo(GraphInfo):
    block_indices: list[int]

    def codegen(self, state: CodegenState) -> list[object]:
        for_node, statements = state.device_function.tile_strategy.codegen_device_loop(
            state, self.block_indices
        )
        args = state.ast_args[1]
        assert isinstance(args, list)
        assert all(isinstance(x, ast.AST) for x in args)
        with state.codegen.set_statements(statements):
            output = codegen_call_with_graph(
                state.codegen,
                self.graph,
                args,
            )
        state.add_statement(for_node)
        return output


class DeviceIR:
    def __init__(self) -> None:
        super().__init__()
        self.graphs: list[GraphInfo] = []
        self.root: torch.fx.GraphModule | None = None

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
        graph_id = len(self.graphs)
        self.graphs.append(graph_info_cls(graph_id=graph_id, graph=graph, **kwargs))
        return graph_id

    def add_root_graph(self, graph: torch.fx.GraphModule) -> None:
        assert self.root is None
        self.root = graph
        self.graphs.append(RootGraphInfo(graph_id=-1, graph=graph))


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
                {k: self.scope[k] for k in rw if k in self.scope}
            )
            outputs: LiftTensorArgs | None = None

            def run_subgraph(*args: object) -> list[object]:
                nonlocal outputs
                subgraph_walker = WalkDeviceAST(self.device_ir)
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

            mode = proxy_tensor.get_proxy_mode()
            assert isinstance(mode, proxy_tensor.ProxyTorchDispatchMode)
            tracer = mode.tracer
            with proxy_tensor.disable_proxy_modes_tracing():
                graph = proxy_tensor.make_fx(
                    run_subgraph, decomposition_table=select_decomp_table()
                )(*inputs.get_tensor_args())
                if isinstance(inner_type, SequenceType):
                    iter_vars = inner_type.unpack()
                else:
                    iter_vars = [inner_type]
                assert all(isinstance(x, TileIndexType) for x in iter_vars)
                graph_idx = self.device_ir.add_graph(
                    graph,
                    ForLoopGraphInfo,
                    block_indices=[x.block_size_idx for x in iter_vars],
                )
                args = (
                    graph_idx,
                    inputs.get_tensor_args(),
                )
                proxy_out = tracer.create_proxy(
                    "call_function",
                    _for_loop,
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
                if name in self.scope:
                    try:
                        self.scope[name] = _phi(self.scope[name], value)
                    except Exception as e:
                        raise exc.CantCombineTypesInControlFlow(
                            name, self.scope[name], value
                        ) from e
                else:
                    self.scope[name] = value
        else:
            raise AssertionError(f"Unexpected loop type {node._loop_type}")

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
        key = slice_node._type_info
        if isinstance(key, SequenceType):
            keys = key.unpack()
        else:
            keys = [key]
        try:
            return [x.proxy() for x in keys]
        except TypeError:
            raise exc.InvalidSliceType(slice_node._type_info) from None

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
        # pyre-ignore[29]
        return self.visit(node.func)(*args, **kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> object:
        assert isinstance(node, ExtendedAST)
        type_info = node._type_info
        if not type_info.contains_tensor() or type_info.origin.is_host():
            try:
                return type_info.proxy()
            except NotImplementedError:
                raise exc.CantReadOnDevice(type_info) from None
        return getattr(self.visit(node.value), node.attr)

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
        return device_ir
