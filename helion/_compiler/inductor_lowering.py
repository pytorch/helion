from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
from operator import getitem
from typing import TYPE_CHECKING
from typing import ContextManager
from typing import NamedTuple

import sympy
import torch
from torch._dynamo.convert_frame import compile_lock
from torch._inductor import config as inductor_config
from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import Pointwise
from torch._inductor.ir import Reduction
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import triton_type
from torch._inductor.virtualized import OpsValue
from torch._inductor.virtualized import V
from torch.fx.experimental import proxy_tensor
from torch.fx.experimental.sym_node import SymNode
from torch.fx.interpreter import Interpreter
from torch.fx.node import Node
from torch.fx.node import map_arg

from .._compat import min_dot_size
from ..exc import InductorLoweringError
from ..language._decorators import APIFunc
from ..language._decorators import is_api_func
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from torch.utils._ordered_set import OrderedSet

    from .. import Config
    from .device_function import DeviceFunction
    from .generate_ast import GenerateAST
    from .tile_dispatch import TileStrategyDispatch

    CodegenHandler = Callable[["GraphInterpreter", torch.fx.Node], object]


def prepare_graph_lowerings(gm: torch.fx.GraphModule) -> None:
    with compile_lock:
        graph_lowering = GraphLowering(
            gm, shape_env=CompileEnvironment.current().shape_env
        )
        # pyre-ignore[19]
        with V.set_graph_handler(graph_lowering):
            for node in gm.graph.nodes:
                assert node.op in {
                    "call_function",
                    "placeholder",
                    "output",
                }, node.op
                if node.op == "call_function":
                    with node.meta["location"]:
                        prepare_node_lowering(graph_lowering, node)


def prepare_node_lowering(
    graph_lowering: GraphLowering,
    node: Node,
) -> None:
    if is_api_func(api := node.target):
        APIFuncLowering.normalize_args_kwargs(api, node)
        node.meta["lowering"] = APIFuncLowering(api)
        return

    if node.target in aten_lowering_dispatch:
        node.meta["lowering"] = aten_lowering_dispatch[node.target](node)
        return

    if isinstance(
        val := node.meta["val"], (torch.SymInt, torch.SymFloat, torch.SymBool)
    ):
        node.meta["lowering"] = SympyExprLowering(val._sympy_())
        return

    def convert_arg(arg: Node) -> TensorBox:
        example = arg.meta["val"]
        input_names.append(name := f"{node.name}_input{len(input_names)}")
        if isinstance(example, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            dtype = {
                torch.SymInt: torch.int64,
                torch.SymFloat: torch.float32,
                torch.SymBool: torch.bool,
            }[type(example)]
            return TensorBox.create(
                InputBuffer(
                    name=name,
                    layout=FixedLayout(
                        CompileEnvironment.current().device,
                        dtype,
                        [],
                        [],
                    ),
                )
            )
        assert isinstance(example, torch.Tensor), (
            f"Expected Tensor, got {type(example)}: {node.target}"
        )
        return TensorBox.create(
            InputBuffer(
                name=name,
                layout=FixedLayout(
                    example.device,
                    example.dtype,
                    [*map(_unpack_symint, example.size())],
                    [*map(_unpack_symint, example.stride())],
                ),
            )
        )

    prior_buffers = len(graph_lowering.buffers)
    input_names: list[str] = []
    with torch._inductor.config.patch(split_reductions=False):
        result = graph_lowering.call_function(
            # pyre-ignore[6]
            node.target,
            # pyre-ignore[6]
            *map_arg((node.args, node.kwargs), convert_arg),
        )
        result.realize()
    if not isinstance(result, TensorBox) or not isinstance(result.data, StorageBox):
        raise InductorLoweringError(
            f"Lowering {node.target} returned type(result), expected TensorBox(StorageBox(...)): {result}"
        )
    if not isinstance(buffer := result.data.data, ComputedBuffer):
        raise InductorLoweringError(
            f"Lowering {node.target} returned buffer type {type(buffer)}, expected ComputedBuffer: {buffer}"
        )

    new_buffers = graph_lowering.buffers[prior_buffers:]
    assert new_buffers[-1] is buffer
    nodes = []
    extra_input_names = []
    new_node: torch.fx.Node
    for i, buffer in enumerate(new_buffers):
        if not isinstance(buffer, ComputedBuffer) or not isinstance(
            buffer.data, (Pointwise, Reduction)
        ):
            raise InductorLoweringError(
                f"Lowering {node.target} returned buffer type {type(buffer)}, expected ComputedBuffer(Pointwise|Reduction): {buffer}"
            )
        if i == len(new_buffers) - 1:
            new_node = node
            if nodes:
                new_node.kwargs = {**new_node.kwargs, "_extra_args": [*nodes]}
        else:
            new_node = create_extra_node(node, buffer, [*node._input_nodes, *nodes])
        lowering_cls = (
            PointwiseLowering
            if isinstance(buffer.data, Pointwise)
            else ReductionLowering
        )
        buffer.freeze_layout()
        used_input_names = strip_unused_inputs(
            new_node,
            buffer.get_read_names(),
            dict(
                zip(
                    node.all_input_nodes,
                    [*input_names, *extra_input_names],
                    strict=True,
                )
            ),
        )
        new_node.meta["lowering"] = lowering_cls(buffer, used_input_names)
        nodes.append(new_node)
        extra_input_names.append(buffer.get_name())


def strip_unused_inputs(
    node: torch.fx.Node,
    used_input_names: OrderedSet[str],
    input_names: dict[torch.fx.Node, str],
) -> list[str]:
    """
    Remove unused inputs from the node.  Inplace updates node.args and
    node.kwargs to replace unused inputs with None.

    :param node: Node to mutate args of
    :param used_input_names: Set of input names that are used in the node's lowering.
    :param input_names: Mapping of node inputs to their names.
    :return:  List of nodes that were used in the lowering.
    """

    def mask_unused_inputs(n: torch.fx.Node) -> torch.fx.Node | None:
        if (name := input_names[n]) in used_input_names and name not in seen_names:
            seen_names.setdefault(name)
            return n
        return None

    assert len(input_names) == len(node._input_nodes)
    seen_names: dict[str, None] = {}
    node.args = map_arg(node.args, mask_unused_inputs)
    node.kwargs = map_arg(node.kwargs, mask_unused_inputs)
    assert len(seen_names) == len(used_input_names)
    return [*seen_names]


def create_extra_node(
    original_node: torch.fx.Node,
    buffer: ComputedBuffer,
    input_nodes: list[torch.fx.Node],
) -> torch.fx.Node:
    """When inductor lowerings produce multiple buffers,
    we add extra nodes to maintain a 1:1 mapping between fx nodes and buffers."""
    from ..language._tracing_ops import _inductor_lowering_extra

    graph = original_node.graph
    with graph.inserting_before(original_node):
        node = graph.create_node(
            "call_function",
            _inductor_lowering_extra,
            (input_nodes,),
            {},
            name=f"{original_node.name}_extra",
        )
    with proxy_tensor.disable_proxy_modes_tracing():
        node.meta["val"] = torch.empty(
            [*map(to_symint, buffer.get_size())],
            dtype=buffer.get_dtype(),
            device=buffer.get_device(),
        )
    for key in ("stack_trace", "original_aten", "location"):
        node.meta[key] = original_node.meta.get(key, None)
    return node


def to_symint(x: object) -> torch.SymInt | int:
    if isinstance(x, (int, sympy.Integer)):
        return int(x)
    assert isinstance(x, sympy.Expr)
    return torch.SymInt(
        SymNode(x, CompileEnvironment.current().shape_env, int, hint=None)
    )


def _unpack_symint(x: torch.SymInt | int) -> sympy.Expr:
    if isinstance(x, torch.SymInt):
        return x._sympy_()
    if isinstance(x, int):
        return sympy.sympify(x)
    raise TypeError(f"Expected SymInt or int, got {type(x)}")


class Lowering:
    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        raise NotImplementedError


@dataclasses.dataclass
class InductorLowering(Lowering):
    buffer: ComputedBuffer
    input_names: list[str]

    def input_asts(self, ctx: GraphInterpreter, node: torch.fx.Node) -> list[ast.AST]:
        def visit(n: torch.fx.Node) -> None:
            ast_val = ctx.env[n]
            if isinstance(fake_val := n.meta["val"], torch.Tensor):
                if fake_val.ndim < ndim:
                    # Broadcast to force ranks to match
                    expand = ["None"] * (ndim - fake_val.ndim) + [":"] * fake_val.ndim
                    ast_val = expr_from_string(
                        "tensor[" + ", ".join(expand) + "]", tensor=ast_val
                    )
            input_asts.append(ast_val)

        ndim: int = max([x.ndim for x in self.input_fake_tensors(node)] or (0,))
        input_asts: list[ast.AST] = []
        map_arg((node.args, node.kwargs), visit)
        assert len(input_asts) == len(self.input_names)
        return input_asts

    @staticmethod
    def input_fake_tensors(node: torch.fx.Node) -> list[torch.Tensor]:
        def visit(n: torch.fx.Node) -> torch.fx.Node:
            if isinstance(val := n.meta["val"], torch.Tensor):
                result.append(val)
            return n

        result: list[torch.Tensor] = []
        map_arg((node.args, node.kwargs), visit)
        return result

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        raise NotImplementedError(
            f"codegen not implemented for {type(self).__name__}: {self.buffer}"
        )

    def install_kernel_handlers(
        self, ctx: GraphInterpreter, node: torch.fx.Node
    ) -> ContextManager[None]:
        return install_inductor_kernel_handlers(
            ctx.cg, dict(zip(self.input_names, self.input_asts(ctx, node), strict=True))
        )


@contextlib.contextmanager
def install_inductor_kernel_handlers(
    cg: GenerateAST, args: dict[str, ast.AST]
) -> Iterator[None]:
    with (
        inductor_config.patch(
            {
                "triton.codegen_upcast_to_fp32": False,
                "split_reductions": False,
            }
        ),
        # pyre-ignore[19]
        V.set_graph_handler(
            GraphLowering(dummy_gm(), shape_env=CompileEnvironment.current().shape_env)
        ),
        # pyre-ignore[19]
        V.set_ops_handler(
            GenerateASTFromInductor(
                cg,
                args,
            )
        ),
        # pyre-ignore[19]
        V.set_kernel_handler(
            TritonKernel({}, features=SIMDKernelFeatures([], sympy.S.One))
        ),
    ):
        yield


@functools.cache
def dummy_gm() -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(lambda: None)


class PointwiseLowering(InductorLowering):
    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        with self.install_kernel_handlers(ctx, node):
            indices = [
                sympy.Symbol(f"i{n}") for n in range(len(self.buffer.data.ranges))
            ]
            output_name = _unpack_opsvalue(self.buffer.data.inner_fn(indices))
            return expr_from_string(output_name)


@dataclasses.dataclass
class ReductionLowering(InductorLowering):
    def __init__(
        self,
        buffer: ComputedBuffer,
        input_names: list[str],
    ) -> None:
        super().__init__(buffer, input_names)
        reduction = self.buffer.data
        assert isinstance(reduction, Reduction)
        reduction_ranges = reduction.reduction_ranges
        if len(reduction_ranges) != 1:
            # TODO(jansel): can this happen?
            raise NotImplementedError("multiple reduction dimensions")
        reduction_var = reduction_ranges[0]
        assert isinstance(reduction_var, sympy.Symbol)

        block_index = TileStrategy.get_block_index(reduction_var)
        assert block_index is not None
        self.block_index: int = block_index

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        reduction = self.buffer.data
        assert isinstance(reduction, Reduction)
        indices = [sympy.Symbol(f"i{n}") for n in range(len(reduction.ranges))]
        reduction_indices = [
            sympy.Symbol(f"i{n}")
            for n in range(len(indices), len(indices) + len(reduction.reduction_ranges))
        ]
        with self.install_kernel_handlers(ctx, node):
            # codegen the pointwise part before reduction
            output_name = _unpack_opsvalue(
                self.buffer.data.inner_fn(indices, reduction_indices)
            )

        state = CodegenState(
            ctx.cg,
            fx_node=node,
        )
        if CompileEnvironment.current().block_sizes[self.block_index].reduction:
            strategy = ctx.cg.device_function.tile_strategy.get_reduction_strategy(
                self.block_index
            )
        else:
            from .reduction_strategy import BlockReductionStrategy

            strategy = BlockReductionStrategy(state, self.block_index)

        inputs = self.input_fake_tensors(node)
        if len(inputs) != 1:
            # TODO(jansel): combine multiple inputs into a single fake value
            raise NotImplementedError("reductions with >1 input")

        # TODO(jansel): find a better way to get dim
        (dim,) = [
            i
            for i, v in enumerate(inputs[0].shape)
            if TileStrategy.get_block_index(v) == self.block_index
        ]

        return strategy.codegen_reduction(
            state,
            output_name,
            reduction.reduction_type,
            dim,
            inputs[0],
            node.meta["val"],
        )


@dataclasses.dataclass
class APIFuncLowering(Lowering):
    api_func: APIFunc

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        assert not node.kwargs
        ast_args = [*map_arg(node.args, lambda arg: ctx.env[arg])]
        proxy_args = [*map_arg(node.args, lambda arg: arg.meta["val"])]

        assert self.api_func._codegen is not None
        return self.api_func._codegen(
            CodegenState(
                ctx.cg,
                fx_node=node,
                # pyre-ignore[6]
                proxy_args=proxy_args,
                # pyre-ignore[6]
                ast_args=ast_args,
            ),
        )

    @staticmethod
    def normalize_args_kwargs(
        api_func: APIFunc,
        node: torch.fx.Node,
    ) -> None:
        bound = api_func._signature.bind(*node.args, **node.kwargs)
        bound.apply_defaults()
        node.args = (*bound.arguments.values(),)
        node.kwargs = {}


@dataclasses.dataclass
class SympyExprLowering(Lowering):
    expr: sympy.Expr

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        return expr_from_string(ctx.cg.device_function.user_sympy_expr(self.expr))


@dataclasses.dataclass
class LambdaLowering(Lowering):
    fn: Callable[..., object]

    def codegen(self, ctx: GraphInterpreter, node: torch.fx.Node) -> object:
        return self.fn(ctx, node)


aten_lowering_dispatch: dict[object, Callable[[torch.fx.Node], Lowering]] = {}


def default_make_lowering(handler: CodegenHandler, node: torch.fx.Node) -> Lowering:
    return LambdaLowering(handler)


def register_lowering(
    fn: object,
    make_lowering: Callable[
        [CodegenHandler, torch.fx.Node], Lowering
    ] = default_make_lowering,
) -> Callable[[CodegenHandler], CodegenHandler]:
    def decorator(handler: CodegenHandler) -> CodegenHandler:
        assert fn not in aten_lowering_dispatch, f"Lowering for {fn} already registered"
        aten_lowering_dispatch[fn] = lambda node: make_lowering(handler, node)
        return handler

    return decorator


# pyre-fixme[56]
@register_lowering(torch.ops.aten.sym_size.int)
def codegen_sym_size(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    val = node.meta["val"]
    assert isinstance(
        val, (int, float, bool, torch.SymInt, torch.SymBool, torch.SymFloat)
    )
    return val


@register_lowering(getitem)
def codegen_getitem(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(lhs, (list, tuple))
    assert isinstance(rhs, int)
    return lhs[rhs]


# pyre-fixme[56]
@register_lowering(torch.ops.aten.full.default)
def codegen_full(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    env = CompileEnvironment.current()
    size, fill_value = map_arg(node.args, lambda n: n.meta["val"])
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    device = node.kwargs.get("device", env.device)
    assert device == env.device, f"expected {env.device}, got {device}"
    assert not node.kwargs.get("pin_memory"), "pin_memory not supported"
    assert isinstance(fill_value, (int, float, bool))
    # pyre-ignore[32]
    shape_str = ctx.cg.device_function.tile_strategy.shape_str([*size])
    return expr_from_string(
        f"tl.full({shape_str}, {constant_repr(fill_value)}, {triton_type(dtype)})"
    )


# pyre-fixme[56]
@register_lowering(torch.ops.aten.unsqueeze.default)
def codegen_unsqueeze(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dim = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    assert isinstance(dim, int)
    ndim = node.args[0].meta["val"].ndim
    if dim < 0:
        dim += ndim
    assert 0 <= dim <= ndim, f"Invalid dim {dim} for tensor with {ndim} dims"
    args = [":"] * ndim
    args.insert(dim, "None")
    return expr_from_string(
        f"tensor[{', '.join(args)}]",
        tensor=tensor,
    )


@register_lowering(torch.ops.aten.squeeze.dim)
@register_lowering(torch.ops.aten.view.default)
# pyre-fixme[56]
@register_lowering(torch.ops.aten.reshape.default)
def codegen_view(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "view kwargs not supported"
    tensor = map_arg(node.args[0], lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    return expr_from_string(f"tl.reshape(tensor, {shape_str})", tensor=tensor)


# pyre-fixme[56]
@register_lowering(torch.ops.aten.permute.default)
def codegen_permute(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dims = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    dims = [*dims]
    assert {*dims} == {*range(len(dims))}, dims
    return expr_from_string(
        f"tl.permute(tensor, {dims!r})",
        tensor=tensor,
    )


# pyre-fixme[56]
@register_lowering(torch.ops.aten.expand.default)
def codegen_expand(ctx: GraphInterpreter, node: torch.fx.Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, _ = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    if node.args[0].meta["val"].ndim != len(shape):
        broadcasting = [":"] * len(shape)
        for i in range(len(shape) - node.args[0].meta["val"].ndim):
            broadcasting[i] = "None"
        tensor = expr_from_string(f"tensor[{', '.join(broadcasting)}]", tensor=tensor)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"tl.broadcast_to(tensor, {shape_str})",
        tensor=tensor,
    )


def apply_dot_requirements(handler: CodegenHandler, node: torch.fx.Node) -> Lowering:
    """Apply min_dot_size requirements to the config_spec"""
    assert not node.kwargs, "dot kwargs not supported"
    assert len(node.args) in (2, 3)
    lproxy, rproxy = map_arg(node.args[-2:], lambda arg: arg.meta["val"])
    assert isinstance(lproxy, torch.Tensor)
    assert isinstance(rproxy, torch.Tensor)
    lshape = lproxy.size()
    rshape = rproxy.size()
    # use last two dimensions for dot (supports 2D and batched 3D tensors)
    n, k = lshape[-2], lshape[-1]
    k2, m = rshape[-2], rshape[-1]
    assert k == k2, f"Mismatched k dimensions for dot: {k} vs {k2}"
    a, b, c = min_dot_size(lproxy.device, lproxy.dtype, rproxy.dtype)
    env = CompileEnvironment.current()
    for shape, min_size in [(n, a), (k, b), (m, c)]:
        block_idx = TileStrategy.get_block_index(shape)
        if block_idx is not None:
            env.block_sizes[block_idx].update_min_block(min_size, allow_flattened=True)
    return LambdaLowering(handler)


@register_lowering(torch.ops.aten.bmm.default, apply_dot_requirements)
# pyre-fixme[56]
@register_lowering(torch.ops.aten.mm.default, apply_dot_requirements)
def codegen_mm(ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)
    tf32 = CompileEnvironment.current().settings.dot_precision
    return expr_from_string(
        f"tl.dot(lhs, rhs, input_precision={tf32!r})", lhs=lhs, rhs=rhs
    )


# pyre-fixme[56]
@register_lowering(torch.ops.aten.addmm.default, apply_dot_requirements)
def codegen_addmm(ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    acc, lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)
    tf32 = CompileEnvironment.current().settings.dot_precision
    return expr_from_string(
        f"tl.dot(lhs, rhs, acc=acc, input_precision={tf32!r})",
        lhs=lhs,
        rhs=rhs,
        acc=acc,
    )


# pyre-fixme[56]
@register_lowering(torch.ops.aten.baddbmm.default, apply_dot_requirements)
def codegen_baddbmm(ctx: GraphInterpreter, node: torch.fx.Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    acc, lhs, rhs = map_arg(node.args, lambda arg: ctx.env[arg])
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)
    tf32 = CompileEnvironment.current().settings.dot_precision
    return expr_from_string(
        f"tl.dot(lhs, rhs, acc=acc, input_precision={tf32!r})",
        lhs=lhs,
        rhs=rhs,
        acc=acc,
    )


class GenerateASTFromInductor(DefaultHandler):
    def __init__(self, cg: GenerateAST, input_name_lookup: dict[str, ast.AST]) -> None:
        super().__init__()
        self.parent_handler = TritonOverrides()
        self.cg = cg
        self.input_name_lookup = input_name_lookup

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> str:
        result_str = _unpack_opsvalue(
            getattr(self.parent_handler, name)(*args, **kwargs)
        )
        return self.cg.lift(expr_from_string(result_str)).id

    def load(self, name: str, index: sympy.Expr) -> str:
        # TODO(jansel): assert the index is correct
        return self.cg.lift(self.input_name_lookup[name]).id

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> str:
        name = self.cg.lift(
            expr_from_string(self.cg.device_function.user_sympy_expr(expr))
        ).id
        return f"{name}.to({triton_type(dtype)})"


def _unpack_opsvalue(value: object) -> str:
    if isinstance(value, OpsValue):
        return str(value)
    assert isinstance(value, str)
    return value


class GraphInterpreter(Interpreter):
    def __init__(self, gm: torch.fx.GraphModule, cg: GenerateAST) -> None:
        super().__init__(gm, garbage_collect_values=False)
        self.cg = cg

    def run_node(self, n: Node) -> object:
        if n.op == "call_function":
            with self._set_current_node(n), n.meta["location"]:
                lowering: Lowering = n.meta["lowering"]
                result = lowering.codegen(self, n)
                if result is None:
                    return None
                if not isinstance(result, ast.AST):
                    return result
                assert isinstance(result, ast.expr)
                if len(n.users) > 0:
                    if isinstance(result, (ast.Name, ast.Constant)):
                        return result
                    name = self.cg.device_function.new_var(n.name)
                    self.cg.add_statement(
                        statement_from_string(f"{name} = result", result=result)
                    )
                    return create(ast.Name, id=name, ctx=ast.Load())
                if not isinstance(result, (ast.Name, ast.Constant)):
                    self.cg.add_statement(create(ast.Expr, value=result))
                return None
        return super().run_node(n)


def codegen_call_with_graph(
    cg: GenerateAST, gm: torch.fx.GraphModule, args: list[ast.AST]
) -> list[object]:
    with compile_lock:
        new_args = []
        placeholders = gm.graph.find_nodes(op="placeholder")
        for arg, placeholder in zip(args, placeholders, strict=True):
            if all(
                user.target == torch.ops.aten.sym_size.int for user in placeholder.users
            ):
                # TODO(jansel): we should remove these sym_size-only args from the graph
                new_args.append(arg)
            elif isinstance(arg, ast.Name):
                # We need to copy the inputs to a loop so that phi nodes are handled properly.
                # Phi nodes will merge variable names from outside the loop, but the old value
                # of those variables could have usages.
                copy_name = cg.device_function.new_var(arg.id + "_copy")
                cg.add_statement(statement_from_string(f"{copy_name} = arg", arg=arg))
                new_args.append(expr_from_string(copy_name))
            else:
                new_args.append(cg.lift(arg))
        return GraphInterpreter(gm, cg).run(*new_args)


class CodegenState(NamedTuple):
    codegen: GenerateAST
    fx_node: torch.fx.Node | None
    proxy_args: list[object] = dataclasses.field(default_factory=list)
    ast_args: list[object] = dataclasses.field(default_factory=list)

    def proxy_arg(self, i: int) -> object:
        return self.proxy_args[i]

    def ast_arg(self, i: int) -> ast.AST:
        rv = self.ast_args[i]
        assert isinstance(rv, ast.AST), "TODO: convert nested/defaults"
        return rv

    @property
    def fake_value(self) -> object:
        assert self.fx_node is not None
        return self.fx_node.meta["val"]

    @property
    def device_function(self) -> DeviceFunction:
        return self.codegen.device_function

    @property
    def tile_strategy(self) -> TileStrategyDispatch:
        return self.codegen.device_function.tile_strategy

    @property
    def config(self) -> Config:
        return self.codegen.device_function.config

    def add_statement(self, statement: ast.AST | str) -> None:
        return self.codegen.add_statement(statement)
