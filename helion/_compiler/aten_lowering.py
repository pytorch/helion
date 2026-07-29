from __future__ import annotations

import ast
from collections.abc import Callable
import dataclasses
from operator import getitem
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx.node import Argument
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc
from ..language.matmul_ops import enforce_dot_requirements
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .node_masking import apply_masking
from .node_masking import cached_masked_value
from .node_masking import getitem_masked_value

if TYPE_CHECKING:
    from .helper_function import CodegenInterface


class LoweringContext:
    cg: CodegenInterface
    env: dict[Node, Argument]

    def to_ast(self, value: object) -> ast.AST:
        raise NotImplementedError


class Lowering:
    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        raise NotImplementedError

    def get_masked_value(self, node: Node) -> float | bool | None:
        """Get the masked value for this node."""
        return None


MaskedValueFn = Callable[[Node], float | bool | None]


CodegenHandler = Callable[[LoweringContext, Node], object]


def _env_arg(ctx: LoweringContext, node: Node) -> Argument:
    return ctx.env[node]


@dataclasses.dataclass
class AtenLowering(Lowering):
    target: object | None = None
    masked_value_fn: MaskedValueFn | None = None
    codegen_impls: dict[str, CodegenHandler] = dataclasses.field(default_factory=dict)

    def register_codegen(
        self, backend: str
    ) -> Callable[[CodegenHandler], CodegenHandler]:
        def decorator(handler: CodegenHandler) -> CodegenHandler:
            assert backend not in self.codegen_impls, (
                f"codegen already registered for backend {backend!r}"
            )
            self.codegen_impls[backend] = handler
            return handler

        return decorator

    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        env = CompileEnvironment.current()
        handler = self.codegen_impls.get(env.codegen_name)
        if handler is None:
            handler = self.codegen_impls.get("common")
        if handler is None:  # pragma: no cover - defensive
            target = self.target or "unknown"
            raise exc.BackendImplementationMissing(
                env.backend_name,
                f"Aten lowering codegen not registered for {target!r}",
            )
        return handler(ctx, node)

    def get_masked_value(self, node: Node) -> float | bool | None:
        if self.masked_value_fn is not None:
            return self.masked_value_fn(node)
        return None


def passthrough_masked_value(node: Node) -> float | bool | None:
    for input_node in node.all_input_nodes:
        if isinstance(input_node.meta["val"], torch.Tensor):
            return cached_masked_value(input_node)
    return None


aten_lowering_dispatch: dict[object, Callable[[Node], Lowering]] = {}


def default_make_lowering(lowering: AtenLowering, node: Node) -> Lowering:
    return lowering


def register_lowering(
    fn: object,
    make_lowering: Callable[[AtenLowering, Node], Lowering] = default_make_lowering,
    masked_value_fn: MaskedValueFn | None = None,
) -> AtenLowering:
    assert fn not in aten_lowering_dispatch, f"Lowering for {fn} already registered"
    lowering = AtenLowering(target=fn, masked_value_fn=masked_value_fn)
    aten_lowering_dispatch[fn] = lambda node: make_lowering(lowering, node)
    return lowering


sym_size_lowering = register_lowering(torch.ops.aten.sym_size.int)


@sym_size_lowering.register_codegen("common")
def codegen_sym_size(ctx: LoweringContext, node: Node) -> object:
    val = node.meta["val"]
    assert isinstance(
        val, (int, float, bool, torch.SymInt, torch.SymBool, torch.SymFloat)
    )
    return val


getitem_lowering = register_lowering(getitem, masked_value_fn=getitem_masked_value)


@getitem_lowering.register_codegen("common")
def codegen_getitem(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(lhs, (list, tuple))
    assert isinstance(rhs, int)
    return lhs[rhs]


full_lowering = register_lowering(
    torch.ops.aten.full.default,
    masked_value_fn=lambda n: (
        n.args[1] if isinstance(n.args[1], (int, float, bool)) else None
    ),
)


scalar_tensor_lowering = register_lowering(
    torch.ops.aten.scalar_tensor.default,
)


where_lowering = register_lowering(torch.ops.aten.where.self)


@where_lowering.register_codegen("common")
def codegen_where(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    cond, x, y = map_arg(node.args, lambda arg: _env_arg(ctx, arg))

    def ensure_ast(value: object) -> ast.AST:
        if isinstance(value, ast.AST):
            return value
        if isinstance(value, (int, float, bool)):
            return expr_from_string(constant_repr(value))
        raise AssertionError(f"unsupported where operand: {type(value)!r}")

    return expr_from_string(
        env.backend.where_expr("{cond}", "{x}", "{y}"),
        cond=ensure_ast(cond),
        x=ensure_ast(x),
        y=ensure_ast(y),
    )


@full_lowering.register_codegen("common")
def codegen_full(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    size = map_arg(node.args[0], lambda n: n.meta["val"])
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    device = node.kwargs.get("device", env.device)
    assert device == env.device, f"expected {env.device}, got {device}"
    assert not node.kwargs.get("pin_memory"), "pin_memory not supported"
    value_ast = map_arg(node.args[1], lambda arg: _env_arg(ctx, arg))
    if isinstance(value_ast, (int, float, bool)):
        value_ast = expr_from_string(constant_repr(value_ast))
    assert isinstance(value_ast, ast.AST), value_ast
    # pyrefly: ignore [not-iterable]
    shape_dims = ctx.cg.device_function.tile_strategy.shape_dims([*size])
    return expr_from_string(
        env.backend.full_expr(shape_dims, "{value}", dtype),
        value=value_ast,
    )


@scalar_tensor_lowering.register_codegen("common")
def codegen_scalar_tensor(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    dtype = node.kwargs.get("dtype", torch.get_default_dtype())
    assert isinstance(dtype, torch.dtype)
    device = node.kwargs.get("device", env.device)
    assert device == env.device, f"expected {env.device}, got {device}"
    layout = node.kwargs.get("layout", torch.strided)
    assert layout in (None, torch.strided), f"layout={layout}"
    assert not node.kwargs.get("pin_memory"), "pin_memory not supported"
    value_arg = node.args[0]
    value_ast = _env_arg(ctx, value_arg) if isinstance(value_arg, Node) else value_arg
    if isinstance(value_ast, (int, float, bool)):
        value_ast = expr_from_string(constant_repr(value_ast))
    assert isinstance(value_ast, ast.AST), value_ast
    return expr_from_string(
        env.backend.full_expr([], "{value}", dtype),
        value=value_ast,
    )


unsqueeze_lowering = register_lowering(
    torch.ops.aten.unsqueeze.default,
    masked_value_fn=passthrough_masked_value,
)


@unsqueeze_lowering.register_codegen("common")
def codegen_unsqueeze(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dim = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    assert isinstance(dim, int)
    # pyrefly: ignore [missing-attribute]
    ndim = node.args[0].meta["val"].ndim
    if dim < 0:
        dim += ndim + 1
    assert 0 <= dim <= ndim, f"Invalid dim {dim} for tensor with {ndim} dims"
    args = [":"] * ndim
    args.insert(dim, "None")
    return expr_from_string(
        f"{{tensor}}[{', '.join(args)}]",
        tensor=tensor,
    )


squeeze_lowering = register_lowering(
    torch.ops.aten.squeeze.dim,
    masked_value_fn=passthrough_masked_value,
)


view_lowering = register_lowering(
    torch.ops.aten.view.default,
    masked_value_fn=passthrough_masked_value,
)


reshape_lowering = register_lowering(
    torch.ops.aten.reshape.default,
    masked_value_fn=passthrough_masked_value,
)


argmax_lowering = register_lowering(torch.ops.aten.argmax.default)


argmin_lowering = register_lowering(torch.ops.aten.argmin.default)


constant_pad_nd_lowering = register_lowering(torch.ops.aten.constant_pad_nd.default)


def _argreduce_schema(node: Node) -> tuple[torch.Tensor, int | None, bool]:
    input_node = cast("Node", node.args[0])
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dim is None:
        keepdim = (
            bool(node.args[2])
            if len(node.args) > 2
            else bool(node.kwargs.get("keepdim", False))
        )
        return input_val, None, keepdim
    if not isinstance(dim, int):
        raise exc.BackendUnsupported(
            CompileEnvironment.current().backend_name,
            f"{node.target} with a non-integer dim",
        )
    if dim < 0:
        dim += input_val.ndim
    if not (0 <= dim < input_val.ndim):
        raise exc.ReductionDimInvalidForShape(dim, input_val.shape)
    keepdim = (
        bool(node.args[2])
        if len(node.args) > 2
        else bool(node.kwargs.get("keepdim", False))
    )
    return input_val, dim, keepdim


def _normalize_argreduce_dim(node: Node) -> tuple[torch.Tensor, int]:
    input_val, dim, _ = _argreduce_schema(node)
    if dim is None:
        raise exc.BackendUnsupported(
            CompileEnvironment.current().backend_name,
            f"{node.target} without an explicit integer dim",
        )
    return input_val, dim


def _should_use_cute_argreduce_lowering(argreduce_node: Node) -> bool:
    from ..language import _tracing_ops
    from ..language._decorators import is_api_func
    from .device_ir import DeviceIR

    if CompileEnvironment.current().backend_name != "cute":
        return False
    if not argreduce_node.args or not isinstance(argreduce_node.args[0], Node):
        return False

    matmul_targets = {
        torch.matmul,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
    try:
        device_ir = DeviceIR.current()
        graph_by_id = {
            idx: graph_info
            for idx, graph_info in enumerate(getattr(device_ir, "graphs", ()))
            if hasattr(graph_info, "graph")
        }
    except (AttributeError, IndexError):
        graph_by_id = {}
    seen_graph_ids: set[int] = set()
    seen_nodes: set[Node] = set()

    def graph_contains_matmul(graph_id: int) -> bool:
        if graph_id in seen_graph_ids:
            return False
        seen_graph_ids.add(graph_id)
        graph_info = graph_by_id.get(graph_id)
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            return False
        return any(node_contains_matmul(node) for node in graph.nodes)

    def node_contains_matmul(node: Node) -> bool:
        if node in seen_nodes:
            return False
        seen_nodes.add(node)
        if node.op != "call_function":
            return False
        if node.target in matmul_targets:
            return True
        if is_api_func(node.target):
            name = getattr(node.target, "__name__", "")
            if name == "dot":
                return True
            if _tracing_ops.is_for_loop_target(node.target):
                graph_id = node.args[0] if node.args else None
                if isinstance(graph_id, int) and graph_contains_matmul(graph_id):
                    return True
        for arg in node.args:
            if isinstance(arg, Node) and node_contains_matmul(arg):
                return True
        for arg in node.kwargs.values():
            if isinstance(arg, Node) and node_contains_matmul(arg):
                return True
        return False

    return node_contains_matmul(argreduce_node.args[0])


def _triton_argreduce(ctx: LoweringContext, node: Node, reduction_type: str) -> ast.AST:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    input_val, dim, keepdim = _argreduce_schema(node)
    assert isinstance(input_val, torch.Tensor)
    fn = "argmax" if reduction_type == "argmax" else "argmin"
    backend = CompileEnvironment.current().backend
    dtype_str = backend.dtype_str(node.meta["val"].dtype)
    if dim is None:
        flat_shape = ctx.cg.device_function.tile_strategy.shape_str([input_val.numel()])
        tensor = expr_from_string(
            backend.reshape_expr("{tensor}", flat_shape), tensor=tensor
        )
        reduced = f"tl.{fn}({{tensor}}, axis=0).to({dtype_str})"
    else:
        reduced = f"tl.{fn}({{tensor}}, axis={dim}).to({dtype_str})"
    if keepdim:
        output_val = node.meta["val"]
        assert isinstance(output_val, torch.Tensor)
        shape_dims = ctx.cg.device_function.tile_strategy.shape_dims(
            [*output_val.size()]
        )
        output_shape = ctx.cg.device_function.tile_strategy.shape_str(
            [*output_val.size()]
        )
        if output_val.numel() == 1:
            reduced = backend.full_expr(shape_dims, reduced, output_val.dtype)
        else:
            reduced = backend.reshape_expr(reduced, output_shape)
    return expr_from_string(reduced, tensor=tensor)


def _pallas_argreduce(ctx: LoweringContext, node: Node, reduction_type: str) -> ast.AST:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    input_val, dim, keepdim = _argreduce_schema(node)
    assert isinstance(input_val, torch.Tensor)
    fn = "argmax" if reduction_type == "argmax" else "argmin"
    backend = CompileEnvironment.current().backend
    dtype_str = backend.dtype_str(node.meta["val"].dtype)
    if dim is None:
        flat_shape = ctx.cg.device_function.tile_strategy.shape_str([input_val.numel()])
        tensor = expr_from_string(
            backend.reshape_expr("{tensor}", flat_shape), tensor=tensor
        )
        reduced = f"{dtype_str}(jnp.{fn}({{tensor}}, axis=0))"
    else:
        reduced = f"{dtype_str}(jnp.{fn}({{tensor}}, axis={dim}))"
    if keepdim:
        output_val = node.meta["val"]
        assert isinstance(output_val, torch.Tensor)
        shape_dims = ctx.cg.device_function.tile_strategy.shape_dims(
            [*output_val.size()]
        )
        output_shape = ctx.cg.device_function.tile_strategy.shape_str(
            [*output_val.size()]
        )
        if output_val.numel() == 1:
            reduced = backend.full_expr(shape_dims, reduced, output_val.dtype)
        else:
            reduced = backend.reshape_expr(reduced, output_shape)
    return expr_from_string(reduced, tensor=tensor)


view_dtype_lowering = register_lowering(
    torch.ops.aten.view.dtype,
    masked_value_fn=passthrough_masked_value,
)


alias_lowering = register_lowering(
    torch.ops.aten.alias.default,
    masked_value_fn=passthrough_masked_value,
)


@alias_lowering.register_codegen("common")
def codegen_alias(ctx: LoweringContext, node: Node) -> object:
    """Alias is a no-op view, just pass through the input tensor."""
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    return tensor


permute_lowering = register_lowering(
    torch.ops.aten.permute.default,
    masked_value_fn=passthrough_masked_value,
)


stack_lowering = register_lowering(
    torch.ops.aten.stack.default,
    masked_value_fn=passthrough_masked_value,
)


expand_lowering = register_lowering(
    torch.ops.aten.expand.default,
    masked_value_fn=passthrough_masked_value,
)


def apply_dot_requirements(lowering: AtenLowering, node: Node) -> Lowering:
    """Apply min_dot_size requirements to the config_spec"""
    assert not node.kwargs, "dot kwargs not supported"
    assert len(node.args) in (2, 3)
    lproxy, rproxy = map_arg(node.args[-2:], lambda arg: arg.meta["val"])
    assert isinstance(lproxy, torch.Tensor)
    assert isinstance(rproxy, torch.Tensor)
    # Update config spec min sizes for M, N, K
    enforce_dot_requirements(lproxy, rproxy)
    # inputs to the dot operation must be zero-masked
    *maybe_acc, lnode, rnode = node.args
    assert isinstance(lnode, Node)
    assert isinstance(rnode, Node)
    lnode = apply_masking(lnode, base_node=node, other=0)
    rnode = apply_masking(rnode, base_node=node, other=0)
    node.args = (*maybe_acc, lnode, rnode)
    return lowering


bmm_lowering = register_lowering(
    torch.ops.aten.bmm.default,
    apply_dot_requirements,
)


mm_lowering = register_lowering(
    torch.ops.aten.mm.default,
    apply_dot_requirements,
)


def _apply_bmm_dot_dtype_requirements(_lowering: AtenLowering, node: Node) -> Lowering:
    """Strip bmm.dtype's ScalarType arg and reuse the ordinary bmm lowering."""
    node.args = tuple(a for a in node.args if isinstance(a, Node))
    return apply_dot_requirements(bmm_lowering, node)


register_lowering(
    torch.ops.aten.bmm.dtype,
    _apply_bmm_dot_dtype_requirements,
)


addmm_lowering = register_lowering(
    torch.ops.aten.addmm.default,
    apply_dot_requirements,
)


baddbmm_lowering = register_lowering(
    torch.ops.aten.baddbmm.default,
    apply_dot_requirements,
)


iota_lowering = register_lowering(torch.ops.prims.iota.default)


arange_default_lowering = register_lowering(torch.ops.aten.arange.default)


def _node_dtype_kwarg(node: Node) -> torch.dtype | None:
    dtype = node.kwargs.get("dtype")
    return dtype if isinstance(dtype, torch.dtype) else None


sort_lowering = register_lowering(torch.ops.aten.sort.default)


gather_lowering = register_lowering(
    torch.ops.aten.gather.default,
    masked_value_fn=passthrough_masked_value,
)


topk_lowering = register_lowering(torch.ops.aten.topk.default)
