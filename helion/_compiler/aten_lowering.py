from __future__ import annotations

import ast
from collections.abc import Callable
import contextlib
import dataclasses
from operator import getitem
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.utils import triton_type
from torch.fx.node import Argument
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc
from .._utils import next_power_of_2
from ..language.matmul_ops import enforce_dot_requirements
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cute.argreduce import codegen_cute_tile_argreduce
from .cute.indexing import CutePackedAffineLoad
from .cute.indexing import CuteShapeChainView
from .cute.indexing import is_cute_shape_chain_target
from .cute.indexing import match_cute_affine_range_iota
from .cute.matmul_fallback import _emit_cute_matmul
from .cute.matmul_utils import cute_lower_rhs_for_matmul
from .cute.matmul_utils import cute_outer_accumulates_result
from .cute.matmul_utils import cute_outer_accumulator_dtype
from .cute.matmul_utils import cute_outer_accumulator_out_dtype
from .cute.matmul_utils import cute_resolve_active_block_id
from .cute.matmul_utils import cute_resolve_active_matmul_k_block_id
from .cute.matmul_utils import cute_static_k_invariant_extent
from .matmul_utils import _emit_pallas_matmul
from .matmul_utils import _needs_f32_accumulator
from .matmul_utils import emit_tl_dot_with_padding
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


@unsqueeze_lowering.register_codegen("cute")
def codegen_unsqueeze_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import resolve_cute_shape_chain_value

    # One scalar per thread — adding a unit dimension cannot change the value.
    assert not node.kwargs, "unsqueeze kwargs not supported"
    tensor = _env_arg(ctx, cast("Node", node.args[0]))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, tensor.node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


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


def _shape_chain_only_users(node: Node) -> bool:
    return bool(node.users) and all(
        user.op == "call_function" and is_cute_shape_chain_target(user.target)
        for user in node.users
    )


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


def _cute_argreduce(ctx: LoweringContext, node: Node, reduction_type: str) -> ast.AST:
    _, dim, keepdim = _argreduce_schema(node)
    return codegen_cute_tile_argreduce(
        ctx,
        node,
        reduction_type,
        dim=dim,
        keepdim=keepdim,
    )


@argmax_lowering.register_codegen("triton")
def codegen_argmax(ctx: LoweringContext, node: Node) -> ast.AST:
    return _triton_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("triton")
def codegen_argmin(ctx: LoweringContext, node: Node) -> ast.AST:
    return _triton_argreduce(ctx, node, "argmin")


@argmax_lowering.register_codegen("pallas")
def codegen_argmax_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("pallas")
def codegen_argmin_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_argreduce(ctx, node, "argmin")


@argmax_lowering.register_codegen("cute")
def codegen_argmax_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    return _cute_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("cute")
def codegen_argmin_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    return _cute_argreduce(ctx, node, "argmin")


@squeeze_lowering.register_codegen("cute")
def codegen_squeeze_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import resolve_cute_shape_chain_value

    # Squeeze removes a dimension of size 1 — no data movement needed
    # since each thread still holds the same element.
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, tensor.node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


@view_lowering.register_codegen("cute")
@reshape_lowering.register_codegen("cute")
def codegen_view_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import codegen_cute_reshape

    return codegen_cute_reshape(ctx, node)


@squeeze_lowering.register_codegen("triton")
@view_lowering.register_codegen("triton")
@reshape_lowering.register_codegen("triton")
def codegen_view(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "view kwargs not supported"
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    return expr_from_string(f"tl.reshape({{tensor}}, {shape_str})", tensor=tensor)


@squeeze_lowering.register_codegen("pallas")
@view_lowering.register_codegen("pallas")
@reshape_lowering.register_codegen("pallas")
def codegen_view_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    return expr_from_string(f"jnp.reshape({{tensor}}, {shape_str})", tensor=tensor)


view_dtype_lowering = register_lowering(
    torch.ops.aten.view.dtype,
    masked_value_fn=passthrough_masked_value,
)


@view_dtype_lowering.register_codegen("triton")
def codegen_view_dtype(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.cast with bitcast=True for dtype reinterpretation."""
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    target_dtype = node.args[1]
    assert isinstance(target_dtype, torch.dtype)
    return expr_from_string(
        f"tl.cast({{tensor}}, {triton_type(target_dtype)}, bitcast=True)",
        tensor=tensor,
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


@permute_lowering.register_codegen("cute")
def codegen_permute_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import codegen_cute_permute

    return codegen_cute_permute(ctx, node)


@permute_lowering.register_codegen("triton")
def codegen_permute(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    # pyrefly: ignore [not-iterable]
    dims = [*dims]
    assert {*dims} == {*range(len(dims))}, dims
    return expr_from_string(
        f"tl.permute({{tensor}}, {dims!r})",
        tensor=tensor,
    )


@permute_lowering.register_codegen("pallas")
def codegen_permute_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    # pyrefly: ignore [not-iterable]
    dims = [*dims]
    return expr_from_string(
        f"jnp.transpose({{tensor}}, {dims!r})",
        tensor=tensor,
    )


stack_lowering = register_lowering(
    torch.ops.aten.stack.default,
    masked_value_fn=passthrough_masked_value,
)


@stack_lowering.register_codegen("triton")
def codegen_stack(ctx: LoweringContext, node: Node) -> object:
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)

    assert isinstance(tensors, (list, tuple))
    # pyrefly: ignore [bad-index]
    tensor_asts = [ctx.env[t] for t in tensors]
    n = len(tensor_asts)

    if n == 0:
        raise ValueError("Cannot stack empty tensor list")

    # Round up to power of 2 for efficient masking
    padded_size = 1 << (n - 1).bit_length()

    # Create index array [0, 1, 2, 3, ...] for tensor selection
    idx = ctx.cg.device_function.new_var("stack_idx")
    ctx.cg.add_statement(statement_from_string(f"{idx} = tl.arange(0, {padded_size})"))

    # Broadcast index to target dimension shape
    # e.g., dim=0: [:, None, None], dim=1: [None, :, None], dim=2: [None, None, :]
    bidx = ctx.cg.device_function.new_var("broadcast_idx")
    assert isinstance(dim, int)
    pattern = "[" + ", ".join(["None"] * dim + [":"] + ["None"] * max(0, 2 - dim)) + "]"
    ctx.cg.add_statement(statement_from_string(f"{bidx} = {idx}{pattern}"))

    # Expand each input tensor along the stack dimension
    expanded = [ctx.cg.device_function.new_var(f"expanded_{i}") for i in range(n)]
    for var, tensor in zip(expanded, tensor_asts, strict=False):
        tensor_ast = cast("ast.AST", tensor)
        ctx.cg.add_statement(
            statement_from_string(f"{var} = tl.expand_dims({{t}}, {dim})", t=tensor_ast)
        )

    # Initialize result with zeros
    result = ctx.cg.device_function.new_var("stacked_result")
    ctx.cg.add_statement(
        statement_from_string(f"{result} = tl.zeros_like({expanded[0]})")
    )

    # Select each tensor using masks
    for i in range(n):
        mask = ctx.cg.device_function.new_var(f"mask_{i}")
        ctx.cg.add_statement(statement_from_string(f"{mask} = {bidx} == {i}"))
        ctx.cg.add_statement(
            statement_from_string(
                f"{result} = tl.where({mask}, {expanded[i]}, {result})"
            )
        )

    return expr_from_string(result)


@stack_lowering.register_codegen("cute")
def codegen_stack_cute(ctx: LoweringContext, node: Node) -> object:
    tensors = node.args[0]
    assert isinstance(tensors, (list, tuple))
    if not tensors:
        raise ValueError("Cannot stack empty tensor list")
    if not all(isinstance(tensor, Node) for tensor in tensors):
        raise exc.BackendUnsupported("cute", "stack inputs")
    if _shape_chain_only_users(node):
        return CuteShapeChainView(node)
    raise exc.BackendUnsupported(
        "cute", "virtual shape-chain direct consumers are not yet supported"
    )


expand_lowering = register_lowering(
    torch.ops.aten.expand.default,
    masked_value_fn=passthrough_masked_value,
)


@expand_lowering.register_codegen("triton")
def codegen_expand(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, _ = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    # pyrefly: ignore [missing-attribute]
    if node.args[0].meta["val"].ndim != len(shape):
        broadcasting = [":"] * len(shape)
        # pyrefly: ignore [missing-attribute]
        for i in range(len(shape) - node.args[0].meta["val"].ndim):
            broadcasting[i] = "None"
        tensor = expr_from_string(
            f"{{tensor}}[{', '.join(broadcasting)}]", tensor=tensor
        )
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"tl.broadcast_to({{tensor}}, {shape_str})",
        tensor=tensor,
    )


@expand_lowering.register_codegen("pallas")
def codegen_expand_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor, _ = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    # pyrefly: ignore [missing-attribute]
    if node.args[0].meta["val"].ndim != len(shape):
        broadcasting = [":"] * len(shape)
        # pyrefly: ignore [missing-attribute]
        for i in range(len(shape) - node.args[0].meta["val"].ndim):
            broadcasting[i] = "None"
        tensor = expr_from_string(
            f"{{tensor}}[{', '.join(broadcasting)}]", tensor=tensor
        )
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"jnp.broadcast_to({{tensor}}, {shape_str})",
        tensor=tensor,
    )


@expand_lowering.register_codegen("cute")
def codegen_expand_cute(ctx: LoweringContext, node: Node) -> object:
    from .cute.cute_reshape import resolve_cute_shape_chain_value

    tensor = _env_arg(ctx, cast("Node", node.args[0]))
    if isinstance(tensor, CuteShapeChainView):
        if _shape_chain_only_users(node):
            return CuteShapeChainView(node)
        materialized = resolve_cute_shape_chain_value(ctx, node)
        if materialized is None:
            raise exc.BackendUnsupported(
                "cute", "virtual shape-chain direct consumers are not yet supported"
            )
        return materialized
    assert isinstance(tensor, ast.AST)
    return tensor


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


def reduce_3d_dot(ctx: LoweringContext, node: Node, with_acc: bool) -> ast.AST:
    acc = None
    acc_node: Node | None = None
    if with_acc:
        acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(acc, ast.AST)
        assert isinstance(node.args[0], Node)
        acc_node = node.args[0]
        lhs_node = node.args[1]
        rhs_node = node.args[2]
    else:
        lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        lhs_node = node.args[0]
        rhs_node = node.args[1]
    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)

    # Check if inputs are FP8 - if so, redirect user to hl.dot()
    lhs_dtype = lhs_node.meta["val"].dtype
    rhs_dtype = rhs_node.meta["val"].dtype
    acc_dtype_meta: torch.dtype | None = None
    if with_acc:
        assert acc_node is not None
        assert isinstance(acc_node, Node)
        acc_dtype_meta = acc_node.meta["val"].dtype
    if lhs_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and rhs_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        raise NotImplementedError(
            "FP8 GEMM via torch API is not supported yet. Please use hl.dot() instead."
        )

    lhs_shape = list(lhs_node.meta["val"].size())
    rhs_shape = list(rhs_node.meta["val"].size())
    acc_shape = (
        list(acc_node.meta["val"].size())
        if (with_acc and acc_node is not None)
        else None
    )

    # Extract expected output dtype from FX node to match PyTorch eager mode behavior
    out_dtype: torch.dtype | None = None
    if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
        out_dtype = node.meta["val"].dtype

    return emit_tl_dot_with_padding(
        lhs,
        rhs,
        acc if with_acc else None,
        lhs_dtype,
        rhs_dtype,
        acc_dtype=acc_dtype_meta if with_acc else None,
        out_dtype=out_dtype,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        acc_shape=acc_shape,
    )


bmm_lowering = register_lowering(
    torch.ops.aten.bmm.default,
    apply_dot_requirements,
)
mm_lowering = register_lowering(
    torch.ops.aten.mm.default,
    apply_dot_requirements,
)


@bmm_lowering.register_codegen("triton")
@mm_lowering.register_codegen("triton")
def codegen_mm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"

    return reduce_3d_dot(ctx, node, False)


addmm_lowering = register_lowering(
    torch.ops.aten.addmm.default,
    apply_dot_requirements,
)


@addmm_lowering.register_codegen("triton")
def codegen_addmm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


baddbmm_lowering = register_lowering(
    torch.ops.aten.baddbmm.default,
    apply_dot_requirements,
)


@baddbmm_lowering.register_codegen("triton")
def codegen_baddbmm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


def _pallas_dot(ctx: LoweringContext, node: Node, with_acc: bool) -> ast.AST:
    """Generate jnp.matmul for Pallas backend.

    Uses ``jnp.matmul`` instead of ``jnp.dot`` for correct batch matmul
    semantics (``jnp.dot`` on 3D tensors produces 4D output).

    When either operand is sub-32-bit (bf16, f16, fp8, int8), we pass
    ``preferred_element_type=jnp.float32`` so TPU uses a 32-bit accumulator.
    If the FX-level output dtype is narrower than f32 we cast back afterwards.
    """
    if with_acc:
        acc_node_arg, lhs_node_arg, rhs_node_arg = node.args[:3]
        acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(acc, ast.AST)
        assert isinstance(lhs, ast.AST)
        assert isinstance(rhs, ast.AST)
    else:
        lhs_node_arg, rhs_node_arg = node.args[:2]
        lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(lhs, ast.AST)
        assert isinstance(rhs, ast.AST)
        acc = None

    assert isinstance(lhs_node_arg, Node)
    assert isinstance(rhs_node_arg, Node)
    lhs_dtype = lhs_node_arg.meta["val"].dtype
    rhs_dtype = rhs_node_arg.meta["val"].dtype
    need_f32_acc = _needs_f32_accumulator(lhs_dtype, rhs_dtype)
    out_dtype = node.meta["val"].dtype if "val" in node.meta else None

    return _emit_pallas_matmul(
        lhs,
        rhs,
        acc=acc if with_acc else None,
        need_f32_acc=need_f32_acc,
        out_dtype=out_dtype,
    )


@bmm_lowering.register_codegen("pallas")
@mm_lowering.register_codegen("pallas")
def codegen_mm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, False)


@addmm_lowering.register_codegen("pallas")
def codegen_addmm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, True)


@baddbmm_lowering.register_codegen("pallas")
def codegen_baddbmm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, True)


@bmm_lowering.register_codegen("cute")
@mm_lowering.register_codegen("cute")
def codegen_mm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"
    lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    lhs_node, rhs_node = node.args[:2]
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    out_dtype = node.meta["val"].dtype if "val" in node.meta else None
    outer_acc_dtype = cute_outer_accumulator_dtype(node, is_acc_none=True)
    effective_out_dtype = (
        cute_outer_accumulator_out_dtype(out_dtype, outer_acc_dtype)
        if out_dtype is not None
        else None
    )
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        accumulate_in_lane_loop=not cute_outer_accumulates_result(
            node,
            is_acc_none=True,
        ),
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        out_dtype=effective_out_dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
    )


@addmm_lowering.register_codegen("cute")
def codegen_addmm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    from .cute.cute_mma import codegen_cute_mma

    result = codegen_cute_mma(ctx, node, with_acc=True)
    if result is not None:
        return result
    acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    acc_node = node.args[0]
    lhs_node = node.args[1]
    rhs_node = node.args[2]
    assert isinstance(acc_node, Node)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
    )


@baddbmm_lowering.register_codegen("cute")
def codegen_baddbmm_cute(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    from .cute.cute_mma import codegen_cute_mma

    result = codegen_cute_mma(ctx, node, with_acc=True)
    if result is not None:
        return result
    acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(acc, ast.AST)
    assert isinstance(lhs, (ast.AST, CutePackedAffineLoad))
    acc_node = node.args[0]
    lhs_node = node.args[1]
    rhs_node = node.args[2]
    assert isinstance(acc_node, Node)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)
    assert isinstance(rhs, ast.AST)
    rhs, packed_rhs = cute_lower_rhs_for_matmul(ctx.env, lhs, rhs_node, rhs)
    k_block_id = cute_resolve_active_matmul_k_block_id(
        ctx.cg,
        lhs_node.meta["val"].shape[-1],
        rhs_node.meta["val"].shape[-2],
        rhs_node.meta["val"].shape[-1],
    )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            ctx.cg, packed_node.meta["val"].shape[0]
        )
    static_k_extent = (
        None
        if k_block_id is not None
        else cute_static_k_invariant_extent(lhs_node, rhs_node)
    )
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = (
        hinted(lhs_node.meta["val"].shape[-1]) == 1
        and hinted(rhs_node.meta["val"].shape[-2]) == 1
    )
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    return _emit_cute_matmul(
        ctx.cg,
        lhs,
        rhs,
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=acc,
        acc_dtype=acc_node.meta["val"].dtype,
        lhs_dtype=lhs_node.meta["val"].dtype,
        rhs_dtype=rhs_node.meta["val"].dtype,
    )


iota_lowering = register_lowering(torch.ops.prims.iota.default)
arange_default_lowering = register_lowering(torch.ops.aten.arange.default)


def _triton_iota_expr(
    ctx: LoweringContext,
    *,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype: torch.dtype | None = None,
) -> object:
    dtype = dtype or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)

    # Pad static non-power-of-2 lengths to next power of 2
    length_expr = "{length}"
    if isinstance(length_arg, int) and length_arg != next_power_of_2(length_arg):
        length_expr = str(next_power_of_2(length_arg))

    expr = f"tl.arange(0, {length_expr})"
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    if dtype != torch.int32:
        expr = f"({expr}).to({triton_type(dtype)})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
        length=ctx.to_ast(length_arg),
    )


def _pallas_iota_expr(
    ctx: LoweringContext,
    *,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype: torch.dtype | None = None,
) -> object:
    dtype = dtype or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)

    dtype_str = CompileEnvironment.current().backend.dtype_str(dtype)
    expr = f"jnp.arange(0, {{length}}, dtype={dtype_str})"
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
        length=ctx.to_ast(length_arg),
    )


def _node_dtype_kwarg(node: Node) -> torch.dtype | None:
    dtype = node.kwargs.get("dtype")
    return dtype if isinstance(dtype, torch.dtype) else None


@iota_lowering.register_codegen("triton")
def codegen_iota(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.arange for torch.ops.prims.iota.default operations with automatic power-of-2 padding."""
    return _triton_iota_expr(
        ctx,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype=_node_dtype_kwarg(node),
    )


@iota_lowering.register_codegen("pallas")
def codegen_iota_pallas(ctx: LoweringContext, node: Node) -> object:
    """Generate jnp.arange for torch.ops.prims.iota.default on Pallas."""
    return _pallas_iota_expr(
        ctx,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype=_node_dtype_kwarg(node),
    )


def _cute_iota_expr(
    ctx: LoweringContext,
    *,
    source_node: Node,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype_arg: object = None,
) -> object:
    from .cute.cute_reshape import _get_dim_local_coord
    from .cute.cute_reshape import _grid_local_coord_expr
    from .device_ir import ForLoopGraphInfo
    from .generate_ast import GenerateAST

    assert isinstance(ctx.cg, GenerateAST)
    cg = ctx.cg
    dtype = (
        dtype_arg
        if isinstance(dtype_arg, torch.dtype)
        else CompileEnvironment.current().index_dtype
    )

    env = CompileEnvironment.current()
    length_hint: int | None = None
    if isinstance(length_arg, int):
        length_hint = length_arg
    elif isinstance(length_arg, torch.SymInt):
        length_hint = env.size_hint(length_arg)

    def active_iota_expr() -> ast.AST | None:
        active_block_ids: list[int] = []
        graph_block_ids = [
            graph_info.block_ids
            for graph_info in cg.codegen_graphs
            if isinstance(graph_info, ForLoopGraphInfo)
            and graph_info.graph is source_node.graph
        ]
        if len(graph_block_ids) == 1:
            active_block_ids = [
                candidate
                for candidate in graph_block_ids[0]
                if cg.active_device_loops.get(candidate)
            ]
        if not active_block_ids and cg.current_grid_state is not None:
            active_block_ids = list(cg.current_grid_state.block_ids)
        if not active_block_ids:
            active_block_ids = [
                candidate
                for candidate, loops in cg.active_device_loops.items()
                if loops
            ]
        if not active_block_ids:
            return None

        def local_expr_and_extent(
            candidate: int,
        ) -> tuple[str | None, int | None]:
            loops = cg.active_device_loops.get(candidate)
            if loops:
                loop_state = loops[-1]
                thread_axis = loop_state.block_thread_axes.get(candidate)
                if thread_axis is None:
                    return None, None
                local_expr = _grid_local_coord_expr(cg, candidate, thread_axis)
                elements_per_thread_fn = getattr(
                    loop_state.strategy, "_elements_per_thread_for_block", None
                )
                elements_per_thread = (
                    elements_per_thread_fn(candidate)
                    if callable(elements_per_thread_fn)
                    else 1
                )
                if not isinstance(elements_per_thread, int):
                    return local_expr, None
                return (
                    local_expr,
                    loop_state.thread_axis_sizes.get(thread_axis, 1)
                    * elements_per_thread,
                )
            if cg.current_grid_state is not None:
                thread_axis = cg.current_grid_state.block_thread_axes.get(candidate)
                if thread_axis is None:
                    return None, None
                local_expr = _grid_local_coord_expr(cg, candidate, thread_axis)
                elements_per_thread_fn = getattr(
                    cg.current_grid_state.strategy,
                    "_elements_per_thread_for_block",
                    None,
                )
                elements_per_thread = (
                    elements_per_thread_fn(candidate)
                    if callable(elements_per_thread_fn)
                    else 1
                )
                if not isinstance(elements_per_thread, int):
                    return local_expr, None
                return (
                    local_expr,
                    cg.current_grid_state.thread_axis_sizes.get(thread_axis, 1)
                    * elements_per_thread,
                )
            return None, None

        matched: list[tuple[int, str]] = []
        for candidate in active_block_ids:
            loops = cg.active_device_loops.get(candidate)
            if loops:
                expr = loops[-1].strategy.index_var(candidate)
            elif (
                cg.current_grid_state is not None
                and candidate in cg.current_grid_state.block_ids
            ):
                expr = cg.current_grid_state.strategy.index_var(candidate)
            else:
                continue

            candidate_size = env.block_sizes[candidate].from_config(
                cg.device_function.config
            )
            if (
                not isinstance(candidate_size, int)
                or candidate_size <= 0
                or not isinstance(length_hint, int)
                or length_hint <= 0
            ):
                continue
            if candidate_size == length_hint:
                matched.append((candidate, expr))
            elif candidate_size % length_hint == 0:
                matched.append(
                    (candidate, f"({expr}) // {candidate_size // length_hint}")
                )
            else:
                local_expr, local_extent = local_expr_and_extent(candidate)
                if (
                    local_expr is not None
                    and isinstance(local_extent, int)
                    and local_extent > 0
                ):
                    if local_extent == length_hint:
                        matched.append((candidate, local_expr))
                    elif local_extent % length_hint == 0:
                        matched.append(
                            (
                                candidate,
                                f"({local_expr}) // {local_extent // length_hint}",
                            )
                        )
        if len(matched) != 1:
            return None
        _, expr = matched[0]
        if step != 1:
            expr = f"{{step}} * ({expr})"
        if start != 0:
            expr = f"{{start}} + ({expr})"
        if dtype != torch.int32:
            expr = f"{env.backend.dtype_str(dtype)}({expr})"
        return expr_from_string(
            expr,
            start=ctx.to_ast(start),
            step=ctx.to_ast(step),
        )

    block_id = env.resolve_block_id(length_arg)
    original_block_id = block_id
    if block_id is None:
        if (affine_range := match_cute_affine_range_iota(source_node)) is not None:
            return affine_range
    if "val" in source_node.meta:
        fake_val = source_node.meta["val"]
        if isinstance(fake_val, torch.Tensor) and fake_val.ndim == 1:
            with contextlib.suppress(Exception):
                length_hint = int(fake_val.shape[0])
            local_coord = _get_dim_local_coord(cg, fake_val, 0)
            if local_coord != "cutlass.Int32(0)":
                expr = local_coord
                if step != 1:
                    expr = f"{{step}} * ({expr})"
                if start != 0:
                    expr = f"{{start}} + ({expr})"
                if dtype != torch.int32:
                    expr = f"{env.backend.dtype_str(dtype)}({expr})"
                return expr_from_string(
                    expr,
                    start=ctx.to_ast(start),
                    step=ctx.to_ast(step),
                )
            if block_id is None:
                block_id = env.resolve_block_id(fake_val.shape[0])
            if block_id is None and cg.current_grid_state is not None:
                grid_candidates = [
                    candidate
                    for candidate in cg.current_grid_state.block_ids
                    if isinstance(length_hint, int)
                    and isinstance(
                        env.block_sizes[candidate].from_config(
                            cg.device_function.config
                        ),
                        int,
                    )
                    and env.block_sizes[candidate].from_config(
                        cg.device_function.config
                    )
                    == length_hint
                ]
                if len(grid_candidates) == 1:
                    block_id = grid_candidates[0]
    if block_id is None:
        if (active_expr := active_iota_expr()) is not None:
            return active_expr
        raise exc.BackendUnsupported(
            "cute",
            "hl.arange() requires an active tile/reduction axis in cute kernels",
        )
    resolved_block_id = env.resolve_codegen_block_id(block_id, cg, source_node.graph)
    candidate_block_ids = [resolved_block_id]
    if (
        original_block_id is not None
        and original_block_id != resolved_block_id
        and original_block_id not in candidate_block_ids
    ):
        candidate_block_ids.append(original_block_id)

    expr: str | None = None
    active_block_id: int | None = None
    for candidate_block_id in candidate_block_ids:
        loops = cg.active_device_loops.get(candidate_block_id)
        if loops:
            expr = loops[-1].strategy.index_var(candidate_block_id)
            active_block_id = candidate_block_id
            break
        if (
            cg.current_grid_state is not None
            and candidate_block_id in cg.current_grid_state.block_ids
        ):
            expr = cg.current_grid_state.strategy.index_var(candidate_block_id)
            active_block_id = candidate_block_id
            break
    block_id = resolved_block_id if active_block_id is None else active_block_id

    if expr is None:
        thread_axis: int | None = None
        if cg.current_grid_state is not None:
            thread_axis = cg.current_grid_state.block_thread_axes.get(block_id)
        if thread_axis is None:
            for loops_for_block in cg.active_device_loops.values():
                for loop_state in loops_for_block:
                    block_axes = getattr(loop_state, "block_thread_axes", {})
                    if isinstance(block_axes, dict) and block_id in block_axes:
                        thread_axis = block_axes[block_id]
                        break
                if thread_axis is not None:
                    break
        if thread_axis is not None:
            expr = _grid_local_coord_expr(cg, block_id, thread_axis)
        elif (active_expr := active_iota_expr()) is not None:
            return active_expr
        else:
            raise exc.BackendUnsupported(
                "cute",
                f"hl.arange() axis block_id={block_id} is not active in this scope",
            )
    if step != 1:
        expr = f"{{step}} * ({expr})"
    if start != 0:
        expr = f"{{start}} + ({expr})"
    if dtype != torch.int32:
        expr = f"{env.backend.dtype_str(dtype)}({expr})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
    )


@iota_lowering.register_codegen("cute")
def codegen_iota_cute(ctx: LoweringContext, node: Node) -> object:
    return _cute_iota_expr(
        ctx,
        source_node=node,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype_arg=node.kwargs.get("dtype"),
    )


@arange_default_lowering.register_codegen("triton")
def codegen_arange_default(ctx: LoweringContext, node: Node) -> object:
    return _triton_iota_expr(
        ctx,
        length_arg=node.args[0],
        dtype=_node_dtype_kwarg(node),
    )


@arange_default_lowering.register_codegen("pallas")
def codegen_arange_default_pallas(ctx: LoweringContext, node: Node) -> object:
    return _pallas_iota_expr(
        ctx,
        length_arg=node.args[0],
        dtype=_node_dtype_kwarg(node),
    )


@arange_default_lowering.register_codegen("cute")
def codegen_arange_default_cute(ctx: LoweringContext, node: Node) -> object:
    return _cute_iota_expr(
        ctx,
        source_node=node,
        length_arg=node.args[0],
        dtype_arg=node.kwargs.get("dtype"),
    )


sort_lowering = register_lowering(torch.ops.aten.sort.default)


@sort_lowering.register_codegen("triton")
def codegen_sort(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.sort-based sort implementation.

    torch.sort(input, dim=-1, descending=False, stable=False) returns (values, indices).
    We implement this using tl.sort for values.
    For indices, we compute the rank of each element to determine its sorted position.

    Note: tl.sort only works on the last dimension currently.
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)

    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    descending = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
    )
    # stable arg (node.args[3]) is ignored - tl.sort is stable

    assert isinstance(dim, int), f"sort dim must be int, got {type(dim)}"
    assert isinstance(descending, bool), (
        f"sort descending must be bool, got {type(descending)}"
    )

    # Get the input tensor shape info
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # tl.sort only supports sorting on the last dimension
    assert dim == ndim - 1, (
        f"tl.sort only supports sorting on last dimension, got dim={dim}"
    )

    descending_str = "True" if descending else "False"

    # Generate sorted values using tl.sort
    sorted_vals = ctx.cg.device_function.new_var("sorted_vals")
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_vals} = tl.sort({{tensor}}, descending={descending_str})",
            tensor=tensor,
        )
    )

    # Skip O(N^2) argsort when indices are not used downstream
    indices_used = any(
        user.target is getitem and user.args[1] == 1 for user in node.users
    )
    if not indices_used:
        return (expr_from_string(sorted_vals), None)

    # For indices, compute argsort using ranking:
    # For each element x[..., i], its rank is count of elements strictly less (or greater for descending)
    # plus count of equal elements with smaller index (for stability).
    # rank[..., i] gives the sorted position of x[..., i], so we need to invert this.
    sorted_indices = ctx.cg.device_function.new_var("sorted_indices")
    rank = ctx.cg.device_function.new_var("rank")
    idx_var = ctx.cg.device_function.new_var("idx")

    # Get size of last dimension (must be power of 2 for tl.sort)
    n = input_tensor.shape[-1]
    env = CompileEnvironment.current()
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    n_pow2 = next_power_of_2(n_hint)

    # Create indices: [0, 1, 2, ..., n-1]
    ctx.cg.add_statement(statement_from_string(f"{idx_var} = tl.arange(0, {n_pow2})"))

    # Set up dimension-specific indexing patterns and comparison operator
    cmp_op = ">" if descending else "<"
    if ndim == 1:
        # 1D: compare [1, n] with [n, 1], reduce over axis 1
        t_a, t_b = "[None, :]", "[:, None]"
        i_a, i_b = "[None, :]", "[:, None]"
        reduce_axis = 1
        # For inverting: [n, 1] == [1, n], reduce axis 0
        r_a, r_b, inv_i_a, _inv_i_b, inv_axis = (
            "[:, None]",
            "[None, :]",
            "[:, None]",
            "[None, :]",
            0,
        )
    elif ndim == 2:
        # 2D: compare [batch, 1, n] with [batch, n, 1], reduce over axis 2
        t_a, t_b = "[:, None, :]", "[:, :, None]"
        i_a, i_b = "[None, None, :]", "[None, :, None]"
        reduce_axis = 2
        # For inverting: [batch, n, 1] == [1, 1, n], reduce axis 1
        r_a, r_b, inv_i_a, _inv_i_b, inv_axis = (
            "[:, :, None]",
            "[None, None, :]",
            "[None, :, None]",
            "[None, None, :]",
            1,
        )
    else:
        raise NotImplementedError

    # Compute rank: count elements that should come before + tie-breaking
    ctx.cg.add_statement(
        statement_from_string(
            f"{rank} = tl.sum(tl.where({{tensor}}{t_a} {cmp_op} {{tensor}}{t_b}, 1, 0), axis={reduce_axis}) + "
            f"tl.sum(tl.where(({{tensor}}{t_a} == {{tensor}}{t_b}) & ({idx_var}{i_a} < {idx_var}{i_b}), 1, 0), axis={reduce_axis})",
            tensor=tensor,
        )
    )

    # Invert the rank permutation: sorted_indices[rank[i]] = i
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_indices} = tl.sum(tl.where({rank}{r_a} == {idx_var}{r_b}, {idx_var}{inv_i_a}, 0), axis={inv_axis})"
        )
    )

    # Return as tuple (values, indices)
    return (expr_from_string(sorted_vals), expr_from_string(sorted_indices))


gather_lowering = register_lowering(
    torch.ops.aten.gather.default,
    masked_value_fn=passthrough_masked_value,
)


@gather_lowering.register_codegen("triton")
def codegen_gather(ctx: LoweringContext, node: Node) -> object:
    """Generate gather implementation using tl.gather.

    torch.gather(input, dim, index) gathers values along dim using index.
    Both input and index must be already-loaded tiles (not host tensors).
    Uses Triton's tl.gather for the actual gather operation.
    """
    # Validate arguments
    assert not node.kwargs, "gather does not support keyword arguments"
    assert len(node.args) == 3, f"gather expects 3 arguments, got {len(node.args)}"

    input_node = node.args[0]
    dim = node.args[1]
    index_node = node.args[2]

    assert isinstance(input_node, Node), "gather input must be a Node"
    assert isinstance(dim, int), f"gather dim must be int, got {type(dim)}"
    assert isinstance(index_node, Node), "gather index must be a Node"

    input_tensor = input_node.meta["val"]

    # Validate that input is a tensor
    assert isinstance(input_tensor, torch.Tensor), (
        f"gather input must be a tensor, got {type(input_tensor)}"
    )

    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # Validate dim is in range
    assert 0 <= dim < ndim, (
        f"gather dim {dim} out of range for tensor with {ndim} dimensions"
    )

    fn = ctx.cg.device_function

    # Get the input and index AST nodes
    input_ast_raw = _env_arg(ctx, input_node)
    assert isinstance(input_ast_raw, ast.AST)
    input_ast = input_ast_raw

    index_ast_raw = _env_arg(ctx, index_node)
    assert isinstance(index_ast_raw, ast.AST)
    index_ast = index_ast_raw

    result_var = fn.new_var("gather_result")

    ctx.cg.add_statement(
        statement_from_string(
            f"{result_var} = tl.gather({{input}}, {{index}}.to(tl.int32), axis={dim})",
            input=input_ast,
            index=index_ast,
        )
    )

    return expr_from_string(result_var)


topk_lowering = register_lowering(torch.ops.aten.topk.default)


@topk_lowering.register_codegen("triton")
def codegen_topk(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.topk-based topk implementation.

    torch.topk(input, k, dim=-1, largest=True, sorted=True) returns (values, indices).
    We use tl.topk for values (when largest=True) or tl.sort (when largest=False).
    For indices, we compute argsort using a ranking approach.

    Note: tl.topk/tl.sort only works on the last dimension currently.
    See: https://github.com/triton-lang/triton/blob/main/python/triton/language/standard.py
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)

    k = node.args[1]
    assert isinstance(k, int), f"topk k must be int, got {type(k)}"

    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    largest = node.args[3] if len(node.args) > 3 else node.kwargs.get("largest", True)
    # sorted arg (node.args[4]) is ignored - tl.topk always returns sorted

    assert isinstance(dim, int), f"topk dim must be int, got {type(dim)}"
    assert isinstance(largest, bool), f"topk largest must be bool, got {type(largest)}"

    # Get the input tensor shape info
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # tl.topk only supports sorting on the last dimension
    assert dim == ndim - 1, f"tl.topk only supports the last dimension, got dim={dim}"

    # Get size of last dimension
    n = input_tensor.shape[-1]
    env = CompileEnvironment.current()
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    n_pow2 = next_power_of_2(n_hint)
    k_pow2 = next_power_of_2(k)

    # Generate top-k values using tl.topk (for largest=True) or tl.sort (for largest=False)
    topk_vals = ctx.cg.device_function.new_var("topk_vals")
    if largest:
        # tl.topk returns top k largest elements directly
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_vals} = tl.topk({{tensor}}, {k_pow2})",
                tensor=tensor,
            )
        )
    else:
        # tl.topk only supports largest=True, so use tl.sort with descending=False
        sorted_vals = ctx.cg.device_function.new_var("sorted_vals")
        ctx.cg.add_statement(
            statement_from_string(
                f"{sorted_vals} = tl.sort({{tensor}}, descending=False)",
                tensor=tensor,
            )
        )
        # Need to gather first k elements from sorted
        k_idx = ctx.cg.device_function.new_var("k_idx")
        idx_n = ctx.cg.device_function.new_var("idx_n")
        ctx.cg.add_statement(statement_from_string(f"{k_idx} = tl.arange(0, {k_pow2})"))
        ctx.cg.add_statement(statement_from_string(f"{idx_n} = tl.arange(0, {n_pow2})"))
        if ndim == 1:
            ctx.cg.add_statement(
                statement_from_string(
                    f"{topk_vals} = tl.sum("
                    f"tl.where(({idx_n}[:, None] == {k_idx}[None, :]) & ({k_idx}[None, :] < {k}), "
                    f"{sorted_vals}[:, None], 0.0), axis=0)"
                )
            )
        else:
            ctx.cg.add_statement(
                statement_from_string(
                    f"{topk_vals} = tl.sum("
                    f"tl.where(({idx_n}[None, :, None] == {k_idx}[None, None, :]) & ({k_idx}[None, None, :] < {k}), "
                    f"{sorted_vals}[:, :, None], 0.0), axis=1)"
                )
            )

    # For indices, compute argsort using ranking approach
    topk_indices = ctx.cg.device_function.new_var("topk_indices")
    rank = ctx.cg.device_function.new_var("rank")
    idx_var = ctx.cg.device_function.new_var("idx")

    ctx.cg.add_statement(statement_from_string(f"{idx_var} = tl.arange(0, {n_pow2})"))

    # Set up dimension-specific indexing patterns and comparison operator
    cmp_op = ">" if largest else "<"
    if ndim == 1:
        t_a, t_b = "[None, :]", "[:, None]"
        i_a, i_b = "[None, :]", "[:, None]"
        reduce_axis = 1
        r_a, r_b, inv_i_a, inv_axis = "[:, None]", "[None, :]", "[:, None]", 0
    elif ndim == 2:
        t_a, t_b = "[:, None, :]", "[:, :, None]"
        i_a, i_b = "[None, None, :]", "[None, :, None]"
        reduce_axis = 2
        r_a, r_b, inv_i_a, inv_axis = (
            "[:, :, None]",
            "[None, None, :]",
            "[None, :, None]",
            1,
        )
    else:
        raise NotImplementedError

    # Compute rank: count elements that should come before + tie-breaking
    ctx.cg.add_statement(
        statement_from_string(
            f"{rank} = tl.sum(tl.where({{tensor}}{t_a} {cmp_op} {{tensor}}{t_b}, 1, 0), axis={reduce_axis}) + "
            f"tl.sum(tl.where(({{tensor}}{t_a} == {{tensor}}{t_b}) & ({idx_var}{i_a} < {idx_var}{i_b}), 1, 0), axis={reduce_axis})",
            tensor=tensor,
        )
    )

    # Invert rank permutation to get sorted indices, then gather first k
    sorted_indices = ctx.cg.device_function.new_var("sorted_indices")
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_indices} = tl.sum(tl.where({rank}{r_a} == {idx_var}{r_b}, {idx_var}{inv_i_a}, 0), axis={inv_axis})"
        )
    )

    # Gather first k indices
    k_idx_final = ctx.cg.device_function.new_var("k_idx")
    ctx.cg.add_statement(
        statement_from_string(f"{k_idx_final} = tl.arange(0, {k_pow2})")
    )

    if ndim == 1:
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_indices} = tl.sum("
                f"tl.where(({idx_var}[:, None] == {k_idx_final}[None, :]) & ({k_idx_final}[None, :] < {k}), "
                f"{sorted_indices}[:, None], 0), axis=0)"
            )
        )
    else:
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_indices} = tl.sum("
                f"tl.where(({idx_var}[None, :, None] == {k_idx_final}[None, None, :]) & ({k_idx_final}[None, None, :] < {k}), "
                f"{sorted_indices}[:, :, None], 0), axis=1)"
            )
        )

    return (expr_from_string(topk_vals), expr_from_string(topk_indices))
