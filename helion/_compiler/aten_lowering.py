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
from ..language._decorators import BackendDispatchTable
from ..language._decorators import is_api_func
from ..language._tracing_ops import _get_symnode
from ..language.matmul_ops import MATMUL_DIM_BLOCK_IDS_META
from ..language.matmul_ops import MATMUL_FACT_ID_META
from ..language.matmul_ops import enforce_dot_requirements
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .compile_environment import _to_sympy
from .host_function import HostFunction
from .node_masking import apply_masking
from .node_masking import broadcasts_onto_masked_lanes
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
    codegen_impls: BackendDispatchTable[CodegenHandler] = dataclasses.field(
        default_factory=BackendDispatchTable
    )

    def register_codegen(
        self, backend: str
    ) -> Callable[[CodegenHandler], CodegenHandler]:
        def decorator(handler: CodegenHandler) -> CodegenHandler:
            from ..language._decorators import _register_codegen_handler

            _register_codegen_handler(self.codegen_impls, backend, handler)
            return handler

        return decorator

    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        env = CompileEnvironment.current()
        handler = self.codegen_impls.lookup(
            env.codegen_name,
            backend_name=env.backend_name,
            what=f"Aten lowering codegen not registered for {self.target or 'unknown'!r}",
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


def _keep_shape_size_reads(node: Node) -> None:
    """Keep physical tile extents in shape parameters."""
    physical: Node | None = None

    def physical_size(arg: Node) -> Node:
        nonlocal physical
        if arg is not node:
            return arg
        if physical is None:
            with node.graph.inserting_before(node):
                physical = node.graph.call_function(
                    _get_symnode,
                    (
                        CompileEnvironment.current().sympy_debug(
                            _to_sympy(node.meta["val"])
                        ),
                    ),
                )
            physical.meta.update(node.meta)
            physical.meta["helion_host_scalar"] = False
        return physical

    for user in list(node.users):
        if isinstance(user.target, torch._ops.OpOverload):
            parameters = [argument.name for argument in user.target._schema.arguments]
        elif is_api_func(user.target):
            parameters = list(user.target._signature.parameters)
        else:
            continue
        args, kwargs = list(user.args), dict(user.kwargs)
        for position, name in enumerate(parameters):
            if name not in {"size", "shape", "length", "output_size"} and not (
                user.target
                in (
                    torch.ops.aten.arange.default,
                    torch.ops.aten.arange.start,
                    torch.ops.aten.arange.start_step,
                )
                and name == "end"
            ):
                continue
            if position < len(args):
                args[position] = map_arg(args[position], physical_size)
            elif name in kwargs:
                kwargs[name] = map_arg(kwargs[name], physical_size)
        if tuple(args) != user.args or kwargs != user.kwargs:
            user.args, user.kwargs = tuple(args), kwargs


def resolve_reduction_sizes(graph: torch.fx.Graph) -> None:
    """Read a full slice's logical extent, independent of padding or chunk size."""
    env = CompileEnvironment.current()
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in (
            torch.ops.aten.sym_size.int,
            _get_symnode,
        ):
            continue
        block_id = env.get_block_id(node.meta["val"])
        if block_id is None or not env.block_sizes[block_id].reduction:
            continue
        _keep_shape_size_reads(node)
        size = env.block_sizes[block_id].size
        assert isinstance(size, (int, torch.SymInt))
        # Keep the scalar node's identity: nested graph captures can refer to
        # it outside FX's users map. Inline literal values on ordinary FX edges
        # as before, while keeping their source addressable for those captures.
        node.target = _get_symnode
        size_expr = _to_sympy(size)
        node.args = (
            str(size) if isinstance(size, int) else env.sympy_debug(size_expr),
        )
        node.kwargs = {}
        node.meta["val"] = size
        node.meta["helion_host_scalar"] = isinstance(size, int) or (
            HostFunction.current().is_host_scalar_expr(size_expr)
        )
        if isinstance(size, int):
            for user in list(node.users):
                user.args, user.kwargs = map_arg(
                    (user.args, user.kwargs),
                    lambda arg, extent=size, source=node: (
                        extent if arg is source else arg
                    ),
                )


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


# The dtypes that read all-clear bits as zero (not float8_e8m0fnu, which reads
# them as 2^-127, nor the packed sub-byte dtypes).
_ZERO_BITS_READ_AS_ZERO = frozenset(
    {
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
)


def view_dtype_masked_value(node: Node) -> float | bool | None:
    """A view reinterprets bits, so its masked lanes keep a known value only
    where the source's is a known bit pattern: an integer or bool zero, all
    bits clear.  A floating masked value of zero may be -0.0, since masked
    values don't track the sign of zero (``-x`` of a masked 0 is tracked as
    0).  All-clear bits read as zero in the target dtype if it is one of
    ``_ZERO_BITS_READ_AS_ZERO``."""
    source, dtype = node.args[:2]
    assert isinstance(source, Node) and isinstance(dtype, torch.dtype)
    source_dtype = source.meta["val"].dtype
    if (
        source_dtype.is_floating_point
        or source_dtype.is_complex
        or dtype not in _ZERO_BITS_READ_AS_ZERO
        or cached_masked_value(source) != 0
    ):
        return None
    if dtype == torch.bool:
        return False
    return 0.0 if dtype.is_floating_point else 0


view_dtype_lowering = register_lowering(
    torch.ops.aten.view.dtype,
    masked_value_fn=view_dtype_masked_value,
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


def stack_masked_value(node: Node) -> float | bool | None:
    # Every input lands in the result, so they must share their masked value
    values = {
        cached_masked_value(input_node)
        for input_node in node.all_input_nodes
        if isinstance(input_node.meta["val"], torch.Tensor)
    }
    return values.pop() if len(values) == 1 else None


stack_lowering = register_lowering(
    torch.ops.aten.stack.default,
    masked_value_fn=stack_masked_value,
)


def expand_masked_value(node: Node) -> float | bool | None:
    (input_node, *_) = node.all_input_nodes
    if broadcasts_onto_masked_lanes(input_node, node):
        return None
    return cached_masked_value(input_node)


expand_lowering = register_lowering(
    torch.ops.aten.expand.default,
    masked_value_fn=expand_masked_value,
)


def apply_dot_requirements(lowering: AtenLowering, node: Node) -> Lowering:
    """Apply min_dot_size requirements to the config_spec"""
    assert not node.kwargs, "dot kwargs not supported"
    assert len(node.args) in (2, 3)
    lproxy, rproxy = map_arg(node.args[-2:], lambda arg: arg.meta["val"])
    assert isinstance(lproxy, torch.Tensor)
    assert isinstance(rproxy, torch.Tensor)
    # Update config spec min sizes for M, N, K
    fact_id = enforce_dot_requirements(lproxy, rproxy)
    node.meta[MATMUL_FACT_ID_META] = fact_id
    fact = CompileEnvironment.current().config_spec.matmul_facts[fact_id]
    output_ndim = max(fact.lhs_ndim, fact.rhs_ndim)
    node.meta[MATMUL_DIM_BLOCK_IDS_META] = (
        *(None for _ in range(output_ndim - 2)),
        fact.m_block_id,
        fact.n_block_id,
    )
    # inputs to the dot operation must be zero-masked
    *maybe_acc, lnode, rnode = node.args
    assert isinstance(lnode, Node)
    assert isinstance(rnode, Node)
    lnode = apply_masking(lnode, base_node=node, other=0)
    rnode = apply_masking(rnode, base_node=node, other=0)
    node.args = (*maybe_acc, lnode, rnode)
    return lowering


def matmul_masked_value(node: Node) -> float | bool | None:
    """Propagate zero padding through a matmul under fast-math semantics."""
    if not CompileEnvironment.current().settings.fast_math:
        # In strict mode, zero multiplied by NaN or infinity is not zero.
        return None
    lhs, rhs = cast("tuple[Node, Node]", node.args[:2])
    return (
        0 if cached_masked_value(lhs) == 0 and cached_masked_value(rhs) == 0 else None
    )


bmm_lowering = register_lowering(
    torch.ops.aten.bmm.default,
    apply_dot_requirements,
    masked_value_fn=matmul_masked_value,
)


mm_lowering = register_lowering(
    torch.ops.aten.mm.default,
    apply_dot_requirements,
    masked_value_fn=matmul_masked_value,
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


def rolled_reduction_iota_index(ctx: LoweringContext, node: Node) -> str | None:
    """The rolled reduction loop's index, for an iota over its reduction dim.

    ``torch.tril``'s column iota of an ``x[tile, :]`` row, or
    ``hl.arange(row.size(1))``, spans the row's reduction dim, so inside the
    rolled reduction loop it holds the positions of the current chunk.
    """
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor) or val.ndim != 1:
        return None
    return ctx.cg.rolled_reduction_index_var(val.shape[0])


def _node_dtype_kwarg(node: Node) -> torch.dtype | None:
    dtype = node.kwargs.get("dtype")
    return dtype if isinstance(dtype, torch.dtype) else None


sort_lowering = register_lowering(torch.ops.aten.sort.default)


# A gather's masked-off lanes read the source at (masked) index 0, which is
# real data, so its masked value is unknown.
gather_lowering = register_lowering(torch.ops.aten.gather.default)


topk_lowering = register_lowering(torch.ops.aten.topk.default)
