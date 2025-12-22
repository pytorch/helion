from __future__ import annotations

import ast
from collections.abc import Callable
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
from triton import next_power_of_2

from .. import exc
from ..language.matmul_ops import enforce_dot_requirements
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
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
        backend = CompileEnvironment.current().backend
        try:
            handler = self.codegen_impls[backend]
        except KeyError as err:  # pragma: no cover - defensive
            target = self.target or "unknown"
            raise exc.BackendImplementationMissing(
                backend,
                f"Aten lowering codegen not registered for {target!r}",
            ) from err
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


@sym_size_lowering.register_codegen("triton")
def codegen_sym_size(ctx: LoweringContext, node: Node) -> object:
    val = node.meta["val"]
    assert isinstance(
        val, (int, float, bool, torch.SymInt, torch.SymBool, torch.SymFloat)
    )
    return val


getitem_lowering = register_lowering(getitem, masked_value_fn=getitem_masked_value)


@getitem_lowering.register_codegen("triton")
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


@full_lowering.register_codegen("triton")
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
    shape_str = ctx.cg.device_function.tile_strategy.shape_str([*size])
    return expr_from_string(
        f"tl.full({shape_str}, {{value}}, {triton_type(dtype)})",
        value=value_ast,
    )


unsqueeze_lowering = register_lowering(
    torch.ops.aten.unsqueeze.default,
    masked_value_fn=passthrough_masked_value,
)


@unsqueeze_lowering.register_codegen("triton")
def codegen_unsqueeze(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dim = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    assert isinstance(dim, int)
    # pyrefly: ignore [missing-attribute]
    ndim = node.args[0].meta["val"].ndim
    if dim < 0:
        dim += ndim
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


@alias_lowering.register_codegen("triton")
def codegen_alias(ctx: LoweringContext, node: Node) -> object:
    """Alias is a no-op view, just pass through the input tensor."""
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    return tensor


permute_lowering = register_lowering(
    torch.ops.aten.permute.default,
    masked_value_fn=passthrough_masked_value,
)


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


iota_lowering = register_lowering(torch.ops.prims.iota.default)


@iota_lowering.register_codegen("triton")
def codegen_iota(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.arange for torch.ops.prims.iota.default operations with automatic power-of-2 padding."""
    start = node.kwargs.get("start", 0)
    step = node.kwargs.get("step", 1)
    dtype = node.kwargs.get("dtype") or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)
    (length_arg,) = node.args  # expecting a single argument for length

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


def _codegen_rng_op(
    ctx: LoweringContext,
    node: Node,
    rng_function: str,
) -> object:
    """Common codegen implementation for all RNG operations.

    Args:
        ctx: The graph interpreter context
        node: The FX node for this operation
        rng_function: Either "rand" or "randn"
    """
    from .generate_ast import GenerateAST

    assert rng_function in ["rand", "randn"]
    assert isinstance(ctx.cg, GenerateAST)

    # Get unique seed index for this RNG operation
    device_fn = ctx.cg.device_function
    seed_index = device_fn.allocate_rng_seed()

    # Get dimensionality and dtype
    assert hasattr(node, "meta") and "val" in node.meta
    fake_value = node.meta["val"]
    ndim = fake_value.ndim
    dtype = node.kwargs.get("dtype", None)

    # Get dimension names for offset calculation
    env = CompileEnvironment.current()
    dim_names = []
    block_ids = []
    for size in fake_value.size():
        block_id = env.get_block_id(size)
        block_ids.append(block_id)
        block_size = env.block_sizes[block_id].size if block_id is not None else size
        dim_names.append(device_fn.literal_expr(block_size))

    offset_parts: list[str] = []

    for i in range(ndim):
        # Create the index variable with proper broadcasting
        if block_ids[i] is not None:
            index_expr = f"indices_{block_ids[i]}"
        else:
            # For constant dimensions (block_id is None), use tl.arange directly
            index_expr = f"tl.arange(0, {dim_names[i]})"

        # Add broadcasting slices for this dimension
        # For 1D tensors, this will just be indices_0 with no slicing
        slice_parts = []
        for j in range(ndim):
            if j < i:
                slice_parts.append("None")
            elif j == i:
                slice_parts.append(":")
            else:
                slice_parts.append("None")

        # Create the broadcasted index expression
        if ndim == 1:
            # For 1D, no broadcasting needed
            broadcasted_index = index_expr
        else:
            broadcasted_index = f"{index_expr}[{', '.join(slice_parts)}]"

        # Calculate stride (product of dimensions after this one)
        if i < ndim - 1:
            # Use the actual dimension variable names
            stride_parts = dim_names[i + 1 :]
            stride_expr = " * ".join(stride_parts)
            offset_parts.append(f"{broadcasted_index} * {stride_expr}")
        else:
            # Last dimension has no stride multiplication
            offset_parts.append(broadcasted_index)

    offset_expr = expr_from_string(" + ".join(offset_parts))

    # Load seed from buffer using the kernel parameter name
    assert device_fn.rng_seed_buffer_param_name is not None
    seed_expr = expr_from_string(
        "tl.load({buffer} + {index})",
        buffer=expr_from_string(device_fn.rng_seed_buffer_param_name),
        index=create(ast.Constant, value=seed_index),
    )

    # Generate the RNG call
    # Note: tl.rand() and tl.randn() always return float32
    rng_expr = expr_from_string(
        f"tl.{rng_function}({{seed}}, {{offset}})", seed=seed_expr, offset=offset_expr
    )

    # Cast to target dtype only if explicitly specified
    if dtype is not None:
        assert isinstance(dtype, torch.dtype)
        rng_expr = expr_from_string(f"{{val}}.to({triton_type(dtype)})", val=rng_expr)

    return rng_expr


rand_lowering = register_lowering(torch.ops.aten.rand.default)


@rand_lowering.register_codegen("triton")
def codegen_rand(ctx: LoweringContext, node: Node) -> object:
    return _codegen_rng_op(ctx, node, "rand")


randn_lowering = register_lowering(torch.ops.aten.randn.default)


@randn_lowering.register_codegen("triton")
def codegen_randn(ctx: LoweringContext, node: Node) -> object:
    return _codegen_rng_op(ctx, node, "randn")


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
