from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from .._compat import min_dot_size
from .._compat import sm100_dot_tmem_lhs_broken
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .dtype_utils import cast_ast

if TYPE_CHECKING:
    import ast

original_matmul = torch.matmul


def torch_matmul_replacement(
    a: torch.Tensor, b: torch.Tensor, *extra_args: object, **extra_kwargs: object
) -> torch.Tensor:
    if extra_kwargs and "out" in extra_kwargs:
        raise NotImplementedError(
            "torch.matmul(..., out=...) is not supported in Helion kernel"
        )
    if a.dim() != b.dim():
        # torch.matmul broadcasts a 2-D operand against a 3-D one; expand the
        # smaller side so the batched path below sees matching ranks.
        if a.dim() == 3 and b.dim() == 2:
            b = b.unsqueeze(0).expand(a.size(0), -1, -1)
        elif a.dim() == 2 and b.dim() == 3:
            a = a.unsqueeze(0).expand(b.size(0), -1, -1)
        else:
            raise NotImplementedError(
                "torch.matmul with different input tensor dims is not supported in Helion kernel"
            )
    if a.dim() == 2 and b.dim() == 2:
        return original_matmul(a, b)
    if a.dim() == 3 and b.dim() == 3:
        return torch.bmm(a, b)
    raise NotImplementedError(
        "torch.matmul with input tensor dim <2 or >3 is not supported in Helion kernel"
    )


def tensor_matmul_replacement(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch_matmul_replacement(self, other)


def _emit_tl_dot(
    lhs: ast.AST,
    rhs: ast.AST,
    *,
    input_precision: str | None = None,
    acc: ast.AST | None = None,
    out_dtype: torch.dtype | None = None,
) -> ast.AST:
    """Build a tl.dot AST with optional acc/input_precision/out_dtype.

    The caller is responsible for ensuring compatible operand/accumulator
    dtypes for fused accumulation when providing `acc`.
    """
    kwargs = {"lhs": lhs, "rhs": rhs}
    parts = ["tl.dot({lhs}, {rhs}"]
    if acc is not None:
        kwargs["acc"] = acc
        parts.append(", acc={acc}")
    if input_precision:
        parts.append(f", input_precision='{input_precision}'")
    if out_dtype:
        parts.append(
            f", out_dtype={CompileEnvironment.current().backend.dtype_str(out_dtype)}"
        )
    parts.append(")")
    return expr_from_string("".join(parts), **kwargs)


def _emit_tl_dot_scaled(
    lhs: ast.AST,
    lhs_scale: ast.AST,
    lhs_format: str,
    rhs: ast.AST,
    rhs_scale: ast.AST,
    rhs_format: str,
    *,
    acc: ast.AST | None = None,
    out_dtype: torch.dtype | None = None,
) -> ast.AST:
    """Build a tl.dot_scaled AST with optional acc/out_dtype.

    Format strings and scale tensors are passed as compile-time constants
    and AST placeholders respectively.
    """
    kwargs = {"lhs": lhs, "lhs_scale": lhs_scale, "rhs": rhs, "rhs_scale": rhs_scale}
    parts = [
        (
            f"tl.dot_scaled({{lhs}}, {{lhs_scale}}, '{lhs_format}', "
            f"{{rhs}}, {{rhs_scale}}, '{rhs_format}'"
        )
    ]
    if acc is not None:
        kwargs["acc"] = acc
        parts.append(", acc={acc}")
    if out_dtype is not None:
        parts.append(
            f", out_dtype={CompileEnvironment.current().backend.dtype_str(out_dtype)}"
        )
    parts.append(")")
    return expr_from_string("".join(parts), **kwargs)


def _compute_out_dtype(
    mat1_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    acc_dtype: torch.dtype | None = None,
) -> torch.dtype:
    """Compute the output dtype for dot operation."""
    if acc_dtype is not None:
        # If accumulator is provided, use its dtype
        return acc_dtype

    # When no accumulator is specified:
    # For int8 inputs, default to int32
    if mat1_dtype == torch.int8 or mat2_dtype == torch.int8:
        return torch.int32
    # For all other inputs (including FP8), default to float32
    return torch.float32


def _needs_f32_accumulator(lhs_dtype: torch.dtype, rhs_dtype: torch.dtype) -> bool:
    """Return True when either operand is sub-32-bit (bf16, f16, fp8, int8).

    When True the Pallas backend should pass
    ``preferred_element_type=jnp.float32`` so the TPU uses a 32-bit
    accumulator for the matmul.
    """
    return lhs_dtype.itemsize < 4 or rhs_dtype.itemsize < 4


def _emit_pallas_matmul(
    lhs: ast.AST,
    rhs: ast.AST,
    *,
    acc: ast.AST | None = None,
    need_f32_acc: bool = False,
    out_dtype: torch.dtype | None = None,
    lhs_ndim: int = 2,
) -> ast.AST:
    """Build a ``lax.dot_general`` AST node for the Pallas backend.

    Parameters
    ----------
    lhs, rhs:
        AST nodes for the left / right operands.
    acc:
        Optional AST node for the accumulator (``acc + dot_general(...)``).
    need_f32_acc:
        When True, emit ``preferred_element_type=jnp.float32`` and, if
        *out_dtype* is narrower than f32, append a
        ``lax.convert_element_type`` cast.
    out_dtype:
        Desired output dtype.  Only used when *need_f32_acc* is True to
        decide whether a cast-back is required.
    lhs_ndim:
        Number of dimensions in the left operand (2 for mm, 3 for bmm).
    """
    if lhs_ndim == 3:
        dim_numbers = "(((2,), (1,)), ((0,), (0,)))"
    elif lhs_ndim == 2:
        dim_numbers = "(((1,), (0,)), ((), ()))"
    else:
        raise ValueError(f"lhs_ndim must be 2 or 3, got {lhs_ndim}")

    env = CompileEnvironment.current()
    precision = env.backend.map_dot_precision(env.settings.dot_precision)
    precision_arg = f", precision={precision!r}" if precision else ""
    if need_f32_acc:
        dot_expr = expr_from_string(
            f"lax.dot_general({{lhs}}, {{rhs}}, dimension_numbers={dim_numbers}{precision_arg}, preferred_element_type=jnp.float32)",
            lhs=lhs,
            rhs=rhs,
        )
    else:
        dot_expr = expr_from_string(
            f"lax.dot_general({{lhs}}, {{rhs}}, dimension_numbers={dim_numbers}{precision_arg})",
            lhs=lhs,
            rhs=rhs,
        )

    if acc is not None:
        dot_expr = expr_from_string("{acc} + {dot}", acc=acc, dot=dot_expr)

    # Cast back if the result should be narrower than f32
    if need_f32_acc and out_dtype is not None and out_dtype.itemsize < 4:
        dtype_str = env.backend.dtype_str(out_dtype)
        dot_expr = expr_from_string(
            f"lax.convert_element_type({{val}}, {dtype_str})", val=dot_expr
        )

    return dot_expr


def _resolve_dim_size(
    v: int | torch.SymInt | sympy.Expr,
) -> int | torch.SymInt | sympy.Expr:
    """Resolve a dimension size value using both direct resolution and config lookup."""
    if isinstance(v, int):
        return v

    env = CompileEnvironment.current()
    sym_expr = (
        v._sympy_()
        if isinstance(v, torch.SymInt)
        else v
        if isinstance(v, sympy.Expr)
        else None
    )

    if sym_expr is not None:
        replaced = env.shape_env.replace(sym_expr)
        if isinstance(replaced, (int, sympy.Integer, sympy.Number)):
            return int(replaced)

    if not isinstance(v, (torch.SymInt, sympy.Expr)):
        return v

    device_fn = DeviceFunction.current()
    cfg = device_fn.config
    block_idx = env.get_block_id(v)

    if block_idx is None:
        return v

    cfg_value = env.block_sizes[block_idx].from_config(cfg)
    if isinstance(cfg_value, int):
        return cfg_value
    if isinstance(cfg_value, (torch.SymInt, sympy.Expr)):
        sym_expr = (
            cfg_value._sympy_() if isinstance(cfg_value, torch.SymInt) else cfg_value
        )
        replaced = env.shape_env.replace(sym_expr)
        if isinstance(replaced, (int, sympy.Integer, sympy.Number)):
            return int(replaced)

    return v


def _dot_lhs_stays_in_smem(node: torch.fx.Node) -> bool:
    """Whether Triton keeps this ``tl.dot`` LHS operand out of tensor memory.

    Mirrors Triton's ``PromoteLHSToTMem`` heuristic on sm100: an LHS that is a
    plain load (possibly through casts/transposes, including values passed in
    from an enclosing graph) stays in shared memory; anything else (another
    dot's output, elementwise math, constants) is staged through tensor
    memory, which is broken for wide accumulators there (see
    ``sm100_dot_tmem_lhs_broken``).  Loop-carried values resolve to their
    initial value, so an LHS that starts as a load but is rebound to computed
    values across iterations can be misclassified as safe.
    """
    # Imported here to avoid circular imports with helion.language.
    from ..language._tracing_ops import _new_var
    from ..language.memory_ops import load
    from .device_ir import NodeArgsGraphInfo
    from .host_function import HostFunction

    transparent_targets = (
        _new_var,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
    )
    # Codegen interprets per-config graph copies, but placeholder_to_outer_arg
    # on a copy returns nodes from the original traced graphs, so search both.
    graphs = [
        *DeviceFunction.current().codegen.codegen_graphs,
        *HostFunction.current().device_ir.graphs,
    ]
    while True:
        if node.op == "placeholder":
            graph_info = next((g for g in graphs if g.graph is node.graph), None)
            if isinstance(graph_info, NodeArgsGraphInfo):
                node = graph_info.placeholder_to_outer_arg(node)
                continue
            return False
        if node.op != "call_function":
            return False
        if node.target is load:
            return True
        if node.target in transparent_targets and isinstance(
            node.args[0], torch.fx.Node
        ):
            node = node.args[0]
            continue
        return False


def _pad_tensor(
    tensor: ast.AST,
    pad_dim: int,
    cur_size: int,
    target_size: int,
    other_dim: int | torch.SymInt | sympy.Expr,
) -> ast.AST:
    """Pad tensor by repeatedly doubling specified dimension."""
    assert pad_dim in (0, 1), f"pad_dim must be 0 or 1, got {pad_dim}"
    x = tensor
    shape_str = DeviceFunction.current().tile_strategy.shape_str
    while cur_size < target_size:
        x = expr_from_string("tl.join({x}, tl.zeros_like({x}))", x=x)
        x = expr_from_string(
            f"tl.permute({{x}}, {'[2, 0, 1]' if pad_dim == 0 else '[0, 2, 1]'})", x=x
        )
        cur_size *= 2
        shape = [cur_size, other_dim] if pad_dim == 0 else [other_dim, cur_size]
        # pyrefly: ignore [bad-argument-type]
        x = expr_from_string(f"tl.reshape({{x}}, {shape_str(shape)})", x=x)
    return x


def emit_tl_dot_with_padding(
    lhs: ast.AST,
    rhs: ast.AST,
    acc: ast.AST | None,
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
    *,
    acc_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
    lhs_shape: list[int | torch.SymInt],
    rhs_shape: list[int | torch.SymInt],
    acc_shape: list[int | torch.SymInt] | None = None,
    lhs_fx_node: torch.fx.Node | None = None,
) -> ast.AST:
    device_fn = DeviceFunction.current()
    shape_str = device_fn.tile_strategy.shape_str

    env = CompileEnvironment.current()
    input_precision = env.backend.map_dot_precision(env.settings.dot_precision)
    config = device_fn.config

    lhs_shape_list = list(lhs_shape)
    rhs_shape_list = list(rhs_shape)
    acc_shape_list = list(acc_shape) if acc_shape is not None else None

    if len(lhs_shape_list) < 2:
        raise ValueError("lhs_shape must have at least two dimensions")
    if len(rhs_shape_list) < 1:
        raise ValueError("rhs_shape must have at least one dimension")

    m, k, n = lhs_shape_list[-2], lhs_shape_list[-1], rhs_shape_list[-1]
    common_dtype = torch.promote_types(lhs_dtype, rhs_dtype)
    lhs_cast, rhs_cast = cast_ast(lhs, common_dtype), cast_ast(rhs, common_dtype)
    m, n, k = (_resolve_dim_size(d) for d in (m, n, k))

    fuse_acc = (
        acc is not None
        and acc_dtype in (common_dtype, torch.float32)
        and (out_dtype is None or out_dtype == acc_dtype)
    )
    acc_out = acc if not fuse_acc else None
    acc_for_dot = acc if fuse_acc else None
    acc_cast_dtype = acc_dtype if not fuse_acc else None

    # Determine the out_dtype to use for tl.dot operation, and whether to
    # explicitly cast the tl.dot result to the expected output dtype
    expected_out_dtype = out_dtype or (
        acc_dtype if fuse_acc else _compute_out_dtype(lhs_dtype, rhs_dtype)
    )
    if expected_out_dtype == torch.float32:
        dot_out_dtype = torch.float32
    elif expected_out_dtype == torch.float16:
        dot_out_dtype = (
            torch.float32
            if common_dtype in {torch.float16, torch.bfloat16} and not fuse_acc
            else torch.float16
        )
    elif common_dtype == torch.int8 and expected_out_dtype == torch.int32:
        dot_out_dtype = torch.int32
    else:
        # Unsupported dtype (like bfloat16), use float32 and cast afterward
        dot_out_dtype = torch.float32

    # Squeeze 3D shapes to 2D when leading dims map to block size 1 for both operands.
    need_squeeze_dim = (
        len(lhs_shape_list) == 3
        and config is not None
        and all(
            block_idx is not None
            and env.block_sizes[block_idx].from_config(config) == 1
            for block_idx in (
                env.get_block_id(lhs_shape_list[0]),
                env.get_block_id(rhs_shape_list[0]),
            )
        )
    )

    if need_squeeze_dim:
        lhs_cast = expr_from_string(
            f"tl.reshape({{x}}, {shape_str(lhs_shape_list[1:])})", x=lhs_cast
        )
        rhs_cast = expr_from_string(
            f"tl.reshape({{x}}, {shape_str(rhs_shape_list[1:])})", x=rhs_cast
        )

        if acc_for_dot is not None:
            assert acc_shape_list is not None
            acc_for_dot = expr_from_string(
                f"tl.reshape({{x}}, {shape_str(acc_shape_list[1:])})",
                x=acc_for_dot,
            )

    min_m, min_n, min_k = min_dot_size(env.device, lhs_dtype, rhs_dtype)
    dims = {
        d: v if isinstance(v, int) else None
        for d, v in zip(("m", "n", "k"), [m, n, k], strict=False)
    }
    min_sizes = {"m": min_m, "n": min_n, "k": min_k}
    pad_needed = {}
    for d in ("m", "n", "k"):
        dim_value = dims[d]
        pad_needed[d] = dim_value is not None and dim_value < min_sizes[d]
    need_padding = any(pad_needed.values())

    # Triton crashes with a misaligned address on sm100 when a tl.dot LHS is
    # promoted to tensor memory while the accumulator is wider than 256
    # columns (triton-lang/triton#10309).  Split such dots into two M=32
    # halves; splitting along N instead runs out of tensor memory because both
    # halves' accumulators are live at the join.
    split_m = (
        lhs_fx_node is not None
        and isinstance(m, int)
        and m == 64
        and isinstance(n, int)
        and n > 256
        and common_dtype.itemsize in (2, 4)
        and (len(lhs_shape_list) == 2 or need_squeeze_dim)
        and not (pad_needed["m"] or pad_needed["n"])
        and env.backend.name == "triton"
        and sm100_dot_tmem_lhs_broken(env.device)
        and not _dot_lhs_stays_in_smem(lhs_fx_node)
    )

    def emit_dot(
        lhs_final: ast.AST,
        rhs_final: ast.AST,
        acc_final: ast.AST | None,
        k_final: int | torch.SymInt | sympy.Expr,
    ) -> ast.AST:
        if not split_m:
            return _emit_tl_dot(
                lhs_final,
                rhs_final,
                acc=acc_final,
                input_precision=input_precision,
                out_dtype=dot_out_dtype,
            )

        def split_half(x: ast.AST, last_dim: object, index: int) -> ast.AST:
            # pyrefly: ignore [bad-argument-type]
            shape = shape_str([2, 32, last_dim])
            return expr_from_string(
                f"tl.split(tl.permute(tl.reshape({{x}}, {shape}), [1, 2, 0]))[{index}]",
                x=x,
            )

        halves = []
        for index in (0, 1):
            halves.append(
                _emit_tl_dot(
                    split_half(lhs_final, k_final, index),
                    rhs_final,
                    acc=split_half(acc_final, n, index)
                    if acc_final is not None
                    else None,
                    input_precision=input_precision,
                    out_dtype=dot_out_dtype,
                )
            )
        # pyrefly: ignore [bad-argument-type]
        out_shape = shape_str([m, n])
        return expr_from_string(
            f"tl.reshape(tl.permute(tl.join({{lo}}, {{hi}}), [2, 0, 1]), {out_shape})",
            lo=halves[0],
            hi=halves[1],
        )

    if not need_padding:
        result = emit_dot(lhs_cast, rhs_cast, acc_for_dot, k)
    else:
        lhs_pad, rhs_pad, acc_pad = lhs_cast, rhs_cast, acc_for_dot
        pad_specs = [
            ("lhs", "k", 1, min_k, m),
            ("rhs", "k", 0, min_k, n),
            ("lhs", "m", 0, min_m, min_k if pad_needed["k"] else k),
            ("rhs", "n", 1, min_n, min_k if pad_needed["k"] else k),
        ]
        pads = {"lhs": lhs_cast, "rhs": rhs_cast}
        for target, dim, axis, target_size, other_dim in pad_specs:
            if pad_needed[dim] and (cur := dims[dim]):
                pads[target] = _pad_tensor(
                    pads[target], axis, cur, target_size, other_dim
                )
        lhs_pad, rhs_pad = pads["lhs"], pads["rhs"]

        if acc_for_dot is not None and (pad_needed["m"] or pad_needed["n"]):
            assert acc_pad is not None  # acc_pad == acc_for_dot when we reach here
            acc_pad_specs = [
                ("m", 0, min_m, min_n if pad_needed["n"] and dims["n"] else n),
                ("n", 1, min_n, min_m if pad_needed["m"] and dims["m"] else m),
            ]
            for dim, axis, min_dim, other in acc_pad_specs:
                if pad_needed[dim] and (cur := dims[dim]):
                    # pyrefly: ignore [unbound-name]
                    acc_pad = _pad_tensor(acc_pad, axis, cur, min_dim, other)

        result = emit_dot(lhs_pad, rhs_pad, acc_pad, min_k if pad_needed["k"] else k)

        unpad_specs = [
            (
                "n",
                min_n,
                min_m if pad_needed["m"] else m,
                lambda cur, other: [other, 2, cur],
                [0, 2, 1],
            ),
            (
                "m",
                min_m,
                dims["n"] or n,
                lambda cur, other: [2, cur, other],
                [1, 2, 0],
            ),
        ]
        for dim, min_dim, other, shape_fn, perm in unpad_specs:
            if pad_needed[dim] and (cur := dims[dim]):
                assert dim in ("m", "n"), f"dim must be 'm' or 'n', got {dim}"
                cur_size = min_dim
                # pyrefly: ignore [unbound-name]
                while cur_size > cur:
                    cur_size //= 2
                    shape = shape_fn(cur_size, other)
                    result = expr_from_string(
                        f"tl.split(tl.permute(tl.reshape({{x}}, {shape_str(shape)}), {perm}))[0]",
                        x=result,
                    )

    if need_squeeze_dim:
        out_shape = [*lhs_shape_list[:-1], rhs_shape_list[-1]]
        result = expr_from_string(
            f"tl.reshape({{x}}, {shape_str(out_shape)})",
            x=result,
        )

    if acc_cast_dtype is not None:
        result = cast_ast(result, acc_cast_dtype)

    # Explicitly cast to expected output dtype if we used a different out_dtype for tl.dot and haven't already cast
    if dot_out_dtype != expected_out_dtype and acc_cast_dtype != expected_out_dtype:
        assert expected_out_dtype is not None
        result = cast_ast(result, expected_out_dtype)

    return (
        expr_from_string("{acc} + {mm}", acc=acc_out, mm=result)
        if not fuse_acc and acc_out is not None
        else result
    )
