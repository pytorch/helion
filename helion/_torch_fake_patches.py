from __future__ import annotations

from typing import Iterable, Sequence

import sympy
import torch
import triton._utils as triton_utils
import triton.language as triton_lang
import triton.language.core as triton_core
from sympy.core.relational import Relational
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
import torch._subclasses.fake_tensor as fake_tensor_module
from torch._inductor import sizevars as inductor_sizevars
from torch._inductor.utils import sympy_subs
from torch._subclasses import fake_impls
import torch.fx.experimental.symbolic_shapes as symbolic_shapes
from torch.fx.experimental.symbolic_shapes import guard_or_false
from torch.fx.experimental.symbolic_shapes import SymBool

_SUM_DIM = torch.ops.aten.sum.dim_IntList

_original_dispatch_impl = FakeTensorMode._dispatch_impl


def _normalize_dims(dim_arg: object, rank: int) -> tuple[int, ...] | None:
    if dim_arg is None:
        return tuple(range(rank))
    if isinstance(dim_arg, (list, tuple)):
        dims = tuple(int(d) for d in dim_arg)
    else:
        dims = (int(dim_arg),)
    normalized = []
    for dim in dims:
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            return None
        normalized.append(dim)
    return tuple(sorted(set(normalized)))


def _canonicalize_sum_output(
    fake_mode: FakeTensorMode,
    result: object,
    args: Sequence[object],
    kwargs: dict[str, object],
) -> object:
    if not isinstance(result, FakeTensor):
        return result
    if not args:
        return result

    input_tensor = args[0]
    if not isinstance(input_tensor, FakeTensor):
        return result

    dim_arg = None
    if len(args) > 1:
        dim_arg = args[1]
    elif "dim" in kwargs:
        dim_arg = kwargs["dim"]

    keepdim = bool(kwargs.get("keepdim", False))

    dims = _normalize_dims(dim_arg, input_tensor.dim())
    if dims is None:
        return result

    input_sizes: list[object] = list(input_tensor.size())
    output_sizes: list[object] = list(result.size())

    def _canonical_size(dim: object) -> object:
        return dim

    if keepdim:
        if len(output_sizes) != len(input_sizes):
            return result
        new_sizes = [output_sizes[i] for i in range(len(output_sizes))]
        for idx in range(len(new_sizes)):
            if idx not in dims:
                new_sizes[idx] = _canonical_size(input_sizes[idx])
    else:
        remaining = [i for i in range(len(input_sizes)) if i not in dims]
        if len(remaining) != len(output_sizes):
            return result
        new_sizes = [_canonical_size(input_sizes[i]) for i in remaining]

    try:
        meta = torch.empty_strided(
            new_sizes,
            result.stride(),
            dtype=result.dtype,
            device=torch.device("meta"),
        )
    except Exception:
        return result

    meta.requires_grad_(result.requires_grad)

    replacement = FakeTensor(
        fake_mode,
        meta,
        result.fake_device,
        constant=result.constant,
        real_tensor=result.real_tensor,
        pytype=result.pytype,
        dispatch_keys=result.dispatch_keys,
    )
    return replacement


def _patched_dispatch_impl(
    self: FakeTensorMode,
    func: torch._ops.OpOverload,
    types: Sequence[type],
    args: Sequence[object],
    kwargs: dict[str, object],
) -> object:
    if func is torch.ops.aten.add.Tensor and len(args) >= 2:
        lhs, rhs = args[0], args[1]
        if isinstance(lhs, FakeTensor) and isinstance(rhs, FakeTensor):
            lhs_sizes = tuple(lhs.size())
            rhs_sizes = tuple(rhs.size())
            if len(lhs_sizes) == len(rhs_sizes):
                try:
                    from helion._compiler.compile_environment import CompileEnvironment

                    env = CompileEnvironment.current()
                except Exception:
                    env = None

                def _resolve(dim: object) -> object:
                    if env is None or not isinstance(dim, torch.SymInt):
                        return dim
                    expr = env.shape_env.simplify(dim._sympy_())
                    seen: set[sympy.Expr] = set()
                    while expr in env.specialized_aliases and expr not in seen:
                        seen.add(expr)
                        expr = env.specialized_aliases[expr]
                    return expr

                if all(
                    (ldim == rdim)
                    or (
                        isinstance(ldim, torch.SymInt)
                        and isinstance(rdim, torch.SymInt)
                        and _resolve(ldim) == _resolve(rdim)
                    )
                    for ldim, rdim in zip(lhs_sizes, rhs_sizes)
                ):
                    try:
                        meta = torch.empty_strided(
                            lhs_sizes,
                            rhs.stride(),
                            dtype=rhs.dtype,
                            device=torch.device("meta"),
                        )
                    except Exception:
                        meta = None
                    if meta is not None:
                        meta.requires_grad_(rhs.requires_grad)
                        rhs = FakeTensor(
                            self,
                            meta,
                            rhs.fake_device,
                            constant=rhs.constant,
                            real_tensor=rhs.real_tensor,
                            pytype=rhs.pytype,
                            dispatch_keys=rhs.dispatch_keys,
                        )
                        args = (lhs, rhs, *args[2:])
    result = _original_dispatch_impl(self, func, types, args, kwargs)
    if func is _SUM_DIM:
        result = _canonicalize_sum_output(self, result, args, dict(kwargs))
    return result


FakeTensorMode._dispatch_impl = _patched_dispatch_impl  # type: ignore[assignment]


def _helion_infer_size(a: Sequence[object], b: Sequence[object]) -> tuple[object, ...]:
    dims_a = len(a)
    dims_b = len(b)
    ndim = max(dims_a, dims_b)
    expanded = [0] * ndim

    def _symbols_match(x: object, y: object) -> bool:
        if not isinstance(x, torch.SymInt) or not isinstance(y, torch.SymInt):
            return False
        node_x = getattr(x, "node", None)
        node_y = getattr(y, "node", None)
        shape_env = getattr(node_x, "shape_env", None)
        if shape_env is None or getattr(node_y, "shape_env", None) is not shape_env:
            return False
        expr_x = node_x.expr
        expr_y = node_y.expr
        try:
            evaluated = shape_env._maybe_evaluate_static(sympy.Eq(expr_x, expr_y))
            if evaluated is not None:
                return bool(evaluated)
        except Exception:
            pass

        val_x = shape_env.var_to_val.get(expr_x)
        val_y = shape_env.var_to_val.get(expr_y)
        if val_x is not None and val_y is not None and val_x == val_y:
            return True

        try:
            from helion._compiler.compile_environment import CompileEnvironment

            env = CompileEnvironment.current()
        except Exception:
            env = None
        if env is not None:
            spec_vals = env.specialized_values
            vx = spec_vals.get(expr_x)
            vy = spec_vals.get(expr_y)
            if vx is not None and vy is not None and vx == vy:
                return True

        return False

    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dim_a = dims_a - 1 - offset
        dim_b = dims_b - 1 - offset
        size_a = a[dim_a] if dim_a >= 0 else 1
        size_b = b[dim_b] if dim_b >= 0 else 1

        broadcastable = (
            guard_or_false(size_a == 1)
            or guard_or_false(size_b == 1)
            or size_a == size_b
            or _symbols_match(size_a, size_b)
        )

        torch._check(
            broadcastable,
            lambda size_a=size_a, size_b=size_b, i=i: (
                f"The size of tensor a ({size_a}) must match the size of tensor b ({size_b}) "
                f"at non-singleton dimension {i})"
            ),
        )

        expanded[i] = size_b if guard_or_false(size_a == 1) else size_a

    return tuple(expanded)


fake_impls.infer_size = _helion_infer_size

_orig_get_fast_op_impls = fake_impls.get_fast_op_impls


def _helion_get_fast_op_impls() -> dict[object, object]:
    impls = _orig_get_fast_op_impls()
    impls.pop(torch.ops.aten.add.Tensor, None)
    return impls


fake_impls.get_fast_op_impls = _helion_get_fast_op_impls
fake_tensor_module.get_fast_op_impls = _helion_get_fast_op_impls


def _helion_expr_equal(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
    try:
        from helion._compiler.compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
    except Exception:
        return False

    lhs = env.resolve_alias(lhs)
    rhs = env.resolve_alias(rhs)

    values = {k: sympy.Integer(v) for k, v in env.specialized_values.items()}
    if not values:
        return False
    try:
        lhs_val = sympy_subs(lhs, values)
        rhs_val = sympy_subs(rhs, values)
    except Exception:
        return False
    print("helion inductor eq", lhs, rhs, lhs_val, rhs_val)
    try:
        lhs_val = sympy.Integer(lhs_val)
        rhs_val = sympy.Integer(rhs_val)
    except Exception:
        pass
    return lhs_val == rhs_val


_orig_expect_true = inductor_sizevars.SizeVarAllocator.expect_true


def _helion_expect_true(self: inductor_sizevars.SizeVarAllocator, expr: sympy.Expr) -> bool:
    if isinstance(expr, sympy.Equality):
        if _helion_expr_equal(expr.lhs, expr.rhs):
            return True
    return _orig_expect_true(self, expr)


inductor_sizevars.SizeVarAllocator.expect_true = _helion_expect_true

_orig_guard_or_false = symbolic_shapes.guard_or_false


def _helion_guard_or_false(a: object) -> bool:
    if isinstance(a, SymBool):
        expr = getattr(a.node, "expr", None)
        if isinstance(expr, Relational):
            if isinstance(expr, sympy.Equality):
                if _helion_expr_equal(expr.lhs, expr.rhs):
                    return True
            elif getattr(expr, "rel_op", None) == "!=":
                if _helion_expr_equal(expr.lhs, expr.rhs):
                    return False
    return _orig_guard_or_false(a)


symbolic_shapes.guard_or_false = _helion_guard_or_false  # type: ignore[assignment]
guard_or_false = symbolic_shapes.guard_or_false  # keep module-level alias in sync

_orig_validate_block_shape = triton_utils.validate_block_shape


def _helion_validate_block_shape(shape: list[int]) -> int:
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]`"
            )
        numel *= d
    if numel > triton_utils.TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(
            f"numel ({numel}) exceeds triton maximum tensor numel "
            f"({triton_utils.TRITON_MAX_TENSOR_NUMEL})"
        )
    return numel


triton_utils.validate_block_shape = _helion_validate_block_shape  # type: ignore[assignment]
triton_core.validate_block_shape = _helion_validate_block_shape  # type: ignore[assignment]
