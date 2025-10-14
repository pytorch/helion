from __future__ import annotations

import ast
import itertools
from typing import TYPE_CHECKING
from typing import Callable

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.host_function import HostFunction
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = [
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
]


_VALID_SEMS: set[str] = {"relaxed", "acquire", "release", "acq_rel"}


def _validate_sem(sem: str) -> None:
    if sem not in _VALID_SEMS:
        raise exc.InternalError(
            ValueError(
                f"Invalid memory semantic '{sem}'. Valid options are: relaxed, acquire, release, acq_rel"
            )
        )


def _prepare_mem_args(
    target: torch.Tensor,
    index: list[object],
    *values: object,
    sem: str = "relaxed",
) -> tuple:
    from .tile_proxy import Tile

    _validate_sem(sem)
    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)
    return (target, index, *values, sem)


def _codegen_common(
    tl_func: str, state: CodegenState, value_exprs: list[ast.AST]
) -> ast.AST:
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    host_function = HostFunction.current()
    if target not in host_function.tensor_to_origin:
        raise exc.AtomicOnDeviceTensor(tl_func)

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    placeholder_names = [f"v{i}" for i in range(len(value_exprs))]
    values_section = (
        ", " + ", ".join([f"{{{n}}}" for n in placeholder_names]) if value_exprs else ""
    )
    placeholders = dict(zip(placeholder_names, value_exprs, strict=False))
    return expr_from_string(
        f"tl.{tl_func}({name} + {{offset}}{values_section}, mask={{mask}}, sem={{sem}})",
        offset=indices.index_expr,
        mask=indices.mask_expr,
        sem=sem,
        **placeholders,
    )


def _to_ast_values(values: list[object]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for v in values:
        if isinstance(v, (int, float, bool)):
            out.append(expr_from_string(constant_repr(v)))
        else:
            assert isinstance(v, ast.AST)
            out.append(v)
    return out


def _ref_apply(
    target: torch.Tensor,
    index: list[object],
    apply_fn: Callable[[torch.Tensor, tuple, object], None],
    value: object,
) -> None:
    from .ref_tile import RefTile

    # Convert indices to proper format
    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)

    # Find tensor indices that need element-wise processing
    tensor_indices = [
        (i, idx)
        for i, idx in enumerate(processed_index)
        if isinstance(idx, torch.Tensor) and idx.numel() > 1
    ]

    if tensor_indices:
        # Element-wise processing for tensor indices (handle first tensor index)
        i, tensor_idx = tensor_indices[0]

        if tensor_idx.ndim == 0:
            coords_iter = [()]
        else:
            ranges = [range(dim) for dim in tensor_idx.shape]
            coords_iter = itertools.product(*ranges)

        for coords in coords_iter:
            elem = tensor_idx[coords].item()
            new_index = processed_index.copy()
            new_index[i] = int(elem)
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                next_value = value[coords]
            else:
                next_value = value
            _ref_apply(target, new_index, apply_fn, next_value)
    else:
        apply_fn(target, tuple(processed_index), value)


# -- atomic_add --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_add(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically add a value to a target tensor.

    Performs an atomic read-modify-write that adds ``value`` to
    ``target[index]``. This is safe for concurrent access from multiple
    threads/blocks.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to add (tensor or scalar).
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.

    Example:
        @helion.kernel
        def global_sum(x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                hl.atomic_add(result, [0], x[tile].sum())
            return result

    Notes:
        - Use for race-free accumulation across parallel execution.
        - Higher memory semantics may reduce performance.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, torch.Tensor | float | int, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_add)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    # Convert indices for shape computation and fast path detection
    processed_index: list[object] = []
    has_tensor_index = False
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor):
            if idx.numel() == 1:
                processed_index.append(int(idx.item()))
            else:
                processed_index.append(idx)
                has_tensor_index = True
        else:
            processed_index.append(idx)

    def _convert_value_to_target_dtype(val: object) -> torch.Tensor:
        if isinstance(val, torch.Tensor):
            vt = val.to(device=target.device)
            if vt.dtype != target.dtype:
                vt = vt.to(dtype=target.dtype)
            return vt
        return torch.as_tensor(val, dtype=target.dtype, device=target.device)

    if has_tensor_index:
        ret_shape = SubscriptIndexing.compute_shape(target, processed_index)
        prev_chunks: list[torch.Tensor] = []

        def apply(t: torch.Tensor, idx_tuple: tuple, v: object) -> None:
            prev_val = t[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
            val_tensor = _convert_value_to_target_dtype(v)
            t[idx_tuple] = t[idx_tuple] + val_tensor  # pyright: ignore[reportArgumentType]
            prev_chunks.append(prev_val.reshape(-1))

        _ref_apply(target, index, apply, value)
        if prev_chunks:
            flat_prev = torch.cat(prev_chunks)
        else:
            flat_prev = target.new_empty(0, dtype=target.dtype, device=target.device)
        return flat_prev.reshape(ret_shape)

    idx_tuple = tuple(processed_index)
    prev = target[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
    val_tensor = _convert_value_to_target_dtype(value)
    target[idx_tuple] = target[idx_tuple] + val_tensor  # pyright: ignore[reportArgumentType]
    return prev


@_decorators.codegen(atomic_add)
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_add", state, _to_ast_values([value_expr]))


# -- atomic_xchg --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_xchg(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically exchange (set) a value at ``target[index]``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: New value(s) to set.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_xchg)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_xchg)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_xchg)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    prev = target[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    target[idx_tuple] = val  # pyright: ignore[reportArgumentType]
    return prev


@_decorators.codegen(atomic_xchg)
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xchg", state, _to_ast_values([value_expr]))


# -- atomic_and/or/xor --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_and(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise AND with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to AND with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_and)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_and)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_and)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    prev = target[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    target[idx_tuple] = target[idx_tuple] & val  # pyright: ignore[reportArgumentType]
    return prev


@_decorators.codegen(atomic_and)
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_and", state, _to_ast_values([value_expr]))


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_or(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise OR with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to OR with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_or)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_or)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_or)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    prev = target[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    target[idx_tuple] = target[idx_tuple] | val  # pyright: ignore[reportArgumentType]
    return prev


@_decorators.codegen(atomic_or)
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_or", state, _to_ast_values([value_expr]))


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_xor(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically apply bitwise XOR with ``value`` to ``target[index]``.

    Args:
        target: Tensor to update (integer/bool dtype).
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to XOR with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_xor)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_xor)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_xor)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | int | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    prev = target[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    target[idx_tuple] = target[idx_tuple] ^ val  # pyright: ignore[reportArgumentType]
    return prev


@_decorators.codegen(atomic_xor)
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_xor", state, _to_ast_values([value_expr]))


# -- atomic_max/min --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_max(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically update ``target[index]`` with the maximum of current value
    and ``value``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to compare with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_max)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_max)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_max)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> None:
    _validate_sem(sem)

    def apply(t: torch.Tensor, idx: tuple, v: object) -> None:
        t[idx] = torch.maximum(
            t[idx], torch.as_tensor(v, dtype=t[idx].dtype, device=t.device)
        )  # pyright: ignore[reportArgumentType]

    _ref_apply(target, index, apply, value)


@_decorators.codegen(atomic_max)
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_max", state, _to_ast_values([value_expr]))


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_min(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically update ``target[index]`` with the minimum of current value
    and ``value``.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        value: Value(s) to compare with.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
        ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_min)
def _(
    target: torch.Tensor, index: list[object], value: object, sem: str = "relaxed"
) -> tuple[torch.Tensor, object, object, str]:
    return _prepare_mem_args(target, index, value, sem=sem)


@_decorators.register_fake(atomic_min)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_min)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    prev = target[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    target[idx_tuple] = torch.minimum(target[idx_tuple], val)  # pyright: ignore[reportArgumentType]
    return prev


@_decorators.codegen(atomic_min)
def _(state: CodegenState) -> ast.AST:
    value_expr = state.ast_args[2]
    return _codegen_common("atomic_min", state, _to_ast_values([value_expr]))


# -- atomic_cas --


@has_side_effect
@_decorators.api(allow_host_tensor=True, tiles_as_sizes=True)
def atomic_cas(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor | float | bool,
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    """
    Atomically compare-and-swap a value at ``target[index]``.

    If the current value equals ``expected``, writes ``value``. Otherwise
    leaves memory unchanged.

    Args:
        target: Tensor to update.
        index: Indices selecting elements to update. Can include tiles.
        expected: Expected current value(s) used for comparison.
        value: New value(s) to write if comparison succeeds.
        sem: Memory ordering semantics. One of ``"relaxed"``, ``"acquire"``,
            ``"release"``, ``"acq_rel"``. Defaults to ``"relaxed"``.

    Returns:
        torch.Tensor: The previous value(s) stored at ``target[index]`` before the compare-and-swap.

    Note:
        Triton CAS doesn’t support a masked form; our generated code uses
        an unmasked CAS and relies on index masking to avoid OOB.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: object,
    value: object,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, object, object, str]:
    return _prepare_mem_args(target, index, expected, value, sem=sem)


@_decorators.register_fake(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor,
    value: torch.Tensor,
    sem: str = "relaxed",
) -> torch.Tensor:
    target_shape = SubscriptIndexing.compute_shape(target, index)
    return target.new_empty(target_shape)


@_decorators.ref(atomic_cas)
def _(
    target: torch.Tensor,
    index: list[object],
    expected: torch.Tensor | float | bool,
    value: torch.Tensor | float | bool,
    sem: str = "relaxed",
) -> torch.Tensor:
    _validate_sem(sem)
    from .ref_tile import RefTile

    processed_index: list[object] = []
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor) and idx.numel() == 1:
            processed_index.append(int(idx.item()))
        else:
            processed_index.append(idx)
    idx_tuple = tuple(processed_index)
    prev = target[idx_tuple].clone()  # pyright: ignore[reportArgumentType]
    exp_t = (
        expected
        if isinstance(expected, torch.Tensor)
        else torch.as_tensor(expected, dtype=target.dtype, device=target.device)
    )
    val_t = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    mask = target[idx_tuple] == exp_t  # pyright: ignore[reportArgumentType]
    target[idx_tuple] = torch.where(mask, val_t, target[idx_tuple])  # pyright: ignore[reportArgumentType]
    return prev


@_decorators.codegen(atomic_cas)
def _(state: CodegenState) -> ast.AST:
    exp_expr = state.ast_args[2]
    val_expr = state.ast_args[3]
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(len(state.ast_args) - 1)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    exp_ast, val_ast = _to_ast_values([exp_expr, val_expr])
    return expr_from_string(
        f"tl.atomic_cas({name} + {{offset}}, {{exp}}, {{val}}, sem={{sem}})",
        offset=indices.index_expr,
        exp=exp_ast,
        val=val_ast,
        sem=sem,
    )
