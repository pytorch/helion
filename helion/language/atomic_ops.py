from __future__ import annotations

import ast
from typing import Callable

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.indexing_strategy import SubscriptIndexing
from .._compiler.utils import normalize_index_slices
from . import _decorators

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
    index = Tile._tiles_to_sizes_for_index(index)
    index = normalize_index_slices(target.shape, index)
    return (target, index, *values, sem)


def _to_ast_values(values: list[object]) -> list[ast.AST]:
    out: list[ast.AST] = []
    for v in values:
        if isinstance(v, (int, float, bool)):
            out.append(expr_from_string(constant_repr(v)))
        else:
            assert isinstance(v, ast.AST)
            out.append(v)
    return out


def _refuse_device_load_target(target: torch.Tensor, op: str) -> None:
    from ..runtime.ref_mode import is_device_load

    if is_device_load(target):
        raise exc.AtomicOnDeviceTensor(op)


def _ref_atomic_binop(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float | bool,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Shared ref implementation for simple atomic binary ops (xchg/and/or/xor/max/min).

    Processes indices, clones the previous value, applies the op, and returns prev.
    For xchg, pass op=lambda old, val: val.
    """
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
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=target.dtype, device=target.device)
    )
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = op(target[idx_tuple], val)
    return prev


def _ref_lane_coords(target: torch.Tensor, index: list[object]) -> list[torch.Tensor]:
    """Each lane's coordinate along every dim of ``target``, laid out like
    the compiled result of ``target[index]``: several 1-D index tensors
    form a cartesian product, one dim each in place, while other index
    tensors broadcast together into dims at the first one's place.  Negative
    indices wrap and out-of-range ones raise, as in eager indexing."""
    tensors = [idx for idx in index if isinstance(idx, torch.Tensor)]
    cartesian = len(tensors) > 1 and all(t.ndim == 1 for t in tensors)
    group = () if cartesian else torch.broadcast_shapes(*(t.shape for t in tensors))
    placed: list[tuple[torch.Tensor, int]] = []
    rank = 0
    group_start: int | None = None
    for dim, (idx, size) in enumerate(zip(index, target.shape, strict=True)):
        if isinstance(idx, slice):
            placed.append(
                (torch.arange(*idx.indices(size), device=target.device), rank)
            )
            rank += 1
            continue
        coord = torch.as_tensor(idx, device=target.device, dtype=torch.int64)
        if bool(((coord < -size) | (coord >= size)).any()):
            raise IndexError(f"index out of range for dim {dim} of size {size}")
        coord = torch.where(coord < 0, coord + size, coord)
        if not isinstance(idx, torch.Tensor):
            placed.append((coord, rank))
        elif cartesian:
            placed.append((coord, rank))
            rank += 1
        else:
            if group_start is None:
                group_start = rank
                rank += len(group)
            placed.append((coord.expand(group), group_start))
    return list(
        torch.broadcast_tensors(
            *(
                coord.view(
                    *[1] * start, *coord.shape, *[1] * (rank - start - coord.ndim)
                )
                for coord, start in placed
            )
        )
    )


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
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.host_function import HostFunction

    if (
        CompileEnvironment.current().backend.name == "cute"
        and target not in HostFunction.current().tensor_to_origin
        and target.dtype == torch.int32
        and target.ndim == 1
        and len(index) == 1
        and isinstance(index[0], torch.Tensor)
    ):
        # Local fetch-add returns one old value per logical index. Normalizing
        # a concrete index extent to a padded reduction block here changes the
        # meaning of downstream views (for example, flip reverses padding).
        return torch.empty_like(index[0], dtype=target.dtype)
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
    _refuse_device_load_target(target, "atomic_add")
    from .ref_tile import RefTile

    processed_index: list[object] = []
    has_tensor_index = False
    for idx in index:
        if isinstance(idx, RefTile):
            processed_index.append(idx._slice)
        elif isinstance(idx, torch.Tensor):
            if idx.ndim == 0:
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
        # Add lane by lane, in the result's layout, so repeated indices
        # accumulate and each lane returns the value its own add replaced.
        coords = _ref_lane_coords(target, processed_index)
        shape = coords[0].shape
        values = _convert_value_to_target_dtype(value).expand(shape).reshape(-1)
        prev = target.new_empty(shape)
        flat_prev = prev.view(-1)
        lanes = zip(*(coord.reshape(-1).tolist() for coord in coords), strict=True)
        for flat, element in enumerate(lanes):
            flat_prev[flat] = target[element]
            target[element] += values[flat]
        return prev

    idx_tuple = tuple(processed_index)
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
    val_tensor = _convert_value_to_target_dtype(value)
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = target[idx_tuple] + val_tensor
    return prev


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
    _refuse_device_load_target(target, "atomic_xchg")
    return _ref_atomic_binop(target, index, value, lambda old, val: val)


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
    _refuse_device_load_target(target, "atomic_and")
    return _ref_atomic_binop(target, index, value, torch.bitwise_and)


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
    _refuse_device_load_target(target, "atomic_or")
    return _ref_atomic_binop(target, index, value, torch.bitwise_or)


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
    _refuse_device_load_target(target, "atomic_xor")
    return _ref_atomic_binop(target, index, value, torch.bitwise_xor)


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
) -> torch.Tensor:
    _validate_sem(sem)
    _refuse_device_load_target(target, "atomic_max")
    return _ref_atomic_binop(target, index, value, torch.maximum)


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
    _refuse_device_load_target(target, "atomic_min")
    return _ref_atomic_binop(target, index, value, torch.minimum)


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
        Triton CAS has no masked form.  A lane of a tile past the target's
        extent instead swaps an in-bounds element (its indices clamped to
        each dim's last element) for that lane's expected value, which
        leaves memory unchanged; its returned value is meaningless.
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
    _refuse_device_load_target(target, "atomic_cas")
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
    # pyrefly: ignore [bad-index]
    prev = target[idx_tuple].clone()
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
    # pyrefly: ignore [bad-index]
    mask = target[idx_tuple] == exp_t
    # pyrefly: ignore [bad-index, unsupported-operation]
    target[idx_tuple] = torch.where(mask, val_t, target[idx_tuple])
    return prev


ATOMIC_OPS: frozenset[Callable[..., object]] = frozenset(
    {
        atomic_add,
        atomic_and,
        atomic_cas,
        atomic_max,
        atomic_min,
        atomic_or,
        atomic_xchg,
        atomic_xor,
    }
)
