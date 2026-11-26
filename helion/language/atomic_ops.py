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

    Writes ``op(old, value)`` at ``target[index]`` through ``_ref_update``,
    which returns the old values. For xchg, pass op=lambda old, val: val:
    each of these ops is associative.
    """
    return _ref_update(target, index, op, _ref_operand(target, value), associative=True)


def _ref_operand(target: torch.Tensor, value: object) -> torch.Tensor:
    """``value`` as a tensor in ``target``'s dtype, as Triton's and the CuTe
    backend's atomics convert it before they combine it with the element: an
    int64 value of 2**32 + 1 is 1 to an int32 target."""
    if isinstance(value, torch.Tensor):
        return value.to(device=target.device, dtype=target.dtype)
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -(2**63) <= value < 2**63
    ):
        # Through int64, which wraps into the target's dtype (0xFFFFFFFF is
        # -1 to int32), where as_tensor refuses a number past its range.
        wide = torch.as_tensor(value, dtype=torch.int64, device=target.device)
        return wide.to(target.dtype)
    return torch.as_tensor(value, dtype=target.dtype, device=target.device)


def _ref_add(old: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``old + value``; for uint16, uint32 and uint64, whose add has no CPU
    kernel, the add of their bits as a signed dtype of the same width, which
    gives the same bits."""
    bits = _UNSIGNED_BITS.get(old.dtype)
    if bits is None:
        return torch.add(old, value)
    return torch.add(old.view(bits), value.view(bits)).view(old.dtype)


def _ref_update(
    target: torch.Tensor,
    index: list[object],
    update: Callable[..., torch.Tensor],
    *operands: torch.Tensor,
    associative: bool = False,
    eager_indices: bool = False,
) -> torch.Tensor:
    """Write ``update(old, *operands)`` at ``target[index]`` and return the
    old values, as an atomic does lane by lane.

    The index takes a tile as its slice and a 0-d tensor as its number, as
    ref mode subscripts with them; an index tensor of another shape keeps
    its dims, as in compiled code. Lanes that name one element of the
    target update it in turn, each finding what the lanes before it left
    (``_ref_in_turn``), as the compiled atomics do, where one setitem keeps
    one lane's update or is refused (more than one element of the
    written-to tensor refers to a single memory location). Such lanes come
    from an index tensor that repeats an element, or from strides that
    overlap: both lanes of a row of ``out.as_strided((32, 2), (1, 0))`` name
    one element. The lanes of an index tensor are laid out as compiled code
    lays them out (``_ref_lane_coords``): its elements out of range raise
    with ``eager_indices``, and are otherwise dropped, as a subscript store
    drops them. A bool index tensor is taken as the offsets 0 and 1, as
    compiled code takes it, not as a mask. Any other index takes one
    setitem, unless the strides of ``target[index]`` may overlap
    (``_ref_may_share``).
    """
    from .ref_tile import RefTile

    tiles = [isinstance(idx, RefTile) for idx in index]
    index = [
        idx._slice
        if isinstance(idx, RefTile)
        else int(idx.item())
        if isinstance(idx, torch.Tensor) and idx.ndim == 0
        else idx
        for idx in index
    ]
    if not any(isinstance(idx, torch.Tensor) for idx in index):
        idx_tuple = tuple(index)
        # pyrefly: ignore [bad-index]
        view = target[idx_tuple]
        if not _ref_may_share(view):
            prev = view.clone()
            # Each entry but a number takes one dim of the view, in order.
            tile_dims = {
                dim
                for dim, tile in enumerate(
                    tile
                    for idx, tile in zip(index, tiles, strict=True)
                    if not isinstance(idx, int)
                )
                if tile
            }
            operands = tuple(
                _ref_broadcast(operand, view.shape, tile_dims) for operand in operands
            )
            # pyrefly: ignore [bad-index, unsupported-operation]
            target[idx_tuple] = update(view, *operands)
            return prev
    coords, keep, tile_dims = _ref_lane_coords(
        target, index, drop_out_of_range=not eager_indices, tiles=tiles
    )
    offsets = torch.full_like(coords[0], target.storage_offset())
    for coord, stride in zip(coords, target.stride(), strict=True):
        offsets = offsets + coord * stride
    operands = tuple(
        _ref_broadcast(operand, offsets.shape, tile_dims) for operand in operands
    )
    return _ref_in_turn(target, offsets, keep, update, operands, associative)


def _ref_may_share(view: torch.Tensor) -> bool:
    """Whether two elements of ``view`` may be one element of its storage, by
    its strides alone. They cannot when, its dims of size > 1 ordered by
    stride, each stride passes the furthest offset the smaller ones reach,
    as the place values of a number do: no stride 0, and no overlap."""
    reach = 0
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(view.shape, view.stride(), strict=True)
        if size > 1
    ):
        if stride <= reach:
            return True
        reach += (size - 1) * stride
    return False


def _ref_in_turn(
    target: torch.Tensor,
    offsets: torch.Tensor,
    keep: torch.Tensor | None,
    update: Callable[..., torch.Tensor],
    operands: tuple[torch.Tensor, ...],
    associative: bool,
) -> torch.Tensor:
    """Apply an atomic at the elements of ``target``'s storage at
    ``offsets``, one per lane in their layout, and return each lane's old
    value. The lanes in ``keep`` (every lane when None) write
    ``update(old, *their operands)`` in lane order at each element, so each
    finds what the lanes before it left; the others only read.

    An associative update (``update(update(a, b), c) == update(a,
    update(b, c))``, exactly) takes a scan: the lanes grouped by element, a
    doubling step combines each lane's operand with those of the lanes
    before it, in a number of steps that grows with the log of the most
    lanes at one element, and each lane's old value is the element's
    combined with those before it. Any other update takes one round per
    lane at the element most lanes share. The gathers and scatters are at
    offsets inside the storage, without the bounds handling of ref mode's
    subscripts."""
    # A uint16, uint32 or uint64 target is gathered, scattered and selected
    # through a signed view of the same width, as CPU kernels for those
    # dtypes are few (index_select and index_put_ have none in PyTorch 2.13);
    # only the update sees the target's dtype.
    bits = _UNSIGNED_BITS.get(target.dtype, target.dtype)
    storage = target.as_strided(
        (target.untyped_storage().nbytes() // target.element_size(),), (1,), 0
    ).view(bits)

    def apply(old: torch.Tensor, *values: torch.Tensor) -> torch.Tensor:
        if bits is target.dtype:
            return update(old, *values)
        unsigned = (value.view(target.dtype) for value in (old, *values))
        return update(*unsigned).view(bits)

    lane_offsets = offsets.reshape(-1)
    prev = storage.index_select(0, lane_offsets)
    lanes = torch.arange(lane_offsets.numel(), device=target.device)
    if keep is not None:
        lanes = lanes.masked_select(keep.reshape(-1))
    if lanes.numel() == 0:
        return prev.view(target.dtype).view(offsets.shape)
    # The kept lanes grouped by element, in lane order within each group.
    lanes = lanes.index_select(
        0, torch.argsort(lane_offsets.index_select(0, lanes), stable=True)
    )
    elements = lane_offsets.index_select(0, lanes)
    values = [
        operand.view(bits).reshape(-1).index_select(0, lanes) for operand in operands
    ]
    true = torch.ones(1, dtype=torch.bool, device=target.device)
    first = torch.cat([true, elements.diff() != 0])
    position = torch.arange(lanes.numel(), device=target.device)
    rank = position - torch.cummax(torch.where(first, position, 0), 0).values
    turns = int(rank.max()) + 1
    if associative:
        (scan,) = values
        step = 1
        while step < turns:
            later = scan.narrow(0, step, scan.numel() - step)
            combined = apply(scan.narrow(0, 0, scan.numel() - step), later)
            scan = torch.cat(
                [
                    scan.narrow(0, 0, step),
                    torch.where(
                        rank.narrow(0, step, rank.numel() - step) >= step,
                        combined,
                        later,
                    ),
                ]
            )
            step *= 2
        original = storage.index_select(0, elements)
        before = torch.cat([scan.narrow(0, 0, 1), scan.narrow(0, 0, scan.numel() - 1)])
        prev.index_put_(
            (lanes,), torch.where(rank == 0, original, apply(original, before))
        )
        last = torch.cat([first.narrow(0, 1, first.numel() - 1), true]).nonzero()
        last = last.reshape(-1)
        final = apply(original.index_select(0, last), scan.index_select(0, last))
        storage.index_put_((elements.index_select(0, last),), final)
        return prev.view(target.dtype).view(offsets.shape)
    # The lanes ordered by turn, so that each round takes a run of them.
    by_turn = torch.argsort(rank, stable=True)
    elements = elements.index_select(0, by_turn)
    lanes = lanes.index_select(0, by_turn)
    values = [value.index_select(0, by_turn) for value in values]
    olds: list[torch.Tensor] = []
    begin = 0
    for end in torch.bincount(rank, minlength=turns).cumsum(0).tolist():
        at = elements.narrow(0, begin, end - begin)
        old = storage.index_select(0, at)
        olds.append(old)
        storage.index_put_(
            (at,),
            apply(old, *(value.narrow(0, begin, end - begin) for value in values)),
        )
        begin = end
    prev.index_put_((lanes,), torch.cat(olds))
    return prev.view(target.dtype).view(offsets.shape)


# The signed dtype of the same width as each unsigned one.
_UNSIGNED_BITS = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
}


def _ref_broadcast(
    value: torch.Tensor, shape: torch.Size, tile_dims: set[int]
) -> torch.Tensor:
    """``value`` broadcast to the lanes' ``shape``: without the extra leading
    dims of size 1 that a subscript store drops, and cut to the lanes along
    the ``tile_dims`` where a tile's padded value runs past them, as a
    subscript store cuts it (``_trim_to_tiles``). A value that runs past the
    lanes along any other dim is refused, as compiled code refuses it."""
    from ..runtime.ref_mode import RefModeTorchFunctionMode

    if value.shape == shape:
        return value
    extra = value.ndim - len(shape)
    if extra > 0:
        value = value.reshape(value.shape[extra:])
    lead = len(shape) - value.ndim
    value = RefModeTorchFunctionMode._trim_to_tiles(
        value, shape[lead:], [dim - lead for dim in tile_dims if dim >= lead]
    )
    return value.expand(shape)


def _ref_lane_coords(
    target: torch.Tensor,
    index: list[object],
    *,
    drop_out_of_range: bool = False,
    tiles: list[bool],
) -> tuple[list[torch.Tensor], torch.Tensor | None, set[int]]:
    """Each lane's coordinate along every dim of ``target``, laid out like
    the compiled result of ``target[index]``: several 1-D index tensors
    form a cartesian product, one dim each in place, while other index
    tensors broadcast together into dims at the first one's place; a
    ``None`` puts a dim of size 1 there, and the dims that ``index`` leaves
    out at the end are taken whole.  Negative indices wrap and out-of-range
    ones raise, as in eager indexing.  With ``drop_out_of_range``, the
    elements of an index tensor outside its dim, the negative ones among
    them, instead leave their lanes out of the returned mask, at the
    clamped coordinate a load in a device loop reads, as a subscript store
    there drops them; the mask is None where nothing can be dropped (no
    index tensor, or eager rules). Last comes the set of the dims that the
    slices ``tiles`` marks as tiles take."""
    tensors = [idx for idx in index if isinstance(idx, torch.Tensor)]
    cartesian = len(tensors) > 1 and all(t.ndim == 1 for t in tensors)
    group = () if cartesian else torch.broadcast_shapes(*(t.shape for t in tensors))
    indexed = sum(idx is not None for idx in index)
    if indexed > target.ndim:
        raise IndexError(f"{indexed} indices for {target.ndim} dims")
    placed: list[tuple[torch.Tensor, int, torch.Tensor | None]] = []
    rank = 0
    group_start: int | None = None
    dim = -1
    tile_dims: set[int] = set()
    trailing = [False] * (target.ndim - indexed)
    for idx, tile in zip(
        [*index, *[slice(None)] * len(trailing)], [*tiles, *trailing], strict=True
    ):
        if idx is None:
            rank += 1
            continue
        dim += 1
        size = target.size(dim)
        if isinstance(idx, slice):
            placed.append(
                (torch.arange(*idx.indices(size), device=target.device), rank, None)
            )
            if tile:
                tile_dims.add(rank)
            rank += 1
            continue
        if isinstance(idx, torch.Tensor) and (
            idx.dtype.is_floating_point or idx.dtype.is_complex
        ):
            raise IndexError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )
        coord = torch.as_tensor(idx, device=target.device, dtype=torch.int64)
        inside: torch.Tensor | None = None
        if drop_out_of_range and isinstance(idx, torch.Tensor):
            inside = (coord >= 0) & (coord < size)
            coord = coord.clamp(0, size - 1)
        else:
            if bool(((coord < -size) | (coord >= size)).any()):
                raise IndexError(f"index out of range for dim {dim} of size {size}")
            coord = torch.where(coord < 0, coord + size, coord)
        start = rank
        if isinstance(idx, torch.Tensor):
            if cartesian:
                rank += 1
            else:
                if group_start is None:
                    group_start = rank
                    rank += len(group)
                start = group_start
                coord = coord.expand(group)
                if inside is not None:
                    inside = inside.expand(group)
        placed.append((coord, start, inside))

    def place(tensor: torch.Tensor, start: int) -> torch.Tensor:
        return tensor.view(
            *[1] * start, *tensor.shape, *[1] * (rank - start - tensor.ndim)
        )

    coords = list(
        torch.broadcast_tensors(*(place(coord, start) for coord, start, _ in placed))
    )
    keep: torch.Tensor | None = None
    for _, start, inside in placed:
        if inside is not None:
            mask = place(inside, start).expand(coords[0].shape)
            keep = mask if keep is None else keep & mask
    return coords, keep, tile_dims


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

    local = CompileEnvironment.current().backend.local_fetch_add_fake_shape(
        target, index
    )
    if local is not None:
        return local
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
    return _ref_update(
        target,
        index,
        _ref_add,
        _ref_operand(target, value),
        # A floating add rounds after each lane, which no other grouping
        # keeps: 2**24 + 1 + 1 rounds back to 2**24 at each lane, where
        # 2**24 + (1 + 1) is 2**24 + 2.
        associative=not (target.dtype.is_floating_point or target.dtype.is_complex),
        eager_indices=True,
    )


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
    return _ref_update(
        target,
        index,
        lambda old, exp, val: torch.where(old == exp, val, old),
        _ref_operand(target, expected),
        _ref_operand(target, value),
    )


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
