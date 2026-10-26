from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


def memory_index_coordinates(
    env: CompileEnvironment,
    indices: list[object],
    tensor_shapes: dict[int, tuple[int, ...]],
    output_shape: tuple[int, ...],
) -> list[tuple[int | None, ...]]:
    """Map each indexer's coordinates using the frontend's indexing rules.

    Tensor indices can share broadcast axes or contribute independent Cartesian
    axes. None entries in a mapping denote singleton coordinates, not new axes.
    This is an emission proof for one resolved configuration, not shape inference.
    """
    tensor_positions = list(tensor_shapes)
    broadcast = env.should_broadcast_tensor_indexers(indices)
    cartesian = all(len(shape) == 1 for shape in tensor_shapes.values())
    position = 0
    shared_start = 0
    shared_rank = 0
    mappings: list[tuple[int | None, ...]] = []
    for ordinal, index in enumerate(indices):
        if ordinal in tensor_shapes:
            shape = tensor_shapes[ordinal]
            if broadcast:
                if ordinal == tensor_positions[0]:
                    shared_start = position
                    tensors = [
                        cast("torch.Tensor", indices[i]) for i in tensor_positions
                    ]
                    shared_rank = len(env.tensor_indexer_broadcast_shape(tensors))
                    position += shared_rank
                start = (
                    shared_start + tensor_positions.index(ordinal)
                    if cartesian
                    else shared_start + shared_rank - len(shape)
                )
                mapping = tuple(
                    None if size == 1 else start + axis
                    for axis, size in enumerate(shape)
                )
            else:
                width = len(env.tensor_indexer_dims(cast("torch.Tensor", index)))
                non_singleton = [axis for axis, size in enumerate(shape) if size != 1]
                if len(non_singleton) != width and not (
                    not non_singleton
                    and width in (0, 1)
                    and (
                        width == 0
                        or position < len(output_shape)
                        and output_shape[position] == 1
                    )
                ):
                    raise exc.BackendUnsupported("cute", "fragment tensor index axes")
                axes = dict(
                    zip(non_singleton, range(position, position + width), strict=False)
                )
                mapping = tuple(axes.get(axis) for axis in range(len(shape)))
                position += width
            for size, axis in zip(shape, mapping, strict=True):
                # Exact static index axes can be smaller than a padded result.
                # The caller guards index reads by the fragment's logical domain.
                if axis is not None and (
                    not 0 <= axis < len(output_shape) or size > output_shape[axis]
                ):
                    raise exc.BackendUnsupported(
                        "cute", "fragment tensor index broadcast"
                    )
            mappings.append(mapping)
        elif (
            index is None
            or isinstance(index, slice)
            or isinstance(index, torch.SymInt)
            and (
                (bid := env.resolve_block_id(index)) is not None
                and index._sympy_() == env.block_sizes[bid].var._sympy_()
            )
        ):
            mappings.append((position,))
            position += 1
        else:
            mappings.append(())
    if position != len(output_shape):
        raise exc.BackendUnsupported("cute", "fragment indexed output axes")
    return mappings
