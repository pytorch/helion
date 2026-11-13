"""Ordinary tensor helpers exercising static logical-coordinate indexing."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def ordered_tree(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(values.shape[0], 4, 32)
    values = values[:, :, :16] + values[:, :, 16:]
    values = values[:, :, :8] + values[:, :, 8:]
    values = values[:, :, :4] + values[:, :, 4:]
    values = values[:, :, :2] + values[:, :, 2:]
    values = values[:, :, 0] + values[:, :, 1]
    return ((values[:, 0] + values[:, 1]) + values[:, 2]) + values[:, 3]


def mixed_views(values: torch.Tensor) -> torch.Tensor:
    computed = (values * 3 + 1).reshape(values.shape[0], 8, 16).transpose(1, 2)
    selected = computed[:, -12:14:2, -7::3]
    return selected[:, :, -1] + computed[:, 4:14:2, 1]


def plain_halves(values: torch.Tensor) -> torch.Tensor:
    computed = values * 2
    return computed[:, :64] + computed[:, 64:]


def plain_select(values: torch.Tensor) -> torch.Tensor:
    return (values * 2)[:, 0]


def plain_last(values: torch.Tensor) -> torch.Tensor:
    return (values * 2)[:, -1]


def shape_only_view(
    values: torch.Tensor, select_singleton: bool | hl.constexpr
) -> torch.Tensor:
    computed = values * 2
    if select_singleton:
        return computed[:, 0]
    return computed[:, 0 : values.shape[1]]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_shape_only(
    x: torch.Tensor, select_singleton: hl.constexpr
) -> torch.Tensor:
    width = hl.specialize(x.shape[1])
    shape = x.shape[0] if select_singleton else x.shape
    out = torch.empty(shape, device=x.device, dtype=x.dtype)
    for row in hl.tile(x.shape[0], block_size=1):
        values = shape_only_view(x[row, hl.arange(width)], select_singleton)
        if select_singleton:
            out[row] = values
        else:
            out[row, hl.arange(width)] = values
    return out


def register_narrow(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.maximum(values, torch.gather(values, 0, indices))[:, ::2]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_tree(x: torch.Tensor) -> torch.Tensor:
    width = hl.specialize(x.shape[1])
    assert width == 128
    out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    for row in hl.tile(x.shape[0], block_size=1):
        out[row] = ordered_tree(x[row, hl.arange(width)])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_mixed(x: torch.Tensor) -> torch.Tensor:
    width = hl.specialize(x.shape[1])
    assert width == 128
    out = torch.empty((x.shape[0], 5), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.shape[0], block_size=1):
        out[row, hl.arange(5)] = mixed_views(x[row, hl.arange(width)])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_halves(x: torch.Tensor) -> torch.Tensor:
    width = hl.specialize(x.shape[1])
    assert width == 128
    out = torch.empty((x.shape[0], 64), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.shape[0], block_size=1):
        out[row, hl.arange(64)] = plain_halves(x[row, hl.arange(width)])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_select(x: torch.Tensor) -> torch.Tensor:
    width = hl.specialize(x.shape[1])
    assert width == 128
    out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    for row in hl.tile(x.shape[0], block_size=1):
        out[row] = plain_select(x[row, hl.arange(width)])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_full_axis(x: torch.Tensor, select_last: hl.constexpr) -> torch.Tensor:
    shape = x.shape[0] if select_last else (x.shape[0], 64)
    out = torch.empty(shape, device=x.device, dtype=x.dtype)
    for row in hl.tile(x.shape[0], block_size=1):
        if select_last:
            out[row] = plain_last(x[row, :])
        else:
            out[row, hl.arange(64)] = plain_halves(x[row, :])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_register(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (x.size(0), x.size(1), x.size(2) // 2), device=x.device, dtype=x.dtype
    )
    for row in hl.grid(x.size(0)):
        groups = hl.arange(x.size(1))[:, None]
        indices = (groups ^ 1).expand(-1, x.size(2))
        out[row, :, :] = register_narrow(x[row, :, :], indices)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_full_axis_first(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    for row in hl.tile(x.shape[0], block_size=1):
        out[row] = plain_select(x[row, :])
    return out


def identity_full_axis(
    values: torch.Tensor, explicit_end: hl.constexpr
) -> torch.Tensor:
    return values[:, :127] if explicit_end else values[:, :]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def static_index_full_axis_identity(
    x: torch.Tensor, explicit_end: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.shape[0], block_size=1):
        out[row, :] = identity_full_axis(x[row, :] * 2, explicit_end)
    return out
