"""Current-storage guards for the selected horizontal descriptor wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast
import weakref

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


# Selected modules bind once at construction. The weak owner prevents search
# candidates from being retained, and lookup does not trust mutable attributes.
_contracts: dict[int, tuple[weakref.ReferenceType[object], object]] = {}


def _plan_snapshot(value: object) -> object:
    if type(value) in (bool, int, str) or value is None:
        return (type(value), value)
    if type(value) in (list, tuple):
        return (
            type(value),
            tuple(_plan_snapshot(x) for x in cast("Sequence[object]", value)),
        )
    if type(value) is dict:
        mapping = cast("dict[object, object]", value)
        if any(type(key) is not str for key in mapping):
            raise ValueError("horizontal plan has a non-string field")
        return (
            dict,
            tuple(
                (key, _plan_snapshot(mapping[key]))
                for key in sorted(cast("dict[str, object]", mapping))
            ),
        )
    raise ValueError("horizontal plan contains nonliteral authority")


def bind_horizontal_state_kernel(kernel: object) -> None:
    """Capture the selected module's literal plan after metadata construction."""
    plans = getattr(kernel, "_helion_cute_wrapper_plans", ())
    if (
        not isinstance(plans, (list, tuple))
        or len(plans) != 1
        or not isinstance(plans[0], dict)
        or plans[0].get("kind") != "horizontal_state_tma"
    ):
        raise ValueError("horizontal module has no exact selected plan")
    snapshot = _plan_snapshot(plans)
    identity = id(kernel)
    prior = _contracts.get(identity)
    if prior is not None:
        if prior[0]() is not kernel or prior[1] != snapshot:
            raise ValueError("horizontal module contract cannot be rebound")
        return

    def release(reference: weakref.ReferenceType[object]) -> None:
        current = _contracts.get(identity)
        if current is not None and current[0] is reference:
            _contracts.pop(identity)

    _contracts[identity] = (weakref.ref(kernel, release), snapshot)


def validate_horizontal_state_kernel(kernel: object, args: tuple[object, ...]) -> None:
    """Validate selected-module authority before either cached launch path."""
    plans = getattr(kernel, "_helion_cute_wrapper_plans", ())
    record = _contracts.get(id(kernel))
    if record is None:
        if isinstance(plans, (list, tuple)) and any(
            isinstance(plan, dict) and plan.get("kind") == "horizontal_state_tma"
            for plan in plans
        ):
            raise ValueError("horizontal module was not bound at construction")
        return
    if record[0]() is not kernel or record[1] != _plan_snapshot(plans):
        raise ValueError("horizontal module plan was removed or changed")
    validate_horizontal_state_arguments(
        cast("Sequence[dict[str, object]]", plans), args
    )


def _indices(value: object, count: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or any(
        type(index) is not int or not 0 <= index < count for index in value
    ):
        raise ValueError("horizontal tensor index schema changed")
    if len(value) != len(set(value)):
        raise ValueError("horizontal tensor indices are duplicated")
    return tuple(value)


def _integers(value: object, *, positive: bool) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or any(
        type(x) is not int or x < int(positive) for x in value
    ):
        raise ValueError("horizontal physical layout schema changed")
    return tuple(value)


def validate_horizontal_state_arguments(
    plans: Sequence[dict[str, object]], args: tuple[object, ...]
) -> None:
    for plan in plans:
        if plan.get("kind") != "horizontal_state_tma":
            continue
        indices = _indices(plan["metadata_indices"], len(args))
        writes = _indices(plan["write_indices"], len(args))
        reads = _indices(plan["readonly_indices"], len(args))
        aligned = _indices(plan["aligned_indices"], len(args))
        metadata = plan["metadata"]
        if not isinstance(metadata, (tuple, list)) or len(indices) != len(metadata):
            raise ValueError("horizontal tensor metadata schema changed")
        if (
            len(writes) != 2
            or set(indices) != set(writes) | set(reads)
            or set(writes) & set(reads)
            or not set(aligned) <= set(reads)
        ):
            raise ValueError("horizontal access ownership is incomplete")
        spans: dict[int, tuple[int, int]] = {}
        owners: dict[int, tuple[int, int]] = {}
        devices = set()
        for index, record in zip(indices, metadata, strict=True):
            if not isinstance(record, (tuple, list)) or len(record) != 3:
                raise ValueError("horizontal metadata record changed")
            shape = _integers(record[0], positive=True)
            stride = _integers(record[1], positive=False)
            dtype = record[2]
            tensor = args[index]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.is_conj()
                or tensor.is_neg()
            ):
                raise ValueError("horizontal argument is not a plain current tensor")
            if (
                len(shape) != len(stride)
                or tuple(tensor.shape) != shape
                or tuple(tensor.stride()) != stride
                or str(tensor.dtype) != dtype
            ):
                raise ValueError("horizontal shape/stride/dtype changed")
            pointer = tensor.data_ptr()
            owner = tensor.untyped_storage()
            end = (
                pointer
                + (
                    1
                    + sum(
                        (size - 1) * step
                        for size, step in zip(shape, stride, strict=True)
                    )
                )
                * tensor.element_size()
            )
            if (
                not owner.data_ptr()
                <= pointer
                < end
                <= owner.data_ptr() + owner.nbytes()
            ):
                raise ValueError("horizontal current tensor exceeds its storage")
            if pointer % tensor.element_size():
                raise ValueError("horizontal scalar pointer is misaligned")
            spans[index] = (pointer, end)
            owners[index] = (owner.data_ptr(), owner.data_ptr() + owner.nbytes())
            devices.add(tensor.device)
            if index in aligned and (
                pointer % 16
                or any(
                    size != 1 and step % 8
                    for size, step in zip(shape[:-1], stride[:-1], strict=True)
                )
            ):
                raise ValueError("horizontal vector packet alignment changed")
        if len(devices) != 1:
            raise ValueError("horizontal tensors have different devices")
        state_index = plan["state_idx"]
        if (
            type(state_index) is not int
            or state_index not in writes
            or spans[state_index][0] % 16
        ):
            raise ValueError("horizontal descriptor pointer is misaligned")
        state = args[state_index]
        assert isinstance(state, torch.Tensor)
        if tuple(state.shape) != _integers(plan["state_shape"], positive=True) or tuple(
            state.stride()
        ) != _integers(plan["state_strides"], positive=False):
            raise ValueError("horizontal descriptor layout differs from state owner")
        for index in writes:
            # The original narrowed scalar index can address a valid storage
            # prefix outside the positive logical view. Preserve that fallback
            # without moving a readonly access across any same-owner write.
            low, high = owners[index]
            for other in indices:
                if other != index and max(low, owners[other][0]) < min(
                    high, owners[other][1]
                ):
                    raise ValueError(
                        "horizontal mutable and reordered-read storage owners overlap"
                    )
