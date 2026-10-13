"""Storage ownership and readonly-byte checks for the SSD objective."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import torch

from helion.autotuner.benchmark_objective import BenchmarkObjectiveError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _storage_bytes(tensor: torch.Tensor) -> torch.Tensor:
    storage = tensor.untyped_storage()
    return torch.empty(0, device=tensor.device, dtype=torch.uint8).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _tensor_abi(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _merged_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclasses.dataclass(frozen=True)
class _PrivateStorage:
    current: torch.Tensor
    pristine: torch.Tensor
    writable: tuple[tuple[int, int], ...]
    protected: tuple[tuple[int, int], ...]

    def reset(self) -> None:
        for start, end in self.writable:
            self.current[start:end].copy_(self.pristine[start:end])

    def validate(self) -> None:
        for start, end in self.protected:
            assert torch.equal(self.current[start:end], self.pristine[start:end]), (
                "SSD pipeline changed readonly input storage or a physical guard"
            )


def _private_storages(
    raw: Mapping[str, torch.Tensor], written_names: frozenset[str]
) -> tuple[_PrivateStorage, ...]:
    groups: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for name, value in raw.items():
        groups.setdefault(value.untyped_storage()._cdata, []).append((name, value))
    result = []
    for tensors in groups.values():
        current = _storage_bytes(tensors[0][1])
        intervals = []
        for name, value in tensors:
            if name in written_names:
                # All writable views in the frozen GDN contract are contiguous.
                # Readonly q/k/v and gate views may have arbitrary row strides.
                if not value.is_contiguous():
                    raise BenchmarkObjectiveError(
                        "SSD requires contiguous writable views"
                    )
                start = int(value.storage_offset()) * value.element_size()
                intervals.append(
                    (start, start + int(value.numel()) * value.element_size())
                )
        writable = _merged_intervals(intervals)
        protected = []
        cursor = 0
        for start, end in writable:
            if cursor < start:
                protected.append((cursor, start))
            cursor = end
        if cursor < current.numel():
            protected.append((cursor, current.numel()))
        result.append(
            _PrivateStorage(current, current.clone(), writable, tuple(protected))
        )
    return tuple(result)
