from __future__ import annotations

from typing import cast

import torch

from helion.autotuner.benchmark_provider import _clone_args


def _storage_id(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


def test_clone_args_preserves_shared_storage_views_and_duplicate_refs() -> None:
    storage = torch.arange(128, dtype=torch.uint8)
    int16_view = storage.view(torch.int16).as_strided((3, 4), (6, 1), 2)
    float32_view = storage[8:72].view(torch.float32)[::2]
    broadcast_view = storage[20:21].expand(3, 4)
    unrelated = torch.ones(4)
    args = ([int16_view, float32_view, int16_view, broadcast_view], unrelated)

    cloned = cast(
        "tuple[list[torch.Tensor], torch.Tensor]",
        _clone_args(args, process_group_name=None, idx_to_clone=(0,)),
    )
    cloned_views, cloned_unrelated = cloned

    assert cloned_views[0] is cloned_views[2]
    assert len({_storage_id(tensor) for tensor in cloned_views}) == 1
    assert _storage_id(cloned_views[0]) != _storage_id(int16_view)
    assert cloned_unrelated is unrelated
    for original, copy_ in zip(
        (int16_view, float32_view, int16_view, broadcast_view),
        cloned_views,
        strict=True,
    ):
        assert copy_.dtype is original.dtype
        assert copy_.shape == original.shape
        assert copy_.stride() == original.stride()
        assert copy_.storage_offset() == original.storage_offset()
        assert copy_.untyped_storage().nbytes() == original.untyped_storage().nbytes()

    original_bytes = storage.clone()
    cloned_views[0].fill_(0)
    assert torch.equal(storage, original_bytes)


def test_clone_args_creates_candidate_private_alias_groups() -> None:
    storage = torch.arange(64, dtype=torch.float32)
    even = storage[3:35:2]
    overlapping = storage[7:23]
    args = (even, overlapping)

    first = cast(
        "tuple[torch.Tensor, torch.Tensor]",
        _clone_args(args, process_group_name=None, idx_to_clone=(0,)),
    )
    second = cast(
        "tuple[torch.Tensor, torch.Tensor]",
        _clone_args(args, process_group_name=None, idx_to_clone=(0,)),
    )

    assert _storage_id(first[0]) == _storage_id(first[1])
    assert _storage_id(second[0]) == _storage_id(second[1])
    assert _storage_id(first[0]) != _storage_id(second[0])
    assert _storage_id(first[0]) != _storage_id(storage)
    first[1].zero_()
    assert not torch.equal(first[0], second[0])
    assert torch.equal(storage, torch.arange(64, dtype=torch.float32))
