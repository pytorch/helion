"""Trusted worker files retain tensor metadata and aliases across dtype views."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion.autotuner.handoff_inputs import _load_inputs
from helion.autotuner.handoff_inputs import _save_inputs
from helion.autotuner.tensor_serialization import save_tensor_tree

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.float16, torch.float32, torch.complex64]
)
def test_mixed_dtype_views_preserve_storage_and_mutations(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    data = torch.arange(32, dtype=torch.float32).to(dtype)
    view = data.as_strided((3, 4), (8, 2), 3)
    byte_view = data.view(torch.uint8)[2::3]
    path = tmp_path / "args.pt"
    save_tensor_tree((view, byte_view, view), path)
    restored, restored_bytes, repeated = torch.load(path, weights_only=False)
    assert restored is repeated
    for before, after in ((view, restored), (byte_view, restored_bytes)):
        torch.testing.assert_close(before, after)
        assert before.stride() == after.stride()
        assert before.storage_offset() == after.storage_offset()
        assert before.untyped_storage().nbytes() == after.untyped_storage().nbytes()
    assert restored.untyped_storage()._cdata == restored_bytes.untyped_storage()._cdata
    # Mutations through either dtype retain the alias relationship.
    restored.fill_(0)
    view.fill_(0)
    torch.testing.assert_close(restored_bytes, byte_view)


def test_tensor_metadata_and_parameter_survive_mixed_dtype_retry(
    tmp_path: Path,
) -> None:
    data = torch.tensor([1 + 2j, 3 + 4j], requires_grad=True)
    conjugate = data.conj()
    negative = torch._neg_view(data)
    data.tag = "payload"  # pyrefly: ignore[missing-attribute]
    parameter = torch.nn.Parameter(data.detach())
    byte_view = data.detach().view(torch.uint8)
    path = tmp_path / "args.pt"
    save_tensor_tree((data, conjugate, negative, parameter, byte_view), path)
    loaded = torch.load(path, weights_only=False)
    assert loaded[0].tag == "payload"
    assert loaded[0].requires_grad and loaded[1].requires_grad
    assert loaded[1].is_conj() and loaded[2].is_neg()
    assert isinstance(loaded[3], torch.nn.Parameter) and loaded[3].requires_grad
    assert len({item.untyped_storage()._cdata for item in loaded}) == 1
    for before, after in zip(
        (data, conjugate, negative, parameter, byte_view), loaded, strict=True
    ):
        torch.testing.assert_close(before, after)


def test_handoff_inputs_preserve_mixed_dtype_aliases(tmp_path: Path) -> None:
    data = torch.arange(16, dtype=torch.float32)
    args = (data[1::2], data.view(torch.uint8)[3::4])
    path = tmp_path / "inputs.pt"
    _save_inputs(args, path)
    loaded = cast("tuple[torch.Tensor, ...]", _load_inputs(path))
    for before, after in zip(args, loaded, strict=True):
        torch.testing.assert_close(before, after)
        assert before.storage_offset() == after.storage_offset()
        assert before.stride() == after.stride()
        assert before.data_ptr() % 256 == after.data_ptr() % 256
    assert loaded[0].untyped_storage()._cdata == loaded[1].untyped_storage()._cdata


def test_native_v3_tensor_requires_grad_in_mixed_dtype_retry(tmp_path: Path) -> None:
    data = torch.arange(8, dtype=torch.float32).to(torch.float8_e4m3fn)
    data.requires_grad_(True)
    byte_view = data.detach().view(torch.int8)
    path = tmp_path / "fp8.pt"
    save_tensor_tree((data, byte_view), path)
    restored, restored_bytes = torch.load(path, weights_only=False)
    assert restored.requires_grad and restored.dtype == data.dtype
    assert restored.untyped_storage()._cdata == restored_bytes.untyped_storage()._cdata
    torch.testing.assert_close(restored.float(), data.float())
    torch.testing.assert_close(restored_bytes, byte_view)


def test_normal_serialization_keeps_tensor_and_explicit_typed_storage(
    tmp_path: Path,
) -> None:
    data = torch.arange(8, dtype=torch.float32)
    path = tmp_path / "typed.pt"
    save_tensor_tree((data, data._typed_storage()), path)
    restored, storage = torch.load(path, weights_only=False)
    torch.testing.assert_close(restored, data)
    assert storage.dtype == data.dtype
    assert restored.untyped_storage()._cdata == storage._untyped_storage._cdata


def test_unrelated_serialization_failure_is_not_retried(tmp_path: Path) -> None:
    with patch("torch.save", side_effect=RuntimeError("disk full")) as save:
        with pytest.raises(RuntimeError, match="disk full"):
            save_tensor_tree((torch.ones(4),), tmp_path / "args.pt")
        assert save.call_count == 1
