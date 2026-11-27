"""Trusted worker files retain tensor metadata and aliases across dtype views."""

from __future__ import annotations

import copyreg
import dataclasses
import threading
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch
import zipfile

import pytest
import torch

from helion.autotuner.handoff_inputs import _load_inputs
from helion.autotuner.handoff_inputs import _save_inputs
from helion.autotuner.handoff_inputs import _SavedInputs
from helion.autotuner.kernel_args import load_trusted_kernel_args
from helion.autotuner.tensor_serialization import save_tensor_tree

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu_only


def _handoff_callback(value=None):
    return value


@dataclasses.dataclass
class _HandoffCallbackHolder:
    callback: object


class _RegisteredTensor(torch.Tensor):
    """A tensor type with unpicklable state that its registered reducer omits."""


def _reduce_registered_tensor(tensor: _RegisteredTensor) -> tuple:
    plain = tensor.detach().as_subclass(torch.Tensor)
    return (_restore_registered_tensor, (plain, tensor.requires_grad))


def _restore_registered_tensor(
    plain: torch.Tensor, requires_grad: bool
) -> _RegisteredTensor:
    return plain.as_subclass(_RegisteredTensor).requires_grad_(requires_grad)


@pytest.mark.parametrize("container", ["direct", "list", "record"])
def test_handoff_refuses_callbacks_before_overwriting_inputs(
    container, monkeypatch, tmp_path
):
    tensor = torch.arange(4)
    monkeypatch.setattr(_handoff_callback, "__defaults__", (tensor,))
    if container == "list":
        value = [_handoff_callback]
    elif container == "record":
        value = _HandoffCallbackHolder(_handoff_callback)
    else:
        value = _handoff_callback
    path = tmp_path / "inputs.pt"
    path.write_bytes(b"existing handoff inputs")
    with pytest.raises(TypeError, match="handoff.*callback"):
        _save_inputs((tensor, value), path)
    assert path.read_bytes() == b"existing handoff inputs"
    assert _handoff_callback() is tensor


@pytest.mark.parametrize("wrapped", [False, True])
def test_handoff_refuses_legacy_callback_files(wrapped, monkeypatch, tmp_path):
    tensor = torch.arange(4)
    monkeypatch.setattr(_handoff_callback, "__defaults__", (tensor,))
    value = (tensor, {"callback": _handoff_callback})
    saved = (
        _SavedInputs(value, (tensor.untyped_storage().data_ptr() % 256,))
        if wrapped
        else value
    )
    path = tmp_path / "legacy.pt"
    torch.save(saved, path)
    # A by-name callback would now observe this new default, not the saved tensor.
    monkeypatch.setattr(_handoff_callback, "__defaults__", (torch.full((4,), 99),))
    with pytest.raises(TypeError, match="handoff.*callback"):
        _load_inputs(path)


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


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("python_state", [False, True])
def test_fp8_tensor_that_requires_grad_loads_in_the_worker(
    tmp_path: Path, dtype: torch.dtype, python_state: bool
) -> None:
    # PyTorch saves such a tensor and then cannot load it: its v3 rebuild
    # calls set_ on a leaf that requires grad. TritonBench's fp8 attention
    # operands are built that way, and the benchmark worker reads the
    # arguments of every candidate from this file.
    data = torch.arange(8, dtype=torch.float32).to(dtype)
    data.requires_grad_(True)
    if python_state:
        # Python state nests the rebuild in ``_rebuild_from_type_v2``.
        data.tag = "payload"  # pyrefly: ignore[missing-attribute]
    plain = torch.arange(4, dtype=torch.float32, requires_grad=True)
    path = tmp_path / "args.pt"
    save_tensor_tree((data, plain, data[2:], 3), path)
    restored, restored_plain, view, scalar = load_trusted_kernel_args(str(path))
    assert isinstance(restored, torch.Tensor) and isinstance(view, torch.Tensor)
    assert isinstance(restored_plain, torch.Tensor)
    assert restored.dtype == dtype and restored.requires_grad and restored.is_leaf
    assert view.requires_grad and restored_plain.requires_grad
    assert view.untyped_storage()._cdata == restored.untyped_storage()._cdata
    assert view.storage_offset() == 2
    assert getattr(restored, "tag", None) == ("payload" if python_state else None)
    torch.testing.assert_close(restored.detach().float(), data.detach().float())
    torch.testing.assert_close(restored_plain, plain)
    assert scalar == 3


def test_normal_serialization_writes_what_pytorch_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only a v3 tensor that requires grad gets another rebuild function. A
    # reducer registered for a tensor type keeps its place ahead of
    # ``__reduce_ex__``, which could not pickle this one's lock.
    monkeypatch.setitem(
        copyreg.dispatch_table, _RegisteredTensor, _reduce_registered_tensor
    )
    registered = torch.arange(3, dtype=torch.float32).as_subclass(_RegisteredTensor)
    registered.lock = threading.Lock()  # pyrefly: ignore[missing-attribute]
    registered.requires_grad_(True)
    value = (
        torch.arange(4, dtype=torch.float32, requires_grad=True),
        torch.arange(4, dtype=torch.float32).to(torch.float8_e4m3fn),
        torch.nn.Parameter(torch.ones(2, dtype=torch.float16)),
        registered,
        {"shape": (2, 3), "dtype": torch.bfloat16},
    )
    streams = []
    for name, save in (("helion", save_tensor_tree), ("torch", torch.save)):
        (tmp_path / name).mkdir()
        path = tmp_path / name / "args.pt"
        save(value, path)
        with zipfile.ZipFile(path) as archive:
            (record,) = [n for n in archive.namelist() if n.endswith("/data.pkl")]
            streams.append(archive.read(record))
    assert streams[0] == streams[1]


@pytest.mark.parametrize("mixed_dtype_view", [False, True])
def test_registered_reducer_of_fp8_tensor_that_requires_grad_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mixed_dtype_view: bool
) -> None:
    # A reducer registered for the type may hand back PyTorch's own v3
    # rebuild; its file has to load as well, on the first attempt and on the
    # mixed-dtype retry.
    monkeypatch.setitem(
        copyreg.dispatch_table, torch.Tensor, lambda tensor: tensor.__reduce_ex__(2)
    )
    data = torch.arange(8, dtype=torch.float32).to(torch.float8_e4m3fn)
    data.requires_grad_(True)
    args = (data, data.detach().view(torch.int8)) if mixed_dtype_view else (data,)
    path = tmp_path / "args.pt"
    save_tensor_tree(args, path)
    restored, *views = load_trusted_kernel_args(str(path))
    assert isinstance(restored, torch.Tensor)
    assert restored.dtype == data.dtype and restored.requires_grad
    torch.testing.assert_close(restored.detach().float(), data.detach().float())
    assert len(views) == len(args) - 1
    for view in views:
        assert isinstance(view, torch.Tensor)
        assert view.untyped_storage()._cdata == restored.untyped_storage()._cdata
        torch.testing.assert_close(view, data.detach().view(torch.int8))


def test_mixed_dtype_retry_uses_a_registered_reducer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        copyreg.dispatch_table, _RegisteredTensor, _reduce_registered_tensor
    )
    registered = torch.arange(3, dtype=torch.float32).as_subclass(_RegisteredTensor)
    registered.lock = threading.Lock()  # pyrefly: ignore[missing-attribute]
    data = torch.arange(8, dtype=torch.float32)
    path = tmp_path / "args.pt"
    save_tensor_tree((registered, data, data.view(torch.int8)), path)
    restored, restored_data, byte_view = torch.load(path, weights_only=False)
    assert type(restored) is _RegisteredTensor
    torch.testing.assert_close(restored.as_subclass(torch.Tensor), torch.arange(3.0))
    torch.testing.assert_close(restored_data, data)
    assert byte_view.untyped_storage()._cdata == restored_data.untyped_storage()._cdata


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


def test_mixed_dtype_retry_keeps_explicit_typed_storage(tmp_path: Path) -> None:
    # The retry writes tensors as untyped bytes; a typed storage written under
    # its own dtype beside them is what ``torch.save`` refuses.
    data = torch.arange(8, dtype=torch.float32)
    other = torch.arange(4, dtype=torch.float64)
    byte_view = data.view(torch.int8)
    path = tmp_path / "args.pt"
    save_tensor_tree(
        (data, byte_view, data._typed_storage(), other, other._typed_storage()), path
    )
    restored, restored_bytes, storage, restored_other, other_storage = torch.load(
        path, weights_only=False
    )
    torch.testing.assert_close(restored, data)
    torch.testing.assert_close(restored_bytes, byte_view)
    torch.testing.assert_close(restored_other, other)
    assert storage.dtype == torch.float32 and storage.tolist() == data.tolist()
    assert other_storage.dtype == torch.float64
    assert other_storage.tolist() == other.tolist()
    shared = restored.untyped_storage()._cdata
    assert restored_bytes.untyped_storage()._cdata == shared
    assert storage._untyped_storage._cdata == shared
    assert (
        restored_other.untyped_storage()._cdata == other_storage._untyped_storage._cdata
    )


def test_unrelated_serialization_failure_is_not_retried(tmp_path: Path) -> None:
    with patch("torch.save", side_effect=RuntimeError("disk full")) as save:
        with pytest.raises(RuntimeError, match="disk full"):
            save_tensor_tree((torch.ones(4),), tmp_path / "args.pt")
        assert save.call_count == 1
