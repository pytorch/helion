"""CPU codegen guards do not assume a fresh xdist worker's CUDA state."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test._cute_binding import _forbid_cuda_work


@pytest.mark.parametrize("initialized", (False, True))
def test_cuda_work_guard_preserves_existing_context_state(initialized: bool) -> None:
    with patch("torch.cuda.is_initialized", return_value=initialized):
        with _forbid_cuda_work():
            assert torch.empty(4).device.type == "cpu"
            assert torch.cuda.is_initialized() is initialized
        assert torch.cuda.is_initialized() is initialized


@pytest.mark.parametrize("initialized", (False, True))
@pytest.mark.parametrize(
    "entry_point", ("_lazy_init", "init", "synchronize", "current_stream", "Stream")
)
def test_cuda_work_guard_rejects_calls_with_or_without_context(
    initialized: bool, entry_point: str
) -> None:
    with (
        patch("torch.cuda.is_initialized", return_value=initialized),
        pytest.raises(AssertionError, match="CUDA work forbidden"),
        _forbid_cuda_work(),
    ):
        getattr(torch.cuda, entry_point)()


def test_cuda_work_guard_detects_a_swallowed_forbidden_call() -> None:
    with (
        pytest.raises(AssertionError, match="to not have been called"),
        _forbid_cuda_work(),
        pytest.raises(AssertionError, match="CUDA work forbidden"),
    ):
        torch.cuda.synchronize()


def test_cuda_work_guard_rejects_native_compile_with_existing_context() -> None:
    cutlass = pytest.importorskip("cutlass")
    with (
        patch("torch.cuda.is_initialized", return_value=True),
        pytest.raises(AssertionError, match="native forbidden"),
        _forbid_cuda_work(),
    ):
        cutlass.cute.compile(None)


def test_cuda_work_guard_detects_context_state_change() -> None:
    with (
        patch("torch.cuda.is_initialized", side_effect=(False, True)),
        pytest.raises(AssertionError),
        _forbid_cuda_work(),
    ):
        pass


@pytest.mark.parametrize("initialized", (False, True))
def test_cuda_work_guard_blocks_direct_aten_allocation(initialized: bool) -> None:
    with (
        patch("torch.cuda.is_initialized", return_value=initialized),
        pytest.raises(AssertionError, match="CUDA device forbidden by dispatch guard"),
        _forbid_cuda_work(),
    ):
        torch.ops.aten.empty.memory_format([4], device=torch.device("cuda:0"))


def test_cuda_work_guard_allows_meta_and_fake_cuda_operations() -> None:
    with _forbid_cuda_work():
        meta = torch.empty(4, device=torch.device("meta"))
        assert (meta + 1).device.type == "meta"
        with FakeTensorMode():
            fake = torch.empty(4, device=torch.device("cuda:0"))
            assert (fake + 1).device.type == "cuda"


def test_cuda_work_guard_disables_fallback_for_fake_used_outside_its_mode() -> None:
    with _forbid_cuda_work():
        with FakeTensorMode() as mode:
            fake = torch.empty(4, device=torch.device("cuda:0"))
        original = mode.dispatch
        observed = []

        def record(*args, **kwargs):
            observed.append(mode.allow_fallback_kernels)
            return original(*args, **kwargs)

        assert mode.allow_fallback_kernels
        with patch.object(mode, "dispatch", side_effect=record):
            assert (fake + 1).device.type == "cuda"
        assert observed and not any(observed)
        assert mode.allow_fallback_kernels
