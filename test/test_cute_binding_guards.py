"""CPU codegen guards do not assume a fresh xdist worker's CUDA state."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from helion._testing.cute_codegen import forbid_cuda_work

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("initialized", (False, True))
def test_cuda_work_guard_preserves_existing_context_state(initialized: bool) -> None:
    with patch("torch.cuda.is_initialized", return_value=initialized):
        with forbid_cuda_work():
            assert torch.empty(4).device.type == "cpu"
            assert torch.cuda.is_initialized() is initialized
        assert torch.cuda.is_initialized() is initialized


@pytest.mark.parametrize("initialized", (False, True))
# Both context states must reject each entry point, including an already-live stream.
@pytest.mark.parametrize(
    "entry_point", ("_lazy_init", "init", "synchronize", "current_stream", "Stream")
)
def test_cuda_work_guard_rejects_calls_with_or_without_context(
    initialized: bool, entry_point: str
) -> None:
    with (
        patch("torch.cuda.is_initialized", return_value=initialized),
        pytest.raises(AssertionError, match="CUDA work forbidden"),
        forbid_cuda_work(),
    ):
        getattr(torch.cuda, entry_point)()


def test_cuda_work_guard_detects_a_swallowed_forbidden_call() -> None:
    with (
        pytest.raises(AssertionError, match="to not have been called"),
        forbid_cuda_work(),
        pytest.raises(AssertionError, match="CUDA work forbidden"),
    ):
        torch.cuda.synchronize()


def test_cuda_work_guard_rejects_native_compile_with_existing_context() -> None:
    cutlass = pytest.importorskip("cutlass")
    with (
        patch("torch.cuda.is_initialized", return_value=True),
        pytest.raises(AssertionError, match="native forbidden"),
        forbid_cuda_work(),
    ):
        cutlass.cute.compile(None)


def test_cuda_work_guard_rejects_native_compile_with_options() -> None:
    # ``cute.compile[options](...)`` is how the launchers compile.
    cutlass = pytest.importorskip("cutlass")
    with (
        patch("torch.cuda.is_initialized", return_value=True),
        pytest.raises(AssertionError, match="native forbidden"),
        forbid_cuda_work(),
    ):
        cutlass.cute.compile[()](None)


def test_cuda_work_guard_detects_context_state_change() -> None:
    with (
        patch("torch.cuda.is_initialized", side_effect=(False, True)),
        pytest.raises(AssertionError),
        forbid_cuda_work(),
    ):
        pass


@pytest.mark.parametrize("initialized", (False, True))
def test_cuda_work_guard_blocks_direct_aten_allocation(initialized: bool) -> None:
    with (
        patch("torch.cuda.is_initialized", return_value=initialized),
        pytest.raises(AssertionError, match="CUDA device forbidden by dispatch guard"),
        forbid_cuda_work(),
    ):
        torch.ops.aten.empty.memory_format([4], device=torch.device("cuda:0"))


def test_cuda_work_guard_allows_meta_and_fake_cuda_operations() -> None:
    with forbid_cuda_work():
        meta = torch.empty(4, device=torch.device("meta"))
        assert (meta + 1).device.type == "meta"
        with FakeTensorMode():
            fake = torch.empty(4, device=torch.device("cuda:0"))
            assert (fake + 1).device.type == "cuda"


def test_cuda_work_guard_disables_fallback_for_fake_used_outside_its_mode() -> None:
    with forbid_cuda_work():
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


def test_offline_compile_option_preserves_the_compiler_call() -> None:
    cutlass = pytest.importorskip("cutlass")
    result = object()
    with (
        patch("torch.cuda.is_initialized", return_value=True),
        patch.object(cutlass.cute, "compile", return_value=result) as compile_call,
        forbid_cuda_work(forbid_native=False),
    ):
        assert cutlass.cute.compile(None, no_jit_engine=True) is result
        compile_call.assert_called_once_with(None, no_jit_engine=True)


@pytest.mark.parametrize("entry_point", ("_lazy_init", "current_stream", "Event"))
def test_offline_compile_option_still_rejects_cuda_work(entry_point) -> None:
    with (
        patch("torch.cuda.is_initialized", return_value=True),
        pytest.raises(AssertionError, match="CUDA work forbidden"),
        forbid_cuda_work(forbid_native=False),
    ):
        getattr(torch.cuda, entry_point)()


def test_offline_compile_option_still_rejects_real_tensor_allocation() -> None:
    with (
        patch("torch.cuda.is_initialized", return_value=True),
        pytest.raises(AssertionError, match="CUDA device forbidden by dispatch guard"),
        forbid_cuda_work(forbid_native=False),
    ):
        torch.ops.aten.empty.memory_format([4], device=torch.device("cuda:0"))


def test_offline_compile_option_still_detects_context_state_change() -> None:
    with (
        patch("torch.cuda.is_initialized", side_effect=(False, True)),
        pytest.raises(AssertionError),
        forbid_cuda_work(forbid_native=False),
    ):
        pass
