"""Actual captured dependencies, not ready-input-only overlap benchmarks."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import is_cuda
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_pdl import CONTROL_DELAY
from helion._testing.cute_pdl import produce
from helion.runtime.cute import capture_pdl_launch
from helion.runtime.cute.capture_pdl import WaitMode


@pytest.fixture(
    scope="module",
    params=[
        (True, "entry", 0),
        (False, "entry", 3),
        (True, "loader_tail", 0),
        (False, "loader_tail", 3),
    ],
)
def kernels(request):
    if not is_cuda():
        pytest.skip("capture PDL requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention requires SM100 tensor memory")
    return_lse, release, lag = request.param
    args = list(paged_attention_args(device=DEVICE, rows=8, requests=2))
    for index, modulus in ((0, 19), (1, 15), (2, 17)):
        tensor = args[index]
        values = torch.arange(tensor.numel(), device=DEVICE).reshape(tensor.shape)
        tensor.copy_((values % modulus - modulus // 2) / (32 if index == 0 else 1))
    args[3].fill_(126)
    args[4].fill_(126)
    args[5].copy_(
        (torch.arange(args[5].numel(), device=DEVICE) * 7 % args[1].size(0)).reshape(
            args[5].shape
        )
    )
    args[6].copy_(torch.tensor([0, 1025], dtype=torch.int32, device=DEVICE))
    args[-1] = -1
    args.extend((None, True, return_lse))
    bound = paged_online_attention.bind(tuple(args))
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_capture_overlap=True,
        # Reuse fixtures for both wrapper ABIs and both V-lag loader branches.
        cute_mma_dag_query_tma=(return_lse == (release == "entry")),
        cute_pdl_release=release,
        cute_mma_dag_v_lag=lag,
    )
    tracked = bound.compile_config(config)
    plain = bound.compile_config(
        helion.Config.from_dict(
            {
                **config.config,
                "cute_pdl": False,
                "cute_pdl_capture_overlap": False,
                "cute_pdl_release": "auto",
            }
        )
    )
    tracked(*args)
    plain(*args)
    torch.cuda.synchronize()
    return args, tracked, plain


@pytest.mark.cpu_only
def test_capture_fixture_skips_before_querying_cuda(monkeypatch):
    monkeypatch.setitem(kernels.__wrapped__.__globals__, "is_cuda", lambda: False)

    def unavailable():
        raise AssertionError("CUDA capability must not be queried")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable)
    with pytest.raises(pytest.skip.Exception, match="capture PDL requires CUDA"):
        kernels.__wrapped__(None)


def close(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        if value is None:
            assert reference is None
        else:
            torch.testing.assert_close(value, reference, rtol=0.002, atol=0.002)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_capture_overlap_retains_non_immediate_ancestor_and_replays(
    kernels, monkeypatch
):
    args, tracked, plain = kernels
    modes = []
    original = capture_pdl_launch.finish_capture_launch

    def observed(*values):
        mode = original(*values)
        modes.append(mode)
        return mode

    monkeypatch.setattr(capture_pdl_launch, "finish_capture_launch", observed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output_a = tracked(*args)
        output_b = tracked(*args)
        dependent_args = [output_a[0], *args[1:]]
        output_c = tracked(*dependent_args)
        output_d = tracked(*args)
    assert modes == [
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
    ], modes
    for iteration in range(2):
        args[0].add_(0.03125)
        args[2].neg_()
        args[6].copy_(
            torch.tensor(
                [129, 0] if iteration else [0, 1025], dtype=torch.int32, device=DEVICE
            )
        )
        expected_a = plain(*args)
        expected_c = plain(expected_a[0], *args[1:])
        graph.replay()
        for output in (output_a, output_b, output_d):
            close(output, expected_a)
        close(output_c, expected_c)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_unknown_delayed_producer_forces_early_wait(kernels, monkeypatch):
    args, tracked, plain = kernels
    lengths = args[6]
    values = torch.tensor([0, 1025], dtype=torch.int32, device=DEVICE)
    produce(values, lengths, delay=CONTROL_DELAY, grid=1, block=32)
    tracked(*args)
    torch.cuda.synchronize()
    modes = []
    original = capture_pdl_launch.finish_capture_launch

    def observed(*arguments):
        mode = original(*arguments)
        modes.append(mode)
        return mode

    monkeypatch.setattr(capture_pdl_launch, "finish_capture_launch", observed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output_a = tracked(*args)
        produce(values, lengths, delay=CONTROL_DELAY, grid=1, block=32)
        output_b = tracked(*args)
    assert modes == [WaitMode.EARLY, WaitMode.EARLY], modes
    for iteration in range(2):
        values.copy_(
            torch.tensor(
                [129, 0] if iteration else [0, 1025], dtype=torch.int32, device=DEVICE
            )
        )
        lengths.copy_(values)
        expected_b = plain(*args)
        lengths.fill_(0)
        expected_a = plain(*args)
        graph.replay()
        close(output_a, expected_a)
        close(output_b, expected_b)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_capture_overlap_on_non_current_device(kernels, monkeypatch):
    args, tracked, plain = kernels
    launch_device = args[0].device
    assert launch_device.index is not None
    other_device = (launch_device.index + 1) % torch.cuda.device_count()
    # Initialize the other context before starting global CUDA capture.
    with torch.cuda.device(other_device):
        torch.empty(1, device=other_device)
        torch.cuda.synchronize()
    modes = []
    original = capture_pdl_launch.finish_capture_launch

    def observed(*values):
        mode = original(*values)
        modes.append(mode)
        return mode

    monkeypatch.setattr(capture_pdl_launch, "finish_capture_launch", observed)
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.device(launch_device),
        torch.cuda.graph(graph),
        torch.cuda.device(other_device),
    ):
        output_a = tracked(*args)
        output_b = tracked(*args)
        assert torch.cuda.current_device() == other_device
    assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES], modes
    for _ in range(2):
        args[0].add_(0.03125)
        args[2].neg_()
        expected = plain(*args)
        graph.replay()
        close(output_a, expected)
        close(output_b, expected)
