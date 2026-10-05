from __future__ import annotations

from functools import partial

from examples.rope import rope_bwd
from examples.rope import rope_fwd
from examples.rope import rope_pytorch
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_da
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_dx
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion.runtime import default_launcher

pytestmark = skipUnlessBackends(["triton", "cute"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4, 2, 128, 64), (2, 3, 1, 33, 128)])
@pytest.mark.parametrize("broadcast_angles", [False, True])
@skipIfNotCUDA()
@skipIfRefEager("compiles a pinned config; ref mode runs the kernel eagerly")
def test_rope_forward_backward_rounding_and_slices(
    dtype: torch.dtype, shape: tuple[int, int, int, int, int], broadcast_angles: bool
) -> None:
    batch, q_heads, k_heads, seq, dim = shape
    q = torch.randn(
        (batch, q_heads, seq, dim), device=DEVICE, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        (batch, k_heads, seq, dim), device=DEVICE, dtype=dtype, requires_grad=True
    )
    angles = torch.randn(
        (1 if broadcast_angles else batch, seq, dim), device=DEVICE, dtype=dtype
    )
    cos, sin = angles.cos(), angles.sin()
    args = (q, k, cos, sin)
    forward = helion.kernel(rope_fwd.fn, static_shapes=True).bind(args)
    backward = helion.kernel(rope_bwd.fn, static_shapes=True).bind(args)
    fwd = forward.compile_config(forward.config_spec.autotune_reference_config())
    bwd = backward.compile_config(backward.config_spec.autotune_reference_config())
    launch_options = {}
    if forward.env.backend_name == "triton":
        # Fusion can replace the rounded product and addition with a half FMA.
        # Disable contraction when checking the explicit rounding exactly.
        launch_options["_launcher"] = partial(default_launcher, enable_fp_fusion=False)
    for _ in range(3):
        with torch.no_grad():
            q.normal_()
            k.normal_()
        expected = rope_pytorch(q, k, cos, sin)
        torch.testing.assert_close(
            fwd(*args, **launch_options), expected, atol=0, rtol=0
        )
        grad_q = torch.randn_like(q)
        grad_k = torch.randn_like(k)
        grads = torch.autograd.grad(expected, (q, k), (grad_q, grad_k))
        torch.testing.assert_close(
            bwd(grad_q, grad_k, cos, sin, **launch_options), grads, atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shared_epilogue", [False, pytest.param(True, marks=skipUnlessBackends(["cute"]))]
)
@skipIfNotCUDA()
@skipIfRefEager("compiles a pinned config; ref mode runs the kernel eagerly")
def test_squeeze_excitation_sigmoid_rounds_matmul_input(
    dtype: torch.dtype, shared_epilogue: bool
) -> None:
    if shared_epilogue and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("Shared tcgen05 epilogues require CUDA capability >= 10.0")
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    args = tuple(torch.empty((128, 128), device=DEVICE, dtype=dtype) for _ in range(3))
    settings = (
        {
            "backend": "cute",
            "autotune_effort": "full",
            "cute_region_fission": True,
            "cute_full_slice_matmul_tiling": True,
        }
        if shared_epilogue
        else {}
    )
    bound = helion.kernel(
        squeeze_and_excitation_net_fwd.fn, static_shapes=True, **settings
    ).bind(args)
    config = (
        helion.Config(
            block_sizes=[128] * 6,
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=4,
            tcgen05_num_epi_warps=4,
            tcgen05_epilogue_fanout="shared",
        )
        if shared_epilogue
        else bound.config_spec.autotune_reference_config()
    )
    compiled = bound.compile_config(config)
    for _ in range(3):
        x, a, b = args
        for tensor in args:
            tensor.copy_(
                torch.randint(
                    -16, 17, tensor.shape, device=DEVICE, generator=generator
                ).to(dtype)
                / 16
            )
        out, c, d = compiled(*args)
        torch.testing.assert_close(c, torch.relu(x @ a), atol=0.1, rtol=0.01)
        # Isolate the second matmul's rounding from first-matmul accumulation
        # differences. Values use exact binary fractions, so its FP32 dot is
        # reproducible before conversion to the input dtype.
        expected_d = torch.sigmoid(c @ b)
        torch.testing.assert_close(d, expected_d, atol=0.001, rtol=0.001)
        torch.testing.assert_close(out, x * d, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(33, 37, 17), (65, 128, 32)])
@skipIfNotCUDA()
@skipIfRefEager("compiles a pinned config; ref mode runs the kernel eagerly")
def test_squeeze_excitation_staged_backward_replays_new_inputs(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    m, n, k = shape
    grad_out = torch.randn((m, n), device=DEVICE, dtype=dtype)
    x = torch.randn((m, n), device=DEVICE, dtype=dtype)
    a = torch.randn((n, k), device=DEVICE, dtype=dtype)
    b = torch.randn((k, n), device=DEVICE, dtype=dtype)
    c = torch.randn((m, k), device=DEVICE, dtype=dtype)
    d = torch.rand((m, n), device=DEVICE, dtype=dtype)
    args_x = (grad_out, x, a, b, c, d)
    args_a = (grad_out, x, b, c, d)
    bound_x = squeeze_and_excitation_net_bwd_dx.bind(args_x)
    bound_a = squeeze_and_excitation_net_bwd_da.bind(args_a)
    config = helion.Config(block_sizes=[16] * 8, pid_type="persistent_blocked")
    compiled_x = bound_x.compile_config(config)
    compiled_a = bound_a.compile_config(config)

    def call() -> tuple[torch.Tensor, torch.Tensor]:
        return compiled_x(*args_x), compiled_a(*args_a)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for _ in range(3):
        grad_out.normal_()
        for value in (x, a, b):
            value.normal_(std=0.25)
        c.normal_()
        d.uniform_(0.1, 0.9)
        grad_cb = grad_out * x * d * (1.0 - d)
        grad_c = (grad_cb @ b.T) * (c > 0)
        expected = (grad_out * d + grad_c @ a.T, x.T @ grad_c)
        torch.testing.assert_close(call(), expected, atol=0.05, rtol=0.01)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0.05, rtol=0.01)
