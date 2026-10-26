"""Runtime numerics for the trimmed tcgen05 GEMM fixed chain (sm_100 GPU).

Each case pins a config that exercises one of the trimmed prologue/teardown
shapes (merged pipeline init, warp-1 TMEM allocator, early permit release,
one-tile-per-CTA scheduler with the in-epilogue TMEM free, short teardown)
and launches it many times: a TMEM leak or a barrier-count mismatch shows up
as a hang or a wrong result within a few launches per SM.
"""

from __future__ import annotations

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import _batched_gemm
from test.test_cute_tcgen05_fixed_chain import _bias_gemm
from test.test_cute_tcgen05_fixed_chain import _compact_group4_fragment
from test.test_cute_tcgen05_fixed_chain import _two_output_gemm

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="tcgen05 requires an sm_100 GPU",
    ),
]

LAUNCHES = 40


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=torch.float16, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(torch.float16)
    return out


def _addmm_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _config(**overrides: object) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": [128, 64, 64],
        "loop_orders": [[0, 1]],
        "indexing": ["tensor_descriptor"] * 3,
        "pid_type": "persistent_blocked",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_tvm_ffi_launch": False,
    }
    values.update(overrides)
    return helion.Config(**values)


def _run(
    kernel_fn,
    args: tuple[torch.Tensor, ...],
    config: helion.Config,
    expected: torch.Tensor,
    *,
    source_needle: str | tuple[str, str] | None = None,
    **tol: float,
) -> None:
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    bound.set_config(config)
    if source_needle is not None:
        with bound.env:
            normalized = bound.config_spec.normalized_config(config)
        source = bound.to_code(normalized)
        if isinstance(source_needle, tuple):
            # Both present, in this order (for the one-tile epilogue: the TMEM
            # free follows the TMA store issue).
            first, second = source_needle
            assert source.index(first) < source.index(second), source
        else:
            assert source_needle in source, source
    outputs = [bound(*args) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    for out in outputs:
        torch.testing.assert_close(out, expected.to(out.dtype), **tol)


def test_one_tile_per_cta_fp16() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    _run(
        _gemm,
        (a, b),
        _config(),
        a @ b,
        source_needle=(
            "cute.copy(tcgen05_tma_store_atom",
            "tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)",
        ),
        rtol=1e-2,
        atol=1e-2,
    )


def test_plain_two_cta_narrow_subtile_fp16() -> None:
    # Plain 16-bit two-CTA 256x256 tiles take the (128, 32) subtile and, at
    # bk=64, the no-unroll K loop; the persistent kernel runs 3.5 tiles per
    # CTA pair here.
    torch.manual_seed(0)
    a = torch.randn(4096, 512, device=DEVICE, dtype=torch.float16)
    b = torch.randn(512, 2048, device=DEVICE, dtype=torch.float16)
    _run(
        _addmm_gemm,
        (a, b),
        _config(
            block_sizes=[256, 256, 64],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=6,
        ),
        a @ b,
        source_needle=(
            "tcgen05_epi_tile = (cute.make_layout(128), cute.make_layout(32))",
            "cutlass.Int32(_BLOCK_SIZE_2), unroll=1):",
        ),
        rtol=2e-2,
        atol=2e-1,
    )


def test_multi_tile_persistent_fp16() -> None:
    torch.manual_seed(0)
    a = torch.randn(2048, 1024, device=DEVICE, dtype=torch.float16)
    b = torch.randn(1024, 4096, device=DEVICE, dtype=torch.float16)
    _run(
        _gemm,
        (a, b),
        _config(tcgen05_ab_stages=3),
        a @ b,
        source_needle="while tcgen05_role_local_0_work_tile.is_valid_tile",
        rtol=2e-2,
        atol=2e-1,
    )


def test_batched_one_tile_per_cta_fp16() -> None:
    torch.manual_seed(0)
    a = torch.randn(8, 256, 512, device=DEVICE, dtype=torch.float16)
    b = torch.randn(8, 512, 256, device=DEVICE, dtype=torch.float16)
    _run(
        _batched_gemm,
        (a, b),
        _config(
            block_sizes=[1, 128, 128, 64], loop_orders=[[0, 1, 2]], tcgen05_ab_stages=3
        ),
        torch.bmm(a, b),
        source_needle=(
            "cute.copy(tcgen05_tma_store_atom",
            "tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)",
        ),
        rtol=2e-2,
        atol=1e-1,
    )


def test_fp8_e4m3_one_tile_per_cta() -> None:
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE).to(torch.float8_e4m3fn)
    b = torch.randn(256, 256, device=DEVICE).to(torch.float8_e4m3fn).T.contiguous().T
    scale = torch.tensor(1.0, device=DEVICE)
    expected = torch._scaled_mm(
        a, b, scale, scale, use_fast_accum=False, out_dtype=torch.float16
    )
    _run(
        _gemm,
        (a, b),
        _config(block_sizes=[128, 64, 128], tcgen05_ab_stages=4, tcgen05_c_stages=4),
        expected,
        source_needle="allocator_warp_id=1",
        rtol=2e-2,
        atol=2e-2,
    )


def test_scheduler_warp_strategy_fp16() -> None:
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)
    _run(
        _gemm,
        (a, b),
        _config(
            block_sizes=[128, 64, 128],
            tcgen05_ab_stages=3,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
        ),
        a @ b,
        source_needle="barrier_id=3, num_threads=160",
        rtol=2e-2,
        atol=1e-1,
    )


def test_two_output_stores_one_tile_per_cta_bf16() -> None:
    # One tile per CTA with two stores of the accumulator: TMEM must be
    # handshaken + freed once (in the teardown), or the second epilogue
    # handshake hangs the CTA.
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
    config = _config()
    bound = helion.kernel(_two_output_gemm, backend="cute", static_shapes=True).bind(
        (a, b)
    )
    bound.set_config(config)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert source.count("tcgen05_tmem_allocator.free(") == 1, source
    assert source.count("tcgen05_tmem_alloc_barrier.arrive_and_wait()") == 1, source
    assert "tcgen05_pipeline_init_barrier.arrive_and_wait()" in source
    expected = a @ b
    outputs = [bound(a, b) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    for out, out2 in outputs:
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=1e-1)
        torch.testing.assert_close(out2, torch.relu(expected), rtol=2e-2, atol=1e-1)


def test_compact_fragment_epilogue_one_tile_per_cta_bf16() -> None:
    # The compact fragment epilogue renders no TMA store body; TMEM must
    # still be freed (in the teardown) or the CTA exits with it allocated
    # ("tensor memory not completely freed", a sticky CUDA error).
    torch.manual_seed(17)
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(1, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    acc = torch.einsum("mk,hkd->hmd", x.float(), weight.float())
    groups = acc.view(1, 128, 32, 4)
    expected = (
        groups[..., 0]
        + 2.0 * groups[..., 1]
        - 3.0 * groups[..., 2]
        + 4.0 * groups[..., 3]
    )
    _run(
        _compact_group4_fragment,
        (x, weight),
        _config(
            block_sizes=[1, 128, 128, 32],
            loop_orders=[[0, 1, 2]],
            pid_type="persistent_interleaved",
        ),
        expected,
        source_needle=("for tcgen05_epi_position", "tcgen05_tmem_allocator.free("),
        rtol=2e-2,
        atol=2e-1,
    )


def test_pure_matmul_lifecycle_flat_fp16() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    _run(
        _addmm_gemm,
        (a, b),
        _config(pid_type="flat", tcgen05_strategy="pure_matmul_role_lifecycle"),
        a @ b,
        source_needle=(
            "allocator_warp_id=1",
            "tcgen05_pipeline_init_barrier.arrive_and_wait()",
        ),
        rtol=1e-2,
        atol=1e-2,
    )


def test_clc_one_cta_scheduler_multiwave_bf16() -> None:
    # 1024 tiles over 148 SMs: several CLC-scheduled waves per CTA.
    torch.manual_seed(0)
    a = torch.randn(2048, 256, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 4096, device=DEVICE, dtype=torch.bfloat16)
    _run(
        _gemm,
        (a, b),
        _config(
            pid_type="persistent_interleaved",
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            tcgen05_persistence_model="clc_persistent",
        ),
        a @ b,
        source_needle=("barrier_id=3, num_threads=160", "cute.arch.clc_response"),
        rtol=2e-2,
        atol=1e-1,
    )


@pytest.mark.parametrize("load_mode", ["simt", "tma"])
def test_c_input_warp_aux_pipeline_bf16(load_mode: str) -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
    _run(
        _bias_gemm,
        (a, b, bias),
        _config(
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            tcgen05_warp_spec_c_input_warps=1,
            tcgen05_aux_load_mode=load_mode,
        ),
        (a.float() @ b.float() + bias.float()).to(torch.bfloat16),
        source_needle=(
            "tcgen05_aux_pipeline = cutlass.pipeline.Pipeline",
            "cute.arch.mbarrier_init_fence()",
        ),
        rtol=2e-2,
        atol=1e-1,
    )


def test_flat_pid_type_fp16() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    _run(
        _gemm,
        (a, b),
        _config(pid_type="flat"),
        a @ b,
        source_needle="tcgen05_pipeline_init_barrier.arrive_and_wait()",
        rtol=1e-2,
        atol=1e-2,
    )
