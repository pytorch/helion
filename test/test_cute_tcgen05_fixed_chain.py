"""Codegen pins for the trimmed fixed chain of the plain tcgen05 GEMM launch.

CPU-only (fake CUDA target): every test binds a kernel, pins a config and
inspects the generated CuTe source.  Runtime coverage lives in
``test_cute_tcgen05_fixed_chain_gpu.py``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_shared_rhs_grouped import _target
from test.test_cute_tcgen05_ordinary_exploration import _cuda_fake
from test.test_cute_tcgen05_ordinary_exploration import _ordinary_gemm

import helion
from helion._compiler.cute.tcgen05_lifecycle import Tcgen05LifecycleContext
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CPU_DEVICE = "cpu"

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel


def _batched_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.shape
    _, _, n = b.shape
    out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    for tile_b, tile_m, tile_n in hl.tile((batch, m, n)):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc.to(out.dtype)
    return out


def _two_output_gemm(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    out2 = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        out2[tile_m, tile_n] = torch.relu(acc).to(out.dtype)
    return out, out2


def _compact_group4_fragment(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compact four adjacent projection columns into one weighted output.

    A shape-changing fragment epilogue: the compact SIMT schedule owns the
    TMEM read and the TMA store body is never rendered.
    """
    m, k = x.size()
    h, _, d = weight.size()
    out = torch.empty([h, m, d // 4], dtype=x.dtype, device=x.device)
    for tile_h, tile_m, tile_d in hl.tile([h, m, d]):
        acc = hl.zeros([tile_h, tile_m, tile_d], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], weight[tile_h, tile_k, tile_d], acc=acc)
        groups = acc.view(
            tile_h.block_size,
            tile_m.block_size,
            tile_d.block_size // 4,
            2,
            2,
        )
        left, right = hl.split(groups)
        x0, x2 = hl.split(left)
        x1, x3 = hl.split(right)
        index_groups = tile_d.index.view(tile_d.block_size // 4, 2, 2)
        index_left, index_right = hl.split(index_groups)
        output_d, output_d_other = hl.split(index_left)
        out[
            tile_h.index[:, None, None],
            tile_m.index[None, :, None],
            (output_d // 4)[None, None, :],
        ] = (x0 + 2.0 * x1 - 3.0 * x2 + 4.0 * x3).to(x.dtype)
    return out


def _bias_gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_m, tile_n]).to(out.dtype)
    return out


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HELION_BACKEND", "cute")
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "full")
    with (
        _target(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _mock_cuda_unavailable(),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch(
            "helion.runtime.kernel._find_device", return_value=torch.device("cuda:0")
        ),
        patch(
            "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
            new=_cuda_fake,
        ),
        patch(
            "torch._inductor.runtime.hints.DeviceProperties.create",
            return_value=SimpleNamespace(
                type="cuda", index=0, cc=100, warp_size=32, multi_processor_count=148
            ),
        ),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind_gemm(shape: tuple[int, int, int], dtype: torch.dtype) -> BoundKernel:
    m, n, k = shape
    args = (
        torch.empty((m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((k, n), dtype=dtype, device=CPU_DEVICE),
    )
    return helion.kernel(_ordinary_gemm, backend="cute", static_shapes=True).bind(args)


def _bind_bmm(shape: tuple[int, int, int, int], dtype: torch.dtype) -> BoundKernel:
    batch, m, n, k = shape
    args = (
        torch.empty((batch, m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((batch, k, n), dtype=dtype, device=CPU_DEVICE),
    )
    return helion.kernel(_batched_gemm, backend="cute", static_shapes=True).bind(args)


def _bind_two_output_gemm(
    shape: tuple[int, int, int], dtype: torch.dtype
) -> BoundKernel:
    m, n, k = shape
    args = (
        torch.empty((m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((k, n), dtype=dtype, device=CPU_DEVICE),
    )
    return helion.kernel(_two_output_gemm, backend="cute", static_shapes=True).bind(
        args
    )


def _bind_bias_gemm(shape: tuple[int, int, int], dtype: torch.dtype) -> BoundKernel:
    m, n, k = shape
    args = (
        torch.empty((m, k), dtype=dtype, device=CPU_DEVICE),
        torch.empty((k, n), dtype=dtype, device=CPU_DEVICE),
        torch.empty((m, n), dtype=dtype, device=CPU_DEVICE),
    )
    return helion.kernel(_bias_gemm, backend="cute", static_shapes=True).bind(args)


def _bind_compact_fragment() -> BoundKernel:
    args = (
        torch.empty((128, 64), dtype=torch.bfloat16, device=CPU_DEVICE),
        torch.empty((1, 64, 128), dtype=torch.bfloat16, device=CPU_DEVICE),
    )
    return helion.kernel(
        _compact_group4_fragment, backend="cute", static_shapes=True
    ).bind(args)


def _source(bound: BoundKernel, requested: helion.Config) -> str:
    with bound.env:
        config = bound.config_spec.normalized_config(requested)
    return bound.to_code(config)


def _plain_config(**overrides: object) -> helion.Config:
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


def _kernel_body(source: str) -> str:
    """The ``@cute.kernel`` function source, without trailing comments."""
    start = source.index("@cute.kernel")
    end = source.index("_helion_cute_wrapper_plans")
    lines = source[start:end].splitlines()
    while lines and (
        not lines[-1].strip()
        or lines[-1].lstrip().startswith("#")
        or not lines[-1].startswith(" ")
    ):
        lines.pop()
    return "\n".join(lines)


PUBLICATION = (
    "if not tcgen05_epi_active or tcgen05_warp_idx == cutlass.Int32(0):\n"
    "        tcgen05_pipeline_init_barrier.arrive_and_wait()"
)
# One-tile-per-CTA kernels run the TMA-load role ahead of the publication;
# the role joins the barrier itself, so the prefix wait excludes that warp.
PUBLICATION_HOISTED = (
    "if not tcgen05_epi_active and (not tcgen05_tma_warp) or "
    "tcgen05_warp_idx == cutlass.Int32(0):\n"
    "        tcgen05_pipeline_init_barrier.arrive_and_wait()"
)


def _assert_merged_pipeline_init(
    kernel: str, *, launched_warps: int, private_mbarrier_inits: int = 0
) -> None:
    # Every pipeline defers its own fence + __syncthreads ...
    creates = re.findall(r"cutlass\.pipeline\.Pipeline\w+\.create\([^\n]*\)", kernel)
    assert creates, kernel
    for create in creates:
        if "PipelineTmaStore" in create:
            continue  # no mbarriers
        assert "defer_sync=True" in create, create
    # ... and one fence + the two named barriers publish all of them.  Only
    # mbarriers outside the pipeline set (the CLC response barrier, private
    # to the scheduler warp) fence themselves.
    assert kernel.count("cute.arch.mbarrier_init_fence()") == 1 + private_mbarrier_inits
    assert kernel.count("cute.arch.sync_threads()") == 0
    assert (
        "tcgen05_pipeline_init_barrier = cutlass.pipeline.NamedBarrier("
        f"barrier_id=3, num_threads={32 * (launched_warps - 4 + 1)})"
    ) in kernel
    hoisted = kernel.count(PUBLICATION_HOISTED)
    assert kernel.count(PUBLICATION) + hoisted == 1
    publication = kernel.index(PUBLICATION_HOISTED if hoisted else PUBLICATION)
    fence = kernel.rindex("cute.arch.mbarrier_init_fence()", 0, publication)
    alloc_wait = kernel.index("tcgen05_tmem_allocator.wait_for_alloc()")
    assert fence < publication < alloc_wait
    if hoisted:
        role_wait = kernel.index("tcgen05_pipeline_init_barrier.arrive_and_wait()")
        first_acquire = kernel.index("tcgen05_ab_pipeline.producer_acquire(")
        assert fence < role_wait < first_acquire < publication
        assert kernel.count("tcgen05_pipeline_init_barrier.arrive_and_wait()") == 2
    else:
        assert kernel.count("tcgen05_pipeline_init_barrier.arrive_and_wait()") == 1
    for create in creates:
        if "PipelineTmaStore" not in create:
            assert kernel.index(create) < fence
    # TMEM: warp 1 allocates (warp 0 initializes the mbarriers) and the
    # allocation permit is relinquished right after the allocation.
    assert kernel.count("allocator_warp_id=1") == 2
    assert kernel.count("allocator_warp_id=0") == 0
    allocate = kernel.index("tcgen05_tmem_allocator.allocate(")
    relinquish = kernel.index("tcgen05_tmem_allocator.relinquish_alloc_permit()")
    assert kernel.count("tcgen05_tmem_allocator.relinquish_alloc_permit()") == 1
    assert allocate < relinquish < fence


def test_plain_persistent_gemm_merges_pipeline_init_and_shortens_teardown() -> None:
    # 2048x4096 output / 128x64 tiles = 1024 tiles > 148 SMs: a real
    # persistent while loop, so the teardown is the post-loop one.
    kernel = _kernel_body(
        _source(_bind_gemm((2048, 4096, 1024), torch.bfloat16), _plain_config())
    )
    assert "while tcgen05_role_local_0_work_tile.is_valid_tile" in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=6)
    # Teardown: AB tail, MMA-warp arrive + acc tail, epilogue handshake +
    # free, and only then the TMA-store drain; no CTA-wide sync.
    ab_tail = kernel.index(
        "tcgen05_ab_pipeline.producer_tail(tcgen05_ab_producer_state)"
    )
    mma_arrive = kernel.index("tcgen05_tmem_alloc_barrier.arrive()")
    acc_tail = kernel.index(
        "tcgen05_acc_pipeline.producer_tail(tcgen05_acc_producer_state)"
    )
    epi_wait = kernel.index("tcgen05_tmem_alloc_barrier.arrive_and_wait()")
    free = kernel.index("tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)")
    store_tail = kernel.index("tcgen05_c_pipeline.producer_tail()")
    assert ab_tail < mma_arrive < acc_tail < epi_wait < free < store_tail
    assert kernel.count("tcgen05_tmem_alloc_barrier.arrive_and_wait()") == 1
    assert kernel.count("tcgen05_tmem_allocator.free(") == 1
    # The store drain is the last statement of the kernel.
    assert kernel.rstrip().endswith("tcgen05_c_pipeline.producer_tail()")


def test_one_tile_per_cta_frees_tmem_inside_the_epilogue() -> None:
    # 512x512 output / 128x64 tiles = 32 tiles <= 148 SMs, swizzle 1: every
    # CTA owns exactly one tile.
    kernel = _kernel_body(
        _source(_bind_gemm((512, 512, 256), torch.float16), _plain_config())
    )
    assert "while tcgen05_role_local" not in kernel
    assert "advance_to_next_work" not in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=6)
    # The last subtile's TMEM-load fence + acc release stay in the subtile
    # loop; the free follows the loop's last TMA store issue and precedes the
    # store drain, so the MMA-warp handshake overlaps the drain.
    release = kernel.index(
        "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)"
    )
    epi_wait = kernel.index("tcgen05_tmem_alloc_barrier.arrive_and_wait()")
    free = kernel.index("tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)")
    staging = kernel.index("cute.copy(tcgen05_tiled_copy_r2s")
    last_store = kernel.rindex("cute.copy(tcgen05_tma_store_atom")
    store_tail = kernel.index("tcgen05_c_pipeline.producer_tail()")
    assert release < staging < last_store < epi_wait < free < store_tail
    assert kernel.count("cute.copy(tcgen05_tma_store_atom") == 1
    assert kernel.count("tcgen05_tmem_alloc_barrier.arrive_and_wait()") == 1
    assert kernel.count("tcgen05_tmem_allocator.free(") == 1
    # The MMA warp still arrives once so the epilogue handshake completes.
    assert kernel.count("tcgen05_tmem_alloc_barrier.arrive()") == 1
    # Post-loop: nothing but the producer tails remains.
    mma_arrive = kernel.index("tcgen05_tmem_alloc_barrier.arrive()")
    assert free < mma_arrive < store_tail
    assert kernel.rstrip().endswith("tcgen05_c_pipeline.producer_tail()")


def test_batched_gemm_one_tile_per_cta_uses_one_shot_scheduler() -> None:
    # 8 batches x (256/128) x (256/64) = 64 tiles <= 148 SMs; the batch axis
    # is the scheduler's L dimension and counts toward the one-shot proof.
    kernel = _kernel_body(
        _source(
            _bind_bmm((8, 256, 256, 512), torch.float16),
            _plain_config(block_sizes=[1, 128, 64, 64], loop_orders=[[0, 1, 2]]),
        )
    )
    # The batch extent (8) is one of the three scheduler dims; its position
    # follows the loop order.
    params = re.search(r"PersistentTileSchedulerParams\(.*", kernel)
    assert params is not None, kernel
    assert "PersistentTileSchedulerParams((8, (" in params.group(
        0
    ) or ", 8), (1, 1, 1))" in params.group(0), params.group(0)
    assert "while tcgen05_role_local" not in kernel
    assert "advance_to_next_work" not in kernel
    assert ".tile_idx[2]" in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=6)
    assert kernel.count("tcgen05_tmem_allocator.free(") == 1
    assert kernel.rindex("cute.copy(tcgen05_tma_store_atom") < kernel.index(
        "tcgen05_tmem_allocator.free("
    )
    assert kernel.index("tcgen05_tmem_allocator.free(") < kernel.index(
        "tcgen05_c_pipeline.producer_tail()"
    )


FRESH_ACQUIRE = "tcgen05_ab_pipeline.producer_acquire(tcgen05_ab_producer_state, True)"
KLOOP_ACQUIRE = (
    "tcgen05_ab_pipeline.producer_acquire("
    "tcgen05_ab_producer_state, tcgen05_ab_producer_try_token)"
)


@pytest.mark.parametrize(
    ("bind", "config", "ab_stages", "fresh"),
    [
        (
            lambda: _bind_bmm((8, 256, 256, 512), torch.float16),
            lambda: _plain_config(
                block_sizes=[1, 128, 64, 64],
                loop_orders=[[0, 1, 2]],
                tcgen05_ab_stages=4,
            ),
            4,
            True,
        ),
        (
            lambda: _bind_gemm((1024, 1024, 1024), torch.bfloat16),
            lambda: _plain_config(
                block_sizes=[256, 256, 128],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_ab_stages=3,
            ),
            3,
            True,
        ),
        (
            lambda: _bind_gemm((2048, 4096, 1024), torch.bfloat16),
            lambda: _plain_config(tcgen05_ab_stages=2),
            2,
            False,
        ),
        (
            lambda: _bind_gemm((2048, 4096, 1024), torch.bfloat16),
            lambda: _plain_config(
                block_sizes=[256, 256, 128],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_ab_stages=3,
            ),
            3,
            False,
        ),
    ],
    ids=["one_shot_bmm", "one_shot_two_cta", "persistent", "persistent_two_cta"],
)
def test_initial_prefill_acquires_skip_the_fresh_ring_wait(
    bind: object, config: object, ab_stages: int, fresh: bool
) -> None:
    # A one-tile-per-CTA kernel prefills every stage of a ring nobody has
    # consumed from: the producer start phase makes the empty-barrier wait
    # pass by construction, so those acquires take ``True`` as the try-acquire
    # token (arming the full barrier without the try_wait round trip). A
    # multi-tile persistent kernel re-runs the prefill per tile on a live ring
    # and keeps the wait. The K loop's acquires always carry the real token.
    kernel = _kernel_body(_source(bind(), config()))  # pyrefly: ignore[not-callable]
    bare = "tcgen05_ab_pipeline.producer_acquire(tcgen05_ab_producer_state)"
    if fresh:
        assert "while tcgen05_role_local" not in kernel
        assert kernel.count(FRESH_ACQUIRE) == ab_stages
        assert bare not in kernel
        assert kernel.rindex(FRESH_ACQUIRE) < kernel.index(KLOOP_ACQUIRE)
    else:
        assert "while tcgen05_role_local" in kernel
        assert kernel.count(bare) == ab_stages
        assert FRESH_ACQUIRE not in kernel
    assert kernel.count(KLOOP_ACQUIRE) >= 1


def test_batched_gemm_with_more_tiles_than_sms_keeps_the_persistent_loop() -> None:
    kernel = _kernel_body(
        _source(
            _bind_bmm((64, 256, 256, 512), torch.float16),
            _plain_config(block_sizes=[1, 128, 64, 64], loop_orders=[[0, 1, 2]]),
        )
    )
    assert "while tcgen05_role_local_0_work_tile.is_valid_tile" in kernel
    # Multi-tile CTAs free TMEM in the teardown, after the last epilogue.
    assert kernel.index("tcgen05_tmem_allocator.free(") > kernel.index(
        "cute.copy(tcgen05_tiled_copy_r2s"
    )


def _assert_single_teardown_free(kernel: str) -> None:
    """TMEM is handshaken and freed exactly once, in the post-loop teardown.

    The MMA warp arrives once; a second epilogue ``arrive_and_wait`` would never
    complete and a second ``tcgen05.dealloc`` is illegal.
    """
    assert kernel.count("tcgen05_tmem_alloc_barrier.arrive()") == 1
    assert kernel.count("tcgen05_tmem_alloc_barrier.arrive_and_wait()") == 1
    assert kernel.count("tcgen05_tmem_allocator.free(") == 1
    mma_arrive = kernel.index("tcgen05_tmem_alloc_barrier.arrive()")
    epi_wait = kernel.index("tcgen05_tmem_alloc_barrier.arrive_and_wait()")
    free = kernel.index("tcgen05_tmem_allocator.free(tcgen05_epi_acc_tmem_ptr)")
    assert mma_arrive < epi_wait < free
    if "cute.copy(tcgen05_tiled_copy_r2s" in kernel:
        # After every output's TMEM read + SMEM staging.
        assert kernel.rindex("cute.copy(tcgen05_tiled_copy_r2s") < epi_wait


def test_two_output_stores_of_one_tile_free_tmem_once_in_the_teardown() -> None:
    # One tile per CTA, but the accumulator reaches two stores (default
    # ``tcgen05_epilogue_fanout="off"``).  The primary store registers the
    # teardown before the final store is lowered, so the in-epilogue free is
    # off and TMEM is freed once, after both outputs have read it.
    kernel = _kernel_body(
        _source(_bind_two_output_gemm((512, 512, 256), torch.bfloat16), _plain_config())
    )
    assert "while tcgen05_role_local" not in kernel
    assert "advance_to_next_work" not in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=6)
    assert kernel.count("cute.copy(tcgen05_tiled_copy_r2s") == 2
    _assert_single_teardown_free(kernel)
    # Both output pipelines drain after the free (short teardown order).
    free = kernel.index("tcgen05_tmem_allocator.free(")
    assert free < kernel.index("tcgen05_c_pipeline.producer_tail()")
    assert free < kernel.index("tcgen05_c_pipeline_1.producer_tail()")


def test_skip_epilogue_store_diagnostic_frees_tmem_in_the_teardown() -> None:
    # The diagnostic swaps in a store body without the t2r region (and thus
    # without the in-epilogue free); the teardown must keep its own free.
    kernel = _kernel_body(
        _source(
            _bind_gemm((512, 512, 256), torch.float16),
            _plain_config(
                tcgen05_c_store_mode="skip_epilogue_store",
                tcgen05_diagnostic_invalid_output=True,
            ),
        )
    )
    assert "while tcgen05_role_local" not in kernel
    assert "cute.copy(tcgen05_tiled_copy_r2s" not in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=6)
    _assert_single_teardown_free(kernel)


def test_compact_fragment_epilogue_of_one_tile_frees_tmem_in_the_teardown() -> None:
    # One tile per CTA, but the compact fragment epilogue never renders the
    # TMA store body (and so no in-epilogue free): the teardown keeps the
    # free, or the CTA exits with TMEM still allocated.
    kernel = _kernel_body(
        _source(
            _bind_compact_fragment(),
            _plain_config(
                block_sizes=[1, 128, 128, 32],
                loop_orders=[[0, 1, 2]],
                pid_type="persistent_interleaved",
            ),
        )
    )
    assert "for tcgen05_epi_position" in kernel
    assert "while tcgen05_role_local" not in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=6)
    _assert_single_teardown_free(kernel)


def test_pure_matmul_lifecycle_merges_the_pipeline_init() -> None:
    # The pure-matmul strategy is flat-grid only, so it never takes the
    # one-shot in-epilogue free; it does get the merged init, the warp-1
    # allocator and the short teardown through its own post-loop emitter.
    kernel = _kernel_body(
        _source(
            _bind_gemm((512, 512, 256), torch.float16),
            _plain_config(
                pid_type="flat", tcgen05_strategy="pure_matmul_role_lifecycle"
            ),
        )
    )
    assert (
        "tcgen05_pipeline_init_barrier = cutlass.pipeline.NamedBarrier("
        "barrier_id=3, num_threads=96)"
    ) in kernel
    assert kernel.count(PUBLICATION) == 1
    assert kernel.count("cute.arch.mbarrier_init_fence()") == 1
    assert kernel.count("allocator_warp_id=1") == 2
    assert kernel.count("allocator_warp_id=0") == 0
    _assert_single_teardown_free(kernel)
    free = kernel.index("tcgen05_tmem_allocator.free(")
    assert kernel.index("tcgen05_acc_pipeline.producer_tail(") < free
    assert free < kernel.index("tcgen05_c_pipeline.producer_tail()")


def test_clc_one_cta_scheduler_merges_the_pipeline_init() -> None:
    kernel = _kernel_body(
        _source(
            _bind_gemm((2048, 4096, 256), torch.bfloat16),
            _plain_config(
                pid_type="persistent_interleaved",
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_warp_spec_scheduler_warps=1,
                tcgen05_persistence_model="clc_persistent",
            ),
        )
    )
    assert "cute.arch.clc_response" in kernel
    assert "while tcgen05_role_local" in kernel
    # The sched pipeline joins the merged init; the CLC response mbarrier is
    # private to the scheduler warp, which initializes and fences it itself.
    assert "tcgen05_sched_pipeline = cutlass.pipeline.PipelineAsync.create(" in kernel
    assert "cute.arch.mbarrier_init(tcgen05_clc_mbar_smem_ptr, 1)" in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=8, private_mbarrier_inits=1)
    _assert_single_teardown_free(kernel)


@pytest.mark.parametrize("load_mode", ["simt", "tma"])
def test_c_input_warp_aux_pipeline_joins_the_merged_init(load_mode: str) -> None:
    kernel = _kernel_body(
        _source(
            _bind_bias_gemm((512, 512, 256), torch.bfloat16),
            _plain_config(
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_warp_spec_scheduler_warps=1,
                tcgen05_warp_spec_c_input_warps=1,
                tcgen05_aux_load_mode=load_mode,
            ),
        )
    )
    pipeline = "PipelineTmaAsync" if load_mode == "tma" else "PipelineAsync"
    assert f"tcgen05_aux_pipeline = cutlass.pipeline.{pipeline}.create(" in kernel
    assert "tcgen05_aux_pipeline.consumer_release" in kernel
    _assert_merged_pipeline_init(kernel, launched_warps=8)
    _assert_single_teardown_free(kernel)


def test_scheduler_warp_strategy_merges_the_sched_pipeline_init() -> None:
    kernel = _kernel_body(
        _source(
            _bind_gemm((2048, 4096, 1024), torch.bfloat16),
            _plain_config(
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_warp_spec_scheduler_warps=1,
            ),
        )
    )
    assert "tcgen05_sched_pipeline = cutlass.pipeline.PipelineAsync.create(" in kernel
    # 8 launched warps: 4 epilogue + MMA + TMA + scheduler + padding.
    _assert_merged_pipeline_init(kernel, launched_warps=8)


def test_two_cta_cluster_keeps_the_cluster_init_protocol() -> None:
    kernel = _kernel_body(
        _source(
            _bind_gemm((2048, 4096, 1024), torch.bfloat16),
            _plain_config(
                block_sizes=[256, 256, 128],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_ab_stages=3,
            ),
        )
    )
    assert "cutlass.pipeline.pipeline_init_arrive(" in kernel
    assert "cutlass.pipeline.pipeline_init_wait(" in kernel
    assert "tcgen05_pipeline_init_barrier" not in kernel
    assert "cute.arch.mbarrier_init_fence()" not in kernel
    assert "allocator_warp_id=0" in kernel
    assert "allocator_warp_id=1" not in kernel
    # The cluster rendezvous is the pipeline-init publication; the TMEM
    # allocation is published by ``wait_for_alloc`` alone (no CTA-wide sync
    # holds the TMA warp behind ``tcgen05.alloc``), and the permit is
    # relinquished right after the allocation instead of in the teardown.
    assert "cute.arch.sync_threads()" not in kernel
    relinquish = kernel.index("tcgen05_tmem_allocator.relinquish_alloc_permit()")
    assert kernel.index("tcgen05_tmem_allocator.allocate(") < relinquish
    assert relinquish < kernel.index("tcgen05_tmem_allocator.wait_for_alloc()")
    # This CPU-mocked bind is a multi-tile persistent kernel (the SM count
    # mock keeps it off the one-shot path), so the TMA-load role stays after
    # the prefix wait and the teardown frees TMEM after the store drain; the
    # one-shot hoist / epilogue free are covered on a real device by
    # ``test_matmul_mma_tcgen05_two_cta_one_shot_prologue_teardown``.
    assert "advance_to_next_work()" in kernel
    assert "if not tcgen05_tma_warp:" not in kernel
    assert kernel.index("cutlass.pipeline.pipeline_init_wait(") < kernel.index(
        "cute.arch.griddepcontrol_wait()"
    )
    assert kernel.index("tcgen05_c_pipeline.producer_tail()") < kernel.index(
        "tcgen05_tmem_allocator.free("
    )


def _lifecycle(**overrides: object) -> Tcgen05LifecycleContext:
    values: dict[str, object] = {
        "exec_active": "exec_w",
        "epi_active": "epi_w",
        "tma_warp": "tma_w",
        "tma_pipeline": "ab_pipe",
        "tma_producer_state": "ab_state",
        "acc_pipeline": "acc_pipe",
        "acc_producer_state": "acc_pstate",
        "acc_consumer_state": "acc_cstate",
        "tmem_alloc_barrier": "tmem_bar",
        "tmem_allocator": "tmem_alloc",
        "tmem_holding_buf": "tmem_buf",
        "tmem_dealloc_mbar_ptr": "tmem_mbar",
        "epi_acc_tmem_ptr": "epi_ptr",
        "acc_tmem_cols": "cols",
        "is_two_cta": False,
        "use_tma": True,
    }
    values.update(overrides)
    return Tcgen05LifecycleContext(**values)  # pyrefly: ignore[bad-argument-type]


def test_lifecycle_short_teardown_orders_free_before_store_drain() -> None:
    lines = _lifecycle(
        use_merged_pipeline_init=True,
        tmem_permit_released_early=True,
        tmem_allocator_warp=1,
    ).render_store_post_loop_lines(
        tma_store_pipeline_tail="if w0:\n    c.producer_tail()"
    )
    assert lines == [
        "if tma_w:\n    ab_pipe.producer_tail(ab_state)",
        "if exec_w:\n    tmem_bar.arrive()",
        "if exec_w:\n    acc_pipe.producer_tail(acc_pstate)",
        (
            "tmem_alloc = cutlass.utils.TmemAllocator(tmem_buf, "
            "barrier_for_retrieve=tmem_bar, allocator_warp_id=1, is_two_cta=False, "
            "two_cta_tmem_dealloc_mbar_ptr=tmem_mbar, num_allocated_columns=cols, "
            "initialize_mbarrier=False)"
        ),
        "if epi_w:\n    tmem_bar.arrive_and_wait()",
        "if epi_w:\n    tmem_alloc.free(epi_ptr)",
        "if w0:\n    c.producer_tail()",
    ]


def test_lifecycle_epilogue_free_leaves_only_the_tails_in_the_teardown() -> None:
    lifecycle = _lifecycle(
        use_merged_pipeline_init=True,
        tmem_permit_released_early=True,
        tmem_allocator_warp=1,
        free_tmem_in_epilogue=True,
    )
    assert lifecycle.render_epilogue_tmem_free_lines() == [
        "tmem_bar.arrive_and_wait()",
        (
            "tmem_alloc = cutlass.utils.TmemAllocator(tmem_buf, "
            "barrier_for_retrieve=tmem_bar, allocator_warp_id=1, is_two_cta=False, "
            "two_cta_tmem_dealloc_mbar_ptr=tmem_mbar, num_allocated_columns=cols, "
            "initialize_mbarrier=False)"
        ),
        "tmem_alloc.free(epi_ptr)",
    ]
    assert lifecycle.render_store_post_loop_lines(
        tma_store_pipeline_tail="if w0:\n    c.producer_tail()"
    ) == [
        "if tma_w:\n    ab_pipe.producer_tail(ab_state)",
        "if exec_w:\n    tmem_bar.arrive()",
        "if exec_w:\n    acc_pipe.producer_tail(acc_pstate)",
        "if w0:\n    c.producer_tail()",
    ]


def test_lifecycle_default_teardown_is_unchanged() -> None:
    lines = _lifecycle().render_store_post_loop_lines(
        tma_store_pipeline_tail="if w0:\n    c.producer_tail()"
    )
    assert lines[0] == "if w0:\n    c.producer_tail()"
    assert "cute.arch.sync_threads()" in lines
    assert lines[-2:] == [
        "if epi_w:\n    tmem_bar.arrive_and_wait()",
        "if epi_w:\n    tmem_alloc.free(epi_ptr)",
    ]
    assert "if epi_w:\n    tmem_alloc.relinquish_alloc_permit()" in lines
