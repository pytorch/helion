from __future__ import annotations

import ast
from itertools import starmap
import math
import os
from typing import Any
import unittest
from unittest.mock import patch

from packaging import version
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import HALF_DTYPE
from helion._testing import LONG_INT_TYPE
from helion._testing import CosSimilarity
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import check_example
from helion._testing import float32_matmul_precision
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfA10G
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfCudaSharedMemoryLessThan
from helion._testing import skipIfFn
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfPallas
from helion._testing import skipIfPallasInterpret
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import xfailIfPallas
from helion._testing import xfailIfPallasInterpret
from helion._testing import xfailIfPallasTpu
import helion.language as hl
from helion.runtime.config import Config
from helion.runtime.ref_mode import is_ref_mode_enabled

_orig_matmul_fp32_precision: str = "none"
_orig_cudnn_fp32_precision: str = "none"


def _compile_only(
    fn: helion.Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> object:
    bound = fn.bind(args)
    if kwargs:
        config = Config(
            # pyrefly: ignore [bad-argument-type]
            **kwargs
        )
    elif fn.configs:
        (config,) = fn.configs
    else:
        config = bound.config_spec.default_config()
    for key in bound.config_spec.unsupported_config_keys(config.config):
        config.config.pop(key, None)
    if is_ref_mode_enabled(bound.kernel.settings):
        bound._config = config
        return bound
    return bound.compile_config(config)


def _cute_wrapper_plans_from_code(code: str) -> list[dict[str, object]]:
    marker = "._helion_cute_wrapper_plans = "
    line = next(line for line in code.splitlines() if marker in line)
    payload = line.split(marker, 1)[1]
    freeze_prefix = "helion.runtime._freeze_cute_wrapper_plans("
    if payload.startswith(freeze_prefix) and payload.endswith(")"):
        payload = payload[len(freeze_prefix) : -1]
    return list(ast.literal_eval(payload))


def _deepgemm_segment_test_kernel(body: object) -> helion.Kernel:
    return helion.kernel(
        static_shapes=False,
        config=helion.Config(block_sizes=[64, 64, 64]),
    )(body)


def _deepgemm_segment_bad_store_row_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index + 1, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _deepgemm_segment_bad_store_mask_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = valid_rows & (local_m >= 0)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _deepgemm_segment_store_m_mask_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        store_m = work_tile_metadata[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def _deepgemm_segment_inverted_row_mask_body(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    m_total_aligned, k = a_packed.shape
    _g, n, k2 = b_grouped.shape
    assert k == k2
    assert work_tile_metadata.size(1) == 4
    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    block_k = hl.register_block_size(64)
    out = torch.zeros(
        m_total_aligned,
        n,
        dtype=a_packed.dtype,
        device=a_packed.device,
    )
    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), 64, n],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = valid_m < local_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a_blk = hl.load(
                a_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                b_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


_deepgemm_segment_bad_store_row_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_bad_store_row_body
)
_deepgemm_segment_bad_store_mask_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_bad_store_mask_body
)
_deepgemm_segment_store_m_mask_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_store_m_mask_body
)
_deepgemm_segment_inverted_row_mask_kernel = _deepgemm_segment_test_kernel(
    _deepgemm_segment_inverted_row_mask_body
)


def setUpModule() -> None:
    global _orig_matmul_fp32_precision, _orig_cudnn_fp32_precision
    _orig_matmul_fp32_precision = torch.backends.cuda.matmul.fp32_precision
    _orig_cudnn_fp32_precision = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"


def tearDownModule() -> None:
    torch.backends.cuda.matmul.fp32_precision = _orig_matmul_fp32_precision
    torch.backends.cudnn.conv.fp32_precision = _orig_cudnn_fp32_precision


@onlyBackends(["triton", "cute", "pallas"])
class TestExamples(RefEagerTestBase, TestCase):
    def test_add(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example("add", args, sum(args), block_sizes=[128, 1], flatten_loop=True)

    def test_add_loop_order(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "add", args, sum(args), block_sizes=[256, 128], loop_orders=[[1, 0]]
        )

    @skipIfCudaSharedMemoryLessThan(
        131072, reason="block sizes exceed device shared memory limit"
    )
    def test_matmul(self):
        args = (
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 512], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "matmul",
            args,
            args[0] @ args[1],
            block_sizes=[128, 128, 128],
        )

    def test_matmul_default(self):
        """Matmul without explicit block_sizes to exercise autotuner defaults."""
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "matmul",
            args,
            args[0] @ args[1],
        )

    @xfailIfPallas(
        "Pallas TPU clamps the N block to the lane width (128) which does"
        " not match the test's N=96 bias dimension"
    )
    def test_matmul_bias_epilogue_wrapper(self):
        from typing import Callable
        from typing import NamedTuple

        class BiasEpilogue(NamedTuple):
            bias: torch.Tensor

            @property
            def fn(
                self,
            ) -> Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]:
                bias = self.bias

                def epilogue(
                    acc: torch.Tensor, tile: tuple[torch.Tensor, ...]
                ) -> torch.Tensor:
                    return acc + bias[tile[1]]

                return epilogue

            def __call__(
                self, acc: torch.Tensor, tile: tuple[torch.Tensor, ...]
            ) -> torch.Tensor:
                return self.fn(acc, tile)

            @property
            def __closure__(self) -> tuple[Any, ...] | None:
                return self.fn.__closure__

        a = torch.randn([128, 64], device=DEVICE, dtype=torch.float32)
        b = torch.randn([64, 96], device=DEVICE, dtype=torch.float32)
        bias = torch.randn([96], device=DEVICE, dtype=torch.float32)

        check_example(
            "matmul",
            (a, b, BiasEpilogue(bias)),
            a @ b + bias,
            block_sizes=[32, 32, 32],
        )

    def test_matmul_bf16_tcgen05(self):
        """Matmul at 256^3 bf16 — fixture sized just above the cute
        tcgen05 admission floor (M >= 64 divisible by 64) so the cute
        autotune sub-sweep fires the ``uses_tcgen05`` codegen marker.
        """
        args = (
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
        )
        check_example(
            "matmul",
            args,
            args[0] @ args[1],
            block_sizes=[64, 64, 32],
        )

    @xfailIfPallas("missing barrier implementation")
    @skipIfTileIR("PassManager::run failed")
    @skipIfXPU("Split-K barrier not supported on XPU backend")
    def test_split_k_barrier(self):
        m, k, n = 64, 512, 64
        a = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        expected = a @ b

        check_example(
            "split_k_barrier",
            (a, b),
            expected,
            fn_name="split_k_matmul",
            block_sizes=[16, 8, 16, 16, 16],
            pid_type="persistent_blocked",
            split_k=64,
        )

    @xfailIfPallas("missing barrier implementation")
    @skipIfTileIR("PassManager::run failed")
    @skipIfRefEager("Test requires compiled kernel with specific config")
    def test_split_k_barrier_accuracy(self):
        """Test split_k_barrier with a shape that exposes accuracy issues.

        This test uses shape (64, 33, 64) where K is not divisible by split_k.
        The bug manifests after multiple kernel executions - errors accumulate
        due to improper handling of the tmp tensor across invocations.
        """
        from examples.split_k_barrier import split_k_matmul

        m, k, n = 64, 33, 64
        config = helion.Config(
            block_sizes=[16, 8, 16, 16, 16],
            pid_type="persistent_blocked",
            split_k=32,
        )

        # Compile once and reuse - this triggers the accumulating error bug
        torch.manual_seed(0)
        a0 = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
        b0 = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        bound = split_k_matmul.bind((a0, b0))
        compiled = bound.compile_config(config)

        # Run multiple iterations - errors accumulate starting around iteration 2-3
        for seed in range(5):
            torch.manual_seed(seed)
            a = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
            b = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
            expected = a @ b
            result = compiled(a, b)

            torch.testing.assert_close(
                result,
                expected,
                atol=1e-1,
                rtol=1e-2,
                msg=f"Accuracy failure at iteration {seed}",
            )

    def test_matmul_bwd(self):
        """Test backward pass for matmul via matmul_autograd."""
        mod = import_path(EXAMPLES_DIR / "matmul.py")
        # Set a fixed config to avoid autotuning in CI.  ``import_path`` caches
        # the module in ``sys.modules``, so ``mod.matmul`` is a process-wide
        # singleton shared with every other test that imports the matmul
        # example.  Restore ``configs`` on teardown so this mutation does not
        # leak: a leaked non-empty ``configs`` makes the kernel skip autotuning
        # entirely, which e.g. breaks ``test_cache``'s cache-miss assertions
        # when that test later runs the same singleton on the same worker.
        config = helion.Config(block_sizes=[16, 16, 16])
        original_configs = mod.matmul.configs
        self.addCleanup(setattr, mod.matmul, "configs", original_configs)
        mod.matmul.configs = [config]

        mat1 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat2 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )

        mat1_ref = mat1.detach().clone().requires_grad_(True)
        mat2_ref = mat2.detach().clone().requires_grad_(True)
        ref_out = torch.matmul(mat1_ref, mat2_ref)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        result = mod.matmul_autograd(mat1, mat2)
        result.backward(grad_out)

        torch.testing.assert_close(result, ref_out, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat1.grad, mat1_ref.grad, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat2.grad, mat2_ref.grad, atol=1e-1, rtol=1e-2)

    def test_addmm_bwd(self):
        """Test backward pass for addmm via addmm_autograd."""
        mod = import_path(EXAMPLES_DIR / "matmul.py")
        # Set a fixed config to avoid autotuning in CI.  ``mod.matmul`` is a
        # process-wide singleton (``import_path`` caches the module), so
        # restore ``configs`` on teardown to avoid leaking the mutation into
        # other tests (a non-empty ``configs`` makes the kernel skip
        # autotuning).  See ``test_matmul_bwd``.
        config = helion.Config(block_sizes=[16, 16, 16])
        original_configs = mod.matmul.configs
        self.addCleanup(setattr, mod.matmul, "configs", original_configs)
        mod.matmul.configs = [config]

        bias = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat1 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        mat2 = torch.randn(
            [128, 128], device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        alpha, beta = 2.0, 0.5

        bias_ref = bias.detach().clone().requires_grad_(True)
        mat1_ref = mat1.detach().clone().requires_grad_(True)
        mat2_ref = mat2.detach().clone().requires_grad_(True)
        ref_out = torch.addmm(bias_ref, mat1_ref, mat2_ref, alpha=alpha, beta=beta)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        result = mod.addmm_autograd(bias, mat1, mat2, alpha, beta)
        result.backward(grad_out)

        torch.testing.assert_close(result, ref_out, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(bias.grad, bias_ref.grad, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat1.grad, mat1_ref.grad, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(mat2.grad, mat2_ref.grad, atol=1e-1, rtol=1e-2)

    def test_matmul_layernorm_static_shapes(self):
        args = (
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "matmul_layernorm",
            args,
            torch.nn.functional.layer_norm(
                (args[0] @ args[1]),
                normalized_shape=(512,),
                weight=args[2],
                bias=args[3],
            ),
            block_sizes=[16, 16],
            static_shapes=True,
        )

    @onlyBackends(["pallas"])
    def test_matmul_layernorm_half_dtype_multi_k_tile(self):
        """Guards K-loop accumulator precision when inputs are half-precision.

        Across multiple K-tile iterations the partial-sum accumulator must
        stay in fp32 to keep the layernorm output within tight tolerance;
        regressions here surface as out-of-tolerance results on bf16/fp16.
        """
        m, k, n = 1024, 1024, 1024
        args = (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
        )
        expected = torch.nn.functional.layer_norm(
            (args[0] @ args[1]).to(torch.float32),
            normalized_shape=(n,),
            weight=args[2].to(torch.float32),
            bias=args[3].to(torch.float32),
        ).to(HALF_DTYPE)
        check_example(
            "matmul_layernorm",
            args,
            expected,
            block_sizes=[32, 256],
            static_shapes=True,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_matmul_layernorm_small_shapes_compile_on_cute(self):
        if _get_backend() != "cute":
            self.skipTest("CuTe-specific compile coverage")

        mod = import_path(EXAMPLES_DIR / "matmul_layernorm.py")
        args = (
            torch.randn([32, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
            torch.randn([128], device=DEVICE, dtype=torch.float32),
        )
        _compile_only(mod.matmul_layernorm, args, block_sizes=[16, 16])

    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe matmul+layernorm example is unsupported and too expensive in-process",
    )
    def test_matmul_layernorm_dynamic_shapes(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
            torch.randn([400], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "matmul_layernorm",
            args,
            torch.nn.functional.layer_norm(
                (args[0] @ args[1]),
                normalized_shape=(400,),
                weight=args[2],
                bias=args[3],
            ),
            block_sizes=[16, 16],
            static_shapes=False,
        )

    @unittest.skipIf(
        version.parse(torch.__version__.split("+")[0]) < version.parse("2.8"),
        "Requires torch 2.8+",
    )
    def test_bmm(self):
        args = (
            torch.randn([16, 512, 768], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([16, 768, 1024], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "bmm",
            args,
            torch.bmm(args[0], args[1]),
            block_sizes=[16, 16, 16, 16],
        )

    def test_bmm_non_divisible_k(self):
        args = (
            torch.randn([4, 128, 384], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([4, 384, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "bmm",
            args,
            torch.bmm(args[0], args[1]),
            block_sizes=[1, 128, 128, 256],
        )

    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_fp8_gemm(self):
        # Create FP32 tensors and convert to FP8
        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        y = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)

        # Convert to FP8 format
        x_fp8 = x.to(torch.float8_e4m3fn)
        y_fp8 = y.to(torch.float8_e4m3fn).T.contiguous().T

        args = (x_fp8, y_fp8)

        # Import the reference implementation
        mod = import_path(EXAMPLES_DIR / "fp8_gemm.py")
        scale_a = torch.tensor(1.0, device=DEVICE)
        scale_b = torch.tensor(1.0, device=DEVICE)
        expected = mod.reference_fp8_gemm_pytorch(x_fp8, y_fp8, scale_a, scale_b)

        check_example(
            "fp8_gemm",
            args,
            expected,
            block_sizes=[16, 16, 32],
            num_warps=4,
            num_stages=3,
        )

    def test_template_via_closure0(self):
        bias = torch.randn([1, 512], device=DEVICE, dtype=HALF_DTYPE)
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        check_example(
            "matmul",
            args,
            torch.relu(args[0] @ args[1] + bias),
            fn_name="matmul",
            emit_code=False,
            block_sizes=[64, 64, 16],
            loop_orders=[[0, 1]],
            num_warps=2,
            num_stages=4,
            indexing="pointer",
            l2_grouping=64,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("Failed on XPU - https://github.com/pytorch/helion/issues/795")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_template_via_closure1(self):
        bias = torch.randn([1, 512], device=DEVICE, dtype=HALF_DTYPE)
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        check_example(
            "matmul",
            args,
            torch.relu(args[0] @ args[1] + bias),
            fn_name="matmul",
            emit_code=False,
            block_sizes=[64, 64, 16],
            loop_orders=[[0, 1]],
            num_warps=2,
            num_stages=4,
            indexing="block_ptr",
            l2_grouping=64,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_template_via_closure2(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
            lambda x, _: torch.nn.functional.relu(x),
        )
        check_example(
            "matmul",
            args,
            torch.relu(args[0] @ args[1]),
            fn_name="matmul",
            emit_code=False,
            block_sizes=[64, 64, 16],
            loop_orders=[[0, 1]],
            num_warps=2,
            num_stages=4,
            indexing="block_ptr",
            l2_grouping=64,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            emit_code=False,
            block_size=1,
            num_warps=4,
            num_stages=1,
            indexing="block_ptr",
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax_looped(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            emit_code=False,
            block_size=1,
            num_warps=4,
            num_stages=1,
            indexing="block_ptr",
            reduction_loop=32,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax_decomposed(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            fn_name="softmax_decomposed",
            emit_code=False,
            block_size=1,
            num_warps=4,
            num_stages=1,
            indexing="block_ptr",
        )

    def test_softmax_two_pass(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            fn_name="softmax_two_pass",
            emit_code=False,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_softmax_two_pass_block_ptr(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "softmax",
            args,
            torch.nn.functional.softmax(*args, dim=1),
            fn_name="softmax_two_pass",
            emit_code=False,
            block_sizes=[8, 64],
            indexing="block_ptr",
        )

    def test_cross_entropy(self):
        n, v = 128, 1000
        logits = torch.randn(n, v, device=DEVICE, dtype=torch.float32)
        labels = torch.randint(0, v, (n,), device=DEVICE, dtype=LONG_INT_TYPE)
        # PyTorch cross_entropy requires Long labels for the reference
        expected = torch.nn.functional.cross_entropy(logits, labels.long())
        check_example(
            "cross_entropy",
            (logits, labels),
            expected,
        )

    def test_welford(self):
        s, d = 128, 1024
        weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
        bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
        x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)

        check_example(
            "welford",
            (weight, bias, x),
            torch.nn.functional.layer_norm(
                x,
                normalized_shape=(x.shape[-1],),
                weight=weight,
                bias=bias,
                eps=1e-05,
            ),
        )

    def test_low_mem_dropout(self):
        from examples.low_mem_dropout import low_mem_dropout
        from examples.low_mem_dropout import low_mem_dropout_bwd

        p = 0.25
        size = 1024
        block_size = 512
        seed = 123
        seed2 = 456
        x = torch.randn(size=(size,)).to(device=DEVICE)
        out_fwd = _compile_only(
            low_mem_dropout, (p, x, seed), block_sizes=[block_size]
        )(p, x, seed)

        grad_y = torch.ones_like(x)
        bwd = _compile_only(
            low_mem_dropout_bwd,
            (p, grad_y, seed),
            block_sizes=[block_size],
        )
        grad_x = bwd(p, grad_y, seed)

        grad_x2 = bwd(p, grad_y, seed2)

        mask_fwd = out_fwd != 0
        mask_bwd = grad_x != 0
        self.assertTrue(
            torch.equal(mask_fwd, mask_bwd),
            "Same elements should be dropped in fwd and bwd with the same seed",
        )

        mask_bwd2 = grad_x2 != 0
        self.assertFalse(
            torch.equal(mask_bwd, mask_bwd2),
            "Different elements should be dropped when using a different seed",
        )

        check_example(
            "low_mem_dropout",
            (p, grad_y, seed),
            grad_x,
            block_sizes=[block_size],
        )

    @skipIfPallasInterpret(
        "65536x1024x1280 GEMM is too slow under CPU interpret -- it exceeds the "
        "300s per-test timeout and (thread timeout method) kills the whole job"
    )
    @xfailIfPallasTpu("precision differences with bf16xint16 operations on pallas")
    @skipIfTileIR("precision differences with bf16xint16 operations on tileir")
    @skipIfRocm("precision differences with bf16xint16 operations on rocm")
    @skipIfXPU("precision differences with bf16xint16 operations on xpu")
    def test_bf16xint16(self):
        from examples.bf16xint16_gemm import reference_bf16xint16_pytorch

        m, k, n = 65536, 1024, 1280

        # The CuTe scalar matmul fallback accumulates each bf16xbf16 product in
        # full fp32 (it never rounds the per-element products back to bf16), so
        # it is *more* accurate than torch's bf16 tensor-core reference. The cute
        # output bit-matches a full-precision (IEEE fp32) products matmul cast to
        # bf16, so use that as the reference. ``setUpModule`` flips the default
        # matmul fp32 precision to TF32, which would make ``torch.matmul`` itself
        # lossy, so force IEEE fp32 for the reference computation.
        is_cute = _get_backend() == "cute"

        def expected(
            xt: torch.Tensor, wt: torch.Tensor, transpose: bool
        ) -> torch.Tensor:
            if not is_cute:
                return reference_bf16xint16_pytorch(xt, wt, transpose)
            if transpose:
                x_f32 = xt.to(torch.bfloat16).float()
                w_f32 = wt.float()
            else:
                x_f32 = xt.float()
                w_f32 = wt.to(torch.bfloat16).float()
            prev = torch.backends.cuda.matmul.fp32_precision
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            try:
                out = torch.matmul(x_f32, w_f32)
            finally:
                torch.backends.cuda.matmul.fp32_precision = prev
            return out.to(torch.bfloat16)

        x = torch.randn([m, k], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randint(-(2**15), 2**15 - 1, (k, n), device=DEVICE, dtype=torch.int16)

        check_example(
            "bf16xint16_gemm",
            (x, w),
            expected(x, w, False),
            fn_name="_bf16xint16_gemm",
        )

        x_int16 = torch.randint(
            -(2**15), 2**15 - 1, (m, k), device=DEVICE, dtype=torch.int16
        )
        w_bf16 = torch.randn([k, n], device=DEVICE, dtype=torch.bfloat16)

        check_example(
            "bf16xint16_gemm",
            (x_int16, w_bf16),
            expected(x_int16, w_bf16, True),
            fn_name="_int16xbf16_gemm",
        )

    def test_rms_norm_fwd(self):
        args = (
            torch.randn([128, 256], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256], device=DEVICE, dtype=HALF_DTYPE),
            1e-5,
        )
        # Import and use the reference implementation from rms_norm.py
        mod = import_path(EXAMPLES_DIR / "rms_norm.py")
        expected = mod.rms_norm_pytorch(*args)

        check_example(
            "rms_norm",
            args,
            (expected, None),  # Expected: (output, 1/rms)
            fn_name="rms_norm_fwd",
            block_sizes=[16],
            indexing="pointer",
        )

    def test_swiglu_bwd(self):
        """Test backward pass for swiglu."""
        x1, x2 = [
            torch.randn(1024, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(2)
        ]

        out = F.silu(x1) * x2

        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        args = (
            grad_out,
            x1,
            x2,
        )

        check_example(
            "swiglu",
            args,
            (x1.grad, x2.grad),
            fn_name="swiglu_bwd",
        )

    @parametrize("dtype", (torch.float32, HALF_DTYPE))
    def test_rms_norm_bwd(self, dtype):
        """Test backward pass for rms norm weight gradient."""
        batch_size, dim = 2048, 2048
        x = torch.randn([batch_size, dim], device=DEVICE, dtype=dtype)
        weight = torch.randn([dim], device=DEVICE, dtype=dtype, requires_grad=True)
        grad_out = torch.randn([batch_size, dim], device=DEVICE, dtype=dtype)
        eps = 1e-5

        # Compute forward pass to get rms
        from examples.rms_norm import rms_norm_fwd

        # Create configured kernel with explicit config
        config = helion.Config(block_size=32, num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(rms_norm_fwd.fn, config=config)
        y, rms = configured_kernel(x, weight, eps)

        # Compute expected gradients with PyTorch
        x_torch = x.detach().clone().requires_grad_(True)
        weight_torch = weight.detach().clone().requires_grad_(True)
        y_torch = torch.nn.functional.rms_norm(x_torch, [dim], weight_torch, eps)
        y_torch.backward(grad_out)

        # Test the kernel using check_example
        args = (
            grad_out,
            x,
            weight,
            rms,
        )

        # rms_norm_bwd_dw returns grad_weight
        check_example(
            "rms_norm",
            args,
            (x_torch.grad, weight_torch.grad),  # Expected: grad_weight
            fn_name="rms_norm_bwd",
            block_size=[32, 1],
            num_warps=4,
            num_stages=3,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_embedding_pointers(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "embedding",
            args,
            torch.nn.functional.embedding(*args),
            block_sizes=[1, 256],
            indexing="pointer",
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_embedding_block_ptr(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        check_example(
            "embedding",
            args,
            torch.nn.functional.embedding(*args),
            block_sizes=[8, 64],
            indexing="block_ptr",
            pid_type="xyz",
        )

    @skipIfTileIR("PassManager::run failed")
    def test_epilogue_subtiling_residual_gelu(self):
        m, k, n = 8192, 8192, 8192
        x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
        w = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.randn([n], device=DEVICE, dtype=HALF_DTYPE)
        residual = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        acc = x.float() @ w.float()
        expected = torch.nn.functional.gelu(
            acc * 1.25 + residual.float() * 0.5 + bias.float()
        ).half()
        block_sizes = [16, 16, 16] if _get_backend() == "cute" else [64, 64, 64]
        check_example(
            "epilogue_subtiling",
            (x, w, bias, residual),
            expected,
            fn_name="matmul_bias_residual_gelu_cast",
            block_sizes=block_sizes,
        )

    @skipIfTileIR("PassManager::run failed")
    def test_epilogue_subtiling_gelu_aux(self):
        m, k, n = 8192, 8192, 8192
        x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
        w = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.randn([n], device=DEVICE, dtype=HALF_DTYPE)
        acc = x.float() @ w.float()
        pre = acc * 1.25 + bias.float()
        expected = (
            torch.nn.functional.gelu(pre).half(),
            pre.half(),
        )
        block_sizes = [16, 16, 16] if _get_backend() == "cute" else [64, 64, 64]
        check_example(
            "epilogue_subtiling",
            (x, w, bias),
            expected,
            fn_name="matmul_bias_gelu_aux",
            block_sizes=block_sizes,
        )

    def test_attention_pointer(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            (torch.nn.functional.scaled_dot_product_attention(*args), None),
            block_sizes=[1, 64, 32],
            indexing="pointer",
        )

    def test_attention_output(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args),
            fn_name="attention_output",
            block_sizes=[1, 64, 32],
        )

    def test_causal_attention_output(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args, is_causal=True),
            fn_name="causal_attention_output",
            block_sizes=[1, 64, 32],
        )

    def test_biased_attention_output(self):
        args = (
            torch.randn(1, 2, 128, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(1, 2, 128, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(1, 2, 128, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(1, 2, 128, 128, dtype=HALF_DTYPE, device=DEVICE) * 0.25,
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(
                args[0],
                args[1],
                args[2],
                attn_mask=args[3],
            ),
            fn_name="biased_attention_output",
            block_sizes=[1, 128, 128],
            atol=5e-2,
            rtol=2e-2,
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("failure on XPU")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_attention_block_pointer(self):
        args = (
            torch.randn(2, 32, 1024, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            (torch.nn.functional.scaled_dot_product_attention(*args), None),
            block_sizes=[16, 32, 16],
            num_stages=1,
            indexing="block_ptr",
        )

    def test_attention_dynamic(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            (torch.nn.functional.scaled_dot_product_attention(*args), None),
            fn_name="attention_dynamic",
            block_sizes=[1, 64, 32],
        )

    def test_xsa(self):
        args = (
            torch.randn(2, 32, 1024, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 1024, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 1024, 64, dtype=HALF_DTYPE, device=DEVICE),
        )
        mod = import_path(EXAMPLES_DIR / "xsa.py")
        check_example(
            "xsa",
            args,
            mod.ref_xsa(*args),
            fn_name="xsa_kernel",
            block_sizes=[1, 64, 32],
        )

    def test_xsa_near_zero_v(self):
        q = torch.randn(2, 4, 128, 64, dtype=HALF_DTYPE, device=DEVICE)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        # Force ||V_i|| = 0 < eps so F.normalize's eps-clamp matters.
        v[..., 0, :] = 0.0
        args = (q, k, v)
        mod = import_path(EXAMPLES_DIR / "xsa.py")
        check_example(
            "xsa",
            args,
            mod.ref_xsa(*args),
            fn_name="xsa_kernel",
            block_sizes=[1, 64, 32],
        )

    @xfailIfPallas("slice-based stores not yet supported")
    def test_concat_simple(self):
        args = (
            torch.randn(512, 500, device=DEVICE),
            torch.randn(512, 512, device=DEVICE),
        )
        check_example(
            "concatenate",
            args,
            torch.cat(args, dim=1),
            fn_name="concat2d_dim1_simple",
        )

    def test_concat(self):
        args = (
            torch.randn(512, 500, device=DEVICE),
            torch.randn(512, 512, device=DEVICE),
        )
        check_example(
            "concatenate",
            args,
            torch.cat(args, dim=1),
            fn_name="concat2d_dim1",
        )

    @xfailIfPallas("BlockSpec tiling failure")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_concat_block_ptr(self):
        args = (
            torch.randn(222, 100, device=DEVICE),
            torch.randn(222, 151, device=DEVICE),
        )
        check_example(
            "concatenate",
            args,
            torch.cat(args, dim=1),
            fn_name="concat2d_dim1",
            indexing="block_ptr",
            block_sizes=[128, 64],
        )

    @skipIfPallas("TODO: follow up on timeout due to google-pytorch/torch_tpu@42d10ff")
    @xfailIfPallas("BlockSpec tiling failure")
    def test_jagged_dense_add(self):
        mod = import_path(EXAMPLES_DIR / "jagged_dense_add.py")
        args = (
            *mod.random_jagged_2d(500, 5000, device=DEVICE),
            torch.randn(500, 5000, device=DEVICE),
        )
        check_example(
            "jagged_dense_add",
            args,
            mod.jagged_dense_add_2d_reference(*args),
            fn_name="jagged_dense_add_2d",
        )

    @xfailIfPallas("Pallas rejects int64 inputs (jagged offsets)")
    @skipIfXPU("Jagged tensor operations not fully supported on XPU")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_dense_bmm(self):
        mod = import_path(EXAMPLES_DIR / "jagged_dense_bmm.py")
        seq_offsets, jagged, dense, bias = mod.random_input(
            D=32, K=24, batch_size=16, max_seq_len=32, dtype=torch.float32
        )
        args = (seq_offsets, jagged, dense, bias)
        check_example(
            "jagged_dense_bmm",
            args,
            mod.jagged_dense_bmm_reference(*args),
        )

    @skipIfRefEager("Test has skip_accuracy=True and doesn't call assert_close")
    def test_moe_matmul_ogs(self):
        mod = import_path(EXAMPLES_DIR / "moe_matmul_ogs.py")

        B = 1000  # tokens / rows
        K = 500  # hidden size
        N = 200  # output size
        n_experts = 30
        A = torch.randn(B, K, device=DEVICE, dtype=HALF_DTYPE)
        W = torch.randn(n_experts, K, N, device=DEVICE, dtype=HALF_DTYPE)
        top1_expert_per_token = torch.randint(n_experts, (B,), device=DEVICE)

        args = (A, W, top1_expert_per_token)
        helion_kernel_args = mod.moe_matmul_ogs_helion_kernel_args_gen(
            A, W, top1_expert_per_token
        )
        check_example(
            "moe_matmul_ogs",
            helion_kernel_args,
            mod.moe_matmul_ogs_reference(*args),
            block_sizes=[16, 16, 16],
            skip_accuracy=True,  # TODO(yf225): fix unstable numerics
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_matmul_split_k(self):
        args = (
            torch.randn(64, 1024, device=DEVICE),
            torch.randn(1024, 64, device=DEVICE),
        )
        check_example(
            "matmul_split_k",
            args,
            torch.matmul(*args),
            indexing="block_ptr",
            block_sizes=[16, 16, 32],
            split_k=8,
        )

    def test_sum(self):
        args = (torch.randn([512, 512], device=DEVICE, dtype=torch.float32),)
        check_example(
            "sum",
            args,
            torch.sum(args[0], dim=-1),
            fn_name="sum_kernel",
            block_sizes=[8],
        )

    def test_long_sum_manual(self):
        # longsum_manual uses hl.register_block_size to get a static bound for the
        # inner reduction loop, so range() receives a plain Python int — no JAX
        # tracer wrapping.  Use n=65536 (2x the 32768 block size) to exercise two
        # reduction loop iterations on Pallas.
        x = torch.randn([4, 65536], device=DEVICE, dtype=torch.float32)
        check_example(
            "long_sum",
            (x,),
            x.sum(-1),
            fn_name="longsum_manual",
        )

    def test_long_sum_manual_non_divisible(self):
        """Reduction loop OOB when block_size doesn't divide the reduction dim.

        longsum_manual uses dynamic shapes (static_shapes=False by default).
        Two different non-divisible N values exercise the runtime pad
        computation with different pad amounts.
        """
        for n in [50000, 40000]:
            x = torch.randn([4, n], device=DEVICE, dtype=torch.float32)
            check_example(
                "long_sum",
                (x,),
                x.sum(-1),
                fn_name="longsum_manual",
                block_sizes=[32768, 1],
            )

    @xfailIfPallas("JAX tracer error with dynamic shapes")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_mean(self):
        num_rows, max_cols = 32, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        feature_counts = torch.randint(
            1, M + 1, (num_rows,), dtype=torch.int32, device=DEVICE
        )
        args = (x_data, x_offsets, feature_counts, M)

        mod = import_path(EXAMPLES_DIR / "jagged_mean.py")
        expected = mod.reference_jagged_mean_kernel_pytorch(
            x_data, x_offsets, feature_counts, M
        )

        check_example(
            "jagged_mean",
            args,
            expected,
            fn_name="jagged_mean_kernel",
            block_sizes=[16, 8, 16],
        )

    @xfailIfPallas("requires triton module")
    @skipIfRefEager(
        "torch._higher_order_ops.associative_scan with tuple arg is not supported by ref eager mode yet"
    )
    def test_segment_reduction(self):
        num_nodes = 100
        num_edges = 1000
        num_features = 32
        dtype = torch.float32

        # Create sorted indices for segmented reduction
        indices = torch.randint(0, num_nodes, (num_edges,), device=DEVICE).sort()[0]
        input_data = torch.randn(num_edges, num_features, device=DEVICE, dtype=dtype)

        args = (indices, input_data, num_nodes)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "segment_reduction.py")
        expected = mod.segmented_reduction_pytorch(*args)

        check_example(
            "segment_reduction",
            args,
            expected,
            fn_name="segmented_reduction_helion",
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfXPU("failure on XPU")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_attention_persistent_interleaved_l2_grouping(self):
        """Test attention with persistent interleaved execution and L2 grouping for optimal performance."""
        args = (
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
        )

        check_example(
            "attention",
            args,
            (torch.nn.functional.scaled_dot_product_attention(*args), None),
            block_sizes=[16, 32, 16],
            num_stages=1,
            pid_type="persistent_interleaved",
            l2_grouping=4,
            indexing="block_ptr",
        )

    @skipIfFn(
        lambda: _get_backend() == "cute",
        "CuTe FP8 attention destabilizes later cute tests when it fails in-process",
    )
    @onlyBackends(["triton", "pallas"])
    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    @xfailIfPallasInterpret("unsupported torch.float8_e4m3fn dtype")
    def test_fp8_attention(self):
        batch = 2
        heads = 4
        seq_len = 256
        head_dim = 64

        # Create FP16 tensors
        q = torch.randn(
            batch, heads, seq_len, head_dim, dtype=HALF_DTYPE, device=DEVICE
        )
        k = torch.randn(
            batch, heads, seq_len, head_dim, dtype=HALF_DTYPE, device=DEVICE
        )
        v = torch.randn(
            batch, heads, seq_len, head_dim, dtype=HALF_DTYPE, device=DEVICE
        )

        # Import the module
        mod = import_path(EXAMPLES_DIR / "fp8_attention.py")

        # Prepare FP8 inputs using the module's preprocessing function
        q_fp8, k_fp8, v_fp8 = mod.preprocess_fp8_attention_inputs(q, k, v)
        args = (q_fp8, k_fp8, v_fp8, batch, heads)

        # Get expected output from kernel
        expected = mod.fp8_attention_pytorch(q, k, v)()

        check_example(
            "fp8_attention",
            args,
            expected,
            fn_name="fp8_attention_kernel",
            block_sizes=[64, 64],
            atol=0.2,
            rtol=0.1,
        )

    def test_layernorm_with_bias(self):
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)

        args = (x, [64], weight, bias)

        # layer_norm_fwd returns (out, mean, rstd)
        # We only check the output tensor, not mean/rstd
        expected_out = torch.nn.functional.layer_norm(*args)

        check_example(
            "layer_norm",
            args,
            (expected_out, None, None),  # Expected: (output, mean, rstd)
            fn_name="layer_norm_fwd",
            block_size=32,
            num_warps=4,
            num_stages=3,
        )

    def test_layernorm_no_bias(self):
        """Test forward pass for layer normalization without bias."""
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)

        args = (x, [64], weight, None)

        # layer_norm_fwd returns (out, mean, rstd)
        # We only check the output tensor, not mean/rstd
        expected_out = torch.nn.functional.layer_norm(*args)

        check_example(
            "layer_norm",
            args,
            (expected_out, None, None),  # Expected: (output, mean, rstd)
            fn_name="layer_norm_fwd",
            block_size=32,
            num_warps=4,
            num_stages=3,
        )

    def test_layernorm_reduction_not_divisible(self):
        """Reduction loop OOB when reduction_loops doesn't divide the reduction dim."""
        batch_size = 4
        dim = 48  # not divisible by reduction_loops=32
        x = torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)
        weight = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)

        args = (x, [dim], weight, bias, 1e-5)
        expected_out = torch.nn.functional.layer_norm(*args)

        check_example(
            "layer_norm",
            args,
            (expected_out, None, None),
            fn_name="layer_norm_fwd",
            block_size=1,
            reduction_loops=32,
        )

    def _run_layernorm_bwd(self, batch_size: int, dim: int, seed: int = 0) -> None:
        eps = 1e-4
        atol = 3e-2
        rtol = 5e-2

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x = -2.3 + 0.5 * torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)
        weight = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
        grad_out = torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)

        x_fp32 = x.to(torch.float32)
        mean = x_fp32.mean(dim=-1)
        var = x_fp32.var(dim=-1, unbiased=False)
        rstd = torch.rsqrt(var + eps)

        x_ref = x.clone().detach().requires_grad_(True)
        weight_ref = weight.clone().detach().requires_grad_(True)
        bias_ref = bias.clone().detach().requires_grad_(True)

        y_ref = torch.nn.functional.layer_norm(x_ref, [dim], weight_ref, bias_ref, eps)
        y_ref.backward(grad_out.detach())

        expected = (
            x_ref.grad.detach(),
            weight_ref.grad.detach(),
            bias_ref.grad.detach(),
        )

        args = (grad_out, x, mean, rstd, weight, True)

        check_example(
            "layer_norm",
            args,
            expected,
            fn_name="layer_norm_bwd",
            block_sizes=[32, 1],
            num_warps=4,
            num_stages=3,
            rtol=rtol,
            atol=atol,
        )

    @skipIfA10G("accuracy check fails on A10G GPUs")
    def test_layernorm_bwd(self):
        """Test combined backward pass for layer norm with bias."""
        self._run_layernorm_bwd(batch_size=32, dim=64)

    @xfailIfPallas("VMEM OOM: untiled block specs load full tensors")
    @skipIfA10G("accuracy check fails on A10G GPUs")
    def test_layernorm_bwd_large_batch(self):
        """Regression test: large batch, small dim."""
        self._run_layernorm_bwd(batch_size=1152 * 1000, dim=16, seed=1)

    def test_softmax_bwd(self):
        m, n = 2048, 2048
        x = torch.randn([m, n], device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        grad_out = torch.randn([m, n], device=DEVICE, dtype=torch.bfloat16)

        from examples.softmax import softmax_two_pass

        config = helion.Config(block_size=[128, 128], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(softmax_two_pass.fn, config=config)
        y = configured_kernel(x)

        x_torch = x.detach().clone().requires_grad_(True)
        y_torch = torch.nn.functional.softmax(x_torch, dim=-1)
        y_torch.backward(grad_out)

        check_example(
            "softmax",
            (grad_out, y),
            x_torch.grad,
            fn_name="softmax_bwd",
            rtol=1e-3,
            atol=1e-3,
        )

    def test_layernorm_without_bias(self):
        x = -2.3 + 0.5 * torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)

        args = (x, [64], weight, None)
        # Test returns (output, mean, rstd) tuple
        expected_out = torch.nn.functional.layer_norm(x, [64], weight)
        expected = (expected_out, None, None)
        check_example(
            "layer_norm",
            args,
            expected,
            fn_name="layer_norm_fwd",
            block_size=32,
            num_warps=4,
            num_stages=3,
        )

    @xfailIfPallas("JAX tracer error with dynamic shapes")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_softmax(self):
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        args = (x_data, x_offsets)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_softmax.py")
        expected = mod.reference_jagged_softmax_pytorch(x_data, x_offsets)

        check_example(
            "jagged_softmax",
            args,
            expected,
            fn_name="jagged_softmax_kernel",
            block_sizes=[16, 8, 16, 16],
        )

    @xfailIfPallas("tensor-derived if-predicates not supported")
    @skipIfXPU("Jagged tensor operations not fully supported on XPU")
    def test_jagged_hstu_attn(self):
        batch_size = 4
        max_seq_len = 64
        heads = 8
        head_dim = 32

        # Generate random sequence lengths
        min_seq_len = max_seq_len // 2
        seq_lengths = torch.randint(
            min_seq_len,
            max_seq_len + 1,
            (batch_size,),
            dtype=torch.int32,
            device=DEVICE,
        )
        seq_offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=DEVICE),
                torch.cumsum(seq_lengths, dim=0),
            ]
        )
        total_seq_len = int(seq_offsets[-1].item())

        # Create input tensors: [total_seq_len, heads, head_dim]
        q = torch.randn(
            (total_seq_len, heads, head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        k = torch.randn(
            (total_seq_len, heads, head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        v = torch.randn(
            (total_seq_len, heads, head_dim),
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        # The kernel expects: max_seq_len, alpha, q, k, v, seq_offsets
        alpha = 1.0 / v.size(2) ** 2
        args = (max_seq_len, alpha, q, k, v, seq_offsets)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_hstu_attn.py")
        expected = mod.reference_jagged_hstu_kernel_pytorch(
            q, k, v, seq_offsets, None, max_seq_len
        )

        # Patch to use core silu decomposition instead of inductor's custom decomposition from pytorch PR #171723.
        # This ensures consistent codegen both torch 2.9 (stable) and nightly versions.
        from torch._decomp.decompositions import silu
        import torch._inductor.decomposition as inductor_decomp

        # Clear cache since fast_random_decomps() caches a copy of decompositions
        if hasattr(inductor_decomp.fast_random_decomps, "cache_clear"):
            inductor_decomp.fast_random_decomps.cache_clear()
        with patch.dict(
            inductor_decomp.decompositions, {torch.ops.aten.silu.default: silu}
        ):
            check_example(
                "jagged_hstu_attn",
                args,
                expected,
                fn_name="_helion_jagged_attention_kernel",
                block_sizes=[16, 16],
                atol=1e-2,
                rtol=1e-2,
            )

    @parametrize(
        "helion_precision,torch_precision,atol,rtol",
        [
            ("highest", "highest", 1e-2, 1e-2),
            ("default", "medium", 1.5, 5e-2),
        ],
    )
    @xfailIfPallasInterpret(
        "The get/set_float32_matmul_precision API needs an actual backend"
    )
    @onlyBackends(["pallas"])
    def test_jagged_hstu_attn_2(self, helion_precision, torch_precision, atol, rtol):
        torch.manual_seed(0)
        num_sequnces = 4
        heads = 4
        head_dim = 64
        max_seq_len = 128
        alpha = 1.23
        attn_scale = 4.56

        lengths = torch.randint(
            max_seq_len // 2,
            max_seq_len + 1,
            (num_sequnces,),
            dtype=torch.int32,
        )
        seq_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(lengths, dim=0).to(torch.int32),
            ]
        ).to(DEVICE)
        L = int(seq_offsets[-1].item())

        q = torch.randn(L, heads, head_dim, dtype=torch.float32, device=DEVICE)
        k = torch.randn(L, heads, head_dim, dtype=torch.float32, device=DEVICE)
        v = torch.randn(L, heads, head_dim, dtype=torch.float32, device=DEVICE)

        args = (max_seq_len, alpha, attn_scale, q, k, v, seq_offsets)

        with float32_matmul_precision(torch_precision):
            mod = import_path(EXAMPLES_DIR / "jagged_hstu_attn_2.py")
            expected = mod.reference_jagged_hstu_attention(*args)

            # Patch to use core silu decomposition instead of inductor's custom decomposition from pytorch PR #171723.
            # This ensures consistent codegen across torch 2.9 (stable) and nightly versions.
            from torch._decomp.decompositions import silu
            import torch._inductor.decomposition as inductor_decomp

            if hasattr(inductor_decomp.fast_random_decomps, "cache_clear"):
                inductor_decomp.fast_random_decomps.cache_clear()

            with (
                patch.object(
                    mod.jagged_hstu_attention.settings,
                    "dot_precision",
                    helion_precision,
                ),
                patch.dict(
                    inductor_decomp.decompositions, {torch.ops.aten.silu.default: silu}
                ),
            ):
                # Clear the cache to ensure the modified settings are used.
                mod.jagged_hstu_attention._bound_kernels.clear()

                check_example(
                    "jagged_hstu_attn_2",
                    args,
                    expected,
                    fn_name="jagged_hstu_attention",
                    block_sizes=[128, 128],
                    cos_sim=CosSimilarity(dim=-1, min_similarity=0.999),
                    atol=atol,
                    rtol=rtol,
                )

    @xfailIfPallasTpu("tensor-derived if-predicates not supported")
    def test_grouped_gemm_jagged(self):
        # Build small jagged grouped GEMM inputs
        torch.manual_seed(0)
        G = 3
        K, N = 64, 64
        dtype = torch.bfloat16
        group_A = [
            torch.randn(32 * (i + 1), K, device=DEVICE, dtype=dtype).contiguous()
            for i in range(G)
        ]
        B_shared = torch.randn(K, N, device=DEVICE, dtype=dtype).contiguous()

        # Pack A and offsets
        M_sizes = [int(a.size(0)) for a in group_A]
        starts = [0]
        for m in M_sizes:
            starts.append(starts[-1] + m)
        group_offsets = torch.tensor(starts, device=DEVICE, dtype=torch.int32)
        A_packed = torch.cat(group_A, dim=0).contiguous()

        # Reference result
        expected = torch.cat([a @ B_shared for a in group_A], dim=0)

        # Run kernel and check
        args = (A_packed, B_shared, group_offsets)
        check_example(
            "grouped_gemm",
            args,
            expected,
            fn_name="grouped_gemm_jagged",
        )

    @xfailIfPallas("Pallas scatter: multiple indirect dims are not supported")
    def test_grouped_gemm_jagged_persistent(self):
        # Build small jagged grouped GEMM inputs
        torch.manual_seed(0)
        G = 3
        K, N = 64, 64
        dtype = torch.bfloat16
        group_A = [
            torch.randn(32 * (i + 1), K, device=DEVICE, dtype=dtype).contiguous()
            for i in range(G)
        ]
        B_shared = torch.randn(K, N, device=DEVICE, dtype=dtype).contiguous()

        # Pack A and offsets
        M_sizes = [int(a.size(0)) for a in group_A]
        starts = [0]
        for m in M_sizes:
            starts.append(starts[-1] + m)
        group_offsets = torch.tensor(starts, device=DEVICE, dtype=torch.int32)
        A_packed = torch.cat(group_A, dim=0).contiguous()

        # Reference result
        expected = torch.cat([a @ B_shared for a in group_A], dim=0)

        # Run kernel and check
        args = (
            A_packed,
            B_shared,
            group_offsets,
        )
        check_example(
            "grouped_gemm",
            args,
            expected,
            fn_name="grouped_gemm_jagged_persistent",
        )

    @onlyBackends(["cute"])
    def test_grouped_gemm_deepgemm_m_grouped_bf16_nt_contiguous(self):
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        use_generated_default = (
            os.environ.get("HELION_CUTE_MMA_IMPL", "").strip().lower() == "tcgen05"
        )
        if use_generated_default:
            from helion._compat import requires_cuda_version
            from helion._compiler.cute.mma_support import get_cute_mma_support

            if DEVICE.type != "cuda":
                self.skipTest("CuTeDSL generated segment path requires CUDA")
            if not requires_cuda_version("13"):
                self.skipTest("CuTeDSL generated segment path requires CUDA >= 13")
            with torch.cuda.device(DEVICE):
                major, _minor = torch.cuda.get_device_capability(DEVICE)
                if major < 10:
                    self.skipTest("CuTeDSL generated segment path requires SM100+")
                if not get_cute_mma_support().tcgen05_f16bf16:
                    self.skipTest(
                        "tcgen05 F16/BF16 MMA is not supported on this machine"
                    )

        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (188, 64, 40, 64) if use_generated_default else (64, 96, 128, 32),
            n=128 if use_generated_default else 64,
            k=64,
            m_alignment=224 if use_generated_default else 64,
            tail_padding=0 if use_generated_default else 64,
            device=DEVICE,
        )
        layout_has_valid_prefix_tiles = (
            mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(*args)
        )
        if use_generated_default:
            self.assertFalse(layout_has_valid_prefix_tiles)
            self.assertTrue(
                mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                    *args,
                    layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
                )
            )
            self.assertTrue(
                mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                    *args
                )
            )

        actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(bool(torch.all(actual[args[2] == -1] == 0).item()))

        row_hole_layout = args[2].clone()
        row_hole_layout[16:32] = -1
        row_hole_layout[32:64] = 0
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                args[0],
                args[1],
                row_hole_layout,
            )

    def _skip_unless_blackwell_grouped_gemm(self) -> None:
        from helion._compat import requires_cuda_version
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if DEVICE.type != "cuda":
            self.skipTest("Blackwell grouped GEMM requires CUDA")
        if not requires_cuda_version("13"):
            self.skipTest("Blackwell grouped GEMM requires CUDA >= 13")
        with torch.cuda.device(DEVICE):
            major, _minor = torch.cuda.get_device_capability(DEVICE)
            if major < 10:
                self.skipTest("Blackwell grouped GEMM requires SM100+")
            if not get_cute_mma_support().tcgen05_f16bf16:
                self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    def test_grouped_gemm_blackwell_nt_accepts_heterogeneous_fp16_metadata(self):
        from helion._compiler.cute import grouped_deepgemm

        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        group_A = tuple(
            torch.empty(m, k, device=DEVICE, dtype=torch.float16)
            for m, _n, k, _l in mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
        )
        group_B = tuple(
            torch.empty(n, k, device=DEVICE, dtype=torch.float16)
            for _m, n, k, _l in mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
        )
        out_groups = tuple(
            torch.empty(m, n, device=DEVICE, dtype=torch.float16)
            for m, n, _k, _l in mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
        )

        metadata = grouped_deepgemm._blackwell_grouped_gemm_nt_metadata(
            group_A,
            group_B,
            out_groups,
        )
        self.assertEqual(
            metadata.problems,
            tuple(
                starmap(
                    grouped_deepgemm.BlackwellGroupedGemmProblem,
                    mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
                )
            ),
        )
        self.assertEqual(
            grouped_deepgemm._compute_blackwell_total_clusters(
                metadata.problems,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            ),
            sum(
                math.ceil(m / 128) * math.ceil(n / 64)
                for m, n, _k, _l in mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES
            ),
        )
        self.assertFalse(metadata.a_mode0_major)
        self.assertFalse(metadata.b_mode0_major)
        self.assertFalse(metadata.c_mode0_major)

        bad_group_A = list(group_A)
        m0, k0 = int(group_A[0].size(0)), int(group_A[0].size(1))
        padded = torch.empty(m0, k0 + 1, device=DEVICE, dtype=torch.float16)
        bad_group_A[0] = padded[:, :k0]
        self.assertEqual(bad_group_A[0].shape, group_A[0].shape)
        self.assertEqual(bad_group_A[0].stride(1), 1)
        leading_stride_bytes = bad_group_A[0].stride(0) * bad_group_A[0].element_size()
        self.assertNotEqual(leading_stride_bytes % 16, 0)
        with self.assertRaisesRegex(ValueError, "leading stride.*16-byte"):
            grouped_deepgemm._blackwell_grouped_gemm_nt_metadata(
                tuple(bad_group_A),
                group_B,
                out_groups,
            )

    @skipIfRefEager("Argument validation does not execute a Helion kernel")
    def test_grouped_gemm_blackwell_nt_arg_builder_validates_alignment(self):
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        device = torch.device("cpu")

        self.assertEqual(
            mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            (
                (128, 128, 128, 1),
                (512, 128, 128, 1),
                (128, 256, 128, 1),
            ),
        )
        with self.assertRaisesRegex(ValueError, "K.*16-byte"):
            mod.make_blackwell_grouped_gemm_nt_args(
                ((128, 128, 130, 1),),
                dtype=torch.float16,
                device=device,
            )
        with self.assertRaisesRegex(ValueError, "N.*16-byte"):
            mod.make_blackwell_grouped_gemm_nt_args(
                ((128, 130, 128, 1),),
                dtype=torch.float16,
                device=device,
            )
        padded = torch.empty(128, 129, device=device, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "leading stride.*16-byte"):
            mod._blackwell_grouped_gemm_nt_generated_check_16b_alignment(
                padded[:, :128],
                major="K",
                mode0="M",
                tensor_name="A",
                group=0,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_direct_source(self):
        self._skip_unless_blackwell_grouped_gemm()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, _expected = mod.make_blackwell_grouped_gemm_nt_args(
            mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )
        prepared = mod._blackwell_grouped_gemm_nt_generated_args(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        self.assertIsNotNone(prepared)
        assert prepared is not None
        kernel_args, _out_tuple = prepared
        direct_pointers = kernel_args[-2]
        self.assertEqual(
            direct_pointers.detach().cpu().tolist(),
            [
                [int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr())]
                for a, b, out in zip(group_A, group_B, out_groups, strict=True)
            ],
        )

        bound = mod._blackwell_grouped_gemm_nt_generated_kernel.bind(kernel_args)
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        code = bound.to_triton_code(
            mod._blackwell_grouped_gemm_nt_generated_config(reserved_sms=4)
        )
        self.assertIn("PipelineTmaUmma.create(num_stages=3", code)
        self.assertIn("'external_direct_pointer_metadata': True", code)
        self.assertIn("'tcgen05_grouped_static_reserved_sms': 4", code)
        self.assertIn("tcgen05_grouped_direct_pointers", code)
        self.assertIn("tcgen05_grouped_direct_strides", code)
        self.assertIn("tcgen05_grouped_tensormap_a_addr = (", code)
        self.assertIn("tcgen05_grouped_tensormap_b_addr = (", code)
        self.assertIn("tcgen05_grouped_d_tensormap_addr = (", code)
        self.assertIn("cute.make_ptr(cutlass.Float16", code)
        self.assertNotIn(
            "tcgen05_grouped_tensormap_a_base = a_placeholder.iterator +",
            code,
        )
        self.assertNotIn(
            "tcgen05_grouped_tensormap_b_base = b_placeholder.iterator +",
            code,
        )
        self.assertNotIn(
            "tcgen05_grouped_d_tensormap_base = out_placeholder.iterator +",
            code,
        )
        self.assertNotIn("_cutlass_grouped_gemm_kernel", code)
        self.assertNotIn("GroupedGemmKernel", code)
        self.assertNotIn("torch.cat", code)
        self.assertNotIn("torch.stack", code)

        default_args, _default_expected = mod.make_blackwell_grouped_gemm_nt_args(
            mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        default_group_A, default_group_B = default_args
        default_out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(default_group_A, default_group_B, strict=True)
        )
        default_prepared = mod._blackwell_grouped_gemm_nt_generated_args(
            default_group_A,
            default_group_B,
            default_out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        self.assertIsNotNone(default_prepared)
        assert default_prepared is not None
        default_kernel_args, _default_out_tuple = default_prepared
        default_bound = mod._blackwell_grouped_gemm_nt_generated_kernel.bind(
            default_kernel_args
        )
        default_bound.env.config_spec.cute_tcgen05_search_enabled = True
        default_code = default_bound.to_triton_code(
            mod._blackwell_grouped_gemm_nt_generated_config()
        )
        self.assertIn("PipelineTmaUmma.create(num_stages=2", default_code)
        self.assertNotIn("tcgen05_grouped_static_reserved_sms", default_code)

    @skipIfRefEager("Generated config selection does not execute a Helion kernel")
    def test_grouped_gemm_blackwell_nt_generated_ab_stage_selector(self):
        captured_configs: list[Config] = []
        call_count = 0
        device = torch.device("cpu")

        class FakeConfigSpec:
            cute_tcgen05_search_enabled = False

        class FakeEnv:
            def __init__(self) -> None:
                self.config_spec = FakeConfigSpec()

        class FakeBound:
            def __init__(self) -> None:
                self.env = FakeEnv()

            def set_config(self, config: Config) -> None:
                captured_configs.append(config)

            def __call__(self, *args: object) -> None:
                nonlocal call_count
                call_count += 1

        def fake_bind(args: tuple[torch.Tensor, ...]) -> FakeBound:
            return FakeBound()

        def run_case(mod: Any, k: int) -> None:
            group_A = (torch.empty(128, k, device=device, dtype=torch.float16),)
            group_B = (torch.empty(64, k, device=device, dtype=torch.float16),)
            out_groups = (torch.empty(128, 64, device=device, dtype=torch.float16),)
            kernel_args = (torch.empty(1, device=device, dtype=torch.float16),)
            with patch.object(
                mod,
                "_blackwell_grouped_gemm_nt_generated_args",
                return_value=(kernel_args, out_groups),
            ) as generated_args:
                result = mod._blackwell_grouped_gemm_nt_generated(
                    group_A,
                    group_B,
                    out_groups,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                    use_2cta_instrs=False,
                )
            generated_args.assert_called_once_with(
                group_A,
                group_B,
                out_groups,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            )
            self.assertIs(result, out_groups)

        with patch.dict(
            os.environ,
            {
                "HELION_BACKEND": "cute",
                "HELION_CUTE_MMA_IMPL": "tcgen05",
                "HELION_AUTOTUNE_EFFORT": "none",
            },
            clear=False,
        ):
            mod: Any = import_path(EXAMPLES_DIR / "grouped_gemm.py")
            with (
                patch.object(
                    mod._blackwell_grouped_gemm_nt_generated_kernel,
                    "bind",
                    side_effect=fake_bind,
                ),
                patch.object(
                    mod,
                    "_blackwell_grouped_gemm_nt_generated_reserved_sms",
                    return_value=0,
                ),
            ):
                run_case(mod, 128)
                run_case(mod, 192)

        self.assertEqual(call_count, 2)
        self.assertEqual(len(captured_configs), 2)
        small_k_config = captured_configs[0].config
        large_k_config = captured_configs[1].config
        self.assertEqual(small_k_config["block_sizes"], [128, 64, 64])
        self.assertEqual(small_k_config["num_stages"], 2)
        self.assertNotIn("tcgen05_ab_stages", small_k_config)
        self.assertEqual(large_k_config["block_sizes"], [128, 64, 64])
        self.assertEqual(large_k_config["tcgen05_ab_stages"], 4)

    @onlyBackends(["cute"])
    def test_grouped_gemm_blackwell_nt_generated_reserved_sms_selector(self):
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        problem = mod._BlackwellGeneratedGemmProblem
        self.assertEqual(
            mod._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(16, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 160, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            3,
        )
        self.assertEqual(
            mod._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(128, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 192, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            0,
        )
        self.assertEqual(
            mod._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(16, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 192, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            3,
        )
        self.assertEqual(
            mod._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8192, 1280, 32),
                    problem(16, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 128, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            0,
        )
        self.assertEqual(
            mod._blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
                (
                    problem(8065, 1280, 32),
                    problem(128, 384, 1536),
                    problem(640, 1280, 16),
                    problem(640, 128, 16),
                ),
                num_sm=148,
                block_m=128,
                block_n=64,
            ),
            2,
        )
        self.assertNotIn(
            "tcgen05_grouped_static_reserved_sms",
            mod._blackwell_grouped_gemm_nt_generated_config().config,
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_declines_common_envelope(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE.clear()
        grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.clear()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, _expected = mod.make_blackwell_grouped_gemm_nt_args(
            ((128, 160, 80, 1),),
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )
        prepared = mod._blackwell_grouped_gemm_nt_generated_args(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        self.assertIsNone(prepared)

        with patch.object(
            mod._blackwell_grouped_gemm_nt_generated_kernel,
            "bind",
            side_effect=AssertionError("generated bind should not be reached"),
        ):
            self.assertIsNone(
                mod._blackwell_grouped_gemm_nt_generated(
                    group_A,
                    group_B,
                    out_groups,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                    use_2cta_instrs=False,
                )
            )

        self.assertEqual(grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE, {})
        self.assertEqual(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE, {})

    def _assert_blackwell_grouped_gemm_nt_correct(
        self,
        problem_sizes_mnkl,
        *,
        mma_tiler_mn,
    ) -> None:
        self._skip_unless_blackwell_grouped_gemm()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        if problem_sizes_mnkl is None:
            args, expected = mod.make_blackwell_grouped_gemm_nt_args(
                dtype=torch.float16,
                device=DEVICE,
            )
            problem_sizes_mnkl = mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES
        else:
            args, expected = mod.make_blackwell_grouped_gemm_nt_args(
                problem_sizes_mnkl,
                dtype=torch.float16,
                device=DEVICE,
            )

        group_A, group_B = args
        self.assertEqual(
            tuple(
                (int(a.size(0)), int(b.size(0)), int(a.size(1)), 1)
                for a, b in zip(group_A, group_B, strict=True)
            ),
            tuple(problem_sizes_mnkl),
        )
        actual = mod.blackwell_grouped_gemm_nt(
            *args,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        for group, (actual_group, expected_group) in enumerate(
            zip(actual, expected, strict=True)
        ):
            with self.subTest(group=group):
                torch.testing.assert_close(
                    actual_group,
                    expected_group,
                    rtol=1e-2,
                    atol=1e-1,
                )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_official_default_fp16_correct(self):
        self._assert_blackwell_grouped_gemm_nt_correct(
            None,
            mma_tiler_mn=(128, 128),
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_documented_mixed_fp16_correct(self):
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        self._assert_blackwell_grouped_gemm_nt_correct(
            mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
            mma_tiler_mn=(128, 64),
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_fp16_correct(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE.clear()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_blackwell_grouped_gemm_nt_args(
            mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        actual = mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE, {})
        for actual_group, expected_group, out_group in zip(
            actual,
            expected,
            out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_cache_freshness(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()

        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, expected = mod.make_blackwell_grouped_gemm_nt_args(
            mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        actual = mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 1)
        cached_launch = next(
            iter(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.values())
        )
        for actual_group, expected_group in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        group_A[0].zero_()
        updated_expected = mod._reference_blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_dtype=torch.float16,
        )
        actual = mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertIs(
            next(iter(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.values())),
            cached_launch,
        )
        for actual_group, expected_group in zip(
            actual,
            updated_expected,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        fresh_group_A = (group_A[0].clone(), *group_A[1:])
        actual = mod.blackwell_grouped_gemm_nt_direct(
            fresh_group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 2)
        for actual_group, expected_group in zip(
            actual,
            updated_expected,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        fresh_out_groups = tuple(torch.empty_like(out) for out in out_groups)
        actual = mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=fresh_out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 3)
        for actual_group, expected_group, fresh_out in zip(
            actual,
            updated_expected,
            fresh_out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, fresh_out)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        actual = mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 1)
        for actual_group, expected_group in zip(
            actual,
            updated_expected,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_graph_capture(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, expected = mod.make_blackwell_grouped_gemm_nt_args(
            mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 1)

        for out in out_groups:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured: list[tuple[torch.Tensor, ...]] = []
        with (
            patch.object(
                grouped_deepgemm,
                "blackwell_grouped_gemm_nt",
                side_effect=AssertionError(
                    "generated graph capture should not fallback"
                ),
            ),
            torch.cuda.graph(graph),
        ):
            captured.append(
                mod.blackwell_grouped_gemm_nt_direct(
                    group_A,
                    group_B,
                    out_groups=out_groups,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                )
            )
        for out in out_groups:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(len(captured), 1)
        for actual_group, expected_group, out_group in zip(
            captured[0],
            expected,
            out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_blackwell_nt_generated_explicit_outputs_graph_capture_lru_hit(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, _expected = mod.make_blackwell_grouped_gemm_nt_args(
            mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args

        def new_out_groups() -> tuple[torch.Tensor, ...]:
            return tuple(
                torch.empty(
                    int(a.size(0)),
                    int(b.size(0)),
                    device=DEVICE,
                    dtype=torch.float16,
                )
                for a, b in zip(group_A, group_B, strict=True)
            )

        out_first = new_out_groups()
        out_second = new_out_groups()
        mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_first,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 1)
        first_key = next(iter(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE))
        first_launch = mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE[first_key]

        mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_second,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 2)
        self.assertIn(first_key, mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE)
        self.assertIsNot(mod._BLACKWELL_GENERATED_LAST_STABLE_LAUNCH, first_launch)

        group_A[0].zero_()
        expected = mod._reference_blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_dtype=torch.float16,
        )
        torch.cuda.synchronize()

        for out in out_first:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured: list[tuple[torch.Tensor, ...]] = []
        with (
            patch.object(
                grouped_deepgemm,
                "blackwell_grouped_gemm_nt",
                side_effect=AssertionError(
                    "generated graph capture should not fallback"
                ),
            ),
            patch.object(
                mod._blackwell_grouped_gemm_nt_generated_kernel,
                "bind",
                side_effect=AssertionError("cached graph capture should not rebind"),
            ),
            torch.cuda.graph(graph),
        ):
            captured.append(
                mod.blackwell_grouped_gemm_nt_direct(
                    group_A,
                    group_B,
                    out_groups=out_first,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                )
            )

        for out in out_first:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(len(captured), 1)
        self.assertIs(mod._BLACKWELL_GENERATED_LAST_STABLE_LAUNCH, first_launch)
        for actual_group, expected_group, out_group in zip(
            captured[0],
            expected,
            out_first,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_explicit_outputs_retains_owned_metadata(
        self,
    ):
        self._skip_unless_blackwell_grouped_gemm()
        import helion.runtime as helion_runtime

        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.clear()
        args, expected = mod.make_blackwell_grouped_gemm_nt_args(
            mod.BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args

        def new_out_groups() -> tuple[torch.Tensor, ...]:
            return tuple(
                torch.empty(
                    int(a.size(0)),
                    int(b.size(0)),
                    device=DEVICE,
                    dtype=torch.float16,
                )
                for a, b in zip(group_A, group_B, strict=True)
            )

        out_groups = new_out_groups()
        actual = mod.blackwell_grouped_gemm_nt_direct(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        self.assertEqual(len(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE), 1)
        first_launch = next(iter(mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.values()))
        self.assertIsNotNone(first_launch.fast_call)
        self.assertIsNotNone(first_launch.runtime_cache_entry)
        runtime_cache_entry = first_launch.runtime_cache_entry
        retained_owned_tensors = runtime_cache_entry.owned_tensors
        self.assertGreater(len(retained_owned_tensors), 0)
        retained_owned_ptrs = tuple(
            int(tensor.data_ptr()) for tensor in retained_owned_tensors
        )
        for actual_group, expected_group in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )

        pressure_launches = helion_runtime._CUTE_LAUNCH_ARG_CACHE_LIMIT + 2
        self.assertLess(
            pressure_launches + 1,
            mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE_MAX,
        )
        pressure_out_groups = []
        for _ in range(pressure_launches):
            pressure_out = new_out_groups()
            pressure_out_groups.append(pressure_out)
            mod.blackwell_grouped_gemm_nt_direct(
                group_A,
                group_B,
                out_groups=pressure_out,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
            )
        torch.cuda.synchronize()
        self.assertIn(
            first_launch.cache_key, mod._BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE
        )
        self.assertIs(first_launch.runtime_cache_entry, runtime_cache_entry)
        self.assertEqual(
            tuple(int(tensor.data_ptr()) for tensor in retained_owned_tensors),
            retained_owned_ptrs,
        )

        junk_tensors = []
        for fill_value in range(64):
            for tensor in retained_owned_tensors:
                junk = torch.empty_like(tensor)
                junk.fill_(fill_value + 1024)
                junk_tensors.append(junk)

        group_A[0].zero_()
        updated_expected = mod._reference_blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_dtype=torch.float16,
        )
        for out in out_groups:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        with patch.object(
            mod._blackwell_grouped_gemm_nt_generated_kernel,
            "bind",
            side_effect=AssertionError("relaunch should use cached public launch"),
        ):
            actual = mod.blackwell_grouped_gemm_nt_direct(
                group_A,
                group_B,
                out_groups=out_groups,
                mma_tiler_mn=(128, 64),
                cluster_shape_mn=(1, 1),
            )
        torch.cuda.synchronize()
        for actual_group, expected_group, out_group in zip(
            actual,
            updated_expected,
            out_groups,
            strict=True,
        ):
            self.assertIs(actual_group, out_group)
            torch.testing.assert_close(
                actual_group,
                expected_group,
                rtol=1e-2,
                atol=1e-1,
            )
        self.assertEqual(len(junk_tensors), 64 * len(retained_owned_tensors))
        self.assertEqual(len(pressure_out_groups), pressure_launches)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_generated_documented_fp16_bf16_correct(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        for dtype, rtol, atol in (
            (torch.float16, 4e-2, 1e-1),
            (torch.bfloat16, 4e-2, 2e-1),
        ):
            with self.subTest(dtype=dtype):
                grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE.clear()
                torch.manual_seed(0)
                args, expected = mod.make_blackwell_grouped_gemm_nt_args(
                    mod.BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES,
                    dtype=dtype,
                    device=DEVICE,
                )
                actual = mod.blackwell_grouped_gemm_nt_direct(
                    *args,
                    mma_tiler_mn=(128, 64),
                    cluster_shape_mn=(1, 1),
                )
                torch.cuda.synchronize()
                self.assertEqual(
                    grouped_deepgemm._BLACKWELL_GROUPED_COMPILED_CACHE,
                    {},
                )
                for actual_group, expected_group in zip(
                    actual,
                    expected,
                    strict=True,
                ):
                    torch.testing.assert_close(
                        actual_group,
                        expected_group,
                        rtol=rtol,
                        atol=atol,
                    )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_blackwell_nt_prepares_stable_output_launch(self):
        self._skip_unless_blackwell_grouped_gemm()
        from helion._compiler.cute import grouped_deepgemm

        grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.clear()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_blackwell_grouped_gemm_nt_args(
            ((128, 128, 128, 1),),
            dtype=torch.float16,
            device=DEVICE,
        )
        group_A, group_B = args
        out_groups = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=DEVICE,
                dtype=torch.float16,
            )
            for a, b in zip(group_A, group_B, strict=True)
        )

        actual = mod.blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-1)
        self.assertEqual(len(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE), 1)
        prepared = next(
            iter(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.values())
        )
        self.assertEqual(len(prepared.stream_executions), 1)

        actual = mod.blackwell_grouped_gemm_nt(
            group_A,
            group_B,
            out_groups=out_groups,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-1)
        self.assertIs(
            next(iter(grouped_deepgemm._BLACKWELL_GROUPED_PREPARED_CACHE.values())),
            prepared,
        )

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            mod.blackwell_grouped_gemm_nt(
                group_A,
                group_B,
                out_groups=out_groups,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out_groups[0], expected[0], rtol=1e-2, atol=1e-1)

    def _skip_unless_deepgemm_segment_tcgen05(self) -> None:
        from helion._compat import requires_cuda_version
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if os.environ.get("HELION_CUTE_MMA_IMPL", "").strip().lower() != "tcgen05":
            self.skipTest("CuTeDSL generated segment path requires tcgen05")
        if not requires_cuda_version("13"):
            self.skipTest("CuTeDSL generated segment path requires CUDA >= 13")
        with torch.cuda.device(DEVICE):
            major, _minor = torch.cuda.get_device_capability(DEVICE)
            if major < 10:
                self.skipTest("CuTeDSL generated segment path requires SM100+")
            if not get_cute_mma_support().tcgen05_f16bf16:
                self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

    def _deepgemm_segment_metadata_fixture(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, _expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (188, 64, 40, 64),
            n=128,
            k=64,
            m_alignment=224,
            tail_padding=0,
            device=DEVICE,
        )
        work_tile_metadata = (
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
                *args
            )
        )
        self.assertIsNotNone(work_tile_metadata)
        return args, work_tile_metadata

    def _patch_torch_empty_for_deepgemm_output(
        self,
        replacement: torch.Tensor,
    ) -> object:
        real_empty = torch.empty

        def patched_empty(*shape: object, **kwargs: object) -> torch.Tensor:
            requested_shape = shape
            if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
                requested_shape = tuple(shape[0])
            requested_device = kwargs.get("device")
            if (
                tuple(int(dim) for dim in requested_shape) == tuple(replacement.shape)
                and kwargs.get("dtype") == replacement.dtype
                and requested_device is not None
                and torch.device(requested_device) == replacement.device
            ):
                return replacement
            return real_empty(*shape, **kwargs)

        return patch.object(torch, "empty", side_effect=patched_empty)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("selected generated segment requires compiled CuTe code")
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_source_and_route(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (188, 64, 40, 64),
            n=128,
            k=64,
            m_alignment=224,
            tail_padding=0,
            device=DEVICE,
        )
        A_packed, B_grouped, grouped_layout = args
        A_packed[grouped_layout == -1] = 0
        route_report = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report(
            *args
        )
        self.assertEqual(route_report.route, "generated_selected_segment")
        self.assertEqual(
            route_report.as_dict(),
            {
                "layout_has_valid_prefix_tiles": False,
                "use_generated_segment_kernel": True,
                "use_selected_segment_kernel": True,
                "selected_metadata_nonempty": True,
                "route": "generated_selected_segment",
            },
        )
        work_tile_metadata = route_report.selected_segment_work_tile_metadata
        self.assertIsNotNone(work_tile_metadata)
        assert work_tile_metadata is not None
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 188, 224],
                [1, 224, 64, 224],
                [2, 448, 40, 224],
                [3, 672, 64, 224],
            ],
        )

        selected_kernel = (
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_kernel
        )
        bound = selected_kernel.bind((A_packed, B_grouped, work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = selected_kernel.configs
        self.assertEqual(config["block_sizes"], [256, 128, 64])
        self.assertEqual(config["tcgen05_cluster_m"], 2)
        self.assertEqual(config["tcgen05_cluster_n"], 1)
        self.assertEqual(config["tcgen05_ab_stages"], 7)
        self.assertTrue(config["tcgen05_deepgemm_selected"])
        self.assertTrue(config["tcgen05_deepgemm_selected_compact_metadata"])
        self.assertEqual(config["tcgen05_selected_accumulator_view"], "nm")
        self.assertEqual(config["tcgen05_selected_d_store_view"], "nm_transposed")
        code = bound.to_triton_code(config)
        self.assertIn("tcgen05_grouped_selected_sched", code)
        self.assertIn("tcgen05_bSG_gD[None, cutlass.Int32(_tcgen05_subtile)]", code)
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertNotIn("CtaGroup.ONE", code)
        self.assertIn("out = torch.empty(", code)
        self.assertNotIn("out = torch.zeros(", code)
        self.assertNotIn(".zero_(", code)
        for helper_marker in (
            "_cutlass_grouped_gemm_kernel",
            "GroupedGemmKernel",
            "grouped_deepgemm",
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_zero_padding_rows",
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_output",
        ):
            self.assertNotIn(helper_marker, code)
        self.assertFalse(
            hasattr(
                mod,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_zero_padding_rows",
            )
        )
        self.assertFalse(
            hasattr(
                mod,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_output",
            )
        )

        grouped_plan = next(
            plan
            for plan in _cute_wrapper_plans_from_code(code)
            if plan["kind"] == "tcgen05_grouped_static_persistent"
        )
        self.assertIs(grouped_plan["deepgemm_selected"], True)
        self.assertIs(grouped_plan["deepgemm_selected_compact_metadata"], True)
        self.assertEqual(grouped_plan["source_m_tile"], 224)
        self.assertEqual(grouped_plan["cluster_m"], 2)
        self.assertEqual(grouped_plan["cluster_n"], 1)
        self.assertEqual(grouped_plan["selected_store_wave"], "nm_explicit_128x32")
        ab_plan = next(
            plan
            for plan in _cute_wrapper_plans_from_code(code)
            if plan["kind"] == "tcgen05_ab_tma"
        )
        self.assertEqual(ab_plan["ab_stage_count"], 7)
        selected_kernel.reset()

        with patch.object(
            mod,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(bool(torch.all(actual[grouped_layout == -1] == 0).item()))

        sentinel = torch.full_like(expected, -77.0)
        with (
            patch.object(
                mod,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
                side_effect=AssertionError("old segment fallback must not run"),
            ),
            self._patch_torch_empty_for_deepgemm_output(sentinel),
        ):
            actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        self.assertIs(actual, sentinel)
        torch.testing.assert_close(
            actual[grouped_layout != -1],
            expected[grouped_layout != -1],
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertTrue(bool(torch.all(actual[grouped_layout == -1] == 0).item()))

    @skipIfRefEager("Route report metadata inspection does not execute a Helion kernel")
    def test_grouped_gemm_deepgemm_route_report_metadata_does_not_alias_cache(self):
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, _expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (188, 64, 40, 64),
            n=128,
            k=64,
            m_alignment=224,
            tail_padding=0,
            device=torch.device("cpu"),
        )
        route_report = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report(
            *args
        )
        reported_metadata = route_report.selected_segment_work_tile_metadata
        self.assertIsNotNone(reported_metadata)
        assert reported_metadata is not None

        cached_metadata = mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(cached_metadata)
        assert cached_metadata is not None
        self.assertIsNot(reported_metadata, cached_metadata)
        expected_metadata = cached_metadata.detach().cpu().tolist()

        reported_metadata[0, 0] = 99
        self.assertEqual(cached_metadata.detach().cpu().tolist(), expected_metadata)

        second_report = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report(
            *args
        )
        second_metadata = second_report.selected_segment_work_tile_metadata
        self.assertIsNotNone(second_metadata)
        assert second_metadata is not None
        self.assertEqual(second_metadata.detach().cpu().tolist(), expected_metadata)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("CUDA graph capture requires compiled kernel")
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_graph_capture_padding(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (188, 64, 40, 64),
            n=128,
            k=64,
            m_alignment=224,
            tail_padding=0,
            device=DEVICE,
        )
        A_packed, _B_grouped, grouped_layout = args
        padding_rows = grouped_layout == -1
        valid_rows = ~padding_rows
        self.assertGreater(int(torch.count_nonzero(padding_rows).item()), 0)
        self.assertTrue(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                *args
            )
        )
        with patch.object(
            mod,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            warm = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(warm, expected, rtol=2e-2, atol=2e-2)

        A_packed[padding_rows] = 7
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with (
            patch.object(
                mod,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
                side_effect=AssertionError("old segment fallback must not run"),
            ),
            torch.cuda.graph(graph),
        ):
            captured = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()

        captured[valid_rows] = -5
        captured[padding_rows] = 13
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            captured[valid_rows],
            expected[valid_rows],
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertTrue(bool(torch.all(captured[padding_rows] == 0).item()))

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("selected generated segment requires compiled CuTe code")
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_full_overwrite(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (224, 224),
            n=128,
            k=64,
            m_alignment=224,
            tail_padding=0,
            device=DEVICE,
        )
        _A_packed, _B_grouped, grouped_layout = args
        self.assertEqual(int(torch.count_nonzero(grouped_layout == -1).item()), 0)
        self.assertTrue(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                *args
            )
        )
        work_tile_metadata = mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(work_tile_metadata)
        assert work_tile_metadata is not None
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 224, 224],
                [1, 224, 224, 224],
            ],
        )

        with patch.object(
            mod,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            warm = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(warm, expected, rtol=2e-2, atol=2e-2)

        sentinel = torch.full_like(expected, float("nan"))
        with (
            patch.object(
                mod,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
                side_effect=AssertionError("old segment fallback must not run"),
            ),
            self._patch_torch_empty_for_deepgemm_output(sentinel),
        ):
            actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        self.assertIs(actual, sentinel)
        self.assertFalse(bool(torch.any(torch.isnan(actual)).item()))
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("selected generated segment requires compiled CuTe code")
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_selected_segment_padding_only_chunk(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (224,),
            n=128,
            k=64,
            m_alignment=448,
            tail_padding=0,
            device=DEVICE,
        )
        A_packed, _B_grouped, grouped_layout = args
        A_packed[grouped_layout == -1] = 7
        self.assertTrue(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
                *args
            )
        )
        work_tile_metadata = mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            *args
        )
        self.assertIsNotNone(work_tile_metadata)
        assert work_tile_metadata is not None
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 224, 448],
            ],
        )

        with patch.object(
            mod,
            "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor",
            side_effect=AssertionError("old segment fallback must not run"),
        ):
            actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(
            actual[grouped_layout != -1],
            expected[grouped_layout != -1],
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertTrue(bool(torch.all(actual[grouped_layout == -1] == 0).item()))

    def _assert_segment_tcgen05_not_admitted(
        self,
        kernel: helion.Kernel,
        args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        work_tile_metadata: torch.Tensor,
    ) -> None:
        bound = kernel.bind((args[0], args[1], work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = kernel.configs
        code = bound.to_triton_code(config)
        self.assertNotIn("cute.nvgpu.tcgen05", code)
        self.assertNotIn("cute.gemm(", code)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("CUDA graph capture requires compiled kernel")
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_mixed_boundary(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (188, 64, 40, 64),
            n=128,
            k=64,
            m_alignment=224,
            tail_padding=0,
            device=DEVICE,
        )
        layout_has_valid_prefix_tiles = (
            mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(*args)
        )
        self.assertFalse(layout_has_valid_prefix_tiles)
        self.assertTrue(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                *args,
                layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
            )
        )
        self.assertFalse(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_output_fully_overwritten(
                *args
            )
        )

        work_tile_metadata = (
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
                *args
            )
        )
        self.assertIsNotNone(work_tile_metadata)
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 128, 0],
                [0, 128, 60, 0],
                [1, 224, 64, 0],
                [2, 448, 40, 0],
                [3, 672, 64, 0],
            ],
        )
        segment_kernel = mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_kernel
        bound = segment_kernel.bind((args[0], args[1], work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = segment_kernel.configs
        self.assertEqual(config["block_sizes"], [128, 128, 64])
        self.assertEqual(config["tcgen05_ab_stages"], 3)
        self.assertEqual(config["tcgen05_c_stages"], 4)
        code = bound.to_triton_code(config)
        self.assertIn("'cta_tile_shape_mnk': (128, 128, 64)", code)
        self.assertIn("'bn': 128", code)
        self.assertIn("'ab_stage_count': 3", code)
        self.assertIn("'c_stage_count': 4", code)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn("PipelineTmaUmma.create", code)
        self.assertIn("StaticPersistentGroupTileScheduler.create", code)
        self.assertIn("TensorMapManager", code)
        self.assertIn("update_tensormap", code)
        self.assertIn("tma_partition", code)
        self.assertIn("CtaGroup.ONE", code)
        self.assertNotIn("CtaGroup.TWO", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)
        self.assertNotIn("_cutlass_grouped_gemm_kernel", code)
        self.assertNotIn("dot_product", code)
        self.assertNotIn("for _load_i", code)
        self.assertIn("work_tile_metadata.size(0)", code)
        self.assertIn("'worklist_metadata': True", code)
        self.assertIn("'real_groups_arg':", code)
        self.assertNotIn("M_total_aligned + _BLOCK_SIZE_0 - 1", code)
        self.assertIn("out = torch.zeros(", code)
        self.assertNotIn("out = torch.empty(", code)

        actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        padding_rows = args[2] == -1
        self.assertTrue(bool(torch.all(actual[padding_rows] == 0).item()))

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()

        captured[padding_rows] = 13
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(bool(torch.all(captured[padding_rows] == 0).item()))

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfRefEager("CUDA graph capture requires compiled kernel")
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_full_overwrite_uses_empty(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (188, 68),
            n=128,
            k=64,
            m_alignment=1,
            tail_padding=0,
            device=DEVICE,
        )
        self.assertEqual(int(torch.count_nonzero(args[2] == -1).item()), 0)
        layout_has_valid_prefix_tiles = (
            mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(*args)
        )
        self.assertFalse(layout_has_valid_prefix_tiles)
        self.assertTrue(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                *args,
                layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
            )
        )
        self.assertTrue(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_output_fully_overwritten(
                *args
            )
        )

        work_tile_metadata = (
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
                *args
            )
        )
        self.assertIsNotNone(work_tile_metadata)
        self.assertEqual(
            work_tile_metadata.detach().cpu().tolist(),
            [
                [0, 0, 128, 0],
                [0, 128, 60, 0],
                [1, 188, 68, 0],
            ],
        )
        segment_kernel = mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_full_overwrite_kernel
        bound = segment_kernel.bind((args[0], args[1], work_tile_metadata))
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        (config,) = segment_kernel.configs
        code = bound.to_triton_code(config)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn("StaticPersistentGroupTileScheduler.create", code)
        self.assertIn("out = torch.empty(", code)
        self.assertNotIn("out = torch.zeros(", code)

        actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_rejects_store_mismatch(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        args, work_tile_metadata = self._deepgemm_segment_metadata_fixture()
        for name, kernel in (
            ("row", _deepgemm_segment_bad_store_row_kernel),
            ("mask", _deepgemm_segment_bad_store_mask_kernel),
            ("store_m", _deepgemm_segment_store_m_mask_kernel),
        ):
            with self.subTest(name=name):
                self._assert_segment_tcgen05_not_admitted(
                    kernel,
                    args,
                    work_tile_metadata,
                )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": "tcgen05",
            "HELION_AUTOTUNE_EFFORT": "none",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_rejects_inverted_row_mask(
        self,
    ):
        self._skip_unless_deepgemm_segment_tcgen05()
        args, work_tile_metadata = self._deepgemm_segment_metadata_fixture()
        self._assert_segment_tcgen05_not_admitted(
            _deepgemm_segment_inverted_row_mask_kernel,
            args,
            work_tile_metadata,
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_generated_segment_cache_keys_preserve_capture_metadata(
        self,
    ):
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        layout_validation_cache = mod._DEEPGEMM_LAYOUT_VALIDATION_CACHE
        layout_segments_cache = mod._DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE
        segment_tensor_cache = mod._DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE
        selected_unsupported_cache = (
            mod._DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE
        )
        saved_layout_validation_cache = layout_validation_cache.copy()
        saved_layout_segments_cache = layout_segments_cache.copy()
        saved_segment_tensor_cache = segment_tensor_cache.copy()
        saved_selected_unsupported_cache = selected_unsupported_cache.copy()
        try:
            layout_validation_cache.clear()
            layout_segments_cache.clear()
            segment_tensor_cache.clear()
            selected_unsupported_cache.clear()

            m_per_group = (188, 64, 40, 64)
            layout: list[int] = []
            for group_id, actual_m in enumerate(m_per_group):
                layout.extend([group_id] * actual_m)
                layout.extend([-1] * ((-actual_m) % 224))
            args = (
                torch.empty(len(layout), 64, device=DEVICE, dtype=torch.bfloat16),
                torch.empty(
                    len(m_per_group),
                    128,
                    64,
                    device=DEVICE,
                    dtype=torch.bfloat16,
                ),
                torch.tensor(layout, device=DEVICE, dtype=torch.int32),
            )
            legacy_metadata_key = mod._deepgemm_layout_validation_cache_key(
                *args,
                block_m=mod.DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
            )
            segment_worklist_key = mod._deepgemm_layout_validation_cache_key(
                *args,
                block_m=mod.DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M,
            )
            selected_worklist_key = mod._deepgemm_layout_validation_cache_key(
                *args,
                block_m=mod.DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_SOURCE_M_TILE,
            )
            self.assertNotEqual(legacy_metadata_key, segment_worklist_key)
            self.assertNotEqual(legacy_metadata_key, selected_worklist_key)
            self.assertNotEqual(segment_worklist_key, selected_worklist_key)

            layout_has_valid_prefix_tiles = (
                mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(*args)
            )
            self.assertFalse(layout_has_valid_prefix_tiles)
            self.assertIsNotNone(
                mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(*args)
            )
            work_tile_metadata = (
                mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
                    *args
                )
            )
            self.assertIsNotNone(work_tile_metadata)
            selected_work_tile_metadata = mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
                *args
            )
            self.assertIsNotNone(selected_work_tile_metadata)

            self.assertIn(legacy_metadata_key, layout_segments_cache)
            self.assertNotIn(segment_worklist_key, layout_segments_cache)
            self.assertNotIn(selected_worklist_key, layout_segments_cache)
            self.assertIn(segment_worklist_key, segment_tensor_cache)
            self.assertIn(selected_worklist_key, segment_tensor_cache)
            self.assertNotIn(legacy_metadata_key, segment_tensor_cache)

            with (
                patch.object(
                    mod,
                    "_parse_deepgemm_m_grouped_layout_segments",
                    side_effect=AssertionError(
                        "unexpected layout parse during capture"
                    ),
                ),
                patch.object(
                    mod,
                    "_deepgemm_cuda_graph_capture_active",
                    return_value=True,
                ),
            ):
                self.assertFalse(
                    mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(*args)
                )
                self.assertIsNotNone(
                    mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
                        *args
                    )
                )
                self.assertTrue(
                    mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                        *args,
                        layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
                    )
                )
                self.assertIs(
                    mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
                        *args
                    ),
                    work_tile_metadata,
                )
                self.assertIs(
                    mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
                        *args
                    ),
                    selected_work_tile_metadata,
                )
        finally:
            layout_validation_cache.clear()
            layout_validation_cache.update(saved_layout_validation_cache)
            layout_segments_cache.clear()
            layout_segments_cache.update(saved_layout_segments_cache)
            segment_tensor_cache.clear()
            segment_tensor_cache.update(saved_segment_tensor_cache)
            selected_unsupported_cache.clear()
            selected_unsupported_cache.update(saved_selected_unsupported_cache)

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @patch.dict(
        os.environ,
        {
            "HELION_BACKEND": "cute",
        },
        clear=False,
    )
    def test_grouped_gemm_deepgemm_valid_prefix_uses_legacy_generated_kernel(
        self,
    ):
        torch.manual_seed(0)
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")
        args, expected = mod.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (64, 96, 128, 32),
            n=64,
            k=64,
            m_alignment=64,
            tail_padding=64,
            device=DEVICE,
        )
        layout_has_valid_prefix_tiles = (
            mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(*args)
        )
        self.assertTrue(layout_has_valid_prefix_tiles)
        self.assertFalse(
            mod._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
                *args,
                layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
            )
        )

        with (
            patch.object(
                mod,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_generated_segment",
                side_effect=AssertionError("segment fallback must not run"),
            ),
            patch.object(
                mod,
                "_deepgemm_m_grouped_bf16_gemm_nt_contiguous_kernel",
                return_value=expected,
            ) as legacy_kernel,
        ):
            actual = mod.deepgemm_m_grouped_bf16_gemm_nt_contiguous(*args)
        legacy_kernel.assert_called_once_with(*args)
        self.assertIs(actual, expected)

    @onlyBackends(["cute"])
    def test_grouped_gemm_deepgemm_m_grouped_layout_parser_rejects_invalid(self):
        mod = import_path(EXAMPLES_DIR / "grouped_gemm.py")

        def validate_layout(layout: list[int]) -> None:
            rows = len(layout)
            a_packed = torch.empty(rows, 8, device=DEVICE, dtype=torch.bfloat16)
            b_grouped = torch.empty(4, 8, 8, device=DEVICE, dtype=torch.bfloat16)
            grouped_layout = torch.tensor(layout, device=DEVICE, dtype=torch.int32)
            mod._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
                a_packed,
                b_grouped,
                grouped_layout,
                block_m=4,
            )

        invalid_layouts = (
            ([0, 0, 2, 2, 1, 1], "strictly increasing"),
            ([0, 0, -1, 1, 1, -1, 1, 1], "strictly increasing"),
            ([0, 0, -1, 0], "strictly increasing"),
            ([0, -2, 1], "values must be -1"),
        )
        for layout, error in invalid_layouts:
            with (
                self.subTest(layout=layout),
                self.assertRaisesRegex(ValueError, error),
            ):
                validate_layout(layout)

    def test_geglu(self):
        args = (
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
        )
        check_example(
            "geglu",
            args,
            torch.nn.functional.gelu(args[0], approximate="tanh") * args[1],
            fn_name="_geglu",
            block_sizes=[1024],
            num_warps=4,
            num_stages=3,
        )

    def test_geglu_bwd(self):
        x1, x2 = [
            torch.randn(1024, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(2)
        ]

        out = torch.nn.functional.gelu(x1, approximate="tanh") * x2
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        args = (grad_out, x1, x2)

        check_example(
            "geglu",
            args,
            (x1.grad, x2.grad),
            fn_name="geglu_bwd",
            block_sizes=[16],
            num_warps=4,
            num_stages=3,
        )

    def test_swiglu(self):
        args = (
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
            torch.randn([256, 256], device=DEVICE, dtype=torch.bfloat16),
        )
        check_example(
            "swiglu",
            args,
            torch.nn.functional.silu(args[0]) * args[1],
            fn_name="_swiglu_fwd",
            block_sizes=[1024],
            num_warps=4,
            num_stages=3,
        )

    @xfailIfPallasInterpret(
        "JAX interpret cannot trace dynamic shapes (TypeError: JitTracer ~int32[])"
    )
    def test_jsd(self):
        args = (
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).log_softmax(
                dim=-1
            ),
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).log_softmax(
                dim=-1
            ),
            None,
        )

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jsd.py")
        expected = mod.TorchJSDBaseline()
        check_example(
            "jsd",
            args,
            (expected(*args), None),
            fn_name="jsd_forward",
            emit_code=False,
            block_sizes=[1, 4096],
            num_warps=4,
            num_stages=3,
        )

    @xfailIfPallasInterpret(
        "JAX interpret cannot trace dynamic shapes (TypeError: JitTracer ~int32[])"
    )
    def test_kl_div(self):
        args = (
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).log_softmax(
                dim=-1
            ),
            torch.randn([1024, 4096], device=DEVICE, dtype=torch.float32).softmax(
                dim=-1
            ),
        )
        torch_kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(
            device=DEVICE
        )
        check_example(
            "kl_div",
            args,
            torch_kl_div(*args),
            fn_name="kl_div_forward",
            emit_code=False,
            block_sizes=[1, 4096],
            num_warps=4,
            num_stages=3,
        )

    def _check_gather_gemv(self, dtype: torch.dtype):
        args = (
            torch.randn([4, 512, 512], device=DEVICE, dtype=dtype),
            torch.randint(0, 4, [2], device=DEVICE, dtype=torch.int32),
            torch.randn([512], device=DEVICE, dtype=dtype),
        )

        def expected(w, idx, x):
            return w[idx].to(x.dtype) @ x

        check_example(
            "gather_gemv",
            args,
            expected(*args),
            fn_name="gather_gemv",
            emit_code=False,
            block_sizes=[16, 16],
            num_warps=8,
            num_stages=1,
        )

    # Pallas f32 succeeds under CPU emulation but fails on TPU.
    @skipIfPallas("Pallas int32 gather coverage uses test_gather_gemv_half")
    @skipIfXPU("Timeout on XPU")
    def test_gather_gemv(self):
        self._check_gather_gemv(torch.float32)

    @skipIfXPU("Timeout on XPU")
    def test_gather_gemv_half(self):
        self._check_gather_gemv(HALF_DTYPE)

    @xfailIfPallas("int4 unpacking not supported on pallas")
    def test_int4_gemm(self):
        # Matrix dimensions
        M, K, N = 256, 512, 256

        # Create bfloat16 matrix A
        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)

        # Create packed int4 matrix B
        # Generate random int4 values in range [-8, 7]
        B_unpacked = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=DEVICE)

        # Pack two int4 values per int8
        B_reshaped = B_unpacked.reshape(K // 2, 2, N).permute(1, 0, 2)
        B_packed = ((B_reshaped[0] & 0xF) | (B_reshaped[1] << 4)).to(torch.int8)

        # Convert unpacked to bfloat16 for expected result
        B_unpacked_bf16 = B_unpacked.to(torch.bfloat16)
        expected = torch.matmul(A, B_unpacked_bf16)

        args = (A, B_packed)

        check_example(
            "int4_gemm",
            args,
            expected,
            fn_name="matmul_bf16_int4",
            block_sizes=[64, 64, 32],
            num_warps=4,
            num_stages=3,
            rtol=2e-1,
            atol=1.0,
        )

    @onlyBackends(["cute"])
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="NVFP4 conversion instructions require Blackwell"
    )
    @skipIfRefEager("inline asm codegen is not available in ref eager mode")
    def test_nvfp4_gemm(self):
        mod = import_path(EXAMPLES_DIR / "nvfp4_gemm.py")

        M, K, N = 64, 128, 64

        A = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
        W = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)

        W_quantized = mod.quantize_fp4_e2m1(W)
        W_packed = mod.pack_fp4(W_quantized).view(torch.float4_e2m1fn_x2)
        weight_scale = mod.make_fp8_scales((N, K // 16), DEVICE)

        result = mod.nvfp4_matmul(A, W_packed, weight_scale)
        expected = mod.reference_nvfp4_matmul(A, W_packed, weight_scale)
        torch.testing.assert_close(
            result,
            expected,
            atol=1.0,
            rtol=2e-1,
        )

        M, K, N = 128, 256, 256
        A_packed = mod.make_random_fp4((M, K), DEVICE)
        B_packed = mod.make_random_fp4((N, K), DEVICE)
        B_packed_t = B_packed.T
        scale_a = mod.make_fp8_scales((M, K // 16), DEVICE)
        scale_b = mod.make_fp8_scales((N, K // 16), DEVICE)

        result = mod.nvfp4_scaled_matmul(A_packed, B_packed_t, scale_a, scale_b)
        expected = mod.reference_nvfp4_scaled_matmul(
            A_packed,
            B_packed_t,
            scale_a,
            scale_b,
        )
        torch.testing.assert_close(
            result,
            expected,
            atol=1.0,
            rtol=2e-1,
        )

    @onlyBackends(["cute", "triton"])
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="NVFP4 conversion instructions require Blackwell"
    )
    @skipIfRefEager("inline asm codegen is not available in ref eager mode")
    def test_nvfp4_gemv(self):
        mod = import_path(EXAMPLES_DIR / "nvfp4_gemv.py")

        M, K_bytes = 64, 128
        weight = torch.randint(0, 256, (M, K_bytes), dtype=torch.uint8, device=DEVICE)
        weight_scale = mod.make_fp8_scales((M, K_bytes // 8), DEVICE)

        x_bf16 = torch.randn(K_bytes * 2, dtype=torch.bfloat16, device=DEVICE)
        bf16_result = mod.nvfp4_gemv_bf16in(weight, x_bf16, weight_scale)
        bf16_expected = mod.reference_nvfp4_gemv_bf16in(weight, x_bf16, weight_scale)
        torch.testing.assert_close(
            bf16_result,
            bf16_expected,
            atol=4.0,
            rtol=2e-1,
        )

        x_packed = torch.randint(0, 256, (K_bytes,), dtype=torch.uint8, device=DEVICE)
        x_scale = mod.make_fp8_scales((K_bytes // 8,), DEVICE)
        fp4_result = mod.nvfp4_gemv_fp4in(weight, x_packed, weight_scale, x_scale)
        fp4_expected = mod.reference_nvfp4_gemv_fp4in(
            weight, x_packed, weight_scale, x_scale
        )
        torch.testing.assert_close(
            fp4_result,
            fp4_expected,
            atol=4.0,
            rtol=2e-1,
        )

    @xfailIfPallas("JAX tracer error")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_sum(self):
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        args = (x_data, x_offsets)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_sum.py")
        expected = mod.reference_jagged_sum_kernel_pytorch(x_data, x_offsets)

        check_example(
            "jagged_sum",
            args,
            expected,
            fn_name="jagged_sum_kernel",
            block_sizes=[16, 8, 16],
        )

    @skipIfXPU("Timeout on XPU")
    def test_fused_linear_jsd(self):
        beta = 0.5
        ignore_index = -100
        temperature = 1.0
        m, n, k = 64, 128, 256

        student_input = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        teacher_input = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        student_weight = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        teacher_weight = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        student_logits = student_input @ student_weight.T
        teacher_logits = teacher_input @ teacher_weight.T

        args = (
            beta,
            ignore_index,
            temperature,
            student_logits,
            teacher_logits,
        )

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "fused_linear_jsd.py")
        # fused_linear_jsd_pytorch signature:
        # (beta, ignore_index, temperature, student_weight, teacher_weight, student_input, teacher_input)
        expected = mod.fused_linear_jsd_pytorch(
            beta,
            ignore_index,
            temperature,
            student_weight,
            teacher_weight,
            student_input,
            teacher_input,
        )

        check_example(
            "fused_linear_jsd",
            args,
            expected,
            fn_name="fused_linear_jsd_kernel",
            block_sizes=[64],
        )

    def test_fused_linear_jsd_fwd(self):
        """Exercise the autograd-wrapped FusedLinearJSDFunction path
        (FusedLinearJSDFunction.forward -> jsd_kernel per chunk).

        This is the user-facing API; the existing `test_fused_linear_jsd`
        only covers the simpler `fused_linear_jsd_kernel` JSD-only path.
        Shape picked so chunk_size > 1 (chunked path actually runs).
        """
        beta = 0.5
        ignore_index = -100
        # Use temperature = sqrt(hidden_dim) so logits/temperature has std ~1
        # and softmax stays in fp32 range.
        m, n, k = 128, 512, 1024
        temperature = float(n) ** 0.5

        student_input = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        teacher_input = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        student_weight = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
        teacher_weight = torch.randn([k, n], device=DEVICE, dtype=torch.float32)

        mod = import_path(EXAMPLES_DIR / "fused_linear_jsd.py")
        # Pin jsd_kernel's config so we skip autotune (CI speed). Block size 16
        # is a small safe pick across backends.  ``mod.jsd_kernel`` is a
        # process-wide singleton (``import_path`` caches the module); restore
        # the mutated ``configs`` / ``settings.static_shapes`` on teardown so
        # the changes don't leak into other tests.
        original_configs = mod.jsd_kernel.configs
        original_static_shapes = mod.jsd_kernel.settings.static_shapes
        self.addCleanup(setattr, mod.jsd_kernel, "configs", original_configs)
        self.addCleanup(
            setattr, mod.jsd_kernel.settings, "static_shapes", original_static_shapes
        )
        mod.jsd_kernel.settings.static_shapes = True
        mod.jsd_kernel.configs = [helion.Config(block_sizes=[16])]
        result = mod.fused_linear_jsd_fwd(
            beta,
            ignore_index,
            temperature,
            student_weight,
            teacher_weight,
            student_input,
            teacher_input,
        )
        expected = mod.fused_linear_jsd_pytorch(
            beta,
            ignore_index,
            temperature,
            student_weight,
            teacher_weight,
            student_input,
            teacher_input,
        )
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)

    @xfailIfPallas("JAX tracer error")
    @skipIfRefEager("hl.jagged_tile does not support ref mode yet")
    def test_jagged_layer_norm(self):
        num_rows, max_cols = 128, 64
        M = 8  # number of features
        lengths = torch.randint(1, max_cols + 1, (num_rows,), device=DEVICE)
        x_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=DEVICE),
                torch.cumsum(lengths, dim=0),
            ]
        )
        nnz = int(x_offsets[-1])
        x_data = torch.randn(nnz, M, dtype=torch.float32, device=DEVICE)
        eps = 1e-6
        args = (x_data, x_offsets, eps)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "jagged_layer_norm.py")
        expected = mod.reference_jagged_layer_norm_pytorch(x_data, x_offsets, eps)

        check_example(
            "jagged_layer_norm",
            args,
            expected,
            fn_name="jagged_layer_norm_kernel",
            block_sizes=[4, 8, 8, 8, 8, 8, 8],
        )

    def test_exp_fwd(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16)
        args = (x,)
        check_example(
            "exp",
            args,
            torch.exp(x),
            fn_name="exp_fwd",
            block_sizes=[16],
            num_warps=4,
            num_stages=3,
        )

    def test_exp_bwd(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16).requires_grad_(
            True
        )
        y = torch.exp(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        torch_out = x.grad
        args = (
            grad_out,
            y,
        )
        check_example(
            "exp",
            args,
            torch_out,
            fn_name="exp_bwd",
            block_sizes=[16],
            num_warps=4,
            num_stages=3,
        )

    @skipIfCudaSharedMemoryLessThan(
        131072, reason="block sizes exceed device shared memory limit"
    )
    @skipIfXPU("Squeeze-and-excitation network not supported on XPU")
    @xfailIfPallasInterpret("numerical mismatch in JAX interpret mode")
    def test_squeeze_and_excitation_net_fwd(self):
        m, n, k = 128, 128, 128
        x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
        a = torch.randn([n, k], device=DEVICE, dtype=torch.float32)
        b = torch.randn([k, n], device=DEVICE, dtype=torch.float32)

        args = (x, a, b)

        expected_out = torch.mul(x, torch.sigmoid(torch.relu(x @ a) @ b))
        c = torch.relu(x @ a)
        d = torch.sigmoid(c @ b)

        check_example(
            "squeeze_and_excitation_net",
            args,
            (expected_out, c, d),
            fn_name="squeeze_and_excitation_net_fwd",
            block_sizes=[128, 128, 128, 128],
            num_warps=4,
            num_stages=2,
            atol=0.15,
        )

    @xfailIfPallas("conflicting tiling patterns")
    @skipIfA10G("failure on a10g")
    @skipIfXPU("Squeeze-and-excitation network not supported on XPU")
    @skipIfTileIR("accuracy failure")
    def test_squeeze_and_excitation_net_bwd_dx(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        a = torch.randn([n, k], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch autograd
        x_torch = x.detach().clone().requires_grad_(True)
        a_torch = a.detach().clone().requires_grad_(True)
        b_torch = b.detach().clone().requires_grad_(True)
        out_torch = torch.mul(
            x_torch, torch.sigmoid(torch.relu(x_torch @ a_torch) @ b_torch)
        )
        out_torch.backward(grad_out)

        args = (grad_out, x, a, b, c, d)
        expected = x_torch.grad

        check_example(
            "squeeze_and_excitation_net",
            args,
            expected,
            fn_name="squeeze_and_excitation_net_bwd_dx",
            block_sizes=[16, 16, 16],
            num_warps=4,
            num_stages=2,
            atol=0.3,
        )

    @xfailIfPallas("tensor accessed with conflicting tiling patterns")
    @skipIfA10G("failure on a10g")
    @skipIfTileIR("accuracy failure")
    @skipIfXPU("ocloc compilation failure with 256-GRF kernel on XPU backend")
    def test_squeeze_and_excitation_net_bwd_da(self):
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        a = torch.randn([n, k], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch autograd
        x_torch = x.detach().clone().requires_grad_(True)
        a_torch = a.detach().clone().requires_grad_(True)
        b_torch = b.detach().clone().requires_grad_(True)
        out_torch = torch.mul(
            x_torch, torch.sigmoid(torch.relu(x_torch @ a_torch) @ b_torch)
        )
        out_torch.backward(grad_out)

        args = (grad_out, x, b, c, d)
        expected = a_torch.grad

        check_example(
            "squeeze_and_excitation_net",
            args,
            expected,
            fn_name="squeeze_and_excitation_net_bwd_da",
            block_sizes=[16, 16, 16],
            num_warps=4,
            num_stages=2,
            atol=0.3,
        )

    @skipIfA10G("failure on a10g")
    @skipIfTileIR("accuracy failure")
    @skipIfXPU("ocloc compilation failure with 256-GRF kernel on XPU backend")
    @xfailIfPallasInterpret(
        "pl.program_id captured into emit_pipeline body is not supported in "
        "JAX interpret mode (program_id_p.bind asserts during trace)"
    )
    def test_squeeze_and_excitation_net_bwd_db(self):
        torch.manual_seed(0)
        m, n, k = 256, 256, 256
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
        a = torch.randn([n, k], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        # Create configured kernel with explicit config
        from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd

        config = helion.Config(block_size=[16, 16, 16, 16], num_warps=4, num_stages=3)
        configured_kernel = helion.kernel(
            squeeze_and_excitation_net_fwd.fn, config=config
        )
        out, c, d = configured_kernel(x, a, b)

        # Create gradient for backward pass
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch autograd
        x_torch = x.detach().clone().requires_grad_(True)
        a_torch = a.detach().clone().requires_grad_(True)
        b_torch = b.detach().clone().requires_grad_(True)
        out_torch = torch.mul(
            x_torch, torch.sigmoid(torch.relu(x_torch @ a_torch) @ b_torch)
        )
        out_torch.backward(grad_out)

        args = (grad_out, x, d, c)
        expected = b_torch.grad

        check_example(
            "squeeze_and_excitation_net",
            args,
            expected,
            fn_name="squeeze_and_excitation_net_bwd_db",
            block_sizes=[16, 16, 16],
            num_warps=4,
            num_stages=2,
            atol=0.4,
        )

    def test_grpo_loss_fwd(self):
        """Test forward pass for GRPO loss."""
        B, L, V = 4, 512, 2048
        temperature = 0.9
        beta = 0.04
        eps_low = 0.2
        eps_high = 0.4

        torch.manual_seed(42)
        logits = torch.randn([B, L + 1, V], device=DEVICE, dtype=torch.bfloat16)
        completion_ids = torch.randint(0, V, (B, L), device=DEVICE, dtype=torch.int64)
        old_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        ref_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        advantages = torch.randn(B, device=DEVICE, dtype=torch.float32)
        completion_mask = torch.ones(B, L, device=DEVICE, dtype=torch.float32)

        from examples.grpo_loss import extract_selected_logits_pytorch

        selected_logits = extract_selected_logits_pytorch(
            logits[:, :-1, :], completion_ids, temperature
        )

        from examples.grpo_loss import torch_grpo_loss

        expected_loss, expected_kl, expected_clipped = torch_grpo_loss(
            logits.float(),
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        args = (
            logits,
            selected_logits,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        # grpo_loss_forward returns (loss, kl_loss, is_clipped, lse)
        # We only check loss, kl_loss, is_clipped (lse is None in expected)
        expected = (expected_loss, expected_kl, expected_clipped, None)

        check_example(
            "grpo_loss",
            args,
            expected,
            fn_name="grpo_loss_forward",
            rtol=1e-2,
            atol=1e-1,
            block_sizes=[4, 16, 16],
        )

    @xfailIfPallas("InductorLoweringError")
    def test_grpo_loss_bwd(self):
        """Test backward pass for GRPO loss."""
        B, L, V = 2, 64, 128
        temperature = 0.9
        beta = 0.04
        eps_low = 0.2
        eps_high = 0.4

        torch.manual_seed(42)
        logits = torch.randn(
            [B, L + 1, V], device=DEVICE, dtype=torch.bfloat16, requires_grad=True
        )
        completion_ids = torch.randint(0, V, (B, L), device=DEVICE, dtype=torch.int64)
        old_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        ref_logp = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
        advantages = torch.randn(B, device=DEVICE, dtype=torch.float32)
        completion_mask = torch.ones(B, L, device=DEVICE, dtype=torch.float32)

        # Pre-compute selected logits and run forward pass to get lse
        from examples.grpo_loss import extract_selected_logits_pytorch
        from examples.grpo_loss import grpo_loss_forward

        from helion._testing import code_and_output

        selected_logits = extract_selected_logits_pytorch(
            logits[:, :-1, :], completion_ids, temperature
        )

        forward_args = (
            logits,
            selected_logits,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        _, (_, _, _, lse) = code_and_output(
            grpo_loss_forward,
            forward_args,
            block_sizes=[4, 16, 16],
        )

        grad_output = torch.randn(B, L, device=DEVICE, dtype=torch.float32)

        logits_torch = logits.detach().clone().float().requires_grad_(True)
        from examples.grpo_loss import torch_grpo_loss

        loss_torch, _, _ = torch_grpo_loss(
            logits_torch,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            temperature,
            beta,
            eps_low,
            eps_high,
        )
        loss_torch.backward(grad_output)
        expected_grad = logits_torch.grad

        args = (
            grad_output,
            logits,
            selected_logits,
            completion_ids,
            old_logp,
            ref_logp,
            advantages,
            completion_mask,
            lse,
            temperature,
            beta,
            eps_low,
            eps_high,
        )

        check_example(
            "grpo_loss",
            args,
            expected_grad,
            fn_name="grpo_loss_backward",
            rtol=1e-2,
            atol=1e-1,
            block_sizes=[4, 16, 16],
        )

    @skipIfCudaSharedMemoryLessThan(
        131072, reason="block sizes exceed device shared memory limit"
    )
    def test_broadcast_matmul(self):
        args = (
            torch.randn([16, 512, 768], device=DEVICE, dtype=torch.float32),
            torch.randn([768, 1024], device=DEVICE, dtype=torch.float32),
        )
        check_example(
            "broadcast_matmul",
            args,
            torch.matmul(args[0], args[1]),
            block_sizes=[128, 128, 128],
        )

    def test_batch_softmax(self):
        args = (torch.randn([16, 512, 1024], device=DEVICE, dtype=torch.bfloat16),)
        check_example(
            "batch_softmax",
            args,
            torch.nn.functional.softmax(args[0], dim=-1),
            block_sizes=[1, 8],
        )

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_batch_softmax_block_ptr(self):
        args = (torch.randn([4, 128, 1024], device=DEVICE, dtype=torch.bfloat16),)
        check_example(
            "batch_softmax",
            args,
            torch.nn.functional.softmax(args[0], dim=-1),
            block_sizes=[1, 8],
            indexing="block_ptr",
        )

    @xfailIfPallasTpu("operation not supported on TPU")
    def test_gdn_fwd_h(self):
        """Test gated delta net forward h kernel."""
        batch = 2
        nheads = 4
        seqlen = 512
        chunk_size = 64
        dhead = 16
        dstate = 32

        k = torch.randn(
            batch, seqlen, nheads, dhead, dtype=torch.bfloat16, device=DEVICE
        )
        k = torch.nn.functional.rms_norm(k, (dhead,))
        w = torch.randn(
            batch,
            seqlen // chunk_size,
            chunk_size,
            nheads,
            dhead,
            dtype=torch.float32,
            device=DEVICE,
        )
        wu, ws, wv = torch.linalg.svd(w.permute(0, 1, 3, 2, 4), full_matrices=False)
        w = torch.einsum("bnhik,bnhkj->bnhij", wu, wv)
        w = (
            w.permute(0, 1, 3, 2, 4)
            .reshape(batch, seqlen, nheads, dhead)
            .to(torch.bfloat16)
        )
        u = torch.randn(
            batch, seqlen, nheads, dstate, dtype=torch.bfloat16, device=DEVICE
        )
        u = torch.nn.functional.rms_norm(u, (dstate,))
        g = torch.cumsum(
            0.5
            * math.log(1 / dhead)
            * torch.rand(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE),
            dim=1,
        )

        args = (k, w, u, g, chunk_size)

        # Import and use the reference implementation
        mod = import_path(EXAMPLES_DIR / "gdn_fwd_h.py")
        expected = mod.ref_gdn_fwd_h(*args)

        check_example(
            "gdn_fwd_h",
            args,
            expected,
            fn_name="helion_gdn_fwd_h",
        )

    @skipIfRocm("default config exceeds thread limit on ROCm")
    @skipIfTileIR("PassManager::run failed on TileIR backend")
    def test_long_sum(self):
        args = (torch.randn([4, 130000], device=DEVICE, dtype=torch.float32),)
        check_example(
            "long_sum",
            args,
            args[0].sum(-1),
            fn_name="longsum",
        )

    def test_long_sum_looped(self):
        args = (torch.randn([4, 130000], device=DEVICE, dtype=torch.float32),)
        check_example(
            "long_sum",
            args,
            args[0].sum(-1),
            fn_name="longsum_w_red_loop",
            block_sizes=[1],
            reduction_loops=[32768],
        )

    @skipIfRefEager("scalar_prefetch indexing not supported in ref interpreter")
    def test_flex_attention(self):
        z, h, n_ctx, head_dim = 2, 4, 256, 64
        q, k, v = [
            torch.randn((z, h, n_ctx, head_dim), dtype=HALF_DTYPE, device=DEVICE)
            for _ in range(3)
        ]

        mod = import_path(EXAMPLES_DIR / "flex_attention.py")
        # Set a fixed config to skip autotuning (exceeds CI timeout).
        # ``mod.helion_flex_attention_kernel`` is a process-wide singleton
        # (``import_path`` caches the module); restore ``configs`` on teardown
        # so the mutation doesn't leak into other tests.
        config = helion.Config(block_sizes=[64, 64])
        original_configs = mod.helion_flex_attention_kernel.configs
        self.addCleanup(
            setattr, mod.helion_flex_attention_kernel, "configs", original_configs
        )
        mod.helion_flex_attention_kernel.configs = [config]
        out = mod.helion_flex_attention(q, k, v)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)

    @xfailIfPallasTpu(
        "dA_cumsum has mixed scalar+slice access (VMEM), but Mosaic requires 32-bit for VMEM scalar extracts"
    )
    def test_mamba2_chunk_state(self):
        batch, nheads, ngroups, seqlen, chunk_size, headdim, dstate = (
            2,
            8,
            1,
            256,
            64,
            32,
            16,
        )
        nchunks = seqlen // chunk_size
        B = torch.rand(batch, seqlen, ngroups, dstate, dtype=HALF_DTYPE, device=DEVICE)
        x = torch.rand(batch, seqlen, nheads, headdim, dtype=HALF_DTYPE, device=DEVICE)
        dt = torch.rand(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        dA_cumsum = torch.rand(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        args = (B, x, dt, dA_cumsum)

        mod = import_path(EXAMPLES_DIR / "mamba2_chunk_state.py")
        expected = mod.ref_chunk_state(*args)

        check_example(
            "mamba2_chunk_state",
            args,
            expected,
            fn_name="helion_mamba2_chunk_state_kernel",
            atol=0.1,
            rtol=0.1,
        )

    @xfailIfPallasTpu("BlockSpec tiling failure")
    def test_mamba2_chunk_scan(self):
        batch, nheads, ngroups, seqlen, chunk_size, headdim, dstate = (
            2,
            8,
            1,
            256,
            64,
            32,
            16,
        )
        nchunks = seqlen // chunk_size
        cb = torch.zeros(
            batch,
            nchunks,
            ngroups,
            chunk_size,
            chunk_size,
            dtype=HALF_DTYPE,
            device=DEVICE,
        )
        torch.manual_seed(0)
        x = torch.zeros(batch, seqlen, nheads, headdim, dtype=HALF_DTYPE, device=DEVICE)
        dt = torch.zeros(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        dA_cumsum = torch.zeros(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        # Keep the recurrence path deterministic and non-zero so backend bugs
        # cannot hide behind allocator state when output buffers are recycled.
        C = torch.randn(batch, seqlen, ngroups, dstate, dtype=HALF_DTYPE, device=DEVICE)
        prev_states = torch.randn(
            batch,
            nchunks,
            nheads,
            headdim,
            dstate,
            dtype=HALF_DTYPE,
            device=DEVICE,
        )
        D_param = torch.zeros(nheads, dtype=HALF_DTYPE, device=DEVICE)
        args = (cb, x, dt, dA_cumsum, C, prev_states, D_param)

        mod = import_path(EXAMPLES_DIR / "mamba2_chunk_scan.py")
        expected = mod.ref_chunk_scan(*args)

        check_example(
            "mamba2_chunk_scan",
            args,
            expected,
            fn_name="helion_mamba2_chunk_scan_kernel",
            atol=0.1,
            rtol=0.1,
        )

    @skipIfRocm("failure on rocm")
    @skipIfA10G("failure on a10g")
    @skipIfCudaCapabilityLessThan((9, 0), reason="se_block CUDA path requires H100+")
    def test_se_block_fwd(self):
        m, n = 128, 128
        x = torch.randn([m, n], device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn([n, n], device=DEVICE, dtype=torch.bfloat16)

        # Compute expected output with PyTorch
        expected = 2 * x * torch.sigmoid(x @ w)

        args = (x, w)

        check_example(
            "se_block",
            args,
            (expected, None),  # (output, sigmoid)
            fn_name="se_block_fwd",
            block_sizes=[32],
            num_warps=4,
            num_stages=3,
        )

    @skipIfRocm("failure on rocm")
    @skipIfA10G("failure on a10g")
    @skipIfCudaCapabilityLessThan((9, 0), reason="se_block CUDA path requires H100+")
    def test_se_block_bwd_dx(self):
        m, n = 128, 128
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
        w = torch.randn([n, n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch
        x_torch = x.detach().clone().requires_grad_(True)
        w_torch = w.detach().clone().requires_grad_(True)
        out_torch = 2 * x_torch * torch.sigmoid(x_torch @ w_torch)
        out_torch.backward(grad_out)

        # Compute sigmoid values using PyTorch reference
        s = torch.sigmoid(x @ w)

        args = (grad_out, x, w, s)

        check_example(
            "se_block",
            args,
            x_torch.grad,
            fn_name="se_block_bwd_dx",
            block_sizes=[32, 32, 32],
            num_warps=4,
            num_stages=3,
        )

    @skipIfRocm("failure on rocm")
    @skipIfA10G("failure on a10g")
    @skipIfCudaCapabilityLessThan((9, 0), reason="se_block CUDA path requires H100+")
    def test_se_block_bwd_dw(self):
        m, n = 128, 128
        x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
        w = torch.randn([n, n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
        grad_out = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)

        # Compute expected gradients with PyTorch
        x_torch = x.detach().clone().requires_grad_(True)
        w_torch = w.detach().clone().requires_grad_(True)
        out_torch = 2 * x_torch * torch.sigmoid(x_torch @ w_torch)
        out_torch.backward(grad_out)

        # Compute sigmoid values using PyTorch reference
        s = torch.sigmoid(x @ w)

        args = (grad_out, x, s)

        check_example(
            "se_block",
            args,
            w_torch.grad,
            fn_name="se_block_bwd_dw",
            block_sizes=[32, 32, 32],
            num_warps=4,
            num_stages=3,
            rtol=1e-2,
        )


instantiate_parametrized_tests(TestExamples)


if __name__ == "__main__":
    unittest.main()
