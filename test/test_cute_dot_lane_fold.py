"""``hl.dot`` whose K axis is split across synthetic lanes folds the full K per
thread (``emit_cute_synthetic_lane_fold_mm``) instead of refusing the config."""

from __future__ import annotations

import unittest

import torch

from test._cute_binding import _cpu_bind

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
import helion.language as hl

# K (head_dim) on the third ``num_threads`` slot: 32 threads x 2 lanes.
_LANE_K_CONFIG = {"block_sizes": [256, 32], "num_threads": [0, 1, 32]}


def _fp8_attention_inputs(
    device: torch.device | str,
) -> tuple[object, tuple[object, ...], tuple[torch.Tensor, ...]]:
    """The example module, its kernel arguments and the fp16 (q, k, v) they
    were quantized from (the input of ``fp8_attention_pytorch``)."""
    mod = import_path(EXAMPLES_DIR / "fp8_attention.py")
    batch, heads, seq_len, head_dim = 1, 4, 256, 64
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    q_fp8, k_fp8, v_fp8 = mod.preprocess_fp8_attention_inputs(q, k, v)
    return mod, (q_fp8, k_fp8, v_fp8, batch, heads), (q, k, v)


def _fp8_attention_cpu_render(config: dict[str, object], **settings: object) -> str:
    mod, args, _ = _fp8_attention_inputs("cpu")
    kernel = helion.kernel(
        mod.fp8_attention_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        **settings,
    )
    return _cpu_bind(kernel, args).to_code(helion.Config(**config))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _e5m2_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    batch_heads, seq_len, _ = q.shape
    out = torch.empty(
        [batch_heads, seq_len, seq_len], dtype=torch.float32, device=q.device
    )
    for bh in hl.grid(batch_heads):
        for tile_m in hl.tile(seq_len):
            q_tile = q[bh, tile_m, :]
            for tile_n in hl.tile(seq_len):
                k_tile = k[bh, tile_n, :]
                out[bh, tile_m, tile_n] = hl.dot(q_tile, k_tile.transpose(0, 1))
    return out


@onlyBackends(["cute"])
class TestCuteDotLaneFold(unittest.TestCase):
    def test_fp8_attention_renders_with_k_on_lanes(self) -> None:
        """The example's QK ``hl.dot`` contracts head_dim; with head_dim split
        32 threads x 2 lanes the dot folds K through ``_cute_fold_dot`` (raw
        e4m3 bytes decoded in the helper, the outer-graph ``q`` load re-read
        through the grid index) and the lane splitter stashes it once per lane.
        """
        code = _fp8_attention_cpu_render(_LANE_K_CONFIG)
        self.assertIn("synthetic_lane_2 in range(2)", code)
        self.assertIn("_cute_fold_dot(q.iterator", code)
        self.assertIn("lhs_e4m3=True, rhs_e4m3=True", code)
        self.assertIn("k=64", code)
        # Computed once per KV lane, read back from the stash in later passes.
        self.assertRegex(code, r"_lane_stash_\d+_mm_fold_acc\[lane_3\] = mm_fold_acc")
        self.assertRegex(code, r"mm_fold_acc = _lane_stash_\d+_mm_fold_acc\[lane_3\]")
        self.assertNotIn("for mm_fold_k", code)
        # The PV dot keeps its in-thread running sum over the KV lanes.
        self.assertIn("dot_sum_0", code)

    def test_fold_step_uses_the_kernel_index_type(self) -> None:
        """The K step is spelled in the kernel's index type like the base
        pointer offsets, so an Int64 kernel (an input past 2**31 elements, or
        ``index_dtype=torch.int64``) cannot wrap a large K stride."""
        code = _fp8_attention_cpu_render(_LANE_K_CONFIG)
        self.assertRegex(code, r"cutlass\.Int32\(q\.layout\.stride\[\d\]\)")
        self.assertNotIn("Int64", code)
        code = _fp8_attention_cpu_render(_LANE_K_CONFIG, index_dtype=torch.int64)
        self.assertRegex(code, r"cutlass\.Int64\(q\.layout\.stride\[\d\]\)")
        self.assertNotRegex(code, r"cutlass\.Int32\(\w+\.layout\.stride")

    def test_fold_refuses_undecodable_storage(self) -> None:
        """Only e4m3 has a scalar decode: e5m2 operands keep the refusal."""
        q = torch.empty(2, 64, 64, dtype=torch.float8_e5m2)
        k = torch.empty(2, 64, 64, dtype=torch.float8_e5m2)
        bound = _cpu_bind(_e5m2_scores, (q, k))
        with self.assertRaisesRegex(exc.BackendUnsupported, "synthetic lanes"):
            bound.to_code(helion.Config(block_sizes=[64, 32], num_threads=[0, 1, 32]))

    def _check_k_on_lanes_against_thread_k(self, **settings: object) -> None:
        """Compile the K-on-lanes config and the thread-K config of the fp8
        example and compare their outputs.  The two accumulate the QK dot in
        fp32 in different orders (a sequential 64-term fold vs the 64-thread
        two-stage reduction), which can round an fp8 bucket either way: the
        outputs must agree within one dequantized fp8 ulp (the autotuner's
        rule, with no extra tolerance) and both must match the example's
        PyTorch reference within its tolerance; a bitwise pin would depend on
        the reduction tree, ptxas and the GPU.
        """
        from helion.autotuner.accuracy import assert_close

        mod, args, (q, k, v) = _fp8_attention_inputs(DEVICE)
        kernel = helion.kernel(
            mod.fp8_attention_kernel.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            **settings,
        )
        bound = kernel.bind(args)
        folded = bound.compile_config(helion.Config(**_LANE_K_CONFIG))(*args)
        reference = bound.compile_config(
            helion.Config(block_sizes=[256, 32], num_threads=[0, 1, 0])
        )(*args)
        assert_close(folded, reference, atol=0.0, rtol=0.0, fp8_accuracy="ulp")
        expected = mod.fp8_attention_pytorch(q, k, v)().float()
        for output in (folded, reference):
            torch.testing.assert_close(output.float(), expected, atol=0.2, rtol=0.1)

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_fp8_attention_k_on_lanes_matches_thread_k(self) -> None:
        self._check_k_on_lanes_against_thread_k()

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_fp8_attention_k_on_lanes_with_an_int64_index_type(self) -> None:
        """The fold's Int64 K step (``cutlass.Integer`` in the helper) compiles
        and reads the same elements."""
        self._check_k_on_lanes_against_thread_k(index_dtype=torch.int64)


if __name__ == "__main__":
    unittest.main()
