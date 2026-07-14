"""Correctness and perf tests for kernels under ``pretuned_kernels/``.

Correctness runs on every CUDA / ROCm runner; perf gating runs only on
the hardware where each kernel's checked-in heuristics apply.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import os
import sys
import unittest

import pytest
import torch
from torch._environment import is_fbcode
import torch.nn.functional as F

from helion import Config
from helion._hardware import get_hardware_info
from helion._testing import DEVICE
from helion._testing import PRETUNED_KERNELS_DIR
from helion._testing import TestCase
from helion._testing import is_cuda
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager


def _under_xdist() -> bool:
    return os.environ.get("PYTEST_XDIST_WORKER") is not None


def _current_compute_capability() -> str | None:
    try:
        return get_hardware_info().compute_capability
    except RuntimeError:
        return None


def _import_pretuned_kernel_module(name):
    # Flat private module name (no dotted parent package, which Helion's
    # global-scope resolution would try to import) avoids clashing with
    # ``examples/<name>.py``.
    module_name = f"_helion_pretuned_kernels_test_{name}"
    if module_name not in sys.modules:
        file_path = PRETUNED_KERNELS_DIR / name / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Register before exec so Helion can resolve kernels that reference
        # module-level globals (global_scope_origin does sys.modules[__name__]).
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def _run_pretuned_kernel_main_and_parse_summary(name):
    # main(verbose=False) returns the metrics dict directly (helion vs the best
    # available baseline) without printing the per-shape table.
    module = _import_pretuned_kernel_module(name)
    metrics = module.main(verbose=False)
    return {
        "helion_wins": int(metrics["helion_wins"]),
        "total": int(metrics["total"]),
        "geomean": float(metrics["geomean"]),
        "best_speedup": float(metrics["best_speedup"]),
    }


_CORRECTNESS_SHAPES = {
    "vector_add": [2**20],
    "attention": [(2, 8, 512, 64)],
    "softmax": [(4096, 1024)],
    "layer_norm": [(4096, 1024)],
    "rms_norm": [(2048, 4096)],
    "cross_entropy": [(4096, 32000)],
    "rope_fwd": [(2048, 2048)],
    "rope_bwd": [(2048, 2048)],
}

_KERNEL_MODULE_NAMES = {
    "rope_fwd": "rope",
    "rope_bwd": "rope",
}


def _make_vector_add_inputs(shape):
    n = shape
    x = torch.randn(n, device=DEVICE, dtype=torch.float32)
    y = torch.randn(n, device=DEVICE, dtype=torch.float32)
    return (x, y), lambda: x + y


def _make_attention_inputs(shape):
    z, h, seq_len, head_dim = shape
    q, k, v = (
        torch.randn(z, h, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        for _ in range(3)
    )
    return (q, k, v), lambda: F.scaled_dot_product_attention(q, k, v)


def _make_softmax_inputs(shape):
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
    return (x,), lambda: F.softmax(x, dim=1)


def _make_layer_norm_inputs(shape):
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
    w = torch.randn(n, device=DEVICE, dtype=torch.float16)
    b = torch.randn(n, device=DEVICE, dtype=torch.float16)
    return (x, w, b), lambda: F.layer_norm(x, [n], w, b, eps=1e-5)


def _make_rms_norm_inputs(shape):
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
    return (x, w), lambda: F.rms_norm(x, [n], w, eps=1e-5)


def _make_cross_entropy_inputs(shape):
    tokens, vocab = shape
    logits = torch.randn(tokens, vocab, device=DEVICE, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (tokens,), device=DEVICE, dtype=torch.int64)
    return (logits, labels), lambda: F.cross_entropy(logits, labels)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


def _rope_fwd_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _rope_bwd_reference(
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def grad_ref(grad_out: torch.Tensor) -> torch.Tensor:
        half_dim = grad_out.shape[-1] // 2
        grad_first_out = grad_out[..., :half_dim]
        grad_second_out = grad_out[..., half_dim:]
        cos_first = cos[:, None, :, :half_dim]
        cos_second = cos[:, None, :, half_dim:]
        sin_first = sin[:, None, :, :half_dim]
        sin_second = sin[:, None, :, half_dim:]
        grad_first = grad_first_out * cos_first + grad_second_out * sin_second
        grad_second = grad_second_out * cos_second - grad_first_out * sin_first
        return torch.cat((grad_first, grad_second), dim=-1)

    return grad_ref(grad_q_out), grad_ref(grad_k_out)


def _make_rope_fwd_inputs(shape):
    hidden_size, seq_length = shape
    q_heads = 32
    k_heads = 8
    head_dim = hidden_size // q_heads
    q = torch.randn(
        [1, q_heads, seq_length, head_dim],
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        [1, k_heads, seq_length, head_dim],
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    angles = torch.randn(
        [1, seq_length, head_dim],
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return (q, k, cos, sin), lambda: _rope_fwd_reference(q, k, cos, sin)


def _make_rope_bwd_inputs(shape):
    hidden_size, seq_length = shape
    q_heads = 32
    k_heads = 8
    head_dim = hidden_size // q_heads
    grad_q_out = torch.randn(
        [1, q_heads, seq_length, head_dim],
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    grad_k_out = torch.randn(
        [1, k_heads, seq_length, head_dim],
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    angles = torch.randn(
        [1, seq_length, head_dim],
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return (grad_q_out, grad_k_out, cos, sin), lambda: _rope_bwd_reference(
        grad_q_out, grad_k_out, cos, sin
    )


_INPUT_BUILDERS = {
    "vector_add": _make_vector_add_inputs,
    "attention": _make_attention_inputs,
    "softmax": _make_softmax_inputs,
    "layer_norm": _make_layer_norm_inputs,
    "rms_norm": _make_rms_norm_inputs,
    "cross_entropy": _make_cross_entropy_inputs,
    "rope_fwd": _make_rope_fwd_inputs,
    "rope_bwd": _make_rope_bwd_inputs,
}

# (atol, rtol) per kernel. Norm/softmax need looser tolerances than
# vector_add because reductions accumulate fp32 and round back to fp16/bf16.
_TOLERANCES = {
    "vector_add": (1e-5, 1e-5),
    "attention": (5e-2, 2e-2),
    "softmax": (1e-3, 1e-3),
    "layer_norm": (1e-2, 1e-2),
    "rms_norm": (1e-2, 1e-2),
    "cross_entropy": (1e-2, 1e-2),
    "rope_fwd": (2e-2, 1e-2),
    "rope_bwd": (2e-2, 1e-2),
}


@dataclass(frozen=True)
class ExpectedPerf:
    helion_wins: int
    total: int
    geomean: float
    wins_slack: int | None


# helion-vs-best-baseline targets. Every kernel's baselines now include
# ``torch_compile`` (torch.compile of the torch reference), which competes to be
# the fastest baseline -- so these numbers are helion vs the best of {torch,
# torch_compile} (the perf test env has no vLLM). ``wins_slack`` lets that many
# near-noise-band shapes flip without failing; ``None`` disables the wins gate
# for kernels with several expected near-parity shapes.
#
# sm90 sampled on H100. sm100 perf gating is deferred until a B200 nightly
# recalibrates it against the torch_compile baseline (the perf test skips
# compute capabilities absent from this map -- so sm100 runs correctness only
# for now).
_EXPECTED_PERF: dict[str, dict[str, ExpectedPerf]] = {
    "vector_add": {
        "sm90": ExpectedPerf(
            helion_wins=5,
            total=10,
            geomean=0.97,
            wins_slack=None,
        ),
    },
    "softmax": {
        "sm90": ExpectedPerf(
            helion_wins=99,
            total=100,
            geomean=1.50,
            wins_slack=7,
        ),
    },
    "layer_norm": {
        "sm90": ExpectedPerf(
            helion_wins=37,
            total=38,
            geomean=1.29,
            wins_slack=2,
        ),
    },
    "rms_norm": {
        "sm90": ExpectedPerf(
            helion_wins=28,
            total=30,
            geomean=1.18,
            wins_slack=5,
        ),
    },
    "cross_entropy": {
        "sm90": ExpectedPerf(
            helion_wins=21,
            total=21,
            geomean=1.68,
            wins_slack=1,
        ),
    },
    "rope": {
        "sm90": ExpectedPerf(
            helion_wins=6,
            total=7,
            geomean=1.45,
            wins_slack=1,
        ),
    },
    # CUDA-graph-timed kernels (scaled_mm + the vLLM-ported ops below). These use
    # tritonbench's L2-cache-clearing cudagraph timer, so numbers reflect cold-L2
    # (realistic) latency -- lower than a cache-warm timer, since torch.compile's
    # fused kernels win more of the memory-bound shapes when L2 is cleared.
    "scaled_mm": {
        # Small decode GEMMs; helion vs the best of {torch._scaled_mm}. Portable
        # across H100 SKUs/torch versions (cudagraph removes host launch overhead).
        "sm90": ExpectedPerf(
            helion_wins=23,
            total=24,
            geomean=1.14,
            wins_slack=4,
        ),
    },
    # vLLM-ported kernels (vllm/kernels/helion/ops): each benchmarks its fused
    # Helion kernel under CUDA graphs against a torch-native reference and its
    # torch.compile (best of the two). sm90 measured on H100 with L2 clearing;
    # sm100 gating is added once a B200 nightly has calibrated it (the perf test
    # skips compute capabilities absent from this map).
    "silu_mul_fp8": {
        # Bandwidth-bound. Re-tuned on H100 with Helion's autotuner under
        # cudagraph timing (see the kernel's heuristic header); helion now wins
        # the majority of shapes vs the best of {torch, torch.compile}.
        "sm90": ExpectedPerf(helion_wins=22, total=36, geomean=1.15, wins_slack=8),
    },
    "dynamic_per_token_scaled_fp8_quant": {
        "sm90": ExpectedPerf(helion_wins=18, total=24, geomean=1.22, wins_slack=5),
    },
    "per_token_group_fp8_quant": {
        "sm90": ExpectedPerf(helion_wins=22, total=24, geomean=2.45, wins_slack=4),
    },
    "rms_norm_dynamic_per_token_quant": {
        "sm90": ExpectedPerf(helion_wins=35, total=36, geomean=1.34, wins_slack=4),
    },
    "rms_norm_per_block_quant": {
        "sm90": ExpectedPerf(helion_wins=24, total=24, geomean=3.30, wins_slack=2),
    },
    "silu_and_mul_per_block_quant": {
        "sm90": ExpectedPerf(helion_wins=24, total=24, geomean=2.68, wins_slack=2),
    },
    "fused_qk_norm_rope": {
        "sm90": ExpectedPerf(helion_wins=21, total=21, geomean=7.2, wins_slack=2),
    },
}

# Geomean must stay within this fraction below expected. Catches regressions
# only; speedups going up is fine.
_GEOMEAN_NOISE_BAND = 0.10


@onlyBackends(["triton"])
@skipIfRefEager("Pretuned kernels use AOT; ref-eager bypasses heuristic logic.")
class TestPretunedKernelsCorrectness(TestCase):
    """Numerical correctness vs. PyTorch eager."""

    def _run_correctness(self, name: str) -> None:
        if not is_cuda():
            self.skipTest("Pretuned kernels require CUDA / ROCm.")
        module = _import_pretuned_kernel_module(_KERNEL_MODULE_NAMES.get(name, name))
        kernel = getattr(module, name)
        builder = _INPUT_BUILDERS[name]
        atol, rtol = _TOLERANCES[name]
        for shape in _CORRECTNESS_SHAPES[name]:
            with self.subTest(shape=shape):
                args, ref_fn = builder(shape)
                actual = kernel(*args)
                expected = ref_fn()
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_vector_add(self):
        self._run_correctness("vector_add")

    def test_attention(self):
        self._run_correctness("attention")

    def test_attention_cute_flash_compiles(self):
        """The pretuned attention kernel compiles on the CuTe (tcgen05) flash
        fast-path -- not the generic cutedsl fallback -- and matches SDPA.

        Guards the CuTe flash enablement the kernel depends on: without it, the
        cute backend falls back to a generic path that cannot lower the
        online-softmax ``torch.maximum``. The pretuned heuristic (and the flash
        envelope) target B200, so this runs only on sm100.
        """
        if not is_cuda():
            self.skipTest("Pretuned kernels require CUDA / ROCm.")
        if _current_compute_capability() != "sm100":
            self.skipTest("CuTe flash attention is pretuned for B200 (sm100).")
        import importlib.util

        module = _import_pretuned_kernel_module("attention")
        heuristic_path = (
            PRETUNED_KERNELS_DIR / "attention" / "_helion_aot_attention_cuda_sm100.py"
        )
        spec = importlib.util.spec_from_file_location("_attn_heuristic", heuristic_path)
        assert spec is not None and spec.loader is not None
        heuristic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(heuristic)

        q, k, v = (
            torch.randn(2, 8, 512, 64, device=DEVICE, dtype=torch.float16)
            for _ in range(3)
        )
        bound = module.attention.bind((q, k, v))
        config = Config(**heuristic.autotune_attention(q, k, v))
        # The fused tcgen05 flash fast-path emits this marker; the generic
        # cutedsl fallback (which errors on the softmax max) never gets here.
        self.assertIn("flash_kv_prod", bound.to_triton_code(config))
        bound.set_config(config)
        out = bound(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=5e-2, rtol=2e-2)

    def test_softmax(self):
        self._run_correctness("softmax")

    def test_layer_norm(self):
        self._run_correctness("layer_norm")

    def test_rms_norm(self):
        self._run_correctness("rms_norm")

    def test_cross_entropy(self):
        self._run_correctness("cross_entropy")

    def test_rope_fwd(self):
        self._run_correctness("rope_fwd")

    def test_rope_bwd(self):
        self._run_correctness("rope_bwd")

    def test_scaled_mm(self):
        if not is_cuda():
            self.skipTest("Pretuned kernels require CUDA / ROCm.")
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("scaled_mm requires FP8 support (SM89+).")
        module = _import_pretuned_kernel_module("scaled_mm")
        kernel = module.scaled_mm
        fp8_dtype = torch.float8_e4m3fn
        for M, K, N in [(16, 4096, 4096), (64, 2048, 2048)]:
            with self.subTest(shape=(M, K, N)):
                scale = 1.0 / math.sqrt(K)
                a = (scale * (0.5 + torch.rand(M, K, device=DEVICE))).to(fp8_dtype)
                b = (scale * (0.5 + torch.rand(N, K, device=DEVICE))).to(fp8_dtype).t()
                c = torch.empty((M, N), dtype=torch.bfloat16, device=DEVICE)
                scale_a = torch.rand((1, 1), device=DEVICE) + 0.5
                scale_b = torch.rand((1, 1), device=DEVICE) + 0.5
                bias = torch.rand(N, dtype=torch.bfloat16, device=DEVICE) - 0.5
                kernel(c, a, b, scale_a, scale_b, bias)
                # Reference dequantizes to fp32, matching the kernel's bias-after-cast.
                ref = ((a.float() @ b.float()) * scale_a * scale_b).to(
                    torch.bfloat16
                ) + bias
                torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-1)

    def _run_vllm_ported_correctness(self, name: str, needs_fp8: bool = True) -> None:
        # vLLM-ported kernels self-verify via the module's correctness_check(),
        # which runs the kernel and its torch-native reference on one shape.
        if not is_cuda():
            self.skipTest("Pretuned kernels require CUDA / ROCm.")
        if needs_fp8 and torch.cuda.get_device_capability() < (8, 9):
            self.skipTest(f"{name} requires FP8 support (SM89+).")
        module = _import_pretuned_kernel_module(name)
        module.correctness_check()

    def test_silu_mul_fp8(self):
        self._run_vllm_ported_correctness("silu_mul_fp8")

    def test_dynamic_per_token_scaled_fp8_quant(self):
        self._run_vllm_ported_correctness("dynamic_per_token_scaled_fp8_quant")

    def test_per_token_group_fp8_quant(self):
        self._run_vllm_ported_correctness("per_token_group_fp8_quant")

    def test_rms_norm_dynamic_per_token_quant(self):
        self._run_vllm_ported_correctness("rms_norm_dynamic_per_token_quant")

    def test_rms_norm_per_block_quant(self):
        self._run_vllm_ported_correctness("rms_norm_per_block_quant")

    def test_silu_and_mul_per_block_quant(self):
        self._run_vllm_ported_correctness("silu_and_mul_per_block_quant")

    def test_fused_qk_norm_rope(self):
        self._run_vllm_ported_correctness("fused_qk_norm_rope", needs_fp8=False)


@onlyBackends(["triton"])
@skipIfRefEager("Pretuned kernels use AOT; ref-eager bypasses heuristic logic.")
class TestPretunedKernelsPerformance(TestCase):
    """Run each kernel's main() on hardware matching its checked-in heuristic."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if _under_xdist():
            raise unittest.SkipTest(
                "Perf gating is unreliable under pytest-xdist GPU contention."
            )
        if is_fbcode():
            raise unittest.SkipTest(
                "Perf gating is unreliable under fbcode GPU contention/deadlines."
            )

    def _run_pretuned_kernel_perf(self, name: str) -> None:
        expected_by_compute = _EXPECTED_PERF[name]
        current_compute = _current_compute_capability()
        if current_compute not in expected_by_compute:
            expected_compute = ", ".join(expected_by_compute)
            self.skipTest(
                f"{name}: pretuned perf target is {expected_compute}; "
                f"current device is {current_compute or 'none'}."
            )
        expected = expected_by_compute[current_compute]

        actual = _run_pretuned_kernel_main_and_parse_summary(name)
        self.assertEqual(
            actual["total"],
            expected.total,
            f"{name}: shape sweep size changed "
            f"({actual['total']} vs expected {expected.total}); "
            f"update _EXPECTED_PERF if intentional.",
        )
        if expected.wins_slack is not None:
            wins_floor = max(0, expected.helion_wins - expected.wins_slack)
            self.assertGreaterEqual(
                actual["helion_wins"],
                wins_floor,
                f"{name}: Helion wins {actual['helion_wins']}/{actual['total']} "
                f"shapes, below floor {wins_floor} "
                f"(expected ~{expected.helion_wins}, slack {expected.wins_slack}).",
            )
        geomean_floor = expected.geomean * (1 - _GEOMEAN_NOISE_BAND)
        self.assertGreaterEqual(
            actual["geomean"],
            geomean_floor,
            f"{name}: geomean {actual['geomean']:.3f}x below floor "
            f"{geomean_floor:.3f}x "
            f"(expected ~{expected.geomean:.3f}x, "
            f"noise band {_GEOMEAN_NOISE_BAND:.0%}).",
        )

    def test_vector_add(self):
        self._run_pretuned_kernel_perf("vector_add")

    # softmax/layer_norm sweep enough shapes to need >60s under xdist contention.
    @pytest.mark.timeout(120)
    def test_softmax(self):
        self._run_pretuned_kernel_perf("softmax")

    @pytest.mark.timeout(120)
    def test_layer_norm(self):
        self._run_pretuned_kernel_perf("layer_norm")

    def test_rms_norm(self):
        self._run_pretuned_kernel_perf("rms_norm")

    def test_cross_entropy(self):
        self._run_pretuned_kernel_perf("cross_entropy")

    @pytest.mark.timeout(120)
    def test_rope(self):
        self._run_pretuned_kernel_perf("rope")

    # The cudagraph-timed kernels below use tritonbench's L2-cache-clearing
    # timer, which is ~9x slower per measurement than triton's do_bench_cudagraph
    # (it captures/replays a large graph and zeroes L2 each iteration), so these
    # get a generous timeout. These perf tests are local-only (skipped under CI
    # xdist), so the long runtime does not affect CI.
    @pytest.mark.timeout(600)
    def test_scaled_mm(self):
        self._run_pretuned_kernel_perf("scaled_mm")

    @pytest.mark.timeout(600)
    def test_silu_mul_fp8(self):
        self._run_pretuned_kernel_perf("silu_mul_fp8")

    @pytest.mark.timeout(600)
    def test_dynamic_per_token_scaled_fp8_quant(self):
        self._run_pretuned_kernel_perf("dynamic_per_token_scaled_fp8_quant")

    @pytest.mark.timeout(600)
    def test_per_token_group_fp8_quant(self):
        self._run_pretuned_kernel_perf("per_token_group_fp8_quant")

    @pytest.mark.timeout(600)
    def test_rms_norm_dynamic_per_token_quant(self):
        self._run_pretuned_kernel_perf("rms_norm_dynamic_per_token_quant")

    @pytest.mark.timeout(600)
    def test_rms_norm_per_block_quant(self):
        self._run_pretuned_kernel_perf("rms_norm_per_block_quant")

    @pytest.mark.timeout(600)
    def test_silu_and_mul_per_block_quant(self):
        self._run_pretuned_kernel_perf("silu_and_mul_per_block_quant")

    @pytest.mark.timeout(600)
    def test_fused_qk_norm_rope(self):
        self._run_pretuned_kernel_perf("fused_qk_norm_rope")


if __name__ == "__main__":
    unittest.main()
