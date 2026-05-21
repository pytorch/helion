"""Correctness and perf tests for kernels under ``pretuned_kernels/``.

Correctness runs on every CUDA / ROCm runner; perf gating runs only on
the hardware where each kernel's checked-in heuristics apply.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib.util
import io
import os
import re
import sys
import unittest

import pytest
import torch
from torch._environment import is_fbcode
import torch.nn.functional as F

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
    # Private module name avoids clashing with ``examples/<name>.py``.
    module_name = f"_helion_pretuned_kernels_test.{name}"
    if module_name not in sys.modules:
        file_path = PRETUNED_KERNELS_DIR / name / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    return sys.modules[module_name]


_SUMMARY_RE = re.compile(
    r"^SUMMARY:\s+helion_wins=(?P<wins>\d+)\s+total=(?P<total>\d+)\s+"
    r"geomean=(?P<geomean>[\d.]+)\s+best_speedup=(?P<best>[\d.]+)\s*$",
    re.MULTILINE,
)


def _run_pretuned_kernel_main_and_parse_summary(name):
    module = _import_pretuned_kernel_module(name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        module.main()
    output = buf.getvalue()
    # Relay output so CI logs show the per-shape table on failure.
    print(output)
    match = _SUMMARY_RE.search(output)
    if match is None:
        raise AssertionError(
            f"Could not find SUMMARY line in pretuned kernel output for {name}.\n"
            f"Output was:\n{output}"
        )
    return {
        "helion_wins": int(match["wins"]),
        "total": int(match["total"]),
        "geomean": float(match["geomean"]),
        "best_speedup": float(match["best"]),
    }


_CORRECTNESS_SHAPES = {
    "vector_add": [2**20],
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


# Sampled on B200 with the checked-in heuristic. ``wins_slack`` lets that
# many near-noise-band shapes flip without failing; ``None`` disables the
# wins gate for kernels with several expected near-parity shapes.
_EXPECTED_PERF: dict[str, dict[str, ExpectedPerf]] = {
    "vector_add": {
        "sm100": ExpectedPerf(
            helion_wins=5,
            total=10,
            geomean=1.009,
            wins_slack=None,
        ),
        "sm90": ExpectedPerf(
            helion_wins=5,
            total=10,
            geomean=0.99,
            wins_slack=None,
        ),
    },
    "softmax": {
        "sm100": ExpectedPerf(
            helion_wins=100,
            total=100,
            geomean=2.304,
            wins_slack=2,
        ),
        "sm90": ExpectedPerf(
            helion_wins=97,
            total=100,
            geomean=1.78,
            wins_slack=7,
        ),
    },
    "layer_norm": {
        "sm100": ExpectedPerf(
            helion_wins=38,
            total=38,
            geomean=1.55,
            wins_slack=1,
        ),
        "sm90": ExpectedPerf(
            helion_wins=37,
            total=38,
            geomean=1.39,
            wins_slack=2,
        ),
    },
    "rms_norm": {
        "sm100": ExpectedPerf(
            helion_wins=30,
            total=30,
            geomean=1.605,
            wins_slack=6,
        ),
        "sm90": ExpectedPerf(
            helion_wins=23,
            total=30,
            geomean=1.17,
            wins_slack=5,
        ),
    },
    "cross_entropy": {
        "sm100": ExpectedPerf(
            helion_wins=21,
            total=21,
            geomean=1.698,
            wins_slack=1,
        ),
        "sm90": ExpectedPerf(
            helion_wins=21,
            total=21,
            geomean=2.35,
            wins_slack=1,
        ),
    },
    "rope": {
        "sm90": ExpectedPerf(
            helion_wins=7,
            total=7,
            geomean=5.0,
            wins_slack=1,
        ),
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


if __name__ == "__main__":
    unittest.main()
