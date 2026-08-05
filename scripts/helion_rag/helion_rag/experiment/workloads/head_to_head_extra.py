"""Additional head-to-head workload shapes across 11 kernel families.

Registers the extra shapes requested for the four-arm sweep. Each family reuses
its existing Helion example kernel; only the input shapes differ. Every torch
import stays inside a per-shape ``build`` closure so importing this module (which
the registry auto-imports) pulls no CUDA -- the running 15-kernel campaign is
unaffected by these extra registry entries.

Shape shorthand decisions (documented so they can be corrected):
- matmul split-K: shorthand gives only K; M=N=64 (the example's own default).
- fp8-attention / attention: (S, D) = seq_len, head_dim; batch/heads default to
  the example/existing-workload conventions (fp8: b2 h4; attention: b2 h8).
- grouped-GEMM: (g, M) = groups, rows-per-group; K=256, N=128 (example defaults).
- gated-delta-net / mamba2: (S, C) = seq_len, state dim (dstate); chunk_size is
  held at 128 (all S values are divisible by it).
"""

from __future__ import annotations

from collections.abc import Callable

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _matmul(n: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.matmul import matmul

        x = torch.randn((n, n), device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn((n, n), device=DEVICE, dtype=HALF_DTYPE)
        return matmul, torch.matmul, (x, y)

    return build


def _split_k(m: int, k: int, n: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.matmul_split_k import matmul_split_k

        x = torch.randn((m, k), device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn((k, n), device=DEVICE, dtype=HALF_DTYPE)
        return matmul_split_k, torch.matmul, (x, y)

    return build


def _attention(z: int, h: int, s: int, d: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.attention import _attention_baseline
        from examples.attention import attention

        q, k, v = (
            torch.randn((z, h, s, d), device=DEVICE, dtype=HALF_DTYPE) for _ in range(3)
        )
        return attention, _attention_baseline, (q, k, v)

    return build


def _fp8_attention(b: int, h: int, s: int, d: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.fp8_attention import fp8_attention
        from examples.fp8_attention import fp8_attention_pytorch

        torch.manual_seed(42)
        q = torch.randn(b, h, s, d, device=DEVICE, dtype=HALF_DTYPE)
        k = torch.randn(b, h, s, d, device=DEVICE, dtype=HALF_DTYPE)
        v = torch.randn(b, h, s, d, device=DEVICE, dtype=HALF_DTYPE)

        # The kernel returns float8 output; cast both sides to fp32 so the
        # correctness comparison is dtype-matched (matching the example's
        # run_example-based check) instead of failing on an fp8-vs-fp16 mismatch.
        def helion(qq: object, kk: object, vv: object) -> object:
            return fp8_attention(qq, kk, vv).to(torch.float32)

        def reference(qq: object, kk: object, vv: object) -> object:
            return fp8_attention_pytorch(qq, kk, vv)().to(torch.float32)

        return helion, reference, (q, k, v)

    return build


def _grouped_gemm(g: int, m: int, k: int, n: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE

        from examples.grouped_gemm import _reference_grouped_gemm
        from examples.grouped_gemm import grouped_gemm_jagged_example

        torch.manual_seed(0)
        dtype = torch.bfloat16
        group_a = [
            torch.randn(m, k, device=DEVICE, dtype=dtype).contiguous() for _ in range(g)
        ]
        group_b = [
            torch.randn(k, n, device=DEVICE, dtype=dtype).contiguous() for _ in range(g)
        ]
        return grouped_gemm_jagged_example, _reference_grouped_gemm, (group_a, group_b)

    return build


def _swiglu(n: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.swiglu import swiglu

        a = torch.randn((n, n), device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn((n, n), device=DEVICE, dtype=HALF_DTYPE)

        def reference(aa: object, bb: object) -> object:
            return torch.nn.functional.silu(aa).to(bb.dtype) * bb

        return swiglu, reference, (a, b)

    return build


def _softmax(rows: int, cols: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.softmax import softmax_two_pass

        x = torch.randn((rows, cols), device=DEVICE, dtype=HALF_DTYPE)
        return softmax_two_pass, lambda t: torch.nn.functional.softmax(t, dim=1), (x,)

    return build


def _rms_norm(rows: int, cols: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE

        from examples.rms_norm import rms_norm
        from examples.rms_norm import rms_norm_pytorch

        x = torch.randn((rows, cols), device=DEVICE, dtype=torch.float32)
        weight = torch.randn((cols,), device=DEVICE, dtype=torch.float32)
        return rms_norm, rms_norm_pytorch, (x, weight)

    return build


def _rope(b: int, qh: int, kh: int, s: int, d: int) -> Callable[[], object]:
    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.rope import rope_fwd
        from examples.rope import rope_pytorch

        q = torch.randn([b, qh, s, d], device=DEVICE, dtype=HALF_DTYPE)
        k = torch.randn([b, kh, s, d], device=DEVICE, dtype=HALF_DTYPE)
        angles = torch.randn([b, s, d], device=DEVICE, dtype=HALF_DTYPE)
        cos, sin = torch.cos(angles), torch.sin(angles)
        return rope_fwd, rope_pytorch, (q, k, cos, sin)

    return build


def _gdn(seqlen: int, dstate: int) -> Callable[[], object]:
    chunk_size, batch, nheads, dhead = 128, 1, 4, 64

    def build() -> object:
        import math

        import torch

        from helion._testing import DEVICE

        from examples.gdn_fwd_h import helion_gdn_fwd_h
        from examples.gdn_fwd_h import ref_gdn_fwd_h

        k = torch.randn(
            batch, seqlen, nheads, dhead, dtype=torch.bfloat16, device=DEVICE
        )
        k = torch.nn.functional.rms_norm(k, [dhead])
        w = torch.randn(
            batch,
            seqlen // chunk_size,
            chunk_size,
            nheads,
            dhead,
            dtype=torch.float32,
            device=DEVICE,
        )
        wu, _ws, wv = torch.linalg.svd(w.permute(0, 1, 3, 2, 4), full_matrices=False)
        w = torch.einsum("bnhik,bnhkj->bnhij", wu, wv)
        w = (
            w.permute(0, 1, 3, 2, 4)
            .reshape(batch, seqlen, nheads, dhead)
            .to(torch.bfloat16)
        )
        u = torch.randn(
            batch, seqlen, nheads, dstate, dtype=torch.bfloat16, device=DEVICE
        )
        u = torch.nn.functional.rms_norm(u, [dstate])
        g = torch.cumsum(
            0.5
            * math.log(1 / dhead)
            * torch.rand(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE),
            dim=1,
        )
        return helion_gdn_fwd_h, ref_gdn_fwd_h, (k, w, u, g, chunk_size)

    return build


def _mamba2(seqlen: int, dstate: int) -> Callable[[], object]:
    chunk_size, batch, nheads, ngroups, headdim = 128, 1, 4, 1, 64

    def build() -> object:
        import torch

        from helion._testing import DEVICE
        from helion._testing import HALF_DTYPE

        from examples.mamba2_chunk_scan import helion_mamba2_chunk_scan_kernel
        from examples.mamba2_chunk_scan import ref_chunk_scan

        nchunks = (seqlen + chunk_size - 1) // chunk_size

        def randn(*shape: int) -> object:
            return torch.randn(*shape, dtype=HALF_DTYPE, device=DEVICE)

        cb = randn(batch, nchunks, ngroups, chunk_size, chunk_size)
        x = randn(batch, seqlen, nheads, headdim)
        dt = randn(batch, nheads, nchunks, chunk_size)
        dA_cumsum = torch.rand(
            batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
        )
        C = torch.zeros(batch, seqlen, ngroups, dstate, dtype=HALF_DTYPE, device=DEVICE)
        prev_states = torch.zeros(
            batch, nchunks, nheads, headdim, dstate, dtype=HALF_DTYPE, device=DEVICE
        )
        D = torch.zeros(nheads, dtype=HALF_DTYPE, device=DEVICE)
        args = (cb, x, dt, dA_cumsum, C, prev_states, D)
        return helion_mamba2_chunk_scan_kernel, ref_chunk_scan, args

    return build


# (workload_id, kernel_name, rtol, atol, build) for every new shape.
_NEW_WORKLOADS: tuple[tuple[str, str, float, float, Callable[[], object]], ...] = (
    # matmul square (1024^3 already registered by matmul.py)
    ("matmul-8192x8192x8192", "matmul", 1e-2, 1e-1, _matmul(8192)),
    ("matmul-4096x4096x4096", "matmul", 1e-2, 1e-1, _matmul(4096)),
    # matmul split-K (M=N=64)
    (
        "matmul_split_k-64x65536x64",
        "matmul_split_k",
        1e-2,
        1.0,
        _split_k(64, 65536, 64),
    ),
    (
        "matmul_split_k-64x16384x64",
        "matmul_split_k",
        1e-2,
        1.0,
        _split_k(64, 16384, 64),
    ),
    ("matmul_split_k-64x1024x64", "matmul_split_k", 1e-2, 1.0, _split_k(64, 1024, 64)),
    # attention (b2 h8; S=512 already registered by attention.py)
    ("attention-2x8x4096x64", "attention", 2e-2, 5e-2, _attention(2, 8, 4096, 64)),
    ("attention-2x8x8192x64", "attention", 2e-2, 5e-2, _attention(2, 8, 8192, 64)),
    # fp8-attention (b2 h4)
    (
        "fp8_attention-2x4x8192x64",
        "fp8_attention_kernel",
        0.1,
        0.1,
        _fp8_attention(2, 4, 8192, 64),
    ),
    (
        "fp8_attention-2x4x2048x64",
        "fp8_attention_kernel",
        0.1,
        0.1,
        _fp8_attention(2, 4, 2048, 64),
    ),
    (
        "fp8_attention-2x4x512x64",
        "fp8_attention_kernel",
        0.1,
        0.1,
        _fp8_attention(2, 4, 512, 64),
    ),
    # grouped-GEMM (K=256, N=128)
    (
        "grouped_gemm-g2m1024",
        "grouped_gemm_jagged",
        1e-2,
        1e-2,
        _grouped_gemm(2, 1024, 256, 128),
    ),
    (
        "grouped_gemm-g4m512",
        "grouped_gemm_jagged",
        1e-2,
        1e-2,
        _grouped_gemm(4, 512, 256, 128),
    ),
    (
        "grouped_gemm-g8m512",
        "grouped_gemm_jagged",
        1e-2,
        1e-2,
        _grouped_gemm(8, 512, 256, 128),
    ),
    # swiglu (n x n)
    ("swiglu-2048x2048", "_swiglu_fwd", 1e-2, 1e-1, _swiglu(2048)),
    ("swiglu-4096x4096", "_swiglu_fwd", 1e-2, 1e-1, _swiglu(4096)),
    ("swiglu-8192x8192", "_swiglu_fwd", 1e-2, 1e-1, _swiglu(8192)),
    # softmax (rows x cols)
    ("softmax-4096x32768", "softmax_two_pass", 1e-2, 1e-1, _softmax(4096, 32768)),
    ("softmax-4096x8192", "softmax_two_pass", 1e-2, 1e-1, _softmax(4096, 8192)),
    ("softmax-4096x1024", "softmax_two_pass", 1e-2, 1e-1, _softmax(4096, 1024)),
    # gated-delta-net (b1 h4; C = dstate; chunk_size 128)
    ("gdn_fwd_h-b1h4s2048ds128", "helion_gdn_fwd_h", 1e-2, 1e-1, _gdn(2048, 128)),
    ("gdn_fwd_h-b1h4s8192ds128", "helion_gdn_fwd_h", 1e-2, 1e-1, _gdn(8192, 128)),
    ("gdn_fwd_h-b1h4s4096ds64", "helion_gdn_fwd_h", 1e-2, 1e-1, _gdn(4096, 64)),
    # rms_norm (fp32)
    ("rms_norm-4096x1024", "rms_norm_fwd", 1e-3, 1e-3, _rms_norm(4096, 1024)),
    ("rms_norm-4096x32768", "rms_norm_fwd", 1e-3, 1e-3, _rms_norm(4096, 32768)),
    ("rms_norm-4096x8192", "rms_norm_fwd", 1e-3, 1e-3, _rms_norm(4096, 8192)),
    # rope (b1 q4 k2; D=128)
    ("rope-1x4x2x512x128", "rope_fwd", 1e-2, 1e-2, _rope(1, 4, 2, 512, 128)),
    ("rope-1x4x2x8192x128", "rope_fwd", 1e-2, 1e-2, _rope(1, 4, 2, 8192, 128)),
    ("rope-1x4x2x2048x128", "rope_fwd", 1e-2, 1e-2, _rope(1, 4, 2, 2048, 128)),
    # mamba2 chunk-scan (b1 h4; C = dstate; chunk_size 128)
    (
        "mamba2_chunk_scan-b1h4s2048ds128",
        "helion_mamba2_chunk_scan_kernel",
        1e-2,
        1e-1,
        _mamba2(2048, 128),
    ),
    (
        "mamba2_chunk_scan-b1h4s4096ds256",
        "helion_mamba2_chunk_scan_kernel",
        1e-2,
        1e-1,
        _mamba2(4096, 256),
    ),
    (
        "mamba2_chunk_scan-b1h4s8192ds256",
        "helion_mamba2_chunk_scan_kernel",
        1e-2,
        1e-1,
        _mamba2(8192, 256),
    ),
)

# The full 33-shape run set: the 31 new shapes plus the two that already exist
# (matmul 1024^3 and attention S=512 D=64) so a subset run can request all of them.
HEAD_TO_HEAD_EXTRA_IDS: tuple[str, ...] = tuple(row[0] for row in _NEW_WORKLOADS) + (
    "matmul-1024x1024x1024",
    "attention-2x8x512x64",
)

for _id, _kernel_name, _rtol, _atol, _build in _NEW_WORKLOADS:
    register(_id, Workload(_kernel_name, _rtol, _atol, _build))
