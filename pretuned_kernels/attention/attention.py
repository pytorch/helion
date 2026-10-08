"""GB300 (sm103) output-only attention for twelve pretuned BHND workloads.

The dense and causal kernel bodies match ``examples/attention.py``. The local
AOT table contains the actual final FULL-search returns; it does not extend
compiler dispatch or choose a nearby configuration for unsupported inputs.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel
import torch.nn.functional as F

import helion
import helion.language as hl

ShapeKey = tuple[int, int, int, int, str]
Shape = tuple[int, int, int, int, str, bool]

# (batch, heads, sequence length, head dimension, dtype, causal), in matrix order.
SHAPES: list[Shape] = [
    (1, 8, 12288, 128, "float16", False),
    (2, 32, 16384, 128, "float16", False),
    (2, 32, 32768, 64, "float16", False),
    (2, 32, 65536, 64, "float16", True),
    (2, 32, 2048, 64, "float16", False),
    (2, 32, 4096, 64, "float16", True),
    (8, 32, 8192, 128, "bfloat16", False),
    (1, 1, 8192, 128, "float16", False),
    (2, 16, 8192, 128, "float16", False),
    (1, 8, 12288, 128, "float16", True),
    (4, 32, 4096, 128, "bfloat16", False),
    (4, 32, 4096, 128, "bfloat16", True),
]


def _input_key(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> ShapeKey:
    """Validate the measured input scope even if no AOT file is available."""
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Pretuned attention requires four-dimensional BHND inputs")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Pretuned attention requires equal Q/K/V shapes")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Pretuned attention supports float16 and bfloat16 only")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Pretuned attention requires matching Q/K/V dtypes")
    if q.device.type != "cuda" or q.device != k.device or q.device != v.device:
        raise ValueError("Pretuned attention requires Q/K/V on the same CUDA device")
    if not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError("Pretuned attention requires contiguous BHND inputs")
    dtype = "float16" if q.dtype == torch.float16 else "bfloat16"
    key = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]), dtype)
    if (*key, causal) not in SHAPES:
        raise ValueError(f"No pretuned attention config for {key}, causal={causal}")
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        raise ValueError("These attention configs are pretuned for GB300/sm103 only")
    # Both dtype objects become the same two-byte size during AOT flattening.
    # Preserve float16 versus bfloat16 with an explicit scalar string.
    return key


def _dense_key(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> ShapeKey:
    return _input_key(q, k, v, causal=False)


def _causal_key(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> ShapeKey:
    return _input_key(q, k, v, causal=True)


def _sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


def _dense_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _sdpa(q, k, v, causal=False)


def _causal_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return _sdpa(q, k, v, causal=True)


def _check_attention_accuracy(actual: object, expected: object) -> None:
    """Check every output element with a dtype-specific rounding allowance."""
    assert isinstance(actual, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert actual.dtype in (torch.float16, torch.bfloat16)
    tolerance = 1e-3 if actual.dtype == torch.float16 else 5e-3
    if not bool((torch.isfinite(actual).all() & torch.isfinite(expected).all()).item()):
        raise AssertionError("Attention outputs must be finite")
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    key=_dense_key,
    autotune_baseline_fn=_dense_reference,
    autotune_baseline_accuracy_check_fn=_check_attention_accuracy,
)
def attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention and returns only the output tensor.
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    key=_causal_key,
    autotune_baseline_fn=_causal_reference,
    autotune_baseline_accuracy_check_fn=_check_attention_accuracy,
)
def causal_attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes causal scaled dot-product attention and returns only the output tensor.
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def use_cudagraph() -> bool:
    """Use plain CUDA events, as in the original pretuning measurements."""
    return False


def main(verbose: bool = True) -> dict:
    """Check every shape before a convenience sweep against output-only cuDNN."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    def make_calls(shape: Shape) -> tuple:
        batch, heads, sequence, head_dim, dtype_name, causal = shape
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
        generator = torch.Generator(device="cuda").manual_seed(101)
        args = tuple(
            torch.randn(
                (batch, heads, sequence, head_dim),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            for _ in range(3)
        )
        kernel = causal_attention_output if causal else attention_output
        reference = _causal_reference if causal else _dense_reference

        def helion_call() -> torch.Tensor:
            return kernel(*args)

        def sdpa_call() -> torch.Tensor:
            return reference(*args)

        # run.py calls main() directly, so this check belongs inside the sweep.
        _check_attention_accuracy(helion_call(), sdpa_call())
        return (
            helion_call,
            [("cuDNN SDPA", sdpa_call)],
            (
                f"{batch:>2d}  {heads:>2d}  {sequence:>6d}  {head_dim:>3d}  "
                f"{dtype_name:>8s}  {causal!s:>6s}"
            ),
        )

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'B':>2s}  {'H':>2s}  {'N':>6s}  {'D':>3s}  {'dtype':>8s}  {'causal':>6s}",
    )


if __name__ == "__main__":
    main()
